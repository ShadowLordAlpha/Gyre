// LlamaLM (dense Llama / SmolLM2 inference). The synthetic tests write a tiny Hugging Face
// LlamaForCausalLM directory with random weights (GQA, odd head counts, widths that are not a
// multiple of the 16-column kernel panel) and check the imported model against an independent
// double-precision reference of the HF forward pass (rotate_half RoPE on the ORIGINAL [out, in]
// weights), so the transposition and the per-head q/k permutation are both covered. The KV cache
// must reproduce a full recompute. Real-model tests skip unless data/smollm2-360m (or -1.7b)
// holds the converted checkpoint.
#include "gradcheck.hpp"

#include "gyre/io/safetensors.hpp"
#include "gyre/nn/llama.hpp"
#include "gyre/nn/llama_import.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/ops.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>

using namespace gyre;
using namespace gyre::testing;

namespace {

struct Tiny {
  int V = 53, d = 40, L = 2, Hq = 6, Hkv = 2, hd = 8, ff = 56;
  bool tied = true;
  double theta = 500.0, eps = 1e-5;
};

using Mat = std::vector<double>;

std::filesystem::path write_tiny_llama(const Tiny& g, Rng& rng, std::map<std::string, Mat>& w,
                                       const std::string& extra_cfg = {}) {
  auto dir = std::filesystem::temp_directory_path() / ("gyre_llama_" + std::to_string(rng.u32(1u << 30)));
  std::filesystem::create_directories(dir);
  {
    std::ofstream c(dir / "config.json");
    c << "{\"architectures\":[\"LlamaForCausalLM\"],\"model_type\":\"llama\",\"hidden_act\":\"silu\","
      << "\"attention_bias\":false,\"rms_norm_eps\":" << g.eps << ",\"rope_theta\":" << g.theta
      << ",\"rope_scaling\":null,\"hidden_size\":" << g.d << ",\"intermediate_size\":" << g.ff
      << ",\"num_attention_heads\":" << g.Hq << ",\"num_key_value_heads\":" << g.Hkv << ",\"head_dim\":" << g.hd
      << ",\"num_hidden_layers\":" << g.L << ",\"vocab_size\":" << g.V << ",\"max_position_embeddings\":64"
      << ",\"tie_word_embeddings\":" << (g.tied ? "true" : "false") << ",\"bos_token_id\":0,\"eos_token_id\":0"
      << extra_cfg << "}";
  }
  std::map<std::string, Tensor> t;
  auto add = [&](const std::string& n, std::vector<std::int64_t> sh, float sd, float mean = 0.f) {
    auto x = rand_t(sh, rng, sd);
    for (auto& v : *x.host_span<float>()) v += mean;
    auto p = x.host_span<float>();
    w[n] = Mat(p->begin(), p->end());
    t.emplace(n, std::move(x));
  };
  const int qd = g.Hq * g.hd, kvd = g.Hkv * g.hd;
  add("model.embed_tokens.weight", {g.V, g.d}, 0.5f);
  for (int i = 0; i < g.L; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    add(p + "input_layernorm.weight", {g.d}, 0.1f, 1.f);
    add(p + "self_attn.q_proj.weight", {qd, g.d}, 0.3f);
    add(p + "self_attn.k_proj.weight", {kvd, g.d}, 0.3f);
    add(p + "self_attn.v_proj.weight", {kvd, g.d}, 0.3f);
    add(p + "self_attn.o_proj.weight", {g.d, qd}, 0.3f);
    add(p + "post_attention_layernorm.weight", {g.d}, 0.1f, 1.f);
    add(p + "mlp.gate_proj.weight", {g.ff, g.d}, 0.3f);
    add(p + "mlp.up_proj.weight", {g.ff, g.d}, 0.3f);
    add(p + "mlp.down_proj.weight", {g.d, g.ff}, 0.3f);
  }
  add("model.norm.weight", {g.d}, 0.1f, 1.f);
  if (!g.tied) add("lm_head.weight", {g.V, g.d}, 0.3f);
  std::vector<NamedTensor> nt;
  for (auto& [n, x] : t) nt.push_back({n, &x});
  if (!safetensors_save(dir / "model.safetensors", nt)) std::abort();
  return dir;
}

// Hugging Face LlamaForCausalLM forward in double precision, straight from the model definition.
Mat reference_logits(const std::map<std::string, Mat>& w, const Tiny& g, const std::vector<int>& tok) {
  const int T = static_cast<int>(tok.size()), d = g.d, hd = g.hd, Hq = g.Hq, Hkv = g.Hkv, half = hd / 2;
  const int qd = Hq * hd, kvd = Hkv * hd, rep = Hq / Hkv;
  auto W = [&](const std::string& n) -> const Mat& { return w.at(n); };
  auto rms = [&](const Mat& x, const Mat& gamma) {
    Mat y(x.size());
    for (int t = 0; t < T; ++t) {
      double ms = 0;
      for (int i = 0; i < d; ++i) ms += x[t * d + i] * x[t * d + i];
      const double inv = 1.0 / std::sqrt(ms / d + g.eps);
      for (int i = 0; i < d; ++i) y[t * d + i] = x[t * d + i] * inv * gamma[i];
    }
    return y;
  };
  // y[t, o] = sum_i x[t, i] * M[o, i]   (nn.Linear, weight [out, in])
  auto lin = [&](const Mat& x, const Mat& M, int out, int in) {
    Mat y(static_cast<std::size_t>(T) * out, 0.0);
    for (int t = 0; t < T; ++t)
      for (int o = 0; o < out; ++o) {
        double s = 0;
        for (int i = 0; i < in; ++i) s += x[t * in + i] * M[static_cast<std::size_t>(o) * in + i];
        y[t * out + o] = s;
      }
    return y;
  };
  auto rotate_half = [&](Mat& x, int heads) {
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < heads; ++h)
        for (int j = 0; j < half; ++j) {
          const double inv = std::pow(g.theta, -2.0 * j / hd), a = t * inv;
          double& x1 = x[t * heads * hd + h * hd + j];
          double& x2 = x[t * heads * hd + h * hd + j + half];
          const double a1 = x1, a2 = x2;
          x1 = a1 * std::cos(a) - a2 * std::sin(a);
          x2 = a2 * std::cos(a) + a1 * std::sin(a);
        }
  };
  Mat x(static_cast<std::size_t>(T) * d);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < d; ++i) x[t * d + i] = W("model.embed_tokens.weight")[static_cast<std::size_t>(tok[t]) * d + i];
  for (int l = 0; l < g.L; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    auto xn = rms(x, W(p + "input_layernorm.weight"));
    auto q = lin(xn, W(p + "self_attn.q_proj.weight"), qd, d);
    auto k = lin(xn, W(p + "self_attn.k_proj.weight"), kvd, d);
    auto v = lin(xn, W(p + "self_attn.v_proj.weight"), kvd, d);
    rotate_half(q, Hq);
    rotate_half(k, Hkv);
    Mat att(static_cast<std::size_t>(T) * qd, 0.0);
    for (int h = 0; h < Hq; ++h) {
      const int kh = h / rep;  // repeat_kv
      for (int t = 0; t < T; ++t) {
        std::vector<double> s(t + 1);
        double mx = -1e300;
        for (int u = 0; u <= t; ++u) {
          double dd = 0;
          for (int j = 0; j < hd; ++j) dd += q[t * qd + h * hd + j] * k[u * kvd + kh * hd + j];
          s[u] = dd / std::sqrt(static_cast<double>(hd));
          mx = std::max(mx, s[u]);
        }
        double z = 0;
        for (auto& e : s) z += (e = std::exp(e - mx));
        for (int u = 0; u <= t; ++u)
          for (int j = 0; j < hd; ++j) att[t * qd + h * hd + j] += s[u] / z * v[u * kvd + kh * hd + j];
      }
    }
    auto o = lin(att, W(p + "self_attn.o_proj.weight"), d, qd);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    xn = rms(x, W(p + "post_attention_layernorm.weight"));
    auto ga = lin(xn, W(p + "mlp.gate_proj.weight"), g.ff, d);
    auto up = lin(xn, W(p + "mlp.up_proj.weight"), g.ff, d);
    for (std::size_t i = 0; i < ga.size(); ++i) ga[i] = ga[i] / (1.0 + std::exp(-ga[i])) * up[i];
    auto dn = lin(ga, W(p + "mlp.down_proj.weight"), d, g.ff);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += dn[i];
  }
  auto xf = rms(x, W("model.norm.weight"));
  return lin(xf, g.tied ? W("model.embed_tokens.weight") : W("lm_head.weight"), g.V, d);
}

double max_abs_diff(std::span<const float> a, std::span<const double> b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

std::vector<int> tokens(int n, int V, Rng& rng) {
  std::vector<int> t(static_cast<std::size_t>(n));
  for (auto& x : t) x = static_cast<int>(rng.u32(static_cast<std::uint32_t>(V)));
  return t;
}

std::filesystem::path smol_ckpt(const char* name) {
  return std::filesystem::path(GYRE_SOURCE_DIR) / "data" / name / (std::string(name) + ".gyre");
}

// Greedy continuation of a factual prompt from a converted checkpoint (`gyre-cli lm import-llama`).
void expect_knows_capital_of_france(const std::filesystem::path& ckpt) {
  auto m = LlamaLM::load_gyre(ckpt);
  ASSERT_TRUE(m) << m.error().message;
  auto doc = peek_gyre(ckpt);
  ASSERT_TRUE(doc);
  auto tok = Tokenizer::from_json(doc->to_json());
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode("The capital of France is");
  ASSERT_TRUE(ids);
  auto out = m->generate(*ids, 6, 0.f, nullptr, true);
  ASSERT_TRUE(out) << out.error().message;
  const auto text = (*tok)->decode(*out);
  EXPECT_NE(text.find("Paris"), std::string::npos) << text;
}

}  // namespace

TEST(Llama, RopePermutationTurnsRotateHalfIntoGyreRope) {
  // HF rotate_half on a head vector == gyre::rope (interleaved) on the permuted vector.
  const int hd = 12, half = hd / 2, pos = 7;
  const float theta = 1000.f;
  Rng rng(3);
  std::vector<float> hf(hd);
  for (auto& x : hf) x = rng.normal(0.f, 1.f);
  std::vector<double> want(hd);
  for (int j = 0; j < half; ++j) {
    const double a = pos * std::pow(static_cast<double>(theta), -2.0 * j / hd);
    want[j] = hf[j] * std::cos(a) - hf[j + half] * std::sin(a);
    want[j + half] = hf[j + half] * std::cos(a) + hf[j] * std::sin(a);
  }
  auto dev = Device::cpu().value();
  std::vector<float> perm(hd);
  for (int o = 0; o < hd; ++o) perm[static_cast<std::size_t>(llama_rope_perm(o, hd))] = hf[o];
  std::int64_t sh[2] = {1, hd};
  auto x = Tensor::from_host(std::as_bytes(std::span(perm)), sh, DType::f32, dev).value();
  std::int32_t p1[1] = {pos};
  std::int64_t psh[1] = {1};
  auto pt = Tensor::from_host(std::as_bytes(std::span(p1)), psh, DType::i32, dev).value();
  auto y = rope(x, pt, theta).value();
  auto py = y.host_span<float>().value();
  for (int o = 0; o < hd; ++o) EXPECT_NEAR(py[static_cast<std::size_t>(llama_rope_perm(o, hd))], want[o], 1e-5) << o;
}

TEST(Llama, ImportedTinyModelMatchesDoubleReference) {
  for (Tiny g : {Tiny{}, Tiny{.V = 41, .d = 36, .L = 2, .Hq = 9, .Hkv = 3, .hd = 6, .ff = 40, .tied = false}}) {
    Rng rng(11 + static_cast<std::uint64_t>(g.Hq));
    std::map<std::string, Mat> w;
    auto dir = write_tiny_llama(g, rng, w);
    auto m = import_llama_hf(dir);
    ASSERT_TRUE(m) << m.error().message;
    EXPECT_EQ(m->config().n_head, g.Hq);
    EXPECT_EQ(m->config().n_kv_head, g.Hkv);
    EXPECT_EQ(m->config().tie_embeddings, g.tied);
    EXPECT_EQ(m->n_params(), m->config().param_count());
    auto tok = tokens(13, g.V, rng);
    std::vector<std::int32_t> ids(tok.begin(), tok.end());
    auto got = m->forward_full(ids);
    ASSERT_TRUE(got) << got.error().message;
    auto want = reference_logits(w, g, tok);
    ASSERT_EQ(got->size(), want.size());
    EXPECT_LT(max_abs_diff(*got, want), 2e-4) << "Hq=" << g.Hq;

    // GYRE1 round trip (arch "llama", weights memory-mapped back) is exact.
    auto path = dir / "tiny.gyre";
    ASSERT_TRUE(m->save_gyre(path, "", "{\"holdout\":0.1}"));
    auto doc = peek_gyre(path);
    ASSERT_TRUE(doc);
    EXPECT_EQ(doc->arch, "llama");
    EXPECT_DOUBLE_EQ(doc->holdout(), 0.1);
    {
      auto back = LlamaLM::load_gyre(path);
      ASSERT_TRUE(back) << back.error().message;
      auto again = back->forward_full(ids);
      ASSERT_TRUE(again);
      EXPECT_EQ(*again, *got);
    }
    std::filesystem::remove_all(dir);
  }
}

TEST(Llama, KvCacheMatchesFullRecompute) {
  LlamaConfig c;
  c.vocab = 61;
  c.d_model = 40;
  c.n_layer = 3;
  c.n_head = 9;
  c.n_kv_head = 3;
  c.head_dim = 8;
  c.d_ff = 72;
  c.max_seq = 64;
  c.rope_theta = 100.f;
  Rng rng(5);
  auto m = LlamaLM::create(c, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::vector<std::int32_t> ids(23);
  for (auto& x : ids) x = static_cast<std::int32_t>(rng.u32(61));
  auto full = m->forward_full(ids);
  ASSERT_TRUE(full);
  const auto V = c.vocab;
  // prefill 7, then one chunk of 5, then single tokens
  auto cache = m->make_cache(32);
  ASSERT_TRUE(cache);
  std::vector<float> inc;
  auto a = m->forward(*cache, std::span(ids).subspan(0, 7), 7);
  ASSERT_TRUE(a);
  inc.insert(inc.end(), a->begin(), a->end());
  auto b = m->forward(*cache, std::span(ids).subspan(7, 5), 5);
  ASSERT_TRUE(b);
  inc.insert(inc.end(), b->begin(), b->end());
  for (std::size_t t = 12; t < ids.size(); ++t) {
    auto s = m->forward(*cache, std::span(ids).subspan(t, 1), 1);
    ASSERT_TRUE(s);
    inc.insert(inc.end(), s->begin(), s->end());
  }
  EXPECT_EQ(cache->len, 23);
  ASSERT_EQ(inc.size(), full->size());
  double md = 0;
  for (std::size_t i = 0; i < inc.size(); ++i) md = std::max(md, static_cast<double>(std::abs(inc[i] - (*full)[i])));
  EXPECT_LE(md, 1e-5);
  (void)V;
  // generation: cached and uncached greedy decode pick the same tokens
  std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + 5);
  auto g1 = m->generate(prompt, 12, 0.f, nullptr, true);
  auto g2 = m->generate(prompt, 12, 0.f, nullptr, false);
  ASSERT_TRUE(g1 && g2);
  EXPECT_EQ(*g1, *g2);
  EXPECT_EQ(g1->size(), 17u);
  // the cache refuses to overflow
  auto small = m->make_cache(4);
  ASSERT_TRUE(small);
  EXPECT_FALSE(m->forward(*small, std::span(ids).subspan(0, 5), 1));
}

TEST(Llama, HfConfigRejectsWhatIsNotImplemented) {
  const std::string base =
      R"({"model_type":"llama","hidden_size":8,"intermediate_size":16,"num_attention_heads":2,)"
      R"("num_key_value_heads":1,"num_hidden_layers":1,"vocab_size":10)";
  EXPECT_TRUE(LlamaConfig::from_hf_json(base + "}"));
  EXPECT_FALSE(LlamaConfig::from_hf_json(base + R"(,"rope_scaling":{"type":"linear","factor":2.0}})"));
  EXPECT_FALSE(LlamaConfig::from_hf_json(base + R"(,"attention_bias":true})"));
  EXPECT_FALSE(LlamaConfig::from_hf_json(base + R"(,"hidden_act":"gelu"})"));
  LlamaConfig c;
  c.n_head = 4;
  c.n_kv_head = 3;  // not a divisor
  EXPECT_FALSE(c.validate());
}

// Real SmolLM2-360M / 1.7B: each skips unless its converted checkpoint
// data/smollm2-<size>/smollm2-<size>.gyre exists (`gyre-cli lm import-llama`).
TEST(Llama, RealSmolLM2KnowsTheCapitalOfFrance) {
  const auto ckpt = smol_ckpt("smollm2-360m");
  if (!std::filesystem::exists(ckpt)) GTEST_SKIP() << "run gyre-cli lm import-llama first";
  expect_knows_capital_of_france(ckpt);
}

TEST(Llama, RealSmolLM2_1_7BKnowsTheCapitalOfFrance) {
  const auto ckpt = smol_ckpt("smollm2-1.7b");
  if (!std::filesystem::exists(ckpt)) GTEST_SKIP() << "run gyre-cli lm import-llama first";
  expect_knows_capital_of_france(ckpt);
}
