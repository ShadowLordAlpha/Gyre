// GPT-2 -> CharLM conversion. The synthetic test builds a tiny GPT-2-format safetensors directory
// with random weights and checks the imported CharLM against an independent double-precision
// reference of the published GPT-2 forward pass (fused c_attn, tied head, gelu_new, pre-LN).
// The real-model test is skipped unless data/gpt2 has been downloaded.
#include "gradcheck.hpp"

#include "gyre/io/safetensors.hpp"
#include "gyre/nn/gpt2_import.hpp"
#include "gyre/nn/tokenize.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>

using namespace gyre;
using namespace gyre::testing;

namespace {

struct Tiny {
  int V = 11, P = 16, L = 2, H = 2, d = 8, ff = 32;
};

using Mat = std::vector<double>;

Mat as_d(const Tensor& t) {
  auto p = t.host_span<float>();
  return Mat(p->begin(), p->end());
}

std::filesystem::path write_tiny_gpt2(const Tiny& g, Rng& rng, const char* activation = "gelu_new") {
  auto dir = std::filesystem::temp_directory_path() / ("gyre_gpt2_" + std::to_string(rng.u32(1u << 30)));
  std::filesystem::create_directories(dir);
  {
    std::ofstream c(dir / "config.json");
    c << "{\"model_type\":\"gpt2\",\"activation_function\":\"" << activation << "\",\"layer_norm_epsilon\":1e-05,"
      << "\"n_ctx\":" << g.P << ",\"n_positions\":" << g.P << ",\"n_embd\":" << g.d << ",\"n_head\":" << g.H
      << ",\"n_layer\":" << g.L << ",\"vocab_size\":" << g.V << "}";
  }
  std::map<std::string, Tensor> t;
  auto add = [&](const std::string& n, std::vector<std::int64_t> sh, float sd, float mean = 0.f) {
    auto x = rand_t(sh, rng, sd);
    for (auto& v : *x.host_span<float>()) v += mean;
    t.emplace(n, std::move(x));
  };
  add("wte.weight", {g.V, g.d}, 0.5f);
  add("wpe.weight", {g.P, g.d}, 0.3f);
  for (int i = 0; i < g.L; ++i) {
    const std::string h = "h." + std::to_string(i) + ".";
    add(h + "ln_1.weight", {g.d}, 0.1f, 1.f);
    add(h + "ln_1.bias", {g.d}, 0.1f);
    add(h + "attn.c_attn.weight", {g.d, 3 * g.d}, 0.4f);
    add(h + "attn.c_attn.bias", {3 * g.d}, 0.1f);
    add(h + "attn.c_proj.weight", {g.d, g.d}, 0.4f);
    add(h + "attn.c_proj.bias", {g.d}, 0.1f);
    add(h + "attn.bias", {1, 1, g.P, g.P}, 0.f);  // causal-mask buffer: must be ignored
    add(h + "ln_2.weight", {g.d}, 0.1f, 1.f);
    add(h + "ln_2.bias", {g.d}, 0.1f);
    add(h + "mlp.c_fc.weight", {g.d, g.ff}, 0.4f);
    add(h + "mlp.c_fc.bias", {g.ff}, 0.1f);
    add(h + "mlp.c_proj.weight", {g.ff, g.d}, 0.4f);
    add(h + "mlp.c_proj.bias", {g.d}, 0.1f);
  }
  add("ln_f.weight", {g.d}, 0.1f, 1.f);
  add("ln_f.bias", {g.d}, 0.1f);
  std::vector<NamedTensor> nt;
  for (auto& [n, x] : t) nt.push_back({n, &x});
  auto r = safetensors_save(dir / "model.safetensors", nt);
  if (!r) std::abort();
  return dir;
}

// Reference GPT-2 forward in double precision, straight from the model definition.
Mat reference_logits(const std::map<std::string, Mat>& w, const Tiny& g, const std::vector<int>& tok) {
  const int T = static_cast<int>(tok.size()), d = g.d, dk = d / g.H;
  auto ln = [&](const Mat& x, const Mat& gamma, const Mat& beta) {
    Mat y(x.size());
    for (int t = 0; t < T; ++t) {
      double m = 0, v = 0;
      for (int i = 0; i < d; ++i) m += x[t * d + i];
      m /= d;
      for (int i = 0; i < d; ++i) v += (x[t * d + i] - m) * (x[t * d + i] - m);
      v /= d;
      for (int i = 0; i < d; ++i) y[t * d + i] = (x[t * d + i] - m) / std::sqrt(v + 1e-5) * gamma[i] + beta[i];
    }
    return y;
  };
  auto matmul = [&](const Mat& x, const Mat& W, const Mat& b, int in, int out) {
    Mat y(static_cast<std::size_t>(T) * out);
    for (int t = 0; t < T; ++t)
      for (int o = 0; o < out; ++o) {
        double s = b[o];
        for (int i = 0; i < in; ++i) s += x[t * in + i] * W[i * out + o];  // Conv1D: W is [in, out]
        y[t * out + o] = s;
      }
    return y;
  };
  Mat x(static_cast<std::size_t>(T) * d);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < d; ++i) x[t * d + i] = w.at("wte.weight")[tok[t] * d + i] + w.at("wpe.weight")[t * d + i];
  for (int l = 0; l < g.L; ++l) {
    const std::string h = "h." + std::to_string(l) + ".";
    Mat a = ln(x, w.at(h + "ln_1.weight"), w.at(h + "ln_1.bias"));
    Mat qkv = matmul(a, w.at(h + "attn.c_attn.weight"), w.at(h + "attn.c_attn.bias"), d, 3 * d);
    Mat y(static_cast<std::size_t>(T) * d, 0.0);
    for (int hd = 0; hd < g.H; ++hd)
      for (int i = 0; i < T; ++i) {
        std::vector<double> s(i + 1);
        double mx = -1e300;
        for (int j = 0; j <= i; ++j) {
          double dot = 0;
          for (int e = 0; e < dk; ++e) dot += qkv[i * 3 * d + hd * dk + e] * qkv[j * 3 * d + d + hd * dk + e];
          s[j] = dot / std::sqrt(double(dk));
          mx = std::max(mx, s[j]);
        }
        double z = 0;
        for (auto& v : s) z += (v = std::exp(v - mx));
        for (int j = 0; j <= i; ++j)
          for (int e = 0; e < dk; ++e) y[i * d + hd * dk + e] += s[j] / z * qkv[j * 3 * d + 2 * d + hd * dk + e];
      }
    Mat o = matmul(y, w.at(h + "attn.c_proj.weight"), w.at(h + "attn.c_proj.bias"), d, d);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    Mat m = ln(x, w.at(h + "ln_2.weight"), w.at(h + "ln_2.bias"));
    Mat f = matmul(m, w.at(h + "mlp.c_fc.weight"), w.at(h + "mlp.c_fc.bias"), d, g.ff);
    for (auto& v : f) v = 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));  // gelu_new
    Mat p = matmul(f, w.at(h + "mlp.c_proj.weight"), w.at(h + "mlp.c_proj.bias"), g.ff, d);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += p[i];
  }
  x = ln(x, w.at("ln_f.weight"), w.at("ln_f.bias"));
  Mat logits(static_cast<std::size_t>(T) * g.V);
  for (int t = 0; t < T; ++t)
    for (int v = 0; v < g.V; ++v) {
      double s = 0;
      for (int i = 0; i < d; ++i) s += x[t * d + i] * w.at("wte.weight")[v * d + i];  // tied head
      logits[t * g.V + v] = s;
    }
  return logits;
}

}  // namespace

TEST(Gpt2Import, MatchesReferenceForward) {
  Rng rng(1);
  Tiny g;
  auto dir = write_tiny_gpt2(g, rng);
  auto d = Device::cpu();
  auto model = import_gpt2(dir, *d);
  ASSERT_TRUE(model) << model.error().message;
  EXPECT_EQ(model->vocab(), g.V);
  EXPECT_EQ(model->block_size(), g.P);
  EXPECT_FALSE(model->config().recency_alibi);

  // load the same weights as doubles for the reference
  auto st = safetensors_open(dir / "model.safetensors");
  ASSERT_TRUE(st);
  std::map<std::string, Mat> w;
  for (auto& info : st->tensors) w[info.name] = as_d(*safetensors_load(*st, info.name, *d, false));

  const std::vector<int> tok = {3, 7, 1, 1, 9, 0, 4, 10, 2, 5};
  std::vector<std::int32_t> ids(tok.begin(), tok.end());
  std::int64_t sh[2] = {1, static_cast<std::int64_t>(tok.size())};
  auto idx = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(ids)), sh, DType::i32, *d);
  ASSERT_TRUE(idx);
  ForwardCtx ctx;
  ctx.train = false;
  auto got = model->forward(*idx, ctx);
  ASSERT_TRUE(got) << got.error().message;
  const Mat want = reference_logits(w, g, tok);
  auto pg = got->host_span<float>();
  ASSERT_EQ(pg->size(), want.size());
  double worst = 0;
  for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(double((*pg)[i]) - want[i]));
  EXPECT_LT(worst, 2e-3) << "max |logit difference| vs the GPT-2 reference";
  std::filesystem::remove_all(dir);
}

TEST(Gpt2Import, RejectsUnsupportedVariantsAndMissingFiles) {
  Rng rng(2);
  Tiny g;
  auto d = Device::cpu();
  auto wrong_act = write_tiny_gpt2(g, rng, "relu");
  auto r = import_gpt2(wrong_act, *d);
  EXPECT_FALSE(r);
  EXPECT_NE(r.error().message.find("gelu_new"), std::string::npos) << r.error().message;
  std::filesystem::remove_all(wrong_act);
  EXPECT_FALSE(import_gpt2("definitely/not/a/dir", *d));
  // weights missing -> clear error
  auto empty = std::filesystem::temp_directory_path() / "gyre_gpt2_empty";
  std::filesystem::create_directories(empty);
  std::ofstream(empty / "config.json") << "{\"vocab_size\":5,\"n_layer\":1,\"n_head\":1,\"n_embd\":4}";
  EXPECT_FALSE(import_gpt2(empty, *d));
  std::filesystem::remove_all(empty);
}

TEST(Gpt2Import, RealGpt2SmallPredictsEnglish) {
  const std::filesystem::path dir = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "gpt2";
  if (!std::filesystem::exists(dir / "model.safetensors")) GTEST_SKIP() << "data/gpt2 not downloaded";
  auto d = Device::cpu();
  auto model = import_gpt2(dir, *d);
  ASSERT_TRUE(model) << model.error().message;
  EXPECT_EQ(model->config().n_layer, 12);
  EXPECT_EQ(model->config().d_model, 768);
  auto tok = Tokenizer::load_huggingface(dir);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->vocab_size(), 50257);
  const std::string text = "The quick brown fox jumps over the lazy dog. It was the best of times, it was the worst of times.";
  auto enc = (*tok)->encode(text);
  ASSERT_TRUE(enc);
  ASSERT_GT(enc->size(), 10u);
  std::vector<std::int32_t> ids(enc->begin(), enc->end());
  std::int64_t sh[2] = {1, static_cast<std::int64_t>(ids.size())};
  auto idx = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(ids)), sh, DType::i32, *d);
  ForwardCtx ctx;
  ctx.train = false;
  auto logits = model->forward(*idx, ctx);
  ASSERT_TRUE(logits) << logits.error().message;
  const auto V = 50257;
  auto pl = logits->host_span<float>();
  double nll = 0;
  for (std::size_t t = 0; t + 1 < ids.size(); ++t) {
    const float* row = pl->data() + t * V;
    float mx = row[0];
    for (int v = 1; v < V; ++v) mx = std::max(mx, row[v]);
    double z = 0;
    for (int v = 0; v < V; ++v) z += std::exp(double(row[v] - mx));
    nll += -(double(row[ids[t + 1]] - mx) - std::log(z));
  }
  nll /= double(ids.size() - 1);
  std::printf("[ gpt2    ] %zu tokens, %.3f nats/token (random weights would be ~10.8)\n", ids.size(), nll);
  EXPECT_LT(nll, 6.0) << "real GPT-2 weights should model plain English far better than chance";
}
