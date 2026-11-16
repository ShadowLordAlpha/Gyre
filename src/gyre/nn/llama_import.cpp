#include "gyre/nn/llama_import.hpp"

#include "gyre/io/safetensors.hpp"
#include "json_parse.hpp"

#include <fstream>
#include <sstream>
#include <tuple>

#if defined(GYRE_OPENMP)
#include <omp.h>
#endif

namespace gyre {
namespace {

Error bad(const std::string& m) { return make_error(Errc::ckpt_corrupt, "llama import: " + m); }

Result<std::string> read_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return std::unexpected(make_error(Errc::io, "open " + p.string()));
  std::ostringstream o;
  o << f.rdbuf();
  return o.str();
}

}  // namespace

std::int64_t llama_rope_perm(std::int64_t o, std::int64_t head_dim) noexcept {
  const auto h = o / head_dim, j = o % head_dim, half = head_dim / 2;
  return h * head_dim + (j < half ? 2 * j : 2 * (j - half) + 1);
}

Result<LlamaLM> import_llama_hf(const std::filesystem::path& dir) {
  auto cfg_text = read_text(dir / "config.json");
  if (!cfg_text) return std::unexpected(cfg_text.error());
  auto cfg = LlamaConfig::from_hf_json(*cfg_text);
  if (!cfg) return std::unexpected(cfg.error());
  if (!std::filesystem::exists(dir / "model.safetensors")) {
    if (std::filesystem::exists(dir / "model.safetensors.index.json"))
      return std::unexpected(make_error(Errc::unsupported, "llama import: sharded safetensors are not supported"));
    return std::unexpected(make_error(Errc::io, "llama import: missing " + (dir / "model.safetensors").string()));
  }
  auto st = safetensors_open(dir / "model.safetensors");
  if (!st) return std::unexpected(st.error());
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  const auto& c = *cfg;
  const auto d = c.d_model, qd = c.n_head * c.head_dim, kvd = c.n_kv_head * c.head_dim;

  // HF tensor as f32 with its [rows, cols] shape checked.
  auto load = [&](const std::string& name, std::int64_t rows, std::int64_t cols) -> Result<Tensor> {
    auto t = safetensors_load(*st, name, *dev, false);
    if (!t) return std::unexpected(bad("missing tensor " + name));
    const auto sh = t->shape();
    const bool ok = cols == 0 ? (sh.size() == 1 && sh[0] == rows) : (sh.size() == 2 && sh[0] == rows && sh[1] == cols);
    if (!ok) return std::unexpected(bad("unexpected shape for " + name));
    return t->to_f32();
  };
  // [out, in] -> [in, out]; with `rope` the output columns are permuted per head.
  auto transpose = [&](const Tensor& src, std::int64_t out, std::int64_t in, bool rope) -> Result<Tensor> {
    std::int64_t sh[2] = {in, out};
    auto dst = Tensor::uninit(sh, DType::f32, *dev);  // every element is written below
    if (!dst) return dst;
    auto ps = src.host_span<float>();
    auto pd = dst->host_span<float>();
    if (!ps || !pd) return std::unexpected(bad("host"));
    const float* s = ps->data();
    float* D = pd->data();
#if defined(GYRE_OPENMP)
    apply_thread_settings();
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t o = 0; o < out; ++o) {
      const auto col = rope ? llama_rope_perm(o, c.head_dim) : o;
      const float* row = s + o * in;
      for (std::int64_t i = 0; i < in; ++i) D[i * out + col] = row[i];
    }
    return dst;
  };

  std::vector<Tensor> ws;
  auto push = [&](Result<Tensor> t) -> Result<void> {
    if (!t) return std::unexpected(t.error());
    ws.push_back(std::move(*t));
    return {};
  };
  if (auto r = push(load("model.embed_tokens.weight", c.vocab, d)); !r) return std::unexpected(r.error());
  for (std::int64_t l = 0; l < c.n_layer; ++l) {
    const auto p = "model.layers." + std::to_string(l) + ".";
    if (auto r = push(load(p + "input_layernorm.weight", d, 0)); !r) return std::unexpected(r.error());
    for (auto&& [name, out, in, rope] : std::initializer_list<std::tuple<const char*, std::int64_t, std::int64_t, bool>>{
             {"self_attn.q_proj.weight", qd, d, true},
             {"self_attn.k_proj.weight", kvd, d, true},
             {"self_attn.v_proj.weight", kvd, d, false},
             {"self_attn.o_proj.weight", d, qd, false}}) {
      auto t = load(p + name, out, in);
      if (!t) return std::unexpected(t.error());
      if (auto r = push(transpose(*t, out, in, rope)); !r) return std::unexpected(r.error());
    }
    if (auto r = push(load(p + "post_attention_layernorm.weight", d, 0)); !r) return std::unexpected(r.error());
    for (auto&& [name, out, in] : std::initializer_list<std::tuple<const char*, std::int64_t, std::int64_t>>{
             {"mlp.gate_proj.weight", c.d_ff, d}, {"mlp.up_proj.weight", c.d_ff, d}, {"mlp.down_proj.weight", d, c.d_ff}}) {
      auto t = load(p + name, out, in);
      if (!t) return std::unexpected(t.error());
      if (auto r = push(transpose(*t, out, in, false)); !r) return std::unexpected(r.error());
    }
  }
  if (auto r = push(load("model.norm.weight", d, 0)); !r) return std::unexpected(r.error());
  if (!c.tie_embeddings) {
    auto t = load("lm_head.weight", c.vocab, d);
    if (!t) return std::unexpected(t.error());
    if (auto r = push(transpose(*t, c.vocab, d, false)); !r) return std::unexpected(r.error());
  }
  return LlamaLM::from_tensors(c, std::move(ws));
}

}  // namespace gyre
