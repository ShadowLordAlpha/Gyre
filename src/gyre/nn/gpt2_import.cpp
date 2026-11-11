#include "gyre/nn/gpt2_import.hpp"

#include "gyre/io/safetensors.hpp"
#include "json_parse.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace gyre {
namespace {

Error bad(const std::string& m) { return make_error(Errc::ckpt_corrupt, "gpt2 import: " + m); }

Result<std::string> read_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return std::unexpected(make_error(Errc::io, "open " + p.string()));
  std::ostringstream o;
  o << f.rdbuf();
  return o.str();
}

}  // namespace

Result<CharLM> import_gpt2(const std::filesystem::path& dir, std::shared_ptr<Device> device) {
  auto cfg_text = read_text(dir / "config.json");
  if (!cfg_text) return std::unexpected(cfg_text.error());
  auto pj = parse_json(*cfg_text);
  if (!pj) return std::unexpected(pj.error());
  const auto& j = *pj;
  CharLMConfig c;
  try {
    if (j.value("model_type", std::string{"gpt2"}) != "gpt2") return std::unexpected(bad("model_type is not gpt2"));
    if (j.value("activation_function", std::string{"gelu_new"}) != "gelu_new")
      return std::unexpected(bad("activation must be gelu_new (tanh GELU)"));
    if (std::abs(j.value("layer_norm_epsilon", 1e-5) - 1e-5) > 1e-12)
      return std::unexpected(bad("layer_norm_epsilon must be 1e-5"));
    if (j.value("scale_attn_by_inverse_layer_idx", false) || j.value("reorder_and_upcast_attn", false))
      return std::unexpected(bad("scale_attn_by_inverse_layer_idx / reorder_and_upcast_attn unsupported"));
    if (!j.value("scale_attn_weights", true)) return std::unexpected(bad("scale_attn_weights=false unsupported"));
    c.vocab = j.at("vocab_size").get<std::int64_t>();
    c.block_size = j.value("n_positions", j.value("n_ctx", std::int64_t{1024}));
    c.n_layer = j.at("n_layer").get<std::int64_t>();
    c.n_head = j.at("n_head").get<std::int64_t>();
    c.d_model = j.at("n_embd").get<std::int64_t>();
    const auto inner = j.contains("n_inner") && !j["n_inner"].is_null() ? j["n_inner"].get<std::int64_t>() : 0;
    c.d_ff = inner > 0 ? inner : 4 * c.d_model;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(bad(std::string("config.json: ") + e.what()));
  }
  c.recency_alibi = false;
  c.dropout = 0.f;
  if (c.n_head <= 0 || c.d_model % c.n_head != 0 || c.n_layer <= 0) return std::unexpected(bad("bad dimensions"));

  auto st = safetensors_open(dir / "model.safetensors");
  if (!st) return std::unexpected(st.error());

  Rng rng(0);  // init values are overwritten below
  auto model = CharLM::create(c, device, rng);
  if (!model) return std::unexpected(model.error());
  const auto names = model->param_names();
  auto params = model->parameters();
  if (names.size() != params.size()) return std::unexpected(bad("internal: param name count"));
  std::unordered_map<std::string, std::size_t> slot;
  for (std::size_t i = 0; i < names.size(); ++i) slot[names[i]] = i;

  const std::int64_t d = c.d_model, V = c.vocab;
  auto dst = [&](const std::string& n) -> Result<std::span<float>> {
    auto it = slot.find(n);
    if (it == slot.end()) return std::unexpected(bad("no CharLM parameter " + n));
    return params[it->second].value.host_span<float>();
  };
  // Loads `name` as f32 and checks its element count.
  auto load = [&](const std::string& name, std::int64_t expect_numel) -> Result<Tensor> {
    auto t = safetensors_load(*st, name, device, false);
    if (!t) return std::unexpected(bad("missing tensor " + name));
    auto f = t->to_f32();
    if (!f) return std::unexpected(f.error());
    if (f->numel() != expect_numel) return std::unexpected(bad("unexpected size for " + name));
    return f;
  };
  auto copy_whole = [&](const std::string& src, const std::string& dstname, std::int64_t numel) -> Result<void> {
    auto t = load(src, numel);
    if (!t) return std::unexpected(t.error());
    auto out = dst(dstname);
    if (!out) return std::unexpected(out.error());
    auto in = t->host_span<float>();
    if (!in) return std::unexpected(in.error());
    std::memcpy(out->data(), in->data(), static_cast<std::size_t>(numel) * sizeof(float));
    return {};
  };
  // Takes columns [off, off+cols) of a row-major [rows, total_cols] matrix.
  auto copy_cols = [&](const Tensor& src, std::int64_t rows, std::int64_t total_cols, std::int64_t off,
                       std::int64_t cols, const std::string& dstname) -> Result<void> {
    auto out = dst(dstname);
    if (!out) return std::unexpected(out.error());
    auto in = src.host_span<float>();
    if (!in) return std::unexpected(in.error());
    for (std::int64_t r = 0; r < rows; ++r)
      std::memcpy(out->data() + r * cols, in->data() + r * total_cols + off,
                  static_cast<std::size_t>(cols) * sizeof(float));
    return {};
  };

  // embeddings (names have no "transformer." prefix in the Hub's safetensors, but accept both)
  auto pre = std::string{};
  for (auto& t : st->tensors)
    if (t.name == "transformer.wte.weight") pre = "transformer.";
  auto step = [&](Result<void> r) -> Result<void> { return r; };
  if (auto r = step(copy_whole(pre + "wte.weight", "wte.weight", V * d)); !r) return std::unexpected(r.error());
  if (auto r = step(copy_whole(pre + "wpe.weight", "wpe.weight", c.block_size * d)); !r)
    return std::unexpected(r.error());

  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    const auto h = pre + "h." + std::to_string(i) + ".";
    const auto b = "blocks." + std::to_string(i) + ".";
    for (auto&& [src, dn, n] : std::initializer_list<std::tuple<const char*, const char*, std::int64_t>>{
             {"ln_1.weight", "ln1.weight", d}, {"ln_1.bias", "ln1.bias", d},
             {"ln_2.weight", "ln2.weight", d}, {"ln_2.bias", "ln2.bias", d},
             {"attn.c_proj.weight", "attn.o.weight", d * d}, {"attn.c_proj.bias", "attn.o.bias", d},
             {"mlp.c_fc.weight", "mlp.fc1.weight", d * c.d_ff}, {"mlp.c_fc.bias", "mlp.fc1.bias", c.d_ff},
             {"mlp.c_proj.weight", "mlp.fc2.weight", c.d_ff * d}, {"mlp.c_proj.bias", "mlp.fc2.bias", d}}) {
      if (auto r = copy_whole(h + src, b + dn, n); !r) return std::unexpected(r.error());
    }
    // fused qkv: weight [d, 3d], bias [3d]
    auto w = load(h + "attn.c_attn.weight", d * 3 * d);
    auto bb = load(h + "attn.c_attn.bias", 3 * d);
    if (!w) return std::unexpected(w.error());
    if (!bb) return std::unexpected(bb.error());
    const char* part[3] = {"q", "k", "v"};
    for (int p = 0; p < 3; ++p) {
      if (auto r = copy_cols(*w, d, 3 * d, p * d, d, b + "attn." + part[p] + ".weight"); !r)
        return std::unexpected(r.error());
      if (auto r = copy_cols(*bb, 1, 3 * d, p * d, d, b + "attn." + part[p] + ".bias"); !r)
        return std::unexpected(r.error());
    }
  }
  if (auto r = copy_whole(pre + "ln_f.weight", "ln_f.weight", d); !r) return std::unexpected(r.error());
  if (auto r = copy_whole(pre + "ln_f.bias", "ln_f.bias", d); !r) return std::unexpected(r.error());

  // tied output head: lm_head.weight [d, V] = wte^T, bias = 0
  {
    auto wte = load(pre + "wte.weight", V * d);
    if (!wte) return std::unexpected(wte.error());
    auto in = wte->host_span<float>();
    auto hw = dst("lm_head.weight");
    auto hb = dst("lm_head.bias");
    if (!in || !hw || !hb) return std::unexpected(bad("lm_head"));
    for (std::int64_t v = 0; v < V; ++v)
      for (std::int64_t k = 0; k < d; ++k) (*hw)[static_cast<std::size_t>(k * V + v)] = (*in)[static_cast<std::size_t>(v * d + k)];
    std::fill(hb->begin(), hb->end(), 0.f);
  }
  return model;
}

}  // namespace gyre
