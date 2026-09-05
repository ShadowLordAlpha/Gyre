#include "gyre/nn/grok.hpp"

#include "gyre/io/safetensors.hpp"
#include "gyre/nn/gqa.hpp"
#include "gyre/nn/moe.hpp"
#include "json_parse.hpp"
#include "gyre/ops.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <span>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace gyre {
namespace {

Result<Param> init_linear(std::int64_t in, std::int64_t out, std::shared_ptr<Device> d, Rng& rng,
                          float std) {
  std::int64_t sh[2] = {in, out};
  auto W = Tensor::empty(sh, DType::f32, d);
  if (!W) return std::unexpected(W.error());
  auto p = W->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = rng.normal(0.f, std);
  return make_param(std::move(*W));
}

Result<Param> init_rms(std::int64_t n, std::shared_ptr<Device> d) {
  std::int64_t sh[1] = {n};
  auto w = Tensor::empty(sh, DType::f32, d);
  if (!w) return std::unexpected(w.error());
  auto p = w->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = 1.f;
  return make_param(std::move(*w));
}

Result<Param> init_embed(std::int64_t v, std::int64_t d, std::shared_ptr<Device> dev, Rng& rng) {
  std::int64_t sh[2] = {v, d};
  auto W = Tensor::empty(sh, DType::f32, dev);
  if (!W) return std::unexpected(W.error());
  auto p = W->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& x : *p) x = rng.normal(0.f, 0.02f);
  return make_param(std::move(*W));
}

Result<Tensor> scale_f32(const Tensor& x, float s) {
  auto y = x.clone();
  if (!y) return y;
  auto p = y->host_span<float>();
  if (!p) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (auto& v : *p) v *= s;
  return y;
}

Result<Tensor> linear_nb(const Tensor& x, const Tensor& W) {
  auto x2 = flatten_leading(x);
  if (!x2) return x2;
  auto y = matmul(*x2, W);
  if (!y) return y;
  return unflatten_like(std::move(*y), x);
}

std::int64_t ji(const nlohmann::json& j, const char* key, std::int64_t def) {
  if (!j.contains(key)) return def;
  return j[key].get<std::int64_t>();
}
float jf(const nlohmann::json& j, const char* key, float def) {
  if (!j.contains(key)) return def;
  return j[key].get<float>();
}
bool jbool(const nlohmann::json& j, const char* key, bool def) {
  if (!j.contains(key)) return def;
  return j[key].get<bool>();
}

constexpr int kLayerBase = 3;  // embed, lm_head, ln_f

int layer_stride(const GrokConfig& c) {
  return 12 + 3 * static_cast<int>(c.n_experts);
}

}  // namespace

GrokConfig GrokConfig::tiny() {
  GrokConfig c;
  c.vocab = 32;
  c.block_size = 16;
  c.n_layer = 2;
  c.n_q_head = 4;
  c.n_kv_head = 1;
  c.d_model = 32;
  c.head_dim = 8;
  c.d_ff = 64;
  c.moe_ff = 32;
  c.n_experts = 2;
  c.n_experts_per_tok = 2;
  c.rope_theta = 10000.f;
  c.rope_scale = 1.f;
  c.embedding_scale = 1.f;
  c.output_scale = 1.f;
  c.attn_temperature_len = 16;
  return c;
}

GrokConfig GrokConfig::mini() {
  GrokConfig c = tiny();
  c.n_layer = 1;
  c.d_model = 16;
  c.head_dim = 4;
  c.n_q_head = 4;
  c.n_kv_head = 1;
  c.d_ff = 32;
  c.moe_ff = 16;
  c.block_size = 8;
  c.vocab = 16;
  return c;
}

GrokConfig GrokConfig::full() {
  GrokConfig c;
  c.vocab = 131072;
  c.block_size = 131072;
  c.n_layer = 64;
  c.n_q_head = 64;
  c.n_kv_head = 8;
  c.d_model = 8192;
  c.head_dim = 128;
  c.d_ff = 32768;
  c.moe_ff = 16384;
  c.n_experts = 8;
  c.n_experts_per_tok = 2;
  c.rms_eps = 1e-5f;
  c.rope_theta = 208533496.f;
  c.rope_scale = 16.f;
  c.embedding_scale = 90.50966799187809f;
  c.output_scale = 0.5f;
  c.attn_softcap = 30.f;
  c.router_softcap = 30.f;
  c.final_softcap = 50.f;
  c.attn_temperature_len = 1024;
  c.residual_moe = true;
  return c;
}

Result<GrokConfig> GrokConfig::from_json(std::string_view json) {
  auto parsed = parse_json(json);
  if (!parsed) return std::unexpected(parsed.error());
  try {
  auto& j = *parsed;
  GrokConfig c = full();
  c.vocab = ji(j, "vocab_size", c.vocab);
  c.block_size = ji(j, "max_position_embeddings", c.block_size);
  c.n_layer = ji(j, "num_hidden_layers", c.n_layer);
  c.n_q_head = ji(j, "num_attention_heads", c.n_q_head);
  c.n_kv_head = ji(j, "num_key_value_heads", c.n_kv_head);
  c.d_model = ji(j, "hidden_size", c.d_model);
  c.head_dim = ji(j, "head_dim", c.head_dim);
  c.d_ff = ji(j, "intermediate_size", c.d_ff);
  c.moe_ff = ji(j, "moe_intermediate_size", c.moe_ff);
  c.n_experts = ji(j, "num_local_experts", c.n_experts);
  c.n_experts_per_tok = ji(j, "num_experts_per_tok", c.n_experts_per_tok);
  c.rms_eps = jf(j, "rms_norm_eps", c.rms_eps);
  c.rope_theta = jf(j, "rope_theta", c.rope_theta);
  c.rope_scale = jf(j, "scaling_factor", c.rope_scale);
  c.embedding_scale = jf(j, "embedding_multiplier_scale", c.embedding_scale);
  c.output_scale = jf(j, "output_multiplier_scale", c.output_scale);
  c.attn_softcap = jf(j, "attn_logit_softcapping", c.attn_softcap);
  c.router_softcap = jf(j, "router_logit_softcapping", c.router_softcap);
  c.final_softcap = jf(j, "final_logit_softcapping", c.final_softcap);
  c.attn_temperature_len = ji(j, "attn_temperature_len", c.attn_temperature_len);
  c.residual_moe = jbool(j, "residual_moe", c.residual_moe);
  return c;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("json: ") + e.what()));
  }
}

Result<GrokConfig> GrokConfig::from_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, path.string()));
  std::ostringstream ss;
  ss << in.rdbuf();
  return from_json(ss.str());
}

std::int64_t GrokConfig::param_count() const {
  const auto D = d_model;
  const auto q = D * (n_q_head * head_dim);
  const auto kv = D * (n_kv_head * head_dim);
  const auto o = (n_q_head * head_dim) * D;
  const auto dense = 2 * D * d_ff + d_ff * D;
  const auto router = D * n_experts;
  const auto exp = n_experts * (2 * D * moe_ff + moe_ff * D);
  const auto rms = 4 * D;
  const auto layer = q + kv + kv + o + dense + router + exp + rms;
  return 2 * vocab * D + D + n_layer * layer;
}

bool GrokConfig::fits_in_ram_create() const {
  if (d_model > 256 || vocab > 4096 || n_layer > 8) return false;
  return param_count() < 8'000'000;
}

std::string GrokConfig::to_string() const {
  std::ostringstream o;
  o << "GrokConfig V=" << vocab << " L=" << n_layer << " d=" << d_model << " Hq=" << n_q_head
    << " Hkv=" << n_kv_head << " E=" << n_experts << " k=" << n_experts_per_tok
    << " params=" << param_count();
  return o.str();
}

std::string GrokConfig::to_json() const {
  nlohmann::json j;
  j["vocab_size"] = vocab;
  j["max_position_embeddings"] = block_size;
  j["num_hidden_layers"] = n_layer;
  j["num_attention_heads"] = n_q_head;
  j["num_key_value_heads"] = n_kv_head;
  j["hidden_size"] = d_model;
  j["head_dim"] = head_dim;
  j["intermediate_size"] = d_ff;
  j["moe_intermediate_size"] = moe_ff;
  j["num_local_experts"] = n_experts;
  j["num_experts_per_tok"] = n_experts_per_tok;
  j["rms_norm_eps"] = rms_eps;
  j["rope_theta"] = rope_theta;
  j["scaling_factor"] = rope_scale;
  j["embedding_multiplier_scale"] = embedding_scale;
  j["output_multiplier_scale"] = output_scale;
  j["attn_logit_softcapping"] = attn_softcap;
  j["router_logit_softcapping"] = router_softcap;
  j["final_logit_softcapping"] = final_softcap;
  j["attn_temperature_len"] = attn_temperature_len;
  j["residual_moe"] = residual_moe;
  return j.dump();
}

Result<GrokLM> GrokLM::create(GrokConfig c, std::shared_ptr<Device> d, Rng& rng) {
  if (c.n_q_head <= 0 || c.n_kv_head <= 0 || c.n_q_head % c.n_kv_head != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "Grok heads"));
  }
  if (c.n_experts_per_tok < 1 || c.n_experts_per_tok > c.n_experts) {
    return std::unexpected(make_error(Errc::invalid_shape, "Grok experts"));
  }
  if (!c.fits_in_ram_create()) {
    return std::unexpected(make_error(Errc::unsupported, "GrokLM::create is tiny-only; bind weights later"));
  }
  GrokLM m(c);
  auto emb = init_embed(c.vocab, c.d_model, d, rng);
  if (!emb) return std::unexpected(emb.error());
  m.params_.push_back(std::move(*emb));
  auto head = init_linear(c.d_model, c.vocab, d, rng, 0.02f);
  if (!head) return std::unexpected(head.error());
  m.params_.push_back(std::move(*head));
  auto lnf = init_rms(c.d_model, d);
  if (!lnf) return std::unexpected(lnf.error());
  m.params_.push_back(std::move(*lnf));

  const float std = 0.02f;
  for (std::int64_t L = 0; L < c.n_layer; ++L) {
    auto push_l = [&](Result<Param> p) -> Result<void> {
      if (!p) return std::unexpected(p.error());
      m.params_.push_back(std::move(*p));
      return {};
    };
    if (auto e = push_l(init_rms(c.d_model, d)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.n_q_head * c.head_dim, d, rng, std)); !e)
      return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.n_kv_head * c.head_dim, d, rng, std)); !e)
      return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.n_kv_head * c.head_dim, d, rng, std)); !e)
      return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.n_q_head * c.head_dim, c.d_model, d, rng, std)); !e)
      return std::unexpected(e.error());
    if (auto e = push_l(init_rms(c.d_model, d)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_rms(c.d_model, d)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.d_ff, d, rng, std)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.d_ff, d, rng, std)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_ff, c.d_model, d, rng, std)); !e) return std::unexpected(e.error());
    if (auto e = push_l(init_linear(c.d_model, c.n_experts, d, rng, std)); !e)
      return std::unexpected(e.error());
    for (std::int64_t e = 0; e < c.n_experts; ++e) {
      if (auto r = push_l(init_linear(c.d_model, c.moe_ff, d, rng, std)); !r)
        return std::unexpected(r.error());
      if (auto r = push_l(init_linear(c.d_model, c.moe_ff, d, rng, std)); !r)
        return std::unexpected(r.error());
      if (auto r = push_l(init_linear(c.moe_ff, c.d_model, d, rng, std)); !r)
        return std::unexpected(r.error());
    }
    if (auto r = push_l(init_rms(c.d_model, d)); !r) return std::unexpected(r.error());
  }
  return m;
}

Result<Tensor> GrokLM::forward(const Tensor& idx, ForwardCtx& ctx) {
  (void)ctx;
  if (idx.dtype() != DType::i32 || idx.rank() != 2) {
    return std::unexpected(make_error(Errc::invalid_shape, "GrokLM idx [B,T] i32"));
  }
  const auto B = idx.shape()[0], T = idx.shape()[1];
  const auto hd = cfg_.head_dim;
  auto h = embedding(params_[0].value, idx);
  if (!h) return h;
  if (cfg_.embedding_scale != 1.f) {
    auto s = scale_f32(*h, cfg_.embedding_scale);
    if (!s) return s;
    h = std::move(s);
  }

  std::vector<std::int32_t> pos(static_cast<std::size_t>(T));
  for (std::int64_t t = 0; t < T; ++t) pos[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(t);
  std::int64_t psh[] = {T};
  auto post = Tensor::from_host(std::as_bytes(std::span(pos.data(), pos.size())), psh, DType::i32,
                                idx.device());
  if (!post) return std::unexpected(post.error());

  const auto stride = layer_stride(cfg_);
  const float atemp = attn_temperature_scale(T, cfg_.attn_temperature_len);

  for (std::int64_t L = 0; L < cfg_.n_layer; ++L) {
    Param* p = params_.data() + kLayerBase + L * stride;
    auto n1 = rms_norm(*h, p[0].value, cfg_.rms_eps);
    if (!n1) return n1;
    const GrokLoraPair *lq = nullptr, *lk = nullptr, *lv = nullptr, *lo = nullptr;
    if (lora_) {
      const auto li = static_cast<std::size_t>(L);
      if (li < lora_->q.size()) lq = &lora_->q[li];
      if (li < lora_->k.size()) lk = &lora_->k[li];
      if (li < lora_->v.size()) lv = &lora_->v[li];
      if (li < lora_->o.size()) lo = &lora_->o[li];
    }
    auto q = linear_lora(*n1, p[1].value, lq);
    auto k = linear_lora(*n1, p[2].value, lk);
    auto v = linear_lora(*n1, p[3].value, lv);
    if (!q || !k || !v) return std::unexpected(q ? (k ? v.error() : k.error()) : q.error());
    std::int64_t qsh[4] = {B, T, cfg_.n_q_head, hd};
    std::int64_t ksh[4] = {B, T, cfg_.n_kv_head, hd};
    auto qr = reshape(*q, qsh);
    auto kr = reshape(*k, ksh);
    auto vr = reshape(*v, ksh);
    if (!qr || !kr || !vr) return std::unexpected(make_error(Errc::invalid_shape, "qkv reshape"));
    auto qh = permute_bthd_bhtd(*qr);
    auto kh = permute_bthd_bhtd(*kr);
    auto vh = permute_bthd_bhtd(*vr);
    if (!qh || !kh || !vh) return std::unexpected(qh ? (kh ? vh.error() : kh.error()) : qh.error());
    auto qrot = rope(*qh, *post, cfg_.rope_theta, cfg_.rope_scale);
    auto krot = rope(*kh, *post, cfg_.rope_theta, cfg_.rope_scale);
    if (!qrot) return qrot;
    if (!krot) return krot;
    auto att = gqa_causal(*qrot, *krot, *vh, cfg_.attn_softcap, atemp);
    if (!att) return att;
    auto yb = permute_bhtd_bthd(*att);
    if (!yb) return yb;
    std::int64_t o3[3] = {B, T, cfg_.n_q_head * hd};
    auto y3 = reshape(*yb, o3);
    if (!y3) return y3;
    auto proj = linear_lora(*y3, p[4].value, lo);
    if (!proj) return proj;
    auto post_a = rms_norm(*proj, p[5].value, cfg_.rms_eps);
    if (!post_a) return post_a;
    auto h1 = add(*h, *post_a);
    if (!h1) return h1;

    auto n2 = rms_norm(*h1, p[6].value, cfg_.rms_eps);
    if (!n2) return n2;
    SwiGLUWeights dense{&p[7].value, &p[8].value, &p[9].value};
    std::vector<SwiGLUWeights> ex(static_cast<std::size_t>(cfg_.n_experts));
    int off = 11;
    for (std::int64_t e = 0; e < cfg_.n_experts; ++e) {
      ex[static_cast<std::size_t>(e)] = SwiGLUWeights{&p[off].value, &p[off + 1].value, &p[off + 2].value};
      off += 3;
    }
    auto ffn = residual_moe(*n2, dense, p[10].value, ex,
                            static_cast<int>(cfg_.n_experts_per_tok), cfg_.router_softcap);
    if (!cfg_.residual_moe) ffn = swiglu_ffn(*n2, dense);
    if (!ffn) return ffn;
    auto post_m = rms_norm(*ffn, p[off].value, cfg_.rms_eps);
    if (!post_m) return post_m;
    auto h2 = add(*h1, *post_m);
    if (!h2) return h2;
    h = std::move(h2);
  }

  auto nf = rms_norm(*h, params_[2].value, cfg_.rms_eps);
  if (!nf) return nf;
  auto logits = linear_nb(*nf, params_[1].value);
  if (!logits) return logits;
  if (cfg_.output_scale != 1.f) {
    auto s = scale_f32(*logits, cfg_.output_scale);
    if (!s) return s;
    logits = std::move(s);
  }
  if (cfg_.final_softcap > 0.f) return softcap(*logits, cfg_.final_softcap);
  return logits;
}

Result<void> GrokLM::backward(const Tensor&, ForwardCtx&) {
  return std::unexpected(make_error(Errc::unsupported, "GrokLM backward later"));
}

Result<std::vector<std::int32_t>> GrokLM::generate(std::vector<std::int32_t> prefix, int max_new,
                                                   std::shared_ptr<Device> d, Rng* rng,
                                                   float temperature) {
  if (prefix.empty()) prefix.push_back(0);
  ForwardCtx ctx;
  ctx.train = false;
  for (int n = 0; n < max_new; ++n) {
    if (static_cast<std::int64_t>(prefix.size()) > cfg_.block_size) {
      prefix.erase(prefix.begin(), prefix.begin() + (prefix.size() - static_cast<std::size_t>(cfg_.block_size)));
    }
    const auto T = static_cast<std::int64_t>(prefix.size());
    std::int64_t sh[] = {1, T};
    auto idx = Tensor::from_host(std::as_bytes(std::span(prefix.data(), prefix.size())), sh, DType::i32, d);
    if (!idx) return std::unexpected(idx.error());
    auto logits = forward(*idx, ctx);
    if (!logits) return std::unexpected(logits.error());
    auto p = logits->host_span<float>();
    if (!p) return std::unexpected(make_error(Errc::not_cpu, "host"));
    const auto V = cfg_.vocab;
    const float* row = p->data() + (T - 1) * V;
    auto tok = sample_logit_row(std::span<const float>(row, static_cast<std::size_t>(V)), temperature, rng);
    if (!tok) return std::unexpected(tok.error());
    prefix.push_back(*tok);
  }
  return prefix;
}

std::vector<std::string> GrokLM::param_names() const {
  std::vector<std::string> n;
  n.push_back("model.embed_tokens.weight");
  n.push_back("lm_head.weight");
  n.push_back("model.norm.weight");
  for (std::int64_t L = 0; L < cfg_.n_layer; ++L) {
    const auto p = "model.layers." + std::to_string(L) + ".";
    n.push_back(p + "pre_attn_norm.weight");
    n.push_back(p + "self_attn.q_proj.weight");
    n.push_back(p + "self_attn.k_proj.weight");
    n.push_back(p + "self_attn.v_proj.weight");
    n.push_back(p + "self_attn.o_proj.weight");
    n.push_back(p + "post_attn_norm.weight");
    n.push_back(p + "pre_moe_norm.weight");
    n.push_back(p + "mlp.gate_proj.weight");
    n.push_back(p + "mlp.up_proj.weight");
    n.push_back(p + "mlp.down_proj.weight");
    n.push_back(p + "mlp.router.weight");
    for (std::int64_t e = 0; e < cfg_.n_experts; ++e) {
      const auto ep = p + "mlp.experts." + std::to_string(e) + ".";
      n.push_back(ep + "gate_proj.weight");
      n.push_back(ep + "up_proj.weight");
      n.push_back(ep + "down_proj.weight");
    }
    n.push_back(p + "post_moe_norm.weight");
  }
  return n;
}

Result<Tensor> GrokLM::linear_lora(const Tensor& x, const Tensor& W, const GrokLoraPair* slot) const {
  auto y = linear_nb(x, W);
  if (!y) return y;
  if (!slot || !lora_ || lora_->rank <= 0) return y;
  auto u = linear_nb(x, slot->A);
  if (!u) return u;
  auto v = linear_nb(*u, slot->B);
  if (!v) return v;
  auto s = scale_f32(*v, lora_->scale());
  if (!s) return s;
  return add(*y, *s);
}

Result<void> GrokLM::save_weights(const std::filesystem::path& dir) const {
  if (dir.extension() == ".gyre") return save_gyre(dir);
  std::filesystem::create_directories(dir);
  {
    std::ofstream cfg(dir / "config.json", std::ios::binary);
    if (!cfg) return std::unexpected(make_error(Errc::io, "config.json"));
    auto j = cfg_.to_json();
    cfg.write(j.data(), static_cast<std::streamsize>(j.size()));
  }
  auto names = param_names();
  if (names.size() != params_.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "name/param count"));
  }
  std::vector<NamedTensor> nt;
  nt.reserve(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) nt.push_back({names[i], &params_[i].value});
  return safetensors_save(dir / "model.safetensors", nt);
}

Result<GrokLM> GrokLM::load_weights(const std::filesystem::path& dir, std::shared_ptr<Device> d) {
  if (std::filesystem::is_regular_file(dir)) return load_gyre(dir, d);
  auto c = GrokConfig::from_file(dir / "config.json");
  if (!c) return std::unexpected(c.error());
  Rng rng(1);
  auto m = create(*c, d, rng);
  if (!m) return m;
  auto st = safetensors_open(dir / "model.safetensors");
  if (!st) return std::unexpected(st.error());
  auto names = m->param_names();
  if (names.size() != m->params_.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "load names"));
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    auto t = safetensors_load(*st, names[i], d, false);
    if (!t) return std::unexpected(t.error());
    if (t->dtype() != DType::f32) {
      auto f = t->to_f32();
      if (!f) return std::unexpected(f.error());
      t = std::move(f);
    }
    auto p = make_param(std::move(*t));
    if (!p) return std::unexpected(p.error());
    m->params_[i] = std::move(*p);
  }
  return m;
}

Result<void> GrokLM::save_gyre(const std::filesystem::path& path, GyreCodec codec) const {
  GyreDoc d;
  d.arch = "grok2";
  d.config_json = cfg_.to_json();
  auto names = param_names();
  return gyre::save_gyre(path, params_, nullptr, d, names, codec);
}

Result<GrokLM> GrokLM::load_gyre(const std::filesystem::path& path, std::shared_ptr<Device> d) {
  auto peek = peek_gyre(path);
  if (!peek) return std::unexpected(peek.error());
  auto cfg = GrokConfig::from_json(peek->config_json);
  if (!cfg) return std::unexpected(cfg.error());
  Rng rng(1);
  auto m = create(*cfg, d, rng);
  if (!m) return m;
  auto f = GyreFile::open(path);
  if (!f) return std::unexpected(f.error());
  auto ld = f->load_params(m->params_, nullptr);
  if (!ld) return std::unexpected(ld.error());
  return m;
}

Result<void> import_safetensors_to_gyre(const std::filesystem::path& dir, const std::filesystem::path& out,
                                       std::shared_ptr<Device> d, GyreCodec codec) {
  auto c = GrokConfig::from_file(dir / "config.json");
  if (!c) return std::unexpected(c.error());
  Rng rng(1);
  auto m = GrokLM::create(*c, d, rng);
  if (!m) return std::unexpected(m.error());
  auto names = m->param_names();
  std::unordered_map<std::string, int> need;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) need[names[static_cast<std::size_t>(i)]] = i;
  for (auto& ent : std::filesystem::directory_iterator(dir)) {
    if (!ent.is_regular_file() || ent.path().extension() != ".safetensors") continue;
    auto st = safetensors_open(ent.path());
    if (!st) return std::unexpected(st.error());
    for (auto& info : st->tensors) {
      auto it = need.find(info.name);
      if (it == need.end()) continue;
      auto t = safetensors_load(*st, info.name, d, false);
      if (!t) return std::unexpected(t.error());
      if (t->dtype() != DType::f32) {
        auto f = t->to_f32();
        if (!f) return std::unexpected(f.error());
        t = std::move(f);
      }
      auto p = make_param(std::move(*t));
      if (!p) return std::unexpected(p.error());
      m->parameters()[static_cast<std::size_t>(it->second)] = std::move(*p);
      need.erase(it);
    }
  }
  return m->save_gyre(out, codec);
}

Result<GrokLora> GrokLora::create(const GrokConfig& c, std::int64_t rank, float alpha,
                                 std::shared_ptr<Device> d, Rng& rng) {
  if (rank < 0) return std::unexpected(make_error(Errc::invalid_shape, "lora rank"));
  GrokLora l;
  l.rank = rank;
  l.alpha = alpha;
  if (rank == 0) return l;
  const auto qin = c.d_model, qout = c.n_q_head * c.head_dim;
  const auto kin = c.d_model, kout = c.n_kv_head * c.head_dim;
  auto make_pair = [&](std::int64_t in, std::int64_t out) -> Result<GrokLoraPair> {
    auto A = init_linear(in, rank, d, rng, 0.02f);
    if (!A) return std::unexpected(A.error());
    std::int64_t bsh[2] = {rank, out};
    auto B = Tensor::zeros(bsh, DType::f32, d);
    if (!B) return std::unexpected(B.error());
    GrokLoraPair p{std::move(A->value), std::move(*B)};
    return p;
  };
  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    auto q = make_pair(qin, qout);
    auto k = make_pair(kin, kout);
    auto v = make_pair(kin, kout);
    auto o = make_pair(qout, qin);
    if (!q || !k || !v || !o) return std::unexpected(q ? (k ? (v ? o.error() : v.error()) : k.error()) : q.error());
    l.q.push_back(std::move(*q));
    l.k.push_back(std::move(*k));
    l.v.push_back(std::move(*v));
    l.o.push_back(std::move(*o));
  }
  return l;
}

Result<void> GrokLM::set_lora(GrokLora lora) {
  if (lora.rank > 0 && static_cast<std::int64_t>(lora.q.size()) != cfg_.n_layer) {
    return std::unexpected(make_error(Errc::invalid_shape, "lora layers"));
  }
  lora_ = std::move(lora);
  return {};
}

Result<void> GrokLM::save_lora(const std::filesystem::path& dir) const {
  if (!lora_) return std::unexpected(make_error(Errc::unsupported, "no lora"));
  std::filesystem::create_directories(dir);
  std::ofstream meta(dir / "lora.json", std::ios::binary);
  if (!meta) return std::unexpected(make_error(Errc::io, "lora.json"));
  nlohmann::json js = {{"gyre", "lora"}, {"base_arch", "grok2"}, {"rank", lora_->rank}, {"alpha", lora_->alpha}};
  auto s = js.dump();
  meta.write(s.data(), static_cast<std::streamsize>(s.size()));
  std::vector<NamedTensor> nt;
  auto add = [&](const char* kind, const std::vector<GrokLoraPair>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto p = std::string("layers.") + std::to_string(i) + "." + kind;
      nt.push_back({p + ".A", &v[i].A});
      nt.push_back({p + ".B", &v[i].B});
    }
  };
  add("q_proj", lora_->q);
  add("k_proj", lora_->k);
  add("v_proj", lora_->v);
  add("o_proj", lora_->o);
  if (nt.empty()) return {};
  return safetensors_save(dir / "lora.safetensors", nt);
}

Result<void> GrokLM::load_lora(const std::filesystem::path& dir, std::shared_ptr<Device> d) {
  auto js = [&]() -> std::string {
    std::ifstream in(dir / "lora.json", std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }();
  std::int64_t rank = 4;
  float alpha = 4.f;
  if (!js.empty()) {
    if (auto parsed = parse_json(js); parsed) {
      rank = (*parsed).value("rank", rank);
      alpha = (*parsed).value("alpha", alpha);
    }
  }
  Rng rng(1);
  auto l = GrokLora::create(cfg_, rank, alpha, d, rng);
  if (!l) return std::unexpected(l.error());
  if (rank == 0) {
    lora_ = std::move(*l);
    return {};
  }
  auto st = safetensors_open(dir / "lora.safetensors");
  if (!st) return std::unexpected(st.error());
  auto loadp = [&](GrokLoraPair& p, const std::string& a, const std::string& b) -> Result<void> {
    auto A = safetensors_load(*st, a, d, false);
    auto B = safetensors_load(*st, b, d, false);
    if (!A) return std::unexpected(A.error());
    if (!B) return std::unexpected(B.error());
    p.A = std::move(*A);
    p.B = std::move(*B);
    return {};
  };
  for (std::int64_t i = 0; i < cfg_.n_layer; ++i) {
    auto s = std::string("layers.") + std::to_string(i);
    auto e = loadp(l->q[static_cast<std::size_t>(i)], s + ".q_proj.A", s + ".q_proj.B");
    if (!e) return e;
    e = loadp(l->k[static_cast<std::size_t>(i)], s + ".k_proj.A", s + ".k_proj.B");
    if (!e) return e;
    e = loadp(l->v[static_cast<std::size_t>(i)], s + ".v_proj.A", s + ".v_proj.B");
    if (!e) return e;
    e = loadp(l->o[static_cast<std::size_t>(i)], s + ".o_proj.A", s + ".o_proj.B");
    if (!e) return e;
  }
  lora_ = std::move(*l);
  return {};
}

}  // namespace gyre

