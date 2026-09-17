#include "gyre/nn/transformer.hpp"

#include "gyre/checkpoint.hpp"
#include "json_parse.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace gyre {
namespace {

void take(std::vector<Param>& flat, std::span<Param> s) {
  for (auto& p : s) flat.push_back(Param{p.value, p.grad});
}

}  // namespace

DecoderBlock::DecoderBlock(LayerNorm ln1, CausalSelfAttention attn, LayerNorm ln2, Linear fc1, Linear fc2,
                           float dropout)
    : ln1_(std::move(ln1)),
      attn_(std::move(attn)),
      ln2_(std::move(ln2)),
      fc1_(std::move(fc1)),
      fc2_(std::move(fc2)),
      dropout_(dropout) {
  rebind();
}

void DecoderBlock::rebind() {
  flat_.clear();
  take(flat_, ln1_.parameters());
  take(flat_, attn_.parameters());
  take(flat_, ln2_.parameters());
  take(flat_, fc1_.parameters());
  take(flat_, fc2_.parameters());
}

Result<DecoderBlock> DecoderBlock::create(const CharLMConfig& c, std::shared_ptr<Device> d, Rng& rng,
                                         float resid_scale) {
  auto ln1 = LayerNorm::create(c.d_model, d);
  auto attn = CausalSelfAttention::create(c.d_model, c.n_head, d, rng, resid_scale, c.recency_alibi,
                                          c.dropout);
  auto ln2 = LayerNorm::create(c.d_model, d);
  auto fc1 = Linear::create(c.d_model, c.d_ff, d, rng, 0.02f, 1.f);
  auto fc2 = Linear::create(c.d_ff, c.d_model, d, rng, 0.02f, resid_scale);
  if (!ln1 || !attn || !ln2 || !fc1 || !fc2) {
    return std::unexpected(ln1 ? (attn ? (ln2 ? (fc1 ? fc2.error() : fc1.error()) : ln2.error()) : attn.error())
                               : ln1.error());
  }
  return DecoderBlock(std::move(*ln1), std::move(*attn), std::move(*ln2), std::move(*fc1), std::move(*fc2),
                      c.dropout);
}

Result<void> DecoderBlock::set_lora(const CharLoraLayer& layer, float scale) {
  auto ra = attn_.set_lora(layer.q, layer.k, layer.v, layer.o, scale);
  if (!ra) return ra;
  auto r1 = fc1_.set_lora(layer.fc1.A, layer.fc1.B, scale);
  auto r2 = fc2_.set_lora(layer.fc2.A, layer.fc2.B, scale);
  if (!r1 || !r2) return std::unexpected(r1 ? r2.error() : r1.error());
  return {};
}

void DecoderBlock::clear_lora() {
  attn_.clear_lora();
  fc1_.clear_lora();
  fc2_.clear_lora();
}

void DecoderBlock::set_freeze_base(bool freeze) {
  attn_.set_freeze_base(freeze);
  fc1_.set_freeze_base(freeze);
  fc2_.set_freeze_base(freeze);
}

std::vector<Param> DecoderBlock::lora_parameters() {
  std::vector<Param> out = attn_.lora_parameters();
  auto take = [&](std::span<Param> s) {
    for (auto& p : s) out.push_back(Param{p.value, p.grad});
  };
  take(fc1_.lora_parameters());
  take(fc2_.lora_parameters());
  return out;
}

namespace {

Result<LoraPair> make_lora_pair(std::int64_t in, std::int64_t out, std::int64_t rank,
                               std::shared_ptr<Device> d, Rng& rng) {
  std::int64_t ash[2] = {in, rank};
  std::int64_t bsh[2] = {rank, out};
  auto A = Tensor::empty(ash, DType::f32, d);
  auto B = Tensor::zeros(bsh, DType::f32, d);
  if (!A || !B) return std::unexpected(A ? B.error() : A.error());
  auto ap = A->host_span<float>();
  if (!ap) {
    auto cpu = Device::cpu();
    if (!cpu) return std::unexpected(cpu.error());
    auto host = A->to(*cpu);
    if (!host) return std::unexpected(host.error());
    auto hp = host->host_span<float>();
    if (!hp) return std::unexpected(hp.error());
    for (auto& v : *hp) v = rng.normal(0.f, 0.02f);
    auto back = host->to(d);
    if (!back) return std::unexpected(back.error());
    A = std::move(*back);
  } else {
    for (auto& v : *ap) v = rng.normal(0.f, 0.02f);
  }
  return LoraPair{std::move(*A), std::move(*B)};
}

}  // namespace

Result<CharLora> CharLora::create(const CharLMConfig& c, std::int64_t rank, float alpha,
                                 std::shared_ptr<Device> d, Rng& rng) {
  if (rank < 0) return std::unexpected(make_error(Errc::invalid_shape, "lora rank"));
  CharLora l;
  l.rank = rank;
  l.alpha = alpha;
  if (rank == 0) return l;
  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    auto q = make_lora_pair(c.d_model, c.d_model, rank, d, rng);
    auto k = make_lora_pair(c.d_model, c.d_model, rank, d, rng);
    auto v = make_lora_pair(c.d_model, c.d_model, rank, d, rng);
    auto o = make_lora_pair(c.d_model, c.d_model, rank, d, rng);
    auto f1 = make_lora_pair(c.d_model, c.d_ff, rank, d, rng);
    auto f2 = make_lora_pair(c.d_ff, c.d_model, rank, d, rng);
    if (!q || !k || !v || !o || !f1 || !f2) {
      return std::unexpected(q ? (k ? (v ? (o ? (f1 ? f2.error() : f1.error()) : o.error()) : v.error())
                                    : k.error())
                               : q.error());
    }
    l.layers.push_back(CharLoraLayer{std::move(*q), std::move(*k), std::move(*v), std::move(*o),
                                     std::move(*f1), std::move(*f2)});
  }
  return l;
}

Result<Tensor> DecoderBlock::forward(const Tensor& x, ForwardCtx& ctx) {
  if (ctx.train) saved_x_ = x;
  auto n1 = ln1_.forward(x, ctx);
  if (!n1) return n1;
  auto att = attn_.forward(*n1, ctx);
  if (!att) return att;
  auto h1 = add(x, *att);
  if (!h1) return h1;
  if (ctx.train) saved_h1_ = *h1;
  auto n2 = ln2_.forward(*h1, ctx);
  if (!n2) return n2;
  auto fc1 = fc1_.forward(*n2, ctx);
  if (!fc1) return fc1;
  if (ctx.train) saved_fc1_ = *fc1;
  auto g = gelu(*fc1);
  if (!g) return g;
  auto fc2 = fc2_.forward(*g, ctx);
  if (!fc2) return fc2;
  auto dropped = dropout(*fc2, dropout_, ctx.train, ctx.rng, &saved_drop_mlp_);
  if (!dropped) return dropped;
  return add(*h1, *dropped);
}

Result<void> DecoderBlock::backward(const Tensor& dh2, ForwardCtx& ctx) {
  if (!saved_x_ || !saved_h1_ || !saved_fc1_) {
    return std::unexpected(make_error(Errc::unsupported, "block tape"));
  }
  Tensor dmlp = dh2;
  if (saved_drop_mlp_) {
    auto dd = dropout_backward(dh2, *saved_drop_mlp_);
    if (!dd) return std::unexpected(dd.error());
    dmlp = std::move(*dd);
  }
  auto r2 = fc2_.backward(dmlp, ctx);
  if (!r2 || !ctx.dx) return r2 ? std::unexpected(make_error(Errc::unsupported, "fc2 dx")) : r2;
  Tensor dgelu_out = std::move(*ctx.dx);
  auto dge = gelu_backward(*saved_fc1_, dgelu_out);
  if (!dge) return std::unexpected(dge.error());
  auto r1 = fc1_.backward(*dge, ctx);
  if (!r1 || !ctx.dx) return r1 ? std::unexpected(make_error(Errc::unsupported, "fc1 dx")) : r1;
  Tensor dfc1 = std::move(*ctx.dx);
  auto rl2 = ln2_.backward(dfc1, ctx);
  if (!rl2 || !ctx.dx) return rl2 ? std::unexpected(make_error(Errc::unsupported, "ln2 dx")) : rl2;
  Tensor dln2 = std::move(*ctx.dx);
  auto dh1 = add(dh2, dln2);
  if (!dh1) return std::unexpected(dh1.error());
  auto ra = attn_.backward(*dh1, ctx);
  if (!ra || !ctx.dx) return ra ? std::unexpected(make_error(Errc::unsupported, "attn dx")) : ra;
  Tensor datt = std::move(*ctx.dx);
  auto rl1 = ln1_.backward(datt, ctx);
  if (!rl1 || !ctx.dx) return rl1 ? std::unexpected(make_error(Errc::unsupported, "ln1 dx")) : rl1;
  Tensor dln1 = std::move(*ctx.dx);
  auto dx = add(*dh1, dln1);
  if (!dx) return std::unexpected(dx.error());
  ctx.dx = std::make_unique<Tensor>(std::move(*dx));
  return {};
}

CharLM::CharLM(CharLMConfig c, Embedding wte, Embedding wpe, std::vector<DecoderBlock> blocks,
               LayerNorm ln_f, Linear lm_head)
    : cfg_(c),
      wte_(std::move(wte)),
      wpe_(std::move(wpe)),
      blocks_(std::move(blocks)),
      ln_f_(std::move(ln_f)),
      lm_head_(std::move(lm_head)) {
  rebind();
}

void CharLM::rebind() {
  params_.clear();
  take(params_, wte_.parameters());
  take(params_, wpe_.parameters());
  for (auto& b : blocks_) take(params_, b.parameters());
  take(params_, ln_f_.parameters());
  take(params_, lm_head_.parameters());
}

Result<CharLM> CharLM::create(CharLMConfig c, std::shared_ptr<Device> d, Rng& rng) {
  if (c.d_model % c.n_head != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "d_model % n_head"));
  }
  const float resid = 1.f / std::sqrt(static_cast<float>(2 * c.n_layer));
  auto wte = Embedding::create(c.vocab, c.d_model, d, rng);
  auto wpe = Embedding::create(c.block_size, c.d_model, d, rng);
  if (!wte || !wpe) return std::unexpected(wte ? wpe.error() : wte.error());
  std::vector<DecoderBlock> blocks;
  blocks.reserve(static_cast<std::size_t>(c.n_layer));
  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    auto b = DecoderBlock::create(c, d, rng, resid);
    if (!b) return std::unexpected(b.error());
    blocks.push_back(std::move(*b));
  }
  auto lnf = LayerNorm::create(c.d_model, d);
  auto head = Linear::create(c.d_model, c.vocab, d, rng, 0.02f, 1.f);
  if (!lnf || !head) return std::unexpected(lnf ? head.error() : lnf.error());
  return CharLM(c, std::move(*wte), std::move(*wpe), std::move(blocks), std::move(*lnf), std::move(*head));
}

Result<Tensor> CharLM::hidden(const Tensor& idx, ForwardCtx& ctx) {
  if (idx.rank() != 2) return std::unexpected(make_error(Errc::invalid_shape, "idx [B,T]"));
  const auto T = idx.shape()[1];
  if (T > cfg_.block_size) return std::unexpected(make_error(Errc::invalid_shape, "T > block"));
  if (ctx.train) saved_idx_ = idx;
  auto tok = wte_.forward(idx, ctx);
  if (!tok) return tok;
  auto pos = arange_i32(T, idx.device());
  if (!pos) return pos;
  auto pe = wpe_.forward(*pos, ctx);
  if (!pe) return pe;
  auto x = add_broadcast_time(*tok, *pe);
  if (!x) return x;
  auto xd = dropout(*x, cfg_.dropout, ctx.train, ctx.rng, &saved_drop_emb_);
  if (!xd) return xd;

  Tensor h = std::move(*xd);
  for (auto& block : blocks_) {
    auto y = block.forward(h, ctx);
    if (!y) return y;
    h = std::move(*y);
  }
  return ln_f_.forward(h, ctx);
}

Result<Tensor> CharLM::forward(const Tensor& idx, ForwardCtx& ctx) {
  auto h = hidden(idx, ctx);
  if (!h) return h;
  return lm_head_.forward(*h, ctx);
}

Result<void> CharLM::hidden_backward(const Tensor& d_hidden, ForwardCtx& ctx) {
  auto rl = ln_f_.backward(d_hidden, ctx);
  if (!rl || !ctx.dx) return rl ? std::unexpected(make_error(Errc::unsupported, "lnf dx")) : rl;
  Tensor dh = std::move(*ctx.dx);
  for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
    auto rb = it->backward(dh, ctx);
    if (!rb || !ctx.dx) return rb ? std::unexpected(make_error(Errc::unsupported, "block dx")) : rb;
    dh = std::move(*ctx.dx);
  }
  // dh is d(tok + pe), after embedding dropout.
  if (saved_drop_emb_) {
    auto dd = dropout_backward(dh, *saved_drop_emb_);
    if (!dd) return std::unexpected(dd.error());
    dh = std::move(*dd);
  }
  auto rtok = wte_.backward(dh, ctx);
  if (!rtok) return rtok;
  auto dpe = sum_batch_to_time(dh);
  if (!dpe) return std::unexpected(dpe.error());
  return wpe_.backward(*dpe, ctx);
}

Result<void> CharLM::backward(const Tensor& d_out, ForwardCtx& ctx) {
  auto rh = lm_head_.backward(d_out, ctx);
  if (!rh || !ctx.dx) return rh ? std::unexpected(make_error(Errc::unsupported, "head dx")) : rh;
  Tensor dlnf = std::move(*ctx.dx);
  return hidden_backward(dlnf, ctx);
}

std::vector<std::string> CharLM::param_names() const {
  std::vector<std::string> n;
  n.push_back("wte.weight");
  n.push_back("wpe.weight");
  for (std::int64_t i = 0; i < cfg_.n_layer; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    n.push_back(p + "ln1.weight");
    n.push_back(p + "ln1.bias");
    n.push_back(p + "attn.q.weight");
    n.push_back(p + "attn.q.bias");
    n.push_back(p + "attn.k.weight");
    n.push_back(p + "attn.k.bias");
    n.push_back(p + "attn.v.weight");
    n.push_back(p + "attn.v.bias");
    n.push_back(p + "attn.o.weight");
    n.push_back(p + "attn.o.bias");
    n.push_back(p + "ln2.weight");
    n.push_back(p + "ln2.bias");
    n.push_back(p + "mlp.fc1.weight");
    n.push_back(p + "mlp.fc1.bias");
    n.push_back(p + "mlp.fc2.weight");
    n.push_back(p + "mlp.fc2.bias");
  }
  n.push_back("ln_f.weight");
  n.push_back("ln_f.bias");
  n.push_back("lm_head.weight");
  n.push_back("lm_head.bias");
  return n;
}

Result<std::vector<std::int32_t>> CharLM::generate(std::vector<std::int32_t> prefix, int max_new,
                                                   std::shared_ptr<Device> d, Rng* rng,
                                                   float temperature) {
  if (temperature > 0.f && !rng) {
    return std::unexpected(make_error(Errc::unsupported, "temperature > 0 requires Rng"));
  }
  if (prefix.empty()) {
    return std::unexpected(make_error(Errc::invalid_shape, "empty prompt"));
  }
  for (int n = 0; n < max_new; ++n) {
    const auto ctx_n = std::min(static_cast<std::int64_t>(prefix.size()), cfg_.block_size);
    std::int64_t sh[2] = {1, ctx_n};
    std::vector<std::byte> bytes(static_cast<std::size_t>(ctx_n) * 4);
    std::memcpy(bytes.data(), prefix.data() + prefix.size() - static_cast<std::size_t>(ctx_n),
                static_cast<std::size_t>(ctx_n) * 4);
    auto idx = Tensor::from_host(bytes, sh, DType::i32, d);
    if (!idx) return std::unexpected(idx.error());
    ForwardCtx ctx;
    ctx.train = false;
    auto logits = forward(*idx, ctx);
    if (!logits) return std::unexpected(logits.error());
    Tensor logits_h = std::move(*logits);
    if (logits_h.device() && logits_h.device()->kind() != DeviceKind::cpu) {
      auto cpu = Device::cpu();
      if (!cpu) return std::unexpected(cpu.error());
      auto c = logits_h.to(*cpu);
      if (!c) return std::unexpected(c.error());
      logits_h = std::move(*c);
    }
    auto p = logits_h.host_span<float>();
    if (!p) return std::unexpected(p.error());
    const auto V = cfg_.vocab;
    const float* last = p->data() + (ctx_n - 1) * V;
    auto next = sample_logit_row(std::span<const float>(last, static_cast<std::size_t>(V)), temperature,
                                 rng);
    if (!next) return std::unexpected(next.error());
    prefix.push_back(*next);
  }
  return prefix;
}

void CharLM::clear_lora() {
  lora_.reset();
  lora_flat_.clear();
  for (auto& b : blocks_) b.clear_lora();
}

void CharLM::set_freeze_base(bool freeze) {
  for (auto& b : blocks_) b.set_freeze_base(freeze);
}

Result<void> CharLM::set_lora(CharLora lora) {
  if (lora.rank > 0 && static_cast<std::int64_t>(lora.layers.size()) != cfg_.n_layer) {
    return std::unexpected(make_error(Errc::invalid_shape, "lora layers"));
  }
  clear_lora();
  if (lora.rank <= 0) {
    lora_ = std::move(lora);
    return {};
  }
  const float scale = lora.scale();
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    auto r = blocks_[i].set_lora(lora.layers[i], scale);
    if (!r) return r;
  }
  lora_ = std::move(lora);
  lora_flat_.clear();
  for (auto& b : blocks_) {
    auto part = b.lora_parameters();
    for (auto& p : part) lora_flat_.push_back(std::move(p));
  }
  return {};
}

Result<void> CharLM::zero_grad() {
  auto r = Module::zero_grad();
  if (!r) return r;
  for (auto& p : lora_flat_) {
    auto z = fill_zero(p.grad);
    if (!z) return z;
  }
  return {};
}

Result<void> CharLM::save_lora(const std::filesystem::path& path) const {
  if (!lora_ || lora_->rank <= 0) return std::unexpected(make_error(Errc::unsupported, "no lora"));
  std::vector<std::string> names;
  names.reserve(lora_flat_.size());
  for (std::int64_t i = 0; i < cfg_.n_layer; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    names.push_back(p + "attn.q.A");
    names.push_back(p + "attn.q.B");
    names.push_back(p + "attn.k.A");
    names.push_back(p + "attn.k.B");
    names.push_back(p + "attn.v.A");
    names.push_back(p + "attn.v.B");
    names.push_back(p + "attn.o.A");
    names.push_back(p + "attn.o.B");
    names.push_back(p + "mlp.fc1.A");
    names.push_back(p + "mlp.fc1.B");
    names.push_back(p + "mlp.fc2.A");
    names.push_back(p + "mlp.fc2.B");
  }
  if (names.size() != lora_flat_.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "lora name count"));
  }
  CheckpointMeta meta;
  meta.json = std::string("{\"arch\":\"char-lm-lora\",\"config\":{\"rank\":")
              + std::to_string(lora_->rank) + ",\"alpha\":" + std::to_string(lora_->alpha)
              + ",\"d_model\":" + std::to_string(cfg_.d_model) + ",\"n_layer\":"
              + std::to_string(cfg_.n_layer) + ",\"d_ff\":" + std::to_string(cfg_.d_ff) + "}}";
  meta.param_names = names;
  return save_gyre1(path, std::span<const Param>(lora_flat_.data(), lora_flat_.size()), nullptr, meta);
}

Result<void> CharLM::load_lora(const std::filesystem::path& path, std::shared_ptr<Device> d) {
  CheckpointMeta meta;
  auto peek = peek_gyre1(path);
  std::int64_t rank = 8;
  float alpha = 16.f;
  if (peek) {
    auto parsed = parse_json(peek->json);
    if (parsed && (*parsed).contains("config") && (*parsed)["config"].is_object()) {
      const auto& cfg = (*parsed)["config"];
      rank = cfg.value("rank", rank);
      alpha = cfg.value("alpha", alpha);
      if (cfg.contains("d_model") && cfg["d_model"].get<std::int64_t>() != cfg_.d_model) {
        return std::unexpected(make_error(Errc::invalid_shape, "lora d_model"));
      }
      if (cfg.contains("n_layer") && cfg["n_layer"].get<std::int64_t>() != cfg_.n_layer) {
        return std::unexpected(make_error(Errc::invalid_shape, "lora n_layer"));
      }
      if (cfg.contains("d_ff") && cfg["d_ff"].get<std::int64_t>() != cfg_.d_ff) {
        return std::unexpected(make_error(Errc::invalid_shape, "lora d_ff"));
      }
    }
  }
  Rng rng(1);
  auto created = CharLora::create(cfg_, rank, alpha, d, rng);
  if (!created) return std::unexpected(created.error());
  auto attached = set_lora(std::move(*created));
  if (!attached) return attached;
  if (rank == 0) return {};
  auto lr = load_gyre1(path, lora_flat_, nullptr, meta);
  if (!lr) return lr;
  return {};
}

}  // namespace gyre
