#include "gyre/nn/transformer.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace gyre {
namespace {

void take(std::vector<Param>& flat, std::span<Param> s) {
  for (auto& p : s) flat.push_back(Param{p.value, p.grad});
}

}  // namespace

DecoderBlock::DecoderBlock(LayerNorm ln1, CausalSelfAttention attn, LayerNorm ln2, Linear fc1, Linear fc2)
    : ln1_(std::move(ln1)),
      attn_(std::move(attn)),
      ln2_(std::move(ln2)),
      fc1_(std::move(fc1)),
      fc2_(std::move(fc2)) {
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
  auto attn = CausalSelfAttention::create(c.d_model, c.n_head, d, rng, resid_scale, c.recency_alibi);
  auto ln2 = LayerNorm::create(c.d_model, d);
  auto fc1 = Linear::create(c.d_model, c.d_ff, d, rng, 0.02f, 1.f);
  auto fc2 = Linear::create(c.d_ff, c.d_model, d, rng, 0.02f, resid_scale);
  if (!ln1 || !attn || !ln2 || !fc1 || !fc2) {
    return std::unexpected(ln1 ? (attn ? (ln2 ? (fc1 ? fc2.error() : fc1.error()) : ln2.error()) : attn.error())
                               : ln1.error());
  }
  return DecoderBlock(std::move(*ln1), std::move(*attn), std::move(*ln2), std::move(*fc1), std::move(*fc2));
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
  return add(*h1, *fc2);
}

Result<void> DecoderBlock::backward(const Tensor& dh2, ForwardCtx& ctx) {
  if (!saved_x_ || !saved_h1_ || !saved_fc1_) {
    return std::unexpected(make_error(Errc::unsupported, "block tape"));
  }
  auto r2 = fc2_.backward(dh2, ctx);
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

Result<Tensor> CharLM::forward(const Tensor& idx, ForwardCtx& ctx) {
  if (idx.rank() != 2) return std::unexpected(make_error(Errc::invalid_shape, "idx [B,T]"));
  const auto B = idx.shape()[0], T = idx.shape()[1];
  if (T > cfg_.block_size) return std::unexpected(make_error(Errc::invalid_shape, "T > block"));
  if (ctx.train) saved_idx_ = idx;
  auto tok = wte_.forward(idx, ctx);
  if (!tok) return tok;
  std::int64_t psh[1] = {T};
  auto pos = Tensor::empty(psh, DType::i32, idx.device());
  if (!pos) return pos;
  auto pp = pos->host_span<std::int32_t>();
  if (!pp) return std::unexpected(pp.error());
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(T); ++i) (*pp)[static_cast<std::size_t>(i)] = i;
  auto pe = wpe_.forward(*pos, ctx);
  if (!pe) return pe;
  auto x = Tensor::empty(tok->shape(), DType::f32, idx.device());
  if (!x) return x;
  auto xp = x->host_span<float>();
  auto tp = tok->host_span<float>();
  auto pep = pe->host_span<float>();
  if (!xp || !tp || !pep) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = cfg_.d_model;
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t c = 0; c < C; ++c)
        (*xp)[static_cast<std::size_t>((b * T + t) * C + c)] =
            (*tp)[static_cast<std::size_t>((b * T + t) * C + c)] +
            (*pep)[static_cast<std::size_t>(t * C + c)];

  Tensor h = std::move(*x);
  for (auto& block : blocks_) {
    auto y = block.forward(h, ctx);
    if (!y) return y;
    h = std::move(*y);
  }
  auto lnf = ln_f_.forward(h, ctx);
  if (!lnf) return lnf;
  return lm_head_.forward(*lnf, ctx);
}

Result<void> CharLM::backward(const Tensor& d_out, ForwardCtx& ctx) {
  auto rh = lm_head_.backward(d_out, ctx);
  if (!rh || !ctx.dx) return rh ? std::unexpected(make_error(Errc::unsupported, "head dx")) : rh;
  Tensor dlnf = std::move(*ctx.dx);
  auto rl = ln_f_.backward(dlnf, ctx);
  if (!rl || !ctx.dx) return rl ? std::unexpected(make_error(Errc::unsupported, "lnf dx")) : rl;
  Tensor dh = std::move(*ctx.dx);
  for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
    auto rb = it->backward(dh, ctx);
    if (!rb || !ctx.dx) return rb ? std::unexpected(make_error(Errc::unsupported, "block dx")) : rb;
    dh = std::move(*ctx.dx);
  }
  // dh is d(tok + pe). tok is [B,T,C], pe is [T,C] broadcast.
  auto rtok = wte_.backward(dh, ctx);
  if (!rtok) return rtok;
  const auto B = dh.shape()[0], T = dh.shape()[1], C = dh.shape()[2];
  std::int64_t psh[2] = {T, C};
  auto dpe = Tensor::zeros(psh, DType::f32, dh.device());
  if (!dpe) return std::unexpected(dpe.error());
  auto a = dh.host_span<float>();
  auto b = dpe->host_span<float>();
  if (!a || !b) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (std::int64_t bi = 0; bi < B; ++bi)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t c = 0; c < C; ++c)
        (*b)[static_cast<std::size_t>(t * C + c)] +=
            (*a)[static_cast<std::size_t>((bi * T + t) * C + c)];
  return wpe_.backward(*dpe, ctx);
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
    auto p = logits->host_span<float>();
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

}  // namespace gyre
