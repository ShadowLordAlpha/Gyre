#include "gyre/rl/policy.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/ops.hpp"

#include <cmath>

namespace gyre {
namespace {
Error err(const char* m) { return make_error(Errc::invalid_shape, m); }

// u8 [B] "keep" row for step t from a reset tensor u8 [T,B] (keep = !reset).
Result<Tensor> keep_row(const Tensor& reset, std::int64_t t) {
  const auto B = reset.shape()[1];
  auto pr = reset.host_span<std::uint8_t>();
  if (!pr) return std::unexpected(pr.error());
  std::int64_t sh[1] = {B};
  auto k = Tensor::empty(sh, DType::u8, reset.device());
  if (!k) return k;
  auto pk = k->host_span<std::uint8_t>();
  for (std::int64_t b = 0; b < B; ++b) (*pk)[static_cast<std::size_t>(b)] = (*pr)[static_cast<std::size_t>(t * B + b)] ? 0 : 1;
  return k;
}
}  // namespace

PolicyNet::PolicyNet(PolicyConfig c, HeadLayout l, SetEncoder enc, std::optional<Gru> gru, Linear heads,
                     Linear value)
    : cfg_(std::move(c)),
      layout_(std::move(l)),
      enc_(std::move(enc)),
      gru_(std::move(gru)),
      heads_(std::move(heads)),
      value_(std::move(value)) {
  for (auto& p : enc_.parameters()) flat_.push_back(Param{p.value, p.grad});
  if (gru_)
    for (auto& p : gru_->parameters()) flat_.push_back(Param{p.value, p.grad});
  for (auto& p : heads_.parameters()) flat_.push_back(Param{p.value, p.grad});
  for (auto& p : value_.parameters()) flat_.push_back(Param{p.value, p.grad});
}

Result<PolicyNet> PolicyNet::create(const PolicyConfig& c, std::shared_ptr<Device> d, Rng& rng) {
  auto layout = HeadLayout::make(c.head_sizes);
  if (!layout) return std::unexpected(layout.error());
  if (c.gru_hidden < 0) return std::unexpected(err("gru_hidden"));
  auto enc = SetEncoder::create(c.encoder, d, rng);
  if (!enc) return std::unexpected(enc.error());
  std::optional<Gru> gru;
  if (c.gru_hidden > 0) {
    auto g = Gru::create(c.encoder.out_dim, c.gru_hidden, d, rng);
    if (!g) return std::unexpected(g.error());
    gru = std::move(*g);
  }
  const auto head_in = c.gru_hidden > 0 ? c.gru_hidden : c.encoder.out_dim;
  // Small policy-head init keeps the initial policy near uniform; value head is O(1/sqrt(in)).
  auto heads = Linear::create(head_in, layout->total(), d, rng, 0.01f);
  auto value = Linear::create(head_in, 1, d, rng, 1.f / std::sqrt(float(head_in)));
  if (!heads || !value) return std::unexpected(heads ? value.error() : heads.error());
  return PolicyNet(c, std::move(*layout), std::move(*enc), std::move(gru), std::move(*heads),
                   std::move(*value));
}

Result<void> PolicyNet::zero_grad() {
  for (auto& p : flat_)
    if (auto r = fill_zero(p.grad); !r) return r;
  return {};
}

Result<PolicyOut> PolicyNet::heads_forward(const Tensor& feat, ForwardCtx& ctx) {
  auto logits = heads_.forward(feat, ctx);
  auto v = value_.forward(feat, ctx);
  if (!logits || !v) return std::unexpected(logits ? v.error() : logits.error());
  std::int64_t sh[1] = {v->shape()[0]};
  auto v1 = reshape(*v, sh);
  if (!v1) return std::unexpected(v1.error());
  return PolicyOut{std::move(*logits), std::move(*v1)};
}

Result<Tensor> PolicyNet::heads_backward(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx) {
  if (auto r = heads_.backward(dlogits, ctx); !r) return std::unexpected(r.error());
  if (!ctx.dx) return std::unexpected(make_error(Errc::unsupported, "heads dx"));
  Tensor df = std::move(*ctx.dx);
  ctx.dx.reset();
  std::int64_t sh[2] = {dvalue.shape()[0], 1};
  auto dv2 = reshape(dvalue, sh);
  if (!dv2) return std::unexpected(dv2.error());
  if (auto r = value_.backward(*dv2, ctx); !r) return std::unexpected(r.error());
  if (!ctx.dx) return std::unexpected(make_error(Errc::unsupported, "value dx"));
  auto dsum = add(df, *ctx.dx);
  ctx.dx.reset();
  if (!dsum) return std::unexpected(dsum.error());
  return std::move(*dsum);
}

// ------------------------------------------------------------- feedforward

Result<PolicyOut> PolicyNet::forward(const Obs& obs, ForwardCtx& ctx) {
  if (recurrent()) return std::unexpected(make_error(Errc::unsupported, "recurrent net: use forward_step/forward_seq"));
  auto feat = enc_.forward(obs.entities, obs.mask, obs.global, ctx);
  if (!feat) return std::unexpected(feat.error());
  return heads_forward(*feat, ctx);
}

Result<void> PolicyNet::backward(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx) {
  auto df = heads_backward(dlogits, dvalue, ctx);
  if (!df) return std::unexpected(df.error());
  auto g = enc_.backward(*df, ctx);
  if (!g) return std::unexpected(g.error());
  return {};
}

// --------------------------------------------------------------- recurrent

Result<StepOut> PolicyNet::forward_step(const Obs& obs, const Tensor& h_in, const Tensor& reset) {
  if (!recurrent()) return std::unexpected(make_error(Errc::unsupported, "net is feedforward"));
  const auto B = obs.entities.shape()[0];
  if (reset.dtype() != DType::u8 || reset.rank() != 1 || reset.shape()[0] != B || h_in.rank() != 2 ||
      h_in.shape()[0] != B || h_in.shape()[1] != hidden_dim())
    return std::unexpected(err("forward_step: h_in [B,H], reset u8 [B]"));
  ForwardCtx ctx;
  ctx.train = false;
  auto feat = enc_.forward(obs.entities, obs.mask, obs.global, ctx);
  if (!feat) return std::unexpected(feat.error());
  // keep = !reset
  std::int64_t sh[1] = {B};
  auto keep = Tensor::empty(sh, DType::u8, reset.device());
  if (!keep) return std::unexpected(keep.error());
  auto pr = reset.host_span<std::uint8_t>();
  auto pk = keep->host_span<std::uint8_t>();
  for (std::int64_t b = 0; b < B; ++b) (*pk)[static_cast<std::size_t>(b)] = (*pr)[static_cast<std::size_t>(b)] ? 0 : 1;
  auto hm = mask_rows(h_in, *keep);
  if (!hm) return std::unexpected(hm.error());
  auto h = gru_->step(*feat, *hm, nullptr);
  if (!h) return std::unexpected(h.error());
  auto out = heads_forward(*h, ctx);
  if (!out) return std::unexpected(out.error());
  return StepOut{std::move(*out), std::move(*h)};
}

Result<SeqOut> PolicyNet::forward_seq(const Obs& obs, const Tensor& h0, const Tensor& reset,
                                      ForwardCtx& ctx) {
  if (!recurrent()) return std::unexpected(make_error(Errc::unsupported, "net is feedforward"));
  if (reset.dtype() != DType::u8 || reset.rank() != 2)
    return std::unexpected(err("forward_seq: reset u8 [T,B]"));
  const auto T = reset.shape()[0], B = reset.shape()[1];
  if (obs.entities.shape()[0] != T * B || h0.rank() != 2 || h0.shape()[0] != B ||
      h0.shape()[1] != hidden_dim())
    return std::unexpected(err("forward_seq: obs rows must be T*B, h0 [B,H]"));
  auto feat = enc_.forward(obs.entities, obs.mask, obs.global, ctx);
  if (!feat) return std::unexpected(feat.error());
  tape_.clear();
  keeps_.clear();
  std::vector<Tensor> hs;
  hs.reserve(static_cast<std::size_t>(T));
  Tensor h = h0;
  for (std::int64_t t = 0; t < T; ++t) {
    auto keep = keep_row(reset, t);
    if (!keep) return std::unexpected(keep.error());
    auto hm = mask_rows(h, *keep);
    if (!hm) return std::unexpected(hm.error());
    auto xt = narrow_rows(*feat, t * B, B);
    if (!xt) return std::unexpected(xt.error());
    GruCache cache{*xt, *xt, *xt, *xt, *xt, *xt};  // placeholders, overwritten by step()
    auto hn = gru_->step(*xt, *hm, ctx.train ? &cache : nullptr);
    if (!hn) return std::unexpected(hn.error());
    if (ctx.train) {
      tape_.push_back(std::move(cache));
      keeps_.push_back(std::move(*keep));
    }
    hs.push_back(*hn);
    h = std::move(*hn);
  }
  tape_T_ = T;
  tape_B_ = B;
  auto hall = concat_rows(hs);
  if (!hall) return std::unexpected(hall.error());
  auto out = heads_forward(*hall, ctx);
  if (!out) return std::unexpected(out.error());
  return SeqOut{std::move(*out), std::move(h)};
}

Result<void> PolicyNet::backward_seq(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx) {
  if (!recurrent() || tape_.empty() || static_cast<std::int64_t>(tape_.size()) != tape_T_)
    return std::unexpected(make_error(Errc::unsupported, "backward_seq: no tape (forward_seq with train=true)"));
  const auto T = tape_T_, B = tape_B_, H = hidden_dim();
  auto dh_all = heads_backward(dlogits, dvalue, ctx);  // [T*B,H]
  if (!dh_all) return std::unexpected(dh_all.error());
  std::int64_t sh[2] = {B, H};
  auto carry = Tensor::zeros(sh, DType::f32, dh_all->device());
  if (!carry) return std::unexpected(carry.error());
  std::vector<Tensor> dxs(static_cast<std::size_t>(T), *carry);
  Tensor dcarry = *carry;
  for (std::int64_t t = T - 1; t >= 0; --t) {
    auto dh_t = narrow_rows(*dh_all, t * B, B);
    if (!dh_t) return std::unexpected(dh_t.error());
    auto dh_out = add(*dh_t, dcarry);
    if (!dh_out) return std::unexpected(dh_out.error());
    auto g = gru_->step_backward(tape_[static_cast<std::size_t>(t)], *dh_out);
    if (!g) return std::unexpected(g.error());
    dxs[static_cast<std::size_t>(t)] = std::move(g->dx);
    auto dm = mask_rows(g->dh_in, keeps_[static_cast<std::size_t>(t)]);
    if (!dm) return std::unexpected(dm.error());
    dcarry = std::move(*dm);
  }
  auto dfeat = concat_rows(dxs);
  if (!dfeat) return std::unexpected(dfeat.error());
  auto g = enc_.backward(*dfeat, ctx);
  if (!g) return std::unexpected(g.error());
  tape_.clear();
  keeps_.clear();
  return {};
}

}  // namespace gyre
