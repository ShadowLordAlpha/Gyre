#include "gyre/nn/attention.hpp"

#include <cmath>
#include <cstring>
#include <span>

namespace gyre {
namespace {

void apply_causal_recency(std::span<float> scores, std::int64_t B, std::int64_t H, std::int64_t T,
                          bool alibi) {
  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t h = 0; h < H; ++h) {
      const float slope =
          alibi ? std::pow(2.f, -8.f * static_cast<float>(h + 1) / static_cast<float>(H)) : 0.f;
      float* s = scores.data() + ((b * H + h) * T * T);
      for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t j = 0; j < T; ++j) {
          if (j > t)
            s[t * T + j] += -1e9f;
          else if (alibi)
            s[t * T + j] += -slope * static_cast<float>(t - j);
        }
      }
    }
  }
}

void take(std::vector<Param>& flat, std::span<Param> s) {
  for (auto& p : s) flat.push_back(Param{p.value, p.grad});
}

}  // namespace

CausalSelfAttention::CausalSelfAttention(Linear q, Linear k, Linear v, Linear o, std::int64_t n_head,
                                         bool alibi)
    : q_(std::move(q)),
      k_(std::move(k)),
      v_(std::move(v)),
      o_(std::move(o)),
      n_head_(n_head),
      recency_alibi_(alibi) {
  rebind();
}

void CausalSelfAttention::rebind() {
  flat_.clear();
  take(flat_, q_.parameters());
  take(flat_, k_.parameters());
  take(flat_, v_.parameters());
  take(flat_, o_.parameters());
}

Result<CausalSelfAttention> CausalSelfAttention::create(std::int64_t d_model, std::int64_t n_head,
                                                        std::shared_ptr<Device> d, Rng& rng,
                                                        float resid_scale, bool recency_alibi) {
  if (n_head <= 0 || d_model % n_head != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "d_model % n_head"));
  }
  auto q = Linear::create(d_model, d_model, d, rng, 0.02f, 1.f);
  auto k = Linear::create(d_model, d_model, d, rng, 0.02f, 1.f);
  auto v = Linear::create(d_model, d_model, d, rng, 0.02f, 1.f);
  auto o = Linear::create(d_model, d_model, d, rng, 0.02f, resid_scale);
  if (!q || !k || !v || !o) {
    return std::unexpected(q ? (k ? (v ? o.error() : v.error()) : k.error()) : q.error());
  }
  return CausalSelfAttention(std::move(*q), std::move(*k), std::move(*v), std::move(*o), n_head,
                             recency_alibi);
}

Result<Tensor> CausalSelfAttention::forward(const Tensor& x, ForwardCtx& ctx) {
  if (x.rank() != 3) return std::unexpected(make_error(Errc::invalid_shape, "attn [B,T,C]"));
  const auto B = x.shape()[0], T = x.shape()[1], C = x.shape()[2];
  const auto dk = C / n_head_;
  if (ctx.train) saved_x_ = x;
  auto q = q_.forward(x, ctx);
  auto k = k_.forward(x, ctx);
  auto v = v_.forward(x, ctx);
  if (!q || !k || !v) return std::unexpected(q ? (k ? v.error() : k.error()) : q.error());
  if (ctx.train) {
    saved_q_ = *q;
    saved_k_ = *k;
    saved_v_ = *v;
  }
  std::int64_t sh[4] = {B, T, n_head_, dk};
  auto qr = reshape(*q, sh);
  auto kr = reshape(*k, sh);
  auto vr = reshape(*v, sh);
  if (!qr || !kr || !vr) return std::unexpected(make_error(Errc::invalid_shape, "reshape qkv"));
  auto qh = permute_bthd_bhtd(*qr);
  auto kh = permute_bthd_bhtd(*kr);
  auto vh = permute_bthd_bhtd(*vr);
  if (!qh || !kh || !vh) return std::unexpected(qh ? (kh ? vh.error() : kh.error()) : qh.error());
  if (ctx.train) {
    saved_qh_ = *qh;
    saved_kh_ = *kh;
    saved_vh_ = *vh;
  }
  auto kt = transpose_last2(*kh);
  if (!kt) return kt;
  auto scores = bmm(*qh, *kt);
  if (!scores) return scores;
  auto sp = scores->host_span<float>();
  if (!sp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  for (auto& z : *sp) z *= scale;
  apply_causal_recency(*sp, B, n_head_, T, recency_alibi_);
  auto w = softmax_last(*scores);
  if (!w) return w;
  if (ctx.train) saved_w_ = *w;
  auto y = bmm(*w, *vh);
  if (!y) return y;
  auto y2 = permute_bhtd_bthd(*y);
  if (!y2) return y2;
  std::int64_t osh[3] = {B, T, C};
  auto y3 = reshape(*y2, osh);
  if (!y3) return y3;
  if (ctx.train) saved_y3_ = *y3;
  return o_.forward(*y3, ctx);
}

Result<void> CausalSelfAttention::backward(const Tensor& d_att, ForwardCtx& ctx) {
  if (!saved_x_ || !saved_qh_ || !saved_kh_ || !saved_vh_ || !saved_w_ || !saved_y3_) {
    return std::unexpected(make_error(Errc::unsupported, "attn tape"));
  }
  const auto B = saved_x_->shape()[0], T = saved_x_->shape()[1], C = saved_x_->shape()[2];
  const auto dk = C / n_head_;
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  auto ro = o_.backward(d_att, ctx);
  if (!ro) return ro;
  if (!ctx.dx) return std::unexpected(make_error(Errc::unsupported, "o dx"));
  Tensor dy3 = std::move(*ctx.dx);
  std::int64_t sh4[4] = {B, T, n_head_, dk};
  auto dy2 = reshape(dy3, sh4);
  if (!dy2) return std::unexpected(dy2.error());
  auto dyh = permute_bthd_bhtd(*dy2);
  if (!dyh) return std::unexpected(dyh.error());
  auto vt = transpose_last2(*saved_vh_);
  if (!vt) return std::unexpected(vt.error());
  auto dw = bmm(*dyh, *vt);
  if (!dw) return std::unexpected(dw.error());
  auto wt = transpose_last2(*saved_w_);
  if (!wt) return std::unexpected(wt.error());
  auto dvh = bmm(*wt, *dyh);
  if (!dvh) return std::unexpected(dvh.error());
  auto dz = softmax_last_backward(*saved_w_, *dw);
  if (!dz) return std::unexpected(dz.error());
  auto dzp = dz->host_span<float>();
  if (!dzp) return std::unexpected(dzp.error());
  for (auto& z : *dzp) z *= scale;
  auto dqh = bmm(*dz, *saved_kh_);
  if (!dqh) return std::unexpected(dqh.error());
  auto dzt = transpose_last2(*dz);
  if (!dzt) return std::unexpected(dzt.error());
  auto dkh = bmm(*dzt, *saved_qh_);
  if (!dkh) return std::unexpected(dkh.error());
  auto q_btc = permute_bhtd_bthd(*dqh);
  auto k_btc = permute_bhtd_bthd(*dkh);
  auto v_btc = permute_bhtd_bthd(*dvh);
  if (!q_btc || !k_btc || !v_btc) return std::unexpected(make_error(Errc::invalid_shape, "perm bwd"));
  std::int64_t osh[3] = {B, T, C};
  auto dq = reshape(*q_btc, osh);
  auto dkey = reshape(*k_btc, osh);
  auto dv = reshape(*v_btc, osh);
  if (!dq || !dkey || !dv) return std::unexpected(make_error(Errc::invalid_shape, "reshape bwd"));
  auto rq = q_.backward(*dq, ctx);
  if (!rq || !ctx.dx) return rq ? std::unexpected(make_error(Errc::unsupported, "q dx")) : rq;
  Tensor dxq = std::move(*ctx.dx);
  auto rk = k_.backward(*dkey, ctx);
  if (!rk || !ctx.dx) return rk ? std::unexpected(make_error(Errc::unsupported, "k dx")) : rk;
  Tensor dxk = std::move(*ctx.dx);
  auto rv = v_.backward(*dv, ctx);
  if (!rv || !ctx.dx) return rv ? std::unexpected(make_error(Errc::unsupported, "v dx")) : rv;
  Tensor dxv = std::move(*ctx.dx);
  auto t = add(dxq, dxk);
  if (!t) return std::unexpected(t.error());
  auto dx = add(*t, dxv);
  if (!dx) return std::unexpected(dx.error());
  ctx.dx = std::make_unique<Tensor>(std::move(*dx));
  return {};
}

}  // namespace gyre
