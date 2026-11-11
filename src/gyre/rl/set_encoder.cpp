#include "gyre/rl/set_encoder.hpp"

#include "gyre/rl/ops.hpp"

#include <cmath>

namespace gyre {
namespace {

void take(std::vector<Param>& flat, std::span<Param> s) {
  for (auto& p : s) flat.push_back(Param{p.value, p.grad});
}

// Unwrap helpers keep the error plumbing readable.
#define GYRE_TRY(var, expr)                                  \
  auto var##_r = (expr);                                     \
  if (!var##_r) return std::unexpected(var##_r.error());     \
  auto& var = *var##_r

Result<Tensor> take_dx(ForwardCtx& ctx, const char* who) {
  if (!ctx.dx) return std::unexpected(make_error(Errc::unsupported, who));
  Tensor t = std::move(*ctx.dx);
  ctx.dx.reset();
  return t;
}

}  // namespace

// ---------------------------------------------------------------- SetBlock

SetBlock::SetBlock(LayerNorm ln1, Linear q, Linear k, Linear v, Linear o, LayerNorm ln2, Linear fc1,
                   Linear fc2, std::int64_t n_head)
    : ln1_(std::move(ln1)),
      q_(std::move(q)),
      k_(std::move(k)),
      v_(std::move(v)),
      o_(std::move(o)),
      ln2_(std::move(ln2)),
      fc1_(std::move(fc1)),
      fc2_(std::move(fc2)),
      n_head_(n_head) {
  rebind();
}

void SetBlock::rebind() {
  flat_.clear();
  take(flat_, ln1_.parameters());
  take(flat_, q_.parameters());
  take(flat_, k_.parameters());
  take(flat_, v_.parameters());
  take(flat_, o_.parameters());
  take(flat_, ln2_.parameters());
  take(flat_, fc1_.parameters());
  take(flat_, fc2_.parameters());
}

Result<SetBlock> SetBlock::create(const SetEncoderConfig& c, std::shared_ptr<Device> d, Rng& rng) {
  if (c.n_head <= 0 || c.d_model % c.n_head != 0)
    return std::unexpected(make_error(Errc::invalid_shape, "d_model % n_head"));
  const float s = 1.f / std::sqrt(static_cast<float>(c.d_model));
  const float sf = 1.f / std::sqrt(static_cast<float>(c.d_ff));
  GYRE_TRY(ln1, LayerNorm::create(c.d_model, d));
  GYRE_TRY(q, Linear::create(c.d_model, c.d_model, d, rng, s));
  GYRE_TRY(k, Linear::create(c.d_model, c.d_model, d, rng, s));
  GYRE_TRY(v, Linear::create(c.d_model, c.d_model, d, rng, s));
  GYRE_TRY(o, Linear::create(c.d_model, c.d_model, d, rng, s));
  GYRE_TRY(ln2, LayerNorm::create(c.d_model, d));
  GYRE_TRY(fc1, Linear::create(c.d_model, c.d_ff, d, rng, s));
  GYRE_TRY(fc2, Linear::create(c.d_ff, c.d_model, d, rng, sf));
  return SetBlock(std::move(ln1), std::move(q), std::move(k), std::move(v), std::move(o),
                  std::move(ln2), std::move(fc1), std::move(fc2), c.n_head);
}

Result<Tensor> SetBlock::forward(const Tensor& x, const Tensor& mask, ForwardCtx& ctx) {
  if (x.rank() != 3) return std::unexpected(make_error(Errc::invalid_shape, "block [B,N,C]"));
  const auto B = x.shape()[0], N = x.shape()[1], C = x.shape()[2];
  const auto dk = C / n_head_;
  GYRE_TRY(n1, ln1_.forward(x, ctx));
  GYRE_TRY(q, q_.forward(n1, ctx));
  GYRE_TRY(k, k_.forward(n1, ctx));
  GYRE_TRY(v, v_.forward(n1, ctx));
  std::int64_t sh[4] = {B, N, n_head_, dk};
  GYRE_TRY(qr, reshape(q, sh));
  GYRE_TRY(kr, reshape(k, sh));
  GYRE_TRY(vr, reshape(v, sh));
  GYRE_TRY(qh, permute_bthd_bhtd(qr));
  GYRE_TRY(kh, permute_bthd_bhtd(kr));
  GYRE_TRY(vh, permute_bthd_bhtd(vr));
  GYRE_TRY(w, attn_probs(qh, kh, &mask, 1.f / std::sqrt(static_cast<float>(dk))));  // fused q.k, mask, softmax
  GYRE_TRY(y, bmm(w, vh));
  GYRE_TRY(y2, permute_bhtd_bthd(y));
  std::int64_t osh[3] = {B, N, C};
  GYRE_TRY(y3, reshape(y2, osh));
  GYRE_TRY(o, o_.forward(y3, ctx));
  GYRE_TRY(h1, add(x, o));
  GYRE_TRY(n2, ln2_.forward(h1, ctx));
  GYRE_TRY(f1, fc1_.forward(n2, ctx));
  GYRE_TRY(g, gelu(f1));
  GYRE_TRY(f2, fc2_.forward(g, ctx));
  if (ctx.train) {
    saved_qh_ = qh;
    saved_kh_ = kh;
    saved_vh_ = vh;
    saved_w_ = w;
    saved_fc1_ = f1;
  }
  return add(h1, f2);
}

Result<Tensor> SetBlock::backward(const Tensor& dy, ForwardCtx& ctx) {
  if (!saved_w_ || !saved_fc1_)
    return std::unexpected(make_error(Errc::unsupported, "set block: no tape"));
  const auto B = saved_qh_->shape()[0], H = saved_qh_->shape()[1], N = saved_qh_->shape()[2],
             dk = saved_qh_->shape()[3];
  const auto C = H * dk;
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));

  // MLP branch: out = h1 + fc2(gelu(fc1(ln2(h1))))
  if (auto r = fc2_.backward(dy, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dg, take_dx(ctx, "fc2 dx"));
  GYRE_TRY(df1, gelu_backward(*saved_fc1_, dg));
  if (auto r = fc1_.backward(df1, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dn2, take_dx(ctx, "fc1 dx"));
  if (auto r = ln2_.backward(dn2, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dh1a, take_dx(ctx, "ln2 dx"));
  GYRE_TRY(dh1, add(dy, dh1a));

  // Attention branch: h1 = x + o(attn(ln1(x)))
  if (auto r = o_.backward(dh1, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dy3, take_dx(ctx, "o dx"));
  std::int64_t sh4[4] = {B, N, H, dk};
  GYRE_TRY(dy2, reshape(dy3, sh4));
  GYRE_TRY(dyh, permute_bthd_bhtd(dy2));
  // dz = scale * softmax-backward(w, dy.v); masked keys have w == 0 -> dz == 0 exactly
  GYRE_TRY(dz, attn_probs_backward(*saved_w_, dyh, *saved_vh_, scale));
  GYRE_TRY(dvh, bmm_tn(*saved_w_, dyh));   // dv_j = sum_i w[i,j] dy_i
  GYRE_TRY(dqh, bmm(dz, *saved_kh_));      // dq_i = sum_j dz[i,j] k_j
  GYRE_TRY(dkh, bmm_tn(dz, *saved_qh_));   // dk_j = sum_i dz[i,j] q_i
  std::int64_t osh[3] = {B, N, C};
  GYRE_TRY(q_bthd, permute_bhtd_bthd(dqh));
  GYRE_TRY(k_bthd, permute_bhtd_bthd(dkh));
  GYRE_TRY(v_bthd, permute_bhtd_bthd(dvh));
  GYRE_TRY(dq, reshape(q_bthd, osh));
  GYRE_TRY(dkey, reshape(k_bthd, osh));
  GYRE_TRY(dv, reshape(v_bthd, osh));
  if (auto r = q_.backward(dq, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dxq, take_dx(ctx, "q dx"));
  if (auto r = k_.backward(dkey, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dxk, take_dx(ctx, "k dx"));
  if (auto r = v_.backward(dv, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dxv, take_dx(ctx, "v dx"));
  GYRE_TRY(t1, add(dxq, dxk));
  GYRE_TRY(dn1, add(t1, dxv));
  if (auto r = ln1_.backward(dn1, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dxa, take_dx(ctx, "ln1 dx"));
  return add(dh1, dxa);
}

// -------------------------------------------------------------- SetEncoder

SetEncoder::SetEncoder(SetEncoderConfig c, Linear embed, std::vector<SetBlock> blocks, LayerNorm ln_f,
                       Linear proj)
    : cfg_(c),
      embed_(std::move(embed)),
      blocks_(std::move(blocks)),
      ln_f_(std::move(ln_f)),
      proj_(std::move(proj)) {
  rebind();
}

void SetEncoder::rebind() {
  flat_.clear();
  take(flat_, embed_.parameters());
  for (auto& b : blocks_) take(flat_, std::span<Param>(b.parameters()));
  take(flat_, ln_f_.parameters());
  take(flat_, proj_.parameters());
}

Result<SetEncoder> SetEncoder::create(const SetEncoderConfig& c, std::shared_ptr<Device> d, Rng& rng) {
  if (c.feat_dim <= 0 || c.d_model <= 0 || c.n_layer < 0 || c.out_dim <= 0 || c.global_dim < 0)
    return std::unexpected(make_error(Errc::invalid_shape, "SetEncoderConfig"));
  GYRE_TRY(embed, Linear::create(c.feat_dim, c.d_model, d, rng, 1.f / std::sqrt(float(c.feat_dim))));
  std::vector<SetBlock> blocks;
  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    GYRE_TRY(b, SetBlock::create(c, d, rng));
    blocks.push_back(std::move(b));
  }
  GYRE_TRY(ln_f, LayerNorm::create(c.d_model, d));
  const auto pin = 2 * c.d_model + c.global_dim;
  GYRE_TRY(proj, Linear::create(pin, c.out_dim, d, rng, 1.f / std::sqrt(float(pin))));
  return SetEncoder(c, std::move(embed), std::move(blocks), std::move(ln_f), std::move(proj));
}

Result<void> SetEncoder::zero_grad() {
  for (auto& p : flat_) {
    if (auto r = fill_zero(p.grad); !r) return r;
  }
  return {};
}

Result<Tensor> SetEncoder::forward(const Tensor& entities, const Tensor& mask, const Tensor& global,
                                   ForwardCtx& ctx) {
  if (entities.rank() != 3 || entities.shape()[2] != cfg_.feat_dim)
    return std::unexpected(make_error(Errc::invalid_shape, "entities [B,N,feat_dim]"));
  if (entities.shape()[1] > cfg_.max_entities)
    return std::unexpected(make_error(Errc::invalid_shape, "N exceeds max_entities"));
  if (global.rank() != 2 || global.shape()[0] != entities.shape()[0] ||
      global.shape()[1] != cfg_.global_dim)
    return std::unexpected(make_error(Errc::invalid_shape, "global [B,global_dim]"));
  // Padded slots are zeroed (copied, not multiplied) before anything else: a NaN/Inf in a padded
  // row must not reach valid rows through 0 * NaN in the attention products.
  const auto B0 = entities.shape()[0], N0 = entities.shape()[1], F0 = entities.shape()[2];
  std::int64_t flat_sh[2] = {B0 * N0, F0}, mflat_sh[1] = {B0 * N0}, ent_sh[3] = {B0, N0, F0};
  GYRE_TRY(ent_flat, reshape(entities, flat_sh));
  GYRE_TRY(mask_flat, reshape(mask, mflat_sh));
  GYRE_TRY(ent_clean, mask_rows(ent_flat, mask_flat));
  GYRE_TRY(ent_in, reshape(ent_clean, ent_sh));
  GYRE_TRY(h0, embed_.forward(ent_in, ctx));
  Tensor h = std::move(h0);
  for (auto& b : blocks_) {
    GYRE_TRY(hn, b.forward(h, mask, ctx));
    h = std::move(hn);
  }
  GYRE_TRY(hf, ln_f_.forward(h, ctx));
  Tensor am = hf;  // placeholder; overwritten by the pool
  GYRE_TRY(pooled, masked_mean_max_pool(hf, mask, am));
  GYRE_TRY(cat, concat_last(pooled, global));
  GYRE_TRY(pre, proj_.forward(cat, ctx));
  if (ctx.train) {
    saved_mask_ = mask;
    saved_argmax_ = am;
    saved_proj_pre_ = pre;
    saved_N_ = entities.shape()[1];
  }
  return gelu(pre);
}

Result<SetEncoderGrads> SetEncoder::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_mask_ || !saved_argmax_ || !saved_proj_pre_)
    return std::unexpected(make_error(Errc::unsupported, "encoder: no tape"));
  GYRE_TRY(dpre, gelu_backward(*saved_proj_pre_, d_out));
  if (auto r = proj_.backward(dpre, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dcat, take_dx(ctx, "proj dx"));
  Tensor dpool = dcat, dglobal = dcat;  // placeholders (Tensor has no default ctor)
  if (auto r = split_last(dcat, 2 * cfg_.d_model, dpool, dglobal); !r) return std::unexpected(r.error());
  GYRE_TRY(dhf, masked_mean_max_pool_backward(dpool, *saved_mask_, *saved_argmax_, saved_N_));
  if (auto r = ln_f_.backward(dhf, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dh0, take_dx(ctx, "ln_f dx"));
  Tensor dh = std::move(dh0);
  for (std::size_t i = blocks_.size(); i-- > 0;) {
    GYRE_TRY(dprev, blocks_[i].backward(dh, ctx));
    dh = std::move(dprev);
  }
  if (auto r = embed_.backward(dh, ctx); !r) return std::unexpected(r.error());
  GYRE_TRY(dent_raw, take_dx(ctx, "embed dx"));
  std::int64_t dflat_sh[2] = {dent_raw.shape()[0] * dent_raw.shape()[1], dent_raw.shape()[2]};
  std::int64_t dmask_sh[1] = {dent_raw.shape()[0] * dent_raw.shape()[1]};
  std::int64_t dent_sh[3] = {dent_raw.shape()[0], dent_raw.shape()[1], dent_raw.shape()[2]};
  GYRE_TRY(dflat, reshape(dent_raw, dflat_sh));
  GYRE_TRY(mflat, reshape(*saved_mask_, dmask_sh));
  GYRE_TRY(dclean, mask_rows(dflat, mflat));  // padded slots: exactly zero
  GYRE_TRY(dent, reshape(dclean, dent_sh));
  return SetEncoderGrads{std::move(dent), std::move(dglobal)};
}

#undef GYRE_TRY

}  // namespace gyre
