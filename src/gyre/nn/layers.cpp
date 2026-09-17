#include "gyre/nn/layers.hpp"

#include <cmath>
#include <vector>

namespace gyre {

Result<Linear> Linear::create(std::int64_t in, std::int64_t out, std::shared_ptr<Device> dev, Rng& rng,
                              float std, float residual_scale) {
  auto cpu = Device::cpu();
  if (!cpu) return std::unexpected(cpu.error());
  std::int64_t wsh[2] = {in, out};
  std::int64_t bsh[1] = {out};
  auto W = Tensor::empty(wsh, DType::f32, *cpu);
  auto b = Tensor::zeros(bsh, DType::f32, *cpu);
  if (!W || !b) return std::unexpected(W ? b.error() : W.error());
  auto wp = W->host_span<float>();
  if (!wp) return std::unexpected(wp.error());
  const float s = std * residual_scale;
  for (auto& v : *wp) v = rng.normal(0.f, s);
  auto pw = make_param(std::move(*W));
  auto pb = make_param(std::move(*b));
  if (!pw || !pb) return std::unexpected(pw ? pb.error() : pw.error());
  std::vector<Param> ps;
  ps.push_back(std::move(*pw));
  ps.push_back(std::move(*pb));
  if (dev && dev->kind() != DeviceKind::cpu) {
    for (auto& p : ps) {
      auto v = p.value.to(dev);
      auto g = p.grad.to(dev);
      if (!v || !g) return std::unexpected(v ? g.error() : v.error());
      p.value = std::move(*v);
      p.grad = std::move(*g);
    }
  }
  return Linear(std::move(ps));
}

void Linear::rebind_lora() {
  lora_flat_.clear();
  if (loraA_ && loraB_) {
    lora_flat_.push_back(Param{loraA_->value, loraA_->grad});
    lora_flat_.push_back(Param{loraB_->value, loraB_->grad});
  }
}

Result<void> Linear::set_lora(Tensor A, Tensor B, float scale) {
  if (params_.empty()) return std::unexpected(make_error(Errc::invalid_shape, "linear empty"));
  const auto in = params_[0].value.shape()[0];
  const auto out = params_[0].value.shape()[1];
  if (A.rank() != 2 || B.rank() != 2 || A.shape()[0] != in || B.shape()[1] != out
      || A.shape()[1] != B.shape()[0]) {
    return std::unexpected(make_error(Errc::invalid_shape, "lora A [in,r] B [r,out]"));
  }
  auto device = params_[0].value.device();
  if (device && A.device() && A.device()->kind() != device->kind()) {
    auto a = A.to(device);
    if (!a) return std::unexpected(a.error());
    A = std::move(*a);
  }
  if (device && B.device() && B.device()->kind() != device->kind()) {
    auto b = B.to(device);
    if (!b) return std::unexpected(b.error());
    B = std::move(*b);
  }
  auto pA = make_param(std::move(A));
  auto pB = make_param(std::move(B));
  if (!pA || !pB) return std::unexpected(pA ? pB.error() : pA.error());
  loraA_ = std::move(*pA);
  loraB_ = std::move(*pB);
  lora_scale_ = scale;
  rebind_lora();
  return {};
}

void Linear::clear_lora() {
  loraA_.reset();
  loraB_.reset();
  lora_flat_.clear();
  lora_scale_ = 0.f;
  saved_xa_.reset();
}

Result<Tensor> Linear::forward(const Tensor& x, ForwardCtx& ctx) {
  if (ctx.train) saved_x_ = x;
  else saved_x_.reset();
  auto y = linear(x, params_[0].value, params_[1].value);
  if (!y) return y;
  if (!loraA_ || !loraB_ || lora_scale_ == 0.f) {
    saved_xa_.reset();
    return y;
  }
  auto xf = flatten_leading(x);
  if (!xf) return std::unexpected(xf.error());
  auto xa = matmul(*xf, loraA_->value);
  if (!xa) return xa;
  if (ctx.train) saved_xa_ = *xa;
  else saved_xa_.reset();
  auto xb = matmul(*xa, loraB_->value);
  if (!xb) return xb;
  auto scaled = mul_scalar(*xb, lora_scale_);
  if (!scaled) return scaled;
  auto extra = unflatten_like(std::move(*scaled), *y);
  if (!extra) return extra;
  return add(*y, *extra);
}

Result<void> Linear::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_x_) return std::unexpected(make_error(Errc::unsupported, "no saved x"));
  const Tensor& x = *saved_x_;
  auto xf = flatten_leading(x);
  auto df = flatten_leading(d_out);
  if (!xf || !df) return std::unexpected(xf ? df.error() : xf.error());
  if (!freeze_base_) {
    auto xt = transpose_last2(*xf);
    if (!xt) return std::unexpected(xt.error());
    auto dW = matmul(*xt, *df);
    if (!dW) return std::unexpected(dW.error());
    auto r1 = add_(params_[0].grad, *dW);
    if (!r1) return r1;
    auto db = sum_dim(*df, 0, false);
    if (!db) return std::unexpected(db.error());
    auto r2 = add_(params_[1].grad, *db);
    if (!r2) return r2;
  }
  auto Wt = transpose_last2(params_[0].value);
  if (!Wt) return std::unexpected(Wt.error());
  auto dx2 = matmul(*df, *Wt);
  if (!dx2) return std::unexpected(dx2.error());
  if (loraA_ && loraB_ && lora_scale_ != 0.f && saved_xa_) {
    auto dfS = mul_scalar(*df, lora_scale_);
    if (!dfS) return std::unexpected(dfS.error());
    auto xaT = transpose_last2(*saved_xa_);
    if (!xaT) return std::unexpected(xaT.error());
    auto dB = matmul(*xaT, *dfS);
    if (!dB) return std::unexpected(dB.error());
    auto rB = add_(loraB_->grad, *dB);
    if (!rB) return rB;
    auto BT = transpose_last2(loraB_->value);
    if (!BT) return std::unexpected(BT.error());
    auto dXa = matmul(*dfS, *BT);
    if (!dXa) return std::unexpected(dXa.error());
    auto xT = transpose_last2(*xf);
    if (!xT) return std::unexpected(xT.error());
    auto dA = matmul(*xT, *dXa);
    if (!dA) return std::unexpected(dA.error());
    auto rA = add_(loraA_->grad, *dA);
    if (!rA) return rA;
    auto AT = transpose_last2(loraA_->value);
    if (!AT) return std::unexpected(AT.error());
    auto dxL = matmul(*dXa, *AT);
    if (!dxL) return std::unexpected(dxL.error());
    auto summed = add(*dx2, *dxL);
    if (!summed) return std::unexpected(summed.error());
    dx2 = std::move(*summed);
  }
  auto dx = unflatten_like(std::move(*dx2), x);
  if (!dx) return std::unexpected(dx.error());
  ctx.dx = std::make_unique<Tensor>(std::move(*dx));
  return {};
}

Result<LayerNorm> LayerNorm::create(std::int64_t n, std::shared_ptr<Device> dev, float eps) {
  auto cpu = Device::cpu();
  if (!cpu) return std::unexpected(cpu.error());
  std::int64_t sh[1] = {n};
  auto w = Tensor::empty(sh, DType::f32, *cpu);
  auto b = Tensor::zeros(sh, DType::f32, *cpu);
  if (!w || !b) return std::unexpected(w ? b.error() : w.error());
  auto wp = w->host_span<float>();
  if (!wp) return std::unexpected(wp.error());
  for (auto& v : *wp) v = 1.f;
  auto pw = make_param(std::move(*w));
  auto pb = make_param(std::move(*b));
  if (!pw || !pb) return std::unexpected(pw ? pb.error() : pw.error());
  std::vector<Param> ps;
  ps.push_back(std::move(*pw));
  ps.push_back(std::move(*pb));
  if (dev && dev->kind() != DeviceKind::cpu) {
    for (auto& p : ps) {
      auto v = p.value.to(dev);
      auto g = p.grad.to(dev);
      if (!v || !g) return std::unexpected(v ? g.error() : v.error());
      p.value = std::move(*v);
      p.grad = std::move(*g);
    }
  }
  return LayerNorm(std::move(ps), eps);
}

Result<Tensor> LayerNorm::forward(const Tensor& x, ForwardCtx& ctx) {
  if (ctx.train) saved_x_ = x;
  else saved_x_.reset();
  return layer_norm(x, params_[0].value, params_[1].value, eps_);
}

Result<void> LayerNorm::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_x_) return std::unexpected(make_error(Errc::unsupported, "no saved x"));
  auto dx = layer_norm_backward(*saved_x_, d_out, params_[0].value, params_[0].grad, params_[1].grad, eps_);
  if (!dx) return std::unexpected(dx.error());
  ctx.dx = std::make_unique<Tensor>(std::move(*dx));
  return {};
}

Result<Embedding> Embedding::create(std::int64_t vocab, std::int64_t d, std::shared_ptr<Device> dev,
                                   Rng& rng, float std) {
  auto cpu = Device::cpu();
  if (!cpu) return std::unexpected(cpu.error());
  std::int64_t sh[2] = {vocab, d};
  auto W = Tensor::empty(sh, DType::f32, *cpu);
  if (!W) return std::unexpected(W.error());
  auto p = W->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = rng.normal(0.f, std);
  auto pw = make_param(std::move(*W));
  if (!pw) return std::unexpected(pw.error());
  std::vector<Param> ps;
  ps.push_back(std::move(*pw));
  if (dev && dev->kind() != DeviceKind::cpu) {
    for (auto& prm : ps) {
      auto v = prm.value.to(dev);
      auto g = prm.grad.to(dev);
      if (!v || !g) return std::unexpected(v ? g.error() : v.error());
      prm.value = std::move(*v);
      prm.grad = std::move(*g);
    }
  }
  return Embedding(std::move(ps));
}

Result<Tensor> Embedding::forward(const Tensor& idx, ForwardCtx& ctx) {
  if (ctx.train) saved_idx_ = idx;
  else saved_idx_.reset();
  return embedding(params_[0].value, idx);
}

Result<void> Embedding::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_idx_) return std::unexpected(make_error(Errc::unsupported, "no saved idx"));
  auto r = embedding_backward(params_[0].grad, *saved_idx_, d_out);
  if (!r) return r;
  ctx.dx.reset();
  return {};
}

}  // namespace gyre
