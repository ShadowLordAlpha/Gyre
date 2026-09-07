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

Result<Tensor> Linear::forward(const Tensor& x, ForwardCtx& ctx) {
  if (ctx.train) saved_x_ = x;
  else saved_x_.reset();
  return linear(x, params_[0].value, params_[1].value);
}

Result<void> Linear::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_x_) return std::unexpected(make_error(Errc::unsupported, "no saved x"));
  const Tensor& x = *saved_x_;
  auto xf = flatten_leading(x);
  auto df = flatten_leading(d_out);
  if (!xf || !df) return std::unexpected(xf ? df.error() : xf.error());
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
  auto Wt = transpose_last2(params_[0].value);
  if (!Wt) return std::unexpected(Wt.error());
  auto dx2 = matmul(*df, *Wt);
  if (!dx2) return std::unexpected(dx2.error());
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
