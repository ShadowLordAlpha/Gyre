#include "gyre/nn/layers.hpp"

#include <cmath>
#include <vector>

namespace gyre {

Result<Linear> Linear::create(std::int64_t in, std::int64_t out, std::shared_ptr<Device> dev, Rng& rng,
                              float std, float residual_scale) {
  std::int64_t wsh[2] = {in, out};
  std::int64_t bsh[1] = {out};
  auto W = Tensor::empty(wsh, DType::f32, dev);
  auto b = Tensor::zeros(bsh, DType::f32, dev);
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
  std::int64_t sh[1] = {n};
  auto w = Tensor::empty(sh, DType::f32, dev);
  auto b = Tensor::zeros(sh, DType::f32, dev);
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
  return LayerNorm(std::move(ps), eps);
}

Result<Tensor> LayerNorm::forward(const Tensor& x, ForwardCtx& ctx) {
  if (ctx.train) saved_x_ = x;
  else saved_x_.reset();
  return layer_norm(x, params_[0].value, params_[1].value, eps_);
}

Result<void> LayerNorm::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_x_) return std::unexpected(make_error(Errc::unsupported, "no saved x"));
  const Tensor& x = *saved_x_;
  auto px = x.host_span<float>();
  auto pd = d_out.host_span<float>();
  auto pw = params_[0].value.host_span<float>();
  auto gw = params_[0].grad.host_span<float>();
  auto gb = params_[1].grad.host_span<float>();
  if (!px || !pd || !pw || !gw || !gb) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  auto dx = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!dx) return std::unexpected(dx.error());
  auto pdx = dx->host_span<float>();
  if (!pdx) return std::unexpected(pdx.error());
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xs = px->data() + r * C;
    const float* gs = pd->data() + r * C;
    float* dxs = pdx->data() + r * C;
    float mean = 0;
    for (std::int64_t i = 0; i < C; ++i) mean += xs[i];
    mean /= static_cast<float>(C);
    float var = 0;
    for (std::int64_t i = 0; i < C; ++i) {
      float d = xs[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(C);
    float inv = 1.f / std::sqrt(var + eps_);
    std::vector<float> xhat(static_cast<std::size_t>(C));
    std::vector<float> dxhat(static_cast<std::size_t>(C));
    for (std::int64_t i = 0; i < C; ++i) {
      xhat[static_cast<std::size_t>(i)] = (xs[i] - mean) * inv;
      dxhat[static_cast<std::size_t>(i)] = gs[i] * (*pw)[static_cast<std::size_t>(i)];
      (*gw)[static_cast<std::size_t>(i)] += gs[i] * xhat[static_cast<std::size_t>(i)];
      (*gb)[static_cast<std::size_t>(i)] += gs[i];
    }
    float sdx = 0, sdxh = 0;
    for (std::int64_t i = 0; i < C; ++i) {
      sdx += dxhat[static_cast<std::size_t>(i)];
      sdxh += dxhat[static_cast<std::size_t>(i)] * xhat[static_cast<std::size_t>(i)];
    }
    const float invC = inv / static_cast<float>(C);
    for (std::int64_t i = 0; i < C; ++i) {
      dxs[i] = invC * (static_cast<float>(C) * dxhat[static_cast<std::size_t>(i)] - sdx -
                       xhat[static_cast<std::size_t>(i)] * sdxh);
    }
  }
  ctx.dx = std::make_unique<Tensor>(std::move(*dx));
  return {};
}

Result<Embedding> Embedding::create(std::int64_t vocab, std::int64_t d, std::shared_ptr<Device> dev,
                                   Rng& rng, float std) {
  std::int64_t sh[2] = {vocab, d};
  auto W = Tensor::empty(sh, DType::f32, dev);
  if (!W) return std::unexpected(W.error());
  auto p = W->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = rng.normal(0.f, std);
  auto pw = make_param(std::move(*W));
  if (!pw) return std::unexpected(pw.error());
  std::vector<Param> ps;
  ps.push_back(std::move(*pw));
  return Embedding(std::move(ps));
}

Result<Tensor> Embedding::forward(const Tensor& idx, ForwardCtx& ctx) {
  if (ctx.train) saved_idx_ = idx;
  else saved_idx_.reset();
  return embedding(params_[0].value, idx);
}

Result<void> Embedding::backward(const Tensor& d_out, ForwardCtx& ctx) {
  if (!saved_idx_) return std::unexpected(make_error(Errc::unsupported, "no saved idx"));
  auto g = params_[0].grad.host_span<float>();
  auto i = saved_idx_->host_span<std::int32_t>();
  auto d = d_out.host_span<float>();
  if (!g || !i || !d) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto dim = params_[0].value.shape()[1];
  const auto V = params_[0].value.shape()[0];
  for (std::int64_t n = 0; n < saved_idx_->numel(); ++n) {
    auto id = (*i)[static_cast<std::size_t>(n)];
    if (id < 0 || id >= V) return std::unexpected(make_error(Errc::invalid_shape, "emb idx"));
    for (std::int64_t c = 0; c < dim; ++c)
      (*g)[static_cast<std::size_t>(id * dim + c)] += (*d)[static_cast<std::size_t>(n * dim + c)];
  }
  ctx.dx.reset();
  return {};
}

}  // namespace gyre
