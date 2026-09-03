#include "gyre/module.hpp"

#include <cmath>
#include <cstring>

namespace gyre {

Result<Param> make_param(Tensor value) {
  auto g = Tensor::zeros(value.shape(), value.dtype(), value.device());
  if (!g) return std::unexpected(g.error());
  return Param{std::move(value), std::move(*g)};
}

Result<void> Module::zero_grad() {
  for (auto& p : parameters()) {
    auto r = fill_zero(p.grad);
    if (!r) return r;
  }
  return {};
}

Result<LossPair> softmax_cross_entropy(const Tensor& logits, const Tensor& targets_i32) {
  if (logits.dtype() != DType::f32 || targets_i32.dtype() != DType::i32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "CE dtypes"));
  }
  if (logits.rank() != 3 || targets_i32.rank() != 2) {
    return std::unexpected(make_error(Errc::invalid_shape, "CE [B,T,V] / [B,T]"));
  }
  const auto B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
  if (targets_i32.shape()[0] != B || targets_i32.shape()[1] != T) {
    return std::unexpected(make_error(Errc::invalid_shape, "CE target shape"));
  }
  auto sm = softmax_last(logits);
  if (!sm) return std::unexpected(sm.error());
  auto p = sm->host_span<float>();
  auto y = targets_i32.host_span<std::int32_t>();
  if (!p || !y) return std::unexpected(make_error(Errc::not_cpu, "host"));

  float loss = 0;
  auto dlog = Tensor::zeros(logits.shape(), DType::f32, logits.device());
  if (!dlog) return std::unexpected(dlog.error());
  auto d = dlog->host_span<float>();
  if (!d) return std::unexpected(d.error());

  const float n = static_cast<float>(B * T);
  for (std::int64_t i = 0; i < B * T; ++i) {
    auto t = (*y)[static_cast<std::size_t>(i)];
    if (t < 0 || t >= V) return std::unexpected(make_error(Errc::invalid_shape, "target OOB"));
    float pt = (*p)[static_cast<std::size_t>(i * V + t)];
    loss += -std::log(std::max(pt, 1e-12f));
    for (std::int64_t v = 0; v < V; ++v) {
      float g = (*p)[static_cast<std::size_t>(i * V + v)];
      if (v == t) g -= 1.f;
      (*d)[static_cast<std::size_t>(i * V + v)] = g / n;
    }
  }
  loss /= n;
  auto lv = Tensor::empty(std::span<const std::int64_t>(), DType::f32, logits.device());
  if (!lv) return std::unexpected(lv.error());
  auto ls = lv->host_span<float>();
  if (!ls) return std::unexpected(ls.error());
  (*ls)[0] = loss;
  return LossPair{std::move(*lv), std::move(*dlog)};
}

}  // namespace gyre
