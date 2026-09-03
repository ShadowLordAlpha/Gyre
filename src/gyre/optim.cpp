#include "gyre/optim.hpp"

#include <cmath>

namespace gyre {

Result<Adam> Adam::create(std::span<Param> params, float lr) {
  Adam a;
  a.lr = lr;
  for (auto& p : params) {
    auto m = Tensor::zeros(p.value.shape(), DType::f32, p.value.device());
    auto v = Tensor::zeros(p.value.shape(), DType::f32, p.value.device());
    if (!m || !v) return std::unexpected(m ? v.error() : m.error());
    a.m.push_back(std::move(*m));
    a.v.push_back(std::move(*v));
  }
  return a;
}

Result<void> Adam::step(std::span<Param> params) {
  if (params.size() != m.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "adam param count"));
  }
  ++t;
  const float b1t = 1.f - std::pow(beta1, static_cast<float>(t));
  const float b2t = 1.f - std::pow(beta2, static_cast<float>(t));
  for (std::size_t i = 0; i < params.size(); ++i) {
    auto g = params[i].grad.host_span<float>();
    auto w = params[i].value.host_span<float>();
    auto pm = m[i].host_span<float>();
    auto pv = v[i].host_span<float>();
    if (!g || !w || !pm || !pv) return std::unexpected(make_error(Errc::not_cpu, "host"));
    for (std::size_t j = 0; j < w->size(); ++j) {
      (*pm)[j] = beta1 * (*pm)[j] + (1.f - beta1) * (*g)[j];
      (*pv)[j] = beta2 * (*pv)[j] + (1.f - beta2) * (*g)[j] * (*g)[j];
      float mhat = (*pm)[j] / b1t;
      float vhat = (*pv)[j] / b2t;
      (*w)[j] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
  }
  return {};
}

}  // namespace gyre
