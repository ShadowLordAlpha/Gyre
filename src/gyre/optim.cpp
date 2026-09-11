#include "gyre/optim.hpp"

#if defined(GYRE_VULKAN)
#include "vk/ops.hpp"
#endif

#include <cmath>
#include <limits>

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
#if defined(GYRE_VULKAN)
  if (!params.empty() && params[0].value.device() &&
      params[0].value.device()->kind() == DeviceKind::vulkan) {
    for (std::size_t i = 0; i < params.size(); ++i) {
      const float wd = (weight_decay > 0.f && params[i].value.rank() >= 2) ? weight_decay : 0.f;
      if (wd > 0.f) {
        auto scaled = vkops::mul_scalar(params[i].value, 1.f - lr * wd);
        if (!scaled) return std::unexpected(scaled.error());
        params[i].value = std::move(*scaled);
      }
      auto r = vkops::adam_step(params[i].value, params[i].grad, m[i], v[i], lr, beta1, beta2, eps, b1t,
                                b2t);
      if (!r) return r;
    }
    return {};
  }
#endif
  for (std::size_t i = 0; i < params.size(); ++i) {
    auto g = params[i].grad.host_span<float>();
    auto w = params[i].value.host_span<float>();
    auto pm = m[i].host_span<float>();
    auto pv = v[i].host_span<float>();
    if (!g || !w || !pm || !pv) return std::unexpected(make_error(Errc::not_cpu, "host"));
    const float wd = (weight_decay > 0.f && params[i].value.rank() >= 2) ? weight_decay : 0.f;
    for (std::size_t j = 0; j < w->size(); ++j) {
      if (wd > 0.f) (*w)[j] *= (1.f - lr * wd);
      float gi = (*g)[j];
      if (!std::isfinite(gi)) continue;
      (*pm)[j] = beta1 * (*pm)[j] + (1.f - beta1) * gi;
      (*pv)[j] = beta2 * (*pv)[j] + (1.f - beta2) * gi * gi;
      float mhat = (*pm)[j] / b1t;
      float vhat = (*pv)[j] / b2t;
      float upd = lr * mhat / (std::sqrt(vhat) + eps);
      if (std::isfinite(upd)) (*w)[j] -= upd;
    }
  }
  return {};
}

Result<float> clip_grad_norm(std::span<Param> params, float max_norm) {
  if (max_norm <= 0.f || params.empty()) return 0.f;
  double ss = 0;
#if defined(GYRE_VULKAN)
  if (params[0].grad.device() && params[0].grad.device()->kind() == DeviceKind::vulkan) {
    for (auto& p : params) {
      auto sq = vkops::mul(p.grad, p.grad);
      if (!sq) return std::unexpected(sq.error());
      auto s = vkops::sum(*sq);
      if (!s) return std::unexpected(s.error());
      auto v = s->item_f32();
      if (!v) return std::unexpected(v.error());
      ss += static_cast<double>(*v);
    }
    if (!std::isfinite(ss)) {
      return std::unexpected(make_error(Errc::overflow, "non-finite gradient"));
    }
    const float n = std::sqrt(static_cast<float>(ss));
    if (n <= max_norm) return n;
    const float scale = max_norm / (n + 1e-6f);
    for (auto& p : params) {
      auto g = vkops::mul_scalar(p.grad, scale);
      if (!g) return std::unexpected(g.error());
      p.grad = std::move(*g);
    }
    return n;
  }
#endif
  for (auto& p : params) {
    auto g = p.grad.host_span<float>();
    if (!g) return std::unexpected(g.error());
    for (float x : *g) ss += static_cast<double>(x) * static_cast<double>(x);
  }
  if (!std::isfinite(ss)) {
    return std::unexpected(make_error(Errc::overflow, "non-finite gradient"));
  }
  const float n = std::sqrt(static_cast<float>(ss));
  if (n <= max_norm) return n;
  const float scale = max_norm / (n + 1e-6f);
  for (auto& p : params) {
    auto g = p.grad.host_span<float>();
    if (!g) return std::unexpected(g.error());
    for (auto& x : *g) x *= scale;
  }
  return n;
}

}  // namespace gyre
