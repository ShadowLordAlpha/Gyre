#pragma once
// Finite-difference gradient checking for tests. Game/model agnostic.
//
// Usage: build a closure that (1) zeroes grads, (2) runs forward, (3) computes a scalar
// loss, (4) if `backward` is true runs backward so Param::grad holds dL/dparam.
// Inputs whose gradient you want checked must be wrapped as Param too (copy ctx.dx
// into their .grad inside the closure).
#include "gyre/module.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace gyre::testing {

inline Tensor make_t(std::vector<std::int64_t> shape, const std::vector<float>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const float>(v)), shape, DType::f32, *d);
  if (!t) std::abort();
  return std::move(*t);
}

inline Tensor rand_t(std::vector<std::int64_t> shape, Rng& rng, float sd = 1.f) {
  std::int64_t n = 1;
  for (auto s : shape) n *= s;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (auto& x : v) x = rng.normal(0.f, sd);
  return make_t(std::move(shape), v);
}

inline Param make_p(Tensor v) {
  auto g = Tensor::zeros(v.shape(), DType::f32, v.device());
  if (!g) std::abort();
  return Param{std::move(v), std::move(*g)};
}

inline void set_grad(Param& p, const Tensor& g) {
  auto dst = p.grad.host_span<float>();
  auto src = g.host_span<float>();
  ASSERT_TRUE(dst && src);
  ASSERT_EQ(dst->size(), src->size());
  std::copy(src->begin(), src->end(), dst->begin());
}

struct GradCheckOpts {
  double h = 1e-2;  // f32 forward: too small a step drowns in rounding noise
  double rtol = 3e-2;
  double atol = 3e-4;
  std::size_t max_per_param = 24;  // elements sampled per tensor (all if fewer)
  std::uint64_t seed = 7;
};

// Returns the number of mismatching elements (0 = pass); failures are also reported
// through gtest with the parameter index and element.
inline int gradcheck(std::span<Param> params,
                     const std::function<Result<double>(bool backward)>& fn,
                     GradCheckOpts o = {}) {
  auto base = fn(true);
  EXPECT_TRUE(base) << (base ? "" : base.error().message);
  if (!base) return 1;
  std::vector<std::vector<float>> analytic;
  for (auto& p : params) {
    auto g = p.grad.host_span<float>();
    analytic.emplace_back(g->begin(), g->end());
  }
  Rng rng(o.seed);
  int bad = 0;
  for (std::size_t pi = 0; pi < params.size(); ++pi) {
    auto v = params[pi].value.host_span<float>();
    const std::size_t n = v->size();
    std::vector<std::size_t> idx;
    if (n <= o.max_per_param) {
      for (std::size_t i = 0; i < n; ++i) idx.push_back(i);
    } else {
      for (std::size_t k = 0; k < o.max_per_param; ++k)
        idx.push_back(rng.u32(static_cast<std::uint32_t>(n)));
    }
    for (auto i : idx) {
      const float orig = (*v)[i];
      (*v)[i] = orig + static_cast<float>(o.h);
      auto lp = fn(false);
      (*v)[i] = orig - static_cast<float>(o.h);
      auto lm = fn(false);
      (*v)[i] = orig;
      if (!lp || !lm) {
        ADD_FAILURE() << "loss fn failed";
        return bad + 1;
      }
      const double num = (*lp - *lm) / (2.0 * o.h);
      const double ana = analytic[pi][i];
      const double tol = o.atol + o.rtol * std::max(std::abs(num), std::abs(ana));
      if (std::abs(num - ana) > tol) {
        ++bad;
        ADD_FAILURE() << "param " << pi << " elem " << i << ": analytic " << ana << " numeric "
                      << num;
      }
    }
  }
  return bad;
}

}  // namespace gyre::testing
