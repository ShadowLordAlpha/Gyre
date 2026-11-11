#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/ops.hpp"

#include <gtest/gtest.h>

using namespace gyre;
using namespace gyre::testing;

namespace {

double dot_all(const Tensor& y, const Tensor& c) {
  auto a = y.host_span<float>();
  auto b = c.host_span<float>();
  double s = 0;
  for (std::size_t i = 0; i < a->size(); ++i) s += double((*a)[i]) * double((*b)[i]);
  return s;
}

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}

Tensor mk_i32(std::vector<std::int64_t> shape, const std::vector<std::int32_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(v)), shape, DType::i32, *d);
  if (!t) std::abort();
  return std::move(*t);
}

// Generic elementwise check: loss = <f(x), c>, grad = bwd(x, y, c).
template <class Fwd, class Bwd>
void check_elementwise(Fwd fwd, Bwd bwd, Tensor x0, std::uint64_t seed) {
  Rng rng(seed);
  auto x = make_p(x0);
  auto y0 = fwd(x0);
  ASSERT_TRUE(y0);
  auto c = rand_t(std::vector<std::int64_t>(y0->shape().begin(), y0->shape().end()), rng);
  std::vector<Param> ps{x};
  auto fn = [&](bool backward) -> Result<double> {
    auto y = fwd(ps[0].value);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto dx = bwd(ps[0].value, *y, c);
      if (!dx) return std::unexpected(dx.error());
      set_grad(ps[0], *dx);
    }
    return dot_all(*y, c);
  };
  GradCheckOpts o;
  o.h = 1e-3;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
}

}  // namespace

TEST(RlOps, Sigmoid) {
  Rng r(1);
  check_elementwise([](const Tensor& x) { return sigmoid(x); },
                    [](const Tensor&, const Tensor& y, const Tensor& dy) { return sigmoid_backward(y, dy); },
                    rand_t({3, 4}, r), 1);
}

TEST(RlOps, Tanh) {
  Rng r(2);
  check_elementwise([](const Tensor& x) { return tanh_act(x); },
                    [](const Tensor&, const Tensor& y, const Tensor& dy) { return tanh_backward(y, dy); },
                    rand_t({3, 4}, r), 2);
}

TEST(RlOps, Relu) {
  // Keep samples away from the kink so the finite difference is valid.
  std::vector<float> v;
  for (int i = 0; i < 12; ++i) v.push_back((i % 2 ? 1.f : -1.f) * (0.3f + 0.1f * float(i)));
  check_elementwise([](const Tensor& x) { return relu(x); },
                    [](const Tensor& x, const Tensor&, const Tensor& dy) { return relu_backward(x, dy); },
                    make_t({3, 4}, v), 3);
}

TEST(RlOps, Clamp) {
  std::vector<float> v = {-2.f, -0.5f, 0.2f, 0.7f, 1.6f, 3.f};
  check_elementwise([](const Tensor& x) { return clamp(x, -1.f, 1.f); },
                    [](const Tensor& x, const Tensor&, const Tensor& dy) {
                      return clamp_backward(x, dy, -1.f, 1.f);
                    },
                    make_t({2, 3}, v), 4);
  auto y = clamp(make_t({2, 3}, v), -1.f, 1.f);
  auto p = y->host_span<float>();
  EXPECT_EQ((*p)[0], -1.f);
  EXPECT_EQ((*p)[5], 1.f);
}

TEST(RlOps, Sub) {
  auto y = sub(make_t({2}, {5.f, 1.f}), make_t({2}, {2.f, 4.f}));
  ASSERT_TRUE(y);
  auto p = y->host_span<float>();
  EXPECT_EQ((*p)[0], 3.f);
  EXPECT_EQ((*p)[1], -3.f);
  EXPECT_FALSE(sub(make_t({2}, {1, 2}), make_t({3}, {1, 2, 3})));
}

TEST(RlOps, LogSoftmax) {
  Rng r(5);
  check_elementwise([](const Tensor& x) { return log_softmax_last(x); },
                    [](const Tensor&, const Tensor& y, const Tensor& dy) {
                      return log_softmax_last_backward(y, dy);
                    },
                    rand_t({2, 3, 5}, r), 5);
  // exp(logp) sums to 1; stable for large logits
  auto y = log_softmax_last(make_t({1, 3}, {1000.f, 1001.f, 999.f}));
  ASSERT_TRUE(y);
  auto p = y->host_span<float>();
  double s = 0;
  for (float v : *p) s += std::exp(v);
  EXPECT_NEAR(s, 1.0, 1e-5);
}

TEST(RlOps, Gather) {
  Rng r(6);
  auto idx = mk_i32({2, 3}, {0, 4, 2, 1, 3, 3});
  check_elementwise([&](const Tensor& x) { return gather_last(x, idx); },
                    [&](const Tensor& x, const Tensor&, const Tensor& dy) {
                      return gather_last_backward(dy, idx, x.shape()[2]);
                    },
                    rand_t({2, 3, 5}, r), 6);
  EXPECT_FALSE(gather_last(rand_t({1, 3}, r), mk_i32({1}, {3})));  // OOB
  EXPECT_FALSE(gather_last(rand_t({1, 3}, r), mk_i32({1}, {-1})));
}

TEST(RlOps, MaskLogitsThenLogSoftmaxGather) {
  Rng r(7);
  auto mask = mk_u8({2, 4}, {1, 0, 1, 1, 0, 1, 1, 0});
  auto idx = mk_i32({2}, {2, 6 - 5});  // legal actions only: row0 -> 2, row1 -> 1
  auto x = make_p(rand_t({2, 4}, r));
  auto c = make_t({2}, {0.7f, -1.3f});
  std::vector<Param> ps{x};
  auto fn = [&](bool backward) -> Result<double> {
    auto ml = mask_logits(ps[0].value, mask);
    if (!ml) return std::unexpected(ml.error());
    auto lp = log_softmax_last(*ml);
    if (!lp) return std::unexpected(lp.error());
    auto g = gather_last(*lp, idx);
    if (!g) return std::unexpected(g.error());
    if (backward) {
      auto dlp = gather_last_backward(c, idx, 4);
      auto dml = log_softmax_last_backward(*lp, *dlp);
      auto dx = mask_logits_backward(*dml, mask);
      if (!dx) return std::unexpected(dx.error());
      set_grad(ps[0], *dx);
    }
    return dot_all(*g, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
  // Illegal actions get zero probability and exactly zero gradient.
  auto g = ps[0].grad.host_span<float>();
  EXPECT_EQ((*g)[1], 0.f);
  EXPECT_EQ((*g)[4], 0.f);
  EXPECT_EQ((*g)[7], 0.f);
  auto ml = mask_logits(ps[0].value, mask);
  auto lp = log_softmax_last(*ml);
  auto p = lp->host_span<float>();
  EXPECT_LT((*p)[1], -1e8f);
}

TEST(RlOps, MaskedMeanMaxPool) {
  Rng r(8);
  auto mask = mk_u8({2, 5}, {1, 1, 0, 1, 0, 0, 1, 0, 0, 0});
  auto x = make_p(rand_t({2, 5, 3}, r));
  auto c = rand_t({2, 6}, r);
  Tensor am = x.value;
  std::vector<Param> ps{x};
  auto fn = [&](bool backward) -> Result<double> {
    auto y = masked_mean_max_pool(ps[0].value, mask, am);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto dx = masked_mean_max_pool_backward(c, mask, am, 5);
      if (!dx) return std::unexpected(dx.error());
      set_grad(ps[0], *dx);
    }
    return dot_all(*y, c);
  };
  GradCheckOpts o;
  o.h = 2e-3;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
  // Padded slots get exactly zero gradient.
  auto g = ps[0].grad.host_span<float>();
  for (int b = 0; b < 2; ++b)
    for (int n = 0; n < 5; ++n) {
      const auto mk = mask.host_span<std::uint8_t>();
      if (!(*mk)[b * 5 + n])
        for (int d = 0; d < 3; ++d) EXPECT_EQ((*g)[(b * 5 + n) * 3 + d], 0.f);
    }
}

TEST(RlOps, PoolEmptySetIsZeroAndFinite) {
  Rng r(9);
  auto mask = mk_u8({1, 4}, {0, 0, 0, 0});
  Tensor am = rand_t({1, 4, 2}, r);
  auto y = masked_mean_max_pool(rand_t({1, 4, 2}, r), mask, am);
  ASSERT_TRUE(y);
  for (float v : *y->host_span<float>()) EXPECT_EQ(v, 0.f);
  auto dx = masked_mean_max_pool_backward(rand_t({1, 4}, r), mask, am, 4);
  ASSERT_TRUE(dx);
  for (float v : *dx->host_span<float>()) EXPECT_EQ(v, 0.f);
}

TEST(RlOps, KeyMask) {
  Rng r(10);
  auto s = rand_t({1, 2, 3, 3}, r);
  auto mask = mk_u8({1, 3}, {1, 0, 1});
  ASSERT_TRUE(key_mask_(s, mask));
  auto w = softmax_last(s);
  auto p = w->host_span<float>();
  for (int row = 0; row < 6; ++row) {
    EXPECT_EQ((*p)[row * 3 + 1], 0.f);
    EXPECT_NEAR((*p)[row * 3] + (*p)[row * 3 + 2], 1.f, 1e-6);
  }
}

TEST(RlOps, ConcatSplitRoundTrip) {
  Rng r(11);
  auto a = rand_t({3, 2}, r), b = rand_t({3, 4}, r);
  auto c = concat_last(a, b);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->shape()[1], 6);
  Tensor da = a, db = b;
  ASSERT_TRUE(split_last(*c, 2, da, db));
  EXPECT_EQ(da.shape()[1], 2);
  EXPECT_EQ(db.shape()[1], 4);
  auto pa = a.host_span<float>(), pda = da.host_span<float>();
  auto pb = b.host_span<float>(), pdb = db.host_span<float>();
  for (std::size_t i = 0; i < pa->size(); ++i) EXPECT_EQ((*pa)[i], (*pda)[i]);
  for (std::size_t i = 0; i < pb->size(); ++i) EXPECT_EQ((*pb)[i], (*pdb)[i]);
}
