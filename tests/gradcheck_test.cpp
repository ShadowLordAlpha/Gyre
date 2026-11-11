#include "gradcheck.hpp"

#include "gyre/nn/attention.hpp"
#include "gyre/nn/layers.hpp"
#include "gyre/ops.hpp"

#include <cstring>
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

// Checks module params and its input gradient with loss = <y, c>.
void check_module(Module& m, Tensor x, std::vector<std::int64_t> yshape, Rng& rng) {
  std::vector<Param> ps;
  for (auto& p : m.parameters()) ps.push_back(Param{p.value, p.grad});
  ps.push_back(make_p(x));
  auto c = rand_t(std::move(yshape), rng);
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    ForwardCtx ctx;
    ctx.train = true;
    auto y = m.forward(ps.back().value, ctx);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto r = m.backward(c, ctx);
      if (!r) return std::unexpected(r.error());
      set_grad(ps.back(), *ctx.dx);
    }
    return dot_all(*y, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

}  // namespace

TEST(GradCheck, Linear) {
  Rng rng(1);
  auto d = Device::cpu();
  auto lin = Linear::create(5, 4, *d, rng, 0.5f);
  ASSERT_TRUE(lin);
  check_module(*lin, rand_t({3, 5}, rng), {3, 4}, rng);
}

TEST(GradCheck, LinearRank3) {
  Rng rng(2);
  auto d = Device::cpu();
  auto lin = Linear::create(4, 3, *d, rng, 0.5f);
  ASSERT_TRUE(lin);
  check_module(*lin, rand_t({2, 3, 4}, rng), {2, 3, 3}, rng);
}

TEST(GradCheck, LayerNorm) {
  Rng rng(3);
  auto d = Device::cpu();
  auto ln = LayerNorm::create(6, *d);
  ASSERT_TRUE(ln);
  for (auto& p : ln->parameters()) {  // non-trivial scale/shift so gw/gb are exercised
    auto v = p.value.host_span<float>();
    for (auto& e : *v) e += rng.normal(0.f, 0.3f);
  }
  check_module(*ln, rand_t({4, 6}, rng), {4, 6}, rng);
}

TEST(GradCheck, CausalAttention) {
  Rng rng(4);
  auto d = Device::cpu();
  auto at = CausalSelfAttention::create(8, 2, *d, rng, 1.f, true, 0.f);
  ASSERT_TRUE(at);
  for (auto& p : at->parameters()) {
    auto v = p.value.host_span<float>();
    for (auto& e : *v) e = rng.normal(0.f, 0.4f);
  }
  check_module(*at, rand_t({2, 4, 8}, rng), {2, 4, 8}, rng);
}

TEST(GradCheck, SoftmaxCrossEntropy) {
  Rng rng(5);
  const std::int64_t B = 2, T = 3, V = 5;
  auto logits = make_p(rand_t({B, T, V}, rng));
  std::vector<std::int32_t> tg = {0, 4, 2, 1, 3, 3};
  auto d = Device::cpu();
  std::int64_t tsh[] = {B, T};
  auto targets = Tensor::from_host(std::as_bytes(std::span(tg)), tsh, DType::i32, *d);
  ASSERT_TRUE(targets);
  std::vector<Param> ps{logits};
  auto fn = [&](bool backward) -> Result<double> {
    auto lp = softmax_cross_entropy(ps[0].value, *targets);
    if (!lp) return std::unexpected(lp.error());
    if (backward) set_grad(ps[0], lp->d_pred);
    auto v = lp->value.item_f32();
    if (!v) return std::unexpected(v.error());
    return double(*v);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

TEST(GradCheck, GeluAndSoftmaxBackward) {
  Rng rng(6);
  auto x = make_p(rand_t({3, 5}, rng));
  auto c = rand_t({3, 5}, rng);
  std::vector<Param> ps{x};
  auto fn = [&](bool backward) -> Result<double> {
    auto g = gelu(ps[0].value);
    if (!g) return std::unexpected(g.error());
    auto s = softmax_last(*g);
    if (!s) return std::unexpected(s.error());
    if (backward) {
      auto ds = softmax_last_backward(*s, c);
      if (!ds) return std::unexpected(ds.error());
      auto dx = gelu_backward(ps[0].value, *ds);
      if (!dx) return std::unexpected(dx.error());
      set_grad(ps[0], *dx);
    }
    return dot_all(*s, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

TEST(Determinism, RngSeeded) {
  Rng a(42), b(42);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a.normal(0, 1), b.normal(0, 1));
    EXPECT_EQ(a.uniform01(), b.uniform01());
    EXPECT_EQ(a.u32(17), b.u32(17));
  }
  Rng c(1);
  for (int i = 0; i < 1000; ++i) {
    float u = c.uniform01();
    EXPECT_GE(u, 0.f);
    EXPECT_LT(u, 1.f);
    EXPECT_LT(c.u32(7), 7u);
  }
  // mt19937_64 output is fixed by the standard: pins the stream across platforms.
  Rng g(5489);
  EXPECT_EQ(g.u64(), 14514284786278117030ull);
}

TEST(Determinism, RngMoments) {
  Rng r(9);
  double s = 0, s2 = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    double x = r.normal(0.f, 1.f);
    s += x;
    s2 += x * x;
  }
  EXPECT_NEAR(s / n, 0.0, 0.03);
  EXPECT_NEAR(s2 / n, 1.0, 0.05);
}

TEST(Determinism, DeterministicModeRepeatable) {
  set_deterministic(true);
  EXPECT_TRUE(deterministic());
  Rng rng(3);
  auto d = Device::cpu();
  auto lin = Linear::create(64, 64, *d, rng, 0.1f);
  ASSERT_TRUE(lin);
  auto x = rand_t({128, 64}, rng);
  ForwardCtx c1, c2;
  auto y1 = lin->forward(x, c1);
  auto y2 = lin->forward(x, c2);
  ASSERT_TRUE(y1 && y2);
  auto a = y1->host_span<float>();
  auto b = y2->host_span<float>();
  EXPECT_EQ(0, std::memcmp(a->data(), b->data(), a->size() * sizeof(float)));
}
