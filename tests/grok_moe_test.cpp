#include "gyre/nn/moe.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace {

gyre::Result<gyre::Tensor> zeros2(std::int64_t r, std::int64_t c, std::shared_ptr<gyre::Device> d) {
  std::int64_t sh[] = {r, c};
  return gyre::Tensor::zeros(sh, gyre::DType::f32, d);
}

void fill_const(gyre::Tensor& t, float v) {
  auto p = t.host_span<float>();
  for (auto& x : *p) x = v;
}

}  // namespace

TEST(GrokMoe, TopKHandLogits) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 4};
  float z[] = {1.f, 3.f, 0.f, 2.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(z)), sh, gyre::DType::f32, *d);
  auto top = gyre::moe_topk(*t, 2, 0.f);
  ASSERT_TRUE(top) << top.error().message;
  auto pi = top->indices.host_span<std::int32_t>();
  EXPECT_EQ((*pi)[0], 1);
  EXPECT_EQ((*pi)[1], 3);
  auto pw = top->weights.host_span<float>();
  EXPECT_NEAR((*pw)[0] + (*pw)[1], 1.f, 1e-5);
  EXPECT_GT((*pw)[0], (*pw)[1]);
}

TEST(GrokMoe, TopKRejectsBadK) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 2};
  auto t = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  auto top = gyre::moe_topk(*t, 3, 0.f);
  ASSERT_FALSE(top);
}

TEST(GrokMoe, Expert0OnlyMatchesDensePlusExpert) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 2, Dim = 4, I = 8, E = 2, k = 1;
  auto x = zeros2(N, Dim, *d);
  fill_const(*x, 0.2f);
  auto dw1 = zeros2(Dim, I, *d);
  auto dw3 = zeros2(Dim, I, *d);
  auto dw2 = zeros2(I, Dim, *d);
  fill_const(*dw1, 0.05f);
  fill_const(*dw3, 0.04f);
  fill_const(*dw2, 0.03f);
  gyre::SwiGLUWeights dense{&*dw1, &*dw3, &*dw2};

  auto e0w1 = zeros2(Dim, I, *d);
  auto e0w3 = zeros2(Dim, I, *d);
  auto e0w2 = zeros2(I, Dim, *d);
  fill_const(*e0w1, 0.1f);
  fill_const(*e0w3, 0.1f);
  fill_const(*e0w2, 0.1f);
  auto z1 = zeros2(Dim, I, *d);
  auto z3 = zeros2(Dim, I, *d);
  auto z2 = zeros2(I, Dim, *d);
  gyre::SwiGLUWeights exp0{&*e0w1, &*e0w3, &*e0w2};
  gyre::SwiGLUWeights exp1{&*z1, &*z3, &*z2};
  gyre::SwiGLUWeights experts[] = {exp0, exp1};

  auto rw = zeros2(Dim, E, *d);
  fill_const(*rw, 0.f);
  auto pr = rw->host_span<float>();
  for (std::int64_t i = 0; i < Dim; ++i) (*pr)[static_cast<std::size_t>(i * E + 0)] = 5.f;

  auto y = gyre::residual_moe(*x, dense, *rw, experts, k, 30.f);
  ASSERT_TRUE(y) << y.error().message;
  auto dens = gyre::swiglu_ffn(*x, dense);
  auto e0 = gyre::swiglu_ffn(*x, exp0);
  ASSERT_TRUE(dens);
  ASSERT_TRUE(e0);
  auto py = y->host_span<float>();
  auto pd = dens->host_span<float>();
  auto pe = e0->host_span<float>();
  for (std::size_t i = 0; i < py->size(); ++i) {
    EXPECT_NEAR((*py)[i], (*pd)[i] + (*pe)[i], 1e-4f);
  }
}

TEST(GrokMoe, ResidualAblationChangesOutput) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 1, Dim = 4, I = 4, E = 2;
  auto x = zeros2(N, Dim, *d);
  fill_const(*x, 0.3f);
  auto dw1 = zeros2(Dim, I, *d);
  auto dw3 = zeros2(Dim, I, *d);
  auto dw2 = zeros2(I, Dim, *d);
  fill_const(*dw1, 0.2f);
  fill_const(*dw3, 0.2f);
  fill_const(*dw2, 0.2f);
  auto z1 = zeros2(Dim, I, *d);
  auto z3 = zeros2(Dim, I, *d);
  auto z2 = zeros2(I, Dim, *d);
  gyre::SwiGLUWeights dense{&*dw1, &*dw3, &*dw2};
  gyre::SwiGLUWeights zexp{&*z1, &*z3, &*z2};
  gyre::SwiGLUWeights experts[] = {zexp, zexp};
  auto rw = zeros2(Dim, E, *d);
  auto y = gyre::residual_moe(*x, dense, *rw, experts, 2, 0.f);
  auto dens = gyre::swiglu_ffn(*x, dense);
  ASSERT_TRUE(y);
  ASSERT_TRUE(dens);
  auto py = y->host_span<float>();
  auto pd = dens->host_span<float>();
  float d2 = 0;
  for (std::size_t i = 0; i < py->size(); ++i) {
    float e = (*py)[i] - (*pd)[i];
    d2 += e * e;
  }
  EXPECT_NEAR(d2, 0.f, 1e-8f);
  fill_const(*dw1, 0.f);
  fill_const(*dw3, 0.f);
  fill_const(*dw2, 0.f);
  auto y2 = gyre::residual_moe(*x, dense, *rw, experts, 2, 0.f);
  ASSERT_TRUE(y2);
  auto p2 = y2->host_span<float>();
  float s = 0;
  for (auto v : *p2) s += v * v;
  EXPECT_NEAR(s, 0.f, 1e-8f);
}

TEST(GrokMoe, TwoTokensDifferentExperts) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 2, Dim = 2, I = 2, E = 2, k = 1;
  float xv[] = {1.f, 0.f, 0.f, 1.f};
  std::int64_t xsh[] = {N, Dim};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(xv)), xsh, gyre::DType::f32, *d);
  auto z1 = zeros2(Dim, I, *d);
  auto z3 = zeros2(Dim, I, *d);
  auto z2 = zeros2(I, Dim, *d);
  auto e0w1 = zeros2(Dim, I, *d);
  auto e0w3 = zeros2(Dim, I, *d);
  auto e0w2 = zeros2(I, Dim, *d);
  auto e1w1 = zeros2(Dim, I, *d);
  auto e1w3 = zeros2(Dim, I, *d);
  auto e1w2 = zeros2(I, Dim, *d);
  fill_const(*e0w1, 1.f);
  fill_const(*e0w3, 1.f);
  fill_const(*e0w2, 1.f);
  fill_const(*e1w1, 0.5f);
  fill_const(*e1w3, 0.5f);
  fill_const(*e1w2, 0.5f);
  gyre::SwiGLUWeights dense{&*z1, &*z3, &*z2};
  gyre::SwiGLUWeights exp0{&*e0w1, &*e0w3, &*e0w2};
  gyre::SwiGLUWeights exp1{&*e1w1, &*e1w3, &*e1w2};
  gyre::SwiGLUWeights experts[] = {exp0, exp1};
  auto rw = zeros2(Dim, E, *d);
  auto pr = rw->host_span<float>();
  (*pr)[0] = 10.f;
  (*pr)[1] = 0.f;
  (*pr)[2] = 0.f;
  (*pr)[3] = 10.f;
  auto y = gyre::residual_moe(*x, dense, *rw, experts, k, 0.f);
  ASSERT_TRUE(y) << y.error().message;
  auto e0 = gyre::swiglu_ffn(*x, exp0);
  auto e1 = gyre::swiglu_ffn(*x, exp1);
  ASSERT_TRUE(e0);
  ASSERT_TRUE(e1);
  auto py = y->host_span<float>();
  auto p0 = e0->host_span<float>();
  auto p1 = e1->host_span<float>();
  EXPECT_NEAR((*py)[0], (*p0)[0], 1e-4f);
  EXPECT_NEAR((*py)[1], (*p0)[1], 1e-4f);
  EXPECT_NEAR((*py)[2], (*p1)[2], 1e-4f);
  EXPECT_NEAR((*py)[3], (*p1)[3], 1e-4f);
}

namespace {

gyre::Tensor randn2(std::int64_t r, std::int64_t c, std::uint64_t seed, std::shared_ptr<gyre::Device> d) {
  gyre::Rng rng(seed);
  auto t = zeros2(r, c, d);
  auto p = t->host_span<float>();
  for (auto& x : *p) x = rng.normal(0.f, 0.5f);
  return std::move(*t);
}

}  // namespace

// Grouped-by-expert compute must equal the per-token definition:
// y[n] = dense(x[n]) + sum_j w[n,j] * expert_{idx[n,j]}(x[n]).
TEST(GrokMoe, ManyTokensMatchPerTokenReference) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 9, Dim = 6, I = 5, E = 3;
  const int k = 2;
  auto x = randn2(N, Dim, 1, *d);
  std::vector<gyre::Tensor> w;
  for (std::uint64_t s = 0; s < 3 * (E + 1); ++s) {
    const bool down = (s % 3) == 2;
    w.push_back(randn2(down ? I : Dim, down ? Dim : I, 100 + s, *d));
  }
  gyre::SwiGLUWeights dense{&w[0], &w[1], &w[2]};
  std::vector<gyre::SwiGLUWeights> ex;
  for (std::int64_t e = 0; e < E; ++e) ex.push_back({&w[3 + 3 * e], &w[4 + 3 * e], &w[5 + 3 * e]});
  auto rw = randn2(Dim, E, 50, *d);
  auto y = gyre::residual_moe(x, dense, rw, ex, k, 0.f);
  ASSERT_TRUE(y) << y.error().message;
  auto y_sparse = gyre::residual_moe(x, gyre::SwiGLUWeights{}, rw, ex, k, 0.f);
  ASSERT_TRUE(y_sparse) << y_sparse.error().message;

  auto logits = gyre::matmul(x, rw);
  auto top = gyre::moe_topk(*logits, k, 0.f);
  auto dens = gyre::swiglu_ffn(x, dense);
  ASSERT_TRUE(top && dens);
  auto pi = top->indices.host_span<std::int32_t>();
  auto pw = top->weights.host_span<float>();
  auto px = x.host_span<float>();
  auto pd = dens->host_span<float>();
  auto py = y->host_span<float>();
  auto ps = y_sparse->host_span<float>();
  for (std::int64_t n = 0; n < N; ++n) {
    std::vector<float> ref(static_cast<std::size_t>(Dim), 0.f);
    std::int64_t one[] = {1, Dim};
    auto tok = gyre::Tensor::from_host(std::as_bytes(std::span(px->data() + n * Dim, Dim)), one,
                                       gyre::DType::f32, *d);
    for (int j = 0; j < k; ++j) {
      auto e = (*pi)[static_cast<std::size_t>(n * k + j)];
      auto ye = gyre::swiglu_ffn(*tok, ex[static_cast<std::size_t>(e)]);
      ASSERT_TRUE(ye);
      auto pe = ye->host_span<float>();
      for (std::int64_t c = 0; c < Dim; ++c)
        ref[static_cast<std::size_t>(c)] += (*pw)[static_cast<std::size_t>(n * k + j)] * (*pe)[static_cast<std::size_t>(c)];
    }
    for (std::int64_t c = 0; c < Dim; ++c) {
      const auto i = static_cast<std::size_t>(n * Dim + c);
      EXPECT_NEAR((*ps)[i], ref[static_cast<std::size_t>(c)], 1e-5f);
      EXPECT_NEAR((*py)[i], (*pd)[i] + ref[static_cast<std::size_t>(c)], 1e-5f);
    }
  }
}

// matmul only checks inner dims, so a dense or expert w2 narrower than D used to be read past its
// end (memcpy of N*D floats / D floats per expert row). Now an invalid_shape error.
TEST(GrokMoe, NarrowOutputWidthIsError) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 3, Dim = 4, I = 5;
  auto x = randn2(N, Dim, 2, *d);
  auto w1 = randn2(Dim, I, 3, *d), w3 = randn2(Dim, I, 4, *d), w2 = randn2(I, Dim, 5, *d);
  auto narrow = randn2(I, Dim - 1, 6, *d);
  auto rw = randn2(Dim, 2, 7, *d);
  gyre::SwiGLUWeights good{&w1, &w3, &w2};
  gyre::SwiGLUWeights bad{&w1, &w3, &narrow};
  gyre::SwiGLUWeights ok_ex[] = {good, good};
  gyre::SwiGLUWeights bad_ex[] = {bad, bad};
  ASSERT_TRUE(gyre::residual_moe(x, good, rw, ok_ex, 1, 0.f));
  auto r1 = gyre::residual_moe(x, bad, rw, ok_ex, 1, 0.f);
  ASSERT_FALSE(r1);
  EXPECT_EQ(r1.error().code, gyre::Errc::invalid_shape);
  auto r2 = gyre::residual_moe(x, good, rw, bad_ex, 1, 0.f);
  ASSERT_FALSE(r2);
  EXPECT_EQ(r2.error().code, gyre::Errc::invalid_shape);
}
