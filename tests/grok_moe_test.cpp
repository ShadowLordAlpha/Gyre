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

void fill(gyre::Tensor& t, float v) {
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
  fill(*x, 0.2f);
  auto dw1 = zeros2(Dim, I, *d);
  auto dw3 = zeros2(Dim, I, *d);
  auto dw2 = zeros2(I, Dim, *d);
  fill(*dw1, 0.05f);
  fill(*dw3, 0.04f);
  fill(*dw2, 0.03f);
  gyre::SwiGLUWeights dense{&*dw1, &*dw3, &*dw2};

  auto e0w1 = zeros2(Dim, I, *d);
  auto e0w3 = zeros2(Dim, I, *d);
  auto e0w2 = zeros2(I, Dim, *d);
  fill(*e0w1, 0.1f);
  fill(*e0w3, 0.1f);
  fill(*e0w2, 0.1f);
  auto z1 = zeros2(Dim, I, *d);
  auto z3 = zeros2(Dim, I, *d);
  auto z2 = zeros2(I, Dim, *d);
  gyre::SwiGLUWeights exp0{&*e0w1, &*e0w3, &*e0w2};
  gyre::SwiGLUWeights exp1{&*z1, &*z3, &*z2};
  gyre::SwiGLUWeights experts[] = {exp0, exp1};

  auto rw = zeros2(Dim, E, *d);
  fill(*rw, 0.f);
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
  fill(*x, 0.3f);
  auto dw1 = zeros2(Dim, I, *d);
  auto dw3 = zeros2(Dim, I, *d);
  auto dw2 = zeros2(I, Dim, *d);
  fill(*dw1, 0.2f);
  fill(*dw3, 0.2f);
  fill(*dw2, 0.2f);
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
  fill(*dw1, 0.f);
  fill(*dw3, 0.f);
  fill(*dw2, 0.f);
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
  fill(*e0w1, 1.f);
  fill(*e0w3, 1.f);
  fill(*e0w2, 1.f);
  fill(*e1w1, 0.5f);
  fill(*e1w3, 0.5f);
  fill(*e1w2, 0.5f);
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
