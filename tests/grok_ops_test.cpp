#include "gyre/ops.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(GrokOps, RmsNormFormula) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {2, 4};
  float x[] = {1.f, 2.f, 3.f, 4.f, 0.f, 1.f, 0.f, -1.f};
  float w[] = {1.f, 1.f, 1.f, 1.f};
  std::int64_t ws[] = {4};
  auto xt = gyre::Tensor::from_host(std::as_bytes(std::span(x)), sh, gyre::DType::f32, *d);
  auto wt = gyre::Tensor::from_host(std::as_bytes(std::span(w)), ws, gyre::DType::f32, *d);
  constexpr float eps = 1e-5f;
  auto y = gyre::rms_norm(*xt, *wt, eps);
  ASSERT_TRUE(y) << y.error().message;
  auto p = y->host_span<float>();
  float ms = (1 + 4 + 9 + 16) / 4.f;
  float inv = 1.f / std::sqrt(ms + eps);
  EXPECT_NEAR((*p)[0], 1.f * inv, 1e-5);
  EXPECT_NEAR((*p)[3], 4.f * inv, 1e-5);
}

TEST(GrokOps, RmsNormBadShape) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {2, 4};
  std::int64_t ws[] = {3};
  auto x = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  auto w = gyre::Tensor::zeros(ws, gyre::DType::f32, *d);
  auto y = gyre::rms_norm(*x, *w, 1e-5f);
  ASSERT_FALSE(y);
  EXPECT_EQ(y.error().code, gyre::Errc::invalid_shape);
}

TEST(GrokOps, SiluAndSwiglu) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {3};
  float g[] = {0.f, 1.f, -1.f};
  float u[] = {2.f, 3.f, 4.f};
  auto gt = gyre::Tensor::from_host(std::as_bytes(std::span(g)), sh, gyre::DType::f32, *d);
  auto ut = gyre::Tensor::from_host(std::as_bytes(std::span(u)), sh, gyre::DType::f32, *d);
  auto s = gyre::silu(*gt);
  ASSERT_TRUE(s);
  auto sp = s->host_span<float>();
  EXPECT_NEAR((*sp)[0], 0.f, 1e-6);
  EXPECT_NEAR((*sp)[1], 1.f / (1.f + std::exp(-1.f)), 1e-5);
  auto sw = gyre::swiglu(*gt, *ut);
  ASSERT_TRUE(sw);
  auto p = sw->host_span<float>();
  EXPECT_NEAR((*p)[1], (*sp)[1] * 3.f, 1e-5);
}

TEST(GrokOps, Softcap) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {3};
  float v[] = {0.f, 100.f, -100.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *d);
  auto y = gyre::softcap(*t, 30.f);
  ASSERT_TRUE(y);
  auto p = y->host_span<float>();
  EXPECT_NEAR((*p)[0], 0.f, 1e-6);
  EXPECT_LT(std::fabs((*p)[1]), 30.f);
  EXPECT_GT((*p)[1], 29.f);
  EXPECT_LT((*p)[2], 0.f);
  auto bad = gyre::softcap(*t, 0.f);
  ASSERT_FALSE(bad);
}

TEST(GrokOps, RopePairAndInverse) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 2, 2};  // T=2, D=2
  float x[] = {1.f, 0.f, 1.f, 0.f};
  std::int32_t pos[] = {0, 1};
  std::int64_t psh[] = {2};
  auto xt = gyre::Tensor::from_host(std::as_bytes(std::span(x)), sh, gyre::DType::f32, *d);
  auto pt = gyre::Tensor::from_host(std::as_bytes(std::span(pos)), psh, gyre::DType::i32, *d);
  auto y = gyre::rope(*xt, *pt, 10000.f, 1.f);
  ASSERT_TRUE(y) << y.error().message;
  auto p = y->host_span<float>();
  EXPECT_NEAR((*p)[0], 1.f, 1e-5);  // pos 0
  EXPECT_NEAR((*p)[1], 0.f, 1e-5);
  EXPECT_NE((*p)[2], (*p)[0]);
  // Inverse: rotate with -pos
  std::int32_t npos[] = {0, -1};
  auto npt = gyre::Tensor::from_host(std::as_bytes(std::span(npos)), psh, gyre::DType::i32, *d);
  auto back = gyre::rope(*y, *npt, 10000.f, 1.f);
  ASSERT_TRUE(back);
  auto b = back->host_span<float>();
  EXPECT_NEAR((*b)[2], 1.f, 1e-4);
  EXPECT_NEAR((*b)[3], 0.f, 1e-4);
}

TEST(GrokOps, RopeScaleChangesAngle) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 1, 2};
  float x[] = {1.f, 0.f};
  std::int32_t pos[] = {16};
  std::int64_t psh[] = {1};
  auto xt = gyre::Tensor::from_host(std::as_bytes(std::span(x)), sh, gyre::DType::f32, *d);
  auto pt = gyre::Tensor::from_host(std::as_bytes(std::span(pos)), psh, gyre::DType::i32, *d);
  auto a = gyre::rope(*xt, *pt, 10000.f, 1.f);
  auto b = gyre::rope(*xt, *pt, 10000.f, 16.f);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  auto pa = a->host_span<float>();
  auto pb = b->host_span<float>();
  EXPECT_NE((*pa)[0], (*pb)[0]);
}

TEST(GrokOps, RopeOddDimRejected) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 1, 3};
  std::int32_t pos[] = {0};
  std::int64_t psh[] = {1};
  auto x = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  auto p = gyre::Tensor::from_host(std::as_bytes(std::span(pos)), psh, gyre::DType::i32, *d);
  auto y = gyre::rope(*x, *p, 10000.f, 1.f);
  ASSERT_FALSE(y);
}

TEST(GrokOps, AttnTemperatureScale) {
  EXPECT_NEAR(gyre::attn_temperature_scale(512, 1024), 1.f, 1e-6);
  EXPECT_NEAR(gyre::attn_temperature_scale(1024, 1024), 1.f, 1e-6);
  EXPECT_GT(gyre::attn_temperature_scale(2048, 1024), 1.f);
  EXPECT_NEAR(gyre::attn_temperature_scale(2048, 1024),
              std::log(2048.f) / std::log(1024.f), 1e-5);
}
