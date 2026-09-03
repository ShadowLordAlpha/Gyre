#include "gyre/ops.hpp"

#include <gtest/gtest.h>

TEST(Ops, Matmul) {
  auto d = gyre::Device::cpu();
  std::int64_t as[] = {2, 3};
  std::int64_t bs[] = {3, 2};
  float A[] = {1, 2, 3, 4, 5, 6};
  float B[] = {1, 0, 0, 1, 1, 1};
  auto a = gyre::Tensor::from_host(std::as_bytes(std::span(A)), as, gyre::DType::f32, *d);
  auto b = gyre::Tensor::from_host(std::as_bytes(std::span(B)), bs, gyre::DType::f32, *d);
  auto c = gyre::matmul(*a, *b);
  ASSERT_TRUE(c);
  auto p = c->host_span<float>();
  EXPECT_NEAR((*p)[0], 4.f, 1e-5);
}

TEST(Ops, Bmm) {
  auto d = gyre::Device::cpu();
  std::int64_t as[] = {2, 3, 4, 5};
  std::int64_t bs[] = {2, 3, 5, 6};
  auto a = gyre::Tensor::zeros(as, gyre::DType::f32, *d);
  auto b = gyre::Tensor::zeros(bs, gyre::DType::f32, *d);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  auto c = gyre::bmm(*a, *b);
  ASSERT_TRUE(c) << c.error().message;
  EXPECT_EQ(c->shape()[3], 6);
}

TEST(Ops, Embedding) {
  auto d = gyre::Device::cpu();
  std::int64_t ws[] = {3, 2};
  float W[] = {1, 2, 3, 4, 5, 6};
  std::int32_t ix[] = {2, 0};
  std::int64_t ish[] = {2};
  auto w = gyre::Tensor::from_host(std::as_bytes(std::span(W)), ws, gyre::DType::f32, *d);
  auto i = gyre::Tensor::from_host(std::as_bytes(std::span(ix)), ish, gyre::DType::i32, *d);
  auto e = gyre::embedding(*w, *i);
  ASSERT_TRUE(e);
  auto p = e->host_span<float>();
  EXPECT_EQ((*p)[0], 5.f);
  EXPECT_EQ((*p)[2], 1.f);
}
