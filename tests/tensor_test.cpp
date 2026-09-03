#include "gyre/tensor.hpp"

#include <gtest/gtest.h>

TEST(Tensor, EmptyZerosHost) {
  auto d = gyre::Device::cpu();
  ASSERT_TRUE(d);
  std::int64_t sh[] = {2, 3};
  auto z = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  ASSERT_TRUE(z);
  EXPECT_EQ(z->numel(), 6);
  auto s = z->host_span<float>();
  ASSERT_TRUE(s);
  for (auto v : *s) EXPECT_EQ(v, 0.f);
}

TEST(Tensor, CopyDeletedClone) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {2};
  float v[] = {1.f, 2.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *d);
  ASSERT_TRUE(t);
  auto c = t->clone();
  ASSERT_TRUE(c);
  auto s = c->host_span<float>();
  EXPECT_EQ((*s)[1], 2.f);
}
