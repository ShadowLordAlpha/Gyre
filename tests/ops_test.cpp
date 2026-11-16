#include "gyre/ops.hpp"

#include <cstdint>
#include <vector>

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

// [N,0] and [0,C] inputs used to compute rows = numel / last = 0 / 0 (integer divide trap).
TEST(Ops, ZeroWidthNormsAndSoftmaxReturnEmpty) {
  auto d = gyre::Device::cpu();
  ASSERT_TRUE(d);
  for (auto shape : {std::vector<std::int64_t>{4, 0}, std::vector<std::int64_t>{0, 3}}) {
    auto x = gyre::Tensor::zeros(shape, gyre::DType::f32, *d);
    ASSERT_TRUE(x);
    std::int64_t csh[] = {shape[1]};
    auto w = gyre::Tensor::zeros(csh, gyre::DType::f32, *d);
    auto b = gyre::Tensor::zeros(csh, gyre::DType::f32, *d);
    auto gw = gyre::Tensor::zeros(csh, gyre::DType::f32, *d);
    auto gb = gyre::Tensor::zeros(csh, gyre::DType::f32, *d);
    ASSERT_TRUE(w && b && gw && gb);
    auto s = gyre::softmax_last(*x);
    ASSERT_TRUE(s) << s.error().message;
    EXPECT_EQ(s->numel(), 0);
    auto ln = gyre::layer_norm(*x, *w, *b, 1e-5f);
    ASSERT_TRUE(ln) << ln.error().message;
    EXPECT_EQ(ln->numel(), 0);
    auto rn = gyre::rms_norm(*x, *w, 1e-5f);
    ASSERT_TRUE(rn) << rn.error().message;
    EXPECT_EQ(rn->numel(), 0);
    auto sb = gyre::softmax_last_backward(*s, *x);
    ASSERT_TRUE(sb) << sb.error().message;
    EXPECT_EQ(sb->numel(), 0);
    auto lb = gyre::layer_norm_backward(*x, *x, *w, *gw, *gb, 1e-5f);
    ASSERT_TRUE(lb) << lb.error().message;
    EXPECT_EQ(lb->numel(), 0);
  }
}

// uniform01() returns exactly 0 with probability 2^-24. With `u <= acc` that picked token 0 even
// when its probability had underflowed to 0. Find such a draw and check it is never chosen.
TEST(Ops, SampleLogitRowNeverPicksZeroProbability) {
  gyre::Rng probe(2024);
  std::uint64_t skip = 0;
  bool found = false;
  for (; skip < (std::uint64_t{1} << 30); ++skip) {
    if ((probe.engine()() >> 40) == 0) {
      found = true;
      break;
    }
  }
  ASSERT_TRUE(found);
  gyre::Rng rng(2024);
  rng.engine().discard(skip);
  // Token 0 at -100 with T=0.01: exp(-1e4) underflows to exactly 0.
  const float logits[] = {-100.f, 0.f, -100.f};
  auto t = gyre::sample_logit_row(logits, 0.01f, &rng);
  ASSERT_TRUE(t);
  EXPECT_EQ(*t, 1);
  // Trailing zero-probability tokens are never a fallback either.
  gyre::Rng r2(5);
  const float tail[] = {0.f, -100.f, -100.f};
  for (int i = 0; i < 1000; ++i) {
    auto s = gyre::sample_logit_row(tail, 0.01f, &r2);
    ASSERT_TRUE(s);
    EXPECT_EQ(*s, 0);
  }
}
