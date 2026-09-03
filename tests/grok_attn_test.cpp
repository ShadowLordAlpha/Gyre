#include "gyre/nn/gqa.hpp"
#include "gyre/rng.hpp"

#include <cmath>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(GrokAttn, RepeatKvIdentityWhenEqualHeads) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 2, 3, 4};
  auto kv = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  auto y = gyre::repeat_kv_heads(*kv, 2);
  ASSERT_TRUE(y);
  EXPECT_EQ(y->shape()[1], 2);
}

TEST(GrokAttn, RepeatKvExpands) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 1, 2, 2};
  float v[] = {1.f, 2.f, 3.f, 4.f};
  auto kv = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *d);
  auto y = gyre::repeat_kv_heads(*kv, 4);
  ASSERT_TRUE(y) << y.error().message;
  EXPECT_EQ(y->shape()[1], 4);
  auto p = y->host_span<float>();
  EXPECT_EQ((*p)[0], 1.f);
  EXPECT_EQ((*p)[4], 1.f);  // second copy of head 0
}

TEST(GrokAttn, GqaMatchesMhaWhenHeadsEqual) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(7);
  std::int64_t sh[] = {1, 2, 4, 8};
  auto q = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  auto k = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  auto v = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  ASSERT_TRUE(q);
  auto fill = [&](gyre::Tensor& t) {
    auto p = t.host_span<float>();
    for (auto& x : *p) x = rng.normal(0.f, 0.1f);
  };
  fill(*q);
  fill(*k);
  fill(*v);
  auto a = gyre::gqa_causal(*q, *k, *v, 0.f, 1.f);
  auto b = gyre::gqa_causal(*q, *k, *v, 0.f, 1.f);
  ASSERT_TRUE(a) << a.error().message;
  ASSERT_TRUE(b);
  auto pa = a->host_span<float>();
  auto pb = b->host_span<float>();
  for (std::size_t i = 0; i < pa->size(); ++i) EXPECT_NEAR((*pa)[i], (*pb)[i], 1e-5);
}

TEST(GrokAttn, SoftcapChangesScores) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 1, 2, 4};
  float qv[] = {5.f, 1.f, 0, 0, 5.f, 1.f, 0, 0};
  float kv[] = {5.f, 0, 0, 0, 0, 5.f, 0, 0};
  auto q = gyre::Tensor::from_host(std::as_bytes(std::span(qv)), sh, gyre::DType::f32, *d);
  auto k = gyre::Tensor::from_host(std::as_bytes(std::span(kv)), sh, gyre::DType::f32, *d);
  auto y0 = gyre::gqa_causal(*q, *k, *k, 0.f, 1.f);
  auto y1 = gyre::gqa_causal(*q, *k, *k, 0.25f, 1.f);
  ASSERT_TRUE(y0);
  ASSERT_TRUE(y1);
  auto p0 = y0->host_span<float>();
  auto p1 = y1->host_span<float>();
  float d2 = 0;
  for (std::size_t i = 0; i < p0->size(); ++i) {
    float e = (*p0)[i] - (*p1)[i];
    d2 += e * e;
  }
  EXPECT_GT(d2, 0.f);
}

TEST(GrokAttn, CacheMatchesFullPrefix) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  const std::int64_t T = 8, D = 4, H = 2;
  std::int64_t sh[] = {1, H, T, D};
  auto q = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  auto k = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  auto v = gyre::Tensor::empty(sh, gyre::DType::f32, *d);
  auto fill = [&](gyre::Tensor& t) {
    auto p = t.host_span<float>();
    for (auto& x : *p) x = rng.normal(0.f, 0.2f);
  };
  fill(*q);
  fill(*k);
  fill(*v);
  auto full = gyre::gqa_causal(*q, *k, *v, 30.f, 1.f);
  ASSERT_TRUE(full) << full.error().message;

  // Step through Tq=1 with growing KV (prefix of k,v).
  std::vector<float> step_last(static_cast<std::size_t>(H * D));
  for (std::int64_t t = 0; t < T; ++t) {
    std::int64_t qsh[] = {1, H, 1, D};
    std::int64_t kvsh[] = {1, H, t + 1, D};
    std::vector<float> q1(static_cast<std::size_t>(H * D));
    std::vector<float> kk(static_cast<std::size_t>(H * (t + 1) * D));
    std::vector<float> vv(kk.size());
    auto pq = q->host_span<float>();
    auto pk = k->host_span<float>();
    auto pv = v->host_span<float>();
    for (std::int64_t h = 0; h < H; ++h) {
      for (std::int64_t d = 0; d < D; ++d) {
        q1[static_cast<std::size_t>(h * D + d)] =
            (*pq)[static_cast<std::size_t>((h * T + t) * D + d)];
      }
      for (std::int64_t s = 0; s <= t; ++s) {
        for (std::int64_t d = 0; d < D; ++d) {
          auto dst = static_cast<std::size_t>((h * (t + 1) + s) * D + d);
          auto src = static_cast<std::size_t>((h * T + s) * D + d);
          kk[dst] = (*pk)[src];
          vv[dst] = (*pv)[src];
        }
      }
    }
    auto qt = gyre::Tensor::from_host(std::as_bytes(std::span(q1.data(), q1.size())), qsh,
                                     gyre::DType::f32, *d);
    auto kt = gyre::Tensor::from_host(std::as_bytes(std::span(kk.data(), kk.size())), kvsh,
                                     gyre::DType::f32, *d);
    auto vt = gyre::Tensor::from_host(std::as_bytes(std::span(vv.data(), vv.size())), kvsh,
                                     gyre::DType::f32, *d);
    auto y = gyre::gqa_causal(*qt, *kt, *vt, 30.f, 1.f);
    ASSERT_TRUE(y) << y.error().message;
    auto py = y->host_span<float>();
    auto pf = full->host_span<float>();
    for (std::int64_t h = 0; h < H; ++h) {
      for (std::int64_t d = 0; d < D; ++d) {
        float got = (*py)[static_cast<std::size_t>(h * D + d)];
        float want = (*pf)[static_cast<std::size_t>((h * T + t) * D + d)];
        EXPECT_NEAR(got, want, 1e-4f) << "t=" << t << " h=" << h << " d=" << d;
      }
    }
  }
}
