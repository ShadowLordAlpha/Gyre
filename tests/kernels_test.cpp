// Tests for the optimised CPU kernels: results must equal straightforward reference code, be
// independent of the thread count, and the buffer pool must never leak stale data.
#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/ops.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

using namespace gyre;
using namespace gyre::testing;

namespace {

std::vector<float> to_vec(const Tensor& t) {
  auto p = t.host_span<float>();
  return std::vector<float>(p->begin(), p->end());
}

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}

void expect_near_vec(const std::vector<float>& a, const std::vector<float>& b, float tol, const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (std::size_t i = 0; i < a.size(); ++i)
    ASSERT_NEAR(a[i], b[i], tol * (1.f + std::abs(b[i]))) << what << " [" << i << "]";
}

}  // namespace

TEST(Kernels, MatmulTnEqualsTransposeThenMatmul) {
  Rng rng(1);
  // small (single block) and large (multi-block, ragged tail) row counts
  for (std::int64_t M : {7, 300, 1030, 4097}) {
    auto a = rand_t({M, 13}, rng), b = rand_t({M, 9}, rng);
    auto got = matmul_tn(a, b);
    ASSERT_TRUE(got);
    auto ref = matmul(*transpose_last2(a), b);
    ASSERT_TRUE(ref);
    EXPECT_EQ(got->shape()[0], 13);
    EXPECT_EQ(got->shape()[1], 9);
    expect_near_vec(to_vec(*got), to_vec(*ref), 2e-4f, "matmul_tn");
  }
  EXPECT_FALSE(matmul_tn(rand_t({4, 3}, rng), rand_t({5, 3}, rng)));  // row mismatch
}

TEST(Kernels, BmmTnEqualsTransposeThenBmm) {
  Rng rng(2);
  auto a = rand_t({3, 2, 11, 5}, rng), b = rand_t({3, 2, 11, 7}, rng);
  auto got = bmm_tn(a, b);
  ASSERT_TRUE(got);
  auto ref = bmm(*transpose_last2(a), b);
  expect_near_vec(to_vec(*got), to_vec(*ref), 1e-5f, "bmm_tn");
}

TEST(Kernels, ReductionsAreThreadCountIndependent) {
  Rng rng(3);
  auto a = rand_t({5000, 33}, rng), b = rand_t({5000, 17}, rng);
  std::vector<float> tn1, tn8, s1, s8;
  set_num_threads(1);
  tn1 = to_vec(*matmul_tn(a, b));
  s1 = to_vec(*sum_dim(a, 0, false));
  set_num_threads(8);
  tn8 = to_vec(*matmul_tn(a, b));
  s8 = to_vec(*sum_dim(a, 0, false));
  EXPECT_EQ(0, std::memcmp(tn1.data(), tn8.data(), tn1.size() * sizeof(float)));
  EXPECT_EQ(0, std::memcmp(s1.data(), s8.data(), s1.size() * sizeof(float)));
  set_num_threads(0);  // back to automatic for the other tests
}

TEST(Kernels, SumDimMatchesNaive) {
  Rng rng(4);
  auto a = rand_t({4, 5, 6, 3}, rng);
  auto pa = a.host_span<float>();
  for (int axis = 0; axis < 4; ++axis) {
    for (bool keep : {false, true}) {
      auto got = sum_dim(a, axis, keep);
      ASSERT_TRUE(got);
      // naive reference
      std::int64_t dims[4] = {4, 5, 6, 3};
      std::int64_t outer = 1, inner = 1;
      for (int i = 0; i < axis; ++i) outer *= dims[i];
      for (int i = axis + 1; i < 4; ++i) inner *= dims[i];
      std::vector<float> ref(static_cast<std::size_t>(outer * inner), 0.f);
      for (std::int64_t o = 0; o < outer; ++o)
        for (std::int64_t l = 0; l < dims[axis]; ++l)
          for (std::int64_t i = 0; i < inner; ++i)
            ref[static_cast<std::size_t>(o * inner + i)] += (*pa)[static_cast<std::size_t>((o * dims[axis] + l) * inner + i)];
      expect_near_vec(to_vec(*got), ref, 1e-5f, "sum_dim");
      EXPECT_EQ(got->rank(), keep ? 4 : 3);
    }
  }
  // long single-row reduction (block path) vs naive double accumulation
  auto big = rand_t({3000, 7}, rng);
  auto g = sum_dim(big, 0, false);
  auto pb = big.host_span<float>();
  for (int j = 0; j < 7; ++j) {
    double s = 0;
    for (int r = 0; r < 3000; ++r) s += (*pb)[r * 7 + j];
    EXPECT_NEAR((*g->host_span<float>())[j], s, 1e-3);
  }
}

TEST(Kernels, TransposeAndPermuteMatchNaive) {
  Rng rng(5);
  for (auto sh : {std::vector<std::int64_t>{2, 37, 53}, std::vector<std::int64_t>{1, 16, 16}, std::vector<std::int64_t>{300, 20}}) {
    auto a = rand_t(sh, rng);
    auto t = transpose_last2(a);
    ASSERT_TRUE(t);
    const auto r = sh.size();
    const std::int64_t d0 = sh[r - 2], d1 = sh[r - 1];
    std::int64_t batch = 1;
    for (std::size_t i = 0; i + 2 < r; ++i) batch *= sh[i];
    auto pa = a.host_span<float>();
    auto pt = t->host_span<float>();
    for (std::int64_t b = 0; b < batch; ++b)
      for (std::int64_t i = 0; i < d0; ++i)
        for (std::int64_t j = 0; j < d1; ++j)
          ASSERT_EQ((*pt)[b * d0 * d1 + j * d0 + i], (*pa)[b * d0 * d1 + i * d1 + j]);
  }
  auto x = rand_t({2, 5, 3, 4}, rng);  // [B,T,H,d]
  auto y = permute_bthd_bhtd(x);
  auto back = permute_bhtd_bthd(*y);
  expect_near_vec(to_vec(*back), to_vec(x), 0.f, "permute round trip");
  auto px = x.host_span<float>();
  auto py = y->host_span<float>();
  for (int b = 0; b < 2; ++b)
    for (int t = 0; t < 5; ++t)
      for (int h = 0; h < 3; ++h)
        for (int i = 0; i < 4; ++i)
          ASSERT_EQ((*py)[((b * 3 + h) * 5 + t) * 4 + i], (*px)[((b * 5 + t) * 3 + h) * 4 + i]);
}

TEST(Kernels, FusedAttentionMatchesUnfusedReference) {
  Rng rng(6);
  const std::int64_t B = 2, H = 3, N = 9, dk = 8;
  auto q = rand_t({B, H, N, dk}, rng), k = rand_t({B, H, N, dk}, rng), v = rand_t({B, H, N, dk}, rng);
  auto mask = mk_u8({B, N}, {1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1});
  const float scale = 1.f / std::sqrt(float(dk));
  auto w = attn_probs(q, k, &mask, scale);
  ASSERT_TRUE(w);
  // reference: transpose + bmm + scale + key mask + softmax
  auto scores = bmm(q, *transpose_last2(k));
  auto scaled = mul_scalar(*scores, scale);
  ASSERT_TRUE(key_mask_(*scaled, mask));
  auto ref = softmax_last(*scaled);
  expect_near_vec(to_vec(*w), to_vec(*ref), 1e-5f, "attn_probs");
  // masked keys: exactly zero weight, rows sum to one
  auto pw = w->host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  for (std::int64_t r = 0; r < B * H * N; ++r) {
    const std::int64_t b = (r / N) / H;
    double sum = 0;
    for (std::int64_t j = 0; j < N; ++j) {
      sum += (*pw)[r * N + j];
      if (!(*pm)[b * N + j]) ASSERT_EQ((*pw)[r * N + j], 0.f);
    }
    ASSERT_NEAR(sum, 1.0, 1e-5);
  }
  // backward vs reference chain: dz = scale * softmax_backward(w, dy v^T)
  auto dy = rand_t({B, H, N, dk}, rng);
  auto dz = attn_probs_backward(*w, dy, v, scale);
  ASSERT_TRUE(dz);
  auto dw = bmm(dy, *transpose_last2(v));
  auto dz_ref = mul_scalar(*softmax_last_backward(*ref, *dw), scale);
  expect_near_vec(to_vec(*dz), to_vec(*dz_ref), 1e-5f, "attn_probs_backward");
  // no mask pointer = all keys valid; all-masked row degrades to uniform instead of NaN
  auto none = attn_probs(q, k, nullptr, scale);
  ASSERT_TRUE(none);
  for (float x : to_vec(*none)) ASSERT_TRUE(std::isfinite(x));
  auto allmasked = mk_u8({B, N}, std::vector<std::uint8_t>(B * N, 0));
  auto uni = attn_probs(q, k, &allmasked, scale);
  for (float x : to_vec(*uni)) ASSERT_NEAR(x, 1.0 / N, 1e-6);
}

TEST(Kernels, FusedAttentionGradCheck) {
  Rng rng(7);
  const std::int64_t B = 1, H = 2, N = 5, dk = 4;
  auto mask = mk_u8({B, N}, {1, 1, 0, 1, 1});
  const float scale = 0.5f;
  auto q = make_p(rand_t({B, H, N, dk}, rng)), k = make_p(rand_t({B, H, N, dk}, rng)),
       v = make_p(rand_t({B, H, N, dk}, rng));
  auto c = rand_t({B, H, N, dk}, rng);
  std::vector<Param> ps{q, k, v};
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    auto w = attn_probs(ps[0].value, ps[1].value, &mask, scale);
    if (!w) return std::unexpected(w.error());
    auto y = bmm(*w, ps[2].value);
    if (!y) return std::unexpected(y.error());
    if (backward) {  // exactly the chain SetBlock::backward uses
      auto dz = attn_probs_backward(*w, c, ps[2].value, scale);
      auto dv = bmm_tn(*w, c);
      auto dq = bmm(*dz, ps[1].value);
      auto dk_ = bmm_tn(*dz, ps[0].value);
      if (!dz || !dv || !dq || !dk_) return std::unexpected(make_error(Errc::unsupported, "bwd"));
      set_grad(ps[0], *dq);
      set_grad(ps[1], *dk_);
      set_grad(ps[2], *dv);
    }
    double s = 0;
    auto a = y->host_span<float>(), b = c.host_span<float>();
    for (std::size_t i = 0; i < a->size(); ++i) s += double((*a)[i]) * (*b)[i];
    return s;
  };
  GradCheckOpts o;
  o.h = 3e-3;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
}

TEST(Kernels, BufferPoolNeverLeaksStaleData) {
  // Allocate big buffers, fill with a pattern, free, then request zeroed memory of similar size.
  auto d = Device::cpu();
  const std::int64_t n = 1 << 20;  // 4 MiB of floats: above the pooling threshold
  std::int64_t sh[1] = {n};
  for (int round = 0; round < 3; ++round) {
    {
      auto t = Tensor::empty(sh, DType::f32, *d);
      ASSERT_TRUE(t);
      for (auto& x : *t->host_span<float>()) x = 123.f;
    }  // returned to the pool
    auto z = Tensor::zeros(sh, DType::f32, *d);
    ASSERT_TRUE(z);
    for (float x : *z->host_span<float>()) ASSERT_EQ(x, 0.f);
    auto e = Tensor::empty(sh, DType::f32, *d);
    for (float x : *e->host_span<float>()) ASSERT_EQ(x, 0.f);  // empty() stays zero-filled
    // from_host overwrites everything
    std::vector<float> src(static_cast<std::size_t>(n), 7.f);
    auto h = Tensor::from_host(std::as_bytes(std::span<const float>(src)), sh, DType::f32, *d);
    ASSERT_TRUE(h);
    for (float x : *h->host_span<float>()) ASSERT_EQ(x, 7.f);
  }
}
