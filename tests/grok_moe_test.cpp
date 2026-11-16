#include "gyre/nn/moe.hpp"

#include <algorithm>
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

// Grok routing (xai-org/grok-1 MoELayer, SGLang fused_moe_router with renormalize=False):
// softmax over ALL experts, then the top-k probabilities as-is (they do not sum to 1).
TEST(GrokMoe, TopKWithoutRenormalizeKeepsSoftmaxProbs) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 4};
  float z[] = {1.f, 3.f, 0.f, 2.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(z)), sh, gyre::DType::f32, *d);
  auto top = gyre::moe_topk(*t, 2, 0.f, /*renormalize=*/false);
  ASSERT_TRUE(top) << top.error().message;
  double se = 0;
  for (float v : z) se += std::exp(static_cast<double>(v));
  auto pi = top->indices.host_span<std::int32_t>();
  auto pw = top->weights.host_span<float>();
  EXPECT_EQ((*pi)[0], 1);
  EXPECT_EQ((*pi)[1], 3);
  EXPECT_NEAR((*pw)[0], std::exp(3.0) / se, 1e-6);
  EXPECT_NEAR((*pw)[1], std::exp(2.0) / se, 1e-6);
  EXPECT_LT((*pw)[0] + (*pw)[1], 0.95f);
}

namespace {

float gelu_tanh_ref(float x) {
  const double u = 0.7978845608028654 * (x + 0.044715 * double(x) * x * x);
  return static_cast<float>(0.5 * x * (1.0 + std::tanh(u)));
}

// act(x@w1) * (x@w3) @ w2 for one row, gelu-tanh gate (Grok GeGLU).
std::vector<double> geglu_row(const float* x, const gyre::SwiGLUWeights& w, std::int64_t D, std::int64_t I) {
  auto p1 = w.w1->host_span<float>(), p3 = w.w3->host_span<float>(), p2 = w.w2->host_span<float>();
  std::vector<double> h(static_cast<std::size_t>(I)), y(static_cast<std::size_t>(D), 0.0);
  for (std::int64_t i = 0; i < I; ++i) {
    double g = 0, u = 0;
    for (std::int64_t c = 0; c < D; ++c) {
      g += double(x[c]) * (*p1)[static_cast<std::size_t>(c * I + i)];
      u += double(x[c]) * (*p3)[static_cast<std::size_t>(c * I + i)];
    }
    h[static_cast<std::size_t>(i)] = gelu_tanh_ref(static_cast<float>(g)) * u;
  }
  for (std::int64_t c = 0; c < D; ++c)
    for (std::int64_t i = 0; i < I; ++i)
      y[static_cast<std::size_t>(c)] += h[static_cast<std::size_t>(i)] * (*p2)[static_cast<std::size_t>(i * D + c)];
  return y;
}

}  // namespace

// kGrokMoe against a scalar transcription of the reference Grok-2 layer FFN (SGLang
// Grok1DecoderLayer.moe_with_rmoe): (GeGLU_dense(x) + sum_j p_j GeGLU_{e_j}(x)) / sqrt(2), with
// p = top_k(softmax(30*tanh(logits/30))) not renormalized; and the sparse (residual_moe=false)
// form without the dense branch or the 1/sqrt(2).
TEST(GrokMoe, GrokSpecMatchesReferenceLayer) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 5, Dim = 6, I = 4, E = 4;
  const int k = 2;
  const float cap = 30.f;
  auto x = randn2(N, Dim, 11, *d);
  std::vector<gyre::Tensor> w;
  for (std::uint64_t s = 0; s < 3 * (E + 1); ++s) {
    const bool down = (s % 3) == 2;
    w.push_back(randn2(down ? I : Dim, down ? Dim : I, 300 + s, *d));
  }
  gyre::SwiGLUWeights dense{&w[0], &w[1], &w[2]};
  std::vector<gyre::SwiGLUWeights> ex;
  for (std::int64_t e = 0; e < E; ++e) ex.push_back({&w[3 + 3 * e], &w[4 + 3 * e], &w[5 + 3 * e]});
  auto rw = randn2(Dim, E, 77, *d);
  auto y = gyre::residual_moe(x, dense, rw, ex, k, cap, gyre::kGrokMoe);
  auto ys = gyre::residual_moe(x, gyre::SwiGLUWeights{}, rw, ex, k, cap, gyre::kGrokMoe);
  ASSERT_TRUE(y) << y.error().message;
  ASSERT_TRUE(ys) << ys.error().message;
  auto px = x.host_span<float>();
  auto pr = rw.host_span<float>();
  auto py = y->host_span<float>();
  auto ps = ys->host_span<float>();
  for (std::int64_t n = 0; n < N; ++n) {
    const float* xr = px->data() + n * Dim;
    std::vector<double> prob(static_cast<std::size_t>(E));
    double mx = -1e30, se = 0;
    for (std::int64_t e = 0; e < E; ++e) {
      double l = 0;
      for (std::int64_t c = 0; c < Dim; ++c) l += double(xr[c]) * (*pr)[static_cast<std::size_t>(c * E + e)];
      l = cap * std::tanh(l / cap);
      prob[static_cast<std::size_t>(e)] = l;
      mx = std::max(mx, l);
    }
    for (auto& p : prob) se += (p = std::exp(p - mx));
    for (auto& p : prob) p /= se;
    std::vector<int> order(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<std::size_t>(e)] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return prob[static_cast<std::size_t>(a)] > prob[static_cast<std::size_t>(b)];
    });
    std::vector<double> moe(static_cast<std::size_t>(Dim), 0.0);
    for (int j = 0; j < k; ++j) {
      const int e = order[static_cast<std::size_t>(j)];
      auto ye = geglu_row(xr, ex[static_cast<std::size_t>(e)], Dim, I);
      for (std::int64_t c = 0; c < Dim; ++c)
        moe[static_cast<std::size_t>(c)] += prob[static_cast<std::size_t>(e)] * ye[static_cast<std::size_t>(c)];
    }
    auto yd = geglu_row(xr, dense, Dim, I);
    for (std::int64_t c = 0; c < Dim; ++c) {
      const auto i = static_cast<std::size_t>(n * Dim + c);
      const double ref = (yd[static_cast<std::size_t>(c)] + moe[static_cast<std::size_t>(c)]) / std::sqrt(2.0);
      EXPECT_NEAR((*py)[i], ref, 1e-4) << "n=" << n << " c=" << c;
      EXPECT_NEAR((*ps)[i], moe[static_cast<std::size_t>(c)], 1e-4) << "sparse n=" << n << " c=" << c;
    }
  }
}

// The default spec stays the generic SwiGLU MoE (silu gate, renormalized, no scaling).
TEST(GrokMoe, DefaultSpecIsUnscaledSwiGLU) {
  auto d = gyre::Device::cpu();
  const std::int64_t N = 3, Dim = 4, I = 3, E = 2;
  auto x = randn2(N, Dim, 21, *d);
  auto w1 = randn2(Dim, I, 22, *d), w3 = randn2(Dim, I, 23, *d), w2 = randn2(I, Dim, 24, *d);
  auto rw = randn2(Dim, E, 25, *d);
  gyre::SwiGLUWeights dense{&w1, &w3, &w2};
  gyre::SwiGLUWeights ex[] = {dense, dense};
  // Both experts identical and k = E: renormalized weights sum to 1, so y = 2 * swiglu(x).
  auto y = gyre::residual_moe(x, dense, rw, ex, 2, 0.f);
  auto s = gyre::swiglu_ffn(x, dense);
  auto g = gyre::swiglu_ffn(x, dense, gyre::GateAct::gelu_tanh);
  ASSERT_TRUE(y && s && g);
  auto py = y->host_span<float>();
  auto ps = s->host_span<float>();
  auto pg = g->host_span<float>();
  bool differs = false;
  for (std::size_t i = 0; i < py->size(); ++i) {
    EXPECT_NEAR((*py)[i], 2.f * (*ps)[i], 1e-5f);
    differs |= std::fabs((*pg)[i] - (*ps)[i]) > 1e-4f;
  }
  EXPECT_TRUE(differs);  // gelu gate is a different activation
}
