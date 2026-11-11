#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/set_encoder.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace gyre;
using namespace gyre::testing;

namespace {

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}

double dot_all(const Tensor& y, const Tensor& c) {
  auto a = y.host_span<float>();
  auto b = c.host_span<float>();
  double s = 0;
  for (std::size_t i = 0; i < a->size(); ++i) s += double((*a)[i]) * double((*b)[i]);
  return s;
}

SetEncoderConfig small_cfg() {
  SetEncoderConfig c;
  c.feat_dim = 4;
  c.global_dim = 3;
  c.max_entities = 8;
  c.d_model = 8;
  c.n_layer = 2;
  c.n_head = 2;
  c.d_ff = 12;
  c.out_dim = 6;
  return c;
}

SetEncoder make_enc(const SetEncoderConfig& c, std::uint64_t seed) {
  Rng rng(seed);
  auto d = Device::cpu();
  auto e = SetEncoder::create(c, *d, rng);
  if (!e) std::abort();
  // Larger weights than init so the gradients are not vanishingly small.
  for (auto& p : e->parameters()) {
    auto v = p.value.host_span<float>();
    if (p.value.rank() == 1) {
      for (auto& x : *v) x += rng.normal(0.f, 0.1f);
    }
  }
  return std::move(*e);
}

Result<Tensor> fwd(SetEncoder& e, const Tensor& ent, const Tensor& m, const Tensor& g, bool train = false) {
  ForwardCtx ctx;
  ctx.train = train;
  return e.forward(ent, m, g, ctx);
}

}  // namespace

TEST(SetEncoder, GradCheckAllParamsAndInputs) {
  auto cfg = small_cfg();
  auto enc = make_enc(cfg, 1);
  Rng rng(2);
  const std::int64_t B = 2, N = 6;
  auto mask = mk_u8({B, N}, {1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0});  // second row: 1 entity
  auto ent = make_p(rand_t({B, N, cfg.feat_dim}, rng));
  auto glob = make_p(rand_t({B, cfg.global_dim}, rng));
  auto c = rand_t({B, cfg.out_dim}, rng);

  std::vector<Param> ps;
  for (auto& p : enc.parameters()) ps.push_back(Param{p.value, p.grad});
  ps.push_back(ent);
  ps.push_back(glob);
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    ForwardCtx ctx;
    ctx.train = true;
    auto y = enc.forward(ps[ps.size() - 2].value, mask, ps.back().value, ctx);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto g = enc.backward(c, ctx);
      if (!g) return std::unexpected(g.error());
      set_grad(ps[ps.size() - 2], g->d_entities);
      set_grad(ps.back(), g->d_global);
    }
    return dot_all(*y, c);
  };
  GradCheckOpts o;
  o.h = 3e-3;  // small step: max-pool argmax must not flip under perturbation
  o.rtol = 5e-2;
  o.atol = 1e-3;
  o.max_per_param = 12;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);

  // Padded entities receive exactly zero gradient.
  auto g = ent.grad.host_span<float>();
  auto mk = mask.host_span<std::uint8_t>();
  for (int b = 0; b < B; ++b)
    for (int n = 0; n < N; ++n)
      if (!(*mk)[b * N + n])
        for (int f = 0; f < cfg.feat_dim; ++f) EXPECT_EQ((*g)[(b * N + n) * cfg.feat_dim + f], 0.f);
}

TEST(SetEncoder, PaddedContentDoesNotAffectOutput) {
  auto cfg = small_cfg();
  auto enc = make_enc(cfg, 3);
  Rng rng(4);
  const std::int64_t B = 2, N = 6;
  auto mask = mk_u8({B, N}, {1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0});
  auto ent = rand_t({B, N, cfg.feat_dim}, rng);
  auto glob = rand_t({B, cfg.global_dim}, rng);
  auto y1 = fwd(enc, ent, mask, glob);
  ASSERT_TRUE(y1);
  // overwrite padded slots with garbage (including huge values)
  auto ent2 = ent.clone();
  auto pe = ent2->host_span<float>();
  auto mk = mask.host_span<std::uint8_t>();
  for (int b = 0; b < B; ++b)
    for (int n = 0; n < N; ++n)
      if (!(*mk)[b * N + n])
        for (int f = 0; f < cfg.feat_dim; ++f) (*pe)[(b * N + n) * cfg.feat_dim + f] = 1e6f * (f + 1);
  auto y2 = fwd(enc, *ent2, mask, glob);
  ASSERT_TRUE(y2);
  auto a = y1->host_span<float>();
  auto b2 = y2->host_span<float>();
  EXPECT_EQ(0, std::memcmp(a->data(), b2->data(), a->size() * sizeof(float)));
}

TEST(SetEncoder, PermutationInvariant) {
  auto cfg = small_cfg();
  auto enc = make_enc(cfg, 5);
  Rng rng(6);
  const std::int64_t N = 6, F = cfg.feat_dim;
  auto ent = rand_t({1, N, F}, rng);
  auto glob = rand_t({1, cfg.global_dim}, rng);
  auto mask = mk_u8({1, N}, {1, 1, 1, 1, 0, 0});
  auto y1 = fwd(enc, ent, mask, glob);
  // reverse the 4 valid entities and move padding around
  auto ent2 = ent.clone();
  auto pe = ent2->host_span<float>();
  auto po = ent.host_span<float>();
  const int perm[6] = {5, 3, 2, 0, 4, 1};  // slot i takes entity perm[i]
  std::vector<std::uint8_t> m2(6);
  auto mk = mask.host_span<std::uint8_t>();
  for (int i = 0; i < 6; ++i) {
    m2[i] = (*mk)[perm[i]];
    for (int f = 0; f < F; ++f) (*pe)[i * F + f] = (*po)[perm[i] * F + f];
  }
  auto y2 = fwd(enc, *ent2, mk_u8({1, N}, m2), glob);
  ASSERT_TRUE(y1 && y2);
  auto a = y1->host_span<float>();
  auto b = y2->host_span<float>();
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_NEAR((*a)[i], (*b)[i], 1e-5f);
}

TEST(SetEncoder, EmptySetIsFinite) {
  auto cfg = small_cfg();
  auto enc = make_enc(cfg, 7);
  Rng rng(8);
  auto ent = rand_t({2, 4, cfg.feat_dim}, rng);
  auto glob = rand_t({2, cfg.global_dim}, rng);
  auto mask = mk_u8({2, 4}, {0, 0, 0, 0, 1, 0, 0, 0});
  ForwardCtx ctx;
  auto y = enc.forward(ent, mask, glob, ctx);
  ASSERT_TRUE(y);
  for (float v : *y->host_span<float>()) EXPECT_TRUE(std::isfinite(v));
  auto g = enc.backward(rand_t({2, cfg.out_dim}, rng), ctx);
  ASSERT_TRUE(g);
  for (auto& p : enc.parameters())
    for (float v : *p.grad.host_span<float>()) ASSERT_TRUE(std::isfinite(v));
}

TEST(SetEncoder, ShapeChecks) {
  auto cfg = small_cfg();
  auto enc = make_enc(cfg, 9);
  Rng rng(10);
  auto glob = rand_t({1, cfg.global_dim}, rng);
  auto mask = mk_u8({1, 3}, {1, 1, 1});
  EXPECT_FALSE(fwd(enc, rand_t({1, 3, cfg.feat_dim + 1}, rng), mask, glob));
  EXPECT_FALSE(fwd(enc, rand_t({1, 3, cfg.feat_dim}, rng), mask, rand_t({1, cfg.global_dim + 1}, rng)));
  auto big = mk_u8({1, 9}, std::vector<std::uint8_t>(9, 1));
  EXPECT_FALSE(fwd(enc, rand_t({1, 9, cfg.feat_dim}, rng), big, glob));  // N > max_entities
}

TEST(SetEncoder, DefaultSizeAt128Entities) {
  SetEncoderConfig c;
  c.feat_dim = 24;
  c.global_dim = 16;
  Rng rng(11);
  auto d = Device::cpu();
  auto enc = SetEncoder::create(c, *d, rng);
  ASSERT_TRUE(enc);
  std::int64_t params = 0;
  for (auto& p : enc->parameters()) params += p.value.numel();
  EXPECT_LT(params, 2'000'000);
  auto ent = rand_t({1, c.max_entities, c.feat_dim}, rng);
  auto mask = mk_u8({1, c.max_entities}, std::vector<std::uint8_t>(c.max_entities, 1));
  auto y = fwd(*enc, ent, mask, rand_t({1, c.global_dim}, rng));
  ASSERT_TRUE(y);
  EXPECT_EQ(y->shape()[1], c.out_dim);
}
