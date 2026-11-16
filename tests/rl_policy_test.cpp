#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/optim.hpp"
#include "gyre/rl/losses.hpp"
#include "gyre/checkpoint.hpp"
#include "gyre/rl/policy.hpp"
#include "gyre/rl/policy_io.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

using namespace gyre;
using namespace gyre::testing;

namespace {

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}
Tensor mk_i32(std::vector<std::int64_t> shape, const std::vector<std::int32_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(v)), shape, DType::i32, *d);
  if (!t) std::abort();
  return std::move(*t);
}

// layout {3,4,2}, B=4. Mix of illegal options and inactive heads. Inactive heads get an
// out-of-range action and an all-illegal mask on purpose: they must be ignored.
struct Fixture {
  HeadLayout L = *HeadLayout::make({3, 4, 2});
  HeadMasks m{mk_u8({4, 9}, {1, 1, 1, 1, 0, 1, 1, 1, 1,    // all legal
                              1, 0, 1, 0, 1, 1, 0, 1, 1,  // some illegal
                              0, 1, 1, 1, 1, 1, 1, 0, 1,
                              1, 1, 0, 0, 0, 0, 0, 1, 1}),  // head1 all-illegal but inactive
              mk_u8({4, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1})};
  // active rows: s0 all; s1 all; s2 heads0,1; s3 heads0,2
  Tensor actions = mk_i32({4, 3}, {0, 3, 1, 2, 2, 0, 1, 1, 99, 0, -7, 1});
};

}  // namespace

TEST(Heads, GradCheckLogProbAndEntropy) {
  Fixture f;
  Rng rng(1);
  auto logits = make_p(rand_t({4, 9}, rng));
  auto c1 = make_t({4}, {0.7f, -1.1f, 0.4f, 1.3f});
  auto c2 = make_t({4}, {0.5f, 0.9f, -0.6f, 0.3f});
  std::vector<Param> ps{logits};
  auto fn = [&](bool backward) -> Result<double> {
    auto d = head_dist(ps[0].value, f.L, f.m);
    if (!d) return std::unexpected(d.error());
    auto lp = log_prob(*d, f.L, f.m, f.actions);
    auto en = entropy(*d, f.L, f.m);
    if (!lp || !en) return std::unexpected(lp ? en.error() : lp.error());
    if (backward) {
      auto g = log_prob_entropy_backward(*d, f.L, f.m, f.actions, c1, c2);
      if (!g) return std::unexpected(g.error());
      set_grad(ps[0], *g);
    }
    double s = 0;
    auto a = lp->host_span<float>(), b = en->host_span<float>();
    auto p1 = c1.host_span<float>(), p2 = c2.host_span<float>();
    for (int i = 0; i < 4; ++i) s += double((*a)[i]) * (*p1)[i] + double((*b)[i]) * (*p2)[i];
    return s;
  };
  GradCheckOpts o;
  o.h = 2e-3;
  o.max_per_param = 36;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);

  // Illegal options and inactive heads: gradient is exactly zero.
  auto g = ps[0].grad.host_span<float>();
  auto legal = f.m.legal.host_span<std::uint8_t>();
  auto act = f.m.active.host_span<std::uint8_t>();
  const std::int64_t off[3] = {0, 3, 7}, sz[3] = {3, 4, 2};
  for (int b = 0; b < 4; ++b)
    for (int h = 0; h < 3; ++h)
      for (int j = 0; j < sz[h]; ++j) {
        const int i = b * 9 + off[h] + j;
        if (!(*act)[b * 3 + h] || !(*legal)[i]) EXPECT_EQ((*g)[i], 0.f) << "b" << b << " h" << h << " j" << j;
      }
}

TEST(Heads, InactiveHeadsContributeNothing) {
  Fixture f;
  Rng rng(2);
  auto z = rand_t({4, 9}, rng);
  auto d1 = head_dist(z, f.L, f.m);
  auto lp1 = log_prob(*d1, f.L, f.m, f.actions);
  auto e1 = entropy(*d1, f.L, f.m);
  // scribble over every inactive head's logits
  auto z2 = z.clone();
  auto p = z2->host_span<float>();
  const int off[3] = {0, 3, 7}, sz[3] = {3, 4, 2};
  auto act = f.m.active.host_span<std::uint8_t>();
  for (int b = 0; b < 4; ++b)
    for (int h = 0; h < 3; ++h)
      if (!(*act)[b * 3 + h])
        for (int j = 0; j < sz[h]; ++j) (*p)[b * 9 + off[h] + j] = 1e4f * (j + 1);
  auto d2 = head_dist(*z2, f.L, f.m);
  auto lp2 = log_prob(*d2, f.L, f.m, f.actions);
  auto e2 = entropy(*d2, f.L, f.m);
  ASSERT_TRUE(lp2 && e2);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ((*lp1->host_span<float>())[i], (*lp2->host_span<float>())[i]);
    EXPECT_EQ((*e1->host_span<float>())[i], (*e2->host_span<float>())[i]);
  }
}

TEST(Heads, UniformEntropyAndLogProb) {
  auto L = *HeadLayout::make({4});
  HeadMasks m{mk_u8({1, 4}, {1, 1, 0, 1}), mk_u8({1, 1}, {1})};
  auto d = head_dist(make_t({1, 4}, {0, 0, 0, 0}), L, m);
  ASSERT_TRUE(d);
  auto e = entropy(*d, L, m);
  EXPECT_NEAR((*e->host_span<float>())[0], std::log(3.0), 1e-6);
  auto lp = log_prob(*d, L, m, mk_i32({1, 1}, {1}));
  EXPECT_NEAR((*lp->host_span<float>())[0], -std::log(3.0), 1e-6);
}

TEST(Heads, SamplingRespectsMasksAndProbs) {
  auto L = *HeadLayout::make({4, 2});
  HeadMasks m{mk_u8({1, 6}, {1, 0, 1, 1, 1, 1}), mk_u8({1, 2}, {1, 0})};
  auto d = head_dist(make_t({1, 6}, {1.f, 9.f, 0.f, -1.f, 0.f, 0.f}), L, m);
  ASSERT_TRUE(d);
  Rng rng(3);
  std::map<int, int> hist;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    auto a = sample_actions(*d, L, m, &rng);
    ASSERT_TRUE(a);
    auto p = a->host_span<std::int32_t>();
    ASSERT_NE((*p)[0], 1);   // illegal despite the huge logit
    ASSERT_EQ((*p)[1], -1);  // inactive head
    hist[(*p)[0]]++;
  }
  auto pr = d->prob.host_span<float>();
  for (int j : {0, 2, 3}) EXPECT_NEAR(double(hist[j]) / n, (*pr)[j], 0.015) << j;
  auto g = sample_actions(*d, L, m, nullptr, true);
  ASSERT_TRUE(g);
  EXPECT_EQ((*g->host_span<std::int32_t>())[0], 0);

  Rng r1(77), r2(77);
  for (int i = 0; i < 50; ++i)
    EXPECT_EQ((*sample_actions(*d, L, m, &r1)->host_span<std::int32_t>())[0],
              (*sample_actions(*d, L, m, &r2)->host_span<std::int32_t>())[0]);
  EXPECT_FALSE(sample_actions(*d, L, m, nullptr));  // sampling without an Rng
}

TEST(Heads, Errors) {
  auto L = *HeadLayout::make({3});
  Rng rng(4);
  auto z = rand_t({1, 3}, rng);
  EXPECT_FALSE(head_dist(z, L, HeadMasks{mk_u8({1, 3}, {0, 0, 0}), mk_u8({1, 1}, {1})}));  // none legal
  EXPECT_TRUE(head_dist(z, L, HeadMasks{mk_u8({1, 3}, {0, 0, 0}), mk_u8({1, 1}, {0})}));   // but inactive
  HeadMasks m{mk_u8({1, 3}, {1, 0, 1}), mk_u8({1, 1}, {1})};
  auto d = head_dist(z, L, m);
  EXPECT_FALSE(log_prob(*d, L, m, mk_i32({1, 1}, {1})));  // illegal action
  EXPECT_FALSE(log_prob(*d, L, m, mk_i32({1, 1}, {3})));  // out of range
  EXPECT_FALSE(HeadLayout::make({}));
  EXPECT_FALSE(HeadLayout::make({2, 0}));
  EXPECT_FALSE(head_dist(rand_t({1, 4}, rng), L, m));  // wrong logit width
}

namespace {
PolicyConfig small_policy_cfg() {
  PolicyConfig c;
  c.encoder.feat_dim = 3;
  c.encoder.global_dim = 2;
  c.encoder.max_entities = 4;
  c.encoder.d_model = 8;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 16;
  c.encoder.out_dim = 6;
  c.head_sizes = {3, 2};
  return c;
}
std::filesystem::path rl_tmp(const char* name) {
  return std::filesystem::temp_directory_path() / (std::string("gyre-rlpol-") + name);
}
}  // namespace

// n_layer=0 skips SetBlock's n_head/d_ff checks, so training accepts configs that
// gyre-infer rejects with "invalid config". The exporter must refuse them.
TEST(PolicyIo, ExportRejectsConfigsGyreInferCannotLoad) {
  auto d = Device::cpu();
  for (int mut = 0; mut < 4; ++mut) {
    auto c = small_policy_cfg();
    c.encoder.n_layer = 0;
    if (mut == 0) c.encoder.n_head = 0;
    if (mut == 1) c.encoder.d_ff = 0;
    if (mut == 2) c.encoder.max_entities = 0;
    if (mut == 3) c.encoder.n_head = 3;  // d_model 8 % 3 != 0
    Rng rng(50 + mut);
    auto net = PolicyNet::create(c, *d, rng);
    ASSERT_TRUE(net) << "training accepts mutation " << mut << ": " << net.error().message;
    auto r = export_policy_flat(rl_tmp("bad.gpol"), *net);
    ASSERT_FALSE(r) << "exported a GPOL1 file gyre-infer refuses (mutation " << mut << ")";
    EXPECT_EQ(r.error().code, Errc::unsupported);
  }
  auto c = small_policy_cfg();
  c.encoder.n_layer = 0;  // valid n_layer=0 net still exports
  Rng rng(60);
  auto net = PolicyNet::create(c, *d, rng);
  ASSERT_TRUE(net);
  EXPECT_TRUE(export_policy_flat(rl_tmp("ok0.gpol"), *net));
}

TEST(PolicyIo, ExportDirectoryErrorIsAResultNotAThrow) {
  auto d = Device::cpu();
  Rng rng(61);
  auto net = PolicyNet::create(small_policy_cfg(), *d, rng);
  ASSERT_TRUE(net);
  const auto file = rl_tmp("plain-file");
  { std::ofstream(file) << "x"; }
  Result<void> r;
  EXPECT_NO_THROW(r = export_policy_flat(file / "sub" / "p.gpol", *net));
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().code, Errc::io);
  std::filesystem::remove(file);
}

TEST(PolicyIo, LoadRejectsDimsThatDisagreeWithStoredWeights) {
  auto d = Device::cpu();
  Rng rng(62);
  auto net = PolicyNet::create(small_policy_cfg(), *d, rng);
  ASSERT_TRUE(net);
  const auto good = rl_tmp("good.gyre");
  ASSERT_TRUE(save_policy(good, *net));
  ASSERT_TRUE(load_policy(good, *d));
  // Same small weights, but a config claiming d_ff = 2^30 (would be terabytes).
  GyreDoc doc;
  doc.arch = "policy-net";
  doc.config_json =
      R"({"feat_dim":3,"global_dim":2,"max_entities":4,"d_model":8,"n_layer":1,"n_head":2,)"
      R"("d_ff":1073741824,"out_dim":6,"gru_hidden":0,"head_sizes":[3,2]})";
  const auto bad = rl_tmp("bad-dims.gyre");
  ASSERT_TRUE(save_gyre(bad, net->parameters(), nullptr, doc));
  Result<PolicyNet> back = std::unexpected(Error{});
  EXPECT_NO_THROW(back = load_policy(bad, *d));
  ASSERT_FALSE(back);
  EXPECT_EQ(back.error().code, Errc::ckpt_corrupt);
  doc.config_json =
      R"({"feat_dim":3,"global_dim":2,"max_entities":4,"d_model":-8,"n_layer":1,"n_head":2,)"
      R"("d_ff":16,"out_dim":6,"gru_hidden":0,"head_sizes":[3,2]})";
  ASSERT_TRUE(save_gyre(bad, net->parameters(), nullptr, doc));
  EXPECT_FALSE(load_policy(bad, *d));
  std::filesystem::remove(good);
  std::filesystem::remove(bad);
}

// A HeadDist from another minibatch (B=1) used with B=4 masks used to be read
// past its end; prob/logp must match the masks' [B,total].
TEST(Heads, DistShapeMustMatchMasks) {
  Fixture f;
  Rng rng(41);
  auto L = f.L;
  HeadMasks small{mk_u8({1, 9}, {1, 1, 1, 1, 1, 1, 1, 1, 1}), mk_u8({1, 3}, {1, 1, 1})};
  auto d1 = head_dist(rand_t({1, 9}, rng), L, small);
  ASSERT_TRUE(d1);
  auto dl = make_t({4}, {1.f, 1.f, 1.f, 1.f});
  auto expect_shape_err = [](const auto& r, const char* what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().code, Errc::invalid_shape) << what;
  };
  expect_shape_err(sample_actions(*d1, L, f.m, &rng), "sample_actions");
  expect_shape_err(sample_actions(*d1, L, f.m, nullptr, true), "sample_actions greedy");
  expect_shape_err(log_prob(*d1, L, f.m, f.actions), "log_prob");
  expect_shape_err(entropy(*d1, L, f.m), "entropy");
  expect_shape_err(log_prob_entropy_backward(*d1, L, f.m, f.actions, dl, dl), "backward");

  // logp and prob that disagree with each other (one right, one wrong).
  auto d4 = head_dist(rand_t({4, 9}, rng), L, f.m);
  ASSERT_TRUE(d4);
  HeadDist mixed{d4->logp, d1->prob};
  expect_shape_err(entropy(mixed, L, f.m), "entropy mixed");
  expect_shape_err(log_prob_entropy_backward(mixed, L, f.m, f.actions, dl, dl), "backward mixed");
  EXPECT_TRUE(log_prob(mixed, L, f.m, f.actions));  // only reads logp

  // dlogp/dent must be f32.
  auto di = mk_i32({4}, {1, 1, 1, 1});
  expect_shape_err(log_prob_entropy_backward(*d4, L, f.m, f.actions, di, dl), "dlogp i32");
  EXPECT_TRUE(log_prob_entropy_backward(*d4, L, f.m, f.actions, dl, dl));
}

TEST(Ppo, GradCheckLogitsAndValue) {
  Fixture f;
  Rng rng(5);
  auto logits = make_p(rand_t({4, 9}, rng));
  auto value = make_p(rand_t({4}, rng));
  PpoConfig cfg;
  cfg.ent_coef = 0.05f;
  // old_logp = current logp + offset -> r = exp(-offset): unclipped and clipped on both sides.
  auto d0 = head_dist(logits.value, f.L, f.m);
  auto lp0 = log_prob(*d0, f.L, f.m, f.actions);
  std::vector<float> old(4);
  const float off[4] = {0.05f, -0.4f, 0.35f, -0.1f};
  for (int i = 0; i < 4; ++i) old[i] = (*lp0->host_span<float>())[i] + off[i];
  PpoBatch batch{f.actions, make_t({4}, old), make_t({4}, {1.2f, 0.8f, -0.9f, -1.5f}),
                 make_t({4}, {0.3f, -0.5f, 1.0f, 0.2f})};
  std::vector<Param> ps{logits, value};
  auto fn = [&](bool backward) -> Result<double> {
    auto l = ppo_loss(ps[0].value, ps[1].value, f.L, f.m, batch, cfg);
    if (!l) return std::unexpected(l.error());
    if (backward) {
      set_grad(ps[0], l->dlogits);
      set_grad(ps[1], l->dvalue);
    }
    return double(l->total);
  };
  GradCheckOpts o;
  o.h = 1e-3;
  o.max_per_param = 36;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
}

TEST(Ppo, ClippingSemantics) {
  auto L = *HeadLayout::make({3});
  HeadMasks m{mk_u8({3, 3}, std::vector<std::uint8_t>(9, 1)), mk_u8({3, 1}, {1, 1, 1})};
  auto logits = make_t({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});  // p = 1/3
  auto acts = mk_i32({3, 1}, {0, 1, 2});
  const float lp = -std::log(3.f);
  // sample 0: r = 1 (A>0). sample 1: A>0 and r far above 1+eps (clipped -> zero grad).
  // sample 2: A<0 and r far above 1+eps (min picks unclipped -> gradient flows).
  auto old = make_t({3}, {lp, lp - 1.f, lp - 1.f});
  PpoBatch b{acts, old, make_t({3}, {2.f, 2.f, -2.f}), make_t({3}, {0, 0, 0})};
  PpoConfig cfg;
  cfg.ent_coef = 0.f;
  auto l = ppo_loss(logits, make_t({3}, {0, 0, 0}), L, m, b, cfg);
  ASSERT_TRUE(l);
  auto g = l->dlogits.host_span<float>();
  const float p = 1.f / 3.f;
  // sample 0: dL/dz_j = -A*r*(1[j==a]-p)/B
  EXPECT_NEAR((*g)[0], -2.f * (1 - p) / 3.f, 1e-6);
  EXPECT_NEAR((*g)[1], -2.f * (0 - p) / 3.f, 1e-6);
  for (int j = 3; j < 6; ++j) EXPECT_EQ((*g)[j], 0.f);  // clipped
  const float r = std::exp(1.f);
  EXPECT_NEAR((*g)[6 + 2], 2.f * r * (1 - p) / 3.f, 1e-5);  // A<0: -A*r with A=-2
  EXPECT_NEAR(l->clip_frac, 2.f / 3.f, 1e-6);
  EXPECT_GT(l->approx_kl, 0.f);
  // r == 1 everywhere -> KL 0, no clipping.
  PpoBatch same{acts, make_t({3}, {lp, lp, lp}), b.advantages, b.returns};
  auto l2 = ppo_loss(logits, make_t({3}, {0, 0, 0}), L, m, same, cfg);
  EXPECT_NEAR(l2->approx_kl, 0.f, 1e-7);
  EXPECT_EQ(l2->clip_frac, 0.f);
}

TEST(Ppo, NormalizeAdvantages) {
  Rng rng(6);
  auto a = make_t({100}, std::vector<float>(100, 0.f));
  auto p = a.host_span<float>();
  for (auto& v : *p) v = rng.normal(3.f, 5.f);
  ASSERT_TRUE(normalize_advantages_(a));
  double mean = 0, var = 0;
  for (float v : *p) mean += v;
  mean /= 100;
  for (float v : *p) var += (v - mean) * (v - mean);
  EXPECT_NEAR(mean, 0.0, 1e-5);
  EXPECT_NEAR(var / 100, 1.0, 1e-4);
  auto c = make_t({4}, {2, 2, 2, 2});  // constant -> zeros, no NaN
  ASSERT_TRUE(normalize_advantages_(c));
  for (float v : *c.host_span<float>()) EXPECT_EQ(v, 0.f);
}

TEST(Imitation, GradCheck) {
  Fixture f;
  Rng rng(7);
  auto logits = make_p(rand_t({4, 9}, rng));
  std::vector<Param> ps{logits};
  auto fn = [&](bool backward) -> Result<double> {
    auto l = imitation_loss(ps[0].value, f.L, f.m, f.actions);
    if (!l) return std::unexpected(l.error());
    if (backward) set_grad(ps[0], l->dlogits);
    return double(l->loss);
  };
  GradCheckOpts o;
  o.h = 2e-3;
  o.max_per_param = 36;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
}

namespace {
PolicyConfig tiny_policy() {
  PolicyConfig c;
  c.encoder.feat_dim = 4;
  c.encoder.global_dim = 2;
  c.encoder.max_entities = 8;
  c.encoder.d_model = 8;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 12;
  c.encoder.out_dim = 10;
  c.head_sizes = {3, 4, 2};
  return c;
}
}  // namespace

TEST(Policy, EndToEndGradCheckPpo) {
  Fixture f;
  Rng rng(8);
  auto d = Device::cpu();
  auto net = PolicyNet::create(tiny_policy(), *d, rng);
  ASSERT_TRUE(net);
  // give the (tiny-init) heads real signal
  for (auto& p : net->parameters()) {
    auto v = p.value.host_span<float>();
    for (auto& x : *v) x += rng.normal(0.f, 0.2f);
  }
  const std::int64_t B = 4, N = 5;
  Obs obs{rand_t({B, N, 4}, rng), mk_u8({B, N}, {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}),
          rand_t({B, 2}, rng)};
  PpoBatch batch{f.actions, make_t({B}, {-3.f, -2.f, -4.f, -1.f}), make_t({B}, {1.f, -1.f, 0.5f, -0.5f}),
                 make_t({B}, {0.1f, 0.2f, -0.3f, 0.4f})};
  PpoConfig cfg;
  std::vector<Param> ps;
  for (auto& p : net->parameters()) ps.push_back(Param{p.value, p.grad});
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    ForwardCtx ctx;
    ctx.train = true;
    auto out = net->forward(obs, ctx);
    if (!out) return std::unexpected(out.error());
    auto l = ppo_loss(out->logits, out->value, net->layout(), f.m, batch, cfg);
    if (!l) return std::unexpected(l.error());
    if (backward) {
      auto r = net->backward(l->dlogits, l->dvalue, ctx);
      if (!r) return std::unexpected(r.error());
    }
    return double(l->total);
  };
  GradCheckOpts o;
  o.h = 3e-3;
  o.rtol = 6e-2;
  o.atol = 1e-3;
  o.max_per_param = 8;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
}

TEST(Policy, ImitationMemorisesDemonstrations) {
  set_deterministic(true);
  Rng rng(9);
  auto d = Device::cpu();
  auto net = PolicyNet::create(tiny_policy(), *d, rng);
  ASSERT_TRUE(net);
  const std::int64_t B = 16, N = 6;
  Obs obs{rand_t({B, N, 4}, rng), mk_u8({B, N}, std::vector<std::uint8_t>(B * N, 1)), rand_t({B, 2}, rng)};
  auto L = net->layout();
  std::vector<std::uint8_t> legal(B * 9, 1), active(B * 3, 1);
  std::vector<std::int32_t> acts(B * 3);
  for (int b = 0; b < B; ++b) {
    acts[b * 3 + 0] = int(rng.u32(3));
    acts[b * 3 + 1] = int(rng.u32(4));
    acts[b * 3 + 2] = int(rng.u32(2));
    if (b % 4 == 3) active[b * 3 + 1] = 0;  // some samples have an irrelevant head
  }
  HeadMasks m{mk_u8({B, 9}, legal), mk_u8({B, 3}, active)};
  auto actions = mk_i32({B, 3}, acts);
  auto adam = Adam::create(net->parameters(), 3e-3f);
  ASSERT_TRUE(adam);
  float first = 0, last = 0;
  for (int step = 0; step < 150; ++step) {
    ASSERT_TRUE(net->zero_grad());
    ForwardCtx ctx;
    ctx.train = true;
    auto out = net->forward(obs, ctx);
    ASSERT_TRUE(out);
    auto l = imitation_loss(out->logits, L, m, actions);
    ASSERT_TRUE(l);
    if (step == 0) first = l->loss;
    last = l->loss;
    auto dv = Tensor::zeros(out->value.shape(), DType::f32, out->value.device());
    ASSERT_TRUE(net->backward(l->dlogits, *dv, ctx));
    ASSERT_TRUE(clip_grad_norm(net->parameters(), 5.f));
    ASSERT_TRUE(adam->step(net->parameters()));
  }
  EXPECT_GT(first, 2.0f);  // ~ log(3)+log(4)+log(2) at init
  EXPECT_LT(last, 0.3f * first) << "first " << first << " last " << last;
}
