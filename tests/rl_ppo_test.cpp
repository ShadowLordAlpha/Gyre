#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/ppo.hpp"

#include <gtest/gtest.h>

#include <cmath>
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
Tensor mk_i32(std::vector<std::int64_t> shape, const std::vector<std::int32_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(v)), shape, DType::i32, *d);
  if (!t) std::abort();
  return std::move(*t);
}

RolloutSpec small_spec(std::int64_t steps, std::int64_t envs) {
  RolloutSpec s;
  s.steps = steps;
  s.envs = envs;
  s.max_entities = 2;
  s.feat_dim = 1;
  s.global_dim = 1;
  s.layout = *HeadLayout::make({2});
  return s;
}

// Record whose entity feature encodes (t, e) so gathers can be verified.
StepRecord rec(std::int64_t E, float tag, std::vector<float> reward, std::vector<std::uint8_t> done,
               std::vector<float> value) {
  std::vector<float> ent(E * 2), lp(E, -0.5f);
  for (std::int64_t e = 0; e < E; ++e) {
    ent[e * 2] = tag * 10 + float(e);
    ent[e * 2 + 1] = -1.f;
  }
  return StepRecord{Obs{make_t({E, 2, 1}, ent), mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)),
                        make_t({E, 1}, std::vector<float>(E, 0.f))},
                    mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)),
                    mk_u8({E, 1}, std::vector<std::uint8_t>(E, 1)),
                    mk_i32({E, 1}, std::vector<std::int32_t>(E, 0)),
                    make_t({E}, lp),
                    make_t({E}, value),
                    std::move(reward),
                    std::move(done)};
}

}  // namespace

TEST(Rollout, GaeHandComputed) {
  auto buf = RolloutBuffer::create(small_spec(3, 2));
  ASSERT_TRUE(buf);
  // env0 has the interesting data; env1 is all zeros and must stay independent.
  const float R[3] = {1.f, 0.f, 2.f}, V[3] = {0.5f, 1.0f, 0.2f};
  const std::uint8_t D[3] = {0, 0, 1};
  for (int t = 0; t < 3; ++t)
    ASSERT_TRUE(buf->push(rec(2, float(t), {R[t], 0.f}, {D[t], 0}, {V[t], 0.f})));
  EXPECT_TRUE(buf->full());
  EXPECT_FALSE(buf->push(rec(2, 9, {0, 0}, {0, 0}, {0, 0})));  // full
  const float last[2] = {3.0f, 0.f};  // ignored for env0 (done at its last step)
  ASSERT_TRUE(buf->compute_gae(last, 0.9f, 0.8f));
  auto adv = buf->advantages();
  auto ret = buf->returns();
  // t=2: 1.8 ; t=1: -0.82 + 0.72*1.8 = 0.476 ; t=0: 1.4 + 0.72*0.476 = 1.74272
  EXPECT_NEAR(adv[2 * 2 + 0], 1.8f, 1e-5);
  EXPECT_NEAR(adv[1 * 2 + 0], 0.476f, 1e-5);
  EXPECT_NEAR(adv[0 * 2 + 0], 1.74272f, 1e-5);
  EXPECT_NEAR(ret[0 * 2 + 0], 1.74272f + 0.5f, 1e-5);
  for (int t = 0; t < 3; ++t) EXPECT_EQ(adv[t * 2 + 1], 0.f);
}

TEST(Rollout, GaeStopsAtEpisodeBoundaryAndBootstraps) {
  auto buf = RolloutBuffer::create(small_spec(3, 1));
  ASSERT_TRUE(buf);
  const float R[3] = {1.f, 0.f, 2.f}, V[3] = {0.5f, 1.0f, 0.2f};
  const std::uint8_t D[3] = {0, 1, 0};
  for (int t = 0; t < 3; ++t) ASSERT_TRUE(buf->push(rec(1, float(t), {R[t]}, {D[t]}, {V[t]})));
  const float last[1] = {2.0f};
  ASSERT_TRUE(buf->compute_gae(last, 0.9f, 0.8f));
  auto adv = buf->advantages();
  EXPECT_NEAR(adv[2], 3.6f, 1e-5);   // 2 + 0.9*2 - 0.2 : bootstraps from last_value
  EXPECT_NEAR(adv[1], -1.0f, 1e-5);  // done: no bootstrap, no carry from t=2
  EXPECT_NEAR(adv[0], 0.68f, 1e-5);  // 1.4 + 0.72*(-1)
}

TEST(Rollout, ValidationAndClear) {
  auto buf = RolloutBuffer::create(small_spec(2, 2));
  ASSERT_TRUE(buf);
  float last[2] = {0, 0};
  EXPECT_FALSE(buf->compute_gae(last, 0.9f, 0.9f));  // not full
  EXPECT_FALSE(buf->push(rec(3, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0})));  // wrong env count
  auto bad = rec(2, 0, {0}, {0, 0}, {0, 0});  // reward size mismatch
  EXPECT_FALSE(buf->push(bad));
  ASSERT_TRUE(buf->push(rec(2, 0, {0, 0}, {0, 0}, {0, 0})));
  ASSERT_TRUE(buf->push(rec(2, 1, {0, 0}, {0, 0}, {0, 0})));
  EXPECT_TRUE(buf->full());
  buf->clear();
  EXPECT_FALSE(buf->full());
  EXPECT_EQ(buf->steps_filled(), 0);
  EXPECT_FALSE(RolloutBuffer::create(small_spec(0, 2)));
}

TEST(Rollout, MinibatchGathersRequestedSamples) {
  auto buf = RolloutBuffer::create(small_spec(2, 3));
  ASSERT_TRUE(buf);
  ASSERT_TRUE(buf->push(rec(3, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0})));
  ASSERT_TRUE(buf->push(rec(3, 1, {0, 0, 0}, {0, 0, 0}, {0, 0, 0})));
  float last[3] = {0, 0, 0};
  ASSERT_TRUE(buf->compute_gae(last, 0.9f, 0.9f));
  const std::int64_t idx[3] = {5, 0, 4};  // (t=1,e=2) (t=0,e=0) (t=1,e=1)
  auto m = buf->minibatch(idx);
  ASSERT_TRUE(m);
  auto e = m->obs.entities.host_span<float>();
  EXPECT_EQ((*e)[0], 12.f);
  EXPECT_EQ((*e)[2], 0.f);
  EXPECT_EQ((*e)[4], 11.f);
  EXPECT_EQ(m->batch.actions.shape()[0], 3);
  const std::int64_t oob[1] = {6};
  EXPECT_FALSE(buf->minibatch(oob));
}

// ------------------------------------------------------------ learning test
//
// A deliberately game-free toy: each step shows a cue (one-hot in the global features) and
// a few noise entities (some padded). Head 0 = intent in {0,1}; head 1 = option in {0,1,2},
// relevant only when intent == 1 (the sampler switches it off otherwise). Reward 1 iff
// intent == cue and (intent == 0 or option == cue2). Legality of options is random but the
// right option is always legal. Chance performance is ~1/3; optimum is 1.

namespace {

struct ToyObs {
  Obs obs;
  Tensor legal;
  std::vector<int> cue, cue2;
};

ToyObs toy_obs(Rng& rng, std::int64_t E) {
  const std::int64_t N = 4, F = 3, G = 5, T = 5;
  std::vector<float> ent(E * N * F), glob(E * G, 0.f);
  std::vector<std::uint8_t> mask(E * N, 0), legal(E * T, 1);
  std::vector<int> cue(E), cue2(E);
  for (auto& x : ent) x = rng.normal(0.f, 1.f);
  for (std::int64_t e = 0; e < E; ++e) {
    const int n = 1 + int(rng.u32(N));
    for (int i = 0; i < n; ++i) mask[e * N + i] = 1;
    cue[e] = int(rng.u32(2));
    cue2[e] = int(rng.u32(3));
    glob[e * G + cue[e]] = 1.f;
    glob[e * G + 2 + cue2[e]] = 1.f;
    for (int o = 0; o < 3; ++o) legal[e * T + 2 + o] = (o == cue2[e] || rng.u32(2)) ? 1 : 0;
  }
  return ToyObs{Obs{make_t({E, N, F}, ent), mk_u8({E, N}, mask), make_t({E, G}, glob)},
                mk_u8({E, T}, legal), cue, cue2};
}

PolicyConfig toy_policy() {
  PolicyConfig c;
  c.encoder.feat_dim = 3;
  c.encoder.global_dim = 5;
  c.encoder.max_entities = 4;
  c.encoder.d_model = 16;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 32;
  c.encoder.out_dim = 32;
  c.head_sizes = {2, 3};
  return c;
}

const ActiveFn kIntentGate = [](std::int64_t, std::int64_t h, std::span<const std::int32_t> a,
                                std::span<std::uint8_t> active) {
  if (h == 0 && a[0] == 0) active[1] = 0;  // option head irrelevant to intent 0
};

struct ToyRun {
  std::vector<float> mean_reward;  // per iteration
  std::vector<float> params0;      // first parameter tensor after training
  int illegal_or_bad{0};
};

ToyRun run_toy(int iters, std::uint64_t seed) {
  ToyRun out;
  Rng rng(seed);
  auto d = Device::cpu();
  auto net = PolicyNet::create(toy_policy(), *d, rng);
  if (!net) std::abort();
  const std::int64_t E = 32, T = 4;
  RolloutSpec spec;
  spec.steps = T;
  spec.envs = E;
  spec.max_entities = 4;
  spec.feat_dim = 3;
  spec.global_dim = 5;
  spec.layout = net->layout();
  auto buf = RolloutBuffer::create(spec);
  PpoTrainConfig tc;
  tc.lr = 3e-3f;
  tc.epochs = 4;
  tc.minibatches = 4;
  tc.seed = seed + 1;
  auto tr = PpoTrainer::create(*net, tc);
  if (!buf || !tr) std::abort();
  auto ones = mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1));
  for (int it = 0; it < iters; ++it) {
    buf->clear();
    double total = 0;
    for (std::int64_t t = 0; t < T; ++t) {
      auto o = toy_obs(rng, E);
      auto a = policy_step(*net, o.obs, o.legal, ones, &rng, false, kIntentGate);
      if (!a) std::abort();
      auto pa = a->actions.host_span<std::int32_t>();
      auto pact = a->active.host_span<std::uint8_t>();
      auto pl = o.legal.host_span<std::uint8_t>();
      std::vector<float> rew(E);
      for (std::int64_t e = 0; e < E; ++e) {
        const int intent = (*pa)[e * 2], opt = (*pa)[e * 2 + 1];
        const bool gate_ok = intent == 1 ? (*pact)[e * 2 + 1] == 1 && opt >= 0 : (*pact)[e * 2 + 1] == 0 && opt == -1;
        if (!gate_ok || (intent == 1 && !(*pl)[e * 5 + 2 + opt])) ++out.illegal_or_bad;
        rew[e] = (intent == o.cue[e] && (intent == 0 || opt == o.cue2[e])) ? 1.f : 0.f;
        total += rew[e];
      }
      StepRecord r{o.obs, o.legal, a->active, a->actions, a->logp, a->value, rew,
                   std::vector<std::uint8_t>(E, 1)};
      if (!buf->push(r)) std::abort();
    }
    std::vector<float> lastv(E, 0.f);
    if (!buf->compute_gae(lastv, tc.gamma, tc.lambda)) std::abort();
    auto st = tr->update(*buf);
    if (!st) {
      ADD_FAILURE() << st.error().message;
      break;
    }
    out.mean_reward.push_back(float(total / double(E * T)));
  }
  auto p = net->parameters()[0].value.host_span<float>();
  out.params0.assign(p->begin(), p->end());
  return out;
}

}  // namespace

TEST(Ppo, StoredLogpMatchesTrainingForward) {
  // The behaviour log-prob saved at acting time must equal what the training forward
  // recomputes on the same data: first-step ratio is exactly 1 (KL ~ 0, nothing clipped).
  Rng rng(21);
  auto d = Device::cpu();
  auto net = PolicyNet::create(toy_policy(), *d, rng);
  ASSERT_TRUE(net);
  const std::int64_t E = 16;
  auto o = toy_obs(rng, E);
  auto ones = mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1));
  auto a = policy_step(*net, o.obs, o.legal, ones, &rng, false, kIntentGate);
  ASSERT_TRUE(a);
  ForwardCtx ctx;
  ctx.train = true;
  auto out = net->forward(o.obs, ctx);
  ASSERT_TRUE(out);
  HeadMasks m{o.legal, a->active};
  PpoBatch b{a->actions, a->logp, make_t({E}, std::vector<float>(E, 1.f)),
             make_t({E}, std::vector<float>(E, 0.f))};
  auto l = ppo_loss(out->logits, out->value, net->layout(), m, b, PpoConfig{});
  ASSERT_TRUE(l);
  EXPECT_NEAR(l->approx_kl, 0.f, 1e-9);
  EXPECT_EQ(l->clip_frac, 0.f);
}

TEST(Ppo, LearnsGatedToyTask) {
  set_deterministic(true);
  auto run = run_toy(200, 5);
  EXPECT_EQ(run.illegal_or_bad, 0) << "sampler produced an illegal option or broke the gate";
  const double first = run.mean_reward[0];  // untrained policy: ~1/3
  double last = 0;
  for (std::size_t i = run.mean_reward.size() - 10; i < run.mean_reward.size(); ++i) last += run.mean_reward[i];
  last /= 10;
  EXPECT_LT(first, 0.5) << "chance level should be ~1/3";
  EXPECT_GT(last, 0.85) << "first " << first << " last " << last;
}

TEST(Ppo, DeterministicGivenSeed) {
  set_deterministic(true);
  auto a = run_toy(8, 11);
  auto b = run_toy(8, 11);
  ASSERT_EQ(a.mean_reward.size(), b.mean_reward.size());
  EXPECT_EQ(0, std::memcmp(a.mean_reward.data(), b.mean_reward.data(), a.mean_reward.size() * sizeof(float)));
  ASSERT_EQ(a.params0.size(), b.params0.size());
  EXPECT_EQ(0, std::memcmp(a.params0.data(), b.params0.data(), a.params0.size() * sizeof(float)));
  auto c = run_toy(8, 12);
  EXPECT_NE(0, std::memcmp(a.params0.data(), c.params0.data(), a.params0.size() * sizeof(float)));
}

TEST(Ppo, ConditionalSamplingHonoursGate) {
  Rng rng(31);
  auto L = *HeadLayout::make({2, 3});
  auto legal = mk_u8({1, 5}, {1, 1, 1, 1, 1});
  auto init = mk_u8({1, 2}, {1, 1});
  int off = 0, on = 0;
  for (int i = 0; i < 500; ++i) {
    auto s = sample_conditional(make_t({1, 5}, {0, 0, 0, 0, 0}), L, legal, init, &rng, false, kIntentGate);
    ASSERT_TRUE(s);
    auto a = s->actions.host_span<std::int32_t>();
    auto f = s->active.host_span<std::uint8_t>();
    if ((*a)[0] == 0) {
      ++off;
      ASSERT_EQ((*a)[1], -1);
      ASSERT_EQ((*f)[1], 0);
    } else {
      ++on;
      ASSERT_GE((*a)[1], 0);
      ASSERT_EQ((*f)[1], 1);
    }
  }
  EXPECT_GT(off, 150);
  EXPECT_GT(on, 150);
}
