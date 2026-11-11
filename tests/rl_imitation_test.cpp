#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/imitation.hpp"

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
Tensor mk_i32(std::vector<std::int64_t> shape, const std::vector<std::int32_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(v)), shape, DType::i32, *d);
  if (!t) std::abort();
  return std::move(*t);
}

PolicyConfig student(std::vector<std::int64_t> heads, std::int64_t gru) {
  PolicyConfig c;
  c.encoder.feat_dim = 2;
  c.encoder.global_dim = 4;
  c.encoder.max_entities = 4;
  c.encoder.d_model = 16;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 32;
  c.encoder.out_dim = 32;
  c.head_sizes = std::move(heads);
  c.gru_hidden = gru;
  return c;
}

// Rule-based expert over a feedforward observation:
//   intent = 1 if the sum of valid entity feature 0 is positive, else 0     (head 0, 2 options)
//   option = floor(3 * global[0])                                            (head 1, only if intent==1)
// reward = global[1] (a known quantity the value head can regress).
RolloutBuffer rule_demos(std::uint64_t seed, std::int64_t E, std::int64_t T) {
  Rng rng(seed);
  RolloutSpec spec;
  spec.steps = T;
  spec.envs = E;
  spec.max_entities = 4;
  spec.feat_dim = 2;
  spec.global_dim = 4;
  spec.layout = *HeadLayout::make({2, 3});
  auto buf = RolloutBuffer::create(spec);
  for (std::int64_t t = 0; t < T; ++t) {
    std::vector<float> ent(E * 8), glob(E * 4);
    std::vector<std::uint8_t> mask(E * 4, 0), active(E * 2, 1);
    std::vector<std::int32_t> act(E * 2, -1);
    std::vector<float> rew(E);
    for (auto& x : ent) x = rng.normal(0.f, 1.f);
    for (auto& x : glob) x = rng.uniform01();
    for (std::int64_t e = 0; e < E; ++e) {
      const int n = 1 + int(rng.u32(4));
      float s = 0;
      for (int i = 0; i < n; ++i) {
        mask[e * 4 + i] = 1;
        s += ent[e * 8 + i * 2];
      }
      const int intent = s > 0 ? 1 : 0;
      act[e * 2] = intent;
      if (intent == 1) act[e * 2 + 1] = std::min(2, int(3 * glob[e * 4]));
      else active[e * 2 + 1] = 0;
      rew[e] = glob[e * 4 + 1];
    }
    Obs obs{make_t({E, 4, 2}, ent), mk_u8({E, 4}, mask), make_t({E, 4}, glob)};
    auto rec = make_demo_step(obs, mk_u8({E, 5}, std::vector<std::uint8_t>(E * 5, 1)), mk_u8({E, 2}, active),
                              mk_i32({E, 2}, act));
    if (!rec) std::abort();
    rec->reward = rew;
    if (!buf->push(*rec)) std::abort();
  }
  std::vector<float> zero(E, 0.f);
  if (!buf->compute_gae(zero, 0.f, 0.f)) std::abort();  // returns == rewards
  return std::move(*buf);
}

// Memory demos: streams of two 4-step episodes. Cue visible at the episode's first step, an
// "answer" flag at its last; the expert repeats the cue there. Only the answer step is active.
RolloutBuffer memory_demos(std::uint64_t seed, std::int64_t E, std::int64_t H) {
  Rng rng(seed);
  const std::int64_t T = 8, L = 4;
  RolloutSpec spec;
  spec.steps = T;
  spec.envs = E;
  spec.max_entities = 2;
  spec.feat_dim = 2;
  spec.global_dim = 4;
  spec.hidden_dim = H;
  spec.layout = *HeadLayout::make({3});
  auto buf = RolloutBuffer::create(spec);
  std::vector<int> cue(E * 2);
  for (auto& c : cue) c = int(rng.u32(3));
  Tensor h0 = *Tensor::zeros(std::vector<std::int64_t>{E, H}, DType::f32, *Device::cpu());
  for (std::int64_t t = 0; t < T; ++t) {
    const int ph = int(t % L), ep = int(t / L);
    std::vector<float> ent(E * 4), glob(E * 4, 0.f);
    std::vector<std::uint8_t> active(E, ph == L - 1 ? 1 : 0);
    std::vector<std::int32_t> act(E, -1);
    for (auto& x : ent) x = rng.normal(0.f, 1.f);
    for (std::int64_t e = 0; e < E; ++e) {
      if (ph == 0) glob[e * 4 + cue[e * 2 + ep]] = 1.f;
      if (ph == L - 1) {
        glob[e * 4 + 3] = 1.f;
        act[e] = cue[e * 2 + ep];
      }
    }
    Obs obs{make_t({E, 2, 2}, ent), mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)), make_t({E, 4}, glob)};
    auto rec = make_demo_step(obs, mk_u8({E, 3}, std::vector<std::uint8_t>(E * 3, 1)), mk_u8({E, 1}, active),
                              mk_i32({E, 1}, act), std::vector<std::uint8_t>(E, ph == 0 ? 1 : 0), h0);
    if (!rec || !buf->push(*rec)) std::abort();
  }
  std::vector<float> zero(E, 0.f);
  if (!buf->compute_gae(zero, 0.f, 0.f)) std::abort();
  return std::move(*buf);
}

}  // namespace

TEST(Imitation, FeedforwardLearnsRuleAndGeneralises) {
  set_deterministic(true);
  Rng rng(1);
  auto d = Device::cpu();
  auto net = PolicyNet::create(student({2, 3}, 0), *d, rng);
  ASSERT_TRUE(net);
  auto train = rule_demos(10, 64, 16);
  auto held = rule_demos(11, 16, 16);
  ImitationConfig cfg;
  cfg.lr = 3e-3f;
  cfg.minibatches = 8;
  cfg.value_coef = 0.5f;
  auto tr = ImitationTrainer::create(*net, cfg);
  ASSERT_TRUE(tr);
  auto before = tr->evaluate(held);
  ASSERT_TRUE(before);
  EXPECT_GT(before->loss, 0.8f);
  EXPECT_LT(before->accuracy, 0.7f);
  for (int ep = 0; ep < 30; ++ep) {
    auto st = tr->epoch(train);
    ASSERT_TRUE(st) << st.error().message;
  }
  auto after = tr->evaluate(held);
  ASSERT_TRUE(after);
  EXPECT_GT(after->accuracy, 0.9f) << "loss " << after->loss;
  EXPECT_LT(after->loss, 0.5f * before->loss);
  EXPECT_LT(after->value_loss, 0.01f) << "value head should learn reward = global[1]; before " << before->value_loss;
}

TEST(Imitation, EvaluateDoesNotTouchParameters) {
  Rng rng(2);
  auto d = Device::cpu();
  auto net = PolicyNet::create(student({2, 3}, 0), *d, rng);
  ASSERT_TRUE(net);
  std::vector<float> snap;
  for (auto& p : net->parameters()) {
    auto v = p.value.host_span<float>();
    snap.insert(snap.end(), v->begin(), v->end());
  }
  auto tr = ImitationTrainer::create(*net, ImitationConfig{});
  auto data = rule_demos(12, 8, 4);
  ASSERT_TRUE(tr->evaluate(data));
  std::size_t k = 0;
  for (auto& p : net->parameters())
    for (float v : *p.value.host_span<float>()) ASSERT_EQ(v, snap[k++]);
}

TEST(Imitation, GruLearnsMemoryFromDemonstrations) {
  set_deterministic(true);
  Rng rng(3);
  auto d = Device::cpu();
  auto net = PolicyNet::create(student({3}, 32), *d, rng);
  ASSERT_TRUE(net);
  auto train = memory_demos(20, 64, 32);
  auto held = memory_demos(21, 32, 32);
  ImitationConfig cfg;
  cfg.lr = 3e-3f;
  cfg.minibatches = 4;
  auto tr = ImitationTrainer::create(*net, cfg);
  ASSERT_TRUE(tr);
  auto before = tr->evaluate(held);
  ASSERT_TRUE(before);
  for (int ep = 0; ep < 150; ++ep) {
    auto st = tr->epoch(train);
    ASSERT_TRUE(st) << st.error().message;
  }
  auto after = tr->evaluate(held);
  ASSERT_TRUE(after);
  // accuracy counts only the answer steps (everything else is inactive and carries no loss)
  EXPECT_LT(before->accuracy, 0.6f);
  EXPECT_GT(after->accuracy, 0.95f) << "before " << before->accuracy << " loss " << after->loss;
}

TEST(Imitation, RejectsRecurrenceMismatch) {
  Rng rng(4);
  auto d = Device::cpu();
  auto ff = PolicyNet::create(student({3}, 0), *d, rng);
  ASSERT_TRUE(ff);
  auto tr = ImitationTrainer::create(*ff, ImitationConfig{});
  auto data = memory_demos(22, 4, 8);  // recurrent buffer
  EXPECT_FALSE(tr->epoch(data));
  EXPECT_FALSE(ImitationTrainer::create(*ff, ImitationConfig{.minibatches = 0}));
}

TEST(Imitation, PretrainedNetContinuesIntoPpo) {
  // The same PolicyNet object moves from supervised pretraining to PPO without conversion.
  set_deterministic(true);
  Rng rng(5);
  auto d = Device::cpu();
  auto net = PolicyNet::create(student({2, 3}, 0), *d, rng);
  ASSERT_TRUE(net);
  auto tr = ImitationTrainer::create(*net, ImitationConfig{.lr = 3e-3f, .minibatches = 8});
  auto demos = rule_demos(30, 32, 8);
  for (int i = 0; i < 5; ++i) ASSERT_TRUE(tr->epoch(demos));
  auto ppo = PpoTrainer::create(*net, PpoTrainConfig{});
  ASSERT_TRUE(ppo);
  // a PPO rollout of the right shape on the same net
  RolloutSpec spec = demos.spec();
  auto buf = RolloutBuffer::create(spec);
  const std::int64_t E = spec.envs;
  auto legal = mk_u8({E, 5}, std::vector<std::uint8_t>(E * 5, 1));
  auto active = mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1));
  for (int t = 0; t < spec.steps; ++t) {
    std::vector<float> ent(E * 8), glob(E * 4);
    for (auto& x : ent) x = rng.normal(0.f, 1.f);
    for (auto& x : glob) x = rng.uniform01();
    Obs obs{make_t({E, 4, 2}, ent), mk_u8({E, 4}, std::vector<std::uint8_t>(E * 4, 1)), make_t({E, 4}, glob)};
    auto a = policy_step(*net, obs, legal, active, &rng, false,
                         [](std::int64_t, std::int64_t h, std::span<const std::int32_t> acts, std::span<std::uint8_t> act) {
                           if (h == 0 && acts[0] == 0) act[1] = 0;
                         });
    ASSERT_TRUE(a);
    StepRecord r{obs, legal, a->active, a->actions, a->logp, a->value, std::vector<float>(E, 1.f),
                 std::vector<std::uint8_t>(E, 0)};
    ASSERT_TRUE(buf->push(r));
  }
  std::vector<float> lv(E, 0.f);
  ASSERT_TRUE(buf->compute_gae(lv, 0.99f, 0.95f));
  auto st = ppo->update(*buf);
  ASSERT_TRUE(st) << st.error().message;
  EXPECT_TRUE(std::isfinite(st->total));
}
