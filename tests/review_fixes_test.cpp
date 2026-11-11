// Regression tests for defects found by the independent (adversarial) review of the RL stack.
#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/imitation.hpp"
#include "gyre/rl/ppo.hpp"
#include "gyre/rl/set_encoder.hpp"
#include "gyre_infer/policy.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

using namespace gyre;
using namespace gyre::testing;

namespace {

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}

// ---- forged GPOL1 headers -------------------------------------------------------------
std::uint32_t fnv(const unsigned char* p, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
  return h;
}
template <class T>
void put(std::vector<unsigned char>& b, T v) {
  const auto* p = reinterpret_cast<const unsigned char*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}
std::vector<unsigned char> forge(const std::vector<std::uint32_t>& cfg10, const std::vector<std::uint32_t>& heads,
                                 std::uint64_t nfloats_declared, std::size_t floats_present) {
  std::vector<unsigned char> b;
  const char magic[8] = {'G', 'P', 'O', 'L', '1', 0, 0, 0};
  b.insert(b.end(), magic, magic + 8);
  put<std::uint32_t>(b, 1);
  put<std::uint32_t>(b, 0);
  for (auto v : cfg10) put<std::uint32_t>(b, v);
  for (auto h : heads) put<std::uint32_t>(b, h);
  put<std::uint64_t>(b, nfloats_declared);
  std::vector<float> w(floats_present, 0.1f);
  const auto* wp = reinterpret_cast<const unsigned char*>(w.data());
  b.insert(b.end(), wp, wp + w.size() * 4);
  put<std::uint32_t>(b, fnv(wp, w.size() * 4));
  return b;
}

}  // namespace

TEST(ReviewFixes, ForgedHeaderWithWrappingWeightCountIsRejected) {
  // feat=2^31-1, global=2^31-107, d=2^31-1, 1 layer, 1 head, ff=1, out=2^31-1, gru=0, one head of 106
  auto f = forge({0x7fffffffu, 0x7fffffffu - 106u, 4, 0x7fffffffu, 1, 1, 1, 0x7fffffffu, 0, 1}, {106}, 100, 100);
  std::string err;
  EXPECT_EQ(gyre_infer::Policy::load_from_memory(f.data(), f.size(), &err), nullptr);
  EXPECT_FALSE(err.empty());
}

TEST(ReviewFixes, HugeHeadSizesAndLayerCountsAreRejected) {
  std::string err;
  auto a = forge({2, 1, 4, 4, 1, 2, 8, 6, 0, 2}, {0xFFFFFFFFu, 2}, 11, 11);
  EXPECT_EQ(gyre_infer::Policy::load_from_memory(a.data(), a.size(), &err), nullptr);
  auto b = forge({2, 1, 4, 4, 1u << 30, 2, 8, 6, 0, 1}, {3}, 100, 100);  // forged n_layer
  EXPECT_EQ(gyre_infer::Policy::load_from_memory(b.data(), b.size(), &err), nullptr);
  auto c = forge({2, 1, 4, 4, 1, 2, 8, 6, 0, 1}, {1u << 21}, 100, 100);  // head larger than the cap
  EXPECT_EQ(gyre_infer::Policy::load_from_memory(c.data(), c.size(), &err), nullptr);
  auto d = forge({2, 1, 4, 4, 1, 3, 8, 6, 0, 1}, {3}, 100, 100);  // d_model % n_head != 0
  EXPECT_EQ(gyre_infer::Policy::load_from_memory(d.data(), d.size(), &err), nullptr);
}

TEST(ReviewFixes, NonFinitePaddedSlotsDoNotPoisonValidRows) {
  SetEncoderConfig c;
  c.feat_dim = 3;
  c.global_dim = 2;
  c.max_entities = 4;
  c.d_model = 8;
  c.n_layer = 2;
  c.n_head = 2;
  c.d_ff = 12;
  c.out_dim = 6;
  Rng rng(1);
  auto d = Device::cpu();
  auto enc = SetEncoder::create(c, *d, rng);
  ASSERT_TRUE(enc);
  auto clean = rand_t({1, 3, 3}, rng);
  auto dirty = clean.clone();
  auto pd = dirty->host_span<float>();
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  (*pd)[6] = nan;  // slot 2 is padded
  (*pd)[7] = inf;
  (*pd)[8] = 1.f;
  auto mask = mk_u8({1, 3}, {1, 1, 0});
  auto glob = rand_t({1, 2}, rng);
  auto cl = clean.clone();
  (*cl->host_span<float>())[6] = (*cl->host_span<float>())[7] = (*cl->host_span<float>())[8] = 0.f;

  ForwardCtx c1, c2;
  auto y_clean = enc->forward(*cl, mask, glob, c1);
  auto y_dirty = enc->forward(*dirty, mask, glob, c2);
  ASSERT_TRUE(y_clean && y_dirty);
  auto a = y_clean->host_span<float>(), b = y_dirty->host_span<float>();
  for (std::size_t i = 0; i < a->size(); ++i) {
    ASSERT_TRUE(std::isfinite((*b)[i]));
    EXPECT_EQ((*a)[i], (*b)[i]);
  }
  // backward: finite everywhere, exactly zero on the padded slot
  auto dy = rand_t({1, 6}, rng);
  ASSERT_TRUE(enc->zero_grad());
  auto g = enc->backward(dy, c2);
  ASSERT_TRUE(g);
  for (auto& p : enc->parameters())
    for (float v : *p.grad.host_span<float>()) ASSERT_TRUE(std::isfinite(v));
  auto pe = g->d_entities.host_span<float>();
  for (int f = 0; f < 3; ++f) EXPECT_EQ((*pe)[2 * 3 + f], 0.f);
}

namespace {
PolicyConfig mini(std::int64_t gru) {
  PolicyConfig c;
  c.encoder.feat_dim = 2;
  c.encoder.global_dim = 1;
  c.encoder.max_entities = 2;
  c.encoder.d_model = 8;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 8;
  c.encoder.out_dim = 8;
  c.head_sizes = {3};
  c.gru_hidden = gru;
  return c;
}
RolloutBuffer one_step_buffer(const HeadLayout& L, std::int64_t E, std::int64_t steps, Rng& rng) {
  RolloutSpec spec;
  spec.steps = steps;
  spec.envs = E;
  spec.max_entities = 2;
  spec.feat_dim = 2;
  spec.global_dim = 1;
  spec.layout = L;
  auto buf = RolloutBuffer::create(spec);
  for (std::int64_t t = 0; t < steps; ++t) {
    Obs obs{rand_t({E, 2, 2}, rng), mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)), rand_t({E, 1}, rng)};
    auto rec = make_demo_step(obs, mk_u8({E, 3}, std::vector<std::uint8_t>(E * 3, 1)),
                              mk_u8({E, 1}, std::vector<std::uint8_t>(E, 1)),
                              [&] {
                                std::vector<std::int32_t> a(E, 1);
                                auto d = Device::cpu();
                                std::int64_t sh[2] = {E, 1};
                                return std::move(*Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(a)), sh,
                                                                    DType::i32, *d));
                              }());
    if (!rec || !buf->push(*rec)) std::abort();
  }
  return std::move(*buf);
}
}  // namespace

TEST(ReviewFixes, TrainerRefusesStaleOrMissingAdvantages) {
  Rng rng(2);
  auto d = Device::cpu();
  auto net = PolicyNet::create(mini(0), *d, rng);
  ASSERT_TRUE(net);
  auto buf = one_step_buffer(net->layout(), 4, 2, rng);
  EXPECT_FALSE(buf.gae_computed());
  auto tr = PpoTrainer::create(*net, PpoTrainConfig{});
  ASSERT_TRUE(tr);
  EXPECT_FALSE(tr->update(buf)) << "update() must demand compute_gae";
  std::vector<float> lv(4, 0.f);
  ASSERT_TRUE(buf.compute_gae(lv, 0.9f, 0.9f));
  EXPECT_TRUE(buf.gae_computed());
  EXPECT_TRUE(tr->update(buf));
  // refill without recomputing: stale advantages must not be reused
  buf.clear();
  EXPECT_FALSE(buf.gae_computed());
  auto fresh = one_step_buffer(net->layout(), 4, 2, rng);
  EXPECT_FALSE(tr->update(fresh));

  // imitation with value_coef > 0 needs returns too
  auto it = ImitationTrainer::create(*net, ImitationConfig{.value_coef = 1.f});
  ASSERT_TRUE(it);
  EXPECT_FALSE(it->epoch(fresh));
  ASSERT_TRUE(fresh.compute_gae(lv, 0.f, 0.f));
  EXPECT_TRUE(it->epoch(fresh));
}

TEST(ReviewFixes, GateCanOnlyChangeLaterHeads) {
  Rng rng(3);
  auto L = *HeadLayout::make({2, 2, 2});
  auto legal = mk_u8({1, 6}, std::vector<std::uint8_t>(6, 1));
  auto logits = make_t({1, 6}, {0, 0, 0, 0, 0, 0});
  // head 0 initially off, head 1 on, head 2 on. The gate tries to (a) switch ON the earlier head 0
  // and (b) switch OFF the head that was just sampled; both edits must be ignored.
  auto init = mk_u8({1, 3}, {0, 1, 1});
  auto bad_gate = [](std::int64_t, std::int64_t h, std::span<const std::int32_t>, std::span<std::uint8_t> a) {
    if (h == 1) {
      a[0] = 1;  // re-activate a never-sampled earlier head
      a[1] = 0;  // deactivate the head just sampled
      a[2] = 0;  // legitimate: a later head
    }
  };
  auto s = sample_conditional(logits, L, legal, init, &rng, false, bad_gate);
  ASSERT_TRUE(s);
  auto act = s->active.host_span<std::uint8_t>();
  auto a = s->actions.host_span<std::int32_t>();
  EXPECT_EQ((*act)[0], 0);
  EXPECT_EQ((*a)[0], -1);
  EXPECT_EQ((*act)[1], 1);
  EXPECT_GE((*a)[1], 0);
  EXPECT_EQ((*act)[2], 0);
  EXPECT_EQ((*a)[2], -1);
  // the stored mask is consistent with log_prob (no "action out of range" error downstream)
  HeadMasks m{legal, s->active};
  auto dist = head_dist(logits, L, m);
  ASSERT_TRUE(dist);
  EXPECT_TRUE(log_prob(*dist, L, m, s->actions));
}

TEST(ReviewFixes, NonFiniteLogitsAreErrorsNotSilentNaN) {
  auto L = *HeadLayout::make({3, 2});
  auto legal = mk_u8({1, 5}, {1, 1, 0, 1, 1});
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  HeadMasks both{legal, mk_u8({1, 2}, {1, 1})};
  EXPECT_FALSE(head_dist(make_t({1, 5}, {inf, 0, 0, 0, 0}), L, both));
  EXPECT_FALSE(head_dist(make_t({1, 5}, {0, nan, 0, 0, 0}), L, both));
  Rng rng(4);
  EXPECT_FALSE(sample_conditional(make_t({1, 5}, {0, nan, 0, 0, 0}), L, legal, both.active, &rng, false));
  EXPECT_FALSE(sample_conditional(make_t({1, 5}, {inf, 0, 0, 0, 0}), L, legal, both.active, &rng, true));
  // NaN in an illegal option or in an inactive head is irrelevant and allowed
  EXPECT_TRUE(head_dist(make_t({1, 5}, {0, 0, nan, 0, 0}), L, both));
  HeadMasks only0{legal, mk_u8({1, 2}, {1, 0})};
  EXPECT_TRUE(head_dist(make_t({1, 5}, {0, 0, 0, nan, inf}), L, only0));
}

TEST(ReviewFixes, DeterministicModeReachesOtherThreads) {
  // set_deterministic(true) on this thread must also govern kernels launched from another thread.
  set_deterministic(true);
  Rng rng(5);
  auto d = Device::cpu();
  auto lin = Linear::create(64, 64, *d, rng, 0.1f);
  ASSERT_TRUE(lin);
  auto x = rand_t({128, 64}, rng);
  std::vector<float> a, b;
  auto run = [&](std::vector<float>& out) {
    ForwardCtx c;
    auto y = lin->forward(x, c);
    auto p = y->host_span<float>();
    out.assign(p->begin(), p->end());
  };
  run(a);
  std::thread t([&] { run(b); });
  t.join();
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
  set_deterministic(false);
  EXPECT_FALSE(deterministic());
  set_deterministic(true);  // leave on for the rest of the process, as other tests assume
}
