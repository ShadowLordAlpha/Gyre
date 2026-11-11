#include "gradcheck.hpp"

#include "gyre/rl/policy_io.hpp"
#include "gyre_infer/policy.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace gyre;
using namespace gyre::testing;

namespace {

Tensor mk_u8(std::vector<std::int64_t> shape, const std::vector<std::uint8_t>& v) {
  auto d = Device::cpu();
  auto t = Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), shape, DType::u8, *d);
  if (!t) std::abort();
  return std::move(*t);
}

std::filesystem::path tmp(const char* name) {
  auto p = std::filesystem::temp_directory_path() / "gyre_infer_tests";
  std::filesystem::create_directories(p);
  return p / name;
}

PolicyConfig cfg(std::int64_t gru) {
  PolicyConfig c;
  c.encoder.feat_dim = 5;
  c.encoder.global_dim = 3;
  c.encoder.max_entities = 12;
  c.encoder.d_model = 16;
  c.encoder.n_layer = 2;
  c.encoder.n_head = 4;
  c.encoder.d_ff = 24;
  c.encoder.out_dim = 20;
  c.head_sizes = {4, 3, 6};
  c.gru_hidden = gru;
  return c;
}

PolicyNet make_net(const PolicyConfig& c, std::uint64_t seed) {
  Rng rng(seed);
  auto d = Device::cpu();
  auto net = PolicyNet::create(c, *d, rng);
  if (!net) std::abort();
  for (auto& p : net->parameters())  // non-trivial LN gains/biases and head weights
    for (auto& v : *p.value.host_span<float>()) v += rng.normal(0.f, 0.15f);
  return std::move(*net);
}

std::unique_ptr<gyre_infer::Policy> export_and_load(PolicyNet& net, const char* name) {
  auto path = tmp(name);
  auto r = export_policy_flat(path, net);
  if (!r) {
    ADD_FAILURE() << r.error().message;
    return nullptr;
  }
  std::string err;
  auto p = gyre_infer::Policy::load(path.string(), &err);
  if (!p) ADD_FAILURE() << err;
  return p;
}

// One training-side observation (batch 1) padded to `slots`, with a validity mask.
struct Sample {
  std::vector<float> ent, glob;
  std::vector<std::uint8_t> mask;
  int slots;
};
Sample rand_sample(Rng& rng, const PolicyConfig& c, int slots, int valid) {
  Sample s;
  s.slots = slots;
  s.ent.resize(slots * c.encoder.feat_dim);
  for (auto& x : s.ent) x = rng.normal(0.f, 1.f);
  s.glob.resize(c.encoder.global_dim);
  for (auto& x : s.glob) x = rng.normal(0.f, 1.f);
  s.mask.assign(slots, 0);
  // valid entities scattered, not just a prefix
  int placed = 0;
  for (int i = 0; i < slots && placed < valid; ++i)
    if (rng.u32(2) || slots - i == valid - placed) {
      s.mask[i] = 1;
      ++placed;
    }
  return s;
}
Obs to_obs(const Sample& s, const PolicyConfig& c) {
  return Obs{make_t({1, s.slots, c.encoder.feat_dim}, s.ent), mk_u8({1, s.slots}, s.mask),
             make_t({1, c.encoder.global_dim}, s.glob)};
}
gyre_infer::Input to_input(const Sample& s, bool reset) {
  gyre_infer::Input in;
  in.entities = s.ent.data();
  in.mask = s.mask.data();
  in.n_slots = s.slots;
  in.global = s.glob.data();
  in.reset = reset;
  return in;
}

void expect_close(const float* a, const float* b, int n, float tol, const char* what) {
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(a[i], b[i], tol * (1.f + std::abs(b[i]))) << what << " [" << i << "]";
}

}  // namespace

TEST(Infer, FeedforwardParityWithTraining) {
  auto c = cfg(0);
  auto net = make_net(c, 1);
  auto pol = export_and_load(net, "ff.gpol");
  ASSERT_TRUE(pol);
  EXPECT_EQ(pol->info().total_logits(), 13);
  EXPECT_EQ(pol->info().head_offset(2), 7);
  EXPECT_FALSE(pol->info().recurrent());
  auto sess = pol->make_session();
  Rng rng(2);
  for (int valid : {1, 3, 7, 12, 0}) {
    auto s = rand_sample(rng, c, 12, valid);
    ForwardCtx ctx;
    ctx.train = false;
    auto ref = net.forward(to_obs(s, c), ctx);
    ASSERT_TRUE(ref);
    std::vector<float> logits(13);
    float value = 0;
    ASSERT_TRUE(pol->step(sess, to_input(s, false), logits.data(), &value)) << sess.error();
    expect_close(logits.data(), ref->logits.host_span<float>()->data(), 13, 2e-4f, "logits");
    EXPECT_NEAR(value, (*ref->value.host_span<float>())[0], 2e-4f * (1.f + std::abs(value))) << "valid=" << valid;
  }
}

TEST(Infer, SlotLayoutDoesNotMatter) {
  // Same entities in different slots / with different padding -> same decision.
  auto c = cfg(0);
  auto net = make_net(c, 3);
  auto pol = export_and_load(net, "ff2.gpol");
  ASSERT_TRUE(pol);
  auto sess = pol->make_session();
  Rng rng(4);
  auto a = rand_sample(rng, c, 12, 5);
  // build b: compact the valid rows to the front of a 6-slot input, mask null (all valid)
  std::vector<float> ent_b;
  for (int i = 0; i < 12; ++i)
    if (a.mask[i]) ent_b.insert(ent_b.end(), a.ent.begin() + i * 5, a.ent.begin() + (i + 1) * 5);
  gyre_infer::Input ib;
  ib.entities = ent_b.data();
  ib.mask = nullptr;
  ib.n_slots = 5;
  ib.global = a.glob.data();
  float la[13], lb[13], va, vb;
  ASSERT_TRUE(pol->step(sess, to_input(a, false), la, &va));
  ASSERT_TRUE(pol->step(sess, ib, lb, &vb));
  expect_close(la, lb, 13, 1e-5f, "logits");
  EXPECT_NEAR(va, vb, 1e-5f);
}

TEST(Infer, RecurrentParityOverEpisodesWithResets) {
  auto c = cfg(24);
  auto net = make_net(c, 5);
  auto pol = export_and_load(net, "rec.gpol");
  ASSERT_TRUE(pol);
  EXPECT_TRUE(pol->info().recurrent());
  auto sess = pol->make_session();
  Rng rng(6);
  Tensor h = *Tensor::zeros(std::vector<std::int64_t>{1, 24}, DType::f32, *Device::cpu());
  const bool resets[8] = {true, false, false, false, true, false, false, true};
  for (int t = 0; t < 8; ++t) {
    auto s = rand_sample(rng, c, 12, 1 + int(rng.u32(12)));
    auto ref = net.forward_step(to_obs(s, c), h, mk_u8({1}, {std::uint8_t(resets[t])}));
    ASSERT_TRUE(ref);
    h = ref->h;
    std::vector<float> logits(13);
    float value = 0;
    ASSERT_TRUE(pol->step(sess, to_input(s, resets[t]), logits.data(), &value));
    expect_close(logits.data(), ref->out.logits.host_span<float>()->data(), 13, 3e-4f, "logits");
    EXPECT_NEAR(value, (*ref->out.value.host_span<float>())[0], 3e-4f * (1.f + std::abs(value))) << "t=" << t;
  }
  // explicit Session::reset() equals Input::reset
  auto s1 = pol->make_session(), s2 = pol->make_session();
  auto smp = rand_sample(rng, c, 12, 6);
  float l1[13], l2[13], v1, v2;
  ASSERT_TRUE(pol->step(s1, to_input(smp, false), l1, &v1));
  ASSERT_TRUE(pol->step(s1, to_input(smp, false), l1, &v1));  // state now non-zero
  s1.reset();
  ASSERT_TRUE(pol->step(s1, to_input(smp, false), l1, &v1));
  ASSERT_TRUE(pol->step(s2, to_input(smp, false), l2, &v2));
  expect_close(l1, l2, 13, 0.f, "after Session::reset");
}

TEST(Infer, CheckpointRoundTrip) {
  auto c = cfg(16);
  auto net = make_net(c, 7);
  auto path = tmp("policy.gyre");
  ASSERT_TRUE(save_policy(path, net, 1234));
  auto d = Device::cpu();
  auto back = load_policy(path, *d);
  ASSERT_TRUE(back) << back.error().message;
  EXPECT_EQ(back->config().gru_hidden, 16);
  EXPECT_EQ(back->config().head_sizes, c.head_sizes);
  auto a = net.parameters(), b = back->parameters();
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto pa = a[i].value.host_span<float>(), pb = b[i].value.host_span<float>();
    ASSERT_EQ(pa->size(), pb->size());
    EXPECT_EQ(0, std::memcmp(pa->data(), pb->data(), pa->size() * sizeof(float))) << "param " << i;
  }
  // exports from the original and the reloaded net are byte-identical
  ASSERT_TRUE(export_policy_flat(tmp("a.gpol"), net));
  ASSERT_TRUE(export_policy_flat(tmp("b.gpol"), *back));
  std::ifstream fa(tmp("a.gpol"), std::ios::binary), fb(tmp("b.gpol"), std::ios::binary);
  std::string sa((std::istreambuf_iterator<char>(fa)), {}), sb((std::istreambuf_iterator<char>(fb)), {});
  EXPECT_EQ(sa, sb);
  EXPECT_FALSE(load_policy(tmp("does_not_exist.gyre"), *d));
}

TEST(Infer, RejectsCorruptFiles) {
  auto c = cfg(0);
  auto net = make_net(c, 8);
  auto path = tmp("corrupt.gpol");
  ASSERT_TRUE(export_policy_flat(path, net));
  std::ifstream f(path, std::ios::binary);
  std::string good((std::istreambuf_iterator<char>(f)), {});
  std::string err;
  EXPECT_TRUE(gyre_infer::Policy::load_from_memory(good.data(), good.size(), &err)) << err;
  auto flip = good;
  flip[flip.size() / 2] ^= 1;
  EXPECT_FALSE(gyre_infer::Policy::load_from_memory(flip.data(), flip.size(), &err));
  EXPECT_NE(err.find("checksum"), std::string::npos) << err;
  EXPECT_FALSE(gyre_infer::Policy::load_from_memory(good.data(), good.size() - 100, &err));
  auto wrong_cfg = good;
  wrong_cfg[8 + 8 + 4 * 3] ^= 1;  // d_model field
  EXPECT_FALSE(gyre_infer::Policy::load_from_memory(wrong_cfg.data(), wrong_cfg.size(), &err));
  EXPECT_FALSE(gyre_infer::Policy::load("no_such_file.gpol", &err));
  // bad inputs to step
  auto pol = gyre_infer::Policy::load_from_memory(good.data(), good.size(), &err);
  ASSERT_TRUE(pol);
  auto sess = pol->make_session();
  Rng rng(9);
  auto s = rand_sample(rng, c, 12, 4);
  float l[13], v;
  auto in = to_input(s, false);
  in.n_slots = 13;  // > max_entities
  EXPECT_FALSE(pol->step(sess, in, l, &v));
  EXPECT_FALSE(sess.error().empty());
  in = to_input(s, false);
  in.global = nullptr;
  EXPECT_FALSE(pol->step(sess, in, l, &v));
}

TEST(Infer, PickMasked) {
  const float lg[5] = {0.f, 5.f, 1.f, 3.f, 2.f};
  const std::uint8_t legal[5] = {1, 0, 1, 1, 0};
  EXPECT_EQ(gyre_infer::pick_masked(lg, legal, 5, -1.f), 3);  // arg-max among legal
  EXPECT_EQ(gyre_infer::pick_masked(lg, nullptr, 5, -1.f), 1);
  const std::uint8_t none[5] = {0, 0, 0, 0, 0};
  EXPECT_EQ(gyre_infer::pick_masked(lg, none, 5, 0.5f), -1);
  int hist[5] = {0};
  Rng rng(10);
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    int k = gyre_infer::pick_masked(lg, legal, 5, rng.uniform01());
    ASSERT_TRUE(k == 0 || k == 2 || k == 3);
    hist[k]++;
  }
  const double z = std::exp(0.0) + std::exp(1.0) + std::exp(3.0);
  EXPECT_NEAR(double(hist[0]) / n, 1.0 / z, 0.015);
  EXPECT_NEAR(double(hist[3]) / n, std::exp(3.0) / z, 0.015);
}

TEST(Infer, SingleCoreLatencyAtFullSetSize) {
  // 128 entities, default-size encoder + GRU: the "ladder bot" case. Prints the measurement.
  PolicyConfig c;
  c.encoder.feat_dim = 24;
  c.encoder.global_dim = 16;
  c.encoder.max_entities = 128;  // d=64, 2 layers, 4 heads, ff=128, out=128 (defaults)
  c.gru_hidden = 128;
  c.head_sizes = {12, 8, 64, 16};
  auto net = make_net(c, 11);
  auto pol = export_and_load(net, "big.gpol");
  ASSERT_TRUE(pol);
  Rng rng(12);
  auto s = rand_sample(rng, c, 128, 128);
  auto sess = pol->make_session();
  std::vector<float> logits(pol->info().total_logits());
  float v;
  for (int i = 0; i < 5; ++i) ASSERT_TRUE(pol->step(sess, to_input(s, i == 0), logits.data(), &v));
  const int iters = 50;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) ASSERT_TRUE(pol->step(sess, to_input(s, false), logits.data(), &v));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::int64_t params = 0;
  for (auto& p : net.parameters()) params += p.value.numel();
  std::printf("[ infer   ] %lld params, 128 entities: %.3f ms / decision (single thread)\n",
              static_cast<long long>(params), ms);
  EXPECT_LT(ms, 10.0) << "budget is ~10 ms per decision";
}

TEST(Infer, LatencyScalingWithModelSize) {
  // Informational: where the ~10 ms budget runs out on one baseline-SSE2 core.
  struct Row { int d, layers, ff; };
  for (Row r : {Row{64, 2, 128}, Row{96, 2, 384}, Row{128, 3, 512}}) {
    PolicyConfig c;
    c.encoder.feat_dim = 24;
    c.encoder.global_dim = 16;
    c.encoder.max_entities = 128;
    c.encoder.d_model = r.d;
    c.encoder.n_layer = r.layers;
    c.encoder.n_head = 4;
    c.encoder.d_ff = r.ff;
    c.encoder.out_dim = 2 * r.d;
    c.gru_hidden = 2 * r.d;
    c.head_sizes = {12, 8, 64, 16};
    auto net = make_net(c, 21);
    auto pol = export_and_load(net, "scale.gpol");
    ASSERT_TRUE(pol);
    Rng rng(22);
    auto s = rand_sample(rng, c, 128, 128);
    auto sess = pol->make_session();
    std::vector<float> logits(pol->info().total_logits());
    float v;
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(pol->step(sess, to_input(s, false), logits.data(), &v));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) ASSERT_TRUE(pol->step(sess, to_input(s, false), logits.data(), &v));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
    std::int64_t params = 0;
    for (auto& p : net.parameters()) params += p.value.numel();
    std::printf("[ infer   ] d=%d layers=%d ff=%d: %lld params, %.2f ms / decision @128 entities\n", r.d, r.layers,
                r.ff, static_cast<long long>(params), ms);
  }
}
