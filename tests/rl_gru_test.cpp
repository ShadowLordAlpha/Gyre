#include "gradcheck.hpp"

#include "gyre/ops.hpp"
#include "gyre/rl/gru.hpp"
#include "gyre/rl/ops.hpp"
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
double dot_all(const Tensor& y, const Tensor& c) {
  auto a = y.host_span<float>();
  auto b = c.host_span<float>();
  double s = 0;
  for (std::size_t i = 0; i < a->size(); ++i) s += double((*a)[i]) * double((*b)[i]);
  return s;
}

}  // namespace

TEST(Gru, MaskRowsAndConcatRows) {
  Rng r(1);
  auto h = rand_t({3, 2}, r);
  auto m = mask_rows(h, mk_u8({3}, {1, 0, 1}));
  ASSERT_TRUE(m);
  auto ph = h.host_span<float>(), pm = m->host_span<float>();
  EXPECT_EQ((*pm)[0], (*ph)[0]);
  EXPECT_EQ((*pm)[2], 0.f);
  EXPECT_EQ((*pm)[3], 0.f);
  EXPECT_EQ((*pm)[4], (*ph)[4]);
  Tensor parts[2] = {h, rand_t({2, 2}, r)};
  auto c = concat_rows(parts);
  ASSERT_TRUE(c);
  EXPECT_EQ(c->shape()[0], 5);
  EXPECT_EQ((*c->host_span<float>())[0], (*ph)[0]);
  Tensor bad[2] = {h, rand_t({2, 3}, r)};
  EXPECT_FALSE(concat_rows(bad));
}

TEST(Gru, StepMatchesHandComputation) {
  Rng rng(2);
  auto d = Device::cpu();
  auto g = Gru::create(1, 1, *d, rng);
  ASSERT_TRUE(g);
  // set weights: Wi=[a_r,a_z,a_n], bi=0, Wh=[b_r,b_z,b_n], bh=[0,0,c]
  auto ps = g->parameters();
  auto wi = ps[0].value.host_span<float>(), bi = ps[1].value.host_span<float>();
  auto wh = ps[2].value.host_span<float>(), bh = ps[3].value.host_span<float>();
  const float ar = 0.5f, az = -0.3f, an = 0.8f, br = 0.2f, bz = 0.4f, bn = -0.6f, c = 0.1f;
  (*wi)[0] = ar; (*wi)[1] = az; (*wi)[2] = an;
  (*wh)[0] = br; (*wh)[1] = bz; (*wh)[2] = bn;
  for (auto& v : *bi) v = 0.f;
  (*bh)[0] = 0.f; (*bh)[1] = 0.f; (*bh)[2] = c;
  const float x = 1.5f, h = 0.7f;
  auto sg = [](float v) { return 1.f / (1.f + std::exp(-v)); };
  const float r = sg(ar * x + br * h), z = sg(az * x + bz * h);
  const float n = std::tanh(an * x + r * (bn * h + c));
  const float expect = (1 - z) * n + z * h;
  auto out = g->step(make_t({1, 1}, {x}), make_t({1, 1}, {h}), nullptr);
  ASSERT_TRUE(out);
  EXPECT_NEAR((*out->host_span<float>())[0], expect, 1e-6);
}

TEST(Gru, BpttGradCheckWithResets) {
  Rng rng(3);
  auto d = Device::cpu();
  auto g = Gru::create(3, 4, *d, rng);
  ASSERT_TRUE(g);
  const std::int64_t T = 4, B = 2;
  std::vector<Param> ps;
  for (auto& p : g->parameters()) ps.push_back(Param{p.value, p.grad});
  for (int t = 0; t < T; ++t) ps.push_back(make_p(rand_t({B, 3}, rng)));  // x_t
  ps.push_back(make_p(rand_t({B, 4}, rng)));                              // h0
  const std::size_t X0 = 4, H0 = X0 + T;
  // resets: stream 0 resets at t=2; stream 1 never (keep=1 everywhere)
  std::vector<Tensor> keep;
  for (int t = 0; t < T; ++t) keep.push_back(mk_u8({B}, {std::uint8_t(t == 2 ? 0 : 1), 1}));
  std::vector<Tensor> c;
  for (int t = 0; t < T; ++t) c.push_back(rand_t({B, 4}, rng));
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    std::vector<GruCache> tape;
    Tensor h = ps[H0].value;
    double loss = 0;
    for (int t = 0; t < T; ++t) {
      auto hm = mask_rows(h, keep[t]);
      if (!hm) return std::unexpected(hm.error());
      GruCache cache{*hm, *hm, *hm, *hm, *hm, *hm};
      auto hn = g->step(ps[X0 + t].value, *hm, &cache);
      if (!hn) return std::unexpected(hn.error());
      tape.push_back(std::move(cache));
      loss += dot_all(*hn, c[t]);
      h = std::move(*hn);
    }
    if (backward) {
      auto carry = Tensor::zeros(c[0].shape(), DType::f32, c[0].device());
      Tensor dcarry = *carry;
      for (int t = T - 1; t >= 0; --t) {
        auto dh = add(c[t], dcarry);
        auto sg = g->step_backward(tape[t], *dh);
        if (!sg) return std::unexpected(sg.error());
        set_grad(ps[X0 + t], sg->dx);
        auto dm = mask_rows(sg->dh_in, keep[t]);
        dcarry = std::move(*dm);
      }
      set_grad(ps[H0], dcarry);
    }
    return loss;
  };
  GradCheckOpts o;
  o.h = 3e-3;
  o.max_per_param = 16;
  EXPECT_EQ(gradcheck(ps, fn, o), 0);
  // Reset at t=2 for stream 0: nothing before it can influence it via h (only via later steps' x).
  // dL/dh0[0] is zero? No: h0 reaches stream 0 only through steps 0-1 outputs, which feed the loss.
  // What must hold is the cut: gradient of the t>=2 loss terms w.r.t. h0[0] is zero. Checked below.
}

TEST(Gru, ResetCutsInformationFlow) {
  Rng rng(4);
  auto d = Device::cpu();
  auto g = Gru::create(2, 3, *d, rng);
  ASSERT_TRUE(g);
  auto run = [&](float x0b0) {
    // two streams, 2 steps; stream 0 resets before step 1
    auto x0 = make_t({2, 2}, {x0b0, 1.f, 0.3f, -0.2f});
    auto x1 = make_t({2, 2}, {0.5f, 0.1f, 0.4f, 0.9f});
    auto h = Tensor::zeros(std::vector<std::int64_t>{2, 3}, DType::f32, *d);
    auto h1 = g->step(x0, *h, nullptr);
    auto hm = mask_rows(*h1, mk_u8({2}, {0, 1}));  // reset stream 0
    auto h2 = g->step(x1, *hm, nullptr);
    return std::vector<float>(h2->host_span<float>()->begin(), h2->host_span<float>()->end());
  };
  auto a = run(0.1f), b = run(7.f);
  for (int j = 0; j < 3; ++j) EXPECT_EQ(a[j], b[j]) << "stream 0 leaked across reset";
  bool differs = false;  // stream 1 has no reset but its own x0 is unchanged -> equal too
  for (int j = 3; j < 6; ++j) differs |= (a[j] != b[j]);
  EXPECT_FALSE(differs);
}

namespace {

PolicyConfig rec_policy(std::int64_t hidden = 8) {
  PolicyConfig c;
  c.encoder.feat_dim = 2;
  c.encoder.global_dim = 4;
  c.encoder.max_entities = 2;
  c.encoder.d_model = 8;
  c.encoder.n_layer = 1;
  c.encoder.n_head = 2;
  c.encoder.d_ff = 12;
  c.encoder.out_dim = 8;
  c.head_sizes = {3};
  c.gru_hidden = hidden;
  return c;
}

Obs rand_obs(Rng& rng, std::int64_t rows) {
  std::vector<std::uint8_t> m(rows * 2, 1);
  m[1] = 0;  // some padding
  return Obs{rand_t({rows, 2, 2}, rng), mk_u8({rows, 2}, m), rand_t({rows, 4}, rng)};
}

}  // namespace

TEST(RecPolicy, SeqMatchesStepByStep) {
  Rng rng(5);
  auto d = Device::cpu();
  auto net = PolicyNet::create(rec_policy(), *d, rng);
  ASSERT_TRUE(net);
  for (auto& p : net->parameters())
    for (auto& v : *p.value.host_span<float>()) v += rng.normal(0.f, 0.2f);
  const std::int64_t T = 4, B = 3;
  auto obs = rand_obs(rng, T * B);
  auto h0 = rand_t({B, 8}, rng);
  std::vector<std::uint8_t> rs(T * B, 0);
  rs[0 * B + 1] = 1;
  rs[2 * B + 0] = 1;
  auto reset = mk_u8({T, B}, rs);
  ForwardCtx ctx;
  ctx.train = false;
  auto seq = net->forward_seq(obs, h0, reset, ctx);
  ASSERT_TRUE(seq);
  Tensor h = h0;
  for (std::int64_t t = 0; t < T; ++t) {
    auto e = narrow_rows(*obs.entities.clone(), 0, 0);  // (unused) keep API exercised below
    (void)e;
    // slice obs rows [t*B,(t+1)*B)
    auto sl = [&](const Tensor& x, DType, std::int64_t per) {
      auto src = x.host_span<float>();
      std::vector<float> v(src->begin() + t * B * per, src->begin() + (t + 1) * B * per);
      return v;
    };
    auto ent = make_t({B, 2, 2}, sl(obs.entities, DType::f32, 4));
    auto glob = make_t({B, 4}, sl(obs.global, DType::f32, 4));
    auto pm = obs.mask.host_span<std::uint8_t>();
    std::vector<std::uint8_t> mv(pm->begin() + t * B * 2, pm->begin() + (t + 1) * B * 2);
    std::vector<std::uint8_t> rv(B);
    for (std::int64_t b = 0; b < B; ++b) rv[b] = rs[t * B + b];
    auto st = net->forward_step(Obs{ent, mk_u8({B, 2}, mv), glob}, h, mk_u8({B}, rv));
    ASSERT_TRUE(st);
    auto a = st->out.logits.host_span<float>();
    auto b = narrow_rows(seq->out.logits, t * B, B);
    auto pb = b->host_span<float>();
    for (std::size_t i = 0; i < a->size(); ++i) EXPECT_NEAR((*a)[i], (*pb)[i], 1e-5f) << "t" << t;
    h = st->h;
  }
  auto hl = seq->h_last.host_span<float>(), hh = h.host_span<float>();
  for (std::size_t i = 0; i < hl->size(); ++i) EXPECT_NEAR((*hl)[i], (*hh)[i], 1e-5f);
}

TEST(RecPolicy, EndToEndBpttGradCheckPpo) {
  Rng rng(6);
  auto d = Device::cpu();
  auto net = PolicyNet::create(rec_policy(6), *d, rng);
  ASSERT_TRUE(net);
  for (auto& p : net->parameters())
    for (auto& v : *p.value.host_span<float>()) v += rng.normal(0.f, 0.2f);
  const std::int64_t T = 3, B = 2, R = T * B;
  auto obs = rand_obs(rng, R);
  auto h0 = rand_t({B, 6}, rng);
  auto reset = mk_u8({T, B}, {1, 0, 0, 1, 0, 0});  // stream0 reset at t=0, stream1 at t=1
  auto L = net->layout();
  HeadMasks m{mk_u8({R, 3}, {1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1}),
              mk_u8({R, 1}, {1, 1, 1, 1, 0, 1})};
  PpoBatch batch{mk_i32({R, 1}, {0, 2, 1, 0, 0, 2}), make_t({R}, {-1.f, -1.4f, -0.9f, -1.2f, 0.f, -1.1f}),
                 make_t({R}, {1.f, -1.f, 0.5f, -0.5f, 0.7f, 0.3f}),
                 make_t({R}, {0.1f, 0.2f, -0.3f, 0.4f, 0.f, 0.5f})};
  std::vector<Param> ps;
  for (auto& p : net->parameters()) ps.push_back(Param{p.value, p.grad});
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    ForwardCtx ctx;
    ctx.train = true;
    auto out = net->forward_seq(obs, h0, reset, ctx);
    if (!out) return std::unexpected(out.error());
    auto l = ppo_loss(out->out.logits, out->out.value, L, m, batch, PpoConfig{});
    if (!l) return std::unexpected(l.error());
    if (backward) {
      auto r = net->backward_seq(l->dlogits, l->dvalue, ctx);
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

TEST(RecPolicy, ErrorsAndModeChecks) {
  Rng rng(7);
  auto d = Device::cpu();
  auto rec = PolicyNet::create(rec_policy(), *d, rng);
  PolicyConfig ffc = rec_policy();
  ffc.gru_hidden = 0;
  auto ff = PolicyNet::create(ffc, *d, rng);
  ASSERT_TRUE(rec && ff);
  EXPECT_TRUE(rec->recurrent());
  EXPECT_FALSE(ff->recurrent());
  ForwardCtx ctx;
  auto obs = rand_obs(rng, 2);
  EXPECT_FALSE(rec->forward(obs, ctx));  // feedforward API on a recurrent net
  EXPECT_FALSE(ff->forward_step(obs, rand_t({2, 8}, rng), mk_u8({2}, {0, 0})));
  EXPECT_FALSE(rec->forward_step(obs, rand_t({2, 7}, rng), mk_u8({2}, {0, 0})));  // wrong H
  EXPECT_FALSE(rec->forward_seq(obs, rand_t({2, 8}, rng), mk_u8({3, 2}, {0, 0, 0, 0, 0, 0}), ctx));  // rows != T*B
}

// ------------------------------------------------------------ memory task
//
// Episodes of length L=4. The cue (one of 3) is visible only on the first step; the last step
// shows an "answer now" flag and rewards the action equal to the cue. A feedforward policy
// cannot beat chance (1/3); a recurrent one must carry the cue through three blank steps.
// Episodes straddle rollout boundaries (T=6) and reset flags mark every episode start.

namespace {

struct MemRun {
  double final_acc{0};  // accuracy on answer steps over the last 20 iterations
  double first_acc{0};
};

MemRun run_memory(bool recurrent, int iters, std::uint64_t seed) {
  Rng rng(seed);
  auto d = Device::cpu();
  PolicyConfig pc = rec_policy(recurrent ? 32 : 0);
  pc.encoder.d_model = 16;
  pc.encoder.d_ff = 32;
  pc.encoder.out_dim = 32;
  auto net = PolicyNet::create(pc, *d, rng);
  if (!net) std::abort();
  const std::int64_t E = 32, T = 6, L = 4, H = pc.gru_hidden;
  RolloutSpec spec;
  spec.steps = T;
  spec.envs = E;
  spec.max_entities = 2;
  spec.feat_dim = 2;
  spec.global_dim = 4;
  spec.hidden_dim = H;
  spec.layout = net->layout();
  auto buf = RolloutBuffer::create(spec);
  PpoTrainConfig tc;
  tc.lr = 3e-3f;
  tc.gamma = 0.95f;
  tc.lambda = 0.95f;
  tc.seed = seed + 1;
  auto tr = PpoTrainer::create(*net, tc);
  if (!buf || !tr) std::abort();

  std::vector<int> phase(E, 0), cue(E);
  for (auto& c : cue) c = int(rng.u32(3));
  std::vector<std::uint8_t> prev_done(E, 1);
  Tensor h = *Tensor::zeros(std::vector<std::int64_t>{E, std::max<std::int64_t>(H, 1)}, DType::f32, *d);
  auto legal = mk_u8({E, 3}, std::vector<std::uint8_t>(E * 3, 1));
  auto active = mk_u8({E, 1}, std::vector<std::uint8_t>(E, 1));
  std::vector<double> acc;
  for (int it = 0; it < iters; ++it) {
    buf->clear();
    int answers = 0, correct = 0;
    for (std::int64_t t = 0; t < T; ++t) {
      std::vector<float> ent(E * 4), glob(E * 4, 0.f);
      for (auto& x : ent) x = rng.normal(0.f, 1.f);
      for (std::int64_t e = 0; e < E; ++e) {
        if (phase[e] == 0) glob[e * 4 + cue[e]] = 1.f;
        if (phase[e] == L - 1) glob[e * 4 + 3] = 1.f;
      }
      Obs obs{make_t({E, 2, 2}, ent), mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)), make_t({E, 4}, glob)};
      auto reset = mk_u8({E}, prev_done);
      Tensor h_before = h;
      ActResult act = [&]() {
        if (recurrent) {
          auto r = policy_step_rec(*net, obs, legal, active, h, reset, &rng);
          if (!r) std::abort();
          h = r->h_out;
          return std::move(r->act);
        }
        auto r = policy_step(*net, obs, legal, active, &rng);
        if (!r) std::abort();
        return std::move(*r);
      }();
      auto pa = act.actions.host_span<std::int32_t>();
      std::vector<float> rew(E, 0.f);
      std::vector<std::uint8_t> done(E, 0);
      for (std::int64_t e = 0; e < E; ++e) {
        if (phase[e] == L - 1) {
          ++answers;
          if ((*pa)[e] == cue[e]) {
            ++correct;
            rew[e] = 1.f;
          }
          done[e] = 1;
        }
      }
      StepRecord rec{obs, legal, act.active, act.actions, act.logp, act.value, rew, done};
      if (recurrent) {
        rec.h_in = h_before;
        rec.reset = prev_done;
      }
      if (!buf->push(rec)) std::abort();
      for (std::int64_t e = 0; e < E; ++e) {
        if (done[e]) {
          phase[e] = 0;
          cue[e] = int(rng.u32(3));
        } else {
          ++phase[e];
        }
      }
      prev_done = done;
    }
    // bootstrap value of the state after the last step
    std::vector<float> lastv(E, 0.f);
    {
      std::vector<float> ent(E * 4), glob(E * 4, 0.f);
      for (auto& x : ent) x = rng.normal(0.f, 1.f);
      for (std::int64_t e = 0; e < E; ++e) {
        if (phase[e] == 0) glob[e * 4 + cue[e]] = 1.f;
        if (phase[e] == L - 1) glob[e * 4 + 3] = 1.f;
      }
      Obs obs{make_t({E, 2, 2}, ent), mk_u8({E, 2}, std::vector<std::uint8_t>(E * 2, 1)), make_t({E, 4}, glob)};
      ActResult a = [&]() {
        if (recurrent) {
          auto r = policy_step_rec(*net, obs, legal, active, h, mk_u8({E}, prev_done), &rng);
          if (!r) std::abort();
          return std::move(r->act);
        }
        auto r = policy_step(*net, obs, legal, active, &rng);
        if (!r) std::abort();
        return std::move(*r);
      }();
      auto pv = a.value.host_span<float>();
      lastv.assign(pv->begin(), pv->end());
    }
    if (!buf->compute_gae(lastv, tc.gamma, tc.lambda)) std::abort();
    auto st = tr->update(*buf);
    if (!st) {
      ADD_FAILURE() << st.error().message;
      break;
    }
    acc.push_back(answers ? double(correct) / answers : 0.0);
  }
  MemRun out;
  out.first_acc = acc.empty() ? 0 : acc[0];
  const std::size_t n = std::min<std::size_t>(20, acc.size());
  for (std::size_t i = acc.size() - n; i < acc.size(); ++i) out.final_acc += acc[i];
  out.final_acc /= double(std::max<std::size_t>(n, 1));
  return out;
}

}  // namespace

TEST(RecPolicy, FeedforwardCannotSolveMemoryTask) {
  set_deterministic(true);
  auto r = run_memory(false, 60, 3);
  EXPECT_LT(r.final_acc, 0.55) << "chance is 1/3; got " << r.final_acc;
}

TEST(RecPolicy, GruSolvesMemoryTaskAcrossRolloutBoundaries) {
  set_deterministic(true);
  auto r = run_memory(true, 250, 3);
  EXPECT_GT(r.final_acc, 0.85) << "first " << r.first_acc << " final " << r.final_acc;
}

TEST(RecPolicy, TruncatedBpttChunksStillTrain) {
  // seq_len shorter than the rollout: trains on stored states at chunk starts without errors.
  set_deterministic(true);
  Rng rng(9);
  auto d = Device::cpu();
  auto net = PolicyNet::create(rec_policy(8), *d, rng);
  ASSERT_TRUE(net);
  RolloutSpec spec;
  spec.steps = 6;
  spec.envs = 4;
  spec.max_entities = 2;
  spec.feat_dim = 2;
  spec.global_dim = 4;
  spec.hidden_dim = 8;
  spec.layout = net->layout();
  auto buf = RolloutBuffer::create(spec);
  ASSERT_TRUE(buf);
  const std::int64_t E = 4;
  auto legal = mk_u8({E, 3}, std::vector<std::uint8_t>(E * 3, 1));
  auto active = mk_u8({E, 1}, std::vector<std::uint8_t>(E, 1));
  Tensor h = *Tensor::zeros(std::vector<std::int64_t>{E, 8}, DType::f32, *d);
  std::vector<std::uint8_t> reset(E, 1);
  for (int t = 0; t < 6; ++t) {
    auto obs = rand_obs(rng, E);
    auto hb = h;
    auto r = policy_step_rec(*net, obs, legal, active, h, mk_u8({E}, reset), &rng);
    ASSERT_TRUE(r);
    h = r->h_out;
    StepRecord rec{obs, legal, r->act.active, r->act.actions, r->act.logp, r->act.value,
                   std::vector<float>(E, 0.5f), std::vector<std::uint8_t>(E, t == 2 ? 1 : 0)};
    rec.h_in = hb;
    rec.reset = reset;
    ASSERT_TRUE(buf->push(rec));
    reset.assign(E, t == 2 ? 1 : 0);
  }
  std::vector<float> lv(E, 0.f);
  ASSERT_TRUE(buf->compute_gae(lv, 0.9f, 0.9f));
  // seq_minibatch content: h0 and reset rows come from the stored steps
  const std::int64_t envs[2] = {3, 1};
  auto sb = buf->seq_minibatch(envs, 2, 3);
  ASSERT_TRUE(sb);
  EXPECT_EQ(sb->len, 3);
  EXPECT_EQ(sb->B, 2);
  EXPECT_EQ(sb->mb.obs.entities.shape()[0], 6);
  auto pr = sb->reset.host_span<std::uint8_t>();
  EXPECT_EQ((*pr)[0], 0);  // step 2: step 1 was not done
  EXPECT_EQ((*pr)[1], 0);
  EXPECT_EQ((*pr)[2], 1);  // step 3 follows done at step 2 -> episode start
  EXPECT_EQ((*pr)[3], 1);
  EXPECT_EQ((*pr)[4], 0);  // step 4
  EXPECT_FALSE(buf->seq_minibatch(envs, 5, 3));  // runs past the end

  PpoTrainConfig tc;
  tc.seq_len = 2;
  auto tr = PpoTrainer::create(*net, tc);
  ASSERT_TRUE(tr);
  auto st = tr->update(*buf);
  ASSERT_TRUE(st) << st.error().message;
  EXPECT_TRUE(std::isfinite(st->total));
  EXPECT_GT(st->updates, 0);
  // Mismatch checks: feedforward trainer on a recurrent buffer is rejected.
  PolicyConfig ffc = rec_policy();
  ffc.gru_hidden = 0;
  auto ff = PolicyNet::create(ffc, *d, rng);
  auto tr2 = PpoTrainer::create(*ff, PpoTrainConfig{});
  EXPECT_FALSE(tr2->update(*buf));
}
