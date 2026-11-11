#include "gyre/rl/ppo.hpp"

#include "gyre/ops.hpp"
#include "gyre/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace gyre {
namespace {

Error err(const char* m) { return make_error(Errc::invalid_shape, m); }

template <class T>
Result<Tensor> from_vec(const T* data, std::size_t n, std::vector<std::int64_t> shape, DType dt) {
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  return Tensor::from_host(std::as_bytes(std::span<const T>(data, n)), shape, dt, *d);
}

}  // namespace

// ---------------------------------------------------------------- acting

Result<ActResult> policy_step(PolicyNet& net, const Obs& obs, const Tensor& legal,
                              const Tensor& initial_active, Rng* rng, bool greedy,
                              const ActiveFn& after_head) {
  ForwardCtx ctx;
  ctx.train = false;
  auto out = net.forward(obs, ctx);
  if (!out) return std::unexpected(out.error());
  auto s = sample_conditional(out->logits, net.layout(), legal, initial_active, rng, greedy, after_head);
  if (!s) return std::unexpected(s.error());
  HeadMasks m{legal, s->active};
  auto dist = head_dist(out->logits, net.layout(), m);
  if (!dist) return std::unexpected(dist.error());
  auto lp = log_prob(*dist, net.layout(), m, s->actions);
  if (!lp) return std::unexpected(lp.error());
  return ActResult{std::move(s->actions), std::move(s->active), std::move(*lp), std::move(out->value)};
}

Result<RecActResult> policy_step_rec(PolicyNet& net, const Obs& obs, const Tensor& legal,
                                     const Tensor& initial_active, const Tensor& h_in,
                                     const Tensor& reset, Rng* rng, bool greedy,
                                     const ActiveFn& after_head) {
  auto st = net.forward_step(obs, h_in, reset);
  if (!st) return std::unexpected(st.error());
  auto smp = sample_conditional(st->out.logits, net.layout(), legal, initial_active, rng, greedy, after_head);
  if (!smp) return std::unexpected(smp.error());
  HeadMasks m{legal, smp->active};
  auto dist = head_dist(st->out.logits, net.layout(), m);
  if (!dist) return std::unexpected(dist.error());
  auto lp = log_prob(*dist, net.layout(), m, smp->actions);
  if (!lp) return std::unexpected(lp.error());
  return RecActResult{ActResult{std::move(smp->actions), std::move(smp->active), std::move(*lp),
                                std::move(st->out.value)},
                      std::move(st->h)};
}

// ---------------------------------------------------------------- rollout

RolloutBuffer::RolloutBuffer(RolloutSpec s) : spec_(std::move(s)) {
  const auto S = static_cast<std::size_t>(spec_.steps * spec_.envs);
  ent_.resize(S * spec_.max_entities * spec_.feat_dim);
  glob_.resize(S * spec_.global_dim);
  mask_.resize(S * spec_.max_entities);
  legal_.resize(S * spec_.layout.total());
  active_.resize(S * spec_.layout.heads());
  act_.resize(S * spec_.layout.heads());
  rew_.resize(S);
  logp_.resize(S);
  val_.resize(S);
  adv_.resize(S);
  ret_.resize(S);
  done_.resize(S);
  if (spec_.hidden_dim > 0) {
    hin_.resize(S * spec_.hidden_dim);
    reset_.resize(S);
  }
}

Result<RolloutBuffer> RolloutBuffer::create(RolloutSpec spec) {
  if (spec.steps < 1 || spec.envs < 1 || spec.max_entities < 1 || spec.feat_dim < 1 ||
      spec.global_dim < 0 || spec.hidden_dim < 0 || spec.layout.heads() < 1)
    return std::unexpected(err("RolloutSpec"));
  return RolloutBuffer(std::move(spec));
}

void RolloutBuffer::clear() {
  t_ = 0;
  gae_done_ = false;
}

Result<void> RolloutBuffer::push(const StepRecord& r) {
  if (full()) return std::unexpected(err("rollout buffer full"));
  const auto E = spec_.envs, N = spec_.max_entities, F = spec_.feat_dim, G = spec_.global_dim,
             T = spec_.layout.total(), H = spec_.layout.heads();
  auto check = [&](const Tensor& t, DType dt, std::initializer_list<std::int64_t> sh,
                   const char* what) -> Result<void> {
    if (t.dtype() != dt || t.rank() != sh.size()) return std::unexpected(err(what));
    std::size_t i = 0;
    for (auto s : sh)
      if (t.shape()[i++] != s) return std::unexpected(err(what));
    return {};
  };
  if (auto c = check(r.obs.entities, DType::f32, {E, N, F}, "entities [E,N,F]"); !c) return c;
  if (auto c = check(r.obs.mask, DType::u8, {E, N}, "mask u8 [E,N]"); !c) return c;
  if (auto c = check(r.obs.global, DType::f32, {E, G}, "global [E,G]"); !c) return c;
  if (auto c = check(r.legal, DType::u8, {E, T}, "legal u8 [E,total]"); !c) return c;
  if (auto c = check(r.active, DType::u8, {E, H}, "active u8 [E,H]"); !c) return c;
  if (auto c = check(r.actions, DType::i32, {E, H}, "actions i32 [E,H]"); !c) return c;
  if (auto c = check(r.logp, DType::f32, {E}, "logp [E]"); !c) return c;
  if (auto c = check(r.value, DType::f32, {E}, "value [E]"); !c) return c;
  if (static_cast<std::int64_t>(r.reward.size()) != E || static_cast<std::int64_t>(r.done.size()) != E)
    return std::unexpected(err("reward/done size != envs"));

  if (spec_.hidden_dim > 0) {
    if (!r.h_in || static_cast<std::int64_t>(r.reset.size()) != E)
      return std::unexpected(err("recurrent rollout needs h_in [E,H] and reset [E]"));
    if (auto c = check(*r.h_in, DType::f32, {E, spec_.hidden_dim}, "h_in [E,H]"); !c) return c;
  }
  const auto base = static_cast<std::size_t>(t_ * E);
  auto cp = [](auto& dst, std::size_t off, auto span) { std::copy(span.begin(), span.end(), dst.begin() + off); };
  auto e = r.obs.entities.template host_span<float>();
  auto m = r.obs.mask.template host_span<std::uint8_t>();
  auto g = r.obs.global.template host_span<float>();
  auto lg = r.legal.template host_span<std::uint8_t>();
  auto ac = r.active.template host_span<std::uint8_t>();
  auto a = r.actions.template host_span<std::int32_t>();
  auto lp = r.logp.template host_span<float>();
  auto v = r.value.template host_span<float>();
  if (!e || !m || !g || !lg || !ac || !a || !lp || !v) return std::unexpected(err("rollout push: host"));
  cp(ent_, base * N * F, *e);
  cp(mask_, base * N, *m);
  cp(glob_, base * G, *g);
  cp(legal_, base * T, *lg);
  cp(active_, base * H, *ac);
  cp(act_, base * H, *a);
  cp(logp_, base, *lp);
  cp(val_, base, *v);
  cp(rew_, base, r.reward);
  cp(done_, base, r.done);
  if (spec_.hidden_dim > 0) {
    auto hh = r.h_in->host_span<float>();
    if (!hh) return std::unexpected(hh.error());
    cp(hin_, base * spec_.hidden_dim, *hh);
    cp(reset_, base, r.reset);
  }
  ++t_;
  gae_done_ = false;
  return {};
}

Result<void> RolloutBuffer::compute_gae(std::span<const float> last_value, float gamma, float lambda) {
  if (!full()) return std::unexpected(err("compute_gae: buffer not full"));
  const auto E = spec_.envs, T = spec_.steps;
  if (static_cast<std::int64_t>(last_value.size()) != E) return std::unexpected(err("last_value [envs]"));
  for (std::int64_t e = 0; e < E; ++e) {
    double next_adv = 0;
    for (std::int64_t t = T - 1; t >= 0; --t) {
      const auto i = static_cast<std::size_t>(t * E + e);
      const double nonterm = done_[i] ? 0.0 : 1.0;
      const double next_v = t == T - 1 ? double(last_value[static_cast<std::size_t>(e)])
                                       : double(val_[static_cast<std::size_t>((t + 1) * E + e)]);
      const double delta = double(rew_[i]) + double(gamma) * next_v * nonterm - double(val_[i]);
      next_adv = delta + double(gamma) * double(lambda) * nonterm * next_adv;
      adv_[i] = static_cast<float>(next_adv);
      ret_[i] = static_cast<float>(next_adv + double(val_[i]));
    }
  }
  gae_done_ = true;
  return {};
}

void RolloutBuffer::normalize_advantages() {
  if (adv_.empty()) return;
  double mean = 0, var = 0;
  for (float v : adv_) mean += v;
  mean /= double(adv_.size());
  for (float v : adv_) var += (v - mean) * (v - mean);
  var /= double(adv_.size());
  const double inv = 1.0 / (std::sqrt(var) + 1e-8);
  for (auto& v : adv_) v = static_cast<float>((v - mean) * inv);
}

Result<Minibatch> RolloutBuffer::minibatch(std::span<const std::int64_t> idx) const {
  const auto M = static_cast<std::int64_t>(idx.size());
  if (M < 1) return std::unexpected(err("empty minibatch"));
  const auto N = spec_.max_entities, F = spec_.feat_dim, G = spec_.global_dim,
             T = spec_.layout.total(), H = spec_.layout.heads();
  std::vector<float> ent(M * N * F), glob(M * G), logp(M), adv(M), ret(M);
  std::vector<std::uint8_t> mask(M * N), legal(M * T), active(M * H);
  std::vector<std::int32_t> act(M * H);
  for (std::int64_t k = 0; k < M; ++k) {
    const auto i = idx[static_cast<std::size_t>(k)];
    if (i < 0 || i >= size()) return std::unexpected(err("minibatch index OOB"));
    std::copy_n(ent_.begin() + i * N * F, N * F, ent.begin() + k * N * F);
    std::copy_n(mask_.begin() + i * N, N, mask.begin() + k * N);
    std::copy_n(glob_.begin() + i * G, G, glob.begin() + k * G);
    std::copy_n(legal_.begin() + i * T, T, legal.begin() + k * T);
    std::copy_n(active_.begin() + i * H, H, active.begin() + k * H);
    std::copy_n(act_.begin() + i * H, H, act.begin() + k * H);
    logp[k] = logp_[i];
    adv[k] = adv_[i];
    ret[k] = ret_[i];
  }
  auto te = from_vec(ent.data(), ent.size(), {M, N, F}, DType::f32);
  auto tm = from_vec(mask.data(), mask.size(), {M, N}, DType::u8);
  auto tg = from_vec(glob.data(), glob.size(), {M, G}, DType::f32);
  auto tl = from_vec(legal.data(), legal.size(), {M, T}, DType::u8);
  auto ta = from_vec(active.data(), active.size(), {M, H}, DType::u8);
  auto tact = from_vec(act.data(), act.size(), {M, H}, DType::i32);
  auto tlp = from_vec(logp.data(), logp.size(), {M}, DType::f32);
  auto tadv = from_vec(adv.data(), adv.size(), {M}, DType::f32);
  auto tret = from_vec(ret.data(), ret.size(), {M}, DType::f32);
  if (!te || !tm || !tg || !tl || !ta || !tact || !tlp || !tadv || !tret)
    return std::unexpected(err("minibatch alloc"));
  return Minibatch{Obs{std::move(*te), std::move(*tm), std::move(*tg)},
                   HeadMasks{std::move(*tl), std::move(*ta)},
                   PpoBatch{std::move(*tact), std::move(*tlp), std::move(*tadv), std::move(*tret)}};
}

Result<SeqBatch> RolloutBuffer::seq_minibatch(std::span<const std::int64_t> envs, std::int64_t t0,
                                              std::int64_t len) const {
  if (spec_.hidden_dim < 1) return std::unexpected(err("seq_minibatch: feedforward buffer"));
  const auto E = spec_.envs, H = spec_.hidden_dim, B = static_cast<std::int64_t>(envs.size());
  if (B < 1 || len < 1 || t0 < 0 || t0 + len > spec_.steps) return std::unexpected(err("seq_minibatch range"));
  std::vector<std::int64_t> idx(static_cast<std::size_t>(len * B));
  std::vector<float> h0(static_cast<std::size_t>(B * H));
  std::vector<std::uint8_t> reset(static_cast<std::size_t>(len * B));
  for (std::int64_t t = 0; t < len; ++t)
    for (std::int64_t b = 0; b < B; ++b) {
      const auto e = envs[static_cast<std::size_t>(b)];
      if (e < 0 || e >= E) return std::unexpected(err("seq_minibatch env OOB"));
      const auto i = (t0 + t) * E + e;
      idx[static_cast<std::size_t>(t * B + b)] = i;
      reset[static_cast<std::size_t>(t * B + b)] = reset_[static_cast<std::size_t>(i)];
      if (t == 0) std::copy_n(hin_.begin() + i * H, H, h0.begin() + b * H);
    }
  auto mb = minibatch(idx);
  if (!mb) return std::unexpected(mb.error());
  auto th = from_vec(h0.data(), h0.size(), {B, H}, DType::f32);
  auto tr = from_vec(reset.data(), reset.size(), {len, B}, DType::u8);
  if (!th || !tr) return std::unexpected(err("seq alloc"));
  return SeqBatch{std::move(*mb), std::move(*th), std::move(*tr), len, B};
}

// ---------------------------------------------------------------- trainer

Result<PpoTrainer> PpoTrainer::create(PolicyNet& net, PpoTrainConfig cfg) {
  if (cfg.epochs < 1 || cfg.minibatches < 1) return std::unexpected(err("PpoTrainConfig"));
  auto adam = Adam::create(net.parameters(), cfg.lr);
  if (!adam) return std::unexpected(adam.error());
  return PpoTrainer(net, cfg, std::move(*adam));
}

Result<PpoStats> PpoTrainer::update(RolloutBuffer& buf) {
  if (!buf.full()) return std::unexpected(err("update: rollout buffer not full"));
  if (!buf.gae_computed()) return std::unexpected(err("update: call compute_gae after filling the buffer"));
  if (net_->recurrent() != (buf.spec().hidden_dim > 0))
    return std::unexpected(err("update: policy and rollout buffer disagree on recurrence"));
  if (cfg_.normalize_advantages) buf.normalize_advantages();
  const bool rec = net_->recurrent();
  const auto S = buf.size();
  // Unit split into minibatches: samples (feedforward) or whole streams (recurrent).
  const auto units = rec ? buf.spec().envs : S;
  const auto mb = std::min<std::int64_t>(cfg_.minibatches, units);
  std::vector<std::int64_t> perm(static_cast<std::size_t>(units));
  PpoStats st;
  double kl_epoch = 0;
  auto apply = [&](ForwardCtx& ctx, PolicyOut& out, const HeadMasks& masks, const PpoBatch& batch,
                   bool seq) -> Result<void> {
    auto l = ppo_loss(out.logits, out.value, net_->layout(), masks, batch, cfg_.loss);
    if (!l) return std::unexpected(l.error());
    auto br = seq ? net_->backward_seq(l->dlogits, l->dvalue, ctx) : net_->backward(l->dlogits, l->dvalue, ctx);
    if (!br) return br;
    auto gn = clip_grad_norm(net_->parameters(), cfg_.max_grad_norm);
    if (!gn) return std::unexpected(gn.error());
    if (auto r = adam_.step(net_->parameters()); !r) return r;
    st.total += l->total;
    st.policy += l->policy;
    st.value += l->value;
    st.entropy += l->entropy;
    st.approx_kl += l->approx_kl;
    st.clip_frac += l->clip_frac;
    st.grad_norm += *gn;
    kl_epoch += l->approx_kl;
    ++st.updates;
    return {};
  };
  for (int ep = 0; ep < cfg_.epochs; ++ep) {
    std::iota(perm.begin(), perm.end(), std::int64_t{0});
    for (std::int64_t i = units - 1; i > 0; --i)  // Fisher-Yates
      std::swap(perm[static_cast<std::size_t>(i)], perm[rng_.u32(static_cast<std::uint32_t>(i + 1))]);
    kl_epoch = 0;
    const int updates_before = st.updates;
    for (std::int64_t k = 0; k < mb; ++k) {
      const auto lo = units * k / mb, hi = units * (k + 1) / mb;
      if (hi <= lo) continue;
      std::span<const std::int64_t> group(perm.data() + lo, static_cast<std::size_t>(hi - lo));
      if (!rec) {
        auto m = buf.minibatch(group);
        if (!m) return std::unexpected(m.error());
        if (auto r = net_->zero_grad(); !r) return std::unexpected(r.error());
        ForwardCtx ctx;
        ctx.train = true;
        auto out = net_->forward(m->obs, ctx);
        if (!out) return std::unexpected(out.error());
        if (auto r = apply(ctx, *out, m->masks, m->batch, false); !r) return std::unexpected(r.error());
      } else {
        const auto T = buf.spec().steps;
        const auto L = cfg_.seq_len > 0 ? std::min<std::int64_t>(cfg_.seq_len, T) : T;
        for (std::int64_t t0 = 0; t0 < T; t0 += L) {
          auto sb = buf.seq_minibatch(group, t0, std::min(L, T - t0));
          if (!sb) return std::unexpected(sb.error());
          if (auto r = net_->zero_grad(); !r) return std::unexpected(r.error());
          ForwardCtx ctx;
          ctx.train = true;
          auto out = net_->forward_seq(sb->mb.obs, sb->h0, sb->reset, ctx);
          if (!out) return std::unexpected(out.error());
          if (auto r = apply(ctx, out->out, sb->mb.masks, sb->mb.batch, true); !r)
            return std::unexpected(r.error());
        }
      }
    }
    ++st.epochs_run;
    const int n_in_epoch = st.updates - updates_before;
    if (cfg_.target_kl > 0.f && n_in_epoch > 0 && kl_epoch / n_in_epoch > 1.5f * cfg_.target_kl) break;
  }
  if (st.updates > 0) {
    const float inv = 1.f / float(st.updates);
    st.total *= inv; st.policy *= inv; st.value *= inv; st.entropy *= inv;
    st.approx_kl *= inv; st.clip_frac *= inv; st.grad_norm *= inv;
  }
  return st;
}

}  // namespace gyre
