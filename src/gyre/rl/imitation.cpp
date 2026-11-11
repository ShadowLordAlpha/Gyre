#include "gyre/rl/imitation.hpp"

#include "gyre/ops.hpp"

#include <algorithm>
#include <numeric>

namespace gyre {
namespace {
Error err(const char* m) { return make_error(Errc::invalid_shape, m); }
}  // namespace

Result<StepRecord> make_demo_step(const Obs& obs, const Tensor& legal, const Tensor& active,
                                  const Tensor& actions, std::vector<std::uint8_t> reset,
                                  std::optional<Tensor> h_in) {
  if (obs.entities.rank() < 1) return std::unexpected(err("make_demo_step: entities rank"));
  const auto E = obs.entities.shape()[0];
  std::vector<float> z(static_cast<std::size_t>(E), 0.f);
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  std::int64_t sh[1] = {E};
  auto zt = Tensor::from_host(std::as_bytes(std::span<const float>(z)), sh, DType::f32, *d);
  if (!zt) return std::unexpected(zt.error());
  return StepRecord{obs, legal, active, actions, *zt, *zt, z, std::vector<std::uint8_t>(z.size(), 0),
                    std::move(h_in), std::move(reset)};
}

Result<ImitationTrainer> ImitationTrainer::create(PolicyNet& net, ImitationConfig cfg) {
  if (cfg.minibatches < 1) return std::unexpected(err("ImitationConfig"));
  auto adam = Adam::create(net.parameters(), cfg.lr);
  if (!adam) return std::unexpected(adam.error());
  return ImitationTrainer(net, cfg, std::move(*adam));
}

Result<ImitationStats> ImitationTrainer::epoch(RolloutBuffer& buf) { return run(buf, true); }
Result<ImitationStats> ImitationTrainer::evaluate(RolloutBuffer& buf) { return run(buf, false); }

Result<ImitationStats> ImitationTrainer::run(RolloutBuffer& buf, bool train) {
  if (!buf.full()) return std::unexpected(err("imitation: buffer not full"));
  if (cfg_.value_coef > 0.f && !buf.gae_computed())
    return std::unexpected(err("imitation: value_coef > 0 needs compute_gae (returns)"));
  const bool rec = net_->recurrent();
  if (rec != (buf.spec().hidden_dim > 0))
    return std::unexpected(err("imitation: policy and buffer disagree on recurrence"));
  const auto units = rec ? buf.spec().envs : buf.size();
  const auto mb = train ? std::min<std::int64_t>(cfg_.minibatches, units) : 1;
  std::vector<std::int64_t> perm(static_cast<std::size_t>(units));
  std::iota(perm.begin(), perm.end(), std::int64_t{0});
  if (train)
    for (std::int64_t i = units - 1; i > 0; --i)
      std::swap(perm[static_cast<std::size_t>(i)], perm[rng_.u32(static_cast<std::uint32_t>(i + 1))]);

  ImitationStats st;
  double nll_sum = 0, vloss_sum = 0, correct = 0, pairs = 0, samples = 0;
  const auto& L = net_->layout();
  for (std::int64_t k = 0; k < mb; ++k) {
    const auto lo = units * k / mb, hi = units * (k + 1) / mb;
    if (hi <= lo) continue;
    std::span<const std::int64_t> group(perm.data() + lo, static_cast<std::size_t>(hi - lo));
    // gather
    std::optional<Minibatch> flat;
    std::optional<SeqBatch> seq;
    if (rec) {
      auto sb = buf.seq_minibatch(group, 0, buf.spec().steps);
      if (!sb) return std::unexpected(sb.error());
      seq = std::move(*sb);
    } else {
      auto m = buf.minibatch(group);
      if (!m) return std::unexpected(m.error());
      flat = std::move(*m);
    }
    const Minibatch& m = rec ? seq->mb : *flat;
    if (train)
      if (auto r = net_->zero_grad(); !r) return std::unexpected(r.error());
    ForwardCtx ctx;
    ctx.train = train;
    std::optional<PolicyOut> out_o;
    if (rec) {
      auto o = net_->forward_seq(m.obs, seq->h0, seq->reset, ctx);
      if (!o) return std::unexpected(o.error());
      out_o = std::move(o->out);
    } else {
      auto o = net_->forward(m.obs, ctx);
      if (!o) return std::unexpected(o.error());
      out_o = std::move(*o);
    }
    PolicyOut& out = *out_o;
    auto il = imitation_loss(out.logits, L, m.masks, m.batch.actions);
    if (!il) return std::unexpected(il.error());
    const auto M = out.logits.shape()[0];
    nll_sum += double(il->loss) * double(M);
    samples += double(M);

    // accuracy over active (sample, head) pairs
    auto dist = head_dist(out.logits, L, m.masks);
    if (!dist) return std::unexpected(dist.error());
    auto greedy = sample_actions(*dist, L, m.masks, nullptr, true);
    if (!greedy) return std::unexpected(greedy.error());
    auto pg = greedy->host_span<std::int32_t>();
    auto pa = m.batch.actions.host_span<std::int32_t>();
    auto pact = m.masks.active.host_span<std::uint8_t>();
    for (std::size_t i = 0; i < pg->size(); ++i)
      if ((*pact)[i]) {
        pairs += 1;
        correct += ((*pg)[i] == (*pa)[i]) ? 1 : 0;
      }

    // optional value regression
    std::int64_t s1[1] = {M};
    auto dval = Tensor::zeros(s1, DType::f32, out.value.device());
    if (!dval) return std::unexpected(dval.error());
    if (cfg_.value_coef > 0.f) {
      auto pv = out.value.host_span<float>();
      auto pr = m.batch.returns.host_span<float>();
      auto pd = dval->host_span<float>();
      for (std::int64_t i = 0; i < M; ++i) {
        const double d = double((*pv)[static_cast<std::size_t>(i)]) - double((*pr)[static_cast<std::size_t>(i)]);
        vloss_sum += 0.5 * d * d;
        (*pd)[static_cast<std::size_t>(i)] = static_cast<float>(double(cfg_.value_coef) * d / double(M));
      }
    }
    if (train) {
      auto br = rec ? net_->backward_seq(il->dlogits, *dval, ctx) : net_->backward(il->dlogits, *dval, ctx);
      if (!br) return std::unexpected(br.error());
      auto gn = clip_grad_norm(net_->parameters(), cfg_.max_grad_norm);
      if (!gn) return std::unexpected(gn.error());
      if (auto r = adam_.step(net_->parameters()); !r) return std::unexpected(r.error());
      st.grad_norm += *gn;
      ++st.updates;
    }
  }
  if (samples > 0) {
    st.loss = static_cast<float>(nll_sum / samples);
    st.value_loss = static_cast<float>(vloss_sum / samples);
  }
  st.accuracy = pairs > 0 ? static_cast<float>(correct / pairs) : 0.f;
  if (st.updates > 0) st.grad_norm /= float(st.updates);
  return st;
}

}  // namespace gyre
