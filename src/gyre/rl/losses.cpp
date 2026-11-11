#include "gyre/rl/losses.hpp"

#include <cmath>

namespace gyre {
namespace {
Error err(const char* m) { return make_error(Errc::invalid_shape, m); }

Result<std::span<const float>> vec(const Tensor& t, std::int64_t B, const char* what) {
  if (t.dtype() != DType::f32 || t.rank() != 1 || t.shape()[0] != B)
    return std::unexpected(err(what));
  return t.host_span<float>();
}
}  // namespace

Result<PpoLoss> ppo_loss(const Tensor& logits, const Tensor& value, const HeadLayout& L,
                         const HeadMasks& m, const PpoBatch& batch, const PpoConfig& cfg) {
  if (logits.rank() != 2) return std::unexpected(err("logits [B,total]"));
  const auto B = logits.shape()[0];
  if (B < 1) return std::unexpected(err("empty batch"));
  auto dist = head_dist(logits, L, m);
  if (!dist) return std::unexpected(dist.error());
  auto lp = log_prob(*dist, L, m, batch.actions);
  auto ent = entropy(*dist, L, m);
  if (!lp || !ent) return std::unexpected(lp ? ent.error() : lp.error());
  auto plp = lp->host_span<float>();
  auto pent = ent->host_span<float>();
  auto pold = vec(batch.old_logp, B, "old_logp [B]");
  auto padv = vec(batch.advantages, B, "advantages [B]");
  auto pret = vec(batch.returns, B, "returns [B]");
  auto pval = vec(value, B, "value [B]");
  if (!pold || !padv || !pret || !pval)
    return std::unexpected(!pold ? pold.error() : !padv ? padv.error() : !pret ? pret.error() : pval.error());

  std::int64_t s1[1] = {B};
  auto dlp = Tensor::zeros(s1, DType::f32, logits.device());
  auto dent = Tensor::zeros(s1, DType::f32, logits.device());
  auto dval = Tensor::zeros(s1, DType::f32, logits.device());
  if (!dlp || !dent || !dval) return std::unexpected(err("alloc"));
  auto pdlp = dlp->host_span<float>();
  auto pdent = dent->host_span<float>();
  auto pdval = dval->host_span<float>();

  const double invB = 1.0 / double(B);
  double pol = 0, vl = 0, en = 0, kl = 0;
  std::int64_t nclip = 0;
  for (std::int64_t i = 0; i < B; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double r = std::exp(double((*plp)[u]) - double((*pold)[u]));
    const double A = (*padv)[u];
    const double a = r * A;
    const double bclip = std::min(std::max(r, 1.0 - cfg.clip), 1.0 + cfg.clip) * A;
    pol += -std::min(a, bclip);
    // d(-min(a,b))/dlogp : unclipped branch is selected whenever a <= b
    (*pdlp)[u] = static_cast<float>((a <= bclip ? -A * r : 0.0) * invB);
    kl += (r - 1.0) - std::log(r);
    if (std::abs(r - 1.0) > cfg.clip) ++nclip;
    const double dv = double((*pval)[u]) - double((*pret)[u]);
    vl += 0.5 * dv * dv;
    (*pdval)[u] = static_cast<float>(cfg.vf_coef * dv * invB);
    en += (*pent)[u];
    (*pdent)[u] = static_cast<float>(-cfg.ent_coef * invB);
  }
  auto dlogits = log_prob_entropy_backward(*dist, L, m, batch.actions, *dlp, *dent);
  if (!dlogits) return std::unexpected(dlogits.error());
  PpoLoss out{.dlogits = std::move(*dlogits), .dvalue = std::move(*dval)};
  out.policy = static_cast<float>(pol * invB);
  out.value = static_cast<float>(vl * invB);
  out.entropy = static_cast<float>(en * invB);
  out.total = out.policy + cfg.vf_coef * out.value - cfg.ent_coef * out.entropy;
  out.approx_kl = static_cast<float>(kl * invB);
  out.clip_frac = static_cast<float>(double(nclip) * invB);
  return out;
}

Result<void> normalize_advantages_(Tensor& adv, float eps) {
  if (adv.dtype() != DType::f32 || adv.rank() != 1) return std::unexpected(err("advantages [B]"));
  auto p = adv.host_span<float>();
  if (!p) return std::unexpected(p.error());
  if (p->empty()) return {};
  double mean = 0;
  for (float v : *p) mean += v;
  mean /= double(p->size());
  double var = 0;
  for (float v : *p) var += (v - mean) * (v - mean);
  var /= double(p->size());
  const double inv = 1.0 / (std::sqrt(var) + eps);
  for (auto& v : *p) v = static_cast<float>((v - mean) * inv);
  return {};
}

Result<ImitationLoss> imitation_loss(const Tensor& logits, const HeadLayout& L, const HeadMasks& m,
                                     const Tensor& actions) {
  if (logits.rank() != 2) return std::unexpected(err("logits [B,total]"));
  const auto B = logits.shape()[0];
  if (B < 1) return std::unexpected(err("empty batch"));
  auto dist = head_dist(logits, L, m);
  if (!dist) return std::unexpected(dist.error());
  auto lp = log_prob(*dist, L, m, actions);
  if (!lp) return std::unexpected(lp.error());
  std::int64_t s1[1] = {B};
  auto dlp = Tensor::zeros(s1, DType::f32, logits.device());
  auto dent = Tensor::zeros(s1, DType::f32, logits.device());
  if (!dlp || !dent) return std::unexpected(err("alloc"));
  double nll = 0;
  auto plp = lp->host_span<float>();
  auto pdlp = dlp->host_span<float>();
  for (std::int64_t i = 0; i < B; ++i) {
    nll -= (*plp)[static_cast<std::size_t>(i)];
    (*pdlp)[static_cast<std::size_t>(i)] = static_cast<float>(-1.0 / double(B));
  }
  auto dlogits = log_prob_entropy_backward(*dist, L, m, actions, *dlp, *dent);
  if (!dlogits) return std::unexpected(dlogits.error());
  return ImitationLoss{static_cast<float>(nll / double(B)), std::move(*dlogits)};
}

}  // namespace gyre
