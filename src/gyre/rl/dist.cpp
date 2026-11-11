#include "gyre/rl/dist.hpp"

#include "gyre/rl/ops.hpp"
#include "gyre/rng.hpp"

#include <algorithm>
#include <cmath>

namespace gyre {
namespace {

Error err(Errc c, const char* m) { return make_error(c, m); }

struct Views {
  std::span<const std::uint8_t> legal, active;
  std::int64_t B{0};
};

Result<Views> views(const HeadLayout& L, const HeadMasks& m) {
  if (m.legal.dtype() != DType::u8 || m.active.dtype() != DType::u8 || m.legal.rank() != 2 ||
      m.active.rank() != 2 || m.legal.shape()[1] != L.total() || m.active.shape()[1] != L.heads() ||
      m.legal.shape()[0] != m.active.shape()[0])
    return std::unexpected(err(Errc::invalid_shape, "masks: legal u8 [B,total], active u8 [B,H]"));
  auto pl = m.legal.host_span<std::uint8_t>();
  auto pa = m.active.host_span<std::uint8_t>();
  if (!pl || !pa) return std::unexpected(pl ? pa.error() : pl.error());
  return Views{*pl, *pa, m.legal.shape()[0]};
}

}  // namespace

Result<HeadLayout> HeadLayout::make(std::vector<std::int64_t> sizes) {
  if (sizes.empty()) return std::unexpected(err(Errc::invalid_shape, "HeadLayout: no heads"));
  for (auto s : sizes)
    if (s < 1) return std::unexpected(err(Errc::invalid_shape, "HeadLayout: size < 1"));
  HeadLayout l;
  l.sizes = std::move(sizes);
  return l;
}

std::int64_t HeadLayout::total() const {
  std::int64_t t = 0;
  for (auto s : sizes) t += s;
  return t;
}

std::int64_t HeadLayout::offset(std::int64_t h) const {
  std::int64_t t = 0;
  for (std::int64_t i = 0; i < h; ++i) t += sizes[static_cast<std::size_t>(i)];
  return t;
}

Result<HeadDist> head_dist(const Tensor& logits, const HeadLayout& L, const HeadMasks& m) {
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  if (logits.dtype() != DType::f32 || logits.rank() != 2 || logits.shape()[0] != v->B ||
      logits.shape()[1] != L.total())
    return std::unexpected(err(Errc::invalid_shape, "logits [B,total]"));
  auto pz = logits.host_span<float>();
  if (!pz) return std::unexpected(pz.error());
  const auto T = L.total();
  auto lp = Tensor::zeros(logits.shape(), DType::f32, logits.device());
  auto pr = Tensor::zeros(logits.shape(), DType::f32, logits.device());
  if (!lp || !pr) return std::unexpected(err(Errc::invalid_shape, "alloc"));
  auto plp = lp->host_span<float>();
  auto ppr = pr->host_span<float>();
  for (std::int64_t b = 0; b < v->B; ++b) {
    for (std::int64_t h = 0; h < L.heads(); ++h) {
      if (!v->active[static_cast<std::size_t>(b * L.heads() + h)]) continue;
      const auto off = L.offset(h), n = L.sizes[static_cast<std::size_t>(h)];
      const auto base = static_cast<std::size_t>(b * T + off);
      float mx = -INFINITY;
      bool any = false;
      for (std::int64_t j = 0; j < n; ++j)
        if (v->legal[base + j]) {
          any = true;
          if (!std::isfinite((*pz)[base + j]))
            return std::unexpected(err(Errc::invalid_shape, "non-finite logit in an active head"));
          mx = std::max(mx, (*pz)[base + j]);
        }
      if (!any) return std::unexpected(err(Errc::invalid_shape, "active head has no legal option"));
      double s = 0;
      for (std::int64_t j = 0; j < n; ++j)
        if (v->legal[base + j]) s += std::exp(double((*pz)[base + j]) - mx);
      const float ls = static_cast<float>(std::log(s));
      for (std::int64_t j = 0; j < n; ++j) {
        if (v->legal[base + j]) {
          const float l = ((*pz)[base + j] - mx) - ls;
          (*plp)[base + j] = l;
          (*ppr)[base + j] = std::exp(l);
        } else {
          (*plp)[base + j] = kMaskFill;
        }
      }
    }
  }
  return HeadDist{std::move(*lp), std::move(*pr)};
}

Result<Tensor> sample_actions(const HeadDist& d, const HeadLayout& L, const HeadMasks& m, Rng* rng,
                              bool greedy) {
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  if (!greedy && !rng) return std::unexpected(err(Errc::unsupported, "sampling needs an Rng"));
  auto pp = d.prob.host_span<float>();
  if (!pp) return std::unexpected(pp.error());
  std::int64_t sh[2] = {v->B, L.heads()};
  auto out = Tensor::empty(sh, DType::i32, d.prob.device());
  if (!out) return std::unexpected(out.error());
  auto po = out->host_span<std::int32_t>();
  const auto T = L.total();
  for (std::int64_t b = 0; b < v->B; ++b) {
    for (std::int64_t h = 0; h < L.heads(); ++h) {
      auto& slot = (*po)[static_cast<std::size_t>(b * L.heads() + h)];
      if (!v->active[static_cast<std::size_t>(b * L.heads() + h)]) {
        slot = -1;
        continue;
      }
      const auto off = L.offset(h), n = L.sizes[static_cast<std::size_t>(h)];
      const auto base = static_cast<std::size_t>(b * T + off);
      std::int64_t pick = -1, last_legal = -1;
      if (greedy) {
        float best = -1.f;
        for (std::int64_t j = 0; j < n; ++j)
          if (v->legal[base + j] && (*pp)[base + j] > best) {
            best = (*pp)[base + j];
            pick = j;
          }
      } else {
        const float u = rng->uniform01();
        float acc = 0.f;
        for (std::int64_t j = 0; j < n && pick < 0; ++j) {
          if (!v->legal[base + j]) continue;
          last_legal = j;
          acc += (*pp)[base + j];
          if (u < acc) pick = j;
        }
        if (pick < 0) pick = last_legal;  // rounding: u landed past the accumulated mass
      }
      slot = static_cast<std::int32_t>(pick);
    }
  }
  return out;
}

Result<SampleOut> sample_conditional(const Tensor& logits, const HeadLayout& L, const Tensor& legal,
                                     const Tensor& initial_active, Rng* rng, bool greedy,
                                     const ActiveFn& after_head) {
  HeadMasks m{legal, initial_active};
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  if (!greedy && !rng) return std::unexpected(err(Errc::unsupported, "sampling needs an Rng"));
  if (logits.dtype() != DType::f32 || logits.rank() != 2 || logits.shape()[0] != v->B ||
      logits.shape()[1] != L.total())
    return std::unexpected(err(Errc::invalid_shape, "logits [B,total]"));
  auto pz = logits.host_span<float>();
  if (!pz) return std::unexpected(pz.error());
  const auto B = v->B, H = L.heads(), T = L.total();
  std::int64_t sh[2] = {B, H};
  auto acts = Tensor::empty(sh, DType::i32, logits.device());
  auto act = Tensor::empty(sh, DType::u8, logits.device());
  if (!acts || !act) return std::unexpected(err(Errc::invalid_shape, "alloc"));
  auto pa = acts->host_span<std::int32_t>();
  auto pf = act->host_span<std::uint8_t>();
  std::copy(v->active.begin(), v->active.end(), pf->begin());
  for (auto& a : *pa) a = -1;
  std::vector<double> pr;
  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t h = 0; h < H; ++h) {
      if (!(*pf)[static_cast<std::size_t>(b * H + h)]) continue;
      const auto off = L.offset(h), n = L.sizes[static_cast<std::size_t>(h)];
      const auto base = static_cast<std::size_t>(b * T + off);
      float mx = -INFINITY;
      bool any_legal = false;
      for (std::int64_t j = 0; j < n; ++j)
        if (v->legal[base + j]) {
          any_legal = true;
          if (!std::isfinite((*pz)[base + j]))
            return std::unexpected(err(Errc::invalid_shape, "non-finite logit in an active head"));
          mx = std::max(mx, (*pz)[base + j]);
        }
      if (!any_legal) return std::unexpected(err(Errc::invalid_shape, "active head has no legal option"));
      pr.assign(static_cast<std::size_t>(n), 0.0);
      double s = 0;
      for (std::int64_t j = 0; j < n; ++j)
        if (v->legal[base + j]) s += (pr[static_cast<std::size_t>(j)] = std::exp(double((*pz)[base + j]) - mx));
      std::int64_t pick = -1, last_legal = -1;
      if (greedy) {
        double best = -1;
        for (std::int64_t j = 0; j < n; ++j)
          if (v->legal[base + j] && pr[static_cast<std::size_t>(j)] > best) {
            best = pr[static_cast<std::size_t>(j)];
            pick = j;
          }
      } else {
        const double u = double(rng->uniform01()) * s;
        double acc = 0;
        for (std::int64_t j = 0; j < n && pick < 0; ++j) {
          if (!v->legal[base + j]) continue;
          last_legal = j;
          acc += pr[static_cast<std::size_t>(j)];
          if (u < acc) pick = j;
        }
        if (pick < 0) pick = last_legal;
      }
      (*pa)[static_cast<std::size_t>(b * H + h)] = static_cast<std::int32_t>(pick);
      if (after_head) {
        // The callback may only change heads AFTER h: flags of sampled/current heads are restored.
        std::vector<std::uint8_t> keep(pf->begin() + b * H, pf->begin() + b * H + h + 1);
        after_head(b, h, std::span<const std::int32_t>(pa->data() + b * H, static_cast<std::size_t>(H)),
                   std::span<std::uint8_t>(pf->data() + b * H, static_cast<std::size_t>(H)));
        std::copy(keep.begin(), keep.end(), pf->begin() + b * H);
      }
    }
  }
  return SampleOut{std::move(*acts), std::move(*act)};
}

namespace {
Result<std::span<const std::int32_t>> check_actions(const Tensor& a, const HeadLayout& L, std::int64_t B) {
  if (a.dtype() != DType::i32 || a.rank() != 2 || a.shape()[0] != B || a.shape()[1] != L.heads())
    return std::unexpected(err(Errc::invalid_shape, "actions i32 [B,H]"));
  return a.host_span<std::int32_t>();
}
}  // namespace

Result<Tensor> log_prob(const HeadDist& d, const HeadLayout& L, const HeadMasks& m,
                        const Tensor& actions) {
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  auto pa = check_actions(actions, L, v->B);
  if (!pa) return std::unexpected(pa.error());
  auto plp = d.logp.host_span<float>();
  if (!plp) return std::unexpected(plp.error());
  std::int64_t sh[1] = {v->B};
  auto out = Tensor::zeros(sh, DType::f32, d.logp.device());
  if (!out) return std::unexpected(out.error());
  auto po = out->host_span<float>();
  const auto T = L.total();
  for (std::int64_t b = 0; b < v->B; ++b) {
    double s = 0;
    for (std::int64_t h = 0; h < L.heads(); ++h) {
      if (!v->active[static_cast<std::size_t>(b * L.heads() + h)]) continue;
      const auto a = (*pa)[static_cast<std::size_t>(b * L.heads() + h)];
      if (a < 0 || a >= L.sizes[static_cast<std::size_t>(h)])
        return std::unexpected(err(Errc::invalid_shape, "action out of range for active head"));
      const auto idx = static_cast<std::size_t>(b * T + L.offset(h) + a);
      if (!v->legal[idx]) return std::unexpected(err(Errc::invalid_shape, "action is illegal"));
      s += (*plp)[idx];
    }
    (*po)[static_cast<std::size_t>(b)] = static_cast<float>(s);
  }
  return out;
}

Result<Tensor> entropy(const HeadDist& d, const HeadLayout& L, const HeadMasks& m) {
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  auto plp = d.logp.host_span<float>();
  auto ppr = d.prob.host_span<float>();
  if (!plp || !ppr) return std::unexpected(plp ? ppr.error() : plp.error());
  std::int64_t sh[1] = {v->B};
  auto out = Tensor::zeros(sh, DType::f32, d.logp.device());
  if (!out) return std::unexpected(out.error());
  auto po = out->host_span<float>();
  const auto T = L.total();
  for (std::int64_t b = 0; b < v->B; ++b) {
    double s = 0;
    for (std::int64_t h = 0; h < L.heads(); ++h) {
      if (!v->active[static_cast<std::size_t>(b * L.heads() + h)]) continue;
      const auto off = L.offset(h), n = L.sizes[static_cast<std::size_t>(h)];
      for (std::int64_t j = 0; j < n; ++j) {
        const auto i = static_cast<std::size_t>(b * T + off + j);
        if (v->legal[i]) s -= double((*ppr)[i]) * double((*plp)[i]);
      }
    }
    (*po)[static_cast<std::size_t>(b)] = static_cast<float>(s);
  }
  return out;
}

Result<Tensor> log_prob_entropy_backward(const HeadDist& d, const HeadLayout& L, const HeadMasks& m,
                                         const Tensor& actions, const Tensor& dlogp,
                                         const Tensor& dent) {
  auto v = views(L, m);
  if (!v) return std::unexpected(v.error());
  auto pa = check_actions(actions, L, v->B);
  if (!pa) return std::unexpected(pa.error());
  if (dlogp.rank() != 1 || dent.rank() != 1 || dlogp.shape()[0] != v->B || dent.shape()[0] != v->B)
    return std::unexpected(err(Errc::invalid_shape, "dlogp/dent [B]"));
  auto plp = d.logp.host_span<float>();
  auto ppr = d.prob.host_span<float>();
  auto pdl = dlogp.host_span<float>();
  auto pde = dent.host_span<float>();
  if (!plp || !ppr || !pdl || !pde) return std::unexpected(err(Errc::not_cpu, "host"));
  auto out = Tensor::zeros(d.logp.shape(), DType::f32, d.logp.device());
  if (!out) return std::unexpected(out.error());
  auto po = out->host_span<float>();
  const auto T = L.total();
  for (std::int64_t b = 0; b < v->B; ++b) {
    const float gl = (*pdl)[static_cast<std::size_t>(b)], ge = (*pde)[static_cast<std::size_t>(b)];
    for (std::int64_t h = 0; h < L.heads(); ++h) {
      if (!v->active[static_cast<std::size_t>(b * L.heads() + h)]) continue;
      const auto off = L.offset(h), n = L.sizes[static_cast<std::size_t>(h)];
      const auto a = (*pa)[static_cast<std::size_t>(b * L.heads() + h)];
      if (a < 0 || a >= n) return std::unexpected(err(Errc::invalid_shape, "action out of range"));
      // head entropy H_h = -sum p logp
      double Hh = 0;
      for (std::int64_t j = 0; j < n; ++j) {
        const auto i = static_cast<std::size_t>(b * T + off + j);
        if (v->legal[i]) Hh -= double((*ppr)[i]) * double((*plp)[i]);
      }
      for (std::int64_t j = 0; j < n; ++j) {
        const auto i = static_cast<std::size_t>(b * T + off + j);
        if (!v->legal[i]) continue;
        const float p = (*ppr)[i];
        // d logp(a)/dz_j = 1[j==a] - p_j ;  dH/dz_j = -p_j (logp_j + H)
        (*po)[i] = gl * ((j == a ? 1.f : 0.f) - p) - ge * p * ((*plp)[i] + static_cast<float>(Hh));
      }
    }
  }
  return out;
}

}  // namespace gyre
