#include "gyre/ga/population.hpp"
#include "gyre/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace gyre::ga {

float onemax_fitness(std::span<const float> genes) {
  float s = 0;
  for (auto g : genes) s += (g > 0.5f) ? 1.f : 0.f;
  return s;
}

Result<Population> random_population(const Config& cfg, std::shared_ptr<Device> dev, Rng& rng) {
  if (cfg.elite >= cfg.n) {
    return std::unexpected(make_error(Errc::invalid_shape, "elite >= n"));
  }
  std::int64_t gsh[2] = {cfg.n, cfg.dim};
  std::int64_t fsh[1] = {cfg.n};
  if (cfg.gene == GeneDType::u8) {
    auto genes = Tensor::empty(gsh, DType::u8, dev);
    auto fit = Tensor::zeros(fsh, DType::f32, dev);
    if (!genes || !fit) return std::unexpected(genes ? fit.error() : genes.error());
    auto p = genes->host_span<std::uint8_t>();
    if (!p) return std::unexpected(p.error());
    auto r = rng.fill_u8(*p, 0, 1);
    if (!r) return std::unexpected(r.error());
    return Population{std::move(*genes), std::move(*fit)};
  }
  auto genes = Tensor::empty(gsh, DType::f32, dev);
  auto fit = Tensor::zeros(fsh, DType::f32, dev);
  if (!genes || !fit) return std::unexpected(genes ? fit.error() : genes.error());
  auto p = genes->host_span<float>();
  if (!p) return std::unexpected(p.error());
  auto r = rng.fill_f32(*p, 0.f, 1.f);
  if (!r) return std::unexpected(r.error());
  return Population{std::move(*genes), std::move(*fit)};
}

Result<void> evaluate(Population& pop, FitnessPtr fitness) {
  auto f = pop.fitness.host_span<float>();
  if (!f) return std::unexpected(f.error());
  const auto N = pop.genes.shape()[0];
  const auto D = pop.genes.shape()[1];
  std::vector<float> row(static_cast<std::size_t>(D));
  if (pop.genes.dtype() == DType::u8) {
    auto g = pop.genes.host_span<std::uint8_t>();
    if (!g) return std::unexpected(g.error());
    for (std::int64_t i = 0; i < N; ++i) {
      for (std::int64_t d = 0; d < D; ++d)
        row[static_cast<std::size_t>(d)] = (*g)[static_cast<std::size_t>(i * D + d)];
      (*f)[static_cast<std::size_t>(i)] = fitness(row);
    }
  } else {
    auto g = pop.genes.host_span<float>();
    if (!g) return std::unexpected(g.error());
    for (std::int64_t i = 0; i < N; ++i) {
      for (std::int64_t d = 0; d < D; ++d)
        row[static_cast<std::size_t>(d)] = (*g)[static_cast<std::size_t>(i * D + d)];
      (*f)[static_cast<std::size_t>(i)] = fitness(row);
    }
  }
  return {};
}

namespace {

// NaN fitness (a 0/0 in a user callback) ranks below every real value. A raw
// `a > b` with NaN breaks std::sort's strict weak ordering (undefined behaviour).
float rank_key(float v) noexcept {
  return std::isnan(v) ? -std::numeric_limits<float>::infinity() : v;
}

std::uint32_t tournament(std::span<const float> f, const Config& cfg, Rng& rng) {
  std::uint32_t best = rng.u32(cfg.n);
  float bf = rank_key(f[best]);
  for (std::uint32_t t = 1; t < cfg.tournament_k; ++t) {
    auto i = rng.u32(cfg.n);
    if (rank_key(f[i]) > bf) {
      bf = rank_key(f[i]);
      best = i;
    }
  }
  return best;
}

// step() indexes genes/fitness with cfg.n and cfg.dim, so they must describe
// this population exactly (a reconfigured or resumed run could pass another cfg).
Result<void> check_matches(const Population& pop, const Config& cfg) {
  if (cfg.n == 0 || cfg.dim == 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "ga: n and dim must be > 0"));
  }
  const auto want_dt = cfg.gene == GeneDType::u8 ? DType::u8 : DType::f32;
  const auto gs = pop.genes.shape();
  const auto fs = pop.fitness.shape();
  if (pop.genes.rank() != 2 || gs[0] != static_cast<std::int64_t>(cfg.n) ||
      gs[1] != static_cast<std::int64_t>(cfg.dim) || pop.genes.dtype() != want_dt) {
    return std::unexpected(make_error(
        Errc::invalid_shape, "ga: population genes do not match cfg (n=" + std::to_string(cfg.n) +
                                 ", dim=" + std::to_string(cfg.dim) + ", gene dtype)"));
  }
  if (pop.fitness.rank() != 1 || fs[0] != static_cast<std::int64_t>(cfg.n) ||
      pop.fitness.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::invalid_shape, "ga: fitness must be f32 [n]"));
  }
  return {};
}

}  // namespace

Result<Population> step(const Population& pop, const Config& cfg, Rng& rng) {
  if (cfg.elite >= cfg.n) {
    return std::unexpected(make_error(Errc::invalid_shape, "elite >= n"));
  }
  if (auto ok = check_matches(pop, cfg); !ok) return std::unexpected(ok.error());
  auto next = pop.genes.clone();
  auto nfit = pop.fitness.clone();
  if (!next || !nfit) return std::unexpected(next ? nfit.error() : next.error());

  const auto N = cfg.n;
  const auto D = cfg.dim;
  std::vector<std::uint32_t> order(N);
  for (std::uint32_t i = 0; i < N; ++i) order[i] = i;
  auto f = pop.fitness.host_span<float>();
  if (!f) return std::unexpected(f.error());
  const std::span<const float> fit(*f);
  std::sort(order.begin(), order.end(),
            [&](auto a, auto b) { return rank_key(fit[a]) > rank_key(fit[b]); });

  if (pop.genes.dtype() == DType::u8) {
    auto src = pop.genes.host_span<std::uint8_t>();
    auto dst = next->host_span<std::uint8_t>();
    if (!src || !dst) return std::unexpected(make_error(Errc::not_cpu, "host"));
    for (std::uint32_t e = 0; e < cfg.elite; ++e) {
      auto row = order[e];
      std::copy(src->begin() + row * D, src->begin() + (row + 1) * D, dst->begin() + e * D);
    }
    const float pflip = (cfg.mutation_sigma > 0.f && cfg.mutation_sigma <= 1.f)
                            ? cfg.mutation_sigma
                            : 1.f / static_cast<float>(D);
    for (std::uint32_t i = cfg.elite; i < N; ++i) {
      auto p0 = tournament(fit, cfg, rng);
      auto p1 = tournament(fit, cfg, rng);
      if (rng.uniform01() < cfg.crossover_p) {
        for (std::uint32_t d = 0; d < D; ++d) {
          auto from = rng.uniform01() < 0.5f ? p0 : p1;
          (*dst)[i * D + d] = (*src)[from * D + d];
        }
      } else {
        std::copy(src->begin() + p0 * D, src->begin() + (p0 + 1) * D, dst->begin() + i * D);
      }
      for (std::uint32_t d = 0; d < D; ++d) {
        if (rng.uniform01() < pflip) (*dst)[i * D + d] = static_cast<std::uint8_t>(1 - (*dst)[i * D + d]);
      }
    }
  } else {
    auto src = pop.genes.host_span<float>();
    auto dst = next->host_span<float>();
    if (!src || !dst) return std::unexpected(make_error(Errc::not_cpu, "host"));
    for (std::uint32_t e = 0; e < cfg.elite; ++e) {
      auto row = order[e];
      std::copy(src->begin() + row * D, src->begin() + (row + 1) * D, dst->begin() + e * D);
    }
    for (std::uint32_t i = cfg.elite; i < N; ++i) {
      auto p0 = tournament(fit, cfg, rng);
      auto p1 = tournament(fit, cfg, rng);
      if (rng.uniform01() < cfg.crossover_p) {
        for (std::uint32_t d = 0; d < D; ++d)
          (*dst)[i * D + d] = 0.5f * ((*src)[p0 * D + d] + (*src)[p1 * D + d]);
      } else {
        std::copy(src->begin() + p0 * D, src->begin() + (p0 + 1) * D, dst->begin() + i * D);
      }
      for (std::uint32_t d = 0; d < D; ++d) (*dst)[i * D + d] += rng.normal(0.f, cfg.mutation_sigma);
    }
  }
  auto z = fill_zero(*nfit);
  if (!z) return std::unexpected(z.error());
  return Population{std::move(*next), std::move(*nfit)};
}

}  // namespace gyre::ga
