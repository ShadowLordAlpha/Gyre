#include "gyre/ga/population.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

TEST(GA, OneMaxImproves) {
  auto d = gyre::Device::cpu();
  gyre::ga::Config cfg;
  cfg.n = 32;
  cfg.dim = 16;
  cfg.elite = 2;
  gyre::Rng rng(1);
  auto pop = gyre::ga::random_population(cfg, *d, rng);
  ASSERT_TRUE(pop);
  ASSERT_TRUE(gyre::ga::evaluate(*pop, gyre::ga::onemax_fitness));
  auto f0 = pop->fitness.host_span<float>();
  float b0 = 0;
  for (auto v : *f0) b0 = std::max(b0, v);
  for (int i = 0; i < 40; ++i) {
    auto n = gyre::ga::step(*pop, cfg, rng);
    ASSERT_TRUE(n);
    *pop = std::move(*n);
    ASSERT_TRUE(gyre::ga::evaluate(*pop, gyre::ga::onemax_fitness));
  }
  auto f1 = pop->fitness.host_span<float>();
  float b1 = 0;
  for (auto v : *f1) b1 = std::max(b1, v);
  EXPECT_GE(b1, b0);
}

TEST(GA, StepRejectsConfigThatDoesNotMatchPopulation) {
  auto d = gyre::Device::cpu();
  gyre::ga::Config cfg;
  cfg.n = 8;
  cfg.dim = 4;
  cfg.elite = 1;
  gyre::Rng rng(3);
  auto pop = gyre::ga::random_population(cfg, *d, rng);
  ASSERT_TRUE(pop);
  ASSERT_TRUE(gyre::ga::evaluate(*pop, gyre::ga::onemax_fitness));
  for (auto mut : {0, 1, 2}) {
    auto bad = cfg;
    if (mut == 0) bad.n = 64;    // more rows than the population has
    if (mut == 1) bad.dim = 32;  // wider rows than the population has
    if (mut == 2) bad.gene = gyre::ga::GeneDType::f32;
    auto n = gyre::ga::step(*pop, bad, rng);
    ASSERT_FALSE(n) << "mismatch " << mut << " was accepted";
    EXPECT_EQ(n.error().code, gyre::Errc::invalid_shape);
  }
  EXPECT_TRUE(gyre::ga::step(*pop, cfg, rng));
}

namespace {
float nan_every_third(std::span<const float> genes) {
  // Distinct per row (sum of genes), with a NaN for some rows: what a 0/0 in a
  // user fitness callback looks like.
  float s = 0.f;
  for (auto g : genes) s += g;
  return (static_cast<int>(s * 7.f) % 3 == 0) ? std::nanf("") : s;
}
}  // namespace

TEST(GA, NanFitnessRanksLastAndElitesAreFinite) {
  auto d = gyre::Device::cpu();
  gyre::ga::Config cfg;
  cfg.n = 256;  // large enough for std::sort to use its unguarded partition loops
  cfg.dim = 8;
  cfg.gene = gyre::ga::GeneDType::f32;
  cfg.elite = 4;
  gyre::Rng rng(5);
  auto pop = gyre::ga::random_population(cfg, *d, rng);
  ASSERT_TRUE(pop);
  ASSERT_TRUE(gyre::ga::evaluate(*pop, nan_every_third));
  auto f = pop->fitness.host_span<float>();
  ASSERT_TRUE(f);
  int nans = 0;
  for (auto v : *f) nans += std::isnan(v) ? 1 : 0;
  ASSERT_GT(nans, 0);
  ASSERT_LT(nans, static_cast<int>(cfg.n) - static_cast<int>(cfg.elite));
  auto next = gyre::ga::step(*pop, cfg, rng);
  ASSERT_TRUE(next) << next.error().message;
  // Elites are copied verbatim into rows 0..elite-1: they must be the best
  // finite rows, never a NaN row.
  auto g = next->genes.host_span<float>();
  ASSERT_TRUE(g);
  std::vector<float> top;
  for (auto v : *f)
    if (!std::isnan(v)) top.push_back(v);
  std::sort(top.begin(), top.end(), std::greater<>());
  for (std::uint32_t e = 0; e < cfg.elite; ++e) {
    std::span<const float> row(g->data() + e * cfg.dim, cfg.dim);
    const float fe = nan_every_third(row);
    EXPECT_FALSE(std::isnan(fe)) << "elite " << e;
    EXPECT_FLOAT_EQ(fe, top[e]) << "elite " << e;
  }
}
