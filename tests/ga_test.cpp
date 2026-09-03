#include "gyre/ga/population.hpp"

#include <gtest/gtest.h>

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
