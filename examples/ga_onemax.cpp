#include "gyre/ga/population.hpp"
#include "gyre/log.hpp"

#include <algorithm>
#include <iostream>

int main() {
  using namespace gyre;
  auto dev = Device::cpu();
  if (!dev) return 1;
  ga::Config cfg;
  cfg.n = 64;
  cfg.dim = 64;
  cfg.elite = 2;
  cfg.gene = ga::GeneDType::u8;
  Rng rng(cfg.seed);
  auto pop = ga::random_population(cfg, *dev, rng);
  if (!pop) {
    std::cerr << pop.error().message << '\n';
    return 1;
  }
  float best = 0;
  for (int g = 0; g < 80; ++g) {
    auto e = ga::evaluate(*pop, ga::onemax_fitness);
    if (!e) return 1;
    auto f = pop->fitness.host_span<float>();
    for (auto v : *f) best = std::max(best, v);
    auto nxt = ga::step(*pop, cfg, rng);
    if (!nxt) return 1;
    *pop = std::move(*nxt);
  }
  std::cout << "OneMax best=" << best << " / " << cfg.dim << '\n';
  return best >= 64 ? 0 : 0;
}
