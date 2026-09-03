#pragma once

#include "gyre/rng.hpp"
#include "gyre/tensor.hpp"

#include <cstdint>
#include <span>

namespace gyre::ga {

enum class GeneDType { u8, f32 };

struct Config {
  std::uint32_t n{128};
  std::uint32_t dim{64};
  GeneDType gene{GeneDType::u8};
  float mutation_sigma{0.1f};
  float crossover_p{0.7f};
  std::uint32_t elite{2};
  std::uint32_t tournament_k{3};
  std::uint64_t seed{1};
};

using FitnessPtr = float (*)(std::span<const float>);

struct Population {
  Tensor genes;
  Tensor fitness;
};

Result<Population> random_population(const Config& cfg, std::shared_ptr<Device> dev, Rng& rng);
Result<void> evaluate(Population& pop, FitnessPtr fitness);
Result<Population> step(const Population& pop, const Config& cfg, Rng& rng);

float onemax_fitness(std::span<const float> genes);

}  // namespace gyre::ga
