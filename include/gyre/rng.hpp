#pragma once

#include "gyre/error.hpp"

#include <cstdint>
#include <random>
#include <span>

namespace gyre {

class Rng {
 public:
  explicit Rng(std::uint64_t seed = 1) : gen_(seed), seed_(seed) {}

  std::uint64_t seed() const noexcept { return seed_; }
  std::mt19937_64& engine() noexcept { return gen_; }

  float uniform01();
  float normal(float mean, float stddev);
  std::uint64_t u64();
  std::uint32_t u32(std::uint32_t n);  // [0, n)

  Result<void> fill_u8(std::span<std::uint8_t> out, std::uint8_t lo, std::uint8_t hi);
  Result<void> fill_f32(std::span<float> out, float lo, float hi);

 private:
  std::mt19937_64 gen_;
  std::uint64_t seed_;
};

}  // namespace gyre
