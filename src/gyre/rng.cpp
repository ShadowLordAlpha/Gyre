#include "gyre/rng.hpp"

namespace gyre {

float Rng::uniform01() {
  std::uniform_real_distribution<float> d(0.f, 1.f);
  return d(gen_);
}

float Rng::normal(float mean, float stddev) {
  std::normal_distribution<float> d(mean, stddev);
  return d(gen_);
}

std::uint64_t Rng::u64() { return gen_(); }

std::uint32_t Rng::u32(std::uint32_t n) {
  if (n == 0) return 0;
  std::uniform_int_distribution<std::uint32_t> d(0, n - 1);
  return d(gen_);
}

Result<void> Rng::fill_u8(std::span<std::uint8_t> out, std::uint8_t lo, std::uint8_t hi) {
  if (lo > hi) return std::unexpected(make_error(Errc::invalid_shape, "fill_u8 range"));
  std::uniform_int_distribution<int> d(lo, hi);
  for (auto& x : out) x = static_cast<std::uint8_t>(d(gen_));
  return {};
}

Result<void> Rng::fill_f32(std::span<float> out, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : out) x = d(gen_);
  return {};
}

}  // namespace gyre
