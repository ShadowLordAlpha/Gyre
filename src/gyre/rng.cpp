#include "gyre/rng.hpp"

#include <cmath>

// Distributions are hand-rolled on top of mt19937_64 (whose output is fixed by the C++
// standard). std::*_distribution output is implementation-defined, so it is not used.
namespace gyre {

float Rng::uniform01() {
  // 24 random bits -> exactly representable float in [0, 1).
  return static_cast<float>(gen_() >> 40) * (1.0f / 16777216.0f);
}

float Rng::normal(float mean, float stddev) {
  if (has_spare_) {
    has_spare_ = false;
    return mean + stddev * spare_;
  }
  // Box-Muller; u1 in (0, 1] so log() is finite.
  const double u1 = (static_cast<double>(gen_() >> 11) + 1.0) * (1.0 / 9007199254740992.0);
  const double u2 = static_cast<double>(gen_() >> 11) * (1.0 / 9007199254740992.0);
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double th = 6.283185307179586476925286766559 * u2;
  spare_ = static_cast<float>(r * std::sin(th));
  has_spare_ = true;
  return mean + stddev * static_cast<float>(r * std::cos(th));
}

std::uint64_t Rng::u64() { return gen_(); }

std::uint32_t Rng::u32(std::uint32_t n) {
  if (n == 0) return 0;
  // Unbiased rejection sampling on the 64-bit output.
  const std::uint64_t range = n;
  const std::uint64_t limit = (~std::uint64_t{0}) - ((~std::uint64_t{0}) % range) - 1;
  std::uint64_t x = gen_();
  while (x > limit) x = gen_();
  return static_cast<std::uint32_t>(x % range);
}

Result<void> Rng::fill_u8(std::span<std::uint8_t> out, std::uint8_t lo, std::uint8_t hi) {
  if (lo > hi) return std::unexpected(make_error(Errc::invalid_shape, "fill_u8 range"));
  const std::uint32_t n = static_cast<std::uint32_t>(hi) - lo + 1u;
  for (auto& x : out) x = static_cast<std::uint8_t>(lo + u32(n));
  return {};
}

Result<void> Rng::fill_f32(std::span<float> out, float lo, float hi) {
  for (auto& x : out) x = lo + (hi - lo) * uniform01();
  return {};
}

}  // namespace gyre
