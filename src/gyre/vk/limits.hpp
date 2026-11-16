#pragma once

// Index limits of the Vulkan compute shaders, kept free of vulkan.h so tests can check them
// without a device.
//
// Every shader addresses its storage buffers as arrays of 4-byte words with 32-bit indices
// (`uint`; the embedding-backward segment tables and arange use `int`), and every count or
// extent travels as a 32-bit push constant. So before an op casts a size to u32 it must know
//   * each bound tensor's element count fits kMaxShaderElems (int32, the tightest index type), and
//   * each bound tensor's byte size fits the device's maxStorageBufferRange (a u32, <= 4 GiB-1);
// every other value an op pushes (rows, cols, M/K/N, batch, T, H, ...) is a factor of one of
// those element counts, so it fits too. Tensors past either limit are rejected with a clear
// error instead of being silently truncated.

#include "gyre/error.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace gyre::vkops {

inline constexpr std::int64_t kMaxShaderElems = std::numeric_limits<std::int32_t>::max();

// a * b for non-negative extents, saturating at INT64_MAX (so an overflowing product is
// still rejected by check_extent instead of wrapping to something small).
constexpr std::int64_t sat_mul(std::int64_t a, std::int64_t b) noexcept {
  if (a <= 0 || b <= 0) return 0;
  if (a > std::numeric_limits<std::int64_t>::max() / b) return std::numeric_limits<std::int64_t>::max();
  return a * b;
}

// numel elements of elem_bytes each, bound as one storage buffer on a device whose
// maxStorageBufferRange is max_range_bytes.
inline Result<void> check_extent(std::int64_t numel, std::size_t elem_bytes, std::uint64_t max_range_bytes,
                                 std::string_view op) {
  if (numel < 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "Vulkan " + std::string(op) + ": negative extent"));
  }
  if (numel > kMaxShaderElems) {
    return std::unexpected(make_error(
        Errc::unsupported, "Vulkan " + std::string(op) + ": tensor of " + std::to_string(numel) +
                               " elements exceeds the shader index limit of " + std::to_string(kMaxShaderElems) +
                               " (32-bit indices); lower --batch or use --device cpu"));
  }
  const auto bytes = static_cast<std::uint64_t>(numel) * static_cast<std::uint64_t>(elem_bytes);
  if (bytes > max_range_bytes) {
    return std::unexpected(make_error(
        Errc::unsupported, "Vulkan " + std::string(op) + ": tensor of " + std::to_string(bytes) +
                               " bytes exceeds maxStorageBufferRange " + std::to_string(max_range_bytes) +
                               "; lower --batch or use --device cpu"));
  }
  return {};
}

// Narrow a value that check_extent has already bounded (directly or as a factor of a checked
// element count). Out-of-range input is a programming error, reported rather than truncated.
inline Result<std::uint32_t> to_u32(std::int64_t v, std::string_view what) {
  if (v < 0 || v > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(make_error(Errc::overflow, "Vulkan " + std::string(what) + " = " + std::to_string(v) +
                                                          " does not fit a 32-bit shader constant"));
  }
  return static_cast<std::uint32_t>(v);
}

}  // namespace gyre::vkops
