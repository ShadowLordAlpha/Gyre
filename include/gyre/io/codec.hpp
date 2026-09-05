#pragma once

#include "gyre/tensor.hpp"

#include <span>
#include <string>
#include <vector>

namespace gyre {

enum class GyreCodec : std::uint8_t {
  identity = 0,
  alp = 1,  // Adaptive lossless float (SIGMOD 2024-style pseudo-decimal + exceptions)
  zfp = 2   // ZFP reversible (lossless) when built with GYRE_ZFP
};

inline const char* gyre_codec_name(GyreCodec c) {
  switch (c) {
    case GyreCodec::identity: return "identity";
    case GyreCodec::alp: return "alp";
    case GyreCodec::zfp: return "zfp";
  }
  return "unknown";
}

inline GyreCodec gyre_codec_from_name(std::string_view s) {
  if (s == "alp") return GyreCodec::alp;
  if (s == "zfp") return GyreCodec::zfp;
  return GyreCodec::identity;
}

bool gyre_codec_available(GyreCodec c);

// Lossless encode/decode of a contiguous tensor payload.
Result<std::vector<std::byte>> gyre_compress(GyreCodec codec, std::span<const std::byte> raw, DType dt,
                                            std::span<const std::int64_t> shape);
Result<std::vector<std::byte>> gyre_decompress(GyreCodec codec, std::span<const std::byte> packed, DType dt,
                                              std::span<const std::int64_t> shape, std::size_t raw_bytes);

struct GyreCodecProbe {
  std::string name;
  std::uint64_t raw_bytes{0};
  std::uint64_t alp_bytes{0};
  std::uint64_t zfp_bytes{0};  // 0 = unavailable or failed
  bool alp_ok{false};
  bool zfp_ok{false};
};

// Compress each f32 tensor with ALP and ZFP (if built); compare packed sizes.
Result<std::vector<GyreCodecProbe>> probe_gyre_codecs(const class GyreFile& file, int max_tensors = 32);

}  // namespace gyre
