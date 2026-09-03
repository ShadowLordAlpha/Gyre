#pragma once

#include "gyre/io/safetensors.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace gyre {

enum class WpackCodec : std::uint8_t {
  identity = 0,
  plane_exc = 1,
  exp_alpha = 2,
  lfsr_pred = 3,
  const_lane = 4
};

inline const char* wpack_codec_name(WpackCodec c) {
  switch (c) {
    case WpackCodec::identity: return "identity";
    case WpackCodec::plane_exc: return "plane_exc";
    case WpackCodec::exp_alpha: return "exp_alpha";
    case WpackCodec::lfsr_pred: return "lfsr_pred";
    case WpackCodec::const_lane: return "const_lane";
  }
  return "unknown";
}

struct WpackTensor {
  std::string name;
  DType dtype{DType::bf16};
  std::vector<std::int64_t> shape;
  WpackCodec codec{WpackCodec::identity};
  std::vector<std::byte> payload;  // encoded
  std::uint64_t raw_bytes{0};
};

struct WpackFile {
  std::string source;  // original shard filename
  std::vector<WpackTensor> tensors;
};

// Lossless encode of a bf16/u16-sized buffer. Keeps a codec only if packed+16 < raw.
Result<WpackTensor> wpack_encode(std::string name, DType dtype, std::span<const std::int64_t> shape,
                                 std::span<const std::byte> raw);
Result<std::vector<std::byte>> wpack_decode(const WpackTensor& t);

Result<void> wpack_save(const std::filesystem::path& path, const WpackFile& file);
Result<WpackFile> wpack_load(const std::filesystem::path& path);

// Pack tensors from a safetensors shard (small tensors only; skip if raw > max_bytes).
Result<WpackFile> wpack_from_safetensors(const SafetensorFile& st, std::uint64_t max_bytes);

}  // namespace gyre
