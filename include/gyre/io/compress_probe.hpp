#pragma once

#include "gyre/io/safetensors.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gyre {

struct CompressProbeRow {
  std::string file;
  std::string name;
  std::string family;
  std::uint64_t raw_bytes{0};
  std::uint64_t probed_bytes{0};
  double entropy_bpb{0};           // Shannon bits/byte on raw
  double xor_entropy_bpb{0};       // after neighbor XOR
  std::uint64_t entropy_bytes{0};  // ceil(entropy_bpb * n / 8)
  std::uint64_t rle_bytes{0};
  std::uint64_t lzss_bytes{0};     // tiny LZSS stand-in (not gzip)
  std::uint64_t sign_ent_bytes{0};
  std::uint64_t exp_ent_bytes{0};
  std::uint64_t mant_ent_bytes{0};
  std::uint64_t plane_pack_bytes{0};  // per-bit-plane majority + exceptions or dense
  std::uint64_t freq16_exc_bytes{0};  // most common u16 + exceptions
  std::uint64_t estimate_bytes{0};    // min of Shannon/RLE/LZSS estimates (not a packed file)
  std::uint64_t stack_bytes{0};       // alias of estimate_bytes (legacy JSON)
  std::uint64_t packed_bytes{0};      // actual wpack_encode payload size
  std::string packed_codec;           // identity | plane_exc | exp_alpha
  int unique_exponents{0};
  double zero_frac{0};
  double top_u16_frac{0};
  std::string best_rule;
  double t_ms{0};
};

struct CompressProbeOpts {
  std::uint64_t chunk_bytes{1u << 20};
  int max_tensors{8};
  std::string file_substr;  // empty = all .safetensors
};

CompressProbeRow probe_bytes(std::span<const std::byte> raw, std::string_view name = {});
Result<std::vector<CompressProbeRow>> compress_probe_dir(const std::filesystem::path& dir,
                                                         const CompressProbeOpts& opts);
std::string compress_probe_json(std::span<const CompressProbeRow> rows);
std::string tensor_family(std::string_view name);

Result<std::vector<std::byte>> safetensors_read_prefix(const SafetensorFile& file,
                                                      const SafetensorInfo& info,
                                                      std::uint64_t max_bytes);

}  // namespace gyre
