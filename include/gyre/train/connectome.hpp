#pragma once

#include "gyre/nn/transformer.hpp"
#include "gyre/rng.hpp"

#include <cstdint>

namespace gyre {

struct ConnectomeStats {
  std::uint32_t generation{0};
  float prune{0};
  std::int64_t d_model{0};
  std::int64_t d_ff{0};
  std::uint64_t n_params{0};
  std::uint64_t n_params_parent{0};
};

std::uint64_t param_count(const Module& m);

// Drop the weakest `prune` fraction of residual channels and of each block's
// FFN width, copy survivors into a new dense CharLM (no masked zeros), then
// collapse remaining 2D weights to sign * mean(|w|).
Result<CharLM> prune_compact_charlm(CharLM& src, float prune, bool shuffle, Rng& rng,
                                    std::shared_ptr<Device> device);

}  // namespace gyre
