#pragma once

#include "gyre/module.hpp"
#include "gyre/optim.hpp"
#include "gyre/rng.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace gyre {

struct CheckpointMeta {
  std::uint64_t rng_seed{0};
  std::uint64_t train_step{0};
  std::string json;  // vocab / arch
};

Result<void> save_gyre1(const std::filesystem::path& path, std::span<Param> params, const Adam* adam,
                        const CheckpointMeta& meta);

Result<void> load_gyre1(const std::filesystem::path& path, std::span<Param> params, Adam* adam,
                        CheckpointMeta& meta);

// Header + trailer JSON only (no tensor load). Used to recover BPE / arch before create().
Result<CheckpointMeta> peek_gyre1(const std::filesystem::path& path);

}  // namespace gyre
