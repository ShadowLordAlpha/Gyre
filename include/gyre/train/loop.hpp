#pragma once

#include "gyre/checkpoint.hpp"
#include "gyre/data.hpp"
#include "gyre/module.hpp"
#include "gyre/optim.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace gyre {

struct Metrics {
  std::uint64_t step{0};
  float loss{0};
  float lr{0};
  float gnorm{0};  // pre-clip L2; 0 if clipping is off
};

struct TrainConfig {
  std::uint32_t steps{100};
  std::uint32_t batch{16};
  std::uint32_t block{64};
  float lr{3e-4f};          // floor / "normal" rate
  float lr_start{0.f};      // 0 = constant lr. else linear decay lr_start -> lr
  float grad_clip{0.f};     // global L2 clip; 0 = off. 1.0 is too tight for Gyre's batch.
  float weight_decay{0.f};  // AdamW; 0 = off
  std::uint32_t lr_decay_steps{0};  // 0 = 1/5 of steps when lr_start > lr
  std::uint32_t log_every{10};
  std::uint32_t ckpt_every{0};
  std::filesystem::path ckpt_dir{"."};
  std::filesystem::path ckpt_path;  // if set, also overwrite this file each checkpoint
  std::uint64_t seed{1};
  std::string ckpt_json;
  std::vector<std::string> param_names;
  std::uint32_t start_step{0};  // run start_step+1 .. start_step+steps
  Adam* adam{nullptr};          // if null, the loop creates one
  bool save_adam{true};
};

inline float scheduled_lr(const TrainConfig& cfg, std::uint32_t step) {
  const float hi = cfg.lr_start > cfg.lr ? cfg.lr_start : cfg.lr;
  auto n = cfg.lr_decay_steps;
  if (n == 0 && cfg.lr_start > cfg.lr) n = cfg.steps / 5;
  if (n == 0 || hi <= cfg.lr) return cfg.lr;
  if (step >= n) return cfg.lr;
  const float t = static_cast<float>(step - 1) / static_cast<float>(n);
  return hi + (cfg.lr - hi) * t;
}

// Seed of the batch/dropout Rng for a run that starts after `start_step`.
// start_step 0 returns `seed` unchanged; otherwise splitmix64 mixes the step in,
// so a resumed run (or a later --prune-gens generation) draws fresh batches
// instead of replaying steps 1..k of the original run.
inline std::uint64_t train_stream_seed(std::uint64_t seed, std::uint32_t start_step) noexcept {
  if (start_step == 0) return seed;
  std::uint64_t z = seed + 0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(start_step) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

class TrainLoop {
 public:
  Result<void> run(Module& model, Dataset& data, const TrainConfig& cfg, std::shared_ptr<Device> device,
                   std::function<void(const Metrics&)> on_log,
                   std::function<void(const Metrics&)> on_progress = {});
};

}  // namespace gyre
