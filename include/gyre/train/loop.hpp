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
};

struct TrainConfig {
  std::uint32_t steps{100};
  std::uint32_t batch{16};
  std::uint32_t block{64};
  float lr{3e-4f};          // floor / "normal" rate
  float lr_start{0.f};      // 0 = constant lr. else linear decay lr_start -> lr
  std::uint32_t lr_decay_steps{0};  // 0 = 1/5 of steps when lr_start > lr
  std::uint32_t log_every{10};
  std::uint32_t ckpt_every{0};
  std::filesystem::path ckpt_dir{"."};
  std::filesystem::path ckpt_path;  // if set, also overwrite this file each checkpoint
  std::uint64_t seed{1};
  std::string ckpt_json;
  std::vector<std::string> param_names;
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

class TrainLoop {
 public:
  Result<void> run(Module& model, Dataset& data, const TrainConfig& cfg, std::shared_ptr<Device> device,
                   std::function<void(const Metrics&)> on_log,
                   std::function<void(const Metrics&)> on_progress = {});
};

}  // namespace gyre
