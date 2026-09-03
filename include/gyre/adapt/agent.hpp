#pragma once

#include "gyre/adapt/environment.hpp"
#include "gyre/nn/layers.hpp"

#include <array>
#include <filesystem>

namespace gyre::adapt {

class Agent {
 public:
  virtual Result<Action> act(const Tensor& observation, Rng&) = 0;
  virtual Result<void> observe(const StepResult&) = 0;
  virtual Result<void> load(const std::filesystem::path&) = 0;
  virtual Result<void> save(const std::filesystem::path&) const = 0;
  virtual ~Agent() = default;
};

// Deterministic discrete agent: argmax of a Linear policy on CPU f32 obs.
class PolicyAgent final : public Agent {
 public:
  static Result<PolicyAgent> create(std::int64_t obs_dim, std::int64_t n_actions,
                                    std::shared_ptr<Device> d, Rng& rng);
  Result<Action> act(const Tensor& observation, Rng&) override;
  Result<void> observe(const StepResult&) override { return {}; }
  Result<void> load(const std::filesystem::path& p) override;
  Result<void> save(const std::filesystem::path& p) const override;

  PolicyAgent(PolicyAgent&&) noexcept = default;
  PolicyAgent& operator=(PolicyAgent&&) noexcept = default;

 private:
  PolicyAgent(Linear lin) : lin_(std::move(lin)) {}
  Linear lin_;
};

class GridWorld final : public Environment {
 public:
  GridWorld();
  Result<Tensor> reset(Rng&) override;
  Result<StepResult> step(const Action&) override;
  std::span<const std::int64_t> observation_shape() const override;
  std::size_t action_dim() const override { return 4; }
  ActionSpace action_space() const override { return ActionSpace::discrete_int; }

 private:
  std::int64_t x_{0}, y_{0};
  std::array<std::int64_t, 1> shape_{2};
  std::shared_ptr<Device> cpu_;
};

}  // namespace gyre::adapt
