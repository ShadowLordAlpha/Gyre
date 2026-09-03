#pragma once

#include "gyre/rng.hpp"
#include "gyre/tensor.hpp"

#include <cstdint>
#include <vector>

namespace gyre::adapt {

enum class ActionSpace : std::uint8_t { discrete_int = 1, continuous_f32 = 2 };

struct Action {
  ActionSpace space{ActionSpace::discrete_int};
  std::int32_t discrete{0};
  std::vector<float> continuous;
};

struct StepResult {
  Tensor observation;
  float reward{0};
  bool done{false};
};

class Environment {
 public:
  virtual Result<Tensor> reset(Rng&) = 0;
  virtual Result<StepResult> step(const Action&) = 0;
  virtual std::span<const std::int64_t> observation_shape() const = 0;
  virtual std::size_t action_dim() const = 0;
  virtual ActionSpace action_space() const = 0;
  virtual ~Environment() = default;
};

}  // namespace gyre::adapt
