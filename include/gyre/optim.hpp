#pragma once

#include "gyre/module.hpp"

#include <vector>

namespace gyre {

struct Adam {
  float lr{3e-4f};
  float beta1{0.9f};
  float beta2{0.999f};
  float eps{1e-8f};
  float weight_decay{0.f};  // AdamW on rank>=2 tensors; 0 = off
  std::uint64_t t{0};
  std::vector<Tensor> m;
  std::vector<Tensor> v;

  static Result<Adam> create(std::span<Param> params, float lr = 3e-4f);
  Result<void> step(std::span<Param> params);
};

// Scale grads so the global L2 norm is at most max_norm. max_norm<=0 is a no-op.
// Errors if any grad is non-finite (so the caller can skip the Adam step).
Result<float> clip_grad_norm(std::span<Param> params, float max_norm);

}  // namespace gyre
