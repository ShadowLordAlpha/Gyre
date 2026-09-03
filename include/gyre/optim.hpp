#pragma once

#include "gyre/module.hpp"

#include <vector>

namespace gyre {

struct Adam {
  float lr{3e-4f};
  float beta1{0.9f};
  float beta2{0.999f};
  float eps{1e-8f};
  std::uint64_t t{0};
  std::vector<Tensor> m;
  std::vector<Tensor> v;

  static Result<Adam> create(std::span<Param> params, float lr = 3e-4f);
  Result<void> step(std::span<Param> params);
};

}  // namespace gyre
