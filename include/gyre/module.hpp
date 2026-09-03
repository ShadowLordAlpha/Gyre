#pragma once

#include "gyre/ops.hpp"
#include "gyre/tensor.hpp"

#include <memory>
#include <span>
#include <vector>

namespace gyre {

struct ForwardCtx {
  std::vector<Tensor> saved;
  bool train = true;
  std::unique_ptr<Tensor> dx;  // ∂L/∂input, filled by backward()
};

struct Param {
  Tensor value;
  Tensor grad;
};

Result<Param> make_param(Tensor value);

class Module {
 public:
  virtual Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) = 0;
  virtual Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) = 0;
  virtual std::span<Param> parameters() noexcept = 0;
  virtual Result<void> zero_grad();
  virtual ~Module() = default;
};

struct LossPair {
  Tensor value;
  Tensor d_pred;
};

Result<LossPair> softmax_cross_entropy(const Tensor& logits, const Tensor& targets_i32);

}  // namespace gyre
