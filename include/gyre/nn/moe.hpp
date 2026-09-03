#pragma once

#include "gyre/ops.hpp"

#include <span>

namespace gyre {

struct SwiGLUWeights {
  const Tensor* w1{nullptr};  // [D, I]
  const Tensor* w3{nullptr};  // [D, I]
  const Tensor* w2{nullptr};  // [I, D]
};

struct MoeTopK {
  Tensor indices;  // i32 [N, k]
  Tensor weights;  // f32 [N, k]  (softmax over experts, then top-k renormalized)
};

// logits [N, E] f32. Softcap then softmax over E, then top-k + renormalize.
Result<MoeTopK> moe_topk(const Tensor& logits, int k, float router_softcap);

// x [N, D] (or [B, T, D] flattened by rows). SwiGLU: silu(x@w1)*(x@w3) @ w2
Result<Tensor> swiglu_ffn(const Tensor& x, const SwiGLUWeights& w);

// residual_moe: dense SwiGLU(x) + sum_j weight_j * expert_{idx_j}(x)
Result<Tensor> residual_moe(const Tensor& x, const SwiGLUWeights& dense, const Tensor& router_w,
                            std::span<const SwiGLUWeights> experts, int k, float router_softcap);

}  // namespace gyre
