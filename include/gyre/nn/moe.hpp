#pragma once

#include "gyre/ops.hpp"

#include <span>

namespace gyre {

struct SwiGLUWeights {
  const Tensor* w1{nullptr};  // [D, I]  gate
  const Tensor* w3{nullptr};  // [D, I]  up
  const Tensor* w2{nullptr};  // [I, D]  down
};

// Activation on the gate branch of a gated FFN: act(x@w1) * (x@w3) @ w2.
enum class GateAct {
  silu,       // SwiGLU (Llama, Mixtral)
  gelu_tanh,  // GeGLU with the tanh GELU (Grok-1 jax.nn.gelu, Grok-2 GeluAndMul(approximate="tanh"))
};

// How a (residual) MoE block combines its parts. The defaults are the generic SwiGLU MoE
// (Mixtral-style renormalized top-k); kGrokMoe is the Grok-1/Grok-2 reference behaviour.
struct MoeSpec {
  GateAct act{GateAct::silu};
  // true: divide the k selected probabilities by their sum (Mixtral). false: use the softmax
  // probabilities over all experts as-is (Grok: top_k(softmax(logits)), renormalize=False).
  bool renormalize{true};
  // true and a dense branch present: (dense + experts) / sqrt(2) (Grok-2 residual MoE,
  // SGLang moe_with_rmoe). Ignored for a plain sparse MoE.
  bool scale_residual{false};
};
inline constexpr MoeSpec kGrokMoe{GateAct::gelu_tanh, false, true};

struct MoeTopK {
  Tensor indices;  // i32 [N, k]
  Tensor weights;  // f32 [N, k]  softmax over all experts, top-k (renormalized if asked)
};

// logits [N, E] f32. Softcap then softmax over E, then top-k (+ renormalize).
Result<MoeTopK> moe_topk(const Tensor& logits, int k, float router_softcap, bool renormalize = true);

// x [N, D] (or [B, T, D] flattened by rows). act(x@w1)*(x@w3) @ w2; silu = SwiGLU.
Result<Tensor> swiglu_ffn(const Tensor& x, const SwiGLUWeights& w, GateAct act = GateAct::silu);

// residual_moe: dense FFN(x) + sum_j weight_j * expert_{idx_j}(x), divided by sqrt(2) when
// spec.scale_residual. A default (all-null) `dense` drops the dense branch: plain sparse MoE,
// experts only (Grok residual_moe = false / Grok-1). Dense and expert outputs must be [N, D];
// a narrower w2 is an invalid_shape error.
Result<Tensor> residual_moe(const Tensor& x, const SwiGLUWeights& dense, const Tensor& router_w,
                            std::span<const SwiGLUWeights> experts, int k, float router_softcap,
                            const MoeSpec& spec = {});

}  // namespace gyre
