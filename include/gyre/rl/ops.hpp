#pragma once
// Generic CPU f32 ops for policy/value networks. No game concepts: inputs are float
// tensors, u8 masks (1 = valid/legal) and i32 indices. Each op that has a gradient
// ships its backward next to it; all are finite-difference checked in tests/.
#include "gyre/tensor.hpp"

#include <span>

namespace gyre {

constexpr float kMaskFill = -1e9f;  // finite, so exp() underflows to exactly 0 and never NaNs

// Elementwise activations. *_backward take the forward OUTPUT (sigmoid/tanh) or INPUT (relu).
Result<Tensor> sigmoid(const Tensor& x);
Result<Tensor> sigmoid_backward(const Tensor& y, const Tensor& dy);
Result<Tensor> tanh_act(const Tensor& x);
Result<Tensor> tanh_backward(const Tensor& y, const Tensor& dy);
Result<Tensor> relu(const Tensor& x);
Result<Tensor> relu_backward(const Tensor& x, const Tensor& dy);
Result<Tensor> sub(const Tensor& a, const Tensor& b);
// clamp to [lo, hi]; gradient passes where lo <= x <= hi.
Result<Tensor> clamp(const Tensor& x, float lo, float hi);
Result<Tensor> clamp_backward(const Tensor& x, const Tensor& dy, float lo, float hi);

// Row-wise over the last dim. backward takes the forward output (log-probs).
Result<Tensor> log_softmax_last(const Tensor& x);
Result<Tensor> log_softmax_last_backward(const Tensor& logp, const Tensor& dy);

// out[...] = x[..., idx[...]]  (x [..., V], idx i32 [...]). Out-of-range index is an error.
Result<Tensor> gather_last(const Tensor& x, const Tensor& idx_i32);
Result<Tensor> gather_last_backward(const Tensor& dy, const Tensor& idx_i32, std::int64_t V);

// Legality masking: logits where mask==0 become kMaskFill. mask u8, same shape as logits.
Result<Tensor> mask_logits(const Tensor& logits, const Tensor& mask_u8);
Result<Tensor> mask_logits_backward(const Tensor& dy, const Tensor& mask_u8);

// In-place key padding mask on attention scores [B,H,T,T] with mask u8 [B,T] (keys).
Result<void> key_mask_(Tensor& scores, const Tensor& mask_u8);

// Masked pooling over the entity axis. h [B,N,D], mask u8 [B,N] -> [B, 2D] = [mean | max].
// Empty sets pool to zeros. `argmax_out` (i32 [B,D], -1 = none) is needed by the backward.
Result<Tensor> masked_mean_max_pool(const Tensor& h, const Tensor& mask_u8, Tensor& argmax_out);
Result<Tensor> masked_mean_max_pool_backward(const Tensor& dy, const Tensor& mask_u8,
                                             const Tensor& argmax, std::int64_t N);

// Concatenate along the last dim (rank-2 inputs [B,Ca],[B,Cb]) and split the gradient back.
Result<Tensor> concat_last(const Tensor& a, const Tensor& b);
Result<void> split_last(const Tensor& dy, std::int64_t ca, Tensor& da, Tensor& db);

// Row masking for recurrent state: out[b,:] = keep[b] ? h[b,:] : 0. h [B,H], keep u8 [B].
// Its own backward (apply to the incoming gradient).
Result<Tensor> mask_rows(const Tensor& h, const Tensor& keep_u8);
// Stack rank-2 tensors with equal width along rows -> [sum rows, C].
Result<Tensor> concat_rows(std::span<const Tensor> parts);

// ---- fused attention (non-causal, optional key padding mask), all [B,H,N,dk] row-major ----
// w[b,h,i,j] = softmax_j( scale * q_i . k_j ), keys with mask==0 get weight exactly 0 (an all-masked
// row degrades to uniform, like softmax over equal fill values). Replaces
// transpose + bmm + mul_scalar + key_mask_ + softmax, which materialised four [B,H,N,N] tensors.
// key_mask: u8 [B,N] or nullptr.
Result<Tensor> attn_probs(const Tensor& qh, const Tensor& kh, const Tensor* key_mask_u8, float scale);
// Gradient w.r.t. the pre-softmax *dot products* of q.k scaled by `scale`, i.e.
//   dz[i,j] = scale * w[i,j] * (dw[i,j] - sum_j' w[i,j'] dw[i,j']),  dw[i,j] = dy_i . v_j,
// so that dq = bmm(dz, k) and dk = bmm_tn(dz, q). dy and v are [B,H,N,dk]; w comes from attn_probs.
Result<Tensor> attn_probs_backward(const Tensor& w, const Tensor& dyh, const Tensor& vh, float scale);

}  // namespace gyre
