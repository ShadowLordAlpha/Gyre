#pragma once

#include "gyre/ops.hpp"

namespace gyre {

// Repeat KV heads so [B, n_kv, T, D] becomes [B, n_q, T, D] (n_q % n_kv == 0).
Result<Tensor> repeat_kv_heads(const Tensor& kv, std::int64_t n_q);

// Causal GQA: q,k,v are [B, H, T, D] (k/v may have fewer heads). Scores are
// (q k^T) / sqrt(D) * attn_temp_scale, then softcap, causal mask, softmax, @ v.
Result<Tensor> gqa_causal(const Tensor& q, const Tensor& k, const Tensor& v, float attn_softcap,
                          float attn_temp_scale);

}  // namespace gyre
