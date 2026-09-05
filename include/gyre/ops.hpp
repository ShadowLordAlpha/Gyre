#pragma once

#include "gyre/rng.hpp"
#include "gyre/tensor.hpp"

namespace gyre {

Result<Tensor> add(const Tensor& a, const Tensor& b);
Result<Tensor> mul(const Tensor& a, const Tensor& b);
Result<Tensor> matmul(const Tensor& a, const Tensor& b);
Result<Tensor> bmm(const Tensor& a, const Tensor& b);
Result<Tensor> sum(const Tensor& a);
Result<Tensor> sum_dim(const Tensor& a, int axis, bool keepdim);
Result<Tensor> reshape(const Tensor& a, std::span<const std::int64_t> new_shape);
Result<Tensor> transpose_last2(const Tensor& a);
Result<Tensor> embedding(const Tensor& weight, const Tensor& indices_i32);
Result<Tensor> permute_bthd_bhtd(const Tensor& x);  // [B,T,H,d] -> [B,H,T,d]
Result<Tensor> permute_bhtd_bthd(const Tensor& x);  // [B,H,T,d] -> [B,T,H,d]
Result<Tensor> narrow_rows(const Tensor& x, std::int64_t start, std::int64_t count);

Result<Tensor> gelu(const Tensor& a);
Result<Tensor> silu(const Tensor& a);
Result<Tensor> swiglu(const Tensor& gate, const Tensor& up);  // silu(gate) * up
Result<Tensor> softmax_last(const Tensor& a);
Result<Tensor> layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps);
Result<Tensor> rms_norm(const Tensor& x, const Tensor& w, float eps);
Result<Tensor> softcap(const Tensor& a, float cap);  // cap * tanh(x / cap)
// Rotary on last dim (even). `x` is [..., T, D]; `positions` i32 [T] (or [1] for decode).
// `pos_scale` > 0 divides positions (linear RoPE interpolation; Grok-2 uses 16).
Result<Tensor> rope(const Tensor& x, const Tensor& positions_i32, float theta, float pos_scale = 1.f);
// Grok attn temperature: log(max(seq, L)) / log(L); 1 when seq <= L.
float attn_temperature_scale(std::int64_t seq_len, std::int64_t temp_len) noexcept;

Result<void> fill_zero(Tensor& t);
Result<void> add_(Tensor& dst, const Tensor& src);

// Rank-3 [B,T,C] or rank-1 [C] → rank-2 rows×C sharing storage.
Result<Tensor> flatten_leading(const Tensor& x);
Result<Tensor> unflatten_like(Tensor y, const Tensor& like);
// y = x @ W + b; x rank 1–3. W [in,out], b [out].
Result<Tensor> linear(const Tensor& x, const Tensor& W, const Tensor& b);
Result<Tensor> gelu_backward(const Tensor& x, const Tensor& dy);
Result<Tensor> softmax_last_backward(const Tensor& softmax, const Tensor& d_out);

// temperature <= 0: argmax. temperature > 0 requires rng.
Result<std::int32_t> sample_logit_row(std::span<const float> logits, float temperature, Rng* rng);

}  // namespace gyre
