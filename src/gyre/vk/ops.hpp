#pragma once

#include "gyre/module.hpp"

namespace gyre::vkops {

Result<Tensor> add(const Tensor& a, const Tensor& b);
Result<Tensor> mul(const Tensor& a, const Tensor& b);
Result<Tensor> matmul(const Tensor& a, const Tensor& b);
Result<Tensor> bmm(const Tensor& a, const Tensor& b);
Result<Tensor> sum(const Tensor& a);
Result<Tensor> sum_dim(const Tensor& a, int axis, bool keepdim);
Result<Tensor> transpose_last2(const Tensor& a);
Result<Tensor> embedding(const Tensor& weight, const Tensor& indices_i32);
Result<Tensor> permute_bthd_bhtd(const Tensor& x);
Result<Tensor> permute_bhtd_bthd(const Tensor& x);
Result<Tensor> narrow_rows(const Tensor& x, std::int64_t start, std::int64_t count);
Result<Tensor> gelu(const Tensor& a);
Result<Tensor> silu(const Tensor& a);
Result<Tensor> softmax_last(const Tensor& a);
Result<Tensor> layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps);
Result<Tensor> rms_norm(const Tensor& x, const Tensor& w, float eps);
Result<Tensor> softcap(const Tensor& a, float cap);
Result<void> fill_zero(Tensor& t);
Result<void> fill(Tensor& t, float value);
Result<void> add_(Tensor& dst, const Tensor& src);
Result<Tensor> gelu_backward(const Tensor& x, const Tensor& dy);
Result<Tensor> softmax_last_backward(const Tensor& softmax, const Tensor& d_out);
Result<Tensor> mul_scalar(const Tensor& a, float s);
Result<void> causal_alibi_(Tensor& scores, bool alibi);
Result<Tensor> add_broadcast_time(const Tensor& tok, const Tensor& pe);
Result<Tensor> sum_batch_to_time(const Tensor& dh);
Result<Tensor> arange_i32(std::int64_t n, std::shared_ptr<Device> d);
Result<Tensor> layer_norm_backward(const Tensor& x, const Tensor& d_out, const Tensor& w, Tensor& gw,
                                   Tensor& gb, float eps);
Result<void> embedding_backward(Tensor& grad_W, const Tensor& idx, const Tensor& d_out);
Result<void> bias_add_(Tensor& y, const Tensor& b);
Result<LossPair> softmax_cross_entropy(const Tensor& logits, const Tensor& targets_i32);
Result<void> adam_step(Tensor& w, const Tensor& g, Tensor& m, Tensor& v, float lr, float beta1,
                       float beta2, float eps, float b1t, float b2t);

}  // namespace gyre::vkops
