#pragma once

#include "gyre/nn/layers.hpp"

#include <optional>
#include <vector>

namespace gyre {

// Causal multi-head attention with optional ALiBi (token-distance) bias.
class CausalSelfAttention final : public Module {
 public:
  static Result<CausalSelfAttention> create(std::int64_t d_model, std::int64_t n_head,
                                            std::shared_ptr<Device> d, Rng& rng, float resid_scale,
                                            bool recency_alibi);

  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return flat_; }

  CausalSelfAttention(CausalSelfAttention&&) noexcept = default;
  CausalSelfAttention& operator=(CausalSelfAttention&&) noexcept = default;

 private:
  CausalSelfAttention(Linear q, Linear k, Linear v, Linear o, std::int64_t n_head, bool alibi);
  void rebind();

  Linear q_, k_, v_, o_;
  std::int64_t n_head_{4};
  bool recency_alibi_{true};
  std::vector<Param> flat_;
  std::optional<Tensor> saved_x_;
  std::optional<Tensor> saved_q_, saved_k_, saved_v_;
  std::optional<Tensor> saved_qh_, saved_kh_, saved_vh_;
  std::optional<Tensor> saved_w_, saved_y3_;
};

}  // namespace gyre
