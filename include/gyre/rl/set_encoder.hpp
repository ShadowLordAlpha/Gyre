#pragma once
// Permutation-invariant encoder over a padded set of entity feature vectors.
//
//   entities [B,N,F] + mask u8 [B,N] (1 = real entity) + global [B,G]
//     -> Linear(F,d) -> L x (pre-LN non-causal self-attention + MLP, keys masked)
//     -> LayerNorm -> masked mean|max pool -> concat(global) -> Linear -> GELU
//     -> [B, out_dim]
//
// No positional encoding: entities are a set. Padded slots never influence the output or
// receive gradient. Which entities occupy slots is the caller's decision; Gyre only sees
// floats and a mask. N is a property of the inputs (config value `max_entities` is only
// validated against them).
#include "gyre/nn/layers.hpp"

#include <vector>

namespace gyre {

struct SetEncoderConfig {
  std::int64_t feat_dim{16};
  std::int64_t global_dim{0};
  std::int64_t max_entities{128};
  std::int64_t d_model{64};
  std::int64_t n_layer{2};
  std::int64_t n_head{4};
  std::int64_t d_ff{128};
  std::int64_t out_dim{128};
};

struct SetEncoderGrads {
  Tensor d_entities;  // [B,N,F]; exactly zero in padded slots
  Tensor d_global;    // [B,G]
};

class SetBlock {
 public:
  static Result<SetBlock> create(const SetEncoderConfig& c, std::shared_ptr<Device> d, Rng& rng);
  Result<Tensor> forward(const Tensor& x, const Tensor& mask, ForwardCtx& ctx);
  Result<Tensor> backward(const Tensor& dy, ForwardCtx& ctx);  // returns dx
  std::vector<Param>& parameters() noexcept { return flat_; }

  SetBlock(SetBlock&&) noexcept = default;
  SetBlock& operator=(SetBlock&&) noexcept = default;

 private:
  SetBlock(LayerNorm ln1, Linear q, Linear k, Linear v, Linear o, LayerNorm ln2, Linear fc1,
           Linear fc2, std::int64_t n_head);
  void rebind();
  LayerNorm ln1_;
  Linear q_, k_, v_, o_;
  LayerNorm ln2_;
  Linear fc1_, fc2_;
  std::int64_t n_head_;
  std::vector<Param> flat_;
  std::optional<Tensor> saved_qh_, saved_kh_, saved_vh_, saved_w_, saved_fc1_;
};

class SetEncoder {
 public:
  static Result<SetEncoder> create(const SetEncoderConfig& c, std::shared_ptr<Device> d, Rng& rng);

  Result<Tensor> forward(const Tensor& entities, const Tensor& mask_u8, const Tensor& global,
                         ForwardCtx& ctx);
  // Accumulates parameter grads. Needs the preceding forward (ctx.train = true).
  Result<SetEncoderGrads> backward(const Tensor& d_out, ForwardCtx& ctx);

  std::span<Param> parameters() noexcept { return flat_; }
  Result<void> zero_grad();
  const SetEncoderConfig& config() const { return cfg_; }

  SetEncoder(SetEncoder&&) noexcept = default;
  SetEncoder& operator=(SetEncoder&&) noexcept = default;

 private:
  SetEncoder(SetEncoderConfig c, Linear embed, std::vector<SetBlock> blocks, LayerNorm ln_f,
             Linear proj);
  void rebind();
  SetEncoderConfig cfg_;
  Linear embed_;
  std::vector<SetBlock> blocks_;
  LayerNorm ln_f_;
  Linear proj_;
  std::vector<Param> flat_;
  std::optional<Tensor> saved_mask_, saved_argmax_, saved_proj_pre_;
  std::int64_t saved_N_{0};
};

}  // namespace gyre
