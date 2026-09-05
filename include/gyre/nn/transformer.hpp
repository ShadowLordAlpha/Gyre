#pragma once

#include "gyre/nn/attention.hpp"
#include "gyre/nn/tokenize.hpp"

#include <optional>
#include <string>
#include <vector>

namespace gyre {

struct CharLMConfig {
  std::int64_t vocab{65};
  std::int64_t block_size{64};
  std::int64_t n_layer{2};
  std::int64_t n_head{4};
  std::int64_t d_model{64};
  std::int64_t d_ff{256};
  bool recency_alibi{true};

  static CharLMConfig tiny() {
    CharLMConfig c;
    c.block_size = 64;
    c.n_layer = 2;
    c.n_head = 4;
    c.d_model = 64;
    c.d_ff = 256;
    return c;
  }

  static CharLMConfig medium() {
    CharLMConfig c;
    c.block_size = 128;
    c.n_layer = 4;
    c.n_head = 4;
    c.d_model = 128;
    c.d_ff = 512;
    return c;
  }

  static CharLMConfig tinygpt() {
    CharLMConfig c;
    c.vocab = 256;
    c.block_size = 256;
    c.n_layer = 6;
    c.n_head = 6;
    c.d_model = 192;
    c.d_ff = 768;
    return c;
  }

  static CharLMConfig nanogpt() {
    CharLMConfig c;
    c.block_size = 256;
    c.n_layer = 6;
    c.n_head = 6;
    c.d_model = 384;
    c.d_ff = 1536;
    return c;
  }
};

class DecoderBlock final : public Module {
 public:
  static Result<DecoderBlock> create(const CharLMConfig& c, std::shared_ptr<Device> d, Rng& rng,
                                     float resid_scale);
  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return flat_; }

  DecoderBlock(DecoderBlock&&) noexcept = default;
  DecoderBlock& operator=(DecoderBlock&&) noexcept = default;

 private:
  DecoderBlock(LayerNorm ln1, CausalSelfAttention attn, LayerNorm ln2, Linear fc1, Linear fc2);
  void rebind();

  LayerNorm ln1_;
  CausalSelfAttention attn_;
  LayerNorm ln2_;
  Linear fc1_, fc2_;
  std::vector<Param> flat_;
  std::optional<Tensor> saved_x_, saved_h1_, saved_fc1_;
};

class CharLM final : public Module {
 public:
  static Result<CharLM> create(CharLMConfig c, std::shared_ptr<Device> d, Rng& rng);
  Result<Tensor> forward(const Tensor& idx, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }
  // Hidden states after final LayerNorm: [B, T, d_model]. Skips lm_head.
  Result<Tensor> hidden(const Tensor& idx, ForwardCtx& ctx);
  // ∂L/∂hidden through the encoder. Pair with hidden() when CharLM is a trunk.
  Result<void> hidden_backward(const Tensor& d_hidden, ForwardCtx& ctx);
  Result<std::vector<std::int32_t>> generate(std::vector<std::int32_t> prefix, int max_new,
                                             std::shared_ptr<Device> d, Rng* rng = nullptr,
                                             float temperature = 0.f);

  CharLM(CharLM&&) noexcept = default;
  CharLM& operator=(CharLM&&) noexcept = default;

  std::int64_t vocab() const { return cfg_.vocab; }
  std::int64_t block_size() const { return cfg_.block_size; }
  const CharLMConfig& config() const { return cfg_; }
  std::vector<std::string> param_names() const;

 private:
  CharLM(CharLMConfig c, Embedding wte, Embedding wpe, std::vector<DecoderBlock> blocks, LayerNorm ln_f,
         Linear lm_head);
  void rebind();

  CharLMConfig cfg_{};
  Embedding wte_;
  Embedding wpe_;
  std::vector<DecoderBlock> blocks_;
  LayerNorm ln_f_;
  Linear lm_head_;
  std::vector<Param> params_;
  std::optional<Tensor> saved_idx_;
};

}  // namespace gyre
