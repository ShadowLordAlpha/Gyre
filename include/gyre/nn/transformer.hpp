#pragma once

#include "gyre/nn/attention.hpp"
#include "gyre/nn/tokenize.hpp"

#include <filesystem>
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
  float dropout{0.f};  // 0 = off; nanoGPT shakespeare-char uses 0.2

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

struct CharLoraLayer {
  LoraPair q, k, v, o, fc1, fc2;
};

struct CharLora {
  std::int64_t rank{8};
  float alpha{16.f};
  std::vector<CharLoraLayer> layers;
  float scale() const noexcept {
    return rank > 0 ? alpha / static_cast<float>(rank) : 0.f;
  }
  static Result<CharLora> create(const CharLMConfig& c, std::int64_t rank, float alpha,
                                 std::shared_ptr<Device> d, Rng& rng);
};

class DecoderBlock final : public Module {
 public:
  static Result<DecoderBlock> create(const CharLMConfig& c, std::shared_ptr<Device> d, Rng& rng,
                                     float resid_scale);
  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return flat_; }

  Result<void> set_lora(const CharLoraLayer& layer, float scale);
  void clear_lora();
  void set_freeze_base(bool freeze);
  std::vector<Param> lora_parameters();

  DecoderBlock(DecoderBlock&&) noexcept = default;
  DecoderBlock& operator=(DecoderBlock&&) noexcept = default;

 private:
  DecoderBlock(LayerNorm ln1, CausalSelfAttention attn, LayerNorm ln2, Linear fc1, Linear fc2,
               float dropout);
  void rebind();

  LayerNorm ln1_;
  CausalSelfAttention attn_;
  LayerNorm ln2_;
  Linear fc1_, fc2_;
  float dropout_{0.f};
  std::vector<Param> flat_;
  std::optional<Tensor> saved_x_, saved_h1_, saved_fc1_;
  std::optional<Tensor> saved_drop_mlp_;
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

  Result<void> set_lora(CharLora lora);
  Result<void> save_lora(const std::filesystem::path& path) const;
  Result<void> load_lora(const std::filesystem::path& path, std::shared_ptr<Device> d);
  void clear_lora();
  void set_freeze_base(bool freeze);
  const CharLora* lora() const { return lora_ ? &*lora_ : nullptr; }
  std::span<Param> lora_parameters() noexcept { return lora_flat_; }
  std::span<const Param> lora_parameters() const noexcept { return lora_flat_; }
  Result<void> zero_grad() override;

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
  std::optional<CharLora> lora_;
  std::vector<Param> lora_flat_;
  LayerNorm ln_f_;
  Linear lm_head_;
  std::vector<Param> params_;
  std::optional<Tensor> saved_idx_;
  std::optional<Tensor> saved_drop_emb_;
};

}  // namespace gyre
