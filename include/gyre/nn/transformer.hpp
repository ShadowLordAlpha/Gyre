#pragma once

#include "gyre/nn/layers.hpp"
#include "gyre/nn/tokenize.hpp"

namespace gyre {

struct CharLMConfig {
  std::int64_t vocab{65};
  std::int64_t block_size{64};
  std::int64_t n_layer{2};
  std::int64_t n_head{4};
  std::int64_t d_model{64};
  std::int64_t d_ff{256};
  bool recency_alibi{true};  // token-distance bias on attention (not wall-clock)

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

  // TinyGPT-style: byte vocab, CPU-realistic (~3M weights), not a 10M gate.
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

  // nanoGPT shakespeare-char scale (~10.7M at V≈65, T=256, 6×384).
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

class CharLM final : public Module {
 public:
  static Result<CharLM> create(CharLMConfig c, std::shared_ptr<Device> d, Rng& rng);
  Result<Tensor> forward(const Tensor& idx, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }
  // temperature <= 0: greedy argmax. temperature > 0: sample softmax(logits/T); rng required.
  Result<std::vector<std::int32_t>> generate(std::vector<std::int32_t> prefix, int max_new,
                                             std::shared_ptr<Device> d, Rng* rng = nullptr,
                                             float temperature = 0.f);

  CharLM(CharLM&&) noexcept = default;
  CharLM& operator=(CharLM&&) noexcept = default;

  std::int64_t vocab() const { return cfg_.vocab; }
  std::int64_t block_size() const { return cfg_.block_size; }
  const CharLMConfig& config() const { return cfg_; }

 private:
  explicit CharLM(CharLMConfig c) : cfg_(c) {}
  CharLMConfig cfg_{};
  std::vector<Param> params_;  // wte, wpe, then per layer, then ln_f, lm_head
  std::int64_t n_head_{4};
};

}  // namespace gyre
