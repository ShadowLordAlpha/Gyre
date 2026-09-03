#pragma once

#include "gyre/module.hpp"
#include "gyre/rng.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gyre {

// Grok-2 decoder (residual MoE + GQA + RoPE). Tiny presets allocate in RAM.
// full() is metadata for later safetensor bind — create() refuses huge allocs.

struct GrokConfig {
  std::int64_t vocab{256};
  std::int64_t block_size{16};
  std::int64_t n_layer{2};
  std::int64_t n_q_head{4};
  std::int64_t n_kv_head{1};
  std::int64_t d_model{32};
  std::int64_t head_dim{8};
  std::int64_t d_ff{64};       // dense SwiGLU inner
  std::int64_t moe_ff{32};     // per-expert inner
  std::int64_t n_experts{2};
  std::int64_t n_experts_per_tok{2};
  float rms_eps{1e-5f};
  float rope_theta{10000.f};
  float rope_scale{1.f};
  float embedding_scale{1.f};
  float output_scale{1.f};
  float attn_softcap{30.f};
  float router_softcap{30.f};
  float final_softcap{50.f};
  std::int64_t attn_temperature_len{1024};
  bool residual_moe{true};

  static GrokConfig tiny();
  static GrokConfig mini();  // 1-layer, even smaller than tiny — pack/generate sandbox
  static GrokConfig full();  // Hub xai-org/grok-2 numbers; not for create()
  static Result<GrokConfig> from_json(std::string_view json);
  static Result<GrokConfig> from_file(const std::filesystem::path& path);

  std::int64_t param_count() const;
  bool fits_in_ram_create() const;
  std::string to_string() const;
  std::string to_json() const;
};

struct GrokLoraPair {
  Tensor A;  // [in, r]
  Tensor B;  // [r, out]
};

struct GrokLora {
  std::int64_t rank{4};
  float alpha{4.f};
  std::vector<GrokLoraPair> q, k, v, o;  // one per layer
  float scale() const noexcept {
    return rank > 0 ? alpha / static_cast<float>(rank) : 0.f;
  }
  static Result<GrokLora> create(const GrokConfig& c, std::int64_t rank, float alpha,
                                 std::shared_ptr<Device> d, Rng& rng);
};

class GrokLM final : public Module {
 public:
  static Result<GrokLM> create(GrokConfig c, std::shared_ptr<Device> d, Rng& rng);
  Result<Tensor> forward(const Tensor& idx, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }

  Result<std::vector<std::int32_t>> generate(std::vector<std::int32_t> prefix, int max_new,
                                             std::shared_ptr<Device> d, Rng* rng = nullptr,
                                             float temperature = 0.f);

  Result<void> save_weights(const std::filesystem::path& dir) const;
  static Result<GrokLM> load_weights(const std::filesystem::path& dir, std::shared_ptr<Device> d);
  Result<void> set_lora(GrokLora lora);
  Result<void> save_lora(const std::filesystem::path& dir) const;
  Result<void> load_lora(const std::filesystem::path& dir, std::shared_ptr<Device> d);
  const GrokLora* lora() const { return lora_ ? &*lora_ : nullptr; }

  GrokLM(GrokLM&&) noexcept = default;
  GrokLM& operator=(GrokLM&&) noexcept = default;

  const GrokConfig& config() const { return cfg_; }
  std::vector<std::string> param_names() const;

 private:
  explicit GrokLM(GrokConfig c) : cfg_(c) {}
  Result<Tensor> linear_lora(const Tensor& x, const Tensor& W, const GrokLoraPair* slot) const;
  GrokConfig cfg_{};
  std::vector<Param> params_;
  std::optional<GrokLora> lora_;
};

}  // namespace gyre
