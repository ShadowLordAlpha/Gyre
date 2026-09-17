#pragma once

#include "gyre/module.hpp"
#include "gyre/rng.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace gyre {

struct LoraPair {
  Tensor A;  // [in, rank]
  Tensor B;  // [rank, out]
};

class Linear final : public Module {
 public:
  static Result<Linear> create(std::int64_t in, std::int64_t out, std::shared_ptr<Device> dev,
                               Rng& rng, float std = 0.02f, float residual_scale = 1.f);

  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }
  std::span<Param> lora_parameters() noexcept { return lora_flat_; }
  std::span<const Param> lora_parameters() const noexcept { return lora_flat_; }

  Result<void> set_lora(Tensor A, Tensor B, float scale);
  void clear_lora();
  void set_freeze_base(bool freeze) { freeze_base_ = freeze; }
  bool freeze_base() const { return freeze_base_; }
  bool has_lora() const { return loraA_.has_value(); }

  Linear(Linear&&) noexcept = default;
  Linear& operator=(Linear&&) noexcept = default;

 private:
  Linear(std::vector<Param> p) : params_(std::move(p)) {}
  void rebind_lora();
  std::vector<Param> params_;  // W [in,out], b [out]
  std::optional<Param> loraA_;
  std::optional<Param> loraB_;
  std::vector<Param> lora_flat_;
  float lora_scale_{0.f};
  bool freeze_base_{false};
  std::optional<Tensor> saved_x_;
  std::optional<Tensor> saved_xa_;
};

class LayerNorm final : public Module {
 public:
  static Result<LayerNorm> create(std::int64_t n, std::shared_ptr<Device> dev, float eps = 1e-5f);
  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }

  LayerNorm(LayerNorm&&) noexcept = default;
  LayerNorm& operator=(LayerNorm&&) noexcept = default;

 private:
  LayerNorm(std::vector<Param> p, float eps) : params_(std::move(p)), eps_(eps) {}
  std::vector<Param> params_;
  float eps_{1e-5f};
  std::optional<Tensor> saved_x_;
};

class Embedding final : public Module {
 public:
  static Result<Embedding> create(std::int64_t vocab, std::int64_t d, std::shared_ptr<Device> dev,
                                  Rng& rng, float std = 0.02f);
  Result<Tensor> forward(const Tensor& idx, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }

  Embedding(Embedding&&) noexcept = default;
  Embedding& operator=(Embedding&&) noexcept = default;

 private:
  explicit Embedding(std::vector<Param> p) : params_(std::move(p)) {}
  std::vector<Param> params_;
  std::optional<Tensor> saved_idx_;
};

}  // namespace gyre
