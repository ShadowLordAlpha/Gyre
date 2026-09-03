#pragma once

#include "gyre/module.hpp"
#include "gyre/rng.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gyre {

class Linear final : public Module {
 public:
  static Result<Linear> create(std::int64_t in, std::int64_t out, std::shared_ptr<Device> dev,
                               Rng& rng, float std = 0.02f, float residual_scale = 1.f);

  Result<Tensor> forward(const Tensor& x, ForwardCtx& ctx) override;
  Result<void> backward(const Tensor& d_out, ForwardCtx& ctx) override;
  std::span<Param> parameters() noexcept override { return params_; }

  Linear(Linear&&) noexcept = default;
  Linear& operator=(Linear&&) noexcept = default;

 private:
  Linear(std::vector<Param> p) : params_(std::move(p)) {}
  std::vector<Param> params_;  // W [in,out], b [out]
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
};

}  // namespace gyre
