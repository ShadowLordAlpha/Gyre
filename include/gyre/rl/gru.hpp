#pragma once
// GRU cell (PyTorch gate convention) with an explicit per-step tape for backprop through time.
//   r = sig(Wir x + bir + Whr h + bhr)      z = sig(Wiz x + biz + Whz h + bhz)
//   n = tanh(Win x + bin + r * (Whn h + bhn))
//   h' = (1 - z) * n + z * h
// Weights are Wi [In,3H], bi [3H], Wh [H,3H], bh [3H]; gate order r, z, n.
// Masking the carried state on episode boundaries is the caller's job (see mask_rows).
#include "gyre/module.hpp"
#include "gyre/rng.hpp"

#include <vector>

namespace gyre {

struct GruCache {
  Tensor x, h_in;      // inputs of the step (h_in already reset-masked)
  Tensor r, z, n, ghn; // [B,H] activations; ghn = Whn h + bhn
};

struct GruStepGrads {
  Tensor dx;     // [B,In]
  Tensor dh_in;  // [B,H]  gradient w.r.t. the (masked) state fed into the step
};

class Gru {
 public:
  static Result<Gru> create(std::int64_t in, std::int64_t hidden, std::shared_ptr<Device> d, Rng& rng);

  // One step. If `cache` is non-null the data needed by step_backward is stored in it.
  Result<Tensor> step(const Tensor& x, const Tensor& h_in, GruCache* cache);
  // Accumulates parameter grads; returns gradients w.r.t. the step's inputs.
  Result<GruStepGrads> step_backward(const GruCache& c, const Tensor& dh_out);

  std::span<Param> parameters() noexcept { return params_; }
  std::int64_t hidden() const { return hidden_; }
  std::int64_t input() const { return in_; }

  Gru(Gru&&) noexcept = default;
  Gru& operator=(Gru&&) noexcept = default;

 private:
  Gru(std::vector<Param> p, std::int64_t in, std::int64_t h) : params_(std::move(p)), in_(in), hidden_(h) {}
  std::vector<Param> params_;  // Wi, bi, Wh, bh
  std::int64_t in_{0}, hidden_{0};
};

}  // namespace gyre
