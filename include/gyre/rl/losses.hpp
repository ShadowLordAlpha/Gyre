#pragma once
// Fused PPO and imitation losses with analytic gradients w.r.t. the policy logits and the
// value prediction. Everything is a mean over the batch of B samples.
#include "gyre/rl/dist.hpp"

namespace gyre {

struct PpoConfig {
  float clip{0.2f};      // epsilon of the clipped surrogate
  float vf_coef{0.5f};   // weight of 0.5*(V - return)^2
  float ent_coef{0.01f}; // weight of the entropy bonus
};

struct PpoBatch {
  Tensor actions;     // i32 [B,H]
  Tensor old_logp;    // f32 [B]   joint log-prob under the behaviour policy
  Tensor advantages;  // f32 [B]   (normalise first: normalize_advantages_)
  Tensor returns;     // f32 [B]   value targets
};

struct PpoLoss {
  float total{0}, policy{0}, value{0}, entropy{0};
  float approx_kl{0};   // mean of (r-1) - log r  >= 0
  float clip_frac{0};   // fraction of samples with |r-1| > clip
  Tensor dlogits;       // [B,total]
  Tensor dvalue;        // [B]
};

// total = policy + vf_coef*value - ent_coef*entropy.
Result<PpoLoss> ppo_loss(const Tensor& logits, const Tensor& value, const HeadLayout& L,
                         const HeadMasks& m, const PpoBatch& batch, const PpoConfig& cfg);

// Zero-mean, unit-variance advantages in place (population std, +eps).
Result<void> normalize_advantages_(Tensor& adv, float eps = 1e-8f);

struct ImitationLoss {
  float loss{0};   // mean over batch of -joint log-prob of the demonstrated actions
  Tensor dlogits;  // [B,total]
};

// Supervised cross-entropy on (observation, action) pairs through the same heads.
Result<ImitationLoss> imitation_loss(const Tensor& logits, const HeadLayout& L, const HeadMasks& m,
                                     const Tensor& actions);

}  // namespace gyre
