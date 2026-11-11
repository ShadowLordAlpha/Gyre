#pragma once
// Composable masked categorical heads.
//
// A policy has H heads. Head h picks one of sizes[h] discrete options. All heads' logits
// are packed side by side in one row of length total = sum(sizes). Per sample, each head
// carries
//   * a legality mask over its options (u8, 1 = legal), and
//   * an "active" flag (u8 [B,H]). Inactive heads are irrelevant to this sample: they have no
//     sampled action (-1) and contribute nothing to log-prob, entropy or gradients.
// The joint log-prob / entropy of a sample is the SUM over its active heads.
//
// Gyre never interprets what a head means; masks and actions are plain integers.
#include "gyre/tensor.hpp"

#include <functional>
#include <span>
#include <vector>

namespace gyre {

class Rng;

struct HeadLayout {
  std::vector<std::int64_t> sizes;
  static Result<HeadLayout> make(std::vector<std::int64_t> sizes);  // all sizes >= 1
  std::int64_t heads() const { return static_cast<std::int64_t>(sizes.size()); }
  std::int64_t total() const;
  std::int64_t offset(std::int64_t h) const;
};

struct HeadMasks {
  Tensor legal;   // u8 [B,total]
  Tensor active;  // u8 [B,H]
};

// Per-head log-softmax over legal options. Illegal options have prob 0 (logp = kMaskFill);
// inactive heads are all zeros. Error if an ACTIVE head has no legal option.
struct HeadDist {
  Tensor logp;  // f32 [B,total]
  Tensor prob;  // f32 [B,total]
};

Result<HeadDist> head_dist(const Tensor& logits, const HeadLayout& L, const HeadMasks& m);

// i32 [B,H]; inactive heads get -1. greedy=true takes the arg-max legal option (rng unused).
Result<Tensor> sample_actions(const HeadDist& d, const HeadLayout& L, const HeadMasks& m, Rng* rng,
                              bool greedy = false);

// Autoregressive sampling where a later head's relevance depends on earlier choices.
// Heads are sampled in order 0..H-1. After head h is sampled for sample b, `after_head` may
// edit that sample's active flags of LATER heads (index > h), e.g. switch off heads irrelevant
// to the chosen intent. Edits to flags of heads <= h are ignored (restored).
// `actions_b` holds the actions sampled so far (-1 for heads not sampled). Returns the
// actions and the FINAL active mask, which is what log_prob / the rollout buffer must use.
using ActiveFn = std::function<void(std::int64_t b, std::int64_t h,
                                    std::span<const std::int32_t> actions_b,
                                    std::span<std::uint8_t> active_b)>;
struct SampleOut {
  Tensor actions;  // i32 [B,H], -1 where inactive
  Tensor active;   // u8  [B,H]
};
Result<SampleOut> sample_conditional(const Tensor& logits, const HeadLayout& L, const Tensor& legal,
                                     const Tensor& initial_active, Rng* rng, bool greedy,
                                     const ActiveFn& after_head = {});

// Sum over active heads of log pi(a_h). actions i32 [B,H] (entries of inactive heads ignored).
// Error if an active head's action is out of range or illegal.
Result<Tensor> log_prob(const HeadDist& d, const HeadLayout& L, const HeadMasks& m,
                        const Tensor& actions);

// Sum over active heads of the entropy. [B]
Result<Tensor> entropy(const HeadDist& d, const HeadLayout& L, const HeadMasks& m);

// Gradient w.r.t. the raw logits [B,total] of  sum_b dlogp[b]*logp(a_b) + dent[b]*entropy_b.
// Exactly zero for illegal options and for inactive heads.
Result<Tensor> log_prob_entropy_backward(const HeadDist& d, const HeadLayout& L, const HeadMasks& m,
                                         const Tensor& actions, const Tensor& dlogp,
                                         const Tensor& dent);

}  // namespace gyre
