#pragma once
// Supervised pretraining from demonstrations: cross-entropy on (observation, action) pairs
// through the SAME PolicyNet used for PPO, so a pretrained net drops straight into RL.
//
// Demonstrations are stored in a RolloutBuffer, one stream per replay (or per replay chunk):
//   * legal / active / actions are the expert's masks and choices (inactive heads: no loss);
//   * logp, value and reward are unused unless value_coef > 0;
//   * recurrent nets: every stream must begin with reset = 1 and is trained over its full
//     length (h_in may be zeros); episode starts inside a stream are marked by reset flags.
// Optional value pretraining: fill rewards, push value = 0 and call compute_gae(.., lambda=1)
// so buffer.returns() are discounted Monte-Carlo returns; set value_coef > 0.
#include "gyre/rl/ppo.hpp"

namespace gyre {

struct ImitationConfig {
  float lr{1e-3f};
  int minibatches{4};         // per pass over the data (samples, or whole streams if recurrent)
  float max_grad_norm{1.f};   // <= 0: no clipping
  float value_coef{0.f};      // > 0: also regress V to buffer.returns()
  std::uint64_t seed{1};
};

struct ImitationStats {
  float loss{0};      // mean NLL of the expert actions
  float accuracy{0};  // greedy arg-max == expert action, over active (sample, head) pairs
  float value_loss{0};
  float grad_norm{0};   // 0 when max_grad_norm <= 0 (clipping off)
  int updates{0};
};

class ImitationTrainer {
 public:
  static Result<ImitationTrainer> create(PolicyNet& net, ImitationConfig cfg);
  // One shuffled pass over the whole buffer with optimiser steps.
  Result<ImitationStats> epoch(RolloutBuffer& buf);
  // Same metrics, no parameter update (held-out data).
  Result<ImitationStats> evaluate(RolloutBuffer& buf);
  void set_lr(float lr) { adam_.lr = lr; }

 private:
  ImitationTrainer(PolicyNet& n, ImitationConfig c, Adam a) : net_(&n), cfg_(c), adam_(std::move(a)), rng_(c.seed) {}
  Result<ImitationStats> run(RolloutBuffer& buf, bool train);
  PolicyNet* net_;
  ImitationConfig cfg_;
  Adam adam_;
  Rng rng_;
};

// Convenience: a StepRecord for a demonstration step (dummy logp/value/reward/done).
// Errors (not aborts) on a malformed observation.
// `h_in` is required (zeros) when the buffer is recurrent.
Result<StepRecord> make_demo_step(const Obs& obs, const Tensor& legal, const Tensor& active,
                          const Tensor& actions, std::vector<std::uint8_t> reset = {},
                          std::optional<Tensor> h_in = std::nullopt);

}  // namespace gyre
