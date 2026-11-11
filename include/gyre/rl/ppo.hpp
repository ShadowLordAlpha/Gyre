#pragma once
// Rollout storage, GAE and the PPO update loop. Game-agnostic: an "environment" is whatever
// the caller runs; Gyre only sees batched observations, masks, actions and rewards.
//
// Layout: `envs` parallel streams, `steps` consecutive steps each. Sample index = t*envs + e.
// Conventions
//   * done[t][e] = 1 means the episode of stream e ended AFTER step t (its action at t was the
//     last one). Step t+1 of that stream then belongs to a fresh episode: GAE does not
//     bootstrap across it, and a recurrent core must reset its state there.
//   * Time-limit truncation: add gamma*V(final obs) to that step's reward, then set done.
//   * `value` and `logp` are what the behaviour policy produced when acting (policy_step).
#include "gyre/optim.hpp"
#include "gyre/rl/losses.hpp"
#include "gyre/rl/policy.hpp"

#include <optional>
#include <vector>

namespace gyre {

// ---------------------------------------------------------------- acting

struct ActResult {
  Tensor actions;  // i32 [B,H]  (-1 where inactive)
  Tensor active;   // u8  [B,H]  final active mask (use it when storing the step)
  Tensor logp;     // f32 [B]    joint log-prob of `actions` under the final active mask
  Tensor value;    // f32 [B]
};

// Inference-mode forward + autoregressive sampling (see sample_conditional).
// `initial_active` is typically all ones; `after_head` switches off irrelevant heads.
Result<ActResult> policy_step(PolicyNet& net, const Obs& obs, const Tensor& legal,
                              const Tensor& initial_active, Rng* rng, bool greedy = false,
                              const ActiveFn& after_head = {});

struct RecActResult {
  ActResult act;
  Tensor h_out;  // f32 [B,H]: carry into the next step
};
// Recurrent acting step. `h_in` is the carried state (BEFORE reset masking); reset u8 [B] = 1
// zeroes it first (episode start). Store h_in and reset in the StepRecord; h_out is next h_in.
Result<RecActResult> policy_step_rec(PolicyNet& net, const Obs& obs, const Tensor& legal,
                                     const Tensor& initial_active, const Tensor& h_in,
                                     const Tensor& reset, Rng* rng, bool greedy = false,
                                     const ActiveFn& after_head = {});

// ---------------------------------------------------------------- rollout

struct RolloutSpec {
  std::int64_t steps{128};
  std::int64_t envs{8};
  std::int64_t max_entities{128};
  std::int64_t feat_dim{1};
  std::int64_t global_dim{0};
  std::int64_t hidden_dim{0};  // > 0 for recurrent policies (stores h_in / reset per step)
  HeadLayout layout;
};

struct StepRecord {
  Obs obs;            // batched over envs: [E,N,F], [E,N], [E,G]
  Tensor legal;       // u8 [E,total]
  Tensor active;      // u8 [E,H]  final
  Tensor actions;     // i32 [E,H]
  Tensor logp;        // f32 [E]
  Tensor value;       // f32 [E]
  std::vector<float> reward;       // [E]
  std::vector<std::uint8_t> done;  // [E]
  // Recurrent policies only (spec.hidden_dim > 0):
  std::optional<Tensor> h_in;       // f32 [E,H] carried state before reset masking
  std::vector<std::uint8_t> reset;  // [E] 1 = episode start at this step
};

struct Minibatch {
  Obs obs;
  HeadMasks masks;
  PpoBatch batch;
};

// A time-contiguous chunk for BPTT: samples ordered t-major (row = t*B + b).
struct SeqBatch {
  Minibatch mb;
  Tensor h0;     // f32 [B,H] stored carried state at the chunk's first step
  Tensor reset;  // u8 [len,B]
  std::int64_t len{0}, B{0};
};

class RolloutBuffer {
 public:
  static Result<RolloutBuffer> create(RolloutSpec spec);

  Result<void> push(const StepRecord& r);
  bool full() const { return t_ == spec_.steps; }
  std::int64_t size() const { return spec_.steps * spec_.envs; }
  std::int64_t steps_filled() const { return t_; }
  void clear();
  bool gae_computed() const { return gae_done_; }  // since the last push/clear

  // Generalised advantage estimation over the filled buffer. `last_value` [envs] is V of the
  // observation after the final step (ignored for streams whose last step was done).
  Result<void> compute_gae(std::span<const float> last_value, float gamma, float lambda);

  // Zero-mean / unit-std advantages over the whole buffer.
  void normalize_advantages();

  std::span<const float> advantages() const { return adv_; }
  std::span<const float> returns() const { return ret_; }
  std::span<const float> rewards() const { return rew_; }
  std::span<const std::uint8_t> dones() const { return done_; }
  const RolloutSpec& spec() const { return spec_; }

  // Gather arbitrary sample indices (0 <= i < size()) into a training minibatch.
  Result<Minibatch> minibatch(std::span<const std::int64_t> idx) const;
  // Steps [t0, t0+len) of the given streams, for recurrent training.
  Result<SeqBatch> seq_minibatch(std::span<const std::int64_t> envs, std::int64_t t0, std::int64_t len) const;

 private:
  explicit RolloutBuffer(RolloutSpec s);
  RolloutSpec spec_;
  std::int64_t t_{0};
  bool gae_done_{false};
  std::vector<float> ent_, glob_, rew_, logp_, val_, adv_, ret_, hin_;
  std::vector<std::uint8_t> mask_, legal_, active_, done_, reset_;
  std::vector<std::int32_t> act_;
};

// ---------------------------------------------------------------- trainer

struct PpoTrainConfig {
  PpoConfig loss;
  float lr{3e-4f};
  float gamma{0.99f};
  float lambda{0.95f};
  int epochs{4};
  int minibatches{4};
  float max_grad_norm{0.5f};  // <= 0: no clipping (and PpoStats::grad_norm then reads 0)
  bool normalize_advantages{true};
  int seq_len{0};             // recurrent: BPTT chunk length (0 = whole rollout)
  float target_kl{0.f};       // > 0: stop the update early once epoch KL exceeds 1.5x this
  std::uint64_t seed{1};      // minibatch shuffling
};

struct PpoStats {
  float total{0}, policy{0}, value{0}, entropy{0}, approx_kl{0}, clip_frac{0}, grad_norm{0};
  int epochs_run{0};
  int updates{0};  // optimiser steps taken
};

class PpoTrainer {
 public:
  static Result<PpoTrainer> create(PolicyNet& net, PpoTrainConfig cfg);
  // The caller runs buffer.compute_gae first. Then epochs x minibatches of Adam steps.
  // Advantages are normalised over the whole buffer when cfg.normalize_advantages.
  Result<PpoStats> update(RolloutBuffer& buf);
  void set_lr(float lr) { adam_.lr = lr; }
  const PpoTrainConfig& config() const { return cfg_; }

 private:
  PpoTrainer(PolicyNet& net, PpoTrainConfig c, Adam a) : net_(&net), cfg_(c), adam_(std::move(a)), rng_(c.seed) {}
  PolicyNet* net_;
  PpoTrainConfig cfg_;
  Adam adam_;
  Rng rng_;
};

}  // namespace gyre
