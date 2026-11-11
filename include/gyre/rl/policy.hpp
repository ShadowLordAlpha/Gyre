#pragma once
// Actor-critic network: SetEncoder trunk -> [optional GRU core] -> {masked categorical heads,
// scalar value head}. Game-agnostic: inputs are entity feature arrays, a validity mask and
// global features.
//
// Feedforward (gru_hidden == 0): use forward()/backward().
// Recurrent   (gru_hidden  > 0): use forward_step() to act (one step, carries state) and
//   forward_seq()/backward_seq() to train (T steps over B streams, backprop through time).
//   State convention: the state fed into step t is h_out of step t-1; reset[t][b] = 1 zeroes it
//   (episode start). reset is typically "done flag of the previous step".
#include "gyre/rl/dist.hpp"
#include "gyre/rl/gru.hpp"
#include "gyre/rl/set_encoder.hpp"

#include <optional>
#include <vector>

namespace gyre {

struct Obs {
  Tensor entities;  // f32 [B,N,F]
  Tensor mask;      // u8  [B,N]   1 = real entity
  Tensor global;    // f32 [B,G]
};

struct PolicyConfig {
  SetEncoderConfig encoder;
  std::vector<std::int64_t> head_sizes;  // options per head
  std::int64_t gru_hidden{0};            // 0 = feedforward
};

struct PolicyOut {
  Tensor logits;  // f32 [B,total]  raw, unmasked
  Tensor value;   // f32 [B]
};

struct StepOut {
  PolicyOut out;
  Tensor h;  // f32 [B,H] new recurrent state
};

struct SeqOut {
  PolicyOut out;  // rows ordered t-major: row = t*B + b
  Tensor h_last;  // f32 [B,H]
};

class PolicyNet {
 public:
  static Result<PolicyNet> create(const PolicyConfig& c, std::shared_ptr<Device> d, Rng& rng);

  // ---- feedforward
  Result<PolicyOut> forward(const Obs& obs, ForwardCtx& ctx);
  // dlogits [B,total], dvalue [B] -> accumulates parameter grads.
  Result<void> backward(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx);

  // ---- recurrent
  bool recurrent() const { return gru_.has_value(); }
  std::int64_t hidden_dim() const { return cfg_.gru_hidden; }
  // Acting: one step for B streams, no tape. reset u8 [B].
  Result<StepOut> forward_step(const Obs& obs, const Tensor& h_in, const Tensor& reset);
  // Training: obs has T*B rows (t-major), h0 [B,H] (state before step 0), reset u8 [T,B].
  Result<SeqOut> forward_seq(const Obs& obs, const Tensor& h0, const Tensor& reset, ForwardCtx& ctx);
  // dlogits [T*B,total], dvalue [T*B]. Truncated BPTT: no gradient into h0.
  Result<void> backward_seq(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx);

  std::span<Param> parameters() noexcept { return flat_; }
  Result<void> zero_grad();
  const HeadLayout& layout() const { return layout_; }
  const PolicyConfig& config() const { return cfg_; }

  PolicyNet(PolicyNet&&) noexcept = default;
  PolicyNet& operator=(PolicyNet&&) noexcept = default;

 private:
  PolicyNet(PolicyConfig c, HeadLayout l, SetEncoder enc, std::optional<Gru> gru, Linear heads,
            Linear value);
  Result<PolicyOut> heads_forward(const Tensor& feat, ForwardCtx& ctx);
  Result<Tensor> heads_backward(const Tensor& dlogits, const Tensor& dvalue, ForwardCtx& ctx);

  PolicyConfig cfg_;
  HeadLayout layout_;
  SetEncoder enc_;
  std::optional<Gru> gru_;
  Linear heads_;
  Linear value_;
  std::vector<Param> flat_;
  // BPTT tape (valid between forward_seq and backward_seq)
  std::vector<GruCache> tape_;
  std::vector<Tensor> keeps_;  // u8 [B] per step: 1 = keep carried state
  std::int64_t tape_T_{0}, tape_B_{0};
};

}  // namespace gyre
