#pragma once
// gyre-infer: forward-only CPU runtime for policy/value nets trained with Gyre.
//
// Standalone: plain C++20, no dependency on the main `gyre` library, OpenMP, Vulkan, an
// autodiff tape, optimisers or a JSON parser. Single-threaded; one decision = one `step()`.
// Weights come from the flat file written by gyre::export_policy_flat (see policy_io.hpp).
//
// Thread safety: a loaded `Policy` is immutable and may be shared; every thread/bot needs its
// own `Session` (recurrent state + scratch buffers).
//
// Inputs are game-agnostic: entity feature rows, an optional validity mask, global features.
// Output: raw logits for all heads packed side by side, and the scalar value. Legality and
// head-gating are the caller's job; `pick_masked` samples/arg-maxes one head.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gyre_infer {

struct Info {
  int feat_dim{0}, global_dim{0}, max_entities{0}, d_model{0}, n_layer{0}, n_head{0}, d_ff{0},
      out_dim{0}, gru_hidden{0};
  std::vector<int> head_sizes;
  int total_logits() const;
  int head_offset(int head) const;
  bool recurrent() const { return gru_hidden > 0; }
};

struct Input {
  const float* entities{nullptr};      // [n_slots, feat_dim], row-major
  const std::uint8_t* mask{nullptr};   // [n_slots], 1 = real entity; nullptr = all valid
  int n_slots{0};                      // <= Info::max_entities
  const float* global{nullptr};        // [global_dim] (may be null when global_dim == 0)
  bool reset{false};                   // episode start: zero the recurrent state first
};

class Session {
 public:
  const std::string& error() const { return error_; }
  // Zero the recurrent state (same as Input::reset on the next step).
  void reset();

 private:
  friend class Policy;
  std::vector<float> h_;                  // recurrent state [gru_hidden]
  std::vector<float> a_, b_, c_, d_, e_;  // scratch, sized lazily
  std::vector<float> scores_;
  std::vector<int> rows_;
  std::string error_;
};

class Policy {
 public:
  static std::unique_ptr<Policy> load(const std::string& path, std::string* error = nullptr);
  static std::unique_ptr<Policy> load_from_memory(const void* data, std::size_t bytes,
                                                  std::string* error = nullptr);
  ~Policy();

  const Info& info() const;
  Session make_session() const;

  // logits: info().total_logits() floats; value: 1 float. Returns false (and sets
  // session.error()) on bad input; outputs are untouched then.
  bool step(Session& s, const Input& in, float* logits, float* value) const;

 private:
  Policy();
  static std::unique_ptr<Policy> load_impl(const void* data, std::size_t bytes, std::string* error);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Pick one option among `n` logits with `legal[i] != 0` (legal may be null = all legal).
// u in [0,1): sample from the softmax over legal options by inverse CDF; u < 0: arg-max.
// Returns the option index, or -1 if nothing is legal.
int pick_masked(const float* logits, const std::uint8_t* legal, int n, float u);

}  // namespace gyre_infer
