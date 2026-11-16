#pragma once
// Dense Llama-family decoder (SmolLM2, TinyLlama, Llama-2/3 shapes without rope scaling), CPU
// inference only: no gradients, no Vulkan.
//
//   x = embed[ids]
//   per layer:  x += Wo · attn(rope(Wq·rms(x)), rope(Wk·rms(x)), Wv·rms(x))   (causal GQA)
//               x += Wdown · (silu(Wgate·rms(x)) * Wup·rms(x))                 (SwiGLU)
//   logits = rms(x) · head      (head = embed^T when tied)
//
// Weights are stored in Gyre's [in, out] layout (Hugging Face nn.Linear is [out, in]). RoPE is
// applied INTERLEAVED like gyre::rope (pairs 2i, 2i+1 rotate with theta^(-2i/hd)); the Hugging
// Face importer permutes the q/k output columns inside every head (HF index i -> 2i, i + hd/2 ->
// 2i + 1) so this equals HF's rotate_half on the original weights. q.k is unchanged by applying
// the same permutation to q and k, so nothing else moves.
//
// A KV cache (`LlamaCache`, K/V for n_kv_head heads only) makes generation O(T) per token:
// `prefill`/`step` append to it; `forward_full` is the reference full recompute.
#include "gyre/checkpoint.hpp"
#include "gyre/rng.hpp"
#include "gyre/tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gyre {

struct LlamaConfig {
  std::int64_t vocab{256};
  std::int64_t d_model{64};
  std::int64_t n_layer{2};
  std::int64_t n_head{4};
  std::int64_t n_kv_head{2};
  std::int64_t head_dim{16};
  std::int64_t d_ff{128};
  std::int64_t max_seq{2048};  // max_position_embeddings (cache capacity default)
  float rms_eps{1e-5f};
  float rope_theta{10000.f};
  bool tie_embeddings{true};
  std::int32_t bos_id{-1};
  std::int32_t eos_id{-1};

  Result<void> validate() const;
  std::int64_t param_count() const;
  std::string to_json() const;  // Gyre doc "config" object
  static Result<LlamaConfig> from_json(std::string_view json);
  // Hugging Face config.json (LlamaForCausalLM). Errors on anything this class does not implement:
  // rope_scaling, attention/mlp bias, hidden_act != silu, pretraining_tp != 1, rope_interleaved.
  static Result<LlamaConfig> from_hf_json(std::string_view json);
};

// Per-sequence K/V cache: [n_layer][n_kv_head][capacity][head_dim] each for K and V.
struct LlamaCache {
  std::int64_t len{0};
  std::int64_t capacity{0};
  std::vector<float> k, v;
  void clear() noexcept { len = 0; }
};

class LlamaLM {
 public:
  // Random init (tests). Weight std 0.02, norms 1.
  static Result<LlamaLM> create(const LlamaConfig& c, Rng& rng);
  // Takes ownership of named f32 CPU tensors in Gyre layout (see param_names()); shapes are checked.
  static Result<LlamaLM> from_tensors(const LlamaConfig& c, std::vector<Tensor> weights);
  // arch "llama" `.gyre` (weights stay memory-mapped).
  static Result<LlamaLM> load_gyre(const std::filesystem::path& path);
  // Writes a GYRE1 v2 file with arch "llama". `tokenizer_json` is the tokenizer object (may be
  // empty); `extra_config` keys (e.g. "holdout", "source") are merged into the config object.
  Result<void> save_gyre(const std::filesystem::path& path, std::string_view tokenizer_json,
                         std::string_view extra_config_json = {}) const;

  const LlamaConfig& config() const { return cfg_; }
  std::vector<std::string> param_names() const;
  std::span<const Tensor> weights() const { return w_; }
  std::int64_t n_params() const;

  Result<LlamaCache> make_cache(std::int64_t capacity = 0) const;  // 0 = config max_seq
  // Appends `ids` at positions cache.len.. and returns logits for the last `n_logits` of them
  // ([n_logits, vocab], row-major). n_logits = 0 returns nothing (hidden states still advance).
  Result<std::vector<float>> forward(LlamaCache& cache, std::span<const std::int32_t> ids,
                                     std::int64_t n_logits = 1);
  // Final-norm hidden states [ids.size(), d_model] for `ids` appended to the cache, then
  // logits for rows [row0, row0 + n) of such a hidden block. Lets an evaluator score a long
  // window without materialising [T, vocab] at once.
  Result<std::vector<float>> forward_hidden(LlamaCache& cache, std::span<const std::int32_t> ids);
  Result<std::vector<float>> logits_from_hidden(std::span<const float> hidden, std::int64_t row0,
                                                std::int64_t n) const;
  // Reference: fresh cache, all positions, logits [ids.size(), vocab].
  Result<std::vector<float>> forward_full(std::span<const std::int32_t> ids);

  struct GenStats {
    double prefill_s{0}, decode_s{0};
    std::int64_t new_tokens{0};
  };
  // temperature <= 0: greedy. use_cache=false recomputes the whole sequence for every token.
  // Stops early when `eos` >= 0 is produced (it is included in the result).
  Result<std::vector<std::int32_t>> generate(std::vector<std::int32_t> prompt, int max_new, float temperature,
                                             Rng* rng, bool use_cache = true, std::int32_t eos = -1,
                                             GenStats* stats = nullptr);

  LlamaLM(LlamaLM&&) noexcept = default;
  LlamaLM& operator=(LlamaLM&&) noexcept = default;

 private:
  LlamaLM() = default;
  Result<void> bind();  // fills the raw pointers below and the transposed tied head
  LlamaConfig cfg_{};
  std::vector<Tensor> w_;  // param_names() order
  struct Layer {
    const float *attn_norm, *wq, *wk, *wv, *wo, *mlp_norm, *wg, *wu, *wd;
  };
  const float* embed_{nullptr};
  const float* final_norm_{nullptr};
  const float* head_{nullptr};      // [d, V]
  std::vector<float> head_t_;       // tied: embed^T
  std::vector<Layer> layers_;
  std::vector<float> rope_cos_, rope_sin_;  // [max_seq][head_dim/2]
  std::int64_t rope_len_{0};
  Result<void> ensure_rope(std::int64_t n);
  // scratch (grown on demand)
  std::vector<float> x_, xn_, q_, k_, v_, att_, g_, u_;
};

}  // namespace gyre
