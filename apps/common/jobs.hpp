#pragma once

#include "gyre/ga/population.hpp"
#include "gyre/io/codec.hpp"
#include "gyre/nn/grok.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/train/loop.hpp"

#include <filesystem>
#include <functional>
#include <string>

namespace gyre::app {

struct CharLMOpts {
  std::filesystem::path data{"data/shakespeare.txt"};  // default corpus, not the model name
  std::filesystem::path ckpt{"data/charlm.gyre"};
  std::filesystem::path tok;       // *.gyre.json — reuse instead of training
  std::filesystem::path hf_dir;    // Hugging Face tokenizer directory
  std::filesystem::path sp_model;  // SentencePiece .model
  std::string preset{"medium"};  // tiny | medium | tinygpt | nanogpt
  std::string tokenizer{"bpe"};  // bpe | chars | bytes | unigram
  std::int32_t vocab_size{2000};  // BPE/unigram target
  double holdout{0.1};  // last fraction of raw bytes reserved; 0 = train on full file
  std::uint32_t steps{2000};
  std::uint32_t batch{4};
  std::uint32_t block{128};
  std::int64_t d_model{128};
  std::int64_t n_layer{4};
  std::int64_t n_head{4};
  std::int64_t d_ff{512};
  float lr{3e-4f};
  float lr_start{1e-3f};  // hotter start; decays to lr
  std::uint32_t lr_decay_steps{0};  // 0 = first 20% of steps
  std::uint32_t log_every{10};
  std::uint32_t ckpt_every{100};
  float temperature{0.8f};  // 0 = greedy
  bool recency_alibi{true};  // ALiBi token-distance bias; off for old checkpoints
};

void apply_charlm_preset(CharLMOpts& o);

struct GaOpts {
  std::uint32_t generations{80};
  std::uint32_t n{64};
  std::uint32_t dim{64};
};

using LogFn = std::function<void(std::string)>;

Result<void> run_onemax(const GaOpts& opts, LogFn log);

// Trains, writes GYRE1 to opts.ckpt, returns last generate sample.
Result<std::string> run_charlm_train(const CharLMOpts& opts, LogFn log);

Result<std::string> run_charlm_generate(const CharLMOpts& opts, std::string prompt, int max_new,
                                        LogFn log);

Result<void> run_charlm_export_onnx(const CharLMOpts& opts, const std::filesystem::path& onnx_path,
                                   LogFn log);

struct EvalReport {
  double nats_per_token{0};
  double nats_per_char{0};
  double bpc{0};
  double chars_per_token{0};
  std::uint64_t n_pred{0};
  std::uint64_t n_chars{0};
  std::uint64_t n_windows{0};
  std::uint64_t n_params{0};
  std::int64_t vocab{0};
  std::int64_t block{0};
};

// Held-out eval on the last `split` fraction of --data. split<=0 uses checkpoint holdout.
Result<EvalReport> run_charlm_eval(const CharLMOpts& opts, double split, LogFn log);

Result<void> run_tok_train(const CharLMOpts& opts, const std::filesystem::path& out, LogFn log);
Result<void> run_tok_export_hf(const std::filesystem::path& tok, const std::filesystem::path& hf_dir,
                               LogFn log);
Result<void> run_tok_import(const CharLMOpts& opts, const std::filesystem::path& out, LogFn log);

struct GrokCliOpts {
  std::filesystem::path config{"data/grok2/config.json"};
  std::filesystem::path weights{"data/grok2"};
  std::filesystem::path out;
  std::string file_substr;
  std::string preset{"mini"};
  std::string prompt{"To be"};
  int max_new{32};
  std::filesystem::path tok;
  std::filesystem::path lora;
  std::uint64_t pack_max{2ull << 20};
  int max_tensors{32};
  std::uint64_t chunk_bytes{1ull << 20};
  std::string codec{"identity"};  // identity | alp | zfp
};

Result<void> run_grok_info(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_inspect(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_compress_probe(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_pack(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_save(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_gen(const GrokCliOpts& o, LogFn log);
Result<void> run_grok_import(const GrokCliOpts& o, LogFn log);
Result<void> run_ckpt_probe(const std::filesystem::path& ckpt, int max_tensors, LogFn log);

}  // namespace gyre::app
