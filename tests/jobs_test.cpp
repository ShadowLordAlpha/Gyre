// App job regressions (apps/common/jobs.cpp).

#include "jobs.hpp"

#include "gyre/checkpoint.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/transformer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path scratch_dir() {
  // One directory per test: ctest runs each TEST in its own process, possibly in parallel.
  const auto* ti = ::testing::UnitTest::GetInstance()->current_test_info();
  auto d = std::filesystem::temp_directory_path() / (std::string("gyre_jobs_test_") + (ti ? ti->name() : "x"));
  std::filesystem::create_directories(d);
  return d;
}

std::filesystem::path write_corpus() {
  auto p = scratch_dir() / "corpus.txt";
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  for (int i = 0; i < 60; ++i) o << "the quick brown fox jumps over the lazy dog " << i << "\n";
  return p;
}

gyre::app::CharLMOpts tiny_opts(const std::filesystem::path& data, const std::filesystem::path& ckpt) {
  gyre::app::CharLMOpts o;
  o.data = data;
  o.ckpt = ckpt;
  o.preset = "tiny";
  gyre::app::apply_charlm_preset(o);
  o.tokenizer = "chars";
  o.block = 16;
  o.d_model = 16;
  o.n_head = 2;
  o.n_layer = 1;
  o.d_ff = 32;
  o.steps = 2;
  o.batch = 2;
  o.log_every = 0;
  o.ckpt_every = 0;
  return o;
}

double doc_number(const gyre::GyreDoc& d, const char* key) {
  // config_json is a flat JSON object; find "key":value without a JSON dep here.
  const std::string k = std::string("\"") + key + "\":";
  auto at = d.config_json.find(k);
  if (at == std::string::npos) return -1.0;
  return std::stod(d.config_json.substr(at + k.size()));
}

}  // namespace

// Regression: resuming or wiring inherited holdout/dropout from the parent but
// saved the CLI values, so eval later scored in-sample text as "held out".
TEST(Jobs, ResumeSavesEffectiveHoldoutAndDropout) {
  const auto data = write_corpus();
  const auto parent = scratch_dir() / "parent.gyre";
  auto o = tiny_opts(data, parent);
  o.holdout = 0.0;  // train on the full file
  o.dropout = 0.2f;
  ASSERT_TRUE(gyre::app::run_charlm_train(o, nullptr));

  const auto child = scratch_dir() / "child.gyre";
  auto r = tiny_opts(data, child);
  r.wire = parent;
  r.holdout = 0.1;  // CLI default; the parent's 0 must win
  r.dropout = 0.f;  // CLI default; the parent's 0.2 must win
  auto t = gyre::app::run_charlm_train(r, nullptr);
  ASSERT_TRUE(t) << t.error().message;
  auto doc = gyre::peek_gyre(child);
  ASSERT_TRUE(doc) << doc.error().message;
  EXPECT_DOUBLE_EQ(doc->holdout(), 0.0);
  EXPECT_NEAR(doc_number(*doc, "dropout"), 0.2, 1e-6);

  // Out-of-range CLI holdout is clamped for training and must be stored clamped.
  const auto fresh = scratch_dir() / "fresh.gyre";
  auto f = tiny_opts(data, fresh);
  f.holdout = -0.5;
  ASSERT_TRUE(gyre::app::run_charlm_train(f, nullptr));
  auto fd = gyre::peek_gyre(fresh);
  ASSERT_TRUE(fd);
  EXPECT_DOUBLE_EQ(fd->holdout(), 0.0);
}

// Regression: export re-applied the CLI preset, so a checkpoint whose shape
// differs from the preset failed to load, and recency came from the CLI.
TEST(Jobs, ExportUsesCheckpointConfig) {
  const auto data = write_corpus();
  const auto ckpt = scratch_dir() / "export.gyre";
  auto o = tiny_opts(data, ckpt);
  o.recency_alibi = false;
  ASSERT_TRUE(gyre::app::run_charlm_train(o, nullptr));

  gyre::app::CharLMOpts e;  // CLI defaults: preset medium, ALiBi on
  e.data = data;
  e.ckpt = ckpt;
  gyre::app::apply_charlm_preset(e);
  std::string log;
  auto onnx = scratch_dir() / "export.onnx";
  auto r = gyre::app::run_charlm_export_onnx(e, onnx, [&](std::string s) { log += s + "\n"; });
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_TRUE(std::filesystem::exists(onnx));
  EXPECT_NE(log.find("d=16"), std::string::npos) << log;
  EXPECT_NE(log.find("recency=none"), std::string::npos) << log;
}

// Regression: eval scored only whole non-overlapping T-windows (dropping the tail and the first
// token) yet divided by every val char, so nats/char and BPC were biased low. Every token but the
// first must be predicted exactly once and the denominator must be the bytes of those tokens.
TEST(Jobs, EvalScoresTailAndNormalisesByScoredChars) {
  const auto data = write_corpus();
  const auto ckpt = scratch_dir() / "eval_tail.gyre";
  auto o = tiny_opts(data, ckpt);
  o.recency_alibi = true;
  ASSERT_TRUE(gyre::app::run_charlm_train(o, nullptr));

  auto rep = gyre::app::run_charlm_eval(o, 0.1, nullptr);
  ASSERT_TRUE(rep) << rep.error().message;

  std::string text;
  {
    std::ifstream in(data, std::ios::binary);
    text.assign(std::istreambuf_iterator<char>(in), {});
  }
  const auto split_at = static_cast<std::size_t>((1.0 - 0.1) * static_cast<double>(text.size()));
  const std::string val = text.substr(split_at);
  const std::int64_t T = 16;
  const std::int64_t n = static_cast<std::int64_t>(val.size());  // chars tokenizer: 1 token per byte
  ASSERT_NE((n - 1) % T, 0) << "corpus must leave a tail for this test";
  EXPECT_EQ(rep->n_pred, static_cast<std::uint64_t>(n - 1));
  EXPECT_EQ(rep->n_chars, static_cast<std::uint64_t>(n - 1));
  EXPECT_EQ(rep->n_windows, static_cast<std::uint64_t>((n - 1 + T - 1) / T));
  EXPECT_NEAR(rep->nats_per_char, rep->nats_per_token, 1e-9);

  // Independent recomputation: every target j in [1, n) from the window that covers it.
  auto pk = gyre::peek_gyre(ckpt);
  ASSERT_TRUE(pk);
  auto tok = gyre::Tokenizer::from_json(pk->to_json());
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode(val);
  ASSERT_TRUE(ids);
  ASSERT_EQ(static_cast<std::int64_t>(ids->size()), n);
  gyre::CharLMConfig cfg;
  cfg.vocab = (*tok)->vocab_size();
  cfg.block_size = T;
  cfg.n_layer = 1;
  cfg.n_head = 2;
  cfg.d_model = 16;
  cfg.d_ff = 32;
  cfg.recency_alibi = true;
  auto dev = gyre::Device::cpu();
  ASSERT_TRUE(dev);
  gyre::Rng rng(1);
  auto model = gyre::CharLM::create(cfg, *dev, rng);
  ASSERT_TRUE(model);
  gyre::CheckpointMeta meta;
  ASSERT_TRUE(gyre::load_gyre1(ckpt, model->parameters(), nullptr, meta));
  const std::int64_t V = cfg.vocab;
  auto window_logits = [&](std::int64_t start) {
    std::vector<std::byte> xb(static_cast<std::size_t>(T) * 4);
    std::memcpy(xb.data(), ids->data() + start, xb.size());
    std::int64_t sh[2] = {1, T};
    auto x = gyre::Tensor::from_host(xb, sh, gyre::DType::i32, *dev);
    EXPECT_TRUE(x);
    gyre::ForwardCtx ctx;
    ctx.train = false;
    auto lg = model->forward(*x, ctx);
    EXPECT_TRUE(lg);
    auto raw = lg->to_host_vec();
    EXPECT_TRUE(raw);
    std::vector<float> f(static_cast<std::size_t>(T * V));
    EXPECT_EQ(raw->size(), f.size() * 4);
    std::memcpy(f.data(), raw->data(), f.size() * 4);
    return f;
  };
  double nll = 0;
  for (std::int64_t j = 1; j < n; ++j) {
    std::int64_t start = ((j - 1) / T) * T;
    if (start + T + 1 > n) start = n - 1 - T;  // tail: window flush with the end
    const auto f = window_logits(start);
    const float* row = f.data() + (j - 1 - start) * V;
    const float mx = *std::max_element(row, row + V);
    double z = 0;
    for (std::int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v] - mx));
    nll += std::log(z) - static_cast<double>(row[(*ids)[static_cast<std::size_t>(j)]] - mx);
  }
  EXPECT_NEAR(nll / static_cast<double>(n - 1), rep->nats_per_token, 2e-4);
}

// Regression: apply_charlm_preset treated batch==4 as "unset", so an explicit --batch 4 on
// tinygpt/nanogpt silently became 2.
TEST(Jobs, ExplicitBatchSurvivesPreset) {
  gyre::app::CharLMOpts d;
  d.preset = "nanogpt";
  gyre::app::apply_charlm_preset(d);
  EXPECT_EQ(d.batch, 2u);  // default is still lowered for CPU

  gyre::app::CharLMOpts e;
  e.preset = "nanogpt";
  e.batch = 4;
  e.batch_set = true;
  gyre::app::apply_charlm_preset(e);
  EXPECT_EQ(e.batch, 4u);
}

TEST(Jobs, KnownPresetAndTokenizerNames) {
  using namespace gyre::app;
  for (auto n : {"tiny", "medium", "tinygpt", "nanogpt"}) EXPECT_TRUE(is_known_preset(n)) << n;
  EXPECT_FALSE(is_known_preset("nano"));
  for (auto n : {"bpe", "chars", "bytes", "unigram"}) EXPECT_TRUE(is_known_tokenizer(n)) << n;
  EXPECT_FALSE(is_known_tokenizer("char"));
}

// Regression: make_tokenizer returned bytes for preset tinygpt before looking at --tokenizer, and
// any unknown --tokenizer silently trained BPE.
TEST(Jobs, TokenizerSelection) {
  const auto data = write_corpus();
  const auto out = scratch_dir() / "tok_sel.gyre.json";
  gyre::app::CharLMOpts o;
  o.data = data;
  o.preset = "tinygpt";
  o.holdout = 0.0;

  // default tokenizer on tinygpt: byte-level (V=256)
  ASSERT_TRUE(gyre::app::run_tok_train(o, out, nullptr));
  auto t0 = gyre::Tokenizer::load(out);
  ASSERT_TRUE(t0);
  EXPECT_EQ((*t0)->vocab_size(), 256);

  // explicit --tokenizer chars wins: only the corpus alphabet
  o.tokenizer = "chars";
  o.tokenizer_set = true;
  ASSERT_TRUE(gyre::app::run_tok_train(o, out, nullptr));
  auto t1 = gyre::Tokenizer::load(out);
  ASSERT_TRUE(t1);
  EXPECT_LT((*t1)->vocab_size(), 100);

  // typo is an error, not a silent BPE
  o.tokenizer = "char";
  auto bad = gyre::app::run_tok_train(o, out, nullptr);
  ASSERT_FALSE(bad);
  EXPECT_NE(bad.error().message.find("unknown tokenizer"), std::string::npos) << bad.error().message;
}
