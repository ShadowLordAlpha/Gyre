// App job regressions (apps/common/jobs.cpp).

#include "jobs.hpp"

#include "gyre/checkpoint.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path scratch_dir() {
  auto d = std::filesystem::temp_directory_path() / "gyre_jobs_test";
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
