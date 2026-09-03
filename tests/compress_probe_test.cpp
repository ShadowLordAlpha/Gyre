#include "gyre/io/compress_probe.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(CompressProbe, OddBitsExceptionShrinks) {
  std::vector<std::uint16_t> w(4096, 0xAAAA);  // repeating 10
  w[10] = 0xAAAB;
  w[100] = 0x0000;
  auto bytes = std::as_bytes(std::span(w.data(), w.size()));
  auto r = gyre::probe_bytes(bytes, "plant.odd");
  EXPECT_LT(r.plane_pack_bytes, r.probed_bytes);
  EXPECT_LT(r.freq16_exc_bytes, r.probed_bytes);
  EXPECT_NE(r.best_rule, "identity");
}

TEST(CompressProbe, TwoLayerPlantStackBeatsIdentity) {
  std::vector<std::uint16_t> w(2048);
  for (std::size_t i = 0; i < w.size(); ++i) {
    // constant exponent 0x40, mantissa alternating
    w[i] = static_cast<std::uint16_t>(0x4000 | (i & 1));
  }
  auto r = gyre::probe_bytes(std::as_bytes(std::span(w.data(), w.size())), "plant.exp");
  EXPECT_EQ(r.unique_exponents, 1);
  EXPECT_LT(r.stack_bytes, r.probed_bytes);
  EXPECT_LT(r.exp_ent_bytes + r.sign_ent_bytes, r.probed_bytes);
}

TEST(CompressProbe, DifferentMasksDifferentRules) {
  std::vector<std::uint16_t> a(512, 0xFFFF);
  std::vector<std::uint16_t> b(512);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint16_t>(i * 251u);
  auto ra = gyre::probe_bytes(std::as_bytes(std::span(a.data(), a.size())), "mlp.gate_proj");
  auto rb = gyre::probe_bytes(std::as_bytes(std::span(b.data(), b.size())), "attn.q_proj");
  EXPECT_EQ(ra.family, "mlp");
  EXPECT_EQ(rb.family, "attn");
  EXPECT_NE(ra.best_rule, rb.best_rule);
}

TEST(CompressProbe, PrngNotWorseThanRawPlusHeader) {
  std::vector<std::byte> b(1024);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::byte>((i * 17 + 31) & 0xFF);
  auto r = gyre::probe_bytes(b, "rand");
  EXPECT_LE(r.stack_bytes, r.probed_bytes + 32);
  EXPECT_GT(r.entropy_bpb, 6.0);
}

TEST(CompressProbe, JsonSchema) {
  std::vector<std::byte> b(16, std::byte{0});
  auto r = gyre::probe_bytes(b, "model.norm.weight");
  r.file = "x.safetensors";
  auto j = gyre::compress_probe_json(std::span<const gyre::CompressProbeRow>(&r, 1));
  EXPECT_NE(j.find("\"family\":\"norm\""), std::string::npos);
  EXPECT_NE(j.find("\"best_rule\""), std::string::npos);
  EXPECT_NE(j.find("\"plane_pack_bytes\""), std::string::npos);
  EXPECT_NE(j.find("\"stack_bytes\""), std::string::npos);
  EXPECT_NE(j.find("\"packed_bytes\""), std::string::npos);
  EXPECT_NE(j.find("\"packed_codec\""), std::string::npos);
  EXPECT_NE(j.find("\"estimate_bytes\""), std::string::npos);
  EXPECT_NE(j.find("\"target_ratio\":0.5"), std::string::npos);
}

TEST(CompressProbe, GrokNormIfPresent) {
#ifdef GYRE_SOURCE_DIR
  auto dir = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "grok2";
#else
  auto dir = std::filesystem::path("data/grok2");
#endif
  if (!std::filesystem::exists(dir / "pytorch_model-00002-TP-common.safetensors")) {
    GTEST_SKIP() << dir.string();
  }
  gyre::CompressProbeOpts o;
  o.max_tensors = 1;
  o.file_substr = "00002";
  o.chunk_bytes = 1 << 16;
  auto rows = gyre::compress_probe_dir(dir, o);
  ASSERT_TRUE(rows) << rows.error().message;
  ASSERT_EQ(rows->size(), 1u);
  EXPECT_EQ((*rows)[0].family, "norm");
  EXPECT_EQ((*rows)[0].raw_bytes, 16384u);
}
