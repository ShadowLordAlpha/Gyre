#include "gyre/io/compress_probe.hpp"
#include "gyre/io/wpack.hpp"
#include "gyre/nn/grok.hpp"

#include <cstring>
#include <filesystem>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(Wpack, PlaneExcRoundTripPlant) {
  std::vector<std::uint16_t> w(2048, 0xAAAA);
  w[3] = 0xAAAB;
  w[9] = 0x0001;
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {2048};
  auto enc = gyre::wpack_encode("plant", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc) << enc.error().message;
  EXPECT_TRUE(enc->codec == gyre::WpackCodec::plane_exc ||
              enc->codec == gyre::WpackCodec::const_lane);
  EXPECT_LT(enc->payload.size(), raw.size());
  auto dec = gyre::wpack_decode(*enc);
  ASSERT_TRUE(dec);
  ASSERT_EQ(dec->size(), raw.size());
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, ExpAlphaConstantExponent) {
  // Two exponents, noisy mantissa: plane_exc stays ~raw (dense planes); exp_alpha drops 8→1 exp bits.
  std::vector<std::uint16_t> w(4096);
  for (std::size_t i = 0; i < w.size(); ++i) {
    const std::uint16_t exp = (i & 1) ? 0x40 : 0x3C;
    const std::uint16_t mant = static_cast<std::uint16_t>((i * 13u) & 0x7F);
    w[i] = static_cast<std::uint16_t>((exp << 7) | mant);
  }
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {4096};
  auto enc = gyre::wpack_encode("exp", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  EXPECT_EQ(enc->codec, gyre::WpackCodec::exp_alpha);
  EXPECT_LT(enc->payload.size(), raw.size());
  auto dec = gyre::wpack_decode(*enc);
  ASSERT_TRUE(dec);
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, ConstLaneHighByteStopsWhenPatternDies) {
  std::vector<std::uint16_t> w(1024);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint16_t>(0x4000 | (i & 0x7F));
  for (std::size_t i = 0; i < 80; ++i) w[i] = 0x1234;
  for (std::size_t i = 900; i < w.size(); ++i) w[i] = 0x55AA;
  w[200] = 0x41FF;
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {1024};
  auto enc = gyre::wpack_encode("lane", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  EXPECT_EQ(enc->codec, gyre::WpackCodec::const_lane);
  EXPECT_LT(enc->payload.size(), raw.size());
  auto& p = enc->payload;
  ASSERT_GE(p.size(), 24u);
  auto u8 = [&](std::size_t i) { return static_cast<unsigned>(p[i]); };
  auto u32 = [&](std::size_t i) {
    return static_cast<std::uint32_t>(u8(i) | (u8(i + 1) << 8) | (u8(i + 2) << 16) | (u8(i + 3) << 24));
  };
  EXPECT_EQ(u32(0), 2048u);
  EXPECT_EQ(u8(4), 2u);     // stride
  EXPECT_EQ(u8(5), 1u);     // high byte
  EXPECT_EQ(u8(6), 0x40u);
  const auto begin = u32(8);
  const auto len = u32(12);
  EXPECT_GE(begin, 70u);
  EXPECT_LE(begin, 90u);
  EXPECT_GE(begin + len, 880u);
  EXPECT_LE(begin + len, 910u);
  auto dec = gyre::wpack_decode(*enc);
  ASSERT_TRUE(dec);
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, LfsrPredPlantSmallerThanInnerAlone) {
  auto lfsr_next = [](std::uint16_t s) {
    const auto bit = static_cast<std::uint16_t>(s & 1u);
    s = static_cast<std::uint16_t>(s >> 1);
    if (bit) s = static_cast<std::uint16_t>(s ^ 0xB400u);
    return s ? s : std::uint16_t{1};
  };
  std::vector<std::uint16_t> w(512);
  std::uint16_t s = 7;
  for (auto& x : w) {
    x = s;
    s = lfsr_next(s);
  }
  w[10] ^= 1;
  w[200] ^= 2;
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {512};
  auto enc = gyre::wpack_encode("lfsr", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  EXPECT_EQ(enc->codec, gyre::WpackCodec::lfsr_pred);
  EXPECT_LT(enc->payload.size(), raw.size());
  auto dec = gyre::wpack_decode(*enc);
  ASSERT_TRUE(dec);
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, FairBitsStayIdentityNotIndexList) {
  std::vector<std::uint16_t> w(2048);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint16_t>(i * 7919u);
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {2048};
  auto enc = gyre::wpack_encode("rand", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  EXPECT_EQ(enc->codec, gyre::WpackCodec::identity);
  EXPECT_EQ(enc->payload.size(), raw.size());
}

TEST(Wpack, PrngStaysIdentity) {
  std::vector<std::uint16_t> w(512);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint16_t>(i * 7919u);
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {512};
  auto enc = gyre::wpack_encode("rand", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  EXPECT_EQ(enc->codec, gyre::WpackCodec::identity);
  auto dec = gyre::wpack_decode(*enc);
  ASSERT_TRUE(dec);
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, SaveLoadFile) {
  std::vector<std::uint16_t> w(64, 0xFFFF);
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {64};
  auto enc = gyre::wpack_encode("t", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  gyre::WpackFile f;
  f.source = "unit";
  f.tensors.push_back(std::move(*enc));
  auto path = std::filesystem::temp_directory_path() / "gyre_wpack" / "t.gyre.wpack";
  ASSERT_TRUE(gyre::wpack_save(path, f));
  auto g = gyre::wpack_load(path);
  ASSERT_TRUE(g) << g.error().message;
  ASSERT_EQ(g->tensors.size(), 1u);
  auto dec = gyre::wpack_decode(g->tensors[0]);
  ASSERT_TRUE(dec);
  EXPECT_EQ(std::memcmp(dec->data(), raw.data(), raw.size()), 0);
}

TEST(Wpack, MiniGrokStillRuns) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(2);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto g = m->generate({1}, 3, *d, nullptr, 0.f);
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_EQ(g->size(), 4u);
}

TEST(Wpack, PackGrokNormIfPresent) {
#ifdef GYRE_SOURCE_DIR
  auto path = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "grok2" /
              "pytorch_model-00002-TP-common.safetensors";
#else
  auto path = std::filesystem::path("data/grok2/pytorch_model-00002-TP-common.safetensors");
#endif
  if (!std::filesystem::exists(path)) GTEST_SKIP();
  auto st = gyre::safetensors_open(path);
  ASSERT_TRUE(st);
  auto pack = gyre::wpack_from_safetensors(*st, 1u << 20);
  ASSERT_TRUE(pack) << pack.error().message;
  ASSERT_EQ(pack->tensors.size(), 1u);
  EXPECT_EQ(pack->tensors[0].codec, gyre::WpackCodec::const_lane);
  EXPECT_LT(pack->tensors[0].payload.size(), pack->tensors[0].raw_bytes);
  auto dec = gyre::wpack_decode(pack->tensors[0]);
  ASSERT_TRUE(dec);
  auto raw = gyre::safetensors_read_prefix(*st, st->tensors[0],
                                            st->tensors[0].data_end - st->tensors[0].data_begin);
  ASSERT_TRUE(raw);
  ASSERT_EQ(dec->size(), raw->size());
  EXPECT_EQ(std::memcmp(dec->data(), raw->data(), raw->size()), 0);
}
