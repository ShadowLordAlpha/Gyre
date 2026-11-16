#include "gyre/io/compress_probe.hpp"
#include "gyre/io/wpack.hpp"
#include "gyre/nn/grok.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

namespace {

void le(std::vector<std::byte>& o, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) o.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
}

/// exp_alpha payload with all 256 exponents (header nalpha=0, ebits=8).
std::vector<std::byte> full_alphabet_payload(const std::vector<std::uint16_t>& w, int ebits = 8) {
  std::vector<std::byte> o;
  le(o, w.size(), 4);
  le(o, 0, 1);
  le(o, static_cast<std::uint64_t>(ebits), 1);
  for (int e = 0; e < 256; ++e) le(o, static_cast<std::uint64_t>(e), 1);
  std::uint8_t acc = 0;
  int nb = 0;
  auto put = [&](std::uint32_t v, int bits) {
    for (int b = 0; b < bits; ++b) {
      if (v & (1u << b)) acc = static_cast<std::uint8_t>(acc | (1u << nb));
      if (++nb == 8) {
        o.push_back(std::byte{acc});
        acc = 0;
        nb = 0;
      }
    }
  };
  for (auto x : w) {
    put((x >> 15) & 1u, 1);
    put((x >> 7) & 0xFFu, 8);
    put(x & 0x7Fu, 7);
  }
  if (nb) o.push_back(std::byte{acc});
  return o;
}

}  // namespace

// Regression: the alphabet loop used a uint8_t counter against 256 and never
// terminated (reading past the payload) for a full exponent alphabet.
TEST(Wpack, ExpAlphaFull256AlphabetDecodes) {
  std::vector<std::uint16_t> w(512);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::uint16_t>(((i & 1) << 15) | ((i & 0xFF) << 7) | ((i * 7) & 0x7F));
  }
  gyre::WpackTensor t;
  t.dtype = gyre::DType::bf16;
  t.codec = gyre::WpackCodec::exp_alpha;
  t.raw_bytes = w.size() * 2;
  t.payload = full_alphabet_payload(w);
  auto dec = gyre::wpack_decode(t);
  ASSERT_TRUE(dec) << dec.error().message;
  ASSERT_EQ(dec->size(), t.raw_bytes);
  EXPECT_EQ(std::memcmp(dec->data(), w.data(), dec->size()), 0);

  // ebits > 8 would shift by >= 32 in the bit reader.
  t.payload = full_alphabet_payload(w);
  t.payload[5] = std::byte{40};
  EXPECT_FALSE(gyre::wpack_decode(t));
  // A count the payload cannot hold must not allocate.
  t.payload = full_alphabet_payload(w);
  t.payload[0] = t.payload[1] = t.payload[2] = t.payload[3] = std::byte{0xff};
  t.raw_bytes = ~0ull;
  EXPECT_FALSE(gyre::wpack_decode(t));
}

TEST(Wpack, LoadRejectsTruncatedAndHostileFiles) {
  std::vector<std::uint16_t> w(256);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint16_t>(0x4000 | (i & 0x7F));
  auto raw = std::as_bytes(std::span(w.data(), w.size()));
  std::int64_t sh[] = {256};
  auto enc = gyre::wpack_encode("t", gyre::DType::bf16, sh, raw);
  ASSERT_TRUE(enc);
  gyre::WpackFile f;
  f.source = "unit";
  f.tensors.push_back(std::move(*enc));
  auto dir = std::filesystem::temp_directory_path() / "gyre_wpack_bad";
  auto good = dir / "good.gywp";
  ASSERT_TRUE(gyre::wpack_save(good, f));
  std::vector<char> bytes;
  {
    std::ifstream in(good, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  ASSERT_GT(bytes.size(), 100u);
  auto write = [&](const std::vector<char>& b) {
    auto p = dir / "bad.gywp";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(b.data(), static_cast<std::streamsize>(b.size()));
    out.close();
    return gyre::wpack_load(p);
  };
  ASSERT_TRUE(write(bytes));
  for (std::size_t n = 0; n < bytes.size(); ++n) {
    std::vector<char> t(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(n));
    EXPECT_FALSE(write(t)) << "len " << n;
  }
  // Header layout: magic(8) ver(4) srclen(2) "unit"(4) count(4) namelen(2) "t"(1) dtype ndim ...
  const std::size_t count_off = 8 + 4 + 2 + 4;
  const std::size_t dtype_off = count_off + 4 + 2 + 1;
  const std::size_t codec_off = dtype_off + 2 + 64;
  auto b = bytes;
  b[count_off] = b[count_off + 1] = b[count_off + 2] = b[count_off + 3] = static_cast<char>(0xff);
  EXPECT_FALSE(write(b));  // used to reserve 4G tensors
  b = bytes;
  b[12] = static_cast<char>(0xff);  // source length past the end
  b[13] = static_cast<char>(0xff);
  EXPECT_FALSE(write(b));
  b = bytes;
  b[dtype_off] = static_cast<char>(99);
  EXPECT_FALSE(write(b));
  b = bytes;
  b[dtype_off + 1] = static_cast<char>(200);  // ndim
  EXPECT_FALSE(write(b));
  b = bytes;
  b[codec_off] = static_cast<char>(77);
  EXPECT_FALSE(write(b));
  b = bytes;
  for (int k = 0; k < 8; ++k) b[codec_off + 9 + static_cast<std::size_t>(k)] = static_cast<char>(0xff);
  EXPECT_FALSE(write(b));  // payload length wraps i + plen
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
