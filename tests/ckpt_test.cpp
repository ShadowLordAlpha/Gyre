#include "gyre/checkpoint.hpp"
#include "gyre/nn/layers.hpp"

#include <gtest/gtest.h>

#include "gyre/io/codec.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

TEST(Ckpt, RoundTrip) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  auto lin = gyre::Linear::create(3, 2, *d, rng);
  ASSERT_TRUE(lin);
  auto opt = gyre::Adam::create(lin->parameters());
  ASSERT_TRUE(opt);
  gyre::CheckpointMeta meta{3, 7, "{\"arch\":\"linear\"}"};
  auto path = std::filesystem::temp_directory_path() / "gyre-test.gyre";
  ASSERT_TRUE(gyre::save_gyre1(path, lin->parameters(), &*opt, meta));
  auto lin2 = gyre::Linear::create(3, 2, *d, rng);
  auto opt2 = gyre::Adam::create(lin2->parameters());
  gyre::CheckpointMeta meta2;
  ASSERT_TRUE(gyre::load_gyre1(path, lin2->parameters(), &*opt2, meta2));
  EXPECT_EQ(meta2.train_step, 7u);
  auto peek = gyre::peek_gyre(path);
  ASSERT_TRUE(peek) << peek.error().message;
  EXPECT_EQ(peek->arch, "linear");
  auto a = lin->parameters()[0].value.host_span<float>();
  auto b = lin2->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(a && b);
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_EQ((*a)[i], (*b)[i]);
}

TEST(Ckpt, JsonTwinAndNamedLoad) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(4);
  auto lin = gyre::Linear::create(2, 2, *d, rng);
  ASSERT_TRUE(lin);
  gyre::GyreDoc doc;
  doc.arch = "linear";
  doc.config_json = "{\"in\":2,\"out\":2}";
  doc.train_step = 11;
  std::vector<std::string> names{"W", "b"};
  auto dir = std::filesystem::temp_directory_path() / "gyre-json-twin";
  std::filesystem::create_directories(dir);
  auto bin = dir / "m.gyre";
  auto js = dir / "m.gyre.json";
  ASSERT_TRUE(gyre::save_gyre(bin, lin->parameters(), nullptr, doc, names));
  ASSERT_TRUE(gyre::save_gyre_json(js, lin->parameters(), nullptr, doc, names, true));

  auto opened = gyre::GyreFile::open(bin);
  ASSERT_TRUE(opened) << opened.error().message;
  EXPECT_EQ(opened->doc().tensors.size(), 2u);
  EXPECT_EQ(opened->doc().tensors[0].name, "W");
  auto w = opened->load_tensor("W", *d);
  ASSERT_TRUE(w) << w.error().message;
  EXPECT_EQ(w->shape()[0], 2);

  auto from_json = gyre::GyreFile::open(js);
  ASSERT_TRUE(from_json) << from_json.error().message;
  auto lin3 = gyre::Linear::create(2, 2, *d, rng);
  ASSERT_TRUE(from_json->load_params(lin3->parameters()));
  auto a = lin->parameters()[0].value.host_span<float>();
  auto b = lin3->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(a && b);
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_NEAR((*a)[i], (*b)[i], 1e-5);
}

TEST(Ckpt, AlpRoundTrip) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(6);
  auto lin = gyre::Linear::create(8, 4, *d, rng);
  ASSERT_TRUE(lin);
  auto hb = lin->parameters()[0].value.host_bytes();
  ASSERT_TRUE(hb);
  auto enc = gyre::gyre_compress(gyre::GyreCodec::alp, *hb, gyre::DType::f32,
                                 lin->parameters()[0].value.shape());
  ASSERT_TRUE(enc) << enc.error().message;
  auto dec = gyre::gyre_decompress(gyre::GyreCodec::alp, *enc, gyre::DType::f32,
                                   lin->parameters()[0].value.shape(), hb->size());
  ASSERT_TRUE(dec) << dec.error().message;
  ASSERT_EQ(dec->size(), hb->size());
  EXPECT_EQ(0, std::memcmp(dec->data(), hb->data(), hb->size()));
  gyre::GyreDoc doc;
  doc.arch = "linear";
  auto path = std::filesystem::temp_directory_path() / "gyre-alp.gyre";
  ASSERT_TRUE(gyre::save_gyre(path, lin->parameters(), nullptr, doc, {}, gyre::GyreCodec::alp));
  auto opened = gyre::GyreFile::open(path);
  ASSERT_TRUE(opened) << opened.error().message;
  EXPECT_EQ(opened->doc().tensors[0].codec, gyre::GyreCodec::alp);
  auto lin2 = gyre::Linear::create(8, 4, *d, rng);
  auto ld = opened->load_params(lin2->parameters());
  ASSERT_TRUE(ld) << ld.error().message;
  auto a = lin->parameters()[0].value.host_span<float>();
  auto b = lin2->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(a && b);
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_EQ((*a)[i], (*b)[i]);
}

TEST(Ckpt, PeekDoesNotNeedFullRead) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(5);
  auto lin = gyre::Linear::create(4, 3, *d, rng);
  ASSERT_TRUE(lin);
  gyre::GyreDoc doc;
  doc.arch = "linear";
  doc.rng_seed = 5;
  doc.train_step = 99;
  auto path = std::filesystem::temp_directory_path() / "gyre-peek.gyre";
  ASSERT_TRUE(gyre::save_gyre(path, lin->parameters(), nullptr, doc, {}));
  auto peek = gyre::peek_gyre(path);
  ASSERT_TRUE(peek) << peek.error().message;
  EXPECT_EQ(peek->train_step, 99u);
  EXPECT_EQ(peek->arch, "linear");
}

// ---- Malformed-file regression tests -------------------------------------
// Each of these used to read or write out of bounds (or silently return
// garbage). They must now fail cleanly with an error.

namespace {

std::uint32_t test_crc32c(const std::byte* p, std::size_t n) {
  std::uint32_t crc = ~0u;
  for (std::size_t k = 0; k < n; ++k) {
    crc ^= static_cast<std::uint8_t>(p[k]);
    for (int i = 0; i < 8; ++i) crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
  }
  return ~crc;
}

void put_le(std::vector<std::byte>& b, std::size_t off, std::uint64_t v, int n) {
  if (b.size() < off + static_cast<std::size_t>(n)) b.resize(off + static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    b[off + static_cast<std::size_t>(i)] = static_cast<std::byte>((v >> (8 * i)) & 0xff);
  }
}

void seal_header(std::vector<std::byte>& b) { put_le(b, 60, test_crc32c(b.data(), 60), 4); }

std::filesystem::path write_file(const char* name, const std::vector<std::byte>& b) {
  auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  return path;
}

constexpr std::uint64_t kKeep = ~0ull;

/// Hand-built v2 file: header, JSON at 64, payload at align64(64 + json).
std::vector<std::byte> make_v2(const std::string& json, const std::vector<std::byte>& payload,
                               std::uint64_t json_off = 64, std::uint64_t json_len = kKeep,
                               std::uint64_t payload_off = kKeep, std::uint32_t flags = 0) {
  std::vector<std::byte> b(64);
  std::memcpy(b.data(), "GYRE1\0\0\0", 8);
  const std::uint64_t real_poff = (64 + json.size() + 63) & ~std::uint64_t{63};
  b.resize(static_cast<std::size_t>(real_poff), std::byte{0});
  std::memcpy(b.data() + 64, json.data(), json.size());
  b.insert(b.end(), payload.begin(), payload.end());
  put_le(b, 8, 2, 4);
  put_le(b, 12, flags, 4);
  put_le(b, 16, 1, 8);
  put_le(b, 24, json_len == kKeep ? json.size() : json_len, 8);
  put_le(b, 32, json_off, 8);
  put_le(b, 40, payload_off == kKeep ? real_poff : payload_off, 8);
  put_le(b, 48, payload.size(), 8);
  put_le(b, 58, 1, 2);
  seal_header(b);
  return b;
}

std::string one_tensor_json(const std::string& shape, std::uint64_t offset, std::uint64_t nbytes,
                            std::uint64_t packed, const char* codec = "identity") {
  return std::string("{\"gyre\":\"model\",\"version\":1,\"arch\":\"t\",\"config\":{},\"tensors\":[{") +
         "\"name\":\"w\",\"dtype\":\"f32\",\"shape\":" + shape + ",\"role\":\"weight\",\"codec\":\"" +
         codec + "\",\"offset\":" + std::to_string(offset) + ",\"nbytes\":" + std::to_string(nbytes) +
         ",\"packed_bytes\":" + std::to_string(packed) + "}]}";
}

std::vector<std::byte> alp_blob(const std::vector<float>& v) {
  std::int64_t sh[] = {static_cast<std::int64_t>(v.size())};
  auto enc = gyre::gyre_compress(gyre::GyreCodec::alp,
                                 std::as_bytes(std::span<const float>(v.data(), v.size())),
                                 gyre::DType::f32, sh);
  EXPECT_TRUE(enc);
  return enc ? *enc : std::vector<std::byte>{};
}

/// Hand-built v1 file: header, name heap "w", one 80-byte descriptor, payload, trailer.
std::vector<std::byte> make_v1(std::uint8_t ndim, std::uint64_t trailer_off_override = kKeep,
                               std::uint32_t flags = 0, std::uint64_t payload_off_override = kKeep) {
  std::vector<std::byte> b(64);
  std::memcpy(b.data(), "GYRE1\0\0\0", 8);
  b.push_back(std::byte{'w'});
  b.resize(128, std::byte{0});  // desc_off = align64(64 + 1)
  std::vector<std::byte> d(80, std::byte{0});
  put_le(d, 4, 1, 2);           // name_off 0, name_len 1
  d[6] = std::byte{1};          // f32
  d[7] = std::byte{ndim};
  put_le(d, 8, 4, 8);           // dim0 = 4
  b.insert(b.end(), d.begin(), d.end());
  const std::uint64_t payload_off = 256;
  b.resize(256 + 64, std::byte{0});
  const std::uint64_t trailer_off = b.size();
  std::vector<std::byte> tr(24, std::byte{0});
  put_le(tr, 0, test_crc32c(b.data() + payload_off, b.size() - payload_off), 4);
  b.insert(b.end(), tr.begin(), tr.end());  // json length 0
  put_le(b, 8, 1, 4);
  put_le(b, 12, flags, 4);
  put_le(b, 16, 1, 8);
  put_le(b, 24, 1, 8);
  put_le(b, 32, payload_off_override == kKeep ? payload_off : payload_off_override, 8);
  put_le(b, 40, trailer_off_override == kKeep ? trailer_off : trailer_off_override, 8);
  put_le(b, 52, 1, 4);
  seal_header(b);
  return b;
}

}  // namespace

TEST(CkptMalformed, AlpRejectsBadFieldsAndTruncation) {
  std::vector<float> v(300);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 17) * 0.25f;
  v[5] = std::numeric_limits<float>::infinity();  // force an exception record
  auto blob = alp_blob(v);
  ASSERT_GT(blob.size(), 40u);
  std::int64_t sh[] = {300};
  const auto raw = v.size() * 4;
  ASSERT_TRUE(gyre::gyre_decompress(gyre::GyreCodec::alp, blob, gyre::DType::f32, sh, raw));
  auto bad = [&](const std::vector<std::byte>& b) {
    return !gyre::gyre_decompress(gyre::GyreCodec::alp, b, gyre::DType::f32, sh, raw);
  };
  // First chunk header starts at byte 12: e f cn(u16) nexc(u16) min(u64) bits(u8).
  auto b = blob;
  b[12] = std::byte{200};  // exponent past the pow10 table
  EXPECT_TRUE(bad(b));
  b = blob;
  b[12] = std::byte{1};
  b[13] = std::byte{9};  // f > e
  EXPECT_TRUE(bad(b));
  b = blob;
  put_le(b, 14, 0xffff, 2);  // chunk longer than the remaining output (heap overflow before)
  EXPECT_TRUE(bad(b));
  b = blob;
  put_le(b, 14, 0, 2);  // empty chunk would never advance
  EXPECT_TRUE(bad(b));
  b = blob;
  b[26] = std::byte{200};  // bit width > 64
  EXPECT_TRUE(bad(b));
  b = blob;
  put_le(b, 16, 0xffff, 2);  // more exceptions than the chunk holds
  EXPECT_TRUE(bad(b));
  for (std::size_t n = 0; n < blob.size(); ++n) {
    std::vector<std::byte> t(blob.begin(), blob.begin() + static_cast<std::ptrdiff_t>(n));
    EXPECT_TRUE(bad(t)) << "len " << n;
  }
  // Single-byte corruption may decode to other values but must never crash.
  for (std::size_t k = 12; k < blob.size(); ++k) {
    auto c = blob;
    c[k] ^= std::byte{0x5a};
    (void)gyre::gyre_decompress(gyre::GyreCodec::alp, c, gyre::DType::f32, sh, raw);
  }
}

TEST(CkptMalformed, AlpBlobExactlyRawSizeIsDecoded) {
  // An ALP blob that is exactly nbytes long used to be aliased as raw floats.
  std::vector<float> v(16);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i) * 0.5f;
  auto blob = alp_blob(v);
  ASSERT_LT(blob.size(), 64u);
  blob.resize(64, std::byte{0});
  auto path =
      write_file("gyre-alp-rawsize.gyre", make_v2(one_tensor_json("[16]", 0, 64, 64, "alp"), blob));
  auto f = gyre::GyreFile::open(path);
  ASSERT_TRUE(f) << f.error().message;
  auto t = f->load_tensor("w", *gyre::Device::cpu());
  ASSERT_TRUE(t) << t.error().message;
  auto s = t->host_span<float>();
  ASSERT_TRUE(s);
  for (std::size_t i = 0; i < v.size(); ++i) EXPECT_EQ((*s)[i], v[i]) << i;
}

TEST(CkptMalformed, V2HeaderRangesDoNotWrap) {
  std::vector<std::byte> payload(64, std::byte{0});
  const auto js = one_tensor_json("[16]", 0, 64, 64);
  ASSERT_TRUE(gyre::GyreFile::open(write_file("gyre-v2ok.gyre", make_v2(js, payload))));
  // json_off + json_len wraps to a small number.
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v2b1.gyre", make_v2(js, payload, ~0ull - 9, 20))));
  // Payload offset past the end of the file.
  EXPECT_FALSE(
      gyre::GyreFile::open(write_file("gyre-v2b2.gyre", make_v2(js, payload, 64, kKeep, 1ull << 40))));
  // CRC range wraps (flags & 2).
  auto b = make_v2(js, payload, 64, kKeep, kKeep, 2);
  put_le(b, 48, ~0ull - 2, 8);
  seal_header(b);
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v2b3.gyre", b)));
}

TEST(CkptMalformed, TensorDescriptorRangesAreChecked) {
  std::vector<std::byte> payload(64, std::byte{0});
  auto cpu = *gyre::Device::cpu();
  // offset near 2^64: start + packed used to wrap past the size check.
  auto f = gyre::GyreFile::open(
      write_file("gyre-d-off.gyre", make_v2(one_tensor_json("[4]", ~0ull - 15, 16, 16), payload)));
  ASSERT_TRUE(f) << f.error().message;
  EXPECT_FALSE(f->load_tensor("w", cpu));
  // nbytes disagrees with dtype*shape: a [1000] view over 16 bytes.
  EXPECT_FALSE(gyre::GyreFile::open(
      write_file("gyre-d-nb.gyre", make_v2(one_tensor_json("[1000]", 0, 16, 16), payload))));
  // Shape product overflows int64.
  EXPECT_FALSE(gyre::GyreFile::open(write_file(
      "gyre-d-ovf.gyre", make_v2(one_tensor_json("[4294967296,4294967296,4]", 0, 0, 0), payload))));
  // Rank > 8.
  EXPECT_FALSE(gyre::GyreFile::open(write_file(
      "gyre-d-rank.gyre", make_v2(one_tensor_json("[1,1,1,1,1,1,1,1,1]", 0, 4, 4), payload))));
  // Identity codec with packed_bytes != nbytes.
  EXPECT_FALSE(gyre::GyreFile::open(
      write_file("gyre-d-pk.gyre", make_v2(one_tensor_json("[4]", 0, 16, 8), payload))));
  // Payload slice past the end of the file.
  auto g = gyre::GyreFile::open(
      write_file("gyre-d-end.gyre", make_v2(one_tensor_json("[16]", 32, 64, 64), payload)));
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_FALSE(g->load_tensor("w", cpu));
}

TEST(CkptMalformed, V1DescriptorAndTrailerChecks) {
  auto ok = gyre::GyreFile::open(write_file("gyre-v1ok.gyre", make_v1(1)));
  ASSERT_TRUE(ok) << ok.error().message;
  auto t = ok->load_tensor("w", *gyre::Device::cpu());
  ASSERT_TRUE(t) << t.error().message;
  EXPECT_EQ(t->numel(), 4);
  // ndim 255 used to read dims far past the 80-byte descriptor.
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v1nd.gyre", make_v1(255))));
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v1nd9.gyre", make_v1(9))));
  // trailer_off + 24 wraps.
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v1tr.gyre", make_v1(1, ~0ull - 8))));
  // Trailer before the payload with a payload CRC: rel used to underflow.
  EXPECT_FALSE(gyre::GyreFile::open(write_file("gyre-v1rel.gyre", make_v1(1, kKeep, 2, 330))));
}

namespace {

std::filesystem::path write_text(const char* name, const std::string& s) {
  auto p = std::filesystem::temp_directory_path() / name;
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  o.write(s.data(), static_cast<std::streamsize>(s.size()));
  return p;
}

std::string twin_with(const std::string& tensor) {
  return R"({"gyre":"model","version":1,"arch":"x","config":{},"tensors":[)" + tensor + "]}";
}

}  // namespace

// Regression: JSON has no NaN, nlohmann writes it as null, and get<float>() on null threw out of
// GyreFile::open, so a model with one NaN weight saved a twin that could never be reloaded.
TEST(CkptMalformed, JsonTwinWithNaNWeightReloads) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(9);
  auto lin = gyre::Linear::create(2, 2, *d, rng);
  ASSERT_TRUE(lin);
  auto w = lin->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(w);
  (*w)[1] = std::numeric_limits<float>::quiet_NaN();
  gyre::GyreDoc doc;
  doc.arch = "linear";
  const auto js = std::filesystem::temp_directory_path() / "gyre-nan-twin.gyre.json";
  ASSERT_TRUE(gyre::save_gyre_json(js, lin->parameters(), nullptr, doc, {}, true));
  auto f = gyre::GyreFile::open(js);
  ASSERT_TRUE(f) << f.error().message;
  auto lin2 = gyre::Linear::create(2, 2, *d, rng);
  ASSERT_TRUE(f->load_params(lin2->parameters()));
  auto w2 = lin2->parameters()[0].value.host_span<float>();
  EXPECT_TRUE(std::isnan((*w2)[1]));
  EXPECT_EQ((*w2)[0], (*w)[0]);
}

// Regression: wrong-typed or out-of-range data elements threw nlohmann::type_error (or wrapped
// silently), and a huge shape was allocated before the data size was checked.
TEST(CkptMalformed, JsonTwinDataIsValidatedBeforeUse) {
  const std::string bad[] = {
      twin_with(R"({"name":"W","dtype":"f32","shape":[2],"data":[1,"x"]})"),
      twin_with(R"({"name":"W","dtype":"f32","shape":[2],"data":[1,[2]]})"),
      twin_with(R"({"name":"W","dtype":"f32","shape":[2],"data":"abc"})"),
      twin_with(R"({"name":"W","dtype":"i32","shape":[1],"data":[30000000000]})"),
      twin_with(R"({"name":"W","dtype":"u8","shape":[1],"data":[300]})"),
      twin_with(R"({"name":"W","dtype":"u8","shape":[1],"data":[-1]})"),
      // 1e9 x 1e9 floats: must be rejected from the data size, not allocated.
      twin_with(R"({"name":"W","dtype":"f32","shape":[1000000000,1000000000],"data":[1]})"),
  };
  int k = 0;
  for (const auto& s : bad) {
    const auto p = write_text(("gyre-bad-twin-" + std::to_string(k++) + ".gyre.json").c_str(), s);
    gyre::Result<gyre::GyreFile> f = std::unexpected(gyre::make_error(gyre::Errc::ok, ""));
    EXPECT_NO_THROW(f = gyre::GyreFile::open(p)) << s;
    EXPECT_FALSE(f) << s;
  }
  auto ok = gyre::GyreFile::open(
      write_text("gyre-ok-twin.gyre.json", twin_with(R"({"name":"W","dtype":"u8","shape":[2],"data":[0,255]})")));
  ASSERT_TRUE(ok) << ok.error().message;
}

// Regression: holdout() did an unguarded get<double>() and terminated on "holdout":"x".
TEST(CkptMalformed, HoldoutFallsBackOnBadValues) {
  gyre::GyreDoc d;
  d.config_json = R"({"holdout":"x"})";
  EXPECT_DOUBLE_EQ(d.holdout(), 0.1);
  d.config_json = R"({"holdout":-3})";
  EXPECT_DOUBLE_EQ(d.holdout(), 0.1);
  d.config_json = R"({"holdout":0.25})";
  EXPECT_DOUBLE_EQ(d.holdout(), 0.25);
  d.config_json = R"({"holdout":0})";
  EXPECT_DOUBLE_EQ(d.holdout(), 0.0);
  d.config_json = "[1]";
  EXPECT_DOUBLE_EQ(d.holdout(), 0.1);
}

// Regression: save_gyre truncated the destination first and wrote the real header last, so a
// failed save destroyed the previous checkpoint. Saves now go through "<path>.tmp" + rename.
TEST(Ckpt, SaveReplacesAtomicallyAndKeepsOldFileOnFailure) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(10);
  auto lin = gyre::Linear::create(2, 2, *d, rng);
  ASSERT_TRUE(lin);
  gyre::GyreDoc doc;
  doc.arch = "linear";
  const auto dir = std::filesystem::temp_directory_path() / "gyre-atomic-save";
  std::error_code ec0;
  std::filesystem::remove_all(dir, ec0);
  std::filesystem::create_directories(dir);
  const auto path = dir / "m.gyre";
  doc.train_step = 1;
  ASSERT_TRUE(gyre::save_gyre(path, lin->parameters(), nullptr, doc, {}));
  doc.train_step = 2;
  ASSERT_TRUE(gyre::save_gyre(path, lin->parameters(), nullptr, doc, {}));  // replaces an existing file
  EXPECT_EQ(gyre::peek_gyre(path)->train_step, 2u);
  EXPECT_FALSE(std::filesystem::exists(dir / "m.gyre.tmp"));
  // Make the temp file impossible to create: the save fails and the old checkpoint survives.
  std::filesystem::create_directories(dir / "m.gyre.tmp");
  doc.train_step = 3;
  EXPECT_FALSE(gyre::save_gyre(path, lin->parameters(), nullptr, doc, {}));
  EXPECT_FALSE(gyre::save_gyre_json(path, lin->parameters(), nullptr, doc, {}, true));
  auto peek = gyre::peek_gyre(path);
  ASSERT_TRUE(peek) << peek.error().message;
  EXPECT_EQ(peek->train_step, 2u);
  {
    auto f = gyre::GyreFile::open(path);  // maps the file: release it before cleanup
    ASSERT_TRUE(f) << f.error().message;
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
