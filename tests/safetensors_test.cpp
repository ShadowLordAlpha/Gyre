#include "gyre/io/safetensors.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

void write_safetensors(const std::filesystem::path& path, const std::string& json,
                       std::span<const std::byte> data) {
  std::uint64_t n = json.size();
  std::ofstream o(path, std::ios::binary);
  o.write(reinterpret_cast<const char*>(&n), 8);
  o.write(json.data(), static_cast<std::streamsize>(json.size()));
  o.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

}  // namespace

TEST(Safetensors, ListAndLoadF32) {
  auto dir = std::filesystem::temp_directory_path() / "gyre_st";
  std::filesystem::create_directories(dir);
  auto path = dir / "two.safetensors";
  float a[] = {1.f, 2.f, 3.f, 4.f};
  std::uint8_t u[] = {9, 8};
  std::vector<std::byte> payload(sizeof(a) + sizeof(u));
  std::memcpy(payload.data(), a, sizeof(a));
  std::memcpy(payload.data() + sizeof(a), u, sizeof(u));
  const std::string json =
      "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
      "\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[16,18]}}";
  write_safetensors(path, json, payload);
  auto f = gyre::safetensors_open(path);
  ASSERT_TRUE(f) << f.error().message;
  ASSERT_EQ(f->tensors.size(), 2u);
  auto d = gyre::Device::cpu();
  auto ta = gyre::safetensors_load(*f, "a", *d, false);
  ASSERT_TRUE(ta) << ta.error().message;
  EXPECT_EQ(ta->dtype(), gyre::DType::f32);
  auto p = ta->host_span<float>();
  EXPECT_EQ((*p)[3], 4.f);
  auto tb = gyre::safetensors_load(*f, "b", *d, false);
  ASSERT_TRUE(tb);
  auto pb = tb->host_span<std::uint8_t>();
  EXPECT_EQ((*pb)[0], 9);
}

TEST(Safetensors, TruncatedHeader) {
  auto dir = std::filesystem::temp_directory_path() / "gyre_st";
  auto path = dir / "bad.safetensors";
  std::filesystem::create_directories(dir);
  std::ofstream o(path, std::ios::binary);
  std::uint64_t n = 100;
  o.write(reinterpret_cast<const char*>(&n), 8);
  o.write("short", 5);
  o.close();
  auto f = gyre::safetensors_open(path);
  ASSERT_FALSE(f);
}

TEST(Safetensors, ConcatLastDim) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {2, 2};
  float a[] = {1, 2, 3, 4};
  float b[] = {5, 6, 7, 8};
  auto ta = gyre::Tensor::from_host(std::as_bytes(std::span(a)), sh, gyre::DType::f32, *d);
  auto tb = gyre::Tensor::from_host(std::as_bytes(std::span(b)), sh, gyre::DType::f32, *d);
  gyre::Tensor shards[] = {std::move(*ta), std::move(*tb)};
  auto c = gyre::concat_tp(shards, 1);
  ASSERT_TRUE(c) << c.error().message;
  EXPECT_EQ(c->shape()[1], 4);
  auto p = c->host_span<float>();
  EXPECT_EQ((*p)[0], 1.f);
  EXPECT_EQ((*p)[2], 5.f);
  EXPECT_EQ((*p)[4], 3.f);
}

TEST(Safetensors, LoadGrokNormIfPresent) {
#ifdef GYRE_SOURCE_DIR
  auto path = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "grok2" /
              "pytorch_model-00002-TP-common.safetensors";
#else
  auto path = std::filesystem::path("data/grok2/pytorch_model-00002-TP-common.safetensors");
#endif
  if (!std::filesystem::exists(path)) GTEST_SKIP() << path.string();
  auto f = gyre::safetensors_open(path);
  ASSERT_TRUE(f) << f.error().message;
  ASSERT_EQ(f->tensors.size(), 1u);
  EXPECT_EQ(f->tensors[0].name, "model.norm.weight");
  EXPECT_EQ(f->tensors[0].dtype, gyre::DType::bf16);
  auto d = gyre::Device::cpu();
  auto t = gyre::safetensors_load(*f, "model.norm.weight", *d, true);
  ASSERT_TRUE(t) << t.error().message;
  EXPECT_EQ(t->numel(), 8192);
  auto f32 = t->to_f32();
  ASSERT_TRUE(f32);
  auto p = f32->host_span<float>();
  EXPECT_TRUE(std::isfinite((*p)[0]));
}
