#include "gyre/tensor.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(Mmap, FileViewMatchesHeap) {
  auto d = gyre::Device::cpu();
  float v[] = {1.f, 2.f, 3.f, 4.f};
  auto dir = std::filesystem::temp_directory_path() / "gyre_mmap";
  std::filesystem::create_directories(dir);
  auto path = dir / "four.f32";
  {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(v), sizeof(v));
  }
  auto st = gyre::Storage::mmap_file(path);
  ASSERT_TRUE(st) << st.error().message;
  std::int64_t sh[] = {4};
  auto t = gyre::Tensor::from_storage(*st, 0, sh, gyre::DType::f32, *d);
  ASSERT_TRUE(t) << t.error().message;
  auto p = t->host_span<float>();
  ASSERT_TRUE(p);
  EXPECT_EQ((*p)[0], 1.f);
  EXPECT_EQ((*p)[3], 4.f);
}

TEST(Mmap, Bf16ToF32) {
  auto d = gyre::Device::cpu();
  float src[] = {0.f, 1.f, -2.f, 0.5f};
  std::uint16_t packed[4];
  for (int i = 0; i < 4; ++i) {
    std::uint32_t u;
    std::memcpy(&u, &src[i], 4);
    packed[i] = static_cast<std::uint16_t>(u >> 16);
  }
  std::int64_t sh[] = {4};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(packed)), sh, gyre::DType::bf16, *d);
  ASSERT_TRUE(t) << t.error().message;
  auto f = t->to_f32();
  ASSERT_TRUE(f) << f.error().message;
  auto p = f->host_span<float>();
  EXPECT_NEAR((*p)[0], 0.f, 1e-5);
  EXPECT_NEAR((*p)[1], 1.f, 1e-3);
  EXPECT_NEAR((*p)[2], -2.f, 1e-3);
  EXPECT_NEAR((*p)[3], 0.5f, 1e-3);
}
