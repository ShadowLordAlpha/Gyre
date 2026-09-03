#include "gyre/checkpoint.hpp"
#include "gyre/nn/layers.hpp"

#include <gtest/gtest.h>

#include <filesystem>

TEST(Ckpt, RoundTrip) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  auto lin = gyre::Linear::create(3, 2, *d, rng);
  ASSERT_TRUE(lin);
  auto opt = gyre::Adam::create(lin->parameters());
  ASSERT_TRUE(opt);
  gyre::CheckpointMeta meta{3, 7, "{\"arch\":\"test\"}"};
  auto path = std::filesystem::temp_directory_path() / "gyre-test.gyre";
  ASSERT_TRUE(gyre::save_gyre1(path, lin->parameters(), &*opt, meta));
  auto lin2 = gyre::Linear::create(3, 2, *d, rng);
  auto opt2 = gyre::Adam::create(lin2->parameters());
  gyre::CheckpointMeta meta2;
  ASSERT_TRUE(gyre::load_gyre1(path, lin2->parameters(), &*opt2, meta2));
  EXPECT_EQ(meta2.train_step, 7u);
  auto a = lin->parameters()[0].value.host_span<float>();
  auto b = lin2->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(a && b);
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_EQ((*a)[i], (*b)[i]);
}
