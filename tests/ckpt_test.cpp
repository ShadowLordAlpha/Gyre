#include "gyre/checkpoint.hpp"
#include "gyre/nn/layers.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
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
