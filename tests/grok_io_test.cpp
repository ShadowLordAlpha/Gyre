#include "gyre/data.hpp"
#include "gyre/nn/grok.hpp"
#include "gyre/nn/tokenize.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(GrokIO, SaveLoadGenerateMatches) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  auto c = gyre::GrokConfig::mini();
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto a = m->generate({1, 2}, 4, *d, nullptr, 0.f);
  ASSERT_TRUE(a);
  auto dir = std::filesystem::temp_directory_path() / "gyre_grok_io";
  ASSERT_TRUE(m->save_weights(dir)) << "save";
  auto m2 = gyre::GrokLM::load_weights(dir, *d);
  ASSERT_TRUE(m2) << m2.error().message;
  auto b = m2->generate({1, 2}, 4, *d, nullptr, 0.f);
  ASSERT_TRUE(b);
  EXPECT_EQ(*a, *b);
}

TEST(GrokIO, ShakespeareCharsPrompt) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(4);
#ifdef GYRE_SOURCE_DIR
  auto path = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "shakespeare.txt";
#else
  auto path = std::filesystem::path("data/shakespeare.txt");
#endif
  if (!std::filesystem::exists(path)) GTEST_SKIP();
  std::ifstream in(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  text.resize(std::min<std::size_t>(text.size(), 8000));
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  auto c = gyre::GrokConfig::mini();
  c.vocab = (*tok)->vocab_size();
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto ids = (*tok)->encode("To be");
  ASSERT_TRUE(ids);
  if (ids->empty()) GTEST_SKIP();
  auto gen = m->generate(*ids, 8, *d, nullptr, 0.f);
  ASSERT_TRUE(gen) << gen.error().message;
  EXPECT_GT(gen->size(), ids->size());
  auto s = (*tok)->decode(*gen);
  EXPECT_FALSE(s.empty());
}

TEST(GrokLora, RankZeroMatchesBase) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(5);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  ASSERT_TRUE(m);
  std::int32_t ids[] = {1, 2, 3};
  std::int64_t sh[] = {1, 3};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y0 = m->forward(*idx, ctx);
  ASSERT_TRUE(y0);
  gyre::Rng rng2(6);
  auto l = gyre::GrokLora::create(m->config(), 0, 4.f, *d, rng2);
  ASSERT_TRUE(l);
  ASSERT_TRUE(m->set_lora(std::move(*l)));
  auto y1 = m->forward(*idx, ctx);
  ASSERT_TRUE(y1);
  auto p0 = y0->host_span<float>();
  auto p1 = y1->host_span<float>();
  for (std::size_t i = 0; i < p0->size(); ++i) EXPECT_EQ((*p0)[i], (*p1)[i]);
}

TEST(GrokLora, ChangesLogitsLeavesW) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(7);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  ASSERT_TRUE(m);
  std::vector<float> wcopy;
  {
    auto& W = m->parameters()[4].value;  // first layer q_proj
    auto p = W.host_span<float>();
    wcopy.assign(p->begin(), p->end());
  }
  gyre::Rng rng2(8);
  auto l = gyre::GrokLora::create(m->config(), 2, 4.f, *d, rng2);
  ASSERT_TRUE(l);
  auto fillb = l->q[0].B.host_span<float>();
  for (auto& x : *fillb) x = 0.1f;
  ASSERT_TRUE(m->set_lora(std::move(*l)));
  std::int32_t ids[] = {1, 2, 3};
  std::int64_t sh[] = {1, 3};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y = m->forward(*idx, ctx);
  ASSERT_TRUE(y);
  auto& W = m->parameters()[4].value;
  auto p = W.host_span<float>();
  ASSERT_EQ(p->size(), wcopy.size());
  EXPECT_EQ(std::memcmp(p->data(), wcopy.data(), wcopy.size() * sizeof(float)), 0);
  gyre::Rng rng3(7);
  auto base = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng3);
  auto yb = base->forward(*idx, ctx);
  auto pa = y->host_span<float>();
  auto pb = yb->host_span<float>();
  float d2 = 0;
  for (std::size_t i = 0; i < pa->size(); ++i) {
    float e = (*pa)[i] - (*pb)[i];
    d2 += e * e;
  }
  EXPECT_GT(d2, 0.f);
}

TEST(GrokLora, SaveLoad) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(9);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  gyre::Rng rng2(10);
  auto l = gyre::GrokLora::create(m->config(), 2, 8.f, *d, rng2);
  ASSERT_TRUE(l);
  ASSERT_TRUE(m->set_lora(std::move(*l)));
  auto dir = std::filesystem::temp_directory_path() / "gyre_lora";
  ASSERT_TRUE(m->save_lora(dir));
  gyre::Rng rng3(9);
  auto m2 = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng3);
  ASSERT_TRUE(m2->load_lora(dir, *d));
  std::int32_t ids[] = {1, 2};
  std::int64_t sh[] = {1, 2};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto a = m->forward(*idx, ctx);
  auto b = m2->forward(*idx, ctx);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  auto pa = a->host_span<float>();
  auto pb = b->host_span<float>();
  for (std::size_t i = 0; i < pa->size(); ++i) EXPECT_NEAR((*pa)[i], (*pb)[i], 1e-5);
}
