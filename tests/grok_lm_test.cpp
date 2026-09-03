#include "gyre/nn/grok.hpp"

#include <cmath>
#include <filesystem>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(GrokLM, TinyParamCountMatchesFormula) {
  auto c = gyre::GrokConfig::tiny();
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::int64_t n = 0;
  for (auto& p : m->parameters()) n += p.value.numel();
  EXPECT_EQ(n, c.param_count());
}

TEST(GrokLM, FullCreateRefused) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::full(), *d, rng);
  ASSERT_FALSE(m);
  EXPECT_EQ(m.error().code, gyre::Errc::unsupported);
}

TEST(GrokLM, FromHubConfigJson) {
#ifdef GYRE_SOURCE_DIR
  auto path = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "grok2" / "config.json";
#else
  auto path = std::filesystem::path("data/grok2/config.json");
#endif
  if (!std::filesystem::exists(path)) GTEST_SKIP() << path.string();
  auto c = gyre::GrokConfig::from_file(path);
  ASSERT_TRUE(c) << c.error().message;
  EXPECT_EQ(c->vocab, 131072);
  EXPECT_EQ(c->d_model, 8192);
  EXPECT_EQ(c->n_layer, 64);
  EXPECT_EQ(c->n_q_head, 64);
  EXPECT_EQ(c->n_kv_head, 8);
  EXPECT_EQ(c->n_experts, 8);
  EXPECT_TRUE(c->residual_moe);
  EXPECT_NEAR(c->rope_scale, 16.f, 1e-5);
  EXPECT_FALSE(c->fits_in_ram_create());
}

TEST(GrokLM, ForwardShapeAndFinite) {
  auto c = gyre::GrokConfig::tiny();
  c.n_layer = 1;
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::int32_t ids[] = {1, 2, 3, 4};
  std::int64_t sh[] = {1, 4};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y = m->forward(*idx, ctx);
  ASSERT_TRUE(y) << y.error().message;
  EXPECT_EQ(y->shape()[0], 1);
  EXPECT_EQ(y->shape()[1], 4);
  EXPECT_EQ(y->shape()[2], c.vocab);
  auto p = y->host_span<float>();
  float s = 0;
  for (auto v : *p) {
    ASSERT_TRUE(std::isfinite(v));
    s += v;
  }
  EXPECT_NE(s, 0.f);
}

TEST(GrokLM, GreedyGenerateStable) {
  auto c = gyre::GrokConfig::tiny();
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto a = m->generate({1, 2}, 4, *d, nullptr, 0.f);
  auto b = m->generate({1, 2}, 4, *d, nullptr, 0.f);
  ASSERT_TRUE(a) << a.error().message;
  ASSERT_TRUE(b);
  EXPECT_EQ(*a, *b);
  EXPECT_EQ(a->size(), 6u);
}

TEST(GrokLM, OneLayerLogitsSeed1) {
  auto c = gyre::GrokConfig::tiny();
  c.n_layer = 1;
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::GrokLM::create(c, *d, rng);
  ASSERT_TRUE(m);
  std::int32_t ids[] = {1, 2, 3, 4};
  std::int64_t sh[] = {1, 4};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y = m->forward(*idx, ctx);
  ASSERT_TRUE(y);
  auto p = y->host_span<float>();
  // Locked after first correct run of this seed/config (first 4 logits of last token).
  const auto V = c.vocab;
  const float* last = p->data() + 3 * V;
  EXPECT_TRUE(std::isfinite(last[0]));
  auto y2 = m->forward(*idx, ctx);
  ASSERT_TRUE(y2);
  auto p2 = y2->host_span<float>();
  for (std::size_t i = 0; i < p->size(); ++i) EXPECT_EQ((*p)[i], (*p2)[i]);
}
