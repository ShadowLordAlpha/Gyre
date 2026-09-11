#include "gyre/checkpoint.hpp"
#include "gyre/data.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/train/connectome.hpp"
#include "gyre/train/loop.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

gyre::CharLMConfig tiny_cfg(std::int64_t vocab) {
  gyre::CharLMConfig c;
  c.vocab = vocab;
  c.block_size = 16;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 16;
  c.d_ff = 32;
  c.recency_alibi = false;
  return c;
}

}  // namespace

TEST(Connectome, PruneRejectsBadFraction) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto m = gyre::CharLM::create(tiny_cfg(20), *d, rng);
  ASSERT_TRUE(m);
  gyre::Rng r2(2);
  EXPECT_FALSE(gyre::prune_compact_charlm(*m, 0.f, false, r2, *d));
  EXPECT_FALSE(gyre::prune_compact_charlm(*m, 1.f, false, r2, *d));
  EXPECT_FALSE(gyre::prune_compact_charlm(*m, -0.2f, false, r2, *d));
}

TEST(Connectome, CompactShrinksDense) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  auto parent = gyre::CharLM::create(tiny_cfg(20), *d, rng);
  ASSERT_TRUE(parent) << parent.error().message;
  const auto n0 = gyre::param_count(*parent);
  gyre::Rng r2(4);
  auto child = gyre::prune_compact_charlm(*parent, 0.2f, false, r2, *d);
  ASSERT_TRUE(child) << child.error().message;
  EXPECT_EQ(child->config().n_head, 2);
  EXPECT_LT(child->config().d_model, parent->config().d_model);
  EXPECT_LT(child->config().d_ff, parent->config().d_ff);
  EXPECT_EQ(child->config().d_model % child->config().n_head, 0);
  const auto n1 = gyre::param_count(*child);
  EXPECT_LT(n1, n0);

  bool any_2d = false;
  auto names = child->param_names();
  auto ps = child->parameters();
  for (std::size_t i = 0; i < ps.size(); ++i) {
    if (ps[i].value.rank() < 2) continue;
    any_2d = true;
    auto h = ps[i].value.host_span<float>();
    ASSERT_TRUE(h);
    double acc = 0;
    for (auto v : *h) acc += std::fabs(static_cast<double>(v));
    const float scale = static_cast<float>(acc / static_cast<double>(h->size()));
    for (auto v : *h) EXPECT_NEAR(std::fabs(v), scale, 1e-5f);
  }
  EXPECT_TRUE(any_2d);

  gyre::Rng r3(5);
  auto g2 = gyre::prune_compact_charlm(*child, 0.2f, false, r3, *d);
  ASSERT_TRUE(g2) << g2.error().message;
  EXPECT_LT(gyre::param_count(*g2), n1);
  EXPECT_LE(g2->config().d_model, child->config().d_model);
}

TEST(Connectome, ShuffleKeepsSizeDiffersWeights) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(6);
  auto parent = gyre::CharLM::create(tiny_cfg(20), *d, rng);
  ASSERT_TRUE(parent);
  gyre::Rng a(7), b(8);
  auto evolved = gyre::prune_compact_charlm(*parent, 0.2f, false, a, *d);
  auto shuffled = gyre::prune_compact_charlm(*parent, 0.2f, true, b, *d);
  ASSERT_TRUE(evolved);
  ASSERT_TRUE(shuffled);
  EXPECT_EQ(evolved->config().d_model, shuffled->config().d_model);
  EXPECT_EQ(evolved->config().d_ff, shuffled->config().d_ff);
  EXPECT_EQ(gyre::param_count(*evolved), gyre::param_count(*shuffled));
  auto we = evolved->parameters()[0].value.host_span<float>();
  auto ws = shuffled->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(we && ws);
  bool differ = we->size() != ws->size();
  for (std::size_t i = 0; i < we->size() && i < ws->size(); ++i) {
    if ((*we)[i] != (*ws)[i]) differ = true;
  }
  EXPECT_TRUE(differ);
}

TEST(Connectome, RoundTripSmallerCkpt) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(9);
  auto parent = gyre::CharLM::create(tiny_cfg(20), *d, rng);
  ASSERT_TRUE(parent);
  const auto parent_bytes = [&] {
    std::uint64_t n = 0;
    for (auto& p : parent->parameters()) n += static_cast<std::uint64_t>(p.value.nbytes());
    return n;
  }();
  gyre::Rng r2(10);
  auto child = gyre::prune_compact_charlm(*parent, 0.2f, false, r2, *d);
  ASSERT_TRUE(child);
  gyre::GyreDoc doc;
  doc.arch = "char-lm";
  doc.config_json = "{\"d_model\":" + std::to_string(child->config().d_model) +
                    ",\"d_ff\":" + std::to_string(child->config().d_ff) +
                    ",\"n_layer\":1,\"n_head\":2,\"block_size\":16,\"vocab_size\":20}";
  auto path = std::filesystem::temp_directory_path() / "gyre-connectome-child.gyre";
  gyre::CheckpointMeta meta{1, 3, doc.to_json(), child->param_names()};
  ASSERT_TRUE(gyre::save_gyre1(path, child->parameters(), nullptr, meta));
  const auto child_bytes = std::filesystem::file_size(path);
  EXPECT_LT(child_bytes, parent_bytes + 4096);

  auto cfg = child->config();
  gyre::Rng r3(11);
  auto loaded = gyre::CharLM::create(cfg, *d, r3);
  ASSERT_TRUE(loaded);
  gyre::CheckpointMeta meta2;
  ASSERT_TRUE(gyre::load_gyre1(path, loaded->parameters(), nullptr, meta2));
  auto a = child->parameters()[0].value.host_span<float>();
  auto b = loaded->parameters()[0].value.host_span<float>();
  ASSERT_TRUE(a && b);
  ASSERT_EQ(a->size(), b->size());
  for (std::size_t i = 0; i < a->size(); ++i) EXPECT_EQ((*a)[i], (*b)[i]);
}

TEST(Connectome, ResumeContinuesStep) {
  auto d = gyre::Device::cpu();
  const std::string text = "hello hello hello hello hello hello hello hello ";
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids);
  auto data = gyre::CharDataset::from_ids(*ids, *d);
  ASSERT_TRUE(data);
  gyre::Rng rng(12);
  auto cfg = tiny_cfg((*tok)->vocab_size());
  auto m = gyre::CharLM::create(cfg, *d, rng);
  ASSERT_TRUE(m);
  auto opt = gyre::Adam::create(m->parameters(), 1e-3f);
  ASSERT_TRUE(opt);
  gyre::TrainConfig tc;
  tc.steps = 2;
  tc.batch = 2;
  tc.block = 16;
  tc.lr = 1e-3f;
  tc.log_every = 0;
  tc.ckpt_every = 0;
  tc.adam = &*opt;
  gyre::TrainLoop loop;
  ASSERT_TRUE(loop.run(*m, *data, tc, *d, {}));
  EXPECT_EQ(opt->t, 2u);
  tc.start_step = 2;
  tc.steps = 2;
  ASSERT_TRUE(loop.run(*m, *data, tc, *d, {}));
  EXPECT_EQ(opt->t, 4u);
}

TEST(Connectome, CompactThenTrainFinite) {
  auto d = gyre::Device::cpu();
  const std::string text = "hello hello hello hello hello hello hello hello ";
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids);
  auto data = gyre::CharDataset::from_ids(*ids, *d);
  ASSERT_TRUE(data);
  gyre::Rng rng(13);
  auto cfg = tiny_cfg((*tok)->vocab_size());
  auto m = gyre::CharLM::create(cfg, *d, rng);
  ASSERT_TRUE(m);
  gyre::TrainConfig tc;
  tc.steps = 4;
  tc.batch = 2;
  tc.block = 16;
  tc.lr = 1e-3f;
  tc.log_every = 0;
  gyre::TrainLoop loop;
  ASSERT_TRUE(loop.run(*m, *data, tc, *d, {}));
  gyre::Rng r2(14);
  auto child = gyre::prune_compact_charlm(*m, 0.2f, false, r2, *d);
  ASSERT_TRUE(child) << child.error().message;
  float last = 0;
  ASSERT_TRUE(loop.run(*child, *data, tc, *d, [&](const gyre::Metrics& met) { last = met.loss; }));
  EXPECT_TRUE(std::isfinite(last));
}
