#include "gyre/data.hpp"
#include "gyre/export/onnx.hpp"
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/layers.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/unigram.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/ops.hpp"
#include "gyre/train/loop.hpp"

#include <filesystem>
#include <memory>

#include <cmath>
#include <gtest/gtest.h>

TEST(NN, LinearForwardShape) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto lin = gyre::Linear::create(4, 3, *d, rng);
  ASSERT_TRUE(lin);
  std::int64_t sh[] = {2, 4};
  auto x = gyre::Tensor::zeros(sh, gyre::DType::f32, *d);
  gyre::ForwardCtx ctx;
  auto y = lin->forward(*x, ctx);
  ASSERT_TRUE(y);
  EXPECT_EQ(y->shape()[1], 3);
}

TEST(NN, SoftmaxLast) {
  auto d = gyre::Device::cpu();
  std::int64_t sh[] = {1, 3};
  float v[] = {1.f, 2.f, 3.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *d);
  auto s = gyre::softmax_last(*t);
  ASSERT_TRUE(s);
  auto p = s->host_span<float>();
  float sum = (*p)[0] + (*p)[1] + (*p)[2];
  EXPECT_NEAR(sum, 1.f, 1e-5);
}

TEST(NN, TransformerOverfitsTinyString) {
  auto d = gyre::Device::cpu();
  const std::string text = "hello hello hello hello hello hello hello hello ";
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids);
  auto data = gyre::CharDataset::from_ids(*ids, *d);
  ASSERT_TRUE(data);
  gyre::Rng rng(1);
  gyre::CharLMConfig c;
  c.vocab = (*tok)->vocab_size();
  c.block_size = 16;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 16;
  c.d_ff = 32;
  auto m = gyre::CharLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  gyre::TrainConfig tc;
  tc.steps = 40;
  tc.batch = 4;
  tc.block = 16;
  tc.lr = 1e-3f;
  tc.log_every = 0;
  float first = -1, last = -1;
  gyre::TrainLoop loop;
  auto r = loop.run(*m, *data, tc, *d, {});
  ASSERT_TRUE(r) << r.error().message;
  // One extra eval batch via another short run of 1 step capturing loss is awkward;
  // instead run 1-step loops around a cloned check: compare train metrics by wrapping.
  gyre::TrainConfig once;
  once.steps = 1;
  once.batch = 4;
  once.block = 16;
  once.lr = 0;
  once.log_every = 1;
  ASSERT_TRUE(loop.run(*m, *data, once, *d, [&](const gyre::Metrics& met) { last = met.loss; }));
  gyre::Rng rng0(1);
  auto m0 = gyre::CharLM::create(c, *d, rng0);
  ASSERT_TRUE(m0);
  ASSERT_TRUE(loop.run(*m0, *data, once, *d, [&](const gyre::Metrics& met) { first = met.loss; }));
  ASSERT_GT(first, 0.f);
  EXPECT_LT(last, first - 0.2f) << "first=" << first << " last=" << last;
}

TEST(NN, TransformerForward) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(2);
  gyre::CharLMConfig c;
  c.vocab = 8;
  c.block_size = 8;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 8;
  c.d_ff = 16;
  auto m = gyre::CharLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::int32_t ids[] = {1, 2, 3, 0};
  std::int64_t sh[] = {1, 4};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  auto y = m->forward(*x, ctx);
  ASSERT_TRUE(y) << y.error().message;
  EXPECT_EQ(y->shape()[2], 8);
}

TEST(NN, TransformerHiddenIsDModelAndBackward) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(3);
  gyre::CharLMConfig c;
  c.vocab = 8;
  c.block_size = 8;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 8;
  c.d_ff = 16;
  auto m = gyre::CharLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::int32_t ids[] = {1, 2, 3, 0};
  std::int64_t sh[] = {1, 4};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = true;
  auto h = m->hidden(*x, ctx);
  ASSERT_TRUE(h) << h.error().message;
  ASSERT_EQ(h->rank(), 3);
  EXPECT_EQ(h->shape()[0], 1);
  EXPECT_EQ(h->shape()[1], 4);
  EXPECT_EQ(h->shape()[2], 8);
  auto dh = gyre::Tensor::zeros(h->shape(), gyre::DType::f32, *d);
  ASSERT_TRUE(dh);
  auto span = dh->host_span<float>();
  ASSERT_TRUE(span);
  (*span)[span->size() - 1] = 1.f;
  auto rb = m->hidden_backward(*dh, ctx);
  ASSERT_TRUE(rb) << rb.error().message;
}

TEST(NN, BpeRoundTrip) {
  const std::string text = "aaabdaaabac aaabdaaabac hello hello world world";
  auto bpe = gyre::Tokenizer::train_bpe(text, 280);
  ASSERT_TRUE(bpe) << bpe.error().message;
  EXPECT_GT((*bpe)->vocab_size(), 256);
  auto ids = (*bpe)->encode(text);
  ASSERT_TRUE(ids);
  EXPECT_LT(ids->size(), text.size());
  EXPECT_EQ((*bpe)->decode(*ids), text);
  auto j = (*bpe)->to_json();
  auto b2 = gyre::Tokenizer::from_json(j);
  ASSERT_TRUE(b2) << b2.error().message;
  auto ids2 = (*b2)->encode(text);
  ASSERT_TRUE(ids2);
  EXPECT_EQ(*ids, *ids2);
}

TEST(NN, LegacyBpeTrailer) {
  const std::string text = "aaabdaaabac hello hello";
  auto bpe = gyre::Tokenizer::train_bpe(text, 270);
  ASSERT_TRUE(bpe);
  const auto& bm = dynamic_cast<const gyre::BpeModel&>((*bpe)->model());
  std::string j = "{\"tokenizer\":\"bpe\",\"merges\":[";
  for (std::size_t i = 0; i < bm.merges().size(); ++i) {
    if (i) j += ',';
    j += '[' + std::to_string(bm.merges()[i].first) + ',' + std::to_string(bm.merges()[i].second) + ']';
  }
  j += "]}";
  auto b2 = gyre::Tokenizer::from_json(j);
  ASSERT_TRUE(b2) << b2.error().message;
  auto a = (*bpe)->encode(text);
  auto b = (*b2)->encode(text);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(*a, *b);
}

TEST(NN, CharsIsBpeWithoutMerges) {
  const std::string text = "abacab";
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  EXPECT_STREQ((*tok)->model_name(), "bpe");
  const auto& bm = dynamic_cast<const gyre::BpeModel&>((*tok)->model());
  EXPECT_TRUE(bm.merges().empty());
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids);
  EXPECT_EQ(ids->size(), text.size());
  EXPECT_EQ((*tok)->decode(*ids), text);
}

TEST(NN, ByteTokenizerAndTinyGptCreate) {
  auto tok = gyre::Tokenizer::bytes();
  ASSERT_TRUE(tok);
  EXPECT_EQ((*tok)->vocab_size(), 256);
  auto ids = (*tok)->encode("Hi");
  ASSERT_TRUE(ids);
  EXPECT_EQ((*ids)[0], static_cast<std::int32_t>('H'));
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto c = gyre::CharLMConfig::tinygpt();
  c.block_size = 8;
  c.n_layer = 1;
  auto m = gyre::CharLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto path = std::filesystem::temp_directory_path() / "gyre-test.onnx";
  auto ex = gyre::export_charlm_onnx(*m, path);
  ASSERT_TRUE(ex) << ex.error().message;
  EXPECT_GT(std::filesystem::file_size(path), 64u);
}

TEST(NN, UnigramViterbi) {
  gyre::PieceTable t;
  t.pieces = {"a", "b", "ab"};
  std::vector<float> scores = {-1.f, -1.f, -0.1f};
  auto um = gyre::UnigramModel::from_scores(scores, false);
  ASSERT_TRUE(um);
  auto tok = std::make_unique<gyre::Tokenizer>(gyre::make_identity_pretok(), std::move(*um), std::move(t));
  auto ids = tok->encode("ab");
  ASSERT_TRUE(ids) << ids.error().message;
  ASSERT_EQ(ids->size(), 1u);
  EXPECT_EQ((*ids)[0], 2);
}

TEST(NN, GyreJsonRoundTripAndHfBpe) {
  const std::string text = "hello hello world world hello";
  auto tok = gyre::Tokenizer::train_bpe(text, 270);
  ASSERT_TRUE(tok);
  auto dir = std::filesystem::temp_directory_path() / "gyre-tok-test";
  std::filesystem::create_directories(dir);
  auto gj = dir / "t.gyre.json";
  ASSERT_TRUE((*tok)->save(gj));
  auto loaded = gyre::Tokenizer::load(gj);
  ASSERT_TRUE(loaded) << loaded.error().message;
  auto a = (*tok)->encode(text);
  auto b = (*loaded)->encode(text);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(*a, *b);
  auto hf = dir / "hf";
  ASSERT_TRUE((*tok)->save_huggingface(hf));
  auto hf_loaded = gyre::Tokenizer::load_huggingface(hf);
  ASSERT_TRUE(hf_loaded) << hf_loaded.error().message;
  auto c = (*hf_loaded)->encode(text);
  ASSERT_TRUE(c);
  EXPECT_EQ(*a, *c);
}
