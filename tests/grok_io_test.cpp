#include "gyre/data.hpp"
#include "gyre/io/safetensors.hpp"
#include "gyre/nn/grok.hpp"
#include "gyre/nn/tokenize.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

TEST(GrokIO, GyreSaveLoadGenerateMatches) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(1);
  auto a = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  ASSERT_TRUE(a);
  auto dir = std::filesystem::temp_directory_path() / "gyre_grok_gyre";
  std::filesystem::create_directories(dir);
  auto path = dir / "mini.gyre";
  auto s = a->save_gyre(path);
  ASSERT_TRUE(s) << s.error().message;
  auto b = gyre::GrokLM::load_gyre(path, *d);
  ASSERT_TRUE(b) << b.error().message;
  std::vector<std::int32_t> prefix{1, 2};
  auto ga = a->generate(prefix, 8, *d);
  auto gb = b->generate(prefix, 8, *d);
  ASSERT_TRUE(ga && gb);
  EXPECT_EQ(*ga, *gb);
}

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

// Regression: import used to write a .gyre full of random init weights when
// tensors (or whole shards) were missing from the directory.
TEST(GrokIO, ImportSafetensorsRequiresEveryTensor) {
  auto d = gyre::Device::cpu();
  gyre::Rng rng(4);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng);
  ASSERT_TRUE(m);
  auto dir = std::filesystem::temp_directory_path() / "gyre_grok_import";
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(m->save_weights(dir));
  auto out = dir / "imported.gyre";
  auto ok = gyre::import_safetensors_to_gyre(dir, out, *d);
  ASSERT_TRUE(ok) << ok.error().message;
  auto back = gyre::GrokLM::load_gyre(out, *d);
  ASSERT_TRUE(back) << back.error().message;
  auto ga = m->generate({1, 2}, 4, *d, nullptr, 0.f);
  auto gb = back->generate({1, 2}, 4, *d, nullptr, 0.f);
  ASSERT_TRUE(ga && gb);
  EXPECT_EQ(*ga, *gb);

  // Same config, no shard: every tensor is missing.
  std::filesystem::remove(dir / "model.safetensors");
  std::filesystem::remove(out);
  auto bad = gyre::import_safetensors_to_gyre(dir, out, *d);
  EXPECT_FALSE(bad);
  EXPECT_FALSE(std::filesystem::exists(out));
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

namespace {

void write_text(const std::filesystem::path& p, const std::string& s) {
  std::ofstream o(p, std::ios::binary);
  o << s;
}

}  // namespace

// Loaded adapters must match the model's [in,rank]/[rank,out] slots (an adapter saved for another
// config used to be accepted as-is), and bad lora.json types are an error, not an exception.
TEST(GrokLora, LoadRejectsMismatchedShapesAndBadJson) {
  auto d = gyre::Device::cpu();
  auto other = gyre::GrokConfig::mini();
  other.n_kv_head = 2;  // k/v adapters are [rank, 8] instead of [rank, 4]
  gyre::Rng rng(21);
  auto src = gyre::GrokLM::create(other, *d, rng);
  ASSERT_TRUE(src) << src.error().message;
  gyre::Rng lr(22);
  auto l = gyre::GrokLora::create(other, 2, 4.f, *d, lr);
  ASSERT_TRUE(l);
  ASSERT_TRUE(src->set_lora(std::move(*l)));
  auto dir = std::filesystem::temp_directory_path() / "gyre_lora_mismatch";
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(src->save_lora(dir));

  gyre::Rng rng2(23);
  auto m = gyre::GrokLM::create(gyre::GrokConfig::mini(), *d, rng2);
  ASSERT_TRUE(m);
  auto r = m->load_lora(dir, *d);
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().code, gyre::Errc::invalid_shape);

  for (const char* js : {R"({"rank":"2","alpha":4})", R"({"rank":2,"alpha":"x"})", R"({"rank":100000000})"}) {
    write_text(dir / "lora.json", js);
    gyre::Result<void> rr;
    EXPECT_NO_THROW(rr = m->load_lora(dir, *d)) << js;
    EXPECT_FALSE(rr) << js;
  }
  std::filesystem::remove_all(dir);
}

// f16 adapters are widened to f32 like base weights (matmul is f32-only).
TEST(GrokLora, LoadF16AdapterMatchesF32) {
  auto d = gyre::Device::cpu();
  const auto c = gyre::GrokConfig::mini();
  const std::int64_t rank = 2, din = c.d_model, qout = c.n_q_head * c.head_dim,
                     kout = c.n_kv_head * c.head_dim;
  auto f16 = [&](std::int64_t r, std::int64_t cc, std::uint16_t bits) {
    std::vector<std::uint16_t> v(static_cast<std::size_t>(r * cc), bits);
    std::int64_t sh[] = {r, cc};
    return *gyre::Tensor::from_host(std::as_bytes(std::span<const std::uint16_t>(v)), sh, gyre::DType::f16, *d);
  };
  // 0x3800 = 0.5, 0x3000 = 0.125 (exact in f16 and f32)
  std::vector<gyre::Tensor> keep;
  std::vector<std::string> names;
  auto add = [&](const std::string& kind, std::int64_t in, std::int64_t out) {
    names.push_back("layers.0." + kind + ".A");
    keep.push_back(f16(in, rank, 0x3800));
    names.push_back("layers.0." + kind + ".B");
    keep.push_back(f16(rank, out, 0x3000));
  };
  add("q_proj", din, qout);
  add("k_proj", din, kout);
  add("v_proj", din, kout);
  add("o_proj", qout, din);
  std::vector<gyre::NamedTensor> nt;
  for (std::size_t i = 0; i < keep.size(); ++i) nt.push_back({names[i], &keep[i]});
  auto dir = std::filesystem::temp_directory_path() / "gyre_lora_f16";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  ASSERT_TRUE(gyre::safetensors_save(dir / "lora.safetensors", nt));
  write_text(dir / "lora.json", R"({"gyre":"lora","base_arch":"grok2","rank":2,"alpha":4})");

  gyre::Rng r1(31), r2(31);
  auto a = gyre::GrokLM::create(c, *d, r1);
  auto b = gyre::GrokLM::create(c, *d, r2);
  ASSERT_TRUE(a && b);
  auto ld = a->load_lora(dir, *d);
  ASSERT_TRUE(ld) << ld.error().message;
  gyre::Rng lr(1);
  auto l = gyre::GrokLora::create(c, rank, 4.f, *d, lr);
  ASSERT_TRUE(l);
  for (auto* v : {&l->q, &l->k, &l->v, &l->o})
    for (auto& p : *v) {
      for (auto& x : *p.A.host_span<float>()) x = 0.5f;
      for (auto& x : *p.B.host_span<float>()) x = 0.125f;
    }
  ASSERT_TRUE(b->set_lora(std::move(*l)));
  std::int32_t ids[] = {1, 2, 3};
  std::int64_t sh[] = {1, 3};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto ya = a->forward(*idx, ctx);
  ASSERT_TRUE(ya) << ya.error().message;
  auto yb = b->forward(*idx, ctx);
  ASSERT_TRUE(yb);
  auto pa = ya->host_span<float>();
  auto pb = yb->host_span<float>();
  for (std::size_t i = 0; i < pa->size(); ++i) EXPECT_NEAR((*pa)[i], (*pb)[i], 1e-5f);
  std::filesystem::remove_all(dir);
}
