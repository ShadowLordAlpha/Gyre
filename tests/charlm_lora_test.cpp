#include "gyre/module.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/optim.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {

auto tiny_cfg() -> gyre::CharLMConfig {
  gyre::CharLMConfig c;
  c.vocab = 8;
  c.block_size = 8;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 8;
  c.d_ff = 16;
  return c;
}

auto idx_tensor(const std::shared_ptr<gyre::Device>& d) -> gyre::Tensor {
  std::int32_t ids[] = {1, 2, 3, 0};
  std::int64_t sh[] = {1, 4};
  return *gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, d);
}

auto last_logit(const gyre::Tensor& y) -> float {
  auto p = y.host_span<float>();
  if (!p || p->empty()) return 0.f;
  return (*p)[p->size() - 1];
}

}  // namespace

TEST(CharLMLora, ZeroBMatchesBaseThenTrainChanges) {
  auto d = gyre::Device::cpu();
  ASSERT_TRUE(d);
  gyre::Rng rng(3);
  auto cfg = tiny_cfg();
  auto m = gyre::CharLM::create(cfg, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  auto x = idx_tensor(*d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y0 = m->forward(x, ctx);
  ASSERT_TRUE(y0) << y0.error().message;
  const float base = last_logit(*y0);

  gyre::Rng lrng(9);
  auto lora = gyre::CharLora::create(cfg, 2, 4.f, *d, lrng);
  ASSERT_TRUE(lora) << lora.error().message;
  ASSERT_TRUE(m->set_lora(std::move(*lora)));
  auto y1 = m->forward(x, ctx);
  ASSERT_TRUE(y1);
  EXPECT_NEAR(last_logit(*y1), base, 1e-5f);

  m->set_freeze_base(true);
  auto params = m->parameters();
  std::vector<float> w0;
  {
    auto s = params[0].value.host_span<float>();
    ASSERT_TRUE(s);
    w0.assign(s->begin(), s->end());
  }
  auto lparams = m->lora_parameters();
  ASSERT_FALSE(lparams.empty());
  auto opt = gyre::Adam::create(lparams, 0.05f);
  ASSERT_TRUE(opt);

  std::int32_t target_ids[] = {2, 3, 0, 1};
  std::int64_t tsh[] = {1, 4};
  auto tgt = gyre::Tensor::from_host(std::as_bytes(std::span(target_ids)), tsh, gyre::DType::i32, *d);
  ASSERT_TRUE(tgt);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(m->zero_grad());
    gyre::ForwardCtx tctx;
    tctx.train = true;
    auto y = m->forward(x, tctx);
    ASSERT_TRUE(y);
    auto loss = gyre::softmax_cross_entropy(*y, *tgt);
    ASSERT_TRUE(loss);
    ASSERT_TRUE(m->backward(loss->d_pred, tctx));
    ASSERT_TRUE(opt->step(m->lora_parameters()));
  }
  auto y2 = m->forward(x, ctx);
  ASSERT_TRUE(y2);
  EXPECT_NE(last_logit(*y2), base);

  auto s = params[0].value.host_span<float>();
  ASSERT_TRUE(s);
  for (std::size_t i = 0; i < w0.size(); ++i) EXPECT_EQ((*s)[i], w0[i]);
}

TEST(CharLMLora, SaveLoadRoundTrip) {
  auto d = gyre::Device::cpu();
  ASSERT_TRUE(d);
  gyre::Rng rng(4);
  auto cfg = tiny_cfg();
  auto m = gyre::CharLM::create(cfg, *d, rng);
  ASSERT_TRUE(m);
  gyre::Rng lrng(5);
  auto lora = gyre::CharLora::create(cfg, 2, 4.f, *d, lrng);
  ASSERT_TRUE(lora);
  auto& b = lora->layers[0].q.B;
  auto bp = b.host_span<float>();
  ASSERT_TRUE(bp);
  if (!bp->empty()) (*bp)[0] = 0.25f;
  ASSERT_TRUE(m->set_lora(std::move(*lora)));

  auto x = idx_tensor(*d);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto y = m->forward(x, ctx);
  ASSERT_TRUE(y);
  const float want = last_logit(*y);

  const auto path = std::filesystem::temp_directory_path() / "gyre_charlm.lora";
  ASSERT_TRUE(m->save_lora(path)) << "save_lora failed";

  gyre::Rng rng2(4);
  auto m2 = gyre::CharLM::create(cfg, *d, rng2);
  ASSERT_TRUE(m2);
  ASSERT_TRUE(m2->load_lora(path, *d)) << "load_lora failed";
  auto y2 = m2->forward(x, ctx);
  ASSERT_TRUE(y2);
  EXPECT_NEAR(last_logit(*y2), want, 1e-5f);
  std::filesystem::remove(path);
}
