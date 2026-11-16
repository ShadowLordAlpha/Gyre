#include "gyre/data.hpp"
#include "gyre/device.hpp"
#include "gyre/rng.hpp"
#include "gyre/module.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/ops.hpp"
#include "gyre/optim.hpp"
#include "gyre/train/loop.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

std::optional<std::shared_ptr<gyre::Device>> try_vk(std::string* err) {
  auto d = gyre::Device::vulkan();
  if (!d) {
    if (err) *err = d.error().message;
    return std::nullopt;
  }
  return *d;
}

void expect_close(const gyre::Tensor& a, const gyre::Tensor& b, float atol = 1e-4f) {
  auto ha = a.to_host_vec();
  auto hb = b.to_host_vec();
  ASSERT_TRUE(ha) << ha.error().message;
  ASSERT_TRUE(hb) << hb.error().message;
  ASSERT_EQ(ha->size(), hb->size());
  auto* fa = reinterpret_cast<const float*>(ha->data());
  auto* fb = reinterpret_cast<const float*>(hb->data());
  const auto n = ha->size() / 4;
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(fa[i], fb[i], atol) << "i=" << i;
  }
}

}  // namespace

TEST(Vulkan, RoundTrip) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t sh[] = {4, 3};
  float v[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto a = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(a);
  auto g = a->to(vkd);
  ASSERT_TRUE(g) << g.error().message;
  auto back = g->to(*cpu);
  ASSERT_TRUE(back);
  expect_close(*a, *back, 0.f);
}

TEST(Vulkan, MatmulMatchesCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t as[] = {2, 3};
  std::int64_t bs[] = {3, 2};
  float A[] = {1, 2, 3, 4, 5, 6};
  float B[] = {1, 0, 0, 1, 1, 1};
  auto a = gyre::Tensor::from_host(std::as_bytes(std::span(A)), as, gyre::DType::f32, *cpu);
  auto b = gyre::Tensor::from_host(std::as_bytes(std::span(B)), bs, gyre::DType::f32, *cpu);
  auto c = gyre::matmul(*a, *b);
  ASSERT_TRUE(c);
  auto ag = a->to(vkd);
  auto bg = b->to(vkd);
  ASSERT_TRUE(ag && bg);
  auto cg = gyre::matmul(*ag, *bg);
  ASSERT_TRUE(cg) << cg.error().message;
  auto ch = cg->to(*cpu);
  ASSERT_TRUE(ch);
  expect_close(*c, *ch);
}

TEST(Vulkan, AddGeluSoftmaxLn) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t sh[] = {2, 4};
  float v[] = {0.1f, -0.2f, 0.3f, 1.4f, -1.f, 0.5f, 2.f, -0.4f};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(x);
  auto xg = x->to(vkd);
  ASSERT_TRUE(xg);
  auto a = gyre::add(*x, *x);
  auto ag = gyre::add(*xg, *xg);
  ASSERT_TRUE(a && ag);
  expect_close(*a, *ag->to(*cpu));
  auto g = gyre::gelu(*x);
  auto gg = gyre::gelu(*xg);
  ASSERT_TRUE(g && gg);
  expect_close(*g, *gg->to(*cpu), 2e-4f);
  auto s = gyre::softmax_last(*x);
  auto sg = gyre::softmax_last(*xg);
  ASSERT_TRUE(s && sg);
  expect_close(*s, *sg->to(*cpu), 2e-4f);
  std::int64_t wsh[] = {4};
  float w[] = {1, 1, 1, 1};
  float b[] = {0, 0, 0, 0};
  auto W = gyre::Tensor::from_host(std::as_bytes(std::span(w)), wsh, gyre::DType::f32, *cpu);
  auto B = gyre::Tensor::from_host(std::as_bytes(std::span(b)), wsh, gyre::DType::f32, *cpu);
  auto ln = gyre::layer_norm(*x, *W, *B, 1e-5f);
  auto lng = gyre::layer_norm(*xg, *W->to(vkd), *B->to(vkd), 1e-5f);
  ASSERT_TRUE(ln && lng) << (lng ? ln.error().message : lng.error().message);
  expect_close(*ln, *lng->to(*cpu), 2e-4f);

  float inf = std::numeric_limits<float>::infinity();
  float iv[] = {inf, 0.f, inf, 1.f, -1.f, inf, 2.f, 3.f};
  std::int64_t ish[] = {2, 4};
  auto xi = gyre::Tensor::from_host(std::as_bytes(std::span(iv)), ish, gyre::DType::f32, *cpu);
  ASSERT_TRUE(xi);
  auto sc = gyre::softmax_last(*xi);
  auto si = gyre::softmax_last(*xi->to(vkd));
  ASSERT_TRUE(sc && si);
  expect_close(*sc, *si->to(*cpu), 2e-4f);
}

TEST(Vulkan, GeluClampMatchesCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t sh[] = {2, 4};
  float v[] = {-40.f, -16.f, -8.f, 0.f, 8.f, 16.f, 40.f, 3.f};
  float dy[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  auto d = gyre::Tensor::from_host(std::as_bytes(std::span(dy)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(x && d);
  auto g = gyre::gelu(*x);
  auto gg = gyre::gelu(*x->to(vkd));
  ASSERT_TRUE(g && gg);
  expect_close(*g, *gg->to(*cpu), 2e-4f);
  auto bg = gyre::gelu_backward(*x, *d);
  auto bgg = gyre::gelu_backward(*x->to(vkd), *d->to(vkd));
  ASSERT_TRUE(bg && bgg) << (bgg ? bg.error().message : bgg.error().message);
  expect_close(*bg, *bgg->to(*cpu), 2e-4f);
}

TEST(Vulkan, TinyTrainStep) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  const std::string text = "hello hello hello hello hello hello hello hello ";
  auto tok = gyre::Tokenizer::chars_from_text(text);
  ASSERT_TRUE(tok);
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids);
  auto data = gyre::CharDataset::from_ids(*ids, vkd);
  ASSERT_TRUE(data);
  gyre::Rng rng(1);
  gyre::CharLMConfig c;
  c.vocab = (*tok)->vocab_size();
  c.block_size = 16;
  c.n_layer = 1;
  c.n_head = 2;
  c.d_model = 16;
  c.d_ff = 32;
  auto m = gyre::CharLM::create(c, vkd, rng);
  ASSERT_TRUE(m) << m.error().message;
  gyre::TrainConfig tc;
  tc.steps = 2;
  tc.batch = 2;
  tc.block = 16;
  tc.lr = 1e-3f;
  tc.log_every = 0;
  tc.grad_clip = 0.f;
  float last = -1;
  gyre::TrainLoop loop;
  auto r = loop.run(*m, *data, tc, vkd, {}, [&](const gyre::Metrics& met) { last = met.loss; });
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_GT(last, 0.f);
  EXPECT_TRUE(std::isfinite(last));
}

TEST(Vulkan, ClipGradNormMatchesCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t sh[] = {4};
  float v[] = {3.f, 4.f, 0.f, 0.f};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  auto g = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(t && g);
  gyre::Param pc{std::move(*t), std::move(*g)};
  auto ncpu = gyre::clip_grad_norm(std::span<gyre::Param>(&pc, 1), 1.f);
  ASSERT_TRUE(ncpu);
  auto tg = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  auto gg = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(tg && gg);
  auto tgv = tg->to(vkd);
  auto ggv = gg->to(vkd);
  ASSERT_TRUE(tgv && ggv);
  gyre::Param pv{std::move(*tgv), std::move(*ggv)};
  auto nvk = gyre::clip_grad_norm(std::span<gyre::Param>(&pv, 1), 1.f);
  ASSERT_TRUE(nvk) << nvk.error().message;
  EXPECT_NEAR(*ncpu, *nvk, 1e-4f);
  auto back = pv.grad.to(*cpu);
  ASSERT_TRUE(back);
  expect_close(pc.grad, *back, 1e-4f);
}

// Regression: Vulkan AdamW decay and clip_grad_norm used to REPLACE the Param
// handle (CharLM::params_ holds copies of the module tensors). The rank>=2
// weights then froze and backward kept writing the old grad buffers. Train a
// tiny CharLM on CPU and Vulkan from the same init with clipping and decay
// forced on, and require the same weights, the same storages, and the same
// logits afterwards.
TEST(Vulkan, AdamDecayAndClipKeepModuleAliasing) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  gyre::CharLMConfig cfg;
  cfg.vocab = 16;
  cfg.block_size = 8;
  cfg.n_layer = 1;
  cfg.n_head = 2;
  cfg.d_model = 16;
  cfg.d_ff = 32;
  cfg.dropout = 0.f;
  gyre::Rng r0(7);
  gyre::Rng r1(7);
  auto mc = gyre::CharLM::create(cfg, *cpu, r0);
  auto mg = gyre::CharLM::create(cfg, vkd, r1);
  ASSERT_TRUE(mc && mg);
  auto oc = gyre::Adam::create(mc->parameters(), 5e-2f);
  auto og = gyre::Adam::create(mg->parameters(), 5e-2f);
  ASSERT_TRUE(oc && og);
  oc->weight_decay = 0.5f;
  og->weight_decay = 0.5f;

  std::vector<const void*> before;
  for (auto& p : mg->parameters()) {
    before.push_back(p.value.storage().get());
    before.push_back(p.grad.storage().get());
  }

  std::vector<std::int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 10, 12, 14, 15};
  std::vector<std::int32_t> tgt = {2, 3, 4, 5, 6, 7, 8, 9, 3, 5, 7, 9, 11, 13, 15, 0};
  std::int64_t sh[] = {2, 8};
  auto mk = [&](const std::vector<std::int32_t>& v, std::shared_ptr<gyre::Device> d) {
    return gyre::Tensor::from_host(std::as_bytes(std::span(v.data(), v.size())), sh,
                                   gyre::DType::i32, std::move(d));
  };
  auto ic = mk(ids, *cpu), ig = mk(ids, vkd), tc = mk(tgt, *cpu), tg = mk(tgt, vkd);
  ASSERT_TRUE(ic && ig && tc && tg);

  auto step = [&](gyre::CharLM& m, gyre::Adam& opt, const gyre::Tensor& x, const gyre::Tensor& y,
                  float* gnorm) {
    ASSERT_TRUE(m.zero_grad());
    gyre::Rng rng(3);
    gyre::ForwardCtx ctx;
    ctx.train = true;
    ctx.rng = &rng;
    auto logits = m.forward(x, ctx);
    ASSERT_TRUE(logits) << logits.error().message;
    auto loss = gyre::softmax_cross_entropy(*logits, y);
    ASSERT_TRUE(loss) << loss.error().message;
    ASSERT_TRUE(m.backward(loss->d_pred, ctx));
    // A tiny max_norm forces the scaling branch on every step.
    auto n = gyre::clip_grad_norm(m.parameters(), 1e-3f);
    ASSERT_TRUE(n) << n.error().message;
    *gnorm = *n;
    ASSERT_TRUE(opt.step(m.parameters()));
  };
  for (int s = 0; s < 3; ++s) {
    float nc = 0.f, ng = 0.f;
    step(*mc, *oc, *ic, *tc, &nc);
    step(*mg, *og, *ig, *tg, &ng);
    ASSERT_GT(nc, 1e-3f);
    // If the grads were not zeroed through the module, Vulkan norms grow.
    EXPECT_NEAR(nc, ng, 1e-3f * std::max(1.f, nc)) << "step " << s;
  }

  std::size_t k = 0;
  for (auto& p : mg->parameters()) {
    EXPECT_EQ(before[k++], p.value.storage().get()) << "param value rebound";
    EXPECT_EQ(before[k++], p.grad.storage().get()) << "param grad rebound";
  }
  auto pc = mc->parameters();
  auto pg = mg->parameters();
  ASSERT_EQ(pc.size(), pg.size());
  for (std::size_t i = 0; i < pc.size(); ++i) {
    auto back = pg[i].value.to(*cpu);
    ASSERT_TRUE(back);
    expect_close(pc[i].value, *back, 1e-4f);
  }
  gyre::ForwardCtx cctx;
  cctx.train = false;
  gyre::ForwardCtx gctx;
  gctx.train = false;
  auto yc = mc->forward(*ic, cctx);
  auto yg = mg->forward(*ig, gctx);
  ASSERT_TRUE(yc && yg);
  expect_close(*yc, *yg->to(*cpu), 1e-3f);
}

TEST(Vulkan, CharLMHiddenEvalIsFinite) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  gyre::CharLMConfig cfg;
  cfg.vocab = 64;
  cfg.block_size = 32;
  cfg.n_layer = 2;
  cfg.n_head = 2;
  cfg.d_model = 16;
  cfg.d_ff = 32;
  cfg.dropout = 0.f;
  gyre::Rng rng(1);
  auto m = gyre::CharLM::create(cfg, *vk, rng);
  ASSERT_TRUE(m) << m.error().message;
  std::int32_t ids[] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::int64_t sh[] = {1, 8};
  auto idx = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, *vk);
  ASSERT_TRUE(idx);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto h = m->hidden(*idx, ctx);
  ASSERT_TRUE(h) << h.error().message;
  auto host = h->to_host_vec();
  ASSERT_TRUE(host);
  auto* f = reinterpret_cast<const float*>(host->data());
  const auto n = host->size() / 4;
  std::size_t nans = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(f[i])) ++nans;
  }
  EXPECT_EQ(nans, 0u) << "vulkan hidden produced " << nans << " non-finite of " << n;
}

TEST(Vulkan, EmbeddingOobMatchesCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t wsh[] = {4, 2};
  float w[] = {1, 2, 3, 4, 5, 6, 7, 8};
  auto W = gyre::Tensor::from_host(std::as_bytes(std::span(w)), wsh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(W);
  std::int32_t bad[] = {0, -1};
  std::int64_t ish[] = {2};
  auto ic = gyre::Tensor::from_host(std::as_bytes(std::span(bad)), ish, gyre::DType::i32, *cpu);
  auto iv = gyre::Tensor::from_host(std::as_bytes(std::span(bad)), ish, gyre::DType::i32, vkd);
  ASSERT_TRUE(ic && iv);
  auto c = gyre::embedding(*W, *ic);
  auto g = gyre::embedding(*W->to(vkd), *iv);
  ASSERT_FALSE(c);
  ASSERT_FALSE(g);
}

TEST(Vulkan, LayerNormMixedDeviceErrors) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  std::int64_t sh[] = {2, 4};
  float v[] = {0.1f, -0.2f, 0.3f, 1.4f, -1.f, 0.5f, 2.f, -0.4f};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(x);
  auto xg = x->to(vkd);
  ASSERT_TRUE(xg);
  std::int64_t wsh[] = {4};
  float w[] = {1, 1, 1, 1};
  float b[] = {0, 0, 0, 0};
  auto W = gyre::Tensor::from_host(std::as_bytes(std::span(w)), wsh, gyre::DType::f32, *cpu);
  auto B = gyre::Tensor::from_host(std::as_bytes(std::span(b)), wsh, gyre::DType::f32, *cpu);
  ASSERT_TRUE(W && B);
  auto ln = gyre::layer_norm(*xg, *W, *B, 1e-5f);
  ASSERT_FALSE(ln);
}

TEST(Vulkan, CharLMHiddenMatchesCpuSovereignWidth) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  auto cpu = gyre::Device::cpu();
  gyre::CharLMConfig cfg = gyre::CharLMConfig::tinygpt();
  cfg.vocab = 2000;
  cfg.dropout = 0.f;
  gyre::Rng r0(1);
  gyre::Rng r1(1);
  auto mc = gyre::CharLM::create(cfg, *cpu, r0);
  auto mg = gyre::CharLM::create(cfg, vkd, r1);
  ASSERT_TRUE(mc) << mc.error().message;
  ASSERT_TRUE(mg) << mg.error().message;
  std::vector<std::int32_t> ids(256);
  for (int i = 0; i < 256; ++i) ids[static_cast<std::size_t>(i)] = i % 1999;
  std::int64_t sh[] = {1, 256};
  auto ic = gyre::Tensor::from_host(std::as_bytes(std::span(ids.data(), ids.size())), sh,
                                   gyre::DType::i32, *cpu);
  auto ig = gyre::Tensor::from_host(std::as_bytes(std::span(ids.data(), ids.size())), sh,
                                   gyre::DType::i32, vkd);
  ASSERT_TRUE(ic && ig);
  gyre::ForwardCtx cctx;
  cctx.train = false;
  gyre::ForwardCtx gctx;
  gctx.train = false;
  auto hc = mc->hidden(*ic, cctx);
  auto hg = mg->hidden(*ig, gctx);
  ASSERT_TRUE(hc) << hc.error().message;
  ASSERT_TRUE(hg) << hg.error().message;
  expect_close(*hc, *hg->to(*cpu), 1e-3f);
  auto yc = mc->forward(*ic, cctx);
  auto yg = mg->forward(*ig, gctx);
  ASSERT_TRUE(yc && yg) << (yg ? yc.error().message : yg.error().message);
  std::vector<std::int32_t> tgt(256);
  for (int i = 0; i < 256; ++i) tgt[static_cast<std::size_t>(i)] = (i + 3) % 1999;
  auto tc = gyre::Tensor::from_host(std::as_bytes(std::span(tgt.data(), tgt.size())), sh,
                                   gyre::DType::i32, *cpu);
  auto tg = gyre::Tensor::from_host(std::as_bytes(std::span(tgt.data(), tgt.size())), sh,
                                   gyre::DType::i32, vkd);
  ASSERT_TRUE(tc && tg);
  auto lc = gyre::softmax_cross_entropy(*yc, *tc);
  auto lg = gyre::softmax_cross_entropy(*yg, *tg);
  ASSERT_TRUE(lc) << lc.error().message;
  ASSERT_TRUE(lg) << lg.error().message;
  auto nc = lc->value.item_f32();
  auto ng = lg->value.item_f32();
  ASSERT_TRUE(nc && ng);
  EXPECT_NEAR(*nc, *ng, 5e-4f);
}
