#include "gyre/data.hpp"
#include "gyre/device.hpp"
#include "gyre/rng.hpp"
#include "gyre/module.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/ops.hpp"
#include "gyre/optim.hpp"
#include "gyre/train/loop.hpp"

#if defined(GYRE_VULKAN)
#include "../src/gyre/vk/runtime.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
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

namespace {

gyre::Tensor rand_cpu(std::vector<std::int64_t> shape, gyre::Rng& rng, float lo = -1.f, float hi = 1.f) {
  auto cpu = gyre::Device::cpu();
  auto t = gyre::Tensor::empty(shape, gyre::DType::f32, *cpu);
  if (!t) std::abort();
  auto s = t->host_span<float>();
  if (!s || !rng.fill_f32(*s, lo, hi)) std::abort();
  return std::move(*t);
}

gyre::Tensor to_dev(const gyre::Tensor& t, const std::shared_ptr<gyre::Device>& d) {
  auto r = t.to(d);
  if (!r) std::abort();
  return std::move(*r);
}

gyre::Tensor to_cpu(const gyre::Tensor& t) {
  auto r = t.to(*gyre::Device::cpu());
  if (!r) std::abort();
  return std::move(*r);
}

}  // namespace

// CPU-vs-Vulkan parity for the backward and reduction kernels a CharLM train
// step uses (and the Grok-only softcap / rms_norm). Shapes are chosen so rows
// have more than one 256-wide chunk and several rows share LN/embedding grads.
TEST(Vulkan, BackwardKernelsMatchCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  gyre::Rng rng(21);

  {  // layer_norm_backward: dx, and gw/gb ACCUMULATE into non-zero buffers
    auto x = rand_cpu({2, 5, 300}, rng, -2.f, 2.f);
    auto dy = rand_cpu({2, 5, 300}, rng);
    auto w = rand_cpu({300}, rng, 0.5f, 1.5f);
    auto gw = rand_cpu({300}, rng), gb = rand_cpu({300}, rng);
    auto gwv = to_dev(gw, vkd), gbv = to_dev(gb, vkd);
    auto gwc = *gw.clone(), gbc = *gb.clone();
    auto dxc = gyre::layer_norm_backward(x, dy, w, gwc, gbc, 1e-5f);
    auto dxv = gyre::layer_norm_backward(to_dev(x, vkd), to_dev(dy, vkd), to_dev(w, vkd), gwv, gbv, 1e-5f);
    ASSERT_TRUE(dxc && dxv) << (dxv ? dxc.error().message : dxv.error().message);
    expect_close(*dxc, to_cpu(*dxv), 1e-4f);
    expect_close(gwc, to_cpu(gwv), 1e-4f);
    expect_close(gbc, to_cpu(gbv), 1e-4f);
  }
  {  // softmax_cross_entropy: loss and dlogits (V > 256)
    auto logits = rand_cpu({2, 3, 300}, rng, -4.f, 4.f);
    std::vector<std::int32_t> t = {0, 299, 5, 128, 256, 17};
    std::int64_t tsh[] = {2, 3};
    auto tc = gyre::Tensor::from_host(std::as_bytes(std::span(t)), tsh, gyre::DType::i32, *gyre::Device::cpu());
    ASSERT_TRUE(tc);
    auto lc = gyre::softmax_cross_entropy(logits, *tc);
    auto lv = gyre::softmax_cross_entropy(to_dev(logits, vkd), to_dev(*tc, vkd));
    ASSERT_TRUE(lc && lv) << (lv ? lc.error().message : lv.error().message);
    EXPECT_NEAR(*lc->value.item_f32(), *lv->value.item_f32(), 1e-4f);
    expect_close(lc->d_pred, to_cpu(lv->d_pred), 1e-5f);
  }
  {  // softmax_last_backward
    auto sm = gyre::softmax_last(rand_cpu({4, 270}, rng, -3.f, 3.f));
    ASSERT_TRUE(sm);
    auto dy = rand_cpu({4, 270}, rng);
    auto bc = gyre::softmax_last_backward(*sm, dy);
    auto bv = gyre::softmax_last_backward(to_dev(*sm, vkd), to_dev(dy, vkd));
    ASSERT_TRUE(bc && bv) << (bv ? bc.error().message : bv.error().message);
    expect_close(*bc, to_cpu(*bv), 1e-5f);
  }
  {  // embedding_backward with repeated ids accumulating into the same rows
    auto gW = rand_cpu({7, 12}, rng);
    std::vector<std::int32_t> ids = {3, 1, 3, 3, 6, 0, 1, 3};
    std::int64_t ish[] = {2, 4};
    auto ic = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), ish, gyre::DType::i32, *gyre::Device::cpu());
    ASSERT_TRUE(ic);
    auto dout = rand_cpu({2, 4, 12}, rng);
    auto gWc = *gW.clone();
    auto gWv = to_dev(gW, vkd);
    ASSERT_TRUE(gyre::embedding_backward(gWc, *ic, dout));
    auto ev = gyre::embedding_backward(gWv, to_dev(*ic, vkd), to_dev(dout, vkd));
    ASSERT_TRUE(ev) << ev.error().message;
    expect_close(gWc, to_cpu(gWv), 1e-5f);
  }
  {  // multi-head attention scores: bmm + causal ALiBi + softmax + bmm
    for (bool alibi : {false, true}) {
      auto q = rand_cpu({2, 3, 5, 4}, rng), kt = rand_cpu({2, 3, 4, 5}, rng), v = rand_cpu({2, 3, 5, 4}, rng);
      auto sc = gyre::bmm(q, kt);
      auto sv = gyre::bmm(to_dev(q, vkd), to_dev(kt, vkd));
      ASSERT_TRUE(sc && sv) << (sv ? sc.error().message : sv.error().message);
      ASSERT_TRUE(gyre::causal_alibi_(*sc, alibi));
      ASSERT_TRUE(gyre::causal_alibi_(*sv, alibi));
      expect_close(*sc, to_cpu(*sv), 1e-3f);
      auto pc = gyre::softmax_last(*sc), pv = gyre::softmax_last(*sv);
      ASSERT_TRUE(pc && pv);
      auto yc = gyre::bmm(*pc, v), yv = gyre::bmm(*pv, to_dev(v, vkd));
      ASSERT_TRUE(yc && yv);
      expect_close(*yc, to_cpu(*yv), 1e-4f);
    }
  }
  {  // sum_dim over rows (the bias-grad reduction)
    auto a = rand_cpu({300, 9}, rng);
    auto c = gyre::sum_dim(a, 0, false);
    auto g = gyre::sum_dim(to_dev(a, vkd), 0, false);
    ASSERT_TRUE(c && g) << (g ? c.error().message : g.error().message);
    expect_close(*c, to_cpu(*g), 1e-3f);
  }
  {  // rms_norm
    auto x = rand_cpu({3, 260}, rng, -2.f, 2.f);
    auto w = rand_cpu({260}, rng, 0.5f, 1.5f);
    auto c = gyre::rms_norm(x, w, 1e-6f);
    auto g = gyre::rms_norm(to_dev(x, vkd), to_dev(w, vkd), 1e-6f);
    ASSERT_TRUE(c && g) << (g ? c.error().message : g.error().message);
    expect_close(*c, to_cpu(*g), 1e-4f);
  }
}

namespace {

std::vector<std::byte> host_bytes_of(const gyre::Tensor& t) {
  auto h = t.to_host_vec();
  if (!h) std::abort();
  return std::move(*h);
}

}  // namespace

// Regression: LayerNorm gamma/beta grads and embedding grads used float
// atomics, so the summation order followed GPU scheduling and two identical
// training runs drifted apart. Every backward reduction now has a fixed order:
// repeating the same work must give byte-identical results.
TEST(Vulkan, BackwardIsBitReproducible) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  constexpr int kRuns = 6;

  {  // layer_norm_backward, many rows (several reduction chunks) and C > 256
    gyre::Rng rng(5);
    auto x = rand_cpu({4, 130, 300}, rng, -2.f, 2.f);
    auto dy = rand_cpu({4, 130, 300}, rng);
    auto w = rand_cpu({300}, rng, 0.5f, 1.5f);
    auto gw = rand_cpu({300}, rng), gb = rand_cpu({300}, rng);
    auto gwc = *gw.clone(), gbc = *gb.clone();
    auto dxc = gyre::layer_norm_backward(x, dy, w, gwc, gbc, 1e-5f);
    ASSERT_TRUE(dxc);
    auto xv = to_dev(x, vkd), dyv = to_dev(dy, vkd), wv = to_dev(w, vkd);
    std::vector<std::byte> dx0, gw0, gb0;
    for (int run = 0; run < kRuns; ++run) {
      auto gwv = to_dev(gw, vkd), gbv = to_dev(gb, vkd);
      auto dxv = gyre::layer_norm_backward(xv, dyv, wv, gwv, gbv, 1e-5f);
      ASSERT_TRUE(dxv) << dxv.error().message;
      if (run == 0) {
        expect_close(*dxc, to_cpu(*dxv), 1e-4f);
        expect_close(gwc, to_cpu(gwv), 2e-3f);
        expect_close(gbc, to_cpu(gbv), 2e-3f);
        dx0 = host_bytes_of(*dxv);
        gw0 = host_bytes_of(gwv);
        gb0 = host_bytes_of(gbv);
        continue;
      }
      EXPECT_TRUE(host_bytes_of(*dxv) == dx0) << "ln dx differs on run " << run;
      EXPECT_TRUE(host_bytes_of(gwv) == gw0) << "ln gw differs on run " << run;
      EXPECT_TRUE(host_bytes_of(gbv) == gb0) << "ln gb differs on run " << run;
    }
  }
  {  // embedding_backward: 4096 tokens over few ids (heavy collisions), C > 256
    gyre::Rng rng(6);
    const std::int64_t V = 40, C = 300, N = 4096;
    std::vector<std::int32_t> ids(static_cast<std::size_t>(N));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      // Mostly ids 0..3, a few others, and some ids never appear.
      const auto r = rng.u64();
      ids[i] = static_cast<std::int32_t>((r % 8) < 6 ? (r >> 8) % 4 : (r >> 8) % 30);
    }
    std::int64_t ish[] = {N};
    auto ic = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), ish, gyre::DType::i32, *gyre::Device::cpu());
    ASSERT_TRUE(ic);
    auto gW = rand_cpu({V, C}, rng);
    auto dout = rand_cpu({N, C}, rng);
    auto gWc = *gW.clone();
    ASSERT_TRUE(gyre::embedding_backward(gWc, *ic, dout));
    auto iv = to_dev(*ic, vkd), dv = to_dev(dout, vkd);
    std::vector<std::byte> g0;
    for (int run = 0; run < kRuns; ++run) {
      auto gWv = to_dev(gW, vkd);
      auto ev = gyre::embedding_backward(gWv, iv, dv);
      ASSERT_TRUE(ev) << ev.error().message;
      if (run == 0) {
        // Same per-id summation order as the CPU loop: identical up to fp contraction.
        expect_close(gWc, to_cpu(gWv), 1e-4f);
        g0 = host_bytes_of(gWv);
        continue;
      }
      EXPECT_TRUE(host_bytes_of(gWv) == g0) << "embedding grad differs on run " << run;
    }
    // Out-of-range ids error like CPU instead of being skipped.
    std::vector<std::int32_t> bad = {0, static_cast<std::int32_t>(V)};
    std::int64_t bsh[] = {2};
    auto bt = gyre::Tensor::from_host(std::as_bytes(std::span(bad)), bsh, gyre::DType::i32, vkd);
    ASSERT_TRUE(bt);
    auto gWv = to_dev(gW, vkd);
    EXPECT_FALSE(gyre::embedding_backward(gWv, *bt, to_dev(rand_cpu({2, C}, rng), vkd)));
  }
  {  // a few CharLM train steps with dropout, AdamW and grad clipping
    const std::string text =
        "to be or not to be that is the question whether tis nobler in the mind to suffer ";
    auto tok = gyre::Tokenizer::chars_from_text(text);
    ASSERT_TRUE(tok);
    auto ids = (*tok)->encode(text);
    ASSERT_TRUE(ids);
    auto data = gyre::CharDataset::from_ids(*ids, vkd);
    ASSERT_TRUE(data);
    gyre::CharLMConfig c;
    c.vocab = (*tok)->vocab_size();
    c.block_size = 32;
    c.n_layer = 2;
    c.n_head = 2;
    c.d_model = 32;
    c.d_ff = 64;
    c.dropout = 0.1f;
    std::vector<std::vector<std::byte>> first;
    for (int run = 0; run < kRuns; ++run) {
      gyre::Rng rng(11);
      auto m = gyre::CharLM::create(c, vkd, rng);
      ASSERT_TRUE(m) << m.error().message;
      gyre::TrainConfig tc;
      tc.steps = 3;
      tc.batch = 4;
      tc.block = 32;
      tc.lr = 3e-3f;
      tc.weight_decay = 0.1f;
      tc.grad_clip = 0.5f;
      tc.log_every = 0;
      tc.seed = 9;
      gyre::TrainLoop loop;
      auto r = loop.run(*m, *data, tc, vkd, {}, {});
      ASSERT_TRUE(r) << r.error().message;
      std::size_t k = 0;
      for (auto& p : m->parameters()) {
        auto b = host_bytes_of(p.value);
        if (run == 0) {
          first.push_back(std::move(b));
        } else {
          ASSERT_LT(k, first.size());
          EXPECT_TRUE(b == first[k]) << "param " << k << " differs on run " << run;
        }
        ++k;
      }
      EXPECT_EQ(k, first.size());
    }
  }
}

// Regression: softcap's tanh had no clamp (GELU's did). Large |x/cap| must
// saturate to +/-cap like CPU std::tanh, never NaN.
TEST(Vulkan, SoftcapSaturatesLikeCpu) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  float v[] = {-1e30f, -1e6f, -3000.f, -50.f, 0.f, 0.5f, 50.f, 3000.f, 1e6f, 1e30f};
  std::int64_t sh[] = {10};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(v)), sh, gyre::DType::f32, *gyre::Device::cpu());
  ASSERT_TRUE(x);
  auto c = gyre::softcap(*x, 30.f);
  auto g = gyre::softcap(to_dev(*x, *vk), 30.f);
  ASSERT_TRUE(c && g) << (g ? c.error().message : g.error().message);
  auto gh = to_cpu(*g);
  auto s = gh.host_span<float>();
  ASSERT_TRUE(s);
  for (auto f : *s) EXPECT_TRUE(std::isfinite(f));
  expect_close(*c, gh, 1e-4f);
}

// Regression: fill_zero rounded the size down and the offset down to 4 bytes,
// leaving the tail of a u8 tensor (and of pooled, reused buffers) unzeroed.
TEST(Vulkan, FillZeroIsByteExact) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto vkd = *vk;
  std::vector<std::uint8_t> ff(7, 0xFF);
  std::int64_t sh[] = {7};
  auto t = gyre::Tensor::from_host(std::as_bytes(std::span(ff)), sh, gyre::DType::u8, vkd);
  ASSERT_TRUE(t) << t.error().message;
  ASSERT_TRUE(gyre::fill_zero(*t));
  auto h = t->to_host_vec();
  ASSERT_TRUE(h);
  ASSERT_EQ(h->size(), 7u);
  for (std::size_t i = 0; i < h->size(); ++i) EXPECT_EQ(static_cast<int>((*h)[i]), 0) << "byte " << i;
  auto z = gyre::Tensor::zeros(sh, gyre::DType::u8, vkd);
  ASSERT_TRUE(z);
  auto hz = z->to_host_vec();
  ASSERT_TRUE(hz);
  for (std::size_t i = 0; i < hz->size(); ++i) EXPECT_EQ(static_cast<int>((*hz)[i]), 0) << "zeros byte " << i;
#if defined(GYRE_VULKAN)
  // Unaligned sub-range [1, 6): neighbours 0, 6 and 7 must keep their value.
  std::vector<std::uint8_t> ff8(8, 0xFF);
  std::int64_t sh8[] = {8};
  auto t8 = gyre::Tensor::from_host(std::as_bytes(std::span(ff8)), sh8, gyre::DType::u8, vkd);
  ASSERT_TRUE(t8);
  auto* dev = gyre::vkrt::VulkanDevice::from(vkd.get());
  ASSERT_NE(dev, nullptr);
  ASSERT_TRUE(dev->fill_zero(*t8->storage(), 1, 5));
  auto h8 = t8->to_host_vec();
  ASSERT_TRUE(h8);
  const int want[8] = {0xFF, 0, 0, 0, 0, 0, 0xFF, 0xFF};
  for (int i = 0; i < 8; ++i) EXPECT_EQ(static_cast<int>((*h8)[static_cast<std::size_t>(i)]), want[i]) << i;
#endif
}

#if defined(GYRE_VULKAN)
// Dispatches past maxComputeWorkGroupCount used to be recorded unchecked.
TEST(Vulkan, DispatchChecksDeviceLimits) {
  std::string err;
  auto vk = try_vk(&err);
  if (!vk) GTEST_SKIP() << err;
  auto* dev = gyre::vkrt::VulkanDevice::from(vk->get());
  ASSERT_NE(dev, nullptr);
  std::array<gyre::vkrt::Bind, 7> b{};
  gyre::vkrt::Push pc;  // elem op 0 with n = 0: a no-op if it were dispatched
  int tested = 0;
  for (int dim = 0; dim < 3; ++dim) {
    const auto lim = dev->limits.maxComputeWorkGroupCount[dim];
    if (lim == UINT32_MAX) continue;
    std::uint32_t g[3] = {1, 1, 1};
    g[dim] = lim + 1;
    auto r = dev->dispatch(gyre::vkrt::Pipe::elem, b, g[0], g[1], g[2], pc);
    ASSERT_FALSE(r) << "dim " << dim;
    EXPECT_EQ(r.error().code, gyre::Errc::unsupported);
    ++tested;
  }
  if (tested == 0) GTEST_SKIP() << "device accepts any group count";
  EXPECT_TRUE(dev->dispatch(gyre::vkrt::Pipe::elem, b, 1, 1, 1, pc));
  dev->synchronize();
}
#endif

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
