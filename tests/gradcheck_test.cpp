#include "gradcheck.hpp"

#include "gyre/nn/attention.hpp"
#include "gyre/nn/layers.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/ops.hpp"

#include <cstring>
#include <gtest/gtest.h>

using namespace gyre;
using namespace gyre::testing;

namespace {

double dot_all(const Tensor& y, const Tensor& c) {
  auto a = y.host_span<float>();
  auto b = c.host_span<float>();
  double s = 0;
  for (std::size_t i = 0; i < a->size(); ++i) s += double((*a)[i]) * double((*b)[i]);
  return s;
}

// Checks module params and its input gradient with loss = <y, c>.
void check_module(Module& m, Tensor x, std::vector<std::int64_t> yshape, Rng& rng) {
  std::vector<Param> ps;
  for (auto& p : m.parameters()) ps.push_back(Param{p.value, p.grad});
  ps.push_back(make_p(x));
  auto c = rand_t(std::move(yshape), rng);
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    ForwardCtx ctx;
    ctx.train = true;
    auto y = m.forward(ps.back().value, ctx);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto r = m.backward(c, ctx);
      if (!r) return std::unexpected(r.error());
      set_grad(ps.back(), *ctx.dx);
    }
    return dot_all(*y, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

// Like check_module, plus extra params (LoRA A/B) and dropout: every forward gets a fresh Rng with
// the same seed, so the dropout masks are identical across the finite-difference evaluations.
void check_module_ex(Module& m, Tensor x, std::vector<std::int64_t> yshape, Rng& rng,
                     std::vector<Param> extra, std::uint64_t drop_seed) {
  std::vector<Param> ps;
  for (auto& p : m.parameters()) ps.push_back(Param{p.value, p.grad});
  for (auto& p : extra) ps.push_back(Param{p.value, p.grad});
  ps.push_back(make_p(x));
  auto c = rand_t(std::move(yshape), rng);
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    Rng drop(drop_seed);
    ForwardCtx ctx;
    ctx.train = true;
    ctx.rng = &drop;
    auto y = m.forward(ps.back().value, ctx);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto r = m.backward(c, ctx);
      if (!r) return std::unexpected(r.error());
      set_grad(ps.back(), *ctx.dx);
    }
    return dot_all(*y, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

void randomize(std::span<Param> ps, Rng& rng, float sd) {
  for (auto& p : ps) {
    auto v = p.value.host_span<float>();
    for (auto& e : *v) e = rng.normal(0.f, sd);
  }
}

LoraPair rand_lora(std::int64_t in, std::int64_t out, std::int64_t rank, Rng& rng) {
  return LoraPair{rand_t({in, rank}, rng, 0.4f), rand_t({rank, out}, rng, 0.4f)};
}

}  // namespace

TEST(GradCheck, Linear) {
  Rng rng(1);
  auto d = Device::cpu();
  auto lin = Linear::create(5, 4, *d, rng, 0.5f);
  ASSERT_TRUE(lin);
  check_module(*lin, rand_t({3, 5}, rng), {3, 4}, rng);
}

TEST(GradCheck, LinearRank3) {
  Rng rng(2);
  auto d = Device::cpu();
  auto lin = Linear::create(4, 3, *d, rng, 0.5f);
  ASSERT_TRUE(lin);
  check_module(*lin, rand_t({2, 3, 4}, rng), {2, 3, 3}, rng);
}

TEST(GradCheck, LayerNorm) {
  Rng rng(3);
  auto d = Device::cpu();
  auto ln = LayerNorm::create(6, *d);
  ASSERT_TRUE(ln);
  for (auto& p : ln->parameters()) {  // non-trivial scale/shift so gw/gb are exercised
    auto v = p.value.host_span<float>();
    for (auto& e : *v) e += rng.normal(0.f, 0.3f);
  }
  check_module(*ln, rand_t({4, 6}, rng), {4, 6}, rng);
}

TEST(GradCheck, CausalAttention) {
  Rng rng(4);
  auto d = Device::cpu();
  auto at = CausalSelfAttention::create(8, 2, *d, rng, 1.f, true, 0.f);
  ASSERT_TRUE(at);
  for (auto& p : at->parameters()) {
    auto v = p.value.host_span<float>();
    for (auto& e : *v) e = rng.normal(0.f, 0.4f);
  }
  check_module(*at, rand_t({2, 4, 8}, rng), {2, 4, 8}, rng);
}

// Dropout on the attention weights and the residual: covers the saved_drop_attn_ (w * mask for dV,
// dropout_backward on dw) and saved_drop_resid_ backward branches.
TEST(GradCheck, CausalAttentionDropout) {
  Rng rng(41);
  auto d = Device::cpu();
  auto at = CausalSelfAttention::create(8, 2, *d, rng, 1.f, true, 0.3f);
  ASSERT_TRUE(at);
  randomize(at->parameters(), rng, 0.4f);
  check_module_ex(*at, rand_t({2, 4, 8}, rng), {2, 4, 8}, rng, {}, 99);
}

TEST(GradCheck, LinearLora) {
  Rng rng(42);
  auto d = Device::cpu();
  auto lin = Linear::create(5, 4, *d, rng, 0.5f);
  ASSERT_TRUE(lin);
  auto lp = rand_lora(5, 4, 3, rng);
  ASSERT_TRUE(lin->set_lora(lp.A, lp.B, 1.5f));
  std::vector<Param> extra(lin->lora_parameters().begin(), lin->lora_parameters().end());
  ASSERT_EQ(extra.size(), 2u);
  check_module_ex(*lin, rand_t({2, 3, 5}, rng), {2, 3, 4}, rng, extra, 1);
}

TEST(GradCheck, CausalAttentionLoraDropout) {
  Rng rng(43);
  auto d = Device::cpu();
  auto at = CausalSelfAttention::create(8, 2, *d, rng, 1.f, true, 0.2f);
  ASSERT_TRUE(at);
  randomize(at->parameters(), rng, 0.4f);
  auto q = rand_lora(8, 8, 2, rng), k = rand_lora(8, 8, 2, rng), v = rand_lora(8, 8, 2, rng),
       o = rand_lora(8, 8, 2, rng);
  ASSERT_TRUE(at->set_lora(q, k, v, o, 0.5f));
  auto extra = at->lora_parameters();
  ASSERT_EQ(extra.size(), 8u);
  check_module_ex(*at, rand_t({2, 4, 8}, rng), {2, 4, 8}, rng, extra, 7);
}

// Full block with MLP dropout + attention dropout + LoRA on every projection.
TEST(GradCheck, DecoderBlockLoraDropout) {
  Rng rng(44);
  auto d = Device::cpu();
  CharLMConfig c;
  c.n_head = 2;
  c.d_model = 8;
  c.d_ff = 12;
  c.dropout = 0.25f;
  auto blk = DecoderBlock::create(c, *d, rng, 1.f);
  ASSERT_TRUE(blk);
  randomize(blk->parameters(), rng, 0.4f);
  CharLoraLayer layer{rand_lora(8, 8, 2, rng),  rand_lora(8, 8, 2, rng),  rand_lora(8, 8, 2, rng),
                      rand_lora(8, 8, 2, rng),  rand_lora(8, 12, 2, rng), rand_lora(12, 8, 2, rng)};
  ASSERT_TRUE(blk->set_lora(layer, 0.5f));
  auto extra = blk->lora_parameters();
  ASSERT_EQ(extra.size(), 12u);
  check_module_ex(*blk, rand_t({2, 4, 8}, rng), {2, 4, 8}, rng, extra, 5);
}

// Whole CharLM: embedding dropout + blocks + head, with LoRA. Input is token ids, so only
// parameters are checked; loss = <logits, c>.
TEST(GradCheck, CharLMLoraDropout) {
  Rng rng(45);
  auto d = Device::cpu();
  CharLMConfig c;
  c.vocab = 7;
  c.block_size = 4;
  c.n_layer = 2;
  c.n_head = 2;
  c.d_model = 8;
  c.d_ff = 12;
  c.dropout = 0.2f;
  auto m = CharLM::create(c, *d, rng);
  ASSERT_TRUE(m) << m.error().message;
  randomize(m->parameters(), rng, 0.3f);
  Rng lrng(46);
  auto lora = CharLora::create(c, 2, 2.f, *d, lrng);
  ASSERT_TRUE(lora);
  for (auto& l : lora->layers)
    for (auto* p : {&l.q, &l.k, &l.v, &l.o, &l.fc1, &l.fc2}) {
      auto b = p->B.host_span<float>();
      for (auto& e : *b) e = lrng.normal(0.f, 0.3f);
    }
  ASSERT_TRUE(m->set_lora(std::move(*lora)));
  std::vector<Param> ps;
  for (auto& p : m->parameters()) ps.push_back(Param{p.value, p.grad});
  for (auto& p : m->lora_parameters()) ps.push_back(Param{p.value, p.grad});
  ASSERT_EQ(m->lora_parameters().size(), 24u);
  std::vector<std::int32_t> ids = {1, 4, 6, 0, 2, 2, 5, 3};
  std::int64_t ish[] = {2, 4};
  auto idx = Tensor::from_host(std::as_bytes(std::span(ids)), ish, DType::i32, *d);
  ASSERT_TRUE(idx);
  auto cc = rand_t({2, 4, 7}, rng);
  auto fn = [&](bool backward) -> Result<double> {
    for (auto& p : ps) (void)fill_zero(p.grad);
    Rng drop(11);
    ForwardCtx ctx;
    ctx.train = true;
    ctx.rng = &drop;
    auto y = m->forward(*idx, ctx);
    if (!y) return std::unexpected(y.error());
    if (backward) {
      auto r = m->backward(cc, ctx);
      if (!r) return std::unexpected(r.error());
    }
    return dot_all(*y, cc);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

TEST(GradCheck, SoftmaxCrossEntropy) {
  Rng rng(5);
  const std::int64_t B = 2, T = 3, V = 5;
  auto logits = make_p(rand_t({B, T, V}, rng));
  std::vector<std::int32_t> tg = {0, 4, 2, 1, 3, 3};
  auto d = Device::cpu();
  std::int64_t tsh[] = {B, T};
  auto targets = Tensor::from_host(std::as_bytes(std::span(tg)), tsh, DType::i32, *d);
  ASSERT_TRUE(targets);
  std::vector<Param> ps{logits};
  auto fn = [&](bool backward) -> Result<double> {
    auto lp = softmax_cross_entropy(ps[0].value, *targets);
    if (!lp) return std::unexpected(lp.error());
    if (backward) set_grad(ps[0], lp->d_pred);
    auto v = lp->value.item_f32();
    if (!v) return std::unexpected(v.error());
    return double(*v);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

TEST(GradCheck, GeluAndSoftmaxBackward) {
  Rng rng(6);
  auto x = make_p(rand_t({3, 5}, rng));
  auto c = rand_t({3, 5}, rng);
  std::vector<Param> ps{x};
  auto fn = [&](bool backward) -> Result<double> {
    auto g = gelu(ps[0].value);
    if (!g) return std::unexpected(g.error());
    auto s = softmax_last(*g);
    if (!s) return std::unexpected(s.error());
    if (backward) {
      auto ds = softmax_last_backward(*s, c);
      if (!ds) return std::unexpected(ds.error());
      auto dx = gelu_backward(ps[0].value, *ds);
      if (!dx) return std::unexpected(dx.error());
      set_grad(ps[0], *dx);
    }
    return dot_all(*s, c);
  };
  EXPECT_EQ(gradcheck(ps, fn), 0);
}

TEST(Determinism, RngSeeded) {
  Rng a(42), b(42);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a.normal(0, 1), b.normal(0, 1));
    EXPECT_EQ(a.uniform01(), b.uniform01());
    EXPECT_EQ(a.u32(17), b.u32(17));
  }
  Rng c(1);
  for (int i = 0; i < 1000; ++i) {
    float u = c.uniform01();
    EXPECT_GE(u, 0.f);
    EXPECT_LT(u, 1.f);
    EXPECT_LT(c.u32(7), 7u);
  }
  // mt19937_64 output is fixed by the standard: pins the stream across platforms.
  Rng g(5489);
  EXPECT_EQ(g.u64(), 14514284786278117030ull);
}

TEST(Determinism, RngMoments) {
  Rng r(9);
  double s = 0, s2 = 0;
  const int n = 20000;
  for (int i = 0; i < n; ++i) {
    double x = r.normal(0.f, 1.f);
    s += x;
    s2 += x * x;
  }
  EXPECT_NEAR(s / n, 0.0, 0.03);
  EXPECT_NEAR(s2 / n, 1.0, 0.05);
}

TEST(Determinism, DeterministicModeRepeatable) {
  set_deterministic(true);
  EXPECT_TRUE(deterministic());
  Rng rng(3);
  auto d = Device::cpu();
  auto lin = Linear::create(64, 64, *d, rng, 0.1f);
  ASSERT_TRUE(lin);
  auto x = rand_t({128, 64}, rng);
  ForwardCtx c1, c2;
  auto y1 = lin->forward(x, c1);
  auto y2 = lin->forward(x, c2);
  ASSERT_TRUE(y1 && y2);
  auto a = y1->host_span<float>();
  auto b = y2->host_span<float>();
  EXPECT_EQ(0, std::memcmp(a->data(), b->data(), a->size() * sizeof(float)));
}
