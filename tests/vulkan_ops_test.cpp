#include "gyre/data.hpp"
#include "gyre/device.hpp"
#include "gyre/nn/transformer.hpp"
#include "gyre/ops.hpp"
#include "gyre/train/loop.hpp"

#include <cmath>
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
  float last = -1;
  gyre::TrainLoop loop;
  auto r = loop.run(*m, *data, tc, vkd, {}, [&](const gyre::Metrics& met) { last = met.loss; });
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_GT(last, 0.f);
  EXPECT_TRUE(std::isfinite(last));
}
