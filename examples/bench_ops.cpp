// Per-op timings at the shapes a 128-entity PolicyNet training step uses (B=64, N=128, d=64, 4 heads).
#if defined(_MSC_VER)
#pragma warning(disable : 4834)  // results of the timed ops are intentionally discarded
#endif
#include "gyre/gyre.hpp"
#include <chrono>
#include <cstdio>
using namespace gyre;
template <class F> double ms(F&& f, int reps = 20) {
  f(); f();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}
Tensor rnd(Rng& r, std::vector<std::int64_t> sh) {
  std::int64_t n = 1; for (auto s : sh) n *= s;
  std::vector<float> v(n); for (auto& x : v) x = r.normal(0, 1);
  auto d = Device::cpu();
  return std::move(*Tensor::from_host(std::as_bytes(std::span<const float>(v)), sh, DType::f32, *d));
}
int main() {
  Rng r(1);
  const int B = 64, N = 128, d = 64, H = 4, dk = 16;
  auto x = rnd(r, {B * N, d}); auto W = rnd(r, {d, d}); auto W2 = rnd(r, {d, 2 * d}); auto b = rnd(r, {d});
  auto x3 = rnd(r, {B, N, d});
  auto q = rnd(r, {B, H, N, dk}); auto kt = rnd(r, {B, H, dk, N}); auto sc = rnd(r, {B, H, N, N});
  auto v4 = rnd(r, {B, N, H, dk});
  auto h2 = rnd(r, {B * N, 2 * d});
  auto lw = rnd(r, {d});
  double t;
  t = ms([&] { matmul(x, W); }); std::printf("matmul 8192x64 @ 64x64        %8.3f ms  %6.1f GFLOP/s\n", t, 2.0 * B * N * d * d / t / 1e6);
  t = ms([&] { matmul(x, W2); }); std::printf("matmul 8192x64 @ 64x128       %8.3f ms  %6.1f GFLOP/s\n", t, 2.0 * B * N * d * 2 * d / t / 1e6);
  t = ms([&] { matmul(h2, *transpose_last2(W2)); }); std::printf("matmul 8192x128 @ 128x64 (+Wt)%8.3f ms\n", t);
  t = ms([&] { bmm(q, kt); }); std::printf("bmm [256]128x16 @ 16x128      %8.3f ms  %6.1f GFLOP/s\n", t, 2.0 * B * H * N * N * dk / t / 1e6);
  t = ms([&] { bmm(sc, q); }); std::printf("bmm [256]128x128 @ 128x16     %8.3f ms  %6.1f GFLOP/s\n", t, 2.0 * B * H * N * N * dk / t / 1e6);
  t = ms([&] { transpose_last2(q); }); std::printf("transpose_last2 [256,128,16]  %8.3f ms\n", t);
  t = ms([&] { transpose_last2(x); }); std::printf("transpose_last2 [8192,64]     %8.3f ms\n", t);
  t = ms([&] { permute_bthd_bhtd(v4); }); std::printf("permute bthd->bhtd            %8.3f ms\n", t);
  t = ms([&] { softmax_last(sc); }); std::printf("softmax_last [256,128,128]    %8.3f ms\n", t);
  { auto w = *softmax_last(sc); auto dw = rnd(r, {B, H, N, N});
  t = ms([&] { softmax_last_backward(w, dw); }); std::printf("softmax_backward              %8.3f ms\n", t); }
  t = ms([&] { mul_scalar(sc, 0.25f); }); std::printf("mul_scalar [256,128,128]      %8.3f ms\n", t);
  t = ms([&] { Tensor s = sc; key_mask_(s, *Tensor::zeros(std::vector<std::int64_t>{B, N}, DType::u8, *Device::cpu())); }); std::printf("key_mask_ (in place)          %8.3f ms\n", t);
  t = ms([&] { layer_norm(x, lw, lw, 1e-5f); }); std::printf("layer_norm 8192x64            %8.3f ms\n", t);
  t = ms([&] { gelu(h2); }); std::printf("gelu 8192x128                 %8.3f ms\n", t);
  { auto dy = rnd(r, {B * N, 2 * d});
  t = ms([&] { gelu_backward(h2, dy); }); std::printf("gelu_backward 8192x128        %8.3f ms\n", t); }
  t = ms([&] { add(x, x); }); std::printf("add 8192x64                   %8.3f ms\n", t);
  t = ms([&] { linear(x, W, b); }); std::printf("linear 8192x64->64            %8.3f ms\n", t);
  t = ms([&] { Tensor::empty(std::vector<std::int64_t>{B, H, N, N}, DType::f32, *Device::cpu()); }); std::printf("alloc [256,128,128]           %8.3f ms\n", t);
  t = ms([&] { sum_dim(x, 0, false); }); std::printf("sum_dim(0) 8192x64            %8.3f ms\n", t);
  return 0;
}
