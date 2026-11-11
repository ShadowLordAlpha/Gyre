// Training / acting throughput of the RL stack at game-like entity counts.
//   gyre-bench-rl [--threads N] [--entities 128] [--d-model 64] [--layers 2] [--gru 128]
// Prints samples/second for: acting (forward_step), feedforward train step, recurrent train step.
// A "sample" is one decision (one set of entities through encoder + GRU + heads).
#include "gyre/gyre.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gyre;

namespace {

template <class T>
T ok(Result<T> r, const char* what) {
  if (!r) {
    std::fprintf(stderr, "error: %s: %s\n", what, r.error().message.c_str());
    std::exit(1);
  }
  return std::move(*r);
}
void ok(Result<void> r, const char* what) {
  if (!r) {
    std::fprintf(stderr, "error: %s: %s\n", what, r.error().message.c_str());
    std::exit(1);
  }
}

Tensor randn(Rng& rng, std::vector<std::int64_t> sh, float sd = 1.f) {
  std::int64_t n = 1;
  for (auto s : sh) n *= s;
  std::vector<float> v(static_cast<std::size_t>(n));
  for (auto& x : v) x = rng.normal(0.f, sd);
  auto d = Device::cpu();
  return ok(Tensor::from_host(std::as_bytes(std::span<const float>(v)), sh, DType::f32, *d), "tensor");
}
Tensor ones_u8(std::vector<std::int64_t> sh) {
  std::int64_t n = 1;
  for (auto s : sh) n *= s;
  std::vector<std::uint8_t> v(static_cast<std::size_t>(n), 1);
  auto d = Device::cpu();
  return ok(Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(v)), sh, DType::u8, *d), "tensor");
}

template <class F>
double per_second(F&& f, double items, double min_seconds = 2.0) {
  f();
  f();  // warm-up
  const auto t0 = std::chrono::steady_clock::now();
  int iters = 0;
  double el = 0;
  do {
    f();
    ++iters;
    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  } while (el < min_seconds);
  return items * iters / el;
}

}  // namespace

int main(int argc, char** argv) {
  int threads = 0, N = 128, d_model = 64, layers = 2, gru = 128;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    int v = std::atoi(argv[i + 1]);
    if (a == "--threads") threads = v;
    else if (a == "--entities") N = v;
    else if (a == "--d-model") d_model = v;
    else if (a == "--layers") layers = v;
    else if (a == "--gru") gru = v;
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 1;
    }
  }
  if (threads > 0) set_num_threads(threads);

  PolicyConfig c;
  c.encoder.feat_dim = 24;
  c.encoder.global_dim = 16;
  c.encoder.max_entities = N;
  c.encoder.d_model = d_model;
  c.encoder.n_layer = layers;
  c.encoder.n_head = 4;
  c.encoder.d_ff = 2 * d_model;
  c.encoder.out_dim = 2 * d_model;
  c.head_sizes = {12, 8, 64, 16};
  Rng rng(1);
  auto dev = ok(Device::cpu(), "device");

  PolicyConfig cr = c;
  cr.gru_hidden = gru;
  auto ff = ok(PolicyNet::create(c, dev, rng), "ff net");
  auto rec = ok(PolicyNet::create(cr, dev, rng), "rec net");
  std::int64_t params = 0;
  for (auto& p : rec.parameters()) params += p.value.numel();
  std::printf("entities=%d d_model=%d layers=%d gru=%d params=%lld threads=%s\n", N, d_model, layers, gru,
              static_cast<long long>(params), threads > 0 ? std::to_string(threads).c_str() : "default");

  // ---- acting: one step for B streams
  {
    const std::int64_t B = 32;
    Obs obs{randn(rng, {B, N, 24}), ones_u8({B, N}), randn(rng, {B, 16})};
    auto h = ok(Tensor::zeros(std::vector<std::int64_t>{B, gru}, DType::f32, dev), "h");
    std::vector<std::uint8_t> z(B, 0);
    auto reset = ok(Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(z)),
                                      std::vector<std::int64_t>{B}, DType::u8, dev), "reset");
    const double sps = per_second([&] { ok(rec.forward_step(obs, h, reset), "step"); }, double(B));
    std::printf("acting (recurrent, batch %lld):      %9.0f decisions/s\n", static_cast<long long>(B), sps);
  }

  auto make_masks = [&](std::int64_t R) {
    HeadMasks m{ones_u8({R, 100}), ones_u8({R, 4})};
    return m;
  };
  auto adam_ff = ok(Adam::create(ff.parameters(), 1e-4f), "adam");
  auto adam_rec = ok(Adam::create(rec.parameters(), 1e-4f), "adam");

  // ---- feedforward train step (forward + backward + Adam), batch of independent samples
  {
    const std::int64_t B = 64;
    Obs obs{randn(rng, {B, N, 24}), ones_u8({B, N}), randn(rng, {B, 16})};
    auto dl = randn(rng, {B, 100}, 0.01f);
    auto dv = randn(rng, {B}, 0.01f);
    const double sps = per_second(
        [&] {
          ok(ff.zero_grad(), "zero");
          ForwardCtx ctx;
          ctx.train = true;
          ok(ff.forward(obs, ctx), "fwd");
          ok(ff.backward(dl, dv, ctx), "bwd");
          ok(adam_ff.step(ff.parameters()), "adam");
        },
        double(B));
    std::printf("train, feedforward (batch %lld):     %9.0f samples/s\n", static_cast<long long>(B), sps);
  }

  // ---- recurrent train step: B streams x T steps (BPTT)
  {
    const std::int64_t B = 8, T = 16, R = B * T;
    Obs obs{randn(rng, {R, N, 24}), ones_u8({R, N}), randn(rng, {R, 16})};
    auto h0 = ok(Tensor::zeros(std::vector<std::int64_t>{B, gru}, DType::f32, dev), "h0");
    std::vector<std::uint8_t> rv(static_cast<std::size_t>(T * B), 0);
    auto reset = ok(Tensor::from_host(std::as_bytes(std::span<const std::uint8_t>(rv)),
                                      std::vector<std::int64_t>{T, B}, DType::u8, dev), "reset");
    auto dl = randn(rng, {R, 100}, 0.01f);
    auto dv = randn(rng, {R}, 0.01f);
    const double sps = per_second(
        [&] {
          ok(rec.zero_grad(), "zero");
          ForwardCtx ctx;
          ctx.train = true;
          ok(rec.forward_seq(obs, h0, reset, ctx), "fwd");
          ok(rec.backward_seq(dl, dv, ctx), "bwd");
          ok(adam_rec.step(rec.parameters()), "adam");
        },
        double(R));
    std::printf("train, recurrent (%lldx%lld BPTT):     %9.0f samples/s\n", static_cast<long long>(B),
                static_cast<long long>(T), sps);
    std::printf("  => one PPO iteration of 100k samples x 4 epochs ~ %.1f min\n", 400000.0 / sps / 60.0);
  }
  (void)make_masks;
  return 0;
}
