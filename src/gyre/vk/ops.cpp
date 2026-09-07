#include "vk/ops.hpp"

#include "vk/runtime.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace gyre::vkops {
namespace {

vkrt::VulkanDevice* dev_of(const Tensor& t) { return vkrt::VulkanDevice::from(t.device().get()); }

std::uint32_t groups(std::int64_t n, std::uint32_t local = 256) {
  if (n <= 0) return 1;
  return static_cast<std::uint32_t>((n + local - 1) / local);
}

std::array<vkrt::Bind, 7> binds3(const Tensor* a, const Tensor* b, const Tensor* c) {
  std::array<vkrt::Bind, 7> o{};
  o[0].t = a;
  o[1].t = b;
  o[2].t = c;
  return o;
}

}  // namespace

Result<Tensor> add(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 0;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, &b, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> mul(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 1;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, &b, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  std::int64_t osh[2] = {M, N};
  auto out = Tensor::empty(osh, DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(M);
  pc.u[1] = static_cast<std::uint32_t>(K);
  pc.u[2] = static_cast<std::uint32_t>(N);
  pc.u[3] = 1;
  auto r = d->dispatch(vkrt::Pipe::gemm, binds3(&a, &b, &*out), groups(N, 16), groups(M, 16), 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> bmm(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const int rnk = a.rank();
  const auto M = a.shape()[rnk - 2], K = a.shape()[rnk - 1], N = b.shape()[rnk - 1];
  std::int64_t batch = 1;
  std::array<std::int64_t, 8> osh{};
  for (int i = 0; i < rnk - 2; ++i) {
    osh[i] = a.shape()[i];
    batch *= a.shape()[i];
  }
  osh[rnk - 2] = M;
  osh[rnk - 1] = N;
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), rnk), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(M);
  pc.u[1] = static_cast<std::uint32_t>(K);
  pc.u[2] = static_cast<std::uint32_t>(N);
  pc.u[3] = static_cast<std::uint32_t>(batch);
  auto r = d->dispatch(vkrt::Pipe::gemm, binds3(&a, &b, &*out), groups(N, 16), groups(M, 16),
                       static_cast<std::uint32_t>(batch), pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> sum(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(std::span<const std::int64_t>(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = 1;
  pc.u[1] = static_cast<std::uint32_t>(a.numel());
  pc.u[2] = 6;
  pc.f[1] = 1.f;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &a;
  b[1].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, 1, 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> sum_dim(const Tensor& a, int axis, bool keepdim) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (a.rank() == 2 && axis == 0 && !keepdim) {
    const auto rows = a.shape()[0], cols = a.shape()[1];
    std::int64_t osh[1] = {cols};
    auto out = Tensor::empty(osh, DType::f32, a.device());
    if (!out) return out;
    vkrt::Push pc;
    pc.u[0] = static_cast<std::uint32_t>(cols);
    pc.u[1] = 9;
    pc.u[2] = static_cast<std::uint32_t>(rows);
    pc.u[3] = static_cast<std::uint32_t>(cols);
    std::array<vkrt::Bind, 7> b{};
    b[0].t = &a;
    b[2].t = &*out;
    auto r = d->dispatch(vkrt::Pipe::idx, b, groups(cols), 1, 1, pc);
    if (!r) return std::unexpected(r.error());
    return out;
  }
  return std::unexpected(make_error(Errc::unsupported, "vulkan sum_dim shape"));
}

Result<Tensor> transpose_last2(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  std::array<std::int64_t, 8> osh{};
  for (int i = 0; i < a.rank(); ++i) osh[i] = a.shape()[i];
  std::swap(osh[a.rank() - 2], osh[a.rank() - 1]);
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), a.rank()), a.dtype(), a.device());
  if (!out) return out;
  const auto d0 = a.shape()[a.rank() - 2];
  const auto d1 = a.shape()[a.rank() - 1];
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 2;
  pc.u[2] = static_cast<std::uint32_t>(d0);
  pc.u[3] = static_cast<std::uint32_t>(d1);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &a;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> embedding(const Tensor& weight, const Tensor& indices_i32) {
  auto* d = dev_of(weight);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto V = weight.shape()[0], dim = weight.shape()[1];
  std::vector<std::int64_t> osh(indices_i32.shape().begin(), indices_i32.shape().end());
  osh.push_back(dim);
  auto out = Tensor::empty(osh, DType::f32, weight.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(indices_i32.numel());
  pc.u[1] = 0;
  pc.u[2] = static_cast<std::uint32_t>(dim);
  pc.u[3] = static_cast<std::uint32_t>(V);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &weight;
  b[2].t = &*out;
  b[3].t = &indices_i32;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(indices_i32.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> permute_bthd_bhtd(const Tensor& x) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto B = x.shape()[0], T = x.shape()[1], H = x.shape()[2], D = x.shape()[3];
  std::int64_t osh[4] = {B, H, T, D};
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(x.numel());
  pc.u[1] = 3;
  pc.u[2] = static_cast<std::uint32_t>(B);
  pc.u[3] = static_cast<std::uint32_t>(T);
  pc.u[4] = static_cast<std::uint32_t>(H);
  pc.u[5] = static_cast<std::uint32_t>(D);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(x.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> permute_bhtd_bthd(const Tensor& x) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto B = x.shape()[0], H = x.shape()[1], T = x.shape()[2], D = x.shape()[3];
  std::int64_t osh[4] = {B, T, H, D};
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(x.numel());
  pc.u[1] = 4;
  pc.u[2] = static_cast<std::uint32_t>(B);
  pc.u[3] = static_cast<std::uint32_t>(H);
  pc.u[4] = static_cast<std::uint32_t>(T);
  pc.u[5] = static_cast<std::uint32_t>(D);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(x.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> narrow_rows(const Tensor& x, std::int64_t start, std::int64_t count) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto D = x.shape()[1];
  std::int64_t osh[2] = {count, D};
  auto out = Tensor::empty(osh, x.dtype(), x.device());
  if (!out) return out;
  const auto es = dtype_size(x.dtype());
  auto r = d->copy(*out->storage(), out->byte_offset(), *x.storage(),
                   x.byte_offset() + static_cast<std::size_t>(start * D * static_cast<std::int64_t>(es)),
                   static_cast<std::size_t>(count * D * static_cast<std::int64_t>(es)));
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> gelu(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 2;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> silu(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 8;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softmax_last(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  const auto cols = a.shape()[a.rank() - 1];
  const auto rows = a.numel() / cols;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(rows);
  pc.u[1] = static_cast<std::uint32_t>(cols);
  pc.u[2] = 0;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &a;
  b[1].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, static_cast<std::uint32_t>(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(rows);
  pc.u[1] = static_cast<std::uint32_t>(C);
  pc.u[2] = 2;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> bd{};
  bd[0].t = &x;
  bd[1].t = &w;
  bd[2].t = &b;
  bd[3].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, bd, static_cast<std::uint32_t>(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor& w, float eps) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(rows);
  pc.u[1] = static_cast<std::uint32_t>(C);
  pc.u[2] = 5;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[1].t = &w;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, static_cast<std::uint32_t>(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softcap(const Tensor& a, float cap) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 9;
  pc.f[0] = cap;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<void> fill_zero(Tensor& t) {
  auto* d = dev_of(t);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  return d->fill_zero(*t.storage(), t.byte_offset(), t.nbytes());
}

Result<void> fill(Tensor& t, float value) {
  auto* d = dev_of(t);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (t.dtype() != DType::f32) return std::unexpected(make_error(Errc::dtype_mismatch, "fill f32"));
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(t.numel());
  pc.u[1] = 5;
  pc.f[0] = value;
  std::array<vkrt::Bind, 7> b{};
  b[2].t = &t;
  return d->dispatch(vkrt::Pipe::elem, b, groups(t.numel()), 1, 1, pc);
}

Result<void> add_(Tensor& dst, const Tensor& src) {
  auto* d = dev_of(dst);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(dst.numel());
  pc.u[1] = 6;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &src;
  b[2].t = &dst;
  return d->dispatch(vkrt::Pipe::elem, b, groups(dst.numel()), 1, 1, pc);
}

Result<Tensor> gelu_backward(const Tensor& x, const Tensor& dy) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(x.numel());
  pc.u[1] = 3;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&x, &dy, &*out), groups(x.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softmax_last_backward(const Tensor& softmax, const Tensor& d_out) {
  auto* d = dev_of(softmax);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(softmax.shape(), DType::f32, softmax.device());
  if (!out) return out;
  const auto cols = softmax.shape()[softmax.rank() - 1];
  const auto rows = softmax.numel() / cols;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(rows);
  pc.u[1] = static_cast<std::uint32_t>(cols);
  pc.u[2] = 1;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &softmax;
  b[1].t = &d_out;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, static_cast<std::uint32_t>(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> mul_scalar(const Tensor& a, float s) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(a.numel());
  pc.u[1] = 4;
  pc.f[0] = s;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<void> causal_alibi_(Tensor& scores, bool alibi) {
  auto* d = dev_of(scores);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto T = scores.shape()[scores.rank() - 1];
  const auto H = scores.shape()[1];
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(scores.numel());
  pc.u[1] = 5;
  pc.u[2] = static_cast<std::uint32_t>(T);
  pc.u[3] = static_cast<std::uint32_t>(H);
  pc.f[0] = alibi ? 1.f : 0.f;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &scores;
  return d->dispatch(vkrt::Pipe::idx, b, groups(scores.numel()), 1, 1, pc);
}

Result<Tensor> add_broadcast_time(const Tensor& tok, const Tensor& pe) {
  auto* d = dev_of(tok);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto out = Tensor::empty(tok.shape(), DType::f32, tok.device());
  if (!out) return out;
  const auto T = tok.shape()[1], C = tok.shape()[2];
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(tok.numel());
  pc.u[1] = 6;
  pc.u[2] = static_cast<std::uint32_t>(C);
  pc.u[3] = static_cast<std::uint32_t>(T);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &tok;
  b[1].t = &pe;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(tok.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> sum_batch_to_time(const Tensor& dh) {
  auto* d = dev_of(dh);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto B = dh.shape()[0], T = dh.shape()[1], C = dh.shape()[2];
  std::int64_t osh[2] = {T, C};
  auto out = Tensor::empty(osh, DType::f32, dh.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(T * C);
  pc.u[1] = 7;
  pc.u[2] = static_cast<std::uint32_t>(C);
  pc.u[3] = static_cast<std::uint32_t>(T);
  pc.u[4] = static_cast<std::uint32_t>(B);
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &dh;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(T * C), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> arange_i32(std::int64_t n, std::shared_ptr<Device> dev) {
  auto* d = vkrt::VulkanDevice::from(dev.get());
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  std::int64_t sh[1] = {n};
  auto out = Tensor::empty(sh, DType::i32, std::move(dev));
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(n);
  pc.u[1] = 8;
  std::array<vkrt::Bind, 7> b{};
  b[5].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(n), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> layer_norm_backward(const Tensor& x, const Tensor& d_out, const Tensor& w, Tensor& gw,
                                   Tensor& gb, float eps) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  auto dx = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!dx) return dx;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(rows);
  pc.u[1] = static_cast<std::uint32_t>(C);
  pc.u[2] = 3;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[1].t = &d_out;
  b[2].t = &w;
  b[3].t = &*dx;
  b[4].t = &gw;
  b[5].t = &gb;
  auto r = d->dispatch(vkrt::Pipe::row, b, static_cast<std::uint32_t>(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return dx;
}

Result<void> embedding_backward(Tensor& grad_W, const Tensor& idx, const Tensor& d_out) {
  auto* d = dev_of(grad_W);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto dim = grad_W.shape()[1];
  const auto V = grad_W.shape()[0];
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(idx.numel());
  pc.u[1] = 1;
  pc.u[2] = static_cast<std::uint32_t>(dim);
  pc.u[3] = static_cast<std::uint32_t>(V);
  std::array<vkrt::Bind, 7> b{};
  b[1].t = &d_out;
  b[3].t = &idx;
  b[4].t = &grad_W;
  return d->dispatch(vkrt::Pipe::idx, b, groups(idx.numel()), 1, 1, pc);
}

Result<void> bias_add_(Tensor& y, const Tensor& b) {
  auto* d = dev_of(y);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(y.numel());
  pc.u[1] = 10;
  pc.u[2] = static_cast<std::uint32_t>(b.numel());
  std::array<vkrt::Bind, 7> bd{};
  bd[0].t = &y;
  bd[1].t = &b;
  bd[2].t = &y;
  return d->dispatch(vkrt::Pipe::elem, bd, groups(y.numel()), 1, 1, pc);
}

Result<LossPair> softmax_cross_entropy(const Tensor& logits, const Tensor& targets_i32) {
  auto* d = dev_of(logits);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
  auto dlog = Tensor::empty(logits.shape(), DType::f32, logits.device());
  if (!dlog) return std::unexpected(dlog.error());
  std::int64_t rsh[1] = {B * T};
  auto rows = Tensor::empty(rsh, DType::f32, logits.device());
  if (!rows) return std::unexpected(rows.error());
  auto lv = Tensor::empty(std::span<const std::int64_t>(), DType::f32, logits.device());
  if (!lv) return std::unexpected(lv.error());
  const float n = static_cast<float>(B * T);
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(B * T);
  pc.u[1] = static_cast<std::uint32_t>(V);
  pc.u[2] = 4;
  pc.f[1] = 1.f / n;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &logits;
  b[1].t = &*dlog;
  b[2].t = &*rows;
  b[6].t = &targets_i32;
  auto r = d->dispatch(vkrt::Pipe::row, b, static_cast<std::uint32_t>(B * T), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  vkrt::Push rp;
  rp.u[0] = 1;
  rp.u[1] = static_cast<std::uint32_t>(B * T);
  rp.u[2] = 6;
  rp.f[1] = 1.f / n;
  std::array<vkrt::Bind, 7> b2{};
  b2[0].t = &*rows;
  b2[1].t = &*lv;
  auto r2 = d->dispatch(vkrt::Pipe::row, b2, 1, 1, 1, rp);
  if (!r2) return std::unexpected(r2.error());
  return LossPair{std::move(*lv), std::move(*dlog)};
}

Result<void> adam_step(Tensor& w, const Tensor& g, Tensor& m, Tensor& v, float lr, float beta1,
                       float beta2, float eps, float b1t, float b2t) {
  auto* d = dev_of(w);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  vkrt::Push pc;
  pc.u[0] = static_cast<std::uint32_t>(w.numel());
  pc.f[0] = lr;
  pc.f[1] = beta1;
  pc.f[2] = beta2;
  pc.f[3] = eps;
  pc.f[4] = b1t;
  pc.f[5] = b2t;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &w;
  b[1].t = &g;
  b[2].t = &m;
  b[3].t = &v;
  return d->dispatch(vkrt::Pipe::adam, b, groups(w.numel()), 1, 1, pc);
}

}  // namespace gyre::vkops
