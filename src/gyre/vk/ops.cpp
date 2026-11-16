#include "vk/ops.hpp"

#include "vk/limits.hpp"
#include "vk/runtime.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

namespace gyre::vkops {
namespace {

vkrt::VulkanDevice* dev_of(const Tensor& t) { return vkrt::VulkanDevice::from(t.device().get()); }

Result<void> on_device(vkrt::VulkanDevice* d, std::initializer_list<const Tensor*> ts) {
  for (auto* t : ts) {
    if (!t) continue;
    if (t->device().get() != d) {
      return std::unexpected(make_error(Errc::mixed_device, "mixed device"));
    }
    if (!t->storage() || !t->storage()->gpu) {
      return std::unexpected(make_error(Errc::mixed_device, "mixed device"));
    }
  }
  return {};
}

Result<void> check_i32_range(const Tensor& idx, std::int64_t hi, const char* what) {
  auto bytes = idx.to_host_vec();
  if (!bytes) return std::unexpected(bytes.error());
  const auto n = idx.numel();
  const auto* p = reinterpret_cast<const std::int32_t*>(bytes->data());
  for (std::int64_t i = 0; i < n; ++i) {
    if (p[i] < 0 || static_cast<std::int64_t>(p[i]) >= hi) {
      return std::unexpected(make_error(Errc::invalid_shape, what));
    }
  }
  return {};
}

// Size limits (see vk/limits.hpp): every op first passes each tensor it binds, and the element
// count of each output it is about to allocate, through fits()/fits_n(). Only then are sizes
// narrowed with u32(): each pushed value is that element count or a factor of it, so it is
// <= kMaxShaderElems and the cast cannot truncate.
Result<void> fits(vkrt::VulkanDevice* d, const char* op, std::initializer_list<const Tensor*> ts) {
  for (auto* t : ts) {
    if (!t) continue;
    auto r = check_extent(t->numel(), dtype_size(t->dtype()), d->limits.maxStorageBufferRange, op);
    if (!r) return r;
  }
  return {};
}

// An f32/i32 output of `numel` elements, checked before it is allocated.
Result<void> fits_n(vkrt::VulkanDevice* d, const char* op, std::int64_t numel) {
  return check_extent(numel, 4, d->limits.maxStorageBufferRange, op);
}

// Only for values bounded by a fits()/fits_n() check (see above).
std::uint32_t u32(std::int64_t v) {
  assert(v >= 0 && v <= kMaxShaderElems);
  return static_cast<std::uint32_t>(v);
}

std::uint32_t groups(std::int64_t n, std::uint32_t local = 256) {
  if (n <= 0) return 1;
  return u32((n + local - 1) / local);
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
  if (auto ok = fits(d, "add", {&a, &b}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 0;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, &b, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> mul(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "mul", {&a, &b}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 1;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, &b, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  if (auto ok = fits(d, "matmul", {&a, &b}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits_n(d, "matmul out", sat_mul(M, N)); !ok) return std::unexpected(ok.error());
  std::int64_t osh[2] = {M, N};
  auto out = Tensor::empty(osh, DType::f32, a.device());
  if (!out) return out;
  if (out->numel() == 0) return out;
  vkrt::Push pc;
  pc.u[0] = u32(M);
  pc.u[1] = u32(K);
  pc.u[2] = u32(N);
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
  if (auto ok = fits(d, "bmm", {&a, &b}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits_n(d, "bmm out", sat_mul(batch, sat_mul(M, N))); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), rnk), DType::f32, a.device());
  if (!out) return out;
  if (out->numel() == 0) return out;  // else batch, M, N are factors of a checked count
  vkrt::Push pc;
  pc.u[0] = u32(M);
  pc.u[1] = u32(K);
  pc.u[2] = u32(N);
  pc.u[3] = u32(batch);
  auto r = d->dispatch(vkrt::Pipe::gemm, binds3(&a, &b, &*out), groups(N, 16), groups(M, 16),
                       u32(batch), pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> sum(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "sum", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(std::span<const std::int64_t>(), DType::f32, a.device());
  if (!out) return out;
  if (a.numel() == 0) return out;  // zero-filled
  vkrt::Push pc;
  pc.u[0] = 1;
  pc.u[1] = u32(a.numel());
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
    if (auto ok = fits(d, "sum_dim", {&a}); !ok) return std::unexpected(ok.error());
    if (auto ok = fits_n(d, "sum_dim out", cols); !ok) return std::unexpected(ok.error());
    std::int64_t osh[1] = {cols};
    auto out = Tensor::empty(osh, DType::f32, a.device());
    if (!out) return out;
    if (a.numel() == 0) return out;  // zero-filled; else rows, cols are factors of a.numel()
    vkrt::Push pc;
    pc.u[0] = u32(cols);
    pc.u[1] = 9;
    pc.u[2] = u32(rows);
    pc.u[3] = u32(cols);
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
  if (auto ok = fits(d, "transpose_last2", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), a.rank()), a.dtype(), a.device());
  if (!out) return out;
  if (a.numel() == 0) return out;
  const auto d0 = a.shape()[a.rank() - 2];
  const auto d1 = a.shape()[a.rank() - 1];
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 2;
  pc.u[2] = u32(d0);
  pc.u[3] = u32(d1);
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
  if (auto ok = on_device(d, {&weight, &indices_i32}); !ok) return std::unexpected(ok.error());
  const auto V = weight.shape()[0], dim = weight.shape()[1];
  if (auto ok = fits(d, "embedding", {&weight, &indices_i32}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits_n(d, "embedding out", sat_mul(indices_i32.numel(), dim)); !ok)
    return std::unexpected(ok.error());
  if (auto ids = check_i32_range(indices_i32, V, "index OOB"); !ids) {
    return std::unexpected(ids.error());
  }
  std::vector<std::int64_t> osh(indices_i32.shape().begin(), indices_i32.shape().end());
  osh.push_back(dim);
  auto out = Tensor::empty(osh, DType::f32, weight.device());
  if (!out) return out;
  if (out->numel() == 0) return out;  // else dim > 0, so V is a factor of weight.numel()
  vkrt::Push pc;
  pc.u[0] = u32(indices_i32.numel());
  pc.u[1] = 0;
  pc.u[2] = u32(dim);
  pc.u[3] = u32(V);
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
  if (auto ok = fits(d, "permute_bthd_bhtd", {&x}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  if (x.numel() == 0) return out;
  vkrt::Push pc;
  pc.u[0] = u32(x.numel());
  pc.u[1] = 3;
  pc.u[2] = u32(B);
  pc.u[3] = u32(T);
  pc.u[4] = u32(H);
  pc.u[5] = u32(D);
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
  if (auto ok = fits(d, "permute_bhtd_bthd", {&x}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  if (x.numel() == 0) return out;
  vkrt::Push pc;
  pc.u[0] = u32(x.numel());
  pc.u[1] = 4;
  pc.u[2] = u32(B);
  pc.u[3] = u32(H);
  pc.u[4] = u32(T);
  pc.u[5] = u32(D);
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
  if (auto ok = fits(d, "gelu", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 2;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> silu(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "silu", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 8;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softmax_last(const Tensor& a) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "softmax_last", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  if (a.numel() == 0) return out;  // also keeps cols = 0 from dividing below
  const auto cols = a.shape()[a.rank() - 1];
  const auto rows = a.numel() / cols;
  vkrt::Push pc;
  pc.u[0] = u32(rows);
  pc.u[1] = u32(cols);
  pc.u[2] = 0;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &a;
  b[1].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, u32(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = on_device(d, {&x, &w, &b}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits(d, "layer_norm", {&x, &w, &b}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  if (x.numel() == 0) return out;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  vkrt::Push pc;
  pc.u[0] = u32(rows);
  pc.u[1] = u32(C);
  pc.u[2] = 2;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> bd{};
  bd[0].t = &x;
  bd[1].t = &w;
  bd[2].t = &b;
  bd[3].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, bd, u32(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor& w, float eps) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = on_device(d, {&x, &w}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits(d, "rms_norm", {&x, &w}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  if (x.numel() == 0) return out;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  vkrt::Push pc;
  pc.u[0] = u32(rows);
  pc.u[1] = u32(C);
  pc.u[2] = 5;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[1].t = &w;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, u32(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softcap(const Tensor& a, float cap) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "softcap", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
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
  if (auto ok = fits(d, "fill", {&t}); !ok) return ok;
  vkrt::Push pc;
  pc.u[0] = u32(t.numel());
  pc.u[1] = 5;
  pc.f[0] = value;
  std::array<vkrt::Bind, 7> b{};
  b[2].t = &t;
  return d->dispatch(vkrt::Pipe::elem, b, groups(t.numel()), 1, 1, pc);
}

Result<void> add_(Tensor& dst, const Tensor& src) {
  auto* d = dev_of(dst);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "add_", {&dst, &src}); !ok) return ok;
  vkrt::Push pc;
  pc.u[0] = u32(dst.numel());
  pc.u[1] = 6;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &src;
  b[2].t = &dst;
  return d->dispatch(vkrt::Pipe::elem, b, groups(dst.numel()), 1, 1, pc);
}

Result<Tensor> gelu_backward(const Tensor& x, const Tensor& dy) {
  auto* d = dev_of(x);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "gelu_backward", {&x, &dy}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(x.numel());
  pc.u[1] = 3;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&x, &dy, &*out), groups(x.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> softmax_last_backward(const Tensor& softmax, const Tensor& d_out) {
  auto* d = dev_of(softmax);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "softmax_last_backward", {&softmax, &d_out}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(softmax.shape(), DType::f32, softmax.device());
  if (!out) return out;
  if (softmax.numel() == 0) return out;
  const auto cols = softmax.shape()[softmax.rank() - 1];
  const auto rows = softmax.numel() / cols;
  vkrt::Push pc;
  pc.u[0] = u32(rows);
  pc.u[1] = u32(cols);
  pc.u[2] = 1;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &softmax;
  b[1].t = &d_out;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::row, b, u32(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<Tensor> mul_scalar(const Tensor& a, float s) {
  auto* d = dev_of(a);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "mul_scalar", {&a}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  vkrt::Push pc;
  pc.u[0] = u32(a.numel());
  pc.u[1] = 4;
  pc.f[0] = s;
  auto r = d->dispatch(vkrt::Pipe::elem, binds3(&a, nullptr, &*out), groups(a.numel()), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  return out;
}

Result<void> mul_scalar_(Tensor& t, float s) {
  auto* d = dev_of(t);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (t.dtype() != DType::f32) return std::unexpected(make_error(Errc::dtype_mismatch, "mul_scalar_ f32"));
  if (auto ok = fits(d, "mul_scalar_", {&t}); !ok) return ok;
  vkrt::Push pc;
  pc.u[0] = u32(t.numel());
  pc.u[1] = 11;
  pc.f[0] = s;
  std::array<vkrt::Bind, 7> b{};
  b[2].t = &t;
  return d->dispatch(vkrt::Pipe::elem, b, groups(t.numel()), 1, 1, pc);
}

Result<void> causal_alibi_(Tensor& scores, bool alibi) {
  auto* d = dev_of(scores);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "causal_alibi_", {&scores}); !ok) return ok;
  if (scores.numel() == 0) return {};
  auto out = Tensor::empty(scores.shape(), DType::f32, scores.device());
  if (!out) return std::unexpected(out.error());
  const auto T = scores.shape()[scores.rank() - 1];
  const auto H = scores.shape()[1];
  vkrt::Push pc;
  pc.u[0] = u32(scores.numel());
  pc.u[1] = 5;
  pc.u[2] = u32(T);
  pc.u[3] = u32(H);
  pc.f[0] = alibi ? 1.f : 0.f;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &scores;
  b[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::idx, b, groups(scores.numel()), 1, 1, pc);
  if (!r) return r;
  scores = std::move(*out);
  return {};
}

Result<Tensor> add_broadcast_time(const Tensor& tok, const Tensor& pe) {
  auto* d = dev_of(tok);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "add_broadcast_time", {&tok, &pe}); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(tok.shape(), DType::f32, tok.device());
  if (!out) return out;
  if (tok.numel() == 0) return out;
  const auto T = tok.shape()[1], C = tok.shape()[2];
  vkrt::Push pc;
  pc.u[0] = u32(tok.numel());
  pc.u[1] = 6;
  pc.u[2] = u32(C);
  pc.u[3] = u32(T);
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
  if (auto ok = fits(d, "sum_batch_to_time", {&dh}); !ok) return std::unexpected(ok.error());
  if (auto ok = fits_n(d, "sum_batch_to_time out", sat_mul(T, C)); !ok) return std::unexpected(ok.error());
  auto out = Tensor::empty(osh, DType::f32, dh.device());
  if (!out) return out;
  if (dh.numel() == 0) return out;  // zero-filled; else B, T, C are factors of dh.numel()
  vkrt::Push pc;
  pc.u[0] = u32(T * C);
  pc.u[1] = 7;
  pc.u[2] = u32(C);
  pc.u[3] = u32(T);
  pc.u[4] = u32(B);
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
  if (auto ok = fits_n(d, "arange_i32", n); !ok) return std::unexpected(ok.error());
  std::int64_t sh[1] = {n};
  auto out = Tensor::empty(sh, DType::i32, std::move(dev));
  if (!out) return out;
  if (n == 0) return out;
  vkrt::Push pc;
  pc.u[0] = u32(n);
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
  if (auto ok = fits(d, "layer_norm_backward", {&x, &d_out, &w, &gw, &gb}); !ok) return std::unexpected(ok.error());
  auto dx = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!dx) return dx;
  if (x.numel() == 0) return dx;
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  if (auto ok = fits_n(d, "layer_norm_backward stats", sat_mul(rows, 2)); !ok) return std::unexpected(ok.error());
  // gamma/beta grads are reduced without atomics so a run is bit-reproducible:
  //   pass 1 (op 3): dx per row, and each row's (mean, inv) into `stats`;
  //   pass 2 (op 7): one invocation per (column, chunk of kChunk rows) sums the
  //     chunk's rows in order;
  //   pass 3 (op 8): gw/gb += the chunk partials in chunk order.
  // With a single chunk, pass 2 accumulates straight into gw/gb (CPU order).
  // kChunk is a constant, so the summation tree is fixed for a given shape.
  constexpr std::int64_t kChunk = 32;
  const std::int64_t chunks = (rows + kChunk - 1) / kChunk;
  std::int64_t ssh[1] = {rows * 2};
  auto stats = Tensor::uninit(ssh, DType::f32, x.device());  // every row writes its pair
  if (!stats) return std::unexpected(stats.error());
  vkrt::Push pc;
  pc.u[0] = u32(rows);
  pc.u[1] = u32(C);
  pc.u[2] = 3;
  pc.f[0] = eps;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &x;
  b[1].t = &d_out;
  b[2].t = &w;
  b[3].t = &*dx;
  b[4].t = &*stats;
  auto r = d->dispatch(vkrt::Pipe::row, b, u32(rows), 1, 1, pc);
  if (!r) return std::unexpected(r.error());

  std::optional<Tensor> pw, pb;
  if (chunks > 1) {
    if (auto ok = fits_n(d, "layer_norm_backward partials", sat_mul(chunks, C)); !ok)
      return std::unexpected(ok.error());
    std::int64_t psh[1] = {chunks * C};
    auto a = Tensor::uninit(psh, DType::f32, x.device());  // pass 2 writes all of it
    auto c = Tensor::uninit(psh, DType::f32, x.device());
    if (!a || !c) return std::unexpected(a ? c.error() : a.error());
    pw = std::move(*a);
    pb = std::move(*c);
  }
  vkrt::Push p2;
  p2.u[0] = u32(rows);
  p2.u[1] = u32(C);
  p2.u[2] = 7;
  p2.u[3] = u32(kChunk);
  p2.u[4] = chunks > 1 ? 0u : 1u;
  std::array<vkrt::Bind, 7> b2{};
  b2[0].t = &x;
  b2[1].t = &d_out;
  b2[2].t = pw ? &*pw : &gw;
  b2[3].t = pb ? &*pb : &gb;
  b2[4].t = &*stats;
  r = d->dispatch(vkrt::Pipe::row, b2, groups(C), u32(chunks), 1, p2);
  if (!r) return std::unexpected(r.error());
  if (chunks > 1) {
    vkrt::Push p3;
    p3.u[0] = u32(chunks);
    p3.u[1] = u32(C);
    p3.u[2] = 8;
    std::array<vkrt::Bind, 7> b3{};
    b3[2].t = &*pw;
    b3[3].t = &*pb;
    b3[4].t = &gw;
    b3[5].t = &gb;
    r = d->dispatch(vkrt::Pipe::row, b3, groups(C), 1, 1, p3);
    if (!r) return std::unexpected(r.error());
  }
  return dx;
}

Result<void> embedding_backward(Tensor& grad_W, const Tensor& idx, const Tensor& d_out) {
  auto* d = dev_of(grad_W);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  const auto dim = grad_W.shape()[1];
  const auto V = grad_W.shape()[0];
  const auto n = idx.numel();
  if (n == 0 || dim == 0) return {};
  if (auto ok = fits(d, "embedding_backward", {&grad_W, &idx, &d_out}); !ok) return ok;  // U*dim <= V*dim
  // Deterministic scatter-add: stable counting sort of token positions by id on
  // the host (ids are small ints), then one invocation per (unique id, column)
  // sums that id's rows in ascending token order. Float atomics would make the
  // order follow GPU scheduling and runs would not be bit-reproducible.
  auto bytes = idx.to_host_vec();
  if (!bytes) return std::unexpected(bytes.error());
  const auto* ids = reinterpret_cast<const std::int32_t*>(bytes->data());
  std::vector<std::int32_t> next(static_cast<std::size_t>(V), 0);  // count, then write cursor
  for (std::int64_t i = 0; i < n; ++i) {
    const auto id = ids[i];
    if (id < 0 || id >= V) return std::unexpected(make_error(Errc::invalid_shape, "emb idx"));
    ++next[static_cast<std::size_t>(id)];
  }
  std::vector<std::int32_t> meta;  // [U unique ids | U+1 segment starts]
  std::vector<std::int32_t> starts;
  std::int32_t run = 0;
  for (std::size_t v = 0; v < next.size(); ++v) {
    const auto c = next[v];
    next[v] = run;
    if (c == 0) continue;
    meta.push_back(static_cast<std::int32_t>(v));
    starts.push_back(run);
    run += c;
  }
  const auto U = static_cast<std::int64_t>(meta.size());
  starts.push_back(run);
  meta.insert(meta.end(), starts.begin(), starts.end());
  std::vector<std::int32_t> order(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    order[static_cast<std::size_t>(next[static_cast<std::size_t>(ids[i])]++)] = static_cast<std::int32_t>(i);
  }
  std::int64_t osh[1] = {n};
  std::int64_t msh[1] = {static_cast<std::int64_t>(meta.size())};
  auto ot = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(order)), osh, DType::i32,
                              grad_W.device());
  if (!ot) return std::unexpected(ot.error());
  auto mt = Tensor::from_host(std::as_bytes(std::span<const std::int32_t>(meta)), msh, DType::i32,
                              grad_W.device());
  if (!mt) return std::unexpected(mt.error());
  vkrt::Push pc;
  pc.u[0] = u32(U * dim);
  pc.u[1] = 1;
  pc.u[2] = u32(dim);
  pc.u[3] = u32(U);
  std::array<vkrt::Bind, 7> b{};
  b[1].t = &d_out;
  b[3].t = &*ot;
  b[4].t = &grad_W;
  b[5].t = &*mt;
  return d->dispatch(vkrt::Pipe::idx, b, groups(U * dim), 1, 1, pc);
}

Result<void> bias_add_(Tensor& y, const Tensor& b) {
  auto* d = dev_of(y);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = on_device(d, {&y, &b}); !ok) return ok;
  if (auto ok = fits(d, "bias_add_", {&y, &b}); !ok) return ok;
  auto out = Tensor::empty(y.shape(), DType::f32, y.device());
  if (!out) return std::unexpected(out.error());
  vkrt::Push pc;
  pc.u[0] = u32(y.numel());
  pc.u[1] = 10;
  pc.u[2] = u32(b.numel());
  std::array<vkrt::Bind, 7> bd{};
  bd[0].t = &y;
  bd[1].t = &b;
  bd[2].t = &*out;
  auto r = d->dispatch(vkrt::Pipe::elem, bd, groups(y.numel()), 1, 1, pc);
  if (!r) return r;
  y = std::move(*out);
  return {};
}

Result<LossPair> softmax_cross_entropy(const Tensor& logits, const Tensor& targets_i32) {
  auto* d = dev_of(logits);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = on_device(d, {&logits, &targets_i32}); !ok) return std::unexpected(ok.error());
  const auto B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
  // B*T is targets_i32.numel() and V a factor of logits.numel(), so both are bounded here.
  if (auto ok = fits(d, "softmax_cross_entropy", {&logits, &targets_i32}); !ok) return std::unexpected(ok.error());
  if (auto ids = check_i32_range(targets_i32, V, "target OOB"); !ids) {
    return std::unexpected(ids.error());
  }
  auto dlog = Tensor::empty(logits.shape(), DType::f32, logits.device());
  if (!dlog) return std::unexpected(dlog.error());
  std::int64_t rsh[1] = {B * T};
  auto rows = Tensor::empty(rsh, DType::f32, logits.device());
  if (!rows) return std::unexpected(rows.error());
  auto lv = Tensor::empty(std::span<const std::int64_t>(), DType::f32, logits.device());
  if (!lv) return std::unexpected(lv.error());
  const float n = static_cast<float>(B * T);
  vkrt::Push pc;
  pc.u[0] = u32(B * T);
  pc.u[1] = u32(V);
  pc.u[2] = 4;
  pc.f[1] = 1.f / n;
  std::array<vkrt::Bind, 7> b{};
  b[0].t = &logits;
  b[1].t = &*dlog;
  b[2].t = &*rows;
  b[6].t = &targets_i32;
  auto r = d->dispatch(vkrt::Pipe::row, b, u32(B * T), 1, 1, pc);
  if (!r) return std::unexpected(r.error());
  vkrt::Push rp;
  rp.u[0] = 1;
  rp.u[1] = u32(B * T);
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
                       float beta2, float eps, float b1t, float b2t, float decay) {
  auto* d = dev_of(w);
  if (!d) return std::unexpected(make_error(Errc::unsupported, "vulkan device"));
  if (auto ok = fits(d, "adam_step", {&w, &g, &m, &v}); !ok) return ok;
  vkrt::Push pc;
  pc.u[0] = u32(w.numel());
  pc.u[1] = decay != 1.f ? 1u : 0u;
  pc.u[2] = std::bit_cast<std::uint32_t>(decay);
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
