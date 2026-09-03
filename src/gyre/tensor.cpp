#include "gyre/tensor.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace gyre {
namespace {

Result<std::int64_t> product_shape(std::span<const std::int64_t> shape) {
  if (shape.size() > 8) {
    return std::unexpected(make_error(Errc::invalid_shape, "rank > 8"));
  }
  std::int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) return std::unexpected(make_error(Errc::invalid_shape, "negative dim"));
    if (d == 0) return 0;
    if (n > (std::numeric_limits<std::int64_t>::max() / d)) {
      return std::unexpected(make_error(Errc::overflow, "numel overflow"));
    }
    n *= d;
  }
  return n;
}

bool v1_dtype(DType d) {
  return d == DType::f32 || d == DType::i32 || d == DType::u8 || d == DType::i8 || d == DType::f16 ||
         d == DType::bf16;
}

float bf16_to_f32(std::uint16_t h) {
  std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<std::uint16_t>(u >> 16);
}

float f16_to_f32(std::uint16_t h) {
  const int sign = (h >> 15) & 1;
  const int exp = (h >> 10) & 0x1f;
  const int frac = h & 0x3ff;
  float f;
  if (exp == 0) {
    f = std::ldexp(static_cast<float>(frac), -24);
  } else if (exp == 31) {
    f = frac ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
  } else {
    f = std::ldexp(static_cast<float>(frac | 0x400), exp - 25);
  }
  return sign ? -f : f;
}

}  // namespace

Tensor::Tensor(std::shared_ptr<Device> d, std::shared_ptr<Storage> s, std::size_t off, DType dt,
               std::array<std::int64_t, 8> sh, std::uint8_t rank, std::int64_t n)
    : device_(std::move(d)),
      storage_(std::move(s)),
      offset_(off),
      shape_(sh),
      rank_(rank),
      dtype_(dt),
      numel_(n) {}

Result<Tensor> Tensor::empty(std::span<const std::int64_t> shape, DType dtype,
                             std::shared_ptr<Device> device) {
  if (!device) return std::unexpected(make_error(Errc::unsupported, "null device"));
  if (device->kind() != DeviceKind::cpu) {
    return std::unexpected(make_error(Errc::unsupported, "v1 CPU only"));
  }
  if (!v1_dtype(dtype)) {
    return std::unexpected(make_error(Errc::unsupported, "dtype not enabled in v1"));
  }
  auto n = product_shape(shape);
  if (!n) return std::unexpected(n.error());
  const auto bytes = static_cast<std::size_t>(*n) * dtype_size(dtype);
  auto st = std::make_shared<Storage>();
  st->heap.resize(bytes);
  std::array<std::int64_t, 8> sh{};
  for (std::size_t i = 0; i < shape.size(); ++i) sh[i] = shape[i];
  return Tensor(std::move(device), std::move(st), 0, dtype, sh,
                static_cast<std::uint8_t>(shape.size()), *n);
}

Result<Tensor> Tensor::zeros(std::span<const std::int64_t> shape, DType dtype,
                             std::shared_ptr<Device> device) {
  auto t = empty(shape, dtype, std::move(device));
  if (!t) return t;
  auto hb = t->host_bytes();
  if (!hb) return std::unexpected(hb.error());
  std::memset(hb->data(), 0, hb->size());
  return t;
}

Result<Tensor> Tensor::from_host(std::span<const std::byte> bytes, std::span<const std::int64_t> shape,
                                 DType dtype, std::shared_ptr<Device> device) {
  auto t = empty(shape, dtype, std::move(device));
  if (!t) return t;
  if (bytes.size() != t->nbytes()) {
    return std::unexpected(make_error(Errc::invalid_shape, "from_host size mismatch"));
  }
  auto hb = t->host_bytes();
  if (!hb) return std::unexpected(hb.error());
  std::memcpy(hb->data(), bytes.data(), bytes.size());
  return t;
}

Result<Tensor> Tensor::clone() const {
  auto hb = host_bytes();
  if (!hb) return std::unexpected(hb.error());
  return from_host(*hb, shape(), dtype_, device_);
}

Result<Tensor> Tensor::to(std::shared_ptr<Device> device) const {
  if (!device) return std::unexpected(make_error(Errc::unsupported, "null device"));
  if (device.get() == device_.get()) return clone();
  if (device->kind() != DeviceKind::cpu || device_->kind() != DeviceKind::cpu) {
    return std::unexpected(make_error(Errc::unsupported, "v1 CPU only"));
  }
  return clone();
}

Result<std::span<std::byte>> Tensor::host_bytes() {
  if (!device_ || device_->kind() != DeviceKind::cpu) {
    return std::unexpected(make_error(Errc::not_cpu, "host_bytes requires CPU"));
  }
  if (!storage_ || offset_ + nbytes() > storage_->size()) {
    return std::unexpected(make_error(Errc::overflow, "host_bytes range"));
  }
  auto* p = storage_->data() + offset_;
  return std::span<std::byte>(p, nbytes());
}

Result<std::span<const std::byte>> Tensor::host_bytes() const {
  if (!device_ || device_->kind() != DeviceKind::cpu) {
    return std::unexpected(make_error(Errc::not_cpu, "host_bytes requires CPU"));
  }
  if (!storage_ || offset_ + nbytes() > storage_->size()) {
    return std::unexpected(make_error(Errc::overflow, "host_bytes range"));
  }
  auto* p = storage_->data() + offset_;
  return std::span<const std::byte>(p, nbytes());
}

Result<Tensor> Tensor::from_storage(std::shared_ptr<Storage> st, std::size_t byte_offset,
                                    std::span<const std::int64_t> shape, DType dtype,
                                    std::shared_ptr<Device> device) {
  if (!device || !st) return std::unexpected(make_error(Errc::unsupported, "from_storage"));
  if (!v1_dtype(dtype)) {
    return std::unexpected(make_error(Errc::unsupported, "dtype not enabled"));
  }
  auto n = product_shape(shape);
  if (!n) return std::unexpected(n.error());
  const auto bytes = static_cast<std::size_t>(*n) * dtype_size(dtype);
  if (byte_offset + bytes > st->size()) {
    return std::unexpected(make_error(Errc::overflow, "from_storage range"));
  }
  std::array<std::int64_t, 8> sh{};
  for (std::size_t i = 0; i < shape.size(); ++i) sh[i] = shape[i];
  return Tensor(std::move(device), std::move(st), byte_offset, dtype, sh,
                static_cast<std::uint8_t>(shape.size()), *n);
}

Result<Tensor> Tensor::to_f32() const {
  if (dtype_ == DType::f32) return clone();
  auto out = empty(shape(), DType::f32, device_);
  if (!out) return out;
  auto src = host_bytes();
  auto dst = out->host_span<float>();
  if (!src || !dst) return std::unexpected(src ? dst.error() : src.error());
  if (dtype_ == DType::bf16) {
    auto* h = reinterpret_cast<const std::uint16_t*>(src->data());
    for (std::int64_t i = 0; i < numel_; ++i) (*dst)[static_cast<std::size_t>(i)] = bf16_to_f32(h[i]);
    return out;
  }
  if (dtype_ == DType::f16) {
    auto* h = reinterpret_cast<const std::uint16_t*>(src->data());
    for (std::int64_t i = 0; i < numel_; ++i) (*dst)[static_cast<std::size_t>(i)] = f16_to_f32(h[i]);
    return out;
  }
  return std::unexpected(make_error(Errc::dtype_mismatch, "to_f32"));
}

Result<Tensor> Tensor::view_reshape(const Tensor& a, std::span<const std::int64_t> new_shape) {
  auto n = product_shape(new_shape);
  if (!n) return std::unexpected(n.error());
  if (*n != a.numel_) {
    return std::unexpected(make_error(Errc::invalid_shape, "reshape numel mismatch"));
  }
  std::array<std::int64_t, 8> sh{};
  for (std::size_t i = 0; i < new_shape.size(); ++i) sh[i] = new_shape[i];
  return Tensor(a.device_, a.storage_, a.offset_, a.dtype_, sh,
                static_cast<std::uint8_t>(new_shape.size()), a.numel_);
}

}  // namespace gyre
