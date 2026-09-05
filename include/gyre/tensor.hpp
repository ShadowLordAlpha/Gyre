#pragma once

#include "gyre/device.hpp"
#include "gyre/dtype.hpp"
#include "gyre/error.hpp"

#include "gyre/detail/storage.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

namespace gyre {

class Tensor {
 public:
  Tensor() = delete;
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  static Result<Tensor> empty(std::span<const std::int64_t> shape, DType dtype,
                              std::shared_ptr<Device> device);
  static Result<Tensor> zeros(std::span<const std::int64_t> shape, DType dtype,
                              std::shared_ptr<Device> device);
  static Result<Tensor> from_host(std::span<const std::byte> bytes,
                                  std::span<const std::int64_t> shape, DType dtype,
                                  std::shared_ptr<Device> device);

  Result<Tensor> clone() const;
  Result<Tensor> to(std::shared_ptr<Device> device) const;
  Result<Tensor> to_f32() const;
  static Result<Tensor> from_storage(std::shared_ptr<Storage> st, std::size_t byte_offset,
                                     std::span<const std::int64_t> shape, DType dtype,
                                     std::shared_ptr<Device> device);

  std::span<const std::int64_t> shape() const noexcept {
    return std::span<const std::int64_t>(shape_.data(), rank_);
  }
  std::uint8_t rank() const noexcept { return rank_; }
  std::int64_t numel() const noexcept { return numel_; }
  DType dtype() const noexcept { return dtype_; }
  std::shared_ptr<Device> device() const noexcept { return device_; }
  std::size_t nbytes() const noexcept { return static_cast<std::size_t>(numel_) * dtype_size(dtype_); }
  std::size_t byte_offset() const noexcept { return offset_; }
  std::shared_ptr<Storage> storage() const noexcept { return storage_; }

  Result<std::span<std::byte>> host_bytes();
  Result<std::span<const std::byte>> host_bytes() const;

  template <class T>
  Result<std::span<T>> host_span() {
    auto b = host_bytes();
    if (!b) return std::unexpected(b.error());
    if (!span_type_ok<T>()) {
      return std::unexpected(make_error(Errc::dtype_mismatch, "host_span type mismatch"));
    }
    return std::span<T>(reinterpret_cast<T*>(b->data()), static_cast<std::size_t>(numel_));
  }

  template <class T>
  Result<std::span<const T>> host_span() const {
    auto b = host_bytes();
    if (!b) return std::unexpected(b.error());
    if (!span_type_ok<T>()) {
      return std::unexpected(make_error(Errc::dtype_mismatch, "host_span type mismatch"));
    }
    return std::span<const T>(reinterpret_cast<const T*>(b->data()),
                              static_cast<std::size_t>(numel_));
  }

  // Metadata-only reshape sharing Storage (used by ops::reshape).
  static Result<Tensor> view_reshape(const Tensor& a, std::span<const std::int64_t> new_shape);

 private:
  Tensor(std::shared_ptr<Device> d, std::shared_ptr<Storage> s, std::size_t off, DType dt,
         std::array<std::int64_t, 8> sh, std::uint8_t rank, std::int64_t n);

  template <class T>
  bool span_type_ok() const noexcept {
    if constexpr (std::is_same_v<T, float>) return dtype_ == DType::f32;
    else if constexpr (std::is_same_v<T, std::int32_t>) return dtype_ == DType::i32;
    else if constexpr (std::is_same_v<T, std::uint8_t>) return dtype_ == DType::u8;
    else return false;
  }

  std::shared_ptr<Device> device_;
  std::shared_ptr<Storage> storage_;
  std::size_t offset_{0};
  std::array<std::int64_t, 8> shape_{};
  std::uint8_t rank_{0};
  DType dtype_{DType::f32};
  std::int64_t numel_{0};
};

}  // namespace gyre
