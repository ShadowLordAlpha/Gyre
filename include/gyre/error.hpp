#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace gyre {

enum class Errc : std::uint16_t {
  ok = 0,
  invalid_shape,
  overflow,
  dtype_mismatch,
  mixed_device,
  not_cpu,
  io,
  ckpt_corrupt,
  unsupported,
};

struct Error {
  Errc code{Errc::unsupported};
  std::string message;
};

template <class T>
using Result = std::expected<T, Error>;

inline Error make_error(Errc c, std::string m) {
  return Error{c, std::move(m)};
}

}  // namespace gyre
