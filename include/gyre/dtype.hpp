#pragma once

#include <cstddef>
#include <cstdint>

namespace gyre {

enum class DType : std::uint8_t {
  f32 = 1,
  f16 = 2,
  bf16 = 3,
  i32 = 4,
  i8 = 5,
  u8 = 6
};

constexpr std::size_t dtype_size(DType d) noexcept {
  switch (d) {
    case DType::f32:
    case DType::i32:
      return 4;
    case DType::f16:
    case DType::bf16:
      return 2;
    case DType::i8:
    case DType::u8:
      return 1;
  }
  return 0;
}

}  // namespace gyre
