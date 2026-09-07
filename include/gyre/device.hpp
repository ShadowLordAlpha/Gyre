#pragma once

#include "gyre/error.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace gyre {

enum class DeviceKind : std::uint8_t {
  cpu = 1,
  vulkan = 2,
  opencl = 3
};

class Device : public std::enable_shared_from_this<Device> {
 public:
  static Result<std::shared_ptr<Device>> cpu();
  static Result<std::shared_ptr<Device>> vulkan();
  static Result<std::shared_ptr<Device>> open(std::string_view name);
  virtual DeviceKind kind() const noexcept = 0;
  virtual void synchronize() = 0;
  virtual ~Device() = default;

 protected:
  Device() = default;
};

}  // namespace gyre
