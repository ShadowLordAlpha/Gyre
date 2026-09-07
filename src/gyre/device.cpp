#include "gyre/device.hpp"

#include <string>

namespace gyre {
namespace {

class CpuDevice final : public Device {
 public:
  DeviceKind kind() const noexcept override { return DeviceKind::cpu; }
  void synchronize() override {}
};

}  // namespace

Result<std::shared_ptr<Device>> Device::cpu() {
  static auto inst = std::static_pointer_cast<Device>(std::make_shared<CpuDevice>());
  return inst;
}

#ifndef GYRE_VULKAN
Result<std::shared_ptr<Device>> Device::vulkan() {
  return std::unexpected(make_error(Errc::unsupported, "built without Vulkan (enable GYRE_ENABLE_VULKAN)"));
}
#endif

Result<std::shared_ptr<Device>> Device::open(std::string_view name) {
  if (name.empty() || name == "cpu") return cpu();
  if (name == "vulkan" || name == "gpu") return vulkan();
  return std::unexpected(make_error(Errc::unsupported, "unknown --device (cpu|vulkan)"));
}

}  // namespace gyre
