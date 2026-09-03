#include "gyre/device.hpp"

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

}  // namespace gyre
