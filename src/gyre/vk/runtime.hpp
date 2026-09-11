#pragma once

#include "gyre/device.hpp"
#include "gyre/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace gyre::vkrt {

enum class Pipe : std::uint32_t { elem = 0, gemm, row, idx, adam, count };

struct Bind {
  const Tensor* t{nullptr};
};

struct Push {
  std::uint32_t u[6]{};
  float f[6]{};
};

class VulkanDevice final : public Device {
 public:
  DeviceKind kind() const noexcept override { return DeviceKind::vulkan; }
  void synchronize() override;
  ~VulkanDevice() override;

  Result<void> init();
  Result<std::shared_ptr<Storage>> alloc(std::size_t bytes);
  Result<void> upload(Storage& st, std::size_t offset, std::span<const std::byte> src);
  Result<void> download(const Storage& st, std::size_t offset, std::span<std::byte> dst);
  Result<void> copy(Storage& dst, std::size_t dst_off, const Storage& src, std::size_t src_off,
                    std::size_t n);
  Result<void> fill_zero(Storage& st, std::size_t offset, std::size_t n);
  Result<void> dispatch(Pipe p, const std::array<Bind, 7>& binds, std::uint32_t gx, std::uint32_t gy,
                        std::uint32_t gz, const Push& pc);

  void destroy_alloc(VkBuffer b, VkDeviceMemory m) noexcept;
  void recycle_alloc(VkBuffer b, VkDeviceMemory m, VkDeviceSize cap) noexcept;

  static VulkanDevice* from(Device* d) noexcept;

  VkInstance instance{VK_NULL_HANDLE};
  VkPhysicalDevice phys{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  std::uint32_t qfamily{0};
  VkQueue queue{VK_NULL_HANDLE};
  VkCommandPool pool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  VkDescriptorSetLayout set_layout{VK_NULL_HANDLE};
  VkPipelineLayout pipe_layout{VK_NULL_HANDLE};
  VkDescriptorPool desc_pool{VK_NULL_HANDLE};
  VkDescriptorSet desc_set{VK_NULL_HANDLE};
  VkPipeline pipes[static_cast<int>(Pipe::count)]{};
  VkShaderModule shaders[static_cast<int>(Pipe::count)]{};
  VkPhysicalDeviceMemoryProperties mem{};
  VkBuffer dummy{VK_NULL_HANDLE};
  VkDeviceMemory dummy_mem{VK_NULL_HANDLE};
  VkBuffer staging{VK_NULL_HANDLE};
  VkDeviceMemory staging_mem{VK_NULL_HANDLE};
  void* staging_ptr{nullptr};
  VkDeviceSize staging_size{0};

 private:
  struct PooledBuf {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize cap{0};
  };

  Result<void> make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                           VkBuffer& buf, VkDeviceMemory& mem, void** mapped);
  Result<void> ensure_staging(VkDeviceSize size);
  Result<void> submit_and_wait();
  uint32_t memory_index(uint32_t type_bits, VkMemoryPropertyFlags flags) const;
  void drain_pool() noexcept;

  std::mutex pool_mu;
  std::vector<PooledBuf> free_bufs;
  static constexpr std::size_t kMaxPooled = 512;
};

Result<std::shared_ptr<Device>> get_or_create();

inline bool is_vulkan(const Tensor& t) noexcept {
  return t.device() && t.device()->kind() == DeviceKind::vulkan;
}

}  // namespace gyre::vkrt
