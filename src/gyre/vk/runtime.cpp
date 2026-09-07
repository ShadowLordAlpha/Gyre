#include "vk/runtime.hpp"

#include "adam.spv.hpp"
#include "elem.spv.hpp"
#include "gemm.spv.hpp"
#include "idx.spv.hpp"
#include "row.spv.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace gyre::vkrt {
namespace {

Error vkerr(VkResult r, const char* what) {
  return make_error(Errc::unsupported, std::string(what) + " vk=" + std::to_string(static_cast<int>(r)));
}

struct VkGpuAlloc final : GpuAlloc {
  std::shared_ptr<VulkanDevice> owner;
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkDeviceSize nbytes{0};
  std::size_t size() const noexcept override { return static_cast<std::size_t>(nbytes); }
  ~VkGpuAlloc() override {
    if (owner) owner->destroy_alloc(buffer, memory);
  }
};

VkGpuAlloc* gpu(const Storage& st) { return static_cast<VkGpuAlloc*>(st.gpu.get()); }

}  // namespace

VulkanDevice* VulkanDevice::from(Device* d) noexcept {
  if (!d || d->kind() != DeviceKind::vulkan) return nullptr;
  return static_cast<VulkanDevice*>(d);
}

uint32_t VulkanDevice::memory_index(uint32_t type_bits, VkMemoryPropertyFlags flags) const {
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags) return i;
  }
  return UINT32_MAX;
}

Result<void> VulkanDevice::make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags props, VkBuffer& buf,
                                       VkDeviceMemory& mem_out, void** mapped) {
  if (size < 16) size = 16;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult r = vkCreateBuffer(device, &bi, nullptr, &buf);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateBuffer"));
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, buf, &req);
  uint32_t mi = memory_index(req.memoryTypeBits, props);
  if (mi == UINT32_MAX) {
    vkDestroyBuffer(device, buf, nullptr);
    buf = VK_NULL_HANDLE;
    return std::unexpected(make_error(Errc::unsupported, "no matching Vulkan memory type"));
  }
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mi;
  r = vkAllocateMemory(device, &ai, nullptr, &mem_out);
  if (r != VK_SUCCESS) {
    vkDestroyBuffer(device, buf, nullptr);
    buf = VK_NULL_HANDLE;
    return std::unexpected(vkerr(r, "vkAllocateMemory"));
  }
  r = vkBindBufferMemory(device, buf, mem_out, 0);
  if (r != VK_SUCCESS) {
    vkFreeMemory(device, mem_out, nullptr);
    vkDestroyBuffer(device, buf, nullptr);
    buf = VK_NULL_HANDLE;
    mem_out = VK_NULL_HANDLE;
    return std::unexpected(vkerr(r, "vkBindBufferMemory"));
  }
  if (mapped) {
    r = vkMapMemory(device, mem_out, 0, size, 0, mapped);
    if (r != VK_SUCCESS) {
      vkFreeMemory(device, mem_out, nullptr);
      vkDestroyBuffer(device, buf, nullptr);
      buf = VK_NULL_HANDLE;
      mem_out = VK_NULL_HANDLE;
      return std::unexpected(vkerr(r, "vkMapMemory"));
    }
  }
  return {};
}

Result<void> VulkanDevice::ensure_staging(VkDeviceSize size) {
  if (staging && staging_size >= size) return {};
  if (staging_ptr && staging_mem) vkUnmapMemory(device, staging_mem);
  if (staging) vkDestroyBuffer(device, staging, nullptr);
  if (staging_mem) vkFreeMemory(device, staging_mem, nullptr);
  staging = VK_NULL_HANDLE;
  staging_mem = VK_NULL_HANDLE;
  staging_ptr = nullptr;
  staging_size = 0;
  auto r = make_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       staging, staging_mem, &staging_ptr);
  if (!r) return r;
  staging_size = std::max<VkDeviceSize>(size, 16);
  return {};
}

Result<void> VulkanDevice::submit_and_wait() {
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkResetFences(device, 1, &fence);
  VkResult r = vkQueueSubmit(queue, 1, &si, fence);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkQueueSubmit"));
  r = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkWaitForFences"));
  return {};
}

void VulkanDevice::synchronize() {
  if (queue) vkQueueWaitIdle(queue);
}

void VulkanDevice::destroy_alloc(VkBuffer b, VkDeviceMemory m) noexcept {
  if (!device) return;
  if (b) vkDestroyBuffer(device, b, nullptr);
  if (m) vkFreeMemory(device, m, nullptr);
}

VulkanDevice::~VulkanDevice() {
  if (queue) vkQueueWaitIdle(queue);
  if (device) {
    for (auto& p : pipes) {
      if (p) vkDestroyPipeline(device, p, nullptr);
      p = VK_NULL_HANDLE;
    }
    for (auto& s : shaders) {
      if (s) vkDestroyShaderModule(device, s, nullptr);
      s = VK_NULL_HANDLE;
    }
    if (desc_pool) vkDestroyDescriptorPool(device, desc_pool, nullptr);
    if (pipe_layout) vkDestroyPipelineLayout(device, pipe_layout, nullptr);
    if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    if (cmd && pool) vkFreeCommandBuffers(device, pool, 1, &cmd);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (staging_ptr && staging_mem) vkUnmapMemory(device, staging_mem);
    if (staging) vkDestroyBuffer(device, staging, nullptr);
    if (staging_mem) vkFreeMemory(device, staging_mem, nullptr);
    if (dummy) vkDestroyBuffer(device, dummy, nullptr);
    if (dummy_mem) vkFreeMemory(device, dummy_mem, nullptr);
    vkDestroyDevice(device, nullptr);
  }
  if (instance) vkDestroyInstance(instance, nullptr);
}

Result<void> VulkanDevice::init() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "Gyre";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ii.pApplicationInfo = &app;
  VkResult r = vkCreateInstance(&ii, nullptr, &instance);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateInstance"));

  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
  if (ndev == 0) return std::unexpected(make_error(Errc::unsupported, "no Vulkan physical device"));
  std::vector<VkPhysicalDevice> devs(ndev);
  vkEnumeratePhysicalDevices(instance, &ndev, devs.data());

  auto score = [&](VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    int s = 1;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) s += 100;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) s += 10;
    return s;
  };
  std::sort(devs.begin(), devs.end(), [&](auto a, auto b) { return score(a) > score(b); });

  bool found = false;
  for (auto pd : devs) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    for (uint32_t i = 0; i < nq; ++i) {
      if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        phys = pd;
        qfamily = i;
        found = true;
        break;
      }
    }
    if (found) break;
  }
  if (!found) return std::unexpected(make_error(Errc::unsupported, "no Vulkan compute queue"));

  float prio = 1.f;
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qi.queueFamilyIndex = qfamily;
  qi.queueCount = 1;
  qi.pQueuePriorities = &prio;
  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  di.queueCreateInfoCount = 1;
  di.pQueueCreateInfos = &qi;
  r = vkCreateDevice(phys, &di, nullptr, &device);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateDevice"));
  vkGetDeviceQueue(device, qfamily, 0, &queue);
  vkGetPhysicalDeviceMemoryProperties(phys, &mem);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = qfamily;
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  r = vkCreateCommandPool(device, &cpi, nullptr, &pool);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateCommandPool"));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  r = vkAllocateCommandBuffers(device, &cai, &cmd);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkAllocateCommandBuffers"));
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  r = vkCreateFence(device, &fi, nullptr, &fence);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateFence"));

  auto dummy_r = make_buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dummy, dummy_mem, nullptr);
  if (!dummy_r) {
    dummy_r = make_buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          dummy, dummy_mem, nullptr);
    if (!dummy_r) return dummy_r;
  }
  auto stg = ensure_staging(1 << 20);
  if (!stg) return stg;

  VkDescriptorSetLayoutBinding binds[7]{};
  for (int i = 0; i < 7; ++i) {
    binds[i].binding = static_cast<uint32_t>(i);
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = 7;
  sl.pBindings = binds;
  r = vkCreateDescriptorSetLayout(device, &sl, nullptr, &set_layout);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateDescriptorSetLayout"));
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 48;
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &set_layout;
  pl.pushConstantRangeCount = 1;
  pl.pPushConstantRanges = &pcr;
  r = vkCreatePipelineLayout(device, &pl, nullptr, &pipe_layout);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreatePipelineLayout"));

  auto make_pipe = [&](Pipe p, const uint32_t* code, std::size_t bytes) -> Result<void> {
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = bytes;
    sm.pCode = code;
    auto idx = static_cast<int>(p);
    VkResult rr = vkCreateShaderModule(device, &sm, nullptr, &shaders[idx]);
    if (rr != VK_SUCCESS) return std::unexpected(vkerr(rr, "vkCreateShaderModule"));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = shaders[idx];
    ci.stage.pName = "main";
    ci.layout = pipe_layout;
    rr = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipes[idx]);
    if (rr != VK_SUCCESS) return std::unexpected(vkerr(rr, "vkCreateComputePipelines"));
    return {};
  };
  if (auto e = make_pipe(Pipe::elem, gyre_elem_spv, sizeof(gyre_elem_spv)); !e) return e;
  if (auto e = make_pipe(Pipe::gemm, gyre_gemm_spv, sizeof(gyre_gemm_spv)); !e) return e;
  if (auto e = make_pipe(Pipe::row, gyre_row_spv, sizeof(gyre_row_spv)); !e) return e;
  if (auto e = make_pipe(Pipe::idx, gyre_idx_spv, sizeof(gyre_idx_spv)); !e) return e;
  if (auto e = make_pipe(Pipe::adam, gyre_adam_spv, sizeof(gyre_adam_spv)); !e) return e;

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 7;
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  r = vkCreateDescriptorPool(device, &dpi, nullptr, &desc_pool);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkCreateDescriptorPool"));
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = desc_pool;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &set_layout;
  r = vkAllocateDescriptorSets(device, &dai, &desc_set);
  if (r != VK_SUCCESS) return std::unexpected(vkerr(r, "vkAllocateDescriptorSets"));
  return {};
}

Result<std::shared_ptr<Storage>> VulkanDevice::alloc(std::size_t bytes) {
  auto st = std::make_shared<Storage>();
  auto g = std::make_unique<VkGpuAlloc>();
  g->owner = std::static_pointer_cast<VulkanDevice>(shared_from_this());
  VkDeviceSize n = bytes < 16 ? 16 : static_cast<VkDeviceSize>((bytes + 15) & ~std::size_t{15});
  auto r = make_buffer(n,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g->buffer, g->memory, nullptr);
  if (!r) {
    r = make_buffer(n,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    g->buffer, g->memory, nullptr);
    if (!r) return std::unexpected(r.error());
  }
  g->nbytes = n;
  st->gpu = std::move(g);
  return st;
}

Result<void> VulkanDevice::upload(Storage& st, std::size_t offset, std::span<const std::byte> src) {
  if (src.empty()) return {};
  auto* g = gpu(st);
  if (!g) return std::unexpected(make_error(Errc::unsupported, "not a Vulkan buffer"));
  auto er = ensure_staging(src.size());
  if (!er) return er;
  std::memcpy(staging_ptr, src.data(), src.size());
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  VkBufferCopy cp{};
  cp.srcOffset = 0;
  cp.dstOffset = offset;
  cp.size = src.size();
  vkCmdCopyBuffer(cmd, staging, g->buffer, 1, &cp);
  vkEndCommandBuffer(cmd);
  return submit_and_wait();
}

Result<void> VulkanDevice::download(const Storage& st, std::size_t offset, std::span<std::byte> dst) {
  if (dst.empty()) return {};
  auto* g = gpu(st);
  if (!g) return std::unexpected(make_error(Errc::unsupported, "not a Vulkan buffer"));
  auto er = ensure_staging(dst.size());
  if (!er) return er;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  VkBufferCopy cp{};
  cp.srcOffset = offset;
  cp.dstOffset = 0;
  cp.size = dst.size();
  vkCmdCopyBuffer(cmd, g->buffer, staging, 1, &cp);
  vkEndCommandBuffer(cmd);
  auto s = submit_and_wait();
  if (!s) return s;
  std::memcpy(dst.data(), staging_ptr, dst.size());
  return {};
}

Result<void> VulkanDevice::copy(Storage& dst, std::size_t dst_off, const Storage& src,
                               std::size_t src_off, std::size_t n) {
  if (n == 0) return {};
  auto* d = gpu(dst);
  auto* s = gpu(src);
  if (!d || !s) return std::unexpected(make_error(Errc::unsupported, "not a Vulkan buffer"));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  VkBufferCopy cp{};
  cp.srcOffset = src_off;
  cp.dstOffset = dst_off;
  cp.size = n;
  vkCmdCopyBuffer(cmd, s->buffer, d->buffer, 1, &cp);
  vkEndCommandBuffer(cmd);
  return submit_and_wait();
}

Result<void> VulkanDevice::fill_zero(Storage& st, std::size_t offset, std::size_t n) {
  auto* g = gpu(st);
  if (!g) return std::unexpected(make_error(Errc::unsupported, "not a Vulkan buffer"));
  std::size_t aligned = n & ~std::size_t{3};
  std::size_t off = offset & ~std::size_t{3};
  if (aligned == 0) {
    std::vector<std::byte> z(n);
    return upload(st, offset, z);
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  vkCmdFillBuffer(cmd, g->buffer, off, aligned, 0);
  vkEndCommandBuffer(cmd);
  return submit_and_wait();
}

Result<void> VulkanDevice::dispatch(Pipe p, const std::array<Bind, 7>& binds, std::uint32_t gx,
                                    std::uint32_t gy, std::uint32_t gz, const Push& pc) {
  if (gx == 0) gx = 1;
  if (gy == 0) gy = 1;
  if (gz == 0) gz = 1;
  VkDescriptorBufferInfo infos[7]{};
  VkWriteDescriptorSet writes[7]{};
  for (int i = 0; i < 7; ++i) {
    infos[i].buffer = dummy;
    infos[i].offset = 0;
    infos[i].range = 16;
    if (binds[i].t && binds[i].t->storage() && binds[i].t->storage()->gpu) {
      auto* g = gpu(*binds[i].t->storage());
      infos[i].buffer = g->buffer;
      infos[i].offset = binds[i].t->byte_offset();
      infos[i].range = std::max<VkDeviceSize>(binds[i].t->nbytes(), 4);
    }
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = desc_set;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[static_cast<int>(p)]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0,
                          nullptr);
  vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 48, &pc);
  vkCmdDispatch(cmd, gx, gy, gz);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &mb, 0, nullptr, 0, nullptr);
  vkEndCommandBuffer(cmd);
  return submit_and_wait();
}

Result<std::shared_ptr<Device>> get_or_create() {
  static std::mutex mu;
  static std::shared_ptr<Device> inst;
  static std::string fail;
  std::lock_guard<std::mutex> lock(mu);
  if (inst) return inst;
  if (!fail.empty()) return std::unexpected(make_error(Errc::unsupported, fail));
  auto d = std::make_shared<VulkanDevice>();
  auto r = d->init();
  if (!r) {
    fail = r.error().message;
    return std::unexpected(r.error());
  }
  inst = std::move(d);
  return inst;
}

}  // namespace gyre::vkrt

namespace gyre {

Result<std::shared_ptr<Device>> Device::vulkan() { return vkrt::get_or_create(); }

}  // namespace gyre
