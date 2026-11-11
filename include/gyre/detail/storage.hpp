#pragma once

#include "gyre/error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace gyre {

// Recycling pool for large CPU buffers: fresh multi-MB allocations page-fault on every use, which
// dominated the cost of elementwise ops on attention-sized tensors. Buffers >= 1 MiB are returned
// here by ~Storage and reused. `zero=false` leaves recycled contents unspecified.
std::vector<std::byte> pool_acquire(std::size_t bytes, bool zero);
void pool_release(std::vector<std::byte>&& buf);

struct GpuAlloc {
  virtual ~GpuAlloc() = default;
  virtual std::size_t size() const noexcept = 0;
};

struct Storage {
  std::vector<std::byte> heap;
  std::byte* mapped{nullptr};
  std::size_t mapped_len{0};
#ifdef _WIN32
  void* win_file{nullptr};
  void* win_map{nullptr};
#else
  int posix_fd{-1};
#endif

  std::unique_ptr<GpuAlloc> gpu;

  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  ~Storage();

  std::byte* data() noexcept { return mapped ? mapped : heap.data(); }
  const std::byte* data() const noexcept { return mapped ? mapped : heap.data(); }
  std::size_t size() const noexcept {
    if (gpu) return gpu->size();
    return mapped ? mapped_len : heap.size();
  }

  static Result<std::shared_ptr<Storage>> mmap_file(const std::filesystem::path& path);
};

}  // namespace gyre
