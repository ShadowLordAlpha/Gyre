#pragma once

#include "gyre/error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace gyre {

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

  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  ~Storage();

  std::byte* data() noexcept { return mapped ? mapped : heap.data(); }
  const std::byte* data() const noexcept { return mapped ? mapped : heap.data(); }
  std::size_t size() const noexcept { return mapped ? mapped_len : heap.size(); }

  static Result<std::shared_ptr<Storage>> mmap_file(const std::filesystem::path& path);
};

}  // namespace gyre
