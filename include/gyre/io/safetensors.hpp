#pragma once

#include "gyre/tensor.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace gyre {

struct SafetensorInfo {
  std::string name;
  DType dtype{DType::f32};
  std::vector<std::int64_t> shape;
  std::uint64_t data_begin{0};  // offset from start of data section
  std::uint64_t data_end{0};
};

struct SafetensorFile {
  std::filesystem::path path;
  std::uint64_t header_len{0};
  std::uint64_t data_offset{0};  // 8 + header_len
  std::vector<SafetensorInfo> tensors;
};

Result<SafetensorFile> safetensors_open(const std::filesystem::path& path);
Result<SafetensorFile> safetensors_open_header(std::string_view header_json, std::uint64_t header_len,
                                               const std::filesystem::path& path = {});

// Heap copy of one tensor (bf16/f16 stay packed). Use to_f32() for compute.
Result<Tensor> safetensors_load(const SafetensorFile& file, std::string_view name,
                                std::shared_ptr<Device> device, bool mmap = false);

Result<Tensor> concat_tp(std::span<const Tensor> shards, int axis);

DType safetensors_dtype(std::string_view s);
const char* safetensors_dtype_str(DType d);

struct NamedTensor {
  std::string name;
  const Tensor* t{nullptr};
};
Result<void> safetensors_save(const std::filesystem::path& path, std::span<const NamedTensor> tensors);

}  // namespace gyre
