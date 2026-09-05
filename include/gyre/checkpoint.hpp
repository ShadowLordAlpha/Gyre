#pragma once

#include "gyre/io/codec.hpp"
#include "gyre/module.hpp"
#include "gyre/optim.hpp"
#include "gyre/rng.hpp"

#include "gyre/detail/storage.hpp"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace gyre {

enum class TensorRole : std::uint8_t { weight = 1, adam_m = 2, adam_v = 3 };

struct GyreTensorDesc {
  std::string name;
  DType dtype{DType::f32};
  std::vector<std::int64_t> shape;
  TensorRole role{TensorRole::weight};
  GyreCodec codec{GyreCodec::identity};
  std::uint64_t offset{0};  // bytes from payload start (.gyre)
  std::uint64_t nbytes{0};  // uncompressed
  std::uint64_t packed_bytes{0};
};

// One model document. Binary `.gyre` and text `.gyre.json` serialize this.
struct GyreDoc {
  std::string arch;           // "char-lm" | "grok2" | "linear" | ...
  std::string config_json;    // JSON object
  std::uint64_t train_step{0};
  std::uint64_t rng_seed{0};
  std::string tokenizer_json;  // JSON object (tokenizer fields only)
  std::vector<GyreTensorDesc> tensors;

  std::string to_json() const;
  static Result<GyreDoc> from_json(std::string_view json);

  bool recency_alibi() const;
  double holdout() const;
};

struct CheckpointMeta {
  std::uint64_t rng_seed{0};
  std::uint64_t train_step{0};
  std::string json;  // full Gyre document (or legacy trailer)
  std::vector<std::string> param_names;
};

class GyreFile {
 public:
  GyreFile() = default;
  GyreFile(GyreDoc doc, std::shared_ptr<Storage> storage, std::uint64_t payload_offset)
      : doc_(std::move(doc)), storage_(std::move(storage)), payload_offset_(payload_offset) {}

  static Result<GyreFile> open(const std::filesystem::path& path);

  const GyreDoc& doc() const { return doc_; }
  GyreDoc& doc() { return doc_; }
  void add_json_tensor(std::string name, Tensor t) { json_owned_.insert_or_assign(std::move(name), std::move(t)); }

  Result<Tensor> load_tensor(std::string_view name, std::shared_ptr<Device> device) const;
  Result<void> load_params(std::span<Param> params, Adam* adam = nullptr) const;

 private:
  GyreDoc doc_{};
  std::shared_ptr<Storage> storage_;
  std::uint64_t payload_offset_{0};
  std::unordered_map<std::string, Tensor> json_owned_;
};

// Skip inlining weight arrays into `.gyre.json` above this payload size.
constexpr std::uint64_t kGyreJsonDataLimit = 8ull << 20;

Result<void> save_gyre(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                       const GyreDoc& doc, std::span<const std::string> param_names = {},
                       GyreCodec codec = GyreCodec::identity);

Result<void> save_gyre_json(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                            const GyreDoc& doc, std::span<const std::string> param_names = {},
                            bool include_data = true);

// Header + document JSON only (no tensor payload).
Result<GyreDoc> peek_gyre(const std::filesystem::path& path);

Result<void> save_gyre1(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                        const CheckpointMeta& meta);
Result<void> load_gyre1(const std::filesystem::path& path, std::span<Param> params, Adam* adam,
                        CheckpointMeta& meta);
Result<CheckpointMeta> peek_gyre1(const std::filesystem::path& path);

}  // namespace gyre
