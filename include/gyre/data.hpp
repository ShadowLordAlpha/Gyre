#pragma once

#include "gyre/nn/tokenize.hpp"
#include "gyre/rng.hpp"
#include "gyre/tensor.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace gyre {

class Dataset {
 public:
  virtual Result<std::pair<Tensor, Tensor>> sample(std::uint32_t batch, std::uint32_t block, Rng&) = 0;
  virtual ~Dataset() = default;
};

class CharDataset final : public Dataset {
 public:
  static Result<CharDataset> from_ids(std::vector<std::int32_t> ids, std::shared_ptr<Device> d);

  Result<std::pair<Tensor, Tensor>> sample(std::uint32_t batch, std::uint32_t block, Rng&) override;
  const std::vector<std::int32_t>& ids() const { return ids_; }

 private:
  std::vector<std::int32_t> ids_;
  std::shared_ptr<Device> device_;
};

}  // namespace gyre
