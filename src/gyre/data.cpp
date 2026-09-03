#include "gyre/data.hpp"

namespace gyre {

Result<CharDataset> CharDataset::from_ids(std::vector<std::int32_t> ids, std::shared_ptr<Device> d) {
  CharDataset ds;
  ds.ids_ = std::move(ids);
  ds.device_ = std::move(d);
  if (ds.ids_.size() < 2) return std::unexpected(make_error(Errc::invalid_shape, "text too short"));
  return ds;
}

Result<std::pair<Tensor, Tensor>> CharDataset::sample(std::uint32_t batch, std::uint32_t block, Rng& rng) {
  if (ids_.size() <= block) return std::unexpected(make_error(Errc::invalid_shape, "block >= n"));
  std::int64_t xsh[2] = {batch, block};
  auto x = Tensor::empty(xsh, DType::i32, device_);
  auto y = Tensor::empty(xsh, DType::i32, device_);
  if (!x || !y) return std::unexpected(x ? y.error() : x.error());
  auto xp = x->host_span<std::int32_t>();
  auto yp = y->host_span<std::int32_t>();
  if (!xp || !yp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto max_start = static_cast<std::uint32_t>(ids_.size() - block - 1);
  for (std::uint32_t b = 0; b < batch; ++b) {
    auto s = rng.u32(max_start + 1);
    for (std::uint32_t t = 0; t < block; ++t) {
      (*xp)[b * block + t] = ids_[s + t];
      (*yp)[b * block + t] = ids_[s + t + 1];
    }
  }
  return std::make_pair(std::move(*x), std::move(*y));
}

}  // namespace gyre
