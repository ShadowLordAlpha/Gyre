#include "gyre/adapt/agent.hpp"

#include "gyre/checkpoint.hpp"

#include <algorithm>
#include <span>

namespace gyre::adapt {

GridWorld::GridWorld() {
  auto d = Device::cpu();
  if (d) cpu_ = *d;
}

Result<Tensor> GridWorld::reset(Rng&) {
  x_ = 0;
  y_ = 0;
  if (!cpu_) {
    auto d = Device::cpu();
    if (!d) return std::unexpected(d.error());
    cpu_ = *d;
  }
  float obs[2] = {0.f, 0.f};
  std::int64_t sh[] = {2};
  return Tensor::from_host(std::as_bytes(std::span(obs, 2)), sh, DType::f32, cpu_);
}

Result<StepResult> GridWorld::step(const Action& a) {
  if (a.space != ActionSpace::discrete_int) {
    return std::unexpected(make_error(Errc::unsupported, "discrete only"));
  }
  switch (a.discrete) {
    case 0:
      y_ = std::min<std::int64_t>(y_ + 1, 4);
      break;
    case 1:
      y_ = std::max<std::int64_t>(y_ - 1, 0);
      break;
    case 2:
      x_ = std::min<std::int64_t>(x_ + 1, 4);
      break;
    case 3:
      x_ = std::max<std::int64_t>(x_ - 1, 0);
      break;
    default:
      break;
  }
  float obs[2] = {static_cast<float>(x_), static_cast<float>(y_)};
  std::int64_t sh[] = {2};
  auto t = Tensor::from_host(std::as_bytes(std::span(obs, 2)), sh, DType::f32, cpu_);
  if (!t) return std::unexpected(t.error());
  bool done = (x_ == 4 && y_ == 4);
  float r = done ? 1.f : -0.01f;
  return StepResult{std::move(*t), r, done};
}

std::span<const std::int64_t> GridWorld::observation_shape() const { return shape_; }

Result<PolicyAgent> PolicyAgent::create(std::int64_t obs_dim, std::int64_t n_actions,
                                        std::shared_ptr<Device> d, Rng& rng) {
  auto lin = Linear::create(obs_dim, n_actions, std::move(d), rng);
  if (!lin) return std::unexpected(lin.error());
  return PolicyAgent(std::move(*lin));
}

Result<Action> PolicyAgent::act(const Tensor& observation, Rng&) {
  ForwardCtx ctx;
  ctx.train = false;
  auto y = lin_.forward(observation, ctx);
  if (!y) return std::unexpected(y.error());
  auto p = y->host_span<float>();
  if (!p) return std::unexpected(p.error());
  std::int32_t best = 0;
  float m = (*p)[0];
  for (std::size_t i = 1; i < p->size(); ++i)
    if ((*p)[i] > m) {
      m = (*p)[i];
      best = static_cast<std::int32_t>(i);
    }
  return Action{ActionSpace::discrete_int, best, {}};
}

Result<void> PolicyAgent::load(const std::filesystem::path& p) {
  CheckpointMeta meta;
  return load_gyre1(p, lin_.parameters(), nullptr, meta);
}

Result<void> PolicyAgent::save(const std::filesystem::path& p) const {
  CheckpointMeta meta;
  return save_gyre1(p, const_cast<Linear&>(lin_).parameters(), nullptr, meta);
}

}  // namespace gyre::adapt
