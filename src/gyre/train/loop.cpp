#include "gyre/train/loop.hpp"

#include "gyre/log.hpp"

#include <cmath>
#include <optional>
#include <string>

namespace gyre {

Result<void> TrainLoop::run(Module& model, Dataset& data, const TrainConfig& cfg,
                            std::shared_ptr<Device> device, std::function<void(const Metrics&)> on_log,
                            std::function<void(const Metrics&)> on_progress) {
  (void)device;
  std::optional<Adam> owned;
  Adam* opt = cfg.adam;
  if (!opt) {
    auto created = Adam::create(model.parameters(), cfg.lr);
    if (!created) return std::unexpected(created.error());
    owned = std::move(*created);
    opt = &*owned;
  }
  if (cfg.weight_decay > 0.f) opt->weight_decay = cfg.weight_decay;
  Rng rng(cfg.seed);
  if (cfg.steps == 0) return {};
  const std::uint32_t last = cfg.start_step + cfg.steps;
  for (std::uint32_t step = cfg.start_step + 1; step <= last; ++step) {
    auto xy = data.sample(cfg.batch, cfg.block, rng);
    if (!xy) return std::unexpected(xy.error());
    auto zg = model.zero_grad();
    if (!zg) return zg;
    ForwardCtx ctx;
    ctx.train = true;
    ctx.rng = &rng;
    auto logits = model.forward(xy->first, ctx);
    if (!logits) return std::unexpected(logits.error());
    auto loss = softmax_cross_entropy(*logits, xy->second);
    if (!loss) return std::unexpected(loss.error());
    auto item = loss->value.item_f32();
    if (!item) return std::unexpected(item.error());
    float loss_v = *item;
    if (!std::isfinite(loss_v)) {
      return std::unexpected(make_error(
          Errc::overflow, "non-finite loss at step " + std::to_string(step) +
                              " (logits overflowed or NaN). Weights were not updated. Resume from the "
                              "last step-*.gyre in the checkpoint directory."));
    }
    auto bw = model.backward(loss->d_pred, ctx);
    if (!bw) return bw;
    float gnorm = 0.f;
    if (cfg.grad_clip > 0.f) {
      auto cg = clip_grad_norm(model.parameters(), cfg.grad_clip);
      if (!cg) return std::unexpected(cg.error());
      gnorm = *cg;
    }
    opt->lr = scheduled_lr(cfg, step);
    auto st = opt->step(model.parameters());
    if (!st) return st;
    Metrics m{step, loss_v, opt->lr, gnorm};
    if (on_progress) on_progress(m);
    if (cfg.log_every && step % cfg.log_every == 0) {
      if (on_log) on_log(m);
      log(LogLevel::info, "step loss logged");
    }
    if (cfg.ckpt_every && step % cfg.ckpt_every == 0) {
      CheckpointMeta meta{cfg.seed, step, cfg.ckpt_json, cfg.param_names};
      if (!cfg.ckpt_dir.empty()) {
        auto p = cfg.ckpt_dir / ("step-" + std::to_string(step) + ".gyre");
        auto s = save_gyre1(p, model.parameters(), cfg.save_adam ? opt : nullptr, meta);
        if (!s) return s;
      }
      if (!cfg.ckpt_path.empty()) {
        auto s = save_gyre1(cfg.ckpt_path, model.parameters(), cfg.save_adam ? opt : nullptr, meta);
        if (!s) return s;
      }
    }
  }
  return {};
}

}  // namespace gyre
