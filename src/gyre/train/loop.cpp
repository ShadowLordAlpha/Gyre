#include "gyre/train/loop.hpp"

#include "gyre/log.hpp"

namespace gyre {

Result<void> TrainLoop::run(Module& model, Dataset& data, const TrainConfig& cfg,
                            std::shared_ptr<Device> device, std::function<void(const Metrics&)> on_log,
                            std::function<void(const Metrics&)> on_progress) {
  (void)device;
  auto opt = Adam::create(model.parameters(), cfg.lr);
  if (!opt) return std::unexpected(opt.error());
  Rng rng(cfg.seed);
  for (std::uint32_t step = 1; step <= cfg.steps; ++step) {
    auto xy = data.sample(cfg.batch, cfg.block, rng);
    if (!xy) return std::unexpected(xy.error());
    auto zg = model.zero_grad();
    if (!zg) return zg;
    ForwardCtx ctx;
    ctx.train = true;
    auto logits = model.forward(xy->first, ctx);
    if (!logits) return std::unexpected(logits.error());
    auto loss = softmax_cross_entropy(*logits, xy->second);
    if (!loss) return std::unexpected(loss.error());
    auto bw = model.backward(loss->d_pred, ctx);
    if (!bw) return bw;
    opt->lr = scheduled_lr(cfg, step);
    auto st = opt->step(model.parameters());
    if (!st) return st;
    float loss_v = 0.f;
    if (auto lv = loss->value.host_span<float>()) loss_v = (*lv)[0];
    Metrics m{step, loss_v, opt->lr};
    if (on_progress) on_progress(m);
    if (cfg.log_every && step % cfg.log_every == 0) {
      if (on_log) on_log(m);
      log(LogLevel::info, "step loss logged");
    }
    if (cfg.ckpt_every && step % cfg.ckpt_every == 0) {
      CheckpointMeta meta{cfg.seed, step, cfg.ckpt_json, cfg.param_names};
      if (!cfg.ckpt_dir.empty()) {
        auto p = cfg.ckpt_dir / ("step-" + std::to_string(step) + ".gyre");
        auto s = save_gyre1(p, model.parameters(), &*opt, meta);
        if (!s) return s;
      }
      if (!cfg.ckpt_path.empty()) {
        auto s = save_gyre1(cfg.ckpt_path, model.parameters(), &*opt, meta);
        if (!s) return s;
      }
    }
  }
  return {};
}

}  // namespace gyre
