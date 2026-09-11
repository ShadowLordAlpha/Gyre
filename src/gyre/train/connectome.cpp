#include "gyre/train/connectome.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>

namespace gyre {
namespace {

Result<std::vector<float>> f32_host(const Tensor& t) {
  const Tensor* tp = &t;
  std::optional<Tensor> cpu;
  if (t.device() && t.device()->kind() != DeviceKind::cpu) {
    auto cdev = Device::cpu();
    if (!cdev) return std::unexpected(cdev.error());
    auto c = t.to(*cdev);
    if (!c) return std::unexpected(c.error());
    cpu = std::move(*c);
    tp = &*cpu;
  }
  auto s = tp->host_span<float>();
  if (!s) return std::unexpected(s.error());
  return std::vector<float>(s->begin(), s->end());
}

Result<void> write_f32(Tensor& t, std::span<const float> src) {
  if (static_cast<std::size_t>(t.numel()) != src.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "write_f32 size"));
  }
  return t.copy_from_host(std::as_bytes(src));
}

void add_in_axis(std::vector<float>& score, std::span<const float> w, std::int64_t in, std::int64_t out) {
  if (static_cast<std::int64_t>(score.size()) != in) return;
  for (std::int64_t i = 0; i < in; ++i) {
    float s = 0;
    for (std::int64_t j = 0; j < out; ++j) s += std::fabs(w[static_cast<std::size_t>(i * out + j)]);
    score[static_cast<std::size_t>(i)] += s;
  }
}

void add_out_axis(std::vector<float>& score, std::span<const float> w, std::int64_t in, std::int64_t out) {
  if (static_cast<std::int64_t>(score.size()) != out) return;
  for (std::int64_t i = 0; i < in; ++i) {
    for (std::int64_t j = 0; j < out; ++j)
      score[static_cast<std::size_t>(j)] += std::fabs(w[static_cast<std::size_t>(i * out + j)]);
  }
}

void add_1d(std::vector<float>& score, std::span<const float> w) {
  if (score.size() != w.size()) return;
  for (std::size_t i = 0; i < w.size(); ++i) score[i] += std::fabs(w[i]);
}

std::int64_t keep_width(std::int64_t n, float prune, std::int64_t align, std::int64_t minv) {
  auto k = static_cast<std::int64_t>(std::floor(static_cast<double>(n) * (1.0 - static_cast<double>(prune))));
  if (k < minv) k = minv;
  if (align > 1) k = (k / align) * align;
  if (k < minv) {
    k = ((minv + align - 1) / align) * align;
  }
  if (k >= n && n > minv && align <= n - minv) k = n - align;
  if (k < minv) k = minv;
  if (k > n) k = n;
  return k;
}

std::vector<std::int64_t> keep_top(const std::vector<float>& score, std::int64_t k, bool shuffle, Rng& rng) {
  const auto n = static_cast<std::int64_t>(score.size());
  std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  if (shuffle) {
    for (std::int64_t i = n - 1; i > 0; --i) {
      auto j = static_cast<std::int64_t>(rng.u32(static_cast<std::uint32_t>(i + 1)));
      std::swap(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
    }
  } else {
    std::stable_sort(idx.begin(), idx.end(), [&](std::int64_t a, std::int64_t b) {
      if (score[static_cast<std::size_t>(a)] != score[static_cast<std::size_t>(b)])
        return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)];
      return a < b;
    });
  }
  if (k > n) k = n;
  idx.resize(static_cast<std::size_t>(k));
  std::sort(idx.begin(), idx.end());
  return idx;
}

std::vector<float> gather_2d(std::span<const float> src, std::int64_t in, std::int64_t out,
                             std::span<const std::int64_t> keep_in, std::span<const std::int64_t> keep_out) {
  std::vector<float> dst(keep_in.size() * keep_out.size());
  const auto no = static_cast<std::int64_t>(keep_out.size());
  for (std::size_t a = 0; a < keep_in.size(); ++a) {
    for (std::size_t b = 0; b < keep_out.size(); ++b) {
      dst[a * static_cast<std::size_t>(no) + b] =
          src[static_cast<std::size_t>(keep_in[a] * out + keep_out[b])];
    }
  }
  (void)in;
  return dst;
}

std::vector<float> gather_1d(std::span<const float> src, std::span<const std::int64_t> keep) {
  std::vector<float> dst(keep.size());
  for (std::size_t i = 0; i < keep.size(); ++i) dst[i] = src[static_cast<std::size_t>(keep[i])];
  return dst;
}

void sign_collapse(std::vector<float>& w) {
  if (w.empty()) return;
  double acc = 0;
  for (auto v : w) acc += std::fabs(static_cast<double>(v));
  const float scale = static_cast<float>(acc / static_cast<double>(w.size()));
  for (auto& v : w) v = std::copysign(scale, v);
}

std::unordered_map<std::string, Tensor*> by_name(CharLM& m) {
  std::unordered_map<std::string, Tensor*> map;
  auto names = m.param_names();
  auto ps = m.parameters();
  for (std::size_t i = 0; i < names.size() && i < ps.size(); ++i) map[names[i]] = &ps[i].value;
  return map;
}

Result<std::vector<float>> req(const std::unordered_map<std::string, Tensor*>& m, const std::string& n) {
  auto it = m.find(n);
  if (it == m.end()) return std::unexpected(make_error(Errc::ckpt_corrupt, "missing " + n));
  return f32_host(*it->second);
}

}  // namespace

std::uint64_t param_count(const Module& m) {
  std::uint64_t n = 0;
  for (auto& p : const_cast<Module&>(m).parameters()) n += static_cast<std::uint64_t>(p.value.numel());
  return n;
}

Result<CharLM> prune_compact_charlm(CharLM& src, float prune, bool shuffle, Rng& rng,
                                    std::shared_ptr<Device> device) {
  if (!(prune > 0.f && prune < 1.f)) {
    return std::unexpected(make_error(Errc::unsupported, "--prune must be in (0, 1)"));
  }
  auto cfg = src.config();
  auto smap = by_name(src);

  std::vector<float> dscore(static_cast<std::size_t>(cfg.d_model), 0.f);
  auto add_d = [&](const std::string& name, bool as_in, bool as_out, bool as_1d) -> Result<void> {
    auto w = req(smap, name);
    if (!w) return std::unexpected(w.error());
    auto it = smap.find(name);
    auto sh = it->second->shape();
    if (as_1d && sh.size() == 1) add_1d(dscore, *w);
    if (sh.size() == 2) {
      const auto in = sh[0], out = sh[1];
      if (as_in) add_in_axis(dscore, *w, in, out);
      if (as_out) add_out_axis(dscore, *w, in, out);
    }
    return {};
  };

  if (auto r = add_d("wte.weight", false, true, false); !r) return std::unexpected(r.error());
  if (auto r = add_d("wpe.weight", false, true, false); !r) return std::unexpected(r.error());
  if (auto r = add_d("ln_f.weight", false, false, true); !r) return std::unexpected(r.error());
  if (auto r = add_d("ln_f.bias", false, false, true); !r) return std::unexpected(r.error());
  if (auto r = add_d("lm_head.weight", true, false, false); !r) return std::unexpected(r.error());
  for (std::int64_t i = 0; i < cfg.n_layer; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    for (auto s : {p + "ln1.weight", p + "ln1.bias", p + "ln2.weight", p + "ln2.bias"}) {
      if (auto r = add_d(s, false, false, true); !r) return std::unexpected(r.error());
    }
    for (auto s : {p + "attn.q.weight", p + "attn.k.weight", p + "attn.v.weight", p + "attn.o.weight"}) {
      if (auto r = add_d(s, true, true, false); !r) return std::unexpected(r.error());
    }
    for (auto s : {p + "attn.q.bias", p + "attn.k.bias", p + "attn.v.bias", p + "attn.o.bias",
                   p + "mlp.fc2.bias"}) {
      if (auto r = add_d(s, false, false, true); !r) return std::unexpected(r.error());
    }
    if (auto r = add_d(p + "mlp.fc1.weight", true, false, false); !r) return std::unexpected(r.error());
    if (auto r = add_d(p + "mlp.fc2.weight", false, true, false); !r) return std::unexpected(r.error());
  }

  const auto kd = keep_width(cfg.d_model, prune, cfg.n_head, cfg.n_head);
  if (kd == cfg.d_model && cfg.d_ff == keep_width(cfg.d_ff, prune, 1, 1)) {
    return std::unexpected(make_error(Errc::unsupported, "prune does not shrink width (already minimum)"));
  }
  auto keep_d = keep_top(dscore, kd, shuffle, rng);

  const auto new_ff = keep_width(cfg.d_ff, prune, 1, std::max<std::int64_t>(1, cfg.n_head));
  std::vector<std::vector<std::int64_t>> keep_ff(static_cast<std::size_t>(cfg.n_layer));
  for (std::int64_t i = 0; i < cfg.n_layer; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    std::vector<float> fs(static_cast<std::size_t>(cfg.d_ff), 0.f);
    auto w1 = req(smap, p + "mlp.fc1.weight");
    auto b1 = req(smap, p + "mlp.fc1.bias");
    auto w2 = req(smap, p + "mlp.fc2.weight");
    if (!w1 || !b1 || !w2) return std::unexpected(w1 ? (b1 ? w2.error() : b1.error()) : w1.error());
    add_out_axis(fs, *w1, cfg.d_model, cfg.d_ff);
    add_1d(fs, *b1);
    add_in_axis(fs, *w2, cfg.d_ff, cfg.d_model);
    keep_ff[static_cast<std::size_t>(i)] = keep_top(fs, new_ff, shuffle, rng);
  }

  CharLMConfig dst_cfg = cfg;
  dst_cfg.d_model = kd;
  dst_cfg.d_ff = new_ff;
  auto dst = CharLM::create(dst_cfg, device, rng);
  if (!dst) return std::unexpected(dst.error());
  auto dmap = by_name(*dst);

  auto put = [&](const std::string& name, std::vector<float> v, bool collapse) -> Result<void> {
    auto it = dmap.find(name);
    if (it == dmap.end()) return std::unexpected(make_error(Errc::ckpt_corrupt, "dst " + name));
    if (collapse && it->second->rank() >= 2) sign_collapse(v);
    return write_f32(*it->second, v);
  };

  auto iota_n = [](std::int64_t n) {
    std::vector<std::int64_t> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), 0);
    return v;
  };

  {
    auto w = req(smap, "wte.weight");
    if (!w) return std::unexpected(w.error());
    auto rows = iota_n(cfg.vocab);
    if (auto r = put("wte.weight", gather_2d(*w, cfg.vocab, cfg.d_model, rows, keep_d), true); !r)
      return std::unexpected(r.error());
  }
  {
    auto w = req(smap, "wpe.weight");
    if (!w) return std::unexpected(w.error());
    auto rows = iota_n(cfg.block_size);
    if (auto r = put("wpe.weight", gather_2d(*w, cfg.block_size, cfg.d_model, rows, keep_d), true); !r)
      return std::unexpected(r.error());
  }
  for (auto s : {"ln_f.weight", "ln_f.bias"}) {
    auto w = req(smap, s);
    if (!w) return std::unexpected(w.error());
    if (auto r = put(s, gather_1d(*w, keep_d), false); !r) return std::unexpected(r.error());
  }
  {
    auto w = req(smap, "lm_head.weight");
    if (!w) return std::unexpected(w.error());
    auto cols = iota_n(cfg.vocab);
    if (auto r = put("lm_head.weight", gather_2d(*w, cfg.d_model, cfg.vocab, keep_d, cols), true); !r)
      return std::unexpected(r.error());
    auto b = req(smap, "lm_head.bias");
    if (!b) return std::unexpected(b.error());
    if (auto r = put("lm_head.bias", *b, false); !r) return std::unexpected(r.error());
  }

  for (std::int64_t i = 0; i < cfg.n_layer; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    const auto& kf = keep_ff[static_cast<std::size_t>(i)];
    for (auto s : {p + "ln1.weight", p + "ln1.bias", p + "ln2.weight", p + "ln2.bias"}) {
      auto w = req(smap, s);
      if (!w) return std::unexpected(w.error());
      if (auto r = put(s, gather_1d(*w, keep_d), false); !r) return std::unexpected(r.error());
    }
    for (auto s : {p + "attn.q.weight", p + "attn.k.weight", p + "attn.v.weight", p + "attn.o.weight"}) {
      auto w = req(smap, s);
      if (!w) return std::unexpected(w.error());
      if (auto r = put(s, gather_2d(*w, cfg.d_model, cfg.d_model, keep_d, keep_d), true); !r)
        return std::unexpected(r.error());
    }
    for (auto s : {p + "attn.q.bias", p + "attn.k.bias", p + "attn.v.bias", p + "attn.o.bias",
                   p + "mlp.fc2.bias"}) {
      auto w = req(smap, s);
      if (!w) return std::unexpected(w.error());
      if (auto r = put(s, gather_1d(*w, keep_d), false); !r) return std::unexpected(r.error());
    }
    {
      auto w = req(smap, p + "mlp.fc1.weight");
      if (!w) return std::unexpected(w.error());
      if (auto r = put(p + "mlp.fc1.weight", gather_2d(*w, cfg.d_model, cfg.d_ff, keep_d, kf), true); !r)
        return std::unexpected(r.error());
      auto b = req(smap, p + "mlp.fc1.bias");
      if (!b) return std::unexpected(b.error());
      if (auto r = put(p + "mlp.fc1.bias", gather_1d(*b, kf), false); !r) return std::unexpected(r.error());
    }
    {
      auto w = req(smap, p + "mlp.fc2.weight");
      if (!w) return std::unexpected(w.error());
      if (auto r = put(p + "mlp.fc2.weight", gather_2d(*w, cfg.d_ff, cfg.d_model, kf, keep_d), true); !r)
        return std::unexpected(r.error());
    }
  }

  return dst;
}

}  // namespace gyre
