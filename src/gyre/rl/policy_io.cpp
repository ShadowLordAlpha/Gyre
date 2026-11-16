#include "gyre/rl/policy_io.hpp"

#include "gyre/checkpoint.hpp"
#include "json_parse.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <optional>
#include <system_error>

namespace gyre {
namespace {

Error bad(const char* m) { return make_error(Errc::ckpt_corrupt, m); }

std::string config_to_json(const PolicyConfig& c) {
  nlohmann::json j;
  j["feat_dim"] = c.encoder.feat_dim;
  j["global_dim"] = c.encoder.global_dim;
  j["max_entities"] = c.encoder.max_entities;
  j["d_model"] = c.encoder.d_model;
  j["n_layer"] = c.encoder.n_layer;
  j["n_head"] = c.encoder.n_head;
  j["d_ff"] = c.encoder.d_ff;
  j["out_dim"] = c.encoder.out_dim;
  j["gru_hidden"] = c.gru_hidden;
  j["head_sizes"] = c.head_sizes;
  return j.dump();
}

Result<PolicyConfig> config_from_json(const std::string& s) {
  auto pj = parse_json(s);
  if (!pj) return std::unexpected(pj.error());
  const auto& j = *pj;
  try {
    PolicyConfig c;
    c.encoder.feat_dim = j.at("feat_dim").get<std::int64_t>();
    c.encoder.global_dim = j.at("global_dim").get<std::int64_t>();
    c.encoder.max_entities = j.at("max_entities").get<std::int64_t>();
    c.encoder.d_model = j.at("d_model").get<std::int64_t>();
    c.encoder.n_layer = j.at("n_layer").get<std::int64_t>();
    c.encoder.n_head = j.at("n_head").get<std::int64_t>();
    c.encoder.d_ff = j.at("d_ff").get<std::int64_t>();
    c.encoder.out_dim = j.at("out_dim").get<std::int64_t>();
    c.gru_hidden = j.at("gru_hidden").get<std::int64_t>();
    c.head_sizes = j.at("head_sizes").get<std::vector<std::int64_t>>();
    return c;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(bad((std::string("policy config: ") + e.what()).c_str()));
  }
}

// Overflow-detecting uint64 arithmetic (same as gyre-infer's loader).
struct Checked {
  bool ok{true};
  std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) ok = false;
    return a * b;
  }
  std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) ok = false;
    return a + b;
  }
};

// Float count of PolicyNet::parameters() for config `c`; the same formula as the
// expected-count check in infer/src/policy.cpp. Empty if a dim is negative or
// the count overflows.
std::optional<std::uint64_t> implied_param_count(const PolicyConfig& c) {
  const auto& e = c.encoder;
  for (std::int64_t v : {e.feat_dim, e.global_dim, e.d_model, e.n_layer, e.d_ff, e.out_dim, c.gru_hidden})
    if (v < 0) return std::nullopt;
  Checked k;
  std::uint64_t T = 0;
  for (auto s : c.head_sizes) {
    if (s < 0) return std::nullopt;
    T = k.add(T, static_cast<std::uint64_t>(s));
  }
  const std::uint64_t F = e.feat_dim, G = e.global_dim, d = e.d_model, ff = e.d_ff, O = e.out_dim,
                      H = c.gru_hidden, L = e.n_layer;
  const std::uint64_t head_in = H > 0 ? H : O;
  std::uint64_t n = k.add(k.mul(F, d), d);
  const std::uint64_t per_layer = k.add(k.add(k.mul(4, d), k.mul(4, k.add(k.mul(d, d), d))),
                                        k.add(k.add(k.mul(d, ff), ff), k.add(k.mul(ff, d), d)));
  n = k.add(n, k.mul(L, per_layer));
  n = k.add(n, k.add(k.mul(2, d), k.add(k.mul(k.add(k.mul(2, d), G), O), O)));
  if (H > 0)
    n = k.add(n, k.add(k.add(k.mul(O, k.mul(3, H)), k.mul(3, H)), k.add(k.mul(H, k.mul(3, H)), k.mul(3, H))));
  n = k.add(n, k.add(k.add(k.mul(head_in, T), T), k.add(head_in, 1)));
  if (!k.ok) return std::nullopt;
  return n;
}

// What gyre-infer's GPOL1 loader accepts (infer/src/policy.cpp, Policy::load_impl).
// Training accepts more (n_layer=0 skips the n_head/d_ff checks; max_entities is
// never checked), so the exporter must reject those here, not at the deploy site.
Result<void> check_gpol1_limits(const PolicyConfig& c) {
  constexpr std::int64_t kDim = 1 << 16, kLayers = 256, kHeads = 1024, kHeadSize = 1 << 20,
                         kTotalLogits = 1 << 22;
  const auto& e = c.encoder;
  auto in = [](std::int64_t v, std::int64_t lo, std::int64_t hi) { return v >= lo && v <= hi; };
  const bool ok = in(e.feat_dim, 1, kDim) && in(e.global_dim, 0, kDim) && in(e.max_entities, 1, kDim) &&
                  in(e.d_model, 1, kDim) && in(e.n_layer, 0, kLayers) && in(e.n_head, 1, kDim) &&
                  e.d_model % e.n_head == 0 && in(e.d_ff, 1, kDim) && in(e.out_dim, 1, kDim) &&
                  in(c.gru_hidden, 0, kDim) && in(static_cast<std::int64_t>(c.head_sizes.size()), 1, kHeads);
  if (!ok)
    return std::unexpected(make_error(
        Errc::unsupported,
        "policy config is outside what gyre-infer loads (GPOL1 needs feat_dim, max_entities, d_model, n_head, "
        "d_ff, out_dim >= 1, d_model % n_head == 0, every dim <= 65536, n_layer <= 256, 1..1024 heads)"));
  std::int64_t total = 0;
  for (auto s : c.head_sizes) {
    if (!in(s, 1, kHeadSize))
      return std::unexpected(make_error(Errc::unsupported, "GPOL1 head sizes must be 1..2^20"));
    total += s;
  }
  if (total > kTotalLogits) return std::unexpected(make_error(Errc::unsupported, "GPOL1: more than 2^22 logits"));
  return {};
}

std::uint32_t fnv1a(const unsigned char* p, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
  return h;
}

template <class T>
void put(std::ofstream& f, T v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

}  // namespace

Result<void> save_policy(const std::filesystem::path& path, PolicyNet& net, std::uint64_t train_step) {
  GyreDoc doc;
  doc.arch = "policy-net";
  doc.config_json = config_to_json(net.config());
  doc.train_step = train_step;
  return save_gyre(path, net.parameters(), nullptr, doc);
}

Result<PolicyNet> load_policy(const std::filesystem::path& path, std::shared_ptr<Device> device) {
  auto f = GyreFile::open(path);
  if (!f) return std::unexpected(f.error());
  if (f->doc().arch != "policy-net") return std::unexpected(bad("not a policy-net checkpoint"));
  auto cfg = config_from_json(f->doc().config_json);
  if (!cfg) return std::unexpected(cfg.error());
  {
    // Untrusted dims: check the parameter count they imply against the weights the
    // file actually stores BEFORE PolicyNet::create allocates (a corrupt d_ff of 1e5
    // would otherwise ask for tens of GB and throw bad_alloc).
    auto want = implied_param_count(*cfg);
    if (!want) return std::unexpected(bad("policy config: negative or overflowing dimensions"));
    std::uint64_t have = 0;
    for (const auto& t : f->doc().tensors) {
      if (t.role != TensorRole::weight) continue;
      std::uint64_t n = 1;
      for (auto s : t.shape) {
        if (s < 0) return std::unexpected(bad("policy tensor with a negative dim"));
        n = (s != 0 && n > UINT64_MAX / static_cast<std::uint64_t>(s)) ? UINT64_MAX : n * static_cast<std::uint64_t>(s);
      }
      have = (have > UINT64_MAX - n) ? UINT64_MAX : have + n;
    }
    if (*want != have)
      return std::unexpected(bad("policy config does not match the stored weight count"));
  }
  Rng rng(0);
  auto net = PolicyNet::create(*cfg, std::move(device), rng);
  if (!net) return std::unexpected(net.error());
  if (auto r = f->load_params(net->parameters(), nullptr); !r) return std::unexpected(r.error());
  return net;
}

Result<void> export_policy_flat(const std::filesystem::path& path, PolicyNet& net) {
  static_assert(std::endian::native == std::endian::little, "flat export is little-endian");
  const auto& c = net.config();
  // Must stay within what gyre-infer's loader accepts (and fit the uint32 header fields).
  if (auto ok = check_gpol1_limits(c); !ok) return std::unexpected(ok.error());
  std::vector<float> data;
  for (auto& p : net.parameters()) {
    auto v = p.value.host_span<float>();
    if (!v) return std::unexpected(v.error());
    data.insert(data.end(), v->begin(), v->end());
  }
  if (auto want = implied_param_count(c); !want || *want != data.size())
    return std::unexpected(make_error(Errc::unsupported,"PolicyNet parameter count does not match the GPOL1 formula"));
  auto parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) return std::unexpected(make_error(Errc::io, "create directory " + parent.string() + ": " + ec.message()));
  }
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return std::unexpected(make_error(Errc::io, "open " + path.string()));
  f.write("GPOL1\0\0\0", 8);
  put<std::uint32_t>(f, 1);
  put<std::uint32_t>(f, 0);
  for (std::int64_t v : {c.encoder.feat_dim, c.encoder.global_dim, c.encoder.max_entities, c.encoder.d_model,
                         c.encoder.n_layer, c.encoder.n_head, c.encoder.d_ff, c.encoder.out_dim, c.gru_hidden,
                         static_cast<std::int64_t>(c.head_sizes.size())})
    put<std::uint32_t>(f, static_cast<std::uint32_t>(v));
  for (auto s : c.head_sizes) put<std::uint32_t>(f, static_cast<std::uint32_t>(s));
  put<std::uint64_t>(f, data.size());
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  put<std::uint32_t>(f, fnv1a(reinterpret_cast<const unsigned char*>(data.data()), data.size() * sizeof(float)));
  if (!f) return std::unexpected(make_error(Errc::io, "write " + path.string()));
  return {};
}

}  // namespace gyre
