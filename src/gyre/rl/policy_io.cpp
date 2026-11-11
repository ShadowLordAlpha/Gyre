#include "gyre/rl/policy_io.hpp"

#include "gyre/checkpoint.hpp"
#include "json_parse.hpp"

#include <bit>
#include <fstream>

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
  Rng rng(0);
  auto net = PolicyNet::create(*cfg, std::move(device), rng);
  if (!net) return std::unexpected(net.error());
  if (auto r = f->load_params(net->parameters(), nullptr); !r) return std::unexpected(r.error());
  return net;
}

Result<void> export_policy_flat(const std::filesystem::path& path, PolicyNet& net) {
  static_assert(std::endian::native == std::endian::little, "flat export is little-endian");
  const auto& c = net.config();
  {  // must stay within what gyre-infer's loader accepts (and fit the uint32 header fields)
    constexpr std::int64_t kDim = 1 << 16, kHead = 1 << 20;
    bool fits = c.encoder.feat_dim <= kDim && c.encoder.global_dim <= kDim && c.encoder.max_entities <= kDim &&
                c.encoder.d_model <= kDim && c.encoder.d_ff <= kDim && c.encoder.out_dim <= kDim &&
                c.encoder.n_head <= kDim && c.encoder.n_layer <= 256 && c.gru_hidden <= kDim &&
                c.head_sizes.size() <= 1024;
    std::int64_t total = 0;
    for (auto s : c.head_sizes) {
      fits = fits && s <= kHead;
      total += s;
    }
    if (!fits || total > (1 << 22))
      return std::unexpected(make_error(Errc::unsupported, "policy exceeds the limits of the GPOL1 format"));
  }
  std::vector<float> data;
  for (auto& p : net.parameters()) {
    auto v = p.value.host_span<float>();
    if (!v) return std::unexpected(v.error());
    data.insert(data.end(), v->begin(), v->end());
  }
  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
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
