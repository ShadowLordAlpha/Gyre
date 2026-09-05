#include "gyre/checkpoint.hpp"

#include "gyre/io/codec.hpp"
#include "json_parse.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <fstream>
#include <span>
#include <sstream>

namespace gyre {
namespace {

std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t crc = 0) {
  crc = ~crc;
  for (auto b : data) {
    crc ^= static_cast<std::uint8_t>(b);
    for (int i = 0; i < 8; ++i) {
      std::uint32_t mask = -(crc & 1u);
      crc = (crc >> 1) ^ (0x82F63B78u & mask);
    }
  }
  return ~crc;
}

std::uint32_t crc32c_update(std::uint32_t crc, std::span<const std::byte> data) {
  return crc32c(data, crc);
}

void put_u16(std::byte* p, std::uint16_t v) {
  p[0] = static_cast<std::byte>(v & 0xff);
  p[1] = static_cast<std::byte>((v >> 8) & 0xff);
}
void put_u32(std::byte* p, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<std::byte>((v >> (8 * i)) & 0xff);
}
void put_u64(std::byte* p, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<std::byte>((v >> (8 * i)) & 0xff);
}
std::uint32_t rd_u32(std::span<const std::byte> s, std::size_t off) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(s[off + i]) << (8 * i);
  return v;
}
std::uint64_t rd_u64(std::span<const std::byte> s, std::size_t off) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(s[off + i]) << (8 * i);
  return v;
}
std::uint16_t rd_u16(std::span<const std::byte> s, std::size_t off) {
  return static_cast<std::uint16_t>(static_cast<unsigned>(s[off]) |
                                    (static_cast<unsigned>(s[off + 1]) << 8));
}

std::size_t align64(std::size_t n) { return (n + 63) & ~std::size_t{63}; }

const char* dtype_name(DType d) {
  switch (d) {
    case DType::f32: return "f32";
    case DType::f16: return "f16";
    case DType::bf16: return "bf16";
    case DType::i32: return "i32";
    case DType::i8: return "i8";
    case DType::u8: return "u8";
  }
  return "f32";
}

Result<DType> dtype_from_name(std::string_view s) {
  if (s == "f32" || s == "F32") return DType::f32;
  if (s == "f16" || s == "F16") return DType::f16;
  if (s == "bf16" || s == "BF16") return DType::bf16;
  if (s == "i32" || s == "I32") return DType::i32;
  if (s == "i8" || s == "I8") return DType::i8;
  if (s == "u8" || s == "U8") return DType::u8;
  return std::unexpected(make_error(Errc::unsupported, "dtype " + std::string(s)));
}

const char* role_name(TensorRole r) {
  switch (r) {
    case TensorRole::weight: return "weight";
    case TensorRole::adam_m: return "adam_m";
    case TensorRole::adam_v: return "adam_v";
  }
  return "weight";
}

TensorRole role_from_name(std::string_view s) {
  if (s == "adam_m" || s == "m") return TensorRole::adam_m;
  if (s == "adam_v" || s == "v") return TensorRole::adam_v;
  return TensorRole::weight;
}

nlohmann::json shape_json(std::span<const std::int64_t> sh) {
  nlohmann::json a = nlohmann::json::array();
  for (auto d : sh) a.push_back(d);
  return a;
}

std::vector<std::int64_t> shape_from_json(const nlohmann::json& a) {
  std::vector<std::int64_t> sh;
  if (!a.is_array()) return sh;
  for (auto& x : a) sh.push_back(x.get<std::int64_t>());
  return sh;
}

std::uint64_t nbytes_of(DType dt, std::span<const std::int64_t> sh) {
  std::int64_t n = 1;
  for (auto d : sh) n *= d;
  return static_cast<std::uint64_t>(n) * dtype_size(dt);
}

struct NamedRef {
  std::string name;
  TensorRole role;
  const Tensor* t;
};

std::vector<NamedRef> collect_named(std::span<const Param> params, const Adam* adam,
                                    std::span<const std::string> param_names) {
  std::vector<NamedRef> out;
  for (std::size_t i = 0; i < params.size(); ++i) {
    std::string n = i < param_names.size() && !param_names[i].empty() ? std::string(param_names[i])
                                                                     : ("w:" + std::to_string(i));
    out.push_back({n, TensorRole::weight, &params[i].value});
    if (adam && i < adam->m.size()) {
      out.push_back({"m:" + n, TensorRole::adam_m, &adam->m[i]});
      out.push_back({"v:" + n, TensorRole::adam_v, &adam->v[i]});
    }
  }
  return out;
}

void apply_named_to_doc(GyreDoc& doc, const std::vector<NamedRef>& items) {
  doc.tensors.clear();
  std::uint64_t rel = 0;
  for (auto& it : items) {
    GyreTensorDesc d;
    d.name = it.name;
    d.dtype = it.t->dtype();
    d.shape.assign(it.t->shape().begin(), it.t->shape().end());
    d.role = it.role;
    d.nbytes = it.t->nbytes();
    d.offset = rel;
    rel = align64(static_cast<std::size_t>(rel + d.nbytes));
    doc.tensors.push_back(std::move(d));
  }
}

Result<void> copy_bytes_into(Tensor& t, std::span<const std::byte> src) {
  auto hb = t.host_bytes();
  if (!hb) return std::unexpected(hb.error());
  if (hb->size() != src.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "tensor size"));
  }
  std::memcpy(hb->data(), src.data(), src.size());
  return {};
}

Result<nlohmann::json> tensor_data_json(const Tensor& t) {
  nlohmann::json a = nlohmann::json::array();
  if (t.dtype() == DType::f32) {
    auto p = t.host_span<float>();
    if (!p) return std::unexpected(p.error());
    for (auto v : *p) a.push_back(v);
  } else if (t.dtype() == DType::i32) {
    auto p = t.host_span<std::int32_t>();
    if (!p) return std::unexpected(p.error());
    for (auto v : *p) a.push_back(v);
  } else if (t.dtype() == DType::u8) {
    auto p = t.host_span<std::uint8_t>();
    if (!p) return std::unexpected(p.error());
    for (auto v : *p) a.push_back(v);
  } else {
    return std::unexpected(make_error(Errc::unsupported, "json data dtype"));
  }
  return a;
}

Result<Tensor> tensor_from_data_json(const GyreTensorDesc& d, const nlohmann::json& data,
                                     std::shared_ptr<Device> device) {
  auto t = Tensor::empty(d.shape, d.dtype, device);
  if (!t) return t;
  if (!data.is_array()) return std::unexpected(make_error(Errc::ckpt_corrupt, "data array"));
  if (d.dtype == DType::f32) {
    auto p = t->host_span<float>();
    if (!p) return std::unexpected(p.error());
    if (data.size() != p->size()) return std::unexpected(make_error(Errc::invalid_shape, d.name));
    for (std::size_t i = 0; i < p->size(); ++i) (*p)[i] = data[i].get<float>();
  } else if (d.dtype == DType::i32) {
    auto p = t->host_span<std::int32_t>();
    if (!p) return std::unexpected(p.error());
    if (data.size() != p->size()) return std::unexpected(make_error(Errc::invalid_shape, d.name));
    for (std::size_t i = 0; i < p->size(); ++i) (*p)[i] = data[i].get<std::int32_t>();
  } else if (d.dtype == DType::u8) {
    auto p = t->host_span<std::uint8_t>();
    if (!p) return std::unexpected(p.error());
    if (data.size() != p->size()) return std::unexpected(make_error(Errc::invalid_shape, d.name));
    for (std::size_t i = 0; i < p->size(); ++i) (*p)[i] = data[i].get<std::uint8_t>();
  } else {
    return std::unexpected(make_error(Errc::unsupported, "json data dtype"));
  }
  return t;
}

Result<void> load_params_from_map(std::span<Param> params, Adam* adam,
                                  const std::function<Result<std::span<const std::byte>>(const std::string&)>& get,
                                  const GyreDoc& doc) {
  std::vector<const GyreTensorDesc*> weights, ms, vs;
  for (auto& t : doc.tensors) {
    if (t.role == TensorRole::weight) weights.push_back(&t);
    else if (t.role == TensorRole::adam_m) ms.push_back(&t);
    else if (t.role == TensorRole::adam_v) vs.push_back(&t);
  }
  if (weights.size() != params.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "weight count"));
  }
  for (std::size_t i = 0; i < params.size(); ++i) {
    auto b = get(weights[i]->name);
    if (!b) return std::unexpected(b.error());
    auto r = copy_bytes_into(params[i].value, *b);
    if (!r) return r;
    if (adam && i < adam->m.size()) {
      if (i >= ms.size() || i >= vs.size()) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "missing adam"));
      }
      auto mb = get(ms[i]->name);
      auto vb = get(vs[i]->name);
      if (!mb) return std::unexpected(mb.error());
      if (!vb) return std::unexpected(vb.error());
      auto r1 = copy_bytes_into(adam->m[i], *mb);
      if (!r1) return r1;
      auto r2 = copy_bytes_into(adam->v[i], *vb);
      if (!r2) return r2;
    }
  }
  return {};
}

Result<GyreFile> open_v1(const std::filesystem::path& path, std::span<const std::byte> file) {
  (void)path;
  if (file.size() < 64) return std::unexpected(make_error(Errc::ckpt_corrupt, "short"));
  auto sp = file;
  if (rd_u32(sp, 8) != 1) return std::unexpected(make_error(Errc::unsupported, "version"));
  auto flags = rd_u32(sp, 12);
  auto tc = rd_u64(sp, 16);
  auto heap_b = rd_u64(sp, 24);
  auto payload_off = rd_u64(sp, 32);
  auto trailer_off = rd_u64(sp, 40);
  if (rd_u32(sp, 52) != 1) return std::unexpected(make_error(Errc::unsupported, "shard"));
  if (file[56] != std::byte{0}) return std::unexpected(make_error(Errc::unsupported, "mix_prec"));
  auto want = crc32c(sp.subspan(0, 60));
  if (rd_u32(sp, 60) != want) return std::unexpected(make_error(Errc::ckpt_corrupt, "header crc"));
  if (tc > 1'000'000 || heap_b > file.size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "limits"));
  }
  const auto desc_off = align64(64 + static_cast<std::size_t>(heap_b));

  GyreFile gf;
  auto st = std::make_shared<Storage>();
  st->heap.assign(file.begin(), file.end());
  gf = GyreFile(GyreDoc{}, st, payload_off);

  for (std::uint64_t i = 0; i < tc; ++i) {
    auto o = desc_off + static_cast<std::size_t>(i) * 80;
    if (o + 80 > file.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "desc"));
    auto name_off = rd_u32(sp, o);
    auto name_len = static_cast<std::uint16_t>(static_cast<unsigned>(file[o + 4]) |
                                              (static_cast<unsigned>(file[o + 5]) << 8));
    if (static_cast<std::uint64_t>(name_off) + name_len > heap_b || name_len > 256) {
      return std::unexpected(make_error(Errc::ckpt_corrupt, "name"));
    }
    std::string name(reinterpret_cast<const char*>(file.data() + 64 + name_off), name_len);
    auto dt = static_cast<DType>(static_cast<std::uint8_t>(file[o + 6]));
    auto ndim = static_cast<std::uint8_t>(file[o + 7]);
    std::vector<std::int64_t> sh(ndim);
    for (int k = 0; k < ndim; ++k) sh[k] = static_cast<std::int64_t>(rd_u64(sp, o + 8 + k * 8));
    auto prel = rd_u64(sp, o + 72);
    GyreTensorDesc d;
    d.name = std::move(name);
    d.dtype = dt;
    d.shape = std::move(sh);
    d.offset = prel;
    d.nbytes = nbytes_of(dt, d.shape);
    if (d.name.rfind("m:", 0) == 0) d.role = TensorRole::adam_m;
    else if (d.name.rfind("v:", 0) == 0) d.role = TensorRole::adam_v;
    else d.role = TensorRole::weight;
    gf.doc().tensors.push_back(std::move(d));
  }

  if (trailer_off + 24 > file.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "trailer"));
  if (flags & 2) {
    auto pcrc = rd_u32(sp, static_cast<std::size_t>(trailer_off));
    auto rel = static_cast<std::size_t>(trailer_off - payload_off);
    auto got = crc32c(sp.subspan(static_cast<std::size_t>(payload_off), rel));
    if (pcrc != got) return std::unexpected(make_error(Errc::ckpt_corrupt, "payload crc"));
  }
  gf.doc().rng_seed = rd_u64(sp, static_cast<std::size_t>(trailer_off + 4));
  gf.doc().train_step = rd_u64(sp, static_cast<std::size_t>(trailer_off + 12));
  auto jl = rd_u32(sp, static_cast<std::size_t>(trailer_off + 20));
  if (jl > (16u << 20) || trailer_off + 24 + jl > file.size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "json"));
  }
  std::string js(reinterpret_cast<const char*>(file.data() + trailer_off + 24), jl);
  auto parsed = GyreDoc::from_json(js);
  if (parsed) {
    auto tens = std::move(gf.doc().tensors);
    auto seed = gf.doc().rng_seed;
    auto step = gf.doc().train_step;
    gf.doc() = std::move(*parsed);
    gf.doc().rng_seed = seed;
    gf.doc().train_step = step;
    if (gf.doc().tensors.empty()) gf.doc().tensors = std::move(tens);
  } else {
    gf.doc().config_json = js;
  }
  return gf;
}

Result<GyreFile> open_v2_mapped(const std::filesystem::path& path, std::shared_ptr<Storage> st) {
  if (!st || st->size() < 64) return std::unexpected(make_error(Errc::ckpt_corrupt, "short"));
  auto sp = std::span<const std::byte>(st->data(), st->size());
  if (std::memcmp(st->data(), "GYRE1\0\0\0", 8) != 0) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "magic"));
  }
  auto ver = rd_u32(sp, 8);
  if (ver == 1) return open_v1(path, sp);
  if (ver != 2) return std::unexpected(make_error(Errc::unsupported, "version"));
  auto flags = rd_u32(sp, 12);
  auto json_len = rd_u64(sp, 24);
  auto json_off = rd_u64(sp, 32);
  auto payload_off = rd_u64(sp, 40);
  auto payload_bytes = rd_u64(sp, 48);
  auto shard_count = rd_u16(sp, 58);
  if (shard_count != 1) return std::unexpected(make_error(Errc::unsupported, "shard"));
  auto want = crc32c(sp.subspan(0, 60));
  if (rd_u32(sp, 60) != want) return std::unexpected(make_error(Errc::ckpt_corrupt, "header crc"));
  if (json_off + json_len > st->size() || json_len > (64ull << 20)) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "json range"));
  }
  std::string js(reinterpret_cast<const char*>(st->data() + json_off), static_cast<std::size_t>(json_len));
  auto doc = GyreDoc::from_json(js);
  if (!doc) return std::unexpected(doc.error());
  if (flags & 2) {
    if (payload_off + payload_bytes + 4 > st->size()) {
      return std::unexpected(make_error(Errc::ckpt_corrupt, "crc range"));
    }
    auto pcrc = rd_u32(sp, static_cast<std::size_t>(payload_off + payload_bytes));
    auto got = crc32c(sp.subspan(static_cast<std::size_t>(payload_off),
                                 static_cast<std::size_t>(payload_bytes)));
    if (pcrc != got) return std::unexpected(make_error(Errc::ckpt_corrupt, "payload crc"));
  }
  (void)path;
  return GyreFile(std::move(*doc), std::move(st), payload_off);
}

Result<GyreFile> open_json_file(std::string raw, std::shared_ptr<Device> device) {
  auto doc = GyreDoc::from_json(raw);
  if (!doc) return std::unexpected(doc.error());
  auto j = parse_json(raw);
  if (!j) return std::unexpected(j.error());
  GyreFile gf{std::move(*doc), nullptr, 0};
  if (j->contains("tensors") && (*j)["tensors"].is_array()) {
    const auto& arr = (*j)["tensors"];
    for (std::size_t i = 0; i < gf.doc().tensors.size() && i < arr.size(); ++i) {
      if (!arr[i].contains("data")) continue;
      auto t = tensor_from_data_json(gf.doc().tensors[i], arr[i]["data"], device);
      if (!t) return std::unexpected(t.error());
      gf.add_json_tensor(gf.doc().tensors[i].name, std::move(*t));
    }
  }
  return gf;
}

}  // namespace

std::string GyreDoc::to_json() const {
  nlohmann::json j;
  j["gyre"] = "model";
  j["version"] = 1;
  j["arch"] = arch;
  if (!config_json.empty()) {
    auto c = parse_json(config_json);
    j["config"] = c ? *c : nlohmann::json::object();
  } else {
    j["config"] = nlohmann::json::object();
  }
  j["train"] = {{"step", train_step}, {"rng_seed", rng_seed}};
  if (!tokenizer_json.empty()) {
    auto t = parse_json(tokenizer_json);
    if (t) j["tokenizer"] = *t;
  }
  nlohmann::json tens = nlohmann::json::array();
  for (auto& t : tensors) {
    nlohmann::json o;
    o["name"] = t.name;
    o["dtype"] = dtype_name(t.dtype);
    o["shape"] = shape_json(t.shape);
    o["role"] = role_name(t.role);
    o["codec"] = gyre_codec_name(t.codec);
    o["offset"] = t.offset;
    o["nbytes"] = t.nbytes;
    o["packed_bytes"] = t.packed_bytes ? t.packed_bytes : t.nbytes;
    tens.push_back(std::move(o));
  }
  j["tensors"] = std::move(tens);
  return j.dump();
}

Result<GyreDoc> GyreDoc::from_json(std::string_view json) {
  auto p = parse_json(json);
  if (!p) return std::unexpected(p.error());
  try {
  auto& j = *p;
  GyreDoc d;
  if (j.contains("arch") && j["arch"].is_string()) d.arch = j["arch"].get<std::string>();
  if (j.contains("config") && j["config"].is_object()) {
    d.config_json = j["config"].dump();
  } else {
    nlohmann::json c = nlohmann::json::object();
    for (const char* k : {"preset", "n_layer", "n_head", "d_model", "d_ff", "block_size", "vocab_size",
                          "holdout", "recency"}) {
      if (j.contains(k)) c[k] = j[k];
    }
    d.config_json = c.dump();
  }
  if (j.contains("train") && j["train"].is_object()) {
    auto& tr = j["train"];
    if (tr.contains("step")) d.train_step = tr["step"].get<std::uint64_t>();
    if (tr.contains("rng_seed")) d.rng_seed = tr["rng_seed"].get<std::uint64_t>();
  }
  if (j.contains("tokenizer")) {
    if (j["tokenizer"].is_object()) d.tokenizer_json = j["tokenizer"].dump();
  } else if (j.contains("gyre") && j["gyre"] == "tokenizer" && j.contains("tokenizer")) {
    d.tokenizer_json = j["tokenizer"].dump();
  }
  if (d.arch.empty() && j.contains("gyre") && j["gyre"] == "tokenizer") d.arch = "tokenizer";
  if (j.contains("tensors") && j["tensors"].is_array()) {
    for (auto& t : j["tensors"]) {
      GyreTensorDesc td;
      td.name = t.value("name", std::string{});
      auto dt = dtype_from_name(t.value("dtype", std::string{"f32"}));
      if (!dt) return std::unexpected(dt.error());
      td.dtype = *dt;
      td.shape = shape_from_json(t.value("shape", nlohmann::json::array()));
      td.role = role_from_name(t.value("role", std::string{"weight"}));
      td.codec = gyre_codec_from_name(t.value("codec", std::string{"identity"}));
      td.offset = t.value("offset", std::uint64_t{0});
      td.nbytes = t.value("nbytes", nbytes_of(td.dtype, td.shape));
      td.packed_bytes = t.value("packed_bytes", td.nbytes);
      d.tensors.push_back(std::move(td));
    }
  }
  return d;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("json: ") + e.what()));
  }
}

bool GyreDoc::recency_alibi() const {
  auto c = parse_json(config_json);
  if (!c || !c->contains("recency")) return false;
  auto r = (*c)["recency"];
  if (r.is_string()) return r.get<std::string>() == "alibi";
  if (r.is_boolean()) return r.get<bool>();
  return false;
}

double GyreDoc::holdout() const {
  auto c = parse_json(config_json);
  if (!c || !c->contains("holdout")) return 0.1;
  return (*c)["holdout"].get<double>();
}

Result<void> save_gyre(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                       const GyreDoc& doc_in, std::span<const std::string> param_names, GyreCodec codec) {
  GyreDoc doc = doc_in;
  auto items = collect_named(params, adam, param_names);
  std::vector<std::vector<std::byte>> packed;
  packed.reserve(items.size());
  doc.tensors.clear();
  std::uint64_t rel = 0;
  for (auto& it : items) {
    auto hb = it.t->host_bytes();
    if (!hb) return std::unexpected(hb.error());
    auto cdc = (it.t->dtype() == DType::f32) ? codec : GyreCodec::identity;
    auto enc = gyre_compress(cdc, *hb, it.t->dtype(), it.t->shape());
    if (!enc) return std::unexpected(enc.error());
    GyreTensorDesc d;
    d.name = it.name;
    d.dtype = it.t->dtype();
    d.shape.assign(it.t->shape().begin(), it.t->shape().end());
    d.role = it.role;
    d.codec = cdc;
    d.nbytes = it.t->nbytes();
    d.packed_bytes = enc->size();
    d.offset = rel;
    rel = align64(static_cast<std::size_t>(rel + d.packed_bytes));
    doc.tensors.push_back(std::move(d));
    packed.push_back(std::move(*enc));
  }
  auto js = doc.to_json();
  if (js.size() > (64ull << 20)) return std::unexpected(make_error(Errc::overflow, "json too large"));

  const auto json_off = std::uint64_t{64};
  const auto json_len = static_cast<std::uint64_t>(js.size());
  const auto payload_off = static_cast<std::uint64_t>(align64(static_cast<std::size_t>(json_off + json_len)));
  std::uint64_t payload_bytes = 0;
  if (!doc.tensors.empty()) {
    auto& last = doc.tensors.back();
    payload_bytes = align64(static_cast<std::size_t>(last.offset + last.packed_bytes));
  }

  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return std::unexpected(make_error(Errc::io, "open write"));

  std::array<std::byte, 64> hdr{};
  out.write(reinterpret_cast<const char*>(hdr.data()), 64);
  out.write(js.data(), static_cast<std::streamsize>(js.size()));
  std::vector<char> pad(static_cast<std::size_t>(payload_off - (json_off + json_len)), 0);
  if (!pad.empty()) out.write(pad.data(), static_cast<std::streamsize>(pad.size()));

  std::uint32_t pcrc = 0;
  std::vector<std::byte> zpad(64, std::byte{0});
  for (auto& blob : packed) {
    out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    pcrc = crc32c_update(pcrc, std::span<const std::byte>(blob.data(), blob.size()));
    const auto need = align64(blob.size()) - blob.size();
    if (need) {
      out.write(reinterpret_cast<const char*>(zpad.data()), static_cast<std::streamsize>(need));
      pcrc = crc32c_update(pcrc, std::span<const std::byte>(zpad.data(), need));
    }
  }
  put_u32(hdr.data(), pcrc);  // temp store
  std::array<std::byte, 4> crcbuf{};
  put_u32(crcbuf.data(), pcrc);
  out.write(reinterpret_cast<const char*>(crcbuf.data()), 4);

  hdr = {};
  std::memcpy(hdr.data(), "GYRE1\0\0\0", 8);
  put_u32(hdr.data() + 8, 2);
  put_u32(hdr.data() + 12, 2);  // payload crc
  put_u64(hdr.data() + 16, items.size());
  put_u64(hdr.data() + 24, json_len);
  put_u64(hdr.data() + 32, json_off);
  put_u64(hdr.data() + 40, payload_off);
  put_u64(hdr.data() + 48, payload_bytes);
  put_u16(hdr.data() + 56, 0);
  put_u16(hdr.data() + 58, 1);
  auto hcrc = crc32c(std::span<const std::byte>(hdr.data(), 60));
  put_u32(hdr.data() + 60, hcrc);
  out.seekp(0);
  out.write(reinterpret_cast<const char*>(hdr.data()), 64);
  if (!out) return std::unexpected(make_error(Errc::io, "write"));
  return {};
}

Result<void> save_gyre_json(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                            const GyreDoc& doc_in, std::span<const std::string> param_names,
                            bool include_data) {
  GyreDoc doc = doc_in;
  auto items = collect_named(params, adam, param_names);
  apply_named_to_doc(doc, items);
  auto j = parse_json(doc.to_json());
  if (!j) return std::unexpected(j.error());
  if (include_data) {
    for (std::size_t i = 0; i < items.size(); ++i) {
      auto data = tensor_data_json(*items[i].t);
      if (!data) return std::unexpected(data.error());
      (*j)["tensors"][i]["data"] = std::move(*data);
      (*j)["tensors"][i].erase("offset");
    }
  }
  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return std::unexpected(make_error(Errc::io, path.string()));
  auto s = j->dump(2);
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
  return {};
}

Result<GyreFile> GyreFile::open(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, "open read"));
  char first = 0;
  in.read(&first, 1);
  if (!in) return std::unexpected(make_error(Errc::ckpt_corrupt, "empty"));
  in.seekg(0);
  if (first == '{') {
    std::ostringstream ss;
    ss << in.rdbuf();
    auto d = Device::cpu();
    if (!d) return std::unexpected(d.error());
    return open_json_file(ss.str(), *d);
  }
  auto st = Storage::mmap_file(path);
  if (st) return open_v2_mapped(path, *st);
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto heap = std::make_shared<Storage>();
  heap->heap.resize(raw.size());
  if (!raw.empty()) std::memcpy(heap->heap.data(), raw.data(), raw.size());
  return open_v2_mapped(path, heap);
}

Result<Tensor> GyreFile::load_tensor(std::string_view name, std::shared_ptr<Device> device) const {
  auto it = json_owned_.find(std::string(name));
  if (it != json_owned_.end()) return it->second;
  const GyreTensorDesc* desc = nullptr;
  for (auto& t : doc_.tensors) {
    if (t.name == name) {
      desc = &t;
      break;
    }
  }
  if (!desc) return std::unexpected(make_error(Errc::ckpt_corrupt, "missing " + std::string(name)));
  if (!storage_) return std::unexpected(make_error(Errc::io, "no storage"));
  const auto packed = desc->packed_bytes ? desc->packed_bytes : desc->nbytes;
  auto start = static_cast<std::size_t>(payload_offset_ + desc->offset);
  if (start + packed > storage_->size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "payload"));
  }
  auto slice = std::span<const std::byte>(storage_->data() + start, static_cast<std::size_t>(packed));
  if (desc->codec == GyreCodec::identity || packed == desc->nbytes) {
    return Tensor::from_storage(storage_, start, desc->shape, desc->dtype, std::move(device));
  }
  auto raw = gyre_decompress(desc->codec, slice, desc->dtype, desc->shape, static_cast<std::size_t>(desc->nbytes));
  if (!raw) return std::unexpected(raw.error());
  return Tensor::from_host(*raw, desc->shape, desc->dtype, std::move(device));
}

Result<void> GyreFile::load_params(std::span<Param> params, Adam* adam) const {
  std::vector<const GyreTensorDesc*> weights, ms, vs;
  for (auto& t : doc_.tensors) {
    if (t.role == TensorRole::weight) weights.push_back(&t);
    else if (t.role == TensorRole::adam_m) ms.push_back(&t);
    else if (t.role == TensorRole::adam_v) vs.push_back(&t);
  }
  if (weights.size() != params.size()) {
    return std::unexpected(make_error(Errc::invalid_shape, "weight count"));
  }
  auto copy_t = [&](const std::string& name, Tensor& dst) -> Result<void> {
    auto t = load_tensor(name, dst.device());
    if (!t) return std::unexpected(t.error());
    auto hb = t->host_bytes();
    if (!hb) return std::unexpected(hb.error());
    return copy_bytes_into(dst, std::span<const std::byte>(hb->data(), hb->size()));
  };
  for (std::size_t i = 0; i < params.size(); ++i) {
    auto r = copy_t(weights[i]->name, params[i].value);
    if (!r) return r;
    if (adam && i < adam->m.size()) {
      if (i >= ms.size() || i >= vs.size()) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "missing adam"));
      }
      auto r1 = copy_t(ms[i]->name, adam->m[i]);
      if (!r1) return r1;
      auto r2 = copy_t(vs[i]->name, adam->v[i]);
      if (!r2) return r2;
    }
  }
  return {};
}

Result<GyreDoc> peek_gyre(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, "open read"));
  char first = 0;
  in.read(&first, 1);
  if (!in) return std::unexpected(make_error(Errc::ckpt_corrupt, "empty"));
  in.seekg(0);
  if (first == '{') {
    std::ostringstream ss;
    ss << in.rdbuf();
    return GyreDoc::from_json(ss.str());
  }
  std::array<char, 64> hdr{};
  in.read(hdr.data(), 64);
  if (in.gcount() != 64) return std::unexpected(make_error(Errc::ckpt_corrupt, "short"));
  auto sp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(hdr.data()), 64);
  if (std::memcmp(hdr.data(), "GYRE1\0\0\0", 8) != 0) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "magic"));
  }
  auto ver = rd_u32(sp, 8);
  if (ver == 1) {
    auto trailer_off = rd_u64(sp, 40);
    in.seekg(static_cast<std::streamoff>(trailer_off));
    std::array<char, 24> tr{};
    in.read(tr.data(), 24);
    if (in.gcount() != 24) return std::unexpected(make_error(Errc::ckpt_corrupt, "trailer"));
    auto tsp = std::span<const std::byte>(reinterpret_cast<const std::byte*>(tr.data()), 24);
    auto jl = rd_u32(tsp, 20);
    if (jl > (16u << 20)) return std::unexpected(make_error(Errc::ckpt_corrupt, "json"));
    std::string js(jl, '\0');
    in.read(js.data(), static_cast<std::streamsize>(jl));
    auto d = GyreDoc::from_json(js);
    if (!d) return d;
    d->rng_seed = rd_u64(tsp, 4);
    d->train_step = rd_u64(tsp, 12);
    return d;
  }
  if (ver != 2) return std::unexpected(make_error(Errc::unsupported, "version"));
  auto json_len = rd_u64(sp, 24);
  auto json_off = rd_u64(sp, 32);
  if (json_len > (64ull << 20)) return std::unexpected(make_error(Errc::ckpt_corrupt, "json"));
  in.seekg(static_cast<std::streamoff>(json_off));
  std::string js(static_cast<std::size_t>(json_len), '\0');
  in.read(js.data(), static_cast<std::streamsize>(json_len));
  if (static_cast<std::uint64_t>(in.gcount()) != json_len) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "json short"));
  }
  return GyreDoc::from_json(js);
}

Result<void> save_gyre1(const std::filesystem::path& path, std::span<const Param> params, const Adam* adam,
                        const CheckpointMeta& meta) {
  auto doc = GyreDoc::from_json(meta.json);
  GyreDoc d;
  if (doc) d = std::move(*doc);
  else {
    d.config_json = meta.json;
    d.arch = "char-lm";
  }
  d.train_step = meta.train_step;
  d.rng_seed = meta.rng_seed;
  return save_gyre(path, params, adam, d, meta.param_names);
}

Result<void> load_gyre1(const std::filesystem::path& path, std::span<Param> params, Adam* adam,
                        CheckpointMeta& meta) {
  auto f = GyreFile::open(path);
  if (!f) return std::unexpected(f.error());
  meta.rng_seed = f->doc().rng_seed;
  meta.train_step = f->doc().train_step;
  meta.json = f->doc().to_json();
  return f->load_params(params, adam);
}

Result<CheckpointMeta> peek_gyre1(const std::filesystem::path& path) {
  auto d = peek_gyre(path);
  if (!d) return std::unexpected(d.error());
  CheckpointMeta m;
  m.rng_seed = d->rng_seed;
  m.train_step = d->train_step;
  m.json = d->to_json();
  return m;
}

}  // namespace gyre
