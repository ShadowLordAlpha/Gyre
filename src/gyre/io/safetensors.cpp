#include "gyre/io/safetensors.hpp"

#include "gyre/nn/tokenize.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

namespace gyre {
namespace {

void skip_ws(std::string_view s, std::size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
}

bool parse_string(std::string_view s, std::size_t& i, std::string& out) {
  skip_ws(s, i);
  return json_unescape_string(s, i, out);
}

bool parse_u64(std::string_view s, std::size_t& i, std::uint64_t& out) {
  skip_ws(s, i);
  char* end = nullptr;
  out = std::strtoull(s.data() + i, &end, 10);
  if (end == s.data() + i) return false;
  i = static_cast<std::size_t>(end - s.data());
  return true;
}

bool parse_i64_array(std::string_view s, std::size_t& i, std::vector<std::int64_t>& out) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '[') return false;
  ++i;
  out.clear();
  while (i < s.size()) {
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
      ++i;
      return true;
    }
    char* end = nullptr;
    auto v = std::strtoll(s.data() + i, &end, 10);
    if (end == s.data() + i) return false;
    out.push_back(v);
    i = static_cast<std::size_t>(end - s.data());
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') ++i;
  }
  return false;
}

bool skip_value(std::string_view s, std::size_t& i) {
  skip_ws(s, i);
  if (i >= s.size()) return false;
  if (s[i] == '"') {
    std::string tmp;
    return parse_string(s, i, tmp);
  }
  if (s[i] == '{') {
    int d = 0;
    for (; i < s.size(); ++i) {
      if (s[i] == '"') {
        std::string tmp;
        if (!parse_string(s, i, tmp)) return false;
        --i;
        continue;
      }
      if (s[i] == '{') ++d;
      else if (s[i] == '}') {
        --d;
        if (d == 0) {
          ++i;
          return true;
        }
      }
    }
    return false;
  }
  if (s[i] == '[') {
    int d = 0;
    for (; i < s.size(); ++i) {
      if (s[i] == '"') {
        std::string tmp;
        if (!parse_string(s, i, tmp)) return false;
        --i;
        continue;
      }
      if (s[i] == '[') ++d;
      else if (s[i] == ']') {
        --d;
        if (d == 0) {
          ++i;
          return true;
        }
      }
    }
    return false;
  }
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
  return true;
}

}  // namespace

DType safetensors_dtype(std::string_view s) {
  if (s == "F32" || s == "f32") return DType::f32;
  if (s == "F16" || s == "f16") return DType::f16;
  if (s == "BF16" || s == "bf16") return DType::bf16;
  if (s == "I32" || s == "i32") return DType::i32;
  if (s == "U8" || s == "u8") return DType::u8;
  if (s == "I8" || s == "i8") return DType::i8;
  return DType::f32;
}

Result<SafetensorFile> safetensors_open_header(std::string_view header_json, std::uint64_t header_len,
                                               const std::filesystem::path& path) {
  SafetensorFile file;
  file.path = path;
  file.header_len = header_len;
  file.data_offset = 8 + header_len;
  std::size_t i = 0;
  skip_ws(header_json, i);
  if (i >= header_json.size() || header_json[i] != '{') {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "st json"));
  }
  ++i;
  while (i < header_json.size()) {
    skip_ws(header_json, i);
    if (i < header_json.size() && header_json[i] == '}') break;
    std::string key;
    if (!parse_string(header_json, i, key)) {
      return std::unexpected(make_error(Errc::ckpt_corrupt, "st key"));
    }
    skip_ws(header_json, i);
    if (i >= header_json.size() || header_json[i] != ':') {
      return std::unexpected(make_error(Errc::ckpt_corrupt, "st colon"));
    }
    ++i;
    if (key == "__metadata__") {
      if (!skip_value(header_json, i)) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "st meta"));
      }
    } else {
      skip_ws(header_json, i);
      auto obj = json_object_field(header_json.substr(i > 0 ? i - 1 : 0), "dtype");
      // Parse this object from current '{'.
      skip_ws(header_json, i);
      if (i >= header_json.size() || header_json[i] != '{') {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "st tensor"));
      }
      std::size_t start = i;
      if (!skip_value(header_json, i)) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "st tensor obj"));
      }
      auto objv = header_json.substr(start, i - start);
      SafetensorInfo info;
      info.name = std::move(key);
      auto dts = json_object_field(objv, "dtype");
      std::string dt;
      if (!dts.empty()) {
        auto q = dts;
        if (q.size() >= 2 && q.front() == '"') {
          std::size_t qi = 0;
          json_unescape_string(q, qi, dt);
        } else
          dt = std::string(q);
      }
      info.dtype = safetensors_dtype(dt);
      auto shv = json_object_field(objv, "shape");
      std::size_t si = 0;
      if (!parse_i64_array(shv, si, info.shape)) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "st shape"));
      }
      auto off = json_object_field(objv, "data_offsets");
      std::vector<std::int64_t> oe;
      std::size_t oi = 0;
      if (!parse_i64_array(off, oi, oe) || oe.size() != 2) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "st offsets"));
      }
      info.data_begin = static_cast<std::uint64_t>(oe[0]);
      info.data_end = static_cast<std::uint64_t>(oe[1]);
      file.tensors.push_back(std::move(info));
      (void)obj;
    }
    skip_ws(header_json, i);
    if (i < header_json.size() && header_json[i] == ',') ++i;
  }
  return file;
}

Result<SafetensorFile> safetensors_open(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, path.string()));
  unsigned char nb[8];
  in.read(reinterpret_cast<char*>(nb), 8);
  if (in.gcount() != 8) return std::unexpected(make_error(Errc::ckpt_corrupt, "st header len"));
  std::uint64_t n = 0;
  for (int k = 0; k < 8; ++k) n |= static_cast<std::uint64_t>(nb[k]) << (8 * k);
  if (n == 0 || n > 64ull * 1024 * 1024) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "st header too large"));
  }
  std::string json(static_cast<std::size_t>(n), '\0');
  in.read(json.data(), static_cast<std::streamsize>(n));
  if (static_cast<std::uint64_t>(in.gcount()) != n) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "st header short"));
  }
  return safetensors_open_header(json, n, path);
}

Result<Tensor> safetensors_load(const SafetensorFile& file, std::string_view name,
                                std::shared_ptr<Device> device, bool mmap) {
  const SafetensorInfo* info = nullptr;
  for (auto& t : file.tensors) {
    if (t.name == name) {
      info = &t;
      break;
    }
  }
  if (!info) return std::unexpected(make_error(Errc::io, "tensor not found"));
  const auto nbytes = info->data_end - info->data_begin;
  const auto file_off = file.data_offset + info->data_begin;
  if (mmap) {
    auto st = Storage::mmap_file(file.path);
    if (!st) return std::unexpected(st.error());
    return Tensor::from_storage(*st, static_cast<std::size_t>(file_off), info->shape, info->dtype,
                                std::move(device));
  }
  std::ifstream in(file.path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, file.path.string()));
  in.seekg(static_cast<std::streamoff>(file_off));
  std::vector<std::byte> buf(static_cast<std::size_t>(nbytes));
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(nbytes));
  if (static_cast<std::uint64_t>(in.gcount()) != nbytes) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "st payload short"));
  }
  return Tensor::from_host(buf, info->shape, info->dtype, std::move(device));
}

Result<Tensor> concat_tp(std::span<const Tensor> shards, int axis) {
  if (shards.empty()) return std::unexpected(make_error(Errc::invalid_shape, "concat empty"));
  const auto r = shards[0].rank();
  if (axis < 0 || axis >= r) return std::unexpected(make_error(Errc::invalid_shape, "concat axis"));
  std::int64_t cat = 0;
  for (const auto& s : shards) {
    if (s.rank() != r || s.dtype() != shards[0].dtype()) {
      return std::unexpected(make_error(Errc::invalid_shape, "concat mismatch"));
    }
    for (int i = 0; i < r; ++i) {
      if (i != axis && s.shape()[i] != shards[0].shape()[i]) {
        return std::unexpected(make_error(Errc::invalid_shape, "concat dim"));
      }
    }
    cat += s.shape()[axis];
  }
  std::array<std::int64_t, 8> osh{};
  for (int i = 0; i < r; ++i) osh[i] = shards[0].shape()[i];
  osh[axis] = cat;
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), r), shards[0].dtype(),
                           shards[0].device());
  if (!out) return out;
  auto dst = out->host_bytes();
  if (!dst) return std::unexpected(dst.error());
  const auto es = dtype_size(shards[0].dtype());
  std::int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; ++i) outer *= shards[0].shape()[i];
  for (int i = axis + 1; i < r; ++i) inner *= shards[0].shape()[i];
  std::size_t doff = 0;
  for (std::int64_t o = 0; o < outer; ++o) {
    for (const auto& s : shards) {
      auto src = s.host_bytes();
      if (!src) return std::unexpected(src.error());
      const auto take = static_cast<std::size_t>(s.shape()[axis] * inner) * es;
      const auto soff = static_cast<std::size_t>(o * s.shape()[axis] * inner) * es;
      std::memcpy(dst->data() + doff, src->data() + soff, take);
      doff += take;
    }
  }
  return out;
}

const char* safetensors_dtype_str(DType d) {
  switch (d) {
    case DType::f32:
      return "F32";
    case DType::f16:
      return "F16";
    case DType::bf16:
      return "BF16";
    case DType::i32:
      return "I32";
    case DType::u8:
      return "U8";
    case DType::i8:
      return "I8";
  }
  return "F32";
}

Result<void> safetensors_save(const std::filesystem::path& path, std::span<const NamedTensor> tensors) {
  if (tensors.empty()) return std::unexpected(make_error(Errc::invalid_shape, "save empty"));
  std::uint64_t off = 0;
  std::string json = "{";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!tensors[i].t) return std::unexpected(make_error(Errc::invalid_shape, "null tensor"));
    if (i) json += ',';
    json += json_escape(tensors[i].name);
    json += ":{\"dtype\":\"";
    json += safetensors_dtype_str(tensors[i].t->dtype());
    json += "\",\"shape\":[";
    auto sh = tensors[i].t->shape();
    for (std::size_t d = 0; d < sh.size(); ++d) {
      if (d) json += ',';
      json += std::to_string(sh[d]);
    }
    const auto n = tensors[i].t->nbytes();
    json += "],\"data_offsets\":[";
    json += std::to_string(off);
    json += ',';
    json += std::to_string(off + n);
    json += "]}";
    off += n;
  }
  json += '}';
  std::uint64_t hlen = json.size();
  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary);
  if (!out) return std::unexpected(make_error(Errc::io, path.string()));
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(json.data(), static_cast<std::streamsize>(json.size()));
  for (auto& nt : tensors) {
    auto hb = nt.t->host_bytes();
    if (!hb) return std::unexpected(hb.error());
    out.write(reinterpret_cast<const char*>(hb->data()), static_cast<std::streamsize>(hb->size()));
  }
  return {};
}

}  // namespace gyre
