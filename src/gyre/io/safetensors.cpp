#include "gyre/io/safetensors.hpp"

#include "atomic_file.hpp"
#include "json_parse.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>

namespace gyre {

DType safetensors_dtype(std::string_view s) {
  if (s == "F32" || s == "f32") return DType::f32;
  if (s == "F16" || s == "f16") return DType::f16;
  if (s == "BF16" || s == "bf16") return DType::bf16;
  if (s == "I32" || s == "i32") return DType::i32;
  if (s == "U8" || s == "u8") return DType::u8;
  if (s == "I8" || s == "i8") return DType::i8;
  return DType::f32;
}

namespace {

bool gyre_dtype_name(std::string_view s) {
  return s == "F32" || s == "f32" || s == "F16" || s == "f16" || s == "BF16" || s == "bf16" || s == "I32" ||
         s == "i32" || s == "U8" || s == "u8" || s == "I8" || s == "i8";
}

// Element size of a safetensors dtype string; 0 when the name is not a safetensors dtype at all.
std::uint64_t st_elem_size(std::string_view s) {
  if (s == "BOOL" || s == "U8" || s == "I8" || s == "F8_E5M2" || s == "F8_E4M3" || s == "u8" || s == "i8")
    return 1;
  if (s == "I16" || s == "U16" || s == "F16" || s == "BF16" || s == "f16" || s == "bf16") return 2;
  if (s == "I32" || s == "U32" || s == "F32" || s == "i32" || s == "f32") return 4;
  if (s == "I64" || s == "U64" || s == "F64") return 8;
  return 0;
}

Error st_corrupt(const std::string& what) { return make_error(Errc::ckpt_corrupt, "safetensors: " + what); }

// numel * es without wraparound; false on overflow or a negative dim.
bool checked_bytes(std::span<const std::int64_t> shape, std::uint64_t es, std::uint64_t& out) {
  std::uint64_t n = es;
  for (auto d : shape) {
    if (d < 0) return false;
    const auto ud = static_cast<std::uint64_t>(d);
    if (ud != 0 && n > std::numeric_limits<std::uint64_t>::max() / ud) return false;
    n *= ud;
  }
  out = n;
  return true;
}

}  // namespace

Result<SafetensorFile> safetensors_open_header(std::string_view header_json, std::uint64_t header_len,
                                               const std::filesystem::path& path) {
  auto parsed = parse_json(header_json);
  if (!parsed) return std::unexpected(parsed.error());
  if (!parsed->is_object()) return std::unexpected(make_error(Errc::ckpt_corrupt, "st json"));
  SafetensorFile file;
  file.path = path;
  file.header_len = header_len;
  file.data_offset = 8 + header_len;
  // Every field is untrusted: check types and ranges here so a bad header is an error, never an
  // exception or a wrapped size later.
  for (auto& [key, val] : parsed->items()) {
    if (key == "__metadata__") continue;
    if (!val.is_object()) continue;
    SafetensorInfo info;
    info.name = key;
    std::string dt = "F32";
    if (val.contains("dtype")) {
      if (!val["dtype"].is_string()) return std::unexpected(st_corrupt(key + ": dtype is not a string"));
      dt = val["dtype"].get<std::string>();
    }
    info.dtype = safetensors_dtype(dt);
    info.dtype_known = gyre_dtype_name(dt);
    if (val.contains("shape")) {
      const auto& sh = val["shape"];
      if (!sh.is_array() || sh.size() > 8) return std::unexpected(st_corrupt(key + ": shape"));
      for (auto& d : sh) {
        if (!d.is_number_integer() || d.get<std::int64_t>() < 0)
          return std::unexpected(st_corrupt(key + ": shape"));
        info.shape.push_back(d.get<std::int64_t>());
      }
    }
    if (val.contains("data_offsets")) {
      const auto& off = val["data_offsets"];
      if (!off.is_array() || off.size() != 2 || !off[0].is_number_unsigned() || !off[1].is_number_unsigned())
        return std::unexpected(st_corrupt(key + ": data_offsets"));
      info.data_begin = off[0].get<std::uint64_t>();
      info.data_end = off[1].get<std::uint64_t>();
    }
    if (info.data_end < info.data_begin) return std::unexpected(st_corrupt(key + ": data_offsets end < begin"));
    if (info.data_begin > std::numeric_limits<std::uint64_t>::max() - file.data_offset ||
        info.data_end > std::numeric_limits<std::uint64_t>::max() - file.data_offset)
      return std::unexpected(st_corrupt(key + ": data_offsets"));
    if (const auto es = st_elem_size(dt)) {
      std::uint64_t want = 0;
      if (!checked_bytes(info.shape, es, want)) return std::unexpected(st_corrupt(key + ": shape overflow"));
      if (want != info.data_end - info.data_begin)
        return std::unexpected(st_corrupt(key + ": data_offsets do not match dtype * shape"));
    }
    file.tensors.push_back(std::move(info));
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
  auto file = safetensors_open_header(json, n, path);
  if (!file) return file;
  std::error_code ec;
  const auto fsz = static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
  if (!ec) {
    for (auto& t : file->tensors) {
      if (file->data_offset > fsz || t.data_end > fsz - file->data_offset)
        return std::unexpected(st_corrupt(t.name + ": data past the end of the file"));
    }
  }
  return file;
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
  if (!info->dtype_known)
    return std::unexpected(make_error(Errc::unsupported, "safetensors: unsupported dtype for " + info->name));
  // SafetensorFile is a plain struct: re-check what safetensors_open_header validated.
  std::uint64_t want = 0;
  if (info->data_end < info->data_begin || !checked_bytes(info->shape, dtype_size(info->dtype), want) ||
      want != info->data_end - info->data_begin ||
      info->data_begin > std::numeric_limits<std::uint64_t>::max() - file.data_offset)
    return std::unexpected(st_corrupt(info->name + ": data_offsets do not match dtype * shape"));
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
  nlohmann::json header = nlohmann::json::object();
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!tensors[i].t) return std::unexpected(make_error(Errc::invalid_shape, "null tensor"));
    nlohmann::json sh = nlohmann::json::array();
    for (auto d : tensors[i].t->shape()) sh.push_back(d);
    const auto n = tensors[i].t->nbytes();
    header[tensors[i].name] = {{"dtype", safetensors_dtype_str(tensors[i].t->dtype())},
                               {"shape", sh},
                               {"data_offsets", {off, off + n}}};
    off += n;
  }
  auto json = header.dump();
  std::uint64_t hlen = json.size();
  return detail::write_file_atomic(path, [&](std::ofstream& out) -> Result<void> {
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    for (auto& nt : tensors) {
      auto hb = nt.t->host_bytes();
      if (!hb) return std::unexpected(hb.error());
      out.write(reinterpret_cast<const char*>(hb->data()), static_cast<std::streamsize>(hb->size()));
    }
    return {};
  });
}

}  // namespace gyre
