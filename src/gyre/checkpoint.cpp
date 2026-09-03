#include "gyre/checkpoint.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <vector>

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

void wr_u32(std::vector<std::byte>& o, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) o.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
}
void wr_u64(std::vector<std::byte>& o, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) o.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
}
void wr_i64(std::vector<std::byte>& o, std::int64_t v) { wr_u64(o, static_cast<std::uint64_t>(v)); }

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

std::size_t align64(std::size_t n) { return (n + 63) & ~std::size_t{63}; }

struct Named {
  std::string name;
  const Tensor* t;
};

}  // namespace

Result<void> save_gyre1(const std::filesystem::path& path, std::span<Param> params, const Adam* adam,
                        const CheckpointMeta& meta) {
  if (params.size() > 1'000'000) {
    return std::unexpected(make_error(Errc::overflow, "too many tensors"));
  }
  std::vector<Named> items;
  for (std::size_t i = 0; i < params.size(); ++i) {
    items.push_back({"w:" + std::to_string(i), &params[i].value});
    if (adam && i < adam->m.size()) {
      items.push_back({"m:" + std::to_string(i), &adam->m[i]});
      items.push_back({"v:" + std::to_string(i), &adam->v[i]});
    }
  }

  std::string heap;
  struct Desc {
    std::uint32_t name_off;
    std::uint16_t name_len;
    DType dt;
    std::uint8_t ndim;
    std::array<std::int64_t, 8> shape{};
    std::uint64_t payload_rel;
    const Tensor* t;
  };
  std::vector<Desc> descs;
  std::uint64_t rel = 0;
  for (auto& it : items) {
    if (it.name.size() > 256) {
      return std::unexpected(make_error(Errc::invalid_shape, "name too long"));
    }
    Desc d;
    d.name_off = static_cast<std::uint32_t>(heap.size());
    d.name_len = static_cast<std::uint16_t>(it.name.size());
    heap += it.name;
    d.dt = it.t->dtype();
    d.ndim = it.t->rank();
    for (int i = 0; i < it.t->rank(); ++i) d.shape[i] = it.t->shape()[i];
    d.payload_rel = rel;
    d.t = it.t;
    rel = align64(rel + it.t->nbytes());
    descs.push_back(d);
  }

  const std::uint64_t heap_bytes = heap.size();
  const std::uint64_t desc_off = align64(64 + heap_bytes);
  const std::uint64_t payload_off = align64(desc_off + descs.size() * 80);
  const std::uint64_t trailer_off = payload_off + rel;

  std::vector<std::byte> file;
  file.resize(static_cast<std::size_t>(trailer_off + 24 + meta.json.size()));

  auto put = [&](std::size_t o, const void* p, std::size_t n) {
    std::memcpy(file.data() + o, p, n);
  };

  put(0, "GYRE1\0\0\0", 8);
  std::uint32_t ver = 1, flags = 2;  // payload crc
  put(8, &ver, 4);
  put(12, &flags, 4);
  std::uint64_t tc = descs.size();
  put(16, &tc, 8);
  put(24, &heap_bytes, 8);
  put(32, &payload_off, 8);
  put(40, &trailer_off, 8);
  std::uint32_t sh = 0, sc = 1;
  put(48, &sh, 4);
  put(52, &sc, 4);
  file[56] = std::byte{0};
  auto hcrc = crc32c(std::span<const std::byte>(file.data(), 60));
  put(60, &hcrc, 4);

  put(64, heap.data(), heap.size());
  std::size_t doff = static_cast<std::size_t>(desc_off);
  for (auto& d : descs) {
    put(doff, &d.name_off, 4);
    put(doff + 4, &d.name_len, 2);
    file[doff + 6] = static_cast<std::byte>(static_cast<std::uint8_t>(d.dt));
    file[doff + 7] = static_cast<std::byte>(d.ndim);
    put(doff + 8, d.shape.data(), 64);
    put(doff + 72, &d.payload_rel, 8);
    doff += 80;
    auto hb = d.t->host_bytes();
    if (!hb) return std::unexpected(hb.error());
    put(static_cast<std::size_t>(payload_off + d.payload_rel), hb->data(), hb->size());
  }

  auto pcrc = crc32c(std::span<const std::byte>(file.data() + payload_off, static_cast<std::size_t>(rel)));
  put(static_cast<std::size_t>(trailer_off), &pcrc, 4);
  put(static_cast<std::size_t>(trailer_off + 4), &meta.rng_seed, 8);
  put(static_cast<std::size_t>(trailer_off + 12), &meta.train_step, 8);
  std::uint32_t jl = static_cast<std::uint32_t>(meta.json.size());
  put(static_cast<std::size_t>(trailer_off + 20), &jl, 4);
  put(static_cast<std::size_t>(trailer_off + 24), meta.json.data(), meta.json.size());

  std::ofstream out(path, std::ios::binary);
  if (!out) return std::unexpected(make_error(Errc::io, "open write"));
  out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  if (!out) return std::unexpected(make_error(Errc::io, "write"));
  return {};
}

Result<void> load_gyre1(const std::filesystem::path& path, std::span<Param> params, Adam* adam,
                        CheckpointMeta& meta) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, "open read"));
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<std::byte> file(raw.size());
  if (!raw.empty()) std::memcpy(file.data(), raw.data(), raw.size());
  if (file.size() < 64) return std::unexpected(make_error(Errc::ckpt_corrupt, "short"));
  if (std::memcmp(file.data(), "GYRE1\0\0\0", 8) != 0) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "magic"));
  }
  auto sp = std::span<const std::byte>(file);
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

  std::unordered_map<std::string, std::span<const std::byte>> blobs;
  std::unordered_map<std::string, std::pair<DType, std::vector<std::int64_t>>> shapes;
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
    if (dt != DType::f32 && dt != DType::i32 && dt != DType::u8) {
      return std::unexpected(make_error(Errc::unsupported, "dtype"));
    }
    auto ndim = static_cast<std::uint8_t>(file[o + 7]);
    std::vector<std::int64_t> sh(ndim);
    for (int k = 0; k < ndim; ++k) sh[k] = static_cast<std::int64_t>(rd_u64(sp, o + 8 + k * 8));
    auto prel = rd_u64(sp, o + 72);
    std::int64_t n = 1;
    for (auto d : sh) n *= d;
    auto nb = static_cast<std::size_t>(n) * dtype_size(dt);
    auto start = static_cast<std::size_t>(payload_off + prel);
    if (start + nb > file.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "payload"));
    blobs[name] = std::span<const std::byte>(file.data() + start, nb);
    shapes[name] = {dt, sh};
  }

  auto load_into = [&](const std::string& name, Tensor& t) -> Result<void> {
    auto it = blobs.find(name);
    if (it == blobs.end()) return std::unexpected(make_error(Errc::ckpt_corrupt, "missing " + name));
    auto hb = t.host_bytes();
    if (!hb) return std::unexpected(hb.error());
    if (hb->size() != it->second.size()) {
      return std::unexpected(make_error(Errc::invalid_shape, "size " + name));
    }
    std::memcpy(hb->data(), it->second.data(), hb->size());
    return {};
  };

  for (std::size_t i = 0; i < params.size(); ++i) {
    auto r = load_into("w:" + std::to_string(i), params[i].value);
    if (!r) return r;
    if (adam && i < adam->m.size()) {
      auto r1 = load_into("m:" + std::to_string(i), adam->m[i]);
      auto r2 = load_into("v:" + std::to_string(i), adam->v[i]);
      if (!r1) return r1;
      if (!r2) return r2;
    }
  }

  if (trailer_off + 24 > file.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "trailer"));
  if (flags & 2) {
    auto pcrc = rd_u32(sp, static_cast<std::size_t>(trailer_off));
    auto rel = static_cast<std::size_t>(trailer_off - payload_off);
    auto got = crc32c(sp.subspan(static_cast<std::size_t>(payload_off), rel));
    if (pcrc != got) return std::unexpected(make_error(Errc::ckpt_corrupt, "payload crc"));
  }
  meta.rng_seed = rd_u64(sp, static_cast<std::size_t>(trailer_off + 4));
  meta.train_step = rd_u64(sp, static_cast<std::size_t>(trailer_off + 12));
  auto jl = rd_u32(sp, static_cast<std::size_t>(trailer_off + 20));
  if (jl > (1u << 20) || trailer_off + 24 + jl > file.size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "json"));
  }
  meta.json.assign(reinterpret_cast<const char*>(file.data() + trailer_off + 24), jl);
  return {};
}

Result<CheckpointMeta> peek_gyre1(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, "open read"));
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (raw.size() < 64) return std::unexpected(make_error(Errc::ckpt_corrupt, "short"));
  if (std::memcmp(raw.data(), "GYRE1\0\0\0", 8) != 0) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "magic"));
  }
  std::vector<std::byte> file(raw.size());
  std::memcpy(file.data(), raw.data(), raw.size());
  auto sp = std::span<const std::byte>(file);
  auto trailer_off = rd_u64(sp, 40);
  if (trailer_off + 24 > file.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "trailer"));
  CheckpointMeta meta;
  meta.rng_seed = rd_u64(sp, static_cast<std::size_t>(trailer_off + 4));
  meta.train_step = rd_u64(sp, static_cast<std::size_t>(trailer_off + 12));
  auto jl = rd_u32(sp, static_cast<std::size_t>(trailer_off + 20));
  if (jl > (1u << 20) || trailer_off + 24 + jl > file.size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "json"));
  }
  meta.json.assign(reinterpret_cast<const char*>(file.data() + trailer_off + 24), jl);
  return meta;
}

}  // namespace gyre
