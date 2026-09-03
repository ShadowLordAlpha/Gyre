#include "gyre/io/compress_probe.hpp"
#include "gyre/io/wpack.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace gyre {
namespace {

double shannon_bpb(const std::uint64_t hist[256], std::uint64_t n) {
  if (n == 0) return 0;
  double h = 0;
  const double inv = 1.0 / static_cast<double>(n);
  for (int i = 0; i < 256; ++i) {
    if (!hist[i]) continue;
    const double p = static_cast<double>(hist[i]) * inv;
    h -= p * std::log2(p);
  }
  return h;
}

void hist_bytes(std::span<const std::byte> b, std::uint64_t hist[256]) {
  std::memset(hist, 0, 256 * sizeof(std::uint64_t));
  for (auto x : b) ++hist[static_cast<unsigned char>(x)];
}

std::uint64_t entropy_bytes(double bpb, std::uint64_t n) {
  return static_cast<std::uint64_t>(std::ceil(bpb * static_cast<double>(n) / 8.0));
}

std::uint64_t rle_size(std::span<const std::byte> b) {
  if (b.empty()) return 0;
  std::uint64_t out = 0;
  std::size_t i = 0;
  while (i < b.size()) {
    auto v = b[i];
    std::size_t j = i + 1;
    while (j < b.size() && b[j] == v && (j - i) < 255) ++j;
    out += 2;  // count + byte
    i = j;
  }
  return out;
}

// Greedy LZSS on a prefix (probe, not a real gzip). 256-byte window.
std::uint64_t lzss_size(std::span<const std::byte> b) {
  if (b.empty()) return 0;
  constexpr std::size_t kSample = 32768;
  const std::size_t nfull = b.size();
  if (nfull > kSample) {
    const auto s = lzss_size(b.subspan(0, kSample));
    return static_cast<std::uint64_t>(static_cast<double>(s) * static_cast<double>(nfull) /
                                      static_cast<double>(kSample));
  }
  const std::size_t n = b.size();
  std::uint64_t bits = 0;
  std::size_t i = 0;
  while (i < n) {
    std::size_t best_l = 0;
    const std::size_t w0 = i > 255 ? i - 255 : 0;
    for (std::size_t j = w0; j < i; ++j) {
      std::size_t l = 0;
      while (i + l < n && j + l < i && b[j + l] == b[i + l] && l < 18) ++l;
      if (l >= 3 && l > best_l) best_l = l;
    }
    if (best_l >= 3) {
      bits += 1 + 12 + 4;
      i += best_l;
    } else {
      bits += 1 + 8;
      ++i;
    }
  }
  return (bits + 7) / 8;
}

std::uint64_t plane_pack_u16(std::span<const std::uint16_t> w) {
  const auto n = w.size();
  if (n == 0) return 0;
  std::uint64_t total = 4;  // 16-bit const mask + 16-bit const values
  for (int b = 0; b < 16; ++b) {
    std::uint64_t ones = 0;
    for (auto x : w)
      if (x & (1u << b)) ++ones;
    const auto zeros = n - ones;
    const auto exc = ones < zeros ? ones : zeros;
    const auto dense = (n + 7) / 8;
    const auto tbl = exc * 4 + 1;  // u32 index list + majority bit
    total += (tbl < dense && exc * 4 < n) ? tbl : dense;
  }
  return total;
}

std::uint64_t freq16_exc(std::span<const std::uint16_t> w, double* top_frac) {
  if (w.empty()) {
    if (top_frac) *top_frac = 0;
    return 0;
  }
  std::unordered_map<std::uint16_t, std::uint32_t> c;
  c.reserve(w.size() / 2 + 1);
  for (auto x : w) ++c[x];
  std::uint32_t best = 0;
  for (auto& [k, n] : c)
    if (n > best) best = n;
  if (top_frac) *top_frac = static_cast<double>(best) / static_cast<double>(w.size());
  const auto exc = w.size() - best;
  return 2 + exc * 6;  // mode u16 + (u32 index, u16 value)
}

void split_bf16(std::span<const std::uint16_t> w, std::vector<std::byte>& sign,
                std::vector<std::byte>& exp, std::vector<std::byte>& mant) {
  sign.assign((w.size() + 7) / 8, std::byte{0});
  exp.resize(w.size());
  mant.resize(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    auto x = w[i];
    if (x & 0x8000) sign[i / 8] |= std::byte{static_cast<unsigned char>(1u << (i % 8))};
    exp[i] = std::byte{static_cast<unsigned char>((x >> 7) & 0xFF)};  // 8-bit exponent
    mant[i] = std::byte{static_cast<unsigned char>(x & 0x7F)};
  }
}

std::span<const std::uint16_t> as_u16(std::span<const std::byte> raw) {
  const auto n = raw.size() / 2;
  return {reinterpret_cast<const std::uint16_t*>(raw.data()), n};
}

}  // namespace

std::string tensor_family(std::string_view name) {
  if (name.find("embed") != std::string_view::npos || name.find("lm_head") != std::string_view::npos)
    return "embed";
  if (name.find("norm") != std::string_view::npos) return "norm";
  if (name.find("attn") != std::string_view::npos || name.find("q_proj") != std::string_view::npos ||
      name.find("k_proj") != std::string_view::npos || name.find("v_proj") != std::string_view::npos ||
      name.find("o_proj") != std::string_view::npos)
    return "attn";
  if (name.find("expert") != std::string_view::npos || name.find("moe") != std::string_view::npos)
    return "moe";
  if (name.find("mlp") != std::string_view::npos || name.find("gate_proj") != std::string_view::npos)
    return "mlp";
  return "other";
}

CompressProbeRow probe_bytes(std::span<const std::byte> raw, std::string_view name) {
  const auto t0 = std::chrono::steady_clock::now();
  CompressProbeRow r;
  r.name = std::string(name);
  r.family = tensor_family(name);
  r.raw_bytes = raw.size();
  r.probed_bytes = raw.size();
  std::uint64_t h[256];
  hist_bytes(raw, h);
  r.entropy_bpb = shannon_bpb(h, raw.size());
  r.entropy_bytes = entropy_bytes(r.entropy_bpb, raw.size());
  r.rle_bytes = rle_size(raw);
  r.lzss_bytes = lzss_size(raw);

  std::vector<std::byte> xored(raw.size());
  if (!raw.empty()) {
    xored[0] = raw[0];
    for (std::size_t i = 1; i < raw.size(); ++i)
      xored[i] = static_cast<std::byte>(static_cast<unsigned char>(raw[i]) ^
                                        static_cast<unsigned char>(raw[i - 1]));
  }
  hist_bytes(xored, h);
  r.xor_entropy_bpb = shannon_bpb(h, xored.size());

  if (raw.size() >= 2 && (raw.size() % 2) == 0) {
    auto w = as_u16(raw);
    std::uint64_t z = 0;
    std::unordered_set<int> exps;
    for (auto x : w) {
      if (x == 0) ++z;
      exps.insert((x >> 7) & 0xFF);
    }
    r.zero_frac = static_cast<double>(z) / static_cast<double>(w.size());
    r.unique_exponents = static_cast<int>(exps.size());
    r.plane_pack_bytes = plane_pack_u16(w);
    r.freq16_exc_bytes = freq16_exc(w, &r.top_u16_frac);
    std::vector<std::byte> sg, ep, mt;
    split_bf16(w, sg, ep, mt);
    hist_bytes(sg, h);
    r.sign_ent_bytes = entropy_bytes(shannon_bpb(h, sg.size()), sg.size());
    hist_bytes(ep, h);
    r.exp_ent_bytes = entropy_bytes(shannon_bpb(h, ep.size()), ep.size());
    hist_bytes(mt, h);
    r.mant_ent_bytes = entropy_bytes(shannon_bpb(h, mt.size()), mt.size());
    r.estimate_bytes = std::min(r.plane_pack_bytes, r.freq16_exc_bytes);
    auto xz = lzss_size(xored);
    if (xz + 8 < r.estimate_bytes) r.estimate_bytes = xz + 8;
    const auto split_sum = r.sign_ent_bytes + r.exp_ent_bytes + r.mant_ent_bytes;
    if (split_sum + 16 < r.estimate_bytes) r.estimate_bytes = split_sum + 16;
  } else {
    r.plane_pack_bytes = r.raw_bytes;
    r.freq16_exc_bytes = r.raw_bytes;
    r.estimate_bytes = std::min(r.lzss_bytes, r.entropy_bytes);
  }
  r.stack_bytes = r.estimate_bytes;

  {
    std::int64_t sh[] = {static_cast<std::int64_t>(raw.size())};
    auto enc = wpack_encode("probe", DType::u8, sh, raw);
    if (raw.size() >= 2 && (raw.size() % 2) == 0) {
      enc = wpack_encode("probe", DType::bf16, sh, raw);
    }
    if (enc) {
      r.packed_bytes = enc->payload.size();
      r.packed_codec = wpack_codec_name(enc->codec);
    } else {
      r.packed_bytes = raw.size();
      r.packed_codec = "identity";
    }
  }

  std::uint64_t best = r.packed_bytes ? r.packed_bytes : r.raw_bytes;
  r.best_rule = r.packed_codec.empty() ? "identity" : r.packed_codec;
  auto consider = [&](std::uint64_t sz, const char* rule) {
    if (sz > 0 && sz < best) {
      best = sz;
      r.best_rule = rule;
    }
  };
  consider(r.entropy_bytes, "entropy_est");
  consider(r.rle_bytes, "rle_est");
  consider(r.lzss_bytes, "lzss_est");
  consider(r.plane_pack_bytes, "plane_est");
  consider(r.freq16_exc_bytes, "freq16_est");
  consider(r.estimate_bytes, "estimate");
  consider(entropy_bytes(r.xor_entropy_bpb, raw.size()), "xor_est");

  const auto t1 = std::chrono::steady_clock::now();
  r.t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return r;
}

Result<std::vector<std::byte>> safetensors_read_prefix(const SafetensorFile& file,
                                                      const SafetensorInfo& info,
                                                      std::uint64_t max_bytes) {
  const auto nbytes = info.data_end - info.data_begin;
  const auto take = nbytes < max_bytes ? nbytes : max_bytes;
  std::vector<std::byte> buf(static_cast<std::size_t>(take));
  if (take == 0) return buf;
  std::ifstream in(file.path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, file.path.string()));
  in.seekg(static_cast<std::streamoff>(file.data_offset + info.data_begin));
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(take));
  if (static_cast<std::uint64_t>(in.gcount()) != take) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "prefix short"));
  }
  return buf;
}

Result<std::vector<CompressProbeRow>> compress_probe_dir(const std::filesystem::path& dir,
                                                         const CompressProbeOpts& opts) {
  std::vector<CompressProbeRow> rows;
  if (!std::filesystem::exists(dir)) {
    return std::unexpected(make_error(Errc::io, dir.string()));
  }
  int left = opts.max_tensors;
  std::vector<std::filesystem::path> files;
  for (auto& ent : std::filesystem::directory_iterator(dir)) {
    if (!ent.is_regular_file() || ent.path().extension() != ".safetensors") continue;
    auto fn = ent.path().filename().string();
    if (!opts.file_substr.empty() && fn.find(opts.file_substr) == std::string::npos &&
        ent.path().string().find(opts.file_substr) == std::string::npos)
      continue;
    files.push_back(ent.path());
  }
  std::sort(files.begin(), files.end());
  for (auto& path : files) {
    if (left <= 0) break;
    auto st = safetensors_open(path);
    if (!st) return std::unexpected(st.error());
    for (auto& info : st->tensors) {
      if (left <= 0) break;
      auto buf = safetensors_read_prefix(*st, info, opts.chunk_bytes);
      if (!buf) return std::unexpected(buf.error());
      auto row = probe_bytes(*buf, info.name);
      row.file = path.filename().string();
      row.raw_bytes = info.data_end - info.data_begin;
      row.probed_bytes = buf->size();
      rows.push_back(std::move(row));
      --left;
    }
  }
  return rows;
}

std::string compress_probe_json(std::span<const CompressProbeRow> rows) {
  std::ostringstream o;
  o << "{\"target_ratio\":0.5,\"rows\":[";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (i) o << ',';
    o << "{\"file\":\"" << r.file << "\",\"name\":\"" << r.name << "\",\"family\":\"" << r.family
      << "\",\"raw_bytes\":" << r.raw_bytes << ",\"probed_bytes\":" << r.probed_bytes
      << ",\"entropy_bpb\":" << r.entropy_bpb << ",\"xor_entropy_bpb\":" << r.xor_entropy_bpb
      << ",\"entropy_bytes\":" << r.entropy_bytes << ",\"rle_bytes\":" << r.rle_bytes
      << ",\"lzss_bytes\":" << r.lzss_bytes << ",\"sign_ent_bytes\":" << r.sign_ent_bytes
      << ",\"exp_ent_bytes\":" << r.exp_ent_bytes << ",\"mant_ent_bytes\":" << r.mant_ent_bytes
      << ",\"plane_pack_bytes\":" << r.plane_pack_bytes
      << ",\"freq16_exc_bytes\":" << r.freq16_exc_bytes << ",\"estimate_bytes\":" << r.estimate_bytes
      << ",\"stack_bytes\":" << r.stack_bytes << ",\"packed_bytes\":" << r.packed_bytes
      << ",\"packed_codec\":\"" << r.packed_codec << "\""
      << ",\"unique_exponents\":" << r.unique_exponents << ",\"zero_frac\":" << r.zero_frac
      << ",\"top_u16_frac\":" << r.top_u16_frac << ",\"best_rule\":\"" << r.best_rule
      << "\",\"t_ms\":" << r.t_ms << "}";
  }
  o << "]}";
  return o.str();
}

}  // namespace gyre
