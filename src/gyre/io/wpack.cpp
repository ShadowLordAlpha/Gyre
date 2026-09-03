#include "gyre/io/wpack.hpp"

#include "gyre/io/compress_probe.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace gyre {
namespace {

void wr_u8(std::vector<std::byte>& o, std::uint8_t v) { o.push_back(static_cast<std::byte>(v)); }
void wr_u16(std::vector<std::byte>& o, std::uint16_t v) {
  wr_u8(o, static_cast<std::uint8_t>(v));
  wr_u8(o, static_cast<std::uint8_t>(v >> 8));
}
void wr_u32(std::vector<std::byte>& o, std::uint32_t v) {
  wr_u16(o, static_cast<std::uint16_t>(v));
  wr_u16(o, static_cast<std::uint16_t>(v >> 16));
}
void wr_u64(std::vector<std::byte>& o, std::uint64_t v) {
  wr_u32(o, static_cast<std::uint32_t>(v));
  wr_u32(o, static_cast<std::uint32_t>(v >> 32));
}
void wr_i64(std::vector<std::byte>& o, std::int64_t v) { wr_u64(o, static_cast<std::uint64_t>(v)); }

std::uint8_t rd_u8(std::span<const std::byte> s, std::size_t& i) {
  return static_cast<std::uint8_t>(s[i++]);
}
std::uint16_t rd_u16(std::span<const std::byte> s, std::size_t& i) {
  auto a = rd_u8(s, i);
  auto b = rd_u8(s, i);
  return static_cast<std::uint16_t>(a | (b << 8));
}
std::uint32_t rd_u32(std::span<const std::byte> s, std::size_t& i) {
  auto a = rd_u16(s, i);
  auto b = rd_u16(s, i);
  return a | (static_cast<std::uint32_t>(b) << 16);
}
std::uint64_t rd_u64(std::span<const std::byte> s, std::size_t& i) {
  auto a = rd_u32(s, i);
  auto b = rd_u32(s, i);
  return a | (static_cast<std::uint64_t>(b) << 32);
}

constexpr std::uint8_t kPlaneList = 0;
constexpr std::uint8_t kPlaneDense = 1;

struct BitWriter {
  std::vector<std::byte> o;
  std::uint8_t acc{0};
  int n{0};
  void put(std::uint32_t v, int bits) {
    for (int i = 0; i < bits; ++i) {
      if (v & (1u << i)) acc = static_cast<std::uint8_t>(acc | (1u << n));
      ++n;
      if (n == 8) {
        o.push_back(std::byte{acc});
        acc = 0;
        n = 0;
      }
    }
  }
  void flush() {
    if (n) {
      o.push_back(std::byte{acc});
      acc = 0;
      n = 0;
    }
  }
};

struct BitReader {
  std::span<const std::byte> s;
  std::size_t i{0};
  int n{0};
  std::uint8_t acc{0};
  Result<std::uint32_t> get(int bits) {
    std::uint32_t v = 0;
    for (int b = 0; b < bits; ++b) {
      if (n == 0) {
        if (i >= s.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "bitstream"));
        acc = static_cast<std::uint8_t>(s[i++]);
        n = 8;
      }
      if (acc & 1) v |= (1u << b);
      acc = static_cast<std::uint8_t>(acc >> 1);
      --n;
    }
    return v;
  }
};

int ceil_log2_u(unsigned n) {
  if (n <= 1) return 0;
  int b = 0;
  unsigned x = n - 1;
  while (x) {
    ++b;
    x >>= 1;
  }
  return b;
}

std::vector<std::byte> encode_plane_exc(std::span<const std::uint16_t> w) {
  std::vector<std::byte> o;
  wr_u32(o, static_cast<std::uint32_t>(w.size()));
  std::uint16_t majority = 0;
  for (int b = 0; b < 16; ++b) {
    std::uint32_t ones = 0;
    for (auto x : w)
      if (x & (1u << b)) ++ones;
    if (ones * 2 >= w.size()) majority |= static_cast<std::uint16_t>(1u << b);
  }
  wr_u16(o, majority);
  const auto n = w.size();
  const auto dense_bytes = (n + 7) / 8;
  for (int b = 0; b < 16; ++b) {
    const bool maj = (majority >> b) & 1;
    std::vector<std::uint32_t> exc;
    std::vector<std::byte> dense(dense_bytes, std::byte{0});
    for (std::uint32_t i = 0; i < n; ++i) {
      const bool bit = (w[i] >> b) & 1;
      if (bit != maj) {
        exc.push_back(i);
        dense[i / 8] |= std::byte{static_cast<unsigned char>(1u << (i % 8))};
      }
    }
    const auto list_bytes = 4u + 4u * exc.size();
    if (list_bytes < dense_bytes) {
      wr_u8(o, kPlaneList);
      wr_u32(o, static_cast<std::uint32_t>(exc.size()));
      for (auto idx : exc) wr_u32(o, idx);
    } else {
      wr_u8(o, kPlaneDense);
      o.insert(o.end(), dense.begin(), dense.end());
    }
  }
  return o;
}

Result<std::vector<std::byte>> decode_plane_exc(std::span<const std::byte> p) {
  std::size_t i = 0;
  if (p.size() < 6) return std::unexpected(make_error(Errc::ckpt_corrupt, "plane_exc"));
  const auto n = rd_u32(p, i);
  const auto majority = rd_u16(p, i);
  std::vector<std::uint16_t> w(n, 0);
  const auto dense_bytes = (static_cast<std::size_t>(n) + 7) / 8;
  for (int b = 0; b < 16; ++b) {
    if (i >= p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "plane"));
    const auto kind = rd_u8(p, i);
    const bool maj = (majority >> b) & 1;
    if (maj) {
      for (auto& x : w) x = static_cast<std::uint16_t>(x | (1u << b));
    }
    if (kind == kPlaneList) {
      if (i + 4 > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "plane"));
      const auto ne = rd_u32(p, i);
      for (std::uint32_t e = 0; e < ne; ++e) {
        if (i + 4 > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "exc"));
        auto idx = rd_u32(p, i);
        if (idx >= n) return std::unexpected(make_error(Errc::ckpt_corrupt, "exc idx"));
        w[idx] ^= static_cast<std::uint16_t>(1u << b);
      }
    } else if (kind == kPlaneDense) {
      if (i + dense_bytes > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "dense"));
      for (std::uint32_t k = 0; k < n; ++k) {
        const auto by = static_cast<std::uint8_t>(p[i + k / 8]);
        if (by & (1u << (k % 8))) w[k] ^= static_cast<std::uint16_t>(1u << b);
      }
      i += dense_bytes;
    } else {
      return std::unexpected(make_error(Errc::ckpt_corrupt, "plane kind"));
    }
  }
  std::vector<std::byte> out(static_cast<std::size_t>(n) * 2);
  std::memcpy(out.data(), w.data(), out.size());
  return out;
}

std::vector<std::byte> encode_exp_alpha(std::span<const std::uint16_t> w) {
  std::vector<std::uint8_t> alpha;
  bool seen[256]{};
  for (auto x : w) {
    const auto e = static_cast<std::uint8_t>((x >> 7) & 0xFF);
    if (!seen[e]) {
      seen[e] = true;
      alpha.push_back(e);
    }
  }
  std::sort(alpha.begin(), alpha.end());
  std::uint8_t idx_of[256];
  std::memset(idx_of, 0, sizeof(idx_of));
  for (std::size_t i = 0; i < alpha.size(); ++i) idx_of[alpha[i]] = static_cast<std::uint8_t>(i);
  const int ebits = ceil_log2_u(static_cast<unsigned>(alpha.size()));

  std::vector<std::byte> o;
  wr_u32(o, static_cast<std::uint32_t>(w.size()));
  wr_u8(o, alpha.size() == 256 ? 0 : static_cast<std::uint8_t>(alpha.size()));
  wr_u8(o, static_cast<std::uint8_t>(ebits));
  for (auto e : alpha) wr_u8(o, e);

  BitWriter bw;
  for (auto x : w) {
    bw.put((x >> 15) & 1u, 1);
    if (ebits) bw.put(idx_of[(x >> 7) & 0xFF], ebits);
    bw.put(x & 0x7Fu, 7);
  }
  bw.flush();
  o.insert(o.end(), bw.o.begin(), bw.o.end());
  return o;
}

Result<std::vector<std::byte>> decode_exp_alpha(std::span<const std::byte> p) {
  std::size_t i = 0;
  if (p.size() < 6) return std::unexpected(make_error(Errc::ckpt_corrupt, "exp_alpha"));
  const auto n = rd_u32(p, i);
  std::uint16_t nalpha = rd_u8(p, i);
  const auto ebits = rd_u8(p, i);
  if (nalpha == 0 && ebits == 8) nalpha = 256;
  if (i + nalpha > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "alphabet"));
  std::vector<std::uint8_t> alpha(nalpha);
  for (std::uint8_t k = 0; k < nalpha; ++k) alpha[k] = rd_u8(p, i);
  BitReader br;
  br.s = p.subspan(i);
  std::vector<std::uint16_t> w(n);
  for (std::uint32_t k = 0; k < n; ++k) {
    auto s = br.get(1);
    if (!s) return std::unexpected(s.error());
    std::uint32_t ei = 0;
    if (ebits) {
      auto e = br.get(ebits);
      if (!e) return std::unexpected(e.error());
      ei = *e;
    }
    auto m = br.get(7);
    if (!m) return std::unexpected(m.error());
    if (ei >= nalpha && nalpha) return std::unexpected(make_error(Errc::ckpt_corrupt, "exp idx"));
    const std::uint16_t exp = nalpha ? alpha[ei] : 0;
    w[k] = static_cast<std::uint16_t>((*s << 15) | (exp << 7) | (*m & 0x7Fu));
  }
  std::vector<std::byte> out(static_cast<std::size_t>(n) * 2);
  std::memcpy(out.data(), w.data(), out.size());
  return out;
}

constexpr std::uint16_t kLfsrPoly = 0xB400;  // 16-bit Galois taps
constexpr std::uint16_t kLfsrBlock = 64;

std::uint16_t lfsr16_next(std::uint16_t s) {
  const std::uint16_t bit = static_cast<std::uint16_t>(s & 1u);
  s = static_cast<std::uint16_t>(s >> 1);
  if (bit) s = static_cast<std::uint16_t>(s ^ kLfsrPoly);
  return s ? s : 1;
}

std::uint32_t plane_score(std::span<const std::uint16_t> w) {
  std::uint32_t ones[16]{};
  for (auto x : w) {
    for (int b = 0; b < 16; ++b)
      if (x & (1u << b)) ++ones[b];
  }
  std::uint32_t t = 0;
  const auto n = static_cast<std::uint32_t>(w.size());
  for (int b = 0; b < 16; ++b) t += ones[b] < n - ones[b] ? ones[b] : n - ones[b];
  return t;
}

void lfsr_xor_block(std::span<std::uint16_t> blk, std::uint16_t seed) {
  auto s = seed ? seed : std::uint16_t{1};
  for (auto& x : blk) {
    x = static_cast<std::uint16_t>(x ^ s);
    s = lfsr16_next(s);
  }
}

std::uint16_t pick_lfsr_seed(std::span<const std::uint16_t> blk) {
  const auto orig = plane_score(blk);
  std::uint16_t best_seed = 0;
  auto best = orig;
  std::uint16_t cands[10];
  int nc = 0;
  auto add = [&](std::uint16_t s) {
    if (!s) return;
    for (int i = 0; i < nc; ++i)
      if (cands[i] == s) return;
    if (nc < 10) cands[nc++] = s;
  };
  add(blk[0]);
  add(static_cast<std::uint16_t>(blk[0] ^ 1u));
  add(1);
  add(2);
  add(3);
  add(5);
  add(7);
  add(11);
  add(13);
  add(17);
  std::vector<std::uint16_t> tmp(blk.size());
  for (int i = 0; i < nc; ++i) {
    std::memcpy(tmp.data(), blk.data(), blk.size() * 2);
    lfsr_xor_block(tmp, cands[i]);
    const auto sc = plane_score(tmp);
    if (sc < best) {
      best = sc;
      best_seed = cands[i];
    }
  }
  return best_seed;
}

std::vector<std::byte> encode_inner(std::span<const std::uint16_t> w, std::uint8_t* codec_out) {
  auto a = encode_plane_exc(w);
  auto b = encode_exp_alpha(w);
  if (b.size() < a.size()) {
    *codec_out = static_cast<std::uint8_t>(WpackCodec::exp_alpha);
    return b;
  }
  *codec_out = static_cast<std::uint8_t>(WpackCodec::plane_exc);
  return a;
}

std::vector<std::byte> encode_lfsr_pred(std::span<const std::uint16_t> w) {
  const auto n = w.size();
  if (n < 8) return {};
  const auto C = kLfsrBlock;
  const auto nblocks = (n + C - 1) / C;
  if (nblocks > 65535) return {};
  std::vector<std::uint16_t> seeds(nblocks, 0);
  std::vector<std::uint16_t> resid(w.begin(), w.end());
  int hits = 0;
  for (std::size_t b = 0; b < nblocks; ++b) {
    const auto lo = b * C;
    const auto hi = std::min(lo + C, n);
    auto blk = std::span<const std::uint16_t>(w.data() + lo, hi - lo);
    const auto seed = pick_lfsr_seed(blk);
    if (!seed) continue;
    seeds[b] = seed;
    auto r = std::span<std::uint16_t>(resid.data() + lo, hi - lo);
    lfsr_xor_block(r, seed);
    ++hits;
  }
  if (!hits) return {};
  std::uint8_t inner_c = 0;
  auto inner = encode_inner(resid, &inner_c);
  std::vector<std::byte> o;
  wr_u32(o, static_cast<std::uint32_t>(n));
  wr_u16(o, C);
  wr_u16(o, static_cast<std::uint16_t>(nblocks));
  for (auto s : seeds) wr_u16(o, s);
  wr_u8(o, inner_c);
  wr_u32(o, static_cast<std::uint32_t>(inner.size()));
  o.insert(o.end(), inner.begin(), inner.end());
  return o;
}

Result<std::vector<std::byte>> decode_lfsr_pred(std::span<const std::byte> p) {
  std::size_t i = 0;
  if (p.size() < 12) return std::unexpected(make_error(Errc::ckpt_corrupt, "lfsr_pred"));
  const auto n = rd_u32(p, i);
  const auto C = rd_u16(p, i);
  const auto nblocks = rd_u16(p, i);
  if (!C || nblocks != (n + C - 1) / C) return std::unexpected(make_error(Errc::ckpt_corrupt, "lfsr blocks"));
  if (i + static_cast<std::size_t>(nblocks) * 2 + 5 > p.size()) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "lfsr seeds"));
  }
  std::vector<std::uint16_t> seeds(nblocks);
  for (std::uint16_t b = 0; b < nblocks; ++b) seeds[b] = rd_u16(p, i);
  const auto inner_c = rd_u8(p, i);
  const auto ilen = rd_u32(p, i);
  if (i + ilen > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "lfsr inner"));
  auto inner = p.subspan(i, ilen);
  Result<std::vector<std::byte>> dec = std::unexpected(make_error(Errc::unsupported, "lfsr inner"));
  if (inner_c == static_cast<std::uint8_t>(WpackCodec::plane_exc)) dec = decode_plane_exc(inner);
  else if (inner_c == static_cast<std::uint8_t>(WpackCodec::exp_alpha)) dec = decode_exp_alpha(inner);
  else if (inner_c == static_cast<std::uint8_t>(WpackCodec::identity)) {
    dec = std::vector<std::byte>(inner.begin(), inner.end());
  }
  if (!dec) return std::unexpected(dec.error());
  if (dec->size() != static_cast<std::size_t>(n) * 2) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "lfsr residual size"));
  }
  auto w = std::span<std::uint16_t>(reinterpret_cast<std::uint16_t*>(dec->data()), n);
  for (std::size_t b = 0; b < nblocks; ++b) {
    if (!seeds[b]) continue;
    const auto lo = b * C;
    const auto hi = std::min(lo + C, static_cast<std::size_t>(n));
    lfsr_xor_block(std::span<std::uint16_t>(w.data() + lo, hi - lo), seeds[b]);
  }
  return *dec;
}

struct LaneWin {
  int stride{2};
  int lane{1};
  std::uint8_t c{0};
  std::uint32_t begin{0};
  std::uint32_t len{0};
  std::int64_t gain{0};
};

bool kadane_lane(std::span<const std::byte> raw, int stride, int lane, std::uint8_t c, int k,
                 LaneWin* out) {
  const auto n = raw.size();
  if (n <= static_cast<std::size_t>(lane)) return false;
  const auto n_lane = (n - static_cast<std::size_t>(lane) + static_cast<std::size_t>(stride) - 1) /
                      static_cast<std::size_t>(stride);
  std::int64_t best = 0, cur = 0;
  std::uint32_t best_b = 0, best_e = 0, cur_b = 0;
  bool in = false;
  for (std::uint32_t t = 0; t < n_lane; ++t) {
    const auto idx = static_cast<std::size_t>(lane) + static_cast<std::size_t>(t) * stride;
    const bool hit = idx < n && static_cast<std::uint8_t>(raw[idx]) == c;
    const std::int64_t v = hit ? 1 : -static_cast<std::int64_t>(k);
    if (!in || cur + v < v) {
      cur = v;
      cur_b = t;
      in = true;
    } else {
      cur += v;
    }
    if (cur > best) {
      best = cur;
      best_b = cur_b;
      best_e = t + 1;
    }
  }
  if (best <= 0 || best_e <= best_b || (best_e - best_b) < 16) return false;
  out->stride = stride;
  out->lane = lane;
  out->c = c;
  out->begin = best_b;
  out->len = best_e - best_b;
  out->gain = best;
  return true;
}

bool lane_covered(std::size_t i, const LaneWin& w) {
  if (static_cast<int>(i % static_cast<std::size_t>(w.stride)) != w.lane) return false;
  const auto t = static_cast<std::uint32_t>(i / static_cast<std::size_t>(w.stride));
  return t >= w.begin && t < w.begin + w.len;
}

std::vector<std::byte> encode_const_lane(std::span<const std::byte> raw) {
  if (raw.size() < 32) return {};
  LaneWin best{};
  bool any = false;
  const int strides[] = {2, 1};
  for (int stride : strides) {
    for (int lane = 0; lane < stride; ++lane) {
      std::uint32_t freq[256]{};
      const auto n = raw.size();
      const auto n_lane = n > static_cast<std::size_t>(lane)
                              ? (n - static_cast<std::size_t>(lane) + stride - 1) / stride
                              : 0;
      for (std::size_t t = 0; t < n_lane; ++t) {
        const auto idx = static_cast<std::size_t>(lane) + t * stride;
        if (idx < n) ++freq[static_cast<std::uint8_t>(raw[idx])];
      }
      std::uint8_t cands[8];
      int nc = 0;
      auto addc = [&](std::uint8_t v) {
        if (!freq[v]) return;
        for (int i = 0; i < nc; ++i)
          if (cands[i] == v) return;
        if (nc < 8) cands[nc++] = v;
      };
      std::uint8_t mode = 0;
      std::uint32_t bestf = 0;
      for (int v = 0; v < 256; ++v) {
        if (freq[v] > bestf) {
          bestf = freq[v];
          mode = static_cast<std::uint8_t>(v);
        }
      }
      addc(mode);
      addc(0x40);
      addc(0x41);
      addc(0x3f);
      addc(0x00);
      addc(0xff);
      const int k = n_lane > 65535 ? 4 : 2;
      for (int i = 0; i < nc; ++i) {
        LaneWin w{};
        if (!kadane_lane(raw, stride, lane, cands[i], k, &w)) continue;
        if (!any || w.gain > best.gain) {
          best = w;
          any = true;
        }
      }
    }
  }
  if (!any) return {};
  const int k = best.len > 65535 ? 4 : 2;
  std::vector<std::uint32_t> exc_i;
  std::vector<std::uint8_t> exc_v;
  for (std::uint32_t t = 0; t < best.len; ++t) {
    const auto idx =
        static_cast<std::size_t>(best.lane) + static_cast<std::size_t>(best.begin + t) * best.stride;
    if (idx >= raw.size()) break;
    const auto v = static_cast<std::uint8_t>(raw[idx]);
    if (v != best.c) {
      exc_i.push_back(t);
      exc_v.push_back(v);
    }
  }
  constexpr int kHdr = 28;
  const std::int64_t saved =
      static_cast<std::int64_t>(best.len - exc_i.size()) -
      kHdr - static_cast<std::int64_t>(exc_i.size()) * k;
  if (saved <= 16) return {};

  std::vector<std::byte> leftover;
  leftover.reserve(raw.size() - (best.len - exc_i.size()));
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (lane_covered(i, best)) continue;
    leftover.push_back(raw[i]);
  }
  std::uint8_t inner_c = static_cast<std::uint8_t>(WpackCodec::identity);
  std::vector<std::byte> inner = leftover;
  if (leftover.size() >= 2 && (leftover.size() % 2) == 0) {
    auto lw = std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(leftover.data()),
                                            leftover.size() / 2);
    inner = encode_inner(lw, &inner_c);
    if (inner.size() + 8 >= leftover.size()) {
      inner = leftover;
      inner_c = static_cast<std::uint8_t>(WpackCodec::identity);
    }
  }

  std::vector<std::byte> o;
  wr_u32(o, static_cast<std::uint32_t>(raw.size()));
  wr_u8(o, static_cast<std::uint8_t>(best.stride));
  wr_u8(o, static_cast<std::uint8_t>(best.lane));
  wr_u8(o, best.c);
  wr_u8(o, static_cast<std::uint8_t>(k));
  wr_u32(o, best.begin);
  wr_u32(o, best.len);
  wr_u32(o, static_cast<std::uint32_t>(exc_i.size()));
  for (std::size_t e = 0; e < exc_i.size(); ++e) {
    if (k == 2) wr_u16(o, static_cast<std::uint16_t>(exc_i[e]));
    else wr_u32(o, exc_i[e]);
    wr_u8(o, exc_v[e]);
  }
  wr_u8(o, inner_c);
  wr_u32(o, static_cast<std::uint32_t>(inner.size()));
  o.insert(o.end(), inner.begin(), inner.end());
  return o;
}

Result<std::vector<std::byte>> decode_const_lane(std::span<const std::byte> p) {
  std::size_t i = 0;
  if (p.size() < 24) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane"));
  const auto n = rd_u32(p, i);
  const auto stride = rd_u8(p, i);
  const auto lane = rd_u8(p, i);
  const auto c = rd_u8(p, i);
  const auto k = rd_u8(p, i);
  const auto begin = rd_u32(p, i);
  const auto len = rd_u32(p, i);
  const auto nexc = rd_u32(p, i);
  if (!stride || lane >= stride || (k != 2 && k != 4)) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane hdr"));
  }
  const auto excsz = static_cast<std::size_t>(nexc) * (k + 1);
  if (i + excsz + 5 > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane exc"));
  std::vector<std::uint32_t> exc_i(nexc);
  std::vector<std::uint8_t> exc_v(nexc);
  for (std::uint32_t e = 0; e < nexc; ++e) {
    exc_i[e] = k == 2 ? rd_u16(p, i) : rd_u32(p, i);
    exc_v[e] = rd_u8(p, i);
    if (exc_i[e] >= len) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane idx"));
  }
  const auto inner_c = rd_u8(p, i);
  const auto ilen = rd_u32(p, i);
  if (i + ilen > p.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane left"));
  auto inner = p.subspan(i, ilen);
  std::vector<std::byte> leftover;
  if (inner_c == static_cast<std::uint8_t>(WpackCodec::identity)) {
    leftover.assign(inner.begin(), inner.end());
  } else if (inner_c == static_cast<std::uint8_t>(WpackCodec::plane_exc)) {
    auto d = decode_plane_exc(inner);
    if (!d) return std::unexpected(d.error());
    leftover = std::move(*d);
  } else if (inner_c == static_cast<std::uint8_t>(WpackCodec::exp_alpha)) {
    auto d = decode_exp_alpha(inner);
    if (!d) return std::unexpected(d.error());
    leftover = std::move(*d);
  } else {
    return std::unexpected(make_error(Errc::unsupported, "const_lane inner"));
  }
  LaneWin w;
  w.stride = stride;
  w.lane = lane;
  w.c = c;
  w.begin = begin;
  w.len = len;
  std::vector<std::byte> out(n);
  std::size_t li = 0;
  for (std::size_t b = 0; b < n; ++b) {
    if (lane_covered(b, w)) {
      out[b] = std::byte{c};
    } else {
      if (li >= leftover.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane left"));
      out[b] = leftover[li++];
    }
  }
  for (std::uint32_t e = 0; e < nexc; ++e) {
    const auto idx = static_cast<std::size_t>(lane) +
                     static_cast<std::size_t>(begin + exc_i[e]) * stride;
    if (idx >= n) return std::unexpected(make_error(Errc::ckpt_corrupt, "const_lane poke"));
    out[idx] = std::byte{exc_v[e]};
  }
  return out;
}

}  // namespace

Result<WpackTensor> wpack_encode(std::string name, DType dtype, std::span<const std::int64_t> shape,
                                 std::span<const std::byte> raw) {
  WpackTensor t;
  t.name = std::move(name);
  t.dtype = dtype;
  t.shape.assign(shape.begin(), shape.end());
  t.raw_bytes = raw.size();
  t.codec = WpackCodec::identity;
  t.payload.assign(raw.begin(), raw.end());
  if ((dtype == DType::bf16 || dtype == DType::f16) && raw.size() >= 2 && (raw.size() % 2) == 0) {
    auto w = std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(raw.data()),
                                            raw.size() / 2);
    auto consider = [&](WpackCodec c, std::vector<std::byte>&& enc) {
      if (enc.size() + 16 < t.payload.size()) {
        t.codec = c;
        t.payload = std::move(enc);
      }
    };
    consider(WpackCodec::plane_exc, encode_plane_exc(w));
    consider(WpackCodec::exp_alpha, encode_exp_alpha(w));
    auto lfsr = encode_lfsr_pred(w);
    if (!lfsr.empty()) consider(WpackCodec::lfsr_pred, std::move(lfsr));
    auto lane = encode_const_lane(raw);
    if (!lane.empty()) consider(WpackCodec::const_lane, std::move(lane));
  } else {
    auto lane = encode_const_lane(raw);
    if (!lane.empty() && lane.size() + 16 < t.payload.size()) {
      t.codec = WpackCodec::const_lane;
      t.payload = std::move(lane);
    }
  }
  return t;
}

Result<std::vector<std::byte>> wpack_decode(const WpackTensor& t) {
  if (t.codec == WpackCodec::identity) return t.payload;
  if (t.codec == WpackCodec::plane_exc) return decode_plane_exc(t.payload);
  if (t.codec == WpackCodec::exp_alpha) return decode_exp_alpha(t.payload);
  if (t.codec == WpackCodec::lfsr_pred) return decode_lfsr_pred(t.payload);
  if (t.codec == WpackCodec::const_lane) return decode_const_lane(t.payload);
  return std::unexpected(make_error(Errc::unsupported, "wpack codec"));
}

Result<void> wpack_save(const std::filesystem::path& path, const WpackFile& file) {
  std::vector<std::byte> o;
  o.insert(o.end(), {std::byte{'G'}, std::byte{'Y'}, std::byte{'W'}, std::byte{'P'}, std::byte{'1'},
                     std::byte{0}, std::byte{0}, std::byte{0}});
  wr_u32(o, 1);
  wr_u16(o, static_cast<std::uint16_t>(file.source.size()));
  for (char c : file.source) wr_u8(o, static_cast<std::uint8_t>(c));
  wr_u32(o, static_cast<std::uint32_t>(file.tensors.size()));
  for (auto& t : file.tensors) {
    wr_u16(o, static_cast<std::uint16_t>(t.name.size()));
    for (char c : t.name) wr_u8(o, static_cast<std::uint8_t>(c));
    wr_u8(o, static_cast<std::uint8_t>(t.dtype));
    wr_u8(o, static_cast<std::uint8_t>(t.shape.size()));
    for (int i = 0; i < 8; ++i) wr_i64(o, i < static_cast<int>(t.shape.size()) ? t.shape[static_cast<std::size_t>(i)] : 0);
    wr_u8(o, static_cast<std::uint8_t>(t.codec));
    wr_u64(o, t.raw_bytes);
    wr_u64(o, t.payload.size());
    o.insert(o.end(), t.payload.begin(), t.payload.end());
  }
  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary);
  if (!out) return std::unexpected(make_error(Errc::io, path.string()));
  out.write(reinterpret_cast<const char*>(o.data()), static_cast<std::streamsize>(o.size()));
  return {};
}

Result<WpackFile> wpack_load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, path.string()));
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::span<const std::byte> s(reinterpret_cast<const std::byte*>(raw.data()), raw.size());
  if (s.size() < 14 || std::memcmp(s.data(), "GYWP1", 5) != 0) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "wpack magic"));
  }
  std::size_t i = 8;
  auto ver = rd_u32(s, i);
  if (ver != 1) return std::unexpected(make_error(Errc::unsupported, "wpack version"));
  auto sl = rd_u16(s, i);
  WpackFile f;
  f.source.assign(reinterpret_cast<const char*>(s.data() + i), sl);
  i += sl;
  auto n = rd_u32(s, i);
  f.tensors.reserve(n);
  for (std::uint32_t t = 0; t < n; ++t) {
    WpackTensor wt;
    auto nl = rd_u16(s, i);
    wt.name.assign(reinterpret_cast<const char*>(s.data() + i), nl);
    i += nl;
    wt.dtype = static_cast<DType>(rd_u8(s, i));
    auto ndim = rd_u8(s, i);
    wt.shape.resize(ndim);
    for (int k = 0; k < 8; ++k) {
      auto v = static_cast<std::int64_t>(rd_u64(s, i));
      if (k < ndim) wt.shape[static_cast<std::size_t>(k)] = v;
    }
    wt.codec = static_cast<WpackCodec>(rd_u8(s, i));
    wt.raw_bytes = rd_u64(s, i);
    auto plen = rd_u64(s, i);
    if (i + plen > s.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "wpack payload"));
    wt.payload.assign(s.begin() + static_cast<std::ptrdiff_t>(i),
                      s.begin() + static_cast<std::ptrdiff_t>(i + plen));
    i += static_cast<std::size_t>(plen);
    f.tensors.push_back(std::move(wt));
  }
  return f;
}

Result<WpackFile> wpack_from_safetensors(const SafetensorFile& st, std::uint64_t max_bytes) {
  WpackFile f;
  f.source = st.path.filename().string();
  for (auto& info : st.tensors) {
    const auto nbytes = info.data_end - info.data_begin;
    if (nbytes > max_bytes) continue;
    auto buf = safetensors_read_prefix(st, info, nbytes);
    if (!buf) return std::unexpected(buf.error());
    auto enc = wpack_encode(info.name, info.dtype, info.shape, *buf);
    if (!enc) return std::unexpected(enc.error());
    f.tensors.push_back(std::move(*enc));
  }
  return f;
}

}  // namespace gyre
