#include "gyre/io/codec.hpp"

#include "gyre/checkpoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(GYRE_ZFP)
#include <zfp.h>
#endif

namespace gyre {
namespace {

constexpr int kAlpVec = 256;
constexpr int kAlpMaxE = 10;

double pow10_tbl[kAlpMaxE + 1];
bool pow10_once = false;

void init_pow10() {
  if (pow10_once) return;
  pow10_once = true;
  pow10_tbl[0] = 1.0;
  for (int i = 1; i <= kAlpMaxE; ++i) pow10_tbl[i] = pow10_tbl[i - 1] * 10.0;
}

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
void wr_f32(std::vector<std::byte>& o, float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  wr_u32(o, u);
}

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
float rd_f32(std::span<const std::byte> s, std::size_t& i) {
  auto u = rd_u32(s, i);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int bit_width(std::uint64_t span) {
  if (span == 0) return 0;
  int b = 0;
  while (span) {
    ++b;
    span >>= 1;
  }
  return b;
}

void pack_bits(std::vector<std::byte>& o, std::span<const std::uint64_t> v, int bits) {
  if (bits <= 0 || v.empty()) return;
  const auto nbytes = (v.size() * static_cast<std::size_t>(bits) + 7) / 8;
  const auto start = o.size();
  o.resize(start + nbytes, std::byte{0});
  const auto mask = (bits == 64) ? ~std::uint64_t{0} : ((1ull << bits) - 1);
  for (std::size_t i = 0; i < v.size(); ++i) {
    auto x = v[i] & mask;
    const auto bitpos = i * static_cast<std::size_t>(bits);
    for (int b = 0; b < bits; ++b) {
      if (x & (1ull << b)) {
        const auto off = bitpos + static_cast<std::size_t>(b);
        o[start + off / 8] |= static_cast<std::byte>(1u << (off % 8));
      }
    }
  }
}

Result<void> unpack_bits(std::span<const std::byte> s, std::size_t& i, std::vector<std::uint64_t>& v,
                         int nvals, int bits) {
  v.assign(static_cast<std::size_t>(nvals), 0);
  if (bits <= 0 || nvals <= 0) return {};
  const auto nbytes = (static_cast<std::size_t>(nvals) * static_cast<std::size_t>(bits) + 7) / 8;
  if (i + nbytes > s.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "alp bitstream"));
  for (int k = 0; k < nvals; ++k) {
    std::uint64_t x = 0;
    const auto bitpos = static_cast<std::size_t>(k) * static_cast<std::size_t>(bits);
    for (int b = 0; b < bits; ++b) {
      const auto off = bitpos + static_cast<std::size_t>(b);
      auto byte = static_cast<std::uint8_t>(s[i + off / 8]);
      if (byte & (1u << (off % 8))) x |= (1ull << b);
    }
    v[static_cast<std::size_t>(k)] = x;
  }
  i += nbytes;
  return {};
}

bool alp_try(std::span<const float> x, int e, int f, std::vector<std::int64_t>& enc,
             std::vector<std::uint16_t>& exc_pos, std::vector<float>& exc_val) {
  enc.assign(x.size(), 0);
  exc_pos.clear();
  exc_val.clear();
  const double scale = pow10_tbl[e] / pow10_tbl[f];
  const double inv = pow10_tbl[f] / pow10_tbl[e];
  for (std::size_t i = 0; i < x.size(); ++i) {
    float xv = x[i];
    if (!std::isfinite(xv)) {
      enc[i] = 0;
      exc_pos.push_back(static_cast<std::uint16_t>(i));
      exc_val.push_back(xv);
      continue;
    }
    double q = std::round(static_cast<double>(xv) * scale);
    if (q > static_cast<double>(std::numeric_limits<std::int64_t>::max()) ||
        q < static_cast<double>(std::numeric_limits<std::int64_t>::min())) {
      enc[i] = 0;
      exc_pos.push_back(static_cast<std::uint16_t>(i));
      exc_val.push_back(xv);
      continue;
    }
    auto iv = static_cast<std::int64_t>(q);
    float back = static_cast<float>(static_cast<double>(iv) * inv);
    std::uint32_t a, b;
    std::memcpy(&a, &xv, 4);
    std::memcpy(&b, &back, 4);
    if (a != b) {
      enc[i] = 0;
      exc_pos.push_back(static_cast<std::uint16_t>(i));
      exc_val.push_back(xv);
    } else {
      enc[i] = iv;
    }
  }
  return true;
}

int alp_score(const std::vector<std::int64_t>& enc, std::size_t n_exc) {
  std::int64_t mn = 0, mx = 0;
  bool any = false;
  for (auto v : enc) {
    if (!any) {
      mn = mx = v;
      any = true;
    } else {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  std::uint64_t span = any ? static_cast<std::uint64_t>(mx - mn) : 0;
  int bw = bit_width(span);
  return static_cast<int>(n_exc) * 40 + bw * static_cast<int>(enc.size());
}

Result<std::vector<std::byte>> alp_compress_f32(std::span<const float> data) {
  init_pow10();
  std::vector<std::byte> o;
  wr_u32(o, 0x31504c41u);  // ALP1
  wr_u32(o, static_cast<std::uint32_t>(data.size()));
  wr_u32(o, kAlpVec);
  std::vector<std::int64_t> enc, best_enc;
  std::vector<std::uint16_t> exc, best_exc;
  std::vector<float> exc_v, best_exc_v;
  for (std::size_t off = 0; off < data.size(); off += kAlpVec) {
    const auto n = std::min(static_cast<std::size_t>(kAlpVec), data.size() - off);
    auto sl = data.subspan(off, n);
    int best = std::numeric_limits<int>::max(), be = 0, bf = 0;
    for (int e = 0; e <= kAlpMaxE; ++e) {
      for (int f = 0; f <= e; ++f) {
        alp_try(sl, e, f, enc, exc, exc_v);
        int sc = alp_score(enc, exc.size());
        if (sc < best) {
          best = sc;
          be = e;
          bf = f;
          best_enc = enc;
          best_exc = exc;
          best_exc_v = exc_v;
        }
      }
    }
    alp_try(sl, be, bf, best_enc, best_exc, best_exc_v);
    wr_u8(o, static_cast<std::uint8_t>(be));
    wr_u8(o, static_cast<std::uint8_t>(bf));
    wr_u16(o, static_cast<std::uint16_t>(n));
    wr_u16(o, static_cast<std::uint16_t>(best_exc.size()));
    std::int64_t mn = 0, mx = 0;
    if (!best_enc.empty()) {
      mn = mx = best_enc[0];
      for (auto v : best_enc) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
    }
    wr_u64(o, static_cast<std::uint64_t>(mn));
    std::vector<std::uint64_t> rel(best_enc.size());
    for (std::size_t i = 0; i < best_enc.size(); ++i)
      rel[i] = static_cast<std::uint64_t>(best_enc[i] - mn);
    int bw = bit_width(static_cast<std::uint64_t>(mx - mn));
    wr_u8(o, static_cast<std::uint8_t>(bw));
    pack_bits(o, rel, bw);
    for (std::size_t i = 0; i < best_exc.size(); ++i) {
      wr_u16(o, best_exc[i]);
      wr_f32(o, best_exc_v[i]);
    }
  }
  return o;
}

Result<std::vector<std::byte>> alp_decompress_f32(std::span<const std::byte> packed, std::size_t nfloat) {
  init_pow10();
  std::size_t i = 0;
  if (packed.size() < 12) return std::unexpected(make_error(Errc::ckpt_corrupt, "alp short"));
  auto mag = rd_u32(packed, i);
  if (mag != 0x31504c41u) return std::unexpected(make_error(Errc::ckpt_corrupt, "alp magic"));
  auto n = rd_u32(packed, i);
  auto vec = rd_u32(packed, i);
  if (n != nfloat) return std::unexpected(make_error(Errc::invalid_shape, "alp n"));
  (void)vec;
  std::vector<float> out(nfloat);
  std::size_t off = 0;
  while (off < nfloat) {
    if (i + 6 > packed.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "alp chunk"));
    int e = rd_u8(packed, i);
    int f = rd_u8(packed, i);
    int cn = rd_u16(packed, i);
    int nexc = rd_u16(packed, i);
    auto mn = static_cast<std::int64_t>(rd_u64(packed, i));
    if (i >= packed.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "alp bits"));
    int bits = rd_u8(packed, i);
    std::vector<std::uint64_t> rel;
    auto up = unpack_bits(packed, i, rel, cn, bits);
    if (!up) return std::unexpected(up.error());
    const double inv = pow10_tbl[f] / pow10_tbl[e];
    for (int k = 0; k < cn; ++k) {
      auto iv = mn + static_cast<std::int64_t>(rel[static_cast<std::size_t>(k)]);
      out[off + static_cast<std::size_t>(k)] = static_cast<float>(static_cast<double>(iv) * inv);
    }
    for (int k = 0; k < nexc; ++k) {
      auto pos = rd_u16(packed, i);
      auto val = rd_f32(packed, i);
      if (pos < static_cast<std::uint16_t>(cn)) out[off + pos] = val;
    }
    off += static_cast<std::size_t>(cn);
  }
  std::vector<std::byte> raw(nfloat * 4);
  std::memcpy(raw.data(), out.data(), raw.size());
  return raw;
}

#if defined(GYRE_ZFP)
Result<std::vector<std::byte>> zfp_compress_f32(std::span<const float> data,
                                               std::span<const std::int64_t> shape) {
  zfp_type type = zfp_type_float;
  zfp_field* field = nullptr;
  const auto n = static_cast<std::size_t>(data.size());
  auto* ptr = const_cast<float*>(data.data());
  if (shape.size() == 1)
    field = zfp_field_1d(ptr, type, static_cast<unsigned>(shape[0]));
  else if (shape.size() == 2)
    field = zfp_field_2d(ptr, type, static_cast<unsigned>(shape[1]), static_cast<unsigned>(shape[0]));
  else if (shape.size() == 3)
    field = zfp_field_3d(ptr, type, static_cast<unsigned>(shape[2]), static_cast<unsigned>(shape[1]),
                         static_cast<unsigned>(shape[0]));
  else
    field = zfp_field_1d(ptr, type, static_cast<unsigned>(n));
  if (!field) return std::unexpected(make_error(Errc::unsupported, "zfp field"));
  zfp_stream* z = zfp_stream_open(nullptr);
  zfp_stream_set_reversible(z);
  auto bufsize = zfp_stream_maximum_size(z, field);
  std::vector<std::byte> buf(bufsize);
  bitstream* stream = stream_open(buf.data(), buf.size());
  zfp_stream_set_bit_stream(z, stream);
  zfp_stream_rewind(z);
  auto sz = zfp_compress(z, field);
  zfp_field_free(field);
  zfp_stream_close(z);
  stream_close(stream);
  if (!sz) return std::unexpected(make_error(Errc::unsupported, "zfp compress"));
  buf.resize(sz);
  return buf;
}

Result<std::vector<std::byte>> zfp_decompress_f32(std::span<const std::byte> packed,
                                                 std::span<const std::int64_t> shape, std::size_t raw_bytes) {
  std::vector<std::byte> raw(raw_bytes);
  auto n = raw_bytes / 4;
  zfp_type type = zfp_type_float;
  zfp_field* field = nullptr;
  auto* ptr = reinterpret_cast<float*>(raw.data());
  if (shape.size() == 1)
    field = zfp_field_1d(ptr, type, static_cast<unsigned>(shape[0]));
  else if (shape.size() == 2)
    field = zfp_field_2d(ptr, type, static_cast<unsigned>(shape[1]), static_cast<unsigned>(shape[0]));
  else if (shape.size() == 3)
    field = zfp_field_3d(ptr, type, static_cast<unsigned>(shape[2]), static_cast<unsigned>(shape[1]),
                         static_cast<unsigned>(shape[0]));
  else
    field = zfp_field_1d(ptr, type, static_cast<unsigned>(n));
  if (!field) return std::unexpected(make_error(Errc::unsupported, "zfp field"));
  zfp_stream* z = zfp_stream_open(nullptr);
  zfp_stream_set_reversible(z);
  bitstream* stream = stream_open(const_cast<std::byte*>(packed.data()), packed.size());
  zfp_stream_set_bit_stream(z, stream);
  zfp_stream_rewind(z);
  auto ok = zfp_decompress(z, field);
  zfp_field_free(field);
  zfp_stream_close(z);
  stream_close(stream);
  if (!ok) return std::unexpected(make_error(Errc::ckpt_corrupt, "zfp decompress"));
  return raw;
}
#endif

}  // namespace

bool gyre_codec_available(GyreCodec c) {
  switch (c) {
    case GyreCodec::identity:
    case GyreCodec::alp:
      return true;
    case GyreCodec::zfp:
#if defined(GYRE_ZFP)
      return true;
#else
      return false;
#endif
  }
  return false;
}

Result<std::vector<std::byte>> gyre_compress(GyreCodec codec, std::span<const std::byte> raw, DType dt,
                                            std::span<const std::int64_t> shape) {
  if (codec == GyreCodec::identity) {
    return std::vector<std::byte>(raw.begin(), raw.end());
  }
  if (dt != DType::f32) {
    return std::vector<std::byte>(raw.begin(), raw.end());  // only f32 compressed
  }
  auto n = raw.size() / 4;
  auto* p = reinterpret_cast<const float*>(raw.data());
  if (codec == GyreCodec::alp) return alp_compress_f32(std::span<const float>(p, n));
  if (codec == GyreCodec::zfp) {
#if defined(GYRE_ZFP)
    return zfp_compress_f32(std::span<const float>(p, n), shape);
#else
    (void)shape;
    return std::unexpected(make_error(Errc::unsupported, "zfp not built"));
#endif
  }
  return std::unexpected(make_error(Errc::unsupported, "codec"));
}

Result<std::vector<std::byte>> gyre_decompress(GyreCodec codec, std::span<const std::byte> packed, DType dt,
                                              std::span<const std::int64_t> shape, std::size_t raw_bytes) {
  (void)dt;
  if (codec == GyreCodec::identity || packed.size() == raw_bytes) {
    if (packed.size() != raw_bytes && codec == GyreCodec::identity) {
      return std::unexpected(make_error(Errc::invalid_shape, "identity size"));
    }
    if (codec == GyreCodec::identity) return std::vector<std::byte>(packed.begin(), packed.end());
  }
  if (codec == GyreCodec::alp) return alp_decompress_f32(packed, raw_bytes / 4);
  if (codec == GyreCodec::zfp) {
#if defined(GYRE_ZFP)
    return zfp_decompress_f32(packed, shape, raw_bytes);
#else
    (void)shape;
    return std::unexpected(make_error(Errc::unsupported, "zfp not built"));
#endif
  }
  return std::unexpected(make_error(Errc::unsupported, "codec"));
}

Result<std::vector<GyreCodecProbe>> probe_gyre_codecs(const GyreFile& file, int max_tensors) {
  auto d = Device::cpu();
  if (!d) return std::unexpected(d.error());
  std::vector<GyreCodecProbe> rows;
  int n = 0;
  for (auto& t : file.doc().tensors) {
    if (t.role != TensorRole::weight) continue;
    if (max_tensors > 0 && n >= max_tensors) break;
    auto ten = file.load_tensor(t.name, *d);
    if (!ten) continue;
    auto hb = ten->host_bytes();
    if (!hb) continue;
    GyreCodecProbe r;
    r.name = t.name;
    r.raw_bytes = hb->size();
    auto alp = gyre_compress(GyreCodec::alp, *hb, ten->dtype(), ten->shape());
    if (alp) {
      r.alp_bytes = alp->size();
      auto back = gyre_decompress(GyreCodec::alp, *alp, ten->dtype(), ten->shape(), hb->size());
      r.alp_ok = back && *back == std::vector<std::byte>(hb->begin(), hb->end());
    }
    if (gyre_codec_available(GyreCodec::zfp)) {
      auto z = gyre_compress(GyreCodec::zfp, *hb, ten->dtype(), ten->shape());
      if (z) {
        r.zfp_bytes = z->size();
        auto back = gyre_decompress(GyreCodec::zfp, *z, ten->dtype(), ten->shape(), hb->size());
        r.zfp_ok = back && *back == std::vector<std::byte>(hb->begin(), hb->end());
      }
    }
    rows.push_back(std::move(r));
    ++n;
  }
  return rows;
}

}  // namespace gyre
