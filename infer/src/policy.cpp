#include "gyre_infer/policy.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>

namespace gyre_infer {

static_assert(std::endian::native == std::endian::little, "flat weights are little-endian");

int Info::total_logits() const {
  int t = 0;
  for (int s : head_sizes) t += s;
  return t;
}
int Info::head_offset(int head) const {
  int t = 0;
  for (int i = 0; i < head && i < static_cast<int>(head_sizes.size()); ++i) t += head_sizes[i];
  return t;
}

void Session::reset() { std::fill(h_.begin(), h_.end(), 0.f); }

namespace {

constexpr float kLnEps = 1e-5f;

struct Lin {  // y = x W + b, W [in,out]
  const float* W{nullptr};
  const float* b{nullptr};
  int in{0}, out{0};
};
struct Ln {
  const float* w{nullptr};
  const float* b{nullptr};
};
struct Block {
  Ln ln1, ln2;
  Lin q, k, v, o, fc1, fc2;
};
struct Gru {
  Lin wi, wh;  // [in,3H], [H,3H]
};

// Rows of x [n,k] times W [k,m] plus bias. The inner loop runs over contiguous output
// columns, which compilers vectorise at baseline SSE2 without any intrinsics.
void linear(const float* x, int n, const Lin& l, float* y) {
  const int k = l.in, m = l.out;
  for (int i = 0; i < n; ++i) {
    float* yi = y + static_cast<std::size_t>(i) * m;
    std::memcpy(yi, l.b, sizeof(float) * m);
    const float* xi = x + static_cast<std::size_t>(i) * k;
    for (int kk = 0; kk < k; ++kk) {
      const float a = xi[kk];
      const float* Wk = l.W + static_cast<std::size_t>(kk) * m;
      for (int j = 0; j < m; ++j) yi[j] += a * Wk[j];
    }
  }
}

void layer_norm(const float* x, int n, int c, const Ln& ln, float* y) {
  for (int i = 0; i < n; ++i) {
    const float* s = x + static_cast<std::size_t>(i) * c;
    float* d = y + static_cast<std::size_t>(i) * c;
    float mean = 0;
    for (int j = 0; j < c; ++j) mean += s[j];
    mean /= static_cast<float>(c);
    float var = 0;
    for (int j = 0; j < c; ++j) {
      const float t = s[j] - mean;
      var += t * t;
    }
    var /= static_cast<float>(c);
    const float inv = 1.f / std::sqrt(var + kLnEps);
    for (int j = 0; j < c; ++j) d[j] = (s[j] - mean) * inv * ln.w[j] + ln.b[j];
  }
}

inline float gelu1(float x) {
  x = std::min(std::max(x, -40.f), 40.f);
  const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + std::tanh(u));
}
inline float sigm(float v) { return 1.f / (1.f + std::exp(-v)); }

std::uint32_t fnv1a(const unsigned char* p, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
  return h;
}

}  // namespace

struct Policy::Impl {
  Info info;
  std::vector<float> weights;
  Lin embed, proj, heads, value;
  Ln ln_f;
  std::vector<Block> blocks;
  Gru gru;
};

Policy::Policy() : impl_(std::make_unique<Impl>()) {}
Policy::~Policy() = default;
const Info& Policy::info() const { return impl_->info; }

Session Policy::make_session() const {
  Session s;
  s.h_.assign(static_cast<std::size_t>(impl_->info.gru_hidden), 0.f);
  return s;
}

namespace {
// Limits for untrusted headers. Generous for any real policy net; they keep every product below
// 2^64 and every int computation in step() far from overflow.
constexpr std::uint32_t kMaxDim = 1u << 16;        // feat/global/entities/d_model/d_ff/out/gru
constexpr std::uint32_t kMaxLayers = 256;
constexpr std::uint32_t kMaxHeads = 1024;
constexpr std::uint32_t kMaxHeadSize = 1u << 20;
constexpr std::uint64_t kMaxTotalLogits = 1ull << 22;

struct Checked {  // overflow-detecting uint64 arithmetic
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

bool fail(std::string* e, const char* m) {
  if (e) *e = m;
  return false;
}
}  // namespace

std::unique_ptr<Policy> Policy::load(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return nullptr;
  }
  const auto n = static_cast<std::size_t>(f.tellg());
  if (n > (1ull << 32)) {
    if (error) *error = "file too large: " + path;
    return nullptr;
  }
  try {
    std::vector<char> buf(n);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(n));
    if (!f) {
      if (error) *error = "read failed: " + path;
      return nullptr;
    }
    return load_from_memory(buf.data(), n, error);
  } catch (const std::exception&) {
    if (error) *error = "out of memory reading " + path;
    return nullptr;
  }
}

std::unique_ptr<Policy> Policy::load_from_memory(const void* data, std::size_t bytes, std::string* error) {
  try {
    return load_impl(data, bytes, error);
  } catch (const std::exception&) {
    fail(error, "out of memory or malformed file");
    return nullptr;
  }
}

std::unique_ptr<Policy> Policy::load_impl(const void* data, std::size_t bytes, std::string* error) {
  const auto* p = static_cast<const unsigned char*>(data);
  std::size_t at = 0;
  auto rd32 = [&](std::uint32_t& v) {
    if (at + 4 > bytes) return false;
    std::memcpy(&v, p + at, 4);
    at += 4;
    return true;
  };
  if (bytes < 16 || std::memcmp(p, "GPOL1\0\0\0", 8) != 0) {
    fail(error, "not a GPOL1 file");
    return nullptr;
  }
  at = 8;
  std::uint32_t ver = 0, reserved = 0;
  if (!rd32(ver) || !rd32(reserved) || ver != 1) {
    fail(error, "unsupported GPOL version");
    return nullptr;
  }
  std::uint32_t v[10];
  for (auto& x : v)
    if (!rd32(x)) {
      fail(error, "truncated header");
      return nullptr;
    }
  const std::uint32_t nh = v[9];
  if (nh == 0 || nh > kMaxHeads || v[0] < 1 || v[0] > kMaxDim || v[1] > kMaxDim || v[2] < 1 || v[2] > kMaxDim ||
      v[3] < 1 || v[3] > kMaxDim || v[4] > kMaxLayers || v[5] < 1 || v[5] > kMaxDim || v[3] % v[5] != 0 ||
      v[6] < 1 || v[6] > kMaxDim || v[7] < 1 || v[7] > kMaxDim || v[8] > kMaxDim) {
    fail(error, "invalid config");
    return nullptr;
  }
  std::unique_ptr<Policy> pol(new Policy());
  Info& I = pol->impl_->info;
  I.feat_dim = static_cast<int>(v[0]);
  I.global_dim = static_cast<int>(v[1]);
  I.max_entities = static_cast<int>(v[2]);
  I.d_model = static_cast<int>(v[3]);
  I.n_layer = static_cast<int>(v[4]);
  I.n_head = static_cast<int>(v[5]);
  I.d_ff = static_cast<int>(v[6]);
  I.out_dim = static_cast<int>(v[7]);
  I.gru_hidden = static_cast<int>(v[8]);
  std::uint64_t total_logits = 0;
  for (std::uint32_t i = 0; i < nh; ++i) {
    std::uint32_t s = 0;
    if (!rd32(s) || s < 1 || s > kMaxHeadSize) {
      fail(error, "invalid head sizes");
      return nullptr;
    }
    total_logits += s;  // <= 1024 * 2^20 fits easily
    I.head_sizes.push_back(static_cast<int>(s));
  }
  if (total_logits > kMaxTotalLogits) {
    fail(error, "too many logits");
    return nullptr;
  }
  if (at + 8 > bytes) {
    fail(error, "truncated");
    return nullptr;
  }
  std::uint64_t nfloats = 0;
  std::memcpy(&nfloats, p + at, 8);
  at += 8;

  // expected parameter count (must match PolicyNet::parameters() order/shapes)
  Checked c;
  const std::uint64_t F = I.feat_dim, G = I.global_dim, d = I.d_model, ff = I.d_ff, O = I.out_dim,
                      H = I.gru_hidden, T = total_logits;
  const std::uint64_t head_in = H > 0 ? H : O;
  std::uint64_t expect = c.add(c.mul(F, d), d);
  const std::uint64_t per_layer =
      c.add(c.add(c.mul(4, d), c.mul(4, c.add(c.mul(d, d), d))),
            c.add(c.add(c.mul(d, ff), ff), c.add(c.mul(ff, d), d)));
  expect = c.add(expect, c.mul(I.n_layer, per_layer));
  expect = c.add(expect, c.add(c.mul(2, d), c.add(c.mul(c.add(c.mul(2, d), G), O), O)));
  if (H > 0)
    expect = c.add(expect, c.add(c.add(c.mul(O, c.mul(3, H)), c.mul(3, H)),
                                 c.add(c.mul(H, c.mul(3, H)), c.mul(3, H))));
  expect = c.add(expect, c.add(c.add(c.mul(head_in, T), T), c.add(head_in, 1)));
  if (!c.ok) {
    fail(error, "config implies an impossible weight count");
    return nullptr;
  }
  if (nfloats != expect) {
    fail(error, "weight count does not match config");
    return nullptr;
  }
  if (nfloats > (1ull << 31) || at + nfloats * 4 + 4 > bytes) {
    fail(error, "truncated payload");
    return nullptr;
  }
  std::uint32_t sum = 0;
  std::memcpy(&sum, p + at + nfloats * 4, 4);
  if (fnv1a(p + at, static_cast<std::size_t>(nfloats) * 4) != sum) {
    fail(error, "checksum mismatch");
    return nullptr;
  }
  auto& W = pol->impl_->weights;
  W.resize(static_cast<std::size_t>(nfloats));
  std::memcpy(W.data(), p + at, static_cast<std::size_t>(nfloats) * 4);

  const float* cur = W.data();
  auto take = [&](std::uint64_t cnt) {
    const float* r = cur;
    cur += cnt;
    return r;
  };
  auto mk_lin = [&](int in, int out) {
    Lin l;
    l.in = in;
    l.out = out;
    l.W = take(static_cast<std::uint64_t>(in) * out);
    l.b = take(out);
    return l;
  };
  auto mk_ln = [&](int c) {
    Ln l;
    l.w = take(c);
    l.b = take(c);
    return l;
  };
  auto& im = *pol->impl_;
  im.embed = mk_lin(I.feat_dim, I.d_model);
  for (int i = 0; i < I.n_layer; ++i) {
    Block b;
    b.ln1 = mk_ln(I.d_model);
    b.q = mk_lin(I.d_model, I.d_model);
    b.k = mk_lin(I.d_model, I.d_model);
    b.v = mk_lin(I.d_model, I.d_model);
    b.o = mk_lin(I.d_model, I.d_model);
    b.ln2 = mk_ln(I.d_model);
    b.fc1 = mk_lin(I.d_model, I.d_ff);
    b.fc2 = mk_lin(I.d_ff, I.d_model);
    im.blocks.push_back(b);
  }
  im.ln_f = mk_ln(I.d_model);
  im.proj = mk_lin(2 * I.d_model + I.global_dim, I.out_dim);
  if (I.gru_hidden > 0) {
    im.gru.wi = mk_lin(I.out_dim, 3 * I.gru_hidden);
    im.gru.wh = mk_lin(I.gru_hidden, 3 * I.gru_hidden);
  }
  const int head_in_i = I.gru_hidden > 0 ? I.gru_hidden : I.out_dim;
  im.heads = mk_lin(head_in_i, static_cast<int>(T));
  im.value = mk_lin(head_in_i, 1);
  return pol;
}

bool Policy::step(Session& s, const Input& in, float* logits, float* value) const {
  const Impl& m = *impl_;
  const Info& I = m.info;
  auto err = [&](const char* msg) {
    s.error_ = msg;
    return false;
  };
  s.error_.clear();
  if (in.n_slots < 0 || in.n_slots > I.max_entities) return err("n_slots out of range");
  if (in.n_slots > 0 && !in.entities) return err("entities is null");
  if (I.global_dim > 0 && !in.global) return err("global is null");
  if (I.recurrent() && s.h_.size() != static_cast<std::size_t>(I.gru_hidden))
    return err("session was created for a different policy");
  if (!logits || !value) return err("output pointers are null");

  // compact the valid rows
  s.rows_.clear();
  for (int i = 0; i < in.n_slots; ++i)
    if (!in.mask || in.mask[i]) s.rows_.push_back(i);
  const int n = static_cast<int>(s.rows_.size());
  const int d = I.d_model, ff = I.d_ff, O = I.out_dim;
  auto fit = [](std::vector<float>& v, std::size_t sz) {
    if (v.size() < sz) v.resize(sz);
  };
  fit(s.a_, static_cast<std::size_t>(std::max(n, 1)) * d);                       // h
  fit(s.b_, static_cast<std::size_t>(std::max(n, 1)) * d);                       // normed / scratch
  fit(s.c_, static_cast<std::size_t>(std::max(n, 1)) * std::max(ff, d) * 3);     // q,k,v / ffn
  fit(s.d_, static_cast<std::size_t>(std::max(n, 1)) * d);                       // attention out
  fit(s.e_, static_cast<std::size_t>(2 * d + I.global_dim + O + 6 * std::max(I.gru_hidden, 1) + 8));
  float* h = s.a_.data();
  float* t = s.b_.data();
  float* qkv = s.c_.data();
  float* att = s.d_.data();

  // embed
  for (int r = 0; r < n; ++r) {
    const float* row = in.entities + static_cast<std::size_t>(s.rows_[r]) * I.feat_dim;
    linear(row, 1, m.embed, h + static_cast<std::size_t>(r) * d);
  }
  const int dk = d / I.n_head;
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  for (const Block& b : m.blocks) {
    layer_norm(h, n, d, b.ln1, t);
    float* q = qkv;
    float* k = qkv + static_cast<std::size_t>(n) * d;
    float* v = qkv + static_cast<std::size_t>(n) * d * 2;
    linear(t, n, b.q, q);
    linear(t, n, b.k, k);
    linear(t, n, b.v, v);
    s.scores_.resize(static_cast<std::size_t>(std::max(n, 1)));
    float* sc = s.scores_.data();
    for (int hd = 0; hd < I.n_head; ++hd) {
      const int off = hd * dk;
      for (int i = 0; i < n; ++i) {
        const float* qi = q + static_cast<std::size_t>(i) * d + off;
        float mx = -INFINITY;
        for (int j = 0; j < n; ++j) {
          const float* kj = k + static_cast<std::size_t>(j) * d + off;
          float dot = 0;
          for (int e = 0; e < dk; ++e) dot += qi[e] * kj[e];
          sc[j] = dot * scale;
          mx = std::max(mx, sc[j]);
        }
        float sum = 0;
        for (int j = 0; j < n; ++j) {
          sc[j] = std::exp(sc[j] - mx);
          sum += sc[j];
        }
        const float inv = 1.f / sum;
        float* yi = att + static_cast<std::size_t>(i) * d + off;
        for (int e = 0; e < dk; ++e) yi[e] = 0.f;
        for (int j = 0; j < n; ++j) {
          const float w = sc[j] * inv;
          const float* vj = v + static_cast<std::size_t>(j) * d + off;
          for (int e = 0; e < dk; ++e) yi[e] += w * vj[e];
        }
      }
    }
    linear(att, n, b.o, t);  // t = o(att)
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * d; ++i) h[i] += t[i];
    layer_norm(h, n, d, b.ln2, t);
    float* f1 = qkv;  // [n, ff] (qkv region is large enough: n*max(ff,d)*3)
    linear(t, n, b.fc1, f1);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * ff; ++i) f1[i] = gelu1(f1[i]);
    linear(f1, n, b.fc2, t);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * d; ++i) h[i] += t[i];
  }
  layer_norm(h, n, d, m.ln_f, t);

  // masked mean | max pool, then global features
  float* pin = s.e_.data();  // [2d + G]
  if (n > 0) {
    for (int j = 0; j < d; ++j) {
      float sum = 0, mx = t[j];
      for (int r = 0; r < n; ++r) {
        const float x = t[static_cast<std::size_t>(r) * d + j];
        sum += x;
        mx = std::max(mx, x);
      }
      pin[j] = sum / static_cast<float>(n);
      pin[d + j] = mx;
    }
  } else {
    for (int j = 0; j < 2 * d; ++j) pin[j] = 0.f;
  }
  for (int j = 0; j < I.global_dim; ++j) pin[2 * d + j] = in.global[j];
  float* feat = pin + 2 * d + I.global_dim;  // [O]
  linear(pin, 1, m.proj, feat);
  for (int j = 0; j < O; ++j) feat[j] = gelu1(feat[j]);

  const float* head_in = feat;
  if (I.recurrent()) {
    const int H = I.gru_hidden;
    if (in.reset) std::fill(s.h_.begin(), s.h_.end(), 0.f);
    float* gi = feat + O;       // [3H]
    float* gh = gi + 3 * H;     // [3H]
    s.e_.resize(std::max<std::size_t>(s.e_.size(), static_cast<std::size_t>(2 * d + I.global_dim + O + 6 * H)));
    // (resize may move storage: recompute pointers)
    pin = s.e_.data();
    feat = pin + 2 * d + I.global_dim;
    gi = feat + O;
    gh = gi + 3 * H;
    linear(feat, 1, m.gru.wi, gi);
    linear(s.h_.data(), 1, m.gru.wh, gh);
    for (int j = 0; j < H; ++j) {
      const float r = sigm(gi[j] + gh[j]);
      const float z = sigm(gi[H + j] + gh[H + j]);
      const float nn = std::tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      s.h_[j] = (1.f - z) * nn + z * s.h_[j];
    }
    head_in = s.h_.data();
  }
  // heads
  const int T = I.total_logits();
  std::memcpy(logits, m.heads.b, sizeof(float) * T);
  float vsum = m.value.b[0];
  for (int kk = 0; kk < m.heads.in; ++kk) {
    const float a = head_in[kk];
    const float* Wk = m.heads.W + static_cast<std::size_t>(kk) * T;
    for (int j = 0; j < T; ++j) logits[j] += a * Wk[j];
    vsum += a * m.value.W[kk];
  }
  *value = vsum;
  return true;
}

int pick_masked(const float* logits, const std::uint8_t* legal, int n, float u) {
  float mx = -INFINITY;
  int best = -1;
  for (int i = 0; i < n; ++i)
    if (!legal || legal[i]) {
      if (best < 0 || logits[i] > mx) {
        mx = logits[i];
        best = i;
      }
    }
  if (best < 0 || u < 0.f) return best;
  double sum = 0;
  for (int i = 0; i < n; ++i)
    if (!legal || legal[i]) sum += std::exp(double(logits[i]) - mx);
  double acc = 0;
  const double target = double(u) * sum;
  int last = best;
  for (int i = 0; i < n; ++i) {
    if (legal && !legal[i]) continue;
    last = i;
    acc += std::exp(double(logits[i]) - mx);
    if (target < acc) return i;
  }
  return last;
}

}  // namespace gyre_infer
