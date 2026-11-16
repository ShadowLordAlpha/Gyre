#include "gyre/nn/llama.hpp"

#include "gyre/ops.hpp"
#include "json_parse.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

#if defined(GYRE_OPENMP)
#include <omp.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define GYRE_LLAMA_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define GYRE_AVX2_FN
#else
#include <cpuid.h>
#define GYRE_AVX2_FN __attribute__((target("avx2,fma")))
#endif
#endif

namespace gyre {
namespace {

Error bad(const std::string& m) { return make_error(Errc::invalid_shape, "llama: " + m); }

// ---------------------------------------------------------------------------------------------
// CPU kernels. The library is built for the baseline ISA; AVX2+FMA paths are chosen at run time.

#if defined(GYRE_LLAMA_X86)
bool detect_avx2_fma() {
#if defined(_MSC_VER) && !defined(__clang__)
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  const bool fma = (r[2] & (1 << 12)) != 0, osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
  if (!fma || !osxsave || !avx) return false;
  if ((_xgetbv(0) & 6) != 6) return false;
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 5)) != 0;
#else
  unsigned a, b, c, d;
  if (__get_cpuid_max(0, nullptr) < 7) return false;
  __cpuid(1, a, b, c, d);
  const bool fma = (c & (1u << 12)) != 0, osxsave = (c & (1u << 27)) != 0, avx = (c & (1u << 28)) != 0;
  if (!fma || !osxsave || !avx) return false;
  unsigned lo, hi;
  __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  if ((lo & 6) != 6) return false;
  __cpuid_count(7, 0, a, b, c, d);
  return (b & (1u << 5)) != 0;
#endif
}
const bool g_avx2 = detect_avx2_fma();

GYRE_AVX2_FN inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
  return _mm_cvtss_f32(lo);
}

// C[MR rows, 16 cols] (+)= A[MR, K] @ B[K, 16-col panel]
template <int MR>
GYRE_AVX2_FN void micro_avx2(const float* A, std::int64_t lda, const float* B, std::int64_t ldb, float* C,
                             std::int64_t ldc, std::int64_t K, bool acc) {
  __m256 c0[MR], c1[MR];
  for (int r = 0; r < MR; ++r) {
    c0[r] = acc ? _mm256_loadu_ps(C + r * ldc) : _mm256_setzero_ps();
    c1[r] = acc ? _mm256_loadu_ps(C + r * ldc + 8) : _mm256_setzero_ps();
  }
  for (std::int64_t k = 0; k < K; ++k) {
    const __m256 b0 = _mm256_loadu_ps(B + k * ldb);
    const __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);
    for (int r = 0; r < MR; ++r) {
      const __m256 a = _mm256_broadcast_ss(A + r * lda + k);
      c0[r] = _mm256_fmadd_ps(a, b0, c0[r]);
      c1[r] = _mm256_fmadd_ps(a, b1, c1[r]);
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm256_storeu_ps(C + r * ldc, c0[r]);
    _mm256_storeu_ps(C + r * ldc + 8, c1[r]);
  }
}

GYRE_AVX2_FN float dot_avx2(const float* a, const float* b, std::int64_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  std::int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  float s = hsum256(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

// out[0..n) = sum_s w[s] * V[s*ldv + 0..n)   (n % 8 == 0)
GYRE_AVX2_FN void wsum_avx2(const float* w, const float* V, std::int64_t ldv, std::int64_t S, float* out,
                            std::int64_t n) {
  std::int64_t c = 0;
  for (; c + 32 <= n; c += 32) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (std::int64_t s = 0; s < S; ++s) {
      const __m256 ws = _mm256_broadcast_ss(w + s);
      const float* v = V + s * ldv + c;
      a0 = _mm256_fmadd_ps(ws, _mm256_loadu_ps(v), a0);
      a1 = _mm256_fmadd_ps(ws, _mm256_loadu_ps(v + 8), a1);
      a2 = _mm256_fmadd_ps(ws, _mm256_loadu_ps(v + 16), a2);
      a3 = _mm256_fmadd_ps(ws, _mm256_loadu_ps(v + 24), a3);
    }
    _mm256_storeu_ps(out + c, a0);
    _mm256_storeu_ps(out + c + 8, a1);
    _mm256_storeu_ps(out + c + 16, a2);
    _mm256_storeu_ps(out + c + 24, a3);
  }
  for (; c + 8 <= n; c += 8) {
    __m256 a0 = _mm256_setzero_ps();
    for (std::int64_t s = 0; s < S; ++s) a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + s), _mm256_loadu_ps(V + s * ldv + c), a0);
    _mm256_storeu_ps(out + c, a0);
  }
}
#else
constexpr bool g_avx2 = false;
#endif

template <int MR>
void micro_scalar(const float* A, std::int64_t lda, const float* B, std::int64_t ldb, float* C, std::int64_t ldc,
                  std::int64_t K, std::int64_t nr, bool acc) {
  float c[MR][16];
  for (int r = 0; r < MR; ++r)
    for (std::int64_t j = 0; j < nr; ++j) c[r][j] = acc ? C[r * ldc + j] : 0.f;
  for (std::int64_t k = 0; k < K; ++k) {
    const float* b = B + k * ldb;
    for (int r = 0; r < MR; ++r) {
      const float a = A[r * lda + k];
      for (std::int64_t j = 0; j < nr; ++j) c[r][j] += a * b[j];
    }
  }
  for (int r = 0; r < MR; ++r)
    for (std::int64_t j = 0; j < nr; ++j) C[r * ldc + j] = c[r][j];
}

template <int MR>
void micro(const float* A, std::int64_t lda, const float* B, std::int64_t ldb, float* C, std::int64_t ldc,
           std::int64_t K, std::int64_t nr, bool acc) {
#if defined(GYRE_LLAMA_X86)
  if (g_avx2 && nr == 16) return micro_avx2<MR>(A, lda, B, ldb, C, ldc, K, acc);
#endif
  micro_scalar<MR>(A, lda, B, ldb, C, ldc, K, nr, acc);
}

int n_threads() {
#if defined(GYRE_OPENMP)
  apply_thread_settings();
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// C[M,N] = A[M,K] @ B[K,N] (row-major; `acc` adds into C). Every output element is one
// k-ordered FMA chain, independent of the tiling and the thread count, so a row gives the same
// result whether it is computed alone (decode) or inside a block (prefill).
void gemm(const float* A, const float* B, float* C, std::int64_t M, std::int64_t K, std::int64_t N, bool acc) {
  constexpr std::int64_t MB = 64;
  const int nthr = n_threads();
  const std::int64_t mt = (M + MB - 1) / MB;
  std::int64_t NB = 256;
  const std::int64_t want_ct = std::max<std::int64_t>(1, (2 * nthr + mt - 1) / mt);
  if (mt * ((N + NB - 1) / NB) < 2 * nthr) NB = std::max<std::int64_t>(16, ((N + want_ct - 1) / want_ct + 15) / 16 * 16);
  const std::int64_t nt = (N + NB - 1) / NB;
  const std::int64_t tiles = mt * nt;
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (tiles > 1 && !omp_in_parallel())
#endif
  for (std::int64_t tile = 0; tile < tiles; ++tile) {
    const std::int64_t i0 = (tile / nt) * MB, j0 = (tile % nt) * NB;
    const std::int64_t i1 = std::min(M, i0 + MB), j1 = std::min(N, j0 + NB);
    for (std::int64_t j = j0; j < j1; j += 16) {
      const std::int64_t nr = std::min<std::int64_t>(16, j1 - j);
      std::int64_t i = i0;
      for (; i + 4 <= i1; i += 4) micro<4>(A + i * K, K, B + j, N, C + i * N + j, N, K, nr, acc);
      for (; i < i1; ++i) micro<1>(A + i * K, K, B + j, N, C + i * N + j, N, K, nr, acc);
    }
  }
}

float dot(const float* a, const float* b, std::int64_t n) {
#if defined(GYRE_LLAMA_X86)
  if (g_avx2) return dot_avx2(a, b, n);
#endif
  float s = 0.f;
  for (std::int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

void wsum(const float* w, const float* V, std::int64_t ldv, std::int64_t S, float* out, std::int64_t n) {
#if defined(GYRE_LLAMA_X86)
  if (g_avx2 && n % 8 == 0) return wsum_avx2(w, V, ldv, S, out, n);
#endif
  for (std::int64_t c = 0; c < n; ++c) out[c] = 0.f;
  for (std::int64_t s = 0; s < S; ++s) {
    const float ws = w[s];
    const float* v = V + s * ldv;
    for (std::int64_t c = 0; c < n; ++c) out[c] += ws * v[c];
  }
}

void rms_rows(const float* x, const float* w, float* y, std::int64_t rows, std::int64_t d, float eps) {
#if defined(GYRE_OPENMP)
  apply_thread_settings();
#pragma omp parallel for schedule(static) if (rows > 8 && !omp_in_parallel())
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * d;
    float* yr = y + r * d;
    double ss = 0;
    for (std::int64_t i = 0; i < d; ++i) ss += static_cast<double>(xr[i]) * xr[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + static_cast<double>(eps)));
    for (std::int64_t i = 0; i < d; ++i) yr[i] = (xr[i] * inv) * w[i];
  }
}

Result<std::vector<std::int64_t>> shape_of(const LlamaConfig& c, const std::string& name) {
  const auto d = c.d_model, qd = c.n_head * c.head_dim, kvd = c.n_kv_head * c.head_dim;
  auto ends = [&](const char* s) {
    const std::string suf(s);
    return name.size() >= suf.size() && name.compare(name.size() - suf.size(), suf.size(), suf) == 0;
  };
  if (name == "embed.weight") return std::vector<std::int64_t>{c.vocab, d};
  if (name == "norm.weight") return std::vector<std::int64_t>{d};
  if (name == "lm_head.weight") return std::vector<std::int64_t>{d, c.vocab};
  if (ends("attn_norm.weight") || ends("mlp_norm.weight")) return std::vector<std::int64_t>{d};
  if (ends("attn.q.weight")) return std::vector<std::int64_t>{d, qd};
  if (ends("attn.k.weight") || ends("attn.v.weight")) return std::vector<std::int64_t>{d, kvd};
  if (ends("attn.o.weight")) return std::vector<std::int64_t>{qd, d};
  if (ends("mlp.gate.weight") || ends("mlp.up.weight")) return std::vector<std::int64_t>{d, c.d_ff};
  if (ends("mlp.down.weight")) return std::vector<std::int64_t>{c.d_ff, d};
  return std::unexpected(bad("unknown parameter " + name));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// LlamaConfig

Result<void> LlamaConfig::validate() const {
  if (vocab <= 0 || d_model <= 0 || n_layer <= 0 || n_head <= 0 || n_kv_head <= 0 || head_dim <= 0 || d_ff <= 0)
    return std::unexpected(bad("dimensions must be positive"));
  if (n_head % n_kv_head != 0) return std::unexpected(bad("n_head must be a multiple of n_kv_head"));
  if (head_dim % 2 != 0) return std::unexpected(bad("head_dim must be even (RoPE pairs)"));
  if (!(rms_eps > 0.f) || !(rope_theta > 0.f)) return std::unexpected(bad("rms_eps / rope_theta must be > 0"));
  if (max_seq <= 0) return std::unexpected(bad("max_seq must be > 0"));
  return {};
}

std::int64_t LlamaConfig::param_count() const {
  const auto d = d_model, qd = n_head * head_dim, kvd = n_kv_head * head_dim;
  const auto layer = 2 * d + d * qd + 2 * d * kvd + qd * d + 3 * d * d_ff;
  return vocab * d + n_layer * layer + d + (tie_embeddings ? 0 : d * vocab);
}

std::string LlamaConfig::to_json() const {
  nlohmann::json j;
  j["vocab_size"] = vocab;
  j["d_model"] = d_model;
  j["n_layer"] = n_layer;
  j["n_head"] = n_head;
  j["n_kv_head"] = n_kv_head;
  j["head_dim"] = head_dim;
  j["d_ff"] = d_ff;
  j["max_seq"] = max_seq;
  j["rms_eps"] = rms_eps;
  j["rope_theta"] = rope_theta;
  j["rope"] = "interleaved";  // q/k columns are stored for gyre::rope's pair layout
  j["tie_embeddings"] = tie_embeddings;
  j["bos_id"] = bos_id;
  j["eos_id"] = eos_id;
  return j.dump();
}

Result<LlamaConfig> LlamaConfig::from_json(std::string_view json) {
  auto p = parse_json(json);
  if (!p) return std::unexpected(p.error());
  try {
    const auto& j = *p;
    LlamaConfig c;
    c.vocab = j.at("vocab_size").get<std::int64_t>();
    c.d_model = j.at("d_model").get<std::int64_t>();
    c.n_layer = j.at("n_layer").get<std::int64_t>();
    c.n_head = j.at("n_head").get<std::int64_t>();
    c.n_kv_head = j.at("n_kv_head").get<std::int64_t>();
    c.head_dim = j.at("head_dim").get<std::int64_t>();
    c.d_ff = j.at("d_ff").get<std::int64_t>();
    c.max_seq = j.value("max_seq", c.max_seq);
    c.rms_eps = j.value("rms_eps", c.rms_eps);
    c.rope_theta = j.value("rope_theta", c.rope_theta);
    c.tie_embeddings = j.value("tie_embeddings", true);
    c.bos_id = j.value("bos_id", -1);
    c.eos_id = j.value("eos_id", -1);
    if (j.value("rope", std::string{"interleaved"}) != "interleaved")
      return std::unexpected(make_error(Errc::unsupported, "llama: only the interleaved RoPE layout is stored"));
    if (auto v = c.validate(); !v) return std::unexpected(v.error());
    return c;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("llama config: ") + e.what()));
  }
}

Result<LlamaConfig> LlamaConfig::from_hf_json(std::string_view json) {
  auto p = parse_json(json);
  if (!p) return std::unexpected(p.error());
  auto unsupported = [](const std::string& m) { return std::unexpected(make_error(Errc::unsupported, "llama import: " + m)); };
  try {
    const auto& j = *p;
    if (j.value("model_type", std::string{}) != "llama") return unsupported("model_type is not llama");
    if (j.value("hidden_act", std::string{"silu"}) != "silu") return unsupported("hidden_act must be silu");
    if (j.value("attention_bias", false)) return unsupported("attention_bias");
    if (j.value("mlp_bias", false)) return unsupported("mlp_bias");
    if (j.value("pretraining_tp", 1) != 1) return unsupported("pretraining_tp != 1");
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) return unsupported("rope_scaling");
    if (j.value("rope_interleaved", false)) return unsupported("rope_interleaved=true");
    LlamaConfig c;
    c.vocab = j.at("vocab_size").get<std::int64_t>();
    c.d_model = j.at("hidden_size").get<std::int64_t>();
    c.n_layer = j.at("num_hidden_layers").get<std::int64_t>();
    c.n_head = j.at("num_attention_heads").get<std::int64_t>();
    c.n_kv_head = j.value("num_key_value_heads", c.n_head);
    c.head_dim = j.contains("head_dim") && !j["head_dim"].is_null() ? j["head_dim"].get<std::int64_t>()
                                                                      : c.d_model / c.n_head;
    c.d_ff = j.at("intermediate_size").get<std::int64_t>();
    c.max_seq = j.value("max_position_embeddings", std::int64_t{2048});
    c.rms_eps = static_cast<float>(j.value("rms_norm_eps", 1e-6));
    c.rope_theta = static_cast<float>(j.value("rope_theta", 10000.0));
    c.tie_embeddings = j.value("tie_word_embeddings", false);
    if (j.contains("bos_token_id") && j["bos_token_id"].is_number_integer()) c.bos_id = j["bos_token_id"].get<std::int32_t>();
    if (j.contains("eos_token_id") && j["eos_token_id"].is_number_integer()) c.eos_id = j["eos_token_id"].get<std::int32_t>();
    if (auto v = c.validate(); !v) return std::unexpected(v.error());
    return c;
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("llama import: config.json: ") + e.what()));
  }
}

// ---------------------------------------------------------------------------------------------
// LlamaLM

std::vector<std::string> LlamaLM::param_names() const {
  std::vector<std::string> n;
  n.push_back("embed.weight");
  for (std::int64_t l = 0; l < cfg_.n_layer; ++l) {
    const auto p = "layers." + std::to_string(l) + ".";
    for (const char* s : {"attn_norm.weight", "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.o.weight",
                          "mlp_norm.weight", "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight"})
      n.push_back(p + s);
  }
  n.push_back("norm.weight");
  if (!cfg_.tie_embeddings) n.push_back("lm_head.weight");
  return n;
}

std::int64_t LlamaLM::n_params() const {
  std::int64_t n = 0;
  for (auto& t : w_) n += t.numel();
  return n;
}

Result<LlamaLM> LlamaLM::create(const LlamaConfig& c, Rng& rng) {
  if (auto v = c.validate(); !v) return std::unexpected(v.error());
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  LlamaLM m;
  m.cfg_ = c;
  std::vector<Tensor> ws;
  for (auto& name : m.param_names()) {
    auto sh = shape_of(c, name);
    if (!sh) return std::unexpected(sh.error());
    auto t = Tensor::empty(*sh, DType::f32, *dev);
    if (!t) return std::unexpected(t.error());
    auto p = t->host_span<float>();
    if (!p) return std::unexpected(p.error());
    const bool norm = sh->size() == 1;
    for (auto& x : *p) x = norm ? 1.f + rng.normal(0.f, 0.1f) : rng.normal(0.f, 0.02f);
    ws.push_back(std::move(*t));
  }
  return from_tensors(c, std::move(ws));
}

Result<LlamaLM> LlamaLM::from_tensors(const LlamaConfig& c, std::vector<Tensor> weights) {
  if (auto v = c.validate(); !v) return std::unexpected(v.error());
  LlamaLM m;
  m.cfg_ = c;
  const auto names = m.param_names();
  if (weights.size() != names.size())
    return std::unexpected(bad("expected " + std::to_string(names.size()) + " tensors, got " +
                               std::to_string(weights.size())));
  for (std::size_t i = 0; i < names.size(); ++i) {
    auto sh = shape_of(c, names[i]);
    if (!sh) return std::unexpected(sh.error());
    const auto& t = weights[i];
    if (t.dtype() != DType::f32) return std::unexpected(bad(names[i] + " must be f32"));
    if (t.device() && t.device()->kind() != DeviceKind::cpu) return std::unexpected(bad(names[i] + " must be on CPU"));
    if (!std::equal(t.shape().begin(), t.shape().end(), sh->begin(), sh->end()))
      return std::unexpected(bad("shape mismatch for " + names[i]));
  }
  m.w_ = std::move(weights);
  if (auto b = m.bind(); !b) return std::unexpected(b.error());
  return m;
}

Result<void> LlamaLM::bind() {
  std::vector<const float*> p;
  p.reserve(w_.size());
  for (auto& t : w_) {
    auto s = std::as_const(t).host_span<float>();
    if (!s) return std::unexpected(s.error());
    p.push_back(s->data());
  }
  std::size_t i = 0;
  embed_ = p[i++];
  layers_.clear();
  for (std::int64_t l = 0; l < cfg_.n_layer; ++l) {
    Layer L{};
    L.attn_norm = p[i++];
    L.wq = p[i++];
    L.wk = p[i++];
    L.wv = p[i++];
    L.wo = p[i++];
    L.mlp_norm = p[i++];
    L.wg = p[i++];
    L.wu = p[i++];
    L.wd = p[i++];
    layers_.push_back(L);
  }
  final_norm_ = p[i++];
  const auto V = cfg_.vocab, d = cfg_.d_model;
  if (cfg_.tie_embeddings) {
    head_t_.assign(static_cast<std::size_t>(V * d), 0.f);
    float* dst = head_t_.data();
    const float* src = embed_;
#if defined(GYRE_OPENMP)
    apply_thread_settings();
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t k = 0; k < d; ++k)
      for (std::int64_t v = 0; v < V; ++v) dst[k * V + v] = src[v * d + k];
    head_ = head_t_.data();
  } else {
    head_ = p[i++];
  }
  return {};
}

Result<void> LlamaLM::ensure_rope(std::int64_t n) {
  if (n <= rope_len_) return {};
  const auto half = cfg_.head_dim / 2;
  const auto len = std::max<std::int64_t>(n, std::min<std::int64_t>(cfg_.max_seq, 2 * rope_len_));
  rope_cos_.resize(static_cast<std::size_t>(len * half));
  rope_sin_.resize(static_cast<std::size_t>(len * half));
  for (std::int64_t i = 0; i < half; ++i) {
    const double inv = std::pow(static_cast<double>(cfg_.rope_theta),
                                -static_cast<double>(2 * i) / static_cast<double>(cfg_.head_dim));
    for (std::int64_t pos = rope_len_; pos < len; ++pos) {
      const double a = static_cast<double>(pos) * inv;
      rope_cos_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::cos(a));
      rope_sin_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::sin(a));
    }
  }
  rope_len_ = len;
  return {};
}

Result<LlamaCache> LlamaLM::make_cache(std::int64_t capacity) const {
  if (capacity <= 0) capacity = cfg_.max_seq;
  LlamaCache c;
  c.capacity = capacity;
  const auto n = static_cast<std::size_t>(cfg_.n_layer * cfg_.n_kv_head * capacity * cfg_.head_dim);
  c.k.assign(n, 0.f);
  c.v.assign(n, 0.f);
  return c;
}

Result<std::vector<float>> LlamaLM::forward_hidden(LlamaCache& cache, std::span<const std::int32_t> ids) {
  const auto T = static_cast<std::int64_t>(ids.size());
  const auto d = cfg_.d_model, hd = cfg_.head_dim, Hq = cfg_.n_head, Hkv = cfg_.n_kv_head;
  const auto qd = Hq * hd, kvd = Hkv * hd, ff = cfg_.d_ff, V = cfg_.vocab;
  const auto pos0 = cache.len, cap = cache.capacity;
  if (T == 0) return std::vector<float>{};
  if (pos0 + T > cap)
    return std::unexpected(make_error(Errc::overflow, "llama: KV cache full (" + std::to_string(pos0 + T) + " > " +
                                                          std::to_string(cap) + ")"));
  if (cache.k.size() != static_cast<std::size_t>(cfg_.n_layer * Hkv * cap * hd))
    return std::unexpected(bad("cache was made for another model"));
  if (auto r = ensure_rope(pos0 + T); !r) return std::unexpected(r.error());
  auto grow = [](std::vector<float>& v, std::int64_t n) {
    if (v.size() < static_cast<std::size_t>(n)) v.resize(static_cast<std::size_t>(n));
  };
  grow(x_, T * d);
  grow(xn_, T * d);
  grow(q_, T * qd);
  grow(k_, T * kvd);
  grow(v_, T * kvd);
  grow(att_, T * qd);
  grow(g_, T * ff);
  grow(u_, T * ff);
  for (std::int64_t t = 0; t < T; ++t) {
    const auto id = ids[static_cast<std::size_t>(t)];
    if (id < 0 || id >= V) return std::unexpected(bad("token id " + std::to_string(id) + " out of range"));
    std::memcpy(x_.data() + t * d, embed_ + static_cast<std::int64_t>(id) * d, static_cast<std::size_t>(d) * sizeof(float));
  }
  const auto half = hd / 2;
  const auto rep = Hq / Hkv;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  const float* cosT = rope_cos_.data();
  const float* sinT = rope_sin_.data();
  auto rope_rows = [&](float* m, std::int64_t heads) {
    const auto width = heads * hd;
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (T * heads > 64 && !omp_in_parallel())
#endif
    for (std::int64_t th = 0; th < T * heads; ++th) {
      const auto t = th / heads, h = th % heads;
      float* r = m + t * width + h * hd;
      const float* cs = cosT + (pos0 + t) * half;
      const float* sn = sinT + (pos0 + t) * half;
      for (std::int64_t i = 0; i < half; ++i) {
        const float x0 = r[2 * i], x1 = r[2 * i + 1];
        r[2 * i] = x0 * cs[i] - x1 * sn[i];
        r[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
      }
    }
  };

  for (std::int64_t l = 0; l < cfg_.n_layer; ++l) {
    const Layer& L = layers_[static_cast<std::size_t>(l)];
    rms_rows(x_.data(), L.attn_norm, xn_.data(), T, d, cfg_.rms_eps);
    gemm(xn_.data(), L.wq, q_.data(), T, d, qd, false);
    gemm(xn_.data(), L.wk, k_.data(), T, d, kvd, false);
    gemm(xn_.data(), L.wv, v_.data(), T, d, kvd, false);
    apply_thread_settings();
    rope_rows(q_.data(), Hq);
    rope_rows(k_.data(), Hkv);
    float* Kl = cache.k.data() + l * Hkv * cap * hd;
    float* Vl = cache.v.data() + l * Hkv * cap * hd;
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t g = 0; g < Hkv; ++g) {
        std::memcpy(Kl + (g * cap + pos0 + t) * hd, k_.data() + t * kvd + g * hd, static_cast<std::size_t>(hd) * sizeof(float));
        std::memcpy(Vl + (g * cap + pos0 + t) * hd, v_.data() + t * kvd + g * hd, static_cast<std::size_t>(hd) * sizeof(float));
      }
    // causal attention: query t (position pos0 + t) sees keys 0 .. pos0 + t
    const std::int64_t items = T * Hq;
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4) if (items > 1)
#endif
    for (std::int64_t it = 0; it < items; ++it) {
      // longest rows first keeps the dynamic schedule balanced
      const auto t = T - 1 - it / Hq, h = it % Hq;
      const auto g = h / rep;
      const auto S = pos0 + t + 1;
      thread_local std::vector<float> sc;
      if (sc.size() < static_cast<std::size_t>(S)) sc.resize(static_cast<std::size_t>(S));
      const float* qv = q_.data() + t * qd + h * hd;
      const float* Kg = Kl + g * cap * hd;
      const float* Vg = Vl + g * cap * hd;
      float mx = -INFINITY;
      for (std::int64_t s = 0; s < S; ++s) {
        const float v = dot(qv, Kg + s * hd, hd) * scale;
        sc[static_cast<std::size_t>(s)] = v;
        mx = std::max(mx, v);
      }
      double sum = 0;
      for (std::int64_t s = 0; s < S; ++s) {
        const float e = std::exp(sc[static_cast<std::size_t>(s)] - mx);
        sc[static_cast<std::size_t>(s)] = e;
        sum += e;
      }
      const float inv = static_cast<float>(1.0 / sum);
      for (std::int64_t s = 0; s < S; ++s) sc[static_cast<std::size_t>(s)] *= inv;
      wsum(sc.data(), Vg, hd, S, att_.data() + t * qd + h * hd, hd);
    }
    gemm(att_.data(), L.wo, x_.data(), T, qd, d, true);  // x += attn @ Wo

    rms_rows(x_.data(), L.mlp_norm, xn_.data(), T, d, cfg_.rms_eps);
    gemm(xn_.data(), L.wg, g_.data(), T, d, ff, false);
    gemm(xn_.data(), L.wu, u_.data(), T, d, ff, false);
    {
      float* gp = g_.data();
      const float* up = u_.data();
      const std::int64_t n = T * ff;
      apply_thread_settings();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 16384)
#endif
      for (std::int64_t i = 0; i < n; ++i) {
        const float a = gp[i];
        gp[i] = a / (1.f + std::exp(-a)) * up[i];
      }
    }
    gemm(g_.data(), L.wd, x_.data(), T, ff, d, true);  // x += mlp
  }
  cache.len += T;
  std::vector<float> out(static_cast<std::size_t>(T * d));
  rms_rows(x_.data(), final_norm_, out.data(), T, d, cfg_.rms_eps);
  return out;
}

Result<std::vector<float>> LlamaLM::logits_from_hidden(std::span<const float> hidden, std::int64_t row0,
                                                       std::int64_t n) const {
  const auto d = cfg_.d_model, V = cfg_.vocab;
  if (row0 < 0 || n < 0 || static_cast<std::size_t>((row0 + n) * d) > hidden.size())
    return std::unexpected(bad("logits rows out of range"));
  std::vector<float> out(static_cast<std::size_t>(n * V));
  if (n > 0) gemm(hidden.data() + row0 * d, head_, out.data(), n, d, V, false);
  return out;
}

Result<std::vector<float>> LlamaLM::forward(LlamaCache& cache, std::span<const std::int32_t> ids,
                                            std::int64_t n_logits) {
  const auto T = static_cast<std::int64_t>(ids.size());
  if (n_logits < 0 || n_logits > T) return std::unexpected(bad("n_logits out of range"));
  auto h = forward_hidden(cache, ids);
  if (!h) return std::unexpected(h.error());
  return logits_from_hidden(*h, T - n_logits, n_logits);
}

Result<std::vector<float>> LlamaLM::forward_full(std::span<const std::int32_t> ids) {
  auto c = make_cache(std::max<std::int64_t>(1, static_cast<std::int64_t>(ids.size())));
  if (!c) return std::unexpected(c.error());
  return forward(*c, ids, static_cast<std::int64_t>(ids.size()));
}

Result<std::vector<std::int32_t>> LlamaLM::generate(std::vector<std::int32_t> prompt, int max_new, float temperature,
                                                    Rng* rng, bool use_cache, std::int32_t eos, GenStats* stats) {
  using clock = std::chrono::steady_clock;
  if (prompt.empty()) return std::unexpected(bad("generate needs a non-empty prompt"));
  if (temperature > 0.f && !rng) return std::unexpected(bad("sampling needs an rng"));
  const auto V = cfg_.vocab;
  GenStats st;
  std::vector<std::int32_t> seq = std::move(prompt);
  auto pick = [&](std::span<const float> row) { return sample_logit_row(row, temperature, rng); };
  if (max_new <= 0) return seq;
  if (use_cache) {
    auto cache = make_cache(static_cast<std::int64_t>(seq.size()) + max_new);
    if (!cache) return std::unexpected(cache.error());
    auto t0 = clock::now();
    auto lg = forward(*cache, seq, 1);
    if (!lg) return std::unexpected(lg.error());
    st.prefill_s = std::chrono::duration<double>(clock::now() - t0).count();
    t0 = clock::now();
    for (int i = 0; i < max_new; ++i) {
      auto nx = pick(std::span<const float>(lg->data(), static_cast<std::size_t>(V)));
      if (!nx) return std::unexpected(nx.error());
      seq.push_back(*nx);
      ++st.new_tokens;
      if (*nx == eos || i + 1 == max_new) break;
      const std::int32_t one[1] = {*nx};
      lg = forward(*cache, one, 1);
      if (!lg) return std::unexpected(lg.error());
    }
    st.decode_s = std::chrono::duration<double>(clock::now() - t0).count();
  } else {
    auto t0 = clock::now();
    for (int i = 0; i < max_new; ++i) {
      auto c = make_cache(static_cast<std::int64_t>(seq.size()));
      if (!c) return std::unexpected(c.error());
      auto lg = forward(*c, seq, 1);
      if (!lg) return std::unexpected(lg.error());
      if (i == 0) st.prefill_s = std::chrono::duration<double>(clock::now() - t0).count();
      auto nx = pick(*lg);
      if (!nx) return std::unexpected(nx.error());
      seq.push_back(*nx);
      ++st.new_tokens;
      if (*nx == eos) break;
    }
    st.decode_s = std::chrono::duration<double>(clock::now() - t0).count() - st.prefill_s;
  }
  if (stats) *stats = st;
  return seq;
}

Result<LlamaLM> LlamaLM::load_gyre(const std::filesystem::path& path) {
  auto f = GyreFile::open(path);
  if (!f) return std::unexpected(f.error());
  if (f->doc().arch != "llama")
    return std::unexpected(make_error(Errc::unsupported, "not a llama checkpoint (arch " + f->doc().arch + ")"));
  auto cfg = LlamaConfig::from_json(f->doc().config_json);
  if (!cfg) return std::unexpected(cfg.error());
  auto dev = Device::cpu();
  if (!dev) return std::unexpected(dev.error());
  LlamaLM probe;
  probe.cfg_ = *cfg;
  std::vector<Tensor> ws;
  for (auto& name : probe.param_names()) {
    auto t = f->load_tensor(name, *dev);
    if (!t) return std::unexpected(t.error());
    ws.push_back(std::move(*t));
  }
  return from_tensors(*cfg, std::move(ws));
}

Result<void> LlamaLM::save_gyre(const std::filesystem::path& path, std::string_view tokenizer_json,
                                std::string_view extra_config_json) const {
  auto cfgj = parse_json(cfg_.to_json());
  if (!cfgj) return std::unexpected(cfgj.error());
  if (!extra_config_json.empty()) {
    auto extra = parse_json(extra_config_json);
    if (!extra || !extra->is_object()) return std::unexpected(bad("extra config must be a JSON object"));
    for (auto& [k, v] : extra->items()) (*cfgj)[k] = v;
  }
  GyreDoc doc;
  doc.arch = "llama";
  doc.config_json = cfgj->dump();
  doc.tokenizer_json = std::string(tokenizer_json);
  // save_gyre only reads Param::value; grad aliases the same storage (no copy, never written).
  std::vector<Param> params;
  params.reserve(w_.size());
  for (auto& t : w_) params.push_back(Param{t, t});
  const auto names = param_names();
  if (auto s = gyre::save_gyre(path, params, nullptr, doc, names); !s) return s;
  auto json_path = path;
  json_path.replace_extension(".gyre.json");
  return save_gyre_json(json_path, params, nullptr, doc, names, static_cast<std::uint64_t>(n_params()) * 4 <= kGyreJsonDataLimit);
}

}  // namespace gyre
