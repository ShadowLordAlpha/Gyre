#include "gyre/rl/ops.hpp"

#include "gyre/ops.hpp"
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define GYRE_RL_SSE2 1
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace gyre {
namespace {

Error shape_err(const char* m) { return make_error(Errc::invalid_shape, m); }

inline float dot_n(const float* a, const float* b, int n) {
#if defined(GYRE_RL_SSE2)
  if ((n & 3) == 0) {
    __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
      acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    for (; i < n; i += 4) acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    acc0 = _mm_add_ps(acc0, acc1);
    alignas(16) float t[4];
    _mm_store_ps(t, acc0);
    return (t[0] + t[1]) + (t[2] + t[3]);
  }
#endif
  float s = 0.f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

bool same_shape(const Tensor& a, const Tensor& b) {
  if (a.rank() != b.rank()) return false;
  for (std::uint8_t i = 0; i < a.rank(); ++i)
    if (a.shape()[i] != b.shape()[i]) return false;
  return true;
}

template <class F>
Result<Tensor> unary(const Tensor& x, F f) {
  auto px = x.host_span<float>();
  if (!px) return std::unexpected(px.error());
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  if (!po) return std::unexpected(po.error());
  for (std::size_t i = 0; i < px->size(); ++i) (*po)[i] = f((*px)[i]);
  return out;
}

template <class F>
Result<Tensor> binary(const Tensor& a, const Tensor& b, F f) {
  if (!same_shape(a, b)) return std::unexpected(shape_err("binary op shape mismatch"));
  auto pa = a.host_span<float>();
  auto pb = b.host_span<float>();
  if (!pa || !pb) return std::unexpected(pa ? pb.error() : pa.error());
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  if (!po) return std::unexpected(po.error());
  for (std::size_t i = 0; i < pa->size(); ++i) (*po)[i] = f((*pa)[i], (*pb)[i]);
  return out;
}

}  // namespace

Result<Tensor> sigmoid(const Tensor& x) {
  return unary(x, [](float v) { return 1.f / (1.f + std::exp(-v)); });
}
Result<Tensor> sigmoid_backward(const Tensor& y, const Tensor& dy) {
  return binary(y, dy, [](float s, float g) { return g * s * (1.f - s); });
}
Result<Tensor> tanh_act(const Tensor& x) {
  return unary(x, [](float v) { return std::tanh(v); });
}
Result<Tensor> tanh_backward(const Tensor& y, const Tensor& dy) {
  return binary(y, dy, [](float t, float g) { return g * (1.f - t * t); });
}
Result<Tensor> relu(const Tensor& x) {
  return unary(x, [](float v) { return v > 0.f ? v : 0.f; });
}
Result<Tensor> relu_backward(const Tensor& x, const Tensor& dy) {
  return binary(x, dy, [](float v, float g) { return v > 0.f ? g : 0.f; });
}
Result<Tensor> sub(const Tensor& a, const Tensor& b) {
  return binary(a, b, [](float p, float q) { return p - q; });
}
Result<Tensor> clamp(const Tensor& x, float lo, float hi) {
  return unary(x, [=](float v) { return std::min(std::max(v, lo), hi); });
}
Result<Tensor> clamp_backward(const Tensor& x, const Tensor& dy, float lo, float hi) {
  return binary(x, dy, [=](float v, float g) { return (v >= lo && v <= hi) ? g : 0.f; });
}

Result<Tensor> log_softmax_last(const Tensor& x) {
  if (x.rank() < 1) return std::unexpected(shape_err("log_softmax rank"));
  auto px = x.host_span<float>();
  if (!px) return std::unexpected(px.error());
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  if (!po) return std::unexpected(po.error());
  const std::int64_t V = x.shape()[x.rank() - 1];
  const std::int64_t R = V > 0 ? x.numel() / V : 0;
  for (std::int64_t r = 0; r < R; ++r) {
    const float* in = px->data() + r * V;
    float* o = po->data() + r * V;
    float m = in[0];
    for (std::int64_t v = 1; v < V; ++v) m = std::max(m, in[v]);
    double s = 0;
    for (std::int64_t v = 0; v < V; ++v) s += std::exp(double(in[v]) - m);
    const float ls = static_cast<float>(std::log(s));
    for (std::int64_t v = 0; v < V; ++v) o[v] = (in[v] - m) - ls;  // avoids m + ls rounding
  }
  return out;
}

Result<Tensor> log_softmax_last_backward(const Tensor& logp, const Tensor& dy) {
  if (!same_shape(logp, dy) || logp.rank() < 1) return std::unexpected(shape_err("lsm bwd shape"));
  auto pl = logp.host_span<float>();
  auto pd = dy.host_span<float>();
  if (!pl || !pd) return std::unexpected(pl ? pd.error() : pl.error());
  auto out = Tensor::empty(logp.shape(), DType::f32, logp.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  const std::int64_t V = logp.shape()[logp.rank() - 1];
  const std::int64_t R = V > 0 ? logp.numel() / V : 0;
  for (std::int64_t r = 0; r < R; ++r) {
    double sd = 0;
    for (std::int64_t v = 0; v < V; ++v) sd += (*pd)[r * V + v];
    for (std::int64_t v = 0; v < V; ++v)
      (*po)[r * V + v] = (*pd)[r * V + v] - std::exp((*pl)[r * V + v]) * static_cast<float>(sd);
  }
  return out;
}

Result<Tensor> gather_last(const Tensor& x, const Tensor& idx) {
  if (idx.dtype() != DType::i32)
    return std::unexpected(make_error(Errc::dtype_mismatch, "gather idx i32"));
  if (x.rank() < 1 || x.rank() - 1 != idx.rank()) return std::unexpected(shape_err("gather ranks"));
  for (std::uint8_t i = 0; i < idx.rank(); ++i)
    if (x.shape()[i] != idx.shape()[i]) return std::unexpected(shape_err("gather lead dims"));
  auto px = x.host_span<float>();
  auto pi = idx.host_span<std::int32_t>();
  if (!px || !pi) return std::unexpected(px ? pi.error() : px.error());
  auto out = Tensor::empty(idx.shape(), DType::f32, x.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  const std::int64_t V = x.shape()[x.rank() - 1];
  for (std::size_t r = 0; r < pi->size(); ++r) {
    const auto k = (*pi)[r];
    if (k < 0 || k >= V) return std::unexpected(shape_err("gather index OOB"));
    (*po)[r] = (*px)[r * static_cast<std::size_t>(V) + static_cast<std::size_t>(k)];
  }
  return out;
}

Result<Tensor> gather_last_backward(const Tensor& dy, const Tensor& idx, std::int64_t V) {
  if (!same_shape(dy, idx)) return std::unexpected(shape_err("gather bwd shape"));
  std::vector<std::int64_t> sh(dy.shape().begin(), dy.shape().end());
  sh.push_back(V);
  auto out = Tensor::zeros(sh, DType::f32, dy.device());
  if (!out) return out;
  auto pd = dy.host_span<float>();
  auto pi = idx.host_span<std::int32_t>();
  auto po = out->host_span<float>();
  if (!pd || !pi || !po) return std::unexpected(shape_err("gather bwd host"));
  for (std::size_t r = 0; r < pi->size(); ++r) {
    const auto k = (*pi)[r];
    if (k < 0 || k >= V) return std::unexpected(shape_err("gather index OOB"));
    (*po)[r * static_cast<std::size_t>(V) + static_cast<std::size_t>(k)] += (*pd)[r];
  }
  return out;
}

Result<Tensor> mask_logits(const Tensor& logits, const Tensor& mask) {
  if (mask.dtype() != DType::u8 || !same_shape(logits, mask))
    return std::unexpected(shape_err("mask_logits: u8 mask, same shape"));
  auto pl = logits.host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  if (!pl || !pm) return std::unexpected(pl ? pm.error() : pl.error());
  auto out = Tensor::empty(logits.shape(), DType::f32, logits.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  for (std::size_t i = 0; i < pl->size(); ++i) (*po)[i] = (*pm)[i] ? (*pl)[i] : kMaskFill;
  return out;
}

Result<Tensor> mask_logits_backward(const Tensor& dy, const Tensor& mask) {
  if (mask.dtype() != DType::u8 || !same_shape(dy, mask))
    return std::unexpected(shape_err("mask_logits bwd shape"));
  auto pd = dy.host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  if (!pd || !pm) return std::unexpected(pd ? pm.error() : pd.error());
  auto out = Tensor::empty(dy.shape(), DType::f32, dy.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  for (std::size_t i = 0; i < pd->size(); ++i) (*po)[i] = (*pm)[i] ? (*pd)[i] : 0.f;
  return out;
}

Result<void> key_mask_(Tensor& scores, const Tensor& mask) {
  if (scores.rank() != 4 || mask.rank() != 2 || mask.dtype() != DType::u8)
    return std::unexpected(shape_err("key_mask_: scores [B,H,T,T], mask u8 [B,T]"));
  const auto B = scores.shape()[0], H = scores.shape()[1], T = scores.shape()[2];
  if (scores.shape()[3] != T || mask.shape()[0] != B || mask.shape()[1] != T)
    return std::unexpected(shape_err("key_mask_ dims"));
  auto ps = scores.host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  if (!ps || !pm) return std::unexpected(ps ? pm.error() : ps.error());
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t i = 0; i < T; ++i) {
        float* row = ps->data() + ((b * H + h) * T + i) * T;
        for (std::int64_t j = 0; j < T; ++j)
          if (!(*pm)[static_cast<std::size_t>(b * T + j)]) row[j] = kMaskFill;
      }
  return {};
}

Result<Tensor> masked_mean_max_pool(const Tensor& h, const Tensor& mask, Tensor& argmax_out) {
  if (h.rank() != 3 || mask.rank() != 2 || mask.dtype() != DType::u8 ||
      mask.shape()[0] != h.shape()[0] || mask.shape()[1] != h.shape()[1])
    return std::unexpected(shape_err("pool: h [B,N,D], mask u8 [B,N]"));
  const auto B = h.shape()[0], N = h.shape()[1], D = h.shape()[2];
  auto ph = h.host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  if (!ph || !pm) return std::unexpected(ph ? pm.error() : ph.error());
  std::int64_t osh[2] = {B, 2 * D};
  auto out = Tensor::zeros(osh, DType::f32, h.device());
  std::int64_t ash[2] = {B, D};
  auto am = Tensor::zeros(ash, DType::i32, h.device());
  if (!out || !am) return std::unexpected(shape_err("pool alloc"));
  auto po = out->host_span<float>();
  auto pa = am->host_span<std::int32_t>();
  for (std::int64_t b = 0; b < B; ++b) {
    std::int64_t cnt = 0;
    for (std::int64_t d = 0; d < D; ++d) (*pa)[b * D + d] = -1;
    for (std::int64_t n = 0; n < N; ++n) {
      if (!(*pm)[b * N + n]) continue;
      ++cnt;
      const float* row = ph->data() + (b * N + n) * D;
      for (std::int64_t d = 0; d < D; ++d) {
        (*po)[b * 2 * D + d] += row[d];
        const auto cur = (*pa)[b * D + d];
        if (cur < 0 || row[d] > ph->data()[(b * N + cur) * D + d])
          (*pa)[b * D + d] = static_cast<std::int32_t>(n);
      }
    }
    const float inv = cnt > 0 ? 1.f / static_cast<float>(cnt) : 0.f;
    for (std::int64_t d = 0; d < D; ++d) {
      (*po)[b * 2 * D + d] *= inv;
      const auto k = (*pa)[b * D + d];
      (*po)[b * 2 * D + D + d] = k >= 0 ? ph->data()[(b * N + k) * D + d] : 0.f;
    }
  }
  argmax_out = std::move(*am);
  return out;
}

Result<Tensor> masked_mean_max_pool_backward(const Tensor& dy, const Tensor& mask,
                                             const Tensor& argmax, std::int64_t N) {
  if (dy.rank() != 2 || argmax.rank() != 2 || dy.shape()[1] != 2 * argmax.shape()[1])
    return std::unexpected(shape_err("pool bwd shapes"));
  const auto B = dy.shape()[0], D = argmax.shape()[1];
  auto pd = dy.host_span<float>();
  auto pm = mask.host_span<std::uint8_t>();
  auto pa = argmax.host_span<std::int32_t>();
  if (!pd || !pm || !pa) return std::unexpected(shape_err("pool bwd host"));
  std::int64_t sh[3] = {B, N, D};
  auto out = Tensor::zeros(sh, DType::f32, dy.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  for (std::int64_t b = 0; b < B; ++b) {
    std::int64_t cnt = 0;
    for (std::int64_t n = 0; n < N; ++n) cnt += (*pm)[b * N + n] ? 1 : 0;
    const float inv = cnt > 0 ? 1.f / static_cast<float>(cnt) : 0.f;
    for (std::int64_t n = 0; n < N; ++n) {
      if (!(*pm)[b * N + n]) continue;
      for (std::int64_t d = 0; d < D; ++d)
        (*po)[(b * N + n) * D + d] += (*pd)[b * 2 * D + d] * inv;
    }
    for (std::int64_t d = 0; d < D; ++d) {
      const auto k = (*pa)[b * D + d];
      if (k >= 0) (*po)[(b * N + k) * D + d] += (*pd)[b * 2 * D + D + d];
    }
  }
  return out;
}

Result<Tensor> concat_last(const Tensor& a, const Tensor& b) {
  if (a.rank() != 2 || b.rank() != 2 || a.shape()[0] != b.shape()[0])
    return std::unexpected(shape_err("concat_last [B,Ca],[B,Cb]"));
  const auto B = a.shape()[0], Ca = a.shape()[1], Cb = b.shape()[1];
  std::int64_t sh[2] = {B, Ca + Cb};
  auto out = Tensor::empty(sh, DType::f32, a.device());
  if (!out) return out;
  auto pa = a.host_span<float>();
  auto pb = b.host_span<float>();
  auto po = out->host_span<float>();
  if (!pa || !pb || !po) return std::unexpected(shape_err("concat host"));
  for (std::int64_t r = 0; r < B; ++r) {
    std::copy_n(pa->data() + r * Ca, Ca, po->data() + r * (Ca + Cb));
    std::copy_n(pb->data() + r * Cb, Cb, po->data() + r * (Ca + Cb) + Ca);
  }
  return out;
}

Result<void> split_last(const Tensor& dy, std::int64_t ca, Tensor& da, Tensor& db) {
  if (dy.rank() != 2 || ca < 0 || ca > dy.shape()[1])
    return std::unexpected(shape_err("split_last"));
  const auto B = dy.shape()[0], C = dy.shape()[1], cb = C - ca;
  std::int64_t sa[2] = {B, ca}, sb[2] = {B, cb};
  auto ta = Tensor::empty(sa, DType::f32, dy.device());
  auto tb = Tensor::empty(sb, DType::f32, dy.device());
  if (!ta || !tb) return std::unexpected(shape_err("split alloc"));
  auto pd = dy.host_span<float>();
  auto pa = ta->host_span<float>();
  auto pb = tb->host_span<float>();
  if (!pd || !pa || !pb) return std::unexpected(shape_err("split host"));
  for (std::int64_t r = 0; r < B; ++r) {
    std::copy_n(pd->data() + r * C, ca, pa->data() + r * ca);
    std::copy_n(pd->data() + r * C + ca, cb, pb->data() + r * cb);
  }
  da = std::move(*ta);
  db = std::move(*tb);
  return {};
}

Result<Tensor> mask_rows(const Tensor& h, const Tensor& keep) {
  if (h.rank() != 2 || keep.rank() != 1 || keep.dtype() != DType::u8 || keep.shape()[0] != h.shape()[0])
    return std::unexpected(shape_err("mask_rows: h [B,H], keep u8 [B]"));
  auto ph = h.host_span<float>();
  auto pk = keep.host_span<std::uint8_t>();
  if (!ph || !pk) return std::unexpected(ph ? pk.error() : ph.error());
  auto out = Tensor::zeros(h.shape(), DType::f32, h.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  const auto B = h.shape()[0], H = h.shape()[1];
  for (std::int64_t b = 0; b < B; ++b)
    if ((*pk)[static_cast<std::size_t>(b)]) std::copy_n(ph->data() + b * H, H, po->data() + b * H);
  return out;
}

Result<Tensor> concat_rows(std::span<const Tensor> parts) {
  if (parts.empty()) return std::unexpected(shape_err("concat_rows: empty"));
  const auto C = parts[0].shape()[1];
  std::int64_t rows = 0;
  for (auto& t : parts) {
    if (t.rank() != 2 || t.shape()[1] != C || t.dtype() != DType::f32)
      return std::unexpected(shape_err("concat_rows: rank-2 f32, equal width"));
    rows += t.shape()[0];
  }
  std::int64_t sh[2] = {rows, C};
  auto out = Tensor::empty(sh, DType::f32, parts[0].device());
  if (!out) return out;
  auto po = out->host_span<float>();
  if (!po) return std::unexpected(po.error());
  std::int64_t at = 0;
  for (auto& t : parts) {
    auto p = t.host_span<float>();
    if (!p) return std::unexpected(p.error());
    std::copy(p->begin(), p->end(), po->data() + at * C);
    at += t.shape()[0];
  }
  return out;
}

Result<Tensor> attn_probs(const Tensor& qh, const Tensor& kh, const Tensor* key_mask, float scale) {
  if (qh.rank() != 4 || kh.rank() != 4 || qh.dtype() != DType::f32 || kh.dtype() != DType::f32)
    return std::unexpected(shape_err("attn_probs: q,k f32 [B,H,N,dk]"));
  for (int i = 0; i < 4; ++i)
    if (qh.shape()[i] != kh.shape()[i]) return std::unexpected(shape_err("attn_probs: q/k shapes differ"));
  const auto B = qh.shape()[0], H = qh.shape()[1], N = qh.shape()[2], dk = qh.shape()[3];
  const std::uint8_t* pm = nullptr;
  if (key_mask) {
    if (key_mask->dtype() != DType::u8 || key_mask->rank() != 2 || key_mask->shape()[0] != B ||
        key_mask->shape()[1] != N)
      return std::unexpected(shape_err("attn_probs: key mask u8 [B,N]"));
    auto m = key_mask->host_span<std::uint8_t>();
    if (!m) return std::unexpected(m.error());
    pm = m->data();
  }
  auto pq = qh.host_span<float>();
  auto pk = kh.host_span<float>();
  if (!pq || !pk) return std::unexpected(pq ? pk.error() : pq.error());
  std::int64_t sh[4] = {B, H, N, N};
  auto out = Tensor::uninit(sh, DType::f32, qh.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  if (!po) return std::unexpected(po.error());
  apply_thread_settings();
  const std::int64_t rows = B * H * N;
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows * N * dk > 65536)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::int64_t bh = r / N, b = bh / H;
    const float* q = pq->data() + r * dk;
    const float* kbase = pk->data() + bh * N * dk;
    float* w = po->data() + r * N;
    float mx = -INFINITY;
    for (std::int64_t j = 0; j < N; ++j) {
      const float sc = (pm && !pm[b * N + j]) ? kMaskFill : scale * dot_n(q, kbase + j * dk, static_cast<int>(dk));
      w[j] = sc;
      mx = std::max(mx, sc);
    }
    float sum = 0.f;
    for (std::int64_t j = 0; j < N; ++j) {
      w[j] = std::exp(w[j] - mx);
      sum += w[j];
    }
    const float inv = 1.f / sum;
    for (std::int64_t j = 0; j < N; ++j) w[j] *= inv;
  }
  return out;
}

Result<Tensor> attn_probs_backward(const Tensor& w, const Tensor& dyh, const Tensor& vh, float scale) {
  if (w.rank() != 4 || dyh.rank() != 4 || vh.rank() != 4)
    return std::unexpected(shape_err("attn_probs_backward ranks"));
  const auto B = dyh.shape()[0], H = dyh.shape()[1], N = dyh.shape()[2], dk = dyh.shape()[3];
  if (w.shape()[0] != B || w.shape()[1] != H || w.shape()[2] != N || w.shape()[3] != N)
    return std::unexpected(shape_err("attn_probs_backward: w [B,H,N,N]"));
  for (int i = 0; i < 4; ++i)
    if (vh.shape()[i] != dyh.shape()[i]) return std::unexpected(shape_err("attn_probs_backward: v/dy shapes"));
  auto pw = w.host_span<float>();
  auto pd = dyh.host_span<float>();
  auto pv = vh.host_span<float>();
  if (!pw || !pd || !pv) return std::unexpected(pw ? (pd ? pv.error() : pd.error()) : pw.error());
  auto out = Tensor::uninit(w.shape(), DType::f32, w.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  apply_thread_settings();
  const std::int64_t rows = B * H * N;
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows * N * dk > 65536)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::int64_t bh = r / N;
    const float* dy = pd->data() + r * dk;
    const float* vbase = pv->data() + bh * N * dk;
    const float* wr = pw->data() + r * N;
    float* dz = po->data() + r * N;
    float t = 0.f;
    for (std::int64_t j = 0; j < N; ++j) {
      if (wr[j] == 0.f) {  // masked (or underflowed) key: no contribution, exactly zero gradient
        dz[j] = 0.f;
        continue;
      }
      const float dw = dot_n(dy, vbase + j * dk, static_cast<int>(dk));
      dz[j] = dw;
      t += wr[j] * dw;
    }
    for (std::int64_t j = 0; j < N; ++j) dz[j] = (wr[j] == 0.f) ? 0.f : scale * wr[j] * (dz[j] - t);
  }
  return out;
}

}  // namespace gyre
