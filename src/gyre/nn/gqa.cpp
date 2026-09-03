#include "gyre/nn/gqa.hpp"

#include <cmath>
#include <cstring>

namespace gyre {

Result<Tensor> repeat_kv_heads(const Tensor& kv, std::int64_t n_q) {
  if (kv.dtype() != DType::f32 || kv.rank() != 4) {
    return std::unexpected(make_error(Errc::invalid_shape, "repeat_kv rank 4 f32"));
  }
  const auto B = kv.shape()[0], Hkv = kv.shape()[1], T = kv.shape()[2], D = kv.shape()[3];
  if (n_q <= 0 || Hkv <= 0 || n_q % Hkv != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "repeat_kv heads"));
  }
  const auto rep = n_q / Hkv;
  if (rep == 1) return kv.clone();
  std::int64_t osh[] = {B, n_q, T, D};
  auto out = Tensor::empty(osh, DType::f32, kv.device());
  if (!out) return out;
  auto ps = kv.host_span<float>();
  auto pd = out->host_span<float>();
  if (!ps || !pd) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto td = T * D;
  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t h = 0; h < Hkv; ++h) {
      const float* src = ps->data() + ((b * Hkv + h) * td);
      for (std::int64_t r = 0; r < rep; ++r) {
        float* dst = pd->data() + ((b * n_q + h * rep + r) * td);
        std::memcpy(dst, src, static_cast<std::size_t>(td) * sizeof(float));
      }
    }
  }
  return out;
}

Result<Tensor> gqa_causal(const Tensor& q, const Tensor& k, const Tensor& v, float attn_softcap,
                          float attn_temp_scale) {
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
    return std::unexpected(make_error(Errc::invalid_shape, "gqa rank"));
  }
  if (q.shape()[0] != k.shape()[0] || q.shape()[3] != k.shape()[3] || k.shape()[1] != v.shape()[1] ||
      k.shape()[2] != v.shape()[2] || k.shape()[3] != v.shape()[3]) {
    return std::unexpected(make_error(Errc::invalid_shape, "gqa qkv"));
  }
  const auto Hq = q.shape()[1];
  const auto D = q.shape()[3];
  const auto Tq = q.shape()[2];
  const auto Tk = k.shape()[2];
  auto kr = repeat_kv_heads(k, Hq);
  if (!kr) return kr;
  auto vr = repeat_kv_heads(v, Hq);
  if (!vr) return vr;
  auto kt = transpose_last2(*kr);
  if (!kt) return kt;
  auto scores = bmm(q, *kt);
  if (!scores) return scores;
  const float inv = attn_temp_scale / std::sqrt(static_cast<float>(D));
  auto sp = scores->host_span<float>();
  if (!sp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (auto& x : *sp) x *= inv;
  if (attn_softcap > 0.f) {
    auto capped = softcap(*scores, attn_softcap);
    if (!capped) return capped;
    *scores = std::move(*capped);
    sp = scores->host_span<float>();
    if (!sp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  }
  // Causal: query t cannot see key s > t + (Tk - Tq)  (decode: Tq=1, Tk=prefix)
  const auto offset = Tk - Tq;
  auto B = q.shape()[0];
  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t h = 0; h < Hq; ++h) {
      for (std::int64_t t = 0; t < Tq; ++t) {
        float* row = sp->data() + (((b * Hq + h) * Tq + t) * Tk);
        const auto max_s = t + offset;
        for (std::int64_t s = 0; s < Tk; ++s) {
          if (s > max_s) row[s] = -1e9f;
        }
      }
    }
  }
  auto att = softmax_last(*scores);
  if (!att) return att;
  return bmm(*att, *vr);
}

}  // namespace gyre
