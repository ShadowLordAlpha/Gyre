#include "gyre/nn/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace gyre {
namespace {

Result<Tensor> as_2d(const Tensor& x) { return flatten_leading(x); }

Result<Tensor> restore_like(Tensor y, const Tensor& like) { return unflatten_like(std::move(y), like); }

}  // namespace

Result<MoeTopK> moe_topk(const Tensor& logits, int k, float router_softcap) {
  if (logits.dtype() != DType::f32 || logits.rank() != 2) {
    return std::unexpected(make_error(Errc::invalid_shape, "moe logits [N,E]"));
  }
  const auto N = logits.shape()[0];
  const auto E = logits.shape()[1];
  if (k < 1 || k > E) return std::unexpected(make_error(Errc::invalid_shape, "moe k"));
  Result<Tensor> scores = logits.clone();
  if (!scores) return std::unexpected(scores.error());
  if (router_softcap > 0.f) {
    auto c = softcap(logits, router_softcap);
    if (!c) return std::unexpected(c.error());
    scores = std::move(c);
  }
  auto sm = softmax_last(*scores);
  if (!sm) return std::unexpected(sm.error());
  auto ps = sm->host_span<float>();
  if (!ps) return std::unexpected(make_error(Errc::not_cpu, "host"));

  std::int64_t ish[] = {N, k};
  auto idx = Tensor::empty(ish, DType::i32, logits.device());
  auto wts = Tensor::empty(ish, DType::f32, logits.device());
  if (!idx || !wts) return std::unexpected(idx ? wts.error() : idx.error());
  auto pi = idx->host_span<std::int32_t>();
  auto pw = wts->host_span<float>();
  if (!pi || !pw) return std::unexpected(make_error(Errc::not_cpu, "host"));

  std::vector<std::pair<float, int>> row(static_cast<std::size_t>(E));
  for (std::int64_t n = 0; n < N; ++n) {
    for (int e = 0; e < E; ++e) {
      row[static_cast<std::size_t>(e)] = {(*ps)[static_cast<std::size_t>(n * E + e)], e};
    }
    std::partial_sort(row.begin(), row.begin() + k, row.end(),
                      [](auto a, auto b) { return a.first > b.first; });
    float z = 0.f;
    for (int j = 0; j < k; ++j) z += row[static_cast<std::size_t>(j)].first;
    if (z <= 0.f) z = 1.f;
    for (int j = 0; j < k; ++j) {
      (*pi)[static_cast<std::size_t>(n * k + j)] = row[static_cast<std::size_t>(j)].second;
      (*pw)[static_cast<std::size_t>(n * k + j)] = row[static_cast<std::size_t>(j)].first / z;
    }
  }
  return MoeTopK{std::move(*idx), std::move(*wts)};
}

Result<Tensor> swiglu_ffn(const Tensor& x, const SwiGLUWeights& w) {
  if (!w.w1 || !w.w3 || !w.w2) {
    return std::unexpected(make_error(Errc::invalid_shape, "swiglu weights"));
  }
  auto x2 = as_2d(x);
  if (!x2) return x2;
  auto g = matmul(*x2, *w.w1);
  if (!g) return g;
  auto u = matmul(*x2, *w.w3);
  if (!u) return u;
  auto h = swiglu(*g, *u);
  if (!h) return h;
  auto y = matmul(*h, *w.w2);
  if (!y) return y;
  return restore_like(std::move(*y), x);
}

Result<Tensor> residual_moe(const Tensor& x, const SwiGLUWeights& dense, const Tensor& router_w,
                            std::span<const SwiGLUWeights> experts, int k, float router_softcap) {
  auto x2 = as_2d(x);
  if (!x2) return x2;
  const auto N = x2->shape()[0];
  const auto D = x2->shape()[1];
  const auto E = static_cast<std::int64_t>(experts.size());
  if (router_w.rank() != 2 || router_w.shape()[0] != D || router_w.shape()[1] != E) {
    return std::unexpected(make_error(Errc::invalid_shape, "router"));
  }
  auto logits = matmul(*x2, router_w);
  if (!logits) return logits;
  auto top = moe_topk(*logits, k, router_softcap);
  if (!top) return std::unexpected(top.error());

  // An all-null `dense` means a plain sparse MoE (experts only, Grok residual_moe = false).
  const bool has_dense = dense.w1 || dense.w2 || dense.w3;

  auto out = Tensor::zeros(x2->shape(), DType::f32, x.device());
  if (!out) return out;
  auto po = out->host_span<float>();
  auto px = x2->host_span<float>();
  auto pi = top->indices.host_span<std::int32_t>();
  auto pw = top->weights.host_span<float>();
  if (!po || !px || !pi || !pw) return std::unexpected(make_error(Errc::not_cpu, "host"));
  if (has_dense) {
    auto dens = swiglu_ffn(*x2, dense);
    if (!dens) return dens;
    // matmul only checks inner dims, so a dense w2 narrower than D would be read past its end.
    if (dens->rank() != 2 || dens->shape()[0] != N || dens->shape()[1] != D) {
      return std::unexpected(make_error(Errc::invalid_shape, "moe dense output width != D"));
    }
    auto pd = dens->host_span<float>();
    if (!pd) return std::unexpected(make_error(Errc::not_cpu, "host"));
    std::memcpy(po->data(), pd->data(), static_cast<std::size_t>(N * D) * sizeof(float));
  }

  // Group tokens by expert so each expert runs one [n_e, D] SwiGLU instead of N*k single-row
  // calls (each of which allocated tensors). Outputs are added in expert-id order, which is fixed,
  // so the result stays deterministic.
  const auto NK = static_cast<std::size_t>(N) * static_cast<std::size_t>(k);
  std::vector<std::vector<std::int64_t>> slots(experts.size());  // (n*k + j) routed to expert e
  for (std::size_t s = 0; s < NK; ++s) {
    const auto e = (*pi)[s];
    if (e < 0 || static_cast<std::size_t>(e) >= experts.size()) {
      return std::unexpected(make_error(Errc::invalid_shape, "expert id"));
    }
    slots[static_cast<std::size_t>(e)].push_back(static_cast<std::int64_t>(s));
  }
  std::vector<float> rows;
  for (std::size_t e = 0; e < experts.size(); ++e) {
    const auto& sl = slots[e];
    if (sl.empty()) continue;
    const auto ne = static_cast<std::int64_t>(sl.size());
    rows.resize(static_cast<std::size_t>(ne * D));
    for (std::int64_t r = 0; r < ne; ++r) {
      const auto n = sl[static_cast<std::size_t>(r)] / k;
      std::memcpy(rows.data() + r * D, px->data() + n * D, static_cast<std::size_t>(D) * sizeof(float));
    }
    std::int64_t rsh[] = {ne, D};
    auto tok = Tensor::from_host(std::as_bytes(std::span<const float>(rows)), rsh, DType::f32, x.device());
    if (!tok) return std::unexpected(tok.error());
    auto ye = swiglu_ffn(*tok, experts[e]);
    if (!ye) return ye;
    if (ye->rank() != 2 || ye->shape()[0] != ne || ye->shape()[1] != D) {
      return std::unexpected(make_error(Errc::invalid_shape, "moe expert output width != D"));
    }
    auto py = ye->host_span<float>();
    if (!py) return std::unexpected(make_error(Errc::not_cpu, "host"));
    for (std::int64_t r = 0; r < ne; ++r) {
      const auto s = sl[static_cast<std::size_t>(r)];
      const auto n = s / k;
      const float wt = (*pw)[static_cast<std::size_t>(s)];
      const float* y = py->data() + r * D;
      float* o = po->data() + n * D;
      for (std::int64_t d = 0; d < D; ++d) o[d] += wt * y[d];
    }
  }
  return restore_like(std::move(*out), x);
}

}  // namespace gyre
