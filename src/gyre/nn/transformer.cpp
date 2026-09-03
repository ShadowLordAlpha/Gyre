#include "gyre/nn/transformer.hpp"

#include "gyre/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gyre {
namespace {

Result<Param> init_linear(std::int64_t in, std::int64_t out, std::shared_ptr<Device> d, Rng& rng,
                          float std, float scale) {
  std::int64_t wsh[2] = {in, out};
  auto W = Tensor::empty(wsh, DType::f32, d);
  if (!W) return std::unexpected(W.error());
  auto p = W->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = rng.normal(0.f, std * scale);
  return make_param(std::move(*W));
}

Result<Param> init_bias(std::int64_t n, std::shared_ptr<Device> d) {
  std::int64_t sh[1] = {n};
  auto b = Tensor::zeros(sh, DType::f32, d);
  if (!b) return std::unexpected(b.error());
  return make_param(std::move(*b));
}

Result<Param> init_ln(std::int64_t n, std::shared_ptr<Device> d) {
  std::int64_t sh[1] = {n};
  auto w = Tensor::empty(sh, DType::f32, d);
  if (!w) return std::unexpected(w.error());
  auto p = w->host_span<float>();
  if (!p) return std::unexpected(p.error());
  for (auto& v : *p) v = 1.f;
  return make_param(std::move(*w));
}

Result<Tensor> as_2d(const Tensor& x) {
  if (x.rank() == 2) return x.clone();
  if (x.rank() == 3) {
    std::int64_t sh[2] = {x.shape()[0] * x.shape()[1], x.shape()[2]};
    auto c = x.clone();
    if (!c) return c;
    return reshape(*c, sh);
  }
  return std::unexpected(make_error(Errc::invalid_shape, "as_2d"));
}

Result<Tensor> restore_rank(Tensor y, const Tensor& like) {
  if (like.rank() == 3) {
    std::int64_t sh[3] = {like.shape()[0], like.shape()[1], y.shape()[1]};
    return reshape(y, sh);
  }
  return y;
}

Result<Tensor> linear_f(const Tensor& x, const Tensor& W, const Tensor& b) {
  auto x2 = as_2d(x);
  if (!x2) return x2;
  auto y = matmul(*x2, W);
  if (!y) return y;
  auto yp = y->host_span<float>();
  auto bp = b.host_span<float>();
  if (!yp || !bp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto out = W.shape()[1];
  const auto rows = y->shape()[0];
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t j = 0; j < out; ++j)
      (*yp)[static_cast<std::size_t>(r * out + j)] += (*bp)[static_cast<std::size_t>(j)];
  return restore_rank(std::move(*y), x);
}

Result<Tensor> linear_bwd(const Tensor& x, Param& W, Param& b, const Tensor& dy) {
  auto xf = as_2d(x);
  auto df = as_2d(dy);
  if (!xf || !df) return std::unexpected(xf ? df.error() : xf.error());
  auto xt = transpose_last2(*xf);
  if (!xt) return xt;
  auto dW = matmul(*xt, *df);
  if (!dW) return dW;
  auto r1 = add_(W.grad, *dW);
  if (!r1) return std::unexpected(r1.error());
  auto db = sum_dim(*df, 0, false);
  if (!db) return db;
  auto r2 = add_(b.grad, *db);
  if (!r2) return std::unexpected(r2.error());
  auto Wt = transpose_last2(W.value);
  if (!Wt) return Wt;
  auto dx = matmul(*df, *Wt);
  if (!dx) return dx;
  return restore_rank(std::move(*dx), x);
}

Result<Tensor> ln_bwd(const Tensor& x, Param& w, Param& b, const Tensor& dy, float eps) {
  auto px = x.host_span<float>();
  auto pd = dy.host_span<float>();
  auto pw = w.value.host_span<float>();
  auto gw = w.grad.host_span<float>();
  auto gb = b.grad.host_span<float>();
  if (!px || !pd || !pw || !gw || !gb) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  auto dx = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!dx) return dx;
  auto pdx = dx->host_span<float>();
  if (!pdx) return std::unexpected(pdx.error());
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xs = px->data() + r * C;
    const float* gs = pd->data() + r * C;
    float* dxs = pdx->data() + r * C;
    float mean = 0;
    for (std::int64_t i = 0; i < C; ++i) mean += xs[i];
    mean /= static_cast<float>(C);
    float var = 0;
    for (std::int64_t i = 0; i < C; ++i) {
      float d = xs[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(C);
    float inv = 1.f / std::sqrt(var + eps);
    std::vector<float> xhat(static_cast<std::size_t>(C));
    std::vector<float> dxhat(static_cast<std::size_t>(C));
    for (std::int64_t i = 0; i < C; ++i) {
      xhat[static_cast<std::size_t>(i)] = (xs[i] - mean) * inv;
      dxhat[static_cast<std::size_t>(i)] = gs[i] * (*pw)[static_cast<std::size_t>(i)];
      (*gw)[static_cast<std::size_t>(i)] += gs[i] * xhat[static_cast<std::size_t>(i)];
      (*gb)[static_cast<std::size_t>(i)] += gs[i];
    }
    float sdx = 0, sdxh = 0;
    for (std::int64_t i = 0; i < C; ++i) {
      sdx += dxhat[static_cast<std::size_t>(i)];
      sdxh += dxhat[static_cast<std::size_t>(i)] * xhat[static_cast<std::size_t>(i)];
    }
    const float invC = inv / static_cast<float>(C);
    for (std::int64_t i = 0; i < C; ++i) {
      dxs[i] = invC * (static_cast<float>(C) * dxhat[static_cast<std::size_t>(i)] - sdx -
                       xhat[static_cast<std::size_t>(i)] * sdxh);
    }
  }
  return dx;
}

Result<Tensor> gelu_bwd(const Tensor& x, const Tensor& dy) {
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto px = x.host_span<float>();
  auto pd = dy.host_span<float>();
  auto po = out->host_span<float>();
  if (!px || !pd || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  constexpr float c = 0.044715f;
  constexpr float s = 0.7978845608028654f;
  for (std::size_t i = 0; i < px->size(); ++i) {
    float xv = (*px)[i];
    float u = s * (xv + c * xv * xv * xv);
    float th = std::tanh(u);
    float du = s * (1.f + 3.f * c * xv * xv);
    float dgelu = 0.5f * (1.f + th) + 0.5f * xv * (1.f - th * th) * du;
    (*po)[i] = (*pd)[i] * dgelu;
  }
  return out;
}

Result<Tensor> softmax_last_bwd(const Tensor& s, const Tensor& ds) {
  auto out = Tensor::empty(s.shape(), DType::f32, s.device());
  if (!out) return out;
  auto ps = s.host_span<float>();
  auto pd = ds.host_span<float>();
  auto po = out->host_span<float>();
  if (!ps || !pd || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto last = s.shape()[s.rank() - 1];
  const auto rows = s.numel() / last;
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* sv = ps->data() + r * last;
    const float* dv = pd->data() + r * last;
    float* ov = po->data() + r * last;
    float dot = 0;
    for (std::int64_t i = 0; i < last; ++i) dot += sv[i] * dv[i];
    for (std::int64_t i = 0; i < last; ++i) ov[i] = sv[i] * (dv[i] - dot);
  }
  return out;
}

Result<void> embedding_bwd(Param& w, const Tensor& idx, const Tensor& dy) {
  auto g = w.grad.host_span<float>();
  auto i = idx.host_span<std::int32_t>();
  auto d = dy.host_span<float>();
  if (!g || !i || !d) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto dim = w.value.shape()[1];
  const auto V = w.value.shape()[0];
  for (std::int64_t n = 0; n < idx.numel(); ++n) {
    auto id = (*i)[static_cast<std::size_t>(n)];
    if (id < 0 || id >= V) return std::unexpected(make_error(Errc::invalid_shape, "emb idx"));
    for (std::int64_t c = 0; c < dim; ++c)
      (*g)[static_cast<std::size_t>(id * dim + c)] += (*d)[static_cast<std::size_t>(n * dim + c)];
  }
  return {};
}

void apply_causal_recency(std::span<float> scores, std::int64_t B, std::int64_t H, std::int64_t T,
                          bool alibi) {
  for (std::int64_t b = 0; b < B; ++b) {
    for (std::int64_t h = 0; h < H; ++h) {
      const float slope =
          alibi ? std::pow(2.f, -8.f * static_cast<float>(h + 1) / static_cast<float>(H)) : 0.f;
      float* s = scores.data() + ((b * H + h) * T * T);
      for (std::int64_t t = 0; t < T; ++t) {
        for (std::int64_t j = 0; j < T; ++j) {
          if (j > t)
            s[t * T + j] += -1e9f;
          else if (alibi)
            s[t * T + j] += -slope * static_cast<float>(t - j);
        }
      }
    }
  }
}

// attn saved: q,k,v,qh,kh,vh,w,y3  then returns proj(y3)
Result<Tensor> attn_fwd(const Tensor& x, Param* qkv, std::int64_t n_head, std::vector<Tensor>& saved,
                        bool recency_alibi) {
  const auto B = x.shape()[0], T = x.shape()[1], C = x.shape()[2];
  const auto dk = C / n_head;
  auto q = linear_f(x, qkv[0].value, qkv[1].value);
  auto k = linear_f(x, qkv[2].value, qkv[3].value);
  auto v = linear_f(x, qkv[4].value, qkv[5].value);
  if (!q || !k || !v) return std::unexpected(q ? (k ? v.error() : k.error()) : q.error());
  saved.push_back(std::move(*q));
  saved.push_back(std::move(*k));
  saved.push_back(std::move(*v));
  std::int64_t sh[4] = {B, T, n_head, dk};
  auto qr = reshape(saved[saved.size() - 3], sh);
  auto kr = reshape(saved[saved.size() - 2], sh);
  auto vr = reshape(saved[saved.size() - 1], sh);
  if (!qr || !kr || !vr) return std::unexpected(make_error(Errc::invalid_shape, "reshape qkv"));
  auto qh = permute_bthd_bhtd(*qr);
  auto kh = permute_bthd_bhtd(*kr);
  auto vh = permute_bthd_bhtd(*vr);
  if (!qh || !kh || !vh) return std::unexpected(qh ? (kh ? vh.error() : kh.error()) : qh.error());
  saved.push_back(std::move(*qh));
  saved.push_back(std::move(*kh));
  saved.push_back(std::move(*vh));
  const std::size_t iQ = saved.size() - 3;
  const std::size_t iK = saved.size() - 2;
  const std::size_t iV = saved.size() - 1;
  auto kt = transpose_last2(saved[iK]);
  if (!kt) return kt;
  auto scores = bmm(saved[iQ], *kt);
  if (!scores) return scores;
  auto scale_t = Tensor::empty(scores->shape(), DType::f32, x.device());
  if (!scale_t) return scale_t;
  auto sp = scale_t->host_span<float>();
  auto scp = scores->host_span<float>();
  if (!sp || !scp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  for (std::size_t i = 0; i < sp->size(); ++i) (*sp)[i] = (*scp)[i] * scale;
  apply_causal_recency(*sp, B, n_head, T, recency_alibi);
  auto w = softmax_last(*scale_t);
  if (!w) return w;
  saved.push_back(std::move(*w));
  auto y = bmm(saved.back(), saved[iV]);
  if (!y) return y;
  auto y2 = permute_bhtd_bthd(*y);
  if (!y2) return y2;
  std::int64_t osh[3] = {B, T, C};
  auto y3 = reshape(*y2, osh);
  if (!y3) return y3;
  saved.push_back(std::move(*y3));
  return linear_f(saved.back(), qkv[6].value, qkv[7].value);
}

Result<Tensor> attn_bwd(const Tensor& x, Param* qkv, std::int64_t n_head, const Tensor& d_att,
                        const Tensor& /*q*/, const Tensor& /*k*/, const Tensor& /*v*/, const Tensor& qh,
                        const Tensor& kh, const Tensor& vh, const Tensor& w, const Tensor& y3) {
  const auto B = x.shape()[0], T = x.shape()[1], C = x.shape()[2];
  const auto dk = C / n_head;
  const float scale = 1.f / std::sqrt(static_cast<float>(dk));
  auto dy3 = linear_bwd(y3, qkv[6], qkv[7], d_att);
  if (!dy3) return dy3;
  std::int64_t sh4[4] = {B, T, n_head, dk};
  auto dy2 = reshape(*dy3, sh4);
  if (!dy2) return dy2;
  auto dyh = permute_bthd_bhtd(*dy2);
  if (!dyh) return dyh;
  auto vt = transpose_last2(vh);
  if (!vt) return vt;
  auto dw = bmm(*dyh, *vt);
  if (!dw) return dw;
  auto wt = transpose_last2(w);
  if (!wt) return wt;
  auto dvh = bmm(*wt, *dyh);
  if (!dvh) return dvh;
  auto dz = softmax_last_bwd(w, *dw);
  if (!dz) return dz;
  auto dzp = dz->host_span<float>();
  if (!dzp) return std::unexpected(dzp.error());
  for (auto& z : *dzp) z *= scale;
  // scores = qh @ kh^T  => d_qh = d_scores @ kh, d_kh = d_scores^T @ qh
  auto dqh = bmm(*dz, kh);
  if (!dqh) return dqh;
  auto dzt = transpose_last2(*dz);
  if (!dzt) return dzt;
  auto dkh = bmm(*dzt, qh);
  if (!dkh) return dkh;

  auto q_btc = permute_bhtd_bthd(*dqh);
  auto k_btc = permute_bhtd_bthd(*dkh);
  auto v_btc = permute_bhtd_bthd(*dvh);
  if (!q_btc || !k_btc || !v_btc) return std::unexpected(make_error(Errc::invalid_shape, "perm bwd"));
  std::int64_t osh[3] = {B, T, C};
  auto dq = reshape(*q_btc, osh);
  auto dkey = reshape(*k_btc, osh);
  auto dv = reshape(*v_btc, osh);
  if (!dq || !dkey || !dv) return std::unexpected(make_error(Errc::invalid_shape, "reshape bwd"));
  auto dxq = linear_bwd(x, qkv[0], qkv[1], *dq);
  auto dxk = linear_bwd(x, qkv[2], qkv[3], *dkey);
  auto dxv = linear_bwd(x, qkv[4], qkv[5], *dv);
  if (!dxq || !dxk || !dxv) return std::unexpected(dxq ? (dxk ? dxv.error() : dxk.error()) : dxq.error());
  auto t = add(*dxq, *dxk);
  if (!t) return t;
  return add(*t, *dxv);
}

constexpr int kPerLayer = 17;
// h, ln1, q,k,v, qh,kh,vh, w, y3, att, h1, ln2, fc1, gelu, fc2, h2

}  // namespace

Result<CharLM> CharLM::create(CharLMConfig c, std::shared_ptr<Device> d,
                                                Rng& rng) {
  if (c.d_model % c.n_head != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "d_model % n_head"));
  }
  CharLM m(c);
  m.n_head_ = c.n_head;
  const float resid = 1.f / std::sqrt(static_cast<float>(2 * c.n_layer));

  std::int64_t wte_sh[2] = {c.vocab, c.d_model};
  std::int64_t wpe_sh[2] = {c.block_size, c.d_model};
  auto wte = Tensor::empty(wte_sh, DType::f32, d);
  auto wpe = Tensor::empty(wpe_sh, DType::f32, d);
  if (!wte || !wpe) return std::unexpected(wte ? wpe.error() : wte.error());
  auto a = wte->host_span<float>();
  auto b = wpe->host_span<float>();
  if (!a || !b) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (auto& v : *a) v = rng.normal(0.f, 0.02f);
  for (auto& v : *b) v = rng.normal(0.f, 0.02f);
  auto p0 = make_param(std::move(*wte));
  auto p1 = make_param(std::move(*wpe));
  if (!p0 || !p1) return std::unexpected(p0 ? p1.error() : p0.error());
  m.params_.push_back(std::move(*p0));
  m.params_.push_back(std::move(*p1));

  auto push_lin = [&](std::int64_t in, std::int64_t out, float scale) -> Result<void> {
    auto W = init_linear(in, out, d, rng, 0.02f, scale);
    auto B = init_bias(out, d);
    if (!W || !B) return std::unexpected(W ? B.error() : W.error());
    m.params_.push_back(std::move(*W));
    m.params_.push_back(std::move(*B));
    return {};
  };
  auto push_ln = [&](std::int64_t n) -> Result<void> {
    auto w = init_ln(n, d);
    auto b = init_bias(n, d);
    if (!w || !b) return std::unexpected(w ? b.error() : w.error());
    m.params_.push_back(std::move(*w));
    m.params_.push_back(std::move(*b));
    return {};
  };

  for (std::int64_t i = 0; i < c.n_layer; ++i) {
    auto r = push_ln(c.d_model);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_model, c.d_model, 1.f);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_model, c.d_model, 1.f);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_model, c.d_model, 1.f);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_model, c.d_model, resid);
    if (!r) return std::unexpected(r.error());
    r = push_ln(c.d_model);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_model, c.d_ff, 1.f);
    if (!r) return std::unexpected(r.error());
    r = push_lin(c.d_ff, c.d_model, resid);
    if (!r) return std::unexpected(r.error());
  }
  auto r = push_ln(c.d_model);
  if (!r) return std::unexpected(r.error());
  r = push_lin(c.d_model, c.vocab, 1.f);
  if (!r) return std::unexpected(r.error());
  return m;
}

Result<Tensor> CharLM::forward(const Tensor& idx, ForwardCtx& ctx) {
  ctx.saved.clear();
  ctx.saved.reserve(8 + static_cast<std::size_t>(cfg_.n_layer) * 24);
  auto ic = idx.clone();
  if (!ic) return ic;
  ctx.saved.push_back(std::move(*ic));
  const auto B = idx.shape()[0], T = idx.shape()[1];
  if (T > cfg_.block_size) return std::unexpected(make_error(Errc::invalid_shape, "T > block"));
  auto tok = embedding(params_[0].value, idx);
  if (!tok) return tok;
  std::int64_t psh[1] = {T};
  auto pos = Tensor::empty(psh, DType::i32, idx.device());
  if (!pos) return pos;
  auto pp = pos->host_span<std::int32_t>();
  if (!pp) return std::unexpected(pp.error());
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(T); ++i) (*pp)[static_cast<std::size_t>(i)] = i;
  ctx.saved.push_back(std::move(*pos));
  auto pe = embedding(params_[1].value, ctx.saved.back());
  if (!pe) return pe;
  auto x = Tensor::empty(tok->shape(), DType::f32, idx.device());
  if (!x) return x;
  auto xp = x->host_span<float>();
  auto tp = tok->host_span<float>();
  auto pep = pe->host_span<float>();
  if (!xp || !tp || !pep) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = cfg_.d_model;
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t c = 0; c < C; ++c)
        (*xp)[static_cast<std::size_t>((b * T + t) * C + c)] =
            (*tp)[static_cast<std::size_t>((b * T + t) * C + c)] + (*pep)[static_cast<std::size_t>(t * C + c)];

  std::size_t pi = 2;
  ctx.saved.push_back(std::move(*x));
  std::size_t h_i = ctx.saved.size() - 1;
  for (std::int64_t layer = 0; layer < cfg_.n_layer; ++layer) {
    auto ln = layer_norm(ctx.saved[h_i], params_[pi].value, params_[pi + 1].value, 1e-5f);
    if (!ln) return ln;
    ctx.saved.push_back(std::move(*ln));
    auto att = attn_fwd(ctx.saved.back(), &params_[pi + 2], n_head_, ctx.saved, cfg_.recency_alibi);
    if (!att) return att;
    ctx.saved.push_back(std::move(*att));
    auto h1 = add(ctx.saved[h_i], ctx.saved.back());
    if (!h1) return h1;
    ctx.saved.push_back(std::move(*h1));
    auto ln2 = layer_norm(ctx.saved.back(), params_[pi + 10].value, params_[pi + 11].value, 1e-5f);
    if (!ln2) return ln2;
    ctx.saved.push_back(std::move(*ln2));
    auto fc1 = linear_f(ctx.saved.back(), params_[pi + 12].value, params_[pi + 13].value);
    if (!fc1) return fc1;
    ctx.saved.push_back(std::move(*fc1));
    auto g = gelu(ctx.saved.back());
    if (!g) return g;
    ctx.saved.push_back(std::move(*g));
    auto fc2 = linear_f(ctx.saved.back(), params_[pi + 14].value, params_[pi + 15].value);
    if (!fc2) return fc2;
    ctx.saved.push_back(std::move(*fc2));
    auto h2 = add(ctx.saved[ctx.saved.size() - 5], ctx.saved.back());  // h1 + fc2
    if (!h2) return h2;
    ctx.saved.push_back(std::move(*h2));
    h_i = ctx.saved.size() - 1;
    pi += 16;
  }
  auto lnf = layer_norm(ctx.saved[h_i], params_[pi].value, params_[pi + 1].value, 1e-5f);
  if (!lnf) return lnf;
  ctx.saved.push_back(std::move(*lnf));
  return linear_f(ctx.saved.back(), params_[pi + 2].value, params_[pi + 3].value);
}

Result<void> CharLM::backward(const Tensor& d_out, ForwardCtx& ctx) {
  // saved: [0]=idx [1]=pos [2]=x_embed
  // per layer starting at 3: ln1, q,k,v, qh,kh,vh, w, y3, att, h1, ln2, fc1, gelu, fc2, h2
  // Wait I also saved h at start of layer as x_embed then h2 of previous.
  // Layout:
  // 0 idx, 1 pos, 2 x0
  // layer: ln1, q,k,v, qh,kh,vh, w, y3, att, h1, ln2, fc1, gelu, fc2, h2   (16) — h_in is previous h2 or x0
  // last: lnf
  const auto nL = cfg_.n_layer;
  const std::size_t lnf_i = ctx.saved.size() - 1;
  const std::size_t expected = 3 + static_cast<std::size_t>(nL) * 16 + 1;
  if (ctx.saved.size() != expected) {
    return std::unexpected(make_error(Errc::unsupported, "saved size mismatch"));
  }
  std::size_t pi = 2 + static_cast<std::size_t>(nL) * 16;
  auto dlnf = linear_bwd(ctx.saved[lnf_i], params_[pi + 2], params_[pi + 3], d_out);
  if (!dlnf) return std::unexpected(dlnf.error());
  const std::size_t h_last = 2 + static_cast<std::size_t>(nL) * 16;  // last residual h2
  auto dh = ln_bwd(ctx.saved[h_last], params_[pi], params_[pi + 1], *dlnf, 1e-5f);
  if (!dh) return std::unexpected(dh.error());

  for (std::int64_t layer = nL - 1; layer >= 0; --layer) {
    const std::size_t base = 3 + static_cast<std::size_t>(layer) * 16;
    pi = 2 + static_cast<std::size_t>(layer) * 16;
    // base+0 ln1, +1 q +2 k +3 v +4 qh +5 kh +6 vh +7 w +8 y3 +9 att
    // +10 h1 +11 ln2 +12 fc1 +13 gelu +14 fc2 +15 h2
    const Tensor& h_in = (layer == 0) ? ctx.saved[2] : ctx.saved[3 + static_cast<std::size_t>(layer - 1) * 16 + 15];
    const Tensor& ln1 = ctx.saved[base + 0];
    const Tensor& q = ctx.saved[base + 1];
    const Tensor& k = ctx.saved[base + 2];
    const Tensor& v = ctx.saved[base + 3];
    const Tensor& qh = ctx.saved[base + 4];
    const Tensor& kh = ctx.saved[base + 5];
    const Tensor& vh = ctx.saved[base + 6];
    const Tensor& w = ctx.saved[base + 7];
    const Tensor& y3 = ctx.saved[base + 8];
    const Tensor& h1 = ctx.saved[base + 10];
    const Tensor& ln2 = ctx.saved[base + 11];
    const Tensor& fc1 = ctx.saved[base + 12];
    const Tensor& ge = ctx.saved[base + 13];
    const Tensor& fc2 = ctx.saved[base + 14];
    (void)fc2;

    // dh is d(h2)= d(h1+fc2)
    auto dfc2 = linear_bwd(ge, params_[pi + 14], params_[pi + 15], *dh);
    if (!dfc2) return std::unexpected(dfc2.error());
    auto dge = gelu_bwd(fc1, *dfc2);
    if (!dge) return std::unexpected(dge.error());
    auto dfc1 = linear_bwd(ln2, params_[pi + 12], params_[pi + 13], *dge);
    if (!dfc1) return std::unexpected(dfc1.error());
    auto dln2 = ln_bwd(h1, params_[pi + 10], params_[pi + 11], *dfc1, 1e-5f);
    if (!dln2) return std::unexpected(dln2.error());
    auto dh1 = add(*dh, *dln2);  // residual h1
    if (!dh1) return std::unexpected(dh1.error());
    auto datt = attn_bwd(ln1, &params_[pi + 2], n_head_, *dh1, q, k, v, qh, kh, vh, w, y3);
    if (!datt) return std::unexpected(datt.error());
    auto dln1 = ln_bwd(h_in, params_[pi], params_[pi + 1], *datt, 1e-5f);
    if (!dln1) return std::unexpected(dln1.error());
    auto next = add(*dh1, *dln1);  // d h_in from residual att + ln path
    // wait: h2 = h1 + fc2; h1 = h_in + att
    // dh2 flows to h1 and fc2
    // dh1 flows to h_in and att
    // so d_h_in = dh1 + d_from_ln1(att path)
    // dh1 already includes dh2 and dln2. Good.
    // d_h_in = dh1 (through residual) + dln1
    if (!next) return std::unexpected(next.error());
    dh = std::move(next);
  }

  // dh is d(x_embed) = d(tok + pe)
  const Tensor& idx = ctx.saved[0];
  const Tensor& pos = ctx.saved[1];
  auto r1 = embedding_bwd(params_[0], idx, *dh);
  if (!r1) return r1;
  // pe was [T,C] broadcast over B — sum dh over batch into [T,C]
  const auto B = dh->shape()[0], T = dh->shape()[1], C = dh->shape()[2];
  std::int64_t psh[2] = {T, C};
  auto dpe = Tensor::zeros(psh, DType::f32, dh->device());
  if (!dpe) return std::unexpected(dpe.error());
  auto a = dh->host_span<float>();
  auto b = dpe->host_span<float>();
  if (!a || !b) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (std::int64_t bi = 0; bi < B; ++bi)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t c = 0; c < C; ++c)
        (*b)[static_cast<std::size_t>(t * C + c)] +=
            (*a)[static_cast<std::size_t>((bi * T + t) * C + c)];
  return embedding_bwd(params_[1], pos, *dpe);
}

Result<std::vector<std::int32_t>> CharLM::generate(std::vector<std::int32_t> prefix, int max_new,
                                                   std::shared_ptr<Device> d, Rng* rng,
                                                   float temperature) {
  if (temperature > 0.f && !rng) {
    return std::unexpected(make_error(Errc::unsupported, "temperature > 0 requires Rng"));
  }
  if (prefix.empty()) {
    return std::unexpected(make_error(Errc::invalid_shape, "empty prompt"));
  }
  for (int n = 0; n < max_new; ++n) {
    const auto ctx_n = std::min(static_cast<std::int64_t>(prefix.size()), cfg_.block_size);
    std::int64_t sh[2] = {1, ctx_n};
    std::vector<std::byte> bytes(static_cast<std::size_t>(ctx_n) * 4);
    std::memcpy(bytes.data(), prefix.data() + prefix.size() - static_cast<std::size_t>(ctx_n),
                static_cast<std::size_t>(ctx_n) * 4);
    auto idx = Tensor::from_host(bytes, sh, DType::i32, d);
    if (!idx) return std::unexpected(idx.error());
    ForwardCtx ctx;
    ctx.train = false;
    auto logits = forward(*idx, ctx);
    if (!logits) return std::unexpected(logits.error());
    auto p = logits->host_span<float>();
    if (!p) return std::unexpected(p.error());
    const auto V = cfg_.vocab;
    const float* last = p->data() + (ctx_n - 1) * V;
    std::int32_t next = 0;
    if (temperature <= 0.f) {
      float m = last[0];
      for (std::int32_t i = 1; i < V; ++i)
        if (last[i] > m) {
          m = last[i];
          next = i;
        }
    } else {
      float maxv = last[0];
      for (std::int32_t i = 1; i < V; ++i) maxv = std::max(maxv, last[i]);
      std::vector<float> pr(static_cast<std::size_t>(V));
      float sum = 0;
      for (std::int32_t i = 0; i < V; ++i) {
        pr[static_cast<std::size_t>(i)] = std::exp((last[i] - maxv) / temperature);
        sum += pr[static_cast<std::size_t>(i)];
      }
      float u = rng->uniform01() * sum;
      float acc = 0;
      next = static_cast<std::int32_t>(V - 1);
      for (std::int32_t i = 0; i < V; ++i) {
        acc += pr[static_cast<std::size_t>(i)];
        if (u <= acc) {
          next = i;
          break;
        }
      }
    }
    prefix.push_back(next);
  }
  return prefix;
}

}  // namespace gyre
