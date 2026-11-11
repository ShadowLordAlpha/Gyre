#include "gyre/rl/gru.hpp"

#include "gyre/ops.hpp"

#include <cmath>

namespace gyre {
namespace {
Error err(const char* m) { return make_error(Errc::invalid_shape, m); }
float sigm(float v) { return 1.f / (1.f + std::exp(-v)); }
}  // namespace

Result<Gru> Gru::create(std::int64_t in, std::int64_t hidden, std::shared_ptr<Device> d, Rng& rng) {
  if (in < 1 || hidden < 1) return std::unexpected(err("Gru dims"));
  const float k = 1.f / std::sqrt(static_cast<float>(hidden));
  std::int64_t s_wi[2] = {in, 3 * hidden}, s_wh[2] = {hidden, 3 * hidden}, s_b[1] = {3 * hidden};
  auto wi = Tensor::zeros(s_wi, DType::f32, d);
  auto wh = Tensor::zeros(s_wh, DType::f32, d);
  auto bi = Tensor::zeros(s_b, DType::f32, d);
  auto bh = Tensor::zeros(s_b, DType::f32, d);
  if (!wi || !wh || !bi || !bh) return std::unexpected(err("Gru alloc"));
  for (Tensor* w : {&*wi, &*wh}) {
    auto p = w->host_span<float>();
    if (!p) return std::unexpected(p.error());
    (void)rng.fill_f32(*p, -k, k);
  }
  std::vector<Param> ps;
  for (Tensor* t : {&*wi, &*bi, &*wh, &*bh}) {
    auto p = make_param(std::move(*t));
    if (!p) return std::unexpected(p.error());
    ps.push_back(std::move(*p));
  }
  return Gru(std::move(ps), in, hidden);
}

Result<Tensor> Gru::step(const Tensor& x, const Tensor& h_in, GruCache* cache) {
  if (x.rank() != 2 || h_in.rank() != 2 || x.shape()[0] != h_in.shape()[0] || x.shape()[1] != in_ ||
      h_in.shape()[1] != hidden_)
    return std::unexpected(err("Gru::step: x [B,In], h [B,H]"));
  const auto B = x.shape()[0], H = hidden_;
  auto gi = linear(x, params_[0].value, params_[1].value);
  auto gh = linear(h_in, params_[2].value, params_[3].value);
  if (!gi || !gh) return std::unexpected(gi ? gh.error() : gi.error());
  std::int64_t sh[2] = {B, H};
  auto hn = Tensor::empty(sh, DType::f32, x.device());
  auto r = Tensor::empty(sh, DType::f32, x.device());
  auto z = Tensor::empty(sh, DType::f32, x.device());
  auto n = Tensor::empty(sh, DType::f32, x.device());
  auto ghn = Tensor::empty(sh, DType::f32, x.device());
  if (!hn || !r || !z || !n || !ghn) return std::unexpected(err("Gru alloc"));
  auto pgi = gi->host_span<float>();
  auto pgh = gh->host_span<float>();
  auto ph = h_in.host_span<float>();
  auto pho = hn->host_span<float>();
  auto pr = r->host_span<float>();
  auto pz = z->host_span<float>();
  auto pn = n->host_span<float>();
  auto pg = ghn->host_span<float>();
  if (!pgi || !pgh || !ph || !pho || !pr || !pz || !pn || !pg) return std::unexpected(err("Gru host"));
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t j = 0; j < H; ++j) {
      const auto i3 = static_cast<std::size_t>(b * 3 * H + j), o = static_cast<std::size_t>(b * H + j);
      const float rr = sigm((*pgi)[i3] + (*pgh)[i3]);
      const float zz = sigm((*pgi)[i3 + H] + (*pgh)[i3 + H]);
      const float gn = (*pgh)[i3 + 2 * H];
      const float nn = std::tanh((*pgi)[i3 + 2 * H] + rr * gn);
      (*pr)[o] = rr;
      (*pz)[o] = zz;
      (*pn)[o] = nn;
      (*pg)[o] = gn;
      (*pho)[o] = (1.f - zz) * nn + zz * (*ph)[o];
    }
  if (cache) *cache = GruCache{x, h_in, std::move(*r), std::move(*z), std::move(*n), std::move(*ghn)};
  return std::move(*hn);
}

Result<GruStepGrads> Gru::step_backward(const GruCache& c, const Tensor& dh_out) {
  const auto B = c.x.shape()[0], H = hidden_;
  if (dh_out.rank() != 2 || dh_out.shape()[0] != B || dh_out.shape()[1] != H)
    return std::unexpected(err("Gru::step_backward: dh [B,H]"));
  std::int64_t s3[2] = {B, 3 * H}, s1[2] = {B, H};
  auto dgi = Tensor::empty(s3, DType::f32, c.x.device());
  auto dgh = Tensor::empty(s3, DType::f32, c.x.device());
  auto dhd = Tensor::empty(s1, DType::f32, c.x.device());
  if (!dgi || !dgh || !dhd) return std::unexpected(err("Gru alloc"));
  auto pdh = dh_out.host_span<float>();
  auto ph = c.h_in.host_span<float>();
  auto pr = c.r.host_span<float>();
  auto pz = c.z.host_span<float>();
  auto pn = c.n.host_span<float>();
  auto pg = c.ghn.host_span<float>();
  auto pgi = dgi->host_span<float>();
  auto pgh = dgh->host_span<float>();
  auto pd = dhd->host_span<float>();
  if (!pdh || !ph || !pr || !pz || !pn || !pg || !pgi || !pgh || !pd) return std::unexpected(err("Gru host"));
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t j = 0; j < H; ++j) {
      const auto o = static_cast<std::size_t>(b * H + j), i3 = static_cast<std::size_t>(b * 3 * H + j);
      const float g = (*pdh)[o], r = (*pr)[o], z = (*pz)[o], n = (*pn)[o];
      const float dn = g * (1.f - z);
      const float dz = g * ((*ph)[o] - n);
      (*pd)[o] = g * z;
      const float dpre_n = dn * (1.f - n * n);
      const float dr = dpre_n * (*pg)[o];
      const float dpre_r = dr * r * (1.f - r);
      const float dpre_z = dz * z * (1.f - z);
      (*pgi)[i3] = dpre_r;
      (*pgi)[i3 + H] = dpre_z;
      (*pgi)[i3 + 2 * H] = dpre_n;
      (*pgh)[i3] = dpre_r;
      (*pgh)[i3 + H] = dpre_z;
      (*pgh)[i3 + 2 * H] = dpre_n * r;
    }
  // parameter grads and input grads
  auto xt = transpose_last2(c.x);
  auto ht = transpose_last2(c.h_in);
  if (!xt || !ht) return std::unexpected(err("Gru transpose"));
  auto dWi = matmul(*xt, *dgi);
  auto dWh = matmul(*ht, *dgh);
  auto dbi = sum_dim(*dgi, 0, false);
  auto dbh = sum_dim(*dgh, 0, false);
  if (!dWi || !dWh || !dbi || !dbh) return std::unexpected(err("Gru grad matmul"));
  if (auto r = add_(params_[0].grad, *dWi); !r) return std::unexpected(r.error());
  if (auto r = add_(params_[1].grad, *dbi); !r) return std::unexpected(r.error());
  if (auto r = add_(params_[2].grad, *dWh); !r) return std::unexpected(r.error());
  if (auto r = add_(params_[3].grad, *dbh); !r) return std::unexpected(r.error());
  auto Wit = transpose_last2(params_[0].value);
  auto Wht = transpose_last2(params_[2].value);
  if (!Wit || !Wht) return std::unexpected(err("Gru transpose W"));
  auto dx = matmul(*dgi, *Wit);
  auto dhr = matmul(*dgh, *Wht);
  if (!dx || !dhr) return std::unexpected(err("Gru dx matmul"));
  auto dh_in = add(*dhd, *dhr);
  if (!dh_in) return std::unexpected(dh_in.error());
  return GruStepGrads{std::move(*dx), std::move(*dh_in)};
}

}  // namespace gyre
