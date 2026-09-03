#include "gyre/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

#if defined(GYRE_OPENMP)
#include <omp.h>
#endif

namespace gyre {
namespace {

Result<void> same_dev_dtype_shape(const Tensor& a, const Tensor& b, bool check_shape) {
  if (a.device().get() != b.device().get()) {
    return std::unexpected(make_error(Errc::mixed_device, "mixed device"));
  }
  if (a.dtype() != b.dtype()) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "dtype mismatch"));
  }
  if (check_shape) {
    auto sa = a.shape();
    auto sb = b.shape();
    if (sa.size() != sb.size() || !std::equal(sa.begin(), sa.end(), sb.begin())) {
      return std::unexpected(make_error(Errc::invalid_shape, "shape mismatch"));
    }
  }
  return {};
}

Result<std::span<float>> f32(Tensor& t) {
  if (t.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "expected f32"));
  }
  return t.host_span<float>();
}

Result<std::span<const float>> f32c(const Tensor& t) {
  if (t.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "expected f32"));
  }
  return t.host_span<float>();
}

void init_threads() {
#if defined(GYRE_OPENMP)
  static bool once = false;
  if (once) return;
  once = true;
#ifdef _MSC_VER
  char* e = nullptr;
  size_t elen = 0;
  if (_dupenv_s(&e, &elen, "GYRE_THREADS") == 0 && e) {
    int n = std::atoi(e);
    free(e);
    if (n > 0) omp_set_num_threads(n);
  }
#else
  if (const char* e = std::getenv("GYRE_THREADS")) {
    int n = std::atoi(e);
    if (n > 0) omp_set_num_threads(n);
  }
#endif
#endif
}

// C[M,N] += A[M,K] @ B[K,N]  (C must be zeroed by caller or we zero here)
void gemm_nn(const float* A, const float* B, float* C, std::int64_t M, std::int64_t K,
             std::int64_t N) {
  init_threads();
#if defined(GYRE_OPENMP)
  const int nested = omp_in_parallel();
#pragma omp parallel for schedule(static) if (!nested)
#endif
  for (std::int64_t i = 0; i < M; ++i) {
    float* Ci = C + i * N;
    std::memset(Ci, 0, static_cast<std::size_t>(N) * sizeof(float));
    for (std::int64_t k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* Bk = B + k * N;
      for (std::int64_t j = 0; j < N; ++j) Ci[j] += a * Bk[j];
    }
  }
}

}  // namespace

Result<Tensor> add(const Tensor& a, const Tensor& b) {
  auto ok = same_dev_dtype_shape(a, b, true);
  if (!ok) return std::unexpected(ok.error());
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::unsupported, "add f32 only"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto pb = f32c(b);
  auto po = f32(*out);
  if (!pa || !pb || !po) return std::unexpected(pa ? (pb ? po.error() : pb.error()) : pa.error());
  init_threads();
  const auto n = a.numel();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
  for (std::int64_t i = 0; i < n; ++i)
    (*po)[static_cast<std::size_t>(i)] =
        (*pa)[static_cast<std::size_t>(i)] + (*pb)[static_cast<std::size_t>(i)];
  return out;
}

Result<Tensor> mul(const Tensor& a, const Tensor& b) {
  auto ok = same_dev_dtype_shape(a, b, true);
  if (!ok) return std::unexpected(ok.error());
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::unsupported, "mul f32 only"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto pb = f32c(b);
  auto po = f32(*out);
  if (!pa || !pb || !po) return std::unexpected(pa ? (pb ? po.error() : pb.error()) : pa.error());
  init_threads();
  const auto n = a.numel();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
  for (std::int64_t i = 0; i < n; ++i)
    (*po)[static_cast<std::size_t>(i)] =
        (*pa)[static_cast<std::size_t>(i)] * (*pb)[static_cast<std::size_t>(i)];
  return out;
}

Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
  if (a.device().get() != b.device().get()) {
    return std::unexpected(make_error(Errc::mixed_device, "mixed device"));
  }
  if (a.dtype() != DType::f32 || b.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "matmul f32"));
  }
  if (a.rank() != 2 || b.rank() != 2) {
    return std::unexpected(make_error(Errc::invalid_shape, "matmul rank-2"));
  }
  const auto M = a.shape()[0], K = a.shape()[1], K2 = b.shape()[0], N = b.shape()[1];
  if (K != K2) return std::unexpected(make_error(Errc::invalid_shape, "matmul inner dim"));
  std::int64_t osh[2] = {M, N};
  auto out = Tensor::empty(osh, DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto pb = f32c(b);
  auto po = f32(*out);
  if (!pa || !pb || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  gemm_nn(pa->data(), pb->data(), po->data(), M, K, N);
  return out;
}

Result<Tensor> bmm(const Tensor& a, const Tensor& b) {
  if (a.device().get() != b.device().get()) {
    return std::unexpected(make_error(Errc::mixed_device, "mixed device"));
  }
  if (a.dtype() != DType::f32 || b.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "bmm f32"));
  }
  if (a.rank() < 3 || b.rank() != a.rank()) {
    return std::unexpected(make_error(Errc::invalid_shape, "bmm rank"));
  }
  const int r = a.rank();
  for (int i = 0; i < r - 2; ++i) {
    if (a.shape()[i] != b.shape()[i]) {
      return std::unexpected(make_error(Errc::invalid_shape, "bmm batch dims"));
    }
  }
  const auto M = a.shape()[r - 2], K = a.shape()[r - 1], K2 = b.shape()[r - 2], N = b.shape()[r - 1];
  if (K != K2) return std::unexpected(make_error(Errc::invalid_shape, "bmm inner"));
  std::int64_t batch = 1;
  std::array<std::int64_t, 8> osh{};
  for (int i = 0; i < r - 2; ++i) {
    osh[i] = a.shape()[i];
    batch *= a.shape()[i];
  }
  osh[r - 2] = M;
  osh[r - 1] = N;
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), r), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto pb = f32c(b);
  auto po = f32(*out);
  if (!pa || !pb || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const std::int64_t a_mat = M * K, b_mat = K * N, o_mat = M * N;
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t bi = 0; bi < batch; ++bi) {
    gemm_nn(pa->data() + bi * a_mat, pb->data() + bi * b_mat, po->data() + bi * o_mat, M, K, N);
  }
  return out;
}

Result<Tensor> sum(const Tensor& a) {
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "sum f32"));
  }
  std::int64_t sh[1] = {1};
  // rank-0: use shape {}
  auto out = Tensor::empty(std::span<const std::int64_t>(), DType::f32, a.device());
  if (!out) return out;
  (void)sh;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  float s = 0;
  for (auto v : *pa) s += v;
  (*po)[0] = s;
  return out;
}

Result<Tensor> sum_dim(const Tensor& a, int axis, bool keepdim) {
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "sum_dim f32"));
  }
  if (axis < 0 || axis >= a.rank()) {
    return std::unexpected(make_error(Errc::invalid_shape, "axis"));
  }
  std::vector<std::int64_t> osh;
  for (int i = 0; i < a.rank(); ++i) {
    if (i == axis) {
      if (keepdim) osh.push_back(1);
    } else {
      osh.push_back(a.shape()[i]);
    }
  }
  auto out = Tensor::zeros(osh, DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));

  std::array<std::int64_t, 8> stride{};
  stride[a.rank() - 1] = 1;
  for (int i = a.rank() - 2; i >= 0; --i) stride[i] = stride[i + 1] * a.shape()[i + 1];

  std::array<std::int64_t, 8> ostr{};
  if (!osh.empty()) {
    ostr[osh.size() - 1] = 1;
    for (int i = static_cast<int>(osh.size()) - 2; i >= 0; --i) ostr[i] = ostr[i + 1] * osh[i + 1];
  }

  const std::int64_t n = a.numel();
  for (std::int64_t lin = 0; lin < n; ++lin) {
    std::array<std::int64_t, 8> idx{};
    auto rem = lin;
    for (int i = 0; i < a.rank(); ++i) {
      idx[i] = rem / stride[i];
      rem %= stride[i];
    }
    std::int64_t olin = 0;
    int oi = 0;
    for (int i = 0; i < a.rank(); ++i) {
      if (i == axis) {
        if (keepdim) {
          olin += 0 * ostr[oi];
          ++oi;
        }
      } else {
        olin += idx[i] * ostr[oi];
        ++oi;
      }
    }
    (*po)[static_cast<std::size_t>(olin)] += (*pa)[static_cast<std::size_t>(lin)];
  }
  return out;
}

Result<Tensor> reshape(const Tensor& a, std::span<const std::int64_t> new_shape) {
  return Tensor::view_reshape(a, new_shape);
}

Result<Tensor> transpose_last2(const Tensor& a) {
  if (a.rank() < 2) return std::unexpected(make_error(Errc::invalid_shape, "transpose rank"));
  if (a.dtype() != DType::f32 && a.dtype() != DType::i32) {
    return std::unexpected(make_error(Errc::unsupported, "transpose dtype"));
  }
  std::array<std::int64_t, 8> osh{};
  for (int i = 0; i < a.rank(); ++i) osh[i] = a.shape()[i];
  std::swap(osh[a.rank() - 2], osh[a.rank() - 1]);
  auto out = Tensor::empty(std::span<const std::int64_t>(osh.data(), a.rank()), a.dtype(), a.device());
  if (!out) return out;
  const auto d0 = a.shape()[a.rank() - 2];
  const auto d1 = a.shape()[a.rank() - 1];
  std::int64_t batch = 1;
  for (int i = 0; i < a.rank() - 2; ++i) batch *= a.shape()[i];
  auto src = a.host_bytes();
  auto dst = out->host_bytes();
  if (!src || !dst) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const std::size_t es = dtype_size(a.dtype());
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t i = 0; i < d0; ++i)
      for (std::int64_t j = 0; j < d1; ++j) {
        auto si = (b * d0 * d1 + i * d1 + j) * es;
        auto di = (b * d1 * d0 + j * d0 + i) * es;
        std::memcpy(dst->data() + di, src->data() + si, es);
      }
  }
  return out;
}

Result<Tensor> embedding(const Tensor& weight, const Tensor& indices_i32) {
  if (weight.rank() != 2 || weight.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::invalid_shape, "embedding weight [V,d] f32"));
  }
  if (indices_i32.dtype() != DType::i32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "indices i32"));
  }
  if (weight.device().get() != indices_i32.device().get()) {
    return std::unexpected(make_error(Errc::mixed_device, "mixed"));
  }
  const auto V = weight.shape()[0], d = weight.shape()[1];
  std::vector<std::int64_t> osh(indices_i32.shape().begin(), indices_i32.shape().end());
  osh.push_back(d);
  auto out = Tensor::empty(osh, DType::f32, weight.device());
  if (!out) return out;
  auto w = f32c(weight);
  auto idx = indices_i32.host_span<std::int32_t>();
  auto o = f32(*out);
  if (!w || !idx || !o) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (std::int64_t i = 0; i < indices_i32.numel(); ++i) {
    auto id = (*idx)[static_cast<std::size_t>(i)];
    if (id < 0 || id >= V) {
      return std::unexpected(make_error(Errc::invalid_shape, "index OOB"));
    }
    std::memcpy(o->data() + i * d, w->data() + static_cast<std::int64_t>(id) * d,
                static_cast<std::size_t>(d) * sizeof(float));
  }
  return out;
}

Result<Tensor> permute_bthd_bhtd(const Tensor& x) {
  if (x.rank() != 4 || x.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::invalid_shape, "permute [B,T,H,d] f32"));
  }
  const auto B = x.shape()[0], T = x.shape()[1], H = x.shape()[2], D = x.shape()[3];
  std::int64_t osh[4] = {B, H, T, D};
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  auto s = f32c(x);
  auto d = f32(*out);
  if (!s || !d) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t t = 0; t < T; ++t)
      for (std::int64_t h = 0; h < H; ++h)
        for (std::int64_t i = 0; i < D; ++i)
          (*d)[static_cast<std::size_t>(((b * H + h) * T + t) * D + i)] =
              (*s)[static_cast<std::size_t>(((b * T + t) * H + h) * D + i)];
  return out;
}

Result<Tensor> permute_bhtd_bthd(const Tensor& x) {
  if (x.rank() != 4 || x.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::invalid_shape, "permute [B,H,T,d] f32"));
  }
  const auto B = x.shape()[0], H = x.shape()[1], T = x.shape()[2], D = x.shape()[3];
  std::int64_t osh[4] = {B, T, H, D};
  auto out = Tensor::empty(osh, DType::f32, x.device());
  if (!out) return out;
  auto s = f32c(x);
  auto d = f32(*out);
  if (!s || !d) return std::unexpected(make_error(Errc::not_cpu, "host"));
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t t = 0; t < T; ++t)
        for (std::int64_t i = 0; i < D; ++i)
          (*d)[static_cast<std::size_t>(((b * T + t) * H + h) * D + i)] =
              (*s)[static_cast<std::size_t>(((b * H + h) * T + t) * D + i)];
  return out;
}

Result<Tensor> narrow_rows(const Tensor& x, std::int64_t start, std::int64_t count) {
  if (x.rank() != 2) return std::unexpected(make_error(Errc::invalid_shape, "narrow_rows rank 2"));
  const auto N = x.shape()[0], D = x.shape()[1];
  if (start < 0 || count < 0 || start + count > N) {
    return std::unexpected(make_error(Errc::invalid_shape, "narrow_rows range"));
  }
  std::int64_t osh[2] = {count, D};
  auto out = Tensor::empty(osh, x.dtype(), x.device());
  if (!out) return out;
  auto src = x.host_bytes();
  auto dst = out->host_bytes();
  if (!src || !dst) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto es = dtype_size(x.dtype());
  std::memcpy(dst->data(), src->data() + start * D * static_cast<std::int64_t>(es),
              static_cast<std::size_t>(count * D * static_cast<std::int64_t>(es)));
  return out;
}

Result<Tensor> gelu(const Tensor& a) {
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "gelu f32"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  constexpr float c = 0.044715f;
  constexpr float s = 0.7978845608028654f;  // sqrt(2/pi)
  init_threads();
  const auto n = static_cast<std::int64_t>(pa->size());
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
  for (std::int64_t i = 0; i < n; ++i) {
    float x = (*pa)[static_cast<std::size_t>(i)];
    float u = s * (x + c * x * x * x);
    (*po)[static_cast<std::size_t>(i)] = 0.5f * x * (1.f + std::tanh(u));
  }
  return out;
}

Result<Tensor> silu(const Tensor& a) {
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "silu f32"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto n = static_cast<std::int64_t>(pa->size());
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
  for (std::int64_t i = 0; i < n; ++i) {
    float x = (*pa)[static_cast<std::size_t>(i)];
    (*po)[static_cast<std::size_t>(i)] = x / (1.f + std::exp(-x));
  }
  return out;
}

Result<Tensor> swiglu(const Tensor& gate, const Tensor& up) {
  auto ok = same_dev_dtype_shape(gate, up, true);
  if (!ok) return std::unexpected(ok.error());
  auto g = silu(gate);
  if (!g) return g;
  return mul(*g, up);
}

Result<Tensor> softmax_last(const Tensor& a) {
  if (a.dtype() != DType::f32 || a.rank() < 1) {
    return std::unexpected(make_error(Errc::invalid_shape, "softmax_last"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto last = a.shape()[a.rank() - 1];
  const auto rows = a.numel() / last;
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows > 8)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src = pa->data() + r * last;
    float* dst = po->data() + r * last;
    float m = src[0];
    for (std::int64_t i = 1; i < last; ++i) m = std::max(m, src[i]);
    float s = 0;
    for (std::int64_t i = 0; i < last; ++i) {
      dst[i] = std::exp(src[i] - m);
      s += dst[i];
    }
    for (std::int64_t i = 0; i < last; ++i) dst[i] /= s;
  }
  return out;
}

Result<Tensor> layer_norm(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
  if (x.dtype() != DType::f32 || w.dtype() != DType::f32 || b.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "ln f32"));
  }
  if (x.rank() < 1 || w.rank() != 1 || b.rank() != 1 || w.shape()[0] != x.shape()[x.rank() - 1] ||
      b.shape()[0] != w.shape()[0]) {
    return std::unexpected(make_error(Errc::invalid_shape, "ln shape"));
  }
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto px = f32c(x);
  auto pw = f32c(w);
  auto pb = f32c(b);
  auto po = f32(*out);
  if (!px || !pw || !pb || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows > 8)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src = px->data() + r * C;
    float* dst = po->data() + r * C;
    float mean = 0;
    for (std::int64_t i = 0; i < C; ++i) mean += src[i];
    mean /= static_cast<float>(C);
    float var = 0;
    for (std::int64_t i = 0; i < C; ++i) {
      float d = src[i] - mean;
      var += d * d;
    }
    var /= static_cast<float>(C);
    float inv = 1.f / std::sqrt(var + eps);
    for (std::int64_t i = 0; i < C; ++i) {
      dst[i] = (src[i] - mean) * inv * (*pw)[static_cast<std::size_t>(i)] +
               (*pb)[static_cast<std::size_t>(i)];
    }
  }
  return out;
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor& w, float eps) {
  if (x.dtype() != DType::f32 || w.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "rms f32"));
  }
  if (x.rank() < 1 || w.rank() != 1 || w.shape()[0] != x.shape()[x.rank() - 1]) {
    return std::unexpected(make_error(Errc::invalid_shape, "rms shape"));
  }
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto px = f32c(x);
  auto pw = f32c(w);
  auto po = f32(*out);
  if (!px || !pw || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto C = x.shape()[x.rank() - 1];
  const auto rows = x.numel() / C;
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows > 8)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* src = px->data() + r * C;
    float* dst = po->data() + r * C;
    float ms = 0;
    for (std::int64_t i = 0; i < C; ++i) ms += src[i] * src[i];
    ms /= static_cast<float>(C);
    float inv = 1.f / std::sqrt(ms + eps);
    for (std::int64_t i = 0; i < C; ++i) {
      dst[i] = src[i] * inv * (*pw)[static_cast<std::size_t>(i)];
    }
  }
  return out;
}

Result<Tensor> softcap(const Tensor& a, float cap) {
  if (a.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "softcap f32"));
  }
  if (!(cap > 0.f)) {
    return std::unexpected(make_error(Errc::invalid_shape, "softcap cap"));
  }
  auto out = Tensor::empty(a.shape(), DType::f32, a.device());
  if (!out) return out;
  auto pa = f32c(a);
  auto po = f32(*out);
  if (!pa || !po) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto n = static_cast<std::int64_t>(pa->size());
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
  for (std::int64_t i = 0; i < n; ++i) {
    (*po)[static_cast<std::size_t>(i)] = cap * std::tanh((*pa)[static_cast<std::size_t>(i)] / cap);
  }
  return out;
}

float attn_temperature_scale(std::int64_t seq_len, std::int64_t temp_len) noexcept {
  if (temp_len <= 1 || seq_len <= 0) return 1.f;
  const auto m = seq_len > temp_len ? seq_len : temp_len;
  return std::log(static_cast<float>(m)) / std::log(static_cast<float>(temp_len));
}

Result<Tensor> rope(const Tensor& x, const Tensor& positions_i32, float theta, float pos_scale) {
  if (x.dtype() != DType::f32 || positions_i32.dtype() != DType::i32) {
    return std::unexpected(make_error(Errc::dtype_mismatch, "rope dtypes"));
  }
  if (x.rank() < 2) return std::unexpected(make_error(Errc::invalid_shape, "rope rank"));
  const auto D = x.shape()[x.rank() - 1];
  const auto T = x.shape()[x.rank() - 2];
  if (D < 2 || (D % 2) != 0) {
    return std::unexpected(make_error(Errc::invalid_shape, "rope last dim even"));
  }
  if (positions_i32.rank() != 1 ||
      (positions_i32.shape()[0] != T && positions_i32.shape()[0] != 1)) {
    return std::unexpected(make_error(Errc::invalid_shape, "rope positions"));
  }
  if (!(theta > 0.f) || !(pos_scale > 0.f)) {
    return std::unexpected(make_error(Errc::invalid_shape, "rope theta/scale"));
  }
  auto out = Tensor::empty(x.shape(), DType::f32, x.device());
  if (!out) return out;
  auto px = f32c(x);
  auto po = f32(*out);
  auto pp = positions_i32.host_span<std::int32_t>();
  if (!px || !po || !pp) return std::unexpected(make_error(Errc::not_cpu, "host"));
  const auto rows = x.numel() / D;  // includes T in the leading product
  const auto half = D / 2;
  init_threads();
#if defined(GYRE_OPENMP)
#pragma omp parallel for schedule(static) if (rows > 8)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    const auto t = r % T;
    const auto pos_i = (*pp)[static_cast<std::size_t>(pp->size() == 1 ? 0 : t)];
    const float pos = static_cast<float>(pos_i) / pos_scale;
    const float* src = px->data() + r * D;
    float* dst = po->data() + r * D;
    for (std::int64_t i = 0; i < half; ++i) {
      const float freq =
          1.f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(D));
      const float ang = pos * freq;
      const float c = std::cos(ang);
      const float s = std::sin(ang);
      const float x0 = src[2 * i];
      const float x1 = src[2 * i + 1];
      dst[2 * i] = x0 * c - x1 * s;
      dst[2 * i + 1] = x0 * s + x1 * c;
    }
  }
  return out;
}

Result<void> fill_zero(Tensor& t) {
  auto hb = t.host_bytes();
  if (!hb) return std::unexpected(hb.error());
  std::memset(hb->data(), 0, hb->size());
  return {};
}

Result<void> add_(Tensor& dst, const Tensor& src) {
  auto ok = same_dev_dtype_shape(dst, src, true);
  if (!ok) return ok;
  if (dst.dtype() != DType::f32) {
    return std::unexpected(make_error(Errc::unsupported, "add_ f32"));
  }
  auto pd = dst.host_span<float>();
  auto ps = src.host_span<float>();
  if (!pd || !ps) return std::unexpected(pd ? ps.error() : pd.error());
  for (std::size_t i = 0; i < pd->size(); ++i) (*pd)[i] += (*ps)[i];
  return {};
}

}  // namespace gyre
