// ONNX export parity: parse the exported protobuf, run it with a tiny reference
// interpreter (only the ops the CharLM exporter emits), and compare the logits
// with CharLM::forward on CPU. Regression for the exporter computing x*tanh(x)
// instead of tanh-GELU and dropping the ALiBi recency bias.

#include "gyre/export/onnx.hpp"
#include "gyre/module.hpp"
#include "gyre/nn/transformer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- minimal protobuf reader ---------------------------------------------

struct Reader {
  const std::uint8_t* p;
  const std::uint8_t* end;
  bool done() const { return p >= end; }
  std::uint64_t varint() {
    std::uint64_t v = 0;
    for (int s = 0; p < end; s += 7) {
      const auto b = *p++;
      v |= static_cast<std::uint64_t>(b & 127) << s;
      if (!(b & 128)) return v;
    }
    throw std::runtime_error("varint");
  }
  // Returns field number and wire type; payload left at p.
  std::pair<int, int> key() {
    auto k = varint();
    return {static_cast<int>(k >> 3), static_cast<int>(k & 7)};
  }
  Reader ld() {
    auto n = varint();
    if (n > static_cast<std::uint64_t>(end - p)) throw std::runtime_error("ld");
    Reader r{p, p + n};
    p += n;
    return r;
  }
  std::string str() {
    auto r = ld();
    return std::string(reinterpret_cast<const char*>(r.p), static_cast<std::size_t>(r.end - r.p));
  }
  std::uint32_t fixed32() {
    if (end - p < 4) throw std::runtime_error("fixed32");
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
  }
  void skip(int wire) {
    if (wire == 0) varint();
    else if (wire == 2) ld();
    else if (wire == 5) fixed32();
    else if (wire == 1) p += 8;
    else throw std::runtime_error("wire");
  }
};

struct T {
  std::vector<std::int64_t> shape;
  std::vector<float> f;
  std::vector<std::int64_t> i;  // int64 tensors
  bool is_int{false};
  std::size_t numel() const {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
  }
};

struct Attr {
  std::int64_t i{0};
  float f{0};
  std::vector<std::int64_t> ints;
};

struct Node {
  std::string op;
  std::vector<std::string> in, out;
  std::map<std::string, Attr> attrs;
};

T parse_tensor(Reader r, std::string* name) {
  T t;
  int dtype = 1;
  std::string raw;
  while (!r.done()) {
    auto [f, w] = r.key();
    if (f == 1 && w == 0) t.shape.push_back(static_cast<std::int64_t>(r.varint()));
    else if (f == 2 && w == 0) dtype = static_cast<int>(r.varint());
    else if (f == 8 && w == 2) *name = r.str();
    else if (f == 9 && w == 2) raw = r.str();
    else r.skip(w);
  }
  t.is_int = dtype == 7;
  // Real runtimes reject an initializer whose raw_data does not match its dims.
  if (raw.size() != t.numel() * (t.is_int ? 8u : 4u)) throw std::runtime_error("raw_data size " + *name);
  if (t.is_int) {
    t.i.resize(raw.size() / 8);
    std::memcpy(t.i.data(), raw.data(), raw.size());
  } else {
    t.f.resize(raw.size() / 4);
    std::memcpy(t.f.data(), raw.data(), raw.size());
  }
  return t;
}

Node parse_node(Reader r) {
  Node n;
  while (!r.done()) {
    auto [f, w] = r.key();
    if (f == 1 && w == 2) n.in.push_back(r.str());
    else if (f == 2 && w == 2) n.out.push_back(r.str());
    else if (f == 4 && w == 2) n.op = r.str();
    else if (f == 5 && w == 2) {
      auto a = r.ld();
      std::string an;
      Attr at;
      while (!a.done()) {
        auto [af, aw] = a.key();
        if (af == 1 && aw == 2) an = a.str();
        else if (af == 3 && aw == 0) at.i = static_cast<std::int64_t>(a.varint());
        else if (af == 4 && aw == 5) {
          auto u = a.fixed32();
          std::memcpy(&at.f, &u, 4);
        } else if (af == 8 && aw == 0) at.ints.push_back(static_cast<std::int64_t>(a.varint()));
        else a.skip(aw);
      }
      n.attrs[an] = at;
    } else r.skip(w);
  }
  return n;
}

// ---- reference ops ---------------------------------------------------------

std::vector<std::int64_t> strides(const std::vector<std::int64_t>& s) {
  std::vector<std::int64_t> st(s.size(), 1);
  for (int k = static_cast<int>(s.size()) - 2; k >= 0; --k) st[k] = st[k + 1] * s[k + 1];
  return st;
}

T broadcast(const T& a, const T& b, float (*op)(float, float)) {
  const auto r = std::max(a.shape.size(), b.shape.size());
  auto pad = [&](const std::vector<std::int64_t>& s) {
    std::vector<std::int64_t> o(r - s.size(), 1);
    o.insert(o.end(), s.begin(), s.end());
    return o;
  };
  auto sa = pad(a.shape), sb = pad(b.shape);
  T out;
  out.shape.resize(r);
  for (std::size_t k = 0; k < r; ++k) out.shape[k] = std::max(sa[k], sb[k]);
  auto sta = strides(sa), stb = strides(sb), sto = strides(out.shape);
  out.f.resize(out.numel());
  for (std::size_t idx = 0; idx < out.f.size(); ++idx) {
    std::int64_t ia = 0, ib = 0, rem = static_cast<std::int64_t>(idx);
    for (std::size_t k = 0; k < r; ++k) {
      const auto c = rem / sto[k];
      rem %= sto[k];
      if (sa[k] != 1) ia += c * sta[k];
      if (sb[k] != 1) ib += c * stb[k];
    }
    out.f[idx] = op(a.f[static_cast<std::size_t>(ia)], b.f[static_cast<std::size_t>(ib)]);
  }
  return out;
}

T matmul(const T& a, const T& b) {
  const auto M = a.shape[a.shape.size() - 2], K = a.shape.back(), N = b.shape.back();
  const std::int64_t batch = static_cast<std::int64_t>(a.numel()) / (M * K);
  const bool b2 = b.shape.size() == 2;
  T out;
  out.shape = a.shape;
  out.shape.back() = N;
  out.f.assign(static_cast<std::size_t>(batch * M * N), 0.f);
  for (std::int64_t z = 0; z < batch; ++z) {
    const float* pa = a.f.data() + z * M * K;
    const float* pb = b.f.data() + (b2 ? 0 : z * K * N);
    float* po = out.f.data() + z * M * N;
    for (std::int64_t m = 0; m < M; ++m)
      for (std::int64_t n = 0; n < N; ++n) {
        float s = 0;
        for (std::int64_t k = 0; k < K; ++k) s += pa[m * K + k] * pb[k * N + n];
        po[m * N + n] = s;
      }
  }
  return out;
}

T transpose(const T& a, const std::vector<std::int64_t>& perm) {
  T out;
  out.shape.resize(perm.size());
  for (std::size_t k = 0; k < perm.size(); ++k) out.shape[k] = a.shape[static_cast<std::size_t>(perm[k])];
  auto sa = strides(a.shape), so = strides(out.shape);
  out.f.resize(a.f.size());
  for (std::size_t idx = 0; idx < out.f.size(); ++idx) {
    std::int64_t rem = static_cast<std::int64_t>(idx), ia = 0;
    for (std::size_t k = 0; k < perm.size(); ++k) {
      const auto c = rem / so[k];
      rem %= so[k];
      ia += c * sa[static_cast<std::size_t>(perm[k])];
    }
    out.f[idx] = a.f[static_cast<std::size_t>(ia)];
  }
  return out;
}

T run(const std::vector<Node>& nodes, std::map<std::string, T> env) {
  for (auto& n : nodes) {
    auto in = [&](std::size_t k) -> const T& { return env.at(n.in[k]); };
    T o;
    if (n.op == "Gather") {
      const auto& d = in(0);
      const auto& ix = in(1);
      const auto C = d.shape[1];
      o.shape = ix.shape;
      o.shape.push_back(C);
      for (auto id : ix.i)
        o.f.insert(o.f.end(), d.f.begin() + id * C, d.f.begin() + (id + 1) * C);
    } else if (n.op == "Add") {
      o = broadcast(in(0), in(1), [](float x, float y) { return x + y; });
    } else if (n.op == "Mul") {
      o = broadcast(in(0), in(1), [](float x, float y) { return x * y; });
    } else if (n.op == "MatMul") {
      o = matmul(in(0), in(1));
    } else if (n.op == "LayerNormalization") {
      const auto& x = in(0);
      const auto& w = in(1);
      const auto& b = in(2);
      const float eps = n.attrs.at("epsilon").f;
      const auto C = static_cast<std::size_t>(x.shape.back());
      o = x;
      for (std::size_t r = 0; r < x.f.size() / C; ++r) {
        float mean = 0, var = 0;
        for (std::size_t c = 0; c < C; ++c) mean += x.f[r * C + c];
        mean /= static_cast<float>(C);
        for (std::size_t c = 0; c < C; ++c) var += (x.f[r * C + c] - mean) * (x.f[r * C + c] - mean);
        var /= static_cast<float>(C);
        const float inv = 1.f / std::sqrt(var + eps);
        for (std::size_t c = 0; c < C; ++c) o.f[r * C + c] = (x.f[r * C + c] - mean) * inv * w.f[c] + b.f[c];
      }
    } else if (n.op == "Reshape") {
      o = in(0);
      o.shape = in(1).i;
    } else if (n.op == "Transpose") {
      o = transpose(in(0), n.attrs.at("perm").ints);
    } else if (n.op == "Softmax") {
      o = in(0);
      const auto C = static_cast<std::size_t>(o.shape.back());
      for (std::size_t r = 0; r < o.f.size() / C; ++r) {
        float mx = -INFINITY, s = 0;
        for (std::size_t c = 0; c < C; ++c) mx = std::max(mx, o.f[r * C + c]);
        for (std::size_t c = 0; c < C; ++c) s += (o.f[r * C + c] = std::exp(o.f[r * C + c] - mx));
        for (std::size_t c = 0; c < C; ++c) o.f[r * C + c] /= s;
      }
    } else if (n.op == "Tanh") {
      o = in(0);
      for (auto& v : o.f) v = std::tanh(v);
    } else if (n.op == "Clip") {
      o = in(0);
      const float lo = in(1).f.at(0), hi = in(2).f.at(0);
      for (auto& v : o.f) v = std::clamp(v, lo, hi);
    } else {
      throw std::runtime_error("unsupported op " + n.op);
    }
    env[n.out.at(0)] = std::move(o);
  }
  return env.at("logits");
}

void check_parity(bool alibi) {
  auto cpu = *gyre::Device::cpu();
  gyre::CharLMConfig c;
  c.vocab = 11;
  c.block_size = 8;
  c.n_layer = 2;
  c.n_head = 2;
  c.d_model = 16;
  c.d_ff = 32;
  c.recency_alibi = alibi;
  gyre::Rng rng(9);
  auto m = gyre::CharLM::create(c, cpu, rng);
  ASSERT_TRUE(m) << m.error().message;
  // Larger weights push fc1 into the range where GELU and x*tanh(x) differ a lot.
  for (auto& p : m->parameters()) {
    auto s = p.value.host_span<float>();
    ASSERT_TRUE(s);
    if (p.value.rank() >= 2) for (auto& v : *s) v *= 10.f;
  }
  auto path = std::filesystem::temp_directory_path() / (alibi ? "gyre-parity-alibi.onnx" : "gyre-parity.onnx");
  auto ex = gyre::export_charlm_onnx(*m, path);
  ASSERT_TRUE(ex) << ex.error().message;

  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  Reader model{bytes.data(), bytes.data() + bytes.size()};
  std::vector<Node> nodes;
  std::map<std::string, T> env;
  try {
  while (!model.done()) {
    auto [f, w] = model.key();
    if (f != 7 || w != 2) {
      model.skip(w);
      continue;
    }
    auto g = model.ld();
    while (!g.done()) {
      auto [gf, gw] = g.key();
      if (gf == 1 && gw == 2) nodes.push_back(parse_node(g.ld()));
      else if (gf == 5 && gw == 2) {
        std::string name;
        auto t = parse_tensor(g.ld(), &name);
        env[name] = std::move(t);
      } else g.skip(gw);
    }
  }
  } catch (const std::exception& e) {
    FAIL() << "parse: " << e.what();
  }
  ASSERT_FALSE(nodes.empty());

  std::vector<std::int32_t> ids = {1, 4, 2, 9, 0, 3, 10, 5};
  T tok;
  tok.is_int = true;
  tok.shape = {1, 8};
  tok.i.assign(ids.begin(), ids.end());
  env["tokens"] = tok;
  T got;
  try {
    got = run(nodes, env);
  } catch (const std::exception& e) {
    FAIL() << "reference run: " << e.what();
  }

  std::int64_t sh[] = {1, 8};
  auto x = gyre::Tensor::from_host(std::as_bytes(std::span(ids)), sh, gyre::DType::i32, cpu);
  ASSERT_TRUE(x);
  gyre::ForwardCtx ctx;
  ctx.train = false;
  auto want = m->forward(*x, ctx);
  ASSERT_TRUE(want) << want.error().message;
  auto ws = want->host_span<float>();
  ASSERT_TRUE(ws);
  ASSERT_EQ(got.f.size(), ws->size());
  {
    // Reference logits next to the .onnx, so the file can also be checked with
    // onnxruntime by hand (same tokens).
    std::ofstream lo(path.string() + ".logits.f32", std::ios::binary | std::ios::trunc);
    lo.write(reinterpret_cast<const char*>(ws->data()), static_cast<std::streamsize>(ws->size() * 4));
  }
  float worst = 0.f;
  for (std::size_t i = 0; i < ws->size(); ++i) {
    worst = std::max(worst, std::abs(got.f[i] - (*ws)[i]) / std::max(1.f, std::abs((*ws)[i])));
  }
  EXPECT_LT(worst, 1e-3f) << "onnx logits diverge from CharLM (alibi=" << alibi << ")";
}

}  // namespace

TEST(OnnxExport, MatchesCharLMNoRecency) { check_parity(false); }

TEST(OnnxExport, MatchesCharLMAlibi) { check_parity(true); }
