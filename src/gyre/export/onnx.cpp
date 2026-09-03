#include "gyre/export/onnx.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gyre {
namespace {

using u8 = std::uint8_t;
using Buf = std::vector<u8>;

void varint(Buf& o, std::uint64_t v) {
  while (v > 127) {
    o.push_back(static_cast<u8>((v & 127) | 128));
    v >>= 7;
  }
  o.push_back(static_cast<u8>(v));
}

void tag(Buf& o, int field, int wire) { varint(o, (static_cast<std::uint64_t>(field) << 3) | wire); }

void f_varint(Buf& o, int field, std::uint64_t v) {
  tag(o, field, 0);
  varint(o, v);
}

void f_ld(Buf& o, int field, std::span<const u8> s) {
  tag(o, field, 2);
  varint(o, s.size());
  o.insert(o.end(), s.begin(), s.end());
}

void f_str(Buf& o, int field, std::string_view s) {
  f_ld(o, field, std::span<const u8>(reinterpret_cast<const u8*>(s.data()), s.size()));
}

Buf tensor_f32(std::string_view name, std::span<const std::int64_t> dims, std::span<const float> data) {
  Buf t;
  for (auto d : dims) f_varint(t, 1, static_cast<std::uint64_t>(d));
  f_varint(t, 2, 1);  // FLOAT
  f_str(t, 8, name);
  f_ld(t, 9, std::span<const u8>(reinterpret_cast<const u8*>(data.data()), data.size() * 4));
  return t;
}

Buf tensor_i64(std::string_view name, std::span<const std::int64_t> dims, std::span<const std::int64_t> data) {
  Buf t;
  for (auto d : dims) f_varint(t, 1, static_cast<std::uint64_t>(d));
  f_varint(t, 2, 7);  // INT64
  f_str(t, 8, name);
  f_ld(t, 9, std::span<const u8>(reinterpret_cast<const u8*>(data.data()), data.size() * 8));
  return t;
}

Buf value_info(std::string_view name, int elem, std::span<const std::int64_t> dims) {
  Buf shape;
  for (auto d : dims) {
    Buf dim;
    f_varint(dim, 1, static_cast<std::uint64_t>(d));
    f_ld(shape, 1, dim);
  }
  Buf tensor;
  f_varint(tensor, 1, static_cast<std::uint64_t>(elem));
  f_ld(tensor, 2, shape);
  Buf type;
  f_ld(type, 1, tensor);
  Buf vi;
  f_str(vi, 1, name);
  f_ld(vi, 2, type);
  return vi;
}

Buf attr_i(std::string_view name, std::int64_t v) {
  Buf a;
  f_str(a, 1, name);
  f_varint(a, 3, static_cast<std::uint64_t>(v));
  f_varint(a, 20, 2);  // INT
  return a;
}

Buf attr_f(std::string_view name, float v) {
  Buf a;
  f_str(a, 1, name);
  tag(a, 4, 5);  // 32-bit
  u8 b[4];
  std::memcpy(b, &v, 4);
  a.insert(a.end(), b, b + 4);
  f_varint(a, 20, 1);  // FLOAT
  return a;
}

Buf attr_ints(std::string_view name, std::span<const std::int64_t> vs) {
  Buf a;
  f_str(a, 1, name);
  for (auto v : vs) f_varint(a, 8, static_cast<std::uint64_t>(v));
  f_varint(a, 20, 7);  // INTS
  return a;
}

Buf node(std::string_view op, std::span<const std::string_view> ins, std::span<const std::string_view> outs,
         std::span<const Buf> attrs, std::string_view nm) {
  Buf n;
  for (auto s : ins) f_str(n, 1, s);
  for (auto s : outs) f_str(n, 2, s);
  f_str(n, 3, nm);
  f_str(n, 4, op);
  for (auto& a : attrs) f_ld(n, 5, a);
  return n;
}

std::string p(int i) { return "p" + std::to_string(i); }

}  // namespace

Result<void> export_charlm_onnx(const CharLM& model, const std::filesystem::path& path) {
  const auto& cfg = model.config();
  auto params = const_cast<CharLM&>(model).parameters();
  const auto T = cfg.block_size;
  const auto d = cfg.d_model;
  const auto H = cfg.n_head;
  const auto dk = d / H;
  const auto V = cfg.vocab;
  if (d % H != 0) return std::unexpected(make_error(Errc::invalid_shape, "d_model % n_head"));

  std::vector<Buf> inits;
  std::vector<Buf> nodes;
  std::vector<std::string> pool;
  pool.reserve(4096);
  auto S = [&](std::string s) -> std::string_view {
    pool.push_back(std::move(s));
    return pool.back();
  };
  int nid = 0;
  auto add_n = [&](std::string_view op, std::vector<std::string_view> ins, std::vector<std::string_view> outs,
                   std::vector<Buf> attrs) {
    auto nn = S("n" + std::to_string(nid++));
    nodes.push_back(node(op, ins, outs, attrs, nn));
  };
  std::vector<std::string_view> pname(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) pname[i] = S(p(static_cast<int>(i)));

  for (std::size_t i = 0; i < params.size(); ++i) {
    auto sp = params[i].value.host_span<float>();
    if (!sp) return std::unexpected(sp.error());
    std::vector<std::int64_t> dims(params[i].value.shape().begin(), params[i].value.shape().end());
    if (dims.empty()) dims.push_back(1);
    inits.push_back(tensor_f32(p(static_cast<int>(i)), dims, *sp));
  }

  std::vector<std::int64_t> pos(static_cast<std::size_t>(T));
  for (std::int64_t i = 0; i < T; ++i) pos[static_cast<std::size_t>(i)] = i;
  std::int64_t pos_dims[] = {T};
  inits.push_back(tensor_i64("pos_ids", pos_dims, pos));

  std::vector<float> mask_data(static_cast<std::size_t>(T * T), 0.f);
  for (std::int64_t i = 0; i < T; ++i)
    for (std::int64_t j = 0; j < T; ++j)
      if (j > i) mask_data[static_cast<std::size_t>(i * T + j)] = -1e9f;
  std::int64_t md[] = {1, 1, T, T};
  inits.push_back(tensor_f32("causal_mask", md, mask_data));

  float scale = 1.f / std::sqrt(static_cast<float>(dk));
  float scv[] = {scale};
  std::int64_t scd[] = {1};
  inits.push_back(tensor_f32("attn_scale", scd, scv));

  std::int64_t qshape[] = {1, T, H, dk};
  inits.push_back(tensor_i64("shape_bthd", std::span<const std::int64_t>(qshape, 4),
                             std::span<const std::int64_t>(qshape, 4)));
  std::int64_t oshape[] = {1, T, d};
  inits.push_back(tensor_i64("shape_btc", std::span<const std::int64_t>(oshape, 3),
                             std::span<const std::int64_t>(oshape, 3)));

  auto tokens = S("tokens");
  auto tok = S("tok");
  auto pe = S("pe");
  auto h = S("h");
  auto posn = S("pos_ids");
  auto shp_bthd = S("shape_bthd");
  auto shp_btc = S("shape_btc");
  auto mask_n = S("causal_mask");
  auto ascale = S("attn_scale");
  add_n("Gather", {pname[0], tokens}, {tok}, {attr_i("axis", 0)});
  add_n("Gather", {pname[1], posn}, {pe}, {attr_i("axis", 0)});
  add_n("Add", {tok, pe}, {h}, std::vector<Buf>{});

  std::int64_t perm_bthd[] = {0, 2, 1, 3};
  std::int64_t perm_last[] = {0, 1, 3, 2};
  int pi = 2;
  for (std::int64_t layer = 0; layer < cfg.n_layer; ++layer) {
    auto L = std::to_string(layer);
    auto ln1 = S("ln1_" + L);
    add_n("LayerNormalization", {h, pname[pi], pname[pi + 1]}, {ln1},
          {attr_i("axis", -1), attr_f("epsilon", 1e-5f)});
    auto qm = S("qm_" + L), qv = S("q_" + L);
    auto km = S("km_" + L), kv = S("k_" + L);
    auto vm = S("vm_" + L), vv = S("v_" + L);
    add_n("MatMul", {ln1, pname[pi + 2]}, {qm}, std::vector<Buf>{});
    add_n("Add", {qm, pname[pi + 3]}, {qv}, std::vector<Buf>{});
    add_n("MatMul", {ln1, pname[pi + 4]}, {km}, std::vector<Buf>{});
    add_n("Add", {km, pname[pi + 5]}, {kv}, std::vector<Buf>{});
    add_n("MatMul", {ln1, pname[pi + 6]}, {vm}, std::vector<Buf>{});
    add_n("Add", {vm, pname[pi + 7]}, {vv}, std::vector<Buf>{});
    auto qr = S("qr_" + L), qh = S("qh_" + L);
    auto kr = S("kr_" + L), kh = S("kh_" + L);
    auto vr = S("vr_" + L), vh = S("vh_" + L);
    add_n("Reshape", {qv, shp_bthd}, {qr}, std::vector<Buf>{});
    add_n("Transpose", {qr}, {qh}, {attr_ints("perm", perm_bthd)});
    add_n("Reshape", {kv, shp_bthd}, {kr}, std::vector<Buf>{});
    add_n("Transpose", {kr}, {kh}, {attr_ints("perm", perm_bthd)});
    add_n("Reshape", {vv, shp_bthd}, {vr}, std::vector<Buf>{});
    add_n("Transpose", {vr}, {vh}, {attr_ints("perm", perm_bthd)});
    auto kt = S("kt_" + L);
    add_n("Transpose", {kh}, {kt}, {attr_ints("perm", perm_last)});
    auto sc = S("sc_" + L), ss = S("ss_" + L), sm = S("sm_" + L), w = S("w_" + L);
    add_n("MatMul", {qh, kt}, {sc}, std::vector<Buf>{});
    add_n("Mul", {sc, ascale}, {ss}, std::vector<Buf>{});
    add_n("Add", {ss, mask_n}, {sm}, std::vector<Buf>{});
    add_n("Softmax", {sm}, {w}, {attr_i("axis", -1)});
    auto y = S("y_" + L), yt = S("yt_" + L), yo = S("yo_" + L);
    add_n("MatMul", {w, vh}, {y}, std::vector<Buf>{});
    add_n("Transpose", {y}, {yt}, {attr_ints("perm", perm_bthd)});
    add_n("Reshape", {yt, shp_btc}, {yo}, std::vector<Buf>{});
    auto pm = S("pm_" + L), po = S("po_" + L);
    add_n("MatMul", {yo, pname[pi + 8]}, {pm}, std::vector<Buf>{});
    add_n("Add", {pm, pname[pi + 9]}, {po}, std::vector<Buf>{});
    auto h1 = S("h1_" + L);
    add_n("Add", {h, po}, {h1}, std::vector<Buf>{});
    auto ln2 = S("ln2_" + L);
    add_n("LayerNormalization", {h1, pname[pi + 10], pname[pi + 11]}, {ln2},
          {attr_i("axis", -1), attr_f("epsilon", 1e-5f)});
    auto fc1m = S("fc1m_" + L), fc1 = S("fc1_" + L), gth = S("gth_" + L), ge = S("ge_" + L);
    add_n("MatMul", {ln2, pname[pi + 12]}, {fc1m}, std::vector<Buf>{});
    add_n("Add", {fc1m, pname[pi + 13]}, {fc1}, std::vector<Buf>{});
    add_n("Tanh", {fc1}, {gth}, std::vector<Buf>{});
    add_n("Mul", {fc1, gth}, {ge}, std::vector<Buf>{});
    auto fc2m = S("fc2m_" + L), fc2 = S("fc2_" + L);
    add_n("MatMul", {ge, pname[pi + 14]}, {fc2m}, std::vector<Buf>{});
    add_n("Add", {fc2m, pname[pi + 15]}, {fc2}, std::vector<Buf>{});
    auto h2 = S("h2_" + L);
    add_n("Add", {h1, fc2}, {h2}, std::vector<Buf>{});
    h = h2;
    pi += 16;
  }
  auto lnf = S("lnf"), hm = S("hm"), logits = S("logits");
  add_n("LayerNormalization", {h, pname[pi], pname[pi + 1]}, {lnf},
        {attr_i("axis", -1), attr_f("epsilon", 1e-5f)});
  add_n("MatMul", {lnf, pname[pi + 2]}, {hm}, std::vector<Buf>{});
  add_n("Add", {hm, pname[pi + 3]}, {logits}, std::vector<Buf>{});

  std::int64_t in_dims[] = {1, T};
  std::int64_t out_dims[] = {1, T, V};
  auto vin = value_info("tokens", 7, in_dims);
  auto vout = value_info("logits", 1, out_dims);

  Buf graph;
  for (auto& n : nodes) f_ld(graph, 1, n);
  f_str(graph, 2, "charlm");
  for (auto& t : inits) f_ld(graph, 5, t);
  f_ld(graph, 11, vin);
  f_ld(graph, 12, vout);

  Buf opset;
  f_varint(opset, 2, 17);

  Buf model_bytes;
  f_varint(model_bytes, 1, 8);  // ir_version
  f_str(model_bytes, 2, "gyre");
  f_str(model_bytes, 3, "0.1.5");
  f_ld(model_bytes, 7, graph);
  f_ld(model_bytes, 8, opset);

  std::ofstream out(path, std::ios::binary);
  if (!out) return std::unexpected(make_error(Errc::io, "onnx open"));
  out.write(reinterpret_cast<const char*>(model_bytes.data()),
            static_cast<std::streamsize>(model_bytes.size()));
  if (!out) return std::unexpected(make_error(Errc::io, "onnx write"));
  return {};
}

}  // namespace gyre
