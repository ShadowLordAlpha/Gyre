// Links ONLY gyre-infer (see CMakeLists.txt). Builds a tiny GPOL1 buffer by hand, loads it,
// runs decisions, and checks the corruption guards. Exit code 0 = pass.
#include "gyre_infer/policy.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

std::uint32_t fnv1a(const unsigned char* p, std::size_t n) {
  std::uint32_t h = 2166136261u;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 16777619u;
  return h;
}

template <class T>
void put(std::vector<unsigned char>& b, T v) {
  const auto* p = reinterpret_cast<const unsigned char*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}

// feat 2, global 1, N 4, d 4, 1 layer, 2 heads, ff 8, out 6, no GRU, heads {3,2}
std::vector<unsigned char> make_buffer() {
  const std::uint64_t F = 2, G = 1, d = 4, ff = 8, O = 6, T = 5;
  std::uint64_t n = F * d + d;
  n += 4 * d + 4 * (d * d + d) + (d * ff + ff) + (ff * d + d);
  n += 2 * d + (2 * d + G) * O + O;
  n += O * T + T + O + 1;
  std::vector<float> w(n);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.05f * std::sin(0.7f * float(i));
  std::vector<unsigned char> b;
  const char magic[8] = {'G', 'P', 'O', 'L', '1', 0, 0, 0};
  b.insert(b.end(), magic, magic + 8);
  put<std::uint32_t>(b, 1);
  put<std::uint32_t>(b, 0);
  for (std::uint32_t v : {2u, 1u, 4u, 4u, 1u, 2u, 8u, 6u, 0u, 2u}) put<std::uint32_t>(b, v);
  put<std::uint32_t>(b, 3);
  put<std::uint32_t>(b, 2);
  put<std::uint64_t>(b, n);
  const auto* wp = reinterpret_cast<const unsigned char*>(w.data());
  b.insert(b.end(), wp, wp + n * 4);
  put<std::uint32_t>(b, fnv1a(wp, n * 4));
  return b;
}

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  auto buf = make_buffer();
  std::string err;
  auto pol = gyre_infer::Policy::load_from_memory(buf.data(), buf.size(), &err);
  check(pol != nullptr, "load valid buffer");
  if (!pol) return 1;
  check(pol->info().total_logits() == 5, "total logits");
  auto s = pol->make_session();
  float ent[8] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f};
  std::uint8_t mask[4] = {1, 1, 0, 1};
  float glob[1] = {0.5f};
  gyre_infer::Input in;
  in.entities = ent;
  in.mask = mask;
  in.n_slots = 4;
  in.global = glob;
  float logits[5], logits2[5], value = 0, value2 = 0;
  check(pol->step(s, in, logits, &value), "step");
  check(pol->step(s, in, logits2, &value2), "step again");
  check(std::memcmp(logits, logits2, sizeof logits) == 0 && value == value2, "deterministic, stateless when feedforward");
  for (float v : logits) check(std::isfinite(v), "finite logits");
  in.n_slots = 5;
  check(!pol->step(s, in, logits, &value), "n_slots > max rejected");
  in.n_slots = 0;
  check(pol->step(s, in, logits, &value), "empty set is allowed");
  check(gyre_infer::pick_masked(logits, nullptr, 3, -1.f) >= 0, "pick");

  auto bad = buf;
  bad[bad.size() / 2] ^= 0x40;
  check(gyre_infer::Policy::load_from_memory(bad.data(), bad.size(), &err) == nullptr, "corrupt payload rejected");
  auto trunc = buf;
  trunc.resize(trunc.size() - 9);
  check(gyre_infer::Policy::load_from_memory(trunc.data(), trunc.size(), &err) == nullptr, "truncated rejected");
  auto magic = buf;
  magic[0] = 'X';
  check(gyre_infer::Policy::load_from_memory(magic.data(), magic.size(), &err) == nullptr, "bad magic rejected");
  if (failures == 0) std::printf("gyre-infer standalone OK\n");
  return failures == 0 ? 0 : 1;
}
