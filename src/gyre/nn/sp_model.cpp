#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/unigram.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace gyre {
namespace {

struct ProtoReader {
  const std::uint8_t* p;
  const std::uint8_t* end;
  bool ok{true};

  bool remain(std::size_t n) const { return static_cast<std::size_t>(end - p) >= n; }

  std::uint64_t varint() {
    std::uint64_t x = 0;
    int s = 0;
    while (p < end) {
      std::uint8_t b = *p++;
      x |= static_cast<std::uint64_t>(b & 0x7F) << s;
      if ((b & 0x80) == 0) return x;
      s += 7;
      if (s > 63) break;
    }
    ok = false;
    return 0;
  }

  bool skip_value(int wt) {
    if (wt == 0) {
      (void)varint();
      return ok;
    }
    if (wt == 1) {
      if (!remain(8)) return ok = false;
      p += 8;
      return true;
    }
    if (wt == 2) {
      auto n = varint();
      if (!ok || !remain(static_cast<std::size_t>(n))) return ok = false;
      p += static_cast<std::size_t>(n);
      return true;
    }
    if (wt == 5) {
      if (!remain(4)) return ok = false;
      p += 4;
      return true;
    }
    return ok = false;
  }
};

float decode_f32(const std::uint8_t* b) {
  float f;
  std::memcpy(&f, b, 4);
  return f;
}

}  // namespace

Result<std::unique_ptr<Tokenizer>> Tokenizer::load_sentencepiece(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected(make_error(Errc::io, path.string()));
  std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  ProtoReader r{buf.data(), buf.data() + buf.size()};
  PieceTable table;
  std::vector<float> scores;
  int model_type = 1;  // UNIGRAM
  while (r.ok && r.p < r.end) {
    auto tag = r.varint();
    if (!r.ok) break;
    int fn = static_cast<int>(tag >> 3);
    int wt = static_cast<int>(tag & 7);
    if (fn == 1 && wt == 2) {  // pieces
      auto n = r.varint();
      if (!r.ok || !r.remain(static_cast<std::size_t>(n)))
        return std::unexpected(make_error(Errc::ckpt_corrupt, "sp piece"));
      auto* sub_end = r.p + static_cast<std::size_t>(n);
      ProtoReader s{r.p, sub_end};
      r.p = sub_end;
      std::string piece;
      float score = 0;
      while (s.ok && s.p < s.end) {
        auto t2 = s.varint();
        int f2 = static_cast<int>(t2 >> 3);
        int w2 = static_cast<int>(t2 & 7);
        if (f2 == 1 && w2 == 2) {
          auto pn = s.varint();
          if (!s.ok || !s.remain(static_cast<std::size_t>(pn))) break;
          piece.assign(reinterpret_cast<const char*>(s.p), static_cast<std::size_t>(pn));
          s.p += static_cast<std::size_t>(pn);
        } else if (f2 == 2 && w2 == 5) {
          if (!s.remain(4)) break;
          score = decode_f32(s.p);
          s.p += 4;
        } else {
          if (!s.skip_value(w2)) break;
        }
      }
      table.pieces.push_back(std::move(piece));
      scores.push_back(score);
    } else if (fn == 2 && wt == 2) {  // trainer_spec
      auto n = r.varint();
      if (!r.ok || !r.remain(static_cast<std::size_t>(n))) break;
      auto* sub_end = r.p + static_cast<std::size_t>(n);
      ProtoReader s{r.p, sub_end};
      r.p = sub_end;
      while (s.ok && s.p < s.end) {
        auto t2 = s.varint();
        int f2 = static_cast<int>(t2 >> 3);
        int w2 = static_cast<int>(t2 & 7);
        if (f2 == 3 && w2 == 0) {
          model_type = static_cast<int>(s.varint());
        } else {
          if (!s.skip_value(w2)) break;
        }
      }
    } else {
      if (!r.skip_value(wt)) break;
    }
  }
  if (table.pieces.empty())
    return std::unexpected(make_error(Errc::ckpt_corrupt, "empty sentencepiece"));
  if (model_type == 2)
    return std::unexpected(make_error(Errc::unsupported, "sentencepiece BPE not imported"));
  if (model_type == 3)
    return std::unexpected(make_error(Errc::unsupported, "sentencepiece WORD"));
  auto um = UnigramModel::from_scores(std::move(scores), true);
  if (!um) return std::unexpected(um.error());
  return std::make_unique<Tokenizer>(make_metaspace_pretok(true), std::move(*um), std::move(table));
}

}  // namespace gyre
