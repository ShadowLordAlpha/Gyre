#include "gyre/nn/tiktoken.hpp"

#include <algorithm>
#include <cctype>
#include <limits>

namespace gyre {
namespace {

#include "unicode_cat.inc"

bool in_ranges(const std::uint32_t tbl[][2], int n, std::uint32_t cp) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (cp < tbl[mid][0]) hi = mid - 1;
    else if (cp > tbl[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_L(std::uint32_t cp) { return in_ranges(kLuL, kLuL_n, cp); }
bool is_N(std::uint32_t cp) { return in_ranges(kNuN, kNuN_n, cp); }
bool is_S(std::uint32_t cp) { return in_ranges(kSp, kSp_n, cp); }

bool next_cp(std::string_view s, std::size_t& i, std::uint32_t& cp, std::size_t& nbytes) {
  if (i >= s.size()) return false;
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    cp = b0;
    nbytes = 1;
    return true;
  }
  if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
    cp = (static_cast<std::uint32_t>(b0 & 0x1F) << 6) |
         (static_cast<unsigned char>(s[i + 1]) & 0x3F);
    nbytes = 2;
    return true;
  }
  if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
    cp = (static_cast<std::uint32_t>(b0 & 0x0F) << 12) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
         (static_cast<unsigned char>(s[i + 2]) & 0x3F);
    nbytes = 3;
    return true;
  }
  if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
    cp = (static_cast<std::uint32_t>(b0 & 0x07) << 18) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
         (static_cast<unsigned char>(s[i + 3]) & 0x3F);
    nbytes = 4;
    return true;
  }
  cp = b0;
  nbytes = 1;
  return true;
}

char ascii_lower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool match_contraction(std::string_view s, std::size_t i, std::size_t& n) {
  if (i >= s.size() || s[i] != '\'') return false;
  if (i + 1 >= s.size()) return false;
  char a = ascii_lower(s[i + 1]);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') {
    n = 2;
    return true;
  }
  if (i + 2 < s.size()) {
    char b = ascii_lower(s[i + 2]);
    if ((a == 'r' || a == 'v') && b == 'e') {
      n = 3;
      return true;
    }
    if (a == 'l' && b == 'l') {
      n = 3;
      return true;
    }
  }
  return false;
}

std::size_t skip_letters(std::string_view s, std::size_t i) {
  while (i < s.size()) {
    std::uint32_t cp{};
    std::size_t n{};
    if (!next_cp(s, i, cp, n)) break;
    if (!is_L(cp)) break;
    i += n;
  }
  return i;
}

bool not_crlf_LN(std::uint32_t cp) {
  if (cp == '\r' || cp == '\n') return false;
  if (is_L(cp) || is_N(cp)) return false;
  return true;
}

bool not_sLN(std::uint32_t cp) {
  if (is_S(cp) || is_L(cp) || is_N(cp)) return false;
  return true;
}

std::size_t special_len_at(std::string_view s, std::size_t i,
                           const std::vector<std::string>& specials) {
  for (const auto& sp : specials) {
    if (!sp.empty() && i + sp.size() <= s.size() && s.compare(i, sp.size(), sp) == 0)
      return sp.size();
  }
  return 0;
}

// PAT_STR_B at position i; returns end offset or i if no match.
std::size_t match_pat_v1(std::string_view s, std::size_t i,
                         const std::vector<std::string>& specials) {
  if (i >= s.size()) return i;
  if (special_len_at(s, i, specials)) return i;  // caller emits the special
  std::size_t n = 0;
  if (match_contraction(s, i, n)) return i + n;

  std::uint32_t cp{};
  std::size_t nb{};
  if (!next_cp(s, i, cp, nb)) return i;

  // [^\r\n\p{L}\p{N}]?\p{L}+
  {
    std::size_t j = i;
    std::uint32_t c0 = cp;
    std::size_t n0 = nb;
    bool took_prefix = false;
    if (not_crlf_LN(c0)) {
      std::size_t k = i + n0;
      std::uint32_t c1{};
      std::size_t n1{};
      if (k < s.size() && next_cp(s, k, c1, n1) && is_L(c1)) {
        j = skip_letters(s, k);
        if (j > k) return j;
      }
      took_prefix = true;
      (void)took_prefix;
    }
    if (is_L(c0)) {
      j = skip_letters(s, i);
      if (j > i) return j;
    }
  }

  // \p{N}  (single number)
  if (is_N(cp)) return i + nb;

  //  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    std::size_t j = i;
    if (cp == ' ') {
      std::size_t k = i + 1;
      std::uint32_t c1{};
      std::size_t n1{};
      if (k < s.size() && next_cp(s, k, c1, n1) && not_sLN(c1) &&
          special_len_at(s, k, specials) == 0) {
        j = k;
        while (j < s.size()) {
          if (special_len_at(s, j, specials)) break;
          std::uint32_t c{};
          std::size_t n2{};
          if (!next_cp(s, j, c, n2) || !not_sLN(c)) break;
          j += n2;
        }
        while (j < s.size() && (s[j] == '\r' || s[j] == '\n') &&
               special_len_at(s, j, specials) == 0)
          ++j;
        if (j > i + 1) return j;
      }
    } else if (not_sLN(cp)) {
      j = i;
      while (j < s.size()) {
        if (special_len_at(s, j, specials)) break;
        std::uint32_t c{};
        std::size_t n2{};
        if (!next_cp(s, j, c, n2) || !not_sLN(c)) break;
        j += n2;
      }
      while (j < s.size() && (s[j] == '\r' || s[j] == '\n') &&
             special_len_at(s, j, specials) == 0)
        ++j;
      if (j > i) return j;
    }
  }

  // \s*[\r\n]+
  {
    std::size_t j = i;
    while (j < s.size()) {
      std::uint32_t c{};
      std::size_t n2{};
      if (!next_cp(s, j, c, n2) || !is_S(c)) break;
      if (c == '\r' || c == '\n') break;
      j += n2;
    }
    std::size_t k = j;
    while (k < s.size() && (s[k] == '\r' || s[k] == '\n')) ++k;
    if (k > j) return k;
  }

  // \s+(?!\S)  then \s+
  if (is_S(cp)) {
    std::size_t j = i;
    while (j < s.size()) {
      std::uint32_t c{};
      std::size_t n2{};
      if (!next_cp(s, j, c, n2) || !is_S(c)) break;
      j += n2;
    }
    if (j > i) return j;
  }

  return i + nb;  // consume one codepoint so we cannot stall
}

class TiktokenV1Pretok final : public Pretokenizer {
 public:
  explicit TiktokenV1Pretok(std::vector<std::string> specials) : specials_(std::move(specials)) {
    std::sort(specials_.begin(), specials_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
  }
  const char* name() const noexcept override { return "tiktoken_v1"; }
  Result<void> split(std::string_view text, std::string& scratch,
                     std::vector<std::string_view>& spans) const override {
    scratch.assign(text.begin(), text.end());
    spans.clear();
    const std::string_view s = scratch;
    std::size_t i = 0;
    while (i < s.size()) {
      if (auto nsp = special_len_at(s, i, specials_)) {
        spans.emplace_back(s.data() + i, nsp);
        i += nsp;
        continue;
      }
      std::size_t j = match_pat_v1(s, i, specials_);
      if (j <= i) j = i + 1;
      spans.emplace_back(s.data() + i, j - i);
      i = j;
    }
    return {};
  }

 private:
  std::vector<std::string> specials_;
};

void skip_ws(std::string_view j, std::size_t& i) {
  while (i < j.size() && std::isspace(static_cast<unsigned char>(j[i]))) ++i;
}

bool parse_int_field(std::string_view obj, std::string_view key, std::int64_t& out) {
  auto v = json_object_field(obj, key);
  if (v.empty()) return false;
  char* end = nullptr;
  out = std::strtoll(v.data(), &end, 10);
  return end != v.data();
}

Result<void> parse_bytes_array(std::string_view arr, std::string& bytes) {
  bytes.clear();
  std::size_t i = 0;
  while (i < arr.size() && arr[i] != '[') ++i;
  if (i >= arr.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "bytes"));
  ++i;
  while (i < arr.size()) {
    skip_ws(arr, i);
    if (i < arr.size() && arr[i] == ']') break;
    char* end = nullptr;
    long v = std::strtol(arr.data() + i, &end, 10);
    if (end == arr.data() + i) break;
    if (v < 0 || v > 255) return std::unexpected(make_error(Errc::ckpt_corrupt, "byte"));
    bytes.push_back(static_cast<char>(static_cast<unsigned char>(v)));
    i = static_cast<std::size_t>(end - arr.data());
    skip_ws(arr, i);
    if (i < arr.size() && arr[i] == ',') ++i;
  }
  return {};
}

}  // namespace

std::unique_ptr<Pretokenizer> make_tiktoken_v1_pretok(std::vector<std::string> specials) {
  return std::make_unique<TiktokenV1Pretok>(std::move(specials));
}

bool is_tiktoken_tok_json(std::string_view json) {
  return json.find("\"regular_tokens\"") != std::string_view::npos &&
         json.find("\"word_split\"") != std::string_view::npos;
}

Result<void> TiktokenModel::encode_span(std::string_view span, const PieceTable& pieces,
                                        std::vector<std::int32_t>& out) const {
  if (span.empty()) return {};
  {
    std::string key(span);
    if (auto it = ranks_.find(key); it != ranks_.end()) {
      out.push_back(it->second);
      return {};
    }
    const auto nspec = std::min<std::size_t>(128, pieces.pieces.size());
    for (std::size_t id = 0; id < nspec; ++id) {
      if (pieces.pieces[id] == key) {
        out.push_back(static_cast<std::int32_t>(id));
        return {};
      }
    }
  }

  std::vector<std::string> parts;
  parts.reserve(span.size());
  for (unsigned char c : span) parts.emplace_back(1, static_cast<char>(c));

  auto rank_of = [&](const std::string& a, const std::string& b) -> std::int32_t {
    auto it = ranks_.find(a + b);
    if (it == ranks_.end()) return std::numeric_limits<std::int32_t>::max();
    return it->second;
  };

  while (parts.size() >= 2) {
    std::int32_t best = std::numeric_limits<std::int32_t>::max();
    std::size_t best_i = 0;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      auto r = rank_of(parts[i], parts[i + 1]);
      if (r < best) {
        best = r;
        best_i = i;
      }
    }
    if (best == std::numeric_limits<std::int32_t>::max()) break;
    parts[best_i] += parts[best_i + 1];
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
  }

  for (const auto& p : parts) {
    auto it = ranks_.find(p);
    if (it == ranks_.end()) {
      return std::unexpected(make_error(Errc::unsupported, "tiktoken byte"));
    }
    out.push_back(it->second);
  }
  return {};
}

std::string TiktokenModel::to_json() const { return "\"ranks\":null"; }

Result<void> TiktokenModel::write_huggingface(const std::filesystem::path&, const Pretokenizer&,
                                              const PieceTable&) const {
  return std::unexpected(make_error(Errc::unsupported, "tiktoken hf export"));
}

Result<std::unique_ptr<Tokenizer>> load_tiktoken_json(std::string_view json) {
  if (!is_tiktoken_tok_json(json)) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, "not tiktoken tok.json"));
  }
  auto ws = json_object_field(json, "word_split");
  std::string wss(ws.begin(), ws.end());
  if (wss.find("V1") == std::string::npos) {
    return std::unexpected(make_error(Errc::unsupported, "word_split"));
  }
  std::int64_t vocab = 0;
  parse_int_field(json, "vocab_size", vocab);

  PieceTable table;
  if (vocab > 0) table.pieces.assign(static_cast<std::size_t>(vocab), {});
  std::unordered_map<std::string, std::int32_t> ranks;
  std::vector<std::string> specials;

  auto parse_token_list = [&](std::string_view arr, bool special) -> Result<void> {
    std::size_t i = 0;
    while (i < arr.size() && arr[i] != '[') ++i;
    if (i >= arr.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "token list"));
    ++i;
    while (i < arr.size()) {
      skip_ws(arr, i);
      if (i < arr.size() && arr[i] == ']') break;
      if (i >= arr.size() || arr[i] != '{') break;
      std::size_t start = i;
      int depth = 0;
      for (; i < arr.size(); ++i) {
        if (arr[i] == '{') ++depth;
        else if (arr[i] == '}') {
          --depth;
          if (depth == 0) {
            ++i;
            break;
          }
        }
      }
      auto obj = arr.substr(start, i - start);
      auto bytesv = json_object_field(obj, "bytes");
      std::string bytes;
      auto pb = parse_bytes_array(bytesv, bytes);
      if (!pb) return pb;
      std::int64_t tok = -1;
      if (!parse_int_field(obj, "token", tok) || tok < 0) {
        return std::unexpected(make_error(Errc::ckpt_corrupt, "token id"));
      }
      auto id = static_cast<std::int32_t>(tok);
      if (static_cast<std::size_t>(id) >= table.pieces.size()) {
        table.pieces.resize(static_cast<std::size_t>(id) + 1);
      }
      table.pieces[static_cast<std::size_t>(id)] = bytes;
      if (special) specials.push_back(bytes);
      else ranks.emplace(bytes, id);
      skip_ws(arr, i);
      if (i < arr.size() && arr[i] == ',') ++i;
    }
    return {};
  };

  auto rt = json_object_field(json, "regular_tokens");
  auto st = json_object_field(json, "special_tokens");
  auto pr = parse_token_list(rt, false);
  if (!pr) return std::unexpected(pr.error());
  auto ps = parse_token_list(st, true);
  if (!ps) return std::unexpected(ps.error());

  if (vocab <= 0) vocab = static_cast<std::int64_t>(table.pieces.size());
  auto model = std::make_unique<TiktokenModel>();
  model->set_vocab_size(vocab);
  model->set_ranks(std::move(ranks));
  auto pretok = make_tiktoken_v1_pretok(std::move(specials));
  return std::make_unique<Tokenizer>(std::move(pretok), std::move(model), std::move(table));
}

}  // namespace gyre
