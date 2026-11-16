#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/tiktoken.hpp"

#include <cctype>

namespace gyre {
namespace {

class IdentityPretok final : public Pretokenizer {
 public:
  const char* name() const noexcept override { return "identity"; }
  Result<void> split(std::string_view text, std::string&,
                     std::vector<std::string_view>& spans) const override {
    spans.clear();
    if (!text.empty()) spans.push_back(text);
    return {};
  }
};

class MetaspacePretok final : public Pretokenizer {
 public:
  explicit MetaspacePretok(bool dummy) : dummy_(dummy) {}
  const char* name() const noexcept override { return "metaspace"; }
  Result<void> split(std::string_view text, std::string& scratch,
                     std::vector<std::string_view>& spans) const override {
    scratch.clear();
    if (dummy_) scratch.push_back('\xe2'), scratch.push_back('\x96'), scratch.push_back('\x81');  // ▁
    for (unsigned char c : text) {
      if (c == ' ') {
        scratch.push_back('\xe2');
        scratch.push_back('\x96');
        scratch.push_back('\x81');
      } else {
        scratch.push_back(static_cast<char>(c));
      }
    }
    spans.clear();
    if (!scratch.empty()) spans.emplace_back(scratch);
    return {};
  }
  // Hugging Face Metaspace decoder: every U+2581 becomes a space, and the space the dummy prefix
  // put in front of the text is dropped again.
  void decode(std::string& text) const override {
    std::string o;
    o.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
      if (text.compare(i, 3, "\xe2\x96\x81") == 0) {
        o.push_back(' ');
        i += 3;
      } else {
        o.push_back(text[i++]);
      }
    }
    if (dummy_ && !o.empty() && o.front() == ' ' && text.compare(0, 3, "\xe2\x96\x81") == 0) o.erase(0, 1);
    text = std::move(o);
  }

 private:
  bool dummy_{true};
};

// GPT-2 byte-level pre-tokenizer. Splits the RAW text with the published pattern
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// (Unicode-aware via the generated category tables) and appends spans of the original bytes. The
// byte -> unicode alphabet of Hugging Face files is only applied when reading/writing those files
// (see bytelevel.hpp); inside Gyre pieces are raw bytes. `text` is a whole input or one piece of an
// earlier split; the (?!\S) look-ahead sees the end of `text` as the end of the string, which is
// how Hugging Face runs ByteLevel after another pre-tokenizer in a Sequence.
void gpt2_split_append(std::string_view text, std::vector<std::string_view>& spans) {
  const std::size_t n = text.size();
  std::size_t i = 0;
  std::uint32_t cp = 0;
  std::size_t nb = 0;
  auto emit = [&](std::size_t from, std::size_t to) { spans.emplace_back(text.data() + from, to - from); };
  auto is_other = [](std::uint32_t c) {
    return !unicode::is_space(c) && !unicode::is_letter(c) && !unicode::is_number(c);
  };
  while (i < n) {
    // 's 't 'm 'd 're 've 'll
    if (text[i] == '\'' && i + 1 < n) {
      const char c1 = text[i + 1];
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        emit(i, i + 2);
        i += 2;
        continue;
      }
      if (i + 2 < n) {
        const char c2 = text[i + 2];
        if (((c1 == 'r' || c1 == 'v') && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
          emit(i, i + 3);
          i += 3;
          continue;
        }
      }
    }
    // optional single space, then a run of letters / numbers / other
    std::size_t q = i;
    if (text[q] == ' ') ++q;
    if (q < n) {
      unicode::next_codepoint(text, q, cp, nb);
      int kind = unicode::is_letter(cp) ? 1 : unicode::is_number(cp) ? 2 : is_other(cp) ? 3 : 0;
      if (kind != 0) {
        std::size_t k = q;
        while (k < n) {
          unicode::next_codepoint(text, k, cp, nb);
          const int kk = unicode::is_letter(cp) ? 1 : unicode::is_number(cp) ? 2 : is_other(cp) ? 3 : 0;
          if (kk != kind) break;
          k += nb;
        }
        emit(i, k);
        i = k;
        continue;
      }
    }
    // whitespace: \s+(?!\S) leaves the last space for the next token; \s+ takes a lone one
    std::size_t k = i, last = i;
    int count = 0;
    while (k < n) {
      unicode::next_codepoint(text, k, cp, nb);
      if (!unicode::is_space(cp)) break;
      last = k;
      k += nb;
      ++count;
    }
    if (count == 0) {  // defensive: never loop forever on malformed input
      emit(i, i + 1);
      ++i;
      continue;
    }
    if (k >= n || count < 2) {
      emit(i, k);
      i = k;
    } else {
      emit(i, last);
      i = last;
    }
  }
}

class Gpt2Pretok final : public Pretokenizer {
 public:
  const char* name() const noexcept override { return "bytelevel"; }
  Result<void> split(std::string_view text, std::string&, std::vector<std::string_view>& spans) const override {
    spans.clear();
    gpt2_split_append(text, spans);
    return {};
  }
};

// Hugging Face Sequence[Digits(individual_digits=true), ByteLevel(use_regex=true)] (SmolLM2,
// StarCoder). Every numeric code point (Rust char::is_numeric = \p{N}: Nd, Nl, No) becomes its own
// piece; each run between them goes through the GPT-2 pattern on its own. Invalid UTF-8 bytes are
// never numeric.
class DigitsBytelevelPretok final : public Pretokenizer {
 public:
  const char* name() const noexcept override { return "digits_bytelevel"; }
  Result<void> split(std::string_view text, std::string&, std::vector<std::string_view>& spans) const override {
    spans.clear();
    const std::size_t n = text.size();
    std::size_t run = 0, i = 0;
    std::uint32_t cp = 0;
    std::size_t nb = 0;
    while (i < n) {
      unicode::next_codepoint(text, i, cp, nb);
      if (nb == 0) nb = 1;
      const bool valid = cp < 0x80 || nb > 1;
      if (valid && unicode::is_number(cp)) {
        if (i > run) gpt2_split_append(text.substr(run, i - run), spans);
        spans.push_back(text.substr(i, nb));
        run = i + nb;
      }
      i += nb;
    }
    if (n > run) gpt2_split_append(text.substr(run), spans);
    return {};
  }
};

}  // namespace

std::unique_ptr<Pretokenizer> make_identity_pretok() {
  return std::make_unique<IdentityPretok>();
}
std::unique_ptr<Pretokenizer> make_gpt2_pretok() { return std::make_unique<Gpt2Pretok>(); }
std::unique_ptr<Pretokenizer> make_digits_bytelevel_pretok() { return std::make_unique<DigitsBytelevelPretok>(); }
std::unique_ptr<Pretokenizer> make_metaspace_pretok(bool dummy_prefix) {
  return std::make_unique<MetaspacePretok>(dummy_prefix);
}

Result<std::unique_ptr<Pretokenizer>> pretok_from_name(std::string_view name) {
  if (name.empty() || name == "identity") return make_identity_pretok();
  if (name == "bytelevel" || name == "gpt2") return make_gpt2_pretok();
  if (name == "digits_bytelevel") return make_digits_bytelevel_pretok();
  if (name == "metaspace") return make_metaspace_pretok(true);
  if (name == "tiktoken_v1") return make_tiktoken_v1_pretok({});
  return std::unexpected(make_error(Errc::unsupported, "unknown pretok"));
}

std::string Pretokenizer::to_json() const {
  return std::string("\"") + name() + '"';
}

}  // namespace gyre
