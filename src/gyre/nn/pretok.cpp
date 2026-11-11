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

 private:
  bool dummy_{true};
};

// GPT-2 byte-level pre-tokenizer. Splits the RAW text with the published pattern
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// (Unicode-aware via the generated category tables) and returns spans of the original bytes. The
// byte -> unicode alphabet of Hugging Face files is only applied when reading/writing those files
// (see bytelevel.hpp); inside Gyre pieces are raw bytes.
class Gpt2Pretok final : public Pretokenizer {
 public:
  const char* name() const noexcept override { return "bytelevel"; }
  Result<void> split(std::string_view text, std::string&, std::vector<std::string_view>& spans) const override {
    spans.clear();
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
    return {};
  }
};

}  // namespace

std::unique_ptr<Pretokenizer> make_identity_pretok() {
  return std::make_unique<IdentityPretok>();
}
std::unique_ptr<Pretokenizer> make_gpt2_pretok() { return std::make_unique<Gpt2Pretok>(); }
std::unique_ptr<Pretokenizer> make_metaspace_pretok(bool dummy_prefix) {
  return std::make_unique<MetaspacePretok>(dummy_prefix);
}

Result<std::unique_ptr<Pretokenizer>> pretok_from_name(std::string_view name) {
  if (name.empty() || name == "identity") return make_identity_pretok();
  if (name == "bytelevel" || name == "gpt2") return make_gpt2_pretok();
  if (name == "metaspace") return make_metaspace_pretok(true);
  if (name == "tiktoken_v1") return make_tiktoken_v1_pretok({});
  return std::unexpected(make_error(Errc::unsupported, "unknown pretok"));
}

std::string Pretokenizer::to_json() const {
  return std::string("\"") + name() + '"';
}

}  // namespace gyre
