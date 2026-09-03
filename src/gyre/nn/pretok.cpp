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

// GPT-2 bytes-to-unicode (same table as OpenAI).
void gpt2_byte_encoder(std::uint32_t enc[256]) {
  int n = 0;
  for (int b = 0; b < 256; ++b) enc[b] = 0;
  for (int b = '!'; b <= '~'; ++b) enc[b] = static_cast<std::uint32_t>(b);
  for (int b = 0xA1; b <= 0xAC; ++b) enc[b] = static_cast<std::uint32_t>(b);
  for (int b = 0xAE; b <= 0xFF; ++b) enc[b] = static_cast<std::uint32_t>(b);
  for (int b = 0; b < 256; ++b) {
    if (enc[b] == 0 && b != 0) {
      // unassigned get 256+n
    }
  }
  n = 0;
  for (int b = 0; b < 256; ++b) {
    bool mapped = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    if (!mapped) {
      enc[b] = static_cast<std::uint32_t>(256 + n);
      ++n;
    } else {
      enc[b] = static_cast<std::uint32_t>(b);
    }
  }
}

void append_utf8(std::string& s, std::uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool is_letter(unsigned char c) {
  return std::isalpha(c) != 0;
}
bool is_digit(unsigned char c) { return std::isdigit(c) != 0; }

class Gpt2Pretok final : public Pretokenizer {
 public:
  Gpt2Pretok() { gpt2_byte_encoder(enc_); }
  const char* name() const noexcept override { return "bytelevel"; }
  Result<void> split(std::string_view text, std::string& scratch,
                     std::vector<std::string_view>& spans) const override {
    scratch.clear();
    for (unsigned char c : text) append_utf8(scratch, enc_[c]);
    spans.clear();
    const std::size_t n = scratch.size();
    std::size_t i = 0;
    auto peek = [&](std::size_t k) -> unsigned char {
      return k < n ? static_cast<unsigned char>(scratch[k]) : 0;
    };
    while (i < n) {
      std::size_t start = i;
      unsigned char c = peek(i);
      // 's | 't | 're | 've | 'm | 'll | 'd
      if (c == '\'' && i + 1 < n) {
        unsigned char n1 = peek(i + 1);
        if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
          i += 2;
        } else if ((n1 == 'r' || n1 == 'v') && i + 2 < n && peek(i + 2) == 'e') {
          i += 3;
        } else if (n1 == 'l' && i + 2 < n && peek(i + 2) == 'l') {
          i += 3;
        } else {
          goto rest;
        }
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
    rest:
      // optional space + letters
      if (c == ' ' && i + 1 < n && is_letter(peek(i + 1))) {
        ++i;
        while (i < n && is_letter(peek(i))) ++i;
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      if (is_letter(c)) {
        while (i < n && is_letter(peek(i))) ++i;
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      if (c == ' ' && i + 1 < n && is_digit(peek(i + 1))) {
        ++i;
        while (i < n && is_digit(peek(i))) ++i;
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      if (is_digit(c)) {
        while (i < n && is_digit(peek(i))) ++i;
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      // optional space + non-letter-digit
      if (c == ' ') {
        ++i;
        if (i < n && !std::isspace(peek(i)) && !is_letter(peek(i)) && !is_digit(peek(i))) {
          while (i < n && !std::isspace(peek(i)) && !is_letter(peek(i)) && !is_digit(peek(i))) ++i;
        } else {
          while (i < n && std::isspace(peek(i))) ++i;
        }
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      if (std::isspace(c)) {
        while (i < n && std::isspace(peek(i))) ++i;
        spans.emplace_back(scratch.data() + start, i - start);
        continue;
      }
      while (i < n && !std::isspace(peek(i)) && !is_letter(peek(i)) && !is_digit(peek(i)) &&
             peek(i) != '\'')
        ++i;
      if (i == start) ++i;
      spans.emplace_back(scratch.data() + start, i - start);
    }
    return {};
  }

 private:
  std::uint32_t enc_[256]{};
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
