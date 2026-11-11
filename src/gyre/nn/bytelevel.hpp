#pragma once
// GPT-2 "bytes to unicode" alphabet used by Hugging Face byte-level BPE files (vocab.json keys,
// merges.txt, tokenizer.json). Inside Gyre pieces are always RAW BYTES; this table is only used
// when reading or writing those foreign files.
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace gyre::bytelevel {

// byte -> code point: printable Latin-1 maps to itself, the 68 others to 256, 257, ...
inline const std::array<std::uint32_t, 256>& encoder() {
  static const std::array<std::uint32_t, 256> tbl = [] {
    std::array<std::uint32_t, 256> e{};
    std::uint32_t n = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
      const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      e[b] = keep ? b : 256 + n++;
    }
    return e;
  }();
  return tbl;
}

inline void append_utf8(std::string& s, std::uint32_t cp) {
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

// raw bytes -> UTF-8 text in the GPT-2 alphabet
inline std::string to_mapped(std::string_view raw) {
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char c : raw) append_utf8(out, encoder()[c]);
  return out;
}

// UTF-8 text in the GPT-2 alphabet -> raw bytes. False if a character is outside the alphabet.
inline bool from_mapped(std::string_view s, std::string& raw) {
  static const std::array<int, 0x200> inv = [] {
    std::array<int, 0x200> t{};
    t.fill(-1);
    for (int b = 0; b < 256; ++b) t[encoder()[static_cast<std::size_t>(b)]] = b;
    return t;
  }();
  raw.clear();
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::uint32_t cp;
    if (c < 0x80) {
      cp = c;
      i += 1;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
      cp = (static_cast<std::uint32_t>(c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
      i += 2;
    } else {
      return false;
    }
    if (cp >= inv.size() || inv[cp] < 0) return false;
    raw.push_back(static_cast<char>(inv[cp]));
  }
  return true;
}

}  // namespace gyre::bytelevel
