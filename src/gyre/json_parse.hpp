#pragma once

#include "gyre/error.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

namespace gyre {

inline Result<nlohmann::json> parse_json(std::string_view s) {
  try {
    return nlohmann::json::parse(s.begin(), s.end());
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("json: ") + e.what()));
  }
}

// Tokenizer pieces are raw bytes. JSON strings must be UTF-8; map 0x80–0xFF
// to U+0080–U+00FF (same as the old `\u00xx` encoder).
inline std::string bytes_to_json_text(std::string_view s) {
  std::string o;
  o.reserve(s.size() * 2);
  for (unsigned char c : s) {
    if (c < 0x80) {
      o.push_back(static_cast<char>(c));
    } else {
      o.push_back(static_cast<char>(0xC0 | (c >> 6)));
      o.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
  }
  return o;
}

inline std::string json_text_to_bytes(std::string_view s) {
  std::string o;
  o.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      o.push_back(s[i]);
      ++i;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
      unsigned cp = (static_cast<unsigned>(c & 0x1F) << 6) |
                    (static_cast<unsigned>(s[i + 1]) & 0x3F);
      o.push_back(static_cast<char>(cp & 0xFF));
      i += 2;
    } else {
      o.push_back(s[i]);
      ++i;
    }
  }
  return o;
}

}  // namespace gyre
