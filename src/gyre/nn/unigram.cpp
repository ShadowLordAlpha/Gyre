#include "gyre/nn/unigram.hpp"

#include "json_parse.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace gyre {

UnigramModel::UnigramModel(std::vector<float> scores, bool byte_fallback)
    : scores_(std::move(scores)), byte_fallback_(byte_fallback) {}

std::int64_t UnigramModel::vocab_size() const noexcept {
  return static_cast<std::int64_t>(scores_.size());
}

namespace {

std::size_t utf8_len(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 1;
}

Error unknown_text(std::string_view s) {
  std::string hex;
  static constexpr char kHex[] = "0123456789ABCDEF";
  for (unsigned char c : s) {
    hex += "\\x";
    hex.push_back(kHex[c >> 4]);
    hex.push_back(kHex[c & 15]);
  }
  return make_error(Errc::unsupported, "unigram: text \"" + hex + "\" has no piece and the model has no unk_id");
}

}  // namespace

// SentencePiece / Hugging Face Unigram lattice: pieces of the matchable types, plus an "unknown"
// edge over every character that has no one-character piece (score min_score - 10). After
// Viterbi an unknown character becomes its BYTE pieces (byte fallback) or unk_id; consecutive
// unks merge into one, as SentencePiece does.
Result<void> UnigramModel::encode_typed(std::string_view span, const PieceTable& pieces,
                                        std::vector<std::int32_t>& out) const {
  const auto n = span.size();
  const auto V = pieces.pieces.size();
  if (types_.size() != V) return std::unexpected(make_error(Errc::invalid_shape, "unigram piece types"));
  std::unordered_map<std::string_view, std::int32_t> inv;
  inv.reserve(V);
  std::int32_t byte_id[256];
  std::fill(std::begin(byte_id), std::end(byte_id), -1);
  std::size_t max_len = 1;
  float min_score = 0.f;
  bool any_score = false;
  for (std::size_t i = 0; i < V; ++i) {
    const auto& p = pieces.pieces[i];
    if (types_[i] == kByte) {
      if (p.size() == 1 && byte_id[static_cast<unsigned char>(p[0])] < 0)
        byte_id[static_cast<unsigned char>(p[0])] = static_cast<std::int32_t>(i);
      continue;
    }
    if ((types_[i] != kNormal && types_[i] != kUserDefined) || p.empty()) continue;
    inv.emplace(p, static_cast<std::int32_t>(i));
    max_len = std::max(max_len, p.size());
    min_score = any_score ? std::min(min_score, scores_[i]) : scores_[i];
    any_score = true;
  }
  const float unk_score = min_score - 10.f;
  constexpr std::int32_t kUnkEdge = -2;
  const float ninf = -std::numeric_limits<float>::infinity();
  std::vector<float> best(n + 1, ninf);
  std::vector<std::size_t> back(n + 1, 0);
  std::vector<std::int32_t> tok(n + 1, -1);
  best[0] = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    if (best[i] == ninf) continue;
    const std::size_t jmax = std::min(n, i + max_len);
    for (std::size_t j = i + 1; j <= jmax; ++j) {
      auto it = inv.find(span.substr(i, j - i));
      if (it == inv.end()) continue;
      const float sc = scores_[static_cast<std::size_t>(it->second)] + best[i];
      if (sc > best[j]) {
        best[j] = sc;
        back[j] = i;
        tok[j] = it->second;
      }
    }
    const std::size_t cl = std::min(n - i, utf8_len(static_cast<unsigned char>(span[i])));
    if (!inv.contains(span.substr(i, cl))) {
      const float sc = best[i] + unk_score;
      if (sc > best[i + cl]) {
        best[i + cl] = sc;
        back[i + cl] = i;
        tok[i + cl] = kUnkEdge;
      }
    }
  }
  if (best[n] == ninf) return std::unexpected(make_error(Errc::unsupported, "unigram cannot segment"));
  std::vector<std::size_t> cuts;  // segment ends, last to first
  for (std::size_t i = n; i > 0; i = back[i]) cuts.push_back(i);
  std::size_t from = 0;
  bool prev_unk = false;
  for (auto c = cuts.rbegin(); c != cuts.rend(); ++c) {
    const std::size_t to = *c;
    if (tok[to] != kUnkEdge) {
      out.push_back(tok[to]);
      prev_unk = false;
    } else {
      bool bytes_ok = byte_fallback_;
      for (std::size_t k = from; bytes_ok && k < to; ++k) bytes_ok = byte_id[static_cast<unsigned char>(span[k])] >= 0;
      if (bytes_ok) {
        for (std::size_t k = from; k < to; ++k) out.push_back(byte_id[static_cast<unsigned char>(span[k])]);
        prev_unk = false;
      } else {
        if (unk_id_ < 0) return std::unexpected(unknown_text(span.substr(from, to - from)));
        if (!prev_unk) out.push_back(unk_id_);
        prev_unk = true;
      }
    }
    from = to;
  }
  return {};
}

Result<void> UnigramModel::encode_span(std::string_view span, const PieceTable& pieces,
                                       std::vector<std::int32_t>& out) const {
  const auto n = span.size();
  if (n == 0) return {};
  const auto V = pieces.pieces.size();
  if (V == 0 || scores_.size() != V)
    return std::unexpected(make_error(Errc::invalid_shape, "unigram pieces"));
  if (!types_.empty()) return encode_typed(span, pieces, out);

  std::unordered_map<std::string_view, std::int32_t> inv;
  inv.reserve(V);
  for (std::size_t i = 0; i < V; ++i) inv.emplace(pieces.pieces[i], static_cast<std::int32_t>(i));

  std::size_t max_len = 1;
  for (auto& p : pieces.pieces) max_len = std::max(max_len, p.size());
  std::vector<float> best(n + 1, -std::numeric_limits<float>::infinity());
  std::vector<int> back(n + 1, -1);
  std::vector<std::int32_t> tok(n + 1, -1);
  best[0] = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    if (best[i] == -std::numeric_limits<float>::infinity()) continue;
    bool any = false;
    const std::size_t jmax = std::min(n, i + max_len);
    for (std::size_t j = i + 1; j <= jmax; ++j) {
      auto sub = span.substr(i, j - i);
      auto it = inv.find(sub);
      if (it == inv.end()) continue;
      any = true;
      float sc = scores_[static_cast<std::size_t>(it->second)] + best[i];
      if (sc > best[j]) {
        best[j] = sc;
        back[j] = static_cast<int>(i);
        tok[j] = it->second;
      }
    }
    if (!any && byte_fallback_) {
      std::size_t j = i + 1;
      float sc = best[i] - 10.f;
      if (sc > best[j]) {
        best[j] = sc;
        back[j] = static_cast<int>(i);
        auto it = inv.find(span.substr(i, 1));
        // A byte with no piece is unk_id (historically 0); -1 marks it for the error below.
        tok[j] = it == inv.end() ? unk_id_ : it->second;
      }
    }
  }
  if (best[n] == -std::numeric_limits<float>::infinity())
    return std::unexpected(make_error(Errc::unsupported, "unigram cannot segment"));
  std::vector<std::int32_t> rev;
  for (int i = static_cast<int>(n); i > 0; i = back[static_cast<std::size_t>(i)]) {
    if (i <= 0) break;
    if (tok[static_cast<std::size_t>(i)] < 0)
      return std::unexpected(unknown_text(span.substr(static_cast<std::size_t>(i) - 1, 1)));
    rev.push_back(tok[static_cast<std::size_t>(i)]);
  }
  out.insert(out.end(), rev.rbegin(), rev.rend());
  return {};
}

std::string UnigramModel::to_json() const {
  nlohmann::json j;
  j["scores"] = scores_;
  j["byte_fallback"] = byte_fallback_;
  // Only written when they differ from what an older file implies, so those re-save unchanged.
  if (unk_id_ != 0) j["unk_id"] = unk_id_;
  if (!types_.empty()) j["piece_types"] = types_;
  return j.dump();
}

Result<void> UnigramModel::write_huggingface(const std::filesystem::path& dir,
                                             const Pretokenizer& pretok, const PieceTable& pieces) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream t(dir / "tokenizer.json", std::ios::binary);
  if (!t) return std::unexpected(make_error(Errc::io, "tokenizer.json"));
  nlohmann::json tok;
  tok["version"] = "1.0";
  tok["pre_tokenizer"] = {{"type", pretok.name()}};
  nlohmann::json vocab = nlohmann::json::array();
  bool has_bytes = false;
  for (std::size_t i = 0; i < pieces.pieces.size(); ++i) {
    float sc = i < scores_.size() ? scores_[i] : 0.f;
    const auto& p = pieces.pieces[i];
    if (i < types_.size() && types_[i] == kByte && p.size() == 1) {
      // Hugging Face names byte-fallback pieces <0xNN>.
      static constexpr char kHex[] = "0123456789ABCDEF";
      const auto b = static_cast<unsigned char>(p[0]);
      vocab.push_back({std::string("<0x") + kHex[b >> 4] + kHex[b & 15] + ">", sc});
      has_bytes = true;
      continue;
    }
    vocab.push_back({bytes_to_json_text(p), sc});
  }
  nlohmann::json unk = unk_id_ >= 0 ? nlohmann::json(unk_id_) : nlohmann::json(nullptr);
  tok["model"] = {{"type", "Unigram"}, {"unk_id", unk}, {"vocab", std::move(vocab)}};
  if (has_bytes) tok["model"]["byte_fallback"] = byte_fallback_;
  t << tok.dump();
  t.flush();
  if (!t) return std::unexpected(make_error(Errc::io, "tokenizer.json write"));
  return {};
}

Result<std::unique_ptr<UnigramModel>> UnigramModel::from_scores(std::vector<float> scores,
                                                               bool byte_fallback) {
  return std::make_unique<UnigramModel>(std::move(scores), byte_fallback);
}

Result<std::pair<std::unique_ptr<UnigramModel>, PieceTable>> UnigramModel::train(
    std::string_view text, std::int32_t vocab_size, std::function<void(int, int)> progress) {
  if (text.empty()) return std::unexpected(make_error(Errc::invalid_shape, "empty text"));
  if (vocab_size < 16) return std::unexpected(make_error(Errc::invalid_shape, "vocab_size"));
  std::unordered_map<std::string, std::int64_t> counts;
  for (unsigned char c : text) counts[std::string(1, static_cast<char>(c))]++;
  const int max_n = 8;
  for (int n = 2; n <= max_n; ++n) {
    if (text.size() < static_cast<std::size_t>(n)) break;
    for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= text.size(); ++i)
      counts[std::string(text.substr(i, static_cast<std::size_t>(n)))]++;
    if (progress) progress(n, max_n);
  }
  // Single bytes are the fallback that makes every text encodable, so they are never cut by the
  // count ranking: every byte seen in `text` is kept, and with room (vocab_size >= 512) the unseen
  // ones too, at the lowest score, so held-out text with a new byte still round-trips. With a
  // smaller vocab an unseen byte is an encode error (unk_id -1), never a silent id 0.
  std::vector<std::pair<std::int64_t, std::string>> singles, multi;
  for (auto& [s, c] : counts) {
    if (s.size() == 1) singles.emplace_back(c, s);
    else if (c >= 2) multi.emplace_back(c, s);
  }
  if (singles.size() > static_cast<std::size_t>(vocab_size))
    return std::unexpected(make_error(Errc::invalid_shape, "unigram: vocab_size " + std::to_string(vocab_size) +
                                                               " is below the " + std::to_string(singles.size()) +
                                                               " distinct bytes of the text"));
  const bool all_bytes = vocab_size >= 512;
  if (all_bytes) {
    for (int b = 0; b < 256; ++b) {
      std::string s(1, static_cast<char>(b));
      if (!counts.contains(s)) singles.emplace_back(0, std::move(s));
    }
  }
  const auto by_rank = [](auto& a, auto& b) {
    if (a.first != b.first) return a.first > b.first;
    if (a.second.size() != b.second.size()) return a.second.size() < b.second.size();
    return a.second < b.second;
  };
  std::sort(multi.begin(), multi.end(), by_rank);
  multi.resize(std::min(multi.size(), static_cast<std::size_t>(vocab_size) - singles.size()));
  std::vector<std::pair<std::int64_t, std::string>> ranked = std::move(singles);
  ranked.insert(ranked.end(), std::make_move_iterator(multi.begin()), std::make_move_iterator(multi.end()));
  std::sort(ranked.begin(), ranked.end(), by_rank);
  PieceTable t;
  std::vector<float> scores;
  const auto total = static_cast<double>(text.size());
  for (auto& [c, s] : ranked) {
    t.pieces.push_back(s);
    scores.push_back(static_cast<float>(std::log(c > 0 ? static_cast<double>(c) / total : 0.5 / total)));
  }
  if (progress) progress(max_n, max_n);
  auto model = std::make_unique<UnigramModel>(std::move(scores), true);
  model->set_unk_id(-1);
  return std::pair{std::move(model), std::move(t)};
}

}  // namespace gyre
