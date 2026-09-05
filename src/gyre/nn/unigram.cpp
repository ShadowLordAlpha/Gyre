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

Result<void> UnigramModel::encode_span(std::string_view span, const PieceTable& pieces,
                                       std::vector<std::int32_t>& out) const {
  const auto n = span.size();
  if (n == 0) return {};
  const auto V = pieces.pieces.size();
  if (V == 0 || scores_.size() != V)
    return std::unexpected(make_error(Errc::invalid_shape, "unigram pieces"));

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
        tok[j] = it == inv.end() ? 0 : it->second;
      }
    }
  }
  if (best[n] == -std::numeric_limits<float>::infinity())
    return std::unexpected(make_error(Errc::unsupported, "unigram cannot segment"));
  std::vector<std::int32_t> rev;
  for (int i = static_cast<int>(n); i > 0; i = back[static_cast<std::size_t>(i)]) {
    if (i <= 0) break;
    rev.push_back(tok[static_cast<std::size_t>(i)]);
  }
  out.insert(out.end(), rev.rbegin(), rev.rend());
  return {};
}

std::string UnigramModel::to_json() const {
  nlohmann::json j;
  j["scores"] = scores_;
  j["byte_fallback"] = byte_fallback_;
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
  for (std::size_t i = 0; i < pieces.pieces.size(); ++i) {
    float sc = i < scores_.size() ? scores_[i] : 0.f;
    vocab.push_back({bytes_to_json_text(pieces.pieces[i]), sc});
  }
  tok["model"] = {{"type", "Unigram"}, {"unk_id", 0}, {"vocab", std::move(vocab)}};
  t << tok.dump();
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
  std::vector<std::pair<std::int64_t, std::string>> ranked;
  ranked.reserve(counts.size());
  for (auto& [s, c] : counts) {
    if (s.size() == 1 || c >= 2) ranked.emplace_back(c, s);
  }
  std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) {
    if (a.first != b.first) return a.first > b.first;
    if (a.second.size() != b.second.size()) return a.second.size() < b.second.size();
    return a.second < b.second;
  });
  PieceTable t;
  std::vector<float> scores;
  const auto total = static_cast<double>(text.size());
  std::size_t take = std::min(static_cast<std::size_t>(vocab_size), ranked.size());
  for (std::size_t i = 0; i < take; ++i) {
    t.pieces.push_back(ranked[i].second);
    scores.push_back(static_cast<float>(std::log(std::max(1.0, static_cast<double>(ranked[i].first)) / total)));
  }
  if (progress) progress(max_n, max_n);
  return std::pair{std::make_unique<UnigramModel>(std::move(scores), true), std::move(t)};
}

}  // namespace gyre
