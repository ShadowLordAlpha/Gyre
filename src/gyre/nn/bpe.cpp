#include "gyre/nn/bpe.hpp"

#include "bytelevel.hpp"
#include "json_parse.hpp"

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>

namespace gyre {
namespace {

struct PairHash {
  std::size_t operator()(std::pair<std::int32_t, std::int32_t> p) const noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
           static_cast<std::uint32_t>(p.second);
  }
};

void apply_merge(std::vector<std::int32_t>& seq, std::int32_t a, std::int32_t b, std::int32_t c) {
  std::size_t w = 0;
  for (std::size_t r = 0; r < seq.size();) {
    if (r + 1 < seq.size() && seq[r] == a && seq[r + 1] == b) {
      seq[w++] = c;
      r += 2;
    } else {
      seq[w++] = seq[r++];
    }
  }
  seq.resize(w);
}

void rebuild_pieces_from_merges(PieceTable& t, const std::vector<std::pair<std::int32_t, std::int32_t>>& merges) {
  t.pieces.assign(256, {});
  t.byte_to_id.clear();
  t.identity256 = true;
  for (int i = 0; i < 256; ++i) {
    t.pieces[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(i));
    t.byte_to_id[static_cast<std::uint8_t>(i)] = i;
  }
  for (auto [a, b] : merges) {
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= t.pieces.size() ||
        static_cast<std::size_t>(b) >= t.pieces.size())
      continue;
    t.pieces.push_back(t.pieces[static_cast<std::size_t>(a)] + t.pieces[static_cast<std::size_t>(b)]);
  }
}

}  // namespace

BpeModel::BpeModel(std::vector<std::pair<std::int32_t, std::int32_t>> merges) : merges_(std::move(merges)) {
  vocab_size_ = 256 + static_cast<std::int64_t>(merges_.size());
  rank_.reserve(merges_.size() * 2);
  for (std::size_t i = 0; i < merges_.size(); ++i) {
    const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(merges_[i].first)) << 32) |
                     static_cast<std::uint32_t>(merges_[i].second);
    rank_.emplace(key, static_cast<std::int32_t>(i));  // first occurrence wins
  }
}

std::int64_t BpeModel::vocab_size() const noexcept { return vocab_size_; }

Result<void> BpeModel::encode_span(std::string_view span, const PieceTable& pieces,
                                   std::vector<std::int32_t>& out) const {
  std::vector<std::int32_t> seq;
  seq.reserve(span.size());
  if (pieces.identity256 && pieces.byte_to_id.empty()) {
    for (unsigned char c : span) seq.push_back(static_cast<std::int32_t>(c));
  } else if (!pieces.byte_to_id.empty()) {
    for (unsigned char c : span) {
      auto it = pieces.byte_to_id.find(c);
      if (it == pieces.byte_to_id.end()) {
        if (!merge_ids_.empty()) continue;  // Hugging Face BPE without unk_token skips it
        return std::unexpected(make_error(Errc::unsupported, "unknown byte"));
      }
      seq.push_back(it->second);
    }
  } else {
    for (unsigned char c : span) seq.push_back(static_cast<std::int32_t>(c));
  }
  std::int32_t nid = static_cast<std::int32_t>(pieces.identity256 ? 256 : pieces.pieces.size() - merges_.size());
  if (pieces.identity256) nid = 256;
  else {
    // restricted alphabet: new merge ids start at alphabet size
    nid = static_cast<std::int32_t>(pieces.byte_to_id.size());
  }
  // Lowest-rank pair first: same result as applying every merge in order, but only merges that
  // actually occur in this span are visited.
  while (seq.size() > 1 && !rank_.empty()) {
    std::int32_t best = -1;
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
      const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(seq[i])) << 32) |
                       static_cast<std::uint32_t>(seq[i + 1]);
      auto it = rank_.find(key);
      if (it != rank_.end() && (best < 0 || it->second < best)) best = it->second;
    }
    if (best < 0) break;
    const auto [a, b] = merges_[static_cast<std::size_t>(best)];
    apply_merge(seq, a, b, merge_ids_.empty() ? nid + best : merge_ids_[static_cast<std::size_t>(best)]);
  }
  out.insert(out.end(), seq.begin(), seq.end());
  return {};
}

std::string BpeModel::to_json() const {
  nlohmann::json j;
  j["merges"] = nlohmann::json::array();
  for (auto [a, b] : merges_) j["merges"].push_back({a, b});
  if (!merge_ids_.empty()) j["merge_ids"] = merge_ids_;
  return j.dump();
}

Result<void> BpeModel::write_huggingface(const std::filesystem::path& dir, const Pretokenizer& pretok,
                                         const PieceTable& pieces) const {
  if (!merge_ids_.empty())
    return std::unexpected(make_error(Errc::unsupported,
                                      "Hugging Face export of an explicit-id BPE (e.g. SmolLM2) is not implemented"));
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  auto vocab_path = dir / "vocab.json";
  auto merges_path = dir / "merges.txt";
  auto tok_path = dir / "tokenizer.json";
  std::ofstream v(vocab_path, std::ios::binary);
  if (!v) return std::unexpected(make_error(Errc::io, "vocab.json"));
  // Byte-level tokenizers are written in the GPT-2 alphabet so other tools (and load_huggingface) read them.
  const bool byte_level = std::string(pretok.name()) == "bytelevel";
  auto key_text = [&](const std::string& raw) { return byte_level ? bytelevel::to_mapped(raw) : bytes_to_json_text(raw); };
  nlohmann::json vocab = nlohmann::json::object();
  for (std::size_t i = 0; i < pieces.pieces.size(); ++i) vocab[key_text(pieces.pieces[i])] = i;
  v << vocab.dump();
  std::ofstream m(merges_path, std::ios::binary);
  if (!m) return std::unexpected(make_error(Errc::io, "merges.txt"));
  m << "#version: 0.2\n";
  for (auto [a, b] : merges_) {
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= pieces.pieces.size() ||
        static_cast<std::size_t>(b) >= pieces.pieces.size())
      continue;
    m << (byte_level ? bytelevel::to_mapped(pieces.pieces[static_cast<std::size_t>(a)])
                     : pieces.pieces[static_cast<std::size_t>(a)])
      << ' '
      << (byte_level ? bytelevel::to_mapped(pieces.pieces[static_cast<std::size_t>(b)])
                     : pieces.pieces[static_cast<std::size_t>(b)])
      << '\n';
  }
  std::ofstream t(tok_path, std::ios::binary);
  if (!t) return std::unexpected(make_error(Errc::io, "tokenizer.json"));
  nlohmann::json tok;
  tok["version"] = "1.0";
  tok["truncation"] = nullptr;
  tok["padding"] = nullptr;
  tok["added_tokens"] = nlohmann::json::array();
  tok["normalizer"] = nullptr;
  tok["pre_tokenizer"] = {{"type", std::string(pretok.name()) == "bytelevel" ? "ByteLevel" : "Sequence"}};
  nlohmann::json model;
  model["type"] = "BPE";
  model["dropout"] = nullptr;
  model["unk_token"] = nullptr;
  model["vocab"] = vocab;
  model["merges"] = nlohmann::json::array();
  for (auto [a, b] : merges_) {
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= pieces.pieces.size() ||
        static_cast<std::size_t>(b) >= pieces.pieces.size())
      return std::unexpected(make_error(Errc::invalid_shape, "BPE export: merge id outside the vocab"));
    const auto& pa = pieces.pieces[static_cast<std::size_t>(a)];
    const auto& pb = pieces.pieces[static_cast<std::size_t>(b)];
    // Non-byte-level pieces may contain spaces ("o w"), so "a b" strings would be ambiguous: use the
    // [a, b] pair form (tokenizers >= 0.20 reads it; load_huggingface reads both).
    if (byte_level) model["merges"].push_back(bytelevel::to_mapped(pa) + " " + bytelevel::to_mapped(pb));
    else model["merges"].push_back(nlohmann::json::array({bytes_to_json_text(pa), bytes_to_json_text(pb)}));
  }
  tok["model"] = std::move(model);
  t << tok.dump();
  return {};
}

Result<std::unique_ptr<BpeModel>> BpeModel::from_merges(
    std::vector<std::pair<std::int32_t, std::int32_t>> merges, std::int64_t vocab) {
  auto m = std::make_unique<BpeModel>(std::move(merges));
  if (vocab > 0) m->vocab_size_ = vocab;
  return m;
}

PieceTable BpeModel::bytes_table() {
  PieceTable t;
  t.identity256 = true;
  t.pieces.resize(256);
  for (int i = 0; i < 256; ++i) {
    t.pieces[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(i));
    t.byte_to_id[static_cast<std::uint8_t>(i)] = i;
  }
  return t;
}

Result<std::pair<std::unique_ptr<BpeModel>, PieceTable>> BpeModel::chars_table(std::string_view text) {
  if (text.empty()) return std::unexpected(make_error(Errc::invalid_shape, "empty text"));
  std::set<unsigned char> uniq;
  for (unsigned char c : text) uniq.insert(c);
  PieceTable t;
  t.identity256 = false;
  t.pieces.reserve(uniq.size());
  std::int32_t i = 0;
  for (unsigned char c : uniq) {
    t.pieces.emplace_back(1, static_cast<char>(c));
    t.byte_to_id[c] = i++;
  }
  auto m = std::make_unique<BpeModel>();
  m->vocab_size_ = t.size();
  return std::pair{std::move(m), std::move(t)};
}

Result<std::pair<std::unique_ptr<BpeModel>, PieceTable>> BpeModel::train(
    std::string_view text, std::int32_t vocab_size, std::function<void(int, int)> progress) {
  if (text.empty()) return std::unexpected(make_error(Errc::invalid_shape, "empty text"));
  if (vocab_size < 256) return std::unexpected(make_error(Errc::invalid_shape, "vocab_size < 256"));
  if (vocab_size > 32000) return std::unexpected(make_error(Errc::unsupported, "vocab_size too large"));

  std::vector<std::int32_t> seq;
  seq.reserve(text.size());
  for (unsigned char c : text) seq.push_back(static_cast<std::int32_t>(c));

  std::vector<std::pair<std::int32_t, std::int32_t>> merges;
  const int n_merges = vocab_size - 256;
  merges.reserve(static_cast<std::size_t>(n_merges));
  for (int m = 0; m < n_merges; ++m) {
    std::unordered_map<std::pair<std::int32_t, std::int32_t>, std::int32_t, PairHash> counts;
    counts.reserve(seq.size() / 2 + 1);
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) counts[{seq[i], seq[i + 1]}]++;
    if (counts.empty()) break;
    auto best = counts.begin()->first;
    std::int32_t best_n = 0;
    for (auto& [k, n] : counts)
      if (n > best_n) {
        best_n = n;
        best = k;
      }
    if (best_n < 2) break;
    const auto id = 256 + static_cast<std::int32_t>(merges.size());
    merges.push_back(best);
    apply_merge(seq, best.first, best.second, id);
    if (progress) progress(m + 1, n_merges);
  }
  if (progress) progress(n_merges, n_merges);
  PieceTable t;
  rebuild_pieces_from_merges(t, merges);
  auto model = std::make_unique<BpeModel>(std::move(merges));
  model->vocab_size_ = t.size();
  return std::pair{std::move(model), std::move(t)};
}

Result<std::vector<std::pair<std::int32_t, std::int32_t>>> BpeModel::parse_merges(std::string_view json) {
  auto parsed = parse_json(json);
  if (!parsed) {
    parsed = parse_json(std::string("{") + std::string(json) + "}");
  }
  if (!parsed) return std::vector<std::pair<std::int32_t, std::int32_t>>{};
  if (!parsed->contains("merges") || !(*parsed)["merges"].is_array()) {
    return std::vector<std::pair<std::int32_t, std::int32_t>>{};
  }
  std::vector<std::pair<std::int32_t, std::int32_t>> m;
  for (auto& row : (*parsed)["merges"]) {
    if (row.is_array() && row.size() == 2 && row[0].is_number()) {
      m.emplace_back(row[0].get<std::int32_t>(), row[1].get<std::int32_t>());
    }
  }
  return m;
}

}  // namespace gyre
