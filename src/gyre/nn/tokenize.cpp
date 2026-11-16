#include "bytelevel.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/tiktoken.hpp"
#include "gyre/nn/unigram.hpp"

#include "atomic_file.hpp"
#include "json_parse.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace gyre {

// tiktoken.cpp: rebuilds a TiktokenModel tokenizer from its nested Gyre document.
Result<std::unique_ptr<Tokenizer>> tiktoken_from_doc(const nlohmann::json& obj);

std::string PieceTable::concat(std::span<const std::int32_t> ids) const {
  std::string s;
  for (auto id : ids) {
    if (id >= 0 && static_cast<std::size_t>(id) < pieces.size()) s += pieces[static_cast<std::size_t>(id)];
  }
  return s;
}

Tokenizer::Tokenizer(std::unique_ptr<Pretokenizer> pretok, std::unique_ptr<VocabModel> model,
                     PieceTable pieces)
    : pretok_(std::move(pretok)), model_(std::move(model)), pieces_(std::move(pieces)) {}

Result<void> Tokenizer::encode_plain(std::string_view text, std::vector<std::int32_t>& out,
                                     std::function<void(int, int)>* progress) const {
  std::string scratch;
  std::vector<std::string_view> spans;
  auto s = pretok_->split(text, scratch, spans);
  if (!s) return std::unexpected(s.error());
  const int n = static_cast<int>(spans.size());
  int k = 0;
  for (auto sp : spans) {
    auto e = model_->encode_span(sp, pieces_, out);
    if (!e) return std::unexpected(e.error());
    ++k;
    if (progress && *progress && (k == n || k % 32 == 0)) (*progress)(k, n == 0 ? 1 : n);
  }
  if (progress && *progress && n == 0) (*progress)(1, 1);
  return {};
}

Result<std::vector<std::int32_t>> Tokenizer::encode(std::string_view text,
                                                    std::function<void(int, int)> progress) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size());
  if (specials_.empty()) {
    auto r = encode_plain(text, out, &progress);
    if (!r) return std::unexpected(r.error());
    return out;
  }
  // Leftmost special first; at the same position the longest one wins.
  std::size_t pos = 0;
  while (pos <= text.size()) {
    std::size_t best_at = std::string_view::npos, best_len = 0;
    std::int32_t best_id = -1;
    for (auto& [content, id] : specials_) {
      if (content.empty()) continue;
      const auto at = text.find(content, pos);
      if (at == std::string_view::npos) continue;
      if (at < best_at || (at == best_at && content.size() > best_len)) {
        best_at = at;
        best_len = content.size();
        best_id = id;
      }
    }
    const auto end = best_at == std::string_view::npos ? text.size() : best_at;
    if (end > pos) {
      auto r = encode_plain(text.substr(pos, end - pos), out, nullptr);
      if (!r) return std::unexpected(r.error());
    }
    if (best_at == std::string_view::npos) break;
    out.push_back(best_id);
    pos = best_at + best_len;
  }
  if (progress) progress(1, 1);
  return out;
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids) const {
  auto s = pieces_.concat(ids);
  pretok_->decode(s);
  return s;
}

std::int64_t Tokenizer::vocab_size() const { return pieces_.size() ? pieces_.size() : model_->vocab_size(); }

const char* Tokenizer::model_name() const { return model_->name(); }
const char* Tokenizer::pretok_name() const { return pretok_->name(); }

std::string Tokenizer::to_json() const {
  nlohmann::json tok;
  auto fields = parse_json(model_->to_json());
  if (fields && fields->is_object()) tok = *fields;
  tok["pretoken"] = pretok_->name();
  tok["model"] = model_->name();
  const bool identity = pieces_.identity256 && pieces_.pieces.size() >= 256;
  if (!identity) {
    nlohmann::json vocab = nlohmann::json::array();
    for (auto& p : pieces_.pieces) vocab.push_back(bytes_to_json_text(p));
    tok["vocab"] = std::move(vocab);
  }
  if (!specials_.empty()) {
    nlohmann::json sp = nlohmann::json::array();
    for (auto& [content, id] : specials_) sp.push_back({{"id", id}, {"content", bytes_to_json_text(content)}});
    tok["special"] = std::move(sp);
  }
  nlohmann::json doc;
  doc["gyre"] = "tokenizer";
  doc["version"] = 1;
  doc["tokenizer"] = std::move(tok);
  return doc.dump();
}

Result<void> Tokenizer::save(const std::filesystem::path& path) const {
  auto j = to_json();
  return detail::write_file_atomic(path, [&](std::ofstream& out) -> Result<void> {
    out.write(j.data(), static_cast<std::streamsize>(j.size()));
    return {};
  });
}

Result<void> Tokenizer::save_huggingface(const std::filesystem::path& dir) const {
  return model_->write_huggingface(dir, *pretok_, pieces_);
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::bytes() {
  return std::make_unique<Tokenizer>(make_identity_pretok(), std::make_unique<BpeModel>(),
                                     BpeModel::bytes_table());
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::chars_from_text(std::string_view text) {
  auto p = BpeModel::chars_table(text);
  if (!p) return std::unexpected(p.error());
  return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(p->first), std::move(p->second));
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::train_bpe(std::string_view text, std::int32_t vocab_size,
                                                       std::function<void(int, int)> progress) {
  auto p = BpeModel::train(text, vocab_size, std::move(progress));
  if (!p) return std::unexpected(p.error());
  return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(p->first), std::move(p->second));
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::train_unigram(std::string_view text, std::int32_t vocab_size,
                                                           std::function<void(int, int)> progress) {
  auto p = UnigramModel::train(text, vocab_size, std::move(progress));
  if (!p) return std::unexpected(p.error());
  return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(p->first), std::move(p->second));
}

static PieceTable table_from_vocab_array(const nlohmann::json& arr) {
  PieceTable t;
  t.identity256 = false;
  if (!arr.is_array()) return t;
  t.pieces.reserve(arr.size());
  for (auto& x : arr) t.pieces.push_back(json_text_to_bytes(x.get<std::string>()));
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(t.pieces.size()); ++i) {
    if (t.pieces[static_cast<std::size_t>(i)].size() == 1)
      t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i)][0])] = i;
  }
  return t;
}

// `gyre_text`: keys were written by bytes_to_json_text (Gyre's own files, raw bytes 0x80-0xFF as
// U+0080-U+00FF). Otherwise keys are real UTF-8 text, which already is the raw bytes Gyre matches.
static PieceTable table_from_vocab_object(const nlohmann::json& obj, bool gyre_text = true) {
  PieceTable t;
  if (!obj.is_object()) return t;
  std::int32_t maxid = -1;
  for (auto& [k, v] : obj.items()) maxid = std::max(maxid, v.get<std::int32_t>());
  if (maxid < 0) return t;
  if (maxid >= kMaxTokenizerVocab) throw std::length_error("vocab id");
  t.pieces.assign(static_cast<std::size_t>(maxid + 1), {});
  t.identity256 = false;
  for (auto& [k, v] : obj.items()) {
    auto id = v.get<std::int32_t>();
    if (id >= 0) t.pieces[static_cast<std::size_t>(id)] = gyre_text ? json_text_to_bytes(k) : k;
  }
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(t.pieces.size()); ++i) {
    if (t.pieces[static_cast<std::size_t>(i)].size() == 1)
      t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i)][0])] = i;
  }
  return t;
}

// Hugging Face byte-level BPE: vocab keys and merges are in the GPT-2 byte->unicode alphabet. Gyre
// pieces are raw bytes, so decode them here.
//  * GPT-2 layout (ByteLevel pretokenizer, 256 single bytes, merge result = 256 + rank): built
//    exactly as before (implicit ids; added tokens such as <|endoftext|> stay plain text), so old
//    checkpoints and their frozen tokenizers are unchanged.
//  * Anything else (SmolLM2: 17 added tokens at 0..16, 235 bytes, merge results in vocab order):
//    explicit byte -> id and per-merge result ids read from the vocab, and `added` tokens matched
//    literally before pretokenizing. A merge whose parts or result are missing is an error.
static Result<std::unique_ptr<Tokenizer>> bytelevel_bpe_tokenizer(
    const nlohmann::json& vocab, const std::vector<std::pair<std::string, std::string>>& merge_text,
    std::unique_ptr<Pretokenizer> pretok = nullptr,
    std::vector<std::pair<std::string, std::int32_t>> added = {}, std::int32_t unk_id = -1,
    bool fuse_unk = false) {
  if (!vocab.is_object() || vocab.empty())
    return std::unexpected(make_error(Errc::invalid_shape, "byte-level BPE: vocab must be an object"));
  const bool gpt2_pretok = !pretok || std::string(pretok->name()) == "bytelevel";
  if (!pretok) pretok = make_gpt2_pretok();
  std::int32_t maxid = -1;
  for (auto& [k, v] : vocab.items()) maxid = std::max(maxid, v.get<std::int32_t>());
  for (auto& [content, id] : added) maxid = std::max(maxid, id);
  if (maxid >= kMaxTokenizerVocab)
    return std::unexpected(make_error(Errc::ckpt_corrupt, "byte-level BPE: token id too large"));
  PieceTable t;
  t.identity256 = false;
  t.pieces.assign(static_cast<std::size_t>(maxid + 1), {});
  std::vector<bool> is_added(t.pieces.size(), false);
  for (auto& [content, id] : added) {
    if (id < 0) return std::unexpected(make_error(Errc::invalid_shape, "byte-level BPE: negative added id"));
    t.pieces[static_cast<std::size_t>(id)] = content;
    is_added[static_cast<std::size_t>(id)] = true;
  }
  std::unordered_map<std::string, std::int32_t> inv;
  for (auto& [k, v] : vocab.items()) {
    const auto id = v.get<std::int32_t>();
    if (id < 0) continue;
    if (is_added[static_cast<std::size_t>(id)]) continue;  // literal content set above
    std::string raw;
    if (!bytelevel::from_mapped(k, raw)) raw = k;  // special tokens such as <|endoftext|>
    t.pieces[static_cast<std::size_t>(id)] = raw;
    inv.emplace(raw, id);
  }
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(t.pieces.size()); ++i)
    if (!is_added[static_cast<std::size_t>(i)] && t.pieces[static_cast<std::size_t>(i)].size() == 1)
      t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i)][0])] = i;

  // Legacy GPT-2 numbering? (the only layout the loader accepted before explicit ids)
  bool legacy = gpt2_pretok && added.empty() && t.byte_to_id.size() == 256;
  if (legacy) {
    std::size_t rank = 0;
    for (auto& [sa, sb] : merge_text) {
      std::string a, b;
      if (!bytelevel::from_mapped(sa, a) || !bytelevel::from_mapped(sb, b)) continue;
      if (!inv.contains(a) || !inv.contains(b)) continue;
      const auto result_id = 256 + rank++;
      if (result_id >= t.pieces.size() || t.pieces[result_id] != a + b) {
        legacy = false;
        break;
      }
    }
  }
  std::vector<std::pair<std::int32_t, std::int32_t>> merges;
  if (legacy) {
    for (auto& [sa, sb] : merge_text) {
      std::string a, b;
      if (!bytelevel::from_mapped(sa, a) || !bytelevel::from_mapped(sb, b)) continue;
      auto ia = inv.find(a);
      auto ib = inv.find(b);
      if (ia == inv.end() || ib == inv.end()) continue;
      merges.emplace_back(ia->second, ib->second);
    }
    auto bm = BpeModel::from_merges(std::move(merges), t.size());
    if (!bm) return std::unexpected(bm.error());
    if (unk_id >= 0) (*bm)->set_unk(unk_id, fuse_unk, /*per_byte=*/true);
    return std::make_unique<Tokenizer>(std::move(pretok), std::move(*bm), std::move(t));
  }
  std::vector<std::int32_t> merge_ids;
  merges.reserve(merge_text.size());
  merge_ids.reserve(merge_text.size());
  for (auto& [sa, sb] : merge_text) {
    std::string a, b;
    if (!bytelevel::from_mapped(sa, a) || !bytelevel::from_mapped(sb, b))
      return std::unexpected(
          make_error(Errc::unsupported, "byte-level BPE: merge outside the byte alphabet: " + sa + " " + sb));
    auto ia = inv.find(a);
    auto ib = inv.find(b);
    auto ic = inv.find(a + b);
    if (ia == inv.end() || ib == inv.end() || ic == inv.end())
      return std::unexpected(make_error(Errc::unsupported, "byte-level BPE: merge not in vocab: " + sa + " " + sb));
    merges.emplace_back(ia->second, ib->second);
    merge_ids.push_back(ic->second);
  }
  auto bm = BpeModel::from_merges(std::move(merges), t.size());
  if (!bm) return std::unexpected(bm.error());
  (*bm)->set_merge_ids(std::move(merge_ids));
  if (unk_id >= 0) (*bm)->set_unk(unk_id, fuse_unk, /*per_byte=*/true);  // a ByteLevel character is one byte
  auto tok = std::make_unique<Tokenizer>(std::move(pretok), std::move(*bm), std::move(t));
  tok->set_specials(std::move(added));
  return tok;
}

// Without explicit merge ids, BpeModel::encode_span numbers the result of merge `rank` as
// alphabet + rank (alphabet = 256 for identity256, else byte_to_id.size()). True when the table puts
// pieces[a] + pieces[b] exactly there for every merge, i.e. encode and decode agree.
static bool implicit_merge_layout(const PieceTable& t,
                                  const std::vector<std::pair<std::int32_t, std::int32_t>>& merges) {
  const std::size_t nid = t.identity256 ? 256 : t.byte_to_id.size();
  for (std::size_t r = 0; r < merges.size(); ++r) {
    const auto [a, b] = merges[r];
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= t.pieces.size() ||
        static_cast<std::size_t>(b) >= t.pieces.size())
      return false;
    if (nid + r >= t.pieces.size() ||
        t.pieces[nid + r] != t.pieces[static_cast<std::size_t>(a)] + t.pieces[static_cast<std::size_t>(b)])
      return false;
  }
  return true;
}

static Result<PieceTable> pieces_from_obj(const nlohmann::json& obj,
                                          const std::vector<std::pair<std::int32_t, std::int32_t>>& merges) {
  if (obj.contains("vocab") && obj["vocab"].is_array()) return table_from_vocab_array(obj["vocab"]);
  if (obj.contains("vocab") && obj["vocab"].is_object()) return table_from_vocab_object(obj["vocab"]);
  if (obj.contains("pieces") && obj["pieces"].is_array() && !obj["pieces"].empty()) {
    auto t = table_from_vocab_array(obj["pieces"]);
    t.identity256 = t.pieces.size() >= 256;
    if (t.identity256) {
      for (int i = 0; i < 256; ++i) t.byte_to_id[static_cast<std::uint8_t>(i)] = i;
    }
    return t;
  }
  PieceTable t = BpeModel::bytes_table();
  for (auto [a, b] : merges) {
    if (static_cast<std::size_t>(a) < t.pieces.size() && static_cast<std::size_t>(b) < t.pieces.size())
      t.pieces.push_back(t.pieces[static_cast<std::size_t>(a)] + t.pieces[static_cast<std::size_t>(b)]);
  }
  return t;
}

static Result<std::unique_ptr<Tokenizer>> from_nested_json(const nlohmann::json& obj) {
  std::string pret = obj.value("pretoken", obj.value("pretok", std::string{"identity"}));
  std::string model = obj.value("model", std::string{"bpe"});
  if (model == "tiktoken_bpe") return tiktoken_from_doc(obj);
  if (model != "bpe" && model != "unigram")
    return std::unexpected(make_error(Errc::unsupported, "tokenizer: unknown model \"" + model + "\""));
  auto pt = pretok_from_name(pret);
  if (!pt) return std::unexpected(pt.error());
  if (model == "unigram") {
    std::vector<float> scores;
    if (obj.contains("scores") && obj["scores"].is_array()) scores = obj["scores"].get<std::vector<float>>();
    PieceTable t;
    if (obj.contains("vocab") && obj["vocab"].is_array()) t = table_from_vocab_array(obj["vocab"]);
    else if (obj.contains("pieces") && obj["pieces"].is_array()) t = table_from_vocab_array(obj["pieces"]);
    bool fallback = obj.value("byte_fallback", false);
    auto um = UnigramModel::from_scores(std::move(scores), fallback);
    if (!um) return std::unexpected(um.error());
    if (obj.contains("unk_id")) {
      const auto unk = obj["unk_id"].get<std::int32_t>();
      if (unk < -1 || unk >= t.size()) return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: unk_id out of range"));
      (*um)->set_unk_id(unk);
    }
    if (obj.contains("piece_types")) {
      auto types = obj["piece_types"].get<std::vector<std::uint8_t>>();
      if (types.size() != t.pieces.size())
        return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: piece_types size != vocab size"));
      (*um)->set_piece_types(std::move(types));
    }
    return std::make_unique<Tokenizer>(std::move(*pt), std::move(*um), std::move(t));
  }
  auto merges = BpeModel::parse_merges(obj.dump());
  if (!merges) return std::unexpected(merges.error());
  auto table = pieces_from_obj(obj, *merges);
  if (!table) return std::unexpected(table.error());
  std::vector<std::int32_t> merge_ids;
  if (obj.contains("merge_ids") && obj["merge_ids"].is_array()) {
    merge_ids = obj["merge_ids"].get<std::vector<std::int32_t>>();
    if (merge_ids.size() != merges->size())
      return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: merge_ids size != merges size"));
    for (auto id : merge_ids)
      if (id < 0 || id >= table->size())
        return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: merge id out of range"));
  } else if (!implicit_merge_layout(*table, *merges)) {
    return std::unexpected(make_error(Errc::ckpt_corrupt,
                                      "tokenizer: merge results are not at alphabet + rank and there are no merge_ids"));
  }
  auto bm = BpeModel::from_merges(std::move(*merges), table->size());
  if (!bm) return std::unexpected(bm.error());
  if (!merge_ids.empty()) (*bm)->set_merge_ids(std::move(merge_ids));
  if (obj.contains("unk_id") && !obj["unk_id"].is_null()) {
    const auto unk = obj["unk_id"].get<std::int32_t>();
    if (unk < 0 || unk >= table->size())
      return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: unk_id out of range"));
    (*bm)->set_unk(unk, obj.value("fuse_unk", false), obj.value("unk_per_byte", false));
  }
  auto tok = std::make_unique<Tokenizer>(std::move(*pt), std::move(*bm), std::move(*table));
  if (obj.contains("special") && obj["special"].is_array()) {
    std::vector<std::pair<std::string, std::int32_t>> sp;
    try {
      for (auto& e : obj["special"]) {
        const auto id = e.at("id").get<std::int32_t>();
        if (id < 0 || id >= tok->vocab_size())
          return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: special id out of range"));
        sp.emplace_back(json_text_to_bytes(e.at("content").get<std::string>()), id);
      }
    } catch (const nlohmann::json::exception& ex) {
      return std::unexpected(make_error(Errc::ckpt_corrupt, std::string("tokenizer special: ") + ex.what()));
    }
    tok->set_specials(std::move(sp));
  }
  return tok;
}

// Tokenizer files are untrusted: a wrong-typed field (nlohmann type_error) or a size that cannot
// be allocated must come back as an error, not escape the Result API.
template <class F>
static Result<std::unique_ptr<Tokenizer>> guard_tokenizer_load(const char* what, F&& f) {
  try {
    return f();
  } catch (const nlohmann::json::exception& e) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string(what) + ": " + e.what()));
  } catch (const std::bad_alloc&) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string(what) + ": too large"));
  } catch (const std::length_error&) {
    return std::unexpected(make_error(Errc::ckpt_corrupt, std::string(what) + ": too large"));
  }
}

static Result<std::unique_ptr<Tokenizer>> from_json_impl(std::string_view json);
static Result<std::unique_ptr<Tokenizer>> load_huggingface_impl(const std::filesystem::path& dir);

Result<std::unique_ptr<Tokenizer>> Tokenizer::from_json(std::string_view json) {
  return guard_tokenizer_load("tokenizer", [&] { return from_json_impl(json); });
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::load_huggingface(const std::filesystem::path& dir) {
  return guard_tokenizer_load("tokenizer.json", [&] { return load_huggingface_impl(dir); });
}

static Result<std::unique_ptr<Tokenizer>> from_json_impl(std::string_view json) {
  auto p = parse_json(json);
  if (!p) return std::unexpected(p.error());
  auto& j = *p;
  if (j.contains("tokenizer") && j["tokenizer"].is_object()) return from_nested_json(j["tokenizer"]);
  if (j.contains("tokenizer") && j["tokenizer"].is_string() && j["tokenizer"] == "bytes") return Tokenizer::bytes();
  if ((j.contains("tokenizer") && j["tokenizer"] == "bpe") || j.contains("merges")) {
    auto merges = BpeModel::parse_merges(std::string(json));
    if (!merges) return std::unexpected(merges.error());
    auto table = pieces_from_obj(j, *merges);
    if (!table) return std::unexpected(table.error());
    if (!implicit_merge_layout(*table, *merges))
      return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer: merge results are not at alphabet + rank"));
    auto bm = BpeModel::from_merges(std::move(*merges), table->size());
    if (!bm) return std::unexpected(bm.error());
    return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(*bm), std::move(*table));
  }
  if (j.contains("vocab") && j["vocab"].is_string() && j["vocab"] == "bytes") return Tokenizer::bytes();
  if (j.contains("vocab") && j["vocab"].is_array()) {
    auto t = table_from_vocab_array(j["vocab"]);
    auto bm = std::make_unique<BpeModel>();
    bm->set_vocab_size(t.size());
    return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(bm), std::move(t));
  }
  return std::unexpected(make_error(Errc::ckpt_corrupt, "no tokenizer"));
}

static std::string read_all(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::load(const std::filesystem::path& path) {
  if (std::filesystem::is_directory(path)) return load_huggingface(path);
  auto ext = path.extension().string();
  if (ext == ".model") return load_sentencepiece(path);
  auto s = read_all(path);
  if (s.empty()) return std::unexpected(make_error(Errc::io, path.string()));
  if (is_tiktoken_tok_json(s)) return load_tiktoken_json(s);
  if (s.find("\"model\"") != std::string::npos && s.find("\"type\"") != std::string::npos &&
      s.find("\"gyre\"") == std::string::npos)
    return load_huggingface(path);
  return from_json(s);
}

// null, or a Sequence whose members are all no-ops (an empty Sequence changes nothing).
static bool normalizer_is_noop(const nlohmann::json& n) {
  if (n.is_null()) return true;
  if (!n.is_object() || n.value("type", std::string{}) != "Sequence") return false;
  if (!n.contains("normalizers") || !n["normalizers"].is_array()) return false;
  for (auto& m : n["normalizers"])
    if (!normalizer_is_noop(m)) return false;
  return true;
}

static Result<std::unique_ptr<Tokenizer>> load_huggingface_impl(const std::filesystem::path& dir) {
  auto tokj = dir;
  if (std::filesystem::is_directory(dir)) tokj = dir / "tokenizer.json";
  if (std::filesystem::exists(tokj) && tokj.filename() == "tokenizer.json") {
    auto s = read_all(tokj);
    auto parsed = parse_json(s);
    if (!parsed) return std::unexpected(parsed.error());
    auto& j = *parsed;
    nlohmann::json model = j.contains("model") ? j["model"] : j;
    std::string type = model.value("type", std::string{});
    if (!type.empty() && type != "BPE" && type != "Unigram")  // WordPiece, WordLevel, ...
      return std::unexpected(make_error(Errc::unsupported, "tokenizer.json: model type \"" + type + "\" is not supported"));
    std::string ptype;
    if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object())
      ptype = j["pre_tokenizer"].value("type", std::string{});
    // Non-byte-level pieces: Hugging Face writes real UTF-8 text, which already is the raw bytes
    // Gyre matches. Gyre's own save_huggingface writes raw bytes 0x80-0xFF as U+0080-U+00FF
    // (bytes_to_json_text) and marks the file with a pre_tokenizer Hugging Face never writes: a
    // "Sequence" without "pretokenizers" (BPE) or a lower-case Gyre pretokenizer name (Unigram).
    const bool gyre_hf = j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object() &&
                         ((ptype == "Sequence" && !j["pre_tokenizer"].contains("pretokenizers")) ||
                          (!ptype.empty() && ptype[0] >= 'a' && ptype[0] <= 'z'));
    auto piece_bytes = [&](const std::string& text) { return gyre_hf ? json_text_to_bytes(text) : text; };
    auto unsupported = [](const std::string& what) {
      return std::unexpected(make_error(Errc::unsupported, "tokenizer.json: " + what));
    };
    // Gyre encodes the raw text: a normalizer (NFC, Lowercase, BertNormalizer, Precompiled, ...)
    // would change what Hugging Face encodes, so refuse it instead of silently skipping it.
    if (j.contains("normalizer") && !normalizer_is_noop(j["normalizer"]))
      return unsupported("normalizer " + j["normalizer"].dump().substr(0, 200) + " is not supported");
    const bool pre_null = !j.contains("pre_tokenizer") || j["pre_tokenizer"].is_null();
    if (!pre_null && !j["pre_tokenizer"].is_object()) return unsupported("pre_tokenizer must be an object or null");
    std::unique_ptr<Pretokenizer> pt;
    bool explicit_ids = false;  // non-GPT-2 byte-level layout: added tokens are honoured
    bool metaspace_first = false;  // prepend_scheme "first": only the first section gets the prefix
    if (gyre_hf && ptype != "Sequence") {
      auto named = pretok_from_name(ptype);
      if (!named) return std::unexpected(named.error());
      pt = std::move(*named);
    } else if (ptype == "ByteLevel") {
      // GPT-2: add_prefix_space false, use_regex true. Older Gyre exports wrote only the type, so a
      // missing add_prefix_space reads as false here (Hugging Face itself would default it to true).
      const auto& p = j["pre_tokenizer"];
      if (p.value("add_prefix_space", false) || !p.value("use_regex", true))
        return unsupported("ByteLevel pre_tokenizer with add_prefix_space=true or use_regex=false is not supported");
      pt = make_gpt2_pretok();
    } else if (ptype == "Metaspace") {
      const auto& p = j["pre_tokenizer"];
      if (p.contains("replacement") && p["replacement"] != "\xe2\x96\x81")
        return unsupported("Metaspace replacement other than U+2581 is not supported");
      bool dummy = p.value("add_prefix_space", true);  // tokenizers < 0.19
      if (p.contains("prepend_scheme")) {
        const auto scheme = p["prepend_scheme"].get<std::string>();
        if (scheme != "always" && scheme != "first" && scheme != "never")
          return unsupported("Metaspace prepend_scheme \"" + scheme + "\" is not supported");
        dummy = scheme != "never";
        metaspace_first = scheme == "first";
      }
      pt = make_metaspace_pretok(dummy, p.value("split", true));
    } else if (ptype == "Sequence" && j["pre_tokenizer"].value("pretokenizers", nlohmann::json::array()).empty()) {
      pt = make_identity_pretok();  // what save_huggingface writes for Gyre's identity pretokenizer
    } else if (ptype == "Sequence") {
      // Only Digits(individual_digits) followed by ByteLevel(use_regex, no prefix space) is known.
      const auto seq = j["pre_tokenizer"].value("pretokenizers", nlohmann::json::array());
      const bool ok = seq.is_array() && seq.size() == 2 && seq[0].is_object() && seq[1].is_object() &&
                      seq[0].value("type", std::string{}) == "Digits" && seq[0].value("individual_digits", false) &&
                      seq[1].value("type", std::string{}) == "ByteLevel" && seq[1].value("use_regex", true) &&
                      !seq[1].value("add_prefix_space", false);
      if (!ok)
        return std::unexpected(make_error(Errc::unsupported,
                                          "tokenizer.json: unsupported pre_tokenizer Sequence " + seq.dump()));
      pt = make_digits_bytelevel_pretok();
      ptype = "ByteLevel";  // same byte-level vocab handling below
      explicit_ids = true;
    } else if (pre_null) {
      pt = make_identity_pretok();  // no pre_tokenizer: the whole text is one word
    } else {
      // Whitespace, WhitespaceSplit, BertPreTokenizer, Punctuation, Split, CharDelimiterSplit, ...
      // Encoding with the identity pretokenizer instead would give different ids.
      return unsupported("pre_tokenizer type \"" + ptype + "\" is not supported");
    }
    // Plain (not byte-level) BPE from another tool: pieces are checked against BpeModel's layout below.
    const bool foreign_bpe = type == "BPE" && ptype != "ByteLevel" && !gyre_hf;
    // BPE model options Gyre does not implement change the ids; refuse them. (GPT-2 writes "" for
    // the prefix/suffix, SmolLM2 null.)
    const bool bpe_model = type == "BPE" || (type.empty() && ptype == "ByteLevel");
    if (bpe_model) {
      auto set_str = [&](const char* k) {
        return model.contains(k) && model[k].is_string() && !model[k].get<std::string>().empty();
      };
      if (set_str("continuing_subword_prefix") || set_str("end_of_word_suffix"))
        return unsupported("BPE continuing_subword_prefix / end_of_word_suffix is not supported");
      if (model.value("byte_fallback", false)) return unsupported("BPE byte_fallback is not supported");
      if (model.value("ignore_merges", false)) return unsupported("BPE ignore_merges is not supported");
      if (model.contains("dropout") && model["dropout"].is_number() && model["dropout"].get<double>() > 0.0)
        return unsupported("BPE dropout is not supported");
    }
    // unk_token: id of the token an unencodable character becomes (null = dropped, as Hugging Face).
    std::int32_t unk_id = -1;
    if (bpe_model && model.contains("unk_token") && model["unk_token"].is_string()) {
      const auto u = model["unk_token"].get<std::string>();
      if (!model.contains("vocab") || !model["vocab"].is_object() || !model["vocab"].contains(u) ||
          !model["vocab"][u].is_number_integer())
        return unsupported("BPE unk_token \"" + u + "\" is not in the vocab");
      unk_id = model["vocab"][u].get<std::int32_t>();
    }
    const bool fuse_unk = model.value("fuse_unk", false);
    std::vector<std::pair<std::string, std::int32_t>> added;
    if (explicit_ids || foreign_bpe) {
      if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (auto& a : j["added_tokens"]) {
          if (!a.is_object() || !a.contains("id") || !a.contains("content")) continue;
          if (a.value("lstrip", false) || a.value("rstrip", false) || a.value("single_word", false))
            return std::unexpected(make_error(
                Errc::unsupported, "tokenizer.json: added token lstrip/rstrip/single_word unsupported: " + a.dump()));
          added.emplace_back(a["content"].get<std::string>(), a["id"].get<std::int32_t>());
        }
      }
    }
    // Gyre encodes the text between added tokens section by section, each with the dummy prefix
    // ("always"); "first" differs as soon as an added token splits the text.
    if (metaspace_first && !added.empty())
      return unsupported("Metaspace prepend_scheme \"first\" together with added tokens is not supported");
    if (type == "Unigram") {
      PieceTable table;
      std::vector<float> scores;
      // byte_fallback models name the byte pieces <0xNN>: keep them as the raw byte, typed BYTE,
      // so unknown characters fall back to them (as Hugging Face does) instead of to unk_id.
      const bool byte_fallback = model.value("byte_fallback", false);
      std::vector<std::uint8_t> types;
      auto byte_piece = [](const std::string& s, char& out) {
        auto hex = [](char c) -> int {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'A' && c <= 'F') return c - 'A' + 10;
          return -1;
        };
        if (s.size() != 6 || s.compare(0, 3, "<0x") != 0 || s[5] != '>') return false;
        const int hi = hex(s[3]), lo = hex(s[4]);
        if (hi < 0 || lo < 0) return false;
        out = static_cast<char>(hi * 16 + lo);
        return true;
      };
      if (model.contains("vocab") && model["vocab"].is_array()) {
        for (auto& row : model["vocab"]) {
          if (row.is_array() && row.size() >= 2) {
            const auto text = row[0].get<std::string>();
            char raw = 0;
            if (byte_fallback && byte_piece(text, raw)) {
              table.pieces.emplace_back(1, raw);
              types.push_back(UnigramModel::kByte);
            } else {
              table.pieces.push_back(piece_bytes(text));
              types.push_back(UnigramModel::kNormal);
            }
            scores.push_back(row[1].get<float>());
          }
        }
      }
      // unk_id null: Hugging Face refuses text no piece covers; Gyre returns an error for it.
      std::int32_t unk = -1;
      if (model.contains("unk_id") && !model["unk_id"].is_null()) unk = model["unk_id"].get<std::int32_t>();
      if (unk < -1 || unk >= table.size())
        return std::unexpected(make_error(Errc::ckpt_corrupt, "tokenizer.json: unk_id out of range"));
      auto um = UnigramModel::from_scores(std::move(scores), true);
      if (!um) return std::unexpected(um.error());
      (*um)->set_unk_id(unk);
      if (byte_fallback) (*um)->set_piece_types(std::move(types));
      return std::make_unique<Tokenizer>(std::move(pt), std::move(*um), std::move(table));
    }
    if (ptype == "ByteLevel" && model.contains("vocab") && model["vocab"].is_object()) {
      std::vector<std::pair<std::string, std::string>> mt;
      if (model.contains("merges") && model["merges"].is_array()) {
        for (auto& m : model["merges"]) {
          if (m.is_string()) {
            auto s2 = m.get<std::string>();
            auto sp = s2.find(' ');
            if (sp != std::string::npos) mt.emplace_back(s2.substr(0, sp), s2.substr(sp + 1));
          } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
            mt.emplace_back(m[0].get<std::string>(), m[1].get<std::string>());
          }
        }
      }
      if (std::string(pt->name()) == "bytelevel") return bytelevel_bpe_tokenizer(model["vocab"], mt, nullptr, {}, unk_id, fuse_unk);
      return bytelevel_bpe_tokenizer(model["vocab"], mt, std::move(pt), std::move(added), unk_id, fuse_unk);
    }
    auto merges = BpeModel::parse_merges(model.dump());
    if (!merges) return std::unexpected(merges.error());
    PieceTable table;
    if (model.contains("vocab") && model["vocab"].is_object())
      table = table_from_vocab_object(model["vocab"], gyre_hf);
    if (table.pieces.empty()) {  // no vocab: id merges over the 256 bytes
      table = BpeModel::bytes_table();
      auto bm = BpeModel::from_merges(std::move(*merges), table.size());
      if (!bm) return std::unexpected(bm.error());
      return std::make_unique<Tokenizer>(std::move(pt), std::move(*bm), std::move(table));
    }
    auto bad = [](const std::string& what) {
      return std::unexpected(make_error(Errc::unsupported, "tokenizer.json (BPE): " + what));
    };
    for (auto& [content, id] : added) {  // added tokens may sit past the model vocab
      if (id < 0) return bad("negative added token id");
      if (id >= kMaxTokenizerVocab) return bad("added token id too large");
      if (static_cast<std::size_t>(id) >= table.pieces.size()) table.pieces.resize(static_cast<std::size_t>(id) + 1);
      if (table.pieces[static_cast<std::size_t>(id)].empty()) table.pieces[static_cast<std::size_t>(id)] = content;
    }
    std::unordered_map<std::string, std::int32_t> inv;
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(table.pieces.size()); ++i)
      if (!table.pieces[static_cast<std::size_t>(i)].empty()) inv[table.pieces[static_cast<std::size_t>(i)]] = i;
    const std::size_t nid = table.byte_to_id.size();  // where BpeModel numbers implicit merge results
    if (merges->empty() && model.contains("merges") && model["merges"].is_array()) {
      for (auto& m : model["merges"]) {
        std::int32_t ia = -1, ib = -1;
        if (m.is_string()) {
          // "a b". A piece may itself hold spaces (old Gyre exports of identity-pretok BPE), so every
          // space is a candidate split. Keep the splits whose parts are both in the vocab, then
          // narrow down: (1) the result sits where the implicit layout puts it (alphabet + rank,
          // parts before it), else (2) the result a+b is in the vocab at all. More than one
          // survivor is truly ambiguous (e.g. "a  b" = "a"+" b" = "a "+"b") and an error: picking
          // one would silently change the merge ranks. New exports write [a, b] pairs instead.
          const auto text = piece_bytes(m.get<std::string>());
          const auto at = nid + merges->size();
          struct Cand {
            std::int32_t a, b;
            bool fits, result_in_vocab;
          };
          std::vector<Cand> cands;
          for (auto sp = text.find(' '); sp != std::string::npos; sp = text.find(' ', sp + 1)) {
            auto a = inv.find(text.substr(0, sp));
            auto b = inv.find(text.substr(sp + 1));
            if (a == inv.end() || b == inv.end()) continue;
            const auto joined = a->first + b->first;
            const bool fits = at < table.pieces.size() && table.pieces[at] == joined &&
                              static_cast<std::size_t>(a->second) < at && static_cast<std::size_t>(b->second) < at;
            cands.push_back({a->second, b->second, fits, inv.contains(joined)});
          }
          if (cands.empty()) return bad("merge parts not in vocab: " + m.get<std::string>());
          auto pick = [&](auto keep) -> int {  // index of the single candidate kept, -1 none, -2 several
            int found = -1;
            for (int c = 0; c < static_cast<int>(cands.size()); ++c) {
              if (!keep(cands[static_cast<std::size_t>(c)])) continue;
              if (found >= 0) return -2;
              found = c;
            }
            return found;
          };
          int c = pick([](const Cand& x) { return x.fits; });
          if (c == -1) c = pick([](const Cand& x) { return x.result_in_vocab; });
          if (c == -1) c = cands.size() == 1 ? 0 : -2;  // result missing: reported below
          if (c == -2) return bad("ambiguous merge \"" + m.get<std::string>() +
                                  "\": more than one split of its spaces matches the vocab");
          ia = cands[static_cast<std::size_t>(c)].a;
          ib = cands[static_cast<std::size_t>(c)].b;
        } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
          auto a = inv.find(piece_bytes(m[0].get<std::string>()));
          auto b = inv.find(piece_bytes(m[1].get<std::string>()));
          if (a == inv.end() || b == inv.end()) return bad("merge parts not in vocab: " + m.dump());
          ia = a->second;
          ib = b->second;
        } else {
          return bad("bad merge entry: " + m.dump());
        }
        merges->emplace_back(ia, ib);
      }
    }
    // BpeModel numbers merge results alphabet + rank unless it has explicit ids. Keep that (Gyre's
    // own export) when the vocab agrees; otherwise (specials first, other numbering) use the vocab's ids.
    std::vector<std::int32_t> merge_ids;
    const bool implicit = implicit_merge_layout(table, *merges);
    if (!implicit) {
      merge_ids.reserve(merges->size());
      for (auto [a, b] : *merges) {
        if (a < 0 || b < 0 || a >= table.size() || b >= table.size()) return bad("merge id out of range");
        auto c = inv.find(table.pieces[static_cast<std::size_t>(a)] + table.pieces[static_cast<std::size_t>(b)]);
        if (c == inv.end()) return bad("merge result not in vocab");
        merge_ids.push_back(c->second);
      }
    }
    // Encode starts from single bytes and can only build merge results. Any other vocab entry (a
    // multi-byte character of a character-level alphabet, a "<0x41>" byte-fallback piece, ...) is
    // one Hugging Face would emit and Gyre never could: refuse instead of encoding differently.
    std::vector<bool> reachable(table.pieces.size(), false);
    for (auto& [byte, id] : table.byte_to_id) reachable[static_cast<std::size_t>(id)] = true;
    for (std::size_t r = 0; r < merges->size(); ++r) reachable[implicit ? nid + r : static_cast<std::size_t>(merge_ids[r])] = true;
    for (auto& [content, id] : added)
      if (id >= 0 && id < table.size()) reachable[static_cast<std::size_t>(id)] = true;
    if (unk_id >= 0 && unk_id < table.size()) reachable[static_cast<std::size_t>(unk_id)] = true;
    for (std::size_t i = 0; i < table.pieces.size(); ++i)
      if (!table.pieces[i].empty() && !reachable[i])
        return bad("vocab entry " + std::to_string(i) + " \"" + table.pieces[i] +
                   "\" is neither a single byte nor a merge result (character-level BPE is not supported)");
    auto bm = BpeModel::from_merges(std::move(*merges), table.size());
    if (!bm) return std::unexpected(bm.error());
    if (!implicit) (*bm)->set_merge_ids(std::move(merge_ids));
    if (unk_id >= 0) {
      if (unk_id >= table.size()) return bad("unk_token id out of range");
      (*bm)->set_unk(unk_id, fuse_unk, /*per_byte=*/false);  // plain BPE: unknown = one UTF-8 character
    }
    auto tok = std::make_unique<Tokenizer>(std::move(pt), std::move(*bm), std::move(table));
    if (!added.empty()) tok->set_specials(std::move(added));
    return tok;
  }
  auto vocabp = std::filesystem::is_directory(dir) ? dir / "vocab.json" : dir;
  auto mergesp = std::filesystem::is_directory(dir) ? dir / "merges.txt" : dir.parent_path() / "merges.txt";
  auto vs = read_all(vocabp);
  if (vs.empty()) return std::unexpected(make_error(Errc::io, "vocab.json"));
  auto vj = parse_json(vs);
  if (!vj) return std::unexpected(vj.error());
  // GPT-2 layout (vocab.json + merges.txt, GPT-2 byte->unicode alphabet)
  auto ms = read_all(mergesp);
  std::istringstream iss(ms);
  std::string line;
  std::vector<std::pair<std::string, std::string>> mt;
  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    auto sp = line.find(' ');
    if (sp == std::string::npos) continue;
    mt.emplace_back(line.substr(0, sp), line.substr(sp + 1));
  }
  return bytelevel_bpe_tokenizer(*vj, mt);
}

}  // namespace gyre
