#include "bytelevel.hpp"
#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/tiktoken.hpp"
#include "gyre/nn/unigram.hpp"

#include "json_parse.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
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

std::string Tokenizer::decode(std::span<const std::int32_t> ids) const { return pieces_.concat(ids); }

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
  auto parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary);
  if (!out) return std::unexpected(make_error(Errc::io, path.string()));
  auto j = to_json();
  out.write(j.data(), static_cast<std::streamsize>(j.size()));
  return {};
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
    std::vector<std::pair<std::string, std::int32_t>> added = {}) {
  if (!vocab.is_object() || vocab.empty())
    return std::unexpected(make_error(Errc::invalid_shape, "byte-level BPE: vocab must be an object"));
  const bool gpt2_pretok = !pretok || std::string(pretok->name()) == "bytelevel";
  if (!pretok) pretok = make_gpt2_pretok();
  std::int32_t maxid = -1;
  for (auto& [k, v] : vocab.items()) maxid = std::max(maxid, v.get<std::int32_t>());
  for (auto& [content, id] : added) maxid = std::max(maxid, id);
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

Result<std::unique_ptr<Tokenizer>> Tokenizer::from_json(std::string_view json) {
  auto p = parse_json(json);
  if (!p) return std::unexpected(p.error());
  auto& j = *p;
  if (j.contains("tokenizer") && j["tokenizer"].is_object()) return from_nested_json(j["tokenizer"]);
  if (j.contains("tokenizer") && j["tokenizer"].is_string() && j["tokenizer"] == "bytes") return bytes();
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
  if (j.contains("vocab") && j["vocab"].is_string() && j["vocab"] == "bytes") return bytes();
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

Result<std::unique_ptr<Tokenizer>> Tokenizer::load_huggingface(const std::filesystem::path& dir) {
  auto tokj = dir;
  if (std::filesystem::is_directory(dir)) tokj = dir / "tokenizer.json";
  if (std::filesystem::exists(tokj) && tokj.filename() == "tokenizer.json") {
    auto s = read_all(tokj);
    auto parsed = parse_json(s);
    if (!parsed) return std::unexpected(parsed.error());
    auto& j = *parsed;
    nlohmann::json model = j.contains("model") ? j["model"] : j;
    std::string type = model.value("type", std::string{});
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
    std::unique_ptr<Pretokenizer> pt;
    bool explicit_ids = false;  // non-GPT-2 byte-level layout: added tokens are honoured
    if (gyre_hf && ptype != "Sequence") {
      auto named = pretok_from_name(ptype);
      if (!named) return std::unexpected(named.error());
      pt = std::move(*named);
    } else if (ptype == "ByteLevel") pt = make_gpt2_pretok();
    else if (ptype == "Metaspace") pt = make_metaspace_pretok(true);
    else if (ptype == "Sequence" && j["pre_tokenizer"].value("pretokenizers", nlohmann::json::array()).empty()) {
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
    } else pt = make_identity_pretok();
    // Plain (not byte-level) BPE from another tool: pieces are checked against BpeModel's layout below.
    const bool foreign_bpe = type == "BPE" && ptype != "ByteLevel" && !gyre_hf;
    std::vector<std::pair<std::string, std::int32_t>> added;
    if (explicit_ids || foreign_bpe) {
      if (explicit_ids && j.contains("normalizer") && !j["normalizer"].is_null())
        return std::unexpected(make_error(Errc::unsupported, "tokenizer.json: normalizer is not supported"));
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
    if (type == "Unigram") {
      PieceTable table;
      std::vector<float> scores;
      if (model.contains("vocab") && model["vocab"].is_array()) {
        for (auto& row : model["vocab"]) {
          if (row.is_array() && row.size() >= 2) {
            table.pieces.push_back(piece_bytes(row[0].get<std::string>()));
            scores.push_back(row[1].get<float>());
          }
        }
      }
      auto um = UnigramModel::from_scores(std::move(scores), true);
      if (!um) return std::unexpected(um.error());
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
      if (std::string(pt->name()) == "bytelevel") return bytelevel_bpe_tokenizer(model["vocab"], mt);
      return bytelevel_bpe_tokenizer(model["vocab"], mt, std::move(pt), std::move(added));
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
          // "a b". A piece may itself hold spaces (Gyre's identity-pretok BPE), so try every space;
          // prefer the split whose result sits where the implicit layout puts it.
          const auto text = piece_bytes(m.get<std::string>());
          const auto at = nid + merges->size();
          for (auto sp = text.find(' '); sp != std::string::npos; sp = text.find(' ', sp + 1)) {
            auto a = inv.find(text.substr(0, sp));
            auto b = inv.find(text.substr(sp + 1));
            if (a == inv.end() || b == inv.end()) continue;
            const bool fits = at < table.pieces.size() && table.pieces[at] == a->first + b->first &&
                              static_cast<std::size_t>(a->second) < at && static_cast<std::size_t>(b->second) < at;
            if (ia < 0 || fits) {
              ia = a->second;
              ib = b->second;
            }
            if (fits) break;
          }
          if (ia < 0) return bad("merge parts not in vocab: " + m.get<std::string>());
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
    if (model.contains("unk_token") && model["unk_token"].is_string())
      if (auto u = inv.find(model["unk_token"].get<std::string>()); u != inv.end()) reachable[static_cast<std::size_t>(u->second)] = true;
    for (std::size_t i = 0; i < table.pieces.size(); ++i)
      if (!table.pieces[i].empty() && !reachable[i])
        return bad("vocab entry " + std::to_string(i) + " \"" + table.pieces[i] +
                   "\" is neither a single byte nor a merge result (character-level BPE is not supported)");
    auto bm = BpeModel::from_merges(std::move(*merges), table.size());
    if (!bm) return std::unexpected(bm.error());
    if (!implicit) (*bm)->set_merge_ids(std::move(merge_ids));
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
