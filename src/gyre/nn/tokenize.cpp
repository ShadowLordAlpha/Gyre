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

Result<std::vector<std::int32_t>> Tokenizer::encode(std::string_view text,
                                                    std::function<void(int, int)> progress) const {
  std::string scratch;
  std::vector<std::string_view> spans;
  auto s = pretok_->split(text, scratch, spans);
  if (!s) return std::unexpected(s.error());
  std::vector<std::int32_t> out;
  out.reserve(text.size());
  const int n = static_cast<int>(spans.size());
  int k = 0;
  for (auto sp : spans) {
    auto e = model_->encode_span(sp, pieces_, out);
    if (!e) return std::unexpected(e.error());
    ++k;
    if (progress && (k == n || k % 32 == 0)) progress(k, n == 0 ? 1 : n);
  }
  if (progress && n == 0) progress(1, 1);
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

static PieceTable table_from_vocab_object(const nlohmann::json& obj) {
  PieceTable t;
  if (!obj.is_object()) return t;
  std::int32_t maxid = -1;
  for (auto& [k, v] : obj.items()) maxid = std::max(maxid, v.get<std::int32_t>());
  if (maxid < 0) return t;
  t.pieces.assign(static_cast<std::size_t>(maxid + 1), {});
  t.identity256 = false;
  for (auto& [k, v] : obj.items()) {
    auto id = v.get<std::int32_t>();
    if (id >= 0) t.pieces[static_cast<std::size_t>(id)] = json_text_to_bytes(k);
  }
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(t.pieces.size()); ++i) {
    if (t.pieces[static_cast<std::size_t>(i)].size() == 1)
      t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i)][0])] = i;
  }
  return t;
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
  auto bm = BpeModel::from_merges(std::move(*merges), table->size());
  if (!bm) return std::unexpected(bm.error());
  return std::make_unique<Tokenizer>(std::move(*pt), std::move(*bm), std::move(*table));
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
    std::unique_ptr<Pretokenizer> pt;
    if (ptype == "ByteLevel") pt = make_gpt2_pretok();
    else if (ptype == "Metaspace") pt = make_metaspace_pretok(true);
    else pt = make_identity_pretok();
    if (type == "Unigram") {
      PieceTable table;
      std::vector<float> scores;
      if (model.contains("vocab") && model["vocab"].is_array()) {
        for (auto& row : model["vocab"]) {
          if (row.is_array() && row.size() >= 2) {
            table.pieces.push_back(json_text_to_bytes(row[0].get<std::string>()));
            scores.push_back(row[1].get<float>());
          }
        }
      }
      auto um = UnigramModel::from_scores(std::move(scores), true);
      if (!um) return std::unexpected(um.error());
      return std::make_unique<Tokenizer>(std::move(pt), std::move(*um), std::move(table));
    }
    auto merges = BpeModel::parse_merges(model.dump());
    if (!merges) return std::unexpected(merges.error());
    PieceTable table;
    if (model.contains("vocab") && model["vocab"].is_object()) table = table_from_vocab_object(model["vocab"]);
    if (table.pieces.empty()) table = BpeModel::bytes_table();
    if (merges->empty() && model.contains("merges") && model["merges"].is_array()) {
      std::unordered_map<std::string, std::int32_t> inv;
      for (std::int32_t i = 0; i < static_cast<std::int32_t>(table.pieces.size()); ++i)
        inv[table.pieces[static_cast<std::size_t>(i)]] = i;
      for (auto& m : model["merges"]) {
        if (m.is_string()) {
          auto pair = json_text_to_bytes(m.get<std::string>());
          auto sp = pair.find(' ');
          if (sp == std::string::npos) continue;
          auto a = inv.find(pair.substr(0, sp));
          auto b = inv.find(pair.substr(sp + 1));
          if (a != inv.end() && b != inv.end()) merges->emplace_back(a->second, b->second);
        } else if (m.is_array() && m.size() == 2 && m[0].is_string()) {
          auto ia = inv.find(json_text_to_bytes(m[0].get<std::string>()));
          auto ib = inv.find(json_text_to_bytes(m[1].get<std::string>()));
          if (ia != inv.end() && ib != inv.end()) merges->emplace_back(ia->second, ib->second);
        }
      }
    }
    auto bm = BpeModel::from_merges(std::move(*merges), table.size());
    if (!bm) return std::unexpected(bm.error());
    return std::make_unique<Tokenizer>(std::move(pt), std::move(*bm), std::move(table));
  }
  auto vocabp = std::filesystem::is_directory(dir) ? dir / "vocab.json" : dir;
  auto mergesp = std::filesystem::is_directory(dir) ? dir / "merges.txt" : dir.parent_path() / "merges.txt";
  auto vs = read_all(vocabp);
  if (vs.empty()) return std::unexpected(make_error(Errc::io, "vocab.json"));
  auto vj = parse_json(vs);
  if (!vj) return std::unexpected(vj.error());
  auto table = table_from_vocab_object(*vj);
  std::unordered_map<std::string, std::int32_t> inv;
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(table.pieces.size()); ++i)
    inv[table.pieces[static_cast<std::size_t>(i)]] = i;
  std::vector<std::pair<std::int32_t, std::int32_t>> merges;
  auto ms = read_all(mergesp);
  std::istringstream iss(ms);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto sp = line.find(' ');
    if (sp == std::string::npos) continue;
    auto a = inv.find(line.substr(0, sp));
    auto b = inv.find(line.substr(sp + 1));
    if (a != inv.end() && b != inv.end()) merges.emplace_back(a->second, b->second);
  }
  auto bm = BpeModel::from_merges(std::move(merges), table.size());
  if (!bm) return std::unexpected(bm.error());
  return std::make_unique<Tokenizer>(make_gpt2_pretok(), std::move(*bm), std::move(table));
}

}  // namespace gyre
