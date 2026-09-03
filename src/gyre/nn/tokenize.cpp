#include "gyre/nn/tokenize.hpp"
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/tiktoken.hpp"
#include "gyre/nn/unigram.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
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

std::string json_escape(std::string_view s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o.push_back('\\');
      o.push_back(static_cast<char>(c));
    } else if (c == '\n') {
      o += "\\n";
    } else if (c == '\r') {
      o += "\\r";
    } else if (c == '\t') {
      o += "\\t";
    } else if (c < 0x20 || c >= 0x7F) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      o += buf;
    } else {
      o.push_back(static_cast<char>(c));
    }
  }
  o.push_back('"');
  return o;
}

bool json_unescape_string(std::string_view in, std::size_t& i, std::string& out) {
  out.clear();
  if (i >= in.size() || in[i] != '"') return false;
  ++i;
  while (i < in.size()) {
    char c = in[i++];
    if (c == '"') return true;
    if (c == '\\' && i < in.size()) {
      char e = in[i++];
      if (e == 'n') out.push_back('\n');
      else if (e == 'r') out.push_back('\r');
      else if (e == 't') out.push_back('\t');
      else if (e == 'u' && i + 4 <= in.size()) {
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
          char h = in[i++];
          v <<= 4;
          if (h >= '0' && h <= '9') v += static_cast<unsigned>(h - '0');
          else if (h >= 'a' && h <= 'f') v += static_cast<unsigned>(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') v += static_cast<unsigned>(h - 'A' + 10);
        }
        out.push_back(static_cast<char>(v & 0xFF));
      } else {
        out.push_back(e);
      }
    } else {
      out.push_back(c);
    }
  }
  return false;
}

std::string_view json_object_field(std::string_view json, std::string_view key) {
  std::string pat = "\"";
  pat += key;
  pat += '"';
  auto p = json.find(pat);
  if (p == std::string_view::npos) return {};
  p = json.find(':', p + pat.size());
  if (p == std::string_view::npos) return {};
  ++p;
  while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
  if (p >= json.size()) return {};
  if (json[p] == '"') {
    auto e = p + 1;
    while (e < json.size() && json[e] != '"') {
      if (json[e] == '\\' && e + 1 < json.size()) e += 2;
      else ++e;
    }
    if (e < json.size()) ++e;
    return json.substr(p, e - p);
  }
  if (json[p] == '{' || json[p] == '[') {
    char open = json[p];
    char close = open == '{' ? '}' : ']';
    int depth = 0;
    auto e = p;
    for (; e < json.size(); ++e) {
      if (json[e] == '"') {
        ++e;
        while (e < json.size() && json[e] != '"') {
          if (json[e] == '\\' && e + 1 < json.size()) e += 2;
          else ++e;
        }
        continue;
      }
      if (json[e] == open) ++depth;
      else if (json[e] == close) {
        --depth;
        if (depth == 0) {
          ++e;
          return json.substr(p, e - p);
        }
      }
    }
    return json.substr(p);
  }
  auto e = p;
  while (e < json.size() && json[e] != ',' && json[e] != '}' && json[e] != ']') ++e;
  return json.substr(p, e - p);
}

static std::string json_unquote(std::string_view quoted) {
  if (quoted.size() >= 2 && quoted.front() == '"') {
    std::size_t i = 0;
    std::string s;
    if (json_unescape_string(quoted, i, s)) return s;
  }
  return std::string(quoted);
}

static Result<std::vector<std::string>> parse_string_array(std::string_view arr) {
  std::vector<std::string> v;
  if (arr.empty() || arr.front() != '[') return v;
  std::size_t i = 1;
  while (i < arr.size()) {
    while (i < arr.size() && (arr[i] == ' ' || arr[i] == '\n' || arr[i] == ',')) ++i;
    if (i < arr.size() && arr[i] == ']') break;
    if (i >= arr.size() || arr[i] != '"') break;
    std::string s;
    if (!json_unescape_string(arr, i, s)) break;
    v.push_back(std::move(s));
  }
  return v;
}

static Result<std::vector<float>> parse_float_array(std::string_view arr) {
  std::vector<float> v;
  if (arr.empty() || arr.front() != '[') return v;
  std::size_t i = 1;
  while (i < arr.size()) {
    while (i < arr.size() && (arr[i] == ' ' || arr[i] == '\n' || arr[i] == ',')) ++i;
    if (i < arr.size() && arr[i] == ']') break;
    char* end = nullptr;
    float f = std::strtof(arr.data() + i, &end);
    if (end == arr.data() + i) break;
    v.push_back(f);
    i = static_cast<std::size_t>(end - arr.data());
  }
  return v;
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
  std::ostringstream ss;
  ss << "{\"gyre\":\"tokenizer\",\"version\":1,\"tokenizer\":{";
  ss << "\"pretoken\":\"" << pretok_->name() << "\",";
  ss << "\"model\":\"" << model_->name() << "\",";
  ss << model_->to_json();
  bool identity = pieces_.identity256 && pieces_.pieces.size() >= 256;
  if (!identity) {
    ss << ",\"vocab\":[";
    for (std::size_t i = 0; i < pieces_.pieces.size(); ++i) {
      if (i) ss << ',';
      ss << json_escape(pieces_.pieces[i]);
    }
    ss << ']';
  }
  if (!pieces_.identity256 && !pieces_.pieces.empty() && std::string(model_->name()) == "unigram") {
    // pieces already in vocab
  } else if (!identity && std::string(model_->name()) == "bpe") {
    // vocab holds restricted alphabet / all pieces
  } else if (identity) {
    ss << ",\"pieces\":[]";
  }
  ss << "}}";
  return ss.str();
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

static Result<PieceTable> pieces_from_vocab_or_merges(
    std::string_view obj, const std::vector<std::pair<std::int32_t, std::int32_t>>& merges) {
  auto vocab = json_object_field(obj, "vocab");
  auto piecesf = json_object_field(obj, "pieces");
  if (!vocab.empty() && vocab.front() == '[') {
    auto arr = parse_string_array(vocab);
    if (!arr) return std::unexpected(arr.error());
    PieceTable t;
    t.identity256 = false;
    t.pieces = std::move(*arr);
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(t.pieces.size()); ++i) {
      if (t.pieces[static_cast<std::size_t>(i)].size() == 1)
        t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i)][0])] = i;
    }
    return t;
  }
  if (!piecesf.empty() && piecesf.front() == '[' && piecesf.size() > 2) {
    auto arr = parse_string_array(piecesf);
    if (!arr) return std::unexpected(arr.error());
    if (!arr->empty()) {
      PieceTable t;
      t.pieces = std::move(*arr);
      t.identity256 = t.pieces.size() >= 256;
      for (int i = 0; i < 256 && static_cast<std::size_t>(i) < t.pieces.size(); ++i)
        t.byte_to_id[static_cast<std::uint8_t>(i)] = i;
      return t;
    }
  }
  PieceTable t = BpeModel::bytes_table();
  for (auto [a, b] : merges) {
    if (static_cast<std::size_t>(a) < t.pieces.size() && static_cast<std::size_t>(b) < t.pieces.size())
      t.pieces.push_back(t.pieces[static_cast<std::size_t>(a)] + t.pieces[static_cast<std::size_t>(b)]);
  }
  return t;
}

static Result<std::unique_ptr<Tokenizer>> from_nested(std::string_view obj) {
  auto pret = json_unquote(json_object_field(obj, "pretoken"));
  if (pret.empty()) pret = json_unquote(json_object_field(obj, "pretok"));
  if (pret.empty()) pret = "identity";
  auto model = json_unquote(json_object_field(obj, "model"));
  if (model.empty()) model = "bpe";
  auto pt = pretok_from_name(pret);
  if (!pt) return std::unexpected(pt.error());
  if (model == "unigram") {
    auto scores = parse_float_array(json_object_field(obj, "scores"));
    if (!scores) return std::unexpected(scores.error());
    auto vocab = parse_string_array(json_object_field(obj, "vocab"));
    if (!vocab) return std::unexpected(vocab.error());
    if (vocab->empty()) vocab = parse_string_array(json_object_field(obj, "pieces"));
    PieceTable t;
    t.pieces = std::move(*vocab);
    auto bf = json_object_field(obj, "byte_fallback");
    bool fallback = bf.find("true") != std::string_view::npos;
    auto um = UnigramModel::from_scores(std::move(*scores), fallback);
    if (!um) return std::unexpected(um.error());
    return std::make_unique<Tokenizer>(std::move(*pt), std::move(*um), std::move(t));
  }
  auto merges = BpeModel::parse_merges(obj);
  if (!merges) return std::unexpected(merges.error());
  auto table = pieces_from_vocab_or_merges(obj, *merges);
  if (!table) return std::unexpected(table.error());
  auto bm = BpeModel::from_merges(std::move(*merges), table->size());
  if (!bm) return std::unexpected(bm.error());
  return std::make_unique<Tokenizer>(std::move(*pt), std::move(*bm), std::move(*table));
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::from_json(std::string_view json) {
  auto nested = json_object_field(json, "tokenizer");
  if (!nested.empty() && nested.front() == '{') return from_nested(nested);

  // Legacy GYRE1 trailer: "tokenizer":"bpe","merges":[...]
  auto tok = json_unquote(json_object_field(json, "tokenizer"));
  auto vocab_field = json_object_field(json, "vocab");
  if (tok == "bpe" || json.find("\"merges\"") != std::string_view::npos) {
    auto merges = BpeModel::parse_merges(json);
    if (!merges) return std::unexpected(merges.error());
    auto table = pieces_from_vocab_or_merges(json, *merges);
    if (!table) return std::unexpected(table.error());
    auto bm = BpeModel::from_merges(std::move(*merges), table->size());
    if (!bm) return std::unexpected(bm.error());
    return std::make_unique<Tokenizer>(make_identity_pretok(), std::move(*bm), std::move(*table));
  }
  if (vocab_field == "\"bytes\"" || tok == "bytes") return bytes();
  if (!vocab_field.empty() && vocab_field.front() == '[') {
    auto arr = parse_string_array(vocab_field);
    if (!arr) return std::unexpected(arr.error());
    PieceTable t;
    std::int32_t i = 0;
    for (auto& s : *arr) {
      t.pieces.push_back(s);
      if (s.size() == 1) t.byte_to_id[static_cast<std::uint8_t>(s[0])] = i;
      ++i;
    }
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

static Result<PieceTable> vocab_json_object(std::string_view obj) {
  PieceTable t;
  std::size_t i = 0;
  while (i < obj.size() && obj[i] != '{') ++i;
  if (i < obj.size()) ++i;
  std::vector<std::pair<std::int32_t, std::string>> items;
  while (i < obj.size()) {
    while (i < obj.size() && (obj[i] == ' ' || obj[i] == '\n' || obj[i] == ',')) ++i;
    if (i < obj.size() && obj[i] == '}') break;
    if (i >= obj.size() || obj[i] != '"') break;
    std::string key;
    if (!json_unescape_string(obj, i, key)) break;
    while (i < obj.size() && obj[i] != ':') ++i;
    if (i < obj.size()) ++i;
    char* end = nullptr;
    auto id = static_cast<std::int32_t>(std::strtoll(obj.data() + i, &end, 10));
    i = static_cast<std::size_t>(end - obj.data());
    items.emplace_back(id, std::move(key));
  }
  std::int32_t maxid = -1;
  for (auto& [id, _] : items) maxid = std::max(maxid, id);
  if (maxid < 0) return t;
  t.pieces.assign(static_cast<std::size_t>(maxid + 1), {});
  for (auto& [id, k] : items) {
    if (id >= 0) t.pieces[static_cast<std::size_t>(id)] = std::move(k);
  }
  t.identity256 = false;
  for (std::int32_t i2 = 0; i2 < static_cast<std::int32_t>(t.pieces.size()); ++i2) {
    if (t.pieces[static_cast<std::size_t>(i2)].size() == 1)
      t.byte_to_id[static_cast<std::uint8_t>(t.pieces[static_cast<std::size_t>(i2)][0])] = i2;
  }
  return t;
}

Result<std::unique_ptr<Tokenizer>> Tokenizer::load_huggingface(const std::filesystem::path& dir) {
  auto tokj = dir;
  if (std::filesystem::is_directory(dir)) tokj = dir / "tokenizer.json";
  if (std::filesystem::exists(tokj) && tokj.filename() == "tokenizer.json") {
    auto s = read_all(tokj);
    auto model = json_object_field(s, "model");
    auto type = json_unquote(json_object_field(model.empty() ? s : model, "type"));
    auto pret = json_object_field(s, "pre_tokenizer");
    auto ptype = json_unquote(json_object_field(pret, "type"));
    std::unique_ptr<Pretokenizer> pt;
    if (ptype == "ByteLevel") pt = make_gpt2_pretok();
    else if (ptype == "Metaspace") pt = make_metaspace_pretok(true);
    else pt = make_identity_pretok();
    if (type == "Unigram") {
      auto vocab = json_object_field(model, "vocab");
      PieceTable table;
      std::vector<float> scores;
      std::size_t i = 0;
      while (i < vocab.size() && vocab[i] != '[') ++i;
      if (i < vocab.size()) ++i;
      while (i < vocab.size()) {
        while (i < vocab.size() && (vocab[i] == ' ' || vocab[i] == ',' || vocab[i] == '\n')) ++i;
        if (i < vocab.size() && vocab[i] == ']') break;
        if (i >= vocab.size() || vocab[i] != '[') break;
        ++i;
        while (i < vocab.size() && vocab[i] != '"') ++i;
        std::string piece;
        if (!json_unescape_string(vocab, i, piece)) break;
        while (i < vocab.size() && vocab[i] != ',') ++i;
        if (i < vocab.size()) ++i;
        char* end = nullptr;
        float sc = std::strtof(vocab.data() + i, &end);
        i = static_cast<std::size_t>(end - vocab.data());
        while (i < vocab.size() && vocab[i] != ']') ++i;
        if (i < vocab.size()) ++i;
        table.pieces.push_back(std::move(piece));
        scores.push_back(sc);
      }
      auto um = UnigramModel::from_scores(std::move(scores), true);
      if (!um) return std::unexpected(um.error());
      return std::make_unique<Tokenizer>(std::move(pt), std::move(*um), std::move(table));
    }
    auto merges = BpeModel::parse_merges(model.empty() ? s : model);
    if (!merges) return std::unexpected(merges.error());
    auto vocab = json_object_field(model.empty() ? s : model, "vocab");
    PieceTable table;
    if (!vocab.empty() && vocab.front() == '{') {
      auto vt = vocab_json_object(vocab);
      if (!vt) return std::unexpected(vt.error());
      table = std::move(*vt);
    }
    if (table.pieces.empty()) table = BpeModel::bytes_table();
    // HF merges as "a b" strings: parse if integer parse yielded empty
    if (merges->empty()) {
      auto marr = json_object_field(model.empty() ? s : model, "merges");
      std::unordered_map<std::string, std::int32_t> inv;
      for (std::int32_t i = 0; i < static_cast<std::int32_t>(table.pieces.size()); ++i)
        inv[table.pieces[static_cast<std::size_t>(i)]] = i;
      std::size_t i = 0;
      while (i < marr.size() && marr[i] != '[') ++i;
      if (i < marr.size()) ++i;
      while (i < marr.size()) {
        while (i < marr.size() && (marr[i] == ' ' || marr[i] == ',' || marr[i] == '\n')) ++i;
        if (i < marr.size() && marr[i] == ']') break;
        if (i < marr.size() && marr[i] == '"') {
          std::string pair;
          if (!json_unescape_string(marr, i, pair)) break;
          auto sp = pair.find(' ');
          if (sp != std::string::npos) {
            auto a = inv.find(pair.substr(0, sp));
            auto b = inv.find(pair.substr(sp + 1));
            if (a != inv.end() && b != inv.end()) merges->emplace_back(a->second, b->second);
          }
        } else if (i < marr.size() && marr[i] == '[') {
          ++i;
          std::string a, b;
          while (i < marr.size() && marr[i] != '"') ++i;
          if (!json_unescape_string(marr, i, a)) break;
          while (i < marr.size() && marr[i] != '"') ++i;
          if (!json_unescape_string(marr, i, b)) break;
          while (i < marr.size() && marr[i] != ']') ++i;
          if (i < marr.size()) ++i;
          auto ia = inv.find(a);
          auto ib = inv.find(b);
          if (ia != inv.end() && ib != inv.end()) merges->emplace_back(ia->second, ib->second);
        } else
          break;
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
  auto table = vocab_json_object(vs);
  if (!table) return std::unexpected(table.error());
  std::unordered_map<std::string, std::int32_t> inv;
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(table->pieces.size()); ++i)
    inv[table->pieces[static_cast<std::size_t>(i)]] = i;
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
  auto bm = BpeModel::from_merges(std::move(merges), table->size());
  if (!bm) return std::unexpected(bm.error());
  return std::make_unique<Tokenizer>(make_gpt2_pretok(), std::move(*bm), std::move(*table));
}

}  // namespace gyre
