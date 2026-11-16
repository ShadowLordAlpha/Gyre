#pragma once

#include "gyre/error.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gyre {

// Tokenizer = Pretokenizer + VocabModel + PieceTable. Encode is non-virtual:
// split the whole string once, then encode_span per span. See docs/tokenizer.md.

// Largest token id + 1 a tokenizer file may declare. Ids from a file size the piece table, so an
// unchecked id is an allocation bomb; real vocabularies stay far below this (Grok-2: 131072).
inline constexpr std::int64_t kMaxTokenizerVocab = std::int64_t{1} << 22;

struct PieceTable {
  std::vector<std::string> pieces;
  // If nonempty, encode maps each input byte through this table. Empty = 1:1 byte id
  // for identity 0..255, or per-byte lookup into pieces for a restricted alphabet.
  std::unordered_map<std::uint8_t, std::int32_t> byte_to_id;
  bool identity256{false};

  std::int64_t size() const noexcept { return static_cast<std::int64_t>(pieces.size()); }
  std::string concat(std::span<const std::int32_t> ids) const;
};

class Pretokenizer {
 public:
  virtual ~Pretokenizer() = default;
  virtual const char* name() const noexcept = 0;
  // `spans` view either `text` or `scratch` (filled when mapping is required).
  virtual Result<void> split(std::string_view text, std::string& scratch,
                             std::vector<std::string_view>& spans) const = 0;
  virtual std::string to_json() const;
  // Undoes the text mapping `split` applies (Metaspace: U+2581 back to spaces, minus the dummy
  // prefix). Tokenizer::decode runs it on the concatenated pieces; identity for the others.
  virtual void decode(std::string& text) const { (void)text; }
};

class VocabModel {
 public:
  virtual ~VocabModel() = default;
  virtual const char* name() const noexcept = 0;
  virtual std::int64_t vocab_size() const noexcept = 0;
  virtual Result<void> encode_span(std::string_view span, const PieceTable& pieces,
                                   std::vector<std::int32_t>& out) const = 0;
  virtual std::string to_json() const = 0;
  virtual Result<void> write_huggingface(const std::filesystem::path& dir,
                                         const Pretokenizer& pretok,
                                         const PieceTable& pieces) const = 0;
};

class Tokenizer {
 public:
  Tokenizer(std::unique_ptr<Pretokenizer> pretok, std::unique_ptr<VocabModel> model,
            PieceTable pieces);

  // Special (added) tokens are matched literally first (leftmost, then longest); the text between
  // them goes through pretok + model. With no specials this is exactly pretok + model.
  Result<std::vector<std::int32_t>> encode(
      std::string_view text, std::function<void(int, int)> progress = {}) const;
  std::string decode(std::span<const std::int32_t> ids) const;
  std::int64_t vocab_size() const;
  const char* model_name() const;
  const char* pretok_name() const;
  const PieceTable& pieces() const { return pieces_; }
  VocabModel& model() { return *model_; }
  const VocabModel& model() const { return *model_; }
  const Pretokenizer& pretok() const { return *pretok_; }
  // Literal tokens (content, id) such as <|endoftext|>. Only Hugging Face files with an explicit-id
  // vocab set these; the legacy GPT-2 layout keeps treating "<|endoftext|>" as plain text.
  void set_specials(std::vector<std::pair<std::string, std::int32_t>> s) { specials_ = std::move(s); }
  const std::vector<std::pair<std::string, std::int32_t>>& specials() const { return specials_; }

  // Nested Gyre tokenizer document (`*.gyre.json`).
  std::string to_json() const;
  Result<void> save(const std::filesystem::path& path) const;
  Result<void> save_huggingface(const std::filesystem::path& dir) const;

  static Result<std::unique_ptr<Tokenizer>> bytes();
  static Result<std::unique_ptr<Tokenizer>> chars_from_text(std::string_view text);
  static Result<std::unique_ptr<Tokenizer>> train_bpe(
      std::string_view text, std::int32_t vocab_size,
      std::function<void(int, int)> progress = {});
  static Result<std::unique_ptr<Tokenizer>> train_unigram(
      std::string_view text, std::int32_t vocab_size,
      std::function<void(int, int)> progress = {});
  static Result<std::unique_ptr<Tokenizer>> from_json(std::string_view json);
  static Result<std::unique_ptr<Tokenizer>> load(const std::filesystem::path& path);
  static Result<std::unique_ptr<Tokenizer>> load_huggingface(const std::filesystem::path& dir);
  static Result<std::unique_ptr<Tokenizer>> load_sentencepiece(const std::filesystem::path& path);

 private:
  std::unique_ptr<Pretokenizer> pretok_;
  std::unique_ptr<VocabModel> model_;
  PieceTable pieces_;
  std::vector<std::pair<std::string, std::int32_t>> specials_;

  Result<void> encode_plain(std::string_view text, std::vector<std::int32_t>& out,
                            std::function<void(int, int)>* progress) const;
};

std::unique_ptr<Pretokenizer> make_identity_pretok();
std::unique_ptr<Pretokenizer> make_gpt2_pretok();
// Hugging Face Sequence[Digits(individual_digits), ByteLevel(use_regex)] (SmolLM2 / StarCoder).
std::unique_ptr<Pretokenizer> make_digits_bytelevel_pretok();
std::unique_ptr<Pretokenizer> make_metaspace_pretok(bool dummy_prefix = true);

Result<std::unique_ptr<Pretokenizer>> pretok_from_name(std::string_view name);

}  // namespace gyre
