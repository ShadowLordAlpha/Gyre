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
};

std::unique_ptr<Pretokenizer> make_identity_pretok();
std::unique_ptr<Pretokenizer> make_gpt2_pretok();
std::unique_ptr<Pretokenizer> make_metaspace_pretok(bool dummy_prefix = true);

Result<std::unique_ptr<Pretokenizer>> pretok_from_name(std::string_view name);

// JSON helpers used by models.
std::string json_escape(std::string_view s);
bool json_unescape_string(std::string_view in, std::size_t& i, std::string& out);
std::string_view json_object_field(std::string_view json, std::string_view key);

}  // namespace gyre
