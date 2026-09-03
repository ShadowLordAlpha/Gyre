#pragma once

#include "gyre/nn/tokenize.hpp"

namespace gyre {

// Byte-pair merges. Empty merges + bytes_table() = 0..255; empty merges +
// chars_table() = observed alphabet. Gyre encode applies merges in train order
// on each pretok span (identity pretok = whole buffer).

class BpeModel final : public VocabModel {
 public:
  BpeModel() = default;
  explicit BpeModel(std::vector<std::pair<std::int32_t, std::int32_t>> merges);

  const char* name() const noexcept override { return "bpe"; }
  std::int64_t vocab_size() const noexcept override;
  Result<void> encode_span(std::string_view span, const PieceTable& pieces,
                           std::vector<std::int32_t>& out) const override;
  std::string to_json() const override;
  Result<void> write_huggingface(const std::filesystem::path& dir, const Pretokenizer& pretok,
                                 const PieceTable& pieces) const override;

  const std::vector<std::pair<std::int32_t, std::int32_t>>& merges() const { return merges_; }
  void set_vocab_size(std::int64_t n) { vocab_size_ = n; }

  static Result<std::unique_ptr<BpeModel>> from_merges(
      std::vector<std::pair<std::int32_t, std::int32_t>> merges, std::int64_t vocab);
  static Result<std::pair<std::unique_ptr<BpeModel>, PieceTable>> train(
      std::string_view text, std::int32_t vocab_size,
      std::function<void(int, int)> progress = {});
  static Result<std::pair<std::unique_ptr<BpeModel>, PieceTable>> chars_table(
      std::string_view text);
  static PieceTable bytes_table();
  static Result<std::vector<std::pair<std::int32_t, std::int32_t>>> parse_merges(
      std::string_view json);

 private:
  std::vector<std::pair<std::int32_t, std::int32_t>> merges_;
  std::int64_t vocab_size_{256};
};

}  // namespace gyre
