#pragma once

#include "gyre/nn/tokenize.hpp"

namespace gyre {

// Viterbi over scored pieces (SentencePiece unigram). Native train is not
// bit-identical to spm_train. See docs/tokenizer.md.

class UnigramModel final : public VocabModel {
 public:
  // SentencePiece piece types (sentencepiece_model.proto). An empty type list means every piece
  // is NORMAL (Gyre-trained and older files).
  enum PieceType : std::uint8_t { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

  UnigramModel() = default;
  UnigramModel(std::vector<float> scores, bool byte_fallback);

  const char* name() const noexcept override { return "unigram"; }
  std::int64_t vocab_size() const noexcept override;
  Result<void> encode_span(std::string_view span, const PieceTable& pieces,
                           std::vector<std::int32_t>& out) const override;
  std::string to_json() const override;
  Result<void> write_huggingface(const std::filesystem::path& dir, const Pretokenizer& pretok,
                                 const PieceTable& pieces) const override;

  const std::vector<float>& scores() const { return scores_; }

  // Id emitted for text no piece covers; -1 makes that an encode error instead. Older files have
  // no "unk_id" and keep the historical 0.
  void set_unk_id(std::int32_t id) { unk_id_ = id; }
  std::int32_t unk_id() const { return unk_id_; }
  // Per-piece SentencePiece types (same length as the piece table). With types, only NORMAL and
  // USER_DEFINED pieces are matched in text; BYTE pieces (stored as the raw byte) are used for
  // byte fallback, and a character without a one-character piece becomes byte pieces or unk_id,
  // as in SentencePiece.
  void set_piece_types(std::vector<std::uint8_t> t) { types_ = std::move(t); }
  const std::vector<std::uint8_t>& piece_types() const { return types_; }

  static Result<std::pair<std::unique_ptr<UnigramModel>, PieceTable>> train(
      std::string_view text, std::int32_t vocab_size,
      std::function<void(int, int)> progress = {});
  static Result<std::unique_ptr<UnigramModel>> from_scores(std::vector<float> scores,
                                                           bool byte_fallback);

 private:
  std::vector<float> scores_;
  bool byte_fallback_{false};
  std::int32_t unk_id_{0};
  std::vector<std::uint8_t> types_;

  Result<void> encode_typed(std::string_view span, const PieceTable& pieces, std::vector<std::int32_t>& out) const;
};

}  // namespace gyre
