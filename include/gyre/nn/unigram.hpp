#pragma once

#include "gyre/nn/tokenize.hpp"

namespace gyre {

// Viterbi over scored pieces (SentencePiece unigram). Native train is not
// bit-identical to spm_train. See docs/tokenizer.md.

class UnigramModel final : public VocabModel {
 public:
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

  static Result<std::pair<std::unique_ptr<UnigramModel>, PieceTable>> train(
      std::string_view text, std::int32_t vocab_size,
      std::function<void(int, int)> progress = {});
  static Result<std::unique_ptr<UnigramModel>> from_scores(std::vector<float> scores,
                                                           bool byte_fallback);

 private:
  std::vector<float> scores_;
  bool byte_fallback_{false};
};

}  // namespace gyre
