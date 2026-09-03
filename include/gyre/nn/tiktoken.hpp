#pragma once

#include "gyre/nn/tokenize.hpp"

namespace gyre {

// Grok-2 / SGLang tokenizer.tok.json: tiktoken BPE + word_split V1 (PAT_STR_B).
// Ranks are token ids of byte pieces. Encode is not Gyre sequential-merge BPE.

class TiktokenModel final : public VocabModel {
 public:
  const char* name() const noexcept override { return "tiktoken_bpe"; }
  std::int64_t vocab_size() const noexcept override { return vocab_size_; }
  Result<void> encode_span(std::string_view span, const PieceTable& pieces,
                           std::vector<std::int32_t>& out) const override;
  std::string to_json() const override;
  Result<void> write_huggingface(const std::filesystem::path& dir, const Pretokenizer& pretok,
                                 const PieceTable& pieces) const override;

  void set_vocab_size(std::int64_t n) { vocab_size_ = n; }
  void set_ranks(std::unordered_map<std::string, std::int32_t> ranks) {
    ranks_ = std::move(ranks);
  }
  const std::unordered_map<std::string, std::int32_t>& ranks() const { return ranks_; }

 private:
  std::unordered_map<std::string, std::int32_t> ranks_;
  std::int64_t vocab_size_{0};
};

std::unique_ptr<Pretokenizer> make_tiktoken_v1_pretok(std::vector<std::string> specials);

bool is_tiktoken_tok_json(std::string_view json);
Result<std::unique_ptr<Tokenizer>> load_tiktoken_json(std::string_view json);

}  // namespace gyre
