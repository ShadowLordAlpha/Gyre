// Byte-level (GPT-2) tokenizer: pre-tokenizer regex semantics, rank-based BPE equivalence with the
// old sequential merges, and (when data/gpt2 exists) the published GPT-2 token ids.
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/tokenize.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace gyre;

namespace {

std::vector<std::string> split(const std::string& text) {
  auto pt = make_gpt2_pretok();
  std::string scratch;
  std::vector<std::string_view> spans;
  EXPECT_TRUE(pt->split(text, scratch, spans));
  std::vector<std::string> out;
  std::string joined;
  for (auto s : spans) {
    out.emplace_back(s);
    joined += s;
  }
  EXPECT_EQ(joined, text) << "spans must tile the input exactly";
  return out;
}

using V = std::vector<std::string>;

const std::filesystem::path kGpt2Dir = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "gpt2";

}  // namespace

TEST(ByteLevel, PretokenizerFollowsTheGpt2Pattern) {
  EXPECT_EQ(split("Hello world"), (V{"Hello", " world"}));
  EXPECT_EQ(split("Hello  world"), (V{"Hello", " ", " world"}));  // \s+(?!\S) leaves the last space
  EXPECT_EQ(split("a  "), (V{"a", "  "}));                        // trailing run is taken whole
  EXPECT_EQ(split("x\n\ny"), (V{"x", "\n", "\n", "y"}));
  EXPECT_EQ(split("don't I'm it's we'll they've I'd"),
            (V{"don", "'t", " I", "'m", " it", "'s", " we", "'ll", " they", "'ve", " I", "'d"}));
  EXPECT_EQ(split("abc123 def"), (V{"abc", "123", " def"}));
  EXPECT_EQ(split("Hi!! ,foo"), (V{"Hi", "!!", " ,", "foo"}));
  EXPECT_EQ(split("1,000.50"), (V{"1", ",", "000", ".", "50"}));
  EXPECT_EQ(split("'S"), (V{"'", "S"}));                          // contractions are lower-case only
  EXPECT_EQ(split(" 42"), (V{" 42"}));
  EXPECT_EQ(split(""), V{});
  // multi-byte letters stay together (Unicode \p{L}), emoji are "other"
  EXPECT_EQ(split("caf\xC3\xA9 na\xC3\xAFve"), (V{"caf\xC3\xA9", " na\xC3\xAFve"}));
  EXPECT_EQ(split("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E!"), (V{"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E", "!"}));
  // arbitrary bytes still tile the input (invalid UTF-8 never loops or drops data)
  const std::string junk = "\xFF\xFE ab \x80\x81 \n\t ";
  split(junk);
}

TEST(ByteLevel, RankBasedEncodeEqualsSequentialMerges) {
  const std::filesystem::path shk = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "shakespeare.txt";
  if (!std::filesystem::exists(shk)) GTEST_SKIP() << "shakespeare.txt missing";
  std::ifstream f(shk, std::ios::binary);
  std::ostringstream o;
  o << f.rdbuf();
  const std::string text = o.str().substr(0, 60000);
  auto trained = BpeModel::train(text, 600);
  ASSERT_TRUE(trained);
  const auto& merges = trained->first->merges();
  const auto& table = trained->second;

  // reference: apply every merge, in order, to the whole span
  auto sequential = [&](std::string_view span) {
    std::vector<std::int32_t> seq;
    for (unsigned char c : span) seq.push_back(c);
    std::int32_t nid = 256;
    for (auto [a, b] : merges) {
      std::size_t w = 0;
      for (std::size_t r = 0; r < seq.size();) {
        if (r + 1 < seq.size() && seq[r] == a && seq[r + 1] == b) {
          seq[w++] = nid;
          r += 2;
        } else {
          seq[w++] = seq[r++];
        }
      }
      seq.resize(w);
      ++nid;
    }
    return seq;
  };
  std::size_t pos = 0;
  int checked = 0;
  while (pos + 40 < text.size() && checked < 400) {
    const std::string_view span(text.data() + pos, 24 + (pos % 17));
    std::vector<std::int32_t> got;
    ASSERT_TRUE(trained->first->encode_span(span, table, got));
    ASSERT_EQ(got, sequential(span)) << "span at " << pos;
    pos += 97;
    ++checked;
  }
  EXPECT_GT(checked, 100);
}

TEST(ByteLevel, RealGpt2MatchesPublishedTokenIds) {
  if (!std::filesystem::exists(kGpt2Dir / "tokenizer.json")) GTEST_SKIP() << "data/gpt2 not downloaded";
  auto tok = Tokenizer::load_huggingface(kGpt2Dir);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->vocab_size(), 50257);
  EXPECT_STREQ((*tok)->pretok_name(), "bytelevel");
  auto ids = [&](const std::string& s) {
    auto e = (*tok)->encode(s);
    EXPECT_TRUE(e) << (e ? "" : e.error().message);
    return e ? *e : std::vector<std::int32_t>{};
  };
  EXPECT_EQ(ids("Hello world"), (std::vector<std::int32_t>{15496, 995}));
  EXPECT_EQ(ids("The quick brown fox jumps over the lazy dog."),
            (std::vector<std::int32_t>{464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13}));
  EXPECT_EQ(ids("<|endoftext|>").size(), 7u) << "special tokens are not parsed unless the caller asks";
  // round trips, including multi-byte text, emoji, runs of spaces and newlines
  for (const std::string s : {"Hello world", "  leading and trailing  ", "line1\nline2\n\n\nline3",
                              "caf\xC3\xA9 na\xC3\xAFve \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x98\x80",
                              "I'm sure they'll say it's 1,234.56 or something!?", "tabs\tand\r\nCRLF"}) {
    auto e = ids(s);
    EXPECT_EQ((*tok)->decode(e), s) << s;
  }
  // Gyre's own JSON round-trips the exact same tokenizer
  auto again = Tokenizer::from_json((*tok)->to_json());
  ASSERT_TRUE(again) << again.error().message;
  const std::string sample = "To be, or not to be: that is the question.\nWhether 'tis nobler in the mind";
  EXPECT_EQ((*again)->encode(sample).value(), (*tok)->encode(sample).value());
  // and so does an export to Hugging Face files followed by a re-import
  auto tmp = std::filesystem::temp_directory_path() / "gyre_gpt2_tok_roundtrip";
  std::filesystem::remove_all(tmp);
  ASSERT_TRUE((*tok)->save_huggingface(tmp));
  auto re = Tokenizer::load_huggingface(tmp);
  ASSERT_TRUE(re) << re.error().message;
  EXPECT_EQ((*re)->encode(sample).value(), (*tok)->encode(sample).value());
  std::filesystem::remove_all(tmp);
}
