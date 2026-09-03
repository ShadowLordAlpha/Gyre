#include "gyre/nn/tiktoken.hpp"
#include "gyre/nn/tokenize.hpp"

#include <fstream>
#include <filesystem>

#include <gtest/gtest.h>

namespace {

std::filesystem::path grok_tok_path() {
#ifdef GYRE_SOURCE_DIR
  return std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "grok2" / "tokenizer.tok.json";
#else
  return std::filesystem::path("data/grok2/tokenizer.tok.json");
#endif
}

void write_file(const std::filesystem::path& p, std::string_view s) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream o(p, std::ios::binary);
  o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

}  // namespace

TEST(Tiktoken, LoadGrok2VocabSize) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->vocab_size(), 131072);
  EXPECT_STREQ((*tok)->pretok_name(), "tiktoken_v1");
  EXPECT_STREQ((*tok)->model_name(), "tiktoken_bpe");
}

TEST(Tiktoken, GoldenHelloWorld) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode("hello world");
  ASSERT_TRUE(ids) << ids.error().message;
  ASSERT_EQ(ids->size(), 2u);
  EXPECT_EQ((*ids)[0], 21517);
  EXPECT_EQ((*ids)[1], 1749);
  EXPECT_EQ((*tok)->decode(*ids), "hello world");
}

TEST(Tiktoken, GoldenChatPrefix) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  const std::string text = "Human: What is Deep Learning?<|separator|>\n\n";
  auto ids = (*tok)->encode(text);
  ASSERT_TRUE(ids) << ids.error().message;
  const std::vector<std::int32_t> want{35406, 186, 2171, 458, 17454, 14803, 191, 1, 417};
  ASSERT_EQ(*ids, want);
  EXPECT_EQ((*tok)->decode(*ids), text);
}

TEST(Tiktoken, SpecialRoundTrip) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode("<|eos|>");
  ASSERT_TRUE(ids);
  ASSERT_EQ(ids->size(), 1u);
  EXPECT_EQ((*ids)[0], 2);
  auto sep = (*tok)->encode("<|separator|>");
  ASSERT_TRUE(sep);
  ASSERT_EQ(sep->size(), 1u);
  EXPECT_EQ((*sep)[0], 1);
}

TEST(Tiktoken, SyntheticMergeAndEmpty) {
  const char* json =
      "{\"reserved_tokens\":0,\"regular_tokens\":["
      "{\"bytes\":[97],\"token\":0},{\"bytes\":[98],\"token\":1},"
      "{\"bytes\":[97,98],\"token\":2}],"
      "\"special_tokens\":[{\"bytes\":[60,124,120,124,62],\"token\":3}],"
      "\"word_split\":\"V1\",\"vocab_size\":4}";
  auto tok = gyre::load_tiktoken_json(json);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->vocab_size(), 4);
  auto ab = (*tok)->encode("ab");
  ASSERT_TRUE(ab) << ab.error().message;
  ASSERT_EQ(ab->size(), 1u);
  EXPECT_EQ((*ab)[0], 2);
  auto empty = (*tok)->encode("");
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->empty());
  auto sp = (*tok)->encode("<|x|>");
  ASSERT_TRUE(sp);
  ASSERT_EQ(sp->size(), 1u);
  EXPECT_EQ((*sp)[0], 3);
}

TEST(Tiktoken, PretokLeadingSpaceAndUtf8) {
  const char* json =
      "{\"reserved_tokens\":0,\"regular_tokens\":["
      "{\"bytes\":[32],\"token\":0},{\"bytes\":[97],\"token\":1},"
      "{\"bytes\":[32,97],\"token\":2}],"
      "\"special_tokens\":[],\"word_split\":\"V1\",\"vocab_size\":3}";
  auto tok = gyre::load_tiktoken_json(json);
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode(" a");
  ASSERT_TRUE(ids) << ids.error().message;
  ASSERT_EQ(ids->size(), 1u);
  EXPECT_EQ((*ids)[0], 2);
}

TEST(Tiktoken, LoadViaTempFile) {
  auto dir = std::filesystem::temp_directory_path() / "gyre_tiktoken_test";
  auto path = dir / "tokenizer.tok.json";
  write_file(path,
             "{\"reserved_tokens\":0,\"regular_tokens\":[{\"bytes\":[97],\"token\":0}],"
             "\"special_tokens\":[],\"word_split\":\"V1\",\"vocab_size\":1}");
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  auto ids = (*tok)->encode("a");
  ASSERT_TRUE(ids);
  ASSERT_EQ(ids->size(), 1u);
  EXPECT_EQ((*ids)[0], 0);
}
