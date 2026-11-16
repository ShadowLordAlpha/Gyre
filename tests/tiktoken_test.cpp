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

namespace {

std::vector<std::string> v1_split(const std::string& text, std::vector<std::string> specials = {}) {
  auto pt = gyre::make_tiktoken_v1_pretok(std::move(specials));
  std::string scratch;
  std::vector<std::string_view> spans;
  EXPECT_TRUE(pt->split(text, scratch, spans));
  std::vector<std::string> out(spans.begin(), spans.end());
  std::string joined;
  for (auto& s : out) joined += s;
  EXPECT_EQ(joined, text) << "spans must tile the input";
  return out;
}

// specials <|x|>=0, <|sep|>=1; ids 2..3 reserved (no bytes); then regular tokens.
const char* kSynthetic =
    "{\"reserved_tokens\":4,\"regular_tokens\":["
    "{\"bytes\":[97],\"token\":4},{\"bytes\":[98],\"token\":5},{\"bytes\":[32],\"token\":6},"
    "{\"bytes\":[10],\"token\":7},{\"bytes\":[195],\"token\":8},{\"bytes\":[169],\"token\":9},"
    "{\"bytes\":[97,98],\"token\":10},{\"bytes\":[32,97],\"token\":11},{\"bytes\":[32,97,98],\"token\":12},"
    "{\"bytes\":[195,169],\"token\":13}],"
    "\"special_tokens\":[{\"bytes\":[60,124,120,124,62],\"token\":0},"
    "{\"bytes\":[60,124,115,101,112,124,62],\"token\":1}],"
    "\"word_split\":\"V1\",\"vocab_size\":14}";

}  // namespace

// Expected spans are what Python `regex` gives for PAT_STR_B:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
TEST(Tiktoken, PretokWhitespaceRunsMatchRegex) {
  using V = std::vector<std::string>;
  // \s+(?!\S): a run before a non-space leaves its last char for the next piece
  EXPECT_EQ(v1_split("    foo"), (V{"   ", " foo"}));
  EXPECT_EQ(v1_split("a  b"), (V{"a", " ", " b"}));
  EXPECT_EQ(v1_split("  1"), (V{" ", " ", "1"}));
  EXPECT_EQ(v1_split("  !x"), (V{" ", " !", "x"}));
  EXPECT_EQ(v1_split("a \t b"), (V{"a", " \t", " b"}));
  EXPECT_EQ(v1_split("a   "), (V{"a", "   "}));  // a run that ends the text is taken whole
  // \s*[\r\n]+ ends after the LAST newline of the run
  EXPECT_EQ(v1_split("x\n    \n y"), (V{"x", "\n    \n", " y"}));
  EXPECT_EQ(v1_split("\n\n  \nz"), (V{"\n\n  \n", "z"}));
  EXPECT_EQ(v1_split("end  \n"), (V{"end", "  \n"}));
  EXPECT_EQ(v1_split("def f():\n    return 1\n"), (V{"def", " f", "():\n", "   ", " return", " ", "1", "\n"}));
  EXPECT_EQ(v1_split("Hi  there,   world!\n\n\tTabbed  line"),
            (V{"Hi", " ", " there", ",", "  ", " world", "!\n\n", "\tTabbed", " ", " line"}));
  // tiktoken runs the regex per segment between specials, so a run before a special is whole
  EXPECT_EQ(v1_split("a  <|x|>b", {"<|x|>"}), (V{"a", "  ", "<|x|>", "b"}));
}

// Ids from tiktoken 0.12: tiktoken.Encoding(pat_str=PAT_STR_B, mergeable_ranks=regular_tokens,
// special_tokens=special_tokens).encode(s, allowed_special="all").
TEST(Tiktoken, GoldenWhitespaceRuns) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases{
      {"    foo", {431, 22681}},
      {"a  b", {225, 160, 412}},
      {"def f():\n    return 1\n", {1406, 409, 10609, 431, 902, 160, 177, 138}},
      {"x\n    \n y", {248, 20795, 462}},
      {"Hi  there,   world!\n\n\tTabbed  line", {14226, 160, 894, 172, 384, 1749, 2588, 137, 14876, 2999, 160, 1638}},
  };
  for (auto& [text, want] : cases) {
    auto ids = (*tok)->encode(text);
    ASSERT_TRUE(ids) << ids.error().message;
    EXPECT_EQ(*ids, want) << text;
    EXPECT_EQ((*tok)->decode(*ids), text);
  }
}

// The nested Gyre document (tok export, checkpoint sidecar) must rebuild the same tiktoken tokenizer,
// not a merge-less BPE.
TEST(Tiktoken, GyreDocumentRoundTrip) {
  auto tok = gyre::load_tiktoken_json(kSynthetic);
  ASSERT_TRUE(tok) << tok.error().message;
  const std::string doc = (*tok)->to_json();
  auto again = gyre::Tokenizer::from_json(doc);
  ASSERT_TRUE(again) << again.error().message;
  EXPECT_STREQ((*again)->model_name(), "tiktoken_bpe");
  EXPECT_STREQ((*again)->pretok_name(), "tiktoken_v1");
  EXPECT_EQ((*again)->vocab_size(), (*tok)->vocab_size());
  const std::vector<std::string> texts{"ab ab", "a<|x|>b<|sep|> ab", "\xC3\xA9 ab\n", "  ab", ""};
  for (auto& t : texts) {
    auto want = (*tok)->encode(t);
    auto got = (*again)->encode(t);
    ASSERT_TRUE(want && got) << t;
    EXPECT_EQ(*got, *want) << t;
  }
  EXPECT_EQ((*again)->encode("ab ab").value(), (std::vector<std::int32_t>{10, 12}));
  EXPECT_EQ((*again)->encode("a<|x|>b<|sep|>").value(), (std::vector<std::int32_t>{4, 0, 5, 1}));

  // through a file, like `gyre-cli tok import ... --out x.gyre.json` and a later --tok x.gyre.json
  auto path = std::filesystem::temp_directory_path() / "gyre_tiktoken_doc" / "t.gyre.json";
  ASSERT_TRUE((*tok)->save(path));
  auto loaded = gyre::Tokenizer::load(path);
  ASSERT_TRUE(loaded) << loaded.error().message;
  EXPECT_STREQ((*loaded)->model_name(), "tiktoken_bpe");
  EXPECT_EQ((*loaded)->encode("a<|x|>ab ab").value(), (*tok)->encode("a<|x|>ab ab").value());
  std::filesystem::remove_all(path.parent_path());

  // Documents written before the ranks were saved fail loudly instead of becoming byte-level BPE.
  std::string old = doc;
  const std::string from = "\"ranks\":\"vocab\"";
  auto at = old.find(from);
  ASSERT_NE(at, std::string::npos);
  old.replace(at, from.size(), "\"ranks\":null");
  EXPECT_FALSE(gyre::Tokenizer::from_json(old));
  // ... and so do unknown model names.
  EXPECT_FALSE(gyre::Tokenizer::from_json(
      "{\"gyre\":\"tokenizer\",\"version\":1,\"tokenizer\":{\"model\":\"wordpiece\",\"vocab\":[\"a\"]}}"));
}

TEST(Tiktoken, Grok2GyreDocumentRoundTrip) {
  auto path = grok_tok_path();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "missing " << path.string();
  }
  auto tok = gyre::Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  auto again = gyre::Tokenizer::from_json((*tok)->to_json());
  ASSERT_TRUE(again) << again.error().message;
  EXPECT_EQ((*again)->vocab_size(), 131072);
  for (const std::string t : {"Human: What is Deep Learning?<|separator|>\n\n", "hello world", "<|eos|>",
                              "def f():\n    return 1\n", "caf\xC3\xA9 \xE6\x97\xA5\xE6\x9C\xAC 2024!"}) {
    EXPECT_EQ((*again)->encode(t).value(), (*tok)->encode(t).value()) << t;
  }
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
