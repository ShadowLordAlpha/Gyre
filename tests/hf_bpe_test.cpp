// Hugging Face byte-level BPE with an explicit-id vocab (SmolLM2-360M): Digits+ByteLevel
// pretokenizer, added tokens at ids 0..16, merge results numbered by the vocab. Expected ids come
// from the Hugging Face `tokenizers` library (Tokenizer.from_file(...).encode(s,
// add_special_tokens=False)). Skips when data/smollm2-360m is not downloaded.
#include "gyre/nn/bpe.hpp"
#include "gyre/nn/tokenize.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>

using namespace gyre;

namespace {

// Writes `json` as <tmp>/<name>/tokenizer.json and returns the directory.
std::filesystem::path write_hf_dir(const std::string& name, std::string_view json) {
  auto dir = std::filesystem::temp_directory_path() / name;
  std::filesystem::create_directories(dir);
  std::ofstream o(dir / "tokenizer.json", std::ios::binary);
  o.write(json.data(), static_cast<std::streamsize>(json.size()));
  return dir;
}

const std::filesystem::path kSmolDir = std::filesystem::path(GYRE_SOURCE_DIR) / "data" / "smollm2-360m";
using Ids = std::vector<std::int32_t>;

std::vector<std::string> split_digits(const std::string& text) {
  auto pt = make_digits_bytelevel_pretok();
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

}  // namespace

TEST(HfBpe, DigitsPretokenizerIsolatesEveryNumber) {
  using V = std::vector<std::string>;
  EXPECT_EQ(split_digits("In 2024, x"), (V{"In", " ", "2", "0", "2", "4", ",", " x"}));
  EXPECT_EQ(split_digits("abc123 def"), (V{"abc", "1", "2", "3", " def"}));
  EXPECT_EQ(split_digits("a  2"), (V{"a", "  ", "2"}));  // each run is regex-split on its own
  // \p{N} beyond ASCII: Arabic-Indic digits (Nd), one half (No), Roman eight (Nl)
  EXPECT_EQ(split_digits("\xD9\xA3\xD9\xA4 \xC2\xBD"), (V{"\xD9\xA3", "\xD9\xA4", " ", "\xC2\xBD"}));
  EXPECT_EQ(split_digits("\xE2\x85\xA7"), (V{"\xE2\x85\xA7"}));
  EXPECT_EQ(split_digits(""), V{});
  split_digits("\xFF\xBD 1 \x80");  // invalid bytes never count as numbers and still tile
}

TEST(HfBpe, SmolLM2MatchesHuggingFaceTokenIds) {
  if (!std::filesystem::exists(kSmolDir / "tokenizer.json")) GTEST_SKIP() << "data/smollm2-360m not downloaded";
  auto tok = Tokenizer::load_huggingface(kSmolDir);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->vocab_size(), 49152);
  EXPECT_STREQ((*tok)->pretok_name(), "digits_bytelevel");
  EXPECT_EQ((*tok)->specials().size(), 17u);
  auto ids = [&](const std::string& s) {
    auto e = (*tok)->encode(s);
    EXPECT_TRUE(e) << (e ? "" : e.error().message);
    return e ? *e : Ids{};
  };
  EXPECT_EQ(ids("Hello world"), (Ids{19556, 905}));
  EXPECT_EQ(ids("The capital of France is"), (Ids{504, 3575, 282, 4649, 314}));
  EXPECT_EQ(ids("In 2024, prices rose 3.5%."),
            (Ids{788, 216, 34, 32, 34, 36, 28, 5770, 8739, 216, 35, 30, 37, 8006}));
  EXPECT_EQ(ids("I'm sure they'll say it's fine, don't you?"),
            (Ids{57, 5248, 2090, 502, 3060, 1643, 357, 506, 4979, 28, 1326, 982, 346, 47}));
  EXPECT_EQ(ids("  leading spaces\n\nand newlines\n"), (Ids{216, 2899, 5600, 198, 198, 397, 725, 5110, 198}));
  EXPECT_EQ(ids("caf\xC3\xA9 na\xC3\xAFve \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x98\x80"),
            (Ids{83, 1939, 2756, 15486, 46494, 17097, 241, 115, 40993, 179, 120, 248, 40303, 218}));
  EXPECT_EQ(ids("Tabs\tand\r\nCRLF 123456789"),
            (Ids{68, 7366, 197, 397, 201, 198, 9088, 60, 54, 216, 33, 34, 35, 36, 37, 38, 39, 40, 41}));
  EXPECT_EQ(ids("\xD9\xA3\xD9\xA4 \xC2\xBD \xE2\x85\xA7"), (Ids{164, 113, 164, 114, 216, 16738, 216, 173, 223, 117}));
  // added tokens are matched literally, also inside text
  EXPECT_EQ(ids("<|endoftext|>"), (Ids{0}));
  EXPECT_EQ(ids("a<|im_start|>user\nhi<|im_end|>b"), (Ids{81, 1, 4093, 198, 6004, 2, 82}));
  // bytes missing from the vocab (0x04, 0xF1 lead byte) are dropped like Hugging Face does
  EXPECT_EQ(ids("x \x04 y \xF1\x90\x80\x80 z"), (Ids{104, 216, 329, 216, 234, 218, 218, 1892}));

  for (const std::string s : {"Hello world", "  leading and trailing  ", "line1\nline2\n\n\nline3",
                              "caf\xC3\xA9 na\xC3\xAFve \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x98\x80",
                              "I'm sure they'll say it's 1,234.56 or something!?", "tabs\tand\r\nCRLF",
                              "<|im_start|>system\nYou are 42.<|im_end|>\n<|endoftext|>"}) {
    EXPECT_EQ((*tok)->decode(ids(s)), s) << s;
  }

  // The Gyre JSON document (what a .gyre checkpoint freezes) rebuilds the same tokenizer.
  auto again = Tokenizer::from_json((*tok)->to_json());
  ASSERT_TRUE(again) << again.error().message;
  EXPECT_STREQ((*again)->pretok_name(), "digits_bytelevel");
  for (const std::string s : {"To be, or not to be: that is the question.\nWhether 'tis nobler in 1603",
                              "a<|im_start|>user\nhi<|im_end|>b", "x \x04 y"}) {
    EXPECT_EQ((*again)->encode(s).value(), (*tok)->encode(s).value()) << s;
  }
  // Writing Hugging Face files for an explicit-id vocab is not implemented: it must say so.
  auto tmp = std::filesystem::temp_directory_path() / "gyre_smol_tok_export";
  EXPECT_FALSE((*tok)->save_huggingface(tmp));
  std::filesystem::remove_all(tmp);
}

// Explicit merge ids (any numbering) give the same pieces as the sequential definition: merging the
// lowest-rank pair repeatedly. Small synthetic vocab so it runs without downloads.
TEST(HfBpe, ExplicitMergeIdsFollowVocabNumbering) {
  PieceTable t;
  t.identity256 = false;
  // ids: 0 "<s>", 1 "a", 2 "b", 3 "c", 4 "bc", 5 "abc", 6 "ab"
  t.pieces = {"<s>", "a", "b", "c", "bc", "abc", "ab"};
  for (std::int32_t i = 1; i <= 3; ++i) t.byte_to_id[static_cast<std::uint8_t>(t.pieces[i][0])] = i;
  // ranks: (b,c)->4, (a,bc)->5, (a,b)->6
  BpeModel m({{2, 3}, {1, 4}, {1, 2}});
  m.set_merge_ids({4, 5, 6});
  std::vector<std::int32_t> out;
  ASSERT_TRUE(m.encode_span("abcab", t, out));
  EXPECT_EQ(out, (Ids{5, 6}));
  out.clear();
  ASSERT_TRUE(m.encode_span("axb", t, out));  // 'x' is not in the vocab: dropped, then a+b merges
  EXPECT_EQ(out, (Ids{6}));
}

// Real Hugging Face files hold real UTF-8 text. "é" must stay the two bytes C3 A9 (not the
// single byte E9) and Cyrillic "а" must not fold onto "0". Expected ids from the Hugging Face
// `tokenizers` library (0.22) for this exact file.
TEST(HfImport, UnigramKeepsRealUtf8Pieces) {
  const auto dir = write_hf_dir("gyre_hf_unigram_utf8", R"({"version":"1.0","truncation":null,"padding":null,
    "added_tokens":[],"normalizer":null,
    "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false},
    "post_processor":null,"decoder":null,
    "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁",-3.0],["▁café",-1.0],
      ["é",-4.0],["c",-4.0],["a",-4.0],["f",-4.0],["а",-2.0],["0",-2.0],["▁naïve",-1.0],
      ["n",-4.0]],"byte_fallback":false}})");
  auto tok = Tokenizer::load_huggingface(dir);
  ASSERT_TRUE(tok) << tok.error().message;
  const auto& p = (*tok)->pieces().pieces;
  ASSERT_EQ(p.size(), 11u);
  EXPECT_EQ(p[2], "\xE2\x96\x81" "caf\xC3\xA9");
  EXPECT_EQ(p[3], "\xC3\xA9");
  EXPECT_EQ(p[7], "\xD0\xB0");
  EXPECT_EQ(p[8], "0");
  EXPECT_EQ((*tok)->encode("caf\xC3\xA9").value(), (Ids{2}));
  EXPECT_EQ((*tok)->encode("\xD0\xB0" "0").value(), (Ids{1, 7, 8}));
  EXPECT_EQ((*tok)->encode("0\xD0\xB0").value(), (Ids{1, 8, 7}));
  EXPECT_EQ((*tok)->encode("caf\xC3\xA9 na\xC3\xAFve").value(), (Ids{2, 9}));
  std::filesystem::remove_all(dir);
}

// A plain (not byte-level) BPE whose specials come first: merge results are numbered by the vocab
// (bc=4, ab=5, abc=6), not alphabet + rank. Expected ids from Hugging Face `tokenizers` (0.22).
TEST(HfImport, PlainBpeUsesVocabMergeIds) {
  const auto dir = write_hf_dir("gyre_hf_plain_bpe", R"({"version":"1.0","truncation":null,"padding":null,
    "added_tokens":[{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,
      "normalized":false,"special":true}],
    "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
    "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,
      "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
      "vocab":{"[UNK]":0,"a":1,"b":2,"c":3,"bc":4,"ab":5,"abc":6},
      "merges":[["b","c"],["a","b"],["a","bc"]]}})");
  auto tok = Tokenizer::load_huggingface(dir);
  ASSERT_TRUE(tok) << tok.error().message;
  const std::vector<std::pair<std::string, Ids>> cases{
      {"ab", {5}}, {"abc", {6}}, {"abcab", {6, 5}}, {"cab", {3, 5}}, {"b[UNK]a", {2, 0, 1}}, {"axb", {5}}};
  for (auto& [text, want] : cases) EXPECT_EQ((*tok)->encode(text).value(), want) << text;
  EXPECT_EQ((*tok)->decode(Ids{6, 5}), "abcab");
  // The frozen Gyre document keeps the vocab numbering.
  auto again = Tokenizer::from_json((*tok)->to_json());
  ASSERT_TRUE(again) << again.error().message;
  for (auto& [text, want] : cases) EXPECT_EQ((*again)->encode(text).value(), want) << text;
  std::filesystem::remove_all(dir);
}

// What BpeModel cannot reproduce is refused at load time instead of encoding differently.
TEST(HfImport, PlainBpeRejectsLayoutsItCannotEncode) {
  auto load = [](const std::string& name, const std::string& vocab, const std::string& merges) {
    const auto dir = write_hf_dir(name, R"({"version":"1.0","added_tokens":[],"normalizer":null,
      "pre_tokenizer":null,"model":{"type":"BPE","unk_token":null,"vocab":)" + vocab +
                                            R"(,"merges":)" + merges + "}}");
    auto tok = Tokenizer::load_huggingface(dir);
    std::filesystem::remove_all(dir);
    return tok;
  };
  // the ASCII subset loads
  ASSERT_TRUE(load("gyre_hf_bpe_ok", R"({"a":0,"b":1,"ab":2})", R"(["a b"])"));
  // a multi-byte alphabet symbol: encode starts from bytes and could never produce id 3
  auto chars = load("gyre_hf_bpe_chars", R"({"a":0,"b":1,"ab":2,"é":3})", R"(["a b"])");
  ASSERT_FALSE(chars);
  EXPECT_NE(chars.error().message.find("character-level"), std::string::npos) << chars.error().message;
  // a merge whose result is missing from the vocab
  EXPECT_FALSE(load("gyre_hf_bpe_nores", R"({"a":0,"b":1,"c":2})", R"(["a b"])"));
  // a merge whose parts are missing (used to be skipped silently, shifting every later rank)
  EXPECT_FALSE(load("gyre_hf_bpe_noparts", R"({"a":0,"b":1,"ab":2})", R"(["a x","a b"])"));
}

// A Gyre document whose merge results are not at alphabet + rank and that has no merge_ids would
// encode to the wrong ids: refuse it.
TEST(HfImport, GyreDocumentWithInconsistentLayoutFails) {
  EXPECT_FALSE(Tokenizer::from_json(R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"bpe",
    "pretoken":"identity","vocab":["<s>","a","b","ab"],"merges":[[1,2]]}})"));
  auto ok = Tokenizer::from_json(R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"bpe",
    "pretoken":"identity","vocab":["<s>","a","b","ab"],"merges":[[1,2]],"merge_ids":[3]}})");
  ASSERT_TRUE(ok) << ok.error().message;
  EXPECT_EQ((*ok)->encode("ab").value(), (Ids{3}));
}

// Gyre's own Hugging Face export (raw bytes as U+0080..U+00FF, pieces with spaces) still round-trips
// for non-ASCII text, for BPE and Unigram.
TEST(HfImport, GyreExportRoundTripNonAscii) {
  const std::string text =
      "caf\xC3\xA9 na\xC3\xAFve fa\xC3\xA7" "ade caf\xC3\xA9 na\xC3\xAFve o w o w \xD0\xB0\xD0\xB1 \xD0\xB0\xD0\xB1 "
      "\xE6\x97\xA5\xE6\x9C\xAC caf\xC3\xA9 na\xC3\xAFve o w";
  auto bpe = Tokenizer::train_bpe(text, 300);
  ASSERT_TRUE(bpe) << bpe.error().message;
  auto uni = Tokenizer::train_unigram(text, 120);
  ASSERT_TRUE(uni) << uni.error().message;
  for (auto* tok : {bpe->get(), uni->get()}) {
    const auto dir = std::filesystem::temp_directory_path() / "gyre_hf_export_utf8";
    std::filesystem::remove_all(dir);
    ASSERT_TRUE(tok->save_huggingface(dir));
    auto re = Tokenizer::load_huggingface(dir);
    ASSERT_TRUE(re) << tok->model_name() << ": " << re.error().message;
    EXPECT_STREQ((*re)->model_name(), tok->model_name());
    EXPECT_EQ((*re)->pieces().pieces, tok->pieces().pieces) << tok->model_name();
    EXPECT_EQ((*re)->encode(text).value(), tok->encode(text).value()) << tok->model_name();
    std::filesystem::remove_all(dir);
  }
}

// Regression: decode was a plain concat, so Metaspace tokenizers returned "▁café▁naïve".
TEST(HfImport, MetaspaceDecodeRestoresSpaces) {
  const auto dir = write_hf_dir("gyre_hf_unigram_decode", R"({"version":"1.0","truncation":null,"padding":null,
    "added_tokens":[],"normalizer":null,
    "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false},
    "post_processor":null,"decoder":null,
    "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁",-3.0],["▁café",-1.0],
      ["é",-4.0],["c",-4.0],["a",-4.0],["f",-4.0],["а",-2.0],["0",-2.0],["▁naïve",-1.0],
      ["n",-4.0]],"byte_fallback":false}})");
  auto tok = Tokenizer::load_huggingface(dir);
  ASSERT_TRUE(tok) << tok.error().message;
  for (const std::string s : {"caf\xC3\xA9 na\xC3\xAFve", "caf\xC3\xA9", "a  0", " caf\xC3\xA9"}) {
    auto ids = (*tok)->encode(s);
    ASSERT_TRUE(ids) << s;
    EXPECT_EQ((*tok)->decode(*ids), s);
  }
  std::filesystem::remove_all(dir);
}

// unk_token: a character missing from the vocab becomes the unk id (Hugging Face BPE.merge_word),
// one per UTF-8 character, and fuse_unk collapses a run. Without unk_token it is dropped.
TEST(HfImport, PlainBpeUnkTokenEmitsUnk) {
  auto load = [](const std::string& name, const std::string& unk, bool fuse) {
    const auto dir = write_hf_dir(name, R"({"version":"1.0","added_tokens":[{"id":0,"content":"[UNK]",
      "single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
      "normalizer":null,"pre_tokenizer":null,
      "model":{"type":"BPE","dropout":null,"unk_token":)" + unk + R"(,"continuing_subword_prefix":null,
        "end_of_word_suffix":null,"fuse_unk":)" + (fuse ? "true" : "false") + R"(,"byte_fallback":false,
        "vocab":{"[UNK]":0,"a":1,"b":2,"c":3,"bc":4,"ab":5,"abc":6},
        "merges":[["b","c"],["a","b"],["a","bc"]]}})");
    auto tok = Tokenizer::load_huggingface(dir);
    std::filesystem::remove_all(dir);
    return tok;
  };
  auto tok = load("gyre_hf_bpe_unk", R"("[UNK]")", false);
  ASSERT_TRUE(tok) << tok.error().message;
  const std::string e_acute = "\xC3\xA9";  // one character, two bytes: one unk
  const std::vector<std::pair<std::string, Ids>> cases{
      {"axb", {1, 0, 2}}, {"a" + e_acute + "b", {1, 0, 2}}, {"axyb", {1, 0, 0, 2}}, {"abc", {6}}, {"x", {0}}};
  for (auto& [text, want] : cases) EXPECT_EQ((*tok)->encode(text).value(), want) << text;
  auto again = Tokenizer::from_json((*tok)->to_json());  // the frozen document keeps the unk id
  ASSERT_TRUE(again) << again.error().message;
  for (auto& [text, want] : cases) EXPECT_EQ((*again)->encode(text).value(), want) << text;

  auto fused = load("gyre_hf_bpe_unk_fuse", R"("[UNK]")", true);
  ASSERT_TRUE(fused) << fused.error().message;
  EXPECT_EQ((*fused)->encode("axy" + e_acute + "b").value(), (Ids{1, 0, 2}));
  EXPECT_EQ((*fused)->encode("xay").value(), (Ids{0, 1, 0}));

  auto none = load("gyre_hf_bpe_nounk", "null", false);  // dropped, as before
  ASSERT_TRUE(none) << none.error().message;
  EXPECT_EQ((*none)->encode("axb").value(), (Ids{5}));
  // an unk_token that is not in the vocab is refused
  auto missing = load("gyre_hf_bpe_badunk", R"("<unk>")", false);
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().message.find("unk_token"), std::string::npos) << missing.error().message;
}

// Same with an alphabet + rank layout (no explicit merge ids): an unknown byte used to be an
// "unknown byte" encode error even though the file names an unk_token.
TEST(HfImport, ImplicitLayoutBpeUnkToken) {
  const auto dir = write_hf_dir("gyre_hf_bpe_unk_implicit", R"({"version":"1.0","added_tokens":[],
    "normalizer":null,"pre_tokenizer":null,"model":{"type":"BPE","unk_token":"<unk>",
    "vocab":{"a":0,"b":1,"ab":2,"<unk>":3},"merges":["a b"]}})");
  auto tok = Tokenizer::load_huggingface(dir);
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->encode("ab?ab").value(), (Ids{2, 3, 2}));
  EXPECT_EQ((*tok)->encode("a\xE6\x97\xA5" "b").value(), (Ids{0, 3, 1}));
}

// Hugging Face pipeline stages Gyre does not implement fail loudly instead of silently encoding
// with the identity pretokenizer / un-normalized text.
TEST(HfImport, UnsupportedPipelineStagesFail) {
  auto try_load = [](const std::string& name, const std::string& normalizer, const std::string& pre,
                     const std::string& model) {
    const auto dir = write_hf_dir(name, R"({"version":"1.0","added_tokens":[],"normalizer":)" + normalizer +
                                            R"(,"pre_tokenizer":)" + pre + R"(,"model":)" + model + "}");
    auto tok = Tokenizer::load_huggingface(dir);
    std::filesystem::remove_all(dir);
    return tok;
  };
  const std::string bpe = R"({"type":"BPE","unk_token":null,"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]})";
  const std::string uni = R"({"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-1.0],["b",-1.0]]})";
  auto expect_refused = [&](const std::string& normalizer, const std::string& pre, const std::string& model,
                            const char* what) {
    auto r = try_load("gyre_hf_refused", normalizer, pre, model);
    ASSERT_FALSE(r) << "loaded: " << normalizer << " " << pre << " " << model;
    EXPECT_EQ(r.error().code, Errc::unsupported) << r.error().message;
    EXPECT_NE(r.error().message.find(what), std::string::npos) << r.error().message;
  };
  for (const char* pre : {R"({"type":"Whitespace"})", R"({"type":"WhitespaceSplit"})", R"({"type":"BertPreTokenizer"})",
                          R"({"type":"Punctuation","behavior":"Isolated"})",
                          R"({"type":"Split","pattern":{"Regex":"\\s+"},"behavior":"Removed","invert":false})",
                          R"({"type":"CharDelimiterSplit","delimiter":" "})"}) {
    expect_refused("null", pre, bpe, "pre_tokenizer");
    expect_refused("null", pre, uni, "pre_tokenizer");
  }
  expect_refused("null", R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true})",
                 R"({"type":"BPE","vocab":{"a":0},"merges":[]})", "add_prefix_space");
  expect_refused("null", R"({"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true})", uni,
                 "replacement");
  for (const char* norm : {R"({"type":"NFC"})", R"({"type":"Lowercase"})",
                           R"({"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true})",
                           R"({"type":"Sequence","normalizers":[{"type":"NFKC"}]})"}) {
    expect_refused(norm, "null", bpe, "normalizer");
    expect_refused(norm, "null", uni, "normalizer");
  }
  expect_refused("null", "null", R"({"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0,"a":1}})", "WordPiece");
  expect_refused("null", "null",
                 R"({"type":"BPE","continuing_subword_prefix":"##","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]})",
                 "continuing_subword_prefix");
  expect_refused("null", "null", R"({"type":"BPE","byte_fallback":true,"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]})",
                 "byte_fallback");

  // What is supported still loads: null stages, an empty normalizer Sequence, GPT-2 style
  // ByteLevel options and the bare {"type":"ByteLevel"} older Gyre exports wrote.
  EXPECT_TRUE(try_load("gyre_hf_ok1", "null", "null", bpe));
  EXPECT_TRUE(try_load("gyre_hf_ok2", R"({"type":"Sequence","normalizers":[]})", "null", uni));
  const std::string bl = R"({"vocab":{"a":0},"merges":[]})";
  EXPECT_TRUE(try_load("gyre_hf_ok3", "null", R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true})", bl));
  EXPECT_TRUE(try_load("gyre_hf_ok4", "null", R"({"type":"ByteLevel"})", bl));
}

// Metaspace options are honoured (and survive the Gyre document): split=true cuts before every
// U+2581, prepend_scheme "never" drops the dummy prefix.
TEST(HfImport, MetaspaceSplitAndPrependScheme) {
  auto load = [](const std::string& name, const std::string& pre) {
    const auto dir = write_hf_dir(name, R"({"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":)" + pre +
                                            R"(,"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],
      ["▁a▁b",-0.5],["▁a",-2.0],["▁b",-2.0],["▁",-3.0],["a",-4.0],["b",-4.0]]}})");
    auto tok = Tokenizer::load_huggingface(dir);
    std::filesystem::remove_all(dir);
    return tok;
  };
  auto whole = load("gyre_hf_ms_whole", R"({"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false})");
  auto split = load("gyre_hf_ms_split", R"({"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true})");
  auto never = load("gyre_hf_ms_never", R"({"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true})");
  ASSERT_TRUE(whole && split && never);
  EXPECT_EQ((*whole)->encode("a b").value(), (Ids{1}));     // "▁a▁b" spans the space
  EXPECT_EQ((*split)->encode("a b").value(), (Ids{2, 3}));  // "▁a" | "▁b"
  EXPECT_EQ((*never)->encode("a b").value(), (Ids{5, 3}));  // "a" | "▁b"
  EXPECT_STREQ((*split)->pretok_name(), "metaspace_split");
  EXPECT_STREQ((*never)->pretok_name(), "metaspace_noprefix_split");
  for (auto* t : {whole->get(), split->get(), never->get()}) {
    auto re = Tokenizer::from_json(t->to_json());
    ASSERT_TRUE(re) << re.error().message;
    EXPECT_EQ((*re)->encode("a b").value(), t->encode("a b").value()) << t->pretok_name();
    EXPECT_EQ(t->decode(t->encode("a b").value()), "a b") << t->pretok_name();
  }
}

// Old Gyre exports wrote merges as "a b" strings although pieces may contain spaces.
TEST(HfImport, StringMergesWithSpacesResolveOrFail) {
  // (1) Truly ambiguous: "a  b" is both "a"+" b" and "a "+"b", each with its result "a b" at
  // alphabet + rank. Picking one silently changes merge ranks, so the file is refused.
  const std::string vocab = R"({"a":0," ":1,"b":2,"a ":3," b":4,"a b":5})";
  auto load = [](const std::string& name, const std::string& pre, const std::string& model) {
    const auto dir = write_hf_dir(name, R"({"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":)" +
                                            pre + R"(,"model":)" + model + "}");
    auto tok = Tokenizer::load_huggingface(dir);
    std::filesystem::remove_all(dir);
    return tok;
  };
  const std::string gyre_pre = R"({"type":"Sequence"})";  // Gyre's identity-export marker
  auto amb = load("gyre_hf_amb", gyre_pre,
                  R"({"type":"BPE","unk_token":null,"vocab":)" + vocab + R"(,"merges":["a  ","  b","a  b"]})");
  ASSERT_FALSE(amb);
  EXPECT_NE(amb.error().message.find("ambiguous"), std::string::npos) << amb.error().message;
  // The same merges as [a, b] pairs are unambiguous.
  auto pairs = load("gyre_hf_amb_pairs", gyre_pre,
                    R"({"type":"BPE","unk_token":null,"vocab":)" + vocab +
                        R"(,"merges":[["a"," "],[" ","b"],["a ","b"]]})");
  ASSERT_TRUE(pairs) << pairs.error().message;
  EXPECT_EQ((*pairs)->encode("a b").value(), (Ids{5}));

  // (2) Resolvable: "a b c" splits as "a"+"b c" or "a b"+"c", but only "a bc" is in the vocab.
  // With [UNK] first the layout is not alphabet + rank, so the old loader took the first split
  // and then failed with "merge result not in vocab".
  auto ok = load("gyre_hf_amb_ok", "null", R"({"type":"BPE","unk_token":"[UNK]",
    "vocab":{"[UNK]":0,"a":1," ":2,"b":3,"c":4,"a ":5,"a b":6," c":7,"b c":8,"a bc":9},
    "merges":["a  ","a  b","  c","b  c","a b c"]})");
  ASSERT_TRUE(ok) << ok.error().message;
  const auto& bm = dynamic_cast<const BpeModel&>((*ok)->model());
  ASSERT_EQ(bm.merges().size(), 5u);
  EXPECT_EQ(bm.merges()[4], (std::pair<std::int32_t, std::int32_t>{6, 4}));
  EXPECT_EQ(bm.merges()[1], (std::pair<std::int32_t, std::int32_t>{5, 3}));
  EXPECT_EQ((*ok)->decode((*ok)->encode("a b c a bc").value()), "a b c a bc");
}

namespace {

void pb_varint(std::string& o, std::uint64_t v) {
  while (v >= 0x80) {
    o.push_back(static_cast<char>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  o.push_back(static_cast<char>(v));
}

// ModelProto.pieces entry {piece, score, type}.
void pb_piece(std::string& model, const std::string& piece, float score, int type) {
  std::string sub;
  pb_varint(sub, (1 << 3) | 2);
  pb_varint(sub, piece.size());
  sub += piece;
  pb_varint(sub, (2 << 3) | 5);
  char f[4];
  std::memcpy(f, &score, 4);
  sub.append(f, 4);
  pb_varint(sub, (3 << 3) | 0);
  pb_varint(sub, static_cast<std::uint64_t>(type));
  pb_varint(model, (1 << 3) | 2);
  pb_varint(model, sub.size());
  model += sub;
}

}  // namespace

// Regression: SentencePiece piece types were ignored. <s>/</s> (CONTROL) matched literal text,
// byte-fallback pieces <0xNN> were never used (unknown characters became id 0), and decode left
// the U+2581 markers in.
TEST(SentencePiece, PieceTypesByteFallbackAndUnk) {
  const std::string ms = "\xE2\x96\x81";
  std::string m;
  pb_piece(m, "<unk>", 0.f, 2);         // 0 UNKNOWN
  pb_piece(m, "<s>", 0.f, 3);           // 1 CONTROL
  pb_piece(m, "</s>", 0.f, 3);          // 2 CONTROL
  pb_piece(m, "<0xE2>", 0.f, 6);        // 3 BYTE
  pb_piece(m, "<0x82>", 0.f, 6);        // 4 BYTE
  pb_piece(m, "<0xAC>", 0.f, 6);        // 5 BYTE
  pb_piece(m, ms, -2.f, 1);             // 6
  pb_piece(m, ms + "hi", -1.f, 1);      // 7
  pb_piece(m, "h", -3.f, 1);            // 8
  pb_piece(m, "i", -3.f, 1);            // 9
  const auto path = std::filesystem::temp_directory_path() / "gyre_sp_types.model";
  {
    std::ofstream o(path, std::ios::binary);
    o.write(m.data(), static_cast<std::streamsize>(m.size()));
  }
  auto tok = Tokenizer::load(path);
  ASSERT_TRUE(tok) << tok.error().message;
  EXPECT_EQ((*tok)->encode("hi").value(), (Ids{7}));
  // "€" has no piece: its UTF-8 bytes become the BYTE pieces, and decode gives the text back.
  const std::string euro = "hi \xE2\x82\xAC";
  auto ids = (*tok)->encode(euro);
  ASSERT_TRUE(ids) << ids.error().message;
  EXPECT_EQ(*ids, (Ids{7, 6, 3, 4, 5}));
  EXPECT_EQ((*tok)->decode(*ids), euro);
  // "<s>" is not matched as the CONTROL piece; uncovered characters without byte pieces become
  // one merged <unk>.
  EXPECT_EQ((*tok)->encode("<s>").value(), (Ids{6, 0}));
  // The Gyre document keeps the types and the unk id.
  auto re = Tokenizer::from_json((*tok)->to_json());
  ASSERT_TRUE(re) << re.error().message;
  EXPECT_EQ((*re)->encode(euro).value(), *ids);
  EXPECT_EQ((*re)->encode("<s>").value(), (Ids{6, 0}));
  std::filesystem::remove(path);
}

// Regression: Unigram train cut rare single bytes by count, and the byte fallback then emitted
// id 0 for them, so decode(encode(x)) != x.
TEST(UnigramTrain, KeepsEveryByteAndNeverEmitsSilentZero) {
  std::string text;
  for (int i = 0; i < 200; ++i) text += "the cat sat on the mat. ";
  text += "\xC3\xA9";  // a rare non-ASCII character, seen once
  auto big = Tokenizer::train_unigram(text, 600);
  ASSERT_TRUE(big) << big.error().message;
  for (const std::string s : {text, std::string("caf\xC3\xA9 \x01\xFF z"), std::string("\xE2\x82\xAC")}) {
    auto ids = (*big)->encode(s);
    ASSERT_TRUE(ids) << ids.error().message;
    EXPECT_EQ((*big)->decode(*ids), s);
  }
  // A small vocab keeps every byte of the training text; an unseen byte is an error, not id 0.
  auto small = Tokenizer::train_unigram(text, 40);
  ASSERT_TRUE(small) << small.error().message;
  auto ids = (*small)->encode(text);
  ASSERT_TRUE(ids) << ids.error().message;
  EXPECT_EQ((*small)->decode(*ids), text);
  EXPECT_FALSE((*small)->encode("zzz"));
  // ... and survives a Gyre document round trip.
  auto re = Tokenizer::from_json((*small)->to_json());
  ASSERT_TRUE(re) << re.error().message;
  EXPECT_FALSE((*re)->encode("zzz"));
  EXPECT_EQ((*re)->encode(text).value(), *ids);
  EXPECT_FALSE(Tokenizer::train_unigram("abcdefghijklmnopqrstu", 16));  // fewer slots than distinct bytes
}

// Regression: BPE train broke count ties by unordered_map iteration order (toolchain dependent).
TEST(BpeTrain, TiesGoToTheSmallestPair) {
  // xy and ab both occur twice; (a,b) < (x,y).
  auto tok = Tokenizer::train_bpe("xyxyabab", 257);
  ASSERT_TRUE(tok) << tok.error().message;
  ASSERT_GE((*tok)->pieces().pieces.size(), 257u);
  EXPECT_EQ((*tok)->pieces().pieces[256], "ab");
}

// Regression: wrong-typed fields in tokenizer files threw nlohmann::type_error out of the Result
// API (terminating the CLI), and huge ids sized the piece table.
TEST(TokenizerLoad, MalformedFilesAreErrorsNotThrows) {
  const char* docs[] = {
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"bpe","pretoken":"identity","vocab":[1]}})",
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"unigram","pretoken":"identity","vocab":["a"],"scores":["x"]}})",
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"bpe","pretoken":"identity","vocab":{"a":"x"}}})",
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"bpe","pretoken":"identity","vocab":{"a":2000000000}}})",
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"unigram","pretoken":"identity","vocab":["a"],"scores":[0],"unk_id":5}})",
      R"({"gyre":"tokenizer","version":1,"tokenizer":{"model":"tiktoken_bpe","ranks":"vocab","vocab":[3]}})",
  };
  for (const char* d : docs) {
    Result<std::unique_ptr<Tokenizer>> r = std::unexpected(make_error(Errc::ok, ""));
    EXPECT_NO_THROW(r = Tokenizer::from_json(d)) << d;
    EXPECT_FALSE(r) << d;
  }
  EXPECT_NO_THROW((void)Tokenizer::from_json(R"({"merges":[[1,"x"]]})"));
  const char* hf[] = {
      R"({"model":{"type":"BPE","vocab":{"a":"x"},"merges":[]}})",
      R"({"model":{"type":"BPE","vocab":{"a":2000000000},"merges":[]}})",
      R"({"model":{"type":"Unigram","unk_id":0,"vocab":[["a","x"]]}})",
      R"({"model":{"type":"Unigram","unk_id":7,"vocab":[["a",0.0]]}})",
  };
  for (const char* j : hf) {
    const auto dir = write_hf_dir("gyre_hf_malformed", j);
    Result<std::unique_ptr<Tokenizer>> r = std::unexpected(make_error(Errc::ok, ""));
    EXPECT_NO_THROW(r = Tokenizer::load_huggingface(dir)) << j;
    EXPECT_FALSE(r) << j;
    std::filesystem::remove_all(dir);
  }
}
