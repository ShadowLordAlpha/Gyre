# Tokenizers

Gyre tokenizers are **not** the decoder. `CharLM` consumes integer ids. A tokenizer maps UTF-8 bytes ↔ ids and is saved separately so you can reuse it, import GPT-2 / Hugging Face files, or export ours.

**Reference:** `include/gyre/nn/tokenize.hpp`, `bpe.hpp`, `unigram.hpp`.  
**CLI:** `gyre-cli tok …` and `lm train --tok`.

## Composition

```
Tokenizer                 encode / decode / save  (not virtual)
  pretok_ : Pretokenizer  split the whole string once
  model_  : VocabModel    span → ids
  pieces_ : PieceTable    id → raw bytes (decode)
```

`Tokenizer::encode` always:

1. `pretok_->split(text, scratch, spans)` — one pass over the full buffer.
2. For each span, `model_->encode_span(span, pieces, out)`.
3. Concatenate ids.

The LM never sees text after that. Heap (`unique_ptr`) is expected; tokenize cost is small vs GEMM.

To add a kind, implement **either** a `Pretokenizer` **or** a `VocabModel` (sometimes both), then a factory + JSON key. Do not add `Gpt2Tokenizer` / `CharTokenizer` product classes.

| Pretokenizer `name()` | Role |
| --- | --- |
| `identity` | Whole UTF-8 as one span (Gyre BPE, chars, bytes) |
| `bytelevel` | GPT-2 regex + bytes-to-unicode (Hugging Face GPT-2) |
| `metaspace` | SentencePiece dummy prefix; space → `▁` |
| `tiktoken_v1` | Grok-2 `word_split=V1` (GPT-4-style `PAT_STR_B` + Unicode `\p{L}`/`\p{N}`) |

| VocabModel `name()` | Role |
| --- | --- |
| `bpe` | Merge ranks. **Zero merges** = chars or bytes |
| `unigram` | Piece scores + Viterbi (SentencePiece default) |
| `tiktoken_bpe` | tiktoken ranks (`bytes` → id). Not Gyre sequential-merge BPE |

**Chars** = `identity` + `bpe` + empty merges + alphabet = unique bytes in the train split (unknown byte → error).  
**Bytes** = `identity` + `bpe` + empty merges + identity 0..255.  
**Gyre BPE** = `identity` + `bpe` + learned merges on base 256 (same encode as before: apply merges in train order on the whole byte stream).  
**GPT-2** = `bytelevel` + `bpe` loaded from HF files. Do **not** run Gyre BPE checkpoints through the GPT-2 pretok.

CLI `--tokenizer chars|bytes|bpe|unigram` only picks how to **create** a tokenizer when `--tok` is omitted. After load, `model_name()` / `pretok_name()` are the source of truth.

## File: `*.gyre.json`

Versioned nested document (not a flat `"tokenizer":"bpe"` string):

```json
{
  "gyre": "tokenizer",
  "version": 1,
  "tokenizer": {
    "pretoken": "identity",
    "model": "bpe",
    "merges": [[97, 97]],
    "vocab": ["a", "b"]
  }
}
```

| Field | Meaning |
| --- | --- |
| `gyre` | Must be `"tokenizer"` |
| `version` | Document version (`1`) |
| `tokenizer.pretoken` | `identity` \| `bytelevel` \| `metaspace`. Omitted → `identity` |
| `tokenizer.model` | `bpe` \| `unigram` |
| `tokenizer.merges` | BPE only: `[left_id, right_id]` in rank order |
| `tokenizer.vocab` | Piece strings (id = index). Restricted alphabets (chars) always write this. Identity-256 BPE may omit it and rebuild 0..255 + merges |
| `tokenizer.scores` | Unigram log scores, parallel to `vocab` |
| `tokenizer.byte_fallback` | Unigram: if true, 1-byte fallback when Viterbi fails |

GYRE1 checkpoint trailers embed the same keys next to arch fields:

```json
{"arch":"char-lm", "...":"...", "gyre":"tokenizer", "version":1, "tokenizer":{...}}
```

**Legacy trailers** still load: `"tokenizer":"bpe","merges":[[a,b],...]`, `"vocab":"bytes"`, or `"vocab":["a",...]`.

`Tokenizer::load(path)`:

1. Directory → Hugging Face (`tokenizer.json` or `vocab.json` + `merges.txt`)
2. `*.model` → SentencePiece proto
3. JSON with `"regular_tokens"` + `"word_split"` → Grok/SGLang `tokenizer.tok.json` (xAI community license; `data/grok2/tokenizer.tok.json`)
4. Else JSON (`*.gyre.json` or HF `tokenizer.json` without `"gyre"`)

Grok-2 goldens: `"hello world"` → `21517 1749`; chat prefix with `<|separator|>` matches the published Hugging Face ids. `gyre-cli tok encode --tok data/grok2/tokenizer.tok.json --text "hello world"`.

## Hugging Face

**Export** (`save_huggingface` / `gyre-cli tok export`):

- BPE: `vocab.json`, `merges.txt` (`#version: 0.2`), `tokenizer.json` (`model.type: BPE`)
- Unigram: `tokenizer.json` with `model.type: Unigram` and `vocab: [[piece, score], ...]`. No fake empty `merges.txt`

**Import** (`load_huggingface` / `tok import --hf-dir`):

- `tokenizer.json` BPE / Unigram
- GPT-2 layout `vocab.json` + `merges.txt` → `bytelevel` + `bpe`

GPT-2 regex without ICU is ASCII/Latin-1 best-effort. English fixtures match; full Unicode `\p{L}` does not.

## SentencePiece

`.model` is a **file format**, not a third algorithm. Minimal protobuf reader (no libprotobuf):

| SP `model_type` | Gyre |
| --- | --- |
| UNIGRAM (1), CHAR (4) | `metaspace` + `unigram` |
| BPE (2) | Error until merge reconstruction is reliable |
| WORD (3) | Error |

NFKC is not applied (no ICU). `add_dummy_prefix` is on for metaspace. Native `train_unigram` is **not** bit-identical `spm_train`.

## CLI

```
gyre-cli tok train --data data/shakespeare.txt --tokenizer bpe|chars|bytes|unigram \
                   [--vocab-size 2000] [--holdout 0.1] --out data/tok.gyre.json

gyre-cli tok export --tok data/tok.gyre.json --hf-dir data/hf-tok

gyre-cli tok import --hf-dir DIR|--sp FILE.model --out data/tok.gyre.json

gyre-cli lm train --data data/shakespeare.txt --tok data/tok.gyre.json \
                  --ckpt data/charlm.gyre
```

- `--tok` **skips** tokenizer training. Vocab size comes from the file.
- Without `--tok`, train creates a tokenizer then writes **`ckpt` with extension replaced by `.gyre.json`** (e.g. `data/charlm.gyre` → `data/charlm.gyre.json`).
- `--hf-dir` / `--sp` on `lm train` load a foreign tokenizer the same way.
- There is no `--tokenizer gpt2` trainer; import HF then `--tok`.

TUI: tokenizer kind + optional `.gyre.json` path.

Library:

```cpp
auto tok = gyre::Tokenizer::load("data/tok.gyre.json");
auto ids = (*tok)->encode(train_text);  // whole split once
auto data = gyre::CharDataset::from_ids(*ids, device);
```

## Eval

Report **nats/char or BPC**, never nats/token across different tokenizers. Holdout is a raw-byte fraction of `--data`; BPE/unigram train and LM steps see only the prefix.

## Limits

- No WordPiece in this pass (`VocabModel` is the hook).
- No GPT-2 / T5 **weight** import.
- Native unigram train: frequent substrings + log-count scores, then Viterbi.
- Unigram encode is O(n × max_piece_len) per span.
