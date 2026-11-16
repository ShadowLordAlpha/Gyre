# SmolLM2-360M on Gyre (dense Llama, CPU, KV cache)

Gyre runs Hugging Face **`HuggingFaceTB/SmolLM2-360M`** (Llama architecture) on the CPU through `LlamaLM`
(`include/gyre/nn/llama.hpp`), an inference-only dense Llama decoder with a KV cache. Nothing is retrained;
the importer repackages the weights into a GYRE1 v2 `.gyre` with `arch: "llama"` and the tokenizer frozen
inside.

## Where files live

Hub files go in the git-ignored `data/smollm2-360m/` (like `data/gpt2/`):

```
data/smollm2-360m/config.json  tokenizer.json  model.safetensors (bf16, 724 MB)  special_tokens_map.json
data/smollm2-360m/smollm2-360m.gyre  smollm2-360m.gyre.json        (converted: f32, 1.45 GB + metadata twin)
```

```
gyre-cli lm import-llama --hf-dir data/smollm2-360m --ckpt data/smollm2-360m/smollm2-360m.gyre
gyre-cli lm eval --ckpt data/smollm2-360m/smollm2-360m.gyre --data data/shakespeare.txt --split 0.1
gyre-cli lm generate --ckpt data/smollm2-360m/smollm2-360m.gyre --prompt "The capital of France is" --chars 16 --temp 0
gyre-cli lm generate ... --no-kv-cache          # full recompute per token (timing reference)
gyre-cli lm logits --ckpt ... --prompt "..." --out logits.f32   # raw f32 [T, vocab] for parity checks
gyre-cli tok encode --tok data/smollm2-360m --text "In 2024"    # Hugging Face tokenizer directly
```

`lm eval` and `lm generate` dispatch on the checkpoint `arch`, so the CharLM flags still apply where they make
sense: `--chars` is the number of new **tokens**, `--temp` (default 0.8; 0 = greedy), plus `--window` /
`--stride` for eval. Generation stops at `eos_token_id` (`<|endoftext|>`, id 0).

Library: `import_llama_hf(dir)` (`gyre/nn/llama_import.hpp`) -> `LlamaLM`; `LlamaLM::load_gyre(path)` maps the
weights straight from the file (no copy); `save_gyre(path, tokenizer_json, extra_config)`.

## Model

Config: hidden 960, 32 layers, 15 query heads / 5 KV heads (head_dim 64, GQA), SwiGLU d_ff 2560, RMSNorm eps
1e-5, RoPE theta 100000 (HF `rotate_half`), tied embeddings, vocab 49152, no biases, no rope scaling.
361,821,120 parameters.

```
x = embed[ids]
per layer:  x += Wo · attn(rope(Wq·rms(x)), rope(Wk·rms(x)), Wv·rms(x))     causal GQA
            x += Wdown · (silu(Wgate·rms(x)) * Wup·rms(x))
logits = rms(x) · embed^T
```

| Hugging Face | Gyre (`[in, out]` layout) |
| --- | --- |
| `model.embed_tokens` `[V, d]` | `embed.weight` (as is; the tied head is `embed^T`, built in RAM at load) |
| `layers.i.input_layernorm` / `post_attention_layernorm` | `layers.i.attn_norm` / `mlp_norm` |
| `self_attn.q_proj` / `k_proj` `[out, in]` | `attn.q` / `attn.k`, transposed **and RoPE-permuted** |
| `self_attn.v_proj` / `o_proj`, `mlp.gate/up/down_proj` | `attn.v` / `attn.o`, `mlp.gate/up/down`, transposed |
| `model.norm` | `norm.weight` |
| `lm_head` (untied models only) | `lm_head.weight` `[d, V]` |

**RoPE layout.** `gyre::rope` rotates interleaved pairs `(2i, 2i+1)`; HF Llama rotates `(i, i + hd/2)`. Both use
`theta^(-2i/hd)` for pair `i`, so the importer moves HF column `j < hd/2` to `2j` and `j >= hd/2` to
`2(j - hd/2) + 1` inside every q and k head. `q·k` is unchanged by permuting both, so V/O and the rest are
untouched. The checkpoint config records `"rope": "interleaved"`. Cos/sin tables are computed in double.

Rejected with a message: `rope_scaling`, `attention_bias`, `mlp_bias`, `hidden_act` other than silu,
`pretraining_tp != 1`, `rope_interleaved`, sharded safetensors. bf16/f16 are widened exactly to f32.

## KV cache and kernels

`LlamaCache` holds K and V for the **5 KV heads only**: `[layer][kv_head][capacity][head_dim]`.
`forward(cache, ids, n_logits)` appends any number of tokens (prefill = many, decode = one) and attends over
everything cached; `forward_full(ids)` is the reference full recompute with a fresh cache.
`forward_hidden` + `logits_from_hidden` let the evaluator score a 2048-token window without materialising
`[2048, 49152]` logits.

CPU kernels live in `src/gyre/nn/llama.cpp`: a tiled GEMM (64-row x 256-col tiles, 4x16 micro-kernel) with an
AVX2+FMA path chosen at **run time** (the library is still built for the baseline ISA), OpenMP over tiles,
per-(query, head) causal attention with an AVX2 dot / weighted-sum. Each output element is a single k-ordered
chain whatever the tiling, so cached decode reproduces the full recompute bit for bit on the same machine. The
residual adds (`x += attn·Wo`, `x += mlp·Wdown`) run that chain from zero and add `x` once at the end; starting
the chain at `x` rounded every FMA at ulp(|x|), which cost 2e-2 in the logits on SmolLM2-1.7B's ~25,000-sized
first-token activations (below).

## Tokenizer

SmolLM2's `tokenizer.json` is a byte-level BPE that the GPT-2 path could not load: pre-tokenizer
`Sequence[Digits(individual_digits=true), ByteLevel(use_regex=true)]`, 17 added tokens at ids 0-16
(`<|endoftext|>`, `<|im_start|>`, ...), only 235 single-byte tokens, and merge results numbered by the vocab
rather than `256 + rank`. Now ([tokenizer.md](tokenizer.md)):

* `digits_bytelevel` pretokenizer: every `\p{N}` code point is its own piece (`"2024"` -> `2 0 2 4`, also
  Arabic-Indic digits, `½`, `Ⅷ`), and the GPT-2 regex runs on the RAW text between them;
* explicit byte -> id and per-merge result ids (`merge_ids` in the Gyre JSON); encode stays rank-based;
* added tokens are matched literally before pretokenizing (`special` in the Gyre JSON);
* bytes not in the vocab (e.g. `0x04`, `0xF1`) are dropped, exactly like Hugging Face without `unk_token`.

The GPT-2 layout is detected and built exactly as before (its frozen tokenizer in `data/gpt2/gpt2.gyre` is
byte-identical, and `<|endoftext|>` is still plain text there).

`tests/hf_bpe_test.cpp` checks ids from the Hugging Face `tokenizers` library for digits (`"In 2024, prices
rose 3.5%."`), contractions, leading spaces and newlines, CRLF/tabs, accented/CJK/emoji text, non-ASCII numbers,
`<|endoftext|>`, chat markers inside text and dropped bytes, plus `decode(encode(s)) == s` and the Gyre-JSON round
trip.

## Verification

* `tests/llama_test.cpp`: writes tiny random HF Llama directories (GQA 6/2 heads; 9/3 heads with head_dim 6,
  untied head; widths not a multiple of 16) and compares the imported model with an independent
  double-precision HF forward (`rotate_half` on the original `[out, in]` weights): max |dlogit| < 2e-4. Also: the
  RoPE permutation against `gyre::rope`, KV cache (prefill 7 + chunk 5 + 11 single steps) == full recompute
  (<= 1e-5), cached and uncached greedy decode pick the same tokens, `.gyre` round trip is exact, config
  rejection, and (with the converted checkpoint) greedy "The capital of France is" -> " Paris".
* Independent numpy float64 reference of the real model (reads the bf16 safetensors as `uint16 << 16`, HF
  `rotate_half`, `repeat_kv`), 32-token prompt *"In 1603, Shakespeare's company became the King's Men; ..."*:
  **max |dlogit| = 1.13e-3** (mean 3.8e-5; the largest error is at position 0, the rest <= 2.3e-4),
  **same top-1 at all 32 positions**, mean next-token NLL 1.993748 (ref) vs 1.993743 (Gyre). Token ids equal the
  `tokenizers` library's. After the residual-add fix (above), on the 64-token prompt used for 1.7B: max |dlogit|
  **2.6e-4**, same top-1 everywhere, NLL 1.262457 both; the Shakespeare eval below is unchanged at 3 decimals.

## Results (Shakespeare, last 10% = 111,540 chars)

`lm eval --split 0.1`: the val slice is `<|endoftext|>` + 35,983 tokens (3.10 chars/token); 2048-token windows,
stride 1024, every token scored exactly once (after the first window each prediction sees >= 1024 tokens of
context). 35 windows, **75 s** on a Ryzen 9 7950X (32 threads).

| model | params | context | nats/char | BPC |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-360M, **zero-shot** | 362M | 1024-2047 tokens (~3-6k chars) | **0.845** | **1.219** |
| GPT-2 small, zero-shot ([gpt2.md](gpt2.md)) | 124M | 1024 tokens | 1.284 | 1.85 |
| nanoGPT shakespeare-char (reference) | 10.7M | 256 chars | ~1.47 | 2.12 |

nats/token 2.618. As with GPT-2 this is a **reference point, not a fair comparison**: SmolLM2 was trained on
trillions of tokens of web text where Shakespeare (and tinyshakespeare itself) is common, so a lot of this is
recall. It recites Romeo and Juliet verbatim from a two-line cue (below).

## Generation speed

Release build, 7950X, 32 OpenMP threads, greedy, f32 weights (1.45 GB):

| prompt | KV cache | no KV cache (full recompute) |
| --- | --- | --- |
| 5 tokens, 16 new | prefill 0.034 s, **0.031 s/token** | 0.036 s/token |
| 846 tokens, 32 new | prefill 0.59 s, **0.035 s/token** | **0.62 s/token** |

Decode is memory-bound (all 1.45 GB of f32 weights are read per token, ~45 GB/s); without the cache the cost
grows with the prefix (an 846-token prefix is ~18x slower per token).

Sample (`--temp 0.7`, prompt "ROMEO:\nBut soft, what light", 60 tokens, 0.038 s/token):

```
ROMEO:
But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid,
```

## SmolLM2-1.7B

`HuggingFaceTB/SmolLM2-1.7B` (base model) goes through the same path with no code changes beyond the residual-add
fix: hidden 2048, 24 layers, 32 query / 32 KV heads (no GQA, head_dim 64), d_ff 8192, RoPE theta 130000, tied,
vocab 49152; **1,711,376,384 parameters**.

```
gyre-cli lm import-llama --hf-dir data/smollm2-1.7b --ckpt data/smollm2-1.7b/smollm2-1.7b.gyre
```

* **Import:** 3.4 GB bf16 -> 6.85 GB f32 `.gyre` (+ 3.6 MB `.gyre.json`) in 80 s. Peak RAM 13.5 GB: the importer
  reads one tensor at a time, but `save_gyre` packs every payload in memory before writing, so the peak is ~2x
  the f32 model. Loading maps the file (only the 0.4 GB `embed^T` head is a copy).
* **Parity** (numpy float64 reference as above, 64-token prompt *"The Great Barrier Reef, off the coast of
  Queensland, ..."*): **max |dlogit| = 8.9e-4** (position 0; positions 1-63 <= 1.5e-4, mean 1.8e-5), **same top-1
  at all 64 positions**, mean next-token NLL 1.138238 (ref) vs 1.138238 (Gyre). Position 0 carries the "massive
  activation": the residual reaches ~25,000 in one channel from layer 7 to 21 and is cancelled again in layers
  22-23, so f32 rounding there shows up in the logits. A numpy f32 forward with one sequential k-chain per output
  gives 8.0e-4 at position 0, i.e. this is the f32 floor of the kernel design.
* **Shakespeare** (`lm eval --split 0.1`, same 35 windows as 360M): **0.4938 nats/char, 0.7124 BPC**
  (1.531 nats/token), **599 s**. Same caveat as above: much of this is recall of a famous public text.
* **Speed** (7950X, 32 threads, f32, greedy):

| prompt | KV cache | no KV cache (full recompute) |
| --- | --- | --- |
| 5 tokens, 16 new | prefill 0.22 s, **0.174 s/token** | 0.258 s/token |
| 839 tokens, 32 new | prefill 4.98 s, **0.170 s/token** | **5.85 s/token** |

Decode reads all 6.85 GB of weights per token (~40 GB/s, memory-bound, ~4.7x the 360M time for 4.7x the
parameters). Greedy "The capital of France is" -> " Paris." (`Llama.RealSmolLM2_1_7BKnowsTheCapitalOfFrance`,
skipped unless the converted checkpoint exists; ~50 s, mostly paging in the file). Sample (`--temp 0.5`, 110
tokens, 0.19 s/token):

```
# Python 3: return the n-th Fibonacci number iteratively.
def fibonacci(n):
    if n < 0:
        return None
    elif n == 0 or n == 1:
        return n
    else:
        a, b = 0, 1
        for i in range(2, n+1):
            a, b = b, a+b
        return b
```

## Limits

* CPU, f32 only (no bf16/int8 weights, no Vulkan); single sequence (no batching).
* No training/gradients for `LlamaLM`; it is not a `Module`.
* Llama family without rope scaling (no Llama-3.1 `rope_scaling`, no Qwen biases); single-file safetensors.
* Hugging Face export of the SmolLM2 tokenizer layout is not implemented (errors instead of writing a wrong file).
