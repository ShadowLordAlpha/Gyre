# GPT-2 on Gyre (pretrained baseline)

Gyre can run the Hugging Face **GPT-2 small (124M)** weights as an ordinary CharLM checkpoint. GPT-2 is a
pre-LayerNorm decoder with learned positions, tanh-GELU and biases everywhere, which is exactly CharLM with
`--recency none`. The conversion only repackages tensors; nothing is retrained.

## Where files live

Like `data/grok2/`, Hub files go in `data/gpt2/` (git-ignored):

```
data/gpt2/config.json  model.safetensors  tokenizer.json  vocab.json  merges.txt   (openai-community/gpt2)
data/gpt2/gpt2.gyre  gpt2.gyre.json                                                (converted, 653 MB)
```

```
gyre-cli lm import-gpt2 --hf-dir data/gpt2 --ckpt data/gpt2/gpt2.gyre
gyre-cli lm eval --ckpt data/gpt2/gpt2.gyre --data data/shakespeare.txt --split 0.1
```

Library: `import_gpt2(dir, device)` in `gyre/nn/gpt2_import.hpp` returns a `CharLM`.

## What is converted

| Hugging Face | CharLM |
| --- | --- |
| `wte`, `wpe` | `wte`, `wpe` |
| `h.i.ln_1` / `ln_2` / `ln_f` | `blocks.i.ln1` / `ln2` / `ln_f` |
| `h.i.attn.c_attn` `[d, 3d]` + bias | `blocks.i.attn.q / k / v` (thirds of the output axis) |
| `h.i.attn.c_proj` | `blocks.i.attn.o` |
| `h.i.mlp.c_fc` / `c_proj` | `blocks.i.mlp.fc1 / fc2` |
| tied output head (`wte`^T) | `lm_head.weight`, bias 0 |
| `h.i.attn.bias` (causal mask buffer) | ignored |

GPT-2's `Conv1D` weights are already `[in, out]` (Gyre's `Linear` layout), so nothing else is transposed.
Unsupported variants are rejected with a message (activation other than `gelu_new`, `layer_norm_epsilon`
other than 1e-5, `scale_attn_by_inverse_layer_idx`, `reorder_and_upcast_attn`). f16/bf16 tensors are widened.
The tied head is stored as a separate copy, so the checkpoint is 163M parameters / 653 MB.

`tests/gpt2_import_test.cpp` builds a tiny GPT-2-format directory with random weights and compares the imported
CharLM with an independent double-precision GPT-2 forward (max logit difference < 2e-3). With the real weights
it checks that plain English scores ~3.3 nats/token (random weights would be ~10.8).

## Tokenizer

GPT-2's byte-level BPE needed real support. The previous `bytelevel` pretokenizer mapped bytes into GPT-2's
unicode alphabet and then failed ("unknown byte") on actual `vocab.json` / `tokenizer.json`. Now:

* the pre-tokenizer splits the **raw** text with the GPT-2 pattern, Unicode-aware (`\p{L}`, `\p{N}`, `\s`);
* Hugging Face files are decoded from the GPT-2 alphabet into raw-byte pieces on import, and written back in
  that alphabet on export (`save_huggingface`);
* the importer requires GPT-2's id layout (merge *k* produces id `256 + k`) and refuses anything else;
* BPE encoding now applies the lowest-rank merge present in a span instead of walking every merge, which gives
  identical ids (tested against the old sequential algorithm) and is far faster for 50k merges.

Goldens (`tests/bytelevel_test.cpp`, need `data/gpt2`): `"Hello world"` -> `15496 995`, the pangram -> the
published ids, exact decode round trips on multi-byte text/emoji/newlines, Gyre-JSON and Hugging Face export
round trips.

## Results (Shakespeare, last 10% = 111,540 chars)

`lm eval --split 0.1`, 1024-token windows (35 windows):

| model | params | context | nats/char | BPC | score /10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small, **zero-shot** (not trained on this file) | 124M (+tied copy) | 1024 tokens (~3k chars) | **1.284** | 1.85 | 10.7 |
| RL-stack GRU (24.6M chars of training) | 0.48M | 128 chars | 1.506 | 2.17 | 9.87 |
| CharLM medium (24.6M chars) | 0.83M | 128 chars | 1.539 | 2.22 | 9.75 |
| nanoGPT shakespeare-char (reference) | 10.7M | 256 chars | ~1.47 | 2.12 | 10 |

Read this as a **reference point, not an apples-to-apples comparison**: GPT-2 is ~250x larger, trained on far
more data, sees ~25x more context per prediction, and tinyshakespeare is widely mirrored online, so the plays
may well be in its training data. The score scale in `wire-evals.md` was defined with 10 = nanoGPT, so >10 is
possible.

## Limits

* No KV cache: `lm generate` re-runs the whole prefix for every new token (hundreds of seconds for a few
  hundred characters). Scoring (`lm eval`) is unaffected.
* f32 compute and weights in RAM only; this path is for models up to roughly 1B parameters.
* GPT-2 family only (small/medium/large/xl with the same tensor names). Llama/Qwen-style models (RMSNorm, RoPE,
  GQA, SwiGLU) need a dense architecture that does not exist yet; `GrokLM` is MoE/softcap specific.
