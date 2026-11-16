# Grok-2 on Gyre

Gyre can tokenize Grok-2 (`data/grok2/tokenizer.tok.json`), run a **tiny/mini residual-MoE Grok-shaped net**, inspect Hub safetensors, and pack small tensors. It cannot yet page-generate the 270B checkpoint.

There is **no official small Grok**. Grok-1 is 314B; Grok-2 is ~270B (~500 GB). GGUF quants are lossy and still huge. Development uses `GrokConfig::mini()` / `tiny()`, optionally with the Shakespeare corpus tokenizer (`chars_from_text`), not the 500 GB dump.

## Done

- Tiktoken V1 load + goldens
- RMSNorm, GeGLU (tanh GELU gate), RoPE, softcap, GQA, residual MoE

## MoE block (matches the reference)

`GrokLM` runs each layer's FFN as `residual_moe(..., kGrokMoe)` (`include/gyre/nn/moe.hpp`), checked against xai-org/grok-1 `model.py` (MoELayer/DecoderLayer) and SGLang `models/grok.py` (Grok-2):

- Gate activation is GELU (tanh form): `gelu(x@w1) * (x@w3) @ w2` (Grok-1 `jax.nn.gelu`, SGLang `GeluAndMul(approximate="tanh")` / experts `activation="gelu"`).
- Routing: `30*tanh(logits/30)` → softmax over **all** experts → top-k probabilities used as-is (**not** renormalized; SGLang `renormalize=False`, Grok-1 `top_k(softmax)`).
- `residual_moe: true` (Grok-2): `(dense(x) + moe(x)) / sqrt(2)` (SGLang `moe_with_rmoe`).
- `residual_moe: false` (Grok-1 layout, SGLang `ffn = block_sparse_moe`): experts only, no dense MLP, no `1/sqrt(2)`.

**Behaviour change (Nov 2026):** earlier builds used a SiLU gate, renormalized the top-k weights, and did not scale the residual sum. Weight layout and names are unchanged, so saved tiny/mini dirs (`config.json` + `model.safetensors`) still load, but their logits/generations differ from before. The generic default `MoeSpec{}` (SiLU, renormalized, unscaled) is still available for non-Grok MoE use.
- `GrokLM` tiny/mini; `full()` is metadata (`create` refused)
- bf16, mmap, safetensors read/write, `grok inspect`
- Compress **probe** + lossless `wpack` (identity / plane_exc / exp_alpha / lfsr_pred / const_lane); **full plan:** `docs/grok-compress.md`
- Save/load tiny weights (`config.json` + `model.safetensors`)
- LoRA on q/k/v/o (apply, save/load); base `W` not written
- `grok gen` on a saved tiny dir

## Not done (revisit)

- Residual stack + chunked 2 GB tensors; packed prefixes are ~0.75–0.81 (embed 1 MiB prefix ~0.61) — see `docs/grok-compress.md`. Target ~250 GB is a packer goal, not a CI gate.
- Bind TP-8 Hub shards + expert paging for 270B generate
- GrokLM full backward / 270B train
- Grok-shaped nets on Vulkan (CharLM GPU path does not cover RoPE/MoE/GQA)
- f16 GEMM, GGUF quant (lossy on purpose)

## CLI

```
gyre-cli grok info --config data/grok2/config.json
gyre-cli grok inspect data/grok2
gyre-cli grok save --preset mini --out data/grok-mini
gyre-cli grok gen --weights data/grok-mini --prompt "To be" --max-new 32
```
