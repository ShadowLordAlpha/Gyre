# Grok-2 on Gyre

Gyre can tokenize Grok-2 (`data/grok2/tokenizer.tok.json`), run a **tiny/mini residual-MoE Grok-shaped net**, inspect Hub safetensors, and pack small tensors. It cannot yet page-generate the 270B checkpoint.

There is **no official small Grok**. Grok-1 is 314B; Grok-2 is ~270B (~500 GB). GGUF quants are lossy and still huge. Development uses `GrokConfig::mini()` / `tiny()`, optionally with the Shakespeare corpus tokenizer (`chars_from_text`), not the 500 GB dump.

## Done

- Tiktoken V1 load + goldens
- RMSNorm, SiLU/SwiGLU, RoPE, softcap, GQA, residual MoE
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
- Vulkan, f16 GEMM, GGUF quant (lossy on purpose)

## CLI

```
gyre-cli grok info --config data/grok2/config.json
gyre-cli grok inspect data/grok2
gyre-cli grok save --preset mini --out data/grok-mini
gyre-cli grok gen --weights data/grok-mini --prompt "To be" --max-new 32
```
