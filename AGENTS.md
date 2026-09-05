# Gyre — agent memory

Gyre is a greenfield C++23 library (static `gyre` + `gyre-cli` + optional `gyre-tui`). Product name is **Gyre**, not Shakespeare. Default `--data data/shakespeare.txt` is a corpus path only. Models are reusable CharLM checkpoints meant to plug into other programs.

Human-facing docs: `README.md`, Helix embed `docs/helix.md`, index `docs/README.md`.

## Checkpoints and train

- User-facing extension is `.gyre` (binary `GYRE1` v2: JSON document first, then aligned payloads). Twin `*.gyre.json` is the same document (inline `data` only for small models). Spec: `docs/gyre-file.md`. Magic inside the binary is `GYRE1`. v1 trailer files still load.
- Default tokenizer is **BPE** (`--tokenizer bpe|chars|bytes|unigram`, `--vocab-size`, default 2000). Chars/bytes are BPE with **no merges**. Full write-up: `docs/tokenizer.md`.
- Reuse: `--tok FILE.gyre.json` (document `{"gyre":"tokenizer","version":1,"tokenizer":{pretoken,model,merges|vocab|scores}}`). `gyre-cli tok train|export|import`. Hugging Face dir and SentencePiece `.model` (unigram/char only). Train without `--tok` also writes `ckpt` with extension `.gyre.json`. Frozen copy in the GYRE1 trailer. Legacy `"tokenizer":"bpe","merges":…` still loads.
- Default recency is **ALiBi** (`--recency alibi|none`): per-head slope `2^{-8(h+1)/H}` on causal scores, **token distance only**. Checkpoints without `"recency":"alibi"` generate without ALiBi (old files stay compatible).
- **Train holdout** (default `0.1`): last raw-byte fraction of `--data` is **not** used for BPE train/encode or LM steps. Sidecar JSON stores `"holdout"`. `lm eval` uses that holdout (or `--split`) as the val slice. `--holdout 0` trains on the full file (in-sample eval). `--split` on train aliases holdout.
- Fair comparison vs other small LMs: **nats/char or BPC**, never nats/token across tokenizers. nanoGPT shakespeare-char (~10.7M, 6×384, T=256, batch 64, 5k iters) reports val **~1.47 nats/char (~2.12 BPC)** on last 10% of Karpathy tinyshakespeare (~1,115,394 chars; val 111,540). Prior Gyre evals on a full-file train were not held-out.

## Presets

| preset   | typical size | T   | notes |
|----------|--------------|-----|--------|
| tiny     | ~0.2M        | 64  | debug |
| medium   | default      | 128 | d=128 L=4 H=4 |
| tinygpt  | ~3–4M        | 256 | d=192 L=6 H=6; batch 2 if left at 4 |
| nanogpt  | ~10.7M @ V≈65 | 256 | d=384 L=6 H=6 d_ff=1536; CPU batch 2 |

CLI can override `--d-model --n-layer --n-head --d-ff --block` after the preset.

## nanoGPT-like recipe on Gyre (CPU)

Closest apples-to-apples **char** run (match their tokenizer and 90/10 split):

```
gyre-cli lm train --data data/shakespeare.txt --preset nanogpt --tokenizer chars \
  --holdout 0.1 --steps 5000 --batch 2 --lr-start 1e-3 --lr 3e-4 \
  --recency none --ckpt data/nanogpt-char.gyre
gyre-cli lm eval --ckpt data/nanogpt-char.gyre --data data/shakespeare.txt --split 0.1
```

`--recency none` matches vanilla nanoGPT attention; ALiBi is an extra Gyre knob (A/B it). nanoGPT batch 64 will not fit comfortably on CPU here — keep batch 2–4; more steps if tokens/step are lower. Use the **Release** OpenMP binary (`cmake-build-release` / `cmake-build-gyre`), not Debug.

BPE variant (not a tokenizer-matched nanoGPT score; still report nats/char):

```
gyre-cli lm train --data data/shakespeare.txt --preset nanogpt --tokenizer bpe \
  --vocab-size 4000 --holdout 0.1 --steps 5000 --batch 2 --ckpt data/nanogpt-bpe.gyre
```

Reuse that BPE on a later run (no merge retrain):

```
gyre-cli tok train --data data/shakespeare.txt --tokenizer bpe --vocab-size 4000 \
  --holdout 0.1 --out data/nanogpt-bpe.gyre.json
gyre-cli lm train --data data/shakespeare.txt --preset nanogpt --tok data/nanogpt-bpe.gyre.json \
  --holdout 0.1 --steps 5000 --batch 2 --ckpt data/nanogpt-bpe.gyre
```

## Speed and quality

- Debug `cmake-build-debug/gyre-cli` is the first training-speed problem. Release + OpenMP GEMM (`omp_in_parallel()` guards nested matmul). Vulkan/OpenCL later for 10M+/T=512+; not CUDA.
- Sampling: temperature 0.65–0.75 + a cue prompt; greedy often loops.
- LR: `lr_start` → `lr` over first 20% of steps unless `--lr-decay-steps` is set.
- Compare only with matching holdout, same file, same tokenizer family.

## Stack / don’t do unless asked

C++23, CPU Device singleton first, then Vulkan/OpenCL. Helix/StarCraft adapters later. Do not start Elman/leaky-state, island GA, or Vulkan unless requested. Recency-biased attention was chosen over RNN fade for v1.5.

Build: `vcvars64` then `cmake --build` the Ninja dir (`cmake-build-gyre` or `cmake-build-release`).
