# CharLM prune evals (local checkpoints)

Not CI. Numbers are `lm eval --split 0.1` on `data/shakespeare.txt` (chars, last 10% = 111,540 val chars). Checkpoints live under `data/wire/` and are **not** in git. How prune works: [connectome.md](connectome.md).

## Rating scale

Linear leftover-entropy map on **val nats/char**:

| Anchor | nats/char | /10 |
| --- | ---: | ---: |
| Uniform 65-char model (`ln 65`) | 4.174 | **0** |
| nanoGPT shakespeare-char (Karpathy, ~10.7M, batch 64, 5k iters) | ~1.47 | **10** |

```
overall/10 = 10 * (4.174 − nats) / (4.174 − 1.47)
vs parent/10 = 10 * (4.174 − nats) / (4.174 − parent_nats)
```

**Vs parent** is 10 at the dense parent of that run; **> 10** means better val than that parent. Tiny nets sit in the mid-6s even when healthy — they are not a 10.7M model.

Fair comparison is **nats/char**, not nats/token across tokenizers. Generate samples in the score dumps are qualitative only (`temp 0.7`, prompt `To be or `).

## Runs

All use `--tokenizer chars --holdout 0.1 --recency none --prune 0.2 --prune-gens 6`. Each generation trains the same `--steps` as the parent (extra compute vs a matched-budget prune). Width snaps `d_model` to a multiple of `n_head`.

| Run | Preset | Device | Batch | Steps/stage | Extra | Score dump |
| --- | --- | --- | ---: | ---: | --- | --- |
| tiny | `tiny` | CPU | 4 | 2000 | — | `data/wire/scores.txt` |
| medium | `medium` | CPU | 4 | 5000 | — | `data/wire/scores_medium.txt` |
| nanogpt batch 8 | `nanogpt` | Vulkan | 8 | 5000 | constant `lr=3e-4`, unclipped | `data/wire/scores_nanogpt.txt` |
| nanogpt batch 64 | `nanogpt` | Vulkan | 64 | 5000 then mixed | no dropout/decay; g2–g6 rest 2k + clip 50 | `data/wire/scores_nanogpt64.txt` |
| nanogpt + dropout/decay | `nanogpt` | Vulkan | 64 | 5000 | `--dropout 0.2 --decay 0.1` | parent only so far |

`--lr-start 1e-3` on nanogpt width NaN’d. `--grad-clip 20` stalled. Stable nanogpt recipe: **unclipped `3e-4`**. Batch 64 without dropout overfits / can blow up (g4 below).

### tiny (`parent.gyre` → `child-gK.gyre`)

`d=64 L=2 H=4 T=64`. ~0.11M parent.

| Gen | File | d / d_ff | Params | % parent | File bytes | nats/char | BPC | overall/10 | vs parent/10 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `parent.gyre` | 64 / 256 | 112,577 | 100 | 457 KiB | 2.333 | 3.37 | 6.81 | 10.00 |
| 1 | `child-g1.gyre` | 48 / 204 | 68,345 | 61 | 273 KiB | 2.234 | 3.22 | 7.17 | 10.54 |
| 2 | `child-g2.gyre` | 36 / 163 | 41,935 | 37 | 171 KiB | **2.202** | 3.18 | **7.29** | **10.71** |
| 3 | `child-g3.gyre` | 28 / 130 | 27,149 | 24 | 113 KiB | 2.208 | 3.19 | 7.27 | 10.68 |
| 4 | `child-g4.gyre` | 20 / 103 | 15,991 | 14 | 70 KiB | 2.257 | 3.26 | 7.09 | 10.42 |
| 5 | `child-g5.gyre` | 12 / 82 | 7,885 | 7 | 37 KiB | 2.384 | 3.44 | 6.62 | 9.72 |
| 6 | `child-g6.gyre` | 8 / 65 | 4,499 | 4 | 25 KiB | 2.493 | 3.60 | 6.22 | 9.13 |

Peak **g2** (still better than parent through g4). g5–g6 are too thin for this preset.

### medium (`med-parent.gyre` → `med-child-gK.gyre`)

`d=128 L=4 H=4 T=128`. ~0.83M parent.

| Gen | File | d / d_ff | Params | % parent | File bytes | nats/char | BPC | overall/10 | vs parent/10 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `med-parent.gyre` | 128 / 512 | 826,433 | 100 | 3.2 MiB | 1.856 | 2.68 | 8.57 | 10.00 |
| 1 | `med-child-g1.gyre` | 100 / 409 | 518,501 | 63 | 2.0 MiB | 1.732 | 2.50 | 9.03 | 10.53 |
| 2 | `med-child-g2.gyre` | 76 / 327 | 315,101 | 38 | 1.2 MiB | **1.717** | 2.48 | **9.09** | **10.60** |
| 3 | `med-child-g3.gyre` | 60 / 261 | 201,749 | 24 | 800 KiB | 1.741 | 2.51 | 9.00 | 10.50 |
| 4 | `med-child-g4.gyre` | 44 / 208 | 118,113 | 14 | 473 KiB | 1.805 | 2.60 | 8.76 | 10.22 |
| 5 | `med-child-g5.gyre` | 32 / 166 | 69,081 | 8 | 281 KiB | 1.889 | 2.73 | 8.45 | 9.85 |
| 6 | `med-child-g6.gyre` | 24 / 132 | 42,257 | 5 | 177 KiB | 1.986 | 2.86 | 8.09 | 9.44 |

Same shape as tiny: mid-gens beat the dense parent, then width runs out.

### nanogpt batch 8 (`nano-parent.gyre` → `nano-child-gK.gyre`)

`d=384 L=6 H=6 d_ff=1536 T=256`. 10,795,841 params. Vulkan, batch 8, 5k steps/stage, `lr=3e-4`, no clip, no dropout. Parent file ~41 MiB (weights; smaller than batch-64 parents that still hold Adam).

| Gen | File | d / d_ff | Params | % parent | File bytes | nats/char | BPC | overall/10 | vs parent/10 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `nano-parent.gyre` | 384 / 1536 | 10,795,841 | 100 | 41.2 MiB | 1.535 | 2.21 | 9.76 | 10.00 |
| 1 | `nano-child-g1.gyre` | 306 / 1228 | 6,899,165 | 64 | 26.3 MiB | 1.601 | 2.31 | 9.52 | 9.75 |
| 2 | `nano-child-g2.gyre` | 240 / 982 | 4,322,597 | 40 | 16.5 MiB | 1.642 | 2.37 | 9.36 | 9.59 |
| 3 | `nano-child-g3.gyre` | 186 / 785 | 2,669,411 | 25 | 10.2 MiB | 1.602 | 2.31 | 9.51 | 9.75 |
| 4 | `nano-child-g4.gyre` | 144 / 627 | 1,648,595 | 15 | 6.3 MiB | 1.560 | 2.25 | 9.67 | 9.91 |
| 5 | `nano-child-g5.gyre` | 114 / 501 | 1,050,731 | 10 | 4.0 MiB | **1.513** | 2.18 | **9.84** | **10.08** |
| 6 | `nano-child-g6.gyre` | 90 / 400 | 668,645 | 6 | 2.6 MiB | 1.526 | 2.20 | 9.79 | 10.04 |

Parent is already close to Karpathy ~1.47 (batch 8 vs their 64 → fewer tokens/step). **g5 is the peak**: slightly **better** val than the 10.7M parent at **10% of weights**. g6 (6%) still matches the parent. g1–g3 dip then recover.

### nanogpt batch 64 (`nano64-parent.gyre` → `nano64-child-gK.gyre`)

Same architecture. Vulkan batch 64, no dropout/decay. Parent + g1 trained 5k unclipped; g2–g6 continued from g1 at 2k steps with `--grad-clip 50` (`nano64-rest-gK.gyre` copies). Parent `.gyre` ~124 MiB includes Adam.

| Gen | File | d / d_ff | Params | % parent | nats/char | BPC | overall/10 | vs parent/10 | Notes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `nano64-parent.gyre` | 384 / 1536 | 10.8M | 100 | 2.834 | 4.09 | 4.96 | 10.00 | Overfit vs batch 8 |
| 1 | `nano64-child-g1.gyre` | 306 / 1228 | 6.9M | 64 | 3.139 | 4.53 | 3.83 | 7.72 | Worse |
| 2 | `nano64-child-g2.gyre` | 240 / 982 | 4.3M | 40 | 2.023 | 2.92 | 7.96 | 16.05 | Rest resume |
| 3 | `nano64-child-g3.gyre` | 186 / 785 | 2.7M | 25 | 1.654 | 2.39 | 9.32 | 18.80 | Recovering |
| 4 | `nano64-child-g4.gyre` | 144 / 627 | 1.6M | 15 | **4.952** | 7.14 | — | — | **Collapsed** (worse than uniform; sample is `a a a`) |
| 5 | `nano64-child-g5.gyre` | 114 / 501 | 1.05M | 10 | 1.572 | 2.27 | 9.62 | 19.42 | Recovered after collapse |
| 6 | `nano64-child-g6.gyre` | 90 / 400 | 0.67M | 6 | 1.572 | 2.27 | 9.62 | 19.42 | Matches g5 |

Do **not** treat vs-parent > 16 as a win: the parent is a weak, overfit baseline. Prefer the batch-8 table for nanogpt prune quality. Batch 64 needs `--dropout 0.2 --decay 0.1` (see next).

### nanogpt dropout + decay (`nano-dd-parent.gyre`)

Intended: same as Karpathy-ish Gyre recipe (`--dropout 0.2 --decay 0.1 --lr-start 1e-3 --lr 3e-4 --batch 64 --steps 5000`). Parent checkpoint exists; prune children (`nano-dd-child-gK`) were **not** written when this doc was filled. Re-eval after that run finishes.

## How to re-eval

```
gyre-cli lm eval --ckpt data/wire/nano-child-g5.gyre --data data/shakespeare.txt --split 0.1 --device vulkan
```

CPU for tiny/medium. Scripts that produced the dumps: `data/wire/run_prune6*.ps1`.
