# CharLM wire maps (optional prune → smaller dense net)

Opt-in training path. Default dense CharLM is unchanged.

After a trained checkpoint, `--prune 0.2` drops the weakest **20% of residual channels** (`d_model`, aligned to `n_head`) and **20% of each block’s FFN width** (`d_ff`), copies survivors into a **new smaller dense** CharLM, and collapses remaining 2D weights to `sign(w) * mean(|w|)`. Dead units are gone: no zero weights, no masked GEMM noops. The next train run is ordinary dense GEMM at the new size.

That is the “shift remaining wires until the net is dense again” step. Unstructured holes are not kept.

`--prune-gens 8` repeats prune→train. Width scales by about `0.8` per generation (`d_model` snapped down to a multiple of `n_head`). Parameter count falls faster than 20% on square matrices (`d×d`). After **six** 20% width cuts you are near **25%** of the original width (`0.8^6 ≈ 0.26`); **eight** cuts land near `0.8^8 ≈ 17%` width. File size follows the live dense tensors.

Not lossless `wpack` ([grok-compress.md](grok-compress.md)). Not a copied fly connectome ([interesting.md](interesting.md)).

## CLI

| Flag | Meaning |
|------|---------|
| `--prune F` | `0 < F < 1`. Each generation drops that fraction of `d_model` / `d_ff`. `0` (default) is off. |
| `--prune-gens N` | Repeat prune→train N times (default 1). |
| `--wire PATH` | Init from this `.gyre` (parent or previous gen). Reuses its tokenizer and holdout. |
| `--resume` | Load `--ckpt` (or `--wire`) and continue; `--steps` are **additional**. Adam is restored when present. |
| `--shuffle-wire` | Keep random channels instead of the strongest (control). |
| `--ckpt-every` | Unchanged: `step-<n>.gyre` in the checkpoint directory (with Adam). |

After generation `k`, writes `STEM-gK.gyre` next to `--ckpt` and overwrites `--ckpt`.

`--steps 0 --wire PARENT --prune 0.2` only carves a smaller dense net (no SGD).

`lm eval` prints `params`, `d`, `d_ff`, `gen`, `file_bytes`, and val **nats/char**.

## Loop

1. If not `--wire`/`--resume`: random init, train `--steps`, save (the parent).
2. For each generation: compact → fresh Adam → train `--steps` → save `--ckpt` and `*-gK.gyre`.

Architecture in the GYRE1 `config` is updated (`d_model`, `d_ff`, `connectome.generation`). Generate/eval load that config; do not pass a larger `--preset` and expect it to override the file.

## Shakespeare comparison (not CI)

Same file, **chars**, `--holdout 0.1`, Release binary, val **nats/char**. Start with `tiny`; scale up if the loop is stable.

Parent:

```
gyre-cli lm train --data data/shakespeare.txt --preset tiny --tokenizer chars \
  --holdout 0.1 --steps 2000 --batch 4 --recency none \
  --ckpt data/wire/parent.gyre
gyre-cli lm eval --ckpt data/wire/parent.gyre --data data/shakespeare.txt --split 0.1
```

Eight prune generations, matched total SGD after the parent exists (`8 × 250`):

```
gyre-cli lm train --data data/shakespeare.txt --preset tiny --tokenizer chars \
  --holdout 0.1 --wire data/wire/parent.gyre --prune 0.2 --prune-gens 8 --steps 250 \
  --batch 4 --recency none --ckpt data/wire/child.gyre
```

Eval `data/wire/parent.gyre` and `data/wire/child-g1.gyre` … `child-g8.gyre`. Record gen, `d_model`, `d_ff`, params, file bytes, val nats/char.

Control: add `--shuffle-wire` on a short run; shuffled channels should not match the evolved subset.

If 250 steps per gen is too little after a prune, use the same `--steps` as the parent each gen and say so (that is extra compute).

Recorded Shakespeare runs (tiny / medium / nanogpt batch 8 / batch 64), nats/char, and /10 ratings: [wire-evals.md](wire-evals.md). Checkpoints stay in `data/wire/` (not git).
