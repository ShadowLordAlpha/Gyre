# Grok weight lossless compression (handoff)

**Status:** `wpack` now reports **actual** packed sizes. Dense bitplane vs u32 exception list is per-plane. `exp_alpha` is a real codec (exponent alphabet + leftover sign/mantissa bits). Real Grok-2 prefixes **do** shrink (~0.60–0.81 packed/raw), but that is **not** the Shannon `estimate` column and **not** 0.5× yet. Quantization (GGUF/Q4) is **out of scope** here.

**Product target (later):** Hub dump ~500 GB bf16 → **~250 GB** packed, **bit-identical** decode, **random access** per tensor/chunk (layer paging). Compare to “two large games” on disk and faster load if `read(packed)+regenerate < read(raw)`.

**Do not** gzip the whole 500 GB tree (no seek). **Do not** treat probe `stack/probed≈0.65` as a 35% smaller file.

---

## Context a new session needs

Gyre already runs a **tiny/mini Grok-shaped** net, tiktoken, safetensors inspect, mmap/bf16. Full Grok-2 generate is **not** blocked on compression; compression is an optional disk/load path.

Weights: official `xai-org/grok-2`, **Grok 2 Community License**. Code stays MIT. Do not vendor the dump.

There is **no official small Grok**. Grok-1 is **314B params / ~318 GB int8 pickle** (`ckpt-0/tensorNNNNN_000`), not safetensors. Community Grok-1 **bf16** conversions are **~633 GB** (worse). Prefer probing **Grok-2 shards we already have**.

### Local Grok-2 files (`data/grok2/`)

| File | Contents | Size (approx) |
|------|----------|----------------|
| `pytorch_model-00000-TP-common.safetensors` | `model.embed_tokens.weight` BF16 `[131072,8192]` | ~2 GB |
| `pytorch_model-00001-TP-common.safetensors` | `lm_head.weight` same | ~2 GB |
| `pytorch_model-00002-TP-common.safetensors` | `model.norm.weight` BF16 `[8192]` | ~16 KB |
| `pytorch_model-00003-TP-common.safetensors` | 64× `model.layers.*.mlp.gate_proj.weight` `[32768,8192]` | ~34 GB |
| `tokenizer.tok.json`, `config.json` | not weights | |

Still missing for a full family picture: **attention** (`q/k/v/o_proj`), **experts**, other MLP (`up`/`down`), per-layer norms, router. Probe **does not need** the rest of the 42 files to iterate codecs; add one expert or attn shard later if a family behaves differently.

PyTorch layout is `[out, in]`; Gyre GEMM is `[in, out]`. Compression must round-trip **file bytes**, not Gyre’s training layout.

---

## Design rules (from the owner)

1. **Lossless only.** Decode `memcmp`s to the original safetensor payload (bf16 bits). `to_f32()` is compute, not the packed format.
2. **Dtype is a view.** Bytes may be treated as `u8`, `u16`, bitplanes, or a bitstream. Hub `BF16` is 16-bit cells, not a requirement to multiply floats.
3. **Pattern + exceptions.** A rule predicts bits (constant plane, repeating mask, “this bit is always 0 after a reversible map”). Store the rule, then a **sparse exception table** (or a dense bitplane if that is smaller). Overflow / non-conforming values are exceptions, not a reason to abandon the rule.
4. **Stacking is on the residual.**  
   - Apply reversible rewrite `f` (may not shrink by itself).  
   - Exception-code / drop bits that became boring.  
   - Next codec sees **only leftover bits**.  
   - Keep a step **only if packed size falls** (include the rule bytes + exception table).  
   - Cap stack depth (e.g. 4). If a step grows the blob → `identity` for that chunk.
5. **Transforms can reduce randomness without shrinking.** Example used in discussion: integer `×2` forces LSB=0 (the example said “always 1”; for integers it is **0**). That bit can then be omitted + overflow list. **Do not** implement `float ×2` as the first catalog entry; on bf16, `×2` is mostly `exponent++`, mantissa unchanged.
6. **Per shard / tensor / ~1–4 MiB chunk**, not one global codec. Attn vs embed vs expert will differ. Prefer **one `.gyre.wpack` per Hub safetensor** so packing can be incremental (no extra 500 GB free space).
7. **Never keep a larger encoding than raw.**
8. **Shannon / gzip-ratio columns are estimates.** Success is **packed_bytes / raw_bytes** on `wpack_encode` output.

Pedagogical ×2 is **not** a required codec. The required idea is: **reversible map → constant or low-entropy bits → drop those bits + exceptions → repeat.**

---

## What is already in the tree

| Piece | Path | Behavior |
|-------|------|----------|
| Probe | `include/gyre/io/compress_probe.hpp`, `src/gyre/io/compress_probe.cpp` | Samples prefixes (`--chunk-bytes`, `--max-tensors`). Reports entropy, RLE, sampled LZSS, XOR entropy, split sign/exp/mant Shannon, **estimated** plane_exc / freq16 sizes. `stack_bytes` is **`min` of those estimates**, **not** a real multi-step encode. |
| Pack | `include/gyre/io/wpack.hpp`, `src/gyre/io/wpack.cpp` | Magic `GYWP1`. Codecs: `identity=0`, `plane_exc=1`, `exp_alpha=2`, `lfsr_pred=3`, `const_lane=4` (stride-1/2 constant byte on a **window** `[offset, offset+len)`, exception `(rel_index, value)`, leftover optionally inner-packed). Keep only if `enc.size()+16 < current`. |
| CLI | `gyre-cli grok compress-probe DIR …` | JSON + one line per tensor: **`packed/probed`** is `wpack_encode`; `estimate/probed` is Shannon/RLE/LZSS (legacy `stack_bytes`). |
| CLI | `gyre-cli grok pack DIR --out FILE [--file SUBSTR] [--max-bytes N]` | Skips tensors larger than `--max-bytes` (default 2 MiB). |
| Tests | `tests/compress_probe_test.cpp`, `tests/wpack_test.cpp` | Planted `0xAAAA` → plane_exc; two-exp noisy mant → exp_alpha; PRNG → identity; 50/50 bits do not use the index list; real `model.norm` memcmp and packed smaller than raw. |

### Actual `wpack_encode` sizes (do not mix with Shannon)

64 KiB prefixes (`--chunk-bytes 65536`):

| Tensor | packed/probed | codec | estimate/probed (Shannon-ish) |
|--------|---------------|--------|-------------------------------|
| embed | **0.712** | const_lane | 0.646 |
| lm_head | **0.784** | plane_exc | 0.690 |
| `model.norm` (full 16 KiB) | **0.659** (10803/16384) | const_lane | 0.478 |
| `layers.0.mlp.gate_proj` | **0.756** | plane_exc | 0.658 |

1 MiB prefixes:

| Tensor | packed/probed | codec | notes |
|--------|---------------|--------|--------|
| embed | **0.607** | plane_exc | first 1 MiB is unusually compressible (`entropy_bpb≈2.0`, `top_u16_frac≈0.48`); do not assume the rest of the 2 GB tensor matches |
| lm_head | **0.804** | plane_exc | |
| gate_proj.0 | **0.754** | plane_exc | ~22 unique exponents |

Neighbor XOR still **increases** entropy on lm_head/mlp. `freq16` exception lists lose on real data.

**Why old plane_exc lost:** u32 index per miss on a ~50/50 plane is `~0.5N×4` vs `N/8` dense. Dense planes now win on Grok-2 prefixes (some bits are biased). Index list is only used when occupancy is sparse (planted `0xAAAA`).

`estimate_bytes` / legacy `stack_bytes` still mix Shannon of sign/exp/mant. That is **not** a packed file. Success is **`packed_bytes / probed_bytes`**.

### LFSR / SeedLM-style predictor (`lfsr_pred`)

Not SeedLM’s lossy \(\hat{w}=U(s)t\). Lossless analog: XOR each 64-word block with a 16-bit Galois stream (`poly 0xB400`) if that **lowers `plane_score`** (sum of `min(ones,zeros)` per bitplane — a cheap “less random” proxy). Then pack the residual with `plane_exc` or `exp_alpha`. Seed table + inner blob is kept **only if smaller than those codecs on the original**.

- Planted LFSR stream + two bit flips → `lfsr_pred`, memcmp, smaller than inner-alone.
- Grok-2 64 KiB prefixes (embed / lm_head / norm / gate_proj): **did not win**. Packed codecs stayed `plane_exc` / `exp_alpha` at the same sizes as before. The map made residuals worse or not enough to pay the seed table.
- Do not treat SeedLM’s 3–4-bit accuracy numbers as packed/raw.

A later catalog entry can try a denser seed search or integer subtract residual; same keep-if-smaller gate.

### `const_lane` (the glance-visible `ab 40 b2 40 …` rule)

Stride-2 high byte of LE bf16 is often `0x40` on **norm** (~89.5% of `00002`). The map is **not** “whole file”: Kadane on `hit=+1, miss=−index_bytes` picks `[begin, begin+len)` of **lane indices**. After the pattern dies, the window stops so a tail of misses does not pay an exception per cell.

Store: stride, lane, const, range, `(rel_index, byte)` exceptions, leftover = all bytes not in that lane window (inner `plane_exc`/`exp_alpha` if smaller).

- `00002` full: **16384 → 10803** (was 13339 `exp_alpha`).
- Plant: prefix junk + `0x40` high bytes + failing tail → window covers the middle only, memcmp.

---

## Implementation plan (when this work resumes)

### P0 — Honesty in the probe (done)

- `estimate_bytes` (JSON also still has `stack_bytes` as an alias) is Shannon/RLE/LZSS **estimates**.
- `packed_bytes` / `packed_codec` come from `wpack_encode` on the same prefix.
- CLI prints `packed/probed` and `estimate/probed` separately.
- CI: planted shrink + `memcmp`; real-shard ratios are logs, not fail-unless-0.5.

### P1 — Chunked residual stack (core)

Replace single `plane_exc` with a **list of steps** per chunk:

```
chunk := raw bytes (1–4 MiB)
for depth in 1..4:
  try catalog of reversible maps f on current bytes
  try exception/dense encode of boring bits
  if size(f, rule, exceptions, leftover) < size(current)+epsilon:
    commit step; current := leftover bits
  else skip
if final >= raw: store identity
```

Decode: reverse steps, emit exact original bytes.

**Catalog (byte/bit first, float layout optional):**

| id | Map / rule | Notes |
|----|------------|--------|
| `identity` | none | fallback |
| `xor_delta` | each u16/u8 XOR previous | **failed** as entropy on Grok-2 prefixes; keep in catalog, don’t assume |
| `bitshuffle` | group bitplanes | often helps entropy coders; we have no zstd yet |
| `const_plane` | majority 0/1 per plane | current `plane_exc`; **store dense bitplane when exceptions are dense** |
| `repeat_mask` | period-1/2/8/16 bit mask | “every other bit” family |
| `freq_sym` | most common u8/u16 + exceptions | current `freq16` estimate |
| `exp_alphabet` | interpret u16 as bf16 **only as a hint**: 8-bit exponent as small alphabet, mantissa residual | probe showed few unique exponents — **best lead** |
| `int_shift` | treat as u16, shift left/right if high/low bits empty + overflow list | pedagogical ×2 class; measure, don’t assume |
| `lfsr_pred` | per-block LFSR XOR, then inner codec on residual | **in tree**; planted wins; Grok-2 prefixes did not; SeedLM paper is lossy |
| `const_lane` | stride-1/2 const byte on `[offset,size)` + exceptions | **in tree**; wins on `00002` and embed 64 KiB prefix |

**Exception encoding:** do not only use `u32` indices. Use: run-length of misses, bitplane of misses (`N/8`), or delta-varint of positions. Pick per plane `min(dense_plane, index_list, rle)`.

**Chunking:** tensors larger than chunk size (embed 2 GB, gate ~512 MB/layer) **must** split; packer currently loads whole tensor up to `max_bytes` and **skips** 2 GB files. Implement chunk index in `GYWP1` or bump to `GYWP2` if the header must change.

### P2 — Format

Keep `GYWP1` if possible: per tensor, `codec` might become `stack=2` with a payload = `[nsteps][step_id…][blobs]`. If that breaks old files, `GYWP2`. One pack file per Hub shard.

Loader for generate: `wpack_decode` → bf16 bytes → existing `to_f32` / mmap path. Generate must still work on **unpacked** safetensors.

### P3 — Measure on real prefixes (no 500 GB encode)

```
gyre-cli grok compress-probe data/grok2 --max-tensors 8 --chunk-bytes 1048576 --out data/grok2/probe.json
gyre-cli grok pack data/grok2 --file 00002 --out data/grok2/norm.gyre.wpack
```

Success for a codec: **packed < raw** on embed **or** gate_proj prefixes, `memcmp` decode, and time `read+decode` vs raw read.

Target 0.5× on the **full dump** is a packer goal, not a CI gate. If embed stays ~0.65 Shannon after a real stack, report which family refused and add a map — do not invent gzip-all.

### P4 — Optional Grok-1

Official `xai-org/grok-1` is **~318 GB**, `ckpt-0/tensor*_000`, **pickle int8 + bf16 scale**, Apache 2.0. Probe **cannot** read it today.

If used: download **one** ~3.2 GB file (`tensor00000_000`), add a prefix reader for the **int8 payload** (avoid naive `pickle.load` of 3 GB as the long-term API). Int8 may exception-code better than bf16 mantissa. Do **not** pull 318 GB or the 633 GB bf16 conversion just to iterate codecs.

### P5 — Out of scope until P1–P3

- zstd as a **final** residual entropy coder only if it shrinks leftover bits (vcpkg `zstd`); not gzip-the-archive
- Quant / GGUF
- Packing 00000/00003 in full in CI
- Claiming 250 GB before `wpack_encode` sizes say so

---

## Tests the next implementer should add

- Planted residual stack: constant exponent **then** repeating mantissa mask → two-step pack **smaller than either step alone**, `memcmp`.
- Two synthetic shards, **different** winning stacks.
- PRNG → identity, size ≤ raw + header.
- Real `00002` still bit-identical; log packed/raw.
- Optional: 1 MiB prefix of `00000` / `00003` if files present; **no** fail if ratio > 0.5.
- Dense bitplane vs u32-index: 50/50 random bits must **not** use the index list.

Existing tests must stay green (`Wpack.*`, `CompressProbe.*`).

---

## CLI cheat sheet

```
gyre-cli grok inspect data/grok2
gyre-cli grok compress-probe data/grok2 --max-tensors 4 --chunk-bytes 65536
gyre-cli grok pack data/grok2 --file 00002 --out data/grok2/norm.gyre.wpack --max-bytes 1000000
```

Tiny model (unrelated to Hub dump):

```
gyre-cli grok save --preset mini --out data/grok-mini
gyre-cli grok gen --weights data/grok-mini --prompt "To be" --max-new 32
```

---

## Suggested first commit in the next session

1. Fix probe `stack_bytes` semantics; report **actual** `wpack_encode` size.  
2. Change `plane_exc` to choose **dense bitplane vs exception list** per plane.  
3. Add `exp_alphabet` + mantissa residual as a real stack (not Shannon sum).  
4. Re-run probe on `00000`/`00003` prefixes and record packed/raw in this doc.

Code entry points: `wpack_encode` / `encode_plane_exc` in `src/gyre/io/wpack.cpp`; `probe_bytes` in `src/gyre/io/compress_probe.cpp`.
