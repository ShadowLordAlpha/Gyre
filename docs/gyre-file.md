# Gyre model file

One **document** describes a model. Two encodings:

| Path | Use |
| --- | --- |
| `*.gyre` | Binary (magic `GYRE1`). Weights as raw little-endian payloads. Default for training and Helix load. |
| `*.gyre.json` | The same document as JSON. For **small** models you can inline weight arrays so a text editor can read the file. Tokenizer-only files (`"gyre":"tokenizer"`) stay valid. |

`gyre-cli lm train` writes both: `.gyre` always, `.gyre.json` with weight `data` when the payload is ≤ 8 MiB, otherwise metadata (arch, tokenizer, tensor names/shapes) without arrays.

## Document (`"gyre":"model"`)

```json
{
  "gyre": "model",
  "version": 1,
  "arch": "char-lm",
  "config": {
    "preset": "tiny",
    "n_layer": 2,
    "n_head": 4,
    "d_model": 64,
    "d_ff": 256,
    "block_size": 64,
    "vocab_size": 65,
    "holdout": 0.1,
    "recency": "alibi"
  },
  "train": { "step": 2000, "rng_seed": 1 },
  "tokenizer": { "pretoken": "identity", "model": "bpe", "merges": [], "vocab": ["a"] },
  "tensors": [
    {
      "name": "wte.weight",
      "dtype": "f32",
      "shape": [65, 64],
      "role": "weight",
      "offset": 0,
      "nbytes": 16640
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `arch` | `char-lm`, `grok2`, `linear`, … |
| `config` | Architecture knobs (nested object, not mixed into the root) |
| `train` | `step`, `rng_seed` — updated when training overwrites the same path |
| `tokenizer` | Nested tokenizer object (same as `*.gyre.json` tokenizer files) |
| `tensors[].name` | Stable name (`wte.weight`, `blocks.0.attn.q.weight`, `m:wte.weight` for Adam) |
| `tensors[].role` | `weight` \| `adam_m` \| `adam_v` |
| `tensors[].codec` | `identity` (default), `alp` (lossless f32), `zfp` (reversible, if `GYRE_ZFP`) |
| `tensors[].nbytes` | Uncompressed size |
| `tensors[].packed_bytes` | Size of the payload blob |
| `tensors[].offset` | Byte offset from the binary payload start (omitted when `data` is inlined) |
| `tensors[].data` | JSON number array — **only** in `.gyre.json` for small models |

Training writes `identity` so weights stay bit-exact. `gyre-cli grok save --codec alp` and `gyre-cli ckpt probe FILE.gyre` compare ALP vs ZFP packed sizes (ZFP is compiled in when `GYRE_ZFP` is set). ALP is a pseudo-decimal + exception encoder (SIGMOD 2024 ALP-style) for f32; non-f32 tensors stay identity.

Load order: tensors with `role=weight` in document order map to `Module::parameters()`. Adam moments follow if present.

Tokenizer-only document (`"gyre":"tokenizer"`) is unchanged; `Tokenizer::load` still accepts it. A full model JSON also loads as a tokenizer via the nested `tokenizer` object.

## Binary layout (format_version = 2)

Little-endian. JSON sits **before** weights so `peek_gyre` reads 64 bytes + `json_len` and never touches the payload.

**Header (64 bytes)**

| Off | Size | Field |
| --- | --- | --- |
| 0 | 8 | magic `GYRE1\0\0\0` |
| 8 | 4 | `format_version` = **2** |
| 12 | 4 | `flags`: bit1 = payload CRC present |
| 16 | 8 | `tensor_count` |
| 24 | 8 | `json_len` |
| 32 | 8 | `json_offset` (64) |
| 40 | 8 | `payload_offset` (`align64(64 + json_len)`) |
| 48 | 8 | `payload_bytes` (padded payload size) |
| 56 | 2 | `shard_index` (0) |
| 58 | 2 | `shard_count` (1; reject otherwise) |
| 60 | 4 | CRC32C of bytes `[0, 60)` |

JSON document at `json_offset`. Each tensor payload is 64-byte aligned. If flag bit1, CRC32C of the padded payload is the 4 bytes after `payload_offset + payload_bytes`.

`GyreFile::open` memory-maps the file when the OS allows it and returns **one tensor** via `load_tensor(name)` without copying the rest. `peek_gyre` only seeks the header + JSON.

Writers stream tensors (no second full-file buffer). Training can overwrite the same `.gyre` path with updated weights.

## Legacy format_version = 1

Old files put JSON in a **trailer** after the payload and named tensors `w:0`, `m:0`, `v:0`. Readers still accept them. New writes are version 2 only.

## Library

```cpp
auto peek = gyre::peek_gyre("data/charlm.gyre");           // metadata only
auto file = gyre::GyreFile::open("data/charlm.gyre");      // mmap / JSON
auto w = file->load_tensor("wte.weight", device);
gyre::save_gyre(path, model.parameters(), adam, doc, model.param_names());
gyre::save_gyre_json(path, model.parameters(), nullptr, doc, names, /*data=*/true);
```

`save_gyre1` / `load_gyre1` / `peek_gyre1` remain as wrappers (`CheckpointMeta.json` is the document).
