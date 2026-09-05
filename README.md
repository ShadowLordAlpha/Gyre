# Gyre

C++23 library for **small neural models** (pre-LN decoder transformers) and **genetic algorithms** on CPU tensors. Product name is **Gyre**. Default `--data data/shakespeare.txt` is a corpus path, not the product name.

**Helix / game hosts:** start at [docs/helix.md](docs/helix.md). Gyre is a **static in-process runtime** (`gyre.lib`). Helix schemas stay out of this tree.

License: [MIT](LICENSE).

## What ships

| Target | Role |
| --- | --- |
| `gyre` | Static library — tensors, ops, NN, GA, checkpoints, adapter stub |
| `gyre-cli` | Train / eval / generate / tok / grok / ga |
| `gyre-tui` | Optional FTXUI dashboard (`GYRE_ENABLE_TUI`) |
| `gyre-tests` | gtest |
| `gyre-charlm`, `gyre-ga-onemax` | Small examples |

Public headers live under `include/gyre/`. Umbrella: `gyre/gyre.hpp`.

## Status (for Helix)

**Usable now**

- CPU Device, f32 tensors, Linear / LayerNorm / CharLM transformer, Adam, GYRE1 `.gyre` checkpoints
- Tokenizers: BPE (default), chars, bytes, unigram; reuse `--tok FILE.gyre.json`
- GA: tournament, elite, OneMax (`gyre-cli ga`)
- Embed stub: `gyre::adapt::{Agent, Environment, PolicyAgent, GridWorld}`
- CharLM ONNX export (no ONNX Runtime link)

**Not in tree (do not wait on these for a first Helix link)**

- Elman / leaky RNN
- Island GA, neuroevolution flatten/unflatten
- Helix or StarCraft observation schemas
- Vulkan / OpenCL / CUDA
- Full Grok-2 270B generate

## Build

Requires CMake ≥ 3.28, C++23, Ninja recommended. Windows: `vcvars64` then:

```
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Options:

| CMake option | Default | Meaning |
| --- | --- | --- |
| `GYRE_BUILD_TESTS` | ON | `gyre-tests` |
| `GYRE_ENABLE_TUI` | ON | Fetch FTXUI, build `gyre-tui` |
| `GYRE_BUILD_EXAMPLE_BINS` | ON | charlm / OneMax examples |
| `GYRE_ENABLE_OPENMP` | ON | Multi-thread matmul |
| `GYRE_ENABLE_VULKAN` | OFF | Unimplemented |

Use **Release + OpenMP** for training. Debug `gyre-cli` is not a fair speed test.

Corpus: `scripts/fetch_shakespeare.ps1` → `data/shakespeare.txt` (Karpathy tinyshakespeare).

```
cmake --build cmake-build-release --target gyre-tests
ctest --test-dir cmake-build-release --output-on-failure
```

Install exports `gyre::gyre` via `cmake/GyreConfig.cmake.in`.

## CLI

```
gyre-cli ga [--gens 80] [--n 64] [--dim 64]

gyre-cli lm train --data data/shakespeare.txt [--preset medium|tiny|tinygpt|nanogpt]
                 [--tokenizer bpe|chars|bytes|unigram] [--tok FILE.gyre.json]
                 [--vocab-size 2000] [--holdout 0.1] [--steps 2000] [--batch 4]
                 [--ckpt data/charlm.gyre] [--recency alibi|none]

gyre-cli lm generate --ckpt data/charlm.gyre --data data/shakespeare.txt
                    [--prompt "To be"] [--chars 200] [--temp 0.8]

gyre-cli lm eval --ckpt data/charlm.gyre --data data/shakespeare.txt [--split 0.1]

gyre-cli lm export --ckpt data/charlm.gyre --onnx data/charlm.onnx

gyre-cli tok train|export|import|encode ...
gyre-cli grok info|inspect|compress-probe|pack|save|gen ...
```

Checkpoints use extension **`.gyre`** (binary `GYRE1` v2: JSON document first, then aligned weights). Small models also write a readable **`.gyre.json`** twin of the same document. Spec: [docs/gyre-file.md](docs/gyre-file.md). Default tokenizer is **BPE** (vocab 2000). Chars/bytes are BPE with **no merges**. Default recency is **ALiBi** (token distance only). Train **holdout** default `0.1` (last raw-byte fraction unused for BPE/LM). Fair LM scores: **nats/char or BPC**, not nats/token across tokenizers.

Presets:

| preset | typical size | T | notes |
| --- | --- | --- | --- |
| tiny | ~0.2M | 64 | debug |
| medium | default | 128 | d=128 L=4 H=4 |
| tinygpt | ~3–4M | 256 | d=192 L=6 H=6 |
| nanogpt | ~10.7M @ V≈65 | 256 | d=384 L=6 H=6 |

Sampling: temperature ~0.65–0.75 plus a cue; greedy often loops.

nanoGPT-like **char** recipe (held-out 10%, no ALiBi): see [AGENTS.md](AGENTS.md).

## Library map

```
include/gyre/
  gyre.hpp              umbrella
  tensor.hpp device.hpp ops.hpp rng.hpp
  module.hpp optim.hpp checkpoint.hpp data.hpp
  ga/population.hpp
  nn/layers.hpp transformer.hpp tokenize.hpp bpe.hpp …
  train/loop.hpp
  adapt/agent.hpp environment.hpp
  export/onnx.hpp
```

Errors: `gyre::Result<T>` (`std::expected`). Logging: `log.hpp` (default no-op).

## Docs

Index: [docs/README.md](docs/README.md).

- [docs/helix.md](docs/helix.md) — embed contract and gap list for Helix
- [docs/design.md](docs/design.md) — design (some “v1 plan” text is historical; code has moved past the original stub)
- [docs/tokenizer.md](docs/tokenizer.md)
- [docs/grok.md](docs/grok.md)

## Data not in git

Large Hub shards and local `.gyre` training runs are gitignored (`data/grok2/*.safetensors`, `data/*.gyre`). Fetch or train locally. `data/shakespeare.txt` may be committed when present (~1 MiB).
