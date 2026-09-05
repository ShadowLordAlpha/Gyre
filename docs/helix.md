# Helix embed: what Gyre can do today

Gyre is a **C++23 static library** (`gyre`) plus `gyre-cli`. Helix (and later StarCraft bots) are **out of tree**. This repo does not contain Helix observation schemas, command enums, or game SDKs.

Intended runtime: Helix **links `gyre` in-process**. It is not a sidecar RPC server. Train offline with `gyre-cli`; load a `.gyre` checkpoint (or a `PolicyAgent`) at tick time.

## Verdict for reviewers

| Helix need | Ready? | Notes |
| --- | --- | --- |
| Separate NVIDIA-free CPU runtime | **Yes** | Static lib, `std::expected` public API, MIT |
| Load / save weights | **Yes** | GYRE1 (`.gyre`), magic `GYRE1` |
| Deterministic `act(obs, rng)` | **Stub** | `gyre::adapt::PolicyAgent` is one `Linear` + argmax |
| Helix obs/action schema | **No** | Host-owned; implement `Environment` or skip it and call `Agent::act` |
| Discrete or continuous actions | **API yes** | `ActionSpace::discrete_int` / `continuous_f32`; PolicyAgent is discrete only |
| Genetic algorithms | **Kernel yes** | Tournament k=3, elite, OneMax. **No islands. No param flatten for neuroevolution** |
| Recurrent controller (Elman) | **No** | Not implemented; ALiBi transformer used for sequence LMs instead |
| Small transformer LM | **Yes** | CharLM presets `tiny` … `nanogpt`; BPE/chars/bytes/unigram |
| ONNX export of CharLM | **Yes** | Write-only; no ORT link |
| Vulkan / OpenCL | **No** | Reserved `DeviceKind` only |
| Full Grok-2 generate | **No** | Tiny/mini Grok-shaped nets + inspect/pack; 270B not runnable |

**You do not need Elman RNN before Helix can try Gyre.** You need a Helix-side adapter and a policy sized to Helix obs/actions. **You need more GA only if Helix evolves network weights** (`flatten_params` is not in the tree).

## Link

CMake package after install (`gyre::gyre`), or add this repo as a subdirectory:

```cmake
add_subdirectory(path/to/Gyre)
target_link_libraries(helix_ai PRIVATE gyre)
```

Public umbrella: `#include "gyre/gyre.hpp"`.

Build: MSVC (`vcvars64`) + Ninja, C++23, CMake ≥ 3.28. Prefer **Release + OpenMP** (`GYRE_ENABLE_OPENMP`, default ON). Debug is too slow for training; inference of a tiny Linear policy is fine.

```
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

TUI is optional (`GYRE_ENABLE_TUI`). Tests: `GYRE_BUILD_TESTS`. Vulkan option exists and is unimplemented.

## Adapter contract

Headers: `include/gyre/adapt/environment.hpp`, `include/gyre/adapt/agent.hpp`.

```cpp
namespace gyre::adapt {

enum class ActionSpace : std::uint8_t { discrete_int = 1, continuous_f32 = 2 };

struct Action {
  ActionSpace space{ActionSpace::discrete_int};
  std::int32_t discrete{0};
  std::vector<float> continuous;
};

struct StepResult {
  Tensor observation;  // CPU, f32, contiguous; copy shares Storage (view)
  float reward{0};
  bool done{false};
};

class Environment {
 public:
  virtual Result<Tensor> reset(Rng&) = 0;
  virtual Result<StepResult> step(const Action&) = 0;
  virtual std::span<const std::int64_t> observation_shape() const = 0;
  virtual std::size_t action_dim() const = 0;
  virtual ActionSpace action_space() const = 0;
  virtual ~Environment() = default;
};

class Agent {
 public:
  virtual Result<Action> act(const Tensor& observation, Rng&) = 0;
  virtual Result<void> observe(const StepResult&) = 0;  // may no-op
  virtual Result<void> load(const std::filesystem::path&) = 0;
  virtual Result<void> save(const std::filesystem::path&) const = 0;
  virtual ~Agent() = default;
};

}  // namespace gyre::adapt
```

Invariants:

- Host thread only. Deterministic given observation bytes + `Rng` state.
- Observations: CPU `DType::f32`, shape equals `observation_shape()`.
- Replay: host records `(seed, actions)`; Gyre does not snapshot the game.
- `GridWorld` is a 5×5 fake env for tests (`tests/adapt_grid_test.cpp`).
- `PolicyAgent::create(obs_dim, n_actions, device, rng)` — Linear logits, greedy discrete.

Helix should:

1. Pack a tick observation into a CPU f32 `Tensor`.
2. Call `agent.act(obs, rng)` (or a custom `Agent` that runs `CharLM` / MLP).
3. Map `Action` to Helix commands.
4. Optionally `observe(step)` for online learning later.

Do **not** put StarCraft or Helix types in this repository.

## GA

`include/gyre/ga/population.hpp`

- Genes: `u8` or `f32` tensors; fitness `float(*)(std::span<const float>)`.
- `random_population` → `evaluate` → `step` (tournament, elite, crossover, mutate).
- Example: `examples/ga_onemax.cpp`, CLI `gyre-cli ga`.

Missing for neuroevolution: island model; flatten/unflatten of `Module` parameters. Fitness stays host-sequential on CPU.

## Neural stack Helix can reuse

| Type | Header | Use |
| --- | --- | --- |
| `Linear`, `LayerNorm` | `nn/layers.hpp` | Policies, heads |
| `CharLM` | `nn/transformer.hpp` | Pre-LN decoder; ALiBi optional |
| Tokenizer | `nn/tokenize.hpp` | BPE / chars / bytes / unigram; `*.gyre.json` |
| Adam + `TrainLoop` | `optim.hpp`, `train/loop.hpp` | Offline train |
| GYRE1 | `checkpoint.hpp` | `.gyre` files |
| ONNX | `export/onnx.hpp` | CharLM inference graph, no ORT |

Presets (CLI `--preset`): `tiny`, `medium` (default), `tinygpt`, `nanogpt`. Recency default **ALiBi**; `--recency none` for vanilla causal attention.

There is **no** `nn/rnn.hpp`. Sequence memory for LMs is attention + ALiBi.

## Errors and threading

Public fallible APIs return `gyre::Result<T>` (`std::expected`). Exceptions inside `.cpp` must not escape. Hosts see `expected` only.

Kernels: CPU f32 (i32/u8 where needed). OpenMP GEMM when built with OpenMP. No CUDA.

## Suggested Helix spike (no new Gyre features)

1. Add Gyre as a CMake subdir; link `gyre`.
2. Implement a 20-line Helix `Agent` wrapper around `PolicyAgent` or a custom Linear.
3. Confirm tick latency on a dummy obs (design target: small nets, milliseconds on CPU).
4. If you need evolved nets, file a request for param flatten; do not block the spike on islands or RNN.

Further reading: [design.md](design.md) (architecture + GYRE1 bytes), [tokenizer.md](tokenizer.md), [grok.md](grok.md) (not required for Helix v1).
