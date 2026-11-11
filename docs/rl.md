# Policy/value nets and PPO (`gyre/rl/*`, `gyre-infer`)

Generic RL building blocks. **Game-agnostic by construction:** inputs are float arrays, `u8`
masks and `i32` actions. Gyre never learns what an entity, unit or head *means*; the host
program decides which entities get slots, what each head chooses, and what the masks say.

Header map (`include/gyre/rl/`):

| Header | What |
| --- | --- |
| `ops.hpp` | sigmoid / tanh / relu / clamp / sub, log-softmax, gather, mask_logits, key mask, masked mean+max pool, concat/split, mask_rows, concat_rows (all with backward where a gradient exists) |
| `set_encoder.hpp` | permutation-invariant transformer over a padded entity set |
| `gru.hpp` | GRU cell with explicit per-step tape (BPTT) |
| `dist.hpp` | masked categorical heads: layout, legality + active masks, log-prob, entropy, sampling |
| `losses.hpp` | fused PPO loss, advantage normalisation, imitation loss (analytic gradients) |
| `policy.hpp` | `PolicyNet`: encoder → optional GRU → heads + value |
| `ppo.hpp` | acting (`policy_step[_rec]`), `RolloutBuffer`, GAE, `PpoTrainer` |
| `imitation.hpp` | `ImitationTrainer` (supervised pretraining through the same net) |
| `policy_io.hpp` | `.gyre` checkpoint + flat export for `gyre-infer` |

Everything here is CPU `f32`. Passing a Vulkan tensor returns an error (the rest of Gyre's GPU
path is CharLM-only, see [vulkan.md](vulkan.md)).

## Observation contract

Per sample (batch dimension `B`):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `entities` | `f32 [B, N, F]` | one feature row per slot; `N` = `max_entities` (default 128) |
| `mask` | `u8 [B, N]` | 1 = real entity, 0 = padding |
| `global` | `f32 [B, G]` | features not tied to an entity |

Entities are a **set**: there is no positional encoding, and permuting real entities (mask moved
with them) changes the output only by float rounding. Padded slots never influence the output or
receive gradient (exactly zero, tested). An empty set is legal and pools to zeros.

`SetEncoder`: `Linear(F→d)` → *L* × pre-LayerNorm blocks (non-causal self-attention with the key
mask, GELU MLP) → LayerNorm → masked **mean | max** pool → concat `global` → `Linear` → GELU.

## Heads, legality and "active" heads

A policy has *H* heads; head *h* picks one of `sizes[h]` options. Logits are packed side by side
in one row of `total = Σ sizes`. Per sample each head has:

* a **legality mask** over its options (`u8 [B, total]`, 1 = legal) — illegal options get
  probability exactly 0 and gradient exactly 0;
* an **active flag** (`u8 [B, H]`). An inactive head has no action (stored as `-1`) and
  contributes nothing to log-prob, entropy or gradients — its logits can be anything.

Joint log-prob and entropy are the **sum over active heads**. Errors (not silent NaNs): an active
head with no legal option, an out-of-range or illegal action for an active head.

**Intent-dependent relevance.** Whether a head is relevant can depend on an earlier head's
sampled action. Use `sample_conditional` (or `policy_step` / `policy_step_rec`, which wrap it)
with an `ActiveFn`:

```cpp
auto gate = [](int64_t b, int64_t h, std::span<const int32_t> acts, std::span<uint8_t> active) {
  if (h == 0 && acts[0] == 0) active[1] = 0;   // head 1 is irrelevant to intent 0
};
auto a = policy_step(net, obs, legal, all_active, &rng, /*greedy=*/false, gate);
// a->actions (-1 where inactive), a->active (FINAL mask), a->logp, a->value
```

Store `a->active` — not the initial mask — in the rollout. Heads are sampled in index order, so
a gate can only switch off *later* heads.

A pointer head (attention scores over entity embeddings as logits) is not implemented.

## PPO

```cpp
PolicyConfig pc;  pc.encoder = {...};  pc.head_sizes = {12, 8, 64};  pc.gru_hidden = 0;
auto net = PolicyNet::create(pc, *Device::cpu(), rng);

RolloutSpec spec{.steps = 128, .envs = 8, .max_entities = 128, .feat_dim = F, .global_dim = G,
                 .hidden_dim = 0 /* = gru_hidden */, .layout = net->layout()};
auto buf = RolloutBuffer::create(spec);
auto trainer = PpoTrainer::create(*net, PpoTrainConfig{});

// each iteration
buf->clear();
for (t in steps) {
  auto a = policy_step(*net, obs, legal, init_active, &rng, false, gate);
  /* step your envs with a->actions ... */
  buf->push(StepRecord{obs, legal, a->active, a->actions, a->logp, a->value, reward, done});
}
buf->compute_gae(last_value /* V(obs after final step), [envs] */, gamma, lambda);
auto stats = trainer->update(*buf);   // epochs × minibatches of Adam steps
```

Conventions:

* Sample index is `t * envs + e`.
* `done[t][e] = 1` means the episode of stream *e* ended **after** step *t*. GAE does not
  bootstrap across it. For time-limit truncation add `gamma * V(final obs)` to that step's reward,
  then set done.
* The PPO loss is a mean over the batch: clipped surrogate + `vf_coef * ½(V − R)²` −
  `ent_coef * entropy`. Reported: `approx_kl` (k3 estimator), `clip_frac`, `grad_norm`.
  `target_kl > 0` stops the update early. No value clipping.
* Advantages are normalised over the whole buffer (`normalize_advantages`), not per minibatch.

## Recurrent core (GRU)

`pc.gru_hidden > 0` inserts a GRU between the encoder and the heads. State convention: the state
fed into step *t* is the output of step *t−1*; **`reset[t][b] = 1` zeroes it first** (episode
start; normally "done of the previous step"). Gradients do not cross a reset.

* Acting: `policy_step_rec(net, obs, legal, init_active, h_in, reset, &rng, ...)` → `act` plus
  `h_out`. Put `h_in` (the state *before* reset masking) and `reset` into the `StepRecord`;
  `h_out` becomes the next `h_in`. Use `spec.hidden_dim = gru_hidden`.
* Training: `PolicyNet::forward_seq / backward_seq` run backprop through time over `T` steps ×
  `B` streams (rows are t-major). The trainer splits whole **streams** into minibatches
  (`seq_minibatch`) and uses `PpoTrainConfig::seq_len` as the BPTT chunk length (0 = whole
  rollout). Chunks after the first start from the state stored at collection time — standard
  recurrent-PPO staleness. No gradient flows into the initial state.
* `forward_seq` and step-by-step `forward_step` agree to ~1e-5 (tested).

## Imitation pretraining

```cpp
auto buf = RolloutBuffer::create(spec);                // one stream per replay (chunk)
buf->push(*make_demo_step(obs, legal, active, expert_actions, reset, h_in_zeros_if_recurrent));  // returns Result
buf->compute_gae(zeros, 0.f, 0.f);                     // only needed with value_coef > 0
auto tr = ImitationTrainer::create(*net, {.lr = 3e-3f, .minibatches = 8});
for (epoch...) tr->epoch(*buf);                        // loss, top-1 accuracy over active heads
tr->evaluate(held_out);                                // same metrics, no update
```

It is the same `PolicyNet`, so a pretrained net goes straight into `PpoTrainer`. Recurrent
streams must start with `reset = 1` and are trained over their full length. For value
pretraining set `value_coef > 0` and fill `reward`; use `compute_gae(..., lambda = 1)` with zero
values to get Monte-Carlo returns.

## Determinism

* `gyre::set_deterministic(true)` forces one OpenMP thread so float reductions have a fixed order
  (also `set_num_threads`). Same seed + same machine + deterministic mode ⇒ bit-identical
  training (tested, including PPO).
* `gyre::Rng` no longer uses `std::*_distribution`: `uniform01`, `u32`, `normal` are hand-written
  on top of `mt19937_64`, so streams do not depend on the STL. `normal` calls libm
  (`log/sin/cos`), which can differ in the last ulp across libms.
* Cross-machine requirement is only that inference output matches the training forward within
  epsilon (tested at ~2e-4 relative).

## Checkpoints and `gyre-infer`

* `save_policy(path, net, step)` / `load_policy(path, device)`: standard `.gyre` model document,
  arch `policy-net`, config in the document. No optimiser state.
* `export_policy_flat(path, net)`: fixed-schema binary for the inference runtime
  (`GPOL1`, little-endian; header + weights in `PolicyNet::parameters()` order + FNV-1a). The
  exact layout is in `policy_io.hpp`. The reader recomputes the expected float count from the
  config and rejects any mismatch, truncation or checksum failure.

`gyre-infer` (`infer/`) is a separate static library that links **nothing** — no `gyre`, OpenMP,
Vulkan, JSON, optimiser or autodiff code (enforced by `tests/infer_standalone.cpp`, which links
only it). Plain C++20, no intrinsics, so no AVX2 requirement.

```cpp
#include "gyre_infer/policy.hpp"
auto pol  = gyre_infer::Policy::load("policy.gpol", &err);
auto sess = pol->make_session();                 // one per bot/thread: recurrent state + scratch
gyre_infer::Input in{.entities = ent, .mask = mask, .n_slots = n, .global = glob, .reset = new_ep};
std::vector<float> logits(pol->info().total_logits());  float value;
pol->step(sess, in, logits.data(), &value);
int a0 = gyre_infer::pick_masked(logits.data() + pol->info().head_offset(0), legal0, size0, u);
```

`Policy` is immutable (share it); `Session` is not (one per thread). Legality, head gating and
action selection are the caller's job, exactly as in training. `n_slots` may be smaller than the
training `max_entities`; padding is skipped, so cost scales with real entities.

Measured on one core (Windows, MSVC `/O2`, baseline x64), 128 real entities, with GRU:

| model | params | ms / decision |
| --- | --- | --- |
| d=64, 2 layers, ff=128 | 0.20M | ~2.0 |
| d=96, 2 layers, ff=384 | 0.51M | ~4.6 |
| d=128, 3 layers, ff=512 | 1.09M | ~9.7 |

Numbers are machine-specific. The next speed-up would be blocked or runtime-dispatched AVX2
kernels (not done: AVX2 must not be required).

## Tests

`tests/gradcheck.hpp` is a finite-difference harness (central differences on sampled elements of
every parameter and input). It covers: Linear, LayerNorm, causal attention, cross-entropy, GELU∘
softmax, every op in `rl/ops.hpp`, the set encoder (parameters + entity + global inputs), GRU BPTT
with resets, heads (log-prob + entropy), PPO loss, imitation loss, and the whole `PolicyNet`
(feedforward and recurrent) through the PPO loss.

Behavioural tests: padded content bit-identical, permutation invariance, gated sampling, GAE
against hand-computed values, stored vs recomputed log-prob (first-step ratio is exactly 1),
PPO learns a gated toy task, a GRU solves a memory task that a feedforward net cannot,
imitation (feedforward + recurrent), checkpoint round-trip, inference parity (feedforward +
recurrent with resets), corruption handling.

## Measured: Shakespeare through the RL stack (sanity check)

`examples/shakespeare_policy.cpp` (`gyre-shakespeare-policy`) does next-character prediction as
imitation learning: observation = last *K* characters as a set (one-hot char | one-hot recency),
one head with *V* options, GRU core, `ImitationTrainer`, then `export_policy_flat` and sampling /
scoring through `gyre-infer`. Same file and held-out slice as `gyre-cli lm eval --split 0.1`
(last 10% = 111,540 chars; the RL example scores 832 of the 871 segments because it evaluates in
batches of 64). Score = `10*(4.174 - nats)/(4.174 - 1.47)` as in [wire-evals.md](wire-evals.md).

| model | params | text seen | nats/char | BPC | /10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CharLM medium (dropout 0.1, no ALiBi) | 826K | 6.1M | 1.746 | 2.52 | 8.98 |
| RL stack, GRU 256, K=1 | 483K | 6.1M | 1.623 | 2.34 | 9.43 |
| RL stack, set encoder K=16 + GRU 256 (stopped at 860/1500 steps) | 517K | ~3.5M | 1.698 | 2.45 | 9.16 |
| CharLM medium | 826K | 24.6M | 1.539 | 2.22 | 9.75 |
| RL stack, GRU 256, K=1 | 483K | 24.6M | **1.506** | 2.17 | **9.87** |

Single runs, different learning-rate schedules and regularisation between the two families: the
0.03 nats/char lead at 24.6M characters is not evidence of an architecture win (the gap shrank
from 0.12 at 6.1M). What it shows: the recurrent path trains stably on real data, and
`gyre-infer` reproduces the trainer's held-out loss to four decimals on that data.

## Measured: training throughput (`gyre-bench-rl`, `gyre-bench-ops`)

One decision = one set of entities through encoder + GRU + heads; forward + backward + Adam.
d_model 64, 2 layers, GRU 128, 0.2M params, 32-thread Windows machine, MSVC `/O2`, `f32`.
"before" is the first measurement (naive kernels, fresh allocation per tensor); "after" is the
current tree.

| entities | train samples/s before | after | acting decisions/s before | after |
| ---: | ---: | ---: | ---: | ---: |
| 32 | ~1150 | ~3300 | ~4900 | ~5800 |
| 64 | ~490 | ~2500 | ~2200 | ~3750 |
| 128 | ~210 | ~1150 | ~970 | ~3200 |

At 128 entities a PPO iteration of 100k samples x 4 epochs now takes ~6 min instead of ~30.
One thread: ~130 -> ~180 samples/s; 8 threads: ~205 -> ~1085 (the old kernels were mostly
single-threaded and memory-bound, so extra cores did nothing).

What changed:

* **Buffer pool** (`Storage`): buffers >= 1 MiB are recycled instead of freshly page-faulted. This
  alone was ~3.5 ms per `[B,H,N,N]` tensor. `Tensor::empty` still returns zeroed memory;
  `Tensor::uninit` skips the zeroing and is used by ops that overwrite their whole output
  (`add`, `matmul`, `bmm`, `transpose`, `permute`, `gelu`, `softmax`, `layer_norm`, ...).
* **Fused attention** (`attn_probs`, `attn_probs_backward`): q.k, scale, key mask and softmax in
  one pass per row instead of transpose + bmm + scale + mask + softmax (four `[B,H,N,N]` tensors),
  and the backward folds `dy.v`, the softmax Jacobian and the scale into one kernel.
* **`matmul_tn` / `bmm_tn`**: `A^T B` without materialising the transpose (used by `Linear`
  weight gradients and the attention backward). Reductions use fixed 256-row blocks, so results
  do not depend on the thread count (tested).
* Parallelised and tightened: `gelu_backward` (was serial, ~13 ms), `softmax_last_backward`,
  `mul_scalar`, `sum_dim` (was a per-element index decode), blocked `transpose_last2`, row-copy
  `permute`.

No AVX2 is required anywhere: the GEMM is still the plain loop (now ~290 GFLOP/s aggregate on
this machine once the memory stalls are gone). A register-blocked / runtime-dispatched AVX2
kernel is the next lever if matmuls become the limiter. The recurrent training loop over many
short GRU steps (e.g. the Shakespeare example, 128 sequential steps) is overhead-bound and did
not speed up.

## Limits / not done

* Verified only on toy tasks, never on real game data.
* Pointer head, value clipping, per-head entropy coefficients, multi-GPU, any GPU path for this
  stack.
* PPO hyperparameters are untuned defaults.
* Chunked BPTT for *imitation* (demos have no stored hidden states).
* Not built or run on non-Windows / non-MSVC toolchains.
