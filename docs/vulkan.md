# Vulkan compute backend

CPU is the **default** device. Vulkan is opt-in for CharLM **train / eval / generate**. The library does **not** read `GYRE_DEVICE` or any other env var to pick a device. Missing Vulkan is an error, not a silent CPU fallback.

```
gyre-cli lm train --data data/shakespeare.txt --preset medium --device vulkan --ckpt data/charlm.gyre
gyre-cli lm eval  --ckpt data/charlm.gyre --data data/shakespeare.txt --device vulkan
gyre-cli lm generate --ckpt data/charlm.gyre --device vulkan --prompt "To be"
```

Tokenizer training, ONNX export, GA, Helix `act`, and Grok helpers stay on CPU. CharLM `--dropout` / `--decay` (AdamW) and `--prune` run on Vulkan the same as CPU when `--device vulkan` is set.

## Build

Needs a Vulkan SDK (`VULKAN_SDK`) with `glslangValidator` on `PATH` (or `$VULKAN_SDK/Bin`). CMake option `GYRE_ENABLE_VULKAN` is **ON** in a fresh tree and turns itself **off** if the SDK or `glslangValidator` is missing (CI without a GPU/SDK still builds CPU-only).

Existing CMake caches that were generated when the option defaulted to OFF need a one-time:

```
cmake -S . -B cmake-build-gyre -DGYRE_ENABLE_VULKAN=ON
```

Force off: `-DGYRE_ENABLE_VULKAN=OFF`.

```
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Use **Release**. Debug Vulkan is not a fair speed test.

## API

```
Device::cpu()                 // default singleton
Device::vulkan()              // compute device, or Error
Device::open("cpu"|"vulkan")  // CLI helper; "gpu" aliases vulkan
```

`Tensor::empty` / `from_host` / `to` / `copy_from_host` / `copy_to_host` work on both devices. `host_span` / `host_bytes` stay CPU-only (`Errc::not_cpu`). Checkpoints stay GYRE1 on disk (download before write).

## What runs on GPU

Weights, activations, Adam moments, and CharLM forward/backward (GEMM, LN, softmax, GELU, attention mask, embeddings, CE). Token windows are sampled on CPU and uploaded each step. Loss scalar and generate sampling download to host.

Storage is **device-local** `VkBuffer` plus a 4-slot host-visible staging ring.
Uploads record a copy into the current compute command buffer (flush only when
all slots are in use, on download, or when the descriptor ring fills). Shaders
live in `src/gyre/vk/shaders/*.comp` and compile to SPIR-V at build time (`raw
vulkan.h`, no volk/VMA). GELU matches CPU: clamp `x` to ±40. `tanh`’s argument
is clamped to ±20 on GPU only so `exp` does not overflow to NaN (`std::tanh`
saturates). Mixed-device LN/CE/bias operands error (`mixed_device`); OOB
embedding/CE ids error (`index OOB` / `target OOB`) like CPU.

Limits and memory are checked, not assumed: a dispatch past
`maxComputeWorkGroupCount`, a tensor larger than `maxStorageBufferRange`, or a
misaligned bind offset is an error (lower `--batch`). Out of VRAM (or the
driver's allocation-count cap) frees the buffer pool and retries once, then
errors; it never falls back to host-visible memory (that would move a discrete
GPU's training to system RAM over PCIe). Host-visible memory is used only on a
device with no device-local storage memory type.

### Reproducibility

Vulkan CharLM training is **bit-reproducible run to run**: the same seed,
data, flags, binary, GPU and driver give a byte-identical `.gyre` (dropout,
AdamW and grad clipping included). No kernel uses float atomics or any other
reduction whose order follows GPU scheduling:

- In-workgroup reductions (softmax, LN stats, CE, `sum`) use a fixed shared-memory
  tree; GEMM sums K in tile order; `sum_dim` / batch-to-time sums loop in order.
- LayerNorm gamma/beta grads: the row pass stores each row's mean and 1/std,
  then one invocation per (column, 32-row chunk) sums its rows in order, and a
  last pass adds the chunk partials into the grad in chunk order (with one chunk
  it accumulates straight into the grad, in the CPU loop's order).
- Embedding grads: the token ids are read back and stable-sorted by id on the
  host (counting sort), then one invocation per (used id, column) adds that id's
  rows in ascending token order, the same order as the CPU loop. Out-of-range ids
  error (`emb idx`) like CPU.

Limits: results are reproducible only on the same GPU model, driver version and
`gyre` build (the shader compiler may contract or reorder float math
differently on another driver), and they are **not** bit-equal to CPU (GEMM and
reduction trees differ; parity is within the tolerances in
`tests/vulkan_ops_test.cpp`). They do not depend on CPU thread count or OpenMP
settings. `Vulkan.BackwardIsBitReproducible` repeats LN/embedding backward and a
few CharLM train steps and requires identical bytes.

Grok RoPE / MoE / GQA kernels are not implemented on Vulkan. OpenCL and CUDA are not in tree.

## Tests

`tests/vulkan_ops_test.cpp` skips if `Device::vulkan()` fails (no GPU or build without Vulkan). CPU tests stay the numeric source of truth.

## Speed notes

Vulkan can use a **larger `--batch`** than CPU if VRAM allows. Preset default batch stays CPU-safe (2–4 for nanogpt). Compare only matching holdout, tokenizer, and file.
