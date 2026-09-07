# Vulkan compute backend

CPU is the **default** device. Vulkan is opt-in for CharLM **train / eval / generate**. The library does **not** read `GYRE_DEVICE` or any other env var to pick a device. Missing Vulkan is an error, not a silent CPU fallback.

```
gyre-cli lm train --data data/shakespeare.txt --preset medium --device vulkan --ckpt data/charlm.gyre
gyre-cli lm eval  --ckpt data/charlm.gyre --data data/shakespeare.txt --device vulkan
gyre-cli lm generate --ckpt data/charlm.gyre --device vulkan --prompt "To be"
```

Tokenizer training, ONNX export, GA, Helix `act`, and Grok helpers stay on CPU.

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

Storage is **device-local** `VkBuffer` plus a host-visible staging buffer. Shaders live in `src/gyre/vk/shaders/*.comp` and compile to SPIR-V at build time (`raw vulkan.h`, no volk/VMA).

Grok RoPE / MoE / GQA kernels are not implemented on Vulkan. OpenCL and CUDA are not in tree.

## Tests

`tests/vulkan_ops_test.cpp` skips if `Device::vulkan()` fails (no GPU or build without Vulkan). CPU tests stay the numeric source of truth.

## Speed notes

Vulkan can use a **larger `--batch`** than CPU if VRAM allows. Preset default batch stays CPU-safe (2–4 for nanogpt). Compare only matching holdout, tokenizer, and file.
