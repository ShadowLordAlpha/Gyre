# Gyre Vulkan CharLM review (train/eval vs CPU)

Copied from the session scratch file (that path is not in either git tree):
`C:\Users\Shadow\.grok\sessions\E%3A%5CGithub%5CHelix\01a0b238-4634-7e31-b0dc-b085fd5c327c\gyre-vulkan-review.md`

Mode: local (Gyre Vulkan stack). Helix is context only (Sovereign: d=192, L=6, nHead=6, dFf=768, train T=256, B=32, V=2000, ALiBi). Live Helix `act` stays CPU.

Issue 1 (GELU ±8 vs CPU ±40) was patched in `src/gyre/vk/shaders/elem.comp` after this review; rebuild Gyre so SPIR-V is regenerated. Other issues below are still open.

## Summary

Vulkan CharLM is a real compute path (device-local buffers, 16×16 f32 GEMM, row softmax/LN/CE, Adam), not a silent CPU fallback: missing Vulkan errors, and `add`/`mul`/`matmul`/`bmm`/`embedding` hard-error on mixed devices. ALiBi is token-distance only and matches CPU (`slope = 2^(-8(h+1)/H)`, causal `-1e9`). Train vs eval dropout matches CPU (`p=0` or `ctx.train=false` is a no-op; train builds the mask on the host). The live math defect is GELU: GPU clamps the pre-activation to ±8 (and the `tanh` argument) while CPU clamps to ±40 with no `u` clamp — tests never hit `|x|>2`, so they cannot catch the old NaN-last-hidden class of bug at Sovereign scale. Speed is dominated by the tiled f32 GEMM (especially `B·H` BMMs of `[T,T]` at T=256), not the ~64 KiB token-window upload; `upload()` still full-flushes the compute stream around every host copy. Existing atol `1e-4`/`2e-4` on 2×4 tensors is not a substitute for hidden/loss parity at B=32, T=256, d=192, V=2000.

## Issues

### Issue 1 -- Severity: bug
- File: E:\Github\Gyre\src\gyre\vk\shaders\elem.comp:36
- Description: Vulkan GELU forward and backward clamp the input to `[-8, 8]` and also clamp `u = s*(v + c*v^3)` to `[-8, 8]`. CPU GELU (`ops.cpp:461`) and `gelu_backward` (`ops.cpp:812`) clamp only the input to `[-40, 40]` and do not clamp `u`. For `|x| in (8, 40]`, CPU computes `GELU(x) ≈ x` (tanh saturated) while GPU emits `GELU(±8) ≈ ±8`, so the MLP residual is hard-clipped on Vulkan only. That is the same family of hidden-state distortion that previously stuck SFT at uniform NLL when GPU GELU diverged; `AddGeluSoftmaxLn` only feeds values in `[-1, 2]` (`vulkan_ops_test.cpp:91`) so the mismatch is untested. DecoderBlock runs this on `fc1` (`transformer.cpp:155`, `:177`) every train step.
- Suggestion: Use the same clamp on both devices (prefer CPU’s `[-40, 40]` on `x` only, or document a shared constant). Drop the extra `u` clamp or apply it on CPU too. Add a parity test with `|x| in {8, 16, 40}` for forward and backward.
- Status: patched in `elem.comp` (clamp ±40, no extra `u` clamp) plus `Vulkan.GeluClampMatchesCpu`

### Issue 2 -- Severity: suggestion
- File: E:\Github\Gyre\tests\vulkan_ops_test.cpp:129
- Description: Numeric tests use `expect_close` atol `1e-4` (default, line 27) or `2e-4` (GELU/softmax/LN, line 103) on tiny tensors (matmul 2×3, LN last-dim 4). `TinyTrainStep` is T=16, d=16, 1 layer, batch 2, and only asserts finite positive loss — no CPU vs Vulkan hidden or NLL compare. `CharLMHiddenEvalIsFinite` is T=8, d=16, 2 heads, vocab 64, eval only. There is no test for GELU backward, LN backward, softmax CE (V=2000), ALiBi, embedding gather, BMM attention scores, or Adam against CPU. Tree reductions in `row.comp` (softmax/LN mean-var over C=192 or V=2000) will differ from CPU OpenMP’s sequential sum in last bits; `2e-4` on 4-col rows does not bound Sovereign CE.
- Suggestion: Add a holdout: same seed/weights, `CharLMConfig` d=192 L=6 H=6 dFf=768 T=256 V=2000 ALiBi, B=2 (or 32 if VRAM allows), compare last-hidden and mean NLL to CPU with a stated atol (e.g. `1e-3` hidden, `5e-4` NLL) and `|x|_max` of pre-GELU. Keep TinyTrainStep as a smoke, not the parity gate.
- Status: patched (`Vulkan.CharLMHiddenMatchesCpuSovereignWidth`: tinygpt width, T=256, V=2000, B=1)

### Issue 3 -- Severity: suggestion
- File: E:\Github\Gyre\src\gyre\vk\shaders\idx.comp:47
- Description: GPU embedding gather zeros OOB ids (`id < 0 || id >= V`); CPU embedding (`ops.cpp:365`) returns `index OOB`. Neither clamps token ids to `[0, vocab-1]`. Vulkan CE (`row.comp:206-208`) treats OOB targets as `pt = 1e-12` and never subtracts 1 from a class; CPU CE (`module.cpp:61`) errors. Valid Helix BPE ids will match. Invalid ids on Vulkan train as a zero row / uniform-ish grad and hide data bugs that CPU would abort.
- Suggestion: Match CPU: reject OOB in `vkops::embedding` / CE (or clamp both sides to `[0, V-1]` if that is the intended Helix policy). Add a unit test for id `-1` and `V`.
- Status: patched (`vkops::embedding` / `softmax_cross_entropy` host-check ids; `Vulkan.EmbeddingOobMatchesCpu`)

### Issue 4 -- Severity: suggestion
- File: E:\Github\Gyre\src\gyre\ops.cpp:568
- Description: Binary ops error on mixed devices (`mixed_device` at `ops.cpp:28`, `:154`, `:180`, `:349`). `layer_norm` / `rms_norm` / `layer_norm_backward` / `linear`’s `bias_add_` / CE only test `is_vk` on the primary tensor. `VulkanDevice::dispatch` (`runtime.cpp:543-548`) then binds the 16-byte dummy SSBO for any tensor without GPU storage. That is not a CPU fallback; it is silent garbage weights/bias/targets. CharLM create() moves all params to the same device, so the train loop does not hit this unless a caller leaves LN scale or CE targets on CPU.
- Suggestion: If `is_vk(x)` and any other operand is not Vulkan, return `mixed_device` before dispatch (same as `embedding`).
- Status: patched (`require_vk` / `on_device` before LN/RMS/CE/bias; `Vulkan.LayerNormMixedDeviceErrors`)

### Issue 5 -- Severity: suggestion
- File: E:\Github\Gyre\src\gyre\vk\shaders\gemm.comp:1
- Description: Train bottleneck is the 16×16 tiled f32 GEMM (`local_size 16×16`, no tensor cores, K loop in 16-wide tiles), especially attention `bmm` at Sovereign shape: batch `B·H = 192`, `M=N=256`, `K=32`, plus MLP `[B·T, 192]×[192, 768]`. Token windows are `2 × 32 × 256 × 4 ≈ 64 KiB` per step (`data.cpp:35-36`) and are not the limiter. Staging is already persistently mapped (`runtime.cpp:69-72`, `ensure_staging`), but `upload()` (`runtime.cpp:453-466`) `flush()`es the recorded compute stream, memcpy’s, then `flush()`es the copy — two full `vkQueueSubmit`+wait per tensor, every batch, even though the copy is tiny. The descriptor ring is 512 (`runtime.cpp:386`); a CharLM step is hundreds of dispatches, so the step also mid-flushes. `clip_grad_norm` on Vulkan (`optim.cpp:75-83`) does `mul`+`sum`+`item_f32()` per parameter (~100 host downloads) when `--grad-clip` is on (default 0). Workgroup 256 for elem/row/idx/adam is reasonable occupancy; GEMM 256-wide tiles are the occupancy/math limit, not those. Debug is not a fair speed test (`docs/vulkan.md:30`). Batch 64–128 at T=256: attention scores `[B,6,256,256]` are ~50 MiB at B=32 and scale linearly; weights+Adam moments are small (~tens of MiB). VRAM blowup is unlikely below ~B=128 on 8 GB; NaNs are more likely from GELU/softmax than from raising batch (no extra Vulkan batch cap).
- Suggestion: Keep tokens on a persistent mapped staging buffer and copy without flushing the previous step’s compute (double-buffer cmd + staging). Larger GEMM tiles and/or fp16/tensor-core paths would move the real wall. If `grad_clip > 0`, reduce to one device-side sum of squares. Compare speed only in Release. Raising Helix fit batch above 32 is fine for VRAM at T=256; re-check finite hidden after Issue 1 is fixed.
- Status: partial — 4-slot staging ring so `upload()` records into the live command buffer instead of flush-per-copy. GEMM remains 16×16 f32 (no tensor cores).
