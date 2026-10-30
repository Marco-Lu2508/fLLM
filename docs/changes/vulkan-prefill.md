# Vulkan: faster prompt processing (tiled quantized matrix-matrix kernel, hardware dot product with every compiler)

Test level: **compiled and unit tested** (all shaders compile with glslc on a Linux build server, the planner, the tile choice and a transcription of the
kernel arithmetic are tested for every weight format, the plans match the compiled modules). **Not run on a GPU**: the speed and the results on real
devices are to be measured with `fllm-rs selftest-gpu --device vulkan` and a prompt benchmark.

## Why

On an Intel integrated GPU (Xe, subgroup 32, int8 dot product) a 505-token prompt of a 2B Q4_K_M model ran at about 100 tokens/s, three to four times
slower than other Vulkan engines, while the answer (a memory-bound matrix-vector product) was already on par. Two findings from reading the kernels:

1. The existing prompt kernel `mm_<fmt>_i` reads the weight tile from shared memory with a stride of 32 words between neighbouring lanes (every lane of a
   subgroup hits the same memory bank, 16-way conflict), scales every 8 values instead of once per scale group and synchronizes twice per 32 values.
2. The Windows release build compiles the shaders with the Vulkan SDK 1.4.309 (glslang 15.0). `GL_EXT_integer_dot_product` only exists since
   glslang 15.2, so those builds silently used the emulation of the dot product (shifts and four multiplies) instead of one dot instruction.

## What changed

* `shaders/vulkan/_q_ext.glsl`: if the compiler lacks `GL_EXT_integer_dot_product` but knows `GL_EXT_spirv_intrinsics` (every glslang since 2022) the
  shaders emit `OpSDot` / `OpSDotAccSat` themselves (same capabilities and extension as the native path). All `_i` shaders (matrix-vector and
  matrix-matrix) benefit. Build-time switch to go back to the emulation on such compilers: `FLLM_VK_NO_SPIRV_DOT=1` when building. A compiler that knows
  neither still gets the emulation. (Alternatively use a Vulkan SDK with glslang >= 15.2.)
* `shaders/vulkan/mm_q.comp` (`mmq_<fmt>_i`, 16 variants: Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL MXFP4 NVFP4 Q4Z32/64/128): the new prompt
  kernel. BM x BN output tile per workgroup (default 64 x 64, 256 invocations, 4 x 4 outputs per invocation), shared-memory tiles stored word-major so
  that a subgroup reads consecutive words (no bank conflicts) and writes consecutive output rows, double-buffered tiles with one barrier per K step of
  32 and the global loads of the next step overlapping the arithmetic of the current one, activations loaded as 16-byte vectors, integer dot products
  accumulated with the accumulating dot instruction over a whole scale group and converted/scaled once per group
  (`acc += float(isum) * (ws * d_act) + wm * (d_act * sum(act))`; the activation sums are computed once while staging). Same inputs, bindings and
  push constants as `mm_<fmt>_i`, so the activation prepass, the slicing of big weights and the unified-memory allocation are unchanged.
* `weights/plan.rs`: `mmq_tile`, `MmqKnobs`. The planner picks `mmq_<fmt>_i` instead of `mm_<fmt>_i` for every batch above the matrix-vector range (9 and more) when the variant is the int8 dot
  product; spec constants are (workgroup size, BM, BN), the grid is `ceil(rows / BM) x ceil(batch / BN)`. The prepass buffer is rounded up to 16 bytes.
* Tests: `weights/tests/mmq.rs` (arithmetic transcription for every format incl. the check that scale and offset are constant inside each scale group,
  selection and fallbacks, tile choice for several device limits, the module contains the dot instruction and its capabilities), the registry/plan
  consistency tests, and new prompt-sized cases (batches 65 and 505, wide rows, odd tile edges) in the fast pass of `selftest-gpu` (group `linear`;
  batches 40, 129 and 356 of the existing fast cases also run through the new kernel).

## Flags (read once per process)

| variable | effect |
|---|---|
| `FLLM_VK_MM=0` (or `legacy`, `off`) | never use the new kernel: `mm_<fmt>_i` as before (A/B test) |
| `FLLM_VK_MM_MIN_BATCH=N` | smallest batch served by the new kernel (default and minimum 9: the first batch above the matrix-vector range; it was 32 until 2026-10-30) |
| `FLLM_VK_MM_BM=32\|64\|128`, `FLLM_VK_MM_BN=64\|128` | tile size; reduced automatically when the device's workgroup or shared-memory limit does not allow it |
| `FLLM_VULKAN_NO_IDOT=1`, `FLLM_VULKAN_EXACT=1` | as before: the new kernel is an int8-dot kernel, so both switch it off (the f32 `mm_<fmt>_s` serves) |
| `FLLM_VK_PROFILE=1` | per-pipeline GPU time table; the new kernels show up as `mmq_<fmt>_i` |

## Memory

No new buffers: the kernel reads the same int8 activation scratch buffer as `mm_<fmt>_i` (batch x cols x 1.125 bytes, 16-byte aligned) and the weight
buffers as uploaded. Shared memory is at most 4 * (24 BM + 22 BN) bytes (11.8 KiB for 64 x 64), checked against the device limit by the planner.

## How to measure

1. `fllm-rs selftest-gpu --device vulkan --only linear` (fast pass; the new kernel needs a device with accelerated int8 dot product).
2. Prompt of 500+ tokens, with and without `FLLM_VK_MM=0`, `FLLM_VK_PROFILE=1` for the table; then `FLLM_VK_MM_BM/BN` (32/64/128, 64/128) to tune a device.
3. If the dot instruction itself misbehaves on a driver, rebuild with `FLLM_VK_NO_SPIRV_DOT=1` (or a Vulkan SDK >= 1.4.3xx) to compare.

## Not done

A cooperative-matrix kernel for formats other than the existing `mm_<fmt>_c` and a dequantize-to-f16 shared-memory variant were not added: `mm_<fmt>_c`
(f16 tiles, 16x16x16 cooperative matrices, preferred by the planner when the device reports them) and `mm_<fmt>_s` (f32) remain the paths for devices
without the int8 dot product. The batched attention, the DeltaNet kernels and the element-wise operations of the prompt path were read but not changed
(no measurement available): attention is a tiled online-softmax kernel, DeltaNet runs one workgroup per value head sequentially over the tokens.
