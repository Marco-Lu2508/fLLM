# Vulkan backend: integration

Level of verification: **L0** (compiles, clippy clean, unit-tested without a device; nothing here has ever run on a Vulkan driver).
`API.md` describes the core (instance, memory, pipelines, recorder), `shaders/vulkan/INTERFACE_QUANT.md` and `INTERFACE_OPS.md` the shaders and their
planners. This file describes what sits on top of them: `backend.rs` (+ `backend/`) and `stream.rs` (+ `stream/`).

## Files

| file | contents |
|---|---|
| `backend.rs` | `VulkanBackend` (`Backend`), `create(index)`, `native_quants()`, KV pools (`allocate_pool`) |
| `backend/shared.rs` | `Shared`: context, pipeline registry, blocking transfer queue, planner capabilities, switches (`set_exact_math`) |
| `backend/mem.rs` | `VkMem` (the `DeviceBuffer::handle`), `view_of`, `slice_table` (big weights), `pool_window` (KV layer windows) |
| `backend/upload.rs` | `upload_weight`: layouts of `weights.rs`, slicing, conversion of formats without a shader |
| `backend/convert.rs` | IQ4_XS -> IQ4_NL, f16 / f32 expansion (pure functions, tested) |
| `backend/experts.rs` | `upload_pointer_table` / `upload_bias_table` as expert arenas |
| `backend/tests.rs` | device-free consistency tests (plans vs. the compiled SPIR-V, registry completeness, float shader mirror) |
| `backend/hw_tests.rs` | `#[ignore]`d smoke tests for a real device |
| `stream.rs` | `VulkanStream` (`Stream`): one method per operation |
| `stream/exec.rs` | operand resolution, `run_plans` (the generic plan executor), submission, read-back |
| `stream/ring.rs` | scratch rings (device and host-visible) |
| `stream/linear.rs` | `quantized_linear`, `host_linear`, float-weight plans, barrier-free projections of the same input |
| `stream/fallback.rs` | CPU fallbacks (`moe_route`, sampling with temperature, attention with head width > 512) |
| `stream/mamba.rs` | Mamba-2 mixer, ready for the trait method proposed for the Backend trait |
| `shaders/vulkan/lin_float.comp`, `variants_float.list` | linear projection for F32 / F16 / BF16 weights |
| `shaders/vulkan/check_all.sh` | compiles and validates *every* shader of the registry |

## Operations

| `Stream` method | how |
|---|---|
| `quantized_linear` | `weights::plan_quantized_linear_with` per slice of the weight (variant `_s` / `_i` / `_c`, prepass) or `lin_f32/f16/bf16` for float weights |
| `host_linear` | the GGUF bytes are packed on the host, copied through the scratch rings and used for this one product (formats of `weights.rs` only) |
| `rms_norm`, `layer_norm`, `rope`, `swiglu`, `geglu`, `swiglu_oai`, `residual_add`, `split_q_gate`, `sigmoid_gate_mul`, `embedding_lookup`, `argmax_rows`, `gather_rows`, `scatter_add_rows`, `moe_topk`, `moe_combine` | `ops::plan_*` -> `run_plans` |
| `kv_cache_write`, `attention_paged` | `ops::attention::plan_*` on one layer window of each pool; two independent dispatches for keys and values |
| `moe_linear` | `moe_linear_<fmt>_arena` on the arena behind the table buffer |
| `delta_layer` (`supports_delta`) | `ops::recurrent::plan_delta_layer`; `mixed` is the caller's scratch buffer |
| `upload`, `download`, `copy_buffer`, `zero_buffer`, `synchronize` | recorded copies / fill; `download` and `synchronize` submit and wait |
| `sampling_logits` | greedy (temperature <= 0 or top-k 1) through `argmax_rows`, otherwise through the CPU backend |
| `moe_route` | CPU backend (the decoder routes with `moe_topk`) |
| `moe_expert_gemm` | `UnsupportedOperation` (the planar weights cannot be read by the CPU backend; `moe_linear` + `moe_combine` is the device path) |
| `graph_*`, `linear_f16_out`, `glu_f16`, `rope_kv_write_f16`, `quantized_linear_add` | defaults ("not done"): the caller takes the unfused path |
| Mamba layers | no trait method yet; `VulkanStream::mamba2_layer` is ready (see below) |

`capabilities()`: `supported_quants` = the 19 formats of `weights::supported_formats()` + F32 / F16 / BF16 + IQ4_XS (converted exactly).
`supports_moe` and `supports_device_sampling` are `false` like on CUDA (they describe `moe_route` / `moe_expert_gemm` / `sampling_logits`).

## Submissions

A `VulkanStream` owns one `Recorder`. Every operation records its dispatches; nothing is submitted until the host needs a result
(`synchronize`, `download`, `argmax_rows`, `sampling_logits`) or a command buffer would grow too large (4096 dispatches, 256 MiB of host scratch,
1 GiB of device scratch). A decode step is therefore one `vkQueueSubmit`. `VulkanStream::recorder_stats()` exposes the counters;
`backend/hw_tests.rs::a_decode_step_is_recorded_into_one_submission` checks it.

Host data of a dispatch (positions, block tables, slots, indices) and the sources of uploads go through a ring of three frames of host-visible memory,
scratch buffers (prepass results, split-attention partials, staged operands) through a ring of device memory. A frame is stamped with the timeline value
of the submission that used it and waited for before it is reused. Chunks of 64 MiB or more are not kept.

Projections that read the same input (q, k, v; gate, up) share the int8 prepass and skip the barrier between them when their outputs do not overlap
(`LinTrace` in `stream/linear.rs`); every other operation resets that state.

An operand whose view does not satisfy `minStorageBufferOffsetAlignment` (or 16 bytes) is copied to an aligned scratch buffer and copied back
afterwards if it is written. This is a safety net for odd `DeviceBuffer::slice` offsets; the decoder's own buffers are aligned.

## Weights

`upload_weight` (and `upload_weight_scaled`, which keeps the quantized bytes and applies `DeviceWeight::scale` in every shader):

* formats with a planar layout (`weights::supported_formats()`): `pack_weight` per slice; Q4ZP permutations become `DeviceWeight::perm`;
* F32 / F16 / BF16: row-major as in the file, read by `lin_float.comp`;
* IQ4_XS: converted exactly to IQ4_NL (the scale of each group of 32 is rounded to binary16);
* everything else (IQ1-3, Q8_K, FP8, shapes the planar layouts cannot hold): dequantized on the host and stored as Q5_K when the rows are a multiple of 256
  (not for FP8 / NVFP4), otherwise as F16 (F32 when a value does not fit). The first occurrence is logged.

A weight larger than `maxStorageBufferRange` is cut into slices of whole rows (`mem::slice_table`, the same function in the upload and in every dispatch);
all slices live in the one buffer of the `DeviceWeight`, each at an offset that is a multiple of 256 and of the device alignment, and are dispatched with
`row_base` / `out_stride`.

If a fast pipeline (`_i`, `_c`) cannot be created by the driver, the variant is switched off for the process and the slice is planned again with
the next best variant (a message is printed once).

## Mixture of experts

There are no buffer device addresses in the core, so `upload_pointer_table` copies the experts of a projection (device to device) into **one arena**
and returns a `uint32[]` of word offsets; the returned `DeviceBuffer`'s handle carries the arena, which `moe_linear` reads back. The decoder then makes the individual weights slices of the arena
(`alias_weights_into_table`), so the experts are not stored twice. An arena must fit one storage-buffer
binding; otherwise the call fails and the decoder falls back to the slower path. `upload_bias_table` does the same for the biases. Enabling
`bufferDeviceAddress` in the core would let `moe_linear_*_bda` replace the copy.

## KV cache

`allocate_kv_cache` allocates the two pools zero-filled; a pool larger than `maxBufferSize` is split into one allocation per group of whole layers. The
attention and write shaders always get **one layer window** of each pool (`mem::pool_window`, `ops::attention::KvLayout::rebased`), so ranges stay small
and pools beyond 4 GiB work. Attention with head widths above 512 goes through the CPU backend and needs unsplit pools.

## Switches

| variable | effect |
|---|---|
| `FLLM_VULKAN_EXACT=1` | only the f32 shaders (`_s`, `lin_*`): meets the FP32 bound of `selftest-gpu`; `backend::vulkan::backend::set_exact_math(bool)` overrides it at run time |
| `FLLM_VULKAN_NO_IDOT=1` | never plan the int8 dot-product variants |
| `FLLM_VULKAN_NO_COOP=1` | never plan the cooperative-matrix variants |
| `FLLM_VK_MM=0`, `FLLM_VK_MM_MIN_BATCH`, `FLLM_VK_MM_BM`, `FLLM_VK_MM_BN` | prompt-path kernel `mmq_<fmt>_i` (`mm_q.comp`): switch off, smallest batch, tile size; see `docs/changes/vulkan-prefill.md` |
| `FLLM_VULKAN_VERBOSE=1` | prints the planner capabilities (cooperative matrices, integer dot, subgroup size ...) when the device is opened and, once per shader, the variant, specialization constants and grid of the first `quantized_linear` that uses it; also the probe messages of `API.md` |
| `FLLM_ALLOW_SOFTWARE_GPU`, `FLLM_VULKAN_VALIDATE`, `FLLM_VK_CACHE_DIR`, `FLLM_VK_PIPELINE_CACHE` | see `API.md` |

The default is the fast variants (int8 activations, f16 tiles): their results differ from the CPU in the last digits. Use `FLLM_VULKAN_EXACT=1` when a greedy text
has to match the CPU token for token.

## Device index, errors, fallback

* `--device vulkan[:N]`: `create(N)` -> `VkContext::new(N)` counts the usable **native** Vulkan devices in driver order, which is the order of
  `device_info::list`. The translation of the global inventory index (which also contains CUDA and wgpu devices) is done by `devices::select_device`.
* No loader / no device / Vulkan < 1.2: `create` returns `Err` with the reason; `Decoder::set_device` then falls back to the wgpu backend when it is compiled in.
* Errors: `VkError` converts to `BackendError` (`OutOfMemory`, `DeviceNotFound`, `BufferError`, `DriverError`); shape problems are `InvalidShape`; formats
  or shapes without a shader are `UnsupportedQuant` / `UnsupportedOperation` (the caller falls back to the host).
* `--gpu-layers`: unchanged. `MemoryPlan` decides how many leading layers keep their feed-forward weights on the device (from `DeviceInfo::free_memory_bytes`);
  the other layers' feed-forward runs on the CPU between the device layers, and prompts may stream those weights through `host_linear`.

## The Mamba hook

`VulkanStream::mamba2_layer(shape, weights, runs, state_offset, projected, mixed, out)` plans and records the Mamba-2 mixer (`plan_mamba2_mixer`). After the
trait gains `MambaShape`, `MambaWeights`, `supports_mamba` and `mamba_layer` (proposed) the implementation is a conversion of the
arguments and one call..

## First run on hardware

1. `rust/engine/shaders/vulkan/check_all.sh` (does the local `glslc` build all modules?).
2. `FLLM_VULKAN_VALIDATE=1 cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::instance::smoke -- --ignored --nocapture --test-threads=1` (core).
3. `FLLM_VULKAN_VALIDATE=1 cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::backend::hw_tests -- --ignored --nocapture --test-threads=1`.
4. `fllm-rs devices`, then `FLLM_VULKAN_VALIDATE=1 FLLM_VULKAN_EXACT=1 fllm-rs selftest-gpu --device vulkan`.
5. A small model (Q4_0 / Q8_0) with `--device vulkan` against `--device cpu` (greedy), first with `FLLM_VULKAN_EXACT=1`, then without.
6. When something differs, isolate with `FLLM_VULKAN_NO_COOP=1`, `FLLM_VULKAN_NO_IDOT=1`, `FLLM_VULKAN_EXACT=1` (the shaders' CPU emulation is
   `shaders/vulkan/emu/run_kernels.sh`).

## Known weaknesses and untested assumptions

* Nothing ran on a device: driver behaviour, barriers (global and conservative), the staging rings, the descriptor paths and all shaders are untested.
* Prompt processing with float weights (`lin_float.comp`) re-reads each weight row once per eight tokens (not tiled).
* The experts of a mixture-of-experts layer live in an arena; the individual weights are slices of it (`alias_weights_into_table`).
* A weight, KV layer window or expert arena larger than `maxStorageBufferRange` is rejected (weights are sliced, the others are not).
* `moe_linear` is a matrix-vector kernel; prompt processing of MoE layers should group tokens per expert as the CPU decoder does.
* The cooperative-matrix path needs the 16x16x16 f16 shape in subgroup scope, `vulkanMemoryModel`, a pinnable subgroup size and `shaderFloat16`; otherwise it is not planned.
* Staged copies of unaligned operands are never needed for the decoder's own buffers, so that path has the least coverage.
* The expert arenas and the scratch copies of `host_linear` (a device and a host copy of one packed weight, kept until their frame has completed) are not in
  `MemoryPlan`'s estimate; on a card whose memory the plan fills almost completely they can run it out of memory.
* The KV layer windows and the whole DeltaNet state buffer of a run are bound as one range each; on a device with a 128 MiB `maxStorageBufferRange` this fails
  in `set_buffers` (no check beforehand). A KV pool that had to be split over several allocations cannot be used by `upload` / `download` / `copy_buffer` /
  `zero_buffer` or the CPU attention fallback (only by attention and `kv_cache_write`).
* Dropping a stream submits and waits for what it still has recorded (as CUDA streams do); the scratch rings of a dropped stream are freed.
* An operand that has to be staged (unaligned slice) costs a copy in and, if written, a copy out per operation, and ends the barrier-free sequence of projections.
* No timing anywhere: tuning constants (ring sizes, submission thresholds, attention targets) are guesses.
