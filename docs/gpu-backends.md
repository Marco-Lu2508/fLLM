# GPU & Accelerator Backends in fLLM

fLLM provides a unified, zero-allocation asynchronous backend abstraction for hardware acceleration across heterogeneous compute devices (CUDA, Metal, Vulkan, DirectX 12, and Host CPU).

This document outlines the architecture of the backend system, describes the operational lifecycle and zero-allocation decode stream, documents memory planning and fallback mechanics, and serves as an implementation guide for Phase 2 (CUDA Backend) and Phase 3 (Vulkan Backend).

> **Native backends of the multi-backend round.** Besides the CPU, wgpu and CUDA backends the tree has native Vulkan (`vulkan`), Metal (`metal`), AMD HIP (`hip`),
> Intel SYCL (`sycl`), NPU (`npu-*`) and TPU (`tpu-pjrt`) backends and new CPU kernels for ARM and AVX-512/AMX. All of them are test level **L0**: compiled and checked
> without the hardware, nothing has run on it. The table of hardware x format x level, the known gaps and the commands for verifying them on foreign hardware are in
> [backends.md](backends.md) and [backends-verify.md](backends-verify.md) (German); the section [Native backends](#native-backends-of-the-multi-backend-round) below lists features and prerequisites.
> Device numbering, `--device auto` and the extended self test are described in `device-selection.md` (package p15).

---

## 1. Architectural Overview

```
                      +-----------------------------------+
                      |      RustDecoder / Scheduler      |
                      +-----------------------------------+
                                        |
                         DeviceBuffer / DeviceStream
                                        |
                      +-----------------+-----------------+
                      | Backend Trait / Stream Trait      |
                      +-----------------+-----------------+
                                        |
        +------------------+------------+------------+------------------+
        |                  |                         |                  |
+---------------+  +---------------+         +---------------+  +---------------+
|  CPU Adapter  |  | wgpu Backend  |         | CUDA Backend  |  | Vulkan Backend|
|  (Reference)  |  | (Vulkan/Metal)|         |   (Phase 2)   |  |   (Phase 3)   |
+---------------+  +---------------+         +---------------+  +---------------+
```

The acceleration subsystem lives in `rust/engine/src/backend/` and consists of five core components:

1. **`Backend` Trait**: Represents a physical compute device. Manages device memory allocation (`DeviceBuffer`), pool management (`BufferPool`), weight lifecycle (`DeviceWeight`), and creates execution streams.
2. **`Stream` Trait**: Represents an asynchronous command stream / queue (CUDA Stream, Vulkan Command Buffer, wgpu CommandEncoder). Dispatches layer-level compute kernels without CPU-GPU synchronization. Execution only blocks when `synchronize()` is called or when transferring tensors back to host memory.
3. **`DeviceBuffer` & `BufferPool`**: Pre-allocated device memory arena. During the autoregressive token-generation decode phase, **zero dynamic allocations** take place on the hot path. Temporary activations reuse pooled scratch buffers sized during model initialization.
4. **`DeviceWeight` & Native Quant Formats**: Model weights reside in device memory in their native GGUF representation and are decoded inside the kernels. The wgpu backend decodes `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K` and `Q6_K`; every other format is dequantized to `F32` once at upload (logged once per format, with its size).
5. **`DeviceKvCache` & Paged Memory**: Device-resident paged KV blocks (F16, BF16, FP8, INT8, Q8_0, Q5, Q4, IQ4_NL, NVFP4 and FP32, keys and values independently; see [KV cache formats](#kv-cache-formats)) in the same layout as the CPU decoder (documented on `DeviceKvCache` and `kv_format.rs`), written by `Stream::kv_cache_write` and read by `Stream::attention_paged`. A decode step never copies the cache between host and device.

---

## 2. Operation Contracts: Required vs. Optional (CPU Fallback)

Backends implement layer-level and block-level operations rather than individual fine-grained vector math. This enables kernel fusion (e.g., fused RMSNorm + QKV projection) and minimizes command submission overhead.

### Required Operations (`Stream`)
Every backend **must** implement these primitives to participate in acceleration:
* `copy_dtod(src, dst, size)`: Device-to-device buffer copy (for residual streams and buffer transfers).
* `synchronize()`: Blocks until all pending recorded commands have completed execution on the hardware.

### Accelerable Layer-Level Operations (`Stream`)
If a backend does not support an operation or a specific tensor quantization format, it returns `Err(BackendError::UnsupportedOp(...))` or `Err(BackendError::FallbackRequested)`.

The engine handles this transparently via **One-Time Fallback Logging**:
* A one-time warning is emitted (`warn_fallback_once`) noting the layer index and missing kernel.
* State is synchronized, buffers downloaded to host memory, computed via reference CPU SIMD kernels, and transparently uploaded back to device memory to continue pipeline execution.

### Operations of the wgpu backend

| Operation | wgpu (WGSL) | Notes |
| :--- | :--- | :--- |
| `quantized_linear` | native, all formats above | bias added in the kernel; four batch rows per workgroup; any row count (2-D grid) |
| `embedding_lookup` | native | the decoder builds input rows on the host (scaling, image overrides, position tables) |
| `rms_norm` | native | |
| `rope` | native | NeoX and interleaved pairs; cos/sin tables uploaded once per stream (`Arc` identity) |
| `swiglu`, `geglu` | native | GELU uses the tanh approximation, like the CPU path |
| `residual_add` | native | |
| `kv_cache_write` | native | f16 and q8_0 (`d = amax/127`, round half away from zero); on CUDA and the CPU also every format of [KV cache formats](#kv-cache-formats) (wgpu: f16 and q8_0 only) |
| `attention_paged` | native, head width <= 512 | decode and prefill rows, GQA, scale, soft cap, sliding window, chunks, sinks, ALiBi, key width != value width; wider heads fall back to the host with a logged reason |
| `layer_norm`, `moe_*`, `sampling_logits` | host fallback | logged once per operation with the reason |

Fallbacks copy their operands to a CPU stream and back and are logged once per operation (`warn_fallback_reason`). Layers whose features the device decoder path does not implement run on the host between the device layers (section 3a); the wgpu backend has no LayerNorm / element-wise architecture kernels, so those layers (and models that need them everywhere) stay on the CPU. Recurrent mixers without a device kernel (DeltaNet, Mamba-1/2 and ShortConv on wgpu) run on the host as well.

Parameters reach the kernels through one per-stream arena buffer: each dispatch gets a 32-word header at a dynamic offset plus variable data (positions, slots, lengths, block tables, sinks) in the same buffer, written once per submit. Bind groups are cached per buffer window. Driver and validation errors are captured by an error handler and returned as `BackendError` from the next `synchronize`/`download`, never as a panic.

---

## 3. Recurrent and Hybrid State Strategy (DeltaNet / Mamba)

Models featuring linear recurrent components (Qwen3.5 Gated DeltaNet, Mamba-1/2, Nemotron-H, Jamba) require persistent recurrent hidden states across token decoding steps:

1. **State Layout**:
   * Recurrent states (e.g., `[batch, heads, d_k, d_v]` in Gated DeltaNet or `[batch, d_inner, d_state]` in Mamba) are allocated in VRAM as dedicated `DeviceBuffer` instances within `DeviceModelWeights`.
2. **State Updates**:
   * State updates occur in-place during the autoregressive step using `Stream::recurrent_state_update`.
   * For partial offloading (where recurrent layers are split between CPU and GPU), recurrent state tensors for GPU layers remain device-resident in VRAM, while states for CPU layers remain in the host `StateStore`.

### 3a. Layers without device kernels: per-layer host fallback

`--device cuda` starts for every architecture the CPU path knows. `RustDecoder::layer_host_reason` decides per layer whether the
device has what it needs; a layer that does not runs on the host (`host_layers`, the CPU reference code, with the host KV cache and the
host recurrent state): the residual stream is downloaded, the layer runs, the stream is uploaded again. Start-up logs
`N of M layers run on the host between the device layers (reasons)`. A model in which no layer can use the device, or whose norm
cannot be done by the device at all (grouped RMSNorm), is refused with the list of reasons (`--device cpu` then). `FLLM_HOST_LAYERS=all|3,4`
forces layers onto the host (the output must then match `--device cpu`); `FLLM_LAYER_CHECK=1` runs every device layer a second time on
the host from the same input and prints the deviation of the residual stream (max, mean, relative to the rms).

Device kernels (CUDA, `cuda/arch_ops.cu`, `mla.cu`, `mamba1.cu`, `mamba2.cu`, `shortconv.cu`; the HIP build has stubs, `Stream::supports_arch_ops` etc.
report them as missing so the host path runs):

| Layer type / feature | Device | Host fallback |
| :--- | :--- | :--- |
| LayerNorm (with bias), post-norm epsilon, parallel residual (GPT-NeoX, Falcon, Phi, Cohere), plain MLPs, GELU erf / ReLU / ReLU^2, QKV clipping, FFN-only layers (Nemotron-H), attention output gates (sigmoid / softplus, also from a separate projection: AFMoE, K2 Horizon) | yes | attention temperature scaling (Llama 4), xIELU, latent-space MoE experts (FFN only), QKV clipping together with a query gate |
| Routed value experts (K2 Horizon MoVA) | the values of the layer come from the host reference (download the normed input, upload the values); q, k, attention, output projection and FFN stay on the device | |
| MLA (DeepSeek V2/V3, Kimi), expanded form: the cache holds the rebuilt per-head keys and values like the CPU reference | yes | when the kv head count differs from the head count |
| Mamba-2 (Nemotron-H, Granite-Hybrid, Bamba): convolution, selective scan (state in registers, one block per sequence run and head) and gated norm; the state of all layers of a sequence lives in one device buffer (same bookkeeping as DeltaNet) | state 16/32/64/128, head width <= 256 | other shapes, and models that also contain DeltaNet |
| LFM2 ShortConv | yes | |
| Mamba-1 (Mamba, Falcon-Mamba, Jamba): convolution, the x_proj / dt_proj GEMMs, dt/B/C normalisation and the selective scan with one thread per channel (state in registers) | state 16/32 | other shapes, and models that also contain DeltaNet or Mamba-2 layers |

With `--gpu-layers N` the feed-forward weights of the layers after N stay in host memory. Gated FFNs and MoE experts stream over PCIe for large
batches; plain (non-gated) MLPs and the exact host FFN of the features above run on the CPU. Layers of the types in this section keep their
mixer weights on the device (their memory is counted by the planner like attention weights); layers that run completely on the host upload nothing.

---

## 4. VRAM Memory Planning & Offloading (`backend::memory`)

fLLM includes an automated VRAM memory planner (`MemoryPlan`) configured via:
* `--gpu-memory-utilization <float>`: Target fraction of free device VRAM to utilize (default: `0.90`).
* `--gpu-layers <int|auto>` / `-ngl <int>`: Explicit layer count to offload, or `auto` to maximize offloaded layers within the budget.

### Memory Planning Formula
$$\text{Budget} = \text{Free VRAM} \times \text{utilization}$$
$$\text{Required} = W_{\text{embeddings+head}} + (N_{\text{layers}} \times W_{\text{layer}}) + M_{\text{scratch}} + M_{\text{KV-cache}}$$

* **Model Weights ($W$)**: Computed directly from GGUF mapped tensor byte sizes.
* **Scratchpad ($M_{\text{scratch}}$)**: Sized for maximum continuous batch capacity:
  $$M_{\text{scratch}} = \text{max\_batched\_tokens} \times (d_{\text{model}} + d_{\text{intermediate}} + d_{\text{vocab}}) \times 4 \text{ bytes}$$
* **KV-Cache ($M_{\text{KV-cache}}$)**:
  $$M_{\text{KV-layer}} = \text{block\_size} \times N_{\text{kv\_heads}} \times (d_{\text{head}} + d_{\text{value}}) \times \text{bytes\_per\_elem}$$
  where `bytes_per_elem` follows the cache format (`KvFormat::bytes_per_slot`: 2 for FP16, 1 + 1/16 for Q8_0, 1 for FP8, 0.5 + scales for the 4-bit formats; keys and values may differ).


When memory is insufficient to hold all layers, layers $0 \dots N-1$ are offloaded to GPU (where high-throughput Prefill benefits most), and layers $N \dots L-1$ are evaluated on the host CPU.

---

## 5. Guide for Implementing Phase 2: CUDA Backend

The CUDA backend provides bare-metal NVIDIA acceleration using `cudarc` or native driver/runtime bindings.

### Step 1: Create Files
Create `rust/engine/src/backend/cuda.rs`.

### Step 2: Implement Traits
1. Implement `Backend` for `CudaBackend`:
   * `name(&self) -> "cuda"`
   * `allocate(size)` -> `cuMemAlloc` or `cudaMallocAsync`
   * `upload` -> `cudaMemcpyAsync(HostToDevice)`
   * `download` -> `cudaMemcpyAsync(DeviceToHost)`
   * `create_stream` -> `cudaStreamCreateWithFlags`
2. Implement `Stream` for `CudaStream`:
   * `linear` -> Dispatch GEMV (decoding) or GEMM (prefill) using cuBLAS / custom CUDA kernels (e.g. Marlin or CUTLASS).
   * `rms_norm`, `swiglu`, `residual_add` -> Fused elementwise CUDA kernels.
   * `paged_attention` -> FlashAttention / PagedAttention CUDA kernel.
   * `synchronize` -> `cudaStreamSynchronize`.

### Step 3: Register in Engine
1. In `rust/engine/Cargo.toml`:
   ```toml
   [features]
   cuda = ["dep:cudarc"]
   ```
2. In `rust/engine/src/backend/mod.rs`:
   ```rust
   #[cfg(feature = "cuda")]
   pub mod cuda;
   ```
   Add `create_backend("cuda", index)` factory arm.
3. In `rust/engine/src/devices.rs`:
   * Add CUDA probe in `list_devices()` using `nvml` or `cudaGetDeviceCount`.
   * Ensure `select_device("auto")` chooses CUDA with highest priority when available.

---

## 6. Guide for Implementing Phase 3: Vulkan Backend

> **Status:** the native Vulkan backend exists (feature `vulkan`, `ash`, own allocator instead of `gpu-allocator`, SPIR-V compiled by `build.rs`): see
> `rust/engine/src/backend/vulkan/{API.md,INTEGRATION.md}` and `rust/engine/shaders/vulkan/`. Level L0. The steps below are the original plan and differ in details.

The native Vulkan backend provides high-performance vendor-independent acceleration on Linux, Windows, and embedded hardware using `ash`.

### Step 1: Create Files
Create `rust/engine/src/backend/vulkan.rs`.

### Step 2: Implement Traits
1. Implement `Backend` for `VulkanBackend`:
   * `name(&self) -> "vulkan"`
   * Memory allocations via `gpu-allocator` or `vkAllocateMemory`.
   * `create_stream` -> Allocates `VkCommandBuffer` from a compute `VkCommandPool`.
2. Implement `Stream` for `VulkanStream`:
   * Pipelines for compute SPIR-V kernels (`gemm.spv`, `rmsnorm.spv`, `paged_attention.spv`).
   * Bind descriptors with push constants for tensor shapes and strides.
   * `vkCmdDispatch` without pipeline barriers between independent operations.
   * `synchronize` -> `vkQueueSubmit` with fence and `vkWaitForFences`.

### Step 3: Register in Engine
1. In `rust/engine/Cargo.toml`:
   ```toml
   [features]
   vulkan = ["dep:ash", "dep:gpu-allocator"]
   ```
2. In `rust/engine/src/backend/mod.rs`:
   Add `create_backend("vulkan", index)` factory arm.

---

## 7. Verification & Quality Gates

Check a backend change in this order; each step assumes the previous one passed.

1. **Build and lint**, with and without the GPU feature (and for aarch64 when CPU code changed):
   ```bash
   cargo check -p fllm-engine
   cargo check -p fllm-engine --features rust-gpu
   cargo clippy -p fllm-engine --all-targets -- -D warnings
   cargo clippy -p fllm-engine --all-targets --features rust-gpu -- -D warnings
   ```
2. **Shader validation without a device**: `cargo test -p fllm-engine --features rust-gpu shaders_parse_and_validate` parses and validates every WGSL module with naga.
3. **Device inventory**: `fllm-rs devices` must list the adapter. Its capability lines come from runtime detection (CPU kernel tier and SIMD extensions) and from the kernel set of this build.
4. **CPU reference**: `fllm-rs selftest-cpu`. `selftest-gpu` compares against the CPU backend, so the CPU kernels must pass first.
5. **Every GPU operation**: `fllm-rs selftest-gpu [--device vulkan]` runs each backend operation on random data on the device and on the CPU backend and compares both with an f64 reference: linear for every weight format (13 rows, batches 1/2/5, odd widths, a bias, more than 65535 rows), embedding, RMSNorm, RoPE, SwiGLU/GeGLU, residual add, KV-cache writes (f16/q8, head widths 16/32/64/80/128, layers starting on odd 16-bit words, untouched slots must stay zero) and paged attention (decode and prefill, f16 and q8 caches, head widths 16/32/64/80/128, GQA and MHA, window, chunks, soft cap, sinks, ALiBi, key width != value width). Each line prints the largest error of device and CPU relative to the element's error scale, the tolerance (a worst-case f32 rounding bound plus WGSL's accuracy for `exp`/`sqrt`/division and f16 conversion), the device time and ok/FAIL; the exit code is non-zero when any check fails. The extended harness (`--only GROUP`, `--quants a,b`, `--list`, `--strict`, `SKIPPED (unsupported)` for operations a backend does not offer, one JSON entry per check) is described in `device-selection.md` (package p15).
6. **A model**: only then compare a short greedy generation (or the first-token logits) of `--device cpu` and `--device vulkan`.

Software adapters (lavapipe, WARP) are refused by default because they are slower than the CPU path; set `FLLM_ALLOW_SOFTWARE_GPU=1` to use one for steps 3 to 6 on machines without a GPU:
```bash
FLLM_ALLOW_SOFTWARE_GPU=1 fllm-rs selftest-gpu --device vulkan
```

---

## Native backends of the multi-backend round

All entries are level **L0** (compiled, nothing run on the hardware). Features are optional and not part of the default build; prerequisites are needed at build or run time as listed.
Per-format and per-operation status: [backends.md](backends.md); verification: [backends-verify.md](backends-verify.md).

| Backend | Cargo feature | `--device` | Needed | Code |
| --- | --- | --- | --- | --- |
| Vulkan (native) | `vulkan` | `vulkan[:N]` (`vulkan-native:N` with p15) | Vulkan driver; `glslc` at build time | `src/backend/vulkan/`, `shaders/vulkan/` |
| Metal (native, Apple only) | `metal` | `metal[:N]` (`metal-native:N` with p15) | Apple SDK to link | `src/backend/metal/`, `metal/` |
| AMD ROCm/HIP | `hip` | `hip[:N]` | ROCm (`FLLM_HIP=1`, `ROCM_PATH`, `FLLM_HIP_ARCHS`) | `hip/`, `cuda/*.cu`, `src/backend/hip/` |
| Intel GPU (SYCL) | `sycl` | `sycl[:N]` | oneAPI DPC++ and Level Zero; `build.rs` patch of p09 | `sycl/`, `src/backend/sycl/` |
| Intel NPU | `npu-openvino` | `npu` | OpenVINO C library at run time | `src/backend/npu/` |
| AMD XDNA, Qualcomm Hexagon, Apple ANE | `npu-xdna`, `npu-qnn`, `npu-ane` | probes only (`examples/npu_probe.rs`) | vendor stack at run time | `src/backend/npu/{xdna,hexagon,ane}*` |
| Google TPU | `tpu-pjrt` | `tpu` | `libtpu.so` at run time | `src/backend/tpu/` |
| ARM CPU | none (`arm-simd` has no effect) | `cpu` | AArch64 | `src/cpu_arm/`, `docs/cpu-arm.md` (p13) |
| x86 AVX-512 / AMX | `avx512` | `cpu` | AVX-512 CPU; AMX needs Linux 5.16 or newer | `src/cpu_avx512/`, `docs/cpu-avx512-amx.md` (p14) |

## CUDA backend (`--features cuda`)

NVIDIA GPUs run through `rust/engine/src/backend/cuda.rs`, which drives the kernels in
`rust/engine/cuda/` (`nvcc`, compiled by `build.rs`). The CUDA runtime is linked statically, so a
binary needs only the NVIDIA driver at run time; without it the CUDA devices are simply not listed
and fLLM keeps using the CPU or Vulkan.

Build and run:

```sh
FLLM_CUDA_ARCHS=120 cargo build --release --features cuda -p fllm-engine   # one architecture: fast
cargo build --release --features "rust-gpu,cuda" -p fllm-engine           # all architectures
fllm-rs devices                                  # lists "CUDA, sm_120, 46 SMs"
fllm-rs selftest-gpu --device cuda               # every operation against the CPU reference
fllm-rs serve --model MODEL.gguf --device cuda   # or --device auto (CUDA is preferred)
```

`CUDA_HOME` selects the toolkit (default `/usr/local/cuda`), `FLLM_CUDA_ARCHS` the architectures
(default `61;70;75;80;86;89;90;120`, plus PTX of the newest). Docker users need no toolkit at all:
`docker/Dockerfile.cuda` builds the image and `docker run --gpus all` supplies the driver.

| Operation | Status |
| --- | --- |
| `quantized_linear`, `embedding_lookup` | native for `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS MXFP4 NVFP4`; only unusual shapes fall back to a one-time conversion (logged) |
| `rms_norm`, `layer_norm`, `rope`, `swiglu`, `geglu`, `residual_add` | native |
| `kv_cache_write`, `attention_paged` | native (f16 and q8_0 cache; CUDA also all other [KV cache formats](#kv-cache-formats), GQA, softcap, window, chunk, sinks, ALiBi; head width up to 512, wider heads fall back to the CPU) |
| `moe_route`, `moe_expert_gemm`, `sampling_logits` | host fallback |

### Kernels

| Path | Used for | Notes |
| --- | --- | --- |
| int8 `dp4a` (`mmvq.cu`) | 1-4 tokens, Q4_K / Q5_K / Q6_K / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / IQ4_NL | activations rounded to int8 (about 1 % per output); ~500 GB/s |
| FP32 GEMV (`gemv2.cu`) | 5-8 tokens and `FLLM_CUDA_NO_MMVQ*` | weights decoded once per chunk |
| tensor-core GEMM (`qgemm4k.cu`, `qgemm6k.cu`) | 2+ tokens, Q4_K / Q6_K | fp16 tensor cores, weights dequantized in registers, one pipeline stage per super-block (long contiguous reads), split-K for small matrices |
| fused-decode GEMM (`tc_gemm.cu`) | other formats, 16+ tokens | |
| generic FP32 kernels (`linear.cu`) | everything else, `FLLM_CUDA_EXACT=1` | exactly the tolerances of `selftest-gpu` |
| NVFP4 (`nvfp4.cu`) | NVFP4 weights on Blackwell (sm_12x), up to 64 tokens | FP4 tensor cores (`mma ... kind::mxf4nvf4`), activations quantized to NVFP4 on the fly |
| NVFP4 (`nvfp4_cutlass.cu`) | NVFP4 weights, more than 64 tokens | CUTLASS block-scaled GEMM (TMA, warp specialization, 128x128x256 tiles). Its epilogue adds the output projections to the residual stream (FP32) and leaves FP16 for gate and up, whose consumer (the activation quantization of the down projection) reads FP16; needs the CUTLASS headers at build time, see below |
| attention (`attention3.cu`, `attention2.cu`, `attention.cu`) | prefill (tensor cores), decode (split-KV), everything else | f16 cache; q8_0 cache and unusual heads use the generic kernel |

Environment switches: `FLLM_CUDA_EXACT=1` (FP32 only), `FLLM_CUDA_NO_FLASHINFER` (prompt attention through the built-in kernel), `FLLM_CUDA_NO_CUTLASS` (NVFP4 prompts through the built-in kernel), `FLLM_CUDA_NO_TC`, `FLLM_CUDA_NO_QGEMM`,
`FLLM_CUDA_NO_MMVQ`, `FLLM_CUDA_NO_MMVQ1` (single-token Q4_K/Q5_K/Q6_K), `FLLM_CUDA_NO_MMVQ_LEG` (32-value formats), `FLLM_CUDA_NO_FLASH`,
`FLLM_CUDA_OLD_GEMV`, `FLLM_CUDA_OLD_ATTN`, `FLLM_CUDA_NO_AUTOTUNE`, `FLLM_CUDA_TUNE_LOG`, `FLLM_CUDA_NO_GLU_FUSE`, `FLLM_CUDA_NO_X16_REUSE`, `FLLM_NO_FUSED_ADD` (output projections added to the residual by a separate kernel), `FLLM_NO_FUSED_QKV` (Marlin q/k/v), `FLLM_NO_F16_FFN` (Marlin and NVFP4 gate/up) and `FLLM_NO_F16_NVFP4` (NVFP4 gate/up) keep FP32 results, `FLLM_FP4_TILE=1..4` (force an NVFP4 prompt tile for tests),
`FLLM_HEAD_FULL=1` (keep a BF16/F16 output projection at full precision instead of Q6_K), `FLLM_FUSE=0` and `FLLM_FUSE_ADD_NORM` /
`FLLM_FUSE_NORM_QUANT` / `FLLM_FUSE_ROPE_KV` / `FLLM_FUSE_GLU_Q8` / `FLLM_FUSE_HEAD_COPY` `=0|1` (the fused decode kernels, see "Fewer kernels per decode token").

The server sizes the KV pool to the free device memory, so `--max-seqs 32` works on a 12 GB card
(sequences share the pool; the scheduler preempts when it runs out).

### Models larger than the card

`--gpu-layers auto` (the default) keeps attention and the KV cache of every layer on the device and leaves the feed-forward
weights of the layers that do not fit in host memory. `--gpu-layers N` fixes the number of layers that are completely on the
device. The experts of a mixture-of-experts layer count as feed-forward weights too: the leading layers keep their experts on the device (decode and prompts run there with routing on the host), the others run as described below.

* **Decode** (one token at a time) runs the host-resident feed-forward on the CPU between the device layers: Qwen3-30B-A3B in
  Q4_K_M (18.5 GB) decodes at ~44 tokens/s on a 12 GB card with 21 of its 48 layers resident (llama.cpp with all experts on the CPU: 24);
  a card that holds the whole model keeps every layer on the device.
* **Prompt processing** (32 or more tokens in a batch) streams the host-resident weights to the device instead: every projection
  is copied over PCIe (the GGUF file mapping is page-locked once, ~28 GB/s on PCIe 5.0 x8) into a staging buffer and multiplied
  there, Q6_K and Q8_0 are converted to their device layout on the way. For mixture-of-experts layers the routing stays on the
  host, the tokens of every expert are gathered and the results accumulated with a weighted scatter-add on the device.
  Batches default to 2048 tokens in this mode so that a weight transfer serves more tokens. Qwen3-30B-A3B prefill: 130 -> 1300
  tokens/s for a 1900-token prompt, 1650 with the resident layers (llama.cpp: 720); Llama-3.1-8B Q4_K_M with only 8 device layers: 57 -> 2020 tokens/s.
  `FLLM_NO_HOST_STREAM=1` keeps the CPU path, `FLLM_HOST_STREAM_MIN=N` changes the batch threshold, `FLLM_NO_HOST_PIN=1` copies
  from pageable memory (half the speed). Streaming covers Q4_K, Q5_K, Q6_K and Q8_0 weights without bias and SwiGLU/GeGLU
  experts; everything else stays on the CPU.

Formats without a CUDA decoder (Q2_K, Q3_K, the IQ family except IQ4_NL/IQ4_XS) are re-quantized to Q5_K when they are uploaded.

### Older GPUs

The tensor-core kernels need compute capability 8.0 (Ampere) or newer. The same binary also contains code for Pascal
(6.x), Volta (7.0) and Turing (7.5): there the dispatcher uses the FP32/`dp4a` kernels (single-token decode runs at
the same bandwidth-bound speed, prompt processing is much slower). `FLLM_CUDA_FORCE_CC=61` forces that path for testing.

### AMD GPUs (ROCm / HIP)

The same kernel sources build for AMD with `hipcc` (`cuda/hip_compat.h` maps the runtime calls and intrinsics; `__dp4a`
becomes `v_dot4_i32_i8` / `v_dot4_i32_iu8`):

```sh
FLLM_HIP=1 FLLM_HIP_ARCHS="gfx1030;gfx1100;gfx1201;gfx942" cargo build --release --features cuda -p fllm-engine
```

`FLLM_HIP_ARCHS` lists the offload targets (default: gfx908, gfx90a, gfx942, gfx1030, gfx1100-1103, gfx1150/1151, gfx1200/1201),
`ROCM_PATH` the install (default `/opt/rocm`). The binary links `libamdhip64` dynamically, so it needs a ROCm runtime. What runs on
AMD today: all integer-dot-product decode kernels (Q4_0 ... Q8_0, Q4_K/Q5_K/Q6_K, IQ4, MXFP4), the FP32 GEMM, the plain and
split-KV attention kernels, MoE and the paged KV cache. The NVIDIA tensor-core kernels (prompt GEMMs, FlashInfer/CUTLASS paths,
NVFP4) are compiled out (`fllm_cc()` reports 10, `hip_stubs.cu` declines the calls), so prompt processing uses the generic decode-and-multiply GEMM of
`tc_gemm.cu` on the matrix cores of RDNA3/4 and CDNA (rocWMMA, 16x16x16 fragments; RDNA2 and older use the FP32 GEMM). The CI job `hip-build` compiles this configuration.

### NVFP4 models

Checkpoints produced with NVIDIA ModelOpt (`quant_algo: NVFP4`, e.g. `nvidia/Llama-3.1-8B-Instruct-NVFP4`)
run through GGUF type 40 (llama.cpp layout: 4 UE4M3 scales + 32 nibble bytes per 64 values; the per-tensor
factor `weight_scale_2` is the F32 tensor `<name>.scale`). Convert with llama.cpp's `convert_hf_to_gguf.py`
until `fllm convert` handles ModelOpt directly. On sm_12x GPUs the FP4 tensor cores are used; elsewhere the
weights are expanded to F16 at load. The CPU decodes them to F32 per row (slow).

Device layout: the weights are stored as CUTLASS expects them (row-major packed E2M1, block scales in the swizzled
atom layout), so one copy serves the small-batch kernel (`nvfp4.cu`) and the CUTLASS GEMM that processes prompts.
The CUTLASS headers are found in `third_party/cutlass` or in `FLLM_CUTLASS_DIR`
(`git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass`; the Dockerfile and CI do
this); without them, or with `FLLM_CUDA_NO_CUTLASS=1`, the built-in kernel handles prompts too, about 25 % slower. The
GEMM configuration follows vLLM and SGLang (see [THIRD_PARTY.md](../THIRD_PARTY.md)).

### Measured (RTX PRO 3000 Blackwell Laptop 12 GB, Llama-3.1-8B-Instruct)

Greedy decoding through the OpenAI-compatible server, one request at a time (`~/bench/stream_bench.py`-style client:
prompts of 8 / 356 / 1864 tokens, 128 generated tokens, median of four runs). llama.cpp is the `ghcr.io/ggml-org/llama.cpp:server-cuda`
image on the same card with `-ngl 99`, measured with the same client. The card runs power-limited (about 110 W, ~1.95 GHz
under load), so the numbers are sustained rather than burst figures.

| Format | decode, context 8 / 356 / 1864 (tok/s) fLLM | llama.cpp | prefill, 356 / 1864 tokens (tok/s) fLLM | llama.cpp |
| --- | --- | --- | --- | --- |
| Q4_K_M | 96 / 96 / 92 | 85 / 84 / 77 | 3050 / 3410 | 2785 / 3145 |
| Q5_K_M | 85 / 85 / 82 | 75 / 75 / 71 | 2855 / 3210 | 2625 / 3005 |
| Q8_0 | 67 / 67 / 65 | 60 / 60 / 57 | 2855 / 3265 | 2810 / 3165 |
| IQ4_XS | 96 / 94 / 91 | 97 / 97 / 91 | 3040 / 3305 | 2965 / 3340 |
| NVFP4 (ModelOpt) | 102 / 101 / 97 | - | 9180 / 10070 | - |

Mixture-of-experts models (same client; the experts of the leading layers sit on the device, see "Models larger than the card"):

| Model | decode tok/s fLLM | llama.cpp | prefill 342-359 / 1800-1870 tokens fLLM | llama.cpp |
| --- | --- | --- | --- | --- |
| OLMoE-1B-7B Q4_K_M (4.2 GB, fully on the device) | 434 / 438 / 377 | 365 / 337 (long prompt) | 4800 / 5750 | 5190 / 6320 |
| Qwen3-30B-A3B Q4_K_M (18.5 GB, 12 GB card, 21 of 48 layers resident) | 44 / 44 / 42 | 24 / 24 / 24 (`--n-cpu-moe 99`) | 560 / 1630 | 414 / 718 |
| gpt-oss-20b MXFP4 (12.1 GB, 12 GB card, 15 of 24 layers resident) | 44 / 44 / 45 | 43-45 (`--n-cpu-moe 8`) | 1025 / 2990 | 1160 / 1910 |

Speculative decoding by prompt lookup (`--speculative-lookup 4`) works on the GPU as well: a prompt that makes the model repeat text (87 of 96 drafted tokens accepted) decodes at 190-210 tokens/s instead of 95 on the same Llama-3.1-8B Q4_K_M; text without repetition is unaffected.

Concurrent requests (Q4_K_M, 200-token prompts, 128 generated tokens each, total tok/s): 1 / 2 / 4 / 8 / 16 / 32 users =
97 / 179 / 297 / 498 / 866 / 1210.

How the numbers come about:

* **Decode** is bandwidth-bound (the card reads ~500 GB/s of its 672 GB/s peak). Single tokens use `dp4a` kernels on int8
  activations (one quantisation shared by consecutive projections with the same input), the same trade llama.cpp makes.
* **Prefill** is limited by the power cap: the FP16 tensor-core kernels reach 72-78 TFLOPS in isolation, ~58 sustained.
  The tile shape of every large GEMM is picked per (shape, batch) by a small autotuner (`FLLM_CUDA_TUNE_LOG=1` prints its
  timings, `FLLM_CUDA_NO_AUTOTUNE=1` disables it). The down projection reads gate and up directly (the GLU is folded
  into the activation conversion).

## Decode with many users (vLLM's sampler and graphs)

- **Greedy sampling on the device.** A decode step in which every sequence just takes the largest logit (temperature 0, no penalties,
  logit bias, logprobs or grammar) runs one argmax kernel per row and downloads the token ids instead of `rows x vocabulary` logits
  (vLLM: `Sampler.greedy_sample`, an argmax on the GPU). Qwen2.5-1.5B GPTQ, 32 concurrent users: 2727 -> 5442 tok/s (vLLM 5095), most
  of it from the vectorized argmax and from not copying the logits row once per sequence. `FLLM_CUDA_EXACT`, penalties, grammar or
  logprobs use the ordinary path.
- **CUDA graphs** (on by default where the model and the device allow it; `FLLM_CUDA_GRAPH=0|off|false|no` switches them off,
  `1|on|true` asks for them explicitly, any other value is the default). A decode step of a dense model (plain attention, every weight
  on the device) is recorded once per padded batch size 1, 2, 4 .. 64 and replayed: the per-step arrays (positions, cache slots, block
  tables, lengths) go through pinned mirrors that copy nodes of the graph upload, the padding rows read and write the last block of the
  KV pool, which the scheduler never hands out, and attention sizes its splits for the whole context. The text is identical to the
  ordinary path (3 models, 1 to 20 concurrent users). On a slow host CPU or a fast GPU the replay is the difference between waiting
  for ~500 to 700 kernel launches per token and not: RTX PRO 6000 Blackwell, one user, greedy, 128 tokens: Qwen2.5-3B AWQ 205 -> 229
  tok/s, Llama-3.2-3B Q4_K_M 316 -> 399, Llama-3.1-8B Q4_K_M 190 -> 219, Llama-3.1-8B NVFP4 194 -> 227. (An earlier revision of this
  switch looked only at whether the variable was *set*, so `FLLM_CUDA_GRAPH=0` turned the graphs on.) On the first decode step the
  engine prints `[fllm-rs] CUDA graphs for decode steps: on|off (...)` to stderr, and `CUDA graph recorded for N rows` per size.
  A failed capture falls back to the ordinary path for the rest of the run.

  Not eligible (these decode without graphs): recurrent and convolutional mixers (Qwen3.5 / DeltaNet hybrids, Mamba 1 and 2, LFM
  short convolutions: they stage a per-step table of state slots that is not one of the four fixed arrays of a graph), multi-head latent
  attention (MLA; its projections are not the q/k/v weights the check requires and its assemble kernels were never recorded), mixtures
  of experts (the routed activations are downloaded to the host or the experts are streamed in), models with layers or feed-forward
  blocks on the host (`--gpu-layers` below the layer count), parallel-residual models (Cohere), QKV clipping, MTP speculation, image
  or embedding steps, and steps of more than 64 rows or with prompt rows. Not a criterion: the KV cache format (f16, q8_0 and the
  quantized formats read the same staged slot, table and length arrays and size their split-KV launch for the longest context while
  recording), sliding-window and soft-capped attention, biases, QK norms and gated attention outputs. Quantized KV caches take part
  in the graph, but only the f16 cache has the fused rotary-and-cache-write kernels (see "Fewer kernels per decode token").

## Fewer kernels per decode token

A decode step is a chain of small kernels, 500 to 700 launches per generated token. CUDA graphs (above) remove the launch cost from the
host, but the GPU still pays a gap of about a microsecond between two dependent kernels and every tiny kernel reads and writes the
activations again. vLLM, SGLang and FlashInfer fuse the obvious chains (`fused_add_rms_norm`, `rotary_embedding` + `reshape_and_cache`,
`silu_and_mul`); fLLM has four fusions of that kind, each with its own switch (`1` or `0`; a switch that is not set follows `FLLM_FUSE`, which is `1`
unless set, so `FLLM_FUSE=0` turns all of them off and `FLLM_FUSE=0 FLLM_FUSE_ROPE_KV=1` runs one alone; `FLLM_FUSE_HEAD_COPY` below belongs to the same
family). The kernels are in `rust/engine/cuda/fused.cu`. They repeat the arithmetic of the
kernels they replace expression by expression (same residual update, same order of the sum of squares, same quantizers, one shared
`rope_pair`), so their results are bit-identical to the unfused sequence and all of them are on by default. This is an argument from
the code and is checked by `selftest-gpu --only fuse`, which compares every fused kernel with the unfused sequence on the device bit for
bit; the kernels were compiled for sm_75 and sm_120 but have not run on a GPU yet (level L0 until that selftest has been run).

| Switch | What it replaces | Where it applies |
| --- | --- | --- |
| `FLLM_FUSE_ADD_NORM` | `residual_add` (or the half-to-float accumulate kernel of the Marlin GEMMs) followed by `rms_norm`: one kernel per boundary (attention output to feed-forward norm, feed-forward output to the next layer's attention norm, last layer to the final norm) | RMS-norm models without post norms and parallel residual; every weight format (f32 branch), Marlin 4-bit as `quantized_linear_add_norm` (the GEMM, then one kernel reading its FP16 result) |
| `FLLM_FUSE_NORM_QUANT` | the activation conversion of the GEMM that follows the norm: NVFP4 blocks (`nvfp4_quantize_kernel`), int8 blocks of the K-quant GEMVs (`quantize_x8_kernel`), FP16 of the Marlin GEMMs (`f32_to_f16_kernel`) | rides on `FLLM_FUSE_ADD_NORM` (decode batches up to 64 for NVFP4, up to the `mmvq` limit for K-quants; the first GEMM after the norm finds its input prepared and skips its own conversion) |
| `FLLM_FUSE_ROPE_KV` | `rope` + two `kv_cache_write` launches (k and v) | f16 KV cache, FP32 projections (the Marlin path has had its own fused kernel for FP16 projections, `rope_kv_f16_kernel`, before); not with QK norm after the rotation (Gemma-style) and not for quantized caches |
| `FLLM_FUSE_GLU_Q8` | the gated activation kernel in front of a K-quant GEMV: the int8 quantization computes `silu(gate) * up` itself (NVFP4 and Marlin already did this) | Q4_K / Q5_K / Q6_K down projections, same batches as `mmvq` |
| `FLLM_FUSE_HEAD_COPY` (host side) | the copy of the final-norm rows into the head GEMM's input buffer; also lets the final norm prepare the head's int8 activations | every step whose logit rows are all rows in order (decode, graph) |

Launches per layer, counted from the code (decode, one user; attention counted as one kernel; estimates, the nsys totals of the
Colab run are about 20 % lower than these counts for reasons that are not yet understood, so only the ratios are meant):

| Model class | before | after |
| --- | --- | --- |
| Marlin AWQ (Qwen2.5-3B): norm, convert, q, k, v, rope+cache, attention, convert, o, accumulate, norm, convert, gate, up, convert+glu, down, accumulate | 17 | 13 (-24 %) |
| K-quant GGUF (Llama-3.2-3B / 3.1-8B Q4_K_M): norm, quantize, q, k, v, rope, 2 cache writes, attention, quantize, o, residual, norm, quantize, gate, up, glu, quantize, down, residual | 20 | 13 (-35 %) |
| NVFP4 (Llama-3.1-8B): the same chain with split-K reduce kernels after the narrow GEMMs | 24 | 18 (-25 %) |

Applied to the measured totals this would be about 497 -> 380 launches per token for the AWQ model and 620 -> 465 for NVFP4. At
roughly one microsecond of idle time per launch inside a graph that is 0.1 to 0.2 ms of a 4.4 ms token (2 to 5 %); this is an
estimate, not a measurement. The fusions do not change what the big GEMMs do: Marlin is 76 % of the AWQ model's kernel time and
`nvfp4_gemm_kernel` 70 % of the NVFP4 model's.

### The Q6_K output projection is at the memory bandwidth

The head GEMV (`mmvq_q6k_kernel`) was measured at 181 us per token for Qwen2.5-3B (vocabulary 151936, width 2048) and 304 us for
Llama-3.1-8B (128256 x 4096) on the RTX PRO 6000 Blackwell (1792 GB/s by specification). Bytes per weight: 210/256 = 0.820 in the GGUF
file, 224/256 = 0.875 in the device layout (the Q6_K blocks are padded to 224 bytes so that every 16-byte load is aligned). Streamed bytes
over time:

| | weights | device bytes | time | device bytes / time | GGUF bytes / time |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | 311.2 M | 272.3 MB | 181 us | 1.50 TB/s (84 % of 1792 GB/s) | 1.41 TB/s (79 %) |
| Llama-3.1-8B | 525.3 M | 459.7 MB | 304 us | 1.51 TB/s (84 %) | 1.42 TB/s (79 %) |

A streaming read kernel on GDDR7 reaches about 90 to 93 % of the specified bandwidth, so the kernel is within 6 to 9 % (11 to 27 us, 0.3 to
0.6 % of a token) of what any rewrite could reach; it already loads 16 bytes per instruction, reads two rows per warp and keeps the
activations in registers. No change was made. The one lever with real size is the padding (6.7 % of the bytes), which needs a different
device layout of the Q6_K blocks and a 2-byte-aligned load path; that costs more in the load instructions than it saves in bytes.

Left out on purpose: fusing the RMS norm of QK-norm models (Qwen3) with the rotation and the cache write (a per-head kernel, as in
SGLang's `fused_qk_norm_rope`), the attention output into the quantization of the output projection, and programmatic dependent launch
between consecutive GEMMs (hides the prologue of each kernel behind the tail of the previous one; needs `griddepcontrol` in the Marlin
and FP4 kernels).

---

## KV cache formats

`--kv-cache-type TYPE` (alias `--kv-cache-dtype`) sets the format of the paged KV cache; `--cache-type-k` and
`--cache-type-v` (llama.cpp's `-ctk` / `-ctv`) set the keys and the values separately. `auto` is `f16`. The formats
exist on the CPU and on CUDA (Turing, sm_75, and newer); the wgpu backends (Vulkan, DX12, Metal) keep `f16` and
`q8_0` and refuse the others at start-up with a clear message. The layout lives in `rust/engine/src/kv_format.rs`
(CPU encoder/decoder and the numbers the kernels use) and `rust/engine/cuda/kv_quant.cu` (CUDA).

Layout of one layer of a pool (16-bit words): the code plane (one entry per value), for the 5-bit formats a plane with
the fifth bit, a plane of scales (f16 per group; nvfp4: one UE4M3 byte per 16 values) and, for the `_1` formats, a plane
of minima. Scales are never global: they belong to a group of 32 values (the whole head when its width is not a
multiple of 32), to a head of one token (`int8`, `fp8_e4m3`: `scale = amax / 127` or `amax / 448`) or to 16 values
(`nvfp4`). `fp8_e5m2`, `bf16`, `f16` and `f32` need no scale. The encoders round to nearest (ties to even for FP8 and
bf16, half away from zero for int8 / q8_0) and the CUDA encoders produce the same words as the Rust ones.

Measured on an RTX PRO 3000 Blackwell Laptop (12 GB), Llama-3.1-8B-Instruct Q4_K_M, context 8192. "Blocks" is the
size of the KV pool the planner fits into the same 2.72 GB (16 tokens per block). Accuracy: 30 prompts of about 1300
tokens from technical text, one token each; "top-1" is the number of prompts whose most likely token equals the
FP16-KV one and "dlogp" the mean |log-probability difference| of that token. Needle: three codes hidden at 15 / 50 /
85 % of a prompt of about 6000 tokens. Speed: decode with 1 user (160 tokens) and prefill of 1500 tokens relative to
the FP16 cache (single runs, noise about 5 %).

| Format | Bits/value | KV memory vs FP16 | CPU | CUDA (sm_75+) | pool blocks | top-1 same | mean abs dlogp | needle | free text 800 | decode 1 user (short / 2k ctx) | prefill 1.5k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f16 | 16 | 100 % | yes | yes | 1299 | 30/30 | 0.000 | 3/3 | ok | 100 % / 100 % | 100 % |
| bf16 | 16 | 100 % | yes | yes | 1299 | 30/30 | 0.008 | 3/3 | ok | 98 % / 81 % | 102 % |
| fp8_e4m3 | 8.125 | 51 % | yes | yes | 2558 | 27/30 | 0.063 | 3/3 | ok | 97 % / 79 % | 101 % |
| fp8_e5m2 | 8 | 50 % | yes | yes | 2598 | 26/30 | 0.104 | 3/3 | ok | 96 % / 80 % | 101 % |
| int8 | 8.125 | 51 % | yes | yes | 2558 | 28/30 | 0.018 | 3/3 | ok | 96 % / 79 % | 100 % |
| q8_0 | 8.5 | 53 % | yes | yes | 2445 | 27/30 | 0.019 | 3/3 | ok | 98 % / 99 % | 98 % |
| q5_0 | 5.5 | 34 % | yes | yes | 3779 | 23/30 | 0.108 | 3/3 | ok | 95 % / 74 % | 99 % |
| q5_1 | 6 | 38 % | yes | yes | 3464 | 28/30 | 0.074 | 3/3 | ok | 93 % / 67 % | 98 % |
| q4_0 | 4.5 | 28 % | yes | yes | 4619 | 25/30 | 0.228 | 3/3 | ok | 96 % / 79 % | 99 % |
| q4_1 | 5 | 31 % | yes | yes | 4157 | 22/30 | 0.162 | 3/3 | ok | 95 % / 74 % | 100 % |
| iq4_nl | 4.5 | 28 % | yes | yes | 4619 | 26/30 | 0.112 | 3/3 | ok | 95 % / 76 % | 98 % |
| nvfp4 | 4.5 | 28 % | yes | yes | 4619 | 26/30 | 0.150 | 3/3 | ok | 96 % / 78 % | 99 % |
| f32 | 32 | 200 % | yes | yes | 939 | 30/30 | 0.004 | 3/3 | ok | n/m | n/m |
| K q8_0 / V q4_0 | 8.5 / 4.5 | ~66 % | yes | yes | 3198 | 27/30 | 0.062 | 3/3 | ok | 96 % / 79 % | 99 % |

Reading the table: a format passes when the needle test is 3/3 and the free generation of 800 tokens shows no loop or
gibberish. 4-bit and 5-bit caches and the 8-bit float formats lose accuracy on models with large key outliers:
Qwen2.5-1.5B (GPTQ) keeps `q8_0` and `int8` (29-30 of 30 identical top-1 tokens, needle 3/3) but breaks with `fp8_e4m3`,
`fp8_e5m2`, `q5_*`, `q4_*`, `iq4_nl` and `nvfp4` (0-6 of 30, needle 0/3, repetitive text). Keys are the sensitive half;
`--cache-type-k q8_0 --cache-type-v q4_0` keeps most of the saving. The server prints a warning when keys use a format
below 8 bits or `fp8`, and `nvfp4` is always marked experimental.

Prompts: for 16 or more rows of one sequence the quantized keys and values of the sequence are expanded once into an
FP16 scratch (at most `FLLM_KVQ_SCRATCH_MB`, default 256 MB) and the FP16 tensor-core / FlashInfer prompt kernels run
on it. Measured on Llama-3.1-8B (1.5k-token prompt): fp8_e4m3 3634 tokens/s and q4_0 3589 tokens/s with the scratch against 617 and 647 tokens/s when the generic attention kernel decodes inside the kernel
(`FLLM_KVQ_NO_EXPAND=1`), so the scratch is the default. Decode reads the quantized values directly inside the attention kernel (no full-cache dequantization); the decode rows of the table are 93-98 % of the FP16 speed at a short context and 67-81 % at 2k tokens (`q8_0` keeps its older dedicated kernel: 98-99 %). On the CPU all formats work through the scalar attention path (slower than f16 and q8_0). The kernels use no instruction newer than Turing (`FLLM_CUDA_ARCHS=75` builds and links); they were run on Blackwell only. The planner counts the exact bytes of a token (`KvFormat::bytes_per_slot`, scales included) for keys and values separately, so `--gpu-memory-utilization` and the context size follow the format; the log prints `paged <format> KV` and the block count of the pool.
The fused FP16 rope + cache-write path of 4-bit weights is used only with an FP16 cache.
Prefix caching works with every format; requests with `n > 1` copy the tail block plane by plane and need a block size that keeps every plane on whole words (the default 16-token blocks do).
