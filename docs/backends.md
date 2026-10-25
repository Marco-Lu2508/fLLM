# Backend matrix: hardware x format x test level

Every cell states the test level reached (defined in [test-levels.md](test-levels.md)). **Unless a cell says otherwise, a backend
other than CPU and CUDA is L0: the code exists and compiles, but its kernels have not run on the target hardware.** Where this page
says "backend X can do Y", it means "the code for it exists and was compiled or syntax-checked", never "it runs" - except where the
README Status table or this page names a run.

How to verify on other hardware: [backends-verify.md](backends-verify.md). Device selection, `--device auto` and the self test:
[device-selection.md](device-selection.md).

## 1. Test levels and cell abbreviations

| Level | Meaning | How to reach it |
| --- | --- | --- |
| **L0** | compiled and/or syntax-checked; possibly also unit tests without hardware or a CPU emulation of the kernel text (stays L0) | `cargo check/clippy`, cross-check scripts for HIP, SYCL and MSL, `glslc` + `spirv-val` |
| **L1** | self test passes on real hardware | `fllm-rs selftest-gpu --device <name> --json out.json` with `failed = 0` (or `selftest-cpu`, or the `#[ignore]` hardware tests of the backend) |
| **L2** | greedy text (temperature 0) of a model on this hardware is identical to the CPU reference | text comparison from [backends-verify.md](backends-verify.md) |
| **L3** | speed measured and compared with a reference implementation on the same hardware | `fllm-rs bench` plus a comparison run |

A cross-compile check only proves that the Rust code compiles, not that it links or runs.

| Abbreviation | Meaning |
| --- | --- |
| **B** | Established: code that, according to the existing docs, has run on hardware |
| **B‡** | Established, but `docs/cpu-kernels.md` says "not run" (compiles, tests written, never executed on matching hardware); effectively L0 |
| **L0** | native kernel/path present, compiled only |
| **L0†** | works only through a conversion at upload (the host dequantises or requantises); no kernel of its own for the format |
| **–** | not present; the operation runs on the host/CPU or reports `Unsupported` |
| **?** | not determined; not guessed |

## 2. Weight formats

Columns: **L** = Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; **K** = Q2_K Q3_K Q4_K Q5_K Q6_K; **IQ4** = IQ4_NL; **MX** = MXFP4; **NV** = NVFP4;
**Q4Z** = AWQ/GPTQ 4-bit (Q4Z32/64/128, Q4ZP with act-order); **F** = F16/BF16 (F32).
Numbers in square brackets refer to the notes below the table.

| Hardware | L | K | IQ4 | MX | NV | Q4Z | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU x86 AVX2 / AVX-VNNI | B [1] | Q4_K-Q6_K: B; Q2_K, Q3_K: B‡ [1] | B‡ [1] | B‡ [1] | – [2] | – [2] | B [1] |
| CPU x86 AVX-512 (BW/VNNI/BF16) | B‡ [1] + L0 (16-row VNNI GEMM) | Q4_K-Q6_K: B‡; Q2_K, Q3_K, IQ4_XS: L0 | L0 | L0 | – [2] | – [2] | L0 (F32/F16/BF16, BF16 with `vdpbf16ps`) |
| CPU x86 AMX (INT8/BF16 tiles) | L0 (Q8_0 Q4_0 Q5_0 Q4_1 Q5_1); run only in an emulator | L0 (Q2_K-Q6_K, IQ4_XS) | L0 | L0 | – | – | BF16: L0 (`cols % 32 == 0`); F16 does not use AMX |
| CPU ARM NEON | B [3] | Q4_K-Q6_K: B; Q2_K, Q3_K: L0 | L0 (`tbl`) | L0 | – [2] | – [2] | L0 (F32/F16/BF16) |
| CPU ARM dotprod (`sdot`) | B [3] | as NEON | L0 | L0 | – | – | L0 |
| CPU ARM i8mm (`smmla`, `bfdot`) | B [3] | as NEON | L0 | L0 | – | – | L0, BF16 with `bfdot` |
| CPU ARM SVE (opt-in `FLLM_ARM_SVE=1`) | Q8_0 decode only: L0; otherwise the NEON tier | – | – | – | – | – | – |
| CUDA | B [4] | B (also IQ1-IQ3, IQ4_XS; see note 4) | B | B | B (FP4 kernels from Blackwell) | B (Marlin, from sm_75) | B |
| wgpu (Vulkan/Metal/DX12) | B [5] | Q4_K, Q5_K, Q6_K: B; Q2_K, Q3_K: F32 expansion at upload [5] | F32 expansion | F32 expansion | F32 expansion | F32 expansion | B |
| Vulkan native | L0; ran on a few devices (NVIDIA, Intel) according to the README Status table | L0 | L0 | L0 | L0 | L0 (Q4Z32/64/128; Q4ZP with a pre-step) | L0 (`lin_float`) |
| Metal native | L1 for the kernel checks on the Apple-silicon CI runner (see [ci.md](ci.md)); no whole model run | same | same | L0 (planar layout) | same | L0 (Q4ZP with a pre-step) | L1 (F32/F16/BF16, FP8) |
| ROCm/HIP RDNA3 (gfx11) | L0 [6] | L0 [6] | L0 [6] | L0 [6] | ? [7] | L0 (`hip_q4z.cu`, Int4 skinny) | L0 (skinny 1-4 tokens + WMMA) |
| ROCm/HIP RDNA4 (gfx12) | L0 [6] | L0 [6] | L0 [6] | L0 [6] | ? [7] | L0 (Int4 skinny + WMMA gfx12) | L0 |
| ROCm/HIP CDNA (gfx908/90a/942/950) | L0 [6] | L0 [6] | L0 [6] | L0 [6] | ? [7] | L0 (own GEMV + MFMA GEMM; Int4 skinny only RDNA3/4) | L0 (skinny + MFMA) |
| Intel SYCL (Level Zero, Arc/Xe) | L0 | L0 | L0 | L0 | L0† (-> f16) | L0† (-> f16) | L0 |
| Intel NPU (OpenVINO) | Q4_0, Q8_0: L0 exact (i4/i8, group 32); Q4_1, Q5_x: L0† (requantised) | L0† | L0† | L0† | L0† | L0† (Q4ZP not registered) | L0 (-> FP16) |
| AMD XDNA NPU | probe only (one MatMul through ONNX Runtime/VitisAI) | – | – | – | – | – | – |
| Qualcomm Hexagon NPU | probe only (FP16 MatMul through QNN HTP; Genie wrapper) | – | – | – | – | – | – |
| Apple Neural Engine | probe only (linear through Core ML) | – | – | – | – | – | – |
| Google TPU (PJRT/StableHLO) | L0† (host decoding, upload as bf16/f32/int8) | L0† | L0† | L0† | L0† | L0† (Q4Z/Q4ZP/FP8M through `compute::dequantize`) | L0 |

Notes:

1. **x86 CPU:** `docs/cpu-kernels.md` ("What is tested") lists the AVX2 and AVX-VNNI projections as "run on hardware by us"; the kernels
   for IQ4_NL, MXFP4 and Q2_K/Q3_K/IQ4_XS are listed there as "unit tests written, not run here" (hence B‡), the AVX-512 BW/VNNI kernels as
   "not run" (B‡). The unit tests of the AVX-512 kernels were run on a CPU machine with only AVX2/AVX-VNNI (247 passed with the
   `avx512` feature) together with `selftest-cpu` over the tiers present there; AVX-512 and AMX themselves have not run in that setup (the
   README Status table reports AVX-512 runs on an AMD Zen 5 machine, and AMX only in an emulator). The AMX model (`EmuTiles`) was written
   by the same author as the drivers.
2. NVFP4 and Q4Z have no SIMD kernel on the CPU (row decoding in `compute.rs`, `docs/cpu-kernels.md`).
3. **ARM CPU:** `docs/cpu-kernels.md`: "NEON `sdot` and `smmla` paths: compiled for aarch64-unknown-linux-gnu and run on hardware by us" (Q8_0,
   Q4_0/1, Q5_0/1, Q4_K, Q5_K, Q6_K). CI additionally runs the CPU self tests (`selftest-cpu`, `selftest-gpu --device cpu`) on the
   `ubuntu-24.04-arm` runner ([ci.md](ci.md)). Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4 and the float formats had no ARM kernel before the ARM kernel
   set was added (f32 row decoding or scalar code).
4. **CUDA:** the native formats are those listed in `NATIVE_KINDS` in `rust/engine/src/backend/cuda.rs`; see [gpu-backends.md](gpu-backends.md)
   for the exact list. Q4Z runs through the Marlin kernels from vLLM from compute capability 7.5. The README measurements come from a
   Blackwell card; [device-selection.md](device-selection.md) names Blackwell, Ada and Turing as hardware on which CUDA has run.
5. **wgpu:** native WGSL kernels for `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K`; everything else is expanded to F32 at upload
   ([gpu-backends.md](gpu-backends.md) section 1). Physically tested is only Intel Arrow Lake through Vulkan; NVIDIA, AMD, Apple/Metal and
   DX12 through wgpu are not validated.
6. **HIP:** the backend is `CudaBackend` with the HIP translation of the CUDA kernels (`hip_kernels()` in `build.rs`: 10 sources plus the
   `hip/*.cu` files). `hip_gemm.cu` (decode-in-LDS GEMM) is stated to cover all GGUF kinds; which format runs through which path is not
   established case by case. Syntax and code generation down to GCN assembler were checked for gfx908, gfx90a, gfx942, gfx950, gfx1030,
   gfx1100, gfx1151, gfx1201 - no execution. Q4Z on AMD requires `rows % 64 == 0` and `cols % 64 == 0` (otherwise expansion to F32).
7. HIP NVFP4: the build runs with `-DFLLM_NO_FP4`; NVFP4/FP8 Marlin remain stubs. Whether NVFP4 weights run on AMD through the generic path
   is not established.

## 3. Operations

Columns: **Attention** = `attention_paged` and `kv_cache_write` (f16 and q8_0 KV cache behind `DeviceKvCache`); **MoE** = routing,
expert-indexed matvec, combine, gather/scatter; **DeltaNet** = Gated DeltaNet (`delta_layer`, Qwen3.5/3.6); **Mamba** = Mamba-1/-2.

| Hardware | Attention | MoE | DeltaNet | Mamba |
| --- | --- | --- | --- | --- |
| CPU x86 AVX2 / AVX-VNNI | B (f16 and q8_0) | B (fused `linear_experts`, x86 only) | B (`cpu_delta.rs`) | B (host path `mamba_forward`) |
| CPU x86 AVX-512 | B‡ (512-bit decode attention) | L0 (per-expert prompts through the 512-bit kernels) | L0 (512-bit state update, bit-identical to AVX2) | – (no new kernel) |
| CPU x86 AMX | – | – (MoE prompts do not use AMX) | – | – |
| CPU ARM NEON / dotprod / i8mm | B (`sdot` + NEON) | single path; no fused `linear_experts` | L0 (NEON step, bit-identical to scalar) | B (host path) |
| CPU ARM SVE | – | – | – | – |
| CUDA | B (head width <= 512, wider goes to the CPU) | experts on the device, `moe_route`/`moe_expert_gemm` on the host (`supports_moe: false`) | present (`deltanet.cu`, `supports_delta`); level not stated here | Mamba-1 and Mamba-2 kernels (`mamba1.cu`, `mamba2.cu`, `supports_mamba1/2`) |
| wgpu | B (head width <= 512) | host fallback | host ([gpu-backends.md](gpu-backends.md) section 2) | host |
| Vulkan native | L0 (decode with split-KV, prompt with tiles; head width <= 512) | L0 (`moe_topk`, `moe_linear` in 12 formats, combine, gather/scatter; `moe_route` on the CPU; `moe_expert_gemm` = `Unsupported`) | L0 (HK 32/64/96/128, HV <= 128) | shaders and plans L0; `VulkanStream::mamba2_layer` exists but is not reachable through the `Stream` trait |
| Metal native | L1 for the CI kernel checks (decode split-KV, prompt with `simdgroup_matrix`; prefill head width <= 256) | L0 (`moe_linear` in 10 formats, matvec without grouped GEMM) | L0 | kernels and plans L0, no `Stream` method |
| ROCm/HIP (RDNA3/RDNA4/CDNA) | L0 (CUDA attention kernels under hipcc; DPP reduction in `attention4.cu`; vLLM paged attention not ported) | CUDA path under hipcc; not reported individually | – (`deltanet.cu` is not built for AMD, `fllm_delta_supported` returns 0, host path) | host (Mamba kernels are built from the CUDA sources only where they compile under hipcc; not verified) |
| Intel SYCL | L0 (scalar FP32, split-KV; head width <= 512) | L0 (`moe_linear`, top-k, combine, gather/scatter; `moe_route` on the CPU) | – (`supports_delta` = false) | – |
| Intel NPU (OpenVINO) | CPU reference stream (hybrid backend, dense models only) | CPU | CPU | CPU |
| AMD XDNA / Qualcomm Hexagon / Apple ANE | – (probe only) | – | – | – |
| Google TPU | L0 (StableHLO, f16 KV only, **no q8_0**) | L0 (top-k, combine, gather/scatter); `moe_linear`, `upload_pointer_table`, `moe_expert_gemm` = `Unsupported` | – (host) | – |

All columns are L0 unless "B" or another level is stated. The Metal kernels have additionally been emulated in C++ on the CPU
(160 + 139 + 2436 cases against the CPU backend or the f64 reference), and so have the Vulkan shader parts (`emu/`, `ops/mirror.rs`;
175 cases plus mirror tests). That proves index and layout logic, not compiler, driver or GPU behaviour. The Metal kernel checks that run on the
Apple-silicon CI runner are the exception: they execute on a GPU.

## 4. Platforms and build

| Target | State |
| --- | --- |
| Linux x86-64 (CPU, CUDA, wgpu) | established; CI builds, tests and runs the CPU self tests ([ci.md](ci.md)) |
| `x86_64-pc-windows-gnu` | cross-check (`check`/`clippy`) green for all features; MinGW + CUDA aborts in `build.rs` with a message |
| Windows MSVC (CPU, wgpu, native Vulkan, CUDA, HIP) | CI builds, runs unit tests and the CPU self tests on `windows-2022`; the CUDA build is compile-only; no GPU run on Windows |
| `aarch64-unknown-linux-gnu` | CI builds, runs unit tests and the CPU self tests on `ubuntu-24.04-arm` |
| `aarch64-apple-darwin` | CI builds and runs unit tests, the CPU self tests and the Metal kernel checks on `macos-14` (Apple silicon) |
| `aarch64-pc-windows-msvc` | not built |
| `--device auto`, device numbers (`vulkan:N` = N-th device of the backend), `devices --json` | selection logic covered by unit tests and by runs without a GPU; L0 on hardware |
| Self-test harness (groups, `--only`, `--quants`, `--list`, `--strict`, `--json`) | the CPU against itself (`selftest-gpu --device cpu`: 220 checks, 0 failed, 35 skipped); this checks the harness, not a GPU backend. CUDA: see the README Status table |
| Autotuner (`src/autotune.rs`) | L0: called by nothing, nothing measured |
| Intel NPU backend in the release binary | the cargo feature `npu-openvino` is in the **default** feature set; the OpenVINO runtime is opened at run time (`dlopen` / `LoadLibrary`, no link-time dependency), so the same binary starts on every machine and uses an NPU once the runtime is installed (section 6). Compiled and unit tested; **L0**: it has not run on NPU hardware |

## 5. Known gaps

Ordered by area.

### 5.1 Device index semantics

* **Rule:** `--device <backend>:<N>` counts the devices *of this backend* from 0, not the number `[n]` from `fllm-rs devices`. `VkContext::new(i)`,
  SYCL `create(i)` and `metal::device_info::open(i)` expect exactly this count.
* Selection with more than one adapter of the same API (numbers > 0) has never been executed.
* **Metal heuristic:** `metal::device_info::open(n)` reads `n` as a *global* number if it lies in `[first_index, first_index + count)`, otherwise as a
  position. The selection code always passes the local number; with two Metal GPUs (Intel Mac with an eGPU) `metal:1` could be read wrongly. The
  heuristic should be removed.
* `auto` never chooses SYCL, NPU or TPU (explicit selection only).

### 5.2 Duplicate native/wgpu entries in `devices`

* With `--features rust-gpu,vulkan` (or `metal`) the same GPU appears twice in the inventory: as a wgpu adapter and as a native entry. Native
  entries start with `native, Vulkan x.y.z, ...` or `Metal (native), ...`.
* Every device is labelled with backend and API (`native`/`wgpu`) and the wgpu adapters are listed as `wgpu-vulkan:N`, `wgpu-metal:N`,
  `wgpu-dx12:N`; one path can be forced with `vulkan-native`/`metal-native` or `wgpu-<api>`. This output has been seen on one machine (one adapter) only.
* With `--features hip` the CUDA list already shows the AMD cards (description "CUDA/ROCm"); there is no second entry.
* `fllm-rs devices` and `serve` under `tpu-pjrt` open the PJRT client already when listing; the TPU belongs to the process until it ends, and a
  second process fails.

### 5.3 Mamba without a `Stream` hook outside CUDA

* Mamba-1/-2 run on devices through the `Stream` trait only on CUDA (`supports_mamba1`, `supports_mamba2`, `mamba2_layer`). On every other
  backend the decoder runs Mamba layers on the host.
* Present but unused: shaders and plans for Vulkan (`plan_mamba2_mixer`, `plan_mamba_conv`, `plan_mamba1_scan`, `plan_mamba1_bc_norms`),
  `VulkanStream::mamba2_layer` (an inherent method, not part of the trait implementation), and MSL kernels and plans for Metal (`mamba_conv`,
  `mamba_scan1/2`, `mamba_gate_norm`, `mamba_bc_norm`). Wiring them in means implementing the trait methods `supports_mamba1/2`, `mamba2_layer`
  and the Mamba-1 steps for these backends. Mamba-1 needs two linears between its parts (`x_proj`, `dt_proj`).

### 5.4 ARM and x86 CPU tier ladders

* `cpu_kernels.rs`: enum `CpuTier`, `ALL`, `level()`, `native()`, `name()`, `supported()`. The ARM side adds `NeonSve` (level 4), the x86 side adds
  `Avx512Bf16` (level 5) and `Amx` (level 6). The ladders are separate per architecture; level 4 may occur twice.
* `devices.rs` (`cpu_features`/`cpu_capabilities`): the ARM probe reports `bf16`, `fhm`; the x86 side prints class and policy.
* `scripts/check-arm-cpu.sh` and `scripts/check_x86_ext.sh` run the architecture-specific checks on matching hardware.

### 5.5 HIP kernels not ported, and HIP gaps

* **Not ported:** vLLM `attention.cu` (paged attention; expects vLLM's cache layout, `DeviceKvCache` has another), `q_gemm_rdna3.cu`,
  `q_gemm_rdna3_wmma.cu`, `moe_q_gemm_rdna3.cu` (GPTQ layout vs. the Marlin-permuted Q4Z layout; fLLM has no Q4Z MoE path), LLMM1,
  `wvSplitK_hf_big_`, `wvSplitKrc`, `wvSplitKQ` (FP8), the MFMA bf16 variant for gfx94x. `hip_q4z.cu` (MMA GEMM + Int4 skinny) and DPP in
  `attention4.cu` take their place.
* DeltaNet is not built for AMD (`hip_stubs.cu`: `fllm_delta_supported` returns 0); `fllm_cuda_graph_*` are placeholders, NVFP4/FP8 Marlin and
  `fllm_q4z_gptq_repack` are stubs.
* All lane layouts (MFMA/WMMA) and DPP control codes come from documentation and rocWMMA headers; the Python emulation checks only the kernel
  indices against these layouts (`scripts/hip_mma_layout_check.py`, `scripts/hip_q4z_layout_check.py`).

### 5.6 Missing Metal parts

* Mamba: no `Stream` method (see 5.3).
* Not present: `linear_f16_out`, `glu_f16`, `rope_kv_write_f16`, graph capture, `host_linear`; GPU sampling (`supports_device_sampling` =
  false, temperature through the CPU backend); `quantized_linear_add` only with `scale == 1.0`.
* Embedding for IQ4_XS, NVFP4, Q4Z, FP8 and scaled tables runs on the host (slow for long prompts).
* Formats without a kernel (IQ1/IQ2/IQ3, Q8_1, Q8_K) are converted to F16 on the host at upload; `moe_linear` knows only 10 formats (no Q2_K, Q3_K,
  NVFP4, Q4Z, F16/BF16).
* Weights larger than `maxBufferLength` are not split into row slices. The decoder does not call `register_host_mapping` (no no-copy GGUF; each
  weight is copied once).
* `x` and `out` must not overlap in `quantized_linear`. The hazard declarations of the plans are checked against no GPU beyond the CI kernel
  checks (`FLLM_METAL_SERIAL=1` is the first thing to try when results are wrong).
* The Metal compiler check script is a C++ front end with a hand-written shim and does not know address-space, attribute or threadgroup-memory
  rules; in directory mode it gives false alarms. The CI runner compiles the real shaders.
* All bandwidth figures for the M1-M5 chips are estimates, especially M3 Max, M4 Max and M5.

### 5.7 Missing Vulkan parts

* Mamba: see 5.3. `moe_expert_gemm` = `Unsupported`; `moe_linear` is a matvec (no grouped GEMM); `lin_float` is not tiled.
* Embedding for NVFP4, Q4Z, IQ1-IQ3, IQ4_XS, Q8_K, FP8 has no shader (host decoding); `sampling_logits` with temperature, `moe_route` and
  attention with head width > 512 run on the CPU.
* MoE experts are held twice in memory (arena plus individual weights); the arena and the `host_linear` scratch are not part of the
  `MemoryPlan`, so OOM is possible when memory is almost full.
* Split KV pools (beyond `maxBufferSize`) work only for attention and KV writes, not for `upload`, `download`, `copy_buffer`, `zero_buffer` or
  CPU attention.
* Cooperative matrix is planned only for 16x16x16 f16/f32 and only with a pinnable subgroup size; `VK_NV_cooperative_matrix` and bf16/FP8
  shader extensions are not used.
* One queue, no asynchronous transfer, no buffer device address (the `_bda` variants are ready but unused), no descriptor indexing.
* By default the fast variants run (`_i`/`_c` round activations to int8 or compute tiles in f16), so greedy text can differ from the CPU. Use
  `FLLM_VULKAN_EXACT=1` for comparisons; the Vulkan self test passes only with it.
* Tuning and measuring a device: `FLLM_VK_PROFILE=1` times every dispatch with GPU timestamps and prints a table of the biggest shaders to stderr
  every 256 submissions; `FLLM_VK_MV_WG` (32..256, power of two) and `FLLM_VK_MV_ROWS` (1..8) override the work-group size and the rows per
  work group of the matrix-vector shaders. On an AMD integrated GPU (Ryzen Z1 Extreme) the dedicated Q4_K / Q6_K matrix-vector shaders
  (`mv_q4k_idot`, `mv_q6k_idot`), 64 invocations with one row and the split of the key range from 64 keys on raised decode of a 1.5B Q4_K_M
  model from 49 to 64 tokens/s.

### 5.8 Other gaps

* **SYCL:** no DeltaNet, no `host_linear`, no graphs, no fused operations; attention is scalar FP32; native only after
  `git apply rust/engine/sycl/build_rs.patch` on `build.rs` (without the patch a stub that reports "no SYCL device");
  `sycl:N` counts the SYCL devices. Verified only with the open-source DPC++ nightly, not with the release `icpx`; `joint_matrix` names may
  differ. Vulkan is the recommended Intel default and SYCL an option (reasoning and a measurement plan in
  [research/README.md](research/README.md)).
* **NPU (OpenVINO):** no decode gain expected (bandwidth bound); the hybrid stream computes everything except `quantized_linear` single-threaded
  on the reference CPU, and weights are held twice; the host hook (a patch to `compute.rs`, active with `FLLM_NPU_HOST=1`) is the more usable
  route but has only been compiled as a patch; the rates of the cost model are placeholders.
* **XDNA/Hexagon/ANE:** probes only, no compute path; `create()` always fails with a state message. The Qualcomm QAIRT headers are "Confidential
  and Proprietary" and are not in the repository. Notes: [research/](research/README.md).
* **TPU:** no q8_0 KV, no DeltaNet, each new shape compiles about 12 programs, no persistent cache; `model.rs` (a whole decoder block as one
  program) is not called by the decoder (no hook); the self-test tolerances assume f32 accuracy, and single FAIL lines are expected for
  `swiglu`, `geglu`, `attention`, `rms_norm`.
* **Self test:** `SKIPPED (unsupported)` changes the count for backends that do not offer operations; the default of `selftest-gpu --device` is `auto`.
  The CPU backend itself reports 7 failed q8 attention checks in the self test.
* **Autotuner:** `src/autotune.rs` is called by nothing.
* **Numbers:** all speed figures for backends other than CPU and CUDA are expectations, not measurements; the L3 column is empty for them.

## 6. NPUs in the app

**Test level: L0 for everything in this section.** The NPU backends compile and their logic is unit tested; the authors have not run them on NPU
hardware. Nothing here says that an NPU works. What exists:

| NPU | Status in the app |
| --- | --- |
| Intel NPU (Core Ultra, via OpenVINO) | detected; usable after the runtime is installed; **experimental** hybrid backend (L0) |
| AMD XDNA (Ryzen AI) | detected and shown as "detected, not supported yet"; probe code only, never usable |
| Qualcomm Hexagon | detected and shown as "detected, not supported yet"; probe code only, never usable |
| Apple Neural Engine | not detected by the app; probe code only (feature `npu-ane`) |

**What the Intel path does.** The "hybrid" backend sends the large quantized projections (`quantized_linear`) of a prompt to the NPU and runs
everything else on the processor's reference stream. No speed-up is expected when the model writes its answer, and the model is often slower than
on the processor alone. The weights are held twice (the processor's copy and the NPU's converted copy). Only dense models work. Details, limits
and the measurement plan: `rust/engine/src/backend/npu/OPENVINO.md`.

**Weight formats on the NPU** (conversion at load time): Q4_0 and Q8_0 exact (INT4 / INT8, group 32); Q4_1, Q2_K, Q3_K, Q4_K, IQ4_NL, MXFP4, NVFP4 and
AWQ/GPTQ 4-bit re-quantised to INT4 (group 128); Q5_0, Q5_1, Q5_K, Q6_K re-quantised to INT8; F16, BF16, F32 as FP16. Nothing else.

**Detection** (works without the OpenVINO runtime, `backend/npu/probe.rs`): Linux, the kernel accelerator class (`/sys/class/accel/accelN`,
`/dev/accel/accelN`; the driver `intel_vpu` names an Intel NPU, `amdxdna` an AMD one); Windows, the present devices of the device-manager classes
`ComputeAccelerator` and `System`, recognised by name (`Intel(R) AI Boost`, `AMD IPU Device`, `Qualcomm(R) Hexagon(TM) NPU`), asked through PowerShell.
When the runtime is installed, OpenVINO's own device list decides what is usable.

**Setting up the runtime.** `fllm-rs npu-setup` (or the **Install the runtime** button in Settings, Hardware) downloads the Python wheel
`openvino` 2026.4.1 for this system (Linux x86-64 / aarch64, Windows x86-64) from `files.pythonhosted.org` only, checks its size and SHA-256 against
the values pinned in `backend/npu/setup.rs` (taken from the file list of the release on PyPI, `pypi.org/project/openvino/2026.4.1`, on 2026-10-25), and
unpacks `openvino/libs/` into `~/.fllm/openvino` (`$FLLM_HOME/openvino`). No Python is needed. An interrupted download is resumed; the run can be
cancelled. `fllm-rs npu-setup --remove` (or **Remove the runtime**) deletes the directory. The OpenVINO runtime is Apache-2.0 licensed and is
downloaded by the user's machine, not redistributed with fLLM. The pinned values were copied from the PyPI page and have not been exercised
against the real download by the authors; a wrong value makes the setup fail with a checksum message and nothing is installed. The NPU driver itself
is separate: on Linux the distribution's `intel_vpu` module and the user-space NPU driver, with access to `/dev/accel/accel0` (group `render`);
on Windows the Intel NPU driver (Windows Update or the maker's site).

**Using it.** `--device auto` never picks the NPU. Choose **NPU** under *Advanced*, *Run on* in the load dialog of the Models page (listed only when the
state is ready) or start with `--device npu`. If the NPU cannot be created or cannot take the model, the server does not stop: it logs the reason,
runs the model on the processor and shows the reason on the NPU card (`last_error` of `GET /api/studio/npu`).

**Routes.** `GET /api/studio/npu` (state, hardware found, runtime path and version, setup progress, last error); `POST /api/studio/npu/setup`
(start the setup, or `{"action":"cancel"}`); `DELETE /api/studio/npu/setup` (remove the runtime). The two changing routes are administration and
only accept this machine, also when an API key is set.

**Cargo feature string for the release and desktop workflows.** Nothing has to be added: `npu-openvino` is in the default features, so
`--features rust-gpu,vulkan` (and any list that does not use `--no-default-features`) already contains it. A build with
`--no-default-features` has no NPU backend; the app then reports "this build has no NPU support".
