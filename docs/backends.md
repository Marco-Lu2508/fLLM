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
**bitsandbytes** (NF4B64/128, FP4B64/128, INT8R; no column): x86 CPU AVX2 and portable kernels, run by us (`docs/cpu-kernels.md`);
CUDA decode GEMV and tensor-core prompt path, compiled for sm_80/89/90/120 and never run on a GPU (L0); Metal has its own kernels (L0, see the
table below the MLXA one); Vulkan native has f32 shaders for them (L0*: ran on Mesa lavapipe, never on a GPU; `FLLM_VULKAN_NO_EXT_FORMATS=1` converts instead); every other backend, including wgpu, the NPU and the TPU path, takes them through a conversion at upload (L0†,
3.6 times the packed size in 16-bit, 7 times in F32; `docs/gpu-backends.md`).

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
| Intel NPU (OpenVINO) | all integer formats as INT8 per channel: Q4_K/Q6_K (Llama-3.2-1B Q4_K_M), Q4_K (Qwen2.5-0.5B): L2 on one Core Ultra 5 235U; the others L0† (requantised; `FLLM_NPU_WEIGHTS=exact` keeps Q4_0/Q8_0 bit for bit) | L0† | L0† | L0† | L0† | L0† (Q4ZP not registered) | L0 (-> FP16) |
| AMD XDNA NPU (native) | **L1 on one Ryzen AI 5 PRO 435**: prompt projections only, bf16 x bf16 -> f32 on fLLM's own kernel; Q4_0, Q8_0, Q4_K, Q6_K, F16, BF16, F32 decoded straight to bf16, every other dequantizable format through the generic dequantizer, the rest stays on the CPU ([npu-xdna.md](npu-xdna.md)) | L1 (same kernel; 183 + 576 self-test checks of the `linear`, `host` and `embedding` groups pass) | – | – | – | – | – |
| AMD XDNA NPU (FastFlowLM) | not fLLM's kernels: the models run in FastFlowLM, which fLLM drives and lists as a provider ([user/amd-npu.md](user/amd-npu.md)); the probe of one MatMul through ONNX Runtime/VitisAI remains | – | – | – | – | – | – |
| Qualcomm Hexagon NPU | probe only (FP16 MatMul through QNN HTP; Genie wrapper) | – | – | – | – | – | – |
| Apple Neural Engine | probe only (linear through Core ML) | – | – | – | – | – | – |
| Google TPU (PJRT/StableHLO) | L0† (host decoding, upload as bf16/f32/int8) | L0† | L0† | L0† | L0† | L0† (Q4Z/Q4ZP/FP8M through `compute::dequantize`) | L0 |


MLX affine group quantization (`MLXA`, 2 to 8 bits, groups of 32 or more, see [rust-formats-and-import.md](rust-formats-and-import.md#mlx-affine-quantization-mlxa)) per hardware, same scale as above:

| Hardware | MLXA |
| --- | --- |
| CPU (all tiers) | L1: portable Rust with an AVX2 + FMA instance chosen at run time (`cpu_mlx.rs`), f32 activations; the unit tests and the engine tests ran on an x86-64 machine (the AArch64 build uses the portable instance and was not run) |
| CUDA (NVIDIA) | L0: own kernels on the packed words (GEMV, FP32 tile, WMMA tensor cores from sm_75; `cuda/mlx_affine.cu`) compiled with nvcc 12.8 for sm_80/89/90/120; the device self test `mlx` has not run on a GPU |
| Metal native | L0: kernels for 36 formats (6 widths x 3 groups x f16/bf16 scales) built on the quantized linear family; the chunk decoder text compiled as C++ and run on the CPU decodes every width bit for bit; never compiled by the Metal compiler. `FLLM_METAL_NO_MLX=1` turns them off |
| Vulkan native | L0*: six shaders (one per bit width, group and scale type in push constants; f32, int8-dot and tiled prompt variants); ran on Mesa lavapipe in the Vulkan self test (`mlx` group, exact and fast pass), never on a GPU |
| wgpu, SYCL, NPU, TPU, ROCm/HIP | L0†: no kernel; the decoder uploads the weight dequantized as F16 (`RustDecoder::init_backend`, one stderr line), so a model that is 4.5 bits per weight on disk is 16 on the device |

Block-wise FP8 (`FP8B`), bitsandbytes (`NF4B64/128`, `FP4B64/128`, `INT8R`) and EXL2 on **Metal native** (same scale as above). All three are written without a
Mac at hand: **L0**. The MSL has been linted (`msl_check`), its chunk decoders compiled as C++ and compared bit for bit with the Rust reference
(`metal/emu/run_dec8.sh`, including the matrix-vector kernel body), the index arithmetic and device images mirrored by unit tests, and the Apple-only code
type-checked and clippy-checked for `aarch64-apple-darwin`; **no Metal compiler and no Apple GPU has seen the kernels**. The Apple-silicon CI runner runs
`metal_quant_kernels` (every format, several batch sizes, CPU dequantization as reference).

| Format | Metal native | Switch off |
| --- | --- | --- |
| `FP8B` (any power-of-two tile, f32 scales) | L0: the GGUF tensor is the device image (no copy); kernels `fllm_q_{mv,mm}_fp8b_*` look the tile scale up from the row and column of every chunk of 8 codes (`q_fmt_fp8b`, tile shape in `q_params::pad`); a tensor that is not one matrix of the shape (stacked experts) is expanded to F16 | `FLLM_METAL_NO_FP8B=1` |
| `NF4B64/128`, `FP4B64/128`, `INT8R` | L0: the GGUF tensor is the device image; code books as exact float bit patterns, `book[nibble] * absmax` per weight | `FLLM_METAL_NO_BNB=1` |
| `EXL2` | L0: own device image (header, rows of the blob, per-chunk table, group maxima), per-group widths 2 to 8 bits, activations gathered into stored order by the act-order prepass; blobs with a group that is not a multiple of 8 inputs are expanded to F16 | `FLLM_METAL_NO_EXL=1` |
| `EXL3` | L0†: expanded to F16 (Hadamard transforms and trellis tiles have no Metal kernel) | – |

The weights keep their packed size on the device (about 8, 4.5 and 2 to 8 bits per weight); the arithmetic follows the CPU kernels (f32 per weight, f32 activations,
simd_sum reduction; half tiles for batches above 8 unless the exact mode is set).

The same formats on **Vulkan native** (**L0\***: every shader ran on Mesa lavapipe, a CPU implementation of Vulkan, against the CPU reference in `selftest-gpu --device vulkan`
(groups `linear`, `fp8blk`, `mlx`; exact and fast pass, the fast pass also with `FLLM_VULKAN_FORCE_IDOT=1`) and in unit tests; no physical GPU has run them, and the
cooperative-matrix variants `_c` have not executed anywhere):

| Format | Vulkan native | Switch off |
| --- | --- | --- |
| `FP8B` (any power-of-two tile with at least 8 columns) and `FP8M` | L0\*: device image = codes plus one f32 scale per row and tile column (the tile scale repeated per row); f32 shaders `mv_fp8b_s`, `mm_fp8b_s`; `FP8M` is converted to tiles of one row; a row that is not a multiple of 8 columns is expanded to F16 | `FLLM_VULKAN_NO_EXT_FORMATS=1` |
| `NF4B64/128`, `FP4B64/128` | L0\*: codes and f32 absmax in two planes; f32 shaders (`book[nibble] * absmax`) | same |
| `INT8R` | L0\*: all variants (f32, int8 dot, tiled prompt kernel, cooperative matrices) | same |
| `MLXA` | L0\*: see the MLXA table above | same |
| `EXL2` | L0\*: rows with the scale nibbles first, a table per block of 32 inputs, activations gathered by the act-order prepass; groups must be a multiple of 32 inputs else F16 | same |
| `EXL3` | L0\*: `exl3_pre` / `exl3_mm` / `exl3_post`, every width and codebook ran on lavapipe; rows and columns multiples of 128 and a blob within one storage binding, else F16 | same |

\* L0 plus an execution on a software Vulkan device; the shaders are validated with `spirv-val`, which is more than L0 but not hardware.

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
| AMD XDNA (native kernels) | CPU reference stream (hybrid backend) or the host decoder with `FLLM_NPU_HOST=1`; only prompt projections run on the NPU | CPU | CPU | CPU |
| Qualcomm Hexagon / Apple ANE | – (probe only) | – | – | – |
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
| `aarch64-apple-darwin` | CI builds and runs unit tests, the CPU self tests and the Metal kernel checks on `macos-15` (Apple silicon) |
| `aarch64-pc-windows-msvc` | not built |
| `--device auto`, device numbers (`vulkan:N` = N-th device of the backend), `devices --json` | selection logic covered by unit tests and by runs without a GPU; L0 on hardware |
| Self-test harness (groups, `--only`, `--quants`, `--list`, `--strict`, `--json`) | the CPU against itself (`selftest-gpu --device cpu`: 220 checks, 0 failed, 35 skipped); this checks the harness, not a GPU backend. CUDA: see the README Status table |
| AMD XDNA native backend (`npu-xdna-native`) | L1: ran on one Ryzen AI 5 PRO 435 laptop under Linux (own `.xclbin`s built with mlir-aie/Peano, run through XRT 2.21.75): the `linear`/`host`/`embedding` self test passes (183 + 576 checks, 388 projections on the NPU, 4-bit weights expanded in the compute tiles and multiplied on the block-floating-point MAC), Llama-3.2-3B Q4_K_M gives the processor's greedy text on a 714-token prompt and processes prompts at 310-314 tok/s (400 tokens) and 222-227 tok/s (1180 tokens) against 155-168 and 117-131 on the 6-thread processor ([npu-xdna.md](npu-xdna.md)). Decode stays on the processor. Windows, macOS, other XDNA chips: untested (the Windows cross-build compiles) |
| Autotuner (`src/autotune.rs`) | L0: called by nothing, nothing measured |
| Intel NPU backend in the release binary | the cargo feature `npu-openvino` is in the **default** feature set; the OpenVINO runtime is opened at run time (`dlopen` / `LoadLibrary`, no link-time dependency), so the same binary starts on every machine and uses an NPU once the runtime is installed (section 6). Compiled, unit tested and run on one Core Ultra 5 235U (see section 6) |

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
* Embedding for NVFP4, Q4Z, IQ1-IQ3, IQ4_XS, Q8_K, FP8, FP8B, MLXA, bitsandbytes and EXL2/EXL3 has no shader (host decoding); `sampling_logits` with temperature, `moe_route` and
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
  work group of the matrix-vector shaders; `FLLM_VK_MV_MAX` (1..16) the largest batch they serve (default 16, on NVIDIA a per-format table: Q4_K 16, Q6_K and Q8_0 12,
  the rest 8, because the cooperative-matrix tile wins above that). `FLLM_VK_BENCH_LINEAR=1` prints, at the first projection, the time of typical projections
  (milliseconds, TMAC/s and GB/s for 1 .. 512 rows of Q4_K / Q6_K / Q4_0 / Q8_0 weights, vocabulary-sized ones included), the cooperative-matrix shapes of the device and the raw
  rates of the matrix units (float16 and int8 cooperative matrices, with and without the conversion and scaling of a quantization group, and the int8 dot product).
  Q4_K from 3 rows on uses `mv_q4_k_g`, a row-group variant: every group of 8 invocations owns its rows and walks all their blocks, so the sum over the blocks needs no
  work-group reduction (3 to 4 times faster from 9 rows on than the shared-memory tree on a Radeon 840M). The planner's memory budget is taken in steps of 256 MiB and the
  server logs the figures it used (`memory for the plan: ...`), see [benchmarks.md](benchmarks.md). On an AMD integrated GPU (Ryzen Z1 Extreme) the dedicated Q4_K / Q6_K matrix-vector shaders
  (`mv_q4k_idot`, `mv_q6k_idot`), 64 invocations with one row and the split of the key range from 64 keys on raised decode of a 1.5B Q4_K_M
  model from 49 to 64 tokens/s.
* NVIDIA prompts (RTX 2000 Ada, 2026-10-30): `mm_<fmt>_cw` is a 128 x 64 cooperative-matrix tile with a 64-wide K step and a prefetch of the next step into
  registers; it reads the activations as binary16 from the prepass `quant_act_f16` and is planned for batches from 64 on, K a multiple of 64, 32-wide
  subgroups and NVIDIA only (`FLLM_VK_COOP_WIDE=0` keeps the 64 x 64 tile of `mm_<fmt>_c`, a number sets the smallest batch). The workgroups are numbered token tile first
  (the token tiles of one block of weight rows run next to each other, so the block is read from memory once), the scale words of a Q4_K super-block are loaded once per
  four K steps and converted after the matrix products of a step, and Q6_K decodes after its products too (+25 % on the 8B model, [benchmarks.md](benchmarks.md)). `attn_prefill_cm_d128` /
  `_d64` is the prompt attention on cooperative matrices (f16 cache, key = value width 64 or 128, 64 query rows per workgroup of four subgroups); it
  replaced a scalar kernel that took 41 % of a 2000-token prompt (`FLLM_VK_ATTN_COOP=0` brings that kernel back; the exact self-test pass never uses the
  new one). Llama-3.2-3B Q4_K_M prompts of 580 / 2100 tokens: 1719 -> 2615 and 1431 -> 2810 tok/s, see [benchmarks.md](benchmarks.md).
* AMD prompts (Radeon 840M, RDNA 3.5, 2026-10-31): the cooperative-matrix pipelines ask for 32-wide subgroups (the default of RDNA compute is 64, which halves
  the matrix instructions; `FLLM_VK_COOP_SG64=1` keeps the default). `mm_q4_k_cx` (`mm_q4_k_cx.comp`) is a 64 x 128 tile derived from `mm_<fmt>_cw` (uvec4 tiles, 16-byte
  matrix loads, the epilogue staging inside the weight tile, the K slices of a step not unrolled): 0.94 - 0.97 TMAC/s against 0.86 for `mmq_q4_k_i` on a
  8192 x 3072 Q4_K weight with 512 tokens. It is planned for Q4_K only, from 128 tokens on and when at most a quarter of the last 128-token tile is empty; the
  device-wide timing at the first prompt (4096 x 4096 Q4_K, 512 tokens) still decides between it and the integer dot product, every other format and
  batch keeps `mmq_<fmt>_i`. The prompt attention on cooperative matrices (`FLLM_VK_ATTN_COOP=0` switches it off) now runs on AMD as well, pinned to 32-wide
  subgroups: 4.9 -> 2.5 ms per layer for a 460-token prompt. Numbers and the measurements that led there: [benchmarks.md](benchmarks.md).

### 5.7b Several devices in one run

`--device A,B,C` splits one model over the devices by layers (pipeline stages), `--tensor-parallel N` cuts the layers over N devices of one backend;
details, measurements and limits in [multi-gpu.md](multi-gpu.md). Per backend (the stage code is the device forward pass of that backend, nothing is
special-cased, the activations cross between stages through host memory):

| Backend | As a stage of a split | As a tensor-parallel rank | Test level |
| --- | --- | --- | --- |
| CPU | yes (also as the first or last stage next to GPUs) | yes (host path) | L2: unit and equivalence tests; L3: with an NVIDIA GPU and an Intel iGPU (Vulkan) |
| Vulkan (native and wgpu) | yes | yes (host all-reduce) | L3: RTX 2000 Ada + Intel RPL-S iGPU + CPU, Llama-3.2-3B Q4_K_M, layer split and tensor-parallel 2 |
| CUDA | yes (KV cache cut per stage, layered layout) | yes; NCCL / peer-to-peer all-reduce between cards is not used by the device pass (host all-reduce) | L3 as a stage next to the CPU and a wgpu-Vulkan iGPU on an RTX 2000 Ada (same text as the single GPU); several CUDA cards and tensor-parallel CUDA never run |
| HIP, Metal, SYCL, TPU | same code path (uniform KV pools cut to the stage's layers); not run in a split | same | L0 (compiles) |
| NPU | not in a list (refused) | no | |

Two backends on one physical GPU (HIP and Vulkan on one Radeon) are one device, not two: do not list both. Mixture-of-experts layers in a
tensor-parallel group run their experts on the device only on backends with the accumulate kernels (checked: none yet in a group; the CPU
backend stand-in has none), else on the host.

### 5.8 Other gaps

* **SYCL:** no DeltaNet, no `host_linear`, no graphs, no fused operations; attention is scalar FP32; native only after
  `git apply rust/engine/sycl/build_rs.patch` on `build.rs` (without the patch a stub that reports "no SYCL device");
  `sycl:N` counts the SYCL devices. Verified only with the open-source DPC++ nightly, not with the release `icpx`; `joint_matrix` names may
  differ. Vulkan is the recommended Intel default and SYCL an option (reasoning and a measurement plan in
  [research/README.md](research/README.md)).
* **NPU (OpenVINO):** no decode gain expected (bandwidth bound); the hybrid stream computes everything except `quantized_linear` single-threaded
  on the reference CPU, and weights are held twice; the host hook (a patch to `compute.rs`, active with `FLLM_NPU_HOST=1`) is the more usable
  route but has only been compiled as a patch; the rates of the cost model are placeholders.
* **AMD XDNA (native):** prompt projections only (64 or more rows, bf16); decode stays on the CPU; the NPU buffers hold bf16 copies of the
  weights (2 bytes per weight; budget = a quarter of the free memory), so a model that does not fit streams them again for every call; the
  `--device npu` hybrid backend needs a second quantized copy and the reference CPU backend (about 6 GB for a 3B model), the host decoder route does not.
  Needs a helper library built from C++ against XRT (`fllm-rs amd-npu build-shim`). Tested on one machine; [npu-xdna.md](npu-xdna.md).
* **Hexagon/ANE:** probes only, no compute path; `create()` always fails with a state message. The Qualcomm QAIRT headers are "Confidential
  and Proprietary" and are not in the repository. Notes: [research/](research/README.md). The old ONNX Runtime / VitisAI probe for AMD
  (`npu-xdna`, one MatMul) is still there and has no compute path.
* **TPU:** no q8_0 KV, no DeltaNet, each new shape compiles about 12 programs, no persistent cache; `model.rs` (a whole decoder block as one
  program) is not called by the decoder (no hook); the self-test tolerances assume f32 accuracy, and single FAIL lines are expected for
  `swiglu`, `geglu`, `attention`, `rms_norm`.
* **MoE experts larger than memory:** the experts of a layer that runs on the host are read from the mapped GGUF on demand and managed by the expert cache ([moe-streaming.md](moe-streaming.md); CPU path run on Linux and Windows with OLMoE). Not done: a hot subset of experts pinned on a CUDA/Vulkan/HIP device (the grouped kernels index a table of all experts of a layer), and a bound for CUDA's page-locking of streamed expert tensors (`CudaStream::pin` locks every streamed tensor and never unlocks it, so a model larger than RAM ends in the pageable-copy fallback).
* **Self test:** `SKIPPED (unsupported)` changes the count for backends that do not offer operations; the default of `selftest-gpu --device` is `auto`.
  The CPU backend itself reports 7 failed q8 attention checks in the self test.
* **Autotuner:** `src/autotune.rs` is called by nothing.
* **Numbers:** all speed figures for backends other than CPU and CUDA are expectations, not measurements; the L3 column is empty for them.

## 6. NPUs in the app

**Test level.** The Intel NPU path ran on one machine (a Core Ultra 5 235U PC, "Intel(R) AI Boost", architecture 3720, Windows 11, OpenVINO 2026.4.1 installed by
the app): the runtime installed itself, the NPU compiled its models, and Llama-3.2-1B and Qwen2.5-0.5B Q4_K_M read prompts on it with the same greedy text as the processor (measurements: `rust/engine/src/backend/npu/OPENVINO.md` section 0). That is one device and two small models; everything else in this section (AMD, Qualcomm, Apple, other Intel parts, other models) has
only been compiled and unit tested (L0). What exists:

| NPU | Status in the app |
| --- | --- |
| Intel NPU (Core Ultra, via OpenVINO) | detected; usable after the runtime is installed; **experimental**: `--device npu` runs the host decoder with the long prompt projections on the NPU (INT8 per channel, compiled models cached on disk); L2/L3 on one Core Ultra 5 235U |
| AMD XDNA (Ryzen AI) | two routes: **fLLM's own kernels** through XRT (feature `npu-xdna-native`, default; usable once the helper library is built with `fllm-rs amd-npu build-shim`; prompt projections only; L1 on one Ryzen AI 5 PRO 435, [npu-xdna.md](npu-xdna.md)) and FastFlowLM ([user/amd-npu.md](user/amd-npu.md)); both are shown on one card |
| Qualcomm Hexagon | detected and shown as "detected, not supported yet"; probe code only, never usable |
| Apple Neural Engine | not detected by the app; probe code only (feature `npu-ane`) |

**What the Intel path does.** `--device npu` (or `FLLM_NPU_HOST=1`) keeps the model on the fast host decoder and sends the projections of prompts of 64 or more rows to the NPU (INT8 weights, one scale per output row; the first start compiles about a second per projection in the background while the processor serves the prompts, later starts load the compiled programs from a cache). Measured on a Core Ultra 5 235U with Llama-3.2-1B Q4_K_M: a 512-token prompt is read about twice as fast as on the processor alone and about as fast as on the integrated graphics, a 130-token prompt 1.0 to 1.4 times as fast; decode is unchanged; with Qwen2.5-0.5B the integrated graphics are 2 to 3 times faster than the NPU. The NPU is therefore never chosen automatically. The older "hybrid" backend (`--npu-mode all` or `FLLM_NPU_HYBRID=1`) sends the large quantized projections (`quantized_linear`) of a prompt to the NPU and runs
everything else (attention, normalisation, the layers with a gate) on the processor's reference stream. By default only blocks of 64 or more rows go
to the NPU, so the answer, one row at a time, is computed by the processor. The load option **What the NPU computes: everything it can**
(`--npu-mode all`) sends every projection to the NPU, also single rows (static chunks of 16 rows, padded); the first start compiles about a hundred
small NPU models (several minutes, kept in OpenVINO's cache afterwards). Measured on the machine above with Qwen3.5-2B Q4_K_M and a short prompt:
answer about 5 tokens/s in that mode, 6 tokens/s by default, 5 tokens/s on the processor alone and 25 tokens/s on the integrated graphics chip, so
the NPU is not the fast choice on this hardware; the mode exists so that the whole model can run on an NPU, and for stronger NPUs. The model is often
slower than on the processor alone. The weights are held twice (the processor's copy and the NPU's converted copy). Only dense models work. Details, limits
and the measurement plan: `rust/engine/src/backend/npu/OPENVINO.md`.

**Weight formats on the NPU** (conversion at first use): every integer format (Q4_0, Q4_1, Q4_K, Q5_x, Q6_K, Q8_0, Q2_K, Q3_K, IQ4_*, MXFP4, NVFP4, AWQ/GPTQ 4-bit) becomes INT8 with one scale per output row (INT4 in groups of 32 or 128 compiles in tens of seconds for large projections; `FLLM_NPU_WEIGHTS=int4` or `exact` select those); F16, BF16, F32 as FP16. Nothing else. Environment: `FLLM_NPU_HOST`, `FLLM_NPU_WEIGHTS`, `FLLM_NPU_STATS`, `FLLM_NPU_BUDGET_MB`, `FLLM_NPU_CACHE_DIR`; `fllm-rs npu-bench` times one projection against the processor.

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
