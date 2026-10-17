# Evaluation: Vulkan versus SYCL (oneAPI) for Intel GPUs

Research note, not user documentation.

Subject: Intel Arc (Alchemist/Battlemage, dGPU), Iris Xe (Gen12, iGPU), Core Ultra iGPUs (Meteor Lake/Arrow Lake: Xe-LPG without XMX; Lunar
Lake/Panther Lake: Xe2/Xe3 with XMX), and data-center GPUs (Max/PVC) at the margin. State: **all backends at test level L0** (see [test-levels.md](../test-levels.md)).

> **Statement of honesty.** There is not a single measurement of our own, neither for Vulkan nor for SYCL. Everything below is (a) expectations derived from
> hardware data and the construction of the kernels, (b) publicly reported measurements of other projects (with links, unverified, different software, different
> drivers) and (c) effort estimates. A backend "runs" only from L1. The measurement plan at the end says what should be measured to
> replace this page.

## 1. Summary and recommendation

1. **Vulkan as the default route for Intel GPUs.** It needs nothing but the graphics driver, runs with the same code on
   Linux (Mesa ANV) and Windows (Intel driver), on iGPU and dGPU, and at the same time on AMD/NVIDIA/Apple; the Vulkan backend already covers more
   formats and architectures than SYCL (16 formats including Q4Z/NVFP4, integer-dot and cooperative-matrix variants, DeltaNet, Mamba, MoE).
2. **Keep SYCL as an optional, explicitly experimental feature, but do not extend it before the measurement plan has a
   result.** It is worthwhile only if, on **Arc dGPUs**, it is at least about 20 % faster than Vulkan under the same conditions
   (decode or prompt) and this holds on Linux **and** Windows. Public reports show both directions (Section 3.4): on
   iGPUs Vulkan is level or ahead, on a large dGPU SYCL is reported at +50 % in token generation. Only measurement decides.
3. If SYCL is not needed, removing it is cheap: one Cargo feature, one folder `rust/engine/sycl/`, three Rust files, one
   function in `build.rs`; none of it touches the CPU or CUDA.

## 2. What is compared

| | Vulkan (native through `ash`) | SYCL (DPC++/oneAPI, Level Zero) |
|---|---|---|
| Programming model | GLSL compute shaders (SPIR-V), compiled by `glslc` at build time, kernels in the repository | C++17 with SYCL 2020 + Intel extensions (`joint_matrix`), compiled by `icpx` (SPIR-V, JIT or AOT) |
| Matrix engine (XMX) | through `VK_KHR_cooperative_matrix` (`mm_coopmat.comp`), driver dependent | directly through `joint_matrix` (`linear_xmx.cpp`, shapes 8x16x16 and 8x8x16), compiler-version dependent |
| Integer dot (DP4A) | `mv_idot.comp`, `mm_idot.comp`, `quant_act_q8.comp` (Q8 activations) | **not implemented** (FP32 FMA with dequantisation; DP4A is possible through `sycl::ext::oneapi::dot_acc`, see 6) |
| Formats (weights) | 16 formats: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ4_NL, MXFP4, NVFP4, Q4Z/Q4ZP | 16 GGUF types (F32/F16/BF16, the legacy and K formats above, IQ4_NL/IQ4_XS, MXFP4); FP8, NVFP4, Q4Z are converted to f16, the rest to Q5_K |
| Operations | attention (decode, prompt, merge), KV write, norm, RoPE, GLU, MoE (top-k, linear, gather/scatter, combine), **DeltaNet, Mamba** | the same without DeltaNet/Mamba/`host_linear`/graphs; attention as a scalar split-KV kernel (no XMX) |
| Host part | one queue, timeline semaphore, one command buffer per step | one `queue.submit` per operation on an in-order queue (about 12-20 launches per layer) |
| Platforms | Intel, AMD, NVIDIA, Apple (MoltenVK), Android | Intel GPUs; formally also NVIDIA/AMD through other SYCL plugins (unused here, `sub_group` 16 is a prerequisite) |
| Size | Vulkan backend plus its shader sets | about 2.4k lines of C++ (without tests), about 1.6k lines of Rust |

## 3. Expected speed

### 3.1 Hardware reference figures (vendor data, rounded)

| Device | Bandwidth | FP32 (vector) | XMX fp16 | DP4A (INT8 on vector) |
|---|---|---|---|---|
| Arc A770 (Alchemist, 32 Xe cores) | about 560 GB/s | about 20 TFLOPS | about 130 TFLOPS | about 4x FP32 |
| Arc B580 (Battlemage, 20 Xe2 cores) | about 456 GB/s | about 14 TFLOPS | about 110 TFLOPS | about 4x FP32 |
| Iris Xe (Gen12, 96 EU) | about 60-85 GB/s (shared) | about 2-2.5 TFLOPS | none | yes |
| Core Ultra 7 155H (Xe-LPG, 128 EU) | about 90-120 GB/s (shared) | about 4-5 TFLOPS | none | yes |
| Lunar Lake / Panther Lake (Xe2/Xe3 iGPU) | about 80-130 GB/s (shared) | 2-4 TFLOPS | yes | yes |

### 3.2 Decode (batch 1): memory bandwidth decides

For every generated token almost the whole weight file is read once. Upper bound: `tok/s ~ bandwidth x utilisation / file size`. A well
built matvec kernel typically reaches 55-70 % of peak bandwidth; both routes (Vulkan shaders and SYCL kernels) are limited equally here,
the differences come from driver code generation and launch overhead, not from the programming language.

| Model (example Llama-3.1-8B, Q4_K_M, about 4.9 GB) | Expected upper bound at 60 % utilisation |
|---|---|
| A770 | about 65-70 tok/s |
| B580 | about 55 tok/s |
| Iris Xe | about 8-10 tok/s |
| Core Ultra 155H | about 10-12 tok/s |

(Calculation, not a measurement. Q8_0 about half as fast, Q6_K/Q5_K in between, F16 a quarter.) Differences per format class at equal bandwidth:

| Format class | Decode in SYCL | Remark |
|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 (blocks of 32) | 1 work-item = 1 block (16 B quants + 2 B scale, separate arrays) -> 16-byte loads, little ALU per weight | bandwidth bound; MXFP4/IQ4_NL with table lookups need a little more ALU |
| Q8_0 | 16 values per unit, 16 B load | bandwidth bound (more bytes per weight) |
| Q4_K, Q5_K | 16 units per super-block, scales/mins are unpacked per unit from 12 bytes | about 10-15 ALU operations per 16 weights; on iGPUs the ALU can be co-limiting |
| Q6_K, Q2_K, Q3_K | 8 units per super-block of 32 values | like K-quants, a little more bit work |
| F16/BF16/F32 | rows padded to 8 values | purely bandwidth bound |

A DP4A path (activations as Int8, `dot_acc`) would reduce the ALU load of the K-quants and is the main gain for iGPUs without XMX,
see 6. Vulkan already has it as a separate shader.

**Launch overhead.** One decode step of a 32-layer model consists of about 400-600 kernel launches. At 10-20 us per launch through the
SYCL runtime (Level Zero) that is 4-12 ms per token - a large share for a target of 60-100 tok/s (10-16 ms per token). The remedy is
the Level Zero variant "immediate command lists" (default in newer runtimes) or `sycl_ext_oneapi_graph`; the SYCL backend uses neither.
Vulkan records one command buffer per step; its host overhead is expected to be lower. This is the most important
point at which SYCL falls behind Vulkan in practice and a possible reason for the deficit in the third-party measurement (3.4) -
unproven.

### 3.3 Prompt processing (batch > 8): compute throughput decides

Per token `2 x parameters` FLOPs are needed (8B model: 16 GFLOP).

| Path | Device | realistic share of peak | rough expectation (8B model) |
|---|---|---|---|
| SYCL `LinTiled` (FP32, batch 9-15 and all devices without XMX) | A770 | 15-25 % of 20 TFLOPS | about 190-310 tok/s |
| | Iris Xe | 15-25 % of 2.2 TFLOPS | about 20-35 tok/s |
| SYCL `LinXmx` (fp16, batch >= 16) | A770 / B580 | 10-25 % of XMX peak (dequantisation and tile staging in local memory limit it) | about 800-2000 tok/s |
| Vulkan `mm_coopmat` / `mm_idot` | same devices | depends on driver support for `VK_KHR_cooperative_matrix` | similar to lower; on iGPUs without XMX `mm_idot` (integer dot) is the better path |

Prompt attention is still scalar in both routes (SYCL: `attention.cpp`, FP32; Vulkan: `attn_prefill.comp`). Public measurements show
that missing XMX flash attention is the largest deficit of llama.cpp-SYCL against vLLM-XPU on long prompts (2.4-15x in the prompt);
this also applies to our kernel. This is the largest open lever for both routes.

### 3.4 Publicly reported measurements (not ours, other software)

* Panther Lake iGPU (Arc B390, Mesa ANV, Qwen3.6-35B-A3B Q4_K_M, llama.cpp): Vulkan pp512 633 tok/s, tg128 33.0 tok/s; SYCL (Level Zero) 615 / 30.9;
  so Vulkan +3 % in the prompt, +7 % in generation ([grigio.org, 16.09.2026](https://grigio.org/benchmarking-llama-cpp-backends-on-intel-panther-lake-vulkan-vs-sycl-vs-openvino-vs-cpu/)).
* Windows, Core Ultra 7 155U (iGPU), Gemma-4-E4B Q4_K_M with MTP drafting: Vulkan 20.1 tok/s, SYCL 7.0 tok/s in generation (2.9x), SYCL slightly ahead in the prompt
  ([llama.cpp issue 26010](https://github.com/ggml-org/llama.cpp/issues/26010)); a single report with speculative decoding, not generalisable.
* Arc Pro B70: llama.cpp-SYCL prompt far behind vLLM-XPU because of missing XMX attention, decode roughly equal (bandwidth bound)
  ([PMZFX, engine-comparison](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/engine-comparison.md)).
* A comparison report gives +52 % generation for SYCL over Vulkan for Qwen3.6-27B on the Arc Pro B70
  ([zenn.dev](https://zenn.dev/eyeofjapetus/articles/015908f77349ca?locale=en), secondary source).
* On Vulkan, llama.cpp uses the cooperative-matrix extension rather than XMX directly; whether it is active depends on the driver and reported features
  ([discussion 29907](https://github.com/ggml-org/llama.cpp/discussions/29907)).

Conclusion: **no consistent winner.** iGPU: Vulkan level or ahead; large dGPU: individual SYCL reports ahead in generation; the prompt depends on
XMX (through cooperative matrix or `joint_matrix`) and on attention.

## 4. Driver and installation effort

| | Vulkan | SYCL |
|---|---|---|
| Building | `glslc` (in the Vulkan SDK; on Linux `glslc` from `shaderc`); the SPIR-V files are produced in `build.rs` | **oneAPI DPC++ compiler** (`icpx`, Base Toolkit about 2 GB offline installer, or the open-source compiler from intel/llvm); the version must match the driver and `libsycl` |
| Run time on Linux | Mesa (ANV) or the Intel driver only | `libsycl` (part of the oneAPI runtime), Level Zero loader (`libze1`), compute runtime (`intel-level-zero-gpu`/`libze-intel-gpu1`), kernel driver `xe` or `i915`; optional AOT with `ocloc` |
| Run time on Windows | Intel graphics driver | Intel graphics driver (contains the Level Zero loader) **plus** the oneAPI runtime DLLs (`sycl*.dll`, UR adapter, SVML/IMF), which must be shipped with the program (about 100-200 MB) |
| First start | pipeline cache on disk | SPIR-V is compiled by the driver at the first call of each kernel; many instances (about 150-200 kernels) -> noticeable warm-up time; AOT solves this but costs a target list |
| Failure picture without the installation | "device not found", fallback to wgpu/CPU | the program does not run without `libsycl`/loader or reports "no GPU device" (tested here: the binary reports this cleanly) |
| Container/WSL | Vulkan through `dzn`/Mesa only to a limited extent | reported: Arc GPU not visible in WSL2 (see zenn.dev above); Docker needs `--device=/dev/dri` |

## 5. Windows suitability

* **Vulkan**: device and driver are enough; no further build effort; the MSVC target is uncritical (`ash` loads `vulkan-1.dll` at run time).
* **SYCL**: `icx` + the MSVC linker are possible; `build.rs` contains an unverified Windows branch (`/std:c++17`, `llvm-lib`, `sycl*.lib`). Open points:
  the dynamic CRT (`/MD`) must match Rust's MSVC target, the import library is called `sycl7.lib`/`sycl8.lib` depending on the version, the DLLs must sit next to
  the `.exe` or be on `PATH`. `x86_64-pc-windows-gnu` cannot do SYCL (no Intel compiler for MinGW). Overall assessment: feasible,
  but a sub-project of its own (MSVC needs to be verified).

## 6. Maintenance effort and risks

* **Two toolchains.** SYCL adds a second kernel language and a second compiler; Vulkan uses the same GLSL base for Intel, AMD, NVIDIA and (MoltenVK) Apple.
* **API churn.** `joint_matrix` has changed several times (layout names, `matrix_combinations`, shapes per architecture); code that compiles with
  oneAPI 2025.x may need to be touched again for 2026.x. The SYCL backend builds against a nightly of the open-source line (clang 24) and
  writes the shape query defensively (`#ifdef SYCL_EXT_ONEAPI_MATRIX`, `try`/`catch`), but remains unverified against the release `icpx`.
* **Shape choice at run time.** The XMX shape (8x16x16 with sub-group 16 versus 8x8x16 with sub-group 8) depends on the architecture; wrong detection = a kernel error when the
  driver compiles. The fallback is `FLLM_SYCL_EXACT=1`.
* **Not covered in SYCL**: DeltaNet, Mamba, native Q4Z/NVFP4/FP8 (conversion to f16 doubles to quadruples the weight memory), `host_linear`
  (models larger than VRAM), graph replay, fused operations (QKV, linear+residual), XMX attention, DP4A.
* **Allocation limit.** Level Zero limits a single allocation to about 4 GiB by default (large vocabulary matrices, KV pools): an environment variable is needed.
* **Size**: the Vulkan backend already contains the broader kernel collection; SYCL would double the maintenance of every new format/operator.

## 7. Recommendation with reasoning

1. **Intel default = Vulkan**, because (a) zero installation beyond the driver, (b) one code base for all vendors, (c) already more complete, (d) the same
   decode upper bound (bandwidth), (e) on iGPUs, the most common Intel device, publicly level or ahead, (f) Windows delivery without a
   runtime package.
2. **Keep SYCL only as a research path**: it offers (a) direct access to XMX without depending on the driver's cooperative-matrix support,
   (b) the possibility of integrating oneMKL/oneDNN GEMMs and Level Zero functions, (c) individual reports of a lead on large dGPUs. Only measurement shows whether that holds.
3. **Decision rule after the measurement plan:** SYCL stays and gets DP4A, XMX attention and graphs if it is at least 20 % faster than Vulkan on at least one Arc dGPU
   (A or B series) on Linux **and** Windows (median of pp512/pp2048 or tg128 over the model list below).
   Otherwise the feature is frozen (unchanged, without maintenance) or removed.
4. Regardless of the outcome: XMX attention (prompt) is the largest open performance reserve of both routes.

## 8. Measurement plan

Prerequisite: the Vulkan and SYCL backends are built on the same machine; first L1 (self test), then L2 (text equality), then L3 (numbers).

**Machines:** (1) an Arc dGPU (A750/A770/B570/B580, Linux **and** Windows), (2) a machine with Iris Xe or a Core Ultra iGPU, (3) if available Lunar/Panther Lake (Xe2/Xe3 iGPU with XMX).
Record: driver version (`fllm-rs devices` shows `driver`), oneAPI version, Mesa version, kernel, power plan, clock limits, memory (channel count/MT/s for iGPUs).

**Models:** Llama-3.1-8B-Instruct in Q4_K_M, Q8_0 and Q4_0; a small model (Qwen3-1.7B Q8_0); a MoE model (Qwen3-30B-A3B Q4_K_M if memory allows);
a gpt-oss model in MXFP4. KV cache f16 and q8_0.

**Steps per machine:**
1. L1: `fllm-rs selftest-gpu --device sycl --json sycl.json` and `--device vulkan`; `cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu`;
   `tests/run_decode_check.sh` (only to check the host side). Narrow down errors first with `FLLM_SYCL_EXACT=1` (disables the fp16 XMX kernels).
2. L2: per model the same 5 prompts with greedy decoding (temperature 0) on CPU, Vulkan, SYCL; the first 64 tokens must match (with fp16 paths, deviations after the first dozen tokens are possible: note the position of the deviation).
3. L3: `fllm-rs bench` (or `llama-bench` as a third-party reference on the same machine, `-p 512,2048 -n 128 -fa 1`, depth 0 and 8192) per backend, per model, 5 repetitions, cold (first measurement separate, contains shader/JIT compilation) and warm.
   In addition `llama-bench` with `-DGGML_VULKAN=ON` and `-DGGML_SYCL=ON` as the **upper bound of expectation** ("best known engine on the same hardware").
4. To record per measurement point: pp512, pp2048, tg128 (tok/s), time to first token, peak memory, power draw (`xpu-smi`/HWiNFO), share of bandwidth
   (`tg x file size / peak bandwidth`) for decode, share of matrix peak for the prompt.
5. Launch overhead separately: time from `fllm-rs serve` to the first token cold/warm (SYCL: JIT; repeat with `FLLM_SYCL_AOT=<architecture>`), number of kernel launches per token (Level Zero trace `ZE_ENABLE_TRACING_LAYER` or VTune/`unitrace`).
6. Windows extra: delivery size (program + DLLs), start on a machine without oneAPI, behaviour on driver update.

**Evaluation:** a table backend x machine x model with median and spread; apply the decision rule of Section 7.3. For SYCL additionally:
the effect of `FLLM_SYCL_EXACT=1` (fp16 XMX off) on pp512, and of `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` on load time.
