# Intel NPU via OpenVINO in fLLM

Research, design, limits and measurement plan for the `npu-openvino` feature. **Test level L2/L3 on one machine** (see
[test-levels.md](../../../../../docs/test-levels.md)): section 0 below holds what was measured on a Core Ultra 5 235U (NPU 3720, Windows 11,
OpenVINO 2026.4.1, driver version 1004841) on 2026-10-30; sections 1 to 10 are the earlier research and design text, whose numbers are quotes or
*assumptions* unless section 0 replaces them. Other Intel NPUs (Lunar Lake 4000 and later), Linux and other models are untested.

## 0. Measured on a Core Ultra 5 235U (2026-10-30)

**What changed because of it.**

* The "195 s to start" seen in the first Windows run was not compilation: 18 models compiled in 1.3 s; the time was the reference CPU backend of the self test.
* Compile time depends hard on the weight form: INT4 in groups of 32 (the exact Q4_0 form) takes 40-90 s per large projection (338 s once), INT4 in groups of 128
  5-12 s, **INT8 or INT4 per output channel 0.7-1.2 s**. The channel-wise forms are also the fastest to run. So the default is now **INT8 per channel for every
  integer format** (`FLLM_NPU_WEIGHTS=int8`; error against the source at most 1/254 of the row maximum, small beside the 4-bit rounding the source has; the model's
  NPU copy is about its size at 8 bit). `FLLM_NPU_WEIGHTS=int4` (groups of 128) halves the memory, `exact` keeps Q4_0/Q8_0 bit for bit; both compile slowly.
* The blob cache works (`CACHE_DIR`, default `~/.cache/fllm/npu`, on Windows `%LOCALAPPDATA%\fllm\npu`): 2048x2048 g32: 1005 ms -> 42 ms, g128 544 -> 97 ms.
  The first compile decides the mode: no cache or a slow first compile means the other models are compiled by a background thread while the processor serves the
  prompts (the server is ready in 2 s and no prompt waits); a warm cache compiles them while the first prompt waits (about 40 ms each).
* Static chunk of 256 rows instead of 128 (the call costs about 0.6 ms whatever its size; 256 rows reach 1.3-2.6 TOPS INT8 on 8192x2048, 128 rows 1.2-1.9).
* `--device npu` now runs the fast host decoder with the prompt projections on the NPU (same as `FLLM_NPU_HOST=1`); the slow hybrid backend only with
  `--npu-mode all` or `FLLM_NPU_HYBRID=1` (or the `selftest-gpu`, which builds it directly).
* A compiled model is now keyed by the address **and** a fingerprint of the tensor, and idle models are dropped when the weight budget is needed (a model switch in the Studio
  leaves the old model's slots behind). The budget is at most 40 % of the free memory of the computer at start (at least 1 GiB) unless `FLLM_NPU_BUDGET_MB` is set.

**Projection microbenchmark** (`fllm-rs npu-bench`, synthetic weights, median of 8-10 calls, the CPU column is fLLM's own kernel on the same Q4_0 data; the processor numbers
vary by a factor of two between runs on this PC, the NPU numbers by 10 %):

| shape (n x k), batch 256 | weights on the NPU | compile (cold) | NPU ms (GOPS) | CPU ms |
|---|---|---|---|---|
| 2048 x 2048 | INT4 g32 (exact) / g128 / per channel | 1.3 s / 0.6 s / 0.2 s | 1.86 (1150) / 1.60 (1340) / 1.55 (1390) | 6.8-9.0 |
| 8192 x 2048 | INT4 g32 / g128 / per channel | 88 s / 5.3 s / 0.7 s | 9.1 (940) / 5.6 (1540) / 3.3 (2600) | 48-90 |
| 8192 x 2048 | INT8 per channel / FP16 | 1.0 s / 1.3 s | 4.5 (1910) / 6.0 (1430) | |
| 2048 x 8192 | INT4 g32 / g128 / per channel | 4.7 s / 9.1 s / 1.1 s | 20 (430) / 8.9 (970) / 6.3 (1380) | 33-240 |
| 2048 x 8192 | INT8 per channel / FP16 | 1.1 s / 1.8 s | 7.1 (1210) / 6.5 (1320) | |
| 512 x 2048 | INT8 per channel | 0.2 s | 0.94 (570) | |

A call has a fixed cost of about 0.6 ms, so small projections (the K and V projections, 512 x 2048) gain little; the NPU is 3-5 times faster than the processor's kernel on the large ones.

**Whole model, prompt tokens per second of the request** (`fllm-rs bench`, Llama-3.2-1B Q4_K_M and Qwen2.5-0.5B Q4_K_M, 8 threads, runs alternated; this PC's speed drifts by 40-100 %
between runs, so only the pairs inside one series mean something; each cell is a median of 2 repetitions):

| model, prompt | processor | NPU prompt offload | iGPU (Vulkan) |
|---|---|---|---|
| Llama-3.2-1B, ~130 tokens | 108, 140, 149, 154, 185 | 130, 157, 159, 175, 231 | 98, 189 |
| Llama-3.2-1B, ~510 tokens | 83, 90, 95, 140, 152 | 176, 180, 224, 245, 319 | 138, 166 |
| Qwen2.5-0.5B, ~130 tokens | 117 | 105, 144 | 347, 446 |
| Qwen2.5-0.5B, ~510 tokens | 138 | 117, 227 | 289, 375 |

Read: on a 1B model and a 512-token prompt the NPU route reads about twice as fast as the processor alone (neighbouring runs: 2.2x three times) and is in the range of the
iGPU; at 130 tokens the gain is 1.0-1.4x; on the 0.5B model most projections are below the offload size (896 hidden) and the gain is small, and the iGPU is 2-3 times faster than the NPU.
**Decode is unchanged** (A/B with a 16-token prompt, 64 generated tokens: 20.7, 11.0, 21.2 tok/s with the NPU route against 8.3, 13.8, 21.6 without; the spread is the PC, not the route).
The greedy answer to an 807-token prompt (Llama-3.2-1B) is the same text with and without the NPU.

**Verdict for this chip and this model class.** The NPU does not beat the iGPU, and decode never benefits; it beats the processor for prompts of a few hundred tokens and more on models of 1B and up.
It therefore stays opt-in (never chosen by `--device auto`); the use cases are a busy or absent iGPU, or keeping the processor free. Not measured: energy, Lunar Lake and later NPUs, 3B+ models, long prompts (2k+), the effect of the NPU running next to the iGPU.

**Where the time goes** (Llama-3.2-1B, 512 tokens, 16 layers x 7 projections): the NPU computes at about 1.2-1.5 TOPS effective, i.e. roughly 0.5 s of the 1.7 s a 512-token prompt needs; the rest is attention, norms, the output
layer and the f32/f16 conversions on the processor, plus the 0.6 ms per call. `FLLM_NPU_STATS=N` prints the split (conversion to f16, the NPU call, conversion back) every N offloaded projections.

**Unproven or not done:** overlapping the processor and the NPU (the processor idles while the NPU runs a chunk; the two together could add about 15 %); converting the activations once per Q/K/V group; offloading the output layer;
bucketed chunk sizes (one static size is used; a 70-token prompt pays for 256 rows); NF4; Linux and Lunar Lake and later.

## 1. Summary

* The Intel NPU is a low-power matrix multiplier (Meteor Lake / Arrow Lake-U: about 9.5 TOPS INT8, half of that in FP16; Lunar Lake: 48 TOPS).
  It shares main memory with the CPU and the iGPU and needs **static shapes**.
* For fLLM it is worthwhile **only for prefill** (many prompt rows per weight read, compute bound). For **decode**
  (one row per weight, memory-bandwidth bound) it gains nothing and, because of the per-call cost, is more likely slower than the CPU.
* Compared with the **iGPU** (Xe) no speed gain is expected, only energy and offload benefits (in one measurement the NPU drew at most 7 W, the iGPU up to 20 W).
  Anyone who wants speed on an Intel laptop is better served by the iGPU backends (Vulkan, SYCL).
* Therefore **no whole model runs on the NPU**; the implementation is a *matmul offload*: `y = x·Wᵀ` for large projections above a minimum
  row count, with weights as INT4/INT8 (symmetric, group-wise) or FP16 in the OpenVINO graph; everything else stays on the CPU.
* Two ways into the decoder: (a) `--device npu`: a hybrid backend (CPU backend plus the NPU for `quantized_linear`), fully present, but
  using the slow reference CPU path for everything else; (b) a hook in the host decoder (`Linear::forward`) that keeps the fast CPU kernels and hands
  only the prompt matmuls to the NPU. (b) is the more sensible route; `NpuEngine::offload_host` for it exists, but the hook in the decoder is **not wired in this tree** (Section 7).

## 2. Hardware

| NPU (Intel name) | Platform code in the driver | Processors | INT8 peak | Source / status |
|---|---|---|---|---|
| NPU 3720 | `3720` | Core Ultra series 1 (Meteor Lake), presumably also the Core Ultra 200U/200H refreshes with "13 TOPS" | 2 NCE tiles x 2048 INT8 MACs per clock, 1.16 GHz -> 9.5 TOPS (marketing: 11-13) | Chips and Cheese, "Intel Meteor Lake's NPU" (2024) |
| NPU 4 | `4000` | Core Ultra series 2 (Lunar Lake) | 48 TOPS (per the vendor; the tile count was not verified here) | TechPowerUp, Lunar Lake deep dive |
| NPU 5 / 5020 / 6 | `5010`, `5020`, `6010` | later generations | not investigated | platform list in `src/plugins/intel_npu/README.md` (OpenVINO master) |

The target hardware (Core Ultra, Intel NPU of the 3720 class) is **not verified on the NPU**. Whether and which NPU is present is shown by
`fllm-rs devices` (with `--features npu-openvino`; only when the driver and the OpenVINO runtime are installed) and by
`cargo test --features npu-openvino -- --ignored npu_hardware_selfcheck`.

Layout of the 3720 according to Chips and Cheese: two *Neural Compute Engines* (tiles), each with a MAC array (FP16 at half the INT8 rate -> about 4.7 TFLOPS
FP16 peak) and **2 MB of software-managed SRAM**, plus Movidius **SHAVE** DSPs for everything that does not fit the MAC array (FP32,
transcendentals, type conversion). The NPU hangs off the scalable fabric and shares the LPDDR/DDR memory with everything else; the
OpenCL copy throughput was below 10 GB/s (iGPU: > 19 GB/s). Measured FP16 matmul throughput through the NPU acceleration library:
**1.35 TFLOPS** (28 % of peak); for very large matrices (7168^2) it collapsed, and the iGPU value was similar at large sizes.

## 3. What the NPU computes and what we use

The NPU compiler (in the driver or the plugin) translates an OpenVINO graph into static command streams. The MAC arrays (DPU) compute
**matmul/convolution in INT8 or FP16**; element-wise operations, softmax, normalisation and type conversions run on the SHAVE DSPs. Officially
supported data types for computation: FP32 (converted to FP16), FP16, U8 quantisation; the hardware's compute precision is FP16
(OpenVINO documentation, "NPU Device"). For LLMs, OpenVINO GenAI requires **symmetric** 4-bit weights (INT4 or NF4, NF4 only from NPU 4),
per channel or in groups of 128, and FP16 activations.

We use exactly one pattern, the one NNCF/OpenVINO write for compressed weights:

```
Const(i4|i8 [N, G, GS]) -> Convert(f16) -> Multiply(Const f16 [N, G, 1]) -> Reshape [N, K] -+
Parameter x f16 [B, K] ---------------------------------------------------------------------+-> MatMul(transpose_b) -> Result y f16 [B, N]
```

(per channel: no reshape, scales `[N, 1]`; FP16 weights: a single constant). Activations and result are f16; the conversion
to and from f32, the bias and the tensor scale factor stay on the host.

## 4. Limits

* **Static shapes.** Every compiled model has a fixed row count `B` (default 256, `FLLM_NPU_CHUNK`). A prompt is split into full
  blocks; a remainder is zero-padded if it has at least `FLLM_NPU_MIN_BATCH` (64) rows, otherwise the CPU computes it.
* **Compile time.** Every projection is its own model (weights as constants). A model with 32 layers has more than 200 of them.
  The compile time per model is **not measured**, and no reliable figure is known for this model shape; it is the most important measurement point (Section 8). `CACHE_DIR` (default `~/.cache/fllm/npu`, disabled with `FLLM_NPU_CACHE_DIR=`) makes the second start cheaper. Compilation
  happens at the first large prompt, not at load time; there is no warm-up.
* **Memory.** The compiled model holds the (compressed) weights in NPU-accessible memory; the host copy comes on top. In the hybrid backend
  the weights also exist twice (CPU buffer and registration). There is a budget limit (`FLLM_NPU_BUDGET_MB`, otherwise 75 % of
  `NPU_DEVICE_TOTAL_MEM_SIZE`); projections above the budget stay on the CPU.
* **Accuracy.** The NPU computes in FP16 (internal accumulation is not documented). Activations with |x| > 65504 (outside f16) or NaN/Inf, and f16 overflows in the
  result, cause a CPU fallback for that call. Q4_0 and Q8_0 are converted **exactly** (same values as the CPU dequantisation);
  all other formats are **re-quantised** from the dequantised values (INT4 for <= 4.5 bit, otherwise INT8; group 128, `FLLM_NPU_GROUP`).
  Asymmetric K-quant parts (minima) are lost in the process; the asymmetric u4 form would be possible in OpenVINO but is deliberately not
  produced, because the NPU compiler prefers symmetric weights.
* **Formats.** Device-specific layouts are not converted (`Nvfp4Marlin`, `Q4_K_Marlin`, `Fp8E4m3/E5m2` without a Rust dequantiser, `Other`). The
  Q4ZP formats (GPTQ with permutation) are not registered in the hybrid backend.
* **Stateful KV / attention.** OpenVINO GenAI puts LLMs on the NPU by compiling prefill and decode as static graphs with a built-in KV cache
  (`MAX_PROMPT_LEN`, `MIN_RESPONSE_LEN`, prefill in blocks of `NPUW_LLM_PREFILL_CHUNK_SIZE`). That does not fit fLLM's shared,
  paged KV cache (`DeviceKvCache`): it would have to be copied into a static NPU graph. Attention therefore stays on the CPU.
* **Versions.** `ov_element_type_e` was renumbered in OpenVINO 2025.2; the numbers are determined at run time (Section 6). The C API is assumed from 2024.0
  on (`ov_core_read_model_from_memory_buffer`); with older runtimes the path goes through temporary files. NPU driver:
  OpenVINO recommends 32.0.100.3104 or newer (GenAI note); compatibility between OpenVINO and driver versions is not guaranteed.
* **One device.** According to its documentation, the NPU plugin currently supports a single device (`NPU`).

## 5. Expected gain: prefill versus decode

*Everything below is an estimate from specifications and third-party measurements, not a measurement of fLLM.*

**Decode.** Per token every weight is read once: 7 billion parameters in Q4 are about 3.8 GB. At 60-90 GB/s of usable memory bandwidth
(laptop DDR5/LPDDR5) the upper bound is 15-25 tokens/s, regardless of which compute unit forms the sum. The NPU cannot use the bandwidth better
than the CPU (copy throughput < 10 GB/s, see above), and every call has a fixed start-up cost; with 7 projections x 32 layers that would be more than 200
calls per token. Third-party measurements show the same picture: in an OpenVINO discussion (Lunar Lake, Discussion #36484) a small draft model
on the NPU is 3.6x slower per token than on the CPU; a mobile study (Qualcomm, not Intel) describes a "phase split": NPUs
lead in compute-bound prefill, the CPU in memory-bound decode. **Expectation for fLLM: no gain, probably a loss -> decode stays on the CPU.**

**Prefill.** Arithmetically 2 x parameters FLOP per token (7 billion -> 14 GFLOP). At the measured 1.35 TFLOPS (NPU 3720, FP16 matmul, large matrices) that is
at most about 95 tokens/s for the matmuls; at the 4.7 TFLOPS peak about 330 tokens/s; NPU 4 has four times the peak. The CPU
(fLLM's AVX2/AVX-VNNI INT8 kernels, `docs/cpu-kernels.md`) lies, on laptop processors and depending on core count, in a similar or lower
order of magnitude; whether the difference is a factor of 1 or 3 cannot be read from the literature. The planner's default assumptions
(`CPU 400 GOPS`, `NPU 1500 GOPS`, `500 us per call`) are **placeholders** that the measurement plan (Section 8) is meant to replace. Compared with the iGPU
(Xe-LPG/Xe2 with their own matrix units) no prefill gain is expected: Chips and Cheese found the iGPU similarly fast on large matmuls
and 62 % faster than the NPU on the Stable Diffusion workload (FP32 model on the iGPU against an INT8 blob on the NPU).

**Conclusion.** A gain over the CPU is plausible for large prompts (>= 128 tokens) and NPU 4, uncertain on the NPU 3720; compared with the
iGPU there are at most energy and offload benefits (the CPU stays free for the rest). The decision to build or not should only be made
after the measurement in Section 8. The implementation is cut so that it can be switched on without touching the CPU paths
(`FLLM_NPU=0` disables every offload; the default build contains none of it).

## 6. Structure of the implementation

Files (all under `rust/engine/src/backend/npu/`, only with the `npu-openvino` feature):

| File | Contents |
|---|---|
| `openvino/ffi.rs` | `dlopen`/`LoadLibrary` of `libopenvino_c.so` / `openvino_c.dll` (no linking), bindings of the required C functions, library search (`FLLM_OPENVINO_C_LIB`, `FLLM_OPENVINO_DIR`, `INTEL_OPENVINO_DIR`, `LD_LIBRARY_PATH`, `PATH`, `VIRTUAL_ENV`, `/opt/intel/openvino*`), determination of the element type numbers |
| `openvino/session.rs` | RAII wrappers: core, tensor, compiled model + inference request; reading IR from memory (fallback via files), `compile_model` with properties (variadic) |
| `openvino/ir.rs` | writer for OpenVINO IR v11 (XML + BIN), only our own test parsers, no dependency |
| `openvino/model.rs` | the matmul graph (Section 3) |
| `openvino/quant.rs` | conversion from fLLM formats to INT4/INT8/FP16 for the NPU |
| `openvino/plan.rs` | `OffloadPlan`, `should_offload`, `plan` (selection of projections under a memory budget) |
| `openvino/engine.rs` | `NpuEngine`: registration, deferred compilation, block-wise execution, CPU fallback, `offload_host`, `selfcheck` |
| `openvino/devices.rs` | device list (`FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE`, `NPU_DRIVER_VERSION`, `NPU_DEVICE_TOTAL_MEM_SIZE`) |
| `backend.rs`, `backend/hybrid.rs` | provider selection (`--device npu[:N]`) and the hybrid backend |
| `device_info.rs` | device inventory of all providers |

**Format conversion.**

| fLLM format | NPU form | Accuracy |
|---|---|---|
| Q4_0 | INT4 symmetric, group 32, f16 scale | exact (bit-identical to the CPU dequantisation, tested) |
| Q8_0 | INT8 symmetric, group 32 | exact (tested) |
| Q4_1, Q2_K, Q3_K, Q4_K, IQ*, MXFP4, NVFP4, Q4Z* | INT4 symmetric, group `FLLM_NPU_GROUP` (128), RTN | re-quantised |
| Q5_0, Q5_1, Q5_K, Q6_K, Q8_1, Q8_K, FP8M | INT8 symmetric, group 128 | re-quantised, error about half an INT8 step |
| F16, BF16, F32 | FP16 | value range checked (Inf -> rejection) |
| odd column count with 4-bit formats | INT8 instead of INT4 | as above |

**Decision per call** (`OffloadPlan::should_offload`): projection >= `FLLM_NPU_MIN_ELEMS` (1 Mi) weights and convertible; at least
`FLLM_NPU_MIN_BATCH` rows in a block or remainder; estimated NPU time (call time + compute time at `FLLM_NPU_GOPS`) below the CPU time
(`FLLM_NPU_CPU_GOPS`). Errors (compilation, execution, value range) fall back silently to the CPU (logged once); three compile errors
in a row switch the NPU off for the process.

**Environment variables.**

| Variable | Effect |
|---|---|
| `FLLM_NPU=0` | offload off |
| `FLLM_NPU_HOST=1` | host-decoder hook active (Section 7) |
| `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_CHUNK`, `FLLM_NPU_MIN_ELEMS`, `FLLM_NPU_BUDGET_MB` | planner, Section 6 |
| `FLLM_NPU_WEIGHTS` | `int8` (default: INT8 per channel), `int4` (INT4 in groups of 128) or `exact` (Q4_0/Q8_0 bit for bit); see section 0 |
| `FLLM_NPU_STATS` | log a summary line (with the time split) every N offloaded projections |
| `FLLM_NPU_COMPILE` | `inline` or `background`: force how models are compiled (default: decided by the first compile) |
| `FLLM_NPU_HYBRID` | `--device npu` uses the hybrid backend instead of the host decoder (`--npu-mode all` sets it) |
| `FLLM_NPU_GROUP` | group size of re-quantised weights (0 = per channel); also disables the exact Q4_0/Q8_0 conversion |
| `FLLM_NPU_CPU_GOPS`, `FLLM_NPU_GOPS`, `FLLM_NPU_OVERHEAD_US` | rates of the cost model (placeholders) |
| `FLLM_NPU_CACHE_DIR` | directory for OpenVINO's `CACHE_DIR` (empty = off) |
| `FLLM_NPU_PROPS` | additional compile properties `KEY=VAL,KEY=VAL` (at most two; e.g. `NPU_TILES=2`) |
| `FLLM_NPU_DEVICE` | another OpenVINO device name instead of `NPU` (only for exercising the flow, e.g. `CPU`) |
| `FLLM_NPU_DUMP_IR` | directory into which every compiled IR is written as `.xml`/`.bin` |
| `FLLM_OPENVINO_C_LIB`, `FLLM_OPENVINO_DIR` | location of the runtime |

**What was verified without hardware:**
* `cargo check`/`clippy -D warnings` without the feature and with `npu-openvino`, `cargo check` for `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-gnu`,
  `aarch64-apple-darwin` (with `npu-openvino`) and with all four NPU features;
* 39 unit tests (IR XML well-formed and consistent, offsets in the BIN, Q4_0/Q8_0 conversion exact, planner, hybrid backend against the CPU backend);
* **IR parser check with the real OpenVINO runtime (pip package `openvino==2026.4.1`, CPU plugin only)**: `read_model` accepts all five
  example graphs; after `ConstantFolding` (OpenVINO's own code, no device) the unfolded f16 weights match the Rust dequantisation
  element by element (largest difference = f16 rounding). This proves nibble order, scale layout, reshape and grouping.
  For this the test `openvino_reader_accepts_generated_ir` (ignored, needs the library) loads the C API through our `dlopen` path,
  determines the type numbers (2026.4.1: f16 = 3, f32 = 4), counts devices (`CPU`) and lets the reader check the graphs and one deliberately broken one.
  **Not** verified: `compile_model` (not even on the CPU plugin), inference, the NPU compiler.

## 7. Integration into the decoder

**What `rust_decoder.rs` can do:** `init_backend` takes *one* complete `Backend`; there is no mechanism to move individual operations to another device
(`docs/gpu-backends.md` describes a fallback with `warn_fallback_once`; no caller of it was found in the decoder; only
individual paths, such as recurrent layers and `host_linear` for streamed weights, run on the host). Two routes:

1. **`--device npu` (done, hybrid backend).** `create_native_backend("npu", N)` returns `NpuHybridBackend`: the reference CPU backend (`backend/cpu.rs`) with
   `quantized_linear` going through the NPU engine. The rest (norms, RoPE, attention, KV write, sampling) is the reference implementation of the CPU backend and
   **single-threaded and not optimised**; decode is therefore much slower than with `--device cpu`. Rows below the threshold are computed by the fast CPU kernel
   (`cpu_quantized::linear`) from the registration copy of the weights. Restrictions: dense models only (the CPU stream has neither delta nor gate kernels nor
   `argmax_rows`; `backend_unsupported` rejects them), doubled weight memory, no MoE acceleration. `selftest-gpu --device npu` checks the backend
   against the CPU backend (note: the CPU backend itself already reports 7 failed q8 attention checks there, because the tolerance is tighter than the q8 cache error; the unit test of the hybrid backend asks for the same verdict as for the CPU backend); with `FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1` every check runs on the NPU (every shape compiles its own model).
2. **Hook in the host decoder (built in, off by default).** `NpuEngine::offload_host(quant, raw, rows, cols, x, batch)` takes the mapped GGUF bytes, copies nothing and
   returns the first `rows_done` rows; the rest runs as before on the fast CPU kernels. It sits in `Linear::forward` (`compute.rs`) before the
   `fast_cpu` branch and in `Linear::forward_group` and `forward_group_with_workspace` (the grouped Q8/K kernels share one activation packing; for NPU-suitable batches the computation goes
   projection by projection). Activation: the feature and `FLLM_NPU_HOST=1`.
   Checked on an OpenVINO CPU device (`FLLM_NPU_DEVICE=CPU`), never on the NPU. Open points: the hook sits in layer loops that must take bias and scale into account
   (included); MoE experts (`forward_experts`) and the output head (`head`) are not covered.

## 8. Verification and measurement plan

**L1 (does it run?)** - Prerequisites: NPU driver (Linux: intel/linux-npu-driver; Windows: Intel NPU driver), OpenVINO >= 2025 with the NPU plugin
(`pip install openvino` or an archive).
1. `python3 -c "import openvino as ov; print(ov.Core().available_devices)"` must contain `NPU`.
2. `FLLM_OPENVINO_C_LIB=.../libopenvino_c.so... cargo test --release -p fllm-engine --features npu-openvino -- --ignored --nocapture` (two ignored tests:
   the reader check without a device, and `npu_hardware_selfcheck` with a device: compiles a 192x512 Q4_0 projection, compares with the host computation, threshold 2 %).
3. `fllm-rs devices` (entry "OpenVINO NPU, platform ...").
4. `FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1 fllm-rs selftest-gpu --device npu --json npu.json`.
5. On errors: `FLLM_NPU_DUMP_IR=/tmp/ir` and check the files with `benchmark_app -m x.xml -d NPU` and `ov.Core().compile_model(model, "NPU")`; `OV_NPU_LOG_LEVEL=LOG_INFO`.

**L2 (correct text?)** - Compare the greedy text of a small Q4_0 model (e.g. 1-3 B) with `FLLM_NPU=0` and with `FLLM_NPU_HOST=1` (or `--device npu`);
deviations in the logit range from the f16 computation are expected (the argmax may only flip within the noise range). In addition log the NPU-versus-CPU deviation per projection.

**L3 (faster?)** - Measurements in this order; the results replace the planner's three placeholders:
1. *Matmul microbenchmark*: 4096x4096 and 14336x4096, Q4_0, batch 16/32/64/128/256/512, on the NPU (`engine.run`), the CPU (`cpu_quantized::linear`) and the iGPU
   (Vulkan/SYCL backend, if present): time per call, GOPS; from this `FLLM_NPU_GOPS`, `FLLM_NPU_CPU_GOPS`, `FLLM_NPU_OVERHEAD_US`.
2. *Compile time* per projection shape without and with the cache, and total time to the first token for a 7B model; memory use (RSS and
   `NPU_DEVICE_ALLOC_MEM_SIZE`); the largest number of simultaneously loaded models before the driver refuses.
3. *Prefill throughput* (tokens/s) for prompts of 128/512/2048 tokens: CPU alone, CPU + NPU, iGPU alone; plus energy (RAPL / `powertop` / Windows power meter).
4. *Decode* with and without offload (must stay the same; proof that the threshold does not touch decode).
5. *Accuracy*: perplexity on a fixed text and logit distance (max/RMS) against the CPU, for Q4_0 (exact weights, only the f16 computation) and Q4_K_M
   (re-quantised).
6. *Group size*: group 32 (exact) against 128 and per channel: speed and accuracy.
Go/no-go: the offload stays only if (3) shows at least 1.3x over the CPU alone for prompts >= 512 tokens, or needs clearly less energy at equal speed;
otherwise `FLLM_NPU` stays off by default and the feature remains an experiment.

## 9. Unverified assumptions and risks

* The NPU compiler recognises the decompression pattern at **group 32** and with **FP16 input/output** just as at group 128 (NNCF default); group 32 may be slow.
* `Reshape` after `Multiply` on the NPU: NNCF also produces this pattern; support by all driver versions is unverified.
* Properties at compile time: only `CACHE_DIR` (and the user list); `PERFORMANCE_HINT`, `NPU_TILES`, `NPU_COMPILATION_MODE_PARAMS` (e.g. `optimization-level=2`)
  are not set because their effect is unknown.
* Concurrent access: there is exactly one inference request per projection under a mutex; the scheduler calls prefill projections one after another, and parallelising
  across projections has not been attempted.
* The Windows path handling (`LOAD_WITH_ALTERED_SEARCH_PATH`, TBB directory in `PATH`) only compiles for `x86_64-pc-windows-gnu`, not for MSVC, and has not been executed.
* The hybrid backend has never been run with a model; which models `init_backend` accepts follows from `backend_unsupported` and the capabilities of the CPU stream.
* The assumptions about the performance figures in Section 5 rest on specifications and third-party measurements of other software stacks.

## 10. Sources

* OpenVINO, `src/bindings/c/include/openvino/c/*.h` (master 2026.x; tags 2023.3, 2024.0, 2025.0, 2025.2, 2025.4 for the numbering of `ov_element_type_e` and the read functions).
* OpenVINO, `src/plugins/intel_npu/README.md` (properties, platform codes), documentation "NPU Device" and "OpenVINO GenAI on NPU" (docs.openvino.ai/2025).
* OpenVINO, `src/frontends/ir/tests/frontend_test_basic.cpp` (IR example with a u4 constant), `src/core/dev_api/openvino/core/type/element_iterator.hpp` (packing order of the 4-bit types).
* Chips and Cheese, "Intel Meteor Lake's NPU" (April 2024); TechPowerUp, "Intel Lunar Lake Technical Deep Dive".
* OpenVINO GitHub Discussion #36484 (NPU draft model on Lunar Lake); arXiv 2607.05475 (mobile NPU study, prefill/decode phase split - Qualcomm hardware, only as a pointer).

## Setting up the runtime: `fllm-rs npu-setup`

`fllm-rs npu-setup` downloads the pinned OpenVINO runtime (the wheel `openvino`, which carries `libopenvino_c` and the NPU plug-in, from
`files.pythonhosted.org`, checked against a pinned SHA-256; no Python needed) and unpacks `openvino/libs` into
`~/.fllm/openvino` (`$FLLM_HOME/openvino`); `--remove` deletes it. The Studio does the same from Settings, Hardware (`setup.rs`, `status.rs`). The backend finds it there without further settings, as it finds a virtual environment
(`VIRTUAL_ENV`), the user and system Python site directories (`~/.local/lib/python3.*/site-packages/openvino/libs`), `/opt/intel/openvino*`
and `FLLM_OPENVINO_C_LIB` / `FLLM_OPENVINO_DIR`. It needs a network connection. The kernel driver (`intel_vpu`, with its
user-space part `intel-level-zero-npu` / `libze_intel_npu`) comes from the distribution, and the user needs access to `/dev/accel/accel0`
(group `render`). Then `fllm-rs devices` lists the NPU and `fllm-rs selftest-gpu --device npu` checks it; select it with `--device npu`.
Measured on a Core Ultra 5 235U (Meteor Lake NPU) with a 1.5B Q4_K_M model: the selftest passes (606 checks, 0 failed) and the answers are
correct, but the hybrid path is far slower than the CPU (the CPU part of the work runs through the generic backend interface on one thread).
