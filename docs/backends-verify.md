# Verification guide for other hardware

This guide takes the backends from **L0** (compiled) to **L1/L2/L3** (see [test-levels.md](test-levels.md) and [backends.md](backends.md),
section 1). It consists of ready-to-copy commands per platform and of scripts (`scripts/verify-<backend>.sh` and `.ps1`) that run the same
sequence automatically and write the result to `report.json`.

**Honest state:** apart from the runs named in the README Status table and in [ci.md](ci.md), neither these commands nor the scripts have
run on the target hardware. The scripts have been checked with `bash -n` and the PowerShell parser, nothing more. A first run with errors is
expected and wanted; the failure patterns in section 4 say where to look first.

## 0. Preparation

* Source: a checkout of the current main branch. The verify scripts call the helper scripts of each backend where they are in the tree and
  report "skipped" with the script name otherwise. The hooks of the HIP, SYCL and OpenVINO backends are built into the tree; a backend whose
  toolchain is missing at build time is a stub (see docs/backends.md).
* Rust (rustup, `cargo`). Per backend additional requirements: see the section of the platform.
* A **small model**: the example used here is `hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf`. For Q4 formats create
  them from it with `fllm-rs quantize IN.gguf OUT.gguf q4_0` (or `q4_k_m`). For MoE, DeltaNet (Qwen3.5/3.6) and Mamba use one model each; which
  formats a GGUF contains is shown by `fllm-rs inspect MODEL.gguf`. `fllm-rs pull hf://owner/repo/file.gguf` downloads a model and prints the path.
* Do not run several processes on the same device at once (TPU: one process per chip).

The scripts write to `verify-out/<backend>/`: one log per step, `report.json`, server logs and the texts. **What to send back:** the whole
directory (or the files named in the respective section) plus the "device" line from `fllm-rs devices`. For the format of a hardware report
see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 1. Common sequence (by hand)

The five steps are the same for all GPU/accelerator backends. `<F>` is the Cargo feature, `<DEV>` the device specification (`vulkan:0`,
`metal:0`, `hip:0`, `sycl:0`, `npu`, `tpu`, `auto`). `<backend>:N` counts the devices of *that* backend from 0; the number `[n]` from
`fllm-rs devices` is **not** meant, the `Select:` line names the right value ([device-selection.md](device-selection.md)).
`vulkan-native:N`/`metal-native:N` force the native backend (no silent fallback to wgpu), `wgpu-vulkan:N` etc. the wgpu backend.

### 1.1 Build

```sh
cargo build --release -p fllm-engine --features <F> --bin fllm-rs      # binary: target/release/fllm-rs (Windows: fllm-rs.exe)
```

### 1.2 Is the device visible?

```sh
target/release/fllm-rs devices                 # the "Select:" line names the value for --device; also: devices --json [--measure]
```
Expected: one entry per device; native ones are recognisable by `native, Vulkan x.y.z, ...` or `Metal (native), ...`; AMD under HIP appears as
"CUDA/ROCm, N CUs". No entry means the driver/runtime is missing (not an error).

### 1.3 Self test

```sh
target/release/fllm-rs selftest-gpu --device <DEV> --json out.json
```
Expected: exit 0, `failed = 0`. Each line gives the largest error of the device and of the CPU relative to the error scale of the element, the
tolerance and the time ([gpu-backends.md](gpu-backends.md) section 7). `out.json` contains `checks`, `failed`, `skipped`, `ok`, `error` and
every single check; further options: `--list` (groups), `--only GROUP[,...]`, `--quants q4_0,q8_0`, `--strict`.
`SKIPPED (unsupported)` means "the backend does not offer it" and is **not** a pass.
Run a second pass with the fast kernels where they exist (CUDA/HIP does this itself; for Vulkan see section 2.2, for Metal
`FLLM_METAL_EXACT=1` against the default).

### 1.4 Text against the CPU reference (L2)

```sh
BIN=target/release/fllm-rs; MODEL=/path/to/model.gguf
ASK='{"model":"m","messages":[{"role":"user","content":"Write one sentence about the sea."}],"temperature":0,"max_tokens":64}'
$BIN serve --model "$MODEL" --device cpu --port 18001 --ctx 2048 --max-seqs 2 & PID=$!
until curl -sf http://127.0.0.1:18001/v1/models >/dev/null; do sleep 2; done
curl -s http://127.0.0.1:18001/v1/chat/completions -H 'Content-Type: application/json' -d "$ASK" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > cpu.txt
kill $PID

$BIN serve --model "$MODEL" --device <DEV> --port 18002 --ctx 2048 --max-seqs 2 & PID=$!
until curl -sf http://127.0.0.1:18002/v1/models >/dev/null; do sleep 2; done      # the first start takes a while (pipelines/shaders/programs)
for i in 1 2; do                                                                    # the first request warms up
  curl -s http://127.0.0.1:18002/v1/chat/completions -H 'Content-Type: application/json' -d "$ASK" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > dev.txt
done
kill $PID
cmp cpu.txt dev.txt && echo "IDENTICAL" || echo "DIFFERENT"
```
Windows (PowerShell), for each server after it has started:
```powershell
$body = '{"model":"m","messages":[{"role":"user","content":"Write one sentence about the sea."}],"temperature":0,"max_tokens":64}'
(Invoke-RestMethod -Uri http://127.0.0.1:18002/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body).choices[0].message.content | Set-Content dev.txt
```
Expected: **identical in exact mode** (Vulkan `FLLM_VULKAN_EXACT=1`, Metal `FLLM_METAL_EXACT=1`, HIP `FLLM_CUDA_EXACT=1`, SYCL
`FLLM_SYCL_EXACT=1`, TPU `FLLM_TPU_EXACT=1`). In the fast default mode a later token can differ (Vulkan: int8 activations, about 0.4 %, and
f16 tiles; Metal: half tiles and fast `exp`/`tanh`; HIP: FP16 rounding of the matrix-core paths; SYCL/XMX: fp16 inputs). In that case note the
position of the first difference (`cmp` prints the byte) instead of reporting "failed".
A matching text alone does not prove that the backend was used: read the server log (for `vulkan`/`metal`, `Decoder::set_device` falls back to
wgpu on error; the verify scripts record `log_mentions_wgpu` in the report when the log mentions "wgpu" - an indication only).

### 1.5 Benchmark

```sh
$BIN serve --model "$MODEL" --device <DEV> --port 18002 --ctx 4096 --max-seqs 2 &        # as above
$BIN bench --url http://127.0.0.1:18002 --scenarios prefill,decode-single --prompt-lens 512 --gen-len 128 \
           --warmup 1 --repeats 3 --label "<hardware>" --json bench-<hardware>.json
$BIN bench-report bench-cpu.json bench-<hardware>.json                                  # Markdown comparison
```
Scenarios according to `fllm-rs bench --help`: `prefill`, `decode-single`, `decode-context`, `concurrency`. **L3** additionally needs a run of the
reference implementation on the *same* hardware (llama.cpp with Vulkan/Metal/HIP/SYCL; vLLM on ROCm); without it a measurement is not L3. The
`tok_s` in the scripts' `report.json` is a rough value from one chat request (including the prompt), not a benchmark. See
[benchmarking.md](benchmarking.md).

### 1.6 The scripts

| Backend | Script (`.sh` and `.ps1`) | Features | Calls (if present in the tree) |
| --- | --- | --- | --- |
| Vulkan native | `verify-vulkan` | `vulkan` | `shaders/vulkan/check_all.sh`, smoke and hardware tests |
| Metal | `verify-metal` | `metal` | `metal_core_smoke`, `metal_quant_kernels`, `metal_ops_kernels` |
| ROCm/HIP | `verify-hip` | `hip` | `scripts/hip-verify.sh` |
| Intel SYCL | `verify-sycl` | `sycl` | test `linear_matches_cpu` |
| Intel NPU (OpenVINO) | `verify-npu-openvino` | `npu-openvino` | Python device check, `npu_hardware_selfcheck`, optionally the XDNA/Hexagon/ANE probes (`V_NPU_PROBES`) |
| TPU | `verify-tpu` | `tpu-pjrt` | `scripts/colab/tpu_check.py` |
| ARM CPU | `verify-arm` | - | `scripts/check-arm-cpu.sh` |
| x86 AVX-512/AMX | `verify-avx512` | `avx512` | `scripts/check_x86_ext.sh` |
| Windows | `verify-windows` | `rust-gpu,vulkan` (through `build-windows.ps1`) | `scripts/build-windows.ps1` |

Common options (bash: `--model`, PowerShell: `-Model`; both spellings work in the `.ps1` files):
`--model FILE`, `--model-uri URI`, `--bin PATH` (finished binary, skips the build), `--device NAME`, `--out DIR`, `--port N`, `--tokens N`,
`--prompt TEXT`, `--skip build,devices,pre,selftest,text,bench`, `--dry-run` (prints the plan and runs nothing). The phases and their order are
given in the header of `scripts/verify-common.sh` (bash 3.2 compatible for macOS) and `scripts/verify-common.ps1`.

`report.json` contains: `backend`, `device`, `commit`, `tree_dirty`, `command`, `features`, `model`, `host`, `date_utc`, `exit_code`,
`duration_seconds`, `tok_s` and `steps[]` with `name`, `command`, `exit_code`, `seconds`, `log`, `note` each (text cases additionally
`text_vs_cpu` and `request` with seconds, tokens and `tok_s`). The exit code of the script is 1 as soon as a step failed; a differing text does
not count as a failure (see 1.4). If the self test fails, the script repeats it once per switch from the backend's switch list
(`selftest-with-<SWITCH>`): **note which switch removes the error.**

## 2. Platforms

### 2.1 Mac (Metal, Apple silicon)

* Requirements: macOS (residency sets and MSL 3.2 need macOS 15), Rust, the Apple SDK (Xcode or Command Line Tools; without the SDK Metal
  cannot be linked). M1 to M5; one line per chip.
* Build: `cargo build --release -p fllm-engine --features metal --bin fllm-rs`
* Script: `scripts/verify-metal.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf`

Order by hand:

| # | Command | Expected |
| --- | --- | --- |
| a | `cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture` | 9 lines `[ok]`/`[skip]`/`[FAIL]` |
| b+c | `FLLM_METAL_PREWARM=1 FLLM_METAL_DEBUG=1 cargo test --release -p fllm-engine --features metal metal_quant_kernels -- --nocapture` | first read the **compiler messages** per `.metal` file; then per line `[ok]`/`[FAIL] <format> batch <n> <kernel>: max rel err ...` |
| d | `cargo test --release -p fllm-engine --features metal metal_ops_kernels -- --nocapture` | `[ok]` per operation; note the `device: ...` line |
| e | `target/release/fllm-rs devices` and `selftest-gpu --device metal:0 --json metal.json` | entry "Metal (native), ..." with chip, families, memory |
| f | text against the CPU (1.4) with a dense Q8_0/Q4_0 model, then `FLLM_METAL_NO_NOCOPY=1` against the default (real file mapping) | identical in exact mode (`FLLM_METAL_EXACT=1`) |
| g | MoE and DeltaNet model against the CPU | as f |

On errors try these switches in order and note which one removes the error: `FLLM_METAL_SERIAL=1` (rules out barrier errors),
`FLLM_METAL_FAST_MATH=0`, `FLLM_METAL_EXACT=1` (float instead of half tiles), `FLLM_METAL_NO_NOCOPY=1`, `FLLM_METAL_SINGLE_LIB=1` (source
layout), `FLLM_METAL_MSL=3.0` or `3.1`, `FLLM_METAL_RESIDENCY=0`.
Failure patterns: only the `_h` kernels wrong means half tiles/`simdgroup_matrix`; only one format wrong means the decoder (it runs correctly on
the CPU in `metal/emu/`); only batch > 8 means the matmul; everything wrong means the parameter block or binding. Only the prefill cases of
attention wrong means the assumption about `simdgroup_load(..., transpose=true)`. CPU emulation for cross-checking: `rust/engine/metal/emu/`,
`emu_ops/`, `emu_recurrent/` (each with a README; needs `python3`, clang++ and an MSL shim header).

Record per chip: name, `summary()` (the "Metal (native), ..." line from `devices`), `recommendedMaxWorkingSetSize`, the result of the three
tests, tokens/s. The bandwidth figures in the description are estimates (M3 Max, M4 Max, M5 uncertain).

### 2.2 AMD (Vulkan and ROCm/HIP)

**Vulkan native** (Linux RADV, Windows AMD driver). Requirements: a Vulkan driver, `glslc` (Vulkan SDK) when building; `check_all.sh` also needs
`spirv-val`.
```sh
cargo build --release -p fllm-engine --features vulkan --bin fllm-rs
scripts/verify-vulkan.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```
Order by hand, everything with `FLLM_VULKAN_VALIDATE=1`:

| # | Command | Expected |
| --- | --- | --- |
| a | `rust/engine/shaders/vulkan/check_all.sh` | `modules: 171 ... FAIL=0` (compiles and validates all shaders with the local `glslc`) |
| b | `FLLM_VULKAN_VALIDATE=1 FLLM_VULKAN_VERBOSE=1 cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::instance::smoke -- --ignored --nocapture --test-threads=1` | 7 tests green, validation layer without errors; `context_reports_capabilities` prints device, caps, cooperative-matrix shapes |
| c | the same with `backend::vulkan::backend::hw_tests` | 7 tests green (among them `a_decode_step_is_recorded_into_one_submission`) |
| d | `fllm-rs devices` | one entry `native, Vulkan ...` per device |
| e | `FLLM_VULKAN_EXACT=1 fllm-rs selftest-gpu --device vulkan:0 --json exact.json`, then without `FLLM_VULKAN_EXACT` | exact: `failed = 0`. The fast pass (`_i` integer dot, `_c` coopmat) does not necessarily meet the FP32 tolerance; whether `_i`/`_c` pass with a relaxed tolerance (1 % for `_i`, 0.5 % for `_c` was proposed) is shown only by the hardware |
| f | small Q4_0/Q8_0 model, greedy, against the CPU (1.4): first with `FLLM_VULKAN_EXACT=1`, then without | identical in exact mode; the fast mode may differ |
| g | narrowing down: `FLLM_VULKAN_NO_COOP=1`, `FLLM_VULKAN_NO_IDOT=1` | the switch that removes the error names the variant family |

Record per device: `VkCaps::describe()` (printed on open with `FLLM_VULKAN_VERBOSE=1`), `push_descriptor`, `unified_memory`,
`external_memory_host`, `int_dot`, `subgroup_size`, `max_storage_buffer_range`, number of cooperative-matrix shapes (`vulkaninfo | grep -A6
CooperativeMatrix` for the shapes), duration of the first pipeline creation, number of submits of one decode step
(`VulkanStream::recorder_stats()`), and that the file `vk-pipeline-v1-<vendor>-<device>-<uuid>.bin` appears in the cache directory and a second
run loads it. Suspect list on errors: pipeline creation (specialisation constants, subgroup pinning), the binding check in `Recorder::set_buffers`,
`lin_float`, rings, KV windows, `attn_prefill_*`, `kv_write_q8`, `moe_linear_*`/`embed_*` (layout assumption), `delta_conv`/`delta_rec_*`, the
expert arena. Reproduce shader errors without a GPU with `rust/engine/shaders/vulkan/emu/run_kernels.sh` (see `emu/README.md`).
To test the slicing logic for large weights use a device with only 128 MiB `maxStorageBufferRange`, for example lavapipe with
`FLLM_ALLOW_SOFTWARE_GPU=1`.

**ROCm/HIP** (RDNA3 gfx11, RDNA4 gfx12, CDNA gfx908/90a/942/950; ROCm 6.2 or newer, 7.x recommended; gfx950 needs 6.4 or newer and belongs in
`FLLM_HIP_ARCHS`):
```sh
FLLM_HIP=1 ROCM_PATH=/opt/rocm FLLM_HIP_ARCHS="gfx1100;gfx90a" cargo build --release -p fllm-engine --features hip --bin fllm-rs
scripts/hip-verify.sh MODEL.gguf target/release/fllm-rs hip:0                                       # helper script of the HIP backend
scripts/verify-hip.sh --model MODEL.gguf                                                            # wrapper: build, devices, bench, report.json
```
Expected: (1) `devices` shows "CUDA/ROCm, N CUs" with a name `... (gfx....)` and no second entry; at start a log line
`[fllm-rs] hip: device 0 is gfx... (waveNN, WMMA|MFMA)`. (2) `selftest-gpu --device hip` (exact, then fast). (3) Greedy text CPU against GPU
with all switches: `FLLM_HIP_NO_MMA=1` (rocWMMA/FP32 instead of MFMA/WMMA), `FLLM_HIP_NO_SKINNY=1`, `FLLM_HIP_NO_Q4Z=1` (AWQ/GPTQ as F32
reference), `FLLM_CUDA_NO_ATTN4=1`, `FLLM_CUDA_EXACT=1`. One model per kernel family: F16 GGUF (skinny + MMA), Q4_K/Q8_0 (MMA from 16 tokens),
AWQ/GPTQ (Q4Z), a GQA model with G = 4 or 8 (DPP attention). Suspects: lane layouts in `hip_mma.cuh`, DPP control codes (0xB1, 0x4E, 0x141,
0x140), nibble order in `hip_q4z.cu`, LDS size of the skinny kernels, register pressure (`-Rpass-analysis=kernel-resource-usage`). **The script
`hip-verify.sh` has never been run on AMD hardware**; the first AMD run is its first run.

### 2.3 NVIDIA (native Vulkan, with CUDA as a cross-check)

On NVIDIA the CUDA backend is the established path; native Vulkan is the alternative. Requirements as in 2.2 (Vulkan). The cross-check compares
three paths on the same card:
```sh
cargo build --release -p fllm-engine --features vulkan,cuda,rust-gpu --bin fllm-rs
fllm-rs selftest-gpu --device vulkan-native:0 --json vk.json     # native
fllm-rs selftest-gpu --device wgpu-vulkan:0   --json wgpu.json   # the wgpu backend
fllm-rs selftest-gpu --device cuda:0          --json cuda.json   # CUDA
scripts/verify-vulkan.sh --model MODEL.gguf
```
Notes: coopmat only with 16x16x16 f16/f32 at subgroup scope; `VK_NV_cooperative_matrix` (older NVIDIA drivers) is not used. Switches as in 2.2.
Speed (L3): against CUDA and against llama.cpp-Vulkan on the same card.

### 2.4 Intel: iGPU (Vulkan, SYCL) and NPU

**Intel iGPU/Arc through Vulkan** (Linux ANV, Windows): as in 2.2 (`scripts/verify-vulkan.sh`). On an iGPU the CPU is often the faster way on
x86, and the inventory cannot know that (`FLLM_AUTO_IGPU=0`).

**Intel GPU through SYCL:** oneAPI (`source /opt/intel/oneapi/setvars.sh`) and a Level Zero driver; `git apply rust/engine/sycl/build_rs.patch`
is a prerequisite for the native backend (otherwise a stub).
```sh
cargo build --release -p fllm-engine --features sycl --bin fllm-rs
fllm-rs devices                                              # must show the device including the XMX shape
fllm-rs selftest-gpu --device sycl:0 --json sycl.json
cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu   # covers all batch paths (tile, XMX); the self test only reaches batch <= 5
scripts/verify-sycl.sh --model MODEL.gguf
```
Troubleshooting: `ONEAPI_DEVICE_SELECTOR=level_zero:0` (first run), `FLLM_SYCL_EXACT=1` (XMX off),
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (single allocation > 4 GiB; the variable name is an assumption taken from third-party documentation),
`FLLM_SYCL_ALL_BACKENDS=1`, `FLLM_VULKAN_VALIDATE=1` (Vulkan cross-check). Open assumptions: XMX shapes (8x16x16 / sub-group 16 and 8x8x16 /
sub-group 8), VNNI layout of the B tile, register spills with Q6_K/Q3_K/Q2_K. Windows: the `icx` branch of `build.rs` has never run. No
DeltaNet models (`supports_delta` = false). Measurement plan for L3: see [research/README.md](research/README.md) (Vulkan against SYCL).

**Intel NPU through OpenVINO** (Core Ultra with NPU, Linux or Windows):
```sh
python3 -c "import openvino as ov; print(ov.Core().available_devices)"       # the NPU must appear
cargo test --release -p fllm-engine --features npu-openvino --lib npu_hardware_selfcheck -- --ignored --nocapture   # 192x512 Q4_0 against the host, threshold 2 %
fllm-rs devices
FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1 fllm-rs selftest-gpu --device npu --json npu.json
scripts/verify-npu-openvino.sh --model MODEL.gguf
```
Expected: the same **7 failed q8 attention checks as with the CPU backend** (tolerance tighter than the q8 cache error), otherwise `ok`. Without
the lowered thresholds (default `FLLM_NPU_MIN_BATCH` = 64, `FLLM_NPU_MIN_ELEMS` = 1 Mi) the self test never reaches the NPU; every shape compiles
its own model (this takes time). L2: text with `FLLM_NPU=0` against `--device npu` or (after the host-hook patch) `FLLM_NPU_HOST=1`; f16
computation can shift logits slightly. On errors: `FLLM_NPU_DUMP_IR=/tmp/ir`, then `benchmark_app -m x.xml -d NPU` or
`ov.Core().compile_model(ov.Core().read_model("x.xml"), "NPU")`, `OV_NPU_LOG_LEVEL=LOG_INFO`. The pip package `openvino` is enough for a pure
parser test without an NPU (`cargo test ... -- --ignored openvino_reader`). Expected gain: prefill only, no decode gain; the rates of the cost
model are placeholders that L3 replaces (measurement plan in the OpenVINO notes under [research/](research/README.md)).

**AMD XDNA, Qualcomm Hexagon, Apple ANE** are probes only (one MatMul through the vendor stack), not a compute path; "runs" must not be claimed
anywhere:
```sh
cargo run --release -p fllm-engine --example npu_probe --features npu-xdna -- list          # AMD Ryzen AI: install Ryzen AI Software; then: -- xdna 1 4096 4096 and -- xdna 128 4096 4096
cargo run --release -p fllm-engine --example npu_probe --features npu-qnn  -- hexagon 1 4096 4096   # Snapdragon: QAIRT SDK, QNN_SDK_ROOT (or FLLM_QNN_HTP_LIB, FLLM_GENIE_LIB)
cargo run --release -p fllm-engine --example npu_probe --features npu-ane  -- ane 1 4096 4096       # Apple silicon
```
(XDNA: `FLLM_XDNA_ORT_LIB`, `FLLM_XDNA_TARGET`, `FLLM_XDNA_CACHE_DIR`, `FLLM_XDNA_CONFIG`; placement with `XLNX_ONNX_EP_REPORT_FILE`.) Copy the
output (ProbeReport) into the matrix row; decide on the recommendations of the studies only afterwards. `V_NPU_PROBES="xdna hexagon ane"` makes
`verify-npu-openvino` run them as well.

### 2.5 ARM CPU (Apple M, Graviton, Raspberry Pi, Ampere, Snapdragon X)

CI already builds and runs the unit tests and the CPU self tests on `ubuntu-24.04-arm` and on Apple silicon ([ci.md](ci.md)); the scripts below
add the per-tier checks and real-model text on your own hardware.
```sh
scripts/check-arm-cpu.sh                  # build, unit tests, selftest-cpu per tier, FLLM_CPU_ISA cap; --quick = self test only; logs in target/check-arm-cpu/
scripts/verify-arm.sh --model MODEL.gguf  # wrapper: text per tier against FLLM_CPU_ISA=neon, bench, report.json
```
Expected: `check-arm-cpu.sh` ends with exit 0 and "ok" per tier; the line "SVE vector length" appears on SVE cores. Target devices: Apple M1 (dotprod),
M2 or newer (i8mm, bf16), Graviton 3 (SVE 256 bit, `FLLM_ARM_SVE=1`), Raspberry Pi 5 (Cortex-A76) and an A72 machine (NEON only: checks that
nothing wrong is dispatched). No performance claim without a measurement; if `neon-sve` passes *and* is measured, `cpu_arm::sve::DEFAULT_ON`
can be set to `true` (one line). `FLLM_CPU_ISA=neon-dotprod` switches `bfdot` off. The NEON intrinsics (`vshlq_u8` with negative shift counts,
`vqtbl1q_s8`, `vcvt_f16_f32`, the `bfdot` pairing) are the most likely first points of failure.

### 2.6 x86: AVX-512 and AMX

```sh
scripts/check_x86_ext.sh                  # build with --features avx512, unit tests, selftest-cpu per tier and policy overrides; exit 0 = all passed
scripts/verify-avx512.sh --model MODEL.gguf
```
Test matrix: (a) Zen 4, (b) Zen 5, (c) Intel Ice/Tiger/Rocket Lake (client, one 512-bit port), (d) Sapphire/Emerald/Granite Rapids for AMX
(cloud, for example AWS m7i/c7i, GCP C3; **Linux >= 5.16**, AMX on Linux only), (e) Arrow/Lunar Lake for AVX-VNNI-INT8. Order:
`check_x86_ext.sh` (L1), then greedy text against `FLLM_CPU_ISA=avx2` (L2), then a comparison with llama.cpp (L3). The first errors are to be
expected in: tile configuration/`#UD`/SIGILL, lane orders in `codebook`/`kextra`, pair order of `vdpbf16ps`, the permission request
(`arch_prctl`) in containers/VMs. Policy switches: `FLLM_X86_WIDE_FLOAT`, `FLLM_X86_WIDE_INT`, `FLLM_X86_BF16`, `FLLM_VNNI_GEMM`,
`FLLM_AMX_MIN_BATCH`, `FLLM_CPU_ISA`. GitHub-hosted runners often have AVX-512-VNNI (Xeon Platinum 8370C) but no AMX.

### 2.7 Windows

Requirements: Rust (MSVC toolchain), Visual Studio 2022 Build Tools ("Desktop development with C++"); for native Vulkan the Vulkan SDK; for CUDA
the CUDA Toolkit; for HIP the HIP SDK ([windows.md](windows.md)).
```powershell
.\scripts\build-windows.ps1 -DryRun                         # shows plan, tools, features
.\scripts\build-windows.ps1                                  # CPU + wgpu (Vulkan/DX12) + native Vulkan; dist\fllm-windows-x86_64-msvc.zip
.\target\release\fllm-rs.exe devices --json > devices.json
.\target\release\fllm-rs.exe selftest-gpu --device auto --json selftest.json
.\scripts\build-windows.ps1 -Cuda -CudaArchs "89;120"        # NVIDIA; report nvcc errors with the full command
.\scripts\build-windows.ps1 -Hip                             # AMD
.\install.ps1 -Build                                         # and -Variant cuda
.\scripts\verify-windows.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf     # everything together
```
Send back: `devices.json`, `selftest.json`, and for build errors the full command. Open assumptions: nvcc accepts `-Xcompiler
/MD,/EHsc,/bigobj,/Zc:__cplusplus`; `lib.exe @file`; `Launch-VsDevShell.ps1` parameters; `hipcc.exe`/`lib\amdhip64.lib`; Marlin/CUTLASS under
MSVC (never tried); static CRT with all C dependencies (onig, ring). CI builds the CPU, wgpu and native Vulkan variants on `windows-2022` and
compiles the CUDA kernels there, but no GPU runs on Windows in CI.

### 2.8 TPU / Google Colab

```python
!python fllm/scripts/colab/tpu_check.py --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
```
Requirements (`scripts/colab/tpu_check.md`): runtime type TPU v5e-1 (or v2-8), **then restart the runtime** (no notebook kernel may hold the TPU),
internet (rustup, `pip install libtpu`). The build takes 10 to 25 minutes (`--jobs 2` when memory is tight). The steps: environment, rust, libtpu,
build, devices, selftest (`FLLM_TPU_EXACT=1`), model (greedy text TPU against CPU). `scripts/verify-tpu.sh` does the same plus bench and
`report.json` (`!bash fllm/scripts/verify-tpu.sh --model-uri ...`).
Send back: `tpu_check_report.json`, `tpu_selftest.json`, `tpu_check_selftest.log`, `tpu_check_devices.log`, `tpu_check_server_tpu.log`,
`tpu_check_server_cpu.log`, `tpu_check_build.log`.
A first run **with errors** is expected: plugin version (PJRT API 0.116 in the header), self-test bounds (single FAILs at `swiglu`, `geglu`,
`attention`, `rms_norm` just over the bound - then the bound should be relaxed, not the code), known FAIL: q8 KV (`kv_cache_write q8`,
`attention q8`); DeltaNet cases are skipped. Every new program shape is compiled once (0.1-1 s), so the self test is slow. Settings:
`FLLM_TPU_LIBRARY`, `FLLM_TPU_EXACT`, `FLLM_TPU_WEIGHTS=bf16|f32|int8`, `FLLM_TPU_NUMERICS=exact|fast`, `FLLM_TPU_DEBUG=1`,
`FLLM_TPU_DUMP_DIR=DIR` (dumped `.mlir` programs can be parsed without a TPU with `scripts/colab/stablehlo_verify.py`). One process per TPU: end a
stuck `fllm-rs` with `pkill fllm-rs`. Two hardware tests without a model: `FLLM_TPU_EXACT=1 cargo test --release -p fllm-engine --features
tpu-pjrt -- --ignored tpu_hardware`.

## 3. Evaluation: when may a cell of the matrix go up?

The rules apply per cell (hardware x format or operation) in [backends.md](backends.md). Everything else stays as it is; nothing is inferred
from a neighbour (a passing Q4_0 says nothing about Q6_K). Level definitions: [test-levels.md](test-levels.md).

| Level | Condition | Evidence in the result |
| --- | --- | --- |
| **L1** | `selftest-gpu` (or `selftest-cpu`, or the backend's hardware tests) reports `ok` for this format or operation group on the real hardware - in the exact run; where a fast run with its own tolerance exists, there too. `SKIPPED` and `failed` do not count. | `selftest*.json`: `failed = 0` for the group; `fllm-rs selftest-gpu --list` names the groups (`delta linear embedding norm rope activation kv attention sampling host moe fuse`), `--only`/`--quants` narrow the run |
| **L2** | additionally: greedy text with a model that uses the cell is identical to the CPU in exact mode. If only the fast mode differs, L2 holds for the exact mode with the note "fast: differs from character N". | `report.json`: `text_vs_cpu` of the case `exact` = `same`; the server log proves the native backend |
| **L3** | additionally: `fllm-rs bench` on the hardware, with a run of a third-party implementation on the same hardware, measurements and versions documented | `bench-*.json` of both runs, `fllm-rs bench-report` |

Special cases:

* **Operations without a self test** stay L0 until a model uses them (then L2 without L1 only with a note): Mamba outside CUDA
  ([backends.md](backends.md) 5.3), DeltaNet without the group `delta`, MoE without the group `moe`.
* **Formats the self test does not know** reach L1 only through the hardware counterpart of the backend (Metal: `metal_quant_kernels`; SYCL:
  `linear_matches_cpu`; Vulkan: `hw_tests`).
* **TPU:** L1 formally requires `failed = 0` yet FAIL is expected for q8 KV; practical is L1 per group with q8 KV as a documented gap.
* **Intel NPU:** L1 only if `npu_hardware_selfcheck` passes *and* the self test is ok except for the 7 q8 attention FAILs known from the CPU.
* **XDNA/Hexagon/ANE:** at most "probe ran" (ProbeReport); there is no compute path, hence no L1 cell.
* **Software adapters** (lavapipe, WARP) are rejected unless `FLLM_ALLOW_SOFTWARE_GPU=1`; a run on them tests the logic, not the hardware, and
  raises no cell.

Result to cell (examples):

| Result on the hardware | Cells |
| --- | --- |
| Metal: `metal_quant_kernels` all `[ok]` for Q4_0...Q6_K, IQ4_NL, MXFP4, NVFP4, Q4Z, F16/BF16 | row "Metal native", columns L, K, IQ4, MX, NV, Q4Z, F: L1 (name the chip) |
| Metal: `metal_ops_kernels` all `[ok]` | "Metal native", attention (f16/q8_0): L1; MoE and DeltaNet only if `selftest-gpu` reports the groups `moe`/`delta` as `ok` |
| Vulkan: `selftest-exact.json` `failed = 0` | "Vulkan native", formats and operations of the `ok` groups: L1 (exact run); the fast variants `_i`/`_c` only with a passing fast run |
| HIP: `selftest-gpu --device hip` passes, text with `FLLM_CUDA_EXACT=1` identical | "ROCm/HIP" of the architecture (note the gfx name): L1/L2; MMA/skinny/Q4Z/DPP paths only if the switch runs are individually identical |
| ARM: `check-arm-cpu.sh` exit 0 | "CPU ARM" of the tiers that report `ok`: Q2_K, Q3_K, IQ4_NL, MXFP4, float, DeltaNet: L1 |
| x86: `check_x86_ext.sh` exit 0 on AVX-512-VNNI or AMX hardware | "AVX-512" or "AMX": L1 (name the CPU class) |
| TPU: `tpu_selftest.json` without FAIL outside the known gaps, `identical: true` in the model comparison | "Google TPU": L1 or L2 (name model and weight format) |

## 4. Failure patterns and switches

| Backend | Switch / aid | Effect |
| --- | --- | --- |
| Vulkan | `FLLM_VULKAN_VALIDATE=1` | validation layer; always use it for the first run |
| Vulkan | `FLLM_VULKAN_VERBOSE=1` | planner capabilities, per shader once the variant, specialisation constants, grid |
| Vulkan | `FLLM_VULKAN_EXACT=1` | only `_s`/`lin_*` (FP32); `set_exact_math(true)` in the code |
| Vulkan | `FLLM_VULKAN_NO_IDOT=1`, `FLLM_VULKAN_NO_COOP=1` | integer-dot or coopmat variants off |
| Vulkan | `FLLM_VK_SUBMIT_EVERY=N` | dispatches per command-buffer submission during a decode step (default 32; 0 = one submission per step, the slowest on both the RTX 2000 Ada and the Radeon 840M; 16 to 128 are within the noise there, [benchmarks.md](benchmarks.md)) |
| Vulkan | `FLLM_VK_SHADER_DIR=<dir>` | takes `<dir>/<name>.spv` instead of the built-in module of that name (compile with `glslc --target-env=vulkan1.2 -O -I shaders/vulkan <defines from variants_quant.list>`), to try a shader change without rebuilding |
| Vulkan | `FLLM_VK_COOP_WIDE_TILE=<bm>x<bn>x<subgroups>` | the planner's view of the wide cooperative-matrix shader module (default `128x64x8`, as built); for a module built with other `-DBM` / `-DBN` / `-DNSGM` / `-DNSGN` through `FLLM_VK_SHADER_DIR`. Tiles other than 128 x 64 were slower on the RTX 2000 Ada |
| Vulkan | `FLLM_ALLOW_SOFTWARE_GPU=1` | allow lavapipe/WARP (for logic tests only) |
| Metal | `FLLM_METAL_SERIAL=1` | every dispatch serialised: rules out hazard/barrier errors |
| Metal | `FLLM_METAL_FAST_MATH=0`, `FLLM_METAL_EXACT=1` | fast math off; float instead of half tiles |
| Metal | `FLLM_METAL_NO_NOCOPY=1`, `FLLM_METAL_SINGLE_LIB=1`, `FLLM_METAL_MSL=3.0\|3.1`, `FLLM_METAL_RESIDENCY=0` | no-copy mapping off; one library instead of one per file; MSL version; residency set off |
| Metal | `FLLM_METAL_PREWARM=1`, `FLLM_METAL_DEBUG=1` | compile all libraries at start; compiler errors per file; also: `FLLM_METAL_DUMP_DIR`, `FLLM_METAL_MEM_LIMIT_MB`, `FLLM_METAL_CB_DISPATCHES` |
| HIP | `FLLM_HIP_NO_MMA`, `FLLM_HIP_NO_SKINNY`, `FLLM_HIP_NO_Q4Z`, `FLLM_HIP_NO_Q4Z_SKINNY` | one kernel family off each |
| HIP | `FLLM_CUDA_EXACT`, `FLLM_CUDA_NO_TC`, `FLLM_CUDA_NO_ATTN4`, `FLLM_CUDA_NO_MMVQ` | switches shared with the CUDA sources |
| CUDA, HIP | `FLLM_CUDA_PROFILE=1`, `FLLM_CUDA_MMVQ_MAXB=N` | GPU time per weight type and batch class of every projection, printed to stderr every 2000 projections (the run is slower: one synchronisation per projection); largest batch of the int8-dot kernels for Q4_K / Q5_K / Q6_K (default 4 on CUDA, 16 on HIP, at most 16) |
| SYCL | `FLLM_SYCL_EXACT=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `FLLM_SYCL_ALL_BACKENDS=1`, `FLLM_SYCL_AOT`, `FLLM_SYCL_FORMATS` | XMX off; device choice; list all SYCL backends; AOT compilation (unchecked) |
| NPU | `FLLM_NPU=0`, `FLLM_NPU_HOST=1`, `FLLM_NPU_WEIGHTS`, `FLLM_NPU_STATS`, `FLLM_NPU_COMPILE`, `FLLM_NPU_HYBRID`, `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_MIN_ELEMS`, `FLLM_NPU_CHUNK`, `FLLM_NPU_GROUP`, `FLLM_NPU_BUDGET_MB`, `FLLM_NPU_CACHE_DIR`, `FLLM_NPU_PROPS`, `FLLM_NPU_DEVICE`, `FLLM_NPU_DUMP_IR` | offload off; host hook; planner; quantisation; cache; properties; another OpenVINO device for practice (for example `CPU`); dump the IR |
| TPU | `FLLM_TPU_EXACT`, `FLLM_TPU_WEIGHTS`, `FLLM_TPU_NUMERICS`, `FLLM_TPU_DEBUG`, `FLLM_TPU_DUMP_DIR`, `FLLM_TPU_LIBRARY` | see 2.8 |
| ARM / x86 | `FLLM_CPU_ISA=<tier>`, `FLLM_ARM_SVE=1`, `FLLM_X86_*`, `FLLM_VNNI_GEMM`, `FLLM_AMX_MIN_BATCH` | cap the tier or override the policy; `fllm-rs selftest-cpu --tier <tier>` |
| all | `FLLM_AUTO_ORDER`, `FLLM_AUTO_IGPU=0`, `selftest-gpu --only/--quants/--list/--strict` | order of `auto`; iGPUs off; narrow the self test |

When a self-test check fails in a backend, this order helps: (1) repeat with the exact switch; (2) narrow it to one line with `--only GROUP
--quants FORMAT`; (3) run the counterpart on the CPU emulation of the kernel text (Vulkan `emu/run_kernels.sh`, Metal `emu*/run_*.sh` - both on
a development machine, not on the test hardware) and check whether the emulation passes the same configuration: if it does, the error lies in
compiler, driver or hardware behaviour, not in the index/layout logic.

## 5. What to send back

1. `verify-out/<backend>/` or the files named in 2.x, per device.
2. One line per device: device, driver/ROCm/oneAPI/macOS version, command, level, last passing switch.
3. The matrix in [backends.md](backends.md) is raised from these files (section 3), not from memory.
