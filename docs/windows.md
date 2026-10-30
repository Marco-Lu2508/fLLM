# fLLM on Windows (native, MSVC)

Level of verification: **L1 for CPU, native Vulkan and the Intel NPU on one machine** (Intel Core Ultra 5 235U, Windows 11): the cross-built `fllm-rs.exe` ran the
self tests, a Studio session and benchmarks there (section "Test guide" below). **L0 for NVIDIA (CUDA) and AMD (HIP, XDNA2 NPU) on Windows**: `build.rs`,
`scripts/build-windows.ps1`, `install.ps1` and the workflows compile and package them in CI, but no CUDA or HIP kernel has run on Windows yet; the first test
round on such hardware will find mistakes.

## Which build do I get?

| Asset | Contents | Needs on the user's machine |
| --- | --- | --- |
| `fllm-windows-x86_64-msvc.zip` | CPU, Vulkan and DirectX 12 through wgpu (`rust-gpu`), native Vulkan when the Vulkan SDK was there at build time | a GPU driver for GPU use |
| `fllm-windows-x86_64-msvc-cuda.zip` | the same plus the CUDA kernels (CUDA runtime linked statically) | the NVIDIA driver |
| `fllm-windows-x86_64-msvc-hip.zip` | the same plus the AMD HIP (ROCm) kernels (optional asset, test level L0: compiles in CI, never run on an AMD GPU); the HIP runtime DLLs of the SDK are inside | an AMD driver that supports the GPU; `fllm-rs.exe` also starts without any AMD software and uses Vulkan or the CPU |
| `fllm-windows-aarch64-msvc.zip` | Windows on ARM64, CPU (NEON) and wgpu | |
| `fllm-windows-x86_64.zip` | older MinGW cross build, CPU and wgpu | |

The MSVC assets use the static C runtime (`-C target-feature=+crt-static`): no Visual C++ redistributable is needed. `install.ps1` picks the first
asset the release has: the CUDA asset when an NVIDIA driver is present (`nvcuda.dll`), then the plain MSVC asset, then the MinGW one; `-Variant default|cuda|hip` overrides
the choice. On ARM64 it takes the ARM64 asset and otherwise the x86_64 one, which runs in Windows' x64 emulation, slowly.

```powershell
irm https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.ps1 | iex   # newest release -> %LOCALAPPDATA%\fllm\bin, verified against its .sha256
.\install.ps1 -Variant cuda -AddToPath    # from a clone: force a variant, add the folder to the user PATH
.\install.ps1 -Uninstall                  # removes fllm-rs.exe and the PATH entry
.\install.ps1 -Build                      # no download: build this checkout (scripts\build-windows.ps1)
fllm-rs devices                           # before the PATH change: & "$env:LOCALAPPDATA\fllm\bin\fllm-rs.exe" devices
fllm-rs run unsloth/Qwen3.5-9B-GGUF:Q4_K_M       # --device auto: the best GPU, else the CPU
```

The binary is not code-signed. A file that the installer downloads carries no "downloaded from the internet" mark, but a zip saved from the browser does: unblock it
(`Unblock-File`) before extracting, or expect a SmartScreen prompt. Defender may scan the first start. The installer never changes Defender or SmartScreen settings.

## Test guide for Windows testers (one page)

What is checked on which hardware (2026-10-30):

| Part | Status |
|---|---|
| Windows build (`x86_64-pc-windows-msvc`, CPU + Vulkan + wgpu DirectX 12), CI on GitHub runners | builds and passes the unit and CPU self tests (workflows `platforms`, `windows-package`) |
| CPU, native Vulkan, Studio, `selftest-gpu`, OpenVINO NPU on an **Intel Core Ultra 5 235U** (Intel Graphics, Intel AI Boost), Windows 11 | **ran**, see "Measured on Intel" below |
| NVIDIA GPUs on Windows (CUDA build; RTX 2070 Super, Turing `sm_75`, is in the default architecture list) | **unverified**: compiles in CI, never ran on an NVIDIA GPU under Windows |
| AMD GPUs on Windows, HIP (`hip:0`) and Radeon 8060S / `gfx1151` | **unverified**: HIP compiles in CI, never ran on Windows. Native Vulkan on a Radeon (Ryzen Z1 Extreme handheld) ran, see the last section |
| AMD XDNA2 NPU through FastFlowLM on Windows | **unverified**: fLLM's `amd-npu` route ran on Linux only; on the Intel machine it correctly reports `no_npu` |

### 1. Get a binary

* Easiest: the newest release (<https://github.com/Marco-Lu2508/fLLM/releases>) has the zips; no account is needed.
  * RTX 2070 Super (or any NVIDIA card): `fllm-windows-x86_64-msvc-cuda.zip` (needs only the NVIDIA driver).
  * Ryzen AI Max 365 (Radeon 8060S): `fllm-windows-x86_64-msvc.zip` (CPU, Vulkan, DirectX 12; works with the normal AMD driver) and, to try the HIP route,
    `fllm-windows-x86_64-msvc-hip.zip` (contains the HIP runtime DLLs; `gfx1151` is in its architecture list).
  * Anyone: `fllm-windows-x86_64-msvc.zip`, or the desktop installer `fLLM_<version>_x64-setup.exe`.
  * Check the download against its `.sha256` file. The binary is unsigned: right-click the zip, Properties, Unblock, then extract.
* Newer than the release: workflow artifacts (needs a GitHub login; `gh run download <run id>` works too). Workflows:
  `windows-package` (artifact `fllm-windows-x86_64-msvc`, CPU + Vulkan, about 10 minutes), `windows-cuda-package` (artifact
  `fllm-windows-x86_64-msvc-cuda`; input `archs`, for example `75;80;89;120`, builds faster than the default list `61;70;75;80;86;89;90;100;110;120`),
  `windows-hip-package` (artifact `fllm-windows-x86_64-msvc-hip`; input `archs`, for example `gfx1151`). Start one with Actions, the workflow, *Run workflow*.
  Every zip contains `fllm-rs.exe` and `windows-check.ps1` (the artifact of `windows-package` carries the script next to the zip).

### 2. Run the commands (PowerShell, in the folder of the extracted zip)

```powershell
.\fllm-rs.exe --version                      # prints the features of this build: cuda / hip / vulkan / ...
.\fllm-rs.exe devices                        # every device with the exact --device value on its "Select:" line
.\fllm-rs.exe pull hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf   # 0.8 GB into %USERPROFILE%\.cache\huggingface
.\fllm-rs.exe selftest-gpu --device cuda:0 --json selftest.json      # or hip:0 / vulkan:0; expect "0 failed"
.\fllm-rs.exe serve --model <the .gguf path printed by pull> --device cuda:0 --port 8000
```

In a second window, with the server running:

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":16}"
.\fllm-rs.exe bench --url http://127.0.0.1:8000 --model <id shown by /v1/models> --scenarios prefill,decode-single --prompt-lens 256 --gen-len 64
```

Studio (the browser app) instead of `serve`: `.\fllm-rs.exe studio --model <gguf> --device cuda:0` (`--no-open` does not start a browser).
The server listens on `127.0.0.1` only, so Windows Firewall does not ask; `--host 0.0.0.0` makes it reachable from the network (and the firewall prompt appears).
Stop it with Ctrl+C (Ctrl+Break, closing the window and logoff also shut it down cleanly since 2026-10-30).

**Choosing the device**: `--device auto` (default) takes the best GPU in the order CUDA, HIP, Metal, Vulkan, DirectX 12, else the CPU. Force one with
`--device cuda:0` (NVIDIA), `--device hip:0` (AMD, HIP build only), `--device vulkan:0` (native Vulkan, any GPU), `--device cpu`. The number counts the devices of
one backend from 0; `fllm-rs devices` prints the value to use. With two GPUs (an iGPU plus a discrete card) check that the line you select names the intended card.

### 3. One command that collects everything

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows-check.ps1 -Model C:\path\to\model.gguf          # devices, nvidia-smi, amd-npu status, CPU and GPU self tests, speed per device
.\windows-check.ps1 -Model C:\path\to\model.gguf -Devices cpu,cuda:0    # only these devices for the speed part
```

It writes **`fllm-report.txt`** next to the script (hardware names, driver versions, self test tables and speeds; no prompts, no file names of yours). Send that
one file. Without `-Model` only the devices and self tests run (a few minutes). The DirectX 12 (wgpu) self test is in the report too and is experimental: failures
there are known (end of this page); failures on `cuda`, `hip` or `vulkan` are what we want to hear about.

Expected: `devices` lists the GPU with a `Select:` line of the backend you installed; `selftest-gpu` ends with `... 0 failed`; the speed part prints a table with
prefill and decode tokens/s per device. A Q4_K_M 1B model should decode at tens of tokens/s on an RTX 2070 Super and a Radeon 8060S (an expectation, not a measurement).

### 4. The AMD NPU (Ryzen AI Max 365, XDNA2): FastFlowLM

fLLM does not drive the XDNA2 NPU itself on Windows; it talks to FastFlowLM (`flm`), which you install:

1. Install the current AMD Ryzen AI / NPU driver (NPU driver 32.0.203.x or newer).
2. Install FastFlowLM with `flm-setup.msi` from <https://github.com/FastFlowLM/FastFlowLM/releases> (its own licence terms apply; fLLM does not install it).
3. `.\fllm-rs.exe amd-npu status` shows the NPU, the FastFlowLM version and what `flm validate` says.
4. `.\fllm-rs.exe amd-npu list`, then `.\fllm-rs.exe amd-npu pull llama3.2:1b` and `.\fllm-rs.exe amd-npu serve llama3.2:1b --register`. FastFlowLM has its own model
   files, not GGUF files. Status: **unverified on Windows**; please send the output of `amd-npu status` before and after the install (the report script includes it).

### 5. Measured on Intel (Core Ultra 5 235U, Windows 11, 2026-10-30)

Cross-built `fllm-rs.exe` (features `avx512,computer-use,npu-openvino,rust-gpu,vulkan`), also started from a folder with a space and non-ASCII letters in its path.

| Check | Result |
|---|---|
| `devices` | CPU, wgpu Vulkan, wgpu DirectX 12, native Vulkan (`vulkan:0`), Intel AI Boost NPU (OpenVINO runtime from `~/.fllm/openvino`, platform 3720) |
| `selftest-cpu` | 0 failed (tiers scalar, avx2, avx-vnni) |
| `selftest-gpu --device vulkan:0` | 1092 checks, **0 failed**, 227 skipped (unsupported on Vulkan), 865 ok |
| `selftest-gpu --device npu:0` | 1092 checks, 0 failed, 230 skipped; 18 NPU model compilations took 195 s (the compiling is the slow part) |
| `selftest-gpu --device dx12:0` (wgpu) | 786 checks, **35 failed** (the known DirectX 12 problem, here on Intel too) |
| Studio (`studio --device vulkan:0 --no-open`, browser) | page loads, chat works on the iGPU |
| model path with spaces and non-ASCII letters | `inspect` and `serve` fine |
| `amd-npu status` | `no_npu`, with the install hint for FastFlowLM |
| speed | [benchmarks.md](benchmarks.md), Intel Core Ultra 5 235U section: Qwen2.5-0.5B Q4_K_M decode 49 tokens/s on the iGPU against 21 to 29 on the CPU |

Not run: the CUDA and HIP builds (no such hardware), Windows on ARM, the desktop installer.

## Building from source

Needed: Rust with the MSVC host toolchain (rustup), Visual Studio 2022 or Build Tools with the "Desktop development with C++" workload (`cl.exe`,
`link.exe`, `lib.exe`), Git. Optional: the Vulkan SDK for the native Vulkan shaders (`glslc`, `VULKAN_SDK`), the CUDA Toolkit for `-Cuda`, the AMD HIP SDK for `-Hip`.

```powershell
.\scripts\build-windows.ps1                       # features rust-gpu (+ vulkan when glslc is found); dist\fllm-windows-x86_64-msvc.zip
.\scripts\build-windows.ps1 -Cuda -CudaArchs "89;120"
.\scripts\build-windows.ps1 -DryRun               # show the plan only
```

The script loads the Visual Studio developer environment itself (`Launch-VsDevShell.ps1`, else `VsDevCmd.bat`), runs `cargo build`, the library tests and a
smoke test (`fllm-rs devices`) and writes `<asset>.zip` and `<asset>.sha256` ("`<hash>  <asset>`", what `install.ps1` verifies).

By hand: `cargo build --release --locked -p fllm-engine --features "rust-gpu,cuda" --bin fllm-rs` from a *Developer PowerShell*.

### What `build.rs` does for MSVC

* **CUDA** (`--features cuda`): `CUDA_PATH` (set by the installer) or the newest `Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`; `nvcc` gets
  `-Xcompiler /MD,/EHsc,/bigobj,/Zc:__cplusplus` instead of `-fPIC` (`/MT` when the crate is built with `+crt-static`, so that the C runtime matches the one rustc links);
  objects are `.obj`, packed with `lib.exe` (response file) into `fllmcuda.lib`; `cudart_static.lib` comes from `lib\x64`. nvcc needs `cl.exe`: from a Developer Prompt it
  is on `PATH`, otherwise the script finds it through `FLLM_MSVC_BIN` (the directory of `cl.exe` and `lib.exe`) or `vswhere` and passes `-ccbin`
  (`FLLM_CUDA_CCBIN` overrides). A Visual Studio newer than the toolkit supports: `FLLM_CUDA_EXTRA_FLAGS=-allow-unsupported-compiler`.
  The MinGW target cannot build CUDA (nvcc only takes `cl.exe` on Windows); `build.rs` says so.
* **HIP** (`FLLM_HIP=1`, `build-windows.ps1 -Hip`, AMD HIP SDK for Windows): `ROCM_PATH` / `HIP_PATH` or the newest `Program Files\AMD\ROCm\X.Y`, no `-fPIC`,
  `-fms-runtime-lib=dll` (`static` with `crt-static`), `amdhip64.lib` from the SDK's `lib`. The compiler is `bin\hipcc.exe`, or the SDK's
  `lib\llvm\bin\clang++.exe` driven as a HIP compiler (`--rocm-path`, `--hip-device-lib-path`) when the SDK has no `hipcc.exe`. The default
  architecture list on Windows is `gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201` (no Vega / Instinct: there is no
  Windows runtime for them); `FLLM_HIP_ARCHS` / `-HipArchs` overrides it, and entries for which the SDK has no `oclc_isa_version_*.bc`
  device library are skipped with a build warning. Every build records its list; at run time a GPU outside it is not listed.
  The HIP runtime is **delay-loaded** (`/DELAYLOAD:amdhip64_N.dll`, the name is read from `amdhip64.lib`), so `fllm-rs.exe` starts on a machine without
  it; `FLLM_HIP_DELAYLOAD=0` links it the ordinary way. Details and the search order of the DLL: [docs/changes/hip.md](changes/hip.md).
  CI: `windows-hip-package.yml` (artifact `fllm-windows-x86_64-msvc-hip`) and the optional `windows-hip` job of the release workflow.
* **Vulkan** (`--features vulkan`): `glslc` from `GLSLC`, else from `VULKAN_SDK\Bin`, else from `PATH`.
* Unchanged on Linux: `.o` objects, `ar crs`, `-fPIC`, the library directories (the Arm CUDA directories `targets/sbsa-linux` and `targets/aarch64-linux` were added).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` | build from a Developer PowerShell, or set `FLLM_MSVC_BIN` / `FLLM_CUDA_CCBIN` |
| `unsupported Microsoft Visual Studio version` | newer Visual Studio than the CUDA Toolkit: `FLLM_CUDA_EXTRA_FLAGS=-allow-unsupported-compiler` |
| `lib.exe not found` | run from a Developer PowerShell or set `FLLM_MSVC_BIN` / `FLLM_LIB_EXE` |
| `LNK2038: mismatch detected for 'RuntimeLibrary'` | the crate and the GPU kernels use different C runtimes: build everything with the same `RUSTFLAGS` (`+crt-static` or not) |
| `cannot run glslc` | install the Vulkan SDK (sets `VULKAN_SDK`) or set `GLSLC`; `build-windows.ps1` drops the native Vulkan backend when it cannot find `glslc` |
| `fllm-rs devices` lists no GPU | install the vendor driver; wgpu uses DirectX 12 and Vulkan, the CUDA build needs `nvcuda.dll` from the NVIDIA driver |
| first GPU test | `fllm-rs selftest-gpu --json out.json` (see [device-selection.md](device-selection.md)) and send `out.json` |

## Checking fLLM on an RTX 20 series card (Turing, sm_75; e.g. an RTX 2070 Super, 8 GB)

**Status: not verified on real Turing hardware on Windows.** What exists: a Tesla T4 (Turing, Linux, Google Colab) showed that the fused gate/up activation of
the quantized prompt GEMM wrote nonsense for Q4_K_M models on sm_75, and with `FLLM_CUDA_NO_GLU_FUSE=1` the T4 wrote the right text. The cause is now
understood and was reproduced on an RTX 2000 Ada laptop GPU by giving the kernels the 64 KB of shared memory per block that Turing has
(`FLLM_CUDA_FORCE_SMEM=65536`): the prompt GEMMs fold the gated activation into the conversion of their input and find out only afterwards that none of their tile
shapes fits, so the fallback kernels read the raw gate (a 20-token prompt gave a run of newlines; with the fusion off the text was right). Two changes:
the fallback computes the activation first (the self-test group `fuse` runs the fused sequence for every legacy and K-quant format with 64 KB and passes;
with `FLLM_CUDA_NO_GLU_RECOVER=1`, the old behaviour, it fails), and the fusion stays **off below Ampere** (sm_80) as before, because that is what the T4 run
showed to work. The dispatch and the 64 KB limit of a Turing card were also run on the Ada GPU (`FLLM_CUDA_FORCE_CC=75 FLLM_CUDA_FORCE_SMEM=65536`, see the
changelog); what no emulation covers is the sm_75 machine code itself (no `cp.async`, the Turing `mma` shapes). The build has **not run on a T4 or any other Turing
card since**, and no CUDA build has run on Windows at all. If you have such a card, this is the shortest useful check (about 15 minutes, 3 GB of download), and the
output of every step is worth sending back (issue or the changelog thread):

1. **Install** the CUDA asset: `.\install.ps1 -Variant cuda -AddToPath` (needs only the NVIDIA driver, no CUDA Toolkit).
2. **See the card:** `fllm-rs devices`. Expected: the card as `cuda:0` with `capability 7.5` and a line `kernels of this binary: sm_75 cubin` (or the
   PTX of a lower architecture). `none for sm_75: rebuild with FLLM_CUDA_ARCHS including 75` means the asset has no Turing kernels: stop and report.
3. **Self test:** `fllm-rs selftest-gpu --device cuda:0 --failures-only --json turing-selftest.json`. Expected: no failure (an RTX 2000 Ada laptop GPU passes every
   check of both passes). Any failed group is a finding; attach the json.
4. **Real model, fused path (the default):** `fllm-rs serve --model <Llama-3.2-3B-Instruct-Q4_K_M.gguf> --model-name m --device cuda:0 --port 8080`, then
   `curl.exe http://127.0.0.1:8080/v1/completions -H "content-type: application/json" -d "{\"model\":\"m\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}"`.
   Expected: a sensible sentence ("Paris ..."). Nonsense or repeated tokens = the sm_75 fix does not hold.
5. **Same prompt with the fusion forced off:** set `FLLM_CUDA_NO_GLU_FUSE=1` (`$env:FLLM_CUDA_NO_GLU_FUSE=1` in PowerShell) before starting the server and ask
   again. The two texts should be the same; if only this one is right, the default path is still broken on Turing. Also try a prompt of about 600 tokens
   (the fused GEMM is used from 16 prompt tokens up) and `--device vulkan` (native Vulkan, if the Vulkan SDK was there when the asset was built) for comparison.
6. **Speed for the record:** `fllm-rs bench` or the server's `usage` numbers: decode tok/s and prompt tok/s of the 3B model (a T4 gave 46 tok/s decode with
   Llama-3.1-8B Q4_K_M; an RTX 2070 Super has about 40 % more memory bandwidth).

## Known issues from the first runs on an AMD handheld (Ryzen Z1 Extreme, Windows 11, `fllm-rs selftest-gpu`)

* Native Vulkan on the Radeon: the f16 KV cache write checks failed on the first run (AMD drivers round `packHalf2x16` toward zero; the shaders now
  convert to binary16 with their own round-to-nearest-even code). Not confirmed on the AMD driver yet, see the next run.
* DirectX 12 (`--device dx12`, wgpu): the block-32 formats Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 produced wrong values or NaN (K-quants were correct). `--device auto`
  picks native Vulkan before DirectX 12, so a normal run does not use that path; the cause is not found.
* Measured with Qwen2.5-1.5B Q4_K_M on that machine: CPU 28 tokens/s decode and 344 prompt tokens/s, Vulkan 46 to 50 decode and 1140 to 1240 prompt tokens/s.

## AMD Ryzen AI Max (Strix Halo, `gfx1151`): checklist for a first test (unverified)

**Nothing here has run on such a machine.** fLLM has not been started on a Ryzen AI Max (40 compute units, RDNA 3.5, up to 96 GB of the unified memory
usable by the GPU); the points below come from reading the code paths and from the tests on a Radeon 840M (4 compute units, the same `gfx115x`
family) under Linux. Treat every statement as a hypothesis, and please send the output of the commands to the maintainers whatever happens.

What the code assumes for this chip:

* `gfx1151` is in the default architecture list of the HIP build on Windows (`build.rs`: `gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201`);
  the HIP code treats every `gfx11xx` as RDNA 3 (wave32, WMMA matrix cores, 64 KB of LDS); `gfx1151` additionally has its own tile table in the skinny GEMM
  (`hip_skinny.cu`, F16 / BF16 weights only, never run).
* The GPU is an integrated one (`hipDeviceProp.integrated`, Vulkan device type `INTEGRATED_GPU`): the memory plan treats it as unified memory.
  The plan sums the device-local and the host-visible Vulkan heaps and, because an integrated GPU has no memory of its own, limits the result to the physical
  memory the system still has free.
* HIP only: for an integrated device the runtime figures are cut to 3/4 (a Linux observation on the 840M: the runtime hands out about 3/4 of the shared memory to
  one process). Set `FLLM_HIP_FULL_APU_MEMORY=1` to keep the runtime's own figures.

Known risk: a Strix Halo machine can reserve a large block of the RAM for the GPU in the BIOS or in the AMD driver settings ("variable graphics memory", up to
96 GB of 128 GB). That block is dedicated: the system does not count it as free, so the limit "what the system still has free" would hide it and a model
that fits the GPU's memory would be planned as a partial offload. If `Memory plan: ... (budget X GB / Y GB total ...)` in the server log shows a budget far below
the dedicated memory, start with `FLLM_NO_HOST_MEMORY_CLAMP=1` (use the device's own figure).

Checklist, Vulkan first (the plain `fllm-windows-x86_64-msvc.zip`, or `.\scripts\build-windows.ps1`), then HIP:

1. **Inventory.** `fllm-rs devices`: expect the Radeon GPU as a native Vulkan entry (`--device vulkan:0`; there may be wgpu / DirectX 12 entries as well), type
   `igpu`, a memory figure, and perhaps `Matrix cores cooperative matrix` (depends on the driver). Note the memory numbers. Install the newest AMD Adrenalin driver first.
2. **Self test.** `fllm-rs selftest-gpu --device vulkan --failures-only` (with the number from step 1 if there are several: `vulkan:0`). The Radeon 840M under
   Linux gives 0 failed checks; any failed check on this chip is a finding (send `--json out.json`).
3. **A small model.** `fllm-rs serve --model Llama-3.2-3B-Instruct-Q4_K_M.gguf --device vulkan --port 8000` and watch the log: the lines `Memory plan:`,
   `Memory split:` and `prompt matrix shaders measured on ...` (the one-time timing of cooperative matrices against the integer dot product), then
   `python scripts/bench_compare.py --model <gguf> --concurrency 1,4,8 --prompt-words 60,380 "fllm=fllm:<path to fllm-rs.exe>:vulkan"`. The numbers to report: prompt and decode
   tokens per second for one request, and the total tokens per second for 4 and 8 simultaneous requests.
4. **A big model.** The point of the chip: a 30 to 70 billion parameter model in 4-bit (for example a 70B Q4_K_M of about 42 GB) on the GPU. Expect `Memory split:
   N/N layers offloaded` (all layers). If fewer layers are offloaded although the dedicated memory is large, apply `FLLM_NO_HOST_MEMORY_CLAMP=1` (see above) and report both
   logs. A model of this size is far beyond anything tested so far (the largest on the 840M was an 8B model of 5 GB), so allocation limits of the Windows driver may show up here.
5. **Per-shader timing.** `set FLLM_VK_PROFILE=1` before `fllm-rs serve ...` prints a table of the GPU time per shader from time to time; with a 40-CU
   GPU the shares may differ from the 4-CU laptop (there `mmq_q4_k_i` took 50 to 70 % of a prompt).
6. **HIP, second.** Install the AMD HIP SDK for Windows that lists `gfx1151` as supported (check the release notes of the SDK version; if it does not, skip this
   part). Build with `.\scripts\build-windows.ps1 -Hip` or take `fllm-windows-x86_64-msvc-hip.zip` from CI. `fllm-rs devices` must list `hip:0` with
   `gfx1151`; then `fllm-rs selftest-gpu --device hip:0 --failures-only` (the Radeon 840M under Linux/ROCm 7.1: 0 failed, 822 ok), the same `serve` and
   `bench_compare.py` run with `...:hip@0`, and for the time of every projection `set FLLM_CUDA_PROFILE=1` (slower; prints the GPU milliseconds per weight type and
   batch class to stderr).
7. **Memory under HIP.** If the plan on HIP is smaller than the GPU memory the driver reports, try `FLLM_HIP_FULL_APU_MEMORY=1`; if a large allocation fails with an
   out-of-memory error with it, leave it off and report the figures.
8. **What to send:** the output of steps 1 to 3 and 6 as text, `Memory plan` / `Memory split` lines for every model, the `bench_compare` result lines, the Windows and driver
   versions (and the HIP SDK version), the BIOS setting for the GPU memory.
