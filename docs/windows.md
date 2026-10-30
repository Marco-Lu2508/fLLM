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

## Known issues from the first runs on an AMD handheld (Ryzen Z1 Extreme, Windows 11, `fllm-rs selftest-gpu`)

* Native Vulkan on the Radeon: the f16 KV cache write checks failed on the first run (AMD drivers round `packHalf2x16` toward zero; the shaders now
  convert to binary16 with their own round-to-nearest-even code). Not confirmed on the AMD driver yet, see the next run.
* DirectX 12 (`--device dx12`, wgpu): the block-32 formats Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 produced wrong values or NaN (K-quants were correct). `--device auto`
  picks native Vulkan before DirectX 12, so a normal run does not use that path; the cause is not found.
* Measured with Qwen2.5-1.5B Q4_K_M on that machine: CPU 28 tokens/s decode and 344 prompt tokens/s, Vulkan 46 to 50 decode and 1140 to 1240 prompt tokens/s.
