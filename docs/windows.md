# fLLM on Windows (native, MSVC)

Level of verification: **L0**. `build.rs`, `scripts/build-windows.ps1`, `install.ps1` and the workflows were written and checked without a Windows
build machine: `build.rs` was dry-run with a fake `nvcc`, `hipcc` and `lib.exe` (argument lists, object and library names, link directives), the
Rust code compiles for `x86_64-pc-windows-gnu`, the PowerShell scripts parse and their `-DryRun` runs under PowerShell 5.1 and 7. No MSVC build,
no `nvcc` on Windows, no Vulkan SDK on Windows has run yet; the first Windows test round will find mistakes.

## Which build do I get?

| Asset | Contents | Needs on the user's machine |
| --- | --- | --- |
| `fllm-windows-x86_64-msvc.zip` | CPU, Vulkan and DirectX 12 through wgpu (`rust-gpu`), native Vulkan when the Vulkan SDK was there at build time | a GPU driver for GPU use |
| `fllm-windows-x86_64-msvc-cuda.zip` | the same plus the CUDA kernels (CUDA runtime linked statically) | the NVIDIA driver |
| `fllm-windows-aarch64-msvc.zip` | Windows on ARM64, CPU (NEON) and wgpu | |
| `fllm-windows-x86_64.zip` | older MinGW cross build, CPU and wgpu | |

The MSVC assets use the static C runtime (`-C target-feature=+crt-static`): no Visual C++ redistributable is needed. `install.ps1` picks the first
asset the release has: the CUDA asset when an NVIDIA driver is present (`nvcuda.dll`), then the plain MSVC asset, then the MinGW one; `-Variant default|cuda|hip` overrides
the choice. On ARM64 it takes the ARM64 asset and otherwise the x86_64 one, which runs in Windows' x64 emulation, slowly.

```powershell
.\install.ps1                  # newest release, verified against its .sha256
.\install.ps1 -Variant cuda
.\install.ps1 -Build           # no download: build this checkout (scripts\build-windows.ps1)
.\.local\bin\fllm-rs.exe devices
.\.local\bin\fllm-rs.exe run unsloth/Qwen3.5-9B-GGUF:Q4_K_M       # --device auto: the best GPU, else the CPU
```

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
* **HIP** (`FLLM_HIP=1`, AMD HIP SDK for Windows): `HIP_PATH` or the newest `Program Files\AMD\ROCm\X.Y`, no `-fPIC`, `-fms-runtime-lib=dll` (`static` with `crt-static`), `amdhip64.lib`.
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
