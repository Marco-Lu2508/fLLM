# AMD (ROCm / HIP) backend: change fragment

All AMD code in fLLM is test level **L0**: it compiles with `hipcc` for the listed gfx targets and its matrix-core index arithmetic is
checked on paper (`tests/hip_layout_check.py`). Nothing in this fragment has run on an AMD GPU. Read "Not done" at the end before relying on it.

## What changed

1. **DeltaNet, Mamba-1, Mamba-2, ShortConv, MLA and the element-wise architecture ops now build for AMD.** `deltanet.cu`, `mamba1.cu`,
   `mamba2.cu`, `shortconv.cu`, `arch_ops.cu`, `mla.cu` are in the HIP source list of `build.rs` and their stubs are gone from
   `cuda/hip_stubs.cu`. They are plain FP32 CUDA-core code (shared memory, `__shfl_xor` with offsets below 32); they compile unchanged for
   gfx908/90a/942/950/1030/1100/1151/1201. A wave64 device (CDNA) runs a 64-lane wave as two independent 32-lane "warps" in these kernels
   (the reductions use offsets 16..1 only), exactly the structure the code has on NVIDIA. `fllm_delta_supported` and the Mamba / ShortConv
   checks therefore answer with the real shape rules instead of 0. Before this change the Rust side reported `supports_arch_ops = true` for
   HIP while the stubs returned errors; that mismatch is fixed.
2. **hipGraph for decode steps.** The graph capture code of `api.cu` (`fllm_cuda_graph_*`) is no longer excluded for HIP; `hip_compat.h` maps
   the `cudaGraph*` calls. It is **off by default** on AMD: `FLLM_HIP_GRAPH=1` turns it on (then `FLLM_CUDA_GRAPH=0` still switches it off,
   the eligibility rules are the CUDA ones in `rust_decoder.rs`).
3. **Build checks without a GPU.** `scripts/hipcheck.sh` compiles every HIP source for every gfx target (and with `RESOURCES=1` prints VGPR /
   AGPR / scratch / LDS per kernel from `-Rpass-analysis=kernel-resource-usage`). `tests/hip_layout_check.py` models the three 16x16x16
   matrix instructions (CDNA `mfma_f32_16x16x16f16`, RDNA3 `wmma_f32_16x16x16_f16_w32`, RDNA4 `..._w32_gfx12`) from AMD's lane layouts,
   reads the index formulas out of `hip/hip_mma.cuh` and `hip/hip_gemm.cuh`, and emulates the whole tile loop of `mma_gemm_kernel` for ragged
   shapes against a float64 reference (also checks the alignment of every fragment load). A deliberately wrong formula makes it fail.
4. **Packaging.** `docker/Dockerfile.rocm`, `.github/workflows/docker-rocm.yml` (manual + `v*` tags, actions pinned by SHA, passes
   `actionlint`), `scripts/verify-hip.sh`. The default offload list of `build.rs` now includes gfx906 and gfx950.

## Commands

Compile checks (no GPU; needs hipcc, e.g. the container the CI uses):

```sh
docker run --rm -v "$PWD":/src -w /src rocm/dev-ubuntu-24.04@sha256:5ed1362ea542a928651e4c710b44024ebe98870f74159cb70f0265aec8ef0abe \
    bash scripts/hipcheck.sh "gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1151;gfx1201"
RESOURCES=1 scripts/hipcheck.sh gfx1100        # register / LDS / spill figures
python3 tests/hip_layout_check.py              # needs numpy
```

**What an AMD owner runs** (Linux, ROCm installed, user in groups `video` and `render`; Windows with the HIP SDK works the same with
`ROCM_PATH`/`HIP_PATH`):

```sh
git clone https://github.com/Marco-Lu2508/fLLM && cd fLLM
scripts/verify-hip.sh /path/to/small-model.gguf     # build, devices, selftest-gpu, CPU-vs-GPU text, bench; writes verify-out/hip/
```

By hand, the same steps:

```sh
FLLM_HIP=1 FLLM_HIP_ARCHS=gfx1100 cargo build --release --locked -p fllm-engine --features hip --bin fllm-rs   # gfx of your card: rocminfo | grep gfx
target/release/fllm-rs devices                              # expect an entry "CUDA/ROCm, N CUs"
target/release/fllm-rs selftest-gpu --device hip:0 --json st.json   # expect exit 0, failed = 0
FLLM_HIP_GRAPH=1 target/release/fllm-rs selftest-gpu --device hip:0    # second run with graphs
target/release/fllm-rs serve --model m.gguf --device hip:0 --port 18002          # compare the text with --device cpu (greedy)
FLLM_CUDA_EXACT=1 target/release/fllm-rs serve --model m.gguf --device hip:0 ...  # exact mode: text must equal the CPU
```

Please send back: the whole `verify-out/hip/` directory, the line of `fllm-rs devices` that names the card, `rocminfo | grep -E "Name|gfx"`, and
the ROCm version (`/opt/rocm/.info/version` or `apt list --installed | grep amdrocm`). If `selftest-gpu` fails, the log names the check
(group, quantization, max error, tolerance, worst element); rerun with `--only GROUP` and the switches below to find the kernel.

Docker:

```sh
docker build -f docker/Dockerfile.rocm -t fllm:rocm --build-arg HIP_ARCHS=gfx1100 .       # one architecture: much faster
docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    -v fllm:/home/fllm/.cache fllm:rocm run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device hip
```

The runtime stage installs the runtime package from the ROCm apt repository of the builder image (`RUNTIME_PACKAGES`, default
`amdrocm-runtime10.1` for the pinned image). The image has not been built to the end in this work (it is several GB and the build slots were
busy); the Dockerfile follows `Dockerfile.cuda` and the individual pieces were looked up in the pinned container.

## Windows (HIP SDK for Windows)

**Test level L0 and less.** The Windows build is compiled only by CI (`windows-hip-package.yml`); until that has run green it is not even known to
compile. Nothing has run on an AMD GPU, with or without Windows. The statements below describe the design and what the CI is meant to show.

* **Build**: `scripts/build-windows.ps1 -Hip [-HipArchs "gfx1100;gfx1151"]` (Visual Studio C++ tools + a ROCm / HIP SDK). `build.rs` uses
  `bin\hipcc.exe`, or the SDK's `lib\llvm\bin\clang++.exe` as HIP compiler when there is none, objects are `.obj`, packed with `lib.exe` through a
  response file (the hipcc command line has one source and the architecture flags, far from the Windows command-line limit). Default
  architectures on Windows: `gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201`; the Linux list is unchanged. Architectures for
  which the SDK has no device library (`oclc_isa_version_<n>.bc`) are skipped with a warning.
* **SDK in CI**: the ROCm wheels of AMD's "TheRock" build (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel` 10.0.0 from `stable.repo.amd.com/rocm/whl-next/`,
  HIP runtime `amdhip64_7.dll`), the way llama.cpp's Windows ROCm job gets its SDK; no GUI installer, no EULA click-through. The version is pinned in
  `.github/actions/setup-rocm-windows/action.yml`. **The file hashes are not pinned yet**: the index does not publish them and nothing was downloaded
  while writing this, so the first run prints `SHA-256` of each wheel as a warning and the `sha256` input of the action (one `<file>  <hash>` line each)
  must be filled in afterwards; a mismatch then fails the job. The wheels are cached per version.
* **Runtime without hard dependency**: the HIP runtime is *delay-loaded* (`/DELAYLOAD:amdhip64_N.dll`, name read from `amdhip64.lib`; `delayimp.lib`).
  `src/backend/hip_runtime.rs` defines the two delay-load hooks. Before a load they look for the DLL in this order: `FLLM_HIP_DLL` (file or
  directory), the folder of `fllm-rs.exe` (the package bundles the SDK's runtime there), `System32` (the Adrenalin driver installs a HIP runtime
  there on recent versions), `%HIP_PATH%\bin`, `%ROCM_PATH%\bin`, `Program Files\AMD\ROCm\<newest>\bin`, then the Windows default search. If
  none loads, the failure hook hands out stubs: the compiler-generated registration calls that run before `main` do nothing, everything else returns
  `hipErrorNoDevice`, `hipGetErrorString` returns a text. `fllm-rs devices` then lists no AMD GPU, prints one line saying what to install
  (driver or HIP SDK, or `FLLM_HIP_DLL`), and `--device auto` goes on to Vulkan / the CPU. The DLL name is fixed at build time (an import lib
  of HIP 6.x names `amdhip64_6.dll`, 7.x `amdhip64_7.dll`), so a DLL of another major version is not picked up.
  `FLLM_HIP_DELAYLOAD=0` links the DLL normally. **What is not known**: whether the SDK's linker accepts the delay-load of this import library and the
  hook symbols without duplicate-symbol errors (the CI step prints `dumpbin /dependents` and `/imports` to show it), and whether clang's eager
  fat-binary registration works through the stubs. If the hooks fail to link, build with `FLLM_HIP_DELAYLOAD=0` (then the exe needs the DLL to start).
  Only `fllm-rs` of this package is delay-loaded (the link argument is set by the engine's build script); a different final binary that links the
  engine with `hip` (e.g. a desktop shell) needs `/DELAYLOAD:` for the DLL named in the build output warning.
* **Which GPUs**: the HIP SDK 7.2 for Windows lists as supported Radeon RX 9070 / 9070 XT / 9060 (gfx1200, gfx1201), RX 7900 / 7800 / 7700 / 7600 and
  matching Radeon PRO (gfx1100, gfx1101, gfx1102), and the Ryzen AI 300 / 400 and AI Max (Strix Halo) APUs (gfx1150, gfx1151). RDNA2 (RX 6000,
  gfx1030..1032), RDNA1 and Vega are marked unsupported by AMD ("might cause runtime errors"); no Instinct card is listed for Windows. The
  iGPU of older Ryzen APUs such as the Radeon 780M (gfx1103) is not on the list either: Vulkan (`--device vulkan`) is the path there. gfx1030
  and gfx1103 are in the build list so that unlisted but working cards can run, with no promise.
* **`HSA_OVERRIDE_GFX_VERSION`** is read by the Linux ROCm user-mode stack. The Windows runtime does not use that stack, so it is not expected to
  have any effect there (not tested); the engine does not let it override the architecture check on Windows.
* **GPUs outside the build list**: on every platform a GPU whose architecture is not in the list the binary was built for is no longer listed (it would
  fail at the first kernel); one stderr line names it and says to rebuild with `FLLM_HIP_ARCHS`. On Linux `HSA_OVERRIDE_GFX_VERSION` keeps the
  old behaviour (device listed). With several AMD GPUs of which only some are built for, the device numbers of the later ones shift.
* **Redistribution**: the package bundles from the SDK only `amdhip64*.dll`, `amd_comgr*.dll`, `rocm_kpack*.dll` (as llama.cpp's package does), plus
  the license files found in the SDK (`licenses\rocm`). These are open-source ROCm components; the build prints which files it took, and
  `README-HIP.txt` in the archive lists them. Not bundled: the graphics driver, rocBLAS / hipBLAS (not used), the MSVC runtime (statically linked).
  Whether the license texts found are complete is not checked: review them before a public release.
* **What the first CI run shows**: see the report of this change; the build log artifact `windows-hip-package-buildlog` keeps everything.

## ROCm versions and unsupported cards

* Pinned compile environment: `rocm/dev-ubuntu-24.04@sha256:5ed1362e...` (HIP 7.16 / "ROCm 10.1" package layout `amdrocm-*`, LLVM 24). All
  numbers in this fragment are from that image. Older classic layouts (6.4.x, 7.0.x with `hip-runtime-amd`) should work with
  `--build-arg BASE_IMAGE=rocm/dev-ubuntu-24.04:6.4.3 --build-arg RUNTIME_PACKAGES=hip-runtime-amd`; they were not tried. The ROCm 6.2
  compatibility notes of the vLLM kernels (FNUZ fp8 types) are not relevant here, fLLM uses no fp8 on AMD.
* The kernels need gfx900 or newer for the portable path, gfx908+ (MFMA) or gfx11/gfx12 (WMMA) for the matrix-core GEMM; gfx906 and RDNA1/2
  (gfx1030) use the FP32 GEMM and the dot-product GEMV.
* A consumer card that the ROCm release does not list can often run the code of a close relative:
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1101/1102/1103, then build with `FLLM_HIP_ARCHS=gfx1100`), `10.3.0` (gfx1031..1036, build gfx1030),
  `9.4.2` is not meaningful for consumer cards. This makes the runtime load the other chip's code object; it is unsupported by AMD and by fLLM.
  APUs (gfx1151 Strix Halo, gfx1150) have their own targets in the default list.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLLM_HIP` | unset | build time: `1` compiles the kernel library with hipcc (with `--features hip`) |
| `FLLM_HIP_ARCHS` | `gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201` | build time: offload architectures (one architecture builds much faster) |
| `ROCM_PATH` / `HIP_PATH` | `/opt/rocm` | ROCm install |
| `FLLM_HIP_GRAPH` | off | `1` allows hipGraph capture of decode steps (new, never run on a GPU) |
| `FLLM_HIP_NO_SSM` | off | `1` keeps DeltaNet, Mamba-1/2 and ShortConv layers on the host path; these device paths are on by default since they passed `selftest-gpu` and the tiny-model texts on a gfx1153 GPU (`FLLM_HIP_SSM=0` does the same) |
| `FLLM_HIP_NO_Q4Z`, `FLLM_HIP_NO_Q4Z_SKINNY`, `FLLM_HIP_NO_MMA` | off | existing: Q4Z kernels off / vLLM skinny kernel off / matrix-core GEMM off |
| `FLLM_CUDA_EXACT=1` | off | exact (FP32) kernels instead of the FP16 matrix-core paths; text must equal the CPU |
| `RESOURCES=1`, `JOBS`, `SOURCES` | | options of `scripts/hipcheck.sh` |

## THIRD_PARTY entries

No third-party code was added by this work. The new files (`scripts/hipcheck.sh`, `tests/hip_layout_check.py`, `docker/Dockerfile.rocm`,
`.github/workflows/docker-rocm.yml`, `scripts/verify-hip.sh`) were written for fLLM. The lane layouts in the Python model are the publicly
documented layouts of AMD's matrix instruction calculator and the RDNA3/RDNA4 ISA guides (documentation, no code copied). The existing
vLLM-derived files (`hip/hip_skinny.cu`, the `q4z_int4_sml_kernel` of `hip/hip_q4z.cu`) are unchanged; the third-party section of the main
file stays as it is. The vLLM files that were identified as not ported (`csrc/rocm/attention.cu`, `q_gemm_rdna3*.cu`, `moe_q_gemm_rdna3.cu`)
are still not used.

## Test levels (honest)

| Item | Level | What was run |
| --- | --- | --- |
| All 21 HIP sources compile | L0 | `scripts/hipcheck.sh` in the pinned ROCm container, targets listed in the report |
| Matrix-core lane layouts / tile loop | L0 (paper) | `python3 tests/hip_layout_check.py`: passes; fails when a formula is mutated |
| DeltaNet / Mamba / ShortConv / MLA on AMD | L0 | compiled for gfx906/908/90a/942/950/1030/1100/1151/1201 as far as listed; never executed |
| hipGraph | L0 | `api.cu` compiles with the HIP mapping; never executed |
| Dockerfile.rocm | not built to the end | pieces looked up in the container; `actionlint` passes for the workflow |
| Everything else in the backend | L0 | as before |

## Known limits and not done

* **Paged attention from vLLM (`csrc/rocm/attention.cu`) is not ported.** It is 3849 lines in three architecture variants with its own KV layout
  (`[blocks, heads, head/x, block, x]`) and PyTorch glue; adapting it to fLLM's KV pool is a separate piece of work. HIP decode attention is
  still the CUDA split-KV kernels (with the DPP reductions of `attention4.cu`). `FLLM_HIP_ATTN` does not exist.
* **No new GEMM formats.** The matrix-core GEMM of `hip_gemm.cu` already serves every GGUF kind through the shared decoder; no MFMA int8/fp8
  or gfx950 scaled-MFMA (MXFP4) variant, no FP8 W8A8 / hipBLASLt path, no NVFP4 kernel for AMD were added.
* **Q4Z still needs rows and cols divisible by 64** (the host-side rearrangement in `backend/q4z.rs` works in 64-blocks); `fllm_q4z_gptq_repack`
  remains a stub, but it is only reached for NVFP4/FP8 Marlin weights, which are not routed to AMD.
* **MoE, multi-stream, hipMallocAsync:** nothing new; MoE runs on the host path as before.
* **Quantized KV cache (`kv_quant.cu`) is not built for AMD**: it uses PTX `prefetch`, 32-bit warp masks that this ROCm rejects, and 32-lane
  shuffle semantics; the ROCm build reports the extra KV formats as unavailable.
* Fused decode kernels (`fused.cu` entry points in `api.cu`) stay disabled for HIP.
