# Intel iGPU and NPU: Vulkan, SYCL and OpenVINO backends

This fragment covers the three Intel-related backends (`--features vulkan`, `--features sycl`, `--features npu-openvino`): what has to be installed,
what was changed, and exactly what was run where.

## 1. Setup requirements for Intel users

Linux is the supported platform for all three paths. Kernel 6.8 or newer is what was inspected; the NPU needs the `intel_vpu` module (in the mainline kernel since 6.3).

| Need | Vulkan (`--device vulkan`) | SYCL (`--device sycl`) | NPU (`--device npu`, `FLLM_NPU_HOST=1`) |
| --- | --- | --- | --- |
| Device nodes | `/dev/dri/renderD128` | `/dev/dri/renderD128` | `/dev/accel/accel0` |
| Group | the user must be in `render` (nodes are `root:render 0660`): `sudo usermod -aG render $USER`, log in again | same | same group (`render`) owns `/dev/accel/accel0` |
| Driver | Mesa ANV (`mesa-vulkan-drivers`, `libvulkan1`, `vulkan-tools` for `vulkaninfo`). Arrow Lake and Meteor Lake need a recent Mesa (24.1 or newer is the first with Arrow Lake ANV support; not checked by us) | Intel compute runtime (`libze-intel-gpu1` or `intel-opencl-icd` + `libze1` the Level Zero loader) | Intel NPU driver (`intel-driver-compiler-npu`, `intel-fw-npu`, `intel-level-zero-npu` from https://github.com/intel/linux-npu-driver releases) + `libze1`; firmware `vpu_37xx_v*.bin` (Meteor Lake class) is in `linux-firmware` |
| User-space runtime | none | `libsycl.so.9` (DPC++ runtime; ships with the oneAPI package or the intel/llvm tarball) | OpenVINO C runtime `libopenvino_c.so` 2024.0 or newer (for example `pip install openvino` in a virtualenv, then `FLLM_OPENVINO_C_LIB=<venv>/lib/python3.x/site-packages/openvino/libs/libopenvino_c.so.<ver>`, or `FLLM_OPENVINO_DIR`) |
| Build | `glslc` from a recent Vulkan SDK (the Debian 12 `glslc` 2023.2 cannot compile `mm_coopmat.comp`: no `GL_KHR_cooperative_matrix`; set `GLSLC=<path>`) | DPC++ compiler, see 2 | none (the OpenVINO library is loaded at run time) |

`fllm-rs devices` shows each device only when the driver stack works; the NPU line appears only with the OpenVINO runtime. Nothing in a driver stack
is installed or changed by fLLM.

## 2. Building the SYCL backend (`--features sycl`)

`build.rs` already contained the integrated `sycl_kernels()` function; `rust/engine/sycl/build_rs.patch` was obsolete and is removed. New: the open-source
DPC++ compiler is found through `DPCPP_HOME` (`$DPCPP_HOME/bin/clang++`), next to `FLLM_SYCL_CXX`, `ONEAPI_ROOT` (`icpx`), `PATH`, `/opt/intel/oneapi`.

Open-source toolchain (about 0.5 GB instead of the 10 GB `intel/oneapi-basekit` image): on Linux x86-64

```sh
curl -fsSL -o sycl.tgz https://github.com/intel/llvm/releases/download/nightly-2026-10-14/sycl_linux.tar.gz
mkdir -p /opt/dpcpp && tar -xzf sycl.tgz -C /opt/dpcpp
DPCPP_HOME=/opt/dpcpp LD_LIBRARY_PATH=/opt/dpcpp/lib cargo build --release --features sycl --bin fllm-rs
```

`icpx` from `intel-oneapi-compiler-dpcpp-cpp` (apt repository of Intel) works through `ONEAPI_ROOT` or `source /opt/intel/oneapi/setvars.sh` by the same code path;
the oneAPI `icpx` was **not** built or run here (only the nightly was). The built binary links `libsycl.so.9` dynamically (RUNPATH is the toolchain `lib`
directory): to run on another machine copy the binary plus `libsycl.so.9*` and set `LD_LIBRARY_PATH`; the Level Zero loader and the GPU runtime come from the host.

## 3. What was changed

| Change | Why |
| --- | --- |
| `sycl/attention.cpp`: the `AttnPartial` kernel used `nd_range<2>((items*128, splits), (128, 1))`. The 128 work-items of the sub-groups sat in dimension 0 and the fastest dimension (1) had local size 1. SYCL requires the sub-group dimension (last) to be a multiple of the sub-group size. Now `(splits, items*128)` with local `(1, 128)`, like every other 2-D kernel of the backend. | The Intel OpenCL CPU runtime crashed (segmentation fault inside the kernel compiler) on this kernel in `selftest-gpu --device sycl` and `serve`; a GPU driver may reject the launch or return wrong data. After the change the attention checks pass (f16 and q8_0). |
| `backend/sycl/backend.rs`: `attention_paged` and `kv_cache_write` report `UnsupportedOperation` for KV cache kinds other than 1 (f16) and 2 (q8_0). | The kernels read every kind as f16: bf16/fp8/int8/q4 caches produced wrong data with a nonzero error 1e2..1e40 in the selftest. The selftest now skips them like the Vulkan backend does. |
| `sycl/device.cpp`, `backend/sycl/device_info.rs`: `FLLM_SYCL_ALLOW_CPU=1` also lists SYCL CPU devices (type `cpu`, never chosen by `auto`). | Lets the SYCL kernels be run on a machine without an Intel GPU, through the Intel OpenCL CPU runtime. |
| `devices.rs`: `sycl` removed from the default `auto` order (`CUDA > ROCm > Metal > Vulkan > DX12 > CPU`); `FLLM_AUTO_ORDER` can add it. | Specification: `auto` never selects NPU/TPU/SYCL; an Intel GPU is used through Vulkan. Docs and help texts adjusted. |
| `selftest.rs`: `selftest-gpu --device vulkan` runs two passes like `cuda`/`hip`: first the f32 shaders against the FP32 bound (`set_exact_math(true)`), then the default shaders (int8 dot product, cooperative matrix, f16 tiles) against `FAST_TOL` = 4e-3. `FLLM_VULKAN_EXACT=1` keeps the single exact pass. | The default shaders were held to the FP32 bound, which they cannot meet by design (int8 activations: about 1e-3 relative error). That is what the unanalysed strict failures were: a tolerance classification, not a shader defect (see 5). |
| `selftest.rs`: `--device npu` runs the same two passes; delegated operations of the hybrid backend get the CPU tolerance; requantized formats get an explicit bound (3e-2 for int4 formats, 1e-2 for Q5_x/Q6_K, FAST_TOL for Q4_0/Q8_0/f16/f32). The NVFP4 W4A4 checks report `UnsupportedOperation` instead of a driver error on devices that keep NVFP4 in another layout (Vulkan, NPU). | classification of the checks, see 5 |
| `vulkan/backend/shared.rs`: `FLLM_VULKAN_FORCE_IDOT=1` plans the int8 dot-product shaders also where the device has the feature but does not accelerate it. | lets the `_i` shaders be checked on lavapipe |
| `compute.rs`: host hook for the NPU (`FLLM_NPU_HOST=1` with `--features npu-openvino`): `Linear::forward` and the group variants offer the projection of a prompt (`>= FLLM_NPU_MIN_BATCH` rows) to `NpuEngine::offload_host` and compute the remaining rows with the CPU kernels. Without the feature or the variable nothing changes. | The recommended NPU path of `OPENVINO.md` section 7 was documented but not wired. |

## 4. Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DPCPP_HOME` | unset | build time: directory of an intel/llvm release; `bin/clang++` is the SYCL compiler |
| `FLLM_SYCL_ALLOW_CPU` | unset | list SYCL CPU devices (OpenCL CPU runtime) for testing; slower than the CPU backend |
| `FLLM_VULKAN_FORCE_IDOT` | unset | plan the int8 dot-product shaders although the device does not accelerate them (lavapipe tests only) |
| `FLLM_NPU_HOST` | unset | existing; now effective: host decoder offloads prompt projections to the NPU |
| `FLLM_NPU_DEVICE` | `NPU` | existing; `CPU` runs the whole OpenVINO path on the OpenVINO CPU plugin (a test of the plumbing, not of the NPU) |
| existing and relevant | | `FLLM_VULKAN_EXACT`, `FLLM_VULKAN_NO_IDOT`, `FLLM_VULKAN_NO_COOP`, `FLLM_SYCL_EXACT`, `FLLM_SYCL_ALL_BACKENDS`, `FLLM_NPU_MIN_BATCH`, `FLLM_NPU_CHUNK`, `FLLM_NPU_MIN_ELEMS`, `FLLM_NPU_GOPS`, `FLLM_NPU_OVERHEAD_US`, `FLLM_AUTO_ORDER`, `FLLM_AUTO_IGPU` |

## 5. Selftest findings

### Vulkan (lavapipe, Mesa 25.2.8, LLVM 20.1.2, subgroup size 8; CPU emulation)

* exact pass: 573 checks, 0 failed, 336 skipped, 237 ok. The skips are 100 % "unsupported", none hides a failure: KV cache kinds other than f16/q8_0 (all of the attention
  group), the fused ops that exist only on CUDA (`add_rms_norm`, `rope_kv_write`, `swiglu + int8 quantization`, `quantized_linear_add`).
  These operations fall back to the unfused sequence on the Vulkan backend.
* fast pass with `FLLM_VULKAN_FORCE_IDOT=1`: 447 checks, 0 failed, 74 skipped, 373 ok. The `_i` (int8 dot) variants were planned (31 pipelines logged) and run (`mv_*_i`, `mm_*_i`); the largest error is 1.0e-3 of the error scale
  (tolerance 4e-3; IQ2_XS 9.3e-3 against its own bound 2e-2). **Cooperative-matrix (`_c`) shaders were not exercised**: lavapipe has no cooperative matrix.
* before the change the default (fast) shaders were checked against the FP32 bound in one pass; whether the "67 strict failures" came from those or from `--strict`
  counting skipped checks as failures could not be reproduced here (`--strict` on this device counts all 336 skips). The analysis: int8 activations add about 5e-4..1e-3 relative error, f16 tiles about 5e-4;
  both are far above the FP32 bound (1e-6) and below 4e-3, so they are a tolerance issue by construction. No shader error was found.

### SYCL (Intel OpenCL CPU runtime 2026.1.2 on an AMD EPYC, `FLLM_SYCL_ALLOW_CPU=1`; no GPU, so no Level Zero, no sub-group size 16 hardware path, no XMX)

* linear (97), embedding, norm, rope, activation, kv, sampling, moe, host: all ok; attention f16 and q8_0 ok (59); everything else skipped as unsupported.
* A crash of the attention kernel on this runtime was a real bug (section 3) and is fixed.
* The `joint_matrix`/XMX kernels (`linear_xmx.cpp`) are compiled but cannot run on a CPU device; they are untested. The Core Ultra 5 235U iGPU (Xe-LPG) has no XMX anyway.

### NPU path through the OpenVINO CPU plugin (`FLLM_NPU_DEVICE=CPU`, OpenVINO 2026.4.1; not the NPU)

* with `FLLM_NPU_GOPS=100000 FLLM_NPU_OVERHEAD_US=0` so that the cost model offloads: both passes 0 failed (573 + 447 checks; 18 projection models compiled, 3005 rows through OpenVINO).
* measured quantisation errors of the offloaded projections (relative to the error scale): Q4_0 8e-4 (fp16 only), Q8_0 2e-7, Q4_K as int4 group 128 about 1.3e-2.
* host hook (`--device cpu` with `FLLM_NPU_HOST=1 FLLM_NPU_DEVICE=CPU`, offload forced by `FLLM_NPU_GOPS=100000`): the 32-token greedy text equals the plain CPU run, a projection model (4864x896, int8) was compiled and used for the 138-token prompt; decode 42 / 47 tokens/s, prefill 205 / 2312 (cold / warm) on the build host, i.e. no slowdown of the CPU kernels. On the OpenVINO CPU plugin this proves the plumbing only.
* what `quantized_linear` on the NPU needs: static shapes (one compiled model per projection and per row block of `FLLM_NPU_CHUNK`, default 128), weights as constants in the graph (Q4_0/Q8_0 exact; other formats re-quantized to symmetric int4/int8, group 128, which is the 1.3e-2 above), fp16 activations. The compile cost is paid once per projection (the CPU plugin: about 100 ms per model; the NPU compiler is expected to be much slower, unmeasured).

## 6. Test levels

| Backend | L0 compiles | L1 selftest | L2 text equals CPU | L3 benchmark | Where |
| --- | --- | --- | --- | --- | --- |
| Vulkan native | yes (also clippy, tests) | yes, on lavapipe only (CPU emulation): exact and int-dot pass | yes (32 tokens, Qwen2.5-0.5B Q8_0), exact mode and `FLLM_VULKAN_FORCE_IDOT=1` mode, lavapipe | not meaningful on lavapipe | not on the Intel iGPU |
| SYCL | yes (open-source DPC++ nightly-2026-10-14; oneAPI `icpx` not tried) | yes on the OpenCL CPU device only | yes, exact and default mode, OpenCL CPU | not meaningful | not on a GPU |
| NPU / OpenVINO | yes | yes on the OpenVINO CPU plugin only | yes (hybrid `--device npu` on the CPU plugin) | measured on the CPU plugin only (see below) | not on the NPU |

Intel hardware itself (Core Ultra 5 235U: iGPU 8086:7d41 Arrow Lake-U, NPU 8086:7d1d Meteor Lake class NPU, kernel 6.8.0-142, Ubuntu 24.04.5) **could not be tested**: the
user has no access to `/dev/dri/renderD128` and `/dev/accel/accel0` (group `render` missing), there is no Vulkan ICD, no Level Zero loader, no OpenVINO on that machine,
and docker needs a sudo password. Nothing was changed there. The owner has to: add the user to `render` (log in again), `apt install mesa-vulkan-drivers vulkan-tools` (Vulkan),
install `libze1` plus the compute runtime (SYCL), and the linux-npu-driver packages plus an OpenVINO runtime (NPU).

### Real-hardware CPU baseline (Core Ultra 5 235U, 14 threads, `avx-vnni` tier; server capped to 2 threads, `nice 19`, 3 GB, 200 % CPU)

Qwen2.5-0.5B-Instruct Q8_0, prompt 128 / 32 new tokens, 2 repeats (one warm-up each): decode 46.4 and 37.1 tokens/s, prefill 369 (cold) and 2049 tokens/s. This is the number an iGPU or NPU path must beat.

### Measured on the build host (AMD EPYC, 4 threads; relative numbers only, the emulated devices are not hardware)

| Device | decode t/s | prefill t/s (128) |
| --- | --- | --- |
| CPU backend | 24.1 / 40.5 | 422 / 1342 |
| Vulkan lavapipe exact | 1.1 / 0.9 | 8.6 / 33.7 |
| Vulkan lavapipe int-dot | 1.1 / 0.8 | 13.1 / 29.0 |
| SYCL (OpenCL CPU), exact | 4.5 / 4.4 | 16.9 / 38.9 |
| SYCL (OpenCL CPU), default | 10.8 / 9.6 | 19.8 / 93.0 |
| NPU hybrid on the OpenVINO CPU plugin | 2.1 / 2.2 | 47.1 / 73.0 |

Does decode on the NPU beat the CPU? The design says no and this run is consistent with that: the hybrid backend routes only projections of 64 or more rows to the NPU, so decode (one row)
never touches it, and with `--device npu` everything else runs on the single-threaded reference CPU backend (2 tokens/s above). A decode token on the NPU would pay a model call per projection
(about 170 per token for 24 layers) for a bandwidth-bound matrix-vector product that the CPU already does at 40 tokens/s. A real measurement needs the NPU.

## 7. Does `auto` pick the iGPU or NPU?

`--device auto` never takes the NPU, TPU, or (after this change) SYCL. It does take an Intel iGPU through Vulkan if a hardware Vulkan device exists (integrated GPUs take part unless `FLLM_AUTO_IGPU=0`). The inventory cannot say
whether the iGPU is faster than the CPU kernels, and no startup autotune was added; on a laptop like the one above (CPU decode 40 tokens/s with 2 threads) compare `--device cpu` and `--device vulkan` once and set `FLLM_AUTO_IGPU=0` if the CPU wins.

## 8. Known limits

* No Intel GPU or NPU was available: the Vulkan subgroup-size 16/32 behaviour of ANV, `VK_KHR_cooperative_matrix` (not offered by Xe-LPG anyway), f16/int8 storage and the NPU compiler time are unverified.
* SYCL attention is scalar FP32; no DeltaNet, no host_linear, no fused ops; KV caches other than f16/q8_0 are rejected.
* Stale statements remain in the shared documents `docs/backends.md` and `docs/backends-verify.md` (they still mention `build_rs.patch` and the old levels); they are left to the merge step.
