# Intel GPU backend through SYCL / oneAPI (Cargo feature `sycl`)

**Test level: L0 for Intel GPUs; L1 only on the OpenCL CPU device (see [intel.md](../../../docs/changes/intel.md)).** The code is written and is accepted by the front end and SPIR-V code generation of the DPC++ compiler and by `cargo check/clippy`.
It has **never run on an Intel GPU**. Nothing in this folder claims that the backend "runs" or is fast; the
numbers in [the evaluation](../../../docs/research/sycl-evaluation.md) are expectations, not measurements.

## Files

| File | Contents |
|---|---|
| `fllm_sycl.h` | C interface of `libfllmsycl` (38 `fllm_sycl_*` functions, raw USM pointers); mirrored in `src/backend/sycl/ffi.rs` |
| `common.hpp` | device/stream types, error handling, forwarding of the launch functions |
| `device.cpp` | device enumeration (Level Zero first), XMX detection (`matrix_combinations`), USM memory, in-order queues, staging arena for small host arrays |
| `weights.cpp` | device layouts of the weights and repacking of the GGUF bytes (host, parallel); `fllm_sycl_pack_weight` |
| `formats.hpp` | device decoder per format (`F::decode`, `F::pos`); one unit = 8-32 values per work-item |
| `linear_kernels.hpp`, `linear_small.cpp`, `linear_tiled.cpp`, `linear_xmx.cpp`, `linear_api.cpp` | linear kernels: sub-group matvec (batch <= 8, decode, also MoE), shared-memory tile in FP32 (from batch 9), `joint_matrix`/XMX in fp16 (from batch 16 on XMX devices) |
| `attention.cpp` | paged attention (f16 and q8_0 cache, GQA, softcap, window/chunk, sinks, ALiBi), split-KV in two stages |
| `elementwise.cpp` | RMSNorm, LayerNorm, RoPE, SwiGLU/GeGLU/OAI-SwiGLU, residual, KV write, argmax, MoE routing/combine, gather/scatter, Q/gate split |
| `build_sycl.sh` | builds `libfllmsycl.a` without cargo (compile, `-fsycl-link`, `ar`) |
| `build_rs.patch` | historical patch for `rust/engine/build.rs`; its content (function `sycl_kernels`) is already part of `build.rs`, so it does not need to be applied |
| `tests/decode_check.cpp`, `tests/run_decode_check.sh` | host test (no device): packs random GGUF blocks and compares the device decoders with a reference following the engine's CPU decoder |

The evaluation of Vulkan versus SYCL for Intel GPUs, the recommendation and the measurement plan are in [docs/research/sycl-evaluation.md](../../../docs/research/sycl-evaluation.md).

## Formats

Read natively (15 decoders, 16 GGUF types): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS
(repacked into IQ4_NL blocks), MXFP4. Everything else is converted on the host during upload, and this is reported once:
FP8, NVFP4 and the 4-bit group formats (Q4Z/Q4ZP, AWQ/GPTQ) become f16, the other formats with 256-wide rows (IQ1/IQ2/IQ3,
Q8_K ...) become Q5_K, the rest becomes f32 (as in the CUDA backend). **Not** present: DeltaNet (`delta_layer`), `host_linear`, CUDA graphs, the
`linear_f16_out` family, device sampling (runs on the CPU).

Device layout (everything 16-byte aligned so that the kernels load 8/16-byte blocks): scales and quants of a row separate
(`[d][m][qh][qs]`, block count padded to a multiple of 8), Q6_K 224, Q2_K 96, Q3_K 112 bytes per super-block, float rows padded to 8
values. See the head of `weights.cpp`.

## Building

Prerequisite: the oneAPI DPC++ compiler (`icpx`; on Windows `icx`) with the Level Zero runtime, or the open-source DPC++ build. `build.rs` is built in (function
`sycl_kernels`; `build_rs.patch` is no longer needed). `cargo build --release --features sycl` finds the compiler through `FLLM_SYCL_CXX`,
`DPCPP_HOME/bin/clang++` (open-source DPC++ from `github.com/intel/llvm/releases`, file `sycl_linux.tar.gz`, tested with `nightly-2026-10-14`),
`ONEAPI_ROOT`, `PATH` (`icpx`) or `/opt/intel/oneapi`. Without a compiler `build.rs` only **warns**: there is then no
`libfllmsycl`, `ffi.rs` uses Rust stubs, the crate builds, `--device sycl` reports "no SYCL runtime in this build", and `fllm-rs devices`
shows no SYCL device. `FLLM_SYCL_AOT=acm-g10,pvc,bmg-g21` compiles ahead of time for Intel GPUs (needs `ocloc`; default: SPIR-V, the driver
compiles on the first call), `FLLM_SYCL_EXTRA_FLAGS` appends flags, `FLLM_SKIP_GPU_KERNELS=1` skips the build.
Important: the library is linked **as a whole** (`static:+whole-archive`), because the pre-linked device object contains only a
constructor that an ordinary archive link would discard (the kernels would then be missing at run time).

`build.rs` sets the cfg `fllm_sycl_native` itself when the library was built; without it the backend builds as a stub.

## Run time

`libsycl` (from the oneAPI directory, `RUNPATH` is set), the Level Zero loader (`libze_loader.so.1`, in the Linux package `libze1`) and the
Intel Compute Runtime driver. `ONEAPI_DEVICE_SELECTOR=level_zero:0` selects as in any SYCL program; `FLLM_SYCL_ALL_BACKENDS=1` also lists
the OpenCL duplicates. Single allocations are limited to about 4 GiB on Level Zero by default: for larger weights or KV pools
set `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. `FLLM_SYCL_EXACT=1` disables the fp16 matrix-engine kernels (everything in FP32).

## Verification

* Host test of the decoders (no device, about 30 s): `tests/run_decode_check.sh` (point `CXX` and `LD_LIBRARY_PATH` at the DPC++ compiler).
* Front end + SPIR-V without a device: compile the file with the DPC++ compiler using `-fsycl -fsycl-device-only -I<this folder>`.
* On an Intel GPU: `fllm-rs selftest-gpu --device sycl --json out.json` (batch <= 5, FP32 exact), then
  `cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu` (covers all batch paths including XMX), then a real
  model against the greedy text of the CPU (L2) and the measurement plan from the evaluation (L3).
