# Hopper, Blackwell and Ada: architecture support, FP8 / int8 W8A8, FA3 and FlashInfer attention, latent MLA

This is the master fragment of the Hopper / Blackwell work. The details of each part are in the topic documents next to it:
`hopper-arch.md` (architectures, NVFP4 on datacenter Blackwell, `fllm-rs devices`), `hopper-w8a8.md` (FP8 / int8 W8A8, cuBLASLt),
`hopper-attn.md` (FA3, FlashInfer variants, tensor-core decode, cascade), `hopper-mla.md` (latent-cache MLA).

## Test levels (honest summary)

**Everything below is L0**: it compiles (CUDA 12.8.1 and 13.0.1), the generated code was inspected with `cuobjdump`, the host-side logic has
unit tests, and the CPU / f64 references of the self tests were exercised where they run on the CPU. **No kernel of this change set ran on a
GPU.** The self-test groups exist so that one Colab session decides each item (L1); the new paths are therefore **off by default** where they
change numerics or can fail at run time (see the table), and the commands below switch them on only for the check.

| item | files | default | test level |
|---|---|---|---|
| Architecture list: sm_100 (B200 / GB200) added, sm_110 on CUDA 13, sm_103 / sm_121 via the plain cubins of sm_100 / sm_120, `sm_121a` FP4 target, toolkit-aware | `build.rs` | on (build) | L0: unit tests of the plan, builds with CUDA 12.8 and 13.0 |
| `sm_90a` / `sm_100a,103a,110a` / `sm_120a,121a` only for the files that need them | `build.rs` (`Flavor`) | on (build) | L0: `cuobjdump` of the objects |
| NVFP4 W4A4 GEMM on sm_100 / 103 / 110 (CUTLASS tcgen05) | `nvfp4_cutlass_sm100.cu` | **off** (`FLLM_NVFP4_SM100=1`) | L0 |
| `fllm-rs devices`: capability, loaded cubin, fast paths | `arch_report.cu`, `devices.rs` | on | L0 (unit test of the parser) |
| FP8 W8A8 GEMM (`mma.sync` e4m3, per-token / per-tensor / 1x128 group activation scales, row and 128x128 block weight scales) | `fp8_gemm.cu` | **off** (`FLLM_FP8_GEMM=auto`) | L0 |
| FP8 W8A8 CUTLASS wgmma GEMM (Hopper) | `fp8_cutlass_sm90.cu` | **off** (`FLLM_FP8_GEMM=cutlass`) | L0 (`QGMMA` in the sm_90a SASS, 12.8 and 13.0) |
| int8 W8A8 tensor-core GEMM (Q8_0 weights, Turing and newer) | `int8_gemm.cu` | **off** (`FLLM_INT8_GEMM=1`) | L0 |
| Dense BF16 / FP16 GEMM through cuBLASLt (dlopen) | `cublaslt_gemm.cu` | **off** (`FLLM_CUBLASLT=auto`) | L0 |
| FA3 prompt attention on Hopper (FlashInfer's port, sm_90a) | `fa3_prefill.cu` | **off** (`FLLM_ATTN_BACKEND=auto`) | L0 |
| FlashInfer prompt attention with window / soft cap / sinks and ragged batches; tensor-core decode; cascade attention | `fi_attn.cu` | **off** (`FLLM_ATTN_FI=1`, `FLLM_ATTN_DECODE=flashinfer`, `FLLM_CASCADE_ATTN=1`) | L0 |
| Latent-cache MLA (compressed KV, absorbed projections, tensor-core kernels) | `mla_latent.cu`, `mla_decode.cu`, `rust_decoder/mla_latent.rs` | **off** (`FLLM_MLA=latent`) | L0 + CPU unit tests |

## Colab: one session per GPU type

Build once per runtime from the repository root (the clones are needed by the Hopper / FlashInfer / CUTLASS files; without them those files are
stubs and the self-test lines say SKIPPED):

```
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
git clone --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer third_party/flashinfer
# FLLM_CUDA_ARCHS: H100 = 90, A100 = 80, L4 = 89, G4 (RTX PRO 6000 Blackwell) = 120; T4 = 75
FLLM_CUDA_ARCHS=90 cargo build --release --locked --features cuda
target/release/fllm-rs devices
```

`fllm-rs devices` must show, for the GPU, `kernels of this binary: sm_90 cubin` (H100; `sm_89`, `sm_80`, `sm_120` on the others) and the
fast paths of the table below. A line `PTX ... compiled at start-up` or `none for sm_XX` means the architecture list was wrong.

Then, per GPU (the self-test groups switch the paths on at run time; no environment variable is needed for them):

```
target/release/fllm-rs selftest-gpu --only w8a8,hgemm,hattn,hmla,fp4sm100 --json hopper.json
FLLM_SELFTEST_BENCH=1 target/release/fllm-rs selftest-gpu --only w8a8,hgemm,hattn,hmla --json hopper-bench.json
# H100 only, the Hopper FP8 kernel:
FLLM_FP8_GEMM=cutlass target/release/fllm-rs selftest-gpu --only w8a8
```

| GPU | what `failed = 0` must cover | expected SKIPPED | timing lines worth reading (`FLLM_SELFTEST_BENCH=1`, column `device ms`) |
|---|---|---|---|
| **H100** (sm_90) | `w8a8` (all FP8 lines, `fp8 w8a8 bound` <= 1), `hgemm`, `hattn` including the `[Fa3]` lines, `hmla` | `fp4sm100` | `bench fp8 w8a8` vs `bench fp8 w8a16`; `bench cublaslt` vs the built-in GEMM; FA3 vs FA2 prompt; tensor-core decode vs builtin; cascade vs plain decode; latent vs expanded MLA |
| **L4** (sm_89) | `w8a8` (FP8 lines through `mma.sync` e4m3), `hgemm`, `hattn` (`[FlashInfer]`, `[Cascade]`), `hmla` | the `[Fa3]` lines, `fp4sm100` | FP8 W8A8 vs W8A16 at batch >= 256 is the decisive line |
| **A100** (sm_80) | `w8a8` int8 lines, `hgemm`, `hattn`, `hmla` | the FP8 lines (no FP8 tensor cores), `[Fa3]`, `fp4sm100` | int8 vs the FP16 Q8_0 kernels |
| **G4** (RTX PRO 6000 Blackwell, sm_120) | as L4; `linear` group for NVFP4 as before | `[Fa3]`, `fp4sm100` | as L4 |
| B200 / GB200 / B300 / Thor (not on Colab) | `FLLM_NVFP4_SM100=1 fllm-rs selftest-gpu --only fp4sm100,linear --quants nvfp4` | | |

Any `FAIL` line prints the group, the shape, the worst element and the tolerance; the first things to try then are the switch set to its
safe value (`FLLM_FP8_GEMM=0`, `FLLM_ATTN_BACKEND=fa2`, `FLLM_ATTN_FI=0`, `FLLM_MLA=expanded`, `FLLM_NVFP4_SM100=0`: these are the defaults) and, for the
attention kernels, `FLLM_CUDA_NO_FLASHINFER=1`.

After a group passes, the end-to-end checks (L2, text against the default path):

```
FLLM_FP8_GEMM=auto       fllm-rs run <FP8 model> ...        # against the default: same text modulo FP8 activation rounding, prompt speed
FLLM_ATTN_BACKEND=auto   fllm-rs run <f16-KV model> ...     # H100: FA3 prompts
FLLM_ATTN_FI=1 FLLM_ATTN_DECODE=flashinfer fllm-rs run <Mistral / Gemma / gpt-oss model> ...
FLLM_MLA=latent          fllm-rs serve --model <DeepSeek-V2-Lite GGUF> --device cuda --ctx 32768
```

When a group passes and the end-to-end text agrees, the default of that switch can be flipped in the file named in its topic document
(one line each: `env_mode()` in `fp8_gemm.cu`, `fllm_env_choice(... "FLLM_ATTN_BACKEND" ...)` in `fi_attn.cu`, ...).

## Environment

| variable | default | meaning |
|---|---|---|
| `FLLM_CUDA_ARCHS` | toolkit-dependent (`hopper-arch.md`) | build: plain GPU targets |
| `FLLM_NO_HOPPER`, `FLLM_NO_SM100` | unset | build: the sm_90a / sm_100a-family files become stubs |
| `FLLM_NVFP4_SM100`, `FLLM_NVFP4_SM100_TILE` | `0`, by shape | NVFP4 W4A4 through CUTLASS tcgen05 on sm_100 / 103 / 110 |
| `FLLM_FP8_GEMM` | `0` | `auto` / `1`: FP8M weights on sm_89+ as W8A8 (`mma.sync`); `cutlass`: Hopper wgmma kernel; set before the model loads (it chooses the weight layout) |
| `FLLM_INT8_GEMM` | `0` | `1`: Q8_0 prompts on the int8 tensor cores |
| `FLLM_CUBLASLT`, `FLLM_CUBLASLT_LIB` | `0`, unset | `auto` / `1`: cuBLASLt (loaded at run time) for F16 / BF16 prompt GEMMs |
| `FLLM_ATTN_BACKEND` | `fa2` | `auto` / `fa3`: FA3 prompt attention on cc 9.0 |
| `FLLM_ATTN_FI`, `FLLM_ATTN_DECODE`, `FLLM_CASCADE_ATTN`, `FLLM_CASCADE_MIN_PREFIX` | `0`, `builtin`, `0`, `128` | FlashInfer variants and batches, tensor-core decode, cascade attention |
| `FLLM_MLA`, `FLLM_MLA_KERNEL` | `expanded`, tensor cores | `latent`: latent MLA cache; `ref`: the CUDA-core reference kernel of that path |
| `FLLM_SELFTEST_BENCH` | `0` | `selftest-gpu`: add timing cases to the groups above |

## THIRD_PARTY entries

The complete lists, with what was adapted, are in the topic documents. In short: **CUTLASS 4.x** (BSD-3-Clause; examples 54, 72, 79, 89 and the Sm90 /
Sm100 / Sm120 collective builders), **vLLM** (Apache-2.0; `nvfp4_scaled_mm_sm100_kernels.cu`, w8a8 conventions, MLA decode design), **FlashInfer 0.6.5**
(Apache-2.0; the Hopper prefill kernels, the prefill / decode templates and variants, the scheduler, `fp4_gemm_template_sm100.h`), **FlashMLA / SGLang**
(MIT / Apache-2.0; design reference of the latent kernels, no code copied unless stated in `hopper-mla.md`), **NVIDIA cuBLASLt** (loaded from the CUDA
installation, not redistributed). Each adapted file keeps the licence header of its source.

## What is not done

* Everything in the table is unverified on a device; performance is unmeasured. The three kernel families written from scratch (FP8 / int8
  `mma.sync` GEMMs, the latent MLA kernels, the cascade merge) have no `ldmatrix` / multi-stage tuning.
* FA3: no soft cap, no sinks, no bf16 cache, no FP8 inputs (those shapes use FlashInfer FA2 or the built-in kernels); FP8 attention inputs of the task text are not done.
* FlashInfer's own CUDA-core decode kernel is not wired (it is the same arithmetic as `attention4.cu`); FlashInfer decode is skipped while a CUDA graph is captured.
* Block-wise FP8 (`weight_scale_inv`, 128x128) and plain int8 (compressed-tensors) weights have kernels and self tests but no weight format: the
  converter still expands block-wise FP8 to f16 and int8 to Q8_0 (changing GGUF types is outside this change set). Per-layer static activation scales are not read.
* The CUTLASS Hopper FP8 kernel covers per-token scales only (no blockwise-scaling variant of CUTLASS example 67).
* NVFP4 on sm_100 / 103 / 110: no autotuner, no grouped / MoE GEMM, no accumulate or FP16-result epilogues, no sm_103 "Ultra" kernels; the Marlin FP4 path stays the default there.
* Latent MLA: no wgmma variant, `FLLM_MLA=latent` only on the CUDA backend with a layer that runs entirely on the device, no windows / chunks / sinks / soft cap; the CPU and the other
  backends keep the expanded cache.
* Not run for the whole change set: the full `cargo test` and clippy for `rust-gpu` on the final merged tree are listed in the report with the results that were obtained.
