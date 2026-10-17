# Architecture support: Hopper, datacenter Blackwell, Thor, DGX Spark (build, runtime selection, NVFP4 on sm_100)

Test levels used below: L0 = compiles (and the generated code was inspected with `cuobjdump`), L1 = self-test ran on the device,
L2 = text matches the CPU, L3 = benchmarked. **Nothing in this fragment ran on a GPU.** Everything here is L0.

## 1. Which GPU architectures a build contains

`rust/engine/build.rs` asks the toolkit what it can compile (`nvcc --list-gpu-code`, a version table when that fails) and builds the
default list from it. `FLLM_CUDA_ARCHS` still overrides the list (`;`-separated numbers, e.g. `FLLM_CUDA_ARCHS="80;90;120"`; entries
the toolkit does not know are dropped with a build warning instead of failing in nvcc; a suffix `a` / `f` is accepted and ignored).

| toolkit | generic targets (plain `sm_N` cubin, PTX of the newest) | architecture-specific (`a`) targets |
|---|---|---|
| CUDA 12.8 | 61 70 75 80 86 89 90 **100** 120 (+ PTX compute_120) | sm_120a (FP4), sm_100a (tcgen05), sm_90a (wgmma/TMA files only) |
| CUDA 12.9 (not installed here; the entries follow NVIDIA's release notes and the probe in `build.rs`, which asks the toolkit) | the same as 12.8 | + sm_121a, sm_103a |
| CUDA 13.0 | 75 80 86 89 90 100 **110** 120 (Pascal and Volta are gone from the toolkit) | sm_120a, sm_121a, sm_100a, sm_103a, sm_110a, sm_90a |

Why this is enough for every GPU family:

* A plain cubin runs on every GPU of the same major compute capability with an equal or higher minor version (sm_100 cubin on a 10.3
  B300 / GB300, sm_120 cubin on a 12.1 DGX Spark). So the generic kernels (Marlin, attention, dense GEMMs, all quantized formats) need
  no extra target for sm_103 / sm_121. **The default list used to have no sm_100 at all: a B200 / GB200 had neither a cubin nor usable PTX
  (compute_120 PTX cannot be compiled down to a 10.0 device) and could not start.** sm_110 (Thor, CUDA 13) is a separate family.
* Architecture-specific cubins (`sm_XXa`: FP4 `mma.sync`, tcgen05 / UMMA, wgmma) run only on exactly that compute capability.
  `build.rs` therefore builds them per capability, only for the few files that need them (`Flavor` in `build.rs`):
  `nvfp4.cu` and `nvfp4_cutlass.cu` for sm_120a **and sm_121a**, `nvfp4_cutlass_sm100.cu` for sm_100a / sm_103a / sm_110a, the wgmma files
  for sm_90a only. These files are not built for the plain targets, the plain targets get the stub branch of the kernels.
* Whether an sm_120a cubin loads on a 12.1 device is not something this host could test; building `sm_121a` as well makes the question moot
  (CUDA 12.9+ / 13.0), and `fllm_cuda_nvfp4_mma_native` (nvfp4.cu) asks the driver at run time (`cudaFuncGetAttributes` on a kernel of the
  file: it fails with `cudaErrorNoKernelImageForDevice` when the device has no cubin) so that a binary built with CUDA 12.8 on a DGX Spark
  keeps the Marlin FP4 weights (W4A16) instead of failing at the first GEMM. For sm_120 the behaviour is unchanged.
* PTX fallback: the newest generic target (compute_120) is embedded as PTX, as before. A GPU that is newer than that gets it JIT-compiled
  (slow first start, the driver caches it). A GPU of a family that has no cubin in the list gets nothing: `fllm-rs devices` says so
  (section 4).

Switches of the build: `FLLM_NO_HOPPER=1` (no sm_90a files), `FLLM_NO_SM100=1` (no sm_100a / 103a / 110a file); without the CUTLASS
headers (`third_party/cutlass` or `FLLM_CUTLASS_DIR`) the files that need them are built as stubs that decline every call.

Verified (L0) on the build host, CUDA 12.8.1 and 13.0.1 images: `rustc --test build.rs` (arch plan unit tests), single-file `nvcc`
compiles of `nvfp4.cu`, `nvfp4_cutlass.cu` (sm_120a + sm_121a on 13.0: 39 functions, ~179k SASS lines in each cubin, i.e. the CUTLASS
kernels are really generated for sm_121a), `nvfp4_cutlass_sm100.cu` (sm_100a on 12.8; sm_100a, sm_103a, sm_110a on 13.0: all four GEMM tile
variants plus the helper kernels in every cubin, ~22.6k SASS lines each). Complete `cargo build --release --features cuda` builds
(CUDA 12.8.1, `FLLM_CUDA_ARCHS="100;120"`, 10 min; CUDA 13.0.1, `FLLM_CUDA_ARCHS="90;100;110;120"`, 18 min) link; `cuobjdump -lelf` of the
objects of the 13.0 library shows `sm_90 sm_100 sm_110 sm_120` cubins for the Marlin and attention objects, `sm_120a sm_121a` for the
two NVFP4 objects, `sm_100a sm_103a sm_110a` for `nvfp4_cutlass_sm100.o`; the 12.8 library has `sm_100 sm_120`, `sm_120a` and `sm_100a`.
The default list itself (all of 61 .. 120) was not built in one piece.

## 2. NVFP4 on datacenter Blackwell (sm_100 / sm_103 / sm_110): `cuda/nvfp4_cutlass_sm100.cu`

Datacenter Blackwell has no `mma.sync` FP4; its FP4 tensor core is tcgen05 / UMMA with the accumulator in tensor memory, which only the
CUTLASS 4.x `Sm100` collectives express. The new file builds the block-scaled NVFP4 GEMM with `cutlass::arch::Sm100`,
`OpClassBlockScaledTensorOp`, `nv_float4_t<float_e2m1_t>` operands and the 1-SM / 2-SM schedules (`KernelScheduleAuto` picks them from the
tile: 128x128x256, 128x256x256 with one SM; 256x128x256, 256x256x256 with a 2x1 cluster). The weights are the same device layout as on
sm_120 (packed E2M1, swizzled scale factors, 16-byte trailer), so nothing changes in the converter or the upload; activations are quantized
by fLLM's kernels into the swizzled layout (same recipe as `nvfp4_cutlass.cu`, per-tensor factor `FLLM_NVFP4_ACT_SCALE`), the epilogue
factor comes from device memory. All batch sizes go through this GEMM (M tile 128; decode is memory bound, the padding rows cost nothing
visible). Dispatch by compute capability is in `fllm_nvfp4_try` (nvfp4.cu): 10.x / 11.0 -> `fllm_nvfp4_sm100_try`, 12.x as before. The
accumulate / FP16-result epilogues and the FP16-activation fusions of the sm_120 path are not implemented here (`fllm_nvfp4_cutlass_ok`
answers no on these devices, so the decoder does not ask for them).

* **Default: off.** `FLLM_NVFP4_SM100=1` turns it on. Without it a B200 keeps the Marlin W4A16 FP4 weights, as before. The reason is
  that this code never ran on a device (L0 only). When the switch is on and the device and binary qualify (`fllm_cuda_nvfp4_sm100_native`),
  `backend/cuda.rs` uploads NVFP4 weights in the native layout (`fp4` flag), otherwise it does not.
* Env: `FLLM_NVFP4_SM100=1`, `FLLM_NVFP4_SM100_TILE=128x128x256|128x256x256|256x128x256|256x256x256`, `FLLM_NVFP4_ACT_SCALE` (as on sm_120).
* Compile-only: everything in this section. Not done: tuning (no autotuner like `FLLM_NVFP4_TUNE`), grouped / MoE GEMM, swap-AB for tiny
  batches, FP8 or MXFP4 variants, the sm_103 "Ultra" FP4 kernels (CUTLASS example 89).
* Self-test group `fp4sm100` (see below).

## 3. Self-test groups and commands

```
FLLM_NVFP4_SM100=1 fllm-rs selftest-gpu --only fp4sm100                     # B200 / GB200 / B300 / Thor
FLLM_NVFP4_SM100=1 FLLM_SELFTEST_BENCH=1 fllm-rs selftest-gpu --only fp4sm100   # + prompt-sized shapes, device ms per case
```
Expected on a datacenter Blackwell with a binary built for it: every `nvfp4 sm100` line `ok` (device error within the W4A4 tolerance of
the `linear` group, two runs bit-identical). On any other device: all lines SKIPPED. If a line fails, the first thing to compare is the
same case with `FLLM_NVFP4_SM100=0` (Marlin) and `fllm-rs selftest-gpu --only linear --quants nvfp4`.

## 4. `fllm-rs devices`

CUDA devices print two more lines (cuda/arch_report.cu, `backend/cuda/hopper_arch.rs`):

```
      Compute:      capability 9.0; kernels of this binary: sm_90 cubin
      Fast paths:   Marlin 4-bit / FP8 / FP4 weights (W4A16, W8A16); FlashInfer prompt attention (f16 KV); ...
```
`kernels` says which cubin of the binary the device loads (`sm_100 cubin (runs on sm_103)`, `PTX of sm_120 compiled at start-up`, or `none
for sm_XX: rebuild with FLLM_CUDA_ARCHS including XX`). `Fast paths` lists what the binary was built with for that compute capability
(from the build description `FLLM_BUILT` that `build.rs` passes to `arch_report.cu`): features that were built as stubs are not listed. The name,
SM count and memory are on the existing lines. `fllm-rs devices --json` carries `compute_capability`, `kernels`, `fast_paths`.

## 5. What else could keep a Hopper GPU slow (audit, no GPU)

* The sm_90 build contains its own sm_90 cubin of every kernel (plain `-gencode ...,code=sm_90`); an H100 loads that, not the sm_80
  cubin. `fllm-rs devices` shows `sm_90 cubin`. The Marlin kernels are the vLLM sm_80 kernels (mma.sync + cp.async), which run on Hopper
  at roughly the speed of the tensor-core `mma.sync` path (about 60 % of wgmma peak): Hopper-specific kernels (wgmma, TMA) exist only
  in the files listed in the other fragments (FA3 prefill, CUTLASS FP8), compiled for sm_90a only.
* Nothing in `cuda/*.cu` tests `__CUDA_ARCH__ == 800` or caps at `< 900`; the only exact checks are `== 750` (Turing) in Marlin. The
  device checks are `fllm_cc() >= 75 / 80`, the FP4 ones `major == 12` (nvfp4.cu); none of them mis-classifies sm_90 / sm_100 / sm_110.
* Shared memory: kernels size their tiles from `fllm_smem_limit()` (opt-in maximum, 227 KB on H100), not from a hard-coded 99 KB.

## 6. THIRD_PARTY entries (this fragment)

| source | licence | used for |
|---|---|---|
| CUTLASS 4.x (`examples/72_blackwell_narrow_precision_gemm`, `include/cutlass/gemm/collective/builders/sm100_*`) | BSD-3-Clause | the Sm100 block-scaled NVFP4 GEMM configuration in `cuda/nvfp4_cutlass_sm100.cu` |
| vLLM `csrc/quantization/fp4/nvfp4_scaled_mm_sm100_kernels.cu` | Apache-2.0 | tile choice by token count, argument assembly (`alpha_ptr`) |
| FlashInfer `include/flashinfer/gemm/fp4_gemm_template_sm100.h` | Apache-2.0 | epilogue without a source tensor, candidate tiles |

## 7. Environment

| variable | default | meaning |
|---|---|---|
| `FLLM_CUDA_ARCHS` | toolkit-dependent list (section 1) | build time: plain GPU targets; unknown entries are dropped with a warning |
| `FLLM_NO_HOPPER`, `FLLM_NO_SM100` | unset | build time: build the sm_90a / sm_100a-family files as stubs |
| `FLLM_NVFP4_SM100` | `0` | run time: NVFP4 W4A4 through CUTLASS tcgen05 on sm_100 / sm_103 / sm_110 |
| `FLLM_NVFP4_SM100_TILE` | by token count | run time: force a tile of that GEMM |
| `FLLM_SELFTEST_BENCH` | `0` | selftest-gpu: add prompt-sized timing cases to the `fp4sm100` and the other Hopper groups |
