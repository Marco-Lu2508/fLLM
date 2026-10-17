# FP8 / int8 W8A8 GEMMs and the cuBLASLt dense GEMM (CUDA)

Part of the Hopper / Blackwell work. Test level of everything below: **L0** (compiled for the listed architectures with CUDA 12.8 and 13.0; no GPU was
available, no kernel has run). The self-test groups `w8a8` and `hgemm` exist to make the first run on a device decisive.

## Features

### FP8 W8A8 GEMM (`cuda/fp8_gemm.cu`, `cuda/fp8_cutlass_sm90.cu`)

FP8M weights (E4M3 bytes plus one f16 scale per row, the format the converter writes for FP8 checkpoints) were run as W8A16 through the Marlin kernel.
On sm_89 and newer they now stay row-major E4M3 and the FP32 activations are quantized to E4M3 in the call:

* per-token dynamic scale (`amax / 448`, default), or one static per-tensor scale (`FLLM_FP8_ACT_STATIC=<factor>`, experiment: FP8M tensors carry no
  `input_scale`), or, for block-wise weights, one scale per 1 x 128 group of a token;
* the product runs on the FP8 tensor cores with FP32 accumulation: a hand-written `mma.sync.aligned.m16n8k32 e4m3` kernel (Ada, Hopper, Blackwell; five
  tile shapes from 16 x 32 for decode to 128 x 128 for prompts, auto-tuned once per shape with `fllm_autotune`), or on Hopper a CUTLASS 3.x warp-specialized
  TMA + wgmma GEMM whose epilogue applies the token scale, the row scale and the bias (`FLLM_FP8_GEMM=cutlass`);
* weights with one scale per 128 x 128 block (DeepSeek `weight_scale_inv`) are supported by the kernel (variant `BLOCK`, partial sums of every K step of
  128 are scaled by `x_scale[token][kb] * w_scale[row/128][kb]`). The engine has no weight format for block-wise FP8 yet: the importer still expands such
  checkpoints to f16, so this variant is reachable only through the self test (`Direct::Fp8Block`) until the converter keeps the scales.

Layout: `upload_weight` of an FP8M tensor keeps the raw tensor in `DeviceWeight::buffer` (so the host fallbacks still read it) and puts the FP32 row scales
in `DeviceWeight::perm`; `perm.is_some()` on an `Fp8Marlin` weight selects the W8A8 kernel in `quantized_linear`. No `QuantType` variant was added.
The Marlin W8A16 layout remains for `FLLM_FP8_GEMM=0`, devices below sm_89 and widths that are not a multiple of 128.

### int8 W8A8 GEMM (`cuda/int8_gemm.cu`)

`mma.sync m16n8k32 s8` (Turing: four `m8n8k16`), int32 accumulation. Two weight layouts:

* Q8_0 (what int8 / compressed-tensors checkpoints are converted to): the kernel reads the device layout of the weight directly, activations are quantized per token and
  32 values (`amax / 127`), each block's int32 sum is scaled by activation scale x block scale. Used for prompt batches (16 and more) when
  `FLLM_INT8_GEMM=1`. **Off by default** (`auto` = off): the FP16 tensor-core kernel (qgemm8) stays until the int8 kernel has been measured faster.
* plain int8 `[rows][cols]` with one FP32 scale per row and per-token activation scales (SmoothQuant style): `fllm_int8_plain_gemm`, reachable through the
  self test only (no engine weight format).

### Dense BF16 / FP16 GEMM through cuBLASLt (`cuda/cublaslt_gemm.cu`)

Chosen route: cuBLASLt opened with `dlopen` / `LoadLibrary` at the first use (no link-time dependency; only the declarations of `<cublasLt.h>` are used
at build time), instead of a CUTLASS Sm90 GEMM: the library already contains the wgmma / TMA kernels for Hopper and the tcgen05 ones for B200 and picks the
algorithm by heuristic; a CUTLASS GEMM for dense weights would need its own tuning on hardware nobody has access to here. When the library is missing, a
symbol is absent or any call fails, the function declines and the built-in tensor-core GEMM (tc_gemm.cu) runs. The activations are converted to the
weight's 16-bit format in a scratch buffer, the result is FP32; a bias is added by a small kernel. Plans (descriptor, layouts, heuristic algorithm) are
cached per shape; the 32 MB workspace and the scratch are allocated at the first uncaptured call (inside a CUDA graph capture the function declines).
`auto` uses it from compute capability 9.0 for 64 tokens and more.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FLLM_FP8_GEMM` | `0` (unset) | `auto`/`1`: FP8M weights on sm_89+ run W8A8 with `mma.sync`; `0`: Marlin W8A16 (the layout is chosen when the weight is uploaded); `cutlass`: Hopper wgmma kernel (sm_90 only, per-token scales; falls back to `mma.sync` elsewhere) |
| `FLLM_FP8_ACT_STATIC` | unset | a positive factor selects one static per-tensor activation scale for every FP8 layer (experiment) |
| `FLLM_INT8_GEMM` | `auto` (= off) | `1`: Q8_0 prompts (16+ tokens) on the int8 tensor cores |
| `FLLM_CUBLASLT` | `0` (unset) | `auto`: cuBLASLt for F16 / BF16 weights from cc 9.0, 64+ tokens; `1`: every cc >= 8.0, 16+ tokens; `0`: never |
| `FLLM_CUBLASLT_LIB` | unset | file name of the cuBLASLt library to open first |
| `FLLM_SELFTEST_BENCH` | unset | `1`: the `w8a8` and `hgemm` groups add timing lines (TFLOPS) of the new path against the one it replaces |
| `FLLM_NO_HOPPER` | unset | build time: do not compile `fp8_cutlass_sm90.cu` for sm_90a (stub) |

Each switch is read once; the self test overrides them at run time (`set_fp8_gemm_mode`, `set_int8_gemm_mode`, `set_cublaslt_mode`).

## Colab: what to run

Build as usual (`cargo build --release --features cuda`; the CUTLASS headers must be in `third_party/cutlass` for the Hopper kernel).

```
# every device: correctness
FLLM_SELFTEST_BENCH=1 fllm-rs selftest-gpu --only w8a8,hgemm --json w8a8.json
```

| Device | Expected |
|---|---|
| L4 (sm_89) | `fp8 w8a8`, `fp8 w8a8 bound`, `fp8 row scales`, `fp8 128x128 blocks`, `fp8 static scale`, `int8 row scales`, `int8 w8a8 q8_0` all ok; the bench lines: `bench fp8 w8a8` vs `bench fp8 w8a16` (batch 256 and up should favour W8A8: that is the decision criterion, there is no measurement yet), `bench q8_0 int8` vs `bench q8_0 fp16`; `cublaslt` lines ok (auto is off on sm_89; forced by the test) |
| A100 (sm_80) | the FP8 lines are SKIPPED (no FP8 tensor cores: the weight stays in the Marlin layout); `int8 row scales` and `int8 w8a8 q8_0` ok (IMMA m16n8k32); `cublaslt` ok |
| H100 (sm_90) | as L4; run twice, the second time with `FLLM_FP8_GEMM=cutlass` (the same lines, now through wgmma; if the CUTLASS kernel declined, the lines still pass through `mma.sync`, so compare the bench times); bench: `bench cublaslt` should beat `bench built-in` clearly at batch 256 and up |
| G4 (RTX PRO 6000 Blackwell, sm_120) | as L4 (the `mma.sync` e4m3 path); `FLLM_FP8_GEMM=cutlass` is a no-op (cc != 90) |

A failing `fp8 w8a8` line prints the worst element, its error relative to the sum of absolute terms and the tolerance (1e-3: the FP8 tensor cores add the
products of a k-step with limited alignment width); a failing `fp8 w8a8 bound` line (> 1) means the activation quantization is wrong (scale or conversion),
not accumulation noise. int8 lines are held to 1e-5 (integer sums).

End to end, once the group passes: `fllm-rs` with an FP8 model (`FLLM_FP8_GEMM=auto` against the default `0`) and compare the text and the prompt / decode speed.

## THIRD_PARTY entries

* **CUTLASS** `examples/54_hopper_fp8_warp_specialized_gemm` (BSD-3-Clause, NVIDIA): collective builders of the Hopper FP8 GEMM, adapted in
  `cuda/fp8_cutlass_sm90.cu` (E4M3 x E4M3 -> FP32, epilogue visitor tree of scale broadcasts).
* **vLLM** `csrc/quantization/w8a8` (Apache-2.0): conventions only (per-token activation scales, per-channel weight scales, 128 x 128 block weights with
  1 x 128 activation groups, scales folded into an epilogue visitor tree); no code copied.
* **NVIDIA cuBLASLt**: loaded from the CUDA installation at run time, not linked or redistributed; only its public header declarations are used at build time.

## Known limits

* Nothing has run on a GPU. Performance of the hand-written kernels is unknown; the FP8 kernel uses a plain 2-stage `cp.async` pipeline and scalar
  fragment loads from shared memory (no `ldmatrix`), the int8 kernel reads fragments straight from global memory.
* W8A8 is OFF unless `FLLM_FP8_GEMM=auto|1|cutlass` is set (before the model loads): the kernels never ran on a device, and the activation rounding
  changes the results of every FP8 model. The intended default is `auto` (W8A8 on sm_89+); flip `env_mode()` in `fp8_gemm.cu` once the `w8a8` group passes on L4 / H100 / G4.
* FP8 activations lose precision against the FP16 activations of the Marlin path (worst case 1/16 of an element, typically 2 % per product, random errors average out); the
  self test bounds it, it does not make it disappear.
* Block-wise FP8 and plain int8 weights have no engine weight format (converter / GGUF type): only the kernels and their self tests exist.
* The CUTLASS kernel handles per-token scales only, `rows % 4 == 0`, 16-byte aligned operands; the static-scale and block variants use `mma.sync`.
* Scratch buffers (quantized activations, cuBLASLt workspace) are allocated at the first call and never freed when they grow (a captured CUDA graph may
  point to them); a W8A8 call that would have to allocate during a graph capture returns an error (the decoder then reports "CUDA graph capture failed"
  and decodes without graphs), cuBLASLt declines. Prompts run before the first capture and size the buffers, so this is not expected in practice.
