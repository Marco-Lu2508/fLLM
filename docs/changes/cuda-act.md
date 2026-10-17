# cuda-act: activation type, Marlin small-M, prompt levers (fragment)

Everything below is **test level L0** unless a row says otherwise: it compiles for sm_75, sm_80, sm_89, sm_90 and sm_120 (BF16 Marlin kernels for sm_80 and
newer), the Rust policy and CLI parsers have unit tests, and the `selftest-gpu` cases are written, but no GPU was available to the author, so **no CUDA kernel
of this change has run**. One Colab run (section 5) decides every open point.

## 1. Features

### 1.1 Narrow activations: `FLLM_ACT_DTYPE=f32|f16|bf16|auto`
The residual stream, norm outputs, q/k/v, attention output and feed-forward intermediates on the device are FP16 or BF16 (sums, residual adds, rotations,
SiLU run in FP32 registers; the add is rounded once and the norm reads the rounded value). Marlin 4-bit GEMMs read and write the narrow arrays directly, the add of
the output/down projection happens inside the next norm (`quantized_linear_add_norm`), the decode attention kernel and FlashInfer take the narrow arrays, K-quant
GEMVs quantize to int8 straight from them. Everything else (NVFP4, FP8, other K-quant batch sizes, activation permutation) runs on FP32 copies in persistent scratch.
Default `f32` (nothing changes, bit-identical). Details, audit table, byte counts and overflow policy: docs/activation-dtype.md.

```
FLLM_ACT_DTYPE=f16 fllm-rs serve --model <AWQ/GPTQ model>          # FP16 activations
FLLM_ACT_DTYPE=bf16 fllm-rs serve --model <AWQ/GPTQ model>         # BF16 (sm_80+; BF16 group scales are written at upload)
FLLM_ACT_DTYPE=auto ...        # BF16 for BF16 files and Gemma, FP16 otherwise; only for models with 4-bit Marlin weights
FLLM_ACT_CHECK=1 ...           # residual max / NaN / Inf after every layer of the first two passes (also with f32: tells the FP16 headroom)
fllm-rs selftest-gpu --device cuda --only actdtype
```

### 1.2 Marlin small-M GEMV: `FLLM_MARLIN_SMALL_M=auto|0|1`
GEMV for 1 to 4 rows over the Marlin weight layout (`cuda/marlin_vllm/fllm_marlin_gemv.cu`); `auto` (default) compares and times it against Marlin the first time a shape
is seen and keeps it only if the results agree and it is faster. Note: **`auto` is on by default**; `FLLM_MARLIN_SMALL_M=0` is the old behaviour. docs/marlin-small-m.md.
```
fllm-rs bench-kernel --op marlin_gemv --m 1 --n 2048 --k 2048 --iters 200
fllm-rs selftest-gpu --device cuda --only smallm
FLLM_CUDA_TUNE_LOG=1 fllm-rs serve ...     # one line per (shape) decision
```

### 1.3 Fused QK-norm + rotary embedding + KV write: `FLLM_FUSE_QK_NORM_ROPE=1`
One kernel per layer for models with per-head q/k norms (Qwen3, Gemma 3, OLMo style with per-head weights): norm, rotation and f16 cache write. FP32 and narrow
pipelines. Default off (the unfused path stays bit-identical). Only norm-before-rotation (`qk_norm_after_rope` models and full-width norms use the old path).

### 1.4 Merged projections: `FLLM_MERGE_PROJ=1`
q/k/v as one Marlin matrix and gate/up as one (7 GEMMs per layer become 4), concatenated at upload (no extra memory). Needs `FLLM_ACT_DTYPE=f16|bf16`, Marlin 4-bit
weights without activation permutation, a dense gated FFN with SiLU or GELU, no q/k/v clipping, no gated attention. The GEMM output is split by one kernel, or read with
row strides by the rotation + cache kernel; the gated activation reads both halves of the gate/up row. Default off: the merged GEMM sums in another order than the
three separate ones, so the last bits can differ (the selftests hold the difference to a few units of the roundoff). With FP32 activations the existing norm/GLU/conversion
fusions of the separate projections are better, so the switch is ignored there (with a log line).

### 1.5 Programmatic dependent launch: `FLLM_PDL=1`
The narrow-activation kernels and the GEMV start their blocks while the previous kernel finishes (`griddepcontrol`, sm_90+); results are unchanged (selftest
`actdtype ... pdl` compares bits). Default off. Marlin, CUTLASS and FlashInfer do not trigger early (they gain nothing, they are not harmed).

### 1.6 Workspace reservation: `FLLM_WORKSPACE_ROWS=N`
Sizes the scratch pool and the library buffers (Marlin activations/partials/locks, conversion scratch, attention workspace, narrow scratch) once for batches of N rows
(layers up to the wider of hidden and ff), so no allocation, free or synchronisation happens later in a run of that size. Default 0 (off, the buffers grow on demand as before).

### 1.7 BF16 Marlin kernels
`generate_kernels.py` also instantiates the 4-bit AWQ (zero points) and GPTQ (symmetric) kernels for BF16 activations, scales and results (sm_80+); the generated
files and `kernel_selector.h` are committed like the FP16 ones. `fllm_vllm_marlin_gemm_t(..., bf16)` selects them. NVFP4/FP8 Marlin stay FP16. `build.rs` compiles the
BF16 files for sm_80 and newer targets only (the template has no BF16 scalar type below sm_80); with no such target in `FLLM_CUDA_ARCHS` they are built for the PTX of compute_80 so that
the kernel selector links, and BF16 activations are refused at run time on GPUs below sm_80.

## 2. Environment variables and flags

| name | default | meaning |
|---|---|---|
| `FLLM_ACT_DTYPE` | `f32` | `f32`, `f16`, `bf16`, `auto` (see 1.1). A model the narrow pipeline cannot run stays f32 with a log line. |
| `FLLM_ACT_CHECK` | off | `1`: largest residual value, layer, FP16 headroom and first NaN/Inf layer for the first two forward passes (synchronizes per layer). |
| `FLLM_MARLIN_SMALL_M` | `auto` | `0` Marlin only, `1` GEMV whenever the shape is covered, `auto` decide per shape by comparing and timing. |
| `FLLM_MARLIN_GEMV_SPLITS` | auto | number of K splits of the GEMV grid. |
| `FLLM_MARLIN_SMALL_M_LOG` | off | one log line per auto decision (also `FLLM_CUDA_TUNE_LOG=1`). |
| `FLLM_FUSE_QK_NORM_ROPE` | `0` | fused per-head q/k RMS norm + rotation + cache write. |
| `FLLM_MERGE_PROJ` | `0` | merged q/k/v and gate/up matrices (narrow activations only). |
| `FLLM_PDL` | `0` | programmatic dependent launch for the kernels of this change (sm_90+). |
| `FLLM_WORKSPACE_ROWS` | `0` | reserve scratch for N rows at start-up. |
| `fllm-rs bench-kernel --op marlin_gemv [--m --n --k --iters --kind awq|gptq --group --device]` | | micro benchmark: us per call of Marlin (FP32 and FP16 reduce) and of the GEMV. |
| `selftest-gpu --only actdtype` / `--only smallm` | | new groups (`--list` shows them). |

`FLLM_MARLIN_PREFILL_CFG` keeps its default (`vllm`): the `smallm` group prints, for five batch sizes, how many values differ in any bit between `fllm` and `vllm`; it may
become the default only if that is 0 of N everywhere (the notes of the previous work say it will not be).

## 3. THIRD_PARTY entries

* **vLLM `csrc/quantization/marlin` kernels (Apache-2.0)**: the existing entry covers the sources in `cuda/marlin_vllm`. New here: `generate_kernels.py` is extended to
  instantiate the `kU4` / `kU4B8` kernels for `kBFloat16` (templates unchanged, sm_80 and newer), two generated files
  (`sm80_kernel_bfloat16_u4_bfloat16.cu`, `sm80_kernel_bfloat16_u4b8_bfloat16.cu`) and the matching `kernel_selector.h` entries.
* **IST-DASLab/marlin and vLLM (Apache-2.0)**: `fllm_marlin_gemv.cu` is original code; it reads the weight layout written by `gptq_marlin_repack_kernel` /
  `awq_marlin_repack_kernel` and the scale/zero-point permutations of `marlin_permute_scales` / `marlin_zero_points` (layout facts only, noted in the file header).
* **FlashInfer (Apache-2.0)**: the existing wrapper additionally accepts FP16 `q` and output arrays (no FlashInfer code changed).
* Programmatic dependent launch (`griddepcontrol`) follows the CUDA Programming Guide; no third-party code.

## 4. Test levels

| item | level | what was run |
|---|---|---|
| Rust: policy (`act_dtype`), `bench-kernel` parser, existing tests | L0 + unit tests | see REPORT.md for the exact `cargo test` / `clippy` results |
| all CUDA sources | L0 | `nvcc` for sm_75, 80, 89, 90, 120 (file by file) and the full crate build with `FLLM_CUDA_ARCHS=80` (and 75, 120: see REPORT.md) |
| `selftest-gpu --only actdtype,smallm` | written, **not run** | needs a GPU (section 5) |
| everything about speed | **not measured** | numbers in docs/activation-dtype.md and docs/marlin-small-m.md are byte counts and hypotheses |

## 5. Colab run (one session, T4 / A100 / H100 / G4 = RTX PRO 6000 Blackwell)

1. Build as usual for the GPU; then `fllm-rs selftest-gpu --device cuda --only smallm,actdtype 2>&1 | tee selftest.txt`. A FAIL line names the case, the worst error and the
   tolerance; the tolerances are derived in the header of `selftest/actdtype.rs` and may be wrong by a factor (report the printed worst element instead of editing them).
   On a T4 (sm_75) the BF16 cases are skipped; PDL cases are plain launches below sm_90.
2. `fllm-rs bench-kernel --op marlin_gemv --m 1 --n 2048 --k 2048` and the shapes of the table in docs/marlin-small-m.md (`--n 256`, `--n 11008`, `--k 11008`, `--m 4`). Expected if the
   hypothesis holds: GEMV 3 to 5 us at 2048 x 2048 (about 1 to 1.4 TB/s on the RTX PRO 6000), Marlin 10 to 13 us, "marlin (fp16 reduce)" a little below Marlin.
3. Serving A/B on one model (Qwen2.5-3B AWQ, 8B AWQ), single user, greedy, text compared with the default run: decode and prefill at 355/1000/1863/4000/8000 tokens with
   `FLLM_MARLIN_SMALL_M=0` vs default; then add `FLLM_ACT_DTYPE=f16` (and `bf16`); then `FLLM_MERGE_PROJ=1`; then `FLLM_PDL=1` (H100/G4); `FLLM_FUSE_QK_NORM_ROPE=1` on Qwen3.
   Expected: the text is identical or differs after some tokens only (FP16/BF16 rounding); decode with the GEMV faster; prefill with narrow activations 3 to 9 % faster.
4. `FLLM_ACT_CHECK=1 FLLM_ACT_DTYPE=f32` once per model family you care about: the printed FP16 headroom decides whether `f16` is safe; below 4x use `bf16`.

## 6. Known limits

* No narrow path for: recurrent layers, MoE, MLA, gated attention, LayerNorm, q/k/v clipping, parallel residual, layers on the host, MTP, NVFP4/FP8/K-quant *GEMMs* (they run on FP32
  copies; the K-quant GEMV reads the narrow array for the int8 quantization). NVFP4 models should keep `f32`; `auto` leaves them there.
* The GEMV covers M <= 4, N % 64 == 0, K % 128 == 0 (64 for group 32), no bias, no act-order; BF16 GEMV only through the dispatcher.
* `FLLM_MERGE_PROJ`: Marlin 4-bit only; K-quant and NVFP4 merging (equal per-tensor scale or folded block scales) is not implemented.
* **Not implemented:** CUDA graphs for prompt chunks (`FLLM_CUDA_GRAPH_PREFILL`). Reason and the measurement to take first are in docs/activation-dtype.md section 4. The
  host-side causes of idle time on the prompt path were removed earlier (step cache); a piecewise graph around attention needs the forward pass restructured into segments
  that can be skipped on replay (buffer identities, padded rows that attention must not see, per-segment staging arrays), which was not attempted blind.
* Fusing the SwiGLU into the int8 quantization of the down projection for the narrow K-quant path, and NVFP4 quantization directly from FP16 activations, are not done (the
  FP32 pipeline has both already).
* The `auto` decision of the GEMV and the compare of its first call cost one stream synchronisation per new shape; the first request after start is slower by a few ms.
* The policy cannot see the checkpoint dtype of a quantized embedding table (see docs/activation-dtype.md).
