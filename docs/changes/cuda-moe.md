# CUDA mixture-of-experts: device routing, expert-major sort, grouped expert GEMM, compact stacked experts

Test level of everything below that runs on a GPU: **L0** (compiles for sm_75, sm_80, sm_89, sm_90 and sm_120; the selftest cases are written but
were never run on a device). The CPU-side parts (converter, loader, reference functions) are L2-style CPU tests, see "Tests".

## Features

### 1. Device MoE layer (`rust/engine/cuda/moe.cu`, `src/backend/cuda/moe.rs`, `src/rust_decoder/moe_device.rs`)

`CudaBackend::supports_moe` is now true (sm_75 or newer, NVIDIA only) and the decoder runs a resident MoE layer entirely on the device:

1. router GEMM (`quantized_linear`), then `Stream::moe_route_ex`: softmax or sigmoid scores, selection bias (`e_score_correction_bias`),
   group-limited selection (`n_group` / `topk_group`, best-two-sum or best-one ranking), renormalisation, `routed_scaling_factor`, an
   epsilon in the denominator (LFM2-MoE); ties to the lower expert index, like the host router. One warp per token, up to 1024 experts, k up to 32.
2. `Stream::moe_align` (the `moe_align_block_size` step): sorts the token-expert pairs by expert and pads every expert to a multiple of the
   tile (16 or 64 pairs). Outputs `sorted`, `block_expert` and the padded length; the buffer sizes depend only on the number of pairs, so nothing
   is read back. `expert_offset` / `num_local` select a slice of the experts (expert parallelism: the pointer tables hold the local experts only).
3. `Stream::moe_grouped_linear`: one launch per projection over all experts. Tensor cores (mma.sync m16n8k16, FP16 inputs, FP32 accumulation),
   weights decoded straight into shared memory with the `decode16` decoders of the dense GEMM. Gate and up run in one pass with the activation
   (SwiGLU, GeGLU, gpt-oss clamped SwiGLU) in the epilogue; bias per expert; optional scaling of the activations by the routing weight
   (Llama 4). Formats: F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K (the others are converted at upload as before).
4. `Stream::moe_combine_ex`: `out[t] = sum_j w[t][j] y[t*k+j] + gate[t] * shared[t]`, fixed summation order, no atomics. The shared expert is the
   existing dense gated FFN (`gated_run`); `Stream::moe_gate_scale` computes `sigmoid(g . x)` for gated shared experts (Qwen2-MoE, Qwen3-Next).

One to four tokens still use the integer dot-product decode kernels (`moe_linear`) when all expert formats allow it; everything else uses the
grouped GEMM. Batches whose intermediate buffers would exceed 256 MB are cut into chunks of rows. `Stream::moe_route` (softmax, renormalised)
and `Stream::moe_expert_gemm` of the old trait now run on the device too; they fall back to the host for weight types the grouped kernel does
not cover.

### 2. Compact stacked experts (converter, loader)

NVFP4 and AWQ / GPTQ / compressed-tensors int4 experts stay 4-bit in the GGUF (`docs/moe-formats.md`): stacked tensors `[inputs, rows, experts]`,
for NVFP4 with one `.scale` and one `.input_scale` value per expert. Old files keep loading.

### 3. CUDA graphs (not done)

The device path reads nothing back and its buffer sizes depend only on the number of rows, so a decode step through it is a candidate for graph
capture, but `graph_eligible` still excludes every layer with MoE experts. Making it eligible needs a device run: the decode branch calls
`moe_linear`, which may reallocate its activation scratch during a capture.

### 3b. Self test and timing mode

```
fllm-rs selftest-gpu --device cuda --only moe            # routing variants, sort, grouped GEMM per format, whole layers
fllm-rs selftest-gpu --device cuda --only moe --quants q4_k,q8_0
fllm-rs selftest-gpu --device cuda --only moebench       # microseconds per call, real model shapes
FLLM_MOE_BENCH_TOKENS=1,16,256,4096 FLLM_MOE_BENCH_ITERS=50 fllm-rs selftest-gpu --device cuda --only moebench --quants q4_k
```

`moebench` prints one line per (model shape, format, token count): route, align, gate+up, down, combine and total in microseconds, the
effective weight bandwidth of the two grouped products, tokens/s, and for up to 256 pairs the time of the earlier `moe_linear` x3 path for
comparison. Shapes: Qwen3-30B-A3B (128 experts, k8, 2048/768), OLMoE (64, k8, 2048/1024), DeepSeek-V2-Lite (64, k6, 2048/1408),
Qwen3-Next-80B (512, k10, 2048/512), gpt-oss-20b (32, k4, 2880/2880). For a comparison with vLLM run its `benchmarks/kernels/benchmark_moe.py`
for the same shape on the same GPU (batch sizes 1 to 4096).

## Environment variables

| variable | default | meaning |
|---|---|---|
| `FLLM_CUDA_MOE` | on | `0` / `off` / `false` / `no`: no device MoE operations (`supports_moe` false); the decoder keeps the earlier softmax-only decode kernels and the host-routed expert path |
| `FLLM_MOE_CHUNK_MB` | 256 | size of the per-pair intermediate buffers above which a large batch is processed in chunks of rows |
| `FLLM_MOE_BENCH_TOKENS` | `1,4,16,64,256,1024,4096` | token counts of `--only moebench` |
| `FLLM_MOE_BENCH_ITERS` | 20 | timed calls per stage of `--only moebench` (after two warm-up calls) |

## THIRD_PARTY entries

* vLLM (https://github.com/vllm-project/vllm, Apache-2.0): the structure of `csrc/moe/moe_align_sum_kernels.cu` (`moe_align_block_size`:
  sorted token ids padded per expert, expert id per block, padded length), `topk_softmax` / `grouped_topk` (routing variants and the group
  selection rule) and `fused_moe` (one launch over expert-major tiles, gate/up fused with the activation). The kernels in `rust/engine/cuda/moe.cu` are written for
  this engine's layouts; **no vLLM source text was copied**, so no file carries their licence header, the header of `moe.cu` names the sources of
  the design. The Marlin MoE kernel (`moe_wna16_marlin`) and SGLang's fused MoE were studied as references only and are not used.
* The weight decoders and the tile loop are the repository's own (`common.cuh` `decode16`, `tc_gemm.cu`).

## Tests

* L0: `nvcc` builds `moe.cu` for sm_75/80/89/90/120 without warnings from the new file (ptxas at sm_80: no spills except 4 to 8 bytes in some gate/up variants of the 64-pair tile, 127 to 164 registers); `cargo clippy -D warnings` is clean for default features, `rust-gpu` and `cuda` (FLLM_CUDA_ARCHS=75).
* CPU (L2-style, the CPU backend implements the same operations): `fllm-rs selftest-gpu --device cpu --only moe` runs the whole new group against the CPU implementation (173 checks, 0 failed, 12 skipped = the old `moe_topk` / `moe_combine` cases, which the CPU backend does not have): this tests the test code, not a device.
* CPU, run in the build container: `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe` (converter tests with tiny
  Mixtral-style checkpoints in NVFP4 and int4, unit tests of the per-expert scale lookup and of the stacked row encoders, the f64 router and sort
  references).
* GPU (not run): the `moe` and `moebench` groups above. Expected on a working device: all `moe route`, `moe align`, `moe grouped`, `moe gate/up`
  and `moe layer` lines `ok` (tolerances: routing weights 2e-5 relative, GEMMs 4e-3 of the sum of absolute terms, the FP16 tensor-core bound).

## Known limits

* Native grouped kernels for stacked NVFP4 experts, Q4Z (Marlin MoE) and block-wise FP8 experts were **not** written. NVFP4 and Q4Z experts are
  stored compactly now but still execute per expert through the host-routed path (`moe_on_device`), without CUDA graphs. Block-wise FP8 experts are
  still expanded to F16 at conversion (needs a new tensor type).
* The shared expert is not overlapped with the routed experts on a second stream (the stream API has no events yet).
* The grouped GEMM streams each expert's weights once per tile row; there is no split-K or persistent scheduling. Expect it to be slower than
  vLLM's Triton `fused_moe` on very large batches until measured; `moebench` is there to find out.
* Experts beyond the first 24 alias earlier weights in `moebench` (generation cost), which flatters the L2 hit rate of models with more experts.
* Expert-parallel structure: `moe_align` takes `expert_offset` / `num_local`; nothing shards experts across devices yet. Pairs routed to experts
  outside the slice leave their rows of `y` untouched (the caller zeroes `y` if it combines them).
