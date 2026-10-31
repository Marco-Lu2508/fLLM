# Q4_K prompt GEMM (qgemm4k.cu)

`rust/engine/cuda/qgemm4k.cu` runs the prompt (prefill) GEMM for Q4_K weights with
`mma.m16n8k16` tensor-core instructions. Small batches go to `qgemm4k_sb_kernel`, everything
above 64 tokens to `qgemm4k_kernel`, and both are entered through `fllm_qgemm4k_try`, which
keeps its signature and its "return 0" fallbacks (compute capability below 8.0, batch below 2,
`cols % 256 != 0`, unaligned pointers, not enough scratch) unchanged.

## Where the time goes

The wide shape is a single 128x256 tile per CTA, eight warps, `WM = 8`, `WN = 1`, two
cp.async stages. Each warp therefore covers 16 rows of the weight matrix and all 256 tokens,
and per 64-value k-tile it reads

- `WM * BN * 128 B` of activations (8 * 256 * 128 = 256 KB per SM and k-tile; each warp loads
  the whole activation tile through `ldmatrix`),
- `WN * BM * 36 B` of weights (1 * 128 * 36 = 4.6 KB),

and executes `MI * NI * 4` mma instructions with `MI = BM / WM / 16` and `NI = BN / WN / 8`,
which for the wide shape is 128 mma per thread and k-tile. The accumulator block alone is 128
registers per thread, so ptxas lands at 243 registers with no spill, and the 76 KB of shared
memory allows exactly one CTA per SM. Three costs follow from that shape:

1. the shared-memory traffic for the activations, which only `BM/WM` and `BN` bound,
2. the per-thread ALU work around the mma pipeline (dequant plus 6-bit scale unpack),
3. the fact that the shape is fixed for every batch size and every GPU.

Each variant attacks one of them. All of them compute the same result as variant 0.

## Variants

Selected with `-DQG_VARIANT=n`; `QG_VARIANT` defaults to 0, which is the current code, so an
unmodified build behaves exactly as before. The variants are cumulative: each step adds to the
previous one, so variant 3 is 1 + 2 and variant 4 is 3 plus the smaller warp tiles.

| Macro | What changes | Expected effect |
| --- | --- | --- |
| `QG_VARIANT=0` | nothing, this is the current kernel | baseline |
| `QG_VARIANT=1` | `mma_util.cuh` gains `dequant_pair_lop3()`: the nibble mask and the 1024+n magic become a single `lop3.b32` (`0xea`) instead of shift, and, or | fewer integer ALU slots per dequant, which are on the critical path next to the mma pipeline |
| `QG_VARIANT=2` | `d * sc` and `dmin * mn` are computed as `__hmul2` in fp16, and the minimum is negated once per group instead of once per dequant | fewer conversions and half negations per k-tile |
| `QG_VARIANT=3` | 1 and 2 together | both of the above |
| `QG_VARIANT=4` | wide shape `128x256` with `WM = 4`, `WN = 2` instead of `WM = 8`, `WN = 1`; narrow/wide heuristic shape `128x128, WM = 4, WN = 2, 4 stages` instead of `128x128, WM = 8, WN = 1, 3 stages` | activation traffic per SM and k-tile from 256 KB to 128 KB, weight traffic from 4.6 KB to 9.2 KB, heuristic shape from 65 KB to 88 KB of shared memory |

The tile shape itself is not a variant: `fllm_autotune` already times the wide-batch candidates
at run time, and `QG_BIG_BM` / `QG_BIG_BN` / `QG_BIG_WM` / `QG_BIG_WN` / `QG_BIG_ST` (plus
`QG_S_*` for the small shapes) pin it for an A/B comparison.

Bit-identity: variant 1 moves the nibble into place with `__byte_perm` before masking, so the
mask is no longer foldable into the shift, and the `lop3` only replaces the `and`/`or` pair.
Variant 2 relies on `__hfma2` rounding `n * s2` exactly like the float product of variant 0
rounded once, which it does, because both operands are fp16. Variant 4 only changes how a block
is split over warps. The bench below checks this instead of trusting it.

`fllm_qgemm4k_try` returns 0 in the same situations as before for every variant.

## Building

```sh
# bench, one binary per variant
cd rust/engine/cuda/bench
for v in 0 1 2 3 4; do
  nvcc -O3 -arch=sm_120 --use_fast_math -std=c++17 -DQG_VARIANT=$v -I.. bench_qgemm.cu -o bq$v
done

# the server
FLLM_CUDA_EXTRA_FLAGS=-DQG_VARIANT=4 cargo build -p fllm-engine --features cuda
```

`FLLM_CUDA_EXTRA_FLAGS` is split on whitespace and appended to every nvcc invocation
(`rust/engine/build.rs`). The bench compiles for sm_75, sm_80, sm_86 and sm_90 as well; sm_75
runs the reference path only, because `FLLM_TC` compiles the tensor-core kernels out below
sm_80.

## Measuring

The Q4_K GEMM micro-benchmark harness prints, per batch, the time, the TFLOPS, the weight bandwidth and the
maximum relative difference against the fused-decode kernel in `tc_gemm.cu`; batches 512 and
1864 additionally get a `M=...` summary line. To compare two builds:

```sh
./bq0 --dump base.bin      # baseline (variant 0)
./bq1 --check base.bin     # candidate, exits non-zero above --tol (default 1e-3)
./bq1 --iters 200          # more iterations for a stable number
```

The dump holds the FP32 results of every batch >= 16; `--check` prints the maximum relative
difference and the number of differing values per batch, so "faster but different" cannot pass
silently. Note that variant 4 changes the tile shape and therefore the summation order inside a
k-tile; a small difference there is expected, a large one is not.

No speedup has been measured yet. The variants exist to be measured on the target GPU
(sm_120); this document describes what each one changes and how to compare it, not what it
achieves.

## Registers and shared memory

From `nvcc -O3 -arch=sm_90 --use_fast_math -Xptxas -v`, no kernel spills in any variant.
Shared memory per CTA is `STAGES * (BM * 48 + BN * 128)` bytes for `qgemm4k_kernel`.

| Shape (BM, BN, WM, WN, STAGES) | Registers, variant 0 / 1 / 2 / 3 / 4 | Shared memory |
| --- | --- | --- |
| 128, 256, 8, 1, 2 (wide, batch > 512) | 243 / 250 / 246 / 251 / 254 | 76 KB |
| 128, 128, 8, 1, 3 (narrow/wide heuristic, v0) | 128 / 128 / 128 / 130 / - | 65 KB |
| 128, 128, 4, 2, 4 (narrow/wide heuristic, v4) | 135 / 142 / 130 / 130 / 130 | 88 KB |
| 128, 64, 4, 1, 4 | 136 / 142 / 128 / 166 / 166 | 56 KB |
| 128, 32, 4, 1, 4 | 127 / 128 / 128 / 148 / 148 | 40 KB |
| 128, 16, 4, 1, 4 | 90 / 112 / 95 / 124 / 124 | 32 KB |

These are the shapes the file instantiates today; the autotuner candidates `fllm_autotune` times
(`run` in `fllm_qgemm4k_try`: 128x128x8x1x3, 128x128x4x2x4, 128x192x4x2x3, 128x256x4x2x2,
256x128x8x2x2, 128x96x4x1x4, 64x128x4x2x4) were not re-measured per variant here.

## Measured on an RTX 2000 Ada laptop (sm_89, 2026-10-31)

`nsys` of four 2100-token Llama-3.1-8B prompts: the wide kernel (128 x 256 tile, 14336 weight rows, 512 tokens) takes 2.35 ms = 25.6 TFLOPS, 79 % of a cuBLAS
float16 product of the same size at the same 60 W power limit (1.86 ms, 32.3 TFLOPS). `QG_VARIANT=4` changed the kernel times by -1 to -6 %, inside the 4 % by which an
unchanged kernel differed between two profiles, and the bursts by +1 % (53.6 / 52.8 against 52.6 / 52.7 tok/s): it stays off by default. `QG_TOKFAST` (default on,
`-DQG_TOKFAST=0` for the old order) numbers the blocks token tile first, so that the token tiles of one block of weight rows run next to each other and read the
weights from memory once (+1.5 % on the burst; the same greedy text with `FLLM_CUDA_NO_AUTOTUNE=1`, since the arithmetic of a block is unchanged). Details in [benchmarks.md](benchmarks.md#third-pass-2026-10-31-same-rtx-2000-ada-laptop).

## Deferred: a repacked device layout

The remaining lever is the device-side weight layout. Q4_K is currently uploaded verbatim: a
thread reads 16 nibble bytes with `ldmatrix4` and then reconstructs the fragment through a
`__byte_perm` per quarter, because GGUF order and fragment order differ. A load-time repack
into fragment order (per 256-value super-block: 128 B of nibbles, 8 u8 scales, 8 u8 mins and
the fp16 pair, about 0.578 B per weight against 0.5625 native, still lossless because Q4_K
stores scales and minima in 6 bits) would let one 128-bit load per thread feed the mma
directly and remove the nibble shuffling from the inner loop.

That repack is deliberately **not** part of this round. It touches `mmvq.cu` (which reads the
same buffer for batches 2 to 16), `decode16`/`weight_at` in `common.cuh`, `upload_weight` in
`rust/engine/src/backend/cuda.rs` and the bench reference, so it cannot be validated without a
GPU and would make the variant comparison ambiguous. It is a follow-up step, once the numbers say which
of the three costs above dominates.