# Marlin for one to four rows (decode)

Measured on an RTX PRO 6000 Blackwell (nsys, Qwen2.5-3B AWQ, one user): Marlin is 76 % of the kernel time of a decode step and a call for a 5.5 MB
matrix takes 12.5 us where the memory system needs 3 us. This note says where the time can go, what was added to find out and to remove it, and what
was not measured (no GPU was available to the author; every statement about speed below is a hypothesis until `selftest-gpu --only smallm` or
`bench-kernel` has run).

## What the Marlin host code does for M <= 4

`marlin_vllm/marlin_vllm.cu` is vLLM's `marlin.cu` host code. For `prob_m <= 16` it picks `thread_m_blocks = 1` (the 16-row tile; for `prob_m <= 8` the
variant that processes 8 rows) and one of `{128,128,256}`, `{64,128,128}`, `{128,64,128}` as (`thread_k`, `thread_n`, threads), and the {128,64,128}
configuration when `n / thread_n * m_tiles * 4 <= SMs`. It always launches `SMs * blocks_per_sm` blocks of a persistent kernel; each block takes a
contiguous range of the (K-tile, N-tile) work (stream-K) and blocks that share an output tile combine their partial sums through a lock array
(`workspace`, zero between calls, reset by the kernel itself: no memset kernel runs per call) and, with `use_fp32_reduce` (on whenever the caller passes
`C_tmp`, as fLLM does), through FP32 partials in `C_tmp`.

Where the difference to the bandwidth bound can come from (none of these is measured; each has a switch or a number in the bench output):

| suspect | why it hurts at M = 1 | how to see it |
|---|---|---|
| prologue of the 4-stage `cp.async` pipeline | every block fills 4 stages of weight tiles and the activation tile before its first MMA: several memory latencies (about 1 us each) of fixed cost per call, for a matrix that takes 3 us to stream | `bench-kernel` for growing `--k`: a fixed offset of 5 to 8 us in the Marlin column but not in the GEMV column |
| lock-ordered split-K | for small N the K range of an output tile is cut over many blocks; they add their partials one after the other (`barrier_acquire` / `barrier_release`) | `FLLM_MARLIN_CFG=128,64,128` (existing) changes the cut; the `GEMV` has its own cut (`FLLM_MARLIN_GEMV_SPLITS`) |
| FP32 reduction through `C_tmp` | extra traffic and a second pass over the tile | the column "marlin (fp16 reduce)" of the benchmark runs the same kernel without `C_tmp` (vLLM's `use_fp32_reduce=False`) |
| grid larger than the work | `SMs` blocks for 32 output tiles of 64 columns: most blocks only take a few K tiles | compare `n / 64` with the SM count in the benchmark line |
| launch and ramp | about 2 us from the launch to the first byte of weights for any kernel | the GEMV pays it too; PDL (`FLLM_PDL=1`) hides it behind the kernel before |

## The GEMV (`marlin_vllm/fllm_marlin_gemv.cu`)

For 1 to 4 rows, AWQ (zero points) and symmetric GPTQ, group 32/64/128, K a multiple of 128 (64 for group 32), N a multiple of 64, no bias:

* grid (N / 64, K splits), 128 threads; lane `l` of a warp loads one 16-byte word per 16 x 64 tile (the 32 codes of its four K rows and eight columns, in the
  layout `gptq_marlin_repack_kernel` writes; the nibble order and the scale / zero-point permutations are in the header of the file), a warp walks "units" of
  2 or 4 K tiles inside one quantization group, so scale and zero point are loaded once per unit; the activations come from global memory (L1/L2 hits);
* `sum_k x_k q_k` and `sum_k x_k` are accumulated in FP32 per group and combined as `s * (sum x q - z * sum x)`; the four lanes that share columns and the four
  warps are reduced with shuffles and shared memory; K splits go through an FP32 scratch (`C_tmp`, big enough by construction: `sms * 16 * 256` floats) and a ticket
  counter per column tile (the lock array, left at zero): the last block of a tile adds the partials in a fixed order and writes the FP16 (or BF16) result.
  The kernel is deterministic and graph-safe;
* bytes read per call: the weights once, the scales and zero points once per unit, the activations from cache; no shared-memory staging, no `cp.async`.

Numerics: the dequantization is done in FP32 (`s * (q - z)` is never rounded to FP16 as in Marlin), so the result is a little more accurate than Marlin's; the check
tolerance against Marlin is 5e-3 of the largest result (both round to FP16).

### Selection: `FLLM_MARLIN_SMALL_M=auto|0|1` (default `auto`)

* `0`: Marlin only. `1`: the GEMV whenever the shape is covered.
* `auto`: the first call of every (device, N, K, M, format, group) runs both kernels on the real data, compares the results (the GEMV is dropped for
  that shape if they differ by more than 1 % of the largest value: this catches a layout mistake without any model-specific knowledge) and times 8 launches of each
  (the GEMV is kept when it is at least 5 % faster). The decision is cached. During CUDA graph capture a shape without a decision uses Marlin; the ordinary pass
  that precedes every capture has made the decisions. The first call of a shape costs a few extra launches and one stream synchronisation.
* `FLLM_CUDA_TUNE_LOG=1` (or `FLLM_MARLIN_SMALL_M_LOG=1`) prints one line per decision with both times.
* `FLLM_MARLIN_GEMV_SPLITS=N` forces the number of K splits (default: enough blocks to cover the device twice over, at least one unit per warp).

### Measuring

```
fllm-rs bench-kernel --op marlin_gemv --m 1 --n 2048 --k 2048 --iters 200
fllm-rs selftest-gpu --device cuda --only smallm
```
The benchmark uses random bytes as weights (they are a valid matrix for both kernels, so no repack is needed), 5 warm-up launches, `--iters` timed launches back to
back on one stream, and prints microseconds per call of Marlin (FP32 reduce), Marlin without `C_tmp`, and the GEMV, the bandwidth the GEMV reaches and the largest
difference of the results. `selftest-gpu --only smallm` runs twelve shapes of 3B to 8B models (attention projections, gate/up, down) with the same numbers in the
shape column, the linear stack with the GEMV forced on against the f64 reference, and the `auto` decision path. The same group reports whether
`FLLM_MARLIN_PREFILL_CFG=fllm` and `vllm` differ in any bit (they are expected to, see docs/performance-notes.md section 6; `fllm` becomes the default only if the line
says 0 of N values differ).

### Not done

* a grouped launch of the seven projections of a layer in one kernel (the weights of different matrices have different shapes and scales; the merged q/k/v and
  gate/up matrices of `FLLM_MERGE_PROJ` reach the same goal for 7 -> 4 launches without a new kernel);
* more than four rows (the register tile of the kernel is 4 x 8 values per lane; Marlin takes over);
* NVFP4 (FP8 scales) and FP8 weights, act-order GPTQ, a bias in the epilogue;
* BF16 is wired (`bf16` flag of `fllm_marlin_gemv`, template instantiated) and compiles, but the selftest only times FP16 weights; the narrow-activation selftests exercise the
  BF16 path through the dispatcher.
