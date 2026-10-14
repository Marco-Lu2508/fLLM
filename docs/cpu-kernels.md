# CPU kernels

fLLM's CPU projections, attention and element-wise kernels are written in Rust with
`std::arch` intrinsics. Every SIMD path is chosen at runtime from the features the CPU
reports; the binary is built for the baseline target and runs on any x86-64 or AArch64 CPU.
Without a usable SIMD tier the scalar fallbacks run.

Check a machine with:

```bash
fllm-rs selftest-cpu            # every supported tier
fllm-rs selftest-cpu --tier avx2
```

It prints the detected features and, per kernel and tier, the largest error relative to a
scalar reference and the duration, and exits non-zero when a check exceeds its tolerance.

## Tiers

Each architecture has a ladder of tiers. Dispatch uses the highest tier the CPU supports;
`FLLM_CPU_ISA=<tier>` caps it for the whole process (useful for A/B comparisons and for
stepping around a misbehaving kernel). A tier name of the other architecture is ignored with
a warning.

| Architecture | Tier | Required features |
| --- | --- | --- |
| any | `scalar` | none |
| x86-64 | `avx2` | AVX2, FMA, F16C |
| x86-64 | `avx-vnni` | `avx2` + AVX-VNNI (VEX `vpdpbusd`) |
| x86-64 | `avx512bw` | `avx2` + AVX512F, AVX512BW, AVX512VL |
| x86-64 | `avx512-vnni` | `avx512bw` + AVX512-VNNI |
| AArch64 (little-endian) | `neon` | Advanced SIMD |
| AArch64 | `neon-dotprod` | `neon` + FEAT_DotProd (`sdot`) |
| AArch64 | `neon-i8mm` | `neon-dotprod` + FEAT_I8MM (`smmla`) |
| AArch64 | `neon-sve` | `neon-dotprod` + SVE with vectors of 256 bits or more; opt-in with `FLLM_ARM_SVE=1` (Q8_0 decode only, see [cpu-arm.md](cpu-arm.md)) |

A capped tier keeps the lower rungs: `avx512bw` on a CPU with AVX-VNNI still uses the
AVX-VNNI kernels where no AVX-512 kernel exists. On CPUs with AVX-512 VNNI but no AVX-VNNI
(Ice Lake, Tiger Lake, Zen 4) the `avx-vnni` rung is absent.

`isa()` (also printed by `fllm-rs devices`) names only the AVX2-level x86 tier and reports
`scalar` on AArch64, because code outside the CPU kernels reads any other value as "the x86
kernels are present". `cpu_kernels::active_tier()` reports the real tier.

## Dispatch per kernel family

Order of preference, left to right; the last entry is the fallback.

| Kernel | x86-64 | AArch64 |
| --- | --- | --- |
| Q8_0 projection, decode | AVX-512 (VNNI, BW), 8 rows, 2 blocks per vector; AVX-VNNI 8 rows; AVX2 4 rows | `sdot`, 8 rows |
| Q8_0 projection, prompt | AVX-512 4x2 tile (row pairs per vector); AVX-VNNI/AVX2 4x2 tile | `smmla` 4x2 tile; `sdot` 4x2 tile |
| Q4_0, Q5_0, prompt (2+ tokens) | Q8_0 tiles on repacked panels: AVX-512 row-pair tile, AVX-VNNI/AVX2 4x2 tile | NEON W panel with `smmla` or `sdot` |
| Q4_1, Q5_1, prompt | AVX-512 W-panel tile; AVX-VNNI/AVX2 W-panel tile | NEON W panel with `smmla` or `sdot` |
| Q4_0, Q4_1, Q5_0, Q5_1, decode | AVX-512 (8 rows); AVX-VNNI and AVX2: direct decode into the dot, 4 rows | NEON decode |
| Q2_K..Q6_K, prompt (4+ tokens) | AVX-VNNI/AVX2: decoded panels (below); AVX-512: streaming panels (8 rows decode, 4x2 prompt) | Q4_K..Q6_K: W panel with `smmla` or `sdot`; Q2_K, Q3_K: unpacked group panels with `smmla` or `sdot` (L0) |
| Q4_K, Q5_K, Q6_K, decode and 2-3 tokens | AVX-512 panels; AVX-VNNI; AVX2 (streaming, 4 rows) | NEON decode |
| Q2_K, Q3_K, IQ4_XS, decode | AVX-VNNI; AVX-512 VNNI (EVEX, 256-bit); AVX2 | `sdot`, 4 rows, groups decoded into registers (L0) |
| IQ4_XS, prompt | as decode | unpacked group panels with `smmla` or `sdot` (L0) |
| IQ4_NL, MXFP4 | AVX2 codebook kernel | `tbl` codebook decode with `sdot` / `smmla` (L0) |
| F32, F16, BF16 | AVX2 + FMA | NEON FMA, `bfdot` for BF16 with FEAT_BF16 (L0) |
| Gated DeltaNet step | AVX2 | NEON, bit-identical to scalar (L0) |
| Decode attention (f16 and Q8 KV) | AVX-512; AVX-VNNI/AVX2 | `sdot` + NEON |
| `swiglu` | AVX2 | NEON |
| `exp_shifted` (softmax) | AVX-512 (masked tail); AVX2 | NEON |
| Activation packing, Q8 KV encode | AVX2 | NEON |

All other packed formats (IQ1/IQ2/IQ3, TQ, NVFP4, ...) decode one f32 row per output row on
demand (`compute.rs`), as does every format on a CPU without a kernel.

Sibling projections (Q/K/V, gate/up) share one activation packing and one work queue:
Q8_0 groups, float groups, and packed groups whose formats consume the same activation
records. The K formats (Q2_K..Q6_K, IQ4_XS) mix freely, so the Q4_K and Q6_K members of a
Q4_K_M layer share one packing; every other packed format groups only with itself.

## Prompt-time repacking

llama.cpp repacks Q4_0, Q4_K and Q2_K into interleaved 4/8-row layouts when the model loads.
fLLM does not keep a second copy of the weights. A worker that claims a four-row panel of a
projection unpacks that panel into its own scratch (once per panel, at most 14 KB for K formats)
and runs every token of the prompt over it, so each weight block is decoded once per prompt
instead of once per token pair, and the scratch is gone when the projection returns. Memory
use is unchanged, there is no load-time step and the unpacked panel adapts to any core count.
`--no-repack` (or `FLLM_NO_REPACK=1`) selects the previous streaming kernels for A/B runs.

| Format | Panel layout | Kernel |
| --- | --- | --- |
| Q4_0, Q5_0 | `WPanel`: four rows of 32 signed weights plus 128 as bytes and one scale each (a Q4_0 weight `d * (q - 8)` is a Q8_0 weight after the offset) | the Q8_0 prompt tiles: 4x2 on AVX2/AVX-VNNI, row pairs on AVX-512; activations are the Q8_0 pairs of `Q8Input` |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | `KBlock` per 256-value block and row: unsigned weights (one byte each), scale words in the order `packs_epi32` gives (`[a0 x4, b0 x4, a1 x4, b1 x4]` per group pair), correction coefficients per 16-value half, `d`, `dmin` | `predecoded_avx2` / `predecoded_vnni`: per group pair two loads, two dots (`maddubs` or `vpdpbusd`), `packs`, one `madd` with the scale words |

For the K formats the offset (Q6_K 32, Q3_K 4) or minimum (Q2_K, Q4_K, Q5_K) of a row is one more
`madd` of the coefficients with the activation sums of the sixteen 16-value halves (the eight
32-value groups for Q4_K/Q5_K). Those sums are computed once per call and token, not per panel.
Offsets are subtracted exactly in the integer domain; minimums stay a float correction with
`dmin`, as before. A row is decoded in chunks of eight blocks so a chunk (14 KB) and the
activations of a token pair (8 KB) fit in L1 whatever the row length; the partial sums of a
row wait between chunks in a 128-byte-per-token buffer. The chunking does not change the
accumulation order.

Used from 4 tokens (Q4_0/Q5_0: from 2) on AVX2 and AVX-VNNI. Not repacked, with the reason:

- AVX-512 K formats keep their streaming kernels (the 512-bit paths cannot be tested here).
- NEON: every packed format with a NEON kernel unpacks panels (Q4_0..Q6_K; Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4 in `cpu_arm/quant.rs`).
- Q4_1, Q5_1: rare; they keep the W-panel tile.
- IQ4_NL, IQ4_XS, MXFP4: signed codebook weights would need the sign-trick dot, which saturates
  `maddubs` on unsigned data; left on the streaming kernels.

## Scheduling on hybrid CPUs

Every projection is one atomic queue of row panels claimed by all rayon workers. A claim is 16
rows (64 bytes of output per token) while many panels remain and shrinks to a single panel
towards the end of the queue (a worker takes at most a quarter of its share of what is left), so
a slow E-core cannot hold the tail of a projection while P-cores idle. `--threads` sizes the pool
and `--pin-cores` sets the process affinity before the pool starts, so pool threads inherit it;
neither was changed. Whether leaving out low-power E-cores helps decode is still unmeasured.

## MoE decode (fused expert passes)

On the host, one decoded token used to run each active expert as its own pair of fork/joins
(gate+up, then down) with fresh buffers: eight experts meant sixteen joined queues per layer.
`cpu_quantized::linear_experts` runs all gate and up rows of the active experts in one queue and
all down rows in a second one, with the (serial) gate activation in between and every buffer
kept in `ExpertScratch`. The activations are quantized once per pass, the down projection of
expert `e` reads the activated row `e`. Row panels are claimed 64 rows at a time (`row_panels_claim`;
dense decode stays at 16) so a core streams a long contiguous block of one expert, and each panel
software-prefetches the next one, because a Q4_K down row is only 432 bytes and the hardware
prefetcher barely ramps up inside it.

Every output row is computed by the same kernel from the same activation records as before, and
the experts are accumulated in ascending expert order, so the result is bit for bit that of the
per-expert path (`fused_experts_equal_the_per_projection_path_bit_for_bit`). The path applies to
one token, gated experts with K-format weights (Q2_K..Q6_K, IQ4_XS) on x86-64, without
`input_scaled`; prompts (several tokens per expert group) keep the grouped per-expert loop, which
reads each expert once. Q8_0/classic experts, aarch64, shared experts and plain up/down experts
run the old path.

## Accuracy

- The packed integer kernels quantize activations to 8 bits: Q8_0-style records with an
  fp16 scale per 32 values for the 32-value formats (Q4_1/Q5_1 additionally use an
  fp16-rounded record sum for their offset, like llama.cpp's Q8_1), and Q8_K-style records
  with one f32 scale per 256 values for the K formats. This is the same activation
  quantization llama.cpp's CPU backend uses.
- The f32 fallback (`compute.rs`) multiplies dequantized weights with full-precision
  activations. **The SIMD and fallback paths therefore do not agree bit for bit.** Measured
  on our hardware: per projection, the SIMD result differs from the f32 reference by 0.3 to
  0.4 %; over whole models (8 random layers) the logits differ between the paths by 2 to 7 %
  of the logit range. This is expected 8-bit activation noise, not a kernel defect.
- Different SIMD tiers consume the same 8-bit activations (the SIMD packings follow the
  scalar packing step for step, and `selftest-cpu` compares them bit for bit) but sum in a
  different order and width, so their outputs differ in the last bits. No tier is
  bit-identical to another.
- Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4 used the f32 row decode before they got packed
  kernels; on CPUs with a kernel they now see the 8-bit activation error as well.
- The SIMD `exp` is a degree-7 polynomial (below f32 rounding on its reduced range); the
  scalar fallback uses `f32::exp`.
- The f32 fallback rejects non-finite dequantized weights; the packed kernels do not check.

## What is tested

| Path | Status |
| --- | --- |
| AVX2-level projections and attention (AVX2, AVX-VNNI) | run on hardware by us; existing unit tests |
| Prompt repacking: Q4_0/Q5_0 Q8_0 tiles, Q2_K..Q6_K decoded panels | AVX2 kernels run by us (unit tests against the f64 reference and the streaming kernels, rows of 1 to 11 blocks, 1 to 9 tokens; `selftest-cpu --tier avx2` with batches 1 to 64); the AVX-VNNI instances are compiled and share the code through macros, their unit tests skip unless AVX-VNNI is present; the AVX-512 tile calls are the Q8_0 ones |
| NEON `sdot` and `smmla` paths (Q8_0, Q4_0..Q6_K) | compiled for aarch64-unknown-linux-gnu and run on hardware by us |
| NEON Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4, float formats, DeltaNet step, SVE Q8_0 decode | compiled (check, clippy, codegen) for aarch64-unknown-linux-gnu; the layout and lane algebra is tested on x86 through scalar models; **not run on ARM hardware** (L0, see [cpu-arm.md](cpu-arm.md)) |
| AVX-512 BW/VNNI kernels (Q8_0, classic, K panels, attention, `exp_shifted`) | compiled; reviewed line by line against the AVX2 and scalar paths; unit tests written but **not run** (no AVX-512 hardware available) |
| EVEX VNNI Q2_K/Q3_K/IQ4_XS panels | compiled; unit test written, not run |
| IQ4_NL/MXFP4 codebook kernel, Q2_K/Q3_K/IQ4_XS AVX2 panels | unit tests written, not run here |
| `selftest-cpu` | compiled, not run here |

The unit tests compare each kernel with an f64 reference built from scalar dequantization
and the 8-bit activations the kernel consumes, at odd block counts, every panel row count and
odd token counts; they skip themselves via `is_x86_feature_detected!` when a feature is
missing. Run `cargo test -p fllm-engine cpu_` and `fllm-rs selftest-cpu` on AVX-512 hardware
(Ice Lake or later, Zen 4/5, Sapphire Rapids) before relying on the AVX-512 tier.

## Not implemented

- **AMX-INT8** (Sapphire Rapids and later): the tile intrinsics (`_tile_loadd`,
  `_tile_dpbssd`, ...) are nightly-only in Rust 1.98 (`x86_amx_intrinsics`), and Linux
  requires `arch_prctl(ARCH_REQ_XCOMP_PERM)` before the first tile instruction. An AMX
  prompt kernel would need inline assembly plus a repacked 16x64-byte weight layout; it is
  a candidate for prefill on Xeon, not for decode.
- **AVX-VNNI-INT8** (`vpdpbssd`, Arrow Lake, Lunar Lake; intrinsics stable since Rust 1.89):
  a signed-by-signed dot would replace the sign-transfer `maddubs` sequence in the IQ4_NL,
  IQ4_XS and MXFP4 kernels and the bias trick of Q8_0. Not implemented.
- **SVE/SVE2**: one opt-in kernel (Q8_0 decode, 256-bit vectors or more); the NEON kernels run on all other SVE machines. See [cpu-arm.md](cpu-arm.md).

## Performance notes

Implemented without measurements (each is a strict reduction of work):

- Shared activation packing for mixed K groups and for all packed formats, and grouping on
  AArch64 (previously each projection repacked the input and woke the thread pool).
- Q4_K/Q5_K minimum correction with one `vpmaddwd` instead of `vpmulld`, scales decoded
  once per row block.
- Attention picks its kernel path once per call instead of probing CPU features per key.

Hypotheses for the remaining gap to llama.cpp (earlier measurements on a Core Ultra 5 235U:
Q4_K prefill 12 to 35 % and decode about 29 % behind). Not implemented because the benefit
could not be measured:

1. **K-quant prefill re-decodes weights per token pair** (implemented, see "Prompt-time
   repacking"; not measured by us). The K panels decode each 256-value
   block once per two tokens. llama.cpp repacks Q4_K into interleaved 8-row blocks at load
   (`block_q4_Kx8`) and runs 4x8 GEMM tiles. Candidates: decode a row panel once into int8
   values plus i16 scale vectors and reuse it for every token of the prompt, or repack at
   load; on AVX-512 (32 registers) a 4x4 token tile.
2. **Decode panel height.** AVX2 K decode uses 4 rows per pass; an 8-row pass halves the
   per-token setup (group sums, Q6_K lane biases) per row, as it did for Q8_0 on AVX-VNNI,
   but risks spills with 16 YMM registers.
3. **Hybrid cores.** Each projection is one dynamic queue over all Rayon workers
   (`rayon::broadcast`); claims now shrink at the tail. Dynamic claiming balances P and E cores, but low-power E cores
   (outside the compute tile on Meteor/Arrow Lake U) are much slower; excluding them with
   `--pin-cores` may help decode. Not measured.
4. **AMD CCDs and NUMA.** Threads are not pinned; mmap'ed weights land wherever the page
   cache placed them. On multi-CCD parts, giving each CCD a contiguous row range would keep
   each CCD's L3 hot. Not measured.
5. **Prefetching.** Decode streams 4 to 8 weight rows at once, which hardware prefetchers
   track; software prefetch has not been tried.
6. **False sharing** is already avoided: a worker claims at least 16 output rows (64 bytes
   per token row) at a time.
7. **Activation record size.** An `X` record stores 32 values in 64 bytes (scale, sums,
   padding). For long prompts the packed activations are twice the size of a compact layout.
8. **AVX-512 clocks.** On Skylake-SP/Cascade Lake 512-bit code can lower the core clock;
   compare with `FLLM_CPU_ISA=avx2`.

Expected, not measured (no benchmarks were run by the author of these changes):

- Q4_0/Q5_0 prompts should move to the speed of the Q8_0 tile, which was ahead of llama.cpp in the
  reviewer's table; the old Q4_0 tile recomputed activation inits, scale products and broadcasts
  per row and token.
- Q6_K and Q2_K prompts should gain the most (the decode was about half of their instructions);
  Q5_K less; Q4_K and Q3_K little (their decode is short) and they may be neutral.
- Decode: only Q4_0/Q5_0/Q4_1/Q5_1 on AVX-VNNI changed (no panel stores and reloads). The K
  formats' decode kernels are unchanged; the per-panel recomputation of activation sums is
  about 3 to 8 percent of a panel and was not removed.
- Weight bytes times token/s against memory bandwidth is the measure for decode. From the reviewer's
  table (TinyLlama 1.1B, rough weight sizes) Q8_0 reaches about 71 GB/s and Q4_0 and Q6_K 34 to 36
  GB/s, which points at instructions per byte (most visibly on E-cores) rather than at DRAM.
