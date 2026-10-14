# CPU kernels

fLLM's CPU projections, attention and element-wise kernels are written in Rust with
`std::arch` intrinsics. Every SIMD path is chosen at runtime from the features the CPU
reports; the binary is built for the baseline target and runs on any x86-64 or AArch64 CPU.
Without a usable SIMD tier the scalar fallbacks run.

Check a machine with:

```bash
fllm-rs selftest-cpu                     # the tiers up to the FLLM_CPU_ISA cap (all supported ones without a cap)
fllm-rs selftest-cpu --tier avx2         # one tier, cap ignored
fllm-rs selftest-cpu --tiers all         # every supported tier, cap ignored, plus the ragged-shape grid
fllm-rs selftest-cpu --tiers avx2,avx512-vnni --full
fllm-rs selftest-cpu --tiers all --gguf model.gguf --tensors 400   # real tensors, outlier activations
```

It prints the detected features, the tiers it runs and why (`running tiers: ...`), per kernel
and tier the largest error relative to a scalar reference and the duration (the tier is the
bracketed name in front of every line), a summary per tier, and exits non-zero when a check
exceeds its tolerance. Projections are compared with the 8-bit activations they consume, at
1e-5 of the sum of the absolute terms (measured worst case on all tiers: 1.7e-6; the same
on 197 real tensors of Llama-3.2-3B Q4_K_M: 4e-7); attention and element-wise kernels at 1e-4.

`--tiers` (a list of tier names or `all`) and `--full` add the ragged grid for every quantized
format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K, IQ4_NL, IQ4_XS, MXFP4, Q4Z and Q4ZP with groups of 32, 64
and 128) and F32/F16/BF16: 1, 7, 15, 17, 33 and 50 rows, one to ten 256-value blocks (or
1 to 25 groups) per row, prompts of 1, 4, 9, 17 and 33 tokens, and a saturation stress (every
payload byte at 0x00, 0x7f, 0x80 or 0xff against activations that quantize to +127, -127 or
alternate). With the `avx512` feature the 16-row VNNI GEMM and the AMX GEMM are also run
directly on every format with a decoder (reported as `vnni_gemm ...` and `amx_gemm ...`),
whatever their default format lists and batch thresholds say. A full run is about 22 000 checks.

Tier support of the machine and of the build: the default release build contains the tiers
`scalar`, `avx2`, `avx-vnni`, `avx512bw` and `avx512-vnni` (512-bit integer dots for Q8_0, the
classic formats and the K formats, 512-bit attention and `exp_shifted`). The Cargo feature
`avx512` adds the tiers `avx512-bf16` and `amx` and the kernels listed in
[cpu-avx512-amx.md](cpu-avx512-amx.md): the 16-row VNNI prompt GEMM, 512-bit Q2_K, Q3_K, IQ4_XS,
IQ4_NL and MXFP4 panels, 512-bit float projections and DeltaNet. Both builds pick the kernels at run time
from CPUID, so the same binary runs on any x86-64 CPU; see "Building with the avx512 feature".

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
| x86-64 | `avx512-bf16` | `avx512-vnni` + AVX512-BF16; only in builds with the `avx512` feature ([cpu-avx512-amx.md](cpu-avx512-amx.md)) |
| x86-64 | `amx` | `avx512-bf16` + AMX-TILE + AMX-INT8 and OS tile state; only with the `avx512` feature, Linux only |
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
| Q4Z, Q4ZP (AWQ, GPTQ int4; groups of 32, 64, 128), decode and prompt | AVX-512 VNNI or BW (512-bit, 4 records per step); AVX-VNNI; AVX2 (256-bit, 2 records per step); 4 rows x 1 token for decode, 4 rows x 4 tokens for prompts; see "Q4Z" below | the generic f32 path (slow) |
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

## Q4Z and Q4ZP (AWQ, GPTQ int4)

`cpu_q4z` computes the 4-bit AWQ/GPTQ tensors of [awq-gptq-marlin.md](awq-gptq-marlin.md) without
dequantizing a row per output element (the former generic path allocated an f32 row for every
output element and token: Qwen2.5-3B AWQ decoded at 6 tok/s on 24 Zen 5 threads, nine times slower
than Q4_K_M of the same size).

- The input of a call is quantized once to 32-value Q8_0-style records (fp16-rounded scale) and
  split into an even-index and an odd-index plane. The two nibbles of a code byte are adjacent
  inputs (`2j` low, `2j + 1` high), so `codes & 15` against the even plane plus `codes >> 4` against
  the odd plane is the whole dot, with no nibble shuffle. One 256-bit load of codes covers two
  records, one 512-bit load four.
- Codes are unsigned (0..15), so the integer dot is `vpdpbusd` (AVX-VNNI, AVX-512 VNNI) or
  `vpmaddubsw` + `vpmaddwd` (AVX2, AVX-512BW); a `maddubs` pair is at most 2 * 15 * 127 and the
  sum of the even and the odd half 7620, so nothing saturates.
- The zero point is applied per group from the activation sums:
  `sum((q - z) a) = sum(q a) - z * sum(a)`; the weight scale times the activation scale of each
  record scales the converted lanes (per 128-bit half, since a vector spans records of different
  scales and, with groups of 32, different groups).
- Decode runs four rows against one token per pass, prompts four rows against four tokens.
  Groups of 32, 64 and 128 are separate instantiations; a row is never read past its end (odd
  record counts use zero-padded activations and per-record loads).
- Q4ZP gathers the input into stored order first (`x_stored[c] = x[perm[c]]`), then runs the same
  kernels; the activations are therefore quantized in stored order, which differs from the
  quantization of the original order by ordinary 8-bit noise.

Measured on the AVX2-only test host (AMD EPYC 9255 VM, 10 shared threads), Qwen2.5-3B-Instruct-AWQ:
decode 2.76 -> 27.5 tok/s, prefill (100 tokens) 1.6 -> 91 tok/s; Llama-3.2-3B Q4_K_M decodes
27.3 tok/s there. The AVX-512 variants were executed under Intel SDE only (see "What is tested"). A per-kernel throughput
benchmark (decode weight bandwidth, prompt MAC/s per group size and kernel, one thread by default) is the
ignored test `cargo test --release -p fllm-engine cpu_q4z -- --ignored --nocapture`; on the AVX2 host it
shows 15 to 22 GB/s of weights per core at decode (the 2 MB test matrix is cache resident) and 51 to 59
GMAC/s per core for a 64-token prompt.

## Building with the `avx512` feature

`cargo build --release --features avx512` adds the kernels of [cpu-avx512-amx.md](cpu-avx512-amx.md)
(16-row VNNI prompt GEMM, AMX tiles, 512-bit Q2_K/Q3_K/IQ4_XS/IQ4_NL/MXFP4 panels, 512-bit float
projections, BF16 dot products, 512-bit DeltaNet) and the tiers `avx512-bf16` and `amx`. Without
it the tiers up to `avx512-vnni` still exist, but a prompt on an AVX-512 VNNI CPU runs the
streaming panels. Measured on a Zen 5 EPYC with the feature built (Colab, Llama-3.2-3B Q4_K_M,
355-token prompt): 449 tok/s on `avx2`, 417 on `avx512bw`, 661 on `avx512-vnni`, 649 on
`avx512-bf16`; decode 54 tok/s on all tiers (memory bound). The gain of `avx512-vnni` is the
16-row GEMM that only the feature contains; the default build has not been measured there.

**The feature does not make the binary less portable.** It only compiles more functions that carry
`#[target_feature(enable = ...)]` and the AMX inline assembly; nothing in the repository sets
`-C target-cpu`, a CPU `target-feature` or `cfg(target_feature)` (the Makefile, `docker/Dockerfile`, the
workflows and `scripts/build-windows.ps1` build for the baseline target), so the compiler never
emits an AVX/AVX-512 instruction outside those functions. Every call is guarded by CPUID
(`is_x86_feature_detected!`, the AMX bits read directly) and the `FLLM_CPU_ISA` cap. Checked:

- the same model gives bit-identical logprobs from the default build and the `avx512` build on an
  AVX2-only CPU (Llama-3.2-3B Q4_K_M and Qwen2.5-3B AWQ, tier `avx2`), and the binary is 0.9 %
  larger (28.28 MB against 28.03 MB);
- the `avx512` build runs `selftest-cpu` under Intel SDE as Nehalem and Sandy Bridge (no AVX2:
  scalar tier; `devices` too), Haswell (`avx2`), Alder Lake (`avx-vnni`), Skylake-X (`avx512bw`),
  Ice Lake (`avx512-vnni`) and Sapphire Rapids (`avx512-bf16`, `amx`), without an illegal instruction.

What the feature does not give you yet is silicon coverage: the AVX-512 BW/VNNI/BF16 kernels ran on
a Zen 5 EPYC (Colab, `selftest-cpu` 413 checks on every tier, text and speed of three models), AMX
and everything on Intel silicon only under SDE.

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
- **Tiers do not reproduce each other's logprobs, and that is not a kernel defect.** Every
  kernel is exact on the integers (for identical inputs the 8-bit activations are bit-identical
  across tiers, `selftest-cpu` checks it) and sums its float partial results in a tier-specific order, so two
  tiers differ in the last bit or two of each projection (relative 1e-7). The activations of the
  next projection are quantized to 8 bits again (and the keys and values go to an f16 cache), and
  that rounding turns a 1e-7 difference into a flipped quantization step in a few elements per
  layer: after one layer the hidden states of two tiers differ by 1e-5 to 1e-4, after 24 layers by
  about 2e-3 (relative L2, the size of the error of the 8-bit activations themselves). That is the
  noise floor of the method, and it is as large between two *variants of the same tier* as
  between tiers. Llama-3.2-3B-Instruct
  Q4_K_M, prompt "The capital of France is", logprobs of the first token (AVX2 default
  " Paris" -0.322, " located" -2.976):

  | run | " Paris" | " located" |
  | --- | --- | --- |
  | avx2, default | -0.322 | -2.976 |
  | avx2, `FLLM_NO_R8=1` (other panel kernels, same integer math) | -0.346 | -2.911 |
  | avx2, `FLLM_NO_REPACK=1` | -0.360 | -2.902 |
  | avx512bw | -0.352 | -2.874 |
  | avx512-vnni | -0.360 | -2.869 |
  | scalar tier (f32 activations, no 8-bit rounding) | -0.349 | -2.854 |

  Over the five best candidates the tiers differ by up to 0.16 (avx2 against avx512-vnni, token " a"),
  and two variants of the same tier (avx2 default against `FLLM_NO_REPACK=1`) by up to 0.22. The
  greedy continuation forks where the model is undecided: at the position after
  "The capital of" the top two candidates " France" (-1.702) and " Italy" (-1.721) differ by
  0.019, less than that noise. With `FLLM_X86_WIDE_INT=0` the 512-bit integer kernels are off and
  all tiers print bit-identical results to `avx2` (their float code is then the same). To compare
  tiers use the mean negative log-likelihood over several texts (`fllm-rs ppl`), not single
  logprobs. Ten texts of 159 tokens (Qwen3-0.6B Q4_K_M, paired per text, difference of the mean
  NLL against avx2): avx512bw +0.006 on average (standard deviation over the texts 0.015), avx512-vnni
  -0.002 (0.021); the two avx2 variants `FLLM_NO_R8=1` -0.002 (0.015) and `FLLM_NO_REPACK=1` +0.001
  (0.020). The means are within their standard error (0.005 to 0.007) of zero, so no tier is better or
  worse, and the scatter of the tiers equals the scatter of the variants of one tier (the avx512 rows
  ran under Intel SDE, `scripts/colab/cpu_confirm.py` repeats the comparison on real hardware). Bit-exact agreement across tiers would need one fixed float summation order in every
  kernel, attention and normalisation; it is not a goal.
- `fllm-rs selftest-cpu --tiers all --gguf FILE` is the per-tensor check behind that statement: the
  same quantized rows and the same outlier-heavy activations through every tier against an f64
  reference on the identical 8-bit activations. Llama-3.2-3B Q4_K_M, all 197 quantized matrices,
  batches 1, 6 and 17: largest error 1.8e-7 (avx2, avx-vnni), 2.9e-7 (avx512bw), 3.9e-7
  (avx512-vnni, avx512-bf16), 3.0e-7 (amx) of the sum of the absolute terms. No tier is closer to
  the reference than the others; there is no saturation (checked also with the worst-case
  stress of `--tiers all`).
- Setting `FLLM_DUMP_LAYERS=<dir>` makes the CPU decoder append the hidden state after the
  attention and the FFN branch (and the projection results) of every layer to `<dir>/l<N>_<tag>.f32`;
  `scripts/cmpdump.py A B [prefix]` prints the relative L2 difference of two such directories, which is
  how the layer-by-layer growth above was measured.
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
| AVX-512 BW/VNNI kernels (Q8_0, classic, K panels, attention, `exp_shifted`), the `avx512` feature kernels (16-row VNNI GEMM, 512-bit codebook, Q2_K/Q3_K/IQ4_XS, float, BF16, DeltaNet), AMX tiles | **Real silicon**: AMD EPYC 9B45 (Zen 5, Colab): `selftest-cpu` 413 checks per tier, three models (avx512bw, avx512-vnni, avx512-bf16; no AMX there). **Emulated**: Intel SDE 9.58 executes the whole unit-test suite (`-spr`: 325 tests, including the AVX-512 and AMX ones, which skip themselves on other hardware), `selftest-cpu --tiers all` (`-spr`: 22 101 checks on the tiers avx2 to amx; `-icl`, `-skx`, `-adl` for their tiers; 0 failed) and runs two models end to end (Llama-3.2-3B Q4_K_M and Qwen2.5-3B AWQ on avx512bw and avx512-vnni); SDE is a functional emulator, so alignment, encoding and result bits are checked but speed is not. Intel silicon, AMX silicon and Zen 4 are untested |
| Q4Z / Q4ZP kernels (`cpu_q4z`) | AVX2 on an AMD EPYC 9255 VM (unit tests, `selftest-cpu`, Qwen2.5-3B AWQ end to end with speed); AVX-VNNI, AVX-512BW and AVX-512 VNNI under SDE; no AVX-512 speed measurement yet |
| `selftest-cpu` | run on the AVX2 host and under SDE for every tier; `--tiers all --gguf` on 197 real tensors |

The unit tests compare each kernel with an f64 reference built from scalar dequantization
and the 8-bit activations the kernel consumes, at odd block counts, every panel row count and
odd token counts; they skip themselves via `is_x86_feature_detected!` when a feature is
missing. Without AVX-512 hardware run the test binary under Intel SDE (`sde64 -spr -- target/release/deps/fllm_engine-<hash>`)
and `sde64 -spr -- fllm-rs selftest-cpu --tiers all`; both are free to download from Intel. Real hardware runs
of `cargo test -p fllm-engine cpu_` and `fllm-rs selftest-cpu --tiers all` on Ice Lake or later, Zen 4/5 and
Sapphire Rapids remain the reference.

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
