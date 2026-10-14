# ARM CPU kernels (AArch64)

This page covers the AArch64 side of the CPU kernels: what exists, how a kernel is chosen, how it
is laid out and, above all, **how far it has been verified**. The shared description of the tier
mechanism is in [cpu-kernels.md](cpu-kernels.md).

## Verification level

Prompt/decode kernels that existed before this work (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K
with `sdot`/`smmla`, decode attention, `swiglu`, `exp_shifted`, activation packing) were run on
hardware by their authors. Everything marked **L0** below was written without access to an ARM
machine:

| Level | Meaning |
| --- | --- |
| L0 | compiles (`cargo check`, `cargo clippy -D warnings`, and a full release codegen of the library) for `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin` (check/clippy); unit-tested where the logic is platform independent |
| L1 | `fllm-rs selftest-cpu` passes on real ARM hardware |
| L2 | a real model produces the same greedy text as the scalar reference |
| L3 | measured against the best engine on the same CPU |

| Kernel | Formats | Status |
| --- | --- | --- |
| `sdot` decode (4 rows) and `sdot` / `smmla` prompt panels | Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4 (`cpu_arm/quant.rs`) | L0 |
| NEON F32 / F16 / BF16 projection, `bfdot` for BF16 with FEAT_BF16 (`cpu_arm/float.rs`) | F32, F16, BF16 weights | L0 |
| f16 encode (KV cache) and input rounding | F16, BF16 | L0 |
| NEON DeltaNet recurrence step (`cpu_arm/delta.rs`) | Gated DeltaNet layers | L0 |
| SVE `sdot` decode matvec (`cpu_arm/sve.rs`), tier `neon-sve`, opt-in | Q8_0 | L0 |
| Everything listed as existing above | Q8_0, Q4_0/1, Q5_0/1, Q4_K..Q6_K | L1 by the earlier authors, not re-run here |

The platform-independent parts are tested on x86 (`cargo test -p fllm-engine cpu_arm`): the block
decoders against `compute::dequantize` bit for bit, and a scalar model of every NEON kernel's
dataflow (`cpu_arm/model.rs`, built from scalar models of `sdot`, `smmla`, `tbl` and the 64-bit
zips in `cpu_arm/emu.rs`) against the f64 reference at odd shapes. That covers block layout, scale
decoding, which `smmla` lane holds which row and token, tails, and the minimum term; it cannot
cover the intrinsics themselves. The AArch64-only tests (`cargo test -p fllm-engine cpu_arm` on an
ARM machine) compare the NEON decoders with the scalar ones bit for bit and the kernels with the
reference.

## Tiers

| Tier | Needs | Used for |
| --- | --- | --- |
| `neon` | Advanced SIMD (baseline) | activation packing, element-wise kernels, float projections, DeltaNet, f16 encode |
| `neon-dotprod` | FEAT_DotProd | `sdot` projections and attention (decode and prompt) |
| `neon-i8mm` | FEAT_I8MM | `smmla` prompt panels; `bfdot` when FEAT_BF16 is also present |
| `neon-sve` | FEAT_SVE with vectors of 256 bits or more, **and** `FLLM_ARM_SVE=1` | Q8_0 decode matvec |

`FLLM_CPU_ISA=<tier>` caps dispatch (and `neon-sve` selected this way is opted in). `fllm-rs devices`
prints the active tier, the supported tiers and the detected features; `selftest-cpu` also prints the SVE vector
length. The detection is the standard library's (`getauxval` on Linux and Android, `sysctl` on macOS,
`IsProcessorFeaturePresent` on Windows); the SVE length is read with `rdvl`. The binary is built for
the baseline target and needs no `-C target-cpu`; there is no Cargo feature to turn (the `arm-simd`
feature is accepted and has no effect, NEON being the AArch64 baseline).

Which kernels each CPU family is expected to take (from its feature set, not from a run):

| CPU | Features | Tier |
| --- | --- | --- |
| Apple M1 | NEON, dotprod, fp16 | `neon-dotprod` |
| Apple M2 and later | + i8mm, bf16 | `neon-i8mm` |
| Cortex-A76/A77/A78/X1, Neoverse N1 | NEON, dotprod | `neon-dotprod` |
| Armv9 cores (Cortex-A510/A710/X2 and later), Neoverse N2/V1/V2, Graviton 3/4 | + i8mm, bf16 | `neon-i8mm` |
| Graviton 3, Neoverse V1 (SVE 256-bit), A64FX (SVE 512-bit, no i8mm) | SVE >= 256 bit | `neon-sve` only with `FLLM_ARM_SVE=1` |
| Raspberry Pi 4 / 5 (Cortex-A72 / A76) | A72: NEON only; A76: + dotprod | `neon` (no packed kernels, f32 row decode) / `neon-dotprod` |

## Formats

Every format is cut into 32-value *groups*. A group is decoded to signed 8-bit weights plus a scale
for each 16-value half and a coefficient of the activation half sums; `cpu_arm/formats.rs` defines
this record and the scalar decoders. The kernels compute, per group and activation record `x`:

```text
out += x.d * (d[0] * dot(q[0..16], x.q[0..16]) + d[1] * dot(q[16..32], x.q[16..32]))
     + m[0] * x.sum[0] + m[1] * x.sum[1]
```

| Format | Block | Group scales | `m` |
| --- | --- | --- | --- |
| Q2_K | 256 values, 84 bytes | one per 16 values (4 bit) | `-dmin * min` per 16 values |
| Q3_K | 256 values, 110 bytes | one per 16 values (6 bit, signed) | 0 |
| IQ4_NL | 32 values, 18 bytes | one (f16) | 0 |
| IQ4_XS | 256 values, 136 bytes | one per 32 values (6 bit, signed) | 0 |
| MXFP4 | 32 values, 17 bytes | one (E8M0, halved; values at twice their magnitude) | 0 |

The 4-bit codebook formats decode with `tbl` (`vqtbl1q_s8`): 16 bytes of nibbles go through the 16-entry
table in one instruction per half group, giving signed weights that `sdot` consumes directly. (On
x86 `maddubs` cannot take signed weights, which is why those formats stay on the streaming kernel
there.) The activations are the same 8-bit records the other formats use: Q8_0-style (fp16 scale per
32 values) for IQ4_NL and MXFP4, Q8_K-style (one f32 scale per 256 values) for the 256-value
formats, so the K formats and IQ4_XS share one packing in mixed groups and a Q2_K/Q3_K/Q4_K/Q6_K
Q/K/V group runs in one work queue.

### Decode and prompt

- **Decode (one token):** four weight rows per claim, each group decoded into registers and
  consumed by `sdot` (two for formats with half-group scales), the dot product converted and
  accumulated with `fmla`. Q2_K adds its minimum term from the activation half sums.
- **Prompt (two or more tokens):** a worker claims a four-row panel, unpacks it once into `Group`
  records in its own scratch (no second copy of the weights, no load-time step, like the existing
  prompt repacking), and runs every token pair over it, with `smmla` when FEAT_I8MM is present:
  row pairs against token pairs, four `smmla` per 32-value group (the two 16-value halves of Q2_K
  and Q3_K are separate `smmla` pairs because they carry separate scales). Short last panels are
  zero padded; an odd last token repeats the previous one and its result is dropped.
  `--no-repack` does not apply: the unpack is what the kernel runs on.

### Float formats

F32, F16 and BF16 weights stay packed. The input is rounded once to the weight type (as the AVX2
kernels do), eight to sixteen independent accumulators hide the fused multiply-add latency, and the
panel is four weight rows by one, two or four tokens. BF16 with FEAT_BF16 uses `bfdot` on an input
rounded to bf16 bits (eight products per instruction; the instruction's own rounding differs from a
fused f32 sum in the last bits, and it flushes bf16 denormals in the non-extended mode, irrelevant
for weights). `FLLM_CPU_ISA=neon-dotprod` switches `bfdot` off. Weight loads go through byte vectors,
so no alignment is required of the weight bytes.

### DeltaNet

`cpu_delta::step` dispatches to a NEON version for column blocks that are a multiple of four. Every
column is a lane, the operations and their order are those of the scalar loop (separate multiplies
and adds, no fused multiply-add), so the result is bit-identical to the scalar step, as the AVX2
version is. The convolution and normalisation loops are the portable ones (the compiler vectorises
them for NEON); SiLU keeps `exp` from libm to stay identical to the token-at-a-time reference.

### SVE

The `std::arch` SVE intrinsics are not stable, so the one SVE kernel (Q8_0 decode matvec, four rows
per call) is inline assembly with `.arch_extension sve`; the crate does not need the unstable `sve`
target feature. One 32-byte block fills one 256-bit vector (predicated `ld1b`), so the kernel is
restricted to vectors of at least 256 bits (Graviton 3, Neoverse V1, A64FX); 128-bit SVE cores
(Graviton 4, Neoverse N2/V2, Grace) use the NEON kernels. No speed-up is claimed. A decode matvec is
limited by memory bandwidth and the NEON kernel already issues only two `sdot`s per block. The tier is
opt-in (`FLLM_ARM_SVE=1`) until `selftest-cpu` has passed on SVE hardware and a benchmark shows a
gain; `cpu_arm::sve::DEFAULT_ON` is the one switch to flip then. `selftest-cpu` enables the tier by
itself to check it.

## What to run on an ARM machine

```bash
cargo test --release -p fllm-engine cpu_arm        # decoders vs scalar, kernels vs reference
cargo test --release -p fllm-engine cpu_delta      # NEON DeltaNet step vs scalar
cargo test --release -p fllm-engine cpu_float      # (x86 float tests are skipped; cpu_arm::float has its own)
fllm-rs selftest-cpu                               # every supported tier, all kernels
fllm-rs selftest-cpu --tier neon-i8mm
FLLM_ARM_SVE=1 fllm-rs selftest-cpu --tier neon-sve
```

`selftest-cpu` covers, per tier, the formats above at odd row, column and batch sizes (13 and 16
rows, batches 1..64), the mixed K group with Q2_K, Q3_K and IQ4_XS, the float formats at three
shapes, the DeltaNet recurrence against the scalar tier (bit for bit), and everything it covered
before. A `FAIL` line names the kernel and tier; `FLLM_CPU_ISA=<lower tier>` steps around it.

## Not done

- `ummla` / `usmmla` / `usdot`: the kernels convert weights to signed 8-bit values (cheap with the
  offsets known), which `smmla` and `sdot` consume; the unsigned-by-signed forms would only save
  that conversion.
- SVE beyond the Q8_0 matvec (SVE `smmla` prompt tiles, K formats, float): nothing to gain on the
  128-bit cores, unmeasurable without hardware elsewhere.
- FEAT_FHM (`fmlal`) for F16 weights and FEAT_FP16 arithmetic: F16 weights are converted with
  `fcvtl` and summed in f32.
- The fused MoE expert pass (`cpu_quantized::linear_experts`) is x86-only; ARM runs the experts one
  by one.
- Q4_K / Q5_K / Q6_K / Q8_0 / Q4_x / Q5_x keep their existing kernels unchanged.
- Windows on ARM (`aarch64-pc-windows-msvc`) compiles the same code; neither it nor Linux/Android
  on ARM has been run.
