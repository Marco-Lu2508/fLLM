# x86 extensions: AVX-512, AMX and the CPU-family policy

Package 14 of the multi-backend round. Everything described here is **test level L0**: it compiles
(`cargo check`, `clippy`, cross targets) and parts of it are unit-tested on an AVX2 machine, but no
AVX-512 or AMX instruction has been executed and nothing has been measured. The performance
statements below are design reasons, not results. See "What was run" and "Hardware test plan".

## Enabling it

All new kernels sit behind the Cargo feature `avx512` (x86-64 only, off by default):

```bash
cargo build --release -p fllm-engine --features avx512
```

Without the feature nothing in this document is compiled and the CPU dispatch is exactly what it
was before (including the existing AVX-512 BW/VNNI tiers described in `cpu-kernels.md`). With the
feature, dispatch is still decided at run time from the CPU; the binary runs on any x86-64 CPU.

## Tiers

The tier ladder of `cpu-kernels.md` gains two rungs above `avx512-vnni`:

| Tier | Level | Required | Uses |
| --- | --- | --- | --- |
| `avx512-bf16` | 5 | `avx512-vnni` + AVX512-BF16 | BF16 projections with `vdpbf16ps` |
| `amx` | 6 | `avx512-bf16` + AMX-TILE + AMX-INT8, OS tile state, Linux permission | tile GEMMs for prompts |

`FLLM_CPU_ISA=<tier>` caps the ladder as before: `FLLM_CPU_ISA=avx512-vnni` turns both new rungs off,
`avx512-bf16` turns AMX off. `fllm-rs selftest-cpu` runs every tier the CPU supports, so a
Sapphire Rapids machine checks `amx` and `avx512-bf16` in addition to the older ones.

Rust cannot report AMX yet (`x86_amx_intrinsics` is unstable), so the CPU bits come from CPUID
directly (leaf 7 EDX bits 24/25/22) plus XCR0 bits 17/18 for the OS tile state. On Linux the first
use asks the kernel for the tile data (`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`);
without that call the first tile instruction raises SIGILL. The request is made once, when the
`amx` tier is first probed, and is process-wide. **AMX is Linux-only here**; on Windows and macOS
the tier is never supported.

## CPU identification and policy

`cpu_avx512::ident` classifies the CPU from CPUID family/model (and stepping for Skylake-SP versus
Cascade Lake) and attaches a policy. The `avx512` feature bit separates Zen 4 from Zen 3 (both
family 0x19).

| Class | Models | `wide_int` | `wide_float` | `bf16_dot` |
| --- | --- | --- | --- | --- |
| Zen 4 | AMD family 0x19 with AVX-512 | yes | **no** | yes |
| Zen 5 and later | AMD family 0x1A+ with AVX-512 | yes | yes | yes |
| Intel client with AVX-512 | Ice/Tiger/Rocket/Cannon Lake client | yes | **no** | yes |
| Skylake-SP, Cascade/Cooper Lake | Intel 0x55 | yes | **no** | yes |
| Ice Lake-SP/D, AMX servers | 0x6A, 0x6C, 0x8F, 0xCF, 0xAD, 0xAE, Diamond Rapids (family 19 model 1) | yes | yes | yes |
| other | | yes | yes | yes |

- `wide_int`: the 512-bit integer dot kernels are preferred to the 256-bit ones (this is the
  existing behaviour of the AVX-512 tiers; the policy only provides the override).
- `wide_float`: the 512-bit F32/F16/BF16 FMA kernels and the 512-bit DeltaNet step replace the AVX2
  ones. Off where a 512-bit operation is issued as two 256-bit halves (Zen 4, the Intel client parts
  with one 512-bit unit) or where heavy 512-bit code lowers the clock (Skylake-SP, Cascade Lake):
  there the wider vector saves nothing in throughput. These are hypotheses taken from the
  micro-architecture descriptions; nothing was measured.
- `bf16_dot`: `vdpbf16ps` does twice the multiply-adds per instruction, so it stays on for every
  CPU that has the tier.

Overrides (all read once): `FLLM_X86_WIDE_INT`, `FLLM_X86_WIDE_FLOAT`, `FLLM_X86_BF16` take
`0|off|false|no` or anything else for "on"; `FLLM_AMX_MIN_BATCH=<n>` sets the smallest prompt that
goes to AMX (default 8); `FLLM_VNNI_GEMM=0|all` (below). `selftest-cpu` prints the detected class,
the extra features and the policy in one line.

## Kernels and dispatch (with `--features avx512`)

Added to the table of `cpu-kernels.md`; "(new)" marks what this package adds.

| Kernel | Choice, left to right |
| --- | --- |
| Q8_0, Q4_0..Q5_1, IQ4_NL, MXFP4, Q2_K..Q6_K, IQ4_XS, prompt of `FLLM_AMX_MIN_BATCH`+ tokens | **AMX tile GEMM (new)**; then the older paths |
| Q8_0 prompt (`q8_linear*`) | AMX (new); AVX-512 4x2 tile; AVX-VNNI/AVX2 |
| Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_XS, IQ4_NL, MXFP4 prompt (4+ tokens), AVX-512 VNNI CPU | AMX if active; **16-row VNNI GEMM (new)**; streaming 512-bit panels |
| Q2_K, Q3_K, IQ4_XS, decode and small prompts, AVX-512 | **512-bit panels (new, 8 rows for decode)**; before: 256-bit EVEX |
| IQ4_NL, MXFP4, decode and small prompts | **AVX-512 VNNI / BW 512-bit (new)**; **AVX-VNNI-INT8 or AVX-VNNI 256-bit (new)**; AVX2 codebook |
| BF16 projections | AMX-BF16 for prompts with `cols % 32 == 0` (new); **`vdpbf16ps` (new)**; 512-bit FMA (new, if `wide_float`); AVX2 |
| F32/F16 projections | 512-bit FMA (new, if `wide_float`); AVX2 |
| DeltaNet state update | 512-bit (new, if `wide_float`, widths 16/32/48/64); AVX2 |

Note on the older tiers: the AVX2-level paths are not switched off when AVX-512 is present. The
existing code already picks the 512-bit kernel in front of AVX-VNNI/AVX2 for Q8_0, the classic
formats, Q4_K..Q6_K and attention. The one inversion it had, Q2_K/Q3_K/IQ4_XS taking the VEX
256-bit AVX-VNNI kernel on CPUs that have both AVX-VNNI and AVX-512 (Sapphire Rapids, Zen 5), is
fixed: the 512-bit kernel is tried first.

### IQ4_NL and MXFP4 (`codebook`)

Two consecutive 32-value blocks of a row share one 512-bit vector in the 128-bit lane order
`[a low nibbles, b low nibbles, a high nibbles, b high nibbles]`, looked up in a 16-entry table with
`vpshufb`; the activation records are arranged the same way with one `vshufi64x2`. The VNNI variants
store the table as `value + 128` and let `vpdpbusd` start from the records' `init` word
(`-128 * sum(x)` spread over the 16-value halves, the convention of the Q8_0 tiles), so the dot
never saturates. The AVX-512 BW variant (no VNNI) keeps the signed table and uses the
sign-transfer dot of the Q8_0 kernels. AVX-VNNI-INT8 (`vpdpbssd`, Arrow Lake, Lunar Lake) needs
neither. All variants produce the same integer dots as the AVX2 kernel.

### Q2_K, Q3_K, IQ4_XS (`kextra`)

After `cpu_kquant::kernel512`: two 32-value groups per 64-byte vector, `packs_epi32` plus one
`madd` with the 16-value scales (Q2_K, Q3_K) or the sign-transfer `maddubs` with the group scale in
the `madd` (IQ4_XS). Q2_K's minimum meets the 16-value activation sums in one `madd`; Q3_K's offset
is folded into the dot's initial value.

### AMX tile GEMM (`amx`, `amx_decode`, `amx_gemm`)

Prompts of `amx_min_batch` tokens or more (default 8; an unmeasured guess: below it the padding of
the 16-token tile and the panel decode outweigh the tile throughput).

- A worker claims a 16-row panel and decodes it once (`amx_decode`) into plain 8-bit values plus a
  float scale per group of 16 or 32 values and, for Q2_K/Q4_K/Q5_K/Q4_1/Q5_1, a float minimum:
  `weight = scale * q - min`. Formats with a stored offset (Q4_0 -8, Q5_0 -16, Q3_K -4, Q6_K -32)
  are decoded signed, so they need no correction; the codebooks are signed bytes. The decoders are
  ordinary Rust loops (and `pshufb` for the codebooks) that need only AVX2 + F16C.
- The activations of the call are packed once (`AmxActs`) into 16-token tiles in the VNNI
  interleave (`[group / 4 rows][16 tokens][4 bytes]`), together with a scale and a sum per group
  and token. Weights are tile A (16 rows x one group), activations tile B, the result C is a 16 x 16
  `i32` tile per group: `tdpbssd` for signed, `tdpbusd` for unsigned weights.
- Per group the epilogue converts the tile to float and does `acc += C * scale[row] * dx[token]`
  and, with a minimum, `acc -= min[row] * sum[token]`. Two groups are in flight in two tile sets
  (tiles 0-2 and 3-5) to overlap the dot latency. 2 x 2 blocking is the obvious next step; not done.
- BF16 projections need no decode: the bf16 weights are tile A directly, the activations tile B
  (`[16 pair rows][16 tokens][2 x bf16]`), and `tdpbf16ps` accumulates over the whole row in four
  C tiles (2 x 2 blocking, 32 weight rows x 32 tokens per pass).
- The tile instructions are issued by inline assembly (`ldtilecfg`, `tileloadd`, `tilestored`,
  `tilezero`, `tdpbssd`, `tdpbusd`, `tdpbf16ps`, `tilerelease`); the generated encodings were
  checked with `objdump` on the build machine. Each panel configures and releases the tiles.
- Unit tests run the same drivers against `EmuTiles`, a scalar model of the instruction semantics
  written from the instruction descriptions, and against the portable epilogue. A fault in that
  model would hide a fault in the drivers (and vice versa); only hardware can close this.

### 16-row VNNI GEMM (`vnni_gemm`)

For AVX-512 VNNI CPUs without AMX (and as an alternative on those with it, if the AMX tier is
capped). The decoded panel is transposed so that one 64-byte vector holds four columns of all 16
rows; one `vpdpbusd` against a broadcast activation dword then advances 16 dot products, with no
horizontal sum and no pack/madd step. Signed formats are stored as `q + 128` and the offset becomes
a minimum of `128 * scale` against the group's activation sum. By default it takes the formats whose
existing prompt kernels decode each block again per token pair (K formats, IQ4_XS, IQ4_NL, MXFP4);
Q8_0, Q4_0, Q5_0 keep their tuned 4x2 tiles. `FLLM_VNNI_GEMM=all` takes every decodable format,
`0` turns the path off. This is the arrangement of `cpu_kquant::r8_*` widened to 16 rows. The kernel
body is one macro instantiated for 512-bit and (in the tests) for 256-bit AVX-VNNI vectors.

### BF16 with `vdpbf16ps` and 512-bit float (`float`)

`cpu_float` rounds the input to the weight type first, so a BF16 projection already multiplies
bf16 by bf16-exact values; `vdpbf16ps` reproduces the products exactly in f32 and adds the pairs with
one rounding each. It flushes denormal inputs and outputs and ignores MXCSR, so results can differ in
the last bits from the FMA path; `FLLM_CPU_ISA=avx512-vnni` or `FLLM_X86_BF16=0` falls back. The
prompt kernel transposes 16 BF16 rows into column pairs (`panel[pair][row]`, reusing the 8x8 dword
transposes of the AVX2 panel); the decode kernel uses 32-column steps with masked loads, so odd
widths need no scalar tail. The F32/F16 kernels are the 16-row panel and the 4 x 2 tile of the AVX2
code widened to 512 bits, 14 tokens per pass.

## What was run

On the build machine (Core i9-13900H: AVX2, AVX-VNNI, no AVX-512, no AMX), `cargo test --release -p
fllm-engine --features avx512` runs 247 tests (207 pre-existing; the default build still runs 207).
`fllm-rs selftest-cpu` of a build with the feature passes 364 checks on its tiers (scalar, avx2,
avx-vnni), including the extended float checks, also under every policy override of
`scripts/check_x86_ext.sh`. Of the 40 new tests:

- executed in full: CPUID classification tables, the policy table, panel decoders for 13 formats
  against `compute::dequantize`, the activation-tile layout, the VNNI interleave, the AMX drivers
  (Q8_0, Q4_0..Q5_1, IQ4_NL, MXFP4, Q2_K..Q6_K, IQ4_XS, mixed K groups, BF16) against the f64 reference with
  emulated tiles, the AVX-VNNI-256 codebook kernels, the 256-bit instantiation of the VNNI-GEMM
  kernel body for all formats, the f32 and bf16-pair panel decoders of the float kernels;
- executed as scalar models of the 512-bit data flow (`model.rs`: shuffles, `packs`, `madd`,
  `dpbusd`, blends restated from their descriptions; the kernel's sequence of operations is replayed
  with them and compared with the reference): the IQ4_NL/MXFP4 kernels and the Q2_K, Q3_K, IQ4_XS
  kernels. This checks the lane order, shift counts, bias distribution and scale mapping, not the
  intrinsics themselves;
- compiled but skipped for lack of the CPU feature: the AVX-512 codebook, Q2_K/Q3_K/IQ4_XS, float,
  BF16 and DeltaNet kernels, the AVX-512 instantiation of the VNNI GEMM, AVX-VNNI-INT8.

Mutation checks (a deliberately wrong offset, scale, sign or index in the source; the suite must
fail) caught every mutation of the executed code; a mutation of the AVX-512 epilogue was, as
expected, not caught because that code is not executed.
## Hardware test plan

On an AVX-512 VNNI machine (Zen 4/5, Ice Lake or later), AMX if available, Linux,
`scripts/check_x86_ext.sh` runs everything below (unit tests, every supported tier, the policy
overrides) and exits non-zero on a failure. By hand:

```bash
cargo test --release -p fllm-engine --features avx512 cpu_avx512      # the new unit tests
cargo test --release -p fllm-engine --features avx512                  # all of them
cargo run --release --features avx512 -- selftest-cpu                  # every supported tier
cargo run --release --features avx512 -- selftest-cpu --tier amx       # AMX only (Sapphire Rapids+)
FLLM_CPU_ISA=avx512-vnni  ... selftest-cpu                             # compare without AMX/bf16
```

Expected: all checks `ok`; a failing line names the format, tier and shape. Failures are most likely
in, in this order: the AMX tile configuration (shapes rejected as `#UD`/SIGILL), the 512-bit lane
orders of `codebook`/`kextra` (wrong values, not crashes), the `vdpbf16ps` pair order, the permission
request in a container or VM that filters `arch_prctl`. For A/B timings use
`FLLM_CPU_ISA=<tier>`, `FLLM_X86_WIDE_FLOAT`, `FLLM_VNNI_GEMM`, `FLLM_AMX_MIN_BATCH` and the existing
`--no-repack`. L1 means `selftest-cpu` passes, L2 means the greedy text of a real model equals the
`avx2` run, L3 a comparison against llama.cpp on the same machine.

## Not implemented

- **AMX-FP16** (Granite Rapids-D): not used; F16 weights go through the 512-bit FMA kernel.
- **AMX on Windows/macOS**: the permission handshake differs; the tier reports "not supported".
- **`vpermb`/AVX512-VBMI**: detected and printed only; the 16-entry codebooks fit `vpshufb`.
- **2 x 2 tile blocking for the integer kernels** and a 16-token-tile `tdpbssd` with K = 64 where
  consecutive groups share a scale: both would raise tile utilisation; neither is measured or built.
- **AVX-VNNI-INT8 for Q8_0 and the classic formats**: only the codebook kernel uses `vpdpbssd`.
- Zen 4 specific tuning beyond the float/integer width choice (for example 256-bit EVEX kernels with
  32 vector registers) is not done.
