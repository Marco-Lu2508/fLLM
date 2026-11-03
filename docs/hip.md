# AMD GPUs: ROCm / HIP

fLLM's HIP backend runs the CUDA kernel sources compiled for AMD GPUs (`cuda/*.cu` and `hip/*.cu`, built with `hipcc`, behind the same C API as the
NVIDIA backend). This page says which AMD GPUs the build covers, what fLLM runs on each family, how a GPU that is not in any list is handled, and how
to report a result. Test levels in general: [backends.md](backends.md). Build details and the Windows SDK: [changes/hip.md](changes/hip.md),
[windows.md](windows.md).

**Short version.** The default build contains a code object for every gfx target the ROCm compiler knows (Vega, CDNA, RDNA 1 to 4) *and* the generic
targets of newer clang (`gfx9-generic`, `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic`, `gfx11-7-generic`, `gfx12-generic`, ...), so a GPU
that is not named anywhere still finds a loadable code object. Nothing is skipped because its name is not in a list. A GPU is only used after a small
check on the device (a matrix product, the integer dot product and one matrix-core step are compared with the host); a GPU that fails it is reported
as *present but not usable* with the reason, and the model runs on the next device. **Only one AMD GPU, a Radeon 840M (`gfx1153`), has ever run fLLM's
HIP code.** Every other family below is compiled in CI and expected to work, not proven.

## What the build contains

| | |
| --- | --- |
| Default targets (Linux) | `gfx900 gfx906 gfx908 gfx90a gfx942 gfx950`, `gfx1010 gfx1011 gfx1012`, `gfx1030 ... gfx1036`, `gfx1100 ... gfx1103`, `gfx1150 ... gfx1153`, `gfx1170 gfx1171 gfx1172`, `gfx1200 gfx1201`, `gfx1250 gfx1251`, plus the generic targets `gfx9-generic`, `gfx9-4-generic`, `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic`, `gfx11-7-generic`, `gfx12-generic`, `gfx12-5-generic` (the list is `rust/engine/hip/default_archs.txt`) |
| Default targets (Windows) | the same without the Instinct targets (`gfx908 gfx90a gfx942 gfx950 gfx9-4-generic`) and without `gfx125x`; targets whose `oclc_isa_version_*.bc` the HIP SDK lacks are dropped with a build warning |
| Targets the compiler does not know | dropped one by one (a probe compiles an empty kernel for each); an explicit `FLLM_HIP_ARCHS` is never filtered and fails loudly |
| Compression | the device code is compressed in the binary (`--offload-compress`, zstd); the HIP runtime of ROCm 6.2 and newer unpacks it. `FLLM_HIP_COMPRESS=0` builds uncompressed code for an older runtime |
| Size | measured with the ROCm 7.16 clang, release build, 4 compile jobs: the old list (14 targets, uncompressed) gave a 155.5 MB `fllm-rs` with 103 MB of device code (22 min build); the new default list (38 targets, compressed) gives a 63.2 MB `fllm-rs` with 10.8 MB of device code (17.6 min) |
| Quick builds | `FLLM_HIP_ARCHS=gfx1100` (one target), `FLLM_HIP_ARCHS=native` (the GPUs of this machine, from `rocm_agent_enumerator` / `amdgpu-arch`); `FLLM_HIP_JOBS=N` sets how many sources compile at once (default: cargo's job count, at most 8) |

Not in the list: `gfx940` / `gfx941` (early CDNA 3 steppings the current compiler no longer builds; `HSA_OVERRIDE_GFX_VERSION=9.4.2` runs the `gfx942`
code on them), `gfx13xx` (the compiler's support is incomplete; such a chip gets the conservative class and, if its runtime accepts it, the plain
kernels), and the pre-Vega GCN chips (Polaris and older, gfx8xx: not built and not tried).

## Which GPU runs what

Three kernel sets exist. The choice is made on the device from the code object that was really loaded (see below), not from the name of the GPU.

* **Matrix cores**: a prompt GEMM that decodes the quantized weights into LDS and multiplies 16x16x16 FP16 tiles with FP32 accumulation (`hip/hip_gemm.cu`);
  MFMA on CDNA, WMMA on RDNA 3 / 3.5 and RDNA 4, plus the skinny 1 to 4 token kernels (`hip_skinny.cu`, `hip_q4z.cu`) where the chip has `v_dot2`.
* **dp4a**: the integer-dot-product matrix-vector kernels (Q4_0 ... Q8_0, K-quants, IQ4, MXFP4) with `v_dot4_i32_i8` (`gfx906`, `gfx908`, `gfx90a`, CDNA 3/4,
  `gfx1011`, `gfx1012`, RDNA 2) or `v_dot4_i32_iu8` (RDNA 3 and newer). Where the chip has neither (`gfx900` family, `gfx1010`, `gfx125x`) the same kernels run
  a byte loop with the same result.
* **Plain**: FP32 arithmetic only (FP32 GEMM, plain and split-KV attention, element-wise kernels, the Q4Z GEMV for GPTQ / AWQ 4-bit weights). No MFMA, WMMA, `v_dot`
  or DPP-row-only instruction; this is the fallback that every family has, and the only set on the families marked "plain".

| Family | gfx names | Examples | Wave | Code object | Matrix cores | Integer dot | Used by fLLM | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vega 10 / GCN 5 | `gfx900` `gfx902` `gfx904` `gfx909` `gfx90c` | Vega 56 / 64, Ryzen 2000 to 5000 APU graphics (`gfx90c`) | 64 | `gfx900`; `gfx9-generic` for the others | none | none | plain, byte-loop dp4a | compiled in CI, untested |
| Vega 20 | `gfx906` | Radeon VII, Instinct MI50 / MI60 | 64 | `gfx906` | none | `v_dot4_i32_i8` | plain, dp4a | compiled in CI, untested |
| CDNA 1 | `gfx908` | Instinct MI100 | 64 | `gfx908` | MFMA | `v_dot4_i32_i8` | matrix cores (MFMA), dp4a | compiled in CI, untested |
| CDNA 2 | `gfx90a` | Instinct MI210 / MI250 | 64 | `gfx90a` | MFMA | `v_dot4_i32_i8` | matrix cores, skinny kernels (`v_dot2c`), dp4a | compiled in CI, untested |
| CDNA 3 | `gfx942` (`gfx940` / `gfx941` with the override) | Instinct MI300 | 64 | `gfx942`; `gfx9-4-generic` | MFMA | `v_dot4_i32_i8` | as CDNA 2 | compiled in CI, untested |
| CDNA 4 | `gfx950` | Instinct MI350 | 64 | `gfx950`; `gfx9-4-generic` | MFMA | `v_dot4_i32_i8` | as CDNA 2, 160 KiB LDS in the skinny kernel (exact code object only) | compiled in CI, untested |
| RDNA 1 | `gfx1010` `gfx1013` | RX 5600 / 5700 | 32 | `gfx1010`; `gfx10-1-generic` | none | none | plain, byte-loop dp4a | compiled in CI, untested |
| RDNA 1 | `gfx1011` `gfx1012` | Radeon Pro V520, RX 5500 | 32 | `gfx1011` / `gfx1012`; `gfx10-1-generic` | none | `v_dot4_i32_i8` | plain, dp4a | compiled in CI, untested |
| RDNA 2 | `gfx1030` `gfx1031` `gfx1032` `gfx1033` `gfx1034` `gfx1035` `gfx1036` | RX 6600 to 6900, Steam Deck (`gfx1033`), Ryzen 6000 / 7000 integrated graphics (`gfx1035` / `gfx1036`) | 32 | exact; `gfx10-3-generic` | none | `v_dot4_i32_i8` | plain, dp4a | compiled in CI, untested |
| RDNA 3 | `gfx1100` `gfx1101` `gfx1102` `gfx1103` | RX 7600 to 7900, Ryzen 7040 (Radeon 780M / 760M) | 32 | exact; `gfx11-generic` | WMMA (16-element fragments) | `v_dot4_i32_iu8` | matrix cores (WMMA), skinny kernels (`v_dot2`), dp4a | compiled in CI, untested |
| RDNA 3.5 | `gfx1150` `gfx1151` `gfx1152` `gfx1154` | Ryzen AI 300 / Max series (Radeon 880M / 890M / 8060S) | 32 | exact (`gfx1154`: `gfx11-generic`) | WMMA | `v_dot4_i32_iu8` | as RDNA 3 (`gfx1151` has a tuned tile table) | compiled in CI, untested |
| RDNA 3.5 | `gfx1153` | Radeon 840M | 32 | `gfx1153` | WMMA | `v_dot4_i32_iu8` | as RDNA 3 | **verified on hardware** (below) |
| RDNA 3.5+ | `gfx1170` `gfx1171` `gfx1172` | newer RDNA APUs | 32 | exact; `gfx11-7-generic` | WMMA in the gfx12 encoding (hardware has it, **not used**) | `v_dot4_i32_iu8` | plain, dp4a, skinny kernels | compiled in CI, untested |
| RDNA 4 | `gfx1200` `gfx1201` | RX 9060 / 9070 | 32 | exact; `gfx12-generic` | WMMA (8-element fragments) | `v_dot4_i32_iu8` | matrix cores, skinny kernels, dp4a | compiled in CI, untested |
| gfx12.5 | `gfx1250` `gfx1251` | | 32 | exact; `gfx12-5-generic` | a different matrix unit (not used) | `v_dot4_i32_iu8`, no `v_dot2` | plain, dp4a | compiled in CI, untested |
| Anything else | unknown names | future chips | by generation | none unless a generic target of its family is in the binary | | | plain (the "conservative class") | not applicable |

"Compiled in CI" means: every HIP source compiles for that target with the ROCm 7.16 (ROCm 10.1 packages) clang, one `PASS` line per target in the `kernels` workflow
([ci.md](ci.md)); the instruction facts of this table (matrix unit, dot product, wave size) are compared with what the compiler defines for each target by
`scripts/hip_arch_check.sh`. It does **not** mean a kernel produced a correct number on such a chip. The generic path is expected to work because it is the same source
compiled for the family's common instruction subset, but no generic code object has ever been loaded on a GPU by fLLM. `gfx117x` is deliberately on the plain
kernels: its matrix unit uses the layouts of RDNA 4, which were never run, and one wrong tile layout would give wrong text instead of a slow answer.

LDS: 64 KiB per workgroup on every target above, 160 KiB on `gfx950`. FP16 is native on all of them, BF16 inputs are converted. Atomics: the kernels use `atomicAdd` and `atomicMax` on 32-bit values only (hardware instructions or the compare-and-swap loop
the compiler emits where the chip has no FP32 atomic add).

## How a GPU gets its code

1. **Before the HIP runtime starts** (Linux) fLLM reads the GPUs the kernel driver lists (`/sys/class/kfd`) and checks the list this binary was built for
   (`FLLM_HIP_BUILT_ARCHS`, printed by the build as `HIP code objects for: ...`):
   * the exact `gfx` target is in the binary: nothing to do;
   * else the generic target of its family is: the HIP runtime loads the generic code object by itself (a generic code object is code object version 6 and runs on
     every processor its target lists; it can be slower than the exact one, [LLVM's AMDGPU usage notes](https://llvm.org/docs/AMDGPUUsage.html#generic-processors)).
     One log line says so. That the runtime prefers the exact code object when both are in the binary, and that an older ROCm runtime loads generic ones
     at all, is how the loader is documented to work but was not tried by fLLM; the device check below decides in every case, and an older runtime that cannot
     load a generic code object reports the GPU as not usable (then `FLLM_HIP_ARCH_OVERRIDE` / `HSA_OVERRIDE_GFX_VERSION` or an exact target applies);
   * else (a build with a short `FLLM_HIP_ARCHS`, or a chip newer than the build): the nearest chip of the same instruction-set generation that *is* in the binary is
     used by setting `HSA_OVERRIDE_GFX_VERSION` for this process (log line: `HSA_OVERRIDE_GFX_VERSION=10.3.0 loads the code of gfx1030`). A chip with no stand-in
     (a new CDNA, RDNA 4, a matrix-core family whose neighbour has other instructions) gets a line that names the `FLLM_HIP_ARCHS` to rebuild with.
   A value of `HSA_OVERRIDE_GFX_VERSION` that you set yourself is never replaced. fLLM sets the variable inside its own process, which works only if the ROCm stack has
   not started yet (whether the HIP runtime starts before `main` on a given ROCm version was not checked); if the GPU then fails the device check, export the variable in
   the shell before starting `fllm-rs`, as with llama.cpp.
2. **After the runtime has started**, every AMD GPU gets a check on the device (`fllm_hip_selfcheck`, `hip/hip_device.cu`): which code object was loaded, a 16x16x16
   FP32 product, the integer dot product on 16 test vectors and, if the code has matrix cores, one 16x16x16 matrix-core step, each compared with the host.
   * all pass: the GPU is listed; the log line says which code object it uses (`its own code object` / `the generic gfx10-3-generic code object` / `another chip's code object`),
     the wave size and the kernel set (`kernels: WMMA`);
   * the matrix-core step gives a wrong result (or the loaded code is for another matrix unit than the chip has): the GPU still works with the **plain kernels**, one line says so;
   * nothing runs (no code object fits, an instruction the chip lacks, wrong data): the GPU is **present but not usable**: it is left out of the device list, `fllm-rs devices`
     prints it with the reason and what to try, the Studio shows it as unavailable, `fllm-rs doctor --gpu` prints `FALLS BACK: present but not usable: ...`, and the placement
     ledger carries a notice (`/api/studio/status`, `device_fallbacks.notices`). `--device auto` goes on to Vulkan or the CPU; an explicit `--device hip:N` for it is an error that names
     the reason. The other AMD GPUs keep their ordinals.
3. **Windows** has no `HSA_OVERRIDE_GFX_VERSION` (the Windows runtime does not read it, untested); there steps 1's first two cases and step 2 apply, and a chip without
   a code object is reported as not usable with the `FLLM_HIP_ARCHS` to rebuild with.

### Overriding by hand

* `FLLM_HIP_ARCH_OVERRIDE=gfxNNNN`: treat every AMD GPU as that chip. The host selects its kernels for that chip, and on Linux fLLM sets the matching
  `HSA_OVERRIDE_GFX_VERSION` (`gfx1030` is `10.3.0`) before the runtime starts unless you set one. The code object that gets loaded still has to exist in the binary and the
  self-check still decides; a matrix-core set that does not fit the chip is switched off (plain kernels).
* `HSA_OVERRIDE_GFX_VERSION=x.y.z` (Linux, read by the ROCm user-mode stack, not by fLLM): makes the runtime treat the GPU as another chip. llama.cpp and Ollama document the same
  variable; with the default build you normally do not need it, because the generic code objects are in the binary. If a GPU is reported as not usable, these are the values
  people use, nearest chip first (the value must belong to a target that is in your binary):

  | GPU | gfx | `HSA_OVERRIDE_GFX_VERSION` | runs the code of |
  | --- | --- | --- | --- |
  | RX 6700 / 6700 XT, RX 6600, RX 6500 / 6400, Ryzen 6000 / 7000 integrated graphics, Steam Deck | `gfx1031` to `gfx1036` | `10.3.0` | `gfx1030` |
  | RX 7600, Radeon 780M / 760M, RX 7700 / 7800 | `gfx1102` `gfx1103` `gfx1101` | `11.0.0` | `gfx1100` |
  | RDNA 3.5 APUs (880M, 890M, 8060S, 840M) | `gfx1150` to `gfx1154` | `11.5.1` | `gfx1151` |
  | RX 5600 / 5700 (RDNA 1), if its own code does not load | `gfx1010` `gfx1013` | `10.1.0` | `gfx1010` |
  | Vega APUs (Ryzen 2000 to 5000 graphics), Vega 12 | `gfx90c` `gfx902` `gfx904` `gfx909` | `9.0.0` | `gfx900` |
  | early CDNA 3 steppings | `gfx940` `gfx941` | `9.4.2` | `gfx942` |

  This is unsupported by AMD and by fLLM: a stand-in whose instructions the chip lacks fails the first kernel, which the self-check reports as not usable (no wrong text). On
  a machine with two different AMD GPUs the variable applies to both; fLLM sets it only for a GPU that has no code object at all.

## Environment variables

| Variable | Where | Meaning |
| --- | --- | --- |
| `FLLM_HIP=1` | build | compile the kernel library with `hipcc` (`--features hip`) |
| `FLLM_HIP_ARCHS` | build | `;`-separated targets, `native`, or empty for the default list (above) |
| `FLLM_HIP_COMPRESS=0` | build | uncompressed device code (for a HIP runtime older than 6.2) |
| `FLLM_HIP_JOBS` | build | sources compiled at once |
| `ROCM_PATH` / `HIP_PATH` | build, run | the ROCm install (Windows: the HIP SDK) |
| `FLLM_HIP_ARCH_OVERRIDE` | run | treat every AMD GPU as this gfx chip (above) |
| `HSA_OVERRIDE_GFX_VERSION` | run | ROCm runtime variable, see above |
| `FLLM_HIP_NO_MMA`, `FLLM_HIP_NO_SKINNY`, `FLLM_HIP_NO_Q4Z`, `FLLM_HIP_NO_Q4Z_SKINNY`, `FLLM_HIP_NO_X16` | run | switch single kernel families off (`1`); `FLLM_HIP_NO_MMA=1` forces the plain kernels on any chip, a good first step when a result looks wrong |
| `FLLM_STRICT_DEVICE=1` | run | an operation that would run on the CPU is an error |

## What is verified on hardware

| Device | What ran | Where |
| --- | --- | --- |
| Radeon 840M, `gfx1153`, ROCm 7.1 (Ubuntu packages), Linux | `fllm-rs selftest-gpu --device hip:0` of 2026-10-31: 0 failed in 976 (exact) and 1365 (fast) checks; dense models write the CPU's text; benchmark rows in [benchmarks.md](benchmarks.md) | README, [benchmarks.md](benchmarks.md) |
| everything else | nothing | |

The AMD handheld with a Radeon 780M (`gfx1103`) in [windows.md](windows.md) and [benchmarks.md](benchmarks.md) ran the **Vulkan** backend, not HIP. No Vega, CDNA, RDNA 1, RDNA 2, RDNA 4 or
generic code object has run. The self-check, the generic-target selection and the `HSA_OVERRIDE_GFX_VERSION` handling described above were written without an AMD GPU: they
compile and their table logic is unit tested on the CPU (`hip_arch.rs`, `hip_arch_test.cpp`), but they have not run on the `gfx1153` laptop either, so even that family needs a
re-run of `selftest-gpu` with the current build.

## Please report your result

On an AMD GPU (any family, especially one marked "untested"), with a build that has the `hip` feature:

```sh
fllm-rs devices                          # does the GPU show up? "present but not usable" lines carry the reason
fllm-rs doctor --gpu                     # one line per GPU: OK (N times faster than the CPU), FALLS BACK, WRONG RESULT, SLOWER THAN THE CPU
fllm-rs selftest-gpu --device hip:0      # the kernel checks against the CPU (a few minutes); paste the last lines
```

and send the output together with the GPU name (`fllm-rs devices` prints it with its gfx name), the ROCm / driver version, the log lines that start with `[fllm-rs] hip:` and whether
`HSA_OVERRIDE_GFX_VERSION` / `FLLM_HIP_ARCH_OVERRIDE` was set. If `selftest-gpu` fails, repeat it with `FLLM_HIP_NO_MMA=1`: a pass there points at the matrix-core kernels of your family.
A passing report moves the family from "compiled in CI, untested" to "verified on hardware" in the table above, with the date and the device.
