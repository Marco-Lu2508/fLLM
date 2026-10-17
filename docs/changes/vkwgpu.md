# Software-Vulkan test rig, Vulkan arch ops, wgpu LayerNorm and adapter selection

Change fragment of the `vkwgpu` work. Test levels: L0 = compiles / unit tests only, L1 = `selftest-gpu` ran on a device, L2 = whole-model text matches
the CPU, L3 = benchmarked. Everything below that ran on a device ran on **Mesa lavapipe (a CPU rasteriser)**, not on a GPU: it validates shader logic
and host code against the CPU reference, not performance, driver quirks, cooperative matrices or subgroup behaviour of real hardware.

## Features

### 1. Software Vulkan rig (`scripts/vulkan-soft-test.sh`, `docker/Dockerfile.vulkan-soft`)

```
scripts/vulkan-soft-test.sh selftest   # fllm-rs selftest-gpu --device vulkan on lavapipe, report in ./vulkan-soft-out/selftest-vulkan.{json,txt}
scripts/vulkan-soft-test.sh unit       # cargo test (vulkan, wgpu, selftest, device fallback tests)
scripts/vulkan-soft-test.sh model      # only the whole-model test, with output
scripts/vulkan-soft-test.sh all        # unit + selftest (default)
scripts/vulkan-soft-test.sh selftest --expect-skips rust/engine/src/backend/vulkan/EXPECTED_SKIPS.md
```

The image (Ubuntu 24.04: Rust, `glslc` 2023.8, Mesa lavapipe, vulkan-tools) is built on first use. glslc of that age compiles every shader of the
tree, including `mm_coopmat.comp`. lavapipe is admitted through `FLLM_ALLOW_SOFTWARE_GPU=1` (set by the script). The first run on the software
driver (the first run of the native Vulkan backend on any driver) found no shader bug: 0 failures in the exact pass. The one failing group of the
fast pass was a defect of the self test (the CUDA-only NVFP4 W4A4 check read a tensor-core trailer that other layouts do not have); it now
reports `SKIPPED (unsupported)`.

Result on lavapipe with `--expect-skips`: exact pass 592 checks (256 ok, 0 failed, 336 skipped), fast pass 447 checks (373 ok, 0 failed, 74 skipped);
1039 checks, 629 ok, 410 skipped, all skips as listed in `EXPECTED_SKIPS.md`. The run takes about 10 minutes on 4 llvmpipe threads (the Marlin-sized
fast cases dominate). Whole-model test (L2 on lavapipe): float model, the same 8 tokens, first-step logits within 2.9e-7 of the largest logit;
Q8_0 model, the same 8 tokens, logits within 9.6e-3.

### 2. Explicit exact / fast policy of the Vulkan self test

`selftest-gpu --device vulkan` now runs two passes like `cuda` and `hip`: pass 1 forces the FP32 shaders (`set_exact_math(true)`, whatever
`FLLM_VULKAN_EXACT` says) and holds every operation to the FP32 error bound; pass 2 uses the shaders a normal run picks for the device
(cooperative matrix, integer dot product, f16 activations) and holds the projections to `FAST_TOL` (4e-3 of the error scale). Each pass prints
`kernel variant: exact ...` / `kernel variant: fast ...` before its table, every record and the JSON carry `pass`, and the JSON has a `passes` array.
Before this change one pass ran with whatever variant the device chose and the FP32 bound, so a device with integer-dot or coopmat kernels failed
checks that the fast pass is meant to cover.

### 3. Skip accounting

- The JSON report has `skipped_ops`: one entry per (group, op) with count and reason; the `groups` object already carried per-group `skipped`.
- `--expect-skips FILE` compares the skipped checks with the `expected-skips` block of FILE and fails on any difference in either direction.
  `rust/engine/src/backend/vulkan/EXPECTED_SKIPS.md` is the committed expectation for the Vulkan backend (62 entries, 410 skipped checks of 1039).
- A backend without arch operations (`supports_arch_ops` false) used to leave out the whole `arch` group silently; it now records one skipped check.

### 4. Vulkan operations added to the Stream (L1 on lavapipe)

| Stream method | shader | note |
|---|---|---|
| `activation`, `activation_glu` (SiLU, GELU tanh, GELU erf, ReLU, ReLU squared; gated forms) | `act_glu.comp` modes 7-11 (`act_gelu_erf`, `act_geglu_erf`, `act_reglu`, `act_reglu2`, `act_copy`) | GELU erf = the decoder's erfc fit |
| `clamp`, `softplus_gate_mul` | `ewise.comp` (`ewise_clamp`, `ewise_softplus`) | in place |
| `mla_latent`, `mla_assemble` | `mla_move.comp` (`mla_move_q/k/v/latent`) | expanded MLA data movement; exact copies |
| `supports_arch_ops` | | now true: LayerNorm models, ReLU^2, GELU-erf and MLA layers stay on the device |
| `supports_mamba2` / `mamba2_layer` | existing `mamba_conv`, `mamba_scan2_*`, `mamba_gate_norm` | `stream/mamba.rs` was prepared but not wired |
| `supports_mamba1`, `mamba1_conv`, `mamba1_prep`, `mamba1_scan` | existing `mamba_conv`, `mamba_scan1_*`, RMS norms, `act_copy` | state layout of the trait: recurrent state, then history |

`arch` group on lavapipe: 19 of 19 checks ok (LayerNorm, five activations, softplus gate, clamp, MLA, three Mamba-1 and four Mamba-2 shapes).

### 5. Whole-model test (`backend/vulkan/model_tests.rs`)

A random Llama-like model (64 hidden, 4/2 heads, 2 layers; the synthetic checkpoint builder of `import_tests.rs`), a 5-token prompt and 8 greedy
tokens on the CPU and on native Vulkan with the exact shaders: `vulkan_whole_model_float_greedy_tokens_match_the_cpu` requires identical tokens
and first-step logits within 2e-4 of the largest logit; `vulkan_whole_model_q8_0_logits_match_the_cpu` (Q8_0 weights; the CPU quantizes the
activations to int8, the exact shaders do not) requires the logits within 3e-2 and reports the tokens. Without a device the tests print why and
pass (GitHub's runners have none); with `FLLM_ALLOW_SOFTWARE_GPU=1` they run on lavapipe. They switch the process-wide exact flag: use `--test-threads=1`.

### 6. wgpu (generic GPU path)

- WGSL `LayerNorm` kernel (two-pass mean / variance, optional bias): `supports_layer_norm` is true, so Falcon, Bloom, MPT, GPT-2/J/NeoX, StarCoder,
  OPT, Phi-1/2, StableLM, Cohere, Nemotron, Orion, DBRX, OLMo-1 and Jais2 are no longer rejected by `init_backend`.
- Arch operations (`supports_arch_ops` true): `activation`, `activation_glu` (SiLU, GELU tanh, GELU erf, ReLU, ReLU^2), `clamp`, `softplus_gate_mul`,
  `mla_latent`, `mla_assemble` as WGSL kernels. This also lets MLA, non-SiLU/GELU activations, QKV clipping and gated attention run on the device.
- L0 only for these kernels: the WGSL parses and validates with naga (`shaders_parse_and_validate`) and the code compiles for Linux, Windows (MSVC) and
  macOS targets; they have not run on a device. The `arch` group of `selftest-gpu --device gpu` is the check to run on a real adapter.

### 7. Adapter selection and refusal of software adapters (DX12 / Vulkan through wgpu)

`GpuContext::new` used `request_adapter` for index 0 (which may return WARP) and an enumeration for index N. It now enumerates once and applies
`choose_adapter`: index 0 is the first discrete GPU, else the first hardware adapter; index N the N-th hardware adapter (the order of `fllm-rs devices`);
software renderers (WARP, lavapipe, SwiftShader) are used only when no hardware adapter exists and `FLLM_ALLOW_SOFTWARE_GPU=1`. Messages:
`no hardware dx12 GPU adapter found; only software renderers are present (...); set FLLM_ALLOW_SOFTWARE_GPU=1 to use one anyway (testing), or run on the CPU with --device cpu`,
`no dx12 GPU adapter found (install the graphics driver ...)`, `no dx12 GPU adapter number 2 (1 hardware adapters found)`. Pure-function unit tests cover the cases.

CI: `scripts/no-gpu-smoke.sh` (new step in `platforms.yml`, Linux and Windows): on a runner without a hardware adapter `--device dx12` and `--device vulkan`
must exit non-zero with the messages above and no panic, `--device cpu` must work. The unit test `device_fallback_tests` checks that `--device auto`
ends on a working device and that an explicit `dx12` without hardware is a clear error. Nothing was run on Windows here: the Windows (MSVC) target only
type-checks (`cargo xwin check`), see the report.

## Environment variables and flags

| Name | Default | Meaning |
|---|---|---|
| `selftest-gpu --expect-skips FILE` | off | fail when the skipped checks differ from the `expected-skips` block of FILE |
| `FLLM_ALLOW_SOFTWARE_GPU` | unset | admit a software renderer (lavapipe, SwiftShader, WARP) as GPU (native Vulkan and wgpu); `0` or empty counts as unset for wgpu |
| `FLLM_VULKAN_EXACT` | off | unchanged; `selftest-gpu --device vulkan` overrides it per pass |
| `scripts/vulkan-soft-test.sh`: `FLLM_VK_IMAGE`, `OUT`, `LP_NUM_THREADS`, `CARGO_BUILD_JOBS`, `WRAP`, `CACHE_VOLUME` | `fllm-vulkan-soft`, `./vulkan-soft-out`, 4, 3, empty, `fllm-cargo-cache` | image, report directory, llvmpipe threads, build jobs, command prefix (build-slot wrapper), cargo registry volume |

## THIRD_PARTY entries

None. The GELU erf fit is the one already used by `rust_decoder.rs` (Numerical Recipes erfc Chebyshev fit, as in the existing code); no code was taken from other projects.

## Known limits / not done

See the report. In short: ShortConv on Vulkan, GPU sampling (wgpu and Vulkan), WGSL kernels for Q2_K/Q3_K/IQ4_NL/IQ4_XS/MXFP4, `moe_expert_gemm` and device-side
`moe_route` on Vulkan, attention for head width > 512, more KV formats, native shaders for the formats still re-quantized at upload, subgroup-size /
cooperative-matrix property queries, buffer-device-address, async transfer, command-buffer reuse, timeline semaphores, expert single storage.
