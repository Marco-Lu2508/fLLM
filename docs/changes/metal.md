# Native Metal backend: whole-model tests, arch ops, MLA, Mamba-2, static MSL checks

This fragment describes what was added to the native Metal backend (feature `metal`), what the Apple-silicon CI runner is expected to show,
and what is not done. Nothing here was run on an Apple GPU by the author: there is no Mac and no GPU on the development host. Everything that is
called "verified" below was verified on Linux (CPU emulation of the MSL text, unit tests of the planners, a clippy cross check for
`aarch64-apple-darwin`); everything that needs Apple's Metal compiler or a GPU is a prediction that the CI run will confirm or refute.

## 1. Features

### Whole-model tests (run by the macOS 14 job of `platforms.yml`)

`rust/engine/src/tiny_models.rs` builds tiny random models in the test (F32 safetensors in the Hugging Face layout, converted with
`fllm_checkpoint::convert_checkpoint`, optionally re-encoded with a quantization preset). `rust/engine/src/backend/metal/whole_model_tests.rs`
loads each model twice, on the host path of the decoder and attached to the Metal backend, prefills a 6-token prompt, decodes 16 tokens and compares:

1. teacher-forced: the Metal run is fed the host's greedy tokens, the logits of all 16 steps must agree within a tolerance relative to the largest logit;
2. free-running: the Metal run's own greedy tokens must equal the host's, up to a step at which the host's two best logits are closer than twice the tolerance;
3. no layer and no feed-forward block may fall back to the host (`RustDecoder::host_fallbacks`).

| test (`whole_model_*`) | architecture | weights | tolerance |
| --- | --- | --- | --- |
| `llama_f32`, `llama_f16`, `qwen3_f32` | Llama, Qwen3 (q/k norm) | F32, F16 | 5e-3 |
| `llama_q8_0`, `llama_q4_0` | Llama | Q8_0, Q4_0 | 8e-2 |
| `llama_q4_k_m` | Llama | Q4_K, Q6_K (output head, last layer attn_v / ffn_down) | 8e-2 |
| `qwen3_q6_k`, `qwen3_q5_k_m` | Qwen3 | Q6_K, Q5_K (+ Q6_K) | 8e-2 |
| `mixtral_f16`, `mixtral_q4_k_m` | Mixtral (4 experts, top 2) | F16, Q4_K/Q6_K | 5e-3, 8e-2 |
| `qwen3_moe_q8_0` | Qwen3-MoE (4 experts, top 2, renormalised) | Q8_0 | 8e-2 |
| `qwen35_f32`, `qwen35_q8_0` | Qwen3.5 style: gated DeltaNet layer + full-attention layer with query gate | F32, Q8_0 | 5e-3, 8e-2 |

The tolerances are estimates: on the CPU backend standing in for the device the Q8_0 model differs by 1.9e-2 and the F32 model by 4e-5. If a Metal
result is slightly above a tolerance the failure message prints the measured value, the step and the vocabulary entry; `FLLM_MODEL_TEST_TOL=0.05`
overrides the tolerance of a run.

The fixtures, the converter and the comparison itself run on every platform: `tiny_models::tests` (Linux CI included) uses the CPU backend in the role of
the device (`cpu_backend_matches_host_path_*`) and checks that the MoE and Qwen3.5 fixtures load and decode on the host path.

Run on a Mac: `cargo test --release -p fllm-engine --lib --features metal whole_model -- --nocapture --test-threads=1`
(`FLLM_METAL_SERIAL=1` first when a model fails: it rules out missing barriers). `platforms.yml` runs exactly this in an extra step of the macOS 14 job
(`--nocapture`, so that the result lines are in the log); the tests also run inside the normal "Unit tests" step.

### Arch ops on the device (`Stream::supports_arch_ops` is now true)

New kernels in `rust/engine/metal/arch_ops.metal` and four more modes in `act_glu.metal`:

* `activation` / `activation_glu` for all five modes (SiLU, GELU tanh, GELU erf, ReLU, ReLU squared; with and without the `up` operand). GELU erf uses
  the Chebyshev fit of erfc that the host reference uses (`o_erfc_pos` in `common_ops.metal`; the Metal standard library has no `erf`).
* `clamp` (QKV clipping), `softplus_gate_mul` (K2 Horizon attention gate).
* `mla_latent`, `mla_assemble`: multi-head latent attention in its expanded form (DeepSeek V2/V3, Kimi, GLM-MoE-DSA). The latent cache of the CUDA `hopper`
  work is not part of this; the expanded cache holds per-head keys and values like the CPU path.
* Effect in the decoder: plain (non-gated) MLPs, ReLU² / GELU-erf models, the attention output gate of Qwen3.5 (it needed `supports_arch_ops`),
  QKV clipping and MLA layers run on the Metal device instead of the host fallback layer.

The kernels have three layers of checks: the CPU emulation below (formulas and indexing), the unit tests of the planners against the
`FLLM-KERNEL` headers (Apple targets), and on the Apple runner `device_arch_ops_*` in `backend/metal/device_arch_tests.rs` (every activation mode plain and
gated, clamp, softplus gate, MLA data movement, each against a formula in double through the `Stream` API) plus the new GLU modes in the existing
`metal_ops_kernels` hardware check (`act_fused_gelu_erf`, `act_fused_geglu_erf`, `act_fused_reglu`, `act_fused_relu2glu`).

### Mamba-2 on the device

`Stream::supports_mamba2` / `mamba2_layer` call the existing planners (`mamba_conv`, `mamba_scan2_*`, `mamba_gate_norm`; emulated on the CPU by the previous
work). Mamba-1 and the LFM2 short convolution still have no Stream method (see section 5).

### Static MSL checks (`rust/engine/src/msl_check.rs`, runs everywhere)

A lexical lint of every `.metal` file: all thread-position attributes of a kernel have one type (the rule the CI violated twice), pointer and reference
parameters name an address space, `[[buffer(n)]]` indices are distinct and below 31, `threadgroup` arrays are declared at the top of a kernel body and the
evaluated static threadgroup memory stays within 32 KiB, program-scope variables are `constant`, no `double` / `printf` / `erf`, kernel names are unique.
`cargo test --release -p fllm-engine --lib msl_check`. The checker is itself tested with bad snippets. It found no violation in the existing sources.

### CPU emulation of the new kernels (`rust/engine/metal/emu_arch/`)

`run_emulation.sh` compiles the real text of `common_ops.metal`, `act_glu.metal` and `arch_ops.metal` as C++ (a 90-line MSL shim, `g++` only) and runs every
new kernel and mode against references in double: 61 checks (all activation modes, strided rows, a GELU-erf sweep over -20..20, clamp and softplus with
grid-stride loops of fewer threads than elements, MLA latent and assemble for three shapes including 128/64/128 with 16 heads). Result of the last run:
`61 checks, 0 failed`, largest error 3.6e-7 (softplus, beta 0.5). It proves index arithmetic and formulas, not Metal compiler acceptance.

### Packaging

* `release.yml`: the Apple-silicon asset is built with `rust-gpu,metal` when the repository variable `FLLM_RELEASE_METAL` is `1` (default: unset, the asset
  is unchanged). Set it after the CI log below looks right.
* `--device auto` already ranks `metal` (native) before wgpu when the binary was built with the `metal` feature (`devices.rs`, `DEFAULT_AUTO_ORDER`), and
  falls back to the next candidate when the native backend fails to start or cannot take the model. Nothing had to change there.
  `fllm-rs devices` already prints chip, family, working set and a bandwidth class for native Metal devices.

## 2. Environment variables and switches

| name | default | meaning |
| --- | --- | --- |
| `FLLM_MODEL_TEST_TOL` | per test | tolerance (relative to the largest logit) of the whole-model tests, test runs only |
| `FLLM_RELEASE_METAL` (repository variable) | unset | `1`: the `aarch64-apple-darwin` release asset includes the native Metal backend |
| `FLLM_METAL_SERIAL=1` | off | existing: serial encoder, first thing to try when a model fails |

## 3. Third-party code

No new third-party code. `o_erfc_pos` uses the Chebyshev coefficients of Numerical Recipes (Press et al.) that `rust_decoder.rs` already contains; the
element-wise kernels are ports of the repository's own `cuda/arch_ops.cu` and `cuda/mla.cu`; `emu_arch/msl_emu/emu.h` derives from the repository's
`emu_recurrent/msl_emu/emu.h`.

## 4. Test levels

| item | level | what was run, where |
| --- | --- | --- |
| whole-model tests, fixtures, comparison logic | L0 for Metal; L2 for the harness | the harness ran on Linux with the CPU backend as the device: Llama F32 differs by 4.4e-5, Qwen3 Q8_0 by 1.9e-2, 16/16 greedy tokens equal; MoE and Qwen3.5 fixtures load and decode on the host path. The Metal side has not run. |
| arch kernels, `act_glu` modes | L0 for Metal; emulated on the CPU (61 checks pass) | `emu_arch/run_emulation.sh` in `rust:1-bookworm` (g++ 12) |
| plans of the new kernels | unit-tested on Linux against the `FLLM-KERNEL` headers | `ops/arch_plan.rs` tests, kernel inventory (`backend/kernels.rs`) |
| Mamba-2 stream wiring | L0 (compiles for `aarch64-apple-darwin`) | clippy cross check only; the kernels were emulated by the earlier work |
| static MSL lint | L2 on Linux | unit tests |
| `aarch64-apple-darwin` build | `cargo clippy --target aarch64-apple-darwin --features rust-gpu,metal --lib --tests -- -D warnings`, see the report | |

### What the Apple runner should print

`cargo test ... whole_model -- --nocapture` (step "Metal whole-model tests (Apple GPU)"), one line per model, shaped like

```
metal llama-f32: 16 steps, largest logit difference 2.1e-6 (tolerance 5.0e-3), 16/16 free-running greedy tokens equal, 0 host layer(s), 0 host feed-forward block(s), host tokens [64, 76, ...]
metal qwen35-f32: 16 steps, largest logit difference ...
```

The same step also runs `device_arch_ops` (three tests, lines like `metal activation mode 2: largest error 8.9e-8`). Expected: 13 `whole_model` lines
and 3 `device_arch_ops` tests, `test result: ok. 16 passed`, differences around 1e-5 for the float models and 1e-2 for the quantized ones. Reading a failure:

* `run on the host instead of the metal backend`: the message names the count; the line `[fllm-rs] N of M layers run on the host between the device layers (reason)` above it names the reason.
* `logits differ by ...`: note which models fail. All fail: `FLLM_METAL_SERIAL=1`, then `FLLM_METAL_FAST_MATH=0`. Only quantized fail: the quantized GEMV/GEMM of that format. Only MoE: `moe_linear`/`moe_route`. Only `qwen35`: DeltaNet or the query gate.
* a compile error of a `.metal` file is reported when the first model creates the backend (`MetalBackend::new` compiles probe kernels) or when a pipeline is first used; the message contains Apple's compiler text with file and line (`#line` directives).

## 5. Not done (precise list)

Needs a Mac or a GPU to be meaningful, or was out of the time budget:

* Mamba-1 (`mamba1_conv`, `mamba1_prep`, `mamba1_scan`; the planners exist, `mamba1_prep` needs a small row-copy kernel for the dt part) and the LFM2
  short convolution (`shortconv_layer`, no kernel) have no Stream method: those layers run on the host between device layers.
* `moe_linear` still handles ~10 formats (no Q2_K, Q3_K, NVFP4, Q4Z, F16); `moe_route` on the device is softmax only. Both use the host fallback of the decoder.
* Grouped MoE GEMM with sorted tokens, IQ1-IQ4 / MXFP4 / NVFP4 / Q4Z / FP8 native kernels (these are converted to F16/F32 at upload, as before),
  BF16 in MSL (`bfloat`).
* xIELU stays on the host (the decoder routes it there).
* Performance: nothing was measured or tuned. Not started: simdgroup-matrix prompt GEMMs beyond the existing `q_mm_t`, 64x64 tiles for M5, fused
  add+RMSNorm / rope+KV-write / SwiGLU+quant kernels, prompt attention for head widths above 256, indirect command buffers, `MTLResidencySet` beyond what the
  Metal core already does, splitting weights above `maxBufferLength`, `newBufferWithBytesNoCopy` for the memory-mapped GGUF in the decoder (the Metal core
  has `register_host_mapping`, the decoder does not call it), GPU sampling (temperature, top-k/p, penalties), async copy/compute overlap.
* Metal 4 / MPP tensor paths for M5: nothing; the code reports `metal4` but uses Metal 3 features only.
* Memory planner for unified memory (working-set warning, CPU+GPU hybrid by layer): not done.
* Quantized KV cache other than f16 and q8_0.
