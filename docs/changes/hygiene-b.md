# Fragment: documentation hygiene, project files, proof harness (hygiene-b)

Base: `main` at 9c97fb2. Nothing in this fragment changes Rust or CUDA sources.

## 1. What changed

* German documents were translated to English and internal development wording was removed: `docs/backends.md`,
  `docs/backends-verify.md`, `docs/model-coverage-vllm.md`, the NPU, SYCL, Metal and Vulkan notes, `scripts/colab/tpu_check.md`
  and the German comments and output labels of `scripts/colab/ab_matrix.py` (JSON keys renamed to English; nothing else read them).
* Research notes moved to `docs/research/` with an index (`npu-study-ane.md`, `npu-study-hexagon.md`, `npu-study-xdna.md`,
  `sycl-evaluation.md`). `docs/release-license-report.md` was deleted.
* New documents: `docs/test-levels.md`, `docs/architecture.md`, `docs/adding-a-backend.md`, `docs/benchmarking.md`,
  `docs/roadmap.md`, `docs/index.md`.
* New project files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `.github/pull_request_template.md`,
  `.github/CODEOWNERS`, `.github/ISSUE_TEMPLATE/feature_request.yml`, `hardware_report.yml`; `config.yml` now disables blank issues
  and links the docs and the private security advisory form.
* Proof harness restored from the private history into `scripts/` and `bench/` (see section 4); `.gitattributes` no longer
  has `export-ignore` for them, so GitHub source archives and `git archive` keep them.
* `scripts/check_docs.py` (links, anchors, referenced paths, German detector, jargon list, YAML syntax) and the workflow
  `.github/workflows/docs.yml` that runs it; `make docs-check`.

Commands: `python3 scripts/check_docs.py [--only links,paths,german,jargon,yaml]`, `make docs-check`.

## 2. Make targets and scripts

| Item | Default | Meaning |
| --- | --- | --- |
| `make venv` | - | creates `.venv` and installs `scripts/requirements.txt` (CPU torch wheel index) |
| `make build-examples` | - | builds `fllm-rs` and the examples the scripts need |
| `make test-arch ARCHS="llama qwen2 ..."` | `llama qwen2 gemma2 qwen3 mixtral qwen3_moe` | tiny random checkpoints vs `transformers` |
| `make test-generation` | same `ARCHS` | greedy text through the server vs `transformers` |
| `make test-model MODEL=x.gguf` | `MODEL` required | `scripts/verify_features.py` against a small real model |
| `make test-encoder MODEL=x.gguf` | `MODEL` required | `scripts/verify_encoder.py` |
| `make docs-check` | - | `scripts/check_docs.py` |
| `FLLM_BIN`, `EXAMPLES`, `PYTHON` (make variables) | `target/release/fllm-rs`, `<bin dir>/examples`, `.venv/bin/python` if present | binary, examples directory and interpreter |
| `FLLM_TEST_MODEL` (env, `scripts/api-tests/cli-tests.sh`) | required | small GGUF for the CLI acceptance test |

No new engine flags or engine environment variables.

## 3. Third-party entries

None. The restored scripts contain no third-party code; they use `transformers`, `torch`, `numpy`, `gguf`, `safetensors`, the OpenAI,
Anthropic and Ollama SDKs as test dependencies (not redistributed).

## 4. Test levels and receipts (honest statement)

Nothing here ran on a GPU. The harness was run on a CPU-only host (AMD EPYC 9255 VM, AVX2) against the CPU build.

Environment: `rust:1-bookworm` (rustc 1.99.0) build of `main` (9c97fb2) with default features, `cargo build --release --locked -p fllm-engine
--bin fllm-rs --example verify_checkpoint --example tokenize --example verify_quant`; Python 3.12 (`python:3.12-bookworm`) with
torch 2.14.1+cpu, transformers 5.19.0, tokenizers 0.23.3, numpy 2.5.4, safetensors 0.8.0, gguf 0.19.0, openai 3.31.0, anthropic 1.14.0,
ollama 0.6.3, requests 2.34.2.

| Command | Result |
| --- | --- |
| `python scripts/verify_architectures.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint llama qwen2 gemma2 qwen3 mixtral qwen3_moe` | all `ok`, 12 positions each, no arg-max mismatch; relative logit difference llama 2.1e-4, qwen2 1.5e-4, gemma2 2.4e-4, qwen3 1.6e-4, mixtral 2.1e-4, qwen3_moe 2.0e-4 (limit 2e-3) |
| `make test-arch ARCHS="llama qwen2"` | same numbers (proves the Makefile target) |
| `python scripts/verify_generation.py --fllm target/release/fllm-rs llama qwen3 qwen3_moe` | 4 of 4 prompts identical to `transformers.generate` for each, with 4 concurrent requests |
| `python scripts/verify_tokenizer.py l.gguf l-hf target/release/examples/tokenize` (tiny llama fixture) | 8/8 lines identical |
| `python scripts/verify_rust_quants.py --create --fixture q.gguf --binary target/release/examples/verify_quant` | CPU matmul max error between 1e-6 and 4e-4 for all fixture types |
| `python scripts/verify_mxfp4.py --fllm ... --example ...` | prints relative logit error 0.068 (quantization error, informational) |
| `python scripts/verify_features.py --binary target/release/fllm-rs --model qwen2.5-0.5b-instruct-q8_0.gguf` | 9 of 9 checks PASS (authentication, tokenize, JSON schema, GBNF, embeddings, tools and streaming, seeded concurrency, invalid parameters, Ollama options) |
| `MODEL=qwen2.5-0.5b-instruct-q8_0.gguf VENV=/venv scripts/api-tests/run_all.sh 8991` | OpenAI, Anthropic and Ollama SDK suites: 3 passed, 0 failed |

`verify_features.py` fails on a random tiny checkpoint (it needs an instruction-tuned model that answers "Berlin"); this is
documented in `scripts/README.md`. Not run: `verify_encoder.py`, `verify_vision.py`, `verify_real_embedding.py` (need real encoder and
vision models), `cli-tests.sh`, `bench-compare.sh`, and every `verify-<backend>` script (need the hardware): L0 only for those.

`actionlint` (docker image `rhysd/actionlint`) passes on all workflows including `docs.yml`.

## 5. Known limits

* `scripts/check_docs.py` currently reports real problems in files owned by others; they are listed in section 7 (checklist).
* The translation of `docs/backends.md`, `docs/backends-verify.md` and `docs/model-coverage-vllm.md` also corrected facts that the
  translators verified against the code (Metal kernel checks run in CI so the Metal rows say L1; CUDA has Mamba, DeltaNet, ShortConv and
  MLA kernels; per-layer host fallback instead of a flat "CPU only"; ARM and macOS CPU self tests run in CI). Re-verify these when the
  final doc pass runs.
* `rust/engine/sycl/build_rs.patch` is stale (`build.rs` already contains `sycl_kernels`); it can be deleted.
* `bench/` contains only the README; the baseline file and the benchmark JSON dumps of the private history were not restored.

## 6. Corrections for README.md (apply in the merge)

| Line (at 9c97fb2) | Old | New |
| --- | --- | --- |
| 107 | `architecture names (German) is in` | `architecture names is in` |
| 210 | `are defined in [docs/backends.md](docs/backends.md) (German).` | `are defined in [docs/test-levels.md](docs/test-levels.md).` |
| 250 | `Some pages are in German or contain development notes; the code and `--help` are authoritative.` | `The code and `--help` are authoritative where a page and the program disagree.` |
| 261 | `... against vLLM/SGLang (German)` | `... against vLLM/SGLang` |
| 276 | `hardware by format matrix, verification steps (German)` | `hardware by format matrix, verification steps` |
| 284 | `, [docs/release-license-report.md](docs/release-license-report.md): security policy, third-party code and licenses` | `: security policy, third-party code and licenses` (the file was deleted) |
| 301 | `optional native backends, test level L0` | `optional native backends; the test level differs per backend (Metal: kernel checks in CI, others L0), see docs/backends.md` |
| Documentation index | - | add `docs/index.md`, `docs/test-levels.md`, `docs/architecture.md`, `docs/adding-a-backend.md`, `docs/benchmarking.md`, `docs/roadmap.md`, `docs/research/README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `scripts/README.md` |
| Building from source | - | add `make test-arch` / `make test-model` and a pointer to `scripts/README.md` |

## 7. Doc audit checklist (apply in one pass after the feature work; not applied here)

An exact patch of the fixes below, produced against 9c97fb2 and reverted from the tree, is delivered as
`doc-audit-fixes.patch` next to this fragment; `git apply` works if the files have not changed, otherwise use it as the reference.
Verify each item against the code first. `python3 scripts/check_docs.py` checks the mechanical parts.

| # | File and line (9c97fb2) | Problem | Correct statement | Verify |
| --- | --- | --- | --- | --- |
| a | `docs/server-api.md:3-8, 12-40` | Opens with "lists every route" but the Route overview lacks `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses[/{id}]`, `/v1/files*`, `/v1/batches*`, `/v1/models/{id}`, `/api/show`, `/api/ps`, `/api/pull`, `/api/delete`, `/completion`, `/infill`, `/apply-template`, `/embedding(s)`, `/props`, `/slots`, `/generate_stream`, `/info`, `HEAD /` | Rebuild the table from the router and add a family column. `/v1/rerank`, `/rerank`, `/v1/score`, `/classify`, `/pooling` answer 501 on a decoder (only the encoder server implements them); `/api/pull` and `/api/delete` answer 501 | `rust/engine/src/server.rs:200-260`, `server/api/*.rs`, `encoder_server.rs:504-521`, `encoder_required` handlers at `server.rs:480-498` |
| a2 | `docs/server-api.md:290` and `docs/configuration.md:24` | Heading "Tools und Reasoning" is German; configuration.md links `#tools-und-reasoning` | Heading "Tools and reasoning", anchor `#tools-and-reasoning`, update the link in configuration.md | `python3 scripts/check_docs.py --only links` |
| b | `docs/configuration.md:9, 31` | Says `--device` accepts only `auto|cpu|vulkan|metal|dx12`, that `auto` and `cpu` select the CPU and that `cuda` is not accepted | See section 8 | `fllm-rs --help`, `fllm-rs devices`, `docs/device-selection.md`, `rust/engine/src/bin/serve.rs:53-54` |
| c | `docs/inference-features.md` (Decoder, Quantization, Hardware rows; `--device auto` paragraph; last paragraph) | States "Qwen3 and SmolVLM's Llama decoder tested", "MoE, MLA, recurrent ... not covered", "NVIDIA, AMD, Apple/Metal and DX12 hardware have not been validated", "LayerNorm, attention, activation and cache handling currently stay on the Rust CPU", only eight quantized formats packed, and `--device auto chooses CPU` | Decoder covers dense, MoE, MLA and recurrent mixers (`docs/architectures.md`); CUDA is a full device-resident decoder (T4 and RTX PRO 6000 run, README Status); native Metal kernel checks run in CI; `--device auto` takes the best GPU (CUDA > ROCm > Metal > Vulkan > DX12 > SYCL) else the CPU; quantization list is `fllm-rs formats`; CPU SIMD tiers include AVX-512 BF16, AMX and NEON i8mm. "Not provided: runtime LoRA switching" in `api-compat.md:171` and README:149 is correct. The exact rewrite is in `doc-audit-fixes.patch` | `fllm-rs formats`, `rust/engine/src/devices.rs` (`auto_chain`), README Status table; file has CRLF line endings: edit in binary mode |
| d1 | `docs/gpu-backends.md:323` | Says Q2_K, Q3_K and the IQ family are re-quantized to Q5_K; line 276 says they are native | They are native (generic decoders in `iq_decode.cuh`); only types outside the list and unusual shapes are converted on upload | `NATIVE_KINDS` in `rust/engine/src/backend/cuda.rs:492` |
| d2 | `docs/gpu-backends.md:327` | "Tensor-core kernels need compute capability 8.0" | Gate is `cc >= 75` for the FP16 tensor-core, Marlin Q4Z and NVFP4-Marlin paths; some paths need 80 (`marlin_q4k`, `cc >= 80` at `cuda.rs:1041`); FP4 kernels need sm_12x | `rust/engine/cuda/api.cu:1100`, `cuda.rs:670, 977, 988, 1041, 1130` |
| d3 | `docs/gpu-backends.md:425-434` | Graph text must say graphs are on by default, `FLLM_CUDA_GRAPH=0|off|false|no` off, `1|on|true` explicit | The text is already correct on this commit; re-check if the feature work changes `parse_graph_switch` | `rust/engine/src/rust_decoder.rs:5593-5613`, `docs/configuration.md:104` |
| d4 | `docs/gpu-backends.md:11, 226`, `:188` | "package p15" jargon; reference to `rust/engine/src/backend/vulkan.rs` (the backend is the directory `vulkan/`) | Neutral wording; `rust/engine/src/backend/vulkan/` | `check_docs.py` |
| e | `docs/ci.md:133` and last section | Claims a `.github/dependabot.yml` and "No workflow has run on GitHub yet" | Workflows: `ci`, `platforms`, `release`, `docker-image`, `docker-image-cuda`, `kernels`, `conformance`, `streaming-benchmark`; Dependabot disabled until further notice (commit 1e44b82); `ci`, `platforms`, `release`, Docker workflows run on hosted runners; `docs` workflow added by this fragment (add a row) | `.github/workflows/*.yml`, `git log` |
| e2 | `docs/ci.md` platform table | Says nothing has a GPU and omits what runs on ARM and macOS | `ubuntu-24.04-arm` runs `selftest-cpu` on ARM hardware; `macos-14` runs the unit tests with `rust-gpu,metal` (Apple-silicon GPU) | `.github/workflows/platforms.yml:55-58, 117` |
| f | `docs/backends.md`, `docs/backends-verify.md` (done in translation) | Metal and Vulkan "L0", CUDA Mamba "Host", ARM "not run" | See section 5; re-verify after the CUDA/Metal feature work | `cuda.rs` `supports_mamba1/2`, README Status table, `platforms.yml` |
| g | `docs/cpu-kernels.md:319-324` ("Not implemented") | Lists AMX-INT8 as not implemented, but the `amx` tier exists (inline assembly, `avx512` feature, default on) | AMX prompt GEMMs exist and have run only in an emulator; AMX is not used for decode | `docs/cpu-avx512-amx.md:34, 116`, `rust/engine/src/cpu_avx512/` |
| h1 | `docs/architectures.md:59-62, 67, 80-81` | "LFM2 also selects CPU under auto and rejects explicit accelerators"; table says CPU implementation | Dense `lfm2` runs on CUDA (ShortConv kernels); `lfm2moe` and `dbrx` are CPU-only | `rust/engine/src/rust_decoder.rs:1011-1020` |
| h2 | `docs/architectures.md:85` | "The DeltaNet recurrence runs on the CPU between device layers" | Runs on the device with CUDA (`deltanet.cu`); other backends: CPU between device layers | `rust/engine/cuda/deltanet.cu`, `supports_delta` in `backend/mod.rs` |
| h3 | `docs/model-coverage-vllm.md` (done in translation) | "CPU only" column partly stale | Per-layer host fallback via `layer_host_reason` / `ffn_host_reason` | `rust_decoder.rs:1727-1790` |
| i | `rust/engine/src/bin/serve.rs:53` (source fix, not docs) | Usage line prints `--max-batched-tokens 1024`, code default is 512 (`serve.rs:254`) | Change the usage string to 512; then drop the "stale usage line" note at `docs/user/commands.md:27-29` | `serve.rs:254` |
| j | See section 6 | README | | |
| k | `THIRD_PARTY.md:126, 199`, `docs/cuda-qgemm4k.md:76`, `docs/ci.md`/`bench/README.md` (`bench/baseline.json`), `rust/gguf-probe/README.md:5,7` | Paths that no longer exist in the public tree: `bench/test_vllm_marlin.cu`, `bench/bench_qgemm.cu`, `rust/engine/sycl/EVALUATION.md` (moved to `docs/research/sycl-evaluation.md`), `benchmarks/20260929/*.json` | Update or remove the references (THIRD_PARTY is owned by another agent) | `python3 scripts/check_docs.py --only paths` |
| l | `install.sh:8,25,27,142,145,147`, `install.ps1:27,28,107,110,111` | Mention Forgejo (the former private forge) | Describe GitHub releases only (installer owner) | `python3 scripts/check_docs.py --only jargon` |
| m | Source comments (a later automated pass) | `STUDY_*.md` references: `rust/engine/src/backend/npu/hexagon.rs` lines 4, 69; `ane.rs` 3, 52; `hexagon/graph.rs` 6; `xdna.rs` 3, 72 | Point at `docs/research/npu-study-*.md` | `grep -rn STUDY_ rust/` |

## 8. Corrections for docs/configuration.md (old line -> new line)

Line 9 (usage):

```text
old: fllm-rs serve --model MODEL.gguf [--device auto|cpu|vulkan|metal|dx12] ...
new: fllm-rs serve --model MODEL.gguf [--device auto|cpu|cuda|hip|metal|vulkan|dx12|sycl|npu|tpu|BACKEND:N] ...
```

Line 31 (`--device` row):

```text
old: | `--device` | string (lowercased) | `auto` | `auto` and `cpu` select the CPU. `vulkan`, `metal`, `dx12` (also `gpu`/`wgpu` = all compiled backends) need the `rust-gpu` feature; a software adapter is refused. Any other value fails with `unsupported Rust GPU backend …` (or, without the feature, `rebuild with --features rust-gpu …`) — including `cuda`, which the built-in usage line mentions but the parser does not accept |
new: | `--device` | string (lowercased) | `auto` | `auto` takes the best GPU of the build in the order CUDA > ROCm > Metal > Vulkan > DX12 > SYCL (`FLLM_AUTO_ORDER` changes the order) and falls back to the CPU when there is none; `cpu` forces the CPU. `cuda`, `hip` (ROCm), `metal`, `vulkan`, `dx12`, `sycl`, `npu`, `tpu` select that backend when it was compiled in; `BACKEND:N` picks device N of a backend (`vulkan:1` is the second Vulkan device; `fllm-rs devices` prints the value to use); `gpu`/`wgpu` mean any wgpu adapter and need the `rust-gpu` feature. An explicit GPU request fails if the backend is unavailable or only a software adapter exists. See [device-selection.md](device-selection.md) |
```

Line 24 (`--tool-parser` row), link text and anchor: `[Tools und Reasoning](server-api.md#tools-und-reasoning)` ->
`[Tools and reasoning](server-api.md#tools-and-reasoning)`.

Line for `--max-batched-tokens` (if the row repeats the 1024 usage value): the default is 512.

Add a row for the new Make variables only if configuration.md documents the Makefile (it does not at present).
