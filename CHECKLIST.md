# README claim checklist

Paths are relative to the repo root. "cmd" = run with the CPU binary built from commit 1e53396
(default features, and a second build with `--features rust-gpu`, both in rust:1-bookworm without cmake).

## Intro
- 35 registered ggml types: cmd `fllm-rs formats` -> "Registered GGML tensor types: 35".
- Import of safetensors / PyTorch: rust/engine/src/cli.rs:106,447 (`.safetensors/.bin/.pt/.pth`), docs/rust-formats-and-import.md "Checkpoint containers".
- AWQ/GPTQ 4-bit incl. act-order: rust/checkpoint/src/convert.rs:1000-1040 (quant_method gptq/awq), docs/awq-gptq-marlin.md (Q4Z, Q4ZP).
- FP8: convert.rs:357-374, 599-608 (FP8M); quantization() arm "fp8" | "compressed-tensors" at convert.rs:1039.
- MXFP4 (gpt-oss): rust/checkpoint/src/gptoss.rs:2,48,175.
- EXL2/EXL3: convert.rs:998-999,1126, docs/exl-formats.md.
- APIs list: rust/engine/src/server.rs:200-252 (route table), docs/api-compat.md table.
- No llama.cpp/GGML dependency: `grep -ci "ggml|llama" Cargo.lock` -> 0; THIRD_PARTY.md lines 1-40.
- Non-Rust sources (CUDA/HIP/Metal/SYCL/shaders): ls rust/engine/{cuda,hip,metal,sycl,shaders}; Cargo features rust/engine/Cargo.toml:40-56.
- Adapted from vLLM (Marlin) / CUTLASS / FlashInfer: THIRD_PARTY.md (FlashInfer at lines 72-76), licenses/ (MARLIN, vLLM, CUTLASS).
- No Python at run time: images contain only the binary (docs/docker.md); the public tree has no Python scripts (`ls scripts` -> build-windows.ps1 only).
- Image input SigLIP/Idefics3 only: rust/engine/src/vision.rs:38 (only the `idefics3` projector type is accepted).

## Quick start
- v0.1.0 assets: GitHub releases API: fllm-linux-aarch64-musl, fllm-linux-x86_64-musl, fllm-macos-aarch64, fllm-macos-x86_64 (.tar.gz + .sha256 each); `prerelease: true`. No Windows asset in the list.
- install.sh: `sh install.sh --help`; header lines 1-35 (installs to ./.local/bin/fllm-rs, verifies .sha256, `--build`).
- install.ps1 falls back to cargo build when no asset matches: install.ps1 header ("The fallback build (also used when the release has no matching asset)").
- Release binaries = rust-gpu, not cuda: .github/workflows/release.yml:41-42, docs/ci.md "The feature set is RUST_FEATURES".
- Docker tags: ghcr.io registry API (tags/list): 0.1.0, v0.1.0, latest, sha-926420a, cuda-v0.1.0, cuda, cuda-sha-926420a; the manifest of `latest` lists amd64 and arm64. `cuda` is documented as linux/amd64 (docs/ci.md "Docker images"); its manifest was not inspected.
- Cache path /home/fllm/.cache: docker/Dockerfile:27-28 (`useradd -m fllm`), docs/quickstart-windows-linux.md.
- CUDA image needs only the driver: docs/docker.md "CUDA image", docs/quickstart-windows-linux.md.
- Source build: cmd `cargo build --release --locked -p fllm-engine --bin fllm-rs` and the same with `--features rust-gpu` succeeded in rust:1-bookworm (gcc, git, no cmake). The old README, docs/getting-started.md and install.sh say CMake is needed; the builds did not need it.
- `run` forms: cmd `fllm-rs run --help`; rust/engine/src/hub.rs:19-68 normalize_reference; catalog alias `qwen3.5:9b`: cmd `fllm-rs search`; rust/engine/src/catalog.rs:172-196 (quant chosen by available memory, default Q4_K_M).
- ":QUANT needed for repos with many files": docs/user/commands.md "Model cache and catalog".

## Models and formats
- serve loads only GGUF; a directory without .gguf is an error pointing at convert: rust/engine/src/bin/serve.rs:476-512.
- ModelOpt not read by convert: cmd `fllm-rs convert <dir with quant_method modelopt> out.gguf` -> `unsupported quantization_config quant_method "modelopt"` (tested with a minimal config.json); no "modelopt" string in rust/checkpoint/src. docs/gpu-backends.md "NVFP4 models" says to use llama.cpp convert_hf_to_gguf.py.
- NVFP4 = GGUF type 40, MXFP4 = type 39: cmd `fllm-rs formats`.
- Q4Z/Q4ZP, group sizes, fallback to F16: docs/awq-gptq-marlin.md lines 1-25.
- Native kernel table. CPU: docs/cpu-kernels.md, docs/rust-formats-and-import.md ("Optimized CPU kernels cover ... on x86-64 also Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4"), docs/backends.md note 2 (NVFP4 and Q4Z have no SIMD kernel). CUDA: docs/gpu-backends.md "CUDA backend" table, rust/engine/src/backend/cuda.rs:433-436, docs/awq-gptq-marlin.md "Execution" and "NVFP4 on GPUs without FP4 tensor cores". wgpu: docs/gpu-backends.md section 1 item 4.
- Architecture families and unsupported list: docs/architectures.md lines 15-100.
- "Verified" definition and "real checkpoints not available": docs/architectures.md lines 3-10.
- Coverage numbers 43/25/2/16: docs/model-coverage.md "Result" table.
- Text model only for multimodal checkpoints: docs/architectures.md lines 12-13.

## Commands
- Subcommand list: cmd `fllm-rs completions bash` (command list) plus `ppl`. `--help` run for: run, pull, convert, quantize, inspect, formats, ppl, chat, bench, bench-report, devices, selftest-cpu, selftest-gpu, merge-lora, gguf-set, gguf-split, gguf-merge; list, ps, search, completions executed (rm and show print their usage).
- Defaults 127.0.0.1:8000, one model per process, API key: docs/configuration.md core options; docs/api-compat.md ("one model per process").

## APIs
- Routes: server.rs:200-252 and encoder_server.rs:503-521. `/api/pull`, `/api/delete` answer 501: docs/api-compat.md table and "Ollama" section.
- Anthropic x-api-key: docs/api-compat.md line 30. Stored responses: api-compat.md "Responses".
- 19 tool parsers: rust/engine/src/tool_parsers.rs:1729-1747 (match arms), docs/configuration.md. 7 reasoning parsers: docs/configuration.md `--reasoning-parser`, docs/server-api.md "Reasoning parsers".
- Structured output: docs/inference-features.md "Structured output" and "API details"; log probabilities: docs/api-compat.md "OpenAI token probabilities".
- Sampling controls: docs/configuration.md "Default sampling values".
- Not provided list: docs/api-compat.md "Not provided".

## Hardware and performance
- `--device auto` order: cmd `fllm-rs devices --help`, docs/device-selection.md section 2. `cuda:1` style: serve usage line.
- One device per process, no multi-GPU: `grep -rn "tensor.parallel|tensor-split|multi.gpu" rust/engine/src` finds nothing; serve takes a single --device. This rests on the absence of code, not on a doc statement.
- CUDA kernel list: docs/gpu-backends.md "Kernels" table and "Decode with many users"; FlashInfer: THIRD_PARTY.md:72-76; CUDA graphs opt-in: rust/engine/src/rust_decoder.rs:5579 (`FLLM_CUDA_GRAPH` must be set).
- 13 KV formats: cmd `fllm-rs --help`, docs/gpu-backends.md "KV cache formats"; wgpu f16/q8_0 only and the key-outlier warning: same section.
- gpu-layers / offload: docs/gpu-backends.md "Models larger than the card".
- Hybrid/state-space models disable prefix sharing and speculation: docs/architectures.md lines 75-76.
- Speculation: docs/configuration.md "Cache and speculation"; MTP is CPU only, DeepSeek-V2/V3 and GLM-4 MoE, greedy: docs/exl-formats.md "MTP / NextN draft source".
- CPU tiers: cmd `fllm-rs selftest-cpu --help`.

## Status (carried over from the previous README unless noted)
- CPU, CUDA, Ampere/Ada/Hopper, Vulkan, ROCm/SYCL/NPU/TPU rows: previous README text; hardware results cannot be re-verified here. The test counts "595 + 395" are unverified.
- Changed: CI row from .github/workflows/platforms.yml:53-55,117,121 (build, unit tests, selftest-cpu, `selftest-gpu --device cpu`; file header: "No job here has a GPU"); ubuntu-24.04-arm is in the matrix. The Metal row was softened for the same reason (the previous text said GPU tests run on Apple silicon in CI; the workflow does not run GPU tests, and conformance.yml with apple-metal needs a self-hosted runner).
- SVE opt-in and not run: docs/cpu-kernels.md "What is tested". Level definitions: docs/backends.md section 1.

## Measurements
- Colab table and the two caveat sentences: unchanged from the previous README.
- Laptop numbers: docs/gpu-backends.md "Measured" tables (Q4_K_M decode 96, prefill 3410 at 1864 tokens vs llama.cpp 85 / 3145; NVFP4 102 / 10070; gpt-oss-20b 44-45). The SGLang comparison (94 and 11,900) is carried over from the previous README and is not in docs/.

## Build and docs
- Features table: rust/engine/Cargo.toml:40-56; `rust-gpu` used by Makefile:6, docker/Dockerfile and release.yml.
- make test: Makefile `test` target. FLLM_CUDA_ARCHS: docs/gpu-backends.md "CUDA backend".
- SECURITY.md scope sentences: SECURITY.md lines 10-14. Issue templates: `ls .github/ISSUE_TEMPLATE`. licenses/: `ls licenses`.
- Link check: every relative link target in README.md exists (script: extract `](path)`, `test -e`); every docs/*.md, docs/user/*.md, compose files, SECURITY.md and THIRD_PARTY.md is linked from the index.
- German pages: docs/backends.md, docs/backends-verify.md, docs/model-coverage-vllm.md.
