# fLLM

fLLM is a local LLM inference engine written in Rust. It reads GGUF files, converts
Hugging Face checkpoints to GGUF, quantizes weights and serves an OpenAI-compatible HTTP API.
The workspace (`rust/engine`, `rust/gguf-probe`, `rust/checkpoint`) has no llama.cpp or GGML
dependency; the only non-Rust sources are the optional CUDA kernels in `rust/engine/cuda/`
(compiled with `nvcc` when the `cuda` feature is on). Python appears only in `scripts/` as development and
verification tooling; nothing at run time needs it.

## Quick start

```sh
./install.sh                                        # newest release binary (Linux, macOS; install.ps1 on Windows)
./run.sh run unsloth/Qwen3.5-9B-GGUF:Q4_K_M         # downloads the model once, then opens a chat in the terminal
```

`run` accepts a Hugging Face link (`https://huggingface.co/owner/repo`, `hf.co/owner/repo:Q4_K_M`),
a short `owner/repo[:QUANT]` name or a local `.gguf`. With Docker (NVIDIA GPU):
`docker run -it --rm --gpus all -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M`
(see [Installation](#installation) for the image and its tags).

## Status

fLLM is young. What has actually been run on hardware, and what has not:

| Target | State |
| --- | --- |
| CPU x86-64 (AVX2, AVX-VNNI, AVX-512) | Runs. Self test and real models on AVX2 and on an AMD Zen 5 (AVX-512); the AMX kernels have only been run in an emulator |
| NVIDIA CUDA, Turing (T4), Blackwell (RTX PRO 6000) | Runs. Self test 595 + 395 checks, real models (Llama, Qwen2.5, Qwen3.5 with DeltaNet, AWQ, GPTQ, NVFP4) |
| NVIDIA Ampere / Ada / Hopper | Builds; the same kernels as the tested cards, not yet measured |
| Apple Metal | Compiles; the GPU tests run on Apple silicon in CI, results are still being worked through |
| Vulkan | Ran on a few devices (NVIDIA, Intel); slower than CUDA, some operations are missing |
| AMD ROCm, Intel SYCL, NPUs, TPU, ARM NEON/SVE | Compile only (test level L0): not run on the hardware |

Measured on Google Colab with a single user, greedy decoding, over HTTP (before the latest performance work):

| Card, model | fLLM decode | fLLM prefill (355 / 1863 tokens) | vLLM on the same card |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell, Llama-3.1-8B NVFP4 | 194 tok/s | 24,900 / 36,100 tok/s | 195 tok/s, 20,400 / 54,500 tok/s |
| RTX PRO 6000 Blackwell, Qwen2.5-3B AWQ | 205 tok/s | 18,100 / 29,500 tok/s | 306 tok/s, 23,200 / 39,100 tok/s |
| RTX PRO 6000 Blackwell, Llama-3.1-8B Q4_K_M | 190 tok/s | 7,400 / 13,800 tok/s | not compared |
| Tesla T4, Llama-3.1-8B Q4_K_M | 46 tok/s | 910 / 980 tok/s | not compared |

So: on par with vLLM for NVFP4 decoding and faster on short prompts, behind it for long prompts and for small
quantized models. Closing that gap is the current work. Numbers are one run on a shared machine; treat differences of a few percent as noise.

## Contents

- [Quick start](#quick-start)
- [Status](#status)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Supported architectures](#supported-architectures)
- [Hardware notes](#hardware-notes)
- [User guides](#user-guides)
- [License](#license)

## Features

- **GGUF read/write.** Memory-mapped GGUF v2/v3 loading and GGUF writing. `fllm-rs formats`
  lists the 35 registered GGML storage types (classic Q, K, IQ, TQ and FP4 formats) with the
  decoder class each one has; registration is not a promise of inference support
  ([format matrix](docs/rust-formats-and-import.md)).
- **Checkpoint import.** `fllm-rs convert` reads safetensors (single file or weight-map shards)
  and restricted PyTorch ZIP state dictionaries (`.bin`, `.pt`, `.pth`) locally or through
  `hf://owner/repo` references, and streams GGUF directly. Config and tokenizer files must sit
  beside the weights.
- **Quantization.** `fllm-rs quantize` and `convert --outtype` write `q4_0`, `q4_1`, `q5_0`,
  `q5_1`, `q8_0`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m` and `q6_k`
  ([details](docs/quantization.md)).
- **GGUF tooling.** `inspect`, `gguf-set` (metadata), `gguf-split` / `gguf-merge` (shards),
  `merge-lora` (PEFT LoRA into the base weights).
- **OpenAI-compatible server** (`fllm-rs serve`): `/v1/chat/completions` (with `tools`,
  `tool_choice`, streaming and `reasoning_content`), `/v1/completions`, `/v1/embeddings`,
  `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/metrics`, plus a self-contained
  browser chat page at `/`
  ([API reference](docs/server-api.md)).
- **Compatibility endpoints:** Ollama (`/api/generate`, `/api/chat`, `/api/tags`,
  `/api/version`) and SGLang (`/generate`, `/get_model_info`, `/get_server_info`).
- **Constrained decoding:** JSON Schema, `json_object` and grammar strings (`gbnf`, `lark`,
  `regex`) run as sampler masks; tool arguments are validated against the declared schema.
- **Sampling:** temperature, `top_p`, `top_k`, `min_p`, `typical_p`,
  repetition/presence/frequency penalties, `repeat_last_n`, DRY, Mirostat 0/1/2, `seed` and
  `logit_bias`.
- **NVIDIA GPUs (CUDA).** Tensor-core and integer dot-product kernels for the K-quants, the classic Q formats,
  Q8_0, IQ4_NL/IQ4_XS, MXFP4 (gpt-oss), BF16/F16 and NVFP4 (native FP4 on Blackwell), flash attention for prompts,
  a quantized KV cache (f16, bf16, fp8, int8, q8_0, q5, q4, nvfp4; keys and values separately), many concurrent users, mixture-of-experts models with the experts on the device, and
  models larger than the card (attention and the leading layers on the GPU, the rest streamed over PCIe for prompts)
  ([CUDA notes](docs/gpu-backends.md#cuda-backend---features-cuda)).
- **More hardware (experimental).** Optional native backends for Vulkan, Metal, AMD ROCm/HIP, Intel GPUs (SYCL), Intel NPUs
  (OpenVINO) and Google TPUs, plus new CPU kernels for ARM (NEON, dotprod, i8mm, SVE) and x86 (AVX-512, AMX). They compile; Metal has been run on Apple silicon in CI, the others have not been run on the target hardware
  (test level L0). Which format and operation exists where, the known gaps and the commands to verify a backend on your machine:
  [docs/backends.md](docs/backends.md), [docs/backends-verify.md](docs/backends-verify.md).
- **Runtime features:** paged KV cache (f16, bf16, fp8 e4m3/e5m2, int8, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl, nvfp4 and f32 on the CPU and CUDA; keys and values may use different formats, see [KV cache formats](docs/gpu-backends.md#kv-cache-formats)), prefix cache, continuous batching with
  chunked prefill, load-time LoRA (`--lora`, `--lora-scaled`) and speculative decoding
  (`--speculative-lookup` prompt lookup or `--draft-model` with `--draft-tokens`).
- **Encoders and vision:** BERT GGUF files serve embeddings (CLS/mean/last pooling); a matching
  `--mmproj` projector enables image chat for supported multimodal checkpoints.
- **CLI clients:** `fllm-rs chat` (interactive chat against a running server) and
  `fllm-rs bench` (request/latency measurements).

## Installation

Prerequisites for a local build: Rust/Cargo, a C compiler, CMake and Git.

### Docker

```sh
FLLM_MODELS_DIR=/path/to/models FLLM_MODEL=model.gguf docker compose up -d
make build-image JOBS=2
```

The image contains only the `fllm-rs` binary. See [docs/docker.md](docs/docker.md).

**NVIDIA GPU (Linux and Windows with Docker Desktop/WSL 2).** The CUDA image needs only the NVIDIA driver and, on
Linux, the NVIDIA Container Toolkit; the model is downloaded from a Hugging Face link on first use and kept in a
Docker volume:

```sh
docker build -f docker/Dockerfile.cuda -t fllm:cuda .
docker run -it --rm --gpus all -v fllm:/home/fllm/.cache fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda
docker run -d --gpus all -v fllm:/home/fllm/.cache -p 8000:8000 fllm:cuda serve --model unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda --host 0.0.0.0 --port 8000
FLLM_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M docker compose -f compose.cuda.yml up -d
```

Step by step (prerequisites, cache, context size, quantization): [docs/quickstart-windows-linux.md](docs/quickstart-windows-linux.md).

### Release binary

```sh
./install.sh              # newest release
./install.sh v0.1.0       # a specific tag
```

`install.sh` detects OS and architecture, downloads the matching asset from this repository's
releases (`https://github.com/Marco-Lu2508/fLLM`), checks it against the published
SHA256 file and installs it as `./.local/bin/fllm-rs` (the path `run.sh` uses). If no matching
asset exists it prints why and falls back to building the checkout with `cargo`.
`./install.sh --no-verify` allows an install when the release publishes no checksum file (a
checksum that exists is always checked); see the script header for all options.
Windows users can run `install.ps1` or use WSL2/Docker.

### Build from source

```sh
make build-rust                       # cargo build --release --locked -p fllm-engine \
                                      #   --features rust-gpu --bin fllm-rs
./install.sh --build                  # build and install into ./.local/bin
```

`make test`, `make lint` and `make test-model MODEL=/models/model.gguf` run the Rust test
suite, `cargo fmt`/`clippy` and the Python verification harnesses.

## Quickstart

```sh
# 1) weights: pull a GGUF from the Hub, or convert a local Hugging Face checkpoint
./run.sh pull hf://owner/repo/model.gguf
./run.sh convert /models/hf-checkpoint /models/model-f16.gguf --outtype f16

# 2) quantize (optional; convert --outtype <type> does both steps in one run)
./run.sh quantize /models/model-f16.gguf /models/model-q4_k_m.gguf q4_k_m

# 3) serve
./run.sh serve --model /models/model-q4_k_m.gguf --host 127.0.0.1 --port 8000

# 4) talk to it
./run.sh chat --url http://127.0.0.1:8000
```

The same endpoint over HTTP:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

More detail (curl, streaming, tools, embeddings, BERT encoder mode):
[docs/getting-started.md](docs/getting-started.md), full route list:
[docs/server-api.md](docs/server-api.md), all `serve` flags:
[docs/configuration.md](docs/configuration.md).

## Supported architectures

Dense decoders (`llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `gemma2`, `gemma3_text`, `olmo`
family, `phi`, `phi3`, `cohere`, `starcoder2`, `gpt_neox`, `gpt2`, `opt`, `falcon`, `bloom`,
`mpt`, `glm`, `glm4`, and others), mixture-of-experts (`mixtral`, `qwen2_moe`, `qwen3_moe`,
`deepseek_v2`, `deepseek_v3`, `kimi_k2`, `gpt_oss`, `llama4`, and others) and recurrent/hybrid
models (`mamba`, `mamba2`, `qwen3_next`, `qwen3_5`, `olmo_hybrid`, `granitemoehybrid`, `bamba`,
`nemotron_h`) convert and load. BERT-style encoders serve embeddings.

The complete, verified list — including families that are explicitly *not* supported — is in
[docs/architectures.md](docs/architectures.md). Conversion refuses an unknown `model_type`
instead of guessing an architecture.

## Hardware notes

**CPU.** On x86-64 the kernels detect AVX2 + FMA + F16C at run time and use AVX-VNNI when the
CPU reports it; on other machines a portable scalar path runs. The CPU line of `fllm-rs
devices` shows the detected ISA (`avx-vnni`, `avx2` or `scalar`), so no build flag is needed
per machine. On Linux, `-C`/`--pin-cores` pins the process to a CPU affinity mask (`N` cores,
`0,2,3` list or `0xFF5` hex mask).

**NVIDIA GPU (CUDA).** `docker build -f docker/Dockerfile.cuda -t fllm:cuda .` builds an image that needs only
the NVIDIA driver on the host (`docker run --gpus all ...`; no CUDA toolkit); `--device auto` prefers CUDA when a
device is present. On an RTX PRO 3000 Blackwell Laptop (12 GB) Llama-3.1-8B Q4_K_M decodes at ~96 tokens/s and
prefills at ~3400 tokens/s (llama.cpp on the same card and client: 85 and 3150), the NVFP4 build at ~102 and
~10000 tokens/s (SGLang on the same card: 94 and 11900; CUTLASS GEMM as in SGLang and vLLM), gpt-oss-20b (MXFP4, more than the card holds) at ~45 and ~3000; the numbers for every format, for
mixture-of-experts models and for concurrent requests are in [docs/gpu-backends.md](docs/gpu-backends.md).

**Vulkan, Metal, DirectX 12.** The `rust-gpu` feature (enabled by `make`, `install.sh` and the Dockerfile) compiles
wgpu backends for Vulkan, Metal and DirectX 12. `--device vulkan|metal|dx12` forces a backend
(`--device auto` prefers CUDA → Metal → Vulkan → DX12 when a device is present, else CPU);
a software adapter is refused. Attention, RMSNorm, activation and KV-cache handling run on the
GPU with host fallback for LayerNorm, MoE and sampling, and hardware
validation covers only some devices — see [docs/inference-features.md](docs/inference-features.md).

## User guides

- [Commands: the `fllm-rs` CLI](docs/user/commands.md) — every subcommand, device selection, wgpu scope
- [Clients](docs/user/clients.md) — curl, verified SDKs, program integrations
- [Quants and formats](docs/user/quants-and-formats.md) — writers, AWQ/GPTQ, KV cache formats
- [Memory](docs/user/memory.md) — VRAM planner, sizing flags, CPU RAM notes
- [Troubleshooting](docs/user/troubleshooting.md) — device errors, HTTP errors, sampling surprises

**Native backends (experimental).** Features `vulkan`, `metal`, `hip`, `sycl`, `npu-openvino`, `tpu-pjrt` and `avx512` add backends next to the wgpu and CUDA ones; none is part of
the default build, and all are level L0 (compiled, not run on the hardware). The per-format matrix, the known gaps (for example Mamba on devices, ROCm kernels that were not ported,
missing Metal and Vulkan parts) and the verification scripts (`scripts/verify-<backend>.sh` / `.ps1`) are in [docs/backends.md](docs/backends.md) and
[docs/backends-verify.md](docs/backends-verify.md); the overview of features and prerequisites is in [docs/gpu-backends.md](docs/gpu-backends.md#native-backends-of-the-multi-backend-round).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party dependencies, their licenses and the
reference implementations used for comparisons are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
