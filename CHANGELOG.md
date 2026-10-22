# Changelog

All notable changes to fLLM are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project aims to follow [Semantic Versioning](https://semver.org/) once it leaves the 0.x series (0.x releases may change
behaviour between versions).

## [Unreleased]

### Added
- Studio, the built-in browser app (`fllm-rs studio`, or `GET /` of any server; [docs/user/studio.md](docs/user/studio.md)): chat with branching message history, Markdown and code highlighting, file and image attachments, presets and chat management; a coding agent that reads, searches and changes files and runs commands inside one folder, with diffs, permission questions (ask, auto-edit, plan), a task list and `@file`; a model manager with Hugging Face search, download progress and switching the loaded model (the server restarts itself); settings, API keys, hardware and tool pages. Backed by `/api/studio/*`. Changes through these routes are accepted without an API key only from the same machine, from the page of this server, addressed by `localhost` or an IP address; the agent tools never leave the machine, also with a key, and `FLLM_NO_AGENT=1` switches them off. Generated keys come from the operating system random source. `fllm-rs studio` also starts without a model and lets you pick and download the first one in the page. A desktop app (Tauri, `desktop/`) shows the Studio in its own window with the server built in; installers for Windows and Linux are built by the `desktop` workflow ([docs/user/desktop.md](docs/user/desktop.md)).
- `install.sh --variant auto|rocm`: the installer picks cuda, rocm, gnu (Vulkan) or cpu from the hardware; `release.yml` builds the optional `fllm-linux-x86_64-rocm.tar.gz` (HIP kernels, untested on AMD GPUs).

### Fixed
- Vulkan on integrated GPUs with a small dedicated heap (AMD APUs): a full device-local heap falls back to host-visible memory instead of failing with "Out of memory".
- Vulkan: conversion to binary16 rounds to nearest even on every driver (AMD rounded toward zero).
- `serve` plans the memory of an integrated GPU with the free system memory.

## [0.1.1] - 2026-10-15

### Added
- Checkpoints are converted to GGUF on first use: `run`, `serve` and `pull` accept a Hugging Face repository or a local directory with
  safetensors / PyTorch weights and keep the converted file in the model cache.
- `convert` imports NVIDIA ModelOpt NVFP4 checkpoints, compressed-tensors int4/int8 and block-wise FP8 checkpoints.
- `fllm-rs selftest-cpu` with kernel tiers (`--tier`, `--tiers`, `--full`), Q4Z/Q4ZP SIMD kernels and a per-layer dump.
- NVFP4 activation scale (the checkpoint's static `input_scale`), round-to-even E2M1 conversion and a tile table; a GPU perplexity and
  quality script (`scripts/colab/nvfp4_quality.py`).
- Colab scripts for A/B measurements of the new defaults (`scripts/colab/ab_matrix.py`, `cpu_confirm.py`).
- `SECURITY.md`.

### Changed
- CUDA decode steps are recorded as CUDA graphs and use fused decode kernels by default; the prompt pass no longer does per-layer host
  work and uses fused add+norm and FP16 copies. Each has a one-line switch back (see `docs/configuration.md`).
- The `avx512` Cargo feature is on by default; the AVX-512 and AMX kernels are chosen at run time from CPUID, so the binary still runs
  on any x86-64 CPU.
- README rewritten with the real feature set, formats, APIs, status table and a documentation index.

### Fixed
- NVFP4 self test w4a4 failures are deterministic and readable; weight upload ordering; IEEE division in the quantizer.
- Waiting for a concurrent conversion of the same checkpoint; compressed-tensors FP8 configs without `config_groups`.
- Windows: the developer-shell launcher takes `amd64` as architecture.

## [0.1.0] - 2026-10-14

### Added
- First public release of the pure-Rust inference engine: GGUF loading, CPU kernels (AVX2, AVX-VNNI, AVX-512, NEON), CUDA backend,
  wgpu GPU path (Vulkan, Metal, DirectX 12) and native Vulkan, Metal, HIP, SYCL, NPU and TPU backends in various states of
  verification (see the Status table in the README).
- OpenAI-compatible server with Anthropic, Ollama, llama.cpp, SGLang and TGI compatible routes, tool calling, structured output,
  speculative decoding, LoRA merge at load time, embeddings and encoder (BERT-style) models, SmolVLM image input.
- Quantization formats of the GGUF family (including K-quants, IQ formats, MXFP4, NVFP4) plus the Q4Z formats for AWQ and GPTQ checkpoints.
- Command line: `run`, `serve`, `chat`, `bench`, `convert`, `pull`, `devices`, `formats`, `inspect`, `selftest-cpu`, `selftest-gpu`.
- GitHub Actions for checks, platform builds, release binaries and container images; install scripts that resolve GitHub releases.
- Docker images (CPU for amd64 and arm64, CUDA for amd64).

### Fixed
- Misaligned vector loads in the AVX-512 weight-panel tile (Q4_1/Q5_1 prompts).
- Metal kernels `fllm_attn_merge_f32` and `fllm_kv_write_q8` thread-index declarations.
- Vulkan: clippy `unnecessary_cast` where `c_char` is `u8` (aarch64 Linux).

[Unreleased]: https://github.com/Marco-Lu2508/fLLM/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Marco-Lu2508/fLLM/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Marco-Lu2508/fLLM/releases/tag/v0.1.0
