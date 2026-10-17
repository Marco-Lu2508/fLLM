# Documentation index

Start with the [README](../README.md) for what fLLM is and its current status. This page groups all documentation by topic.

## Getting started

* [Getting started](getting-started.md): first model, first requests with curl
* [Quick start for Windows and Linux](quickstart-windows-linux.md), [Windows](windows.md), [Docker](docker.md)
* [Commands: the `fllm-rs` CLI](user/commands.md)
* [Troubleshooting](user/troubleshooting.md)

## Using the server

* [HTTP API reference](server-api.md) and [compatibility with other APIs](api-compat.md)
* [Clients: SDKs, editors, chat front-ends](user/clients.md)
* [Configuration: flags and environment variables](configuration.md)
* [Inference features: sampling, structured output, tools, LoRA, speculation, vision](inference-features.md)
* [Memory planning: VRAM and RAM](user/memory.md)

## Models and formats

* [Supported architectures](architectures.md)
* [Model coverage](model-coverage.md) and [coverage against vLLM's list](model-coverage-vllm.md)
* [Quantization](quantization.md), [quants and formats for users](user/quants-and-formats.md)
* [AWQ and GPTQ through Marlin](awq-gptq-marlin.md), [EXL2/EXL3 formats](exl-formats.md)
* [Rust formats and checkpoint import](rust-formats-and-import.md)

## Hardware and performance

* [Device selection](device-selection.md)
* [GPU backends (CUDA, ROCm, Vulkan, Metal, ...)](gpu-backends.md) and [backend status matrix](backends.md)
* [CPU kernels](cpu-kernels.md), [Arm](cpu-arm.md), [AVX-512 and AMX](cpu-avx512-amx.md)
* [CUDA Q4_K prompt GEMM](cuda-qgemm4k.md), [performance notes](performance-notes.md)
* [Benchmarking](benchmarking.md)

## Verification and test levels

* [Test levels L0 to L3](test-levels.md)
* [Backend verification procedures](backends-verify.md)
* The proof scripts: `scripts/README.md`
* [CI and releases](ci.md)

## Contributing and internals

* [Contributing](../CONTRIBUTING.md), [Code of Conduct](../CODE_OF_CONDUCT.md), [Security policy](../SECURITY.md)
* [Architecture overview](architecture.md)
* [Adding a model](adding-a-model.md), [adding a backend](adding-a-backend.md)
* [Roadmap](roadmap.md), [Changelog](../CHANGELOG.md)
* [Research notes](research/README.md) (design studies, not user documentation)
* [Third-party code and licences](../THIRD_PARTY.md)
