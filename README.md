# fLLM

fLLM is an LLM inference engine written in Rust. It loads GGUF files (all 35 ggml storage types it
registers, including the K and IQ quants), imports Hugging Face checkpoints (safetensors and PyTorch,
including 4-bit AWQ and GPTQ, FP8, MXFP4 and EXL2/EXL3 weights) by converting them to GGUF, quantizes,
and serves OpenAI, Anthropic, Ollama, SGLang, TGI and llama.cpp style HTTP APIs from one port. It also
has a terminal chat (`fllm-rs run`), encoder and embedding models, image input for SigLIP/Idefics3
projectors, and CPU, CUDA and wgpu (Vulkan, Metal, DirectX 12) backends.

The Rust workspace (`rust/engine`, `rust/checkpoint`, `rust/gguf-probe`) has no llama.cpp or GGML
dependency. The only non-Rust sources are the GPU kernels (CUDA, HIP, Metal, SYCL, shaders), which are
compiled only when the matching Cargo feature is on; a few CUDA parts are adapted from vLLM (Marlin),
CUTLASS and FlashInfer, see [THIRD_PARTY.md](THIRD_PARTY.md). The binary does not need Python.

fLLM is young. Read [Status](#status) for what has actually run on which hardware.

## Quick start

**1. Release binary** (Linux x86_64/aarch64, macOS aarch64/x86_64):

```sh
git clone https://github.com/Marco-Lu2508/fLLM && cd fLLM
./install.sh                                          # newest release -> ./.local/bin/fllm-rs, checksum verified
./run.sh run unsloth/Qwen3.5-9B-GGUF:Q4_K_M           # downloads the model once, then chats in the terminal
```

The v0.1.0 release has `fllm-linux-x86_64-musl`, `fllm-linux-aarch64-musl`, `fllm-macos-aarch64` and
`fllm-macos-x86_64` archives, each with a `.sha256` file. It is marked as a pre-release. There are no
Windows assets yet; on Windows `install.ps1` builds from source with `cargo` until a release has them.
The release binaries are built with the `rust-gpu` feature (CPU, Vulkan, Metal, DirectX 12 through
wgpu), not with CUDA: for NVIDIA GPUs use the CUDA image or build with `--features cuda`.

**2. Docker** (images on ghcr.io: `latest` for CPU, linux/amd64 and linux/arm64; `cuda` for linux/amd64):

```sh
docker run -it --rm -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:latest run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
docker run -it --rm --gpus all -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda
docker run -d --gpus all -v fllm:/home/fllm/.cache -p 8000:8000 ghcr.io/marco-lu2508/fllm:cuda \
    serve --model unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda --host 0.0.0.0 --port 8000
```

The CUDA image needs only the NVIDIA driver (and the NVIDIA Container Toolkit on Linux), see
[docs/quickstart-windows-linux.md](docs/quickstart-windows-linux.md) and [docs/docker.md](docs/docker.md).

**3. From source** (Rust stable, a C compiler, Git; CUDA builds also need the CUDA toolkit):

```sh
cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs
target/release/fllm-rs run unsloth/Qwen3.5-9B-GGUF:Q4_K_M
```

`run` takes a Hugging Face link (`https://huggingface.co/owner/repo`, `hf.co/owner/repo:Q4_K_M`),
`owner/repo[:QUANT]`, `hf://owner/repo[:QUANT]`, a catalog name such as `qwen3.5:9b` (`fllm-rs search`
lists them) or a local `.gguf`. It starts a server on a free local port, opens a chat and stops the
server when you quit with Ctrl-D. Extra options go to `serve`, for example `--ctx 8192 --device cpu`.
A repository with several GGUF files needs the `:QUANT` suffix.

## What it can do

### Models and formats

`serve` and `run` load **GGUF** files. Everything else is converted to GGUF first with
`fllm-rs convert`; `serve` on a checkpoint directory without `.gguf` files stops and points to `convert`.

| Input | How it is handled | Notes |
| --- | --- | --- |
| GGUF v2/v3 with F32/F16/BF16, Q4_0 ... Q8_0, Q2_K ... Q6_K, IQ1 ... IQ4, TQ, MXFP4, NVFP4 | loaded directly | `fllm-rs formats` lists the 35 registered types; registration is not a promise of a fast kernel |
| safetensors (single or sharded) and PyTorch `.bin` / `.pt` / `.pth` | `convert` writes GGUF (`--outtype preserve`, `f16`, `f32` or a quant type) | local directory or file, or a Hub repository; `config.json` and tokenizer files are required |
| AWQ 4-bit (AutoAWQ GEMM) and GPTQ 4-bit, including act-order | `convert` keeps them 4-bit as fLLM-specific `Q4Z` / `Q4ZP` tensors; nothing is re-quantized | group sizes 32, 64, 128; other sizes, fp32 scales and stacked experts are expanded to F16 |
| FP8 checkpoints (`quant_method` fp8) | `convert`; stays FP8 with row scales for `--outtype preserve` | |
| gpt-oss MXFP4 | `convert` repacks the MXFP4 blocks | |
| EXL2 / EXL3 | `convert` expands to F16 (or re-quantizes with `--outtype`) | import only, not native EXL inference ([docs/exl-formats.md](docs/exl-formats.md)) |
| NVIDIA ModelOpt NVFP4 | runs as GGUF type 40 (llama.cpp layout) | `convert` does not read ModelOpt checkpoints directly; convert them with another tool or use an NVFP4 GGUF |

Where the weight formats run natively ([docs/gpu-backends.md](docs/gpu-backends.md),
[docs/backends.md](docs/backends.md), [docs/cpu-kernels.md](docs/cpu-kernels.md)):

| Device | Native kernels | Everything else |
| --- | --- | --- |
| CPU (x86-64, AArch64) | F32/F16/BF16, Q4_0 ... Q8_0, Q4_K ... Q6_K; on x86-64 also Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4 | rows decoded on demand (functional, slow): the other IQ/TQ types, NVFP4, Q4Z/Q4ZP |
| CUDA | all of the above, the IQ family, MXFP4, NVFP4 (FP4 tensor cores on Blackwell sm_12x, Marlin FP4 on other GPUs from Turing), AWQ/GPTQ `Q4Z`/`Q4ZP` through Marlin (Turing and newer, shapes in multiples of 128) | expanded at upload |
| wgpu (Vulkan, Metal, DX12) | F32/F16/BF16, Q4_0 ... Q8_0, Q4_K, Q5_K, Q6_K | expanded to F32 at upload |

### Architectures

Dense decoders (Llama, Mistral, Qwen2/3, Gemma 1-3, OLMo, Phi, Cohere, StarCoder2, GPT-NeoX, Falcon,
GLM, InternLM, Nemotron and more), mixture-of-experts (Mixtral, Qwen MoE, DeepSeek V2/V3 and Kimi K2
with multi-head latent attention, gpt-oss, Llama 4, DBRX, PhiMoE and more), and recurrent or hybrid
models (Mamba 1/2, Qwen3-Next, Qwen3.5/3.6 with gated DeltaNet in HF and llama.cpp GGUF layouts,
Nemotron-H, Granite hybrid, LFM2). BERT, RoBERTa/XLM-R and ModernBERT serve embeddings and rerank/classify.
The full list, with the families that are **not** supported (for example Gemma 4, Gemma 3n, Jamba,
Falcon-H1, DeepSeek V3.2/V4), is in [docs/architectures.md](docs/architectures.md). Conversion refuses
an unknown `model_type`.

"Verified" in these tables means: a tiny random checkpoint of the architecture gives the same logits as
`transformers`. Most real checkpoints were not available for testing. The coverage against the open
models shown on artificialanalysis.ai (2026-10-13 snapshot: 43 models, 25 verified, 2 partly, 16 not
supported) is in [docs/model-coverage.md](docs/model-coverage.md); the checklist against vLLM and SGLang
architecture names (German) is in [docs/model-coverage-vllm.md](docs/model-coverage-vllm.md).

Image input: `--mmproj` loads a SigLIP/Idefics3 projector (SmolVLM). Other vision families, audio and
speech are not supported; for multimodal checkpoints only the text model is converted.

### Run and serve

| Command | Purpose |
| --- | --- |
| `run MODEL` | download if needed, serve on a free port, chat in the terminal |
| `serve --model MODEL` | HTTP server (the word `serve` is optional) |
| `pull`, `list`, `rm`, `ps`, `show`, `search` | model cache and catalog: download, list, remove, running servers, details, catalog names |
| `chat`, `bench`, `bench-report`, `ppl` | terminal client for a running server, load benchmark and comparison, perplexity |
| `convert`, `quantize`, `inspect`, `formats` | checkpoint to GGUF, quantizer, file inspection, type list |
| `merge-lora`, `gguf-set`, `gguf-split`, `gguf-merge` | LoRA merge, GGUF metadata and shard tools |
| `devices`, `selftest-cpu`, `selftest-gpu`, `completions` | device inventory, kernel self tests, shell completion |

`fllm-rs <command> --help` prints the usage of each command; [docs/user/commands.md](docs/user/commands.md)
and [docs/configuration.md](docs/configuration.md) cover every flag and environment variable.
The server listens on `127.0.0.1:8000` by default, serves one model per process, and `--api-key` (or
`FLLM_API_KEY`) turns on bearer-token checks.

### APIs

One port, several dialects, all translated onto the same chat pipeline
([docs/api-compat.md](docs/api-compat.md), [docs/server-api.md](docs/server-api.md)):

| Dialect | Routes |
| --- | --- |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses` (with stored responses), `/v1/files` and `/v1/batches` (JSONL batch jobs) |
| Anthropic | `/v1/messages` (streaming, tools, images, thinking), `/v1/messages/count_tokens`; `x-api-key` header accepted |
| Ollama | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` (`/api/pull` and `/api/delete` answer 501) |
| SGLang, TGI | `/generate`, `/generate_stream`, `/info`, `/get_model_info`, `/get_server_info` |
| llama.cpp | `/completion`, `/infill`, `/tokenize`, `/detokenize`, `/embedding`, `/props`, `/slots`, `/apply-template` |
| Encoders | `/v1/embeddings`, `/v1/rerank`, `/rerank`, `/v1/score`, `/classify`, `/pooling` |
| Other | `/` built-in web chat page, `/health`, `/metrics` (Prometheus) |

Common to the chat routes: streaming (SSE); tool calling with 19 tool-call parsers (Hermes/Qwen, Qwen3-Coder,
GLM, Llama 3, Mistral, DeepSeek, Kimi, gpt-oss and others; `--tool-parser auto` picks from the chat
template); reasoning split into `reasoning_content` with 7 parsers; structured output (`response_format`
JSON schema and `json_object`, GBNF, Lark and regex grammars); log probabilities; and sampling controls
(temperature, top-p/k, min-p, typical-p, penalties, DRY, Mirostat, seed, logit bias).
Not provided: audio, image generation, realtime, moderation, Assistants, fine-tuning and runtime LoRA switching.

### Quantization and conversion

- `fllm-rs quantize IN.gguf OUT.gguf TYPE` and `convert --outtype TYPE` write `q4_0`, `q4_1`, `q5_0`, `q5_1`,
  `q8_0`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m` and `q6_k` with llama.cpp's per-tensor mixes. The classic
  types are byte-identical to `llama-quantize`; K-quants differ only in rare float-summation ties.
- There is no writer for IQ, TQ or FP4 types (they can be read, not produced), and `quantize` leaves
  already quantized tensors unchanged: start from an F16/F32 file to change a quantization.
- `merge-lora` merges a PEFT adapter into a GGUF file; `--lora` / `--lora-scaled` merge one at load time.
- `gguf-set`, `gguf-split` and `gguf-merge` edit metadata and shards.

Details: [docs/quantization.md](docs/quantization.md), [docs/user/quants-and-formats.md](docs/user/quants-and-formats.md).

### Hardware and performance

- **Backends.** CPU (runtime-selected AVX2, AVX-VNNI, AVX-512, NEON dotprod/i8mm kernels), CUDA, wgpu
  (Vulkan, Metal, DirectX 12), and optional native Vulkan, Metal, ROCm/HIP, SYCL, Intel NPU (OpenVINO) and
  TPU backends. `--device auto` takes the first GPU of CUDA, ROCm, Metal, Vulkan, DX12, SYCL, else the
  CPU; names like `cuda:1` pick a device. One server process uses one device: there is no
  multi-GPU splitting of a single model. See [docs/device-selection.md](docs/device-selection.md).
- **CUDA.** int8 dot-product and tensor-core GEMMs for quantized weights, Marlin for AWQ/GPTQ/FP4,
  CUTLASS block-scaled NVFP4 GEMM on Blackwell, FlashInfer prompt attention (f16 KV cache), split-KV decode
  attention, argmax sampling on the device, and CUDA graphs for decode of dense models (opt-in with
  `FLLM_CUDA_GRAPH=1`).
- **Memory.** Paged KV cache with 13 formats (`f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `int8`, `q8_0`, `q5_0`,
  `q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `nvfp4`, `f32`); keys and values can differ (`--cache-type-k`,
  `--cache-type-v`). wgpu devices keep `f16` and `q8_0` only, and 4-bit and 5-bit keys can break models
  with large key outliers. For models larger than the card, `--gpu-layers` keeps attention on the GPU and
  streams feed-forward weights and MoE experts from host memory ([docs/user/memory.md](docs/user/memory.md)).
- **Scheduling.** Continuous batching, chunked prefill, prefix cache (hybrid and state-space models
  switch off prefix sharing and speculation).
- **Speculative decoding.** Prompt lookup (`--speculative-lookup`), a draft model (`--draft-model`) and
  the multi-token-prediction heads of DeepSeek-V2/V3 and GLM-4 MoE (`--speculative mtp`, CPU only); greedy
  requests only.

### Tools and clients

`fllm-rs chat` and `bench` talk to any running OpenAI-compatible server. [docs/user/clients.md](docs/user/clients.md)
covers curl, the OpenAI, Anthropic and Ollama SDKs and program integrations.

## Status

What has actually been run on hardware, and what has not:

| Target | State |
| --- | --- |
| CPU x86-64 (AVX2, AVX-VNNI, AVX-512) | Runs. Self test and real models on AVX2 and on an AMD Zen 5 (AVX-512); the AMX kernels have only been run in an emulator |
| NVIDIA CUDA, Turing (T4), Blackwell (RTX PRO 6000) | Runs. Self test 595 + 395 checks, real models (Llama, Qwen2.5, Qwen3.5 with DeltaNet, AWQ, GPTQ, NVFP4) |
| NVIDIA Ampere / Ada / Hopper | Builds; the same kernels as the tested cards, not yet measured |
| Linux aarch64, macOS (Apple silicon), Windows (MSVC) | Build, unit tests and the CPU self tests run in CI on GitHub-hosted runners (the Linux and Windows runners have no GPU). The SVE kernel is opt-in and was not run. Windows release assets are not published yet |
| Apple Metal | The native Metal kernels (attention for f16 and q8_0 KV caches in decode and prompt, KV-cache writes and more) run their GPU checks on the Apple-silicon CI runner and match the CPU reference to about 1e-7. A whole model has not been run on Metal yet |
| Vulkan | Ran on a few devices (NVIDIA, Intel); slower than CUDA, some operations are missing |
| AMD ROCm, Intel SYCL, NPUs, TPU | Compile only (test level L0): not run on the hardware |

The test levels (L0 compiles, L1 self test passes on the device, L2 same greedy text as the CPU, L3 speed
measured) are defined in [docs/backends.md](docs/backends.md) (German).

## Measurements

Measured on Google Colab with a single user, greedy decoding, over HTTP (before the latest performance work):

| Card, model | fLLM decode | fLLM prefill (355 / 1863 tokens) | vLLM on the same card |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell, Llama-3.1-8B NVFP4 | 194 tok/s | 24,900 / 36,100 tok/s | 195 tok/s, 20,400 / 54,500 tok/s |
| RTX PRO 6000 Blackwell, Qwen2.5-3B AWQ | 205 tok/s | 18,100 / 29,500 tok/s | 306 tok/s, 23,200 / 39,100 tok/s |
| RTX PRO 6000 Blackwell, Llama-3.1-8B Q4_K_M | 190 tok/s | 7,400 / 13,800 tok/s | not compared |
| Tesla T4, Llama-3.1-8B Q4_K_M | 46 tok/s | 910 / 980 tok/s | not compared |

So: on par with vLLM for NVFP4 decoding and faster on short prompts, behind it for long prompts and for small
quantized models. Closing that gap is the current work. Numbers are one run on a shared machine; treat differences of a few percent as noise.

On an RTX PRO 3000 Blackwell Laptop (12 GB, power-limited), one request at a time, Llama-3.1-8B Q4_K_M decodes at
96 tok/s and prefills a 1,863-token prompt at 3,410 tok/s (llama.cpp on the same card and client: 85 and 3,145). The
NVFP4 build reaches 102 tok/s and 10,070 tok/s (SGLang on the same card: 94 and 11,900, so fLLM is behind for long
prompts there too). gpt-oss-20b (MXFP4, larger than the card) decodes at about 45 tok/s. More formats, MoE models and
concurrent requests are in [docs/gpu-backends.md](docs/gpu-backends.md).

## Documentation

Some pages are in German or contain development notes; the code and `--help` are authoritative.

**Start here**
- [docs/getting-started.md](docs/getting-started.md): install, get weights, quantize, serve, curl, chat, embeddings
- [docs/quickstart-windows-linux.md](docs/quickstart-windows-linux.md): Docker with an NVIDIA GPU, step by step
- [docs/user/commands.md](docs/user/commands.md): every `fllm-rs` subcommand
- [docs/configuration.md](docs/configuration.md): every `serve` flag and environment variable
- [docs/user/clients.md](docs/user/clients.md), [docs/user/memory.md](docs/user/memory.md), [docs/user/troubleshooting.md](docs/user/troubleshooting.md): clients, memory planning, error messages

**Models and formats**
- [docs/architectures.md](docs/architectures.md): supported and unsupported architectures, verification method
- [docs/model-coverage.md](docs/model-coverage.md), [docs/model-coverage-vllm.md](docs/model-coverage-vllm.md): coverage against popular open models and against vLLM/SGLang (German)
- [docs/adding-a-model.md](docs/adding-a-model.md): how to add an architecture
- [docs/rust-formats-and-import.md](docs/rust-formats-and-import.md): ggml storage types and checkpoint containers
- [docs/quantization.md](docs/quantization.md), [docs/user/quants-and-formats.md](docs/user/quants-and-formats.md): quantizer, GGUF tools, KV cache formats
- [docs/awq-gptq-marlin.md](docs/awq-gptq-marlin.md): AWQ/GPTQ and NVFP4 through Marlin
- [docs/exl-formats.md](docs/exl-formats.md): EXL2/EXL3 import, MTP draft source

**Server and APIs**
- [docs/server-api.md](docs/server-api.md): request and response fields, tool and reasoning parsers (its route overview predates the Anthropic, Responses, batch and llama.cpp routes, see api-compat)
- [docs/api-compat.md](docs/api-compat.md): all dialects, files and batches, encoder tasks
- [docs/inference-features.md](docs/inference-features.md): sampling, structured output, LoRA, speculation, vision

**Hardware**
- [docs/device-selection.md](docs/device-selection.md): `--device`, `devices`, `auto`, self test
- [docs/gpu-backends.md](docs/gpu-backends.md): backend design, CUDA kernels, offload, KV cache formats, measurements
- [docs/backends.md](docs/backends.md), [docs/backends-verify.md](docs/backends-verify.md): hardware by format matrix, verification steps (German)
- [docs/cpu-kernels.md](docs/cpu-kernels.md), [docs/cpu-arm.md](docs/cpu-arm.md), [docs/cpu-avx512-amx.md](docs/cpu-avx512-amx.md): CPU kernels
- [docs/cuda-qgemm4k.md](docs/cuda-qgemm4k.md): Q4_K prompt GEMM
- [docs/windows.md](docs/windows.md): Windows builds

**Deployment and project**
- [docs/docker.md](docs/docker.md), [compose.yml](compose.yml), [compose.cuda.yml](compose.cuda.yml): images and compose files
- [docs/ci.md](docs/ci.md): workflows, release assets, image tags
- [SECURITY.md](SECURITY.md), [THIRD_PARTY.md](THIRD_PARTY.md), [docs/release-license-report.md](docs/release-license-report.md): security policy, third-party code and licenses

## Building from source

```sh
cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs           # CPU + wgpu
cargo build --release --locked -p fllm-engine --features "rust-gpu,cuda" --bin fllm-rs    # plus CUDA, needs nvcc
FLLM_CUDA_ARCHS=120 cargo build --release --locked -p fllm-engine --features cuda          # one GPU generation, faster
./install.sh --build                                                                       # build and install into ./.local/bin
```

| Cargo feature | Adds |
| --- | --- |
| (none) | CPU only |
| `rust-gpu` | wgpu backends: Vulkan, Metal, DirectX 12 (what `make`, `install.sh`, the release binaries and the CPU image use) |
| `cuda` | NVIDIA kernels, compiled with `nvcc` (architectures from `FLLM_CUDA_ARCHS`) |
| `hip`, `sycl`, `vulkan`, `metal`, `npu-openvino`, `tpu-pjrt`, `avx512` | optional native backends and kernels, test level L0 ([docs/backends.md](docs/backends.md)) |

`make test` runs the Rust tests. On Windows see [docs/windows.md](docs/windows.md).

## Contributing and security

Bug reports go to the GitHub issue tracker (templates in `.github/ISSUE_TEMPLATE`). To add a model
architecture see [docs/adding-a-model.md](docs/adding-a-model.md). Please report security problems
privately as described in [SECURITY.md](SECURITY.md). The server is meant for your own machine or a
trusted network: if you expose it, set an API key and put a reverse proxy in front.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party dependencies, their licenses and the reference
implementations used for comparisons are listed in [THIRD_PARTY.md](THIRD_PARTY.md); license texts of
adapted code are in [licenses/](licenses/).
