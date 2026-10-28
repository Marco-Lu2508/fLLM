# fLLM

fLLM is an LLM inference engine written in Rust, with a desktop app (chat, coding agent and model manager)
on top. It loads GGUF files (all 35 ggml storage types it registers, including the K and IQ quants) and
Hugging Face checkpoints (safetensors and PyTorch, including 4-bit AWQ and GPTQ, FP8, MXFP4 and EXL2/EXL3
weights): a checkpoint is translated into the GGUF tensor layout in memory while it loads, with no GGUF file written. It quantizes, and serves OpenAI, Anthropic, Ollama, SGLang, TGI and llama.cpp style HTTP APIs from one port. It also
has a terminal chat (`fllm-rs run`), encoder and embedding models, image input for SigLIP/Idefics3
projectors, and CPU, CUDA and wgpu (Vulkan, Metal, DirectX 12) backends.

The Rust workspace (`rust/engine`, `rust/checkpoint`, `rust/gguf-probe`) has no llama.cpp or GGML
dependency. The only non-Rust sources are the GPU kernels (CUDA, HIP, Metal, SYCL, shaders), which are
compiled only when the matching Cargo feature is on; a few CUDA parts are adapted from vLLM (Marlin),
CUTLASS and FlashInfer, see [THIRD_PARTY.md](THIRD_PARTY.md). The binary does not need Python.

fLLM is young. Read [Status](#status) for what has actually run on which hardware.

## Quick start

**1. Desktop app** (Windows and Linux): download the installer from the
[latest release](https://github.com/Marco-Lu2508/fLLM/releases) (`fLLM_<version>_x64-setup.exe` or `.msi` on Windows;
`.deb`, `.rpm` or `.AppImage` on Linux), install it and start it. On the first start the app shows your hardware and
suggests models that fit into its memory; pick one and chat. No terminal needed. The installers are not signed yet,
so Windows may show a warning ([docs/user/desktop.md](docs/user/desktop.md)). macOS is not packaged yet.

**2. Release binary** (command line; Linux x86_64/aarch64, macOS aarch64/x86_64, Windows x86_64):

```sh
git clone https://github.com/Marco-Lu2508/fLLM && cd fLLM
curl -fsSL https://raw.githubusercontent.com/Marco-Lu2508/fLLM/main/install.sh | sh        # Linux, macOS -> ~/.local/bin/fllm-rs
./run.sh run unsloth/Qwen3.5-9B-GGUF:Q4_K_M           # downloads the model once, then chats in the terminal
```

Each release has archives for Linux (musl and gnu, x86_64 and aarch64), macOS (aarch64 and x86_64) and Windows
(`fllm-windows-x86_64-msvc.zip`), each with a `.sha256` file; 0.x releases are marked as pre-releases. The archives
are built with the `rust-gpu` feature (CPU, Vulkan, Metal, DirectX 12 through wgpu); the Linux `gnu` and desktop
builds also contain the native Vulkan backend. CUDA and ROCm archives are optional release assets, listed on the
release page when their build succeeded; otherwise use the CUDA image or build with `--features cuda`.

**3. Docker** (images on ghcr.io: `latest` for CPU, linux/amd64 and linux/arm64; `cuda` for linux/amd64):

```sh
docker run -it --rm -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:latest run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
docker run -it --rm --gpus all -v fllm:/home/fllm/.cache ghcr.io/marco-lu2508/fllm:cuda run unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda
docker run -d --gpus all -v fllm:/home/fllm/.cache -p 8000:8000 ghcr.io/marco-lu2508/fllm:cuda \
    serve --model unsloth/Qwen3.5-9B-GGUF:Q4_K_M --device cuda --host 0.0.0.0 --port 8000
```

The CUDA image needs only the NVIDIA driver (and the NVIDIA Container Toolkit on Linux), see
[docs/quickstart-windows-linux.md](docs/quickstart-windows-linux.md) and [docs/docker.md](docs/docker.md).

**4. From source** (Rust stable, a C compiler, Git; CUDA builds also need the CUDA toolkit):

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

`serve` and `run` load **GGUF** files directly. A **checkpoint** (safetensors or PyTorch) needs no manual step either:
a local directory or file, or a downloaded Hugging Face repository, with safetensors or PyTorch weights is translated into the GGUF tensor layout in memory while it loads, with no GGUF file written (`--no-direct-load` converts to a cached GGUF instead); a Hugging Face
repository is converted automatically on first use: `fllm-rs run owner/repo` on a repository without
GGUF files downloads the weights, checks the architecture from `config.json` first, converts once and caches the GGUF
next to the other models (`fllm-rs list` shows it, `fllm-rs rm owner/repo` removes it, `rm owner/repo --converted` keeps
the downloaded checkpoint and removes only the GGUF; the cache is `FLLM_CACHE`, else `$HF_HOME/hub`, a local directory is
converted into `fllm-local/` there). The second start finds the file again. `fllm-rs convert` stays available for an
explicit conversion. What was verified: the conversion of every format below with tiny synthetic checkpoints, run on the
CPU against a dequantized reference ([docs/rust-formats-and-import.md](docs/rust-formats-and-import.md)); real multi-GB
checkpoints and GPU runs of converted files were **not** tested.

| Input | How it is handled | Notes |
| --- | --- | --- |
| GGUF v2/v3 with F32/F16/BF16, Q4_0 ... Q8_0, Q2_K ... Q6_K, IQ1 ... IQ4, TQ, MXFP4, NVFP4 | loaded directly | `fllm-rs formats` lists the 35 registered types; registration is not a promise of a fast kernel |
| safetensors (single or sharded) and PyTorch `.bin` / `.pt` / `.pth` | checkpoints of unquantized, AWQ, GPTQ, FP8, NVFP4, compressed-tensors, MXFP4, bitsandbytes, MLX, EXL2 and EXL3 weights, in safetensors or PyTorch ZIP files, are loaded directly by `run` / `serve` (no GGUF file; `--no-direct-load` converts instead; proven on tiny synthetic models, not yet on large real ones); `convert` writes a file you name | Hub repository, local directory or file; `config.json` and `tokenizer.json` are required. Unquantized bf16/f16/f32 stay as they are (`--outtype f16`/`f32` to change) or are re-encoded with `--quant q4_k_m`, `q8_0`, ... |
| bitsandbytes 4-bit (NF4, FP4, double quantization) and LLM.int8() | converted automatically or loaded directly; the packed codes stay 4-bit / 8-bit as fLLM-specific `NF4B64/128`, `FP4B64/128` and `INT8R` tensors, nothing is expanded when the rows are a multiple of 64 values | CPU kernels (AVX2 and portable) are tested; the CUDA kernels compile for sm_80/89/90/120 and have not run on a GPU yet; Vulkan, Metal and wgpu expand the weights to 16-bit on upload ([docs/rust-formats-and-import.md](docs/rust-formats-and-import.md)) |
| AWQ 4-bit (AutoAWQ GEMM) and GPTQ 4-bit, including act-order | converted automatically, kept 4-bit as fLLM-specific `Q4Z` / `Q4ZP` tensors; nothing is re-quantized | group sizes 32, 64, 128; other sizes, fp32 scales and stacked experts are expanded to F16 |
| MLX affine quantization (mlx-community and mlx-lm exports: 2, 3, 4, 5, 6 and 8 bits, `quantization` in `config.json`), embeddings and head included | loaded directly or converted automatically, kept packed as fLLM-specific `MLXA` tensors: the MLX words, scales and biases as they are (4.5 bits per weight for 4-bit with group 64, like the checkpoint); nothing is re-quantized | groups that are a multiple of 32 (any size up to 8160), scales and biases exact in f16 or bfloat16 (the usual case); 7-bit weights, other group sizes and float32 scales without an exact 16-bit form are expanded to F16 with a warning; the `mxfp4` / `nvfp4` modes of MLX and stacked experts are refused |
| FP8 checkpoints (`quant_method` fp8, ModelOpt FP8, compressed-tensors float) | converted automatically and kept 8-bit: per-tensor or per-row scales as `FP8M` (f16 scale per row), block-wise `weight_scale_inv` (DeepSeek-V3, Qwen3-FP8, GLM-FP8: 128 x 128 tiles, any other power-of-two tile, f32 / bf16 / f16 / UE8M0 scales, experts included) as fLLM-specific `FP8B` tensors: the E4M3 codes as they are plus one f32 scale per tile, 1.0002 bytes per weight | the CPU multiplies `FP8B` natively (AVX2+FMA / NEON / portable, tests pass); NVIDIA GPUs from sm_60 read it in place (FP32 dequantizing GEMV for decode, FP16 tensor cores for prompts from sm_75, FP8 tensor cores for prompts from sm_89; the kernels compile for sm_75/80/89/90/120 but **have not run on a GPU**); Vulkan, wgpu, Metal, SYCL, TPU and AMD HIP expand `FP8B` to F16 at upload (2 bytes per weight) with a notice; tile shapes that are not powers of two, a row permutation that cuts through tile rows and the direct load of the codes (they are built in memory, not mapped) are the limits; no tensor-parallel splitting ([docs/rust-formats-and-import.md](docs/rust-formats-and-import.md)) |
| NVIDIA ModelOpt NVFP4 (`quant_algo` NVFP4, `hf_quant_config.json` or `quantization_config`) | converted automatically to GGUF type 40 (the layout of llama.cpp, no llama.cpp needed): block scales copied, `weight_scale_2` stored as `<name>.scale` | dense models; excluded modules (for example `lm_head`) stay bf16/f16; FP8 KV-cache scales are ignored; NVFP4 mixture-of-experts checkpoints are expanded to F16 |
| compressed-tensors int4 / int8 (`pack-quantized`, `int-quantized`, symmetric) | converted automatically: int4 groups of 32/64/128 become `Q4Z`, int8 becomes `Q8_0` | asymmetric, act-order, other bit widths, sparse and compressed-tensors NVFP4 are refused with a message |
| gpt-oss MXFP4 | repacks the MXFP4 blocks (loaded directly, or written by `convert`) | |
| EXL2 / EXL3 | stays packed: the codes, scales, group widths and permutation (EXL2) or trellis tiles and Hadamard signs (EXL3) are kept in fLLM's `EXL2` / `EXL3` GGUF types, loaded directly or written by `convert`; `--dtype f16` expands | runs natively on the CPU and on NVIDIA GPUs (CUDA-core kernels, **not yet run on a GPU**); other backends expand to F16 at upload; stacked MoE experts and fractional EXL3 are expanded or refused ([docs/exl-formats.md](docs/exl-formats.md)) |

Where the weight formats run natively ([docs/gpu-backends.md](docs/gpu-backends.md),
[docs/backends.md](docs/backends.md), [docs/cpu-kernels.md](docs/cpu-kernels.md)):

| Device | Native kernels | Everything else |
| --- | --- | --- |
| CPU (x86-64, AArch64) | F32/F16/BF16, Q4_0 ... Q8_0, Q4_K ... Q6_K; on x86-64 also Q2_K, Q3_K, IQ4_NL, IQ4_XS, MXFP4; MLX affine `MLXA` (all widths and groups, f32 activations, AVX2 where present) | rows decoded on demand (functional, slow): the other IQ/TQ types, NVFP4, Q4Z/Q4ZP |
| CUDA | all of the above, the IQ family, MXFP4, NVFP4 (FP4 tensor cores on Blackwell sm_12x, Marlin FP4 on other GPUs from Turing), AWQ/GPTQ `Q4Z`/`Q4ZP` through Marlin (Turing and newer, shapes in multiples of 128), MLX affine `MLXA` (own kernels on the packed words; not yet run on a GPU) | expanded at upload |
| wgpu (Vulkan, Metal, DX12) | F32/F16/BF16, Q4_0 ... Q8_0, Q4_K, Q5_K, Q6_K | expanded to F32 at upload (MLX affine: to F16) |
| Metal (native) | the formats of the table in [docs/gpu-backends.md](docs/gpu-backends.md), MLX affine `MLXA` (kernels written, checked as C++ on the CPU, awaiting the Apple-silicon run) | expanded at upload |

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
architecture names is in [docs/model-coverage-vllm.md](docs/model-coverage-vllm.md).

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
| Other | `/` the built-in Studio (browser app), `/health`, `/metrics` (Prometheus) |

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
  attention, argmax sampling on the device, and, on by default, CUDA graphs for decode of dense models,
  fused decode kernels (residual add + norm + activation quantization, rotary + cache write) and a prompt
  pass without per-layer host work. Each has a one-line switch back (`FLLM_CUDA_GRAPH=0`, `FLLM_FUSE=0`,
  `FLLM_PREFILL_STEP_CACHE=0`, ...), listed with all defaults in
  [docs/configuration.md](docs/configuration.md#switching-the-new-defaults-back). NVFP4 checkpoints are
  computed as W4A4 with the checkpoint's per-tensor `input_scale` (`FLLM_NVFP4_ACT_SCALE=static`, the
  default); `FLLM_NVFP4_ACT_SCALE=off` returns to quantizing the activations without it.
- **Memory.** Paged KV cache with 13 formats (`f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `int8`, `q8_0`, `q5_0`,
  `q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `nvfp4`, `f32`); keys and values can differ (`--cache-type-k`,
  `--cache-type-v`). wgpu devices keep `f16` and `q8_0` only, and 4-bit and 5-bit keys can break models
  with large key outliers. For models larger than the card, `--gpu-layers` keeps attention on the GPU and
  streams feed-forward weights and MoE experts from host memory ([docs/user/memory.md](docs/user/memory.md)).
- **Scheduling.** Continuous batching, chunked prefill, prefix cache (hybrid and state-space models
  switch off prefix sharing and speculation), beam search (`use_beam_search`, CPU backend), `--multi-step N`
  (several decode rounds per scheduler round, off by default and CPU only until it is verified elsewhere) and
  jump-forward decoding for grammar-constrained requests (text identical to token-by-token decoding; large gains
  only with compact JSON schemas). Optional: prompt blocks saved on disk (`FLLM_KV_DISK_DIR`) and an output cache
  for repeated deterministic requests ([docs/output-cache.md](docs/output-cache.md)).
- **Speculative decoding.** `--speculative off|ngram|suffix|draft|mtp`: conversation lookup (`suffix`, on by default on
  the CPU and on Vulkan/DirectX; on CUDA, HIP and Metal it stays off until its kernels have been checked on hardware),
  prompt lookup, a draft model of your own (`--speculative-model`, browse for it in the load dialog) and the
  multi-token-prediction heads of DeepSeek-V2/V3 and GLM-4 MoE (`--speculative mtp`, CPU only); greedy requests only,
  and not for models with a recurrent state (Qwen3.5, Mamba).

### Tools and clients

`fllm-rs studio` starts the server and opens the Studio, the built-in browser app with chat, a coding agent and a
model manager (the desktop app shows the same page in its own window): [docs/user/studio.md](docs/user/studio.md).
Inside the Studio:

- **Chat**: projects, assistants, documents, web access and research, sampling controls, thinking levels.
- **Coding agent** (Code page): reads, searches and edits files and runs commands in one folder, with permission
  rules, plan and build modes, sub-agents the agent can create itself, hooks, plugins, memory, skills and MCP tools.
  Optional and off by default: a **browser** the agent drives (a real Chrome, Chromium, Edge or Brave through the
  DevTools protocol, shown live in a panel where you can click and leave comments), **computer use** (screenshots,
  mouse and keyboard, every action asked first, a STOP button and a panic key) and a **command sandbox** (bubblewrap on
  Linux; a restricted token and job object on Windows, checked on one Windows 11 machine; Seatbelt on macOS, unchecked).
- **Skills and MCP servers**: install skills from any GitHub repository or the skills.sh directory, and find MCP servers
  in the official registry or on Smithery; nothing runs before you read the command and save it.
- **Settings > Engine** for the output cache, prompt sharing, telemetry (OTLP) and the disk tier; **API keys with an
  admin right** protect the server's own settings. Settings, plugins, agents and notes live in fLLM's own folder
  (`%LOCALAPPDATA%\fLLM`, `~/.local/share/fllm`, or `FLLM_HOME`).
- **Operation**: Docker images and Compose files, Kubernetes manifests and a Helm chart in [deploy/](deploy), and
  `fllm-router`, a prefix-aware router for several servers ([docs/router.md](docs/router.md)).

`fllm-rs chat` and `bench` talk to any running OpenAI-compatible server;
[docs/user/clients.md](docs/user/clients.md) covers curl, the OpenAI, Anthropic and Ollama SDKs and program integrations.

## Status

What has actually been run on hardware, and what has not:

| Target | State |
| --- | --- |
| CPU x86-64 (AVX2, AVX-VNNI, AVX-512) | Runs. Self test and real models on AVX2 and on an AMD Zen 5 (AVX-512); the AMX kernels have only been run in an emulator |
| NVIDIA CUDA, Turing (T4), Blackwell (RTX PRO 6000) | Runs. Self test 595 + 395 checks, real models (Llama, Qwen2.5, Qwen3.5 with DeltaNet, AWQ, GPTQ, NVFP4) |
| NVIDIA Ampere (A100) and Ada (L4) | Ran on Google Colab with v0.1.11: 587 of 644 self-test checks passed; the 57 failures (quantized KV-cache writes, a log-probability kernel, a DeltaNet head width, FP16 activations) have found causes and fixes that are **not yet re-run on a GPU**. MoE, GPTQ, CUDA graphs and prompt-lookup gave the CPU's text; Llama-3.2-3B Q4_K_M decodes at 218 tok/s (A100) and 95 tok/s (L4) |
| NVIDIA Hopper | Builds; not run |
| Linux aarch64, macOS (Apple silicon), Windows (MSVC) | Build, unit tests and the CPU self tests run in CI on GitHub-hosted runners (the Linux and Windows runners have no GPU). The SVE kernel is opt-in and was not run. Windows release assets and the desktop installer exist since 0.1.2; the installer was tried on one Windows PC only |
| Apple Metal | The native Metal kernels (attention for f16 and q8_0 KV caches in decode and prompt, KV-cache writes and more) run their GPU checks on the Apple-silicon CI runner and match the CPU reference to about 1e-7. A whole model has not been run on Metal yet |
| Vulkan | Ran on a few devices (NVIDIA, Intel, and an AMD handheld with a Radeon 780M, where the native backend runs the whole model on the GPU and the integrated-GPU memory falls back to shared host memory); slower than CUDA. DirectX 12 through wgpu has a known bug with legacy quantizations on AMD |
| Windows agent tools | The browser tool, computer use (screenshot, displays, refused key chords) and the sandbox ran on one Windows 11 machine with three displays; clicking and typing there were not tried |
| AMD ROCm (HIP), Intel SYCL, NPUs, TPU | Compile only (test level L0): not run on the hardware |

The test levels (L0 compiles, L1 self test passes on the device, L2 same greedy text as the CPU, L3 speed
measured) are defined in [docs/test-levels.md](docs/test-levels.md).

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

**With the latest performance work** (this code: CUDA graphs and fused decode kernels on, prompt pass without per-layer host
work, NVFP4 activation scale and tile table), same RTX PRO 6000 Blackwell Colab machine, same client (single user, greedy,
HTTP). Arrows show the table above (before) and the new value:

| Model | fLLM decode | fLLM prefill 355 / 1000 / 1863 / 8000 tokens (tok/s) | vLLM, same card and session |
| --- | --- | --- | --- |
| Llama-3.1-8B NVFP4 | 194 -> 228 tok/s | 24,700 -> 28,800 / 34,800 -> 43,300 / 35,500 -> 40,700 / 32,700 -> 34,200 | 201 tok/s; 20,400 / 50,300 / 54,700 / 44,900 |
| Qwen2.5-3B AWQ | 200 -> 238 tok/s | 1863 tokens: 29,000 -> 34,800 | 261 tok/s; 24,000 / 36,100 / 39,400 / 37,800 |
| Llama-3.1-8B Q4_K_M | 190 -> 225 tok/s | not re-measured | not compared |
| Llama-3.2-3B Q4_K_M | 315 -> 419 tok/s | not re-measured | not compared |

Read honestly: for NVFP4, fLLM now decodes faster than vLLM (228 vs 201 tok/s) and prefills short prompts faster (355 tokens:
28.8k vs 20.4k tok/s), but is still about 15 to 25 % behind it for long prompts (1863 tokens 40.7k vs 54.7k; 8000 tokens 34.2k vs
44.9k). For the AWQ model fLLM is 9 to 17 % behind vLLM (decode 238 vs 261, prefill at 1863 tokens 34.8k vs 39.4k). The machine
is shared and run-to-run noise is around 15 %, so single percentages mean little. The "before" values in the arrows come from the A/B run of the same session and differ slightly from the first table (a different run).

On an RTX PRO 3000 Blackwell Laptop (12 GB, power-limited), one request at a time, Llama-3.1-8B Q4_K_M decodes at
96 tok/s and prefills a 1,863-token prompt at 3,410 tok/s (llama.cpp on the same card and client: 85 and 3,145). The
NVFP4 build reaches 102 tok/s and 10,070 tok/s (SGLang on the same card: 94 and 11,900, so fLLM is behind for long
prompts there too). gpt-oss-20b (MXFP4, larger than the card) decodes at about 45 tok/s. More formats, MoE models and
concurrent requests are in [docs/gpu-backends.md](docs/gpu-backends.md).

On Google Colab with the v0.1.11 release binary (single user, greedy, over HTTP; Llama-3.2-3B-Instruct Q4_K_M, contexts 355 /
1863 / 8000 tokens, one run each): an A100 80 GB decodes at 218 / 244 / 232 tok/s and prefills at 5,128 / 16,090 / 19,952 tok/s; an
L4 decodes at 95 / 96 / 78 tok/s and prefills at 4,088 / 6,540 / 6,909 tok/s. No other engine was measured on these two runs.

## Documentation

The code and `--help` are authoritative where a page and the program disagree.

**Start here**
- [docs/index.md](docs/index.md), [docs/getting-started.md](docs/getting-started.md): documentation index, install, get weights, quantize, serve, curl, chat, embeddings
- [docs/output-cache.md](docs/output-cache.md), [docs/router.md](docs/router.md): the output cache (deterministic answers are not computed twice), the prefix-aware router for a fleet, and the switches that decide how much state requests share
- [benchmarks/README.md](benchmarks/README.md): the benchmark client and what a published number has to state
- [deploy/](deploy): Kubernetes manifests and a Helm chart
- [docs/quickstart-windows-linux.md](docs/quickstart-windows-linux.md): Docker with an NVIDIA GPU, step by step
- [docs/user/studio.md](docs/user/studio.md): the Studio and desktop app: chat, the coding agent, browser, computer use, sandbox, skills, MCP, settings, where files are kept
- [docs/user/commands.md](docs/user/commands.md): every `fllm-rs` subcommand
- [docs/configuration.md](docs/configuration.md): every `serve` flag and environment variable
- [docs/user/clients.md](docs/user/clients.md), [docs/user/memory.md](docs/user/memory.md), [docs/user/troubleshooting.md](docs/user/troubleshooting.md): clients, memory planning, error messages

**Models and formats**
- [docs/architectures.md](docs/architectures.md): supported and unsupported architectures, verification method
- [docs/model-coverage.md](docs/model-coverage.md), [docs/model-coverage-vllm.md](docs/model-coverage-vllm.md): coverage against popular open models and against vLLM/SGLang
- [docs/architecture.md](docs/architecture.md), [docs/adding-a-model.md](docs/adding-a-model.md), [docs/adding-a-backend.md](docs/adding-a-backend.md): engine structure, adding an architecture or backend
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
- [docs/backends.md](docs/backends.md), [docs/backends-verify.md](docs/backends-verify.md): hardware by format matrix, verification steps
- [docs/cpu-kernels.md](docs/cpu-kernels.md), [docs/cpu-arm.md](docs/cpu-arm.md), [docs/cpu-avx512-amx.md](docs/cpu-avx512-amx.md): CPU kernels
- [docs/cuda-qgemm4k.md](docs/cuda-qgemm4k.md): Q4_K prompt GEMM
- [docs/windows.md](docs/windows.md), [docs/test-levels.md](docs/test-levels.md), [docs/benchmarking.md](docs/benchmarking.md), [docs/roadmap.md](docs/roadmap.md), [docs/research/README.md](docs/research/README.md): Windows builds, test levels, benchmarking, roadmap, and research notes

**Deployment and project**
- [docs/docker.md](docs/docker.md), [compose.yml](compose.yml), [compose.cuda.yml](compose.cuda.yml): images and compose files
- [docs/ci.md](docs/ci.md), [scripts/README.md](scripts/README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md): workflows, verification scripts, contributing, and changelog
- [SECURITY.md](SECURITY.md), [THIRD_PARTY.md](THIRD_PARTY.md), [docs/releasing.md](docs/releasing.md): security policy, third-party code, licenses, and release procedure

## Building from source

```sh
cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs           # CPU + wgpu
cargo build --release --locked -p fllm-engine --features "rust-gpu,cuda" --bin fllm-rs    # plus CUDA, needs nvcc
FLLM_CUDA_ARCHS=120 cargo build --release --locked -p fllm-engine --features cuda          # one GPU generation, faster
./install.sh --build                                                                       # build and install into ~/.local/bin
```

| Cargo feature | Adds |
| --- | --- |
| (none) | CPU only (plus the default `avx512` feature on x86-64 and the default `npu-openvino` feature, which only loads the OpenVINO runtime when one is installed) |
| `rust-gpu` | wgpu backends: Vulkan, Metal, DirectX 12 (what `make`, `install.sh`, the release binaries and the CPU image use) |
| `cuda` | NVIDIA kernels, compiled with `nvcc` (architectures from `FLLM_CUDA_ARCHS`) |
| `avx512` | x86-64 AVX-512 / AMX CPU kernels; **on by default** (`--no-default-features` leaves them out; chosen at run time by CPUID, so the binary still runs on any x86-64 CPU) |
| `npu-openvino` | Intel NPU backend (experimental, L0); **on by default**, loads the OpenVINO runtime at run time (`fllm-rs npu-setup`); see [docs/backends.md](docs/backends.md) section 6 |
| `hip`, `sycl`, `vulkan`, `metal`, `tpu-pjrt` | optional native backends; the test level differs per backend (Metal: kernel checks in CI, others L0), see [docs/backends.md](docs/backends.md) |

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
