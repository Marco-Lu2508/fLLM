# Roadmap

An honest list of what is missing and what is being worked on. "In progress" means work has started and is not merged or not yet
verified; "planned" means wanted but not started; "needs hardware" means it depends on results from machines the maintainers do not
own ([CONTRIBUTING.md](../CONTRIBUTING.md), hardware reports). There are no dates: items move when they are done and verified at the
level stated in [test-levels.md](test-levels.md). Details of what exists today are in the README Status table, [backends.md](backends.md)
and [model-coverage.md](model-coverage.md).

## Performance and GPU

| Item | State | Notes |
| --- | --- | --- |
| Mixture-of-experts on the GPU (CUDA) | in progress | Today routing and expert GEMMs run on the host for most models, with GPU streaming of host weights for prompts. Device-resident experts are the goal |
| Newer NVIDIA generations (Hopper, Blackwell data-centre tuning) | in progress | The kernels build for these cards; tuning and measurements are open (needs hardware) |
| Closing the prefill gap to vLLM for long prompts and small quantized models | in progress | See the measurement tables in the README and [gpu-backends.md](gpu-backends.md) |
| Multi-GPU (tensor or pipeline parallel) | in progress | Implemented and proven on the CPU path ([multi-gpu.md](multi-gpu.md)); the CUDA forward does not call the collectives yet, so the server refuses it on a GPU |
| Act-order GPTQ, other AWQ/GPTQ group sizes, GPTQ 8-bit | planned | Unsupported variants fall back to dense F16 at conversion |
| Whole-model runs on Metal, ROCm, SYCL, NPU and TPU | needs hardware | The kernels exist; the levels are in the Status table |
| Vulkan completeness (missing operations) | planned | Some operations fall back to the host |

## Models and formats

| Item | State | Notes |
| --- | --- | --- |
| New architectures (Gemma 4 text models exist and are checked against `transformers`; others, for example Jamba, are open) | in progress | Adding a model: [adding-a-model.md](adding-a-model.md); each needs a tiny-checkpoint check against `transformers` |
| Loading checkpoints directly, without writing a GGUF first | done and the default for unquantized, AWQ, GPTQ, FP8, ModelOpt NVFP4, compressed-tensors, MXFP4, bitsandbytes, MLX, EXL2 and EXL3 weights in safetensors or PyTorch ZIP files; `--quant` while loading | Proven byte for byte against the converted file by tests, see [rust-formats-and-import.md](rust-formats-and-import.md). Fixtures hold random codes in each layout; EXL3 logits are not compared through the engine (its tiles need dimensions of 128). Not measured on real multi-GB models |
| More vision families (Qwen2/2.5/3-VL and SmolVLM/Idefics3 exist) | planned | Image input, projector and tokenizer-side placeholders per family |
| Audio and speech input | in progress | The WAV reader, log-mel frontend and Whisper encoder exist; the text decoder is missing, so the transcription routes answer 501 |
| Full SentencePiece/Unigram tokenizer support | planned | BPE families are covered |
| Remaining quantization paths (for example more EXL3 variants) | planned | See [exl-formats.md](exl-formats.md) |

## Inference features

| Item | State | Notes |
| --- | --- | --- |
| Speculative decoding for sampled (non-greedy) requests with exact distribution | planned | Greedy requests only today |
| Runtime LoRA switching per request | planned | Adapters are merged at load time |
| Embedding adapters and activated LoRA | planned | |
| Request-level scheduling policies (priorities, fairness limits) | planned | |

## Agent and gateway

| Item | State | Notes |
| --- | --- | --- |
| Headless agent (`fllm-rs agent -p`) | done | Studio tools over HTTP, permissions by flag, sessions, trajectory export ([user/agent-cli.md](user/agent-cli.md)); needs a running server on the same machine |
| Telegram and Matrix gateway (`fllm-rs gateway telegram|matrix`) | done | Allowlist, per-chat session, saved stream position, same permissions as `agent`; run against stubs only, not against the real services; Matrix rooms must be unencrypted |
| Delivery of scheduled-task results (webhook, ntfy, Telegram) | done | `deliver` field of a task; Telegram token from `FLLM_TELEGRAM_TOKEN` |
| Other messengers (Discord, Signal, mail) | planned | The gateway talks to a `Chat` trait; Telegram and Matrix implement it |
| Profiles that split skills, memory and tasks | planned | `--profile` splits only sessions, `SOUL.md` and gateway positions |
| Tasks that run while the app is closed | planned | The scheduler is a thread of the server and runs only while a model is loaded |
| Remote execution backends (SSH, container) for the agent tools | planned | Tools run on the machine of the server |
| IDE extension, GitHub pull-request workflow | planned | Not started |

## Platforms, packaging and project health

| Item | State | Notes |
| --- | --- | --- |
| Windows release assets | planned | Windows builds run in CI; installers build from source until assets are published |
| Automated dependency updates | planned | Dependabot is disabled until main is stable |
| `cargo fmt` enforced in CI | planned | Needs one formatting commit first |
| Published reproducible verification receipts for every architecture | in progress | The proof scripts are in `scripts/`; see `scripts/README.md` |
| AVX-512 and AMX on real silicon beyond the Zen 5 results | needs hardware | AMX has only run in an emulator |

Suggestions and contributions are welcome: open a feature request, or pick an item and say so in an issue before you start.
