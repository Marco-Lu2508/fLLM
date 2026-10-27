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
| Multi-GPU (tensor or pipeline parallel) | planned | One model runs on one device today; layers that do not fit are split between device and host |
| Act-order GPTQ, other AWQ/GPTQ group sizes, GPTQ 8-bit | planned | Unsupported variants fall back to dense F16 at conversion |
| Whole-model runs on Metal, ROCm, SYCL, NPU and TPU | needs hardware | The kernels exist; the levels are in the Status table |
| Vulkan completeness (missing operations) | planned | Some operations fall back to the host |

## Models and formats

| Item | State | Notes |
| --- | --- | --- |
| New architectures (for example the Gemma 4 family) | planned | Adding a model: [adding-a-model.md](adding-a-model.md); each needs a tiny-checkpoint check against `transformers` |
| Loading safetensors directly, without writing a GGUF first | done for unquantized, AWQ, GPTQ, FP8, ModelOpt NVFP4 and compressed-tensors safetensors (default); `--quant` while loading; EXL2/EXL3, MXFP4, bitsandbytes, MLX and PyTorch weights still convert | Proven byte for byte against the converted file by tests, see [rust-formats-and-import.md](rust-formats-and-import.md). Not measured on real multi-GB models |
| More vision families (beyond SmolVLM/Idefics3) | planned | Image input, projector and tokenizer-side placeholders per family |
| Audio and speech input | planned | Not started |
| Full SentencePiece/Unigram tokenizer support | planned | BPE families are covered |
| Remaining quantization paths (for example more EXL3 variants) | planned | See [exl-formats.md](exl-formats.md) |

## Inference features

| Item | State | Notes |
| --- | --- | --- |
| Speculative decoding for sampled (non-greedy) requests with exact distribution | planned | Greedy requests only today |
| Runtime LoRA switching per request | planned | Adapters are merged at load time |
| Embedding adapters and activated LoRA | planned | |
| Request-level scheduling policies (priorities, fairness limits) | planned | |

## Platforms, packaging and project health

| Item | State | Notes |
| --- | --- | --- |
| Windows release assets | planned | Windows builds run in CI; installers build from source until assets are published |
| Automated dependency updates | planned | Dependabot is disabled until main is stable |
| `cargo fmt` enforced in CI | planned | Needs one formatting commit first |
| Published reproducible verification receipts for every architecture | in progress | The proof scripts are in `scripts/`; see `scripts/README.md` |
| AVX-512 and AMX on real silicon beyond the Zen 5 results | needs hardware | AMX has only run in an emulator |

Suggestions and contributions are welcome: open a feature request, or pick an item and say so in an issue before you start.
