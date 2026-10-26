# Independent Rust inference

fLLM runs its own inference engine, written in Rust. Model layers execute with
fLLM's Rust tensor operations, its own CPU SIMD kernels and independently authored GPU kernels
(CUDA, WGSL, and the native Vulkan, Metal, HIP and SYCL backends).
There is no llama.cpp mode, libllama inference, GGML dependency, upstream server
process or silent switch to an external runtime.

| Feature | Own implementation and current boundaries |
| --- | --- |
| Decoder | Rust embedding lookup, RMSNorm/grouped RMSNorm, GQA, model-specific RoPE, attention, residuals, SwiGLU and LM head. Dense and fused Q/K/V and gate/up projections, MoE (softmax and sigmoid routing, shared experts), MLA, and recurrent mixers (gated DeltaNet, Mamba 1/2, LFM2 short convolution). The architecture families, each checked against `transformers` on a tiny random checkpoint (`scripts/verify_architectures.py`), are listed in [architectures.md](architectures.md); real-model results are in [model-coverage.md](model-coverage.md). Where a backend lacks a kernel for a layer, that layer runs on the host (CPU) while the rest stays on the device. |
| Quantization | Rust readers for the GGUF tensor types listed by `fllm-rs formats` (F32, F16, BF16, the Q4/Q5/Q8 and K-quant families, the IQ family, TQ, MXFP4, NVFP4) plus fLLM's own AWQ/GPTQ-derived Q4Z formats. Weights stay packed during projection. Own CPU kernels are chosen at runtime (AVX2/FMA/F16C, AVX-VNNI, AVX-512 BW/VNNI/BF16, AMX, NEON dotprod/i8mm); they quantize activations to 8 bits, see [cpu-kernels.md](cpu-kernels.md). The CUDA backend decodes the K-quant, IQ, MXFP4, NVFP4 and Q4Z formats natively ([gpu-backends.md](gpu-backends.md)); the wgpu backend has WGSL projection kernels for eight quantized formats. Formats without a packed kernel on a backend decode rows on demand or are converted once on upload. |
| KV cache | Own paged storage and attention in FP16, BF16, FP8, INT8, Q8_0, Q5, Q4, IQ4_NL, NVFP4 and FP32 (keys and values separately, CPU and CUDA; wgpu: FP16 and Q8_0). Prefix reuse, continuous batching, chunked prefill, admission limits, cancellation and bounded output buffers. |
| Sampling | Temperature, top_p/top_k, min_p, typical_p, repetition/presence/frequency penalties and history windows, DRY/breakers, Mirostat 1/2, logit bias and per-sequence seeds. |
| Structured output | Sampler masks enforce JSON Schema, JSON objects, Lark, regex and a common GBNF CFG subset. Unsupported grammar constructs reject before generation. |
| Tools | OpenAI/Ollama declarations, required/named choices, Qwen tags/Mistral arrays, argument-schema validation and streaming. Tool execution stays with the client. |
| Encoders | Own Rust WordPiece tokenizer and bidirectional BERT layers. CLS/mean/last pooling and L2 normalization; BGE-small tested against PyTorch. Causal decoder embeddings remain available separately. BGE is English trained; tokenizer handling of other text is not a claim of multilingual retrieval quality. |
| Vision | Own Rust SigLIP encoder, patch convolution, normalization, attention, GELU, pixel shuffle and Idefics3 projector. The SmolVLM (Idefics3) family is the only supported one; it supports OpenAI image chat and a diagnostic image-feature endpoint. One resized global image view; no high-resolution crop splitting or audio. |
| LoRA | Load-time projection merge, multiple adapters/scales. Applied weights transfer to own Rust compute. Adapted matrices use F32 RAM. Embedding adapters, activated LoRA and request-time switching remain unsupported. |
| Speculative decoding | A start option of the server (`--speculative ngram`, `--speculative draft --speculative-model FILE`, `--speculative mtp`, `--speculative suffix`; the Studio sets it in the load dialog, Advanced). Prompt lookup (no extra model), a cached draft model of the same tokenizer, or the model's own multi-token prediction heads; target verification and KV rollback. Greedy requests give exactly the tokens of plain decoding; sampled and grammar-constrained requests are verified with the speculative sampling rule and keep the distribution of plain sampling. Speed only: it is not faster when the guesses are rarely right. Refused for models with recurrent layers (Qwen3.5, Mamba, ...); MTP needs a model with the heads and runs on the processor or CUDA. A request can opt out with `"speculative": false`. Details: [speculative-decoding.md](speculative-decoding.md). |
| Thinking control | `capabilities.thinking` in the status route says per model whether thinking can be switched (`toggle`), scaled (`effort`), or not at all (`always`, `none`), read from the chat template. `thinking_budget` (tokens) or `reasoning_effort` low/medium ends the reasoning block after that many tokens and lets the model answer (budget forcing). See [server-api.md](server-api.md#thinking). |
| Hardware | CPU (x86-64, AArch64), NVIDIA CUDA (full device-resident decoder: norms, attention with paged KV cache, MLP, sampling; run on Turing T4 and Blackwell RTX PRO 6000, other generations build but are not measured), wgpu (Vulkan, Metal, DX12), and the native Vulkan, Metal, HIP, SYCL, NPU and TPU backends. Vulkan ran on a few devices (NVIDIA, Intel Arrow Lake); the native Metal kernel checks run on an Apple-silicon CI runner; ROCm, SYCL, NPUs and TPU are compile-only. The per-target state is the Status table of the README; the test levels are defined in [test-levels.md](test-levels.md). |

## Build and run

Build and install with `./install.sh`, then:

```sh
fllm-rs serve --model qwen.gguf --device cpu
fllm-rs serve --model qwen.gguf --device vulkan
fllm-rs serve --model qwen.gguf --draft-model small-draft.gguf --draft-tokens 5
fllm-rs serve --model qwen.gguf --lora adapter.gguf --kv-cache-type q8_0
fllm-rs serve --model bge-small.gguf --embeddings --pooling cls
fllm-rs serve --model SmolVLM.gguf --mmproj mmproj-SmolVLM.gguf --device vulkan
```

`--device auto` chooses the best GPU (CUDA, then ROCm, Metal, Vulkan, DX12, SYCL) and falls back to the CPU when no GPU is
found; see [device-selection.md](device-selection.md). Explicit GPU requests fail if the requested backend
is unavailable or only a software adapter exists. GPU weights are uploaded once;
large projections are split to respect device buffer/dispatch limits. Input and
output transfers between projections are currently retained for correctness,
so device support alone is not a performance claim.

Rust decoder embeddings and projection weights share the mapped GGUF bytes.
Only requested embedding rows are decoded to F32; the full vocabulary is not
expanded into a second F32 table. LoRA merges can increase memory further.
Running processes are not restarted by a source update. The older `--compute rust|auto` flag is
accepted and ignored; `--compute native` was removed together with the C++ kernels.

CPU numerical checks cover all eleven listed tensor types, grouped projection
tails, packed activation rounding, paged Q8/F16 attention and split-KV. Current
measurements are reproduced with `rust/engine/examples/perf_decode.rs`.

## API details

OpenAI requests accept the sampling fields above, `grammar`/`grammar_type`
(`gbnf`, `lark`, `regex`), and `response_format` of text/json_object/json_schema.
For JSON Schema supply `response_format.json_schema.schema`. Chat accepts tools,
tool_choice and parallel_tool_calls; required tools cannot be combined with
another output schema. Validated calls stream after their complete arguments.

Ollama sampling lives in `options`; `format` accepts JSON or a schema.
`/v1/embeddings`, `/api/embed`, `/api/embeddings` accept strings, token IDs and
batches up to 32. Dedicated BERT defaults to the model's GGUF pooling metadata
(BGE: CLS), accepts CLS/mean/last, and rejects generation endpoints. Dedicated
encoder inputs must fit their model context; invalid IDs/dimensions fail clearly.
Float and base64 little-endian F32 vectors are supported.

`POST /tokenize` accepts `{"content":"Grüße!","add_special":false}`;
`POST /detokenize` accepts `{"tokens":[...]}`. BERT uses a real WordPiece model;
the existing decoder tokenizer covers its supported BPE families. General
SentencePiece/Unigram support is still outstanding.

With a matching SmolVLM projector, OpenAI chat accepts `image_url` content parts
containing PNG/JPEG base64 data URLs, plus ordinary text parts. Up to four images
are encoded, resized to the configured square global view, and projected into
the decoder's image-token positions. Image requests disable prefix caching and
speculation, preventing reused KV state for a different picture. The custom
`/v1/images/embeddings` endpoint accepts `{"image_url":"data:image/png;base64,..."}`
and returns projected image tokens for numerical verification. Other vision
families, remote image downloads, Ollama image payloads and audio are not covered.

Draft/target must share token IDs and BOS/EOS metadata. `/metrics` exports draft
proposed/accepted counts. Speculation's speed depends on draft cost, acceptance
and hardware; enabling it does not establish a speedup. Tokenization, prompt
prefill, retrieval/vision encoding and speculative decoding are distinct tasks.

## Reproducible checks

```sh
make test
make lint
python3 scripts/verify_features.py --binary /path/fllm-rs --model qwen.gguf
python3 scripts/verify_features.py --binary /path/fllm-rs --model qwen.gguf --draft-model small-draft.gguf
python3 scripts/verify_encoder.py --binary /path/fllm-rs --model bge-small.gguf
python3 scripts/verify_vision.py --binary /path/fllm-rs --model SmolVLM.gguf --mmproj mmproj.gguf
cargo build --release --locked --features rust-gpu --manifest-path rust/engine/Cargo.toml --example verify_quant
python3 scripts/verify_rust_quants.py --create --fixture /tmp/quants.gguf --binary /path/examples/verify_quant
```

Test-only PyTorch/numpy/gguf/tokenizers packages provide independent references;
production encoder/vision/decoder computations do not import them. Small test
model downloads are pinned by `scripts/fetch_feature_models.py`; models stay
outside Git. CPU/GPU quantization fixtures exercise high bits, signed scales, row
boundaries, multiple input rows and dequantization against gguf/PyTorch.

The earlier upstream runtime measurements are not evidence for these paths.
Remaining work is listed in [roadmap.md](roadmap.md): more architectures, full SentencePiece,
stochastic exact speculation, dynamic adapters, other image families and audio, and
physical validation on more devices.
