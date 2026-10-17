# Architecture overview

This page describes how fLLM is put together: the crates, the path of a request, the scheduler, the backend abstraction, weight
formats and kernels, the KV cache and the import pipeline. It is a map, not a specification; flags, defaults and numbers live in
`fllm-rs --help` and [configuration.md](configuration.md).

## Crates

| Crate | Directory | Role |
| --- | --- | --- |
| `fllm-engine` | `rust/engine` | The engine and the `fllm-rs` binary: server, scheduler, decoder, tokenizer, sampling, CPU kernels, backends, CLI |
| `fllm-checkpoint` | `rust/checkpoint` | Converts Hugging Face checkpoints (safetensors, PyTorch, AWQ, GPTQ, FP8, ModelOpt NVFP4, EXL) to GGUF; also quantizes |
| `fllm-gguf-probe` | `rust/gguf-probe` | Read-only, memory-mapped GGUF reader: metadata, tokenizer arrays, tensor index, validated zero-copy tensor bytes |

GPU kernels live next to the engine as source files that `build.rs` compiles when the matching Cargo feature is on:
`rust/engine/cuda` (also built with `hipcc` for AMD), `rust/engine/metal`, `rust/engine/shaders/vulkan`, `rust/engine/sycl`.

## Overview

```text
 client (OpenAI / Anthropic / Ollama / llama.cpp / SGLang / TGI protocol)
        |  HTTP (axum)
        v
 +--------------------------+     +-------------------+
 | server + server/api      |---->| tokenizer, chat   |  template (minijinja), tools, reasoning parsers
 +--------------------------+     +-------------------+
        | requests (prompt ids, sampling params, grammar)
        v
 +--------------------------+     block allocator + prefix cache (mem/)
 | scheduler                 |<--------------------------------------+
 | continuous batching,      |                                       |
 | chunked prefill, admission|                                       |
 +--------------------------+                                        |
        | batch: token ids, lengths, block tables                     |
        v                                                             |
 +--------------------------+     +----------------------------+     |
 | runtime (step loop)       |---->| rust_decoder (layers)      |-----+
 +--------------------------+     |  embedding, norm, attention|  paged KV cache
        ^  sampled tokens         |  MLP/MoE/MLA/recurrent...  |
        |                         +-------------+--------------+
 +--------------------------+                   | Stream / compute calls
 | sampling + grammar masks  |<-- logits --------+
 +--------------------------+       +-----------+-----------+------------+
                                    | CPU SIMD  | CUDA/HIP  | Vulkan, Metal, SYCL, NPU, TPU, wgpu
                                    +-----------+-----------+------------+
```

## Request flow

1. **HTTP.** `server.rs` builds one axum router for decoder models and `encoder_server.rs` a smaller one for BERT-style encoders. The
   protocol-specific handlers (Anthropic, Responses, Files/Batches, llama.cpp, TGI, Ollama extras) are in `server/api/` and translate to
   the same internal request type. Authentication, the request-size limit and Prometheus metrics are middleware.
   [server-api.md](server-api.md) and [api-compat.md](api-compat.md) specify the routes.
2. **Prompt preparation.** The chat template stored in the GGUF is rendered with minijinja (`template_json.rs`), tools are declared
   in the model's own format (`tool_parsers.rs`), and the text is tokenized by the tokenizer rebuilt from the GGUF (`tokenizer.rs`).
   Images go through the vision encoder (`vision.rs`) and replace the image placeholder positions.
3. **Admission.** The request is registered with the scheduler. A bounded semaphore limits queued requests; invalid parameters fail here
   with an OpenAI-style error.
4. **Scheduling.** Each step the scheduler (`scheduler.rs`) picks sequences for a batch within the token budget: decode tokens
   first, then chunks of prompts that still need prefill. Cached prefix blocks (`mem/prefix_cache.rs`) are reused, and blocks come
   from the block allocator (`mem/block_allocator.rs`). Finished or cancelled sequences release their blocks.
5. **Forward pass.** `runtime.rs` runs the step loop on a worker thread and hands the batch to the decoder (`rust_decoder.rs`), which
   executes the layers on the selected device, or per layer on the host when a backend lacks a kernel.
6. **Sampling.** Logits become tokens in `sampling.rs`: penalties, top-k/top-p/min-p, Mirostat, DRY, logit bias and seeds; grammar masks
   (`structured.rs`, built on llguidance) restrict the allowed tokens. On the GPU, greedy requests can be sampled on the device.
7. **Streaming.** Token events go back to the HTTP task, which detokenizes, applies the reasoning and tool-call parsers and writes
   Server-Sent Events or one JSON body.

## Scheduler and runtime

The scheduler is device independent: it knows sequences, token budgets and KV blocks, not tensors. The runtime owns the model and the
step loop; a speculative-decoding draft model (`draft.rs`) or prompt lookup plugs in at the runtime level and rolls the KV cache back
when proposed tokens are rejected. Recurrent layers (DeltaNet, Mamba, short convolutions) keep per-sequence state next to the KV cache,
which is why prefix sharing is restricted for hybrid models ([architectures.md](architectures.md)).

## Backends and streams

All accelerators implement two traits in `rust/engine/src/backend/mod.rs`:

* **`Backend`** is the device: name and index, capabilities (supported weight formats and operations), buffer allocation, weight
  upload in native quantized form, and allocation of the paged KV cache. It creates streams.
* **`Stream`** is an ordered queue of operations on that device: embedding lookup, norms, RoPE, quantized linear layers, attention over
  the paged cache, KV writes, activations, MoE routing and expert GEMMs, MLA, Mamba, DeltaNet and short-convolution layers, residuals,
  device sampling, and optional fused or graph-captured variants. Every method has a default that reports
  `UnsupportedOperation`; the decoder then runs that operation on the host (CPU) and continues on the device.

Backends are selected with `--device` (`auto` takes the best GPU, else the CPU); [device-selection.md](device-selection.md) explains the
policy. The CPU path is itself a backend plus a set of SIMD kernels chosen at run time from CPUID ([cpu-kernels.md](cpu-kernels.md)).
How to write a new one: [adding-a-backend.md](adding-a-backend.md).

## Weight formats and kernels

GGUF is the internal model format. Weights stay memory-mapped and packed; kernels read quantized blocks directly and quantize the
activations to 8 bits (or use floating-point tensor-core paths). `fllm-rs formats` lists the supported tensor types. In addition to the
GGUF families (legacy Q4/Q5/Q8, K-quants, IQ, MXFP4, NVFP4, F16/BF16/F32) fLLM defines its own Q4Z formats that hold AWQ and GPTQ
4-bit weights with group scales and zero points ([awq-gptq-marlin.md](awq-gptq-marlin.md)). Per backend the status of every format and
operation is in [backends.md](backends.md) and [gpu-backends.md](gpu-backends.md); [quantization.md](quantization.md) explains the
formats from a user's view.

## KV cache

The cache is a pool of fixed-size blocks per layer; a sequence owns a block table. Keys and values can use different storage formats
(`kv_format.rs` defines the layout shared by the CPU decoder and the CUDA kernels), from F32 and FP16 down to 4-bit codes. Attention
kernels read blocks through the table, so sequences of different lengths batch without padding, and equal prompt prefixes share blocks.
On devices the pool size is planned from the free memory after the weights have been placed (`backend/memory.rs`, `mem/`).

## Import pipeline

`fllm-rs run`, `serve` and `pull` accept a GGUF file, a Hugging Face reference or a checkpoint directory. `hub.rs` resolves and downloads
files into an immutable snapshot cache. For non-GGUF checkpoints `import.rs` calls `fllm-checkpoint`, which reads safetensors (or PyTorch
zip) tensors memory-mapped, maps names and layouts of the architecture to the GGUF layout, optionally re-quantizes, writes the GGUF once
and records its source and converter version; later starts find the cached file. [rust-formats-and-import.md](rust-formats-and-import.md)
and [adding-a-model.md](adding-a-model.md) describe the details.

## Verification

Every layer of this picture has a check: unit tests per module, `selftest-cpu` and `selftest-gpu` for kernels against f64 references,
tiny random checkpoints compared with `transformers` for each architecture, and greedy-text comparisons against the CPU for backends.
The levels are defined in [test-levels.md](test-levels.md); the scripts are described in `scripts/README.md`.
