# Rust serving binary (fllm-rs)

`fllm-rs` is the whole engine: GGUF model setup, tokenizers and chat templates, scheduler with paged
KV memory, prefix reuse and continuous batching, the decoder, BERT and SigLIP/Idefics3 vision
implementations, and the OpenAI/Ollama-compatible server. Build with `--features rust-gpu` for the
quantized WGSL projections. See
[current feature limits and real-device verification](../../docs/inference-features.md).

## Build and run

From the repository root:

    cargo build --release --locked --features rust-gpu -p fllm-engine --bin fllm-rs

Start the server:

    target/release/fllm-rs serve --model /path/to/model.gguf --host 0.0.0.0 --port 8000 --ctx 4096 --max-seqs 16 --max-batched-tokens 512 --threads 8

Options include:
- `--model PATH`: GGUF model file or `hf://owner/repo/file.gguf` (required).
- `--host HOST`: Bind host (default: `127.0.0.1`).
- `--port PORT`: Port number (default: `8000`).
- `--ctx N` (or `--max-model-len`): Context length per sequence (default: `4096`).
- `--max-seqs N`: Maximum concurrent sequences (default: `16`).
- `--max-batched-tokens N`: Max tokens per forward pass budget (default: `512`).
- `--threads N`: Worker threads (default: available CPU parallelism).
- `--device auto|cpu|vulkan|metal|dx12`: Compute device (default: `auto`, which means CPU).
- `--kv-cache-type f16|q8_0`: KV cache storage (default: `f16`).
- `--api-key KEY`: Optional bearer token auth (or via `FLLM_API_KEY`).
- `--model-name NAME` (or `--served-name`): Custom model alias name in API routes.

## HTTP surface and capabilities

The server exposes:
- `/health`: Health status probe.
- `/metrics`: Prometheus metrics.
- `/v1/models`: OpenAI-compatible model listing.
- `/v1/completions`: OpenAI-compatible text completions (streaming SSE and non-streaming).
- `/v1/chat/completions`: OpenAI-compatible chat completions (streaming SSE and non-streaming).
- `/api/tags`: Ollama-compatible model tag listing.
- `/api/generate`: Ollama-compatible completions (NDJSON streaming and non-streaming).
- `/api/chat`: Ollama-compatible chat completions (NDJSON streaming and non-streaming).
- `/api/version`: Ollama version endpoint.

### Sampling and Stop Strings
- Exact greedy sampling (`temperature=0.0`) with exact tie-breaking.
- Multi-string stop sequence filtering (`StopFilter`) with partial-match buffer retention across token boundaries, verified for single strings, string arrays, and UTF-8 multi-byte sequences.

## Verification

Run the engine test suite:

    cargo test --workspace --locked --features rust-gpu

Numerical and end-to-end harnesses live in `scripts/` (`make test-model MODEL=...`).
