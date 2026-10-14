# Getting started

Install fLLM, turn a Hugging Face checkpoint into a GGUF file, quantize it, start the server
and call it. Every command below is the exact CLI of the `fllm-rs` binary
(`fllm-rs <command> --help` prints the usage strings).

## 1. Install

Three routes; pick one.

**Release binary** (downloads from this repository's GitHub releases and installs to
`./.local/bin/fllm-rs`, the path `run.sh` uses):

```sh
./install.sh              # newest release
./install.sh v0.1.0       # a specific tag
```

The script detects OS/architecture, verifies the asset against the published `.sha256` file
(a release without that file is a hard error unless you pass `--no-verify`) and, when no asset
exists for the platform, prints a clear message and builds the checkout with `cargo` instead.
That fallback needs a Rust toolchain, a C compiler, CMake and Git.

**Docker** (CPU by default, model directory mounted read-only):

```sh
FLLM_MODELS_DIR=/path/to/models FLLM_MODEL=model.gguf docker compose up -d
```

See [docker.md](docker.md) for the available `FLLM_*` variables.

**From source:**

```sh
make build-rust     # cargo build --release --locked -p fllm-engine --features rust-gpu --bin fllm-rs
./install.sh --build
```

Confirm the installation:

```sh
./run.sh formats        # registered GGML storage types
./run.sh devices        # compiled backends, detected devices, CPU ISA
```

## 2. Get weights

Either pull a ready-made GGUF file from the Hugging Face Hub:

```sh
./run.sh pull hf://owner/repo/model.gguf
./run.sh pull hf://owner/repo/model.gguf --revision main
./run.sh pull hf://owner/repo:q4_k_m       # pick a quantization from the repository
./run.sh pull hf://owner/repo/model.gguf --local-files-only   # offline: cache only
./run.sh pull --list                       # what is in the local cache
./run.sh pull --remove hf://owner/repo     # drop it again
```

Downloads resolve to immutable snapshots in the standard Hub cache, resume interrupted
transfers, fetch every shard of a GGUF file, and honour `FLLM_CACHE`, `HF_HOME`, `HF_HUB_CACHE`
and `HF_TOKEN`.

or convert a local checkpoint. The input must be a checkpoint directory (or a single
`.safetensors`/`.bin`/`.pt`/`.pth` file) with `config.json` and the tokenizer files beside the
weights:

```sh
./run.sh convert /models/hf-checkpoint /models/model-f16.gguf --outtype f16
./run.sh convert /models/model.safetensors /models/model.gguf --outtype preserve --overwrite
```

`--outtype` accepts `preserve` (default: keep the checkpoint dtypes), `f16`, `f32` or one of
the quantization presets (`q4_k_m`, `q8_0`, …), which converts and quantizes in a single run.
Conversion never loads the model into the inference runtime.

Inspect the result (`./run.sh inspect PATH` works for GGUF files and checkpoint directories
alike):

```sh
./run.sh inspect /models/model-f16.gguf
```

The output has one line per property — `GGUF: <path>`, `Version:`, `Metadata fields:`,
`Tensors:`, `Tensor types:` (each type with its GGML id and count) and `Architecture:`.

## 3. Quantize (optional)

```sh
./run.sh quantize /models/model-f16.gguf /models/model-q4_k_m.gguf q4_k_m
./run.sh quantize /models/model-f16.gguf /models/model-q8_0.gguf q8_0 --overwrite
```

Supported types: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_k_s`, `q4_k_m`, `q5_k_s`,
`q5_k_m`, `q6_k`. Float tensors are re-encoded; norms, routers and already quantized tensors
are copied unchanged. Details, including per-tensor mixes and GGUF split/merge, are in
[quantization.md](quantization.md).

## 4. Serve

```sh
./run.sh serve --model /models/model-q4_k_m.gguf --host 127.0.0.1 --port 8000 \
    --ctx 4096 --max-seqs 16 --max-batched-tokens 512 --threads 8
```

Startup prints the chosen settings and then:

```
fllm-rs listening on http://127.0.0.1:8000
```

`--model` accepts a file, a directory containing `.gguf` files (a file with `q4_k_m` in its
name is preferred, otherwise the first one) or an `hf://owner/repo/file.gguf` reference.
Every flag with its default is documented in [configuration.md](configuration.md).

Optional access control and served name:

```sh
./run.sh serve --model /models/model-q4_k_m.gguf --api-key secret --model-name my-model
```

## 5. Call it with curl

Health and model list:

```sh
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Chat completion:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

Streaming (server-sent events, terminated by `data: [DONE]`):

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"stream":true}'
```

Text completion:

```sh
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16}'
```

Embeddings:

```sh
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["first sentence","second sentence"]}'
```

If the server was started with `--api-key`, add `-H 'Authorization: Bearer secret'` to every
request. The full route list, request fields and response fields are in
[server-api.md](server-api.md).

## 6. Interactive chat (optional)

`fllm-rs chat` is a small client for a running server; it keeps the history and sends it with
each turn:

```sh
./run.sh chat --url http://127.0.0.1:8000 --model my-model --max-new 256 \
    --temperature 0.7 --top-p 0.9
```

Defaults: `--url http://127.0.0.1:8000`, `--model fllm-model`, `--max-new 512`,
`--temperature 0.7`, `--top-p 0.9`; `--api-key` defaults to `$FLLM_API_KEY`.
Exit with Ctrl-D.

A request and latency check against the same endpoint:

```sh
./run.sh bench --url http://127.0.0.1:8000 --requests 16 --concurrency 1 --input-len 256
```

`bench` defaults: `--requests 16`, `--concurrency 1`, `--input-len 256`, `--max-new 128`,
`--model fllm-model`; the requests it sends use `temperature: 0`, `stream: false`. It prints
request count and concurrency, elapsed time, requests/s, generated tokens and tokens/s,
prompt tokens and prefill rate, and mean/p50/p95 latency. `--input-len` counts repeated prompt
words, not tokenizer tokens; use the server's `usage` object for real token counts.

## 7. Embedding models (optional)

A BERT GGUF file starts the dedicated encoder server instead of the decoder:

```sh
./run.sh serve --model /models/bge-small.gguf --port 8001 --pooling cls
curl -s http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":"a sentence to embed"}'
```

`--pooling` accepts `cls`, `mean` and `last`; without it the value comes from the GGUF
`bert.pooling_type` metadata. This mode exposes `/health`, `/v1/models`, `/v1/embeddings`,
`/api/embed`, `/api/embeddings`, `/tokenize` and `/detokenize` only — the generation
endpoints do not exist here. `--embeddings`/`--pooling` on a decoder model is rejected.

## Next steps

- All HTTP routes and payloads: [server-api.md](server-api.md)
- Every `serve` flag, default and environment variable: [configuration.md](configuration.md)
- Quantization types, mixes, split/merge: [quantization.md](quantization.md)
- Supported model families: [architectures.md](architectures.md)
- Docker and compose: [docker.md](docker.md)
- Adding a new architecture: [adding-a-model.md](adding-a-model.md)
