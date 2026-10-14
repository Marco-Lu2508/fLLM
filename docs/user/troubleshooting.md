# Troubleshooting

Symptoms, causes and fixes. Exact messages come from
`rust/engine/src/bin/serve.rs`, `rust/engine/src/devices.rs`,
`rust/engine/src/backend/wgpu_backend.rs`, `rust/engine/src/gpu.rs`,
`rust/engine/src/encoder.rs`, `rust/engine/src/server.rs`, and from
[quickstart-windows-linux.md](../quickstart-windows-linux.md) (Docker GPU),
[docker.md](../docker.md), [configuration.md](../configuration.md).

## Device selection errors

Which string you see depends on the code path (see
[commands](commands.md#device-selection)):

| Message | Meaning / fix |
| --- | --- |
| `unsupported GPU backend X` | Decoder with the `rust-gpu` feature rejected `X`. Valid: `vulkan`, `metal`, `dx12`, `gpu`/`wgpu` (= all compiled backends), `auto`, `cuda` (CUDA builds only), `cpu`, `kind:index`. |
| `unsupported Rust GPU backend X` | BERT encoder / vision path (or the legacy GPU context) rejected `X` — including `cuda`, which the encoder never accepts, and `kind:N`, which that path does not split. Same valid set otherwise, bare kinds only. |
| `rebuild with --features rust-gpu for native Rust GPU projections` | Binary built without `rust-gpu` (decoder asked for GPU). Rebuild with the feature or use `--device cpu`. |
| `build with --features rust-gpu to enable the independent Rust GPU backend` | Same situation on the encoder path. |
| `invalid device index in '…'` | The `:N` in `kind:N` is not a number (decoder path). |
| `no native GPU adapter: …` | No adapter for the requested backend (missing driver, headless Vulkan, container without GPU). |
| `<name> is not a hardware GPU (set FLLM_ALLOW_SOFTWARE_GPU=1 to use it anyway)` | Decoder wgpu path refused a software adapter (lavapipe, WARP). Set `FLLM_ALLOW_SOFTWARE_GPU=1` to use it anyway (slower than CPU; meant for `devices`/`selftest-gpu` checks). |
| `<name> is not a hardware GPU; refusing silent CPU fallback` | Encoder/vision path (`gpu.rs`) refused a software adapter — this path honors no opt-out variable. Use `--device cpu`. |
| `the vulkan backend only supports the f16 and q8_0 KV cache (requested …)` | wgpu backends keep only `f16`/`q8_0`. Use `--device cuda` or `--device cpu` for other KV formats. |

`--device cuda` fails on purpose when no GPU is visible (e.g. `--gpus all`
missing) instead of silently running on the CPU. `--device auto` uses CUDA
when available; note its startup line reads `--device auto selects
CUDA/Metal/Vulkan when available, otherwise CPU`.

## Docker / GPU host problems

| Symptom | Cause / fix |
| --- | --- |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing (Linux) or Docker Desktop without WSL 2 backend / driver (Windows). |
| `docker run --gpus all ubuntu nvidia-smi` fails on Windows | Update the Windows NVIDIA driver, run `wsl --update`, restart Docker Desktop. |
| `fllm-rs devices` shows no CUDA device | Add `--gpus all`; check `nvidia-smi` on the host. |
| Runs but slowly | GPU not used or model does not fit VRAM (log shows layers offloaded); compare `--device cpu` vs `--device cuda`. |
| Container OOM | Smaller quantization, smaller `--ctx`, `--kv-cache-type q8_0`, `--gpu-memory-utilization 0.8`. |

Check what the container sees: `docker run --rm --gpus all <registry>/fllm:cuda devices`.

## Requests and HTTP errors

- `prompt + max_tokens` must fit `--ctx`; the prompt must encode to at least
  one token and every token id must be inside the vocabulary — otherwise `400`.
- `401 {"error":{…}}`: missing/wrong `Authorization: Bearer <key>` (Anthropic
  clients: `x-api-key`). Routes `/`, `/health`, `/metrics`, `/api/tags`,
  `/api/version`, `/get_model_info`, `/get_server_info` stay open.
- `404` on embeddings: the `model` field does not match the served model name
  (`--model-name` or the file stem).
- `429 inference request limit reached`: admission limit (128 concurrent
  requests) exhausted — retry later, raise client backoff, or lower fan-out.
- `503`: model worker unavailable (still starting, or a failed load).
- Bodies are capped at 16 MiB (2 MiB on the encoder server).
- `n: 1..16`; `n > 1` currently needs explicit CPU compute on a stateless
  model without draft/MTP and unconstrained text — GPU/stateful setups answer
  a validation error.
- Generation endpoints do not exist on the BERT encoder server (only
  `/health`, `/metrics`, `/v1/models`, `/v1/embeddings`, `/api/embed`,
  `/api/embeddings`, `/tokenize`, `/detokenize`, rerank/score/classify/pooling).
- Ollama model management (`/api/pull`, `/api/delete`, `/api/copy`) always
  answers `501` — fLLM serves one model per process. `/infill` answers `501`
  without FIM tokens in the GGUF.

## Models, downloads and files

- `GGUF selection is ambiguous`: a repository holds many files — add the
  quant suffix (`owner/repo:Q4_K_M`) or link the exact file.
- `directory '…' does not contain any .gguf models`: point `--model` at a GGUF
  file, or `convert` the checkpoint first.
- `unknown model_type` at convert: the architecture is not recognized —
  conversion refuses to guess (see [architectures.md](../architectures.md)).
- Download failures: connectivity, wrong repository/quant name, or a gated
  repository needing `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`. `--revision`,
  `--local-files-only`/`--offline` restrict Hub resolution to cache/revision.
- `… already exists` from `quantize`/`convert`/`merge-lora`/`gguf-split`:
  existing outputs are refused — pass `--overwrite` (where supported) or
  remove the file.
- Slow first token on big prompts with a quantized KV cache is expected when
  the generic attention kernel runs; the FP16 scratch path (default) is much
  faster (see [gpu-backends.md](../gpu-backends.md#kv-cache-formats)).

## Wrong output shape / sampling surprises

- The server default is greedy (`temperature 0.0`) unless `--temp` or the
  request says otherwise — silent-looking determinism is usually this.
- Tool calls not parsed: check `--tool-parser` (default `auto` picks from chat
  template/architecture); force one (e.g. `--tool-parser hermes`) to test.
  Reasoning leaking into text: same for `--reasoning-parser`.
- Quantized KV caches below 8 bits (and `fp8`) lose accuracy on models with
  large key outliers; keep keys at `q8_0` (`--cache-type-k q8_0`). `nvfp4`
  cache is experimental.
- `FLLM_CPU_ISA` caps the CPU kernel tier (e.g. `scalar`) to isolate a
  misbehaving SIMD path; `fllm-rs selftest-cpu [--tier TIER]` checks kernels
  against scalar references.

- CUDA decode uses CUDA graphs when the model allows it (the startup log says
  `CUDA graphs for decode steps: on`). If you suspect them for odd output or a
  failing capture, start with `FLLM_CUDA_GRAPH=0`: same text, slower; please
  report the difference.

## Getting help from the binary

```sh
fllm-rs --help            # usage summary (note: --max-batched-tokens 1024 there is stale; code default is 512)
fllm-rs devices           # compiled backends, detected devices, CPU ISA
fllm-rs selftest-cpu      # CPU kernels vs scalar references
fllm-rs selftest-gpu --device cuda    # every GPU op vs CPU reference (CUDA builds)
fllm-rs <command> --help  # per-command usage
```
