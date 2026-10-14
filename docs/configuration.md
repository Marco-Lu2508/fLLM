# Server configuration

All `fllm-rs serve` options with their exact flag names and defaults, taken from
`rust/engine/src/bin/serve.rs`. The `serve` keyword itself is optional — `fllm-rs serve
--model x.gguf` and `fllm-rs --model x.gguf` are the same command — and `fllm-rs` with no
arguments prints the usage summary.

```sh
fllm-rs serve --model MODEL.gguf [--device auto|cpu|vulkan|metal|dx12] ...
fllm-rs devices            # compiled backends, detected devices, CPU ISA
fllm-rs selftest-cpu       # check every CPU kernel tier against scalar references (docs/cpu-kernels.md)
fllm-rs --help             # usage summary
```

## Core options

| Flag | Value | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | path or `hf://owner/repo[/file.gguf]` | required | Model to load. A directory is scanned for `*.gguf` (a name containing `q4_k_m` wins, otherwise the first file); a directory without GGUF files is an error that points at `convert` |
| `--host` | string | `127.0.0.1` | Listen address |
| `--port` | integer | `8000` | Listen port |
| `--model-name`, `--served-name` | string | model file stem | Model id reported by `/v1/models`, `/api/tags` and responses |
| `--api-key` | string | value of `FLLM_API_KEY`, if set | Requires `Authorization: Bearer <key>` on all routes except `/health`, `/metrics`, `/api/tags`, `/api/version`, `/get_model_info`, `/get_server_info` |
| `--tool-parser` | `auto` or a parser id | `auto` | How tool calls are read from the reply: `hermes`, `qwen3_coder`, `glm`, `minicpm`, `llama3`, `mistral`, `deepseek`, `deepseek_v32`, `deepseek_v4`, `kimi`, `gpt_oss`, `gemma4`, `functiongemma`, `minimax`, `minimax_m1`, `granite3`, `phi4`, `lfm2`, `k2horizon`, `none`. `auto` picks from the chat template and architecture; see [Tools und Reasoning](server-api.md#tools-und-reasoning) |
| `--reasoning-parser` | `auto` or a format id | `auto` | How reasoning is split from the answer: `think`, `ifm`, `mistral`, `seed`, `minimax_m3`, `gemma4`, `gpt_oss`, `none`. `auto` picks from the chat template, default `think` |
| `--no-repack` | flag | off | CPU only: prompts use the streaming kernels instead of the prompt-time repacking (Q4_0/Q5_0 into Q8_0 tiles, K formats on decoded panels). Nothing is stored either way; the switch exists for A/B measurements. `FLLM_NO_REPACK=1` does the same |
| `--ctx`, `--ctx-size`, `--max-model-len` | integer | `4096` | Context window; `prompt + max_tokens` must fit into it |
| `--max-seqs` | integer | `16` | Sequences scheduled concurrently |
| `--max-batched-tokens` | integer | `512` | Token budget per model call (also the prefill chunk size) |
| `--threads` | integer | all available cores | CPU worker threads (`std::thread::available_parallelism`) |
| `--device` | string (lowercased) | `auto` | `auto` and `cpu` select the CPU. `vulkan`, `metal`, `dx12` (also `gpu`/`wgpu` = all compiled backends) need the `rust-gpu` feature; a software adapter is refused. Any other value fails with `unsupported Rust GPU backend …` (or, without the feature, `rebuild with --features rust-gpu …`) — including `cuda`, which the built-in usage line mentions but the parser does not accept |
| `-C`, `--pin-cores`, `--p-cores`, `--pin-mask` | string | none | Linux only: pin the process to a CPU affinity mask — `8` (first 8 cores), `0,2,3` (core list) or `0xFF5` (hex mask) |

`--revision REV`, `--local-files-only` and `--offline` apply to `hf://` model references:
`REV` selects a Hub revision (default `main`), the boolean flags restrict resolution to the
local Hub cache.

## Cache and speculation

| Flag | Value | Default | Meaning |
| --- | --- | --- | --- |
| `--kv-cache-type`, `--kv-cache-dtype` | string (lowercased) | `f16` | KV cache format of keys and values: `f16` (`auto`), `bf16`, `fp8_e4m3` (`fp8`), `fp8_e5m2`, `int8`, `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl`, `nvfp4` (experimental), `f32`; see [KV cache formats](gpu-backends.md#kv-cache-formats) |
| `--cache-type-k`, `--cache-type-v` | string (lowercased) | the common type | Format of the keys / of the values alone, like llama.cpp (`-ctk`, `-ctv`); overrides `--kv-cache-type` for that half |
| `--speculative-lookup`, `--speculative`, `-sp` | integer | `0` | Prompt-lookup speculative decoding; `0` disables it |
| `--speculative-window`, `-sw` | integer | `32` | Lookahead window used with `--speculative-lookup` |
| `--draft-model`, `--model-draft` | path | none | Second GGUF used as draft model; token ids and BOS/EOS metadata must match the target |
| `--draft-tokens` | integer | `5` | Candidate tokens verified per step against the draft model |

Speculative decoding and LoRA apply to decoder models only; passing them with a BERT encoder
is rejected.

## Adapters and encoder/vision modes

| Flag | Value | Default | Meaning |
| --- | --- | --- | --- |
| `--lora` | path | none (repeatable) | Merge a PEFT directory or `.safetensors` adapter at load time with scale `1.0` |
| `--lora-scaled` | path + scale | none (repeatable) | Same, with an explicit scale: `--lora-scaled adapter.safetensors 0.5` |
| `--embeddings` | flag | off | Accepted for BERT models (the encoder server is selected by `general.architecture == bert`); rejected for decoder models |
| `--pooling` | `cls` \| `mean` \| `last` | GGUF `bert.pooling_type` | Pooling for the BERT encoder; rejected for decoder models |
| `--mmproj` | path | none | Vision projector for image chat; hidden size and `<image>` template markers must match the decoder |

## Default sampling values

These flags set server-wide defaults used when a request omits the corresponding field; a
request value always wins. `--min-p` and friends are stored as the sampling defaults for all
requests (see [server-api.md](server-api.md) for the accepted ranges).

| Flag | Value | Default |
| --- | --- | --- |
| `--temp`, `--temperature` | number | unset → `0.0` when a request does not send `temperature` |
| `--top-p` | number | unset → `1.0` |
| `--top-k` | integer | unset → `0` (disabled) |
| `--min-p` | number | `0.0` |
| `--typical-p` | number | `1.0` |
| `--presence-penalty` | number | `0.0` |
| `--frequency-penalty` | number | `0.0` |
| `--repeat-penalty` | number | `1.0` |
| `--repeat-last-n` | integer | `-1` |
| `--dry-multiplier` | number | `0.0` |
| `--dry-base` | number | `1.75` |
| `--dry-allowed-length` | integer | `2` |
| `--dry-penalty-last-n` | integer | `-1` |
| `--mirostat` | `0` \| `1` \| `2` | `0` |
| `--mirostat-tau` | number | `5.0` |
| `--mirostat-eta` | number | `0.1` |
| `--seed` | integer | unset (no fixed seed) |

Values are parsed as JSON numbers, so `--seed 42`, `--mirostat 2` and `--dry-multiplier 0.5`
are valid; anything else exits with `invalid number for <flag>`.

## Compatibility flags (accepted, no effect)

| Flag | Behaviour |
| --- | --- |
| `--engine auto\|rust` | Only `auto` and `rust` are accepted; the Rust engine is the only one |
| `--compute rust\|auto` | Accepted and ignored. `--compute native` was removed and exits with an error |

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `FLLM_API_KEY` | `serve` (and the `chat`/`bench` clients as their default `--api-key`) | Initial value of `--api-key`; the flag overrides it |
| `FLLM_RUST_PROFILE` | Rust decoder | When set (any value), a traced forward call prints one `{"rust_cpu_profile":{"rows":…,"phases_ms":{…}}}` line (total rows plus per-phase milliseconds) to stderr |
| `FLLM_CUDA_GRAPH` | CUDA decoder | CUDA graphs of decode steps: on by default for eligible models; `0`, `off`, `false` or `no` switch them off, `1`, `on` or `true` ask for them explicitly (an ineligible model still decodes without graphs); any other value is the default. See docs/gpu-backends.md |
| `FLLM_FUSE`, `FLLM_FUSE_ADD_NORM`, `FLLM_FUSE_NORM_QUANT`, `FLLM_FUSE_ROPE_KV`, `FLLM_FUSE_GLU_Q8`, `FLLM_FUSE_HEAD_COPY` | CUDA decoder | Fused decode kernels (residual add + norm + activation quantization, rotary + cache write, gated activation + int8 quantization, final-norm copy); all on by default, `0`, `off`, `false` or `no` switch one off; a switch that is not set follows `FLLM_FUSE` (`FLLM_FUSE=0` turns all of them off, and `FLLM_FUSE=0 FLLM_FUSE_ROPE_KV=1` runs one alone). See docs/gpu-backends.md |
| `FLLM_MARLIN_CFG` | CUDA Marlin | `K,N,T` (thread_k, thread_n, threads, e.g. `128,64,128`): every Marlin call tries that thread-block configuration first and uses it where it is valid for the shape; unset, `vllm`, `fllm` or `auto` keep vLLM's choice. A tuning aid for decode (docs/awq-gptq-marlin.md) |
| `FLLM_PREFILL_STEP_CACHE` | CUDA prompt pass | Host work once per forward pass instead of once per layer (block tables, row layout, Marlin launch attributes). Default on; `0`, `off` or `false` switch it off. See docs/performance-notes.md |
| `FLLM_PREFILL_PAR_EMBED` | CUDA prompt pass | Embedding rows of a prompt are looked up on several threads. Default on; `0` switches it off |
| `FLLM_FUSE_PREFILL_ADD_NORM` | CUDA prompt pass (Marlin 4-bit, 16 rows or more) | Residual add of the output projection done inside the next RMS norm. Default on; `0`, `off` or `false` switch it off |
| `FLLM_FUSE_PREFILL_NORM_F16` | CUDA prompt pass (Marlin 4-bit, 16 rows or more) | RMS norm also writes the FP16 copy the next Marlin GEMM reads. Default on; `0`, `off` or `false` switch it off |
| `FLLM_FUSE_PREFILL_ATTN_F16` | CUDA prompt pass (Marlin 4-bit, FlashInfer attention) | Attention output also written as FP16 for the output projection. Default on; `0`, `off` or `false` switch it off |
| `FLLM_MARLIN_PREFILL_CFG` | CUDA Marlin, prompt batches | `fllm` runs more than 64 rows as whole 64-row tiles in one launch; anything else (default, `vllm`) keeps vLLM's split into launches. Independent of `FLLM_MARLIN_CFG` |
| `FLLM_NVFP4_ACT_SCALE` | NVFP4 GEMMs (sm_120) | Per-tensor activation factor of the checkpoint: `static` (default, also `auto`) uses `input_scale` stored with the weight, `dynamic` uses amax of each activation tensor, `off` (or any other value) uses 1 as before the switch existed. Also read by the fused decode norm (`dynamic` makes it decline the quantization). See docs/gpu-backends.md |
| `FLLM_NVFP4_TILE` | NVFP4 prompt GEMMs (sm_120, CUTLASS) | `legacy` (default; 128x128x128 up to 256 tokens, 128x128x256 above), `auto` (cost model) or a tile: `256x128x128`, `128x256x128`, `128x128x128`, `128x128x256`, `128x128x128-sk`, `128x128x256-sk`, `128x128x128-pp`, `128x64x256` |
| `FLLM_NVFP4_TUNE` | NVFP4 prompt GEMMs (sm_120, CUTLASS) | `1` times every tile once per shape at first use and keeps the fastest; default off |
| `FLLM_FUSE_NVFP4_NORM` | NVFP4 prompt GEMMs (sm_120, CUTLASS, more than 64 rows) | RMS norm and the activation quantization of the next CUTLASS GEMM in one kernel. Default off; any value but `0` switches it on |
| `FLLM_FUSE_NVFP4_ATTN_OUT` | NVFP4 prompt GEMMs (sm_120, FlashInfer) | FP16 attention output quantized directly by the output projection. Default off; any value but `0` switches it on |
| `FLLM_CPU_ISA` | CPU kernels | Cap kernel dispatch at a tier: `scalar`, `avx2`, `avx-vnni`, `avx512bw`, `avx512-vnni` (x86-64) or `neon`, `neon-dotprod`, `neon-i8mm` (AArch64); see docs/cpu-kernels.md |
| `FLLM_CACHE` | `hub` resolver | Hub cache root; takes precedence over `HF_HUB_CACHE` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | `hub` resolver | Bearer token for `hf://` downloads (first match wins) |
| `HF_HUB_CACHE` | `hub` resolver | Hub cache root when `FLLM_CACHE` is unset; default `$HF_HOME/hub` |
| `HF_HOME` | `hub` resolver | Default `$XDG_CACHE_HOME`/`$HOME/.cache`, then `huggingface` |
| `XDG_CACHE_HOME`, `HOME`, `USERPROFILE` | `hub` resolver | Fallbacks for the cache location when `HF_HOME` is unset |
| `FLLM_FILES_DIR` | batch store | Directory for uploaded batch files, results and `store.json`; default `<cache>/fllm-files` (cache root as resolved above) |
| `FLLM_MAX_FILE_SIZE` | batch store | Maximum upload size in bytes (default `536870912`, 512 MB); larger uploads are rejected with `413` |
| `FLLM_BATCH_CONCURRENCY` | batch store | How many JSONL lines of a batch run at once (default `2`); keep it low so interactive traffic is not crowded out |

The batch variables are read once, on the first upload or batch request. CLI-Flags folgen, sobald
`serve.rs` frei ist — `--files-dir`, `--max-file-size` and `--batch-concurrency` will replace them.

Interaction rules:

- `--api-key` beats `FLLM_API_KEY`; if neither is set, all routes are open.
- Request fields beat `--temp`/`--top-p`/`--top-k`; those beat the built-in defaults
  (`temperature 0.0`, `top_p 1.0`, `top_k 0`).
- `--local-files-only`/`--offline` makes `hf://` resolution fail unless the revision is
  already in the local cache, regardless of the token variables.
- The Docker image passes `FLLM_API_KEY` through and adds `FLLM_*` compose variables on top
  of the command line — see [docker.md](docker.md).

## Examples

```sh
# small local server, fixed sampling defaults
fllm-rs serve --model /models/model.gguf --ctx 8192 --max-seqs 8 --threads 6 \
    --temp 0.7 --top-p 0.95 --pin-cores 0,1,2,3,4,5

# quantized KV cache + prompt-lookup speculation
fllm-rs serve --model /models/model.gguf --kv-cache-type q8_0 --speculative-lookup 4

# draft model + LoRA adapter + API key
fllm-rs serve --model /models/model.gguf --draft-model /models/draft.gguf \
    --draft-tokens 5 --lora /models/adapter.safetensors --api-key secret

# BERT embedding server
fllm-rs serve --model /models/bge-small.gguf --pooling cls --port 8001
```
