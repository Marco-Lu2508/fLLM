# Commands: the `fllm-rs` CLI

Every subcommand with its usage string, from `rust/engine/src/cli.rs`
(`help()`), `rust/engine/src/bin/serve.rs` (`usage()`),
`rust/engine/src/bench.rs` (`HELP`), `rust/engine/src/cpu_selftest.rs`
(`HELP`) and `rust/engine/src/ppl.rs`. `scripts/gen-cli-reference.sh`
regenerates `docs/cli-reference.md` from these same strings
(`BIN=… bash scripts/gen-cli-reference.sh [OUT]`).

Conventions: `fllm-rs` with no arguments prints the usage summary; the
`serve` keyword is optional (`fllm-rs serve --model x.gguf` ≡
`fllm-rs --model x.gguf`); unknown subcommands/options exit non-zero with the
usage text. All serve flags with defaults: [configuration.md](../configuration.md).

## Serve (summary)

```text
fllm-rs [serve] --model MODEL.gguf [--device auto|cpu|cuda|metal|vulkan|dx12|device:N] …
```

Key code defaults: `--host 127.0.0.1`, `--port 8000`, `--ctx 4096`,
`--max-seqs 16`, `--max-batched-tokens 512`, `--threads` = all cores,
`--device auto`, `--gpu-memory-utilization 0.90`, `--gpu-layers auto`,
`--kv-cache-type f16`, `--speculative-lookup 0`, `--speculative-window 32`,
`--draft-tokens 5`, `--tool-parser auto`, `--reasoning-parser auto`.

Other serve flags
(`--gpu-layers`, `--speculative mtp|off`, `--pin-cores`, `--no-repack`,
sampling `--temp`/`--top-p`/`--top-k`/`--min-p`/…, `--lora`,
`--draft-model`, `--embeddings`/`--pooling`, `--mmproj`, `--api-key`,
`--model-name`) behave as [configuration.md](../configuration.md) documents.

## Device selection

`--device` is lowercased; `serve.rs` only logs a note for `auto` (`--device
auto selects CUDA/Metal/Vulkan when available, otherwise CPU`) and `cpu`
(`Using explicit CPU runtime`) and passes everything else to backend
selection. The true accepted set is decided downstream:

| Value | Decoder (`--model …gguf`) | BERT encoder / vision (`--mmproj`) |
| --- | --- | --- |
| `auto` (default) | First available CUDA → Metal → Vulkan → DX12, else CPU | CPU (never picks a GPU) |
| `cpu` | CPU | CPU |
| `cuda`, `cuda:N` | CUDA backend — **only in `--features cuda` builds** (CUDA image, `compose.cuda.yml`) | Rejected (`unsupported Rust GPU backend cuda`) |
| `vulkan`, `metal`, `dx12`, `gpu`/`wgpu`, `kind:N` | wgpu backend (`rust-gpu` builds); `gpu`/`wgpu` = all compiled backends | Bare kinds accepted (`rust-gpu` builds); `kind:N` is **not** split on this path (`unsupported Rust GPU backend kind:N`) |
| anything else | `unsupported GPU backend X` | `unsupported Rust GPU backend X` |

Without the `rust-gpu` feature a GPU request fails with
`rebuild with --features rust-gpu for native Rust GPU projections` (decoder)
or `build with --features rust-gpu to enable the independent Rust GPU
backend` (encoder). The CPU image builds with `rust-gpu` (no CUDA), so there
`cuda` is rejected while `vulkan`/`metal`/`dx12` parse; the CUDA image builds
`rust-gpu,cuda`, so `cuda` works there.

Two repo docs disagree here, and the code sides as follows:
[configuration.md](../configuration.md) ("the parser does not accept `cuda`")
is correct for non-CUDA builds and the encoder path, but incomplete for CUDA
builds, where the decoder accepts `cuda`/`cuda:N` and `auto` prefers CUDA —
which is what the usage line's `cuda` describes. `device:N` is generic on the
decoder path: `select_device` splits any `kind:index` (the index must parse
as a number); the kind is validated by the backend as in the table. The
encoder/vision path (`encoder.rs`, `vision.rs` → `gpu::GpuContext::new`)
takes the raw string with no splitting.

## wgpu scope (narrower verified claim)

[gpu-backends.md](../gpu-backends.md) documents the full wgpu operation set
(native: quantized linear for all 11 formats, embedding lookup, RMSNorm,
RoPE, SwiGLU/GeGLU, residual add, KV writes and paged attention for `f16` /
`q8_0` with head width ≤ 512; host fallback: LayerNorm, MoE, sampling logits,
per-layer architecture fallbacks, recurrent mixers). This matches the code
(`supports_paged_attention: true`, `selftest-gpu` coverage).

[inference-features.md](../inference-features.md) instead says LayerNorm,
attention, activation and cache handling "stay on the Rust CPU" — that
sentence is **stale** relative to the backend doc and the code. What remains
authoritative in `inference-features.md` is the hardware claim: physically
validated on Intel Arrow Lake Vulkan only; NVIDIA/AMD/Apple-Metal/DX12
hardware unvalidated. Verified scope for users: the operation set of
`gpu-backends.md` on Arrow Lake Vulkan; anywhere else, run
`fllm-rs selftest-gpu [--device vulkan]` plus a short greedy
`--device cpu` vs `--device vulkan` comparison before trusting output.

Other constraints: software adapters refused unless
`FLLM_ALLOW_SOFTWARE_GPU=1`; wgpu KV cache limited to `f16`/`q8_0` (startup
error otherwise); `--device auto` on the encoder path stays on CPU.

## Model cache and catalog

```text
fllm-rs pull owner/repo[:QUANT] [--revision REV] [--local-files-only]
       fllm-rs pull owner/repo [--quant TYPE] [--outtype preserve|f16|f32] [--no-convert]
       fllm-rs pull hf://owner/repo/file.gguf [--revision REV] [--local-files-only]
       fllm-rs pull --list
       fllm-rs pull --remove hf://owner/repo[:QUANT]
fllm-rs list        # complete models in the local cache (reference, bytes, path; converted files with their origin)
fllm-rs rm MODEL [--converted]   # removes the cached repository (--converted: only the GGUF files made from a checkpoint); a local directory removes its conversions
fllm-rs ps          # servers recorded in the status dir (pid, port, model, host)
fllm-rs show MODEL  # path, size, GGUF architecture + selected metadata; a checkpoint: architecture, quantization, converted GGUF files
fllm-rs search [QUERY]  # catalog names/repos/default quants (empty query lists all)
```

`run` downloads on first use, serves on a free local port, chats in the
terminal, and stops the server on quit (Ctrl-D):

```text
fllm-rs run MODEL [--quant TYPE] [--outtype preserve|f16|f32] [--revision REV] [--local-files-only] [serve options]
MODEL: https://huggingface.co/owner/repo, .../blob/main/file.gguf,
hf.co/owner/repo:Q4_K_M, owner/repo, hf://owner/repo[:QUANT], a local .gguf
or a local checkpoint directory.
```

Repos with many GGUF files need the `:QUANT` suffix (`GGUF selection is
ambiguous` otherwise). `FLLM_CACHE` overrides the Hub cache location.

**Checkpoints.** A repository or directory without GGUF files but with a safetensors or
PyTorch checkpoint (plain, AWQ, GPTQ, FP8, compressed-tensors int4/int8, NVIDIA ModelOpt
NVFP4) is converted on first use by `run`, `serve --model` and `pull`; the same options
work for `serve`:

```sh
fllm-rs run nvidia/Llama-3.1-8B-Instruct-NVFP4            # ModelOpt NVFP4 -> GGUF type 40
fllm-rs run Qwen/Qwen2.5-7B-Instruct-AWQ                  # stays 4-bit (Q4Z)
fllm-rs run meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m   # bf16 checkpoint -> Q4_K_M
fllm-rs serve --model ./my-checkpoint --outtype f16
fllm-rs pull owner/repo --quant q8_0                      # convert only, print the GGUF path
```

stderr shows the download size and "converting ... this happens once"; the GGUF is cached
next to the downloaded files (see `list`), so the next start only loads it. `--quant TYPE`
(`q4_0 q4_1 q5_0 q5_1 q8_0 q4_k_s q4_k_m q5_k_s q5_k_m q6_k`) applies to unquantized
checkpoints only, `--outtype` to the 16/32-bit type. An unsupported architecture, a missing
`tokenizer.json` or a gated repository without `HF_TOKEN` is reported before the weights are
downloaded. `pull --no-convert` only downloads the checkpoint (the same download as
`convert hf://owner/repo OUT.gguf`). Details: [rust-formats-and-import.md](../rust-formats-and-import.md).

## Convert, quantize, inspect, formats

```text
fllm-rs convert CHECKPOINT_PATH OUTPUT.gguf [--outtype preserve|f16|f32|TYPE] [--overwrite] [--revision REV] [--local-files-only]
fllm-rs quantize INPUT.gguf OUTPUT.gguf TYPE [--overwrite]
  TYPE: q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k.
fllm-rs inspect PATH [--revision REV] [--local-files-only]   # GGUF file or dense checkpoint dir
fllm-rs formats     # registered GGML storage types + decoder class (not an inference promise)
fllm-rs ppl --model M --file text.txt [--ctx 2048] [--compare-to OTHER]
```

Details (mixes, fallbacks, requant rule, AWQ/GPTQ): [quants-and-formats](quants-and-formats.md).

## GGUF tools

```text
fllm-rs merge-lora BASE.gguf ADAPTER OUT.gguf [--scale S] [--outtype f16|f32|TYPE]
fllm-rs gguf-set KEY=VALUE ... IN.gguf OUT.gguf
fllm-rs gguf-split IN.gguf --max-size 4G OUT-PREFIX
fllm-rs gguf-merge FIRST-SHARD.gguf OUT.gguf
```

`ADAPTER` is a PEFT directory or safetensors file beside
`adapter_config.json` (defaults: scale 1, outtype f16). Split suffixes
`K`/`M`/`G`/`T` are powers of 1024 and count metadata/padding; merge takes
the first `…-00001-of-0000N.gguf` shard. Existing outputs are refused.

## Chat and bench clients

```text
fllm-rs chat [--url http://127.0.0.1:8000] [--model MODEL_ID] [--api-key KEY] [--max-new N] [--temperature T] [--top-p P]
```

Interactive chat against a running server (history kept per session, Ctrl-D
to exit). Defaults: `--url http://127.0.0.1:8000`, `--model fllm-model`,
`--max-new 512`, `--temperature 0.7`, `--top-p 0.9`; `--api-key` defaults to
`$FLLM_API_KEY`.

```text
fllm-rs bench [OPTIONS]
  --url URL                 Server root or /v1 (default http://127.0.0.1:8000)
  --model ID                Default: first model reported by GET /v1/models
  --api-key KEY             Or FLLM_API_KEY; never written to results
  --endpoint chat|completions (default chat)
  --scenarios LIST          prefill,decode-single,decode-context,concurrency
  --prompt-lens LIST        128,512,2048,8192
  --context-lens LIST       1024,4096,16384
  --concurrency LIST        1,2,4,8,16,32
  --gen-len N               256 (alias --max-new)
  --input-len N             Short/concurrency prompt target: 64
  --requests N              Concurrency requests/repeat; default N simultaneous streams
  --shared-prefix           Share 90% of concurrency prompt words
  --warmup N                1 (zero allowed)
  --repeats N               3
  --seed N                  42
  --timeout SECONDS         Entire HTTP request deadline: 600
  --length-control MODE     auto|none|min-tokens|ignore-eos|both (default auto)
  --label TEXT --notes TEXT --json PATH
```

Schema, definitions and comparison limits: `bench/README.md`. Note:
[getting-started.md](../getting-started.md) §6 describes an older bench
(`--requests 16`, `--concurrency 1`, …); the `HELP` text above is
authoritative for this checkout.

```text
fllm-rs bench-report A.json B.json [C.json ...]
fllm-rs bench-report --baseline BASE.json --current NEW.json --max-regression 7
```

Prints Markdown; regression mode fails on missing/invalid cases, request
errors, incompatible workloads, or a primary rate dropping more than the
limit.

## Devices, selftests, completions

```text
fllm-rs devices            # compiled backends, detected devices, CPU ISA
fllm-rs selftest-cpu [--tier TIER | --tiers all|TIER,TIER] [--full] [--gguf FILE [--tensors N]]
  # the tiers up to the FLLM_CPU_ISA cap (or only TIER) vs scalar references; non-zero exit on excess error.
  # --tiers all: every supported tier, cap ignored, plus ragged shapes and a saturation stress for every format;
  # --gguf: real tensors of a model with outlier activations.
  # TIER: scalar, avx2, avx-vnni, avx512bw, avx512-vnni, avx512-bf16, amx (the last two need a build with the
  #       avx512 feature), neon, neon-dotprod, neon-i8mm, neon-sve
fllm-rs selftest-gpu [--device vulkan]   # every GPU op vs CPU reference (see gpu-backends.md §7)
fllm-rs completions [bash|zsh|fish|powershell]   # shell completions (default bash)
```

## See also

- Every serve flag and env var: [configuration.md](../configuration.md)
- Sizing: [memory](memory.md) — failures: [troubleshooting](troubleshooting.md)
- Talking to the server: [clients](clients.md)
- Weight formats: [quants-and-formats](quants-and-formats.md)
