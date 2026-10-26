# Memory: VRAM and RAM planning

How fLLM sizes the GPU KV pool and layer offload, which flags move the
numbers, and what is (not) documented about CPU RAM. Sources:
[gpu-backends.md](../gpu-backends.md) (VRAM planner, offload, KV table),
[configuration.md](../configuration.md) (flag defaults),
[quickstart-windows-linux.md](../quickstart-windows-linux.md) (rule of thumb),
`rust/engine/src/bin/serve.rs` (code defaults).

## Server sizing flags and their code defaults

| Flag | Code default | Meaning |
| --- | --- | --- |
| `--ctx`, `--ctx-size`, `--max-model-len` | `4096` | Context window; `prompt + max_tokens` must fit |
| `--max-seqs` | `16` | Sequences scheduled concurrently |
| `--max-batched-tokens` | `512` | Token budget per model call (prefill chunk size) |
| `--threads` | `auto` | CPU worker threads (`auto` picks a count that suits the processor and measures it once) |
| `--gpu-memory-utilization` | `0.90` | Fraction of free device memory the planner may use |
| `--gpu-layers`, `-ngl`, `--n-gpu-layers` | `auto` | Layers fully on the device, or `auto` to fit the budget |

Note: the built-in usage line (`fllm-rs --help`) prints
`[--max-batched-tokens 1024]`, but the code default is **512**
(`serve.rs`: `let mut max_batched_tokens = 512usize`). The usage string is
stale; 512 is what you get when the flag is omitted. (The same usage line's
`--device` list is discussed in [commands](commands.md#device-selection).)

Two automatic adjustments apply when `--max-batched-tokens` is **not** given
explicitly: with partially offloaded layers it is raised to 2048 (host
weight streaming amortizes better over big batches), otherwise to 1024 when
below that. Pass the flag to take control.

## VRAM planner

Budget and requirement (`backend::memory`, see
[gpu-backends.md](../gpu-backends.md#4-vram-memory-planning--offloading-backendmemory)):

```text
Budget   = Free VRAM × --gpu-memory-utilization
Required = W(embeddings+head) + N(layers) × W(layer) + M(scratch) + M(KV cache)
```

- Weights come from the mapped GGUF tensor byte sizes.
- Scratch: `max_batched_tokens × (d_model + d_intermediate + d_vocab) × 4 bytes`.
- KV cache per layer: `block_size × kv_heads × (d_head + d_value) × bytes_per_elem`,
  where `bytes_per_elem` follows the cache format (keys and values counted
  separately, so a split `K q8_0 / V q4_0` cache plans exactly).

When everything does not fit, layers `0 … N-1` stay on the device (prefill
throughput matters most there) and the rest evaluate on the host CPU; the
experts of a mixture-of-experts layer count as feed-forward weights. Startup
logs the outcome, e.g. how many layers fit completely and the KV pool size in
blocks. With `--gpu-layers N` you fix N instead of letting `auto` maximize it.

Watch for these startup lines:

- `a context of {} tokens needs more KV cache than the device has left; using
  {} tokens` — the context was cut to fit; lower `--ctx`, use a smaller KV
  format (`q8_0` or `fp8_e4m3` roughly doubles what fits), or raise VRAM headroom.
- `KV pool limited to {} of {} blocks …` — the pool, not the context, was cut.
- `{} of {} layers fit on the device completely …` — the rest streams or runs
  on the host (see "Models larger than the card" below).

## Models larger than the card (CUDA)

`--gpu-layers auto` (default) keeps attention and every layer's KV cache on
the device and leaves only feed-forward weights of the overflowing layers in
host memory. Decode runs those layers on the CPU between device layers;
prompt processing (batches of ~32 tokens and up) streams the host weights to
the device over PCIe instead. Formats without a CUDA decoder (Q2_K, Q3_K, IQ
except IQ4_NL/IQ4_XS) are re-quantized to Q5_K at upload. Full numbers and
tuning switches (`FLLM_NO_HOST_STREAM`, `FLLM_HOST_STREAM_MIN`,
`FLLM_NO_HOST_PIN`): [gpu-backends.md](../gpu-backends.md#models-larger-than-the-card).

## KV format memory savings

Relative to an FP16 cache (see
[gpu-backends.md](../gpu-backends.md#kv-cache-formats) for the measured
accuracy/speed table): `fp8_e4m3`/`int8` ≈ 51 %, `q8_0` ≈ 53 %,
`q5_0` ≈ 34 %, `q4_0`/`iq4_nl`/`nvfp4` ≈ 28 %, `f32` = 200 %. The planner
counts exact per-token bytes per half, so `--gpu-memory-utilization` and the
fitted context follow the format automatically; the log prints the format and
block count (`paged <format> KV`).

## Agents and parallel requests

Coding agents and sub-agents send several requests at once that start with the same long
system prompt and tool list.

- **Batching.** All running requests share one decode step (continuous batching). Raise
  `--max-seqs` to the number of parallel sub-agents (default 16; each sequence costs KV
  memory, see above). Requests beyond it wait in the queue.
- **Prefix cache.** On by default (not for stateful/hybrid models and not for requests with
  images or audio). Whole KV blocks (16 tokens) of an identical prefix are reused; the last
  partial block and everything after the first differing token are computed again. The reply
  reports the reused tokens as `usage.prompt_tokens_details.cached_tokens`. Blocks enter the
  cache while the first request prefills, so requests that arrive after it has started reuse
  them; requests that are admitted in the very same step each compute the prefix once. For a
  burst, send one request first, then the rest. Keep the shared part byte-identical and put
  anything that varies (task text, time) at the end.
- **KV cache type.** `--kv-cache-type q8_0` or `fp8_e4m3` roughly halves KV memory, so about
  twice as many sequences or cached prefixes fit. Prefix reuse works with every cache format,
  because cached blocks are stored in the cache's own format and never converted.
- **Tool calls.** `"tool_call_grammar": true` ([server-api.md](../server-api.md#constrained-tool-calls))
  works with batching, the prefix cache and every KV type; it only adds a mask per token.
- To measure a setup, `scripts/agent_load_test.py` sends N parallel chat requests with a shared
  prefix and prints the time and tokens per second.

## Quick rule of thumb

Model file size + KV cache + about 1 GB overhead ≤ VRAM. A 12 GB card runs a
9B model at Q4_K_M with several thousand context tokens. Use
`--gpu-memory-utilization 0.85` to leave room for other programs (e.g. the
Windows desktop). Compose defaults match the code (`FLLM_MAX_MODEL_LEN=4096`,
`FLLM_MAX_SEQS=16`, `FLLM_MAX_BATCHED_TOKENS=512`; the CUDA compose file uses
`FLLM_CTX=8192`, `FLLM_KV_CACHE=f16`) — see [docker.md](../docker.md).

## CPU RAM: what is documented

The exact CPU RAM formula is **undocumented** — do not estimate from the VRAM
formula. What is confirmed:

- The GGUF file is memory-mapped; decoder embeddings and projection weights
  share the mapped bytes (no second F32 vocabulary table — only requested
  embedding rows are decoded).
- fLLM keeps no repacked second copy of the weights (prompt repacking unpacks
  ≤14 KB panels into transient per-worker scratch).
- Load-time LoRA merges increase memory (adapted matrices use F32 RAM).
- The Hub cache holds downloaded models (`FLLM_CACHE` / `HF_HOME` /
  `HF_HUB_CACHE`; `fllm-rs pull --list` shows what is cached).
- Batch uploads/results live under `FLLM_FILES_DIR` (default
  `<cache>/fllm-files`), capped by `FLLM_MAX_FILE_SIZE` (default 512 MB).
