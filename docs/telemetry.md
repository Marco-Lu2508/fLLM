# Telemetry

fLLM exports one span per finished answer over **OTLP/HTTP** as JSON — no OpenTelemetry SDK, no extra dependency.

```bash
FLLM_OTEL_ENDPOINT=http://127.0.0.1:4318 fllm-rs serve --model model.gguf
```

| variable | default | meaning |
| --- | --- | --- |
| `FLLM_OTEL_ENDPOINT` | empty (off) | collector base URL, spans go to `{endpoint}/v1/traces` |
| `FLLM_OTEL_SERVICE` | `fllm` | `service.name` of the resource |

Span `chat.completions` / `completions` with `model`, `prompt_tokens`, `stream` and — for a cache hit — `cache=hit` and
`completion_tokens`. Errors carry `status.code = 2`.

Spans are queued and sent in batches of 128 (or every two seconds) by one background thread, so a slow collector never
slows a request down; if the queue overflows the oldest spans are dropped. `GET /admin/telemetry` shows whether the
exporter is on and how many spans were sent.

A span attribute is a key/value pair: `{"key": "model", "value": {"stringValue": "llama"}}`. The counters move only when the
collector answers 2xx — a refused batch is not reported as sent.

**Not instrumented yet:** the streaming path (a stream's end is not observed) and the encoder server. Both are one hook
each away.

## What `/metrics` carries

Beyond the model counters: `fllm_output_cache_*` (hits, misses, evictions, bytes), `fllm_draining` (1 while new work is
refused), `fllm_coalescing_leaders_total` / `fllm_coalesced_requests_total` (requests that computed for others / waited),
and `fllm_otel_spans_sent_total`. One scrape answers "is this instance healthy, busy, and answering from cache".

## Inspecting the prefix cache

`GET /admin/prefix-cache` answers with `enabled`, `entries`, `hits`, the blocks (`hash`, `tokens`) and the radix `tree`.
Live example from a tiny model after one request: four entries of 16 tokens, each with the chain hash that identifies it.
This is what makes a "why was my prompt not reused" question answerable instead of guessed. `DELETE /admin/prefix-cache`
empties it and gives the blocks back to the allocator — useful after a template change, when the old blocks are dead weight.

## The disk tier

`GET /admin/kv-tier` reports `enabled`, `dir`, `max_bytes`, `used_bytes`, `spills`, `hits`, `misses` and `drops`;
`DELETE /admin/kv-tier` removes every stored block. Both go through the inference worker, like the prefix cache.

## Rate limits already exist

Before building anything new: per-key limits are in the server already —
`RateLimitSpec { requests_per_minute, tokens_per_minute }` with a token bucket per API key in
`server/security.rs`, answered with `429` and `Retry-After`. What was missing is documentation, not code.
