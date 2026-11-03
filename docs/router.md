# fllm-router

A small HTTP router in front of several fLLM servers. Its job is not load balancing in general, but keeping the **prefix
cache** useful across a fleet.

```bash
fllm-router --listen 0.0.0.0:8080 \
  --backend http://10.0.0.5:8000 \
  --backend http://10.0.0.6:8000 \
  --api-key "$(cat key)"
```

## How it routes

The router hashes up to 2048 characters of the stable beginning of the prompt — system message, then the first user message
(or the `prompt` of a completion) — and sends the request to the backend that hash points at. Same prefix, same backend,
cache hit. Different conversations still spread over the fleet.

A backend that answers 5xx or does not answer is marked unhealthy for 15 seconds and the request is retried once on another
backend. Streaming answers are passed through chunk by chunk (`stream: true` stays streaming), headers are copied, and the
chosen backend is in the `x-fllm-backend` response header.

`GET /router/stats` shows the backends, their health, their request and failure counts. `GET /router/metrics` is the same
in Prometheus format: `fllm_router_routed_total`, `fllm_router_backend_requests_total{backend=…}`,
`fllm_router_backend_failures_total{backend=…}` and `fllm_router_backend_healthy{backend=…}` (1 = used, 0 = skipped).

## Disaggregated prefill / decode

```sh
fllm-router --listen 0.0.0.0:8080 --prefill http://prefill-1:8000 --prefill http://prefill-2:8000 --decode http://decode-1:8000
```

With `--prefill` the servers given by `--decode` (or `--backend`) generate and the prefill servers only compute prompts. For a chat or completion
request the router (1) asks a prefill server for the token ids the request turns into (`/v1/chat/completions/render`, `/tokenize`), (2) lets it
prefill them (`max_tokens` 1), (3) tells it to push the KV blocks of the prompt to the chosen decode server (`/admin/disagg/export` with the decode
server as `target`, which the decode server takes at `/admin/disagg/import`), then (4) sends the original request to the decode server, which
finds the prompt in its prefix cache and computes only the last partial block. Both choices use the prefix hash of the routing above. If any
step fails the decode server prefills itself: same answer, slower. The reply carries `x-fllm-disagg: transferred` or `fallback`, and
`/router/stats` counts both. Requirements: same model, block size and KV format on both sides (the import checks a fingerprint and refuses
otherwise), no `FLLM_PREFIX_SALT` and no `cache_salt` (blocks under another identity would never be reused), the same API key everywhere. The
blocks travel through host memory over HTTP; a shared directory (`dir:/dev/shm/kv` as `target`) is the other transport (`kv_transfer.rs`), an
RDMA transport is not built. Tested with the CPU runtime ([datacenter-gap.md](datacenter-gap.md)); not on GPUs.

## What it does not do

No queueing, no retries on client errors, no auth beyond the single `--api-key` it checks itself, no KV-cache-aware routing
beyond the prefix hash (a real router would ask the backends which prefixes they hold — that needs an endpoint on the server
side and is on the list).
