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

## What it does not do

No queueing, no retries on client errors, no auth beyond the single `--api-key` it checks itself, no KV-cache-aware routing
beyond the prefix hash (a real router would ask the backends which prefixes they hold — that needs an endpoint on the server
side and is on the list).
