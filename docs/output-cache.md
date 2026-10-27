# The output cache

A request whose answer is already determined — `temperature = 0`, one choice, no streaming, no tools, no images — does not
have to be computed twice. The output cache keeps the answer and hands it out again, including a replay of the token stream
for streaming clients.

## What the key is

SHA-256 over everything that changes the text:

- the model and template identity (computed once at startup; a different model, a different chat template or a different
  architecture gives a different identity)
- the sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, the penalties, `max_tokens`, `seed`, `n`,
  `best_of`, `stop`)
- the grammar / `response_format`, when one is used
- the prompt **token ids** (not the text: the same text can tokenize differently)
- the tenant salt (see below)

Two requests share an entry only if they would have produced the same text. Anything non-deterministic is refused:
`temperature > 0`, `n > 1`, `best_of > 1`, logprobs, tools, images, `echo`, LoRA adapters (a separate adapter id would have
to be part of the key; until then those requests stay out).

## Configuration

| variable | default | meaning |
| --- | --- | --- |
| `FLLM_OUTPUT_CACHE` | `exact` | `exact` or `off` |
| `FLLM_OUTPUT_CACHE_BYTES` | `268435456` (256 MiB) | byte budget, LRU eviction |
| `FLLM_OUTPUT_CACHE_TTL` | `3600` | seconds an answer stays valid |
| `FLLM_CACHE_SCOPE` | `global` | `global`, `tenant` or `off` |

With `FLLM_CACHE_SCOPE=tenant` the namespace is the `user` field of the request. A request without a `user` field is then
**not cached at all** — no hit is better than a hit that belongs to somebody else.

## Routes and metrics

- `GET /admin/cache` — entries, bytes, hits, misses, evictions, saved prompt tokens
- `DELETE /admin/cache` — drops everything (after switching models, or when memory is needed)
- `/metrics` gains `fllm_output_cache_hits_total`, `..._misses_total`, `..._stored_total`, `..._evictions_total`,
  `..._expirations_total`, `..._saved_prompt_tokens_total`, `fllm_output_cache_bytes`, `fllm_output_cache_entries`

A hit answers with the OpenAI shape and one extra object, `fllm_cache: {hit: true, stored_at}`.

## What it is not

It is not a semantic cache: only the *same* request is a hit. Near-duplicate questions are the domain of the prefix cache
(which saves the prefill) and of a semantic cache that would have to be opt-in — a plausible wrong answer is worse than a
slower right one.

## Limits

- Streaming requests can read from the cache but do not fill it: a streamed answer is stored only when the same prompt is
  also requested without streaming. (Filling it from a stream means collecting the chunks in the stream path; that is the
  next step.)
- Tool calls are not stored (their answer has a different shape), which is why requests with `tools` never take part.
- `logprobs` are stored with the answer and replayed with it, so those requests do take part — the numbers belong in the key
  and in the entry, otherwise a replayed answer would claim probabilities it never had.

## Related switches

Two more knobs decide how much state is shared between requests:

| variable | default | meaning |
| --- | --- | --- |
| `FLLM_PREFIX_SALT` | empty | separates prefix-block identities between deployments: two servers with different salts can never reuse each other's KV blocks, even for identical prompts |
| `FLLM_PREFIX_CACHE` | on | `off` disables the prefix cache completely: no KV block is taken from or given to another request. Use it when requests of different tenants must never share state — the cheaper variant is one process per tenant. |
| `FLLM_PREEMPTION` | `recompute` | `swap` moves the KV blocks of a preempted sequence into host memory instead of throwing them away and prefilling again. `swap` needs a backend that can copy KV blocks; without that the server says so and keeps recomputing. |

**Per-deployment isolation is a salt:** `FLLM_PREFIX_SALT=<name>` seeds every prefix chain, so two servers cannot share
blocks even if they see the same prompt. It is per process, not per key — a deployment that serves several tenants from one
process needs one process per tenant (or `FLLM_PREFIX_CACHE=off`).

**Tenant isolation is honest, not implied:** `FLLM_CACHE_SCOPE=tenant` scopes the *output* cache per `user`, but the prefix
cache still shares KV blocks between requests of different tenants. Real isolation means `FLLM_PREFIX_CACHE=off` or one
server per tenant; the server prints a warning when a tenant scope meets a running prefix cache.

The prefix cache and the output cache answer different questions: the prefix cache saves the prefill of a shared beginning and
is always safe, the output cache saves the whole generation but only for requests whose answer is determined.

## Draining before a restart

`POST /admin/drain` lets a server finish what it is doing and refuse anything new: every inference route answers 503 with
`Retry-After: 30`, and `/ready` reports `draining` so a load balancer takes the instance out of rotation. `DELETE
/admin/drain` takes it back in. That is what makes a rolling restart (`maxSurge: 1, maxUnavailable: 0` in the Kubernetes
manifest) safe: the old pod stops receiving work before it stops.

## One place for the effective settings

`GET /admin/config` answers what the server is actually running with: model and cache identity, context and sequence limits,
ready/draining, the output-cache statistics, whether the prefix cache is on, which preemption policy is active, the telemetry
and coalescing counters, and the sandbox state. Useful when a deployment behaves differently from what the environment
suggests — the answer comes from the process, not from the manifest.

## Settings page and the admin right

Every option above is also on **Settings > Engine** in the app. The page saves them on this computer and the server uses them after its next start
(the page has a button for that); an option that the environment of the server sets wins and is listed on the page as such.

The `/admin/*` routes answer this computer, and any machine that sends an API key with the **admin right**. Keys given with `--api-key` have it. Keys made
on the Studio page (Settings > Server and API) have it only when "Admin" is ticked. While keys are in use, only an admin key (or this computer) can make
or remove keys. An API key for chatting does not open any of this.
