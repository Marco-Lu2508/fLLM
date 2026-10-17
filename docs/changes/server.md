# Change fragment: HTTP server (OpenAI / vLLM / Ollama compatibility, hardening, tests)

Files: `rust/engine/src/server.rs`, `rust/engine/src/server/**`, `rust/engine/src/bin/serve.rs`,
`rust/engine/src/metrics.rs`, `rust/engine/src/encoder_server.rs`, `rust/engine/src/tiny_model.rs`, `scripts/api-tests/`,
`fuzz/`, `.github/workflows/api.yml`. The checkpoint converter (`rust/checkpoint/src/convert.rs`) and one scheduler fix (section 7) are outside that area.

## 1. Features and commands

### Defaults that now match OpenAI, vLLM and Ollama

| Before | Now |
| --- | --- |
| `max_tokens` omitted: 128 | the rest of the context (`prompt + max_tokens <= --ctx`). Too long: `400`, code `context_length_exceeded`, with the numbers |
| Ollama `num_predict` omitted or `-1`: 128 / 512 | Ollama's own meaning: until the model stops or the context is full |
| temperature omitted: 0 (greedy) | `--temp`, else the model's own default, else 1.0 (OpenAI/vLLM/SGLang). Ollama routes: 0.8, top-p 0.9, top-k 40 |
| `--ctx 4096` | `--ctx auto` (default): the trained context of the model, at most 32768 tokens on a GPU (the memory planner lowers it to what fits and says so) and at most 8192 on the CPU, where nothing plans the KV cache; the KV pool keeps the size of the former default (`--max-seqs` x 4096 tokens, never below one full-length sequence), so the memory use does not grow; the start-up line `model=... context=N (auto: ...)` names the value |
| admission limit 128, then `429` | requests running plus `--max-queue` waiting (default `8 * --max-seqs`); beyond that `429` with `Retry-After` |
| `model` in the request ignored | served name, `--model-alias` names, empty or missing: accepted; anything else `404` code `model_not_found` (Ollama routes: `404 {"error":"model 'x' not found"}`) |

Sampling defaults of the model: the checkpoint converter copies `temperature`, `top_p`, `top_k`, `min_p` and
`repetition_penalty` of `generation_config.json` into the GGUF (`general.sampling.temp`, `.top_p`, `.top_k`, `.min_p`,
`.penalty_repeat`, the `general.sampling.*` key family for sampling defaults in GGUF metadata). The server reads them (or a `generation_config.json` next to the file) at start. Order of
precedence: request field, command line (`--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`), model, API default.
GGUF files converted before this change have no such keys: convert again, or pass `--generation-config DIR`.

```sh
fllm-rs --model hf://org/repo --generation-config auto          # default
fllm-rs --model m.gguf --generation-config ./some-dir           # DIR/generation_config.json or a JSON file
fllm-rs --model m.gguf --no-generation-config                   # vLLM's "vllm" mode: no model defaults
fllm-rs --model m.gguf --ctx auto --max-queue 64 --model-name llama3,gpt-4o --model-alias 'claude-*'
```

### Authentication, rate limits, exposure

```sh
fllm-rs --model m.gguf --host 0.0.0.0 --api-key k1 --api-key k2 --rate-limit 600,200000
fllm-rs --model m.gguf --api-key-file keys.txt --metrics-no-auth            # one key per line, # comments
FLLM_API_KEY=k1,k2 fllm-rs --model m.gguf                                   # used when no --api-key/--api-key-file is given
```

* The presented key (`Authorization: Bearer`, `x-api-key` or `api-key`) is hashed with SHA-256 and the digest is compared with every
  configured key in a fixed-time loop. Key length and the position of the first difference do not influence the time.
* With a key configured every route needs it, except `GET/HEAD /health`, `GET /ready` and `GET /` (the web page, which asks for the key).
  `/metrics`, `/api/tags`, `/api/version`, `/get_model_info`, `/props`, `/slots` ... were open before. Relax with
  `--public-routes /metrics,/api/*` (exact path, or prefix when it ends in `*`) or `--metrics-no-auth`.
* `--rate-limit RPM[,TPM]`: per key, token buckets. RPM counts POST/PUT/DELETE requests, TPM counts prompt tokens when a request is admitted and
  generated tokens as they appear (a request may overdraw; the next one is refused until the bucket refills). `429` with `Retry-After`.
  Without keys the limit applies to all callers together.
* Error bodies have the shape of the route (OpenAI `{"error":{"message","type","param","code"}}`, Anthropic `{"type":"error",...}`, Ollama
  `{"error":"..."}`), contain the request id, and the `X-Request-Id` header is accepted from the client (up to 128 characters of
  `A-Za-z0-9._:-`) or generated and always returned.

### Timeouts, limits, shutdown, TLS

| Mechanism | Flag | Behaviour |
| --- | --- | --- |
| request timeout | `--request-timeout S` | a non-streaming request that has no response after S seconds is cancelled (`504`, code `request_timeout`); the generation stops |
| header read | `--header-read-timeout S` | request headers must be complete S seconds after their first byte; a connection that sends nothing is dropped after S as well |
| keep-alive idle | `--keep-alive-timeout S` | an idle connection is closed after S seconds |
| connections | `--max-connections N` | more connections get `503 Retry-After` and are closed |
| body size | `--max-body-size 16MiB` | `413` with the OpenAI error object. `/v1/files` keeps its own limit (`FLLM_MAX_FILE_SIZE`) |
| shutdown | `--shutdown-grace S` | SIGINT/SIGTERM: the listener closes, `/ready` turns 503, requests in flight finish for up to S seconds; a second Ctrl-C ends the process at once |
| TLS | `--tls-cert F --tls-key F` | rustls (ring provider, HTTP/1.1); certificate problems are reported before the model loads |

Slow SSE clients are still cancelled when their 256-event buffer fills.

`GET /health` is liveness and always `{"status":"ok"}` (unchanged, container health checks keep working). `GET /ready` answers `200` only
while the model is loaded, the worker thread makes progress (it wakes at least once a second when idle; 120 s without progress counts as
stuck) and the server is not shutting down; otherwise `503`.

### A panic in model code

The worker loop catches panics per request set-up, per embedding and per step. The affected requests fail with `500` and
"... (error id err_...)" (an SSE `error` event while streaming), the id is in the log line with the panic message, sequences are cancelled, the
counter `fllm_worker_panics_total` increases, and the server keeps serving. Three panicking steps in a row end the process (a supervisor restarts it).
Mutexes that a panic poisoned are recovered instead of failing every later request.

### CORS

Default: none (the web page is same-origin and needs no headers). `--cors-origins '*'` or `--cors-origins https://app.example,http://localhost:3000`.
Preflights (`OPTIONS` with `Origin` and `Access-Control-Request-Method`) are answered before authentication with the requested headers (or a
list that has `Authorization`, `x-api-key`, `Content-Type`, `anthropic-version`, `x-stainless-*`), `Access-Control-Max-Age: 600` and, for Chrome's
private-network preflight, `Access-Control-Allow-Private-Network`. Normal and error responses carry `Access-Control-Allow-Origin`
and expose `x-request-id` and `retry-after`. A disallowed origin gets `403` on the preflight and no CORS headers.

### Remote images

`image_url` (OpenAI) and `source.type=url` (Anthropic) with an `http(s)` URL are fetched only with `--allow-remote-images`; otherwise the request
fails with a message that says so. The fetch resolves the host name itself, refuses loopback, private, link-local (including 169.254.169.254),
CGNAT, multicast, reserved and documentation ranges for IPv4 and IPv6 (also IPv4-mapped, NAT64 and 6to4 forms), connects to the address it checked,
follows at most 3 redirects (each hop checked again), allows no credentials in the URL, stops after 20 MiB or 15 s and identifies PNG/JPEG by their
magic bytes (the `Content-Type` of the server is not trusted).

### Logging

`tracing` with `--log-level LEVEL` (or `RUST_LOG`; default `info`) and `--log-format text|json`; `--access-log` adds one line per request
(target `fllm::access`: request id, client, method, path, status, latency, prompt and completion tokens, written when the body has been sent).
The messages of `bin/serve.rs` and the server use it; the engine internals (backends, importer) still print their `[fllm-rs] ...` lines to stderr.

### Metrics

`--metrics-style fllm|vllm|both` (default both). New `fllm_*`: `fllm_request_queue_seconds`, `fllm_request_e2e_seconds`, `fllm_inter_token_seconds`
(histograms), `fllm_prompt_tokens_total`, `fllm_kv_cache_usage_ratio`, `fllm_requests_finished_total{reason}`, `fllm_worker_panics_total`,
`fllm_queue_rejections_total`, `fllm_rate_limited_total`, `fllm_http_connections`, `fllm_http_connections_rejected_total`,
`fllm_http_request_timeouts_total`. The `vllm:*` series carry `model_name`: `vllm:num_requests_running`, `vllm:num_requests_waiting`,
`vllm:kv_cache_usage_perc` (and the old `vllm:gpu_cache_usage_perc`), `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`,
`vllm:prefix_cache_hits_total`, `vllm:request_success_total{finished_reason=stop|length|abort}`, and the histograms `vllm:time_to_first_token_seconds`,
`vllm:time_per_output_token_seconds` (= `vllm:inter_token_latency_seconds`), `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`.
Histogram buckets start at 1 ms now. The queue time is the time until the scheduler thread accepted the request (HTTP admission and channel);
waiting for a free sequence slot inside the scheduler is not part of it (`vllm:num_requests_waiting` shows that backlog). Memory use of a GPU is not
exported.

### New and changed routes

* vLLM: `GET /version`, `GET /is_sleeping` (always false), `POST /v1/chat/completions/render` (`[conversation, [{prompt, prompt_token_ids}]]`),
  `POST /tokenize` also takes `prompt` or chat `messages` and answers `count` and `max_model_len`, `POST /detokenize` answers `prompt` next to `content`.
* llama.cpp: `POST /props` (accepted, ignored), `POST /slots/{id}?action=save|restore|erase` (`501`, clear message), `GET /lora-adapters` (`[]`),
  `POST /lora-adapters` (`501`).
* OpenAI: `service_tier` echoed, `user`, `metadata`, `store` accepted, `logit_bias` honoured, `best_of` and unsupported `n` rejected with a clear error,
  `stream_options.include_usage` as before. `/v1/models` lists aliases (without wildcards) with `max_model_len`.
* Anthropic: `message_start` carries the real `input_tokens`; `cache_control` is accepted and ignored (there is no prompt-cache API; common prefixes are
  reused automatically); model validation as above.
* Ollama: `/api/tags`, `/api/ps` and `/api/show` fill `size`, `digest`, `parameter_size` and `quantization_level` from the GGUF (the digest is
  a SHA-256 over the file length, its first 16 MiB and last MiB, not over the whole file, so that start-up stays fast); `keep_alive` is validated
  (`"5m"`, `"1h30m"`, seconds, negative = forever, 0 = unload) and sets `expires_at` of `/api/ps`. An empty prompt or empty `messages` list returns the
  Ollama "load"/"unload" reply. **HOOK(multi-model)**: nothing is unloaded here; `note_keep_alive` in `server/api/ollama_extra.rs` is where
  a model registry takes over. `/api/embed` honours `truncate` (default true; false: `400` for inputs longer than the context) and returns
  `prompt_eval_count`.
* Shape changes: the key `quantization` of `/api/tags` details is now `quantization_level` (what Ollama sends); `/rerank` on a decoder answers the
  OpenAI error object instead of `{"message": ...}`; `code` of generic `400` errors is `null` instead of the string `invalid_request_error`.

### Tools

* `fllm-rs make-tiny-model OUT.gguf [--seed N] [--no-generation-config]` (hidden, not in `--help`): writes a random 2-layer Llama with a one-token-per-ASCII-character
  vocabulary in a few seconds. The output is gibberish by design; it exists so that HTTP, streaming, sampling and structured-output behaviour can be tested on
  the CPU without a download.
* `scripts/api-tests/run_all.sh`: the official OpenAI, Anthropic and Ollama Python SDKs (pinned in `requirements.txt`) against a running server, see the header of the script.
* `fuzz/`: cargo-fuzz targets `gguf`, `safetensors`, `pytorch`, `chat_template`. Run: `cd fuzz && cargo +nightly fuzz run gguf -- -max_total_time=60`.
  `fuzz/make_seeds.py` writes small valid seed inputs. Not part of the workspace; `.github/workflows/fuzz.yml` runs each target weekly and on demand.

## 2. Flags and environment variables

New or changed flags of `fllm-rs serve` (all optional):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ctx auto\|N` (aliases `--ctx-size`, `--max-model-len`) | `auto` | context window; `auto` as described above (was 4096) |
| `--generation-config auto\|vllm\|PATH` | `auto` | where model sampling defaults come from |
| `--no-generation-config` | off | same as `--generation-config vllm` |
| `--max-queue N` | `8 * --max-seqs` | requests allowed to wait for a sequence slot before `429` (was a fixed total of 128) |
| `--model-name NAME[,ALIAS...]` (alias `--served-name`) | file name | served name; further comma separated names are aliases |
| `--model-alias NAME` (repeatable) | none | extra accepted `model` values; a trailing `*` matches any suffix |
| `--api-key KEY` (repeatable) | none | accepted keys |
| `--api-key-file FILE` (repeatable) | none | keys, one per line |
| `--public-routes LIST` | none | paths without authentication in addition to `/health`, `/ready`, `/` |
| `--metrics-no-auth` | off | add `/metrics` to the public routes |
| `--rate-limit RPM[,TPM]` | off | per key requests and tokens per minute |
| `--cors-origins '*'\|LIST` | none | allowed browser origins |
| `--request-timeout S` | 600 | non-streaming request timeout (0 = off) |
| `--header-read-timeout S` | 30 | header read timeout (0 = off) |
| `--keep-alive-timeout S` | 75 | idle connection timeout (0 = off) |
| `--max-connections N` | 1024 | open connections |
| `--max-body-size N[K\|M\|G]` | 16M | request body limit |
| `--shutdown-grace S` | 30 | drain time on SIGINT/SIGTERM |
| `--tls-cert FILE`, `--tls-key FILE` | off | HTTPS (both needed) |
| `--allow-remote-images` | off | fetch `http(s)` image URLs |
| `--metrics-style fllm\|vllm\|both` | `both` | metric names |
| `--log-level LEVEL` | `info` | `tracing` filter (`RUST_LOG` syntax); wins over `RUST_LOG` |
| `--log-format text\|json` | `text` | log format |
| `--access-log` | off | one line per request |
| `--max-batched-tokens` | 512 | unchanged; the usage text said 1024 |

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FLLM_API_KEY` | unset | comma separated keys; used only when no `--api-key`/`--api-key-file` is given |
| `RUST_LOG` | unset | log filter when `--log-level` is not given |

## 3. Security notes

* Constant-time key comparison (digest based, all keys visited), several keys, per-key budgets, authentication for every informative route.
* Connection hardening: header-read and idle timeouts, request timeout, connection limit, body limit, 256-event output buffer per stream.
* SSRF-safe image fetching, off by default (section 1).
* Panic isolation: a bug in model code fails requests, not the process; poisoned locks are recovered.
* The chat template of a model repository is executed by a Jinja interpreter; rendering is limited to 50 million instructions per call (minijinja `fuel`
  feature) so that a looping template cannot stall a request thread.
* Inputs reach the GGUF, safetensors and PyTorch (ZIP + restricted pickle) readers: fuzz targets exist for them (`fuzz/`).
* What is still true: the server has no users or roles (a key is all-or-nothing), no audit log beyond `--access-log`, and TLS 1.3/1.2 only through rustls'
  defaults. Put a reverse proxy in front of it when the network is not trusted. Request bodies are buffered in memory up to the body limit per connection.

### New text for SECURITY.md (replace the "Scope and what to expect" section)

```md
## Scope and what to expect

- `fllm-rs serve` is built for use on your own machine or inside a trusted network. It listens on `127.0.0.1` by default.
- If you expose the server beyond that, set API keys (`--api-key`, repeatable, `--api-key-file`, or `FLLM_API_KEY`). Keys are compared in
  constant time. With a key set, every route except `/health`, `/ready` and the browser page needs it, including `/metrics`; widen that
  deliberately with `--public-routes` or `--metrics-no-auth`. `--rate-limit` caps requests and tokens per key.
- The server enforces its own header-read, idle and request timeouts, a connection limit and a request body limit (`--header-read-timeout`,
  `--keep-alive-timeout`, `--request-timeout`, `--max-connections`, `--max-body-size`), and can serve HTTPS itself (`--tls-cert`, `--tls-key`).
  For an untrusted network a reverse proxy with its own rate limiting and logging in front of it is still the better setup.
- Cross-origin browser access is off unless you list origins with `--cors-origins`.
- Image URLs are never fetched unless you start the server with `--allow-remote-images`; the fetch refuses loopback, private, link-local and
  cloud-metadata addresses after name resolution, limits redirects, size and time, and checks the file type itself.
- A panic in model code fails the requests it touched and is logged with an error id; it does not end the process.
- Model files are data, but a malformed GGUF or checkpoint can still be rejected only after expensive work. The GGUF, safetensors, PyTorch
  and chat-template readers have fuzz targets (`fuzz/`); only load models you trust anyway.
- Release assets carry a `.sha256` file next to them; `install.sh` and `install.ps1` verify it.
```

## 4. Third-party entries

No third-party code was adapted. The vLLM, llama.cpp, Ollama, OpenAI and Anthropic API names, field names and metric names are interface
facts; no source of those projects was copied. New crates (all MIT or Apache-2.0; `Cargo.lock` is updated): `tracing-subscriber` with `matchers`,
`nu-ansi-term`, `sharded-slab`, `thread_local`, `tracing-serde`, `tracing-attributes`, `valuable`; direct dependencies on `hyper`, `hyper-util`,
`http-body`, `tower`, `rustls`, `tokio-rustls`, `tracing` that were already in the tree through `axum` and `reqwest`; development only: `rcgen` with
`pem`, `time`, `yasna`, `deranged`, `num-conv`, `powerfmt`, `time-core` (self-signed certificate for the TLS test). TLS needs no Cargo feature.

## 5. Test levels (honest)

Everything here runs on the CPU and was run on the build host (Linux x86-64, Docker image `rust:1-bookworm`). There is no GPU result in this change.

* L2, CPU, executed: `cargo test --release -p fllm-engine --lib server::` runs the router, all middleware and the real worker loop on the tiny model:
  every dialect (OpenAI chat/completions/embeddings/models/responses/files/batches, Anthropic incl. SSE, Ollama chat/generate/tags/embed/ps/show, SGLang, TGI,
  llama.cpp, vLLM extras), SSE framing, a tool-call round trip (grammar-forced Hermes call, parsed to `tool_calls`, streamed and replayed), structured output,
  authentication with several keys, CORS, request and queue limits, body limit, 404 for a wrong model, panic isolation, `/ready`, metric names and values,
  request ids, a real TCP listener for header/idle timeouts, connection limit, graceful shutdown with and without a stuck request, and TLS.
  The SSRF block list and the fetch (against a loopback test server) have unit tests.
* L2, CPU, executed: the SDK scripts against a release binary and the tiny model (`scripts/api-tests/run_all.sh`, openai 3.31.0, anthropic 1.14.0, ollama 0.6.3, pytest 9.1.1 in a `python:3.12-bookworm` container): 25 passed; SIGTERM ended the server with exit code 0 in under a second.
* Fuzzing, executed once for 150 s per target in `rustlang/rust:nightly-bookworm` (AddressSanitizer): `gguf` 355k runs (no seeds), `safetensors` (no seeds), `pytorch` 227k runs (seeded) found no crash. `chat_template` found one crash, see "Known limits".
* Not run: anything on a GPU; `--allow-remote-images` against a real vision model (`--mmproj` models are not part of the test data), TLS against a browser,
  `--ctx auto` on a GPU (the planner path is unchanged code, but the `auto` start value was only exercised on the CPU).

## 6. Known limits

* OPEN: the `chat_template` fuzz target found a stack overflow (AddressSanitizer) in the chat-template renderer for the input `{% set ns = namespace(n=0) %}{or%  fm in messages %}{% set ns.n = ns%}{{ ns.n }}{{ raise_exception('x') if ns.nf > 100 }}` followed by a newline and `[{""content":"yo"}]` (base64 of the whole input: `eyUgc2V0IG5zID0gbmFtZXNwYWNlKG49MCkgJX17b3IlICBmbSBpbiBtZXNzYWdlcyAlfXslIHNldCBucy5uID0gbnMlfXt7IG5zLm4gfX17eyByYWlzZV9leGNlcHRpb24oJ3gnKSBpZiBucy5uZiA+IDEwMCB9fQpbeyIiY29udGVudCI6InlvIn1d`). Not analysed further and not fixed: it is unknown whether the recursion is in minijinja or in the engine's template handling, and whether a normal (non-sanitizer) build overflows its stack. Reproduce: `cd fuzz && cargo +nightly fuzz run chat_template` with the input as an artifact.
* Rate limits and the queue limit are per process and in memory. Batch jobs run in process without a caller, so they are not rate limited.
* `keep_alive` does not unload anything (single resident model); `POST /slots/{id}` actions and runtime LoRA changes answer `501`.
* The `--ctx auto` cap of 32768 on a GPU exists because the per-step block tables grow with the window; use `--ctx N` for longer windows.
* The queue-time histogram does not include the wait for a free sequence slot inside the scheduler.
* The digest in the Ollama listings is a file fingerprint, not Ollama's manifest digest.
* The panic recovery cannot know which request of a batch panicked: all requests of that step fail.

## 7. Fixes found on the way

* `rust/engine/src/scheduler.rs`: cancelling a request that was still waiting for a free sequence slot (a client that disconnects while queued) left its
  id in the waiting queue; the next scheduler step failed with `unknown sequence`, the worker returned an error and the process exited. `free_request` now
  also takes never-admitted sequences out of the queues. Regression test: `a_request_that_leaves_the_queue_early_does_not_stop_the_worker`.
