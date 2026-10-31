# HTTP API

`fllm-rs serve` exposes one router for decoder models and a smaller one for BERT encoders.
This page lists every route of the decoder and the encoder server. The OpenAI, Ollama and SGLang
routes are specified in detail below; the Anthropic, OpenAI Responses/Files/Batches, llama.cpp and TGI
routes are specified in [api-compat.md](api-compat.md). Field names and defaults come from
`rust/engine/src/server.rs`, `rust/engine/src/server/api/*.rs`, `rust/engine/src/encoder_server.rs` and
`rust/engine/src/parameters.rs`.

- Getting started with curl: [getting-started.md](getting-started.md)
- Server flags and environment variables: [configuration.md](configuration.md)

## Route overview

Decoder server (`serve --model …gguf`):

| Method | Path | Family | Purpose |
| --- | --- | --- | --- |
| GET | `/` | fLLM | Self-contained browser chat page (static, no build step); `HEAD /` answers for Ollama clients |
| GET | `/health` | fLLM | Liveness probe |
| GET | `/metrics` | fLLM | Prometheus text metrics |
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI | The loaded model id |
| POST | `/v1/completions` | OpenAI | Text completion |
| POST | `/v1/chat/completions` | OpenAI | Chat completion (tools, streaming, reasoning) |
| POST | `/v1/embeddings` | OpenAI | Embeddings |
| POST | `/v1/images/embeddings` | fLLM | Projected image tokens (diagnostic, needs `--mmproj`) |
| POST | `/v1/messages` | Anthropic | Messages API (streaming, tools, images, `thinking`) |
| POST | `/v1/messages/count_tokens` | Anthropic | Token count of the rendered prompt |
| POST | `/v1/responses` | OpenAI | Responses API |
| GET, DELETE | `/v1/responses/{id}` | OpenAI | Stored responses |
| GET, POST | `/v1/files` | OpenAI | List and upload files (`multipart/form-data`) |
| GET, DELETE | `/v1/files/{id}` | OpenAI | File metadata and deletion |
| GET | `/v1/files/{id}/content` | OpenAI | File content |
| GET, POST | `/v1/batches` | OpenAI | List and create batches (background execution) |
| GET | `/v1/batches/{id}` | OpenAI | Batch status |
| POST | `/v1/batches/{id}/cancel` | OpenAI | Cancel a batch |
| POST | `/v1/rerank`, `/rerank`, `/v1/score`, `/classify`, `/pooling` | vLLM/Cohere | Answer `501` on a decoder; only an encoder server implements them (see below) |
| POST | `/api/embed`, `/api/embeddings` | Ollama | Embeddings (`/api/embeddings` shares the `/v1/embeddings` handler) |
| GET | `/api/tags` | Ollama | Model list |
| GET | `/api/version` | Ollama | Version string |
| POST | `/api/generate` | Ollama | Completion |
| POST | `/api/chat` | Ollama | Chat |
| POST | `/api/show` | Ollama | Model details |
| GET | `/api/ps` | Ollama | Loaded models |
| POST | `/api/pull` | Ollama | Answers `501` (one model per process; start the server with `--model`) |
| DELETE | `/api/delete` | Ollama | Answers `501` (the model is part of the running process) |
| POST | `/tokenize` | llama.cpp | Text → token ids (`with_pieces` supported) |
| POST | `/detokenize` | llama.cpp | Token ids → text |
| POST | `/completion` | llama.cpp | Completion |
| POST | `/infill` | llama.cpp | Fill-in-the-middle completion |
| POST | `/apply-template` | llama.cpp | Render the chat template |
| POST | `/embedding`, `/embeddings` | llama.cpp | Embeddings |
| GET | `/props` | llama.cpp | Server properties |
| GET | `/slots` | llama.cpp | Slot list |
| POST | `/generate` | SGLang | Completion |
| GET | `/get_model_info` | SGLang | Model info |
| GET | `/get_server_info` | SGLang | Server info |
| POST | `/generate_stream` | TGI | Streaming completion |
| GET | `/info` | TGI | Model info |

Request bodies up to 16 MiB are accepted (`/v1/files` has its own upload limit).

Encoder server (`serve --model bge….gguf`, a BERT-style encoder): `/health`, `/metrics`,
`/v1/models`, `/v1/embeddings`, `/api/embed`, `/api/embeddings`, `/tokenize`, `/detokenize`,
`/v1/rerank`, `/rerank`, `/v1/score`, `/classify`, `/pooling`. Generation routes do not exist
there. The rerank, score, classify and pooling routes are implemented only by the encoder server:
a decoder answers them with `501 Not Implemented` (rerank in the Cohere-style `{"message": …}`
shape, the others as an OpenAI error object).

## Cross-cutting behaviour

**Authentication.** When the server was started with `--api-key` (or `FLLM_API_KEY` is set),
every route except `/`, `/health`, `/metrics`, `/api/tags`, `/api/version`, `/get_model_info`
and `/get_server_info` requires `Authorization: Bearer <key>`; otherwise
`401 {"error":{…}}`.

**Errors.** Generation and embedding errors use the OpenAI shape; rerank errors
use the Cohere-style `{"message":"..."}` body. Other encoder task errors use the
existing OpenAI error object:

```json
{"error":{"message":"prompt is required","type":"invalid_request_error","code":"invalid_request_error"}}
```

`type` is `invalid_request_error` for 4xx and `server_error` for 5xx. Frequently seen cases:
`400` for malformed input, `401` for a bad key, `404` when `model` does not match the loaded
embedding model, `429 inference request limit reached` when the admission limit (128
concurrent requests) is exhausted, `503` when the model worker is unavailable.

**Limits.** Request bodies are capped at 16 MiB (2 MiB on the encoder server).
`prompt + max_tokens` must not exceed the configured context (`--ctx`), the prompt must encode
to at least one token and every token id must be inside the vocabulary. `n` is `1..16`; multiple choices currently require explicit CPU compute and
unconstrained text generation on a stateless model without draft/MTP.

**Request ids.** OpenAI and SGLang responses use `req_<16 hex digits>`, Ollama uses
`ollama_<16 hex digits>`; chat tool calls embed the same id in their call ids.

## Shared sampling parameters

Read by `/v1/chat/completions`, `/v1/completions` (and, from `options` / `sampling_params`,
by the Ollama and SGLang routes). Values present in the request override the server defaults
set with `--temp`, `--top-p`, `--top-k` and the `--min-p`-style flags; invalid values are
rejected instead of ignored.

| Field | Type / range | Default |
| --- | --- | --- |
| `temperature` | number 0 … 100 | `0.0` (greedy) unless `--temp` was given |
| `top_p` | number 0 … 1 | `1.0` unless `--top-p` was given |
| `top_k` | integer 0 … 2147483647 | `0` (disabled) unless `--top-k` was given |
| `min_p` | number 0 … 1 | `0.0` |
| `typical_p` | number 0 … 1 | `1.0` |
| `repetition_penalty` (alias `repeat_penalty`) | number 0.0001 … 100 | `1.0` |
| `presence_penalty` | number −2 … 2 | `0.0` |
| `frequency_penalty` | number −2 … 2 | `0.0` |
| `repeat_last_n` | integer −1 … 2147483647 | `-1` |
| `dry_multiplier` | number 0 … 100 | `0.0` |
| `dry_base` | number 1 … 10 | `1.75` |
| `dry_allowed_length` | integer 0 … 1024 | `2` |
| `dry_penalty_last_n` | integer −1 … 2147483647 | `-1` |
| `dry_sequence_breakers` | array of strings | `["\n", ":", "\"", "*"]` |
| `mirostat` | integer 0, 1 or 2 | `0` |
| `mirostat_tau` | number 0.0001 … 100 | `5.0` |
| `mirostat_eta` | number 0 … 1 | `0.1` |
| `seed` | non-negative integer, or `-1` for “no seed” | unset |
| `logit_bias` | object `{token_id: number}` (values −100 … 100) | empty |

`stop` accepts a string or an array of strings (empty strings are dropped). Matching a stop
string ends generation with `finish_reason: "stop"` and the matched text is not emitted.

**Structured output** (chat, completions and the SGLang/Ollama routes):

- `response_format`: `{"type":"text"}` (no constraint), `{"type":"json_object"}` or
  `{"type":"json_schema","json_schema":{"name":"…","schema":{…}}}`.
- `grammar` plus `grammar_type`: `gbnf` (default), `lark` or `regex`.
- Ollama's `format`: `"json"`, `""` (none) or a JSON schema object.

Constraints are sampler masks, not post-processing; unsupported constructs are rejected
before generation starts.

### Jump-forward decoding

Where a grammar allows exactly one continuation for the next several characters (the fixed keys,
quotes and punctuation of a JSON schema, a literal in a Lark or GBNF rule), the model has nothing to
decide. fLLM asks the grammar for those forced characters, appends them as tokens and lets the model
process all of them in **one** forward pass (the same kind of pass as a prompt chunk) instead of one
pass per token. Sampling continues afterwards as usual. This is the "compressed finite state
machine" idea of SGLang. It applies to `response_format`, `grammar` and `tool_call_grammar`
requests and needs nothing in the request.

What it changes and what it does not:

- The reply is the same as before: forced tokens are streamed as tokens like sampled ones, and
  `usage.completion_tokens` counts them. `finish_reason`, `max_tokens` and stop tokens behave as
  without it.
- With a vocabulary of single-character or byte tokens (for example the test model of
  `fllm-rs make-tiny-model`) a forced character is also the only token the grammar mask would allow, so
  the output is identical to decoding without jump-forward, token for token and seed for seed. The
  tests check this (see below).
- With ordinary subword vocabularies the same text is produced, but the token sequence can differ
  from token-by-token decoding: the forced text is tokenised with the model's tokenizer together with
  the token before it, and the last token of each forced run is not jumped over (it could merge with
  what the model writes next); its characters stay forced, so the next sampled token has to start with
  them. Text that the model chooses afterwards can therefore differ from a run without
  jump-forward, as any change in the token history can.
- Nothing is forced where the grammar may also end or may choose between continuations. JSON schemas
  allow whitespace between tokens by default, which leaves little to force; adding
  `"x-guidance": {"whitespace_flexible": false}` to the schema selects compact JSON and leaves long
  forced runs (the keys of an object, enum values after their first character).
- It is skipped for a sequence (which then decodes token by token) when the request asks for
  `logprobs`, `logit_bias`, `allowed_token_ids`, bad words, `min_tokens` that is not reached yet,
  Mirostat or several choices, for images in the prompt, and while a draft model, MTP or a
  speculative head is active. N-gram speculation works together with it (tested); suffix speculation takes the same path.

Turn it off with `--no-jump-forward` or `FLLM_JUMP_FORWARD=off` (`--jump-forward` forces it on).
`/metrics` reports `fllm_jump_forward_tokens_total` (tokens appended without sampling) and
`fllm_jump_forward_steps_total` (jumps; each costs one forward pass). The same numbers appear as
`forced_tokens`/`jumps` in the runtime's batch statistics.

What has been measured: forward-pass counts on the CPU test model, 12 seeds per grammar, with and
without jump-forward and identical output (`server::tests::jump_forward_matches_plain_decoding_and_needs_fewer_forward_passes`;
the counts include the one prompt pass of every request):

| grammar | forward passes without | with | completion tokens forced |
|---|---|---|---|
| object with fixed keys, enum, boolean (compact) | 486 | 96 | 390 of 486 |
| nested arrays of objects (compact) | 293 | 63 | 230 of 293 |
| Lark literal with alternatives | 440 | 91 | 349 of 440 |
| GBNF literal with alternatives | 182 | 36 | 146 of 182 |
| same object, default flexible whitespace | 827 | 742 | 85 of 827 |
| nested arrays, flexible whitespace | 409 | 409 | 0 of 409 |
| `json_object` | 1150 | 1146 | 4 of 1150 |

What has **not** been measured: wall-clock speed-up on a real model. A forward pass over several
tokens costs more than one over a single token, but on memory-bound decoding (most GPUs and CPUs
at batch size 1) it costs much less than the same number of single-token passes; how much depends on
the model, the hardware and how much of a real schema is fixed text. The numbers above are step counts,
not seconds, and the test model has one character per token, which is the best case for the
amount of forced text (real vocabularies need fewer tokens for the same text, and keep the last
token of each run for the model). A schema with free text everywhere gains nothing.

---

## GET /

No parameters (unauthenticated). Returns the built-in chat page as HTML with
`Cache-Control: no-store` and a restrictive Content-Security-Policy. The page is static —
API credentials are never embedded in the source; it asks for a key when the server runs with
`--api-key`.

```sh
curl -s http://127.0.0.1:8000/ | head -n 3
```

## GET /health

No parameters. Decoder response: `{"status":"ok"}`. Encoder response:
`{"status":"ok","engine":"fllm-rust","architecture":"bert"}`.

```sh
curl -s http://127.0.0.1:8000/health
```

## GET /metrics

No parameters; unauthenticated. Returns `text/plain; version=0.0.4` Prometheus exposition
with two groups of series.

Model/scheduler counters and gauges: `fllm_requests_total`, `fllm_completion_tokens_total`,
`fllm_model_forward_calls_total`, `fllm_model_batch_rows_total`,
`fllm_model_multirow_calls_total`, gauge `fllm_model_batch_max_rows`,
`fllm_prefill_rows_total`, `fllm_decode_rows_total`, `fllm_draft_proposed_tokens_total`,
`fllm_draft_accepted_tokens_total`, `fllm_scheduler_seconds_total`,
`fllm_batch_build_seconds_total`, `fllm_model_forward_seconds_total`,
`fllm_postprocess_seconds_total`.

API observability (also the whole output of the encoder server's `/metrics`):
`fllm_api_requests_total`, gauges `fllm_requests_running`, `fllm_requests_waiting`,
`fllm_kv_cache_blocks_used`, `fllm_kv_cache_blocks_total`, counters
`fllm_processed_tokens_total`, `fllm_generated_tokens_total`,
`fllm_prefix_cache_hits_total`, `fllm_request_errors_total{status="400"}` (only non-zero
statuses appear), gauge `fllm_tokens_per_second` (last ten seconds) and the histograms
`fllm_time_to_first_token_seconds` and `fllm_request_duration_seconds`.

```sh
curl -s http://127.0.0.1:8000/metrics
```

## GET /v1/models

No parameters (auth applies). Returns the single loaded model:

```json
{"object":"list","data":[{"id":"model-q4_k_m","object":"model","created":1730000000,"owned_by":"fllm"}]}
```

The id is the `--model-name` value or, by default, the model file stem. On the encoder server
the entry carries `architecture: "bert"` and `embedding_dimensions` instead of `created`.

```sh
curl -s http://127.0.0.1:8000/v1/models
```

## POST /v1/chat/completions

Request fields:

| Field | Notes |
| --- | --- |
| `messages` | Required, non-empty array. Roles: `system`, `developer`, `user`, `assistant`, `tool`, `function`. Content is a string, or an array of `{"type":"text","text":…}` parts (vision models also accept `image_url` parts with PNG/JPEG data URLs, at most four images). Assistant messages may carry `tool_calls`. |
| `model` | Optional; echoed back, not validated against the loaded model |
| `max_tokens`, `max_completion_tokens` | Integer; default `128` |
| `stream` | Boolean; default `false` |
| `temperature`, `top_p`, `top_k`, … | See the shared sampling table |
| `stop` | String or array of strings |
| `n` | Integer `1..16`; see shared-prompt limits in [api-compat.md](api-compat.md) |
| `logprobs`, `top_logprobs` | Boolean and integer `0..20` (top_logprobs requires logprobs:true) |
| `stream_options.include_usage` | Boolean; final usage-only chunk, default false |
| `min_tokens`, `ignore_eos`, `stop_token_ids` | Minimum generation length, EOS suppression, explicit vocabulary stop IDs |
| `reasoning_effort` | Passed through to chat_template_kwargs.reasoning_effort. On a model with a thinking switch (`capabilities.thinking.control` is `toggle`) it also switches thinking on, `none` switches it off, and `low` / `medium` set a thinking budget of 512 / 2048 tokens (`high` is unlimited); see [Thinking](#thinking) |
| `thinking_budget` | Integer, at least 1: the most tokens the reasoning block may use before the server ends it itself, see [Thinking](#thinking). Not with `n > 1` |
| `tools` | Array of `{"type":"function","function":{"name":…,"parameters":{…}}}`; only `function` tools |
| `tool_choice` | `"none"`, `"auto"`, `"required"` or `{"type":"function","function":{"name":"…"}}`; default `auto` when tools are present, otherwise `none` |
| `parallel_tool_calls` | Boolean; default `true`. `false` rejects a reply containing more than one call |
| `tool_call_grammar` | Boolean; default `false`. With `tools`, constrains decoding, see [Constrained tool calls](#constrained-tool-calls) |
| `response_format` | See structured output |
| `grammar`, `grammar_type` | See structured output |
| `chat_template_kwargs` | Object merged into the Jinja chat-template context |

Rendering: messages go through the GGUF chat template; without a template a plain
`role: content` transcript is used. When `tool_choice` forces a call (or `response_format` is
not `text`), `enable_thinking` is set to `false` in the template context. A required tool call
and `response_format` cannot be combined (`400`).

Response (non-streaming):

```json
{
  "id": "req_0000000000000001",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "model-q4_k_m",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "…",
      "reasoning_content": "…",
      "tool_calls": [{"id":"call_req_…_0","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Berlin\"}"}}]
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 21, "completion_tokens": 34, "total_tokens": 55},
  "timings": {"prompt_n": 21, "prompt_ms": 4.2, "prompt_per_second": 5000.0,
              "predicted_n": 34, "predicted_ms": 680.0, "predicted_per_second": 50.0}
}
```

- `reasoning_content` appears only when the model's output contained a reasoning block
  (`…` or a template-specific `<…think…>` span).
- `tool_calls` appears only when calls were produced; then `content` is `null` and
  `finish_reason` is `tool_calls`.
- `finish_reason` is `stop`, `length` or `tool_calls`.
- `usage` and `timings` are present on every non-streaming completion; the shortcut response
  for `max_tokens: 0` carries `usage` but no `timings` and answers with
  `finish_reason: "length"` and zero `completion_tokens`.

Simple request:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"max_tokens":32}'
```

With tools (the model answers with `tool_calls` instead of text when it decides to call one):

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Weather in Berlin?"}],
    "max_tokens": 128,
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
      }
    }],
    "tool_choice": "auto"
  }'
```

Streaming (`stream: true`) emits server-sent events: one `data:` line per chunk,
a finish chunk per choice with `timings`, then `data: [DONE]`. With
`stream_options.include_usage: true`, a separate final chunk has `choices: []`
and aggregate `usage`. Chunk objects are
`chat.completion.chunk` with `choices[0].delta` holding `content`, `reasoning_content` or
`tool_calls`; the closing chunk has `"delta":{}` plus `finish_reason`. Generation errors
arrive as an `event: error` whose `data` is `{"error": "…"}`.

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"stream":true}'
```

JSON-Schema constrained reply:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Give me a person."}],
       "max_tokens":64,
       "response_format":{"type":"json_schema","json_schema":{"name":"person","schema":{
         "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
         "required":["name","age"],"additionalProperties":false}}}}'
```

## Tools and reasoning

Models write tool calls and reasoning in family-specific formats. The server reads them with a
*tool parser* (`rust/engine/src/tool_parsers.rs`) and a *reasoning parser*
(`rust/engine/src/reasoning.rs`), both chosen when the model is loaded and shared by
`/v1/chat/completions`, `/v1/messages`, `/v1/responses` and the Ollama chat routes. The
startup log names the choice (`tool parser: …, reasoning parser: …`).

Selection, first match wins: `--tool-parser` / `--reasoning-parser` (see
[configuration](configuration.md)), then markers in the model's chat template, then
`general.architecture`, then the Hermes/Qwen format and `<think>` tags. `auto` (the default)
means "choose from the model"; `none` turns a parser off.

### What every parser guarantees

- A tool call leaves the server as `{"id","type":"function","function":{"name","arguments"}}`
  with `arguments` a JSON *string*; `/v1/messages` and `/v1/responses` read exactly these
  fields. Call ids are `call_<request>_<n>`.
- Streaming never shows half a marker: text is buffered at marker boundaries, and a call
  appears once its span is complete. Content, reasoning and the list of calls only grow.
- A span that does not decode (broken JSON, unterminated tag, generation cut off by
  `max_tokens`) is returned unchanged as `content`; it never fails the request and never
  panics. The same holds for a call to a function the request did not declare.
- A call to a declared function whose arguments violate its JSON schema is an error
  (`422` / SSE `error` event), as before.
- Formats that carry arguments as text (XML styles) are typed with the tool's parameter
  schema: `3` becomes a number for an `integer` parameter and stays `"50667"` for a `string`
  parameter. Without a schema type, JSON literals (numbers, booleans, `null`, arrays, objects)
  are interpreted and everything else stays text.
- Many families put their markers into control tokens that plain detokenization drops
  (MiniCPM5's `<function`, `<param`, Mistral's `[TOOL_CALLS]`, Harmony's `<|channel|>` …). While a
  request declares tools, the server keeps exactly these tokens visible.

### Constrained tool calls

Small models often write a command as plain text or break the JSON of a call. Send
`"tool_call_grammar": true` together with `tools` and the reply is limited to either

- ordinary text that cannot contain the opening marker of a call (prose never turns into a
  call by accident), or
- one call, or several unless `parallel_tool_calls` is `false`, in the exact format of the
  model's tool parser, with a declared function name and arguments that follow that
  function's JSON schema (required keys, types, enums). For the `<tool_call>` formats
  (`hermes`, `qwen3_coder`) a sentence may come before the first call ("Let me read the
  file." followed by the call), because models such as Qwen3-4B-Instruct-2507 write one.

`tool_choice: "required"` removes the text branch, `{"type":"function","function":{"name":…}}`
also limits the names to that function. Without `tool_call_grammar`, a required call is
generated as bare, compact JSON (no free whitespace). A forced call without `max_tokens` is
limited to 8192 tokens, not to the rest of the context; when the limit ends it before the
call is complete, the reply is the error "generation ended before a complete function call
was produced; increase max_tokens". If the chat template opens the reasoning block itself
(Qwen3-Thinking-2507, DeepSeek-R1 distills), a forced call or a `response_format` request
gets the block closed empty first, so the JSON is the answer and not reasoning.

Formats with a grammar: `hermes` (also Qwen2.5/3, Granite 4), `qwen3_coder` (XML; integer,
number, boolean and enum parameters are typed, nested values use their JSON schema; other
string constraints such as `pattern` are not enforced), `llama3`, `mistral`
(`[TOOL_CALLS]name[ARGS]{…}`) and `granite3`. For any other parser the server logs one line and
leaves generation unconstrained (the field is accepted). The grammar starts at the first
generated token, so `enable_thinking` is set to `false` like for other constrained requests;
when the prompt still ends inside an open reasoning block the grammar is skipped. It cannot
be combined with `response_format` or `grammar` (`400`), and not with `n > 1` / `best_of`.
Only `/v1/chat/completions` reads the field.

### Tool parsers (`--tool-parser`)

| id (aliases) | Wire format | Source |
| --- | --- | --- |
| `hermes` (`qwen`, `qwen3`, `granite`) | `<tool_call>{"name":…,"arguments":…}</tool_call>` | derived from the Qwen2.5/Qwen3, Hermes-2/3 and Granite-4 templates |
| `qwen3_coder` (`nemotron`, `qwen3.5`) | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` | derived from the Qwen3-Coder, Qwen3.5, Nemotron-3 and Step-3.5 templates |
| `glm` (`glm-4.6`, `laguna`) | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` | derived from the GLM-4.6/4.7 and Laguna templates |
| `minicpm` (`minicpm5`) | `<function name="NAME"><param name="KEY">VALUE</param></function>`; values with `<`, `&` or a newline sit in `<![CDATA[…]]>` | derived from the MiniCPM5 template and vocabulary (`<function`, `<param`, `</param>`, `</function>` are control tokens) |
| `llama3` (`llama`, `python_tag`) | `{"name":…,"parameters":…}` as the whole reply, or after `<\|python_tag\|>`; several calls separated by `;` | derived from the Llama 3.1/3.3 templates. UNVERIFIED: the built-in tool syntax `brave_search.call(query=…)` is not parsed (it stays content) |
| `mistral` (`ministral`, `devstral`) | `[TOOL_CALLS][{…}]` (v0.3, Nemo) and `[TOOL_CALLS]NAME[CALL_ID]id[ARGS]{…}` (Small 3.x, Devstral, Ministral) | derived from the Nemo, Small-3.2 and Devstral templates |
| `deepseek` (`r1`) | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>…`; R1 and V3 write `function<｜tool▁sep｜>NAME` + a ```` ```json ```` block | derived from the V3.1 and R1-Distill templates |
| `deepseek_v32`, `deepseek_v4` | `<｜DSML｜function_calls>` (V3.2) / `<｜DSML｜tool_calls>` (V4) with `<｜DSML｜invoke name>` and `<｜DSML｜parameter name string="true\|false">` | derived from the V3.2/V4 templates. UNVERIFIED: whether the DSML tags are single tokens in the vocabulary |
| `kimi` (`k2`) | `<\|tool_calls_section_begin\|><\|tool_call_begin\|>functions.NAME:N<\|tool_call_argument_begin\|>{json}<\|tool_call_end\|>…` | derived from the Kimi-K2 templates |
| `gpt_oss` (`harmony`) | Harmony: `<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>{json}<\|call\|>`; with the default reasoning parser the channels are split first | derived from the gpt-oss template |
| `gemma4` | `<\|tool_call>call:NAME{key:<\|"\|>text<\|"\|>,n:3}<tool_call\|>` | derived from the Gemma 4 template |
| `functiongemma` | `<start_function_call>call:NAME{key:<escape>text<escape>}<end_function_call>` | UNVERIFIED: from Google's FunctionGemma card, no template available |
| `minimax` (`minimax_m2`) | `<minimax:tool_call><invoke name="NAME"><parameter name="K">V</parameter></invoke></minimax:tool_call>` | derived from the MiniMax-M2 template. MiniMax-M3's namespaced, nested-XML format is not supported |
| `minimax_m1` (`apriel`) | `<tool_calls>` with one JSON call per line or a JSON array, `</tool_calls>` | derived from the MiniMax-M1 and Apriel templates |
| `granite3` | `<\|tool_call\|>[{"name":…,"arguments":…}]` | derived from the Granite-3.3 template (Granite 4 uses `hermes`) |
| `phi4` (`functools`) | `functools[{"name":…,"arguments":…}]`; the word alone in prose is ignored | UNVERIFIED: from vLLM's Phi-4-mini parser and the Firefunction-v2 template, no Phi-4-mini template available |
| `lfm2` (`lfm`) | `<\|tool_call_start\|>[name(key="value", n=3), …]<\|tool_call_end\|>` (Python call syntax; JSON inside the brackets is accepted too) | derived from the LFM2.5 template. Positional arguments are not supported |
| `k2horizon` | `<ifm\|tool_calls><ifm\|tool_call>…</ifm\|tool_call></ifm\|tool_calls>` with JSON, `NAME` + `<ifm\|arg_key>`/`<ifm\|arg_value>`, or the typed form with `<ifm\|arg_type>` | derived from the K2-Horizon template (`tool_call_format` json, xml, xml_typed) |
| `none` | no parsing \| |

Not covered: Seed-OSS (`<seed:tool_call>`), Hunyuan (`<tool_sep>` tags), Command-R, Functionary and
other families; pick the closest id with `--tool-parser` if the wire format matches, or `none`.

If a model's calls come back as plain text, the log line shows which parser was chosen; the
usual causes are a template without the model's marker (use `--tool-parser`) and a tokenizer
that does not store the markers as tokens.

### Thinking

`capabilities.thinking` of `GET /api/studio/status` (`model.capabilities`) and of every entry of `GET /v1/models` tells what the loaded
model can do about thinking. The server reads it from the chat template (the variables `enable_thinking`, `thinking`,
`reasoning_effort`, the reasoning markers), the reasoning parser it picked and the architecture:

```json
{"supported": true, "control": "toggle", "variable": "enable_thinking", "levels": ["off","low","medium","high"], "default": "high"}
```

| `control` | Models | What a client can do |
| --- | --- | --- |
| `none` | Llama, Mistral, Gemma 3 and others without a thinking mode (`supported` is `false`) | nothing; the Studio hides its Thinking chip |
| `toggle` | Qwen3, Qwen3.5, GLM, Gemma 4, Granite 3.3 and other templates with a switch | `off`, or `low` / `medium` / `high`, where the levels are budgets enforced by the server |
| `effort` | gpt-oss and other templates that take `reasoning_effort` | `low`, `medium`, `high`, handed to the template; the model scales its own thinking |
| `always` | DeepSeek-R1 distills, QwQ and Qwen3-Thinking-2507: templates whose generation prompt opens a reasoning block | nothing to switch (an explicit `thinking_budget` still applies) |

A template that merely mentions `<think>` and `</think>` (Qwen3 templates do, to show earlier reasoning in the history) is not `always`:
what counts is the part after the last `add_generation_prompt`. Qwen3-4B-Instruct-2507 therefore reports `none`, Qwen3-4B-Thinking-2507
`always` and the hybrid Qwen3-4B `toggle`.

`variable` is the template variable the switch or level goes into, `default` the level the model uses when a request says nothing.
The detection is a reading of the template, not a test of the model: a template that handles thinking in a way the reader does not
know shows as `none` or `always`, and `chat_template_kwargs` still reaches the template.

**Thinking budget.** With `thinking_budget` (or `reasoning_effort` `low` / `medium` on a `toggle` model) the server watches the
reasoning block. When it has used the budget's number of tokens and is still open, the server closes it itself: it appends the
model's closing tag and a line break to the context (`</think>` for the Qwen/DeepSeek family, the matching tag of the other
formats), ends the first generation and continues from prompt + reasoning + closing tag, so the model writes its answer next.
The client sees one uninterrupted stream (`reasoning_content` deltas, then `content`). Details:

* The forced tokens count as completion tokens (they are part of the context), the reasoning part of `reasoning_tokens` is
  at most the budget plus the closing tag. `prompt_tokens` stays the size of the original prompt.
* The continuation is a second request internally. It reuses the prefix cache for the prompt; the reasoning tokens are
  prefilled again unless the cache still holds them (`timings` then describe the continuation).
* A block the model closed before the budget is left alone, as is a reply without a reasoning block.
* Not applied (the request runs as without a budget) for constrained output (`response_format`, grammars, forced tool
  calls), images, and for reasoning formats that have no closing tag to write (`gpt_oss`, which uses `reasoning_effort`).
  `n > 1` with an explicit `thinking_budget` is refused (`400`).
* If `max_tokens` leaves no room for an answer after the closing tag, the budget is not applied.

### Reasoning parsers (`--reasoning-parser`)

The text is split into `reasoning_content` (in streams: `delta.reasoning_content`) and
`content`; the prompt decides whether the thinking block is already open (templates such as
K2-Horizon, QwQ or DeepSeek-R1 end the prompt with an open tag).

| id (aliases) | Reasoning block |
| --- | --- |
| `think` (`qwen3`, `deepseek_r1`, `glm`, `minimax`) | `<think>` … `</think>` (default); also any open `<…think…>` tag at the end of the prompt |
| `ifm` (`k2horizon`) | `<ifm\|think>`, `<ifm\|think_fast>`, `<ifm\|think_faster>`; the block ends at *any* of `</ifm\|think>`, `</ifm\|think_fast>`, `</ifm\|think_faster>` |
| `mistral` | `[THINK]` … `[/THINK]` |
| `seed` | `<seed:think>` … `</seed:think>` |
| `minimax_m3` | `<mm:think>` … `</mm:think>` |
| `gemma4` | `<\|channel>thought` … `<channel\|>`. UNVERIFIED: token spelling taken from the Gemma 4 template only |
| `gpt_oss` (`harmony`) | channel `analysis` is reasoning; `final` and a recipient-less `commentary` preamble are the answer; messages to `functions.NAME` go to the tool parser; messages to other recipients are not shown |
| `none` | no reasoning split |

## POST /v1/completions

Request fields: `prompt` (required — a string, or a non-empty array of integer token ids),
`model`, `max_tokens` / `max_completion_tokens` (default `128`), `stream` (default `false`),
`stop`, `temperature`, `top_p`, `top_k`, the shared sampling fields, `n` (`1..16` under the shared-prompt limits) and the
structured-output fields (`response_format`, `grammar`, `grammar_type`). `tools` is ignored on
this route.

Response uses `object: "text_completion"` and `choices[0].text`:

```json
{"id":"req_…","object":"text_completion","created":…,"model":"model-q4_k_m",
 "choices":[{"index":0,"text":" Paris","finish_reason":"stop"}],
 "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7},
 "timings":{"prompt_n":5,"prompt_ms":1.1,"prompt_per_second":4545.4,
            "predicted_n":2,"predicted_ms":30.0,"predicted_per_second":33.3}}
```

Streaming chunks are `text_completion` objects with indexed choices carrying `text`.
Finish chunks contain `timings`; `stream_options.include_usage: true` adds a separate
usage-only chunk with empty choices. The stream ends with `data: [DONE]`.
Completions also accepts integer `logprobs: 0..5` and `echo: true`. Echoed prompt text
has no probabilities because the prefill LM head retains only the last position.
All non-streaming usage includes cached prompt and reasoning token details; see
[api-compat.md](api-compat.md) for probability and visibility conventions.

```sh
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

```sh
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"1, 2, 3,","max_tokens":24,"stream":true}'
```

## POST /v1/embeddings (aliases: /api/embed, /api/embeddings)

Request fields:

| Field | Notes |
| --- | --- |
| `input` or `prompt` | String, integer token array, or an array of either; batch size 1 … 32. Each input must fit the model context |
| `model` | Optional; must equal the served model name when present (decoder: `404`, encoder: `400` otherwise) |
| `encoding_format` | `float` (default) or `base64` (little-endian F32 bytes, standard base64) |
| `dimensions` | Decoder: must equal the model hidden size. Encoder: optional truncation to `1 … hidden`, followed by re-normalization |
| `pooling` | Decoder: `mean` (default) or `last`. Encoder: `cls`/`mean`/`last`, default taken from the GGUF `bert.pooling_type` metadata |

Response:

```json
{"object":"list","model":"bge-small","data":[{"object":"embedding","index":0,"embedding":[0.01,-0.02]}],
 "usage":{"prompt_tokens":6,"total_tokens":6},
 "embeddings":[[0.01,-0.02]],"embedding":[0.01,-0.02],
 "prompt_eval_count":6,"total_duration":1234567}
```

`embeddings`, `embedding`, `prompt_eval_count` and `total_duration` (nanoseconds) are extra
fields beyond the OpenAI shape; the Ollama-style clients read them.

```sh
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["a sentence","another sentence"],"pooling":"mean"}'
```

```sh
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer secret' \
  -d '{"input":"single text","encoding_format":"base64"}'
```

## POST /v1/images/embeddings

Requires a loaded `--mmproj`. Request: `{"image_url":"data:image/png;base64,…"}` (PNG/JPEG
data URL). Response: `{"model":…,"engine":"fllm-rust","tokens":<image token count>,
"dimensions":<hidden size>,"data":[[ … ]]}` where `data[0]` is the projected image embedding.

```sh
curl -s http://127.0.0.1:8000/v1/images/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"data:image/png;base64,iVBORw0KGgo…"}'
```

## POST /tokenize

Request: `content` (string, required), `add_special` (boolean).

- Decoder server: `add_special` defaults to `false` (no BOS); response
  `{"tokens":[…],"count":N}`.
- Encoder server: `add_special` defaults to `true`; response `{"tokens":[…]}` (no `count`).

```sh
curl -s http://127.0.0.1:8000/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"content":"Grüße!","add_special":false}'
```

## POST /detokenize

Request: `tokens` (array of integer ids; the decoder route also accepts a non-empty string,
which is tokenized first). Response: `{"content":"…"}`.

```sh
curl -s http://127.0.0.1:8000/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"tokens":[15043,29892]}'
```

---

## Ollama compatibility

### GET /api/tags

No parameters (unauthenticated). Returns one entry for the loaded model:
`{"models":[{"name":…,"model":…,"modified_at":<ISO-8601>,"size":0,"digest":"",
"details":{"family":<model arch>,"parameter_size":"","quantization":""}}]}`.

### GET /api/version

No parameters (unauthenticated). Response: `{"version":"0.4.1-rust"}`.

### POST /api/generate

Request fields: `prompt` (required string, or array of integer token ids), `model` (optional),
`stream` (default **`true`**), `options` (object; non-object is rejected) and the
structured-output fields.

`options` keys: `temperature` (default `0.0`), `top_p` (default `1.0`), `top_k` (default `0`),
`repeat_penalty`/`repetition_penalty` (default `1.0`), `stop`, `num_predict`, plus the shared
sampling fields. `num_predict` defaults to `min(128, context − prompt_tokens)`; a negative
value selects `min(512, context − prompt_tokens)`.

Streaming uses `application/x-ndjson`: intermediate lines carry `"done":false` and either
`response` (text delta) or an `error`, the final line has `"done":true` plus
`total_duration` (nanoseconds), `load_duration` (0), `prompt_eval_count`, `eval_count` and
`done_reason`.

```sh
curl -s http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Why is the sky blue?","stream":false,"options":{"temperature":0.7,"num_predict":64}}'
```

### POST /api/chat

Request fields: `messages` (required, non-empty), `model`, `stream` (default `true`),
`options` (as above), `tools`, `tool_choice`, `parallel_tool_calls` (as in the OpenAI route)
and `format` (`"json"`, `""` or a schema object; `format` also disables template thinking).

Non-streaming response: `{"model":…,"created_at":…,"message":{"role":"assistant","content":…,
"tool_calls":[…]},"done":true,"total_duration":…,"load_duration":0,"prompt_eval_count":…,
"eval_count":…,"done_reason":…}`. Tool calls use the Ollama shape:
`[{"function":{"name":"get_weather","arguments":{"city":"Berlin"}}}]`.

```sh
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"stream":false,"messages":[{"role":"user","content":"Hello"}]}'
```

---

## SGLang compatibility

### POST /generate

Request fields: `prompt` (required; `text` is accepted as an alias), `sampling_params`
(object) and/or top-level `max_new_tokens`/`max_tokens`, `stop`, `temperature`, `top_p`,
`top_k`, `repetition_penalty`, plus the structured-output fields.

`max_tokens` resolution: `sampling_params.max_new_tokens` → top-level `max_new_tokens` →
top-level `max_tokens` → `128`. Inside `sampling_params`: `temperature` (default `0.0`),
`top_p` (default `1.0`), `top_k` (default `0`), `repetition_penalty` (default `1.0`), `stop`.
Responses are always non-streaming:

```json
{"text":"…","meta_info":{"id":"req_…","finish_reason":{"type":"stop"},
                          "prompt_tokens":5,"completion_tokens":12}}
```

```sh
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"def add(a, b):","sampling_params":{"max_new_tokens":32,"temperature":0}}'
```

### GET /get_model_info

Unauthenticated. `{"model_path":<model name>,"model_type":<GGUF architecture>,
"max_total_num_tokens":<configured context>,"is_generation":true}`.

### GET /get_server_info

Unauthenticated. `{"model_path":…,"model_type":…,"version":"0.4.3-fllm-sglang"}`.


## Encoder classification, scoring and reranking

Convert a `BertForSequenceClassification`, `RobertaForSequenceClassification` or
`XLMRobertaForSequenceClassification` checkpoint, then serve it with `--embeddings`.
Classifier tensors, the BERT pooler or RoBERTa classification dense layer, label count
and `id2label` are stored in GGUF. BERT pairs use `[CLS] query [SEP] document [SEP]`
with document segment IDs of one; RoBERTa/XLM-R use `<s>query</s></s>document</s>` and
zero segment IDs. Inputs are neither truncated nor silently replaced.

The head operates on the unnormalized first hidden row, applies its dense projection
and tanh, then the classifier. One-label scores are sigmoid(logit); multiple-label
probabilities are softmax(logits). This postprocessing is the requested scoring
convention, not a claim that Transformers itself applies these activations to logits.
Regression and multi-label activation policies are not inferred from `problem_type`.

| POST route | Request | Response |
| --- | --- | --- |
| `/v1/rerank`, `/rerank` | `model?`, `query`, `documents`, `top_n?`, `return_documents?` | `results[{index,relevance_score,document?}]`, `usage`, `meta` |
| `/v1/score` | `model?`, `text_1`, `text_2` | `data[{index,object:"score",score}]`, `usage` |
| `/classify` | `model?`, `input` | `data[{index,label,probs,num_classes}]`, `usage` |
| `/pooling` | `model?`, `input`, `pooling?` | `data[{index,data}]`, `usage` |

Rerank accepts 1..256 strings or objects with a string `text` field. Additional object
fields are preserved when `return_documents: true`; strings return `{text: ...}`.
Results sort by descending relevance and by original index for ties. `top_n` must be
positive and is capped at the document count; omission returns all documents.
Rerank and score require a one-label head. Score accepts strings or batches of at most
32 strings on either side; a singleton broadcasts, otherwise lengths must match.
Classify and pooling accept one string or up to 32 strings. Pooling returns the same
normalized cls/mean/last vector as the embedding route; no classifier is required.

Each string is limited to 65536 bytes, each sequence to model context, and a request
to 32768 total input tokens. The existing encoder 2 MiB body limit, Bearer authentication
and single-request admission permit apply. Blocking compute processes a ragged batch
in the encoder Rayon pool, without padding tokens or cross-sequence attention.
Rerank errors use `{message: ...}`; score/classify/pooling retain the OpenAI error object,
including invalid-JSON and oversized-body responses. Unsupported decoder task routes
return 501 after authentication. Decoder-based Qwen3 reranking is not implemented:
its yes/no-logit and model-specific template contract needs a separate worker path.

`usage.prompt_tokens` counts actual pair input tokens, including query copies and
special separators; `total_tokens` is the same. `meta.api_version.version` is `"2"`
for the Cohere-style envelope. `meta.billed_units.search_units` is zero because the
local server does not bill searches; input token counts are also in that envelope.
The response/request subset includes Jina/vLLM document objects and return_documents,
not every vendor option (`max_tokens_per_doc`, priority, image ranking, calibration).

Verifier fixtures (commands for the external checker; not executed during this change):

```sh
python scripts/hf_golden.py bert_classifier_1 /tmp/bert-one
python scripts/hf_golden.py bert_classifier_3 /tmp/bert-three
python scripts/hf_golden.py xlm_roberta_classifier_1 /tmp/xlm-one
fllm-rs convert /tmp/bert-one /tmp/bert-one.gguf --outtype f32
python scripts/verify_encoder.py --binary target/release/fllm-rs \
  --model /tmp/bert-one.gguf --hf-reference /tmp/bert-one
```

The checker loads only the supplied local Transformers checkpoint, compares probabilities
at relative error below 2e-3 and matching argmax, then checks broadcasting, rerank order,
document preservation, authentication and validation. It can use `--url` for an already
running encoder. The small classification fixtures have 64 hidden units and three layers.
