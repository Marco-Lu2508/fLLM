# API compatibility

fLLM serves several API dialects from one port. Every dialect is translated onto the same
chat-completions pipeline, so sampling, tools, reasoning and structured output behave the same
everywhere.

| Route | Dialect | Notes |
| --- | --- | --- |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI | see [server-api.md](server-api.md) |
| `GET /v1/models`, `GET /v1/models/{id}` | OpenAI, Anthropic | Anthropic shape when the request has an `anthropic-version` header |
| `POST /v1/responses` | OpenAI Responses | streaming, tools, `previous_response_id` |
| `GET`, `DELETE /v1/responses/{id}` | OpenAI Responses | stored responses |
| `POST /v1/messages` | Anthropic Messages | streaming, tools, images, `thinking` |
| `POST /v1/messages/count_tokens` | Anthropic Messages | counts the rendered prompt |
| `POST /api/chat`, `/api/generate`, `POST /api/embed`, `GET /api/tags` | Ollama | see [Ollama](#ollama) |
| `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama | `HEAD /` answers the CLI liveness probe |
| `POST /api/pull`, `DELETE /api/delete` | Ollama | always `501`; fLLM serves one model per process |
| `POST /generate`, `/generate_stream`, `GET /info` | TGI | `/generate` dispatches on the body: `inputs` → TGI, `text`/`input_ids` → SGLang |
| `GET /get_model_info`, `GET /get_server_info` | SGLang | |
| `POST /completion`, `/infill` | llama.cpp | `/infill` needs FIM tokens in the GGUF, otherwise `501` |
| `GET /props`, `GET /slots`, `POST /apply-template` | llama.cpp | |
| `POST /embedding`, `/embeddings`, `/tokenize`, `/detokenize` | llama.cpp | |
| `POST /v1/files`, `GET /v1/files` | OpenAI Files | `multipart/form-data`, field `file` |
| `GET`, `DELETE /v1/files/{id}`, `GET /v1/files/{id}/content` | OpenAI Files | |
| `POST /v1/batches`, `GET /v1/batches` | OpenAI Batches | JSONL input file, executed in the background |
| `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` | OpenAI Batches | |
| `POST /v1/rerank`, `/rerank` | Jina/Cohere response subset | encoder with a one-label classifier |
| `POST /v1/score`, `/classify`, `/pooling` | vLLM-style encoder tasks | see encoder limits below |

Authentication: `Authorization: Bearer KEY` or, for Anthropic clients, `x-api-key: KEY`.

## Messages (Anthropic)

```sh
curl http://127.0.0.1:8000/v1/messages -H 'x-api-key: KEY' -H 'anthropic-version: 2023-06-01' \
  -H 'content-type: application/json' \
  -d '{"model":"m","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Supported: `system` (string or blocks), text and image (base64 or URL) blocks, `tool_use` and
`tool_result`, `tools` with `input_schema`, `tool_choice` (`auto`, `any`, `tool`, `none`),
`stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, and `thinking`
(`enabled` or `disabled`, mapped to the chat template's `enable_thinking`). Reasoning is returned
as `thinking` blocks. `budget_tokens` is accepted and ignored. Server tools (web search and
similar) and prompt caching fields are ignored.

`message_start` reports the size of the rendered prompt; the final `message_delta` carries the real input and output token counts, with the part served from the prefix cache in `cache_read_input_tokens`. `max_tokens` is cut to what the context still holds (Claude Code asks for 32000). A `claude-*` model name the server does not serve is answered by the loaded model. `{"role":"system"}` entries inside `messages` (Claude Code) are accepted: before the first turn they join the system prompt, later ones become a `<system-reminder>` user message; the `x-anthropic-billing-header` system block and `defer_loading` tools are dropped. Long waits send `ping` events every ten seconds.

## Responses (OpenAI)

```sh
curl http://127.0.0.1:8000/v1/responses -H 'Authorization: Bearer KEY' \
  -H 'content-type: application/json' \
  -d '{"model":"m","input":"Hello","instructions":"Be brief."}'
```

Supported: `input` as a string or a list of `message`, `function_call`, `function_call_output`
and `reasoning` items, `instructions`, function `tools`, `tool_choice`, `max_output_tokens`,
`temperature`, `top_p`, `text.format` (`json_schema`, `json_object`), `reasoning.effort`
(forwarded as `chat_template_kwargs.reasoning_effort`), `chat_template_kwargs`, `stream`, `store`
and `previous_response_id`. Built-in tools (web search, file search, code interpreter) are not
available and are ignored.

`max_output_tokens` is cut to what the context still holds, and a silent stream repeats `response.in_progress` every ten seconds (Codex counts events, not SSE comments, against its 300 s idle limit).

Stored responses live in memory. `FLLM_RESPONSE_STORE_SIZE` sets how many are kept (default 256);
they are lost on restart.

## Completion (llama.cpp)

```sh
curl http://127.0.0.1:8000/completion -H 'Authorization: Bearer KEY' \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","n_predict":64,"temperature":0.7,"stop":["\n\n"]}'
```

Supported: `prompt` as a string, a list of token ids, or a list of strings/`{"text":…}` pieces,
`n_predict` (`-1` or omitted fills the remaining context), `temperature`, `top_k`, `top_p`,
`min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `seed`,
`stop`, `stream`, `grammar` and `json_schema`. `cache_prompt` and `n_probs` are accepted and
ignored (no logprobs, no prompt cache reuse), `ignore_eos` is accepted and ignored.

The reply is `{"content", "stop", "stop_type", "stopped_eos", "stopped_limit", "stopped_word",
"tokens_predicted", "tokens_evaluated", "tokens_cached", "timings", "model"}`; streaming sends
`data: {"content":…}` chunks followed by one final object with `stop: true`, the stop flags and
the counters. Errors use `{"error":{"code","message","type"}}`.

```sh
curl http://127.0.0.1:8000/infill -H 'Authorization: Bearer KEY' -H 'content-type: application/json' \
  -d '{"input_prefix":"def fib(n):\n    ","input_suffix":"\n\nprint(fib(10))","input_extra":[{"filename":"a.py","text":"# helper\n"}]}'
```

`/infill` builds the FIM prompt from the GGUF `tokenizer.ggml.fim_*_token_id` metadata (falling
back to the token strings of the Qwen2.5-Coder, StarCoder, CodeLlama and DeepSeek-Coder
families: Qwen2.5-Coder, CodeGemma, StarCoder/StarCoder2, CodeLlama, DeepSeek-Coder, Codestral), prefixes `input_extra` as repository context and orders the blocks per family (markers before the text they announce). Models
without FIM tokens and without an explicit `prompt` answer `501`.

`GET /props` reports `default_generation_settings`, `total_slots`, `model_path`, `chat_template`,
`bos_token`, `eos_token` and `build_info`; `GET /slots` lists the running sequences (an idle entry
when the scheduler is empty); `POST /apply-template` renders `messages`/`tools` with the chat
template and returns `{"prompt": …}`. `POST /embedding` and `POST /embeddings` take
`{"content": …}` (string or array) and return `[{"index":0,"embedding":[[…]]}]`; `/tokenize`
additionally understands `with_pieces` and returns `{"tokens":[{"id","piece"}]}`, `/detokenize`
accepts `skip_special`.

## Generate (TGI)

```sh
curl http://127.0.0.1:8000/generate -H 'Authorization: Bearer KEY' \
  -H 'content-type: application/json' \
  -d '{"inputs":"Hello","parameters":{"max_new_tokens":64,"temperature":0.5,"do_sample":true}}'
```

`inputs` is a string or a list of strings (joined with newlines); a list of token ids is passed
through as ids. `parameters` supports `max_new_tokens`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `stop`, `seed`, `do_sample` (`false` forces greedy decoding),
`return_full_text` and `details`. The reply is `{"generated_text", "details":{"finish_reason",
"generated_tokens", "seed"}}`.

`POST /generate_stream` streams NDJSON-free SSE with `{"token":{"id","text","logprob","special"}}`
events followed by a final event carrying `generated_text` and `details`. `GET /info` returns
`model_id`, `max_input_tokens`, `max_total_tokens` and `version`. Errors use
`{"error": "...", "error_type": "..."}`.

## Ollama

```sh
curl http://127.0.0.1:8000/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hello","images":["<base64 png>"]}],"think":false,"format":"json","tools":[{"type":"function","function":{"name":"now","description":"time"}}],"keep_alive":"10m"}'
```

Supported: `model`, `messages`, `prompt`, `tools`, `format` (`"json"` or a JSON schema), `options`
(`num_predict`, `num_ctx`, `temperature`, `top_k`, `top_p`, `seed`, `stop`, `repeat_penalty`),
`stream`, `keep_alive` (accepted, ignored), `think` (`false` disables reasoning in the chat
template, a level string enables it) and base64 `images` in chat messages and at the top level of
`/api/generate` (needs `--mmproj`). `/api/generate` also takes `suffix`, `system`, `template`,
`raw` and `context`; a non-empty `suffix` is rejected with a pointer to `/infill`.

`POST /api/embed` takes `model`, `input` (string or array of strings) and `dimensions` and
returns `{"model","embeddings":[[…]]}`; `truncate` and `keep_alive` are ignored.
`POST /api/show` reports `license`, `modelfile`, `parameters`, `template`, `details`,
`model_info`, `capabilities` and `modified_at`; `GET /api/ps` reports the loaded model with
`expires_at`. Model management (`/api/pull`, `/api/delete`, `/api/copy`) answers `501` with
`{"error": …}` because fLLM serves one model per process. Errors use `{"error": "..."}`.

## Files and batches

```sh
curl http://127.0.0.1:8000/v1/files -H 'Authorization: Bearer KEY' -F purpose=batch -F file=@batch.jsonl
curl http://127.0.0.1:8000/v1/batches -H 'Authorization: Bearer KEY' -H 'content-type: application/json' \
  -d '{"input_file_id":"file-…","endpoint":"/v1/chat/completions","completion_window":"24h"}'
```

A batch file is JSONL with one `{"custom_id","method":"POST","url","body"}` object per line; every
line is validated when the batch is created, so a malformed line is reported immediately as `400`.
`url` may point at `/v1/chat/completions`, `/v1/completions` or `/v1/embeddings` (`/v1/responses`
runs as a chat request) and `streaming` is always forced off. Lines are executed in the background
as internal requests with bounded concurrency, so interactive traffic is not crowded out.

States are `validating`, `in_progress`, `completed`, `failed`, `cancelling` and `cancelled`;
`request_counts` reports `total`, `completed` and `failed`. Results are written to an output file
and an error file in OpenAI's format (`id`, `custom_id`, `response{status_code,request_id,body}`,
`error`), each exposed as a `batch_output`/`batch_error` file object and downloadable through
`/v1/files/{id}/content`. A running batch is marked `failed` after a restart.

Files live in `FLLM_FILES_DIR` (default `<cache>/fllm-files`), are limited by
`FLLM_MAX_FILE_SIZE` bytes (default 536870912) and batch concurrency comes from
`FLLM_BATCH_CONCURRENCY` (default 2) — see [configuration.md](configuration.md). File ids are
assigned by the server, upload names are sanitized and no path from a request ever reaches the
filesystem.

## Not provided

Audio (transcription, speech), image generation, realtime/WebSocket, moderation, Assistants,
fine-tuning and runtime LoRA switching. fLLM cannot serve these, so the routes do not exist.


## OpenAI token probabilities and usage

Chat accepts `logprobs: true` and `top_logprobs: 0..20`; completions accepts an
integer `logprobs: 0..5`. Both JSON and SSE return the corresponding OpenAI
structures. Token bytes come from the existing grammar tokenizer, including
individual bytes of split UTF-8 characters. Internal llguidance control-token
markers are stripped before exposing token text/bytes. No vocabulary probability
scan or probability records are created when logprobs are absent. Requested logprobs
use raw LM logits before temperature, penalties, bias, grammar and stop-token
masks. This is an explicit fLLM convention: the official API specifies the
response structure, but does not establish that probability calculation stage.
Speculative generation is bypassed when probabilities or stop-token controls
are requested so each returned probability belongs to a target-model row.

Only original tokens fully inside the visible content span receive logprobs.
Reasoning/tool markers and tokens crossing a removed/stop boundary are omitted;
if parser output is not a contiguous span of original token bytes, no probabilities
are fabricated for it. `echo: true` on completions prepends the original string
prompt (or decoded text for token-ID prompts). Prompt logprobs are omitted because
prefill evaluates the LM head only at the last
prompt position. Generated text offsets include the echoed prompt character count.
`best_of` (without `use_beam_search`) samples `best_of` candidates and returns the `n` with the highest cumulative
log-probability; it cannot be streamed. See "Beam search" below for `use_beam_search`.

Legacy completions follow the documented `logprobs: 0..5` limit. The per-token
`top_logprobs` map includes the sampled token even when it is outside the top N,
so its size can be N+1. Text offsets count Unicode characters, including shared
offsets for multiple byte tokens making up one character.

`stream_options.include_usage: true` emits a separate final SSE chunk with empty
`choices`, then `[DONE]`. Otherwise no streaming usage chunk is emitted. JSON
usage includes `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens`. Cached tokens count only reused
prompt KV blocks, excluding recomputation following preemption. Reasoning tokens
count original generated tokens overlapping the parser's reasoning text; delimiters
are excluded. Anthropic/Responses translators request final streaming usage.
Responses preserves cached/reasoning details; Messages maps cached input to
`cache_read_input_tokens` and excludes it from `input_tokens`. Messages has no
standard reasoning usage field, so `output_tokens_details.reasoning_tokens` is a
fLLM extension; output token totals already include reasoning.

Generation accepts `min_tokens`, `ignore_eos`, `stop_token_ids`, `echo` and `seed`.
Before the minimum, EOS and explicit stop token IDs are masked, and string
stops beginning before the minimum are ignored. `ignore_eos` allows EOS to be
generated without ending the sequence; explicit stop IDs still end generation. Ordinary stop-ID tokens remain in the returned text,
while control tokens are hidden by the tokenizer. These extension semantics follow
[vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/).
Seed state is per sequence and therefore independent of unrelated batched requests. `reasoning_effort` is
passed to the chat template as a keyword. `service_tier`, `store`, `metadata` and
`prediction` are accepted without affecting chat/completion generation.

Sources: [OpenAI chat creation](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[OpenAI legacy completions](https://developers.openai.com/api/reference/resources/completions/methods/create)
and [OpenAI streaming schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events).


`n: 1..16` produces independently sampled, indexed choices in JSON and SSE. With
`n > 1`, the scheduler computes the entire prompt once, shares complete KV blocks,
copies the writable tail (including q8 scales), and copies the final prompt logits.
Seeded choice zero uses the supplied seed; later choices use a fixed, documented
64-bit offset `index * 0x9e3779b97f4a7c15` modulo 2^64, so choices remain reproducible
without sampling identically. Prompt usage is counted once and completion usage is
summed across choices. Every choice owns independent text/stop/tool/reasoning state.

Shared choices currently require CPU compute (`--device cpu` or `--device auto`
resolving to the CPU), a stateless model, no draft/MTP engine, and unconstrained
text input. GPU cache state, hybrid recurrent state, image embeddings and grammar
matchers do not yet have a supported shared-prompt fork operation; those combinations
return a validation error. The prompt is computed once and only complete state is shared.

## Beam search (`use_beam_search`)

`POST /v1/completions` and `/v1/chat/completions` accept the vLLM fields for beam search. It is deterministic: the same
request gives the same hypotheses.

| Field | Meaning |
|---|---|
| `use_beam_search: true` | run a beam search instead of sampling |
| `best_of` | the beam width (1 to 8). Without it the width is `n` |
| `n` | how many of the finished hypotheses to return (at most the width), best first, as `choices` |
| `length_penalty` | default `1.0`. A hypothesis scores `sum of token log-probabilities / length ^ length_penalty`, where `length` counts the generated tokens including the final stop token. Larger values favour longer text |
| `early_stopping` | `false` (default): stop when no live beam can beat the worst of the `best_of` finished hypotheses at its current length; `true`: stop as soon as `best_of` hypotheses are finished; `"never"`: stop only when no live beam can still win at its best reachable length |

Beams share the KV blocks of the prompt and of every common prefix (complete blocks are shared by reference count, only
the block a beam writes into is copied), so a width of 8 costs far less KV than eight separate requests. A hypothesis
ends at the end-of-sequence token (unless `ignore_eos`), at a `stop_token_ids` token, at a `stop` string, or at
`max_tokens` (`finish_reason` `stop` or `length`). Scores use the raw log-softmax of the model's logits after
`logit_bias`, `min_tokens`, `allowed_token_ids` and `bad_words`. `temperature`, `top_p`, `top_k`, `min_p`, `seed` and the
other sampling fields do not apply to a beam search.

`usage.completion_tokens` counts the tokens of the hypotheses that are returned. The work spent on the beams that lost is
not billed.

Refused with `400` and a message that names the cause: `stream: true` (no token is final before the search ends),
`logprobs`, repetition/presence/frequency penalties, DRY, mirostat, grammar-constrained generation (`response_format`,
`grammar`, forced tool calls), images, a width above 8, and a width or token budget the server cannot hold. The budget
is checked before the search starts: it needs `best_of` sequence slots (`--max-seqs`) and, in the worst case, a KV tail of
`max_tokens` for every beam, so a large width together with the default `max_tokens` (the rest of the context) can be
refused; set `max_tokens`. When other beam searches hold the slots the answer is `503`. Beam search with a width above 1
needs the same backend as `n > 1` (CPU, a stateless model, no draft/MTP engine); a width of 1 is plain greedy decoding.
Beam requests never use the output cache.


## Encoder task compatibility

BERT, RoBERTa and XLM-R sequence classifiers, including compatible BGE cross-encoders,
are converted with their actual classifier/pooler weights and label metadata. The
encoder serves rerank, score, classify and pooling under existing authentication and
admission controls. Pair layout follows the tokenizer family, with BERT segment IDs
and XLM-R double separators. Ragged batches avoid padding overhead. Classification
returns sigmoid scores for one label and softmax probabilities for multiple labels;
Transformers returns raw logits, so the external verifier applies the same activation.
These are single-label classification/ranking conventions, not regression or
multi-label semantic detection. No calibration is applied.

Rerank preserves document objects, supports `top_n` and `return_documents`, and sorts
stable ties by input index. Limits and schemas are in [server-api.md](server-api.md).
The local meta envelope has zero billed search units. Vendor truncation, priority,
image inputs, calibration and decoder-based Qwen3 yes/no reranking are not implemented.
Decoder task routes return clear authenticated 501 responses. Numerical parity remains
unverified until the external checker runs the new 1/3-label BERT and XLM-R fixtures.

Sources: [Cohere rerank schema](https://docs.cohere.com/reference/rerank),
[vLLM scoring](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/),
[vLLM classification](https://docs.vllm.ai/en/stable/models/pooling_models/classify/)
and [Transformers BERT outputs](https://huggingface.co/docs/transformers/model_doc/bert).
The operation order was read directly from the installed `modeling_bert.py`,
`modeling_roberta.py` and `modeling_xlm_roberta.py` on a test machine.
