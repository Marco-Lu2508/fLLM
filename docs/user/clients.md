# Clients: talking to the server

fLLM serves OpenAI, Anthropic, Ollama, SGLang/TGI and llama.cpp dialects from
one port (route list: [server-api.md](../server-api.md), [api-compat.md](../api-compat.md)).
This guide covers the three verified SDK clients plus curl and program
integrations.

## Basics (all clients)

- Start the server and note the **served model name**: `--model-name`, or by
  default the model file stem. Clients must send that id as `model` where the
  API requires it.
- Auth: if the server runs with `--api-key` (or `FLLM_API_KEY`), send
  `Authorization: Bearer <key>` on every route except `/`, `/health`,
  `/metrics`, `/api/tags`, `/api/version`, `/get_model_info`,
  `/get_server_info`. Anthropic clients send `x-api-key: <key>` instead.
  Without a key, all routes are open.
- Health check: `curl -s http://127.0.0.1:8000/health` →
  `{"status":"ok"}`.

## curl (OpenAI dialect)

```sh
# chat
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# streaming (server-sent events ending in data: [DONE])
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"Count to five."}],"max_tokens":64,"stream":true}'

# text completion
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","prompt":"The capital of France is","max_tokens":16}'

# embeddings
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["first sentence","second sentence"]}'

# with an API key, add -H 'Authorization: Bearer secret' to each request
```

Defaults worth knowing: greedy decoding (`temperature 0.0`) unless set;
`prompt + max_tokens` must fit `--ctx`; `n` is `1..16` (with `n > 1` CPU-only,
see [troubleshooting](troubleshooting.md)). Structured output via
`response_format` (`json_object`/`json_schema`) or `grammar`+`grammar_type`
(`gbnf`/`lark`/`regex`); tools via `tools`/`tool_choice` with streaming and
`reasoning_content`. Sampling ranges: [server-api.md](../server-api.md#shared-sampling-parameters).

## Verified SDK clients

The matrix below is the tested environment (library versions
`openai 3.29`, `anthropic 1.13`, `ollama 0.6.3`). Reproduce it with
`scripts/api-tests/` (see below) — the versions name the tested environment,
not a minimum requirement.

| Library (test-env version) | Base URL | Model | Script |
| --- | --- | --- | --- |
| `openai` (3.29) | `http://127.0.0.1:8000/v1` | served name | `scripts/api-tests/openai_client.py` |
| `anthropic` (1.13) | `http://127.0.0.1:8000` (no `/v1`) | served name | `scripts/api-tests/anthropic_client.py` |
| `ollama` (0.6.3) | `http://127.0.0.1:8000` (host) | served name | `scripts/api-tests/ollama_client.py` |

```python
# OpenAI
from openai import OpenAI
client = OpenAI(api_key="sk-test", base_url="http://127.0.0.1:8000/v1")
print(client.models.list().data[0].id)
print(client.chat.completions.create(
    model="<served-name>",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
).choices[0].message.content)
```

```python
# Anthropic (needs the anthropic-version header; the SDK sends it)
import anthropic
client = anthropic.Anthropic(api_key="sk-test", base_url="http://127.0.0.1:8000")
m = client.messages.create(
    model="<served-name>", max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}])
print(m.content[0].text)
```

```python
# Ollama
import ollama
client = ollama.Client(host="http://127.0.0.1:8000")
print(client.list())  # served name appears as the model
print(client.chat(model="<served-name>",
                  messages=[{"role": "user", "content": "Hello"}])["message"]["content"])
```

Notes per dialect:

- **Anthropic** (`POST /v1/messages`, `…/count_tokens`): `system`, text/image
  blocks, `tool_use`/`tool_result`, `tools`+`input_schema`, `tool_choice`,
  `stop_sequences`, `temperature`/`top_p`/`top_k`, `stream`, `thinking`
  (mapped to the template's `enable_thinking`). `budget_tokens` accepted and
  ignored; server tools and prompt-caching fields ignored.
- **Ollama** (`/api/generate`, `/api/chat`, `/api/tags`, `/api/version`,
  `/api/embed`, `/api/show`, `/api/ps`): sampling lives in `options`
  (`temperature` default `0.0`, `top_p` `1.0`, `top_k` `0`,
  `repeat_penalty`/`repetition_penalty` `1.0`, `stop`, `num_predict`
  default `min(128, context − prompt)`, `num_ctx`, `seed`). `stream` defaults
  to **`true`** — set `"stream":false` for one JSON reply. `keep_alive`
  accepted, ignored. `format`: `"json"` or a schema. `think: false` disables
  template thinking. Images need `--mmproj`. Model management
  (`/api/pull`, `/api/delete`, `/api/copy`) answers `501`; errors use
  `{"error": "…"}`.
- **OpenAI Responses** (`/v1/responses` + stored-response routes): streaming,
  tools, `previous_response_id`; built-in tools unavailable. Stored responses
  are in-memory (`FLLM_RESPONSE_STORE_SIZE`, default 256; lost on restart).
- **SGLang** (`POST /generate`, `GET /get_model_info`, `/get_server_info`),
  **TGI** (`POST /generate`, `/generate_stream`, `GET /info`), **llama.cpp**
  (`/completion`, `/infill`, `/props`, `/slots`, `/apply-template`,
  `/embedding`, `/tokenize`, …): translated onto the same pipeline; details in
  [api-compat.md](../api-compat.md).

### Reproducing the client tests

```sh
bash scripts/api-tests/run_all.sh [--bin=PATH] [--model=PATH] [--port=PORT]
```

It starts `fllm-rs serve` on a test port (`--ctx 1024 --threads 4 --device
cpu`, model name `test-model`), waits for `/health`, runs the three client
scripts with `FLLM_URL`/`FLLM_MODEL` from a venv, stops the server, and fails
unless all three suites pass. Flags take `=` (`--bin=PATH`; a bare
`8991`-style first argument also sets the port). Defaults:
`BIN=$HOME/fllm/target/release/fllm-rs`,
`MODEL=$HOME/q35/ref-BF16.gguf`, `PORT=8991`, `VENV=$HOME/ux2venv` (a Python
venv with the pinned client libraries); `BIN=`/`MODEL=`/`VENV=` env vars
override without flags. The script sets no API key — the test server runs
open and the client scripts default to `sk-test`.

## Program integrations (UNTESTED placeholders)

The following were **not** tested against fLLM — standard OpenAI-compatible
settings only; report what works and what does not:

| Program | Settings (start here) | Status |
| --- | --- | --- |
| Continue | provider `openai`, `apiBase http://127.0.0.1:8000/v1`, `apiKey` as configured (any string when open), `model` = served name | UNTESTED |
| Open WebUI | OpenAI connection, base `http://127.0.0.1:8000/v1`, key as configured | UNTESTED |
| Aider | `--openai-api-base http://127.0.0.1:8000/v1`, `--openai-api-key …`, `--model openai/<served-name>` | UNTESTED |
| Zed | custom OpenAI-compatible provider, endpoint `http://127.0.0.1:8000/v1`, model = served name | UNTESTED |
| Jan | OpenAI-compatible engine, base URL `http://127.0.0.1:8000/v1`, model = served name | UNTESTED |
| LibreChat | OpenAI-compatible custom endpoint `http://127.0.0.1:8000/v1`, model = served name | UNTESTED |
| Claude Code bridge (Anthropic dialect) | base URL `http://127.0.0.1:8000`, `x-api-key`, model = served name, `/v1/messages` shape | UNTESTED |

Streaming, tool use and thinking support vary by program; the server side of
each is documented in [server-api.md](../server-api.md) and
[api-compat.md](../api-compat.md). The built-in browser chat page (`GET /`)
needs no client at all.

## See also

- Full route/field reference: [server-api.md](../server-api.md)
- Dialect details and limits: [api-compat.md](../api-compat.md)
- Server flags and auth: [configuration.md](../configuration.md)
- Failures: [troubleshooting](troubleshooting.md)
