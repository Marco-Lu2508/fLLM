# Coding with fLLM: Claude Code, Codex, OpenCode, Aider, Continue

fLLM serves the OpenAI Chat Completions and Responses APIs, the Anthropic Messages API and the llama.cpp `/infill` route from one
port, so the coding tools below talk to it as they would to a hosted service. `fllm-rs connect TOOL` prints the exact
environment or config for the server that is running (or about to start) and writes it with `--write`.

```sh
fllm-rs serve --model Qwen/Qwen3-4B-Instruct-2507-GGUF --ctx 32768 &   # any tool-capable model
fllm-rs connect claude-code            # prints the environment and the settings file it would write
fllm-rs connect claude-code --write    # writes .claude/settings.local.json of the current folder (old content kept as .fllm-backup)
```

`connect` asks the server for the model name and context length (`GET /v1/models`); for a server that is not started yet use
`--model NAME --ctx N`. Nothing is changed on disk without `--write`; without it the command shows the diff `--write` would
apply. The same text is in the Studio under *Settings > Connect other apps > Connect a coding tool* (copy buttons, one tab per
tool). `fllm-rs connect --list` names the tools.

## Choose a model and a context

* Use a model that follows tool calls: Qwen3 instruct (2507) or Qwen3-Coder, Qwen2.5-Coder 7B and up, gpt-oss, Llama 3.1+, Mistral
  and Devstral, GLM, Kimi. The tool format is chosen from the model's chat template (`--tool-parser` overrides it).
  Qwen2.5-Coder 1.5B/3B often answer a tool request with a ```json block or bare JSON instead of `<tool_call>`; the Hermes parser reads both
  when the function is declared in the request.
* Give the server a long context. A coding session resends instructions, tool descriptions and every earlier tool result:
  Claude Code's first request is about 12,000 tokens (measured with 2.1.314, headless, no MCP servers), Codex about 6,300, and
  both grow with each file read. `--ctx 32768` is a good start; on a CPU the default context is only 8192 and too small.
* Output ceilings: Claude Code asks for 32,000 output tokens, OpenCode and Codex for about as many. On the Messages and
  Responses routes fLLM cuts the ceiling to what the context still holds (the reply then ends with `max_tokens` /
  `incomplete`) instead of answering 400. Chat Completions keeps OpenAI's strict check.
* Prefix reuse is on by default: the key/value cache of the conversation prefix is kept between requests, so a tool loop only reads the
  new tool result, not the history again (numbers below).
* A request may wait minutes for its first token on a slow processor. fLLM sends `response.in_progress` (Responses) and `ping`
  (Messages) events every ten seconds, and `connect` sets the client-side timeouts that exist (`API_TIMEOUT_MS` for Claude Code,
  `stream_idle_timeout_ms` for Codex).

## Setup per tool

### Claude Code

```sh
fllm-rs connect claude-code --write          # or: copy the printed environment lines
claude
```

Sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, the Opus/Sonnet/Haiku/subagent model variables (all the served
model), `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, `API_TIMEOUT_MS` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`. Any `claude-*` model name the server does
not serve is answered by the loaded model, so Claude Code's background requests (`claude-haiku-*`) do not fail. fLLM understands what Claude
Code 2.1 sends beyond the public API: `{"role":"system"}` entries inside `messages`, the `x-anthropic-billing-header` system block (dropped, it
would change the prefix on every request), `thinking: {"type":"adaptive"}`, `context_management`, `output_config` and deferred-loading
tool placeholders. Claude Code prints `[claude-code:unrecognized_model]` for a model name it does not know; the request is still sent.
If your shell has `CLAUDE_CODE_USE_BEDROCK` or `CLAUDE_CODE_USE_VERTEX` set, Claude Code ignores `ANTHROPIC_BASE_URL`: unset them.

### Codex CLI

```sh
fllm-rs connect codex --write      # writes ~/.codex/fllm.config.toml (or $CODEX_HOME)
export FLLM_API_KEY=fllm
codex --profile fllm
```

Codex 0.166 reads a named profile from `NAME.config.toml`; the file holds the provider (`wire_api = "responses"`), the model, the context
window and a long stream idle timeout. Your default provider stays as it is. Codex warns that it has no metadata for the model name; that is harmless.

### OpenCode

```sh
fllm-rs connect opencode --write           # opencode.json in the current folder; --global for ~/.config/opencode/opencode.json
opencode --model fllm/MODEL
```

An `@ai-sdk/openai-compatible` provider named `fllm` with the base URL, the key and the model's context and output limits.

### Aider

```sh
fllm-rs connect aider --write      # .aider.conf.yml and .aider.model.metadata.json in the current folder
aider --model openai/MODEL --model-metadata-file .aider.model.metadata.json
```

Aider reads `OPENAI_API_BASE` and `OPENAI_API_KEY`; the metadata file gives it the context window so it neither warns nor guesses.

### Continue (VS Code, JetBrains)

`fllm-rs connect continue --write` creates `~/.continue/config.yaml` when it does not exist (an existing file is never merged; copy the models from
the printed text). Chat and edit use the OpenAI provider; autocomplete uses the llama.cpp provider, which calls `POST /infill`.

### Cline, Roo Code, Kilo Code

`fllm-rs connect cline` prints the four fields of the "OpenAI Compatible" provider (base URL ending in `/v1`, key, model ID, context window).

### SDKs and anything else

`fllm-rs connect openai-sdk` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`), `anthropic-sdk` (`ANTHROPIC_BASE_URL`, no `/v1`) and `env` (all of them).
With `--api-key` on the server the key is a real one; without it any value is accepted.

## Autocomplete: `/infill`

`POST /infill` builds the fill-in-the-middle prompt from the model's FIM tokens (GGUF `tokenizer.ggml.fim_*_token_id`, else the literals of
the family): Qwen2.5-Coder and CodeGemma (`<|fim_prefix|>`), StarCoder, StarCoder2 and Granite code (`<fim_prefix>`), CodeLlama (`<PRE>`),
DeepSeek-Coder (`<｜fim▁begin｜>` and the prefix/suffix spelling) and Codestral (`[SUFFIX]`/`[PREFIX]`). `input_extra` files are put in front as
repository context. Before this version the code and the prefix were placed on the wrong side of the markers, and Qwen2.5-Coder-3B answered
`"\n"`; it now completes `def fib(n):` with the function body and stops at the end of the middle. Only Qwen2.5-Coder was run (see the matrix).

## Compatibility matrix

What was run, with the version and the result. "Config only" means `fllm-rs connect` writes it and the format was checked against the tool's
documentation or schema, but the tool was not started.

| Tool | Version | Model, device | Result |
| --- | --- | --- | --- |
| Claude Code (`claude -p`, Messages API, streaming) | 2.1.314 | Qwen3-4B-Instruct-2507 Q4_K_M, CPU | `Read` tool call, tool result, final answer in 2 turns: works. Settings written by `connect --write`. Before the fixes above every request failed with HTTP 400 (output ceiling 32000, system entry in `messages`). Interactive mode, other tools, subagents: not run. |
| Codex CLI (`codex exec`, Responses API) | 0.166.0 | Qwen3-4B-Instruct-2507 Q4_K_M, CPU (a plain chat also on Qwen2.5-Coder-3B) | `exec` with a shell tool: the function call came back as a `function_call` item, Codex ran it (its own Windows read-only policy refused `cat`), sent the output item back and printed the model's answer: works over 2 requests (6,336 prompt tokens in 575 s on the processor, then 6,441 tokens in 27 s because the prefix was cached). Before the keep-alive events Codex gave up on a 6,300 token prompt after 300 s of silence. Profile from `connect codex --write`. |
| OpenCode (`opencode run`) | 1.19.5 | Qwen3-4B-Instruct-2507 Q4_K_M, CPU | `run` with its `read` tool: tool call, result, answer ("... returns their sum") in 2 requests (10,333 prompt tokens in 1,233 s while a title request shared the processor, then 10,429 tokens in 29 s). Config from `connect opencode --write`. |
| OpenAI Python SDK: chat, streaming usage, streamed tool calls, tool round trip, Responses (create, stream, function calls) | 3.41.0 | Qwen2.5-Coder-3B Q4_K_M, CPU | `scripts/api-tests/test_coding.py`: 11 passed |
| Anthropic Python SDK: messages, streaming, tool use and result turn, count_tokens, errors | 2.3.0 | same | same file, passed |
| Aider | not run | | config only |
| Continue | not run | | config only |
| Cline, Roo Code, Kilo Code | not run | | instructions only |
| Hermes Agent, OpenClaw | not run | | they speak the OpenAI API; use `connect openai-sdk` for the values |

Reproduce the SDK part with `FLLM_BASE_URL=... FLLM_MODEL=... FLLM_TOOL_MODEL=1 pytest scripts/api-tests/test_coding.py`.

## Speed on a coding session

`scripts/coding/replay_session.py` replays a multi-turn tool session (a system prompt, six tools and a file read per turn, or your own
recording with `--session`) and prints, per turn, the prompt size, the tokens served from the prefix cache, the time to the first token
and the decode speed. `--cold` defeats the cache for comparison.

Measured on a laptop processor (Intel Core Ultra 5 235U, 14 threads, shared with other work, so the absolute numbers move by tens of
percent), Qwen3-1.7B Q4_K_M, `--ctx 16384`, 12 turns that each add about 550 tokens:

| Turn | Prompt tokens | From the cache | First token, cache on | First token, cache defeated |
| --- | --- | --- | --- | --- |
| 1 | 753 | 0 | 7.1 s | 9.2 s |
| 5 | 2,943 | 2,384 | 10.3 s | 62.8 s |
| 10 | 5,685 | 5,136 | 17.3 s | 173.6 s |
| 12 | 6,788 | 6,224 | 22.3 s | not run |

With the prefix cache the tenth turn reads only the new ~550 tokens (10 times sooner than reading all 5,700 again). The first token still gets
later with the context (7 s at turn 1, 22 s at turn 12 for the same 550 new tokens): the processor attention kernel re-reads the whole cache for
every group of query rows and is the next thing to optimise. Prompt reading itself is about 45 tokens/s for this 1.7B model on this processor,
which is slow (the Intel iGPU through the wgpu Vulkan path timed out on a 3,500 token prompt, so it was not used). GPUs are far faster; none was measured here.

`scripts/coding/spec_code_bench.py` measures the lookup speculation (on by default, `--speculative off` disables it) on code-editing requests from the
`/metrics` counters; on Qwen2.5-Coder-3B Q4_K_M (CPU, 4 requests that ask for a changed copy of a 235-token Python file) 82 % of 654 proposed draft tokens were accepted and decoding ran at 8.0 tok/s on average against 5.5 tok/s with `--speculative off` (same prompts, shared processor, so a rough figure).

## Known limits

* Tool search / deferred tools of Claude Code and Anthropic server tools (web search) are not available; they are left out of the request.
* `tool_choice` `required` or a named function allows one call per reply (the reply is constrained to one call object), so parallel calls need `auto`.
  If the model runs out of `max_tokens` inside a forced call the request fails with HTTP 422 instead of returning a partial call.
* The output cache (`FLLM_OUTPUT_CACHE`) answers repeated greedy requests from memory; benchmarks must vary their text.
