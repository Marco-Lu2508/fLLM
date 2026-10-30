# The agent without a browser: `fllm-rs agent` and `fllm-rs gateway`

The coding agent of the [Studio](studio.md) can run from a terminal, a script, cron or a chat. Both commands talk to a running
`fllm-rs` server **on the same machine** (the agent tools accept requests from this machine only), so the tools, the sandbox,
the hooks, the `.gitignore` handling and the edit log are exactly those of the Studio. Only the loop is different: it runs in the
command, calls `/v1/chat/completions` and runs each tool call through `POST /api/studio/agent/tool`.

## `fllm-rs agent`

```sh
fllm-rs serve --model owner/repo:Q4_K_M &                      # or the desktop app / studio
fllm-rs agent -p "find the TODOs in src and list them" --folder ~/project
fllm-rs agent -p "fix the failing test" --allow-edit --allow-command --max-steps 40
echo "summarise the changes of today" | fllm-rs agent -p - --json
```

Nobody can answer a question in a script, so permissions are decided up front. Reading and searching files is always allowed.
Everything else is refused unless a flag allows it:

| Flag | Allows | Tools |
| --- | --- | --- |
| (always) | read | `list_dir`, `read_file`, `glob`, `grep`, `diagnostics`, `lsp`, `bash_output` ... |
| `--allow-edit` | edit | `write_file`, `edit_file`, `multi_edit`, `patch`, `notebook_edit` |
| `--allow-command` | command | `bash`, `bash_kill` |
| `--allow-network` | network | `web_fetch`, `web_search`, the browser tools |
| `--allow-memory` | memory | the `memory` tool |
| `--yes` | all four | |

A refused call is **not run**: the model is told which flag would allow it and not to retry. Computer use (screen and mouse) is
never offered to a headless run. The category of each tool comes from the server's catalog (`GET /api/studio/agent/tools`), so a
tool added later is covered by the same rule; a tool without a known category is not offered.

Other options: `--profile NAME`, `--ask`, `--url` (default `http://127.0.0.1:8000`), `--model`, `--api-key` (or `FLLM_API_KEY`), `--max-steps N` (default 30),
`--max-tokens N` (limited to a quarter of the model's context), `--temperature`, `--system TEXT`, `--quiet`.

* **Output.** The final answer goes to standard output, progress (one line per tool call) to standard error. `--json` prints one
  object with `answer`, `status`, `steps`, `folder` and the list of tool calls with their status
  (`ok`, `error`, `denied`, `unknown`, `bad_args`, `repeated`).
* **Exit codes.** 0 finished, 1 error (server not reachable, empty answer ...), 3 stopped at `--max-steps`. In the last case the
  model is asked for a report without tools; if it gives none, the report lists the tool calls so far and says it did not finish.
* **Sessions.** `--session NAME` keeps the conversation in `agent-sessions/NAME.json` of the Studio folder and continues it the
  next time (at most the last 120 messages, cut at a user message). A damaged session file is reported, not replaced.
* **Trajectories.** `--trajectory FILE` appends one JSON line per run (time, folder, model, status, tool calls, the whole
  conversation including the system prompt). Keys are never part of it. Use it to study runs or as training data.
* **Project instructions and memory.** The run reads `AGENTS.md` / `FLLM.md` / `CLAUDE.md` of the folder (and its parents) and the
  memories of the Studio (`POST /api/studio/agent/instructions`) and puts them into the system prompt, capped at 12,000 characters.
  A server that cannot answer that route (older build) does not stop the run.
* **Persona and profiles.** `SOUL.md` in the Studio folder is added to the system prompt as the persona (at most 4,000 characters;
  empty or missing: none). `--profile NAME` switches to `profiles/NAME/` inside the Studio folder, which then holds that profile's own
  `SOUL.md`, sessions and gateway positions, so a work persona and a private one do not mix. A profile is only a folder of files here:
  skills, memory and scheduled tasks of the Studio are not split by profile.
* **`--ask`.** A call that the flags do not allow is shown in the terminal first: `y` runs it once (status `approved`), `n` refuses it,
  `a` allows that kind of action for the rest of the run. Without a terminal on standard input nothing can be asked and the call stays
  refused. Unknown tools and calls with broken arguments are never put to the question.
* **Loop guard.** The same call three times in a row is not run a third time.
* **Small contexts.** The tool list alone needs about 4,000 tokens. With a model whose context is smaller than twice that, the
  command prints a warning at the start: such a server will answer HTTP 400 for most requests (start it with a larger `--ctx`).

## `fllm-rs gateway telegram` and `fllm-rs gateway matrix`

```sh
export FLLM_TELEGRAM_TOKEN=123456:ABC...      # from @BotFather; never taken from the command line
fllm-rs gateway telegram --chat 123456789 --folder ~/project --allow-edit
```

The gateway long-polls the Bot API and runs each text message of an allowed chat through the same agent loop. The answer is sent
back (split at line breaks below the 4096-character limit). Each chat has its own session (`agent-sessions/tg-<id>.json`; a group id
`-100123` becomes `tg-n100123`); `/new` starts a fresh conversation, `/help` lists the commands.

* **`--chat` is required.** Messages from any other chat are dropped without an answer and without a model call. Without a list the
  bot would answer everyone, so the command refuses to start.
* **Permissions** are the same flags as for `agent` and cannot be widened from the chat.
* `--once` handles the waiting messages and exits (for cron). `FLLM_TELEGRAM_API` names another Bot API address.
* The token is read from the environment only and is removed from every error message.

### Matrix

```sh
export FLLM_MATRIX_TOKEN=syt_...                 # access token of the bot's own Matrix account
fllm-rs gateway matrix --homeserver https://matrix.example --chat '!abc123:matrix.example' --folder ~/project
```

The same loop, allowlist and permission flags, over the Client-Server API (`/sync`). `--chat` takes room ids (`!abc:server`; an alias
like `#room:server` is refused because it can be pointed elsewhere). An invite to a listed room is accepted, any other invite is not.
The bot answers with `m.notice`, ignores its own messages, notices, edits and everything that is not plain text, and answers in the
room (long answers are split at 8,000 characters). Encrypted rooms are not supported: the gateway cannot read them and says so on
standard error. `FLLM_MATRIX_HOMESERVER` may replace `--homeserver`. The token goes in the `Authorization` header only.

### Position in the stream

Telegram's offset and Matrix's `next_batch` are saved in `gateway-<kind>.cursor` (in the Studio folder or the profile folder), so a
restart or the next `--once` run does not answer the same messages again. The first Matrix start only records the position: what was
written before it is not answered. If the file cannot be written, the gateway says so and carries on (a restart could then repeat
messages).

Other messengers would implement the same small `Chat` trait (`updates`, `send`, `session_name`, `max_len`); Telegram and Matrix exist.

**What was verified.** The loop, the permission gate, sessions, the allowlist, offsets and message splitting are unit tested with a
scripted model (mutation-checked); the commands ran against a real `fllm-rs serve` with a scripted model and a Bot API stub on
localhost. Matrix ran against a stub homeserver the same way (first sync, invite, own and foreign messages, positions across three runs). Neither gateway was run against a real Telegram or Matrix service, and no real model drove the agent.

## Results of scheduled tasks

A task of the Studio ([studio.md](studio.md)) can send its result somewhere besides the page: the field *Also send the result to* takes
`webhook:URL` (a JSON `POST` with `task`, `ok`, `text`, `time`), `ntfy:URL` (the text as the body, the task name as the title) or
`telegram:CHAT_ID` (the token comes from `FLLM_TELEGRAM_TOKEN` of the server). Whether it worked is stored with the result
(`sent` or `failed: reason`) and shown under the task. Addresses inside the home network are allowed on purpose; the schedule
routes are reachable from this machine only. Tasks still run only while the app is open and a model is loaded.
