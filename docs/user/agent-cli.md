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
Newer options have their own sections below: `--plan`, `--allow-rule`/`--deny-rule`/`--ask-rule` ([rules](#permission-rules-plan-mode-and-settings-files)),
`--review` and `--verify` ([self-review](#self-review---review)), `--output-format`, `--continue`, `--resume`, `--image`
([formats](#output-formats-sessions-and-pictures)), `--propose-skills` ([skills](#skills-proposals-and-exchange)).
How this compares with Hermes Agent, Claude Code and Codex CLI: [code-mode-gap.md](../code-mode-gap.md).

* **Output.** The final answer goes to standard output, progress (one line per tool call) to standard error. `--json` prints one
  object with `answer`, `status`, `steps`, `folder` and the list of tool calls with their status
  (`ok`, `error`, `denied`, `unknown`, `bad_args`, `repeated`).
* **Exit codes.** 0 finished, 1 error (server not reachable, empty answer ...), 3 stopped at `--max-steps`, 4 the `--verify` command still fails after the run (see [self-review](#self-review---review)). In the last case the
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

## Permission rules, plan mode and settings files

The `--allow-*` flags decide by kind of action. Rules decide by pattern, in a settings file or on the command line:

```json
{ "allow": ["bash: npm test*", "edit: src/**"],
  "deny":  ["bash: git push*", "read: .env"],
  "ask":   ["bash: npm publish*"],
  "mode":  "default" }
```

* **Where.** `permissions.json` in the Studio folder (yours, every project), then `.fllm/permissions.json` in the project (shared, can be
  committed), then `.fllm/permissions.local.json` (yours, not committed). The lists of all three are joined. `--allow-rule`,
  `--deny-rule` and `--ask-rule` add one rule each for a single run. A file that is not valid JSON stops the run: a typo must not
  silently drop a deny rule.
* **Order.** A **deny** rule always wins, also over `--yes`. Next an **ask** rule: the call is put to the question (a terminal with
  `y/n/a`; without a terminal it stays refused). Then an **allow** rule: the call runs even when no `--allow-*` flag covers it. A call
  no rule mentions falls back to the flags.
* **Patterns.** `category: pattern` with the categories `bash`, `edit`, `read`, `network`, `memory` as in the Studio
  ([studio.md](studio.md)). `*` matches anything inside a command; in paths `*` stops at `/` and `**` does not. A patch is judged by
  every file it touches. An allow rule never covers a command with shell operators (`;`, `&&`, `|`, redirects, substitutions) unless
  the rule is that exact command, and a risky program (`rm`, `curl`, `sudo`, `git`, shells ...) is only covered by a rule that names
  it. A deny rule looks at every part of a compound command.
* **Claude Code spelling.** `Bash(npm test:*)`, `Edit(src/**)`, `Read(.env)`, `WebFetch(domain:example.com)` and the bare tool names
  are read too, and so is a `permissions` object (with `defaultMode: "plan"`), so a project's `.claude/settings.json` can be copied as
  it is. `fllm-rs import claude` does that for you.
* **Plan mode.** `--plan` (or `"mode": "plan"`) allows reading only, whatever the flags say, and asks for a numbered plan. On a terminal
  the plan is shown and `Run this plan now?` starts a second pass with your flags (each action asked unless allowed). Without a
  terminal the plan is the answer, and `--json` marks it with `"plan": true`.

Honest limit: a rule is a pattern on the text of a command, not an analysis of what it does. `bash: npm test*` allows `npm test --
--eval ...` if the project's test script does something odd. Keep the sandbox on ([studio.md](studio.md)) and use deny rules for what
must never happen.

## Self-review: `--review`

A small model often says "done" with a typo in the file. With `--review`, when a run has changed files the command checks them before
it counts as finished: the `review` tool (the diff plus the findings of a fixed rule set), `diagnostics` for every changed file
and, with `--verify "COMMAND"`, your own test command (it runs even without `--allow-command`, because you typed it; deny rules still
stop it). The results go back to the model as a message and the agent works again. Round one is always sent (the model reads its own
diff once more); a second round (`--review-rounds N`, at most 5, default 2) only happens when a machine-stated problem remains: a
diagnostics error, a failing verify command or a high-severity finding. A clean run costs one extra model step. The Studio has the same
switch, *Review my work before it counts as done* (Settings box of the Code page, or `/autoreview on`).

Good default for local models: on, with `--verify` set to the project's quick test command. Honest limit: the rule set is small
(secrets, `eval`, merge markers, leftover debug output ...) and the "read your diff again" step depends on the model; it catches typos
and unfinished edits much more often than design mistakes. The `review` tool needs a git repository; without one only the
diagnostics and the verify command are used.

## Output formats, sessions and pictures

* `--output-format text|json|stream-json`. `json` is `--json`. `stream-json` prints one JSON object per line while the run goes on, in
  the shape of Claude Code's: `{"type":"system","subtype":"init",...}`, `{"type":"assistant","message":{"content":[text,
  tool_use...]}}`, `{"type":"user","message":{"content":[tool_result]}}` and finally `{"type":"result","subtype":"success"` or
  `"error_max_turns"` or `"error_during_execution"`, `"result":...}`.
* **Every run is saved** as `agent-sessions/run-<time>.json` (the newest 200 automatic ones are kept; a run with `--session NAME` is
  kept for ever). `--continue` continues the newest session, `--resume NAME` a named one. The file also records the folder, the model
  and the files changed, which `fllm-rs insights` reads. A picture is not kept in the file.
* `--image FILE` (repeatable) attaches a png, jpeg, gif or webp picture to the task. The loaded model must be able to see; the server
  answers with an error otherwise.

## Search of earlier sessions

Every headless run is copied into the search index of the Studio (the same BM25 index the Code and chat pages use). The agent has a
read-only tool `session_search`: `{query}` lists the best matches with a snippet, `{id}` shows how a session started and ended. You can
use it too: `fllm-rs sessions search WORDS` (against the running server) and `fllm-rs sessions list`; in the Studio `/search WORDS`.
The "summary" is the start and end of a session, not a model-written summary: with a small model one more call per hit costs more than
it gives.

## Skills: proposals and exchange

* `--propose-skills`: after a successful run with at least eight tool calls the model is asked once whether the way it worked is worth
  keeping and, if so, to write it as a skill. The result waits in `skills-proposed/` of the Studio folder: `fllm-rs skills proposed`
  lists it, `fllm-rs skills approve NAME` moves it into your skills, `reject NAME` deletes it. `--auto-skills` saves it at once
  (never over an existing skill). In the Studio the same idea is `/skill`, which shows the draft and saves nothing before you approve.
* A skill is a folder with `SKILL.md` and a `name`, `description`, `when` front matter: the format of the
  [agentskills.io](https://agentskills.io) standard and of Claude Code, Codex and Hermes. `fllm-rs skills export NAME --to claude`
  (or `codex`, `hermes`, `agentskills`, a folder) writes it in the other layout (a valid hyphenated name, the `when` line folded into
  the description, other files copied); `fllm-rs skills import PATH` and `fllm-rs skills install OWNER/REPO[/SUBFOLDER]` read every
  `SKILL.md` below a folder or a git repository (`--dry-run` first). Existing skills are kept unless `--force`. fLLM only reads
  `SKILL.md`; scripts that ship with a skill are copied but the model has to be told to run them.

## `doctor`, `update`, `insights`, `import`

* `fllm-rs doctor` checks the server and model, the tool list against the context window, the sandbox, git, the Studio folder, the
  permission files, skills and the terminal programs (bubblewrap, docker, ssh). Exit code 1 when something fails.
* `fllm-rs update --check` asks GitHub for the newest release (exit code 10 when it is newer than this build). It never downloads or
  replaces anything; the desktop app installs updates.
* `fllm-rs insights [--days N]` (also `usage`): runs, steps, tool calls by tool, edited files and folders of the saved sessions, and
  an approximate token count (characters / 4).
* `fllm-rs import claude|codex|hermes [--from HOME] [--folder PROJECT] [--apply] [--force]` shows what it would bring over and writes
  nothing without `--apply`: skills, custom commands, sub-agents, permission rules (`settings.json`, Codex rules, a Hermes command
  allowlist), the persona (`SOUL.md`) and memories (`CLAUDE.md`, `AGENTS.md`, `MEMORY.md`, `USER.md`) as short memory entries. Hooks and
  model settings are not converted; API keys and tokens are never read. The file layouts come from those tools' documentation and
  were not checked against a real installation of each.

## As an MCP server and as an ACP agent

`fllm-rs mcp-server [--folder DIR] [--allow-edit] ...` speaks MCP on standard input and output, so Claude Code, Codex, Cursor or Zed
can use fLLM's tools: `claude mcp add fllm -- fllm-rs mcp-server --folder /path/to/project`. The tools are those of the running fLLM
server (only the kinds you allow with the same `--allow-*` flags are listed; reading is always there), plus `fllm_agent`, which runs the
whole coding agent on a task with the local model and returns its report (`--no-agent` removes it). Deny rules apply; an ask rule cannot
be answered over MCP and refuses.

`fllm-rs acp` runs the agent as an [Agent Client Protocol](https://agentclientprotocol.com) agent for editors such as Zed: the editor
starts it as a subprocess, sends `session/new` with the folder and `session/prompt` with the text, and receives `session/update`
messages (text, tool calls with their results). Actions that the flags and rules do not allow are put to the editor with
`session/request_permission` (allow once, allow always for this kind, reject); `session/cancel` stops the run before its next model call.
It supports no session loading, no pictures and no editor-side file access. It was written from the public protocol pages and tested
against a scripted client, not against an editor.

## Discord, Slack and who may use a chat

```sh
export FLLM_DISCORD_TOKEN=...                 # Developer Portal > Bot > Reset Token
fllm-rs gateway discord --chat 123456789012345678 --user 111222333444555666 --folder ~/project
export FLLM_SLACK_TOKEN=xoxb-...              # OAuth & Permissions > Bot User OAuth Token
fllm-rs gateway slack --chat C024BE91L --pair --folder ~/project
```

Both poll over HTTPS with the bot token (no websocket, no inbound port). Discord: `GET /channels/{id}/messages?after=` and
`POST /channels/{id}/messages`; answers carry an empty `allowed_mentions`, so no answer pings anybody; reading text in a server channel
needs the *Message Content* intent in the bot's settings, direct messages do not. Slack: `conversations.history` and `chat.postMessage`;
the bot needs `channels:history` (or `groups:history`, `im:history`) and `chat:write`, and Slack limits `conversations.history` to one
call a minute for apps distributed outside a workspace (an app made for your own workspace is not limited). `--chat` takes channel ids
(for Slack the id such as `C024BE91L`, not the name) and is required. The first poll of a channel only records its newest message.
Same session per channel, same permission flags and rules as `agent`.

A chat on the list can still hold people you do not know. `--user ID[,ID]` limits the agent to those senders. `--pair` lets a stranger
ask for access: the bot answers once with an 8-character code (no look-alike letters, from the operating system's random source, valid
for an hour, at most three waiting per platform) and runs nothing; you approve on the machine with
`fllm-rs gateway pair approve discord CODE`, list with `pair list`, remove with `pair revoke discord SENDER`. The store is
`gateway-pairing.json` in the Studio (or profile) folder and is read on every message, so an approval works at once; a damaged file
lets nobody in. `--user` and `--pair` work for Telegram and Matrix too.

WhatsApp, Signal and Email are not there: see [code-mode-gap.md](../code-mode-gap.md#why-some-things-stay-missing).

## What was verified

Checked on a Windows 11 PC (Intel integrated graphics through Vulkan) with Qwen2.5-Coder-3B-Instruct Q4_K_M, a 24k context, and a small
JavaScript project whose test fails because `add` subtracts:

* `fllm-rs agent --review --verify "node test.js" --output-format stream-json`: the model called `read_file` and `edit_file`, the review
  round ran (diff, diagnostics, the model re-read its change), the verify command passed and the exit code was 0. A 3B model does this in
  some runs and not in others: other runs of the same task ended with the model describing a fix it never made (the review then
  caught it through the failing `--verify`, but the model kept insisting) or with a loop of identical edit calls. Use a larger model for
  real work; the review step is a safety net, not a substitute.
* Also run with that model and server: `--continue`, `sessions list|search`, `insights`, `doctor`, `update --check` (GitHub had no
  release to compare with), `--plan` (two attempted writes were refused, the file stayed unchanged), a deny rule against `--yes`
  (the read was refused), `mcp-server` over a pipe (`initialize`, `tools/list`, `read_file`, a refused `bash`, and `fllm_agent`
  answering from the model), `acp` over a pipe (`initialize`, `session/new`, `session/prompt` with a tool call, its result and the answer,
  stop reason `end_turn`), `import claude` and `import codex` against a made-up home folder (dry run wrote nothing; `--apply` wrote
  skills, memory entries, merged rules and a sub-agent; the API key in the made-up `settings.json` appeared nowhere), `skills
  export|import`, `gateway pair list`.
* The Studio (Playwright against the same server): `/autoreview`, `/personality`, `/insights`, `/search` (it found the headless runs),
  the settings persisting over a reload, a real edit followed by the notice "Reviewing my own changes" and the model's review
  answer, `/retry` (reverted the file and sent the message again), and the refusal to attach a picture while a model without
  vision is loaded.
* Docker backend: the command line it builds ran against a real Docker daemon on a Linux machine (alpine image): the workspace
  appeared at `/workspace`, a file created in the container belonged to the workspace owner, `cd` was mapped back, `--network none`
  blocked the network (the ignored test `real_docker_runs_a_command_in_the_mapped_folder`, see its comment for how to run it).
* Unit tests for the rules, the review feedback, stream events, skills, import, session search, gateway chats against stub servers
  (Discord and Slack through the real HTTP client), pairing, terminal backends, the MCP server and ACP against scripted clients.

**Not verified:** a model that can see (the picture path ends at the refusal), the self-proposed skill with a real model (it needs a run of
eight tool calls, which the 3B model did not reach), the Discord, Slack, Telegram and Matrix gateways against the real services, the ACP
adapter in an editor, the MCP server inside Claude Code or Codex, the SSH backend against a real host, the Docker backend on Windows,
and the importers against real installations of the other tools. With Qwen3-4B-Instruct-2507 the first headless runs answered with
sentences about what it would do instead of calling tools, and a run with `tool_choice: required` produced no answer within ten
minutes (the server logged `thinking: always` for that model); that model was not used for the checks above.
