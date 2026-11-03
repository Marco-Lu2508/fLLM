# Work flows around the coding agent: specs, steering, hooks, auto mode, heartbeat, routes, embedding, review

This page covers what surrounds a single `fllm-rs agent -p "task"` run: planning a feature in files before coding it, standing
instructions that apply to some files only, hooks on more events and on file changes, a model that judges risky actions, a gateway that
wakes up on its own and routes chats to different agents, embedding the agent in other programs, pull request review, and plugins from git
repositories. Everything here works with a small local model; each part says what it costs in model calls and where it was tried.
[code-mode-gap.md](../code-mode-gap.md) lists what other coding agents have and why each item here was built or skipped.

Every command needs a running fLLM server on this machine with a model that can call tools (`fllm-rs serve ...` or the Studio), as for
`fllm-rs agent`. All of it is off unless you use it.

## Specs: plan a feature in three files, then build it task by task

```text
fllm-rs spec new csv-export "Export the report table as a CSV file from the report page."
fllm-rs spec list
fllm-rs spec run csv-export                  # the next unchecked task
fllm-rs spec run csv-export --all --verify "pytest -q"
```

`spec new` asks the agent for three Markdown files under `.fllm/specs/<name>/`, one after the other:

| File | What it holds |
| --- | --- |
| `requirements.md` | numbered requirements, each with a user story and acceptance criteria written as `WHEN ... THEN the system SHALL ...` |
| `design.md` | overview, the files that change (real paths the agent found), data and interfaces, error handling, tests |
| `tasks.md` | a checklist, `- [ ] 1. Add the writer`, ordered so that each task can be checked after the one before |

The agent only reads the code for these steps; the command writes the files, so a small model has to produce a document, not to call a write
tool correctly. A reply that is unusable (no checklist in `tasks.md`, far too short) is asked for once more with the reason. On a terminal each
file is shown and you are asked before the next one; `--phase requirements|design|tasks` makes just one file, so you can edit the Markdown
and carry on later. Nothing else in the project is touched.

`spec run` does the unchecked tasks. Each task is its own agent run that sees the requirements, the design and the whole list and is told to
do only its task; it may edit files (not the spec files) and, with `--allow-command` or `--verify`, run commands. **The command ticks the
box, never the model**: only when the run finished (and the `--verify` command, run after the task with the review step, passes). A task
that fails stays open, `spec run --all` stops at the first failure, and `spec tick NAME N [--undo]` changes a box by hand. Options for the
model and the server (`--url`, `--model`, `--max-steps` ...) are passed on to `fllm-rs agent`.

In the Studio the Code page has `/spec NAME what you want`, which makes the agent write the three files itself, stopping after each for your
review, `/spec run NAME [N]`, and a **Specs** block in the side panel: every spec with the three files shown as filled or empty dots, its
tasks as check boxes (a click ticks a task in `tasks.md`) and a *Run next task* button. In the Studio the agent ticks the box itself when
the task is done; the check box lets you correct it.

## Steering files

Standing instructions in `.fllm/steering/*.md` (for one project) or `steering/` in the Studio folder (for all projects) next to `AGENTS.md`.
An optional front matter says when a file applies:

```markdown
---
inclusion: fileMatch
fileMatchPattern: ["**/*.tsx", "src/ui/**"]
---
Components are function components. Styles live in the .module.css next to them.
```

| `inclusion` | Effect |
| --- | --- |
| `always` (default, also without front matter) | part of every prompt, like `AGENTS.md` |
| `fileMatch` | added to the result of a tool call that reads or changes a matching file, once per session (the model sees it when it matters, the prompt stays short) |
| `manual` | listed by name and path in the prompt; the agent reads it when the task needs it; `fllm-rs agent --steering NAME` puts it into the prompt |
| `auto` | like `manual`, with a one-line `description` that says when it applies |

`fileMatchPattern` takes one glob or a list (`*`, `**`, `?`, `{a,b}`; a pattern without `/` matches the file name at any depth). The workspace
file wins over a user file of the same name. Each file is cut at 6,000 characters and the always-files together at 10,000, so they cannot fill a
small model's context. Steering files are read on every run; there is nothing to enable.

## Hooks: more events, prompts, and file changes

The hooks of the Studio ([studio.md](studio.md)) (Settings, *Hooks*, or `hooks.json` in the Studio folder, or a plugin) now know eleven events:
`pre_tool_call`, `post_tool_call`, `on_file_edit`, `on_session_start`, `on_session_end` as before and

| Event | Fired by | A hook can |
| --- | --- | --- |
| `user_prompt_submit` | the headless agent, the RPC mode, the Studio, when a task is sent | block it (exit 2, stderr is the reason) or add notes the agent sees with the task |
| `stop` | the same, when the agent is about to finish | block: the reason goes back to the agent as its next instruction (twice at most), for example "run the tests before you finish" |
| `subagent_stop` | reserved for a sub-agent that has reported back | notes |
| `notification` | the Studio, when a permission question waits for you | run a command (a sound, a message to your phone) |
| `pre_compact` | the Studio, before older turns are summarised | save the transcript |
| `file_changed` | `fllm-rs agent --watch` | react to a created, changed or deleted file |

A hook can carry a `prompt` instead of a `command`: its text (with `{{file}}` replaced by the path) is given to the agent as context. The
spellings of Claude Code and Kiro (`PreToolUse`, `UserPromptSubmit`, `Stop`, `PostFileSave` ...) are accepted in the files. A hook that
fails never stops the agent; an unreachable hook route (an older server) means no hooks.

`fllm-rs agent --watch --folder DIR [--allow-edit ...]` watches a folder (it polls modification times, so it works the same everywhere and
skips `.git`, `target`, `node_modules`, `.fllm` and names in the top-level `.gitignore`). For every created, changed or deleted file the
`file_changed` hooks whose globs match run; a hook with a `command` runs on the server like every other hook, a hook with a `prompt` starts one
agent run with that text and the path. Changes the agent makes during its own run are not reported back, `--max-runs N` (default 20) bounds the
runs, and a file that is still being written is handled when it has been quiet for one scan.

```json
{"hooks": {"file_changed": [
  {"matcher": "docs/**/*.md", "on": "modified", "prompt": "{{file}} changed: check its links and fix dead ones."},
  {"matcher": "src/**/*.rs", "command": "cargo fmt --check"}
]}}
```

## Auto mode: `--auto`

Instead of refusing everything the `--allow-*` flags do not cover (or asking you each time), `fllm-rs agent --auto` lets a short model call judge
each such action against your task. The order: a **deny rule** always wins and an **ask rule** always goes to you; reading, what a flag or allow
rule covers runs as before; a short list of **hard blocks** (`curl ... | sh`, `rm -rf` of a root or home folder, a forced push, writing to a disk
device or to system folders, reading credentials) is refused without asking the model; everything else goes to the judge, which sees your task
and the one action, never tool results (text inside a file or web page cannot talk to it), and answers `ALLOW` or `BLOCK: reason`. An answer it
cannot read counts as a block. After three blocks in a row, or twelve in all, it stops judging; with `--ask` a blocked action is put to you,
otherwise it stays refused. The judge is one more call of at most 48 tokens per risky action, using the same model. It is a middle way
between asking every time and `--yes`; a small model judges less reliably than a large one, so use deny rules for what must never happen.

## The gateway: heartbeat and routes

**Heartbeat.** `fllm-rs gateway telegram --chat 123 --heartbeat 30m --heartbeat-chat 123` makes the agent read `HEARTBEAT.md` in its folder every
30 minutes, check the list with its tools and message the chosen chats **only when something needs attention**:

```markdown
# Heartbeat
- Is the disk of /srv more than 85% full? (run `df -h /srv`)
- Did backup.log end with an error?
```

The agent answers `HEARTBEAT_OK` when all is well; that answer (also with a few words around it) is dropped. A file with only headings costs no model call.
A heartbeat run may only read unless you add `--heartbeat-act`; `--heartbeat-hours 08:00-22:00` keeps it quiet at night (a window across midnight
works), `--heartbeat-idle 10m` waits until nobody has written for that long, `--heartbeat-file PATH` names another list. Every wake-up starts from the list
without memory of earlier ones, so a weak model sees a short prompt. Alerts only go to chats that are allowed in `--chat` or in the routes; it does not run with `--once`.
It costs one agent run per interval, and the gateway has to keep running (it is a process, not a schedule inside the app).

**Routes.** `--routes FILE` (or `gateway-routes.json` in the profile folder) gives chats their own agents:

```json
{"routes": [
  {"chat": "123456789", "profile": "home", "folder": "/srv/notes", "allow": ["edit"]},
  {"chat": "-1001234567", "profile": "work", "folder": "/srv/app", "allow": ["edit", "command"], "system": "Answer in German.", "model": "qwen3-4b"}
]}
```

A routed chat is allowed, and runs in its folder with its profile (the persona `SOUL.md` and the sessions of `profiles/NAME`), its permissions
(reading plus what `allow` names: `edit`, `command`, `network`, `memory`, `all`), its extra system text and model. Other allowed chats use the command line
defaults. The tools follow the chat (the server's folder is switched before every run); nothing in a chat message can change a route.
The heartbeat uses the default folder.

## Embedding the agent: `fllm-rs agent --rpc` and the SDKs

`fllm-rs agent --rpc` runs the agent as a JSON-RPC 2.0 server on standard input and output, one message per line, for editors, scripts and services:

| Method | Does |
| --- | --- |
| `initialize` | name, version, protocol (1), methods, permission categories |
| `session/new` `{folder, allow, plan, model, system, max_steps, allow_rules, deny_rules, ask_rules}` | a session in an absolute folder; `allow` names the categories beyond reading |
| `session/prompt` `{session, text}` | runs one task; the result is `{answer, status, steps, edited, tool_calls}` |
| `session/cancel` (notification), `session/history`, `session/close`, `tools/list`, `shutdown` | |

While a prompt runs the agent sends `session/event` notifications (`text`, `tool_call`, `tool_result`) and, for an action the permissions do not
cover, the request `session/approve` `{call_id, tool, category, args}`, which the client answers with `{allow, always}`. Deny rules, plan mode, the
sandbox and the `user_prompt_submit` and `stop` hooks work as in the headless agent.

Clients without dependencies: [scripts/sdk/fllm_agent.py](../../scripts/sdk/fllm_agent.py) (Python 3) and
[scripts/sdk/fllm-agent.mjs](../../scripts/sdk/fllm-agent.mjs) with [fllm-agent.d.ts](../../scripts/sdk/fllm-agent.d.ts) (Node 18 or newer):

```python
from fllm_agent import Agent
with Agent(url="http://127.0.0.1:8000") as agent:
    session = agent.new_session("/path/to/project", allow=["edit"])
    result = session.prompt("Fix the failing test", on_event=print,
                            approve=lambda call: call["tool"] == "bash" and call["args"]["command"].startswith("pytest"))
    print(result["status"], result["answer"])
```

## Review a pull request: `fllm-rs review`

`fllm-rs review --pr 42` fetches the diff and text of pull request 42 with the GitHub CLI (`gh`, signed in), lets the agent read it together with the
repository (read-only: it cannot edit, run commands or use the network) and prints findings as `path:line - severity - problem - what to change`, ending
in `VERDICT: ok` or `VERDICT: issues`. `--comment` posts it as a comment that starts with a hidden marker, `--security` looks for security problems only,
`--base main` reviews the branch against main and `--staged` the staged changes (both with plain git), `--fail-on-issues` exits with 5 on a verdict of
issues, and a diff longer than `--max-diff-chars` (default 40,000) is cut with a note so the agent reads the rest itself.

**On GitHub Actions** copy [.github/workflows/fllm-review.yml.example](../../.github/workflows/fllm-review.yml.example) into
the `.github/workflows` folder of your repository and drop the `.example` ending. It needs a machine with `fllm-rs`, `gh` and a running fLLM server; the simplest setup is a self-hosted runner on that
machine (server on `127.0.0.1:8000`), otherwise give the server an address and an API key and store them as the secrets `FLLM_URL` and `FLLM_API_KEY`.
The workflow replaces its own earlier comment on every push and skips pull requests from forks. It has not run on a real runner yet; the command it
calls was tested against a fake `gh`.

## Plugins from git repositories

```text
fllm-rs plugins add owner/repo                 a repository with plugin.json at its root
fllm-rs plugins add owner/repo@v1.2 --yes      at a tag or branch, without the question
fllm-rs plugins add owner/market               a repository with marketplace.json lists what it offers
fllm-rs plugins add owner/market --plugin rust-review      (or --all)
fllm-rs plugins list | remove NAME
```

A marketplace is any git repository with a `marketplace.json`: `{"name": "my-plugins", "plugins": [{"name": "rust-review", "source": "plugins/rust-review",
"description": "..."}, {"name": "other", "source": "someone/other-plugin"}]}`; a `source` is a folder of that repository or another repository. Before it
installs, the command lists what the plugin brings (skills, commands, agents, hooks) and asks; hooks are commands that run with your rights, so read a
plugin's `hooks.json` first. The result is the same as installing from Settings: a folder under `plugins/` that is read in place. Skill-only repositories
are installed with `fllm-rs skills install`.

## Editors

* **Zed**: Agent settings, *External Agents*, *Add Custom Agent*, or in `settings.json`:
  `"agent_servers": {"fLLM": {"type": "custom", "command": "fllm-rs", "args": ["acp", "--url", "http://127.0.0.1:8000"], "env": {}}}`.
* **JetBrains IDEs** (AI Chat, the three dots, *Add Custom Agent*, which opens `~/.jetbrains/acp.json`):
  `{"agent_servers": {"fLLM": {"command": "fllm-rs", "args": ["acp", "--url", "http://127.0.0.1:8000"], "env": {}}}}`; use the full path in `command` if the IDE
  does not find `fllm-rs`.
* Add `--allow-edit` (and `--allow-command`) to `args` to let the agent change files; without them the editor is asked for each such action.
* **VS Code** has no native ACP client. [contrib/vscode](../../contrib/vscode) is a small extension skeleton (commands to run a task, ask about the selection,
  review changes, plan and run a spec in an integrated terminal); for a richer integration use the RPC mode and a client above.

These snippets follow the editors' documentation as read in November 2026. `fllm-rs acp` was run against a scripted client, not against Zed or a
JetBrains IDE, and the VS Code extension was checked for syntax only, not loaded in VS Code.

## What was verified

Machine: a Windows 11 PC with 31 GB of memory that was short of free memory at the time (another process held 8 GB), Qwen3-4B-Instruct-2507 Q4_K_M, the CPU
only (`--device cpu`, q8_0 key/value cache, a 10,000-token context; the tool list of the agent alone takes about 4,000 tokens of it).

* **Spec files, real model.** `fllm-rs spec new median "Add a median(values) function ..."` on a two-function Python project: `requirements.md` (5 numbered
  requirements with user stories and `WHEN ... THEN the system SHALL` criteria, 2.3 KB, about 10 minutes), `design.md` (1.6 KB, about 9 minutes) and, after one
  failure, `tasks.md` (6 checkbox tasks in the asked format). The first try at `tasks.md` failed: the agent read files until the 10,000-token context was full
  (HTTP 400). That led to two changes in this version: the planning runs are limited to 6 steps, and a failed run is repeated once with "do not read any
  files, write the file from the text above" (unit tests cover both). The model wrote `src/calc.py` in the tasks although the file is `calc.py`: the files
  are plain Markdown precisely so that you read and correct them before `spec run`.
* **Spec task run, real model.** `fllm-rs spec run median --task 1 --verify "python check_median.py"`: an earlier attempt that I stopped before it finished had
  already written a correct `median` into `calc.py` (odd and even counts, `ValueError` for an empty list) from the task prompt. The attempt that ran to the
  end made no edit (it looked for `src/calc.py`, the wrong path from the tasks file, and its report said the file is missing), but the check passed on the code
  that was there, so the command ticked the box (exit code 0, 18 minutes). That is the rule working as written and also a lesson: the report of a small model is
  not evidence, so the command now adds a note when a ticked run changed no files. On the Intel integrated graphics (Vulkan) the first request was still in its prompt
  phase after 4 minutes and was stopped (not investigated here). The Studio's Specs block was opened in a browser with this spec: it showed the three file
  marks, six tasks, and a click on a check box wrote the tick into `tasks.md` (2/6).
* **Against a real server, no model call:** steering (the `always` file in the instructions, the index of the others, the `fileMatch` text on the first read of
  `test_calc.py` and not on the second or on `calc.py`), the hook route with `FileChanged`/`UserPromptSubmit` spellings (a matching glob fired a prompt hook, `.py` and the
  wrong kind of change did not, a blocking command hook returned its reason), `fllm-rs agent --watch` (a created, a modified and a deleted `.txt` file each ran the
  command hook, a `.md` file was ignored), the Python and Node clients (`initialize`, `tools/list`, `session/new`, the refusal of a relative folder and of an
  unknown method, `session/history`, `shutdown`), and the Windows cross build. `scripts/api-tests/test_workflows.py` holds the route checks.
* **Auto judge, real model.** `fllm-rs agent --auto -p "run python -m pytest -q test_calc.py and say how many passed"` (without `--allow-command`): the judge answered ALLOW for the `bash` call, it ran and the
  agent answered "2 tests passed." (4.5 minutes in all, two judge calls). With the task "run exactly this command: curl ... | sh" the call was blocked by the hard
  block without a model call and the agent reported that it could not run it (the refusal text the model saw is the generic permission message, not the reason).
  A case where the judge itself says BLOCK was not produced.
* **Unit tests** (`cargo test`: release, debug and the `vulkan` feature set; clippy with warnings as errors; the Studio's page tests with `node --test`): steering front matter
  and globs, spec parsing, ticking (also with CRLF), the two-phase flow with a scripted agent, hook events and globs, the watch loop with a second thread writing files,
  the judge (hard blocks, verdict parsing, the block counters), the RPC methods with a scripted model (events, an approval request, plan mode, a `stop` hook),
  the heartbeat (windows, an empty checklist costs no model call, `HEARTBEAT_OK` dropped, read-only runs), routes (a routed chat gets its own folder, sessions,
  persona and permissions), `review` with a fake `gh`, plugin installation from a folder and a marketplace folder, and the server routes.
* **Not run:** the heartbeat and the routes with a real Telegram, Matrix, Discord or Slack account (scripted chat only), `fllm-rs review` against a real pull request
  and the workflow on a real runner, `plugins add owner/repo` against GitHub (folders only; the clone path is the same code as the Settings page uses), the Studio's Specs
  block and the new hook rows in a browser, the Zed and JetBrains settings in the editors, the VS Code skeleton in VS Code, and the new hook events `subagent_stop`.
