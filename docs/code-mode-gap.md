# Code mode compared with Hermes Agent, Claude Code, Codex CLI, Kiro, Pi and OpenClaw

This page lists what the coding agent of fLLM (the Studio *Code* surface, `fllm-rs agent`, `fllm-rs gateway`, hooks, skills,
plugins, sub-agents, scheduled tasks, MCP) does and does not do next to three agents it is measured against:
[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research), [Claude Code](https://code.claude.com/docs/en/overview)
and [Codex CLI](https://github.com/openai/codex); the [second matrix](#second-matrix-kiro-pi-openclaw-and-newer-claude-code-features) adds
[Kiro](https://kiro.dev/docs/), [Pi](https://github.com/badlogic/pi-mono), [OpenClaw](https://github.com/openclaw/openclaw) and what Claude Code,
Hermes and Codex shipped since. The reference columns come from their public documentation and READMEs; the first matrix was read in October 2026,
the second in early November 2026. They are summaries, not a promise that every detail is current. The fLLM column was checked against the code
and, for everything marked *new*, run for real (see [What was run](#what-was-run)).

Marks in the fLLM column: **works** (used and tested), **partly** (some of it, the gap is named), **missing** (not there; the reason
is in the last section). *new* marks what was added together with this page. Priority: how much it matters for a coding agent that
runs on a local model (high, medium, low).

Local models are weaker than the models these agents were built for. Where a feature costs extra model calls (self-review, skill
proposals) it is off by default and the good default for a small model is named in [the feature pages](user/agent-cli.md).

## Matrix

### The loop and its tools

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Read, search, edit, patch files; notebooks | yes (toolsets) | yes | yes (`apply_patch`) | **works**: `list_dir`, `glob`, `grep`, `read_file`, `write_file`, `edit_file`, `multi_edit`, `patch`, `notebook_edit` | done |
| Shell with background jobs | yes | yes | yes | **works**: `bash` (`background`), `bash_output`, `bash_kill`; a `cd` stays in effect | done |
| Web fetch and search | yes | yes | yes (`--search`) | **works**: `web_fetch`, `web_search` (no key; private addresses refused) | done |
| Language diagnostics | via MCP | IDE diagnostics | no | **works**: `diagnostics` (fixed checks), `lsp` (rust-analyzer, pyright, tsserver, gopls, clangd) | done |
| Browser and desktop control | yes | Chrome extension | no (app only) | **works**: `browser_*`, `computer_*` (Studio only; never offered headless) | done |
| Picture input | yes (vision) | yes (paste) | yes (`-i`) | **works** (*new*): paste or attach in the Code box, `fllm-rs agent --image FILE`. Needs a model that can see; others are told so | medium |
| Toolsets (enable groups of tools) | yes (`hermes tools`) | `--allowedTools` | config | **partly**: tools are grouped by kind of action (read, edit, command, network, memory) and each kind is allowed or not; no named toolsets | low |

### Permissions, sandbox, isolation

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Allow / deny / ask rules with patterns | command allowlist | `permissions.allow/deny/ask` with patterns | execpolicy rules, approval modes | **works**: Studio rules table and `/allow` `/deny`; headless (*new*) `permissions.json` with `allow`, `deny`, `ask` lists, `--allow-rule`, `--deny-rule`, `--ask-rule`; the Claude Code spelling (`Bash(npm test:*)`) is read too | high |
| Settings hierarchy | profiles | managed > user > project > local | `config.toml` + profiles | **partly** (*new*): rules from the Studio folder, then `.fllm/permissions.json`, then `.fllm/permissions.local.json`; other settings are per Studio folder and per `--profile` | medium |
| Plan mode (read only, user approves) | no | yes | read-only approval mode | **works**: Studio *Plan* agent with *Run the plan*; headless (*new*) `--plan` (also `"mode": "plan"` in a settings file), on a terminal it asks before running the plan | high |
| Approval prompts | yes | yes | `-a on-request` | **works**: Studio asks per action; headless `--ask` asks in the terminal, `ask` rules ask even with a flag | done |
| Skip all questions | yolo | `--dangerously-skip-permissions` | `--yolo` | **works**: `--yes`, Studio *Turbo* (`/turbo on`) | done |
| Command sandbox | container | bash sandbox (bubblewrap, Seatbelt) | Seatbelt, Landlock, Windows | **partly**: bubblewrap on Linux; restricted token and job object on Windows (one machine checked, off by default); Seatbelt on macOS (unverified). No Landlock | medium |
| Terminal backends | local, Docker, SSH, Modal, Daytona, Vercel | local | local | **partly** (*new*): local, Docker (`FLLM_TERMINAL=docker:IMAGE`), SSH (`ssh://host/ROOT`). Modal, Daytona, Vercel missing | medium |
| Who may use the bot | DM pairing, allowlists | n/a | n/a | **works** (*new*): `--chat` list (before), `--user` list and `--pair` one-time codes with `fllm-rs gateway pair approve` | high |

### Sessions, memory, context

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Project instructions | context files | `CLAUDE.md` (reads `AGENTS.md`) | `AGENTS.md` | **works**: `AGENTS.md`, `FLLM.md`, `CLAUDE.md`, nearest first | done |
| Memory | agent-curated, user model (Honcho) | `/memory`, auto memory | no | **works**: `memory` tool, `/remember`, `/memory`; (*new*) a reflective step saves entries after a run and every 12 tool steps, deduped and with the source session, and keeps an editable profile of the user (on by default, see [Self-improvement](#self-improvement-learning-from-use)). A Honcho-style dialectic model is **missing** | done |
| Context compression | `/compress` | `/compact`, auto | auto | **works**: `/compact`, automatic at 70% of the context | done |
| Cross-session search | FTS5 with model summaries | `/resume` search | `resume` picker | **works** (*new* for the agent): `session_search` tool, `/search`, `fllm-rs sessions search`; every headless run is indexed. Ranking is BM25 (no SQLite); the "summary" is the start and end of the session, not a model summary | high |
| Resume, continue, fork | yes | `-c`, `-r`, `/resume` | `resume`, `fork` | **works**: Studio `/resume`, `/fork`; headless (*new*) `--continue`, `--resume NAME`, `--session NAME`. Forking a headless session is **missing** (copy the file) | medium |
| Checkpoints and rewind | yes | `/rewind` | no | **works** in the Studio (`/rewind`, `/undo`, `/revert`, `/retry`); **missing** headless | medium |
| @-file mentions | yes | yes (fuzzy) | `@` | **works** in the Studio (fuzzy over the folder); headless: write the path | done |
| Output style / personality | `/personality`, `SOUL.md` | output styles | no | **works**: `SOUL.md` and profiles (headless); Studio (*new*) five styles, `/personality` | medium |
| Thinking level | yes | thinking budget | reasoning effort | **works**: the chat page level is sent to models that support it | done |
| Usage and insights | `/usage`, `/insights` | `/cost`, `/context` | `/status` | **works**: Studio `/cost`; (*new*) `/insights`, `fllm-rs insights`, `fllm-rs usage` over saved sessions. Counts are characters / 4, not the model's own | medium |
| `/retry`, `/undo`, `/stop` | yes | Esc, `/rewind` | Esc | **works**: Studio `/undo`, (*new*) `/retry`, `/stop`, Esc | done |

### Skills, extensions, agents

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Skills as `SKILL.md` | agentskills.io | yes | yes | **works**: library, user and project folders, plugins; (*new*) `fllm-rs skills import/export/install` in both directions | done |
| Skills the agent creates itself | yes, autonomously | no (skill-creator on request) | no | **works**: Studio `/skill` (user approves); headless (*new*) `--propose-skills` waits in `skills-proposed/` for `fllm-rs skills approve`, `--auto-skills` saves at once | high |
| Skills improve during use | "improves them during use" | no | no | **works** (*new*): usage log per skill, automatic revision after a failure or five uses with checks, history, automatic and manual rollback (below) | high |
| Skills hub | Skills Hub | marketplace for plugins | no | **partly**: built-in library of six, import from URL, `fllm-rs skills install OWNER/REPO` for any repository with `SKILL.md` files, (*new*) `fllm-rs plugins add owner/repo` for plugin repositories and for git repositories that act as a marketplace (`marketplace.json`). No hosted search page: a hub needs a service that someone runs | low |
| Custom slash commands | yes | `.claude/commands` | custom prompts | **works**: `.fllm/commands/*.md` with `$ARGUMENTS` | done |
| Hooks | plugin hooks, `pre_llm_call` | many events (33 in the newest docs) | notify | **works** (*new*): eleven events (`pre_tool_call`, `post_tool_call`, `on_file_edit`, session start and end, `user_prompt_submit`, `stop`, `subagent_stop`, `notification`, `pre_compact`, `file_changed`), a hook can be a command or a prompt for the agent, Claude Code and Kiro spellings are read, `fllm-rs agent --watch` fires file events. Not there: the HTTP, MCP-tool and model-judged hook kinds, rewriting a tool call's arguments, most of the 33 events (permission, worktree, elicitation ...) | medium |
| Plugins | yes | yes | yes | **works**: folders with skills, commands, hooks, agents | done |
| Sub-agents | delegation | subagents | agents | **works**: `task` tool, own agents in `agents/`, parallel runs | done |
| MCP client | yes | yes | yes | **works**: tools, resources | done |
| MCP server | ACP/MCP | `claude mcp serve` | `codex mcp-server` (replaced by app-server) | **works** (*new*): `fllm-rs mcp-server` with the agent tools and `fllm_agent` | high |
| ACP (editor protocol) | yes | via adapter | via adapter | **works** (*new*): `fllm-rs acp`. Run over a pipe with a scripted client and a real model; not tried in a real editor | medium |
| Programmatic access from scripts | tool RPC from scripts | Agent SDK, `-p` | `codex exec`, SDK | **works** (*new*): `fllm-rs agent -p`, `--output-format json` and `stream-json`, the MCP server, and `fllm-rs agent --rpc` (JSON-RPC over standard input and output) with a Python and a TypeScript client in `scripts/sdk/` | done |
| Headless mode | `-q` | `-p`, `--output-format` | `exec --json` | **works**: `fllm-rs agent -p`, `--output-format text|json|stream-json` (*new*), exit codes, stdin | done |
| Self-review before "done" | no | `/review`, GitHub review | `/review`, `codex review` | **works**: `/review`, `/security-review`; (*new*) automatic *Review before done* (Settings box, `/autoreview on`) and `--review` with `--verify COMMAND` | high |
| Worktrees | no | `--worktree` | `exec --worktree` | **works**: `/worktree`, `/batch` | done |
| Scheduled / recurring tasks | cron in plain language | `/loop`, cloud tasks | no | **partly**: schedules (`daily at 08:30`, cron lines), only while the app runs; no free-text parsing; results to webhook, ntfy, Telegram. (*new*) The gateway's heartbeat wakes the agent on an interval with a checklist (`HEARTBEAT.md`) and messages only when something needs attention | medium |
| GitHub Action / CI | no | yes | yes | **partly** (*new*): `fllm-rs review --pr N [--comment]` (through `gh`) and `.github/workflows/fllm-review.yml.example`; `fllm-rs agent` runs in CI as before. No published action; the workflow has not run on a real runner | low |
| Cloud tasks | Modal, Daytona | Claude Code on the web | Codex cloud | **missing** on purpose: fLLM runs on your hardware | low |

### Self-improvement (learning from use)

Hermes calls itself the self-improving agent: its README lists skills created after complex tasks, skills that improve during use, memory
with periodic nudges, a deepening user model (Honcho) and session search with model summaries. fLLM does the same steps **by default and
without asking**, with checks in front of every write and an undo behind it, and with prompts that a 3B to 4B model can follow. How to switch it off:
`--no-learn`, `fllm-rs learn off` (per project: `--folder DIR`), `FLLM_NO_LEARN=1`, the toggle in the Code settings.

| Hermes claim | fLLM today |
| --- | --- |
| Creates skills after complex tasks | **works** (*new*): a run that finished, passed its `--verify`, used at least 6 tool calls and no skill is turned into a skill by the model (JSON schema, validated, secrets refused) and saved at once |
| Skills improve during use | **works** (*new*): each use is logged (outcome, `--verify`, retries, thumbs); a failure, a thumb down, many retries or five uses make the agent revise the skill itself. Applied after validation and a second model call that checks the revision; the old text is kept; **automatic rollback** when one of the next two uses fails or gets a thumb down; a skill a person wrote or edited is never overwritten (sibling `NAME-agent`) |
| Nudges itself to persist knowledge | **works** (*new*): reflective step after the run and every N tool results; notes are saved at once (dedup by normalised words plus BM25, no embeddings; secrets refused; at most 5 per run; 100 per file); `learn forget` |
| Deepening user model (Honcho dialectic) | **partly** (*new*): short dated lines in a plain `user-profile.md`, derived only from the user's own messages (a line must rest on their words), editable, exportable, deletable, in the prompt within 1,200 characters. No dialectic reasoning, no second model, nothing hidden |
| Session search with LLM summaries | **partly**: BM25 search of sessions as before; the summary is still the start and end of a session, not a model summary |
| Transparency | **works** (*new*): `fllm-rs learn log` and the *What I learned* feed in the Studio list every automatic change with time, session, reason, diff and one-step Undo; a `Learned: ...` line after a run and a toast in the Studio |
| Runs on weak models | **works** (*new*): `response_format` JSON schema, fallback without it, validation (nothing is written when the answer does not fit), token and time caps, per-run write limits, run after the answer; proof and limits in [What was run](#what-was-run) |
| Scheduled housekeeping | **works** (*new*): a schedule task "Consolidate memory and skills" (you add it; nothing is scheduled by default) |

Safety nets that replace the question: validation and the second model call before a revision is applied, the previous text in the history, automatic rollback, protection of
hand-written skills, redaction and caps, the log with Undo, `--ask-first` for those who want the earlier proposals-only behaviour. What they do not cover: a small model can write a
plausible but wrong note or skill that no check catches; the log and the undo are the remedy, so read `fllm-rs learn log` now and then.

### Where the agent can be reached

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Terminal UI | yes | yes | yes | **partly**: `fllm-rs agent` runs one task and prints; `fllm-rs run` is a plain chat. No full-screen TUI | medium |
| Messaging | Telegram, Discord, Slack, WhatsApp, Signal, Email | channels (Telegram, Discord), Slack app | no | **partly**: Telegram, Matrix (before); Discord, Slack; (*new*) routes give chats their own profile, folder and permissions, and a heartbeat sends alerts. WhatsApp, Signal, Email missing | medium |
| IDE | ACP | VS Code, JetBrains | IDE extension | **partly**: ACP; (*new*) tested-against-the-docs configuration for Zed and JetBrains in [agent-workflows.md](user/agent-workflows.md#editors), a VS Code extension skeleton in `contrib/vscode`, and the RPC mode for your own integration. [coding.md](user/coding.md) shows how to point other agents at fLLM | medium |
| `doctor` / `update` | yes | `/doctor`, `claude update` | `doctor`, `update` | **works** (*new*): `fllm-rs doctor`, `fllm-rs update --check` (looks, never replaces) | medium |
| Import from other agents | `import-agent` | no | no | **works** (*new*): `fllm-rs import claude|codex|hermes` (dry run by default). Formats taken from the documentation, not checked against a real installation | medium |
| Status line | no | yes | no | **missing**: the Code page has a context meter | low |

## Second matrix: Kiro, Pi, OpenClaw and newer Claude Code features

Read in November 2026 from the documentation of each tool (Kiro: specs, steering, hooks; Pi: README and its sessions, slash-command and RPC pages;
OpenClaw: README and heartbeat page; Claude Code: hooks, permission modes and the weekly digest up to week 37; Codex 0.166; Hermes 0.22.6), then
checked against the code of fLLM. "Who has it" names the tools whose documentation describes the feature. *Decision* says what was done on this
page's change (**built**), what stays out and why (**skip**), or what is left for later (**later**).

| Feature | Who has it | fLLM today (checked in the code) | Priority | Decision |
| --- | --- | --- | --- | --- |
| Spec-driven development: requirements, design, tasks files, tasks run one by one | Kiro | before: none. Now `fllm-rs spec new/run/list/tick`, `/spec` and a Specs block in the Studio side panel (`rust/engine/src/spec.rs`) | high | **built**; the runner ticks the box, not the model |
| Bugfix specs (current, expected, unchanged behaviour) | Kiro | none; a bug is a spec whose requirements describe the failing case | low | **skip**: same files, other headings |
| Run independent spec tasks in parallel waves | Kiro | tasks run one after the other | low | **skip**: a local model serves one run at a time anyway |
| Steering files with inclusion modes (always, fileMatch, manual, auto), user and workspace level | Kiro | before: `AGENTS.md`, `FLLM.md`, `CLAUDE.md` only. Now `.fllm/steering/*.md` and `steering/` of the Studio folder (`steering.rs`) | high | **built**; no `#[[file:...]]` references |
| `AGENTS.md` in sub folders | Kiro, Codex, Claude Code | nearest first up to the repository root (`agent_mem.rs`) | done | exists |
| Hooks on file events (save, create, delete) with a prompt or a command | Kiro, Claude Code (`FileChanged`) | before: `on_file_edit` for the agent's own edits only. Now `file_changed` and `fllm-rs agent --watch` (`agent_watch.rs`) | high | **built** (polling, no OS watcher) |
| Hook events for prompt submit, stop, sub-agent stop, notification, pre-compact | Claude Code, Kiro (`Stop`), Hermes (`pre_llm_call`) | before: five events. Now eleven (`hooks.rs`); headless and RPC fire `user_prompt_submit` and `stop`, the Studio fires those plus `notification` and `pre_compact` | medium | **built**; `subagent_stop` is declared but nothing fires it yet (**later**) |
| Hook kinds beyond a command: HTTP, MCP tool, model-judged, sub-agent | Claude Code | a command, or a prompt text for the agent | low | **skip**: a command can call curl; a judged hook costs a model call per event |
| Hooks rewriting tool input or output (`updatedInput`) | Claude Code | no | low | **skip** |
| Autopilot and supervised modes | Kiro | Plan, ask, auto-edit, Turbo in the Studio; flags and rules headless | done | exists |
| Auto mode: a classifier decides risky actions | Claude Code | before: none. Now `fllm-rs agent --auto` (`agent_auto.rs`): rules first, hard blocks, then a short model call that sees the task and the action but no tool results | high | **built** |
| Checkpoints and rewind | Kiro, Claude Code | Studio `/rewind`, `/undo`, `/revert`; none headless | medium | **later** (headless) |
| Context providers `#file`, `#codebase`, `#docs` | Kiro | `@file` mentions with fuzzy search and `!command` in the Studio; no semantic code index, no docs index | medium | **later**: an index needs embeddings and a store |
| Powers (tools plus knowledge that load on demand) | Kiro | skills load on demand (the `skill` tool), plugins bundle skills, commands, agents and hooks | low | **skip**: covered by plugins and skills |
| Ignore file for secrets | Kiro (`.kiroignore`) | `.gitignore` handling and deny rules (`agent_ignore.rs`, permission rules) | done | exists |
| Extensions in TypeScript that add tools, commands and events | Pi | plugins add skills, commands, agents and hooks; MCP servers add tools in any language; no in-process script API | medium | **later**: a script-tool type needs its own permission design; MCP covers it for now |
| Session tree: branch inside one file (`/tree`), `/fork`, `/clone` | Pi | Studio `/fork` copies a session, `/rewind` goes back; no tree view | low | **skip** |
| Prompt templates | Pi, Claude Code | `.fllm/commands/*.md` with `$ARGUMENTS`, plugin commands | done | exists |
| Packages of extensions, skills and templates shared through npm or git | Pi, Claude Code marketplaces | before: install by folder or one git URL. Now `fllm-rs plugins add owner/repo`, repositories with `marketplace.json` as indexes (`plugins_cli.rs`) | medium | **built** |
| RPC mode and an SDK for embedding | Pi, Claude Code (Agent SDK), Codex | before: `-p`, stream-json, MCP server, ACP. Now `fllm-rs agent --rpc` plus Python and TypeScript clients in `scripts/sdk/` | high | **built** |
| Minimal system prompt for small models | Pi | compact prompts with explicit steps for small models, a fuller form for large ones (`codeprompts.js`) | done | exists |
| Messages queued while the agent works | Pi, Codex | Studio: Enter queues, Esc stops | done | exists |
| Proactive heartbeat: wake up on a schedule, tell the owner only when needed | OpenClaw | before: schedules deliver results every time. Now the gateway's `--heartbeat` with `HEARTBEAT.md` (`gateway_heartbeat.rs`) | high | **built** |
| Many messaging channels | OpenClaw (20+), Hermes (6) | Telegram, Matrix, Discord, Slack | medium | **skip** for WhatsApp, Signal, iMessage, Email: see the end of this page |
| Multi-agent routing: channels or chats mapped to different agents | OpenClaw, Hermes (profiles) | before: one agent per gateway process. Now `--routes` (`gateway_route.rs`): profile, folder, permissions, model per chat | medium | **built** |
| DM pairing for unknown senders | OpenClaw, Hermes | `--pair`, `gateway pair approve` | done | exists |
| Canvas (an agent-drawn live surface) | OpenClaw | the chat's artifact panel previews HTML, SVG, Markdown and CSV blocks with versions and a sandboxed runner (`artifacts.js`); the agent does not drive it | low | **skip**: partly covered |
| Companion apps for phone and desktop (nodes), voice wake | OpenClaw | the Studio runs in a browser and in the desktop app; voice input and output in the chat (`voice.js`); no phone app, no wake word | low | **skip**: apps are a product of their own |
| Remote control and phone access to a running session | Claude Code (`remote-control`), Hermes (`webapp`) | the Studio can be opened from another device only with an API key; no pairing flow, no QR code | medium | **later**: needs a pairing design (keys, read-only first); not done to avoid a half-safe version |
| Background agents with a board of tasks | Claude Code (`claude agents`, background sub-agents), Kiro | sub-agents run in parallel inside one run; `/batch` runs a task in its own worktree; no board | medium | **later** |
| Status line and themes in the terminal | Claude Code, Pi | the Studio shows a context meter; there is no full-screen terminal UI to put a line into | low | **skip** |
| Output styles | Claude Code | five styles in the Studio (`/personality`), `SOUL.md`; a manual steering file can be added with `--steering NAME` | medium | exists (partly) |
| GitHub integration: review pull requests, comment, run in CI | Claude Code (Action, `ultrareview`), Codex, Kiro | before: none. Now `fllm-rs review --pr N --comment` and a workflow example | medium | **built** (not run on a real runner) |
| IDE integration | Claude Code, Kiro, Codex, Zed through ACP | ACP agent, configuration for Zed and JetBrains, a VS Code skeleton | medium | **built** as documentation and a skeleton |
| Sub-agents with their own model | Claude Code, Kiro | `.fllm/agents/NAME.md` with `tools`, `model`, `max_steps` | done | exists |
| Memory the agent writes itself | Claude Code (auto memory), Hermes | `--learn`, `/remember`, proposals with approval | done | exists |
| Per-skill locks against autonomous writes | Hermes 0.22 | skill changes always wait for approval; no per-skill lock | low | **skip**: approval already covers it |
| Scheduled routines in the cloud | Claude Code, Codex | local schedules and the heartbeat; no cloud | low | **skip** on purpose |
| Skills with their own tool limits and model | Claude Code | `SKILL.md` carries a name, a description and a body | low | **later** |

**Other tools in this class**, one line each for what they have that is not above (from their public descriptions; these were not re-read for this
page): Cline (a plan/act split with step-by-step approval inside the editor, and a browser tool), Aider (a repository map, automatic git commits for
every change, voice input), OpenCode (a client/server split so a terminal UI and other front ends share one session, a themeable TUI), Goose
(recipes: shareable, parameterised agent runs, and MCP as the extension mechanism), Continue (rules files and a model-agnostic autocomplete), Cursor
agent (background agents on cloud machines, a checkpoint per prompt), Amp (threads that can be shared and searched by a team), Droid (specialised
droids for review, migration and incident work). None of these needed a change beyond the table; the repository map of Aider and the shared threads of
Amp are the two ideas that may be worth a later look.

## What was added with this page

| Area | Command or place | Documentation |
| --- | --- | --- |
| Permission rules, plan mode, settings files | `fllm-rs agent --plan`, `--allow-rule`, `permissions.json` | [agent-cli.md](user/agent-cli.md) |
| Self-review | Code settings, `/autoreview`, `fllm-rs agent --review` | [agent-cli.md](user/agent-cli.md), [studio.md](user/studio.md) |
| Output formats, resume, pictures | `--output-format stream-json`, `--continue`, `--resume`, `--image` | [agent-cli.md](user/agent-cli.md) |
| Session search | `session_search` tool, `/search`, `fllm-rs sessions` | [agent-cli.md](user/agent-cli.md) |
| Skills | `fllm-rs skills ...`, `--propose-skills` | [agent-cli.md](user/agent-cli.md) |
| MCP server and ACP | `fllm-rs mcp-server`, `fllm-rs acp` | [agent-cli.md](user/agent-cli.md), [coding.md](user/coding.md) |
| Gateways | `fllm-rs gateway discord|slack|pair` | [agent-cli.md](user/agent-cli.md) |
| Terminal backends | `FLLM_TERMINAL`, `terminal.json` | [studio.md](user/studio.md) |
| Everyday commands | `doctor`, `update --check`, `insights`, `usage`, `import` | [agent-cli.md](user/agent-cli.md) |
| Studio | `/search`, `/retry`, `/stop`, `/insights`, `/personality`, pictures | [studio.md](user/studio.md) |
| Specs, steering, hooks, auto mode, heartbeat, routes, RPC mode, review, plugins from git | `fllm-rs spec`, `.fllm/steering/`, `agent --watch`, `agent --auto`, `gateway --heartbeat --routes`, `agent --rpc`, `review`, `plugins add` | [agent-workflows.md](user/agent-workflows.md) |

## Why some things stay missing

* **WhatsApp, Signal, Email.** WhatsApp needs a Meta business account and a webhook that is reachable from the internet; Signal has no
  official bot interface (the community route is a separate `signal-cli` daemon); Email needs IMAP and SMTP clients with their
  own TLS handling. None of the three can be done cleanly with a bot token and plain HTTPS polling, which is what Discord, Slack,
  Telegram and Matrix use here.
* **Modal, Daytona, Vercel sandboxes, cloud tasks.** They are paid services with accounts. fLLM is built to run on hardware you own;
  a Docker or SSH backend reaches your own machines.
* **A skills hub page.** A search page needs a hosted index. Installing from any repository with `fllm-rs skills install` covers the
  use without running a service.
* **A status line, a TUI, a published GitHub Action.** A status line needs a full-screen terminal UI to live in; the Studio has its meter. The review
  workflow is an example file, because an action would need a place to host and version it.
* **Script tools, remote access from a phone, a board of background agents, a semantic code index.** Judged in the second matrix as worth doing and left for
  later: a script tool type needs its own permission design (MCP servers cover tools in any language meanwhile), phone access needs a pairing design with read-only
  first, the board is a Studio feature of its own, and `#codebase` needs an embeddings index.
* **Per-skill tool limits, `subagent_stop` firing, HTTP and model-judged hook kinds, rewriting tool input in a hook.** Small, and each adds more surface than it
  saves for a local model; the event name `subagent_stop` is reserved so that hook files written now keep working.
* **Honcho-style dialectic user modelling.** The profile file is derived from the same reflective step and is fully visible; a separate
  model that reasons about the user over many turns would cost another call per message and hide state, which this design avoids.
* **Model-written session summaries.** They need a call per stored session; the search keeps ranking the text itself.

## What was run

* Unit tests for every new module (rules, extras, import, skills, operations, MCP server, ACP, gateway chats, pairing, terminal
  backends) and for the page logic (`node --test`).
* A real small model (Qwen2.5-Coder-3B-Instruct, Q4_K_M) driving `fllm-rs agent` and the Studio on a Windows PC: see the verified
  list in [agent-cli.md](user/agent-cli.md#what-was-verified). A 3B model completes the fix-the-test task in some runs and not in
  others; that is the model, and the page says where a larger one is needed.
* The learning loop, on by default, with Qwen3-4B-Instruct-2507 Q4_K_M on the same PC (context 8,192): a matching skill was used, notes were saved on their own, and a revision of a hand-written
  skill was written as a sibling skill, listed in the log and undone. The note and the revision were plausible but not sharp, the profile stayed empty, and a third run ended in the server's
  600 s timeout, so a skill written from scratch and an automatic rollback were **not** seen with a real model (unit tests with scripted answers cover them). Details:
  [agent-cli.md](user/agent-cli.md#what-was-verified). The Studio pages were clicked through with Playwright (feed, undo of a note, a skill revision and a profile line, the switch).
  Unit tests cover the usage log, creation, revision with the self-check, the watch and rollback, protection of hand-written skills, dedup, redaction, schema validation, the caps, the log and undo,
  the switches; the routes were run by `scripts/api-tests/test_learn.py` against a server with the tiny test model.
* The additions of the second matrix: unit tests for each module, route checks against a real server, and a real Qwen3-4B-Instruct-2507 run that wrote the
  three spec files (the task run itself did not finish on this machine) and, for `--auto`, a real judge; the list of what was and was not run is at the end of
  [agent-workflows.md](user/agent-workflows.md#what-was-verified).
* The Docker backend against a real Docker daemon on Linux (alpine image).
* Not run against the real service: Discord, Slack (stub servers only), Telegram and Matrix as before; ACP against a real editor;
  the MCP server inside Claude Code or Codex; the importers against real installations of the other agents; the SSH backend beyond
  argument building; picture input with a model that can see.
