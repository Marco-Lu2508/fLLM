# Code mode compared with Hermes Agent, Claude Code and Codex CLI

This page lists what the coding agent of fLLM (the Studio *Code* surface, `fllm-rs agent`, `fllm-rs gateway`, hooks, skills,
plugins, sub-agents, scheduled tasks, MCP) does and does not do next to three agents it is measured against:
[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research), [Claude Code](https://code.claude.com/docs/en/overview)
and [Codex CLI](https://github.com/openai/codex). The reference columns come from their public documentation and READMEs; they were
read in October 2026 and are summaries, not a promise that every detail is current. The fLLM column was checked against the code
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
| Memory | agent-curated, user model (Honcho) | `/memory`, auto memory | no | **works**: `memory` tool, `/remember`, `/memory`. A learned user model (Honcho style) is **missing** | low |
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
| Skills hub | Skills Hub | marketplace for plugins | no | **partly**: built-in library of six, import from URL, (*new*) `fllm-rs skills install OWNER/REPO` for any repository with `SKILL.md` files. No hosted search page: a hub needs a service that someone runs | low |
| Custom slash commands | yes | `.claude/commands` | custom prompts | **works**: `.fllm/commands/*.md` with `$ARGUMENTS` | done |
| Hooks | plugin hooks | many events | notify | **partly**: five events (`pre_tool_call`, `post_tool_call`, `on_file_edit`, session start and end). Missing: prompt submit, stop, notification, before compaction | medium |
| Plugins | yes | yes | yes | **works**: folders with skills, commands, hooks, agents | done |
| Sub-agents | delegation | subagents | agents | **works**: `task` tool, own agents in `agents/`, parallel runs | done |
| MCP client | yes | yes | yes | **works**: tools, resources | done |
| MCP server | ACP/MCP | `claude mcp serve` | `codex mcp-server` (replaced by app-server) | **works** (*new*): `fllm-rs mcp-server` with the agent tools and `fllm_agent` | high |
| ACP (editor protocol) | yes | via adapter | via adapter | **works** (*new*): `fllm-rs acp`. Checked with a scripted client only, not with a real editor | medium |
| Programmatic access from scripts | tool RPC from scripts | Agent SDK, `-p` | `codex exec`, SDK | **partly**: `fllm-rs agent -p`, `--output-format json` and `stream-json` (*new*), the MCP server; the tool route `/api/studio/agent/tool` on this machine. No separate SDK | medium |
| Headless mode | `-q` | `-p`, `--output-format` | `exec --json` | **works**: `fllm-rs agent -p`, `--output-format text|json|stream-json` (*new*), exit codes, stdin | done |
| Self-review before "done" | no | `/review`, GitHub review | `/review`, `codex review` | **works**: `/review`, `/security-review`; (*new*) automatic *Review before done* (Settings box, `/autoreview on`) and `--review` with `--verify COMMAND` | high |
| Worktrees | no | `--worktree` | `exec --worktree` | **works**: `/worktree`, `/batch` | done |
| Scheduled / recurring tasks | cron in plain language | `/loop`, cloud tasks | no | **partly**: schedules (`daily at 08:30`, cron lines), only while the app runs; no free-text parsing; results to webhook, ntfy, Telegram | medium |
| GitHub Action / CI | no | yes | yes | **missing**: `fllm-rs agent` can run in CI (documented) but there is no published action | low |
| Cloud tasks | Modal, Daytona | Claude Code on the web | Codex cloud | **missing** on purpose: fLLM runs on your hardware | low |

### Where the agent can be reached

| Feature | Hermes | Claude Code | Codex | fLLM today | Priority |
| --- | --- | --- | --- | --- | --- |
| Terminal UI | yes | yes | yes | **partly**: `fllm-rs agent` runs one task and prints; `fllm-rs run` is a plain chat. No full-screen TUI | medium |
| Messaging | Telegram, Discord, Slack, WhatsApp, Signal, Email | channels (Telegram, Discord), Slack app | no | **partly**: Telegram, Matrix (before); Discord, Slack (*new*). WhatsApp, Signal, Email missing | medium |
| IDE | ACP | VS Code, JetBrains | IDE extension | **partly**: ACP (*new*); no VS Code extension. [coding.md](user/coding.md) shows how to point other agents at fLLM | medium |
| `doctor` / `update` | yes | `/doctor`, `claude update` | `doctor`, `update` | **works** (*new*): `fllm-rs doctor`, `fllm-rs update --check` (looks, never replaces) | medium |
| Import from other agents | `import-agent` | no | no | **works** (*new*): `fllm-rs import claude|codex|hermes` (dry run by default). Formats taken from the documentation, not checked against a real installation | medium |
| Status line | no | yes | no | **missing**: the Code page has a context meter | low |

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

## Why some things stay missing

* **WhatsApp, Signal, Email.** WhatsApp needs a Meta business account and a webhook that is reachable from the internet; Signal has no
  official bot interface (the community route is a separate `signal-cli` daemon); Email needs IMAP and SMTP clients with their
  own TLS handling. None of the three can be done cleanly with a bot token and plain HTTPS polling, which is what Discord, Slack,
  Telegram and Matrix use here.
* **Modal, Daytona, Vercel sandboxes, cloud tasks.** They are paid services with accounts. fLLM is built to run on hardware you own;
  a Docker or SSH backend reaches your own machines.
* **A skills hub page.** A search page needs a hosted index. Installing from any repository with `fllm-rs skills install` covers the
  use without running a service.
* **More hook events, a status line, a TUI, a VS Code extension, a published GitHub Action.** Real work with real value, not done in
  this change; the order above says which matter most.
* **Honcho-style user modelling.** It depends on a second model call after every session; with local models that costs more than
  the memory tool and `SOUL.md` give back.

## What was run

* Unit tests for every new module (rules, extras, import, skills, operations, MCP server, ACP, gateway chats, pairing, terminal
  backends) and for the page logic (`node --test`).
* A real small model (Qwen2.5-Coder-3B-Instruct, Q4_K_M) driving `fllm-rs agent` and the Studio on a Windows PC: see the verified
  list in [agent-cli.md](user/agent-cli.md#what-was-verified) and [studio.md](user/studio.md).
* Not run against the real service: Discord, Slack (stub servers only), Telegram and Matrix as before; ACP against a real editor;
  the importers against real installations of the other agents; the Docker and SSH backends beyond argument building and, where
  noted, a local Docker run.
