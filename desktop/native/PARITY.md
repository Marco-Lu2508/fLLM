# Native app: parity with the Studio

The goal: the native app (this folder) does everything the web Studio (`rust/engine/src/webui`) and the Tauri window did, and then the Studio
and the Tauri app are removed. Every row has a state: `done` (built and tested), `part` (built, something missing, said in the row), `todo`.
The server routes (`/api/studio/*`, `/v1/*`) stay: the native app is a client of them.

Order of work: chat, then the pages people use every day (assistants, projects, providers, usage, voice), then the settings pages, then the
rest of the Code mode, then the app chrome (palette, search, shortcuts, first start, tray). The Studio is removed in the last step, after the
checks at the bottom pass.

## Chat

| Studio feature | State |
|---|---|
| Streaming chat, markdown, code blocks with copy | done |
| Stats line (tokens/s, tokens, first token, prompt speed), context meter | done |
| Stop | done (Esc: todo) |
| Chat list, rename, delete, search over chats | done (date groups, pinned, archived; search in titles and messages) |
| Conversation tree: edit a message / regenerate keeps the old version, arrows between versions | todo |
| Thinking chip (off/low/medium/high, per model), collapsible reasoning | part (chip done, one level for all chats; reasoning shown, collapsing: todo) |
| Default system prompt builder (date, language, model family, tools), response style | part (built from date, language, model family and size, prompt length auto/full/short, your text replaces or is added, clock held 30 min; tools and response style: todo) |
| Assistants (name, icon, color, instructions, sampling, model, suggestions; import/export) | done (six built-ins plus your own ones; the menu under the name in the top bar; editor with icon, color, sampling, model, starters and the per-assistant default prompt; export and import as JSON) |
| Projects (instructions + knowledge base, move chats) | done (a *Projects* section in the sidebar with a row per project, its chats under it and rename/delete; the page with the name, the used characters, the instructions box, the knowledge with *Add files* and its documents, the chats of the project with their dates, *New chat in project* and *Delete project* behind the Studio's confirm dialog; chats move through the picker of the chat menu and of the select row, a new chat joins the open project; the instructions enter the system prompt between the default prompt and the assistant's own text with the Studio's header line, and the best passages of the knowledge travel with the newest question, shown under it; limits and messages follow PRJ_LIMITS and projects.js) |
| Presets / prompt library (`/` in the box) | todo |
| Personalization and memory | todo |
| Sampling panel (temperature, top-p/k, min-p, penalties, mirostat, ...) | done (constraint/grammar and logprobs: todo) |
| Attach images (vision), text files and documents to a chat (PDF, DOCX, ... via `/api/studio/extract`) with the knowledge of the chat itself (the search itself is built: `rag.rs`, projects use it) | todo |
| Use the web / Allow web access (tools) / Research mode | todo |
| Message actions (copy, copy Markdown, read aloud, rewind) | part (copy) |
| Temporary chat | todo |
| Artifacts (HTML/SVG/Markdown/CSV preview), run JavaScript | todo |
| Math (LaTeX) and tables | part (tables via markdown; math: todo) |
| Organizing chats (bulk delete, move, archive, export) | part (select many, delete, archive, duplicate, export Markdown/JSON to Downloads; import and move to a project: todo) |

## Everyday

| Studio feature | State |
|---|---|
| Welcome tour / first start | todo |
| Keyboard shortcuts list | todo |
| Command palette (Ctrl+K) | todo |
| Dictation, reading aloud, alerts | todo |
| Start screen: what the engine is doing right now | todo |
| Usage and cost | todo |
| Session mailbox (inbox) | todo |
| Look: themes, accent, density | part (light/dark) |
| Desktop: tray, autostart, quick-entry hotkey, updates | todo |

## Models and settings

| Studio feature | State |
|---|---|
| Models page: loaded, on disk, search and download from Hugging Face, switch model | done |
| Model picker (file / folder / HF name, inspect first) | part (list; inspect and file dialog: todo) |
| Settings > General / Appearance / Personalization | part |
| Settings > Engine (caches, order of dropping work, telemetry, live controls) | part |
| Settings > Hardware (GPU check, device list, NPU, AMD NPU) | todo |
| Settings > Runtime (sandbox, turbo, automatic approval) | todo |
| Settings > Providers (cloud and other servers, keys) | todo |
| Settings > Local API server, API keys, logs, tokenizer | todo |
| Settings > Connect other apps (Claude Code, Codex, ...) | todo |
| Settings > Search (web search engines) | todo |
| Settings > MCP servers | todo |
| Settings > Skills, scheduled tasks, skill library | todo |
| Settings > Plugins, hooks, sub-agents | todo |
| Settings search | todo |

## Code mode

| Studio feature | State |
|---|---|
| Folder, agent loop, tool cards, approvals, previews, Stop, undo | done |
| Modes: plan / ask / edit freely / everything | done |
| Project instructions, memory, profile; MCP tools; task list; questions | done |
| Sessions kept per folder | done |
| Sub-agents (`task`, custom agents) | todo |
| Hooks | todo |
| Slash commands, skills, `/init` | todo |
| Specs and steering files | todo |
| Review, self-review | todo |
| Worktrees | todo |
| Language servers, diagnostics view | todo |
| Notebooks, MCP resources | todo |
| Browser view (live page) | todo |
| Computer use | todo |
| Terminal view | todo |
| Learning from sessions | todo |

## Removal of the Studio

Done only when every row above is `done` or has been decided out by Marco. Then: the files of `rust/engine/src/webui` and the route that serves
them, `desktop/src-tauri` and its workflows, the packages built from them in `release.yml`, and the docs that describe them; the native bundle
(`native-bundle.yml`) takes over the release. The API routes stay.

Checks before the removal: all tests of the app green on the three systems, a session with a real model (chat, a tool-using code run, a
download) on this computer, the bundle starts the engine by itself.
