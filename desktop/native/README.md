# fLLM app, native (Rust, no web view)

The Studio as a real desktop program: one Rust binary, drawn by [iced](https://iced.rs) (wgpu, with a software renderer as fallback),
no Chromium, no Electron and no web view. The Tauri app in `../src-tauri` shows the web Studio in the system web view; this one
replaces it step by step.

Opening the app is enough: when no server answers, it starts `fllm-rs studio` itself (found by `FLLM_RS`, next to the app, or on the PATH) on a free
port and stops it when the window closes; a server that is already there (`FLLM_URL`, a Docker container, one on port 8080) is used as it is and left
alone. If the server cannot be started the "no server" screen says why, and `server.log` in the data folder has its output. The engine moves into
the app's own process later; the window talks to it the same way.

It talks to an fLLM server over HTTP, the same `/v1` API every other client uses (`GET /v1/models`, `POST /v1/chat/completions` as an event
stream), so everything the server can do is within reach and the window stays small. The server is not inside the window yet (stage 4).

```sh
fllm-rs studio --no-open --port 8080 --model model.gguf     # or any fLLM server
cd desktop/native && cargo run --release                    # FLLM_URL=http://host:port points the window elsewhere
```

## What is built (stage 1: the chat, stage 2: Models and Settings)

- Chat with streaming answers, Markdown (headings, lists, tables, quotes, code blocks with highlighting and a copy button, inline code),
  stop, copy, the reasoning of thinking models kept apart, tokens per second and prompt speed under every answer.
- Chats on disk (`~/.local/share/fllm-app/chats.json`, `$FLLM_APP_DATA` moves it), written after every change, a damaged file is moved aside
  instead of overwritten. Title from the first message, delete, open, new chat.
- The design system of the Studio (`design-system/fllm-studio/MASTER.md`): the same tokens in both themes (copper accent, system grays,
  hairlines), Inter shipped inside the program, a context meter with a track, 780-pixel reading column shared by messages and composer.
- No server: a clear screen with the address and a retry every four seconds. A server without a model has its own screen that leads to Models.
- Settings (sidebar > Settings). Everything is *Automatic* until you switch it to *Manual*; a manual value is sent with every message, an
  automatic one is left to the server and the model. Answers: sampling (temperature, top-p), the system prompt (the short built-in one or your own
  text), the longest answer; kept in the chat file, "Reset all answer settings" puts them back. The engine of the server (the Studio's own
  routes `/api/studio/engine` and `/api/studio/settings`): sharing the start of prompts, the state checkpoints of Qwen3.5 / Mamba / RWKV and their
  memory, prompt blocks in RAM or on disk, what happens when memory runs out, saved answers and who shares them, loading the last model at start,
  unloading when idle. A switch or choice is written at once, a number or text when you press Enter (checked first); the server says which
  options its environment fixes; engine options apply after a restart, which the page offers ("Restart now") and waits for.
- The model name in the top bar of the chat is a picker: the models on this computer with the loaded one marked, a click loads another
  (a file from a folder by its path, a downloaded one by its name), "Manage models…" opens the Models page. While the server switches its model
  (it restarts its engine for a load, an unload and a settings restart) the window says "Loading …" instead of "No server" and looks every second.
- Code (sidebar > Code): a coding agent in a folder. Open a folder (type the path, *Choose…* for the folder dialog of the server's computer, or one of the last six), tell the agent what to do; the model
  gets the server's agent tools (read, search, edit, run commands, ...) and answers with tool calls, which the window runs through
  `/api/studio/agent/tool` and feeds back until the model answers without one (at most 30 model calls per message). Every call is a card:
  an edit shows the lines it removes and adds on tinted bands before it runs, a command shows itself, the result follows. Reading and the
  agent's own notes run alone; the rest asks (*Allow* / *Deny*; a denial goes back to the model) unless you pick *Edit freely* or *Everything*,
  and switching the mode releases the calls that wait. *Stop* ends the stream and the running commands. The bar under the conversation counts
  the file changes and undoes the last one or all of them. Not yet: sub-agents, plan mode, MCP tools, hooks, sessions that survive a restart.
- Models (sidebar > Models): what is loaded (Unload), the downloads with progress, speed, time left and Pause / Resume / Cancel / Remove, the
  models on this computer (Load: a file from a folder by its path, a downloaded one by its name), and the catalog with a search field, the
  quantizations as chips (the recommended one is chosen; you pick another before you press Download), the size of the chosen one and whether
  it fits in the memory of this computer. Everything is the Studio's own routes (`/api/studio/models`, `.../load`, `.../unload`, `.../pull`,
  `.../progress`, `.../downloads`); a refused load shows the message of the server. The page polls once a second while it is open or a download runs.

## What is not built yet

| Stage | Content |
|---|---|
| 2c | Load options (context, device), the measured prompt speed with the compact prompt, projects and memory, web tools |
| 3b | Code mode: sessions that survive a restart, sub-agents and plan mode, MCP tools, a terminal view |
| 4 | The engine inside the process (no second program; for now the app starts the server next to it, see above), installers per system that put both side by side, the tray, autostart, updates; then the Tauri app and the web Studio files can go |
| later | Voice, artifacts, the research mode, the command palette |

Until stage 4 the Tauri app stays the supported desktop app.

## Tests

```sh
cargo test --locked      # 89 tests: the SSE parser and the client against a small server, the chat store, the contrast of the palette, the
                         # window driven headless (new chat, send, stream, stop, error, delete, theme) and drawn to PNG files
```

The window tests use iced's software renderer, so they need no display and no GPU. The pictures go to
`target/tmp/ui-snapshots/` (conversation, start screen and offline screen, light and dark). A mutation of the line buffer, of the save in
`finish` and of the guard against late events each turn a test red.

Not checked: a window on a real screen (this machine has none), scaling on a HiDPI display, input methods, the macOS and Windows builds
(the CI workflow `native-app` builds and tests all three).
