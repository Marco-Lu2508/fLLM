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
  instead of overwritten. Title from the first message; the sidebar opens, renames, duplicates, exports, deletes and searches them, groups
  them by the day they were last used, pins and archives them, and *Select* takes many at once.
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
- The assistant beside the model name in the top bar (the Studio's presets as assistants): six built-ins — Standard, Precise and brief,
  Creative, Programmer, Translator, Editor — with their icon, colour and four starter suggestions, your own ones below them, and
  *New* / *Import* / *Save current*. Choosing one stores it on the open chat and for new chats, sends its sampling values (temperature,
  top-p, top-k; the sampler switches to manual, an assistant brings its own), loads its model when it is on this computer (a missing one
  keeps the current model and says so) and shows its suggestions on the empty chat. The editor window has the name, eight icons, twelve
  colours, the instructions, the three sampling values, the model as a select of this computer's models, up to four starters, and how the
  assistant treats the default system prompt (*Use the global setting*, *Add to the default prompt*, *Replace the default prompt*, with
  `{{date}}` / `{{time}}` / `{{language}}` filled in). *Export* writes `<name>.assistant.json` to the Downloads folder, *Import* reads one
  back through `POST /api/studio/pick-file` with `kind: "file"`, which opens the dialog of this machine and answers the text of the file it
  chose. A built-in is edited as a copy; deleting clears the choice; chats from before the feature load unchanged.
- Projects: a *Projects* section in the sidebar between the navigation and the chats — a row per project with its folder, its count of
  chats, rename and delete, its chats below it when the chevron is open, and *New project* as an input row at the top. The page of a
  project (the row or the badge in the top bar) has the name, how many of the 2,000,000 characters of knowledge are used, the
  *Instructions for every chat of this project* box, the *Knowledge* with *Add files* (the dialog of this machine; text files are read
  here, PDF, DOCX, XLSX, EPUB and HTML through `POST /api/studio/extract`, images are refused with the line of the Studio) and every
  document with its size, pages and a cross, the chats of the project with their dates, *New chat in project*, *Rename* and
  *Delete project* — the last one asks first, with the sentence of the Studio's dialog. Chats move with *Move* in the chat menu or in
  the select row: the picker lists *No project* and every project. A new chat joins the project of the page it was started from, the
  badge in the top bar opens the page, the sidebar only lists chats without a project (a search finds them all, archived ones stay
  under *Archived*). Every question in a chat of a project searches its knowledge (BM25, `src/rag.rs`); the passages travel with the
  newest question and the window shows them under it, and the instructions sit in the system prompt between the default prompt and the
  assistant's own text with the Studio's header line. Limits, clamps and messages follow `PRJ_LIMITS` and `projects.js`.
- Attachments: the + beside the input (and files dropped onto the window) attaches an image or a file to the next question. An image is
  shrunk the way the Studio shrinks one (at most 1280 pixels on the long side, JPEG) and travels as an image part with the question — an
  image without a question says *Describe the image.* A text file goes in whole when it fits 35% of the context of the model (the
  Studio's line *too long for the context* when it does not, and then it becomes a document of this chat), PDF, DOCX, XLSX, EPUB and HTML
  are read by the server (`POST /api/studio/extract`, the 50 MB line of the Studio), a text file over 20 MB says so. The chips over the
  input show the documents of this chat — a cross takes one out — and what is attached to the next question; the bubble of the question
  shows the thumbnails and the file chips. Every question searches the documents of the chat together with the knowledge of its project
  (the passages travel with the question as they do for a project), the context meter counts an image with 256 tokens, and a question of
  nothing but an attachment is sent and names the chat after its first attachment. Chats from before the feature load unchanged. Not
  ported: pasting an image from the clipboard (the window gets no such event) and several files in one dialog (the dialog of the server
  picks one file at a time).
- Personalization and memory (Settings > Personalization): *Custom instructions* in a box with its counter and the switch *Use these
  instructions*, and the remembered facts with *Use memory*, a row per entry (a click changes it, the cross deletes it), a row to add one,
  the line *N of 100 entries …* and *Forget everything* behind its dialog. What of yours goes into the system prompt between the default
  prompt and the project with the Studio's headers (*About the user and how they want answers …*, *Saved memory about the user …* with
  the entries that fit — the useful ones first, newest first); a chat switches either off in the chat menu (*Personalization: on/off*,
  *Memory: on/off* with the lines of the Studio). A question that starts with "remember that …" saves its thing, the *Remember* button
  under a message asks first with the sentence of the Studio, secrets are refused with its line and the list holds a hundred entries. The
  model gets the `memory` tool when the switch is on (not for a chat that says no, not for a context under 2048 tokens and not for a model
  that reads the prompt slower than 120 tokens a second): its calls run in the window, the answer carries them and their results back, a
  line under the answer says what changed (*Saved to memory: …*) with *Undo*, and an unknown tool or bad arguments get the Studio's error
  lines. The copy on the server (`GET`/`PUT /api/studio/memory`) is taken over on a first run and written when it changes. Not ported: the
  retry of a request a server refuses when it is asked with tools.
- Prompt library (Settings > Personalization, and "/" in the box): a row per saved prompt — the
  title in its box, the text under it, the cross in the corner of the row —, "No saved prompts
  yet." when there is none, "Add a prompt" with the line "The library holds at most 100 prompts."
  at the hundredth, and the limits of the Studio (60 characters for a title, 8,000 for the text).
  While the message box holds one line that starts with a "/" it lists the saved prompts that
  match what is typed (the built-in commands `clear`, `new`, `system`, `export`, `help` and
  `retry` are left alone): the arrows move the highlight, Enter or Tab puts the whole text of the
  picked prompt into the box with the cursor behind it, Escape closes it and a click on an entry
  picks it; eight at most, the titles that start with what is typed first, the preview under a
  title the text with the whitespace collapsed and cut at 80 characters. The matching is checked
  against `plibMatch` of the Studio itself (the fixtures are produced by that code). Not ported:
  the built-in commands as commands of their own.
- Code (sidebar > Code): a coding agent in a folder. Open a folder (type the path, *Choose…* for the folder dialog of the server's computer, or one of the last six), tell the agent what to do; the model
  gets the server's agent tools (read, search, edit, run commands, ...) and answers with tool calls, which the window runs through
  `/api/studio/agent/tool` and feeds back until the model answers without one (at most 30 model calls per message). Every call is a card:
  an edit shows the lines it removes and adds on tinted bands before it runs, a command shows itself, the result follows. Reading and the
  agent's own notes run alone; the rest asks (*Allow* / *Deny*; a denial goes back to the model) unless you pick *Edit freely* or *Everything*,
  and switching the mode releases the calls that wait. *Stop* ends the stream and the running commands. The bar under the conversation counts
  the file changes and undoes the last one or all of them. *Plan* is the plan agent: it is offered only the tools that read, refuses anything else, and answers with a plan; *Carry out the plan* then goes back to the mode you had and tells the agent to do it. The instructions of the folder (`AGENTS.md`, `CLAUDE.md` and the like, nearest first), the notes the agent kept and your profile are read from the server and are part of the model's instructions (the empty screen names the files it read). Besides the server's tools the model has `todo_write` / `todo_read` (a task list that the window keeps and shows above the box) and `ask_user` (a question with answer buttons; you can also type the answer in the box), and the tools of the connected MCP servers (`/api/studio/mcp/tools`, named `mcp__server__tool`; they ask first like a command). The screen and input tools are not offered. Not yet: sub-agents (`task`), hooks. The conversation of each folder is kept (`code-sessions.json` next to the chats, the newest 200 items; a call that had not finished comes back as interrupted) and returns when the folder is opened again; *New session* forgets it.
- Long chats: the conversation draws the newest 60 messages; *Show 60 earlier messages* adds the next older ones. Laying out every message on every
  frame of a stream cost 63 ms for 800 messages (release build, `cargo run --release --example view_bench -- 50 200 800`); with the window the
  build and layout of the view stay at about 6 ms however long the chat is (opening an 800-message chat still parses its markdown once, 124 ms).
- Models (sidebar > Models): what is loaded (Unload), the downloads with progress, speed, time left and Pause / Resume / Cancel / Remove, the
  models on this computer (Load: a file from a folder by its path, a downloaded one by its name), and the catalog with a search field, the
  quantizations as chips (the recommended one is chosen; you pick another before you press Download), the size of the chosen one and whether
  it fits in the memory of this computer. Everything is the Studio's own routes (`/api/studio/models`, `.../load`, `.../unload`, `.../pull`,
  `.../progress`, `.../downloads`); a refused load shows the message of the server. The page polls once a second while it is open or a download runs.

## What is not built yet

| Stage | Content |
|---|---|
| 2c | Load options (context, device), the measured prompt speed with the compact prompt, memory, web tools |
| 3b | Code mode: sub-agents, a terminal view |
| 4a | One download per system with the app and the engine side by side: the `native-bundle` workflow (started by hand with a release tag; it takes the engine of that release, checks its checksum, adds the app and, with `upload`, signs and attaches the archives) |
| 4 | The engine inside the process (no second program; for now the app starts the server next to it, see above), installers per system that put both side by side, the tray, autostart, updates; then the Tauri app and the web Studio files can go |
| later | Voice, artifacts, the research mode, the command palette |

Until stage 4 the Tauri app stays the supported desktop app.

## Tests

```sh
cargo test --locked      # 190 tests: the SSE parser and the client against a small server, the chat store, the contrast of the palette, the
                         # window driven headless (new chat, send, stream, stop, error, delete, theme, assistants, projects,
                         # attachments, personalization, prompt library) and drawn to PNG files
```

The window tests use iced's software renderer, so they need no display and no GPU. The pictures go to
`target/tmp/ui-snapshots/` (conversation, start screen and offline screen, light and dark, the assistant menu and editor, the projects
with their page, dialogs and passage chip, the attachment chips and the bubble with an image). A mutation of the line buffer, of the save
in `finish`, of the guard against late events, of the project filter of the sidebar, of the passage prefix, of the project header line, of
the files in the request, of the image parts, of the documents of the chat in the search, of the personalization in the prompt, of the
capture of "remember that …", of the saving of the memory tool, of the sort of the library, of
the menu on every change of the box and of the limit of a hundred prompts each turn a test red.

Not checked: a window on a real screen (this machine has none), scaling on a HiDPI display, input methods, the macOS and Windows builds
(the CI workflow `native-app` builds and tests all three).
