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
  stop (the button or Esc over the whole window: it ends a running answer and a research run, and does
  nothing when nothing runs), copy, the reasoning of thinking models kept apart and foldable (the line *Thought
  process (n characters)* of the Studio with a click; while the answer has not started it says *Thinking* and is
  open), tokens per second and prompt speed under every answer.
- Chats on disk (`~/.local/share/fllm-app/chats.json`, `$FLLM_APP_DATA` moves it), written after every change, a damaged file is moved aside
  instead of overwritten. Title from the first message; the sidebar opens, renames, duplicates, exports, deletes and searches them, groups
  them by the day they were last used, pins and archives them, and *Select* takes many at once.
- The design system of the Studio (`design-system/fllm-studio/MASTER.md`): the same tokens in both themes (copper accent, system grays,
  hairlines), Inter shipped inside the program, a context meter with a track, 780-pixel reading column shared by messages and composer.
- No server: a clear screen with the address and a retry every four seconds. A server without a model has its own screen that leads to Models.
- Settings (sidebar > Settings). Everything is *Automatic* until you switch it to *Manual*; a manual value is sent with every message, an
  automatic one is left to the server and the model. Answers: sampling (temperature, top-p), the advanced values (top-k, min-p, penalties, DRY,
  mirostat, seed, stop words), the output format (free text, a JSON object, a JSON schema, a GBNF grammar or a regular expression, with the box for
  the schema, grammar or pattern below when one needs it — a box that does not parse is ignored, like in the Studio), the token probabilities of an
  answer (every word of the answer comes back with the chance the model gave it and is drawn as a chip under the text, the warmer the color the more
  certain), the system prompt (the short built-in one or your own text), the longest answer; kept in the chat file, "Reset all answer settings" puts
  them back. The engine of the server (the Studio's own
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
- Response style and the tool texts of the prompt: the chip *Style* in the composer (with the
  lines icon; "Style: Concise" for a chat that has a style) opens the menu of the five styles of
  the Studio — Default, Concise, Explanatory, Formal, Casual — the current one tinted. The choice
  goes into the chat (saved with it), or without a chat into the staging that the next new chat
  takes. The instruction of the style rides at the very end of the system prompt, project and
  assistant included; the default adds nothing. The texts for the tools of the prompt (web,
  memory, chat search, the call line and the line for a chat without tools) are the SP_TOOL_TEXT
  of the Studio word for word — memory is live with the memory round, the web and chat-search
  texts wait for their rounds.
- Temporary chat: the ghost button in the top bar beside the token counter stages the next chat
  (the banner of the Studio sits over the messages: "Temporary chat: it is not saved and
  disappears when you close it or start another chat."), the chat that is made when the first
  question is sent carries it, and the button is tinted while it is open or staged. Pressing the
  button on a temporary chat with messages asks first with the Studio's dialog ("Close the
  temporary chat?" / "It was never saved and will be deleted now." / "Close it"), one without
  messages is gone at once, and leaving it — another chat or *New chat* — drops it for good
  without asking. The file never gets one: the save leaves it out the way the Studio's
  saveThreads does.
- Conversation tree: a *Retry* under an answer and an *Edit* under a question keep the old
  version — the chat holds its versions (`Variant`, the slot of the shown one) and under a
  message with more than one the arrows `‹ 1 / 2 ›` step through them: a click switches, the
  numbers say which one is shown, and they are away while an answer is written. What a rewind
  hides becomes one of those versions when the row moves on.
- Message actions: *Copy answer* carries the visible text of the answer without the Markdown
  marks, *MD* the text as the model wrote it, *Read aloud* strips it for speech (no code, no
  marks, no web addresses — the rules of `stripForSpeech` of the Studio, checked against that
  code) and hands it to the voice of the computer (`spd-say`, `espeak-ng`, `espeak`, `say`, the
  synthesizer of Windows; a computer without one gets "This computer cannot read aloud.").
  *Rewind* sits under a question and under an answer: the conversation ends at that message and
  nothing is deleted — the notice under the messages says "Rewound: N message(s) after this one
  are hidden." with *Undo* for nine seconds, at the last message it says "Nothing after this
  message to rewind.", and the Undo says "Rewind undone". A new question or a retry keeps what
  was hidden as a version of the chat, so the conversation tree can show it later.
- Use the web: the chip *Allow web access* in the composer switches the web tools of the server on
  for the window (one press; the line of the Studio says what that means, and the chip is tinted
  while it is on). The request then carries `web_search` and `web_fetch` beside the memory with
  the descriptions of the Studio, the prompt gets the web line (the compact one for a small
  model), a context under 2,048 tokens gets no tools at all and a temporary or private chat gets
  no memory. The calls run against `POST /api/studio/web/search` and `/api/studio/web/fetch`, one
  after the other, with the room for results the Studio budgets for one answer (five rounds of
  four calls); the sources are numbered `[n]` under the answer (*Sources*, a click opens the page
  in the browser). The texts, cards, budget and error lines are checked against `ctRunTool` of the
  Studio itself — the fixtures are produced by that code.
- Appearance: the settings card of the Studio — theme (system/dark/light), the eight accent
  colors with the chosen one ringed, the base size of the text (14–18px), *Tint your messages* and
  *Show generation speed*. The accent colours the whole page exactly as the Studio computes it
  (dark theme takes the accent, light theme its lighter twin, the text on it chosen by luminance),
  without the tint the messages fall back to the Studio's secondary. Three unit tests over every
  accent, three window tests with one drawing, four mutations.
- Session mailbox: notes one session leaves for another wait in the workspace and show up as a badge
  *Inbox (n)* on the top bar. The panel lists them with what they say about themselves; one click puts
  a note into the message (a blank line between what was written and what comes from another session),
  another marks it read. Four window tests with one drawing, three mutations.
- Usage and cost: the settings card counts the tokens of every answer per model and shows what
  they cost with the prices entered (dollars per million; the longest matching pattern wins, `*`
  matches any tail). The table carries every model with its answers, input, output and cost plus the
  row of everything together; *Save prices* writes the rows back to the server, *Reset counters*
  sets them to zero. Seven unit tests, three window tests with one drawing, four mutations.
- Start screen: the welcome view says what the engine is doing right now, like the Studio's hero: *No model
  loaded* with the plan of what can be done about it (every model on this computer counts — the one used last first,
  the recommended one when it is on disk, the largest that fits otherwise — or the recommended download when nothing
  is here) and its buttons to load, download, choose a file or browse; a loaded model shows its name, its facts and
  the two-minute line of tokens per second with the value of the last moment. Twelve fixtures of the Studio's own
  pure functions, seven unit tests, four window tests with three drawings, four mutations.
- Voice and alerts: Settings shows the cards *Dictation* (with what the Studio says when there is no speech
  recognition, and its seventeen languages), *Reading aloud* (voice, speed, pitch, read answers automatically,
  *Read a sample*) and *Alerts*. Every answer is read with the chosen voice through the speech programs of the
  computer (`voice_args` puts rate and pitch on each program's own scale), and a run that took twenty seconds
  (an agent five) announces itself while the window is behind — with the two-tone chime of the Studio when
  *Sound* is on and the system notification when *System notification* is on. Seven tests, one drawing, three
  mutations.
- The command palette (`Mod+K`): the commands of the app and the chats of the store in one list, reached
  by typing and the arrow keys — the score, the ranking in blocks per kind and the arrow steps are the
  `@pure` block of `palette.js` itself (37 fixture cases), the chats come with the text around a hit under
  their title the way the sidebar searches, Enter runs a row and closes, Esc does too. 4 window tests with
  3 drawings, 3 mutations.
- The keyboard shortcuts: the overview (`Mod+/`, or Settings, General) lists what every one of the
  fifteen shortcuts does with the keys it has now, the fixed rows below them, and Change / Reset per row.
  Change waits for the new keys; `checkCombo` of `boot.js` refuses a bare letter, `Shift` alone, a fixed key
  and a combination another shortcut has, with the Studio's own words; the changes are saved (`fllm.studio
  .shortcuts.v1`) and the window reads them (`comboFromEvent`: `Mod` is the command key on macOS and Ctrl
  elsewhere, a modifier alone is nothing, a plain letter is for the text field). 74 fixture cases produced
  by the `@pure` block of `boot.js` itself, five window tests, three mutations.
- The welcome tour: the first start shows it once (`TOUR_KEY` of the Studio: no key and no chats), four
  steps in the modal of `shell.html` — *Welcome to fLLM*, *Your computer* with the line of `describeComputer`
  (`models.js`, one entry per physical device through the merging of `npu.js`, the processor first), *Pick a
  model* with the loaded model, up to four suggestions of the hub by display name and *Open Models*, and
  *Three tips* with the keys as `formatCombo` writes them. Skip, Back, Next / *Get started*, a click beside it
  closes; Settings > General brings it back. Checked in the window with five drawings and three mutations.
- The math of an answer, typeset: `$x$`, `$$x$$`, `\(x\)` and `\[x\]` in the Markdown of the chat are found
  by the recognizers of `markdown.js` (with their lookbehind and lookahead as character checks, which a Rust regex
  cannot express) and given to the typesetter of the Studio (`tex.rs`, the `typeset` and `toHTML` of `markdown.js`
  as fixtures): groups, scripts, fractions, roots, symbols, Greek letters, accents, big operators with their power
  above and their subscript below in a display answer, `\text`, thin spaces of the units. A display piece gets its
  own line, an inline piece keeps its place between the words with the typeset source instead of the dollars, and a
  piece that will not typeset is shown as its source — checked in the window and drawn to PNG, with the fixtures
  produced by `markdown.js` itself (45 typeset cases, 7 span places, 15 inline patterns).
- Research mode: the chip *Research* in the composer (between web and style; one press and the
  line of the Studio says what it means) turns a question without attachments into a research run:
  the plan asks the model for 3–6 search queries (with the JSON schema, read plain when the server
  refuses it), the searches of a round run together and the best pages are read and given their
  passages by the same BM25 as a local document; per page the model takes notes (in parallel, with
  the room of the context bounded to 200–500 tokens) and the reflection says whether the notes are
  enough — if not, one more round with the queries it names, two rounds at most. The notes are
  budgeted for the context (the reserve of 400 tokens, at most 240 characters per note, condensed
  per page when they are too big) and the report is streamed as its own answer with what is left
  of the context after the margin (at least 200 tokens), ending with the sources under *##
  Sources* (`n. [title](url)`). While it runs, the answer shows the progress line with the round,
  step, phase and pages read, the notice of a fallback or *Research stopped.* over it, and under it
  the log with a click: queries, pages (date, notes), problems, steps with the seconds. A server
  without the web routes (404) or with them blocked (403), an unreadable plan, no pages and no
  notes all end the run before the model is asked again — it answers on its own with *Research
  could not run: … The model answers on its own instead.*; *Stop* cancels the run and its requests,
  and the run comes back with the chat. Checked against the routes themselves with a small server
  that answers plan, search, fetch, notes, reflection and report (4 tests, 5 mutations).
- Chats: *Select* beside the search takes several at once to delete, archive, move to a project
  or export; *Import* beside it takes a file of exports back (an array of chats or one, the shape
  the export writes — an id that this window already has gets a new one), with the lines of the
  Studio (*The file is not valid JSON.*, *No chats were found in the file.*, *2 chats imported*).
- Artifacts: a fenced block of an answer with a title (`title="…"`) or that can be previewed
  (HTML, SVG, Markdown, CSV) becomes an artifact; *Preview* or *Open* in the header of the block
  opens the panel beside the chat with the artifact and all its versions (`‹ v N / M ›`) — the
  blocks of a conversation group into families exactly the way the Studio's `buildFamilies` does,
  checked against fixtures the JavaScript itself produced. The panel has the *Preview* and *Code*
  tabs (Markdown renders, a CSV becomes its table of at most 500 rows with the note of the rest,
  code and pages show their text), *Reload*, *Copy*, *Download* (your downloads, the name of the
  Studio) and *Larger* — the large view sits over a dimmed chat. A page has no window to be drawn
  in here, so *Open in browser* writes the file and shows it in the browser of your computer. A
  JavaScript block has a *Run* button: it runs on the quickjs engine with the script of the
  Studio's `runner.js` — the same console lines with the same formatting and limits, no network,
  the async fallback, errors as `name: message`, the value of the last line, five seconds at most
  — and its output appears under the block (*Running…* then *Done in N ms*, the lines and `->`
  value; the same button becomes *Stop*).
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
| later | Voice, artifacts, the command palette |

Until stage 4 the Tauri app stays the supported desktop app.

## Tests

```sh
cargo test --locked      # 213 tests: the SSE parser and the client against a small server, the chat store, the contrast of the palette, the
                         # window driven headless (new chat, send, stream, stop, error, delete, theme, assistants, projects,
                         # attachments, personalization, prompt library, response style, temporary chat, message actions, web tools) and
  drawn to PNG files
```

The window tests use iced's software renderer, so they need no display and no GPU. The pictures go to
`target/tmp/ui-snapshots/` (conversation, start screen and offline screen, light and dark, the assistant menu and editor, the projects
with their page, dialogs and passage chip, the attachment chips and the bubble with an image). A mutation of the line buffer, of the save
in `finish`, of the guard against late events, of the project filter of the sidebar, of the passage prefix, of the project header line, of
the files in the request, of the image parts, of the documents of the chat in the search, of the personalization in the prompt, of the
capture of "remember that …", of the saving of the memory tool, of the sort of the library, of
the menu on every change of the box, of the limit of a hundred prompts, of the style in the
prompt, of the staged style for a new chat, of the web text of the Studio, of the save without
its filter, of the staging into the chat, of the drop when a temporary chat is left, of the
rewind that hides, of the version it keeps for a new question, of the plain copy, of the web
tools in the request, of their line in the prompt and of the sources under an answer each turn a
test red.

Not checked: a window on a real screen (this machine has none), scaling on a HiDPI display, input methods, the macOS and Windows builds
(the CI workflow `native-app` builds and tests all three).
