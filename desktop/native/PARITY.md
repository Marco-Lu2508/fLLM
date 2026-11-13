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
| Stop | done (Esc: the key ends a running answer and a research run that is still planning — the title of the Studio's send button says *Stop the answer (Esc)*; when nothing runs the key does nothing; it works over the whole window, not only with the composer focused) |
| Chat list, rename, delete, search over chats | done (date groups, pinned, archived; search in titles and messages) |
| Conversation tree: edit a message / regenerate keeps the old version, arrows between versions | done (a *Retry* under an answer and an *Edit* under a question keep the old version: the chat holds its versions (`Variant`, the slot of the shown one) and under a message with more than one the arrows `‹ 1 / 2 ›` step through them — a click switches, the numbers say which one is shown, and they are away while an answer is written; what a rewind hides becomes one of those versions when the row moves on) |
| Thinking chip (off/low/medium/high, per model), collapsible reasoning | part (chip done, one level for all chats; reasoning shown and foldable as the box of the Studio: the line *Thought process (n characters)* with a click, open while the answer has not started where it says *Thinking*; the level per chat: todo) |
| Default system prompt builder (date, language, model family, tools), response style | done (built from date, language, model family and size, prompt length auto/full/short, your text replaces or is added, clock held 30 min; the tool texts of the prompt are SP_TOOL_TEXT of the Studio word for word — the memory and web ones are live with their rounds, the chat-search one waits for its round, and the line for a chat without tools is the Studio's; the response style: the chip *Style* (or *Style: Concise*) in the composer with the menu of the five styles of the Studio, the choice goes into the chat or, without one, into the staging that the next new chat takes, and the instruction rides at the very end of the prompt) |
| Assistants (name, icon, color, instructions, sampling, model, suggestions; import/export) | done (six built-ins plus your own ones; the menu under the name in the top bar; editor with icon, color, sampling, model, starters and the per-assistant default prompt; export and import as JSON) |
| Projects (instructions + knowledge base, move chats) | done (a *Projects* section in the sidebar with a row per project, its chats under it and rename/delete; the page with the name, the used characters, the instructions box, the knowledge with *Add files* and its documents, the chats of the project with their dates, *New chat in project* and *Delete project* behind the Studio's confirm dialog; chats move through the picker of the chat menu and of the select row, a new chat joins the open project; the instructions enter the system prompt between the default prompt and the assistant's own text with the Studio's header line, and the best passages of the knowledge travel with the newest question, shown under it; limits and messages follow PRJ_LIMITS and projects.js) |
| Presets / prompt library (`/` in the box) | done (Settings > Personalization: a row per saved prompt — the title in a box, the text under it, the cross in the corner of the row —, "No saved prompts yet." when there are none, "Add a prompt" with the line "The library holds at most 100 prompts." at the hundredth, and the limits of the Studio (60 characters for a title, 8,000 for the text). While the message box holds one line that starts with a "/" it lists the saved prompts that match what is typed (the built-in commands `clear`, `new`, `system`, `export`, `help` and `retry` are left alone): the arrows move the highlight, Enter or Tab puts the whole text of the picked prompt into the box with the cursor behind it, Escape closes, a click picks it; eight at most, the titles that start with what is typed first, the preview collapsed and cut at 80 characters — match and sort checked against `plibMatch` of the Studio itself (the fixtures are produced by that code). Not ported: the built-in commands as commands of their own (the row "Slash commands, skills, `/init`" below).) |
| Personalization and memory | done (Settings > Personalization: the *About me* box with its counter and the *Use these instructions* switch, the remembered facts with *Use memory*, a row per entry that changes with a click, the row to add one, the count line of the Studio and *Forget everything* behind its dialog; the chat menu says *Personalization: on/off* and *Memory: on/off* with the lines of the Studio; "remember that …" at the start of a question and the *Remember* button under a message save with its lines and refuse secrets with its line; about me and the memory enter the system prompt between the default prompt and the project with the Studio's headers and a chat can switch either off; the model gets the `memory` tool with the card under its answer and Undo — a round of tool calls is asked and answered the way the Studio's ctGenerate does it, including the lines for an unknown tool and bad arguments; the copy on the server is taken over on a first run and written when it changes (`GET`/`PUT /api/studio/memory`). Not ported: the retry of a request a server refuses when it is asked with tools (the grammar of the Studio)) |
| Sampling panel (temperature, top-p/k, min-p, penalties, mirostat, ...) | done (the output format with its picker of the five shapes of the Studio — free text, JSON object, JSON schema, GBNF grammar, regular expression — the box under it for the schema, grammar or pattern when one is needed, and it rides the request exactly as `requestBody` builds it: `response_format` for the two JSON shapes (an unparseable box is ignored), `grammar`+`grammar_type` for GBNF, `regex` for the pattern; the switch *Show token probabilities* sends `logprobs` with `top_logprobs: 5`, the words of the answer come back from `choice.logprobs.content` and are drawn under the text as the chips of `logprobsHtml` — the background `hsl(hue, 70%, 42%, 0.22)` with `hue = round(p · 130)` from `exp(logprob)`, so a certain word is green and an uncertain one red; kept with the message) |
| Attach images (vision), text files and documents to a chat (PDF, DOCX, ... via `/api/studio/extract`) with the knowledge of the chat itself | done (the + of the composer and a drop onto the window: images are shrunk to 1280 px JPEG like the Studio and sent as content parts with the question — an image without a question says *Describe the image.*; text files ride whole when they fit 35% of the context and otherwise become documents of the chat with the Studio's line; PDF, DOCX, XLSX, EPUB and HTML go through `POST /api/studio/extract` behind the 50 MB line, a bigger text file behind the 20 MB line, an image is refused with the Studio's line; the chips over the input show the documents of this chat (a cross takes them out) and the pending attachments, the bubble shows the thumbnails and the file chips; the documents of the chat are searched together with the knowledge of the project for every question, the passages travel with the question, the meter counts an image with 256 tokens; chats from before the feature load unchanged. Not ported: pasting an image from the clipboard (the window gets no such event) and several files in one dialog (the dialog of the server picks one file at a time)) |
| Use the web / Allow web access (tools) / Research mode | part (the chip *Allow web access* in the composer: one press switches the web tools of the server on for the window (the Studio's `state.useWeb`, not kept between starts) with the line of the Studio, tinted while it is on; the request then carries `web_search` and `web_fetch` beside the memory with the descriptions of the Studio, the prompt gets the web line (the compact one for a small model), a tiny context gets no tools at all, and a temporary or private chat gets no memory; the calls run against `POST /api/studio/web/search` and `/api/studio/web/fetch` in order, with the room for results the Studio budgets per answer (five rounds of four calls), the sources numbered `[n]` under the answer (`Sources`, a click opens the page) and the texts, cards, budget and error lines checked against `ctRunTool` of the Studio itself — the fixtures are produced by that code; research mode: the chip *Research* in the composer between web and style (the Studio's `btnToggleResearch`, one press and the line of the Studio says what it means) turns a question without attachments into a research run (`researchRun`): the plan of 3–6 queries is asked with the JSON schema (and read plain when the server refuses the schema: `parsePlan` takes JSON, an embedded JSON, then lines), the web searches of a run go out together (the pool of the Studio, the first round with six queries, every further round the up to three the reflection asks, two rounds at most), the best pages are read through `fetch`, their passages taken with the same BM25 (`rag::search`, `k` from the room of the context), notes are taken per page in parallel (`researchNoteSystem`, room bounded to 200–500), and the reflection of the notes decides whether they are enough — not: one more round with the new queries. The notes are budgeted with `researchBudget`/`fitNotes` (the reserve of 400 tokens, at most 240 characters per note) and, when they are too big, condensed per page in the order of the longest ones. The report is then streamed as its own answer with the size of `researchStream` (`researchMaxTokens` cut to what is left after the margin, at least 200) and ends with the sources of the Studio (`researchSourcesMarkdown`: `n. [title](url)` under *## Sources*; a page without a number is no source). While it runs the answer shows the progress line (*Research · round … · step … of 5* with the phase and the pages read), the notice of a fallback or *Research stopped.* over it, and under it the log of the Studio (`researchLogSummary`, opening with a click: Queries, Pages with the date and the notes, Problems, Steps with the seconds; at most eighty events, the oldest leave). A server whose web route is missing (404) or blocked (403) ends the run before anything is read and the model answers on its own with the notice *Research could not run: … The model answers on its own instead.* — which is what an unreadable plan, no pages and no notes at all also fall back to; *Stop* stops the run and its request with it, and the run is kept with the chat. Checked against the routes themselves: a small server that answers plan, search, fetch, notes, reflection and report, four tests for the run and the window, five mutations each turn one red) |
| Message actions (copy, copy Markdown, read aloud, rewind) | done (copy: the visible text of the answer without the Markdown marks; copy MD: the text as the model wrote it; read aloud: the text stripped for speech — no code, no marks, no web addresses — and handed to the voice of the computer (`spd-say`, `espeak-ng`, `espeak`, `say`, the synthesizer of Windows), a computer without one gets "This computer cannot read aloud."; rewind under a question and under an answer: the conversation ends at that message, nothing is deleted — the notice under the messages says "Rewound: N message(s) after this one are hidden." with *Undo* for nine seconds (the Studio's rewindNotice), "Nothing after this message to rewind." at the end and "Rewind undone" on the Undo; a new question or a retry keeps what was hidden as a version of the chat, and what a plain copy and the speech strip do is checked against the Studio's own code) |
| Temporary chat | done (the ghost button in the top bar beside the token counter: one press stages the next chat, the chat that is made when the first question is sent carries it, and the banner of the Studio sits over the messages; the button is tinted while a temporary chat is open or one is staged; pressing it on a temporary chat with messages asks first with the Studio's dialog — "Close the temporary chat?" / "It was never saved and will be deleted now." / "Close it" —, one without messages is gone at once, and leaving it (another chat, *New chat*) drops it for good without asking; the file never gets one: the save leaves it out the way the Studio's saveThreads does) |
| Artifacts (HTML/SVG/Markdown/CSV preview), run JavaScript | done (a fenced block of an answer becomes an artifact when it can be previewed or carries a title — `title=`/`artifact=`/`name=` in its info string, exactly the `extractArtifacts` of the Studio, checked against fixtures its own `@pure` block produced; the blocks of a conversation group into families with versions exactly like `buildFamilies`: a titled block joins the artifact of its title and kind, an untitled one the identical code or the most recently changed of its kind, one message changes one at most; the block header offers *Preview*/*Open* like `MD.hooks.actions` and *Run* for JavaScript; the panel sits beside the chat (46%, at most 580px) with the title and kind, `‹ v N / M ›` when it has versions, *Close*, the *Preview*/*Code* tabs (code has no preview) and *Reload* (a page), *Copy*, *Download* to the downloads and *Larger* — the large view is the fixed box over a dimmed chat, *Esc* only from that one; the body renders Markdown, the table of a CSV (500 rows, then *Showing the first 500 of N rows*, the parse of `parseCsv`) and the code; a page has no window to be drawn in, so its preview shows the source with *Open in browser*, which writes the file and opens it like the links of an answer; *Download* writes the file of `fileNameFor` with the mime of the kind. Run: `run_javascript` executes the block on the quickjs engine with the script of the Studio's `runner.js` — same `fmt`, 500 lines and 20,000 characters then *... output truncated*, the network functions throw *… is disabled: code blocks run offline*, the sync code first and the async code as the fallback, errors as `name: message`, the value of the last line as `-> value`, and the five seconds of the Studio that stop with *Stopped after 5 s (time limit)*; the lines and the status (*Running…*, *Done in N ms*, *Failed*) are drawn under the block, the same button says *Stop* while it goes. Ten tests for the pure part against the Studio's fixtures and for the runner itself, two for the window (panel with versions, run with its output), six mutations each turn one red) |
| Math (LaTeX) and tables | part (tables via markdown; math: todo) |
| Organizing chats (bulk delete, move, archive, export) | done (select many, delete, archive, move to a project, duplicate, export Markdown/JSON to Downloads; the import beside the select mode takes a file of exports back — an array of chats or one, in the shape `json_of` writes, an id of the file that is already taken gets a new one like `importThreads` does — with the lines of the Studio: *The file is not valid JSON.*, *No chats were found in the file.*, *1 chat imported* / *n chats imported*) |

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
