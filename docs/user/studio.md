# Studio: the browser app

The Studio is the app that ships inside `fllm-rs`: a chat, a coding agent, a model manager and the server settings in one page. It needs
nothing but the binary and a browser; the page and all its scripts are embedded and nothing is loaded from the internet.

```sh
fllm-rs studio --model owner/repo:Q4_K_M     # first start: pick a model once
fllm-rs studio                                # later starts reuse the last model and open the page
                                              # (with no model yet, the page opens and helps you pick one)
fllm-rs serve --model model.gguf              # the same page is served at / by every fllm-rs server
```

`studio` is `serve` plus two conveniences: it remembers the model it started with (and the one you switch to in the page) and opens the
page in your default browser (`--no-open` keeps it closed). All other `serve` options work unchanged.

## Chat

* Every chat is a tree. Editing one of your messages or regenerating an answer keeps the old version; switch between versions with the
  arrows under the message.
* Markdown with tables, lists and highlighted code, collapsible reasoning blocks, a context meter, and live tokens-per-second.
* Attach images (for vision models), documents and text or code files by button, drag and drop or paste. A long paste becomes a file.
* **Documents.** Attach PDF (with a text layer), DOCX, XLSX, EPUB, HTML, CSV, Markdown, code and other text files by button, drag and
  drop or paste. The server reads them (`POST /api/studio/extract`, up to 50 MB; a scanned PDF without a text layer, an encrypted PDF
  and the old binary Office formats are refused with a message). A document that fits the model's context (about a third of it) goes
  into the message whole. A longer one is not stuffed in: it is cut into passages of about 500 tokens, the passages that match your
  question best (BM25 ranking in the browser, no embedding model) are added to the prompt with labels such as `[report.pdf, chunk 3/42]`,
  and a chip under your message ("Using 6 passages from report.pdf") shows them. Several documents can be in one chat; they stay in
  the chat and are searched for every later question until you remove them (the x on their chip). The extracted text is stored in the
  browser with the chat, at most 600,000 characters per document and 1,500,000 per chat; the page says so when a limit cuts a document.
* **Thinking.** The *Thinking* chip in the composer appears only when the loaded model can think, and offers what the model supports:
  a model with a thinking switch (Qwen3, Qwen3.5, GLM, Gemma 4, ...) steps through Off, Low, Medium and High on each click (Low and
  Medium make the server end the thinking after about 512 or 2048 tokens and let the model answer; High lets it think as long as it
  wants); a model that takes an effort (gpt-oss) offers Low, Medium and High; a model that always thinks (DeepSeek-R1 style) shows the
  chip greyed out; models without a thinking mode (Llama, Mistral, Gemma 3, ...) do not show it. The level is remembered per chat, and a
  new chat starts at the model's own default. With a model of another provider the chip sends the level as the reasoning effort.
* **Use the web.** Switch on *Use the web* in the composer and each question is searched on the web first. The first three results
  are read, their best passages are added to the prompt with their source numbers, and the sources are listed under the answer. This
  uses the web tools of the server (`/api/studio/web/search` and `/api/studio/web/fetch`); if they are missing or blocked, the page
  says so and answers without the web. The pages are fetched by the server, so the computer running fLLM needs internet access.
* **Allow web access.** Switch on *Allow web access* in the composer and the model is offered the tools `web_search` and `web_fetch`. It decides
  itself: current or unknown facts (news, prices, versions, a link you give) lead to a search; explanations, writing, code and small
  talk are answered directly without one. Each tool call shows as a small card ("Searched: ...", "Read: ..."), the results go back to the
  model, and the sources it cites as [n] are listed under the answer. At most 5 rounds of tool calls per answer, then one more answer
  without tools; Stop works at any point. Results are cut to a share of the context size, so small contexts get short results. A model
  that never calls tools simply answers; a server or chat template that refuses tools gets a plain request (a toast says so). The tools
  use the web routes of the server (`/api/studio/web/search` and `/api/studio/web/fetch`), which refuse private addresses; if they
  are missing or blocked the model is told and the card shows the error. The pages are fetched by the server, so the computer running
  fLLM needs internet access. Tool calls are constrained with `tool_call_grammar` unless *Thinking* is on (the grammar would switch
  thinking off).
* **Research.** Switch on *Research* in the composer for questions that need more than one page. The model plans 3 to 6 search queries,
  the pages found are read in parallel (at most 4 requests at a time; duplicate and junk results such as social media, PDFs and
  login pages are skipped), the best passages of each page are picked with the same BM25 ranking as for documents, and the model
  writes short notes per page. It then checks whether the notes answer the question and, if not, runs one more round of searches
  (at most 2 rounds; a line above the answer shows round, step and progress). The answer is a report with *Summary*, *Findings* and
  *Open questions*, citing sources as [1], [2], followed by a source list with title, link and date when the page shows one. Under
  the answer, the collapsible *Research log* lists the queries, every page with its notes, failures and the time spent. The notes
  never take more than about 60 % of the model's context: longer notes are condensed, then cut. The Stop button cancels every
  pending request. Research needs the web tools of the server; when they are missing or blocked, or nothing usable is found, the
  page says why and the model answers on its own. Small models make weaker plans and notes, so check important claims in the
  linked pages. Research is not used for messages with images; edited or regenerated messages get a normal answer.
* Presets store a system prompt together with sampling values. The sampling panel covers temperature, top-p/top-k/min-p, penalties,
* **Assistants.** The pill next to the model name picks an assistant: a name, icon and color, instructions (system prompt), default
  sampling values, an optional default model and up to four suggestions for an empty chat (editable; the chips on a new chat come from
  the assistant). A new chat starts with the chosen assistant. Assistants can be exported and imported as JSON. The older presets are
  migrated once into assistants with the same ids; the built-in ones are read-only (edit makes a copy).
* **Projects.** The *Projects* section of the sidebar groups chats. A project has instructions that are added to the system prompt of
  all its chats and a knowledge base: add PDF, Office, EPUB, HTML, text and code files to the project page (the server extracts the
  text as for chat documents). Every question in a chat of the project searches those documents together with the chat's own ones
  (BM25 in the browser), and the passages are shown as chips under your message. Only the extracted text is stored: at most 600,000
  characters per document, 2,000,000 per project and 3,500,000 for all projects; the page says so when a limit applies. Create, rename
  and delete projects in the sidebar, drag chats onto a project (or use *Move to a project* in the chat menu), and open the project
  page for its chats, instructions and files with sizes. Deleting a project keeps its chats.
* **System prompt.** Every chat starts with a default system prompt made by the page: the date and time (kept for 30 minutes so the
  server can reuse its cache), your language, concise answers by default, formatting rules, honesty about uncertainty, no invented
  sources, and how to use the tools that are on. It never mentions a tool that is not offered. It comes in a compact size (models under
  about 3.5 billion parameters or contexts under 12,288 tokens) and a full size, with a short adjustment for the model family (Qwen,
  Llama, Mistral, Gemma, gpt-oss, DeepSeek, Phi, SmolLM) that is read from the model name and chat template. It says that the model
  runs locally on your computer, except for models of an external provider. After it come, in this order: your personalization and
  memory, the instructions of the project, the instructions of the assistant. *Settings > Personalization > System prompt* shows the
  prompt as it is sent now and lets you add your own text or replace the default prompt (`{{date}}`, `{{time}}` and `{{language}}` are
  filled in); an assistant can do the same for itself in its editor. Titles of new chats use a separate, short prompt that never
  reveals the system prompt and works with thinking models.
* **Personalization and memory** (Settings > Personalization). *About me and how to answer* is put in front of the instructions of every
  chat (2,000 characters; switch it off for one chat in the chat menu). Memory holds short facts about you. The model manages it itself
  with a `memory` tool (add, update, remove, list): when it learns something durable (name, preferences, projects, constraints) a card
  "Saved to memory: ..." appears in the chat with an **Undo** button. It is told never to save trivia or secrets, and the page refuses
  texts that look like passwords, keys or card numbers, also when you type them. You can still start a message with "remember that ..."
  or press the bookmark under a message. Edit, delete or *Forget everything* in Settings. At most 3,000 characters (1,200 for compact
  prompts) go into a prompt, entries that were come back to first, then the newest. *Memory* can be switched off globally (Settings) and
  per chat (chat menu, which also stops the model from changing it). The memory lives in the browser and as a copy in `memory.json`
  next to `studio.json` (`GET`/`PUT /api/studio/memory`, this machine only); a browser with no stored personalization restores it from
  that copy. The *search_chats* tool lets the model look through the titles and text of your earlier chats when you refer to one; the
  page asks once per chat whether to allow it, and temporary chats are never searched.
* **Prompt library.** Save prompts with a title in Settings > Personalization, then type `/` in the message box to list them, filter by
  typing, and insert one with Enter or Tab.
* **Organizing chats.** The select button above the chat list allows bulk delete, move, archive and export. Drag a chat onto the
  "pin" target to pin it. Archived chats leave the main list (*Show archived* brings them back) and are still found by the search.
* The sampling panel covers temperature, top-p/top-k/min-p, penalties,
  mirostat, DRY, a seed and constrained output (JSON object, JSON schema, GBNF grammar, regular expression).
* Chats can be pinned, renamed, duplicated, exported (Markdown or JSON), imported and searched by title and content. New chats get a
  short title from the model (switch off in Settings).
* Chats are stored in the browser (`localStorage`) and never leave it.
* **Artifacts.** A code block of `html`, `svg`, `markdown` or `csv` has a *Preview* button that opens a side panel next to the chat
  with the tabs *Preview* and *Code*, and the actions Reload, Copy, Download (as a file) and Larger (a large view). HTML and SVG run
  in a sandboxed frame (`sandbox="allow-scripts"`, no same-origin access, and a policy that forbids every network request), so a
  page can use scripts and styles but cannot reach the Studio, the server or the internet. Markdown and CSV are rendered as a document
  and a table (first 500 rows). An artifact is versioned: when a later answer contains a block of the same kind, it becomes the next
  version of that artifact (version switcher in the panel header; the panel jumps to the new version when the answer ends). Give a block
  a title in its info string, for example ```` ```html title="Todo app" ````, to keep several artifacts apart: blocks with the same title
  are versions of each other, and a titled block of any language (also `python`) gets an *Open* button.
* **Run JavaScript.** A `javascript` block has a *Run* button. The code runs in a Web Worker with no page access and no network
  (`fetch`, `XMLHttpRequest`, `WebSocket` and `importScripts` throw). The output of `console`, the final value and errors appear under the
  block. It stops after 5 seconds or with the *Stop* button; top-level `await` and `return` work. The worker is served by the Studio
  itself at `/studio/runner.js` with its own policy (it may evaluate code but can connect to nothing).
* **Message actions.** Under an answer: copy (as visible text), copy as Markdown, read aloud (shown when a speech add-on provides
  `window.fllmSpeak`), thumbs up or down with an optional note (stored with the chat and part of its JSON and Markdown export),
  regenerate, delete. When an answer stopped at the token limit, *Continue* asks the model to go on and appends to the same message.
* **Response style.** The *Style* button in the composer (Default, Concise, Explanatory, Formal, Casual) adds one short instruction to
  the system prompt of that chat.
* **Rewind.** Every message has a rewind button (the clock): the conversation ends at that message. Nothing is deleted —
  the messages below stay attached to the node and are only detached from the visible path, so the notice's *Undo* brings
  them back, and a reload keeps the state. That is the chat counterpart of what the Code page does with files.
* **Temporary chat.** The ghost button in the header starts a chat that is never saved: it is kept out of the chat list and of
  `localStorage`, a banner says so, and it is gone when you close it, open another chat or reload the page.
* **Math and tables.** `$...$`, `$$...$$`, `\(...\)` and `\[...\]` are typeset by a small built-in renderer (fractions, powers,
  subscripts, greek letters, roots, sums, integrals and common symbols). Anything it does not understand (matrices, for example) is
  shown as source text in a chip. Wide tables scroll sideways instead of overflowing.

## Everyday features

* **Welcome tour.** On the first start (once the page knows your hardware) a short tour explains what fLLM is, shows your computer and
  the memory a model can use, points to the suggested models and gives three tips. Skip it any time; Settings, General, *Show the
  welcome tour* brings it back.
* **Keyboard shortcuts.** Press `?` (outside a text field) or Ctrl/Cmd+`/` for the list. New chat, search, focus the message field,
  show or hide the sidebar, stop, go to Chat, Code or Models, open Settings, next and previous chat, and regenerate have keys that you
  can change: choose *Change* and press the new keys. A key that is already in use, or that would disturb typing, is refused with a
  message. The choices are kept in the browser. Ctrl becomes Cmd on macOS.
* **Dictation.** The microphone button in the message field fills it with what you say, live. It uses the speech recognition of the
  browser (`webkitSpeechRecognition`), so the audio may go to the speech service of the browser's maker; fLLM never receives or
  stores audio. Where the browser or the desktop app's web view has no speech recognition the button is hidden, and Settings, Voice
  and alerts says so. The language defaults to the browser language and can be changed there.
* **Reading aloud.** Uses the speech synthesis of the browser. Settings, Voice and alerts has the voice, speed and pitch, a button to
  try them, and *Read answers automatically*. Markdown marks and code blocks are not spoken; long text is read in pieces. A stop
  button appears next to the microphone while it speaks (Esc stops it too). Other parts of the page can call
  `window.fllmSpeak(text)` and `window.fllmStopSpeaking()`.
* **Alerts.** When an answer that took more than 20 seconds, or a Code agent run, finishes while the window is not in front, a system
  notification is shown (the browser asks for permission once) and, if switched on, a short tone is played. Both are optional in
  Settings, Voice and alerts.

## Code: the coding agent

Pick a folder and describe a task. The model works through tools that the server runs inside that folder, plus tools of the page itself
(task list, `ask_user`, `task` for sub-agents). Tools of configured MCP servers are added too. Every step is shown as a card, file
changes with a diff, long output cut with a "Show all" button. The design borrows ideas from [OpenCode](https://opencode.ai) (MIT): build
and plan agents, sub-agents by type, permissions per tool, tolerant edits, a diff tool, sessions you can fork and revert, project
instruction files and custom commands. Nothing was copied from it; the prompts and the code are fLLM's own.

| Tool | What it does |
| --- | --- |
| `list_dir` | Folder tree (`depth` 1 to 4), folders first. Build output and what `.gitignore` excludes is listed but not opened. |
| `glob` | Find files by pattern, newest first. `.gitignore` and `.ignore` are respected. |
| `read_file` | Text file with line numbers, `offset` and `limit`. Binary files and images are refused with a clear message; a missing file suggests similar names. A Jupyter notebook (`.ipynb`) is shown as text cells with their outputs. |
| `grep` | Regular-expression search (or `literal` text, `word` matches). `output_mode` (`content`, `files`, `count`), `context` lines, `case_insensitive`, `head_limit`. Respects `.gitignore`. |
| `write_file` | Create a file or replace it completely. |
| `edit_file` | Replace text. Exact match first; if there is none, a unique match that ignores indentation and trailing spaces is accepted. CRLF files stay CRLF. A failing edit says where the closest text is, which lines differ and the line numbers of every place when the text is not unique. |
| `multi_edit` | Several edits to one file, all or nothing, one undo step. |
| `patch` | A unified diff over one or several files (create, change, delete), all or nothing. Hunks are placed by their content, line numbers only need to be roughly right. |
| `diagnostics` | A quick check without a language server: `python -m py_compile`, `node --check`, JSON parse for a file; `cargo check`, `tsc --noEmit` or `go vet` for the project when it has `Cargo.toml`, `tsconfig.json` or `go.mod`. Returns the first problems. Only fixed commands run. |
| `bash` | Run a shell command; stdout and stderr keep their order, long output keeps its start and its end. A `cd` stays in effect for the next call (never outside the folder). With `background: true` it starts a job and returns a job id. A timeout or Stop ends the whole process tree. |
| `bash_output`, `bash_kill` | Read the output of a background job (the newest 256 KB are kept) or stop it. Jobs are stopped when the session ends, the folder changes or the server stops. |
| `memory` | List, add, update or remove short memory entries (see below). |
| `web_fetch`, `web_search` | Download an http(s) page as readable text (10 s connect and 30 s total timeout, 2 MB body cap, at most 5 redirects); best-effort web search through an HTML search page that needs no key. |
| `todo_write`, `todo_read` | The task list shown in the side panel (page tool). |
| `task` | Start a sub-agent with a fresh context (page tool, below). |
| `ask_user` | Ask you a question with a few answer choices (page tool). |

The server lists its tools, with the permission category the page asks about (`read`, `edit`, `command`, `network`, `memory`), at
`GET /api/studio/agent/tools`. `GET /api/studio/agent/changes` lists every file change of the session (id, path, `create` or `edit`,
time), `POST /api/studio/agent/undo` takes `{"steps": n}` or `{"to": id}` to go back several changes at once (newest first) and
`{"redo": n}` to apply undone changes again. `POST /api/studio/agent/cancel` ends running commands (the Stop button),
`POST /api/studio/agent/files` (`{query, limit}`) lists workspace files and folders by fuzzy match, `POST /api/studio/agent/instructions`
returns the project instruction files and the memories, and `POST /api/studio/agent/memory` (`{scope, op: get | set | append, text}`)
reads and writes a memory file. All of them, like the tools, can only be reached from the machine fLLM runs on. The chat page can use
the network tools without an agent session through `POST /api/studio/web/fetch` (`{url, max_chars?}`) and `POST /api/studio/web/search`
(`{query, limit?}`).

**Agents and system prompts.** The header switches between the *Build* agent (reads, changes files, runs commands) and the *Plan* agent
(only reads, then writes a plan; "Run the plan" hands it to Build). Alt+B does the same. Under Build, *Ask* and *Auto-edit* choose
whether file changes are asked about. The system prompt is assembled from separate parts: the agent's rules, a note for the model family
(Qwen, Llama, Mistral, Gemma, gpt-oss or generic, read from the model name) with an example of a correct tool call, an environment block
(folder, system, shell, git branch and state, date), the project instruction files and the memory. Small models (8B parameters or less,
or a context of 8192 tokens or less) get a short numbered-rules form and one tool call at a time; larger ones get the full guidance
(how to choose a tool, how to edit safely, how to check the work, when to use the task list, when to ask, what not to do). Mentions of
tools the server does not offer are left out.

| Mode | What the agent does on its own |
| --- | --- |
| Build, Ask (default) | Reads and searches. Asks before every file change and every command. |
| Build, Auto-edit | Also changes files. Still asks before commands. |
| Plan | Only reads, then writes a plan. |

**Permissions.** The *Permissions* box of the side panel has a table: for each kind of action (read and search, change files, commands,
web, MCP tools, memory) and for each agent (build, plan) *allow*, *ask* or *deny*; "default" keeps the built-in behaviour. Below it,
pattern rules per folder (or `/allow` and `/deny`): one `category: pattern` per line with the categories `bash`, `edit`, `network`, `mcp`,
`read` and `memory`, for example `bash: npm test*`, `edit: src/**`, `network: *`, optionally for one agent: `[plan] bash: ls*`. A deny
rule or a *deny* mode always wins; an allow rule answers the question but never overrides a *deny* mode. An allow rule never covers a
command with shell operators, and a risky program (`rm`, `curl`, `sudo`, `git`, shells, ...) is only covered by a rule that names it
(`bash: git status*`, not `bash: *`). "Always" on a question allows that kind of action for the rest of the session. A patch is checked
against the rules for every file it touches. A call that is repeated three times in a row with the same arguments is not run again; the
model is told to try something else.

**Check my edits automatically.** In the *Settings* box: after every edit the agent runs `diagnostics` for the changed file and the
first problems are added to the edit result (and shown on the card). It is off by default because it can take seconds per edit.
`/check [path]` runs the same check by hand.

**Sub-agents.** The model can hand a self-contained job to a sub-agent with a fresh context and gets back only its report. `task` with
`agent: "explore"` (default) can only read and search; `agent: "general"` may also change files and run commands (the usual questions
still apply; the Plan agent can only start explore agents). Several started in one turn run in parallel as nested cards with live
status. They have a step and context budget and stop when you stop the agent. All requests share the same system prompt and tool list,
so the server can reuse its prompt cache.

**Your own sub-agents.** Settings, *Sub-agents* defines further agents as Markdown files: `agents/<name>.md` in the Studio folder (all
folders) or `.fllm/agents/<name>.md` in a workspace. The front matter holds `name`, `description`, `tools` (a comma-separated list of
tool names; empty keeps the tools of the session), `model` (empty uses the model of the session, so one agent can run on another model or
provider) and `max_steps`; the body is the instruction the sub-agent starts with. The `task` tool lists them by name, and an agent whose
tool list contains no writing, command or memory tool is treated as read-only. The list is read when a folder is opened, so a new agent
shows up without a restart.

**Hooks.** Settings, *Hooks* attaches shell commands to five points of the loop: `pre_tool_call`, `post_tool_call`, `on_file_edit`,
`on_session_start` and `on_session_end`. A hook gets one JSON object on stdin (`event`, `tool`, `arguments`, `workspace`) and runs in the
workspace folder with your rights; its exit code decides: `0` passes and its output is added to the tool result as notes, `2` blocks the
call and its stderr is the reason the model reads, any other code is reported as a failed hook without stopping the agent. A hook may
also print `{"decision": "block", "reason": "..."}` or `{"context": "..."}`. The matcher is a comma-separated list of tool names, a
prefix such as `edit*`, or empty for every tool; a timeout of 1 to 300 s per hook keeps a hanging command from blocking the turn. That is
how a formatter runs after every edit, a guard refuses writes outside a folder, or a script records what the agent did — deterministically,
instead of hoping the prompt is followed. The list lives in `hooks.json` next to `studio.json`; *Test* runs the matching hooks once with a
sample payload. The routes are `GET`/`POST /api/studio/hooks` and `POST /api/studio/hooks/test`; the agents use
`GET`/`POST`/`DELETE /api/studio/agents`.

**Sessions.** Sessions of each folder are saved in the browser (up to 30) and get a short title from the model after the first answer.
The side panel lists them with search, *Rename*, *Fork*, *Export* (a Markdown file with the messages and the tools used) and *Delete*.
Hover a message of yours for *Fork* (a new session that continues from before that message; its text goes back into the input) and
*Revert* (the conversation and the file changes since that message are taken back, through the same change log as `/rewind`; the
notice offers *Undo the revert*). Messages from before a compaction cannot be forked or reverted to. Stopping the agent leaves the
session valid: calls that did not finish get an "interrupted" result, so the next request works with every chat template, and commands
the server was running are ended.

**Message queue and shell.** Enter while the agent works queues the message (shown as a chip above the input; ✕ removes it) and sends
it when the run is over; Stop returns queued text to the input. `!command` runs a shell command yourself; its output is shown and
given to the model with your next message. `@` completes file and folder names with a fuzzy search over the folder (it respects
`.gitignore`); the content of mentioned files is attached. The header has a model switch for the agent.

**Long sessions.** When the conversation passes about 70% of the model's context, older turns are summarised by the model and replaced
(the task list and the changed files are kept, and a notice says so). `/compact [focus]` does it on demand. Tool output keeps its start
and its end when it is cut.

**Project instructions and memory.** `AGENTS.md`, `FLLM.md` or `CLAUDE.md` (the first one found per folder, in this order) of the folder
and of its parent folders up to the repository root are part of the prompt, nearest first, capped. `/init` has the agent write an
`AGENTS.md`. The memory is two short lists: `.fllm/memory.md` in the folder and one for you in the Studio folder (`agent-memory.md`).
The model adds entries itself with the `memory` tool when it learns something lasting (your preferences, project conventions, commands
that work); `/remember [global] text` adds one by hand, and the *Memory* box in the side panel shows and edits both. Both files are
plain Markdown lists (`- entry`), at most 100 entries of 400 characters each.

Commands (type `/` for the list, with search): `/init`, `/compact`, `/resume`, `/fork`, `/revert`, `/rename`, `/export`, `/rewind` (undo
file changes back to a chosen point), `/undo [n]`, `/plan`, `/build`, `/ask`, `/edit`, `/agent`, `/model`, `/remember`, `/memory`,
`/check`, `/diff`, `/review` (the changes with the findings of a fixed set of rules; `/review staged` and `/review branch <base>` too), `/security-review` (only the security rules), `/status`, `/cd path`, `/allow`, `/deny`, `/rules`, `/clear`. Your own commands are Markdown files in
`.fllm/commands/` of the folder: `review.md` becomes `/review` (a built-in command of the same name wins), an optional `description:`
front matter line is shown in the list, and
`$ARGUMENTS` (or `$1` ... `$9`) is replaced by what follows the command. Keyboard: Alt+B switches the agent, Alt+L picks a session,
Esc stops, Mod+N starts a new session (all rebindable under Keyboard shortcuts).

Commands started in the background appear in the *Background jobs* panel, where you can read their output and stop them. Requests of
agent turns ask the server to constrain tool calls to valid ones (`tool_call_grammar`); servers that do not know it ignore it.

**Sandbox.** The approval rules decide *whether* a command may run; the sandbox decides what it can reach once it runs. With
`bwrap` (bubblewrap) installed, every `bash` command and every background job runs with the root filesystem mounted read-only, the
workspace writable, a private `/tmp` and its own PID and IPC namespace; `FLLM_SANDBOX=strict` also takes the network away and
`FLLM_SANDBOX=off` switches it off. The server checks at the first command whether bubblewrap really works (a test run, not just the
binary); if it does not — a missing `bwrap`, or a kernel or container policy that forbids user namespaces — the command runs as before
and `GET /api/studio/agent/sandbox` says why. Nothing is reported as sandboxed that is not; `/sandbox` shows the state.

On **Windows** there is a different mechanism (implemented and checked on one Windows 11 machine, **not yet on others**, so it is off by default
there and you switch it on under Settings > Runtime or with `FLLM_SANDBOX=workspace`). Each command is started through
`fllm-rs sandbox-exec`, which runs it under a restricted token at low integrity level inside a job object:

- *Enforced (mode `workspace`)*: the whole process tree is killed on stop, timeout or when the server ends; writes are refused outside
  the workspace and a private temp folder under `%USERPROFILE%\AppData\LocalLow\fLLM`; privileges are removed and administrator groups
  are deny-only; at most 512 processes and three quarters of the physical memory per command; no clipboard, desktop or window-handle
  access; environment variables that look like secrets (`..._TOKEN`, `..._KEY`, `..._PASSWORD`, ...) are removed and `TEMP`/`TMP` point
  to the private folder.
- *Not enforced*: **reading** (the whole disk stays readable, including your profile), **the network**, and other places that are
  writable at low integrity (such as `AppData\LocalLow`). `strict` is **unavailable** on Windows because a network cut-off needs an
  AppContainer, which is not implemented: in that mode commands are refused with a message instead of running with a network.
- *Side effect*: the workspace folder gets a low integrity label (with `icacls`, once, remembered in `sandbox-labelled.txt`), which is
  what lets a low-integrity process write there. Drive roots, the user profile, personal folders such as Documents and system folders
  are refused as workspaces for this reason. Programs that need to write elsewhere (`~/.cargo`, `%APPDATA%`) fail in `workspace` mode,
  as they do on Linux with the read-only root.
- *Self-test*: before the Windows sandbox counts as available, fLLM runs real commands through it on your machine and demands that a
  write into the allowed folder works and a write outside fails. If that does not hold, the status says why and commands run unwrapped
  (as with a missing `bwrap` on Linux). If the sandbox works but cannot start one particular command, the command fails with the
  reason; it never silently runs unsandboxed.

On **macOS** a Seatbelt (`sandbox-exec`) profile exists for the three modes, also off by default and **unverified** (profile text and
arguments are unit-tested, no Mac was available; the same self-test guards it). Diagnostics and hook commands are not sandboxed on any system.

Safety: the agent tools are reachable from the machine fLLM runs on only, also when an API key is set. Paths cannot leave the folder
(also through symbolic links; on Windows the drive letter and folder names are compared without regard to case). The agent can still run
any command you allow, so choose the folder with care. Set `FLLM_NO_AGENT=1` to switch the agent routes off completely. On Windows
commands run with `cmd.exe`; a command written over several lines is joined with `&`, and a `!` inside a command may be taken by
cmd's delayed expansion, which the working-directory tracking needs (known limit, not worked around yet).

Network safety: `web_fetch`, `web_search` and the two web routes only speak http and https, refuse URLs with credentials, and refuse
targets on loopback, private, link-local (including cloud metadata) and other reserved addresses. The host name is resolved by the
server and the connection is pinned to the checked address; every redirect is checked again. They are reachable from the machine
fLLM runs on only, like the other agent routes. To let them reach a documentation server on your own network, start fLLM with
`FLLM_AGENT_ALLOW_PRIVATE_URLS=1`. `FLLM_NO_AGENT=1` switches them off too.

A small model often writes a command into the answer instead of calling the tool. Use a model with good tool calling for real work.

The pure logic of this page (rule matching, compaction, command expansion) has tests: `node rust/engine/src/webui/tests/code.test.js`.
The logic of assistants, memory and project retrieval is tested by `node rust/engine/src/webui/tests/projects.test.js`; the chat tool loop,
the memory tool and secret refusal by `tools.test.js`; system prompts, titles and the parsing of all page scripts by `prompts.test.js` (same folder).
The pure logic of this page (rule matching, permission modes, compaction, message repair, session cutting, command expansion, the
prompts) has tests: `node rust/engine/src/webui/tests/code.test.js` and `node rust/engine/src/webui/tests/codeprompts.test.js`.
The logic of assistants, memory and project retrieval is tested by `node rust/engine/src/webui/tests/projects.test.js`.

## Models

Shows the loaded model, downloaded models and suggestions. Search Hugging Face (GGUF repositories, most downloaded first), look at the
files with a hint whether they fit into the free memory, and download with a progress bar. **Load** restarts the server with another
model: the page asks first, the server starts a new process with the same options and the new model, and the page reloads when it is
back. This needs the server to run in a terminal or service you can restart; the new process takes over the port.

If the model that should load cannot be used (an architecture fLLM does not support yet, a damaged file, not enough memory), the Studio
still opens on the Models page and shows the reason; the model is forgotten so that the next start does not fail the same way. A model
is also checked before the server is restarted with it.

**Load options.** *Load* opens a dialog with an *Advanced* section: context length, layers on the graphics card (a number or
`all`), processor threads, the memory type of the conversation (f16, q8_0, q4_0, ...) and how many chats are answered at once.
Everything left on *auto* is chosen by the server. The choices are remembered per model file and shown on the loaded-model card.
*Speculative decoding* (guessing ahead to answer faster, a start option of the server, so it is chosen here and not per chat) can be off, *prompt lookup* (no extra model; guesses from the text of the conversation, good for code edits, summaries and quoted text), *draft model* (pick a smaller downloaded model of the same family) or *multi-token prediction* (only models that have the heads, on the processor or a CUDA graphics card). The answers are the same as without it, only the speed changes, and it can also be slower when the guesses are rarely right. Models with recurrent layers (for example Qwen3.5 or Mamba) cannot use it. The choice is remembered per model.
Combinations that cannot work (for example q4_0 conversation memory on a Vulkan, DirectX 12 or Metal device, or GPU layers while
running on the processor) are refused with a message. There is no switch for prefix caching: the engine has no such option, so none is offered.

**Unload.** *Unload* on the loaded-model card restarts the server on the same port without a model, keeping all other options.
*Settings > Models* has "Unload the model when idle" (minutes without a request; 0 never); the next *Load* works as usual. Chat and
Code ask you to load a model while none is loaded.

**Downloads.** Downloads are queued: one runs, the others wait. Each can be paused, resumed or cancelled; the finished part stays on
disk, so a later download continues where it stopped. The list shows speed and time left and is kept by the server, so it is the
same after a page reload (but not after the server restarts). *Settings > Models* sets the models folder for new downloads
(models that are already downloaded stay where they are).

**Browse.** The *Browse* buttons open the file or folder dialog of the operating system on the machine that runs fLLM (PowerShell on
Windows, `osascript` on macOS, `zenity` or `kdialog` on Linux; without one the button explains that and you type the path). It works
from that machine only.

**Model details.** The file list of a Hugging Face repository shows the start of the model card, the quantization of each file, its
size and whether it should fit into the free memory, with the files that fit first.

**Your own GGUF files.** The *Add a model file* card takes the path of a .gguf file, or of a folder (its .gguf files are added). The files stay where they are and appear in the list of models on the computer; *Remove* only takes them off the list.

**Hugging Face account.** Private and gated models need an access token: paste one (read access is enough) into the Hugging Face card
of the Models page. It is checked with Hugging Face and kept in the same file the Hugging Face tools use (`~/.cache/huggingface/token`),
so downloads from the page, `fllm-rs pull` and `fllm-rs run` all use it. A `HF_TOKEN` variable wins over the saved token.

**NPU (experimental).** If your computer has an Intel NPU, Settings, Hardware shows an NPU card. When the NPU is found but its runtime (OpenVINO,
a download of about 60 to 85 MB from the official Python package host, checked against a fixed checksum) is missing, the card has an **Install the runtime**
button with a progress bar and a Cancel button, and **Remove the runtime** takes it away again. Once the card says *Ready*, the load dialog of the
Models page (Advanced, *Run on*) lists the NPU. Be aware: it is experimental and has not been run on NPU hardware by the authors yet; it is slower than the
processor for many models, the weights are held twice in memory and only dense models work. The automatic choice never uses the NPU. If the NPU cannot
take a model, the model runs on the processor and the card shows why. AMD and Qualcomm NPUs are shown as *detected, not supported yet*.

**Updates.** The Studio asks GitHub for a newer release once per start. If there is one, the Models page offers to download the
installer of this system and start it (Windows: the setup program, Linux: the AppImage or the package).
## Settings

Grouped as General, Appearance, Models (models folder, idle unload), Hardware (devices, the NPU card, a speed test of the loaded model, processor
feature limit, memory speed, a processor check), Server and API (address, request limits, API keys, a tester), Logs (the latest
server messages with refresh and copy), Usage and cost (tokens per model counted in this browser, with the prices you enter),
and under More: model details, text tools and ready-made examples for connecting Claude Code,
the OpenAI libraries and curl.

**Usage and cost.** Every answer returns its token counts; the page adds them per model and keeps them in the browser
(Settings, *Usage and cost*; `/cost` shows the summary in the Code page). A local model costs nothing and needs no entry.
For a model behind a provider, add a price row — a pattern matched against the model name (the longest wins, `*` matches any
tail, for example `gpt-6*`), dollars per million input and output tokens, and an optional note. Models without a row show
tokens only: no dollar amount is invented. The rows live in `prices.json`; `GET`/`POST /api/studio/prices` are the routes.

## Tools (MCP)

Settings has a Tools (MCP) page for Model Context Protocol servers: programs that offer tools (files, search, databases, ...) over their
standard input and output. Add a server with a name, a command (for example `npx`), arguments (one per line) and optional environment
variables. Each server shows its state (starting, ready, or failed with the reason), its tools and its last error output; you can restart,
edit or remove it. The list is saved in `mcp.json` next to `studio.json` in the common format
`{"mcpServers": {"name": {"command": "npx", "args": ["-y", "pkg"], "env": {"KEY": "value"}}}}`, so a file from another MCP client works as is.
Edits to the file are picked up the next time the page or the API reads it. An optional `"timeout_seconds"` per server sets the limit for
one tool call (default 60). A server that stops is started again on the next call, with a growing pause after repeated failures.

**Remote servers.** A server can also be reached over HTTP (MCP "Streamable HTTP") instead of being started as a program:
fill the *URL* field and leave the command empty. One POST per message; the answer arrives as JSON or as an event stream, and
the `Mcp-Session-Id` the server hands out is sent with every later request. Extra headers go in the same form
(`NAME=value` per line); a value may reference an environment variable as `${NAME}`, so a token does not have to be written
into `mcp.json`. In the file that is `{"mcpServers": {"name": {"url": "https://...", "headers": {"Authorization": "Bearer ${MY_TOKEN}"}}}}`.
A URL is refused unless it starts with `http://` or `https://`.

**Warning:** an MCP server runs on this computer with your user rights. Only add servers you trust. Saved environment variable values are
never sent back to the page.

API: `GET /api/studio/mcp/servers`, `POST` and `DELETE /api/studio/mcp/servers`, `POST /api/studio/mcp/restart`,
`GET /api/studio/mcp/tools` (`{tools: [{server, name, description, input_schema}]}`) and `POST /api/studio/mcp/call`
(`{server, name, arguments}` gives `{content, is_error}`). Everything except the two GET routes is administration (see below).

## Model providers

Settings has a Model providers page for models that do not run on this computer: cloud services and other servers. A provider is an
OpenAI-compatible address (OpenAI, OpenRouter, Groq, Together, Mistral, the Google Gemini OpenAI-compatible endpoint, LM Studio, vLLM and
many more), Anthropic, or an Ollama server. Pick a template (it fills in the address only), enter a key if the service needs one and press
Test: fLLM lists the models of the provider and tells you plainly what went wrong (for example "the key was not accepted"). Providers can be
switched off, edited, refreshed and removed. Their models appear in the model menu of the header after the local ones, as
"Provider · model". Choosing one sends that conversation to the provider; streaming, the stop button and the token meter work as usual.
Settings of the local engine (top-k, min-p, mirostat, DRY, grammars, speculative decoding) are not sent to other services, and the context
meter shows tokens used without a limit. Anthropic requests are translated to the Messages API and the answers back to the OpenAI format
(text, tool calls, images and token usage).

The list is saved in `providers.json` next to `studio.json`: `{"providers": [{"id", "name", "kind", "base_url", "api_key", "enabled",
"local", "models"}]}` with `kind` one of `openai`, `anthropic` or `ollama`. A server that runs on this computer or on your network
(Ollama, LM Studio, vLLM) needs "This server runs on this computer or network" (`"local": true`); without it addresses on this computer, in
private networks and other internal ranges are refused, and redirects are never followed. A saved key is dropped when the address of a
provider is changed to another host without entering the key again.

**Security:** keys stay on this computer, in `providers.json` (readable only by your user on Linux and macOS). The page never sees them: every
route returns `has_key` instead of the key, and the chat route adds the key when it passes your request on. Keys are not written to logs and
are removed from error messages. Everything you type in a chat with such a model is sent to that service; read its privacy terms. All
provider routes work only from this machine, even when an API key is configured.

API: `GET`, `POST` and `DELETE /api/studio/providers`, `POST /api/studio/providers/test`, `GET /api/studio/providers/models[?refresh=1]`
(`{models: [{provider, provider_name, kind, local, id}], errors}`) and `POST /api/studio/providers/chat` (`{provider, request}` with an
OpenAI chat-completions body; answers like `/v1/chat/completions`, as JSON or as an event stream).

## Skills, scheduled tasks and search of past chats

Settings has a "Skills and schedules" page. Details and the page API are in `docs/changes/personal-agent.md`.

**Skills.** A skill is a saved procedure: a Markdown file `<name>/SKILL.md` with a front matter (`name`, `description`, `when`) and the
steps. Your own skills are kept in the `skills` folder next to `studio.json`; the skills of a project in `<project>/.fllm/skills/`. The page
lists, edits and deletes them. The model can read them with the `skill` tool and, after a task worth repeating, propose a new one that you
approve before it is saved.

**Scheduled tasks.** A task runs a prompt on a schedule (`every 30 minutes`, `hourly`, `daily at 08:30`, `weekdays at 09:00`,
`weekly on mon,thu at 18:00`, or a cron line) in local time, either as a chat answer or as an agent run in a folder that may only read,
search and look things up on the web. The last 20 results of each task are kept in the page. **Tasks run only while the app is running
with a model loaded**; runs missed while it was closed are not made up. A task never runs twice at once. "Run now" starts one at once, and
a task can ask for a desktop notification while a page is open.

**Search of past chats.** Chats and code sessions are searched by their words. A plain-text copy is kept in `sessions.jsonl` next to
`studio.json`, so they can be found after the browser storage was cleared; it can be switched off and deleted in the same page.
Temporary chats are never copied.

These routes (`/api/studio/skills`, `/api/studio/sessions/*`, `/api/studio/schedule*`) work only from this machine, reads included.

## Plugins

A plugin is a folder that adds several things at once, without copying anything into your own configuration:

```
plugins/<name>/plugin.json          {"name", "version", "description", "author"}
plugins/<name>/skills/<skill>/SKILL.md
plugins/<name>/agents/<name>.md     same front matter as your own sub-agents
plugins/<name>/commands/<name>.md   your own slash commands (description in the front matter)
plugins/<name>/hooks.json           the same shape as Settings > Hooks
```

Settings, *Plugins* installs a plugin from a local folder or from a repository (`git clone --depth 1`, with an optional
branch; the clone gets `GIT_TERMINAL_PROMPT=0`, so it can never ask for a password, and only https, http, ssh, `git@` and
`file://` addresses are accepted). The folder is copied into `plugins/`, lists what each plugin
contributes and removes it again; removing the folder removes every contribution, because nothing was merged into your
files. Contributions are read-only: a skill from your own folder wins over one of the same name, a plugin's sub-agents
cannot be edited in place (copy the file into `agents/`), and its hooks appear in Settings > Hooks under *From plugins*
with the plugin's name. A hook of a plugin gets `FLLM_PLUGIN_DIR` in its environment, so it can call the scripts that ship
with the plugin. Installation refuses links, more than 5,000 files and more than 50 MB, and a plugin name has to be a
lowercase slug. Routes: `GET`/`POST`/`DELETE /api/studio/plugins`, `GET /api/studio/plugins/commands`,
`GET /api/studio/plugins/command?plugin=&name=`.

## Worktrees

A change can run in its own checkout on its own branch, while the main folder keeps its state: `/worktree new <name>`
creates `git worktree add .fllm/worktrees/<name> -b <name>` and switches the agent there, `/worktree leave` goes back to
the main worktree, `/worktree remove <name>` deletes the checkout (the branch stays), `/worktree` lists them. `/batch <name>
<task>` is the same in one step: it makes the worktree and runs the task there, so a risky change never touches the main
tree and you decide afterwards whether to merge the branch. `.fllm/worktrees/` is added to `.git/info/exclude`, so the
worktrees never show up as untracked files. Everything goes through `git worktree`, never through a shell; the route is
`POST /api/studio/agent/worktree` with `{action: new|list|leave|remove, name, branch}`.

## Language servers

The `lsp` tool asks a real language server instead of guessing: `diagnostics` for a file (types, imports, borrows —
exactly what the server says, with file, line and column), `symbols` for a workspace-wide symbol search and `definition`
for a position. It uses `rust-analyzer` for Rust, `typescript-language-server` for TS/JS, `pyright-langserver` or `pylsp`
for Python, `gopls` for Go and `clangd` for C/C++ — whichever is first on `PATH`. Without a server the tool says so and
names the ones that are installed; the `diagnostics` tool then still works, because it calls the compiler directly. Each
call starts the server, asks and stops it, so the first answer of a big project takes as long as the server needs to
index.

## Document search with an embeddings server

Documents are cut into passages and ranked by keyword (BM25) in the browser. Settings > *Search* can point at an
embeddings server (`fllm-rs serve --encoder <model>`, or any OpenAI-compatible `/v1/embeddings`); then the ranking is
hybrid: 60 % keyword and 40 % cosine, and a passage is also found when the question uses another word for the same thing
(cosine has to reach 0.3 to count as a hit on its own). The vectors are computed once per chat and kept with the index;
the *Test* button shows how many numbers an embedding has and how close two sentences are. Without an address the search
stays keyword-only and says so — the chat server itself does not embed. The setting lives in `embeddings.json` next to
`studio.json`; routes `GET`/`POST /api/studio/embeddings` and `POST /api/studio/embed`.

## Sandbox and turbo

Settings > *Runtime* decides how freely the app may work, and the choice is kept in `runtime.json`:

- **sandbox** — `workspace` (system read-only, workspace writable, private `/tmp`, own PID namespace), `strict` (also no
  network) or `off` (a plain shell). On Linux it needs `bwrap`; Windows (low-integrity token in a job object, no `strict`) and
  macOS (Seatbelt) are implemented but not yet verified on real machines and default to `off`. Without a working mechanism the
  command runs unwrapped and the page lists what is enforced, what is not and why instead of pretending. `FLLM_SANDBOX` in the
  service file wins over the page.
- **turbo** — no permission questions at all: every tool call runs, deny rules and the sandbox included. It is a loud red
  switch with a confirmation, it stays off until somebody turns it on, and `/turbo on|off` does the same from the agent.
  `/sandbox off|workspace|strict` shows or sets the mode.

The agent reads both from the server (`GET /api/studio/runtime`), so page and process cannot disagree.

## The browser

The Code agent can use a real web browser, the way a person does: open a page, read it, click, type, scroll, take a
screenshot. You watch the same page live in the **Browser** panel of the Code page (the globe button in the header, closed by
default), can click, type and scroll in it yourself, and can leave comments on elements that the agent then acts on.
The main use is testing the app the agent is building: it starts the dev server, opens `http://localhost:3000`, clicks through
it and reads the console errors and failed requests that come back with every result.

**It is off until you switch it on:** Settings > *Browser* > *Let the agent use a browser*. Until then the tools do not even
appear in the tool list.

How it works: fLLM starts a Chromium-family browser that is already on the computer (Google Chrome, Chromium, Microsoft
Edge or Brave; Edge is part of Windows) and drives it over the Chrome DevTools Protocol from the server, so it behaves the same
in the desktop app, in the Studio in a browser tab and in Docker. Nothing is downloaded and no automation framework is
used. If no browser is found, the tools and the panel say so and name what to install; nothing else is affected. To choose
one yourself: the environment variable `FLLM_BROWSER` or Settings > *Browser* > *Browser program*.

- **Own profile.** The browser runs with its own profile folder (`browser-profile` inside fLLM's data folder, see
  [Where fLLM keeps its files](#where-fllm-keeps-its-files)), never with your own browser profile, without extensions and
  without first-run screens. Cookies stay in that profile between sessions (switch *Keep logins and cookies* off for a
  throw-away profile each time).
- **Headless by default.** *Show the browser window* opens a real window instead, for example to log in to a site once; the
  agent then works with that session. Do this only for sites you are willing to let the agent act on.
- **The tools** (all in the permission category `network`: they ask first, "Always" allows them for the session, a rule such as
  `network: localhost*` or `network: *.example.com*` allows or denies by the page the browser is on, deny rules win, turbo skips
  the questions): `browser_open`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_select`, `browser_press`,
  `browser_scroll`, `browser_back`, `browser_screenshot`, `browser_tabs`, `browser_close`.
  `browser_snapshot` is what a model without vision uses: the page as compact text (title, URL, headings, links, buttons, fields)
  with a number `[ref=N]` on everything that can be operated; a ref stays the same for the same element while the page stays on
  the same document. A screenshot is shown in the tool card and, when the loaded model has vision, attached to the conversation
  (only the newest one is kept); a model without vision gets the text description only.
- **Waiting.** After opening a page or after a click the tool waits (up to 20 s and 8 s) until the page has loaded and the
  network is quiet, then returns the new snapshot, console errors, uncaught exceptions, failed requests and HTTP errors.
  JavaScript dialogs are answered by the server (alerts accepted, the rest dismissed) and reported.
- **Safety.** Only `http` and `https` pages can be opened; `file:`, `chrome:`, `devtools:`, `view-source:`, `javascript:`,
  `data:` and other schemes are refused, for navigations of the agent, of you, of links and of redirects. Link-local and
  cloud-metadata addresses (169.254.x.x and the like) are never reachable. `localhost` is allowed while a workspace is open
  (the default *local addresses* setting `auto`), the local network (10.x, 172.16-31.x, 192.168.x) only with the setting `all`,
  and `none` forbids both. A name is resolved and every address it returns is checked, so a name that points to an internal
  address is refused; because the browser resolves the name again itself, a server that changes its answer between the two
  lookups is not fully excluded. Every request of a page is checked too: a page from the internet cannot load anything from
  this machine or your network. Downloads are refused; camera, microphone, location and notification requests are denied.
- **Only on this computer.** The panel and its controls (`/api/studio/agent/browser/*`) are refused from every other machine,
  with or without an API key, like the other agent routes: the browser can hold logged-in sessions.
- **It stops with the server.** The browser is closed when the agent session ends, after 20 minutes without use, when the
  server stops, and when the server is killed (a process-group watchdog on Linux and macOS, a job object on Windows).

**Taking over and comments.** In the panel you can type an address, go back and forward, reload, and use the page with the
mouse (click, double click, wheel) and the keyboard (click the picture first; Tab and Enter go to the page, not to the
Studio). The comment button switches on comment mode: click an element, write what should be done, and *Add comment*. The
comments (element role, name, text, `ref` and your note) are put in front of your next message to the agent, or sent at once
with *Send to the agent*.

**What is verified.** Linux with Chromium 156 in Docker (Debian 13), headless and with a window on a virtual display (Xvfb): opening a local page, snapshot, typing into a
field, choosing an option, clicking a button, the page changing, screenshots (PNG), tabs (a link that opens a new tab), going back, cookies kept across a restart of the browser, the refusals above, the
panel routes, closing, and the browser disappearing when the server is killed. The URL policy, the CDP framing (against a fake
server), the snapshot and the key mapping have unit tests. **Not verified:** Windows (Edge and Chrome, the job object; that code was not even compiled by the Linux build used for testing), macOS (its watchdog is the same shell script as on Linux),
a window on a real desktop, and the panel in the desktop app. The panel itself was exercised in a real browser
tab against a running server (clicks, typing and comment mode, on a wide and a narrow window); keyboard input into the preview from a physical keyboard and an IME was not.
Only Chromium-family browsers work; Firefox and Safari are not supported.

## Web search that needs no server

Two engines work the moment the app starts: DuckDuckGo over its HTML endpoint and Wikipedia over its public API — no key, no
instance, nothing to host. Settings > *Search* holds the list; a SearXNG instance or a Brave key can be added, and every
engine is asked on each search, with the results merged by URL. The answer says per engine what happened (`DuckDuckGo: 8
hit(s)`, `SearXNG: HTTP 403`), so a blocked engine never looks like "nothing found" — DuckDuckGo, for one, asks for a
captcha on datacenter addresses. The *Test a search* button runs one real query. The engine list lives in `search.json`,
the routes are `GET`/`POST /api/studio/search/engines` and `POST /api/studio/search`.

## Skills that come with the app

Settings > *Skills* has a library of six skills inside the binary (commit message, code review, debugging, tests first, SQL,
release notes). One click writes one into your skills folder as a normal file you can edit. *Import from URL* fetches any
`SKILL.md` over http(s) — a raw GitHub link, for example — and stores it under its front-matter name.

## Automatic approval

Asking a question for `ls` and `git status` costs attention, so harmless commands run without one: reading (`ls`, `cat`,
`rg`, `find`, `jq`, `sed`, `awk`), building and testing (`cargo build/check/test/clippy/fmt`, `npm run`, `npm test`, `tsc`,
`pytest`, `go build/test/vet`, `make`, `deno`, `bun`) and git reading or staging (`git status/diff/log/show/branch/add/…`).
The decision comes from a fixed table, not from a model: every part of the command line (split at `&&`, `||`, `;` and `|`)
has to be on the list. Four hard rules then reject a command no matter what: a redirect into anything but `/dev/null`, a
heredoc, a substitution (`$(…)`, backticks) whose inside is not harmless again, and any interpreter that would run
arbitrary code (`python3 script.py`, `node -e`, `node script.js`). Instead of the table, `python3 -m pytest` and
`node --check` pass, because they run tests or a syntax check.

Everything else still asks: `rm`, `mv`, `git push/reset/clean/checkout`, package installs, `curl`/`wget`, `docker`,
`sudo`, `kill`, and every network, MCP or edit call — auto approval only ever covers plain shell commands. A deny rule
beats it. `/autoapprove` shows the switch, `/autoapprove on|off` sets it (kept in the browser). Approved calls say why in
the tool card: “Approved on its own: reads, builds or tests only”.

## Notebooks

`read_file` shows a `.ipynb` as text cells; `notebook_edit` changes one cell without touching the JSON around it:
`{path, action: replace|insert|delete, cell, source, cell_type}` with 1-based cell numbers. Replacing the source of a code
cell clears its old output and its execution count, because they no longer match the code. `insert` takes the position the
new cell gets (1 to count+1) and a `cell_type` of code, markdown or raw; `delete` refuses to remove the last cell. The tool
answers with the cell list, so the model sees the notebook rather than the JSON, and it refuses anything that is not a
`.ipynb`.

## MCP resources

Besides tools, an MCP server can publish resources (files, records, documents it owns). The MCP page has a *Resources*
button per server that lists them with their uri and reads one on click; the model has the `mcp_resource` tool for the
same thing: only `server` lists the resources, `server` plus `uri` reads one. A server that does not support resources is
not an error — it simply has none. Routes: `POST /api/studio/mcp/resources` and `POST /api/studio/mcp/resource`.

## Notes between sessions

A chat lives in the browser, so one session cannot talk to another session that is closed. What works is a handover: the
`session_note` tool (`{text, to?}`, or `{list: true}`) writes a note into `.fllm/messages.json` of the workspace, and the
next chat shows it — the header has an *Inbox* button with the number of waiting notes, one click puts a note into the
composer, another marks it done. `to` narrows a note to one chat title; empty means every chat of the workspace sees it.
That is deliberately not a fake message bus: nothing runs in the background, the note simply waits. Routes:
`GET`/`POST`/`DELETE /api/studio/agent/messages`.

## The first start

Install, open, and the chat screen says what is missing instead of showing an empty box: *No model loaded* with one
button — *Download and start* — for the model the hub recommends for this machine (name, quantization and size are on the
button, and it says when the file may not fit into the free memory). The download runs with progress, speed and an ETA,
can be paused and resumed, and loads the model when it is done. The Models page does the same with the whole catalog,
Hugging Face search and a field for a GGUF that is already on disk. Nothing has to be typed in a terminal, and nothing is
downloaded without a click.

## The look

`webui/theme.css` is the design system and is loaded after the older sheets, so it refines them without a rewrite: one accent
colour per personality (**copper** by default — the colour of the material, since this is a Rust engine — with seven
alternatives in the accent picker; green is kept for “ok” states only), a slate-black background for the
dark theme and warm paper for the light one, an 8-pixel grid, two radii (8 for controls, 14 for surfaces), a type scale from
11 to 34 px, mono for meta and numbers, and motion tokens that respect `prefers-reduced-motion`. The rules follow the UI/UX
Pro Max guidance for developer tools (dark-first Swiss minimalism, bento surfaces, contrast of at least 4.5:1, visible focus
rings, no colour as the only signal). The sidebar is a floating rail with a hairline border, chat text is centred in a 900 px
reading column, the composer is the one elevated card, and the empty chat hides the sparkline instead of showing a flat line
and a lonely counter. Deliberate deviations from the generated `design-system/fllm-studio/MASTER.md`: copper instead of acid
green (the database default reads as every other AI tool), no uppercase eyebrows, no dotted meta strings, and cards sit on
hairlines instead of shadows.

## The desktop app

The Tauri app in `desktop/` starts `fllm-rs studio` on a free port and shows that page in **its own window** — no
browser, no address bar, no tab; the window is a webview of the running server, the way VS Code, Slack or Discord are built.
It passes `--no-open` and removes `FLLM_OPEN_BROWSER` from the server's environment, so a browser cannot open even if the
variable is set globally. The tray menu only handles the server and the window: *Open fLLM* brings the window forward,
*Restart server* restarts it, *Open log folder*, *Quit*.

The command line version is separate: `fllm-rs studio` opens your browser once (for people who downloaded only the binary
or run it in Docker) and `fllm-rs studio --no-open` does not; `fllm-rs serve` opens nothing at all.

## Where fLLM keeps its files

fLLM has its own folder (Settings > Models shows the path): `%LOCALAPPDATA%LLM` on Windows, `~/Library/Application Support/fLLM` on macOS and
`~/.local/share/fllm` elsewhere (`$XDG_DATA_HOME/fllm`). It holds the settings (`studio.json`), `plugins/`, `agents/`, `hooks.json`, notes and prices, and for the agent's browser `browser.json` and the profile folder `browser-profile/`.
`FLLM_HOME` puts it somewhere else. Earlier versions kept these files inside the Hugging Face cache (`~/.cache/huggingface/fllm`); the first start of
a newer version moves them into the new folder.

Downloaded models stay in the Hugging Face cache (`~/.cache/huggingface/hub`) when it exists, so a model downloaded by another tool is not downloaded again;
on a computer without one they go to `models/` inside fLLM's folder. Settings > Models can choose any other folder, and `FLLM_MODELS_DIR` / `FLLM_CACHE` do the
same for a single run. The GPU tuning results and the compiled shaders are caches and stay in the cache folder of the system (`~/.cache/fllm`, `%LOCALAPPDATA%llm`).

## Who may change things

Reading (status, hardware, models) is open like the rest of the API. Changing the server through `/api/studio/*` (settings, model downloads, loading and
deletions, updates, API keys, the engine options) needs, when API keys are in use, a key with the **admin right** (Settings > Server and API: tick
"Admin"; keys given with `--api-key` have it). A key made for chatting cannot do any of that, so one user cannot change what everybody else uses. Without
keys the changes are accepted only from the same machine, from the page of this server, addressed by `localhost` or an IP address. Other websites in your
browser cannot use them. Everything that runs commands or reads files on the server (agent tools, hooks, plugins, MCP servers, providers, the runtime
switch, the engine options) stays on the machine itself, with or without a key.

## Docker and Kubernetes

The page works in a container the same way. Two things differ:

- **Settings are kept in a volume.** The images set `FLLM_HOME=/home/fllm/.cache/fllm` and the Compose files mount a volume there (`fllm-data`; the GPU files keep the whole `~/.cache`). Without a volume the
  options saved on the page are gone when the container is removed.
- **Set an API key.** A container sees every request as coming from another machine (Docker's gateway), so without `FLLM_API_KEY` the page can show things but not change them. With the key (it is an admin key) the page works as on your own computer: enter the key in the page when it asks.
- **No browser inside.** The agent's browser needs a Chromium-family browser in the container; the images do not contain one (see [docker.md](../docker.md)).
- **Restarting.** In a container the server starts itself again in place (same process, so the container keeps running). That is what "Restart the server now"
  on Settings > Engine and loading another model do. If that ever fails the process ends and the restart policy of the container (`restart: unless-stopped`,
  Kubernetes' default) starts it again. `FLLM_CONTAINER=1` forces this behaviour where fLLM cannot tell it runs in a container.
