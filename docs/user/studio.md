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

Pick a folder and describe a task. The model works through tools that the server runs inside that folder (`list_dir`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`, `bash`, and more when the server offers them, such as `multi_edit`, background shell jobs,
`web_fetch` and `web_search`), plus tools of the page itself: a task list, `ask_user` and `spawn_agent`. Tools of configured MCP servers
are added too. Every step is shown as a card, file changes with a diff, long output cut with a "Show all" button.

| Tool | What it does |
| --- | --- |
| `list_dir`, `glob` | List a folder, find files by pattern. |
| `read_file` | Read a text file with line numbers. A Jupyter notebook (`.ipynb`) is shown as text cells with their outputs. |
| `grep` | Regular-expression search. Optional `output_mode` (`content`, `files`, `count`), `context` lines, `case_insensitive` and `head_limit`. |
| `write_file`, `edit_file` | Create or overwrite a file; replace text in a file. |
| `multi_edit` | Several edits to one file, all or nothing, one undo step. |
| `bash` | Run a shell command. A `cd` stays in effect for the next call (never outside the folder). With `background: true` it starts a job and returns a job id. |
| `bash_output`, `bash_kill` | Read the output of a background job (the newest 256 KB are kept) or stop it. Jobs are stopped when the session ends, the folder changes or the server stops. |
| `web_fetch` | Download an http(s) page and return it as readable text (10 s connect and 30 s total timeout, 2 MB body cap, at most 5 redirects). |
| `web_search` | Best-effort web search through an HTML search page that needs no key. Returns title, URL and snippet; says so clearly when the search page blocks the request. |

The server lists its tools, with the permission category the page asks about (`read`, `edit`, `command`, `network`), at
`GET /api/studio/agent/tools`. `GET /api/studio/agent/changes` lists every file change of the session (id, path, `create` or `edit`,
time), and `POST /api/studio/agent/undo` takes `{"steps": n}` or `{"to": id}` to go back several changes at once (newest first).
The chat page can use the network tools without an agent session through `POST /api/studio/web/fetch` (`{url, max_chars?}`) and
`POST /api/studio/web/search` (`{query, limit?}`).

| Mode | What the agent does on its own |
| --- | --- |
| Ask (default) | Reads and searches. Asks before every file change and every command. |
| Auto-edit | Also changes files. Still asks before commands. |
| Plan | Only reads, then writes a plan. "Run the plan" switches to Ask. |

**Permissions.** "Always" on a question allows that kind of action for the rest of the session. Commands containing `;`, `&`, `|`,
redirections or substitutions, and risky programs (`rm`, `curl`, `sudo`, `git`, shells, ...), can only be allowed once at a time.
Network tools and MCP tools always ask first. Persistent rules per folder live in the *Permission rules* box of the side panel (or
`/allow` and `/deny`): one `category: pattern` per line with the categories `bash`, `edit`, `network`, `mcp` and `read`, for example
`bash: npm test*`, `edit: src/**`, `network: *`. Deny beats allow. An allow rule never covers a command with shell operators, and a
risky program is only covered by a rule that names it (`bash: git status*`, not `bash: *`).

**Sub-agents.** The model can hand a self-contained job (a broad search, an independent investigation) to a sub-agent with a fresh
context and gets back only its report. Several sub-agents started in one turn run in parallel, shown as nested cards with live status.
They are read-only unless the model asks for `full` mode (the usual questions still apply), have a step and context budget, and stop
when you stop the agent. All requests share the same system prompt and tool list, so the server can reuse its prompt cache.

**Long sessions.** When the conversation passes about 70% of the model's context, older turns are summarised by the model and replaced
(the task list and the changed files are kept, and a notice says so). `/compact [focus]` does it on demand. Sessions of each folder are
saved in the browser automatically; `/resume` or the side panel brings an earlier one back.

Commands (type `/` for the list): `/init` (the agent writes `FLLM.md`), `/compact`, `/resume`, `/rewind` (undo file changes back to a
chosen point), `/undo [n]`, `/plan`, `/ask`, `/edit`, `/diff`, `/status`, `/cd path`, `/allow`, `/deny`, `/rules`, `/clear`. Your own
commands are Markdown files in `.fllm/commands/` of the folder: `review.md` becomes `/review`, an optional `description:` front matter
line is shown in the list, and `$ARGUMENTS` (or `$1` ... `$9`) is replaced by what follows the command. `@file` adds a file to the
message. A `FLLM.md`, `AGENTS.md` or `CLAUDE.md` in the folder is read as project instructions.

Commands started in the background appear in the *Background jobs* panel, where you can read their output and stop them. Requests of
agent turns ask the server to constrain tool calls to valid ones (`tool_call_grammar`); servers that do not know it ignore it.

Safety: the agent tools are reachable from the machine fLLM runs on only, also when an API key is set. Paths cannot leave the folder
(also through symbolic links). The agent can still run any command you allow, so choose the folder with care. Set `FLLM_NO_AGENT=1` to
switch the agent routes off completely.

Network safety: `web_fetch`, `web_search` and the two web routes only speak http and https, refuse URLs with credentials, and refuse
targets on loopback, private, link-local (including cloud metadata) and other reserved addresses. The host name is resolved by the
server and the connection is pinned to the checked address; every redirect is checked again. They are reachable from the machine
fLLM runs on only, like the other agent routes. To let them reach a documentation server on your own network, start fLLM with
`FLLM_AGENT_ALLOW_PRIVATE_URLS=1`. `FLLM_NO_AGENT=1` switches them off too.

A small model often writes a command into the answer instead of calling the tool. Use a model with good tool calling for real work.

The pure logic of this page (rule matching, compaction, command expansion) has tests: `node rust/engine/src/webui/tests/code.test.js`.
The logic of assistants, memory and project retrieval is tested by `node rust/engine/src/webui/tests/projects.test.js`; the chat tool loop,
the memory tool and secret refusal by `tools.test.js`; system prompts, titles and the parsing of all page scripts by `prompts.test.js` (same folder).

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
server messages with refresh and copy), and under More: model details, text tools and ready-made examples for connecting Claude Code,
the OpenAI libraries and curl.

## Tools (MCP)

Settings has a Tools (MCP) page for Model Context Protocol servers: programs that offer tools (files, search, databases, ...) over their
standard input and output. Add a server with a name, a command (for example `npx`), arguments (one per line) and optional environment
variables. Each server shows its state (starting, ready, or failed with the reason), its tools and its last error output; you can restart,
edit or remove it. The list is saved in `mcp.json` next to `studio.json` in the common format
`{"mcpServers": {"name": {"command": "npx", "args": ["-y", "pkg"], "env": {"KEY": "value"}}}}`, so a file from another MCP client works as is.
Edits to the file are picked up the next time the page or the API reads it. An optional `"timeout_seconds"` per server sets the limit for
one tool call (default 60). A server that stops is started again on the next call, with a growing pause after repeated failures.

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

## Who may change things

Reading (status, hardware, models) is open like the rest of the API. Changing things through `/api/studio/*` (settings, downloads,
deletions, API keys, model switch, installing or removing the NPU runtime) needs an API key when one is configured. Without a key it is accepted only from the same machine, from
the page of this server, addressed by `localhost` or an IP address. Other websites in your browser cannot use it.
