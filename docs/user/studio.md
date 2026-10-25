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
* **Use the web.** Switch on *Use the web* in the composer and each question is searched on the web first. The first three results
  are read, their best passages are added to the prompt with their source numbers, and the sources are listed under the answer. This
  uses the web tools of the server (`/api/studio/web/search` and `/api/studio/web/fetch`); if they are missing or blocked, the page
  says so and answers without the web. The pages are fetched by the server, so the computer running fLLM needs internet access.
* Presets store a system prompt together with sampling values. The sampling panel covers temperature, top-p/top-k/min-p, penalties,
  mirostat, DRY, a seed and constrained output (JSON object, JSON schema, GBNF grammar, regular expression).
* Chats can be pinned, renamed, duplicated, exported (Markdown or JSON), imported and searched by title and content. New chats get a
  short title from the model (switch off in Settings).
* Chats are stored in the browser (`localStorage`) and never leave it.

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

**Updates.** The Studio asks GitHub for a newer release once per start. If there is one, the Models page offers to download the
installer of this system and start it (Windows: the setup program, Linux: the AppImage or the package).
## Settings

Grouped as General, Appearance, Models (models folder, idle unload), Hardware (devices, a speed test of the loaded model, processor
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

## Who may change things

Reading (status, hardware, models) is open like the rest of the API. Changing things through `/api/studio/*` (settings, downloads,
deletions, API keys, model switch) needs an API key when one is configured. Without a key it is accepted only from the same machine, from
the page of this server, addressed by `localhost` or an IP address. Other websites in your browser cannot use it.
