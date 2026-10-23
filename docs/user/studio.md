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
* Attach images (for vision models) and text or code files by button, drag and drop or paste. A long paste becomes a file.
* Presets store a system prompt together with sampling values. The sampling panel covers temperature, top-p/top-k/min-p, penalties,
  mirostat, DRY, a seed and constrained output (JSON object, JSON schema, GBNF grammar, regular expression).
* Chats can be pinned, renamed, duplicated, exported (Markdown or JSON), imported and searched by title and content. New chats get a
  short title from the model (switch off in Settings).
* Chats are stored in the browser (`localStorage`) and never leave it.

## Code: the coding agent

Pick a folder and describe a task. The model works through tools that the server runs inside that folder (`list_dir`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`, `bash`, and more when the server offers them, such as `multi_edit`, background shell jobs,
`web_fetch` and `web_search`), plus tools of the page itself: a task list, `ask_user` and `spawn_agent`. Tools of configured MCP servers
are added too. Every step is shown as a card, file changes with a diff, long output cut with a "Show all" button.

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

**Your own GGUF files.** The *Add a model file* card takes the path of a .gguf file, or of a folder (its .gguf files are added). The files stay where they are and appear in the list of models on the computer; *Remove* only takes them off the list.

**Hugging Face account.** Private and gated models need an access token: paste one (read access is enough) into the Hugging Face card
of the Models page. It is checked with Hugging Face and kept in the same file the Hugging Face tools use (`~/.cache/huggingface/token`),
so downloads from the page, `fllm-rs pull` and `fllm-rs run` all use it. A `HF_TOKEN` variable wins over the saved token.

**Updates.** The Studio asks GitHub for a newer release once per start. If there is one, the Models page offers to download the
installer of this system and start it (Windows: the setup program, Linux: the AppImage or the package).
## Settings

Appearance, the local API server (endpoint, rate limits, API keys, route tester), hardware (devices, CPU instruction set, memory
bandwidth, a kernel self test), the tokenizer and chat template tools, and ready-made snippets for Claude Code, the OpenAI SDKs and curl.

## Who may change things

Reading (status, hardware, models) is open like the rest of the API. Changing things through `/api/studio/*` (settings, downloads,
deletions, API keys, model switch) needs an API key when one is configured. Without a key it is accepted only from the same machine, from
the page of this server, addressed by `localhost` or an IP address. Other websites in your browser cannot use it.
