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

Pick a folder and describe a task. The model works through tools that the server runs inside that folder: `list_dir`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`, `bash`, and a task list. Every step is shown as a card, file changes with a diff.

| Mode | What the agent does on its own |
| --- | --- |
| Ask (default) | Reads and searches. Asks before every file change and every command. |
| Auto-edit | Also changes files. Still asks before commands. |
| Plan | Only reads, then writes a plan. "Run the plan" switches to Ask. |

"Always" on a question allows that kind of action for the rest of the session. Commands containing `;`, `&`, `|`, redirections or
substitutions, and risky programs (`rm`, `curl`, `sudo`, `git`, shells, ...), can only be allowed once at a time.

Commands: `/plan`, `/ask`, `/edit`, `/undo` (last file change), `/diff`, `/status`, `/cd path`, `/clear`. `@file` adds a file to the
message. A `FLLM.md`, `AGENTS.md` or `CLAUDE.md` in the folder is read as project instructions.

Safety: the agent tools are reachable from the machine fLLM runs on only, also when an API key is set. Paths cannot leave the folder
(also through symbolic links). The agent can still run any command you allow, so choose the folder with care. Set `FLLM_NO_AGENT=1` to
switch the agent routes off completely.

A small model often writes a command into the answer instead of calling the tool. Use a model with good tool calling for real work.

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
