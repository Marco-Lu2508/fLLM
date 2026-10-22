# Studio: the browser app

The Studio is the app that ships inside `fllm-rs`: a chat, a coding agent, a model manager and the server settings in one page. It needs
nothing but the binary and a browser; the page and all its scripts are embedded and nothing is loaded from the internet.

```sh
fllm-rs studio --model owner/repo:Q4_K_M     # first start: pick a model once
fllm-rs studio                                # later starts reuse the last model and open the page
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

## Settings

Appearance, the local API server (endpoint, rate limits, API keys, route tester), hardware (devices, CPU instruction set, memory
bandwidth, a kernel self test), the tokenizer and chat template tools, and ready-made snippets for Claude Code, the OpenAI SDKs and curl.

## Who may change things

Reading (status, hardware, models) is open like the rest of the API. Changing things through `/api/studio/*` (settings, downloads,
deletions, API keys, model switch) needs an API key when one is configured. Without a key it is accepted only from the same machine, from
the page of this server, addressed by `localhost` or an IP address. Other websites in your browser cannot use it.
