# Changelog

All notable changes to fLLM are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project aims to follow [Semantic Versioning](https://semver.org/) once it leaves the 0.x series (0.x releases may change
behaviour between versions).

## [Unreleased]

### Added
- Gemma 4 on the graphics chip: the device pass now runs the per-layer embeddings, the layer output scale, the value norm, layers whose values are their keys and layers that read an earlier layer's cache (the per-layer weights are uploaded as Q8_0). Measured on an Intel Core Ultra 5 235U (Vulkan) with Gemma 4 E4B Q4_K_M: answers at 12.7 tokens/s (llama.cpp 12.6), prompts at 115 tokens/s (llama.cpp 167); before, the model failed on the device.
- Vulkan prompt speed: the quantized matrix kernel computes 8 x 4 outputs per invocation on a 128 x 64 tile and its loaders read neighbouring bytes of a row (`FLLM_VK_MM_TM`, `FLLM_VK_MM_KS` for tuning), and the Gated DeltaNet recurrence runs over 16 value columns per workgroup with subgroup shuffles instead of one workgroup and a dozen barriers per token and head (`FLLM_VK_DELTA_COLS=0` keeps the old shader). Qwen3.5-2B Q4_K_M, 505-token prompt on the same laptop: 351 tokens/s (0.1.8: 245, llama.cpp: 374).
- Vulkan: a prompt attention kernel with 64 query rows per workgroup on devices that allow 512 invocations (`attn_prefill_*_r64`; `FLLM_VK_ATTN_ROWS=16` keeps the 16-row kernel): +7 % prompt speed at 2000 tokens on Intel Xe-LPG.
- The scheduler now gives running answers their token before new prompt chunks (`--chunked-prefill-policy decode-first` is the default): with several users the first ones get their answer streamed while the later prompts are still read, instead of everyone waiting until the last prompt is done (4 users with 2000-token prompts on Vulkan: first tokens after 9 to 32 s instead of all after 31 s). `prefill-first` restores the old order.
- `scripts/bench-load.ps1`: fLLM, llama.cpp and Ollama under load (prompts of several lengths, several users at once, total and per-user speed, time to the first token).
- `/skill` in the chat and on the Code page: the model writes a skill from the conversation and shows it for approval.
- `FLLM_VK_PROFILE=N` prints the shader table every N command buffers, summed over all recorders of the process.

### Fixed
- An explicitly chosen device that cannot run a model (for example Gemma 3n on Vulkan) no longer stops the server: the model runs on the CPU and the reason is logged.
- Gemma 4 on Vulkan and the NPU failed with "rope width 512 invalid for head width 256": the device pass used one rotary width for all layers.
- Windows: the low-power efficiency cores of Intel Core Ultra are told from the other efficiency cores by their cache (Windows gives them the same efficiency class), so `--threads auto` picks 8 instead of 9 threads on a Core Ultra 5 235U; an explicit `--threads n` gets the same core placement as the automatic count (a 10-thread run answered at 9 tokens/s without it, 24 with it).
- The code page sent `reasoning_effort: high` whenever thinking was on; it now sends what the loaded model understands for the chosen level.
- KV block copies (several choices per request, swap preemption) work with KV pools whose layers have their own sizes.

## [0.1.8] - 2026-10-26

### Added
- Studio: Settings "Skills and schedules": skills (Markdown procedures per user and per workspace, a `skill` tool and a propose-a-skill helper), scheduled tasks that the server runs on a clock (chat answers or read-only agent runs, last results kept, optional desktop notification), and full-text search of past chats and code sessions with a persistent copy on the computer. See `docs/changes/personal-agent.md`.
- Thinking only where the model can think: the status route and `/v1/models` report `capabilities.thinking` (`control` none / toggle / effort / always, the template variable, the levels and the default), read from the chat template, the reasoning parser and the architecture. The Studio shows its Thinking chip only for models that can think, with the levels the model has (Off / Low / Medium / High for models with a switch, Low / Medium / High for effort models such as gpt-oss), remembered per chat.
- Thinking budget: `thinking_budget` (tokens) in chat requests, and `reasoning_effort` low / medium on models with a thinking switch (512 / 2048 tokens), end the reasoning block after that many tokens: the server appends the closing tag and the model writes its answer, in streams and normal replies alike ([docs/server-api.md](docs/server-api.md#thinking)).
- Speculative decoding is chosen in the load dialog (Models, Advanced: off, prompt lookup, draft model, multi-token prediction) and remembered per model; impossible choices (models with recurrent layers, a model without MTP heads, MTP on a backend without it, a draft model that is larger or has another vocabulary) are refused with a message. The switch in the sampling panel is gone: it only asked the server to skip speculation per request, and the server already does what the start option says.
- Studio chat that decides for itself: *Use the web* became *Allow web access* and offers the model `web_search` and `web_fetch` as tools instead of searching every question (compact "Searched" / "Read" cards, sources under the answer, at most 5 rounds, Stop works, results sized to the context, `tool_call_grammar`). A default system prompt in a compact and a full size with adjustments per model family (date and time, language, honesty, tool rules; shown and overridable or extendable in Settings > Personalization and per assistant). A `memory` tool the model manages (add, update, remove, list; "Saved to memory" cards with Undo; secrets refused; per-chat and global switch) with a server-side copy (`GET`/`PUT /api/studio/memory`, `memory.json`, this machine only), and a `search_chats` tool for earlier chats (asked once per chat). Chat titles use a short prompt that works with thinking models.
- Studio, Code: a larger step towards a full coding agent ([docs/user/studio.md](docs/user/studio.md)). New tools: `patch` (unified diff over several files, all or nothing), `diagnostics` (syntax / build check with `cargo check`, `tsc`, `go vet`, `py_compile`, `node --check`; optional "check my edits automatically"), `memory`, `todo_read`, and `task` with sub-agent types `explore` (read only) and `general`. `edit_file` tolerates indentation differences, keeps CRLF files CRLF and, when it fails, names the closest text with line numbers; `read_file` suggests similar names for a missing file and refuses binary files and images clearly; `list_dir` shows a tree (`depth`); `grep` has `literal` and `word`; `glob`, `grep` and the tree respect `.gitignore`.
- Studio, Code: separate system prompts for the build, plan and sub-agents, a compact form for small models and contexts, a note with an example call per model family (Qwen, Llama, Mistral, Gemma, gpt-oss), an environment block (folder, system, shell, git state, date), project instructions from `AGENTS.md` / `FLLM.md` / `CLAUDE.md` of the folder and its parents, and a persistent memory per folder and per user that the model maintains itself (`.fllm/memory.md`, `agent-memory.md`; panel in the side bar, `/remember`, `/memory`).
- Studio, Code: sessions can be forked from any message, reverted to a message (conversation and file changes, with an undo of the revert), renamed, deleted, searched and exported as Markdown; the model titles them. Build / plan agent switch (Alt+B) and a model switch in the header, a message queue while the agent works, `!command` shell escape whose output the model sees, fuzzy `@` file search over the folder, a permission table (allow / ask / deny per kind of action and agent) next to the pattern rules, and new routes `/api/studio/agent/{cancel,files,memory,instructions}`; `POST /api/studio/agent/undo` takes `{"redo": n}`.
- Vulkan: a new tiled matrix-matrix kernel for prompt processing with quantized weights (`mmq_<format>_i`, used from 32 rows up on GPUs with the int8 dot product), and the shaders now emit the dot-product instruction themselves when the shader compiler is too old to do it (the Windows release build was affected). Compiled and unit tested, not yet measured on a GPU; `FLLM_VK_MM=0` switches the kernel off, `FLLM_VK_MM_BM` / `FLLM_VK_MM_BN` tune the tile ([docs/changes/vulkan-prefill.md](docs/changes/vulkan-prefill.md)).
- NPU: a load option "What the NPU computes" (`--npu-mode all`): besides the long prompt blocks the NPU then also computes the single rows of the answer, the way later and stronger NPUs will be used. Measured on an Intel Core Ultra NPU with Qwen3.5-2B Q4_K_M: the answer runs at about 5 tokens/s this way (after a one-time compile of a few minutes that OpenVINO keeps in its cache), about as fast as the processor and well below the graphics chip.
- `--threads auto` (the new default): the CPU thread count follows the processor's topology (one thread per physical core; on hybrid processors all fast cores plus three quarters of the efficiency cores, never the low-power ones) and is confirmed by a measurement of about a second on the real weights at start-up, remembered per CPU, weight type and matrix shape (`FLLM_NO_AUTOTUNE=1` skips it). The old default of one thread per logical processor was less than half as fast for answers as 8 threads on a 14-thread hybrid laptop. Prompts keep all logical processors when all cores are alike; on Windows hybrid processors the process leaves power throttling and stays on the chosen cores (`FLLM_NO_PIN=1` disables that). `fllm-rs devices` and `/api/studio/hardware` report the topology and the thread count ([docs/cpu-kernels.md](docs/cpu-kernels.md#thread-count)).

### Fixed
- Studio, Code: a command that hit its time limit left its child processes running and the call waited on their output pipes; the whole process tree is ended now, also when Stop is pressed. Command output keeps stdout and stderr in order and keeps its end when it is cut. Stopping the agent now leaves valid history (unanswered tool calls get an "interrupted" result, two user messages in a row are separated) and runs one after another instead of overlapping. A failed edit no longer leaves an entry in the change log. On Windows, an absolute path with a different drive-letter case is accepted, glob patterns ignore case and multi-line commands are joined for `cmd`.

## [0.1.7] - 2026-10-26

### Fixed
- Desktop app 0.1.6: the window stopped responding as soon as the page asked for the desktop settings (the settings page, and at start). The command locked its settings twice in one expression and waited for itself; it now reads them under one lock.

### Added
- AMD GPUs on Windows (experimental, never run on an AMD GPU): `scripts/build-windows.ps1 -Hip` builds the HIP kernels with the ROCm / HIP SDK for Windows, the HIP runtime DLL is delay-loaded and searched at run time (the binary starts without it and falls back to Vulkan / the CPU), GPUs the build has no code for are no longer listed, and the CI workflow `windows-hip-package` and an optional `windows-hip` release job produce `fllm-windows-x86_64-msvc-hip.zip` ([docs/changes/hip.md](docs/changes/hip.md)).
- Desktop app: engine variants. Releases also carry `fLLM-cuda` installers for Windows and Linux with a second server program (CUDA + Vulkan) next to the base one (CPU + Vulkan + DirectX); the app picks the CUDA one when an NVIDIA driver and GPU are found, falls back to the base one if it stops right after the start, and can be told otherwise in Settings, Desktop app, or with `FLLM_DESKTOP_VARIANT` ([docs/user/desktop.md](docs/user/desktop.md)).

### Fixed
- Release workflow: the `windows-cuda` job no longer clones FlashInfer, whose headers do not compile with the MSVC host compiler.
- Intel NPU from the app (experimental, test level L0: not run on NPU hardware): the `npu-openvino` feature is now in the default features and loads the OpenVINO runtime at run time, so the standard binary can use an NPU without a rebuild. `fllm-rs devices` and `GET /api/studio/hardware` list an NPU the system finds even when the runtime is missing, and AMD / Qualcomm NPUs as "detected, not supported yet". New routes `GET /api/studio/npu`, `POST` and `DELETE /api/studio/npu/setup` install (pinned OpenVINO 2026.4.1, checked size and SHA-256, official host only, resumable, cancellable, no Python needed) or remove the runtime; `fllm-rs npu-setup [--remove]` uses the same code. Settings, Hardware has an NPU card, and the load options have a *Run on* choice that lists the NPU when it is ready. `--device auto` still never picks an NPU. An NPU that cannot take a model now falls back to the processor with a readable reason instead of ending the server ([docs/backends.md](docs/backends.md) section 6).

## [0.1.6] - 2026-10-25

### Added
- Studio chat: a *Research* mode that plans search queries, reads web pages in parallel, takes notes within the model's context, reflects for one more round and writes a cited report, with a collapsible research log and a Stop button that cancels every request.
- Studio: Model providers in Settings. Use models of cloud services and other servers (OpenAI-compatible services, Anthropic, Ollama, LM Studio, vLLM) in the chat next to the local model; they appear in the model menu as "Provider · model". Keys stay in `providers.json` on this computer and never reach the page; the routes (`/api/studio/providers/*`) work only from this machine, refuse internal addresses unless a provider is marked local, and translate Anthropic requests and streams to the OpenAI format ([docs/user/studio.md](docs/user/studio.md)).
- Studio: assistants (the old presets, now with icon, color, default model, suggestions and JSON import/export), projects with instructions and a searchable document knowledge base, personalization with memory ("remember that ..."), a `/` prompt library, and chat organization (bulk select, drag to pin, archive).
- Studio: everyday features of a chat app. Dictation with the microphone button (browser speech recognition, hidden where unavailable), reading answers aloud (voice, speed, pitch, optional automatic reading), rebindable keyboard shortcuts with a panel on `?` / Ctrl+`/`, system notifications and an optional tone when a long answer or an agent run finishes in the background, and a welcome tour on the first start (Settings, General brings it back).
- Desktop app: a global quick-entry hotkey (default Ctrl+Alt+Space, configurable under Settings, Desktop app) that brings the window to the front with a new chat.
- Studio chat: a preview panel with versions for HTML, SVG, Markdown and CSV code blocks (sandboxed, no network), a sandboxed JavaScript runner for code blocks, copy-as-Markdown, thumbs up/down with a note, a *Continue* button for answers cut by the token limit, response styles, temporary chats that are never saved, a small built-in LaTeX renderer and scrollable tables. The Studio now serves its code-runner worker at `/studio/runner.js`.

## [0.1.5] - 2026-10-25

### Fixed
- Desktop app: switching the model in the app no longer ends with an empty window and a stopped server. The app closed the pipe that keeps the new server alive as soon as the old server ended; it now waits for the new one. Models such as Qwen3.5 whose chat template counts with `ns.x = ns.x + ...` load again.
- Gemma 4 GGUF files from llama.cpp (such as `unsloth/gemma-4-E4B-it-GGUF`) now load and answer correctly on the CPU: a new tokenizer type `gemma4`, a chat-template check that accepts `set ns.x = ns.x + ...`, and the attention scale 1.0 these files rely on. `gemma4:e4b` is in the model catalog.

## [0.1.4] - 2026-10-23

### Added
- Studio: chat with documents and the web. PDF (text layer), DOCX, XLSX, EPUB and HTML files can be attached (`POST /api/studio/extract`, text extraction in plain Rust without new dependencies); documents longer than the context are cut into passages and the ones that match the question (BM25 in the browser) go into the prompt with source labels, several documents per chat. A "Use the web" switch searches the question, reads the first results and lists the sources under the answer ([docs/user/studio.md](docs/user/studio.md)).
- Studio coding agent (server side): `multi_edit`, richer `grep` (output modes, context, case folding, limits), notebook reading, background shell jobs (`bash_output`, `bash_kill`) with a working directory that persists between commands, `web_fetch` and `web_search` with private-address protection (`FLLM_AGENT_ALLOW_PRIVATE_URLS=1` lifts it), the routes `GET /api/studio/agent/tools`, `GET /api/studio/agent/changes`, `POST /api/studio/web/fetch` and `/search`, and undo by `steps` or up to a change id.
- Server: `tool_call_grammar: true` on `/v1/chat/completions` limits a reply with tools to plain text or valid tool calls in the model's own call format (Hermes/Qwen, Qwen3-Coder, Llama 3, Mistral, Granite 3) with declared names and schema-conforming arguments; also used for `tool_choice` `required` and named functions. Docs: "Agents and parallel requests" in the memory guide, and `scripts/agent_load_test.py`.
- Studio Code: sub-agents that run in parallel (nested cards, read-only by default), context compaction (automatic at about 70 percent and /compact), ask_user questions, /init, custom slash commands from `.fllm/commands/*.md`, /rewind and /resume with automatic per-folder sessions, persistent allow/deny permission rules, a background jobs panel, MCP tools, multi_edit and web tools support, constrained tool-call output, and nicer diffs and long-output handling.
- Studio: Hugging Face sign-in with an access token (private and gated models), a check for a newer release with download and start of the installer, and curated model suggestions that fit the memory (and the processor, without a discrete graphics card), your own GGUF files (a path or a folder) in the model list, and a hardware page that names the device in use.
- Studio model management: advanced load options (context length, GPU layers, threads, conversation memory type, parallel chats) remembered per model and shown on the loaded-model card, Unload and "unload after N idle minutes", native file and folder dialogs (Browse), a download queue with pause, resume, cancel, speed and time left that survives page reloads, a setting for the models folder, model card summaries and quantization and fit hints in the Hugging Face file list, a Logs page, a speed test, and plainer wording across Settings.
- Studio: Tools (MCP) in Settings. The server starts Model Context Protocol servers (stdio) from `mcp.json`, lists their tools and calls them (`/api/studio/mcp/*`).

- Desktop app: one window only (a second start focuses the first), a tray icon with a menu, optional keep running in the tray and start at sign-in, remembered window size and position, a clear message when the server stops, and a Desktop app section in the Studio settings (open log folder, restart server).

### Fixed
- Studio: a model that cannot be loaded (for example a Gemma 4 file) no longer ends the server and leaves an empty window: the Models page opens and shows the reason. Models the engine cannot read are turned down before a switch. The desktop app waits for a slow model load and shows how long it has been loading. The list of downloaded models is no longer taken from the browser cache. The Gemma 4 entries left the catalog (not supported yet).

## [0.1.3] - 2026-10-23

### Fixed
- Release: the CUDA archive is built with the CUDA backend (a global feature setting had overridden it), and the ROCm build script is executable. A tag push removes releases left over from an earlier run of the same tag.
- Studio: a model download runs in the background of the app and resumes after a dropped connection (it was cancelled after 10 minutes, and a download took at most one hour); the models page shows the computer, suggests models that run well in its memory and hides the ones that are too large; memory for integrated and unreported GPUs comes from the computer. The desktop app keeps a `server.log`.

## [0.1.2] - 2026-10-22

### Added
- Studio, the built-in browser app (`fllm-rs studio`, or `GET /` of any server; [docs/user/studio.md](docs/user/studio.md)): chat with branching message history, Markdown and code highlighting, file and image attachments, presets and chat management; a coding agent that reads, searches and changes files and runs commands inside one folder, with diffs, permission questions (ask, auto-edit, plan), a task list and `@file`; a model manager with Hugging Face search, download progress and switching the loaded model (the server restarts itself); settings, API keys, hardware and tool pages. Backed by `/api/studio/*`. Changes through these routes are accepted without an API key only from the same machine, from the page of this server, addressed by `localhost` or an IP address; the agent tools never leave the machine, also with a key, and `FLLM_NO_AGENT=1` switches them off. Generated keys come from the operating system random source. `fllm-rs studio` also starts without a model and lets you pick and download the first one in the page. A desktop app (Tauri, `desktop/`) shows the Studio in its own window with the server built in; installers for Windows and Linux are attached to the release ([docs/user/desktop.md](docs/user/desktop.md)).
- `install.sh --variant auto|rocm`: the installer picks cuda, rocm, gnu (Vulkan) or cpu from the hardware; `release.yml` builds the optional `fllm-linux-x86_64-rocm.tar.gz` (HIP kernels, untested on AMD GPUs).

### Fixed
- Vulkan on integrated GPUs with a small dedicated heap (AMD APUs): a full device-local heap falls back to host-visible memory instead of failing with "Out of memory".
- Vulkan: conversion to binary16 rounds to nearest even on every driver (AMD rounded toward zero).
- `serve` plans the memory of an integrated GPU with the free system memory.

## [0.1.1] - 2026-10-15

### Added
- Checkpoints are converted to GGUF on first use: `run`, `serve` and `pull` accept a Hugging Face repository or a local directory with
  safetensors / PyTorch weights and keep the converted file in the model cache.
- `convert` imports NVIDIA ModelOpt NVFP4 checkpoints, compressed-tensors int4/int8 and block-wise FP8 checkpoints.
- `fllm-rs selftest-cpu` with kernel tiers (`--tier`, `--tiers`, `--full`), Q4Z/Q4ZP SIMD kernels and a per-layer dump.
- NVFP4 activation scale (the checkpoint's static `input_scale`), round-to-even E2M1 conversion and a tile table; a GPU perplexity and
  quality script (`scripts/colab/nvfp4_quality.py`).
- Colab scripts for A/B measurements of the new defaults (`scripts/colab/ab_matrix.py`, `cpu_confirm.py`).
- `SECURITY.md`.

### Changed
- CUDA decode steps are recorded as CUDA graphs and use fused decode kernels by default; the prompt pass no longer does per-layer host
  work and uses fused add+norm and FP16 copies. Each has a one-line switch back (see `docs/configuration.md`).
- The `avx512` Cargo feature is on by default; the AVX-512 and AMX kernels are chosen at run time from CPUID, so the binary still runs
  on any x86-64 CPU.
- README rewritten with the real feature set, formats, APIs, status table and a documentation index.

### Fixed
- NVFP4 self test w4a4 failures are deterministic and readable; weight upload ordering; IEEE division in the quantizer.
- Waiting for a concurrent conversion of the same checkpoint; compressed-tensors FP8 configs without `config_groups`.
- Windows: the developer-shell launcher takes `amd64` as architecture.

## [0.1.0] - 2026-10-14

### Added
- First public release of the pure-Rust inference engine: GGUF loading, CPU kernels (AVX2, AVX-VNNI, AVX-512, NEON), CUDA backend,
  wgpu GPU path (Vulkan, Metal, DirectX 12) and native Vulkan, Metal, HIP, SYCL, NPU and TPU backends in various states of
  verification (see the Status table in the README).
- OpenAI-compatible server with Anthropic, Ollama, llama.cpp, SGLang and TGI compatible routes, tool calling, structured output,
  speculative decoding, LoRA merge at load time, embeddings and encoder (BERT-style) models, SmolVLM image input.
- Quantization formats of the GGUF family (including K-quants, IQ formats, MXFP4, NVFP4) plus the Q4Z formats for AWQ and GPTQ checkpoints.
- Command line: `run`, `serve`, `chat`, `bench`, `convert`, `pull`, `devices`, `formats`, `inspect`, `selftest-cpu`, `selftest-gpu`.
- GitHub Actions for checks, platform builds, release binaries and container images; install scripts that resolve GitHub releases.
- Docker images (CPU for amd64 and arm64, CUDA for amd64).

### Fixed
- Misaligned vector loads in the AVX-512 weight-panel tile (Q4_1/Q5_1 prompts).
- Metal kernels `fllm_attn_merge_f32` and `fllm_kv_write_q8` thread-index declarations.
- Vulkan: clippy `unnecessary_cast` where `c_char` is `u8` (aarch64 Linux).

[Unreleased]: https://github.com/Marco-Lu2508/fLLM/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Marco-Lu2508/fLLM/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Marco-Lu2508/fLLM/releases/tag/v0.1.0
