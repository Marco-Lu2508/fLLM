# Desktop app

The desktop app is the [Studio](studio.md) in its own window, with the server built in. Install it like any other program, start it
from the menu, and pick a model on the first start. No terminal is needed.

* **Windows:** run the setup (`.exe`) or the `.msi`.
* **Linux:** install the `.deb` or `.rpm`, or make the `.AppImage` executable and start it.
* **macOS:** planned.

The installers are attached to every [release](https://github.com/Marco-Lu2508/fLLM/releases); the `desktop` workflow builds them for every change (Actions, then the artifacts of a run). They are not signed yet, so
Windows may show a warning on the first start.

How it works: the app starts the `fllm-rs` program that is installed next to it (`fllm-rs studio` on a fixed local port, only reachable from
your computer) and shows its page. Closing the app stops the server. Models, chats and settings are kept between starts; the models are
stored in the usual fLLM model folder, shared with the command line.

The source is in `desktop/` (Tauri 2). To build it yourself, build `fllm-rs`, copy it to
`desktop/src-tauri/binaries/fllm-rs-<target triple>`, then run `cargo tauri icon ../app-icon.png` and `cargo tauri build` in
`desktop/src-tauri`.