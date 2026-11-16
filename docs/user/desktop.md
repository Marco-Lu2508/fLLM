# Desktop app

The desktop app is a native program written in Rust ([iced](https://iced.rs): no web view, no Chromium). It has the chat, the coding agent
(browser, computer use, sub-agents, skills, MCP), the model manager and the settings of the [Studio](studio.md), and it comes with the server.
Unpack it, start `fllm-app`, and pick a model on the first start. No terminal is needed.

## Download

One archive per system and build, attached to every [release](https://github.com/Marco-Lu2508/fLLM/releases):

| System | Standard (CPU, Vulkan, DirectX, Metal) | NVIDIA CUDA | AMD |
|---|---|---|---|
| Windows x86_64 | `fllm-desktop-windows-x86_64.zip` | `fllm-desktop-windows-x86_64-cuda.zip` | `fllm-desktop-windows-x86_64-hip.zip` |
| Linux x86_64 | `fllm-desktop-linux-x86_64.tar.gz` | `fllm-desktop-linux-x86_64-cuda.tar.gz` | `fllm-desktop-linux-x86_64-rocm.tar.gz` |
| macOS (Apple silicon) | `fllm-desktop-macos-aarch64.tar.gz` | | |

Every archive holds the app (`fllm-app`), the engine of the same release (`fllm-rs`) and a `README.txt`; unpacking it is the whole installation.
A `.sha256` and a signature (`.sigstore.json`, cosign keyless) lie beside each archive. The apps are not signed with a system certificate, so
Windows may show a warning on the first start and macOS asks you to confirm it (right click, Open). The CUDA and AMD builds exist when the
release has the matching engine; the AMD kernels are untested on AMD GPUs. Pascal and Volta cards (GTX 10 series, V100) are not in the CUDA
build: the standard one drives them through Vulkan.

How it works: when no server answers, the app starts `fllm-rs studio` (next to the app, or `FLLM_RS`, or on the PATH) on a free local port and
stops it when the window closes. A server that is already there (`FLLM_URL`, a Docker container, port 8080) is used as it is and left alone.
If the server cannot start, the screen says why, and `server.log` in the data folder has its output. Models are kept in the usual fLLM
model folder, shared with the command line; chats and settings of the app are in `~/.local/share/fllm-app` (Windows and macOS: the data folder of
the system; `FLLM_APP_DATA` moves it).

## Updates

*Settings, Desktop app, Updates* looks at the releases (at the start and on a button) and, when a newer one has the bundle of **your build** (a
standard app gets the standard bundle, a CUDA app the CUDA one, an AMD app the AMD one), downloads it, checks it against its `.sha256`, unpacks it
and offers *Restart to update*. A release without your build is not replaced by another one: the app says so and opens the release page.
*Update automatically* (on by default) only downloads; installing is always your button. The check finds damaged downloads; it does not prove who
made the file, because the cosign signature is not verified by the app yet.

## Features

* **One window:** starting the app again brings the open window to the front instead of starting a second one.
* **Tray icon and keeping running** (off by default), **start when I sign in** (off), a global **quick entry hotkey** (default Ctrl+Alt+Space) that
  opens a new chat with the cursor in the message field. Settings, Desktop app. Some Linux desktops (Wayland) do not allow global hotkeys, and on
  GNOME the tray needs an app-indicator extension.
* **Window place:** size and position are remembered. A maximized window comes back as a window of the screen's size.
* **Browser and computer use** of the coding agent: [studio.md](studio.md). The browser panel shows the page live and takes your clicks and keys;
  computer use asks before every action and stops with STOP, Esc three times or Ctrl+Alt+Shift+Esc.
* **Answers with pictures:** math in LaTeX, charts (```` ```chart ````), Mermaid diagrams and SVG are drawn in the answer (Settings, General,
  *Charts, diagrams and math in answers*).
* **Pasting:** an image on the clipboard is attached to the message, a very long text becomes a file.
* **Links to other sites** open in your default browser.
* **Light or dark:** follows the system until you choose a theme.

The behaviour of every part is compared with the web Studio's own scripts in `desktop/native/tests`; what is built, what was only tried without
a display and what is not done is listed in [desktop/native/PARITY.md](../../desktop/native/PARITY.md) and [desktop/native/README.md](../../desktop/native/README.md).

## Build it yourself

```sh
cargo build --release --locked        # in desktop/native; needs no system library besides the graphics driver
FLLM_RS=/path/to/fllm-rs ./target/release/fllm-app
```

The release workflow (`release.yml`) builds the engine archives for a tag and then the bundles (`native-bundle.yml`, also startable by hand with a tag).
