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

## Features

* **One window:** starting the app again brings the open window to the front instead of starting a second server.
* **Tray icon:** a menu with Open fLLM, Restart server, Open log folder and Quit. Left click on the icon opens the window.
* **Keep running in the tray:** off by default. When on, closing the window hides it and the server keeps running; use Quit in the tray menu to end the app.
* **Start fLLM when I sign in:** off by default. When it is on together with the tray option, the app starts hidden in the tray.
* **Window place:** size and position are remembered between starts.
* **If the server stops,** the window shows the loading page with a message. Use Restart server in the tray menu (the log is in the log folder, `server.log`).

The options are in the Studio under Settings, Desktop app (the section only exists inside the app). The same page has the buttons
Open log folder and Restart server.

Where things are kept: the options are in `settings.json` in the config folder of the app (Windows `%APPDATA%\io.github.marco-lu2508.fllm`,
Linux `~/.config/io.github.marco-lu2508.fllm`, macOS `~/Library/Application Support/io.github.marco-lu2508.fllm`); the page shows the exact folders. The start at sign-in is
registered with the system (a Run entry on Windows, an autostart `.desktop` file on Linux, a launch agent on macOS), and the window place is kept by the window-state plugin in the same config folder. On
Linux the tray icon needs an app-indicator extension on some desktops (for example GNOME).

Security: the page of the local server may call exactly five commands of the app (read the options, set the tray option, set the
start at sign-in, open the log folder, restart the server), and only when it is shown from `http://127.0.0.1` in the app window (a Tauri 2
remote capability, `desktop/src-tauri/capabilities/studio-page.json`). No file, shell or network access of the app is exposed.

The source is in `desktop/` (Tauri 2). To build it yourself, build `fllm-rs`, copy it to
`desktop/src-tauri/binaries/fllm-rs-<target triple>`, then run `cargo tauri icon ../app-icon.png` and `cargo tauri build` in
`desktop/src-tauri`.