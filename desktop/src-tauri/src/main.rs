//! The fLLM desktop app: a window around the Studio. The server is the `fllm-rs` program that is installed next to this one; the window
//! shows it as soon as it answers. When the app ends, the pipe on the standard input of the server closes and the server stops with it.
//! Around the window: one instance only, a tray icon, an optional start at sign-in, the remembered window place, and a few commands
//! the Studio page may call (see `capabilities/studio-page.json`).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod variants;

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, RunEvent, State, Url, WindowEvent};
use tauri_plugin_autostart::{MacosLauncher, ManagerExt};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_window_state::StateFlags;

/// The page keeps its chats per address, so the app prefers the same port every time.
const FIRST_PORT: u16 = 47821;
/// Given to the program when the system starts it at sign-in.
const AUTOSTART_ARG: &str = "--autostarted";

struct Server {
    child: Child,
    _pipe: Option<ChildStdin>,
}

/// What the app keeps in `settings.json` in its config folder.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Settings {
    keep_in_tray: bool,
    /// The global hotkey that brings the window to the front with a new chat ("" = off).
    quick_entry: String,
    /// Which engine variant to start: "auto" or the name of a variant (see `variants.rs`).
    variant: String,
}

const DEFAULT_QUICK_ENTRY: &str = "Ctrl+Alt+Space";

impl Default for Settings {
    fn default() -> Self {
        Settings { keep_in_tray: false, quick_entry: DEFAULT_QUICK_ENTRY.to_owned(), variant: variants::AUTO.to_owned() }
    }
}

/// The variant the server runs now, for the settings page.
#[derive(Default, Clone)]
struct Active {
    name: String,
    label: String,
    /// Why this variant runs when it is not the plain choice (a fallback, an ignored setting).
    note: String,
}

struct Shell {
    port: u16,
    /// The address of the loading page (taken from the window at the start).
    loading_url: Mutex<Option<Url>>,
    running: Mutex<Option<Server>>,
    /// Counts the starts of the server, so an older wait or watch ends when a newer start has begun.
    generation: AtomicU64,
    settings: Mutex<Settings>,
    active: Mutex<Active>,
}

fn free_port() -> u16 {
    (FIRST_PORT..FIRST_PORT + 40)
        .find(|port| TcpListener::bind(("127.0.0.1", *port)).is_ok())
        .unwrap_or(FIRST_PORT)
}

fn settings_file(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|dir| dir.join("settings.json"))
}

fn load_settings(app: &AppHandle) -> Settings {
    settings_file(app)
        .and_then(|file| std::fs::read(file).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_settings(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let file = settings_file(app).ok_or("There is no config folder.")?;
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(file, text).map_err(|e| e.to_string())
}

/// The server writes what it prints into `server.log` in the log folder of the app, so a problem can be looked at afterwards. The app
/// starts the file with a line of its own that names the variant (`append`: a second start after a fallback keeps the first one's output).
fn server_log(dir: Option<PathBuf>, header: &str, append: bool) -> (Stdio, Stdio) {
    let file = dir.and_then(|dir| {
        std::fs::create_dir_all(&dir).ok()?;
        let path = dir.join("server.log");
        let mut file = if append { std::fs::OpenOptions::new().create(true).append(true).open(path).ok()? } else { std::fs::File::create(path).ok()? };
        let _ = writeln!(file, "[desktop] {header}");
        Some(file)
    });
    match file.and_then(|f| f.try_clone().ok().map(|clone| (f, clone))) {
        Some((out, err)) => (Stdio::from(out), Stdio::from(err)),
        None => (Stdio::null(), Stdio::null()),
    }
}

fn start_server(port: u16, variant: &variants::Variant, log_dir: Option<PathBuf>, header: &str, append: bool) -> Result<Server, String> {
    let mut command = Command::new(variants::sidecar_path(variant));
    let (out, err) = server_log(log_dir, header, append);
    command
        .args(["studio", "--no-open", "--host", "127.0.0.1", "--port", &port.to_string()])
        .env("FLLM_EXIT_ON_STDIN_EOF", "1")
        .stdin(Stdio::piped())
        .stdout(out)
        .stderr(err);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut command, 0x0800_0000); // no console window
    let mut child = command.spawn().map_err(|e| format!("The server could not be started: {e}"))?;
    let pipe = child.stdin.take();
    Ok(Server { child, _pipe: pipe })
}

fn stop_server(app: &AppHandle) {
    if let Some(mut server) = app.state::<Shell>().running.lock().unwrap().take() {
        drop(server._pipe.take());
        let _ = server.child.kill();
        let _ = server.child.wait();
    }
}

/// Shows the main window again (tray, second start of the app).
fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// The quick-entry hotkey: bring the window to the front and ask the page for a new chat with the cursor in the composer. Before the
/// page has loaded (loading page) the call finds nothing to run and does nothing more than show the window.
fn quick_entry(app: &AppHandle) {
    show_main(app);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.eval("if (window.fllmQuickEntry) window.fllmQuickEntry();");
    }
}

/// Registers `keys` (for example "Ctrl+Alt+Space") as the only global hotkey; an empty text switches the hotkey off.
fn register_quick_entry(app: &AppHandle, keys: &str) -> Result<(), String> {
    let hotkeys = app.global_shortcut();
    let _ = hotkeys.unregister_all();
    if keys.is_empty() {
        return Ok(());
    }
    let shortcut: Shortcut = keys.parse().map_err(|_| format!("\"{keys}\" is not a valid key combination."))?;
    hotkeys.register(shortcut).map_err(|e| format!("The hotkey could not be registered (another program may use it): {e}"))
}

/// Shows the loading page with a message; the page reads the message from the fragment of its address.
fn show_loading(app: &AppHandle, message: &str) {
    let Some(window) = app.get_webview_window("main") else { return };
    let Some(mut url) = app.state::<Shell>().loading_url.lock().unwrap().clone() else { return };
    let encoded: String = message
        .bytes()
        .map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect();
    url.set_fragment(Some(&encoded));
    let _ = window.navigate(url);
}

/// Waits until the server answers. A model is loaded before the server listens, which takes a while for a big one, so the wait is long;
/// it ends early when the server program has stopped or a newer start has begun.
fn wait_until_answers(app: &AppHandle, generation: u64) -> Result<(), Wait> {
    let shell = app.state::<Shell>();
    let address = SocketAddr::from(([127, 0, 0, 1], shell.port));
    let started = Instant::now();
    let mut last_note = Duration::ZERO;
    while started.elapsed() < Duration::from_secs(30 * 60) {
        if shell.generation.load(Ordering::SeqCst) != generation {
            return Err(Wait::Superseded);
        }
        if TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_ok() {
            return Ok(());
        }
        if let Some(server) = shell.running.lock().unwrap().as_mut() {
            if let Ok(Some(status)) = server.child.try_wait() {
                return Err(Wait::Stopped(status.to_string()));
            }
        }
        if started.elapsed() - last_note >= Duration::from_secs(2) && started.elapsed() > Duration::from_secs(6) {
            last_note = started.elapsed();
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.eval(&format!(
                    "var n=document.getElementById('note'); if(n) n.textContent='Loading the last model ... {} s'",
                    started.elapsed().as_secs()
                ));
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err(Wait::Failed("The server did not start in time.".to_owned()))
}

/// Why the wait for the server ended without an answer.
enum Wait {
    /// A newer start has begun; nothing more to do.
    Superseded,
    /// The server program ended before it answered (the text is its exit status). A missing driver library shows up like this.
    Stopped(String),
    Failed(String),
}

fn stopped_message(what: &str) -> String {
    format!("{what} Choose \"Restart server\" in the menu of the tray icon, or start the app again. The log is in the log folder of the app (server.log).")
}

/// Starts (or restarts) the server, then shows its page once it answers, and shows the loading page again if the server stops.
fn launch(app: &AppHandle) {
    let shell = app.state::<Shell>();
    let generation = shell.generation.fetch_add(1, Ordering::SeqCst) + 1;
    stop_server(app);
    let port = shell.port;
    // which variant: the environment, then the saved setting, then the hardware that is present
    let saved = shell.settings.lock().unwrap().variant.clone();
    let (request, ignored) = variants::request_from(std::env::var("FLLM_DESKTOP_VARIANT").ok().as_deref(), &saved);
    let (first, chosen_note) = variants::choose(request, &|v| variants::sidecar_path(v).is_file(), &variants::Probe::detect());
    let mut notice = [ignored, chosen_note].into_iter().flatten().collect::<Vec<_>>().join(" ");
    let log_dir = app.path().app_log_dir().ok();
    let app = app.clone();
    std::thread::spawn(move || {
        let shell = app.state::<Shell>();
        let mut variant = first;
        let mut append = false;
        let outcome = loop {
            *shell.active.lock().unwrap() = Active { name: variant.name.to_owned(), label: variant.label.to_owned(), note: notice.clone() };
            let header = if notice.is_empty() { format!("variant {} ({})", variant.name, variant.label) } else { format!("variant {} ({}): {notice}", variant.name, variant.label) };
            let result = match start_server(port, variant, log_dir.clone(), &header, append) {
                Ok(server) => {
                    *shell.running.lock().unwrap() = Some(server);
                    wait_until_answers(&app, generation)
                }
                Err(e) => Err(Wait::Stopped(e)),
            };
            match result {
                Ok(()) => break Ok(()),
                Err(Wait::Superseded) => return,
                Err(Wait::Failed(message)) => break Err(message),
                Err(Wait::Stopped(reason)) => {
                    if shell.generation.load(Ordering::SeqCst) != generation {
                        return;
                    }
                    // a variant that stops right after the start (driver missing, library error): the base variant takes over, once
                    let Some(base) = variants::fallback(variant) else {
                        break Err(stopped_message(&format!("The server stopped right after the start ({reason}).")));
                    };
                    notice = format!("The {} variant stopped right after the start ({reason}); the {} variant runs instead.", variant.label, base.label);
                    show_loading(&app, &notice);
                    stop_server(&app);
                    variant = base;
                    append = true;
                }
            }
        };
        if shell.generation.load(Ordering::SeqCst) != generation {
            return;
        }
        if let Err(message) = outcome {
            show_loading(&app, &message);
            return;
        }
        if let Some(window) = app.get_webview_window("main") {
            if let Ok(url) = format!("http://127.0.0.1:{port}/").parse::<Url>() {
                let _ = window.navigate(url);
            }
        }
        // Watch the server. When the model is switched, the server starts a replacement that takes over the port and then ends itself: the
        // replacement holds the same standard-input pipe, so the pipe must stay open (dropping it stops the replacement). After the
        // program has ended, only the port tells whether a server is there.
        let address = SocketAddr::from(([127, 0, 0, 1], port));
        let mut child_ended: Option<String> = None;
        let mut down_since: Option<Instant> = None;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            if shell.generation.load(Ordering::SeqCst) != generation {
                return;
            }
            if child_ended.is_none() {
                let ended = shell.running.lock().unwrap().as_mut().and_then(|s| s.child.try_wait().ok().flatten());
                if let Some(status) = ended {
                    child_ended = Some(status.to_string());
                }
            }
            if TcpStream::connect_timeout(&address, Duration::from_millis(500)).is_ok() {
                down_since = None;
                continue;
            }
            let Some(status) = &child_ended else { continue };
            // gone and silent: a replacement may still be loading its model, so give it time
            let since = *down_since.get_or_insert_with(Instant::now);
            if since.elapsed() > Duration::from_secs(10 * 60) {
                shell.running.lock().unwrap().take();
                show_loading(&app, &stopped_message(&format!("The server has stopped ({status}).")));
                return;
            }
        }
    });
}

fn open_folder(path: &std::path::Path) -> Result<(), String> {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    Command::new(program).arg(path).spawn().map(|_| ()).map_err(|e| format!("The folder could not be opened: {e}"))
}

fn open_logs(app: &AppHandle) -> Result<(), String> {
    let dir = app.path().app_log_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    open_folder(&dir)
}

// ---- the commands the Studio page may call ----

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopSettings {
    keep_in_tray: bool,
    autostart: bool,
    log_folder: String,
    config_folder: String,
    quick_entry: String,
    default_quick_entry: String,
    /// The saved choice: "auto" or a variant name.
    variant: String,
    /// Set when the environment variable FLLM_DESKTOP_VARIANT decides (it beats the saved choice).
    variant_from_env: Option<String>,
    active_variant: String,
    active_variant_label: String,
    variant_note: String,
    variants: Vec<VariantInfo>,
}

#[derive(Serialize)]
struct VariantInfo {
    name: &'static str,
    label: &'static str,
    /// The server program of this variant is part of this installation.
    installed: bool,
    /// The hardware it is for was found.
    detected: bool,
}

#[tauri::command]
fn get_desktop_settings(app: AppHandle, shell: State<Shell>) -> Result<DesktopSettings, String> {
    let folder = |dir: tauri::Result<PathBuf>| dir.map(|d| d.display().to_string()).unwrap_or_default();
    let active = shell.active.lock().unwrap().clone();
    Ok(DesktopSettings {
        keep_in_tray: shell.settings.lock().unwrap().keep_in_tray,
        autostart: app.autolaunch().is_enabled().map_err(|e| e.to_string())?,
        log_folder: folder(app.path().app_log_dir()),
        config_folder: folder(app.path().app_config_dir()),
        quick_entry: shell.settings.lock().unwrap().quick_entry.clone(),
        default_quick_entry: DEFAULT_QUICK_ENTRY.to_owned(),
        variant: shell.settings.lock().unwrap().variant.clone(),
        variant_from_env: std::env::var("FLLM_DESKTOP_VARIANT").ok().filter(|v| !v.trim().is_empty()),
        active_variant: active.name,
        active_variant_label: active.label,
        variant_note: active.note,
        variants: {
            let probe = variants::Probe::detect();
            variants::VARIANTS
                .iter()
                .map(|v| VariantInfo { name: v.name, label: v.label, installed: variants::sidecar_path(v).is_file(), detected: (v.detect)(&probe) })
                .collect()
        },
    })
}

/// Sets the engine variant ("auto" or a variant name); it takes effect at the next start of the server.
#[tauri::command]
fn set_variant(app: AppHandle, shell: State<Shell>, value: String) -> Result<(), String> {
    let request = variants::parse_request(&value)?;
    let mut settings = shell.settings.lock().unwrap();
    settings.variant = match request {
        variants::Request::Auto => variants::AUTO.to_owned(),
        variants::Request::Named(v) => v.name.to_owned(),
    };
    save_settings(&app, &settings)
}

/// Sets the quick-entry hotkey ("" switches it off). The old hotkey stays when the new one cannot be registered.
#[tauri::command]
fn set_quick_entry(app: AppHandle, shell: State<Shell>, value: String) -> Result<(), String> {
    let value = value.trim().to_owned();
    if let Err(e) = register_quick_entry(&app, &value) {
        let old = shell.settings.lock().unwrap().quick_entry.clone();
        let _ = register_quick_entry(&app, &old);
        return Err(e);
    }
    let mut settings = shell.settings.lock().unwrap();
    settings.quick_entry = value;
    save_settings(&app, &settings)
}

#[tauri::command]
fn set_keep_in_tray(app: AppHandle, shell: State<Shell>, value: bool) -> Result<(), String> {
    let mut settings = shell.settings.lock().unwrap();
    settings.keep_in_tray = value;
    save_settings(&app, &settings)
}

#[tauri::command]
fn set_autostart(app: AppHandle, value: bool) -> Result<(), String> {
    let manager = app.autolaunch();
    if value { manager.enable() } else { manager.disable() }.map_err(|e| e.to_string())
}

#[tauri::command]
fn open_log_folder(app: AppHandle) -> Result<(), String> {
    open_logs(&app)
}

#[tauri::command]
fn restart_server(app: AppHandle) {
    show_loading(&app, "Restarting the server ...");
    launch(&app);
}

fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open fLLM", true, None::<&str>)?;
    let restart = MenuItem::with_id(app, "restart", "Restart server", true, None::<&str>)?;
    let logs = MenuItem::with_id(app, "logs", "Open log folder", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &restart, &logs, &quit])?;
    let mut tray = TrayIconBuilder::new()
        .tooltip("fLLM")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => show_main(app),
            "restart" => restart_server(app.clone()),
            "logs" => {
                let _ = open_logs(app);
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                show_main(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

fn main() {
    let app = tauri::Builder::default()
        // must be the first plugin: a second start only brings the first window to the front
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_main(app)))
        .plugin(tauri_plugin_autostart::init(MacosLauncher::LaunchAgent, Some(vec![AUTOSTART_ARG])))
        .plugin(tauri_plugin_window_state::Builder::default().with_state_flags(StateFlags::all() - StateFlags::VISIBLE).build())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        quick_entry(app);
                    }
                })
                .build(),
        )
        .manage(Shell {
            port: free_port(),
            loading_url: Mutex::new(None),
            running: Mutex::new(None),
            generation: AtomicU64::new(0),
            settings: Mutex::new(Settings::default()),
            active: Mutex::new(Active::default()),
        })
        .invoke_handler(tauri::generate_handler![get_desktop_settings, set_keep_in_tray, set_autostart, open_log_folder, restart_server, set_quick_entry, set_variant])
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" && window.state::<Shell>().settings.lock().unwrap().keep_in_tray {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .setup(|app| {
            let handle = app.handle().clone();
            let shell = app.state::<Shell>();
            *shell.settings.lock().unwrap() = load_settings(&handle);
            let window = app.get_webview_window("main").expect("main window");
            *shell.loading_url.lock().unwrap() = window.url().ok();
            let keys = shell.settings.lock().unwrap().quick_entry.clone();
            if let Err(e) = register_quick_entry(&handle, &keys) {
                eprintln!("{e}");
            }
            // started by the system at sign-in and kept in the tray: stay out of the way
            if std::env::args().any(|a| a == AUTOSTART_ARG) && shell.settings.lock().unwrap().keep_in_tray {
                let _ = window.hide();
            }
            if let Err(e) = build_tray(app) {
                eprintln!("The tray icon could not be created: {e}");
            }
            launch(&handle);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the fLLM app");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            handle.state::<Shell>().generation.fetch_add(1, Ordering::SeqCst);
            stop_server(handle);
        }
    });
}
