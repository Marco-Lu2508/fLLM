//! The fLLM desktop app: a window around the Studio. The server is the `fllm-rs` program that is installed next to this one; the window
//! shows it as soon as it answers. When the app ends, the pipe on the standard input of the server closes and the server stops with it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent};

/// The page keeps its chats per address, so the app prefers the same port every time.
const FIRST_PORT: u16 = 47821;

struct Server {
    child: Child,
    _pipe: Option<ChildStdin>,
}

#[derive(Default)]
struct Running(Mutex<Option<Server>>);

fn free_port() -> u16 {
    (FIRST_PORT..FIRST_PORT + 40)
        .find(|port| TcpListener::bind(("127.0.0.1", *port)).is_ok())
        .unwrap_or(FIRST_PORT)
}

fn server_program() -> PathBuf {
    let name = if cfg!(windows) { "fllm-rs.exe" } else { "fllm-rs" };
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(name)))
        .unwrap_or_else(|| PathBuf::from(name))
}

/// The server writes what it prints into `server.log` in the log folder of the app, so a problem can be looked at afterwards.
fn server_log(dir: Option<PathBuf>) -> (Stdio, Stdio) {
    let file = dir.and_then(|dir| {
        std::fs::create_dir_all(&dir).ok()?;
        std::fs::File::create(dir.join("server.log")).ok()
    });
    match file.and_then(|f| f.try_clone().ok().map(|clone| (f, clone))) {
        Some((out, err)) => (Stdio::from(out), Stdio::from(err)),
        None => (Stdio::null(), Stdio::null()),
    }
}

fn start_server(port: u16, log_dir: Option<PathBuf>) -> Result<Server, String> {
    let mut command = Command::new(server_program());
    let (out, err) = server_log(log_dir);
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

/// Waits until the server answers. A model is loaded before the server listens, which takes a while for a big one, so the wait is long;
/// it ends early when the server program has stopped.
fn wait_until_answers(port: u16, handle: &tauri::AppHandle, window: &tauri::WebviewWindow) -> Result<(), String> {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let started = Instant::now();
    let mut last_note = Duration::ZERO;
    while started.elapsed() < Duration::from_secs(30 * 60) {
        if TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_ok() {
            return Ok(());
        }
        if let Some(server) = handle.state::<Running>().0.lock().unwrap().as_mut() {
            if let Ok(Some(status)) = server.child.try_wait() {
                return Err(format!("The server stopped right after the start ({status}). Its log is in the log folder of the app (server.log)."));
            }
        }
        if started.elapsed() - last_note >= Duration::from_secs(2) && started.elapsed() > Duration::from_secs(6) {
            last_note = started.elapsed();
            let _ = window.eval(&format!(
                "var n=document.getElementById('note'); if(n) n.textContent='Loading the last model ... {} s'",
                started.elapsed().as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err("The server did not start in time.".to_owned())
}
fn main() {
    let app = tauri::Builder::default()
        .manage(Running::default())
        .setup(|app| {
            let port = free_port();
            let started = start_server(port, app.path().app_log_dir().ok());
            let window = app.get_webview_window("main").expect("main window");
            let waiter = window.clone();
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let outcome = started.and_then(|server| {
                    *handle.state::<Running>().0.lock().unwrap() = Some(server);
                    wait_until_answers(port, &handle, &waiter)
                });
                let script = match outcome {
                    Ok(()) => format!("location.replace('http://127.0.0.1:{port}/')"),
                    Err(message) => format!("document.getElementById('note').textContent = {message:?}"),
                };
                let _ = window.eval(&script);
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the fLLM app");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut server) = handle.state::<Running>().0.lock().unwrap().take() {
                drop(server._pipe.take());
                let _ = server.child.kill();
            }
        }
    });
}