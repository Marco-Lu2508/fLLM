//! The server that comes with the app. Opening the app starts `fllm-rs studio` next to it when no server answers, and closing the app stops it,
//! so there is nothing to start by hand. (Later the engine moves into this process; the window talks to it the same way.)
//!
//! The rules, in order: `FLLM_URL` set: that server, nothing started. A server already answers on the usual address (a Docker container, a
//! server of your own): that one. Otherwise the binary is looked for (`FLLM_RS`, next to the app, on the PATH) and started on a free port.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

pub const DEFAULT_URL: &str = "http://127.0.0.1:8080";

/// A server this app started; it is stopped when this value is dropped.
pub struct Sidecar {
    child: Child,
    pub url: String,
}

impl std::fmt::Debug for Sidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sidecar({}, pid {})", self.url, self.child.id())
    }
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "fllm-rs.exe"
    } else {
        "fllm-rs"
    }
}

/// Where the server binary is: `FLLM_RS`, then next to this program, then the PATH.
pub fn locate() -> Option<PathBuf> {
    locate_in(std::env::var_os("FLLM_RS").map(PathBuf::from), std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)), std::env::var_os("PATH"))
}

pub fn locate_in(explicit: Option<PathBuf>, beside: Option<PathBuf>, path: Option<std::ffi::OsString>) -> Option<PathBuf> {
    if let Some(p) = explicit.filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) = beside.map(|d| d.join(binary_name())).filter(|p| p.is_file()) {
        return Some(p);
    }
    std::env::split_paths(&path?).map(|d| d.join(binary_name())).find(|p| p.is_file())
}

/// Whether something listens on the address of `url` (a quick connect; no request is made).
pub fn answers(url: &str) -> bool {
    let host = url.trim_start_matches("http://").trim_start_matches("https://").trim_end_matches('/');
    let Ok(addr) = host.parse::<SocketAddr>() else { return false };
    // connecting to a closed port in the range of the system's own ports can connect the socket to itself (TCP simultaneous open): not a server
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok_and(|s| s.local_addr().ok() != s.peer_addr().ok())
}

/// A port nobody listens on: the usual 8080 when it is free, else one the system picks.
pub fn free_port() -> u16 {
    if TcpListener::bind(("127.0.0.1", 8080)).is_ok() {
        return 8080;
    }
    TcpListener::bind(("127.0.0.1", 0)).and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(8080)
}

impl Sidecar {
    /// Starts `binary studio --no-open --host 127.0.0.1 --port PORT` with its output in `log`.
    pub fn start(binary: &Path, port: u16, log: &Path) -> std::io::Result<Sidecar> {
        if let Some(dir) = log.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let out = std::fs::File::create(log)?;
        let err = out.try_clone()?;
        let mut cmd = Command::new(binary);
        cmd.args(["studio", "--no-open", "--host", "127.0.0.1", "--port", &port.to_string()]).stdin(Stdio::null()).stdout(out).stderr(err);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let child = cmd.spawn()?;
        Ok(Sidecar { child, url: format!("http://127.0.0.1:{port}") })
    }

    /// Whether the process still runs (it ends by itself when the binary refuses its arguments).
    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// What the app does at start: the server to use, the sidecar it started (if any), and a sentence for the "no server" screen.
pub fn choose(log: &Path) -> (String, Option<Sidecar>, Option<String>) {
    if let Ok(url) = std::env::var("FLLM_URL") {
        return (url.trim_end_matches('/').to_string(), None, None);
    }
    if answers(DEFAULT_URL) {
        return (DEFAULT_URL.to_string(), None, None);
    }
    match locate() {
        None => (DEFAULT_URL.to_string(), None, Some(format!("The app did not find `{}` next to itself or on the PATH, so it could not start a server of its own.", binary_name()))),
        Some(binary) => match Sidecar::start(&binary, free_port(), log) {
            Ok(s) => (s.url.clone(), Some(s), None),
            Err(e) => (DEFAULT_URL.to_string(), None, Some(format!("The app could not start {}: {e}.", binary.display()))),
        },
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// The tests start processes and watch sockets: a process that is forked while another test holds a listener keeps that socket open until
    /// it has exec'd, which makes a closed port look open for a moment. So they run one after the other.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[test]
    fn the_binary_is_found_explicit_first_then_beside_the_app_then_on_the_path() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let c = tempfile::tempdir().unwrap();
        let (pa, pb, pc) = (script(a.path(), "fllm-rs", ""), script(b.path(), "fllm-rs", ""), script(c.path(), "fllm-rs", ""));
        let path = std::env::join_paths([c.path()]).unwrap();
        assert_eq!(locate_in(Some(pa.clone()), Some(b.path().into()), Some(path.clone())), Some(pa));
        assert_eq!(locate_in(Some("/nowhere/fllm-rs".into()), Some(b.path().into()), Some(path.clone())), Some(pb), "a wrong FLLM_RS falls through");
        assert_eq!(locate_in(None, Some(a.path().join("missing")), Some(path)), Some(pc));
        assert_eq!(locate_in(None, None, None), None);
    }

    #[test]
    fn a_started_server_gets_the_arguments_and_a_log_and_ends_with_the_app() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let d = tempfile::tempdir().unwrap();
        // a stand-in that writes its arguments and waits, like a server
        let fake = script(d.path(), "fllm-rs", "echo \"args: $@\"\nexec sleep 30");
        let log = d.path().join("logs/server.log");
        let mut s = Sidecar::start(&fake, 8123, &log).unwrap();
        assert_eq!(s.url, "http://127.0.0.1:8123");
        std::thread::sleep(Duration::from_millis(300));
        assert!(s.running());
        let pid = s.pid();
        assert!(std::fs::read_to_string(&log).unwrap().contains("args: studio --no-open --host 127.0.0.1 --port 8123"));
        drop(s);
        // the process is gone (a killed and reaped child has no /proc entry)
        assert!(!Path::new(&format!("/proc/{pid}")).exists(), "the server ends when the app lets go of it");
    }

    #[test]
    fn a_binary_that_cannot_run_is_an_error_and_one_that_exits_is_noticed() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let d = tempfile::tempdir().unwrap();
        assert!(Sidecar::start(&d.path().join("nope"), 8124, &d.path().join("l.log")).is_err());
        let quick = script(d.path(), "fllm-rs", "exit 3");
        let mut s = Sidecar::start(&quick, 8125, &d.path().join("l2.log")).unwrap();
        std::thread::sleep(Duration::from_millis(400));
        assert!(!s.running());
    }

    #[test]
    fn a_free_port_is_free_and_an_answering_one_is_seen() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let port = free_port();
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
        let l = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let busy = l.local_addr().unwrap().port();
        assert!(answers(&format!("http://127.0.0.1:{busy}")));
        drop(l);
        assert!(!answers(&format!("http://127.0.0.1:{busy}/")));
        assert!(!answers("not a url"));
    }
}
