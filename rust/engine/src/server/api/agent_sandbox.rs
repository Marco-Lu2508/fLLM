//! The sandbox for the commands the agent runs (`bash`, background jobs).
//!
//! Approvals decide whether a command may run; the sandbox decides what it can reach once it runs. That is the
//! difference between asking the user about every command and letting routine work happen inside a boundary. The
//! mechanic is bubblewrap (`bwrap`), the same tool Codex uses on Linux: the root filesystem is mounted read-only, the
//! workspace stays writable, `/proc`, `/dev` and a private `/tmp` are provided, and the command runs in its own PID and
//! IPC namespace. Mode `strict` also takes the network away.
//!
//! Everything degrades honestly: without `bwrap` (or without permission to create user namespaces) nothing is wrapped,
//! the command runs as before, and [`status_json`] says so — the agent never silently loses a permission the user
//! granted, and it never pretends to be sandboxed either.
use super::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

/// `FLLM_SANDBOX=off|workspace|strict`. Default: `workspace`, used only when bubblewrap actually works here.
pub(super) const DEFAULT_MODE: &str = "workspace";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    Workspace,
    Strict,
}

impl Mode {
    pub(super) fn parse(text: &str) -> Mode {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "no" | "false" => Mode::Off,
            "strict" => Mode::Strict,
            _ => Mode::Workspace,
        }
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Workspace => "workspace",
            Mode::Strict => "strict",
        }
    }
}

pub(super) fn mode() -> Mode {
    // the page can choose too (Settings > Runtime); the environment still wins, see studio_mode
    Mode::parse(&super::studio_mode::sandbox_mode())
}

/// Whether this machine can sandbox at all — for the page, so it does not promise what the machine cannot do.
pub(super) fn available() -> bool {
    probe().is_ok()
}

/// Whether this machine can sandbox: the `bwrap` binary, and a test run that proves user namespaces work. Probed once.
fn probe() -> &'static Result<PathBuf, String> {
    static RESULT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    RESULT.get_or_init(|| {
        if cfg!(windows) {
            return Err("bubblewrap is a Linux tool; the Windows sandbox of fLLM is not implemented yet".to_owned());
        }
        let path = find_bwrap().ok_or_else(|| "bubblewrap (bwrap) is not installed".to_owned())?;
        // a real run: some systems have the binary but forbid unprivileged user namespaces
        let mut child = Command::new(&path)
            .args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--", "/bin/true"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let ok = match child.as_mut() {
            Ok(child) => {
                let started = std::time::Instant::now();
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => break status.success(),
                        Ok(None) if started.elapsed() > Duration::from_secs(5) => {
                            let _ = child.kill();
                            break false;
                        }
                        Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                        Err(_) => break false,
                    }
                }
            }
            Err(_) => false,
        };
        if ok {
            Ok(path)
        } else {
            Err("bubblewrap cannot create a user namespace here (AppArmor or a container policy blocks it)".to_owned())
        }
    })
}

fn find_bwrap() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("bwrap")).find(|candidate| candidate.is_file())
}

/// The argv that runs `line` inside the sandbox, or `None` when nothing can be wrapped here.
///
/// Pure: the caller only has to spawn it. `root` is the workspace that stays writable, `cwd` the folder the command
/// starts in (both inside the workspace).
pub(super) fn wrap(line: &str, root: &Path, cwd: &Path) -> Option<Vec<String>> {
    let mode = mode();
    if mode == Mode::Off {
        return None;
    }
    let bwrap = probe().as_ref().ok()?;
    Some(argv_for(mode, bwrap, line, root, cwd))
}

/// The argument list for one mode. Split out from [`wrap`] so the shape is testable without a machine that has bwrap.
fn argv_for(mode: Mode, bwrap: &Path, line: &str, root: &Path, cwd: &Path) -> Vec<String> {
    let path = |p: &Path| p.to_string_lossy().into_owned();
    let mut argv = vec![
        path(bwrap),
        "--ro-bind".to_owned(), "/".to_owned(), "/".to_owned(),
        "--dev".to_owned(), "/dev".to_owned(),
        "--proc".to_owned(), "/proc".to_owned(),
        "--tmpfs".to_owned(), "/tmp".to_owned(),
        "--bind".to_owned(), path(root), path(root),
        "--chdir".to_owned(), path(cwd),
        "--unshare-pid".to_owned(),
        "--unshare-ipc".to_owned(),
        "--unshare-uts".to_owned(),
        "--die-with-parent".to_owned(),
        "--new-session".to_owned(),
    ];
    if mode == Mode::Strict {
        argv.push("--unshare-net".to_owned());
    }
    argv.push("--".to_owned());
    argv.push("/bin/sh".to_owned());
    argv.push("-c".to_owned());
    argv.push(line.to_owned());
    argv
}

/// The state the page shows: which mode is asked for, whether it works here, and why not.
pub(super) fn status_json() -> Value {
    let mode = mode();
    let status = match probe() {
        Ok(path) => json!({"available": true, "path": path.display().to_string(), "reason": Value::Null}),
        Err(reason) => json!({"available": false, "path": Value::Null, "reason": reason}),
    };
    let active = mode != Mode::Off && status["available"].as_bool().unwrap_or(false);
    json!({
        "mode": mode.name(),
        "active": active,
        "requested": std::env::var("FLLM_SANDBOX").ok(),
        "sandbox": status,
    })
}

/// `GET /api/studio/agent/sandbox`: mode, availability and the reason when it is missing.
pub(in crate::server) async fn status() -> Response {
    Json(status_json()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_is_parsed_leniently() {
        assert_eq!(Mode::parse("off"), Mode::Off);
        assert_eq!(Mode::parse("0"), Mode::Off);
        assert_eq!(Mode::parse("STRICT"), Mode::Strict);
        assert_eq!(Mode::parse("workspace"), Mode::Workspace);
        assert_eq!(Mode::parse("nonsense"), Mode::Workspace);
        assert_eq!(Mode::Off.name(), "off");
    }

    #[test]
    fn the_argv_keeps_the_root_read_only_and_the_workspace_writable() {
        let argv = argv_for(Mode::Workspace, Path::new("/usr/bin/bwrap"), "echo hi", Path::new("/w"), Path::new("/w/sub"));
        let joined = argv.join(" ");
        assert!(joined.starts_with("/usr/bin/bwrap "), "{joined}");
        assert!(joined.contains("--ro-bind / /"), "{joined}");
        assert!(joined.contains("--bind /w /w"), "{joined}");
        assert!(joined.contains("--chdir /w/sub"), "{joined}");
        assert!(joined.contains("--unshare-pid"), "{joined}");
        assert!(!joined.contains("--unshare-net"), "the workspace mode keeps the network: {joined}");
        assert!(joined.ends_with("-- /bin/sh -c echo hi"), "{joined}");
    }

    #[test]
    fn the_strict_mode_takes_the_network_away() {
        let argv = argv_for(Mode::Strict, Path::new("/usr/bin/bwrap"), "curl example.com", Path::new("/w"), Path::new("/w"));
        assert!(argv.join(" ").contains("--unshare-net"));
        // and the command line is passed as one argument, never spliced into the argv
        assert_eq!(argv.last().map(String::as_str), Some("curl example.com"));
    }

    #[test]
    fn the_status_reports_a_reason_when_no_sandbox_works() {
        let status = status_json();
        assert!(status["mode"].is_string());
        assert!(status["active"].is_boolean());
        if !status["active"].as_bool().unwrap_or(false) {
            assert!(!status["sandbox"]["reason"].is_null(), "a missing sandbox must say why: {status}");
        }
    }
}
