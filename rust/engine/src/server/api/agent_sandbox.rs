//! The sandbox for the commands the agent runs (`bash`, background jobs).
//!
//! Approvals decide whether a command may run; the sandbox decides what it can reach once it runs. That is the
//! difference between asking the user about every command and letting routine work happen inside a boundary. Each system
//! has its own mechanism, and [`status_json`] tells the truth about every one of them:
//!
//! - **Linux**: bubblewrap (`bwrap`), the same tool Codex uses. The root filesystem is mounted read-only, the workspace
//!   stays writable, `/proc`, `/dev` and a private `/tmp` are provided, and the command runs in its own PID and IPC
//!   namespace. Mode `strict` also takes the network away.
//! - **Windows**: a launcher (`fllm-rs sandbox-exec`) that runs the command under a restricted low-integrity token inside a
//!   job object (`agent_sandbox_win` decides, `agent_sandbox_native` applies). Implemented, not yet verified on real
//!   Windows machines; `strict` is unavailable there (no network cut-off).
//! - **macOS**: Seatbelt (`sandbox-exec`) with a profile per mode (`agent_sandbox_mac`). Unverified.
//!
//! Everything degrades honestly: when the mechanism does not work here (no `bwrap`, a failed self-test, ...) nothing is
//! wrapped, the command runs as before, and [`status_json`] says so with the reason — the agent never silently loses a
//! permission the user granted, and it never pretends to be sandboxed either. The one exception is `strict` on Windows:
//! it promises "no network", which cannot be kept there, so the command is refused with that reason instead of running
//! with a network.
use super::*;
use std::path::{Path, PathBuf};
#[cfg(not(windows))]
use std::process::{Command, Stdio};
use std::sync::OnceLock;
#[cfg(not(windows))]
use std::time::Duration;

/// `FLLM_SANDBOX=off|workspace|strict`. The default is `workspace` on Linux, used only when bubblewrap actually works. On
/// Windows and macOS the mechanism is not verified on real machines yet, so nothing changes until somebody chooses it.
pub(super) const DEFAULT_MODE: &str = if cfg!(target_os = "linux") { "workspace" } else { "off" };

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

/// Whether this machine can sandbox, and the program that does it. Probed once, with a real run.
fn probe() -> &'static Result<PathBuf, String> {
    static RESULT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    RESULT.get_or_init(probe_platform)
}

#[cfg(windows)]
fn probe_platform() -> Result<PathBuf, String> {
    super::agent_sandbox_native::self_test().map_err(|reason| format!("the Windows sandbox self-test failed: {reason}"))
}

#[cfg(target_os = "macos")]
fn probe_platform() -> Result<PathBuf, String> {
    use super::agent_sandbox_mac as mac;
    let program = PathBuf::from(mac::SANDBOX_EXEC);
    if !program.is_file() {
        return Err("/usr/bin/sandbox-exec is not there".to_owned());
    }
    let root = std::fs::canonicalize(std::env::temp_dir())
        .map_err(|e| e.to_string())?
        .join(format!("fllm-sandbox-probe-{}", std::process::id()));
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let denied = root.with_extension("denied");
    let run = |line: &str| -> Result<bool, String> {
        let argv = mac::argv(Mode::Workspace, line, &root.to_string_lossy(), &root.to_string_lossy());
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start sandbox-exec: {e}"))?;
        let started = std::time::Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status.success()),
                Ok(None) if started.elapsed() > Duration::from_secs(10) => {
                    let _ = child.kill();
                    return Err("sandbox-exec did not finish within 10 seconds".to_owned());
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => return Err(e.to_string()),
            }
        }
    };
    let write_line = |path: &str| format!("echo x > '{path}'");
    let result = super::agent_sandbox_win::enforcement_selftest(
        &run,
        &write_line,
        &root.join("allowed.txt").to_string_lossy(),
        &denied.to_string_lossy(),
    );
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_file(&denied);
    result.map(|()| program).map_err(|reason| format!("the Seatbelt self-test failed: {reason}"))
}

#[cfg(not(any(windows, target_os = "macos")))]
fn probe_platform() -> Result<PathBuf, String> {
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
}

#[cfg(not(any(windows, target_os = "macos")))]
fn find_bwrap() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|dir| dir.join("bwrap")).find(|candidate| candidate.is_file())
}

/// The argv that runs `line` inside the sandbox: `Ok(None)` when nothing is wrapped (the sandbox is off, or does not work
/// here and the command runs as before), `Err` when the command must not run at all.
///
/// Pure for the caller: it only has to spawn the argv. `root` is the workspace that stays writable, `cwd` the folder the
/// command starts in (both inside the workspace).
pub(super) fn wrap(line: &str, root: &Path, cwd: &Path) -> Result<Option<Vec<String>>, String> {
    let mode = mode();
    if mode == Mode::Off {
        return Ok(None);
    }
    wrap_platform(mode, line, root, cwd)
}

#[cfg(windows)]
fn wrap_platform(mode: Mode, line: &str, root: &Path, cwd: &Path) -> Result<Option<Vec<String>>, String> {
    // strict promises "no network", which Windows cannot keep: refuse instead of running with a network
    super::agent_sandbox_win::policy_for(mode)?;
    // workspace: like a missing bwrap on Linux, a sandbox that does not work here (the self-test failed) leaves the command
    // unwrapped and the status says why. A sandbox that works but fails to start THIS command is an error, never a bypass.
    let Ok(launcher) = probe().as_ref() else { return Ok(None) };
    let temp = super::agent_sandbox_native::prepare_workspace(root).map_err(|e| format!("the sandbox cannot start the command: {e}"))?;
    let mut argv = vec![launcher.to_string_lossy().into_owned()];
    argv.extend(super::agent_sandbox_native::launcher_command(mode, line, root, cwd, &temp));
    Ok(Some(argv))
}

#[cfg(target_os = "macos")]
fn wrap_platform(mode: Mode, line: &str, root: &Path, _cwd: &Path) -> Result<Option<Vec<String>>, String> {
    let Ok(_program) = probe().as_ref() else { return Ok(None) };
    let real_root = std::fs::canonicalize(root).map_err(|e| format!("the sandbox cannot resolve {}: {e}", root.display()))?;
    let tmp = std::fs::canonicalize(std::env::temp_dir())
        .map_err(|e| format!("the sandbox cannot resolve the temp folder: {e}"))?
        .join(format!("fllm-sandbox-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| format!("the sandbox cannot create its temp folder: {e}"))?;
    Ok(Some(super::agent_sandbox_mac::argv(mode, line, &real_root.to_string_lossy(), &tmp.to_string_lossy())))
}

#[cfg(not(any(windows, target_os = "macos")))]
fn wrap_platform(mode: Mode, line: &str, root: &Path, cwd: &Path) -> Result<Option<Vec<String>>, String> {
    let Ok(bwrap) = probe().as_ref() else { return Ok(None) };
    Ok(Some(argv_for(mode, bwrap, line, root, cwd)))
}

/// The argument list for one mode. Split out from [`wrap`] so the shape is testable without a machine that has bwrap.
#[cfg_attr(any(windows, target_os = "macos"), allow(dead_code))]
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

/// What a mechanism does and does not do, for the status. Written per system and mode so it can be tested for every
/// system on any machine. `working` is whether the mechanism passed its probe here.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Description {
    pub mechanism: &'static str,
    pub enforced: Vec<&'static str>,
    pub not_enforced: Vec<&'static str>,
    pub verification: &'static str,
}

pub(super) fn describe(os: &str, mode: Mode, working: bool) -> Description {
    match os {
        "windows" => {
            let mechanism = "restricted low-integrity token + job object, started through the fllm-rs sandbox-exec launcher";
            let verification = "implemented and cross-compiled, its decisions are unit-tested; not yet verified on real Windows machines (the first command runs a self-test on this machine and the sandbox is reported unavailable when it fails)";
            let (enforced, not_enforced) = match mode {
                Mode::Off => (vec![], vec!["everything: the sandbox is off"]),
                Mode::Strict => (vec![], vec!["everything: strict is unavailable on Windows (no network cut-off), commands are refused in this mode"]),
                Mode::Workspace if !working => (vec![], vec!["everything: the sandbox does not work on this machine, commands run unwrapped"]),
                Mode::Workspace => (
                    vec![
                        "the whole process tree of a command is killed on stop, timeout or when the launcher ends (job object)",
                        "writes are refused outside the workspace and a private temp folder (low integrity level; the workspace is labelled low once)",
                        "privileges are removed from the token and administrator-like groups are deny-only",
                        "at most 512 processes and three quarters of the physical memory per command",
                        "no clipboard, desktop, global atom or window handle access (job UI restrictions)",
                        "environment variables that look like secrets (..._TOKEN, ..._KEY, ...) are removed; TEMP and TMP point to the private folder",
                    ],
                    vec![
                        "reading: the whole disk stays readable, including your profile",
                        "the network: commands can reach it (strict, which would cut it, is unavailable on Windows)",
                        "other low-integrity places (for example AppData\\LocalLow) stay writable",
                    ],
                ),
            };
            Description { mechanism, enforced, not_enforced, verification }
        }
        "macos" => {
            let mechanism = "Seatbelt (sandbox-exec) with a profile per mode";
            let verification = "unverified: profile text and arguments are unit-tested, no Mac was available to run them (a self-test runs on this machine and the sandbox is reported unavailable when it fails)";
            match (mode, working) {
                (Mode::Off, _) => Description { mechanism, enforced: vec![], not_enforced: vec!["everything: the sandbox is off"], verification },
                (_, false) => Description { mechanism, enforced: vec![], not_enforced: vec!["everything: the sandbox does not work on this machine, commands run unwrapped"], verification },
                (m, true) => {
                    let mut not_enforced = vec!["reading: the whole disk stays readable", "separate process and IPC namespaces"];
                    if m == Mode::Workspace {
                        not_enforced.push("the network: commands can reach it (strict cuts it)");
                    }
                    let mut enforced = vec!["writes are refused outside the workspace and a private temp folder"];
                    if m == Mode::Strict {
                        enforced.push("no network");
                    }
                    Description { mechanism, enforced, not_enforced, verification }
                }
            }
        }
        "linux" => {
            let mechanism = "bubblewrap (bwrap)";
            let verification = "verified on Linux (a real bwrap run decides availability)";
            match (mode, working) {
                (Mode::Off, _) => Description { mechanism, enforced: vec![], not_enforced: vec!["everything: the sandbox is off"], verification },
                (_, false) => Description { mechanism, enforced: vec![], not_enforced: vec!["everything: bubblewrap does not work on this machine, commands run unwrapped"], verification },
                (m, true) => {
                    let mut enforced = vec![
                        "the root filesystem is read-only, the workspace is writable, /tmp is private",
                        "own PID, IPC and UTS namespaces; the command dies with the server",
                    ];
                    let mut not_enforced = vec!["reading: the whole filesystem stays readable"];
                    if m == Mode::Strict {
                        enforced.push("no network");
                    } else {
                        not_enforced.push("the network: commands can reach it (strict cuts it)");
                    }
                    Description { mechanism, enforced, not_enforced, verification }
                }
            }
        }
        _ => Description {
            mechanism: "none",
            enforced: vec![],
            not_enforced: vec!["everything: there is no sandbox for this system, commands run unwrapped"],
            verification: "no sandbox exists for this system",
        },
    }
}

/// The state the page shows: which mode is asked for, whether it works here, and why not.
pub(crate) fn status_json() -> Value {
    let mode = mode();
    let mut status = match probe() {
        Ok(path) => json!({"available": true, "path": path.display().to_string(), "reason": Value::Null}),
        Err(reason) => json!({"available": false, "path": Value::Null, "reason": reason}),
    };
    let mut working = status["available"].as_bool().unwrap_or(false);
    // strict does not exist on Windows even when the rest works: say that instead of "available"
    if cfg!(windows) && mode == Mode::Strict {
        working = false;
        status = json!({"available": false, "path": Value::Null, "reason": super::agent_sandbox_win::STRICT_UNAVAILABLE});
    }
    let active = mode != Mode::Off && working;
    let description = describe(std::env::consts::OS, mode, active);
    json!({
        "mode": mode.name(),
        "active": active,
        "requested": std::env::var("FLLM_SANDBOX").ok(),
        "sandbox": status,
        "platform": std::env::consts::OS,
        "mechanism": description.mechanism,
        "enforced": description.enforced,
        "not_enforced": description.not_enforced,
        "verification": description.verification,
        "default_mode": DEFAULT_MODE,
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
    fn only_linux_turns_the_sandbox_on_by_default() {
        assert_eq!(DEFAULT_MODE, if cfg!(target_os = "linux") { "workspace" } else { "off" });
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
        assert!(status["enforced"].is_array() && status["not_enforced"].is_array());
        assert!(status["verification"].is_string() && status["mechanism"].is_string());
        if !status["active"].as_bool().unwrap_or(false) {
            assert!(!status["sandbox"]["reason"].is_null() || status["mode"] == "off", "a missing sandbox must say why: {status}");
        }
    }

    #[test]
    fn the_description_of_windows_is_honest_about_what_it_cannot_do() {
        let on = describe("windows", Mode::Workspace, true);
        assert!(on.enforced.iter().any(|e| e.contains("killed")) && on.enforced.iter().any(|e| e.contains("low integrity")));
        // reading and the network are listed as NOT enforced, never claimed
        assert!(on.not_enforced.iter().any(|e| e.contains("reading")));
        assert!(on.not_enforced.iter().any(|e| e.contains("network")));
        assert!(!on.enforced.iter().any(|e| e.contains("no network")));
        assert!(on.verification.contains("not yet verified"));
        // strict claims nothing on Windows
        let strict = describe("windows", Mode::Strict, false);
        assert!(strict.enforced.is_empty() && strict.not_enforced[0].contains("unavailable"));
        // a sandbox that failed its self-test claims nothing either
        assert!(describe("windows", Mode::Workspace, false).enforced.is_empty());
    }

    #[test]
    fn every_system_describes_each_mode_and_claims_the_network_cut_only_in_strict() {
        for os in ["linux", "macos", "windows", "freebsd"] {
            for mode in [Mode::Off, Mode::Workspace, Mode::Strict] {
                for working in [false, true] {
                    let d = describe(os, mode, working);
                    assert!(!d.mechanism.is_empty() && !d.verification.is_empty() && !d.not_enforced.is_empty(), "{os} {mode:?}");
                    let claims_no_network = d.enforced.iter().any(|e| e.contains("no network"));
                    assert_eq!(claims_no_network, working && mode == Mode::Strict && (os == "linux" || os == "macos"), "{os} {mode:?} {working}");
                    if mode == Mode::Off || !working {
                        assert!(d.enforced.is_empty(), "{os} {mode:?}: nothing is enforced");
                    }
                }
            }
        }
    }
}
