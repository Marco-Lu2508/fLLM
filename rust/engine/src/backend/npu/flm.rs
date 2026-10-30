//! AMD NPU (XDNA 2, Ryzen AI) through FastFlowLM.
//!
//! fLLM has no kernels of its own for the XDNA NPU. FastFlowLM (https://github.com/FastFlowLM/FastFlowLM; the NPU binaries are AMD's and may be used and
//! redistributed unmodified) does: it runs Llama, Qwen, Gemma, gpt-oss and others on the NPU with its own model files and serves them over an
//! OpenAI-compatible HTTP interface. This module finds an installed `flm`, asks it for the state of the NPU stack and its models, downloads
//! models (`flm pull`) and runs `flm serve` as a process that fLLM manages; the Studio then lists the model as a provider ("AMD NPU
//! (FastFlowLM)"), so a chat can use it like any other model. Nothing of FastFlowLM is bundled: it is installed by the user (Windows: the
//! installer of the release page; Linux: the `.deb` for the distribution), and the NPU needs a current AMD driver (Windows) or the `amdxdna` driver
//! of the kernel (Linux), see `flm validate`.
//!
//! Everything here is plain `std` (processes and files) and `reqwest::blocking` for the readiness probe; the routes of the Studio call it from
//! blocking tasks. The same code serves `fllm-rs amd-npu`.
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::probe::{self, Vendor};

/// The id of the provider the running model is registered under.
pub const PROVIDER_ID: &str = "amd-npu";
pub const PROVIDER_NAME: &str = "AMD NPU (FastFlowLM)";
/// FastFlowLM loads a model at start; the first start of a model compiles it and can take minutes.
const READY_TIMEOUT: Duration = Duration::from_secs(300);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const RELEASES: &str = "https://github.com/FastFlowLM/FastFlowLM/releases";

#[cfg(windows)]
const EXE: &str = "flm.exe";
#[cfg(not(windows))]
const EXE: &str = "flm";

// ---------------------------------------------------------------------------------------------------------------
// Finding and asking `flm`
// ---------------------------------------------------------------------------------------------------------------

/// The `flm` program: `FLLM_FLM_PATH`, the `PATH`, then the places the installers use.
pub fn flm_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FLLM_FLM_PATH").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(EXE);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let mut fixed: Vec<PathBuf> = Vec::new();
    #[cfg(windows)]
    {
        for var in ["ProgramFiles", "ProgramW6432", "LOCALAPPDATA"] {
            if let Some(base) = std::env::var_os(var).map(PathBuf::from) {
                fixed.push(base.join("flm").join(EXE));
                fixed.push(base.join("FastFlowLM").join(EXE));
                fixed.push(base.join("Programs").join("flm").join(EXE));
            }
        }
    }
    #[cfg(not(windows))]
    {
        for dir in ["/usr/bin", "/usr/local/bin", "/opt/fastflowlm/bin", "/opt/flm/bin"] {
            fixed.push(Path::new(dir).join(EXE));
        }
    }
    fixed.into_iter().find(|p| p.is_file())
}

fn command(path: &Path) -> Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut c = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashes up next to the app
    }
    c
}

/// Runs `flm ARGS` and returns its output (both streams, escape codes removed) when it ends within `timeout`.
fn run(path: &Path, args: &[&str], timeout: Duration) -> Result<String, String> {
    let mut child = command(path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", path.display()))?;
    let drain = |stream: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(mut s) = stream {
                let mut bytes = Vec::new();
                let _ = s.read_to_end(&mut bytes);
                text = String::from_utf8_lossy(&bytes).into_owned();
            }
            text
        })
    };
    let out = drain(child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>));
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("`flm {}` did not finish within {} seconds", args.join(" "), timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(40)),
            Err(e) => return Err(e.to_string()),
        }
    }
    let mut text = out.join().unwrap_or_default();
    text.push_str(&err.join().unwrap_or_default());
    Ok(strip_ansi(&text))
}

/// Removes terminal colour codes (`ESC [ ... letter`) and carriage returns.
pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        if c != '\r' {
            out.push(c);
        }
    }
    out
}

/// `flm version` prints `FLM v1.0.11`.
pub fn version(path: &Path) -> Option<String> {
    let text = run(path, &["version"], COMMAND_TIMEOUT).ok()?;
    text.lines().map(str::trim).find(|l| !l.is_empty()).map(|l| l.trim_start_matches("FLM").trim().to_owned()).filter(|v| !v.is_empty())
}

/// What `flm validate` found out about the NPU stack.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Validation {
    /// No line starts with `[ERROR]`.
    pub ok: bool,
    /// The `[ERROR]` lines, without the tag.
    pub problems: Vec<String>,
    /// All lines (the kernel, the driver, the firmware, the NPU platform), at most 30.
    pub lines: Vec<String>,
}

pub fn parse_validate(text: &str) -> Validation {
    let lines: Vec<String> = text.lines().map(|l| l.trim().to_owned()).filter(|l| !l.is_empty()).take(30).collect();
    let problems: Vec<String> = lines.iter().filter_map(|l| l.strip_prefix("[ERROR]").map(|r| r.trim().to_owned())).collect();
    Validation { ok: problems.is_empty() && !lines.is_empty(), problems, lines }
}

/// A model of the FastFlowLM catalogue.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub tag: String,
    pub installed: bool,
}

/// Model tags are passed as arguments, never through a shell, but they also become file names and URLs: letters, digits and `. : _ -` only.
pub fn valid_tag(tag: &str) -> bool {
    !tag.is_empty() && tag.len() <= 80 && tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '_' | '-'))
}

/// `flm list`: one `- tag marker` line per model, a check mark for the installed ones.
pub fn parse_list(text: &str) -> Vec<Model> {
    let mut models = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("- ") else { continue };
        let mut words = rest.split_whitespace();
        let Some(tag) = words.next().filter(|t| valid_tag(t)) else { continue };
        let marker: String = words.collect::<Vec<_>>().join(" ");
        models.push(Model { tag: tag.to_owned(), installed: marker.contains('\u{2705}') || marker.to_ascii_lowercase().contains("installed") });
    }
    models
}

pub fn list_models(path: &Path) -> Result<Vec<Model>, String> {
    run(path, &["list"], COMMAND_TIMEOUT).map(|t| parse_list(&t))
}

// ---------------------------------------------------------------------------------------------------------------
// The managed server and the downloads
// ---------------------------------------------------------------------------------------------------------------

struct Running {
    child: Child,
    tag: String,
    port: u16,
    since: Instant,
}

/// What the Studio asks for: nothing running, starting, running or the last start failed.
#[derive(Debug, Clone, PartialEq)]
pub enum Serving {
    Idle,
    Starting { tag: String },
    Running { tag: String, port: u16 },
    Failed { tag: String, error: String },
}

static SERVER: Mutex<Option<Running>> = Mutex::new(None);
static SERVING: Mutex<Serving> = Mutex::new(Serving::Idle);
static HOOK: std::sync::OnceLock<fn(&Serving)> = std::sync::OnceLock::new();

/// `hook` is called after every change of [`serving`] (the Studio lists the running model as a provider and removes it again).
pub fn set_hook(hook: fn(&Serving)) {
    let _ = HOOK.set(hook);
}

fn set_serving(state: Serving) {
    *lock(&SERVING) = state.clone();
    if let Some(hook) = HOOK.get() {
        hook(&state);
    }
}

/// The download in progress or the last one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pulling {
    pub busy: bool,
    pub tag: String,
    /// The last line `flm pull` printed (it prints the progress there).
    pub line: String,
    pub error: Option<String>,
}

static PULL: Mutex<Pulling> = Mutex::new(Pulling { busy: false, tag: String::new(), line: String::new(), error: None });
static PULL_CHILD: Mutex<Option<Child>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn serving() -> Serving {
    lock(&SERVING).clone()
}

pub fn pulling() -> Pulling {
    lock(&PULL).clone()
}

fn log_path() -> PathBuf {
    crate::hub::studio_data_file("flm.log")
}

fn free_port() -> Result<u16, String> {
    std::net::TcpListener::bind(("127.0.0.1", 0)).and_then(|l| l.local_addr()).map(|a| a.port()).map_err(|e| format!("no free port: {e}"))
}

fn tail(path: &Path, lines: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let clean = strip_ansi(&text);
    let kept: Vec<&str> = clean.lines().filter(|l| !l.trim().is_empty()).collect();
    kept[kept.len().saturating_sub(lines)..].join("
")
}

fn ready(port: u16) -> bool {
    let Ok(client) = reqwest::blocking::Client::builder().timeout(Duration::from_secs(3)).no_proxy().build() else { return false };
    client.get(format!("http://127.0.0.1:{port}/v1/models")).send().map(|r| r.status().is_success()).unwrap_or(false)
}

/// Starts `flm serve TAG` in the background and returns at once; [`serving`] tells how it goes. A server that is already running for another
/// model is stopped first.
pub fn start(tag: &str) -> Result<(), String> {
    if !valid_tag(tag) {
        return Err("that is not a FastFlowLM model name".into());
    }
    let path = flm_path().ok_or("FastFlowLM is not installed")?;
    if let Serving::Starting { .. } = serving() {
        return Err("a model is being started already".into());
    }
    if let Serving::Running { tag: running, .. } = serving() {
        if running == tag {
            return Ok(());
        }
    }
    stop();
    let port = free_port()?;
    let log = log_path();
    if let Some(dir) = log.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = std::fs::File::create(&log).map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    let file2 = file.try_clone().map_err(|e| e.to_string())?;
    let child = command(&path)
        .args(["serve", tag, "--host", "127.0.0.1", "--port", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(file2))
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", path.display()))?;
    *lock(&SERVER) = Some(Running { child, tag: tag.to_owned(), port, since: Instant::now() });
    set_serving(Serving::Starting { tag: tag.to_owned() });
    let tag = tag.to_owned();
    std::thread::Builder::new()
        .name("fllm-flm-start".into())
        .spawn(move || {
            let deadline = Instant::now() + READY_TIMEOUT;
            loop {
                let exited = {
                    let mut guard = lock(&SERVER);
                    match guard.as_mut() {
                        // stopped (or replaced) while it was starting
                        Some(r) if r.tag == tag && r.port == port => r.child.try_wait().ok().flatten().map(|s| s.to_string()),
                        _ => return,
                    }
                };
                if let Some(status) = exited {
                    let detail = tail(&log_path(), 8);
                    *lock(&SERVER) = None;
                    set_serving(Serving::Failed { tag, error: format!("flm stopped ({status}). {detail}") });
                    return;
                }
                if ready(port) {
                    set_serving(Serving::Running { tag, port });
                    return;
                }
                if Instant::now() > deadline {
                    stop();
                    set_serving(Serving::Failed { tag, error: format!("the model did not become ready within {} seconds. {}", READY_TIMEOUT.as_secs(), tail(&log_path(), 8)) });
                    return;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Stops the managed `flm serve` (if any).
pub fn stop() {
    if let Some(mut running) = lock(&SERVER).take() {
        let _ = running.child.kill();
        let _ = running.child.wait();
    }
    let keep_failure = matches!(serving(), Serving::Failed { .. });
    if !keep_failure {
        set_serving(Serving::Idle);
    }
}

/// Seconds the managed server has been running, if it is.
pub fn uptime_seconds() -> Option<u64> {
    lock(&SERVER).as_ref().map(|r| r.since.elapsed().as_secs())
}

/// Starts `flm pull TAG` in the background; [`pulling`] shows its last output line.
pub fn pull(tag: &str) -> Result<(), String> {
    if !valid_tag(tag) {
        return Err("that is not a FastFlowLM model name".into());
    }
    let path = flm_path().ok_or("FastFlowLM is not installed")?;
    {
        let mut state = lock(&PULL);
        if state.busy {
            return Err("a download is running already".into());
        }
        *state = Pulling { busy: true, tag: tag.to_owned(), line: "starting ...".into(), error: None };
    }
    let mut child = match command(&path).args(["pull", tag]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            *lock(&PULL) = Pulling { busy: false, tag: tag.to_owned(), line: String::new(), error: Some(format!("cannot start {}: {e}", path.display())) };
            return Err(format!("cannot start {}: {e}", path.display()));
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    *lock(&PULL_CHILD) = Some(child);
    let tag = tag.to_owned();
    let reader = move |stream: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            if let Some(s) = stream {
                // the progress is redrawn in place with carriage returns: both end a "line"
                let mut buf = Vec::new();
                for byte in BufReader::new(s).bytes().map_while(Result::ok) {
                    if byte == b'\n' || byte == b'\r' {
                        let line = strip_ansi(&String::from_utf8_lossy(&buf)).trim().to_owned();
                        if !line.is_empty() {
                            lock(&PULL).line = line.chars().take(200).collect();
                        }
                        buf.clear();
                    } else {
                        buf.push(byte);
                    }
                }
            }
        })
    };
    let r1 = reader(stdout.map(|s| Box::new(s) as Box<dyn Read + Send>));
    let r2 = reader(stderr.map(|s| Box::new(s) as Box<dyn Read + Send>));
    std::thread::Builder::new()
        .name("fllm-flm-pull".into())
        .spawn(move || {
            let status = loop {
                let done = {
                    let mut guard = lock(&PULL_CHILD);
                    match guard.as_mut() {
                        Some(c) => c.try_wait().ok().flatten(),
                        None => break None, // cancelled
                    }
                };
                if let Some(s) = done {
                    break Some(s);
                }
                std::thread::sleep(Duration::from_millis(200));
            };
            let _ = r1.join();
            let _ = r2.join();
            *lock(&PULL_CHILD) = None;
            let mut state = lock(&PULL);
            state.busy = false;
            state.tag = tag;
            match status {
                Some(s) if s.success() => state.line = "done".into(),
                Some(s) => state.error = Some(format!("flm pull ended with {s}: {}", state.line)),
                None => state.error = Some("cancelled".into()),
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Ends a running download.
pub fn cancel_pull() {
    if let Some(mut child) = lock(&PULL_CHILD).take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Stops everything this module started (the server shutting down).
pub fn shutdown() {
    cancel_pull();
    stop();
}

// ---------------------------------------------------------------------------------------------------------------
// The state for the Studio and the command line
// ---------------------------------------------------------------------------------------------------------------

/// The facts the state is derived from.
#[derive(Debug, Clone, Default)]
pub struct Facts {
    pub npu_found: bool,
    pub other_npu_found: bool,
    pub flm_installed: bool,
    pub validation_ok: Option<bool>,
}

/// `no_npu`, `flm_missing`, `needs_attention` (the NPU stack reported a problem), `ready`.
pub fn state_of(f: &Facts) -> &'static str {
    if !f.npu_found && !f.flm_installed {
        return "no_npu";
    }
    if !f.flm_installed {
        return "flm_missing";
    }
    if f.validation_ok == Some(false) {
        return "needs_attention";
    }
    "ready"
}

fn install_hint() -> Value {
    if cfg!(windows) {
        json!({
            "text": "Install FastFlowLM with its installer (flm-setup.msi from the release page). It needs the current AMD Ryzen AI driver (NPU driver 32.0.203.x or newer).",
            "url": RELEASES,
        })
    } else {
        json!({
            "text": "Install the FastFlowLM package for your distribution (a .deb on Ubuntu / Debian) from the release page. It uses the amdxdna driver of the kernel (Linux 6.14 or newer) and needs a memlock limit of unlimited for your user (an `fwuser - memlock unlimited` line in /etc/security/limits.d/ and a new login).",
            "url": RELEASES,
        })
    }
}

/// The JSON of `GET /api/studio/npu/amd`.
pub fn status_json() -> Value {
    let detected = probe::detect();
    let amd: Vec<&probe::Detected> = detected.iter().filter(|d| d.vendor == Vendor::Amd).collect();
    let path = flm_path();
    let (mut flm_version, mut validation, mut models, mut error) = (None, None, Vec::new(), None);
    if let Some(p) = &path {
        flm_version = version(p);
        validation = run(p, &["validate"], COMMAND_TIMEOUT).ok().map(|t| parse_validate(&t));
        match list_models(p) {
            Ok(m) => models = m,
            Err(e) => error = Some(e),
        }
    }
    let facts = Facts {
        npu_found: !amd.is_empty(),
        other_npu_found: detected.iter().any(|d| d.vendor != Vendor::Amd),
        flm_installed: path.is_some(),
        validation_ok: validation.as_ref().map(|v| v.ok),
    };
    let state = state_of(&facts);
    let serving = serving();
    let pull = pulling();
    json!({
        "state": state,
        "provider_id": PROVIDER_ID,
        "hardware": amd.iter().map(|d| json!({"name": d.name, "source": d.source, "problem": d.problem})).collect::<Vec<_>>(),
        "flm": {
            "installed": path.is_some(),
            "path": path.as_ref().map(|p| p.display().to_string()),
            "version": flm_version,
            "install": install_hint(),
        },
        "validation": validation.as_ref().map(|v| json!({"ok": v.ok, "problems": v.problems, "lines": v.lines})),
        "models": models.iter().map(|m| json!({"tag": m.tag, "installed": m.installed})).collect::<Vec<_>>(),
        "serving": match &serving {
            Serving::Idle => json!({"state": "idle"}),
            Serving::Starting { tag } => json!({"state": "starting", "model": tag}),
            Serving::Running { tag, port } => json!({"state": "running", "model": tag, "port": port, "base_url": format!("http://127.0.0.1:{port}/v1"), "uptime_seconds": uptime_seconds()}),
            Serving::Failed { tag, error } => json!({"state": "failed", "model": tag, "error": error}),
        },
        "pull": {"busy": pull.busy, "model": pull.tag, "line": pull.line, "error": pull.error},
        "error": error,
        "native": native_status(),
    })
}

/// State of fLLM's own XDNA kernels (`--device npu`), shown on the same card as FastFlowLM.
fn native_status() -> Value {
    #[cfg(feature = "npu-xdna-native")]
    {
        super::xdna::native::status()
    }
    #[cfg(not(feature = "npu-xdna-native"))]
    {
        json!({"compiled": false})
    }
}

// ---------------------------------------------------------------------------------------------------------------
// fllm-rs amd-npu
// ---------------------------------------------------------------------------------------------------------------

pub const HELP: &str = "usage: fllm-rs amd-npu status | list | pull MODEL | serve MODEL [--register]
Runs models on the NPU of an AMD Ryzen AI processor through FastFlowLM (https://github.com/FastFlowLM/FastFlowLM), which you install yourself.
  status           what was found: the NPU, FastFlowLM, its check of the NPU stack, the models
  list             the models FastFlowLM offers (a check mark = downloaded)
  pull MODEL       download a model (for example llama3.2:3b)
  serve MODEL      run it until you press Ctrl-C and print the address of its OpenAI-compatible interface;
                   --register also lists it in the Studio as the provider \"AMD NPU (FastFlowLM)\" while it runs
  build-shim       builds the small helper library over XRT that fLLM's own NPU kernels need (C++ compiler and libxrt-dev required)
  native-bench [ROWS COLS BATCH FORMAT [ITERS]]
                   times one projection on the NPU with fLLM's own kernels against the CPU kernels (default 3072 3072 512 q4_k);
                   needs the XDNA shim, see docs/npu-xdna.md
FLLM_FLM_PATH names the flm program when it is not on the PATH.";

pub fn cli(args: &[String], register: &dyn Fn(&str, &str) -> Result<(), String>, unregister: &dyn Fn()) -> Result<(), String> {
    let sub = args.first().map(String::as_str).unwrap_or("status");
    match sub {
        "-h" | "--help" | "help" => {
            println!("{HELP}");
            Ok(())
        }
        "status" => {
            println!("{}", serde_json::to_string_pretty(&status_json()).map_err(|e| e.to_string())?);
            Ok(())
        }
        "build-shim" => {
            #[cfg(feature = "npu-xdna-native")]
            {
                let path = super::xdna::native::shim::build_from_embedded()?;
                println!("built {}", path.display());
                Ok(())
            }
            #[cfg(not(feature = "npu-xdna-native"))]
            Err("this build has no native XDNA backend".to_owned())
        }
        "native-bench" => {
            #[cfg(feature = "npu-xdna-native")]
            {
                let num = |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
                let format = args.get(4).map_or("q4_k", String::as_str);
                println!("{}", super::xdna::native::bench::run(num(1, 3072), num(2, 3072), num(3, 512), format, num(5, 5))?);
                Ok(())
            }
            #[cfg(not(feature = "npu-xdna-native"))]
            Err("this build has no native XDNA backend".to_owned())
        }
        "list" => {
            let path = flm_path().ok_or("FastFlowLM is not installed (see `fllm-rs amd-npu status`)")?;
            for m in list_models(&path)? {
                println!("{} {}", if m.installed { "[x]" } else { "[ ]" }, m.tag);
            }
            Ok(())
        }
        "pull" => {
            let tag = args.get(1).ok_or(HELP)?;
            pull(tag)?;
            let mut last = String::new();
            loop {
                std::thread::sleep(Duration::from_millis(500));
                let p = pulling();
                if p.line != last {
                    println!("{}", p.line);
                    last = p.line.clone();
                }
                if !p.busy {
                    return match p.error {
                        Some(e) => Err(e),
                        None => Ok(()),
                    };
                }
            }
        }
        "serve" => {
            let tag = args.get(1).ok_or(HELP)?.clone();
            let registered = args.iter().any(|a| a == "--register");
            start(&tag)?;
            println!("starting {tag} on the NPU (the first start of a model can take minutes) ...");
            loop {
                std::thread::sleep(Duration::from_millis(500));
                match serving() {
                    Serving::Running { port, .. } => {
                        let base = format!("http://127.0.0.1:{port}/v1");
                        println!("ready: {base}   (Ctrl-C stops it)");
                        if registered {
                            register(&tag, &base)?;
                        }
                        loop {
                            std::thread::sleep(Duration::from_secs(2));
                            if !matches!(serving(), Serving::Running { .. }) {
                                break;
                            }
                        }
                        if registered {
                            unregister();
                        }
                        return Err("flm stopped".into());
                    }
                    Serving::Failed { error, .. } => return Err(error),
                    _ => {}
                }
            }
        }
        other => Err(format!("unknown command {other}\n{HELP}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_codes_and_carriage_returns_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[32m[Linux]  NPU\u{1b}[0m\r\n"), "[Linux]  NPU\n");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn the_validation_names_the_problems() {
        let good = parse_validate("[Linux]  Kernel: 7.0.0\n[Linux]  NPU: /dev/accel/accel0 with 8 columns\n[Linux]  NPU platform: aie2p\n");
        assert!(good.ok);
        assert!(good.problems.is_empty());
        let bad = parse_validate("[Linux]  Kernel: 7.0.0\n[ERROR]  Memlock limit is too low (8MB). Please raise the limit or set to infinity.\n");
        assert!(!bad.ok);
        assert_eq!(bad.problems, vec!["Memlock limit is too low (8MB). Please raise the limit or set to infinity."]);
        assert!(!parse_validate("").ok, "no output is not a pass");
    }

    #[test]
    fn the_model_list_marks_the_installed_ones() {
        let text = "Models:\n  - d3:lite \u{23EC}\n  - llama3.2:3b \u{2705}\n  - llama3.1:8b \u{2705}\n  - qwen3:0.6b \u{23EC}\nnot a model line\n  - bad;tag \u{2705}\n";
        let models = parse_list(text);
        assert_eq!(models.len(), 4, "the line with a semicolon is not a tag");
        assert!(!models[0].installed && models[0].tag == "d3:lite");
        assert!(models[1].installed && models[1].tag == "llama3.2:3b");
        assert!(models[2].installed);
    }

    #[test]
    fn tags_are_checked() {
        for ok in ["llama3.2:3b", "gpt-oss:20b", "qwen3-it:4b", "embed-gemma:300m"] {
            assert!(valid_tag(ok), "{ok}");
        }
        for bad in ["", "a b", "a;b", "$(x)", "../x", "a/b", "x\n", &"a".repeat(81)] {
            assert!(!valid_tag(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_state_follows_the_facts() {
        assert_eq!(state_of(&Facts::default()), "no_npu");
        assert_eq!(state_of(&Facts { npu_found: true, ..Facts::default() }), "flm_missing");
        assert_eq!(state_of(&Facts { npu_found: true, flm_installed: true, validation_ok: Some(false), ..Facts::default() }), "needs_attention");
        assert_eq!(state_of(&Facts { npu_found: true, flm_installed: true, validation_ok: Some(true), ..Facts::default() }), "ready");
        assert_eq!(state_of(&Facts { flm_installed: true, ..Facts::default() }), "ready", "an installed flm is shown even when the system query found no NPU");
    }

    #[cfg(unix)]
    #[test]
    fn a_stand_in_for_flm_is_asked() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("fllm-flm-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let flm = dir.join("flm");
        // the real `flm list` marks the models with U+2705 / U+23EC; the stand-in prints the same UTF-8 bytes
        let script = r#"#!/bin/sh
case "$1" in
  version) echo 'FLM v9.9.9';;
  validate) echo '[Linux]  NPU platform: aie2p';;
  list) printf 'Models:\n  - llama3.2:3b \342\234\205\n  - gemma3:4b \342\217\254\n';;
esac
"#;
        std::fs::write(&flm, script).unwrap();
        std::fs::set_permissions(&flm, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(version(&flm).as_deref(), Some("v9.9.9"));
        let models = list_models(&flm).unwrap();
        assert_eq!(models, vec![Model { tag: "llama3.2:3b".into(), installed: true }, Model { tag: "gemma3:4b".into(), installed: false }]);
        assert!(parse_validate(&run(&flm, &["validate"], COMMAND_TIMEOUT).unwrap()).ok);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
