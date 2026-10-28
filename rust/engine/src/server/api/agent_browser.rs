//! The in-app browser of the coding agent: a real Chromium-family browser (Chrome, Chromium, Edge, Brave) driven
//! over the Chrome DevTools Protocol, so the agent can open a page, read it, click, type, scroll and take screenshots,
//! while the user watches the same page in the Code page and can take over.
//!
//! - **Off until the user switches it on** (Settings > Browser). Every tool is permission category `network`.
//! - The browser is started on first use, headless by default, with its own profile folder (never the user's own
//!   browser profile), a debugging port on 127.0.0.1 only, no extensions, downloads denied, permission prompts denied.
//!   It dies with the server (a process group watchdog on Unix, a job object on Windows).
//! - Every navigation and every request is checked against the URL policy (`agent_browser_policy`): http and https
//!   only, no link-local or metadata addresses, loopback and the local network only as the settings allow, and a
//!   public page cannot reach into the local machine.
//! - The routes for the panel (`/api/studio/agent/browser/*`) stay on this machine like the other agent routes: the
//!   browser can hold logged-in sessions.
use super::agent_browser_policy::{self as policy, Reach, Settings, UrlPolicy};
use super::agent_browser_snap::{self as snap, RefMap, SnapshotOptions};
use super::agent_ws::{Cdp, Event};
use super::*;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Mutex, OnceLock, Weak,
    },
    time::{Duration, Instant},
};

const VIEW_W: u32 = 1280;
const VIEW_H: u32 = 800;
const CALL: Duration = Duration::from_secs(20);
const NAV_WAIT: Duration = Duration::from_secs(20);
const CLICK_WAIT: Duration = Duration::from_secs(8);
const IDLE_LIMIT: Duration = Duration::from_secs(20 * 60);
const SNAPSHOT_CHARS: usize = 9000;
const MAX_ISSUES: usize = 30;

pub(super) const OFF_MESSAGE: &str = "the browser is switched off. Ask the user to turn on 'Let the agent use a browser' in Settings > Browser; until then use web_fetch to read pages.";

// ---- the tools --------------------------------------------------------------------------------------------------

/// The browser tools for the catalog (the same shape as the other server tools); permission category `network`.
pub(super) fn catalog() -> Vec<Value> {
    let tool = |name: &str, description: &str, properties: Value, required: &[&str]| {
        json!({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}},
            "permission": "network",
        })
    };
    vec![
        tool(
            "browser_open",
            "Open a web page (http or https) in a real browser that the user can watch, and return a text snapshot of it. Local dev servers (http://localhost:3000) work when a workspace is open. The result lists console errors and failed requests of the page. Use browser_snapshot to read the page again, browser_click and browser_type to operate it.",
            json!({
                "url": {"type": "string", "description": "http:// or https:// address."},
                "new_tab": {"type": "boolean", "description": "Open it in a new tab instead of the current one."},
            }),
            &["url"],
        ),
        tool(
            "browser_snapshot",
            "Read the current page as text: title, URL, headings, links, buttons and fields. Every element you can operate has a [ref=N]; pass N to browser_click, browser_type, browser_select or browser_scroll. Refs stay valid while the page stays on the same document. Take a new snapshot after the page changed.",
            json!({}),
            &[],
        ),
        tool(
            "browser_click",
            "Click an element of the page by its ref from browser_snapshot (links, buttons, checkboxes, tabs ...). Waits for the page to settle and returns the new snapshot.",
            json!({"ref": {"type": "integer", "description": "The number after ref= in the snapshot."}}),
            &["ref"],
        ),
        tool(
            "browser_type",
            "Type text into a field (input, textarea, editable element) by its ref. The text replaces what is in the field. With submit=true Enter is pressed afterwards and the new snapshot returned.",
            json!({
                "ref": {"type": "integer"},
                "text": {"type": "string"},
                "submit": {"type": "boolean", "description": "Press Enter after typing (submits a search or a form)."},
            }),
            &["ref", "text"],
        ),
        tool(
            "browser_select",
            "Choose an option of a <select> drop-down by its ref, by the visible text or the value of the option.",
            json!({"ref": {"type": "integer"}, "value": {"type": "string"}}),
            &["ref", "value"],
        ),
        tool(
            "browser_press",
            "Press a key (Enter, Tab, Escape, Backspace, ArrowDown, PageDown, F5 ... or a character) with optional modifiers, for example Control+a or Shift+Tab. It goes to the element that has the focus.",
            json!({"key": {"type": "string"}}),
            &["key"],
        ),
        tool(
            "browser_scroll",
            "Scroll the page: direction down, up, top or bottom (amount in pixels, default one screen), or bring the element with the given ref into view.",
            json!({
                "direction": {"type": "string", "enum": ["down", "up", "top", "bottom"]},
                "amount": {"type": "integer", "description": "Pixels for up and down (default 700)."},
                "ref": {"type": "integer", "description": "Scroll this element into view instead."},
            }),
            &[],
        ),
        tool("browser_back", "Go back one page in the history of the current tab and return the snapshot.", json!({}), &[]),
        tool(
            "browser_screenshot",
            "Take a screenshot (PNG) of the visible part of the page, of the whole page (full_page) or of one element (ref). A vision model gets the picture; a text-only model gets the description only, so use browser_snapshot to read the page.",
            json!({
                "full_page": {"type": "boolean", "description": "The whole page instead of the visible part (limited to 8000 px of height)."},
                "ref": {"type": "integer", "description": "Only this element."},
            }),
            &[],
        ),
        tool(
            "browser_tabs",
            "List the open tabs; with select (the number from the list) switch to that tab.",
            json!({"select": {"type": "integer", "description": "Tab number to switch to."}}),
            &[],
        ),
        tool(
            "browser_close",
            "Close a tab (default: the current one), or with all=true close the browser completely.",
            json!({"tab": {"type": "integer", "description": "Tab number (default: the current tab)."}, "all": {"type": "boolean"}}),
            &[],
        ),
    ]
}

/// Whether the user has switched the browser on.
pub(super) fn enabled() -> bool {
    policy::read_settings().enabled
}

fn is_tool(name: &str) -> bool {
    catalog().iter().any(|t| t["function"]["name"] == name)
}

pub(super) struct Reply {
    pub text: String,
    /// A `data:image/png;base64,...` URL for vision models.
    pub image: Option<String>,
}

impl Reply {
    fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), image: None }
    }
}

/// `browser_*` tool call; `None` when the name is not a browser tool. `workspace` tells whether the Code page has a workspace.
pub(super) async fn run_tool(workspace: bool, name: &str, args: &Value) -> Option<(Result<String, String>, Option<String>)> {
    if !name.starts_with("browser_") {
        return None;
    }
    Some(match run(workspace, name, args).await {
        Ok(reply) => (Ok(reply.text), reply.image),
        Err(error) => (Err(error), None),
    })
}

async fn run(workspace: bool, name: &str, args: &Value) -> Result<Reply, String> {
    let settings = policy::read_settings();
    if !settings.enabled {
        return Err(OFF_MESSAGE.into());
    }
    if !is_tool(name) {
        return Err(format!("unknown tool {name}"));
    }
    if name == "browser_close" && args.get("all").and_then(Value::as_bool) == Some(true) {
        shutdown().await;
        return Ok(Reply::text("The browser is closed."));
    }
    let session = session(&settings, workspace).await?;
    let _turn = session.action.lock().await;
    session.touch();
    match tokio::time::timeout(Duration::from_secs(90), session.dispatch(name, args)).await {
        Ok(result) => result,
        Err(_) => Err("the browser action took longer than 90 s and was given up".into()),
    }
}

// ---- state ------------------------------------------------------------------------------------------------------

struct Tab {
    target_id: String,
    session_id: String,
    ready: bool,
    refs: RefMap,
    url: String,
    title: String,
    /// Requests in flight: id -> url.
    inflight: HashMap<String, String>,
    last_activity: Instant,
    issues: Vec<String>,
    /// HTTP status of the last main document.
    status: Option<u16>,
    /// The page is on this machine or the local network: it may load local resources.
    main_local: bool,
}

impl Tab {
    fn new(target_id: &str, session_id: &str) -> Self {
        Self {
            target_id: target_id.to_owned(),
            session_id: session_id.to_owned(),
            ready: false,
            refs: RefMap::default(),
            url: String::new(),
            title: String::new(),
            inflight: HashMap::new(),
            last_activity: Instant::now(),
            issues: Vec::new(),
            status: None,
            main_local: true,
        }
    }
    fn issue(&mut self, text: String) {
        if self.issues.last() != Some(&text) && self.issues.len() < MAX_ISSUES {
            self.issues.push(text);
        }
    }
}

struct BrowserState {
    tabs: Vec<Tab>,
    current: usize,
    workspace: bool,
    local: policy::LocalAddresses,
    dns: HashMap<String, (Instant, Reach)>,
}

struct Session {
    cdp: Cdp,
    child: Mutex<Option<Child>>,
    temp_profile: Option<PathBuf>,
    state: Mutex<BrowserState>,
    /// One tool call at a time.
    action: tokio::sync::Mutex<()>,
    last_used: Mutex<Instant>,
    headed: bool,
    closing: AtomicBool,
}

fn slot() -> &'static tokio::sync::Mutex<Option<Arc<Session>>> {
    static SLOT: OnceLock<tokio::sync::Mutex<Option<Arc<Session>>>> = OnceLock::new();
    SLOT.get_or_init(|| tokio::sync::Mutex::new(None))
}

/// The running session, started when there is none (or the old one died).
async fn session(settings: &Settings, workspace: bool) -> Result<Arc<Session>, String> {
    let mut guard = slot().lock().await;
    if let Some(existing) = guard.as_ref() {
        if existing.alive() && existing.headed == settings.show_window {
            existing.configure(settings, workspace);
            return Ok(existing.clone());
        }
        let old = guard.take();
        if let Some(old) = old {
            old.kill().await;
        }
    }
    let fresh = launch(settings, workspace).await?;
    *guard = Some(fresh.clone());
    Ok(fresh)
}

/// The running session, if any (no launch).
async fn running() -> Option<Arc<Session>> {
    slot().lock().await.as_ref().filter(|s| s.alive()).cloned()
}

/// Closes the browser (also called when the server stops and when the agent session ends).
pub(in crate::server) async fn shutdown() {
    let old = slot().lock().await.take();
    if let Some(old) = old {
        old.kill().await;
    }
}

#[cfg(unix)]
fn is_root() -> bool {
    extern "C" {
        fn geteuid() -> u32;
    }
    // SAFETY: geteuid takes no arguments and cannot fail.
    unsafe { geteuid() == 0 }
}

#[cfg(not(unix))]
fn is_root() -> bool {
    false
}

/// Makes the browser die with the server, also when the server is killed.
#[cfg(unix)]
fn guard_child(child: &Child, temp: Option<&Path>) {
    // A shell that outlives the server notices that it is gone and kills the browser's process group (the browser was
    // started as the leader of its own group). It ignores INT/HUP/TERM so that a Ctrl-C in the terminal does not take it down first.
    let script = r#"trap '' INT HUP TERM; while kill -0 "$1" 2>/dev/null && kill -0 "$2" 2>/dev/null; do sleep 1; done; kill -s KILL -- "-$2" 2>/dev/null; kill -s KILL "$2" 2>/dev/null; if [ -n "$3" ]; then sleep 1; rm -rf "$3"; fi; exit 0"#;
    let watchdog = Command::new("sh")
        .arg("-c")
        .arg(script)
        .arg("fllm-browser-watchdog")
        .arg(std::process::id().to_string())
        .arg(child.id().to_string())
        .arg(temp.map(|t| t.display().to_string()).unwrap_or_default())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Ok(mut watchdog) = watchdog {
        // reaped when it ends, so that it does not stay a zombie of this process
        std::thread::spawn(move || {
            let _ = watchdog.wait();
        });
    }
}

#[cfg(windows)]
fn guard_child(child: &Child, _temp: Option<&Path>) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    // One job object for the life of the server, never closed by hand: Windows closes it when the server process ends
    // (however it ends), and the job kills the browser with it.
    static JOB: OnceLock<usize> = OnceLock::new();
    let job = *JOB.get_or_init(|| {
        // SAFETY: plain data passed by pointer with its exact size; the handle stays open on purpose.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let size = std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32;
            if SetInformationJobObject(job, JobObjectExtendedLimitInformation, (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), size) == 0 {
                return 0;
            }
            job as usize
        }
    });
    if job != 0 {
        // SAFETY: both handles are valid; the child handle belongs to `child`.
        unsafe {
            AssignProcessToJobObject(job as *mut std::ffi::c_void, child.as_raw_handle() as *mut std::ffi::c_void);
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn guard_child(_child: &Child, _temp: Option<&Path>) {}

async fn launch(settings: &Settings, workspace: bool) -> Result<Arc<Session>, String> {
    let exe = policy::find_browser(settings)?;
    let mut attempts: Vec<(PathBuf, bool)> = Vec::new();
    if settings.persistent_profile {
        attempts.push((policy::profile_dir(), false));
    }
    let mut random = [0u8; 6];
    let _ = getrandom::fill(&mut random);
    let temp: String = random.iter().map(|b| format!("{b:02x}")).collect();
    attempts.push((std::env::temp_dir().join(format!("fllm-browser-{}-{temp}", std::process::id())), true));
    let mut last = String::new();
    for (dir, is_temp) in attempts {
        match start(&exe, &dir, is_temp, settings, workspace).await {
            Ok(session) => return Ok(session),
            Err(error) => {
                last = error;
                if is_temp {
                    let _ = std::fs::remove_dir_all(&dir);
                }
            }
        }
    }
    Err(last)
}

async fn start(exe: &Path, dir: &Path, is_temp: bool, settings: &Settings, workspace: bool) -> Result<Arc<Session>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create the browser profile folder {}: {e}", dir.display()))?;
    let _ = std::fs::remove_file(dir.join("DevToolsActivePort"));
    let args = policy::launch_args(dir, settings.show_window, is_root(), VIEW_W, VIEW_H);
    let mut command = Command::new(exe);
    command.args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|e| format!("cannot start the browser {}: {e}", exe.display()))?;
    guard_child(&child, is_temp.then_some(dir));
    let port_file = dir.join("DevToolsActivePort");
    let started = Instant::now();
    let (port, path) = loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "the browser {} exited right after it was started ({status}). Another copy may be using the profile folder {}, or the browser cannot run here (in a container it needs --no-sandbox, which is added when running as root)",
                exe.display(),
                dir.display()
            ));
        }
        if let Some(found) = std::fs::read_to_string(&port_file).ok().and_then(|t| policy::parse_active_port(&t)) {
            break found;
        }
        if started.elapsed() > Duration::from_secs(25) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("the browser did not report its debugging port within 25 s".into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    let cdp = match Cdp::connect("127.0.0.1", port, &path).await {
        Ok(cdp) => cdp,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let session = Arc::new(Session {
        cdp,
        child: Mutex::new(Some(child)),
        temp_profile: is_temp.then(|| dir.to_path_buf()),
        state: Mutex::new(BrowserState { tabs: Vec::new(), current: 0, workspace, local: settings.local, dns: HashMap::new() }),
        action: tokio::sync::Mutex::new(()),
        last_used: Mutex::new(Instant::now()),
        headed: settings.show_window,
        closing: AtomicBool::new(false),
    });
    // events first, so that nothing between the setup and the first tab is missed
    let events = session.cdp.subscribe();
    tokio::spawn(pump(Arc::downgrade(&session), events));
    session.browser_setup().await;
    Ok(session)
}

/// The events of the browser: tab bookkeeping, console and network problems, dialogs, and the request gate.
async fn pump(weak: Weak<Session>, mut events: tokio::sync::broadcast::Receiver<Event>) {
    use tokio::sync::broadcast::error::RecvError;
    loop {
        let received = tokio::time::timeout(Duration::from_secs(30), events.recv()).await;
        let Some(session) = weak.upgrade() else { break };
        // nobody used the browser for a long time: let it go (the next use starts it again)
        if session.idle_for() > IDLE_LIMIT {
            drop(session);
            tokio::spawn(shutdown());
            break;
        }
        match received {
            Err(_) | Ok(Err(RecvError::Lagged(_))) => {}
            Ok(Err(RecvError::Closed)) => break,
            Ok(Ok(event)) => session.on_event(event),
        }
    }
}

impl Session {
    fn st(&self) -> std::sync::MutexGuard<'_, BrowserState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn alive(&self) -> bool {
        if self.cdp.is_closed() || self.closing.load(AtomicOrdering::Relaxed) {
            return false;
        }
        let mut child = self.child.lock().unwrap_or_else(|e| e.into_inner());
        match child.as_mut() {
            Some(c) => matches!(c.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn current_url(&self) -> String {
        let st = self.st();
        st.tabs.get(st.current).map(|t| t.url.clone()).unwrap_or_default()
    }

    fn touch(&self) {
        *self.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        self.last_used.lock().unwrap_or_else(|e| e.into_inner()).elapsed()
    }

    fn configure(&self, settings: &Settings, workspace: bool) {
        let mut st = self.st();
        st.workspace = workspace;
        st.local = settings.local;
    }

    fn url_policy(&self) -> UrlPolicy {
        let st = self.st();
        UrlPolicy { local: st.local, workspace: st.workspace }
    }

    async fn cmd(&self, method: &str, params: Value, sid: Option<&str>) -> Result<Value, String> {
        self.cdp.call(method, params, sid, CALL).await
    }

    async fn kill(&self) {
        self.closing.store(true, AtomicOrdering::Relaxed);
        // a clean close lets the browser write cookies to the profile
        let _ = self.cdp.call("Browser.close", json!({}), None, Duration::from_secs(3)).await;
        self.cdp.close().await;
        for _ in 0..30 {
            let done = {
                let mut child = self.child.lock().unwrap_or_else(|e| e.into_inner());
                child.as_mut().is_none_or(|c| !matches!(c.try_wait(), Ok(None)))
            };
            if done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if let Some(mut child) = self.child.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(dir) = &self.temp_profile {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// Browser-wide settings: no downloads, no permission prompts, new tabs set up before they load anything.
    async fn browser_setup(self: &Arc<Self>) {
        let _ = self.cmd("Browser.setDownloadBehavior", json!({"behavior": "deny"}), None).await;
        for name in ["geolocation", "notifications", "midi", "midiSysex", "audioCapture", "videoCapture", "clipboardReadWrite", "displayCapture", "idleDetection", "sensors", "persistentStorage"] {
            let _ = self.cmd("Browser.setPermission", json!({"permission": {"name": name}, "setting": "denied"}), None).await;
        }
        // every page that appears is attached, set up and only then released (it waits for the debugger)
        let _ = self
            .cmd("Target.setAutoAttach", json!({"autoAttach": true, "waitForDebuggerOnStart": true, "flatten": true}), None)
            .await;
        let _ = self.cmd("Target.setDiscoverTargets", json!({"discover": true}), None).await;
    }

    fn on_event(self: &Arc<Self>, event: Event) {
        let me = self.clone();
        match event.method.as_str() {
            "Target.attachedToTarget" => {
                tokio::spawn(async move { me.attached(event.params).await });
            }
            "Target.targetDestroyed" => {
                let id = event.params["targetId"].as_str().unwrap_or_default().to_owned();
                let mut st = self.st();
                if let Some(at) = st.tabs.iter().position(|t| t.target_id == id) {
                    st.tabs.remove(at);
                    if st.current >= st.tabs.len() {
                        st.current = st.tabs.len().saturating_sub(1);
                    } else if at < st.current {
                        st.current -= 1;
                    }
                }
            }
            "Fetch.requestPaused" => {
                if let Some(sid) = event.session {
                    tokio::spawn(async move { me.gate(&sid, event.params).await });
                }
            }
            "Page.javascriptDialogOpening" => {
                let Some(sid) = event.session else { return };
                let kind = event.params["type"].as_str().unwrap_or("alert").to_owned();
                let message = event.params["message"].as_str().unwrap_or_default().to_owned();
                self.with_tab(&sid, |t| t.issue(format!("a {kind} dialog appeared and was {}: \"{}\"", if kind == "alert" || kind == "beforeunload" { "accepted" } else { "dismissed" }, clip(&message, 160))));
                tokio::spawn(async move {
                    let accept = kind == "alert" || kind == "beforeunload";
                    let _ = me.cmd("Page.handleJavaScriptDialog", json!({"accept": accept}), Some(&sid)).await;
                });
            }
            _ => {
                if let Some(sid) = event.session.as_deref() {
                    self.tab_event(sid, &event.method, &event.params);
                }
            }
        }
    }

    fn with_tab<R>(&self, sid: &str, f: impl FnOnce(&mut Tab) -> R) -> Option<R> {
        let mut st = self.st();
        st.tabs.iter_mut().find(|t| t.session_id == sid).map(f)
    }

    fn tab_event(&self, sid: &str, method: &str, p: &Value) {
        self.with_tab(sid, |tab| match method {
            "Page.frameNavigated" => {
                if p["frame"].get("parentId").is_none() {
                    let url = p["frame"]["url"].as_str().unwrap_or_default().to_owned();
                    tab.refs.reset();
                    tab.main_local = policy::is_local_page(&url);
                    tab.url = url;
                    tab.inflight.clear();
                    tab.status = None;
                    tab.last_activity = Instant::now();
                }
            }
            "Page.navigatedWithinDocument" => {
                if let Some(url) = p["url"].as_str() {
                    tab.url = url.to_owned();
                }
            }
            "Network.requestWillBeSent" => {
                let url = p["request"]["url"].as_str().unwrap_or_default();
                if !url.starts_with("data:") {
                    tab.inflight.insert(p["requestId"].as_str().unwrap_or_default().to_owned(), url.to_owned());
                    tab.last_activity = Instant::now();
                }
            }
            "Network.loadingFinished" => {
                tab.inflight.remove(p["requestId"].as_str().unwrap_or_default());
                tab.last_activity = Instant::now();
            }
            "Network.loadingFailed" => {
                let id = p["requestId"].as_str().unwrap_or_default();
                let url = tab.inflight.remove(id).unwrap_or_default();
                tab.last_activity = Instant::now();
                let reason = p["errorText"].as_str().unwrap_or("failed");
                if p["canceled"].as_bool() != Some(true) && reason != "net::ERR_ABORTED" && !url.is_empty() && !url.ends_with("/favicon.ico") {
                    tab.issue(format!("request failed: {} ({reason})", clip(&url, 140)));
                }
            }
            "Network.responseReceived" => {
                let status = p["response"]["status"].as_u64().unwrap_or(0) as u16;
                let url = p["response"]["url"].as_str().unwrap_or_default();
                if p["type"] == "Document" && p["frameId"].as_str().is_some() && p["loaderId"].as_str().is_some() {
                    tab.status = Some(status);
                }
                if status >= 400 && !url.ends_with("/favicon.ico") {
                    tab.issue(format!("HTTP {status}: {}", clip(url, 140)));
                }
            }
            "Runtime.consoleAPICalled" => {
                if p["type"] == "error" {
                    let text = p["args"].as_array().map(|a| a.iter().map(remote_text).collect::<Vec<_>>().join(" ")).unwrap_or_default();
                    tab.issue(format!("console.error: {}", clip(&text, 200)));
                }
            }
            "Runtime.exceptionThrown" => {
                let d = &p["exceptionDetails"];
                let text = d["exception"]["description"].as_str().or_else(|| d["text"].as_str()).unwrap_or("uncaught exception");
                tab.issue(format!("uncaught exception: {}", clip(text.lines().next().unwrap_or(text), 200)));
            }
            "Log.entryAdded" => {
                let e = &p["entry"];
                if e["level"] == "error" && e["source"] != "network" {
                    tab.issue(format!("{}: {}", e["source"].as_str().unwrap_or("log"), clip(e["text"].as_str().unwrap_or_default(), 200)));
                }
            }
            _ => {}
        });
    }

    /// A new target is attached: pages are registered and set up, everything else is let go.
    async fn attached(self: Arc<Self>, p: Value) {
        let Some(sid) = p["sessionId"].as_str().map(str::to_owned) else { return };
        let info = &p["targetInfo"];
        let target_id = info["targetId"].as_str().unwrap_or_default().to_owned();
        let waiting = p["waitingForDebugger"].as_bool().unwrap_or(false);
        if info["type"] != "page" {
            if waiting {
                let _ = self.cmd("Runtime.runIfWaitingForDebugger", json!({}), Some(&sid)).await;
            }
            // frames and workers need no attention of ours
            let _ = self.cmd("Target.detachFromTarget", json!({"sessionId": sid}), None).await;
            return;
        }
        {
            let mut st = self.st();
            if let Some(known) = st.tabs.iter_mut().find(|t| t.target_id == target_id) {
                // the same page again under a new session (the browser moved it to another process): it keeps its place
                known.session_id = sid.clone();
                known.ready = false;
            } else {
                st.tabs.push(Tab::new(&target_id, &sid));
                // a page opened by a click becomes the current one: that is where the person expects to be
                if st.tabs.len() > 1 {
                    st.current = st.tabs.len() - 1;
                }
            }
        }
        // A page that waits for the debugger must not be able to hold the setup up: all commands go out together, each with a short limit.
        let agent = match self.cmd("Browser.getVersion", json!({}), None).await {
            // sites that turn headless browsers away look for this word
            Ok(version) => version["userAgent"].as_str().unwrap_or_default().replace("HeadlessChrome", "Chrome"),
            Err(_) => String::new(),
        };
        let mut setup: Vec<(&str, Value)> = vec![
            ("Page.enable", json!({})),
            ("Runtime.enable", json!({})),
            ("Network.enable", json!({})),
            ("Log.enable", json!({})),
            ("DOM.enable", json!({})),
            ("Accessibility.enable", json!({})),
            ("Fetch.enable", json!({"patterns": [{"urlPattern": "*"}]})),
            ("Emulation.setFocusEmulationEnabled", json!({"enabled": true})),
        ];
        if !self.headed {
            setup.push(("Emulation.setDeviceMetricsOverride", json!({"width": VIEW_W, "height": VIEW_H, "deviceScaleFactor": 1, "mobile": false})));
        }
        if !agent.is_empty() {
            setup.push(("Network.setUserAgentOverride", json!({"userAgent": agent})));
        }
        let calls = setup.into_iter().map(|(method, params)| self.cdp.call(method, params, Some(&sid), Duration::from_secs(6)));
        let _ = futures_util::future::join_all(calls).await;
        self.with_tab(&sid, |t| t.ready = true);
        if waiting {
            let _ = self.cmd("Runtime.runIfWaitingForDebugger", json!({}), Some(&sid)).await;
        }
    }

    // ---- the request gate -------------------------------------------------------------------------------------

    async fn reach_of_host(&self, url: &reqwest::Url) -> Option<Reach> {
        if let Some(reach) = policy::reach_by_name(url) {
            return Some(reach);
        }
        let host = url.host_str()?.to_ascii_lowercase();
        if let Some((at, reach)) = self.st().dns.get(&host) {
            if at.elapsed() < Duration::from_secs(60) {
                return Some(*reach);
            }
        }
        let port = url.port_or_known_default().unwrap_or(80);
        let found: Vec<std::net::IpAddr> = tokio::net::lookup_host((host.as_str(), port)).await.ok()?.map(|a| a.ip()).collect();
        if found.is_empty() {
            return None;
        }
        let reach = policy::worst_reach(&found);
        let mut st = self.st();
        if st.dns.len() > 512 {
            st.dns.clear();
        }
        st.dns.insert(host, (Instant::now(), reach));
        Some(reach)
    }

    /// Decides one request of a page: continue it or fail it.
    async fn gate(self: Arc<Self>, sid: &str, p: Value) {
        let id = p["requestId"].as_str().unwrap_or_default().to_owned();
        let url_text = p["request"]["url"].as_str().unwrap_or_default().to_owned();
        let kind = p["resourceType"].as_str().unwrap_or_default().to_owned();
        let verdict = self.verdict(sid, &url_text, &kind).await;
        match verdict {
            Ok(()) => {
                let _ = self.cmd("Fetch.continueRequest", json!({"requestId": id}), Some(sid)).await;
            }
            Err(reason) => {
                self.with_tab(sid, |t| t.issue(format!("blocked {}: {reason}", clip(&url_text, 120))));
                let _ = self.cmd("Fetch.failRequest", json!({"requestId": id, "errorReason": "BlockedByClient"}), Some(sid)).await;
            }
        }
    }

    async fn verdict(&self, sid: &str, url_text: &str, kind: &str) -> Result<(), String> {
        let Some(url) = judge_scheme(url_text, kind)? else { return Ok(()) };
        let Some(reach) = self.reach_of_host(&url).await else { return Ok(()) };
        let main_local = self.with_tab(sid, |t| t.main_local).unwrap_or(false);
        judge_reach(reach, kind, self.url_policy(), main_local)
    }

    // ---- helpers of the tools ---------------------------------------------------------------------------------

    /// The session id of the current tab; waits for the first tab and opens one when the browser has none.
    async fn current(&self) -> Result<String, String> {
        for round in 0..60 {
            let (ready, none) = {
                let st = self.st();
                (st.tabs.get(st.current).filter(|t| t.ready).map(|t| t.session_id.clone()), st.tabs.is_empty())
            };
            if let Some(sid) = ready {
                return Ok(sid);
            }
            if none && round == 25 {
                let _ = self.cmd("Target.createTarget", json!({"url": "about:blank"}), None).await;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err("the browser has no open tab".into())
    }

    async fn eval(&self, sid: &str, expression: &str) -> Result<Value, String> {
        let answer = self.cmd("Runtime.evaluate", json!({"expression": expression, "returnByValue": true, "awaitPromise": false}), Some(sid)).await?;
        if let Some(text) = answer["exceptionDetails"]["exception"]["description"].as_str() {
            return Err(text.to_owned());
        }
        Ok(answer["result"]["value"].clone())
    }

    /// Waits until the page has loaded and the network is quiet (or the time is up).
    async fn settle(&self, sid: &str, max: Duration) {
        let start = Instant::now();
        tokio::time::sleep(Duration::from_millis(150)).await;
        while start.elapsed() < max {
            let (inflight, quiet) = self.with_tab(sid, |t| (t.inflight.len(), t.last_activity.elapsed())).unwrap_or((0, Duration::from_secs(60)));
            let ready = self.eval(sid, "document.readyState").await.map(|v| v == "complete").unwrap_or(false);
            // streams and long polls never finish: a page that has been quiet for a while counts as settled
            if ready && ((inflight == 0 && quiet >= Duration::from_millis(400)) || quiet >= Duration::from_millis(1500)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn refresh_info(&self, sid: &str) -> (String, String) {
        let target = self.with_tab(sid, |t| t.target_id.clone()).unwrap_or_default();
        if let Ok(info) = self.cmd("Target.getTargetInfo", json!({"targetId": target}), None).await {
            let title = info["targetInfo"]["title"].as_str().unwrap_or_default().to_owned();
            let url = info["targetInfo"]["url"].as_str().unwrap_or_default().to_owned();
            self.with_tab(sid, |t| {
                t.title = title.clone();
                t.url = url.clone();
            });
            return (title, url);
        }
        self.with_tab(sid, |t| (t.title.clone(), t.url.clone())).unwrap_or_default()
    }

    fn take_issues(&self, sid: &str) -> Vec<String> {
        self.with_tab(sid, |t| std::mem::take(&mut t.issues)).unwrap_or_default()
    }

    async fn snapshot_text(&self, sid: &str, max_chars: usize) -> Result<String, String> {
        let tree = self.cmd("Accessibility.getFullAXTree", json!({}), Some(sid)).await?;
        let nodes = tree["nodes"].as_array().cloned().unwrap_or_default();
        let (title, url) = self.refresh_info(sid).await;
        let options = SnapshotOptions { title: &title, url: &url, max_chars };
        let mut st = self.st();
        let tab = st.tabs.iter_mut().find(|t| t.session_id == sid).ok_or("the tab is gone")?;
        Ok(snap::format_snapshot(&nodes, &mut tab.refs, &options))
    }

    /// The end of an action: what happened to the page, problems seen on the way, and optionally the snapshot.
    async fn conclude(&self, sid: &str, lead: &str, with_snapshot: bool) -> Result<Reply, String> {
        let mut out = lead.to_owned();
        let issues = self.take_issues(sid);
        let status = self.with_tab(sid, |t| t.status).flatten();
        if let Some(code) = status.filter(|c| *c >= 400) {
            out.push_str(&format!("\nThe server answered HTTP {code} for this page."));
        }
        if with_snapshot {
            out.push_str("\n\n");
            out.push_str(&self.snapshot_text(sid, SNAPSHOT_CHARS).await?);
        } else {
            let (title, url) = self.refresh_info(sid).await;
            out.push_str(&format!("\nPage: {}\nURL: {url}", clip(&title, 120)));
        }
        if !issues.is_empty() {
            out.push_str("\n\nProblems seen while this happened:");
            for issue in issues.iter().take(8) {
                out.push_str(&format!("\n- {issue}"));
            }
            if issues.len() > 8 {
                out.push_str(&format!("\n- ... and {} more", issues.len() - 8));
            }
        }
        Ok(Reply::text(out))
    }

    fn resolve(&self, sid: &str, args: &Value) -> Result<snap::RefTarget, String> {
        let n = ref_arg(args)?;
        self.with_tab(sid, |t| t.refs.get(n).cloned())
            .flatten()
            .ok_or_else(|| format!("ref {n} is not on the current page. Take a new browser_snapshot and use a ref from it."))
    }

    /// The centre of an element in viewport coordinates, after scrolling it into view.
    async fn center_of(&self, sid: &str, backend: i64) -> Result<(f64, f64, [f64; 4]), String> {
        let _ = self.cmd("DOM.scrollIntoViewIfNeeded", json!({"backendNodeId": backend}), Some(sid)).await;
        let quads = self
            .cmd("DOM.getContentQuads", json!({"backendNodeId": backend}), Some(sid))
            .await
            .map_err(|_| "the element is not visible (it may be hidden, or gone: take a new browser_snapshot)".to_owned())?;
        for quad in quads["quads"].as_array().cloned().unwrap_or_default() {
            let p: Vec<f64> = quad.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            if p.len() == 8 {
                let xs = [p[0], p[2], p[4], p[6]];
                let ys = [p[1], p[3], p[5], p[7]];
                let (x0, x1) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
                let (y0, y1) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
                if (x1 - x0) * (y1 - y0) > 1.0 {
                    return Ok(((x0 + x1) / 2.0, (y0 + y1) / 2.0, [x0, y0, x1 - x0, y1 - y0]));
                }
            }
        }
        Err("the element has no visible area (hidden or collapsed)".into())
    }

    async fn click_at(&self, sid: &str, x: f64, y: f64, count: u32) -> Result<(), String> {
        self.cmd("Input.dispatchMouseEvent", json!({"type": "mouseMoved", "x": x, "y": y}), Some(sid)).await?;
        self.cmd("Input.dispatchMouseEvent", json!({"type": "mousePressed", "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": count}), Some(sid)).await?;
        self.cmd("Input.dispatchMouseEvent", json!({"type": "mouseReleased", "x": x, "y": y, "button": "left", "buttons": 0, "clickCount": count}), Some(sid)).await?;
        Ok(())
    }

    async fn press_key(&self, sid: &str, spec: &str) -> Result<(), String> {
        for event in snap::key_events(spec)? {
            self.cmd("Input.dispatchKeyEvent", event, Some(sid)).await?;
        }
        Ok(())
    }

    /// Calls a JavaScript function with the DOM node of a ref as `this`.
    async fn call_on(&self, sid: &str, backend: i64, function: &str) -> Result<Value, String> {
        let node = self
            .cmd("DOM.resolveNode", json!({"backendNodeId": backend}), Some(sid))
            .await
            .map_err(|_| "the element is gone from the page: take a new browser_snapshot".to_owned())?;
        let object = node["object"]["objectId"].as_str().ok_or("the element cannot be reached")?.to_owned();
        let answer = self
            .cmd("Runtime.callFunctionOn", json!({"objectId": object, "functionDeclaration": function, "returnByValue": true}), Some(sid))
            .await?;
        if let Some(text) = answer["exceptionDetails"]["exception"]["description"].as_str() {
            return Err(text.to_owned());
        }
        Ok(answer["result"]["value"].clone())
    }

    /// A picture of the page. A tab that is not the active one of the browser (after another tab was closed) cannot be photographed:
    /// it is brought to the front and the picture is taken again.
    async fn capture(&self, sid: &str, params: Value) -> Result<Value, String> {
        match self.cmd("Page.captureScreenshot", params.clone(), Some(sid)).await {
            Err(e) if e.contains("Not attached to an active page") || e.contains("not visible") => {
                let target = self.with_tab(sid, |t| t.target_id.clone()).unwrap_or_default();
                self.cmd("Target.activateTarget", json!({"targetId": target}), None).await?;
                tokio::time::sleep(Duration::from_millis(200)).await;
                self.cmd("Page.captureScreenshot", params, Some(sid)).await
            }
            other => other,
        }
    }

    async fn page_count(&self) -> usize {
        self.cmd("Target.getTargets", json!({}), None)
            .await
            .ok()
            .and_then(|v| v["targetInfos"].as_array().map(|a| a.iter().filter(|t| t["type"] == "page").count()))
            .unwrap_or(0)
    }

    // ---- the tools --------------------------------------------------------------------------------------------

    async fn dispatch(&self, name: &str, args: &Value) -> Result<Reply, String> {
        match name {
            "browser_open" => self.open(args).await,
            "browser_snapshot" => {
                let sid = self.current().await?;
                self.take_issues(&sid);
                let text = self.snapshot_text(&sid, SNAPSHOT_CHARS * 2).await?;
                Ok(Reply::text(text))
            }
            "browser_click" => self.click(args).await,
            "browser_type" => self.type_text(args).await,
            "browser_select" => self.select(args).await,
            "browser_press" => {
                let sid = self.current().await?;
                self.take_issues(&sid);
                let key = args["key"].as_str().ok_or("browser_press needs a key")?;
                self.press_key(&sid, key).await?;
                self.settle(&sid, CLICK_WAIT).await;
                self.conclude(&sid, &format!("Pressed {key}."), true).await
            }
            "browser_scroll" => self.scroll(args).await,
            "browser_back" => self.back().await,
            "browser_screenshot" => self.screenshot(args).await,
            "browser_tabs" => self.tabs(args).await,
            "browser_close" => self.close_tab(args).await,
            other => Err(format!("unknown tool {other}")),
        }
    }

    async fn open(&self, args: &Value) -> Result<Reply, String> {
        let text = args["url"].as_str().ok_or("browser_open needs a url")?;
        let url = policy::check_url(text, self.url_policy()).await?;
        if args["new_tab"].as_bool() == Some(true) {
            let before = self.st().tabs.len();
            self.cmd("Target.createTarget", json!({"url": "about:blank"}), None).await?;
            for _ in 0..60 {
                if self.st().tabs.len() > before && self.st().tabs.last().is_some_and(|t| t.ready) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        let sid = self.current().await?;
        self.take_issues(&sid);
        let answer = self.cmd("Page.navigate", json!({"url": url.as_str()}), Some(&sid)).await?;
        if let Some(error) = answer["errorText"].as_str().filter(|e| !e.is_empty()) {
            let blocked = self.take_issues(&sid).into_iter().find(|i| i.starts_with("blocked "));
            return Err(match blocked {
                Some(why) => format!("the page was not opened: {why}"),
                None if error.contains("CONNECTION_REFUSED") => format!("could not open {url}: nothing is listening there ({error}). Is the server running?"),
                None => format!("could not open {url}: {error}"),
            });
        }
        self.settle(&sid, NAV_WAIT).await;
        self.conclude(&sid, &format!("Opened {url}."), true).await
    }

    async fn click(&self, args: &Value) -> Result<Reply, String> {
        let sid = self.current().await?;
        let target = self.resolve(&sid, args)?;
        self.take_issues(&sid);
        let (x, y, _) = self.center_of(&sid, target.backend).await?;
        let pages = self.page_count().await;
        self.click_at(&sid, x, y, 1).await?;
        self.settle(&sid, CLICK_WAIT).await;
        let label = if target.name.is_empty() { target.role.clone() } else { format!("{} \"{}\"", target.role, target.name) };
        let mut lead = format!("Clicked {label}.");
        if self.page_count().await > pages {
            tokio::time::sleep(Duration::from_millis(600)).await;
            lead.push_str(" The click opened a new tab, which is now the current one (browser_tabs lists them).");
            let now = self.current().await?;
            if now != sid {
                self.settle(&now, CLICK_WAIT).await;
                return self.conclude(&now, &lead, true).await;
            }
        }
        self.conclude(&sid, &lead, true).await
    }

    async fn type_text(&self, args: &Value) -> Result<Reply, String> {
        let sid = self.current().await?;
        let target = self.resolve(&sid, args)?;
        let text = args["text"].as_str().ok_or("browser_type needs text")?;
        self.take_issues(&sid);
        let _ = self.center_of(&sid, target.backend).await; // scrolls it into view
        let prepared = self
            .call_on(
                &sid,
                target.backend,
                "function(){ this.focus(); if (typeof this.select === 'function') { this.select(); return 'field'; } if (this.isContentEditable) { document.execCommand('selectAll'); return 'editable'; } return 'other'; }",
            )
            .await?;
        if prepared == "other" {
            return Err(format!("{} is not a field that takes text: choose a textbox, searchbox or combobox ref", target.role));
        }
        if text.is_empty() {
            self.press_key(&sid, "Delete").await?;
        } else {
            self.cmd("Input.insertText", json!({"text": text}), Some(&sid)).await?;
        }
        let label = if target.name.is_empty() { target.role.clone() } else { format!("\"{}\"", target.name) };
        if args["submit"].as_bool() == Some(true) {
            self.press_key(&sid, "Enter").await?;
            self.settle(&sid, CLICK_WAIT).await;
            return self.conclude(&sid, &format!("Typed into {label} and pressed Enter."), true).await;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
        self.conclude(&sid, &format!("Typed {} characters into {label}.", text.chars().count()), false).await
    }

    async fn select(&self, args: &Value) -> Result<Reply, String> {
        let sid = self.current().await?;
        let target = self.resolve(&sid, args)?;
        let want = args["value"].as_str().ok_or("browser_select needs a value")?;
        self.take_issues(&sid);
        let function = format!(
            "function(){{ if (this.tagName !== 'SELECT') return 'not a select'; const want = {}; const o = [...this.options].find(o => o.value === want || o.text.trim() === want) || [...this.options].find(o => o.text.trim().toLowerCase().includes(want.toLowerCase())); if (!o) return 'no such option: ' + [...this.options].map(o => o.text.trim()).slice(0, 20).join(' | '); this.value = o.value; this.dispatchEvent(new Event('input', {{bubbles: true}})); this.dispatchEvent(new Event('change', {{bubbles: true}})); return 'ok:' + o.text.trim(); }}",
            Value::String(want.to_owned())
        );
        let outcome = self.call_on(&sid, target.backend, &function).await?;
        let outcome = outcome.as_str().unwrap_or_default();
        match outcome.strip_prefix("ok:") {
            Some(chosen) => {
                self.settle(&sid, Duration::from_secs(3)).await;
                self.conclude(&sid, &format!("Selected \"{chosen}\"."), false).await
            }
            None => Err(outcome.to_owned()),
        }
    }

    async fn scroll(&self, args: &Value) -> Result<Reply, String> {
        let sid = self.current().await?;
        self.take_issues(&sid);
        if args.get("ref").is_some_and(|v| !v.is_null()) {
            let target = self.resolve(&sid, args)?;
            self.cmd("DOM.scrollIntoViewIfNeeded", json!({"backendNodeId": target.backend}), Some(&sid)).await?;
            return self.conclude(&sid, "Scrolled the element into view.", false).await;
        }
        let amount = args["amount"].as_i64().unwrap_or(700).clamp(1, 20000);
        let script = match args["direction"].as_str().unwrap_or("down") {
            "down" => format!("window.scrollBy(0, {amount})"),
            "up" => format!("window.scrollBy(0, -{amount})"),
            "top" => "window.scrollTo(0, 0)".to_owned(),
            "bottom" => "window.scrollTo(0, document.documentElement.scrollHeight)".to_owned(),
            other => return Err(format!("direction '{other}' is not down, up, top or bottom")),
        };
        self.eval(&sid, &script).await?;
        tokio::time::sleep(Duration::from_millis(250)).await;
        let position = self
            .eval(&sid, "[Math.round(window.scrollY), document.documentElement.scrollHeight, window.innerHeight]")
            .await
            .ok()
            .and_then(|v| v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect::<Vec<_>>()))
            .unwrap_or_default();
        let lead = match position.as_slice() {
            [y, total, view] => format!("Scrolled to {y} px of {total} px ({}).", if y + view >= *total { "at the bottom" } else if *y == 0 { "at the top" } else { "more below" }),
            _ => "Scrolled.".to_owned(),
        };
        self.conclude(&sid, &lead, false).await
    }

    async fn back(&self) -> Result<Reply, String> {
        let sid = self.current().await?;
        self.take_issues(&sid);
        let history = self.cmd("Page.getNavigationHistory", json!({}), Some(&sid)).await?;
        let index = history["currentIndex"].as_i64().unwrap_or(0);
        if index <= 0 {
            return Err("there is no earlier page in the history of this tab".into());
        }
        let entry = history["entries"][(index - 1) as usize]["id"].as_i64().ok_or("the history entry is missing")?;
        let target_url = history["entries"][(index - 1) as usize]["url"].as_str().unwrap_or_default().to_owned();
        policy::check_url(&target_url, self.url_policy()).await.map_err(|e| format!("going back is not allowed: {e}"))?;
        self.cmd("Page.navigateToHistoryEntry", json!({"entryId": entry}), Some(&sid)).await?;
        self.settle(&sid, NAV_WAIT).await;
        self.conclude(&sid, "Went back.", true).await
    }

    async fn screenshot(&self, args: &Value) -> Result<Reply, String> {
        let sid = self.current().await?;
        let mut params = json!({"format": "png"});
        let mut what = "the visible part of the page".to_owned();
        if args.get("ref").is_some_and(|v| !v.is_null()) {
            let target = self.resolve(&sid, args)?;
            let (_, _, rect) = self.center_of(&sid, target.backend).await?;
            params["clip"] = json!({"x": rect[0], "y": rect[1], "width": rect[2].max(1.0), "height": rect[3].max(1.0), "scale": 1});
            what = format!("{} \"{}\"", target.role, target.name);
        } else if args["full_page"].as_bool() == Some(true) {
            let metrics = self.cmd("Page.getLayoutMetrics", json!({}), Some(&sid)).await?;
            let w = metrics["cssContentSize"]["width"].as_f64().unwrap_or(VIEW_W as f64).max(1.0);
            let h = metrics["cssContentSize"]["height"].as_f64().unwrap_or(VIEW_H as f64).clamp(1.0, 8000.0);
            params["clip"] = json!({"x": 0, "y": 0, "width": w, "height": h, "scale": 1});
            params["captureBeyondViewport"] = json!(true);
            what = "the whole page".to_owned();
        }
        let shot = self.capture(&sid, params).await?;
        let data = shot["data"].as_str().ok_or("the browser returned no picture")?.to_owned();
        let (title, url) = self.refresh_info(&sid).await;
        let kib = data.len() * 3 / 4 / 1024;
        Ok(Reply {
            text: format!("Screenshot of {what} (PNG, {kib} KiB). Page: {} - {url}\nIf you cannot see images, use browser_snapshot to read the page.", clip(&title, 100)),
            image: Some(format!("data:image/png;base64,{data}")),
        })
    }

    async fn tabs(&self, args: &Value) -> Result<Reply, String> {
        if let Some(n) = args["select"].as_u64() {
            let (sid, target) = {
                let mut st = self.st();
                let at = n.checked_sub(1).map(|i| i as usize).filter(|i| *i < st.tabs.len()).ok_or_else(|| format!("there is no tab {n}"))?;
                st.current = at;
                (st.tabs[at].session_id.clone(), st.tabs[at].target_id.clone())
            };
            let _ = self.cmd("Target.activateTarget", json!({"targetId": target}), None).await;
            self.refresh_info(&sid).await;
        }
        let sids: Vec<String> = self.st().tabs.iter().map(|t| t.session_id.clone()).collect();
        for sid in &sids {
            self.refresh_info(sid).await;
        }
        let st = self.st();
        if st.tabs.is_empty() {
            return Ok(Reply::text("No tabs are open."));
        }
        let lines: Vec<String> = st
            .tabs
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}. {}{} - {}", i + 1, if i == st.current { "[current] " } else { "" }, clip(&t.title, 80), t.url))
            .collect();
        Ok(Reply::text(lines.join("\n")))
    }

    async fn close_tab(&self, args: &Value) -> Result<Reply, String> {
        let target = {
            let st = self.st();
            let at = match args["tab"].as_u64() {
                Some(n) => n.checked_sub(1).map(|i| i as usize).filter(|i| *i < st.tabs.len()).ok_or_else(|| format!("there is no tab {n}"))?,
                None => st.current,
            };
            st.tabs.get(at).map(|t| t.target_id.clone()).ok_or("there is no tab to close")?
        };
        self.cmd("Target.closeTarget", json!({"targetId": target}), None).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let next = {
            let st = self.st();
            st.tabs.get(st.current).map(|t| t.target_id.clone())
        };
        if let Some(next) = next {
            let _ = self.cmd("Target.activateTarget", json!({"targetId": next}), None).await;
        }
        let left = self.st().tabs.len();
        Ok(Reply::text(if left == 0 { "Closed the tab. No tab is open; browser_open opens a new one.".to_owned() } else { format!("Closed the tab. {left} tab(s) left.") }))
    }

    // ---- what the panel needs ---------------------------------------------------------------------------------

    async fn frame(&self, since: &str) -> Result<Value, String> {
        let sid = self.current().await?;
        let shot = match self.capture(&sid, json!({"format": "jpeg", "quality": 62})).await {
            Ok(shot) => shot,
            Err(error) => {
                // the browser's own error pages (connection refused ...) cannot be photographed: say where the tab is instead
                let (title, url) = self.refresh_info(&sid).await;
                return Ok(json!({"running": true, "url": url, "title": title, "undrawable": error}));
            }
        };
        let data = shot["data"].as_str().unwrap_or_default();
        let key = format!("{}-{}", data.len(), data.bytes().step_by((data.len() / 64).max(1)).fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)));
        let (title, url) = self.refresh_info(&sid).await;
        let metrics = self.cmd("Page.getLayoutMetrics", json!({}), Some(&sid)).await.ok();
        let width = metrics.as_ref().and_then(|m| m["cssVisualViewport"]["clientWidth"].as_f64()).unwrap_or(VIEW_W as f64);
        let height = metrics.as_ref().and_then(|m| m["cssVisualViewport"]["clientHeight"].as_f64()).unwrap_or(VIEW_H as f64);
        let mut out = json!({"running": true, "url": url, "title": title, "width": width, "height": height, "key": key});
        if key != since {
            out["image"] = json!(format!("data:image/jpeg;base64,{data}"));
        }
        Ok(out)
    }

    /// An element of the preview, for comments: its ref, role, name and a bit of its text.
    async fn locate(&self, x: f64, y: f64) -> Result<Value, String> {
        let sid = self.current().await?;
        let node = self.cmd("DOM.getNodeForLocation", json!({"x": x.round() as i64, "y": y.round() as i64, "includeUserAgentShadowDOM": false}), Some(&sid)).await?;
        let backend = node["backendNodeId"].as_i64().ok_or("there is no element at that point")?;
        let (role, name) = match self.cmd("Accessibility.getPartialAXTree", json!({"backendNodeId": backend, "fetchRelatives": false}), Some(&sid)).await {
            Ok(tree) => {
                let first = &tree["nodes"][0];
                (first["role"]["value"].as_str().unwrap_or("generic").to_owned(), first["name"]["value"].as_str().unwrap_or_default().to_owned())
            }
            Err(_) => ("generic".to_owned(), String::new()),
        };
        let info = self
            .call_on(
                &sid,
                backend,
                "function(){ const r = this.getBoundingClientRect(); return {tag: this.tagName ? this.tagName.toLowerCase() : '', text: (this.innerText || this.value || '').trim().slice(0, 300), x: r.x, y: r.y, w: r.width, h: r.height}; }",
            )
            .await
            .unwrap_or(Value::Null);
        let number = self.with_tab(&sid, |t| t.refs.assign(backend, &role, &name)).unwrap_or(0);
        let (_, url) = self.refresh_info(&sid).await;
        Ok(json!({
            "ref": number, "role": role, "name": name, "tag": info["tag"], "text": info["text"], "url": url,
            "rect": {"x": info["x"], "y": info["y"], "w": info["w"], "h": info["h"]},
        }))
    }
}

/// The first half of the decision about a request of a page: `Ok(None)` for what never reaches the network (data and blob
/// URLs, the empty page), `Ok(Some(url))` for an address whose host still has to be judged, `Err` for a scheme that is refused.
/// `kind` is the CDP resource type; `Document` is a page or frame being loaded.
fn judge_scheme(url_text: &str, kind: &str) -> Result<Option<reqwest::Url>, String> {
    let scheme = url_text.split(':').next().unwrap_or_default().to_ascii_lowercase();
    match scheme.as_str() {
        "data" | "blob" | "about" | "filesystem" | "chrome-error" => Ok(None),
        "http" | "https" | "ws" | "wss" => Ok(reqwest::Url::parse(url_text).ok()),
        other if kind == "Document" => Err(format!("the scheme '{other}:' is not allowed")),
        other => Err(format!("requests to '{other}:' are not allowed")),
    }
}

/// The second half: what the address leads to, under the settings. A page from the internet (`main_local` false) may not
/// load subresources from this machine or the local network, whatever the settings say.
fn judge_reach(reach: Reach, kind: &str, policy: UrlPolicy, main_local: bool) -> Result<(), String> {
    if !policy.allows(reach) {
        return Err(match reach {
            Reach::Forbidden => "link-local, metadata and reserved addresses are never reachable".to_owned(),
            _ => "a local address that the browser settings do not allow".to_owned(),
        });
    }
    if kind != "Document" && matches!(reach, Reach::Loopback | Reach::Private) && !main_local {
        return Err("a page from the internet may not load local addresses".into());
    }
    Ok(())
}

/// The scheme for an address typed without one: `http` for `localhost` and IP addresses (dev servers rarely have a certificate),
/// `https` for everything else.
fn bare_scheme(text: &str) -> &'static str {
    let host = text.split(['/', '?', '#']).next().unwrap_or_default();
    let host = if host.starts_with('[') { host.split(']').next().unwrap_or_default().trim_start_matches('[') } else { host.rsplit_once(':').map_or(host, |(h, _)| h) };
    if host.eq_ignore_ascii_case("localhost") || host.to_ascii_lowercase().ends_with(".localhost") || host.parse::<std::net::IpAddr>().is_ok() {
        "http"
    } else {
        "https"
    }
}

fn ref_arg(args: &Value) -> Result<u32, String> {
    match args.get("ref") {
        Some(Value::Number(n)) => n.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| "ref has to be a whole number from browser_snapshot".to_owned()),
        Some(Value::String(s)) => s.trim().trim_start_matches("ref=").trim_start_matches('e').parse().map_err(|_| format!("'{s}' is not a ref: use the number after ref= in browser_snapshot")),
        _ => Err("this tool needs a ref from browser_snapshot".into()),
    }
}

fn remote_text(object: &Value) -> String {
    match &object["value"] {
        Value::String(s) => s.clone(),
        Value::Null => object["description"].as_str().unwrap_or_else(|| object["type"].as_str().unwrap_or("")).to_owned(),
        other => other.to_string(),
    }
}

fn clip(text: &str, cap: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= cap {
        flat
    } else {
        format!("{}...", flat.chars().take(cap).collect::<String>())
    }
}

// ---- routes for the Code page -------------------------------------------------------------------------------------

fn status_json(state: &AppState, settings: &Settings, session: Option<&Arc<Session>>) -> Value {
    let found = policy::find_browser(settings);
    json!({
        "settings": settings.to_json(),
        "running": session.is_some(),
        "url": session.map(|s| s.current_url()),
        "workspace": state.agent.current_root().is_some(),
        "browser": found.as_ref().ok().map(|p| p.display().to_string()),
        "browser_error": found.err(),
        "profile": policy::profile_dir().display().to_string(),
        "env_browser": std::env::var("FLLM_BROWSER").ok().filter(|v| !v.trim().is_empty()),
        "view": {"width": VIEW_W, "height": VIEW_H},
    })
}

fn blocked() -> Option<Response> {
    if super::agent::agent_disabled() {
        return Some(super::agent::disabled_reply());
    }
    None
}

/// `GET /api/studio/agent/browser`: the settings, whether a browser is installed and whether it runs.
pub(in crate::server) async fn status(State(state): State<Arc<AppState>>) -> Response {
    if let Some(r) = blocked() {
        return r;
    }
    let settings = policy::read_settings();
    let session = running().await;
    Json(status_json(&state, &settings, session.as_ref())).into_response()
}

/// `POST /api/studio/agent/browser/settings`: `{enabled?, show_window?, local_addresses?, executable?, persistent_profile?}`.
pub(in crate::server) async fn save_settings(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if let Some(r) = blocked() {
        return r;
    }
    let before = policy::read_settings();
    let after = match before.updated(&body) {
        Ok(next) => next,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    if let Err(error) = policy::write_settings(&after) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error);
    }
    // switching it off, or changing how it starts, closes the running browser; the next use starts it again
    if !after.enabled || after.show_window != before.show_window || after.executable != before.executable || after.persistent_profile != before.persistent_profile {
        shutdown().await;
    }
    let session = running().await;
    Json(status_json(&state, &after, session.as_ref())).into_response()
}

/// `POST /api/studio/agent/browser/frame` `{since?}`: the current picture of the page for the panel (JPEG as a data URL, left out when it
/// is the same as the one the panel already has).
pub(in crate::server) async fn frame(Json(body): Json<Value>) -> Response {
    if let Some(r) = blocked() {
        return r;
    }
    let Some(session) = running().await else { return Json(json!({"running": false})).into_response() };
    if session.idle_for() > IDLE_LIMIT {
        // nobody used it for a long time (the panel's own polling does not count): let it go
        drop(session);
        shutdown().await;
        return Json(json!({"running": false})).into_response();
    }
    let since = body.get("since").and_then(Value::as_str).unwrap_or_default().to_owned();
    match session.frame(&since).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => Json(json!({"running": true, "error": e})).into_response(),
    }
}

/// `POST /api/studio/agent/browser/input`: what the user does in the preview.
/// `{type: "click"|"dblclick"|"wheel"|"key"|"text"|"navigate"|"back"|"forward"|"reload"|"locate"|"close", x?, y?, dx?, dy?, key?, text?, url?}`.
pub(in crate::server) async fn input(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if let Some(r) = blocked() {
        return r;
    }
    let settings = policy::read_settings();
    if !settings.enabled {
        return api_error(StatusCode::FORBIDDEN, OFF_MESSAGE);
    }
    let kind = body.get("type").and_then(Value::as_str).unwrap_or_default().to_owned();
    if kind == "close" {
        shutdown().await;
        return Json(json!({"ok": true})).into_response();
    }
    let workspace = state.agent.current_root().is_some();
    let session = match kind.as_str() {
        // typing an address starts the browser; the other actions need one that runs
        "navigate" => match session(&settings, workspace).await {
            Ok(s) => s,
            Err(e) => return Json(json!({"ok": false, "error": e})).into_response(),
        },
        _ => match running().await {
            Some(s) => s,
            None => return Json(json!({"ok": false, "error": "the browser is not running"})).into_response(),
        },
    };
    session.configure(&settings, workspace);
    session.touch();
    let num = |key: &str| body.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    let result: Result<Value, String> = async {
        let sid = session.current().await?;
        match kind.as_str() {
            "click" | "dblclick" => {
                session.click_at(&sid, num("x"), num("y"), if kind == "dblclick" { 2 } else { 1 }).await?;
                Ok(json!({}))
            }
            "wheel" => {
                session
                    .cmd("Input.dispatchMouseEvent", json!({"type": "mouseWheel", "x": num("x"), "y": num("y"), "deltaX": num("dx"), "deltaY": num("dy")}), Some(&sid))
                    .await?;
                Ok(json!({}))
            }
            "key" => {
                session.press_key(&sid, body.get("key").and_then(Value::as_str).unwrap_or_default()).await?;
                Ok(json!({}))
            }
            "text" => {
                let text = body.get("text").and_then(Value::as_str).unwrap_or_default();
                session.cmd("Input.insertText", json!({"text": text}), Some(&sid)).await?;
                Ok(json!({}))
            }
            "navigate" => {
                let text = body.get("url").and_then(Value::as_str).unwrap_or_default().trim().to_owned();
                // a bare host name is what people type
                let text = if !text.contains("://") && !text.starts_with("about:") && !text.is_empty() { format!("{}://{text}", bare_scheme(&text)) } else { text };
                let url = policy::check_url(&text, session.url_policy()).await?;
                let answer = session.cmd("Page.navigate", json!({"url": url.as_str()}), Some(&sid)).await?;
                if let Some(error) = answer["errorText"].as_str().filter(|e| !e.is_empty()) {
                    return Err(format!("could not open {url}: {error}"));
                }
                Ok(json!({}))
            }
            "back" | "forward" => {
                let history = session.cmd("Page.getNavigationHistory", json!({}), Some(&sid)).await?;
                let index = history["currentIndex"].as_i64().unwrap_or(0) + if kind == "back" { -1 } else { 1 };
                let entry = usize::try_from(index).ok().and_then(|i| history["entries"].get(i).cloned()).ok_or("there is no such page in the history")?;
                policy::check_url(entry["url"].as_str().unwrap_or_default(), session.url_policy()).await?;
                session.cmd("Page.navigateToHistoryEntry", json!({"entryId": entry["id"]}), Some(&sid)).await?;
                Ok(json!({}))
            }
            "reload" => {
                session.cmd("Page.reload", json!({}), Some(&sid)).await?;
                Ok(json!({}))
            }
            "locate" => session.locate(num("x"), num("y")).await,
            other => Err(format!("unknown input type '{other}'")),
        }
    }
    .await;
    match result {
        Ok(mut v) => {
            v["ok"] = json!(true);
            Json(v).into_response()
        }
        Err(e) => Json(json!({"ok": false, "error": e})).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalog_has_the_browser_tools_with_schemas_and_the_network_permission() {
        let tools = catalog();
        let names: Vec<&str> = tools.iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        for expected in [
            "browser_open", "browser_snapshot", "browser_click", "browser_type", "browser_select", "browser_press", "browser_scroll", "browser_back", "browser_screenshot",
            "browser_tabs", "browser_close",
        ] {
            assert!(names.contains(&expected), "{expected} missing");
        }
        for t in &tools {
            assert_eq!(t["permission"], "network", "every browser tool asks like a network tool");
            assert_eq!(t["type"], "function");
            let props = t["function"]["parameters"]["properties"].as_object().unwrap();
            for required in t["function"]["parameters"]["required"].as_array().unwrap() {
                assert!(props.contains_key(required.as_str().unwrap()));
            }
        }
        assert!(is_tool("browser_click") && !is_tool("browser_hack") && !is_tool("web_fetch"));
    }

    #[test]
    fn requests_of_a_page_are_judged_by_scheme_and_by_where_the_page_comes_from() {
        use super::policy::LocalAddresses::{All, Auto, Off};
        let pol = |local, workspace| UrlPolicy { local, workspace };
        // schemes: files and browser pages are never loaded, data and blob never touch the network
        for url in ["file:///etc/passwd", "chrome://settings", "view-source:https://x.test/", "javascript:alert(1)", "ftp://x.test/", "chrome-extension://abc/p.html"] {
            assert!(judge_scheme(url, "Document").is_err(), "{url} as a page");
            assert!(judge_scheme(url, "Image").is_err(), "{url} as a resource");
        }
        for url in ["data:text/html,hi", "blob:https://x.test/uuid", "about:blank", "chrome-error://chromewebdata/"] {
            assert!(judge_scheme(url, "Document").unwrap().is_none(), "{url}");
        }
        assert!(judge_scheme("https://example.com/a", "Script").unwrap().is_some());
        // a page from the internet cannot read from this machine or the local network, even with everything allowed
        for (reach, name) in [(Reach::Loopback, "loopback"), (Reach::Private, "private")] {
            assert!(judge_reach(reach, "XHR", pol(All, true), false).is_err(), "{name} resource of a public page");
            assert!(judge_reach(reach, "Script", pol(All, true), false).is_err());
            assert!(judge_reach(reach, "Document", pol(All, true), false).is_ok(), "{name} page the user or agent opened on purpose");
            assert!(judge_reach(reach, "XHR", pol(All, true), true).is_ok(), "{name} resource of a local page");
        }
        // the settings still apply to local pages
        assert!(judge_reach(Reach::Loopback, "XHR", pol(Auto, false), true).is_err(), "no workspace");
        assert!(judge_reach(Reach::Loopback, "XHR", pol(Auto, true), true).is_ok());
        assert!(judge_reach(Reach::Loopback, "Document", pol(Off, true), true).is_err());
        assert!(judge_reach(Reach::Private, "Document", pol(Auto, true), true).is_err());
        // metadata and link-local never, from anywhere
        for kind in ["Document", "XHR", "Image"] {
            for main_local in [true, false] {
                assert!(judge_reach(Reach::Forbidden, kind, pol(All, true), main_local).is_err());
            }
        }
        assert!(judge_reach(Reach::Public, "Script", pol(Off, false), false).is_ok());
    }

    #[test]
    fn an_address_typed_without_a_scheme_gets_http_for_local_hosts_and_https_otherwise() {
        for local in ["localhost:3000", "localhost", "127.0.0.1:8080/x", "[::1]:5173/", "app.localhost/p", "192.168.1.2"] {
            assert_eq!(bare_scheme(local), "http", "{local}");
        }
        for public in ["example.com", "example.com:8443/a?b=c", "docs.rs/x"] {
            assert_eq!(bare_scheme(public), "https", "{public}");
        }
    }

    #[test]
    fn refs_are_accepted_as_numbers_or_the_text_of_the_snapshot() {
        assert_eq!(ref_arg(&json!({"ref": 7})).unwrap(), 7);
        assert_eq!(ref_arg(&json!({"ref": "12"})).unwrap(), 12);
        assert_eq!(ref_arg(&json!({"ref": "ref=3"})).unwrap(), 3);
        assert_eq!(ref_arg(&json!({"ref": "e5"})).unwrap(), 5);
        for bad in [json!({}), json!({"ref": -1}), json!({"ref": "x"}), json!({"ref": 1.5}), json!({"ref": null})] {
            assert!(ref_arg(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn tab_problems_are_deduplicated_and_capped() {
        let mut tab = Tab::new("t", "s");
        tab.issue("console.error: boom".into());
        tab.issue("console.error: boom".into());
        assert_eq!(tab.issues.len(), 1);
        for i in 0..100 {
            tab.issue(format!("problem {i}"));
        }
        assert_eq!(tab.issues.len(), MAX_ISSUES);
    }

    #[test]
    fn console_arguments_and_long_text_are_shortened_for_the_model() {
        assert_eq!(remote_text(&json!({"type": "string", "value": "Failed to fetch"})), "Failed to fetch");
        assert_eq!(remote_text(&json!({"type": "object", "description": "TypeError: x is undefined"})), "TypeError: x is undefined");
        assert_eq!(remote_text(&json!({"type": "number", "value": 404})), "404");
        assert_eq!(clip("a   b\n c", 50), "a b c");
        assert!(clip(&"x".repeat(500), 20).ends_with("..."));
    }

    #[tokio::test]
    async fn a_browser_tool_says_what_to_do_when_the_feature_is_off() {
        // the settings file of a test machine does not exist: the default is off
        if policy::read_settings().enabled {
            return;
        }
        let (result, image) = run_tool(false, "browser_open", &json!({"url": "https://example.com"})).await.unwrap();
        let error = result.unwrap_err();
        assert!(error.contains("Settings > Browser") && image.is_none(), "{error}");
        assert!(run_tool(false, "web_fetch", &json!({})).await.is_none(), "other tools are not ours");
    }
}
