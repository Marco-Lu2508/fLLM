//! Artifacts: the fenced blocks of an answer that carry a title or can be previewed, grouped into
//! families with versions (the `@pure` block of the Studio's `artifacts.js`).

use serde::{Deserialize, Serialize};

/// What the window keeps of the Studio's artifact code: the rows the table preview shows, the
/// seconds a run of JavaScript gets and the length of a file name.
pub mod limits {
    /// Rows of a CSV preview; the rest is named under the table.
    pub const CSV_ROWS: usize = 500;
    /// The milliseconds a run of JavaScript gets before it is stopped.
    pub const RUN_MS: u64 = 5000;
    /// Characters of the file name before the extension.
    pub const NAME: usize = 60;
    /// Console lines a run of JavaScript prints before the output is cut.
    pub const RUN_LINES: usize = 500;
    /// Characters of that output before the output is cut.
    pub const RUN_CHARS: usize = 20_000;
}

/// What a fenced block can be previewed as: html, svg, markdown, csv; anything else is code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Html,
    Svg,
    Markdown,
    Csv,
    Code,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Html => "HTML page",
            Kind::Svg => "SVG image",
            Kind::Markdown => "Markdown document",
            Kind::Csv => "Table (CSV)",
            Kind::Code => "Code",
        }
    }

    pub fn ext(self) -> &'static str {
        match self {
            Kind::Html => "html",
            Kind::Svg => "svg",
            Kind::Markdown => "md",
            Kind::Csv => "csv",
            Kind::Code => "",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            Kind::Html => "text/html",
            Kind::Svg => "image/svg+xml",
            Kind::Markdown => "text/markdown",
            Kind::Csv => "text/csv",
            Kind::Code => "text/plain",
        }
    }

    pub fn is_previewable(self) -> bool {
        self != Kind::Code
    }
}

/// The languages that make a block previewable (`KIND_BY_LANG`).
pub fn kind_by_lang(lang: &str) -> Option<Kind> {
    match lang.to_lowercase().as_str() {
        "html" | "htm" | "xhtml" => Some(Kind::Html),
        "svg" => Some(Kind::Svg),
        "md" | "markdown" => Some(Kind::Markdown),
        "csv" | "tsv" => Some(Kind::Csv),
        _ => None,
    }
}

/// The file extension of a language (`EXT_BY_LANG`).
pub fn ext_by_lang(lang: &str) -> Option<&'static str> {
    match lang.to_lowercase().as_str() {
        "python" | "py" => Some("py"),
        "javascript" | "js" => Some("js"),
        "typescript" | "ts" => Some("ts"),
        "rust" | "rs" => Some("rs"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" => Some("c"),
        "cpp" => Some("cpp"),
        "sh" | "bash" => Some("sh"),
        "sql" => Some("sql"),
        "json" => Some("json"),
        "css" => Some("css"),
        "yaml" | "yml" => Some("yaml"),
        "toml" => Some("toml"),
        _ => None,
    }
}

/// What a fenced block is: html, svg, markdown, csv; anything else is plain code.
pub fn artifact_kind(lang: &str) -> Kind {
    kind_by_lang(lang).unwrap_or(Kind::Code)
}

/// The JavaScript languages a block can be run as (`isJsLang`).
pub fn is_js_lang(lang: &str) -> bool {
    matches!(lang.to_lowercase().as_str(), "js" | "javascript" | "mjs" | "node")
}

/// Reads `title="..."` (also `artifact=` or `name=`, quoted or bare) from the info string of a
/// fence (`parseInfo`).
pub fn parse_info(info: &str) -> String {
    let re = crate::speech::re(r#"(?i)(?:^|\s)(?:title|artifact|name)\s*=\s*("([^"]*)"|'([^']*)'|(\S+))"#);
    match re.captures(info) {
        Some(c) => c
            .get(2)
            .or_else(|| c.get(3))
            .or_else(|| c.get(4))
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default(),
        None => String::new(),
    }
}

/// One fenced block of an answer that is previewable or carries a title.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub lang: String,
    pub kind: Kind,
    pub title: String,
    pub code: String,
}

/// A version of an artifact: which message wrote it, and what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub n: u32,
    #[serde(rename = "msgId")]
    pub msg_id: String,
    pub code: String,
}

/// An artifact of the conversation with all its versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Family {
    pub id: String,
    pub kind: Kind,
    pub title: String,
    pub lang: String,
    pub versions: Vec<Version>,
    pub touched: u64,
}

/// Closed fenced blocks of a message that are previewable or carry a title
/// (`extractArtifacts`).
pub fn extract_artifacts(text: &str) -> Vec<Artifact> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    let open = crate::speech::re(r"^(\s*(`{3,}|~{3,}))\s*([\w+#.-]*)(.*)$");
    let close_pure = crate::speech::re(r"^\s*[`~]+\s*$");
    let mut found = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let Some(caps) = open.captures(lines[i]) else {
            i += 1;
            continue;
        };
        let fence: String = caps[2].trim().to_string();
        let fence_char = fence.chars().next().unwrap_or('`');
        let fence_str: String = std::iter::repeat_n(fence_char, fence.len()).collect();
        let lang = caps[3].to_string();
        let info = caps[4].to_string();
        let mut body: Vec<&str> = Vec::new();
        let mut closed = false;
        i += 1;
        while i < lines.len() {
            let line = lines[i];
            if line.trim().starts_with(&fence_str) && close_pure.is_match(line) {
                closed = true;
                break;
            }
            body.push(line);
            i += 1;
        }
        if !closed {
            i += 1;
            continue;
        }
        i += 1;
        let kind = artifact_kind(&lang);
        let title = parse_info(&info);
        if kind != Kind::Code || !title.is_empty() {
            found.push(Artifact { lang: lang.to_lowercase(), kind, title, code: body.join("\n") });
        }
    }
    found
}

/// Groups the artifacts of a conversation (assistant messages, oldest first) into artifacts with
/// versions (`buildFamilies`): a titled block joins the artifact of the same title and kind; an
/// untitled one joins the artifact with identical code, else the most recently changed of its
/// kind, else it starts a new one. One message changes an artifact at most once.
pub fn build_families(messages: &[(String, String)]) -> Vec<Family> {
    let mut families: Vec<Family> = Vec::new();
    let mut clock: u64 = 0;
    for (msg_id, text) in messages {
        let mut touched: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (index, a) in extract_artifacts(text).into_iter().enumerate() {
            let hit: Option<usize> = if !a.title.is_empty() {
                (0..families.len()).find(|&i| {
                    families[i].kind == a.kind
                        && families[i].title.to_lowercase() == a.title.to_lowercase()
                        && !touched.contains(&i)
                })
            } else {
                let same: Vec<usize> = families.iter().enumerate().filter(|(_, f)| f.kind == a.kind).map(|(i, _)| i).collect();
                same.iter()
                    .copied()
                    .find(|&i| families[i].versions.last().is_some_and(|v| v.code == a.code))
                    .or_else(|| {
                        same.iter()
                            .copied()
                            .filter(|&i| !touched.contains(&i))
                            .max_by_key(|&i| families[i].touched)
                    })
            };
            let fam = match hit {
                Some(i) => i,
                None => {
                    families.push(Family {
                        id: format!("{msg_id}#{index}"),
                        kind: a.kind,
                        title: a.title.clone(),
                        lang: a.lang.clone(),
                        versions: Vec::new(),
                        touched: 0,
                    });
                    families.len() - 1
                }
            };
            let same_code = families[fam].versions.last().is_some_and(|v| v.code == a.code);
            if !same_code {
                let n = families[fam].versions.len() as u32 + 1;
                families[fam].versions.push(Version { n, msg_id: msg_id.clone(), code: a.code.clone() });
            }
            clock += 1;
            families[fam].touched = clock;
            touched.insert(fam);
        }
    }
    families
}

/// The file a download of this artifact gets (`fileNameFor`).
pub fn file_name(title: &str, kind: Kind, lang: &str) -> String {
    let cleaned: String = title.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ' ' | '-')).collect();
    let base = cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(limits::NAME)
        .collect::<String>();
    let base = if base.is_empty() { "artifact".to_string() } else { base };
    let ext = if !kind.ext().is_empty() {
        kind.ext().to_string()
    } else {
        ext_by_lang(lang).unwrap_or("txt").to_string()
    };
    format!("{base}.{ext}")
}

/// Comma, semicolon or tab separated text -> rows. Quoted fields may hold the separator and line
/// breaks (`parseCsv`).
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let src = text.replace("\r\n", "\n").replace('\r', "\n");
    let first = src.split('\n').next().unwrap_or("");
    let count = |c: char| first.matches(c).count();
    let delim = [',', ';', '\t'].iter().copied().fold(',', |best, c| if count(c) > count(best) { c } else { best });
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if quoted {
            if c == '"' && chars.get(i + 1) == Some(&'"') {
                cell.push('"');
                i += 1;
            } else if c == '"' {
                quoted = false;
            } else {
                cell.push(c);
            }
        } else if c == '"' && cell.is_empty() {
            quoted = true;
        } else if c == delim {
            row.push(std::mem::take(&mut cell));
        } else if c == '\n' {
            row.push(std::mem::take(&mut cell));
            rows.push(std::mem::take(&mut row));
        } else {
            cell.push(c);
        }
        i += 1;
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    rows.retain(|r| r.len() > 1 || r.first().is_none_or(|c| !c.is_empty()));
    rows
}

/// The buttons of the artifact panel (`artAction` of the Studio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtAction {
    Close,
    Prev,
    Next,
    TabPreview,
    TabCode,
    Reload,
    Copy,
    Download,
    Larger,
    /// Write the page and open it in the browser of the computer (the window draws no HTML).
    OpenInBrowser,
}

/// What the panel shows: the preview tab or the code of the current version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Preview,
    Code,
}

/// The artifact panel: which family, version and tab are open (the Studio's `ART` state).
#[derive(Debug)]
pub struct Panel {
    pub fam: Family,
    pub ver: usize,
    pub tab: Tab,
    pub large: bool,
    /// Set when the panel was opened from a block that belongs to no family of the chat.
    pub adhoc: bool,
    /// The rendered body of the preview tab (Markdown and the table of a CSV).
    pub body: Option<iced::widget::markdown::Content>,
}

impl Panel {
    /// Opens a family at a version, on the tab the Studio picks (`openFamily`).
    pub fn open(fam: Family, ver: usize, adhoc: bool) -> Panel {
        let ver = ver.min(fam.versions.len().saturating_sub(1));
        let kind = fam.kind;
        let tab = if kind == Kind::Code { Tab::Code } else { Tab::Preview };
        let mut panel = Panel { fam, ver, tab, large: false, adhoc, body: None };
        panel.render_body();
        panel
    }

    /// The version the panel shows.
    pub fn version(&self) -> &Version {
        &self.fam.versions[self.ver.min(self.fam.versions.len().saturating_sub(1))]
    }

    /// Rebuilds the rendered body of the preview (Markdown and CSV render; a page and code do not
    /// need it, they are shown as their text).
    pub fn render_body(&mut self) {
        self.body = match self.fam.kind {
            Kind::Markdown => Some(iced::widget::markdown::Content::parse(&self.version().code)),
            Kind::Csv => Some(iced::widget::markdown::Content::parse(&csv_markdown(&parse_csv(&self.version().code)))),
            _ => None,
        };
    }

    /// Moves to another version, clamped like the Studio's own clamp.
    pub fn set_version(&mut self, ver: usize) {
        self.ver = ver.min(self.fam.versions.len().saturating_sub(1));
        self.render_body();
    }

    /// The tab of the Studio: a family of code can only show its code.
    pub fn set_tab(&mut self, tab: Tab) {
        self.tab = if self.fam.kind == Kind::Code { Tab::Code } else { tab };
    }
}

/// The families of the open chat, oldest first: the assistant messages that are not being written
/// anymore, as `(id, text)` the way `buildFamilies` takes them.
pub fn chat_families(messages: &[(String, String, bool)]) -> Vec<Family> {
    let msgs: Vec<(String, String)> = messages
        .iter()
        .filter(|(_, _, streaming)| !*streaming)
        .map(|(id, text, _)| (id.clone(), text.clone()))
        .collect();
    build_families(&msgs)
}

/// What `__done` reported: the value of the run and its error.
type Outcome = Option<(Option<String>, Option<String>)>;

/// What a running JavaScript reports back (the messages of the Studio's `runner.js`).
#[derive(Debug, Clone, PartialEq)]
pub enum RunEvent {
    /// One line of console output, with its level: log, info, warn or error.
    Log { level: &'static str, text: String },
    /// The run is over: how long it took, and what came out of it.
    Done { ms: u64, value: Option<String>, error: Option<String> },
    /// The run could not even start.
    Failed(String),
}

/// One line of the output of a run (`run-line` of the Studio's card).
#[derive(Debug, Clone, PartialEq)]
pub struct RunLine {
    pub level: String,
    pub text: String,
}

/// How a run of JavaScript stands while it goes (the `.run-out` of a code card).
#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Running,
    Finished { ms: u64, failed: bool, line: String },
    Stopped(String),
}

/// The output under a code block while its JavaScript runs (`runBlock`).
#[derive(Debug, Clone, PartialEq)]
pub struct CodeRun {
    pub lines: Vec<RunLine>,
    pub status: RunStatus,
    /// The flags of the studio's runner: the stop of a finished run says why.
    pub warn: bool,
}

impl CodeRun {
    /// The card shows "Running..." under the block (`runBlock` starts this way).
    pub fn starting() -> CodeRun {
        CodeRun { lines: Vec::new(), status: RunStatus::Running, warn: false }
    }

    /// One message of the runner: a line, or the end of the run.
    pub fn apply(&mut self, ev: RunEvent) {
        match ev {
            RunEvent::Log { level, text } => self.lines.push(RunLine { level: level.to_string(), text }),
            RunEvent::Done { ms, value, error } => {
                if let Some(err) = error {
                    self.lines.push(RunLine { level: "error".into(), text: err });
                }
                if let Some(v) = value {
                    self.lines.push(RunLine { level: "value".into(), text: format!("-> {v}") });
                }
                let failed = self.lines.iter().any(|l| l.level == "error");
                self.status = RunStatus::Finished {
                    ms,
                    failed,
                    line: format!("{} in {ms} ms", if failed { "Failed" } else { "Done" }),
                };
                self.warn = failed;
            }
            RunEvent::Failed(msg) => {
                self.status = RunStatus::Stopped(msg);
                self.warn = true;
            }
        }
    }

    /// The stop of the card (`stopRun`): it says why the run ended.
    pub fn stop(&mut self, why: &str) {
        self.status = RunStatus::Stopped(why.to_string());
        self.warn = why != "Stopped";
    }
}

/// The script of the Studio's `runner.js`, wrapped for the quickjs context: same `fmt`, same
/// limits, same blocked functions, the sync code first and the async code as a fallback; the
/// callbacks `__emit` and `__done` carry its messages out.
const RUNNER_SCRIPT: &str = r#"
'use strict';
const MAX_LINES = 500;
const MAX_CHARS = 20000;
const BLOCKED = ['fetch', 'XMLHttpRequest', 'WebSocket', 'EventSource', 'importScripts', 'Worker', 'SharedWorker', 'WebTransport'];
let lines = 0;
let chars = 0;
let cut = false;
const fmt = (v, quote) => {
  if (typeof v === 'string') return quote ? JSON.stringify(v) : v;
  if (v === undefined) return 'undefined';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'function') return `[Function${v.name ? `: ${v.name}` : ''}]`;
  if (typeof v === 'bigint') return `${v}n`;
  if (typeof v === 'symbol') return String(v);
  if (v instanceof Error) return `${v.name || 'Error'}: ${v.message}`;
  try {
    const seen = new Set();
    const s = JSON.stringify(v, (k, x) => {
      if (typeof x === 'bigint') return `${x}n`;
      if (typeof x === 'function') return '[Function]';
      if (x instanceof Map) return { Map: Array.from(x.entries()) };
      if (x instanceof Set) return { Set: Array.from(x.values()) };
      if (x && typeof x === 'object') { if (seen.has(x)) return '[Repeated]'; seen.add(x); }
      return x;
    });
    return s === undefined ? String(v) : s;
  } catch (err) { return String(v); }
};
const emit = (level, text) => {
  if (cut) return;
  lines++;
  chars += text.length;
  if (lines > MAX_LINES || chars > MAX_CHARS) {
    cut = true;
    __emit('warn', '... output truncated');
    return;
  }
  __emit(level, text);
};
const con = {
  log: (...a) => emit('log', a.map((x) => fmt(x, false)).join(' ')),
  info: (...a) => emit('info', a.map((x) => fmt(x, false)).join(' ')),
  debug: (...a) => emit('log', a.map((x) => fmt(x, false)).join(' ')),
  warn: (...a) => emit('warn', a.map((x) => fmt(x, false)).join(' ')),
  error: (...a) => emit('error', a.map((x) => fmt(x, false)).join(' ')),
  table: (...a) => emit('log', a.map((x) => fmt(x, false)).join(' ')),
  dir: (...a) => emit('log', a.map((x) => fmt(x, false)).join(' ')),
  assert: (ok, ...a) => { if (!ok) emit('error', ['Assertion failed:', ...a].map((x) => fmt(x, false)).join(' ')); },
  clear: () => {},
};
const stub = (name) => function blocked() { throw new Error(`${name} is disabled: code blocks run offline`); };
BLOCKED.forEach((name) => {
  try { Object.defineProperty(globalThis, name, { value: stub(name), configurable: false, writable: false }); } catch (err) { /* already locked */ }
});
let started = false;
const mark = () => { started = true; };
const exec = (source) => new Function('console', '__mark', ...BLOCKED, '__code', '"use strict"; return eval(__code);')(con, mark, ...BLOCKED.map(stub), source);
const finish = (res) => __done(JSON.stringify({ value: res === undefined ? undefined : fmt(res, true) }));
try {
  let res;
  try {
    res = exec(`__mark();\n${__CODE}`);
  } catch (first) {
    if (started || !(first instanceof SyntaxError)) throw first;
    try { res = exec(`(async () => { __mark();\n${__CODE}\n})()`); } catch (second) { throw !started && second instanceof SyntaxError ? first : second; }
  }
  if (res && typeof res.then === 'function') {
    res.then(
      (v) => finish(v),
      (e) => __done(JSON.stringify({ error: (e && e.name) ? `${e.name}: ${e.message}` : String(e) }))
    );
  } else {
    finish(res);
  }
} catch (e) {
  __done(JSON.stringify({ error: (e && e.name) ? `${e.name}: ${e.message}` : String(e) }));
}
"#;

/// The channel a run reports over: the window takes the messages, the runner sends them.
pub fn run_channel() -> (
    futures::channel::mpsc::UnboundedSender<RunEvent>,
    futures::channel::mpsc::UnboundedReceiver<RunEvent>,
) {
    futures::channel::mpsc::unbounded()
}

/// Runs one code block like the Studio's worker: same script, same limits, no network (the
/// blocked functions throw), 5 seconds at most (`runBlock`), one line per console call — the
/// messages go out over `tx`, the run ends with [`RunEvent::Done`].
pub fn run_javascript(code: &str, tx: futures::channel::mpsc::UnboundedSender<RunEvent>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    let send = |ev: RunEvent| {
        let _ = tx.unbounded_send(ev);
    };
    let started = std::time::Instant::now();
    let timed_out = Arc::new(AtomicBool::new(false));
    let Ok(rt) = rquickjs::Runtime::new() else {
        send(RunEvent::Failed("The runner could not start: no runtime".into()));
        return;
    };
    {
        let timed_out = timed_out.clone();
        rt.set_interrupt_handler(Some(Box::new(move || {
            let over = started.elapsed() > std::time::Duration::from_millis(limits::RUN_MS);
            if over {
                timed_out.store(true, Ordering::SeqCst);
            }
            over
        })));
    }
    let Ok(ctx) = rquickjs::Context::full(&rt) else {
        send(RunEvent::Failed("The runner could not start: no context".into()));
        return;
    };
    let done = Arc::new(AtomicBool::new(false));
    let outcome: Arc<Mutex<Outcome>> = Arc::new(Mutex::new(None));
    let ran = ctx.with(|ctx| -> Result<(), rquickjs::Error> {
        // console, one line per call (the `emit` of runner.js)
        let obj = rquickjs::Object::new(ctx.clone())?;
        for (name, level) in [("log", "log"), ("info", "info"), ("debug", "log"), ("warn", "warn"), ("error", "error"), ("table", "log"), ("dir", "log")] {
            let tx = tx.clone();
            let f = rquickjs::Function::new(ctx.clone(), move |args: rquickjs::prelude::Rest<rquickjs::Value>| {
                let parts: Vec<String> = args.0.iter().filter_map(|v| v.as_string().and_then(|s| s.to_string().ok())).collect();
                let _ = tx.unbounded_send(RunEvent::Log { level, text: parts.join(" ") });
            })?;
            obj.set(name, f)?;
        }
        {
            let tx = tx.clone();
            let f = rquickjs::Function::new(ctx.clone(), move |ok: bool, rest: rquickjs::prelude::Rest<rquickjs::Value>| {
                if !ok {
                    let parts: Vec<String> = std::iter::once("Assertion failed:".to_string())
                        .chain(rest.0.iter().filter_map(|v| v.as_string().and_then(|s| s.to_string().ok())))
                        .collect();
                    let _ = tx.unbounded_send(RunEvent::Log { level: "error", text: parts.join(" ") });
                }
            })?;
            obj.set("assert", f)?;
        }
        obj.set("clear", rquickjs::Function::new(ctx.clone(), || {}))?;
        ctx.globals().set("console", obj)?;
        ctx.globals().set("__CODE", code)?;
        let tx2 = tx.clone();
        let f = rquickjs::Function::new(ctx.clone(), move |level: String, text: String| {
            let level: &'static str = match level.as_str() {
                "error" => "error",
                "warn" => "warn",
                "info" => "info",
                _ => "log",
            };
            let _ = tx2.unbounded_send(RunEvent::Log { level, text });
        })?;
        ctx.globals().set("__emit", f)?;
        let done_c = done.clone();
        let outcome_c = outcome.clone();
        let f = rquickjs::Function::new(ctx.clone(), move |json: String| {
            done_c.store(true, Ordering::SeqCst);
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap_or(serde_json::Value::Null);
            let value = parsed.get("value").and_then(|v| v.as_str()).map(str::to_string);
            let error = parsed.get("error").and_then(|v| v.as_str()).map(str::to_string);
            *outcome_c.lock().unwrap() = Some((value, error));
        })?;
        ctx.globals().set("__done", f)?;
        let result: rquickjs::Value = ctx.eval(RUNNER_SCRIPT)?;
        let _ = result;
        Ok(())
    });
    if let Err(e) = ran {
        if timed_out.load(Ordering::SeqCst) {
            // the five seconds ended the script itself (the interrupt of the runtime)
            send(RunEvent::Failed(format!("Stopped after {} s (time limit)", limits::RUN_MS / 1000)));
        } else {
            send(RunEvent::Failed(format!("The runner could not start: {e}")));
        }
        return;
    }
    // the pending jobs of an async answer are driven here; five seconds are the limit of the Studio
    while !done.load(Ordering::SeqCst) {
        if started.elapsed() > std::time::Duration::from_millis(limits::RUN_MS) {
            send(RunEvent::Failed(format!("Stopped after {} s (time limit)", limits::RUN_MS / 1000)));
            return;
        }
        match rt.execute_pending_job() {
            Ok(true) => continue,
            Ok(false) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => {
                done.store(true, Ordering::SeqCst);
                if timed_out.load(Ordering::SeqCst) {
                    send(RunEvent::Failed(format!("Stopped after {} s (time limit)", limits::RUN_MS / 1000)));
                } else {
                    send(RunEvent::Done { ms: started.elapsed().as_millis() as u64, value: None, error: Some(format!("{e}")) });
                }
                return;
            }
        }
    }
    let (value, error) = outcome.lock().unwrap().take().unwrap_or((None, None));
    send(RunEvent::Done { ms: started.elapsed().as_millis() as u64, value, error });
}

/// The table of a CSV as Markdown rows (the panel's `csv -> table` path): the header line, then at
/// most [`limits::CSV_ROWS`] rows, and the line that says how many were left out.
pub fn csv_markdown(rows: &[Vec<String>]) -> String {
    let escape = |c: &str| c.replace('|', "\\|").replace('\n', " ");
    let shown: &[Vec<String>] = &rows[..rows.len().min(limits::CSV_ROWS)];
    if shown.is_empty() {
        return "The table is empty.".to_string();
    }
    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&shown[0].iter().map(|c| escape(c)).collect::<Vec<_>>().join(" | "));
    out.push_str(" |\n|");
    out.push_str(&shown[0].iter().map(|_| " --- ").collect::<Vec<_>>().join("|"));
    out.push_str("|\n");
    for r in &shown[1..] {
        out.push_str("| ");
        out.push_str(&r.iter().map(|c| escape(c)).collect::<Vec<_>>().join(" | "));
        out.push_str(" |\n");
    }
    if rows.len() > shown.len() {
        out.push_str(&format!("\nShowing the first {} of {} rows.", shown.len(), rows.len()));
    }
    out
}
