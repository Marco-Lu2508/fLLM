//! The tools of the Studio coding agent (`/api/studio/agent/*`): read, search and change files below one
//! workspace folder and run shell commands in it. The agent loop itself lives in the page; this module only
//! executes single tool calls. Every path is resolved against the workspace and may not leave it.
//!
//! These routes change files and run programs, so they are administration like the other changing Studio
//! routes: without an API key only this machine may call them (see `http::studio_admin_blocked`).
use super::*;
use regex::Regex;
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

const MAX_FILE_BYTES: u64 = 2 << 20;
const MAX_READ_LINES: usize = 2000;
const MAX_LINE_CHARS: usize = 2000;
const MAX_RESULTS: usize = 200;
const MAX_OUTPUT_BYTES: usize = 30_000;
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MAX_UNDO: usize = 50;
/// Folders that are never walked by `glob` and `grep` (build output and dependencies).
const SKIPPED_DIRS: [&str; 6] = [".git", "node_modules", "target", ".venv", "__pycache__", ".idea"];

struct UndoEntry {
    path: PathBuf,
    /// The content before the change; `None` when the file did not exist.
    before: Option<Vec<u8>>,
}

#[derive(Default)]
struct Inner {
    root: Option<PathBuf>,
    undo: Vec<UndoEntry>,
}

/// The workspace of the agent and the undo stack of its file changes.
#[derive(Default)]
pub(in crate::server) struct AgentSession {
    inner: Mutex<Inner>,
}

type ToolResult = Result<String, String>;

fn lock(m: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `.` and `..` removed without touching the file system.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Without the `\\?\` prefix Windows adds when it canonicalises a path.
fn plain(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => path,
    }
}

impl AgentSession {
    fn root(&self) -> Result<PathBuf, String> {
        lock(&self.inner)
            .root
            .clone()
            .ok_or_else(|| "no workspace folder is chosen yet; choose one first".to_owned())
    }

    /// Choose the workspace folder.
    pub(in crate::server) fn set_root(&self, path: &str) -> Result<PathBuf, String> {
        let candidate = PathBuf::from(path.trim());
        if path.trim().is_empty() || !candidate.is_absolute() {
            return Err("the workspace must be an absolute folder path".into());
        }
        let root = plain(fs::canonicalize(&candidate).map_err(|e| format!("cannot open {path}: {e}"))?);
        if !root.is_dir() {
            return Err(format!("{} is not a folder", root.display()));
        }
        let mut inner = lock(&self.inner);
        if inner.root.as_ref() != Some(&root) {
            inner.undo.clear();
        }
        inner.root = Some(root.clone());
        Ok(root)
    }

    pub(in crate::server) fn current_root(&self) -> Option<PathBuf> {
        lock(&self.inner).root.clone()
    }

    /// A path of the model (relative to the workspace or absolute) inside the workspace.
    fn resolve(&self, given: &str) -> Result<PathBuf, String> {
        let root = self.root()?;
        let given = given.trim();
        let joined = if Path::new(given).is_absolute() { PathBuf::from(given) } else { root.join(given) };
        let path = normalize(&joined);
        if !path.starts_with(&root) {
            return Err(format!("{given} is outside the workspace {}", root.display()));
        }
        // a link inside the workspace must not lead out of it: check the deepest part that exists
        let mut probe = path.as_path();
        while !probe.exists() {
            match probe.parent() {
                Some(parent) => probe = parent,
                None => break,
            }
        }
        if probe.exists() {
            let real = plain(fs::canonicalize(probe).map_err(|e| e.to_string())?);
            if !real.starts_with(&root) {
                return Err(format!("{given} leads outside the workspace through a link"));
            }
        }
        Ok(path)
    }

    fn shown(&self, path: &Path) -> String {
        match self.current_root() {
            Some(root) => path.strip_prefix(&root).unwrap_or(path).to_string_lossy().replace('\\', "/"),
            None => path.to_string_lossy().into_owned(),
        }
    }

    pub(in crate::server) fn run_tool(&self, name: &str, args: &Value) -> ToolResult {
        let text = |key: &str| args.get(key).and_then(Value::as_str);
        let need = |key: &str| text(key).ok_or_else(|| format!("the argument '{key}' is missing"));
        let number = |key: &str| args.get(key).and_then(Value::as_u64);
        match name {
            "list_dir" => self.list_dir(text("path").unwrap_or(".")),
            "read_file" => self.read_file(need("path")?, number("offset"), number("limit")),
            "write_file" => self.write_file(need("path")?, need("content")?),
            "edit_file" => self.edit_file(
                need("path")?,
                need("old_string")?,
                text("new_string").unwrap_or_default(),
                args.get("replace_all").and_then(Value::as_bool).unwrap_or(false),
            ),
            "glob" => self.glob(need("pattern")?),
            "grep" => self.grep(need("pattern")?, text("path"), text("glob")),
            "bash" => self.bash(need("command")?, number("timeout_ms")),
            other => Err(format!("unknown tool '{other}'")),
        }
    }

    fn list_dir(&self, path: &str) -> ToolResult {
        let dir = self.resolve(path)?;
        let mut entries: Vec<(bool, String, u64)> = fs::read_dir(&dir)
            .map_err(|e| format!("cannot list {}: {e}", self.shown(&dir)))?
            .filter_map(Result::ok)
            .map(|entry| {
                let meta = entry.metadata().ok();
                (
                    meta.as_ref().is_some_and(fs::Metadata::is_dir),
                    entry.file_name().to_string_lossy().into_owned(),
                    meta.map_or(0, |m| m.len()),
                )
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase())));
        let shown = self.shown(&dir);
        let mut out = if shown.is_empty() { "./:\n".to_owned() } else { format!("{}/:\n", shown.trim_end_matches('/')) };
        for (is_dir, name, size) in entries.iter().take(MAX_RESULTS) {
            if *is_dir {
                out.push_str(&format!("  {name}/\n"));
            } else {
                out.push_str(&format!("  {name}  ({size} bytes)\n"));
            }
        }
        if entries.len() > MAX_RESULTS {
            out.push_str(&format!("  ... and {} more\n", entries.len() - MAX_RESULTS));
        }
        if entries.is_empty() {
            out.push_str("  (empty)\n");
        }
        Ok(out)
    }

    fn read_text(&self, path: &Path) -> Result<String, String> {
        let meta = fs::metadata(path).map_err(|e| format!("cannot read {}: {e}", self.shown(path)))?;
        if meta.is_dir() {
            return Err(format!("{} is a folder; use list_dir", self.shown(path)));
        }
        if meta.len() > MAX_FILE_BYTES {
            return Err(format!("{} is larger than {} MB", self.shown(path), MAX_FILE_BYTES >> 20));
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.contains(&0) {
            return Err(format!("{} looks like a binary file", self.shown(path)));
        }
        String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8 text", self.shown(path)))
    }

    fn read_file(&self, path: &str, offset: Option<u64>, limit: Option<u64>) -> ToolResult {
        let path = self.resolve(path)?;
        let text = self.read_text(&path)?;
        let start = offset.unwrap_or(1).max(1) as usize - 1;
        let limit = (limit.unwrap_or(MAX_READ_LINES as u64) as usize).clamp(1, MAX_READ_LINES);
        let total = text.lines().count();
        let mut out = String::new();
        for (index, line) in text.lines().enumerate().skip(start).take(limit) {
            let mut shown: String = line.chars().take(MAX_LINE_CHARS).collect();
            if line.chars().count() > MAX_LINE_CHARS {
                shown.push_str(" ...");
            }
            out.push_str(&format!("{:>6}\t{shown}\n", index + 1));
        }
        if out.is_empty() {
            return Ok(if total == 0 { "(the file is empty)".into() } else { format!("(the file has only {total} lines)") });
        }
        if start + limit < total {
            out.push_str(&format!("... {} more lines; continue with offset {}\n", total - start - limit, start + limit + 1));
        }
        Ok(out)
    }

    fn remember(&self, path: &Path, before: Option<Vec<u8>>) {
        let mut inner = lock(&self.inner);
        inner.undo.push(UndoEntry { path: path.to_owned(), before });
        if inner.undo.len() > MAX_UNDO {
            inner.undo.remove(0);
        }
    }

    fn write_file(&self, path: &str, content: &str) -> ToolResult {
        let path = self.resolve(path)?;
        if content.len() as u64 > MAX_FILE_BYTES {
            return Err(format!("the content is larger than {} MB", MAX_FILE_BYTES >> 20));
        }
        if path.is_dir() {
            return Err(format!("{} is a folder", self.shown(&path)));
        }
        let before = fs::read(&path).ok();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("cannot create the folder: {e}"))?;
        }
        fs::write(&path, content).map_err(|e| format!("cannot write {}: {e}", self.shown(&path)))?;
        let verb = if before.is_some() { "Overwrote" } else { "Created" };
        self.remember(&path, before);
        Ok(format!("{verb} {} ({} lines)", self.shown(&path), content.lines().count()))
    }

    fn edit_file(&self, path: &str, old: &str, new: &str, replace_all: bool) -> ToolResult {
        let path = self.resolve(path)?;
        if old.is_empty() {
            return Err("old_string must not be empty; use write_file to create a file".into());
        }
        if old == new {
            return Err("old_string and new_string are the same".into());
        }
        let text = self.read_text(&path)?;
        let count = text.matches(old).count();
        if count == 0 {
            return Err(format!("old_string was not found in {}; read the file again and copy the text exactly", self.shown(&path)));
        }
        if count > 1 && !replace_all {
            return Err(format!(
                "old_string occurs {count} times in {}; add surrounding lines to make it unique, or set replace_all",
                self.shown(&path)
            ));
        }
        let changed = if replace_all { text.replace(old, new) } else { text.replacen(old, new, 1) };
        self.remember(&path, Some(text.into_bytes()));
        fs::write(&path, changed).map_err(|e| format!("cannot write {}: {e}", self.shown(&path)))?;
        Ok(format!("Edited {} ({count} replacement{})", self.shown(&path), if count == 1 { "" } else { "s" }))
    }

    /// Revert the latest file change; returns what was reverted.
    pub(in crate::server) fn undo(&self) -> ToolResult {
        let entry = lock(&self.inner).undo.pop().ok_or_else(|| "there is nothing to undo".to_owned())?;
        match &entry.before {
            Some(bytes) => fs::write(&entry.path, bytes).map_err(|e| e.to_string())?,
            None => fs::remove_file(&entry.path).map_err(|e| e.to_string())?,
        }
        Ok(format!(
            "{} {}",
            if entry.before.is_some() { "Restored" } else { "Removed" },
            self.shown(&entry.path)
        ))
    }

    /// Files below the workspace (without build output and dependencies) with their path relative to the workspace.
    fn walk(&self, start: &Path, visit: &mut dyn FnMut(&Path, &str) -> bool) {
        let root = self.current_root().unwrap_or_default();
        let mut stack = vec![start.to_owned()];
        while let Some(dir) = stack.pop() {
            let Ok(read) = fs::read_dir(&dir) else { continue };
            let mut entries: Vec<_> = read.filter_map(Result::ok).collect();
            entries.sort_by_key(|e| e.file_name());
            for entry in entries.into_iter().rev() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Ok(kind) = entry.file_type() else { continue };
                let path = entry.path();
                if kind.is_dir() {
                    if !SKIPPED_DIRS.contains(&name.as_str()) {
                        stack.push(path);
                    }
                } else if kind.is_file() {
                    let rel = path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                    if !visit(&path, &rel) {
                        return;
                    }
                }
            }
        }
    }

    fn glob(&self, pattern: &str) -> ToolResult {
        let matcher = glob_regex(pattern)?;
        let root = self.root()?;
        let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
        self.walk(&root, &mut |path, rel| {
            if matcher.is_match(rel) {
                let modified = fs::metadata(path).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                found.push((modified, rel.to_owned()));
            }
            found.len() < 5000
        });
        found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        if found.is_empty() {
            return Ok("no files match".into());
        }
        let more = found.len().saturating_sub(MAX_RESULTS);
        let mut out: String = found.iter().take(MAX_RESULTS).map(|(_, rel)| format!("{rel}\n")).collect();
        if more > 0 {
            out.push_str(&format!("... and {more} more\n"));
        }
        Ok(out)
    }

    fn grep(&self, pattern: &str, path: Option<&str>, glob: Option<&str>) -> ToolResult {
        let regex = Regex::new(pattern).map_err(|e| format!("the pattern is not a valid regular expression: {e}"))?;
        let filter = glob.map(glob_regex).transpose()?;
        let start = self.resolve(path.unwrap_or("."))?;
        let single = start.is_file().then(|| start.clone());
        let mut hits: Vec<String> = Vec::new();
        let mut total = 0usize;
        let mut scan = |path: &Path, rel: &str| -> bool {
            if let Some(filter) = &filter {
                if !filter.is_match(rel) && !filter.is_match(rel.rsplit('/').next().unwrap_or(rel)) {
                    return true;
                }
            }
            let Ok(meta) = fs::metadata(path) else { return true };
            if meta.len() > MAX_FILE_BYTES {
                return true;
            }
            let Ok(bytes) = fs::read(path) else { return true };
            if bytes.contains(&0) {
                return true;
            }
            let text = String::from_utf8_lossy(&bytes);
            for (index, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    total += 1;
                    if hits.len() < MAX_RESULTS {
                        let shown: String = line.trim_end().chars().take(300).collect();
                        hits.push(format!("{rel}:{}: {shown}", index + 1));
                    }
                }
            }
            total < 20_000
        };
        match single {
            Some(file) => {
                let rel = self.shown(&file);
                scan(&file, &rel);
            }
            None => self.walk(&start, &mut scan),
        }
        if hits.is_empty() {
            return Ok("no matches".into());
        }
        let mut out = hits.join("\n");
        out.push('\n');
        if total > hits.len() {
            out.push_str(&format!("... {} more matching lines\n", total - hits.len()));
        }
        Ok(out)
    }

    fn bash(&self, command: &str, timeout_ms: Option<u64>) -> ToolResult {
        let root = self.root()?;
        let limit = Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS).clamp(1000, MAX_TIMEOUT_MS));
        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = Command::new("sh");
            c.arg("-c").arg(command);
            c
        };
        let mut child = cmd
            .current_dir(&root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start the command: {e}"))?;
        let out_reader = collect_stream(child.stdout.take());
        let err_reader = collect_stream(child.stderr.take());
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if started.elapsed() >= limit => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => return Err(e.to_string()),
            }
        };
        let stdout = out_reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();
        let mut text = String::from_utf8_lossy(&stdout).into_owned();
        if !stderr.is_empty() {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&String::from_utf8_lossy(&stderr));
        }
        if text.len() > MAX_OUTPUT_BYTES {
            let mut cut = MAX_OUTPUT_BYTES;
            while !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
            text.push_str("\n... output cut");
        }
        match status {
            None => Err(format!("the command ran longer than {} s and was stopped\n{text}", limit.as_secs())),
            Some(status) if status.success() => Ok(if text.is_empty() { "(no output)".into() } else { text }),
            Some(status) => Err(format!("exit code {}\n{text}", status.code().map_or_else(|| "unknown".into(), |c| c.to_string()))),
        }
    }
}

/// Reads a process pipe to its end on its own thread (the output is capped, the pipe is always drained).
fn collect_stream<R: Read + Send + 'static>(stream: Option<R>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(mut stream) = stream {
            let mut chunk = [0u8; 8192];
            while let Ok(n) = stream.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if buffer.len() < MAX_OUTPUT_BYTES * 4 {
                    buffer.extend_from_slice(&chunk[..n]);
                }
            }
        }
        buffer
    })
}
/// `**` crosses folders, `*` and `?` stay inside one; the pattern is matched against the path relative to the workspace.
fn glob_regex(pattern: &str) -> Result<Regex, String> {
    let mut out = String::from("^");
    let chars: Vec<char> = pattern.trim_start_matches("./").chars().collect();
    let mut depth = 0usize;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' if chars.get(i + 1) == Some(&'*') => {
                i += 1;
                if chars.get(i + 1) == Some(&'/') {
                    i += 1;
                    out.push_str("(?:.*/)?");
                } else {
                    out.push_str(".*");
                }
            }
            '*' => out.push_str("[^/]*"),
            '?' => out.push_str("[^/]"),
            '{' => {
                depth += 1;
                out.push_str("(?:");
            }
            '}' if depth > 0 => {
                depth -= 1;
                out.push(')');
            }
            ',' if depth > 0 => out.push('|'),
            c => out.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    out.push('$');
    Regex::new(&out).map_err(|e| format!("bad glob pattern: {e}"))
}

/// `FLLM_NO_AGENT=1` switches the agent routes off (for servers that must never run commands or touch files).
fn agent_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLLM_NO_AGENT").is_some_and(|v| !v.is_empty() && v != "0"))
}

fn disabled_reply() -> Response {
    api_error(StatusCode::FORBIDDEN, "the coding agent is switched off on this server (FLLM_NO_AGENT)")
}

fn result_json(result: ToolResult) -> Value {
    match result {
        Ok(output) => json!({"ok": true, "output": output}),
        Err(error) => json!({"ok": false, "error": error}),
    }
}

fn workspace_json(state: &AppState) -> Value {
    match state.agent.current_root() {
        None => json!({"root": null}),
        Some(root) => {
            let head = fs::read_to_string(root.join(".git").join("HEAD")).ok();
            let branch = head.as_deref().and_then(|h| h.trim().strip_prefix("ref: refs/heads/")).map(str::to_owned);
            let name = root.file_name().map_or_else(|| root.display().to_string(), |n| n.to_string_lossy().into_owned());
            json!({"root": root.display().to_string(), "name": name, "git_branch": branch, "os": std::env::consts::OS, "shell": if cfg!(windows) { "cmd.exe" } else { "sh" }})
        }
    }
}

/// `POST /api/studio/agent/workspace`: `{path}` chooses the folder, an empty body reports the current one.
pub(in crate::server) async fn workspace(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    if let Some(path) = body.get("path").and_then(Value::as_str) {
        if let Err(error) = state.agent.set_root(path) {
            return api_error(StatusCode::BAD_REQUEST, &error);
        }
    }
    Json(workspace_json(&state)).into_response()
}

/// `POST /api/studio/agent/tool`: `{name, args}` runs one tool and returns `{ok, output}` or `{ok: false, error}`.
pub(in crate::server) async fn tool(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let Some(name) = body.get("name").and_then(Value::as_str).map(str::to_owned) else {
        return api_error(StatusCode::BAD_REQUEST, "the tool name is missing");
    };
    let args = body.get("args").cloned().unwrap_or_else(|| json!({}));
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || worker.agent.run_tool(&name, &args)).await {
        Ok(result) => Json(result_json(result)).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `POST /api/studio/agent/undo`: reverts the latest file change of the agent.
pub(in crate::server) async fn undo(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    Json(result_json(state.agent.undo())).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> (AgentSession, PathBuf) {
        let dir = std::env::temp_dir().join(format!("fllm-agent-{}-{}", std::process::id(), super::super::new_id("t")));
        fs::create_dir_all(dir.join("src")).unwrap();
        let session = AgentSession::default();
        let root = session.set_root(dir.to_str().unwrap()).unwrap();
        (session, root)
    }

    #[test]
    fn paths_stay_inside_the_workspace() {
        let (s, root) = session();
        assert!(s.resolve("src/main.rs").unwrap().starts_with(&root));
        assert!(s.resolve("./a/../b.txt").unwrap().starts_with(&root));
        assert!(s.resolve("../outside.txt").is_err());
        assert!(s.resolve("src/../../outside.txt").is_err());
        let elsewhere = std::env::temp_dir().join("elsewhere.txt");
        assert!(s.resolve(elsewhere.to_str().unwrap()).is_err());
        assert!(s.set_root("relative/path").is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_read_edit_and_undo() {
        let (s, root) = session();
        assert!(s.run_tool("write_file", &json!({"path": "src/a.txt", "content": "one\ntwo\nthree\n"})).unwrap().starts_with("Created"));
        let read = s.run_tool("read_file", &json!({"path": "src/a.txt"})).unwrap();
        assert!(read.contains("     2\ttwo"), "{read}");
        // a unique match is replaced, an ambiguous one is refused
        s.run_tool("edit_file", &json!({"path": "src/a.txt", "old_string": "two", "new_string": "2"})).unwrap();
        assert_eq!(fs::read_to_string(root.join("src/a.txt")).unwrap(), "one\n2\nthree\n");
        s.run_tool("write_file", &json!({"path": "src/b.txt", "content": "x x"})).unwrap();
        let many = s.run_tool("edit_file", &json!({"path": "src/b.txt", "old_string": "x", "new_string": "y"}));
        assert!(many.unwrap_err().contains("2 times"));
        s.run_tool("edit_file", &json!({"path": "src/b.txt", "old_string": "x", "new_string": "y", "replace_all": true})).unwrap();
        assert_eq!(fs::read_to_string(root.join("src/b.txt")).unwrap(), "y y");
        assert!(s.run_tool("edit_file", &json!({"path": "src/b.txt", "old_string": "zzz", "new_string": "y"})).is_err());
        // undo walks back: the edit of b, the creation of b, the edit of a
        assert!(s.undo().unwrap().starts_with("Restored"));
        assert_eq!(fs::read_to_string(root.join("src/b.txt")).unwrap(), "x x");
        assert!(s.undo().unwrap().starts_with("Removed"));
        assert!(!root.join("src/b.txt").exists());
        assert!(s.undo().unwrap().starts_with("Restored"));
        assert_eq!(fs::read_to_string(root.join("src/a.txt")).unwrap(), "one\ntwo\nthree\n");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn search_tools_find_files_and_lines() {
        let (s, root) = session();
        s.run_tool("write_file", &json!({"path": "src/lib.rs", "content": "fn alpha() {}\nfn beta() {}\n"})).unwrap();
        s.run_tool("write_file", &json!({"path": "notes/todo.md", "content": "alpha first\n"})).unwrap();
        s.run_tool("write_file", &json!({"path": "target/skip.rs", "content": "fn alpha() {}\n"})).unwrap();
        let globbed = s.run_tool("glob", &json!({"pattern": "**/*.rs"})).unwrap();
        assert!(globbed.contains("src/lib.rs") && !globbed.contains("target"), "{globbed}");
        let found = s.run_tool("grep", &json!({"pattern": "fn (alpha|beta)"})).unwrap();
        assert!(found.contains("src/lib.rs:1:") && found.contains("src/lib.rs:2:") && !found.contains("target"), "{found}");
        let only_md = s.run_tool("grep", &json!({"pattern": "alpha", "glob": "*.md"})).unwrap();
        assert!(only_md.contains("notes/todo.md:1:") && !only_md.contains("lib.rs"), "{only_md}");
        assert_eq!(s.run_tool("grep", &json!({"pattern": "nothing_like_this"})).unwrap(), "no matches");
        let listing = s.run_tool("list_dir", &json!({"path": "."})).unwrap();
        assert!(listing.contains("src/") && listing.contains("notes/"), "{listing}");
        assert!(s.run_tool("grep", &json!({"pattern": "("})).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shell_commands_run_in_the_workspace_with_a_timeout() {
        let (s, root) = session();
        let echoed = s.run_tool("bash", &json!({"command": "echo hello"})).unwrap();
        assert!(echoed.contains("hello"));
        let failed = s.run_tool("bash", &json!({"command": "echo oops 1>&2 && exit 3"})).unwrap_err();
        assert!(failed.contains("exit code 3") && failed.contains("oops"), "{failed}");
        let slow = if cfg!(windows) { "ping -n 6 127.0.0.1" } else { "sleep 5" };
        let stopped = s.run_tool("bash", &json!({"command": slow, "timeout_ms": 1000})).unwrap_err();
        assert!(stopped.contains("was stopped"), "{stopped}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn glob_patterns() {
        let m = |p: &str, s: &str| glob_regex(p).unwrap().is_match(s);
        assert!(m("*.rs", "main.rs") && !m("*.rs", "src/main.rs"));
        assert!(m("**/*.rs", "main.rs") && m("**/*.rs", "a/b/main.rs"));
        assert!(m("src/**", "src/a/b.rs") && !m("src/**", "other/a.rs"));
        assert!(m("*.{ts,tsx}", "app.tsx") && m("*.{ts,tsx}", "app.ts") && !m("*.{ts,tsx}", "app.js"));
        assert!(m("a?c.txt", "abc.txt") && !m("a?c.txt", "a/c.txt"));
    }
}
