//! The tools of the Studio coding agent (`/api/studio/agent/*`): read, search and change files below one
//! workspace folder and run shell commands in it. The agent loop itself lives in the page; this module only
//! executes single tool calls. Every path is resolved against the workspace and may not leave it.
//!
//! These routes change files and run programs, so they are administration like the other changing Studio
//! routes: without an API key only this machine may call them (see `http::studio_admin_blocked`).
use super::agent_edit;
use super::agent_ignore::{self, Chain};
use super::*;
use regex::Regex;
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    process::Stdio,
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
const MAX_CHANGES: usize = 500;
const MAX_NOTEBOOK_BYTES: u64 = 20 << 20;
const MAX_NOTEBOOK_OUTPUT_CHARS: usize = 2000;
/// Printed by the wrapper around a foreground command so the working directory can be carried to the next call.
const CWD_MARK: &str = "__FLLM_CWD__";
/// Folders that are never walked by `glob` and `grep` (build output and dependencies).
const SKIPPED_DIRS: [&str; 6] = [".git", "node_modules", "target", ".venv", "__pycache__", ".idea"];

/// One file change of the session: enough to put the file back as it was.
struct Change {
    id: u64,
    path: PathBuf,
    /// `create` or `edit`.
    kind: &'static str,
    /// Unix time in seconds.
    time: u64,
    /// The content before the change; `None` when the file did not exist.
    before: Option<Vec<u8>>,
}

#[derive(Default)]
struct Inner {
    root: Option<PathBuf>,
    /// Working directory of the shell (inside the root); `None` means the root itself.
    cwd: Option<PathBuf>,
    changes: Vec<Change>,
    next_id: u64,
    /// Changes that were undone and can be done again; any new change clears it. The last one is the next to redo.
    redo: Vec<Redo>,
}

/// An undone change: the content after it, so that it can be applied again.
struct Redo {
    id: u64,
    path: PathBuf,
    kind: &'static str,
    time: u64,
    before: Option<Vec<u8>>,
    after: Option<Vec<u8>>,
}

/// The workspace of the agent, its shell state, background jobs and the change log of the session.
#[derive(Default)]
pub(in crate::server) struct AgentSession {
    inner: Mutex<Inner>,
    jobs: super::agent_jobs::Jobs,
    /// Process ids of the foreground commands that run right now, so that a stop in the page can end them.
    foreground: Mutex<Vec<u32>>,
}

/// What `undo` put back.
pub(in crate::server) struct Restored {
    pub id: u64,
    pub path: String,
    /// `restored` (an older content is back) or `removed` (a created file is gone).
    pub action: &'static str,
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

type ToolResult = Result<String, String>;

fn lock(m: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A path as the model sees it: relative to the workspace, with forward slashes.
fn shown_under(root: Option<&Path>, path: &Path) -> String {
    match root {
        Some(root) => path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/"),
        None => path.to_string_lossy().into_owned(),
    }
}

/// `path` below `root` when it is inside it, with the spelling of `root` (Windows paths are not case sensitive:
/// `c:\work\x` is inside `C:\work`). `ci` switches the case-insensitive comparison on.
fn rebase_under(root: &Path, path: &Path, ci: bool) -> Option<PathBuf> {
    if path.starts_with(root) {
        return Some(path.to_owned());
    }
    if !ci {
        return None;
    }
    let (mut a, mut b) = (root.components(), path.components());
    loop {
        match (a.next(), b.next()) {
            (Some(x), Some(y)) if x.as_os_str().to_string_lossy().to_lowercase() == y.as_os_str().to_string_lossy().to_lowercase() => {}
            (None, rest) => {
                let mut out = root.to_owned();
                if let Some(first) = rest {
                    out.push(first.as_os_str());
                }
                out.extend(b.map(|c| c.as_os_str().to_owned()));
                return Some(out);
            }
            _ => return None,
        }
    }
}

/// The head and the tail of a long text; errors and summaries usually sit at the end of command output.
fn clip_middle(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let (head, tail) = (max * 2 / 5, max * 3 / 5);
    let mut a = head;
    while !text.is_char_boundary(a) {
        a -= 1;
    }
    let mut b = text.len() - tail;
    while !text.is_char_boundary(b) {
        b += 1;
    }
    format!("{}\n... [{} bytes cut from the middle] ...\n{}", &text[..a], b - a, &text[b..])
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
        let changed = {
            let mut inner = lock(&self.inner);
            let changed = inner.root.as_ref() != Some(&root);
            if changed {
                inner.changes.clear();
                inner.redo.clear();
                inner.cwd = None;
            }
            inner.root = Some(root.clone());
            changed
        };
        if changed {
            // a new workspace is a new session: nothing started in the old one keeps running
            self.jobs.kill_all();
        }
        Ok(root)
    }

    /// Ends the session's background jobs; returns how many were still running.
    pub(in crate::server) fn end_session(&self) -> usize {
        self.jobs.kill_all()
    }

    /// The folder shell commands start in: the last `cd` of a command, or the workspace root.
    fn shell_dir(&self) -> Result<PathBuf, String> {
        let root = self.root()?;
        let cwd = lock(&self.inner).cwd.clone();
        Ok(cwd.filter(|dir| dir.starts_with(&root) && dir.is_dir()).unwrap_or(root))
    }

    pub(in crate::server) fn current_root(&self) -> Option<PathBuf> {
        lock(&self.inner).root.clone()
    }

    /// A path of the model (relative to the workspace or absolute) inside the workspace.
    fn resolve(&self, given: &str) -> Result<PathBuf, String> {
        let root = self.root()?;
        let given = given.trim();
        let joined = if Path::new(given).is_absolute() { PathBuf::from(given) } else { root.join(given) };
        let path = rebase_under(&root, &normalize(&joined), cfg!(windows))
            .ok_or_else(|| format!("{given} is outside the workspace {}", root.display()))?;
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
        shown_under(self.current_root().as_deref(), path)
    }

    /// The workspace folder (for the sibling modules of the agent).
    pub(super) fn root_dir(&self) -> Result<PathBuf, String> {
        self.root()
    }

    /// A path of the model resolved inside the workspace (for the sibling modules of the agent).
    pub(super) fn resolve_path(&self, given: &str) -> Result<PathBuf, String> {
        self.resolve(given)
    }

    /// Ends the foreground commands that run right now (the user pressed Stop); returns how many.
    pub(in crate::server) fn cancel_foreground(&self) -> usize {
        let pids: Vec<u32> = std::mem::take(&mut *self.foreground.lock().unwrap_or_else(|e| e.into_inner()));
        for pid in &pids {
            super::agent_jobs::kill_tree(*pid);
        }
        pids.len()
    }

    /// The ignore rules that apply in `dir`.
    fn ignore_chain(&self, dir: &Path) -> Chain {
        match self.current_root() {
            Some(root) if dir.starts_with(&root) => Chain::for_dir(&root, dir),
            _ => Chain::default(),
        }
    }

    pub(in crate::server) fn run_tool(&self, name: &str, args: &Value) -> ToolResult {
        let dir = self.current_root();
        let result = self.run_tool_inner(name, args);
        let post = super::hooks::dispatch("post_tool_call", name, args, dir.as_deref());
        let result = super::hooks::with_notes(result, &post);
        if super::hooks::is_edit_tool(name) && result.is_ok() {
            let edited = super::hooks::dispatch("on_file_edit", name, args, dir.as_deref());
            return super::hooks::with_notes(result, &edited);
        }
        result
    }

    fn run_tool_inner(&self, name: &str, args: &Value) -> ToolResult {
        let text = |key: &str| args.get(key).and_then(Value::as_str);
        let need = |key: &str| text(key).ok_or_else(|| format!("the argument '{key}' is missing"));
        let number = |key: &str| args.get(key).and_then(Value::as_u64);
        match name {
            "list_dir" => self.list_dir(text("path").unwrap_or("."), number("depth")),
            "read_file" => self.read_file(need("path")?, number("offset"), number("limit")),
            "write_file" => self.write_file(need("path")?, need("content")?),
            "patch" => self.patch(need("patch")?),
            "diagnostics" => super::agent_diag::run(self, text("path"), number("timeout_s")),
            "review" => super::agent_review::run(self, args),
            "lsp" => super::agent_lsp::run(self, args),
            "notebook_edit" => super::agent_notebook::run(self, args),
            "mcp_resource" => super::mcp::read_tool(args),
            "session_note" => super::agent_messages::tool(self, args),
            "memory" => super::agent_mem::tool(self, args),
            "edit_file" => self.edit_file(
                need("path")?,
                need("old_string")?,
                text("new_string").unwrap_or_default(),
                args.get("replace_all").and_then(Value::as_bool).unwrap_or(false),
            ),
            "multi_edit" => self.multi_edit(need("path")?, args.get("edits")),
            "glob" => self.glob(need("pattern")?),
            "grep" => self.grep(need("pattern")?, text("path"), text("glob"), &GrepOptions::from_args(args)?),
            "bash" if args.get("background").and_then(Value::as_bool).unwrap_or(false) => {
                self.jobs.start(need("command")?, &self.shell_dir()?)
            }
            "bash" => self.bash(need("command")?, number("timeout_ms")),
            "bash_output" => self.jobs.output(need("job_id")?, number("since")),
            "bash_kill" => self.jobs.kill(need("job_id")?),
            other => Err(format!("unknown tool '{other}'")),
        }
    }

    fn list_dir(&self, path: &str, depth: Option<u64>) -> ToolResult {
        let dir = self.resolve(path)?;
        let depth = depth.unwrap_or(1).clamp(1, 4) as usize;
        let chain = self.ignore_chain(&dir);
        let shown = self.shown(&dir);
        let mut out = if shown.is_empty() { "./:\n".to_owned() } else { format!("{}/:\n", shown.trim_end_matches('/')) };
        let mut lines = 0usize;
        let mut cut = false;
        self.tree(&dir, &chain, 1, depth, &mut out, &mut lines, &mut cut)?;
        if lines == 0 {
            out.push_str("  (empty)\n");
        }
        if cut {
            out.push_str("  ... the listing was cut; list a sub folder for more\n");
        }
        Ok(out)
    }

    /// One level of the folder tree; folders first, then files, each sorted by name. Ignored and build folders
    /// are listed but not opened below the first level.
    #[allow(clippy::too_many_arguments)]
    fn tree(&self, dir: &Path, chain: &Chain, level: usize, depth: usize, out: &mut String, lines: &mut usize, cut: &mut bool) -> Result<(), String> {
        let mut entries: Vec<(bool, String, u64)> = fs::read_dir(dir)
            .map_err(|e| format!("cannot list {}: {e}", self.shown(dir)))?
            .filter_map(Result::ok)
            .map(|entry| {
                // `metadata` follows links: a link to a folder is a folder
                let meta = entry.metadata().ok();
                (meta.as_ref().is_some_and(fs::Metadata::is_dir), entry.file_name().to_string_lossy().into_owned(), meta.map_or(0, |m| m.len()))
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase())));
        let limit = if depth > 1 { MAX_RESULTS * 2 } else { MAX_RESULTS };
        let pad = "  ".repeat(level);
        for (is_dir, name, size) in &entries {
            if *lines >= limit {
                *cut = true;
                return Ok(());
            }
            *lines += 1;
            if !*is_dir {
                out.push_str(&format!("{pad}{name}  ({size} bytes)\n"));
                continue;
            }
            let child = dir.join(name);
            let hidden = SKIPPED_DIRS.contains(&name.as_str()) || chain.ignores(&child, true);
            if level < depth && !hidden {
                out.push_str(&format!("{pad}{name}/\n"));
                let below = chain.entering(&child);
                self.tree(&child, &below, level + 1, depth, out, lines, cut)?;
            } else {
                let note = if hidden && depth > 1 { "  (ignored, not opened)" } else { "" };
                out.push_str(&format!("{pad}{name}/{note}\n"));
            }
        }
        Ok(())
    }

    /// Names in the folder of a missing file that look like what the model meant.
    fn similar_names(&self, path: &Path) -> String {
        let (Some(parent), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str())) else { return String::new() };
        let wanted = name.to_lowercase();
        let stem_of = |s: &str| s.rsplit_once('.').map_or(s.to_owned(), |(a, _)| a.to_owned());
        let stem = stem_of(&wanted);
        let Ok(read) = fs::read_dir(parent) else {
            return " The folder does not exist either; use list_dir or glob to find the file.".into();
        };
        let mut names: Vec<String> = read
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| {
                let l = n.to_lowercase();
                let s = stem_of(&l);
                l == wanted || (stem.len() >= 2 && s.len() >= 2 && (s.contains(&stem) || stem.contains(&s)))
            })
            .collect();
        names.sort();
        names.truncate(4);
        if names.is_empty() {
            " Use list_dir or glob to find the file.".into()
        } else {
            format!(" Did you mean: {}?", names.join(", "))
        }
    }

    fn read_bytes(&self, path: &Path, max: u64) -> Result<Vec<u8>, String> {
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(format!("{} does not exist.{}", self.shown(path), self.similar_names(path))),
            Err(e) => return Err(format!("cannot read {}: {e}", self.shown(path))),
        };
        if meta.is_dir() {
            return Err(format!("{} is a folder; use list_dir", self.shown(path)));
        }
        if meta.len() > max {
            return Err(format!(
                "{} is {} MB, larger than the {} MB limit; search it with grep or read parts of it with bash (for example sed -n '1,100p')",
                self.shown(path),
                meta.len() >> 20,
                max >> 20
            ));
        }
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", self.shown(path)))?;
        if bytes.contains(&0) {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
            let kind = if ["png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tif", "tiff"].contains(&ext.as_str()) { "an image" } else { "a binary file" };
            return Err(format!("{} is {kind} ({} bytes); this tool reads text only", self.shown(path), bytes.len()));
        }
        Ok(bytes)
    }

    fn read_text(&self, path: &Path) -> Result<String, String> {
        self.read_text_limited(path, MAX_FILE_BYTES)
    }

    fn read_text_limited(&self, path: &Path, max: u64) -> Result<String, String> {
        let bytes = self.read_bytes(path, max)?;
        String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8 text (read_file shows it with replacement characters; change it with bash)", self.shown(path)))
    }

    fn read_file(&self, path: &str, offset: Option<u64>, limit: Option<u64>) -> ToolResult {
        let path = self.resolve(path)?;
        let is_notebook = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ipynb"));
        let mut lossy = false;
        let text = if is_notebook {
            notebook_text(&self.read_text_limited(&path, MAX_NOTEBOOK_BYTES)?).map_err(|e| format!("{} is not a readable notebook: {e}", self.shown(&path)))?
        } else {
            let bytes = self.read_bytes(&path, MAX_FILE_BYTES)?;
            match String::from_utf8(bytes) {
                Ok(t) => t,
                Err(e) => {
                    lossy = true;
                    String::from_utf8_lossy(e.as_bytes()).into_owned()
                }
            }
        };
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
            return Ok(if total == 0 { "(the file is empty)".into() } else { format!("(the file has only {total} lines; offset {} is past the end)", start + 1) });
        }
        if start + limit < total {
            out.push_str(&format!("... {} more lines; continue with offset {}\n", total - start - limit, start + limit + 1));
        } else {
            out.push_str(&format!("(end of file, {total} lines)\n"));
        }
        if lossy {
            out.insert_str(0, "(the file is not valid UTF-8; some characters are replaced)\n");
        }
        Ok(out)
    }

    fn remember(&self, path: &Path, before: Option<Vec<u8>>) {
        let mut inner = lock(&self.inner);
        inner.next_id += 1;
        let id = inner.next_id;
        let kind = if before.is_some() { "edit" } else { "create" };
        inner.changes.push(Change { id, path: path.to_owned(), kind, time: unix_now(), before });
        inner.redo.clear();
        if inner.changes.len() > MAX_CHANGES {
            inner.changes.remove(0);
        }
    }

    /// The file changes of this session, oldest first.
    pub(in crate::server) fn changes(&self) -> Vec<Value> {
        let root = self.current_root();
        let inner = lock(&self.inner);
        inner
            .changes
            .iter()
            .map(|c| json!({"id": c.id, "path": shown_under(root.as_deref(), &c.path), "kind": c.kind, "time": c.time}))
            .collect()
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
        let shown = self.shown(&path);
        let raw = self.read_text(&path)?;
        let file = agent_edit::normalize_eol(&raw);
        // a model never sees the carriage returns of a CRLF file
        let (old, new) = if file.crlf { (old.replace("\r\n", "\n"), new.replace("\r\n", "\n")) } else { (old.to_owned(), new.to_owned()) };
        let edited = agent_edit::replace(&file.text, &old, &new, replace_all, &shown)?;
        fs::write(&path, agent_edit::restore_eol(edited.text, file.crlf)).map_err(|e| format!("cannot write {shown}: {e}"))?;
        self.remember(&path, Some(raw.into_bytes()));
        Ok(format!("Edited {shown} ({} replacement{}){}", edited.count, if edited.count == 1 { "" } else { "s" }, edited.note))
    }

    /// Several edits of one file, applied in order to the result of the previous one. All or nothing: the file
    /// is written once, after every edit worked, and counts as one undo step.
    fn multi_edit(&self, path: &str, edits: Option<&Value>) -> ToolResult {
        let path = self.resolve(path)?;
        let edits = edits.and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or("'edits' must be a non-empty list")?;
        let raw = self.read_text(&path)?;
        let shown = self.shown(&path);
        let file = agent_edit::normalize_eol(&raw);
        let mut current = file.text.clone();
        let mut replacements = 0usize;
        let mut note = "";
        for (index, edit) in edits.iter().enumerate() {
            let field = |key: &str| edit.get(key).and_then(Value::as_str);
            let old = field("old_string").ok_or_else(|| format!("edit {}: 'old_string' is missing (nothing was changed)", index + 1))?;
            let new = field("new_string").unwrap_or_default();
            let all = edit.get("replace_all").and_then(Value::as_bool).unwrap_or(false);
            let (old, new) = if file.crlf { (old.replace("\r\n", "\n"), new.replace("\r\n", "\n")) } else { (old.to_owned(), new.to_owned()) };
            let edited = agent_edit::replace(&current, &old, &new, all, &shown)
                .map_err(|e| format!("edit {} of {}: {e} (nothing was changed)", index + 1, edits.len()))?;
            current = edited.text;
            replacements += edited.count;
            if !edited.note.is_empty() {
                note = edited.note;
            }
        }
        fs::write(&path, agent_edit::restore_eol(current, file.crlf)).map_err(|e| format!("cannot write {shown}: {e}"))?;
        self.remember(&path, Some(raw.into_bytes()));
        Ok(format!("Edited {shown} ({} edits, {replacements} replacements){note}", edits.len()))
    }

    /// Applies a unified diff to one or more files, all or nothing; every changed file is one undo step.
    fn patch(&self, patch: &str) -> ToolResult {
        let files = agent_edit::parse_patch(patch)?;
        // plan every file first, write only when all of them work
        type Planned = (PathBuf, String, Option<Vec<u8>>, Option<String>); // path, shown, before, new text (None: delete)
        let mut plan: Vec<Planned> = Vec::new();
        let mut notes = Vec::new();
        for file in &files {
            let path = self.resolve(&file.path)?;
            let shown = self.shown(&path);
            if plan.iter().any(|p| p.0 == path) {
                return Err(format!("{shown} appears twice in the patch; put all its hunks under one header"));
            }
            if file.create {
                if path.exists() {
                    return Err(format!("{shown} already exists, but the patch creates it; use hunks against the current content"));
                }
                let (text, _) = agent_edit::apply_hunks("", &file.hunks, &shown)?;
                plan.push((path, shown, None, Some(text)));
                continue;
            }
            let raw = self.read_text(&path)?;
            if file.delete {
                plan.push((path, shown, Some(raw.into_bytes()), None));
                continue;
            }
            let normalized = agent_edit::normalize_eol(&raw);
            let (text, note) = agent_edit::apply_hunks(&normalized.text, &file.hunks, &shown)?;
            if !note.is_empty() {
                notes.push(format!("{shown}{note}"));
            }
            plan.push((path, shown, Some(raw.into_bytes()), Some(agent_edit::restore_eol(text, normalized.crlf))));
        }
        let mut report = Vec::new();
        for (path, shown, before, after) in plan {
            match after {
                Some(text) => {
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent).map_err(|e| format!("cannot create the folder of {shown}: {e}"))?;
                    }
                    fs::write(&path, text).map_err(|e| format!("cannot write {shown}: {e}"))?;
                    report.push(format!("{} {shown}", if before.is_some() { "Patched" } else { "Created" }));
                }
                None => {
                    fs::remove_file(&path).map_err(|e| format!("cannot delete {shown}: {e}"))?;
                    report.push(format!("Deleted {shown}"));
                }
            }
            self.remember(&path, before);
        }
        let mut out = report.join("\n");
        if !notes.is_empty() {
            out.push_str(&format!("\n(matched with tolerance: {})", notes.join("; ")));
        }
        Ok(out)
    }

    /// Put back the latest change; returns what was reverted.
    #[cfg(test)]
    fn undo(&self) -> ToolResult {
        let restored = self.undo_many(Some(1), None)?;
        Ok(restored.iter().map(|r| format!("{} {}", if r.action == "removed" { "Removed" } else { "Restored" }, r.path)).collect::<Vec<_>>().join("\n"))
    }

    /// Revert the last `steps` changes, or every change from the one with id `to` on (that change included),
    /// newest first. Without either, one step.
    pub(in crate::server) fn undo_many(&self, steps: Option<u64>, to: Option<u64>) -> Result<Vec<Restored>, String> {
        if steps.is_some() && to.is_some() {
            return Err("give either 'steps' or 'to', not both".into());
        }
        let root = self.current_root();
        let mut inner = lock(&self.inner);
        if inner.changes.is_empty() {
            return Err("there is nothing to undo".into());
        }
        let keep = match to {
            Some(id) => inner.changes.iter().position(|c| c.id == id).ok_or_else(|| format!("there is no change with id {id}"))?,
            None => {
                let steps = steps.unwrap_or(1);
                if steps == 0 {
                    return Err("'steps' must be at least 1".into());
                }
                inner.changes.len().saturating_sub(usize::try_from(steps).unwrap_or(usize::MAX))
            }
        };
        let mut restored = Vec::new();
        while inner.changes.len() > keep {
            let Some(change) = inner.changes.pop() else { break };
            let after = fs::read(&change.path).ok();
            let outcome = match &change.before {
                Some(bytes) => fs::write(&change.path, bytes),
                None => fs::remove_file(&change.path).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(e) }),
            };
            let path = shown_under(root.as_deref(), &change.path);
            if let Err(e) = outcome {
                let done = restored.iter().map(|r: &Restored| r.path.clone()).collect::<Vec<_>>().join(", ");
                inner.changes.push(change);
                return Err(format!("cannot restore {path}: {e}{}", if done.is_empty() { String::new() } else { format!(" (already restored: {done})") }));
            }
            restored.push(Restored { id: change.id, path, action: if change.before.is_some() { "restored" } else { "removed" } });
            inner.redo.push(Redo { id: change.id, path: change.path, kind: change.kind, time: change.time, before: change.before, after });
            if inner.redo.len() > MAX_CHANGES {
                inner.redo.remove(0);
            }
        }
        Ok(restored)
    }

    /// Apply the last `count` undone changes again, oldest first.
    pub(in crate::server) fn redo_many(&self, count: u64) -> Result<Vec<Restored>, String> {
        let root = self.current_root();
        let mut inner = lock(&self.inner);
        if inner.redo.is_empty() {
            return Err("there is nothing to redo (a new change clears the undone ones)".into());
        }
        let mut done = Vec::new();
        for _ in 0..count.max(1) {
            let Some(r) = inner.redo.pop() else { break };
            let outcome = match &r.after {
                Some(bytes) => fs::write(&r.path, bytes),
                None => fs::remove_file(&r.path).or_else(|e| if e.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(e) }),
            };
            let path = shown_under(root.as_deref(), &r.path);
            if let Err(e) = outcome {
                inner.redo.push(r);
                return Err(format!("cannot redo {path}: {e}"));
            }
            done.push(Restored { id: r.id, path, action: if r.after.is_some() { "restored" } else { "removed" } });
            inner.changes.push(Change { id: r.id, path: r.path, kind: r.kind, time: r.time, before: r.before });
        }
        Ok(done)
    }

    /// Files below `start` (without build output, dependencies and what `.gitignore` / `.ignore` exclude) with their
    /// path relative to the workspace. Links are not followed.
    pub(super) fn walk(&self, start: &Path, visit: &mut dyn FnMut(&Path, &str) -> bool) {
        let root = self.current_root().unwrap_or_default();
        let mut stack = vec![(start.to_owned(), self.ignore_chain(start))];
        while let Some((dir, chain)) = stack.pop() {
            let Ok(read) = fs::read_dir(&dir) else { continue };
            let mut entries: Vec<_> = read.filter_map(Result::ok).collect();
            entries.sort_by_key(|e| e.file_name());
            // files of this folder in name order first, then the sub folders in name order
            let mut folders = Vec::new();
            for entry in entries {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Ok(kind) = entry.file_type() else { continue };
                let path = entry.path();
                if kind.is_dir() {
                    if !SKIPPED_DIRS.contains(&name.as_str()) && !chain.ignores(&path, true) {
                        let below = chain.entering(&path);
                        folders.push((path, below));
                    }
                } else if kind.is_file() && !chain.ignores(&path, false) {
                    let rel = path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                    if !visit(&path, &rel) {
                        return;
                    }
                }
            }
            stack.extend(folders.into_iter().rev());
        }
    }

    /// The files of the workspace that match `query` (fuzzy), best first, plus the folders above them.
    pub(in crate::server) fn find_files(&self, query: &str, limit: usize) -> Result<Vec<String>, String> {
        let root = self.root()?;
        let mut all: Vec<String> = Vec::new();
        self.walk(&root, &mut |_, rel| {
            all.push(rel.to_owned());
            all.len() < 30_000
        });
        let mut folders = std::collections::BTreeSet::new();
        for rel in &all {
            let mut at = rel.as_str();
            while let Some((dir, _)) = at.rsplit_once('/') {
                if !folders.insert(format!("{dir}/")) {
                    break;
                }
                at = dir;
            }
        }
        all.extend(folders);
        Ok(agent_ignore::rank(query, &all, limit.clamp(1, 100)))
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

    fn grep(&self, pattern: &str, path: Option<&str>, glob: Option<&str>, options: &GrepOptions) -> ToolResult {
        let mut source = if options.literal { regex::escape(pattern) } else { pattern.to_owned() };
        if options.word {
            source = format!(r"\b(?:{source})\b");
        }
        let regex = regex::RegexBuilder::new(&source)
            .case_insensitive(options.case_insensitive)
            .build()
            .map_err(|e| format!("the pattern is not a valid regular expression: {e}"))?;
        let filter = glob.map(glob_regex).transpose()?;
        let start = self.resolve(path.unwrap_or("."))?;
        let single = start.is_file().then(|| start.clone());
        let mut hits: Vec<String> = Vec::new();
        let mut files: Vec<(String, usize)> = Vec::new();
        let mut total = 0usize;
        let mut shown_matches = 0usize;
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
            let lines: Vec<&str> = text.lines().collect();
            let matched: Vec<usize> = (0..lines.len()).filter(|&i| regex.is_match(lines[i])).collect();
            if matched.is_empty() {
                return true;
            }
            total += matched.len();
            files.push((rel.to_owned(), matched.len()));
            if options.mode == GrepMode::Content && shown_matches < options.limit {
                let clip = |line: &str| -> String { line.trim_end().chars().take(300).collect() };
                if options.context > 0 && !hits.is_empty() {
                    hits.push("--".into());
                }
                let mut printed_until = 0usize; // lines before this index are already in the output
                for &index in &matched {
                    if shown_matches >= options.limit {
                        break;
                    }
                    let from = index.saturating_sub(options.context).max(printed_until);
                    if options.context > 0 && printed_until > 0 && from > printed_until {
                        hits.push("--".into());
                    }
                    for (n, line) in lines.iter().enumerate().take(index).skip(from) {
                        hits.push(format!("{rel}-{}- {}", n + 1, clip(line)));
                    }
                    hits.push(format!("{rel}:{}: {}", index + 1, clip(lines[index])));
                    shown_matches += 1;
                    let until = (index + 1 + options.context).min(lines.len());
                    let next_match = matched.iter().copied().find(|&m| m > index).unwrap_or(usize::MAX);
                    for (n, line) in lines.iter().enumerate().take(until.min(next_match)).skip(index + 1) {
                        hits.push(format!("{rel}-{}- {}", n + 1, clip(line)));
                    }
                    printed_until = until.min(next_match).max(index + 1);
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
        if total == 0 {
            return Ok("no matches".into());
        }
        let mut out = String::new();
        match options.mode {
            GrepMode::Content => {
                out = hits.join("\n");
                out.push('\n');
                if total > shown_matches {
                    out.push_str(&format!("... {} more matching lines\n", total - shown_matches));
                }
            }
            GrepMode::Files => {
                for (rel, _) in files.iter().take(options.limit) {
                    out.push_str(&format!("{rel}\n"));
                }
                if files.len() > options.limit {
                    out.push_str(&format!("... {} more files\n", files.len() - options.limit));
                }
            }
            GrepMode::Count => {
                for (rel, n) in files.iter().take(options.limit) {
                    out.push_str(&format!("{rel}:{n}\n"));
                }
                if files.len() > options.limit {
                    out.push_str(&format!("... {} more files\n", files.len() - options.limit));
                }
                out.push_str(&format!("total: {total} matching lines in {} files\n", files.len()));
            }
        }
        Ok(out)
    }

    fn bash(&self, command: &str, timeout_ms: Option<u64>) -> ToolResult {
        let root = self.root()?;
        let dir = self.shell_dir()?;
        let limit = Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS).clamp(1000, MAX_TIMEOUT_MS));
        let mut cmd = super::agent_jobs::shell_command_in(&super::agent_jobs::with_cwd_report(command, CWD_MARK), &root, &dir)?;
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start the command: {e}"))?;
        let out_reader = collect_output(child.stdout.take());
        let err_reader = collect_output(child.stderr.take());
        let pid = child.id();
        self.foreground.lock().unwrap_or_else(|e| e.into_inner()).push(pid);
        let started = Instant::now();
        let mut timed_out = false;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if started.elapsed() >= limit => {
                    // the whole process tree: a program the shell started would keep the output pipes open
                    super::agent_jobs::kill_tree(pid);
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => {
                    self.foreground.lock().unwrap_or_else(|e| e.into_inner()).retain(|p| *p != pid);
                    return Err(e.to_string());
                }
            }
        };
        // a pid that is gone from the list was ended by `cancel_foreground`
        let cancelled = {
            let mut running = self.foreground.lock().unwrap_or_else(|e| e.into_inner());
            let before = running.len();
            running.retain(|p| *p != pid);
            running.len() == before
        };
        let stdout = out_reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();
        let (stdout, reported) = split_cwd_report(&stdout, CWD_MARK);
        let mut moved = String::new();
        if let Some(new_dir) = reported.and_then(|p| fs::canonicalize(p).ok()).map(plain) {
            // a `cd` out of the workspace is ignored: the next command starts where this one started
            if new_dir.starts_with(&root) && new_dir.is_dir() && new_dir != dir {
                moved = format!("\n(working directory is now {})", match self.shown(&new_dir).as_str() {
                    "" => ".".to_owned(),
                    other => other.to_owned(),
                });
                lock(&self.inner).cwd = Some(new_dir);
            }
        }
        let mut text = String::from_utf8_lossy(&stdout).into_owned();
        if !stderr.is_empty() {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&String::from_utf8_lossy(&stderr));
        }
        let text = clip_middle(&text, MAX_OUTPUT_BYTES);
        match status {
            _ if cancelled && !timed_out => Err(format!("the command was stopped by the user\n{text}")),
            None => Err(format!("the command ran longer than {} s and was stopped\n{text}", limit.as_secs())),
            Some(status) if status.success() => Ok(if text.is_empty() { format!("(no output){moved}") } else { format!("{text}{moved}") }),
            Some(status) => Err(format!("exit code {}\n{text}{moved}", status.code().map_or_else(|| "unknown".into(), |c| c.to_string()))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GrepMode {
    Content,
    Files,
    Count,
}

/// The optional arguments of `grep`.
struct GrepOptions {
    mode: GrepMode,
    context: usize,
    case_insensitive: bool,
    /// The pattern is plain text, not a regular expression.
    literal: bool,
    /// Only whole words match.
    word: bool,
    limit: usize,
}

impl GrepOptions {
    fn from_args(args: &Value) -> Result<Self, String> {
        let mode = match args.get("output_mode").and_then(Value::as_str).unwrap_or("content") {
            "content" => GrepMode::Content,
            "files" | "files_with_matches" => GrepMode::Files,
            "count" => GrepMode::Count,
            other => return Err(format!("output_mode must be content, files or count, not '{other}'")),
        };
        let number = |key: &str| args.get(key).and_then(Value::as_u64).map(|n| usize::try_from(n).unwrap_or(usize::MAX));
        Ok(Self {
            mode,
            context: number("context").unwrap_or(0).min(20),
            case_insensitive: args.get("case_insensitive").and_then(Value::as_bool).unwrap_or(false),
            literal: args.get("literal").and_then(Value::as_bool).unwrap_or(false),
            word: args.get("word").and_then(Value::as_bool).unwrap_or(false),
            limit: number("head_limit").filter(|&n| n > 0).unwrap_or(MAX_RESULTS).min(2000),
        })
    }
}

/// A Jupyter notebook as text: one block per cell with its source and the text of its outputs (no images).
fn notebook_text(raw: &str) -> Result<String, String> {
    let notebook: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    let cells = notebook.get("cells").and_then(Value::as_array).ok_or("it has no 'cells' list")?;
    // `source` and output `text` are a string or a list of strings
    let joined = |value: Option<&Value>| -> String {
        match value {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts.iter().filter_map(Value::as_str).collect(),
            _ => String::new(),
        }
    };
    let clip = |s: String| -> String {
        if s.chars().count() > MAX_NOTEBOOK_OUTPUT_CHARS {
            let mut short: String = s.chars().take(MAX_NOTEBOOK_OUTPUT_CHARS).collect();
            short.push_str("\n... output cut");
            short
        } else {
            s
        }
    };
    let mut out = String::new();
    for (index, cell) in cells.iter().enumerate() {
        let kind = cell.get("cell_type").and_then(Value::as_str).unwrap_or("code");
        out.push_str(&format!("# [cell {} | {kind}]\n", index + 1));
        out.push_str(joined(cell.get("source")).trim_end());
        out.push('\n');
        for output in cell.get("outputs").and_then(Value::as_array).into_iter().flatten() {
            let text = match output.get("output_type").and_then(Value::as_str) {
                Some("stream") => joined(output.get("text")),
                Some("error") => format!(
                    "{}: {}",
                    output.get("ename").and_then(Value::as_str).unwrap_or("error"),
                    output.get("evalue").and_then(Value::as_str).unwrap_or_default()
                ),
                _ => joined(output.get("data").and_then(|d| d.get("text/plain"))),
            };
            if !text.trim().is_empty() {
                out.push_str("# output:\n");
                out.push_str(clip(text).trim_end());
                out.push('\n');
            }
        }
        out.push('\n');
    }
    if cells.is_empty() {
        out.push_str("(the notebook has no cells)\n");
    }
    Ok(out)
}

/// Cuts the working-directory report the wrapper printed at the end of the output.
fn split_cwd_report(stdout: &[u8], mark: &str) -> (Vec<u8>, Option<PathBuf>) {
    let needle = format!("\n{mark}");
    let text = String::from_utf8_lossy(stdout);
    match text.rfind(&needle) {
        Some(at) => {
            let report = text[at + needle.len()..].trim();
            let dir = (!report.is_empty()).then(|| PathBuf::from(report));
            (text.as_bytes()[..at].to_vec(), dir)
        }
        None => (stdout.to_vec(), None),
    }
}

/// Reads a process pipe to its end on its own thread (the output is capped, the pipe is always drained).
pub(super) fn collect_output<R: Read + Send + 'static>(stream: Option<R>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        // the beginning and the end are kept: what a build prints last is usually what matters
        let (mut head, mut tail, mut dropped) = (Vec::new(), Vec::new(), false);
        if let Some(mut stream) = stream {
            let mut chunk = [0u8; 8192];
            while let Ok(n) = stream.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let room = MAX_OUTPUT_BYTES.saturating_sub(head.len());
                let take = room.min(n);
                head.extend_from_slice(&chunk[..take]);
                tail.extend_from_slice(&chunk[take..n]);
                if tail.len() > MAX_OUTPUT_BYTES * 4 {
                    let cut = tail.len() - MAX_OUTPUT_BYTES * 2;
                    tail.drain(..cut);
                    dropped = true;
                }
            }
        }
        if dropped {
            head.extend_from_slice(b"\n... [output cut] ...\n");
        }
        head.extend_from_slice(&tail);
        head
    })
}
/// `**` crosses folders, `*` and `?` stay inside one; the pattern is matched against the path relative to the workspace.
fn glob_regex(pattern: &str) -> Result<Regex, String> {
    // Windows file names are not case sensitive
    let mut out = String::from(if cfg!(windows) { "(?i)^" } else { "^" });
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
pub(super) fn agent_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLLM_NO_AGENT").is_some_and(|v| !v.is_empty() && v != "0"))
}

pub(super) fn disabled_reply() -> Response {
    api_error(StatusCode::FORBIDDEN, "the coding agent is switched off on this server (FLLM_NO_AGENT)")
}

pub(super) fn result_json(result: ToolResult) -> Value {
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

/// `POST /api/studio/agent/workspace`: `{path}` chooses the folder the agent works in.
pub(in crate::server) async fn workspace(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    if let Some(path) = body.get("path").and_then(Value::as_str) {
        let before = state.agent.current_root();
        if let Err(error) = state.agent.set_root(path) {
            return api_error(StatusCode::BAD_REQUEST, &error);
        }
        let after = state.agent.current_root();
        if before != after {
            let started = super::hooks::dispatch("on_session_start", "", &json!({}), after.as_deref());
            let mut answer = workspace_json(&state);
            if let Some(object) = answer.as_object_mut() {
                object.insert("hooks".to_owned(), json!({"context": started.notes(), "blocked": started.blocked}));
            }
            return Json(answer).into_response();
        }
    }
    Json(workspace_json(&state)).into_response()
}

/// `POST /api/studio/agent/tool`: `{name, args}` runs one tool and returns `{ok, output}` or `{ok: false, error}`.
/// Hooks of the user run around it: `pre_tool_call` may block the call, `post_tool_call` and `on_file_edit` add notes.
pub(in crate::server) async fn tool(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let Some(name) = body.get("name").and_then(Value::as_str).map(str::to_owned) else {
        return api_error(StatusCode::BAD_REQUEST, "the tool name is missing");
    };
    let args = body.get("args").cloned().unwrap_or_else(|| json!({}));
    let dir = state.agent.current_root();
    let pre = super::hooks::dispatch("pre_tool_call", &name, &args, dir.as_deref());
    if pre.blocked {
        return Json(result_json(Err(format!("a hook blocked this call: {}", pre.reason)))).into_response();
    }
    // computer use (screen and input of this machine) has its own switch, rate limit and stop; the scheduled tasks never reach it
    if super::agent_computer::is_computer_tool(&name) {
        let result = tokio::task::spawn_blocking(move || super::agent_computer::run(&name, &args)).await;
        return match result {
            Ok(result) => Json(super::agent_computer::outcome_json(result)).into_response(),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
    }
    // the network tools are async and need no workspace
    if let Some(result) = super::agent_net::run_tool(&name, &args).await {
        return Json(result_json(super::hooks::with_notes(result, &pre))).into_response();
    }
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || worker.agent.run_tool(&name, &args)).await {
        Ok(result) => Json(result_json(super::hooks::with_notes(result, &pre))).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `GET /api/studio/agent/tools`: every server-side tool as an OpenAI function schema plus its `permission`.
pub(in crate::server) async fn tools() -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let mut list = super::agent_tools::catalog();
    // the screen and input tools are only offered while the user has switched computer use on and this machine can do it
    if super::agent_computer::offered() {
        list.extend(super::agent_computer::catalog());
    }
    Json(json!({"tools": list})).into_response()
}

/// `GET /api/studio/agent/changes`: the file changes of this session, oldest first.
pub(in crate::server) async fn changes(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    Json(json!({"changes": state.agent.changes()})).into_response()
}

/// `POST /api/studio/agent/end`: ends the session's background jobs.
pub(in crate::server) async fn end(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let root = state.agent.current_root();
    let finished = super::hooks::dispatch("on_session_end", "", &json!({}), root.as_deref());
    let worker = state.clone();
    let stopped = tokio::task::spawn_blocking(move || worker.agent.end_session()).await.unwrap_or(0);
    Json(json!({"ok": true, "stopped": stopped, "hooks": {"context": finished.notes()}})).into_response()
}

/// `POST /api/studio/agent/cancel`: ends the foreground commands that run right now (the Stop button).
pub(in crate::server) async fn cancel(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    Json(json!({"ok": true, "stopped": state.agent.cancel_foreground()})).into_response()
}

/// `POST /api/studio/agent/files`: `{query, limit}` lists workspace files and folders that match the query
/// (fuzzy, `.gitignore` respected), best first. It is a POST so that it stays on this machine like the other
/// routes that name workspace files.
pub(in crate::server) async fn files(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let query = body.get("query").and_then(Value::as_str).unwrap_or_default().to_owned();
    let limit = body.get("limit").and_then(Value::as_u64).map_or(20, |n| n as usize);
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || worker.agent.find_files(&query, limit)).await {
        Ok(Ok(files)) => Json(json!({"files": files})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, &e),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `POST /api/studio/agent/undo`: reverts the latest file change of the agent (`{}` or no body), the last
/// `{steps: n}` changes, or everything back to `{to: id}` (that change included). Answers
/// `{ok, output, restored: [{id, path, action}]}`; `restored` is newest first.
pub(in crate::server) async fn undo(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let body: Value = if body.iter().all(u8::is_ascii_whitespace) {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &format!("invalid JSON: {e}")),
        }
    };
    let steps = body.get("steps").and_then(Value::as_u64);
    let to = body.get("to").and_then(Value::as_u64);
    let outcome = match body.get("redo").and_then(Value::as_u64) {
        Some(count) => state.agent.redo_many(count),
        None => state.agent.undo_many(steps, to),
    };
    match outcome {
        Ok(restored) => {
            let output = restored
                .iter()
                .map(|r| format!("{} {}", if r.action == "removed" { "Removed" } else { "Restored" }, r.path))
                .collect::<Vec<_>>()
                .join("\n");
            let list: Vec<Value> = restored.iter().map(|r| json!({"id": r.id, "path": r.path, "action": r.action})).collect();
            Json(json!({"ok": true, "output": output, "restored": list})).into_response()
        }
        Err(error) => Json(json!({"ok": false, "error": error, "restored": []})).into_response(),
    }
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
    fn multi_edit_is_all_or_nothing_and_one_undo_step() {
        let (s, root) = session();
        s.run_tool("write_file", &json!({"path": "m.txt", "content": "alpha beta beta gamma\n"})).unwrap();
        let ok = s
            .run_tool(
                "multi_edit",
                &json!({"path": "m.txt", "edits": [
                    {"old_string": "alpha", "new_string": "ALPHA"},
                    {"old_string": "beta", "new_string": "B", "replace_all": true},
                    {"old_string": "ALPHA B", "new_string": "first"},
                ]}),
            )
            .unwrap();
        assert!(ok.contains("3 edits, 4 replacements"), "{ok}");
        assert_eq!(fs::read_to_string(root.join("m.txt")).unwrap(), "first B gamma\n");
        // the second edit fails (ambiguous): the file stays as it was
        let bad = s
            .run_tool("multi_edit", &json!({"path": "m.txt", "edits": [
                {"old_string": "first", "new_string": "1st"},
                {"old_string": "zzz", "new_string": "y"},
            ]}))
            .unwrap_err();
        assert!(bad.contains("edit 2 of 2") && bad.contains("nothing was changed"), "{bad}");
        assert_eq!(fs::read_to_string(root.join("m.txt")).unwrap(), "first B gamma\n");
        assert!(s.run_tool("multi_edit", &json!({"path": "m.txt", "edits": []})).is_err());
        assert!(s.run_tool("multi_edit", &json!({"path": "m.txt"})).is_err());
        // one undo step brings back the text from before all three edits
        assert_eq!(s.changes().len(), 2);
        s.undo().unwrap();
        assert_eq!(fs::read_to_string(root.join("m.txt")).unwrap(), "alpha beta beta gamma\n");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn changes_are_listed_and_undone_by_steps_or_up_to_an_id() {
        let (s, root) = session();
        assert!(s.undo_many(None, None).is_err());
        s.run_tool("write_file", &json!({"path": "a.txt", "content": "1"})).unwrap();
        s.run_tool("write_file", &json!({"path": "a.txt", "content": "2"})).unwrap();
        s.run_tool("edit_file", &json!({"path": "a.txt", "old_string": "2", "new_string": "3"})).unwrap();
        s.run_tool("write_file", &json!({"path": "sub/b.txt", "content": "b"})).unwrap();
        let list = s.changes();
        let kinds: Vec<_> = list.iter().map(|c| c["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["create", "edit", "edit", "create"]);
        assert_eq!(list[3]["path"], "sub/b.txt");
        assert!(list[0]["time"].as_u64().unwrap() > 0);
        let ids: Vec<u64> = list.iter().map(|c| c["id"].as_u64().unwrap()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert!(s.undo_many(Some(1), Some(ids[0])).is_err(), "steps and to together are refused");
        assert!(s.undo_many(Some(0), None).is_err());
        assert!(s.undo_many(None, Some(9999)).is_err());
        // two steps: the creation of b and the edit "2" -> "3"
        let two = s.undo_many(Some(2), None).unwrap();
        assert_eq!(two.iter().map(|r| (r.path.as_str(), r.action)).collect::<Vec<_>>(), [("sub/b.txt", "removed"), ("a.txt", "restored")]);
        assert!(!root.join("sub/b.txt").exists());
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "2");
        assert_eq!(s.changes().len(), 2);
        // back to just before the second change
        let rest = s.undo_many(None, Some(ids[1])).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "1");
        // more steps than changes undo everything
        assert_eq!(s.undo_many(Some(50), None).unwrap().len(), 1);
        assert!(!root.join("a.txt").exists());
        assert!(s.changes().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn grep_modes_context_case_and_limits() {
        let (s, root) = session();
        s.run_tool("write_file", &json!({"path": "one.txt", "content": "a\nNeedle 1\nb\nc\nd\ne\nneedle 2\nf\n"})).unwrap();
        s.run_tool("write_file", &json!({"path": "two.txt", "content": "needle 3\nneedle 4\n"})).unwrap();
        let grep = |extra: Value| {
            let mut args = json!({"pattern": "needle"});
            args.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            s.run_tool("grep", &args)
        };
        let sensitive = grep(json!({})).unwrap();
        assert!(sensitive.contains("one.txt:7: needle 2") && !sensitive.contains("Needle 1"), "{sensitive}");
        let folded = grep(json!({"case_insensitive": true, "path": "one.txt"})).unwrap();
        assert!(folded.contains("one.txt:2: Needle 1") && folded.contains("one.txt:7:"), "{folded}");
        let files = grep(json!({"output_mode": "files"})).unwrap();
        assert_eq!(files, "one.txt\ntwo.txt\n");
        let count = grep(json!({"output_mode": "count", "case_insensitive": true})).unwrap();
        assert!(count.contains("one.txt:2\n") && count.contains("two.txt:2\n") && count.contains("total: 4 matching lines in 2 files"), "{count}");
        let limited = grep(json!({"head_limit": 1})).unwrap();
        assert!(limited.contains("one.txt:7:") && limited.contains("2 more matching lines"), "{limited}");
        let one_file = grep(json!({"output_mode": "files", "head_limit": 1})).unwrap();
        assert!(one_file.contains("one.txt") && one_file.contains("1 more files"), "{one_file}");
        // context: neighbours with `-`, a gap marker between distant blocks and files
        let ctx = grep(json!({"case_insensitive": true, "context": 1, "path": "one.txt"})).unwrap();
        assert_eq!(ctx, "one.txt-1- a\none.txt:2: Needle 1\none.txt-3- b\n--\none.txt-6- e\none.txt:7: needle 2\none.txt-8- f\n");
        let merged = grep(json!({"context": 2, "path": "two.txt"})).unwrap();
        assert_eq!(merged, "two.txt:1: needle 3\ntwo.txt:2: needle 4\n");
        assert!(grep(json!({"output_mode": "bogus"})).is_err());
        assert_eq!(grep(json!({"pattern": "zzz", "output_mode": "count"})).unwrap(), "no matches");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn notebooks_are_read_as_text_cells() {
        let (s, root) = session();
        let notebook = json!({"cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "text"]},
            {"cell_type": "code", "source": "print(1)\nx = 2", "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
                {"output_type": "execute_result", "data": {"text/plain": "2", "image/png": "AAAA"}},
                {"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": []},
            ]},
        ], "metadata": {}, "nbformat": 4});
        fs::write(root.join("n.ipynb"), notebook.to_string()).unwrap();
        let read = s.run_tool("read_file", &json!({"path": "n.ipynb"})).unwrap();
        assert!(read.contains("\t# [cell 1 | markdown]") && read.contains("\t# Title\n") && read.contains("\ttext\n"), "{read}");
        assert!(read.contains("\t# [cell 2 | code]") && read.contains("\tprint(1)\n") && read.contains("\tx = 2\n"), "{read}");
        assert_eq!(read.matches("\t# output:\n").count(), 3, "{read}");
        assert!(read.contains("\t1\n") && read.contains("\t2\n") && read.contains("\tValueError: bad\n"), "{read}");
        assert!(!read.contains("AAAA"), "images are left out: {read}");
        fs::write(root.join("broken.ipynb"), "{not json").unwrap();
        assert!(s.run_tool("read_file", &json!({"path": "broken.ipynb"})).unwrap_err().contains("not a readable notebook"));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn the_shell_keeps_its_directory_inside_the_workspace_and_runs_background_jobs() {
        let (s, root) = session();
        fs::create_dir_all(root.join("src/deep")).unwrap();
        let moved = s.run_tool("bash", &json!({"command": "cd src/deep && echo in-deep"})).unwrap();
        assert!(moved.starts_with("in-deep\n") && moved.contains("(working directory is now src/deep)"), "{moved}");
        let pwd = s.run_tool("bash", &json!({"command": "pwd"})).unwrap();
        assert!(pwd.trim_start().starts_with(root.join("src/deep").to_str().unwrap()), "{pwd}");
        // the exit code survives the wrapper, and a `cd` out of the workspace is ignored
        assert!(s.run_tool("bash", &json!({"command": "cd .. && exit 3"})).is_err());
        let up = s.run_tool("bash", &json!({"command": "cd /"})).unwrap();
        assert!(!up.contains("working directory is now"), "{up}");
        let still = s.run_tool("bash", &json!({"command": "pwd"})).unwrap();
        assert!(!still.trim().is_empty() && PathBuf::from(still.trim()).starts_with(&root), "{still}");
        // file tools still use paths relative to the workspace root
        s.run_tool("write_file", &json!({"path": "r.txt", "content": "x"})).unwrap();
        assert!(root.join("r.txt").exists());
        // background jobs start in the shell directory and are stopped with the session
        let started = s.run_tool("bash", &json!({"command": "pwd; sleep 30", "background": true})).unwrap();
        assert!(started.contains("job1"), "{started}");
        let mut seen = String::new();
        for _ in 0..100 {
            seen = s.run_tool("bash_output", &json!({"job_id": "job1"})).unwrap();
            if seen.contains(root.to_str().unwrap()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(seen.contains("status: running") && seen.contains(root.to_str().unwrap()), "{seen}");
        assert_eq!(s.end_session(), 1);
        assert!(s.run_tool("bash_output", &json!({"job_id": "job1"})).is_err());
        // choosing another workspace also ends the jobs
        s.run_tool("bash", &json!({"command": "sleep 30", "background": true})).unwrap();
        let other = std::env::temp_dir().join(format!("fllm-agent-other-{}", std::process::id()));
        fs::create_dir_all(&other).unwrap();
        s.set_root(other.to_str().unwrap()).unwrap();
        assert!(s.run_tool("bash_output", &json!({"job_id": "job2"})).is_err());
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(other);
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

#[cfg(test)]
mod tests_more {
    use super::*;

    fn session() -> (AgentSession, PathBuf) {
        let dir = std::env::temp_dir().join(format!("fllm-agent2-{}-{}", std::process::id(), super::super::new_id("t")));
        fs::create_dir_all(&dir).unwrap();
        let session = AgentSession::default();
        let root = session.set_root(dir.to_str().unwrap()).unwrap();
        (session, root)
    }

    #[test]
    fn edit_keeps_crlf_and_explains_misses() {
        let (s, root) = session();
        fs::write(root.join("a.txt"), "one\r\ntwo\r\nthree\r\n").unwrap();
        s.run_tool("edit_file", &json!({"path": "a.txt", "old_string": "two\nthree", "new_string": "2\n3"})).unwrap();
        assert_eq!(fs::read(root.join("a.txt")).unwrap(), b"one\r\n2\r\n3\r\n");
        let err = s.run_tool("edit_file", &json!({"path": "a.txt", "old_string": "nothing like it here at all", "new_string": "x"})).unwrap_err();
        assert!(err.contains("not found"), "{err}");
        // a failed edit leaves no phantom entry in the change log
        assert_eq!(s.changes().len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn patch_changes_several_files_as_one_unit_and_can_be_undone() {
        let (s, root) = session();
        fs::write(root.join("a.txt"), "one\ntwo\nthree\n").unwrap();
        fs::write(root.join("b.txt"), "x\ny\n").unwrap();
        let patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n--- a/b.txt\n+++ b/b.txt\n@@ -1,2 +1,2 @@\n x\n-y\n+Y\n--- /dev/null\n+++ b/c.txt\n@@ -0,0 +1 @@\n+new\n";
        let out = s.run_tool("patch", &json!({"patch": patch})).unwrap();
        assert!(out.contains("Patched a.txt") && out.contains("Created c.txt"), "{out}");
        assert_eq!(fs::read_to_string(root.join("b.txt")).unwrap(), "x\nY\n");
        // one broken hunk changes nothing
        let bad = "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-missing line\n+x\n";
        assert!(s.run_tool("patch", &json!({"patch": bad})).unwrap_err().contains("does not apply"));
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "one\nTWO\nthree\n");
        assert_eq!(s.changes().len(), 3);
        s.undo_many(Some(3), None).unwrap();
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "one\ntwo\nthree\n");
        assert!(!root.join("c.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn undone_changes_can_be_redone_until_a_new_change_happens() {
        let (s, root) = session();
        s.run_tool("write_file", &json!({"path": "a.txt", "content": "1"})).unwrap();
        s.run_tool("edit_file", &json!({"path": "a.txt", "old_string": "1", "new_string": "2"})).unwrap();
        s.run_tool("write_file", &json!({"path": "b.txt", "content": "b"})).unwrap();
        s.undo_many(None, Some(2)).unwrap();
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "1");
        assert!(!root.join("b.txt").exists());
        let again = s.redo_many(2).unwrap();
        assert_eq!(again.len(), 2);
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "2");
        assert_eq!(fs::read_to_string(root.join("b.txt")).unwrap(), "b");
        assert_eq!(s.changes().len(), 3);
        // the log is whole again: everything can be undone once more
        s.undo_many(Some(3), None).unwrap();
        assert!(!root.join("a.txt").exists());
        s.run_tool("write_file", &json!({"path": "c.txt", "content": "c"})).unwrap();
        assert!(s.redo_many(1).is_err(), "a new change clears the redo stack");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn walks_respect_gitignore_and_the_tree_shows_depth() {
        let (s, root) = session();
        fs::create_dir_all(root.join("src/deep")).unwrap();
        fs::create_dir_all(root.join("out")).unwrap();
        fs::write(root.join(".gitignore"), "out/\n*.log\n").unwrap();
        fs::write(root.join("src/a.rs"), "fn needle() {}\n").unwrap();
        fs::write(root.join("src/deep/b.rs"), "fn other() {}\n").unwrap();
        fs::write(root.join("out/c.rs"), "fn needle() {}\n").unwrap();
        fs::write(root.join("x.log"), "needle\n").unwrap();
        let found = s.run_tool("grep", &json!({"pattern": "needle", "output_mode": "files"})).unwrap();
        assert!(found.contains("src/a.rs") && !found.contains("out/c.rs") && !found.contains("x.log"), "{found}");
        assert!(!s.run_tool("glob", &json!({"pattern": "**/*.rs"})).unwrap().contains("out/"));
        let tree = s.run_tool("list_dir", &json!({"depth": 3})).unwrap();
        assert!(tree.contains("    b.rs") && tree.contains("out/  (ignored, not opened)"), "{tree}");
        let flat = s.run_tool("list_dir", &json!({})).unwrap();
        assert!(!flat.contains("b.rs"), "{flat}");
        let literal = s.run_tool("grep", &json!({"pattern": "fn needle()", "literal": true, "output_mode": "count"})).unwrap();
        assert!(literal.contains("src/a.rs:1"), "{literal}");
        let files = s.find_files("deep b", 5).unwrap();
        assert_eq!(files[0], "src/deep/b.rs");
        assert!(s.find_files("src", 10).unwrap().contains(&"src/".to_owned()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_files_suggest_names_and_binary_files_are_refused() {
        let (s, root) = session();
        fs::write(root.join("config.toml"), "a = 1\n").unwrap();
        fs::write(root.join("logo.png"), [0u8, 1, 2]).unwrap();
        let err = s.run_tool("read_file", &json!({"path": "Config.tml"})).unwrap_err();
        assert!(err.contains("does not exist") && err.contains("config.toml"), "{err}");
        assert!(s.run_tool("read_file", &json!({"path": "logo.png"})).unwrap_err().contains("an image"));
        let ok = s.run_tool("read_file", &json!({"path": "config.toml"})).unwrap();
        assert!(ok.contains("(end of file, 1 lines)"), "{ok}");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn windows_style_paths_and_long_output() {
        // drive letters and folders are not case sensitive on Windows
        let root = Path::new("C:/Users/me/work");
        let p = Path::new("c:/users/ME/work/src/a.rs");
        let same = rebase_under(root, p, true);
        assert!(same.is_some() && !(cfg!(windows) && same.as_deref() != Some(Path::new("C:/Users/me/work/src/a.rs"))));
        assert!(rebase_under(Path::new("/w"), Path::new("/other/x"), true).is_none());
        assert!(rebase_under(Path::new("/w"), Path::new("/W/x"), false).is_none());
        assert_eq!(rebase_under(Path::new("/w"), Path::new("/W/x/y"), true).unwrap(), Path::new("/w/x/y"));
        let text = format!("{}END", "x".repeat(100_000));
        let clipped = clip_middle(&text, 1000);
        assert!(clipped.len() < 1100 && clipped.ends_with("END") && clipped.contains("cut from the middle"));
    }

    #[test]
    fn a_timeout_ends_the_whole_process_tree_and_stderr_keeps_its_place() {
        let (s, root) = session();
        let started = Instant::now();
        let err = s.run_tool("bash", &json!({"command": "sleep 30 & sleep 30", "timeout_ms": 1000})).unwrap_err();
        assert!(err.contains("longer than"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(15), "the output pipes stayed open");
        let out = s.run_tool("bash", &json!({"command": "echo out; echo err 1>&2; echo last"})).unwrap();
        let (a, b, c) = (out.find("out"), out.find("err"), out.find("last"));
        assert!(a < b && b < c, "{out}");
        let _ = fs::remove_dir_all(root);
    }
}
