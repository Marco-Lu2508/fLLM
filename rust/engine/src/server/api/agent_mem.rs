//! Personal context of the coding agent: a memory file for the workspace (`.fllm/memory.md`) and one for the
//! user (in the Studio folder), the `memory` tool that edits them, and the project instruction files
//! (`AGENTS.md`, `FLLM.md`, `CLAUDE.md`) of the workspace and its parent folders.
//!
//! A memory file is a list of short entries, one `- text` line each; people may edit the file by hand.
use super::agent::{agent_disabled, disabled_reply, AgentSession};
use super::*;
use std::{fs, path::{Path, PathBuf}};

const MAX_ENTRY_CHARS: usize = 400;
const MAX_ENTRIES: usize = 100;
const MAX_FILE_BYTES: usize = 24_000;
const INSTRUCTION_FILES: [&str; 3] = ["AGENTS.md", "FLLM.md", "CLAUDE.md"];
const MAX_INSTRUCTION_CHARS: usize = 6_000;
const MAX_INSTRUCTIONS_TOTAL: usize = 14_000;
const MAX_PARENTS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Scope {
    Workspace,
    Global,
}

impl Scope {
    fn parse(text: Option<&str>) -> Result<Scope, String> {
        match text.unwrap_or("workspace") {
            "workspace" | "project" => Ok(Scope::Workspace),
            "global" | "user" => Ok(Scope::Global),
            other => Err(format!("scope must be 'workspace' or 'global', not '{other}'")),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Scope::Workspace => "workspace",
            Scope::Global => "global",
        }
    }
}

fn file_for(scope: Scope, root: Option<&Path>) -> Result<PathBuf, String> {
    match scope {
        Scope::Global => Ok(crate::hub::studio_data_file("agent-memory.md")),
        Scope::Workspace => root.map(|r| r.join(".fllm").join("memory.md")).ok_or_else(|| "no workspace folder is chosen yet".to_owned()),
    }
}

fn read(scope: Scope, root: Option<&Path>) -> String {
    file_for(scope, root).ok().and_then(|f| fs::read_to_string(f).ok()).unwrap_or_default()
}

fn write(scope: Scope, root: Option<&Path>, text: &str) -> Result<(), String> {
    if text.len() > MAX_FILE_BYTES {
        return Err(format!("the memory is limited to {} KB; remove entries first", MAX_FILE_BYTES / 1000));
    }
    let file = file_for(scope, root)?;
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    fs::write(&file, text).map_err(|e| format!("cannot write {}: {e}", file.display()))
}

/// The entry lines (`- text`) of a memory file, without the dash.
pub(super) fn entries(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.trim().strip_prefix("- ").map(|e| e.trim().to_owned())).filter(|e| !e.is_empty()).collect()
}

fn render(entries: &[String]) -> String {
    let mut out = String::from("# Memory\n\n");
    for e in entries {
        out.push_str(&format!("- {e}\n"));
    }
    out
}

fn one_line(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    flat.chars().take(MAX_ENTRY_CHARS).collect()
}

fn listing(scope: Scope, list: &[String]) -> String {
    if list.is_empty() {
        return format!("The {} memory is empty.", scope.name());
    }
    let mut out = format!("{} memory ({} entries):\n", scope.name(), list.len());
    for (i, e) in list.iter().enumerate() {
        out.push_str(&format!("{}. {e}\n", i + 1));
    }
    out
}

/// The `memory` tool: `action` is list, add, update or remove.
pub(super) fn tool(session: &AgentSession, args: &Value) -> Result<String, String> {
    let text = |key: &str| args.get(key).and_then(Value::as_str);
    let scope = Scope::parse(text("scope"))?;
    let root = session.current_root();
    if scope == Scope::Workspace && root.is_none() {
        return Err("no workspace folder is chosen yet; choose one first".into());
    }
    let mut list = entries(&read(scope, root.as_deref()));
    let id = args.get("id").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())));
    let index = |list: &[String]| -> Result<usize, String> {
        let n = id.ok_or("'id' is missing: give the number from the list")? as usize;
        if n == 0 || n > list.len() {
            return Err(format!("there is no entry {n}; the {} memory has {} entries", scope.name(), list.len()));
        }
        Ok(n - 1)
    };
    match text("action").unwrap_or("list") {
        "list" => Ok(listing(scope, &list)),
        "add" => {
            let entry = one_line(text("text").ok_or("'text' is missing")?);
            if entry.is_empty() {
                return Err("'text' is empty".into());
            }
            if list.iter().any(|e| e.eq_ignore_ascii_case(&entry)) {
                return Ok(format!("Already remembered.\n{}", listing(scope, &list)));
            }
            if list.len() >= MAX_ENTRIES {
                return Err(format!("the {} memory is full ({MAX_ENTRIES} entries); remove or merge entries first", scope.name()));
            }
            list.push(entry);
            write(scope, root.as_deref(), &render(&list))?;
            Ok(format!("Remembered.\n{}", listing(scope, &list)))
        }
        "update" => {
            let at = index(&list)?;
            let entry = one_line(text("text").ok_or("'text' is missing")?);
            if entry.is_empty() {
                return Err("'text' is empty; use action remove to delete an entry".into());
            }
            list[at] = entry;
            write(scope, root.as_deref(), &render(&list))?;
            Ok(format!("Updated.\n{}", listing(scope, &list)))
        }
        "remove" => {
            let at = index(&list)?;
            list.remove(at);
            write(scope, root.as_deref(), &render(&list))?;
            Ok(format!("Removed.\n{}", listing(scope, &list)))
        }
        other => Err(format!("action must be list, add, update or remove, not '{other}'")),
    }
}

/// The instruction files that apply to `root`: its own folder first, then the parents up to the repository
/// root (or a few levels when there is no repository). Each entry is `(path shown, text)`.
pub(super) fn instruction_files(root: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut total = 0usize;
    let mut dir = Some(root);
    let mut levels = 0usize;
    while let Some(d) = dir {
        for name in INSTRUCTION_FILES {
            let file = d.join(name);
            let Ok(raw) = fs::read_to_string(&file) else { continue };
            let text: String = raw.trim().chars().take(MAX_INSTRUCTION_CHARS).collect();
            if text.is_empty() || total + text.len() > MAX_INSTRUCTIONS_TOTAL {
                continue;
            }
            total += text.len();
            let shown = if d == root { name.to_owned() } else { file.display().to_string().replace('\\', "/") };
            found.push((shown, text));
            break; // one file per folder: they usually say the same
        }
        if d.join(".git").exists() || levels >= MAX_PARENTS {
            break;
        }
        levels += 1;
        dir = d.parent();
    }
    found
}

/// `POST /api/studio/agent/instructions`: the instruction files and both memories, for the system prompt.
pub(in crate::server) async fn instructions(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let root = state.agent.current_root();
    let files: Vec<Value> = match &root {
        Some(r) => instruction_files(r).into_iter().map(|(file, text)| json!({"file": file, "text": text})).collect(),
        None => Vec::new(),
    };
    let memory = |scope| entries(&read(scope, root.as_deref()));
    Json(json!({"files": files, "memory": {"workspace": memory(Scope::Workspace), "global": memory(Scope::Global)}})).into_response()
}

/// `POST /api/studio/agent/memory`: `{scope, op: get | set | append, text}` reads or writes one memory file as text.
pub(in crate::server) async fn memory(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let scope = match Scope::parse(body.get("scope").and_then(Value::as_str)) {
        Ok(s) => s,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    let root = state.agent.current_root();
    let text = body.get("text").and_then(Value::as_str).unwrap_or_default();
    let result = match body.get("op").and_then(Value::as_str).unwrap_or("get") {
        "get" => Ok(()),
        "set" => write(scope, root.as_deref(), text),
        "append" => {
            let mut list = entries(&read(scope, root.as_deref()));
            let entry = one_line(text);
            if !entry.is_empty() && !list.iter().any(|e| e.eq_ignore_ascii_case(&entry)) && list.len() < MAX_ENTRIES {
                list.push(entry);
            }
            write(scope, root.as_deref(), &render(&list))
        }
        other => Err(format!("op must be get, set or append, not '{other}'")),
    };
    match result {
        Ok(()) => {
            let file = file_for(scope, root.as_deref()).map(|f| f.display().to_string()).unwrap_or_default();
            Json(json!({"ok": true, "text": read(scope, root.as_deref()), "file": file})).into_response()
        }
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_bullet_lines() {
        assert_eq!(entries("# Memory\n\n- one\n- two  \nnot an entry\n-\n"), vec!["one", "two"]);
        assert_eq!(entries(&render(&["a".into(), "b".into()])), vec!["a", "b"]);
        assert_eq!(one_line("a\n  b\tc"), "a b c");
    }

    #[test]
    fn the_workspace_memory_tool_adds_updates_and_removes() {
        let dir = std::env::temp_dir().join(format!("fllm-mem-{}-{}", std::process::id(), super::super::new_id("t")));
        fs::create_dir_all(&dir).unwrap();
        let s = AgentSession::default();
        s.set_root(dir.to_str().unwrap()).unwrap();
        assert!(tool(&s, &json!({"action": "add", "text": "tests run with npm test"})).unwrap().contains("1. tests run"));
        assert!(tool(&s, &json!({"action": "add", "text": "TESTS RUN WITH NPM TEST"})).unwrap().contains("Already"));
        tool(&s, &json!({"action": "add", "text": "use tabs"})).unwrap();
        assert!(tool(&s, &json!({"action": "update", "id": 2, "text": "use two spaces"})).unwrap().contains("2. use two spaces"));
        assert!(tool(&s, &json!({"action": "remove", "id": 1})).unwrap().contains("1. use two spaces"));
        assert!(tool(&s, &json!({"action": "remove", "id": 9})).is_err());
        assert!(dir.join(".fllm").join("memory.md").exists());
        assert!(tool(&s, &json!({"scope": "nowhere"})).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn instruction_files_are_read_nearest_first_up_to_the_repository() {
        let base = std::env::temp_dir().join(format!("fllm-ins-{}-{}", std::process::id(), super::super::new_id("t")));
        let sub = base.join("pkg");
        fs::create_dir_all(base.join(".git")).unwrap();
        fs::create_dir_all(&sub).unwrap();
        fs::write(base.join("CLAUDE.md"), "root rules").unwrap();
        fs::write(sub.join("AGENTS.md"), "package rules").unwrap();
        fs::write(sub.join("FLLM.md"), "ignored: one file per folder").unwrap();
        let found = instruction_files(&sub);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], ("AGENTS.md".to_owned(), "package rules".to_owned()));
        assert!(found[1].0.ends_with("CLAUDE.md") && found[1].1 == "root rules");
        let _ = fs::remove_dir_all(base);
    }
}
