//! Custom sub-agents of the coding agent: Markdown files in `agents/` next to `studio.json` (scope "user") and in
//! `.fllm/agents/` of a workspace folder (scope "workspace").
//!
//! A file is `<name>.md` with a small front matter and the instructions as the body:
//!
//! ```markdown
//! ---
//! name: reviewer
//! description: Reviews a diff for bugs
//! tools: read_file, grep, glob, list_dir
//! model: qwen3.5:9b
//! max_steps: 12
//! ---
//! You review changes. Read the diff, name the file and line, and explain the bug you found.
//! ```
//!
//! `tools` limits what the sub-agent may call (empty = the parent's tools), `model` runs that sub-agent on another
//! model (empty = the model of the session) and `max_steps` caps its loop. The page offers the names of these agents
//! in the `task` tool, so the model delegates by name instead of guessing.
use super::*;
use axum::extract::Query;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_AGENT_BYTES: usize = 32 * 1024;
const MAX_AGENTS: usize = 200;
const MAX_STEPS: u64 = 60;

/// A sub-agent as the page sees it.
fn to_json(name: &str, scope: &str, path: &Path, meta: &std::collections::BTreeMap<String, String>, body: &str) -> Value {
    let tools: Vec<String> = meta
        .get("tools")
        .map(|list| list.split(',').map(str::trim).filter(|t| !t.is_empty()).map(str::to_owned).collect())
        .unwrap_or_default();
    json!({
        "name": name,
        "description": meta.get("description").cloned().unwrap_or_default(),
        "tools": tools,
        "model": meta.get("model").cloned().unwrap_or_default(),
        "max_steps": meta.get("max_steps").and_then(|s| s.parse::<u64>().ok()).map(|n| n.clamp(1, MAX_STEPS)).unwrap_or(0),
        "scope": scope,
        "path": path.display().to_string(),
        "body": body.trim().to_owned(),
    })
}

fn write_text(path: &Path, name: &str, description: &str, tools: &[String], model: &str, max_steps: u64, body: &str) -> Result<(), String> {
    let text = format!(
        "---\nname: {name}\ndescription: {}\ntools: {}\nmodel: {}\nmax_steps: {}\n---\n{}\n",
        description.replace(['\n', '\r'], " ").trim(),
        tools.join(", "),
        model.trim(),
        if max_steps == 0 { String::new() } else { max_steps.to_string() },
        body.trim_end(),
    );
    if text.len() > MAX_AGENT_BYTES {
        return Err(format!("an agent file is limited to {} KB", MAX_AGENT_BYTES / 1024));
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Every agent of one folder, by file name.
fn list_in(base: &Path, scope: &str) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(base) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name().to_string_lossy().into_owned();
        let Some(stem) = file_name.strip_suffix(".md") else { continue };
        if !super::skills::valid_name(stem) {
            continue;
        }
        let path = entry.path();
        // a link is never followed: an agent file must live inside the agents folder
        if !fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_symlink()) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else { continue };
        if text.len() > MAX_AGENT_BYTES {
            continue;
        }
        let (meta, body) = super::skills::front_matter(&text);
        out.push(to_json(stem, scope, &path, &meta, body));
        if out.len() >= MAX_AGENTS {
            break;
        }
    }
    out.sort_by(|a, b| a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or("")));
    out
}

fn base_for(scope: &str, workspace: Option<&str>) -> Result<PathBuf, String> {
    match scope {
        "user" => Ok(crate::hub::studio_data_file("agents")),
        "workspace" => {
            let Some(workspace) = workspace else { return Err("the workspace folder is missing".to_owned()) };
            let path = Path::new(workspace);
            if !path.is_absolute() || !path.is_dir() {
                return Err("the workspace must be an existing absolute folder".to_owned());
            }
            Ok(path.join(".fllm").join("agents"))
        }
        other if other.starts_with("plugin:") => Err("the agents of a plugin are read-only: copy the file into your own agents/ folder to change it".to_owned()),
        other => Err(format!("scope must be 'user' or 'workspace', not '{other}'")),
    }
}

pub(super) fn list_all(workspace: Option<&str>) -> Value {
    let mut agents = list_in(&crate::hub::studio_data_file("agents"), "user");
    if let Ok(base) = base_for("workspace", workspace) {
        agents.extend(list_in(&base, "workspace"));
    }
    for (plugin, base) in super::plugins::agent_bases() {
        let scope = format!("plugin:{plugin}");
        for mut agent in list_in(&base, &scope) {
            agent["read_only"] = json!(true);
            agents.push(agent);
        }
    }
    json!({
        "agents": agents,
        "folders": {
            "user": crate::hub::studio_data_file("agents").display().to_string(),
            "workspace": workspace.map(|w| Path::new(w).join(".fllm").join("agents").display().to_string()),
        },
    })
}

/// Writes one agent file. The same name in the workspace scope wins over the user scope when the model delegates.
pub(super) fn save(body: &Value) -> Result<Value, String> {
    let name = body.get("name").and_then(Value::as_str).unwrap_or("").trim();
    if !super::skills::valid_name(name) {
        return Err("an agent name has 1 to 64 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    let scope = body.get("scope").and_then(Value::as_str).unwrap_or("user");
    let workspace = body.get("workspace").and_then(Value::as_str);
    let base = base_for(scope, workspace)?;
    let description = body.get("description").and_then(Value::as_str).unwrap_or("").trim();
    let model = body.get("model").and_then(Value::as_str).unwrap_or("").trim();
    if model.chars().count() > 120 {
        return Err("the model name is longer than 120 characters".to_owned());
    }
    let tools: Vec<String> = match body.get("tools") {
        Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_owned).collect(),
        Some(Value::String(list)) => list.split(',').map(str::trim).filter(|t| !t.is_empty()).map(str::to_owned).collect(),
        _ => Vec::new(),
    };
    let max_steps = body.get("max_steps").and_then(Value::as_u64).unwrap_or(0).min(MAX_STEPS);
    let instructions = body.get("body").and_then(Value::as_str).unwrap_or("").trim();
    if instructions.is_empty() {
        return Err("the instructions of an agent must not be empty".to_owned());
    }
    write_text(&base.join(format!("{name}.md")), name, description, &tools, model, max_steps, instructions)?;
    Ok(list_all(workspace))
}

pub(super) fn remove(scope: &str, name: &str, workspace: Option<&str>) -> Result<Value, String> {
    if !super::skills::valid_name(name) {
        return Err("an agent name has 1 to 64 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    let base = base_for(scope, workspace)?;
    let path = base.join(format!("{name}.md"));
    if !path.exists() {
        return Err(format!("there is no agent '{name}' in this folder"));
    }
    fs::remove_file(&path).map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
    Ok(list_all(workspace))
}

/// `GET /api/studio/agents?workspace=<folder>`: the agents of both folders.
pub(in crate::server) async fn list(Query(query): Query<HashMap<String, String>>) -> Response {
    Json(list_all(query.get("workspace").map(String::as_str))).into_response()
}

/// `POST /api/studio/agents`: `{name, description, tools, model, max_steps, body, scope, workspace}` writes one file.
pub(in crate::server) async fn write(Json(body): Json<Value>) -> Response {
    match save(&body) {
        Ok(saved) => Json(saved).into_response(),
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}

/// `DELETE /api/studio/agents?name=&scope=&workspace=`: removes one agent file.
pub(in crate::server) async fn remove_route(Query(query): Query<HashMap<String, String>>) -> Response {
    let name = query.get("name").map(String::as_str).unwrap_or("");
    let scope = query.get("scope").map(String::as_str).unwrap_or("user");
    match remove(scope, name, query.get("workspace").map(String::as_str)) {
        Ok(saved) => Json(saved).into_response(),
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-agents-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_written_agent_is_read_back_with_its_fields() {
        let dir = temp("roundtrip");
        let path = dir.join("reviewer.md");
        write_text(&path, "reviewer", "Reviews a diff", &["read_file".to_owned(), "grep".to_owned()], "qwen3.5:9b", 12, "You review.").unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let (meta, body) = super::super::skills::front_matter(&text);
        assert_eq!(meta["description"], "Reviews a diff");
        assert_eq!(meta["tools"], "read_file, grep");
        assert_eq!(meta["model"], "qwen3.5:9b");
        assert_eq!(meta["max_steps"], "12");
        assert!(body.contains("You review."));
        let listed = list_in(&dir, "user");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["name"], "reviewer");
        assert_eq!(listed[0]["tools"][0], "read_file");
        assert_eq!(listed[0]["max_steps"], 12);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_with_an_invalid_name_is_ignored() {
        let dir = temp("names");
        fs::write(dir.join("Bad Name.md"), "---\nname: x\n---\nbody").unwrap();
        fs::write(dir.join("no-extension.txt"), "---\nname: y\n---\nbody").unwrap();
        assert!(list_in(&dir, "user").is_empty(), "only <slug>.md files are agents");
        fs::write(dir.join("good.md"), "---\ndescription: d\n---\nbody").unwrap();
        let listed = list_in(&dir, "user");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["name"], "good");
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_linked_agent_file_is_not_followed() {
        let dir = temp("links");
        let outside = temp("links-outside");
        fs::write(outside.join("secret.md"), "---\ndescription: leaked\n---\nbody").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.md"), dir.join("linked.md")).unwrap();
        assert!(list_in(&dir, "user").is_empty());
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside);
    }

    #[test]
    fn the_workspace_scope_needs_an_existing_absolute_folder() {
        assert!(base_for("workspace", None).is_err());
        assert!(base_for("workspace", Some("relative/path")).is_err());
        assert!(base_for("elsewhere", None).is_err());
        let dir = temp("scope");
        let base = base_for("workspace", Some(dir.to_str().unwrap())).unwrap();
        assert!(base.ends_with(Path::new(".fllm").join("agents")));
        let _ = fs::remove_dir_all(&dir);
    }
}
