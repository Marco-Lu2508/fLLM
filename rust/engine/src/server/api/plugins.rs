//! Plugins: one folder that adds skills, sub-agents, commands and hooks at once.
//!
//! A plugin lives in `plugins/<name>/` next to `studio.json` and is described by `plugin.json`:
//!
//! ```json
//! {"name": "rust-review", "version": "1.0.0", "description": "Review helpers for Rust", "author": "you"}
//! ```
//!
//! Optional sub folders and files next to it: `skills/<skill>/SKILL.md`, `agents/<name>.md` (same front matter as your
//! own sub-agents), `commands/<name>.md` (your own slash commands) and `hooks.json` (the same shape as the Studio's own
//! hooks file). Nothing is copied into your configuration: the contributions are read from the plugin folder every time,
//! so removing the folder removes everything the plugin added. Plugin contributions are read-only and cannot shadow your
//! own: a skill of the same name from your folder wins, and a plugin cannot write into `hooks.json` or `agents/`.
use super::*;
use axum::extract::Query;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_PLUGINS: usize = 50;
const MAX_COPY_BYTES: u64 = 50 * 1024 * 1024;
const MAX_COPY_ENTRIES: usize = 5_000;

pub(super) fn dir() -> PathBuf {
    crate::hub::studio_data_file("plugins")
}

#[derive(Clone)]
pub(super) struct Plugin {
    pub name: String,
    pub path: PathBuf,
    pub meta: Value,
}

/// Every installed plugin (a folder with a `plugin.json`), sorted by name.
pub(super) fn installed() -> Vec<Plugin> {
    let Ok(entries) = fs::read_dir(dir()) else { return Vec::new() };
    let mut out: Vec<Plugin> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if !super::skills::valid_name(&name) || !path.is_dir() {
            continue;
        }
        // a link is not followed: a plugin must be a real folder inside plugins/
        if !fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_symlink()) {
            continue;
        }
        let Ok(text) = fs::read_to_string(path.join("plugin.json")) else { continue };
        let meta = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({}));
        out.push(Plugin { name, path, meta });
        if out.len() >= MAX_PLUGINS {
            break;
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn count_dir(path: &Path) -> usize {
    fs::read_dir(path).map(|entries| entries.flatten().count()).unwrap_or(0)
}

/// The plugin as the page shows it, with what it contributes.
fn describe(plugin: &Plugin) -> Value {
    let skills = count_dir(&plugin.path.join("skills"));
    let agents = count_dir(&plugin.path.join("agents"));
    let commands = count_dir(&plugin.path.join("commands"));
    let hooks = fs::read_to_string(plugin.path.join("hooks.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| {
            value.get("hooks").and_then(Value::as_object).map(|events| {
                events.iter().map(|(_, list)| list.as_array().map_or(0, Vec::len)).sum::<usize>()
            })
        })
        .unwrap_or(0);
    json!({
        "name": plugin.name,
        "version": plugin.meta.get("version").and_then(Value::as_str).unwrap_or(""),
        "description": plugin.meta.get("description").and_then(Value::as_str).unwrap_or(""),
        "author": plugin.meta.get("author").and_then(Value::as_str).unwrap_or(""),
        "path": plugin.path.display().to_string(),
        "skills": skills,
        "agents": agents,
        "commands": commands,
        "hooks": hooks,
    })
}

pub(super) fn list_json() -> Value {
    json!({
        "folder": dir().display().to_string(),
        "plugins": installed().iter().map(describe).collect::<Vec<_>>(),
    })
}

/// Copies a directory into `plugins/`, refusing links, a file that is too large and an absurd number of entries.
fn copy_dir(from: &Path, to: &Path, bytes: &mut u64, entries: &mut usize) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|e| format!("cannot create {}: {e}", to.display()))?;
    for entry in fs::read_dir(from).map_err(|e| format!("cannot read {}: {e}", from.display()))?.flatten() {
        let kind = match fs::symlink_metadata(entry.path()) {
            Ok(meta) => meta,
            Err(e) => return Err(format!("cannot read {}: {e}", entry.path().display())),
        };
        if kind.file_type().is_symlink() {
            return Err(format!("a plugin may not contain a link: {}", entry.path().display()));
        }
        *entries += 1;
        if *entries > MAX_COPY_ENTRIES {
            return Err(format!("a plugin is limited to {MAX_COPY_ENTRIES} files"));
        }
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_dir(&entry.path(), &target, bytes, entries)?;
        } else if kind.is_file() {
            *bytes += kind.len();
            if *bytes > MAX_COPY_BYTES {
                return Err(format!("a plugin is limited to {} MB", MAX_COPY_BYTES / (1024 * 1024)));
            }
            fs::copy(entry.path(), &target).map_err(|e| format!("cannot copy {}: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Installs a plugin from a local folder. The folder must hold a `plugin.json` with a `name`.
pub(super) fn install(source: &str) -> Result<Value, String> {
    let from = PathBuf::from(source.trim());
    if !from.is_absolute() || !from.is_dir() {
        return Err("give the absolute path of an existing folder".to_owned());
    }
    let text = fs::read_to_string(from.join("plugin.json")).map_err(|_| format!("{} holds no plugin.json", from.display()))?;
    let meta = serde_json::from_str::<Value>(&text).map_err(|e| format!("plugin.json is not valid JSON: {e}"))?;
    let name = meta.get("name").and_then(Value::as_str).unwrap_or_default().trim().to_owned();
    if !super::skills::valid_name(&name) {
        return Err("plugin.json needs a name of 1 to 64 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    let target = dir().join(&name);
    if target.exists() {
        return Err(format!("a plugin named '{name}' is installed already; remove it first"));
    }
    let staging = dir().join(format!(".{name}.incoming"));
    let _ = fs::remove_dir_all(&staging);
    let mut bytes = 0u64;
    let mut entries = 0usize;
    if let Err(error) = copy_dir(&from, &staging, &mut bytes, &mut entries) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    fs::rename(&staging, &target).map_err(|e| {
        let _ = fs::remove_dir_all(&staging);
        format!("cannot move the plugin into place: {e}")
    })?;
    Ok(list_json())
}

/// A plugin can also come from a repository: `git clone --depth 1` into a temporary folder, then the same install as a
/// local folder. Only transports that need no typing at a prompt are allowed, and the clone cannot ask for a password.
pub(super) fn install_from_git(url: &str, reference: &str) -> Result<Value, String> {
    let url = url.trim();
    let allowed = ["https://", "http://", "ssh://", "git@", "file://"];
    if !allowed.iter().any(|prefix| url.starts_with(prefix)) {
        return Err("the address has to start with https://, http://, ssh://, git@ or file://".to_owned());
    }
    if url.contains(char::is_whitespace) || url.contains('\u{0}') {
        return Err("the address holds a space or a control character".to_owned());
    }
    if reference.contains(char::is_whitespace) || reference.starts_with('-') {
        return Err("the branch name looks wrong".to_owned());
    }
    let temp = dir().join(format!(".clone-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp);
    fs::create_dir_all(&temp).map_err(|e| format!("cannot create {}: {e}", temp.display()))?;
    let checkout = temp.join("plugin");
    let mut command = std::process::Command::new("git");
    command.arg("clone").arg("--quiet").arg("--depth").arg("1");
    if !reference.trim().is_empty() {
        command.arg("--branch").arg(reference.trim());
    }
    command.arg(url).arg(&checkout).env("GIT_TERMINAL_PROMPT", "0").env("GIT_ASKPASS", "true");
    let output = command.output().map_err(|e| {
        let _ = fs::remove_dir_all(&temp);
        format!("cannot run git: {e}")
    })?;
    if !output.status.success() {
        let _ = fs::remove_dir_all(&temp);
        let detail: String = String::from_utf8_lossy(&output.stderr).trim().chars().take(300).collect();
        return Err(format!("git clone failed: {detail}"));
    }
    let result = install(&checkout.to_string_lossy());
    let _ = fs::remove_dir_all(&temp);
    result
}

pub(super) fn remove(name: &str) -> Result<Value, String> {
    if !super::skills::valid_name(name) {
        return Err("a plugin name has 1 to 64 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    let path = dir().join(name);
    if !path.is_dir() {
        return Err(format!("there is no plugin named '{name}'"));
    }
    if !fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_symlink()) {
        return Err("that folder is a link, not a plugin".to_owned());
    }
    fs::remove_dir_all(&path).map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
    Ok(list_json())
}

// ---- contributions ------------------------------------------------------------------------------------------

/// `(plugin, skills folder)` for every plugin that brings skills.
pub(super) fn skill_bases() -> Vec<(String, PathBuf)> {
    installed()
        .into_iter()
        .map(|p| (p.name.clone(), p.path.join("skills")))
        .filter(|(_, path)| path.is_dir())
        .collect()
}

/// `(plugin, agents folder)` for every plugin that brings sub-agents.
pub(super) fn agent_bases() -> Vec<(String, PathBuf)> {
    installed()
        .into_iter()
        .map(|p| (p.name.clone(), p.path.join("agents")))
        .filter(|(_, path)| path.is_dir())
        .collect()
}

/// `(plugin, hooks value)` for every plugin that brings hooks.
pub(super) fn hook_files() -> Vec<(String, Value)> {
    installed()
        .into_iter()
        .filter_map(|p| {
            let text = fs::read_to_string(p.path.join("hooks.json")).ok()?;
            let value = serde_json::from_str::<Value>(&text).ok()?;
            Some((p.name, value))
        })
        .collect()
}

/// The commands of every plugin, as `{name, description, plugin}`.
pub(super) fn commands() -> Vec<Value> {
    let mut out = Vec::new();
    for (plugin, base) in installed().into_iter().map(|p| (p.name.clone(), p.path.join("commands"))).filter(|(_, p)| p.is_dir()) {
        let Ok(entries) = fs::read_dir(&base) else { continue };
        for entry in entries.flatten() {
            let file = entry.file_name().to_string_lossy().into_owned();
            let Some(stem) = file.strip_suffix(".md") else { continue };
            if !super::skills::valid_name(stem) || !entry.path().is_file() {
                continue;
            }
            let description = fs::read_to_string(entry.path())
                .ok()
                .and_then(|text| {
                    let (meta, body) = super::skills::front_matter(&text);
                    meta.get("description").cloned().or_else(|| body.lines().map(str::trim).find(|l| !l.is_empty()).map(str::to_owned))
                })
                .unwrap_or_default()
                .chars()
                .take(160)
                .collect::<String>();
            out.push(json!({"name": stem, "description": description, "plugin": plugin}));
        }
    }
    out.sort_by(|a, b| a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or("")));
    out
}

/// The body of one plugin command.
pub(super) fn read_command(plugin: &str, name: &str) -> Result<String, String> {
    if !super::skills::valid_name(plugin) || !super::skills::valid_name(name) {
        return Err("a plugin and a command name are lowercase slugs".to_owned());
    }
    let path = dir().join(plugin).join("commands").join(format!("{name}.md"));
    let text = fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (_, body) = super::skills::front_matter(&text);
    Ok(body.trim().to_owned())
}

/// The folder of a plugin, for reading its skills and agents (`scope` = `plugin:<name>`).
pub(super) fn folder(name: &str) -> Option<PathBuf> {
    if !super::skills::valid_name(name) {
        return None;
    }
    let path = dir().join(name);
    path.is_dir().then_some(path)
}

// ---- routes --------------------------------------------------------------------------------------------------

/// `GET /api/studio/plugins`: the installed plugins and the folder they live in.
pub(in crate::server) async fn list() -> Response {
    Json(list_json()).into_response()
}

/// `POST /api/studio/plugins` `{path}`: installs (copies) a plugin from a local folder.
pub(in crate::server) async fn install_route(Json(body): Json<Value>) -> Response {
    let path = body.get("path").and_then(Value::as_str).unwrap_or("").to_owned();
    let url = body.get("url").and_then(Value::as_str).unwrap_or("").to_owned();
    let reference = body.get("ref").and_then(Value::as_str).unwrap_or("").to_owned();
    // cloning is slow and blocking: it must not stop the server
    let result = tokio::task::spawn_blocking(move || {
        if url.trim().is_empty() {
            install(&path)
        } else {
            install_from_git(&url, &reference)
        }
    })
    .await;
    match result {
        Ok(Ok(list)) => Json(list).into_response(),
        Ok(Err(message)) => api_error(StatusCode::BAD_REQUEST, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `DELETE /api/studio/plugins` `{name}`: removes a plugin folder and everything it contributed.
pub(in crate::server) async fn remove_route(Json(body): Json<Value>) -> Response {
    let name = body.get("name").and_then(Value::as_str).unwrap_or("");
    match remove(name) {
        Ok(list) => Json(list).into_response(),
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}

/// `GET /api/studio/plugins/commands`: the slash commands every plugin brings.
pub(in crate::server) async fn commands_route() -> Response {
    Json(json!({"commands": commands()})).into_response()
}

/// `GET /api/studio/plugins/command?plugin=&name=`: the body of one plugin command.
pub(in crate::server) async fn command_route(Query(query): Query<HashMap<String, String>>) -> Response {
    let plugin = query.get("plugin").map(String::as_str).unwrap_or("");
    let name = query.get("name").map(String::as_str).unwrap_or("");
    match read_command(plugin, name) {
        Ok(body) => Json(json!({"plugin": plugin, "name": name, "body": body})).into_response(),
        Err(message) => api_error(StatusCode::NOT_FOUND, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-plugins-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_plugin_folder_is_found_and_counted() {
        let base = temp("describe");
        let plugin = Plugin {
            name: "demo".to_owned(),
            path: base.clone(),
            meta: json!({"version": "1.2.0", "description": "a demo", "author": "me"}),
        };
        fs::create_dir_all(base.join("skills/alphabet")).unwrap();
        fs::write(base.join("skills/alphabet/SKILL.md"), "---\nname: alphabet\n---\nx").unwrap();
        fs::create_dir_all(base.join("agents")).unwrap();
        fs::write(base.join("agents/reviewer.md"), "---\ndescription: r\n---\nbody").unwrap();
        fs::create_dir_all(base.join("commands")).unwrap();
        fs::write(base.join("commands/ship.md"), "---\ndescription: Ship it\n---\ngo").unwrap();
        fs::write(base.join("hooks.json"), "{\"hooks\":{\"on_file_edit\":[{\"command\":\"true\"},{\"command\":\"true\"}]}}").unwrap();
        let described = describe(&plugin);
        assert_eq!(described["skills"], 1);
        assert_eq!(described["agents"], 1);
        assert_eq!(described["commands"], 1);
        assert_eq!(described["hooks"], 2);
        assert_eq!(described["version"], "1.2.0");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn a_link_inside_a_plugin_is_refused() {
        #[cfg(unix)]
        {
            let from = temp("link-src");
            let to = temp("link-dst");
            fs::write(from.join("plugin.json"), "{\"name\": \"x\"}").unwrap();
            std::os::unix::fs::symlink("/etc/passwd", from.join("escape")).unwrap();
            let mut bytes = 0;
            let mut entries = 0;
            let error = copy_dir(&from, &to.join("x"), &mut bytes, &mut entries).unwrap_err();
            assert!(error.contains("link"), "{error}");
            let _ = fs::remove_dir_all(&from);
            let _ = fs::remove_dir_all(&to);
        }
    }

    #[test]
    fn commands_need_valid_slugs_and_the_body_is_returned() {
        assert!(read_command("../etc", "passwd").is_err());
        assert!(read_command("demo", "no/slash").is_err());
    }
    #[test]
    fn a_bad_address_or_branch_is_refused_before_git_runs() {
        assert!(install_from_git("ftp://x/y", "").unwrap_err().contains("https://"));
        assert!(install_from_git("/tmp/x", "").unwrap_err().contains("https://"));
        assert!(install_from_git("https://x/y z", "").unwrap_err().contains("space"));
        assert!(install_from_git("https://x/y", "-b main").unwrap_err().contains("branch"));
        assert!(install_from_git("https://x/y", "ma in").unwrap_err().contains("branch"));
    }

}
