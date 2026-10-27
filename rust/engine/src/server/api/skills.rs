//! Skills: procedural memory as Markdown files. A skill is a folder `<name>/SKILL.md` with a small front matter
//! (`name`, `description`, `when`) followed by the steps. Two places hold them: the user's own folder `skills/` next
//! to `studio.json` (`scope` "user") and `.fllm/skills/` inside a workspace folder (`scope` "workspace"). The routes
//! list, read, write and delete them; a name is a short lowercase slug, so no request can leave the skills folder.
use super::*;
use axum::extract::Query;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SKILL_BYTES: usize = 64 * 1024;
const MAX_SKILLS: usize = 500;
const FILE: &str = "SKILL.md";

/// A skill name: 1 to 64 characters, lowercase letters, digits, `-` and `_`, starting with a letter or digit.
pub(super) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else { return false };
    name.len() <= 64
        && (first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// The front matter (`key: value` lines between two `---` lines at the top) and the rest of the text.
pub(super) fn front_matter(text: &str) -> (BTreeMap<String, String>, &str) {
    let mut meta = BTreeMap::new();
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let Some(rest) = text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n")) else { return (meta, text) };
    let mut consumed = 0;
    for line in rest.split_inclusive('\n') {
        consumed += line.len();
        let trimmed = line.trim_end();
        if trimmed == "---" {
            return (meta, &rest[consumed..]);
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
            if !key.trim().is_empty() {
                meta.insert(key.trim().to_ascii_lowercase(), value.to_owned());
            }
        }
    }
    (BTreeMap::new(), text)
}

fn skill_dir(base: &Path, name: &str) -> Result<PathBuf, String> {
    if !valid_name(name) {
        return Err("a skill name has 1 to 64 characters: lowercase letters, digits, '-' and '_'".into());
    }
    Ok(base.join(name))
}

/// A folder or file that is a link is never followed: a skill must live inside the skills folder.
fn is_plain(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| !m.file_type().is_symlink())
}

pub(super) fn list_in(base: &Path) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(base) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let file = entry.path().join(FILE);
        if !valid_name(&name) || !is_plain(&entry.path()) || !is_plain(&file) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&file) else { continue };
        let (meta, body) = front_matter(&text);
        let description = meta.get("description").cloned().filter(|d| !d.is_empty()).unwrap_or_else(|| {
            body.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#')).unwrap_or_default().chars().take(160).collect()
        });
        out.push(json!({"name": name, "description": description, "when": meta.get("when").cloned().unwrap_or_default(), "bytes": text.len()}));
        if out.len() >= MAX_SKILLS {
            break;
        }
    }
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    out
}

pub(super) fn read_in(base: &Path, name: &str) -> Result<String, String> {
    let dir = skill_dir(base, name)?;
    let file = dir.join(FILE);
    if !is_plain(&dir) || !is_plain(&file) {
        return Err(format!("no skill named '{name}'"));
    }
    fs::read_to_string(&file).map_err(|_| format!("no skill named '{name}'"))
}

pub(super) fn write_in(base: &Path, name: &str, content: &str) -> Result<(), String> {
    let dir = skill_dir(base, name)?;
    if content.trim().is_empty() {
        return Err("the skill is empty".into());
    }
    if content.len() > MAX_SKILL_BYTES {
        return Err(format!("a skill can be at most {} KB", MAX_SKILL_BYTES / 1024));
    }
    if dir.exists() && !is_plain(&dir) {
        return Err("that skill folder is a link".into());
    }
    if !dir.exists() && list_in(base).len() >= MAX_SKILLS {
        return Err(format!("at most {MAX_SKILLS} skills"));
    }
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create the skill folder: {e}"))?;
    let file = dir.join(FILE);
    if file.exists() && !is_plain(&file) {
        return Err("SKILL.md is a link".into());
    }
    fs::write(&file, content).map_err(|e| format!("cannot write the skill: {e}"))
}

pub(super) fn delete_in(base: &Path, name: &str) -> Result<(), String> {
    let dir = skill_dir(base, name)?;
    let file = dir.join(FILE);
    if !is_plain(&dir) || !is_plain(&file) {
        return Err(format!("no skill named '{name}'"));
    }
    fs::remove_file(&file).map_err(|e| format!("cannot delete the skill: {e}"))?;
    // other files the user put next to it stay; an empty folder goes
    let _ = fs::remove_dir(&dir);
    Ok(())
}

fn user_base() -> PathBuf {
    crate::hub::studio_data_file("skills")
}

/// The folder of the skills of a scope. A workspace is an absolute folder path given by the page.
fn base_for(scope: &str, workspace: Option<&str>) -> Result<PathBuf, String> {
    match scope {
        "" | "user" => Ok(user_base()),
        "workspace" => {
            let given = workspace.map(str::trim).filter(|w| !w.is_empty()).ok_or("choose a workspace folder first")?;
            let path = PathBuf::from(given);
            if !path.is_absolute() || !path.is_dir() {
                return Err("the workspace must be an existing absolute folder".into());
            }
            Ok(path.join(".fllm").join("skills"))
        }
        other => Err(format!("unknown scope '{other}' (user or workspace)")),
    }
}

fn bad(message: &str) -> Response {
    api_error(StatusCode::BAD_REQUEST, message)
}

/// `GET /api/studio/skills?workspace=<folder>`: the skills of the user and, when a folder is given, of that workspace.
pub(in crate::server) async fn list(Query(query): Query<HashMap<String, String>>) -> Response {
    let mut skills: Vec<Value> = Vec::new();
    for scope in ["user", "workspace"] {
        let Ok(base) = base_for(scope, query.get("workspace").map(String::as_str)) else { continue };
        for mut skill in list_in(&base) {
            skill["scope"] = json!(scope);
            skills.push(skill);
        }
    }
    // skills of plugins come last and cannot shadow your own
    for (plugin, base) in super::plugins::skill_bases() {
        let scope = format!("plugin:{plugin}");
        for mut skill in list_in(&base) {
            let name = skill["name"].as_str().unwrap_or("");
            if skills.iter().any(|s| s["name"].as_str() == Some(name)) {
                continue;
            }
            skill["scope"] = json!(scope);
            skill["read_only"] = json!(true);
            skills.push(skill);
        }
    }
    Json(json!({"skills": skills, "user_folder": user_base().display().to_string()})).into_response()
}

/// `GET /api/studio/skills/read?name=&scope=&workspace=`: `{name, scope, content}`.
pub(in crate::server) async fn read(Query(query): Query<HashMap<String, String>>) -> Response {
    let scope = query.get("scope").map_or("user", String::as_str);
    let name = query.get("name").map_or("", String::as_str);
    let found = match scope.strip_prefix("plugin:") {
        Some(plugin) => super::plugins::folder(plugin)
            .map(|path| path.join("skills"))
            .ok_or_else(|| format!("there is no plugin '{plugin}'"))
            .and_then(|base| read_in(&base, name)),
        None => base_for(scope, query.get("workspace").map(String::as_str)).and_then(|base| read_in(&base, name)),
    };
    match found {
        Ok(content) => Json(json!({"name": name, "scope": scope, "content": content})).into_response(),
        Err(e) => api_error(StatusCode::NOT_FOUND, &e),
    }
}

/// `POST /api/studio/skills` `{name, scope?, workspace?, content}` creates or replaces a skill.
pub(in crate::server) async fn write(Json(body): Json<Value>) -> Response {
    let scope = body["scope"].as_str().unwrap_or("user");
    let result = base_for(scope, body["workspace"].as_str())
        .and_then(|base| write_in(&base, body["name"].as_str().unwrap_or_default(), body["content"].as_str().unwrap_or_default()));
    match result {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => bad(&e),
    }
}

/// `DELETE /api/studio/skills?name=&scope=&workspace=`.
pub(in crate::server) async fn remove(Query(query): Query<HashMap<String, String>>) -> Response {
    let scope = query.get("scope").map_or("user", String::as_str);
    let name = query.get("name").map_or("", String::as_str);
    match base_for(scope, query.get("workspace").map(String::as_str)).and_then(|base| delete_in(&base, name)) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => api_error(StatusCode::NOT_FOUND, &e),
    }
}

// ---- skills that come with the app -----------------------------------------------------------

/// Skills that ship with the binary: they are installed with one click, and installing writes a normal skill file that the
/// user can change afterwards. Nothing here is magic — the same file could be typed by hand.
pub(super) const LIBRARY: [(&str, &str); 6] = [
    ("commit-message", include_str!("../../builtin_skills/commit-message.md")),
    ("code-review", include_str!("../../builtin_skills/code-review.md")),
    ("debug-bisect", include_str!("../../builtin_skills/debug-bisect.md")),
    ("tests-first", include_str!("../../builtin_skills/tests-first.md")),
    ("sql-query", include_str!("../../builtin_skills/sql-query.md")),
    ("release-notes", include_str!("../../builtin_skills/release-notes.md")),
];

/// `GET /api/studio/skills/library`: the built-in skills with their description, and whether each is installed.
pub(in crate::server) async fn library() -> Response {
    let installed: Vec<String> = list_in(&user_base()).into_iter().filter_map(|s| s["name"].as_str().map(str::to_owned)).collect();
    let items: Vec<Value> = LIBRARY
        .iter()
        .map(|(name, text)| {
            let (meta, _) = front_matter(text);
            json!({
                "name": name,
                "description": meta.get("description").cloned().unwrap_or_default(),
                "when": meta.get("when").cloned().unwrap_or_default(),
                "installed": installed.iter().any(|i| i == name),
            })
        })
        .collect();
    Json(json!({"skills": items, "folder": user_base().display().to_string()})).into_response()
}

/// `POST /api/studio/skills/library` `{name}`: writes the built-in skill into the user folder.
pub(in crate::server) async fn install_builtin(Json(body): Json<Value>) -> Response {
    let name = body.get("name").and_then(Value::as_str).unwrap_or("");
    let Some((_, text)) = LIBRARY.iter().find(|(n, _)| *n == name) else {
        return api_error(StatusCode::NOT_FOUND, "there is no built-in skill with that name");
    };
    match write_in(&user_base(), name, text) {
        Ok(()) => Json(json!({"ok": true, "name": name})).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

/// A skill from a URL: anything that serves a SKILL.md as text, for example a raw GitHub link.
fn clean_skill_url(raw: &str) -> Result<String, String> {
    let url = raw.trim();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("the address has to start with http:// or https://".to_owned());
    }
    if url.len() > 500 {
        return Err("the address is too long".to_owned());
    }
    Ok(url.to_owned())
}

/// `POST /api/studio/skills/import` `{url, name?}`: fetches a SKILL.md and stores it.
pub(in crate::server) async fn import(Json(body): Json<Value>) -> Response {
    let Ok(url) = clean_skill_url(body.get("url").and_then(Value::as_str).unwrap_or("")) else {
        return api_error(StatusCode::BAD_REQUEST, "give an http(s) address of a SKILL.md");
    };
    let fetched = match super::agent_net::fetch_public_text(&url).await {
        Ok(text) => text,
        Err(e) => return api_error(StatusCode::BAD_GATEWAY, &e),
    };
    let (meta, _) = front_matter(&fetched);
    let from_body = body.get("name").and_then(Value::as_str).unwrap_or("").trim().to_owned();
    let name = if !from_body.is_empty() {
        from_body
    } else {
        meta.get("name").cloned().unwrap_or_default()
    };
    if !valid_name(&name) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "the file needs a front matter with a name (1 to 64 characters: lowercase letters, digits, '-' and '_'), or pass a name",
        );
    }
    match write_in(&user_base(), &name, &fetched) {
        Ok(()) => Json(json!({"ok": true, "name": name, "bytes": fetched.len()})).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-skills-{}-{}", std::process::id(), new_id("t")));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn names_are_slugs_and_cannot_climb_out() {
        for good in ["a", "deploy-app", "x_1", "0day"] {
            assert!(valid_name(good), "{good}");
        }
        let long = "a".repeat(65);
        for bad in ["", "../x", "a/b", "a\\b", "A", "-a", ".hidden", "a b", "a.md", "..", long.as_str()] {
            assert!(!valid_name(bad), "{bad}");
        }
        let base = temp();
        assert!(write_in(&base, "../escape", "x").is_err());
        assert!(read_in(&base, "../../etc/passwd").is_err());
        assert!(delete_in(&base, "a/../b").is_err());
        assert!(!base.parent().unwrap().join("escape").exists());
    }

    #[test]
    fn front_matter_is_read_and_missing_front_matter_is_tolerated() {
        let (meta, body) = front_matter("---\nname: deploy\ndescription: \"Ship it\"\nwhen: after a release\n---\n# Steps\n1. go\n");
        assert_eq!(meta["name"], "deploy");
        assert_eq!(meta["description"], "Ship it");
        assert_eq!(meta["when"], "after a release");
        assert!(body.starts_with("# Steps"));
        let (meta, body) = front_matter("# Just text\n");
        assert!(meta.is_empty() && body == "# Just text\n");
        let (meta, body) = front_matter("---\nname: open\nno closing line\n");
        assert!(meta.is_empty() && body.starts_with("---"));
    }

    #[test]
    fn write_list_read_delete() {
        let base = temp();
        write_in(&base, "release", "---\nname: release\ndescription: Cut a release\nwhen: the user says release\n---\n1. tag\n").unwrap();
        write_in(&base, "notes", "# Notes\nPlain text without front matter.\n").unwrap();
        let listed = list_in(&base);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0]["name"], "notes");
        assert_eq!(listed[0]["description"], "Plain text without front matter.");
        assert_eq!(listed[1]["description"], "Cut a release");
        assert!(read_in(&base, "release").unwrap().contains("1. tag"));
        assert!(write_in(&base, "empty", "  \n").is_err());
        assert!(write_in(&base, "big", &"x".repeat(MAX_SKILL_BYTES + 1)).is_err());
        delete_in(&base, "release").unwrap();
        assert!(read_in(&base, "release").is_err());
        assert!(!base.join("release").exists());
        assert!(delete_in(&base, "release").is_err());
    }

    #[test]
    fn workspace_scope_needs_an_existing_absolute_folder() {
        assert!(base_for("workspace", None).is_err());
        assert!(base_for("workspace", Some("relative")).is_err());
        assert!(base_for("elsewhere", None).is_err());
        let dir = temp();
        let base = base_for("workspace", Some(dir.to_str().unwrap())).unwrap();
        assert!(base.ends_with(Path::new(".fllm").join("skills")));
    }

    #[cfg(unix)]
    #[test]
    fn a_linked_skill_folder_is_not_followed() {
        let base = temp();
        let outside = temp();
        fs::write(outside.join(FILE), "secret").unwrap();
        std::os::unix::fs::symlink(&outside, base.join("linked")).unwrap();
        assert!(read_in(&base, "linked").is_err());
        assert!(list_in(&base).is_empty());
    }
    #[test]
    fn every_built_in_skill_has_a_name_description_and_when() {
        for (name, text) in LIBRARY {
            assert!(valid_name(name), "{name}");
            let (meta, body) = front_matter(text);
            assert_eq!(meta.get("name").map(String::as_str), Some(name), "{name}: front matter name has to match");
            assert!(!meta.get("description").unwrap_or(&String::new()).is_empty(), "{name}: no description");
            assert!(!meta.get("when").unwrap_or(&String::new()).is_empty(), "{name}: no when");
            assert!(body.lines().count() >= 3, "{name}: the body is too short to be useful");
        }
    }

    #[test]
    fn a_skill_url_has_to_be_http() {
        assert!(clean_skill_url("https://example.org/SKILL.md").is_ok());
        assert!(clean_skill_url(" file:///etc/passwd ").is_err());
        assert!(clean_skill_url("").is_err());
        assert!(clean_skill_url(&format!("https://{}", "a".repeat(600))).is_err());
    }

}
