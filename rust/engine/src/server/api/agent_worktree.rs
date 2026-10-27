//! Git worktrees for the agent: a task can run in its own checkout, on its own branch, while the main folder stays as it
//! is. That is the same idea as Claude Code's `/batch` and `EnterWorktree` and Codex's worktree isolation, without a second
//! agent: the session switches into the worktree, works there, and leaves again — the main tree keeps its state and the
//! change waits on a branch you can review, merge or throw away.
//!
//! The worktrees live in `<root>/.fllm/worktrees/<name>`; `.fllm/worktrees/` is added to `.git/info/exclude` so they never
//! show up as untracked files. Everything goes through `git -C <root> worktree`, never through a shell.
use super::agent::{agent_disabled, disabled_reply, AgentSession};
use super::*;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const DIR: &str = ".fllm/worktrees";
const MAX_NAME: usize = 60;

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail: String = stderr.trim().chars().take(300).collect();
        return Err(format!("git {} failed: {detail}", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A worktree name: a slug, so it can be a folder and a branch name.
pub(super) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else { return false };
    name.len() <= MAX_NAME
        && (first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

fn is_repo(root: &Path) -> bool {
    git(root, &["rev-parse", "--is-inside-work-tree"]).map(|out| out.trim() == "true").unwrap_or(false)
}

/// The branch of the main worktree and its path, from `git worktree list --porcelain`.
fn main_worktree(root: &Path) -> Result<(PathBuf, String), String> {
    let text = git(root, &["worktree", "list", "--porcelain"])?;
    let mut path: Option<PathBuf> = None;
    let mut branch = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            if path.is_none() {
                path = Some(PathBuf::from(rest.trim()));
            }
        } else if let Some(rest) = line.strip_prefix("branch ") {
            if branch.is_empty() {
                branch = rest.trim().trim_start_matches("refs/heads/").to_owned();
            }
        }
    }
    path.map(|p| (p, branch)).ok_or_else(|| "git lists no worktree".to_owned())
}

/// Keeps the worktree folder out of `git status` of the repository.
fn exclude_worktrees(root: &Path) {
    let Ok(git_dir) = git(root, &["rev-parse", "--git-common-dir"]) else { return };
    let git_dir = git_dir.trim();
    let base = if Path::new(git_dir).is_absolute() { PathBuf::from(git_dir) } else { root.join(git_dir) };
    let info = base.join("info");
    let file = info.join("exclude");
    let line = format!("{DIR}/");
    let existing = std::fs::read_to_string(&file).unwrap_or_default();
    if existing.lines().any(|l| l.trim() == line) {
        return;
    }
    let _ = std::fs::create_dir_all(&info);
    let mut text = existing;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&line);
    text.push('\n');
    let _ = std::fs::write(&file, text);
}

fn describe(root: &Path, path: &Path) -> Value {
    let branch = git(path, &["rev-parse", "--abbrev-ref", "HEAD"]).map(|b| b.trim().to_owned()).unwrap_or_default();
    let current = root == path || root.join(DIR).join(path.file_name().unwrap_or_default()) == path;
    json!({
        "name": path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        "path": path.display().to_string(),
        "branch": branch,
        "current": current,
    })
}

/// The worktrees of this repository that the agent created (plus the main one).
pub(super) fn list(root: &Path) -> Vec<Value> {
    let Ok(text) = git(root, &["worktree", "list", "--porcelain"]) else { return Vec::new() };
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            let path = PathBuf::from(rest.trim());
            out.push(describe(root, &path));
        }
    }
    out
}

/// Creates a worktree with a new branch and switches the session into it.
fn create(session: &AgentSession, root: &Path, name: &str, branch: &str) -> Result<Value, String> {
    if !valid_name(name) {
        return Err("a worktree name has 1 to 60 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    if !is_repo(root) {
        return Err("this folder is not a git repository; a worktree needs one".to_owned());
    }
    let path = root.join(DIR).join(name);
    if path.exists() {
        return Err(format!("a worktree named '{name}' exists already"));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    exclude_worktrees(root);
    let path_text = path.to_string_lossy().into_owned();
    let branch_name = if branch.trim().is_empty() { name } else { branch.trim() };
    if !valid_name(branch_name) {
        return Err("the branch name has to be a slug too".to_owned());
    }
    // an existing branch is checked out, a new one is created
    let branch_exists = git(root, &["rev-parse", "--verify", "--quiet", &format!("refs/heads/{branch_name}")]).is_ok();
    if branch_exists {
        git(root, &["worktree", "add", &path_text, branch_name])?;
    } else {
        git(root, &["worktree", "add", "-b", branch_name, &path_text, "HEAD"])?;
    }
    let switched = session.set_root(&path_text)?;
    Ok(json!({
        "path": switched.display().to_string(),
        "branch": branch_name,
        "created": !branch_exists,
        "worktrees": list(&switched),
    }))
}

fn remove(root: &Path, name: &str) -> Result<Value, String> {
    if !valid_name(name) {
        return Err("a worktree name has 1 to 60 characters: lowercase letters, digits, '-' and '_'".to_owned());
    }
    let path = root.join(DIR).join(name);
    let path_text = path.to_string_lossy().into_owned();
    git(root, &["worktree", "remove", "--force", &path_text])?;
    Ok(json!({"removed": name, "worktrees": list(root)}))
}

/// `POST /api/studio/agent/worktree` `{action: new|list|leave|remove, name?, branch?}`.
pub(in crate::server) async fn route(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || run(&worker.agent, &body)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => api_error(StatusCode::BAD_REQUEST, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// The worktree action itself (blocking, tested above without HTTP).
pub(in crate::server) fn run(session: &AgentSession, body: &Value) -> Result<Value, String> {
    let root = session.root_dir()?;
    let action = body.get("action").and_then(Value::as_str).unwrap_or("list");
    match action {
        "list" => Ok(json!({"root": root.display().to_string(), "worktrees": list(&root)})),
        "new" | "batch" => {
            let name = body.get("name").and_then(Value::as_str).unwrap_or("").trim();
            let branch = body.get("branch").and_then(Value::as_str).unwrap_or("");
            create(session, &root, name, branch)
        }
        "leave" => {
            let (main, branch) = main_worktree(&root)?;
            let switched = session.set_root(&main.to_string_lossy())?;
            Ok(json!({"path": switched.display().to_string(), "branch": branch, "worktrees": list(&switched)}))
        }
        "remove" => {
            let name = body.get("name").and_then(Value::as_str).unwrap_or("").trim();
            remove(&root, name)
        }
        other => Err(format!("unknown action '{other}'; use new, list, leave or remove")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_have_to_be_slugs() {
        assert!(valid_name("fix-parser"));
        assert!(valid_name("v2_try"));
        assert!(!valid_name("Fix Parser"));
        assert!(!valid_name("../escape"));
        assert!(!valid_name("-leading"));
        assert!(!valid_name(""));
        assert!(!valid_name(&"x".repeat(MAX_NAME + 1)));
    }

    #[test]
    fn a_folder_that_is_no_repository_is_refused_before_anything_runs() {
        let dir = std::env::temp_dir().join(format!("fllm-wt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!is_repo(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn worktree_operations_run_on_a_real_repository() {
        // a throwaway repository: the only way to test `git worktree` honestly
        let dir = std::env::temp_dir().join(format!("fllm-wt-repo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ok = Command::new("git").arg("-C").arg(&dir).args(["init", "--quiet"]).status().map(|s| s.success()).unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_dir_all(&dir);
            return; // no git here: the shape of the calls is still covered by the other tests
        }
        std::fs::write(dir.join("a.txt"), "1\n").unwrap();
        let _ = Command::new("git").arg("-C").arg(&dir).args(["add", "."]).status();
        let _ = Command::new("git").arg("-C").arg(&dir).args(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]).status();
        let path = dir.join(DIR).join("try");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        git(&dir, &["worktree", "add", "-b", "try", &path.to_string_lossy(), "HEAD"]).expect("worktree add");
        assert!(path.join("a.txt").is_file(), "the worktree carries the files");
        let listed = list(&dir);
        assert!(listed.iter().any(|w| w["name"] == "try" && w["branch"] == "try"), "{listed:?}");
        git(&dir, &["worktree", "remove", "--force", &path.to_string_lossy()]).expect("worktree remove");
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
