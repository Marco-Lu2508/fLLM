//! The `diagnostics` tool: a quick syntax or build check of the workspace without a language server.
//! It picks a check from the file (or the project files), runs it with a time limit and returns the first
//! problems. Only fixed commands are ever run; the model cannot choose the command line.
use super::agent::{collect_output, AgentSession};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

const MAX_OUTPUT_LINES: usize = 40;
const MAX_OUTPUT_CHARS: usize = 5_000;
const DEFAULT_LIMIT: Duration = Duration::from_secs(120);

/// A check to run: a label for the model and a command line for the platform shell.
#[derive(Debug, PartialEq)]
pub(super) struct Check {
    pub label: &'static str,
    pub line: String,
    /// Where to run it, relative to the workspace root (empty: the root).
    pub dir: PathBuf,
}

fn has(root: &Path, dir: &Path, name: &str) -> bool {
    root.join(dir).join(name).is_file()
}

/// The nearest folder at or above `start` (inside `root`) that contains `marker`.
fn up_to(root: &Path, start: &Path, marker: &str) -> Option<PathBuf> {
    let mut dir = start.to_owned();
    loop {
        if has(root, &dir, marker) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Characters that could change the meaning of a quoted argument in `sh` or `cmd`.
fn safe_arg(text: &str) -> bool {
    !text.is_empty() && !text.chars().any(|c| matches!(c, '"' | '\'' | '`' | '$' | '%' | '!' | '&' | '|' | ';' | '<' | '>' | '^' | '\n' | '\r'))
}

/// Which check fits `file` (relative to `root`, forward slashes) or, without a file, the project.
pub(super) fn choose(root: &Path, file: Option<&str>) -> Result<Check, String> {
    let python = if cfg!(windows) { "python" } else { "python3" };
    if let Some(file) = file {
        if !safe_arg(file) {
            return Err("this file name has characters the check cannot pass safely".into());
        }
        let rel = Path::new(file);
        let parent = rel.parent().unwrap_or(Path::new("")).to_owned();
        let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        match ext.as_str() {
            "py" => return Ok(Check { label: "python -m py_compile", line: format!("{python} -m py_compile \"{file}\""), dir: PathBuf::new() }),
            "js" | "mjs" | "cjs" => return Ok(Check { label: "node --check", line: format!("node --check \"{file}\""), dir: PathBuf::new() }),
            "rs" => {
                if let Some(dir) = up_to(root, &parent, "Cargo.toml") {
                    return Ok(Check { label: "cargo check", line: "cargo check --message-format short --quiet".into(), dir });
                }
            }
            "ts" | "tsx" | "mts" | "cts" => {
                if let Some(dir) = up_to(root, &parent, "tsconfig.json") {
                    return Ok(Check { label: "tsc --noEmit", line: "npx --no-install tsc --noEmit --pretty false".into(), dir });
                }
            }
            "go" => {
                if let Some(dir) = up_to(root, &parent, "go.mod") {
                    return Ok(Check { label: "go vet", line: "go vet ./...".into(), dir });
                }
            }
            _ => {}
        }
    }
    let top = PathBuf::new();
    if has(root, &top, "Cargo.toml") {
        return Ok(Check { label: "cargo check", line: "cargo check --message-format short --quiet".into(), dir: top });
    }
    if has(root, &top, "tsconfig.json") {
        return Ok(Check { label: "tsc --noEmit", line: "npx --no-install tsc --noEmit --pretty false".into(), dir: top });
    }
    if has(root, &top, "go.mod") {
        return Ok(Check { label: "go vet", line: "go vet ./...".into(), dir: top });
    }
    if has(root, &top, "pyproject.toml") || has(root, &top, "setup.py") || has(root, &top, "requirements.txt") {
        return Ok(Check { label: "python -m compileall", line: format!("{python} -m compileall -q ."), dir: top });
    }
    Err(match file {
        Some(f) => format!("no automatic check is known for {f} and the workspace has no Cargo.toml, tsconfig.json, go.mod or Python project file; run the project's own build or test command with bash"),
        None => "no automatic check found: the workspace has no Cargo.toml, tsconfig.json, go.mod or Python project file; run the project's own build or test command with bash".into(),
    })
}

fn first_problems(text: &str) -> String {
    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let more = lines.len().saturating_sub(MAX_OUTPUT_LINES);
    lines.truncate(MAX_OUTPUT_LINES);
    let mut out = lines.join("\n");
    if out.chars().count() > MAX_OUTPUT_CHARS {
        out = out.chars().take(MAX_OUTPUT_CHARS).collect();
        out.push_str(" ...");
    }
    if more > 0 {
        out.push_str(&format!("\n... and {more} more lines"));
    }
    out
}

fn looks_missing(code: Option<i32>, output: &str) -> bool {
    let lower = output.to_lowercase();
    code == Some(127)
        || ["is not recognized as", "command not found", "cannot find the path", "could not determine which binary", "npm err! could not"].iter().any(|m| lower.contains(m))
}

/// Runs the check for `path` (a file of the workspace or empty for the whole project).
pub(super) fn run(session: &AgentSession, path: Option<&str>, timeout_s: Option<u64>) -> Result<String, String> {
    let root = session.root_dir()?;
    let rel = match path.map(str::trim).filter(|p| !p.is_empty() && *p != ".") {
        Some(p) => {
            let full = session.resolve_path(p)?;
            if full.is_dir() {
                None
            } else {
                Some(full.strip_prefix(&root).unwrap_or(&full).to_string_lossy().replace('\\', "/"))
            }
        }
        None => None,
    };
    if let Some(rel) = rel.as_deref() {
        if rel.ends_with(".json") && !rel.ends_with("tsconfig.json") {
            let text = std::fs::read_to_string(root.join(rel)).map_err(|e| format!("cannot read {rel}: {e}"))?;
            return Ok(match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(_) => format!("No problems found in {rel} (valid JSON)."),
                Err(e) => format!("Problems in {rel} (JSON): {e}"),
            });
        }
    }
    let check = choose(&root, rel.as_deref())?;
    let dir = root.join(&check.dir);
    let limit = timeout_s.map_or(DEFAULT_LIMIT, |s| Duration::from_secs(s.clamp(5, 300)));
    let mut cmd = super::agent_jobs::shell_command(&check.line);
    let mut child = cmd
        .current_dir(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", check.label))?;
    let (out, err) = (collect_output(child.stdout.take()), collect_output(child.stderr.take()));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() >= limit => {
                super::agent_jobs::kill_tree(child.id());
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(e.to_string()),
        }
    };
    let mut text = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    text.push_str(&String::from_utf8_lossy(&err.join().unwrap_or_default()));
    let Some(status) = status else {
        return Ok(format!("The check ({}) did not finish within {} s and was stopped.", check.label, limit.as_secs()));
    };
    if status.success() {
        return Ok(format!("No problems found ({}).", check.label));
    }
    if looks_missing(status.code(), &text) {
        return Ok(format!("The check ({}) is not available here: the tool is not installed or not on the path. Run the project's own build or test command with bash instead.", check.label));
    }
    Ok(format!("Problems found ({}, exit code {}):\n{}", check.label, status.code().map_or_else(|| "unknown".into(), |c| c.to_string()), first_problems(&text)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-diag-{}-{}", std::process::id(), super::super::new_id("t")));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_check_follows_the_file_and_the_project() {
        let root = tmp();
        fs::create_dir_all(root.join("crate/src")).unwrap();
        fs::write(root.join("crate/Cargo.toml"), "").unwrap();
        let c = choose(&root, Some("crate/src/lib.rs")).unwrap();
        assert_eq!((c.label, c.dir), ("cargo check", PathBuf::from("crate")));
        assert!(choose(&root, Some("tool.js")).unwrap().line.contains("node --check"));
        assert!(choose(&root, Some("notes.txt")).is_err());
        assert!(choose(&root, Some("a\"b.py")).is_err());
        fs::write(root.join("go.mod"), "").unwrap();
        assert_eq!(choose(&root, None).unwrap().label, "go vet");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn long_output_is_cut_to_the_first_problems() {
        let text: String = (0..100).map(|i| format!("error {i}\n")).collect();
        let cut = first_problems(&text);
        assert!(cut.starts_with("error 0") && cut.contains("and 60 more lines"));
    }

    #[test]
    fn json_files_are_checked_without_a_process() {
        let root = tmp();
        let s = AgentSession::default();
        s.set_root(root.to_str().unwrap()).unwrap();
        fs::write(root.join("ok.json"), "{\"a\": 1}").unwrap();
        fs::write(root.join("bad.json"), "{\"a\": ").unwrap();
        assert!(run(&s, Some("ok.json"), None).unwrap().contains("valid JSON"));
        assert!(run(&s, Some("bad.json"), None).unwrap().starts_with("Problems in bad.json"));
        let _ = fs::remove_dir_all(root);
    }
}
