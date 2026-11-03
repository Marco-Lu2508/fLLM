//! `fllm-rs agent --watch`: file-event hooks. The agent watches a folder; when files are created, changed or deleted, the
//! `file_changed` hooks of the Studio folder (`hooks.json`, or a plugin) run for each file whose path matches the hook's globs:
//!
//! ```json
//! {"hooks": {"file_changed": [
//!   {"matcher": "src/**/*.rs", "on": "modified", "command": "cargo fmt --check"},
//!   {"matcher": "docs/**/*.md", "prompt": "{{file}} changed: check the links in it and fix dead ones."}
//! ]}}
//! ```
//!
//! A hook with a `command` runs on the server like every other hook (its output is printed here). A hook with a `prompt` starts
//! one agent run with that text, the path and the kind of change; the run uses the agent options given after `--watch`
//! (`--allow-edit`, `--url`, ...). Files are found by polling (modification time and size), so it works the same on every
//! system and needs nothing installed; `.git`, `target`, `node_modules`, `.fllm` and everything `.gitignore` names at the top
//! level are skipped. Changes the agent makes during its own run are not reported back to it, so a hook cannot start itself.
use crate::agent_cli::{HookReply, HttpBackend};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub const HELP: &str = "usage: fllm-rs agent --watch [--folder DIR] [--interval MS] [--max-runs N] [--url URL] [AGENT OPTIONS]
Watches the folder and fires the file_changed hooks (hooks.json in the Studio folder, or a plugin) for every created, changed or deleted
file whose path matches the hook's globs. A hook with a command runs it; a hook with a prompt starts an agent run with that text
(AGENT OPTIONS such as --allow-edit, --allow-command, --model are passed to it). --interval: milliseconds between scans (default 1000).
--max-runs: stop after this many agent runs (default 20), so a prompt that keeps changing files cannot run for ever. Ctrl-C ends it.";

const SKIP_DIRS: [&str; 6] = [".git", "target", "node_modules", ".fllm", ".venv", "__pycache__"];
const MAX_FILES: usize = 30_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub path: String,
    /// `created`, `modified` or `deleted`.
    pub kind: &'static str,
}

pub type Snapshot = BTreeMap<String, (u64, SystemTime)>;

fn top_level_ignores(root: &Path) -> Vec<String> {
    // only plain names at the top level (`dist/`, `secret.env`); patterns are not interpreted
    std::fs::read_to_string(root.join(".gitignore"))
        .map(|t| {
            t.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('!') && !l.contains('*'))
                .map(|l| l.trim_matches('/'))
                .filter(|l| !l.is_empty() && !l.contains('/'))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub fn snapshot(root: &Path) -> Snapshot {
    let ignores = top_level_ignores(root);
    let mut out = Snapshot::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            let at_top = dir == root;
            if meta.is_dir() {
                if SKIP_DIRS.contains(&name.as_str()) || (at_top && ignores.contains(&name)) {
                    continue;
                }
                stack.push(path);
            } else if meta.is_file() {
                if at_top && ignores.contains(&name) {
                    continue;
                }
                if out.len() >= MAX_FILES {
                    return out;
                }
                let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                out.insert(rel, (meta.len(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH)));
            }
        }
    }
    out
}

pub fn diff(old: &Snapshot, new: &Snapshot) -> Vec<Change> {
    let mut changes = Vec::new();
    for (path, stamp) in new {
        match old.get(path) {
            None => changes.push(Change { path: path.clone(), kind: "created" }),
            Some(before) if before != stamp => changes.push(Change { path: path.clone(), kind: "modified" }),
            _ => {}
        }
    }
    for path in old.keys().filter(|p| !new.contains_key(*p)) {
        changes.push(Change { path: path.clone(), kind: "deleted" });
    }
    changes
}

/// One scan-and-react loop. `fire` runs the hooks for a change and says what they asked for; `run_agent` runs one agent task.
/// Stops after `cycles` scans (`None` = never) or `max_runs` agent runs. Returns the number of agent runs.
pub fn watch_loop(
    root: &Path,
    interval: Duration,
    cycles: Option<usize>,
    max_runs: usize,
    fire: &mut dyn FnMut(&Change) -> HookReply,
    run_agent: &mut dyn FnMut(&str) -> bool,
    say: &mut dyn FnMut(&str),
) -> usize {
    let mut known = snapshot(root);
    let mut runs = 0usize;
    let mut scans = 0usize;
    let mut pending: Vec<Change> = Vec::new();
    loop {
        if cycles.is_some_and(|c| scans >= c) || runs >= max_runs {
            break;
        }
        std::thread::sleep(interval);
        scans += 1;
        let now = snapshot(root);
        let found = diff(&known, &now);
        known = now;
        // a file still being written changes on two scans in a row: react when a scan finds nothing new
        if !found.is_empty() {
            for c in found {
                if !pending.iter().any(|p| p.path == c.path) {
                    pending.push(c);
                }
            }
            continue;
        }
        if pending.is_empty() {
            continue;
        }
        for change in std::mem::take(&mut pending) {
            let reply = fire(&change);
            if reply.blocked {
                say(&format!("[watch] {} {}: a hook reported a problem: {}", change.kind, change.path, reply.reason));
            }
            if !reply.context.trim().is_empty() && reply.prompts.is_empty() {
                say(&format!("[watch] {} {}: {}", change.kind, change.path, reply.context.trim()));
            }
            for prompt in &reply.prompts {
                if runs >= max_runs {
                    say("[watch] --max-runs reached");
                    return runs;
                }
                runs += 1;
                say(&format!("[watch] {} {}: starting an agent run", change.kind, change.path));
                let task = format!("The file {} was {}. {}", change.path, change.kind, prompt.trim());
                let ok = run_agent(&task);
                say(&format!("[watch] agent run {}", if ok { "finished" } else { "failed" }));
                // what the agent changed itself is not news to it
                known = snapshot(root);
            }
        }
    }
    runs
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let mut folder: Option<PathBuf> = None;
    let mut interval = 1000u64;
    let mut max_runs = 20usize;
    let mut forward: Vec<String> = Vec::new();
    let mut url = "http://127.0.0.1:8000".to_owned();
    let mut api_key = std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty());
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {a}"))
        };
        match a {
            "--folder" => folder = Some(PathBuf::from(next(&mut i)?)),
            "--interval" => interval = next(&mut i)?.parse::<u64>().ok().filter(|n| (100..=60_000).contains(n)).ok_or("--interval is 100 to 60000 milliseconds")?,
            "--max-runs" => max_runs = next(&mut i)?.parse().map_err(|_| "--max-runs is a number".to_owned())?,
            "--url" => {
                url = next(&mut i)?;
                forward.extend(["--url".into(), url.clone()]);
            }
            "--api-key" => {
                let key = next(&mut i)?;
                api_key = Some(key.clone());
                forward.extend(["--api-key".into(), key]);
            }
            "--model" | "--max-steps" | "--max-tokens" | "--temperature" | "--profile" | "--system" | "--allow-rule" | "--deny-rule" | "--ask-rule" => {
                forward.push(a.to_owned());
                forward.push(next(&mut i)?);
            }
            "--allow-edit" | "--allow-command" | "--allow-network" | "--allow-memory" | "--yes" | "--plan" => forward.push(a.to_owned()),
            other => return Err(format!("unknown option {other} for --watch; see fllm-rs agent --watch --help")),
        }
        i += 1;
    }
    let folder = match folder {
        Some(f) => f,
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let folder = crate::agent_cli::plain_path(&folder.canonicalize().map_err(|e| format!("the folder cannot be used: {e}"))?);
    let mut http = HttpBackend::new(&url, api_key)?;
    http.set_folder(&folder.display().to_string())?;
    eprintln!("[watch] watching {} (every {interval} ms); Ctrl-C ends it", folder.display());
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut fire = |c: &Change| crate::agent_cli::Backend::fire_hook(&mut http, "file_changed", &c.path, &json!({"path": c.path, "kind": c.kind}));
    let folder_text = folder.display().to_string();
    let mut run_agent = |task: &str| {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("agent").arg("-p").arg(task).arg("--folder").arg(&folder_text).arg("--quiet").args(&forward);
        cmd.status().map(|s| s.success()).unwrap_or(false)
    };
    let runs = watch_loop(&folder, Duration::from_millis(interval), None, max_runs, &mut fire, &mut run_agent, &mut |line| eprintln!("{line}"));
    eprintln!("[watch] finished after {runs} agent run(s)");
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-watch-{tag}-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(3)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn snapshots_skip_build_folders_and_ignored_names() {
        let root = temp("snap");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::create_dir_all(root.join(".fllm")).unwrap();
        std::fs::create_dir_all(root.join("dist")).unwrap();
        std::fs::write(root.join("src/a.rs"), "x").unwrap();
        std::fs::write(root.join("target/b.rs"), "x").unwrap();
        std::fs::write(root.join(".fllm/c.md"), "x").unwrap();
        std::fs::write(root.join("dist/d.js"), "x").unwrap();
        std::fs::write(root.join("secret.env"), "x").unwrap();
        std::fs::write(root.join(".gitignore"), "# build\ndist/\nsecret.env\n*.log\n").unwrap();
        let keys: Vec<String> = snapshot(&root).keys().cloned().collect();
        assert_eq!(keys, vec![".gitignore", "src/a.rs"]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn diff_names_created_modified_and_deleted() {
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(5);
        let old: Snapshot = [("a".to_owned(), (1, t0)), ("b".to_owned(), (1, t0)), ("c".to_owned(), (1, t0))].into_iter().collect();
        let new: Snapshot = [("a".to_owned(), (1, t0)), ("b".to_owned(), (2, t1)), ("d".to_owned(), (1, t0))].into_iter().collect();
        assert_eq!(
            diff(&old, &new),
            vec![Change { path: "b".into(), kind: "modified" }, Change { path: "d".into(), kind: "created" }, Change { path: "c".into(), kind: "deleted" }]
        );
        assert!(diff(&old, &old).is_empty());
    }

    #[test]
    fn a_prompt_hook_starts_a_run_once_and_the_runs_own_edits_are_ignored() {
        let root = temp("loop");
        std::fs::write(root.join("a.md"), "one").unwrap();
        let writer_root = root.clone();
        // a second thread edits the file while the loop polls
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            std::fs::write(writer_root.join("a.md"), "two, longer").unwrap();
            std::fs::write(writer_root.join("new.txt"), "n").unwrap();
        });
        let fired = std::cell::RefCell::new(Vec::<String>::new());
        let tasks = std::cell::RefCell::new(Vec::<String>::new());
        let root_for_agent = root.clone();
        let runs = watch_loop(
            &root,
            Duration::from_millis(60),
            Some(14),
            5,
            &mut |c| {
                fired.borrow_mut().push(format!("{} {}", c.kind, c.path));
                if c.path.ends_with(".md") {
                    HookReply { prompts: vec!["Check the links.".into()], ..HookReply::default() }
                } else {
                    HookReply { context: "- lint ok".into(), ..HookReply::default() }
                }
            },
            &mut |task| {
                tasks.borrow_mut().push(task.to_owned());
                // the agent edits the file it was started for: this must not start another run
                std::fs::write(root_for_agent.join("a.md"), "three, edited by the agent").unwrap();
                true
            },
            &mut |_| {},
        );
        writer.join().unwrap();
        assert_eq!(runs, 1, "{:?}", fired.borrow());
        assert_eq!(tasks.borrow().len(), 1);
        assert!(tasks.borrow()[0].contains("a.md was modified") && tasks.borrow()[0].contains("Check the links."));
        assert!(fired.borrow().iter().any(|f| f == "created new.txt"));
        assert_eq!(fired.borrow().iter().filter(|f| f.ends_with("a.md")).count(), 1);
        let _ = std::fs::remove_dir_all(root);
    }
}
