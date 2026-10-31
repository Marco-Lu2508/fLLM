//! Everyday commands around the agent: `fllm-rs doctor`, `fllm-rs update --check`, `fllm-rs insights` (also `usage`) and
//! `fllm-rs sessions list|search`. They read files of the Studio folder or ask the running server on this machine.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub const DOCTOR_HELP: &str = "usage: fllm-rs doctor [--url http://127.0.0.1:8000] [--api-key KEY] [--folder DIR]
Checks what the coding agent needs: the server and its model, the tool list, the command sandbox, git, the Studio folder, the
permission settings, skills and the terminal programs (bubblewrap, docker, ssh). Exit code 1 when a check fails.";

pub const INSIGHTS_HELP: &str = "usage: fllm-rs insights [--days N] [--profile NAME] [--json]
What the headless agent did in the last N days (default 30), from the sessions in the Studio folder: runs, steps, tool calls by
tool, edited files, folders and a rough token count (characters / 4, not the model's own count). `fllm-rs usage` is the same.";

pub const SESSIONS_HELP: &str = "usage: fllm-rs sessions list [--profile NAME]
       fllm-rs sessions search QUERY [--url URL] [--api-key KEY] [--limit N]
list shows the saved agent sessions (resume one with `fllm-rs agent --resume NAME`). search asks the running server for the
best matches over all stored chats and code sessions (the same index the agent tool `session_search` uses).";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn backend(args: &[String]) -> Result<crate::agent_cli::HttpBackend, String> {
    crate::agent_cli::HttpBackend::new(&arg(args, "--url").unwrap_or_else(|| "http://127.0.0.1:8000".into()), arg(args, "--api-key").or_else(|| std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty())))
}

fn program_exists(name: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    std::process::Command::new(probe).arg(name).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|s| s.success())
}

#[derive(Debug, PartialEq)]
pub struct Check {
    pub level: &'static str, // ok, warn, fail
    pub text: String,
}

fn check(level: &'static str, text: impl Into<String>) -> Check {
    Check { level, text: text.into() }
}

/// The checks that do not need a server.
pub fn local_checks(studio: &Path, folder: &Path) -> Vec<Check> {
    let mut out = Vec::new();
    out.push(check("ok", format!("fllm-rs {}", env!("CARGO_PKG_VERSION"))));
    match std::fs::create_dir_all(studio).and_then(|()| std::fs::write(studio.join(".doctor-write-test"), "x")) {
        Ok(()) => {
            let _ = std::fs::remove_file(studio.join(".doctor-write-test"));
            out.push(check("ok", format!("Studio folder is writable: {}", studio.display())));
        }
        Err(e) => out.push(check("fail", format!("Studio folder {} is not writable: {e}", studio.display()))),
    }
    out.push(if program_exists("git") { check("ok", "git is installed") } else { check("warn", "git was not found: /review, worktrees and the review tool need it") });
    match crate::agent_rules::load_hierarchy(studio, folder) {
        Ok(rules) => out.push(check("ok", format!("permission settings: {} allow, {} deny, {} ask rule(s){}", rules.allow.len(), rules.deny.len(), rules.ask.len(), if rules.mode.as_deref() == Some("plan") { ", default mode plan" } else { "" }))),
        Err(e) => out.push(check("fail", e)),
    }
    let skills = crate::skills_cli::find_skills(&studio.join("skills")).len() + crate::skills_cli::find_skills(&folder.join(".fllm").join("skills")).len();
    let proposed = crate::skills_cli::find_skills(&crate::agent_extras::proposals_dir(studio)).len();
    out.push(check("ok", format!("{skills} skill(s){}", if proposed > 0 { format!(", {proposed} proposed (fllm-rs skills proposed)") } else { String::new() })));
    let sessions = std::fs::read_dir(studio.join("agent-sessions")).map_or(0, |r| r.flatten().count());
    out.push(check("ok", format!("{sessions} saved agent session(s)")));
    for (name, why) in [("bwrap", "command sandbox on Linux"), ("docker", "terminal backend docker"), ("ssh", "terminal backend ssh")] {
        if program_exists(name) {
            out.push(check("ok", format!("{name} found ({why})")));
        } else if cfg!(not(windows)) || name != "bwrap" {
            out.push(check("warn", format!("{name} not found ({why} unavailable)")));
        }
    }
    out
}

pub fn doctor(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{DOCTOR_HELP}");
        return Ok(0);
    }
    let folder = arg(args, "--folder").map(std::path::PathBuf::from).map_or_else(std::env::current_dir, Ok).map_err(|e| e.to_string())?;
    let studio = crate::agent_cli::studio_folder();
    let mut checks = local_checks(&studio, &folder);
    match backend(args) {
        Err(e) => checks.push(check("fail", e)),
        Ok(mut http) => {
            match http.model_info() {
                Ok((id, ctx)) => {
                    checks.push(check("ok", format!("server reachable, model {id}{}", ctx.map_or(String::new(), |c| format!(", context {c}")))));
                    match crate::agent_cli::Backend::tools(&mut http) {
                        Ok(tools) => {
                            let chars: usize = tools.iter().map(|t| t.to_string().len()).sum();
                            match crate::agent_cli::context_warning(chars, ctx) {
                                Some(w) => checks.push(check("fail", w)),
                                None => checks.push(check("ok", format!("{} agent tools, about {} tokens of schema", tools.len(), chars as u64 * 2 / 7))),
                            }
                        }
                        Err(e) => checks.push(check("fail", format!("the tool list is not available: {e}"))),
                    }
                }
                Err(e) => checks.push(check("fail", format!("{e} (start one with `fllm-rs serve --model ...` or the desktop app)"))),
            }
            if let Ok(s) = http.get_json("/api/studio/agent/sandbox") {
                let active = s["active"].as_bool().unwrap_or(false);
                let mode = s["mode"].as_str().unwrap_or("?");
                checks.push(if active { check("ok", format!("command sandbox active (mode {mode})")) } else { check("warn", format!("command sandbox not active (mode {mode}): {}", s["sandbox"]["reason"].as_str().unwrap_or("unavailable"))) });
            }
            if let Ok(s) = http.get_json("/api/studio/update") {
                if s["newer"].as_bool().unwrap_or(false) {
                    checks.push(check("warn", format!("a newer release exists: {}", s["latest"].as_str().unwrap_or("?"))));
                }
            }
        }
    }
    let mut failed = false;
    for c in &checks {
        println!("[{:<4}] {}", c.level, c.text);
        failed |= c.level == "fail";
    }
    Ok(i32::from(failed))
}

// ---------------------------------------------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------------------------------------------

fn version_parts(version: &str) -> Vec<u64> {
    version.trim_start_matches('v').split(['.', '-']).map_while(|p| p.parse().ok()).collect()
}

/// `(newest tag, is it newer than current)` from the list of releases GitHub returned (drafts and prereleases are ignored).
pub fn newest_release(releases: &[Value], current: &str) -> Option<(String, bool, String)> {
    let best = releases
        .iter()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false) && !r["prerelease"].as_bool().unwrap_or(false))
        .max_by_key(|r| version_parts(r["tag_name"].as_str().unwrap_or("")))?;
    let tag = best["tag_name"].as_str()?.to_owned();
    let newer = version_parts(&tag) > version_parts(current);
    Some((tag, newer, best["html_url"].as_str().unwrap_or("").to_owned()))
}

pub fn update(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("usage: fllm-rs update [--check]\nAsks GitHub for the newest release and says whether it is newer than this build. It never downloads or replaces anything: the\ndesktop app installs updates (Settings > About), or download the release from the page shown.");
        return Ok(0);
    }
    let client = reqwest::blocking::Client::builder().connect_timeout(std::time::Duration::from_secs(15)).timeout(std::time::Duration::from_secs(30)).user_agent("fllm-update-check/0.1").build().map_err(|e| e.to_string())?;
    let releases: Vec<Value> = client
        .get("https://api.github.com/repos/Marco-Lu2508/fLLM/releases?per_page=10")
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("cannot ask GitHub for updates: {e}"))?
        .json()
        .map_err(|e| e.to_string())?;
    let current = env!("CARGO_PKG_VERSION");
    match newest_release(&releases, current) {
        None => println!("fllm-rs {current}: no release found"),
        Some((tag, true, page)) => {
            println!("fllm-rs {current}: a newer release exists: {tag}\n{page}");
            return Ok(10);
        }
        Some((tag, false, _)) => println!("fllm-rs {current} is up to date (newest release {tag})"),
    }
    Ok(0)
}

// ---------------------------------------------------------------------------------------------------------------
// insights
// ---------------------------------------------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
pub struct Insights {
    pub sessions: usize,
    pub steps: usize,
    pub user_messages: usize,
    pub tool_calls: BTreeMap<String, usize>,
    pub folders: BTreeMap<String, usize>,
    pub edited_files: usize,
    pub approx_tokens: usize,
}

/// Adds one saved session (`{"messages": [...], "folder": ..., "edited": [...]}`) to the totals.
pub fn add_session(ins: &mut Insights, session: &Value) {
    ins.sessions += 1;
    if let Some(folder) = session["folder"].as_str() {
        *ins.folders.entry(folder.to_owned()).or_default() += 1;
    }
    ins.edited_files += session["edited"].as_array().map_or(0, Vec::len);
    for m in session["messages"].as_array().into_iter().flatten() {
        let content = match &m["content"] {
            Value::String(s) => s.chars().count(),
            other => other.to_string().len(),
        };
        ins.approx_tokens += content / 4 + m["tool_calls"].to_string().len() / 4;
        match m["role"].as_str() {
            Some("user") => ins.user_messages += 1,
            Some("assistant") => {
                ins.steps += 1;
                for c in m["tool_calls"].as_array().into_iter().flatten() {
                    *ins.tool_calls.entry(c["function"]["name"].as_str().unwrap_or("?").to_owned()).or_default() += 1;
                }
            }
            _ => {}
        }
    }
}

pub fn collect_insights(base: &Path, days: u64) -> Insights {
    let mut ins = Insights::default();
    let cutoff = std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(days * 86_400));
    for entry in std::fs::read_dir(base.join("agent-sessions")).into_iter().flatten().flatten() {
        if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let modified = entry.metadata().and_then(|m| m.modified()).ok();
        if let (Some(cut), Some(m)) = (cutoff, modified) {
            if m < cut {
                continue;
            }
        }
        if let Some(v) = std::fs::read_to_string(entry.path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
            add_session(&mut ins, &v);
        }
    }
    ins
}

pub fn insights(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{INSIGHTS_HELP}");
        return Ok(0);
    }
    let days: u64 = arg(args, "--days").map_or(Ok(30), |d| d.parse().map_err(|_| "--days is a number".to_owned()))?;
    let base = crate::agent_cli::profile_base(arg(args, "--profile").as_deref())?;
    let ins = collect_insights(&base, days);
    if args.iter().any(|a| a == "--json") {
        println!("{}", json!({"days": days, "sessions": ins.sessions, "steps": ins.steps, "user_messages": ins.user_messages, "tool_calls": ins.tool_calls, "folders": ins.folders, "edited_files": ins.edited_files, "approx_tokens": ins.approx_tokens}));
        return Ok(0);
    }
    println!("Last {days} days: {} session(s), {} user message(s), {} agent step(s), {} edited file(s), about {} tokens (characters / 4).", ins.sessions, ins.user_messages, ins.steps, ins.edited_files, ins.approx_tokens);
    let mut tools: Vec<_> = ins.tool_calls.iter().collect();
    tools.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    if !tools.is_empty() {
        println!("Tool calls:");
        for (name, n) in tools.into_iter().take(12) {
            println!("  {n:>5}  {name}");
        }
    }
    let mut folders: Vec<_> = ins.folders.iter().collect();
    folders.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    if !folders.is_empty() {
        println!("Folders:");
        for (name, n) in folders.into_iter().take(8) {
            println!("  {n:>5}  {name}");
        }
    }
    if ins.sessions == 0 {
        println!("(No sessions yet: every `fllm-rs agent` run is saved, see `fllm-rs sessions list`.)");
    }
    Ok(0)
}

// ---------------------------------------------------------------------------------------------------------------
// sessions
// ---------------------------------------------------------------------------------------------------------------

pub fn sessions(args: &[String]) -> Result<i32, String> {
    match args.first().map(String::as_str) {
        Some("list") => {
            let base = crate::agent_cli::profile_base(arg(args, "--profile").as_deref())?;
            let mut rows: Vec<(std::time::SystemTime, String, usize, String)> = Vec::new();
            for e in std::fs::read_dir(base.join("agent-sessions")).into_iter().flatten().flatten() {
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let Some(v) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else { continue };
                let (title, _) = crate::agent_extras::session_text(v["messages"].as_array().map_or(&[][..], Vec::as_slice));
                rows.push((e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_owned(), v["messages"].as_array().map_or(0, Vec::len), title));
            }
            rows.sort_by_key(|r| std::cmp::Reverse(r.0));
            if rows.is_empty() {
                println!("no saved sessions");
            }
            for (_, name, n, title) in rows.into_iter().take(50) {
                println!("{name:<28} {n:>4} messages  {title}");
            }
            Ok(0)
        }
        Some("search") => {
            let query: Vec<&str> = args[1..].iter().take_while(|a| !a.starts_with("--")).map(String::as_str).collect();
            if query.is_empty() {
                return Err("give the words to search for".into());
            }
            let limit = arg(args, "--limit").unwrap_or_else(|| "10".into());
            let http = backend(args)?;
            let q: String = query.join(" ").bytes().map(|b| if b.is_ascii_alphanumeric() { (b as char).to_string() } else { format!("%{b:02X}") }).collect();
            let answer = http.get_json(&format!("/api/studio/sessions/search?q={q}&limit={limit}"))?;
            let hits = answer["hits"].as_array().cloned().unwrap_or_default();
            if hits.is_empty() {
                println!("no stored session matches");
            }
            for h in hits {
                println!("{} [{}] {}\n    {}", h["id"].as_str().unwrap_or("?"), h["kind"].as_str().unwrap_or("?"), h["title"].as_str().unwrap_or(""), h["snippet"].as_str().unwrap_or(""));
            }
            Ok(0)
        }
        _ => {
            println!("{SESSIONS_HELP}");
            Ok(i32::from(!args.is_empty() && args[0] != "--help" && args[0] != "-h"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_are_compared_by_version_and_prereleases_are_ignored() {
        let releases = vec![
            json!({"tag_name": "v0.1.9", "html_url": "u1"}),
            json!({"tag_name": "v0.2.0-rc1", "prerelease": true, "html_url": "u2"}),
            json!({"tag_name": "v0.1.10", "html_url": "u3"}),
            json!({"tag_name": "v9.0.0", "draft": true, "html_url": "u4"}),
        ];
        assert_eq!(newest_release(&releases, "0.1.9"), Some(("v0.1.10".into(), true, "u3".into())));
        assert_eq!(newest_release(&releases, "0.1.10").map(|r| r.1), Some(false));
        assert!(newest_release(&[], "1.0.0").is_none());
    }

    #[test]
    fn insights_count_tools_steps_and_folders() {
        let session = json!({"folder": "/work/a", "edited": ["x.rs", "y.rs"], "messages": [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}, {"function": {"name": "edit_file"}}]},
            {"role": "tool", "content": "text"},
            {"role": "assistant", "content": "done"},
        ]});
        let mut ins = Insights::default();
        add_session(&mut ins, &session);
        add_session(&mut ins, &json!({"messages": [{"role": "user", "content": "x"}]}));
        assert_eq!((ins.sessions, ins.user_messages, ins.steps, ins.edited_files), (2, 2, 2, 2));
        assert_eq!(ins.tool_calls.get("read_file"), Some(&1));
        assert_eq!(ins.folders.get("/work/a"), Some(&1));
    }

    #[test]
    fn insights_read_recent_session_files() {
        let dir = std::env::temp_dir().join(format!("fllm-insights-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("agent-sessions")).unwrap();
        std::fs::write(dir.join("agent-sessions/run-1.json"), r#"{"messages":[{"role":"user","content":"hello"}]}"#).unwrap();
        std::fs::write(dir.join("agent-sessions/bad.json"), "{oops").unwrap();
        assert_eq!(collect_insights(&dir, 30).sessions, 1);
        assert_eq!(collect_insights(&dir, 0).sessions, 0, "nothing is newer than now");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_local_checks_report_a_broken_permission_file_as_a_failure() {
        let dir = std::env::temp_dir().join(format!("fllm-doctor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("proj/.fllm")).unwrap();
        let ok = local_checks(&dir.join("studio"), &dir.join("proj"));
        assert!(ok.iter().all(|c| c.level != "fail"), "{ok:?}");
        std::fs::write(dir.join("proj/.fllm/permissions.json"), "{no").unwrap();
        let bad = local_checks(&dir.join("studio"), &dir.join("proj"));
        assert!(bad.iter().any(|c| c.level == "fail" && c.text.contains("not valid JSON")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
