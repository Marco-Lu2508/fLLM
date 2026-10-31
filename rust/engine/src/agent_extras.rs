//! Helpers of the headless agent that sit around the loop: `stream-json` events, the latest session (`--continue`),
//! the self-review that runs before the agent is called done (`--review`) and the skill a finished run proposes
//! (`--propose-skills`). Everything here is pure or touches only files, so it is tested without a server.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Output of `--output-format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    StreamJson,
}

impl OutputFormat {
    pub fn parse(text: &str) -> Option<OutputFormat> {
        match text {
            "text" => Some(OutputFormat::Text),
            "json" => Some(OutputFormat::Json),
            "stream-json" => Some(OutputFormat::StreamJson),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// stream-json: one JSON object per line while the run goes on, in the shape of Claude Code's `--output-format stream-json`
// ---------------------------------------------------------------------------------------------------------------

pub fn init_event(session: &str, model: &str, cwd: &str, tools: &[String], mode: &str) -> Value {
    json!({"type": "system", "subtype": "init", "session_id": session, "model": model, "cwd": cwd, "tools": tools, "permissionMode": mode})
}

/// The events of one message the loop appended: an assistant message (text and `tool_use` blocks) or a tool result.
pub fn message_events(message: &Value, session: &str) -> Vec<Value> {
    match message["role"].as_str() {
        Some("assistant") => {
            let mut content = Vec::new();
            if let Some(text) = message["content"].as_str().filter(|t| !t.trim().is_empty()) {
                content.push(json!({"type": "text", "text": text}));
            }
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                let args = match &call["function"]["arguments"] {
                    Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
                    other => other.clone(),
                };
                content.push(json!({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": args}));
            }
            vec![json!({"type": "assistant", "message": {"role": "assistant", "content": content}, "session_id": session})]
        }
        Some("tool") => vec![json!({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}]},
            "session_id": session,
        })],
        _ => Vec::new(),
    }
}

pub fn result_event(answer: &str, status: &str, steps: usize, session: &str, duration_ms: u128, error: Option<&str>) -> Value {
    let is_error = error.is_some() || status == "out_of_steps";
    let subtype = if error.is_some() {
        "error_during_execution"
    } else if status == "out_of_steps" {
        "error_max_turns"
    } else {
        "success"
    };
    json!({"type": "result", "subtype": subtype, "is_error": is_error, "result": error.unwrap_or(answer), "num_turns": steps, "session_id": session, "duration_ms": duration_ms})
}

// ---------------------------------------------------------------------------------------------------------------
// sessions
// ---------------------------------------------------------------------------------------------------------------

/// The name of the most recently written session file in `agent-sessions` (for `--continue`).
pub fn latest_session(base: &Path) -> Option<String> {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(base.join("agent-sessions")).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()).map(str::to_owned) else { continue };
        if !crate::agent_cli::valid_session_name(&name) {
            continue;
        }
        let time = entry.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, _)| time > *t) {
            best = Some((time, name));
        }
    }
    best.map(|(_, n)| n)
}

/// The title and the searchable text of a conversation: what the user asked, what the agent answered and the names of the tools
/// it used. Tool output is left out (it is large and mostly noise for a search).
pub fn session_text(messages: &[Value]) -> (String, String) {
    let mut title = String::new();
    let mut text = String::new();
    for m in messages {
        let content = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(" "),
            _ => String::new(),
        };
        match m["role"].as_str() {
            Some("user") if !content.trim().is_empty() => {
                if title.is_empty() {
                    title = content.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect();
                }
                text.push_str(&format!("User: {}\n", content.trim()));
            }
            Some("assistant") => {
                let tools: Vec<&str> = m["tool_calls"].as_array().into_iter().flatten().filter_map(|c| c["function"]["name"].as_str()).collect();
                if !tools.is_empty() {
                    text.push_str(&format!("Tools: {}\n", tools.join(", ")));
                }
                if !content.trim().is_empty() {
                    text.push_str(&format!("Agent: {}\n", content.trim()));
                }
            }
            _ => {}
        }
    }
    (title, text)
}

/// Keeps the newest `keep` automatic `run-<time>` session files and deletes the rest. Named sessions are never touched.
pub fn prune_auto_sessions(base: &Path, keep: usize) {
    let Ok(read) = std::fs::read_dir(base.join("agent-sessions")) else { return };
    let mut runs: Vec<(std::time::SystemTime, PathBuf)> = read
        .flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with("run-") && n.ends_with(".json")))
        .map(|e| (e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), e.path()))
        .collect();
    runs.sort_by_key(|r| std::cmp::Reverse(r.0));
    for (_, path) in runs.into_iter().skip(keep) {
        let _ = std::fs::remove_file(path);
    }
}

/// A fresh session name for a run without `--session`, so every run can be found again with `--resume` and by the search.
pub fn fresh_session_name(now: i64) -> String {
    format!("run-{now}")
}

// ---------------------------------------------------------------------------------------------------------------
// pictures
// ---------------------------------------------------------------------------------------------------------------

/// A chat-completions image part (`image_url` with a data URL) from a picture file. Only png, jpeg, gif and webp, at most 10 MB.
pub fn image_part(path: &Path) -> Result<Value, String> {
    use base64::Engine as _;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return Err(format!("{}: only png, jpg, gif and webp pictures can be attached", path.display())),
    };
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.len() > 10 << 20 {
        return Err(format!("{}: the picture is larger than 10 MB", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))}}))
}

/// Messages for the session file: a picture is replaced by a note (the data would only bloat the file).
pub fn strip_images(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| match m["content"].as_array() {
            Some(parts) if parts.iter().any(|p| p["type"] == "image_url") => {
                let text: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
                let mut copy = m.clone();
                copy["content"] = json!(format!("{}\n[a picture was attached here]", text.join("\n")).trim().to_owned());
                copy
            }
            _ => m.clone(),
        })
        .collect()
}

/// The content of the user message: the text alone, or the text followed by the pictures.
pub fn user_content(text: &str, images: &[Value]) -> Value {
    if images.is_empty() {
        return json!(text);
    }
    let mut parts = vec![json!({"type": "text", "text": text})];
    parts.extend(images.iter().cloned());
    json!(parts)
}

// ---------------------------------------------------------------------------------------------------------------
// self-review
// ---------------------------------------------------------------------------------------------------------------

/// What the checks after an edit found.
#[derive(Debug, Default)]
pub struct ReviewReport {
    /// Output of the `review` tool (diff plus findings of the fixed rules); empty when the folder is no git repository.
    pub review: String,
    /// `(path, output)` of `diagnostics` for each changed file that reported problems.
    pub problems: Vec<(String, String)>,
    /// Exit status and output of the verification command, when one was given and allowed.
    pub verify: Option<(bool, String)>,
}

impl ReviewReport {
    /// Problems a machine can state (not the model's opinion): diagnostics errors, a failing verify command, a high-severity finding.
    pub fn objective_problems(&self) -> bool {
        !self.problems.is_empty() || self.verify.as_ref().is_some_and(|(ok, _)| !*ok) || self.review.contains("[high]")
    }
}

pub fn is_clean_diagnostic(output: &str) -> bool {
    let o = output.trim_start();
    o.starts_with("No problems found") || o.contains("is not available here") || o.contains("no automatic check is known") || o.starts_with("No ")
}

/// The message that goes back to the model. Round 1 is always sent (the model reads its own diff once more); later rounds only when
/// a check found something, so a clean run costs exactly one extra step.
pub fn review_feedback(report: &ReviewReport, round: usize) -> Option<String> {
    if round > 1 && !report.objective_problems() {
        return None;
    }
    let mut text = String::from("Self-review before you finish. Check your own work against the task now.\n");
    if !report.problems.is_empty() {
        text.push_str("\nDiagnostics of the files you changed:\n");
        for (path, out) in &report.problems {
            text.push_str(&format!("- {path}: {}\n", crate::agent_cli::clip(out.trim(), 1500)));
        }
    }
    if let Some((ok, out)) = &report.verify {
        text.push_str(&format!("\nVerification command {}:\n{}\n", if *ok { "passed" } else { "FAILED" }, crate::agent_cli::clip(out.trim(), 3000)));
    }
    if !report.review.trim().is_empty() {
        text.push_str("\nYour changes and the findings of a fixed rule set:\n");
        text.push_str(&crate::agent_cli::clip(report.review.trim(), 12_000));
        text.push('\n');
    }
    text.push_str(if report.objective_problems() {
        "\nFix every real problem above, run the check again, then give the final report. If a finding is a false alarm, say so in one line."
    } else {
        "\nNothing was flagged automatically. Re-read the diff: does it do exactly what the task asked, with no leftover debug code, no unrelated change and no missing case? Fix what is wrong, otherwise give the final report."
    });
    Some(text)
}

// ---------------------------------------------------------------------------------------------------------------
// skills that the agent proposes
// ---------------------------------------------------------------------------------------------------------------

pub const SKILL_PROMPT: &str = "The task is finished. If the way you solved it is a repeatable procedure that is worth saving for the next time (a workflow with several steps, project-specific commands, a trap you hit), write it as a skill file; otherwise answer exactly NONE. Format: first line `name: a-short-slug` (lowercase letters, digits, - and _), second line `description: one sentence on what it does`, third line `when: one sentence on when to use it`, then a blank line and the steps as a short Markdown list with exact commands. No secrets, no paths that only exist on this machine. At most 40 lines.";

/// Parses the model's skill text into `(name, SKILL.md)`; `None` for `NONE` or an unusable text.
pub fn parse_skill(text: &str) -> Option<(String, String)> {
    let text = text.trim().trim_start_matches("```markdown").trim_start_matches("```md").trim_start_matches("```").trim_end_matches("```").trim();
    if text.is_empty() || text.trim_end_matches('.').eq_ignore_ascii_case("none") {
        return None;
    }
    // two shapes: a `---` front matter block, or the key lines at the top followed by a blank line
    let (head, body) = if let Some(rest) = text.strip_prefix("---") {
        rest.split_once("\n---").map(|(h, b)| (h.to_owned(), b.trim().to_owned()))?
    } else {
        let split = text.find("\n\n")?;
        (text[..split].to_owned(), text[split..].trim().to_owned())
    };
    let key = |k: &str| head.lines().find_map(|l| l.trim().strip_prefix(k)).map(|v| v.trim().trim_matches(['"', '\'']).to_owned()).unwrap_or_default();
    let (name, description, when) = (key("name:"), key("description:"), key("when:"));
    let body = body.as_str();
    let name: String = name.to_ascii_lowercase().chars().map(|c| if c == ' ' { '-' } else { c }).collect();
    if !crate::agent_cli::valid_session_name(&name) || description.is_empty() || body.is_empty() || body.len() > 60_000 {
        return None;
    }
    let mut file = format!("---\nname: {name}\ndescription: {description}\n");
    if !when.is_empty() {
        file.push_str(&format!("when: {when}\n"));
    }
    file.push_str(&format!("---\n{body}\n"));
    Some((name, file))
}

/// Where proposals wait for approval.
pub fn proposals_dir(base: &Path) -> PathBuf {
    base.join("skills-proposed")
}

/// Saves a proposal (`approved` false) or a ready skill (`approved` true) as `<dir>/<name>/SKILL.md`; never overwrites a skill of the
/// user (a number is appended to a proposal name that exists).
pub fn save_skill(base: &Path, name: &str, file: &str, approved: bool) -> Result<PathBuf, String> {
    let root = if approved { base.join("skills") } else { proposals_dir(base) };
    let mut slug = name.to_owned();
    let mut n = 1;
    while root.join(&slug).exists() {
        if approved {
            return Err(format!("a skill named '{name}' already exists; it was not replaced"));
        }
        n += 1;
        slug = format!("{name}-{n}");
    }
    let dir = root.join(&slug);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("SKILL.md");
    std::fs::write(&path, file).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_events_follow_the_claude_code_shape() {
        let assistant = json!({"role": "assistant", "content": "looking", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}]});
        let events = message_events(&assistant, "s1");
        assert_eq!(events[0]["type"], "assistant");
        assert_eq!(events[0]["message"]["content"][0]["text"], "looking");
        assert_eq!(events[0]["message"]["content"][1]["type"], "tool_use");
        assert_eq!(events[0]["message"]["content"][1]["input"]["path"], "a");
        let result = message_events(&json!({"role": "tool", "tool_call_id": "c1", "content": "text"}), "s1");
        assert_eq!(result[0]["message"]["content"][0]["type"], "tool_result");
        assert_eq!(result[0]["message"]["content"][0]["tool_use_id"], "c1");
        assert!(message_events(&json!({"role": "user", "content": "x"}), "s").is_empty());
        assert_eq!(result_event("ok", "done", 3, "s", 5, None)["subtype"], "success");
        assert_eq!(result_event("ok", "out_of_steps", 3, "s", 5, None)["is_error"], true);
        assert_eq!(result_event("", "done", 0, "s", 5, Some("boom"))["result"], "boom");
        assert_eq!(OutputFormat::parse("stream-json"), Some(OutputFormat::StreamJson));
        assert_eq!(OutputFormat::parse("xml"), None);
    }

    #[test]
    fn the_latest_session_is_the_newest_valid_file() {
        let dir = std::env::temp_dir().join(format!("fllm-latest-{}", std::process::id()));
        let sessions = dir.join("agent-sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        assert_eq!(latest_session(&dir), None);
        std::fs::write(sessions.join("old.json"), "{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(sessions.join("new-one.json"), "{}").unwrap();
        std::fs::write(sessions.join("Bad Name.json"), "{}").unwrap();
        std::fs::write(sessions.join("note.txt"), "x").unwrap();
        assert_eq!(latest_session(&dir).as_deref(), Some("new-one"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_search_text_has_the_questions_the_answers_and_the_tool_names_but_no_tool_output() {
        let messages = vec![
            json!({"role": "user", "content": "fix   the lexer bug"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "HUGE FILE CONTENT"}),
            json!({"role": "assistant", "content": "Fixed it."}),
        ];
        let (title, text) = session_text(&messages);
        assert_eq!(title, "fix the lexer bug");
        assert!(text.contains("User: fix   the lexer bug") && text.contains("Tools: read_file") && text.contains("Agent: Fixed it."));
        assert!(!text.contains("HUGE"));
    }

    #[test]
    fn only_automatic_runs_are_pruned() {
        let dir = std::env::temp_dir().join(format!("fllm-prune-{}", std::process::id()));
        let sessions = dir.join("agent-sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        for n in 0..5 {
            std::fs::write(sessions.join(format!("run-{n}.json")), "{}").unwrap();
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        std::fs::write(sessions.join("nightly.json"), "{}").unwrap();
        prune_auto_sessions(&dir, 2);
        let mut left: Vec<String> = std::fs::read_dir(&sessions).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, vec!["nightly.json", "run-3.json", "run-4.json"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pictures_become_image_parts_and_other_files_are_refused() {
        let dir = std::env::temp_dir().join(format!("fllm-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.png"), [137, 80, 78, 71]).unwrap();
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        let part = image_part(&dir.join("a.png")).unwrap();
        assert_eq!(part["type"], "image_url");
        assert_eq!(part["image_url"]["url"], "data:image/png;base64,iVBORw==");
        assert!(image_part(&dir.join("notes.txt")).is_err() && image_part(&dir.join("missing.png")).is_err());
        assert_eq!(user_content("hi", &[]), json!("hi"));
        let both = user_content("look", &[part]);
        assert_eq!((both[0]["type"].as_str(), both[1]["type"].as_str()), (Some("text"), Some("image_url")));
        let stripped = strip_images(&[json!({"role": "user", "content": both}), json!({"role": "assistant", "content": "ok"})]);
        assert_eq!(stripped[0]["content"], "look\n[a picture was attached here]");
        assert_eq!(stripped[1]["content"], "ok");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_review_asks_once_when_clean_and_again_only_for_real_problems() {
        let clean = ReviewReport { review: "Review (diff): 1 file(s), +1/-0 lines, 0 finding(s)".into(), ..Default::default() };
        assert!(!clean.objective_problems());
        assert!(review_feedback(&clean, 1).unwrap().contains("Re-read the diff"));
        assert!(review_feedback(&clean, 2).is_none());
        let bad = ReviewReport { problems: vec![("a.py".into(), "Problems found (python): SyntaxError".into())], verify: Some((false, "1 failed".into())), ..Default::default() };
        assert!(bad.objective_problems());
        let text = review_feedback(&bad, 2).unwrap();
        assert!(text.contains("a.py") && text.contains("FAILED") && text.contains("Fix every real problem"));
        let high = ReviewReport { review: "- [high] possible secret".into(), ..Default::default() };
        assert!(high.objective_problems());
        assert!(is_clean_diagnostic("No problems found (node --check)."));
        assert!(!is_clean_diagnostic("Problems found (node --check, exit code 1):\nx"));
    }

    #[test]
    fn a_proposed_skill_is_parsed_checked_and_never_overwrites() {
        let good = "name: run-tests\ndescription: Run the test suite\nwhen: after a change\n\n1. `cargo test`\n2. read failures";
        let (name, file) = parse_skill(good).unwrap();
        assert_eq!(name, "run-tests");
        assert!(file.starts_with("---\nname: run-tests\ndescription: Run the test suite\nwhen: after a change\n---\n1. `cargo test`"));
        assert!(parse_skill("NONE").is_none());
        assert!(parse_skill("name: x\ndescription: d\n").is_none(), "a skill without steps is not saved");
        assert!(parse_skill("name: Bad/Name\ndescription: d\n\nsteps").is_none());
        let fenced = "```markdown\n---\nname: deploy\ndescription: Deploy it\n---\n- step one\n```";
        assert_eq!(parse_skill(fenced).unwrap().0, "deploy");
        let dir = std::env::temp_dir().join(format!("fllm-skill-{}", std::process::id()));
        let p1 = save_skill(&dir, "run-tests", &file, false).unwrap();
        let p2 = save_skill(&dir, "run-tests", &file, false).unwrap();
        assert_ne!(p1, p2);
        assert!(save_skill(&dir, "run-tests", &file, true).is_ok());
        assert!(save_skill(&dir, "run-tests", &file, true).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
