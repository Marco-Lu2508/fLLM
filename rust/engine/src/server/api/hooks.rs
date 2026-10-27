//! Hooks: shell commands the user attaches to fixed points of the agent loop.
//!
//! A hook is a command the server runs when the agent reaches an event, the way Claude Code and Codex do it: before and
//! after every tool call, after a file was changed, when a workspace session starts and when it ends. That is the
//! deterministic half of the loop a prompt cannot give: a formatter that must run after every edit, a linter that must
//! block a bad write, a small script that records what the model did.
//!
//! The list lives in `hooks.json` next to `studio.json`:
//!
//! ```json
//! {"hooks": {"on_file_edit": [{"matcher": "edit_file,write_file", "command": "cargo fmt", "timeout_seconds": 20}]}}
//! ```
//!
//! A hook gets one JSON object on stdin (`event`, `tool`, `arguments`, `workspace`) and decides with its exit code:
//! `0` passes (its stdout is added to the tool result as notes), `2` blocks the tool call (its stderr is the reason),
//! anything else is a failure that is reported but does not stop the agent. A hook can also print
//! `{"decision": "block", "reason": "..."}` or `{"context": "..."}` instead of using the exit code.
use super::*;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// The events a hook can listen to, in the order the page lists them.
pub(super) const EVENTS: [&str; 5] = ["pre_tool_call", "post_tool_call", "on_file_edit", "on_session_start", "on_session_end"];
/// Tool names that change files: their successful call fires `on_file_edit`.
const EDIT_TOOLS: [&str; 4] = ["write_file", "edit_file", "multi_edit", "patch"];

const MAX_HOOKS: usize = 64;
const MAX_COMMAND_CHARS: usize = 2_000;
const MAX_OUTPUT_CHARS: usize = 4_000;
const MAX_CONTEXT_CHARS: usize = 8_000;
const MAX_TOOL_OUTPUT_CHARS: usize = 20_000;

pub(super) fn file() -> PathBuf {
    crate::hub::studio_data_file("hooks.json")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Hook {
    pub event: String,
    pub matcher: String,
    pub command: String,
    pub timeout_seconds: u64,
    pub enabled: bool,
    /// Set when the hook comes from a plugin: its folder, handed to the command as `FLLM_PLUGIN_DIR`.
    pub plugin: Option<String>,
}

/// What the hooks of one event did: whether the call must stop, why, and the notes to add to its result.
#[derive(Default, Debug)]
pub(super) struct Outcome {
    pub blocked: bool,
    pub reason: String,
    pub context: String,
    pub ran: Vec<Value>,
}

impl Outcome {
    pub(super) fn notes(&self) -> String {
        self.context.trim().to_owned()
    }
}

/// Reads `hooks.json` and the hooks of every installed plugin. A missing or broken file means no hooks (the agent never
/// fails because of it).
pub(super) fn load() -> Vec<Hook> {
    let mut hooks = Vec::new();
    if let Ok(text) = fs::read_to_string(file()) {
        if let Ok(value) = serde_json::from_str::<Value>(&text) {
            hooks = parse(&value, None);
        }
    }
    for (plugin, value) in super::plugins::hook_files() {
        hooks.extend(parse(&value, Some(&plugin)));
        if hooks.len() >= MAX_HOOKS {
            break;
        }
    }
    hooks.truncate(MAX_HOOKS);
    hooks
}

/// The hooks of one file (or one plugin file) for every event.
fn parse(value: &Value, plugin: Option<&str>) -> Vec<Hook> {
    let list = match value.get("hooks") {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::Array(items)) => {
            // flat form: [{"event": "...", ...}]
            let mut map = serde_json::Map::new();
            for item in items {
                let Some(event) = item.get("event").and_then(Value::as_str) else { continue };
                let entry = map.entry(event.to_owned()).or_insert_with(|| Value::Array(Vec::new()));
                if let Some(array) = entry.as_array_mut() {
                    array.push(item.clone());
                }
            }
            map
        }
        _ => return Vec::new(),
    };
    let mut hooks = Vec::new();
    for event in EVENTS {
        let Some(Value::Array(items)) = list.get(event) else { continue };
        for item in items {
            let command = item.get("command").and_then(Value::as_str).unwrap_or("").trim().to_owned();
            if command.is_empty() {
                continue;
            }
            hooks.push(Hook {
                event: event.to_owned(),
                matcher: item.get("matcher").and_then(Value::as_str).unwrap_or("").trim().to_owned(),
                command: command.chars().take(MAX_COMMAND_CHARS).collect(),
                timeout_seconds: item.get("timeout_seconds").and_then(Value::as_u64).unwrap_or(30).clamp(1, 300),
                enabled: item.get("enabled").and_then(Value::as_bool).unwrap_or(true),
                plugin: plugin.map(str::to_owned),
            });
        }
    }
    hooks
}

/// The file as the page shows it: every event with its hooks, plus the path. Hooks of plugins are listed separately:
/// they are read-only and belong to the plugin.
pub(super) fn list_json() -> Value {
    let hooks = load();
    let mut events = serde_json::Map::new();
    let mut from_plugins = Vec::new();
    for event in EVENTS {
        let mut items: Vec<Value> = Vec::new();
        for h in hooks.iter().filter(|h| h.event == event) {
            let entry = json!({
                "event": h.event,
                "matcher": h.matcher,
                "command": h.command,
                "timeout_seconds": h.timeout_seconds,
                "enabled": h.enabled,
            });
            match &h.plugin {
                Some(plugin) => {
                    let mut entry = entry;
                    entry["plugin"] = json!(plugin);
                    from_plugins.push(entry);
                }
                None => items.push(entry),
            }
        }
        events.insert(event.to_owned(), Value::Array(items));
    }
    json!({
        "file": file().display().to_string(),
        "events": EVENTS,
        "hooks": Value::Object(events),
        "from_plugins": from_plugins,
    })
}

/// Replaces the whole list. Unknown events, empty commands and absurd timeouts are refused with a message.
pub(super) fn save(body: &Value) -> Result<Value, String> {
    let source = body.get("hooks").cloned().unwrap_or_else(|| body.clone());
    let map = match source {
        Value::Object(map) => map,
        Value::Array(_) => {
            // flat form -> group by event
            let mut grouped = serde_json::Map::new();
            for item in source.as_array().cloned().unwrap_or_default() {
                let event = item.get("event").and_then(Value::as_str).unwrap_or("").to_owned();
                if !EVENTS.contains(&event.as_str()) {
                    return Err(format!("unknown event '{event}'; known: {}", EVENTS.join(", ")));
                }
                let entry = grouped.entry(event).or_insert_with(|| Value::Array(Vec::new()));
                if let Some(array) = entry.as_array_mut() {
                    array.push(item);
                }
            }
            grouped
        }
        _ => return Err("hooks must be an object of events".to_owned()),
    };
    let mut out = serde_json::Map::new();
    let mut total = 0usize;
    for (event, items) in &map {
        if !EVENTS.contains(&event.as_str()) {
            return Err(format!("unknown event '{event}'; known: {}", EVENTS.join(", ")));
        }
        let Some(items) = items.as_array() else { return Err(format!("'{event}' must be a list")) };
        let mut cleaned = Vec::new();
        for item in items {
            let command = item.get("command").and_then(Value::as_str).unwrap_or("").trim();
            if command.is_empty() {
                continue;
            }
            if command.chars().count() > MAX_COMMAND_CHARS {
                return Err(format!("a command is longer than {MAX_COMMAND_CHARS} characters"));
            }
            total += 1;
            if total > MAX_HOOKS {
                return Err(format!("at most {MAX_HOOKS} hooks are allowed"));
            }
            cleaned.push(json!({
                "matcher": item.get("matcher").and_then(Value::as_str).unwrap_or("").trim(),
                "command": command,
                "timeout_seconds": item.get("timeout_seconds").and_then(Value::as_u64).unwrap_or(30).clamp(1, 300),
                "enabled": item.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            }));
        }
        if !cleaned.is_empty() {
            out.insert(event.clone(), Value::Array(cleaned));
        }
    }
    let text = serde_json::to_string_pretty(&json!({ "hooks": Value::Object(out) })).map_err(|e| e.to_string())?;
    if let Some(dir) = file().parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    fs::write(file(), text + "\n").map_err(|e| format!("cannot write {}: {e}", file().display()))?;
    Ok(list_json())
}

/// Does the matcher of this hook cover the tool? Empty and `*` mean every tool; `bash,edit_file` lists names, and
/// `edit*` matches by prefix.
fn matches(hook: &Hook, tool: &str) -> bool {
    let matcher = hook.matcher.trim();
    if matcher.is_empty() || matcher == "*" {
        return true;
    }
    matcher
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .any(|part| match part.strip_suffix('*') {
            Some(prefix) => tool.starts_with(prefix),
            None => part == tool,
        })
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push_str("\n[... hook output was cut]");
    out
}

/// Runs `command` with the payload on stdin and returns (exit code, stdout, stderr, milliseconds).
fn run_command(command: &str, payload: &Value, dir: Option<&Path>, timeout: Duration, plugin_dir: Option<&Path>) -> (i32, String, String, u64) {
    let started = Instant::now();
    let mut builder = if cfg!(windows) {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(command);
        cmd
    } else {
        // the shell is the user's own: a hook is a line they wrote, not a program name
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned());
        let mut cmd = Command::new(shell);
        cmd.arg("-c").arg(command);
        cmd
    };
    builder
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("FLLM_HOOK", "1");
    if let Some(plugin) = plugin_dir {
        // a hook of a plugin can find its own files (scripts, templates) with this path
        builder.env("FLLM_PLUGIN_DIR", plugin);
    }
    if let Some(dir) = dir {
        builder.current_dir(dir);
    }
    let mut child = match builder.spawn() {
        Ok(child) => child,
        Err(e) => return (126, String::new(), format!("cannot start the hook: {e}"), 0),
    };
    if let Some(mut stdin) = child.stdin.take() {
        let text = payload.to_string();
        // a hook that does not read its stdin must not make us wait: the write is best effort
        let _ = stdin.write_all(text.as_bytes());
    }
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(40));
            }
            Err(_) => break None,
        }
    };
    let output = child.wait_with_output().unwrap_or_else(|_| std::process::Output {
        status: std::process::ExitStatus::default(),
        stdout: Vec::new(),
        stderr: Vec::new(),
    });
    let millis = started.elapsed().as_millis() as u64;
    let code = match status {
        Some(s) => s.code().unwrap_or(-1),
        None => 124, // timed out
    };
    (
        code,
        String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        millis,
    )
}

fn payload(event: &str, tool: &str, args: &Value, dir: Option<&Path>) -> Value {
    json!({
        "event": event,
        "tool": tool,
        "arguments": args,
        "workspace": dir.map(|d| d.display().to_string()),
    })
}

/// Runs every enabled hook of `event` that covers `tool`.
pub(super) fn dispatch(event: &str, tool: &str, args: &Value, dir: Option<&Path>) -> Outcome {
    let mut outcome = Outcome::default();
    let hooks = load();
    for hook in hooks.iter().filter(|h| h.enabled && h.event == event && matches(h, tool)) {
        let body = payload(event, tool, args, dir);
        let plugin_dir = hook.plugin.as_deref().and_then(super::plugins::folder);
        let (code, stdout, stderr, millis) =
            run_command(&hook.command, &body, dir, Duration::from_secs(hook.timeout_seconds), plugin_dir.as_deref());
        // a hook may answer with JSON instead of an exit code
        let answer = serde_json::from_str::<Value>(&stdout).ok();
        let decision = answer.as_ref().and_then(|v| v.get("decision")).and_then(Value::as_str).unwrap_or("");
        let json_reason = answer.as_ref().and_then(|v| v.get("reason")).and_then(Value::as_str).unwrap_or("");
        let json_context = answer.as_ref().and_then(|v| v.get("context")).and_then(Value::as_str).unwrap_or("");

        let mut blocked_here = false;
        let mut reason = String::new();
        if decision == "block" {
            blocked_here = true;
            reason = if json_reason.is_empty() { "a hook blocked this call".to_owned() } else { json_reason.to_owned() };
        } else if code == 2 {
            blocked_here = true;
            reason = if stderr.is_empty() { "a hook blocked this call".to_owned() } else { stderr.clone() };
        } else if code == 124 {
            outcome.context.push_str(&format!("- hook `{}` timed out after {} s\n", hook.command, hook.timeout_seconds));
        } else if code != 0 {
            let detail = if stderr.is_empty() { stdout.clone() } else { stderr.clone() };
            outcome.context.push_str(&format!("- hook `{}` failed (exit {code}): {}\n", hook.command, clip(&detail, MAX_OUTPUT_CHARS)));
        } else if !json_context.is_empty() {
            outcome.context.push_str(&format!("- {}\n", clip(json_context, MAX_OUTPUT_CHARS)));
        } else if !stdout.is_empty() && answer.is_none() {
            outcome.context.push_str(&format!("- hook `{}`: {}\n", hook.command, clip(&stdout, MAX_OUTPUT_CHARS)));
        } else if decision == "allow" && !json_context.is_empty() {
            outcome.context.push_str(&format!("- {}\n", clip(json_context, MAX_OUTPUT_CHARS)));
        }

        outcome.ran.push(json!({
            "command": hook.command,
            "exit_code": code,
            "blocked": blocked_here,
            "milliseconds": millis,
            "stdout": clip(&stdout, 1_000),
            "stderr": clip(&stderr, 1_000),
        }));

        if blocked_here {
            outcome.blocked = true;
            if outcome.reason.is_empty() {
                outcome.reason = reason;
            }
            break; // a blocked call does not run the remaining hooks of this event
        }
        if outcome.context.chars().count() > MAX_CONTEXT_CHARS {
            outcome.context = clip(&outcome.context, MAX_CONTEXT_CHARS);
            break;
        }
    }
    outcome
}

pub(super) fn is_edit_tool(name: &str) -> bool {
    EDIT_TOOLS.contains(&name)
}

/// Adds the notes of a hook outcome to a tool result (a blocked call is handled by the caller).
pub(super) fn with_notes(result: Result<String, String>, outcome: &Outcome) -> Result<String, String> {
    let notes = outcome.notes();
    if notes.is_empty() {
        return result;
    }
    match result {
        Ok(mut text) => {
            if text.chars().count() > MAX_TOOL_OUTPUT_CHARS {
                text = clip(&text, MAX_TOOL_OUTPUT_CHARS);
            }
            text.push_str("\n\n[hooks]\n");
            text.push_str(&notes);
            Ok(text)
        }
        Err(mut text) => {
            text.push_str("\n\n[hooks]\n");
            text.push_str(&notes);
            Err(text)
        }
    }
}

/// `GET /api/studio/hooks`: the hook list from `hooks.json`.
pub(in crate::server) async fn list() -> Response {
    Json(list_json()).into_response()
}

/// `POST /api/studio/hooks`: replaces the list. The body is `{"hooks": {...}}` or `{"hooks": [{"event": ...}]}`.
pub(in crate::server) async fn save_handler(Json(body): Json<Value>) -> Response {
    match save(&body) {
        Ok(saved) => Json(saved).into_response(),
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}

/// `POST /api/studio/hooks/test`: `{event, tool, arguments}` runs the matching hooks now and reports what they did.
pub(in crate::server) async fn test(Json(body): Json<Value>) -> Response {
    let event = body.get("event").and_then(Value::as_str).unwrap_or("pre_tool_call");
    if !EVENTS.contains(&event) {
        return api_error(StatusCode::BAD_REQUEST, &format!("unknown event '{event}'"));
    }
    let tool = body.get("tool").and_then(Value::as_str).unwrap_or("");
    let args = body.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let dir = body
        .get("workspace")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .filter(|p| p.is_dir());
    let outcome = dispatch(event, tool, &args, dir.as_deref());
    Json(json!({
        "blocked": outcome.blocked,
        "reason": outcome.reason,
        "context": outcome.context,
        "ran": outcome.ran,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hook(matcher: &str) -> Hook {
        Hook {
            event: "pre_tool_call".to_owned(),
            matcher: matcher.to_owned(),
            command: "true".to_owned(),
            timeout_seconds: 5,
            enabled: true,
            plugin: None,
        }
    }

    #[test]
    fn an_empty_matcher_and_a_star_cover_every_tool() {
        assert!(matches(&hook(""), "bash"));
        assert!(matches(&hook("*"), "edit_file"));
    }

    #[test]
    fn matchers_are_names_lists_or_prefixes() {
        assert!(matches(&hook("bash"), "bash"));
        assert!(!matches(&hook("bash"), "bash_output"));
        assert!(matches(&hook("edit*,multi_edit"), "edit_file"));
        assert!(matches(&hook("edit_file, write_file"), "write_file"));
        assert!(!matches(&hook("edit_file"), "read_file"));
    }

    #[test]
    fn the_edit_tools_are_the_ones_that_write() {
        assert!(is_edit_tool("edit_file"));
        assert!(is_edit_tool("patch"));
        assert!(!is_edit_tool("read_file"));
    }

    #[test]
    fn notes_are_added_to_a_result_and_to_an_error() {
        let outcome = Outcome { context: "- cargo fmt ran\n".to_owned(), ..Outcome::default() };
        assert_eq!(with_notes(Ok("done".to_owned()), &outcome).unwrap(), "done\n\n[hooks]\n- cargo fmt ran");
        assert!(with_notes(Err("boom".to_owned()), &outcome).unwrap_err().contains("[hooks]"));
        let empty = Outcome::default();
        assert_eq!(with_notes(Ok("done".to_owned()), &empty).unwrap(), "done");
    }
}
