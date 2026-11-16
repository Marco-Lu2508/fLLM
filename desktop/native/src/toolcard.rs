//! Rows 83 and 84 of PARITY.md: the words and shapes of a tool card of the Code page — what
//! `TOOL_LABEL` and `toolLabel`, `diffRows`, `outputHtml` and `defaultOpen` of `code.js` make of a
//! call. The page names the diagnostics tool *Check code*, its notebook edits are edits like any
//! other, and a card is open or closed by the same rule for all of them.
//!
//! `tests/toolcard_fixtures.rs` runs the page's own functions in a vm (`tests/fixtures/
//! toolcard_fixtures.js`) and compares the port with them case by case.

use crate::computer::{js_string, slice_units};
use crate::jobs::truthy;
use serde_json::Value;

fn or(v: Option<&Value>, fallback: &str) -> String {
    if truthy(v) { js_string(v.unwrap_or(&Value::Null)) } else { fallback.to_string() }
}

fn truthy_member(args: &Value, k: &str) -> bool {
    truthy(args.get(k))
}

/// `Object.values(a)[0] ?? ''`.
fn first_value(args: &Value) -> String {
    match args.as_object().and_then(|o| o.values().next()) {
        None | Some(Value::Null) => String::new(),
        Some(v) => js_string(v),
    }
}

/// `toolLabel(item)`: the verb and the target of a card. `mcp` is the server and the tool of an
/// MCP call (its card is named by them).
pub fn label(name: &str, args: &Value, mcp: Option<(&str, &str)>) -> (String, String) {
    let empty = Value::Object(Default::default());
    let args = if truthy(Some(args)) { args } else { &empty };
    let a = |k: &str| args.get(k);
    let t = |s: &str| s.to_string();
    match name {
        "list_dir" => (t("List folder"), or(a("path"), ".")),
        "read_file" => (t("Read file"), format!("{}{}", or(a("path"), ""), if truthy_member(args, "offset") { format!(" from line {}", or(a("offset"), "")) } else { String::new() })),
        "write_file" => (t("Write file"), or(a("path"), "")),
        "edit_file" => (t("Edit file"), or(a("path"), "")),
        "multi_edit" => (t("Edit file"), format!("{}{}", or(a("path"), ""), match a("edits") { Some(Value::Array(e)) => format!("  ({} changes)", e.len()), _ => String::new() })),
        "glob" => (t("Find files"), or(a("pattern"), "")),
        "grep" => (t("Search text"), format!("{}{}", or(a("pattern"), ""), if truthy_member(args, "glob") { format!("  in {}", or(a("glob"), "")) } else { String::new() })),
        "bash" => (t(if truthy_member(args, "background") { "Background command" } else { "Command" }), or(a("command"), "")),
        "bash_output" => (t("Job output"), first_value(args)),
        "bash_kill" => (t("Stop job"), first_value(args)),
        "web_fetch" => (t("Fetch page"), or(a("url"), "")),
        "web_search" => (t("Web search"), or(a("query"), "")),
        "todo_write" => (t("Task list"), String::new()),
        "todo_read" => (t("Read task list"), String::new()),
        "ask_user" => (t("Question"), or(a("question"), "")),
        "spawn_agent" => (t("Sub-agent"), or(a("description"), "")),
        "task" => crate::subagents::task_label(args),
        "patch" => (t("Apply patch"), crate::commands::patch_paths(&crate::computer::text_or_empty(a("patch").filter(|p| truthy(Some(p))))).join(", ")),
        "review" => (t(if truthy_member(args, "security") { "Security review" } else { "Code review" }), or(a("scope"), "diff")),
        "diagnostics" => (t("Check code"), or(a("path"), "project")),
        "memory" => (
            format!("Memory {}", or(a("action"), "list")),
            format!("{}{}", or(a("scope"), "workspace"), if truthy_member(args, "text") { format!(": {}", or(a("text"), "")) } else { String::new() }),
        ),
        _ => {
            if let Some(l) = crate::computer::tool_label(name, args).or_else(|| crate::browser::tool_label(name, args)) {
                return l;
            }
            if let Some((server, tool)) = mcp {
                return (tool.to_string(), server.to_string());
            }
            let first = args.as_object().and_then(|o| o.values().find_map(Value::as_str)).unwrap_or_default();
            (t(name), slice_units(first, 80))
        }
    }
}

/// The kind of a line of a diff (`diff-add`, `diff-del`, `diff-more`, `diff-ctx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Add,
    Del,
    More,
    Ctx,
}

/// The lines a diff shows at most before *Show n more lines* (`DIFF_LINES`).
pub const DIFF_LINES: usize = 60;
/// The lines an output shows at most before *Show all* (`OUT_LINES`).
pub const OUT_LINES: usize = 40;

/// `diffRows(item)`: what an edit removes and adds, line by line.
pub fn diff_rows(name: &str, args: &Value) -> Vec<(Kind, String)> {
    let empty = Value::Object(Default::default());
    let args = if truthy(Some(args)) { args } else { &empty };
    let text = |v: Option<&Value>| if truthy(v) { js_string(v.unwrap_or(&Value::Null)) } else { String::new() };
    let mut rows = Vec::new();
    let pair = |rows: &mut Vec<(Kind, String)>, old: String, new: String| {
        rows.extend(old.split('\n').map(|l| (Kind::Del, l.to_string())));
        rows.extend(new.split('\n').map(|l| (Kind::Add, l.to_string())));
    };
    match name {
        "edit_file" => pair(&mut rows, text(args.get("old_string")), text(args.get("new_string"))),
        "multi_edit" => {
            if let Some(Value::Array(edits)) = args.get("edits") {
                for (i, e) in edits.iter().enumerate() {
                    if i > 0 {
                        rows.push((Kind::More, "···".into()));
                    }
                    pair(&mut rows, text(e.get("old_string")), text(e.get("new_string")));
                }
            }
        }
        "write_file" => rows.extend(text(args.get("content")).split('\n').map(|l| (Kind::Add, l.to_string()))),
        "patch" => {
            for l in text(args.get("patch")).split('\n') {
                if l.starts_with("+++ ") || l.starts_with("--- ") || l.starts_with("@@") {
                    rows.push((Kind::More, l.to_string()));
                } else if let Some(rest) = l.strip_prefix('+') {
                    rows.push((Kind::Add, rest.to_string()));
                } else if let Some(rest) = l.strip_prefix('-') {
                    rows.push((Kind::Del, rest.to_string()));
                } else {
                    // `l.slice(1)`: the first character is the space of a context line
                    rows.push((Kind::Ctx, l.chars().skip(1).collect()));
                }
            }
        }
        _ => {}
    }
    rows
}

/// What an output shows: the text, and the words of the *Show all* button when there is more
/// (`outputHtml`).
pub fn output_view(output: &str, expanded: bool) -> (String, Option<String>) {
    let units = |s: &str| s.encode_utf16().count();
    let all: Vec<&str> = output.split('\n').collect();
    let mut shown = output.to_string();
    let mut hidden = 0usize;
    if !expanded && (all.len() > OUT_LINES || units(output) > 6000) {
        shown = slice_units(&all[..all.len().min(OUT_LINES)].join("\n"), 6000);
        hidden = all.len().saturating_sub(shown.split('\n').count());
    } else if units(output) > 60000 {
        shown = format!("{}\n... [cut]", slice_units(output, 60000));
    }
    let more = (units(&shown) < units(output) && !expanded).then(|| format!("Show all{}", if hidden > 0 { format!(" ({hidden} more lines)") } else { String::new() }));
    (shown, more)
}

/// The state of a card (`item.status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pending,
    Running,
    Ok,
    Error,
    Denied,
}

impl Status {
    /// `STATUS_TEXT`: the title of the icon.
    pub fn title(self) -> &'static str {
        match self {
            Status::Pending => "waiting for you",
            Status::Running => "running",
            Status::Ok => "done",
            Status::Error => "error",
            Status::Denied => "denied",
        }
    }
}

/// `defaultOpen(item)`: whether a card starts open.
pub fn default_open(name: &str, status: Status, has_output: bool, has_image: bool, has_shot: bool, has_check: bool) -> bool {
    status == Status::Pending
        || matches!(name, "write_file" | "edit_file" | "multi_edit" | "patch")
        || (name.starts_with("computer_") && (has_image || has_shot))
        || name == "ask_user"
        || (name == "bash" && has_output && status == Status::Error)
        || (status == Status::Running && name == "bash")
        || (matches!(name, "task" | "spawn_agent") && status == Status::Running)
        || has_check
}
