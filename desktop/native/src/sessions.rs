//! Row 79 of PARITY.md: the sessions of a folder and what the session commands need of the pure
//! part of `code.js` — the titles (`heuristicTitle`, `cleanTitle`), the search of the list
//! (`filterSessions`), the cut of a conversation before a message of yours (`cutBefore`,
//! `firstChangeAfter`), the Markdown of an export (`exportMarkdown`) and the arithmetic of the
//! compaction (`estimateTokens`, `needsCompaction`, `splitForCompaction`,
//! `buildCompactedMessages`).
//!
//! `tests/sessions_fixtures.rs` runs the page's own functions in a vm (`tests/fixtures/
//! sessions_fixtures.js`) and compares the port with them case by case.

use crate::api::WireMessage;
use crate::code::Saved;
use crate::computer::slice_units;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// `MAX_SESSIONS`: how many sessions the page keeps per folder (the newest).
pub const MAX_SESSIONS: usize = 30;

/// `COMPACT_RATIO`: the share of the context the conversation may fill before it is summarised.
pub const COMPACT_RATIO: f64 = 0.7;

/// One task of the list, as it is kept with a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedTodo {
    pub content: String,
    pub status: String,
}

/// What the window remembers of the moment a message of yours was sent (the page's `msgIndex`,
/// `todosBefore` and `changeMark` of a user item): the item and the wire message it stands at,
/// the newest change of the server then and the task list then. A revert and a fork cut here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    /// The index of the user item among the items.
    pub item: usize,
    /// The id of the newest change the server knew when the message was sent.
    #[serde(default)]
    pub changes: u64,
    #[serde(default)]
    pub todos: Vec<SavedTodo>,
}

/// The summary that stands for the first `covered` messages of a conversation after
/// `/compact` (`code.messages` of the page is replaced; here the items stay and the wire is cut).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compaction {
    /// The messages that stand in for the covered ones: the summary and, when the next message is
    /// yours, the line `Understood. …`.
    pub prefix: Vec<(String, String)>,
    pub covered: usize,
    /// The first message of yours that is not summarised: a fork or a revert can go back to it and
    /// to every later one, not to an earlier one (the page forgets their `msgIndex`).
    #[serde(default)]
    pub from_item: usize,
}

/// A session as it is kept: the page's `{ id, title, titleLocked, time, view, messages, todos }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub title_locked: bool,
    /// Milliseconds since 1970 (`Date.now()`).
    #[serde(default)]
    pub time: f64,
    pub items: Vec<Saved>,
    #[serde(default)]
    pub todos: Vec<SavedTodo>,
    #[serde(default)]
    pub marks: Vec<Mark>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<Compaction>,
}

impl Record {
    /// The title the list shows: the one it has, else the first words of the first message.
    pub fn shown_title(&self) -> String {
        if self.title.is_empty() {
            heuristic_title(first_user_text(&self.items).as_deref())
        } else {
            self.title.clone()
        }
    }
}

/// The text of the first message of yours that is no summary (what `heuristicTitle` reads).
pub fn first_user_text(items: &[Saved]) -> Option<String> {
    items.iter().find_map(|i| match i {
        Saved::User { text, sent } => {
            let model_text = sent.clone().unwrap_or_else(|| text.clone());
            (!model_text.starts_with("[Summary")).then_some(model_text)
        }
        _ => None,
    })
}

/// `heuristicTitle(messages)`: a short title from the first user message.
pub fn heuristic_title(first: Option<&str>) -> String {
    let text = first.unwrap_or("Session");
    let text = text.split("\n\nContents of @").next().unwrap_or("");
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let short = slice_units(&collapsed, 60);
    if short.is_empty() {
        "Session".to_string()
    } else {
        short
    }
}

/// `cleanTitle(text)`: one line, no quotes or markdown, at most 60 characters.
pub fn clean_title(text: &str) -> String {
    let line = text.split('\n').map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let lead = regex::Regex::new(r"(?i)^(title|session)\s*:\s*").expect("the title lead");
    let line = lead.replace(line, "").to_string();
    let is_start = |c: char| matches!(c, '#' | '*' | '_' | '`' | '"' | '\'') || c.is_whitespace();
    let is_end = |c: char| matches!(c, '#' | '*' | '_' | '`' | '"' | '\'' | '.') || c.is_whitespace();
    let trimmed = line.trim_start_matches(is_start).trim_end_matches(is_end);
    slice_units(trimmed, 60)
}

/// `filterSessions(sessions, query)`: the sessions whose title or messages hold every word.
pub fn filter_sessions<'a>(sessions: &'a [Record], query: &str) -> Vec<&'a Record> {
    let words: Vec<String> = query.to_lowercase().split_whitespace().map(str::to_string).collect();
    if words.is_empty() {
        return sessions.iter().collect();
    }
    sessions
        .iter()
        .filter(|s| {
            let users: Vec<&str> = s.items.iter().filter_map(|i| if let Saved::User { text, .. } = i { Some(text.as_str()) } else { None }).collect();
            let hay = format!("{} {}", s.title, users.join(" ")).to_lowercase();
            words.iter().all(|w| hay.contains(w))
        })
        .collect()
}

/// `firstChangeAfter(changes, mark)`: the oldest change newer than `mark`.
pub fn first_change_after(ids: &[u64], mark: u64) -> Option<u64> {
    ids.iter().copied().filter(|id| *id > mark).min()
}

fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

fn utf16_slice(s: &str, from: usize, to: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    let to = to.min(units.len());
    let from = from.min(to);
    String::from_utf16_lossy(&units[from..to])
}

/// `clipMiddle(text, max)`: the start and the end of a long text with a marker between.
pub fn clip_middle(text: &str, max: usize) -> String {
    let len = utf16_len(text);
    if len <= max {
        return text.to_string();
    }
    let head = (max as f64 * 0.4).floor() as usize;
    let tail = max - head;
    format!("{}\n... [{} characters cut from the middle] ...\n{}", utf16_slice(text, 0, head), len - max, utf16_slice(text, len - tail, len))
}

fn fence(text: &str) -> String {
    let longest = regex::Regex::new(r"`+").expect("ticks").find_iter(text).map(|m| m.as_str().len()).max().unwrap_or(0);
    let ticks = "`".repeat(3.max(longest + 1));
    format!("{ticks}\n{text}\n{ticks}")
}

fn truthy_str(v: Option<&Value>) -> Option<String> {
    let v = v?;
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) if n.as_f64().is_some_and(|f| f != 0.0) => Some(crate::computer::js_string(v)),
        Value::Bool(true) => Some("true".into()),
        Value::Array(_) | Value::Object(_) => Some(crate::computer::js_string(v)),
        _ => None,
    }
}

fn walk(items: &[Saved], depth: usize, out: &mut Vec<String>) {
    for i in items {
        match i {
            Saved::User { text, .. } => {
                out.push("## User".into());
                out.push(String::new());
                out.push(text.clone());
                out.push(String::new());
            }
            Saved::Assistant { text } => {
                if !text.is_empty() {
                    out.push(if depth > 0 { "**Sub-agent:**" } else { "## Assistant" }.into());
                    out.push(String::new());
                    out.push(text.clone());
                    out.push(String::new());
                }
            }
            Saved::Call { name, args, ok, output, children, .. } => {
                let a: Value = serde_json::from_str(args).unwrap_or(Value::Null);
                let what = ["command", "path", "pattern", "url", "query", "description"].iter().find_map(|k| truthy_str(a.get(k))).unwrap_or_default();
                let first = what.split('\n').next().unwrap_or("");
                let what_part = if what.is_empty() { String::new() } else { format!(" `{}`", slice_units(first, 120)) };
                let status = if *ok { "ok" } else { "error" };
                out.push(format!("- Tool `{name}`{what_part} ({status})"));
                if name == "bash" && !output.is_empty() {
                    out.push(String::new());
                    out.push(fence(&clip_middle(output, 1500)));
                    out.push(String::new());
                }
                if !children.is_empty() {
                    walk(children, depth + 1, out);
                }
                if (name == "task" || name == "spawn_agent") && !output.is_empty() {
                    out.push(String::new());
                    out.push(format!("> {}", output.split('\n').collect::<Vec<_>>().join("\n> ")));
                    out.push(String::new());
                }
            }
            Saved::Note { text } => {
                out.push(format!("> {}", text.split('\n').collect::<Vec<_>>().join("\n> ")));
                out.push(String::new());
            }
            Saved::ReviewFeedback { .. } | Saved::StopHook { .. } => {}
        }
    }
}

/// `exportMarkdown(session, root)`.
pub fn export_markdown(title: &str, time_ms: f64, items: &[Saved], root: &str) -> String {
    let mut out = vec![format!("# {}", if title.is_empty() { "Session" } else { title }), String::new()];
    if !root.is_empty() {
        out.push(format!("Workspace: `{root}`"));
        out.push(String::new());
    }
    // `session.time || Date.now()`
    let time_ms = if time_ms == 0.0 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_millis() as f64) } else { time_ms };
    let when = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(time_ms as i64).map(|d| d.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default();
    out.push(format!("Exported {when} UTC"));
    out.push(String::new());
    walk(items, 0, &mut out);
    let joined = out.join("\n");
    let squeezed = regex::Regex::new(r"\n{3,}").expect("blank lines").replace_all(&joined, "\n\n").to_string();
    format!("{}\n", squeezed.trim())
}

/// `fileSafe(s)` of `chat.js`: letters, digits, `-`, `.` and spaces stay, the rest becomes `_`.
pub fn file_safe(s: &str) -> String {
    let re = regex::Regex::new(r"[^\p{L}\p{N}\-. ]+").expect("file safe");
    let out = slice_units(&re.replace_all(s, "_"), 60);
    if out.is_empty() {
        "chat".into()
    } else {
        out
    }
}

// ---- the arithmetic of the compaction ---------------------------------------------------------

/// `estimateTokens(text)`: `Math.ceil(length / 3.6)`.
pub fn estimate_tokens(text: &str) -> usize {
    (utf16_len(text) as f64 / 3.6).ceil() as usize
}

fn content_tokens(m: &WireMessage) -> usize {
    match &m.parts {
        Some(parts) => parts
            .iter()
            .map(|p| if p.get("type").and_then(Value::as_str) == Some("image_url") { 1000 } else { estimate_tokens(&crate::computer::text_or_empty(p.get("text"))) })
            .sum(),
        None => estimate_tokens(&m.content),
    }
}

/// `estimateMessages(messages)`.
pub fn estimate_messages(messages: &[WireMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let calls = match &m.tool_calls {
                Some(v) => serde_json::to_string(v).unwrap_or_default(),
                None => "\"\"".to_string(),
            };
            content_tokens(m) + estimate_tokens(&calls) + 6
        })
        .sum()
}

/// `needsCompaction(estimate, max, ratio)`.
pub fn needs_compaction(estimate: usize, max: usize, ratio: f64) -> bool {
    max > 0 && estimate as f64 > max as f64 * ratio
}

/// `splitForCompaction(messages, keepLast, minHead)`: where the part to keep begins. A tool result
/// never loses its call.
pub fn split_for_compaction(messages: &[WireMessage], keep_last: usize, min_head: usize) -> Option<usize> {
    if messages.len() < keep_last {
        return None;
    }
    let mut start = messages.len() - keep_last;
    if start < min_head {
        return None;
    }
    while start > 0 && messages[start].role == "tool" {
        start -= 1;
    }
    (start >= min_head).then_some(start)
}

/// `buildCompactedMessages(summary, tail, { todos, files })` without the tail: the messages that
/// stand in front of it.
pub fn build_prefix(summary: &str, todos: &[SavedTodo], files: &[String], tail_first_is_user: bool) -> Vec<WireMessage> {
    let mut parts = vec!["[Summary of the earlier conversation, written by you so that work can continue from it]".to_string(), summary.trim().to_string()];
    if !todos.is_empty() {
        parts.push(format!("Task list:\n{}", todos.iter().map(|t| format!("- [{}] {}", t.status, t.content)).collect::<Vec<_>>().join("\n")));
    }
    if !files.is_empty() {
        parts.push(format!("Files changed in this session: {}", files.iter().take(40).cloned().collect::<Vec<_>>().join(", ")));
    }
    let mut out = vec![WireMessage::new("user", parts.join("\n\n"))];
    if tail_first_is_user {
        out.push(WireMessage::new("assistant", "Understood. I will continue from this summary."));
    }
    out
}

/// `MIN_COMPLETION` of the page.
pub const MIN_COMPLETION: usize = 200;

/// `computeMaxTokens(maxContext, promptTokens, wanted)`: what the answer may use of the context
/// that is left, 0 when less than `MIN_COMPLETION` remains.
pub fn compute_max_tokens(max_context: usize, prompt_tokens: usize, wanted: usize) -> usize {
    let margin = 32 + (max_context as f64 * 0.02).ceil() as i64;
    let room = max_context as i64 - prompt_tokens as i64 - margin;
    let n = (wanted as i64).min(room);
    if n >= MIN_COMPLETION as i64 {
        n as usize
    } else {
        0
    }
}

/// `splitReasoning(text).answer` of the page: what stands outside the `<think>` block.
pub fn answer_of(text: &str) -> String {
    match text.find("<think>") {
        None => text.to_string(),
        Some(open) => match text.find("</think>") {
            None => text[..open].to_string(),
            Some(close) => format!("{}{}", &text[..open], &text[close + "</think>".len()..]),
        },
    }
}

/// The question that asks the model for the summary (`summarise` of the page).
pub fn summary_request(focus: &str) -> String {
    format!(
        "Summarise the conversation above so that work can continue from your summary alone. Include: what the user wants, decisions made, files read or changed (paths) and what changed, commands run and their results, open errors, and the next steps. Keep it under 500 words. Do not call tools.{}",
        if focus.is_empty() { String::new() } else { format!(" Pay special attention to: {focus}") }
    )
}

/// The question that asks the model for a title of the session (`generateTitle`).
pub fn title_request(first: &str) -> String {
    format!("Write a title of at most six words for a coding session that starts with this request. Reply with the title only, without quotes.\n\n{first}")
}

pub const TOO_SMALL: &str = "This model's context is too small for the agent; load it with a larger context under Models > Advanced.";

// ---- the file ----------------------------------------------------------------------------------

/// Sessions by folder, in one file next to the chats. An older file kept one conversation per
/// folder; it is read as one session.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(from = "RawSessions")]
pub struct Sessions(pub BTreeMap<String, Vec<Record>>);

#[derive(Deserialize)]
#[serde(untagged)]
enum Folder {
    New(Vec<Record>),
    Old(Vec<Saved>),
}

#[derive(Deserialize)]
struct RawSessions(BTreeMap<String, Folder>);

impl From<RawSessions> for Sessions {
    fn from(raw: RawSessions) -> Sessions {
        Sessions(
            raw.0
                .into_iter()
                .map(|(root, folder)| {
                    let list = match folder {
                        Folder::New(list) => list,
                        Folder::Old(items) if items.is_empty() => Vec::new(),
                        Folder::Old(items) => vec![Record { id: format!("kept-{}", root.len()), title: String::new(), title_locked: false, time: 0.0, items, todos: Vec::new(), marks: Vec::new(), compaction: None }],
                    };
                    (root, list)
                })
                .collect(),
        )
    }
}

impl Sessions {
    pub fn load(path: &Path) -> Sessions {
        let Ok(text) = std::fs::read_to_string(path) else { return Sessions::default() };
        match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(_) => {
                let _ = std::fs::rename(path, path.with_extension("damaged.json"));
                Sessions::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self).map_err(std::io::Error::other)?)?;
        std::fs::rename(tmp, path)
    }

    /// The sessions of a folder, the newest first.
    pub fn of(&self, root: &str) -> &[Record] {
        self.0.get(root).map_or(&[], Vec::as_slice)
    }

    /// Puts a session at the top of its folder's list (replacing the one with its id) and keeps the newest
    /// `MAX_SESSIONS`.
    pub fn put(&mut self, root: &str, record: Record) {
        let list = self.0.entry(root.to_string()).or_default();
        list.retain(|s| s.id != record.id);
        list.insert(0, record);
        list.truncate(MAX_SESSIONS);
    }

    pub fn remove(&mut self, root: &str, id: &str) {
        if let Some(list) = self.0.get_mut(root) {
            list.retain(|s| s.id != id);
            if list.is_empty() {
                self.0.remove(root);
            }
        }
    }
}

/// `cid()` of the page: an id for a session or an item.
pub fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    format!("s{ms:x}{:x}", N.fetch_add(1, Ordering::Relaxed))
}
