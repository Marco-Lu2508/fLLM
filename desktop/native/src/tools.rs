//! The tools of a chat: what is offered, how a call runs and what the answer carries back (the
//! Studio's tools.js — every function here is checked against that code, the fixtures are
//! produced by `ctRunTool` itself with a mock server).

use crate::memory::Change;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// How many rounds of tool calls an answer may run (`CT_MAX_ROUNDS`).
pub const MAX_ROUNDS: usize = 5;
/// How many calls of one round are honoured (`CT_MAX_CALLS`).
pub const MAX_CALLS: usize = 4;

/// What a bad call is told to look like (`CT_EXAMPLE`).
pub fn example(name: &str) -> &'static str {
    match name {
        "web_search" => "{\"query\": \"latest stable version of Rust\"}",
        "web_fetch" => "{\"url\": \"https://example.com/page\"}",
        "memory" => "{\"action\": \"add\", \"text\": \"Prefers metric units\"}",
        "search_chats" => "{\"query\": \"trip to Lisbon\"}",
        "skill" => "{\"action\": \"read\", \"name\": \"release-checklist\"}",
        "schedule" => "{\"name\": \"Morning news\", \"prompt\": \"Summarise the top technology news\", \"when\": \"daily at 08:00\"}",
        _ => "{}",
    }
}

/// What one tool call may put into the answer: `per` for one result, `left` for all of them
/// (`ctBudget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub per: usize,
    pub left: usize,
}

/// The budget of an answer for a model with `max_context` tokens (`ctBudget`).
pub fn budget(max_context: u64) -> Budget {
    let max = if max_context == 0 { 4096 } else { max_context };
    let chars = (max as f64 * 3.4).floor() as usize;
    // the Studio: Math.max(1200, Math.min(12000, floor(chars * 0.15)))
    let per = (((chars as f64 * 0.15).floor()) as usize).clamp(1200, 12000);
    Budget { per, left: ((chars as f64 * 0.4).floor()) as usize }
}

/// Text cut to `max` characters with a line that says how much was cut (`ctClip`).
pub fn clip(text: &str, max: usize) -> String {
    let len = text.chars().count();
    if len <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    format!("{cut}\n... [{} characters cut]", len - max)
}

/// The host of an address, or the first forty characters when there is none (`ctHost`).
pub fn host(url: &str) -> String {
    if let Some(rest) = url.split_once("://").map(|(_, r)| r) {
        let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
        if !authority.is_empty() {
            return authority.to_string();
        }
    }
    url.chars().take(40).collect()
}

/// One line of text, the whitespace collapsed, cut to `n` with an ellipsis (`ctShort`).
pub fn short(s: &str, n: usize) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<&str>>().join(" ");
    if collapsed.chars().count() > n {
        let cut: String = collapsed.chars().take(n.saturating_sub(1)).collect();
        format!("{cut}...")
    } else {
        collapsed
    }
}

/// What a refused web request says (`ctHttpError`).
pub fn http_error(status: u16, detail: &str) -> String {
    match status {
        404 => "Web access is not available: this server has no web tools.".to_string(),
        403 => format!("Web access is blocked{}.", if detail.is_empty() { " by this server".to_string() } else { format!(": {detail}") }),
        _ => format!("The web request failed{}.", if detail.is_empty() { format!(" (error {status})") } else { format!(": {detail}") }),
    }
}

/// Which tools the chat is offered (`ctOfferedFlags`; `chats` is not ported yet).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    pub web: bool,
    pub memory: bool,
    pub chats: bool,
}

/// The tool definitions in the OpenAI format for the flags (`ctDefs`).
pub fn defs(flags: Flags, compact: bool) -> Value {
    let text = |name: &str| -> String {
        let (full, small) = match name {
            "web_search" => (
                "Search the web. Use only for current, specific or unknown facts. Returns titles, links and snippets.",
                "Search the web for current or unknown facts.",
            ),
            "web_fetch" => (
                "Open a web page and read it as text. Use it on a result of web_search, or on a link the user gave.",
                "Read a web page as text.",
            ),
            "memory" => (
                "Save, change or delete a short fact about the user that stays useful in later chats (name, preferences, projects, constraints). Never trivia or secrets.",
                "Save durable facts about the user.",
            ),
            "search_chats" => (
                "Search the title and text of the user's earlier chats and return matching snippets. Use when the user refers to an earlier conversation.",
                "Search the user's earlier chats.",
            ),
            _ => ("", ""),
        };
        if compact {
            small.to_string()
        } else {
            full.to_string()
        }
    };
    let d = |name: &str, props: Value, required: &[&str]| -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": text(name),
                "parameters": { "type": "object", "properties": props, "required": required },
            }
        })
    };
    let mut out: Vec<Value> = Vec::new();
    if flags.web {
        out.push(d(
            "web_search",
            serde_json::json!({ "query": { "type": "string", "description": "What to search for, as short search keywords" } }),
            &["query"],
        ));
        out.push(d(
            "web_fetch",
            serde_json::json!({ "url": { "type": "string", "description": "The full http or https address" } }),
            &["url"],
        ));
    }
    if flags.memory {
        out.push(d(
            "memory",
            serde_json::json!({
                "action": { "type": "string", "enum": ["add", "update", "remove", "list"] },
                "text": { "type": "string", "description": "The fact, one short sentence (add, update)" },
                "category": { "type": "string", "description": "profile, preference, project, constraint or other" },
                "match": { "type": "string", "description": "Part of the existing entry to update or remove" },
                "id": { "type": "string", "description": "Id of the entry, from list (instead of match)" },
            }),
            &["action"],
        ));
    }
    if flags.chats {
        out.push(d(
            "search_chats",
            serde_json::json!({ "query": { "type": "string", "description": "Keywords to look for" } }),
            &["query"],
        ));
    }
    Value::Array(out)
}

/// Arguments of a call: a JSON object, also when wrapped in a code fence; None when unusable
/// (`ctParseArgs`).
pub fn parse_args(text: &str) -> Option<Value> {
    let raw = text.trim();
    if raw.is_empty() {
        return Some(Value::Object(serde_json::Map::new()));
    }
    let unfenced = raw
        .strip_prefix("```json")
        .or_else(|| raw.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .map(|s| s.trim())
        .unwrap_or("");
    for candidate in [raw, unfenced] {
        if candidate.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(candidate) {
            return Some(v);
        }
    }
    None
}

/// What the label of a card says while the tool runs and when it is done (`ctLabel`).
pub fn label(name: &str, args: &Value, done: bool) -> String {
    let arg = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    match name {
        "web_search" => format!("{}: {}", if done { "Searched" } else { "Searching" }, short(&arg("query"), 80)),
        "web_fetch" => {
            let url = arg("url");
            let bare = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://")).unwrap_or(&url);
            format!("{}: {}", if done { "Read" } else { "Reading" }, short(bare, 80))
        }
        "memory" => if done { "Memory".to_string() } else { "Updating memory".to_string() },
        "search_chats" => format!("{}: {}", if done { "Searched your chats" } else { "Searching your chats" }, short(&arg("query"), 60)),
        _ => name.to_string(),
    }
}

/// One source of an answer: numbered, so the model can cite it as `[n]` (`ctSourceNumber`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub n: u32,
    pub title: String,
    pub url: String,
    #[serde(default)]
    pub fetched: bool,
}

/// The number `[n]` of a source: an existing one for the same address, else the next free one.
pub fn source_number(sources: &mut Vec<Source>, url: &str, title: &str, fetched: bool) -> u32 {
    if let Some(hit) = sources.iter().find(|s| s.url == url) {
        return hit.n;
    }
    let n = sources.len() as u32 + 1;
    sources.push(Source { n, title: if title.is_empty() { url.to_string() } else { title.to_string() }, url: url.to_string(), fetched });
    n
}

/// What one call of a tool gives back: the text for the model and the card under the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub text: String,
    pub card: Card,
}

/// The card of a call under the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub label: String,
    pub status: &'static str,
    pub detail: Option<String>,
    pub change: Option<Change>,
    pub quiet: bool,
}

/// What the memory of the window gives back when the model calls the tool.
pub struct MemoryRun {
    pub text: String,
    pub change: Option<Change>,
    /// The list of entries after the call (the window puts it into the settings).
    pub list: Vec<crate::memory::Memory>,
}

/// The answer of the server to one web request: the status and the body (`ctRunTool` reads
/// `ok`, `error`, `results`, `url`, `text` from it).
pub struct Resp {
    pub status: u16,
    pub body: Value,
}

/// How a call reaches the server for its web tools: the routes of the Studio
/// (`/api/studio/web/search` and `/api/studio/web/fetch`).
pub trait Fetcher: Send + Sync {
    fn post(&self, route: String, body: Value) -> Pin<Box<dyn Future<Output = Result<Resp, String>> + Send>>;
}

/// What the memory of the window does when the model calls the tool.
pub type MemoryFn = Box<dyn FnMut(&Value) -> MemoryRun + Send>;

/// The environment of one call, as `ctRunTool` sees it.
pub struct Env {
    pub budget: Budget,
    pub sources: Vec<Source>,
    /// What the memory of the window does with a call (None: the chat has no memory).
    pub memory: Option<MemoryFn>,
}

/// Runs one tool call: the web tools ask the server, the memory asks the window (the Studio's
/// `ctRunTool`).
pub async fn run<F: Fetcher>(name: &str, raw_args: &str, env: &mut Env, fetcher: &F) -> Outcome {
    let parsed = parse_args(raw_args);
    let label_of = |parsed: Option<&Value>, done: bool| -> String {
        // the Studio passes the empty object when the arguments are unusable (a || {})
        match parsed {
            Some(a) => label(name, a, done),
            None => label(name, &Value::Object(serde_json::Map::new()), done),
        }
    };
    // the Studio's fail(): the text is also the detail of the card
    let fail_with = |parsed: Option<&Value>, text: String| Outcome {
        card: Card { label: label_of(parsed, true), status: "error", detail: Some(text.clone()), change: None, quiet: false },
        text,
    };
    if !matches!(name, "web_search" | "web_fetch" | "memory" | "search_chats") {
        // the Studio's own branch: the detail of the card is the literal "unknown tool"
        return Outcome {
            text: format!("Error: unknown tool '{name}'."),
            card: Card { label: name.to_string(), status: "error", detail: Some("unknown tool".into()), change: None, quiet: false },
        };
    }
    let Some(args) = parsed else {
        return fail_with(
            None,
            format!(
                "Error: the arguments of {name} are not valid JSON. Call it again with a JSON object such as {}.",
                example(name)
            ),
        );
    };
    let room = || env.budget.per.min(env.budget.left);
    let spend = |text: &str, budget: &mut Budget| {
        budget.left = budget.left.saturating_sub(text.chars().count());
    };

    if (name == "web_search" || name == "web_fetch") && env.budget.left < 400 {
        return fail_with(Some(&args), "Skipped: the room for web results in this answer is used up. Answer with what you have.".to_string());
    }

    if name == "web_search" {
        let query = args.get("query").and_then(Value::as_str).unwrap_or("").trim().to_string();
        if query.is_empty() {
            return fail_with(Some(&args), "Error: \"query\" is required.".to_string());
        }
        let resp = match fetcher.post("/api/studio/web/search".to_string(), serde_json::json!({ "query": query, "limit": 6 })).await {
            Ok(r) => r,
            Err(e) => return fail_with(Some(&args), format!("Error: the web search failed: {e}")),
        };
        if resp.status >= 400 {
            return fail_with(Some(&args), format!("Error: {}", http_error(resp.status, "")));
        }
        if resp.body.get("ok").and_then(Value::as_bool) == Some(false) {
            let detail = resp.body.get("error").and_then(Value::as_str).unwrap_or("the web search failed");
            return fail_with(Some(&args), format!("Error: {detail}"));
        }
        let hits: Vec<Value> = resp
            .body
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|r| {
                r.get("url")
                    .and_then(Value::as_str)
                    .map(|u| u.starts_with("http://") || u.starts_with("https://"))
                    .unwrap_or(false)
            })
            .collect();
        if hits.is_empty() {
            return Outcome {
                text: format!("No results for \"{query}\". Try other keywords or answer without the web."),
                card: Card { label: label_of(Some(&args), true), status: "done", detail: Some("no results".into()), change: None, quiet: false },
            };
        }
        let mut lines: Vec<String> = Vec::new();
        for hit in &hits {
            let url = hit.get("url").and_then(Value::as_str).unwrap_or("");
            let title = hit.get("title").and_then(Value::as_str).unwrap_or("");
            let snippet = hit.get("snippet").and_then(Value::as_str).unwrap_or("");
            let n = source_number(&mut env.sources, url, title, false);
            lines.push(format!("[{n}] {} ({url})\n{}", short(title, 120).if_empty(url), short(snippet, 300)));
        }
        let text = clip(
            &format!("Search results for \"{query}\". Cite sources as [n].\n\n{}", lines.join("\n\n")),
            room(),
        );
        spend(&text, &mut env.budget);
        let detail = format!("{} result{}", hits.len(), if hits.len() == 1 { "" } else { "s" });
        return Outcome { text, card: Card { label: label_of(Some(&args), true), status: "done", detail: Some(detail), change: None, quiet: false } };
    }

    if name == "web_fetch" {
        let url = args.get("url").and_then(Value::as_str).unwrap_or("").trim().to_string();
        // /^https?:\/\/\S+$/i — a scheme of http or https and something without a space behind it
        let looks_http = match url.find("://") {
            Some(at) => {
                let scheme = &url[..at];
                let rest = &url[at + 3..];
                (scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
                    && !rest.is_empty()
                    && !rest.chars().any(char::is_whitespace)
            }
            None => false,
        };
        if !looks_http {
            return fail_with(Some(&args), "Error: \"url\" must be a full http or https address.".to_string());
        }
        let resp = match fetcher
            .post("/api/studio/web/fetch".to_string(), serde_json::json!({ "url": url, "max_chars": room().max(2000) }))
            .await
        {
            Ok(r) => r,
            Err(e) => return fail_with(Some(&args), format!("Error: the page could not be read: {e}")),
        };
        if resp.status >= 400 {
            return fail_with(Some(&args), format!("Error: {}", http_error(resp.status, "")));
        }
        if resp.body.get("ok").and_then(Value::as_bool) == Some(false) {
            let detail = resp.body.get("error").and_then(Value::as_str).unwrap_or("the page could not be read");
            return fail_with(Some(&args), format!("Error: {detail}"));
        }
        let page = resp.body.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string();
        if page.is_empty() {
            return fail_with(Some(&args), "Error: the page has no readable text.".to_string());
        }
        let page_url = resp.body.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()).unwrap_or(&url).to_string();
        let n = source_number(&mut env.sources, &page_url, &host(&url), true);
        let text = clip(&format!("[{n}] {page_url}\n{page}"), room());
        spend(&text, &mut env.budget);
        return Outcome {
            text,
            card: Card { label: label_of(Some(&args), true), status: "done", detail: Some(format!("{} characters", page.chars().count())), change: None, quiet: false },
        };
    }

    // memory: the window of the chat does it (the Studio's env.memory.apply)
    let Some(apply) = env.memory.as_mut() else {
        return fail_with(Some(&args), "Error: memory is switched off in this chat.".to_string());
    };
    let run = apply(&args);
    let text = run.text.clone();
    let Some(change) = run.change.clone() else {
        let refused = text.starts_with("Not saved");
        let quiet = args.get("action").and_then(Value::as_str).map(|a| a.eq_ignore_ascii_case("list")).unwrap_or(false) || text == "Already remembered.";
        let status = if refused || text.starts_with("Error") { "error" } else { "done" };
        return Outcome {
            text,
            card: Card {
                label: if refused { "Memory: not saved (looks like a secret)".into() } else if quiet { "Memory checked".into() } else { format!("Memory: {}", short(&run.text, 80)) },
                status,
                detail: None,
                change: None,
                quiet: quiet && !refused,
            },
        };
    };
    let label_text = match change.kind.as_str() {
        "add" => format!("Saved to memory: {}", change.after.as_ref().map(|m| m.text.clone()).unwrap_or_default()),
        "update" => format!("Updated memory: {}", change.after.as_ref().map(|m| m.text.clone()).unwrap_or_default()),
        _ => format!("Removed from memory: {}", change.before.as_ref().map(|m| m.text.clone()).unwrap_or_default()),
    };
    Outcome {
        text,
        card: Card { label: short(&label_text, 140), status: "done", detail: None, change: Some(change), quiet: false },
    }
}

/// A tiny helper for the search lines: an empty title falls back to the address.
trait IfEmpty {
    fn if_empty(self, fallback: &str) -> String;
}
impl IfEmpty for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixtures() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/tools_fixtures.json")).expect("the fixtures are json")
    }

    /// Every helper and every call of `ctRunTool` against the Studio's own output.
    #[test]
    fn the_tool_machinery_is_the_studios() {
        let cases = fixtures();
        for case in cases.as_array().expect("the cases") {
            if let Some(kind) = case.get("kind").and_then(Value::as_str) {
                match kind {
                    "budget" => {
                        for (i, want) in case["outputs"].as_array().expect("budgets").iter().enumerate() {
                            let max = [4096u64, 16384, 200, 0, 999999][i];
                            let got = budget(max);
                            assert_eq!(json!({ "per": got.per, "total": got.left }), *want, "budget for {max}");
                        }
                    }
                    "clip" => {
                        let pairs = [("short", 10usize), ("a much longer text that has to be cut in the middle", 20), ("", 5)];
                        for (i, want) in case["outputs"].as_array().expect("clips").iter().enumerate() {
                            assert_eq!(clip(pairs[i].0, pairs[i].1), *want, "clip {:?}", pairs[i]);
                        }
                    }
                    "short" => {
                        let pairs = [("  spaced   out  ", 100usize), ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 10), ("", 5)];
                        for (i, want) in case["outputs"].as_array().expect("shorts").iter().enumerate() {
                            assert_eq!(short(pairs[i].0, pairs[i].1), *want, "short {:?}", pairs[i]);
                        }
                    }
                    "host" => {
                        let urls = ["https://example.com:8443/a/b", "not a url", "ftp://x.test/p"];
                        for (i, want) in case["outputs"].as_array().expect("hosts").iter().enumerate() {
                            assert_eq!(host(urls[i]), *want, "host {:?}", urls[i]);
                        }
                    }
                    "http" => {
                        let pairs = [(404u16, ""), (403, "because"), (500, ""), (401, "")];
                        for (i, want) in case["outputs"].as_array().expect("http").iter().enumerate() {
                            assert_eq!(http_error(pairs[i].0, pairs[i].1), *want, "http {:?}", pairs[i]);
                        }
                    }
                    "limits" => {
                        assert_eq!(case["rounds"], MAX_ROUNDS, "CT_MAX_ROUNDS");
                        assert_eq!(case["calls"], MAX_CALLS, "CT_MAX_CALLS");
                        for name in ["web_search", "web_fetch", "memory", "search_chats", "skill", "schedule"] {
                            assert_eq!(example(name), case["example"][name], "example for {name}");
                        }
                    }
                    "defs" => {
                        let flags = [(true, false, false), (false, true, false), (true, true, true), (false, false, false)];
                        let compacts = [false, false, true, false];
                        for (i, want) in case["outputs"].as_array().expect("defs").iter().enumerate() {
                            let f = Flags { web: flags[i].0, memory: flags[i].1, chats: flags[i].2 };
                            assert_eq!(defs(f, compacts[i]), *want, "defs for {flags:?} compact={}", compacts[i]);
                        }
                    }
                    "parse" => {
                        let inputs = ["{\"a\": 1}", "```json\n{\"a\": 1}\n```", "not json", "", "  {\"query\":\"x\"}  "];
                        for (i, want) in case["outputs"].as_array().expect("parses").iter().enumerate() {
                            let got = parse_args(inputs[i]).unwrap_or(Value::Null);
                            assert_eq!(got, *want, "parse {:?}", inputs[i]);
                        }
                    }
                    "label" => {
                        let pairs = [
                            (json!({ "query": "rust version" }), false),
                            (json!({ "query": "rust version" }), true),
                            (json!({ "url": "https://example.com/page" }), false),
                            (json!({ "url": "https://example.com/page" }), true),
                        ];
                        for (i, want) in case["outputs"].as_array().expect("labels").iter().enumerate() {
                            let name = if i < 2 { "web_search" } else { "web_fetch" };
                            assert_eq!(label(name, &pairs[i].0, pairs[i].1), *want, "label {i}");
                        }
                    }
                    "sources" => {
                        let mut sources: Vec<Source> = Vec::new();
                        let seq = [
                            ("https://a.test/one", "One", false),
                            ("https://a.test/two", "Two", false),
                            ("https://a.test/one", "One again", false),
                            ("https://a.test/page", "a.test", true),
                        ];
                        let mut numbers = Vec::new();
                        for (url, title, fetched) in seq {
                            numbers.push(source_number(&mut sources, url, title, fetched));
                        }
                        let numbers: Vec<Value> = numbers.into_iter().map(Value::from).collect();
                        assert_eq!(Value::Array(numbers), case["numbers"], "the numbers [n]");
                        let want: Vec<Value> = case["list"].as_array().expect("the list").clone();
                        let got: Vec<Value> = sources.iter().map(|s| serde_json::to_value(s).unwrap()).collect();
                        assert_eq!(got, want, "the sources of the answer");
                    }
                    other => panic!("unknown fixture kind {other}"),
                }
                continue;
            }
            // a call of ctRunTool against the mock server
            futures::executor::block_on(run_case(case));
        }
    }

    async fn run_case(case: &Value) {
        let name = case["name"].as_str().expect("a tool");
        let args = match &case["args"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let spec = &case["spec"];
        let memory_holder: Option<MemoryFn> = if name == "memory" {
            let m = spec.get("memory").cloned().unwrap_or(json!({}));
            let text = m.get("text").and_then(Value::as_str).unwrap_or("Already remembered.").to_string();
            let change: Option<Change> = m
                .get("change")
                .cloned()
                .filter(|c| !c.is_null())
                .and_then(|c| serde_json::from_value(c).ok());
            Some(Box::new(move |_args: &Value| MemoryRun { text: text.clone(), change: change.clone(), list: Vec::new() })
                as Box<dyn FnMut(&Value) -> MemoryRun + Send>)
        } else {
            None
        };
        let mut env = Env {
            budget: Budget { per: 2000, left: spec["left"].as_u64().unwrap_or(5000) as usize },
            sources: Vec::new(),
            memory: memory_holder,
        };
        if let Some(list) = spec.get("sources") {
            env.sources = serde_json::from_value(list.clone()).expect("sources of the spec");
        }
        // the mock of the fixtures: canned search and pages
        struct SpecFetcher { spec: Value }
        impl Fetcher for SpecFetcher {
            fn post(&self, route: String, body: Value) -> Pin<Box<dyn Future<Output = Result<Resp, String>> + Send>> {
                let spec = self.spec.clone();
                let body = body.clone();
                Box::pin(async move {
                    if route == "/api/studio/web/search" {
                        let s = spec.get("search").cloned().unwrap_or(json!({ "ok": true, "results": [] }));
                        let status = s.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
                        Ok(Resp { status, body: s })
                    } else {
                        let url = body.get("url").and_then(Value::as_str).unwrap_or("").to_string();
                        let p = spec
                            .get("pages")
                            .and_then(|pages| pages.get(&url))
                            .cloned()
                            .unwrap_or(json!({ "ok": false, "error": "not found" }));
                        let status = p.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
                        Ok(Resp { status, body: p })
                    }
                })
            }
        }
        let fetcher = SpecFetcher { spec: spec.clone() };
        let out = run(name, &args, &mut env, &fetcher).await;
        assert_eq!(out.text, case["out"]["text"], "text of {name} {args}");
        assert_eq!(out.card.label, case["out"]["card"]["label"], "label of {name} {args}");
        assert_eq!(out.card.status, case["out"]["card"]["status"], "status of {name} {args}");
        if let Some(detail) = case["out"]["card"]["detail"].as_str() {
            assert_eq!(out.card.detail.as_deref(), Some(detail), "detail of {name} {args}");
        }
        assert_eq!(json!(env.budget.left), case["after"]["left"], "budget after {name} {args}");
        let want_sources = case["after"]["sources"].clone();
        let got_sources: Vec<Value> = env.sources.iter().map(|s| serde_json::to_value(s).unwrap()).collect();
        assert_eq!(Value::Array(got_sources), want_sources, "sources after {name} {args}");
    }
}
