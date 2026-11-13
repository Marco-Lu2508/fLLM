//! Research mode: the pure helpers of the run — the plan, the pages that are worth reading, the
//! notes that fit and the size of the report (the Studio's research.js — every function here is
//! checked against that code, the fixtures are produced by it).

use crate::rag;
use crate::tools::Fetcher;
use serde::{Deserialize, Serialize};

/// How many rounds of searching the Studio does at most (`RESEARCH_MAX_ROUNDS`).
pub const MAX_ROUNDS: u32 = 2;
/// How many requests of one phase run at the same time (`RESEARCH_CONCURRENCY`).
pub const CONCURRENCY: usize = 4;
/// Which share of the context the notes may take when the report is written
/// (`RESEARCH_NOTE_SHARE`).
pub const NOTE_SHARE: f64 = 0.6;
/// How much of a page is read (`RESEARCH_PAGE_CHARS`).
pub const PAGE_CHARS: usize = 60000;

/// Removes reasoning blocks and code fences from a model answer (`researchClean`).
pub fn clean(text: &str) -> String {
    let without_think: String = crate::speech::re(r"(?i)<!--[\s\S]*?(<!-->|$)")
        .replace_all(text, "")
        .into_owned();
    let without_fences: String = crate::speech::re(r"(?i)```[a-z]*\n?")
        .replace_all(&without_think, "")
        .into_owned();
    without_fences.trim().to_string()
}

/// Search queries from the model's plan: strict JSON first, then the first JSON value in the text,
/// then list lines (`parsePlan`).
pub fn parse_plan(text: &str, max: usize) -> Vec<String> {
    let s = clean(text);
    let mut list: Option<Vec<String>> = None;
    let take = |v: serde_json::Value| -> Option<Vec<String>> {
        if let Some(arr) = v.as_array() {
            return Some(arr.iter().map(|q| q.as_str().unwrap_or("").to_string()).collect());
        }
        if v.is_object() {
            for key in ["queries", "searches", "search_queries"] {
                if let Some(arr) = v.get(key).and_then(|a| a.as_array()) {
                    return Some(arr.iter().map(|q| q.as_str().unwrap_or("").to_string()).collect());
                }
            }
        }
        None
    };
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        list = take(v);
    }
    if list.is_none() {
        // the first JSON value in the text
        let json = crate::speech::re(r"\{[\s\S]*\}|\[[\s\S]*\]");
        if let Some(m) = json.find(&s) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(m.as_str()) {
                list = take(v);
            }
        }
    }
    if list.is_none() {
        let line = crate::speech::re(r"^\s*(?:[-*•]|\d+[.)])\s*");
        let edge = crate::speech::re(r#"^["'`]|["'`,]+$"#);
        let braces = crate::speech::re(r"^[{}\[\]]");
        let queries_word = crate::speech::re(r"(?i)^queries\b");
        list = Some(
            s.split('\n')
                .map(|l| edge.replace_all(&line.replace_all(l, ""), "").trim().to_string())
                .filter(|l| !l.is_empty() && !braces.is_match(l) && !queries_word.is_match(l))
                .collect(),
        );
    }
    let mut seen: Vec<String> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for q in list.unwrap_or_default() {
        let t: String = crate::speech::re(r"\s+").replace_all(q.trim(), " ").into_owned();
        let key = t.to_lowercase();
        if t.chars().count() < 3 || t.chars().count() > 200 || seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(t);
        if out.len() >= max {
            break;
        }
    }
    out
}

/// The model's verdict after reading the notes: `{enough, queries}`. Unreadable answers count as
/// "enough" (`parseReflection`).
pub fn parse_reflection(text: &str) -> (bool, Vec<String>) {
    let s = clean(text);
    let mut v: Option<serde_json::Value> = serde_json::from_str(&s).ok();
    if v.is_none() {
        let object = crate::speech::re(r"\{[\s\S]*\}");
        if let Some(m) = object.find(&s) {
            v = serde_json::from_str(m.as_str()).ok();
        }
    }
    let Some(v) = v.filter(|v| v.is_object()) else {
        return (true, Vec::new());
    };
    let queries_raw = v.get("queries").cloned().unwrap_or_else(|| serde_json::json!([]));
    let queries = parse_plan(&serde_json::json!({ "queries": queries_raw }).to_string(), 3);
    let enough = v.get("enough").and_then(|e| e.as_bool()).unwrap_or(queries.is_empty());
    if enough {
        (true, Vec::new())
    } else {
        (false, queries)
    }
}

/// Identity of a page for deduplication: host without "www", path without a trailing slash or
/// index file; no query, no fragment (`urlKey`).
pub fn url_key(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let (authority, path) = match rest.find(['/', '?', '#']) {
        Some(at) => (&rest[..at], &rest[at..]),
        None => (rest, ""),
    };
    if authority.is_empty() {
        return None;
    }
    let host = authority.to_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    // no query, no fragment (the Studio takes u.pathname only)
    let path = path.split(['?', '#']).next().unwrap_or("");
    let path = crate::speech::re(r"(?i)/index\.(html?|php)$").replace_all(path, "").into_owned();
    let path = crate::speech::re(r"/+$").replace_all(&path, "").into_owned();
    Some(format!("{}{}", host.to_lowercase(), path.to_lowercase()))
}

fn junk_host(host: &str) -> bool {
    let host = host.to_lowercase();
    let host = host.strip_prefix("www.").unwrap_or(&host);
    const JUNK: [&str; 12] = [
        "facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "twitter.com", "x.com", "linkedin.com", "youtube.com", "youtu.be", "vimeo.com", "spotify.com", "facebook",
    ];
    host == "pinterest.com" || host.starts_with("pinterest.") || JUNK.iter().any(|j| host == *j || host.ends_with(&format!(".{j}")))
}

fn split_url(url: &str) -> Option<(&str, &str, &str)> {
    let (scheme, rest) = url.split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let (authority, tail) = match rest.find(['/', '?', '#']) {
        Some(at) => (&rest[..at], &rest[at..]),
        None => (rest, ""),
    };
    let path = tail.split(['?', '#']).next().unwrap_or("");
    Some((scheme, authority, path))
}

/// One picked page of `selectUrls`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    pub key: String,
    pub url: String,
    pub title: String,
    pub snippet: String,
    pub hits: usize,
}

/// The best distinct pages from the results of several queries: no junk, no page twice, at most
/// `per_host` per site, pages found by several queries first. `seen` holds the keys of pages
/// already read (`selectUrls`).
pub fn select_urls(results_by_query: &[Vec<serde_json::Value>], limit: usize, per_host: usize, seen: &[String]) -> Vec<Pick> {
    struct Found {
        url: String,
        title: String,
        snippet: String,
        hits: usize,
        rank: usize,
        order: usize,
        host: String,
    }
    let mut found: Vec<(String, Found)> = Vec::new();
    for (qi, results) in results_by_query.iter().enumerate() {
        for (rank, r) in results.iter().enumerate() {
            let url = r.get("url").and_then(|u| u.as_str()).unwrap_or("");
            let Some(key) = url_key(url) else { continue };
            if seen.contains(&key) {
                continue;
            }
            let Some((_scheme, authority, path)) = split_url(url) else { continue };
            let host = authority.to_lowercase();
            let host = host.strip_prefix("www.").unwrap_or(&host).to_string();
            let lower_path = path.to_lowercase();
            if junk_host(&host)
                || crate::speech::re(r"(?i)\.(pdf|zip|gz|tar|7z|png|jpe?g|gif|svg|webp|mp[34]|avi|mov|exe|dmg|iso|docx?|xlsx?|pptx?)$")
                    .is_match(&lower_path)
                || crate::speech::re(r"(?i)/(login|log-in|signin|sign-in|signup|sign-up|register|account|cart|checkout)(/|$)")
                    .is_match(&lower_path)
            {
                continue;
            }
            let title = r.get("title").and_then(|t| t.as_str()).filter(|t| !t.is_empty()).unwrap_or(url).to_string();
            let snippet = r.get("snippet").and_then(|s| s.as_str()).unwrap_or("").to_string();
            if let Some(hit) = found.iter_mut().find(|(k, _)| *k == key) {
                hit.1.hits += 1;
                hit.1.rank = hit.1.rank.min(rank);
                continue;
            }
            found.push((
                key.clone(),
                Found { url: url.to_string(), title, snippet, hits: 1, rank, order: qi * 100 + rank, host },
            ));
        }
    }
    // the pages found by several queries first, then by rank, then by the order they came in
    found.sort_by(|a, b| b.1.hits.cmp(&a.1.hits).then(a.1.rank.cmp(&b.1.rank)).then(a.1.order.cmp(&b.1.order)));
    let mut per_host_count: Vec<(String, usize)> = Vec::new();
    let mut out: Vec<Pick> = Vec::new();
    for (key, c) in found {
        let n = per_host_count.iter().find(|(h, _)| *h == c.host).map(|(_, n)| *n).unwrap_or(0);
        if n >= per_host {
            continue;
        }
        per_host_count.retain(|(h, _)| *h != c.host);
        per_host_count.push((c.host, n + 1));
        out.push(Pick { key, url: c.url, title: c.title, snippet: c.snippet, hits: c.hits });
        if out.len() >= limit {
            break;
        }
    }
    out
}

/// A publication date in the opening of a page ("2025-03-14", "14 March 2025", "March 14,
/// 2025"), or "" (`findDate`).
pub fn find_date(text: &str) -> String {
    let head: String = text.chars().take(3000).collect();
    if let Some(m) = crate::speech::re(r"\b((?:19|20)\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b").find(&head) {
        return m.as_str().to_string();
    }
    let mon = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)";
    let a = crate::speech::re(&format!(r"\b(\d{{1,2}})\s+{mon}\.?,?\s+((?:19|20)\d\d)\b")).find(&head);
    if let Some(m) = a {
        return m.as_str().to_string();
    }
    let b = crate::speech::re(&format!(r"\b{mon}\.?\s+\d{{1,2}},?\s+((?:19|20)\d\d)\b")).find(&head);
    b.map(|m| m.as_str().to_string()).unwrap_or_default()
}

/// What the notes look like after they were shrunk (`fitNotes`).
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub notes: Vec<(u32, String)>,
    pub cut: bool,
    pub dropped: usize,
}

/// Shrinks notes (`[{n, text}]`, best first) to `limit_tokens`: every note is cut in proportion
/// (not below `min_chars`), and when even that is too much the last notes are dropped
/// (`fitNotes`; `estimate` gives the tokens of a text).
pub fn fit_notes(notes: &[(u32, String)], limit_tokens: usize, min_chars: usize) -> Fit {
    // the estimate of the Studio: one token per four characters, rounded up
    let cost = |list: &[(u32, String)]| -> usize { list.iter().map(|(_, t)| t.chars().count().div_ceil(4) + 8).sum() };
    if cost(notes) <= limit_tokens {
        return Fit { notes: notes.to_vec(), cut: false, dropped: 0 };
    }
    let mut list: Vec<(u32, String)> = notes.to_vec();
    let mut dropped = 0;
    loop {
        let fixed = list.len() * 8;
        let body = limit_tokens.saturating_sub(fixed);
        let total: usize = list.iter().map(|(_, t)| t.chars().count().div_ceil(4)).sum::<usize>().max(1);
        let ratio = (body as f64 / total as f64).min(1.0);
        let shrunk: Vec<(u32, String)> = list
            .iter()
            .map(|(n, text)| {
                let len = text.chars().count();
                let keep = (len as f64 * ratio * 0.95).floor() as usize;
                let keep = keep.saturating_sub(4).max(min_chars);
                if len <= keep {
                    return (*n, text.clone());
                }
                let slice: String = text.chars().take(keep).collect();
                let nl = slice.rfind('\n');
                let kept = match nl {
                    Some(at) if at as f64 > keep as f64 * 0.5 => slice[..at].to_string(),
                    _ => slice.trim_end().to_string(),
                };
                (*n, format!("{kept} ..."))
            })
            .collect();
        if cost(&shrunk) <= limit_tokens || list.len() <= 1 {
            return Fit { notes: shrunk, cut: true, dropped };
        }
        list.pop();
        dropped += 1;
    }
}

/// What the notes may take when the report is written (`noteBudget`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteBudget {
    pub limit: usize,
    pub total: usize,
    pub needs_condense: bool,
}

/// How many tokens the notes may take when the report is written, and whether they must be
/// condensed first (`noteBudget`; `estimate` gives the tokens of a text).
pub fn note_budget(notes: &[(u32, String)], max_context: u64, fixed_tokens: usize, reserve_out: usize) -> NoteBudget {
    let estimate = |t: &str| t.chars().count().div_ceil(4);
    let total: usize = notes.iter().map(|(_, t)| estimate(t) + 8).sum();
    let margin = 32 + (max_context as f64 * 0.02).ceil() as usize;
    let limit_f = (max_context as f64 * NOTE_SHARE).min(max_context as f64 - fixed_tokens as f64 - reserve_out as f64 - margin as f64);
    let limit = (limit_f.max(0.0).floor()) as usize;
    NoteBudget { limit, total, needs_condense: total > limit }
}

/// Tokens the answer may use: the wanted size cut to what is left of the context, 0 when under
/// 200 remain (`researchMaxTokens`).
pub fn research_max_tokens(max_context: u64, prompt_tokens: usize, wanted: usize) -> usize {
    let margin = 32 + (max_context as f64 * 0.02).ceil() as usize;
    let n = wanted.min((max_context as f64 - prompt_tokens as f64 - margin as f64).floor() as usize);
    if n >= 200 { n } else { 0 }
}

/// The prompt that asks the model for the plan of the search (`RESEARCH_PLAN_SYSTEM`).
pub const PLAN_SYSTEM: &str = "You plan web research. Given a question, write 3 to 6 focused web search queries that look at it from different angles. Use short keyword queries, not full sentences, and do not repeat the same query in other words. Answer with JSON only, in the form {\"queries\": [\"first query\", \"second query\"]}.";
/// The schema the plan is asked for (`RESEARCH_PLAN_SCHEMA`).
pub const PLAN_SCHEMA: &str = r#"{"type":"object","properties":{"queries":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":6}},"required":["queries"],"additionalProperties":false}"#;
/// The prompt that asks whether the notes are enough (`RESEARCH_REFLECT_SYSTEM`).
pub const REFLECT_SYSTEM: &str = "You check research notes. Decide whether the notes contain enough to answer the question well. If they do not, name up to 3 new web search queries that would fill the gaps. Answer with JSON only, in the form {\"enough\": true, \"queries\": []} or {\"enough\": false, \"queries\": [\"new query\"]}.";
/// The schema the reflection is asked for (`RESEARCH_REFLECT_SCHEMA`).
pub const REFLECT_SCHEMA: &str = r#"{"type":"object","properties":{"enough":{"type":"boolean"},"queries":{"type":"array","items":{"type":"string"},"maxItems":3}},"required":["enough","queries"],"additionalProperties":false}"#;

/// The system prompt for taking notes: the same for every page of a run, so the server can reuse
/// its prefix cache (`researchNoteSystem`).
pub fn note_system(question: &str) -> String {
    format!(
        "You take research notes from one web source at a time.\nResearch question: {question}\nWrite 3 to 5 short bullet points with the facts in the source that help answer the question. Where you can, add a short exact quote in quotation marks. Use only what the source says. If the source has nothing relevant to the question, answer exactly NONE."
    )
}

/// The system prompt of the report (`researchReportSystem`).
pub const REPORT_SYSTEM: &str = "You write research reports from notes. Use only the notes you are given; never add facts or sources of your own.\nStructure the report with these Markdown headings: \"## Summary\" (2 to 4 sentences), \"## Findings\" (bullet points or short paragraphs, each claim followed by the numbers of its sources, for example [1] or [2][3]), \"## Open questions\" (what the notes do not settle). Cite only numbers that appear in the notes. If sources disagree, say so. Answer in the language of the question. Do not write a list of sources: it is added afterwards.";

/// The system prompt that shortens notes before the report (`researchRun`, the condense step).
pub const CONDENSE_SYSTEM: &str = "You shorten research notes. Keep the numbers, names and exact quotes; cut everything else. Answer with at most 2 bullet points.";

/// The line of the log summary of one page (`researchLogHtml`).
pub fn log_summary(queries: usize, ok: usize, bad: usize, seconds: u64) -> String {
    format!(
        "Research log · {queries} {}, {ok} {} read{}, {seconds} s",
        if queries == 1 { "query" } else { "queries" },
        if ok == 1 { "page" } else { "pages" },
        if bad > 0 { format!(", {bad} failed") } else { String::new() }
    )
}

/// The progress line above the answer while the research runs (`researchProgressHtml`).
pub fn progress_line(round: u32, step: u32, steps: u32, phase: &str, done: usize, total: usize) -> String {
    let count = if total > 0 { format!(" ({done}/{total})") } else { String::new() };
    format!("Research · round {round} of {MAX_ROUNDS} · step {step} of {steps}: {phase}{count}")
}

/// The section of the log that the sources of the report add (`researchSourcesMarkdown`).
pub fn sources_markdown(pages: &[(u32, String, String, String)]) -> String {
    // (n, title, url, date) — only pages with a number are cited
    let used: Vec<&(u32, String, String, String)> = pages.iter().filter(|(n, _, _, _)| *n > 0).collect();
    if used.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = used
        .iter()
        .map(|(n, title, url, date)| {
            let title = title.replace(['[', ']'], "");
            if date.is_empty() {
                format!("{n}. [{title}]({url})")
            } else {
                format!("{n}. [{title}]({url}), {date}")
            }
        })
        .collect();
    format!("\n\n## Sources\n\n{}\n", lines.join("\n"))
}


/// What one page of the run gave back (`researchRound`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub n: u32,
    pub title: String,
    pub url: String,
    pub date: String,
    pub ok: bool,
    pub round: u32,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub snippet_only: bool,
    #[serde(default)]
    pub notes: String,
}

/// What the window shows while the research runs (the `r` on the node of the Studio).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchUi {
    pub round: u32,
    pub step: u32,
    pub steps: u32,
    pub phase: String,
    pub done: usize,
    pub total: usize,
    pub queries: Vec<String>,
    pub pages: Vec<Page>,
    pub failures: Vec<String>,
    /// `(seconds after the start, what happened)` — at most eighty, the oldest leave
    pub events: Vec<(f64, String)>,
    pub notice: Option<String>,
    pub running: bool,
    pub stopped: bool,
    pub seconds: u64,
    pub log_open: bool,
    /// When the run started (milliseconds since 1970), so the log can count the seconds.
    pub t0_ms: u64,
}

impl ResearchUi {
    /// The state the window shows the moment the question is asked (`researchRun`).
    pub fn new(t0_ms: u64) -> Self {
        ResearchUi {
            round: 1,
            step: 1,
            steps: 5,
            phase: "planning".into(),
            done: 0,
            total: 0,
            queries: Vec::new(),
            pages: Vec::new(),
            failures: Vec::new(),
            events: Vec::new(),
            notice: None,
            running: true,
            stopped: false,
            seconds: 0,
            log_open: false,
            t0_ms,
        }
    }

    /// One line of the log (`researchEvent`).
    pub fn event(&mut self, seconds: f64, text: impl Into<String>) {
        self.events.push((seconds, text.into()));
        if self.events.len() > 80 {
            self.events.remove(0);
        }
    }

    /// The progress line above the answer, the notice of a fallback or the stopped line
    /// (`researchProgressHtml`).
    pub fn progress(&self, _now_ms: u64) -> String {
        if let Some(notice) = &self.notice {
            return notice.clone();
        }
        if self.running {
            return progress_line(self.round, self.step, self.steps, &self.phase, self.done, self.total);
        }
        if self.stopped {
            return "Research stopped.".to_string();
        }
        String::new()
    }

    /// The summary line of the log under the answer (`researchLogHtml`).
    pub fn summary(&self, now_ms: u64) -> String {
        let ok = self.pages.iter().filter(|p| p.ok).count();
        let bad = self.pages.len() - ok;
        let secs = if self.running { now_ms.saturating_sub(self.t0_ms) / 1000 } else { self.seconds };
        log_summary(self.queries.len(), ok, bad, secs)
    }
}

/// What the run prepared for the window: the report to stream, or the fallback to a normal answer.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// The system and the user message of the report (empty for a fallback).
    pub system: String,
    pub user: String,
    pub ui: ResearchUi,
    /// The line of the Studio when the research could not run.
    pub notice: Option<String>,
    /// The model answers on its own instead (the window starts a normal answer).
    pub fallback: bool,
    /// The cited pages for the "## Sources" at the end of the report.
    pub sources: Vec<(u32, String, String, String)>,
    /// Whether the server refused the JSON schema (the run asks plainly from then on).
    pub no_schema: bool,
}

/// The error line of a refused web request in a research run (the Studio's `ragWebError`).
pub fn web_error(status: u16, detail: &str) -> String {
    match status {
        404 => "Web search is not available: this server has no web tools (the route was not found).".to_string(),
        403 => format!("Web access is blocked{}.", if detail.is_empty() { " by this server".to_string() } else { format!(": {detail}") }),
        _ => format!("Web search failed{}.", if detail.is_empty() { format!(" (error {status})") } else { format!(": {detail}") }),
    }
}

fn wire(role: &str, content: String) -> crate::api::WireMessage {
    crate::api::WireMessage { role: role.into(), content, parts: None, tool_calls: None, tool_call_id: None }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The whole research run for one question (the Studio's `researchRun`): plan the queries, search
/// and read the best pages, take notes, ask whether they are enough — one more round at most —
/// and prepare the report. Every phase is sent to the window through `tx`; when the window is gone
/// (`tx.is_closed()`) the run stops where it is.
pub async fn prepare(
    base: &str,
    model: &str,
    ctx: u64,
    question: &str,
    no_schema: bool,
    tx: futures::channel::mpsc::UnboundedSender<ResearchUi>,
) -> Option<Prepared> {
    let started = std::time::Instant::now();
    let t0 = now_ms();
    let mut ui = ResearchUi::new(t0);
    let mut no_schema = no_schema;
    let seconds = || (started.elapsed().as_millis() as f64 / 100.0).round() / 10.0;
    // every result leaves the run as "over", with the time it took (researchRun's finally)
    let over = |ui: &mut ResearchUi, what: &str| {
        ui.seconds = started.elapsed().as_secs();
        ui.running = false;
        ui.event(started.elapsed().as_millis() as f64 / 1000.0, what.to_string());
    };
    let push_progress = |ui: &ResearchUi| {
        let _ = tx.unbounded_send(ui.clone());
    };
    let push_event = |ui: &mut ResearchUi, text: String| {
        ui.event(seconds(), text);
        push_progress(ui);
    };
    let fetcher = crate::api::EngineFetcher { base: base.to_string() };

    push_event(&mut ui, "Planning the search queries.".to_string());
    let mut queries: Vec<String> = Vec::new();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let plan_system = format!("{PLAN_SYSTEM} Today is {today}.");
    match crate::api::chat_once(base, model, vec![wire("system", plan_system), wire("user", question.to_string())], 300, Some(PLAN_SCHEMA), no_schema).await
    {
        Ok((text, refused)) => {
            no_schema |= refused;
            queries = parse_plan(&text, 6);
        }
        Err(e) => push_event(&mut ui, format!("Planning: {e}")),
    }
    if tx.is_closed() {
        return None;
    }
    if queries.is_empty() {
        let cut: String = question.chars().take(200).collect();
        queries = vec![cut];
        push_event(&mut ui, "The plan was unreadable: the question itself is searched.".to_string());
    }
    ui.queries.extend(queries.iter().cloned());
    push_progress(&ui);

    let mut seen: Vec<String> = Vec::new();
    let mut round_queries = queries.clone();
    let mut round: u32 = 1;
    while round <= MAX_ROUNDS {
        ui.round = round;
        ui.step = 2;
        let wanted = if round == 1 { if ctx <= 8192 { 4 } else { 6 } } else { 3 };
        // ---- search the web for this round's queries (researchRound) ----
        let mut search_done = 0;
        ui.phase = "searching the web".into();
        ui.done = 0;
        ui.total = round_queries.len();
        push_progress(&ui);
        let mut results: Vec<Vec<serde_json::Value>> = Vec::new();
        for chunk in round_queries.chunks(CONCURRENCY) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|q| {
                    let fetcher = &fetcher;
                    let q = q.clone();
                    async move {
                        let resp = match fetcher.post("/api/studio/web/search".to_string(), serde_json::json!({ "query": q, "limit": 8 })).await {
                            Ok(r) => r,
                            Err(e) => return (q, Err(e), Vec::new()),
                        };
                        if resp.status == 404 || resp.status == 403 {
                            return (q, Err(web_error(resp.status, "")), Vec::new());
                        }
                        if resp.status >= 400 {
                            // a refused search is a problem of the log, not the end of the run
                            return (q, Err(web_error(resp.status, "")), Vec::new());
                        }
                        let hits = resp.body.get("results").and_then(|r| r.as_array()).cloned().unwrap_or_default();
                        (q, Ok(()), hits)
                    }
                })
                .collect();
            for (q, verdict, hits) in futures::future::join_all(futures).await {
                match verdict {
                    Err(msg) if msg.contains("Web access is blocked") || msg.contains("no web tools") => {
                        // the server has no web tools: the research cannot run at all
                        let notice = format!("Research could not run: {msg} The model answers on its own instead.");
                        push_event(&mut ui, format!("Fell back to a normal answer: {msg}"));
                        over(&mut ui, "Fell back to a normal answer.");
                        ui.notice = Some(notice.clone());
                        return Some(Prepared {
                            system: String::new(),
                            user: String::new(),
                            ui,
                            notice: Some(notice),
                            fallback: true,
                            sources: Vec::new(),
                            no_schema,
                        });
                    }
                    Err(msg) => {
                        ui.failures.push(format!("Search \"{q}\": {msg}"));
                        results.push(Vec::new());
                    }
                    Ok(()) => results.push(hits),
                }
                search_done += 1;
                ui.done = search_done;
                push_progress(&ui);
            }
            if tx.is_closed() {
                return None;
            }
        }
        let total_results: usize = results.iter().map(|r| r.len()).sum();
        let picks = select_urls(&results, wanted, 2, &seen);
        push_event(&mut ui, format!("Round {round}: {total_results} results, {} new pages chosen.", picks.len()));
        if picks.is_empty() {
            break;
        }
        for p in &picks {
            seen.push(p.key.clone());
        }

        // ---- read the pages and take their passages (researchRound) ----
        ui.phase = "reading pages".into();
        ui.done = 0;
        ui.total = picks.len();
        push_progress(&ui);
        let full_query = std::iter::once(question.to_string()).chain(round_queries.iter().cloned()).collect::<Vec<_>>().join(" ");
        let k = 2.max(6.min(((ctx as f64 * 0.3 / (rag::CHUNK_CHARS as f64 / 3.6)).floor()) as usize));
        let mut entries: Vec<Page> = Vec::new();
        let mut read_done = 0;
        for chunk in picks.chunks(CONCURRENCY) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|p| {
                    let fetcher = &fetcher;
                    let p = p.clone();
                    let full_query = full_query.clone();
                    async move {
                        let mut entry = Page { n: 0, title: p.title.clone(), url: p.url.clone(), date: String::new(), ok: false, round, error: String::new(), snippet_only: false, notes: String::new() };
                        let text;
                        let mut failure: Option<String> = None;
                        match fetcher.post("/api/studio/web/fetch".to_string(), serde_json::json!({ "url": p.url, "max_chars": PAGE_CHARS })).await {
                            Ok(resp) if resp.status < 400 => {
                                let page = resp.body.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string();
                                if page.is_empty() {
                                    if p.snippet.is_empty() {
                                        entry.error = "the page has no readable text".into();
                                        return (entry, String::new(), None);
                                    }
                                    text = format!("{}\n\n{}", p.title, p.snippet);
                                    entry.snippet_only = true;
                                    failure = Some(format!("{}: the page has no readable text (using the search snippet)", p.url));
                                } else {
                                    text = page;
                                }
                            }
                            Ok(resp) => {
                                let why = format!("error {}", resp.status);
                                if p.snippet.is_empty() {
                                    entry.error = why;
                                    return (entry, String::new(), None);
                                }
                                text = format!("{}\n\n{}", p.title, p.snippet);
                                entry.snippet_only = true;
                                failure = Some(format!("{}: {why} (using the search snippet)", p.url));
                            }
                            Err(e) => {
                                if p.snippet.is_empty() {
                                    entry.error = e;
                                    return (entry, String::new(), None);
                                }
                                text = format!("{}\n\n{}", p.title, p.snippet);
                                entry.snippet_only = true;
                                failure = Some(format!("{}: {e} (using the search snippet)", p.url));
                            }
                        }
                        entry.date = find_date(&text);
                        let doc = rag::IndexedDoc { id: "1", name: &entry.title, text: &text };
                        let index = rag::build_index(&[doc]);
                        let mut found = rag::search(&index, &full_query, k, None);
                        if found.is_empty() {
                            found = rag::opening(&index, k.min(3));
                        }
                        found.sort_by_key(|p| p.n);
                        let passages: Vec<&str> = found.iter().map(|p| p.text.as_str()).collect();
                        entry.ok = !passages.is_empty();
                        if !entry.ok {
                            entry.error = "no passage matched the question".into();
                        }
                        (entry, passages.join("\n\n"), failure)
                    }
                })
                .collect();
            for (entry, _passages, failure) in futures::future::join_all(futures).await {
                if let Some(line) = failure {
                    ui.failures.push(line);
                }
                entries.push(entry);
                read_done += 1;
                ui.done = read_done;
                push_progress(&ui);
            }
            if tx.is_closed() {
                return None;
            }
        }
        ui.pages.extend(entries.iter().cloned());

        // ---- take notes of the readable pages ----
        let readable: Vec<Page> = entries.into_iter().filter(|e| e.ok).collect();
        ui.phase = "taking notes".into();
        ui.done = 0;
        ui.total = readable.len();
        push_progress(&ui);
        let note_sys = note_system(question);
        let room_tokens = research_max_tokens(ctx, note_sys.chars().count().div_ceil(4) + k * 500 + 100, 500);
        let room = 200.max(500.min(if room_tokens == 0 { 200 } else { room_tokens }));
        let mut note_done = 0;
        let mut with_notes: Vec<Page> = Vec::new();
        for chunk in readable.chunks(CONCURRENCY) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|e| {
                    let e = e.clone();
                    let note_sys = note_sys.clone();
                    async move {
                        let user = format!("Source: {} ({})\n\n{}", e.title, e.url, e.notes);
                        let mut e = e;
                        match crate::api::chat_once(base, model, vec![wire("system", note_sys), wire("user", user)], room as u32, None, no_schema).await {
                            Ok((text, _)) => {
                                let none = crate::speech::re(r"(?s)^\W*NONE\W*$").is_match(&text);
                                e.notes = if none { String::new() } else { text };
                            }
                            Err(err) => {
                                e.ok = false;
                                e.error = format!("notes failed: {err}");
                            }
                        }
                        e
                    }
                })
                .collect();
            for e in futures::future::join_all(futures).await {
                if !e.ok && e.error.starts_with("notes failed") {
                    ui.failures.push(format!("{}: {}", e.url, e.error));
                }
                note_done += 1;
                ui.done = note_done;
                with_notes.push(e);
                push_progress(&ui);
            }
            if tx.is_closed() {
                return None;
            }
        }
        // the pages go back into the list of the run, with their notes
        for e in with_notes {
            if let Some(slot) = ui.pages.iter_mut().find(|p| p.url == e.url && p.round == e.round) {
                *slot = e;
            }
        }

        let notes: Vec<Page> = ui.pages.iter().filter(|p| p.ok && !p.notes.is_empty()).cloned().collect();
        if round >= MAX_ROUNDS || notes.is_empty() {
            break;
        }
        // ---- ask whether the notes are enough (researchRun) ----
        ui.step = 3;
        ui.phase = "checking whether the notes are enough".into();
        ui.done = 0;
        ui.total = 0;
        push_progress(&ui);
        let digest: String = notes
            .iter()
            .map(|p| {
                let collapsed: String = p.notes.split_whitespace().collect::<Vec<_>>().join(" ");
                let short: String = collapsed.chars().take(300).collect();
                format!("- {}: {}", p.title, short)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let room_chars = ((ctx as f64 * 0.5 * 3.6).floor()) as usize;
        let digest: String = digest.chars().take(room_chars).collect();
        let mut verdict = (true, Vec::<String>::new());
        match crate::api::chat_once(
            base,
            model,
            vec![wire("system", REFLECT_SYSTEM.to_string()), wire("user", format!("Question: {question}\n\nNotes:\n{digest}"))],
            200,
            Some(REFLECT_SCHEMA),
            no_schema,
        )
        .await
        {
            Ok((text, refused)) => {
                no_schema |= refused;
                verdict = parse_reflection(&text);
            }
            Err(e) => push_event(&mut ui, format!("Reflection: {e}")),
        }
        if tx.is_closed() {
            return None;
        }
        let (enough, more) = verdict;
        let fresh: Vec<String> = more
            .into_iter()
            .filter(|q| !ui.queries.iter().any(|x| x.to_lowercase() == q.to_lowercase()))
            .collect();
        if enough || fresh.is_empty() {
            push_event(&mut ui, "The notes are enough.".to_string());
            break;
        }
        push_event(&mut ui, format!("Gaps found, one more round: {}", fresh.join("; ")));
        ui.queries.extend(fresh.iter().cloned());
        round_queries = fresh;
        round += 1;
    }
    ui.round = ui.round.min(MAX_ROUNDS);

    // ---- write the report (researchRun) ----
    let any_usable = ui.pages.iter().any(|p| p.ok && !p.notes.is_empty());
    if !any_usable {
        let why = if ui.pages.is_empty() { "the web search found no usable pages." } else { "none of the pages had anything relevant to the question." };
        let notice = format!("Research could not run: {why} The model answers on its own instead.");
        push_event(&mut ui, format!("Fell back to a normal answer: {why}"));
        over(&mut ui, "Fell back to a normal answer.");
        ui.notice = Some(notice.clone());
        return Some(Prepared {
            system: String::new(),
            user: String::new(),
            ui,
            notice: Some(notice),
            fallback: true,
            sources: Vec::new(),
            no_schema,
        });
    }
    ui.step = 4;
    // the pages get their numbers on the run itself, so the window and the sources see them
    let mut number = 0u32;
    for p in ui.pages.iter_mut() {
        if p.ok && !p.notes.is_empty() {
            number += 1;
            p.n = number;
        }
    }
    let usable: Vec<Page> = ui.pages.iter().filter(|p| p.n > 0).cloned().collect();
    let mut list: Vec<(u32, String)> = usable.iter().map(|p| (p.n, p.notes.clone())).collect();
    let system = REPORT_SYSTEM.to_string();
    let fixed = system.chars().count().div_ceil(4) + question.chars().count().div_ceil(4) + 60;
    let budget = note_budget(&list, ctx, fixed, 1200);
    if budget.needs_condense {
        push_event(&mut ui, format!("Notes take {} of {} allowed tokens: condensing.", budget.total, budget.limit));
        let mut order: Vec<usize> = (0..list.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(list[i].1.chars().count()));
        let mut total = budget.total;
        let mut targets: Vec<usize> = Vec::new();
        for i in order {
            targets.push(i);
            total -= list[i].1.chars().count().div_ceil(4) / 2;
            if total <= budget.limit {
                break;
            }
        }
        ui.phase = "condensing notes".into();
        ui.done = 0;
        ui.total = targets.len();
        push_progress(&ui);
        let mut condensed = 0;
        for chunk in targets.chunks(CONCURRENCY) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|&i| {
                    let text = list[i].1.clone();
                    async move {
                        let out = crate::api::chat_once(
                            base,
                            model,
                            vec![wire("system", CONDENSE_SYSTEM.to_string()), wire("user", text)],
                            200,
                            None,
                            no_schema,
                        )
                        .await;
                        (i, out)
                    }
                })
                .collect();
            for (i, out) in futures::future::join_all(futures).await {
                if let Ok((text, _)) = out {
                    if !text.is_empty() {
                        list[i].1 = text.clone();
                        // the condensed note goes back onto the run, so the log shows it too
                        if let Some(p) = ui.pages.iter_mut().find(|p| p.n == i as u32 + 1) {
                            p.notes = text;
                        }
                    }
                }
                condensed += 1;
                ui.done = condensed;
                push_progress(&ui);
            }
            if tx.is_closed() {
                return None;
            }
        }
    }
    let fit = fit_notes(&list, budget.limit, 240);
    if fit.cut {
        push_event(
            &mut ui,
            format!(
                "Notes were cut to fit the context{}.",
                if fit.dropped > 0 { format!(" and {} were left out", fit.dropped) } else { String::new() }
            ),
        );
    }
    list = fit.notes;
    let keep: Vec<u32> = list.iter().map(|(n, _)| *n).collect();
    for p in ui.pages.iter_mut() {
        if p.n > 0 && !keep.contains(&p.n) {
            // the page was left out of the report: it is not a source of it
            p.n = 0;
        }
    }
    ui.step = 5;
    ui.phase = "writing the report".into();
    ui.done = 0;
    ui.total = 0;
    push_progress(&ui);
    push_event(&mut ui, "Writing the report.".to_string());
    let notes_text: Vec<String> = list
        .iter()
        .map(|(n, text)| {
            let page = usable.iter().find(|p| p.n == *n).expect("the page of the note");
            format!("[{n}] {} ({})\n{text}", page.title, page.url)
        })
        .collect();
    let user = format!("Question: {question}\n\nNotes:\n\n{}\n\nWrite the report.", notes_text.join("\n\n"));
    over(&mut ui, "Done.");
    let sources: Vec<(u32, String, String, String)> = ui
        .pages
        .iter()
        .filter(|p| p.n > 0)
        .map(|p| (p.n, p.title.clone(), p.url.clone(), p.date.clone()))
        .collect();
    Some(Prepared {
        system,
        user,
        ui,
        notice: None,
        fallback: false,
        sources,
        no_schema,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixtures() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/research_fixtures.json")).expect("the fixtures are json")
    }

    /// Every helper against the Studio's own output (the fixtures come from research.js).
    #[test]
    fn the_research_helpers_are_the_studios() {
        let f = fixtures();
        assert_eq!(f["limits"]["rounds"], MAX_ROUNDS, "RESEARCH_MAX_ROUNDS");
        assert_eq!(f["limits"]["concurrency"], CONCURRENCY, "RESEARCH_CONCURRENCY");
        assert_eq!(f["limits"]["noteShare"], NOTE_SHARE, "RESEARCH_NOTE_SHARE");
        assert_eq!(f["limits"]["pageChars"], PAGE_CHARS, "RESEARCH_PAGE_CHARS");

        for (i, want) in f["clean"].as_array().expect("the cleans").iter().enumerate() {
            let input = ["plain text", " ```rust\nfn x() {}\n```  ", "reason  think block  visible", "  spaced   out  ", ""][i];
            assert_eq!(clean(input), *want, "clean {input:?}");
        }

        for case in f["plans"].as_array().expect("the plans") {
            let text = case["text"].as_str().expect("a plan text");
            let max = case["max"].as_u64().map(|m| m as usize).unwrap_or(6);
            let want: Vec<String> = case["out"].as_array().expect("the queries").iter().map(|q| q.as_str().unwrap().to_string()).collect();
            assert_eq!(parse_plan(text, max), want, "plan {text:?}");
        }

        for case in f["reflections"].as_array().expect("the reflections") {
            let text = case["text"].as_str().expect("a reflection");
            let (enough, queries) = parse_reflection(text);
            let want = &case["out"];
            assert_eq!(enough, want["enough"].as_bool().expect("enough"), "enough for {text:?}");
            let want_queries: Vec<String> = want["queries"].as_array().expect("queries").iter().map(|q| q.as_str().unwrap().to_string()).collect();
            assert_eq!(queries, want_queries, "queries for {text:?}");
        }

        for case in f["urls"].as_array().expect("the urls") {
            let url = case["url"].as_str().expect("a url");
            let want = case["out"].as_str();
            assert_eq!(url_key(url).as_deref(), want, "key of {url:?}");
        }

        for case in f["selects"].as_array().expect("the selects") {
            let by_query: Vec<Vec<serde_json::Value>> = case["byQuery"].as_array().expect("queries").iter().map(|q| q.as_array().expect("results").clone()).collect();
            let want: Vec<String> = case["out"].as_array().expect("picks").iter().map(|p| p["url"].as_str().unwrap().to_string()).collect();
            let picks = select_urls(&by_query, 6, 2, &[]);
            let got: Vec<String> = picks.iter().map(|p| p.url.clone()).collect();
            assert_eq!(got, want, "the best pages");
            // the hits of the first pick, as the Studio counts them
            let want_hits: Vec<usize> = case["out"].as_array().expect("picks").iter().map(|p| p["hits"].as_u64().unwrap() as usize).collect();
            let got_hits: Vec<usize> = picks.iter().map(|p| p.hits).collect();
            assert_eq!(got_hits, want_hits, "how many queries found each page");
            // with the first page already read the list is what the Studio returned for that
            let want_seen: Vec<String> = case["seen"].as_array().expect("seen").iter().map(|s| s.as_str().unwrap().to_string()).collect();
            if let Some(first) = picks.first() {
                let rest = select_urls(&by_query, 6, 2, std::slice::from_ref(&first.key));
                let rest_keys: Vec<String> = rest.iter().map(|p| p.key.clone()).collect();
                assert_eq!(rest_keys, want_seen, "with the first page already read");
            } else {
                assert!(want_seen.is_empty(), "nothing was picked, nothing can be read twice");
            }
        }

        for (i, want) in f["dates"].as_array().expect("the dates").iter().enumerate() {
            let text = [
                "The post from 2025-03-14 says hi.",
                "Published 14 March 2025 by someone.",
                "Dated March 14, 2025 — old news.",
                "Nothing here.",
                "2026-11-13 is the date at position beyond three thousand characters but only after the head: xxx",
            ][i];
            let text = if i == 4 { format!("{}{}", text, "x".repeat(3100)) } else { text.to_string() };
            assert_eq!(find_date(&text), *want, "date of case {i}");
        }

        for case in f["fits"].as_array().expect("the fits") {
            let limit = case["limit"].as_u64().unwrap() as usize;
            let notes: Vec<(u32, String)> = case["notes"].as_array().expect("notes").iter().map(|n| (n["n"].as_u64().unwrap() as u32, "x".repeat(n["len"].as_u64().unwrap() as usize))).collect();
            let fit = fit_notes(&notes, limit, 240);
            let want = &case["out"];
            assert_eq!(json!(fit.cut), want["cut"], "cut for limit {limit}");
            assert_eq!(json!(fit.dropped), want["dropped"], "dropped for limit {limit}");
            let lens: Vec<usize> = fit.notes.iter().map(|(_, t)| t.chars().count()).collect();
            let want_lens: Vec<usize> = want["lens"].as_array().expect("lens").iter().map(|l| l.as_u64().unwrap() as usize).collect();
            assert_eq!(lens, want_lens, "the lengths for limit {limit}");
            for (i, tail) in want["tails"].as_array().expect("tails").iter().enumerate() {
                let got_tail: String = fit.notes[i].1.chars().rev().take(5).collect::<String>().chars().rev().collect();
                assert_eq!(got_tail, *tail, "the tail of note {i}");
            }
        }

        for case in f["noteBudgets"].as_array().expect("the budgets") {
            let notes: Vec<(u32, String)> = case["notes"].as_array().expect("notes").iter().enumerate().map(|(i, n)| (i as u32, "x".repeat(n.as_u64().unwrap() as usize))).collect();
            let max = case["max"].as_u64().unwrap();
            let fixed = case["fixed"].as_u64().unwrap() as usize;
            let budget = note_budget(&notes, max, fixed, 1200);
            let want = &case["out"];
            assert_eq!(json!({ "limit": budget.limit, "total": budget.total, "needsCondense": budget.needs_condense }), *want, "budget for {max}/{fixed}");
        }

        for (i, want) in f["maxTokens"].as_array().expect("the max tokens").iter().enumerate() {
            let (c, p, w) = [(8192u64, 1000usize, 1800usize), (8192, 8000, 1800), (4096, 4000, 1800), (2048, 100, 1800)][i];
            assert_eq!(research_max_tokens(c, p, w), *want, "max tokens for {c}/{p}");
        }
    }

    #[test]
    fn the_prompts_of_the_run_are_the_studios() {
        // the prompts are copied from research.js; the shapes of the answers they ask for
        assert!(PLAN_SYSTEM.starts_with("You plan web research."));
        assert!(PLAN_SYSTEM.ends_with("{\"queries\": [\"first query\", \"second query\"]}."));
        assert!(REFLECT_SYSTEM.contains("{\"enough\": true, \"queries\": []}"));
        assert!(REPORT_SYSTEM.contains("## Summary") && REPORT_SYSTEM.contains("## Findings") && REPORT_SYSTEM.contains("## Open questions"));
        assert!(note_system("q").contains("Research question: q"));
        assert!(CONDENSE_SYSTEM.contains("at most 2 bullet points"));
        // the log and the progress line of the Studio
        assert_eq!(log_summary(1, 1, 0, 12), "Research log · 1 query, 1 page read, 12 s");
        assert_eq!(log_summary(2, 3, 1, 5), "Research log · 2 queries, 3 pages read, 1 failed, 5 s");
        assert_eq!(progress_line(1, 2, 5, "reading pages", 3, 6), "Research · round 1 of 2 · step 2 of 5: reading pages (3/6)");
        assert_eq!(progress_line(1, 4, 5, "writing the report", 0, 0), "Research · round 1 of 2 · step 4 of 5: writing the report");
        // the sources at the end of the report
        let pages = vec![
            (1u32, "The blog".to_string(), "https://a.test/x".to_string(), "2025-03-14".to_string()),
            (0u32, "Unused".to_string(), "https://b.test/y".to_string(), String::new()),
        ];
        assert_eq!(
            sources_markdown(&pages),
            "\n\n## Sources\n\n1. [The blog](https://a.test/x), 2025-03-14\n",
            "only the cited pages, the brackets of the title taken out"
        );
        assert_eq!(sources_markdown(&[]), "", "without pages there is no list");
    }
}
