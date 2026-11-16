//! Search over past chats and code sessions (`sessions.js` of the Studio): a BM25 index over the
//! words of every chat and saved code session, the search box in Settings > Skills, and the text
//! copy that is kept on the computer (`/api/studio/sessions/index`) so a session can still be found
//! when the file of the window is gone. Temporary chats are never copied.

use crate::app::Message;
use crate::sessions::{self, utf16_len, utf16_slice};
use crate::store::{ChatRecord, Role};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, text, text_input, toggler, Space};
use iced::{Alignment, Element, Font, Length};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

/// What one document may hold (`SS_MAX_TEXT`).
pub const MAX_TEXT: usize = 200_000;

/// `ssTokenize`: lowercase words of letters and digits, at least two characters.
pub fn tokenize(text: &str) -> Vec<String> {
    static SPLIT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let split = SPLIT.get_or_init(|| regex::Regex::new(r"[^\p{L}\p{N}]+").expect("split"));
    split.split(&text.to_lowercase()).filter(|w| utf16_len(w) >= 2).map(str::to_string).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Chat,
    Code,
}

impl Kind {
    pub fn word(self) -> &'static str {
        match self {
            Kind::Chat => "chat",
            Kind::Code => "code",
        }
    }
}

/// `{id, kind, title, text, updated}` of the page.
#[derive(Debug, Clone, PartialEq)]
pub struct Doc {
    pub id: String,
    pub kind: Kind,
    pub title: String,
    pub text: String,
    /// Milliseconds since 1970.
    pub updated: f64,
    pub workspace: String,
}

fn cut(text: String) -> String {
    if utf16_len(&text) > MAX_TEXT {
        utf16_slice(&text, 0, MAX_TEXT)
    } else {
        text
    }
}

/// `ssThreadDoc`: every message of the chat that is yours or the model's, the other versions too.
pub fn chat_doc(c: &ChatRecord) -> Doc {
    let mut lines: Vec<&str> = Vec::new();
    let keep = |role: &Role| matches!(role, Role::User | Role::Assistant);
    for m in c.messages.iter().filter(|m| keep(&m.role) && !m.content.is_empty()) {
        lines.push(&m.content);
    }
    // a version shares its beginning with the shown one: only what differs is new
    for v in &c.variants {
        let shared = v.messages.iter().zip(&c.messages).take_while(|(a, b)| a.role == b.role && a.content == b.content).count();
        for m in v.messages[shared..].iter().filter(|m| keep(&m.role) && !m.content.is_empty()) {
            lines.push(&m.content);
        }
    }
    Doc { id: c.id.to_string(), kind: Kind::Chat, title: c.title.clone(), text: cut(lines.join("\n")), updated: c.updated as f64 * 1000.0, workspace: String::new() }
}

/// `ssCodeDoc`: the words of a saved code session of the folder `root`; tool results are left out.
pub fn code_doc(s: &sessions::Record, root: &str) -> Doc {
    let lines: Vec<String> = s
        .items
        .iter()
        .filter_map(|i| match i {
            crate::code::Saved::User { text, sent } => Some(sent.clone().unwrap_or_else(|| text.clone())),
            crate::code::Saved::Assistant { text } => Some(text.clone()),
            _ => None,
        })
        .filter(|t| !t.is_empty())
        .collect();
    Doc { id: format!("code:{}", s.id), kind: Kind::Code, title: s.title.clone(), text: cut(lines.join("\n")), updated: s.time, workspace: root.to_string() }
}

/// `ssBuildIndex`: term counts per document (the title counts three times), document frequencies, average length.
pub struct Index {
    pub docs: Vec<Doc>,
    tfs: Vec<HashMap<String, f64>>,
    df: HashMap<String, f64>,
    lens: Vec<f64>,
    avg: f64,
}

pub fn build_index(docs: Vec<Doc>) -> Index {
    let mut tfs = Vec::new();
    let mut df: HashMap<String, f64> = HashMap::new();
    let mut lens = Vec::new();
    for d in &docs {
        let mut tf: HashMap<String, f64> = HashMap::new();
        let mut len = 0.0;
        let words = tokenize(&d.title);
        for _ in 0..3 {
            for w in &words {
                *tf.entry(w.clone()).or_insert(0.0) += 1.0;
                len += 1.0;
            }
        }
        for w in tokenize(&d.text) {
            *tf.entry(w).or_insert(0.0) += 1.0;
            len += 1.0;
        }
        for w in tf.keys() {
            *df.entry(w.clone()).or_insert(0.0) += 1.0;
        }
        tfs.push(tf);
        lens.push(len);
    }
    let avg = (lens.iter().sum::<f64>() / (docs.len().max(1) as f64)).max(1.0);
    Index { docs, tfs, df, lens, avg }
}

fn unique_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    tokenize(query).into_iter().filter(|t| seen.insert(t.clone())).collect()
}

/// `ssSearch`: BM25 (k1 1.5, b 0.75), best first, the newer one first among equals: `(index of the document, score)`.
pub fn search(index: &Index, query: &str, limit: usize) -> Vec<(usize, f64)> {
    let terms = unique_terms(query);
    if terms.is_empty() || index.docs.is_empty() {
        return Vec::new();
    }
    let n = index.docs.len() as f64;
    let mut hits = Vec::new();
    for (i, doc) in index.docs.iter().enumerate() {
        let mut score = 0.0;
        for t in &terms {
            let Some(f) = index.tfs[i].get(t).copied().filter(|f| *f != 0.0) else { continue };
            let df = index.df.get(t).copied().unwrap_or(0.0);
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += (idf * f * 2.5) / (f + 1.5 * (0.25 + (0.75 * index.lens[i]) / index.avg));
        }
        if score > 0.0 {
            hits.push((i, score, doc.updated));
        }
    }
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)));
    hits.into_iter().take(limit).map(|(i, s, _)| (i, s)).collect()
}

/// `ssSnippet`: about `width` characters of `text` around the first word of the query that is in it.
pub fn snippet(text: &str, query: &str, width: usize) -> String {
    let lower: Vec<u16> = text.to_lowercase().encode_utf16().collect();
    let mut at: Option<usize> = None;
    for t in unique_terms(query) {
        let needle: Vec<u16> = t.encode_utf16().collect();
        if let Some(i) = lower.windows(needle.len().max(1)).position(|w| w == needle.as_slice()) {
            if at.is_none_or(|a| i < a) {
                at = Some(i);
            }
        }
    }
    let len = utf16_len(text);
    let start = at.unwrap_or(0).saturating_sub(width / 3);
    let end = len.min(start + width);
    static WS: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let ws = WS.get_or_init(|| regex::Regex::new(r"[\s\u{FEFF}]+").expect("ws"));
    let piece = ws.replace_all(&utf16_slice(text, start, end), " ").trim().to_string();
    format!("{}{}{}", if start > 0 { "..." } else { "" }, piece, if end < len { "..." } else { "" })
}

/// A hit of the search: of the window itself, or only of the copy on the computer.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub snippet: String,
    pub time: f64,
    pub score: f64,
    pub from_copy: bool,
}

/// `ssMerge`: the hits of this window first, then those only the copy knows, without repeats.
pub fn merge(local: Vec<Hit>, remote: Vec<Hit>, limit: usize) -> Vec<Hit> {
    let mut seen: HashSet<String> = local.iter().map(|h| h.id.clone()).collect();
    let mut out = local;
    for h in remote {
        if seen.insert(h.id.clone()) {
            out.push(h);
        }
    }
    out.truncate(limit);
    out
}

/// The hits of the documents of this window.
pub fn local_hits(docs: Vec<Doc>, query: &str, limit: usize) -> Vec<Hit> {
    let index = build_index(docs);
    search(&index, query, limit)
        .into_iter()
        .map(|(i, score)| {
            let d = &index.docs[i];
            Hit { id: d.id.clone(), kind: d.kind.word().to_string(), title: d.title.clone(), snippet: snippet(&d.text, query, 160), time: d.updated, score, from_copy: false }
        })
        .collect()
}

// ---- the copy on the computer ----------------------------------------------------------------------------

/// What the server's copy answers to a question (`GET /api/studio/sessions/search`).
pub fn parse_remote(v: &Value) -> Vec<Hit> {
    v.get("hits")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|h| Hit {
                    id: h["id"].as_str().unwrap_or_default().into(),
                    kind: h["kind"].as_str().unwrap_or_default().into(),
                    title: h["title"].as_str().unwrap_or_default().into(),
                    snippet: h["snippet"].as_str().unwrap_or_default().into(),
                    time: h["updated"].as_f64().unwrap_or(0.0),
                    score: h["score"].as_f64().unwrap_or(0.0),
                    from_copy: true,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The local hits and, when the copy is on, the ones only the server's copy knows.
pub async fn search_all(base: String, docs: Vec<Doc>, query: String, limit: usize, copy_on: bool) -> Vec<Hit> {
    let local = local_hits(docs, &query, limit);
    if !copy_on {
        return local;
    }
    let url = format!("{base}/api/studio/sessions/search?q={}&limit={limit}", crate::modelpicker::urlencode(&query));
    let remote = match reqwest::get(url).await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.map(|v| parse_remote(&v)).unwrap_or_default(),
        _ => Vec::new(),
    };
    merge(local, remote, limit)
}

/// What `ssPush` found to do: documents to send, ones with nothing in them to mark, and the ids of the chats that exist.
#[derive(Debug, Clone, Default)]
pub struct PushPlan {
    pub send: Vec<Doc>,
    pub empty: Vec<(String, f64)>,
    pub live: HashSet<String>,
    pub done: BTreeMap<String, f64>,
}

/// The documents that changed since they were sent, a few per call (`chats` and `codes` at most) or all of them.
pub fn push_plan(chats: &[&ChatRecord], codes: &[(String, &sessions::Record)], done: &BTreeMap<String, f64>, streaming: Option<u64>, limits: Option<(usize, usize)>) -> PushPlan {
    let threads: Vec<&&ChatRecord> = chats.iter().filter(|c| !c.temporary).collect();
    let (chat_max, code_max) = limits.unwrap_or((usize::MAX, usize::MAX));
    let mut docs: Vec<Doc> = threads
        .iter()
        .filter(|c| Some(c.id) != streaming)
        .map(|c| chat_doc(c))
        .filter(|d| done.get(&d.id) != Some(&d.updated))
        .take(chat_max)
        .collect();
    docs.extend(codes.iter().map(|(root, s)| code_doc(s, root)).filter(|d| done.get(&d.id) != Some(&d.updated)).take(code_max));
    let mut plan = PushPlan { done: done.clone(), live: threads.iter().map(|c| c.id.to_string()).collect(), ..Default::default() };
    for d in docs {
        if d.text.trim().is_empty() && d.title.is_empty() {
            plan.empty.push((d.id.clone(), d.updated));
        } else {
            plan.send.push(d);
        }
    }
    plan
}

/// `ssPush` without the choosing: the documents go out one by one (the first refusal stops it), then the copies of the chats that are gone are removed.
pub async fn run_push(base: String, plan: PushPlan) -> (BTreeMap<String, f64>, usize) {
    let http = reqwest::Client::new();
    let mut done = plan.done;
    for (id, updated) in plan.empty {
        done.insert(id, updated);
    }
    let mut sent = 0;
    for d in plan.send {
        let body = json!({ "id": d.id, "title": d.title, "text": d.text, "kind": d.kind.word(), "updated": d.updated });
        match http.post(format!("{base}/api/studio/sessions/index")).json(&body).send().await {
            Ok(r) if r.status().is_success() => {
                done.insert(d.id, d.updated);
                sent += 1;
            }
            _ => break,
        }
    }
    let gone: Vec<String> = done.keys().filter(|id| !id.starts_with("code:") && !plan.live.contains(*id)).cloned().collect();
    for id in gone {
        match http.delete(format!("{base}/api/studio/sessions/index?id={}", crate::modelpicker::urlencode(&id))).send().await {
            Ok(r) if r.status().is_success() => {
                done.remove(&id);
            }
            _ => break,
        }
    }
    (done, sent)
}

/// `DELETE /api/studio/sessions/index?all=1`.
pub async fn delete_copy(base: String) -> Result<(), String> {
    let r = reqwest::Client::new().delete(format!("{base}/api/studio/sessions/index?all=1")).send().await.map_err(|e| e.to_string())?;
    if r.status().is_success() {
        Ok(())
    } else {
        Err(format!("Request failed ({})", r.status().as_u16()))
    }
}

/// `fllmSessionForget`: the copy of one session goes (a chat that was deleted).
pub async fn forget(base: String, id: String) {
    let _ = reqwest::Client::new().delete(format!("{base}/api/studio/sessions/index?id={}", crate::modelpicker::urlencode(&id))).send().await;
}

// ---- the card ---------------------------------------------------------------------------------------------------

pub const HELP: &str = "Chats and code sessions are searched by their words. A text copy is also kept in the Studio folder on this computer, so they can still be found after the browser storage was cleared. Temporary chats are never copied.";

/// `fllmSessionIndexInfo`: what is searchable and what was copied.
pub fn info_line(searchable: usize, copied: usize, enabled: bool) -> String {
    format!("{searchable} chats and code sessions in this app, {copied} copied to the computer. {}", if enabled { "" } else { "The copy is switched off." })
}

pub struct PastView<'a> {
    pub query: &'a str,
    pub hits: &'a [Hit],
    /// A question was asked and its answer is here (else *Nothing found.* would show too early).
    pub answered: bool,
    pub enabled: bool,
    pub info: String,
}

pub fn card<'a>(t: Tokens, v: PastView<'a>) -> Element<'a, Message> {
    let mut body = column![
        text(HELP).size(12).color(t.muted),
        row![
            column![text("Keep a searchable copy on this computer").size(14)].width(Length::Fill),
            toggler(v.enabled).on_toggle(Message::PastIndexOn).size(22),
        ]
        .align_y(Alignment::Center),
        text_input("Search chats and code sessions", v.query).on_input(Message::PastQuery).size(13).padding([8, 12]),
    ]
    .spacing(10);
    if v.query.trim().is_empty() {
        // nothing asked, nothing listed
    } else if v.hits.is_empty() {
        if v.answered {
            body = body.push(text("Nothing found.").size(12).color(t.muted));
        }
    } else {
        let mut list = column![].spacing(2);
        for (i, h) in v.hits.iter().enumerate() {
            let title = if h.title.is_empty() { "(untitled)" } else { h.title.as_str() };
            let chip = if h.from_copy { format!("{} - copy", h.kind) } else { h.kind.clone() };
            list = list.push(
                button(
                    row![column![text(title).size(13), text(h.snippet.as_str()).size(11).color(t.muted)].spacing(2).width(Length::Fill), text(chip).size(11).color(t.muted)]
                        .spacing(8)
                        .align_y(Alignment::Center),
                )
                .on_press(Message::PastOpen(i))
                .width(Length::Fill)
                .padding([6, 8])
                .style(theme::ghost_button(t)),
            );
        }
        body = body.push(list);
    }
    body = body.push(text(v.info).size(12).color(t.muted));
    body = body.push(
        row![
            button(text("Copy everything now").size(12)).on_press(Message::PastNow).padding([4, 8]).style(theme::ghost_button(t)),
            button(text("Delete the copy").size(12)).on_press(Message::PastClearAsk).padding([4, 8]).style(theme::ghost_button(t)),
        ]
        .spacing(6),
    );
    crate::models::card(t, "Search of past chats", body.into())
}

/// The question before the copy is deleted (`askDialog` of `btnSessionIndexClear`).
pub fn clear_window<'a>(t: Tokens) -> Element<'a, Message> {
    let window = container(
        column![
            text("Delete the copy?").size(16).font(Font { weight: iced::font::Weight::Bold, ..Font::default() }),
            text("The text copy of your chats on this computer is deleted. The chats in this app stay.").size(13).color(t.fg_soft),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13)).on_press(Message::PastClearCancel).padding([6, 12]).style(theme::ghost_button(t)),
                button(text("Delete the copy").size(13)).on_press(Message::PastClearOk).padding([6, 12]).style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.danger, border: iced::border::rounded(8), ..Default::default() }
                }),
            ]
            .spacing(8),
        ]
        .spacing(10),
    )
    .padding(16)
    .width(420)
    .style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window)).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}
