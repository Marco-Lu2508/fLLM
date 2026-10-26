//! A persistent copy of the text of the chats and code sessions, so that they can be searched after the browser cache
//! was cleared. The page sends each changed session (`POST /api/studio/sessions/index`); the copy is a JSON-lines file
//! `sessions.jsonl` next to `studio.json`, appended to and compacted when it grows past a size cap (the newest
//! record of each session is kept, the oldest sessions go first). Search (`GET /api/studio/sessions/search?q=`) is
//! BM25 over title and text, computed on the records of the file.
use super::*;
use axum::extract::Query;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_TEXT_CHARS: usize = 200_000;
const MAX_FILE_BYTES: u64 = 24 << 20;
const COMPACT_TO_BYTES: usize = 12 << 20;
const MAX_SESSIONS: usize = 5000;
const SNIPPET_CHARS: usize = 160;

static FILE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Record {
    pub id: String,
    pub title: String,
    pub text: String,
    pub kind: String,
    pub updated: u64,
}

impl Record {
    fn to_json(&self) -> Value {
        json!({"id": self.id, "title": self.title, "text": self.text, "kind": self.kind, "updated": self.updated})
    }

    fn from_json(v: &Value) -> Option<Record> {
        let id = v["id"].as_str().filter(|s| !s.is_empty() && s.len() <= 100)?.to_owned();
        let kind = if v["kind"].as_str() == Some("code") { "code" } else { "chat" };
        Some(Record {
            id,
            title: clip(v["title"].as_str().unwrap_or_default(), 200),
            text: clip(v["text"].as_str().unwrap_or_default(), MAX_TEXT_CHARS),
            kind: kind.to_owned(),
            updated: v["updated"].as_u64().unwrap_or(0),
        })
    }
}

fn clip(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn data_file() -> PathBuf {
    crate::hub::studio_data_file("sessions.jsonl")
}

pub(super) fn read_all(file: &Path) -> Vec<Record> {
    let Ok(text) = fs::read_to_string(file) else { return Vec::new() };
    text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter_map(|v| Record::from_json(&v)).collect()
}

/// The newest record of every session (later lines win), newest session first.
pub(super) fn latest(records: Vec<Record>) -> Vec<Record> {
    let mut by_id: HashMap<String, Record> = HashMap::new();
    for r in records {
        by_id.insert(r.id.clone(), r);
    }
    let mut out: Vec<Record> = by_id.into_values().collect();
    out.sort_by(|a, b| b.updated.cmp(&a.updated).then_with(|| a.id.cmp(&b.id)));
    out
}

fn write_all(file: &Path, records: &[Record]) -> Result<(), String> {
    let mut text = String::new();
    for r in records {
        text.push_str(&r.to_json().to_string());
        text.push('\n');
    }
    let tmp = file.with_extension("jsonl.tmp");
    fs::write(&tmp, text).map_err(|e| e.to_string())?;
    fs::rename(&tmp, file).map_err(|e| e.to_string())
}

/// Keeps the newest sessions whose records fit into `budget` bytes (and at most `MAX_SESSIONS`).
pub(super) fn compact_records(records: Vec<Record>, budget: usize) -> Vec<Record> {
    let mut kept = Vec::new();
    let mut size = 0;
    for r in latest(records) {
        size += r.to_json().to_string().len() + 1;
        if size > budget || kept.len() >= MAX_SESSIONS {
            break;
        }
        kept.push(r);
    }
    kept
}

/// Appends one record; compacts the file when it is past the cap.
pub(super) fn append(file: &Path, record: &Record, cap: u64, budget: usize) -> Result<(), String> {
    let _guard = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let mut out = fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| e.to_string())?;
    writeln!(out, "{}", record.to_json()).map_err(|e| e.to_string())?;
    drop(out);
    if fs::metadata(file).is_ok_and(|m| m.len() > cap) {
        write_all(file, &compact_records(read_all(file), budget))?;
    }
    Ok(())
}

pub(super) fn forget(file: &Path, id: &str) -> Result<(), String> {
    let _guard = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let kept: Vec<Record> = latest(read_all(file)).into_iter().filter(|r| r.id != id).collect();
    write_all(file, &kept)
}

pub(super) fn forget_all(file: &Path) -> Result<(), String> {
    let _guard = FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match fs::remove_file(file) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
        _ => Ok(()),
    }
}

/// Lowercase words of letters and digits (at least two characters).
pub(super) fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(str::to_owned)
        .collect()
}

pub(super) struct Hit {
    pub score: f64,
    pub index: usize,
}

/// BM25 (k1 1.5, b 0.75) of the query words over title (counted three times) and text. Best first.
pub(super) fn rank(docs: &[Record], query: &str) -> Vec<Hit> {
    let mut terms = tokenize(query);
    terms.sort();
    terms.dedup();
    if terms.is_empty() || docs.is_empty() {
        return Vec::new();
    }
    let counted: Vec<(HashMap<String, f64>, f64)> = docs
        .iter()
        .map(|d| {
            let mut tf: HashMap<String, f64> = HashMap::new();
            let mut len = 0.0;
            for _ in 0..3 {
                for w in tokenize(&d.title) {
                    *tf.entry(w).or_default() += 1.0;
                    len += 1.0;
                }
            }
            for w in tokenize(&d.text) {
                *tf.entry(w).or_default() += 1.0;
                len += 1.0;
            }
            (tf, len)
        })
        .collect();
    let n = docs.len() as f64;
    let average = (counted.iter().map(|(_, l)| l).sum::<f64>() / n).max(1.0);
    let mut hits = Vec::new();
    for (index, (tf, len)) in counted.iter().enumerate() {
        let mut score = 0.0;
        for t in &terms {
            let Some(&f) = tf.get(t) else { continue };
            let df = counted.iter().filter(|(m, _)| m.contains_key(t)).count() as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len / average));
        }
        if score > 0.0 {
            hits.push(Hit { score, index });
        }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| docs[b.index].updated.cmp(&docs[a.index].updated)));
    hits
}

/// About `SNIPPET_CHARS` characters of `text` around the first occurrence of a query word.
pub(super) fn snippet(text: &str, query: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let lower: Vec<char> = chars.iter().map(|c| c.to_lowercase().next().unwrap_or(*c)).collect();
    let terms: HashSet<String> = tokenize(query).into_iter().collect();
    let mut first: Option<usize> = None;
    for t in &terms {
        let needle: Vec<char> = t.chars().collect();
        if let Some(at) = lower.windows(needle.len()).position(|w| w == needle.as_slice()) {
            first = Some(first.map_or(at, |f| f.min(at)));
        }
    }
    let at = first.unwrap_or(0);
    let start = at.saturating_sub(SNIPPET_CHARS / 3);
    let end = (start + SNIPPET_CHARS).min(chars.len());
    let body: String = chars[start..end].iter().map(|c| if c.is_whitespace() { ' ' } else { *c }).collect();
    format!("{}{}{}", if start > 0 { "..." } else { "" }, body.trim(), if end < chars.len() { "..." } else { "" })
}

pub(super) fn search_in(file: &Path, query: &str, limit: usize) -> Vec<Value> {
    let docs = latest(read_all(file));
    rank(&docs, query)
        .into_iter()
        .take(limit)
        .map(|h| {
            let d = &docs[h.index];
            json!({"id": d.id, "title": d.title, "kind": d.kind, "updated": d.updated, "score": h.score, "snippet": snippet(&d.text, query)})
        })
        .collect()
}

/// `POST /api/studio/sessions/index` `{id, title, text, kind, updated}`.
pub(in crate::server) async fn index(Json(body): Json<Value>) -> Response {
    let Some(record) = Record::from_json(&body) else { return api_error(StatusCode::BAD_REQUEST, "id is missing or too long") };
    match tokio::task::spawn_blocking(move || append(&data_file(), &record, MAX_FILE_BYTES, COMPACT_TO_BYTES)).await {
        Ok(Ok(())) => Json(json!({"ok": true})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `DELETE /api/studio/sessions/index?id=` drops the copy of one session; `?all=1` drops every copy.
pub(in crate::server) async fn remove(Query(query): Query<HashMap<String, String>>) -> Response {
    let id = query.get("id").cloned().unwrap_or_default();
    let all = query.get("all").is_some_and(|v| v == "1");
    if id.is_empty() && !all {
        return api_error(StatusCode::BAD_REQUEST, "id is missing");
    }
    match tokio::task::spawn_blocking(move || if all { forget_all(&data_file()) } else { forget(&data_file(), &id) }).await {
        Ok(Ok(())) => Json(json!({"ok": true})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `GET /api/studio/sessions/search?q=&limit=` answers `{hits: [{id, title, kind, updated, score, snippet}]}`.
pub(in crate::server) async fn search(Query(query): Query<HashMap<String, String>>) -> Response {
    let q = query.get("q").cloned().unwrap_or_default();
    let limit = query.get("limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
    match tokio::task::spawn_blocking(move || search_in(&data_file(), &q, limit)).await {
        Ok(hits) => Json(json!({"hits": hits})).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, title: &str, text: &str, updated: u64) -> Record {
        Record { id: id.into(), title: title.into(), text: text.into(), kind: "chat".into(), updated }
    }

    fn temp_file() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-sessions-{}-{}", std::process::id(), new_id("t")));
        fs::create_dir_all(&dir).unwrap();
        dir.join("sessions.jsonl")
    }

    #[test]
    fn ranking_prefers_rare_words_and_titles() {
        let docs = vec![
            rec("a", "Cooking", "pasta with tomato sauce and basil", 1),
            rec("b", "Rust borrow checker", "lifetimes and ownership", 2),
            rec("c", "Misc", "the borrow of a book from the library", 3),
        ];
        let hits = rank(&docs, "borrow checker");
        assert_eq!(docs[hits[0].index].id, "b");
        assert_eq!(hits.len(), 2);
        assert!(rank(&docs, "zzzz").is_empty());
        assert!(rank(&docs, "a").is_empty(), "one-letter words are ignored");
    }

    #[test]
    fn snippets_surround_the_match() {
        let text = format!("{} needle {}", "x ".repeat(100), "y ".repeat(100));
        let s = snippet(&text, "NEEDLE");
        assert!(s.contains("needle") && s.starts_with("...") && s.ends_with("..."), "{s}");
        assert!(s.chars().count() < 200);
    }

    #[test]
    fn append_keeps_the_newest_record_and_search_finds_it() {
        let file = temp_file();
        append(&file, &rec("1", "First", "alpha", 1), MAX_FILE_BYTES, COMPACT_TO_BYTES).unwrap();
        append(&file, &rec("1", "First", "beta gamma", 2), MAX_FILE_BYTES, COMPACT_TO_BYTES).unwrap();
        append(&file, &rec("2", "Second", "gamma", 3), MAX_FILE_BYTES, COMPACT_TO_BYTES).unwrap();
        assert!(search_in(&file, "alpha", 10).is_empty(), "the older text of session 1 is gone");
        let hits = search_in(&file, "gamma", 10);
        assert_eq!(hits.len(), 2);
        forget(&file, "1").unwrap();
        assert_eq!(search_in(&file, "gamma", 10).len(), 1);
        forget_all(&file).unwrap();
        assert!(search_in(&file, "gamma", 10).is_empty());
        forget_all(&file).unwrap();
    }

    #[test]
    fn the_file_is_compacted_past_the_cap_and_old_sessions_go_first() {
        let file = temp_file();
        for i in 0..30u64 {
            append(&file, &rec(&format!("s{i}"), "t", &"word ".repeat(100), i), 4000, 2500).unwrap();
        }
        let left = latest(read_all(&file));
        assert!(!left.is_empty() && left.len() < 30);
        assert_eq!(left[0].id, "s29", "the newest session stays");
        assert!(fs::metadata(&file).unwrap().len() < 4000 + 700);
    }

    #[test]
    fn records_need_an_id() {
        assert!(Record::from_json(&json!({"title": "x"})).is_none());
        let r = Record::from_json(&json!({"id": "i", "kind": "other", "text": "x".repeat(MAX_TEXT_CHARS + 5)})).unwrap();
        assert_eq!(r.kind, "chat");
        assert_eq!(r.text.chars().count(), MAX_TEXT_CHARS);
    }
}
