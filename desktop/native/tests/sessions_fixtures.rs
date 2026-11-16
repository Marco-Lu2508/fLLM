//! Row 79: the sessions of a folder and the arithmetic of the compaction against the page's own
//! functions (`tests/fixtures/sessions_fixtures.js` runs them in a vm).

use fllm_app::api::WireMessage;
use fllm_app::code::Saved;
use fllm_app::sessions::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sessions_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

/// A view item of the page as the window keeps it.
fn saved_of(item: &Value) -> Saved {
    let text = |k: &str| item[k].as_str().unwrap_or("").to_string();
    match item["kind"].as_str().unwrap() {
        "user" => Saved::User { text: text("text"), sent: None },
        "assistant" => Saved::Assistant { text: text("content") },
        "notice" => Saved::Note { text: text("text") },
        _ => Saved::Call {
            id: "c".into(),
            name: text("name"),
            args: item["args"].to_string(),
            permission: "read".into(),
            // the page's `ok` is the only status the window keeps besides an error
            ok: item["status"] == "ok",
            output: text("output"),
            sent: None,
            children: item["children"].as_array().map(|c| c.iter().map(saved_of).collect()).unwrap_or_default(),
            agent: None,
            gone: false,
        },
    }
}

fn record(id: &str, title: &str, view: &[Value]) -> Record {
    Record { id: id.into(), title: title.into(), title_locked: false, time: 0.0, items: view.iter().map(saved_of).collect(), todos: vec![], marks: vec![], compaction: None }
}

#[test]
fn a_title_is_made_the_way_the_page_makes_it() {
    let d = fixtures();
    assert!(d["heuristic"].as_array().unwrap().len() >= 10);
    for row in d["heuristic"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let first = match text {
            "__none__" => None,
            "__summary_then_real__" => Some(row["second"].as_str().unwrap().to_string()),
            t if t.starts_with("[Summary") => None,
            t => Some(t.to_string()),
        };
        // a summary is skipped by the first-message rule; the window finds the message the same way
        let items: Vec<Saved> = match text {
            "__none__" => vec![],
            "__summary_then_real__" => vec![Saved::User { text: "[Summary x".into(), sent: None }, Saved::User { text: "Real question".into(), sent: None }],
            t => vec![Saved::User { text: t.to_string(), sent: None }],
        };
        let _ = first;
        assert_eq!(heuristic_title(first_user_text(&items).as_deref()), row["title"].as_str().unwrap(), "heuristicTitle({text:?})");
    }
    assert!(d["clean"].as_array().unwrap().len() >= 15);
    for row in d["clean"].as_array().unwrap() {
        assert_eq!(clean_title(row["text"].as_str().unwrap()), row["title"].as_str().unwrap(), "cleanTitle({:?})", row["text"]);
    }
}

#[test]
fn the_list_is_searched_by_every_word() {
    let d = fixtures();
    let sessions: Vec<Record> = d["filter_sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| record(s["id"].as_str().unwrap(), s["title"].as_str().unwrap(), s["view"].as_array().unwrap()))
        .collect();
    for row in d["filter"].as_array().unwrap() {
        let got: Vec<&str> = filter_sessions(&sessions, row["query"].as_str().unwrap()).iter().map(|s| s.id.as_str()).collect();
        let want: Vec<&str> = row["ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(got, want, "filterSessions({:?})", row["query"]);
    }
}

#[test]
fn the_first_change_after_a_mark_is_the_oldest_newer_one() {
    let d = fixtures();
    assert!(d["change"].as_array().unwrap().len() >= 18);
    for row in d["change"].as_array().unwrap() {
        let ids: Vec<u64> = row["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let mark = row["mark"].as_u64().unwrap_or(0);
        assert_eq!(first_change_after(&ids, mark), row["got"].as_u64(), "firstChangeAfter({ids:?}, {mark})");
    }
}

#[test]
fn an_export_is_the_markdown_of_the_page() {
    let d = fixtures();
    for row in d["markdown"].as_array().unwrap() {
        let items: Vec<Saved> = row["view"].as_array().unwrap().iter().map(saved_of).collect();
        let got = export_markdown(row["title"].as_str().unwrap(), row["time"].as_f64().unwrap(), &items, row["root"].as_str().unwrap());
        assert_eq!(got, row["md"].as_str().unwrap(), "exportMarkdown of {:?}", row["title"]);
    }
    for row in d["clip"].as_array().unwrap() {
        assert_eq!(clip_middle(row["text"].as_str().unwrap(), row["max"].as_u64().unwrap() as usize), row["got"].as_str().unwrap());
    }
    for row in d["file_safe"].as_array().unwrap() {
        assert_eq!(file_safe(row["text"].as_str().unwrap()), row["got"].as_str().unwrap(), "fileSafe({:?})", row["text"]);
    }
}

fn msgs(roles: &str) -> Vec<WireMessage> {
    roles.chars().map(|c| WireMessage::new(match c { 'u' => "user", 'a' => "assistant", _ => "tool" }, "c")).collect()
}

#[test]
fn the_compaction_counts_and_cuts_the_way_the_page_does() {
    let d = fixtures();
    assert_eq!(d["ratio"].as_f64(), Some(COMPACT_RATIO));
    for row in d["estimate"].as_array().unwrap() {
        assert_eq!(estimate_tokens(row["text"].as_str().unwrap()) as u64, row["got"].as_u64().unwrap(), "estimateTokens({:?})", row["text"]);
    }
    for row in d["needs"].as_array().unwrap() {
        let (e, m) = (row["estimate"].as_f64().unwrap(), row["max"].as_f64().unwrap());
        // a fractional estimate does not occur in the window; the whole numbers decide
        if e.fract() == 0.0 {
            assert_eq!(needs_compaction(e as usize, m as usize, COMPACT_RATIO), row["got"].as_bool().unwrap(), "needsCompaction({e}, {m})");
        }
    }
    assert!(d["split"].as_array().unwrap().len() >= 9);
    for row in d["split"].as_array().unwrap() {
        let got = split_for_compaction(&msgs(row["roles"].as_str().unwrap()), 6, 3);
        assert_eq!(got.map(|s| s as u64), row["start"].as_u64(), "splitForCompaction({:?})", row["roles"]);
    }
    for row in d["split_custom"].as_array().unwrap() {
        let got = split_for_compaction(&msgs(row["roles"].as_str().unwrap()), row["keepLast"].as_u64().unwrap() as usize, row["minHead"].as_u64().unwrap() as usize);
        assert_eq!(got.map(|s| s as u64), row["start"].as_u64(), "splitForCompaction({:?}, custom)", row["roles"]);
    }
    assert!(d["build"].as_array().unwrap().len() >= 18);
    for row in d["build"].as_array().unwrap() {
        let todos: Vec<SavedTodo> = row["todos"].as_array().unwrap().iter().map(|t| SavedTodo { content: t["content"].as_str().unwrap().into(), status: t["status"].as_str().unwrap().into() }).collect();
        let files: Vec<String> = row["files"].as_array().unwrap().iter().map(|f| f.as_str().unwrap().to_string()).collect();
        let first_is_user = row["tailRoles"].as_array().unwrap().first().map(|r| r == "user").unwrap_or(true);
        let got: Vec<(String, String)> = build_prefix("  A summary.\nline two  ", &todos, &files, first_is_user).into_iter().map(|m| (m.role, m.content)).collect();
        let want: Vec<(String, String)> = row["prefix"].as_array().unwrap().iter().map(|m| (m[0].as_str().unwrap().into(), m[1].as_str().unwrap().into())).collect();
        assert_eq!(got, want, "buildCompactedMessages({:?}, {:?}, tail {:?})", row["todos"], row["files"], row["tailRoles"]);
    }
}

#[test]
fn an_older_file_of_one_conversation_per_folder_is_read_as_one_session() {
    let old = r#"{"/w":[{"kind":"user","text":"hi","sent":null},{"kind":"assistant","text":"hello"}],"/empty":[]}"#;
    let all: Sessions = serde_json::from_str(old).unwrap();
    assert_eq!(all.of("/w").len(), 1);
    assert_eq!(all.of("/w")[0].items.len(), 2);
    assert!(all.of("/empty").is_empty());
    let new = serde_json::to_string(&all).unwrap();
    let again: Sessions = serde_json::from_str(&new).unwrap();
    assert_eq!(again.of("/w")[0].items.len(), 2);
}

#[test]
fn the_list_keeps_the_newest_and_puts_a_saved_one_on_top() {
    let mut all = Sessions::default();
    for i in 0..(MAX_SESSIONS + 5) {
        all.put("/w", Record { id: format!("s{i}"), title: String::new(), title_locked: false, time: i as f64, items: vec![], todos: vec![], marks: vec![], compaction: None });
    }
    assert_eq!(all.of("/w").len(), MAX_SESSIONS);
    assert_eq!(all.of("/w")[0].id, format!("s{}", MAX_SESSIONS + 4));
    all.put("/w", Record { id: "s10".into(), title: "again".into(), title_locked: true, time: 0.0, items: vec![], todos: vec![], marks: vec![], compaction: None });
    assert_eq!(all.of("/w")[0].id, "s10");
    assert_eq!(all.of("/w").iter().filter(|s| s.id == "s10").count(), 1);
    all.remove("/w", "s10");
    assert!(all.of("/w").iter().all(|s| s.id != "s10"));
}
