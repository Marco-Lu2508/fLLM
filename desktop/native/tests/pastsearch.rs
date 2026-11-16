//! Search of past chats: the index and the snippets are the page's own (`sessions.js` runs in a vm for
//! `pastsearch_fixtures`), and the card, the copy on the computer and the pushes are tested around them.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::code::Saved;
use fllm_app::pastsearch::{self, build_index, chat_doc, code_doc, merge, parse_remote, push_plan, search, snippet, tokenize, Doc, Hit, Kind};
use fllm_app::sessions::Record as CodeRecord;
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced_test::Simulator;
use serde_json::Value;
use std::collections::BTreeMap;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pastsearch_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).unwrap()
}

fn doc_of(v: &Value) -> Doc {
    Doc {
        id: v["id"].as_str().unwrap().into(),
        kind: if v["kind"] == "code" { Kind::Code } else { Kind::Chat },
        title: v["title"].as_str().unwrap().into(),
        text: v["text"].as_str().unwrap().into(),
        updated: v["updated"].as_f64().unwrap(),
        workspace: v["workspace"].as_str().unwrap_or("").into(),
    }
}

#[test]
fn the_words_are_the_ones_of_the_page() {
    for case in fixtures()["tokens"].as_array().unwrap() {
        let want: Vec<String> = case["got"].as_array().unwrap().iter().map(|w| w.as_str().unwrap().to_string()).collect();
        assert_eq!(tokenize(case["text"].as_str().unwrap_or("")), want, "ssTokenize of {}", case["text"]);
    }
}

#[test]
fn the_ranking_is_the_one_of_the_page() {
    let data = fixtures();
    let docs: Vec<Doc> = data["docs"].as_array().unwrap().iter().map(doc_of).collect();
    for case in data["searches"].as_array().unwrap() {
        let index = build_index(docs.clone());
        let got = search(&index, case["query"].as_str().unwrap(), case["limit"].as_u64().unwrap() as usize);
        let want = case["hits"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "number of hits for {}", case["query"]);
        for ((i, score), w) in got.iter().zip(want) {
            assert_eq!(index.docs[*i].id, w["id"].as_str().unwrap(), "order for {}", case["query"]);
            assert!((score - w["score"].as_f64().unwrap()).abs() < 1e-9, "score of {} for {}: {score}", w["id"], case["query"]);
        }
    }
    assert!(search(&build_index(Vec::new()), "borrow", 20).is_empty());
    assert!(data["empty_index"].as_array().unwrap().is_empty());
}

#[test]
fn the_snippets_and_the_merge_are_the_ones_of_the_page() {
    let data = fixtures();
    for case in data["snippets"].as_array().unwrap() {
        let got = snippet(case["text"].as_str().unwrap(), case["query"].as_str().unwrap(), case["width"].as_u64().unwrap() as usize);
        assert_eq!(got, case["got"].as_str().unwrap(), "ssSnippet of {} for {}", case["text"].as_str().unwrap().chars().take(30).collect::<String>(), case["query"]);
    }
    let hit = |v: &Value| Hit { id: v["id"].as_str().unwrap().into(), kind: String::new(), title: String::new(), snippet: String::new(), time: 0.0, score: 0.0, from_copy: false };
    for case in data["merges"].as_array().unwrap() {
        let local = case["local"].as_array().unwrap().iter().map(hit).collect();
        let remote = case["remote"].as_array().unwrap().iter().map(hit).collect();
        let got: Vec<String> = merge(local, remote, case["limit"].as_u64().unwrap() as usize).into_iter().map(|h| h.id).collect();
        let want: Vec<String> = case["got"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(got, want, "ssMerge");
    }
}

#[test]
fn a_code_session_is_read_like_the_page_reads_it() {
    for case in fixtures()["code_docs"].as_array().unwrap() {
        let s = &case["session"];
        // the messages of the page are the items of the window: yours, the answers; the rest has no text
        let items: Vec<Saved> = s["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| {
                let text = m["content"].as_str().unwrap_or("").to_string();
                match m["role"].as_str().unwrap() {
                    "user" => Some(Saved::User { text, sent: None }),
                    "assistant" => Some(Saved::Assistant { text }),
                    _ => None,
                }
            })
            .collect();
        let rec = CodeRecord { id: s["id"].as_str().unwrap().into(), title: s["title"].as_str().unwrap().into(), title_locked: false, time: s["time"].as_f64().unwrap(), items, todos: Vec::new(), marks: Vec::new(), compaction: None };
        let got = code_doc(&rec, case["root"].as_str().unwrap());
        let want = &case["got"];
        assert_eq!((got.id.as_str(), got.title.as_str(), got.updated, got.workspace.as_str()), (want["id"].as_str().unwrap(), want["title"].as_str().unwrap(), want["updated"].as_f64().unwrap(), want["workspace"].as_str().unwrap()));
        assert_eq!(got.text, want["text"].as_str().unwrap(), "ssCodeDoc of {}", s["id"]);
    }
    assert_eq!(fixtures()["max"].as_u64().unwrap() as usize, pastsearch::MAX_TEXT);
}

fn rec(role: Role, text: &str) -> Record {
    Record {
        role,
        content: text.into(),
        reasoning: String::new(),
        stats: None,
        passages: None,
        images: Vec::new(),
        files: Vec::new(),
        tool_calls: None,
        tool_call_id: None,
        tool_cards: Vec::new(),
    sources: Vec::new(),
    research: None,
    logprobs: Vec::new(),
    }
}

#[test]
fn a_chat_is_every_message_of_every_version_and_its_title() {
    let mut c = ChatRecord::new(7, "Parsers", vec![rec(Role::User, "first"), rec(Role::Assistant, "answer one")]);
    c.updated = 12;
    let d = chat_doc(&c);
    assert_eq!((d.id.as_str(), d.kind, d.title.as_str(), d.updated), ("7", Kind::Chat, "Parsers", 12000.0));
    assert_eq!(d.text, "first\nanswer one");
}

#[test]
fn the_server_copy_is_read_like_the_page_reads_it() {
    let hits = parse_remote(&serde_json::json!({ "hits": [{ "id": "9", "kind": "chat", "title": "T", "snippet": "s", "updated": 5, "score": 1.5 }] }));
    assert_eq!(hits.len(), 1);
    assert!(hits[0].from_copy && hits[0].id == "9" && hits[0].time == 5.0);
    assert!(parse_remote(&serde_json::json!({})).is_empty());
}

#[test]
fn a_push_sends_what_changed_a_few_at_a_time_and_marks_what_is_empty() {
    let chats: Vec<ChatRecord> = (1..=6)
        .map(|i| {
            let mut c = ChatRecord::new(i, "t", vec![rec(Role::User, &format!("question {i}"))]);
            c.updated = i as i64;
            c
        })
        .collect();
    let mut empty = ChatRecord::new(9, "", vec![]);
    empty.updated = 3;
    let mut temp = ChatRecord::new(10, "t", vec![rec(Role::User, "secret")]);
    temp.temporary = true;
    let mut all: Vec<&ChatRecord> = chats.iter().collect();
    all.push(&empty);
    all.push(&temp);
    let mut done = BTreeMap::new();
    done.insert("2".to_string(), 2000.0); // as it is now: not sent again
    done.insert("77".to_string(), 1.0); // a chat that is gone
    let plan = push_plan(&all, &[], &done, Some(5), Some((4, 2)));
    let ids: Vec<&str> = plan.send.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, ["1", "3", "4", "6"], "four at most, not the one that is answering, not the one that is done, not a temporary one");
    assert_eq!(plan.empty, vec![]);
    assert!(plan.live.contains("1") && plan.live.contains("9") && !plan.live.contains("10") && !plan.live.contains("77"));
    let plan = push_plan(&all, &[], &done, None, None);
    assert_eq!(plan.send.len(), 5, "everything that changed");
    assert_eq!(plan.empty, vec![("9".to_string(), 3000.0)], "a chat with nothing in it is marked and not sent");
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app_with(chats: Vec<ChatRecord>) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let store = Store { chats, ..Default::default() };
    store.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::SettingsQuery("skills".into()));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1200.0, 3000.0), app.view())
}

fn chat(id: u64, title: &str, text: &str) -> ChatRecord {
    let mut c = ChatRecord::new(id, title, vec![rec(Role::User, text)]);
    c.updated = id as i64;
    c
}

#[test]
fn the_card_has_the_words_of_the_page_and_finds_a_chat() {
    let (mut app, _dir) = app_with(vec![chat(1, "Borrow checker", "why does the borrow checker complain"), chat(2, "Docker", "compose up")]);
    let mut ui = sim(&app);
    for words in ["Search of past chats", pastsearch::HELP, "Keep a searchable copy on this computer", "Search chats and code sessions", "Copy everything now", "Delete the copy", "2 chats and code sessions in this app, 0 copied to the computer. "] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    drop(ui);
    let _ = app.update(Message::PastQuery("borrow".into()));
    // the search runs a quarter of a second later; an older question is not answered
    let _ = app.update(Message::PastQuery("borrow checker".into()));
    let docs = app.past_docs();
    let hits = pastsearch::local_hits(docs, "borrow checker", 12);
    let _ = app.update(Message::PastHits(1, hits.clone()));
    assert!(sim(&app).find("Nothing found.").is_err(), "the answer of the first question does not belong to the second");
    let _ = app.update(Message::PastHits(2, hits));
    let mut ui = sim(&app);
    assert!(ui.find("Borrow checker").is_ok() && ui.find("chat").is_ok());
    ui.click("why does the borrow checker complain").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::PastOpen(0)]), "{sent:?}");
    let _ = app.update(Message::PastOpen(0));
    assert_eq!(app.page(), View::Chat, "a chat of this window opens");
    // nothing found says so, only once the answer is there
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::PastQuery("zzzz".into()));
    assert!(sim(&app).find("Nothing found.").is_err());
    let _ = app.update(Message::PastHits(3, Vec::new()));
    assert!(sim(&app).find("Nothing found.").is_ok());
    // a hit of the copy only says where it is
    let _ = app.update(Message::PastHits(3, vec![Hit { id: "x".into(), kind: "code".into(), title: String::new(), snippet: "s".into(), time: 0.0, score: 1.0, from_copy: true }]));
    assert!(sim(&app).find("code - copy").is_ok() && sim(&app).find("(untitled)").is_ok());
    let _ = app.update(Message::PastOpen(0));
    assert_eq!(app.toast(), Some("This session is only in the copy on the computer; it is not in this app."));
}

#[test]
fn the_switch_the_push_and_the_deletion_of_the_copy() {
    let (mut app, _dir) = app_with(vec![chat(1, "a", "alpha")]);
    assert!(app.store().prefs.session_index);
    let _ = app.update(Message::PastIndexOn(false));
    assert!(!app.store().prefs.session_index);
    assert!(sim(&app).find("1 chats and code sessions in this app, 0 copied to the computer. The copy is switched off.").is_ok());
    assert_eq!(app.update(Message::PastPush).units(), 0, "switched off, nothing is sent");
    let _ = app.update(Message::PastIndexOn(true));
    let mut sent = BTreeMap::new();
    sent.insert("1".to_string(), 1000.0);
    let _ = app.update(Message::PastPushed(sent, 1));
    assert_eq!(app.store().prefs.session_sent.len(), 1);
    let _ = app.update(Message::PastNowDone(app.store().prefs.session_sent.clone(), 3));
    assert_eq!(app.toast(), Some("3 sessions copied"));
    // the deletion asks first
    let _ = app.update(Message::PastClearAsk);
    let mut ui = sim(&app);
    assert!(ui.find("Delete the copy?").is_ok());
    assert!(ui.find("The text copy of your chats on this computer is deleted. The chats in this app stay.").is_ok());
    drop(ui);
    let _ = app.update(Message::PastClearCancel);
    assert!(!app.store().prefs.session_sent.is_empty());
    let _ = app.update(Message::PastCleared(Err("Request failed (500)".into())));
    assert_eq!(app.toast(), Some("Request failed (500)"));
    assert!(!app.store().prefs.session_sent.is_empty(), "a refused deletion forgets nothing");
    let _ = app.update(Message::PastCleared(Ok(())));
    assert!(app.store().prefs.session_sent.is_empty());
    assert_eq!(app.toast(), Some("Copy deleted"));
}

async fn serve(refuse: &'static str) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let log: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let first = text.lines().next().unwrap_or("").to_string();
                let body = text.split_once("\r\n\r\n").map(|x| x.1).unwrap_or("").to_string();
                seen.lock().unwrap().push(format!("{first} {body}"));
                let status = if !refuse.is_empty() && first.contains(refuse) { "500 Internal Server Error" } else { "200 OK" };
                let reply = if first.starts_with("GET") { r#"{"hits":[{"id":"42","kind":"chat","title":"Copy only","snippet":"s","updated":1,"score":2}]}"# } else { "{}" };
                let _ = sock.write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_push_goes_through_the_routes_of_the_copy_and_stops_at_a_refusal() {
    let docs: Vec<Doc> = (1..=3).map(|i| Doc { id: i.to_string(), kind: Kind::Chat, title: format!("t{i}"), text: format!("text {i}"), updated: i as f64, workspace: String::new() }).collect();
    let mut done = BTreeMap::new();
    done.insert("77".to_string(), 1.0);
    done.insert("code:keep".to_string(), 1.0);
    let plan = pastsearch::PushPlan { send: docs.clone(), empty: vec![("9".into(), 3.0)], live: ["1", "2", "3", "9"].iter().map(|s| s.to_string()).collect(), done: done.clone() };
    let (base, log) = serve("").await;
    let (now, sent) = pastsearch::run_push(base.clone(), plan.clone()).await;
    assert_eq!(sent, 3);
    assert_eq!(now.keys().cloned().collect::<Vec<_>>(), ["1", "2", "3", "9", "code:keep"], "the gone chat 77 is removed, a code session is never removed here");
    let seen = log.lock().unwrap().clone();
    assert!(seen[0].starts_with("POST /api/studio/sessions/index") && seen[0].contains(r#""id":"1""#) && seen[0].contains(r#""kind":"chat""#), "{seen:?}");
    assert!(seen.iter().any(|s| s.starts_with("DELETE /api/studio/sessions/index?id=77")), "{seen:?}");
    // the second document is refused: the first one counts, nothing after it is tried
    let (base, log) = serve("POST").await;
    let (now, sent) = pastsearch::run_push(base, plan).await;
    assert_eq!(sent, 0);
    assert!(!now.contains_key("1") && now.contains_key("9"));
    assert_eq!(log.lock().unwrap().len(), 2, "the first POST and the DELETE of the gone chat");
    // the question goes to the copy too and its extra hit comes after the ones of this window
    let (base, _log) = serve("").await;
    let hits = pastsearch::search_all(base.clone(), docs.clone(), "text".into(), 12, true).await;
    assert_eq!(hits.last().map(|h| (h.id.as_str(), h.from_copy)), Some(("42", true)));
    assert!(pastsearch::search_all(base, docs, "text".into(), 12, false).await.iter().all(|h| !h.from_copy), "switched off: only this window");
    assert!(pastsearch::delete_copy("http://127.0.0.1:9".into()).await.is_err());
}
