//! The research run: the whole run against a small server (plan, search, read, notes, report), the
//! fallback when the server has no web tools, and the window — chip, progress and log (researchRun).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use futures::StreamExt;
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The model, the plan, the reflection, the notes, the report, the search and the fetch — the
/// routes the run walks through, with what it asked for in the log.
async fn serve(with_web: bool) -> (String, Arc<Mutex<Vec<String>>>) {
    serve_reply(with_web, false).await
}

/// The same server, with a plan answer that cannot be read as queries.
async fn serve_garbage_plan() -> (String, Arc<Mutex<Vec<String>>>) {
    serve_reply(true, true).await
}

async fn serve_reply(with_web: bool, garbage_plan: bool) -> (String, Arc<Mutex<Vec<String>>>) {
    let log: Arc<Mutex<Vec<String>>> = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
                seen.lock().unwrap().push(format!("{method} {path} {body}"));
                let (status, reply, kind) = match (method, path) {
                    ("POST", "/api/studio/web/search") if with_web => (
                        "200 OK",
                        json!({"results": [
                            {"title": "Gold hits a record", "url": "https://example.com/gold-record",
                             "snippet": "Gold rose to a record as rate cuts are priced in."},
                            {"title": "Why gold keeps climbing", "url": "https://example.com/why",
                             "snippet": "Real yields and central bank buying."}
                        ]})
                        .to_string(),
                        "application/json",
                    ),
                    ("POST", "/api/studio/web/fetch") if with_web => (
                        "200 OK",
                        json!({"text": "Gold climbed to a record on Friday. Updated November 12, 2026. Traders priced in rate cuts and central bank buying."}).to_string(),
                        "application/json",
                    ),
                    ("POST", "/api/studio/web/search" | "/api/studio/web/fetch") => ("404 Not Found", "{}".to_string(), "application/json"),
                    ("POST", "/v1/chat/completions") => {
                        let parsed: Value = serde_json::from_str(body).unwrap_or_default();
                        let sys = parsed["messages"]
                            .as_array()
                            .and_then(|m| m.iter().find(|m| m["role"] == "system"))
                            .and_then(|m| m["content"].as_str())
                            .unwrap_or("");
                        let usr = parsed["messages"]
                            .as_array()
                            .and_then(|m| m.iter().find(|m| m["role"] == "user"))
                            .and_then(|m| m["content"].as_str())
                            .unwrap_or("");
                        let has_schema = parsed.get("response_format").is_some();
                        let content = if has_schema && usr.starts_with("Question: ") {
                            // the reflection of the notes
                            json!({"notes_ok": true, "more": []}).to_string()
                        } else if has_schema {
                            // the plan of the queries (or a plan that cannot be read)
                            if garbage_plan { String::new() } else { "- \"gold price\"".to_string() }
                        } else if sys.contains("research notes") {
                            "A record on Friday as rate cuts are priced in.".to_string()
                        } else if sys.contains("trusted research REPORT") {
                            "The report says gold rose.".to_string()
                        } else {
                            "practice answer".to_string()
                        };
                        (
                            "200 OK",
                            json!({"choices": [{"message": {"role": "assistant", "content": content}}]}).to_string(),
                            "application/json",
                        )
                    }
                    _ => ("404 Not Found", "{}".to_string(), "application/json"),
                };
                let _ = sock
                    .write_all(
                        format!("HTTP/1.1 {status}\r\ncontent-type: {kind}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}", reply.len()).as_bytes(),
                    )
                    .await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_research_run_walks_its_phases_and_prepares_the_report() {
    let (base, log) = serve(true).await;
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let prepared = fllm_app::research::prepare(&base, "m", 128_000, "what about gold", false, tx).await.expect("the run prepares the report");
    assert!(!prepared.fallback, "the run finished");
    assert!(prepared.system.starts_with("You write research reports from notes."), "the system of the report: {:?}", prepared.system);
    assert!(prepared.user.starts_with("Question: what about gold"), "the question: {:?}", prepared.user);
    assert!(prepared.user.contains("Notes:"), "the report answers from the notes: {:?}", prepared.user);
    assert!(!prepared.user.contains("MCP"), "the report does not ride on the MCP layer");
    // the pages of the sources, numbered in the run
    assert_eq!(prepared.sources.len(), 2, "both pages made it: {:?}", prepared.sources);
    assert_eq!((prepared.sources[0].0, prepared.sources[0].1.as_str()), (1, "Gold hits a record"));
    assert_eq!(prepared.sources[0].2, "https://example.com/gold-record", "the first source");
    // the window had its steps
    let ui = &prepared.ui;
    assert_eq!(ui.queries, ["gold price"], "the plan of the server became the searches: {:?}", ui.queries);
    assert_eq!(ui.pages.len(), 2, "the two pages of the run");
    assert!(!ui.events.is_empty(), "the log of the steps: {:?}", ui.events);
    assert!(!ui.running, "the run is over");
    assert!(ui.seconds < 600, "the run counted its seconds: {}", ui.seconds);
    assert!(ui.notice.is_none(), "no notice for a run that worked");
    // what the server saw: the plan, the search, the fetch, the notes, the reflection, the report
    let seen = log.lock().unwrap().clone();
    assert!(seen.iter().any(|r| r.starts_with("POST /v1/chat/completions") && r.contains("response_format") && !r.contains("Question:")), "the plan was asked: {seen:?}");
    assert!(seen.iter().any(|r| r.contains("POST /api/studio/web/search") && r.contains("gold price")), "the search ran: {seen:?}");
    assert!(seen.iter().any(|r| r.contains("POST /api/studio/web/fetch") && r.contains("example.com/gold-record")), "the page was read: {seen:?}");
    assert!(seen.iter().any(|r| r.contains("take research notes")), "the notes were taken: {seen:?}");
    assert!(seen.iter().any(|r| r.contains("You check research notes")), "the notes were reflected: {seen:?}");
    // the report itself is streamed by the window (researchStream): the wanted size, cut to what
    // is left of the context after the margin — and 0 when too little is left
    assert_eq!(fllm_app::research::research_max_tokens(128_000, 72, 1800), 1800, "a big context gets the wanted size");
    assert_eq!(fllm_app::research::research_max_tokens(4_096, 3_800, 1800), 0, "a context without room gives nothing (stream_research lifts it to 200)");
    // the progress went over the channel of the run
    let steps: Vec<_> = rx.collect().await;
    assert!(steps.len() >= 3, "the window saw the phases: {}", steps.len());
    assert!(steps.iter().any(|u| u.phase == "reading pages"), "the reading phase was shown");
    assert!(steps.iter().any(|u| u.phase == "writing the report"), "the writing phase was shown");
}

/// A plan that cannot be read: the run searches the question itself, and keeps going.
#[tokio::test]
async fn an_unreadable_plan_falls_back_to_the_question_itself() {
    let (base, log) = serve_garbage_plan().await;
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let prepared = fllm_app::research::prepare(&base, "m", 128_000, "what about gold", false, tx)
        .await
        .expect("the run continues with the question");
    assert_eq!(prepared.ui.queries, ["what about gold"], "the question itself is searched: {:?}", prepared.ui.queries);
    assert!(
        prepared.ui.events.iter().any(|(_, e)| e.contains("plan was unreadable")),
        "the run says why it searched the question: {:?}",
        prepared.ui.events
    );
    let seen = log.lock().unwrap().clone();
    assert!(seen.iter().any(|r| r.contains("POST /api/studio/web/search") && r.contains("what about gold")), "the search ran: {seen:?}");
    // the sources of the report: numbered markdown, pages without a number are not sources
    assert_eq!(
        fllm_app::research::sources_markdown(&[(1, "Gold hits a record".into(), "https://example.com/gold-record".into(), String::new())]),
        "\n\n## Sources\n\n1. [Gold hits a record](https://example.com/gold-record)\n",
        "the section of the Studio"
    );
    assert_eq!(fllm_app::research::sources_markdown(&[(0, "not cited".into(), "https://example.com/x".into(), String::new())]), "", "a page without a number is no source");
    let _ = rx.collect::<Vec<_>>().await;
}

/// The server without web tools: the run stops before reading anything and the model answers on
/// its own, with the notice of the Studio's ResearchWebError.
#[tokio::test]
async fn research_without_web_tools_falls_back_with_the_notice() {
    let (base, log) = serve(false).await;
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let prepared = fllm_app::research::prepare(&base, "m", 128_000, "what about gold", false, tx).await.expect("the run answers with the fallback");
    assert!(prepared.fallback, "the run gave up");
    assert!(prepared.system.is_empty() && prepared.user.is_empty(), "no report was written");
    assert_eq!(prepared.sources.len(), 0, "no sources without a report");
    let notice = prepared.notice.as_deref().expect("the notice of the run");
    assert!(notice.starts_with("Research could not run: "), "the notice of the Studio: {notice:?}");
    assert!(notice.contains("no web tools"), "the 404 explains itself: {notice:?}");
    assert!(notice.ends_with("The model answers on its own instead."), "the notice tells what happens instead");
    assert_eq!(prepared.ui.notice.as_deref(), Some(notice), "the window shows the notice");
    let seen = log.lock().unwrap().clone();
    assert!(!seen.iter().any(|r| r.contains("trusted research REPORT")), "no report was asked for: {seen:?}");
    assert!(!seen.iter().any(|r| r.contains("take research notes")), "no notes were taken: {seen:?}");
    let _ = rx.collect::<Vec<_>>().await;
}

fn rec(role: Role, content: &str) -> Record {
    Record {
        role,
        content: content.into(),
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

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view())
}

/// The chip of the research mode with its toast, and the progress of a run over an answer.
#[test]
fn the_research_chip_and_the_progress_line_are_in_the_window() {
    let (mut app, _dir) = app_with(Store::default());
    {
        let mut ui = sim(&app);
        ui.find("Research").expect("the chip of the Studio in the composer");
    }
    let _ = app.update(Message::ResearchToggle);
    assert!(app.research_state(), "the chip is on");
    let toast = app.toast().unwrap_or_default();
    assert!(toast.starts_with("Research:"), "the toast explains the mode: {toast:?}");

    // an answer whose run is going shows its step over the text
    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "practice answer");
    let mut run = fllm_app::research::ResearchUi::new(1_700_000_000_000);
    run.queries.push("gold price".into());
    run.pages.push(fllm_app::research::Page {
        n: 0,
        title: "Gold hits a record".into(),
        url: "https://example.com/gold-record".into(),
        date: "November 12, 2026".into(),
        ok: true,
        round: 1,
        error: String::new(),
        snippet_only: true,
        notes: String::new(),
    });
    run.events.push((1.0, "Planning the search queries.".into()));
    answer.research = Some(run);
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "what about gold"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Research · round 1 of 2 · step 1 of 5: planning").expect("the progress over the answer");
    }
    // the log opens with its sections
    let _ = app.update(Message::ResearchLogToggle(1));
    {
        let mut ui = sim(&app);
        ui.find("Queries").expect("the section of the searches");
        ui.find("1. gold price").expect("the query of the run");
        ui.find("Pages").expect("the section of the pages");
        ui.find("Gold hits a record").expect("the page of the run");
        ui.find("search snippet only").expect("the run says how the page came");
        ui.find("Steps").expect("the section of the steps");
        ui.find("1 s  Planning the search queries.").expect("the step of the log");
    }
    // kept with the chat
    let after = super_persisted(&dir);
    let run = after.chats[0].messages[1].research.clone().expect("the run is kept with the chat");
    assert!(run.log_open, "the state of the log is kept too");

    // stopped and noticed runs say so above the answer
    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "");
    let mut run = fllm_app::research::ResearchUi::new(1_700_000_000_000);
    run.running = false;
    run.stopped = true;
    answer.research = Some(run);
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "q"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Research stopped.").expect("the line of a stopped run");
    }

    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "practice answer");
    let mut run = fllm_app::research::ResearchUi::new(1_700_000_000_000);
    run.running = false;
    run.notice = Some("Research could not run: no web tools. The model answers on its own instead.".into());
    answer.research = Some(run);
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "q"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Research could not run: no web tools. The model answers on its own instead.").expect("the notice over the answer");
    }
}

/// Esc stops a research run that is still going (the same as the stop button).
#[test]
fn esc_stops_a_running_research_run() {
    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "");
    let run = fllm_app::research::ResearchUi::new(1_700_000_000_000);
    answer.research = Some(run);
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "q"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Research · round 1 of 2 · step 1 of 5: planning").expect("the run is going");
    }
    let _ = app.update(Message::Esc);
    {
        let mut ui = sim(&app);
        ui.find("Research stopped.").expect("Esc stopped it like the stop button");
    }
    // and nothing happens when nothing runs
    let _ = app.update(Message::Esc);
    {
        let mut ui = sim(&app);
        ui.find("Research stopped.").expect("still stopped, not undone");
    }
}

fn super_persisted(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/web.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| "/opt/data/cache/scratch/fLLM/desktop/native/ui-snapshots".into()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}

/// The chip of the research mode in the composer, and an answer with its run, as PNGs.
#[test]
fn the_research_views_are_painted() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::ResearchToggle);
    save_png(&app, "research-chip");

    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "The report says gold rose as rate cuts are priced in.\n\n## Summary\nGold hit a record.");
    let mut run = fllm_app::research::ResearchUi::new(1_700_000_000_000);
    run.running = false;
    run.seconds = 12;
    run.queries.push("gold price".into());
    run.queries.push("gold outlook".into());
    run.pages.push(fllm_app::research::Page {
        n: 1,
        title: "Gold hits a record".into(),
        url: "https://example.com/gold-record".into(),
        date: "November 12, 2026".into(),
        ok: true,
        round: 1,
        error: String::new(),
        snippet_only: false,
        notes: "A record on Friday as rate cuts are priced in.".into(),
    });
    run.events.push((1.0, "Planning the search queries.".into()));
    run.events.push((4.2, "Round 1: 16 results, 2 new pages chosen.".into()));
    run.events.push((12.0, "Done.".into()));
    run.log_open = true;
    answer.research = Some(run);
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "what about gold"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    save_png(&app, "research-report-log");
}
