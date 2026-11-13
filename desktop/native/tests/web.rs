//! The web tools of a chat: the chip in the composer, the tools in the request, the line in the
//! prompt and the sources under an answer (the Studio's rag.js, tools.js and artifacts sources).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use fllm_app::tools::Source;
use iced_test::Simulator;

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

fn app_with_context(max_context: u64) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", max_context))));
    (app, dir)
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn saved(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

fn names(app: &App) -> Vec<String> {
    app.offered_tools()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|d| d.pointer("/function/name").and_then(|n| n.as_str()).map(String::from))
        .collect()
}

#[test]
fn the_web_chip_offers_the_tools_and_describes_them() {
    // a big context, so the prompt is the full one (not the compact with its short tool lines)
    let (mut app, _dir) = app_with_context(65536);
    assert!(app.offered_tools().is_some(), "the memory of the settings is offered");
    assert!(!app.system_prompt().contains("Web tools (web_search"), "without the chip the prompt says nothing about the web");
    {
        let mut ui = sim(&app);
        ui.find("Allow web access").expect("the chip of the Studio in the composer");
    }

    let _ = app.update(Message::ToggleWeb);
    assert_eq!(
        app.toast(),
        Some("The model may now search the web and read pages when it needs to. Questions it can answer itself are not searched."),
        "the line of the Studio"
    );
    assert_eq!(names(&app), ["web_search", "web_fetch", "memory"], "the tools of the request");
    let prompt = app.system_prompt();
    assert!(
        prompt.contains("Web tools (web_search, web_fetch): use them only for current or changing facts"),
        "and the prompt describes them: {prompt}"
    );

    // switched off again: the tools and their line are gone
    let _ = app.update(Message::ToggleWeb);
    assert_eq!(names(&app), ["memory"], "only the memory is left");
    assert!(!app.system_prompt().contains("Web tools (web_search"), "no line about the web");
}

#[test]
fn a_tiny_context_gets_no_tools_at_all() {
    // a context under 2,048 tokens has no room for the schemas (the Studio's ctOfferedFlags)
    let (mut app, _dir) = app_with_context(2047);
    let _ = app.update(Message::ToggleWeb);
    assert!(app.offered_tools().is_none(), "no tools for a tiny context");
    assert!(!app.system_prompt().contains("Memory tool (memory)"), "and the prompt says so");
}

#[test]
fn the_sources_of_an_answer_are_shown_and_open() {
    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "The release is out, see [1].");
    answer.sources = vec![Source { n: 1, title: "The fLLM blog".into(), url: "https://example.com/blog".into(), fetched: false }];
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "news?"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Sources").expect("the line under the answer");
        ui.find("[1] The fLLM blog").expect("the numbered link");
    }
    // a click opens the address in the browser of the computer
    let mut ui = sim(&app);
    ui.click("[1] The fLLM blog").expect("the link is clickable");
    let msgs: Vec<Message> = ui.into_messages().collect();
    assert!(msgs.iter().any(|m| matches!(m, Message::OpenUrl(url) if url == "https://example.com/blog")), "{msgs:?}");

    // the sources are part of the chat, they come back when it is opened again
    let after = saved(&dir);
    assert_eq!(after.chats[0].messages[1].sources.len(), 1, "kept with the chat");
}

/// The composer with both chips and the sources under an answer as PNGs into the snapshot dir.
#[test]
fn the_web_views_are_painted() {
    let (mut app, _dir) = app_with_context(8192);
    let _ = app.update(Message::ToggleWeb);
    save_png(&app, "web-chips");

    let mut store = Store::default();
    let mut answer = rec(Role::Assistant, "Two places to read more.");
    answer.sources = vec![
        Source { n: 1, title: "The fLLM blog".into(), url: "https://example.com/blog".into(), fetched: false },
        Source { n: 2, title: "Release notes".into(), url: "https://other.test/notes".into(), fetched: true },
    ];
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "news?"), answer]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    save_png(&app, "web-sources");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
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
