//! Settings > Search in the window: the *Document search* card of `searchSettings` and the
//! *Web search engines* card that `searchengines.js` appends to the pane.

use fllm_app::app::{App, Message, View};
use fllm_app::search::{test_text, TestAnswer, DOC_HELP, ENGINES_HELP};
use fllm_app::store::Store;
use iced::Size;
use iced_test::Simulator;
use serde_json::json;

fn settings() -> iced::Settings {
    iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 4600.0), app.view())
}

/// Both cards start the way the page builds them: their titles, their descriptions as one
/// sentence each, and an empty list.
#[test]
fn the_pane_starts_the_way_the_page_builds_it() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("Document search").expect("the title of the first card");
    ui.find("Web search engines").expect("the title of the second card");
    // both descriptions are one paragraph, the way `app-item-desc` of the page is drawn — the
    // sentence is found whole, so it is not put together out of pieces beside each other
    ui.find(DOC_HELP).expect("the description of the first card as one sentence");
    ui.find(ENGINES_HELP).expect("the description of the second card as one sentence");
    ui.find("No engine yet.").expect("the list before anything was added");
    ui.find("Save").expect("the save button of the embeddings card");
    ui.find("Test").expect("the test button of the embeddings card");
    ui.find("Save engines").expect("the save button of the engines card");
    ui.find("Test a search").expect("the test button of the engines card");
    ui.find("Add").expect("the button over them");
    assert!(app.engines.is_empty(), "nothing was read yet");
}

/// The list follows its answer, Add takes what the checks allow, and Remove takes one out.
#[test]
fn the_list_follows_its_answer_and_the_buttons_work_on_it() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let answer: fllm_app::search::EngineAnswer = serde_json::from_value(json!({
        "engines": [
            {"kind": "ddg", "name": "mine", "url": "", "key": ""},
            {"kind": "searxng", "name": "", "url": "http://127.0.0.1:8888", "key": ""}
        ],
        "file": "/home/m/.fllm/search.json"
    }))
    .expect("an answer of the engines");
    let _ = app.update(Message::EngLoaded(Ok(answer)));

    let mut ui = sim(&app);
    ui.find("2 engine(s) · saved in /home/m/.fllm/search.json").expect("the line of `load()`");
    ui.find("mine").expect("the name of the first engine");
    ui.find("ddg").expect("the kind under it");
    ui.find("http://127.0.0.1:8888").expect("the address of the second one");
    drop(ui);

    // a SearXNG without an address says so and is not added
    let _ = app.update(Message::EngKind("searxng".into()));
    let _ = app.update(Message::EngAdd);
    assert_eq!(app.engines.len(), 2, "nothing was added");
    let mut ui = sim(&app);
    ui.find("A SearXNG engine needs its address.").expect("the line of `#seAdd`");
    drop(ui);

    // with an address it goes in, and the three boxes are cleared while the kind stays
    let _ = app.update(Message::EngUrl("http://10.7.0.6:8888".into()));
    let _ = app.update(Message::EngName("office".into()));
    let _ = app.update(Message::EngAdd);
    assert_eq!(app.engines.len(), 3, "the engine was added");
    assert!(app.eng_name.is_empty() && app.eng_url.is_empty(), "the boxes are cleared");
    assert_eq!(app.eng_kind, "searxng", "the kind stays where it was");

    // and the row of it is drawn
    let mut ui = sim(&app);
    ui.find("office").expect("the name of the new engine");
    ui.find("searxng · http://10.7.0.6:8888").expect("its kind and address");
    drop(ui);

    // the first one comes out again
    let _ = app.update(Message::EngRemove(0));
    assert_eq!(app.engines.len(), 2, "one was removed");

    // the test sends its one query and shows what the answer writes
    let _ = app.update(Message::EngTest);
    assert_eq!(
        app.eng_status, "Searching \"rust axum middleware\" ...",
        "the line of `#seTest` before the answer comes"
    );
    let test: TestAnswer = serde_json::from_value(json!({
        "engines": [{"ok": true, "engine": "searxng", "found": 7}],
        "results": [{"title": "Rust axum"}]
    }))
    .expect("an answer of the search");
    assert_eq!(test_text(&test), "searxng: 7 Treffer\n\n1. Rust axum");
    let _ = app.update(Message::EngTested(Ok(test)));
    let mut ui = sim(&app);
    ui.find("searxng: 7 Treffer\n\n1. Rust axum").expect("the lines of the answer");
    drop(ui);
}

/// The embeddings card reads its setting, saves what is in the box and tests the server.
#[test]
fn the_embeddings_card_reads_saves_and_tests() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    // `load()` puts the saved address in the box and says which search runs
    let _ = app.update(Message::EmbedLoaded(Ok(fllm_app::search::EmbedConfig {
        url: "http://127.0.0.1:8100".into(),
        configured: true,
    })));
    assert_eq!(app.embed_url, "http://127.0.0.1:8100", "the box holds what is saved");
    let mut ui = sim(&app);
    ui.find("Hybrid search is on.").expect("the line of `load()`");
    drop(ui);

    let _ = app.update(Message::EmbedLoaded(Ok(fllm_app::search::EmbedConfig::default())));
    let mut ui = sim(&app);
    ui.find("Keyword search only.").expect("the line for a server without an address");
    drop(ui);

    // a read that failed says so the way the page says it
    let _ = app.update(Message::EmbedLoaded(Err("no server at http://127.0.0.1:9".into())));
    let mut ui = sim(&app);
    ui.find("Could not read the setting: no server at http://127.0.0.1:9")
        .expect("the line of the catch of `load()`");
    drop(ui);

    // what is typed goes into the box, and Save sends it trimmed
    let _ = app.update(Message::EmbedUrl("  http://10.7.0.4:8100  ".into()));
    let _ = app.update(Message::EmbedTest);
    assert_eq!(app.embed_status, "Asking the server ...", "the line before the answer");

    let vectors: fllm_app::search::Vectors = serde_json::from_value(json!({
        "vectors": [[1.0, 0.0], [1.0, 0.0]],
        "dims": 2
    }))
    .expect("an answer of /api/studio/embed");
    let _ = app.update(Message::EmbedTested(Ok(vectors)));
    let mut ui = sim(&app);
    ui.find(
        "2 vector(s), 2 numbers each. Similarity of two similar sentences: 1.000.",
    )
    .expect("the line of `btnEmbedTest`");
    drop(ui);

    // a failure stands as the page writes it
    let _ = app.update(Message::EmbedSaved(Err("Could not save it.".into())));
    let mut ui = sim(&app);
    ui.find("Could not save it.").expect("the line of a failed save");
    drop(ui);
}
