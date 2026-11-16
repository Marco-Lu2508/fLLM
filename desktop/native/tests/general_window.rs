//! Settings > General: the version, *Delete all chats*, *Name new chats automatically*, and the card
//! *Start it from a terminal* (its two snippets are pinned against the page's own template in
//! `general_fixtures`).

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::generalcards::{snippets, ModelFacts};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::Value;
use std::sync::Arc;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    (app, dir)
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


fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1300.0, 3800.0), app.view())
}

#[test]
fn the_snippets_are_the_ones_of_the_page() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/general_fixtures.json")).expect("the fixtures of the page");
    let data: Value = serde_json::from_str(&raw).unwrap();
    assert!(data["cases"].as_array().unwrap().len() >= 5);
    for case in data["cases"].as_array().unwrap() {
        let m = &case["model"];
        let facts = ModelFacts {
            name: m["name"].as_str().unwrap_or_default().to_string(),
            max_context: m["max_context"].as_u64().unwrap_or(0),
            max_sequences: m["max_sequences"].as_u64().unwrap_or(0),
            aliases: m["aliases"].as_array().map(|a| a.iter().map(|s| s.as_str().unwrap().to_string()).collect()).unwrap_or_default(),
        };
        let (cli, toml) = snippets(&facts);
        assert_eq!(cli, case["cli"].as_str().unwrap(), "genCliPre of {m}");
        assert_eq!(toml, case["toml"].as_str().unwrap(), "genTomlPre of {m}");
    }
}

#[test]
fn the_general_cards_have_the_words_of_the_page() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    for words in [
        "General",
        "Version",
        "Refresh",
        "Models",
        "Download, load and remove models.",
        "Open Models",
        "Welcome tour",
        "Keyboard shortcuts",
        "Delete all chats",
        "Delete chats",
        "Name new chats automatically",
        "After the first answer the model gives a new chat a short title.",
        "Start it from a terminal",
        "Command",
        "Starts the server the way it runs now, without this app.",
        "Settings file (fllm.toml)",
    ] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
}

#[test]
fn the_refresh_and_the_models_button_do_what_they_say() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.click("Open Models").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Go(View::Models)]), "{sent:?}");
    let _ = app.update(Message::GeneralRefresh);
    assert_eq!(app.toast(), Some("Refreshed from the server"));
}

#[test]
fn delete_all_chats_asks_first_and_then_empties_the_list() {
    let mut store = Store::default();
    store.chats.push(ChatRecord::new(0, "one", vec![rec(Role::User, "a"), rec(Role::Assistant, "b")]));
    store.chats.push(ChatRecord::new(1, "two", vec![rec(Role::User, "c"), rec(Role::Assistant, "d")]));
    store.next_id = 2;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::ChatsDeleteAsk);
    let mut ui = sim(&app);
    assert!(ui.find("Delete all chats?").is_ok());
    assert!(ui.find("2 chats will be deleted for good. Export them first if you want to keep them.").is_ok());
    drop(ui);
    let _ = app.update(Message::ChatsDeleteCancel);
    assert_eq!(app.store().chats.len(), 2, "cancel keeps them");
    let _ = app.update(Message::ChatsDeleteAsk);
    let _ = app.update(Message::ChatsDeleteOk);
    assert!(app.store().chats.is_empty());
    assert_eq!(app.toast(), Some("All chats deleted"));
    // nothing to delete: no question
    let _ = app.update(Message::ChatsDeleteAsk);
    assert!(sim(&app).find("Delete all chats?").is_err());
}

#[test]
fn the_model_names_a_new_chat_after_its_first_answer_unless_you_switched_it_off() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new("please explain borrowing".into())))));
    let _ = app.update(Message::Send);
    let id = app.store().chats[0].id;
    assert!(app.store().chats[0].auto_title, "a new chat waits for its title");
    let _ = app.update(Message::Stream(id, StreamEvent::Delta("A borrow is a reference.".into())));
    let with = app.update(Message::Stream(id, StreamEvent::Done));
    assert!(!app.store().chats[0].auto_title, "asked once");
    let _ = app.update(Message::ChatTitled(id, Ok("<think>hm</think>Title: \"Rust borrowing\"".into())));
    assert_eq!(app.store().chats[0].title, "Rust borrowing");

    // the switch off: no request, the first words stay
    let (mut app2, _dir2) = app_with(Store::default());
    let _ = app2.update(Message::AutoTitle(false));
    let _ = app2.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new("please explain borrowing".into())))));
    let _ = app2.update(Message::Send);
    let id2 = app2.store().chats[0].id;
    let _ = app2.update(Message::Stream(id2, StreamEvent::Delta("A borrow is a reference.".into())));
    let without = app2.update(Message::Stream(id2, StreamEvent::Done));
    assert_eq!(app2.store().chats[0].title, "please explain borrowing");
    assert_eq!(with.units(), without.units() + 1, "the request for the title is the one difference");
    // a title that gives the prompt away is not kept
    let _ = app2.update(Message::ChatTitled(id2, Ok("My instructions say".into())));
    assert_eq!(app2.store().chats[0].title, "please explain borrowing");
}
