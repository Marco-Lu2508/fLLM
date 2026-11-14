//! Temporary chats: the staging for the next new chat, the banner, the dialog before one with
//! messages is gone for good, and that the file never gets one (the Studio's artifacts.js).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

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

fn saved(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

fn say(app: &mut App, text: &str) {
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new(text.into())))));
    let _ = app.update(Message::Send);
    // the answer of the dead server never comes: stop it, as a user would
    let _ = app.update(Message::Stop);
}

const BANNER: &str = "Temporary chat: it is not saved and disappears when you close it or start another chat.";

#[test]
fn the_next_chat_is_staged_and_never_reaches_the_file() {
    let (mut app, dir) = app_with(Store::default());
    assert!(!app.staged_temporary(), "nothing is staged at first");

    let _ = app.update(Message::TempToggle);
    assert!(app.staged_temporary(), "the button stages the next chat (state.tempNext)");
    // pressing it again takes the staging back
    let _ = app.update(Message::TempToggle);
    assert!(!app.staged_temporary(), "the staging is taken back");

    let _ = app.update(Message::TempToggle);
    say(&mut app, "hello");
    assert!(!app.staged_temporary(), "the staging went into the chat");
    // a save with the chat in the store still leaves it out of the file (saveThreads)
    let _ = app.update(Message::StylePick("concise".into()));
    assert!(saved(&dir).chats.is_empty(), "the temporary chat is not written to the file");
    assert_eq!(app.chat_count(), 1, "but it is in the store while it is open");
    {
        let mut ui = sim(&app);
        ui.find(BANNER).expect("the banner of the Studio over the messages");
    }
}

#[test]
fn a_temporary_chat_with_messages_asks_before_it_is_gone() {
    let (mut app, dir) = app_with(Store::default());
    let _ = app.update(Message::TempToggle);
    say(&mut app, "hello");

    // with messages the Studio asks first
    let _ = app.update(Message::TempToggle);
    {
        let mut ui = sim(&app);
        ui.find("Close the temporary chat?").expect("the dialog of the Studio");
        ui.find("It was never saved and will be deleted now.").expect("its sentence");
        ui.find("Close it").expect("the button of the Studio");
    }
    // cancel keeps it
    let _ = app.update(Message::TempCloseCancel);
    {
        let mut ui = sim(&app);
        ui.find("Close the temporary chat?").expect_err("the dialog is closed");
        ui.find(BANNER).expect("the chat is still there");
    }
    // and the dialog itself deletes it
    let _ = app.update(Message::TempToggle);
    let _ = app.update(Message::TempCloseOk);
    assert_eq!(app.chat_count(), 0, "the chat is gone for good");
    assert!(!app.staged_temporary(), "and nothing is staged");
    {
        let mut ui = sim(&app);
        ui.find(BANNER).expect_err("no banner without a temporary chat");
        ui.find("Close the temporary chat?").expect_err("no dialog either");
    }
    let _ = app.update(Message::StylePick("concise".into())); // a save
    assert!(saved(&dir).chats.is_empty(), "nothing of it was ever written");
}

#[test]
fn switching_to_another_chat_drops_the_temporary_one() {
    let mut store = Store::default();
    let mut normal = ChatRecord::new(0, "kept", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    normal.updated = 1;
    store.chats.push(normal);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::OpenChat(0));
    // a temporary chat beside it
    let _ = app.update(Message::TempToggle);
    say(&mut app, "secret");
    assert_eq!(app.chat_count(), 2, "both are in the store");
    {
        let mut ui = sim(&app);
        ui.find(BANNER).expect("the temporary one is open");
    }

    // opening the other chat leaves it behind for good (leaveThread)
    let _ = app.update(Message::OpenChat(0));
    assert_eq!(app.chat_count(), 1, "the temporary chat is gone");
    {
        let mut ui = sim(&app);
        ui.find(BANNER).expect_err("no banner for a normal chat");
    }
    let _ = app.update(Message::StylePick("concise".into())); // a save
    assert_eq!(saved(&dir).chats.len(), 1, "and the file only holds the normal one");
    assert_eq!(saved(&dir).chats[0].title, "kept");
}

#[test]
fn the_new_chat_button_drops_a_temporary_chat_without_asking() {
    // what the Studio does: the dialog belongs to the temporary button, not to starting a new chat
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::TempToggle);
    say(&mut app, "hello");
    assert_eq!(app.chat_count(), 1);
    let _ = app.update(Message::NewChat);
    assert_eq!(app.chat_count(), 0, "the temporary chat is dropped silently");
    {
        let mut ui = sim(&app);
        ui.find("Close the temporary chat?").expect_err("no dialog here");
        ui.find(BANNER).expect_err("no banner either");
    }
}

/// The banner and the dialog as PNGs into the snapshot dir of the run.
#[test]
fn the_temporary_views_are_painted() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::TempToggle);
    say(&mut app, "hello");
    save_png(&app, "temp-banner");
    let _ = app.update(Message::TempToggle);
    save_png(&app, "temp-dialog");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}
