//! The actions under an answer and a question: the plain copy, the copy as Markdown, reading
//! aloud, and rewind with its notice and Undo (the Studio's artifacts.js and chat.js).

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

/// A chat with four messages, open in the window.
fn open_chat() -> (App, tempfile::TempDir) {
    let mut store = Store::default();
    let mut chat = ChatRecord::new(
        0,
        "one",
        vec![
            rec(Role::User, "first"),
            rec(Role::Assistant, "answer one"),
            rec(Role::User, "second"),
            rec(Role::Assistant, "answer two"),
        ],
    );
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    (app, dir)
}

fn say(app: &mut App, text: &str) {
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new(text.into())))));
    let _ = app.update(Message::Send);
    let _ = app.update(Message::Stop);
}

#[test]
fn rewind_hides_the_rest_and_undo_brings_it_back() {
    let (mut app, dir) = open_chat();
    assert!(app.message_text(3).is_some(), "all four are there");

    let _ = app.update(Message::RewindTo(1));
    assert!(app.message_text(1).is_some(), "up to the message it was rewound to");
    assert!(app.message_text(2).is_none(), "the rest is out of the row");
    {
        let mut ui = sim(&app);
        ui.find("Rewound: 2 message(s) after this one are hidden.").expect("the notice of the Studio");
        ui.find("Undo").expect("its button");
    }
    // the hidden part is kept in the chat, not thrown away
    let after = saved(&dir);
    assert_eq!(after.chats[0].messages.len(), 2, "the row itself is shorter");
    assert_eq!(after.chats[0].rewinds.len(), 1, "one rewind can be taken back");
    assert_eq!(after.chats[0].rewinds[0].tail.len(), 2, "with the two messages in it");

    // the notice is no toast: a tick does not take it away
    let _ = app.update(Message::Tick);
    {
        let mut ui = sim(&app);
        ui.find("Rewound: 2 message(s) after this one are hidden.").expect("still there after a tick");
    }

    // and the last message has nothing after it
    let _ = app.update(Message::RewindTo(3));
    assert_eq!(app.toast(), Some("Nothing after this message to rewind."), "the line of the Studio");
    assert!(app.message_text(1).is_some(), "the rewind before stands");

    let _ = app.update(Message::RewindUndo);
    assert!(app.message_text(3).is_some(), "the rest is back");
    assert_eq!(app.toast(), Some("Rewind undone"), "the line of the Studio");
    let after = saved(&dir);
    assert_eq!(after.chats[0].messages.len(), 4, "the row is whole again");
    assert!(after.chats[0].rewinds.is_empty(), "nothing left to take back");
    {
        let mut ui = sim(&app);
        ui.find("Rewound: 2 message(s) after this one are hidden.").expect_err("the notice is gone");
    }
}

#[test]
fn a_new_question_keeps_what_a_rewind_hid_as_a_version() {
    let (mut app, dir) = open_chat();
    let _ = app.update(Message::RewindTo(1));
    say(&mut app, "third");
    // what was hidden is a version of the chat now (the conversation tree shows those later)
    let after = saved(&dir);
    assert!(after.chats[0].rewinds.is_empty(), "the rewind was committed");
    assert_eq!(after.chats[0].variants.len(), 1, "one version was kept");
    assert_eq!(after.chats[0].variants[0].messages.len(), 2, "with the two messages in it");
    assert_eq!(after.chats[0].next_slot, 1, "the slot behind the shown version");
    assert!(app.message_text(3).is_some(), "the new question and answer are in the row");
    {
        let mut ui = sim(&app);
        ui.find("Rewound: 2 message(s) after this one are hidden.").expect_err("the notice went with the new question");
    }
}

#[test]
fn the_buttons_of_the_answer_and_the_question_are_there() {
    let (app, _dir) = open_chat();
    {
        let mut ui = sim(&app);
        ui.find("Copy answer").expect("the copy of the answer");
        ui.find("MD").expect("the copy as Markdown");
        ui.find("Read aloud").expect("reading aloud");
        ui.find("Retry").expect("the retry");
        ui.find("Rewind").expect("rewind is offered under the answer and the question");
        ui.find("Edit").expect("the edit of the question");
        ui.find("Remember").expect("remember under the question");
    }
}

/// The bar of the answer and the notice of a rewind as PNGs into the snapshot dir of the run.
#[test]
fn the_action_views_are_painted() {
    let (mut app, _dir) = open_chat();
    save_png(&app, "actions-row");
    let _ = app.update(Message::RewindTo(1));
    save_png(&app, "actions-rewind");
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
