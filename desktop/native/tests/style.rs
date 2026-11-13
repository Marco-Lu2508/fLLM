//! The response style of a chat: the chip over the message box, its menu, the choice for the next
//! new chat and the instruction at the end of the prompt (the Studio's artifacts.js).

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

fn with_chat() -> (App, tempfile::TempDir) {
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    app_with(store)
}

#[test]
fn the_instruction_of_the_style_rides_at_the_end_of_the_prompt() {
    // the default adds nothing
    let (app, _dir) = with_chat();
    assert!(!app.system_prompt().contains("Keep answers short and to the point."), "no style by default");

    // a chosen style is appended after everything else, project instructions included
    let (mut app, dir) = with_chat();
    let _ = app.update(Message::OpenChat(0));
    let _ = app.update(Message::StylePick("concise".into()));
    assert_eq!(saved(&dir).chats[0].style, "concise", "the choice went into the chat");
    let prompt = app.system_prompt();
    assert!(
        prompt.ends_with("Keep answers short and to the point. Skip preamble and repetition."),
        "the style is the last thing in the prompt: {prompt}"
    );

    // and for a chat that has not started yet the staged one is used (the Studio's state.newStyle)
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::StylePick("formal".into()));
    assert!(app.system_prompt().ends_with("Use a formal, professional tone and precise wording."), "the staged style counts");
}

#[test]
fn the_chip_says_the_style_and_the_menu_picks_one() {
    let (mut app, dir) = with_chat();
    let _ = app.update(Message::OpenChat(0));
    {
        let mut ui = sim(&app);
        ui.find("Style").expect("the chip says just Style at first");
    }

    let _ = app.update(Message::StyleMenu);
    {
        let mut ui = sim(&app);
        for label in ["Default", "Concise", "Explanatory", "Formal", "Casual"] {
            ui.find(label).expect("every style of the Studio is in the menu");
        }
    }
    let _ = app.update(Message::StylePick("formal".into()));
    assert_eq!(saved(&dir).chats[0].style, "formal");
    {
        let mut ui = sim(&app);
        ui.find("Style: Formal").expect("the chip says the label of the Studio");
        ui.find("Concise").expect_err("the menu closed after the pick");
    }

    // without a chat the choice goes into the staging, and the next new chat gets it
    let (mut app, dir) = app_with(Store::default());
    let _ = app.update(Message::StyleMenu);
    let _ = app.update(Message::StylePick("casual".into()));
    let _ = app.update(Message::NewChat);
    // the chat is only made when the first question is sent (as in the Studio)
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new("hello".into())))));
    let _ = app.update(Message::Send);
    assert_eq!(saved(&dir).chats[0].style, "casual", "the new chat took the staged style");
    {
        let mut ui = sim(&app);
        ui.find("Style: Casual").expect("the chip follows");
    }
    assert!(app.system_prompt().ends_with("Use a relaxed, friendly, conversational tone."), "and so does the prompt");
}

#[test]
fn an_empty_or_unknown_style_is_the_default() {
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.style = String::new(); // a chat from before the feature
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    assert!(!app.system_prompt().contains("Keep answers short and to the point."), "empty means the default");
    {
        let mut ui = sim(&app);
        ui.find("Style").expect("the chip says just Style");
    }
    // an unknown id in the data behaves the same (styleOf falls back)
    let (mut app, dir) = with_chat();
    let _ = app.update(Message::OpenChat(0));
    let _ = app.update(Message::StylePick("not-a-style".into()));
    assert_eq!(saved(&dir).chats[0].style, "not-a-style", "what was picked is kept as it is");
    assert!(!app.system_prompt().contains("Keep answers short and to the point."), "but the prompt treats it as the default");
    {
        let mut ui = sim(&app);
        ui.find("Style").expect("and the chip says just Style");
    }
}

/// The chip and its open menu as a PNG into the snapshot dir of the run.
#[test]
fn the_style_menu_is_painted() {
    let (mut app, _dir) = with_chat();
    let _ = app.update(Message::OpenChat(0));
    let _ = app.update(Message::StyleMenu);
    save_png(&app, "style-menu");
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
