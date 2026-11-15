//! The command of the Code page in the window: `/autoapprove on|off` answers in the box of that
//! page instead of going to the model, says what the switch stands for, and writes the
//! preference the next start reads (`case 'autoapprove'` of `code.js`).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::code::{CodeMsg, Item, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced::Size;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

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
    Store { theme: theme.into(), ..Default::default() }
        .save(&path)
        .unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    // the page draws its list only over an open folder, so the folder is opened here the way
    // `code_page.rs` opens it
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace {
        root: "/home/marco/projects/fllm".into(),
        name: "fllm".into(),
        branch: Some("main".into()),
    }))));
    let tools: Vec<Tool> = [("read_file", Permission::Read), ("bash", Permission::Command)]
        .into_iter()
        .map(|(n, p)| Tool::server(n, p))
        .collect();
    let _ = app.update(Message::Code(CodeMsg::ToolsLoaded(Ok((tools, serde_json::json!([]))))));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 9000.0), app.view())
}

/// Writes the words into the box of the Code page and sends them, as the buttons of the page do.
fn say(app: &mut App, words: &str) {
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::Edit(Edit::Paste(
        Arc::new(words.to_string()),
    )))));
    let _ = app.update(Message::Code(CodeMsg::Send));
}

/// The command is answered in the box of the Code page — with no workspace and no model, as the
/// page answers it out of its own state — and the notice says what the switch stands for now.
#[test]
fn the_command_turns_the_switch_and_says_what_it_means() {
    let (mut app, _dir) = app("dark");
    assert!(app.code().auto_approve, "the switch starts on, as on the page");
    assert!(app.store().prefs.auto_approve, "and the preference starts with it");

    say(&mut app, "/autoapprove off");
    assert!(!app.code().auto_approve, "the switch is off");
    assert!(!app.store().prefs.auto_approve, "and saved for the next start");
    assert!(
        app.code().editor.text().is_empty(),
        "the box goes away before the command runs, as the page clears it first"
    );

    // the notice is in the list of the page …
    assert!(
        app.code()
            .items
            .iter()
            .any(|item| matches!(item, Item::Note(text) if text.starts_with("Automatic approval is off:"))),
        "the words of the notice are in the view of the page"
    );
    // … and the page draws it, with both of its lines
    let mut ui = sim(&app);
    ui.find("Automatic approval is off: every command asks again.\n/autoapprove on · /autoapprove off")
        .expect("the notice, both of its lines as the window draws them");
    drop(ui);

    // any other word changes nothing and says the same as the last time
    let items = app.code().items.len();
    say(&mut app, "/autoapprove banana");
    assert!(!app.code().auto_approve, "only on and off change it");
    assert_eq!(app.code().items.len(), items + 1, "the notice stands anyway");

    // the word of the command is lower-cased on the page, so its own case stands too
    say(&mut app, "/AutoApprove on");
    assert!(app.code().auto_approve, "the page lower-cases the name of the command");
    assert!(app.store().prefs.auto_approve, "and saves it");
    let mut ui = sim(&app);
    ui.find(
        "Automatic approval is on: harmless commands (reading, building, testing) run without a question; everything else still asks.\n/autoapprove on · /autoapprove off",
    )
    .expect("the notice of the switch that stands, both of its lines");
    drop(ui);
}

/// The preference is what the next start reads: the words of the switch are written with the
/// store the window loads again.
#[test]
fn the_preference_is_what_the_next_start_reads() {
    let (mut app, dir) = app("dark");
    say(&mut app, "/autoapprove off");
    assert!(!app.store().prefs.auto_approve, "and the words go with the store as the switch is set");

    let store = Store::load(&dir.path().join("chats.json"));
    assert!(!store.prefs.auto_approve, "the words of the switch are on disk");
}

/// The command belongs to the Code page: in the box of the chat it is a question like any other,
/// and a word that only starts like the command is another command of the page entirely.
#[test]
fn the_command_belongs_to_the_code_page() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Chat));
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new(
        "/autoapprove off".to_string(),
    )))));
    let _ = app.update(Message::Send);
    assert!(app.code().auto_approve, "the chat does not know the command");
    assert!(app.code().items.is_empty(), "and writes no notice");

    let _ = app.update(Message::Go(View::Code));
    say(&mut app, "/autoapprover off");
    assert!(app.code().auto_approve, "another command with the same beginning");
    assert!(
        !app.code().items.iter().any(|item| matches!(item, Item::Note(_))),
        "and no notice for it either"
    );
}
