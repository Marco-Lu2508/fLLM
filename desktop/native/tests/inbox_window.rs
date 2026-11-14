//! The session mailbox in the window: the badge on the button, the panel over the top bar and the
//! two things a note can do — insert it into the message, or mark it read (`inbox.js` of the
//! Studio). The route itself is the engine's.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::models::{parse_notes, Note};
use iced_test::Simulator;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store::default().save(&path).unwrap();
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
    Simulator::with_size(settings, iced::Size::new(1100.0, 800.0), app.view())
}

fn note(id: u64, from: &str, to: &str, text: &str) -> Note {
    Note { id, from: from.into(), to: to.into(), text: text.into() }
}

/// The badge on the top bar, and the panel with both of its buttons (`render` of `inbox.js`).
#[test]
fn the_badge_counts_and_the_panel_shows_both_buttons() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::InboxLoaded(Ok(vec![
        note(1, "session-a", "session-b", "the coffee machine is fixed"),
        note(2, "", "", "look at the log of the migration"),
    ])));

    let mut ui = sim(&app);
    ui.find("Inbox (2)").expect("the badge counts the notes (btnInbox)");
    drop(ui);

    let _ = app.update(Message::InboxToggle);
    let mut ui = sim(&app);
    ui.find("Notes from other sessions").expect("the title of the panel");
    ui.find("Close").expect("the close button");
    ui.find("#1 · session-a → session-b").expect("the line of what it says about itself");
    ui.find("#2 · a session").expect("a note without a sender says so (the Studio's fallback)");
    ui.find("the coffee machine is fixed").expect("the text of the note");
    ui.find("look at the log of the migration").expect("the second note");
    ui.find("Insert into the message").expect("the button that puts it into the composer");
    ui.find("Done").expect("the button that marks it read");
    save_png(&app, "inbox-panel");
}

/// One drawing of the window with the panel open under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR")
            .unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    {
        let _ = std::fs::remove_file(old.path());
    }
    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 800.0),
        app.view(),
    );
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// Inserting puts the note into the message the way the Studio does: a blank line between what
/// was written and what comes from another session, and the panel goes away.
#[test]
fn inserting_puts_the_note_into_the_message() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::InboxLoaded(Ok(vec![note(7, "session-a", "", "check the backup")])));
    let _ = app.update(Message::InboxToggle);
    let _ = app.update(Message::InboxInsert(7));
    assert_eq!(app.composer_text(), "check the backup", "the note is in the composer");
    assert!(!app.inbox_open, "the panel is shut after inserting");
    // an unknown id does nothing but shut the panel
    let _ = app.update(Message::InboxToggle);
    let _ = app.update(Message::InboxInsert(999));
    assert!(!app.inbox_open);
}

/// Close shuts the panel, and without notes there is no badge and nothing to open.
#[test]
fn close_shuts_the_panel_and_without_notes_there_is_no_badge() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::InboxLoaded(Ok(Vec::new())));
    assert!(app.inbox.is_empty(), "no notes are kept");
    assert!(!app.inbox_open, "and nothing stays open");
    let mut ui = sim(&app);
    assert!(
        ui.find("Inbox").is_err(),
        "without notes the button of the Studio is hidden"
    );
}

/// The answer of the route is the list of messages with the four fields the panel shows.
#[test]
fn the_answer_of_the_route_is_read_with_its_fields() {
    let notes = parse_notes(&serde_json::json!({
        "messages": [
            {"id": 1, "from": "session-a", "to": "session-b", "text": "hello"},
            {"id": 2},
        ]
    }));
    assert_eq!(
        notes,
        vec![
            Note { id: 1, from: "session-a".into(), to: "session-b".into(), text: "hello".into() },
            Note { id: 2, from: String::new(), to: String::new(), text: String::new() },
        ],
        "missing fields are empty, not an error"
    );
}
