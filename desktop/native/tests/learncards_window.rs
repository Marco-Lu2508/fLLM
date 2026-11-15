//! Row 88, Settings > Personalization: the three cards of `learn.js` — what they draw of what
//! the server answers, what each button asks the server for, and the dialogs that stand before
//! a deletion.

use fllm_app::app::{App, Message, View};
use fllm_app::learncards::*;
use fllm_app::learn::Status;
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::json;

fn settings() -> iced::Settings {
    iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Go(View::Settings));
    (app, dir)
}

fn lp(app: &mut App, m: LpMsg) -> iced::Task<Message> {
    app.update(Message::Lp(m))
}

fn profile() -> Profile {
    Profile {
        text: "## Preferences\n- Short answers".into(),
        in_prompt: Some(true),
        max_chars: Some(3000),
        proposals: vec![ProfileProposal { id: "p1".into(), text: "Writes Rust".into(), section: "languages_used".into(), session: Some("s7".into()) }],
    }
}

fn memory() -> Memory {
    Memory {
        global: vec![MemEntry { index: 0, text: "Prefers tabs".into(), source: None }],
        workspace: vec![MemEntry { index: 1, text: "Tests run with cargo test".into(), source: Some(MemSource { session: Some("s3".into()), created: None }) }],
        proposals: vec![
            MemProposal { id: "m1".into(), action: "add".into(), text: "Use rustfmt".into(), scope: "workspace".into(), kind: "command".into(), session: Some("s9".into()), reason: Some("seen twice".into()) },
            MemProposal { id: "m2".into(), action: "remove".into(), text: "Prefers tabs".into(), scope: "global".into(), kind: "repeat".into(), session: None, reason: None },
        ],
    }
}

#[test]
fn the_lines_of_the_cards_are_the_pages_lines() {
    let e = MemEntry { index: 2, text: "x".into(), source: Some(MemSource { session: Some("s3".into()), created: None }) };
    assert_eq!(mem_source_line("workspace", &e), "this project · learned in session s3");
    assert_eq!(mem_source_line("global", &MemEntry { source: None, ..e.clone() }), "every project · written by hand or by the memory tool");
    let no_session = MemEntry { source: Some(MemSource { session: None, created: None }), ..e };
    assert_eq!(mem_source_line("global", &no_session), "every project · learned in session ?");
    let m = memory();
    assert_eq!(mem_proposal_line(&m.proposals[0]), "this project · command · from session s9 · seen twice");
    assert_eq!(mem_proposal_line(&m.proposals[1]), "every project · repeat");
    assert_eq!(profile_proposal_line(&profile().proposals[0]), "languages used · from session s7");
    assert_eq!(profile_count("héllo", Some(3000)), "5 / 3000");
    assert_eq!(profile_count("😀", None), "2 / 3000", "the page counts UTF-16 units");
}

#[test]
fn the_cards_show_what_the_server_answered() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::ProfileLoaded(Ok(profile())));
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(memory())));
    let _ = lp(&mut app, LpMsg::StatusLoaded(Some(Status { enabled: Some(true), ask_first: true, project_override: Some(false), ..Default::default() })));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
    for text in [
        "Learning from my sessions",
        "What the coding agent learned about you",
        "Memory of the coding agent",
        "Nothing was learned automatically yet.",
        "Learning is off for the open project (its .fllm/learn.json).",
        "Writes Rust",
        "languages used · from session s7",
        "Use rustfmt",
        "Forget: Prefers tabs",
        "Forget it",
        "Remember",
        "Prefers tabs",
        "this project · learned in session s3",
        "Forget everything that was learned",
        "Lines that wait for your yes (only when you chose to be asked first)",
    ] {
        assert!(ui.find(text).is_ok(), "the cards draw {text:?}");
    }
    assert!(ui.find("No profile line waits for a yes.").is_err());
}

#[test]
fn empty_lists_say_so() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::ProfileLoaded(Ok(Profile::default())));
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(Memory::default())));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
    for text in ["No profile line waits for a yes.", "No memory entry waits for a yes.", "The agent memory is empty."] {
        assert!(ui.find(text).is_ok(), "{text:?}");
    }
    drop(ui);
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Err("no route".into())));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
    assert!(ui.find("no route").is_ok(), "a failed read is written into the card, as the page writes it");
}

#[test]
fn every_button_asks_the_server_for_what_the_page_asks() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::ProfileLoaded(Ok(profile())));
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(memory())));
    let mut pane = LearnPane::default();
    // Save
    pane.editor = iced::widget::text_editor::Content::with_text("## Languages\n- Rust");
    let post = pane.post_of(&LpMsg::Save, "/w").unwrap();
    assert_eq!((post.path, post.body, post.done), ("/api/studio/learn/profile", json!({ "action": "save", "text": "## Languages\n- Rust" }), Some("Profile saved.")));
    // the switch of the prompt
    let post = pane.post_of(&LpMsg::UseInPrompt(false), "/w").unwrap();
    assert_eq!((post.body, post.done), (json!({ "action": "prompt", "on": false }), None));
    // a proposal
    assert_eq!(pane.post_of(&LpMsg::ProfileProposal("p1".into(), "accept"), "/w").unwrap().body, json!({ "action": "accept", "id": "p1" }));
    assert_eq!(pane.post_of(&LpMsg::MemProposal("m1".into(), "reject"), "/w").unwrap().body, json!({ "action": "reject", "id": "m1", "workspace": "/w" }));
    // the switch that asks first
    let post = pane.post_of(&LpMsg::AskFirst(true), "/w").unwrap();
    assert_eq!((post.path, post.body, post.done), ("/api/studio/learn/settings", json!({ "ask_first": true }), Some("The agent now proposes and waits for your yes.")));
    assert_eq!(pane.post_of(&LpMsg::AskFirst(false), "/w").unwrap().done, Some("The agent applies what it learns and logs it."));
    // the dialogs
    assert!(pane.post_of(&LpMsg::DialogOk, "/w").is_none(), "no dialog, no request");
    let _ = pane.update(LpMsg::DeleteProfileAsk, "http://127.0.0.1:9", "/w", &mut None);
    assert_eq!(pane.dialog, Some(Dialog::DeleteProfile));
    let post = pane.post_of(&LpMsg::DialogOk, "/w").unwrap();
    assert_eq!((post.body, post.done), (json!({ "action": "delete" }), Some("Profile deleted.")));
    assert!(pane.dialog.is_none());
    let _ = pane.update(LpMsg::MemDeleteAsk("global".into(), 4, "old".into()), "http://127.0.0.1:9", "/w", &mut None);
    assert_eq!(pane.post_of(&LpMsg::DialogOk, "/w").unwrap().body, json!({ "action": "delete", "scope": "global", "index": 4, "workspace": "/w" }));
    let _ = pane.update(LpMsg::MemEditAsk("workspace".into(), 1, "Tests".into()), "http://127.0.0.1:9", "/w", &mut None);
    let _ = pane.update(LpMsg::MemEditInput("Tests run with make".into()), "http://127.0.0.1:9", "/w", &mut None);
    let post = pane.post_of(&LpMsg::DialogOk, "/w").unwrap();
    assert_eq!((post.body, post.done), (json!({ "action": "edit", "scope": "workspace", "index": 1, "text": "Tests run with make", "workspace": "/w" }), Some("Saved.")));
    // an empty text deletes the entry
    let _ = pane.update(LpMsg::MemEditAsk("workspace".into(), 1, "Tests".into()), "http://127.0.0.1:9", "/w", &mut None);
    let _ = pane.update(LpMsg::MemEditInput("   ".into()), "http://127.0.0.1:9", "/w", &mut None);
    assert_eq!(pane.post_of(&LpMsg::DialogOk, "/w").unwrap().body["action"], "delete");
    // forget everything
    let _ = pane.update(LpMsg::ForgetAsk, "http://127.0.0.1:9", "/w", &mut None);
    let post = pane.post_of(&LpMsg::DialogOk, "/w").unwrap();
    assert_eq!((post.body, post.done), (json!({ "action": "forget", "all_learned": true, "text": "", "workspace": "/w" }), Some("Forgotten.")));
    // cancel leaves everything
    let _ = pane.update(LpMsg::ForgetAsk, "http://127.0.0.1:9", "/w", &mut None);
    let _ = pane.update(LpMsg::DialogCancel, "http://127.0.0.1:9", "/w", &mut None);
    assert!(pane.dialog.is_none() && pane.post_of(&LpMsg::DialogOk, "/w").is_none());
}

#[test]
fn an_answer_is_a_toast_and_every_list_is_read_again() {
    let (mut app, _dir) = app();
    let done = lp(&mut app, LpMsg::Posted(Ok(json!({})), Some("Saved.".into())));
    assert_eq!(app.toast.as_deref(), Some("Saved."));
    assert!(done.units() >= 4, "the lists of the page are read again: {} units", done.units());
    app.toast = None;
    let _ = lp(&mut app, LpMsg::Posted(Err("the memory is full".into()), Some("Saved.".into())));
    assert_eq!(app.toast.as_deref(), Some("the memory is full"), "a failure is the toast, not the success");
    // the switch of learning
    let _ = lp(&mut app, LpMsg::StatusLoaded(Some(Status { enabled: Some(true), ..Default::default() })));
    let _ = lp(&mut app, LpMsg::Enabled(false));
    let _ = lp(&mut app, LpMsg::Posted(Ok(json!({})), Some("Learning is off. What was learned stays; nothing new is saved.".into())));
    assert_eq!(app.toast.as_deref(), Some("Learning is off. What was learned stays; nothing new is saved."));
}

#[test]
fn a_dialog_stands_over_the_page() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(memory())));
    let _ = lp(&mut app, LpMsg::ForgetAsk);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
    assert!(ui.find("Forget everything that was learned?").is_ok());
    assert!(ui.find("Removes every memory entry the learning loop wrote and the memory entries that wait for a yes. Entries you wrote yourself stay.").is_ok());
    drop(ui);
    let _ = lp(&mut app, LpMsg::DialogCancel);
    let _ = lp(&mut app, LpMsg::MemEditAsk("global".into(), 0, "Prefers tabs".into()));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
    assert!(ui.find("Edit this memory entry").is_ok());
    assert!(ui.find("An empty text deletes the entry.").is_ok());
}

fn picture(app: &App, name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 2600.0), app.view());
    let shot = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    for old in std::fs::read_dir(&out).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    assert!(shot.matches_image(out.join(format!("{name}.png"))).unwrap(), "the picture of {name}");
}

/// The three cards drawn, for the eye of the vision check (the PNG is an artifact, not a gate).
#[test]
fn the_learning_cards_are_drawn() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::ProfileLoaded(Ok(profile())));
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(memory())));
    let _ = lp(&mut app, LpMsg::StatusLoaded(Some(Status { enabled: Some(true), ask_first: true, ..Default::default() })));
    picture(&app, "settings-learning-cards");
}

#[test]
fn the_buttons_of_the_proposals_belong_to_their_proposal() {
    let (mut app, _dir) = app();
    let _ = lp(&mut app, LpMsg::MemoryLoaded(Ok(memory())));
    let click = |app: &App, label: &str| {
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view());
        ui.click(label).unwrap_or_else(|_| panic!("{label}"));
        ui.into_messages().collect::<Vec<_>>()
    };
    let sent = click(&app, "Forget it");
    assert!(matches!(sent.as_slice(), [Message::Lp(LpMsg::MemProposal(id, "accept"))] if id == "m2"), "a removal is forgotten: {sent:?}");
    let sent = click(&app, "Remember");
    assert!(matches!(sent.as_slice(), [Message::Lp(LpMsg::MemProposal(id, "accept"))] if id == "m1"), "an addition is remembered: {sent:?}");
}

#[test]
fn the_switch_that_asks_first_is_what_the_card_shows_at_once() {
    let mut pane = LearnPane { status: Some(Status { enabled: Some(true), ..Default::default() }), ..Default::default() };
    let _ = pane.post_of(&LpMsg::AskFirst(true), "/w");
    assert!(pane.status.as_ref().unwrap().ask_first);
    let _ = pane.post_of(&LpMsg::AskFirst(false), "/w");
    assert!(!pane.status.as_ref().unwrap().ask_first);
}
