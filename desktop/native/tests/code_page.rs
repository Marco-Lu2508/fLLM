//! Code mode in the window: the folder screen, a session with its tool cards, the buttons that ask and the changes bar.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{Change, CodeMsg, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::json;
use std::sync::Arc;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo { id: "qwen".into(), max_context: 8192 })));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn code(app: &mut App, m: CodeMsg) {
    let _ = app.update(Message::Code(m));
}

fn open_folder(app: &mut App) {
    code(app, CodeMsg::Opened(Ok(Workspace { root: "/home/marco/projects/fllm".into(), name: "fllm".into(), branch: Some("main".into()) })));
    let tools: Vec<Tool> = [("read_file", Permission::Read), ("grep", Permission::Read), ("edit_file", Permission::Edit), ("bash", Permission::Command)].into_iter().map(|(n, p)| Tool { name: n.into(), permission: p }).collect();
    code(app, CodeMsg::ToolsLoaded(Ok((tools, json!([])))));
}

fn tool_call(app: &mut App, index: usize, id: &str, name: &str, args: &str) {
    code(app, CodeMsg::Stream(StreamEvent::ToolCall { index, id: Some(id.into()), name: Some(name.into()), arguments: args.into() }));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

fn session(app: &mut App) {
    open_folder(app);
    say(app, "Fix the off-by-one error in the tokenizer and run the tests.");
    code(app, CodeMsg::Stream(StreamEvent::Delta("I'll start by reading the tokenizer, then fix the loop bound.".into())));
    tool_call(app, 0, "c1", "read_file", r#"{"path":"src/tokenizer.rs"}"#);
    tool_call(app, 1, "c2", "edit_file", r#"{"path":"src/tokenizer.rs","old_string":"for i in 0..=n {","new_string":"for i in 0..n {"}"#);
    tool_call(app, 2, "c3", "bash", r#"{"command":"cargo test -p tokenizer"}"#);
    code(app, CodeMsg::Stream(StreamEvent::Done));
    code(app, CodeMsg::ToolDone("c1".into(), Ok("     1\tfn tokenize(s: &str) {\n     2\t    for i in 0..=n {\n     3\t    }\n(end of file, 3 lines)".into())));
    code(app, CodeMsg::ChangesLoaded(Ok(vec![Change { id: 1, path: "src/lib.rs".into(), kind: "edit".into() }, Change { id: 2, path: "src/tokenizer.rs".into(), kind: "edit".into() }])));
}

fn click(app: &App, label: &str) -> Vec<Message> {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    ui.click(label).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    ui.into_messages().collect()
}

fn picture(app: &App, name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    let shot = ui.snapshot(&app.theme()).unwrap();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
    std::fs::create_dir_all(&dir).unwrap();
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).unwrap();
}

#[test]
fn the_code_page_asks_for_a_folder_first_and_the_button_opens_it() {
    let (mut app, _dir) = app("light");
    assert!(app.code().workspace.is_none());
    code(&mut app, CodeMsg::PathInput("/home/marco/projects/fllm".into()));
    let msgs = click(&app, "Open");
    assert!(msgs.iter().any(|m| matches!(m, Message::Code(CodeMsg::Open))), "{msgs:?}");
    code(&mut app, CodeMsg::Opened(Err("/nope is not a folder".into())));
    assert_eq!(app.code().error.as_deref(), Some("/nope is not a folder"));
    picture(&app, "code-folder-light");
    // the sidebar has the page
    let (app2, _d) = self::app("dark");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 800.0), app2.view());
    ui.click("Code").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::Go(View::Code))));
}

#[test]
fn a_session_draws_in_both_themes_with_the_waiting_edit_the_command_and_the_changes() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app(theme);
        session(&mut app);
        assert!(app.code().running);
        picture(&app, &format!("code-session-{theme}"));
    }
}

#[test]
fn the_buttons_of_a_waiting_call_allow_or_deny_that_call() {
    let (mut app, _dir) = app("light");
    session(&mut app);
    // two calls wait (the edit and the command): the first button belongs to the first card
    let allow = click(&app, "Allow");
    assert!(allow.iter().any(|m| matches!(m, Message::Code(CodeMsg::Approve(id)) if id == "c2")), "{allow:?}");
    let deny = click(&app, "Deny");
    assert!(deny.iter().any(|m| matches!(m, Message::Code(CodeMsg::Deny(id)) if id == "c2")), "{deny:?}");
    // undo is for when the agent is not working
    code(&mut app, CodeMsg::Stop);
    let undo = click(&app, "Undo last");
    assert!(undo.iter().any(|m| matches!(m, Message::Code(CodeMsg::Undo(None)))));
    let all = click(&app, "Undo all");
    assert!(all.iter().any(|m| matches!(m, Message::Code(CodeMsg::Undo(Some(1))))), "undo all goes back to before the first change: {all:?}");
}

#[test]
fn undo_is_not_offered_while_the_agent_works_and_stop_is() {
    let (mut app, _dir) = app("light");
    session(&mut app);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    ui.find("Stop").expect("a running agent can be stopped");
    drop(ui);
    code(&mut app, CodeMsg::Stop);
    assert!(!app.code().running);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    assert!(ui.find("Stop").is_err());
    ui.find("2 changes in 2 files").expect("the bar counts changes and files");
}

#[test]
fn without_a_model_the_agent_cannot_be_started() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Connected(Err(fllm_app::models::NO_MODEL.into())));
    open_folder(&mut app);
    say(&mut app, "hello");
    assert!(app.code().items.is_empty() && !app.code().running);
}

#[test]
fn the_folder_screen_has_a_dialog_and_the_recent_folders() {
    let (mut app, _dir) = app("light");
    // a folder that was opened is remembered, once, newest first, at most six, and kept on disk
    for f in ["/a/one", "/a/two", "/a/three", "/a/four", "/a/five", "/a/six", "/a/seven", "/a/three"] {
        code(&mut app, CodeMsg::Opened(Ok(Workspace { root: f.into(), name: f.into(), branch: None })));
    }
    assert_eq!(app.prefs().recent_folders, ["/a/three", "/a/seven", "/a/six", "/a/five", "/a/four", "/a/two"]);
    code(&mut app, CodeMsg::Close);
    let msgs = click(&app, "Choose…");
    assert!(msgs.iter().any(|m| matches!(m, Message::Code(CodeMsg::Pick))), "{msgs:?}");
    let msgs = click(&app, "seven");
    assert!(msgs.iter().any(|m| matches!(m, Message::Code(CodeMsg::OpenPath(p)) if p == "/a/seven")), "{msgs:?}");
    picture(&app, "code-folder-recent-light");
}

#[test]
fn the_dialog_fills_the_path_and_a_closed_dialog_changes_nothing() {
    let (mut app, _dir) = app("light");
    code(&mut app, CodeMsg::PathInput("/typed".into()));
    code(&mut app, CodeMsg::Picked(Ok(None)));
    assert_eq!(app.code().path_input, "/typed");
    code(&mut app, CodeMsg::Picked(Err("This computer has no program that can open a file dialog".into())));
    assert!(app.code().error.as_deref().unwrap().contains("no program"));
    code(&mut app, CodeMsg::Picked(Ok(Some("/chosen".into()))));
    assert_eq!(app.code().path_input, "/chosen", "a chosen folder is opened at once");
}

#[test]
fn a_session_is_kept_per_folder_and_comes_back_when_the_folder_is_opened_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let boot = || {
        let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path.clone());
        let _ = app.update(Message::Connected(Ok(ModelInfo { id: "qwen".into(), max_context: 8192 })));
        let _ = app.update(Message::Go(View::Code));
        app
    };
    let mut app = boot();
    session(&mut app);
    // the agent is stopped half way: the open calls are kept as interrupted, the text as it was
    code(&mut app, CodeMsg::Stop);
    let before = app.code().saved();
    assert!(before.len() >= 5, "{before:?}");
    assert!(dir.path().join("code-sessions.json").exists());
    drop(app);

    // a new start of the app, the same folder
    let mut app = boot();
    assert!(app.code().items.is_empty());
    open_folder(&mut app);
    assert_eq!(app.code().saved(), before, "the conversation is back");
    // another folder has its own
    code(&mut app, CodeMsg::Close);
    code(&mut app, CodeMsg::Opened(Ok(Workspace { root: "/other".into(), name: "other".into(), branch: None })));
    assert!(app.code().items.is_empty());
    // and a new session forgets the kept one
    code(&mut app, CodeMsg::Close);
    open_folder(&mut app);
    assert!(!app.code().items.is_empty());
    code(&mut app, CodeMsg::NewSession);
    drop(app);
    let mut app = boot();
    open_folder(&mut app);
    assert!(app.code().items.is_empty(), "a session that was cleared stays cleared");
}
