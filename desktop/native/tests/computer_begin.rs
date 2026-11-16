//! `computerBegin`: with computer use on, the first request of a run waits until the server has
//! answered `computer/begin`, as the page awaits it.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use serde_json::Value;
use std::sync::Arc;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None }))));
    let (tools, defs) = client_tools();
    let _ = app.update(Message::Code(CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs))))));
    (app, dir)
}

fn computer_on(app: &mut App) {
    let epoch = app.code().computer.epoch;
    let _ = app.update(Message::Code(CodeMsg::ComputerStatus(epoch, Some(fllm_app::code::CuStatus { enabled: true, available: true, steps_used: 0, max_steps: 100 }))));
    assert!(app.code().computer_on());
}

fn say(app: &mut App, text: &str) -> iced::Task<Message> {
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string()))))));
    app.update(Message::Code(CodeMsg::Send))
}

#[test]
fn without_computer_use_the_run_starts_at_once() {
    let (mut app, _dir) = app();
    let _ = say(&mut app, "hello");
    assert!(app.code().running, "nothing to wait for");
    assert!(!app.code().beginning);
}

#[test]
fn with_computer_use_the_run_waits_for_the_server() {
    let (mut app, _dir) = app();
    computer_on(&mut app);
    let _ = say(&mut app, "hello");
    assert!(!app.code().running && app.code().beginning, "the first request waits for `computer/begin`");
    // a second message in the meantime waits in the queue and is not lost
    let _ = say(&mut app, "second");
    assert_eq!(app.code().queue, ["second"]);
    let _ = app.update(Message::Code(CodeMsg::ComputerBegun { prompt: "hello".into(), display: "hello".into() }));
    assert!(app.code().running && !app.code().beginning, "the answer starts the run");
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
}

#[test]
fn a_stop_while_the_server_is_asked_takes_the_run_back() {
    let (mut app, _dir) = app();
    computer_on(&mut app);
    let _ = say(&mut app, "hello");
    assert!(app.code().beginning);
    let _ = say(&mut app, "");
    assert!(!app.code().beginning, "an empty line stops it");
    let _ = app.update(Message::Code(CodeMsg::ComputerBegun { prompt: "hello".into(), display: "hello".into() }));
    assert!(!app.code().running, "the late answer starts nothing");
}
