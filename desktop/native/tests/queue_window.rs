//! Messages typed while the agent works wait in a queue and go out one by one when the run ends
//! (`code.queue` of the page); a stop gives them back in the box.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, Item, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::Value;
use std::sync::Arc;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None }))));
    let (tools, defs) = client_tools();
    let _ = app.update(Message::Code(CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs))))));
    (app, dir)
}

fn say(app: &mut App, text: &str) {
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string()))))));
    let _ = app.update(Message::Code(CodeMsg::Send));
}

fn users(app: &App) -> Vec<String> {
    app.code().items.iter().filter_map(|i| if let Item::User { text, .. } = i { Some(text.clone()) } else { None }).collect()
}

fn finish(app: &mut App, answer: &str) {
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Delta(answer.into()))));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
}

#[test]
fn a_message_typed_while_the_agent_works_waits_and_goes_out_when_it_is_done() {
    let (mut app, _dir) = app();
    say(&mut app, "first");
    assert!(app.code().running);
    say(&mut app, "second");
    say(&mut app, "  third\n  line  ");
    assert_eq!(app.code().queue, ["second", "third\n  line"], "queued as typed, trimmed");
    assert!(app.code().editor.text().trim().is_empty(), "the box is empty again");
    assert_eq!(users(&app), ["first"]);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1200.0, 1500.0), app.view());
    assert!(ui.find("second").is_ok() && ui.find("third line").is_ok(), "a chip for each, its words on one line");
    assert!(ui.find("The agent is working. Enter queues your message, Esc stops it.").is_ok());
    drop(ui);
    // a chip is taken away with its ×
    let _ = app.update(Message::Code(CodeMsg::QueueDrop(0)));
    assert_eq!(app.code().queue, ["third\n  line"]);
    let _ = app.update(Message::Code(CodeMsg::QueueDrop(7)));
    assert_eq!(app.code().queue.len(), 1, "a place that is not there changes nothing");
    // the run ends: the next one starts by itself
    finish(&mut app, "answer one");
    assert_eq!(users(&app), ["first", "third\n  line"]);
    assert!(app.code().running && app.code().queue.is_empty());
    assert!(Simulator::with_size(settings(), iced::Size::new(1200.0, 1500.0), app.view()).find("Enter sends, Shift+Enter adds a new line").is_err());
    finish(&mut app, "answer two");
    assert!(!app.code().running);
    assert!(Simulator::with_size(settings(), iced::Size::new(1200.0, 1500.0), app.view()).find("Enter sends, Shift+Enter adds a new line").is_ok());
}

#[test]
fn a_stop_gives_the_queued_messages_back_in_the_box() {
    let (mut app, _dir) = app();
    say(&mut app, "first");
    say(&mut app, "second");
    say(&mut app, "third");
    // an empty line stops, and `/stop` too
    say(&mut app, "   ");
    assert!(!app.code().running);
    assert!(app.code().queue.is_empty());
    assert_eq!(app.code().editor.text().trim_end(), "second\n\nthird");
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Stopped. The messages you had queued are back in the input.")));
    assert_eq!(users(&app), ["first"], "nothing of the queue was sent");
    // `/stop` does the same and is not queued itself
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::SelectAll)));
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::Edit(Edit::Delete))));
    say(&mut app, "again");
    say(&mut app, "queued");
    say(&mut app, "/STOP");
    assert!(!app.code().running && app.code().queue.is_empty());
    assert_eq!(app.code().editor.text().trim_end(), "queued");
}

#[test]
fn a_new_session_forgets_the_queue() {
    let (mut app, _dir) = app();
    say(&mut app, "first");
    say(&mut app, "second");
    assert_eq!(app.code().queue.len(), 1);
    let _ = app.update(Message::Code(CodeMsg::Stop));
    let _ = app.update(Message::Code(CodeMsg::NewSession));
    assert!(app.code().queue.is_empty());
}
