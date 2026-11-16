//! The thinking chip keeps its level per chat (`t.thinkingLevel` of the Studio): a new chat starts
//! with the last choice, and every chat comes back with the level it was left at.

use fllm_app::api::{ModelInfo, Thinking};
use fllm_app::app::{App, Message};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use std::sync::Arc;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let mut model = ModelInfo::new("qwen", 8192);
    model.thinking = Thinking::from_caps(Some(&serde_json::json!({"supported": true, "control": "toggle", "variable": "enable_thinking", "levels": ["off", "low", "medium", "high"], "default": "off"})));
    let _ = app.update(Message::Connected(Ok(model)));
    (app, dir)
}

fn say(app: &mut App, text: &str) {
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    let _ = app.update(Message::Send);
    // the answer comes and ends, so that the next message can be sent
    let id = app.store().chats.iter().map(|c| c.id).max().unwrap_or(0);
    let _ = app.update(Message::Stream(id, fllm_app::api::StreamEvent::Delta("ok".into())));
    let _ = app.update(Message::Stream(id, fllm_app::api::StreamEvent::Done));
}

#[test]
fn a_chat_remembers_the_level_it_was_left_at() {
    let (mut app, dir) = app();
    let _ = app.update(Message::CycleThinking);
    assert_eq!(app.thinking_level(), "low");
    say(&mut app, "first chat");
    let a = app.store().chats[0].id;
    assert_eq!(app.store().chats[0].thinking_level, "low", "a new chat starts with the last choice");
    let _ = app.update(Message::CycleThinking);
    let _ = app.update(Message::CycleThinking);
    assert_eq!(app.thinking_level(), "high");
    assert_eq!(app.store().chats[0].thinking_level, "high", "the open chat keeps it");
    // a second chat starts with that, and changes it for itself
    let _ = app.update(Message::NewChat);
    say(&mut app, "second chat");
    let b = app.store().chats.iter().find(|c| c.id != a).unwrap().id;
    assert_eq!(app.store().chats.iter().find(|c| c.id == b).unwrap().thinking_level, "high");
    let _ = app.update(Message::CycleThinking);
    assert_eq!(app.thinking_level(), "off", "after high comes off");
    let _ = app.update(Message::OpenChat(a));
    assert_eq!(app.thinking_level(), "high", "the first chat has its own level again");
    assert_eq!(app.chat_params().extra["reasoning_effort"], "high", "and it is what the request carries");
    let _ = app.update(Message::OpenChat(b));
    assert_eq!(app.thinking_level(), "off");
    assert_eq!(app.chat_params().extra["chat_template_kwargs"], serde_json::json!({"enable_thinking": false}));
    let _ = dir;
}
