//! Switching the model: the server answers "restarting" and keeps answering for a moment with the old model. The window
//! must not take that answer for the new model (it then showed the old name and no overlay while the engine went away);
//! it waits until the engine has gone and is back, as the Studio does (`waitForRestart` of `models.js`).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, Server};
use fllm_app::models::Status;
use fllm_app::store::Store;
use std::time::Duration;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let mut old = ModelInfo::new("old", 4096);
    old.detail.name = "Old Model".into();
    let _ = app.update(Message::Connected(Ok(old)));
    (app, dir)
}

fn status(name: &str, setup: bool) -> Status {
    Status { model_name: name.into(), setup, ..Default::default() }
}

#[test]
fn the_old_engine_that_still_answers_is_not_the_new_model() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::LoadModel("/m/new.gguf".into()));
    assert!(app.waits_for_the_engine() && app.switching());
    let accepted = app.update(Message::ModelAction(Ok("/m/new.gguf".into())));
    assert_eq!(accepted.units(), 1, "one look at the engine, not the three reads of an ended switch");
    assert!(app.switching(), "the overlay stays");
    // the old engine answers with the old model: that is not the end
    let again = app.update(Message::SwitchProbe(Ok(status("Old Model", false))));
    assert_eq!(again.units(), 1, "look again");
    assert!(app.switching() && app.waits_for_the_engine());
    assert!(matches!(app.server_state(), Server::Ready(m) if m.id == "old"), "and the window still names what it knew");
    // the engine goes away, then something answers: that is the new engine
    let _ = app.update(Message::SwitchProbe(Err("connection refused".into())));
    assert!(app.waits_for_the_engine());
    let done = app.update(Message::SwitchProbe(Ok(status("Old Model", false))));
    assert!(done.units() >= 3, "the overview, the downloads and the model are read now");
    assert!(!app.waits_for_the_engine());
    // the model's own answer ends the overlay
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("new", 4096))));
    assert!(!app.switching());
    assert!(matches!(app.server_state(), Server::Ready(m) if m.id == "new"));
}

#[test]
fn a_status_with_another_model_ends_the_wait_at_once() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::LoadModel("/m/new.gguf".into()));
    let _ = app.update(Message::ModelAction(Ok("/m/new.gguf".into())));
    let done = app.update(Message::SwitchProbe(Ok(status("New Model", false))));
    assert!(done.units() >= 3);
    assert!(!app.waits_for_the_engine());
}

#[test]
fn an_engine_that_restarted_between_two_looks_is_believed_after_a_while() {
    let (mut app, _dir) = app();
    // the same model again: the name does not change, and the polls may have missed the moment it was away
    let _ = app.update(Message::LoadModel("/m/old.gguf".into()));
    let _ = app.update(Message::ModelAction(Ok("/m/old.gguf".into())));
    let wait = app.update(Message::SwitchProbe(Ok(status("Old Model", false))));
    assert_eq!(wait.units(), 1);
    app.age_the_switch(Duration::from_secs(16));
    let done = app.update(Message::SwitchProbe(Ok(status("Old Model", false))));
    assert!(done.units() >= 3 && !app.waits_for_the_engine());
}

#[test]
fn an_unload_waits_for_the_server_without_a_model() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::UnloadModel);
    let _ = app.update(Message::ModelAction(Ok(String::new())));
    let still = app.update(Message::SwitchProbe(Ok(status("Old Model", false))));
    assert_eq!(still.units(), 1, "the old engine still has its model");
    let done = app.update(Message::SwitchProbe(Ok(status("", true))));
    assert!(done.units() >= 3 && !app.waits_for_the_engine());
}

#[test]
fn a_server_that_does_not_come_back_ends_in_a_message_not_in_an_endless_overlay() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::LoadModel("/m/new.gguf".into()));
    let _ = app.update(Message::ModelAction(Ok("/m/new.gguf".into())));
    let _ = app.update(Message::SwitchProbe(Err("connection refused".into())));
    app.age_the_switch(Duration::from_secs(241));
    let end = app.update(Message::SwitchProbe(Err("connection refused".into())));
    assert_eq!(end.units(), 0);
    assert!(!app.switching() && !app.waits_for_the_engine());
    assert!(matches!(app.server_state(), Server::Down(_)));
}

#[test]
fn a_refused_load_ends_the_wait() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::LoadModel("/m/huge.gguf".into()));
    let _ = app.update(Message::ModelAction(Err("not enough memory".into())));
    assert!(!app.switching() && !app.waits_for_the_engine());
}

#[test]
fn the_start_screen_loads_with_the_same_wait() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HeroLoad("/m/new.gguf".into()));
    assert!(app.switching() && app.waits_for_the_engine());
}

#[test]
fn a_stray_probe_changes_nothing() {
    let (mut app, _dir) = app();
    let none = app.update(Message::SwitchProbe(Ok(status("Whatever", false))));
    assert_eq!(none.units(), 0);
    assert!(!app.switching());
}
