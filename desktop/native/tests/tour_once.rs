//! The welcome tour comes once, at the very first start of the program, and never again: not when the model is switched (the server
//! answers again then), not when the answer settings are reset, not for a program that found data of its own.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::models::parse_overview;
use fllm_app::tour::Hardware;
use fllm_app::store::Store;
use serde_json::json;

fn loaded() -> Message {
    Message::TourLoaded {
        overview: Ok(parse_overview(&json!({"available_memory_bytes": 6_000_000_000u64, "loaded": {"name": "m"}, "recommended": [], "catalog": [], "local": [], "cached": []}))),
        hardware: Ok(Hardware::from_json(&json!({"devices": []}))),
    }
}

fn model() -> Message {
    Message::Connected(Ok(ModelInfo::new("m", 4096)))
}

#[test]
fn the_first_start_shows_it_once_and_a_switch_of_the_model_does_not_bring_it_back() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), dir.path().join("chats.json"));
    let first = app.update(model());
    assert_eq!(first.units(), 1, "the first answer asks for the hardware");
    let _ = app.update(loaded());
    assert_eq!(app.tour, Some(0), "the tour opens");
    let _ = app.update(Message::TourSkip);
    assert_eq!(app.tour, None);

    // the user switches the model: the server answers again, twice, and the data of the tour is there again
    let _ = app.update(Message::LoadModel("/m/x.gguf".into()));
    let again = app.update(model());
    assert_eq!(again.units(), 0, "no look at the hardware");
    let _ = app.update(loaded());
    assert_eq!(app.tour, None, "the tour does not come back");
}

#[test]
fn a_tour_that_was_never_seen_does_not_come_with_the_second_answer_of_the_server() {
    // the key is not set and there are no chats (nothing was written yet), but the server has already answered once
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), dir.path().join("chats.json"));
    let _ = app.update(model());
    let _ = app.update(model());
    let _ = app.update(loaded());
    assert_eq!(app.tour, Some(0), "once, whoever answers first");
    let _ = app.update(Message::TourSkip);
    let _ = app.update(loaded());
    assert_eq!(app.tour, None);
}

#[test]
fn a_program_that_finds_its_own_data_has_no_first_start() {
    // data of an older version: there is a file, no chats in it, and the key of the tour is not in it
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let answer = app.update(model());
    assert_eq!(answer.units(), 0);
    let _ = app.update(loaded());
    assert_eq!(app.tour, None);
    assert!(!app.prefs().tour_seen, "and nothing is written");
}

#[test]
fn back_to_automatic_keeps_what_is_no_answer_setting() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let mut store = Store::default();
    store.prefs.tour_seen = true;
    store.prefs.auto_approve = true;
    store.prefs.learn = true;
    store.prefs.manual_sampling = true;
    store.prefs.temperature = 1.4;
    store.prefs.recent_folders = vec!["/work".into()];
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::ResetPrefs);
    let p = app.prefs();
    assert!(!p.manual_sampling && (p.temperature - Store::default().prefs.temperature).abs() < 1e-6, "the answer settings are back");
    assert!(p.tour_seen, "the tour stays seen (it came again after the reset)");
    assert!(p.auto_approve && p.learn, "the settings of the code page stay");
    assert_eq!(p.recent_folders, vec!["/work".to_string()]);
}
