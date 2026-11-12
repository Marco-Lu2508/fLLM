//! The model picker in the top bar of the chat, and the screen while the server switches its model.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, Server, View};
use fllm_app::models::{parse_overview, NO_MODEL};
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::json;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen2.5-0.5b-instruct", 8192))));
    let o = parse_overview(&json!({
        "loaded": {"name": "qwen2.5-0.5b-instruct"},
        "local": [{"name": "my-model.gguf", "path": "/models/my-model.gguf", "size_bytes": 2_000_000_000u64}],
        "cached": [{"name": "qwen2.5-0.5b-instruct", "path": "/c/q.gguf", "size_bytes": 428_000_000u64}, {"name": "gemma3:4b", "path": "/c/g.gguf", "size_bytes": 2_500_000_000u64}],
        "catalog": []
    }));
    let _ = app.update(Message::OverviewLoaded(Ok(o)));
    (app, dir)
}

fn click(app: &App, label: &str) -> Vec<Message> {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 800.0), app.view());
    ui.click(label).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    ui.into_messages().collect()
}

#[test]
fn the_model_name_opens_a_list_and_a_choice_loads_that_model() {
    let (mut app, _dir) = app("light");
    assert!(!app.picker_open());
    let opened = click(&app, "qwen2.5-0.5b-instruct");
    assert!(opened.iter().any(|m| matches!(m, Message::TogglePicker)), "{opened:?}");
    let _ = app.update(Message::TogglePicker);
    assert!(app.picker_open());
    let load = click(&app, "my-model.gguf");
    assert!(load.iter().any(|m| matches!(m, Message::LoadModel(r) if r == "/models/my-model.gguf")), "a model from a folder is loaded by its path: {load:?}");
    let gemma = click(&app, "gemma3:4b");
    assert!(gemma.iter().any(|m| matches!(m, Message::LoadModel(r) if r == "gemma3:4b")), "a downloaded one by its name: {gemma:?}");
    let manage = click(&app, "Manage models…");
    assert!(manage.iter().any(|m| matches!(m, Message::Go(View::Models))));
    let _ = app.update(Message::Go(View::Models));
    assert!(!app.picker_open(), "leaving the page closes the list");
}

#[test]
fn the_list_draws_in_both_themes_with_the_loaded_model_marked() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app(theme);
        let _ = app.update(Message::TogglePicker);
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 800.0), app.view());
        let shot = ui.snapshot(&app.theme()).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("picker-{theme}");
        for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
            let _ = std::fs::remove_file(old.path());
        }
        shot.matches_image(dir.join(format!("{name}.png"))).unwrap();
    }
}

#[test]
fn clicking_outside_closes_the_list_and_the_loaded_model_does_not_reload() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::TogglePicker);
    // the loaded model appears twice (the top bar and the list); the list entry only closes
    let _ = app.update(Message::ClosePicker);
    assert!(!app.picker_open());
    let _ = app.update(Message::TogglePicker);
    let _ = app.update(Message::TogglePicker);
    assert!(!app.picker_open(), "the name toggles the list");
}

#[test]
fn while_the_server_switches_the_window_says_so_instead_of_no_server() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::LoadModel("/models/my-model.gguf".into()));
    assert!(app.switching() && !app.picker_open());
    // the server is away for a moment
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    assert!(matches!(app.server_state(), Server::Down(_)));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 800.0), app.view());
    ui.find("Loading my-model.gguf…").expect("the window says what it waits for");
    assert!(ui.find("No fLLM server found").is_err(), "and does not claim that there is no server");
    drop(ui);
    // back with the new model
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("my-model", 4096))));
    assert!(!app.switching());
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 800.0), app.view());
    assert!(ui.find("Loading my-model.gguf…").is_err());
}

#[test]
fn a_failed_switch_and_an_unload_end_the_waiting() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::LoadModel("/models/huge.gguf".into()));
    let _ = app.update(Message::ModelAction(Err("not enough memory".into())));
    assert!(!app.switching(), "a refused load does not leave the window waiting");
    let _ = app.update(Message::UnloadModel);
    assert!(app.switching());
    let _ = app.update(Message::Connected(Err(NO_MODEL.into())));
    assert!(!app.switching() && matches!(app.server_state(), Server::NoModel));
}
