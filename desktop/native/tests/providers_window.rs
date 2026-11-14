//! Settings > Providers in the window: the two cards of `spane-providers` with the list the
//! server sent, the templates in their groups, the form that fills itself and the question before
//! a provider is removed (`providers.js` of the Studio).

use fllm_app::app::{App, Message, View};
use fllm_app::providers::{ProviderList, TestAnswer};
use fllm_app::store::Store;
use iced::Size;
use iced_test::Simulator;
use serde_json::json;

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
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 2200.0), app.view())
}

/// Writes the drawing under `ui-snapshots/<name>.png`: the two cards of the pane on their own,
/// with the question over them when there is one. The settings page is taller than a texture the
/// CI GPUs accept — wgpu refuses more than 4096 there — and the drawing is of the section either
/// way; the `find` assertions of the tests walk the whole page, whatever the window is.
fn save_png(app: &App, name: &str, list: &ProviderList, remove: bool) {
    let t = app.tokens();
    let theme = app.theme();
    let form = fllm_app::providers::Form::default();
    let models = fllm_app::providers::ModelList::default();
    let file_info = format!("Saved in {}", list.config_file);
    let view = fllm_app::providers::ProvidersView {
        list,
        models: &models,
        form: &form,
        result: None,
        file_info: &file_info,
        network_open: false,
    };
    let mut content = fllm_app::providers::section(t, view);
    if remove {
        content = iced::widget::stack![
            content,
            fllm_app::app::backdrop(Message::ProvRemoveCancel),
            fllm_app::providers::remove_window(t)
        ]
        .into();
    }
    let mut ui = Simulator::with_size(settings(), Size::new(1100.0, 3000.0), content);
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// The answer of `GET /api/studio/providers`: one provider with a key and two templates (one of
/// them a server on this computer).
fn list(with_provider: bool) -> ProviderList {
    let providers = if with_provider {
        json!([
            {"id": "ollama", "name": "Ollama here", "kind": "ollama", "base_url": "http://127.0.0.1:11434/v1",
             "has_key": false, "enabled": true, "local": true, "models": ["llama3:8b", "qwen2.5:0.5b"]}
        ])
    } else {
        json!([])
    };
    serde_json::from_value(json!({
        "providers": providers,
        "presets": [
            {"id": "openai", "name": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1",
             "local": false, "group": "cloud", "key_url": "https://platform.openai.com/keys"},
            {"id": "anthropic", "name": "Anthropic", "kind": "anthropic", "base_url": "https://api.anthropic.com",
             "local": false, "group": "cloud", "key_url": null},
            {"id": "ollama", "name": "Ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434/v1",
             "local": true, "group": null, "key_url": null}
        ],
        "config_file": "/home/marco/.fllm/providers.json"
    }))
    .expect("the answer of the server")
}

/// The empty list says what to do, the form is there, and the file line names the file.
#[test]
fn the_two_cards_say_what_the_server_said() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::ProvLoaded(Ok(list(false))));

    let mut ui = sim(&app);
    ui.find("Model providers").expect("the first card");
    ui.find(fllm_app::providers::LIST_HELP).expect("its description");
    ui.find(fllm_app::providers::PRIVACY_HELP).expect("the privacy note");
    ui.find("No providers yet. Choose a service below, paste its key and test the connection.")
        .expect("the empty list");
    ui.find("Refresh all model lists").expect("the button under the list");
    ui.find("Saved in /home/marco/.fllm/providers.json").expect("the line under the form");
    ui.find("Add a provider").expect("the title of the form");
    ui.find("Choose a service").expect("the templates");
    ui.find("OpenAI").expect("a cloud template");
    ui.find("Other servers on your network").expect("the group behind its own line");
    ui.find("Name").expect("the first box");
    ui.find("Address").expect("the third box");
    ui.find("Test connection").expect("the button of the form");
    save_png(&app, "settings-providers", &list(false), false);
}

/// A provider in the list with its line and its actions, and the question before it is removed.
#[test]
fn a_provider_row_says_what_it_is_and_asks_before_it_is_removed() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::ProvLoaded(Ok(list(true))));

    let mut ui = sim(&app);
    ui.find("Ollama here").expect("the title of the row");
    ui.find("ollama, this computer or network, no key, 2 models").expect("the line of `provRender`");
    ui.find("http://127.0.0.1:11434/v1").expect("the address under it");
    ui.find("Test and refresh models").expect("its actions");
    ui.find("Edit").expect("the second one");
    ui.find("Remove").expect("the third one");
    // the network group is closed at first (the `<details>` of the Studio)
    assert!(ui.find(fllm_app::providers::NETWORK_HELP).is_err(), "the group is closed");
    drop(ui);

    let _ = app.update(Message::ProvRemove("ollama".into()));
    assert_eq!(app.prov_remove.as_deref(), Some("ollama"), "the window asks first");
    let mut ui = sim(&app);
    ui.find("Remove this provider and its saved key?").expect("the question of `confirmDialog`");
    save_png(&app, "settings-providers-remove", &list(true), true);
    drop(ui);

    // the group opens on its line
    let _ = app.update(Message::ProvNetOpen);
    let mut ui = sim(&app);
    ui.find(fllm_app::providers::NETWORK_HELP).expect("the group is open now");
    drop(ui);

    // the question is answered with Cancel: nothing is asked of the server again
    let _ = app.update(Message::ProvRemoveCancel);
    assert!(app.prov_remove.is_none(), "the dialog is closed");
}

/// The form fills itself from a template, edits a provider with its own values and clears itself.
#[test]
fn the_form_fills_itself_from_a_template_and_from_a_row() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::ProvLoaded(Ok(list(true))));

    // the click on a template fills the boxes (`provPresetBox` of `providers.js`)
    let _ = app.update(Message::ProvPreset("openai".into()));
    let mut ui = sim(&app);
    ui.find("OpenAI").expect("the name of the template is in the box");
    ui.find("https://api.openai.com/v1").expect("and its address");
    ui.find("Get a key at platform.openai.com").expect("the link under the key box");
    drop(ui);

    // *Edit* takes the row's values and the title says which provider it is
    let _ = app.update(Message::ProvEdit("ollama".into()));
    let mut ui = sim(&app);
    ui.find("Edit Ollama here").expect("the title of the form");
    ui.find("http://127.0.0.1:11434/v1").expect("the address of the row");
    drop(ui);

    // *Clear* empties it again (the row of the list still shows its address, so the boxes are
    // read from the form itself)
    let _ = app.update(Message::ProvClear);
    assert!(app.prov_form.name.is_empty() && app.prov_form.url.is_empty(), "the boxes are empty");
    assert!(app.prov_form.editing.is_none(), "and nothing is edited");
    let mut ui = sim(&app);
    ui.find("Add a provider").expect("the title of a new one");
}

/// The line under the list after a test, with the summary the Studio writes, and the model count
/// after a refresh (`provShow` and `btnProvRefreshModels`).
#[test]
fn the_line_under_the_list_says_what_a_test_said() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::ProvLoaded(Ok(list(true))));

    let _ = app.update(Message::ProvTest("ollama".into()));
    let mut ui = sim(&app);
    ui.find("Testing...").expect("the line while the test runs");
    drop(ui);

    let answer: TestAnswer = serde_json::from_value(json!({"ok": true, "count": 2, "models": ["llama3:8b", "qwen2.5:0.5b"]}))
        .expect("the answer of the server");
    let _ = app.update(Message::ProvTested(Ok(answer)));
    let mut ui = sim(&app);
    ui.find("Connected. 2 models found: llama3:8b, qwen2.5:0.5b.")
        .expect("the summary of `providerTestSummary`");
    drop(ui);

    // a refused test shows the message of the server in the destructive colour
    let refused: TestAnswer =
        serde_json::from_value(json!({"ok": false, "error": "the key was refused"})).expect("the answer");
    let _ = app.update(Message::ProvTested(Ok(refused)));
    let mut ui = sim(&app);
    ui.find("the key was refused").expect("the message of the server");
}
