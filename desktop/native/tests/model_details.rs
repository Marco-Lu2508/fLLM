//! Settings > Model details and *Save as default* of the answer settings.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::modeldetails::{Facts, FactsMsg};
use fllm_app::settings::defaults_body;
use fllm_app::store::{Prefs, Store};
use iced_test::Simulator;
use serde_json::json;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 4096))));
    let _ = app.update(Message::Go(View::Settings));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1000.0, 9000.0), app.view())
}

fn facts() -> Facts {
    Facts::from_json(&json!({"model": {"name": "qwen", "arch": "qwen3", "parameter_size": "4B", "parameters": 4022468096u64, "size_bytes": 2497280256u64,
        "quantization": "Q4_K_M", "max_context": 8192, "trained_context": 40960, "vocab_size": 151936, "aliases": ["gpt-4o-mini"], "chat_template": "<|im_start|>{{ role }}"}}))
    .unwrap()
}

#[test]
fn the_page_shows_the_facts_the_other_names_and_the_chat_format() {
    let (mut app, _dir) = app();
    let mut ui = sim(&app);
    ui.find("Reading …").expect("before the server has answered");
    drop(ui);
    let _ = app.update(Message::Facts(FactsMsg::Loaded(Ok(Some(facts())))));
    let mut ui = sim(&app);
    ui.find("Model details").expect("the card");
    ui.find("qwen3 · 4B (4,022,468,096 parameters) · 2.5 GB").expect("the facts");
    ui.find("8192 tokens now (the model was trained for 40960) · 151936 known word pieces").expect("the length");
    ui.find("Q4_K_M").expect("the quantization");
    ui.find("gpt-4o-mini").expect("an other name");
    ui.find("Chat format of the model").expect("the second card");
    ui.find("<|im_start|>{{ role }}").expect("the template");
}

#[test]
fn without_a_model_the_page_says_so() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Facts(FactsMsg::Loaded(Ok(None))));
    let mut ui = sim(&app);
    ui.find("No model is loaded. Load one on the Models page.").expect("the words");
    ui.find("The model carries no chat format.").expect("and no format");
}

#[test]
fn a_name_is_added_to_the_loaded_model_and_taken_away_again() {
    let (mut app, _dir) = app();
    let none = app.update(Message::Facts(FactsMsg::Add));
    assert_eq!(none.units(), 0, "an empty field sends nothing");
    let _ = app.update(Message::Facts(FactsMsg::Alias("  claude-3-sonnet ".into())));
    let nomodel = app.update(Message::Facts(FactsMsg::Add));
    assert_eq!(nomodel.units(), 0, "an alias points to the loaded model: none loaded, nothing sent");
    let _ = app.update(Message::Facts(FactsMsg::Loaded(Ok(Some(facts())))));
    let add = app.update(Message::Facts(FactsMsg::Add));
    assert_eq!(add.units(), 1);
    let twice = app.update(Message::Facts(FactsMsg::Add));
    assert_eq!(twice.units(), 0, "not twice at once");
    let done = app.update(Message::Facts(FactsMsg::Done(Ok("The name claude-3-sonnet was added.".into()))));
    assert_eq!(done.units(), 1, "the facts are read again");
    let mut ui = sim(&app);
    ui.find("The name claude-3-sonnet was added.").expect("the note");
    drop(ui);
    let remove = app.update(Message::Facts(FactsMsg::Remove("gpt-4o-mini".into())));
    assert_eq!(remove.units(), 1);
    let _ = app.update(Message::Facts(FactsMsg::Done(Err("not found".into()))));
    let mut ui = sim(&app);
    ui.find("not found").expect("the reason");
}

#[test]
fn the_defaults_are_the_manual_values_and_the_prompt() {
    let mut p = Prefs::default();
    // automatic: no sampling values, no prompt of one's own
    assert_eq!(defaults_body(&p), json!({"system_prompt": ""}));
    p.manual_sampling = true;
    p.temperature = 0.5;
    p.top_p = 0.8;
    p.sampling.top_k = 40;
    p.sampling.seed = Some(7);
    p.custom_system = true;
    p.system_prompt = "Be brief.".into();
    let b = defaults_body(&p);
    assert_eq!(b["temperature"], json!(0.5));
    assert_eq!(b["top_p"], json!(0.8f32));
    assert_eq!(b["top_k"], json!(40));
    assert_eq!(b["seed"], json!(7));
    assert_eq!(b["system_prompt"], json!("Be brief."));
    assert!(b.get("mirostat").is_some() && b.get("dry_multiplier").is_some());
    p.sampling.seed = None;
    assert_eq!(defaults_body(&p)["seed"], serde_json::Value::Null, "no seed: random");
    // the prompt is the user's only while it is custom
    p.custom_system = false;
    assert_eq!(defaults_body(&p)["system_prompt"], json!(""));
}

#[test]
fn saving_the_defaults_says_what_the_server_answered() {
    let (mut app, _dir) = app();
    let sent = app.update(Message::SaveDefaults);
    assert_eq!(sent.units(), 1);
    let _ = app.update(Message::DefaultsSaved(Ok(())));
    let mut ui = sim(&app);
    ui.find("Save as default").expect("the button");
    ui.find("Saved as the engine default.").expect("the answer");
    drop(ui);
    let _ = app.update(Message::DefaultsSaved(Err("temperature must be a non-negative finite number".into())));
    let mut ui = sim(&app);
    ui.find("It could not be saved: temperature must be a non-negative finite number").expect("the reason");
}

/// The drawing of the page for the search "model details", in both themes (`ui-snapshots/`).
#[test]
fn the_cards_draw_in_both_themes() {
    for theme in ["light", "dark"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chats.json");
        Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
        let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
        let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 4096))));
        let _ = app.update(Message::Go(View::Settings));
        let _ = app.update(Message::Facts(FactsMsg::Loaded(Ok(Some(facts())))));
        let _ = app.update(Message::SettingsSearch("chat format".into()));
        let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
        let mut ui = Simulator::with_size(settings, iced::Size::new(1000.0, 900.0), app.view());
        ui.find("Model details").expect("the card is kept by the search");
        let shot = ui.snapshot(&app.theme()).expect("the software renderer draws the view");
        let out = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))));
        std::fs::create_dir_all(&out).unwrap();
        shot.matches_image(out.join(format!("model-details-{theme}.png"))).expect("the picture is written");
    }
}
