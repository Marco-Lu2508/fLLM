//! The start screen in the window: what the engine is doing right now — the heading and the words
//! under it of the Studio's `updateHero`, the line of tokens per second, and the buttons of
//! `renderHeroStart`. The fixtures of the plan itself are `panels.rs`.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::models::{parse_status, CatalogEntry, Installed, Overview, Status, Variant};
use iced_test::Simulator;
use serde_json::json;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

/// Puts the status of a server without a model into the app (the answer of the setup server).
fn setup(app: &mut App) {
    let status = parse_status(&json!({
        "setup": true,
        "last_model": "",
        "model": {"name": "", "parameters": 0, "quantization": "", "max_context": 4096},
        "metrics": {"uptime_secs": 3.0, "generated_tokens_total": 0, "slots_running": 0},
    }));
    let _ = app.update(Message::HeroStatus(Ok(status)));
}

/// One drawing of the current window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1100.0, 800.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR")
            .unwrap_or_else(|_| "/opt/data/cache/scratch/fLLM/desktop/native/ui-snapshots".into()),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// Nothing loaded, models here: the heading and the words of the Studio, the model used last
/// first with its little words, and the two pointers to the rest.
#[test]
fn nothing_loaded_offers_the_models_on_this_computer() {
    let (mut app, _dir) = app();
    setup(&mut app);
    let overview = Overview {
        loaded: None,
        loaded_detail: String::new(),
        memory: 8_000_000_000,
        installed: vec![Installed {
            name: "used.gguf".into(),
            reference: "used.gguf".into(),
            bytes: 1_000_000_000,
            where_: "downloaded".into(),
            shown: "/m/used.gguf".into(),
        }],
        catalog: vec![],
        recommended: vec![],
    };
    let _ = app.update(Message::HeroOverview(Ok(overview)));

    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 800.0),
        app.view(),
    );
    ui.find("No model loaded").expect("the heading of the Studio");
    ui.find("A model on this computer is ready. Nothing to download.").expect("the words of the plan");
    ui.find("Load used").expect("the model used last is first");
    ui.find("Choose a model file…").expect("the file dialog button");
    ui.find("Browse models").expect("the pointer to the models page");
    save_png(&app, "welcome-setup");
}

/// Nothing on disk: the words of the download, the recommended model and the dialog buttons.
#[test]
fn without_any_model_the_screen_offers_the_download() {
    let (mut app, _dir) = app();
    setup(&mut app);
    let overview = Overview {
        loaded: None,
        loaded_detail: String::new(),
        memory: 8_000_000_000,
        installed: vec![],
        catalog: vec![CatalogEntry {
            id: "qwen3-4b".into(),
            display_name: "Qwen3 4B".into(),
            default_quant: "Q4_K_M".into(),
            parameters_b: 4.0,
            context: 32768,
            vision: false,
            fits: true,
            // this variant does not fit into the memory: the Studio's meta line says so
            variants: vec![Variant { quant: "Q4_K_M".into(), bytes: 2_500_000_000, fits: false }],
            installed_ref: None,
        }],
        recommended: vec!["qwen3-4b".into()],
    };
    let _ = app.update(Message::HeroOverview(Ok(overview)));

    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 800.0),
        app.view(),
    );
    ui.find("No model loaded").expect("the heading");
    ui.find("Download one, or choose a model file you already have.").expect("the words");
    ui.find("Download and start").expect("the main button of renderHeroStart");
    ui.find("Qwen3 4B, Q4_K_M, 2.5 GB, may not fit into free memory")
        .expect("the meta line beside the button (name, quantization, weight, warning)");
    save_png(&app, "welcome-download");
}

/// A model is loaded: the heading is its name, the words under it carry its facts, and the
/// two-minute line of tokens per second is drawn with the words beside it.
#[test]
fn a_loaded_model_shows_its_name_facts_and_the_line() {
    let (mut app, _dir) = app();
    // three moments of the server's own clock: nothing, then 30 tokens in three seconds, then
    // twenty more in two — the line takes the samples while a slot runs
    for (tokens, uptime, running) in [(0u64, 0.0f64, 0u64), (30, 3.0, 2), (50, 5.0, 2)] {
        let status = parse_status(&json!({
            "setup": false,
            "last_model": "",
            "model": {
                "name": "qwen3-4b-instruct", "parameters": 4_000_000_000u64,
                "parameter_size": "4B", "quantization": "Q4_K_M", "max_context": 32768,
            },
            "metrics": {"uptime_secs": uptime, "generated_tokens_total": tokens, "slots_running": running},
        }));
        let _ = app.update(Message::HeroStatus(Ok(status)));
    }

    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 800.0),
        app.view(),
    );
    ui.find("qwen3-4b-instruct").expect("the heading is the name of the model");
    ui.find("Running locally on this computer: 4 billion parameters, 4B, Q4_K_M, 32768 tokens of context.")
        .expect("the facts line of the Studio");
    ui.find("tokens per second, the model is answering").expect("the label while a slot runs");
    ui.find("10.0").expect("the rate of the last moment: 20 tokens in two seconds");
    save_png(&app, "welcome-loaded");
}

/// The status of the Studio's own route is read with the fields the hero needs (its tests of the
/// parsing stay in models.rs; this one is the shape of the setup server).
#[test]
fn the_status_of_the_setup_server_has_what_the_screen_needs() {
    let status = parse_status(&json!({
        "setup": true,
        "last_model": "/models/used.gguf",
        "model": {"name": "", "parameters": 0, "quantization": "", "max_context": 4096},
        "metrics": {"uptime_secs": 0, "generated_tokens_total": 0, "slots_running": 0},
    }));
    let expected = Status {
        setup: true,
        last_model: "/models/used.gguf".into(),
        model_name: String::new(),
        parameters: 0,
        parameter_size: String::new(),
        quantization: String::new(),
        max_context: 4096,
        uptime_secs: 0.0,
        generated_tokens_total: 0,
        slots_running: 0,
        device_fallbacks: None,
        load_device: String::from("auto"),
    };
    assert_eq!(status, expected, "the setup server's answer");
    let plain = parse_status(&json!({}));
    assert!(!plain.setup && plain.model_name.is_empty(), "without the fields there is no setup");
}
