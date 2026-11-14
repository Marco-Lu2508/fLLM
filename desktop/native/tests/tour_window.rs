//! The welcome tour in the window: the first start opens it once, the steps walk with Next, Back
//! and Skip, the last step says *Get started*, Settings brings it back, and *Open Models* goes to
//! the Models page — the `tourModal` of the Studio drawn by iced.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::models::parse_overview;
use fllm_app::store::{ChatRecord, Role, Store};
use fllm_app::tour::Hardware;
use iced_test::Simulator;
use serde_json::json;

/// The hub and the hardware of a small computer, as the two routes answer them.
fn tour_loaded() -> Message {
    let overview = parse_overview(&json!({
        "available_memory_bytes": 6_000_000_000u64,
        "loaded": {"name": "Qwen3-4B"},
        "recommended": ["a:4b"],
        "catalog": [
            {"id": "a:4b", "display_name": "Alpha 4B", "default_quant": "Q4_K_M",
             "parameters_billion": 4.0, "context_length": 32768, "vision": false, "fits": true,
             "variants": []}
        ],
        "local": [], "cached": []
    }));
    let hardware = Hardware::from_json(&json!({
        "devices": [
            {"id": "cpu0", "backend": "cpu", "api": "native", "name": "Intel Core i5",
             "dev_type": "cpu", "available": true, "total_memory_bytes": 0u64},
            {"id": "gpu0", "backend": "cuda", "api": "native", "name": "NVIDIA GeForce RTX 4060",
             "dev_type": "gpu", "available": true, "total_memory_bytes": 8_000_000_000u64}
        ]
    }));
    Message::TourLoaded {
        overview: Ok(overview),
        hardware: Ok(hardware),
    }
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view())
}

/// Writes the drawing of the current window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR")
            .unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))),
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

/// A first start with nothing on the machine opens the tour once, with the title of the first step
/// and its two paragraphs (the first start of `voice.js`: no key, no chats).
#[test]
fn a_first_start_opens_the_tour_once_with_its_welcome() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(tour_loaded());
    assert_eq!(app.tour, Some(0), "the first step opens");
    assert!(app.prefs().tour_seen, "the key is written when it opens");

    let mut ui = sim(&app);
    ui.find("Welcome to fLLM").expect("the title of the first step");
    ui.find("fLLM runs language models on your own computer, on the processor or on any graphics card. Your chats, files and models stay on this machine.")
        .expect("the first paragraph of the Studio");
    ui.find("Next").expect("the step walks forward");
    ui.find("Skip").expect("the tour can be left at once");
    ui.find("Back").expect_err("the first step has nothing behind it");
    save_png(&app, "tour-welcome");

    // once it has been shown, the same answer does not open it again
    drop(ui);
    app.tour = None;
    let _ = app.update(tour_loaded());
    assert_eq!(app.tour, None, "it is shown once");
}

/// The steps walk forward and back: the second step says what this computer is, the third what
/// model is there, the last one has the tips and says *Get started* without *Skip*.
#[test]
fn the_steps_walk_forward_and_back() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(tour_loaded());

    let _ = app.update(Message::TourNext);
    assert_eq!(app.tour, Some(1));
    {
        let mut ui = sim(&app);
        ui.find("Your computer").expect("the title of the second step");
        // the line of the Studio: the processor, the card with its memory, the usable memory
        let line = format!(
            "Intel Core i5 \u{b7} NVIDIA GeForce RTX 4060 ({}) \u{b7} {} usable for a model",
            fllm_app::app::format_bytes(8_000_000_000),
            fllm_app::app::format_bytes(6_000_000_000)
        );
        ui.find(line.as_str()).expect("the line about this computer");
        ui.find("This is the memory a model can use. Bigger models are smarter but need more memory and are slower; the Models page tells you which ones fit.")
            .expect("the hint of the Studio");
        save_png(&app, "tour-computer");
    }

    let _ = app.update(Message::TourNext);
    assert_eq!(app.tour, Some(2));
    {
        let mut ui = sim(&app);
        ui.find("Pick a model").expect("the title of the third step");
        // the sentence runs over three spans, so the selector is the first of them with its space
        ui.find("A model is already loaded: ").expect("the loaded model is named");
        ui.find("Qwen3-4B").expect("the name itself, in bold");
        ui.find("Open Models").expect("the button of the step");
        save_png(&app, "tour-model");
    }

    let _ = app.update(Message::TourNext);
    assert_eq!(app.tour, Some(3));
    {
        let mut ui = sim(&app);
        ui.find("Three tips").expect("the title of the last step");
        ui.find("Get started").expect("the last step says Get started");
        ui.find("Skip").expect_err("the last step has no Skip");
        save_png(&app, "tour-tips");
    }

    // Back walks one step back
    let _ = app.update(Message::TourBack);
    assert_eq!(app.tour, Some(2));

    // *Get started* closes it
    let _ = app.update(Message::TourNext);
    assert_eq!(app.tour, Some(3));
    let _ = app.update(Message::TourNext);
    assert_eq!(app.tour, None, "the last step closes the tour");
}

/// *Open Models* closes the tour and goes to the Models page, *Skip* only closes it.
#[test]
fn open_models_leaves_the_tour_and_goes_to_the_models_page() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(tour_loaded());
    let _ = app.update(Message::TourOpenModels);
    assert_eq!(app.tour, None, "the tour closes");
    assert_eq!(app.page(), View::Models, "and the Models page is what opens");

    let _ = app.update(Message::TourShow);
    assert_eq!(app.tour, Some(0), "it opens again on demand");
    let _ = app.update(Message::TourSkip);
    assert_eq!(app.tour, None, "Skip closes it");
}

/// Settings > General shows the tour again: the button is on the page, and it opens the tour over
/// the chat (the `btnShowTour` of `shell.html` with `switchSurface('chat')`).
#[test]
fn the_settings_page_brings_the_tour_back() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("Show the welcome tour").expect("the button of the General card");
    ui.find("A short introduction to fLLM, your computer and the first steps.")
        .expect("the description of the Studio");
    save_png(&app, "tour-settings");
    drop(ui);

    let _ = app.update(Message::TourShow);
    assert_eq!(app.tour, Some(0), "the tour opens");
    assert_eq!(app.page(), View::Chat, "over the chat, the way the Studio opens it");
    assert!(app.prefs().tour_seen, "the key is written");
}

/// An existing user (chats on the machine) never sees the tour, however the answers come in.
#[test]
fn an_existing_user_never_sees_the_tour() {
    let mut store = Store::default();
    store.chats.push(ChatRecord::new(
        1,
        "old chat",
        vec![fllm_app::store::Record {
            role: Role::Assistant,
            content: "hello".into(),
            reasoning: String::new(),
            stats: None,
            passages: None,
            images: Vec::new(),
            files: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            tool_cards: Vec::new(),
            sources: Vec::new(),
            research: None,
            logprobs: Vec::new(),
        }],
    ));
    let (mut app, _dir) = app_with(store);
    let _ = app.update(tour_loaded());
    assert_eq!(app.tour, None, "the chats say this is no first start");
    assert!(!app.prefs().tour_seen, "and the key is untouched");
}
