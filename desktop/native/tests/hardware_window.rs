//! The hardware card in the settings window: the devices of this machine, the amber notice of what
//! runs on the CPU and the check of one card — the `spane-hardware` card of the Studio with the
//! `gpucheck.js` behind it. The pure parts are `gpucheck.rs`'s own tests.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::gpucheck::GpuResult;
use fllm_app::models::parse_status;
use fllm_app::tour::{Device, Hardware};
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

fn device(id: &str, name: &str, backend: &str, bytes: u64) -> Device {
    Device {
        id: id.into(),
        free_memory_bytes: 0,
        name: name.into(),
        backend: backend.into(),
        api: String::new(),
        dev_type: String::new(),
        total_memory_bytes: bytes,
        available: true,
        has_id: !id.is_empty(),
    }
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1100.0, 900.0), app.view())
}

/// One drawing of the window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
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
    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 2800.0),
        app.view(),
    );
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// The card says what the Studio's card says: the description of its header, the devices with the
/// check button where one belongs, and no button on the CPU and the NPU (`gpuCheckHtml`).
#[test]
fn the_card_lists_the_devices_and_tests_the_ones_that_can_be_tested() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HardwareLoaded(Ok(Hardware {
        auto_choice: String::from("cuda:0"),
        devices: vec![
            device("cpu0", "Intel Core i5", "cpu", 0),
            device("cuda:0", "NVIDIA GeForce RTX 4060", "cuda", 8_000_000_000),
            // an NPU with memory stays in the list (one without memory would be out, as in panels.js)
            device("npu0", "AMD XDNA", "npu", 4_000_000_000),
            device("ghost", "Ghost GPU", "vulkan", 0),
        ],
    })));

    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("The processor and graphics cards fLLM can use, and a few ways to check that they work.")
        .expect("the words of the Studio's header");
    ui.find("Compute devices").expect("the title of the second card");
    ui.find("Intel Core i5").expect("the processor is in the list");
    ui.find("NVIDIA GeForce RTX 4060").expect("the card is in the list");
    ui.find("· 7.5 GB").expect("the memory of the card beside it (format_bytes of the app)");
    ui.find("AMD XDNA").expect("the NPU is in the list");
    assert!(
        ui.find("Ghost GPU").is_err(),
        "a device without memory stays out of the list (panels.js)"
    );
    // the sentence above the list is the Studio's (`hwChoiceText`), and the card of the automatic
    // choice says so beside its name
    ui.find(
        "With the automatic choice fLLM runs models on: NVIDIA GeForce RTX 4060 (NVIDIA CUDA). The processor is the \
fallback.",
    )
    .expect("the choice sentence of the Studio");
    ui.find("· the automatic choice").expect("the role of the card the automatic choice picks (deviceRole)");
    ui.find("Processor").expect("the backend label of the CPU is its own word (deviceBackendLabel)");
    ui.find("NVIDIA CUDA").expect("the backend label of the card");
    drop(ui);

    // the check button exists for the card, and not for the processor or the NPU
    let cuda_checks = app
        .update(Message::GpuCheck("cuda:0".into()));
    drop(cuda_checks);
    assert_eq!(app.gpu_busy, vec!["cuda:0".to_string()], "the card is testing now");
    let mut ui = sim(&app);
    ui.find("Testing ...").expect("the button says so while it runs");
    drop(ui);

    // the answer puts its line under the button, coloured by the Studio's three classes
    let _ = app.update(Message::GpuChecked {
        id: String::from("cuda:0"),
        results: Ok(vec![
            GpuResult { device: "cuda:0".into(), status: "ok".into(), verdict: "the card does the work".into() },
        ]),
    });
    assert!(app.gpu_busy.is_empty(), "the test is done");
    let mut ui = sim(&app);
    ui.find("cuda:0: the card does the work").expect("the result line of gpuVerdictLine");
    save_png(&app, "settings-hardware");
}

/// The NPU card says what `npuCardModel` says: the summary of its state with the version and the
/// device, the formats the API names, and the four paragraphs of the Studio's explanation — while
/// nothing was asked yet it only says it is checking.
#[test]
fn the_npu_card_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("NPU (neural processing unit)").expect("the title of the card");
        ui.find("Checking for an NPU ...").expect("nothing asked yet (npuCardModel with null)");
    }

    let state = json!({
        "state": "ready",
        "runtime": {"version": "2025.1.0", "managed": true},
        "devices": [{"full_name": "Intel AI Boost", "architecture": "Arrow Lake"}],
        "hardware": [{"name": "Intel AI Boost", "vendor": "intel", "supported": true}],
        "quantizations": {"eight_bit": ["q8_0"], "float16": ["f16"]},
    });
    let _ = app.update(Message::NpuState(Ok(state)));

    let mut ui = sim(&app);
    ui.find("Ready (experimental), OpenVINO 2025.1.0. Choose NPU under Advanced when you load a model; \
long prompts then run on it. Intel AI Boost (platform Arrow Lake).")
        .expect("the summary of npuCardModel");
    ui.find("Intel AI Boost (Intel)").expect("the hardware row");
    ui.find("Intel NPU").expect("the badge of a supported one");
    ui.find("Computed as 8 bit with one scale per output row (the default)").expect("the first format");
    ui.find("Half precision").expect("the second format");
    ui.find("Explanation").expect("the explanation of the Studio stands under it");
    ui.find(
        "It is never chosen automatically. To use it, pick NPU under Advanced when you load a model, or start fLLM \
with --device npu (or FLLM_NPU_HOST=1). If the NPU cannot take a model, fLLM runs it on the processor and says why \
here.",
    )
    .expect("the fourth paragraph as one line");
    save_png(&app, "settings-npu");
}

/// The amber notice follows the status: without fallbacks nothing shows, with them the title, the
/// summary and the lines of the Studio appear (`hwFallbackNotice`).
#[test]
fn the_notice_of_what_runs_on_the_cpu_follows_the_status() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("Reading the devices of this machine …").expect("nothing was asked yet");
        assert!(ui.find("The whole model runs on the CPU").is_err(), "no status, no notice");
    }

    let status = parse_status(&json!({
        "setup": false,
        "model": {"name": "m", "parameters": 0, "max_context": 4096},
        "metrics": {"uptime_secs": 1.0, "generated_tokens_total": 0, "slots_running": 0},
        "device_fallbacks": {
            "fallback": true, "mode": "cpu", "device": "NVIDIA GPU", "backend": "cuda",
            "whole_model_on_cpu": true,
        },
    }));
    let _ = app.update(Message::HeroStatus(Ok(status)));

    let mut ui = sim(&app);
    ui.find("The whole model runs on the CPU").expect("the title of the notice");
    ui.find("NVIDIA GPU (cuda) is not used for this model.").expect("the line of the Studio");
    ui.find(
        "Run fllm-rs doctor --gpu in a terminal, or press \"Test this GPU\" below, to see in one line whether the card does the work.",
    )
    .expect("the hint under the lines (one line, as the Studio says it)");

    drop(ui);

    // without fallbacks it goes away again
    let status = parse_status(&json!({
        "setup": false,
        "model": {"name": "m", "parameters": 0, "max_context": 4096},
        "metrics": {"uptime_secs": 2.0, "generated_tokens_total": 0, "slots_running": 0},
    }));
    let _ = app.update(Message::HeroStatus(Ok(status)));
    let mut ui = sim(&app);
    assert!(
        ui.find("The whole model runs on the CPU").is_err(),
        "everything runs where it should again"
    );
}
