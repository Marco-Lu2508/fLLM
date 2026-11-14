//! The hardware card in the settings window: the devices of this machine, the amber notice of what
//! runs on the CPU and the check of one card — the `spane-hardware` card of the Studio with the
//! `gpucheck.js` behind it. The pure parts are `gpucheck.rs`'s own tests.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::gpucheck::GpuResult;
use fllm_app::models::{parse_status, Slot};
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
            // the processor carries the memory of the machine (as `host_memory` reports it)
            device("cpu0", "Intel Core i5", "cpu", 16_000_000_000),
            device("cuda:0", "NVIDIA GeForce RTX 4060", "cuda", 8_000_000_000),
            // an NPU with memory stays in the list (one without memory would be out, as in panels.js)
            device("npu0", "AMD XDNA", "npu", 4_000_000_000),
            device("ghost", "Ghost GPU", "vulkan", 0),
        ],
        host_free_memory_bytes: 4_000_000_000,
        active_cpu_tier: String::from("avx2-fma"),
        supported_cpu_tiers: vec![String::from("avx512-vnni"), String::from("avx2-fma")],
        threads: 8,
        ..Hardware::default()
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

/// The first card of the pane: the two buttons of the header and the four rows over the devices
/// (speed test, instruction set, repack switch, system memory) with the words of the Studio.
#[test]
fn the_top_card_of_the_hardware_pane_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HardwareLoaded(Ok(Hardware {
        auto_choice: String::from("cuda:0"),
        devices: vec![device("cpu0", "Intel Core i5", "cpu", 16_000_000_000)],
        host_free_memory_bytes: 4_000_000_000,
        active_cpu_tier: String::from("avx2-fma"),
        supported_cpu_tiers: vec![String::from("avx512-vnni"), String::from("avx2-fma")],
        threads: 8,
        ..Hardware::default()
    })));
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("Measure memory speed").expect("the button that measures the bandwidth");
    ui.find("Check the processor").expect("the button that runs the self test");
    ui.find("Speed test").expect("the row of the Studio");
    ui.find("Run speed test").expect("the button of the row");
    ui.find("Runs a short answer with the loaded model and measures how fast it reads and writes.")
        .expect("the help of `speedTestResult` until the first run");
    ui.find("Processor feature limit").expect("the second row");
    ui.find("Stops the processor from using its newest features. Only useful when something misbehaves; leave it on Auto.")
        .expect("its help");
    ui.find("Faster reading of long prompts").expect("the third row");
    ui.find("Prepares the model so that long texts are read faster. Uses some extra memory.")
        .expect("its help");
    ui.find("System memory").expect("the fourth row");
    ui.find("Instruction set: avx2-fma (available: avx512-vnni, avx2-fma). 8 threads.")
        .expect("the line of `refreshHardware`");
    ui.find("3.7 GB free of 14.9 GB").expect("the numbers beside the bar (format_bytes)");
    drop(ui);
    save_png(&app, "settings-hardware-top");
}

/// The speed test: refused without a model with the Studio's words, *Testing ...* while it runs,
/// and the line of the answer afterwards (`btnSpeedTest` of `models.js`).
#[test]
fn the_speed_test_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HardwareLoaded(Ok(Hardware {
        auto_choice: String::from("cuda:0"),
        devices: vec![device("cpu0", "Intel Core i5", "cpu", 16_000_000_000)],
        ..Hardware::default()
    })));
    let _ = app.update(Message::Go(View::Settings));

    // the setup server has no model to test with
    let setup = parse_status(&json!({"setup": true, "model": {"name": ""}}));
    let _ = app.update(Message::HeroStatus(Ok(setup)));
    let _ = app.update(Message::SpeedTest);
    assert_eq!(app.toast(), Some("Load a model first."));
    assert!(app.speed_test.is_none(), "nothing was sent");
    assert!(!app.speed_test_busy);

    // a loaded model: the line says so while it runs, and the button is off
    let loaded = parse_status(&json!({
        "setup": false,
        "model": {"name": "qwen2.5-0.5b-instruct", "load_options": {"device": "cuda:0"}},
    }));
    let _ = app.update(Message::HeroStatus(Ok(loaded)));
    let _ = app.update(Message::SpeedTest);
    assert!(app.speed_test_busy);
    {
        let mut ui = sim(&app);
        ui.find("Testing, this takes a few seconds ...").expect("the words while it runs");
    }

    // what the answer weighed, how fast, and the device that did it
    let timings = json!({
        "prompt_per_second": 812.5, "prompt_n": 1234,
        "predicted_per_second": 42.0, "predicted_n": 64,
    });
    let _ = app.update(Message::SpeedTestDone(Ok(timings)));
    assert!(!app.speed_test_busy, "the run is over");
    let line = app.speed_test.clone().expect("the line of the answer");
    assert_eq!(
        line,
        "Reading the prompt: 812.5 tokens per second (1,234 tokens). Writing the answer: 42.0 tokens per second (64 tokens). Device: cuda:0."
    );
    {
        let mut ui = sim(&app);
        ui.find(line.as_str()).expect("the line is what the row shows now");
    }

    // a failure says so, in the Studio's own words
    let _ = app.update(Message::SpeedTestDone(Err(String::from("The test failed."))));
    assert_eq!(
        app.speed_test.as_deref(),
        Some("The speed test failed: The test failed.")
    );
}

/// The two buttons of the header say what they do, and the report of *Check the processor* draws
/// its own card when it answered (`btnRunSimdSelftest`).
#[test]
fn the_header_buttons_say_what_they_do_and_the_processor_check_draws_itself() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        assert!(ui.find("Processor check").is_err(), "no report, no card");
    }

    let _ = app.update(Message::MeasureBandwidth);
    assert_eq!(app.toast(), Some("Measuring memory bandwidth ..."));
    let _ = app.update(Message::SelftestCpu);
    assert_eq!(app.toast(), Some("Self test running, about 5 seconds ..."));
    // the answer of the server is the card
    let _ = app.update(Message::SelftestDone(Ok(fllm_app::gpucheck::Selftest::from_json(
        &json!({
            "ok": false, "checks": 80, "failed": 1, "skipped": 2, "duration_ms": 4711.6,
            "tested_tiers": [
                {"tier": "avx2-fma", "checks": 40, "passed": 40, "failed": 0, "ok": true},
                {"tier": "scalar", "checks": 40, "passed": 39, "failed": 1, "ok": false},
            ],
        }),
    ))));

    let mut ui = sim(&app);
    ui.find("Processor check").expect("the title of the card");
    ui.find("Processor check: 79 of 80 checks passed in 4712 ms, 2 skipped (feature not supported by this processor)")
        .expect("the line of the Studio");
    ui.find("avx2-fma").expect("the first feature level");
    ui.find("PASS").expect("a row that passed");
    ui.find("FAIL").expect("the row that failed");
    ui.find("The calculations the model needs are checked against exact results for every processor feature level. Some results deviate.")
        .expect("the closing sentence");
    drop(ui);
    save_png(&app, "settings-hardware-selftest");
}

/// The AMD NPU card: away when this machine has none of it, and with the words of `amdNpuModel`
/// when it does.
#[test]
fn the_amd_npu_card_shows_itself_only_when_there_is_one() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        assert!(
            ui.find("AMD NPU (Ryzen AI): native kernels and FastFlowLM").is_err(),
            "no answer yet: no card"
        );
    }

    let _ = app.update(Message::AmdNpuState(Ok(json!({"state": "no_npu"}))));
    {
        let mut ui = sim(&app);
        assert!(ui.find("AMD NPU (Ryzen AI): native kernels and FastFlowLM").is_err(), "no NPU: no card");
    }

    let state = json!({
        "state": "ready",
        "flm": {"version": "0.4.1"},
        "hardware": [{"name": "Ryzen AI 9 HX 370"}],
        "models": [
            {"tag": "qwen-7b-q4", "installed": true},
            {"tag": "llama-8b", "installed": false},
        ],
    });
    let _ = app.update(Message::AmdNpuState(Ok(state)));

    let mut ui = sim(&app);
    ui.find("AMD NPU (Ryzen AI): native kernels and FastFlowLM").expect("the title of the card");
    ui.find("Ready. Choose a model and run it on the NPU.").expect("the summary of amdNpuModel");
    ui.find("Ryzen AI 9 HX 370").expect("the hardware row");
    ui.find("Run on the NPU").expect("the chosen model is on disk: it can start");
    ui.find("FastFlowLM is a separate program (its NPU kernels are AMD's, free to use). It has its own model files, so a model here is not the same file as one in your fLLM model list.")
        .expect("the note the Studio always writes");
    assert!(ui.find("Download").is_err(), "the model is on disk: nothing to download");
    assert!(ui.find("Stop").is_err(), "nothing is running");
    drop(ui);
    save_png(&app, "settings-amd-npu");
}

/// The runtime actions of the NPU card: the button of the Studio's card, and the question in
/// front of the DELETE (`npuSetupAction` of `npu.js`).
#[test]
fn the_runtime_can_be_installed_and_asked_about_before_it_goes() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::NpuState(Ok(json!({
        "state": "runtime_missing",
        "runtime": {"managed": true},
        "setup": {"download_bytes": 140_000_000u64, "pinned_version": "2025.0.1"},
    }))));
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("Install the runtime (about 140 MB)").expect("the button of npuCardModel");
    ui.find("Remove the runtime").expect("a managed runtime can go away");
    drop(ui);

    // the button asks first (`confirmDialog`)
    let _ = app.update(Message::NpuRemove);
    assert!(app.npu_remove, "the dialog is open");
    let mut ui = sim(&app);
    ui.find("Remove the OpenVINO runtime?").expect("the question of the Studio");
    ui.find("The NPU cannot be used until it is installed again.").expect("what it means");
    ui.find("Cancel").expect("the way out");
    ui.find("OK").expect("the way on");
    drop(ui);
    save_png(&app, "settings-npu-remove");

    let _ = app.update(Message::NpuRemoveCancel);
    assert!(!app.npu_remove, "cancelled: nothing happens");
    let _ = app.update(Message::NpuRemove);
    let _ = app.update(Message::NpuRemoveOk);
    assert!(!app.npu_remove, "confirmed: the dialog is gone and the request goes out");
}

/// Writing one of the two engine settings says what the Studio says, and one the server refused
/// adds nothing (`selSimdTierCap` and `chkCpuRepack` of `listeners-panels.js`).
#[test]
fn a_setting_that_was_written_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::ConfigSaved {
        result: Ok(json!({})),
        toast: "CPU instruction set changed",
        hardware: true,
    });
    assert_eq!(app.toast(), Some("CPU instruction set changed"));

    // the Studio shows nothing when the setting could not be written: the last words stay
    let _ = app.update(Message::ConfigSaved {
        result: Err(String::from("the server answered 500")),
        toast: "Weight repacking changed",
        hardware: false,
    });
    assert_eq!(
        app.toast(),
        Some("CPU instruction set changed"),
        "a refused write adds nothing of its own"
    );
}

/// The table of the scheduler: which slot is answering a chat right now
/// (`hwSlotsTableWrap` of `panels.js`).
#[test]
fn the_slots_table_says_which_chat_is_being_answered() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("Measuring ...")
            .expect("the row holds what shell.html writes into it until the report arrives");
    }
    let _ = app.update(Message::SlotsLoaded(Ok(vec![
        Slot { id: 0, state: String::from("idle"), n_ctx: 8192, running: false },
        Slot { id: 1, state: String::from("processing"), n_ctx: 4096, running: true },
    ])));

    let mut ui = sim(&app);
    ui.find("Chats answered at the same time").expect("the title of the Studio");
    ui.find("Slot").expect("the first column");
    ui.find("Max Context").expect("the third column");
    ui.find("#0").expect("the first slot");
    ui.find("8192 tok").expect("the context it was started with");
    ui.find("Idle").expect("it waits for a question");
    ui.find("#1").expect("the second slot");
    ui.find("Running").expect("that one is working");
    drop(ui);
    save_png(&app, "settings-hardware-slots");
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
