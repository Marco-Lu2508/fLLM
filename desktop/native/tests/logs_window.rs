//! Settings > Logs in the window: the card of `spane-logs` with the lines of the server, the two
//! buttons of its header and the box that keeps its height (`models.js` fills it the same way).

use fllm_app::app::{App, Message, View};
use fllm_app::localapi::Logs;
use fllm_app::store::Store;
use iced::Size;
use iced_test::Simulator;

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
    Simulator::with_size(settings(), Size::new(1280.0, 2600.0), app.view())
}

/// Writes the drawing of the Logs card alone under `ui-snapshots/<name>.png`: the settings page is
/// taller than a texture the CI GPUs accept (wgpu refuses more than 4096 there).
fn save_png(app: &App, name: &str, body: &str) {
    let t = app.tokens();
    let theme = app.theme();
    let content = fllm_app::localapi::logs_card(t, body);
    let mut ui = Simulator::with_size(settings(), Size::new(1100.0, 800.0), content);
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

/// The card before the answer, the lines after it, and the copy of exactly what the box holds.
#[test]
fn the_card_says_what_the_server_wrote() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    // until the answer came the box says what it says at first (`Loading ...` of `shell.html`)
    let mut ui = sim(&app);
    ui.find("Logs").expect("the title of the card");
    ui.find(fllm_app::localapi::LOGS_HELP).expect("the line under it");
    ui.find("Refresh").expect("the first button of the header");
    ui.find("Copy").expect("the second one");
    ui.find("Loading ...").expect("the box while the read runs");
    drop(ui);

    // the answer of `GET /api/studio/logs?lines=300`
    let _ = app.update(Message::LogsLoaded(Ok(Logs {
        lines: vec!["the engine started".into(), "listening on 127.0.0.1:8000".into()],
    })));
    let mut ui = sim(&app);
    ui.find("the engine started\nlistening on 127.0.0.1:8000")
        .expect("the lines of the server in the box");
    drop(ui);

    // *Copy* takes exactly that text to the clipboard (`btnCopyLogs` of `models.js`)
    let mut ui = sim(&app);
    ui.click("Copy").expect("the button of the header");
    let copied: Vec<String> = ui
        .into_messages()
        .filter_map(|m| match m {
            Message::CopyText(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(copied, ["the engine started\nlistening on 127.0.0.1:8000"], "the whole box");

    save_png(&app, "settings-logs", "the engine started\nlistening on 127.0.0.1:8000");
}

/// An empty log and a read that failed say what `refreshLogs` writes for them.
#[test]
fn an_empty_log_and_a_failed_read_say_so() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::LogsLoaded(Ok(Logs { lines: Vec::new() })));
    let mut ui = sim(&app);
    ui.find("Nothing has been logged yet.").expect("the sentence of an empty log");
    drop(ui);

    let _ = app.update(Message::LogsLoaded(Err("no server".into())));
    let mut ui = sim(&app);
    ui.find("The logs could not be read.").expect("the sentence of a failed read");
    assert_eq!(app.logs_text, "The logs could not be read.", "and the box holds it");
}
