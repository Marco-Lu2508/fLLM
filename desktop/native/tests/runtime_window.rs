//! Settings > Runtime in the window: the three cards of `runtime.js` with the lines the server
//! sent, the two questions before a switch goes on, and the line that says a save worked (the
//! `spane-runtime` pane of `shell.html`).

use fllm_app::app::{App, Message, View};
use fllm_app::runtime::{Confirm, Status};
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
    Simulator::with_size(settings(), Size::new(1280.0, 1600.0), app.view())
}

/// Writes the drawing of the current window under `ui-snapshots/<name>.png`. The window is as
/// tall as the page needs: the three cards sit under six other ones, and the scrollable of the
/// settings clips at its own height (the drawings of the hardware card are taken the same way).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let mut ui = Simulator::with_size(settings(), Size::new(1280.0, 6200.0), app.view());
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

/// The answer of `GET /api/studio/runtime` as the server sends it: a workspace sandbox that
/// works, turbo off, a machine that can do computer use, and a status that says what is enforced.
fn answer(turbo: bool, enabled: bool) -> Status {
    serde_json::from_value(json!({
        "sandbox": if turbo { "off" } else { "workspace" },
        "sandbox_from_env": false,
        "turbo": turbo,
        "computer": {
            "compiled": true, "enabled": enabled, "forbidden": false, "available": true,
            "reason": null, "platform": "linux", "rate_per_minute": 30, "max_steps": 40,
            "steps_used": 0, "has_screenshot": false, "panic_chord": "Ctrl+Alt+Shift+Esc", "max_side": 1280
        },
        "file": "/home/marco/.fllm/runtime.json",
        "sandbox_available": true,
        "sandbox_status": {
            "mode": "workspace", "active": !turbo, "requested": null,
            "sandbox": {"available": true, "path": "/usr/bin/bwrap", "reason": null},
            "platform": "linux", "mechanism": "bubblewrap",
            "enforced": ["the filesystem outside the workspace"],
            "not_enforced": ["the network"],
            "verification": "the paths of /proc/self/mountinfo",
            "default_mode": "workspace", "terminal": {"available": true}
        }
    }))
    .expect("the answer of the server")
}

/// The three cards with the lines of that answer, and the boxes with the numbers in it.
#[test]
fn the_three_cards_say_what_the_server_said() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::RuntimeLoaded(Ok(answer(false, false))));

    let mut ui = sim(&app);
    ui.find("Sandbox for agent commands").expect("the first card");
    ui.find(fllm_app::runtime::SANDBOX_HELP).expect("its description");
    ui.find("sandbox: workspace · bubblewrap is active").expect("the line under the radios");
    let status = answer(false, false);
    let details = fllm_app::runtime::sandbox_details(status.sandbox_status.as_ref().expect("the status"));
    ui.find(details.as_str()).expect("the block of the status");
    ui.find("Computer use").expect("the second card");
    ui.find("Possible on this computer (linux).").expect("what this machine can do");
    ui.find("At most").expect("the two limits");
    ui.find("actions per task").expect("the second limit");
    ui.find("Turbo").expect("the third card");
    ui.find("Enable turbo").expect("its switch");
    save_png(&app, "runtime-cards");
}

/// A switch asks before it goes on, and the answer of the save is drawn and said (`Saved.` of
/// `rtStatus`).
#[test]
fn a_switch_asks_before_it_goes_on_and_the_answer_says_so() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::RuntimeLoaded(Ok(answer(false, false))));

    let _ = app.update(Message::RuntimeTurbo(true));
    assert_eq!(app.runtime_confirm, Some(Confirm::Turbo), "the Studio asks first");
    let mut ui = sim(&app);
    ui.find("Turbo removes every permission question and the sandbox. Continue?")
        .expect("the question of `confirmDialog`");
    save_png(&app, "runtime-confirm");
    drop(ui);

    // the dialog is answered with Cancel: nothing is asked of the server again
    let _ = app.update(Message::RuntimeConfirmCancel);
    assert!(app.runtime_confirm.is_none(), "the dialog is closed");

    // computer use asks its own question
    let _ = app.update(Message::RuntimeComputer(true));
    assert_eq!(app.runtime_confirm, Some(Confirm::Computer));
    let mut ui = sim(&app);
    ui.find(Confirm::Computer.question()).expect("the question of computer use");
    drop(ui);
    let _ = app.update(Message::RuntimeConfirmOk);
    assert!(app.runtime_confirm.is_none(), "the dialog is answered");

    // the answer of the save draws the card again and says it worked
    let _ = app.update(Message::RuntimeSaved(Ok(answer(true, true))));
    let mut ui = sim(&app);
    ui.find("Saved.").expect("the line of rtStatus");
    ui.find("sandbox: off · sandbox is off · turbo is on: no questions, sandbox off")
        .expect("the card follows the answer");
    ui.find("Possible on this computer (linux). It is ON.").expect("the switch went on");
}

/// A limit box keeps what was typed, and a text that is not a number of at least one sends
/// nothing (`if (n >= 1)` of `runtime.js`).
#[test]
fn a_limit_box_keeps_what_was_typed_and_a_text_sends_nothing() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::RuntimeLoaded(Ok(answer(false, false))));

    let _ = app.update(Message::RuntimeRate("12.5".into()));
    let mut ui = sim(&app);
    ui.find("12.5").expect("the box holds what was typed");
    drop(ui);

    // a text in the box is not a number: the save sends nothing and says nothing
    let _ = app.update(Message::RuntimeRate("abc".into()));
    let _ = app.update(Message::RuntimeRateSave);
    let mut ui = sim(&app);
    ui.find("abc").expect("the box still holds it");
    assert!(ui.find("Saved.").is_err(), "nothing was sent, so nothing was saved");
}

/// A server that does not answer says so in the line of `rtStatus` (`Could not read the
/// setting: …` of `load()`).
#[test]
fn a_read_that_failed_says_what_went_wrong() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::RuntimeLoaded(Err("no server at http://127.0.0.1:9: connection refused".into())));

    let mut ui = sim(&app);
    ui.find("Could not read the setting: no server at http://127.0.0.1:9: connection refused")
        .expect("the line of rtStatus");
    // nothing of the card is drawn before an answer came (`status: None`)
    assert!(ui.find("sandbox:").is_err(), "no line of a status that was never read");
}
