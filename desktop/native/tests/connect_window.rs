//! Settings > Connect other apps in the window: the four examples of the SDK card and the plan
//! the server writes for a coding tool (`spane-claude-code`).

use fllm_app::app::{App, Message, View};
use fllm_app::connect::{snippet, Plan, CONNECT_HELP, SDK_HELP, SHELLS};
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
    Simulator::with_size(settings(), Size::new(1280.0, 4200.0), app.view())
}

/// The pane starts the way the page starts it: the four tabs, the example of the first one and
/// nothing of the plan yet.
#[test]
fn the_pane_starts_the_way_the_page_starts_it() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("Connect other apps").expect("the title of the first card");
    // both descriptions are one paragraph, the way `app-item-desc` of the page is drawn — the
    // sentence is found whole, so it is not put together out of pieces beside each other
    ui.find(SDK_HELP).expect("the description of the first card as one sentence");
    ui.find(CONNECT_HELP).expect("the description of the second card as one sentence");
    ui.find("Connect a coding tool").expect("the title of the second card");
    for tab in ["Claude Code CLI", "OpenAI Python", "OpenAI TypeScript", "cURL"] {
        ui.find(tab).expect("one of the four tabs");
    }
    // the example of the tab that is chosen, drawn whole (`#preSdkSnippet`)
    let example = snippet("claude", "http://127.0.0.1:9", "fllm-sk-local", "fllm-studio-demo", 0.7);
    ui.find(example.as_str()).expect("the example of Claude Code");
    // the header of that card says bash for every tab — the Studio never writes it again
    ui.find("bash").expect("the language of the header");
    // the line about the key: nothing here requires one yet
    ui.find("The key \"fllm\" is a placeholder: this server accepts any value.")
        .expect("the line of statusLine");
}

/// A plan is drawn in the order of the page, and the shell of the select chooses which
/// environment is shown (`render()` of `connect.js`).
#[test]
fn a_plan_is_drawn_in_the_order_of_the_page_and_by_the_shell() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let plan: Plan = serde_json::from_value(json!({
        "notes": ["Read the notes of the plan."],
        "env": {"sh": ["export FOO=bar"], "powershell": ["$env:FOO = 'bar'"], "cmd": []},
        "files": [{"path": "codex-profile.toml", "content": "[model]"}],
        "snippet": {"title": "Settings", "text": "the program of the plan"},
        "start": ["codex"],
        "tools": [
            {"id": "claude-code", "what": "Claude Code (Anthropic Messages API)"},
            {"id": "codex", "what": "Codex CLI (OpenAI Responses API)"}
        ]
    }))
    .expect("a plan of the server");
    let _ = app.update(Message::ConnectLoaded(Ok(plan)));

    let mut ui = sim(&app);
    ui.find("Read the notes of the plan.").expect("the notes come first");
    ui.find("Environment").expect("the environment of the chosen shell");
    ui.find("export FOO=bar").expect("the lines of bash, the shell this machine starts with");
    ui.find("codex-profile.toml").expect("the file as its own card");
    ui.find("Settings").expect("the snippet as its own card");
    ui.find("Start").expect("the lines to start it with");
    drop(ui);

    // the shell of the select only changes which environment is drawn
    let _ = app.update(Message::ConnectShell("powershell"));
    let mut ui = sim(&app);
    ui.find("$env:FOO = 'bar'").expect("the environment of PowerShell");
    assert!(
        ui.find("export FOO=bar").is_err(),
        "the shell was chosen, so the other lines are not drawn"
    );
    drop(ui);

    // and a shell whose environment is empty draws none (`env[shell] || []`)
    let _ = app.update(Message::ConnectShell("cmd"));
    let mut ui = sim(&app);
    assert!(ui.find("Environment").is_err(), "an empty environment is not drawn");
    drop(ui);

    // a plan the server could not write says so under its select
    let _ = app.update(Message::ConnectLoaded(Err("bad tool".into())));
    let mut ui = sim(&app);
    ui.find("Could not read the settings: bad tool").expect("the line of the page's catch");
    drop(ui);
    let _ = app.update(Message::ConnectShell(SHELLS[1].0));
}

/// The tabs choose which example is drawn, and the Copy button of the card sends its text away
/// with the toast of `btnCopySdkSnippet`.
#[test]
fn the_tabs_choose_the_example_and_the_button_copies_it() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::SdkTab("python"));
    assert_eq!(app.sdk_lang, "python", "the tab that was clicked");
    // the temperature that is in the box of the sampler, which the example writes after
    // `getSampler().temperature`
    let python = snippet(
        "python",
        "http://127.0.0.1:9",
        "fllm-sk-local",
        "fllm-studio-demo",
        fllm_app::store::Prefs::default().temperature,
    );
    let mut ui = sim(&app);
    ui.find(python.as_str()).expect("the example of OpenAI Python");
    drop(ui);

    let _ = app.update(Message::SdkCopy);
    assert_eq!(app.toast.as_deref(), Some("Example copied"), "the toast of the page");
}
