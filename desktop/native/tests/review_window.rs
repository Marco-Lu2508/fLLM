//! Row 81: what the review features do once they run in the window — the switch of
//! `/autoreview`, the three tools the user runs as cards, the check after an edit and the
//! self-review between a run's last answer and its `done`. (`tests/review_fixtures.rs` pins
//! what the page's own functions answer; this file drives the machinery.)

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::code::{CallState, Change, CodeMsg, Mode, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::json;
use std::sync::Arc;

const FOLDER: &str = "/home/marco/projects/fllm";

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
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn code(app: &mut App, m: CodeMsg) {
    let _ = app.update(Message::Code(m));
}

fn open_folder(app: &mut App) {
    code(
        app,
        CodeMsg::Opened(Ok(Workspace {
            root: FOLDER.into(),
            name: "fllm".into(),
            branch: Some("main".into()),
        })),
    );
    let tools: Vec<Tool> = [
        ("read_file", Permission::Read),
        ("edit_file", Permission::Edit),
        ("bash", Permission::Command),
        ("review", Permission::Read),
        ("diagnostics", Permission::Read),
    ]
    .into_iter()
    .map(|(n, p)| Tool::server(n, p))
    .collect();
    code(app, CodeMsg::ToolsLoaded(Ok((tools, json!([])))));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

fn notes(app: &App) -> Vec<String> {
    app.code()
        .items
        .iter()
        .filter_map(|i| match i {
            fllm_app::code::Item::Note(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

/// The last card with this id (`None` when the id is not in the list).
fn call(app: &App, id: &str) -> Option<fllm_app::code::Call> {
    app.code().items.iter().find_map(|i| match i {
        fllm_app::code::Item::Call(c) if c.id == id => Some(c.clone()),
        _ => None,
    })
}

/// The newest card (the tool the user just ran).
fn last_call(app: &App) -> fllm_app::code::Call {
    app.code()
        .items
        .iter()
        .rev()
        .find_map(|i| match i {
            fllm_app::code::Item::Call(c) => Some(c.clone()),
            _ => None,
        })
        .expect("a card")
}

fn tool_call(app: &mut App, index: usize, id: &str, name: &str, arguments: &str) {
    code(
        app,
        CodeMsg::Stream(fllm_app::api::StreamEvent::ToolCall {
            index,
            id: Some(id.into()),
            name: Some(name.into()),
            arguments: arguments.into(),
        }),
    );
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 4000.0), app.view())
}

/// `/autoreview`: only `on` and `off` change the switch, the notice follows either way, and the
/// window keeps the choice like the page keeps it in `fllm.studio.codesettings.v1`.
#[test]
fn autoreview_switches_and_notices_like_the_page() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/autoreview on");
    assert!(app.code().auto_review, "the switch is on");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Review before done: on. When a run changes files the agent reads its own diff and the diagnostics once more and fixes what is wrong."),
        "the notice of the page"
    );
    assert!(app.store().prefs.auto_review, "the window keeps it");

    // an argument the switch does not know changes nothing, and the notice reads the switch
    say(&mut app, "/autoreview maybe");
    assert!(app.code().auto_review, "an unknown word does not switch it off");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Review before done: on. When a run changes files the agent reads its own diff and the diagnostics once more and fixes what is wrong."),
        "the notice still reads the switch"
    );

    say(&mut app, "/autoreview off");
    assert!(!app.code().auto_review, "the switch is off");
    assert_eq!(notes(&app).last().map(String::as_str), Some("Review before done: off."));
    assert!(!app.store().prefs.auto_review, "the window keeps that too");
}

/// `/check`, `/review` and `/security-review` run their tool as a card (`userTool` of the page)
/// and show what the server answers.
#[test]
fn the_review_tools_run_as_cards_with_the_arguments_of_the_page() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/check src");
    let card = last_call(&app);
    assert_eq!(card.name, "diagnostics", "/check runs the diagnostics tool");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&card.args).unwrap(),
        json!({ "path": "src" }),
        "the whole tail is the path, as the page passes it"
    );
    assert!(matches!(card.state, CallState::Running), "the card runs");
    let id = card.id.clone();
    code(&mut app, CodeMsg::UserRunDone(id, Ok("No problems found".into()), false));
    assert!(
        matches!(last_call(&app).state, CallState::Done { ok: true, output } if output == "No problems found"),
        "the answer stands under the card"
    );

    say(&mut app, "/review staged");
    let card = last_call(&app);
    assert_eq!(card.name, "review", "/review runs the review tool");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&card.args).unwrap(),
        json!({ "scope": "staged" }),
        "the first word picks the scope"
    );
    code(&mut app, CodeMsg::UserRunDone(card.id.clone(), Ok("looks good".into()), false));

    say(&mut app, "/security-review");
    let card = last_call(&app);
    assert_eq!(card.name, "review", "the security review is the review tool");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&card.args).unwrap(),
        json!({ "scope": "diff", "security": true }),
        "its own two arguments"
    );
    code(&mut app, CodeMsg::UserRunDone(card.id.clone(), Ok("clean".into()), false));
}

/// The check after an edit: the card is done while it runs, the findings stand in their own
/// block, and the model gets the answer with them between the two markers.
#[test]
fn the_check_after_an_edit_gives_the_model_its_problems() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ToggleAutoCheck(true));
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));

    say(&mut app, "fix the line");
    tool_call(&mut app, 0, "c1", "edit_file", r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#);
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited one line.".into())));
    let card = call(&app, "c1").expect("the card of the edit");
    assert!(matches!(card.state, CallState::Done { ok: true, .. }), "the edit is done");

    // the diagnostics runs and comes back (its task has no executor in this test)
    code(
        &mut app,
        CodeMsg::CheckDone {
            id: "c1".into(),
            reply: "Edited one line.".into(),
            outputs: vec!["src/lib.rs:3: unused variable".into()],
        },
    );
    let card = call(&app, "c1").expect("the card of the edit");
    assert_eq!(
        card.check.as_deref(),
        Some("src/lib.rs:3: unused variable"),
        "the block of the card is the raw finding"
    );
    let sent = card.sent.expect("the model gets the findings with the answer");
    assert!(sent.starts_with("Edited one line."), "the answer stays first");
    assert!(sent.contains("[automatic check found problems]"), "and the findings between the markers");
    // what the model is sent back over the wire is exactly that text
    let wire = app.code().wire(None);
    let tool = wire.iter().find(|m| m.role == "tool").expect("the answer of the call");
    assert_eq!(tool.content, sent, "the wire sends what the page sends");
}

/// The switch off: nothing holds the round and the model gets the plain answer.
#[test]
fn without_the_switch_the_edit_is_answered_plain() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));

    say(&mut app, "fix the line");
    tool_call(&mut app, 0, "c1", "edit_file", r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#);
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited one line.".into())));
    let card = call(&app, "c1").expect("the card of the edit");
    assert!(card.check.is_none() && card.sent.is_none(), "no check ran");
    let wire = app.code().wire(None);
    let tool = wire.iter().find(|m| m.role == "tool").expect("the answer of the call");
    assert_eq!(tool.content, "Edited one line.", "the plain answer");
}

/// The self-review: a run that changed files asks the review tool and the diagnostics before it
/// counts as done, says so in the list, sends the model its own feedback — unseen in the window,
/// seen over the wire — and stops after the second round when nothing a machine can state is
/// left.
#[test]
fn the_self_review_reads_the_work_and_sends_the_model_back_once() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));
    // the run starts after a change that is already there: only what comes after it is reviewed
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![Change { id: 5, path: "old.rs".into(), kind: "edit".into() }])),
    );

    say(&mut app, "fix the bug");
    tool_call(&mut app, 0, "c1", "edit_file", r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#);
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited.".into())));

    // the model is done answering — the review starts between that and the `done`
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(app.code().running, "the run stays open while the review works");
    assert!(notes(&app).last().map(String::as_str) != Some("Reviewing my own changes (1 file) before finishing."), "the notice follows the answers");

    // the changes are read again: this run changed `src/lib.rs`, the old change stays out
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![
            Change { id: 5, path: "old.rs".into(), kind: "edit".into() },
            Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() },
        ])),
    );
    code(
        &mut app,
        CodeMsg::ReviewTools(fllm_app::code::ReviewAnswers {
            review: Some("[low] the diff matches the task".into()),
            diagnostics: vec![
                ("src/lib.rs".into(), "src/lib.rs:3: unused".into()),
                // a clean answer stays out of the feedback (`isCleanDiagnostic` of the page)
                ("readme.md".into(), "No problems found".into()),
            ],
        }),
    );
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Reviewing my own changes (1 file) before finishing."),
        "the notice of round one"
    );
    let feedback = app
        .code()
        .items
        .iter()
        .find_map(|i| match i {
            fllm_app::code::Item::ReviewFeedback(t) => Some(t.clone()),
            _ => None,
        })
        .expect("the feedback is in the list");
    assert!(feedback.starts_with("Self-review before you finish."), "the words of the page");
    assert!(feedback.contains("[low] the diff matches the task"), "with the review behind them");
    assert!(feedback.contains("src/lib.rs:3: unused"), "and the diagnostics that are not clean");
    assert!(!feedback.contains("No problems found"), "a clean answer is not sent");
    let mut ui = sim(&app);
    assert!(
        ui.find("Self-review before you finish. Check your own work against the task now.").is_err(),
        "the window does not draw the feedback"
    );
    drop(ui);
    let wire = app.code().wire(None);
    assert!(
        wire.iter().any(|m| m.role == "user" && m.content == feedback),
        "the model gets the feedback as its next user message"
    );

    // the second round answers with nothing a machine can state — the run is simply over
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() }])),
    );
    code(&mut app, CodeMsg::ReviewTools(fllm_app::code::ReviewAnswers::default()));
    assert!(!app.code().running, "the second round with nothing to fix ends the run");
    assert!(app.code().review.is_none(), "the review is over");
    assert_eq!(
        notes(&app).iter().filter(|n| n.starts_with("The review found problems")).count(),
        0,
        "the page notices nothing when the second round has no problem"
    );
}

/// The two switches are the window's store, so they are there on the next start too (the page
/// reads `fllm.studio.codesettings.v1` at startup).
#[test]
fn the_two_switches_come_back_from_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store {
        theme: "light".into(),
        prefs: fllm_app::store::Prefs { auto_review: true, auto_check: true, ..Default::default() },
        ..Default::default()
    }
    .save(&path)
    .unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    assert!(app.code().auto_review, "the review comes back on");
    assert!(app.code().auto_check, "the check comes back on");
}

/// Two rounds at most — even when the second one found something: after its feedback ran, the
/// page's `for (round <= 2)` is over and the run ends (`selfReview`).
#[test]
fn the_second_round_is_the_last_even_when_it_found_problems() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));

    say(&mut app, "fix the bug");
    tool_call(&mut app, 0, "c1", "edit_file", r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#);
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited.".into())));
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done)); // round one ends, review starts

    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() }])),
    );
    code(&mut app, CodeMsg::ReviewTools(fllm_app::code::ReviewAnswers::default()));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Reviewing my own changes (1 file) before finishing."),
        "round one"
    );

    // the model worked on the feedback and answered again: round two begins
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() }])),
    );
    code(
        &mut app,
        CodeMsg::ReviewTools(fllm_app::code::ReviewAnswers {
            review: None,
            diagnostics: vec![("src/lib.rs".into(), "src/lib.rs:9: error".into())],
        }),
    );
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("The review found problems: another round (1 file with diagnostics)."),
        "round two is announced with what it found"
    );
    assert!(app.code().running, "the second feedback goes to the model");

    // the second round ran and answered — now the page's loop is over, whatever it found
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(!app.code().running, "the second round is the last");
    assert!(app.code().review.is_none(), "no third round begins");
}

/// Plan mode never reviews: the page calls `selfReview` only when `code.perm !== 'plan'`.
#[test]
fn plan_mode_ends_without_a_review() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::Plan));

    say(&mut app, "read the file");
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(!app.code().running, "plan mode ends the run at once");
    assert!(app.code().review.is_none(), "no review starts");
    assert!(notes(&app).is_empty(), "nothing stands in the list");
}
