//! Row 78: the hooks of the agent loop, fired from the window (`fireHook` of code.js). The
//! Settings card with the list itself is row 66; here is what the list changes at the four
//! places the page asks a hook — the submit of a prompt, the stop after a run, the moment a
//! card starts waiting — and what its answers do.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::code::{CodeMsg, CodeState, HookFire, Item, Mode, Permission, Tool, Workspace};
use fllm_app::hooks::{Hook, HooksAnswer, PluginHook};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use serde_json::json;
use std::sync::Arc;

const FOLDER: &str = "/home/marco/projects/fllm";

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
    let tools: Vec<Tool> = [("read_file", Permission::Read), ("bash", Permission::Command)]
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
            Item::Note(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

/// The list of hooks as the pane reads it: one own hook per event the test names.
fn hooks_for(events: &[&str]) -> HooksAnswer {
    HooksAnswer {
        hooks: events
            .iter()
            .map(|e| {
                (
                    (*e).to_string(),
                    vec![Hook {
                        matcher: String::new(),
                        command: "echo hi".into(),
                        prompt: String::new(),
                        on: String::new(),
                        timeout_seconds: 30.0,
                        enabled: None,
                    }],
                )
            })
            .collect(),
        file: ".fllm/hooks.json".into(),
        from_plugins: Vec::new(),
    }
}

/// Without a list the window asks nothing and the turn starts at once — the state every other
/// test of this repo starts from.
#[test]
fn without_a_known_hook_the_turn_starts_at_once() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "do it");
    assert!(app.code().running, "the run began without waiting for anything");
    assert_eq!(app.code().hook_events, Vec::<String>::new(), "nothing is known");
}

/// `user_prompt_submit`: its `context` rides along with the message the model gets, in its own
/// block after the words of the box (`expanded` of `submitText`), and the words stay as typed.
#[test]
fn the_submit_hook_puts_its_context_before_the_model() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["user_prompt_submit"]))));
    assert_eq!(
        app.code().hook_events,
        ["user_prompt_submit".to_string()],
        "the events of the list are the gates"
    );

    say(&mut app, "do it");
    assert!(!app.code().running, "the turn waits for the hook");
    code(
        &mut app,
        CodeMsg::SubmitHookDone {
            prompt: "do it".into(),
            display: "do it".into(),
            answer: HookFire { blocked: false, reason: String::new(), context: "check the tests".into() },
        },
    );
    assert!(app.code().running, "the turn starts with the answer");
    let sent = app
        .code()
        .items
        .iter()
        .find_map(|i| match i {
            Item::User { sent, .. } => sent.clone(),
            _ => None,
        })
        .expect("the message carries what the model is sent");
    assert_eq!(sent, "do it\n\n[hooks]\ncheck the tests", "the words and its block");

    // an answer without a context leaves the message alone
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(!app.code().running, "the turn is over");
    say(&mut app, "again");
    code(
        &mut app,
        CodeMsg::SubmitHookDone {
            prompt: "again".into(),
            display: "again".into(),
            answer: HookFire::default(),
        },
    );
    let second = app
        .code()
        .items
        .iter()
        .rev()
        .find_map(|i| match i {
            Item::User { sent, .. } => Some(sent.clone()),
            _ => None,
        })
        .expect("the second message");
    assert_eq!(second, None, "nothing to append, nothing to send beyond it");
}

/// The stop hook after the run: a blocked answer notices in the list, gives the model its words
/// unseen over the wire, and the model runs once more — that run ends plainly. A hook brought by
/// a plugin opens the gate as well.
#[test]
fn a_blocked_stop_hook_sends_the_model_back_exactly_once() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(HooksAnswer {
        hooks: std::collections::HashMap::new(),
        file: ".fllm/hooks.json".into(),
        from_plugins: vec![PluginHook {
            event: "stop".into(),
            matcher: None,
            command: "echo go on".into(),
            plugin: "test".into(),
            enabled: None,
            timeout_seconds: None,
        }],
    })));
    assert_eq!(app.code().hook_events, ["stop".to_string()], "the plugin's event counts too");

    say(&mut app, "do it");
    assert!(app.code().running, "no submit hook was asked for");
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(
        app.code().running,
        "the stop hook is being asked: the turn is not over yet"
    );
    code(
        &mut app,
        CodeMsg::StopHookDone(HookFire { blocked: true, reason: "run the tests".into(), context: String::new() }),
    );
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("A stop hook asks the agent to continue: run the tests"),
        "the notice of the page"
    );
    assert!(app.code().running, "the model runs once more");
    let wire = app.code().wire(None);
    assert!(
        wire.iter().any(|m| m.role == "user"
            && m.content == "A stop hook asks you to continue before you finish: run the tests"),
        "the words go to the model, not into the window"
    );

    // that run ends plainly: the page checks its hook once
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(!app.code().running, "the second end does not ask again");

    // a hook that does not block ends the turn at once, without a notice
    say(&mut app, "again");
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(app.code().running, "the hook is asked again for this turn");
    code(&mut app, CodeMsg::StopHookDone(HookFire::default()));
    assert!(!app.code().running, "the turn is over");
    assert!(
        notes(&app).last().map(String::as_str) != Some("A stop hook asks the agent to continue: "),
        "nothing stands in the list when the hook passes"
    );
}

/// The window's own list is the gate: a waiting card fires `notification` (the page fires it in
/// `waitFor`, without awaiting) — and with the gate off the card simply waits.
#[test]
fn a_waiting_card_fires_the_notification_hook_only_when_the_list_has_one() {
    // The fire itself has no answer to observe (the page ignores it); what is observable is the
    // gate: with `notification` in the list a waiting card reaches `fire_hook`, without it the
    // run never leaves `finish_round` for one. Both states stand here; the request that is made
    // is the page's own (`{ kind: 'tool', name }`) and lives in the code.
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    // the gate is off: a risky call waits without anything else happening
    say(&mut app, "clean up");
    code(
        &mut app,
        CodeMsg::Stream(fllm_app::api::StreamEvent::ToolCall {
            index: 0,
            id: Some("c1".into()),
            name: Some("bash".into()),
            arguments: r#"{"command":"rm -rf /"}"#.into(),
        }),
    );
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(
        matches!(
            app.code().items.iter().find_map(|i| match i {
                Item::Call(c) if c.id == "c1" => Some(c.state.clone()),
                _ => None,
            }),
            Some(fllm_app::code::CallState::Pending)
        ),
        "the card waits (a risky command asks however the switch stands)"
    );
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["notification"]))));
    assert_eq!(app.code().hook_events, ["notification".to_string()], "the gate opens");
    // the next waiting card leaves `finish_round` with a request — nothing answers, which is
    // exactly what the page's fire-and-forget does with it
    code(
        &mut app,
        CodeMsg::Stream(fllm_app::api::StreamEvent::ToolCall {
            index: 1,
            id: Some("c2".into()),
            name: Some("bash".into()),
            arguments: r#"{"command":"rm -rf /tmp/x"}"#.into(),
        }),
    );
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(
        app.code().items.iter().any(|i| matches!(i, Item::Call(c) if c.id == "c2")),
        "the waiting card is there either way"
    );
}

/// The order of the tail: the self-review runs first, and when it is over the stop hook is
/// asked (`await selfReview(…)` before `fireHook('stop', …)` in `submitText`) — one turn, both.
#[test]
fn after_the_review_the_stop_hook_is_asked() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["stop"]))));
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![fllm_app::code::Change { id: 5, path: "old.rs".into(), kind: "edit".into() }])),
    );

    say(&mut app, "fix the bug");
    code(
        &mut app,
        CodeMsg::Stream(fllm_app::api::StreamEvent::ToolCall {
            index: 0,
            id: Some("c1".into()),
            name: Some("edit_file".into()),
            arguments: r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#.into(),
        }),
    );
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited.".into())));
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done)); // the review begins
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![fllm_app::code::Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() }])),
    );
    code(&mut app, CodeMsg::ReviewTools(fllm_app::code::ReviewAnswers::default()));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Reviewing my own changes (1 file) before finishing."),
        "the review ran"
    );

    // the feedback ran and the model answered: the second round finds something a machine can
    // state, sends the model back once more, and when THAT answer ends the review is over —
    // only now does the page ask its stop hook
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    code(
        &mut app,
        CodeMsg::ChangesLoaded(Ok(vec![fllm_app::code::Change { id: 7, path: "src/lib.rs".into(), kind: "edit".into() }])),
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
        "round two is announced"
    );
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(
        app.code().running,
        "the stop hook is being asked after the review — the turn waits for its answer"
    );
    code(&mut app, CodeMsg::StopHookDone(HookFire::default()));
    assert!(!app.code().running, "and its answer ends the turn");
}

/// The pane keeps the list it read; a `HooksReload` (its Reload button) brings the new events
/// into the gates with it.
#[test]
fn reloading_the_list_updates_the_gates() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["user_prompt_submit"]))));
    assert_eq!(app.code().hook_events, ["user_prompt_submit".to_string()]);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["stop", "notification"]))));
    assert_eq!(app.code().hook_events, ["notification".to_string(), "stop".to_string()], "sorted");
}

/// Plan mode reaches the stop hook as any run does — the page checks it with no mode in sight
/// (`if (code.running)` only) — and its review stays away as row 81 pinned.
#[test]
fn plan_mode_still_asks_the_stop_hook_and_never_reviews() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["stop"]))));
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::Plan));
    say(&mut app, "read the file");
    code(&mut app, CodeMsg::Stream(fllm_app::api::StreamEvent::Done));
    assert!(app.code().running, "the stop hook is being asked, as on the page");
    assert!(
        !notes(&app).iter().any(|n| n.starts_with("Reviewing my own changes")),
        "no self-review in plan mode"
    );
    code(&mut app, CodeMsg::StopHookDone(HookFire::default()));
    assert!(!app.code().running, "and the turn ends with its answer");
}

/// The state of one call, read the way the window keeps it.
#[allow(dead_code)]
fn state_of(s: &CodeState, id: &str) -> Option<fllm_app::code::CallState> {
    s.items.iter().find_map(|i| match i {
        Item::Call(c) if c.id == id => Some(c.state.clone()),
        _ => None,
    })
}
