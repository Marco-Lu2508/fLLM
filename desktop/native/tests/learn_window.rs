//! Row 88 in the Code page: the switch and its sync with the server, the run that is handed over,
//! the thumbs, the watch for the learning step and `/learn`, `/good`, `/bad`.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, Permission, Tool, Workspace};
use fllm_app::learn::{LastRun, Status};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use serde_json::{json, Value};
use std::sync::Arc;

const FOLDER: &str = "/home/marco/projects/fllm";

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn code(app: &mut App, m: CodeMsg) {
    let _ = app.update(Message::Code(m));
}

fn task(app: &mut App, m: CodeMsg) -> iced::Task<Message> {
    app.update(Message::Code(m))
}

fn open_folder(app: &mut App) {
    code(app, CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: Some("main".into()) })));
    let def = |n: &str| json!({ "type": "function", "function": { "name": n, "description": n, "parameters": { "type": "object", "properties": {} } } });
    let mut tools: Vec<Tool> = vec![Tool::server("read_file", Permission::Read), Tool::server("skill", Permission::Read)];
    let mut defs: Vec<Value> = tools.iter().map(|t| def(&t.name)).collect();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

fn status(enabled: Option<bool>) -> Status {
    Status { enabled, ..Default::default() }
}

#[test]
fn the_switch_of_the_server_is_the_truth() {
    let (mut app, _dir) = app();
    assert!(app.code().learn, "learning is on by default");
    assert!(app.store().prefs.learn);
    code(&mut app, CodeMsg::LearnStatus(Some(status(Some(false)))));
    assert!(!app.code().learn, "the server says off");
    assert!(!app.store().prefs.learn, "and the store keeps it");
    code(&mut app, CodeMsg::LearnStatus(Some(status(Some(true)))));
    assert!(app.code().learn && app.store().prefs.learn);
    // a server that does not say reads as on (`undefined !== false`)
    code(&mut app, CodeMsg::LearnStatus(Some(status(Some(false)))));
    code(&mut app, CodeMsg::LearnStatus(Some(status(None))));
    assert!(app.code().learn, "no answer for `enabled` is on");
    // a failed read changes nothing
    code(&mut app, CodeMsg::LearnStatus(Some(status(Some(false)))));
    code(&mut app, CodeMsg::LearnStatus(None));
    assert!(!app.code().learn);
}

#[test]
fn the_first_start_notice_is_asked_for_only_while_learning_is_on() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let off = task(&mut app, CodeMsg::LearnStatus(Some(status(Some(false)))));
    assert_eq!(off.units(), 0, "no notice is fetched for a switch that is off");
    let on = task(&mut app, CodeMsg::LearnStatus(Some(status(Some(true)))));
    assert_eq!(on.units(), 1, "the notice is fetched");
    let n = app.code().items.len();
    code(&mut app, CodeMsg::LearnNotice(String::new()));
    assert_eq!(app.code().items.len(), n, "nothing to say, nothing written");
    code(&mut app, CodeMsg::LearnNotice("Learning is on.".into()));
    assert!(matches!(app.code().items.last(), Some(fllm_app::code::Item::Note(t)) if t == "Learning is on."));
}

#[test]
fn the_checkbox_sets_the_switch_and_tells_the_server() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let task = task(&mut app, CodeMsg::LearnToggle(false));
    assert!(!app.code().learn && !app.store().prefs.learn);
    assert_eq!(task.units(), 1, "the server is told");
    code(&mut app, CodeMsg::LearnSwitched(Err("no route".into())));
    assert_eq!(app.toast.as_deref(), Some("no route"), "a refusal is a toast, as the page toasts it");
}

#[test]
fn a_run_is_handed_over_with_the_skills_it_read() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    say(&mut app, "deploy it");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c1".into()), name: Some("skill".into()), arguments: r#"{"action":"read","name":"deploy"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("the skill".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Delta("Done.".into())));
    let end = task(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(!app.code().running);
    let last = app.code().learn_last.clone().expect("the run was handed over");
    assert_eq!(last.skills, ["deploy"]);
    assert_eq!(last.workspace, FOLDER);
    assert!(!last.session.is_empty());
    assert!(end.units() >= 2, "the specs are read again and the run goes to the server: {} units", end.units());
}

#[test]
fn nothing_is_handed_over_while_learning_is_off() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    code(&mut app, CodeMsg::LearnToggle(false));
    say(&mut app, "hi");
    code(&mut app, CodeMsg::Stream(StreamEvent::Delta("Hello.".into())));
    let end = task(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(app.code().learn_last.is_none());
    assert_eq!(end.units(), 1, "only the specs are read");
}

#[test]
fn a_failed_run_is_learned_from_too() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    say(&mut app, "hi");
    let end = task(&mut app, CodeMsg::Stream(StreamEvent::Error("context length exceeded".into())));
    assert!(app.code().learn_last.is_some(), "a run that ended in an error is handed over, as a failed one");
    assert_eq!(end.units(), 1);
}

#[test]
fn a_stopped_run_is_not_learned_from() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    say(&mut app, "hi");
    code(&mut app, CodeMsg::Stop);
    assert!(app.code().learn_last.is_none());
}

#[test]
fn the_thumbs_need_a_skill_of_the_last_run() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let none = task(&mut app, CodeMsg::LearnThumb("up"));
    assert_eq!(none.units(), 0);
    assert_eq!(app.toast.as_deref(), Some("No skill was used in the last run."));
    // a run that read no skill: the thumb has nothing to be about
    say(&mut app, "hello");
    code(&mut app, CodeMsg::Stream(StreamEvent::Delta("Hi.".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(app.code().learn_last.as_ref().is_some_and(|l| l.skills.is_empty()));
    app.toast = None;
    assert_eq!(task(&mut app, CodeMsg::LearnThumb("up")).units(), 0, "no request for a run that read no skill");
    assert_eq!(app.toast.as_deref(), Some("No skill was used in the last run."));
    // a run that read a skill
    say(&mut app, "deploy it");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c1".into()), name: Some("skill".into()), arguments: r#"{"action":"read","name":"deploy"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("x".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    app.toast = None;
    let sent = task(&mut app, CodeMsg::LearnThumb("down"));
    assert_eq!(sent.units(), 1, "the thumb goes to the server");
    code(&mut app, CodeMsg::LearnThumbDone("down", vec!["deploy".into()], Ok((1, vec!["Two failures.".into()]))));
    assert_eq!(app.toast.as_deref(), Some("Noted: deploy did not work. A revision is drafted on the next run; Two failures.."));
    code(&mut app, CodeMsg::LearnThumbDone("up", vec!["deploy".into()], Ok((1, vec![]))));
    assert_eq!(app.toast.as_deref(), Some("Noted: deploy worked."));
    code(&mut app, CodeMsg::LearnThumbDone("up", vec![], Err("no server".into())));
    assert_eq!(app.toast.as_deref(), Some("no server"));
}

#[test]
fn the_watch_ends_with_a_toast_when_the_step_is_over_and_stops_after_sixty_polls() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let queued_not = task(&mut app, CodeMsg::LearnQueued(false, 10.0));
    assert_eq!(queued_not.units(), 0, "a run the server did not take is not watched");
    let first = task(&mut app, CodeMsg::LearnQueued(true, 10.0));
    assert_eq!(first.units(), 1, "the first look comes after four seconds");
    let running = Status { running: true, last_run: Some(LastRun { t: 11.0, summary: "x".into(), lines: vec![] }), ..Default::default() };
    let again = task(&mut app, CodeMsg::LearnPolled(Some(running), 10.0, 1));
    assert_eq!(again.units(), 1, "a step that still runs is asked about again");
    let old = Status { last_run: Some(LastRun { t: 9.0, summary: "old".into(), lines: vec![] }), ..Default::default() };
    assert_eq!(task(&mut app, CodeMsg::LearnPolled(Some(old), 10.0, 5)).units(), 1, "a step older than the run is not the one");
    assert_eq!(task(&mut app, CodeMsg::LearnPolled(None, 10.0, 59)).units(), 1, "a failed read is asked again");
    assert_eq!(task(&mut app, CodeMsg::LearnPolled(None, 10.0, 60)).units(), 0, "after sixty polls the page gives up");
    assert!(app.toast.is_none());
    let done = Status { last_run: Some(LastRun { t: 12.0, summary: "wrote the skill \"deploy\"".into(), lines: vec![] }), ..Default::default() };
    let end = task(&mut app, CodeMsg::LearnPolled(Some(done), 10.0, 3));
    assert!(end.units() >= 1, "and the lists of Settings are read again: {} units", end.units());
    assert_eq!(app.toast.as_deref(), Some("Learned: wrote the skill \"deploy\""));
    // a step with nothing to say is no toast
    app.toast = None;
    let quiet = Status { last_run: Some(LastRun { t: 12.0, ..Default::default() }), ..Default::default() };
    code(&mut app, CodeMsg::LearnPolled(Some(quiet), 10.0, 3));
    assert!(app.toast.is_none());
}

fn typed(app: &mut App, text: &str) -> iced::Task<Message> {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    task(app, CodeMsg::Send)
}

#[test]
fn the_commands_learn_good_and_bad() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    // `/learn off` changes the switch at once and asks the server for the rest
    let off = typed(&mut app, "/learn off");
    assert!(!app.code().learn && !app.store().prefs.learn);
    assert!(off.units() >= 1);
    use fllm_app::learn::LearnPlan;
    code(&mut app, CodeMsg::LearnCommand(LearnPlan::Show(Some(false)), Some(Status { enabled: Some(false), ask_first: true, pending: Some(2.0), ..Default::default() }), None));
    assert!(matches!(app.code().items.last(), Some(fllm_app::code::Item::Note(t)) if t == "Learning from my sessions: off. It proposes and waits for your yes. 2 proposals wait for a decision."));
    // `/learn project off` is read from the server's answer
    code(&mut app, CodeMsg::LearnCommand(LearnPlan::Project(Some(false)), Some(Status { enabled: Some(false), ..Default::default() }), None));
    assert!(matches!(app.code().items.last(), Some(fllm_app::code::Item::Note(t)) if t == "Learning in this folder: off."));
    assert!(!app.code().learn);
    code(&mut app, CodeMsg::LearnCommand(LearnPlan::Project(None), Some(Status { enabled: Some(true), ..Default::default() }), None));
    assert!(app.code().learn, "the folder follows the global setting: the answer says on");
    assert!(matches!(app.code().items.last(), Some(fllm_app::code::Item::Note(t)) if t == "Learning in this folder: follows the global setting."));
    // the usage
    let _ = typed(&mut app, "/learn project maybe");
    assert!(matches!(app.code().items.last(), Some(fllm_app::code::Item::Note(t)) if t.starts_with("Usage: /learn project on|off|default")));
    // `/good` and `/bad` are the thumbs
    app.toast = None;
    let _ = typed(&mut app, "/good");
    assert_eq!(app.toast.as_deref(), Some("No skill was used in the last run."));
}

#[test]
fn a_second_run_hands_over_only_its_own_skills() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    say(&mut app, "deploy it");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c1".into()), name: Some("skill".into()), arguments: r#"{"action":"read","name":"deploy"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("x".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert_eq!(app.code().learn_last.as_ref().unwrap().skills, ["deploy"]);
    say(&mut app, "and now something else");
    code(&mut app, CodeMsg::Stream(StreamEvent::Delta("ok".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(app.code().learn_last.as_ref().unwrap().skills.is_empty(), "the second run read no skill: {:?}", app.code().learn_last);
}

#[test]
fn the_side_has_the_switch_and_the_thumbs() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let click = |app: &App, label: &str| {
        let mut ui = iced_test::Simulator::with_size(
            iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() },
            iced::Size::new(1280.0, 1800.0),
            app.view(),
        );
        ui.click(label).unwrap_or_else(|_| panic!("{label}"));
        ui.into_messages().collect::<Vec<_>>()
    };
    let sent = click(&app, "Learn from my sessions");
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::LearnToggle(false))]), "the box is on, a click turns it off: {sent:?}");
    let sent = click(&app, "Skills helped");
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::LearnThumb("up"))]), "{sent:?}");
    let sent = click(&app, "Skills did not help");
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::LearnThumb("down"))]), "{sent:?}");
    code(&mut app, CodeMsg::LearnToggle(false));
    let sent = click(&app, "Learn from my sessions");
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::LearnToggle(true))]), "{sent:?}");
}
