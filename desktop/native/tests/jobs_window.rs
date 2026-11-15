//! Row 87: the *Background jobs* block of the Code page — a command the agent starts with
//! `background: true` is kept with its state, *Output* opens its box and reads it, *Stop* ends it,
//! a new session forgets them. The logic is pinned against the page in `tests/jobs_fixtures.rs`.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::Arc;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

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

fn open_folder(app: &mut App, bash_output_arg: &str) {
    code(app, CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None })));
    let def = |n: &str, prop: &str| json!({ "type": "function", "function": { "name": n, "description": n, "parameters": { "type": "object", "properties": { prop: {} } } } });
    let mut tools = vec![Tool::server("bash", Permission::Command), Tool::server("bash_output", Permission::Read), Tool::server("bash_kill", Permission::Command)];
    let mut defs = vec![def("bash", "command"), def("bash_output", bash_output_arg), def("bash_kill", bash_output_arg)];
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
    code(app, CodeMsg::SetMode(fllm_app::code::Mode::Auto));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

fn start_job(app: &mut App, id: &str, command: &str) {
    say(app, "start it");
    code(app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some(id.into()), name: Some("bash".into()), arguments: json!({ "command": command, "background": true }).to_string() }));
    code(app, CodeMsg::Stream(StreamEvent::Done));
    code(app, CodeMsg::ToolDone(id.into(), Ok("Started background job j1. Read its output with bash_output.".into())));
    code(app, CodeMsg::Stop); // the composer's own Stop button is not the job's
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 1800.0), app.view())
}

#[test]
fn a_command_in_the_background_is_kept_with_its_state() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    assert!(sim(&app).find("Commands started in the background show up here.").is_ok(), "an empty block says what it is for");
    start_job(&mut app, "c1", "npm run dev");
    let jobs = &app.code().jobs;
    assert_eq!(jobs.len(), 1);
    assert_eq!((jobs[0].id.as_str(), jobs[0].command.as_str(), jobs[0].status.as_str(), jobs[0].open), ("j1", "npm run dev", "running", false));
    let mut ui = sim(&app);
    assert!(ui.find("Background jobs").is_ok());
    assert!(ui.find("npm run dev").is_ok());
    assert!(ui.find("running").is_ok());
    assert!(ui.find("Stop").is_ok(), "a job that runs can be stopped");
    assert!(ui.find("Commands started in the background show up here.").is_err());
}

#[test]
fn a_command_that_is_no_background_job_is_no_job() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    say(&mut app, "list");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c1".into()), name: Some("bash".into()), arguments: r#"{"command":"ls"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Started background job j1".into())));
    assert!(app.code().jobs.is_empty());
    // a failed call is no job either
    say(&mut app, "again");
}

#[test]
fn the_output_button_opens_the_box_reads_it_and_closes_it() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "make");
    assert_eq!(app.code().job_arg_name(), "job_id", "the server names the id");
    let first = app.update(Message::Code(CodeMsg::JobOutput("j1".into())));
    assert_eq!(first.units(), 1, "the output is read from the server");
    assert!(app.code().jobs[0].open);
    assert!(sim(&app).find("(no output yet)").is_ok(), "an open box that has nothing yet says so");
    code(&mut app, CodeMsg::JobOutputDone("j1".into(), Ok("building...\nprocess exited with exit code 0".into())));
    let jobs = &app.code().jobs;
    assert_eq!((jobs[0].status.as_str(), jobs[0].out.as_str()), ("done", "building...\nprocess exited with exit code 0"));
    let mut ui = sim(&app);
    assert!(ui.find("building...\nprocess exited with exit code 0").is_ok(), "the output is drawn");
    assert!(ui.find("Stop").is_err(), "a job that is over has no Stop");
    drop(ui);
    // the second click closes the box and asks nothing
    let second = app.update(Message::Code(CodeMsg::JobOutput("j1".into())));
    assert_eq!(second.units(), 0);
    assert!(!app.code().jobs[0].open);
}

#[test]
fn stop_marks_the_job_and_a_refusal_is_a_toast() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "make");
    let stop = app.update(Message::Code(CodeMsg::JobKill("j1".into())));
    assert_eq!(stop.units(), 1);
    code(&mut app, CodeMsg::JobKillDone("j1".into(), Err("not running".into())));
    assert_eq!(app.toast.as_deref(), Some("not running"));
    assert_eq!(app.code().jobs[0].status, "running", "a refusal changes nothing");
    code(&mut app, CodeMsg::JobKillDone("j1".into(), Err(String::new())));
    assert_eq!(app.toast.as_deref(), Some("The job could not be stopped."));
    code(&mut app, CodeMsg::JobKillDone("j1".into(), Ok("killed".into())));
    assert_eq!(app.code().jobs[0].status, "killed");
}

#[test]
fn what_the_model_reads_of_a_job_updates_the_list_too() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "make");
    // the model reads the output with the tool itself: the box keeps it
    say(&mut app, "read it");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c2".into()), name: Some("bash_output".into()), arguments: r#"{"job_id":"j1"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c2".into(), Ok("all tests finished".into())));
    let jobs = &app.code().jobs;
    assert_eq!((jobs[0].status.as_str(), jobs[0].out.as_str()), ("done", "all tests finished"));
}

#[test]
fn the_buttons_send_the_id_of_their_job() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "make");
    for (label, want_output) in [("Output", true), ("Stop", false)] {
        let mut ui = sim(&app);
        ui.click(label).unwrap_or_else(|_| panic!("{label}"));
        let sent: Vec<Message> = ui.into_messages().collect();
        match (sent.as_slice(), want_output) {
            ([Message::Code(CodeMsg::JobOutput(id))], true) | ([Message::Code(CodeMsg::JobKill(id))], false) => assert_eq!(id, "j1"),
            other => panic!("{label}: {other:?}"),
        }
    }
}

#[test]
fn a_new_session_forgets_the_jobs() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "make");
    code(&mut app, CodeMsg::Stop);
    say(&mut app, "/new");
    assert!(app.code().jobs.is_empty(), "`newCodeSession` empties `code.jobs`");
}

#[test]
fn a_sub_agent_may_start_a_job() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    say(&mut app, "go");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("t1".into()), name: Some("task".into()), arguments: r#"{"description":"d","prompt":"p","agent":"general"}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::KidStream("t1".into(), StreamEvent::ToolCall { index: 0, id: Some("k".into()), name: Some("bash".into()), arguments: r#"{"command":"serve","background":true}"#.into() }));
    code(&mut app, CodeMsg::KidStream("t1".into(), StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("job s9 started".into())));
    assert_eq!(app.code().jobs.len(), 1, "`runToolCall` is the page's one place for it");
    assert_eq!(app.code().jobs[0].id, "s9");
}

fn picture(app: &App, name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    let shot = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    for old in std::fs::read_dir(&out).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    assert!(shot.matches_image(out.join(format!("{name}.png"))).unwrap(), "the picture of {name}");
}

/// The block drawn with a running and a finished job, for the eye of the vision check.
#[test]
fn the_background_jobs_are_drawn() {
    let (mut app, _dir) = app();
    open_folder(&mut app, "job_id");
    start_job(&mut app, "c1", "npm run dev -- --port 3000");
    code(&mut app, CodeMsg::Stop);
    say(&mut app, "more");
    code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c2".into()), name: Some("bash".into()), arguments: r#"{"command":"cargo build","background":true}"#.into() }));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c2".into(), Ok("job j2".into())));
    code(&mut app, CodeMsg::JobOutput("j2".into()));
    code(&mut app, CodeMsg::JobOutputDone("j2".into(), Ok("Compiling fllm-app\nFinished dev profile".into())));
    picture(&app, "code-jobs");
}
