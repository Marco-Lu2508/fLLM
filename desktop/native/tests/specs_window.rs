//! Row 80: the specs of `.fllm/specs/` in the window — what `tests/commands_fixtures.rs` does
//! not pin (it pins the requests `/spec` plans): the notices that answer them, the run a prompt
//! of the server starts, the *Specs* block of the side and its checkboxes, and the reload the
//! page does when a run is over.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{Change, CodeMsg, HookFire, Item, Mode, Permission, Tool, Workspace};
use fllm_app::hooks::{Hook, HooksAnswer};
use fllm_app::specs::{Spec, SpecTask};
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
        CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: Some("main".into()) })),
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
            Item::Note(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

fn users(app: &App) -> Vec<(String, Option<String>)> {
    app.code()
        .items
        .iter()
        .filter_map(|i| match i {
            Item::User { text, sent } => Some((text.clone(), sent.clone())),
            _ => None,
        })
        .collect()
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 1600.0), app.view())
}

fn task(index: u64, text: &str, done: bool, depth: u64) -> SpecTask {
    SpecTask { index, text: text.into(), done, depth }
}

/// The spec of the fixtures: three tasks, the first done, the last one nested.
fn export() -> Spec {
    Spec {
        name: "export".into(),
        requirements: true,
        design: true,
        tasks_file: true,
        total: 3,
        done: 1,
        tasks: vec![
            task(1, "Write the exporter", true, 0),
            task(2, "Wire the button", false, 0),
            task(3, "Add a test", false, 1),
        ],
    }
}

fn finished() -> Spec {
    Spec {
        name: "search".into(),
        requirements: true,
        design: false,
        tasks_file: true,
        total: 1,
        done: 1,
        tasks: vec![task(1, "Index the pages", true, 0)],
    }
}

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

/// The listing `/spec` writes once the list is read again: the pitch of the page for a folder
/// without a spec, the counts for one with specs — and the list of the side is the fresh one.
#[test]
fn the_listing_is_written_from_the_list_that_was_just_read() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/spec");
    assert!(notes(&app).is_empty(), "the listing waits for the list the server sends");
    assert!(!app.code().running, "a listing is no run");

    code(&mut app, CodeMsg::SpecListDone(Vec::new()));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("No specs yet. /spec NAME what you want writes requirements.md, design.md and tasks.md under .fllm/specs/NAME and stops after each file for your review."),
        "the empty folder"
    );

    code(&mut app, CodeMsg::SpecListDone(vec![export(), finished()]));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Specs of this folder:\nexport: 1/3 tasks done\nsearch: 1/1 tasks done\n/spec run NAME [N] does the next open task (or task N); /spec NAME what you want starts a new spec."),
        "the specs with their counts"
    );
    assert_eq!(app.code().specs.len(), 2, "the side holds the list that was read");
}

/// `/spec NAME` without words says how it goes, before any request, and starts nothing.
#[test]
fn a_spec_without_words_says_the_usage() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/spec csv-export");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Usage: /spec NAME what you want. Example: /spec csv-export Export the report as CSV."),
        "the usage line of the page"
    );
    assert!(!app.code().running);
    say(&mut app, "/spec csv-export    ");
    assert_eq!(notes(&app).len(), 2, "spaces after the name are no words");
}

/// What the page says when a spec route fails: in the list where it notices, as a toast where
/// it toasts (the checkbox of the side) — and the toast reads the list again.
#[test]
fn a_refusal_is_a_note_or_a_toast_as_the_page_shows_it() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    code(&mut app, CodeMsg::SpecFailed { message: "There is no spec \"nope\". /spec lists them.".into(), toast: false });
    assert_eq!(notes(&app).last().map(String::as_str), Some("There is no spec \"nope\". /spec lists them."));
    assert!(app.toast().is_none(), "a note is no toast");

    code(&mut app, CodeMsg::SpecFailed { message: "The spec action failed.".into(), toast: true });
    assert_eq!(app.toast(), Some("The spec action failed."), "the checkbox toasts its refusal");
    assert_eq!(notes(&app).len(), 1, "and writes nothing into the list");
}

/// The prompt of the server starts a run with the page's display, as every `agentRun` does:
/// the model gets the prompt, the list shows `/spec run export 2`.
#[test]
fn a_prompt_of_the_server_starts_a_run_with_its_display() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    code(&mut app, CodeMsg::SpecPrompt { prompt: "Do task 2 of export.".into(), display: "/spec run export 2".into() });
    assert!(app.code().running, "the run started");
    assert_eq!(
        users(&app),
        vec![("/spec run export 2".to_string(), Some("Do task 2 of export.".to_string()))],
        "the words of the page, the prompt behind them"
    );
}

/// A run of the page carries the submit hook whatever started it (`fireHook` sits in
/// `agentRunNow`, not in `submitText`): a spec's run and `/init` wait for it, with its context
/// in a block of its own after the prompt.
#[test]
fn every_run_of_a_command_fires_the_submit_hook() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["user_prompt_submit"]))));

    code(&mut app, CodeMsg::SpecPrompt { prompt: "Do task 2 of export.".into(), display: "/spec run export 2".into() });
    assert!(!app.code().running, "the run waits for the hook");
    code(
        &mut app,
        CodeMsg::SubmitHookDone {
            prompt: "Do task 2 of export.".into(),
            display: "/spec run export 2".into(),
            answer: HookFire { blocked: false, reason: String::new(), context: "mind the tests".into() },
        },
    );
    assert!(app.code().running);
    assert_eq!(
        users(&app).last().and_then(|(_, sent)| sent.clone()).as_deref(),
        Some("Do task 2 of export.\n\n[hooks]\nmind the tests"),
        "the prompt and the block of the hook"
    );
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(!app.code().running);

    // `/init` is an `agentRun` too
    say(&mut app, "/init");
    assert!(!app.code().running, "`/init` waits for the hook as well");
}

/// `/batch` runs its task through `agentRun` once the new folder's tools are read, so the submit
/// hook is asked there too.
#[test]
fn a_batch_task_waits_for_the_submit_hook() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["user_prompt_submit"]))));
    say(&mut app, "/batch feat do the thing");
    code(
        &mut app,
        CodeMsg::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: Some("do the thing".into()),
            result: Ok(json!({ "path": "/home/marco/p/.fllm/worktrees/feat", "branch": "feat" })),
        },
    );
    code(
        &mut app,
        CodeMsg::Opened(Ok(Workspace { root: "/home/marco/p/.fllm/worktrees/feat".into(), name: "feat".into(), branch: Some("feat".into()) })),
    );
    code(&mut app, CodeMsg::ToolsLoaded(Ok((Vec::new(), json!([])))));
    assert!(!app.code().running, "the task waits for the hook like every run");
}

/// The self-review closes a spec's run like any other: the page's `agentRunNow` runs it after
/// the loop whenever the switch is on and files changed.
#[test]
fn a_spec_run_is_reviewed_like_any_run() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ToggleAutoReview(true));
    code(&mut app, CodeMsg::SetMode(Mode::AutoEdit));
    code(&mut app, CodeMsg::ChangesLoaded(Ok(vec![Change { id: 5, path: "old.rs".into(), kind: "edit".into() }])));

    code(&mut app, CodeMsg::SpecPrompt { prompt: "Do task 2 of export.".into(), display: "/spec run export 2".into() });
    assert!(app.code().running);
    code(
        &mut app,
        CodeMsg::Stream(StreamEvent::ToolCall {
            index: 0,
            id: Some("c1".into()),
            name: Some("edit_file".into()),
            arguments: r#"{"path":"src/lib.rs","old_string":"1","new_string":"2"}"#.into(),
        }),
    );
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    code(&mut app, CodeMsg::ToolDone("c1".into(), Ok("Edited.".into())));
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(app.code().running, "the run stays open while the review works");
    assert!(app.code().review.is_some(), "the review of the spec's run started");
}

/// A prompt that comes while a run is going does not touch that run.
#[test]
fn a_prompt_while_a_run_goes_leaves_the_run_alone() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "first");
    assert!(app.code().running);
    let before = users(&app);
    code(&mut app, CodeMsg::SpecPrompt { prompt: "Do task 2.".into(), display: "/spec run export 2".into() });
    assert_eq!(users(&app), before, "no second message");

    // with a hook on the list the second run must not even ask it
    let _ = app.update(Message::HooksLoaded(Ok(hooks_for(&["user_prompt_submit"]))));
    let task = app.update(Message::Code(CodeMsg::SpecPrompt { prompt: "Do task 2.".into(), display: "/spec run export 2".into() }));
    assert_eq!(users(&app), before, "still no second message");
    assert_eq!(task.units(), 0, "and no hook is fired for a run that cannot start");
}

/// The end of a run reads the list again (`loadSpecs()` in the `finally` of `agentRunNow`),
/// for every run, because the agent may have written the files.
#[test]
fn the_end_of_a_run_reads_the_specs_again() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "write the spec files");
    assert!(app.code().running);
    let task = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
    assert!(!app.code().running, "the run is over");
    assert!(task.units() >= 1, "the list is requested again when the run ends: {} units", task.units());

    code(&mut app, CodeMsg::SpecsLoaded(vec![export()]));
    assert_eq!(app.code().specs, vec![export()], "the answer replaces the list");
}

/// The checkbox of a task: a tick asks the server, the answer replaces the whole list, and a
/// refusal toasts and reads the list again.
#[test]
fn a_tick_asks_the_server_and_the_answer_replaces_the_list() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SpecsLoaded(vec![export()]));

    let task = app.update(Message::Code(CodeMsg::SpecTick { name: "export".into(), index: 2, done: true }));
    assert!(task.units() >= 1, "the tick is a request");
    assert_eq!(app.code().specs, vec![export()], "the list changes with the answer, not before it");

    let mut ticked = export();
    ticked.tasks[1].done = true;
    ticked.done = 2;
    code(&mut app, CodeMsg::SpecReplaced(vec![ticked.clone()]));
    assert_eq!(app.code().specs, vec![ticked], "what the server sends is the list");

    let task = app.update(Message::Code(CodeMsg::SpecFailed { message: "nope".into(), toast: true }));
    assert_eq!(app.toast(), Some("nope"));
    assert!(task.units() >= 1, "the failed tick reads the list again");
}

/// The block of the side: the pitch for an empty folder; the box of a spec with its counts,
/// its three files, its tasks and the button of the next one.
#[test]
fn the_side_shows_the_pitch_or_the_specs() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    {
        let mut ui = sim(&app);
        assert!(ui.find("Specs").is_ok(), "the block is there");
        assert!(ui.find("0").is_err(), "no count beside the heading while there is no spec");
        assert!(
            ui.find("Plan a feature before coding it: /spec name what you want writes requirements, design and a task list under .fllm/specs/ and builds it task by task.").is_ok(),
            "an empty folder reads the pitch"
        );
    }

    code(&mut app, CodeMsg::SpecsLoaded(vec![export(), finished()]));
    {
        let mut ui = sim(&app);
        assert!(
            ui.find("Plan a feature before coding it: /spec name what you want writes requirements, design and a task list under .fllm/specs/ and builds it task by task.").is_err(),
            "the pitch is gone"
        );
    }
    // a spec without a task has nothing to run and nothing to claim done
    code(&mut app, CodeMsg::SpecsLoaded(vec![Spec { name: "fresh".into(), ..Default::default() }]));
    {
        let mut ui = sim(&app);
        assert!(ui.find("fresh").is_ok(), "its box is there");
        assert!(ui.find("Every task is done.").is_err(), "a spec without tasks is not done");
        assert!(ui.find("Run next task").is_err(), "and has no next task");
    }
    code(&mut app, CodeMsg::SpecsLoaded(vec![export(), finished()]));
    let mut ui = sim(&app);
    for label in ["export", "search", "1/3", "1/1", "Req", "Design", "Tasks", "Write the exporter", "Wire the button", "Add a test", "Index the pages", "Run next task", "Every task is done.", "2"] {
        assert!(ui.find(label).is_ok(), "the side draws {label:?}");
    }
}

/// The checkboxes and the button send what the page sends: the task's own index, the new state
/// of the box; the button is for the first open task and waits while a run goes.
#[test]
fn the_controls_of_the_side_send_what_the_page_sends() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SpecsLoaded(vec![export(), finished()]));

    let sent = {
        let mut ui = sim(&app);
        ui.click("Run next task").expect("the button");
        ui.into_messages().collect::<Vec<_>>()
    };
    assert!(
        matches!(sent.as_slice(), [Message::Code(CodeMsg::SpecRunTask { name, index: 2 })] if name == "export"),
        "the first open task of its spec: {sent:?}"
    );

    let sent = {
        let mut ui = sim(&app);
        ui.click("Wire the button").expect("the checkbox of task two");
        ui.into_messages().collect::<Vec<_>>()
    };
    assert!(
        matches!(sent.as_slice(), [Message::Code(CodeMsg::SpecTick { name, index: 2, done: true })] if name == "export"),
        "ticking an open task: {sent:?}"
    );
    let sent = {
        let mut ui = sim(&app);
        ui.click("Write the exporter").expect("the checkbox of task one");
        ui.into_messages().collect::<Vec<_>>()
    };
    assert!(
        matches!(sent.as_slice(), [Message::Code(CodeMsg::SpecTick { name, index: 1, done: false })] if name == "export"),
        "unticking a done task: {sent:?}"
    );

    // a run is going: the button has nothing to start
    say(&mut app, "something else");
    assert!(app.code().running);
    let sent = {
        let mut ui = sim(&app);
        let _ = ui.click("Run next task");
        ui.into_messages().collect::<Vec<_>>()
    };
    assert!(sent.is_empty(), "no run beside a run: {sent:?}");
}

/// `/spec run NAME N` plans its own request through the dispatcher and starts nothing by
/// itself: the run begins with the prompt the server answers.
#[test]
fn the_commands_wait_for_the_server_before_a_run_starts() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SpecsLoaded(vec![export()]));
    for text in ["/spec run export", "/spec run export 3", "/spec run nope", "/spec csv-export Export the report as CSV."] {
        say(&mut app, text);
        assert!(!app.code().running, "{text} starts no run before the server answers");
    }
    assert!(users(&app).is_empty(), "no message of a run yet");
}

/// The steering files of the row belong to the server: the `always` ones come with the
/// instructions as files of their own, the others as one `steering index` file, and the model
/// reads them like AGENTS.md, in the order the server lists them (the page has no steering pane
/// either, only the line that names the files it read).
#[test]
fn the_steering_files_reach_the_model_with_the_instructions() {
    let answer = json!({
        "files": [
            { "file": "AGENTS.md", "text": "Run cargo test." },
            { "file": "steering/style.md", "text": "Always use tabs." },
            { "file": "steering index", "text": "steering/api.md (when src/api/* is touched): API rules" }
        ],
        "memory": { "workspace": [], "global": [] },
        "profile": ""
    });
    let context = fllm_app::code::parse_context(&answer);
    assert_eq!(context.files.len(), 3, "every file the server lists is kept");
    let prompt = context.prompt();
    assert!(prompt.contains("(steering/style.md)\nAlways use tabs."), "an always-steering file is an instruction: {prompt}");
    assert!(prompt.contains("(steering index)\nsteering/api.md (when src/api/* is touched): API rules"), "the index too: {prompt}");
    assert!(
        prompt.find("(AGENTS.md)").unwrap() < prompt.find("(steering/style.md)").unwrap(),
        "the order of the server stands"
    );

    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::ContextLoaded(Ok(context)));
    let mut ui = sim(&app);
    assert!(
        ui.find("Project instructions read: AGENTS.md, steering/style.md, steering index").is_ok(),
        "the line of the page names all of them"
    );
}

fn picture(app: &App, name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    let shot = ui.snapshot(&app.theme()).unwrap();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
    std::fs::create_dir_all(&dir).unwrap();
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or(dir);
    std::fs::create_dir_all(&out).unwrap();
    for old in std::fs::read_dir(&out).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    assert!(shot.matches_image(out.join(format!("{name}.png"))).unwrap(), "the picture of {name}");
}

/// The side with specs, drawn: for the eye of the vision check (the PNGs are artifacts, not a
/// pixel gate).
#[test]
fn the_side_with_specs_is_drawn() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    picture(&app, "specs-none");
    code(&mut app, CodeMsg::SpecsLoaded(vec![export(), finished()]));
    picture(&app, "code-specs");
}

/// Opening a folder reads its specs along with the rest (`loadSpecs` runs with the other loads
/// of `openWorkspace`): one more request in the batch than before this row.
#[test]
fn opening_a_folder_reads_the_specs_too() {
    let (mut app, _dir) = app("light");
    let task = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: Some("main".into()) }))));
    assert_eq!(task.units(), 5, "the loads of an opened folder, the specs among them");
}
