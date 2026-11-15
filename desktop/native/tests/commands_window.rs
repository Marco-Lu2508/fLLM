//! What the commands of row 79 do once they run: the state they change, the notes they write,
//! the cards and shares they leave behind, and the two switches that go through the runtime of
//! Settings > Runtime. (`tests/commands_fixtures.rs` pins what the page plans for every word;
//! this file drives the plans through the window.)

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{CodeMsg, Permission, Tool, Workspace};
use fllm_app::commands::CustomCommand;
use fllm_app::runtime::{Confirm, Status};
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
            fllm_app::code::Item::Note(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 9000.0), app.view())
}

/// `/stop` stops a running agent, an empty line does too, and everything else waits in the box
/// where the page queues it — this window has no queue, so the words stay until you send them.
#[test]
fn while_the_agent_works_only_stop_is_answered() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "Fix the tokenizer.");
    assert!(app.code().running);

    say(&mut app, "/stop");
    assert!(!app.code().running, "`/stop` stops it, as the page does");

    say(&mut app, "Fix it.");
    assert!(app.code().running);
    say(&mut app, "one more thing");
    assert_eq!(
        app.code().editor.text().trim(),
        "one more thing",
        "the words wait in the box instead of being queued"
    );
    assert!(
        !app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::User { text, .. } if text == "one more thing")),
        "and they did not become a message"
    );
}

/// `/clear` saves the conversation of the folder first — the file is written again after it was
/// taken away — and then resets everything the page's `resetSessionState` clears.
#[test]
fn clear_saves_the_session_and_starts_over() {
    let (mut app, dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "Fix the tokenizer.");
    code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    assert!(!app.code().running);
    let file = dir.path().join("code-sessions.json");
    assert!(file.exists(), "the run already saved it");
    std::fs::remove_file(&file).unwrap();

    say(&mut app, "/clear");
    assert!(app.code().items.is_empty(), "the conversation is gone");
    assert!(app.code().todos.is_empty(), "and the task list with it");
    let saved = std::fs::read_to_string(&file).expect("/clear wrote the session before it reset");
    assert!(saved.contains("Fix the tokenizer."), "{saved}");
}

/// `/status` leaves a card of the command itself; when it is done its output is shared with the
/// next message and never enters the conversation (`pending` + `takePending` of the page).
#[test]
fn status_runs_a_card_and_shares_its_output_with_the_next_message() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/status");

    let id = app
        .code()
        .items
        .iter()
        .find_map(|i| match i {
            fllm_app::code::Item::Call(c) => Some(c.id.clone()),
            _ => None,
        })
        .expect("the command left a card");
    assert!(app.code().pending.is_empty(), "the output is not in yet");

    code(&mut app, CodeMsg::UserRunDone(id, Ok("On branch main".into()), true));
    assert_eq!(
        app.code().pending,
        ["I ran this command myself:\n$ git status -sb\n```\nOn branch main\n```".to_string()],
        "the words the page puts in front of the next message"
    );
    // a normal message carries them to the model without showing them
    say(&mut app, "anything new?");
    let wire = app.code().wire(None);
    let last = wire.last().expect("the question");
    assert!(
        last.content.starts_with("I ran this command myself:"),
        "the share rides in front of the question: {:?}",
        last.content
    );
    assert!(
        app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::User { text, .. } if text == "anything new?")),
        "the view still shows only what was typed"
    );
}

/// `/allow` writes the rule into the state the decisions read, into the store under this folder
/// (the page keys its own storage the same way), and says so.
#[test]
fn allow_writes_the_rule_for_this_folder_and_says_so() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/allow bash: npm test*");

    assert_eq!(app.code().rules.allow, "bash: npm test*");
    assert_eq!(notes(&app).last().map(String::as_str), Some("Rule saved: allow bash: npm test*"));
    assert_eq!(
        app.store().prefs.code_rules.get(FOLDER).map(|r| r.allow.as_str()),
        Some("bash: npm test*"),
        "one block per folder, as the page keeps it"
    );

    say(&mut app, "/rules");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Allow:\nbash: npm test*\n\nDeny:\n(none)"),
        "the words of `/rules`"
    );

    // and the folder it belongs to is read again when it is opened
    code(&mut app, CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: None })));
    assert_eq!(app.code().rules.allow, "bash: npm test*", "the rules come back with the folder");
}

/// `/remember` notes both sentences of the page and reads the instructions again so the fact
/// joins the next prompt; a refused save goes to the toast with the words of the server.
#[test]
fn remember_saves_notes_and_toasts_a_refusal() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/remember the tests need llvm");
    code(
        &mut app,
        CodeMsg::MemorySaved(Ok(()), "Remembered (for this workspace): the tests need llvm".into()),
    );
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Remembered (for this workspace): the tests need llvm")
    );

    say(&mut app, "/remember global ships on fridays");
    code(&mut app, CodeMsg::MemorySaved(Err("The memory could not be saved.".into()), String::new()));
    assert!(
        !notes(&app).iter().any(|n| n.contains("Remembered (for every workspace)")),
        "nothing was remembered"
    );
    let mut ui = sim(&app);
    ui.find("The memory could not be saved.").expect("the toast of the refusal");
}

/// `/turbo on` asks the very question the switches of Settings > Runtime ask, and only the
/// answer of the runtime becomes the note of the page; cancelling asks nothing.
#[test]
fn turbo_asks_the_runtime_and_notes_its_answer() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/turbo on");
    assert_eq!(app.runtime_confirm, Some(Confirm::Turbo), "the confirm is up at once");
    let mut ui = sim(&app);
    ui.find("Turbo removes every permission question and the sandbox. Continue?")
        .expect("the question of the dialog");
    drop(ui);

    // a cancelled question is nothing, and a later answer changes no note
    let _ = app.update(Message::RuntimeConfirmCancel);
    assert_eq!(app.runtime_confirm, None);
    let before = notes(&app).len();
    let _ = app.update(Message::RuntimeSaved(Ok(serde_json::from_value::<Status>(json!({ "turbo": true })).unwrap())));
    assert_eq!(notes(&app).len(), before, "nothing was asked, nothing is noted");

    // the answer to the question is the note
    say(&mut app, "/turbo on");
    let _ = app.update(Message::RuntimeConfirmOk);
    let _ = app.update(Message::RuntimeSaved(Ok(serde_json::from_value::<Status>(json!({ "turbo": true })).unwrap())));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Turbo is on: every tool call runs, nothing asks, sandbox off.")
    );

    // off asks no question
    say(&mut app, "/turbo off");
    assert_eq!(app.runtime_confirm, None, "nothing to confirm");
    let _ = app.update(Message::RuntimeSaved(Ok(serde_json::from_value::<Status>(json!({ "turbo": false })).unwrap())));
    assert_eq!(notes(&app).last().map(String::as_str), Some("Turbo is off: the usual permission questions are back."));
}

/// `/sandbox <mode>` posts without a question and notes the mode the runtime reports back,
/// including the line for a machine where the sandbox does not work.
#[test]
fn sandbox_posts_and_notes_what_the_runtime_answers() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/sandbox strict");
    assert_eq!(app.runtime_confirm, None, "the page asks no question for a mode");
    let answered = serde_json::from_value::<Status>(json!({ "sandbox": "strict", "sandbox_available": true })).unwrap();
    let _ = app.update(Message::RuntimeSaved(Ok(answered)));
    assert_eq!(notes(&app).last().map(String::as_str), Some("Sandbox: strict."));

    // a word that is no mode is refused at once, with the state as the server last said it
    say(&mut app, "/sandbox sometimes");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Sandbox is strict. /sandbox off | workspace | strict")
    );
}

/// `/personality` keeps its style with the store and puts it into the system prompt where
/// `codeSystemPrompt` of the page puts it — before the instructions of the folder.
#[test]
fn personality_goes_into_the_prompt_and_the_store() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/personality concise");

    assert_eq!(app.code().personality, "concise");
    assert_eq!(app.store().prefs.code_personality, "concise");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("Style: Concise. Use as few words as the job allows. No preamble and no recap of the request. Report what changed, what you checked and what is left.")
    );
    let wire = app.code().wire(None);
    let system = &wire[0].content;
    let style = system.find("Style: use as few words").expect("the style is in the prompt");
    let instructions = system.find("Instructions of this project").unwrap_or(usize::MAX);
    assert!(style < instructions, "before the instructions, as the page writes them");

    // an unknown style changes nothing and says what the choices are
    say(&mut app, "/personality poet");
    assert_eq!(app.code().personality, "concise");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("There is no style \"poet\". Choose one of: default, concise, teacher, reviewer, pair.")
    );
}

/// `/init` shows the command in the list and sends its prompt to the model — `display` and
/// `userText` of `agentRun`.
#[test]
fn init_shows_what_was_typed_and_sends_the_prompt() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/init");

    let user = app
        .code()
        .items
        .iter()
        .find_map(|i| match i {
            fllm_app::code::Item::User { text, sent } => Some((text.clone(), sent.clone())),
            _ => None,
        })
        .expect("the command became a turn");
    assert_eq!(user.0, "/init", "the view shows the command");
    let sent = user.1.expect("the model gets something else than the view");
    assert!(sent.starts_with("Explore this workspace and write a file named AGENTS.md"), "…the prompt");
    assert!(sent.ends_with("instead of starting over."), "…word for word");
    let wire = app.code().wire(None);
    assert_eq!(wire.last().expect("the turn").content, sent, "the request carries the prompt");
}

/// A custom command is read, expanded and run with what was typed as its display (`expandCommand`
/// and `agentRun(expandCommand(body, argText), text)` of the page).
#[test]
fn a_custom_command_runs_with_its_body_and_shows_what_was_typed() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(
        &mut app,
        CodeMsg::CustomLoaded(vec![CustomCommand {
            name: "deploy".into(),
            description: "Ship it".into(),
            path: Some(".fllm/commands/deploy.md".into()),
            plugin: None,
        }]),
    );
    say(&mut app, "/deploy now");
    code(
        &mut app,
        CodeMsg::CustomBody(Ok("Ship the current branch.\n\n$ARGUMENTS".into()), "now".into(), "/deploy now".into()),
    );

    let user = app
        .code()
        .items
        .iter()
        .find_map(|i| match i {
            fllm_app::code::Item::User { text, sent } => Some((text.clone(), sent.clone())),
            _ => None,
        })
        .expect("the command became a turn");
    assert_eq!(user.0, "/deploy now");
    assert_eq!(
        user.1.as_deref(),
        Some("Ship the current branch.\n\nnow"),
        "`$ARGUMENTS` is the words behind the name"
    );

    // a command whose file cannot be read is a note, as the page's catch writes it
    say(&mut app, "/deploy");
    code(&mut app, CodeMsg::CustomBody(Err("The command file could not be read.".into()), String::new(), "/deploy".into()));
    assert_eq!(notes(&app).last().map(String::as_str), Some("The command file could not be read."));
}

/// `/search` notes the hits as the page writes them, `/insights` counts what this folder did.
#[test]
fn search_and_insights_note_what_they_found() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/search error");
    let hits = vec![fllm_app::commands::SearchHit {
        id: "h1".into(),
        title: Some("The tokenizer bug".into()),
        kind: "code".into(),
        updated: None,
        snippet: "the loop bound was off by one".into(),
    }];
    code(&mut app, CodeMsg::SearchDone(Ok(hits), "error".into()));
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("1 stored session for \"error\":\n- The tokenizer bug (code, undated)\n  the loop bound was off by one")
    );

    say(&mut app, "/insights");
    assert_eq!(
        notes(&app).last().map(String::as_str),
        Some("No agent work saved for this folder yet."),
        "no agent step has finished here"
    );
}

/// The words of `/help` are the table of the page with its tail (`case 'help'`), and `/agent`
/// waits like the page when the agent runs.
#[test]
fn help_lists_the_table_and_agent_waits_while_it_runs() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "/help");
    let help = notes(&app).last().cloned().expect("the notice of /help");
    assert!(help.starts_with("Commands: /clear, /compact, /init, /spec, /resume,"), "…the table");
    assert!(help.contains("/sandbox, /autoapprove, /turbo, /sandbox,"), "sandbox stands twice, as on the page");
    assert!(help.ends_with("Alt+B switches between the build and plan agent."), "…the tail");

    // while it runs, `/agent` waits in the box like every other word (the page queues it); the
    // guard toast of `toggleAgent` belongs to Alt+B, which the command itself never reaches
    say(&mut app, "work on it");
    let mode_before = app.code().mode;
    say(&mut app, "/agent");
    assert_eq!(app.code().editor.text().trim(), "/agent", "it waits in the box");
    assert_eq!(app.code().mode, mode_before, "and the mode did not change");
}
