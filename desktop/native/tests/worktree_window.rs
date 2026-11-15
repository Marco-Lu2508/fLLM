//! Row 82: what `/worktree` and `/batch` do once the server answers (`tests/commands_fixtures.rs`
//! pins the request the steps plan and the usage lines; here is the answer side — the notices,
//! the folder switch and the batch's run).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::code::{CodeMsg, CodeMsg as C, Item, Tool, Workspace};
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
        C::Opened(Ok(Workspace {
            root: FOLDER.into(),
            name: "fllm".into(),
            branch: Some("main".into()),
        })),
    );
    let tools: Vec<Tool> = [("read_file", fllm_app::code::Permission::Read), ("bash", fllm_app::code::Permission::Command)]
        .into_iter()
        .map(|(n, p)| Tool::server(n, p))
        .collect();
    code(app, C::ToolsLoaded(Ok((tools, json!([])))));
}

fn say(app: &mut App, text: &str) {
    code(app, C::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, C::Send);
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

fn last_note(app: &App) -> Option<String> {
    notes(app).last().cloned()
}

/// The usage lines: the dispatch answers them itself, without a request.
#[test]
fn the_usage_lines_are_answered_without_asking_the_server() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/worktree new");
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Usage: /worktree new <name> · /worktree leave · /worktree remove <name> · /worktree (list)"),
        "an action without a name says how it goes"
    );
    say(&mut app, "/worktree remove");
    assert!(
        last_note(&app).is_some_and(|n| n.starts_with("Usage: /worktree new <name>")),
        "remove needs a name too"
    );
    say(&mut app, "/batch");
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Usage: /batch <name> <task> — the task runs in a new worktree on branch <name>."),
        "a batch without a task says how it goes"
    );
    say(&mut app, "/batch feat");
    assert!(
        last_note(&app).is_some_and(|n| n.starts_with("Usage: /batch <name> <task>")),
        "a batch without a task says how it goes"
    );
    assert!(!app.code().running, "none of them started anything");
}

/// The rows of `/worktree` with the page's own spacing: `* (main)` for the folder you are in,
/// two spaces between the parts, an empty branch a space shorter.
#[test]
fn the_list_shows_the_rows_the_page_joins() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    code(
        &mut app,
        C::WorktreeDone {
            action: "list",
            name: None,
            batch: None,
            result: Ok(json!({
                "worktrees": [
                    { "current": true, "branch": "main", "path": "/home/marco/p" },
                    { "current": false, "name": "feat", "branch": "feat", "path": "/home/marco/p/.fllm/worktrees/feat" },
                    { "current": false, "name": "bare", "path": "/somewhere" }
                ]
            })),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("* (main)  main  /home/marco/p\n  feat  feat  /home/marco/p/.fllm/worktrees/feat\n  bare    /somewhere"),
        "the rows exactly as the page joins them"
    );

    code(
        &mut app,
        C::WorktreeDone { action: "list", name: None, batch: None, result: Ok(json!({ "worktrees": [] })) },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("No worktrees. /worktree new <name> creates one."),
        "an empty list says how to make one"
    );
}

/// `/worktree new` switches the folder first and notices after — the page awaits
/// `setWorkspace` — and `/worktree leave` comes back to the main one with its branch.
#[test]
fn a_new_worktree_opens_its_folder_before_the_notice() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    code(
        &mut app,
        C::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: None,
            result: Ok(json!({ "path": "/home/marco/p/.fllm/worktrees/feat", "branch": "feat" })),
        },
    );
    assert!(
        last_note(&app).is_none(),
        "the notice waits for the folder (the page awaits setWorkspace first)"
    );
    assert_eq!(app.code().workspace.as_ref().map(|w| w.root.as_str()), Some(FOLDER), "not open yet");
    // the folder of the worktree is opened (its fetch has no server here)
    code(
        &mut app,
        C::Opened(Ok(Workspace {
            root: "/home/marco/p/.fllm/worktrees/feat".into(),
            name: "feat".into(),
            branch: Some("feat".into()),
        })),
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Worktree feat at /home/marco/p/.fllm/worktrees/feat. The agent works there now; /worktree leave brings you back, /worktree remove feat deletes it."),
        "the notice of the page, with its name in the last place"
    );
    assert_eq!(
        app.code().workspace.as_ref().map(|w| w.root.as_str()),
        Some("/home/marco/p/.fllm/worktrees/feat"),
        "the agent works there now"
    );

    // back to the main worktree, with the branch the answer carries
    code(
        &mut app,
        C::WorktreeDone {
            action: "leave",
            name: None,
            batch: None,
            result: Ok(json!({ "path": "/home/marco/p", "branch": "main" })),
        },
    );
    code(
        &mut app,
        C::Opened(Ok(Workspace { root: "/home/marco/p".into(), name: "fllm".into(), branch: Some("main".into()) })),
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Back in the main worktree (/home/marco/p, branch main)."),
        "the branch stands behind the path when the answer has one"
    );
}

/// `/worktree remove` notices at once (no folder to open), and the failures keep the two
/// families of the page: the message of the server's body, its own fallback, and the caught
/// request.
#[test]
fn a_removed_worktree_notices_at_once_and_the_failures_keep_the_page_texts() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    code(
        &mut app,
        C::WorktreeDone {
            action: "remove",
            name: Some("feat".into()),
            batch: None,
            result: Ok(json!({ "removed": "feat" })),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Removed the worktree feat. Its branch stays until you delete it."),
        "the notice of the page"
    );

    // the server's own message, its fallback without one, and the request that did not arrive
    code(
        &mut app,
        C::WorktreeDone {
            action: "leave",
            name: None,
            batch: None,
            result: Err("server:not a git repository".into()),
        },
    );
    assert_eq!(last_note(&app).as_deref(), Some("not a git repository"), "the message of the body");
    code(
        &mut app,
        C::WorktreeDone {
            action: "list",
            name: None,
            batch: None,
            result: Err("server:".into()),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("The worktree action failed."),
        "the fallback of the page when the body has no message"
    );
    code(
        &mut app,
        C::WorktreeDone {
            action: "list",
            name: None,
            batch: None,
            result: Err("transport:no server at http://127.0.0.1:9".into()),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("The worktree action failed: no server at http://127.0.0.1:9"),
        "the caught request of the page"
    );
    // a batch that the server refused says its own line, without the body's message
    code(
        &mut app,
        C::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: Some("do it".into()),
            result: Err("server:not a git repository".into()),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("The worktree could not be created."),
        "the page reads no message of the server here"
    );
    code(
        &mut app,
        C::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: Some("do it".into()),
            result: Err("transport:no server at http://127.0.0.1:9".into()),
        },
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("The batch could not start: no server at http://127.0.0.1:9"),
        "the caught request of the batch"
    );
}

/// `/batch` end to end: the worktree opens, its notice stands, and the task runs as a run of
/// its own — display `/batch`, not a turn of the conversation.
#[test]
fn a_batch_opens_its_worktree_and_runs_the_task_as_its_own_run() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);

    say(&mut app, "/batch feat do the thing");
    code(
        &mut app,
        C::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: Some("do the thing".into()),
            result: Ok(json!({ "path": "/home/marco/p/.fllm/worktrees/feat", "branch": "feat" })),
        },
    );
    assert!(last_note(&app).is_none(), "still waiting for the folder");
    code(
        &mut app,
        C::Opened(Ok(Workspace {
            root: "/home/marco/p/.fllm/worktrees/feat".into(),
            name: "feat".into(),
            branch: Some("feat".into()),
        })),
    );
    assert_eq!(
        last_note(&app).as_deref(),
        Some("Batch feat: working on branch feat in /home/marco/p/.fllm/worktrees/feat."),
        "the notice of the page"
    );
    assert!(!app.code().running, "the task waits for the tools of the folder");
    // the tools arrive — now the page's `agentRun(task, '/batch')` runs
    code(&mut app, C::ToolsLoaded(Ok((Vec::new(), json!([])))));
    assert!(app.code().running, "the task runs");
    let display = app
        .code()
        .items
        .iter()
        .rev()
        .find_map(|i| match i {
            Item::User { text, .. } => Some(text.clone()),
            _ => None,
        })
        .expect("the task is a message");
    assert_eq!(display, "/batch", "with `/batch` as its display, as the page passes it");
    assert!(
        app.code().review.is_none(),
        "a batch is not a turn of the conversation: no self-review"
    );
}

/// A folder that could not be opened keeps the worktree's notice: it exists either way.
#[test]
fn the_notice_survives_a_folder_that_could_not_be_opened() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(
        &mut app,
        C::WorktreeDone {
            action: "new",
            name: Some("feat".into()),
            batch: None,
            result: Ok(json!({ "path": "/nowhere", "branch": "feat" })),
        },
    );
    code(&mut app, C::Opened(Err("the server did not name the folder".into())));
    assert!(
        last_note(&app).is_some_and(|n| n.starts_with("Worktree feat at /nowhere.")),
        "the worktree exists even when its folder failed"
    );
}
