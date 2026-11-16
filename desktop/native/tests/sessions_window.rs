//! Row 79: the sessions of a folder in the window — the list in the side, *Fork* and *Revert*
//! under your messages, `/fork`, `/revert`, `/retry`, `/rewind`, `/compact`, the notices with
//! buttons, and the blocks *Tasks*, *Changed files*, *Memory* and *Permissions*.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, Change, ChoiceAction, CodeMsg, Item, SessMsg, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::Value;
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

fn sess(app: &mut App, m: SessMsg) {
    code(app, CodeMsg::Sess(m));
}

fn open_folder(app: &mut App) {
    code(app, CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None })));
    let (tools, defs) = client_tools();
    code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

/// One finished turn: your message and a plain answer.
fn turn(app: &mut App, question: &str, answer: &str) {
    say(app, question);
    code(app, CodeMsg::Stream(StreamEvent::Delta(answer.into())));
    code(app, CodeMsg::Stream(StreamEvent::Done));
    assert!(!app.code().running, "the turn ended");
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 3000.0), app.view())
}

fn users(app: &App) -> Vec<String> {
    app.code().items.iter().filter_map(|i| if let Item::User { text, .. } = i { Some(text.clone()) } else { None }).collect()
}

fn change(id: u64, path: &str) -> Change {
    Change { id, path: path.into(), kind: "edit".into(), time: 0.0 }
}

#[test]
fn a_message_is_marked_and_a_finished_session_is_in_the_list() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    assert!(sim(&app).find("Sessions of this folder are saved automatically.").is_ok());
    turn(&mut app, "first question", "first answer");
    turn(&mut app, "second question", "second answer");
    let s = app.code();
    assert_eq!(s.marks.len(), 2);
    assert_eq!(s.marks.iter().map(|m| m.item).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(s.sessions.len(), 1, "the session was written when the run ended");
    assert_eq!(s.sessions[0].shown_title(), "first question");
    let mut ui = sim(&app);
    assert!(ui.find("Sessions").is_ok());
    assert!(ui.find("first question").is_ok(), "the list shows the first words of the first message");
    assert!(ui.find("Fork").is_ok() && ui.find("Revert").is_ok(), "an idle conversation offers both under a message");
}

#[test]
fn a_fork_continues_from_before_a_message_and_leaves_the_original_alone() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    turn(&mut app, "first question", "first answer");
    turn(&mut app, "second question", "second answer");
    sess(&mut app, SessMsg::TurnFork(2));
    let s = app.code();
    assert_eq!(users(&app), vec!["first question".to_string()], "the second message and its answer are cut off");
    assert_eq!(s.title, "first question (fork)");
    assert!(s.title_locked);
    assert_eq!(s.editor.text().trim_end(), "second question", "its text goes back to the input");
    assert!(s.items.iter().any(|i| matches!(i, Item::Note(n) if n == "Forked from \"first question\". The original session is unchanged.")));
    assert_eq!(s.sessions.len(), 2, "the original and the fork");
    let original = s.sessions.iter().find(|r| r.title != "first question (fork)").unwrap();
    assert_eq!(original.items.len(), 4, "the original still holds both turns");
}

#[test]
fn a_session_of_the_list_comes_back_and_the_commands_offer_it() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    turn(&mut app, "old work", "done");
    let old_id = app.code().session_id.clone();
    code(&mut app, CodeMsg::NewSession);
    assert!(app.code().items.is_empty());
    turn(&mut app, "new work", "ok");
    say(&mut app, "/resume");
    let set = &app.code().choices[0];
    assert_eq!(set.action, ChoiceAction::Resume);
    assert_eq!(set.text, "Pick a session to continue:");
    assert_eq!(set.rows.len(), 1, "the session that is open is not offered");
    assert!(set.rows[0].0.starts_with("old work  ("), "{}", set.rows[0].0);
    assert_eq!(set.rows[0].1, old_id);
    sess(&mut app, SessMsg::Choice(0, 0));
    assert_eq!(users(&app), vec!["old work".to_string()]);
    assert!(app.code().choices.is_empty(), "a press takes the notice away");
    // the one that was open is in the list and can be resumed again
    assert_eq!(app.code().sessions.len(), 2);
}

#[test]
fn a_session_is_renamed_exported_and_deleted() {
    let (mut app, dir) = app();
    open_folder(&mut app);
    turn(&mut app, "name me", "ok");
    let id = app.code().session_id.clone();
    sess(&mut app, SessMsg::RenameStart(id.clone()));
    sess(&mut app, SessMsg::RenameEdit("\"A better title\"".into()));
    sess(&mut app, SessMsg::RenameOk);
    assert_eq!(app.code().title, "A better title", "the page's cleanTitle took the quotes off");
    assert!(app.code().title_locked);
    assert_eq!(app.code().sessions[0].title, "A better title");
    // `/rename words` does it without the box
    say(&mut app, "/rename Third name");
    assert_eq!(app.code().title, "Third name");
    // the export is the page's Markdown (only this test names a folder for it)
    std::env::set_var("FLLM_EXPORT_DIR", dir.path());
    sess(&mut app, SessMsg::ExportOf(id.clone()));
    let file = dir.path().join("Third name.md");
    let md = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    assert!(md.starts_with("# Third name\n\nWorkspace: `/w`\n\nExported "), "{md}");
    assert!(md.contains("## User\n\nname me\n\n## Assistant\n\nok\n"), "{md}");
    // a delete asks first
    sess(&mut app, SessMsg::DeleteAsk(id.clone()));
    assert!(sim(&app).find("Delete this session?").is_ok());
    sess(&mut app, SessMsg::DeleteCancel);
    assert_eq!(app.code().sessions.len(), 1);
    sess(&mut app, SessMsg::DeleteAsk(id));
    sess(&mut app, SessMsg::DeleteOk);
    assert!(app.code().sessions.is_empty());
    assert!(app.code().items.is_empty(), "the session that was open is gone with its conversation");
}

#[test]
fn a_revert_takes_the_conversation_and_the_files_back_and_can_be_undone() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    code(&mut app, CodeMsg::ChangesLoaded(Ok(vec![change(4, "a.rs")])));
    turn(&mut app, "first question", "first answer");
    turn(&mut app, "second question", "second answer");
    // the second turn changed a file: the server's list has a newer change than the mark of that message
    code(&mut app, CodeMsg::ChangesLoaded(Ok(vec![change(4, "a.rs"), change(9, "b.rs")])));
    say(&mut app, "/revert");
    let set = app.code().choices[0].clone();
    assert_eq!(set.action, ChoiceAction::Revert);
    assert_eq!(set.rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), vec!["second question", "first question"], "newest first");
    // the server's list is read, the first change after the mark is found, and the files are asked to go back
    sess(&mut app, SessMsg::Choice(0, 0));
    sess(&mut app, SessMsg::RevertList(2, Ok(vec![change(4, "a.rs"), change(9, "b.rs")])));
    // the server put one file back
    let answer: Value = serde_json::json!({ "ok": true, "output": "restored", "restored": ["b.rs"] });
    sess(&mut app, SessMsg::RevertUndone(2, Ok(answer)));
    let s = app.code();
    assert_eq!(users(&app), vec!["first question".to_string()]);
    assert_eq!(s.editor.text().trim_end(), "second question");
    assert_eq!(s.reverted.as_ref().map(|r| r.restored), Some(1));
    assert_eq!(s.choices.len(), 1);
    assert_eq!(s.choices[0].text, "Reverted to before that message and put 1 file change back. Its text is in the input.");
    // undo the revert: the conversation is whole again
    sess(&mut app, SessMsg::Choice(0, 0));
    sess(&mut app, SessMsg::UnrevertDone(Ok(serde_json::json!({ "ok": true }))));
    assert_eq!(users(&app), vec!["first question".to_string(), "second question".to_string()]);
    assert!(app.code().reverted.is_none());
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "The revert was undone.")));
    assert!(app.code().editor.text().trim().is_empty());
}

#[test]
fn a_revert_that_the_server_refuses_stops_with_its_words() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    turn(&mut app, "only question", "answer");
    sess(&mut app, SessMsg::RevertUndone(0, Ok(serde_json::json!({ "ok": false, "error": { "message": "b.rs changed since" } }))));
    assert_eq!(users(&app), vec!["only question".to_string()], "nothing was cut");
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Revert stopped: b.rs changed since")));
    // no message to go back to
    let (mut app, _dir) = self::app();
    open_folder(&mut app);
    say(&mut app, "/revert");
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "There is no message to revert to.")));
    say(&mut app, "/retry");
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "There is no message to retry.")));
}

#[test]
fn rewind_lists_the_changes_newest_first() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    sess(&mut app, SessMsg::RewindList(Err("no".into())));
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Rewind is not available on this server. /undo reverts the last file change.")));
    sess(&mut app, SessMsg::RewindList(Ok(vec![])));
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "No file changes to rewind.")));
    let many: Vec<Change> = (1..=20).map(|i| change(i, &format!("f{i}.rs"))).collect();
    sess(&mut app, SessMsg::RewindList(Ok(many)));
    let set = &app.code().choices[0];
    assert_eq!(set.action, ChoiceAction::Rewind);
    assert_eq!(set.rows.len(), 15, "at most fifteen");
    assert_eq!(set.rows[0], ("edit f20.rs".to_string(), "20".to_string()));
    sess(&mut app, SessMsg::RewindDone(Ok(serde_json::json!({ "ok": true, "output": "3 files" }))));
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Rewound: 3 files")));
    // the block of the side offers the same list once the server answered
    assert!(sim(&app).find("Rewind ...").is_ok());
}

#[test]
fn compact_puts_a_summary_in_front_of_the_recent_turns() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    for i in 0..5 {
        turn(&mut app, &format!("question {i}"), &format!("answer {i}"));
    }
    let before = app.code().wire(None).len();
    assert_eq!(before, 11, "the system prompt and ten messages");
    say(&mut app, "/compact focus on tests");
    assert!(app.code().compacting, "the summary is asked for");
    let note = app.code().items.len() - 1;
    // ten messages: the part before the last six is summarised
    sess(&mut app, SessMsg::Compacted { start: 4, before: 900, resume: false, note, result: Ok("The user asked five questions.".into()) });
    let s = app.code();
    assert!(!s.compacting && !s.running);
    let wire = s.wire(None);
    assert_eq!(wire.len(), 1 + 2 + 6, "the system prompt, the summary with its answer, the six recent messages");
    assert!(wire[1].content.starts_with("[Summary of the earlier conversation, written by you so that work can continue from it]\n\nThe user asked five questions."));
    assert_eq!(wire[2].content, "Understood. I will continue from this summary.");
    assert_eq!(wire[3].content, "question 2");
    assert!(matches!(&s.items[note], Item::Note(n) if n.starts_with("Context compacted: 4 earlier messages became a summary (about 900 to ")));
    assert_eq!(users(&app).len(), 5, "everything stays on the screen");
    // a message that was summarised can no longer be reverted to, the ones after it can
    assert!(!s.turn_allowed(0) && !s.turn_allowed(2));
    assert!(s.turn_allowed(4) && s.turn_allowed(6) && s.turn_allowed(8));
    // the next message goes on from the summary
    say(&mut app, "question 5");
    let wire = app.code().wire(None);
    assert!(wire[1].content.starts_with("[Summary of the earlier conversation"));
    assert_eq!(wire.last().map(|m| m.content.as_str()), Some("question 5"));
    assert_eq!(wire.len(), 1 + 2 + 6 + 1);
}

#[test]
fn a_summary_that_failed_leaves_the_conversation_as_it_was() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    for i in 0..5 {
        turn(&mut app, &format!("q{i}"), &format!("a{i}"));
    }
    say(&mut app, "/compact");
    let note = app.code().items.len() - 1;
    sess(&mut app, SessMsg::Compacted { start: 4, before: 10, resume: false, note, result: Err("down".into()) });
    assert!(app.code().compaction.is_none());
    assert!(matches!(&app.code().items[note], Item::Note(n) if n == "Compacting failed; older steps are trimmed instead."));
    say(&mut app, "/compact");
    // too little conversation says so
    let (mut few, _d) = self::app();
    open_folder(&mut few);
    turn(&mut few, "q", "a");
    say(&mut few, "/compact");
    assert!(few.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "There is not enough conversation to compact yet.")));
}

#[test]
fn the_side_has_tasks_memory_permissions_and_the_style() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let mut ui = sim(&app);
    assert!(ui.find("Tasks").is_ok());
    assert!(ui.find("For bigger jobs the agent keeps a task list here.").is_ok());
    assert!(ui.find("Changed files").is_ok() && ui.find("Nothing changed yet.").is_ok());
    assert!(ui.find("Memory").is_ok() && ui.find("Permissions").is_ok());
    assert!(ui.find("Style of the agent").is_ok());
    drop(ui);
    // the memory opens with `/memory` and holds the facts, one per line
    say(&mut app, "/memory");
    assert!(app.code().mem_open);
    assert!(sim(&app).find("Save memory").is_ok());
    sess(&mut app, SessMsg::PermsToggle);
    let mut ui = sim(&app);
    assert!(ui.find("Always allow").is_ok() && ui.find("Always deny").is_ok() && ui.find("Save rules").is_ok());
    drop(ui);
    // rules are written and counted as the page counts them
    sess(&mut app, SessMsg::RulesEditAllow(Action::Edit(Edit::Paste(Arc::new("bash: npm test*\n# a comment\nedit: src/**".to_string())))));
    sess(&mut app, SessMsg::RulesEditDeny(Action::Edit(Edit::Paste(Arc::new("bash: rm *".to_string())))));
    sess(&mut app, SessMsg::RulesSave);
    assert_eq!(app.code().rules.allow, "bash: npm test*\n# a comment\nedit: src/**");
    assert_eq!(app.code().rules.deny, "bash: rm *");
    // the style is a choice of the five
    sess(&mut app, SessMsg::Personality("Concise".into()));
    assert!(["concise", "default"].contains(&app.code().personality.as_str()));
}

#[test]
fn the_title_comes_from_the_model_unless_it_was_chosen() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    turn(&mut app, "please fix the parser", "ok");
    let id = app.code().session_id.clone();
    sess(&mut app, SessMsg::TitleMade(id.clone(), Ok("Title: \"Fix the parser\"".into())));
    assert_eq!(app.code().title, "Fix the parser");
    assert_eq!(app.code().sessions[0].title, "Fix the parser");
    // a locked title is not replaced
    say(&mut app, "/rename Mine");
    sess(&mut app, SessMsg::TitleMade(id, Ok("Other".into())));
    assert_eq!(app.code().title, "Mine");
}

#[test]
fn the_models_of_the_computer_are_offered_by_the_model_command() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    say(&mut app, "/model");
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "No model is loaded.")), "an empty list says so");
}

#[test]
fn the_page_is_drawn_with_the_blocks_of_the_side() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    code(&mut app, CodeMsg::ChangesLoaded(Ok(vec![change(4, "src/a.rs"), change(9, "src/b.rs")])));
    turn(&mut app, "first question about the parser", "first answer");
    turn(&mut app, "second question", "second answer");
    say(&mut app, "/rewind");
    sess(&mut app, SessMsg::RewindList(Ok(vec![change(4, "src/a.rs"), change(9, "src/b.rs")])));
    sess(&mut app, SessMsg::MemToggle);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1400.0, 1900.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    let _ = std::fs::remove_file(out.join("code-sessions-tiny-skia.png"));
    assert!(image.matches_image(out.join("code-sessions.png")).unwrap());
}
