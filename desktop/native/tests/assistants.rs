//! Assistants in the window: choosing one, editing it, importing it, and what it changes in a request.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::assistants::{self, Assistant};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced_test::Simulator;

fn rec(role: Role, content: &str) -> Record {
    Record { role, content: content.into(), reasoning: String::new(), stats: None, passages: None , images: Vec::new(), files: Vec::new(), tool_calls: None, tool_call_id: None, tool_cards: Vec::new(), sources: Vec::new(), research: None }
}

fn chat(id: u64) -> ChatRecord {
    ChatRecord::new(id, "A chat", vec![rec(Role::User, "hi"), rec(Role::Assistant, "hello")])
}

/// A window over a store on disk, with one chat that is open (like the first start of the app).
fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn fresh() -> (App, tempfile::TempDir) {
    app_with(Store { chats: vec![chat(0)], next_id: 1, theme: "light".into(), ..Default::default() })
}

fn fresh_with(assistants: Vec<Assistant>) -> (App, tempfile::TempDir) {
    app_with(Store { chats: vec![chat(0)], next_id: 1, theme: "light".into(), assistants, ..Default::default() })
}

fn apply(app: &mut App, id: &str) {
    let _ = app.update(Message::AsstApply(id.to_string()));
}

fn reload(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn saved_assistants(dir: &tempfile::TempDir) -> Vec<Assistant> {
    reload(dir).assistants
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

/// Writes the drawing of the current window under `ui-snapshots/<prefix>-<renderer>.png`.
fn snap(app: &App, prefix: &str) {
    let mut ui = sim(app);
    let shot = ui.snapshot(&app.theme()).unwrap();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
    std::fs::create_dir_all(&dir).unwrap();
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{prefix}.png"))).unwrap();
}

fn custom(name: &str) -> Assistant {
    Assistant::sanitize(
        &serde_json::json!({ "id": "as-custom", "name": name, "system": "INSTR-MARKER of the helper.",
            "temperature": 0.4, "top_p": 0.7, "top_k": 5, "starters": ["Custom starter one"] }),
        assistants::new_id,
    )
    .unwrap()
}

fn variant(name: &str, id: &str, mode: &str, sys_text: &str) -> Assistant {
    Assistant::sanitize(
        &serde_json::json!({ "id": id, "name": name, "sys_mode": mode, "sys_text": sys_text }),
        assistants::new_id,
    )
    .unwrap()
}

#[test]
fn choosing_an_assistant_sets_the_chat_the_sampler_and_the_line_under_the_model() {
    let (mut app, dir) = fresh();
    assert!(app.active_assistant().is_none(), "nothing is chosen at first");
    assert!(!app.prefs().manual_sampling, "the answers start automatic");
    apply(&mut app, "precise");
    let a = app.active_assistant().unwrap();
    assert_eq!(a.id, "precise");
    assert_eq!(app.active_assistant_id().as_deref(), Some("precise"));
    // the open chat remembers it, and so does the file
    let store = reload(&dir);
    assert_eq!(store.chats[0].assistant.as_deref(), Some("precise"));
    assert_eq!(store.prefs.assistant, "precise", "the choice is kept for new chats");
    // the sampling values of the assistant are sent from now on
    assert!(app.prefs().manual_sampling, "an assistant sets the sampler to manual");
    assert_eq!(app.prefs().temperature, 0.2);
    assert_eq!(app.prefs().top_p, 0.9);
    assert_eq!(app.prefs().sampling.top_k, 20);
    assert_eq!(app.toast(), Some("Assistant \"Precise and brief\" selected"));
    // a chat that starts later carries the choice of the new chats with it
    let _ = app.update(Message::NewChat);
    let _ = app.update(Message::Starter(0));
    let store = reload(&dir);
    assert_eq!(store.chats.len(), 2, "the starter wrote a new chat");
    let fresh = store.chats.iter().find(|c| c.title != "A chat").expect("the new chat");
    assert_eq!(fresh.assistant.as_deref(), Some("precise"), "a new chat keeps the assistant of the new chats");
}

#[test]
fn the_instructions_of_the_assistant_are_added_to_the_system_prompt() {
    let (mut app, _dir) = fresh_with(vec![custom("Helper")]);
    let plain = app.system_prompt();
    assert!(!plain.contains("INSTR-MARKER"), "no assistant is chosen yet");
    apply(&mut app, "as-custom");
    let with = app.system_prompt();
    assert!(with.ends_with("INSTR-MARKER of the helper."), "the instructions come last: {with}");
    assert_ne!(with, "INSTR-MARKER of the helper.", "the built-in prompt stays before them");
}

#[test]
fn a_sys_mode_that_is_not_inherit_replaces_or_adds_to_the_default_prompt() {
    let (mut app, _dir) = fresh_with(vec![
        variant("Override one", "as-ovr", "override", "OVERRIDE-MARKER only."),
        variant("Append one", "as-app", "append", "APPEND-MARKER."),
        variant("Inherit one", "as-inh", "inherit", "IGNORED-MARKER."),
    ]);
    apply(&mut app, "as-ovr");
    assert_eq!(app.system_prompt(), "OVERRIDE-MARKER only.", "override replaces the whole default prompt");
    apply(&mut app, "as-app");
    let appended = app.system_prompt();
    assert!(appended.ends_with("APPEND-MARKER."), "append adds behind the default prompt");
    assert!(appended.starts_with("You are a helpful"), "the default prompt is still first: {appended}");
    apply(&mut app, "as-inh");
    let plain = app.system_prompt();
    assert!(!plain.contains("APPEND-MARKER"), "inherit takes the choice of the settings again");
    assert!(!plain.contains("IGNORED-MARKER"), "the sys_text of an inheriting assistant is not used");
    assert!(plain.starts_with("You are a helpful"));
}

#[test]
fn the_assistant_of_a_chat_comes_back_after_a_restart() {
    let (mut app, dir) = fresh_with(vec![custom("Helper")]);
    apply(&mut app, "as-custom");
    assert_eq!(app.prefs().temperature, 0.4);
    let saved = reload(&dir);
    assert_eq!(saved.assistants.len(), 1);
    assert_eq!(saved.assistants[0].name, "Helper");
    assert_eq!(saved.chats[0].assistant.as_deref(), Some("as-custom"));
    assert_eq!(saved.prefs.assistant, "as-custom");
    // a second start over the same file shows the same assistant and the same instructions
    let (app, _dir2) = app_with(saved);
    assert_eq!(app.active_assistant_id().as_deref(), Some("as-custom"));
    assert!(app.system_prompt().ends_with("INSTR-MARKER of the helper."));
}

#[test]
fn a_new_assistant_is_saved_edited_and_deleted() {
    let (mut app, dir) = fresh();
    let _ = app.update(Message::AsstNew);
    sim(&app).find("New assistant").expect("the editor window is open");
    let _ = app.update(Message::AsstName("  My   helper ".into()));
    let _ = app.update(Message::AsstTemp("0.3".into()));
    let _ = app.update(Message::AsstSave);
    let list = saved_assistants(&dir);
    assert_eq!(list.len(), 1, "the new assistant is saved");
    assert_eq!(list[0].name, "My helper");
    assert_eq!(list[0].temperature, 0.3);
    assert!(list[0].id.starts_with("as-"));
    assert_eq!(app.toast(), Some("Assistant \"My helper\" saved"));
    assert!(app.active_assistant_id().is_some(), "a new assistant becomes the chosen one");
    sim(&app).find("New assistant").expect_err("the editor is closed again");
    // editing keeps the id and replaces the entry
    let id = list[0].id.clone();
    let _ = app.update(Message::AsstEditOpen(id.clone()));
    sim(&app).find("Edit assistant").expect("the editor for the existing one");
    let _ = app.update(Message::AsstName("Renamed".into()));
    let _ = app.update(Message::AsstSave);
    let list = saved_assistants(&dir);
    assert_eq!(list.len(), 1, "the edit replaces the entry");
    assert_eq!(list[0].id, id);
    assert_eq!(list[0].name, "Renamed");
    // deleting takes it away and nothing points at it any more
    let _ = app.update(Message::AsstEditOpen(id));
    let _ = app.update(Message::AsstDelete);
    assert!(saved_assistants(&dir).is_empty());
    assert!(app.active_assistant_id().is_none());
    assert_eq!(app.prefs().assistant, "");
}

#[test]
fn a_builtin_is_edited_as_a_copy_and_never_changed() {
    let (mut app, dir) = fresh();
    let _ = app.update(Message::AsstEditOpen("programmer".into()));
    let _ = app.update(Message::AsstName("Programmer, changed".into()));
    let _ = app.update(Message::AsstSave);
    let list = saved_assistants(&dir);
    assert_eq!(list.len(), 1, "a copy of a built-in is a new assistant");
    assert!(list[0].id.starts_with("as-"), "the copy has its own id");
    assert_eq!(assistants::builtins()[3].name, "Programmer", "the built-in itself is untouched");
    // a built-in has no id in the editor, so Delete does nothing to it
    let _ = app.update(Message::AsstEditOpen("translator".into()));
    let _ = app.update(Message::AsstDelete);
    assert_eq!(saved_assistants(&dir).len(), 1, "the copy is still there");
}

#[test]
fn importing_an_exported_assistant_adds_it_and_junk_says_why() {
    let (mut app, dir) = fresh();
    let file = assistants::export_json(&custom("Imported"));
    let _ = app.update(Message::AsstImported(Ok(file)));
    let list = saved_assistants(&dir);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "Imported");
    assert!(list[0].id.starts_with("as-"), "an import always gets a new id");
    assert_eq!(app.toast(), Some("Assistant \"Imported\" imported"));
    let _ = app.update(Message::AsstImported(Ok("nope".into())));
    assert_eq!(app.toast(), Some("The file is not valid JSON."));
    let _ = app.update(Message::AsstImported(Ok("{\"format\":\"other\"}".into())));
    assert_eq!(app.toast(), Some("This file is not an exported assistant."));
    // a closed dialog says nothing (the empty error is the cancel of the picker)
    let _ = app.update(Message::DismissToast);
    let _ = app.update(Message::AsstImported(Err(String::new())));
    assert!(app.toast().is_none(), "closing the dialog shows no line");
    let _ = app.update(Message::AsstImported(Err("no file dialog here".into())));
    assert_eq!(app.toast(), Some("no file dialog here"));
}

#[test]
fn the_menu_lists_every_assistant_and_the_welcome_screen_takes_theirs() {
    let (mut app, _dir) = fresh_with(vec![custom("Helper")]);
    let _ = app.update(Message::AsstMenuToggle);
    {
        let mut ui = sim(&app);
        ui.find("Assistants").expect("the menu head");
        ui.find("Standard").expect("the first built-in");
        ui.find("Save current").expect("the actions of the head");
        ui.find("T 0.2").expect("the temperature of Precise and brief");
        ui.find("No instructions").expect("Standard has no instructions");
        ui.find("Helper").expect("the own assistant below the built-ins");
    }
    let _ = app.update(Message::AsstMenuClose);
    // the welcome screen shows the four suggestions of the chosen assistant
    apply(&mut app, "creative");
    let _ = app.update(Message::NewChat);
    {
        let mut ui = sim(&app);
        ui.find("Write the opening paragraph of a mystery set on a night train.").expect("the starter of Creative");
        ui.find("Edit suggestions").expect_err("a built-in has nothing to edit");
    }
    apply(&mut app, "as-custom");
    {
        let mut ui = sim(&app);
        ui.find("Custom starter one").expect("the starter of the own assistant");
        ui.find("Edit suggestions").expect("an own assistant can change its suggestions");
    }
    // the suggestions drive the messages of the welcome screen
    assert_eq!(app.starter_list()[0], "Custom starter one");
}

#[test]
fn the_editor_window_and_the_menu_draw() {
    let (mut app, _dir) = fresh();
    let _ = app.update(Message::AsstMenuToggle);
    snap(&app, "assistants-menu");
    let _ = app.update(Message::AsstMenuClose);
    let _ = app.update(Message::AsstEditOpen("translator".into()));
    snap(&app, "assistants-editor");
    let _ = app.update(Message::AsstClose);
    snap(&app, "assistants-topbar");
}

#[test]
fn an_old_file_without_the_fields_of_assistants_still_loads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    std::fs::write(
        &path,
        r#"{"prefs":{"manual_sampling":false,"temperature":0.8},"chats":[{"id":0,"title":"old","messages":[]}],"next_id":1,"theme":"dark"}"#,
    )
    .unwrap();
    let store = Store::load(&path);
    assert_eq!(store.prefs.temperature, 0.8, "the file is read");
    assert!(store.assistants.is_empty(), "no assistants in an old file");
    assert_eq!(store.prefs.assistant, "", "no choice in an old file");
    assert_eq!(store.chats[0].assistant, None, "no assistant on an old chat");
    assert_eq!(store.theme, "dark");
}
