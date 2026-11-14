//! The prompt library: the rows in Settings > Personalization and the menu of them under a "/" in
//! the message box (the Studio's plibList and plibUpdateSlash/plibPick of projects.js).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::promptlib::SavedPrompt;
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn saved(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

fn paste(text: &str) -> Action {
    Action::Edit(Edit::Paste(Arc::new(text.to_string())))
}

fn prompt(id: &str, title: &str, text: &str) -> SavedPrompt {
    SavedPrompt { id: id.into(), title: title.into(), text: text.into() }
}

fn library() -> Store {
    let mut store = Store::default();
    store.prefs.saved_prompts = vec![
        prompt("pr1", "Email draft", "Write an email about: "),
        prompt("pr2", "Weekly report", "Summarize my week: "),
    ];
    store
}

#[test]
fn the_settings_draws_the_rows_of_the_library_and_adds_and_deletes_them() {
    // an empty library says what the Studio says
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("Saved prompts").expect("the title of the section");
        ui.find("Type / in the message box to list them, pick one with the arrow keys and Enter, or keep typing to filter by title.")
            .expect("the description of the Studio");
        ui.find("No saved prompts yet.").expect("the empty state");
        ui.find("Add a prompt").expect("the button of the Studio");
    }

    // two rows, each with its box, added through the button
    let (mut app, dir) = app_with(Store::default());
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    let _ = app.update(Message::PromptAdd);
    let _ = app.update(Message::PromptAdd);
    assert_eq!(saved(&dir).prefs.saved_prompts.len(), 2, "two prompts were added");
    assert_eq!(saved(&dir).prefs.saved_prompts[0].title, "Prompt 1", "the title of the Studio");
    assert_eq!(saved(&dir).prefs.saved_prompts[1].title, "Prompt 2");
    {
        let mut ui = sim(&app);
        ui.find("Prompt 1").expect("the title in its field");
        ui.find("Prompt 2").expect("the second row");
    }

    // the rows change and delete
    let id = saved(&dir).prefs.saved_prompts[0].id.clone();
    let _ = app.update(Message::PromptTitle(id.clone(), "  Email for the boss  ".into()));
    assert_eq!(saved(&dir).prefs.saved_prompts[0].title, "  Email for the boss  ", "what you type is kept");
    // a title over the limit is clamped to sixty characters (PRJ_LIMITS.promptTitle)
    let long: String = "t".repeat(90);
    let _ = app.update(Message::PromptTitle(id.clone(), long));
    assert_eq!(saved(&dir).prefs.saved_prompts[0].title.chars().count(), 60, "the limit of the Studio");
    // the text of the row through its box
    let _ = app.update(Message::PromptText(id.clone(), paste("Make it short: ")));
    assert_eq!(saved(&dir).prefs.saved_prompts[0].text, "Make it short: ");
    let huge: String = "x".repeat(9000);
    let _ = app.update(Message::PromptText(id.clone(), paste(&huge)));
    assert_eq!(saved(&dir).prefs.saved_prompts[0].text.chars().count(), 8000, "PRJ_LIMITS.promptText");
    let _ = app.update(Message::PromptDel(id.clone()));
    assert_eq!(saved(&dir).prefs.saved_prompts.len(), 1, "the row is gone");
    {
        let mut ui = sim(&app);
        ui.find("Email for the boss").expect_err("the deleted row is not drawn");
    }
}

#[test]
fn the_library_holds_at_most_a_hundred_prompts() {
    let mut store = Store::default();
    store.prefs.saved_prompts = (0..100).map(|i| prompt(&format!("pr{i}"), &format!("Title {i}"), "")).collect();
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::PromptAdd);
    assert_eq!(saved(&dir).prefs.saved_prompts.len(), 100, "no hundred-and-first");
    assert_eq!(app.toast(), Some("The library holds at most 100 prompts."), "the line of the Studio");
}

#[test]
fn a_slash_in_the_box_lists_the_prompts_and_enter_picks_one() {
    let (mut app, dir) = app_with(library());
    // typing "/" starts the menu: the matching titles with their collapsed first line
    let _ = app.update(Message::Edit(paste("/e")));
    assert!(app.slash_open(), "the menu is open");
    {
        let mut ui = sim(&app);
        ui.find("/Email draft").expect("the title that starts with e");
        ui.find("/Weekly report").expect("the title that merely contains e");
        ui.find("Write an email about:").expect("the preview under the first entry");
    }
    // the second entry is picked with the arrows and Enter
    let _ = app.update(Message::SlashDown);
    let _ = app.update(Message::SlashEnter);
    assert_eq!(app.composer_text(), "Summarize my week: ", "the picked prompt became the text");
    assert!(!app.slash_open(), "the menu closed");

    // and the highlighted one with a click (the box is emptied first: a paste replaces the selection)
    let _ = app.update(Message::Edit(Action::SelectAll));
    let _ = app.update(Message::Edit(paste("/e")));
    let _ = app.update(Message::SlashPick(0));
    assert_eq!(app.composer_text(), "Write an email about: ");

    // the built-in commands keep the library out of their way
    let _ = app.update(Message::Edit(Action::SelectAll));
    let _ = app.update(Message::Edit(paste("/clear")));
    assert!(!app.slash_open(), "/clear is a command of the box, not a search");
    // a newline ends the search (the Studio only looks while it is one line)
    let _ = app.update(Message::Edit(Action::SelectAll));
    let _ = app.update(Message::Edit(paste("/e\nsecond line")));
    assert!(!app.slash_open(), "no menu over two lines");
    // Escape closes it (the key binding of the box sends SlashClose)
    let _ = app.update(Message::Edit(Action::SelectAll));
    let _ = app.update(Message::Edit(paste("/e")));
    assert!(app.slash_open());
    let _ = app.update(Message::SlashClose);
    assert!(!app.slash_open());
    assert!(!saved(&dir).prefs.saved_prompts.is_empty(), "picking never changes the library");
}

/// The three views of the feature as PNGs into the snapshot dir of the run.
#[test]
fn the_prompt_library_views_are_painted() {
    // the rows in the settings
    let mut store = library();
    store.prefs.saved_prompts.push(prompt("pr3", "Bug report", "Write up this bug: what happened, what I did, what I expected."));
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    save_png_h(&app, "plib-settings", 2000.0);

    // the menu over the message box
    let (mut app, _dir) = app_with(library());
    let _ = app.update(Message::Edit(paste("/e")));
    save_png(&app, "plib-menu");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    save_png_h(app, name, 900.0)
}

fn save_png_h(app: &App, name: &str, height: f32) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, height), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}
