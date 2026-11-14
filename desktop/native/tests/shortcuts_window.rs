//! The keyboard shortcuts in the window: the overview lists what every key does with the keys it
//! has now (the `helpModal` of `shell.html`), Change waits for the new keys, a combination in use
//! is refused with the Studio's words, Reset undoes one and *Reset all* undoes everything — and
//! the keys do the work of the Studio in the app.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::store::Store;
use iced::keyboard::{key::Named, Key, Modifiers};
use iced_test::Simulator;

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view())
}

/// Writes the drawing of the current window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR")
            .unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// Ctrl and a letter, as the window sees it.
fn ctrl(key: char) -> (Key, Modifiers) {
    (Key::Character(key.to_string().into()), Modifiers::CTRL)
}

/// The overview lists the shortcuts with their keys, the fixed rows without the row of `?`, and
/// what the Studio says about changing them (`renderShortcutsPanel` of `boot.js`).
#[test]
fn the_overview_lists_the_shortcuts_and_the_fixed_rows() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::ShortcutsOpen);
    assert!(app.shortcuts_open, "the overview is open");

    let mut ui = sim(&app);
    ui.find("Keyboard shortcuts").expect("the title of the helpModal");
    ui.find("Choose Change, then press the new keys. A combination that is already in use is refused.")
        .expect("the sentence of the Studio");
    ui.find("New chat (new session in Code)").expect("the first row says what it does");
    ui.find("Ctrl+N").expect("its keys, as people see them");
    ui.find("Stop the answer or the agent, close a window").expect("the stop row");
    ui.find("Go to Chat").expect("a row of the surfaces");
    ui.find("Send").expect("the fixed row of Enter");
    ui.find("New line").expect("the fixed row of Shift+Enter");
    ui.find("Edit your last message (empty field)").expect("the fixed row of ArrowUp");
    ui.find("?").expect_err("the row of ? is left out");
    ui.find("Close").expect("the overview closes with Close");
    ui.find("Reset all").expect_err("there is nothing to undo yet");
    save_png(&app, "shortcuts-overview");
}

/// Change waits for the new keys; the combination goes in when it may be given and stays out with
/// the reason when it may not (the capture of `boot.js` with `checkCombo`).
#[test]
fn a_shortcut_is_changed_and_a_combination_in_use_is_refused() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::ShortcutsOpen);
    {
        let mut ui = sim(&app);
        ui.find("Press the new keys (Esc cancels)")
            .expect_err("nothing waits yet");
    }

    // Change on the first row: it waits, and the button says Cancel
    let _ = app.update(Message::ScChange("newChat".into()));
    assert_eq!(app.sc_capture.as_deref(), Some("newChat"));
    {
        let mut ui = sim(&app);
        ui.find("Press the new keys (Esc cancels)").expect("the waiting line");
        ui.find("Cancel").expect("the Change button says Cancel");
    }
    save_png(&app, "shortcuts-capture");

    // the new keys: saved, and the row shows them
    let (key, modifiers) = ctrl('j');
    let _ = app.update(Message::Key(key, modifiers));
    assert_eq!(app.sc_capture, None, "the capture ends");
    assert_eq!(app.sc_message, "Saved.", "the panel says Saved.");
    assert_eq!(
        app.shortcut_override("newChat").map(String::as_str),
        Some("Mod+J"),
        "the change is kept"
    );
    {
        let mut ui = sim(&app);
        ui.find("Ctrl+J").expect("the row shows the new keys");
        ui.find("Reset").expect("the row can be undone now");
        ui.find("Reset all").expect("so can everything");
    }

    // a combination another shortcut still has is refused with the Studio's words (Alt+1 is the
    // way to Chat; newChat gave up Mod+N when it moved to Mod+J)
    let _ = app.update(Message::ScChange("toggleSidebar".into()));
    let (key, modifiers) = (Key::Character("1".into()), Modifiers::ALT);
    let _ = app.update(Message::Key(key, modifiers));
    assert_eq!(
        app.sc_message,
        "Alt+1 is already used for: Go to Chat.",
        "checkCombo of boot.js says so"
    );
    assert_eq!(
        app.sc_capture.as_deref(),
        Some("toggleSidebar"),
        "it keeps waiting after a refusal"
    );

    // Escape ends the capture without a change
    let _ = app.update(Message::Key(Key::Named(Named::Escape), Modifiers::empty()));
    assert_eq!(app.sc_capture, None, "Escape cancels");
    assert_eq!(
        app.shortcut_override("toggleSidebar"),
        None,
        "and nothing was written"
    );

    // a bare letter is refused too (typing must not be disturbed)
    let _ = app.update(Message::ScChange("stop".into()));
    let _ = app.update(Message::Key(Key::Character("q".into()), Modifiers::empty()));
    assert_eq!(
        app.sc_message,
        "Use at least one of Ctrl, Alt or Cmd with the key, so typing is not disturbed.",
        "checkCombo of boot.js says so"
    );
    let _ = app.update(Message::ShortcutsClose);
    assert!(!app.shortcuts_open, "Close closes it");
}

/// Reset puts one shortcut back, *Reset all* puts every one back (the buttons of boot.js).
#[test]
fn reset_puts_the_keys_back() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::ShortcutsOpen);
    let _ = app.update(Message::ScChange("newChat".into()));
    let (key, modifiers) = ctrl('j');
    let _ = app.update(Message::Key(key, modifiers));
    assert_eq!(
        app.shortcut_override("newChat").map(String::as_str),
        Some("Mod+J")
    );

    let _ = app.update(Message::ScReset("newChat".into()));
    assert_eq!(app.shortcut_override("newChat"), None, "one is undone");

    let _ = app.update(Message::ScChange("newChat".into()));
    let (key, modifiers) = ctrl('j');
    let _ = app.update(Message::Key(key, modifiers));
    let _ = app.update(Message::ScResetAll);
    assert_eq!(app.shortcut_override("newChat"), None, "everything is undone");
    assert_eq!(
        app.sc_message,
        "All shortcuts are back to their defaults.",
        "the panel says so"
    );
    let _ = app.update(Message::ShortcutsClose);
}

/// The keys do the work of the Studio: the surfaces open, the overview opens on its own keys, and
/// a plain letter does nothing (the keydown of `boot.js`: only combinations count).
#[test]
fn the_keys_do_the_work_of_the_studio() {
    let (mut app, _dir) = app_with(Store::default());
    app.store_mut().chats.push(fllm_app::store::ChatRecord::new(
        1,
        "chat one",
        Vec::new(),
    ));
    app.store_mut().chats.push(fllm_app::store::ChatRecord::new(
        2,
        "chat two",
        Vec::new(),
    ));

    // Alt+3 opens Models (the combo of boot.js, read by the window)
    let (key, modifiers) = (Key::Character("3".into()), Modifiers::ALT);
    let _ = app.update(Message::Key(key, modifiers));
    assert_eq!(app.page(), View::Models, "Alt+3 opens Models");

    let _ = app.update(Message::Key(Key::Character("2".into()), Modifiers::ALT));
    assert_eq!(app.page(), View::Code, "Alt+2 goes to Code");
    let _ = app.update(Message::Key(Key::Character("1".into()), Modifiers::ALT));
    assert_eq!(app.page(), View::Chat, "Alt+1 goes to Chat");

    // Mod+/ opens the overview itself
    let _ = app.update(Message::Key(Key::Character("/".into()), Modifiers::CTRL));
    assert!(app.shortcuts_open, "Mod+/ opens the overview");
    let _ = app.update(Message::ShortcutsClose);

    // a plain letter is for the text field, it runs nothing
    let _ = app.update(Message::Key(Key::Character("q".into()), Modifiers::empty()));
    assert!(!app.shortcuts_open, "a plain key runs no shortcut");
    assert_eq!(app.page(), View::Chat, "and nothing else happened");

    // the keys of a changed shortcut are the ones in force: change Mod+N to Mod+J
    let _ = app.update(Message::ShortcutsOpen);
    let _ = app.update(Message::ScChange("newChat".into()));
    let _ = app.update(Message::Key(Key::Character("j".into()), Modifiers::CTRL));
    let _ = app.update(Message::ShortcutsClose);
    let before = app.store().prefs.shortcut_overrides.clone();
    let _ = app.update(Message::Key(Key::Character("n".into()), Modifiers::CTRL));
    assert_eq!(
        app.store().prefs.shortcut_overrides,
        before,
        "Mod+N is no longer the new chat after the change"
    );
    let _ = app.update(Message::ShortcutsOpen);
    {
        let mut ui = sim(&app);
        ui.find("Ctrl+J").expect("the changed keys are in force");
        save_png(&app, "shortcuts-changed");
    }
}

/// Settings, General, opens the overview (the `btnShowShortcuts` of `shell.html`).
#[test]
fn the_settings_page_opens_the_overview() {
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("See and change the keys.").expect("the description of the Studio");
        ui.find("Keyboard shortcuts").expect("the button");
    }
    let _ = app.update(Message::ShortcutsOpen);
    assert!(app.shortcuts_open, "the overview opens");
    assert_eq!(app.page(), View::Settings, "the page stays where it was");
}
