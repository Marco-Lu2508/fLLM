//! Row 47: *Settings > Desktop app* — the words, the hotkey field that waits for keys, the choices
//! that are kept in `desktop.json`.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::desktop::Desktop;
use fllm_app::desktopcards::*;
use iced::keyboard::{key::Named, Key, Modifiers};
use iced_test::Simulator;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Settings));
    (app, dir)
}

fn file(dir: &tempfile::TempDir) -> Desktop {
    Desktop::load(&dir.path().join("desktop.json"))
}

fn press(app: &mut App, key: Key, mods: Modifiers) -> iced::Task<Message> {
    app.update(Message::Key(key, mods))
}

#[test]
fn the_card_has_the_words_of_the_page() {
    let (app, _dir) = app();
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1300.0, 20000.0), app.view());
    for words in [TITLE, TRAY_TITLE, AUTOSTART_TITLE, AUTOSTART_TEXT, HOTKEY_TITLE, HOTKEY_TEXT, VARIANT_TITLE, VARIANT_TEXT, LOG_TITLE, SERVER_TITLE, SERVER_TEXT, "Open log folder", "Restart server", "Default", "Ctrl+Alt+Space", "Automatic"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    assert!(ui.find(TRAY_TEXT).is_ok());
}

#[test]
fn the_field_waits_for_keys_and_keeps_the_combination() {
    let (mut app, dir) = app();
    assert_eq!(file(&dir).quick_entry, "Ctrl+Alt+Space");
    let _ = app.update(Message::Dk(DkMsg::Record));
    assert!(app.dk.recording);
    // a modifier alone does nothing yet, and the field goes on waiting
    let _ = press(&mut app, Key::Named(Named::Control), Modifiers::CTRL);
    assert!(app.dk.recording);
    // a combination without Ctrl, Alt or Cmd is refused with the page's sentence
    let _ = press(&mut app, Key::Character("k".into()), Modifiers::empty());
    assert_eq!(app.dk.status, "Use at least one of Ctrl, Alt or Cmd.");
    assert!(app.dk.recording);
    let _ = press(&mut app, Key::Character("k".into()), Modifiers::CTRL | Modifiers::SHIFT);
    assert!(!app.dk.recording);
    assert_eq!(app.dk.s.quick_entry, "Ctrl+Shift+K");
    assert_eq!(app.dk.status, "");
    assert_eq!(file(&dir).quick_entry, "Ctrl+Shift+K", "kept in the file");
    // the keys of the field are not the window's shortcuts: nothing else was done with them
    assert!(app.dk.file.ends_with("desktop.json"));
}

#[test]
fn off_and_default_and_tab_work_as_in_the_page() {
    let (mut app, dir) = app();
    let _ = app.update(Message::Dk(DkMsg::Off));
    assert_eq!(file(&dir).quick_entry, "");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1300.0, 20000.0), app.view());
    assert!(ui.find("Off").is_ok());
    drop(ui);
    let _ = app.update(Message::Dk(DkMsg::Default));
    assert_eq!(file(&dir).quick_entry, "Ctrl+Alt+Space");
    // Tab leaves the field
    let _ = app.update(Message::Dk(DkMsg::Record));
    let _ = press(&mut app, Key::Named(Named::Tab), Modifiers::empty());
    assert!(!app.dk.recording);
    assert_eq!(file(&dir).quick_entry, "Ctrl+Alt+Space");
}

#[test]
fn the_tray_choice_is_kept() {
    let (mut app, dir) = app();
    let _ = app.update(Message::Dk(DkMsg::Tray(true)));
    assert!(file(&dir).keep_in_tray);
    let _ = app.update(Message::Dk(DkMsg::Tray(false)));
    assert!(!file(&dir).keep_in_tray);
}

#[test]
fn the_start_at_sign_in_is_asked_of_the_system_and_shows_what_it_says() {
    let (mut app, _dir) = app();
    let task = app.update(Message::Dk(DkMsg::Autostart(true)));
    assert_eq!(task.units(), 1, "the system is asked");
    assert!(app.dk.autostart);
    // the system said no: the sentence is shown and the box is looked at again
    let again = app.update(Message::Dk(DkMsg::AutostartIs(Err("The registry did not take the entry.".into()))));
    assert_eq!(app.dk.status, "The registry did not take the entry.");
    assert_eq!(again.units(), 1);
    let _ = app.update(Message::Dk(DkMsg::AutostartIs(Ok(false))));
    assert!(!app.dk.autostart);
}

#[test]
fn a_hotkey_that_is_none_is_refused_before_it_is_kept() {
    let (mut app, dir) = app();
    app.dk.set_quick_entry("Banana", None);
    assert_eq!(app.dk.status, "\"Banana\" is not a valid key combination.");
    assert_eq!(file(&dir).quick_entry, "Ctrl+Alt+Space", "the old one stays");
}

#[test]
fn the_restart_button_is_the_servers_restart() {
    let (app, _dir) = app();
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1300.0, 20000.0), app.view());
    ui.click("Restart server").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::RestartServer]), "{sent:?}");
}
