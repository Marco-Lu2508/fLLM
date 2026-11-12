//! The answer settings: automatic until switched to manual, remembered on the disk, and what each one puts into the request.

use fllm_app::api::{chat_body, ModelInfo};
use fllm_app::app::{App, Message, View};
use fllm_app::store::{Store, BUILT_IN_SYSTEM_PROMPT};
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo { id: "tiny".into(), max_context: 4096 })));
    (app, dir)
}

fn reload(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn paste(text: &str) -> Message {
    Message::SystemEdit(Action::Edit(Edit::Paste(Arc::new(text.to_string()))))
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

#[test]
fn everything_is_automatic_at_the_start_and_sends_nothing_extra() {
    let (app, _dir) = app("light");
    assert_eq!(app.chat_params(), Default::default());
    assert_eq!(app.system_prompt(), BUILT_IN_SYSTEM_PROMPT);
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert!(body.get("temperature").is_none() && body.get("top_p").is_none() && body.get("max_tokens").is_none(), "{body}");
    assert_eq!(body["messages"][0]["role"], "system");
}

#[test]
fn manual_sampling_is_sent_remembered_and_switched_off_again() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::Temperature(0.3));
    let _ = app.update(Message::TopP(0.8));
    let _ = app.update(Message::SavePrefs);
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!((body["temperature"].as_f64().map(|t| (t * 100.0).round()), body["top_p"].as_f64().map(|t| (t * 100.0).round())), (Some(30.0), Some(80.0)));
    let saved = reload(&dir).prefs;
    assert!(saved.manual_sampling && (saved.temperature - 0.3).abs() < 1e-6 && (saved.top_p - 0.8).abs() < 1e-6);
    // back to automatic: the chosen values stay in the file for the next time, but are not sent
    let _ = app.update(Message::SamplingManual(false));
    assert_eq!(app.chat_params(), Default::default());
    assert!((reload(&dir).prefs.temperature - 0.3).abs() < 1e-6);
}

#[test]
fn a_custom_system_prompt_replaces_the_built_in_one_and_an_empty_one_falls_back() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SystemCustom(true));
    assert_eq!(app.system_prompt(), BUILT_IN_SYSTEM_PROMPT, "custom but empty: the built-in prompt");
    let _ = app.update(paste("Du bist ein knapper Assistent. Antworte immer auf Deutsch."));
    assert_eq!(app.system_prompt(), "Du bist ein knapper Assistent. Antworte immer auf Deutsch.");
    assert_eq!(app.wire_messages()[0].content, "Du bist ein knapper Assistent. Antworte immer auf Deutsch.");
    assert_eq!(reload(&dir).prefs.system_prompt, "Du bist ein knapper Assistent. Antworte immer auf Deutsch.", "written after the edit");
    let _ = app.update(Message::SystemCustom(false));
    assert_eq!(app.system_prompt(), BUILT_IN_SYSTEM_PROMPT, "switched back: the text is kept but not used");
    assert!(!reload(&dir).prefs.system_prompt.is_empty());
}

#[test]
fn the_longest_answer_takes_digits_only_and_empty_means_automatic() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::MaxTokensText("abc 12x3".into()));
    assert_eq!(app.chat_params().max_tokens, Some(123));
    let _ = app.update(Message::SavePrefs);
    assert_eq!(reload(&dir).prefs.max_tokens, 123);
    let _ = app.update(Message::MaxTokensText(String::new()));
    assert_eq!(app.chat_params().max_tokens, None);
    let _ = app.update(Message::MaxTokensText("99999999999".into()));
    assert!(app.chat_params().max_tokens.unwrap() <= 999_999, "a very long number is cut");
}

#[test]
fn back_to_automatic_resets_every_answer_setting() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::SystemCustom(true));
    let _ = app.update(paste("x"));
    let _ = app.update(Message::MaxTokensText("50".into()));
    let _ = app.update(Message::ResetPrefs);
    assert_eq!(app.prefs(), &Default::default());
    assert_eq!(reload(&dir).prefs, Default::default());
}

#[test]
fn settings_from_an_older_file_without_them_load_with_the_automatic_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    std::fs::write(&path, r#"{"chats":[],"next_id":3,"theme":"dark"}"#).unwrap();
    let store = Store::load(&path);
    assert_eq!(store.prefs, Default::default());
    assert_eq!((store.next_id, store.theme.as_str()), (3, "dark"));
    // a file that has only some of them keeps those and defaults the rest
    std::fs::write(&path, r#"{"chats":[],"prefs":{"manual_sampling":true,"temperature":1.2}}"#).unwrap();
    let p = Store::load(&path).prefs;
    assert!(p.manual_sampling && (p.temperature - 1.2).abs() < 1e-6 && (p.top_p - 0.95).abs() < 1e-6 && p.max_tokens == 0);
}

#[test]
fn the_settings_page_draws_in_both_modes_and_its_buttons_send_their_messages() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app(theme);
        let _ = app.update(Message::Go(View::Settings));
        let _ = app.update(Message::SamplingManual(true));
        let _ = app.update(Message::SystemCustom(true));
        let _ = app.update(paste("Be brief."));
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        let shot = ui.snapshot(&app.theme()).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("settings-{theme}");
        for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
            let _ = std::fs::remove_file(old.path());
        }
        shot.matches_image(dir.join(format!("{name}.png"))).unwrap();
    }
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
    ui.click("Manual").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::SamplingManual(true))));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
    ui.click("Custom").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::SystemCustom(true))));
}
