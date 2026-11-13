//! Headless checks of the window: the real `view` is drawn by iced's software renderer (no display needed), driven with clicks and keys,
//! and compared with what the state should show. The PNGs go to `target/ui-snapshots/` (set `FLLM_SNAPSHOT_DIR` to move them).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, Server};
use fllm_app::store::{ChatRecord, Record, Role, Stats, Store};
use iced_test::Simulator;
use std::path::PathBuf;

const MARKDOWN_ANSWER: &str = "**GGUF** is a single-file format for model weights: a header with metadata, then the tensors, ready to be memory-mapped.\n\n```python\nwith open(\"model.gguf\", \"rb\") as f:\n    magic = f.read(4)\nprint(magic)  # b'GGUF'\n```\n\n1. Pick a size that fits your memory.\n2. Prefer `Q4_K_M` first.\n3. Move up only if answers get worse.\n\n> Rule of thumb: the file size plus the KV cache has to fit in memory.";

fn snapshot_dir() -> PathBuf {
    std::env::var_os("FLLM_SNAPSHOT_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"))
}

fn model() -> ModelInfo {
    ModelInfo::new("qwen2.5-0.5b-instruct", 8192)
}

fn rec(role: Role, content: &str, stats: Option<Stats>) -> Record {
    Record { role, content: content.into(), reasoning: String::new(), stats, passages: None , images: Vec::new(), files: Vec::new() }
}

/// An app on a temporary data folder, with the given chats and theme, connected to a model.
fn app_with(chats: Vec<ChatRecord>, theme: &str, connected: bool) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let next_id = chats.iter().map(|c| c.id + 1).max().unwrap_or(0);
    Store { chats, next_id, theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    if connected {
        let _ = app.update(Message::Connected(Ok(model())));
    }
    (app, dir)
}

fn sample_chats() -> Vec<ChatRecord> {
    vec![
        ChatRecord {
            id: 2,
            title: "What is a GGUF file?".into(),
            messages: vec![
                rec(Role::User, "What is a GGUF file? Give a code sample and a short list.", None),
                rec(Role::Assistant, MARKDOWN_ANSWER, Some(Stats { tokens_per_second: 41.8, completion_tokens: 164, prompt_per_second: 612.0 })),
            ],
            ..Default::default()
        },
        ChatRecord { id: 1, title: "Hi was geht?".into(), messages: vec![rec(Role::User, "Hi was geht?", None), rec(Role::Assistant, "Hallo! Was ist los? Wie kann ich dir helfen?", None)], ..Default::default() },
        ChatRecord { id: 0, title: "Explain in simple words how a language…".into(), messages: vec![], ..Default::default() },
    ]
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn save_png(app: &App, name: &str) -> PathBuf {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 800.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = snapshot_dir();
    std::fs::create_dir_all(&dir).unwrap();
    // iced_test names the file `<name>-<renderer>.png`; remove the old one first, because it only writes when there is none
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    let path = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).find(|p| p.file_name().unwrap().to_string_lossy().starts_with(&prefix)).expect("the picture exists");
    path
}

/// The share of pixels that are not the most common color of the picture (the page): a blank render is 0, a window with content is well above.
fn painted_share(path: &PathBuf) -> f64 {
    let img = image::open(path).unwrap().to_rgba8();
    let mut counts = std::collections::HashMap::<[u8; 4], usize>::new();
    for p in img.pixels() {
        *counts.entry(p.0).or_default() += 1;
    }
    let most = counts.values().copied().max().unwrap_or(0);
    1.0 - most as f64 / (img.width() * img.height()) as f64
}

#[test]
fn the_conversation_draws_in_both_themes() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app_with(sample_chats(), theme, true);
        let _ = app.update(Message::OpenChat(2));
        let path = save_png(&app, &format!("conversation-{theme}"));
        let share = painted_share(&path);
        assert!(share > 0.02, "{theme}: the window is not blank ({share:.3})");
    }
}

#[test]
fn the_start_screen_and_the_offline_screen_draw() {
    let (app, _d1) = app_with(vec![], "light", true);
    assert!(painted_share(&save_png(&app, "start-light")) > 0.02);
    let (app, _d2) = app_with(vec![], "dark", true);
    assert!(painted_share(&save_png(&app, "start-dark")) > 0.02);
    let (mut app, _d3) = app_with(vec![], "light", false);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:8080: connection refused".into())));
    assert!(matches!(app.server_state(), Server::Down(_)));
    assert!(painted_share(&save_png(&app, "offline-light")) > 0.01);
}

#[test]
fn clicking_new_chat_empties_the_view_and_selecting_a_chat_brings_it_back() {
    let (mut app, _dir) = app_with(sample_chats(), "light", true);
    assert_eq!(app.open_chat(), Some(2), "the newest chat is open at the start");
    let _ = app.update(Message::NewChat);
    assert_eq!(app.open_chat(), None);
    assert_eq!(app.message_count(), 0);
    let _ = app.update(Message::OpenChat(1));
    assert_eq!(app.open_chat(), Some(1));
    assert_eq!(app.message_count(), 2);
}

#[test]
fn the_composer_sends_only_with_text_and_a_server() {
    let (mut app, _dir) = app_with(vec![], "light", false);
    let _ = app.update(Message::Send);
    assert_eq!(app.message_count(), 0, "no server, nothing sent");
    let _ = app.update(Message::Connected(Ok(model())));
    let _ = app.update(Message::Send);
    assert_eq!(app.message_count(), 0, "no text, nothing sent");
}

#[test]
fn deleting_a_chat_removes_it_from_the_disk_too() {
    let (mut app, dir) = app_with(sample_chats(), "light", true);
    let _ = app.update(Message::DeleteChat(1));
    let saved = Store::load(&dir.path().join("chats.json"));
    assert_eq!(saved.chats.iter().map(|c| c.id).collect::<Vec<_>>(), vec![2, 0]);
}

#[test]
fn the_theme_toggle_is_remembered() {
    let (mut app, dir) = app_with(vec![], "light", true);
    let _ = app.update(Message::ToggleTheme);
    assert_eq!(Store::load(&dir.path().join("chats.json")).theme, "dark");
    let _ = app.update(Message::ToggleTheme);
    assert_eq!(Store::load(&dir.path().join("chats.json")).theme, "light");
}

#[test]
fn a_reply_streams_into_the_chat_and_is_saved() {
    use fllm_app::api::StreamEvent;
    use iced::widget::text_editor::{Action, Edit};
    let (mut app, dir) = app_with(vec![], "light", true);
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(std::sync::Arc::new("Hi was geht?".to_string())))));
    let _ = app.update(Message::Send);
    assert!(app.is_streaming());
    assert_eq!(app.message_count(), 2, "the question and an empty answer");
    let id = app.open_chat().expect("a chat was started");
    let _ = app.update(Message::Send);
    assert_eq!(app.message_count(), 2, "nothing is sent while a reply is being written");
    for ev in [
        StreamEvent::Delta("Hallo! ".into()),
        StreamEvent::Delta("Was ist los?".into()),
        StreamEvent::Usage { prompt: 33, completion: 9 },
        StreamEvent::Timings { tokens_per_second: 5.9, prompt_per_second: 66.0, prompt_tokens: 33 },
        StreamEvent::Done,
    ] {
        let _ = app.update(Message::Stream(id, ev));
    }
    assert!(!app.is_streaming());
    assert_eq!(app.last_text(), Some("Hallo! Was ist los?"));
    let saved = Store::load(&dir.path().join("chats.json"));
    let chat = saved.chat(id).unwrap();
    assert_eq!(chat.title, "Hi was geht?");
    assert_eq!(chat.messages.len(), 2);
    assert_eq!(chat.messages[1].content, "Hallo! Was ist los?");
    let stats = chat.messages[1].stats.as_ref().unwrap();
    assert_eq!((stats.completion_tokens, stats.tokens_per_second), (9, 5.9));
}

#[test]
fn an_error_before_the_first_word_leaves_no_empty_answer_and_frees_the_composer() {
    use fllm_app::api::StreamEvent;
    use iced::widget::text_editor::{Action, Edit};
    let (mut app, _dir) = app_with(vec![], "light", true);
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(std::sync::Arc::new("Hi".to_string())))));
    let _ = app.update(Message::Send);
    let id = app.open_chat().unwrap();
    let _ = app.update(Message::Stream(id, StreamEvent::Error("the server could not be reached".into())));
    assert!(!app.is_streaming());
    assert_eq!(app.message_count(), 1, "only the question stays");
    let _ = app.update(Message::Stream(id, StreamEvent::Delta("late".into())));
    assert_eq!(app.message_count(), 1, "events of a finished reply are ignored");
}

#[test]
fn stop_keeps_what_was_written_so_far() {
    use fllm_app::api::StreamEvent;
    use iced::widget::text_editor::{Action, Edit};
    let (mut app, dir) = app_with(vec![], "light", true);
    let _ = app.update(Message::Edit(Action::Edit(Edit::Paste(std::sync::Arc::new("Tell me a story".to_string())))));
    let _ = app.update(Message::Send);
    let id = app.open_chat().unwrap();
    let _ = app.update(Message::Stream(id, StreamEvent::Delta("Once upon".into())));
    let _ = app.update(Message::Stop);
    assert!(!app.is_streaming());
    assert_eq!(Store::load(&dir.path().join("chats.json")).chat(id).unwrap().messages[1].content, "Once upon");
    let _ = app.update(Message::Stream(id, StreamEvent::Delta(" a time".into())));
    assert_eq!(app.last_text(), Some("Once upon"), "a stopped reply does not grow");
}

#[test]
fn a_long_chat_draws_only_the_newest_messages_until_more_are_asked_for() {
    use fllm_app::app::CHAT_WINDOW;
    let messages: Vec<Record> = (0..200).map(|i| rec(if i % 2 == 0 { Role::User } else { Role::Assistant }, &format!("message number {i:03}"), None)).collect();
    let (mut app, _dir) = app_with(vec![ChatRecord::new(0, "long", messages)], "light", true);
    let _ = app.update(Message::OpenChat(0));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 12000.0), app.view());
    ui.find("Show 60 earlier messages").expect("older messages are one click away");
    assert!(ui.find("message number 198").is_ok());
    assert!(ui.find("message number 140").is_ok(), "the window starts at the 61st message from the end");
    assert!(ui.find("message number 138").is_err());
    assert!(ui.find("message number 100").is_err(), "an older message is not built");
    ui.click("Show 60 earlier messages").unwrap();
    let msgs: Vec<Message> = ui.into_messages().collect();
    assert!(msgs.iter().any(|m| matches!(m, Message::ShowEarlier)));
    // asking again twice reaches the start; the button is gone when nothing is left
    for _ in 0..3 {
        let _ = app.update(Message::ShowEarlier);
    }
    assert_eq!(CHAT_WINDOW, 60);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 12000.0), app.view());
    assert!(ui.find("message number 000").is_ok());
    assert!(ui.find("Show 60 earlier messages").is_err() && ui.find("Show 1 earlier message").is_err());
}

#[test]
fn a_new_answer_and_a_changed_question_keep_the_old_ones_as_versions_you_can_switch_between() {
    use fllm_app::api::StreamEvent;
    let chats = vec![ChatRecord::new(0, "t", vec![rec(Role::User, "first question", None), rec(Role::Assistant, "first answer", None)])];
    let (mut app, dir) = app_with(chats, "light", true);
    let _ = app.update(Message::OpenChat(0));
    let finish = |app: &mut App, text: &str| {
        let id = app.open_chat().unwrap();
        let _ = app.update(Message::Stream(id, StreamEvent::Delta(text.into())));
        let _ = app.update(Message::Stream(id, StreamEvent::Done));
    };
    let texts = |app: &App| (0..app.message_count()).map(|i| app.message_text(i).unwrap().to_string()).collect::<Vec<_>>();

    // Retry writes the answer again
    let _ = app.update(Message::Regenerate(1));
    assert!(app.is_streaming());
    assert_eq!(texts(&app), vec!["first question", ""], "the old answer is out of the way while the new one is written");
    finish(&mut app, "second answer");
    assert_eq!(texts(&app), vec!["first question", "second answer"]);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 900.0), app.view());
    ui.find("2 / 2").expect("the arrows say which version is shown");
    drop(ui);
    // the arrow back shows the first answer again, and the second is still there
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 900.0), app.view());
    ui.click("‹").unwrap();
    let msgs: Vec<Message> = ui.into_messages().collect();
    assert!(msgs.iter().any(|m| matches!(m, Message::SwitchVersion(1, 0))), "{msgs:?}");
    let _ = app.update(Message::SwitchVersion(1, 0));
    assert_eq!(texts(&app), vec!["first question", "first answer"]);
    let _ = app.update(Message::SwitchVersion(1, 1));
    assert_eq!(texts(&app), vec!["first question", "second answer"]);

    // changing the question: the box gets its text, Cancel leaves everything as it was
    let _ = app.update(Message::EditMessage(0));
    assert_eq!(app.editor_text(), "first question");
    let _ = app.update(Message::CancelEdit);
    assert_eq!(app.editor_text(), "");
    let _ = app.update(Message::EditMessage(0));
    let _ = app.update(Message::Edit(iced::widget::text_editor::Action::Move(iced::widget::text_editor::Motion::DocumentEnd)));
    let _ = app.update(Message::Edit(iced::widget::text_editor::Action::Edit(iced::widget::text_editor::Edit::Paste(std::sync::Arc::new(" (changed)".to_string())))));
    let _ = app.update(Message::Send);
    assert_eq!(app.message_text(0).unwrap(), "first question (changed)");
    finish(&mut app, "answer to the changed question");
    assert_eq!(app.message_count(), 2);
    assert_eq!(app.message_text(1).unwrap(), "answer to the changed question");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 900.0), app.view());
    ui.find("2 / 2").expect("the question has two versions now");
    drop(ui);
    // everything is on disk: a new start finds all the versions
    let store = Store::load(&dir.path().join("chats.json"));
    let chat = store.chat(0).unwrap();
    assert_eq!(chat.variants.len(), 2, "the first and the second answer");
    assert_eq!(chat.versions_at(0).count, 2);
    assert_eq!(chat.versions_at(1).count, 1);
    // no switching, retrying or changing while an answer is being written
    let _ = app.update(Message::Regenerate(1));
    assert!(app.is_streaming());
    let before = texts(&app);
    let _ = app.update(Message::SwitchVersion(0, 0));
    let _ = app.update(Message::EditMessage(0));
    let _ = app.update(Message::Regenerate(1));
    assert_eq!(texts(&app), before);
    assert_eq!(app.editor_text(), "");
}
