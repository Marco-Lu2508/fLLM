//! Attachments: images, whole files and documents of a chat; their chips over the input; the parts
//! of a request with images; and the passages of the documents of the chat and of its project.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Project, ProjectDoc, Record, Role, Store};
use iced_test::Simulator;

fn rec(role: Role, content: &str) -> Record {
    Record {
        role,
        content: content.into(),
        reasoning: String::new(),
        stats: None,
        passages: None,
        images: Vec::new(),
        files: Vec::new(),
        tool_calls: None,
        tool_call_id: None,
        tool_cards: Vec::new(),
    }
}

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

/// A text file of the test on disk, so the flow has a real size to read.
fn file_of(dir: &tempfile::TempDir, name: &str, text: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// A 1x1 PNG as a data URL (the thumbnail of the window draws it).
const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

#[test]
fn a_text_file_rides_whole_with_the_question_and_the_chat_is_named_after_it() {
    let (mut app, dir) = app_with(Store::default());
    let path = file_of(&dir, "notes.txt", "The pump runs dry never.");
    let _ = app.update(Message::AttachChosen(Ok((path, Some("The pump runs dry never.".into())))));
    assert_eq!(app.attachments().len(), 1, "the file is attached");
    assert!(app.can_send(), "a question may be nothing but an attachment");

    let _ = app.update(Message::Send);
    assert!(app.attachments().is_empty(), "the attachments are used up");

    let wire = app.wire_messages();
    let user = wire.iter().find(|m| m.role == "user").expect("the question is on the wire");
    assert_eq!(
        user.content,
        "File notes.txt:\n```\nThe pump runs dry never.\n```\n\n",
        "the whole file goes in front of the (empty) question"
    );
    assert!(user.parts.is_none(), "without images the content stays a string");

    let after = saved(&dir);
    assert_eq!(after.chats.len(), 1);
    let message = &after.chats[0].messages[0];
    assert_eq!(message.files[0].name, "notes.txt");
    assert_eq!(message.files[0].text, "The pump runs dry never.");
    assert_eq!(after.chats[0].title, "notes.txt", "the Studio names a chat after its first attachment");
}

#[test]
fn a_file_that_does_not_fit_becomes_a_document_of_the_chat_and_is_searched() {
    let (mut app, dir) = app_with(Store::default());
    let big = "The pump must never run dry. ".repeat(400);
    let path = file_of(&dir, "big.txt", &big);
    let _ = app.update(Message::AttachChosen(Ok((path, Some(big)))));
    assert_eq!(app.attachments().len(), 1, "the file is attached as a document");
    assert!(
        app.toast().is_some_and(|t| t.contains("too long for the context: the passages that match your question are used")),
        "the line of the Studio's ragPlace: {:?}",
        app.toast()
    );

    let question = std::sync::Arc::new("When may it never run dry?".to_string());
    let _ = app.update(Message::Edit(iced::widget::text_editor::Action::Edit(
        iced::widget::text_editor::Edit::Paste(question),
    )));
    let _ = app.update(Message::Send);

    let wire = app.wire_messages();
    let user = wire.iter().find(|m| m.role == "user").expect("the question is on the wire");
    assert!(
        user.content.starts_with("Passages from the attached documents"),
        "the passages of the document of this chat travel with the question: {}",
        &user.content[..120.min(user.content.len())]
    );
    assert!(user.content.contains("big.txt, chunk"), "the source of the passage is named");

    let after = saved(&dir);
    assert_eq!(after.chats[0].docs.len(), 1, "the document is stored in the chat");
    assert_eq!(after.chats[0].docs[0].name, "big.txt");

    // the cross of the chip takes it out of the chat again
    let id = after.chats[0].docs[0].id.clone();
    let _ = app.update(Message::ChatDocRemove(id));
    assert!(saved(&dir).chats[0].docs.is_empty(), "the document is gone");
}

#[test]
fn an_image_travels_as_parts_and_the_meter_counts_it() {
    let (mut app, dir) = app_with(Store::default());
    let _ = app.update(Message::ImageReady(Ok(("picture.png".into(), PNG.into()))));
    assert_eq!(app.attachments().len(), 1);

    let _ = app.update(Message::Send);
    let wire = app.wire_messages();
    let user = wire.iter().find(|m| m.role == "user").expect("the question is on the wire");
    let parts = user.parts.as_ref().expect("images make the content an array");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "Describe the image.", "an image without a question says so");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], PNG, "the data URL travels unchanged");
    assert!(user.content.is_empty(), "the plain text stays for the meter");

    let after = saved(&dir);
    assert_eq!(after.chats[0].messages[0].images, vec![PNG.to_string()], "the image is kept with the message");
    assert_eq!(after.chats[0].title, "picture.png");
    // the meter counts from the estimate when the chat is (re)opened: an image is 256 tokens
    let _ = app.update(Message::Stream(0, StreamEvent::Done));
    let _ = app.update(Message::OpenChat(0));
    assert!(app.used_tokens() >= 256, "an image counts 256 tokens in the meter: {}", app.used_tokens());
}

#[test]
fn the_documents_of_the_project_and_of_the_chat_are_both_searched() {
    let mut store = Store::default();
    let mut p = Project { id: "p1".into(), name: "Work".into(), ..Default::default() };
    p.docs.push(ProjectDoc {
        id: "pd1".into(),
        name: "project.pdf".into(),
        text: "Alpha is the first letter of the project.".into(),
        pages: Some(1),
        chars: 43,
        added: 0,
    });
    store.projects.push(p);
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::PrjOpen("p1".into()));
    let _ = app.update(Message::PrjNewChat);
    let big = "Omega is the last letter of this chat. ".repeat(400);
    let path = file_of(&dir, "chatdoc.txt", &big);
    let _ = app.update(Message::AttachChosen(Ok((path, Some(big)))));
    let question = std::sync::Arc::new("alpha and omega?".to_string());
    let _ = app.update(Message::Edit(iced::widget::text_editor::Action::Edit(
        iced::widget::text_editor::Edit::Paste(question),
    )));
    let _ = app.update(Message::Send);

    let wire = app.wire_messages();
    let user = wire.iter().find(|m| m.role == "user").expect("the question is on the wire");
    assert!(user.content.contains("project.pdf, chunk"), "the knowledge of the project is searched: {}", &user.content[..200]);
    assert!(user.content.contains("chatdoc.txt, chunk"), "the documents of the chat are searched too");
}

#[test]
fn the_chips_over_the_input_and_the_attachments_in_the_bubble_draw() {
    // the row: a stored document, a whole file and an image
    let (mut app, dir) = app_with(Store::default());
    let big = "Stored for every question of this chat. ".repeat(400);
    let path = file_of(&dir, "stored.txt", &big);
    let _ = app.update(Message::AttachChosen(Ok((path, Some(big)))));
    let _ = app.update(Message::Send);
    let _ = app.update(Message::Stream(0, StreamEvent::Done));
    let path = file_of(&dir, "notes.txt", "small");
    let _ = app.update(Message::AttachChosen(Ok((path, Some("small".into())))));
    let _ = app.update(Message::ImageReady(Ok(("picture.png".into(), PNG.into()))));
    {
        let mut ui = sim(&app);
        ui.find("stored.txt").expect("the chip of the document of this chat");
        ui.find("notes.txt").expect("the chip of the attached file");
    }
    save_png(&app, "att-row");

    // the bubble: an image and a file with an empty question
    let (mut app2, dir2) = app_with(Store::default());
    let path = file_of(&dir2, "notes.txt", "small");
    let _ = app2.update(Message::AttachChosen(Ok((path, Some("small".into())))));
    let _ = app2.update(Message::ImageReady(Ok(("picture.png".into(), PNG.into()))));
    let _ = app2.update(Message::Send);
    {
        let mut ui = sim(&app2);
        ui.find("notes.txt").expect("the file chip inside the bubble");
    }
    save_png(&app2, "att-bubble");
}

#[test]
fn a_chat_file_from_before_the_feature_loads_and_a_too_big_document_is_cut() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let mut chat = ChatRecord::new(0, "old", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.docs.push(ProjectDoc {
        id: "d1".into(),
        name: "huge.txt".into(),
        text: "t".repeat(700_000),
        pages: None,
        chars: 999_999,
        added: 0,
    });
    Store { chats: vec![chat], next_id: 1, theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let store = Store::load(&path);
    assert_eq!(store.chats[0].docs.len(), 1, "the document stays");
    assert_eq!(store.chats[0].docs[0].text.chars().count(), fllm_app::rag::MAX_DOC_CHARS, "the text is cut to the limit");
    assert!(store.chats[0].docs[0].chars <= fllm_app::rag::MAX_DOC_CHARS, "the counted characters follow the text");
    // a message with images and files from a future version loads too (the fields are optional)
    let json = r#"{"chats":[{"id":0,"title":"t","messages":[{"role":"user","content":"q","images":["data:image/png;base64,AA=="],"files":[{"name":"a.txt","text":"x"}]}],"updated":0,"archived":false,"pinned":false}],"next_id":1,"theme":"light"}"#;
    std::fs::write(&path, json).unwrap();
    let store = Store::load(&path);
    assert_eq!(store.chats[0].messages[0].images.len(), 1);
    assert_eq!(store.chats[0].messages[0].files[0].name, "a.txt");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| "/opt/data/cache/scratch/fLLM/desktop/native/ui-snapshots".into()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}
