//! A picture of the clipboard is attached to the next question like a file you pick; text on the
//! clipboard and the other pages are not touched.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Attachment, Message, View};
use fllm_app::store::Store;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    (app, dir)
}

#[test]
fn only_the_chat_looks_at_the_clipboard() {
    let (mut app, _dir) = app();
    assert!(app.update(Message::PasteKey).units() >= 1, "the chat box takes pictures");
    let _ = app.update(Message::Go(View::Settings));
    assert_eq!(app.update(Message::PasteKey).units(), 0, "another page does not");
}

#[test]
fn nothing_on_it_changes_nothing_and_a_failure_is_said() {
    let (mut app, _dir) = app();
    assert_eq!(app.update(Message::PasteImage(Ok(None))).units(), 0);
    assert!(app.attachments().is_empty());
    let _ = app.update(Message::PasteImage(Err("The clipboard could not be read: x".into())));
    assert_eq!(app.toast(), Some("The clipboard could not be read: x"));
}

#[test]
fn a_picture_becomes_an_attachment_named_image_png() {
    let (mut app, _dir) = app();
    let png = fllm_app::clipboard_image::png_of(2, 2, &[255; 16]).unwrap();
    assert!(app.update(Message::PasteImage(Ok(Some(png.clone())))).units() >= 1);
    // the answer of the task that shrinks it is the one of every picture
    let data_url = format!("data:image/png;base64,{}", "AAAA");
    let _ = app.update(Message::ImageReady(Ok(("image.png".into(), data_url.clone()))));
    assert!(matches!(app.attachments(), [Attachment::Image { name, data_url: d }] if name == "image.png" && *d == data_url));
}

#[test]
fn a_long_text_is_attached_as_a_file_and_a_short_one_goes_into_the_box() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::ChatPasted(Some("short words".into())));
    assert_eq!(app.editor_text().trim_end(), "short words");
    assert!(app.attachments().is_empty());
    let _ = app.update(Message::ChatPasted(None));
    let _ = app.update(Message::ChatPasted(Some(String::new())));
    assert_eq!(app.editor_text().trim_end(), "short words", "nothing on the clipboard changes nothing");
    // 6000 units still go into the box, 6001 do not (the page counts the units of UTF-16)
    let _ = app.update(Message::ChatPasted(Some("a".repeat(fllm_app::app::PASTE_AS_FILE))));
    assert!(app.attachments().is_empty());
    let long = "é".repeat(fllm_app::app::PASTE_AS_FILE + 1);
    let _ = app.update(Message::ChatPasted(Some(long.clone())));
    assert!(matches!(app.attachments(), [Attachment::Text { name, text, size }] if name == "Pasted text.txt" && *text == long && *size == 6001));
    assert_eq!(app.toast(), Some("Long text attached as a file"));
    // a emoji is two units: 3001 of them are over the limit
    let (mut app, _dir2) = self::app();
    let _ = app.update(Message::ChatPasted(Some("😀".repeat(3001))));
    assert_eq!(app.attachments().len(), 1);
}
