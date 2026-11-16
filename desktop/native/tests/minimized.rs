//! The panel of the browser does not poll while the window is minimized (the page checks
//! `document.hidden`); it looks again at once when the window comes back.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
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
fn the_browser_is_looked_at_only_when_the_code_page_is_shown_and_the_window_is_not_minimized() {
    let (mut app, _dir) = app();
    assert!(!app.browser_visible(), "the chat page is shown");
    let _ = app.update(Message::Go(View::Code));
    assert!(app.browser_visible());
    let _ = app.update(Message::Minimized(true));
    assert!(!app.browser_visible(), "minimized");
    let back = app.update(Message::Minimized(false));
    assert!(app.browser_visible());
    assert!(back.units() >= 1, "the panel looks again as the window comes back");
    // another page: no look, and no wake-up on the way back
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::Minimized(true));
    assert_eq!(app.update(Message::Minimized(false)).units(), 0);
    assert!(!app.browser_visible());
}
