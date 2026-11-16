//! Row 85: the places of the panel's logic that the page's scenarios do not reach (a tick counter
//! that runs on, a key held alone, a comment that is too long) and the hooks of the Code page
//! around them. Each test names the sentence a mutation of `browser.rs` / `code.rs` has to break.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::browser::{Input, KeyPress, Panel, Rect, WheelUnit};
use fllm_app::code::{BrMsg, CodeMsg, Workspace};
use fllm_app::store::Store;
use serde_json::{json, Value};

const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn status(running: bool, url: &str) -> Value {
    json!({ "browser": "/usr/bin/chromium", "running": running, "url": url, "profile": "/p", "settings": { "enabled": true, "show_window": false, "persistent_profile": true, "local_addresses": "auto", "executable": "" } })
}

fn running_panel() -> Panel {
    let mut p = Panel::new("linux");
    p.apply_status(Some(&status(true, "http://localhost:3000/")));
    p.apply_frame(&json!({ "running": true, "image": format!("data:image/png;base64,{PIXEL}"), "key": "k1", "width": 1280, "height": 800, "url": "http://localhost:3000/", "title": "App" }));
    p
}

const RECT: Rect = Rect { left: 0.0, top: 0.0, width: 640.0, height: 400.0 };

#[test]
fn the_status_is_read_at_the_first_tick_then_on_every_twelfth() {
    let mut p = Panel::new("linux");
    // a server that does not answer leaves the status empty: every tick asks again
    let asked: Vec<bool> = (0..5).map(|_| p.tick_reads_status()).collect();
    assert_eq!(asked, [true; 5], "no status yet");
    let mut p = Panel::new("linux");
    assert!(p.tick_reads_status());
    p.apply_status(Some(&status(true, "")));
    let rest: Vec<bool> = (1..40).map(|_| p.tick_reads_status()).collect();
    let want: Vec<bool> = (1..40).map(|n| n % 12 == 0).collect();
    assert_eq!(rest, want, "the twelfth, the 24th and the 36th tick read it");
}

#[test]
fn a_tick_every_650_ms_while_it_runs_and_every_2000_while_it_does_not() {
    assert_eq!(running_panel().tick_pause_ms(), 650);
    assert_eq!(Panel::new("linux").tick_pause_ms(), 2000);
    let mut stopped = running_panel();
    stopped.apply_frame(&json!({ "running": false }));
    assert_eq!(stopped.tick_pause_ms(), 2000);
}

#[test]
fn the_turns_of_the_wheel_add_up_until_they_are_sent() {
    let mut p = running_panel();
    assert!(p.wheel((10.0, 10.0), RECT, (1.0, 2.0), WheelUnit::Pixels), "the first turn starts the timer");
    assert!(!p.wheel((20.0, 20.0), RECT, (3.0, 4.0), WheelUnit::Pixels), "the second joins it");
    match p.flush_wheel() {
        Some(Input::Wheel { dx, dy, x, y }) => {
            assert_eq!((dx, dy), (4.0, 6.0));
            assert_eq!((x, y), (40.0, 40.0), "at the place of the last turn, in the pixels of the page");
        }
        other => panic!("{other:?}"),
    }
    assert!(p.flush_wheel().is_none(), "nothing is left");
}

#[test]
fn a_modifier_held_alone_and_a_function_key_that_is_none() {
    let p = running_panel();
    for alone in ["Control", "Shift", "Alt", "Meta"] {
        assert!(p.key(&KeyPress { key: alone.into(), ctrl: alone == "Control", ..Default::default() }).is_none(), "{alone} alone is no key");
    }
    assert_eq!(p.key(&KeyPress { key: "F1".into(), ..Default::default() }), Some(Input::Key("F1".into())));
    assert_eq!(p.key(&KeyPress { key: "F12".into(), ..Default::default() }), Some(Input::Key("F12".into())));
    for none in ["F01", "F0", "F13", "F+1"] {
        assert!(p.key(&KeyPress { key: none.into(), ..Default::default() }).is_none(), "{none} is no key");
    }
}

fn draft_of(p: &mut Panel, text: &str) {
    p.toggle_commenting();
    let answer = json!({ "ok": true, "ref": 4, "role": "button", "name": "Save", "tag": "button", "text": text, "url": "http://localhost:3000/a", "rect": { "x": 10, "y": 20, "w": 100, "h": 40 } });
    assert!(p.input_answered(&Input::Locate { x: 1.0, y: 1.0 }, Ok(answer)).is_none());
}

#[test]
fn comments_the_way_the_page_keeps_them() {
    let mut p = running_panel();
    draft_of(&mut p, &"é".repeat(300));
    assert!(p.draft.is_some());
    p.toggle_commenting();
    assert!(p.draft.is_none(), "leaving comment mode drops the element that was picked");
    draft_of(&mut p, &"x".repeat(300));
    assert!(p.add_comment("one"));
    assert_eq!(p.comments[0].text.encode_utf16().count(), 160, "the text of a comment is clipped at 160 units");
    draft_of(&mut p, "second");
    assert!(p.add_comment("two"));
    draft_of(&mut p, "third");
    assert!(p.add_comment("three"));
    p.delete_comment(1);
    let notes: Vec<&str> = p.comments.iter().map(|c| c.note.as_str()).collect();
    assert_eq!(notes, ["one", "three"], "the comment that was asked for goes");
    p.delete_comment(9);
    assert_eq!(p.comments.len(), 2, "a place that is not there changes nothing");
    assert_eq!(p.send_text("   "), "Please act on my comments on the browser page.", "a box of spaces is empty");
    assert_eq!(p.send_text(" fix it "), " fix it ");
}

#[test]
fn the_page_the_browser_is_on_comes_from_the_panel_first() {
    let mut p = Panel::new("linux");
    p.apply_status(Some(&status(true, "http://status/")));
    assert_eq!(p.current_url(), "http://status/", "from the status while no frame came");
    p.apply_frame(&json!({ "running": true, "image": format!("data:image/png;base64,{PIXEL}"), "key": "k", "width": 100, "height": 50, "url": "http://frame/", "title": "" }));
    assert_eq!(p.current_url(), "http://frame/", "the frame is newer");
    let mut none = Panel::new("linux");
    assert_eq!(none.current_url(), "");
    none.apply_status(None);
    assert_eq!(none.current_url(), "");
}

#[test]
fn a_page_that_can_be_drawn_again_is_drawn_again() {
    let mut p = running_panel();
    p.apply_frame(&json!({ "running": true, "undrawable": true, "url": "http://down/" }));
    assert_eq!(p.undrawable.as_deref(), Some("http://down/"));
    assert!(p.empty().text.contains("cannot be drawn"));
    p.apply_frame(&json!({ "running": true, "image": format!("data:image/png;base64,{PIXEL}"), "key": "k2", "width": 1280, "height": 800, "url": "http://up/", "title": "Up" }));
    assert!(p.undrawable.is_none(), "the next picture clears the note");
    assert!(p.show_image());
}

#[test]
fn there_is_no_picture_of_a_browser_that_is_not_running() {
    let mut p = running_panel();
    assert!(p.show_image());
    p.running = false;
    assert!(p.image.is_some(), "the picture is still kept");
    assert!(!p.show_image(), "but not shown");
    assert!(!p.view().show_image);
}

#[test]
fn an_element_without_height_gets_no_frame() {
    let mut p = running_panel();
    p.toggle_commenting();
    let flat = json!({ "ok": true, "ref": 1, "role": "separator", "name": "", "tag": "hr", "text": "", "url": "u", "rect": { "x": 10, "y": 20, "w": 100, "h": 0 } });
    assert!(p.input_answered(&Input::Locate { x: 1.0, y: 1.0 }, Ok(flat)).is_none());
    assert!(p.pin(640.0).is_none(), "a frame of no height is not drawn");
    let tall = json!({ "ok": true, "ref": 1, "role": "button", "name": "", "tag": "button", "text": "", "url": "u", "rect": { "x": 10, "y": 20, "w": 100, "h": 40 } });
    p.input_answered(&Input::Locate { x: 1.0, y: 1.0 }, Ok(tall));
    assert_eq!(p.pin(640.0), Some([5.0, 10.0, 50.0, 20.0]), "half the width of the page: half the numbers");
}

#[test]
fn the_label_of_a_typed_text_is_clipped_at_sixty_units() {
    let long = "a".repeat(100);
    let (verb, what) = fllm_app::browser::tool_label("browser_type", &json!({ "ref": 3, "text": long, "submit": true })).unwrap();
    assert_eq!(verb, "Browser type");
    assert_eq!(what, format!("ref 3: {} + Enter", "a".repeat(60)));
}

// ---- the hooks in the window --------------------------------------------------------------------------------------

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None }))));
    (app, dir)
}

fn br(app: &mut App, m: BrMsg) -> iced::Task<Message> {
    app.update(Message::Code(CodeMsg::Br(m)))
}

fn live(app: &mut App) {
    let _ = br(app, BrMsg::Toggle);
    let _ = br(app, BrMsg::Status { then_tick: false, answer: Some(status(true, "http://localhost:3000/")) });
    let epoch = app.code().br_epoch;
    let frame = json!({ "running": true, "image": format!("data:image/png;base64,{PIXEL}"), "key": "k1", "width": 1280, "height": 800, "url": "http://localhost:3000/", "title": "App" });
    let _ = br(app, BrMsg::TickDone { epoch, status: None, frame: Some(frame) });
}

fn focused(app: &mut App) {
    let _ = br(app, BrMsg::Move(iced::Point::new(100.0, 100.0), iced::Size::new(352.0, 220.0)));
    let _ = br(app, BrMsg::Click);
    let _ = br(app, BrMsg::Answered(Input::Click { x: 0.0, y: 0.0 }, Ok(json!({ "ok": true }))));
    assert!(app.code().br_focus);
}

#[test]
fn hiding_the_panel_ends_the_chain_of_ticks() {
    let (mut app, _dir) = app();
    live(&mut app);
    let before = app.code().br_epoch;
    let _ = br(&mut app, BrMsg::Hide);
    assert!(!app.code().browser.open);
    assert_ne!(app.code().br_epoch, before, "a tick that is on its way finds a new epoch");
    assert_eq!(br(&mut app, BrMsg::Tick(before)).units(), 0, "and does not go on");
}

#[test]
fn a_sent_address_lets_the_frames_write_the_box_again() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::UrlEdit("local".into()));
    assert!(app.code().browser.url_focused, "while you type, a frame must not overwrite the box");
    let _ = br(&mut app, BrMsg::Submit);
    assert!(!app.code().browser.url_focused);
}

#[test]
fn a_window_that_is_not_shown_gets_no_keys_and_a_stopped_browser_no_paste() {
    let (mut app, _dir) = app();
    live(&mut app);
    focused(&mut app);
    let a = KeyPress { key: "a".into(), ..Default::default() };
    assert_eq!(app.update(Message::BrKey(a.clone())).units(), 1);
    let _ = app.update(Message::Minimized(true));
    assert_eq!(app.update(Message::BrKey(a.clone())).units(), 0, "minimized: the keys stay here");
    let _ = app.update(Message::Minimized(false));
    // the browser stops while the panel still has the focus
    let epoch = app.code().br_epoch;
    let _ = br(&mut app, BrMsg::TickDone { epoch, status: None, frame: Some(json!({ "running": false })) });
    assert!(!app.code().browser.running);
    let paste = KeyPress { key: "v".into(), ctrl: true, ..Default::default() };
    assert_eq!(app.update(Message::BrKey(paste)).units(), 0, "nothing to paste into");
}

#[test]
fn the_usual_addresses_are_offered_only_while_nothing_is_open() {
    use iced_test::Simulator;
    let (mut app, _dir) = app();
    live(&mut app);
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    // a page is open: the stage shows it, there is nothing to offer
    let mut ui = Simulator::with_size(settings.clone(), iced::Size::new(1500.0, 900.0), app.view());
    assert!(ui.find("localhost:5173").is_err());
    drop(ui);
    // the browser runs but the page cannot be drawn (the stage is the empty state) and has an address
    let epoch = app.code().br_epoch;
    let _ = br(&mut app, BrMsg::TickDone { epoch, status: None, frame: Some(json!({ "running": true, "undrawable": true, "url": "http://down/" })) });
    let ui = Simulator::with_size(settings, iced::Size::new(1500.0, 900.0), app.view());
    let mut ui = ui;
    assert!(ui.find("localhost:5173").is_err(), "a browser that is on a page does not offer other pages");
}
