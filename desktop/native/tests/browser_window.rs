//! Row 85: the *Browser* panel of the Code page and the browser tools — the live page, what the
//! user does in it, the comments for the agent, and what the model gets back. The state machine is
//! compared with the page in `tests/browser_fixtures.rs`; here the window around it.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::browser::{Input, KeyPress};
use fllm_app::code::{client_tools, BrMsg, CodeMsg, Mode, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::Arc;

const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app_with(vision: bool) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let mut info = ModelInfo::new("qwen", 8192);
    info.vision = vision;
    let _ = app.update(Message::Connected(Ok(info)));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn app() -> (App, tempfile::TempDir) {
    app_with(true)
}

fn code(app: &mut App, m: CodeMsg) -> iced::Task<Message> {
    app.update(Message::Code(m))
}

fn br(app: &mut App, m: BrMsg) -> iced::Task<Message> {
    code(app, CodeMsg::Br(m))
}

fn open_folder(app: &mut App) {
    let _ = code(app, CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None })));
    let def = |n: &str| json!({ "type": "function", "function": { "name": n, "description": n, "parameters": { "type": "object", "properties": {} } } });
    let names = [("browser_open", Permission::Network), ("browser_click", Permission::Network), ("browser_screenshot", Permission::Network)];
    let mut tools: Vec<Tool> = names.iter().map(|(n, p)| Tool::server(n, *p)).collect();
    let mut defs: Vec<Value> = names.iter().map(|(n, _)| def(n)).collect();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    let _ = code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1500.0, 1800.0), app.view())
}

fn status(enabled: bool, running: bool, url: &str) -> Value {
    json!({ "browser": "/usr/bin/chromium", "running": running, "url": url, "profile": "/p", "settings": { "enabled": enabled, "show_window": false, "persistent_profile": true, "local_addresses": "auto", "executable": "" } })
}

/// The panel is open and the page is shown.
fn live(app: &mut App) {
    open_folder(app);
    let _ = br(app, BrMsg::Toggle);
    let _ = br(app, BrMsg::Status { then_tick: false, answer: Some(status(true, true, "http://localhost:3000/")) });
    let epoch = app.code().br_epoch;
    let frame = json!({ "running": true, "image": format!("data:image/png;base64,{PIXEL}"), "key": "k1", "width": 1280, "height": 800, "url": "http://localhost:3000/", "title": "App" });
    let _ = br(app, BrMsg::TickDone { epoch, status: None, frame: Some(frame) });
}

fn mouse(app: &mut App, x: f32, y: f32) {
    let _ = br(app, BrMsg::Move(iced::Point::new(x, y), iced::Size::new(352.0, 220.0)));
}

fn say(app: &mut App, text: &str) {
    let _ = code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    let _ = code(app, CodeMsg::Send);
}

#[test]
fn the_header_has_the_button_and_the_panel_opens_and_closes() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    assert!(sim(&app).find("Browser").is_ok());
    assert!(!app.code().browser.open);
    let task = br(&mut app, BrMsg::Toggle);
    assert_eq!(task.units(), 1, "the status is read first");
    assert!(app.code().browser.open);
    assert!(sim(&app).find("Looking for the browser ...").is_ok(), "until the server says something");
    let _ = br(&mut app, BrMsg::Hide);
    assert!(!app.code().browser.open);
}

#[test]
fn a_switched_off_browser_says_so_and_points_to_the_settings() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = br(&mut app, BrMsg::Toggle);
    let _ = br(&mut app, BrMsg::Status { then_tick: true, answer: Some(status(false, false, "")) });
    let mut ui = sim(&app);
    assert!(ui.find("The browser is switched off. When it is on, the agent can open pages, read them, click and type, and you see what it sees here and can take over.").is_ok());
    assert!(ui.find("off").is_ok());
    ui.click("Open Browser settings").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::Br(BrMsg::OpenSettings))]), "{sent:?}");
}

#[test]
fn the_page_is_shown_and_the_header_says_it_runs() {
    let (mut app, _dir) = app();
    live(&mut app);
    let b = &app.code().browser;
    assert!(b.running && b.show_image());
    assert_eq!(b.url_input, "http://localhost:3000/");
    let mut ui = sim(&app);
    assert!(ui.find("Browser ●").is_ok(), "a dot while it runs");
    assert!(ui.find("Click, type and scroll in the page to use it yourself. The agent sees the result.").is_ok());
    assert!(ui.find("running").is_ok());
}

#[test]
fn an_old_chain_of_ticks_is_ignored() {
    let (mut app, _dir) = app();
    live(&mut app);
    let stale = app.code().br_epoch.wrapping_sub(1);
    let _ = br(&mut app, BrMsg::TickDone { epoch: stale, status: None, frame: Some(json!({ "running": false })) });
    assert!(app.code().browser.running, "a frame of an older chain changes nothing");
    let none = br(&mut app, BrMsg::Tick(stale));
    assert_eq!(none.units(), 0);
}

#[test]
fn a_click_on_the_page_is_sent_in_the_pixels_of_the_page_and_in_order() {
    let (mut app, _dir) = app();
    live(&mut app);
    mouse(&mut app, 176.0, 110.0);
    let first = br(&mut app, BrMsg::Click);
    assert_eq!(first.units(), 1, "the click goes to the server");
    assert!(app.code().br_focus, "the stage has the keys now");
    mouse(&mut app, 0.0, 0.0);
    let second = br(&mut app, BrMsg::Click);
    assert_eq!(second.units(), 0, "it waits for the first (fast input arrives in the order it was made)");
    let after = br(&mut app, BrMsg::Answered(Input::Click { x: 640.0, y: 400.0 }, Ok(json!({ "ok": true }))));
    assert!(after.units() >= 2, "the next input and the look at the page");
}

#[test]
fn a_refusal_is_a_toast() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::Answered(Input::Navigate("http://10.0.0.1".into()), Ok(json!({ "ok": false, "error": "That address is not allowed." }))));
    assert_eq!(app.toast.as_deref(), Some("That address is not allowed."));
    let _ = br(&mut app, BrMsg::Answered(Input::Reload, Err("connection refused".into())));
    assert_eq!(app.toast.as_deref(), Some("The browser did not answer: connection refused"));
}

#[test]
fn keys_go_to_the_page_only_when_it_has_the_focus() {
    let (mut app, _dir) = app();
    live(&mut app);
    let a = KeyPress { key: "a".into(), ..Default::default() };
    assert_eq!(app.update(Message::BrKey(a.clone())).units(), 0, "before a click nothing is passed on");
    mouse(&mut app, 100.0, 100.0);
    let _ = br(&mut app, BrMsg::Click);
    let _ = br(&mut app, BrMsg::Answered(Input::Click { x: 0.0, y: 0.0 }, Ok(json!({ "ok": true }))));
    assert_eq!(app.update(Message::BrKey(a)).units(), 1);
    // a key the page does not forward
    assert_eq!(app.update(Message::BrKey(KeyPress { key: "Shift".into(), ..Default::default() })).units(), 0);
    // the paste shortcut reads the clipboard
    let paste = app.update(Message::BrKey(KeyPress { key: "v".into(), ctrl: true, ..Default::default() }));
    assert_eq!(paste.units(), 1);
    // typing in the address box takes the keys back
    let _ = br(&mut app, BrMsg::UrlEdit("local".into()));
    assert!(!app.code().br_focus);
    assert_eq!(app.update(Message::BrKey(KeyPress { key: "b".into(), ..Default::default() })).units(), 0);
}

#[test]
fn the_wheel_is_collected_for_a_moment_and_sent_once() {
    let (mut app, _dir) = app();
    live(&mut app);
    mouse(&mut app, 176.0, 110.0);
    let first = br(&mut app, BrMsg::Scroll(iced::mouse::ScrollDelta::Lines { x: 0.0, y: -3.0 }));
    assert_eq!(first.units(), 1, "the 50 ms start");
    let second = br(&mut app, BrMsg::Scroll(iced::mouse::ScrollDelta::Lines { x: 0.0, y: -2.0 }));
    assert_eq!(second.units(), 0, "the second turn joins the first");
    let sent = br(&mut app, BrMsg::WheelFlush);
    assert_eq!(sent.units(), 1);
}

#[test]
fn the_address_bar_sends_what_was_typed() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::UrlEdit("  localhost:5173/app ".into()));
    assert_eq!(br(&mut app, BrMsg::Submit).units(), 1);
    let _ = br(&mut app, BrMsg::UrlEdit("   ".into()));
    // the first one is still on its way: the second is not even queued
    assert_eq!(br(&mut app, BrMsg::Submit).units(), 0);
    assert_eq!(br(&mut app, BrMsg::Back).units(), 0, "it queues behind the first");
}

#[test]
fn closing_asks_first() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::CloseAsk);
    assert!(sim(&app).find("Close the browser? Pages that are open are lost; logins stay in its profile.").is_ok());
    assert_eq!(br(&mut app, BrMsg::CloseCancel).units(), 0);
    let _ = br(&mut app, BrMsg::CloseAsk);
    assert_eq!(br(&mut app, BrMsg::CloseOk).units(), 1);
    let _ = br(&mut app, BrMsg::Answered(Input::Close, Ok(json!({ "ok": true }))));
    assert!(!app.code().browser.running && !app.code().browser.show_image(), "the picture goes at once");
}

fn comment_on_something(app: &mut App, note: &str) {
    let _ = br(app, BrMsg::Comment);
    mouse(app, 176.0, 110.0);
    let _ = br(app, BrMsg::Click);
    let located = json!({ "ok": true, "ref": 7, "role": "button", "name": "Save", "tag": "button", "text": "Save", "url": "http://localhost:3000/settings", "rect": { "x": 128, "y": 80, "w": 256, "h": 40 } });
    let _ = br(app, BrMsg::Answered(Input::Locate { x: 640.0, y: 400.0 }, Ok(located)));
    let _ = br(app, BrMsg::NoteEdit(Action::Edit(Edit::Paste(Arc::new(note.to_string())))));
    let _ = br(app, BrMsg::NoteAdd);
}

#[test]
fn a_comment_goes_with_the_next_message_once() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::Comment);
    assert!(sim(&app).find("Comment mode: click an element on the page, then write what the agent should do about it.").is_ok());
    let _ = br(&mut app, BrMsg::Comment);
    comment_on_something(&mut app, "make it blue");
    assert!(sim(&app).find("Comments for the agent (1)").is_ok());
    assert!(sim(&app).find("Send to the agent").is_ok());
    // a message: the comments ride in front of it
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    say(&mut app, "please fix it");
    let item = app.code().items.iter().find_map(|i| if let fllm_app::code::Item::User { text, sent } = i { Some((text.clone(), sent.clone())) } else { None }).expect("a message");
    assert_eq!(item.0, "please fix it", "the card shows only what was typed");
    let sent = item.1.expect("the model gets more");
    assert!(sent.starts_with("[The user pointed at elements of the browser page and left comments. Act on them."), "{sent}");
    assert!(sent.contains("1. button \"Save\" <button> (ref 7) on http://localhost:3000/settings: make it blue"), "{sent}");
    assert!(sent.ends_with("\n\nplease fix it"), "{sent}");
    assert!(app.code().browser.comments.is_empty(), "taken once");
}

#[test]
fn send_to_the_agent_fills_an_empty_box_and_goes() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    comment_on_something(&mut app, "check this");
    let _ = br(&mut app, BrMsg::SendComments);
    let item = app.code().items.iter().find_map(|i| if let fllm_app::code::Item::User { text, .. } = i { Some(text.clone()) } else { None }).expect("a message");
    assert_eq!(item, "Please act on my comments on the browser page.");
}

#[test]
fn the_note_is_drawn_while_an_element_is_chosen_and_can_be_dropped() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = br(&mut app, BrMsg::Comment);
    mouse(&mut app, 176.0, 110.0);
    let _ = br(&mut app, BrMsg::Click);
    let located = json!({ "ok": true, "ref": 7, "role": "button", "name": "Save", "tag": "button", "text": "Save   it", "rect": { "x": 128, "y": 80, "w": 256, "h": 40 } });
    let _ = br(&mut app, BrMsg::Answered(Input::Locate { x: 640.0, y: 400.0 }, Ok(located)));
    let mut ui = sim(&app);
    assert!(ui.find("button \"Save\" ref 7, button").is_ok() || ui.find("button \"Save\" ref 7, buttonSave it").is_ok(), "the element");
    assert!(ui.find("Add comment").is_ok());
    drop(ui);
    let _ = br(&mut app, BrMsg::NoteCancel);
    assert!(app.code().browser.draft.is_none());
    // an empty note adds nothing
    let _ = br(&mut app, BrMsg::NoteAdd);
    assert!(app.code().browser.comments.is_empty());
}

#[test]
fn a_comment_can_be_removed() {
    let (mut app, _dir) = app();
    live(&mut app);
    comment_on_something(&mut app, "one");
    comment_on_something(&mut app, "two");
    assert_eq!(app.code().browser.comments.len(), 2);
    let _ = br(&mut app, BrMsg::Del(0));
    assert_eq!(app.code().browser.comments.len(), 1);
    assert_eq!(app.code().browser.comments[0].note, "two");
}

#[test]
fn the_browser_tools_are_named_as_the_page_names_them() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    say(&mut app, "go");
    for (i, (name, args)) in [("browser_open", json!({ "url": "http://localhost:3000" })), ("browser_click", json!({ "ref": 12 }))].iter().enumerate() {
        let _ = code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: i, id: Some(format!("b{i}")), name: Some((*name).into()), arguments: args.to_string() }));
    }
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    let mut ui = sim(&app);
    assert!(ui.find("Open in browser").is_ok() && ui.find("http://localhost:3000").is_ok());
    assert!(ui.find("Browser click").is_ok() && ui.find("ref 12").is_ok());
}

#[test]
fn a_browser_screenshot_is_on_the_card_and_goes_to_a_model_that_sees() {
    let (mut app, _dir) = app_with(true);
    live(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    let shot = fllm_app::code::Shot::from_reply(&json!({ "data": PIXEL, "mime": "image/png", "width": 1280, "height": 800 })).unwrap();
    for round in 0..2 {
        say(&mut app, "look");
        let id = format!("s{round}");
        let _ = code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some(id.clone()), name: Some("browser_screenshot".into()), arguments: "{}".into() }));
        let _ = code(&mut app, CodeMsg::Stream(StreamEvent::Done));
        let _ = code(&mut app, CodeMsg::ToolReply(id, Ok("Screenshot taken.".into()), Some(shot.clone())));
        let _ = code(&mut app, CodeMsg::Stop);
    }
    let wire = app.code().wire(None);
    let pictures: Vec<_> = wire.iter().filter(|m| m.role == "user" && m.content.starts_with("[Screenshot of the browser page,") || m.content == "[an older screenshot of the browser was removed to save space]").collect();
    assert_eq!(pictures.len(), 2);
    assert_eq!(pictures[0].content, "[an older screenshot of the browser was removed to save space]");
    assert!(pictures[0].parts.is_none());
    assert_eq!(pictures[1].content, "[Screenshot of the browser page, taken by browser_screenshot]");
    assert!(pictures[1].parts.is_some(), "only the newest keeps its picture");
}

#[test]
fn a_model_that_cannot_see_gets_no_picture_of_the_browser() {
    let (mut app, _dir) = app_with(false);
    live(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    let shot = fllm_app::code::Shot::from_reply(&json!({ "data": PIXEL, "mime": "image/png", "width": 10, "height": 10 })).unwrap();
    say(&mut app, "look");
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("s0".into()), name: Some("browser_screenshot".into()), arguments: "{}".into() }));
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    let _ = code(&mut app, CodeMsg::ToolReply("s0".into(), Ok("Screenshot taken.".into()), Some(shot)));
    let wire = app.code().wire(None);
    assert!(!wire.iter().any(|m| m.content.starts_with("[Screenshot of the browser page,")));
}

#[test]
fn the_browser_tools_are_judged_by_the_page_the_browser_is_on() {
    let (mut app, _dir) = app();
    live(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    // a rule against the page the browser is on stops a click that names no address of its own
    say(&mut app, "/deny network: *localhost*");
    say(&mut app, "go");
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("c1".into()), name: Some("browser_click".into()), arguments: r#"{"ref":3}"#.into() }));
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::ToolCall { index: 1, id: Some("c2".into()), name: Some("browser_open".into()), arguments: r#"{"url":"https://example.com"}"#.into() }));
    let _ = code(&mut app, CodeMsg::Stream(StreamEvent::Done));
    let state = |id: &str| app.code().items.iter().find_map(|i| if let fllm_app::code::Item::Call(c) = i { (c.id == id).then(|| format!("{:?}", c.state)) } else { None }).unwrap_or_default();
    assert!(state("c1").contains("Blocked by a permission rule."), "{}", state("c1"));
    assert!(state("c2").contains("Running"), "an address of its own is judged by that address: {}", state("c2"));
}

#[test]
fn the_settings_have_the_three_cards_and_ask_before_the_browser_is_switched_on() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1300.0, 9000.0), app.view());
    assert!(ui.find("Browser for the coding agent").is_ok());
    assert!(ui.find("Let the agent use a browser").is_ok());
    assert!(ui.find("How it runs").is_ok());
    assert!(ui.find("Safety").is_ok());
    drop(ui);
    let _ = app.update(Message::Bp(fllm_app::browsercards::BpMsg::Enable(true)));
    assert!(app.bp.confirming);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1300.0, 1200.0), app.view());
    assert!(ui.find("Let the agent open web pages in a real browser on this computer? Each browser action still asks first. Pages are limited to http and https; local addresses follow the setting below.").is_ok());
    drop(ui);
    let _ = app.update(Message::Bp(fllm_app::browsercards::BpMsg::EnableCancel));
    assert!(!app.bp.confirming && !app.bp.enabled, "no means off");
    let task = app.update(Message::Bp(fllm_app::browsercards::BpMsg::Enable(false)));
    assert_eq!(task.units(), 1, "switching off is saved at once");
}

#[test]
fn saving_the_settings_makes_the_panel_read_again() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let done = app.update(Message::Bp(fllm_app::browsercards::BpMsg::Saved(fllm_app::browsercards::SaveResult::Ok(json!({ "settings": { "enabled": true } })))));
    assert!(done.units() >= 2, "the status and the tool list");
    assert_eq!(app.bp.message, "Saved.");
}

#[test]
fn the_panel_is_drawn() {
    let (mut app, _dir) = app();
    live(&mut app);
    comment_on_something(&mut app, "make it blue");
    let _ = br(&mut app, BrMsg::Comment);
    mouse(&mut app, 100.0, 100.0);
    let _ = br(&mut app, BrMsg::Click);
    let located = json!({ "ok": true, "ref": 3, "role": "link", "name": "Docs", "tag": "a", "text": "Docs", "rect": { "x": 100, "y": 100, "w": 200, "h": 60 } });
    let _ = br(&mut app, BrMsg::Answered(Input::Locate { x: 1.0, y: 1.0 }, Ok(located)));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 1000.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    assert!(image.matches_image(out.join("code-browser.png")).unwrap(), "the picture of the panel");
}
