//! Row 85, the look of the *Browser* panel: the page of a real Chromium (the picture in
//! `fixtures/browser-page.b64` is a frame the engine's `/api/studio/agent/browser/frame` gave),
//! the address pill, what the agent is doing, comment mode, the wide panel, and the empty states.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::browser::KeyPress;
use fllm_app::code::{client_tools, BrMsg, CodeMsg, Mode, Permission, Tool, Workspace};
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::{json, Value};

const PAGE: &str = include_str!("fixtures/browser-page.b64");

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let mut info = ModelInfo::new("qwen", 8192);
    info.vision = true;
    let _ = app.update(Message::Connected(Ok(info)));
    let _ = app.update(Message::Go(View::Code));
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None }))));
    let def = |n: &str| json!({ "type": "function", "function": { "name": n, "description": n, "parameters": { "type": "object", "properties": {} } } });
    let names = [("browser_open", Permission::Network), ("browser_click", Permission::Network)];
    let mut tools: Vec<Tool> = names.iter().map(|(n, p)| Tool::server(n, *p)).collect();
    let mut defs: Vec<Value> = names.iter().map(|(n, _)| def(n)).collect();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    let _ = app.update(Message::Code(CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs))))));
    (app, dir)
}

fn br(app: &mut App, m: BrMsg) -> iced::Task<Message> {
    app.update(Message::Code(CodeMsg::Br(m)))
}

fn status(running: bool, url: &str) -> Value {
    json!({ "browser": "/usr/bin/chromium", "running": running, "url": url, "profile": "/p", "settings": { "enabled": true, "show_window": false, "persistent_profile": true, "local_addresses": "auto", "executable": "" } })
}

/// The panel is open on the dashboard page of the fixture.
fn live(app: &mut App) {
    let _ = br(app, BrMsg::Toggle);
    let _ = br(app, BrMsg::Status { then_tick: false, answer: Some(status(true, "http://127.0.0.1:18091/")) });
    let epoch = app.code().br_epoch;
    let frame = json!({ "running": true, "image": format!("data:image/jpeg;base64,{}", PAGE.trim()), "key": "k1", "width": 1280, "height": 800, "url": "http://127.0.0.1:18091/", "title": "fLLM Studio – Dashboard" });
    let _ = br(app, BrMsg::TickDone { epoch, status: None, frame: Some(frame) });
}

fn the_agent_clicks(app: &mut App) {
    let _ = app.update(Message::Code(CodeMsg::SetMode(Mode::Auto)));
    let _ = app.update(Message::Code(CodeMsg::Edit(iced::widget::text_editor::Action::Edit(iced::widget::text_editor::Edit::Paste(std::sync::Arc::new("go".to_string()))))));
    let _ = app.update(Message::Code(CodeMsg::Send));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("b0".into()), name: Some("browser_click".into()), arguments: json!({ "ref": 7 }).to_string() })));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
}

fn picture(app: &App, size: (f32, f32), name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(size.0, size.1), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    assert!(image.matches_image(out.join(format!("{name}.png"))).unwrap(), "the picture {name}");
}

#[test]
fn the_page_of_a_real_browser_is_shown_in_the_panel() {
    let (mut app, _dir) = app("light");
    live(&mut app);
    assert!(app.code().browser.show_image(), "the fixture frame decodes");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 1000.0), app.view());
    assert!(ui.find("fLLM Studio – Dashboard").is_ok(), "the title under the picture");
    assert!(ui.find("127.0.0.1:18091").is_ok(), "and the host");
    assert!(ui.find("page 1280×800").is_ok());
    drop(ui);
    picture(&app, (1500.0, 1000.0), "browser-narrow-light");
}

#[test]
fn what_the_agent_does_stands_over_the_page_and_the_ring_is_lit() {
    let (mut app, _dir) = app("dark");
    live(&mut app);
    the_agent_clicks(&mut app);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 1000.0), app.view());
    assert!(ui.find("The agent: Browser click ref 7").is_ok());
    drop(ui);
    picture(&app, (1500.0, 1000.0), "browser-agent-dark");
}

#[test]
fn the_wide_panel_and_the_comments() {
    let (mut app, _dir) = app("light");
    live(&mut app);
    let _ = br(&mut app, BrMsg::Wide);
    assert!(app.code().br_wide);
    let _ = br(&mut app, BrMsg::Comment);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 1000.0), app.view());
    assert!(ui.find("Comment mode: click an element").is_ok());
    drop(ui);
    picture(&app, (1500.0, 1000.0), "browser-wide-comment-light");
    let _ = br(&mut app, BrMsg::Wide);
    assert!(!app.code().br_wide, "and back");
}

#[test]
fn nothing_open_yet_offers_the_usual_addresses() {
    let (mut app, _dir) = app("light");
    let _ = br(&mut app, BrMsg::Toggle);
    let _ = br(&mut app, BrMsg::Status { then_tick: false, answer: Some(status(true, "")) });
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 900.0), app.view());
    ui.click("localhost:5173").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Code(CodeMsg::Br(BrMsg::Quick(a)))] if a == "localhost:5173"), "{sent:?}");
    let _ = br(&mut app, BrMsg::Quick("localhost:5173".into()));
    assert_eq!(app.code().browser.url_input, "localhost:5173");
    picture(&app, (1500.0, 900.0), "browser-empty-light");
}

#[test]
fn ctrl_l_goes_to_the_address_box_when_the_panel_is_open() {
    let (mut app, _dir) = app("light");
    let key = KeyPress { key: "l".into(), ctrl: true, ..Default::default() };
    assert_eq!(app.update(Message::BrKey(key.clone())).units(), 0, "nothing while the panel is closed");
    let _ = br(&mut app, BrMsg::Toggle);
    assert!(app.update(Message::BrKey(key)).units() > 0, "an operation that focuses the box");
    let plain = KeyPress { key: "l".into(), ..Default::default() };
    assert_eq!(app.update(Message::BrKey(plain)).units(), 0, "a plain l is only a letter");
}

#[test]
fn a_call_that_waits_for_you_is_said_so_over_the_page() {
    let (mut app, _dir) = app("light");
    live(&mut app);
    // the mode that asks first: the call stands there until you decide
    let _ = app.update(Message::Code(CodeMsg::SetMode(Mode::Ask)));
    let _ = app.update(Message::Code(CodeMsg::Edit(iced::widget::text_editor::Action::Edit(iced::widget::text_editor::Edit::Paste(std::sync::Arc::new("go".to_string()))))));
    let _ = app.update(Message::Code(CodeMsg::Send));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::ToolCall { index: 0, id: Some("w0".into()), name: Some("browser_click".into()), arguments: json!({ "ref": 3 }).to_string() })));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 1000.0), app.view());
    assert!(ui.find("Waiting for you: Browser click ref 3").is_ok());
    assert!(ui.find("The agent: Browser click ref 3").is_err(), "not as if it ran");
}

#[test]
fn the_size_of_the_page_is_not_said_while_there_is_no_page() {
    let (mut app, _dir) = app("light");
    let _ = br(&mut app, BrMsg::Toggle);
    let _ = br(&mut app, BrMsg::Status { then_tick: false, answer: Some(status(true, "")) });
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1500.0, 900.0), app.view());
    assert!(ui.find("page 1280×800").is_err());
}
