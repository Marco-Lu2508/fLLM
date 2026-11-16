//! Row 86: computer use in the Code page — the banner with its STOP, the permission cards that ask
//! before every action, the screenshots on the cards and in the conversation, the emergency stop.
//! The pure part is pinned against the page in `tests/computer_fixtures.rs`.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, CuStatus, Mode, Permission, Shot, Tool, Workspace};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::Arc;

/// A picture of one pixel: what the server's `image.data` is, in small.
const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let mut info = ModelInfo::new("qwen", 8192);
    info.vision = true;
    let _ = app.update(Message::Connected(Ok(info)));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn code(app: &mut App, m: CodeMsg) -> iced::Task<Message> {
    app.update(Message::Code(m))
}

fn open_folder(app: &mut App) {
    let _ = code(app, CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None })));
    let def = |n: &str| json!({ "type": "function", "function": { "name": n, "description": n, "parameters": { "type": "object", "properties": {} } } });
    let names = ["computer_screenshot", "computer_click", "computer_type", "computer_key", "computer_wait"];
    let mut tools: Vec<Tool> = names.iter().map(|n| Tool::server(n, Permission::Computer)).collect();
    let mut defs: Vec<Value> = names.iter().map(|n| def(n)).collect();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    let _ = code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
}

fn status(app: &mut App, on: bool, steps: u64) {
    let epoch = app.code().computer.epoch;
    let _ = code(app, CodeMsg::ComputerStatus(epoch, Some(CuStatus { enabled: on, available: true, steps_used: steps, max_steps: 100 })));
}

fn say(app: &mut App, text: &str) {
    let _ = code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    let _ = code(app, CodeMsg::Send);
    // with computer use on, the run waits for the answer of `computer/begin` (a server of its own is not here)
    if app.code().beginning {
        let _ = code(app, CodeMsg::ComputerBegun { prompt: text.to_string(), display: text.to_string() });
    }
}

fn model_calls(app: &mut App, calls: &[(&str, &str, Value)]) {
    for (i, (id, name, args)) in calls.iter().enumerate() {
        let _ = code(app, CodeMsg::Stream(StreamEvent::ToolCall { index: i, id: Some((*id).into()), name: Some((*name).into()), arguments: args.to_string() }));
    }
    let _ = code(app, CodeMsg::Stream(StreamEvent::Done));
}

fn shot() -> Shot {
    Shot::from_reply(&json!({ "data": PIXEL, "mime": "image/png", "width": 1280, "height": 800 })).expect("a picture")
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 1800.0), app.view())
}

fn state_of(app: &App, id: &str) -> String {
    app.code().items.iter().find_map(|i| if let fllm_app::code::Item::Call(c) = i { (c.id == id).then(|| format!("{:?}", c.state)) } else { None }).unwrap_or_default()
}

fn wire_of(app: &App) -> Vec<fllm_app::api::WireMessage> {
    app.code().wire(None)
}

#[test]
fn the_banner_is_there_only_while_computer_use_is_on() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    assert!(sim(&app).find("STOP").is_err(), "no banner before the server says it is on");
    status(&mut app, true, 0);
    let mut ui = sim(&app);
    assert!(ui.find("Computer use is ON.").is_ok());
    assert!(ui.find("STOP").is_ok());
    assert!(ui.find("Emergency stop: press Esc three times, or Ctrl+Alt+Shift+Esc anywhere.").is_ok());
    drop(ui);
    status(&mut app, true, 7);
    assert!(sim(&app).find("Emergency stop: press Esc three times, or Ctrl+Alt+Shift+Esc anywhere. 7 of 100 actions used in this task.").is_ok(), "the actions used");
    status(&mut app, false, 0);
    assert!(sim(&app).find("STOP").is_err());
}

#[test]
fn a_server_that_says_enabled_but_not_available_is_off() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let epoch = app.code().computer.epoch;
    let _ = code(&mut app, CodeMsg::ComputerStatus(epoch, Some(CuStatus { enabled: true, available: false, steps_used: 0, max_steps: 0 })));
    assert!(!app.code().computer_on());
    let _ = code(&mut app, CodeMsg::ComputerStatus(epoch, None));
    assert!(!app.code().computer_on(), "a server that does not answer is off");
}

#[test]
fn an_older_chain_of_polls_is_ignored() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let epoch = app.code().computer.epoch;
    let _ = code(&mut app, CodeMsg::ComputerStatus(epoch + 5, Some(CuStatus { enabled: true, available: true, steps_used: 0, max_steps: 1 })));
    assert!(!app.code().computer_on());
}

#[test]
fn the_tool_list_follows_the_setting() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    // the list as it stands has the tools, so switching on changes nothing; switching off asks the server again
    let epoch = app.code().computer.epoch;
    let on = CuStatus { enabled: true, available: true, steps_used: 0, max_steps: 100 };
    let first = code(&mut app, CodeMsg::ComputerStatus(epoch, Some(on)));
    assert_eq!(first.units(), 1, "only the next poll");
    let off = code(&mut app, CodeMsg::ComputerStatus(epoch, Some(CuStatus::default())));
    assert_eq!(off.units(), 2, "the next poll and the tool list again");
}

#[test]
fn switched_off_behind_the_back_stops_the_run_and_says_so() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "look");
    assert!(app.code().running);
    status(&mut app, false, 0);
    assert!(!app.code().running, "the run ends");
    assert!(app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::Note(n) if n == "Computer use was stopped and switched off. Switch it on again in Settings > Runtime if you want to continue.")));
}

#[test]
fn operating_asks_every_time_and_looking_follows_the_mode() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    status(&mut app, true, 0);
    say(&mut app, "do it");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    assert!(state_of(&app, "s1").contains("Running"), "a screenshot in the mode that allows everything just runs");
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("1280x800".into()), Some(shot())));
    say(&mut app, "click");
    model_calls(&mut app, &[("k1", "computer_click", json!({ "x": 10, "y": 20 }))]);
    assert!(state_of(&app, "k1").contains("Pending"), "a click asks even where the mode allows everything");
    // and its words
    let mut ui = sim(&app);
    assert!(ui.find("Click at the marked point?").is_ok(), "the question of the card");
    assert!(ui.find("Waiting for your answer: Click 10, 20").is_ok(), "the banner says what it waits for");
    assert!(ui.find("Allow").is_ok());
    assert!(ui.find("Always").is_err(), "an action that operates cannot be remembered");
}

#[test]
fn the_screenshot_card_may_be_remembered_for_the_session() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "look");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    let mut ui = sim(&app);
    assert!(ui.find("Let the agent look at your screen?").is_ok());
    assert!(ui.find("Always").is_ok(), "only the screenshot has the button");
    drop(ui);
    let _ = code(&mut app, CodeMsg::Always("s1".into(), "computer:look".into()));
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("ok".into()), Some(shot())));
    say(&mut app, "again");
    model_calls(&mut app, &[("s2", "computer_screenshot", json!({}))]);
    assert!(state_of(&app, "s2").contains("Running"), "the next look needs no question");
    model_calls(&mut app, &[("c2", "computer_cursor", json!({}))]);
}

#[test]
fn an_action_before_the_first_screenshot_is_an_error() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "click");
    model_calls(&mut app, &[("k1", "computer_click", json!({ "x": 1, "y": 2 }))]);
    assert!(
        state_of(&app, "k1").contains("Take a screenshot first (computer_screenshot): coordinates refer to the last screenshot you took."),
        "{}",
        state_of(&app, "k1")
    );
    // and the model is told so
    let wire = wire_of(&app);
    assert!(wire.iter().any(|m| m.role == "tool" && m.content.starts_with("Error: Take a screenshot first")), "{wire:?}");
}

#[test]
fn a_screenshot_is_on_the_card_and_goes_to_the_model_with_the_reply() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "look");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    let _ = code(&mut app, CodeMsg::Always("s1".into(), "computer:look".into()));
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("Screen 1280x800".into()), Some(shot())));
    assert!(app.code().computer.last_shot.is_some(), "the coordinates of the next action refer to it");
    let wire = wire_of(&app);
    let m = wire.last().expect("a message");
    assert_eq!(m.role, "user");
    assert_eq!(m.content, "[Screenshot from the computer tools: 1280x800 px. Screen 1280x800]");
    let parts = m.parts.as_ref().expect("with the picture");
    assert_eq!(parts[0], json!({ "type": "text", "text": m.content }));
    assert_eq!(parts[1]["type"], "image_url");
    assert!(parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,iVBOR"));
}

#[test]
fn only_the_last_two_screenshots_keep_their_pictures() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    status(&mut app, true, 0);
    for i in 0..3 {
        say(&mut app, "look");
        let id = format!("s{i}");
        model_calls(&mut app, &[(&id, "computer_screenshot", json!({}))]);
        let _ = code(&mut app, CodeMsg::ToolReply(id.clone(), Ok(format!("shot {i}")), Some(shot())));
        let _ = code(&mut app, CodeMsg::Stop);
    }
    let wire = wire_of(&app);
    let shots: Vec<_> = wire.iter().filter(|m| m.role == "user" && m.content.starts_with("[Screenshot")).collect();
    assert_eq!(shots.len(), 3);
    assert!(shots[0].parts.is_none(), "the oldest has lost its picture");
    assert!(shots[0].content.ends_with("[The picture was removed from the conversation to save space. Take a new screenshot if you need to see the screen.]"), "{}", shots[0].content);
    assert!(shots[1].parts.is_some() && shots[2].parts.is_some());
}

#[test]
fn the_card_draws_the_screen_and_the_mark_of_the_action() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    status(&mut app, true, 0);
    say(&mut app, "go");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("ok".into()), Some(shot())));
    say(&mut app, "click");
    model_calls(&mut app, &[("k1", "computer_click", json!({ "x": 640, "y": 400 }))]);
    let card = app.code().items.iter().find_map(|i| if let fllm_app::code::Item::Call(c) = i { (c.id == "k1").then(|| c.clone()) } else { None }).unwrap();
    let cu = card.cu.as_ref().expect("the card keeps the screenshot the click refers to");
    assert!(cu.shot.is_some() && cu.image.is_none());
    let mut ui = sim(&app);
    assert!(ui.find("Click").is_ok() && ui.find("640, 400").is_ok(), "the card is named as the page names it");
}

#[test]
fn stop_ends_the_run_and_switches_the_tools_off() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "look");
    let _ = code(&mut app, CodeMsg::ToolReply("none".into(), Ok(String::new()), Some(shot())));
    let task = code(&mut app, CodeMsg::ComputerStop("You pressed STOP.".into()));
    assert_eq!(task.units(), 1, "the server is told");
    assert!(app.code().computer.stopping);
    let _ = code(&mut app, CodeMsg::ComputerStopped("You pressed STOP.".into()));
    assert!(!app.code().running);
    assert!(app.code().computer.last_shot.is_none());
    assert!(app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::Note(n) if n == "You pressed STOP. Computer use is switched off; the agent cannot see or touch the screen any more. Switch it on again in Settings > Runtime.")));
    assert_eq!(app.toast.as_deref(), Some("Computer use stopped and switched off"));
}

#[test]
fn a_stop_that_the_window_itself_asked_for_is_no_surprise() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    let _ = code(&mut app, CodeMsg::ComputerStop("You pressed STOP.".into()));
    status(&mut app, false, 0);
    assert!(!app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::Note(n) if n.starts_with("Computer use was stopped and switched off."))), "no second notice");
    assert!(!app.code().computer.stopping);
}

#[test]
fn escape_three_times_stops_and_one_or_two_do_not() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    // off: the key is nothing special
    for _ in 0..3 {
        assert_eq!(app.update(Message::Esc).units(), 0);
    }
    status(&mut app, true, 0);
    assert_eq!(app.update(Message::Esc).units(), 0);
    assert_eq!(app.update(Message::Esc).units(), 0);
    let third = app.update(Message::Esc);
    assert_eq!(third.units(), 1, "the third press tells the server");
    assert!(app.code().computer.stopping);
}

#[test]
fn a_held_escape_does_not_count_and_the_chord_stops_at_once() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    status(&mut app, true, 0);
    for _ in 0..5 {
        assert_eq!(app.update(Message::EscHeld).units(), 0, "a repeat is no press");
    }
    assert!(!app.code().computer.stopping);
    let chord = app.update(Message::Panic);
    assert_eq!(chord.units(), 1);
    assert!(app.code().computer.stopping);
}

#[test]
fn a_model_that_cannot_see_is_told_once() {
    let (mut app, _dir) = app();
    let mut blind = ModelInfo::new("qwen", 8192);
    blind.vision = false;
    let _ = app.update(Message::Connected(Ok(blind)));
    open_folder(&mut app);
    status(&mut app, true, 0);
    for i in 0..2 {
        say(&mut app, "look");
        let id = format!("s{i}");
        model_calls(&mut app, &[(&id, "computer_screenshot", json!({}))]);
        let _ = code(&mut app, CodeMsg::Always(id.clone(), "computer:look".into()));
        let _ = code(&mut app, CodeMsg::ToolReply(id, Ok("ok".into()), Some(shot())));
        let _ = code(&mut app, CodeMsg::Stop);
    }
    let notes = app.code().items.iter().filter(|i| matches!(i, fllm_app::code::Item::Note(n) if n.starts_with("The loaded model cannot look at pictures"))).count();
    assert_eq!(notes, 1);
}

#[test]
fn a_model_of_a_provider_is_not_warned_about_pictures() {
    // the page skips the warning for a remote model (`fllmRemoteSelected`): nothing is known about what it can see
    let (mut app, _dir) = app();
    let mut remote = ModelInfo::new("remote:openai/gpt-4o", 8192);
    remote.vision = false;
    let _ = app.update(Message::Connected(Ok(remote)));
    open_folder(&mut app);
    status(&mut app, true, 0);
    say(&mut app, "look");
    model_calls(&mut app, &[("s0", "computer_screenshot", json!({}))]);
    let _ = code(&mut app, CodeMsg::Always("s0".into(), "computer:look".into()));
    let _ = code(&mut app, CodeMsg::ToolReply("s0".into(), Ok("ok".into()), Some(shot())));
    assert!(!app.code().items.iter().any(|i| matches!(i, fllm_app::code::Item::Note(n) if n.starts_with("The loaded model cannot look at pictures"))));
    assert!(!app.code().computer.warned_vision);
}

#[test]
fn a_saved_session_does_not_keep_the_pictures() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    status(&mut app, true, 0);
    say(&mut app, "look");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("ok".into()), Some(shot())));
    let saved = app.code().saved();
    let text = serde_json::to_string(&saved).unwrap();
    assert!(!text.contains(PIXEL), "no picture on the disk");
    assert!(text.contains("\"gone\":true"), "but the card knows it had one");
    // restored, the card knows it had a picture and says so
    let mut restored = fllm_app::code::CodeState::default();
    restored.restore(saved);
    let card = restored.items.iter().find_map(|i| if let fllm_app::code::Item::Call(c) = i { Some(c.clone()) } else { None }).expect("the call");
    assert!(card.cu.as_ref().is_some_and(|v| v.gone && v.image.is_none() && v.shot.is_none()));
}

#[test]
fn the_computer_tools_are_for_the_model_only_while_it_is_on() {
    let on = fllm_app::code::parse_tools_with(&json!({ "tools": [
        { "function": { "name": "computer_click" }, "permission": "computer" },
        { "function": { "name": "read_file" }, "permission": "read" } ] }), true);
    assert_eq!(on.0.len(), 2);
    let off = fllm_app::code::parse_tools(&json!({ "tools": [
        { "function": { "name": "computer_click" }, "permission": "computer" },
        { "function": { "name": "read_file" }, "permission": "read" } ] }));
    assert_eq!(off.0.len(), 1);
}

#[test]
fn the_banner_is_drawn() {
    let (mut app, _dir) = app();
    open_folder(&mut app);
    let _ = code(&mut app, CodeMsg::SetMode(Mode::Auto));
    status(&mut app, true, 3);
    say(&mut app, "go");
    model_calls(&mut app, &[("s1", "computer_screenshot", json!({}))]);
    let _ = code(&mut app, CodeMsg::ToolReply("s1".into(), Ok("ok".into()), Some(shot())));
    say(&mut app, "click");
    model_calls(&mut app, &[("k1", "computer_click", json!({ "x": 640, "y": 400, "double": true }))]);
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    assert!(image.matches_image(out.join("code-computer.png")).unwrap(), "the picture of the banner");
}
