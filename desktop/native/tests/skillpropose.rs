//! `/skill`: the model writes a skill from the session and you decide. What the page computes is
//! pinned against its own code (`skills.js` runs in a vm for `skillpropose_fixtures`).

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CodeMsg, Item, SessMsg, Workspace};
use fllm_app::skills::{parse_proposal, proposal_prompt, transcript, SessionMessage};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::Value;
use std::sync::Arc;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/skillpropose_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).unwrap()
}

fn messages_of(v: &Value) -> Vec<SessionMessage> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| SessionMessage {
            role: m["role"].as_str().unwrap_or("").to_string(),
            content: m["content"].as_str().unwrap_or("").to_string(),
            calls: m["tool_calls"].as_array().map(|a| a.iter().map(|c| (c.pointer("/function/name").and_then(Value::as_str).unwrap_or("undefined").to_string(), c.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("").to_string())).collect()).unwrap_or_default(),
        })
        .collect()
}

#[test]
fn the_transcript_is_the_one_of_the_page() {
    let data = fixtures();
    for case in data["transcripts"].as_array().unwrap() {
        let got = transcript(&messages_of(&case["messages"]), 14000);
        assert_eq!(got, case["got"].as_str().unwrap(), "skTranscript of {}", case["name"]);
    }
    for case in data["small_cap"].as_array().unwrap() {
        let got = transcript(&messages_of(&case["messages"]), case["max"].as_u64().unwrap() as usize);
        assert_eq!(got, case["got"].as_str().unwrap(), "skTranscript of {} cut at {}", case["name"], case["max"]);
    }
    assert!(data["transcripts"].as_array().unwrap().len() >= 8);
}

#[test]
fn the_request_and_the_reading_of_the_reply_are_the_ones_of_the_page() {
    let data = fixtures();
    for case in data["prompts"].as_array().unwrap() {
        assert_eq!(proposal_prompt(case["transcript"].as_str().unwrap()), case["got"].as_str().unwrap());
    }
    for case in data["replies"].as_array().unwrap() {
        let want = &case["got"];
        match parse_proposal(case["reply"].as_str().unwrap()) {
            Ok(p) => {
                assert_eq!((p.name.as_str(), p.content.as_str()), (want["name"].as_str().expect("the page found a skill"), want["content"].as_str().unwrap()), "skParseProposal of {}", case["name"]);
            }
            Err(e) => assert_eq!(e, want["error"].as_str().expect("the page found no skill"), "skParseProposal of {}", case["name"]),
        }
    }
    assert!(data["replies"].as_array().unwrap().len() >= 18);
    assert_eq!(parse_proposal("").unwrap_err(), data["null_reply"]["error"].as_str().unwrap(), "a reply that is not there");
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    let _ = app.update(Message::Code(CodeMsg::Opened(Ok(Workspace { root: "/w".into(), name: "w".into(), branch: None }))));
    let (tools, defs) = client_tools();
    let _ = app.update(Message::Code(CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs))))));
    (app, dir)
}

fn say(app: &mut App, text: &str) -> iced::Task<Message> {
    let _ = app.update(Message::Code(CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string()))))));
    app.update(Message::Code(CodeMsg::Send))
}

fn sess(app: &mut App, m: SessMsg) {
    let _ = app.update(Message::Code(CodeMsg::Sess(m)));
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 3000.0), app.view())
}

const GOOD: &str = "---\nname: run-tests\ndescription: Runs the tests of the project\nwhen: after a change\n---\n# Run the tests\n\n1. npm test\n";

#[test]
fn nothing_to_learn_from_says_nothing_like_the_page() {
    let (mut app, _dir) = app();
    let task = say(&mut app, "/skill");
    assert_eq!(task.units(), 0, "no request without a session");
    assert!(app.code().skill.is_none());
    assert!(!app.code().skill_asking);
}

#[test]
fn the_model_writes_a_skill_and_it_is_saved_only_when_you_agree() {
    let (mut app, _dir) = app();
    let _ = say(&mut app, "make the tests pass");
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Delta("done, npm test is green".into()))));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
    let task = say(&mut app, "/skill");
    assert!(task.units() >= 1, "the request goes out");
    assert!(app.code().skill_asking);
    // a second /skill while one is being written does nothing
    assert_eq!(say(&mut app, "/skill").units(), 0);
    sess(&mut app, SessMsg::SkillWritten(Ok(GOOD.into())));
    assert!(!app.code().skill_asking);
    let draft = app.code().skill.as_ref().expect("the dialog is open");
    assert_eq!(draft.name, "run-tests");
    assert!(!draft.workspace, "for every project to begin with");
    let mut ui = sim(&app);
    for words in ["Save this as a skill?", "The model wrote this skill from the session. Edit it if needed; it is only saved when you press Save.", "For every project", "This folder only", "Discard", "Save skill"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    drop(ui);
    // nothing was saved by now; a bad name is refused before any request
    sess(&mut app, SessMsg::SkillName("Run Tests".into()));
    sess(&mut app, SessMsg::SkillScope(true));
    assert!(app.code().skill.as_ref().unwrap().workspace);
    sess(&mut app, SessMsg::SkillSave);
    assert!(!app.code().skill.as_ref().unwrap().busy, "the name is wrong");
    assert_eq!(app.toast(), Some("The name uses lowercase letters, digits, - and _."));
    sess(&mut app, SessMsg::SkillName("run-tests".into()));
    sess(&mut app, SessMsg::SkillSave);
    assert!(app.code().skill.as_ref().unwrap().busy, "the save is under way");
    // a refusal keeps the dialog and says why
    sess(&mut app, SessMsg::SkillSaved(Err("a skill of that name exists".into())));
    assert!(app.code().skill.is_some() && !app.code().skill.as_ref().unwrap().busy);
    assert_eq!(app.toast(), Some("a skill of that name exists"));
    sess(&mut app, SessMsg::SkillSaved(Ok(())));
    assert!(app.code().skill.is_none());
    assert_eq!(app.toast(), Some("Skill \"run-tests\" saved"));
    assert!(app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Skill saved.")));
}

#[test]
fn a_reply_without_a_skill_says_why_and_discard_leaves_nothing() {
    let (mut app, _dir) = app();
    let _ = say(&mut app, "hello");
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Delta("hi".into()))));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
    let _ = say(&mut app, "/skill");
    sess(&mut app, SessMsg::SkillWritten(Ok("NONE".into())));
    assert!(app.code().skill.is_none());
    assert_eq!(app.toast(), Some("The model found nothing reusable in this session."));
    let _ = say(&mut app, "/skill");
    sess(&mut app, SessMsg::SkillWritten(Err("The server answered with error 500".into())));
    assert_eq!(app.toast(), Some("Could not write a skill: The server answered with error 500"));
    assert!(!app.code().skill_asking);
    let _ = say(&mut app, "/skill");
    sess(&mut app, SessMsg::SkillWritten(Ok(GOOD.into())));
    sess(&mut app, SessMsg::SkillDiscard);
    assert!(app.code().skill.is_none());
    assert!(!app.code().items.iter().any(|i| matches!(i, Item::Note(n) if n == "Skill saved.")));
}

#[test]
fn the_picture_of_the_dialog() {
    let (mut app, _dir) = app();
    let _ = say(&mut app, "make the tests pass");
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Delta("done".into()))));
    let _ = app.update(Message::Code(CodeMsg::Stream(StreamEvent::Done)));
    sess(&mut app, SessMsg::SkillWritten(Ok(GOOD.into())));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1200.0, 1000.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    let _ = std::fs::remove_file(out.join("code-skill-proposal-tiny-skia.png"));
    assert!(image.matches_image(out.join("code-skill-proposal.png")).unwrap());
}
