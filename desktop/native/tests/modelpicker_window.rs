//! The dialog *Choose a model*: what it computes is pinned against the page's own code in
//! `modelpicker_fixtures`, the rest is the way through the dialog.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::models::Download;
use fllm_app::modelpicker::{clean_path_text, fact_rows, format_bytes, mtp_summary, parse_facts, parse_listing, urlencode, Facts, MpMsg};
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::{json, Value};

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Models));
    (app, dir)
}

fn mp(app: &mut App, m: MpMsg) {
    let _ = app.update(Message::Mp(m));
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1200.0, 2400.0), app.view())
}

fn found(kind: &str, path: &str) -> Facts {
    parse_facts(&json!({ "ok": true, "kind": kind, "path": path, "title": "Qwen3 8B", "format": "GGUF", "size_bytes": 4831838208u64, "files": 1,
        "architecture": "qwen3", "quantization": "Q4_K_M", "context_length": 40960, "fits": true, "mtp": false, "notes": ["Only the first part was found."] }))
}

fn checked(app: &mut App, path: &str, facts: Facts) {
    mp(app, MpMsg::Edit(path.into()));
    mp(app, MpMsg::Check);
    mp(app, MpMsg::Checked(path.into(), Ok(facts)));
}

#[test]
fn what_the_dialog_reads_is_what_the_page_reads() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/modelpicker_fixtures.json")).expect("the fixtures of the page");
    let data: Value = serde_json::from_str(&raw).unwrap();
    for case in data["facts"].as_array().unwrap() {
        let facts = if case["facts"].is_null() { None } else { Some(parse_facts(&case["facts"])) };
        assert_eq!(mtp_summary(facts.as_ref()), case["mtp"].as_str().unwrap(), "mtpSummary of {}", case["facts"]);
        let rows: Vec<(String, String)> = facts.as_ref().map(|f| fact_rows(f).into_iter().map(|(k, v)| (k.to_string(), v)).collect()).unwrap_or_default();
        let want: Vec<(String, String)> = case["rows"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_str().unwrap().to_string())).collect();
        assert_eq!(rows, want, "factRows of {}", case["facts"]);
    }
    for case in data["paths"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap_or("");
        let text = if text == "__undefined__" { "" } else { text };
        assert_eq!(clean_path_text(text), case["got"].as_str().unwrap(), "cleanPathText of {text:?}");
    }
    for case in data["bytes"].as_array().unwrap() {
        let bytes = match &case["bytes"] {
            Value::Number(n) => n.as_f64().unwrap(),
            Value::String(s) if s != "__undefined__" => s.parse().unwrap(),
            _ => 0.0,
        };
        assert_eq!(format_bytes(bytes), case["got"].as_str().unwrap(), "formatBytes of {}", case["bytes"]);
    }
}

#[test]
fn a_path_for_the_server_is_encoded_like_encode_uri_component() {
    assert_eq!(urlencode("C:\\Models\\a b"), "C%3A%5CModels%5Ca%20b");
    assert_eq!(urlencode("/home/ü"), "%2Fhome%2F%C3%BC");
    assert_eq!(urlencode("a-b_c.d~e!f*g'h(i)"), "a-b_c.d~e!f*g'h(i)");
}

#[test]
fn the_list_of_folders_is_read_the_way_the_server_sends_it() {
    let l = parse_listing(&json!({ "path": "/m", "parent": "/", "roots": [{ "label": "Home", "path": "/h" }], "truncated": true,
        "entries": [{ "kind": "folder", "name": "q", "path": "/m/q", "model_folder": true }, { "kind": "file", "name": "a.gguf", "path": "/m/a.gguf", "format": "GGUF", "size_bytes": 1048576 }] }));
    assert_eq!(l.parent.as_deref(), Some("/"));
    assert!(l.truncated && l.entries[0].folder && l.entries[0].model_folder && !l.entries[1].folder);
    assert_eq!(l.roots[0].label, "Home");
    assert!(parse_listing(&json!({ "path": "/", "parent": "" })).parent.is_none(), "the top has no way up");
}

#[test]
fn the_card_opens_the_dialog_and_the_buttons_wait_for_a_found_model() {
    let (mut app, _dir) = app();
    let mut ui = sim(&app);
    assert!(ui.find("Add a model").is_ok());
    assert!(ui.find("Choose…").is_ok());
    drop(ui);
    let _ = app.update(Message::LocalPath("/models/q.gguf".into()));
    let mut ui = sim(&app);
    ui.click("Choose…").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Mp(MpMsg::Open(p))] if p == "/models/q.gguf"), "{sent:?}");
    mp(&mut app, MpMsg::Open(String::new()));
    let mut ui = sim(&app);
    assert!(ui.find("Choose a model").is_ok());
    assert!(ui.find("Choose a file…").is_ok() && ui.find("Choose a folder…").is_ok() && ui.find("Look through folders").is_ok());
    // nothing found yet: Add and Load do not send anything
    ui.click("Add to my models").unwrap();
    ui.click("Load").unwrap();
    assert_eq!(ui.into_messages().count(), 0, "both wait for a model that was found");
}

#[test]
fn a_found_model_shows_its_facts_and_can_be_added_or_loaded() {
    let (mut app, _dir) = app();
    mp(&mut app, MpMsg::Open(String::new()));
    // typed with quotes: the box is cleaned when it is checked, and an answer for another text is dropped
    mp(&mut app, MpMsg::Edit("\"/models/q.gguf\"".into()));
    mp(&mut app, MpMsg::Check);
    assert_eq!(app.models().picker.path, "/models/q.gguf");
    mp(&mut app, MpMsg::Checked("/models/other.gguf".into(), Ok(found("file", "/models/other.gguf"))));
    assert!(app.models().picker.facts.is_none() && app.models().picker.busy, "an older check does not count");
    checked(&mut app, "/models/q.gguf", found("file", "/models/q.gguf"));
    let mut ui = sim(&app);
    for words in ["Qwen3 8B", "Format", "GGUF", "Size", "4.5 GB", "Architecture", "qwen3", "Quantization", "Q4_K_M", "Context length", "40,960 tokens", "Memory", "Fits into the free memory", "Only the first part was found."] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    assert!(ui.find("This model has no multi-token prediction heads (MTP), so that option cannot speed it up. Prompt lookup and a draft model work without them. Heads are part of the model files: a model converted without them cannot get them back.").is_ok());
    ui.click("Add to my models").unwrap();
    ui.click("Load").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Mp(MpMsg::Add), Message::Mp(MpMsg::Load)]), "{sent:?}");
    // loading adds it quietly first and then loads what the server named
    let after = app.update(Message::Mp(MpMsg::Load));
    assert!(after.units() >= 1);
    assert!(!app.models().picker.open, "the dialog closed for the load");
    let _ = app.update(Message::Mp(MpMsg::LoadAdded(Ok(Some("q.gguf".into())))));
    assert_eq!(app.models().busy.as_deref(), Some("q.gguf"), "the model is being loaded");
}

#[test]
fn a_hugging_face_name_is_downloaded_and_loaded_when_it_is_done() {
    let (mut app, _dir) = app();
    mp(&mut app, MpMsg::Open(String::new()));
    checked(&mut app, "owner/name", found("repo", "owner/name"));
    let mut ui = sim(&app);
    assert!(ui.find("Download and load").is_ok());
    ui.click("Add to my models").unwrap();
    assert_eq!(ui.into_messages().count(), 0, "a name is not a file that could be added");
    mp(&mut app, MpMsg::Pulled(Ok(7)));
    assert_eq!(app.toast(), Some("Added to the downloads"));
    assert_eq!(app.models().pending_load, Some(7));
    let d = |state: &str| Download { id: 7, model: "owner/name".into(), quant: String::new(), state: state.into(), written: 0, total: 0, mb_per_s: 0.0, eta_seconds: 0, error: "disk full".into(), reference: "owner--name".into() };
    let _ = app.update(Message::DownloadsLoaded(Ok(vec![d("running")])));
    assert_eq!(app.models().pending_load, Some(7), "still running");
    let _ = app.update(Message::DownloadsLoaded(Ok(vec![d("done")])));
    assert_eq!(app.models().pending_load, None);
    assert_eq!(app.models().busy.as_deref(), Some("owner--name"), "its reference is loaded");
    // a failed one tells why and stops waiting
    mp(&mut app, MpMsg::Pulled(Ok(7)));
    let _ = app.update(Message::DownloadsLoaded(Ok(vec![d("failed")])));
    assert_eq!(app.toast(), Some("disk full"));
    assert_eq!(app.models().pending_load, None);
    // a cancelled one is not waited for
    mp(&mut app, MpMsg::Pulled(Ok(7)));
    let _ = app.update(Message::DownloadsLoaded(Ok(vec![])));
    assert_eq!(app.models().pending_load, None);
}

#[test]
fn a_problem_is_shown_with_what_to_do_and_candidates_can_be_picked() {
    let (mut app, _dir) = app();
    mp(&mut app, MpMsg::Open(String::new()));
    let facts = parse_facts(&json!({ "ok": false, "kind": "not_a_model", "problem": "This folder has no model.", "fix": "Choose the folder with config.json.",
        "candidates": [{ "label": "/models/a", "path": "/models/a" }] }));
    checked(&mut app, "/models", facts);
    let mut ui = sim(&app);
    assert!(ui.find("This folder has no model.").is_ok());
    assert!(ui.find("Choose the folder with config.json.").is_ok());
    ui.click("/models/a").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Mp(MpMsg::CheckPath(p))] if p == "/models/a"), "{sent:?}");
    mp(&mut app, MpMsg::CheckPath("/models/a".into()));
    assert_eq!(app.models().picker.path, "/models/a");
    assert!(app.models().picker.busy, "the new path is being looked at");
    assert!(sim(&app).find("Looking at it ...").is_ok());
    // a server that is gone is said the way the page says it
    mp(&mut app, MpMsg::Checked("/models/a".into(), Err("no server".into())));
    let mut ui = sim(&app);
    assert!(ui.find("The model could not be checked: no server").is_ok());
    assert!(ui.find("The server may be restarting. Try again in a moment.").is_ok());
}

#[test]
fn the_list_of_folders_opens_walks_and_hands_a_folder_back() {
    let (mut app, _dir) = app();
    mp(&mut app, MpMsg::Open(String::new()));
    mp(&mut app, MpMsg::Browse);
    assert!(sim(&app).find("Reading the folder ...").is_ok());
    let l = parse_listing(&json!({ "path": "/models", "parent": "/", "roots": [{ "label": "Home", "path": "/home/m" }],
        "entries": [{ "kind": "folder", "name": "qwen", "path": "/models/qwen", "model_folder": true }, { "kind": "file", "name": "a.gguf", "path": "/models/a.gguf", "format": "GGUF", "size_bytes": 5242880 }] }));
    mp(&mut app, MpMsg::Listed(Ok(l)));
    let mut ui = sim(&app);
    for words in ["Home", "/models", "Up one folder", "qwen", "model folder", "a.gguf", "GGUF, 5.0 MB", "Use this folder"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    ui.click("qwen").unwrap();
    ui.click("a.gguf").unwrap();
    ui.click("Use this folder").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::Mp(MpMsg::Dir(d)), Message::Mp(MpMsg::CheckPath(f)), Message::Mp(MpMsg::UseFolder)] if d == "/models/qwen" && f == "/models/a.gguf"), "{sent:?}");
    mp(&mut app, MpMsg::UseFolder);
    assert_eq!(app.models().picker.path, "/models");
    // a folder that cannot be read says why
    mp(&mut app, MpMsg::Dir("/nope".into()));
    mp(&mut app, MpMsg::Listed(Err("Permission denied".into())));
    assert!(sim(&app).find("Permission denied").is_ok());
    // closing the list and the dialog
    mp(&mut app, MpMsg::Browse);
    assert!(sim(&app).find("Use this folder").is_err());
    mp(&mut app, MpMsg::Cancel);
    assert!(sim(&app).find("Choose a model").is_err());
}

#[test]
fn the_picture_of_the_dialog() {
    let (mut app, _dir) = app();
    mp(&mut app, MpMsg::Open(String::new()));
    checked(&mut app, "/models/q.gguf", found("file", "/models/q.gguf"));
    let l = parse_listing(&json!({ "path": "/models", "parent": "/", "roots": [{ "label": "Home", "path": "/home/m" }, { "label": "Models", "path": "/models" }],
        "entries": [{ "kind": "folder", "name": "qwen", "path": "/models/qwen", "model_folder": true }, { "kind": "file", "name": "q.gguf", "path": "/models/q.gguf", "format": "GGUF", "size_bytes": 4831838208u64 }] }));
    mp(&mut app, MpMsg::Browse);
    mp(&mut app, MpMsg::Listed(Ok(l)));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1100.0, 1100.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    let _ = std::fs::remove_file(out.join("model-picker-tiny-skia.png"));
    assert!(image.matches_image(out.join("model-picker.png")).unwrap());
}
