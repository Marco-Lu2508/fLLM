//! The engine and server settings against a small server that answers like the Studio routes, and the page drawn with them.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::engine_settings::{self, parse_engine, parse_server};
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

type Log = Arc<Mutex<Vec<(String, String)>>>;

const ENGINE: &str = r#"{"values":{"state_cache":false,"kv_ram_mb":512},
 "defaults":{"output_cache":"exact","output_cache_mb":64,"cache_scope":"global","prefix_cache":true,"preemption":"recompute","kv_disk_dir":"","kv_ram_mb":0,"state_cache":true,"state_cache_mb":1024},
 "set_by_environment":["cache_scope"]}"#;
const SERVER: &str = r#"{"idle_unload_minutes":15,"load_on_start":true}"#;

async fn serve() -> (String, Log) {
    let log: Log = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16384];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let mut first = head.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
                seen.lock().unwrap().push((format!("{method} {path}"), body.to_string()));
                let (status, reply) = match (method, path) {
                    ("GET", "/api/studio/engine") => ("200 OK", ENGINE.to_string()),
                    ("GET", "/api/studio/settings") => ("200 OK", SERVER.to_string()),
                    ("POST", "/api/studio/engine") if body.contains("kv_disk_dir") => ("400 Bad Request", r#"{"error":{"message":"`kv_disk_dir` must be a full folder path."}}"#.to_string()),
                    ("POST", _) => ("200 OK", r#"{"status":"restarting"}"#.to_string()),
                    _ => ("404 Not Found", "{}".to_string()),
                };
                let _ = sock.write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_engine_options_and_the_server_settings_are_read_and_written_through_the_studio_routes() {
    let (base, log) = serve().await;
    let (engine, server) = engine_settings::fetch(base.clone()).await.unwrap();
    assert_eq!(engine.values["kv_ram_mb"], json!(512));
    assert_eq!(engine.locked, vec!["cache_scope"]);
    assert_eq!((server.idle_unload_minutes, server.load_on_start), (15, true));
    engine_settings::save_engine(base.clone(), json!({"state_cache": true})).await.unwrap();
    engine_settings::save_server(base.clone(), json!({"idle_unload_minutes": 30})).await.unwrap();
    engine_settings::restart(base.clone()).await.unwrap();
    let seen = log.lock().unwrap().clone();
    assert!(seen.contains(&("POST /api/studio/engine".into(), r#"{"state_cache":true}"#.into())), "{seen:?}");
    assert!(seen.contains(&("POST /api/studio/settings".into(), r#"{"idle_unload_minutes":30}"#.into())), "{seen:?}");
    assert!(seen.iter().any(|(r, _)| r == "POST /api/studio/restart"));
    let err = engine_settings::save_engine(base, json!({"kv_disk_dir": "relative"})).await.unwrap_err();
    assert_eq!(err, "`kv_disk_dir` must be a full folder path.", "the message of the server is what the user reads");
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("tiny", 4096))));
    let _ = app.update(Message::EngineLoaded(Ok((parse_engine(&serde_json::from_str::<Value>(ENGINE).unwrap()), parse_server(&serde_json::from_str::<Value>(SERVER).unwrap())))));
    let _ = app.update(Message::Go(View::Settings));
    (app, dir)
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

#[test]
fn a_changed_switch_is_shown_at_once_and_asks_for_a_restart() {
    let (mut app, _dir) = app("light");
    assert!(!app.engine().restart_needed);
    let _ = app.update(Message::EngineSet("state_cache".into(), json!(true)));
    assert_eq!(app.engine().value("state_cache"), json!(true));
    assert!(app.engine().restart_needed, "engine options apply after a restart");
    let _ = app.update(Message::EngineSet("cache_scope".into(), json!("off")));
    assert_eq!(app.engine().value("cache_scope"), json!("global"), "an option the environment fixes cannot be changed");
}

#[test]
fn a_number_is_checked_before_it_is_sent_and_a_wrong_one_stays_in_the_box() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::EngineDraft("state_cache_mb".into(), "0".into()));
    let _ = app.update(Message::EngineSubmit("state_cache_mb".into()));
    assert!(app.engine().error.as_deref().unwrap().contains("1 or more"));
    assert_eq!(app.engine().shown("state_cache_mb"), "0", "the wrong text stays to be corrected");
    assert_eq!(app.engine().value("state_cache_mb"), json!(1024), "nothing was changed");
    let _ = app.update(Message::EngineDraft("state_cache_mb".into(), "2048".into()));
    let _ = app.update(Message::EngineSubmit("state_cache_mb".into()));
    assert_eq!(app.engine().error, None);
    assert_eq!(app.engine().value("state_cache_mb"), json!(2048));
    assert_eq!(app.engine().shown("state_cache_mb"), "2048");
    // the idle time has its own range
    let _ = app.update(Message::EngineDraft("idle_unload_minutes".into(), "99999".into()));
    let _ = app.update(Message::EngineSubmit("idle_unload_minutes".into()));
    assert!(app.engine().error.as_deref().unwrap().contains("10080"));
    assert_eq!(app.engine().server.idle_unload_minutes, 15);
}

#[test]
fn a_refused_save_shows_the_message_and_the_page_goes_back_to_what_the_server_has() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::EngineSet("kv_ram_mb".into(), json!(9)));
    assert_eq!(app.engine().value("kv_ram_mb"), json!(9));
    let _ = app.update(Message::EngineSaved(Err("`kv_ram_mb` must be a whole number".into())));
    assert_eq!(app.engine().error.as_deref(), Some("`kv_ram_mb` must be a whole number"));
    // the follow-up read replaces the guess
    let (e, s) = (parse_engine(&serde_json::from_str::<Value>(ENGINE).unwrap()), parse_server(&serde_json::from_str::<Value>(SERVER).unwrap()));
    let _ = app.update(Message::EngineLoaded(Ok((e, s))));
    assert_eq!(app.engine().value("kv_ram_mb"), json!(512));
    assert_eq!(app.engine().error, None);
}

#[test]
fn restarting_waits_for_the_server_to_come_back() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::EngineSet("state_cache".into(), json!(true)));
    let _ = app.update(Message::RestartServer);
    assert!(app.engine().restarting);
    let _ = app.update(Message::RestartDone(Ok(())));
    assert!(!app.engine().restarting && !app.engine().restart_needed && !app.engine().loaded, "the settings are read again when the server is back");
    assert!(matches!(app.server_state(), fllm_app::app::Server::Connecting));
}

#[test]
fn the_page_draws_with_the_engine_cards_and_a_choice_sends_its_message() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app(theme);
        let _ = app.update(Message::EngineSet("state_cache".into(), json!(true))); // shows the restart bar too
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1700.0), app.view());
        let shot = ui.snapshot(&app.theme()).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("settings-engine-{theme}");
        for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
            let _ = std::fs::remove_file(old.path());
        }
        shot.matches_image(dir.join(format!("{name}.png"))).unwrap();
    }
    let (app, _dir) = app("light");
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1700.0), app.view());
    ui.click("Move to RAM").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::EngineSet(k, v) if k == "preemption" && v == "swap")));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1700.0), app.view());
    ui.find("Fixed by the environment of the server.").expect("the locked option says why it cannot be changed");
    ui.find("Reset engine options").expect("the reset button");
}
