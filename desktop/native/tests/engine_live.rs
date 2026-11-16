//! Settings > Engine, the rest of the pane: the live numbers, the controls that act on the running
//! server (empty the cache, delete the blocks, forget the shared prompts, stop taking requests), and
//! the rows, cards and sentences of the page itself (`engine.js` runs in a vm for the fixtures).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::engine_settings::{self, all_fields, live_texts, parse_engine, parse_server, Ask, Live};
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/engine_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).unwrap()
}

#[test]
fn every_row_of_the_page_has_its_option_with_the_same_words() {
    let data = fixtures();
    let rows = data["rows"].as_array().unwrap();
    let mut seen = Vec::new();
    for row in rows {
        let title = row["title"].as_str().unwrap();
        match row["id"].as_str() {
            Some("engDrain") => assert_eq!((title, row["desc"].as_str().unwrap()), engine_settings::DRAIN),
            Some(id) => {
                let f = all_fields().find(|f| f.id == id).unwrap_or_else(|| panic!("the page's {id} ({title}) has no option here"));
                assert_eq!(f.title, title, "title of {id}");
                assert_eq!(f.help, row["desc"].as_str().unwrap(), "description of {id}");
                seen.push(id.to_string());
            }
            None => {
                assert_eq!((title, row["desc"].as_str().unwrap().replace("<b>", "").replace("</b>", "")), (engine_settings::ADMIN.0, engine_settings::ADMIN.1.to_string()));
            }
        }
    }
    let mut ours: Vec<String> = all_fields().map(|f| f.id.to_string()).collect();
    ours.sort();
    seen.sort();
    assert_eq!(ours, seen, "no option that the page does not have, none missing");
    // the order of the page inside a card
    let order: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
    let ours: Vec<&str> = all_fields().map(|f| f.id).collect();
    let order: Vec<&str> = order.into_iter().filter(|id| *id != "engDrain").collect();
    assert_eq!(ours, order, "the order of the page");
}

#[test]
fn the_cards_and_buttons_are_the_ones_of_the_page() {
    let data = fixtures();
    let cards: Vec<&str> = data["cards"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(cards, ["Answers that were given before", "Speed and memory", "Prompt blocks on disk", "Telemetry", "Administration"]);
    let buttons: Vec<(&str, &str)> = data["buttons"].as_array().unwrap().iter().map(|b| (b[0].as_str().unwrap(), b[1].as_str().unwrap())).collect();
    assert!(buttons.contains(&("engCacheClear", "Empty the cache now")) && buttons.contains(&("engKvClear", "Delete the blocks on disk")) && buttons.contains(&("engPrefixClear", "Forget the shared prompts in memory")));
    assert!(buttons.contains(&("engRestart", "Restart the server now")) && buttons.contains(&("engReset", "Back to the defaults")));
}

/// What the page's `live()` and `liveKv()` wrote for each scenario of answers.
#[test]
fn the_live_lines_are_what_the_page_writes() {
    let data = fixtures();
    for (name, sc) in data["scenarios"].as_object().unwrap() {
        let route = |p: &str| sc["routes"].get(p).filter(|v| v.is_object()).cloned();
        let kv = route("/admin/kv-tier");
        let live = Live { config: route("/admin/config"), prefix: if kv.is_some() { route("/admin/prefix-cache") } else { None }, kv };
        let got = live_texts(&live);
        assert_eq!(got.cache, sc["cacheStats"]["text"].as_str().unwrap(), "{name}: engCacheStats");
        assert_eq!(got.kv, sc["kvStats"]["text"].as_str().unwrap(), "{name}: engKvStats");
        assert_eq!(got.cache_controls, !sc["cacheClear"]["disabled"].as_bool().unwrap(), "{name}: engCacheClear");
        assert_eq!(got.cache_controls, !sc["drain"]["disabled"].as_bool().unwrap(), "{name}: engDrain disabled");
        assert_eq!(got.kv_controls, !sc["kvClear"]["disabled"].as_bool().unwrap(), "{name}: engKvClear");
        assert_eq!(got.kv_controls, !sc["prefixClear"]["disabled"].as_bool().unwrap(), "{name}: engPrefixClear");
        assert_eq!(got.draining, sc["drain"]["checked"].as_bool().unwrap(), "{name}: engDrain checked");
        // the telemetry line is only written when the config answered
        if got.cache_controls {
            assert_eq!(got.otel, sc["otelStats"]["text"].as_str().unwrap(), "{name}: engOtelStats");
        } else {
            assert_eq!(got.otel, "", "{name}");
        }
        let state = if sc["stateStats"].is_null() { "" } else { sc["stateStats"]["text"].as_str().unwrap() };
        assert_eq!(got.state, state, "{name}: engStateStats");
    }
    assert!(data["scenarios"].as_object().unwrap().len() >= 8);
}

type Log = Arc<Mutex<Vec<String>>>;

async fn serve(up: bool) -> (String, Log) {
    let log: Log = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = log.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let mut first = text.lines().next().unwrap_or("").split_whitespace();
                let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
                seen.lock().unwrap().push(format!("{method} {path}"));
                let (status, reply) = match (method.as_str(), path.as_str(), up) {
                    ("GET", "/admin/config", true) => ("200 OK", r#"{"output_cache":{"entries":2,"bytes":2048},"draining":false,"telemetry":{"enabled":false}}"#),
                    ("GET", "/admin/kv-tier", true) => ("200 OK", r#"{"enabled":true,"dir":"/d","used_bytes":1024,"max_bytes":2048}"#),
                    ("GET", "/admin/prefix-cache", true) => ("200 OK", r#"{"state_checkpoints":{"enabled":false}}"#),
                    ("DELETE", "/admin/cache", true) => ("200 OK", r#"{"cleared":4}"#),
                    ("DELETE", "/admin/kv-tier", true) => ("200 OK", r#"{"removed":7}"#),
                    ("DELETE", "/admin/prefix-cache", true) => ("200 OK", r#"{"dropped":2}"#),
                    ("POST", "/admin/drain", true) | ("DELETE", "/admin/drain", true) => ("200 OK", "{}"),
                    _ => ("404 Not Found", "{}"),
                };
                let _ = sock.write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await;
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_live_routes_are_read_and_the_controls_call_the_admin_routes() {
    let (base, log) = serve(true).await;
    let live = engine_settings::fetch_live(base.clone()).await;
    assert!(live.config.is_some() && live.kv.is_some() && live.prefix.is_some());
    let t = live_texts(&live);
    assert_eq!(t.cache, "now: 2 saved answers, 2.0 KB · answered from the cache 0 times, computed 0 times, 0 prompt tokens saved");
    assert_eq!(t.kv, "now: 1.0 KB of 2.0 KB on disk · 0 blocks read back, 0 written, 0 skipped");
    assert_eq!(t.state, "This model does not need copies of its state (or they are off).");
    assert_eq!(engine_settings::admin_delete(base.clone(), "/admin/cache", "cleared").await, Ok(4));
    assert_eq!(engine_settings::admin_delete(base.clone(), "/admin/kv-tier", "removed").await, Ok(7));
    assert_eq!(engine_settings::admin_delete(base.clone(), "/admin/prefix-cache", "dropped").await, Ok(2));
    assert_eq!(engine_settings::admin_drain(base.clone(), true).await, Ok(true));
    assert_eq!(engine_settings::admin_drain(base.clone(), false).await, Ok(false));
    let seen = log.lock().unwrap().clone();
    for want in ["POST /admin/drain", "DELETE /admin/drain", "DELETE /admin/cache", "DELETE /admin/kv-tier", "DELETE /admin/prefix-cache"] {
        assert!(seen.iter().any(|s| s == want), "{want} in {seen:?}");
    }
    // the copies of the state are asked for only when the disk tier answered
    let (down, log) = serve(false).await;
    let live = engine_settings::fetch_live(down.clone()).await;
    assert_eq!(live, Live::default());
    assert!(!log.lock().unwrap().iter().any(|s| s.contains("prefix-cache")), "no question for the copies without the disk tier");
    assert_eq!(engine_settings::admin_delete(down.clone(), "/admin/cache", "cleared").await, Err(String::new()));
    assert_eq!(engine_settings::admin_drain(down, true).await, Err(String::new()));
}

const ENGINE: &str = r#"{"values":{"state_cache":false,"kv_ram_mb":512},"defaults":{"output_cache":"exact","output_cache_mb":64,"output_cache_ttl":3600,"cache_scope":"global","prefix_cache":true,"prefix_salt":"","preemption":"recompute","kv_disk_dir":"","kv_disk_mb":2048,"kv_ram_mb":0,"state_cache":true,"state_cache_mb":1024,"otel_endpoint":"","otel_service":"fllm"},"set_by_environment":["cache_scope"]}"#;
const SERVER: &str = r#"{"idle_unload_minutes":15,"load_on_start":true}"#;

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("tiny", 4096))));
    let _ = app.update(Message::EngineLoaded(Ok((parse_engine(&serde_json::from_str::<Value>(ENGINE).unwrap()), parse_server(&serde_json::from_str::<Value>(SERVER).unwrap())))));
    let _ = app.update(Message::Go(View::Settings));
    // the whole page is taller than a texture: the search leaves the Engine alone
    let _ = app.update(Message::SettingsQuery("engine".into()));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 4090.0), app.view())
}

fn live() -> Live {
    Live {
        config: Some(json!({ "output_cache": { "entries": 12, "bytes": 1048576, "hits": 5, "misses": 7, "saved_prompt_tokens": 900 }, "coalescing": { "followers": 3 }, "telemetry": { "enabled": true, "sent": 20, "dropped": 1 }, "draining": true })),
        kv: Some(json!({ "enabled": true, "dir": "/fast", "used_bytes": 5242880, "max_bytes": 1073741824u64, "hits": 9, "spills": 8, "drops": 1 })),
        prefix: Some(json!({ "state_checkpoints": { "enabled": true, "entries": 3, "bytes": 314572800, "budget_bytes": 1073741824u64, "hits": 6, "tokens_skipped": 12000 } })),
    }
}

#[test]
fn the_pane_has_every_card_and_the_buttons_wait_for_a_running_server() {
    let (mut app, _dir) = app();
    // no numbers yet: the controls of the live server do nothing
    let mut ui = sim(&app);
    for words in ["Answers that were given before", "Speed and memory", "Prompt blocks on disk", "Telemetry", "Administration", "Remember identical answers", "Keep answers for", "Separate this installation", "Disk space for them", "Send traces to", "Name of this server", "Who may administer", "Stop taking new requests", "Back to the defaults"] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    ui.click("Empty the cache now").unwrap();
    ui.click("Delete the blocks on disk").unwrap();
    ui.click("Forget the shared prompts in memory").unwrap();
    assert_eq!(ui.into_messages().count(), 0, "a server without a model has no numbers and no controls");
    let _ = app.update(Message::EngineLive(live()));
    let mut ui = sim(&app);
    for words in [
        "now: 12 saved answers, 1.0 MB · answered from the cache 5 times, computed 7 times, 900 prompt tokens saved · identical requests shared: 3",
        "sending traces · 20 batches sent, 1 dropped",
        "now: 5.0 MB of 1.0 GB on disk · 9 blocks read back, 8 written, 1 skipped",
        "now: 3 copies, 300.0 MB of 1.0 GB · 6 messages continued from one, 12000 prompt tokens not read again",
    ] {
        assert!(ui.find(words).is_ok(), "{words}");
    }
    ui.click("Empty the cache now").unwrap();
    ui.click("Delete the blocks on disk").unwrap();
    ui.click("Forget the shared prompts in memory").unwrap();
    ui.click("Back to the defaults").unwrap();
    let sent: Vec<Message> = ui.into_messages().collect();
    assert!(matches!(sent.as_slice(), [Message::EngineCacheClear, Message::EngineAsk(Ask::KvClear), Message::EnginePrefixClear, Message::EngineAsk(Ask::Reset)]), "{sent:?}");
    assert!(app.engine().live.draining, "the switch shows what the server reported");
}

#[test]
fn deleting_the_blocks_and_the_reset_ask_first() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::EngineLive(live()));
    let _ = app.update(Message::EngineAsk(Ask::KvClear));
    let mut ui = sim(&app);
    assert!(ui.find("Delete every prompt block that is saved on disk?").is_ok());
    ui.click("OK").unwrap();
    assert!(matches!(ui.into_messages().next(), Some(Message::EngineAskOk)));
    let _ = app.update(Message::EngineAskCancel);
    assert!(app.engine().ask.is_none());
    assert!(sim(&app).find("Delete every prompt block that is saved on disk?").is_err());
    // the reset puts every default back, but only after OK
    let _ = app.update(Message::EngineAsk(Ask::Reset));
    assert!(sim(&app).find("Set every option on this page back to its default?").is_ok());
    assert_eq!(app.engine().value("kv_ram_mb"), json!(512), "nothing changed before OK");
    let _ = app.update(Message::EngineAskOk);
    assert!(app.engine().ask.is_none());
    assert_eq!(app.engine().value("kv_ram_mb"), json!(0));
    assert_eq!(app.engine().value("state_cache"), json!(true));
    assert_eq!(app.engine().value("cache_scope"), json!("global"), "the locked option stays");
    // the numbers' answers become words for the toast
    let _ = app.update(Message::EngineCacheCleared(Ok(4)));
    assert_eq!(app.toast(), Some("4 saved answers removed."));
    let _ = app.update(Message::EngineCacheCleared(Err(String::new())));
    assert_eq!(app.toast(), Some("The cache could not be emptied."));
    let _ = app.update(Message::EngineKvCleared(Ok(7)));
    assert_eq!(app.toast(), Some("7 blocks deleted."));
    let _ = app.update(Message::EngineKvCleared(Err(String::new())));
    assert_eq!(app.toast(), Some("The blocks could not be deleted."));
    let _ = app.update(Message::EnginePrefixCleared(Ok(2)));
    assert_eq!(app.toast(), Some("2 shared prompts forgotten."));
}

#[test]
fn the_drain_switch_goes_back_when_the_server_refuses() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::EngineLive(Live { config: Some(json!({ "draining": false })), ..Default::default() }));
    assert!(!app.engine().live.draining);
    let _ = app.update(Message::EngineDrain(true));
    assert!(app.engine().live.draining, "shown at once");
    let _ = app.update(Message::EngineDrained(Err(String::new())));
    assert!(!app.engine().live.draining, "back to what the server has");
    assert_eq!(app.toast(), Some("The server did not accept that."));
    let _ = app.update(Message::EngineDrained(Ok(true)));
    assert_eq!(app.toast(), Some("The server takes no new requests."));
    let _ = app.update(Message::EngineDrained(Ok(false)));
    assert_eq!(app.toast(), Some("The server takes requests again."));
}

#[test]
fn the_variables_of_the_environment_are_named_like_the_page_names_them() {
    let (app, _dir) = app();
    assert!(sim(&app).find("Set by the environment of the server and not changeable here: cache_scope.").is_ok());
}

#[test]
fn the_pane_is_drawn() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::EngineLive(live()));
    let _ = app.update(Message::EngineSet("state_cache".into(), json!(true)));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1200.0, 4000.0), app.view());
    let image = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    let _ = std::fs::remove_file(out.join("settings-engine-live-tiny-skia.png"));
    assert!(image.matches_image(out.join("settings-engine-live.png")).unwrap());
}
