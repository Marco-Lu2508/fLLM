//! The Models page against a small server of our own that answers like the fLLM Studio routes, and the window with that page drawn headless.

use fllm_app::api::{fetch_model, ModelInfo};
use fllm_app::app::{App, Message, Server, View};
use fllm_app::models::{self, NO_MODEL};
use iced_test::Simulator;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// What the server was asked: `(method, path, body)` in order.
type Log = Arc<Mutex<Vec<(String, String, String)>>>;

const OVERVIEW: &str = r#"{"available_memory_bytes":8309764096,
 "loaded":{"name":"qwen2.5-0.5b-instruct","parameter_size":"630M","quantization":"Q4_0"},
 "local":[{"name":"my-model.gguf","path":"/models/my-model.gguf","size_bytes":2000000000}],
 "cached":[{"name":"qwen3.5:4b","path":"/cache/q.gguf","size_bytes":2800000000}],
 "catalog":[{"id":"aya-expanse:8b","recommended_quant":"Q4_K_M","parameters_billion":8.0,"context_length":32768,"fits":true,
   "variants":[{"quant":"Q8_0","estimated_bytes":8908435456,"fits_memory":false},{"quant":"Q4_K_M","estimated_bytes":4908435456,"fits_memory":true}]},
   {"id":"gemma3:4b","default_quant":"Q4_K_M","parameters_billion":4.0,"context_length":131072,"vision":true,"fits":true,"variants":[{"quant":"Q4_K_M","estimated_bytes":2500000000,"fits_memory":true}]}]}"#;
const DOWNLOADS: &str = r#"{"active":true,"downloads":[{"id":7,"model":"gemma3:4b","quant":"Q4_K_M","state":"running","written":900000000,"total":2500000000,"mb_per_s":38.4,"eta_seconds":42}]}"#;

async fn respond(sock: &mut tokio::net::TcpStream, status: &str, body: &str) {
    let _ = sock.write_all(format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await;
}

/// `models_json` is what `GET /v1/models` answers; loads of the model named `bad` fail with a message the user should read.
async fn serve(models_json: &'static str) -> (String, Log) {
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
                let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
                seen.lock().unwrap().push((method.clone(), path.clone(), body.to_string()));
                match (method.as_str(), path.as_str()) {
                    ("GET", "/v1/models") if models_json.is_empty() => respond(&mut sock, "404 Not Found", r#"{"error":{"message":"unknown route /v1/models","code":"not_found"}}"#).await,
                    ("GET", "/v1/models") => respond(&mut sock, "200 OK", models_json).await,
                    ("GET", "/api/studio/models") => respond(&mut sock, "200 OK", OVERVIEW).await,
                    ("GET", "/api/studio/models/progress") => respond(&mut sock, "200 OK", DOWNLOADS).await,
                    ("POST", "/api/studio/models/load") if body.contains("bad") => respond(&mut sock, "400 Bad Request", r#"{"error":{"message":"not enough memory for bad.gguf"}}"#).await,
                    ("POST", _) => respond(&mut sock, "200 OK", r#"{"ok":true}"#).await,
                    _ => respond(&mut sock, "404 Not Found", "{}").await,
                }
            });
        }
    });
    (format!("http://{addr}"), log)
}

#[tokio::test]
async fn the_overview_and_the_downloads_are_read_from_the_studio_routes() {
    let (base, _log) = serve(r#"{"data":[]}"#).await;
    let o = models::fetch_overview(base.clone()).await.unwrap();
    assert_eq!(o.loaded.as_deref(), Some("qwen2.5-0.5b-instruct"));
    assert_eq!(o.installed.len(), 2);
    assert_eq!(o.catalog.len(), 2);
    let d = models::fetch_downloads(base).await.unwrap();
    assert_eq!((d[0].id, d[0].state.as_str(), d[0].written), (7, "running", 900_000_000));
}

#[tokio::test]
async fn load_unload_pull_and_the_download_actions_send_what_the_server_expects() {
    let (base, log) = serve(r#"{"data":[]}"#).await;
    assert_eq!(models::load(base.clone(), "/models/my-model.gguf".into()).await.unwrap(), "/models/my-model.gguf");
    models::unload(base.clone()).await.unwrap();
    models::pull(base.clone(), "gemma3:4b".into(), "Q4_K_M".into()).await.unwrap();
    models::download_action(base, 7, "pause").await.unwrap();
    let seen: Vec<(String, serde_json::Value)> = log.lock().unwrap().iter().map(|(_, p, b)| (p.clone(), serde_json::from_str(b).unwrap_or_default())).collect();
    assert_eq!(seen[0], ("/api/studio/models/load".into(), serde_json::json!({"model": "/models/my-model.gguf"})));
    assert_eq!(seen[1].0, "/api/studio/models/unload");
    assert_eq!(seen[2], ("/api/studio/models/pull".into(), serde_json::json!({"model": "gemma3:4b", "quant": "Q4_K_M"})));
    assert_eq!(seen[3], ("/api/studio/models/downloads".into(), serde_json::json!({"id": 7, "action": "pause"})));
}

#[tokio::test]
async fn a_refused_load_shows_the_message_of_the_server() {
    let (base, _log) = serve(r#"{"data":[]}"#).await;
    let err = models::load(base, "bad.gguf".into()).await.unwrap_err();
    assert_eq!(err, "not enough memory for bad.gguf");
}

#[tokio::test]
async fn a_server_without_a_model_is_told_apart_from_no_server() {
    let (base, _log) = serve(r#"{"object":"list","data":[]}"#).await;
    assert_eq!(fetch_model(&base).await.unwrap_err(), NO_MODEL);
    // the Studio of a server that unloaded its model has no /v1/models route at all
    let (studio_only, _log) = serve("").await;
    assert_eq!(fetch_model(&studio_only).await.unwrap_err(), NO_MODEL);
    assert!(fetch_model("http://127.0.0.1:9").await.unwrap_err().contains("127.0.0.1:9"));
}

// ---- the window ---------------------------------------------------------------------------------------------------------------------

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store { chats: vec![], next_id: 0, theme: theme.into() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo { id: "qwen2.5-0.5b-instruct".into(), max_context: 8192 })));
    let o = models::parse_overview(&serde_json::from_str(OVERVIEW).unwrap());
    let d = models::parse_downloads(&serde_json::from_str(DOWNLOADS).unwrap());
    let _ = app.update(Message::OverviewLoaded(Ok(o)));
    let _ = app.update(Message::DownloadsLoaded(Ok(d)));
    let _ = app.update(Message::Go(View::Models));
    (app, dir)
}

fn picture(app: &App, name: &str) -> std::path::PathBuf {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the page");
    let dir = std::env::var_os("FLLM_SNAPSHOT_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).find(|p| p.file_name().unwrap().to_string_lossy().starts_with(&prefix)).unwrap()
}

#[test]
fn the_models_page_draws_in_both_themes_with_its_cards() {
    for theme in ["light", "dark"] {
        let (app, _dir) = app(theme);
        assert_eq!(app.page(), View::Models);
        let path = picture(&app, &format!("models-{theme}"));
        let img = image::open(&path).unwrap().to_rgba8();
        let mut counts = std::collections::HashMap::<[u8; 4], usize>::new();
        img.pixels().for_each(|p| *counts.entry(p.0).or_default() += 1);
        let page = counts.values().copied().max().unwrap();
        assert!(1.0 - page as f64 / (img.width() * img.height()) as f64 > 0.05, "{theme}: the page has content");
    }
}

#[test]
fn clicking_load_asks_for_that_model_and_clicking_download_asks_for_the_chosen_quantization() {
    let (mut app, _dir) = app("light");
    let click = |app: &App, label: &str| {
        let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
        let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
        ui.click(label).unwrap_or_else(|e| panic!("{label}: {e:?}"));
        ui.into_messages().collect::<Vec<Message>>()
    };
    let loads = click(&app, "Load");
    assert!(loads.iter().any(|m| matches!(m, Message::LoadModel(r) if r == "/models/my-model.gguf")), "{loads:?}");
    let pulls = click(&app, "Download");
    assert!(pulls.iter().any(|m| matches!(m, Message::Pull(id) if id == "aya-expanse:8b")), "{pulls:?}");
    // a chip changes the quantization, and the next download would use it
    let _ = app.update(Message::PickQuant("aya-expanse:8b".into(), "Q8_0".into()));
    assert_eq!(app.models().chosen_quant(&app.models().overview.catalog[0]), "Q8_0");
    let navigation = click(&app, "Chat");
    assert!(navigation.iter().any(|m| matches!(m, Message::Go(View::Chat))), "{navigation:?}");
}

#[test]
fn a_failed_action_shows_its_message_and_frees_the_buttons() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::LoadModel("/models/my-model.gguf".into()));
    assert_eq!(app.models().busy.as_deref(), Some("/models/my-model.gguf"));
    let _ = app.update(Message::LoadModel("/models/other.gguf".into()));
    assert_eq!(app.models().busy.as_deref(), Some("/models/my-model.gguf"), "one load at a time");
    let _ = app.update(Message::ModelAction(Err("not enough memory".into())));
    assert_eq!(app.models().busy, None);
    assert_eq!(app.models().error.as_deref(), Some("not enough memory"));
    let _ = app.update(Message::OverviewLoaded(Ok(models::Overview::default())));
    assert_eq!(app.models().error, None, "a fresh answer clears the old message");
}

#[test]
fn a_server_without_a_model_sends_the_user_to_the_models_page() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Chat));
    let _ = app.update(Message::Connected(Err(NO_MODEL.into())));
    assert!(matches!(app.server_state(), Server::NoModel));
    let _ = picture(&app, "no-model-dark");
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 800.0), app.view());
    ui.click("Open Models").expect("the card has the button");
    assert!(ui.into_messages().any(|m| matches!(m, Message::Go(View::Models))));
    let _ = app.update(Message::Send);
    assert_eq!(app.message_count(), 0, "nothing is sent without a model");
}
