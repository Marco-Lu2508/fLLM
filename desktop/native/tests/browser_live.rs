//! Row 85 against the real thing: the panel's logic talks to a running engine that drives a real
//! Chromium. It needs both, so it only runs on request:
//!
//! ```text
//! fllm-rs studio --model any.gguf --port 18090 &        # FLLM_BROWSER=<chromium> if it is not found
//! python3 -m http.server 18091 --directory <a folder with tests/fixtures/browser-live-page.html as index.html>
//! FLLM_LIVE_ENGINE=http://127.0.0.1:18090 FLLM_LIVE_PAGE=http://127.0.0.1:18091/ \
//!   cargo test --test browser_live -- --ignored --nocapture
//! ```
//!
//! The page has a heading, a button that rewrites the heading ("Benchmark started") and a text box; the
//! picture `fixtures/browser-page.b64` is a frame of it. Chromium needs `--no-sandbox` when it is not run as root
//! (point `FLLM_BROWSER` at a one-line script that adds it).

use fllm_app::browser::{Input, KeyPress, Panel, Rect};
use serde_json::{json, Value};
use std::time::Duration;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set (see the top of this file)"))
}

async fn post(base: &str, path: &str, body: Value) -> Value {
    let r = reqwest::Client::new().post(format!("{base}/api/studio/agent/browser{path}")).json(&body).send().await.unwrap();
    r.json().await.unwrap()
}

async fn status(base: &str) -> Value {
    reqwest::get(format!("{base}/api/studio/agent/browser")).await.unwrap().json().await.unwrap()
}

/// What the panel does for one `Input`: the request, the answer, and the answer given back to the panel.
async fn send(base: &str, panel: &mut Panel, input: Input) -> Value {
    let answer = post(base, "/input", input.body()).await;
    assert!(panel.input_answered(&input, Ok(answer.clone())).is_none(), "the engine refused {input:?}: {answer}");
    answer
}

/// One tick: the frame since the last one, given to the panel.
async fn frame(base: &str, panel: &mut Panel) {
    let data = post(base, "/frame", json!({ "since": panel.key })).await;
    panel.apply_frame(&data);
}

/// The element under a point of the page, as the *Comment* mode finds it.
async fn locate(base: &str, panel: &mut Panel, x: f64, y: f64) -> Value {
    send(base, panel, Input::Locate { x, y }).await
}

#[tokio::test]
#[ignore = "needs a running engine and a Chromium: see the top of the file"]
async fn the_panel_drives_a_real_browser() {
    let base = env("FLLM_LIVE_ENGINE");
    let page = env("FLLM_LIVE_PAGE");
    let mut panel = Panel::new("linux");

    // the browser is switched on (what Settings > Browser does), the panel reads its status
    let settings = json!({ "enabled": true, "show_window": false, "local_addresses": "auto", "executable": "", "persistent_profile": false });
    reqwest::Client::new().post(format!("{base}/api/studio/agent/browser/settings")).json(&settings).send().await.unwrap();
    panel.apply_status(Some(&status(&base).await));
    assert!(panel.status.as_ref().is_some_and(|s| s.enabled), "switched on");

    // an address typed in the box and sent
    panel.url_input = page.clone();
    let go = panel.submit().expect("an address is a request");
    assert_eq!(go, Input::Navigate(page.clone()));
    send(&base, &mut panel, go).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    frame(&base, &mut panel).await;
    assert!(panel.running && panel.show_image(), "the page is drawn: {:?}", panel.view());
    assert_eq!(panel.title, "fLLM Studio – Dashboard");
    assert_eq!(panel.view, (1280.0, 800.0));
    let image = panel.image.clone().unwrap();
    assert!(image.starts_with("data:image/jpeg;base64,"), "{}", &image[..30]);
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, image.split_once(',').unwrap().1).unwrap();
    assert!(::image::load_from_memory(&bytes).is_ok(), "the window can decode what the engine sends");
    let key = panel.key.clone();

    // nothing changed since: the engine says so and the panel keeps its picture
    frame(&base, &mut panel).await;
    assert_eq!(panel.key, key);

    // the panel is 400 wide in the window: a click at a place of it is a click at that place of the page
    let stage = Rect { left: 0.0, top: 0.0, width: 400.0, height: 250.0 };
    let heading_at = (100.0 / 1280.0 * 400.0, 130.0 / 800.0 * 250.0);
    let (x, y) = panel.page_point(heading_at, stage).unwrap();
    assert!((x - 100.0).abs() < 0.01 && (y - 130.0).abs() < 0.01);
    let found = locate(&base, &mut panel, x, y).await;
    assert_eq!(found["tag"], "h1");
    assert_eq!(found["text"], "Dashboard");

    // the button is somewhere above the bottom of the content: look for it the way a user would
    let mut button = None;
    for y in (300..520).step_by(10) {
        let found = locate(&base, &mut panel, 120.0, f64::from(y)).await;
        if found["tag"] == "button" {
            button = Some((120.0, f64::from(y)));
            break;
        }
    }
    let (bx, by) = button.expect("a button is on the page");
    let at = (bx / 1280.0 * 400.0, by / 800.0 * 250.0);
    let click = panel.click(at, stage, false).expect("a click on a live page");
    assert!(matches!(click, Input::Click { .. }));
    send(&base, &mut panel, click).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    let heading = locate(&base, &mut panel, 100.0, 130.0).await;
    assert_eq!(heading["text"], "Benchmark started", "the click reached the button: {heading}");
    frame(&base, &mut panel).await;
    assert_ne!(panel.key, key, "and the picture is a new one");

    // text and keys go to the box that has the focus
    let mut boxed = None;
    for y in (380..560).step_by(8) {
        let found = locate(&base, &mut panel, 700.0, f64::from(y)).await;
        if found["tag"] == "input" {
            boxed = Some(f64::from(y));
            break;
        }
    }
    let by = boxed.expect("the text box is on the page");
    send(&base, &mut panel, Input::Click { x: 700.0, y: by }).await;
    let typed = panel.paste("Qwen3").expect("a paste is text for the page");
    send(&base, &mut panel, typed).await;
    let enter = panel.key(&KeyPress { key: "Enter".into(), ..Default::default() }).expect("Enter goes to the page");
    send(&base, &mut panel, enter).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let boxed = locate(&base, &mut panel, 700.0, by).await;
    assert_eq!(boxed["tag"], "input");
    assert!(boxed.to_string().contains("Qwen3"), "what was typed is in the box: {boxed}");

    // comment mode: the element under the click becomes a comment for the agent
    panel.toggle_commenting();
    let c = panel.click(heading_at, stage, false).expect("in comment mode a click asks which element it is");
    assert!(matches!(c, Input::Locate { .. }));
    send(&base, &mut panel, c).await;
    assert!(panel.draft.is_some());
    assert!(panel.add_comment("Make this heading bigger"));
    let text = panel.comments_text();
    assert!(text.contains("heading") && text.contains("Make this heading bigger"), "{text}");

    // scrolling, back and reload are answered
    send(&base, &mut panel, Input::Wheel { x: 640.0, y: 400.0, dx: 0.0, dy: 300.0 }).await;
    send(&base, &mut panel, Input::Reload).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    frame(&base, &mut panel).await;
    assert!(panel.running);
    let heading = locate(&base, &mut panel, 100.0, 130.0).await;
    assert_eq!(heading["text"], "Dashboard", "a reload brings the page back as it was loaded");

    // closing the browser: the panel sees it stopped
    send(&base, &mut panel, Input::Close).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    panel.apply_status(Some(&status(&base).await));
    frame(&base, &mut panel).await;
    assert!(!panel.running && !panel.show_image(), "{:?}", panel.view());
}
