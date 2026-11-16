//! Row 86 against a real screen: the server's computer tools (`/api/studio/agent/tool`) drive an X server
//! and the answers are the ones the window's cards and banner are built for. It needs an engine with the
//! `computer-use` feature on a machine that has a display — a virtual one is enough — so it only runs on
//! request:
//!
//! ```text
//! Xvfb :99 -screen 0 1280x800x24 &
//! DISPLAY=:99 xev -geometry 400x300+0+0 > /tmp/xev.log &      # a window that logs what reaches it
//! DISPLAY=:99 fllm-rs studio --model any.gguf --port 18092 &
//! DISPLAY=:99 FLLM_LIVE_ENGINE=http://127.0.0.1:18092 FLLM_LIVE_XEV=/tmp/xev.log \
//!   cargo test --test computer_live -- --ignored --nocapture      # (or run the built test binary)
//! ```
//!
//! `xdotool` is only the second pair of eyes: it says where the pointer really is.

use serde_json::{json, Value};
use std::process::Command;
use std::time::Duration;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set (see the top of this file)"))
}

async fn get(base: &str, path: &str) -> Value {
    reqwest::get(format!("{base}{path}")).await.unwrap().json().await.unwrap()
}

async fn post(base: &str, path: &str, body: Value) -> Value {
    reqwest::Client::new().post(format!("{base}{path}")).json(&body).send().await.unwrap().json().await.unwrap()
}

/// One computer tool, as the window runs it after the user said yes.
async fn tool(base: &str, name: &str, args: Value) -> Value {
    post(base, "/api/studio/agent/tool", json!({ "name": name, "args": args })).await
}

fn pointer() -> (i64, i64) {
    let out = Command::new("xdotool").arg("getmouselocation").output().expect("xdotool");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let n = |key: &str| text.split_whitespace().find_map(|w| w.strip_prefix(key)).and_then(|v| v.parse().ok()).unwrap_or(-1);
    (n("x:"), n("y:"))
}

#[tokio::test]
#[ignore = "needs an engine with computer use and an X display: see the top of the file"]
async fn the_computer_tools_drive_a_real_screen() {
    let base = env("FLLM_LIVE_ENGINE");
    let xev = env("FLLM_LIVE_XEV");

    // switched on in Settings > Runtime: the server says it can, and how many actions it allows
    let before = get(&base, "/api/studio/agent/computer").await;
    assert_eq!(before["compiled"], true, "{before}");
    assert_eq!(before["available"], true, "the display is there: {before}");
    assert_eq!(before["enabled"], false, "off until the user switches it on");
    let on = post(&base, "/api/studio/runtime", json!({ "computer_use": true })).await;
    assert!(on.get("error").is_none(), "{on}");
    let status = get(&base, "/api/studio/agent/computer").await;
    assert_eq!(status["enabled"], true);
    assert_eq!(status["steps_used"], 0);
    assert_eq!(status["panic_chord"], "Ctrl+Alt+Shift+Esc");
    // the tool list has them now
    let tools = get(&base, "/api/studio/agent/tools").await;
    let names: Vec<&str> = tools["tools"].as_array().unwrap().iter().filter_map(|t| t.pointer("/function/name").and_then(Value::as_str)).collect();
    for want in ["computer_screenshot", "computer_click", "computer_type", "computer_key", "computer_scroll", "computer_drag"] {
        assert!(names.contains(&want), "{want} is offered: {names:?}");
    }

    // computerBegin: the server starts counting the actions of a task
    let begun = post(&base, "/api/studio/agent/computer/begin", json!({})).await;
    assert!(begun.get("error").is_none(), "{begun}");

    // an action before the first screenshot has nothing to refer to, but a screenshot works at once
    let shot = tool(&base, "computer_screenshot", json!({})).await;
    assert_eq!(shot["ok"], true, "{shot}");
    let image = &shot["image"];
    assert_eq!(image["mime"], "image/png");
    let (w, h) = (image["width"].as_u64().unwrap(), image["height"].as_u64().unwrap());
    assert_eq!((w, h), (1280, 800), "the whole virtual screen");
    let png = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, image["data"].as_str().unwrap()).unwrap();
    let decoded = ::image::load_from_memory(&png).expect("the window can decode the screenshot").to_rgb8();
    assert_eq!(decoded.dimensions(), (1280, 800));
    let colours: std::collections::HashSet<[u8; 3]> = decoded.pixels().map(|p| p.0).take(200_000).collect();
    assert!(colours.len() > 1, "a window is on the screen, not one colour");
    // what the window draws over it is computed from the same numbers
    assert!(!fllm_app::computer::marks_svg(w as f64, h as f64, &fllm_app::computer::computer_marks("computer_click", &json!({ "x": 200, "y": 150 }))).is_empty());

    // the pointer goes where it is told, in the pixels of the screenshot
    let moved = tool(&base, "computer_move", json!({ "x": 200, "y": 150 })).await;
    assert_eq!(moved["ok"], true, "{moved}");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(pointer(), (200, 150), "the pointer of the X server");

    // a click reaches the window under it
    let clicked = tool(&base, "computer_click", json!({ "x": 220, "y": 160 })).await;
    assert_eq!(clicked["ok"], true, "{clicked}");
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(pointer(), (220, 160));
    let log = std::fs::read_to_string(&xev).unwrap_or_default();
    assert!(log.contains("ButtonPress"), "the window got the click:\n{}", &log[log.len().saturating_sub(400)..]);

    // typing and a key
    let typed = tool(&base, "computer_type", json!({ "text": "hi" })).await;
    assert_eq!(typed["ok"], true, "{typed}");
    let key = tool(&base, "computer_key", json!({ "keys": "Return" })).await;
    assert_eq!(key["ok"], true, "{key}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let log = std::fs::read_to_string(&xev).unwrap_or_default();
    assert!(log.contains("keysym 0x68, h") && log.contains("keysym 0x69, i"), "the letters arrived:\n{}", &log[log.len().saturating_sub(600)..]);
    assert!(log.contains("keysym 0xff0d, Return"), "and Enter");

    // a scroll and a drag are answered, a wait is not counted
    assert_eq!(tool(&base, "computer_scroll", json!({ "x": 200, "y": 150, "dy": 3 })).await["ok"], true);
    assert_eq!(tool(&base, "computer_drag", json!({ "from": { "x": 100, "y": 100 }, "to": { "x": 180, "y": 140 } })).await["ok"], true);
    let steps_before_wait = get(&base, "/api/studio/agent/computer").await["steps_used"].as_u64().unwrap();
    assert_eq!(tool(&base, "computer_wait", json!({ "ms": 50 })).await["ok"], true);
    assert_eq!(get(&base, "/api/studio/agent/computer").await["steps_used"].as_u64().unwrap(), steps_before_wait, "a wait is not an action");
    assert!(steps_before_wait >= 6, "the actions are counted: {steps_before_wait}");

    // an action that is not on the screen is refused with words, not a crash
    let off = tool(&base, "computer_click", json!({ "x": 99999, "y": 5 })).await;
    assert_eq!(off["ok"], false, "{off}");
    assert!(off["error"].as_str().is_some_and(|e| !e.is_empty()));

    // STOP: cancels and switches computer use off, which the window's poll then sees
    let stopped = post(&base, "/api/studio/agent/computer/stop", json!({})).await;
    assert!(stopped.get("error").is_none(), "{stopped}");
    let after = get(&base, "/api/studio/agent/computer").await;
    assert_eq!(after["enabled"], false, "STOP switches it off: {after}");
    let refused = tool(&base, "computer_screenshot", json!({})).await;
    assert_eq!(refused["ok"], false, "and nothing runs any more: {refused}");
}
