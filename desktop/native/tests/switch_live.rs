//! Switching the model against a real engine (it only runs on request): the sequence the window goes through — the load is
//! accepted, the old engine still answers for a moment, it goes away, the new one is back — and the model the window reads when
//! its wait ends is the NEW one.
//!
//! ```text
//! fllm-rs studio --model a.gguf --port 18090 &
//! FLLM_LIVE_ENGINE=http://127.0.0.1:18090 FLLM_LIVE_MODEL_B=/path/b.gguf \
//!   cargo test --test switch_live -- --ignored --nocapture
//! ```

use fllm_app::app::{App, Message, Server};
use fllm_app::store::Store;
use std::time::{Duration, Instant};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set (see the top of this file)"))
}

#[tokio::test]
#[ignore = "needs a running engine and a second model: see the top of the file"]
async fn the_window_ends_its_wait_with_the_new_model() {
    let base = env("FLLM_LIVE_ENGINE");
    let other = env("FLLM_LIVE_MODEL_B");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with(base.clone(), path);
    let before = fllm_app::api::fetch_model(&base).await.expect("an engine with a model");
    let _ = app.update(Message::Connected(Ok(before.clone())));

    let started = Instant::now();
    let _ = app.update(Message::LoadModel(other.clone()));
    fllm_app::models::load(base.clone(), other.clone()).await.expect("the load is accepted");
    println!("accepted after {:?}", started.elapsed());

    // what the window does: look every half second until the wait ends
    let mut looks = 0;
    let mut answers_of_the_old_engine = 0;
    let mut down = 0;
    while app.waits_for_the_engine() {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let answer = tokio::time::timeout(Duration::from_secs(4), fllm_app::models::fetch_status(base.clone())).await.unwrap_or_else(|_| Err("no answer".into()));
        looks += 1;
        match &answer {
            Ok(s) if s.model_name == before.detail.name => answers_of_the_old_engine += 1,
            Err(_) => down += 1,
            _ => {}
        }
        let _ = app.update(Message::SwitchProbe(answer));
        assert!(started.elapsed() < Duration::from_secs(60), "the wait must end");
    }
    println!("wait ended after {:?}: {looks} looks, {answers_of_the_old_engine} answers of the old engine, {down} without answer", started.elapsed());
    assert!(down >= 1, "the engine was away at least once");

    // the model the window reads now is the new one
    let now = fllm_app::api::fetch_model(&base).await.expect("the engine is back");
    assert_ne!(now.detail.name, before.detail.name, "the window would have shown the old model");
    let _ = app.update(Message::Connected(Ok(now.clone())));
    assert!(!app.switching());
    assert!(matches!(app.server_state(), Server::Ready(m) if m.id == now.id));
}
