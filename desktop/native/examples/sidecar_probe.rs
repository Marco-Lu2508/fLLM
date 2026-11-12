//! Starts the server the way the app does and waits for it: `FLLM_RS=/path/to/fllm-rs cargo run --example sidecar_probe [-- model.gguf]`
//! (nothing may listen on port 8080 and FLLM_URL must be unset). Prints what the app would see. Not part of the tests.
use fllm_app::{api, models, sidecar};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let log = std::env::temp_dir().join("fllm-app-probe/server.log");
    let (base, started, note) = sidecar::choose(&log);
    println!("base={base} started_by_app={} note={note:?}", started.is_some());
    for i in 0..60 {
        match api::fetch_model(&base).await {
            Ok(m) => {
                println!("after {i} s: model {:?}", m);
                break;
            }
            Err(e) if e == models::NO_MODEL => {
                println!("after {i} s: server up, no model");
                if let Some(path) = std::env::args().nth(1) {
                    println!("load: {:?}", models::load(base.clone(), path).await);
                    for _ in 0..90 {
                        if let Ok(m) = api::fetch_model(&base).await {
                            println!("model: {m:?}");
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
                break;
            }
            Err(_) => tokio::time::sleep(std::time::Duration::from_secs(1)).await,
        }
    }
    let pid = started.as_ref().map(|s| s.pid());
    drop(started);
    if let Some(pid) = pid {
        std::thread::sleep(std::time::Duration::from_millis(300));
        println!("server process {pid} after drop: {}", if std::path::Path::new(&format!("/proc/{pid}")).exists() { "STILL RUNNING" } else { "gone" });
    }
}
