//! Runs the Settings client against a real server: `FLLM_URL=http://127.0.0.1:8080 cargo run --example engine_probe`
//! (reads the engine options, writes `state_cache_mb`, reads again, puts it back). Not part of the tests.
use fllm_app::engine_settings as e;
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let base = std::env::var("FLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let (engine, server) = e::fetch(base.clone()).await.expect("fetch");
    println!("defaults: {}", serde_json::Value::Object(engine.defaults.clone()));
    println!("values: {}  locked: {:?}  server: {:?}", serde_json::Value::Object(engine.values.clone()), engine.locked, server);
    println!("save state_cache_mb=2048: {:?}", e::save_engine(base.clone(), json!({"state_cache_mb": 2048, "state_cache": true})).await);
    let (after, _) = e::fetch(base.clone()).await.unwrap();
    println!("after: state_cache_mb={:?} state_cache={:?}", after.values.get("state_cache_mb"), after.values.get("state_cache"));
    println!("bad value: {:?}", e::save_engine(base.clone(), json!({"state_cache_mb": 0})).await);
    println!("reset: {:?}", e::save_engine(base.clone(), json!({"state_cache_mb": 1024, "state_cache": true})).await);
    println!("server settings: {:?}", e::save_server(base, json!({"idle_unload_minutes": 0})).await);
}
