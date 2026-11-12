//! Runs the Models client against a real server: `FLLM_URL=http://127.0.0.1:8080 cargo run --example models_probe -- /path/to/model.gguf`
//! (lists, unloads, loads the given file, lists again). Used to check the page against the real routes; not part of the tests.
use fllm_app::{api, models};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let base = std::env::var("FLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into());
    let o = models::fetch_overview(base.clone()).await.expect("overview");
    println!("loaded={:?} detail={:?} installed={} catalog={} memory={}", o.loaded, o.loaded_detail, o.installed.len(), o.catalog.len(), models::size(o.memory));
    println!("first catalog entry: {:?}", o.catalog.first().map(|e| (&e.id, &e.default_quant, e.variants.len())));
    println!("unload: {:?}", models::unload(base.clone()).await);
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    println!("after unload: {:?}", api::fetch_model(&base).await);
    if let Some(path) = std::env::args().nth(1) {
        println!("load: {:?}", models::load(base.clone(), path).await);
        // the Studio restarts its engine for a load: wait until /v1/models answers
        for _ in 0..90 {
            let now = api::fetch_model(&base).await;
            if now.is_ok() {
                println!("after load: {now:?}");
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }
    println!("downloads: {:?}", models::fetch_downloads(base).await.map(|d| d.len()));
}
