//! The server without a model: `fllm-rs studio` on a machine that has not chosen a model yet. It serves the Studio page and the model routes
//! (search, download, delete, load) so that the first model can be picked in the browser. "Load" starts the real server, which takes over the
//! port (see `studio::load_model`).
use super::*;
use axum::routing::{delete, get, post};

fn status() -> Value {
    json!({
        "status": "ok",
        "ready": true,
        "setup": true,
        "version": env!("CARGO_PKG_VERSION"),
        "build_info": crate::version::long_version(),
        "model": {
            "name": "", "arch": "", "size_bytes": 0, "parameters": 0, "parameter_size": "", "quantization": "",
            "max_context": 4096, "trained_context": 0, "max_sequences": 1, "vocab_size": 0, "aliases": [],
            "capabilities": {}, "chat_template": "", "tool_parser": "",
        },
        "config": {"active_cpu_tier": "-"},
        "http": {"auth_required": false, "keys_count": 0, "rate_limit": {}},
        "metrics": {"uptime_secs": 0, "slots_running": 0, "slots_total": 0, "generated_tokens_total": 0},
    })
}

fn router(context: Arc<http::HttpContext>) -> Router {
    Router::new()
        .route("/", get(|| async { crate::webui::page(false) }))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/ready", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/api/studio/status", get(|| async { Json(status()) }))
        .route("/api/studio/models", get(|| async { Json(api::studio::models_hub_json(Value::Null)) }))
        .route("/api/studio/hardware", get(api::studio::hardware))
        .route("/api/studio/hf-files", get(api::studio::hf_files))
        .route("/api/studio/models/pull", post(api::studio::pull_model))
        .route("/api/studio/models/load", post(api::studio::load_model))
        .route("/api/studio/models/search", get(api::studio::search_models))
        .route("/api/studio/models/progress", get(api::studio::model_progress))
        .route("/api/studio/models/cache", delete(api::studio::delete_cached_model))
        .layer(axum::middleware::from_fn_with_state(context, http::layer))
}

/// Serves the page and the model routes on `host:port` until the process is stopped.
pub fn serve_setup(host: &str, port: u16) -> Result<(), String> {
    let address = listen_address(host, port)?;
    let context = http::HttpContext::new(options::HttpOptions::default());
    let ready = Arc::new(AtomicBool::new(false));
    let app = router(context.clone());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start HTTP runtime: {e}"))?;
    runtime.block_on(async move {
        let patient = std::env::var_os("FLLM_REPLACE_SERVER").is_some();
        let waiting_since = std::time::Instant::now();
        let listener = loop {
            match tokio::net::TcpListener::bind(address).await {
                Ok(listener) => break listener,
                Err(_) if patient && waiting_since.elapsed() < std::time::Duration::from_secs(90) => {
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
                Err(e) => return Err(format!("cannot bind HTTP listener: {e}")),
            }
        };
        println!("{}", crate::version::long_version());
        println!("fllm-rs studio: no model yet, pick one in the browser at http://{address}/");
        ready.store(true, Ordering::Relaxed);
        if std::env::var_os("FLLM_OPEN_BROWSER").is_some() && !patient {
            open_browser(&format!("http://{}:{}/", if address.ip().is_unspecified() { "127.0.0.1".to_owned() } else { address.ip().to_string() }, address.port()));
        }
        let (_keep, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        http::serve(listener, app, context, ready.clone(), shutdown_signal(shutdown_rx)).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    async fn get(path: &str) -> (u16, Value) {
        let app = router(http::HttpContext::new(options::HttpOptions::default()));
        let request = Request::builder()
            .uri(path)
            .header("host", "127.0.0.1:8000")
            .extension(http::ClientAddr("127.0.0.1:50000".parse().unwrap()))
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let code = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        (code, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn the_server_without_a_model_offers_the_model_page() {
        let (code, status) = get("/api/studio/status").await;
        assert_eq!(code, 200);
        assert_eq!(status["setup"], true);
        let (code, hub) = get("/api/studio/models").await;
        assert_eq!(code, 200);
        assert!(hub["loaded"].is_null());
        assert!(hub["catalog"].as_array().is_some_and(|c| !c.is_empty()));
    }
}
