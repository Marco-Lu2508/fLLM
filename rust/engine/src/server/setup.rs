//! The server without a model: `fllm-rs studio` on a machine that has not chosen a model yet. It serves the Studio page and the model routes
//! (search, download, delete, load) so that the first model can be picked in the browser. "Load" starts the real server, which takes over the
//! port (see `studio::load_model`).
use super::*;
use axum::routing::{delete, get, post};

static PROBLEM: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn status() -> Value {
    json!({
        "status": "ok",
        "ready": true,
        "setup": true,
        "setup_error": PROBLEM.get(),
        // the model that loads with the first message (the page offers it instead of sending the user to the model list)
        "last_model": if PROBLEM.get().is_some() { None } else { crate::hub::last_studio_model() },
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
        .route("/studio/runner.js", get(|| async { crate::webui::runner() }))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/ready", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/api/studio/status", get(|| async { Json(status()) }))
        .route("/api/studio/models", get(|| api::studio::models_hub_with(Value::Null)))
        .route("/api/studio/hardware", get(api::studio::hardware))
        .route("/api/studio/npu", get(api::npu::state))
        .route("/api/studio/npu/setup", post(api::npu::start_setup).delete(api::npu::remove_runtime))
        .route("/api/studio/hf-account", get(api::studio::hf_account).post(api::studio::hf_login))
        .route("/api/studio/open-url", post(api::studio::open_url))
        .route("/api/studio/extract", post(api::extract::extract).layer(axum::extract::DefaultBodyLimit::max(api::extract::MAX_UPLOAD_BYTES)))
        .route("/api/studio/update", get(api::studio::update_status))
        .route("/api/studio/update/install", post(api::studio::update_install))
        .route("/api/studio/mcp/servers", get(api::mcp::servers).post(api::mcp::upsert).delete(api::mcp::remove))
        .route("/api/studio/mcp/restart", post(api::mcp::restart))
        .route("/api/studio/mcp/tools", get(api::mcp::tools))
        .route("/api/studio/mcp/call", post(api::mcp::call))
        .route("/api/studio/providers", get(api::providers::list).post(api::providers::upsert).delete(api::providers::remove))
        .route("/api/studio/providers/test", post(api::providers::test_connection))
        .route("/api/studio/providers/models", get(api::providers::models))
        .route("/api/studio/providers/chat", post(api::providers::chat).layer(axum::extract::DefaultBodyLimit::max(api::providers::MAX_CHAT_BODY)))
        .route("/api/studio/hf-files", get(api::studio::hf_files))
        .route("/api/studio/models/pull", post(api::studio::pull_model))
        .route("/api/studio/models/load", post(api::studio::load_model))
        .route("/api/studio/models/search", get(api::studio::search_models))
        .route("/api/studio/models/progress", get(api::studio::model_progress))
        .route("/api/studio/models/cache", delete(api::studio::delete_cached_model))
        .route("/api/studio/models/local", post(api::studio::add_local_model).delete(api::studio::remove_local_model))
        .route("/api/studio/models/downloads", post(api::studio_ops::download_action))
        .route("/api/studio/memory", get(api::memory::get_memory).put(api::memory::put_memory).layer(axum::extract::DefaultBodyLimit::max(api::memory::MAX_BODY)))
        .route("/api/studio/pick-file", post(api::studio_ops::pick_file))
        .route("/api/studio/settings", get(api::studio_ops::get_settings).post(api::studio_ops::update_settings))
        .route("/api/studio/engine", get(api::studio_ops::get_engine).post(api::studio_ops::update_engine))
        .route("/api/studio/restart", post(api::studio_ops::restart_server))
        .route("/api/studio/logs", get(api::studio_ops::logs))
        .route("/api/studio/hf-readme", get(api::studio_ops::hf_readme))
        .route("/api/studio/skills", get(api::skills::list).post(api::skills::write).delete(api::skills::remove))
        .route("/api/studio/skills/read", get(api::skills::read))
        .route("/api/studio/sessions/index", post(api::sessions_index::index).delete(api::sessions_index::remove))
        .route("/api/studio/sessions/search", get(api::sessions_index::search))
        .route("/api/studio/schedule", get(api::schedule::list).post(api::schedule::upsert).delete(api::schedule::remove))
        .route("/api/studio/schedule/run", post(api::schedule::run_now))
        .route("/api/studio/schedule/results", get(api::schedule::results))
        .route("/api/studio/schedule/check", get(api::schedule::check))
        // the settings pages that need no model: the app opens without one now and loads the last model with the first message
        .route("/api/studio/skills/library", get(api::skills::library).post(api::skills::install_builtin))
        .route("/api/studio/skills/import", post(api::skills::import))
        .route("/api/studio/skills/github", get(api::skills::github_list))
        .route("/api/studio/skills/directory", get(api::skills::directory_list))
        .route("/api/studio/skills/directory/install", post(api::skills::directory_install))
        .route("/api/studio/skills/directory/token", post(api::skills::directory_token_save))
        .route("/api/studio/hooks", get(api::hooks::list).post(api::hooks::save_handler))
        .route("/api/studio/hooks/test", post(api::hooks::test))
        .route("/api/studio/agents", get(api::agents::list).post(api::agents::write).delete(api::agents::remove_route))
        .route("/api/studio/prices", get(api::prices::list).post(api::prices::save_handler))
        .route("/api/studio/plugins", get(api::plugins::list).post(api::plugins::install_route).delete(api::plugins::remove_route))
        .route("/api/studio/plugins/commands", get(api::plugins::commands_route))
        .route("/api/studio/plugins/command", get(api::plugins::command_route))
        .route("/api/studio/runtime", get(api::studio_mode::route_status).post(api::studio_mode::route_save))
        .route("/api/studio/embeddings", get(api::studio_embed::route_status).post(api::studio_embed::route_save))
        .route("/api/studio/search", post(api::websearch::search_route))
        .route("/api/studio/search/engines", get(api::websearch::engines_route).post(api::websearch::engines_save))
        .route("/api/studio/web/search", post(api::websearch::search_route))
        .layer(axum::middleware::from_fn_with_state(context, http::layer))
}

/// Serves the page and the model routes on `host:port` until the process is stopped. `problem` is why there is no model (the one
/// that was meant to load failed): the page shows it.
pub fn serve_setup(host: &str, port: u16, problem: Option<String>) -> Result<(), String> {
    if let Some(problem) = problem {
        let _ = PROBLEM.set(problem);
    }
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
