//! `GET /api/studio/connect?tool=claude-code`: the settings that point a coding tool at this server, for the Studio
//! section "Connect a coding tool". The text comes from the same code as `fllm-rs connect`.
use super::*;
use crate::connect_cli::{describe, Settings, Targets, TOOLS};

pub(in crate::server) async fn connect(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let tool = query.get("tool").map(String::as_str).unwrap_or("claude-code");
    let host = headers.get("host").and_then(|h| h.to_str().ok()).unwrap_or("127.0.0.1:8000");
    let settings = Settings {
        base: format!("http://{host}"),
        model: state.model_name.clone(),
        key: "fllm".into(),
        key_given: false,
        context: Some(state.max_context as u64),
        live: true,
    };
    let targets = Targets { folder: std::path::PathBuf::from("."), home: std::path::PathBuf::from("~"), global: false };
    match describe(tool, &settings, &targets) {
        Ok(mut plan) => {
            plan["tools"] = json!(TOOLS.iter().map(|(name, what)| json!({"id": name, "what": what})).collect::<Vec<_>>());
            Json(plan).into_response()
        }
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}
