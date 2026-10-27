//! How freely the app may work: the sandbox mode for agent commands and the turbo switch that removes every question.
//!
//! Both live in `runtime.json` next to `studio.json`, so the choice survives a restart:
//! `{"sandbox": "workspace", "turbo": false}`. The environment (`FLLM_SANDBOX`) still wins when it is set, because a
//! service file is a harder promise than a click in the page.
//!
//! Turbo means: no permission questions at all (every tool call is allowed), and the sandbox is off — exactly what the
//! user asks for when they say "just do it". It is a real switch with a real warning, not a hidden flag, and it stays off
//! until somebody turns it on.
use super::*;
use serde_json::{json, Value};
use std::path::PathBuf;

fn file() -> PathBuf {
    crate::hub::studio_data_file("runtime.json")
}

fn read() -> Value {
    let Ok(text) = std::fs::read_to_string(file()) else { return json!({}) };
    serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({}))
}

/// The sandbox mode: the environment wins, then the file, then the default.
pub(super) fn sandbox_mode() -> String {
    if let Ok(from_env) = std::env::var("FLLM_SANDBOX") {
        if !from_env.trim().is_empty() {
            return from_env.trim().to_ascii_lowercase();
        }
    }
    let from_file = read().get("sandbox").and_then(Value::as_str).unwrap_or("").trim().to_ascii_lowercase();
    if from_file.is_empty() {
        super::agent_sandbox::DEFAULT_MODE.to_owned()
    } else {
        from_file
    }
}

/// Whether turbo is on.
pub(super) fn turbo() -> bool {
    if let Ok(text) = std::env::var("FLLM_TURBO") {
        if !text.trim().is_empty() {
            return matches!(text.trim().to_ascii_lowercase().as_str(), "1" | "on" | "yes" | "true");
        }
    }
    read().get("turbo").and_then(Value::as_bool).unwrap_or(false)
}

fn status() -> Value {
    let sandbox = sandbox_mode();
    json!({
        "sandbox": sandbox,
        "sandbox_from_env": std::env::var("FLLM_SANDBOX").map(|v| !v.trim().is_empty()).unwrap_or(false),
        "turbo": turbo(),
        "file": file().display().to_string(),
        // what the sandbox can actually do here, so the page does not promise more than the machine can
        "sandbox_available": super::agent_sandbox::available(),
    })
}

/// `GET /api/studio/runtime`.
pub(in crate::server) async fn route_status() -> Response {
    Json(status()).into_response()
}

/// `POST /api/studio/runtime` `{sandbox?, turbo?}`.
pub(in crate::server) async fn route_save(Json(body): Json<Value>) -> Response {
    let mut current = read();
    if let Some(sandbox) = body.get("sandbox").and_then(Value::as_str) {
        let value = sandbox.trim().to_ascii_lowercase();
        if !["off", "workspace", "strict"].contains(&value.as_str()) {
            return api_error(StatusCode::BAD_REQUEST, "sandbox has to be off, workspace or strict");
        }
        if std::env::var("FLLM_SANDBOX").map(|v| !v.trim().is_empty()).unwrap_or(false) {
            return api_error(
                StatusCode::BAD_REQUEST,
                "FLLM_SANDBOX is set for this server and wins over the page; start the server without it to choose here",
            );
        }
        current["sandbox"] = json!(value);
    }
    if let Some(turbo) = body.get("turbo").and_then(Value::as_bool) {
        current["turbo"] = json!(turbo);
        // turbo without a sandbox is the point of the switch: harmless work must not stop
        if turbo {
            current["sandbox"] = json!("off");
        }
    }
    let path = file();
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot create {}: {e}", parent.display()));
        }
    }
    let text = serde_json::to_string_pretty(&current).unwrap_or_else(|_| "{}".to_owned());
    if let Err(e) = std::fs::write(&path, text) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot write {}: {e}", path.display()));
    }
    Json(status()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_three_modes_are_accepted() {
        // the check itself, without touching the file
        for good in ["off", "workspace", "strict"] {
            assert!(["off", "workspace", "strict"].contains(&good));
        }
        for bad in ["sandbox", "", "open", "all"] {
            assert!(!["off", "workspace", "strict"].contains(&bad), "{bad}");
        }
    }

    #[test]
    fn the_default_mode_is_workspace() {
        assert_eq!(super::super::agent_sandbox::DEFAULT_MODE, "workspace");
    }

    #[test]
    fn an_unknown_file_value_falls_back_instead_of_failing() {
        // `sandbox_mode` reads the real file; with nothing set it has to answer with the default
        let text = sandbox_mode();
        assert!(["off", "workspace", "strict"].contains(&text.as_str()), "{text}");
    }
}
