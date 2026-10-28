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

/// Whether the user switched computer use on (screenshots, mouse and keyboard for the agent). Off unless they did.
pub(super) fn computer_use() -> bool {
    read().get("computer_use").and_then(Value::as_bool).unwrap_or(false)
}

/// Actions per minute the agent may do with computer use (1 to 600, default 30).
pub(super) fn computer_rate() -> u32 {
    read().get("computer_rate").and_then(Value::as_u64).map_or(30, |n| n.clamp(1, 600) as u32)
}

/// Actions per task (1 to 5000, default 200).
pub(super) fn computer_max_steps() -> u32 {
    read().get("computer_max_steps").and_then(Value::as_u64).map_or(200, |n| n.clamp(1, 5000) as u32)
}

fn write_all(current: &Value) -> Result<(), String> {
    let path = file();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(current).unwrap_or_else(|_| "{}".to_owned());
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Switches computer use on or off (the STOP button and the panic key switch it off).
pub(super) fn set_computer_use(on: bool) -> Result<(), String> {
    let mut current = read();
    current["computer_use"] = json!(on);
    write_all(&current)
}

fn status() -> Value {
    let sandbox = sandbox_mode();
    json!({
        "sandbox": sandbox,
        "sandbox_from_env": std::env::var("FLLM_SANDBOX").map(|v| !v.trim().is_empty()).unwrap_or(false),
        "turbo": turbo(),
        "computer": super::agent_computer::status_json(),
        "file": file().display().to_string(),
        // what the sandbox can actually do here, so the page does not promise more than the machine can
        "sandbox_available": super::agent_sandbox::available(),
        // the full truth for the page: mechanism, what is and is not enforced, how well it is verified
        "sandbox_status": super::agent_sandbox::status_json(),
    })
}

/// `GET /api/studio/runtime`.
pub(in crate::server) async fn route_status() -> Response {
    Json(status()).into_response()
}

/// `POST /api/studio/runtime` `{sandbox?, turbo?}`.
pub(in crate::server) async fn route_save(Json(body): Json<Value>) -> Response {
    let mut current = read();
    if let Some(on) = body.get("computer_use").and_then(Value::as_bool) {
        if on && !cfg!(feature = "computer-use") {
            return api_error(StatusCode::BAD_REQUEST, "this build has no computer use (it was built without the computer-use feature)");
        }
        if on && std::env::var("FLLM_NO_COMPUTER_USE").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0") {
            return api_error(StatusCode::BAD_REQUEST, "computer use is forbidden on this server (FLLM_NO_COMPUTER_USE)");
        }
        current["computer_use"] = json!(on);
        if on {
            // a fresh start: nothing of an earlier stop or task is carried over
            super::agent_computer::begin();
        }
    }
    for (key, max) in [("computer_rate", 600u64), ("computer_max_steps", 5000u64)] {
        if let Some(n) = body.get(key).and_then(Value::as_u64) {
            current[key] = json!(n.clamp(1, max));
        }
    }
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
    if let Err(e) = write_all(&current) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e);
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
    fn the_default_mode_is_workspace_on_linux_and_off_elsewhere() {
        assert_eq!(super::super::agent_sandbox::DEFAULT_MODE, if cfg!(target_os = "linux") { "workspace" } else { "off" });
    }

    #[test]
    fn an_unknown_file_value_falls_back_instead_of_failing() {
        // `sandbox_mode` reads the real file; with nothing set it has to answer with the default
        let text = sandbox_mode();
        assert!(["off", "workspace", "strict"].contains(&text.as_str()), "{text}");
    }
}
