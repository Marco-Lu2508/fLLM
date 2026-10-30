//! The NPU of an AMD Ryzen AI processor in the Studio, through FastFlowLM ([`crate::backend::npu::flm`]):
//! `GET /api/studio/npu/amd` (what was found: the NPU, FastFlowLM, its check of the NPU stack, the models, the running model) and
//! `POST /api/studio/npu/amd` with `{"action": "start" | "stop" | "pull" | "cancel_pull", "model": "llama3.2:3b"}`.
//!
//! A running model is listed as the provider "AMD NPU (FastFlowLM)" (see [`on_change`]), so a chat can pick it like the model of any other
//! server. The routes start programs and download models, so they are administration: this machine only, reads included (`http::studio_admin_blocked`).
use super::*;
use crate::backend::npu::flm::{self, Serving};

/// Installed once when the routes are built: the provider list follows the model FastFlowLM runs.
pub(in crate::server) fn install() {
    flm::set_hook(on_change);
    // a provider entry left behind by an earlier run points to a server that is gone
    on_change(&flm::serving());
}

fn on_change(state: &Serving) {
    let result = match state {
        Serving::Running { tag, port } => providers::register_local(flm::PROVIDER_ID, flm::PROVIDER_NAME, &format!("http://127.0.0.1:{port}/v1"), std::slice::from_ref(tag)),
        _ => providers::unregister(flm::PROVIDER_ID),
    };
    if let Err(e) = result {
        eprintln!("[fllm-rs] cannot update the provider list for the AMD NPU: {e}");
    }
}

/// For `fllm-rs amd-npu serve --register`.
pub fn register_for_cli(base_url: &str, tag: &str) -> Result<(), String> {
    providers::register_local(flm::PROVIDER_ID, flm::PROVIDER_NAME, base_url, &[tag.to_owned()])
}

pub fn unregister_for_cli() {
    let _ = providers::unregister(flm::PROVIDER_ID);
}

async fn current() -> Response {
    // asks `flm` (a few commands) and the operating system: never on the request thread, never for ever
    match tokio::time::timeout(std::time::Duration::from_secs(90), tokio::task::spawn_blocking(flm::status_json)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("the NPU check failed: {e}")),
        Err(_) => api_error(StatusCode::GATEWAY_TIMEOUT, "the NPU check did not finish within 90 seconds"),
    }
}

pub(in crate::server) async fn state() -> Response {
    current().await
}

/// What a POST asks for.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::server) enum Action {
    Start(String),
    Stop,
    Pull(String),
    CancelPull,
}

pub(in crate::server) fn parse_action(body: &Value) -> Result<Action, String> {
    let action = body.get("action").and_then(Value::as_str).ok_or("the body needs an \"action\"")?;
    let model = || -> Result<String, String> {
        let tag = body.get("model").and_then(Value::as_str).unwrap_or_default().trim().to_owned();
        if flm::valid_tag(&tag) {
            Ok(tag)
        } else {
            Err("\"model\" must be a FastFlowLM model name such as llama3.2:3b".into())
        }
    };
    match action {
        "start" => model().map(Action::Start),
        "stop" => Ok(Action::Stop),
        "pull" => model().map(Action::Pull),
        "cancel_pull" => Ok(Action::CancelPull),
        other => Err(format!("unknown action \"{other}\" (start, stop, pull or cancel_pull)")),
    }
}

pub(in crate::server) async fn action(Json(body): Json<Value>) -> Response {
    let action = match parse_action(&body) {
        Ok(a) => a,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, &message),
    };
    let result = tokio::task::spawn_blocking(move || match action {
        Action::Start(tag) => flm::start(&tag),
        Action::Stop => {
            flm::stop();
            Ok(())
        }
        Action::Pull(tag) => flm::pull(&tag),
        Action::CancelPull => {
            flm::cancel_pull();
            Ok(())
        }
    })
    .await;
    match result {
        Ok(Ok(())) => current().await,
        Ok(Err(message)) => api_error(StatusCode::CONFLICT, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn actions_are_read_and_bad_ones_refused() {
        assert_eq!(parse_action(&json!({"action": "start", "model": "llama3.2:3b"})), Ok(Action::Start("llama3.2:3b".into())));
        assert_eq!(parse_action(&json!({"action": "pull", "model": " gemma3:4b "})), Ok(Action::Pull("gemma3:4b".into())));
        assert_eq!(parse_action(&json!({"action": "stop"})), Ok(Action::Stop));
        assert_eq!(parse_action(&json!({"action": "cancel_pull"})), Ok(Action::CancelPull));
        assert!(parse_action(&json!({"action": "start"})).is_err(), "a model is needed");
        assert!(parse_action(&json!({"action": "start", "model": "x; rm -rf /"})).is_err());
        assert!(parse_action(&json!({"action": "format"})).unwrap_err().contains("unknown action"));
        assert!(parse_action(&json!({})).is_err());
    }
}
