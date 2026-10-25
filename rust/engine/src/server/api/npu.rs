//! Intel NPU support in the Studio: `GET /api/studio/npu` (state), `POST /api/studio/npu/setup` (download and install the OpenVINO
//! runtime in a background thread, or `{"action":"cancel"}`) and `DELETE /api/studio/npu/setup` (remove the runtime).
//!
//! The POST and DELETE routes change files and download programs, so they are administration: without a key they only accept this
//! machine, and (see `http::studio_admin_blocked`) they never accept another machine, with or without a key.
use super::*;
use crate::backend::npu::{setup, status};

/// What the body of `POST /api/studio/npu/setup` asks for.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::server) enum SetupAction {
    Install,
    Cancel,
}

/// An empty body means install.
pub(in crate::server) fn parse_action(body: &[u8]) -> Result<SetupAction, String> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(SetupAction::Install);
    }
    let value: Value = serde_json::from_slice(body).map_err(|_| "the request body must be JSON".to_owned())?;
    match value.get("action").and_then(Value::as_str).unwrap_or("install") {
        "install" => Ok(SetupAction::Install),
        "cancel" => Ok(SetupAction::Cancel),
        other => Err(format!("unknown action \"{other}\" (install or cancel)")),
    }
}

async fn current() -> Response {
    // the check asks the operating system and may load the OpenVINO runtime: never on the request thread, never for ever
    match tokio::time::timeout(std::time::Duration::from_secs(60), tokio::task::spawn_blocking(status::to_json)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("the NPU check failed: {e}")),
        Err(_) => api_error(StatusCode::GATEWAY_TIMEOUT, "the NPU check did not finish within 60 seconds"),
    }
}

pub(in crate::server) async fn state() -> Response {
    current().await
}

pub(in crate::server) async fn start_setup(body: axum::body::Bytes) -> Response {
    match parse_action(&body) {
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
        Ok(SetupAction::Cancel) => {
            setup::cancel();
            current().await
        }
        Ok(SetupAction::Install) => match setup::start() {
            Ok(()) => current().await,
            Err(message) => api_error(StatusCode::CONFLICT, &message),
        },
    }
}

pub(in crate::server) async fn remove_runtime() -> Response {
    match tokio::task::spawn_blocking(setup::remove).await {
        Ok(Ok(())) => current().await,
        Ok(Err(message)) => api_error(StatusCode::CONFLICT, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_action_is_read_from_the_body() {
        assert_eq!(parse_action(b""), Ok(SetupAction::Install));
        assert_eq!(parse_action(b" \n"), Ok(SetupAction::Install));
        assert_eq!(parse_action(b"{}"), Ok(SetupAction::Install));
        assert_eq!(parse_action(br#"{"action":"install"}"#), Ok(SetupAction::Install));
        assert_eq!(parse_action(br#"{"action":"cancel"}"#), Ok(SetupAction::Cancel));
        assert!(parse_action(br#"{"action":"format"}"#).unwrap_err().contains("unknown action"));
        assert!(parse_action(b"nope").is_err());
    }
}
