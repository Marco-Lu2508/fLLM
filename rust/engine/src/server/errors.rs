//! Error bodies of every API dialect. The OpenAI shape is the default; the Anthropic Messages routes
//! and the Ollama routes answer in their own shapes, because their SDKs parse them.
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

/// OpenAI `error.type` of a status.
pub fn openai_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        429 => "rate_limit_error",
        s if s >= 500 => "server_error",
        _ => "invalid_request_error",
    }
}

/// The `code` OpenAI clients see when the handler did not name a more specific one.
pub fn default_code(status: StatusCode) -> Option<&'static str> {
    match status.as_u16() {
        401 => Some("invalid_api_key"),
        404 => Some("not_found"),
        405 => Some("method_not_allowed"),
        408 | 504 => Some("request_timeout"),
        413 => Some("request_too_large"),
        415 => Some("unsupported_media_type"),
        429 => Some("rate_limit_exceeded"),
        503 => Some("service_unavailable"),
        _ => None,
    }
}

/// `{"error": {"message", "type", "param", "code"}}`.
pub fn openai_body(status: StatusCode, message: &str, code: Option<&str>, param: Option<&str>) -> Value {
    json!({"error": {
        "message": message,
        "type": openai_type(status),
        "param": param,
        "code": code.or_else(|| default_code(status)),
    }})
}

pub fn openai(status: StatusCode, message: &str, code: Option<&str>, param: Option<&str>) -> Response {
    (status, Json(openai_body(status, message, code, param))).into_response()
}

fn anthropic_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 413 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        504 => "timeout_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    OpenAi,
    Anthropic,
    Ollama,
}

impl Dialect {
    pub fn of_path(path: &str) -> Self {
        if path.starts_with("/v1/messages") {
            Self::Anthropic
        } else if path.starts_with("/api/") {
            Self::Ollama
        } else {
            Self::OpenAi
        }
    }
}

/// An error response for `path`, in the shape the clients of that route expect.
pub fn for_path(path: &str, status: StatusCode, message: &str, code: Option<&str>, request_id: &str) -> Response {
    match Dialect::of_path(path) {
        Dialect::Anthropic => (
            status,
            Json(json!({"type": "error", "error": {"type": anthropic_type(status), "message": message}, "request_id": request_id})),
        )
            .into_response(),
        Dialect::Ollama => (status, Json(json!({"error": message}))).into_response(),
        Dialect::OpenAi => {
            let mut body = openai_body(status, message, code, None);
            body["error"]["request_id"] = json!(request_id);
            (status, Json(body)).into_response()
        }
    }
}
