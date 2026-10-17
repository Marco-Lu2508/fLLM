//! vLLM-only routes: `GET /version`, `GET /is_sleeping` and `POST /v1/chat/completions/render`.
//! (`/tokenize` and `/detokenize` accept the vLLM request shapes next to llama.cpp's, see `llamacpp.rs`.)
use super::*;

/// `GET /version`: the server version, like vLLM's `{"version": "..."}`.
pub(in crate::server) async fn version() -> Response {
    Json(json!({"version": env!("CARGO_PKG_VERSION"), "engine": "fllm-rs"})).into_response()
}

/// `GET /is_sleeping`: fLLM has no sleep mode, the model is always awake.
pub(in crate::server) async fn is_sleeping() -> Response {
    Json(json!({"is_sleeping": false})).into_response()
}

/// `POST /v1/chat/completions/render`: what a chat request turns into before generation, without running the model.
/// The reply is vLLM's `[conversation, [engine_prompt]]`; the prompt is the chat-template output and the token ids
/// are exactly those `/v1/chat/completions` would feed the model.
pub(in crate::server) async fn render_chat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = state.resolve_model(&body) {
        return response;
    }
    let Some(messages) = body.get("messages").filter(|m| m.as_array().is_some_and(|m| !m.is_empty())) else {
        return api_error_with(StatusCode::BAD_REQUEST, "messages must be a non-empty array", None, Some("messages"));
    };
    let kwargs = body.get("chat_template_kwargs").and_then(Value::as_object).cloned();
    let prompt = match render_messages(&state, messages.clone(), body.get("tools").cloned(), kwargs) {
        Ok(prompt) => prompt,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, &message),
    };
    let ids = match state.tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, &message),
    };
    Json(json!([messages, [{"prompt": prompt, "prompt_token_ids": ids}]])).into_response()
}
