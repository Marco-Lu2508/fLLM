//! Additional API surfaces (Anthropic Messages, OpenAI Responses) translated onto the chat
//! completions pipeline. Each surface rewrites its request into a chat-completions body, runs
//! `chat_completions` in process and rewrites the reply (or the SSE stream) into its own format,
//! so sampling, tools, reasoning and structured output behave identically everywhere.
use super::*;
use futures_util::StreamExt;
use std::pin::Pin;

pub(super) mod anthropic;
pub(super) mod batches;
pub(super) mod extract;
pub(super) mod llamacpp;
pub(super) mod mcp;
pub(super) mod ollama_extra;
pub(super) mod providers;
pub(super) mod responses;
pub(super) mod agent;
mod agent_jobs;
pub(super) mod agent_net;
mod agent_tools;
pub(super) mod studio;
pub(super) mod studio_ops;
pub(super) mod tgi;
pub(super) mod vllm;

/// One parsed event of the internal chat-completions SSE stream.
pub(super) enum Chunk {
    Data(Value),
    Error(String),
}

pub(super) type ChunkStream = Pin<Box<dyn futures_util::Stream<Item = Chunk> + Send>>;

/// Fresh identifier such as `msg_0123456789abcdef`.
pub(super) fn new_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{:012x}{:04x}", nanos & 0xffff_ffff_ffff, count & 0xffff)
}

/// Message text of an OpenAI-style error response, with its status.
pub(super) async fn error_message(response: Response) -> (StatusCode, String) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
    (status, message)
}

/// The JSON body of a successful non-streaming reply.
pub(super) async fn read_json(response: Response) -> Result<Value, (StatusCode, String)> {
    if !response.status().is_success() {
        return Err(error_message(response).await);
    }
    let bytes = axum::body::to_bytes(response.into_body(), 256 << 20)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Parses the SSE body of a streaming chat-completions reply into JSON chunks.
pub(super) fn chunks(response: Response) -> ChunkStream {
    Box::pin(async_stream::stream! {
        let mut body = response.into_body().into_data_stream();
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(piece) = body.next().await {
            let Ok(piece) = piece else {
                yield Chunk::Error("stream interrupted".into());
                return;
            };
            buffer.extend_from_slice(&piece);
            while let Some(end) = buffer.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buffer.drain(..end + 2).collect();
                let block = String::from_utf8_lossy(&block).replace("\r\n", "\n");
                let mut event = "";
                let mut data = String::new();
                for line in block.lines() {
                    if let Some(name) = line.strip_prefix("event:") {
                        event = name.trim();
                    } else if let Some(text) = line.strip_prefix("data:") {
                        if !data.is_empty() {
                            data.push('\n');
                        }
                        data.push_str(text.trim_start());
                    }
                }
                if data == "[DONE]" {
                    return;
                }
                if data.is_empty() {
                    continue;
                }
                let value = serde_json::from_str::<Value>(&data).unwrap_or(Value::Null);
                if event == "error" {
                    let message = value["error"].as_str().map_or_else(
                        || value["error"]["message"].as_str().unwrap_or("generation failed").to_owned(),
                        str::to_owned,
                    );
                    yield Chunk::Error(message);
                    return;
                }
                if !value.is_null() {
                    yield Chunk::Data(value);
                }
            }
        }
    })
}

/// Text of a message `content` that is a string or a list of `{type:"text"}` blocks.
pub(super) fn plain_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                Value::String(s) => Some(s.as_str()),
                _ => b["text"].as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Collapses user content parts into a plain string when they are text only.
pub(super) fn user_content(parts: Vec<Value>) -> Value {
    if parts.iter().all(|p| p["type"] == "text") {
        Value::String(
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        Value::Array(parts)
    }
}

/// `GET /v1/models/{id}` in the OpenAI or (with `anthropic-version`) the Anthropic shape.
pub(super) async fn model_by_id(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if id.is_empty() || !state.knows_model(&id) {
        return model_not_found(&id);
    }
    if headers.contains_key("anthropic-version") {
        return Json(anthropic::model_object(&state)).into_response();
    }
    Json(json!({"id":id,"object":"model","created":unix_now(),"owned_by":"fllm","root":state.model_name,"parent":null,"max_model_len":state.max_context}))
        .into_response()
}
