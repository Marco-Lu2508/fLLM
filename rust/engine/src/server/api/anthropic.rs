//! Anthropic Messages API: `POST /v1/messages`, `POST /v1/messages/count_tokens` and the
//! Anthropic shape of `GET /v1/models`.
use super::*;

/// `{type:"error", error:{type, message}}` with the status code Anthropic clients expect.
fn error(status: StatusCode, message: &str) -> Response {
    let kind = match status.as_u16() {
        400 | 413 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    };
    (
        status,
        Json(json!({"type":"error","error":{"type":kind,"message":message}})),
    )
        .into_response()
}

async fn convert_error(response: Response) -> Response {
    let (status, message) = error_message(response).await;
    error(status, &message)
}

pub(super) fn model_object(state: &AppState) -> Value {
    json!({"type":"model","id":state.model_name,"display_name":state.model_name,
           "created_at":iso_now()})
}

/// Anthropic-format model list (used when the request carries `anthropic-version`).
pub(in crate::server) fn models(state: &AppState) -> Response {
    let model = model_object(state);
    Json(json!({"data":[model],"has_more":false,"first_id":state.model_name,
                "last_id":state.model_name}))
    .into_response()
}

fn tool_result_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        other => plain_text(other),
    }
}

fn image_url(block: &Value) -> Option<String> {
    let source = &block["source"];
    match source["type"].as_str()? {
        "base64" => Some(format!(
            "data:{};base64,{}",
            source["media_type"].as_str().unwrap_or("image/png"),
            source["data"].as_str()?
        )),
        "url" => source["url"].as_str().map(str::to_owned),
        _ => None,
    }
}

/// Rewrites a Messages request into a chat-completions body.
pub(super) fn to_chat_body(body: &Value) -> Result<Value, String> {
    let mut messages: Vec<Value> = Vec::new();
    let system = plain_text(body.get("system").unwrap_or(&Value::Null));
    if !system.is_empty() {
        messages.push(json!({"role":"system","content":system}));
    }
    let input = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages: Field required")?;
    for message in input {
        let role = message["role"]
            .as_str()
            .filter(|r| matches!(*r, "user" | "assistant"))
            .ok_or("messages.role must be \"user\" or \"assistant\"")?;
        match &message["content"] {
            Value::String(text) => messages.push(json!({"role":role,"content":text})),
            Value::Array(blocks) => {
                let mut parts: Vec<Value> = Vec::new();
                let mut text = String::new();
                let mut calls: Vec<Value> = Vec::new();
                let mut results: Vec<Value> = Vec::new();
                for block in blocks {
                    match block["type"].as_str() {
                        Some("text") => {
                            let t = block["text"].as_str().unwrap_or_default();
                            parts.push(json!({"type":"text","text":t}));
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        Some("image") => {
                            let url = image_url(block).ok_or("unsupported image source")?;
                            parts.push(json!({"type":"image_url","image_url":{"url":url}}));
                        }
                        Some("tool_use") => calls.push(json!({
                            "id": block["id"],
                            "type": "function",
                            "function": {"name": block["name"], "arguments": block["input"].to_string()},
                        })),
                        Some("tool_result") => results.push(json!({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": tool_result_text(&block["content"]),
                        })),
                        // earlier reasoning is not replayed; unknown blocks are ignored
                        _ => {}
                    }
                }
                if role == "assistant" {
                    let mut out = json!({"role":"assistant","content":text});
                    if !calls.is_empty() {
                        out["tool_calls"] = Value::Array(calls);
                    }
                    messages.push(out);
                } else {
                    messages.extend(results);
                    if !parts.is_empty() {
                        messages.push(json!({"role":"user","content":user_content(parts)}));
                    }
                }
            }
            _ => return Err("messages.content must be a string or an array of blocks".into()),
        }
    }
    let mut chat = json!({"messages": messages});
    if let Some(model) = body.get("model") {
        chat["model"] = model.clone();
    }
    for (from, to) in [
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stream", "stream"),
        ("stop_sequences", "stop"),
    ] {
        if let Some(value) = body.get(from).filter(|v| !v.is_null()) {
            chat[to] = value.clone();
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let functions: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("input_schema").is_some())
            .map(|t| {
                json!({"type":"function","function":{
                    "name": t["name"],
                    "description": t["description"].as_str().unwrap_or_default(),
                    "parameters": t["input_schema"],
                }})
            })
            .collect();
        if !functions.is_empty() {
            chat["tools"] = Value::Array(functions);
        }
    }
    if chat.get("tools").is_some() {
        match body["tool_choice"]["type"].as_str() {
            Some("any") => chat["tool_choice"] = json!("required"),
            Some("none") => chat["tool_choice"] = json!("none"),
            Some("tool") => {
                chat["tool_choice"] =
                    json!({"type":"function","function":{"name":body["tool_choice"]["name"]}})
            }
            _ => {}
        }
    }
    match body["thinking"]["type"].as_str() {
        Some("enabled") => chat["chat_template_kwargs"] = json!({"enable_thinking":true}),
        Some("disabled") => chat["chat_template_kwargs"] = json!({"enable_thinking":false}),
        _ => {}
    }
    if chat["stream"].as_bool() == Some(true) {
        chat["stream_options"] = json!({"include_usage":true});
    }
    Ok(chat)
}

fn stop_reason(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

/// Rewrites a chat-completions reply into a Messages response.
pub(super) fn to_message(chat: &Value, model: &str) -> Value {
    let choice = &chat["choices"][0];
    let message = &choice["message"];
    let mut content: Vec<Value> = Vec::new();
    if let Some(reasoning) = message["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        content.push(json!({"type":"thinking","thinking":reasoning,"signature":""}));
    }
    if let Some(text) = message["content"].as_str().filter(|s| !s.is_empty()) {
        content.push(json!({"type":"text","text":text}));
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
            content.push(json!({
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({})),
            }));
        }
    }
    json!({
        "id": new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(choice["finish_reason"].as_str().unwrap_or("stop")),
        "stop_sequence": Value::Null,
        "usage": message_usage(&chat["usage"]),
    })
}

fn message_usage(usage: &Value) -> Value {
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    json!({"input_tokens":usage["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached),
        "output_tokens":usage["completion_tokens"].as_u64().unwrap_or(0),
        "cache_creation_input_tokens":0,"cache_read_input_tokens":cached,
        "output_tokens_details":{"reasoning_tokens":usage["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0)}})
}
fn event(name: &str, mut data: Value) -> Event {
    data["type"] = json!(name);
    Event::default().event(name).data(data.to_string())
}

#[derive(PartialEq, Clone, Copy)]
enum Open {
    None,
    Thinking,
    Text,
    Tool,
}

/// Rewrites the chat-completions stream into Messages events.
fn stream_events(chunks: ChunkStream, model: String) -> Response {
    let events = async_stream::stream! {
        let mut chunks = chunks;
        yield Ok::<Event, Infallible>(event("message_start", json!({"message":{
            "id": new_id("msg"), "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }})));
        yield Ok(event("ping", json!({})));
        let mut open = Open::None;
        let mut index = 0usize;
        let mut finish = String::from("stop");
        let mut final_usage = message_usage(&Value::Null);
        while let Some(chunk) = chunks.next().await {
            let value = match chunk {
                Chunk::Data(value) => value,
                Chunk::Error(message) => {
                    yield Ok(event("error", json!({"error":{"type":"api_error","message":message}})));
                    return;
                }
            };
            if !value["usage"].is_null() {
                final_usage = message_usage(&value["usage"]);
            }
            let choice = &value["choices"][0];
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish = reason.to_owned();
            }
            let delta = &choice["delta"];
            if let Some(text) = delta["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
                if open != Open::Thinking {
                    if open != Open::None {
                        yield Ok(event("content_block_stop", json!({"index":index})));
                        index += 1;
                    }
                    yield Ok(event("content_block_start", json!({"index":index,
                        "content_block":{"type":"thinking","thinking":"","signature":""}})));
                    open = Open::Thinking;
                }
                yield Ok(event("content_block_delta", json!({"index":index,
                    "delta":{"type":"thinking_delta","thinking":text}})));
            }
            if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
                if open != Open::Text {
                    if open != Open::None {
                        yield Ok(event("content_block_stop", json!({"index":index})));
                        index += 1;
                    }
                    yield Ok(event("content_block_start", json!({"index":index,
                        "content_block":{"type":"text","text":""}})));
                    open = Open::Text;
                }
                yield Ok(event("content_block_delta", json!({"index":index,
                    "delta":{"type":"text_delta","text":text}})));
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    if call["id"].as_str().is_some_and(|id| !id.is_empty()) {
                        if open != Open::None {
                            yield Ok(event("content_block_stop", json!({"index":index})));
                            index += 1;
                        }
                        yield Ok(event("content_block_start", json!({"index":index,
                            "content_block":{"type":"tool_use","id":call["id"],
                                             "name":call["function"]["name"],"input":{}}})));
                        open = Open::Tool;
                    }
                    if let Some(part) = call["function"]["arguments"].as_str().filter(|s| !s.is_empty()) {
                        if open == Open::Tool {
                            yield Ok(event("content_block_delta", json!({"index":index,
                                "delta":{"type":"input_json_delta","partial_json":part}})));
                        }
                    }
                }
            }
        }
        if open != Open::None {
            yield Ok(event("content_block_stop", json!({"index":index})));
        }
        yield Ok(event("message_delta", json!({
            "delta": {"stop_reason": stop_reason(&finish), "stop_sequence": Value::Null},
            "usage": final_usage,
        })));
        yield Ok(event("message_stop", json!({})));
    };
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

pub(in crate::server) async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let chat = match to_chat_body(&body) {
        Ok(chat) => chat,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let stream = chat["stream"] == true;
    let model = body["model"]
        .as_str()
        .unwrap_or(&state.model_name)
        .to_owned();
    let response = chat_completions(State(state), headers, Json(chat)).await;
    if !response.status().is_success() {
        return convert_error(response).await;
    }
    if stream {
        return stream_events(chunks(response), model);
    }
    match read_json(response).await {
        Ok(chat) => Json(to_message(&chat, &model)).into_response(),
        Err((status, message)) => error(status, &message),
    }
}

pub(in crate::server) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let chat = match to_chat_body(&body) {
        Ok(chat) => chat,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let kwargs = chat
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned();
    let prompt = match render_messages(
        &state,
        chat["messages"].clone(),
        chat.get("tools").cloned(),
        kwargs,
    ) {
        Ok(prompt) => prompt,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    match state.tokenizer.encode(&prompt) {
        Ok(ids) => Json(json!({"input_tokens": ids.len()})).into_response(),
        Err(message) => error(StatusCode::BAD_REQUEST, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_blocks_become_chat_messages() {
        let body = json!({
            "model": "m", "max_tokens": 64, "system": [{"type":"text","text":"be brief"}],
            "tools": [{"name":"get_weather","description":"d","input_schema":{"type":"object"}}],
            "tool_choice": {"type":"any"},
            "stop_sequences": ["END"],
            "messages": [
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"text","text":"checking"},
                    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Köln"}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"sunny"}]},
                    {"type":"text","text":"thanks"}]},
            ],
        });
        let chat = to_chat_body(&body).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages[0], json!({"role":"system","content":"be brief"}));
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Köln\"}"
        );
        assert_eq!(
            messages[3],
            json!({"role":"tool","tool_call_id":"toolu_1","content":"sunny"})
        );
        assert_eq!(messages[4], json!({"role":"user","content":"thanks"}));
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["stop"], json!(["END"]));
        assert_eq!(chat["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn reply_becomes_message_blocks() {
        let chat = json!({"choices":[{"finish_reason":"tool_calls","message":{
            "reasoning_content":"think","content":"ok",
            "tool_calls":[{"id":"call_1","type":"function",
                           "function":{"name":"f","arguments":"{\"a\":1}"}}]}}],
            "usage":{"prompt_tokens":5,"completion_tokens":7}});
        let message = to_message(&chat, "m");
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(message["content"][0]["type"], "thinking");
        assert_eq!(message["content"][1]["text"], "ok");
        assert_eq!(message["content"][2]["input"], json!({"a":1}));
        assert_eq!(message["usage"]["output_tokens"], 7);
    }

    #[test]
    fn invalid_requests_are_rejected() {
        assert!(to_chat_body(&json!({"messages":[{"role":"system","content":"x"}]})).is_err());
        assert!(to_chat_body(&json!({"model":"m"})).is_err());
    }
}
