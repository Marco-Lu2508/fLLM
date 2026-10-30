//! OpenAI Responses API: `POST /v1/responses`, `GET` and `DELETE /v1/responses/{id}`.
use super::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Mutex, OnceLock},
};

/// A stored response and the chat history that produced it, for `previous_response_id`.
struct Stored {
    response: Value,
    history: Vec<Value>,
}

#[derive(Default)]
struct Store {
    order: VecDeque<String>,
    items: HashMap<String, Stored>,
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(Store::default()))
}

/// Number of responses kept in memory (`FLLM_RESPONSE_STORE_SIZE`, default 256).
fn capacity() -> usize {
    std::env::var("FLLM_RESPONSE_STORE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

fn remember(response: &Value, history: Vec<Value>) {
    let Some(id) = response["id"].as_str() else {
        return;
    };
    let mut store = security::lock(store());
    store.order.push_back(id.to_owned());
    store.items.insert(
        id.to_owned(),
        Stored {
            response: response.clone(),
            history,
        },
    );
    while store.order.len() > capacity() {
        if let Some(old) = store.order.pop_front() {
            store.items.remove(&old);
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    api_error(status, message)
}

fn content_parts(content: &Value, role: &str) -> Result<Value, String> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(parts) => {
            let mut out: Vec<Value> = Vec::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        out.push(json!({"type":"text","text":part["text"]}))
                    }
                    Some("input_image") => {
                        let url = part["image_url"]
                            .as_str()
                            .or_else(|| part["image_url"]["url"].as_str())
                            .ok_or("input_image requires image_url")?;
                        out.push(json!({"type":"image_url","image_url":{"url":url}}));
                    }
                    Some("refusal") => out.push(json!({"type":"text","text":part["refusal"]})),
                    Some(other) => return Err(format!("unsupported content part type {other:?}")),
                    None => return Err("content part without type".into()),
                }
            }
            if role == "assistant" {
                Ok(Value::String(plain_text(&Value::Array(out))))
            } else {
                Ok(user_content(out))
            }
        }
        _ => Err("message content must be a string or an array".into()),
    }
}

fn append_call(messages: &mut Vec<Value>, call: Value) {
    if let Some(last) = messages.last_mut() {
        if last["role"] == "assistant" && last.get("tool_calls").is_some() {
            if let Some(calls) = last["tool_calls"].as_array_mut() {
                calls.push(call);
                return;
            }
        }
    }
    messages.push(json!({"role":"assistant","content":"","tool_calls":[call]}));
}

/// Rewrites a Responses request into a chat-completions body plus the history that precedes
/// the new input (used to store the exchange).
pub(super) fn to_chat_body(body: &Value, history: Vec<Value>) -> Result<(Value, Vec<Value>), String> {
    let mut input_messages: Vec<Value> = Vec::new();
    match body.get("input") {
        Some(Value::String(text)) => input_messages.push(json!({"role":"user","content":text})),
        Some(Value::Array(items)) => {
            for item in items {
                let kind = item["type"]
                    .as_str()
                    .unwrap_or(if item.get("role").is_some() { "message" } else { "" });
                match kind {
                    "message" => {
                        let role = match item["role"].as_str().ok_or("message without role")? {
                            "developer" => "system",
                            other => other,
                        };
                        let content = content_parts(&item["content"], role)?;
                        input_messages.push(json!({"role":role,"content":content}));
                    }
                    "function_call" => append_call(
                        &mut input_messages,
                        json!({
                            "id": item["call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": item["arguments"]},
                        }),
                    ),
                    "function_call_output" => input_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": match &item["output"] {
                            Value::String(s) => s.clone(),
                            other => plain_text(other),
                        },
                    })),
                    "reasoning" => {}
                    other => return Err(format!("unsupported input item type {other:?}")),
                }
            }
        }
        _ => return Err("input: Field required".into()),
    }
    let mut messages: Vec<Value> = Vec::new();
    if let Some(text) = body["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role":"system","content":text}));
    }
    messages.extend(history.iter().cloned());
    messages.extend(input_messages.iter().cloned());
    let mut chat = json!({"messages": messages});
    if let Some(model) = body.get("model") {
        chat["model"] = model.clone();
    }
    for (from, to) in [
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stream", "stream"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        // sampling extras that clients such as LiteLLM or the OpenAI SDK's `extra_body` pass through
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("seed", "seed"),
        ("stop", "stop"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("repetition_penalty", "repetition_penalty"),
        ("logit_bias", "logit_bias"),
        ("user", "user"),
    ] {
        if let Some(value) = body.get(from).filter(|v| !v.is_null()) {
            chat[to] = value.clone();
        }
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let functions: Vec<Value> = tools
            .iter()
            .filter(|t| t["type"] == "function")
            .map(|t| {
                json!({"type":"function","function":{
                    "name": t["name"],
                    "description": t["description"].as_str().unwrap_or_default(),
                    "parameters": t["parameters"],
                }})
            })
            .collect();
        if !functions.is_empty() {
            chat["tools"] = Value::Array(functions);
        }
    }
    if chat.get("tools").is_some() {
        match &body["tool_choice"] {
            Value::String(choice) => chat["tool_choice"] = json!(choice),
            Value::Object(object) if object.get("type") == Some(&json!("function")) => {
                chat["tool_choice"] = json!({"type":"function","function":{"name":object["name"]}})
            }
            _ => {}
        }
    }
    let format = &body["text"]["format"];
    match format["type"].as_str() {
        Some("json_schema") => {
            chat["response_format"] = json!({"type":"json_schema","json_schema":{
                "name": format["name"].as_str().unwrap_or("response"),
                "schema": format["schema"],
                "strict": format["strict"],
            }})
        }
        Some("json_object") => chat["response_format"] = json!({"type":"json_object"}),
        _ => {}
    }
    let mut kwargs = body
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(effort) = body["reasoning"]["effort"].as_str() {
        kwargs.entry("reasoning_effort").or_insert(json!(effort));
    }
    if !kwargs.is_empty() {
        chat["chat_template_kwargs"] = Value::Object(kwargs);
    }
    if chat["stream"].as_bool() == Some(true) {
        chat["stream_options"] = json!({"include_usage":true});
    }
    // `max_output_tokens` is a ceiling here, not a promise: cut it to what the context still holds
    chat["x_fllm_clamp_max_tokens"] = json!(true);
    Ok((chat, input_messages))
}

struct Output {
    items: Vec<Value>,
    /// the assistant turn in chat format, kept for `previous_response_id`
    turn: Value,
}

fn output_items(message: &Value) -> Output {
    let mut items: Vec<Value> = Vec::new();
    if let Some(text) = message["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
        items.push(json!({"id":new_id("rs"),"type":"reasoning","summary":[],
                          "content":[{"type":"reasoning_text","text":text}]}));
    }
    let text = message["content"].as_str().unwrap_or_default();
    let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
    if !text.is_empty() || calls.is_empty() {
        items.push(json!({"id":new_id("msg"),"type":"message","status":"completed","role":"assistant",
            "content":[{"type":"output_text","text":text,"annotations":[],"logprobs":[]}]}));
    }
    for call in &calls {
        items.push(json!({"id":new_id("fc"),"type":"function_call","status":"completed",
            "call_id":call["id"],"name":call["function"]["name"],
            "arguments":call["function"]["arguments"]}));
    }
    let mut turn = json!({"role":"assistant","content":text});
    if !calls.is_empty() {
        turn["tool_calls"] = Value::Array(calls);
    }
    Output { items, turn }
}

fn usage(chat_usage: &Value) -> Value {
    let input = chat_usage["prompt_tokens"].as_u64().unwrap_or(0);
    let output = chat_usage["completion_tokens"].as_u64().unwrap_or(0);
    json!({"input_tokens":input,"input_tokens_details":{"cached_tokens":chat_usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0)},
           "output_tokens":output,"output_tokens_details":{"reasoning_tokens":chat_usage["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0)},
           "total_tokens":input + output})
}

#[allow(clippy::too_many_arguments)]
fn response_object(
    id: &str,
    created: u64,
    body: &Value,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Value,
    finish: &str,
) -> Value {
    let incomplete = (status == "completed" && finish == "length")
        .then(|| json!({"reason":"max_output_tokens"}));
    json!({
        "id": id, "object": "response", "created_at": created,
        "status": if incomplete.is_some() { "incomplete" } else { status },
        "error": Value::Null,
        "incomplete_details": incomplete.unwrap_or(Value::Null),
        "instructions": body["instructions"],
        "max_output_tokens": body["max_output_tokens"],
        "model": model,
        "output": output,
        "parallel_tool_calls": body["parallel_tool_calls"].as_bool().unwrap_or(true),
        "previous_response_id": body["previous_response_id"],
        "reasoning": {"effort": body["reasoning"]["effort"], "summary": Value::Null},
        "store": body["store"].as_bool().unwrap_or(true),
        "temperature": body["temperature"].as_f64().unwrap_or(1.0),
        "text": {"format": body["text"]["format"].as_object().map_or(json!({"type":"text"}), |_| body["text"]["format"].clone())},
        "tool_choice": if body["tool_choice"].is_null() { json!("auto") } else { body["tool_choice"].clone() },
        "tools": body["tools"].as_array().cloned().unwrap_or_default(),
        "top_p": body["top_p"].as_f64().unwrap_or(1.0),
        "truncation": "disabled",
        "usage": usage,
        "user": body["user"],
        "metadata": body["metadata"].as_object().map_or(json!({}), |_| body["metadata"].clone()),
    })
}

fn sse(name: &str, sequence: &mut u64, mut data: Value) -> Event {
    data["type"] = json!(name);
    data["sequence_number"] = json!(*sequence);
    *sequence += 1;
    Event::default().event(name).data(data.to_string())
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Reasoning,
    Message,
    Call,
}

struct Open {
    kind: Kind,
    index: usize,
    item_id: String,
    call_id: String,
    name: String,
    text: String,
}

impl Open {
    /// Events that finish the open item, and the finished item itself.
    fn close(&self, sequence: &mut u64) -> (Vec<Event>, Value) {
        let mut events = Vec::new();
        let item = match self.kind {
            Kind::Reasoning => {
                events.push(sse("response.reasoning_text.done", sequence, json!({
                    "item_id": self.item_id, "output_index": self.index, "content_index": 0,
                    "text": self.text})));
                json!({"id":self.item_id,"type":"reasoning","summary":[],
                       "content":[{"type":"reasoning_text","text":self.text}]})
            }
            Kind::Message => {
                events.push(sse("response.output_text.done", sequence, json!({
                    "item_id": self.item_id, "output_index": self.index, "content_index": 0,
                    "text": self.text, "logprobs": []})));
                let part = json!({"type":"output_text","text":self.text,"annotations":[],"logprobs":[]});
                events.push(sse("response.content_part.done", sequence, json!({
                    "item_id": self.item_id, "output_index": self.index, "content_index": 0,
                    "part": part})));
                json!({"id":self.item_id,"type":"message","status":"completed","role":"assistant",
                       "content":[part]})
            }
            Kind::Call => {
                events.push(sse("response.function_call_arguments.done", sequence, json!({
                    "item_id": self.item_id, "output_index": self.index, "name": self.name,
                    "arguments": self.text})));
                json!({"id":self.item_id,"type":"function_call","status":"completed",
                       "call_id":self.call_id,"name":self.name,"arguments":self.text})
            }
        };
        events.push(sse("response.output_item.done", sequence, json!({
            "output_index": self.index, "item": item})));
        (events, item)
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_events(
    chunks: ChunkStream,
    body: Value,
    model: String,
    id: String,
    created: u64,
    history: Vec<Value>,
    store_it: bool,
) -> Response {
    let events = async_stream::stream! {
        let mut chunks = chunks;
        let mut sequence = 0u64;
        let skeleton = response_object(&id, created, &body, &model, "in_progress", vec![], Value::Null, "stop");
        yield Ok::<Event, Infallible>(sse("response.created", &mut sequence, json!({"response":skeleton})));
        yield Ok(sse("response.in_progress", &mut sequence, json!({"response":skeleton})));
        let mut open: Option<Open> = None;
        let mut done: Vec<Value> = Vec::new();
        let mut turn_text = String::new();
        let mut turn_calls: Vec<Value> = Vec::new();
        let mut finish = String::from("stop");
        let mut chat_usage = Value::Null;
        while let Some(chunk) = chunks.next().await {
            let value = match chunk {
                Chunk::Data(value) => value,
                Chunk::Error(message) => {
                    yield Ok(sse("error", &mut sequence, json!({"code":"server_error","message":message,"param":Value::Null})));
                    return;
                }
            };
            if !value["usage"].is_null() {
                chat_usage = value["usage"].clone();
            }
            let choice = &value["choices"][0];
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish = reason.to_owned();
            }
            let delta = &choice["delta"];
            let mut want: Vec<(Kind, String, String, String)> = Vec::new(); // kind, text, call_id, name
            if let Some(text) = delta["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
                want.push((Kind::Reasoning, text.to_owned(), String::new(), String::new()));
            }
            if let Some(text) = delta["content"].as_str().filter(|s| !s.is_empty()) {
                want.push((Kind::Message, text.to_owned(), String::new(), String::new()));
            }
            if let Some(calls) = delta["tool_calls"].as_array() {
                for call in calls {
                    let call_id = call["id"].as_str().unwrap_or_default().to_owned();
                    let name = call["function"]["name"].as_str().unwrap_or_default().to_owned();
                    let arguments = call["function"]["arguments"].as_str().unwrap_or_default().to_owned();
                    want.push((Kind::Call, arguments, call_id, name));
                }
            }
            for (kind, text, call_id, name) in want {
                let starts_call = kind == Kind::Call && !call_id.is_empty();
                let reuse = open.as_ref().is_some_and(|o| o.kind == kind) && !starts_call;
                if !reuse {
                    if let Some(old) = open.take() {
                        let (closing, item) = old.close(&mut sequence);
                        for event in closing { yield Ok(event); }
                        done.push(item);
                    }
                    let index = done.len();
                    let prefix = match kind { Kind::Reasoning => "rs", Kind::Message => "msg", Kind::Call => "fc" };
                    let item_id = new_id(prefix);
                    let item = match kind {
                        Kind::Reasoning => json!({"id":item_id,"type":"reasoning","summary":[],"content":[]}),
                        Kind::Message => json!({"id":item_id,"type":"message","status":"in_progress","role":"assistant","content":[]}),
                        Kind::Call => json!({"id":item_id,"type":"function_call","status":"in_progress","call_id":call_id,"name":name,"arguments":""}),
                    };
                    yield Ok(sse("response.output_item.added", &mut sequence, json!({"output_index":index,"item":item})));
                    if kind == Kind::Message {
                        yield Ok(sse("response.content_part.added", &mut sequence, json!({
                            "item_id": item_id, "output_index": index, "content_index": 0,
                            "part": {"type":"output_text","text":"","annotations":[],"logprobs":[]}})));
                    }
                    open = Some(Open { kind, index, item_id, call_id: call_id.clone(), name: name.clone(), text: String::new() });
                    if kind == Kind::Call {
                        turn_calls.push(json!({"id":call_id,"type":"function","function":{"name":name,"arguments":""}}));
                    }
                }
                if text.is_empty() { continue; }
                let Some(current) = open.as_mut() else { continue };
                current.text.push_str(&text);
                let (event_name, extra) = match kind {
                    Kind::Reasoning => ("response.reasoning_text.delta", json!({"content_index":0})),
                    Kind::Message => {
                        turn_text.push_str(&text);
                        ("response.output_text.delta", json!({"content_index":0,"logprobs":[]}))
                    }
                    Kind::Call => {
                        if let Some(last) = turn_calls.last_mut() {
                            let joined = format!("{}{}", last["function"]["arguments"].as_str().unwrap_or_default(), text);
                            last["function"]["arguments"] = json!(joined);
                        }
                        ("response.function_call_arguments.delta", json!({}))
                    }
                };
                let mut data = extra;
                data["item_id"] = json!(current.item_id);
                data["output_index"] = json!(current.index);
                data["delta"] = json!(text);
                yield Ok(sse(event_name, &mut sequence, data));
            }
        }
        if let Some(old) = open.take() {
            let (closing, item) = old.close(&mut sequence);
            for event in closing { yield Ok(event); }
            done.push(item);
        }
        let mut turn = json!({"role":"assistant","content":turn_text});
        if !turn_calls.is_empty() {
            turn["tool_calls"] = Value::Array(turn_calls);
        }
        let response = response_object(&id, created, &body, &model, "completed", done, usage(&chat_usage), &finish);
        if store_it {
            let mut full = history;
            full.push(turn);
            remember(&response, full);
        }
        let name = if response["status"] == "incomplete" { "response.incomplete" } else { "response.completed" };
        yield Ok(sse(name, &mut sequence, json!({"response":response})));
    };
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

pub(in crate::server) async fn create(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let mut history: Vec<Value> = Vec::new();
    if let Some(previous) = body["previous_response_id"].as_str() {
        let found = security::with_lock(store(), |s| s.items.get(previous).map(|stored| stored.history.clone()));
        match found {
            Some(h) => history = h,
            None => {
                return error(StatusCode::NOT_FOUND, &format!("previous response {previous} not found"))
            }
        }
    }
    let (chat, input_messages) = match to_chat_body(&body, history.clone()) {
        Ok(pair) => pair,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let stream = chat["stream"] == true;
    let model = body["model"].as_str().unwrap_or(&state.model_name).to_owned();
    let store_it = body["store"].as_bool().unwrap_or(true);
    let mut full_history = history;
    full_history.extend(input_messages);
    let id = new_id("resp");
    let created = unix_now();
    let response = chat_completions(State(state), Json(chat)).await;
    if !response.status().is_success() {
        let (status, message) = error_message(response).await;
        return error(status, &message);
    }
    if stream {
        return stream_events(chunks(response), body, model, id, created, full_history, store_it);
    }
    let chat = match read_json(response).await {
        Ok(chat) => chat,
        Err((status, message)) => return error(status, &message),
    };
    let output = output_items(&chat["choices"][0]["message"]);
    let finish = chat["choices"][0]["finish_reason"].as_str().unwrap_or("stop");
    let response = response_object(
        &id, created, &body, &model, "completed", output.items, usage(&chat["usage"]), finish,
    );
    if store_it {
        full_history.push(output.turn);
        remember(&response, full_history);
    }
    Json(response).into_response()
}

pub(in crate::server) async fn get_one(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let found = security::with_lock(store(), |s| s.items.get(&id).map(|s| s.response.clone()));
    match found {
        Some(response) => Json(response).into_response(),
        None => error(StatusCode::NOT_FOUND, &format!("response {id} not found")),
    }
}

pub(in crate::server) async fn delete_one(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let removed = security::with_lock(store(), |s| {
        s.order.retain(|o| o != &id);
        s.items.remove(&id).is_some()
    });
    if removed {
        Json(json!({"id":id,"object":"response","deleted":true})).into_response()
    } else {
        error(StatusCode::NOT_FOUND, &format!("response {id} not found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_become_chat_messages() {
        let body = json!({
            "model": "m", "instructions": "be brief", "max_output_tokens": 32,
            "tools": [{"type":"function","name":"f","description":"d","parameters":{"type":"object"}}],
            "tool_choice": {"type":"function","name":"f"},
            "text": {"format":{"type":"json_schema","name":"x","schema":{"type":"object"}}},
            "reasoning": {"effort":"low"},
            "input": [
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
                {"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},
                {"type":"function_call_output","call_id":"c1","output":"42"},
            ],
        });
        let (chat, input) = to_chat_body(&body, vec![]).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages[0], json!({"role":"system","content":"be brief"}));
        assert_eq!(messages[1], json!({"role":"user","content":"hi"}));
        assert_eq!(messages[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(messages[3], json!({"role":"tool","tool_call_id":"c1","content":"42"}));
        assert_eq!(input.len(), 3);
        assert_eq!(chat["max_tokens"], 32);
        assert_eq!(chat["tool_choice"]["function"]["name"], "f");
        assert_eq!(chat["response_format"]["json_schema"]["name"], "x");
        assert_eq!(chat["chat_template_kwargs"]["reasoning_effort"], "low");
    }

    #[test]
    fn string_input_and_errors() {
        let (chat, _) = to_chat_body(&json!({"input":"hello"}), vec![]).unwrap();
        assert_eq!(chat["messages"][0], json!({"role":"user","content":"hello"}));
        assert!(to_chat_body(&json!({}), vec![]).is_err());
        assert!(to_chat_body(&json!({"input":[{"type":"mystery"}]}), vec![]).is_err());
    }

    #[test]
    fn reply_becomes_output_items() {
        let message = json!({"reasoning_content":"think","content":"ok",
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]});
        let output = output_items(&message);
        assert_eq!(output.items[0]["type"], "reasoning");
        assert_eq!(output.items[1]["content"][0]["text"], "ok");
        assert_eq!(output.items[2]["call_id"], "call_1");
        assert_eq!(output.turn["tool_calls"][0]["id"], "call_1");
    }
}
