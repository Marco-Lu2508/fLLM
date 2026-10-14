//! llama.cpp server API: `POST /completion`, `POST /infill`, `GET /props`, `GET /slots`,
//! `POST /apply-template`, `POST /embedding`, `POST /embeddings`, `POST /tokenize` and
//! `POST /detokenize`.
//!
//! Completion requests are rewritten into the OpenAI text-completions body and run through
//! `completions`, so sampling, stop strings, grammars and JSON schemas behave exactly like
//! `/v1/completions`; only the request field names and the response shape differ.
use super::*;

/// llama.cpp error object: a numeric code plus a message and a type.
fn error(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "exception"
    };
    (
        status,
        Json(json!({"error":{"code":status.as_u16(),"message":message,"type":kind}})),
    )
        .into_response()
}

async fn convert_error(response: Response) -> Response {
    let (status, message) = error_message(response).await;
    error(status, &message)
}

/// Why generation ended, in llama.cpp's vocabulary.
struct Stop {
    stop_type: &'static str,
    eos: bool,
    limit: bool,
    word: bool,
}

fn stop_of(finish_reason: &str, has_stop_words: bool) -> Stop {
    match finish_reason {
        "length" => Stop {
            stop_type: "limit",
            eos: false,
            limit: true,
            word: false,
        },
        "stop" if has_stop_words => Stop {
            stop_type: "word",
            eos: false,
            limit: false,
            word: true,
        },
        "stop" => Stop {
            stop_type: "eos",
            eos: true,
            limit: false,
            word: false,
        },
        _ => Stop {
            stop_type: "none",
            eos: false,
            limit: false,
            word: false,
        },
    }
}

fn stop_fields(stop: &Stop, into: &mut serde_json::Map<String, Value>) {
    into.insert("stop_type".into(), json!(stop.stop_type));
    into.insert("stopped_eos".into(), json!(stop.eos));
    into.insert("stopped_limit".into(), json!(stop.limit));
    into.insert("stopped_word".into(), json!(stop.word));
}

/// The four timing counters llama.cpp reports; the per-second rates are dropped.
fn timings_of(reply: &Value) -> Value {
    json!({
        "predicted_n": reply["timings"]["predicted_n"],
        "predicted_ms": reply["timings"]["predicted_ms"],
        "prompt_n": reply["timings"]["prompt_n"],
        "prompt_ms": reply["timings"]["prompt_ms"],
    })
}

/// llama.cpp accepts a plain string, token IDs, or a list of prompt pieces.
fn prompt_value(prompt: &Value) -> Result<Value, String> {
    match prompt {
        Value::String(_) | Value::Number(_) => Ok(prompt.clone()),
        Value::Array(items) if items.is_empty() => Err("prompt must not be empty".into()),
        Value::Array(items) if items.iter().all(|item| item.is_number()) => {
            Ok(Value::Array(items.clone()))
        }
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                match item {
                    Value::String(part) => text.push_str(part),
                    Value::Object(_) => {
                        let part = item
                            .get("content")
                            .and_then(Value::as_str)
                            .ok_or("prompt list entries must be strings or {content}")?;
                        text.push_str(part);
                    }
                    _ => return Err("prompt must be a string, token IDs or a list of pieces".into()),
                }
            }
            Ok(Value::String(text))
        }
        _ => Err("prompt must be a string, token IDs or a list of pieces".into()),
    }
}

/// `n_predict` is a token budget; llama.cpp's `-1` fills the rest of the context window.
fn budget(state: &AppState, prompt: &Value, n_predict: Option<&Value>) -> Result<usize, String> {
    let requested = match n_predict {
        None | Some(Value::Null) => -1,
        Some(value) => value
            .as_i64()
            .ok_or("n_predict must be an integer")?,
    };
    if requested >= 0 {
        return usize::try_from(requested).map_err(|_| "n_predict is too large".to_owned());
    }
    let used = match prompt {
        Value::Array(items) => items.len(),
        Value::String(text) => state.tokenizer.encode(text)?.len(),
        _ => 0,
    };
    Ok(state.max_context.saturating_sub(used).max(1))
}

/// Rewrites a `/completion` or `/infill` request into the OpenAI text-completions body.
fn to_completion_body(state: &AppState, body: &Value) -> Result<Value, String> {
    let prompt = prompt_value(
        body.get("prompt")
            .ok_or("prompt is required (use input_prefix/input_suffix for /infill)")?,
    )?;
    let mut chat = json!({"prompt": prompt, "stream": body["stream"].as_bool().unwrap_or(false)});
    chat["max_tokens"] = json!(budget(state, &prompt, body.get("n_predict"))?);
    for field in [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
    ] {
        if let Some(value) = body.get(field).filter(|value| !value.is_null()) {
            chat[field] = value.clone();
        }
    }
    if let Some(model) = body.get("model") {
        chat["model"] = model.clone();
    }
    if let Some(schema) = body.get("json_schema").filter(|s| !s.is_null()) {
        chat["response_format"] = json!({"type":"json_schema","json_schema":{
            "name": "response", "schema": schema, "strict": body["json_schema"]["strict"]
        }});
    }
    if let Some(grammar) = body.get("grammar").filter(|g| !g.is_null()) {
        chat["grammar"] = grammar.clone();
        if let Some(kind) = body.get("grammar_type") {
            chat["grammar_type"] = kind.clone();
        }
    }
    Ok(chat)
}

/// Rewrites a text-completions reply into the llama.cpp completion object.
fn to_completion(reply: &Value, has_stop_words: bool) -> Value {
    let stop = stop_of(
        reply["choices"][0]["finish_reason"].as_str().unwrap_or("stop"),
        has_stop_words,
    );
    let predicted = reply["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let mut out = serde_json::Map::new();
    out.insert("content".into(), json!(reply["choices"][0]["text"]));
    out.insert("stop".into(), json!(true));
    stop_fields(&stop, &mut out);
    out.insert("tokens_predicted".into(), json!(predicted));
    out.insert(
        "tokens_evaluated".into(),
        json!(reply["usage"]["prompt_tokens"].as_u64().unwrap_or(0)),
    );
    out.insert("tokens_cached".into(), json!(0));
    out.insert("timings".into(), timings_of(reply));
    Value::Object(out)
}

/// llama.cpp streams `data: {"content": …}` per chunk and a final object with `stop: true`.
fn stream_completion(stream: ChunkStream, has_stop_words: bool) -> Response {
    let events = async_stream::stream! {
        let mut stream = stream;
        while let Some(chunk) = stream.next().await {
            let value = match chunk {
                Chunk::Data(value) => value,
                Chunk::Error(message) => {
                    yield Ok::<Event, Infallible>(
                        Event::default().event("error").data(json!({"error":message}).to_string()),
                    );
                    return;
                }
            };
            let choice = &value["choices"][0];
            if let Some(text) = choice["text"].as_str().filter(|text| !text.is_empty()) {
                yield Ok(Event::default().data(json!({"content":text}).to_string()));
            }
            let Some(reason) = choice["finish_reason"].as_str() else {
                continue;
            };
            let stop = stop_of(reason, has_stop_words);
            let mut last = serde_json::Map::new();
            last.insert("content".into(), json!(""));
            last.insert("stop".into(), json!(true));
            stop_fields(&stop, &mut last);
            last.insert("tokens_predicted".into(), value["usage"]["completion_tokens"].clone());
            last.insert("tokens_evaluated".into(), value["usage"]["prompt_tokens"].clone());
            last.insert("tokens_cached".into(), json!(0));
            last.insert("timings".into(), timings_of(&value));
            yield Ok(Event::default().data(Value::Object(last).to_string()));
            return;
        }
    };
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

fn has_stop_words(body: &Value) -> bool {
    body.get("stop")
        .map(|stop| match stop {
            Value::String(text) => !text.is_empty(),
            Value::Array(items) => items
                .iter()
                .any(|item| item.as_str().is_some_and(|text| !text.is_empty())),
            _ => false,
        })
        .unwrap_or(false)
}

/// Runs a completion body through `/v1/completions` and rewrites the reply (or the SSE
/// stream) into llama.cpp's shape.
async fn run_completion(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: &Value,
    chat: Value,
    model: String,
) -> Response {
    let stream = chat["stream"] == true;
    let stops = has_stop_words(body);
    let response = completions(State(state), headers, Json(chat)).await;
    if !response.status().is_success() {
        return convert_error(response).await;
    }
    if stream {
        return stream_completion(chunks(response), stops);
    }
    match read_json(response).await {
        Ok(reply) => {
            let mut completion = to_completion(&reply, stops);
            completion["model"] = json!(model);
            Json(completion).into_response()
        }
        Err((status, message)) => error(status, &message),
    }
}

pub(in crate::server) async fn completion(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let model = body["model"].as_str().unwrap_or(&state.model_name).to_owned();
    let chat = match to_completion_body(&state, &body) {
        Ok(chat) => chat,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    run_completion(state, headers, &body, chat, model).await
}

/// Literal text of a FIM control token, so families with custom markers still render correctly.
fn marker(state: &AppState, id: u32) -> String {
    state
        .tokenizer
        .decode(&[id], false)
        .unwrap_or_default()
        .to_owned()
}

/// Builds the FIM prompt of `/infill`: repository context, then prefix, suffix and the middle
/// marker in the order the family was trained on.
fn to_infill_body(state: &AppState, body: &Value) -> Result<Value, String> {
    let prefix = body["input_prefix"].as_str().unwrap_or_default();
    let suffix = body["input_suffix"].as_str().unwrap_or_default();
    let extra = body["input_extra"].as_array();
    if (prefix.is_empty() && suffix.is_empty() && extra.is_none()) || body.get("prompt").is_some() {
        let prompt = body
            .get("prompt")
            .ok_or("input_prefix, input_suffix or prompt is required")?;
        let mut chat = json!({"prompt": prompt_value(prompt)?, "stream": body["stream"].as_bool().unwrap_or(false)});
        chat["max_tokens"] = json!(budget(state, &chat["prompt"], body.get("n_predict"))?);
        return Ok(chat);
    }
    let Some(fim) = state.tokenizer.fim() else {
        return Err("this model has no fill-in-the-middle tokens, so /infill is unavailable"
            .into());
    };
    let mut context = String::new();
    for file in extra.into_iter().flatten() {
        let text = file["text"].as_str().unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let name = file["filename"]
            .as_str()
            .or_else(|| file["file_name"].as_str())
            .unwrap_or("file");
        match fim.sep {
            Some(sep) => {
                context.push_str(&marker(state, sep));
                context.push_str(name);
            }
            None => {
                context.push_str(name);
            }
        }
        context.push('\n');
        context.push_str(text);
    }
    let pre = marker(state, fim.prefix);
    let suf = marker(state, fim.suffix);
    let mid = marker(state, fim.middle);
    // StarCoder2 and its derivatives were trained suffix-first (llama.cpp `FIM_SPM`).
    let suffix_first = state.model_name.to_lowercase().contains("starcoder");
    let prompt = if suffix_first {
        format!("{context}{suffix}{suf}{prefix}{pre}{mid}")
    } else {
        format!("{context}{prefix}{pre}{suffix}{suf}{mid}")
    };
    let mut chat = json!({"prompt": prompt, "stream": body["stream"].as_bool().unwrap_or(false)});
    chat["max_tokens"] = json!(budget(state, &chat["prompt"], body.get("n_predict"))?);
    Ok(chat)
}

pub(in crate::server) async fn infill(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let fim_missing = state.tokenizer.fim().is_none()
        && body.get("prompt").is_none()
        && body["input_prefix"].as_str().unwrap_or_default().is_empty()
        && body["input_suffix"].as_str().unwrap_or_default().is_empty();
    let model = body["model"].as_str().unwrap_or(&state.model_name).to_owned();
    let chat = match to_infill_body(&state, &body) {
        Ok(chat) => chat,
        Err(message) => {
            let status = if fim_missing {
                StatusCode::NOT_IMPLEMENTED
            } else {
                StatusCode::BAD_REQUEST
            };
            return error(status, &message);
        }
    };
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    run_completion(state, headers, &body, chat, model).await
}

/// Sequences the scheduler currently runs, with idle slots filled up to the observed capacity.
fn slots(state: &AppState) -> (usize, usize) {
    let running = state.metrics.slots_running.load(Ordering::Relaxed) as usize;
    let waiting = state.metrics.slots_waiting.load(Ordering::Relaxed) as usize;
    (running, (running + waiting).max(1))
}

pub(in crate::server) async fn props(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let defaults = &state.sampling_defaults;
    let token = |ids: &[u32]| {
        ids.first()
            .and_then(|id| state.tokenizer.decode(&[*id], false).ok())
            .unwrap_or_default()
    };
    Json(json!({
        "default_generation_settings": {
            "n_predict": -1,
            "temperature": state.default_temperature.unwrap_or(defaults.temperature),
            "top_k": state.default_top_k.unwrap_or(defaults.top_k),
            "top_p": state.default_top_p.unwrap_or(defaults.top_p),
            "min_p": defaults.min_p,
            "repeat_penalty": defaults.repetition_penalty,
            "repeat_last_n": defaults.repeat_last_n,
            "presence_penalty": defaults.presence_penalty,
            "frequency_penalty": defaults.frequency_penalty,
            "seed": defaults.seed,
        },
        "total_slots": slots(&state).1,
        "model_path": state.model_name,
        "chat_template": state.chat_template,
        "bos_token": token(state.tokenizer.bos_ids()),
        "eos_token": token(state.tokenizer.eos_ids()),
        "build_info": format!("fllm-rs {}", env!("CARGO_PKG_VERSION")),
    }))
    .into_response()
}

pub(in crate::server) async fn slot_list(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let (running, total) = slots(&state);
    let list: Vec<Value> = (0..total)
        .map(|id| {
            json!({
                "id": id,
                "state": if id < running { "processing" } else { "idle" },
                "n_ctx": state.max_context,
                "prompt": "",
                "next_token": Value::Null,
            })
        })
        .collect();
    Json(list).into_response()
}

pub(in crate::server) async fn apply_template(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let messages = body
        .get("messages")
        .cloned()
        .ok_or(())
        .map_err(|()| error(StatusCode::BAD_REQUEST, "messages is required"));
    let messages = match messages {
        Ok(messages) => messages,
        Err(response) => return response,
    };
    let kwargs = body
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned();
    match render_messages(&state, messages, body.get("tools").cloned(), kwargs) {
        Ok(prompt) => Json(json!({"prompt": prompt})).into_response(),
        Err(message) => error(StatusCode::BAD_REQUEST, &message),
    }
}

/// llama.cpp sends `{"content": "…"}` and expects `[{index, embedding:[[floats]]}]`.
async fn embedding_request(state: Arc<AppState>, headers: HeaderMap, body: Value) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let content = body
        .get("content")
        .or_else(|| body.get("input"))
        .or_else(|| body.get("prompt"))
        .cloned()
        .ok_or(())
        .map_err(|()| error(StatusCode::BAD_REQUEST, "content is required"));
    let content = match content {
        Ok(content) => content,
        Err(response) => return response,
    };
    let mut inner = json!({"input": content});
    for field in ["pooling", "encoding_format"] {
        if let Some(value) = body.get(field) {
            inner[field] = value.clone();
        }
    }
    let response = super::embeddings(State(state), headers, Json(inner)).await;
    if !response.status().is_success() {
        return convert_error(response).await;
    }
    match read_json(response).await {
        Ok(reply) => {
            let vectors = reply
                .get("embeddings")
                .or_else(|| reply.get("embedding"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let data: Vec<Value> = vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| json!({"index": index, "embedding": [vector]}))
                .collect();
            Json(data).into_response()
        }
        Err((status, message)) => error(status, &message),
    }
}

pub(in crate::server) async fn embedding(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    embedding_request(state, headers, body).await
}

pub(in crate::server) async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    embedding_request(state, headers, body).await
}

pub(in crate::server) async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let Some(text) = body.get("content").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "content must be a string");
    };
    let add_special = body
        .get("add_special")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let ids = if add_special {
        state.tokenizer.encode(text)
    } else {
        state.tokenizer.encode_without_bos(text)
    };
    let ids = match ids {
        Ok(ids) => ids,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let tokens = if body
        .get("with_pieces")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        ids.iter()
            .map(|id| {
                json!({"id": id, "piece": state.tokenizer.decode(&[*id], false).unwrap_or_default()})
            })
            .collect::<Vec<_>>()
    } else {
        ids.iter().map(|id| json!(id)).collect::<Vec<_>>()
    };
    Json(json!({"tokens": tokens, "count": ids.len()})).into_response()
}

pub(in crate::server) async fn detokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let headers = with_bearer(&headers);
    if let Err(response) = authorize(&state, &headers) {
        return convert_error(response).await;
    }
    let skip_special = body
        .get("skip_special")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    match input_tokens(&state, body.get("tokens").unwrap_or(&Value::Null)) {
        Ok(ids) => {
            let ids: Vec<u32> = ids.into_iter().filter_map(|id| u32::try_from(id).ok()).collect();
            match state.tokenizer.decode(&ids, skip_special) {
                Ok(content) => Json(json!({"content": content})).into_response(),
                Err(message) => error(StatusCode::BAD_REQUEST, &message),
            }
        }
        Err(message) => error(StatusCode::BAD_REQUEST, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reasons_map_to_llama_cpp_flags() {
        let limit = stop_of("length", false);
        assert_eq!(limit.stop_type, "limit");
        assert!(limit.limit && !limit.eos && !limit.word);
        let word = stop_of("stop", true);
        assert_eq!(word.stop_type, "word");
        assert!(word.word);
        let eos = stop_of("stop", false);
        assert_eq!(eos.stop_type, "eos");
        assert!(eos.eos);
        assert_eq!(stop_of("tool_calls", false).stop_type, "none");
    }

    #[test]
    fn prompt_accepts_strings_token_ids_and_piece_lists() {
        assert_eq!(prompt_value(&json!("hello")).unwrap(), json!("hello"));
        assert_eq!(prompt_value(&json!([1, 2, 3])).unwrap(), json!([1, 2, 3]));
        assert_eq!(
            prompt_value(&json!(["a", {"content": "b"}])).unwrap(),
            json!("ab")
        );
        assert!(prompt_value(&json!([])).is_err());
        assert!(prompt_value(&json!([true])).is_err());
    }

    #[test]
    fn reply_becomes_completion_object() {
        let reply = json!({"choices":[{"text":"hi","finish_reason":"length"}],
            "usage":{"prompt_tokens":7,"completion_tokens":3},
            "timings":{"predicted_n":3,"predicted_ms":12.5,"prompt_n":7,"prompt_ms":4.5,
                       "predicted_per_second":1.0,"prompt_per_second":2.0}});
        let completion = to_completion(&reply, false);
        assert_eq!(completion["content"], "hi");
        assert_eq!(completion["stop"], true);
        assert_eq!(completion["stop_type"], "limit");
        assert_eq!(completion["stopped_limit"], true);
        assert_eq!(completion["tokens_predicted"], 3);
        assert_eq!(completion["tokens_evaluated"], 7);
        assert_eq!(completion["tokens_cached"], 0);
        assert_eq!(completion["timings"]["predicted_ms"], 12.5);
        assert!(completion["timings"].get("prompt_per_second").is_none());
    }

    #[test]
    fn timings_are_taken_from_the_final_stream_chunk() {
        let chunk = json!({"choices":[{"text":"","finish_reason":"stop"}],
            "usage":{"prompt_tokens":2,"completion_tokens":1},
            "timings":{"predicted_n":1,"predicted_ms":3.0,"prompt_n":2,"prompt_ms":1.0}});
        assert_eq!(timings_of(&chunk), json!({"predicted_n":1,"predicted_ms":3.0,
                                              "prompt_n":2,"prompt_ms":1.0}));
    }
}
