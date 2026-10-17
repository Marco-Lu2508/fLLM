//! Text Generation Inference server API: `POST /generate`, `POST /generate_stream` and
//! `GET /info`.
//!
//! Requests are rewritten into the OpenAI text-completions body and run through `completions`,
//! so sampling, stop strings and grammars behave like `/v1/completions`. `POST /generate` also
//! serves SGLang clients: bodies carrying `text` or `input_ids` are dispatched to the SGLang
//! handler in `server.rs`, bodies carrying `inputs` are handled here.
use super::*;

/// Text Generation Inference error object: a message plus a coarse error type.
fn error(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_server_error() {
        "generation"
    } else {
        "invalid_request"
    };
    (status, Json(json!({"error": message, "error_type": kind}))).into_response()
}

async fn convert_error(response: Response) -> Response {
    let (status, message) = error_message(response).await;
    error(status, &message)
}

/// `inputs` is one prompt or a list of prompts; fLLM runs a single sequence, so a batch is
/// concatenated the way the reference server treats a multi-line input.
fn input_value(inputs: &Value) -> Result<Value, String> {
    match inputs {
        Value::String(_) | Value::Number(_) => Ok(inputs.clone()),
        Value::Array(items) if items.is_empty() => Err("inputs must not be empty".into()),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                let part = item.as_str().ok_or("inputs list entries must be strings")?;
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part);
            }
            Ok(Value::String(text))
        }
        _ => Err("inputs must be a string or a list of strings".into()),
    }
}

/// Sampling knobs of a request, in TGI spelling.
struct Parameters {
    max_new_tokens: usize,
    temperature: f64,
    return_full_text: bool,
    details: bool,
    seed: Option<i64>,
}

fn parameter<'a>(body: &'a Value, name: &str) -> Option<&'a Value> {
    body.get("parameters")
        .filter(|parameters| !parameters.is_null())
        .and_then(|parameters| parameters.get(name))
        .filter(|value| !value.is_null())
}

fn parameters_of(state: &AppState, body: &Value) -> Parameters {
    Parameters {
        max_new_tokens: parameter(body, "max_new_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(128)
            .max(1) as usize,
        temperature: parameter(body, "temperature")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| state.default_temperature.unwrap_or(0.0)),
        return_full_text: parameter(body, "return_full_text")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        details: parameter(body, "details").and_then(Value::as_bool).unwrap_or(false),
        seed: parameter(body, "seed").and_then(Value::as_i64),
    }
}

/// Rewrites a TGI request into the OpenAI text-completions body.
fn to_completion_body(
    prompt: Value,
    params: &Parameters,
    body: &Value,
    stream: bool,
) -> Value {
    let do_sample = parameter(body, "do_sample").and_then(Value::as_bool).unwrap_or(true);
    let mut chat = json!({
        "prompt": prompt,
        "max_tokens": params.max_new_tokens,
        "stream": stream,
        "temperature": if do_sample { params.temperature } else { 0.0 },
    });
    for (from, to) in [
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("repetition_penalty", "repeat_penalty"),
        ("stop", "stop"),
        ("seed", "seed"),
    ] {
        if let Some(value) = parameter(body, from) {
            chat[to] = value.clone();
        }
    }
    if let Some(model) = body.get("model").filter(|model| !model.is_null()) {
        chat["model"] = model.clone();
    }
    chat
}

fn finish_reason_of(reply: &Value) -> &'static str {
    match reply["choices"][0]["finish_reason"].as_str() {
        Some("length") => "length",
        Some(_) => "eos_token",
        None => "error",
    }
}

/// The `details` object of a finished generation.
fn details_of(reply: &Value, seed: Option<i64>, generated_tokens: u64) -> Value {
    json!({
        "finish_reason": finish_reason_of(reply),
        "generated_tokens": generated_tokens,
        "seed": seed,
    })
}

fn token_event(id: u64, text: &str) -> Value {
    json!({"id": id, "text": text, "logprob": Value::Null, "special": false})
}

/// `data: {"token": …}` per chunk, then one final event carrying `generated_text` and
/// `details`. `generated_text` is only sent when `return_full_text` was requested.
fn stream_generation(stream: ChunkStream, prompt: String, params: Parameters) -> Response {
    let events = async_stream::stream! {
        let mut stream = stream;
        let mut generated = String::new();
        let mut id = 0u64;
        while let Some(chunk) = stream.next().await {
            let value = match chunk {
                Chunk::Data(value) => value,
                Chunk::Error(message) => {
                    yield Ok::<Event, Infallible>(
                        Event::default()
                            .data(json!({"error": message, "error_type": "generation"}).to_string()),
                    );
                    return;
                }
            };
            let choice = &value["choices"][0];
            let text = choice["text"].as_str().unwrap_or("");
            if !text.is_empty() {
                generated.push_str(text);
                let token = token_event(id, text);
                id += 1;
                yield Ok(Event::default().data(json!({"token": token}).to_string()));
            }
            if choice["finish_reason"].is_null() {
                continue;
            }
            let mut last = serde_json::Map::new();
            last.insert("token".into(), token_event(id, ""));
            last.insert(
                "generated_text".into(),
                // TGI puts the generated text in the final event; the prompt is added on request
                if params.return_full_text {
                    json!(format!("{prompt}{generated}"))
                } else {
                    json!(generated)
                },
            );
            last.insert(
                "details".into(),
                if params.details {
                    details_of(&value, params.seed, id)
                } else {
                    Value::Null
                },
            );
            yield Ok(Event::default().data(Value::Object(last).to_string()));
            return;
        }
    };
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// Runs a TGI body through `/v1/completions` and rewrites the reply into TGI's shape.
async fn run(state: Arc<AppState>, body: &Value, stream: bool) -> Response {
    let params = parameters_of(&state, body);
    let prompt = match input_value(body.get("inputs").unwrap_or(&Value::Null)) {
        Ok(prompt) => prompt,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let chat = to_completion_body(prompt.clone(), &params, body, stream);
    let response = completions(State(state), Json(chat)).await;
    if !response.status().is_success() {
        return convert_error(response).await;
    }
    if stream {
        return stream_generation(chunks(response), plain_text(&prompt), params);
    }
    match read_json(response).await {
        Ok(reply) => {
            let generated_tokens = reply["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            Json(json!({
                "generated_text": reply["choices"][0]["text"],
                "details": details_of(&reply, params.seed, generated_tokens),
            }))
            .into_response()
        }
        Err((status, message)) => error(status, &message),
    }
}

/// `POST /generate` with a TGI body; also reached through the SGLang dispatch.
pub(in crate::server) async fn generate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    run(state, &body, false).await
}

/// `POST /generate_stream`.
pub(in crate::server) async fn generate_stream(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    run(state, &body, true).await
}

/// `GET /info`.
pub(in crate::server) async fn info(
    State(state): State<Arc<AppState>>,
) -> Response {
    Json(json!({
        "model_id": state.model_name,
        "max_input_tokens": state.max_context,
        "max_total_tokens": state.max_context,
        "version": env!("CARGO_PKG_VERSION"),
        "sha": crate::version::GIT_SHORT,
        "build": crate::version::long_version(),
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(temperature: f64) -> Parameters {
        Parameters {
            max_new_tokens: 128,
            temperature,
            return_full_text: false,
            details: false,
            seed: None,
        }
    }

    #[test]
    fn inputs_accepts_single_and_batched_prompts() {
        assert_eq!(input_value(&json!("hi")).unwrap(), json!("hi"));
        assert_eq!(input_value(&json!(["a", "b"])).unwrap(), json!("a\nb"));
        assert!(input_value(&json!([])).is_err());
        assert!(input_value(&json!([1, 2])).is_err());
    }

    #[test]
    fn parameters_map_onto_the_completions_body() {
        let body = json!({
            "inputs": "hello",
            "model": "m",
            "parameters": {
                "max_new_tokens": 7,
                "temperature": 0.5,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "stop": ["\n\n"],
                "return_full_text": true
            }
        });
        let mut parsed = params(0.5);
        parsed.max_new_tokens = 7;
        parsed.return_full_text = true;
        let chat = to_completion_body(json!("hello"), &parsed, &body, false);
        assert_eq!(chat["max_tokens"], json!(7));
        assert_eq!(chat["temperature"], json!(0.5));
        assert_eq!(chat["top_k"], json!(20));
        assert_eq!(chat["repeat_penalty"], json!(1.1));
        assert_eq!(chat["stop"], json!(["\n\n"]));
        assert_eq!(chat["model"], json!("m"));
        assert_eq!(chat["stream"], json!(false));
    }

    #[test]
    fn do_sample_false_greedy_decodes() {
        let body = json!({"inputs": "hi", "parameters": {"do_sample": false}});
        let chat = to_completion_body(json!("hi"), &params(0.9), &body, true);
        assert_eq!(chat["temperature"], json!(0.0));
        assert_eq!(chat["stream"], json!(true));
    }

    #[test]
    fn finish_reasons_become_tgi_names() {
        let length = json!({"choices": [{"finish_reason": "length"}]});
        assert_eq!(finish_reason_of(&length), "length");
        let stop = json!({"choices": [{"finish_reason": "stop"}]});
        assert_eq!(finish_reason_of(&stop), "eos_token");
        let details = details_of(&stop, Some(42), 3);
        assert_eq!(details["finish_reason"], json!("eos_token"));
        assert_eq!(details["generated_tokens"], json!(3));
        assert_eq!(details["seed"], json!(42));
    }

    #[test]
    fn token_events_carry_id_text_and_special_flag() {
        assert_eq!(
            token_event(2, "hi"),
            json!({"id": 2, "text": "hi", "logprob": Value::Null, "special": false})
        );
    }
}