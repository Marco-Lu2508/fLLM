//! Tests of the HTTP layer: the real router with all middleware and the real worker loop, on the tiny random model
//! (CPU, built once per test run). Requests go through `tower::ServiceExt::oneshot`; the tests of connection handling
//! (timeouts, connection limit, TLS, graceful shutdown) use a real listener on port 0.
use super::*;
use crate::{tiny_model, GgufModel, InferenceRuntime};
use axum::{
    body::Body,
    http::{header, HeaderMap, Method, Request, StatusCode},
};
use fllm_gguf_probe::MappedGguf;
use options::MetricsStyle;
use std::{path::PathBuf, sync::OnceLock, time::Duration};
use tower::ServiceExt;

fn tiny_gguf() -> &'static std::path::Path {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("fllm-server-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // the file store of /v1/files and /v1/batches reads this once, on first use
        std::env::set_var("FLLM_FILES_DIR", dir.join("files"));
        let path = dir.join("tiny.gguf");
        tiny_model::write_gguf(&path, &tiny_model::Options::default()).unwrap();
        path
    })
}

#[derive(Clone)]
struct Settings {
    http: HttpOptions,
    max_seqs: usize,
    ctx: usize,
    temperature: Option<f64>,
    tool_parser: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            http: HttpOptions::default(),
            max_seqs: 4,
            ctx: 256,
            temperature: None,
            tool_parser: Some("hermes".into()),
        }
    }
}

struct Harness {
    router: Router,
    state: Arc<AppState>,
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.text()))
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn header(&self, name: &str) -> &str {
        self.headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or("")
    }
    /// `(event name, data)` pairs of a Server-Sent-Events body.
    fn events(&self) -> Vec<(String, String)> {
        self.text()
            .split("\n\n")
            .filter(|block| !block.trim().is_empty())
            .map(|block| {
                let mut name = String::new();
                let mut data = Vec::new();
                for line in block.lines() {
                    if let Some(v) = line.strip_prefix("event:") {
                        name = v.trim().to_owned();
                    } else if let Some(v) = line.strip_prefix("data:") {
                        data.push(v.trim_start().to_owned());
                    }
                }
                (name, data.join("\n"))
            })
            .collect()
    }
    /// The JSON payloads of the data lines of an SSE body, without `[DONE]`.
    fn chunks(&self) -> Vec<Value> {
        self.events()
            .into_iter()
            .filter(|(_, data)| data != "[DONE]")
            .map(|(_, data)| serde_json::from_str(&data).unwrap_or_else(|e| panic!("bad SSE data {data:?}: {e}")))
            .collect()
    }
}

fn start(settings: Settings) -> Harness {
    let gguf = tiny_gguf();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mapping = MappedGguf::open(gguf).unwrap();
        let facts = ModelFacts::from_gguf(gguf, &mapping);
        let tokenizer = TokenizerBundle::from_gguf(&mapping).unwrap();
        let eos: Vec<i64> = tokenizer.eos_ids().iter().map(|id| i64::from(*id)).collect();
        let model = GgufModel::load(gguf.to_str().unwrap(), settings.ctx, settings.max_seqs, 2).unwrap();
        let mut runtime = InferenceRuntime::with_speculative(model, settings.max_seqs, 128, eos, 0, 32).unwrap();
        runtime.attach_compute("cpu").unwrap();
        let config = ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            model_name: "tiny".into(),
            http: settings.http,
            facts,
            default_temperature: settings.temperature,
            default_top_p: None,
            default_top_k: None,
            sampling_defaults: Default::default(),
            tool_parser: settings.tool_parser,
            reasoning_parser: None,
        };
        let Built { state, router, worker_rx, metrics } = build(&runtime, tokenizer, config).unwrap();
        state.ready.store(true, Ordering::Relaxed);
        sender.send((router, state)).unwrap();
        // the worker ends when the last sender of its channel (the router state) is dropped
        let _ = run_worker_loop(runtime, worker_rx, metrics);
    });
    let (router, state) = receiver.recv_timeout(Duration::from_secs(60)).expect("harness start");
    Harness { router, state }
}

fn shared() -> &'static Harness {
    static SHARED: OnceLock<Harness> = OnceLock::new();
    SHARED.get_or_init(|| start(Settings::default()))
}

fn with_keys(keys: &[&str]) -> Settings {
    let mut settings = Settings::default();
    settings.http.api_keys = keys.iter().map(|k| k.to_string()).collect();
    settings
}

impl Harness {
    async fn send(&self, request: Request<Body>) -> Reply {
        let response = self.router.clone().oneshot(request).await.unwrap();
        let (parts, body) = response.into_parts();
        let body = axum::body::to_bytes(body, 64 << 20).await.unwrap();
        Reply { status: parts.status, headers: parts.headers, body: body.to_vec() }
    }

    async fn call(&self, method: Method, path: &str, body: Option<Value>, headers: &[(&str, &str)]) -> Reply {
        let mut builder = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let request = match body {
            Some(body) => builder
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string())),
            None => builder.body(Body::empty()),
        };
        self.send(request.unwrap()).await
    }

    async fn get(&self, path: &str) -> Reply {
        self.call(Method::GET, path, None, &[]).await
    }

    async fn post(&self, path: &str, body: Value) -> Reply {
        self.call(Method::POST, path, Some(body), &[]).await
    }

    async fn post_with(&self, path: &str, body: Value, headers: &[(&str, &str)]) -> Reply {
        self.call(Method::POST, path, Some(body), headers).await
    }

    /// A chat request that is deterministic and short.
    async fn chat(&self, extra: Value) -> Reply {
        let mut body = json!({"model": "tiny", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 6, "temperature": 0});
        merge(&mut body, extra);
        self.post("/v1/chat/completions", body).await
    }
}

fn merge(target: &mut Value, extra: Value) {
    if let (Some(target), Some(extra)) = (target.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            target.insert(k.clone(), v.clone());
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// OpenAI

#[tokio::test]
async fn models_list_the_served_name_and_the_aliases() {
    let mut settings = Settings::default();
    settings.http.model_aliases = vec!["gpt-4o".into(), "claude-*".into()];
    let h = start(settings);
    let reply = h.get("/v1/models").await;
    assert_eq!(reply.status, StatusCode::OK);
    let listed: Vec<String> = reply.json()["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(listed, ["tiny", "gpt-4o"], "wildcard aliases are not listed");
    assert_eq!(reply.json()["data"][0]["max_model_len"], 256);
    assert_eq!(h.get("/v1/models/gpt-4o").await.status, StatusCode::OK);
    assert_eq!(h.get("/v1/models/claude-sonnet-4").await.status, StatusCode::OK);
    let missing = h.get("/v1/models/other").await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.json()["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn model_field_is_validated() {
    let mut settings = Settings::default();
    settings.http.model_aliases = vec!["gpt-4o".into()];
    let h = start(settings);
    for model in [json!("tiny"), json!("gpt-4o"), json!(""), json!(null)] {
        let reply = h.chat(json!({"model": model})).await;
        assert_eq!(reply.status, StatusCode::OK, "model {model}: {}", reply.text());
    }
    // omitted entirely
    let reply = h
        .post("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}))
        .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json()["model"], "tiny");
    // the alias is echoed back as the client wrote it
    assert_eq!(h.chat(json!({"model": "gpt-4o"})).await.json()["model"], "gpt-4o");
    let reply = h.chat(json!({"model": "gpt-9"})).await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "model_not_found");
    assert_eq!(error["type"], "invalid_request_error");
    assert_eq!(error["param"], "model");
    assert!(error["message"].as_str().unwrap().contains("gpt-9"));
    assert!(error["request_id"].as_str().unwrap().starts_with("req-"));
    for path in ["/v1/completions", "/v1/embeddings"] {
        let reply = h.post(path, json!({"model": "gpt-9", "prompt": "a", "input": "a"})).await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn chat_completion_has_the_openai_shape() {
    let h = shared();
    let reply = h.chat(json!({"user": "u-1", "metadata": {"k": "v"}, "store": false, "service_tier": "auto"})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let body = reply.json();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["service_tier"], "default");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_string());
    assert!(["stop", "length"].contains(&body["choices"][0]["finish_reason"].as_str().unwrap()));
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage["total_tokens"].as_u64().unwrap(), usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap());
    // greedy decoding is repeatable
    assert_eq!(h.chat(json!({})).await.json()["choices"], h.chat(json!({})).await.json()["choices"]);
}

#[tokio::test]
async fn chat_stream_follows_the_sse_framing() {
    let h = shared();
    let reply = h.chat(json!({"stream": true, "stream_options": {"include_usage": true}, "max_tokens": 5, "ignore_eos": true})).await;
    assert_eq!(reply.status, StatusCode::OK);
    assert!(reply.header("content-type").starts_with("text/event-stream"), "{}", reply.header("content-type"));
    let text = reply.text();
    assert!(text.ends_with("data: [DONE]\n\n"), "stream must end with [DONE]: {text:?}");
    for block in text.split("\n\n").filter(|b| !b.is_empty()) {
        assert!(block.starts_with("data: ") || block.starts_with(':'), "unexpected SSE block {block:?}");
    }
    let chunks = reply.chunks();
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk"));
    let finish: Vec<_> = chunks.iter().filter(|c| c["choices"][0]["finish_reason"].is_string()).collect();
    assert_eq!(finish.len(), 1);
    assert_eq!(finish[0]["choices"][0]["finish_reason"], "length");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 5);
    // without stream_options there is no usage chunk
    let plain = h.chat(json!({"stream": true, "ignore_eos": true})).await;
    assert!(plain.chunks().iter().all(|c| c["usage"].is_null() || c["choices"] != json!([])));
}

#[tokio::test]
async fn default_max_tokens_is_the_rest_of_the_context() {
    let h = shared();
    let reply = h
        .post(
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "hello"}], "temperature": 0, "ignore_eos": true}),
        )
        .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let usage = &reply.json()["usage"];
    assert_eq!(usage["total_tokens"], 256, "prompt plus completion fill the context: {usage}");
    assert_eq!(reply.json()["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn too_long_requests_name_the_context_limit() {
    let h = shared();
    let reply = h.chat(json!({"max_tokens": 10_000})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "context_length_exceeded");
    assert!(error["message"].as_str().unwrap().contains("256"));
    let long = "x".repeat(2000);
    let reply = h.chat(json!({"messages": [{"role": "user", "content": long}], "max_tokens": null})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn completions_accept_text_and_token_prompts() {
    let h = shared();
    let reply = h.post("/v1/completions", json!({"model": "tiny", "prompt": "abc", "max_tokens": 4, "temperature": 0, "echo": true})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json()["object"], "text_completion");
    assert!(reply.json()["choices"][0]["text"].as_str().unwrap().starts_with("abc"));
    let reply = h.post("/v1/completions", json!({"prompt": [10, 11, 12], "max_tokens": 3, "temperature": 0})).await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json()["usage"]["prompt_tokens"], 3);
    let reply = h.post("/v1/completions", json!({"prompt": "abc", "max_tokens": 3, "n": 2, "temperature": 1.0})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json()["choices"].as_array().unwrap().len(), 2);
    // best_of samples more candidates than it returns and keeps the n with the highest log-probability
    let reply = h.post("/v1/completions", json!({"prompt": "abc", "max_tokens": 3, "best_of": 3, "temperature": 1.0})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json()["choices"].as_array().unwrap().len(), 1);
    let reply = h.post("/v1/completions", json!({"prompt": "abc", "best_of": 99})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert!(reply.json()["error"]["message"].as_str().unwrap().contains("best_of"));
    let reply = h.post("/v1/completions", json!({"prompt": "abc", "best_of": 3, "stream": true})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert!(reply.json()["error"]["message"].as_str().unwrap().contains("stream"));
    let reply = h.post("/v1/completions", json!({"prompt": "abc", "max_tokens": 3, "logit_bias": {"5": 100}, "temperature": 0})).await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json()["choices"][0]["text"], "!!!", "token 5 is the exclamation mark; the bias forces it");
}

#[tokio::test]
async fn embeddings_return_vectors() {
    let h = shared();
    let reply = h.post("/v1/embeddings", json!({"model": "tiny", "input": ["a", "bc"]})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let body = reply.json();
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 64);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    let reply = h.post("/v1/embeddings", json!({"input": "abc", "encoding_format": "base64"})).await;
    assert!(reply.json()["data"][0]["embedding"].is_string());
    let reply = h.post("/v1/embeddings", json!({"input": "x".repeat(400)})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn structured_output_follows_the_schema() {
    let h = shared();
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
                        "required": ["a", "b"], "additionalProperties": false});
    // A random model may spend its budget on whitespace, which the grammar allows between tokens. With fixed seeds the outcome is
    // reproducible: at least one of the first eight must finish the object, and whatever finishes must follow the schema.
    let mut finished = 0;
    for seed in 0..8u64 {
        let reply = h
            .chat(json!({"max_tokens": 60, "temperature": 1.0, "seed": seed, "response_format": {"type": "json_schema",
                "json_schema": {"name": "t", "schema": schema, "strict": true}}}))
            .await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
        let choice = &reply.json()["choices"][0];
        let content = choice["message"]["content"].as_str().unwrap().to_owned();
        if choice["finish_reason"] == "stop" {
            let value: Value = serde_json::from_str(&content).unwrap_or_else(|e| panic!("not JSON ({e}): {content:?}"));
            assert!(value["a"].is_number() && value["b"].is_boolean(), "{content}");
            finished += 1;
        } else {
            assert!(content.trim_start().starts_with('{'), "{content}");
        }
    }
    assert!(finished > 0, "none of eight seeded requests completed a schema-conforming object");
    // a random model may not close the object within the budget; what it writes must at least open one
    let object = h.chat(json!({"max_tokens": 80, "response_format": {"type": "json_object"}})).await;
    assert!(object.json()["choices"][0]["message"]["content"].as_str().unwrap().trim_start().starts_with('{'), "{}", object.text());
}

const WEATHER_CALL: &str = "<tool_call>\\n{\\\"name\\\": \\\"get_weather\\\", \\\"arguments\\\": {\\\"city\\\": \\\"Berlin\\\"}}\\n</tool_call>";

fn weather_tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

#[tokio::test]
async fn tool_calls_round_trip() {
    let h = shared();
    // The grammar makes the random model emit exactly a Hermes-format call; the parser must turn it into tool_calls.
    let grammar = format!("start: \"{WEATHER_CALL}\"");
    let reply = h
        .chat(json!({"max_tokens": 120, "tools": weather_tools(), "tool_choice": "auto", "grammar": grammar, "grammar_type": "lark"}))
        .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let choice = &reply.json()["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{choice}");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    let arguments: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments, json!({"city": "Berlin"}));
    let id = call["id"].as_str().unwrap().to_owned();
    // streaming delivers the same call as deltas
    let stream = h
        .chat(json!({"max_tokens": 120, "stream": true, "tools": weather_tools(), "grammar": format!("start: \"{WEATHER_CALL}\""), "grammar_type": "lark"}))
        .await;
    let chunks = stream.chunks();
    let streamed: Vec<&Value> = chunks.iter().filter(|c| c["choices"][0]["delta"]["tool_calls"].is_array()).collect();
    assert!(!streamed.is_empty(), "{}", stream.text());
    assert_eq!(streamed[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
    assert!(chunks.iter().any(|c| c["choices"][0]["finish_reason"] == "tool_calls"));
    // the history with the call and its result renders and generates
    let followup = h
        .chat(json!({"messages": [
            {"role": "user", "content": "weather in Berlin?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": id, "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}}]},
            {"role": "tool", "tool_call_id": id, "content": "sunny"}], "tools": weather_tools()}))
        .await;
    assert_eq!(followup.status, StatusCode::OK, "{}", followup.text());
}

#[tokio::test]
async fn responses_api_stores_and_returns_responses() {
    let h = shared();
    let reply = h.post("/v1/responses", json!({"model": "tiny", "input": "hello", "max_output_tokens": 4, "temperature": 0})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let body = reply.json();
    assert_eq!(body["object"], "response");
    assert!(["completed", "incomplete"].contains(&body["status"].as_str().unwrap()), "{body}");
    let id = body["id"].as_str().unwrap().to_owned();
    let again = h.get(&format!("/v1/responses/{id}")).await;
    assert_eq!(again.status, StatusCode::OK);
    assert_eq!(again.json()["id"], id.as_str());
    let stream = h.post("/v1/responses", json!({"input": "hello", "max_output_tokens": 4, "stream": true})).await;
    let names: Vec<String> = stream.events().into_iter().map(|(name, _)| name).collect();
    assert_eq!(names[0], "response.created");
    assert!(["response.completed", "response.incomplete"].contains(&names.last().unwrap().as_str()), "{names:?}");
    let deleted = h.call(Method::DELETE, &format!("/v1/responses/{id}"), None, &[]).await;
    assert_eq!(deleted.status, StatusCode::OK);
    assert_eq!(h.get(&format!("/v1/responses/{id}")).await.status, StatusCode::NOT_FOUND);
}

fn multipart(purpose: &str, filename: &str, data: &str) -> (String, Vec<u8>) {
    let boundary = "----fllmtestboundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/jsonl\r\n\r\n{data}\r\n--{boundary}--\r\n"
    );
    (format!("multipart/form-data; boundary={boundary}"), body.into_bytes())
}

#[tokio::test]
async fn files_and_batches_run_requests_offline() {
    let h = shared();
    let line = |id: &str| {
        json!({"custom_id": id, "method": "POST", "url": "/v1/chat/completions",
               "body": {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0}})
        .to_string()
    };
    let (content_type, body) = multipart("batch", "in.jsonl", &format!("{}\n{}\n", line("a"), line("b")));
    let upload = h
        .send(Request::builder().method(Method::POST).uri("/v1/files").header(header::CONTENT_TYPE, content_type).body(Body::from(body)).unwrap())
        .await;
    assert_eq!(upload.status, StatusCode::OK, "{}", upload.text());
    let file_id = upload.json()["id"].as_str().unwrap().to_owned();
    assert_eq!(h.get(&format!("/v1/files/{file_id}")).await.json()["purpose"], "batch");
    assert!(h.get("/v1/files").await.json()["data"].as_array().unwrap().iter().any(|f| f["id"] == file_id.as_str()));
    let batch = h.post("/v1/batches", json!({"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"})).await;
    assert_eq!(batch.status, StatusCode::OK, "{}", batch.text());
    let batch_id = batch.json()["id"].as_str().unwrap().to_owned();
    let mut finished = Value::Null;
    for _ in 0..200 {
        let status = h.get(&format!("/v1/batches/{batch_id}")).await.json();
        if status["status"] == "completed" || status["status"] == "failed" {
            finished = status;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(finished["status"], "completed", "{finished}");
    assert_eq!(finished["request_counts"]["completed"], 2);
    let output = h.get(&format!("/v1/files/{}/content", finished["output_file_id"].as_str().unwrap())).await;
    let lines: Vec<Value> = output.text().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2);
    assert!(lines.iter().all(|l| l["response"]["status_code"] == 200));
    let gone = h.call(Method::DELETE, &format!("/v1/files/{file_id}"), None, &[]).await;
    assert_eq!(gone.status, StatusCode::OK);
}

// ---------------------------------------------------------------------------------------------------------------
// Anthropic

#[tokio::test]
async fn anthropic_messages_non_streaming_and_streaming() {
    let h = shared();
    let body = json!({"model": "tiny", "max_tokens": 5, "temperature": 0,
        "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}]});
    let reply = h.post("/v1/messages", body.clone()).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let message = reply.json();
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert!(message["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(["end_turn", "max_tokens"].contains(&message["stop_reason"].as_str().unwrap()));

    let mut streaming = body.clone();
    streaming["stream"] = json!(true);
    let reply = h.post("/v1/messages", streaming).await;
    assert_eq!(reply.status, StatusCode::OK);
    let events = reply.events();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names[0], "message_start");
    assert_eq!(*names.last().unwrap(), "message_stop");
    assert!(names.contains(&"message_delta"));
    let start: Value = serde_json::from_str(&events[0].1).unwrap();
    assert!(start["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0, "message_start reports the prompt size: {start}");
    // the count matches /v1/messages/count_tokens
    let counted = h.post("/v1/messages/count_tokens", body).await.json()["input_tokens"].clone();
    assert_eq!(start["message"]["usage"]["input_tokens"], counted);
}

#[tokio::test]
async fn anthropic_errors_use_the_anthropic_shape() {
    let h = shared();
    let reply = h.post("/v1/messages", json!({"model": "claude-x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert_eq!(reply.json()["type"], "error");
    assert_eq!(reply.json()["error"]["type"], "not_found_error");
    let reply = h.post("/v1/messages", json!({"max_tokens": 5})).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    // models in the Anthropic shape
    let models = h.call(Method::GET, "/v1/models", None, &[("anthropic-version", "2023-06-01")]).await;
    assert_eq!(models.json()["data"][0]["type"], "model");
}

// ---------------------------------------------------------------------------------------------------------------
// Ollama

#[tokio::test]
async fn ollama_tags_report_the_loaded_file() {
    let h = shared();
    let reply = h.get("/api/tags").await;
    assert_eq!(reply.status, StatusCode::OK);
    let model = &reply.json()["models"][0];
    assert_eq!(model["name"], "tiny");
    assert!(model["size"].as_u64().unwrap() > 100_000, "{model}");
    let digest = model["digest"].as_str().unwrap();
    assert!(digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()), "{digest}");
    assert_eq!(model["details"]["family"], "llama");
    assert_eq!(model["details"]["format"], "gguf");
    assert_eq!(model["details"]["quantization_level"], "F32");
    assert!(!model["details"]["parameter_size"].as_str().unwrap().is_empty());
    let show = h.post("/api/show", json!({"model": "tiny"})).await.json();
    assert!(show["model_info"]["general.parameter_count"].as_u64().unwrap() > 0);
    assert_eq!(h.post("/api/show", json!({"model": "tiny:latest"})).await.status, StatusCode::OK);
}

#[tokio::test]
async fn ollama_model_management_and_blobs() {
    let h = shared();
    let tags = h.get("/api/tags").await.json();
    let digest = tags["models"][0]["digest"].as_str().unwrap().to_owned();

    // Blobs
    let head_valid = h.call(Method::HEAD, &format!("/api/blobs/{digest}"), None, &[]).await;
    assert_eq!(head_valid.status, StatusCode::OK);
    let head_invalid = h.call(Method::HEAD, "/api/blobs/deadbeef00000000", None, &[]).await;
    assert_eq!(head_invalid.status, StatusCode::NOT_FOUND);
    let post_blob = h.call(Method::POST, "/api/blobs/anyblob", None, &[]).await;
    assert_eq!(post_blob.status, StatusCode::CREATED);

    // Pull & Push
    let pull_res = h.post("/api/pull", json!({"model": "tiny", "stream": false})).await;
    assert_eq!(pull_res.status, StatusCode::OK);
    assert_eq!(pull_res.json()["status"], "success");
    let pull_err = h.post("/api/pull", json!({"model": "unknown-model", "stream": false})).await;
    assert_eq!(pull_err.status, StatusCode::NOT_FOUND);

    let push_res = h.post("/api/push", json!({"model": "tiny", "stream": false})).await;
    assert_eq!(push_res.status, StatusCode::OK);
    assert_eq!(push_res.json()["status"], "success");
    let push_err = h.post("/api/push", json!({"model": "unknown-model", "stream": false})).await;
    assert_eq!(push_err.status, StatusCode::NOT_FOUND);

    // Create & Copy & Delete
    let create_res = h.post("/api/create", json!({"model": "test-create", "stream": false})).await;
    assert_eq!(create_res.status, StatusCode::OK);

    let copy_res = h.post("/api/copy", json!({"source": "tiny", "destination": "tiny-copied"})).await;
    assert_eq!(copy_res.status, StatusCode::OK);
    let tags_after = h.get("/api/tags").await.json();
    let names: Vec<&str> = tags_after["models"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"tiny-copied"));

    let del_res = h.call(Method::DELETE, "/api/delete", Some(json!({"model": "tiny-copied"})), &[]).await;
    assert_eq!(del_res.status, StatusCode::OK);
    let tags_final = h.get("/api/tags").await.json();
    let names_final: Vec<&str> = tags_final["models"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert!(!names_final.contains(&"tiny-copied"));
}

#[tokio::test]
async fn ollama_chat_generate_and_errors() {
    let h = shared();
    let reply = h.post("/api/chat", json!({"model": "tiny", "stream": false, "messages": [{"role": "user", "content": "hi"}],
        "options": {"temperature": 0, "num_predict": 4}})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let body = reply.json();
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["eval_count"].as_u64().unwrap().min(4), body["eval_count"].as_u64().unwrap());
    // default is a stream of JSON lines that ends with done:true
    let stream = h.post("/api/chat", json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "options": {"temperature": 0, "num_predict": 4}})).await;
    let lines: Vec<Value> = stream.text().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.last().unwrap()["done"], true);
    assert!(lines[..lines.len() - 1].iter().all(|l| l["done"] == false));
    let generate = h.post("/api/generate", json!({"model": "tiny", "prompt": "abc", "stream": false, "options": {"num_predict": 3, "temperature": 0}})).await;
    assert_eq!(generate.status, StatusCode::OK);
    assert!(generate.json()["response"].is_string());
    // unknown model: 404 in Ollama's error shape
    let missing = h.post("/api/chat", json!({"model": "nope", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.json()["error"], "model 'nope' not found");
    // default num_predict is the rest of the context
    let full = h.post("/api/generate", json!({"model": "tiny", "prompt": "abc", "stream": false, "options": {"temperature": 0, "stop": []}})).await;
    assert_eq!(full.status, StatusCode::OK);
    assert!(full.json()["eval_count"].as_u64().unwrap() > 128 || full.json()["done_reason"] == "stop");
}

#[tokio::test]
async fn ollama_keep_alive_and_loading() {
    let h = shared();
    for keep_alive in [json!("5m"), json!(-1), json!("1h30m"), json!(0), json!("0")] {
        let reply = h.post("/api/generate", json!({"model": "tiny", "prompt": "", "keep_alive": keep_alive})).await;
        assert_eq!(reply.status, StatusCode::OK, "{keep_alive}: {}", reply.text());
        assert_eq!(reply.json()["done"], true);
    }
    let unload = h.post("/api/generate", json!({"model": "tiny", "prompt": "", "keep_alive": 0})).await;
    assert_eq!(unload.json()["done_reason"], "unload");
    let load = h.post("/api/chat", json!({"model": "tiny", "messages": []})).await;
    assert_eq!(load.json()["done_reason"], "load");
    let bad = h.post("/api/generate", json!({"model": "tiny", "prompt": "x", "keep_alive": "soon"})).await;
    assert_eq!(bad.status, StatusCode::BAD_REQUEST);
    h.post("/api/generate", json!({"model": "tiny", "prompt": "", "keep_alive": "10m"})).await;
    let ps = h.get("/api/ps").await.json();
    assert_eq!(ps["models"][0]["name"], "tiny");
    assert!(ps["models"][0]["expires_at"].as_str().unwrap().starts_with("20"));
    h.post("/api/generate", json!({"model": "tiny", "prompt": "", "keep_alive": -1})).await;
    assert_eq!(h.get("/api/ps").await.json()["models"][0]["expires_at"], "0001-01-01T00:00:00Z");
}

#[test]
fn keep_alive_values_parse_like_ollama() {
    use api::ollama_extra::{parse_keep_alive, KeepAlive};
    assert_eq!(parse_keep_alive(&json!("5m")).unwrap(), KeepAlive::For(Duration::from_secs(300)));
    assert_eq!(parse_keep_alive(&json!("1h30m")).unwrap(), KeepAlive::For(Duration::from_secs(5400)));
    assert_eq!(parse_keep_alive(&json!("250ms")).unwrap(), KeepAlive::For(Duration::from_millis(250)));
    assert_eq!(parse_keep_alive(&json!(90)).unwrap(), KeepAlive::For(Duration::from_secs(90)));
    assert_eq!(parse_keep_alive(&json!(0)).unwrap(), KeepAlive::Unload);
    assert_eq!(parse_keep_alive(&json!(-1)).unwrap(), KeepAlive::Forever);
    assert_eq!(parse_keep_alive(&json!("-1")).unwrap(), KeepAlive::Forever);
    assert_eq!(parse_keep_alive(&json!("-5m")).unwrap(), KeepAlive::Forever);
    for bad in [json!("soon"), json!(""), json!("5x"), json!(true), json!(null)] {
        assert!(parse_keep_alive(&bad).is_err(), "{bad}");
    }
}

#[tokio::test]
async fn ollama_embed_truncates_unless_told_not_to() {
    let h = shared();
    let long = "y".repeat(400);
    let reply = h.post("/api/embed", json!({"model": "tiny", "input": [long.clone(), "short"]})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json()["embeddings"].as_array().unwrap().len(), 2);
    assert!(reply.json()["prompt_eval_count"].as_u64().unwrap() > 0);
    let strict = h.post("/api/embed", json!({"model": "tiny", "input": long, "truncate": false})).await;
    assert_eq!(strict.status, StatusCode::BAD_REQUEST);
    assert!(strict.json()["error"].is_string());
    let bad = h.post("/api/embed", json!({"model": "tiny", "input": "a", "truncate": "yes"})).await;
    assert_eq!(bad.status, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------------------------------------------
// SGLang, TGI, llama.cpp, vLLM

#[tokio::test]
async fn sglang_tgi_and_llama_cpp_routes_answer() {
    let h = shared();
    let reply = h.post("/generate", json!({"text": "abc", "sampling_params": {"max_new_tokens": 3, "temperature": 0}})).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json()["meta_info"]["completion_tokens"].as_u64().unwrap().min(3), reply.json()["meta_info"]["completion_tokens"].as_u64().unwrap());
    assert_eq!(h.get("/get_model_info").await.json()["model_path"], "tiny");
    assert!(h.get("/get_server_info").await.json()["version"].is_string());
    // TGI
    let tgi = h.post("/generate", json!({"inputs": "abc", "parameters": {"max_new_tokens": 3}})).await;
    assert_eq!(tgi.status, StatusCode::OK, "{}", tgi.text());
    assert!(tgi.json()["generated_text"].is_string());
    assert!(h.get("/info").await.json()["model_id"].is_string());
    let tgi_stream = h.post("/generate_stream", json!({"inputs": "abc", "parameters": {"max_new_tokens": 3}})).await;
    assert_eq!(tgi_stream.status, StatusCode::OK);
    assert!(tgi_stream.events().iter().any(|(_, d)| d.contains("generated_text")), "{}", tgi_stream.text());
    // llama.cpp
    let completion = h.post("/completion", json!({"prompt": "abc", "n_predict": 3, "temperature": 0})).await;
    assert_eq!(completion.status, StatusCode::OK, "{}", completion.text());
    assert_eq!(completion.json()["stop"], true);
    let streamed = h.post("/completion", json!({"prompt": "abc", "n_predict": 3, "stream": true})).await;
    assert!(streamed.chunks().last().unwrap()["stop"] == true);
    let props = h.get("/props").await;
    assert_eq!(props.status, StatusCode::OK);
    assert!(props.json()["default_generation_settings"]["temperature"].is_number());
    assert_eq!(h.post("/props", json!({"anything": 1})).await.json()["success"], true);
    assert!(h.get("/slots").await.json().as_array().is_some());
    for action in ["save", "restore", "erase"] {
        let reply = h.post(&format!("/slots/0?action={action}"), json!({"filename": "x"})).await;
        assert_eq!(reply.status, StatusCode::NOT_IMPLEMENTED, "{action}");
        assert!(reply.json()["error"]["message"].as_str().unwrap().contains(action));
    }
    assert_eq!(h.post("/slots/0?action=bogus", json!({})).await.status, StatusCode::BAD_REQUEST);
    assert_eq!(h.get("/lora-adapters").await.json(), json!([]));
    assert_eq!(h.post("/lora-adapters", json!([{"id": 0, "scale": 0.5}])).await.status, StatusCode::NOT_IMPLEMENTED);
    let template = h.post("/apply-template", json!({"messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(template.json()["prompt"], "user: hi\nassistant:");
    let infill = h.post("/infill", json!({"input_prefix": "a", "input_suffix": "b"})).await;
    assert_eq!(infill.status, StatusCode::BAD_REQUEST, "{}", infill.text());
    assert!(infill.json()["error"]["message"].as_str().unwrap().contains("fill-in-the-middle"), "the tiny model has no FIM tokens");
    assert_eq!(h.post("/embedding", json!({"content": "abc"})).await.status, StatusCode::OK);
}

#[tokio::test]
async fn vllm_routes_and_tokenizer_endpoints() {
    let h = shared();
    assert_eq!(h.get("/version").await.json()["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(h.get("/is_sleeping").await.json(), json!({"is_sleeping": false}));
    let tokens = h.post("/tokenize", json!({"prompt": "abc"})).await.json();
    assert_eq!(tokens["tokens"].as_array().unwrap().len(), 3);
    assert_eq!(tokens["count"], 3);
    assert_eq!(tokens["max_model_len"], 256);
    let from_messages = h.post("/tokenize", json!({"messages": [{"role": "user", "content": "hi"}]})).await.json();
    assert_eq!(from_messages["count"], "user: hi\nassistant:".chars().count());
    // llama.cpp's field name still works
    assert_eq!(h.post("/tokenize", json!({"content": "abc"})).await.json()["tokens"], tokens["tokens"]);
    let text = h.post("/detokenize", json!({"tokens": tokens["tokens"]})).await.json();
    assert_eq!(text["prompt"], "abc");
    assert_eq!(text["content"], "abc");
    let render = h.post("/v1/chat/completions/render", json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(render.status, StatusCode::OK, "{}", render.text());
    let rendered = render.json();
    assert_eq!(rendered[0][0]["content"], "hi");
    assert_eq!(rendered[1][0]["prompt"], "user: hi\nassistant:");
    assert_eq!(rendered[1][0]["prompt_token_ids"].as_array().unwrap().len(), 19);
    assert_eq!(h.post("/v1/chat/completions/render", json!({"messages": []})).await.status, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------------------------------------------
// Errors, request ids, limits

#[tokio::test]
async fn framework_errors_get_the_route_error_shape() {
    let h = shared();
    // malformed JSON
    let reply = h
        .send(Request::builder().method(Method::POST).uri("/v1/chat/completions").header(header::CONTENT_TYPE, "application/json").body(Body::from("{not json")).unwrap())
        .await;
    assert!(reply.status.is_client_error());
    assert!(reply.json()["error"]["message"].is_string(), "{}", reply.text());
    assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    // wrong content type
    let reply = h
        .send(Request::builder().method(Method::POST).uri("/v1/chat/completions").header(header::CONTENT_TYPE, "text/plain").body(Body::from("{}")).unwrap())
        .await;
    assert_eq!(reply.status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert!(reply.json()["error"]["message"].is_string());
    // unknown route and wrong method
    let reply = h.get("/v1/nothing").await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert_eq!(reply.json()["error"]["code"], "not_found");
    let reply = h.get("/v1/chat/completions").await;
    assert_eq!(reply.status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(reply.json()["error"]["code"], "method_not_allowed");
    // the Ollama and Anthropic routes keep their own shapes
    let reply = h.send(Request::builder().method(Method::POST).uri("/api/chat").header(header::CONTENT_TYPE, "application/json").body(Body::from("[")).unwrap()).await;
    assert!(reply.json()["error"].is_string(), "{}", reply.text());
    let reply = h.send(Request::builder().method(Method::POST).uri("/v1/messages").header(header::CONTENT_TYPE, "application/json").body(Body::from("[")).unwrap()).await;
    assert_eq!(reply.json()["type"], "error");
}

#[tokio::test]
async fn request_ids_are_accepted_generated_and_put_into_errors() {
    let h = shared();
    let reply = h.call(Method::GET, "/health", None, &[("x-request-id", "trace-123.a:b")]).await;
    assert_eq!(reply.header("x-request-id"), "trace-123.a:b");
    let reply = h.get("/health").await;
    assert!(reply.header("x-request-id").starts_with("req-"));
    // unusable ids are replaced
    let reply = h.call(Method::GET, "/health", None, &[("x-request-id", "bad id with spaces")]).await;
    assert!(reply.header("x-request-id").starts_with("req-"));
    let reply = h.post_with("/v1/chat/completions", json!({"messages": []}), &[("x-request-id", "abc-1")]).await;
    assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    assert_eq!(reply.json()["error"]["request_id"], "abc-1");
    assert_eq!(reply.header("x-request-id"), "abc-1");
}

#[tokio::test]
async fn bodies_over_the_limit_are_refused() {
    let mut settings = Settings::default();
    settings.http.max_body_bytes = 1024;
    let h = start(settings);
    let big = json!({"messages": [{"role": "user", "content": "a".repeat(4000)}]});
    let reply = h.post("/v1/chat/completions", big).await;
    assert_eq!(reply.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(reply.json()["error"]["code"], "request_too_large");
    assert!(reply.json()["error"]["message"].as_str().unwrap().contains("--max-body-size"));
    assert_eq!(h.chat(json!({})).await.status, StatusCode::OK);
}

#[tokio::test]
async fn queue_limit_answers_429_with_retry_after() {
    let mut settings = Settings { max_seqs: 1, ..Settings::default() };
    settings.http.max_queue = Some(1);
    let h = Arc::new(start(settings));
    h.state.metrics.inject_delay_ms.store(150, Ordering::Relaxed);
    // one running + one queued fill the admission limit of 2
    let mut slow = Vec::new();
    for _ in 0..2 {
        let h = h.clone();
        slow.push(tokio::spawn(async move { h.chat(json!({"max_tokens": 4, "ignore_eos": true})).await }));
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    let rejected = h.chat(json!({"max_tokens": 2})).await;
    assert_eq!(rejected.status, StatusCode::TOO_MANY_REQUESTS, "{}", rejected.text());
    assert!(rejected.header("retry-after").parse::<u64>().unwrap() >= 1);
    assert_eq!(rejected.json()["error"]["code"], "queue_full");
    h.state.metrics.inject_delay_ms.store(0, Ordering::Relaxed);
    for task in slow {
        assert_eq!(task.await.unwrap().status, StatusCode::OK);
    }
    // the queue drained: new requests are admitted again
    assert_eq!(h.chat(json!({})).await.status, StatusCode::OK);
    assert!(h.get("/metrics").await.text().contains("fllm_queue_rejections_total 1"));
}

#[tokio::test]
async fn slow_requests_hit_the_request_timeout() {
    let mut settings = Settings::default();
    settings.http.request_timeout = Duration::from_millis(150);
    let h = start(settings);
    h.state.metrics.inject_delay_ms.store(100, Ordering::Relaxed);
    let reply = h.chat(json!({"max_tokens": 20, "ignore_eos": true})).await;
    assert_eq!(reply.status, StatusCode::GATEWAY_TIMEOUT, "{}", reply.text());
    assert_eq!(reply.json()["error"]["code"], "request_timeout");
    h.state.metrics.inject_delay_ms.store(0, Ordering::Relaxed);
    // the cancelled request does not keep the worker busy: after the step it was in has ended, a new request is served (a slow machine may
    // still be inside that step for a moment, and such a request would itself hit the 150 ms limit)
    let mut status = StatusCode::GATEWAY_TIMEOUT;
    for _ in 0..40 {
        status = h.chat(json!({})).await.status;
        if status == StatusCode::OK {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(status, StatusCode::OK);
    assert!(h.get("/metrics").await.text().contains("fllm_http_request_timeouts_total"));
}

// ---------------------------------------------------------------------------------------------------------------
// Authentication and rate limits

#[tokio::test]
async fn keys_are_checked_on_every_route_but_the_open_ones() {
    let h = start(with_keys(&["alpha", "beta"]));
    // open
    for path in ["/health", "/ready", "/"] {
        assert_eq!(h.get(path).await.status, StatusCode::OK, "{path}");
    }
    // closed without a key, in the dialect of the route
    let body = h.chat(json!({})).await;
    assert_eq!(body.status, StatusCode::UNAUTHORIZED);
    assert_eq!(body.json()["error"]["code"], "missing_api_key");
    assert_eq!(body.header("www-authenticate"), "Bearer");
    for path in ["/metrics", "/v1/models", "/api/tags", "/api/version", "/get_model_info", "/props", "/info", "/version", "/slots"] {
        assert_eq!(h.get(path).await.status, StatusCode::UNAUTHORIZED, "{path} must need a key");
    }
    assert!(h.get("/api/tags").await.json()["error"].is_string());
    let anthropic = h.post("/v1/messages", json!({"max_tokens": 1, "messages": []})).await;
    assert_eq!(anthropic.status, StatusCode::UNAUTHORIZED);
    assert_eq!(anthropic.json()["error"]["type"], "authentication_error");
    // wrong keys
    let wrong = h.post_with("/v1/chat/completions", json!({}), &[("authorization", "Bearer alph")]).await;
    assert_eq!(wrong.status, StatusCode::UNAUTHORIZED);
    assert_eq!(wrong.json()["error"]["code"], "invalid_api_key");
    // right keys, every header spelling
    for headers in [
        vec![("authorization", "Bearer alpha")],
        vec![("authorization", "Bearer beta")],
        vec![("x-api-key", "beta")],
    ] {
        let reply = h.post_with("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}), &headers).await;
        assert_eq!(reply.status, StatusCode::OK, "{headers:?}: {}", reply.text());
    }
    let tags = h.call(Method::GET, "/api/tags", None, &[("authorization", "Bearer alpha")]).await;
    assert_eq!(tags.status, StatusCode::OK);
}

#[tokio::test]
async fn encoder_and_audio_routes_need_a_key() {
    let h = start(with_keys(&["alpha"]));
    for path in ["/pooling", "/classify", "/v1/score", "/v1/rerank", "/v1/audio/transcriptions", "/v1/audio/translations"] {
        let without = h.post(path, json!({})).await;
        assert_eq!(without.status, StatusCode::UNAUTHORIZED, "{path} must need a key");
        let wrong = h.post_with(path, json!({}), &[("authorization", "Bearer nope")]).await;
        assert_eq!(wrong.status, StatusCode::UNAUTHORIZED, "{path} must refuse a wrong key");
        let right = h.post_with(path, json!({}), &[("authorization", "Bearer alpha")]).await;
        assert_ne!(right.status, StatusCode::UNAUTHORIZED, "{path} must accept the right key");
    }
}

#[tokio::test]
async fn public_routes_can_be_widened() {
    let mut settings = with_keys(&["k"]);
    settings.http.public_routes = vec!["/metrics".into(), "/api/*".into()];
    let h = start(settings);
    assert_eq!(h.get("/metrics").await.status, StatusCode::OK);
    assert_eq!(h.get("/api/tags").await.status, StatusCode::OK);
    assert_eq!(h.get("/v1/models").await.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn rate_limits_apply_per_key() {
    let mut settings = with_keys(&["a", "b"]);
    settings.http.rate_limit = options::RateLimitSpec { requests_per_minute: Some(2.0), tokens_per_minute: None };
    let h = start(settings);
    let call = |key: &'static str| {
        let h = &h;
        async move {
            h.post_with("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}), &[("authorization", &format!("Bearer {key}"))])
                .await
        }
    };
    assert_eq!(call("a").await.status, StatusCode::OK);
    assert_eq!(call("a").await.status, StatusCode::OK);
    let limited = call("a").await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert!(limited.header("retry-after").parse::<u64>().unwrap() >= 1);
    assert_eq!(limited.json()["error"]["code"], "rate_limit_exceeded");
    assert_eq!(call("b").await.status, StatusCode::OK, "the second key has its own budget");
    let metrics = h
        .call(Method::GET, "/metrics", None, &[("authorization", "Bearer b")])
        .await
        .text();
    assert!(metrics.contains("fllm_rate_limited_total 1"), "{metrics}");
}

#[tokio::test]
async fn token_budget_is_charged_with_the_tokens_used() {
    let mut settings = Settings::default();
    settings.http.rate_limit = options::RateLimitSpec { requests_per_minute: None, tokens_per_minute: Some(30.0) };
    let h = start(settings);
    let first = h.chat(json!({"max_tokens": 40, "ignore_eos": true})).await;
    assert_eq!(first.status, StatusCode::OK, "{}", first.text());
    let second = h.chat(json!({"max_tokens": 1})).await;
    assert_eq!(second.status, StatusCode::TOO_MANY_REQUESTS, "40 generated tokens overdrew a budget of 30");
}

// ---------------------------------------------------------------------------------------------------------------
// CORS

#[tokio::test]
async fn cors_preflight_and_headers() {
    let mut settings = Settings::default();
    settings.http.cors_origins = vec!["https://app.example".into()];
    let h = start(settings);
    let preflight = |origin: &'static str| {
        let h = &h;
        async move {
            h.call(
                Method::OPTIONS,
                "/v1/chat/completions",
                None,
                &[
                    ("origin", origin),
                    ("access-control-request-method", "POST"),
                    ("access-control-request-headers", "authorization,content-type,x-api-key"),
                ],
            )
            .await
        }
    };
    let ok = preflight("https://app.example").await;
    assert_eq!(ok.status, StatusCode::NO_CONTENT);
    assert_eq!(ok.header("access-control-allow-origin"), "https://app.example");
    assert!(ok.header("access-control-allow-methods").contains("POST"));
    assert_eq!(ok.header("access-control-allow-headers"), "authorization,content-type,x-api-key");
    assert!(ok.header("vary").contains("Origin"));
    let refused = preflight("https://evil.example").await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    assert!(refused.headers.get("access-control-allow-origin").is_none());
    // the real request carries the header too
    let real = h.post_with("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}), &[("origin", "https://app.example")]).await;
    assert_eq!(real.header("access-control-allow-origin"), "https://app.example");
    assert!(real.header("access-control-expose-headers").contains("x-request-id"));
    let other = h.post_with("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 1}), &[("origin", "https://evil.example")]).await;
    assert!(other.headers.get("access-control-allow-origin").is_none());
    // errors carry it as well, so that a browser app can read them
    let error = h.post_with("/v1/chat/completions", json!({}), &[("origin", "https://app.example")]).await;
    assert_eq!(error.header("access-control-allow-origin"), "https://app.example");
}

#[tokio::test]
async fn cors_is_off_by_default_and_wildcard_works() {
    let h = shared();
    let reply = h
        .call(Method::OPTIONS, "/v1/models", None, &[("origin", "https://x.example"), ("access-control-request-method", "GET")])
        .await;
    assert_eq!(reply.status, StatusCode::FORBIDDEN);
    assert!(reply.json()["error"]["message"].as_str().unwrap().contains("--cors-origins"));
    // same-origin use of the web page needs no CORS headers
    let page = h.call(Method::GET, "/", None, &[("origin", "http://127.0.0.1:8000")]).await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(page.headers.get("access-control-allow-origin").is_none());

    let mut settings = Settings::default();
    settings.http.cors_origins = vec!["*".into()];
    let open = start(settings);
    let reply = open
        .call(Method::OPTIONS, "/v1/messages", None, &[("origin", "https://any.example"), ("access-control-request-method", "POST"), ("access-control-request-private-network", "true")])
        .await;
    assert_eq!(reply.header("access-control-allow-origin"), "*");
    assert_eq!(reply.header("access-control-allow-private-network"), "true");
    assert!(reply.header("access-control-allow-headers").to_ascii_lowercase().contains("x-api-key"));
    assert!(reply.header("access-control-allow-headers").to_ascii_lowercase().contains("anthropic-version"));
}

// ---------------------------------------------------------------------------------------------------------------
// Worker health

#[tokio::test]
async fn ready_follows_the_state_of_the_worker() {
    let h = start(Settings::default());
    let reply = h.get("/ready").await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json()["status"], "ready");
    // /health stays ok in every state
    h.state.ready.store(false, Ordering::Relaxed);
    assert_eq!(h.get("/ready").await.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(h.get("/health").await.status, StatusCode::OK);
    h.state.ready.store(true, Ordering::Relaxed);
    assert_eq!(h.get("/ready").await.status, StatusCode::OK);
    // a worker that stops reporting progress makes the server not ready
    h.state.metrics.stall_ms.store(1, Ordering::Relaxed);
    let mut saw_unready = false;
    for _ in 0..100 {
        if h.get("/ready").await.status == StatusCode::SERVICE_UNAVAILABLE {
            saw_unready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(saw_unready, "a heartbeat older than the stall limit must report 503");
}

#[tokio::test]
async fn a_panic_in_model_code_fails_the_request_and_not_the_process() {
    let h = start(Settings::default());
    h.state.metrics.inject_panic.store(1, Ordering::Relaxed);
    let reply = h.chat(json!({"max_tokens": 8, "ignore_eos": true})).await;
    assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR, "{}", reply.text());
    let message = reply.json()["error"]["message"].as_str().unwrap().to_owned();
    assert!(message.contains("error id err_"), "{message}");
    assert_eq!(reply.json()["error"]["type"], "server_error");
    // streaming requests get the error as an SSE event
    h.state.metrics.inject_panic.store(1, Ordering::Relaxed);
    let stream = h.chat(json!({"stream": true, "max_tokens": 8, "ignore_eos": true})).await;
    assert!(stream.events().iter().any(|(name, data)| name == "error" && data.contains("error id")), "{}", stream.text());
    // the server keeps serving, is still ready and counts the panics
    let after = h.chat(json!({})).await;
    assert_eq!(after.status, StatusCode::OK, "{}", after.text());
    assert_eq!(h.get("/ready").await.status, StatusCode::OK);
    assert!(h.get("/metrics").await.text().contains("fllm_worker_panics_total 2"));
}

// ---------------------------------------------------------------------------------------------------------------
// Metrics

fn metric_value(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find(|line| line.starts_with(series) && line[series.len()..].starts_with(' '))
        .and_then(|line| line[series.len() + 1..].trim().parse().ok())
}

#[tokio::test]
async fn metrics_expose_fllm_and_vllm_series() {
    let h = start(Settings::default());
    let reply = h.chat(json!({"max_tokens": 5, "ignore_eos": true})).await;
    assert_eq!(reply.status, StatusCode::OK);
    let streamed = h.chat(json!({"max_tokens": 5, "ignore_eos": true, "stream": true})).await;
    assert_eq!(streamed.status, StatusCode::OK);
    let metrics = h.get("/metrics").await;
    assert_eq!(metrics.status, StatusCode::OK);
    assert!(metrics.header("content-type").starts_with("text/plain"));
    let text = metrics.text();
    for name in [
        "fllm_requests_total", "fllm_generated_tokens_total", "fllm_kv_cache_usage_ratio", "fllm_prompt_tokens_total",
        "fllm_request_queue_seconds_count", "fllm_request_e2e_seconds_count", "fllm_inter_token_seconds_count",
        "fllm_time_to_first_token_seconds_count", "fllm_worker_panics_total", "fllm_http_connections",
    ] {
        assert!(text.contains(&format!("\n{name} ")) || text.starts_with(&format!("{name} ")), "missing {name}\n{text}");
    }
    let labels = "{model_name=\"tiny\"}";
    for name in [
        "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    ] {
        assert!(metric_value(&text, &format!("{name}{labels}")).is_some(), "missing {name}\n{text}");
    }
    assert_eq!(metric_value(&text, &format!("vllm:generation_tokens_total{labels}")), Some(10.0));
    assert!(metric_value(&text, &format!("vllm:prompt_tokens_total{labels}")).unwrap() > 0.0);
    for histogram in [
        "vllm:time_to_first_token_seconds", "vllm:time_per_output_token_seconds", "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds", "vllm:request_queue_time_seconds",
    ] {
        assert_eq!(metric_value(&text, &format!("{histogram}_count{labels}")).map(|c| c > 0.0), Some(true), "{histogram}\n{text}");
        assert!(text.contains(&format!("{histogram}_bucket{{model_name=\"tiny\",le=\"+Inf\"}}")), "{histogram}");
    }
    // 5 tokens per request: 4 gaps each
    assert_eq!(metric_value(&text, &format!("vllm:time_per_output_token_seconds_count{labels}")), Some(8.0));
    assert_eq!(metric_value(&text, "vllm:request_success_total{model_name=\"tiny\",finished_reason=\"length\"}"), Some(2.0));
    assert!(text.contains("# TYPE vllm:request_success_total counter"));
    assert!(text.contains("# TYPE vllm:num_requests_running gauge"));
}

#[tokio::test]
async fn metrics_style_selects_the_series() {
    for (style, fllm, vllm) in [(MetricsStyle::Fllm, true, false), (MetricsStyle::Vllm, false, true), (MetricsStyle::Both, true, true)] {
        let mut settings = Settings::default();
        settings.http.metrics_style = style;
        let h = start(settings);
        let text = h.get("/metrics").await.text();
        assert_eq!(text.contains("fllm_"), fllm, "{style:?}");
        assert_eq!(text.contains("vllm:"), vllm, "{style:?}");
    }
}

#[tokio::test]
async fn aborted_requests_are_counted_as_aborted() {
    let h = start(Settings::default());
    h.state.metrics.inject_delay_ms.store(30, Ordering::Relaxed);
    let request = h.chat(json!({"max_tokens": 50, "ignore_eos": true}));
    // the client gives up: the future is dropped
    assert!(tokio::time::timeout(Duration::from_millis(100), request).await.is_err());
    h.state.metrics.inject_delay_ms.store(0, Ordering::Relaxed);
    let mut aborted = 0.0;
    for _ in 0..100 {
        let text = h.get("/metrics").await.text();
        aborted = metric_value(&text, "vllm:request_success_total{model_name=\"tiny\",finished_reason=\"abort\"}").unwrap_or(0.0);
        if aborted > 0.0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(aborted, 1.0);
}

// ---------------------------------------------------------------------------------------------------------------
// Sampling defaults

#[tokio::test]
async fn sampling_defaults_apply_when_the_request_omits_them() {
    // 0: greedy by server default, so two identical requests agree
    let settings = Settings { temperature: Some(0.0), ..Settings::default() };
    let greedy = start(settings);
    async fn ask(h: &Harness) -> Reply {
        h.post("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, "ignore_eos": true})).await
    }
    let a = ask(&greedy).await.json()["choices"][0]["message"]["content"].clone();
    let b = ask(&greedy).await.json()["choices"][0]["message"]["content"].clone();
    assert_eq!(a, b);
    // a request value wins over the server default
    let hot = greedy
        .chat(json!({"temperature": 5.0, "max_tokens": 8, "ignore_eos": true, "seed": 3}))
        .await
        .json()["choices"][0]["message"]["content"]
        .clone();
    assert_ne!(hot, a, "temperature 5 on a random model does not repeat the greedy text");
    // no model default: OpenAI's temperature 1.0, so repeated requests differ
    let sampled = shared();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..6 {
        seen.insert(ask(sampled).await.json()["choices"][0]["message"]["content"].to_string());
    }
    assert!(seen.len() > 1, "sampling at temperature 1 returned the same text six times");
}

#[test]
fn the_converted_model_carries_the_checkpoint_sampling_defaults() {
    let mapping = MappedGguf::open(tiny_gguf()).unwrap();
    let defaults = gencfg::GenerationDefaults::from_gguf(&mapping.index().metadata);
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.top_p, Some(0.95));
    assert_eq!(defaults.top_k, Some(20));
    let (resolved, origin) = gencfg::GenerationDefaults::resolve(&gencfg::Source::Auto, &mapping.index().metadata, tiny_gguf()).unwrap();
    assert_eq!(resolved, defaults);
    assert_eq!(origin, "GGUF metadata");
    let (none, _) = gencfg::GenerationDefaults::resolve(&gencfg::Source::Vllm, &mapping.index().metadata, tiny_gguf()).unwrap();
    assert!(none.is_empty());
    let facts = ModelFacts::from_gguf(tiny_gguf(), &mapping);
    assert_eq!(facts.trained_context, tiny_model::CONTEXT);
    assert_eq!(facts.quantization, "F32");
    assert!(facts.parameters > 50_000, "{}", facts.parameters);
}

#[tokio::test]
async fn ollama_routes_sample_with_ollama_defaults_unless_told_otherwise() {
    // Ollama's own defaults (temperature 0.8, top_k 40, top_p 0.9) apply when nothing else is set, so the seed matters;
    // an explicit temperature 0 is greedy and ignores it
    let h = shared();
    let ask = |seed: u64, temperature: Option<f64>| {
        let mut options = json!({"num_predict": 8, "seed": seed});
        if let Some(t) = temperature {
            options["temperature"] = json!(t);
        }
        async move { h.post("/api/generate", json!({"model": "tiny", "prompt": "abc", "stream": false, "options": options})).await.json()["response"].clone() }
    };
    let mut sampled = std::collections::HashSet::new();
    for seed in 1..=6 {
        sampled.insert(ask(seed, None).await.to_string());
    }
    assert!(sampled.len() > 1, "different seeds must give different samples at temperature 0.8");
    assert_eq!(ask(1, Some(0.0)).await, ask(2, Some(0.0)).await);
}

// ---------------------------------------------------------------------------------------------------------------
// Connections: a real listener

struct Listening {
    address: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), String>>,
}

impl Listening {
    async fn shutdown(mut self) -> Result<(), String> {
        let _ = self.stop.take().unwrap().send(());
        self.task.await.unwrap()
    }
}

async fn listen(h: &Harness) -> Listening {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let (router, context, ready) = (h.router.clone(), h.state.http.clone(), h.state.ready.clone());
    let task = tokio::spawn(async move {
        http::serve(listener, router, context, ready, async move {
            let _ = stopped.await;
        })
        .await
    });
    Listening { address, stop: Some(stop), task }
}

async fn read_response(stream: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;
    let mut data = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buf)).await {
            Ok(Ok(0)) | Ok(Err(_)) | Err(_) => break,
            Ok(Ok(n)) => {
                data.extend_from_slice(&buf[..n]);
                if data.windows(4).any(|w| w == b"\r\n\r\n") && (data.ends_with(b"}") || data.ends_with(b"\n")) {
                    break;
                }
            }
        }
    }
    String::from_utf8_lossy(&data).into_owned()
}

/// Whether the peer closes the connection within `within` (reads EOF or an error).
async fn closed_within(stream: &mut tokio::net::TcpStream, within: Duration) -> bool {
    use tokio::io::AsyncReadExt;
    let mut buf = [0u8; 1024];
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match tokio::time::timeout_at(deadline, stream.read(&mut buf)).await {
            Err(_) => return false,
            Ok(Ok(0)) | Ok(Err(_)) => return true,
            Ok(Ok(_)) => {}
        }
    }
}

#[tokio::test]
async fn the_listener_serves_http_and_applies_header_and_idle_timeouts() {
    use tokio::io::AsyncWriteExt;
    let mut settings = Settings::default();
    settings.http.header_read_timeout = Duration::from_millis(400);
    settings.http.keep_alive_timeout = Duration::from_millis(600);
    let h = start(settings);
    let server = listen(&h).await;

    // a normal request over a keep-alive connection
    let mut conn = tokio::net::TcpStream::connect(server.address).await.unwrap();
    conn.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
    let reply = read_response(&mut conn).await;
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.to_ascii_lowercase().contains("x-request-id: req-"));
    // the idle connection is closed after --keep-alive-timeout
    assert!(!closed_within(&mut conn, Duration::from_millis(250)).await, "still within the keep-alive time");
    assert!(closed_within(&mut conn, Duration::from_secs(3)).await, "idle connection must be closed");

    // headers that trickle in are cut off after --header-read-timeout
    let mut slow = tokio::net::TcpStream::connect(server.address).await.unwrap();
    slow.write_all(b"GET /health HTTP/1.1\r\nHo").await.unwrap();
    assert!(closed_within(&mut slow, Duration::from_secs(3)).await, "incomplete header must be dropped");

    // a connection that never sends anything is dropped as well
    let mut silent = tokio::net::TcpStream::connect(server.address).await.unwrap();
    assert!(closed_within(&mut silent, Duration::from_secs(3)).await);

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn the_connection_limit_refuses_with_503() {
    use tokio::io::AsyncWriteExt;
    let mut settings = Settings::default();
    settings.http.max_connections = 1;
    let h = start(settings);
    let server = listen(&h).await;
    let mut first = tokio::net::TcpStream::connect(server.address).await.unwrap();
    first.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
    assert!(read_response(&mut first).await.starts_with("HTTP/1.1 200"));
    let mut second = tokio::net::TcpStream::connect(server.address).await.unwrap();
    let refused = read_response(&mut second).await;
    assert!(refused.starts_with("HTTP/1.1 503"), "{refused}");
    assert!(refused.to_ascii_lowercase().contains("retry-after"));
    drop(first);
    // the slot frees up
    let mut again = None;
    for _ in 0..50 {
        let mut c = tokio::net::TcpStream::connect(server.address).await.unwrap();
        c.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
        if read_response(&mut c).await.starts_with("HTTP/1.1 200") {
            again = Some(c);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(again.is_some());
    let rejected = metric_value(&h.get("/metrics").await.text(), "fllm_http_connections_rejected_total");
    assert!(rejected.is_some_and(|n| n >= 1.0), "{rejected:?}");
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_drains_requests_in_flight() {
    let mut settings = Settings::default();
    settings.http.shutdown_grace = Duration::from_secs(10);
    let h = start(settings);
    h.state.metrics.inject_delay_ms.store(40, Ordering::Relaxed);
    let server = listen(&h).await;
    let url = format!("http://{}/v1/chat/completions", server.address);
    let client = reqwest::Client::new();
    let request = tokio::spawn(async move {
        client
            .post(url)
            .json(&json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, "ignore_eos": true}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(120)).await;
    let address = server.address;
    let stopping = tokio::spawn(server.shutdown());
    tokio::time::sleep(Duration::from_millis(50)).await;
    let body = request.await.unwrap();
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["usage"]["completion_tokens"], 8, "the request in flight must finish: {body}");
    stopping.await.unwrap().unwrap();
    assert!(tokio::net::TcpStream::connect(address).await.is_err(), "the listener is closed after shutdown");
    assert!(!h.state.ready.load(Ordering::Relaxed), "readiness drops when draining starts");
}

#[tokio::test]
async fn the_grace_period_ends_a_stuck_drain() {
    let mut settings = Settings::default();
    settings.http.shutdown_grace = Duration::from_millis(200);
    let h = start(settings);
    h.state.metrics.inject_delay_ms.store(500, Ordering::Relaxed);
    let server = listen(&h).await;
    let url = format!("http://{}/v1/chat/completions", server.address);
    let request = tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .json(&json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, "ignore_eos": true}))
            .send()
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = std::time::Instant::now();
    server.shutdown().await.unwrap();
    assert!(started.elapsed() < Duration::from_secs(3), "shutdown must not wait for the full generation");
    assert!(request.await.unwrap().is_err(), "the request is cut off after the grace period");
    h.state.metrics.inject_delay_ms.store(0, Ordering::Relaxed);
}

#[tokio::test]
async fn tls_listener_serves_https() {
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_owned(), "127.0.0.1".to_owned()]).unwrap();
    let dir = std::env::temp_dir().join(format!("fllm-tls-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("cert.pem"), cert.pem()).unwrap();
    std::fs::write(dir.join("key.pem"), key_pair.serialize_pem()).unwrap();
    let mut settings = with_keys(&["tls-key"]);
    settings.http.tls_cert = Some(dir.join("cert.pem"));
    settings.http.tls_key = Some(dir.join("key.pem"));
    settings.http.validate().unwrap();
    let h = start(settings);
    let server = listen(&h).await;
    let client = reqwest::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_pem(cert.pem().as_bytes()).unwrap())
        .build()
        .unwrap();
    let health = client.get(format!("https://localhost:{}/health", server.address.port())).send().await.unwrap();
    assert_eq!(health.status(), 200);
    let denied = client.get(format!("https://localhost:{}/v1/models", server.address.port())).send().await.unwrap();
    assert_eq!(denied.status(), 401);
    let allowed = client
        .get(format!("https://localhost:{}/v1/models", server.address.port()))
        .bearer_auth("tls-key")
        .send()
        .await
        .unwrap();
    assert_eq!(allowed.status(), 200);
    // plain HTTP to the TLS port gets no answer
    let plain = reqwest::get(format!("http://127.0.0.1:{}/health", server.address.port())).await;
    assert!(plain.is_err());
    server.shutdown().await.unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_bad_certificate_is_reported_before_start() {
    let options = HttpOptions { tls_cert: Some("/nonexistent/cert.pem".into()), tls_key: Some("/nonexistent/key.pem".into()), ..Default::default() };
    let error = options.validate().unwrap_err();
    assert!(error.contains("--tls-cert"), "{error}");
}

#[tokio::test]
async fn access_log_leaves_responses_intact() {
    let mut settings = Settings::default();
    settings.http.access_log = true;
    let h = start(settings);
    let reply = h.chat(json!({"stream": true, "max_tokens": 3, "ignore_eos": true})).await;
    assert!(reply.text().ends_with("data: [DONE]\n\n"));
    let plain = h.chat(json!({})).await;
    assert_eq!(plain.status, StatusCode::OK);
    assert!(plain.json()["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn a_request_that_leaves_the_queue_early_does_not_stop_the_worker() {
    // The scheduler used to keep the id of a cancelled request that was still waiting for a slot; the next step then
    // failed with "unknown sequence" and the worker (and with it the process) ended.
    let settings = Settings { max_seqs: 1, ..Settings::default() };
    let h = Arc::new(start(settings));
    h.state.metrics.inject_delay_ms.store(60, Ordering::Relaxed);
    let busy = {
        let h = h.clone();
        tokio::spawn(async move { h.chat(json!({"max_tokens": 6, "ignore_eos": true})).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    // this request waits for the only slot; its client gives up
    let queued = h.chat(json!({"max_tokens": 6, "ignore_eos": true}));
    assert!(tokio::time::timeout(Duration::from_millis(150), queued).await.is_err());
    h.state.metrics.inject_delay_ms.store(0, Ordering::Relaxed);
    assert_eq!(busy.await.unwrap().status, StatusCode::OK);
    let after = h.chat(json!({})).await;
    assert_eq!(after.status, StatusCode::OK, "{}", after.text());
    assert_eq!(h.get("/ready").await.status, StatusCode::OK);
}

#[test]
fn listen_addresses_accept_ipv4_ipv6_and_names() {
    assert!(listen_address("127.0.0.1", 1).unwrap().is_ipv4());
    assert!(listen_address("::1", 1).unwrap().is_ipv6());
    assert!(listen_address("[::]", 1).unwrap().is_ipv6());
    assert!(listen_address("localhost", 1).is_ok());
    assert!(listen_address("no such host", 1).is_err());
}
