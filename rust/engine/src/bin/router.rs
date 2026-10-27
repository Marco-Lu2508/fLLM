//! `fllm-router`: a small HTTP router in front of several fLLM servers.
//!
//! Why it exists: the prefix cache of a server only helps the requests that land on *that* server. A plain round robin
//! spreads one conversation over every backend and every one of them has to prefill the same system prompt. This router
//! hashes the **beginning of the prompt** (system message and first user message) and always sends the same prefix to the
//! same backend, so the cache keeps working while the load is still spread over the fleet.
//!
//! It is a proxy, not a queue: the body is forwarded unchanged, streaming answers are passed through chunk by chunk, and a
//! backend that fails is skipped once and marked unhealthy for a while.
//!
//! ```text
//! fllm-router --listen 127.0.0.1:8080 --backend http://10.0.0.5:8000 --backend http://10.0.0.6:8000
//! ```
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How many characters of the prompt decide the backend. Long enough to separate conversations, short enough that a
/// different first message moves to another server.
const PREFIX_CHARS: usize = 2048;
/// A backend that failed is not used for this long.
const UNHEALTHY_FOR: Duration = Duration::from_secs(15);

struct Backend {
    url: String,
    healthy_until: Mutex<Option<Instant>>,
    requests: AtomicU64,
    failures: AtomicU64,
}

impl Backend {
    fn new(url: String) -> Backend {
        Backend { url, healthy_until: Mutex::new(None), requests: AtomicU64::new(0), failures: AtomicU64::new(0) }
    }

    fn healthy(&self) -> bool {
        let guard = self.healthy_until.lock().unwrap_or_else(|e| e.into_inner());
        match *guard {
            Some(until) => Instant::now() > until,
            None => true,
        }
    }

    fn fail(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.healthy_until.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(Instant::now() + UNHEALTHY_FOR);
    }

    fn ok(&self) {
        let mut guard = self.healthy_until.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }
}

struct State_ {
    backends: Vec<Arc<Backend>>,
    client: reqwest::Client,
    api_key: Option<String>,
    routed: AtomicU64,
}

/// The text whose hash decides the backend: system message, then the first user message, then the raw prompt.
fn prefix_text(body: &Value) -> String {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        let mut out = String::new();
        for message in messages {
            let role = message.get("role").and_then(Value::as_str).unwrap_or("");
            let text = match message.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };
            if text.is_empty() {
                continue;
            }
            out.push_str(&text);
            out.push('\u{1f}');
            if out.chars().count() >= PREFIX_CHARS {
                break;
            }
            // a system message and the first user message are the stable part; a second user turn already depends on them
            if role == "user" {
                break;
            }
        }
        return out.chars().take(PREFIX_CHARS).collect();
    }
    body.get("prompt")
        .and_then(Value::as_str)
        .map(|p| p.chars().take(PREFIX_CHARS).collect())
        .unwrap_or_default()
}

/// Which backend takes this prefix: a hash, mapped to the healthy list.
fn pick(prefix: &str, backends: &[Arc<Backend>]) -> Option<Arc<Backend>> {
    let healthy: Vec<Arc<Backend>> = backends.iter().filter(|b| b.healthy()).cloned().collect();
    if healthy.is_empty() {
        return None;
    }
    let digest = Sha256::digest(prefix.as_bytes());
    let mut slot = [0u8; 8];
    slot.copy_from_slice(&digest[..8]);
    let index = (u64::from_le_bytes(slot) % healthy.len() as u64) as usize;
    Some(healthy[index].clone())
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, Json(json!({"error": {"message": "missing or wrong API key", "type": "invalid_request_error"}}))).into_response()
}

async fn proxy(State(state): State<Arc<State_>>, request: Request) -> Response {
    if let Some(expected) = &state.api_key {
        let bearer = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::to_owned);
        let given = bearer.or_else(|| request.headers().get("x-api-key").and_then(|v| v.to_str().ok()).map(str::to_owned));
        if given.as_deref() != Some(expected.as_str()) {
            return unauthorized();
        }
    }
    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, 64 << 20).await {
        Ok(bytes) => bytes,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": format!("cannot read the body: {e}")}}))).into_response(),
    };
    let parsed = serde_json::from_slice::<Value>(&bytes).ok();
    let prefix = parsed.as_ref().map(prefix_text).unwrap_or_default();
    let Some(first) = pick(&prefix, &state.backends) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": {"message": "no healthy backend"}}))).into_response();
    };
    let mut candidates = vec![first.clone()];
    for backend in &state.backends {
        if backend.url != first.url && backend.healthy() {
            candidates.push(backend.clone());
        }
    }
    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST);
    let mut last_error = String::from("no backend answered");
    for backend in candidates {
        let mut url = format!("{}{}", backend.url.trim_end_matches('/'), parts.uri.path());
        if let Some(query) = parts.uri.query() {
            url.push('?');
            url.push_str(query);
        }
        let builder = state.client.request(method.clone(), url);
        let mut headers = HeaderMap::new();
        for (name, value) in parts.headers.iter() {
            if name == axum::http::header::HOST || name == axum::http::header::CONTENT_LENGTH {
                continue;
            }
            headers.insert(name.clone(), value.clone());
        }
        let response = builder.headers(headers).body(bytes.clone()).send().await;
        match response {
            Ok(response) => {
                let status = StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                if status.is_server_error() {
                    backend.fail();
                    last_error = format!("{} answered {status}", backend.url);
                    continue;
                }
                backend.ok();
                backend.requests.fetch_add(1, Ordering::Relaxed);
                state.routed.fetch_add(1, Ordering::Relaxed);
                // the headers are copied before the body is taken: streaming moves the response
                let copied: Vec<(axum::http::HeaderName, HeaderValue)> =
                    response.headers().iter().map(|(name, value)| (name.clone(), value.clone())).collect();
                let mut out = Response::new(Body::from_stream(response.bytes_stream()));
                *out.status_mut() = status;
                for (name, value) in copied {
                    out.headers_mut().insert(name, value);
                }
                out.headers_mut().insert("x-fllm-backend", HeaderValue::from_str(&backend.url).unwrap_or(HeaderValue::from_static("?")));
                return out;
            }
            Err(e) => {
                backend.fail();
                last_error = format!("{} did not answer: {e}", backend.url);
            }
        }
    }
    (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": last_error}}))).into_response()
}

async fn stats(State(state): State<Arc<State_>>) -> Response {
    let backends: Vec<Value> = state
        .backends
        .iter()
        .map(|b| {
            json!({
                "url": b.url,
                "healthy": b.healthy(),
                "requests": b.requests.load(Ordering::Relaxed),
                "failures": b.failures.load(Ordering::Relaxed),
            })
        })
        .collect();
    Json(json!({
        "routed": state.routed.load(Ordering::Relaxed),
        "backends": backends,
        "prefix_chars": PREFIX_CHARS,
    }))
    .into_response()
}

/// `GET /router/metrics`: the same numbers as `/router/stats`, in the format Prometheus scrapes.
async fn metrics(State(state): State<Arc<State_>>) -> Response {
    let mut out = String::new();
    out.push_str("# HELP fllm_router_routed_total Requests the router forwarded\n");
    out.push_str("# TYPE fllm_router_routed_total counter\n");
    out.push_str(&format!("fllm_router_routed_total {}\n", state.routed.load(Ordering::Relaxed)));
    out.push_str("# HELP fllm_router_backend_requests_total Requests forwarded to one backend\n");
    out.push_str("# TYPE fllm_router_backend_requests_total counter\n");
    for backend in &state.backends {
        out.push_str(&format!(
            "fllm_router_backend_requests_total{{backend=\"{}\",}} {}\n",
            backend.url,
            backend.requests.load(Ordering::Relaxed)
        ));
    }
    out.push_str("# HELP fllm_router_backend_failures_total Failures per backend\n");
    out.push_str("# TYPE fllm_router_backend_failures_total counter\n");
    for backend in &state.backends {
        out.push_str(&format!(
            "fllm_router_backend_failures_total{{backend=\"{}\",}} {}\n",
            backend.url,
            backend.failures.load(Ordering::Relaxed)
        ));
    }
    out.push_str("# HELP fllm_router_backend_healthy Whether the backend is used right now (1) or was skipped (0)\n");
    out.push_str("# TYPE fllm_router_backend_healthy gauge\n");
    for backend in &state.backends {
        out.push_str(&format!(
            "fllm_router_backend_healthy{{backend=\"{}\",}} {}\n",
            backend.url,
            u8::from(backend.healthy())
        ));
    }
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut listen = "127.0.0.1:8080".to_string();
    let mut backends: Vec<Arc<Backend>> = Vec::new();
    let mut api_key: Option<String> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--listen" => {
                listen = args.get(index + 1).cloned().unwrap_or(listen);
                index += 2;
            }
            "--backend" => {
                if let Some(url) = args.get(index + 1) {
                    backends.push(Arc::new(Backend::new(url.trim_end_matches('/').to_owned())));
                }
                index += 2;
            }
            "--api-key" => {
                api_key = args.get(index + 1).cloned();
                index += 2;
            }
            "--help" | "-h" => {
                println!("usage: fllm-router --listen 127.0.0.1:8080 --backend http://host:port [--backend ...] [--api-key KEY]");
                return Ok(());
            }
            _ => index += 1,
        }
    }
    if backends.is_empty() {
        eprintln!("no backend given: --backend http://host:port (repeat it for every server)");
        std::process::exit(2);
    }
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(600)) // long generations must not be cut off
        .build()?;
    let state = Arc::new(State_ {
        backends: backends.clone(),
        client,
        api_key,
        routed: AtomicU64::new(0),
    });
    let app = Router::new()
        .route("/router/stats", get(stats))
        .route("/router/metrics", get(metrics))
        .fallback(any(proxy))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    println!("fllm-router listening on http://{listen} with {} backend(s)", backends.len());
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn backends(urls: &[&str]) -> Vec<Arc<Backend>> {
        urls.iter().map(|url| Arc::new(Backend::new((*url).to_owned()))).collect()
    }

    #[test]
    fn the_same_prefix_always_lands_on_the_same_backend() {
        let list = backends(&["http://a", "http://b", "http://c"]);
        let first = pick("You are a helpful assistant", &list).unwrap();
        for _ in 0..20 {
            assert_eq!(pick("You are a helpful assistant", &list).unwrap().url, first.url);
        }
    }

    #[test]
    fn different_prefixes_spread_over_the_backends() {
        let list = backends(&["http://a", "http://b", "http://c"]);
        let mut seen = HashMap::new();
        for i in 0..64 {
            let backend = pick(&format!("system prompt number {i}"), &list).unwrap();
            *seen.entry(backend.url.clone()).or_insert(0) += 1;
        }
        assert_eq!(seen.len(), 3, "all backends are used: {seen:?}");
    }

    #[test]
    fn the_prefix_is_system_plus_first_user_turn() {
        let body = json!({"messages": [
            {"role": "system", "content": "Du bist knapp."},
            {"role": "user", "content": "Wie geht's?"},
            {"role": "assistant", "content": "Gut."},
            {"role": "user", "content": "Und weiter?"}
        ]});
        let text = prefix_text(&body);
        assert!(text.contains("Du bist knapp."));
        assert!(text.contains("Wie geht's?"));
        assert!(!text.contains("Und weiter?"), "later turns must not move the routing: {text}");
    }

    #[test]
    fn a_completion_prompt_works_too() {
        assert_eq!(prefix_text(&json!({"prompt": "ein satz"})), "ein satz");
    }

    #[test]
    fn a_failed_backend_is_skipped() {
        let list = backends(&["http://a", "http://b"]);
        list[0].fail();
        for i in 0..32 {
            let backend = pick(&format!("prefix {i}"), &list).unwrap();
            assert_eq!(backend.url, "http://b");
        }
    }

    #[test]
    fn without_a_healthy_backend_there_is_no_route() {
        let list = backends(&["http://a"]);
        list[0].fail();
        assert!(pick("x", &list).is_none());
    }
}
