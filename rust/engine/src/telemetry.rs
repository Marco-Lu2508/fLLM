//! A small OTLP/HTTP trace exporter: one span per completed request, sent as JSON to a collector that the user already
//! runs. No SDK, no OpenTelemetry dependency — the wire format is small enough to write by hand, and a local inference
//! server should not pull in a tracing ecosystem for it.
//!
//! Configuration: `FLLM_OTEL_ENDPOINT=http://127.0.0.1:4318` (the collector's HTTP port) and `FLLM_OTEL_SERVICE` (default
//! `fllm`). Without an endpoint the tracer is off and every call is a no-op.
//!
//! Spans are queued and sent in batches by a background thread, so a slow collector never slows a request down. If the
//! queue is full the oldest spans are dropped — a trace is worth less than a token.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_QUEUE: usize = 4096;
const BATCH: usize = 128;

/// One finished unit of work.
#[derive(Debug, Clone)]
pub struct Span {
    pub name: String,
    pub start_unix_ns: u128,
    pub end_unix_ns: u128,
    pub attributes: Vec<(String, Attr)>,
    pub error: bool,
}

#[derive(Debug, Clone)]
pub enum Attr {
    Text(String),
    Int(i64),
    Bool(bool),
}

impl Attr {
    /// The OTLP `AnyValue` of this attribute (the `value` half of a key/value pair).
    fn to_json(&self) -> serde_json::Value {
        match self {
            Attr::Text(value) => serde_json::json!({"stringValue": value}),
            Attr::Int(value) => serde_json::json!({"intValue": value.to_string()}),
            Attr::Bool(value) => serde_json::json!({"boolValue": value}),
        }
    }
}

/// A span being built.
pub struct SpanBuilder {
    name: String,
    started: Instant,
    started_unix_ns: u128,
    attributes: Vec<(String, Attr)>,
    error: bool,
}

impl SpanBuilder {
    pub fn new(name: &str) -> SpanBuilder {
        SpanBuilder {
            name: name.to_owned(),
            started: Instant::now(),
            started_unix_ns: now_ns(),
            attributes: Vec::new(),
            error: false,
        }
    }

    pub fn attr(mut self, key: &str, value: Attr) -> SpanBuilder {
        self.attributes.push((key.to_owned(), value));
        self
    }

    pub fn text(self, key: &str, value: impl Into<String>) -> SpanBuilder {
        self.attr(key, Attr::Text(value.into()))
    }

    pub fn int(self, key: &str, value: i64) -> SpanBuilder {
        self.attr(key, Attr::Int(value))
    }

    pub fn fail(mut self) -> SpanBuilder {
        self.error = true;
        self
    }

    pub fn finish(self) -> Span {
        let elapsed = self.started.elapsed();
        Span {
            name: self.name,
            start_unix_ns: self.started_unix_ns,
            end_unix_ns: self.started_unix_ns + elapsed.as_nanos(),
            attributes: self.attributes,
            error: self.error,
        }
    }
}

fn now_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// The exporter. `None` inside means: off.
pub struct Tracer {
    tx: Option<Sender<Span>>,
    dropped: AtomicU64,
    sent: Arc<AtomicU64>,
    service: String,
}

impl Tracer {
    /// Reads the environment and, when an endpoint is set, starts the sending thread.
    pub fn from_env() -> Tracer {
        let endpoint = std::env::var("FLLM_OTEL_ENDPOINT").unwrap_or_default().trim().to_owned();
        let service = std::env::var("FLLM_OTEL_SERVICE").unwrap_or_else(|_| "fllm".to_owned());
        if endpoint.is_empty() {
            return Tracer { tx: None, dropped: AtomicU64::new(0), sent: Arc::new(AtomicU64::new(0)), service };
        }
        let (tx, rx) = channel();
        let thread_service = service.clone();
        // one counter, shared by the export thread and the stats: two separate ones would never agree
        let sent = Arc::new(AtomicU64::new(0));
        let thread_sent = Arc::clone(&sent);
        std::thread::Builder::new()
            .name("fllm-otel".into())
            .spawn(move || exporter_loop(endpoint, thread_service, rx, thread_sent))
            .ok();
        Tracer { tx: Some(tx), dropped: AtomicU64::new(0), sent, service }
    }

    pub fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    /// Queues a finished span. Never blocks.
    pub fn record(&self, span: Span) {
        if let Some(tx) = &self.tx {
            if tx.send(span).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Spans that left the process.
    pub fn sent(&self) -> u64 {
        self.sent.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.enabled(),
            "service": self.service,
            "sent": self.sent.load(Ordering::Relaxed),
            "dropped": self.dropped.load(Ordering::Relaxed),
        })
    }

    /// Only for tests: count what left the process.
    pub fn mark_sent(&self, count: u64) {
        self.sent.fetch_add(count, Ordering::Relaxed);
    }
}

fn exporter_loop(endpoint: String, service: String, rx: Receiver<Span>, sent: Arc<AtomicU64>) {
    let client = match reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build() {
        Ok(client) => client,
        Err(_) => return,
    };
    let url = format!("{}/v1/traces", endpoint.trim_end_matches('/'));
    let mut batch: Vec<Span> = Vec::new();
    let mut last_send = Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(span) => {
                batch.push(span);
                while let Ok(more) = rx.try_recv() {
                    batch.push(more);
                    if batch.len() >= BATCH {
                        break;
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if !batch.is_empty() && send_batch(&client, &url, &service, &batch) {
                    sent.fetch_add(batch.len() as u64, Ordering::Relaxed);
                }
                return;
            }
        }
        if batch.len() >= BATCH || (!batch.is_empty() && last_send.elapsed() > Duration::from_secs(2)) {
            if send_batch(&client, &url, &service, &batch) {
                sent.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
            batch.clear();
            last_send = Instant::now();
        }
        if batch.len() > MAX_QUEUE {
            batch.drain(..batch.len() - MAX_QUEUE);
        }
    }
}

/// Sends one batch; `true` when the collector accepted it.
fn send_batch(client: &reqwest::blocking::Client, url: &str, service: &str, spans: &[Span]) -> bool {
    let payload = payload(service, spans);
    // a trace that cannot be sent is lost, the request is not: the answer is only interesting as a yes/no
    client.post(url).json(&payload).send().map(|r| r.status().is_success()).unwrap_or(false)
}

/// The OTLP/HTTP JSON shape. Exposed for the test, which checks the wire format.
pub fn payload(service: &str, spans: &[Span]) -> serde_json::Value {
    let items: Vec<serde_json::Value> = spans
        .iter()
        .map(|span| {
            let mut attributes: Vec<serde_json::Value> = vec![serde_json::json!({"key": "service.name", "value": {"stringValue": service}})];
            for (key, value) in &span.attributes {
                attributes.push(serde_json::json!({"key": key, "value": value.to_json()}));
            }
            serde_json::json!({
                "traceId": trace_id(span.start_unix_ns),
                "spanId": span_id(span.start_unix_ns, &span.name),
                "name": span.name,
                "kind": 2, // SPAN_KIND_SERVER
                "startTimeUnixNano": span.start_unix_ns.to_string(),
                "endTimeUnixNano": span.end_unix_ns.to_string(),
                "attributes": attributes,
                "status": if span.error { serde_json::json!({"code": 2}) } else { serde_json::json!({"code": 1}) },
            })
        })
        .collect();
    serde_json::json!({
        "resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service}}]},
            "scopeSpans": [{"scope": {"name": "fllm"}, "spans": items}],
        }]
    })
}

/// A trace id derived from the start time, so spans of one second share a trace.
fn trace_id(nanos: u128) -> String {
    format!("{:032x}", nanos / 1_000_000_000)
}

/// A span id derived from start time and name: unique enough for a single process.
fn span_id(nanos: u128, name: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in nanos.to_le_bytes().iter().chain(name.as_bytes()) {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_keeps_its_duration_and_attributes() {
        let span = SpanBuilder::new("chat")
            .text("model", "llama")
            .int("prompt_tokens", 512)
            .attr("stream", Attr::Bool(false))
            .finish();
        assert_eq!(span.name, "chat");
        assert!(span.end_unix_ns >= span.start_unix_ns);
        assert_eq!(span.attributes.len(), 3);
        assert!(!span.error);
    }

    #[test]
    fn the_payload_has_the_otlp_shape() {
        let span = SpanBuilder::new("chat").text("model", "llama").finish();
        let payload = payload("fllm", &[span]);
        let spans = &payload["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert_eq!(spans[0]["name"], "chat");
        assert_eq!(spans[0]["kind"], 2);
        assert_eq!(spans[0]["traceId"].as_str().unwrap().len(), 32);
        assert_eq!(spans[0]["spanId"].as_str().unwrap().len(), 16);
        let attributes = spans[0]["attributes"].as_array().unwrap();
        let model = attributes.iter().find(|a| a["key"] == "model").expect("the model attribute");
        assert_eq!(model["value"]["stringValue"], "llama", "OTLP shape: key next to a value object");
        assert!(attributes.iter().all(|a| a.get("value").is_some() && a.get("key").is_some()));
        assert!(attributes.iter().any(|a| a["key"] == "service.name"));
    }

    #[test]
    fn an_error_span_is_marked() {
        let span = SpanBuilder::new("chat").fail().finish();
        let payload = payload("fllm", &[span]);
        assert_eq!(payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"], 2);
    }

    #[test]
    fn without_an_endpoint_the_tracer_is_off() {
        // the environment of the test process decides; only the off case is asserted here
        let tracer = Tracer { tx: None, dropped: AtomicU64::new(0), sent: Arc::new(AtomicU64::new(0)), service: "fllm".into() };
        assert!(!tracer.enabled());
        tracer.record(SpanBuilder::new("x").finish());
        assert_eq!(tracer.stats()["dropped"], 0);
    }

    #[test]
    fn trace_ids_group_by_second_and_span_ids_differ_per_name() {
        let base = 1_700_000_000_000_000_000u128;
        assert_eq!(trace_id(base), trace_id(base + 500_000_000));
        assert_ne!(trace_id(base), trace_id(base + 1_500_000_000));
        assert_ne!(span_id(base, "chat"), span_id(base, "completion"));
    }
}
