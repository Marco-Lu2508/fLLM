//! Per-server Prometheus counters and inference observations, without exporter dependencies.
use axum::{
    body::{Body, HttpBody},
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::{
    fmt::Write,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Instant,
};

const BOUNDS: [f64; 13] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
];
struct Histogram {
    bins: [AtomicU64; 14],
    sum_ns: AtomicU64,
}
impl Histogram {
    fn new() -> Self {
        Self {
            bins: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_ns: AtomicU64::new(0),
        }
    }
    fn observe(&self, seconds: f64) {
        let bin = BOUNDS
            .iter()
            .position(|&bound| seconds <= bound)
            .unwrap_or(BOUNDS.len());
        self.bins[bin].fetch_add(1, Relaxed);
        self.sum_ns.fetch_add((seconds * 1e9) as u64, Relaxed);
    }
    fn render(&self, name: &str, help: &str, out: &mut String) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut count = 0;
        for (i, bin) in self.bins.iter().enumerate() {
            count += bin.load(Relaxed);
            let bound = BOUNDS
                .get(i)
                .map_or_else(|| "+Inf".into(), |n| n.to_string());
            let _ = writeln!(out, "{name}_bucket{{le=\"{bound}\"}} {count}");
        }
        let _ = writeln!(
            out,
            "{name}_sum {:.9}\n{name}_count {count}",
            self.sum_ns.load(Relaxed) as f64 / 1e9
        );
    }
}

pub struct Metrics {
    started: Instant,
    requests: AtomicU64,
    queued: AtomicU64,
    running: AtomicU64,
    waiting: AtomicU64,
    processed: AtomicU64,
    generated: AtomicU64,
    mtp_proposed: AtomicU64,
    mtp_accepted: AtomicU64,
    prefix_hits: AtomicU64,
    kv_used: AtomicU64,
    kv_total: AtomicU64,
    errors: [AtomicU64; 600],
    rate_buckets: [AtomicU64; 10],
    first_token: Histogram,
    duration: Histogram,
}
impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
impl Metrics {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            requests: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            running: AtomicU64::new(0),
            waiting: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            generated: AtomicU64::new(0),
            mtp_proposed: AtomicU64::new(0),
            mtp_accepted: AtomicU64::new(0),
            prefix_hits: AtomicU64::new(0),
            kv_used: AtomicU64::new(0),
            kv_total: AtomicU64::new(0),
            errors: std::array::from_fn(|_| AtomicU64::new(0)),
            rate_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            first_token: Histogram::new(),
            duration: Histogram::new(),
        }
    }
    pub fn queue(&self) {
        self.queued.fetch_add(1, Relaxed);
    }
    pub fn dequeue(&self) {
        self.queued.fetch_sub(1, Relaxed);
    }
    pub fn mtp_proposed(&self, n: usize) {
        self.mtp_proposed.fetch_add(n as u64, Relaxed);
    }
    pub fn mtp_accepted(&self, n: usize) {
        self.mtp_accepted.fetch_add(n as u64, Relaxed);
    }
    pub fn processed(&self, n: usize) {
        self.processed.fetch_add(n as u64, Relaxed);
    }
    pub fn generated(&self) {
        self.generated.fetch_add(1, Relaxed);
        // One atomic word holds a seconds-since-start stamp and its token count.
        let second = self.started.elapsed().as_secs();
        let stamp = (second + 1) << 32;
        let bucket = &self.rate_buckets[second as usize % 10];
        #[allow(deprecated)] // renamed to try_update in newer toolchains
        let _ = bucket.fetch_update(Relaxed, Relaxed, |old| {
            Some(if old & !0xffff_ffff == stamp {
                stamp | (old as u32).saturating_add(1) as u64
            } else {
                stamp | 1
            })
        });
    }
    pub fn snapshot(&self, running: usize, waiting: usize, free: usize, total: usize, hits: u64) {
        self.running.store(running as u64, Relaxed);
        self.waiting.store(waiting as u64, Relaxed);
        self.kv_total.store(total as u64, Relaxed);
        self.kv_used
            .store(total.saturating_sub(free) as u64, Relaxed);
        self.prefix_hits.store(hits, Relaxed);
    }
    pub fn generation(self: &Arc<Self>) -> Generation {
        Generation {
            metrics: self.clone(),
            started: Instant::now(),
            first: None,
            tokens: 0,
        }
    }
    pub fn embedding(self: &Arc<Self>) -> Embedding {
        self.running.fetch_add(1, Relaxed);
        Embedding(self.clone())
    }
    pub fn render(&self) -> String {
        let mut out = String::new();
        let values = [
            (
                "fllm_api_requests_total",
                "counter",
                "API requests received",
                self.requests.load(Relaxed),
            ),
            (
                "fllm_requests_running",
                "gauge",
                "Inference requests executing",
                self.running.load(Relaxed),
            ),
            (
                "fllm_requests_waiting",
                "gauge",
                "Inference requests queued for execution",
                self.waiting.load(Relaxed) + self.queued.load(Relaxed),
            ),
            (
                "fllm_processed_tokens_total",
                "counter",
                "Prompt token rows processed, including recomputation",
                self.processed.load(Relaxed),
            ),
            (
                "fllm_generated_tokens_total",
                "counter",
                "Tokens generated by the inference runtime",
                self.generated.load(Relaxed),
            ),
            (
                "fllm_prefix_cache_hits_total",
                "counter",
                "Successful admissions reusing one or more prefix blocks",
                self.prefix_hits.load(Relaxed),
            ),
            (
                "fllm_kv_cache_blocks_used",
                "gauge",
                "Allocated KV blocks, including retained prefixes",
                self.kv_used.load(Relaxed),
            ),
            (
                "fllm_kv_cache_blocks_total",
                "gauge",
                "KV block pool capacity",
                self.kv_total.load(Relaxed),
            ),
        ];
        for (name, kind, help, value) in values {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}"
            );
        }
        let proposed = self.mtp_proposed.load(Relaxed);
        let accepted = self.mtp_accepted.load(Relaxed);
        let _ = writeln!(out, "# HELP fllm_mtp_proposed_tokens_total MTP candidates actually scheduled for target verification\n# TYPE fllm_mtp_proposed_tokens_total counter\nfllm_mtp_proposed_tokens_total {proposed}\n# HELP fllm_mtp_accepted_tokens_total MTP candidates accepted by the target\n# TYPE fllm_mtp_accepted_tokens_total counter\nfllm_mtp_accepted_tokens_total {accepted}\n# HELP fllm_mtp_acceptance_ratio Accepted MTP candidates divided by proposed candidates\n# TYPE fllm_mtp_acceptance_ratio gauge\nfllm_mtp_acceptance_ratio {:.6}", if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 });
        let now = self.started.elapsed();
        let second = now.as_secs();
        let tokens: u64 = self
            .rate_buckets
            .iter()
            .map(|bucket| {
                let word = bucket.load(Relaxed);
                let stamp = word >> 32;
                if stamp > second.saturating_sub(9) && stamp <= second + 1 {
                    word as u32 as u64
                } else {
                    0
                }
            })
            .sum();
        let rate = tokens as f64 / now.as_secs_f64().clamp(1.0, 10.0);
        let _ = writeln!(out, "# HELP fllm_tokens_per_second Generated tokens per second over the last ten seconds\n# TYPE fllm_tokens_per_second gauge\nfllm_tokens_per_second {rate:.6}");
        out.push_str("# HELP fllm_request_errors_total API responses with HTTP error status\n# TYPE fllm_request_errors_total counter\n");
        for status in 400..600 {
            let n = self.errors[status].load(Relaxed);
            if n > 0 {
                let _ = writeln!(out, "fllm_request_errors_total{{status=\"{status}\"}} {n}");
            }
        }
        self.first_token.render(
            "fllm_time_to_first_token_seconds",
            "Inference queue and compute latency until the first generated token",
            &mut out,
        );
        self.duration.render(
            "fllm_request_duration_seconds",
            "API latency through body completion or client cancellation",
            &mut out,
        );
        out
    }
}

pub struct Generation {
    metrics: Arc<Metrics>,
    started: Instant,
    first: Option<Instant>,
    tokens: usize,
}
impl Generation {
    pub fn token(&mut self) {
        if self.first.is_none() {
            self.metrics
                .first_token
                .observe(self.started.elapsed().as_secs_f64());
            self.first = Some(Instant::now());
        }
        self.tokens += 1;
    }
    pub fn timings(&self, prompt: usize) -> serde_json::Value {
        let prompt_secs = self
            .first
            .map_or_else(
                || self.started.elapsed(),
                |t| t.duration_since(self.started),
            )
            .as_secs_f64();
        let decode_secs = self.first.map_or(0.0, |t| t.elapsed().as_secs_f64());
        serde_json::json!({"prompt_n":prompt,"prompt_ms":prompt_secs * 1000.0,"prompt_per_second":if prompt_secs > 0.0 { prompt as f64 / prompt_secs } else { 0.0 },"predicted_n":self.tokens,"predicted_ms":decode_secs * 1000.0,"predicted_per_second":if decode_secs > 0.0 { self.tokens.saturating_sub(1) as f64 / decode_secs } else { 0.0 }})
    }
}
pub struct Embedding(Arc<Metrics>);
impl Drop for Embedding {
    fn drop(&mut self) {
        self.0.running.fetch_sub(1, Relaxed);
    }
}
struct Lifetime {
    metrics: Arc<Metrics>,
    started: Instant,
}
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.metrics
            .duration
            .observe(self.started.elapsed().as_secs_f64());
    }
}

/// Observe response lifetimes as well as immediate errors; SSE bodies remain streaming.
pub async fn observe(
    State(metrics): State<Arc<Metrics>>,
    request: Request,
    next: Next,
) -> Response {
    if !request.uri().path().starts_with("/v1/")
        && !matches!(
            request.uri().path(),
            "/generate"
                | "/api/chat"
                | "/api/generate"
                | "/api/embed"
                | "/api/embeddings"
                | "/tokenize"
                | "/detokenize"
        )
    {
        return next.run(request).await;
    }
    metrics.requests.fetch_add(1, Relaxed);
    let lifetime = Lifetime {
        metrics: metrics.clone(),
        started: Instant::now(),
    };
    let response = next.run(request).await;
    let status = response.status().as_u16() as usize;
    if (400..600).contains(&status) {
        metrics.errors[status].fetch_add(1, Relaxed);
    }
    let (parts, mut body) = response.into_parts();
    let stream = async_stream::stream! {
        let _lifetime = lifetime;
        while let Some(frame) = std::future::poll_fn(|cx| std::pin::Pin::new(&mut body).poll_frame(cx)).await {
            match frame { Ok(frame) => if let Ok(data) = frame.into_data() { yield Ok::<_, axum::Error>(data); }, Err(e) => { yield Err(e); break; } }
        }
    };
    Response::from_parts(parts, Body::from_stream(stream))
}
