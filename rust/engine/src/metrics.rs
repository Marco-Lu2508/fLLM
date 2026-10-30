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
        Arc, OnceLock,
    },
    time::Instant,
};

const BOUNDS: [f64; 16] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];
struct Histogram {
    bins: [AtomicU64; 17],
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
    /// `labels` is empty or `key="value"`; it goes in front of `le` and on the sum and count lines.
    fn render(&self, name: &str, help: &str, labels: &str, out: &mut String) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let separator = if labels.is_empty() { "" } else { "," };
        let mut count = 0;
        for (i, bin) in self.bins.iter().enumerate() {
            count += bin.load(Relaxed);
            let bound = BOUNDS
                .get(i)
                .map_or_else(|| "+Inf".into(), |n| n.to_string());
            let _ = writeln!(out, "{name}_bucket{{{labels}{separator}le=\"{bound}\"}} {count}");
        }
        let braces = if labels.is_empty() { String::new() } else { format!("{{{labels}}}") };
        let _ = writeln!(
            out,
            "{name}_sum{braces} {:.9}\n{name}_count{braces} {count}",
            self.sum_ns.load(Relaxed) as f64 / 1e9
        );
    }
}

/// `finished_reason` label values and their slot in `Metrics::finished`.
const FINISH_REASONS: [(&str, usize); 3] = [("stop", 0), ("length", 1), ("abort", 2)];

pub struct Metrics {
    started: Instant,
    requests: AtomicU64,
    queued: AtomicU64,
    running: AtomicU64,
    waiting: AtomicU64,
    processed: AtomicU64,
    generated: AtomicU64,
    mtp_proposed: AtomicU64,
    jump_tokens: AtomicU64,
    jump_steps: AtomicU64,
    mtp_accepted: AtomicU64,
    spec_steps: AtomicU64,
    spec_proposed: AtomicU64,
    spec_accepted: AtomicU64,
    spec_emitted: AtomicU64,
    spec_draft_ns: AtomicU64,
    spec_verify_ns: AtomicU64,
    prefix_hits: AtomicU64,
    kv_used: AtomicU64,
    kv_total: AtomicU64,
    errors: [AtomicU64; 600],
    rate_buckets: [AtomicU64; 10],
    first_token: Histogram,
    duration: Histogram,
    queue_time: Histogram,
    e2e: Histogram,
    inter_token: Histogram,
    prompt_tokens: AtomicU64,
    finished: [AtomicU64; 3],
    model_name: OnceLock<String>,
    queue_wait: Histogram,
    preemptions: AtomicU64,
    timeouts: AtomicU64,
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
            jump_tokens: AtomicU64::new(0),
            jump_steps: AtomicU64::new(0),
            mtp_accepted: AtomicU64::new(0),
            spec_steps: AtomicU64::new(0),
            spec_proposed: AtomicU64::new(0),
            spec_accepted: AtomicU64::new(0),
            spec_emitted: AtomicU64::new(0),
            spec_draft_ns: AtomicU64::new(0),
            spec_verify_ns: AtomicU64::new(0),
            prefix_hits: AtomicU64::new(0),
            kv_used: AtomicU64::new(0),
            kv_total: AtomicU64::new(0),
            errors: std::array::from_fn(|_| AtomicU64::new(0)),
            rate_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            first_token: Histogram::new(),
            duration: Histogram::new(),
            queue_time: Histogram::new(),
            e2e: Histogram::new(),
            inter_token: Histogram::new(),
            prompt_tokens: AtomicU64::new(0),
            finished: std::array::from_fn(|_| AtomicU64::new(0)),
            model_name: OnceLock::new(),
            queue_wait: Histogram::new(),
            preemptions: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
        }
    }
    /// The `model_name` label of the `vllm:*` series.
    pub fn set_model_name(&self, name: &str) {
        let _ = self.model_name.set(name.to_owned());
    }
    /// Seconds a request waited between admission and the scheduler thread accepting it.
    pub fn queue_time(&self, seconds: f64) {
        self.queue_time.observe(seconds);
    }
    pub fn prompt_tokens(&self, n: usize) {
        self.prompt_tokens.fetch_add(n as u64, Relaxed);
    }
    fn kv_usage(&self) -> f64 {
        let total = self.kv_total.load(Relaxed);
        if total == 0 {
            0.0
        } else {
            self.kv_used.load(Relaxed) as f64 / total as f64
        }
    }
    /// The same measurements under the names and labels vLLM exports, for dashboards built for vLLM.
    pub fn render_vllm(&self) -> String {
        let model = self
            .model_name
            .get()
            .map_or("", String::as_str)
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        let labels = format!("model_name=\"{model}\"");
        let mut out = String::new();
        let gauges = [
            ("vllm:num_requests_running", "Requests currently executing", self.running.load(Relaxed) as f64),
            (
                "vllm:num_requests_waiting",
                "Requests waiting for a sequence slot",
                (self.waiting.load(Relaxed) + self.queued.load(Relaxed)) as f64,
            ),
            ("vllm:kv_cache_usage_perc", "KV cache usage; 1 means 100 percent", self.kv_usage()),
            ("vllm:gpu_cache_usage_perc", "Deprecated vLLM name of vllm:kv_cache_usage_perc", self.kv_usage()),
        ];
        for (name, help, value) in gauges {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n{name}{{{labels}}} {value}");
        }
        let counters = [
            ("vllm:prompt_tokens_total", "Number of prefill tokens processed", self.prompt_tokens.load(Relaxed)),
            ("vllm:generation_tokens_total", "Number of generation tokens processed", self.generated.load(Relaxed)),
            ("vllm:prefix_cache_hits_total", "Admissions that reused cached prefix blocks", self.prefix_hits.load(Relaxed)),
        ];
        for (name, help, value) in counters {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter\n{name}{{{labels}}} {value}");
        }
        out.push_str("# HELP vllm:request_success_total Count of successfully processed requests\n# TYPE vllm:request_success_total counter\n");
        for (reason, index) in FINISH_REASONS {
            let _ = writeln!(
                out,
                "vllm:request_success_total{{{labels},finished_reason=\"{reason}\"}} {}",
                self.finished[index].load(Relaxed)
            );
        }
        self.first_token.render("vllm:time_to_first_token_seconds", "Time to first token in seconds", &labels, &mut out);
        self.inter_token.render(
            "vllm:time_per_output_token_seconds",
            "Time per output token (inter-token latency) in seconds",
            &labels,
            &mut out,
        );
        self.inter_token.render("vllm:inter_token_latency_seconds", "Inter-token latency in seconds", &labels, &mut out);
        self.e2e.render("vllm:e2e_request_latency_seconds", "End-to-end request latency in seconds", &labels, &mut out);
        self.queue_time.render(
            "vllm:request_queue_time_seconds",
            "Time requests spend waiting before being scheduled",
            &labels,
            &mut out,
        );
        out
    }
    pub fn queue(&self) {
        self.queued.fetch_add(1, Relaxed);
    }
    pub fn dequeue(&self) {
        self.queued.fetch_sub(1, Relaxed);
    }
    /// One jump-forward step that appended `forced` grammar-forced tokens without sampling them.
    pub fn jump_forward(&self, forced: usize) {
        self.jump_tokens.fetch_add(forced as u64, Relaxed);
        self.jump_steps.fetch_add(1, Relaxed);
    }
    pub fn mtp_proposed(&self, n: usize) {
        self.mtp_proposed.fetch_add(n as u64, Relaxed);
    }
    /// Add the counters of speculative verification steps (all methods together).
    pub fn speculative(&self, delta: &crate::speculative::SpecStats) {
        self.spec_steps.fetch_add(delta.steps, Relaxed);
        self.spec_proposed.fetch_add(delta.proposed, Relaxed);
        self.spec_accepted.fetch_add(delta.accepted, Relaxed);
        self.spec_emitted.fetch_add(delta.emitted, Relaxed);
        self.spec_draft_ns.fetch_add(delta.draft_ns, Relaxed);
        self.spec_verify_ns.fetch_add(delta.verify_ns, Relaxed);
    }
    pub fn speculative_totals(&self) -> crate::speculative::SpecStats {
        crate::speculative::SpecStats {
            steps: self.spec_steps.load(Relaxed),
            proposed: self.spec_proposed.load(Relaxed),
            accepted: self.spec_accepted.load(Relaxed),
            emitted: self.spec_emitted.load(Relaxed),
            draft_ns: self.spec_draft_ns.load(Relaxed),
            verify_ns: self.spec_verify_ns.load(Relaxed),
        }
    }
    /// Seconds a sequence waited between submission and admission to the running set.
    pub fn queue_wait(&self, seconds: f64) {
        self.queue_wait.observe(seconds);
    }
    pub fn set_preemptions(&self, total: u64) {
        self.preemptions.store(total, Relaxed);
    }
    pub fn timeout(&self) {
        self.timeouts.fetch_add(1, Relaxed);
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
            last: None,
            finished: false,
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
        let _ = writeln!(out, "# HELP fllm_jump_forward_tokens_total Tokens forced by a grammar and appended without a forward pass of their own (counted in completion tokens)
# TYPE fllm_jump_forward_tokens_total counter
fllm_jump_forward_tokens_total {}
# HELP fllm_jump_forward_steps_total Jump-forward steps (one forward pass each covers all tokens forced at that point)
# TYPE fllm_jump_forward_steps_total counter
fllm_jump_forward_steps_total {}", self.jump_tokens.load(Relaxed), self.jump_steps.load(Relaxed));
        let proposed = self.mtp_proposed.load(Relaxed);
        let accepted = self.mtp_accepted.load(Relaxed);
        let _ = writeln!(out, "# HELP fllm_mtp_proposed_tokens_total MTP candidates actually scheduled for target verification\n# TYPE fllm_mtp_proposed_tokens_total counter\nfllm_mtp_proposed_tokens_total {proposed}\n# HELP fllm_mtp_accepted_tokens_total MTP candidates accepted by the target\n# TYPE fllm_mtp_accepted_tokens_total counter\nfllm_mtp_accepted_tokens_total {accepted}\n# HELP fllm_mtp_acceptance_ratio Accepted MTP candidates divided by proposed candidates\n# TYPE fllm_mtp_acceptance_ratio gauge\nfllm_mtp_acceptance_ratio {:.6}", if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 });
        self.queue_wait.render("fllm_queue_wait_seconds", "Time a sequence waited for admission to the running set", "", &mut out);
        let _ = writeln!(
            out,
            "# HELP fllm_preemptions_total Sequences preempted to free KV blocks or slots\n# TYPE fllm_preemptions_total counter\nfllm_preemptions_total {}\n\
             # HELP fllm_request_timeouts_total Requests aborted because their timeout passed\n# TYPE fllm_request_timeouts_total counter\nfllm_request_timeouts_total {}",
            self.preemptions.load(Relaxed),
            self.timeouts.load(Relaxed)
        );
        let spec = self.speculative_totals();
        let _ = writeln!(
            out,
            "# HELP fllm_spec_steps_total Target passes that verified speculative draft tokens
# TYPE fllm_spec_steps_total counter
fllm_spec_steps_total {}
             # HELP fllm_spec_proposed_tokens_total Draft tokens sent to verification (all speculation methods)
# TYPE fllm_spec_proposed_tokens_total counter
fllm_spec_proposed_tokens_total {}
             # HELP fllm_spec_accepted_tokens_total Draft tokens accepted by the target
# TYPE fllm_spec_accepted_tokens_total counter
fllm_spec_accepted_tokens_total {}
             # HELP fllm_spec_acceptance_ratio Accepted divided by proposed draft tokens
# TYPE fllm_spec_acceptance_ratio gauge
fllm_spec_acceptance_ratio {:.6}
             # HELP fllm_spec_mean_accepted_length Mean accepted draft tokens per verification step
# TYPE fllm_spec_mean_accepted_length gauge
fllm_spec_mean_accepted_length {:.6}
             # HELP fllm_spec_mean_tokens_per_step Mean tokens produced per verification step (1 = no gain)
# TYPE fllm_spec_mean_tokens_per_step gauge
fllm_spec_mean_tokens_per_step {:.6}
             # HELP fllm_spec_speedup_estimate Estimated decode speedup of speculation over plain decoding
# TYPE fllm_spec_speedup_estimate gauge
fllm_spec_speedup_estimate {:.6}",
            spec.steps,
            spec.proposed,
            spec.accepted,
            spec.acceptance_rate(),
            spec.mean_accepted_length(),
            spec.mean_emitted(),
            spec.speedup_estimate()
        );
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
            "",
            &mut out,
        );
        self.duration.render(
            "fllm_request_duration_seconds",
            "API latency through body completion or client cancellation",
            "",
            &mut out,
        );
        self.queue_time.render(
            "fllm_request_queue_seconds",
            "Time from admission until the scheduler thread accepted the request",
            "",
            &mut out,
        );
        self.e2e.render(
            "fllm_request_e2e_seconds",
            "Time from admission to the last token of requests that finished",
            "",
            &mut out,
        );
        self.inter_token.render(
            "fllm_inter_token_seconds",
            "Time between two consecutive generated tokens of a request",
            "",
            &mut out,
        );
        let _ = writeln!(
            out,
            "# HELP fllm_prompt_tokens_total Prompt tokens of admitted requests\n# TYPE fllm_prompt_tokens_total counter\nfllm_prompt_tokens_total {}\n# HELP fllm_kv_cache_usage_ratio Share of the KV block pool in use\n# TYPE fllm_kv_cache_usage_ratio gauge\nfllm_kv_cache_usage_ratio {:.6}",
            self.prompt_tokens.load(Relaxed),
            self.kv_usage()
        );
        out.push_str("# HELP fllm_requests_finished_total Requests by how they ended\n# TYPE fllm_requests_finished_total counter\n");
        for (reason, index) in FINISH_REASONS {
            let _ = writeln!(
                out,
                "fllm_requests_finished_total{{reason=\"{reason}\"}} {}",
                self.finished[index].load(Relaxed)
            );
        }
        crate::expert_cache::render_metrics(&mut out);
        out
    }
}

pub struct Generation {
    metrics: Arc<Metrics>,
    started: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    finished: bool,
    tokens: usize,
}
impl Generation {
    pub fn token(&mut self) {
        let now = Instant::now();
        match self.last {
            None => {
                self.metrics
                    .first_token
                    .observe(self.started.elapsed().as_secs_f64());
                self.first = Some(now);
            }
            Some(previous) => self
                .metrics
                .inter_token
                .observe(now.duration_since(previous).as_secs_f64()),
        }
        self.last = Some(now);
        self.tokens += 1;
    }
    /// The request ended on its own (`stop` or `length`); a request that is dropped before this counts as aborted.
    pub fn finish(&mut self, reason: &str) {
        if !self.finished {
            self.finished = true;
            let slot = if reason == "length" { 1 } else { 0 };
            self.metrics.finished[slot].fetch_add(1, Relaxed);
            self.metrics.e2e.observe(self.started.elapsed().as_secs_f64());
        }
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
impl Drop for Generation {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.finished[2].fetch_add(1, Relaxed);
        }
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
