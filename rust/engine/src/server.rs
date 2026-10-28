use crate::reasoning::{Family, ReasoningParser, ReasoningParts};
use crate::runtime::InferenceRuntime;
use crate::tokenizer::TokenizerBundle;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, head, post},
    Json, Router,
};
mod api;
mod cache_gate;
pub(crate) mod errors;
pub mod gencfg;
pub(crate) mod http;
pub mod logging;
pub mod model_facts;
pub mod options;
mod remote_image;
mod setup;
pub use setup::serve_setup;
pub(crate) mod security;
mod tls;
#[cfg(test)]
mod tests;

pub use model_facts::ModelFacts;
use options::HttpOptions;
mod media;

use minijinja::Environment;
use serde_json::{json, Map, Value};
use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc as std_mpsc, Arc,
    },
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc as tokio_mpsc;

struct Metrics {
    observability: Arc<crate::metrics::Metrics>,
    started: Instant,
    /// Milliseconds since `started` at which the worker loop last made progress (it also wakes once a second when idle).
    heartbeat_ms: AtomicU64,
    worker_panics: AtomicU64,
    queue_rejections: AtomicU64,
    /// How long the worker may go without progress before `/ready` reports 503 (`WORKER_STALL_MS`).
    stall_ms: AtomicU64,
    /// Test hook: the next worker step panics (see `tests.rs`).
    #[cfg(test)]
    inject_panic: AtomicU64,
    /// Test hook: every worker step sleeps this many milliseconds first.
    #[cfg(test)]
    inject_delay_ms: AtomicU64,
    requests: AtomicU64,
    generated_tokens: AtomicU64,
    forward_calls: AtomicU64,
    forced_tokens: AtomicU64,
    jumps: AtomicU64,
    rows_total: AtomicU64,
    multi_row_calls: AtomicU64,
    max_rows: AtomicU64,
    prefill_rows: AtomicU64,
    decode_rows: AtomicU64,
    draft_proposed: AtomicU64,
    draft_accepted: AtomicU64,
    scheduler_ns: AtomicU64,
    batch_build_ns: AtomicU64,
    model_forward_ns: AtomicU64,
    slots_running: AtomicU64,
    slots_waiting: AtomicU64,
    postprocess_ns: AtomicU64,
}

struct AppState {
    model_name: String,
    model_arch: String,
    chat_template: String,
    /// `tool_use` template, used instead of `chat_template` when a request carries tools (empty: none)
    tool_chat_template: String,
    /// Tool-call format of the model (--tool-parser, else chosen from the chat template).
    tool_parser: Arc<dyn crate::tool_parsers::ToolParser>,
    /// Reasoning format of the model (--reasoning-parser, else chosen from the chat template).
    reasoning_family: Family,
    /// What the loaded model can do about thinking (`capabilities.thinking` of the status route).
    thinking: crate::thinking::ThinkingCaps,
    tokenizer: Arc<TokenizerBundle>,
    grammar_factory: Arc<llguidance::ParserFactory>,
    worker: std_mpsc::Sender<WorkerCommand>,
    next_request: AtomicU64,
    metrics: Arc<Metrics>,
    /// Authentication, rate limits and the other HTTP-layer settings.
    http: Arc<http::HttpContext>,
    /// Further names a request may put in its `model` field.
    aliases: std::sync::RwLock<Vec<String>>,
    facts: ModelFacts,
    /// `false` while the server is starting or shutting down (`/ready`).
    ready: Arc<AtomicBool>,
    /// Last Ollama `keep_alive` the clients asked for. HOOK(multi-model): unloading idle models is not implemented
    /// here, the single resident model only reports the expiry in `/api/ps`.
    keep_alive: std::sync::Mutex<Option<(SystemTime, api::ollama_extra::KeepAlive)>>,
    max_context: usize,
    max_sequences: usize,
    shared_choices: bool,
    hidden_size: usize,
    admission: Arc<tokio::sync::Semaphore>,
    studio_config: std::sync::RwLock<api::studio::StudioRuntimeConfig>,
    agent: api::agent::AgentSession,
    vision: Option<Arc<dyn crate::vision::MultimodalEncoder>>,
    vision_admission: Arc<tokio::sync::Semaphore>,
    /// Answers that were computed before, handed out again for identical deterministic requests.
    cache: Arc<crate::cache::response_cache::ResponseCache>,
    /// Model and template identity: part of every cache key, so a model change invalidates old answers.
    cache_identity: String,
    /// Set by `POST /admin/drain` before a planned restart: new work is refused, running work finishes.
    draining: Arc<AtomicBool>,
    /// OTLP trace exporter (off unless `FLLM_OTEL_ENDPOINT` is set).
    tracer: Arc<crate::telemetry::Tracer>,
    /// Identical requests that are in flight at the same time share one computation.
    coalesce: Arc<crate::cache::coalescer::Coalescer>,
}

enum WorkerCommand {
    Submit {
        request_id: String,
        prompt: Vec<i64>,
        max_tokens: usize,
        /// `cache_salt` of the request: 0 when it does not ask for isolation.
        prefix_salt: u64,
        sampling: Box<crate::sampling::SamplingParams>,
        constraint: Option<Box<llguidance::Matcher>>,
        images: Box<ImageOverrides>,
        /// Set for a beam search (`use_beam_search`): the request is run by `submit_beam` instead of the sampler.
        beam: Option<Box<crate::beam::BeamParams>>,
        output: GenerationTx,
    },
    Embed {
        tokens: Vec<i64>,
        pooling: String,
        output: tokio::sync::oneshot::Sender<Result<Vec<f32>, String>>,
    },
    /// `GET /admin/prefix-cache`: what the prefix cache holds right now.
    Stats {
        output: tokio::sync::oneshot::Sender<Value>,
    },
    /// `DELETE /admin/prefix-cache`: empty it and give the blocks back.
    ClearPrefixCache {
        output: tokio::sync::oneshot::Sender<Result<usize, String>>,
    },
    /// `GET /admin/kv-tier`: the disk tier's counters.
    KvTierStats {
        output: tokio::sync::oneshot::Sender<Value>,
    },
    /// `DELETE /admin/kv-tier`: drop every stored block.
    ClearKvTier {
        output: tokio::sync::oneshot::Sender<usize>,
    },
}

/// `cache_salt` of a request: a string or a number, folded into 64 bits. 0 means "no isolation".
fn cache_salt_of(body: &Value) -> u64 {
    let Some(value) = body.get("cache_salt") else {
        return 0;
    };
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        _ => return 0,
    };
    if text.is_empty() {
        return 0;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}


#[derive(Clone)]
struct GenerationTx {
    inner: tokio_mpsc::Sender<Result<crate::runtime::TokenEvent, String>>,
    error: Arc<std::sync::Mutex<Option<String>>>,
    /// When the request was handed to the worker, for the queue-time histogram.
    submitted: Instant,
}
impl GenerationTx {
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    /// Events that fit into the channel right now.
    fn room(&self) -> usize {
        self.inner.capacity()
    }
    fn send(&self, event: Result<crate::runtime::TokenEvent, String>) -> Result<(), ()> {
        self.inner.try_send(event).map_err(|error| {
            if matches!(error, tokio_mpsc::error::TrySendError::Full(_)) {
                *security::lock(&self.error) =
                    Some("client output buffer is full; generation cancelled".into());
            }
        })
    }
}
struct GenerationRx {
    observation: crate::metrics::Generation,
    inner: tokio_mpsc::Receiver<Result<crate::runtime::TokenEvent, String>>,
    error: Arc<std::sync::Mutex<Option<String>>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
    /// Token budget and counters of the calling API key / HTTP request.
    caller: Option<security::Caller>,
}
impl GenerationRx {
    async fn recv(&mut self) -> Option<Result<crate::runtime::TokenEvent, String>> {
        match self.inner.recv().await {
            Some(event) => {
                if let Ok(token) = &event {
                    self.observation.token();
                    if let Some(caller) = &self.caller {
                        caller.stats.completion_tokens.fetch_add(1, Ordering::Relaxed);
                        if let Some(limiter) = &caller.limiter {
                            limiter.charge_tokens(1);
                        }
                    }
                    if token.request_finished {
                        self.observation.finish(token.finish_reason.unwrap_or("stop"));
                    }
                }
                Some(event)
            }
            None => security::lock(&self.error).take().map(Err),
        }
    }
}

/// Budget forcing for `thinking_budget`: the state of a request whose reasoning block is cut after a number of tokens.
struct BudgetControl {
    state: Arc<AppState>,
    request_id: String,
    prompt: Vec<i64>,
    sampling: crate::sampling::SamplingParams,
    max_tokens: usize,
    spec: ReasoningSpec,
    machine: crate::thinking::BudgetMachine,
    /// Every token produced so far, the forced ones included.
    ids: Vec<u32>,
}

impl BudgetControl {
    /// Looks at one more generated token. When the reasoning block is still open at the budget, returns the tokens that end it
    /// (the closing tag and a line break); the caller continues the generation from there.
    fn observe(&mut self, event: &crate::runtime::TokenEvent) -> Option<Vec<u32>> {
        use crate::thinking::{Probe, Step};
        self.ids.push(event.token_id as u32);
        if event.finished {
            self.machine.finish();
            return None;
        }
        let (state, spec, ids) = (&self.state, &self.spec, &self.ids);
        let mut close = None;
        let step = self.machine.on_token(|| {
            let Ok(text) = decode_chat_text(&state.tokenizer, ids, Some(spec)) else {
                return Probe::Absent;
            };
            let probe = spec.parser.probe(&text);
            if probe == Probe::Open {
                close = spec.parser.close_tag(&text);
            }
            probe
        });
        if step != Step::Force {
            return None;
        }
        let tail = state.tokenizer.encode_without_bos(&format!("
{}

", close?)).ok()?;
        // no room left for an answer: the length limit ends the reply as it would without a budget
        if tail.is_empty() || self.max_tokens <= self.ids.len() + tail.len() {
            return None;
        }
        self.ids.extend(&tail);
        Some(tail)
    }

    /// The second part of the request: the prompt, the reasoning so far and the closing tag, then the answer.
    fn continuation(&self) -> Result<GenerationRx, String> {
        let mut prompt = self.prompt.clone();
        prompt.extend(self.ids.iter().map(|id| i64::from(*id)));
        let mut sampling = self.sampling.clone();
        sampling.min_tokens = sampling.min_tokens.saturating_sub(self.ids.len());
        submit_constrained(
            &self.state,
            &format!("{}.answer", self.request_id),
            prompt,
            self.max_tokens - self.ids.len(),
            sampling,
            None,
            ImageOverrides::default(),
            0,
        )
        .map_err(|_| "the server could not continue the reply after the thinking budget (it is busy)".to_owned())
    }
}

/// The token stream of a chat request. With a thinking budget it ends the reasoning block itself: when the budget's tokens are
/// used up and the block is open, the closing tag is delivered as if the model had written it, the first generation is
/// cancelled and a second one continues from prompt + reasoning + tag. The caller sees one uninterrupted stream.
struct ReasoningRx {
    rx: Option<GenerationRx>,
    control: Option<BudgetControl>,
    queue: std::collections::VecDeque<crate::runtime::TokenEvent>,
    /// The reasoning block was cut: the tokens in `queue` come first, then the continuation starts.
    resume: bool,
    resumed: bool,
    prompt_tokens: usize,
}

impl ReasoningRx {
    fn new(rx: GenerationRx, control: Option<BudgetControl>) -> Self {
        let prompt_tokens = control.as_ref().map_or(0, |c| c.prompt.len());
        Self { rx: Some(rx), control, queue: Default::default(), resume: false, resumed: false, prompt_tokens }
    }

    fn timings(&self, prompt_tokens: usize) -> Value {
        self.rx.as_ref().map_or(Value::Null, |rx| rx.observation.timings(prompt_tokens))
    }

    async fn recv(&mut self) -> Option<Result<crate::runtime::TokenEvent, String>> {
        if let Some(event) = self.queue.pop_front() {
            return Some(Ok(event));
        }
        if self.resume {
            self.resume = false;
            // the first part must release its slot before the second one asks for one
            self.rx = None;
            match self.control.as_ref()?.continuation() {
                Ok(rx) => {
                    self.rx = Some(rx);
                    self.resumed = true;
                }
                Err(error) => return Some(Err(error)),
            }
        }
        let item = self.rx.as_mut()?.recv().await?;
        let mut event = match item {
            Ok(event) => event,
            Err(error) => return Some(Err(error)),
        };
        if self.resumed {
            // the continuation reports the cache hits of the longer prompt
            event.cached_prompt_tokens = event.cached_prompt_tokens.min(self.prompt_tokens);
        } else if let Some(control) = self.control.as_mut() {
            if let Some(tail) = control.observe(&event) {
                for id in tail {
                    self.queue.push_back(crate::runtime::TokenEvent {
                        token_id: i64::from(id),
                        finished: false,
                        finish_reason: None,
                        logprobs: None,
                        request_finished: false,
                        ..event.clone()
                    });
                }
                self.resume = true;
            }
        }
        Some(Ok(event))
    }
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_name: String,
    /// Authentication, limits, timeouts, CORS, TLS, logging (see `options.rs`).
    pub http: HttpOptions,
    pub facts: ModelFacts,
    /// Sampling defaults used when a request does not set the parameter (`--temp`, `--top-p`, `--top-k`).
    pub default_temperature: Option<f64>,
    pub default_top_p: Option<f64>,
    pub default_top_k: Option<usize>,
    pub sampling_defaults: crate::sampling::SamplingParams,
    /// --tool-parser: a parser id, or auto/absent to choose from the chat template.
    pub tool_parser: Option<String>,
    /// --reasoning-parser: a reasoning format id, or auto/absent to choose from the chat template.
    pub reasoning_parser: Option<String>,
    /// Jump-forward decoding of grammar-constrained requests (`--no-jump-forward`, `FLLM_JUMP_FORWARD=off`).
    pub jump_forward: bool,
}

/// Temperature of an OpenAI-style request that sets none and whose model ships no `generation_config.json` sampling
/// values: the OpenAI and vLLM default (the engine used to answer greedily).
const OPENAI_DEFAULT_TEMPERATURE: f64 = 1.0;

/// Ollama built-in sampling defaults, used on `/api/chat` and `/api/generate` when neither the request, the command
/// line nor the model sets a value.
const OLLAMA_DEFAULT_TEMPERATURE: f64 = 0.8;
const OLLAMA_DEFAULT_TOP_P: f64 = 0.9;
const OLLAMA_DEFAULT_TOP_K: usize = 40;

/// What a worker that has not made progress for this long counts as: stuck. `/ready` answers 503 then.
const WORKER_STALL_MS: u64 = 120_000;

/// A router and worker channel for `state`, without any socket (the unit tests drive them directly).
struct Built {
    state: Arc<AppState>,
    router: Router,
    worker_rx: std_mpsc::Receiver<WorkerCommand>,
    metrics: Arc<Metrics>,
    jump_forward: Option<Arc<crate::structured::JumpForward>>,
}

fn build(runtime: &InferenceRuntime, tokenizer: TokenizerBundle, config: ServerConfig) -> Result<Built, String> {
    validate_chat_template(runtime.model.chat_template())?;
    let chat_template = normalize_chat_template(runtime.model.chat_template());
    let tool_chat_template = normalize_chat_template(runtime.model.tool_chat_template());
    let max_context = runtime.model.max_context();
    let model_arch = runtime.model.model_arch().to_owned();
    let hidden_size = runtime.model.hidden_size();
    let max_sequences = runtime.model.max_sequences();
    let queue = config.http.max_queue.unwrap_or(8 * max_sequences);
    let server_metrics = Arc::new(Metrics {
        observability: Arc::new(crate::metrics::Metrics::new()),
        started: Instant::now(),
        heartbeat_ms: AtomicU64::new(0),
        worker_panics: AtomicU64::new(0),
        queue_rejections: AtomicU64::new(0),
        stall_ms: AtomicU64::new(WORKER_STALL_MS),
        #[cfg(test)]
        inject_panic: AtomicU64::new(0),
        #[cfg(test)]
        inject_delay_ms: AtomicU64::new(0),
        requests: AtomicU64::new(0),
        generated_tokens: AtomicU64::new(0),
        forward_calls: AtomicU64::new(0),
        forced_tokens: AtomicU64::new(0),
        jumps: AtomicU64::new(0),
        rows_total: AtomicU64::new(0),
        multi_row_calls: AtomicU64::new(0),
        max_rows: AtomicU64::new(0),
        prefill_rows: AtomicU64::new(0),
        decode_rows: AtomicU64::new(0),
        draft_proposed: AtomicU64::new(0),
        draft_accepted: AtomicU64::new(0),
        scheduler_ns: AtomicU64::new(0),
        batch_build_ns: AtomicU64::new(0),
        model_forward_ns: AtomicU64::new(0),
        postprocess_ns: AtomicU64::new(0),
        slots_running: AtomicU64::new(0),
        slots_waiting: AtomicU64::new(0),
    });
    server_metrics.observability.set_model_name(&config.model_name);
    let (worker_tx, worker_rx) = std_mpsc::channel::<WorkerCommand>();
    let grammar_factory = Arc::new(tokenizer.grammar_factory()?);
    let tokenizer = Arc::new(tokenizer);
    let jump_forward = config.jump_forward.then(|| {
        let encoder = tokenizer.clone();
        Arc::new(crate::structured::JumpForward::new(
            grammar_factory.tok_env().clone(),
            Some(Box::new(move |text: &str| encoder.encode_plain(text).ok())),
        ))
    });
    let tool_parser = crate::tool_parsers::select(&model_arch, &format!("{chat_template}\n{tool_chat_template}"), config.tool_parser.as_deref())?;
    let reasoning_family = Family::select(&chat_template, config.reasoning_parser.as_deref())?;
    let thinking = crate::thinking::ThinkingCaps::detect(&chat_template, reasoning_family, &model_arch);
    tracing::info!("tool parser: {}, reasoning parser: {:?}, thinking: {}", tool_parser.id(), reasoning_family, thinking.control.name());
    let cache_identity = cache_gate::identity(&config.model_name, &model_arch, &chat_template, &tool_chat_template);
    let cache = cache_gate::from_env();
    // tenant-scoped answers with a shared prefix cache would still share KV blocks: say it loudly instead of pretending
    if matches!(cache.scope(), crate::cache::response_cache::Scope::Tenant)
        && !matches!(std::env::var("FLLM_PREFIX_CACHE").unwrap_or_default().trim().to_ascii_lowercase().as_str(), "off" | "0" | "no" | "false")
    {
        tracing::warn!(
            "FLLM_CACHE_SCOPE=tenant isolates cached answers, but the prefix cache still shares KV blocks between requests; set FLLM_PREFIX_CACHE=off (or run one server per tenant) for real isolation"
        );
    }
    let cache_stats = cache.stats();
    tracing::info!(
        "output cache: {} (scope {}, {} MiB, ttl {} s)",
        if cache.enabled() { "on" } else { "off" },
        cache.scope().name(),
        cache_stats["max_bytes"].as_u64().unwrap_or(0) / (1 << 20),
        cache_stats["ttl_seconds"].as_u64().unwrap_or(0)
    );
    let http_context = http::HttpContext::new(config.http.clone());
    let ready = Arc::new(AtomicBool::new(false));
    let state = Arc::new(AppState {
        model_name: config.model_name,
        model_arch,
        chat_template,
        tool_chat_template,
        tool_parser,
        reasoning_family,
        thinking,
        tokenizer,
        grammar_factory,
        worker: worker_tx,
        next_request: AtomicU64::new(1),
        metrics: server_metrics.clone(),
        http: http_context.clone(),
        aliases: std::sync::RwLock::new(config.http.model_aliases.clone()),
        facts: config.facts,
        ready,
        keep_alive: std::sync::Mutex::new(None),
        max_context,
        max_sequences,
        shared_choices: runtime.supports_shared_choices(),
        hidden_size,
        // requests running plus requests waiting for a sequence slot; a request beyond that gets 429 with Retry-After
        admission: Arc::new(tokio::sync::Semaphore::new(max_sequences + queue)),
        agent: api::agent::AgentSession::default(),
        studio_config: std::sync::RwLock::new(api::studio::StudioRuntimeConfig {
            sampling: config.sampling_defaults,
            override_temperature: config.default_temperature,
            override_top_p: config.default_top_p,
            override_top_k: config.default_top_k,
            system_prompt: String::new(),
            cpu_tier_cap: None,
        }),
        vision: runtime.vision.clone(),
        vision_admission: Arc::new(tokio::sync::Semaphore::new(crate::vision::preprocess::MediaLimits::get().concurrency)),
        cache,
        cache_identity,
        draining: Arc::new(AtomicBool::new(false)),
        tracer: Arc::new(crate::telemetry::Tracer::from_env()),
        coalesce: crate::cache::coalescer::Coalescer::new(),
    });
    let upload_limit = api::batches::max_upload_bytes();
    let router = Router::new()
        .route("/", get(webui).head(api::ollama_extra::head_root))
        .route("/studio/runner.js", get(|| async { crate::webui::runner() }))
        .route("/health", get(health))
        .route("/ready", get(ready_check))
        .route("/metrics", get(metrics))
        .route("/admin/cache", get(admin_cache).delete(admin_cache_clear))
        .route("/admin/drain", post(admin_drain).delete(admin_undrain))
        .route("/admin/telemetry", get(admin_telemetry))
        .route("/admin/config", get(admin_config))
        .route("/admin/prefix-cache", get(admin_prefix_cache).delete(admin_prefix_cache_clear))
        .route("/admin/kv-tier", get(admin_kv_tier).delete(admin_kv_tier_clear))
        .route("/version", get(api::vllm::version))
        .route("/is_sleeping", get(api::vllm::is_sleeping))
        .route("/api/studio/status", get(api::studio::status))
        .route("/api/studio/config", post(api::studio::update_config))
        .route("/api/studio/hardware", get(api::studio::hardware))
        .route("/api/studio/npu", get(api::npu::state))
        .route("/api/studio/npu/setup", post(api::npu::start_setup).delete(api::npu::remove_runtime))
        .route("/api/studio/hf-account", get(api::studio::hf_account).post(api::studio::hf_login))
        .route("/api/studio/open-url", post(api::studio::open_url))
        .route("/api/studio/extract", post(api::extract::extract).layer(DefaultBodyLimit::max(api::extract::MAX_UPLOAD_BYTES)))
        .route("/api/studio/update", get(api::studio::update_status))
        .route("/api/studio/update/install", post(api::studio::update_install))
        .route("/api/studio/selftest-cpu", post(api::studio::selftest_cpu))
        .route("/api/studio/models", get(api::studio::models_hub))
        .route("/api/studio/hf-files", get(api::studio::hf_files))
        .route("/api/studio/models/pull", post(api::studio::pull_model))
        .route("/api/studio/models/load", post(api::studio::load_model))
        .route("/api/studio/models/search", get(api::studio::search_models))
        .route("/api/studio/models/progress", get(api::studio::model_progress))
        .route("/api/studio/models/cache", delete(api::studio::delete_cached_model))
        .route("/api/studio/models/local", post(api::studio::add_local_model).delete(api::studio::remove_local_model))
        .route("/api/studio/models/unload", post(api::studio_ops::unload_model))
        .route("/api/studio/models/downloads", post(api::studio_ops::download_action))
        .route("/api/studio/pick-file", post(api::studio_ops::pick_file))
        .route("/api/studio/settings", get(api::studio_ops::get_settings).post(api::studio_ops::update_settings))
        .route("/api/studio/engine", get(api::studio_ops::get_engine).post(api::studio_ops::update_engine))
        .route("/api/studio/restart", post(api::studio_ops::restart_server))
        .route("/api/studio/logs", get(api::studio_ops::logs))
        .route("/api/studio/hf-readme", get(api::studio_ops::hf_readme))
        .route("/api/studio/keys", get(api::studio::list_keys).post(api::studio::create_key))
        .route("/api/studio/keys/{id}", delete(api::studio::delete_key))
        .route("/api/studio/mcp/servers", get(api::mcp::servers).post(api::mcp::upsert).delete(api::mcp::remove))
        .route("/api/studio/mcp/registry", get(api::mcp_registry::search))
        .route("/api/studio/mcp/restart", post(api::mcp::restart))
        .route("/api/studio/mcp/tools", get(api::mcp::tools))
        .route("/api/studio/mcp/call", post(api::mcp::call))
        .route("/api/studio/mcp/resources", post(api::mcp::resources))
        .route("/api/studio/mcp/resource", post(api::mcp::resource))
        .route("/api/studio/providers", get(api::providers::list).post(api::providers::upsert).delete(api::providers::remove))
        .route("/api/studio/providers/test", post(api::providers::test_connection))
        .route("/api/studio/providers/models", get(api::providers::models))
        .route("/api/studio/providers/chat", post(api::providers::chat).layer(DefaultBodyLimit::max(api::providers::MAX_CHAT_BODY)))
        .route("/api/studio/agent/workspace", post(api::agent::workspace))
        .route("/api/studio/agent/tool", post(api::agent::tool))
        .route("/api/studio/agent/undo", post(api::agent::undo))
        .route("/api/studio/agent/tools", get(api::agent::tools))
        .route("/api/studio/agent/changes", get(api::agent::changes))
        .route("/api/studio/agent/end", post(api::agent::end))
        .route("/api/studio/memory", get(api::memory::get_memory).put(api::memory::put_memory).layer(DefaultBodyLimit::max(api::memory::MAX_BODY)))
        .route("/api/studio/agent/cancel", post(api::agent::cancel))
        .route("/api/studio/agent/files", post(api::agent::files))
        .route("/api/studio/agent/memory", post(api::agent_mem::memory))
        .route("/api/studio/agent/instructions", post(api::agent_mem::instructions))
        .route("/api/studio/hooks", get(api::hooks::list).post(api::hooks::save_handler))
        .route("/api/studio/hooks/test", post(api::hooks::test))
        .route("/api/studio/agents", get(api::agents::list).post(api::agents::write).delete(api::agents::remove_route))
        .route("/api/studio/prices", get(api::prices::list).post(api::prices::save_handler))
        .route("/api/studio/agent/sandbox", get(api::agent_sandbox::status))
        .route("/api/studio/agent/worktree", post(api::agent_worktree::route))
        .route("/api/studio/embeddings", get(api::studio_embed::route_status).post(api::studio_embed::route_save))
        .route("/api/studio/embed", post(api::studio_embed::route_embed))
        .route("/api/studio/runtime", get(api::studio_mode::route_status).post(api::studio_mode::route_save))
        .route("/api/studio/skills/library", get(api::skills::library).post(api::skills::install_builtin))
        .route("/api/studio/skills/import", post(api::skills::import))
        .route("/api/studio/skills/github", get(api::skills::github_list))
        .route("/api/studio/skills/directory", get(api::skills::directory_list))
        .route("/api/studio/skills/directory/install", post(api::skills::directory_install))
        .route("/api/studio/skills/directory/token", post(api::skills::directory_token_save))
        .route(
            "/api/studio/agent/messages",
            get(api::agent_messages::route_list).post(api::agent_messages::route_add).delete(api::agent_messages::route_dismiss),
        )
        .route("/api/studio/plugins", get(api::plugins::list).post(api::plugins::install_route).delete(api::plugins::remove_route))
        .route("/api/studio/plugins/commands", get(api::plugins::commands_route))
        .route("/api/studio/plugins/command", get(api::plugins::command_route))
        .route("/api/studio/web/fetch", post(api::agent_net::fetch_route))
        .route("/api/studio/web/search", post(api::websearch::search_route))
        .route("/api/studio/search", post(api::websearch::search_route))
        .route("/api/studio/search/engines", get(api::websearch::engines_route).post(api::websearch::engines_save))
        .route("/api/studio/skills", get(api::skills::list).post(api::skills::write).delete(api::skills::remove))
        .route("/api/studio/skills/read", get(api::skills::read))
        .route("/api/studio/sessions/index", post(api::sessions_index::index).delete(api::sessions_index::remove))
        .route("/api/studio/sessions/search", get(api::sessions_index::search))
        .route("/api/studio/schedule", get(api::schedule::list).post(api::schedule::upsert).delete(api::schedule::remove))
        .route("/api/studio/schedule/run", post(api::schedule::run_now))
        .route("/api/studio/schedule/results", get(api::schedule::results))
        .route("/api/studio/schedule/check", get(api::schedule::check))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(api::model_by_id))
        .route("/v1/messages", post(api::anthropic::messages))
        .route("/v1/messages/count_tokens", post(api::anthropic::count_tokens))
        .route("/v1/responses", post(api::responses::create))
        .route("/v1/responses/{id}", get(api::responses::get_one).delete(api::responses::delete_one))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/render", post(api::vllm::render_chat))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        .route("/v1/score", post(score))
        .route("/classify", post(classify))
        .route("/pooling", post(pooling))
        .route(
            "/v1/files",
            get(api::batches::list_files)
                .post(api::batches::upload)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        .route("/v1/files/{id}", get(api::batches::get_file).delete(api::batches::delete_file))
        .route("/v1/files/{id}/content", get(api::batches::file_content))
        .route("/v1/batches", post(api::batches::create_batch).get(api::batches::list_batches))
        .route("/v1/batches/{id}", get(api::batches::get_batch))
        .route("/v1/batches/{id}/cancel", post(api::batches::cancel_batch))
        .route("/v1/images/embeddings", post(image_embeddings))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcriptions).layer(DefaultBodyLimit::max(25 * 1024 * 1024)),
        )
        .route(
            "/v1/audio/translations",
            post(audio_translations).layer(DefaultBodyLimit::max(25 * 1024 * 1024)),
        )
        .route("/api/embed", post(api::ollama_extra::embed))
        .route("/api/embeddings", post(embeddings))
        .route("/api/show", post(api::ollama_extra::show))
        .route("/api/ps", get(api::ollama_extra::ps))
        .route("/api/pull", post(api::ollama_extra::pull))
        .route("/api/push", post(api::ollama_extra::push))
        .route("/api/create", post(api::ollama_extra::create))
        .route("/api/copy", post(api::ollama_extra::copy))
        .route("/api/delete", delete(api::ollama_extra::delete_model))
        .route("/api/blobs/{digest}", head(api::ollama_extra::head_blob).post(api::ollama_extra::post_blob))
        .route("/tokenize", post(api::llamacpp::tokenize))
        .route("/detokenize", post(api::llamacpp::detokenize))
        .route("/completion", post(api::llamacpp::completion))
        .route("/infill", post(api::llamacpp::infill))
        .route("/props", get(api::llamacpp::props).post(api::llamacpp::set_props))
        .route("/slots", get(api::llamacpp::slot_list))
        .route("/slots/{id}", post(api::llamacpp::slot_action))
        .route("/lora-adapters", get(api::llamacpp::lora_adapters).post(api::llamacpp::set_lora_adapters))
        .route("/apply-template", post(api::llamacpp::apply_template))
        .route("/embedding", post(api::llamacpp::embedding))
        .route("/embeddings", post(api::llamacpp::embeddings))
        .route("/api/tags", get(ollama_tags))
        .route("/api/version", get(ollama_version))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/generate", post(sglang_generate))
        .route("/generate_stream", post(api::tgi::generate_stream))
        .route("/info", get(api::tgi::info))
        .route("/get_model_info", get(sglang_model_info))
        .route("/get_server_info", get(sglang_server_info))
        .layer(DefaultBodyLimit::max(config.http.max_body_bytes))
        .layer(axum::middleware::from_fn_with_state(server_metrics.observability.clone(), crate::metrics::observe))
        .layer(axum::middleware::from_fn_with_state(http_context, http::layer))
        .with_state(state.clone());
    Ok(Built { state, router, worker_rx, metrics: server_metrics, jump_forward })
}

pub fn serve(
    mut runtime: InferenceRuntime,
    tokenizer: TokenizerBundle,
    config: ServerConfig,
) -> Result<(), String> {
    let address = listen_address(&config.host, config.port)?;
    let tls = config.http.tls_cert.is_some();
    let Built { state, router, worker_rx, metrics, jump_forward } = build(&runtime, tokenizer, config)?;
    runtime.set_jump_forward(jump_forward);
    let ready = state.ready.clone();
    let context = state.http.clone();
    api::studio_ops::spawn_idle_unload(state.metrics.clone());
    let scheduler_state = Arc::downgrade(&state);
    // the router owns the last sender of the worker channel; the worker must see it disconnect to stop
    drop(state);
    if !context.security.auth_required() && !address.ip().is_loopback() {
        tracing::warn!("listening on {address} without an API key: anyone who can reach this port can use the model (--api-key)");
    }
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let http_thread = thread::Builder::new()
        .name("fllm-http-runtime".into())
        .spawn(move || {
            let http_runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("cannot start HTTP runtime: {e}"))?;
            http_runtime.block_on(async move {
                // a server that replaces the one running before (model change in the Studio) waits until the old one has let go of the port
                let patient = std::env::var_os("FLLM_REPLACE_SERVER").is_some();
                let waiting_since = std::time::Instant::now();
                let listener = loop {
                    match tokio::net::TcpListener::bind(address).await {
                        Ok(listener) => break listener,
                        Err(_) if patient && waiting_since.elapsed() < std::time::Duration::from_secs(90) => {
                            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                        }
                        Err(e) => return Err(format!("cannot bind HTTP listener: {e}")),
                    }
                };
                println!("{}", crate::version::long_version());
                println!("fllm-rs listening on http{}://{address}", if tls { "s" } else { "" });
                if std::env::var_os("FLLM_OPEN_BROWSER").is_some() && !patient {
                    let host = if address.ip().is_unspecified() { "127.0.0.1".to_owned() } else { address.ip().to_string() };
                    open_browser(&format!("http{}://{}:{}/", if tls { "s" } else { "" }, if host.contains(':') { format!("[{host}]") } else { host }, address.port()));
                }
                ready.store(true, Ordering::Relaxed);
                api::schedule::start(scheduler_state, tokio::runtime::Handle::current());
                // the router (and with it the sender of the worker channel) lives until the last connection is gone
                http::serve(listener, router, context, ready.clone(), shutdown_signal(shutdown_rx)).await
            })
        })
        .map_err(|e| format!("cannot start HTTP thread: {e}"))?;

    // Keep the native OpenMP model path on the process main thread. Profiling
    // showed the same model call was 2.5-3x slower from a secondary thread.
    let worker_result = run_worker_loop(runtime, worker_rx, metrics);
    let _ = shutdown_tx.send(());
    let http_result = http_thread
        .join()
        .map_err(|_| "HTTP runtime thread panicked".to_owned())?;
    worker_result?;
    http_result?;
    Ok(())
}

/// Opens a page in the default browser (best effort; llm-rs studio uses it).
fn open_browser(url: &str) {
    let result = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if result.is_err() {
        println!("Open {url} in your browser.");
    }
}

/// --host as an address: IPv4, IPv6 (with or without brackets) or a name that resolves to one.
fn listen_address(host: &str, port: u16) -> Result<SocketAddr, String> {
    let bare = host.trim_matches(['[', ']']);
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return Ok(SocketAddr::new(ip, port));
    }
    std::net::ToSocketAddrs::to_socket_addrs(&(bare, port))
        .ok()
        .and_then(|mut found| found.next())
        .ok_or_else(|| format!("invalid listen address: {host}"))
}

/// Completes on the first SIGINT/SIGTERM (or when the worker has stopped); a second SIGINT ends the process at once.
async fn shutdown_signal(mut shutdown: tokio::sync::oneshot::Receiver<()>) {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("register SIGTERM handler");
        tokio::select! {
            _ = &mut shutdown => return,
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::select! {
        _ = &mut shutdown => return,
        _ = tokio::signal::ctrl_c() => {},
    }
    api::mcp::shutdown();
    tracing::info!("shutdown requested; finishing requests in flight (press Ctrl-C again to stop immediately)");
    tokio::spawn(async {
        let _ = tokio::signal::ctrl_c().await;
        std::process::exit(130);
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

/// Identifier that ties the 500 response of a client to the log line with the panic.
fn new_error_id() -> String {
    api::new_id("err")
}

/// Logs a panic of model code, counts it, and returns the message the affected request fails with.
fn record_panic(metrics: &Metrics, what: &str, payload: &(dyn std::any::Any + Send)) -> String {
    let id = new_error_id();
    metrics.worker_panics.fetch_add(1, Ordering::Relaxed);
    tracing::error!(error_id = %id, "panic while {what}: {}", panic_message(payload));
    format!("internal error while {what} (error id {id}); the failure was logged by the server")
}

fn run_worker_loop(
    mut runtime: InferenceRuntime,
    rx: std_mpsc::Receiver<WorkerCommand>,
    metrics: Arc<Metrics>,
) -> Result<(), String> {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut outputs = std::collections::HashMap::<String, GenerationTx>::new();
    runtime.attach_metrics(metrics.observability.clone());
    let mut previous_stats = crate::runtime::BatchStats::default();
    let mut consecutive_panics = 0u32;
    let beat = |metrics: &Metrics| {
        metrics
            .heartbeat_ms
            .store(metrics.started.elapsed().as_millis() as u64, Ordering::Relaxed)
    };
    loop {
        beat(&metrics);
        let (running, waiting, free, total, hits) = runtime.metrics_snapshot();
        metrics.observability.snapshot(running, waiting, free, total, hits);
        metrics.slots_running.store(running as u64, Ordering::Relaxed);
        metrics.slots_waiting.store(waiting as u64, Ordering::Relaxed);
        let cancelled = outputs
            .iter()
            .filter(|(_, output)| output.is_closed())
            .map(|(request_id, _)| request_id.clone())
            .collect::<Vec<_>>();
        for request_id in cancelled {
            outputs.remove(&request_id);
            let _ = runtime.cancel_request(&request_id);
        }
        for request_id in runtime.expire_requests() {
            if let Some(output) = outputs.remove(&request_id) {
                let _ = output.send(Err("the request exceeded its timeout".to_owned()));
            }
        }
        if runtime.active_requests() == 0 {
            // wake once a second while idle so that /ready can tell an idle worker from a stuck one
            match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(command) => { metrics.observability.dequeue(); accept_command(command, &mut runtime, &mut outputs, &metrics); },
                Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        while let Ok(command) = rx.try_recv() {
            metrics.observability.dequeue();
            accept_command(command, &mut runtime, &mut outputs, &metrics);
        }
        if runtime.active_requests() == 0 {
            continue;
        }
        let stepped = catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            {
                let delay = metrics.inject_delay_ms.load(Ordering::Relaxed);
                if delay > 0 {
                    thread::sleep(std::time::Duration::from_millis(delay));
                }
            }
            #[cfg(test)]
            if metrics.inject_panic.swap(0, Ordering::Relaxed) > 0 {
                panic!("injected test panic");
            }
            runtime.step()
        }));
        let stepped = match stepped {
            Ok(result) => result,
            Err(payload) => {
                // The batch that was running is in an unknown state: fail all requests in flight with an error id and
                // drop their sequences, then keep serving. Repeated panics without a good step in between stop the
                // server (a supervisor restarts it) instead of looping forever.
                let message = record_panic(&metrics, "running the model", payload.as_ref());
                for (request_id, output) in outputs.drain() {
                    let _ = output.send(Err(message.clone()));
                    let _ = catch_unwind(AssertUnwindSafe(|| runtime.cancel_request(&request_id)));
                }
                consecutive_panics += 1;
                if consecutive_panics >= 3 {
                    return Err("the model panicked in three consecutive steps; stopping".into());
                }
                continue;
            }
        };
        match stepped {
            Ok(events) => {
                consecutive_panics = 0;
                let stats = runtime.batch_stats();
                metrics.observability.processed(stats.prefill_rows.saturating_sub(previous_stats.prefill_rows));
                metrics.forward_calls.fetch_add(
                    (stats.forward_calls - previous_stats.forward_calls) as u64,
                    Ordering::Relaxed,
                );
                metrics.forced_tokens.fetch_add((stats.forced_tokens - previous_stats.forced_tokens) as u64, Ordering::Relaxed);
                metrics.jumps.fetch_add((stats.jumps - previous_stats.jumps) as u64, Ordering::Relaxed);
                metrics.rows_total.fetch_add(
                    (stats.rows_total - previous_stats.rows_total) as u64,
                    Ordering::Relaxed,
                );
                metrics.multi_row_calls.fetch_add(
                    (stats.calls_with_multiple_rows - previous_stats.calls_with_multiple_rows)
                        as u64,
                    Ordering::Relaxed,
                );
                metrics
                    .max_rows
                    .fetch_max(stats.max_rows as u64, Ordering::Relaxed);
                metrics.prefill_rows.fetch_add(
                    (stats.prefill_rows - previous_stats.prefill_rows) as u64,
                    Ordering::Relaxed,
                );
                metrics.decode_rows.fetch_add(
                    (stats.decode_rows - previous_stats.decode_rows) as u64,
                    Ordering::Relaxed,
                );
                metrics.draft_proposed.fetch_add(
                    (stats.draft_proposed - previous_stats.draft_proposed) as u64,
                    Ordering::Relaxed,
                );
                metrics.draft_accepted.fetch_add(
                    (stats.draft_accepted - previous_stats.draft_accepted) as u64,
                    Ordering::Relaxed,
                );
                metrics.scheduler_ns.fetch_add(
                    stats.scheduler_ns - previous_stats.scheduler_ns,
                    Ordering::Relaxed,
                );
                metrics.batch_build_ns.fetch_add(
                    stats.batch_build_ns - previous_stats.batch_build_ns,
                    Ordering::Relaxed,
                );
                metrics.model_forward_ns.fetch_add(
                    stats.model_forward_ns - previous_stats.model_forward_ns,
                    Ordering::Relaxed,
                );
                metrics.postprocess_ns.fetch_add(
                    stats.postprocess_ns - previous_stats.postprocess_ns,
                    Ordering::Relaxed,
                );
                let did_work = stats.rows_total > previous_stats.rows_total;
                previous_stats = stats;
                let no_events = events.is_empty();
                for event in events {
                    metrics.observability.generated();
                    let request_id = event.request_id.clone();
                    let finished = event.request_finished;
                    if let Some(output) = outputs.get(&request_id) {
                        if output.send(Ok(event)).is_err() {
                            let _ = runtime.cancel_request(&request_id);
                            outputs.remove(&request_id);
                        } else if finished {
                            outputs.remove(&request_id);
                        }
                    }
                }
                // a beam search that failed after it was accepted (the KV pool ran out while forking)
                for (request_id, error) in runtime.take_failures() {
                    if let Some(output) = outputs.remove(&request_id) {
                        let _ = output.send(Err(error));
                    }
                }
                // finished beam searches hand out their hypotheses as fast as the output channel takes them
                let mut waiting_for_room = false;
                for request_id in runtime.beam_outbox_requests() {
                    let Some(output) = outputs.get(&request_id) else {
                        let _ = runtime.cancel_request(&request_id);
                        continue;
                    };
                    let room = output.room();
                    if room == 0 {
                        waiting_for_room = true;
                        continue;
                    }
                    let (mut dropped, mut done) = (false, false);
                    for event in runtime.take_beam_events(&request_id, room) {
                        metrics.observability.generated();
                        done |= event.request_finished;
                        if output.send(Ok(event)).is_err() {
                            dropped = true;
                            break;
                        }
                    }
                    if dropped {
                        let _ = runtime.cancel_request(&request_id);
                        outputs.remove(&request_id);
                    } else if done {
                        outputs.remove(&request_id);
                    }
                }
                if waiting_for_room {
                    thread::sleep(std::time::Duration::from_millis(1));
                } else if no_events && runtime.active_requests() > 0 && !did_work {
                    thread::yield_now();
                }
            }
            Err(error) => {
                // like a panic: the requests of the failed step get the error and the server keeps serving; only repeated
                // failures without a good step in between stop it
                tracing::error!("a model step failed: {error}");
                for (request_id, output) in outputs.drain() {
                    let _ = output.send(Err(error.clone()));
                    let _ = catch_unwind(AssertUnwindSafe(|| runtime.cancel_request(&request_id)));
                }
                consecutive_panics += 1;
                if consecutive_panics >= 3 {
                    return Err(format!("three consecutive model steps failed; stopping (last error: {error})"));
                }
            }
        }
    }
    Ok(())
}

fn accept_command(
    command: WorkerCommand,
    runtime: &mut InferenceRuntime,
    outputs: &mut std::collections::HashMap<String, GenerationTx>,
    metrics: &Metrics,
) {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    match command {
        WorkerCommand::Embed {
            tokens,
            pooling,
            output,
        } => {
            if !output.is_closed() {
                let _active = metrics.observability.embedding();
                let result = catch_unwind(AssertUnwindSafe(|| runtime.embedding(&tokens, &pooling)))
                    .unwrap_or_else(|payload| Err(record_panic(metrics, "computing an embedding", payload.as_ref())));
                if result.is_ok() { metrics.observability.processed(tokens.len()); }
                let _ = output.send(result);
            }
        }
        WorkerCommand::Stats { output } => {
            let report = runtime.prefix_cache_report();
            // the receiver may be gone (the client hung up); that is not an error
            let _ = output.send(report);
        }
        WorkerCommand::ClearPrefixCache { output } => {
            let _ = output.send(runtime.clear_prefix_cache());
        }
        WorkerCommand::KvTierStats { output } => {
            let _ = output.send(runtime.kv_tier_stats());
        }
        WorkerCommand::ClearKvTier { output } => {
            let _ = output.send(runtime.clear_kv_tier());
        }
        WorkerCommand::Submit {
            request_id,
            prompt,
            max_tokens,
            prefix_salt,
            sampling,
            constraint,
            images,
            beam,
            output,
        } => {
            metrics.observability.queue_time(output.submitted.elapsed().as_secs_f64());
            // A panic in model code while a request is set up fails this request only.
            let admitted = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                let sequence_id = match beam {
                    Some(beam) => runtime.submit_beam(request_id.clone(), prompt, max_tokens, *sampling, *beam)?,
                    None => runtime.submit_with_sampling(request_id.clone(), prompt, max_tokens, *sampling)?,
                };
                // the salt has to be on the sequence before the first admission computes its prefix chain
                runtime.set_request_prefix_salt(&request_id, prefix_salt);
                let attached = runtime
                    .attach_media(sequence_id, *images)
                    .and_then(|()| match constraint {
                        Some(constraint) => runtime.set_constraint(sequence_id, *constraint),
                        None => Ok(()),
                    });
                if let Err(error) = attached {
                    let _ = runtime.cancel_request(&request_id);
                    return Err(error);
                }
                Ok(())
            }));
            match admitted {
                Ok(Ok(())) => {
                    outputs.insert(request_id, output);
                }
                Ok(Err(error)) => {
                    let _ = output.send(Err(error));
                }
                Err(payload) => {
                    let message = record_panic(metrics, "starting a request", payload.as_ref());
                    let _ = catch_unwind(AssertUnwindSafe(|| runtime.cancel_request(&request_id)));
                    let _ = output.send(Err(message));
                }
            }
        }
    }
}

async fn pooling(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let input = body
        .get("input")
        .or_else(|| body.get("prompt"))
        .unwrap_or(&Value::Null);
    let inputs = match input {
        Value::Array(items) if items.first().is_some_and(|v| v.is_string() || v.is_array()) => {
            items.clone()
        }
        _ => vec![input.clone()],
    };
    if inputs.is_empty() || inputs.len() > 32 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "pooling batch must contain 1 to 32 inputs",
        );
    }
    let mut tokenized = Vec::new();
    for item in &inputs {
        match input_tokens(&state, item) {
            Ok(ids) if ids.len() <= state.max_context => tokenized.push(ids),
            Ok(_) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "pooling input exceeds model context",
                )
            }
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        }
    }
    let task = body.get("task").and_then(Value::as_str).unwrap_or("embed");
    let pooling_mode = match task {
        "token_embed" | "last" => "last",
        _ => body.get("pooling").and_then(Value::as_str).unwrap_or("mean"),
    };
    let mode = if matches!(pooling_mode, "last" | "mean") { pooling_mode } else { "mean" };

    // a draining server takes no new work: the load balancer should send it to another one
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let _permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "inference request limit reached",
            )
        }
    };
    let mut data = Vec::new();
    let mut total_tokens = 0;
    for (index, tokens) in tokenized.into_iter().enumerate() {
        total_tokens += tokens.len();
        let (tx, rx) = tokio::sync::oneshot::channel();
        if state
            .worker
            .send(WorkerCommand::Embed {
                tokens,
                pooling: mode.into(),
                output: tx,
            })
            .is_err()
        {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable");
        }
        let vector = match rx.await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return api_error(StatusCode::SERVICE_UNAVAILABLE, &e),
            Err(_) => {
                return api_error(StatusCode::SERVICE_UNAVAILABLE, "embedding worker stopped")
            }
        };
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": vector,
        }));
    }
    Json(json!({
        "object": "list",
        "data": data,
        "model": state.model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }))
    .into_response()
}

async fn score(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let (t1, t2s) = if let (Some(t1), Some(t2)) = (
        body.get("text_1").and_then(Value::as_str),
        body.get("text_2"),
    ) {
        let docs = match t2 {
            Value::Array(arr) => arr
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect::<Vec<_>>(),
            Value::String(s) => vec![s.clone()],
            _ => return api_error(StatusCode::BAD_REQUEST, "text_2 must be string or string array"),
        };
        (t1.to_string(), docs)
    } else {
        return api_error(StatusCode::BAD_REQUEST, "request must include text_1 and text_2");
    };

    // a draining server takes no new work: the load balancer should send it to another one
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let _permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "inference request limit reached",
            )
        }
    };

    let embed_one = |text: String| {
        let st = state.clone();
        async move {
            let ids = input_tokens(&st, &Value::String(text))
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
            let tokens_len = ids.len();
            let (tx, rx) = tokio::sync::oneshot::channel();
            st.worker
                .send(WorkerCommand::Embed {
                    tokens: ids,
                    pooling: "mean".into(),
                    output: tx,
                })
                .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable".into()))?;
            let vec = rx.await
                .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "embedding worker stopped".into()))?
                .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e))?;
            Ok::<_, (StatusCode, String)>((vec, tokens_len))
        }
    };

    let (v1, t1_tokens) = match embed_one(t1).await {
        Ok(v) => v,
        Err((sc, msg)) => return api_error(sc, &msg),
    };

    let mut scores = Vec::new();
    let mut total_tokens = t1_tokens;
    for (index, doc) in t2s.into_iter().enumerate() {
        let (v2, d_tokens) = match embed_one(doc).await {
            Ok(v) => v,
            Err((sc, msg)) => return api_error(sc, &msg),
        };
        total_tokens += d_tokens;
        let dot: f32 = v1.iter().zip(&v2).map(|(&a, &b)| a * b).sum();
        let n1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let n2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
        let sim = if n1 > 0.0 && n2 > 0.0 { dot / (n1 * n2) } else { 0.0 };
        scores.push(json!({
            "index": index,
            "object": "score",
            "score": sim,
        }));
    }

    Json(json!({
        "object": "list",
        "model": state.model_name,
        "data": scores,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }))
    .into_response()
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let query = match body.get("query").and_then(Value::as_str) {
        Some(q) => q.to_string(),
        None => return api_error(StatusCode::BAD_REQUEST, "missing query"),
    };
    let documents = match body.get("documents").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect::<Vec<_>>(),
        None => return api_error(StatusCode::BAD_REQUEST, "missing documents array"),
    };
    let top_n = body
        .get("top_n")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(documents.len())
        .min(documents.len());
    let return_docs = body
        .get("return_documents")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // a draining server takes no new work: the load balancer should send it to another one
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let _permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "inference request limit reached",
            )
        }
    };

    let embed_one = |text: String| {
        let st = state.clone();
        async move {
            let ids = input_tokens(&st, &Value::String(text))
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
            let tokens_len = ids.len();
            let (tx, rx) = tokio::sync::oneshot::channel();
            st.worker
                .send(WorkerCommand::Embed {
                    tokens: ids,
                    pooling: "mean".into(),
                    output: tx,
                })
                .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable".into()))?;
            let vec = rx.await
                .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "embedding worker stopped".into()))?
                .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e))?;
            Ok::<_, (StatusCode, String)>((vec, tokens_len))
        }
    };

    let (q_vec, q_tokens) = match embed_one(query).await {
        Ok(v) => v,
        Err((sc, msg)) => return api_error(sc, &msg),
    };

    let mut ranked = Vec::new();
    let mut total_tokens = q_tokens;
    for (index, doc) in documents.into_iter().enumerate() {
        let (d_vec, d_tokens) = match embed_one(doc.clone()).await {
            Ok(v) => v,
            Err((sc, msg)) => return api_error(sc, &msg),
        };
        total_tokens += d_tokens;
        let dot: f32 = q_vec.iter().zip(&d_vec).map(|(&a, &b)| a * b).sum();
        let n1: f32 = q_vec.iter().map(|a| a * a).sum::<f32>().sqrt();
        let n2: f32 = d_vec.iter().map(|a| a * a).sum::<f32>().sqrt();
        let sim = if n1 > 0.0 && n2 > 0.0 { dot / (n1 * n2) } else { 0.0 };
        ranked.push((index, sim, doc));
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_n);

    let results: Vec<Value> = ranked
        .into_iter()
        .map(|(index, score, doc)| {
            if return_docs {
                json!({
                    "index": index,
                    "relevance_score": score,
                    "document": { "text": doc }
                })
            } else {
                json!({
                    "index": index,
                    "relevance_score": score,
                })
            }
        })
        .collect();

    let id = format!(
        "rerank_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    Json(json!({
        "id": id,
        "results": results,
        "model": state.model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        },
        "meta": {
            "api_version": { "version": "2" },
            "billed_units": { "search_units": 0, "input_tokens": total_tokens }
        }
    }))
    .into_response()
}

async fn classify(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let input = body.get("input").unwrap_or(&Value::Null);
    let inputs = match input {
        Value::Array(items) => items.clone(),
        _ => vec![input.clone()],
    };
    if inputs.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "input must not be empty");
    }
    // a draining server takes no new work: the load balancer should send it to another one
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let _permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "inference request limit reached",
            )
        }
    };
    let mut results = Vec::new();
    for item in inputs {
        let text = match item.as_str() {
            Some(s) => s.to_string(),
            None => return api_error(StatusCode::BAD_REQUEST, "each input must be a string"),
        };
        let ids = match input_tokens(&state, &Value::String(text)) {
            Ok(t) => t,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        if state
            .worker
            .send(WorkerCommand::Embed {
                tokens: ids,
                pooling: "last".into(),
                output: tx,
            })
            .is_err()
        {
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable");
        }
        let vec = match rx.await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return api_error(StatusCode::SERVICE_UNAVAILABLE, &e),
            Err(_) => return api_error(StatusCode::SERVICE_UNAVAILABLE, "worker stopped"),
        };
        let max_val = vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = vec.iter().map(|&x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = vec.iter().map(|&x| if exp_sum > 0.0 { (x - max_val).exp() / exp_sum } else { 0.0 }).collect();
        results.push(json!({
            "label": "label_0",
            "probs": probs,
            "logits": vec,
        }));
    }
    Json(json!({
        "object": "list",
        "model": state.model_name,
        "data": results,
    }))
    .into_response()
}

/// Liveness: the process answers HTTP. It stays `ok` while the model loads or drains so that orchestrators do not
/// restart a healthy process; `/ready` says whether requests can be served.
async fn health() -> Json<Value> {
    Json(json!({"status":"ok"}))
}

/// Readiness: 200 only while the listener is up, the model is loaded and the worker thread keeps making progress
/// (it wakes at least once a second when idle); 503 during start-up, shutdown or when the worker is stuck.
async fn ready_check(State(state): State<Arc<AppState>>) -> Response {
    let now = state.metrics.started.elapsed().as_millis() as u64;
    let idle_ms = now.saturating_sub(state.metrics.heartbeat_ms.load(Ordering::Relaxed));
    let serving = state.ready.load(Ordering::Relaxed) && !state.draining.load(Ordering::Relaxed);
    let responsive = idle_ms < state.metrics.stall_ms.load(Ordering::Relaxed);
    let status = if serving && responsive { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    let reason = if state.draining.load(Ordering::Relaxed) { "draining" } else if !serving { "starting or shutting down" } else if !responsive { "worker not responding" } else { "ready" };
    (
        status,
        Json(json!({
            "status": if status == StatusCode::OK { "ready" } else { "not_ready" },
            "reason": reason,
            "model": state.model_name,
            "worker_idle_ms": idle_ms,
            "worker_panics": state.metrics.worker_panics.load(Ordering::Relaxed),
        })),
    )
        .into_response()
}

async fn webui(State(state): State<Arc<AppState>>) -> Response {
    crate::webui::page(state.http.security.auth_required())
}

/// Counters of the HTTP layer and the worker that are not part of the inference metrics.
fn server_health_metrics(state: &AppState) -> String {
    let counters = &state.http.counters;
    format!(
        "# HELP fllm_worker_panics_total Panics in model code that were contained and failed the affected requests\n# TYPE fllm_worker_panics_total counter\nfllm_worker_panics_total {}\n\
         # HELP fllm_queue_rejections_total Requests answered with 429 because the admission queue was full\n# TYPE fllm_queue_rejections_total counter\nfllm_queue_rejections_total {}\n\
         # HELP fllm_rate_limited_total Requests answered with 429 because an API key exceeded --rate-limit\n# TYPE fllm_rate_limited_total counter\nfllm_rate_limited_total {}\n\
         # HELP fllm_http_connections Open client connections\n# TYPE fllm_http_connections gauge\nfllm_http_connections {}\n\
         # HELP fllm_http_connections_rejected_total Connections refused because --max-connections was reached\n# TYPE fllm_http_connections_rejected_total counter\nfllm_http_connections_rejected_total {}\n\
         # HELP fllm_http_request_timeouts_total Requests cancelled by --request-timeout\n# TYPE fllm_http_request_timeouts_total counter\nfllm_http_request_timeouts_total {}\n",
        state.metrics.worker_panics.load(Ordering::Relaxed),
        state.metrics.queue_rejections.load(Ordering::Relaxed),
        state.http.security.rejected_total(),
        counters.connections_open.load(Ordering::Relaxed),
        counters.connections_rejected.load(Ordering::Relaxed),
        counters.request_timeouts.load(Ordering::Relaxed),
    )
}

/// `GET /admin/prefix-cache`: the entries the prefix cache holds, their hashes and the hit counter. Goes through the worker,
/// because the cache lives on the inference thread.
async fn admin_prefix_cache(State(state): State<Arc<AppState>>) -> Response {
    let (output, answer) = tokio::sync::oneshot::channel();
    if state.worker.send(WorkerCommand::Stats { output }).is_err() {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker is not running");
    }
    match answer.await {
        Ok(report) => Json(report).into_response(),
        Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker did not answer"),
    }
}

/// `DELETE /admin/prefix-cache`: drop every entry and hand the blocks back to the allocator.
async fn admin_prefix_cache_clear(State(state): State<Arc<AppState>>) -> Response {
    let (output, answer) = tokio::sync::oneshot::channel();
    if state.worker.send(WorkerCommand::ClearPrefixCache { output }).is_err() {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker is not running");
    }
    match answer.await {
        Ok(Ok(dropped)) => Json(json!({"dropped": dropped})).into_response(),
        Ok(Err(error)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
        Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker did not answer"),
    }
}

/// `GET /admin/kv-tier`: how much the disk tier holds and how often it was used.
async fn admin_kv_tier(State(state): State<Arc<AppState>>) -> Response {
    let (output, answer) = tokio::sync::oneshot::channel();
    if state.worker.send(WorkerCommand::KvTierStats { output }).is_err() {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker is not running");
    }
    match answer.await {
        Ok(stats) => Json(stats).into_response(),
        Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker did not answer"),
    }
}

/// `DELETE /admin/kv-tier`: drop every stored block on disk.
async fn admin_kv_tier_clear(State(state): State<Arc<AppState>>) -> Response {
    let (output, answer) = tokio::sync::oneshot::channel();
    if state.worker.send(WorkerCommand::ClearKvTier { output }).is_err() {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker is not running");
    }
    match answer.await {
        Ok(removed) => Json(json!({"removed": removed})).into_response(),
        Err(_) => api_error(StatusCode::SERVICE_UNAVAILABLE, "the inference worker did not answer"),
    }
}

/// `GET /admin/config`: the settings that are actually in force — what the environment and the page decided, in one place.
async fn admin_config(State(state): State<Arc<AppState>>) -> Response {
    let env = |name: &str| std::env::var(name).ok().filter(|value| !value.trim().is_empty());
    Json(json!({
        "model": state.model_name,
        "model_arch": state.model_arch,
        "cache_identity": state.cache_identity,
        "max_context": state.max_context,
        "max_sequences": state.max_sequences,
        "ready": state.ready.load(Ordering::Relaxed),
        "draining": state.draining.load(Ordering::Relaxed),
        "output_cache": state.cache.stats(),
        "prefix_cache": !matches!(
            env("FLLM_PREFIX_CACHE").unwrap_or_default().trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "no" | "false"
        ),
        "prefix_salt_set": env("FLLM_PREFIX_SALT").is_some(),
        "prefix_seed": crate::scheduler::prefix_seed(),
        "preemption": match env("FLLM_PREEMPTION").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "swap" => "swap",
            _ => "recompute",
        },
        "telemetry": state.tracer.stats(),
        "coalescing": state.coalesce.stats(),
        "sandbox": crate::server::api::agent_sandbox::status_json(),
    }))
    .into_response()
}

/// `GET /admin/telemetry`: whether the OTLP exporter is on and how many spans left the process.
async fn admin_telemetry(State(state): State<Arc<AppState>>) -> Response {
    Json(state.tracer.stats()).into_response()
}

/// `POST /admin/drain`: no new requests, running ones finish (for a rolling restart).
async fn admin_drain(State(state): State<Arc<AppState>>) -> Response {
    state.draining.store(true, Ordering::Relaxed);
    tracing::info!("draining: no new requests are accepted, running work continues");
    Json(json!({"draining": true})).into_response()
}

/// `DELETE /admin/drain`: accept work again.
async fn admin_undrain(State(state): State<Arc<AppState>>) -> Response {
    state.draining.store(false, Ordering::Relaxed);
    tracing::info!("draining cancelled");
    Json(json!({"draining": false})).into_response()
}

/// `GET /admin/cache`: what the output cache holds and how often it was used.
async fn admin_cache(State(state): State<Arc<AppState>>) -> Response {
    Json(state.cache.stats()).into_response()
}

/// `DELETE /admin/cache`: drops every cached answer (after a model change, or when memory is needed).
async fn admin_cache_clear(State(state): State<Arc<AppState>>) -> Response {
    let cleared = state.cache.clear();
    tracing::info!("output cache cleared ({cleared} answers)");
    Json(json!({"cleared": cleared})).into_response()
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    use options::MetricsStyle;
    let style = state.http.options.metrics_style;
    let mut body = String::new();
    if style != MetricsStyle::Vllm {
        body = format!(
        "# HELP fllm_requests_total Requests accepted by the Rust server\n# TYPE fllm_requests_total counter\nfllm_requests_total {}\n# HELP fllm_completion_tokens_total Tokens emitted by the Rust server\n# TYPE fllm_completion_tokens_total counter\nfllm_completion_tokens_total {}\n# HELP fllm_model_forward_calls_total Model forward calls\n# TYPE fllm_model_forward_calls_total counter\nfllm_model_forward_calls_total {}\n# HELP fllm_model_batch_rows_total Rows passed to the model\n# TYPE fllm_model_batch_rows_total counter\nfllm_model_batch_rows_total {}\n# HELP fllm_model_multirow_calls_total Model calls with multiple rows\n# TYPE fllm_model_multirow_calls_total counter\nfllm_model_multirow_calls_total {}\n# HELP fllm_model_batch_max_rows Largest model batch observed\n# TYPE fllm_model_batch_max_rows gauge\nfllm_model_batch_max_rows {}\n# HELP fllm_prefill_rows_total Prefill token rows submitted to the model\n# TYPE fllm_prefill_rows_total counter\nfllm_prefill_rows_total {}\n# HELP fllm_decode_rows_total Decode token rows submitted to the model\n# TYPE fllm_decode_rows_total counter\nfllm_decode_rows_total {}\n# HELP fllm_scheduler_seconds_total Time spent scheduling\n# TYPE fllm_scheduler_seconds_total counter\nfllm_scheduler_seconds_total {:.9}\n# HELP fllm_batch_build_seconds_total Time spent preparing model batches\n# TYPE fllm_batch_build_seconds_total counter\nfllm_batch_build_seconds_total {:.9}\n# HELP fllm_model_forward_seconds_total Time spent in model forward calls\n# TYPE fllm_model_forward_seconds_total counter\nfllm_model_forward_seconds_total {:.9}\n# HELP fllm_postprocess_seconds_total Time spent applying logits and updating sequences\n# TYPE fllm_postprocess_seconds_total counter\nfllm_postprocess_seconds_total {:.9}\n",
        state.metrics.requests.load(Ordering::Relaxed),
        state.metrics.generated_tokens.load(Ordering::Relaxed),
        state.metrics.forward_calls.load(Ordering::Relaxed),
        state.metrics.rows_total.load(Ordering::Relaxed),
        state.metrics.multi_row_calls.load(Ordering::Relaxed),
        state.metrics.max_rows.load(Ordering::Relaxed),
        state.metrics.prefill_rows.load(Ordering::Relaxed),
        state.metrics.decode_rows.load(Ordering::Relaxed),
        state.metrics.scheduler_ns.load(Ordering::Relaxed) as f64 / 1e9,
        state.metrics.batch_build_ns.load(Ordering::Relaxed) as f64 / 1e9,
        state.metrics.model_forward_ns.load(Ordering::Relaxed) as f64 / 1e9,
        state.metrics.postprocess_ns.load(Ordering::Relaxed) as f64 / 1e9,
        );
    // the output cache reports itself: hits, misses, evictions, bytes
    body.push_str(&state.cache.prometheus());
    let (leaders, followers) = state.coalesce.counters();
    body.push_str(&format!(
        "# HELP fllm_draining Whether the server refuses new work (1) or not (0)\n# TYPE fllm_draining gauge\nfllm_draining {}\n\
         # HELP fllm_coalescing_leaders_total Requests that computed an answer others waited for\n# TYPE fllm_coalescing_leaders_total counter\nfllm_coalescing_leaders_total {}\n\
         # HELP fllm_coalesced_requests_total Requests that waited for another request's answer\n# TYPE fllm_coalesced_requests_total counter\nfllm_coalesced_requests_total {}\n\
         # HELP fllm_otel_spans_sent_total Spans exported over OTLP\n# TYPE fllm_otel_spans_sent_total counter\nfllm_otel_spans_sent_total {}\n",
        u8::from(state.draining.load(Ordering::Relaxed)),
        leaders,
        followers,
        state.tracer.sent(),
    ));
        body.push_str(&format!("# TYPE fllm_draft_proposed_tokens_total counter\nfllm_draft_proposed_tokens_total {}\n# TYPE fllm_draft_accepted_tokens_total counter\nfllm_draft_accepted_tokens_total {}\n",state.metrics.draft_proposed.load(Ordering::Relaxed),state.metrics.draft_accepted.load(Ordering::Relaxed)));
        body.push_str(&state.metrics.observability.render());
        body.push_str(&server_health_metrics(&state));
    }
    if style != MetricsStyle::Fllm {
        body.push_str(&state.metrics.observability.render_vllm());
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

impl AppState {
    /// `Some(503)` while the server is draining.
    fn draining_response(&self) -> Option<Response> {
        if !self.draining.load(Ordering::Relaxed) {
            return None;
        }
        let mut response = api_error(StatusCode::SERVICE_UNAVAILABLE, "the server is draining: it accepts no new requests");
        response.headers_mut().insert("retry-after", axum::http::HeaderValue::from_static("30"));
        Some(response)
    }

    /// Whether a request's `model` field names this server's model: the served name, an alias (`*` at the end of an
    /// alias matches any suffix), or nothing at all. HOOK(multi-model): the lookup in a model registry goes here.
    fn knows_model(&self, requested: &str) -> bool {
        let requested = requested.trim();
        let aliases = self.aliases.read().unwrap();
        requested.is_empty()
            || requested == self.model_name
            || aliases.iter().any(|alias| match alias.strip_suffix('*') {
                Some(prefix) => requested.starts_with(prefix),
                None => alias == requested,
            })
    }

    /// The `model` of an OpenAI/Anthropic-style request body. `Ok` carries the name to echo back.
    #[allow(clippy::result_large_err)]
    fn resolve_model(&self, body: &Value) -> Result<String, Response> {
        match body.get("model") {
            None | Some(Value::Null) => Ok(self.model_name.clone()),
            Some(Value::String(name)) if name.trim().is_empty() => Ok(self.model_name.clone()),
            Some(Value::String(name)) if self.knows_model(name) => Ok(name.clone()),
            Some(Value::String(name)) => Err(model_not_found(name)),
            Some(_) => Err(api_error_with(StatusCode::BAD_REQUEST, "model must be a string", None, Some("model"))),
        }
    }
}

/// OpenAI's reply for an unknown model.
fn model_not_found(name: &str) -> Response {
    api_error_with(
        StatusCode::NOT_FOUND,
        &format!("The model `{name}` does not exist or you do not have access to it."),
        Some("model_not_found"),
        Some("model"),
    )
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if headers.contains_key("anthropic-version") {
        return api::anthropic::models(&state);
    }
    let created = unix_now();
    // capabilities let clients discover media input; they mirror what the loaded encoder accepts
    let vision = state.vision.as_ref().is_some_and(|v| v.marker(crate::vision::MediaKind::Image).is_some());
    let video = state.vision.as_ref().is_some_and(|v| v.marker(crate::vision::MediaKind::Video).is_some());
    let audio = state.vision.as_ref().is_some_and(|v| v.family() == "whisper");
    let mut modalities = vec!["text"];
    modalities.extend(vision.then_some("image"));
    modalities.extend(video.then_some("video"));
    modalities.extend(audio.then_some("audio"));
    let aliases = state.aliases.read().unwrap();
    let data: Vec<Value> = std::iter::once(&state.model_name)
        .chain(aliases.iter().filter(|alias| !alias.ends_with('*')))
        .map(|id| {
            json!({"id":id,"object":"model","created":created,"owned_by":"fllm","root":state.model_name,
                   "parent":null,"max_model_len":state.max_context,
                   "capabilities":{"vision":vision,"video":video,"audio":audio,"thinking":state.thinking.to_json()},"modalities":modalities})
        })
        .collect();
    Json(json!({"object":"list","data":data})).into_response()
}

async fn ollama_tags(State(state): State<Arc<AppState>>) -> Json<Value> {
    let aliases = state.aliases.read().unwrap();
    let base = api::ollama_extra::model_entry(&state);
    let mut models = vec![base.clone()];
    for alias in aliases.iter().filter(|a| !a.ends_with('*')) {
        let mut entry = base.clone();
        entry["name"] = json!(alias);
        entry["model"] = json!(alias);
        models.push(entry);
    }
    Json(json!({"models": models}))
}

async fn ollama_version() -> Json<Value> {
    Json(json!({"version":"0.4.1-rust", "fllm_version": env!("CARGO_PKG_VERSION"), "build": crate::version::long_version()}))
}

async fn completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if let Err(response) = state.resolve_model(&body) {
        return response;
    }
    let prompt_val = match body.get("prompt") {
        Some(v) => v,
        None => return api_error(StatusCode::BAD_REQUEST, "prompt is required"),
    };
    let prompt_ids = match prompt_val {
        Value::String(s) => match state.tokenizer.encode(s) {
            Ok(ids) => ids.into_iter().map(i64::from).collect::<Vec<_>>(),
            Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
        },
        Value::Array(arr) => {
            let mut ids = Vec::with_capacity(arr.len());
            for item in arr {
                match item.as_i64() {
                    Some(id) => ids.push(id),
                    None => {
                        return api_error(
                            StatusCode::BAD_REQUEST,
                            "prompt array elements must be integer token IDs",
                        )
                    }
                }
            }
            if ids.is_empty() {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "prompt token array cannot be empty",
                );
            }
            ids
        }
        _ => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "prompt must be a string or array of integer token IDs",
            )
        }
    };
    openai_generate(state, body, prompt_ids, false, None, ImageOverrides::default()).await
}

async fn chat_completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if let Err(response) = state.resolve_model(&body) {
        return response;
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "messages must be a non-empty array",
        );
    };
    if messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "messages must be a non-empty array",
        );
    }
    if state.vision.is_some() && media::request_has_media(messages) {
        let (prompt, ids, images) = match media::prepare_media_chat(&state, messages).await {
            Ok(p) => p,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        };
        let vision_tools = match tool_spec_for(&state, &body) {
            Ok(spec) => spec,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        };
        let reasoning = ReasoningSpec::for_request(&state, &prompt, &vision_tools);
        return openai_generate(state, body, ids, true, Some(reasoning), images).await;
    }
    let tools = body.get("tools");
    if tools.is_some_and(|v| !v.is_array()) {
        return api_error(StatusCode::BAD_REQUEST, "tools must be an array");
    }
    let tool_spec = match tool_spec_for(&state, &body) {
        Ok(s) => s,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    let kwargs = body.get("chat_template_kwargs");
    if kwargs.is_some_and(|v| !v.is_object()) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "chat_template_kwargs must be an object",
        );
    }
    let mut template_kwargs = kwargs
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if tool_spec.forced_schema.is_some()
        || tool_spec.has_native_grammar()
        || body
            .get("response_format")
            .is_some_and(|f| f["type"] != "text")
    {
        template_kwargs.insert("enable_thinking".into(), json!(false));
    }
    if let Some(effort) = body.get("reasoning_effort") {
        if !effort.is_string() {
            return api_error(StatusCode::BAD_REQUEST, "reasoning_effort must be a string");
        }
        template_kwargs.insert("reasoning_effort".into(), effort.clone());
    }
    // thinking budget: `thinking_budget` tokens, or on a model with a thinking switch the budget of a reasoning_effort level
    let thinking_budget = match crate::thinking::request_budget(
        body.get("thinking_budget"),
        body.get("reasoning_effort").and_then(Value::as_str),
        state.thinking.control,
    ) {
        Ok(budget) => budget,
        Err(error) => return api_error_with(StatusCode::BAD_REQUEST, &error, None, Some("thinking_budget")),
    };
    if thinking_budget.is_some() && body.get("thinking_budget").is_some_and(|v| !v.is_null()) && body.get("n").and_then(Value::as_u64).unwrap_or(1) > 1 {
        return api_error_with(StatusCode::BAD_REQUEST, "thinking_budget cannot be combined with n > 1", None, Some("thinking_budget"));
    }
    if state.thinking.control == crate::thinking::Control::Toggle && !state.thinking.variable.is_empty() {
        // asking for an effort or a budget switches thinking on, "none" off, unless the request set the switch itself
        let asked = thinking_budget.is_some() || body.get("reasoning_effort").is_some();
        if asked && !template_kwargs.contains_key(state.thinking.variable) {
            let off = body.get("reasoning_effort").and_then(Value::as_str).is_some_and(crate::thinking::effort_is_off);
            template_kwargs.insert(state.thinking.variable.into(), json!(!off));
        }
    }
    let prompt = match render_messages(
        &state,
        Value::Array(messages.clone()),
        if tool_spec.disabled {
            None
        } else {
            tools.cloned()
        },
        Some(template_kwargs),
    ) {
        Ok(prompt) => prompt,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    let prompt_ids = match state.tokenizer.encode(&prompt) {
        Ok(ids) => ids.into_iter().map(i64::from).collect::<Vec<_>>(),
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    let mut reasoning = ReasoningSpec::for_request(&state, &prompt, &tool_spec);
    reasoning.budget = thinking_budget;
    openai_generate(
        state,
        body,
        prompt_ids,
        true,
        Some(reasoning),
        ImageOverrides::default(),
    )
    .await
}

async fn image_embeddings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(url) = body.get("image_url").and_then(Value::as_str) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "image_url must contain a PNG/JPEG data URL",
        );
    };
    match media::encode_single_image(&state, url).await {
        Ok(rows) => Json(json!({"model":state.model_name,"engine":"fllm-rust","tokens":rows.len(),"dimensions":state.hidden_size,"data":rows})).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

type ImageOverrides = crate::vision::pipeline::MediaInputs;
fn input_tokens(state: &AppState, input: &Value) -> Result<Vec<i64>, String> {
    match input {
        Value::String(s) if !s.is_empty() => state
            .tokenizer
            .encode(s)
            .map(|v| v.into_iter().map(i64::from).collect()),
        Value::Array(items) if !items.is_empty() => items
            .iter()
            .map(|v| {
                v.as_i64()
                    .filter(|n| *n >= 0 && (*n as usize) < state.tokenizer.vocab_size())
                    .ok_or_else(|| {
                        "token IDs must be integers within the model vocabulary".to_string()
                    })
            })
            .collect(),
        _ => Err("input must be a non-empty string or token ID array".into()),
    }
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    if let Err(response) = state.resolve_model(&body) {
        return response;
    }
    let encoding = body
        .get("encoding_format")
        .and_then(Value::as_str)
        .unwrap_or("float");
    if !matches!(encoding, "float" | "base64") {
        return api_error(
            StatusCode::BAD_REQUEST,
            "encoding_format must be float or base64",
        );
    }
    if body
        .get("dimensions")
        .is_some_and(|d| d.as_u64() != Some(state.hidden_size as u64))
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "dimensions must equal the loaded model hidden size",
        );
    }
    let input = body
        .get("input")
        .or_else(|| body.get("prompt"))
        .unwrap_or(&Value::Null);
    let inputs = match input {
        Value::Array(items) if items.first().is_some_and(|v| v.is_string() || v.is_array()) => {
            items.clone()
        }
        _ => vec![input.clone()],
    };
    if inputs.is_empty() || inputs.len() > 32 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "embedding batch must contain 1 to 32 inputs",
        );
    }
    // Ollama (`truncate`, default true) cuts inputs that are too long; OpenAI and vLLM reject them
    let truncate = body.get("truncate").and_then(Value::as_bool).unwrap_or(false);
    let mut tokenized = Vec::new();
    for input in &inputs {
        match input_tokens(&state, input) {
            Ok(ids) if ids.len() <= state.max_context => tokenized.push(ids),
            Ok(mut ids) if truncate => {
                ids.truncate(state.max_context);
                tokenized.push(ids);
            }
            Ok(_) => {
                return api_error_with(
                    StatusCode::BAD_REQUEST,
                    "embedding input exceeds the model context; shorten it or send truncate:true (Ollama API)",
                    Some("context_length_exceeded"),
                    Some("input"),
                )
            }
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        }
    }
    let pooling = body
        .get("pooling")
        .and_then(Value::as_str)
        .unwrap_or("mean");
    if !matches!(pooling, "mean" | "last") {
        return api_error(StatusCode::BAD_REQUEST, "pooling must be mean or last");
    }
    // a draining server takes no new work: the load balancer should send it to another one
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let _permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return queue_full(&state),
    };
    if let Some(caller) = security::current_caller() {
        let total: usize = tokenized.iter().map(Vec::len).sum();
        caller.stats.prompt_tokens.fetch_add(total as u64, Ordering::Relaxed);
        if let Some(limiter) = &caller.limiter {
            limiter.charge_tokens(total);
        }
        state.metrics.observability.prompt_tokens(total);
    }
    let started = Instant::now();
    let mut data = Vec::new();
    let mut vectors = Vec::new();
    let mut count = 0;
    for (index, tokens) in tokenized.into_iter().enumerate() {
        count += tokens.len();
        let (tx, rx) = tokio::sync::oneshot::channel();
        state.metrics.observability.queue();
        if state
            .worker
            .send(WorkerCommand::Embed {
                tokens,
                pooling: pooling.into(),
                output: tx,
            })
            .is_err()
        {
            state.metrics.observability.dequeue();
            return api_error(StatusCode::SERVICE_UNAVAILABLE, "model worker unavailable");
        }
        let vector = match rx.await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return api_error(StatusCode::SERVICE_UNAVAILABLE, &e),
            Err(_) => {
                return api_error(StatusCode::SERVICE_UNAVAILABLE, "embedding worker stopped")
            }
        };
        let encoded = if encoding == "base64" {
            use base64::Engine;
            let bytes: Vec<u8> = vector
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            json!(base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            json!(vector)
        };
        data.push(json!({"object":"embedding","index":index,"embedding":encoded}));
        vectors.push(vector);
    }
    // Both standard API surfaces use the same transformer execution and pooling.
    Json(json!({"object":"list","model":state.model_name,"data":data,"usage":{"prompt_tokens":count,"total_tokens":count},
                "embeddings":vectors,"embedding":vectors.first(),"prompt_eval_count":count,"total_duration":started.elapsed().as_nanos() as u64})).into_response()
}

#[derive(Debug)]
struct StopUpdate {
    delta: String,
    matched: bool,
}

struct StopFilter {
    stops: Vec<String>,
    emitted: String,
    ignore_before: usize,
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            emitted: String::new(),
            ignore_before: 0,
        }
    }

    fn update_with_minimum(&mut self, full_text: &str, final_chunk: bool, allow_stop: bool) -> StopUpdate {
        if allow_stop { return self.update(full_text, final_chunk); }
        self.ignore_before = full_text.trim_end_matches('\u{fffd}').len();
        let stops = std::mem::take(&mut self.stops);
        let result = self.update(full_text, final_chunk);
        self.stops = stops;
        result
    }
    fn update(&mut self, full_text: &str, final_chunk: bool) -> StopUpdate {
        // A byte-level token can end halfway through a UTF-8 code point. The
        // tokenizer temporarily decodes that suffix as U+FFFD; hold it until
        // the next token completes the bytes, or generation actually ends.
        let full_text = if final_chunk {
            full_text
        } else {
            full_text.trim_end_matches('\u{fffd}')
        };
        let matched_at = self
            .stops
            .iter()
            .filter_map(|stop| full_text.get(self.ignore_before..)?.find(stop).map(|at| at + self.ignore_before))
            .min();
        let visible_end = if let Some(index) = matched_at {
            index
        } else if final_chunk {
            full_text.len()
        } else {
            safe_stop_prefix_end(full_text, &self.stops)
        };
        let visible = &full_text[..visible_end];
        let delta = visible
            .strip_prefix(&self.emitted)
            .unwrap_or(visible)
            .to_owned();
        self.emitted = visible.to_owned();
        StopUpdate {
            delta,
            matched: matched_at.is_some(),
        }
    }

    fn text(&self) -> &str {
        &self.emitted
    }
}

fn request_constraint(
    state: &AppState,
    body: &Value,
) -> Result<Option<llguidance::Matcher>, String> {
    crate::structured::request_grammar(body)?
        .map(|grammar| crate::structured::matcher(&state.grammar_factory, grammar))
        .transpose()
}

fn request_sampling(
    state: &AppState,
    body: &Value,
    defaults: crate::sampling::SamplingParams,
) -> Result<crate::sampling::SamplingParams, String> {
    let mut merged = state.studio_config.read().unwrap().sampling.clone();
    merged.temperature = defaults.temperature;
    merged.top_p = defaults.top_p;
    merged.top_k = defaults.top_k;
    let mut params = crate::parameters::parse_sampling(body, merged, state.tokenizer.vocab_size())?;
    if let Some(words) = body.get("bad_words") {
        let words = words
            .as_array()
            .filter(|w| w.len() <= 64)
            .ok_or("bad_words must be an array of at most 64 strings")?;
        for word in words {
            let word = word.as_str().ok_or("bad_words must contain strings")?;
            // a word is blocked both at the start of a text and after a space, as the tokenizer splits them differently
            for variant in [word.to_owned(), format!(" {word}")] {
                if variant.trim().is_empty() {
                    continue;
                }
                let ids: Vec<i64> = state.tokenizer.encode_without_bos(&variant)?.into_iter().map(i64::from).collect();
                if !ids.is_empty() && !params.bad_word_ids.contains(&ids) {
                    params.bad_word_ids.push(ids);
                }
            }
        }
    }
    if params.dry_multiplier > 0.0 {
        let breakers = match body.get("dry_sequence_breakers") {
            None => vec!["\n", ":", "\"", "*"],
            Some(value) => value
                .as_array()
                .ok_or("dry_sequence_breakers must be an array of strings")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or("dry_sequence_breakers must contain strings")
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        for breaker in breakers {
            if !breaker.is_empty() {
                params.dry_breakers.push(
                    state
                        .tokenizer
                        .encode_without_bos(breaker)?
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                );
            }
        }
    }
    Ok(params)
}

fn parse_stop_strings(value: Option<&Value>) -> Result<Vec<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(stop)) => Ok(if stop.is_empty() {
            Vec::new()
        } else {
            vec![stop.clone()]
        }),
        Some(Value::Array(values)) => {
            let mut stops = Vec::with_capacity(values.len());
            for value in values {
                let Some(stop) = value.as_str() else {
                    return Err("stop must be a string or an array of strings".into());
                };
                if !stop.is_empty() {
                    stops.push(stop.to_owned());
                }
            }
            Ok(stops)
        }
        _ => Err("stop must be a string or an array of strings".into()),
    }
}

fn safe_stop_prefix_end(text: &str, stops: &[String]) -> usize {
    let max_stop_chars = stops
        .iter()
        .map(|stop| stop.chars().count())
        .max()
        .unwrap_or(0);
    if max_stop_chars <= 1 || text.is_empty() {
        return text.len();
    }
    let offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
        .collect::<Vec<_>>();
    let char_count = offsets.len() - 1;
    let max_suffix = (max_stop_chars - 1).min(char_count);
    for held_chars in (1..=max_suffix).rev() {
        let start = offsets[char_count - held_chars];
        let suffix = &text[start..];
        if stops.iter().any(|stop| stop.starts_with(suffix)) {
            return start;
        }
    }
    text.len()
}

fn openai_sampling_controls(
    body: &Value,
    chat: bool,
    vocab: usize,
    sampling: &mut crate::sampling::SamplingParams,
) -> Result<(), String> {
    let top = body
        .get("top_logprobs")
        .map(|v| {
            v.as_u64()
                .filter(|v| *v <= 20)
                .ok_or("top_logprobs must be an integer from 0 to 20")
        })
        .transpose()?;
    sampling.logprobs = if chat {
        match body.get("logprobs") {
            None | Some(Value::Null) | Some(Value::Bool(false)) if top.is_none() => None,
            Some(Value::Bool(true)) => Some(top.unwrap_or(0) as usize),
            _ => {
                return Err(
                    "chat logprobs must be boolean; top_logprobs requires logprobs:true".into(),
                )
            }
        }
    } else {
        match body.get("logprobs") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_u64()
                    .filter(|n| *n <= 5)
                    .ok_or("logprobs must be an integer from 0 to 5")? as usize,
            ),
        }
    };
    if let Some(v) = body.get("min_tokens") {
        sampling.min_tokens = usize::try_from(
            v.as_u64()
                .ok_or("min_tokens must be a non-negative integer")?,
        )
        .map_err(|_| "min_tokens is too large")?;
    }
    if let Some(v) = body.get("ignore_eos") {
        sampling.ignore_eos = v.as_bool().ok_or("ignore_eos must be boolean")?;
    }
    if let Some(v) = body.get("echo") {
        v.as_bool().ok_or("echo must be boolean")?;
    }
    if let Some(v) = body.get("stop_token_ids") {
        sampling.stop_token_ids = v
            .as_array()
            .filter(|v| v.len() <= 256)
            .ok_or("stop_token_ids must be an array with at most 256 IDs")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|id| *id < vocab as u64)
                    .map(|id| id as i64)
                    .ok_or_else(|| "stop_token_ids contains an invalid vocabulary ID".to_string())
            })
            .collect::<Result<_, _>>()?;
    }
    if let Some(v) = body.get("stream_options") {
        if !v.is_object() || v.get("include_usage").is_some_and(|v| !v.is_boolean()) {
            return Err("stream_options must be an object with boolean include_usage".into());
        }
    }
    Ok(())
}

fn openai_usage(prompt: usize, completion: usize, cached: usize, reasoning: usize) -> Value {
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion,
        "prompt_tokens_details":{"cached_tokens":cached.min(prompt)},
        "completion_tokens_details":{"reasoning_tokens":reasoning.min(completion)}})
}

// llguidance prefixes control-token vocabulary entries with an internal 0xff.
// That marker is not a byte produced by the model's textual token representation.
fn probability_token_bytes(trie: &llguidance::toktrie::TokTrie, id: u32) -> &[u8] {
    let bytes = trie.token(id);
    if trie.is_special_token(id) {
        bytes.get(1..).unwrap_or_default()
    } else {
        bytes
    }
}

fn count_reasoning_tokens(state: &AppState, ids: &[u32], reasoning: &str) -> usize {
    if reasoning.is_empty() {
        return 0;
    }
    // Match original generated bytes, rather than counting a second tokenization.
    let trie = state.grammar_factory.tok_env().tok_trie();
    let raw: Vec<u8> = ids
        .iter()
        .flat_map(|id| probability_token_bytes(trie, *id).iter().copied())
        .collect();
    let Some(start) = raw
        .windows(reasoning.len())
        .position(|w| w == reasoning.as_bytes())
    else {
        return 0;
    };
    let end = start + reasoning.len();
    let mut at = 0;
    ids.iter()
        .filter(|id| {
            let a = at;
            at += probability_token_bytes(trie, **id).len();
            a < end && at > start
        })
        .count()
}
fn empty_completion_logprobs() -> Value {
    json!({"tokens":[],"token_logprobs":[],"top_logprobs":[],"text_offset":[]})
}
#[derive(Default)]
struct VisibleLogprobs {
    raw: Vec<u8>,
    records: Vec<(usize, usize, Value, bool)>,
}
impl VisibleLogprobs {
    fn push(&mut self, state: &AppState, event: &crate::runtime::TokenEvent) {
        let Some(lp) = &event.logprobs else {
            return;
        };
        let trie = state.grammar_factory.tok_env().tok_trie();
        let bytes = probability_token_bytes(trie, event.token_id as u32);
        let start = self.raw.len();
        self.raw.extend_from_slice(bytes);
        let top: Vec<Value> = lp.top.iter().map(|&(id, logprob)| {
            let bytes = probability_token_bytes(trie, id);
            json!({"token":String::from_utf8_lossy(bytes),"logprob":logprob.max(-9999.0),"bytes":bytes})
        }).collect();
        self.records.push((start, self.raw.len(), json!({"token":String::from_utf8_lossy(bytes),"logprob":lp.logprob.max(-9999.0),"bytes":bytes,"top_logprobs":top}), false));
    }
    fn take(&mut self, visible: &str, chat: bool, echo_offset: usize) -> Value {
        let mut content = Vec::new();
        let mut offsets = Vec::new();
        if !visible.is_empty() {
            if let Some(start) = self
                .raw
                .windows(visible.len())
                .rposition(|w| w == visible.as_bytes())
            {
                let end = start + visible.len();
                for (a, b, lp, sent) in &mut self.records {
                    if !*sent && *a >= start && *b <= end && b > a {
                        content.push(lp.clone());
                        let mut offset = *a - start;
                        while !visible.is_char_boundary(offset) {
                            offset = offset.saturating_sub(1);
                        }
                        offsets.push(echo_offset + visible[..offset].chars().count());
                        *sent = true;
                    }
                }
            }
        }
        if chat {
            return json!({"content":content,"refusal":null});
        }
        let tops: Vec<Value> = content
            .iter()
            .map(|v| {
                let mut map = Map::new();
                if let Some(top) = v["top_logprobs"].as_array() {
                    for t in top {
                        if let Some(token) = t["token"].as_str() {
                            map.insert(token.to_string(), t["logprob"].clone());
                        }
                    }
                }
                if let Some(token) = v["token"].as_str() {
                    map.insert(token.to_owned(), v["logprob"].clone());
                }
                Value::Object(map)
            })
            .collect();
        json!({"tokens":content.iter().map(|v| v["token"].clone()).collect::<Vec<_>>(),
            "token_logprobs":content.iter().map(|v| v["logprob"].clone()).collect::<Vec<_>>(),
            "top_logprobs":tops,"text_offset":offsets})
    }
}

struct OpenaiChoice {
    ids: Vec<u32>,
    probabilities: VisibleLogprobs,
    cached_tokens: usize,
    reasoning_tokens: usize,
    filter: StopFilter,
    sent_reasoning: usize,
    sent_content: usize,
    sent_tools: usize,
    done: bool,
    reason: &'static str,
    /// Cumulative log-probability of the generated tokens (ranks the candidates of best_of).
    score: f64,
}
impl OpenaiChoice {
    fn new(stops: &[String]) -> Self {
        Self {
            ids: Vec::new(),
            probabilities: VisibleLogprobs::default(),
            cached_tokens: 0,
            reasoning_tokens: 0,
            filter: StopFilter::new(stops.to_vec()),
            sent_reasoning: 0,
            sent_content: 0,
            sent_tools: 0,
            done: false,
            reason: "stop",
            score: 0.0,
        }
    }
}

async fn openai_generate(
    state: Arc<AppState>,
    mut body: Value,
    prompt: Vec<i64>,
    is_chat: bool,
    reasoning: Option<ReasoningSpec>,
    images: ImageOverrides,
) -> Response {
    // a draining server takes no new inference work, on any route
    if let Some(refusal) = state.draining_response() {
        return refusal;
    }
    let empty_tool_request = json!({});
    let tool_spec = match tool_spec_for(&state, if is_chat {
        &body
    } else {
        &empty_tool_request
    }) {
        Ok(spec) => spec,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    if let Some(schema) = &tool_spec.forced_schema {
        if body.get("response_format").is_some() {
            return api_error(
                StatusCode::BAD_REQUEST,
                "required tool calls and response_format cannot be combined",
            );
        }
        body["response_format"] =
            json!({"type":"json_schema","json_schema":{"name":"tool_call","schema":schema}});
    }
    // tool_call_grammar: text or calls in the parser's format. The grammar starts with the first
    // generated token, so it is skipped when the prompt already opened a reasoning block.
    let mut native_grammar = tool_spec.native_grammar();
    if native_grammar.is_some() {
        if body.get("response_format").is_some() || body.get("grammar").is_some() {
            return api_error(
                StatusCode::BAD_REQUEST,
                "tool_call_grammar cannot be combined with response_format or grammar",
            );
        }
        if reasoning.as_ref().is_some_and(ReasoningSpec::starts_in_reasoning) {
            tracing::info!("tool_call_grammar: the prompt opens a reasoning block, generation is unconstrained");
            native_grammar = None;
        }
    }
    let n = match body.get("n") {
        None => 1,
        Some(v) => match v.as_u64().filter(|n| (1..=16).contains(n)) {
            Some(n) => n as usize,
            None => return api_error(StatusCode::BAD_REQUEST, "n must be an integer from 1 to 16"),
        },
    };
    // best_of: sample `best_of` candidates, return the `n` with the highest cumulative log-probability (not with streaming)
    let best_of = match body.get("best_of") {
        None | Some(Value::Null) => n,
        Some(v) => match v.as_u64().filter(|b| (n as u64..=16).contains(b)) {
            Some(b) => b as usize,
            None => return api_error(StatusCode::BAD_REQUEST, "best_of must be an integer from n to 16"),
        },
    };
    if best_of > n && body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return api_error(StatusCode::BAD_REQUEST, "best_of cannot be combined with stream");
    }
    let use_beam_search = match body.get("use_beam_search") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(_) => return api_error(StatusCode::BAD_REQUEST, "use_beam_search must be a boolean"),
    };
    let mut beam_rules: Option<(f64, crate::beam::EarlyStopping)> = None;
    if use_beam_search {
        // best_of is the beam width (vLLM), n the number of hypotheses returned
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return api_error(
                StatusCode::BAD_REQUEST,
                "use_beam_search cannot be combined with stream: the beams are pruned until the search ends, so no token is final before then",
            );
        }
        if best_of > crate::beam::MAX_BEAMS {
            return api_error(
                StatusCode::BAD_REQUEST,
                &format!("use_beam_search accepts at most {} beams (best_of, or n when best_of is absent)", crate::beam::MAX_BEAMS),
            );
        }
        if !images.is_empty()
            || tool_spec.forced_schema.is_some()
            || native_grammar.is_some()
            || body.get("response_format").is_some_and(|v| !v.is_null())
            || body.get("grammar").is_some_and(|v| !v.is_null())
            || body.get("grammar_type").is_some_and(|v| !v.is_null())
        {
            return api_error(
                StatusCode::BAD_REQUEST,
                "use_beam_search cannot be combined with grammar-constrained generation (response_format, grammar, forced tool calls) or images",
            );
        }
        let length_penalty = match body.get("length_penalty") {
            None | Some(Value::Null) => 1.0,
            Some(value) => match value.as_f64().filter(|v| v.is_finite()) {
                Some(value) => value,
                None => return api_error(StatusCode::BAD_REQUEST, "length_penalty must be a finite number"),
            },
        };
        let early_stopping = match body.get("early_stopping") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => crate::beam::EarlyStopping::Heuristic,
            Some(Value::Bool(true)) => crate::beam::EarlyStopping::Always,
            Some(Value::String(rule)) if rule == "never" => crate::beam::EarlyStopping::Never,
            Some(_) => return api_error(StatusCode::BAD_REQUEST, "early_stopping must be true, false or \"never\""),
        };
        beam_rules = Some((length_penalty, early_stopping));
    }
    if best_of > 1 && !use_beam_search
        && (!state.shared_choices
            || !images.is_empty()
            || tool_spec.forced_schema.is_some()
            || native_grammar.is_some()
            || body.get("response_format").is_some()
            || body.get("grammar").is_some()
            || body.get("grammar_type").is_some())
    {
        return api_error(StatusCode::BAD_REQUEST, "n > 1 and best_of need a stateless model on a backend with KV block copy, no draft/MTP engine, and unconstrained text generation");
    }
    let studio_cfg = state.studio_config.read().unwrap().clone();
    let temperature = body
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_temperature.unwrap_or(OPENAI_DEFAULT_TEMPERATURE));
    if !temperature.is_finite() || temperature < 0.0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "temperature must be a non-negative finite number",
        );
    }
    let top_p = body
        .get("top_p")
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_top_p.unwrap_or(1.0));
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "top_p must be a finite number between 0.0 and 1.0",
        );
    }
    let top_k = body
        .get("top_k")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(studio_cfg.override_top_k.unwrap_or(0));
    let repetition_penalty = body
        .get("repetition_penalty")
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(studio_cfg.sampling.repetition_penalty);
    let sampling = crate::sampling::SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        ..Default::default()
    };
    let mut sampling = match request_sampling(&state, &body, sampling) {
        Ok(params) => params,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    if let Err(error) =
        openai_sampling_controls(&body, is_chat, state.tokenizer.vocab_size(), &mut sampling)
    {
        return api_error(StatusCode::BAD_REQUEST, &error);
    }
    sampling.choices = if use_beam_search { 1 } else { best_of };
    let min_tokens = sampling.min_tokens;
    let logprobs_requested = sampling.logprobs.is_some();
    if use_beam_search && logprobs_requested {
        return api_error(StatusCode::BAD_REQUEST, "logprobs are not available with use_beam_search");
    }
    if best_of > n && !use_beam_search && sampling.logprobs.is_none() {
        // the ranking needs the log-probability of every sampled token
        sampling.logprobs = Some(0);
    }
    let include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let echo = !is_chat && body["echo"].as_bool().unwrap_or(false);
    let echo_text = if echo {
        if let Some(text) = body["prompt"].as_str() {
            text.to_owned()
        } else {
            match state.tokenizer.decode(
                &prompt.iter().map(|id| *id as u32).collect::<Vec<_>>(),
                false,
            ) {
                Ok(text) => text,
                Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
            }
        }
    } else {
        String::new()
    };
    let stop_strings = match parse_stop_strings(body.get("stop")) {
        Ok(stops) => stops,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    let max_tokens = match body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        // OpenAI and vLLM: without a limit the reply may use the rest of the context window
        None | Some(Value::Null) => {
            let remaining = state.max_context.saturating_sub(prompt.len());
            if remaining == 0 {
                return context_length_error(state.max_context, prompt.len(), 0);
            }
            remaining
        }
        Some(value) => match value.as_u64().and_then(|n| usize::try_from(n).ok()) {
            Some(n) => n,
            None => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "max_completion_tokens/max_tokens must be a non-negative integer",
                )
            }
        },
    };
    if sampling.min_tokens > max_tokens {
        return api_error(
            StatusCode::BAD_REQUEST,
            "min_tokens must not exceed max_completion_tokens/max_tokens",
        );
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let model = match state.resolve_model(&body) {
        Ok(model) => model,
        Err(response) => return response,
    };
    let service_tier = body
        .get("service_tier")
        .and_then(Value::as_str)
        .map_or("default", |tier| if tier == "auto" { "default" } else { tier })
        .to_owned();
    let id = state.next_request.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("req_{id:016x}");
    let created = unix_now();
    let span = crate::telemetry::SpanBuilder::new(if is_chat { "chat.completions" } else { "completions" })
        .text("model", &model)
        .int("prompt_tokens", prompt.len() as i64)
        .attr("stream", crate::telemetry::Attr::Bool(stream));
    // a deterministic request may have been answered before, or may be running right now: check both before any work starts
    let mut coal_lead: Option<crate::cache::coalescer::Leader> = None;
    // only a *budgeted* reasoning request changes the text; the parser alone is part of the model identity
    let reasoning_budgeted = reasoning.as_ref().is_some_and(|spec| spec.budget.is_some());
    if !stream && n == 1 && best_of == 1 && images.is_empty() && !reasoning_budgeted {
        if let Some(cached) = cache_gate::lookup(&state, &body, &prompt, is_chat, &request_id, created, &model) {
            let hit = cached.get("usage").and_then(|u| u.get("completion_tokens")).and_then(Value::as_i64).unwrap_or(0);
            state.tracer.record(span.text("cache", "hit").int("completion_tokens", hit).finish());
            return Json(cached).into_response();
        }
        if let Some(key) = cache_gate::key_of(&state, &body, &prompt, is_chat) {
            match state.coalesce.join(key) {
                crate::cache::coalescer::Role::Follower(mut rx) => {
                    // somebody is already computing exactly this: wait for the answer instead of a second model run
                    return match rx.recv().await {
                        Ok(Some(payload)) => Json(payload).into_response(),
                        _ => api_error(StatusCode::SERVICE_UNAVAILABLE, "the request this one joined did not finish; try again"),
                    };
                }
                crate::cache::coalescer::Role::Leader(leader) => coal_lead = Some(leader),
            }
        }
    }
    if max_tokens == 0 {
        return empty_openai_response(
            (include_usage, logprobs_requested, &echo_text, n),
            &request_id,
            created,
            &model,
            is_chat,
            stream,
            prompt.len(),
        );
    }
    let constraint = match native_grammar {
        Some(grammar) => crate::structured::matcher(&state.grammar_factory, grammar).map(Some),
        None => request_constraint(&state, &body),
    };
    let constraint = match constraint {
        Ok(c) => c,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    // thinking budget: only for one unconstrained, text-only choice
    let budget_plan = reasoning
        .as_ref()
        .and_then(|spec| spec.budget.map(|budget| (spec.clone(), budget)))
        .filter(|_| best_of == 1 && !use_beam_search && constraint.is_none() && images.is_empty());
    let budget_control = budget_plan.map(|(spec, budget)| BudgetControl {
        state: state.clone(),
        request_id: request_id.clone(),
        prompt: prompt.clone(),
        sampling: sampling.clone(),
        max_tokens,
        spec,
        machine: crate::thinking::BudgetMachine::new(budget),
        ids: Vec::new(),
    });
    let beam = beam_rules.map(|(length_penalty, early_stopping)| {
        let mut params = crate::beam::BeamParams::new(best_of, n);
        params.length_penalty = length_penalty;
        params.early_stopping = early_stopping;
        if stop_strings.iter().any(|stop| !stop.is_empty()) {
            // a hypothesis ends where its text first contains a stop string, the same rule the reply is cut by afterwards
            let tokenizer = state.tokenizer.clone();
            let reasoning = reasoning.clone();
            let stops = stop_strings.clone();
            params.stop_check = Some(Arc::new(move |tokens: &[i64]| {
                let ids: Vec<u32> = tokens.iter().map(|id| *id as u32).collect();
                decode_chat_text(&tokenizer, &ids, reasoning.as_ref())
                    .is_ok_and(|text| StopFilter::new(stops.clone()).update(&text, false).matched)
            }));
        }
        params
    });
    let rx = match submit_request(
        &state,
        &request_id,
        prompt.clone(),
        max_tokens,
        sampling,
        constraint,
        images,
        cache_salt_of(&body),
        beam,
    ) {
        Ok(rx) => rx,
        Err(response) => return response,
    };
    let mut rx = ReasoningRx::new(rx, budget_control);
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    if stream {
        let state2 = state.clone();
        let reasoning = reasoning.clone();
        let stream_body = async_stream::stream! {
            let mut choice_states = (0..n).map(|_| OpenaiChoice::new(&stop_strings)).collect::<Vec<_>>();
            if echo {
              for choice_index in 0..n {
                let mut choice = json!({"index":choice_index,"text":echo_text,"finish_reason":null,"logprobs":null});
                if logprobs_requested { choice["logprobs"] = empty_completion_logprobs(); }
                yield Ok::<Event, Infallible>(Event::default().data(json!({"id":request_id,"object":"text_completion","created":created,"model":model,"choices":[choice],"usage":null}).to_string()));
              }
            }
            while let Some(message) = rx.recv().await {
                match message {
                    Ok(event) => {
                        let choice_index = event.choice_index;
                        let Some(current) = choice_states.get_mut(choice_index) else {
                            yield Ok::<Event, Infallible>(Event::default().event("error").data(json!({"error":"invalid runtime choice index"}).to_string())); return;
                        };
                        if current.done { continue; }
                        current.cached_tokens = current.cached_tokens.max(event.cached_prompt_tokens);
                        if logprobs_requested { current.probabilities.push(&state2, &event); }
                        current.ids.push(event.token_id as u32);
                        state2.metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);
                        let full = match decode_chat_text(&state2.tokenizer, &current.ids, reasoning.as_ref()) {
                            Ok(full) => full,
                            Err(error) => {
                                yield Ok::<Event, Infallible>(Event::default().event("error").data(json!({"error":error}).to_string()));
                                return;
                            }
                        };
                        let update = current.filter.update_with_minimum(&full, event.finished, current.ids.len() >= min_tokens);
                        let held_text = hold_incomplete_utf8(current.filter.text(), update.matched || event.finished);
                        let tool_text = reasoning.as_ref().map(|spec| spec.split_views(held_text, update.matched || event.finished).content).unwrap_or_else(|| held_text.to_string());
                        let (visible_content, tool_calls) = match tool_spec.parse_reply(&tool_text, update.matched || event.finished, &format!("{request_id}_{choice_index}")) {
                            Ok(parts) => parts,
                            Err(error) => { yield Ok::<Event,Infallible>(Event::default().event("error").data(json!({"error":error}).to_string())); return; }
                        };
                        if let Some(spec) = reasoning.as_ref() {
                            let parts = spec.split_views(held_text, update.matched || event.finished);
                            current.reasoning_tokens = count_reasoning_tokens(&state2, &current.ids, &parts.reasoning);
                        }
                        if let (true, Some(spec)) = (is_chat, reasoning.as_ref()) {
                            let parts = spec.split_views(held_text, update.matched || event.finished);
                            if parts.reasoning.len() > current.sent_reasoning {
                                let delta = parts.reasoning.get(current.sent_reasoning..).unwrap_or("").to_owned();
                                current.sent_reasoning = parts.reasoning.len();
                                let choice = json!({"index":choice_index,"delta":{"reasoning_content":delta},"finish_reason":null,"logprobs":null});
                                let obj = json!({"id":request_id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[choice],"usage":null});
                                yield Ok::<Event, Infallible>(Event::default().data(obj.to_string()));
                            }
                            if visible_content.len() > current.sent_content {
                                let delta = visible_content.get(current.sent_content..).unwrap_or("").to_owned();
                                current.sent_content = visible_content.len();
                                let mut choice = json!({"index":choice_index,"delta":{"content":delta},"finish_reason":null,"logprobs":null});
                                if logprobs_requested { choice["logprobs"] = current.probabilities.take(&visible_content, true, echo_text.chars().count()); }
                                let obj = json!({"id":request_id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[choice],"usage":null});
                                yield Ok::<Event, Infallible>(Event::default().data(obj.to_string()));
                            }
                        } else if !is_chat && !update.delta.is_empty() || is_chat && visible_content.len() > current.sent_content {
                            let delta = if is_chat { let delta = visible_content.get(current.sent_content..).unwrap_or("").to_owned(); current.sent_content = visible_content.len(); delta } else { update.delta.clone() };
                            let mut choice = if is_chat {
                                json!({"index":choice_index,"delta":{"content":delta},"finish_reason":null,"logprobs":null})
                            } else {
                                json!({"index":choice_index,"text":update.delta,"finish_reason":null,"logprobs":null})
                            };
                            if logprobs_requested { choice["logprobs"] = current.probabilities.take(if is_chat { &visible_content } else { held_text }, is_chat, echo_text.chars().count()); }
                            let obj = json!({"id":request_id,"object":if is_chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":model,"choices":[choice],"usage":null});
                            yield Ok::<Event, Infallible>(Event::default().data(obj.to_string()));
                        }
                        for (index, call) in tool_calls.iter().enumerate().skip(current.sent_tools) {
                            let mut call = call.clone(); call["index"] = json!(index);
                            let obj = json!({"id":request_id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":choice_index,"delta":{"tool_calls":[call]},"finish_reason":null,"logprobs":null}],"usage":null});
                            yield Ok::<Event,Infallible>(Event::default().data(obj.to_string()));
                        }
                        current.sent_tools = current.sent_tools.max(tool_calls.len());
                        if update.matched || event.finished {
                            let reason = if current.sent_tools > 0 { "tool_calls" } else if update.matched { "stop" } else { event.finish_reason.unwrap_or("stop") };
                            let choice = if is_chat {
                                json!({"index":choice_index,"delta":{},"finish_reason":reason,"logprobs":null})
                            } else {
                                json!({"index":choice_index,"text":"","finish_reason":reason,"logprobs":null})
                            };
                            let final_obj = json!({"id":request_id,"object":if is_chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":model,"service_tier":service_tier,"choices":[choice],"usage":null,"timings":rx.timings(prompt.len())});
                            yield Ok::<Event, Infallible>(Event::default().data(final_obj.to_string()));
                            current.done = true;
                            if choice_states.iter().all(|choice| choice.done) {
                                if include_usage {
                                    let completion = choice_states.iter().map(|c| c.ids.len()).sum();
                                    let cached = choice_states.iter().map(|c| c.cached_tokens).max().unwrap_or(0);
                                    let reasoning_count = choice_states.iter().map(|c| c.reasoning_tokens).sum();
                                    let usage_chunk = json!({"id":request_id,"object":if is_chat {"chat.completion.chunk"} else {"text_completion"},"created":created,"model":model,"choices":[],"usage":openai_usage(prompt.len(), completion, cached, reasoning_count)});
                                    yield Ok::<Event, Infallible>(Event::default().data(usage_chunk.to_string()));
                                }
                                yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        yield Ok::<Event, Infallible>(Event::default().event("error").data(json!({"error":error}).to_string()));
                        return;
                    }
                }
            }
        };
        Sse::new(stream_body)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut choice_states = (0..if use_beam_search { n } else { best_of })
            .map(|_| OpenaiChoice::new(&stop_strings))
            .collect::<Vec<_>>();
        while let Some(message) = rx.recv().await {
            let event = match message {
                Ok(event) => event,
                Err(error) => return generation_error(&error),
            };
            let Some(current) = choice_states.get_mut(event.choice_index) else {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "invalid runtime choice index",
                );
            };
            if current.done {
                continue;
            }
            current.cached_tokens = current.cached_tokens.max(event.cached_prompt_tokens);
            if logprobs_requested {
                current.probabilities.push(&state, &event);
            }
            current.ids.push(event.token_id as u32);
            current.score += event.logprobs.as_ref().map_or(0.0, |l| l.logprob.max(-9999.0));
            state
                .metrics
                .generated_tokens
                .fetch_add(1, Ordering::Relaxed);
            let visible_ids = &current.ids;
            let full = match decode_chat_text(&state.tokenizer, visible_ids, reasoning.as_ref()) {
                Ok(full) => full,
                Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
            };
            let update = current.filter.update_with_minimum(&full, event.finished, current.ids.len() >= min_tokens);
            if update.matched || event.finished {
                current.reason = if update.matched {
                    "stop"
                } else {
                    event.finish_reason.unwrap_or("stop")
                };
                current.done = true;
            }
            if choice_states.iter().all(|c| c.done) {
                break;
            }
        }
        if use_beam_search {
            // fewer than n hypotheses can exist (a narrow vocabulary); the ranks that exist are returned
            choice_states.retain(|c| c.done);
        }
        if choice_states.iter().any(|c| !c.done) || choice_states.is_empty() {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation ended before all choices completed",
            );
        }
        let completion = choice_states.iter().map(|c| c.ids.len()).sum();
        let cached = choice_states
            .iter()
            .map(|c| c.cached_tokens)
            .max()
            .unwrap_or(0);
        if best_of > n && !use_beam_search {
            // keep the candidates with the highest cumulative log-probability, best first
            choice_states.sort_by(|a, b| b.score.total_cmp(&a.score));
            choice_states.truncate(n);
        }
        let mut reasoning_count = 0;
        let mut choices = Vec::with_capacity(n);
        for (index, current) in choice_states.iter_mut().enumerate() {
            let text = current.filter.text();
            let parts = reasoning.as_ref().map(|spec| spec.split_views(text, true));
            let reasoning_text = parts.as_ref().map(|p| p.reasoning.as_str()).unwrap_or("");
            reasoning_count += count_reasoning_tokens(&state, &current.ids, reasoning_text);
            let mut choice = if is_chat {
                let final_content = parts.as_ref().map(|p| p.content.as_str()).unwrap_or(text);
                let (content, calls) = match tool_spec.parse_reply(
                    final_content,
                    true,
                    &format!("{request_id}_{index}"),
                ) {
                    Ok(parts) => parts,
                    Err(error) => return api_error(StatusCode::UNPROCESSABLE_ENTITY, &error),
                };
                let mut msg = json!({"role":"assistant","content":content});
                if !calls.is_empty() {
                    if content.is_empty() {
                        msg["content"] = Value::Null;
                    }
                    msg["tool_calls"] = json!(calls);
                    current.reason = "tool_calls";
                }
                if parts.as_ref().is_some_and(|p| p.reasoning_found) {
                    msg["reasoning_content"] = json!(reasoning_text);
                }
                json!({"index":index,"message":msg,"finish_reason":current.reason,"logprobs":null})
            } else {
                json!({"index":index,"text":format!("{echo_text}{text}"),"finish_reason":current.reason,"logprobs":null})
            };
            if logprobs_requested {
                let visible = if is_chat {
                    choice["message"]["content"].as_str().unwrap_or("")
                } else {
                    text
                };
                choice["logprobs"] =
                    current
                        .probabilities
                        .take(visible, is_chat, echo_text.chars().count());
            }
            choices.push(choice);
        }
        let payload = json!({"id":request_id,"object":if is_chat {"chat.completion"} else {"text_completion"},"created":created,"model":model,"service_tier":service_tier,"choices":choices,
            "usage":openai_usage(prompt.len(), completion, cached, reasoning_count),"timings":rx.timings(prompt.len())});
        // remember the answer for an identical, deterministic request
        if !reasoning_budgeted {
            let text: String = choice_states.iter().map(|c| c.filter.text()).collect();
            let tokens: Vec<u32> = choice_states.iter().flat_map(|c| c.ids.iter().copied()).collect();
            let finish = choice_states.first().map(|c| c.reason).unwrap_or("stop");
            let logprobs = choices.first().and_then(|choice| choice.get("logprobs")).cloned().filter(|v| !v.is_null());
            cache_gate::store(
                &state,
                &body,
                &prompt,
                is_chat,
                cache_gate::Stored { text: &text, tokens, finish_reason: finish, logprobs },
            );
        }
        // wake everyone who joined this request
        if let Some(leader) = coal_lead.take() {
            leader.publish(payload.clone());
        }
        // one span per finished answer: the streaming path reports its own end
        state.tracer.record(span.int("completion_tokens", completion as i64).finish());
        Json(payload).into_response()
    }
}

/// See [`crate::chat_template::normalize`].
fn normalize_chat_template(template: &str) -> String {
    crate::chat_template::normalize(template)
}

/// Instructions one template render may execute. Chat templates come from model repositories; the limit keeps a
/// looping template from stalling a request thread (a normal template needs thousands, a long conversation millions).
const TEMPLATE_FUEL: u64 = 50_000_000;

/// Renders a chat template over a message list with the server's Jinja environment, as the fuzz target `chat_template`
/// does.
#[doc(hidden)]
/// Validates a chat template before rendering to reject malformed or adversarial templates
/// that could trigger infinite recursion or stack overflows.
pub fn validate_chat_template(template: &str) -> Result<(), String> {
    let mut depth = 0usize;
    for c in template.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                if depth > 64 {
                    return Err("chat template exceeds maximum nesting depth of 64".into());
                }
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }

    let mut rest = template;
    while let Some(start) = rest.find("{%") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("%}") else { break; };
        let tag = after[..end].trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        if let Some(set_stmt) = tag.strip_prefix("set ") {
            if let Some((lhs, rhs)) = set_stmt.split_once('=') {
                let lhs = lhs.trim();
                let rhs = rhs.trim();
                if let Some((obj, _)) = lhs.split_once(['.', '[']) {
                    let obj = obj.trim();
                    if !obj.is_empty() {
                        // `set ns.n = ns.n + 1` reads an attribute of the object and is the usual way to count in a template; what
                        // makes a cycle is the object itself as a value (`set ns.n = ns`, `set ns.n = [ns]`)
                        let is_ident_char = |c: char| c.is_ascii_alphanumeric() || c == '_';
                        let chars: Vec<char> = rhs.chars().collect();
                        let mut i = 0;
                        let mut cyclic = false;
                        while i < chars.len() {
                            if is_ident_char(chars[i]) {
                                let start = i;
                                while i < chars.len() && is_ident_char(chars[i]) {
                                    i += 1;
                                }
                                let word: String = chars[start..i].iter().collect();
                                let after_word = chars[i..].iter().find(|c| !c.is_whitespace());
                                let member_of_other = start > 0 && chars[..start].iter().rev().find(|c| !c.is_whitespace()) == Some(&'.');
                                if word == obj && !member_of_other && !matches!(after_word, Some('.') | Some('[')) {
                                    cyclic = true;
                                }
                            } else {
                                i += 1;
                            }
                        }
                        if cyclic {
                            return Err("chat template contains cyclic self-referential assignment".into());
                        }                    }
                }
            }
        }
        rest = &after[end + 2..];
    }

    Ok(())
}

pub fn render_template_for_fuzz(template: &str, messages: &Value) -> Result<String, String> {
    validate_chat_template(template)?;
    chat_environment()
        .render_str(&normalize_chat_template(template), json!({"messages": messages, "add_generation_prompt": true}))
        .map_err(|e| e.to_string())
}

/// See [`crate::chat_template::environment`]; the fuel limit bounds the work of one render.
fn chat_environment() -> Environment<'static> {
    let mut env = crate::chat_template::environment();
    env.set_fuel(Some(TEMPLATE_FUEL));
    env
}

/// A token can end in the middle of a multi-byte character (emoji). The decoder shows that as
/// U+FFFD until the next token completes it; streamed text must only ever grow, so trailing
/// replacement characters are held back while generation is still running.
fn hold_incomplete_utf8(text: &str, done: bool) -> &str {
    if done {
        text
    } else {
        text.trim_end_matches('\u{FFFD}')
    }
}

fn decode_chat_text(
    tokenizer: &TokenizerBundle,
    ids: &[u32],
    spec: Option<&ReasoningSpec>,
) -> Result<String, String> {
    match spec {
        Some(spec) if !spec.keep.is_empty() => tokenizer.decode_keeping(ids, &spec.keep),
        _ => tokenizer.decode(ids, true),
    }
}

/// Where chat output switches from reasoning to the answer, plus the marker tokens the chat
/// decoder keeps visible. The format lives in a [`ReasoningParser`] (see `reasoning.rs`);
/// `--reasoning-parser` or the chat template chooses it, the prompt decides whether the
/// thinking block is already open (templates such as K2-Horizon, DeepSeek-R1 or QwQ end the
/// prompt with `<ifm|think>\n` / `<think>\n`).
#[derive(Clone)]
struct ReasoningSpec {
    parser: Arc<dyn ReasoningParser>,
    /// Marker tokens that are special tokens: kept as literal text when decoding. Holds the
    /// reasoning tags and, when the request declares tools, the tool-call markers.
    keep: Vec<(u32, String)>,
    /// Reasoning tokens after which the server ends the reasoning block itself (`thinking_budget`).
    budget: Option<usize>,
}

impl ReasoningSpec {
    fn for_request(state: &AppState, prompt: &str, tools: &crate::tools::ToolSpec) -> Self {
        let parser = state.reasoning_family.parser(prompt);
        let mut literals = parser.markers();
        literals.extend(tools.markers().into_iter().map(str::to_owned));
        Self {
            keep: keep_ids(state, literals),
            parser,
            budget: None,
        }
    }

    /// `true` when the prompt ended inside an open reasoning block (the first generated text is reasoning).
    fn starts_in_reasoning(&self) -> bool {
        !self.parser.split("x", false).reasoning.is_empty()
    }

    #[cfg(test)]
    fn from_prompt(prompt: &str) -> Self {
        Self {
            parser: Family::Think.parser(prompt),
            keep: Vec::new(),
            budget: None,
        }
    }

    /// Splits text produced so far. While generation is running a trailing partial tag is held
    /// back, and outer whitespace of each part is trimmed so streamed deltas never need undoing.
    fn split_views(&self, text: &str, complete: bool) -> ReasoningParts {
        self.parser.split(text, complete)
    }
}

/// The token ids of the given literals, for the literals that are special tokens.
fn keep_ids<S: AsRef<str>>(state: &AppState, literals: impl IntoIterator<Item = S>) -> Vec<(u32, String)> {
    let mut keep: Vec<(u32, String)> = Vec::new();
    for literal in literals {
        let literal = literal.as_ref();
        if let Some(id) = state.tokenizer.special_token_id(literal) {
            if !keep.iter().any(|(known, _)| *known == id) {
                keep.push((id, literal.to_owned()));
            }
        }
    }
    keep
}

/// The request's tools, read with the tool-call format of the loaded model.
fn tool_spec_for(state: &AppState, body: &Value) -> Result<crate::tools::ToolSpec, String> {
    crate::tools::ToolSpec::from_request(body).map(|spec| spec.with_parser(state.tool_parser.clone()))
}
async fn sglang_model_info(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "model_path": state.model_name,
        "model_type": state.model_arch,
        "max_total_num_tokens": state.max_context,
        "is_generation": true,
    }))
}

async fn sglang_server_info(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "model_path": state.model_name,
        "model_type": state.model_arch,
        "version": "0.4.3-fllm-sglang",
    }))
}

/// SGLang and Text Generation Inference share `POST /generate`: TGI sends `inputs`, SGLang
/// sends `prompt`, `text` or `input_ids`, so the body decides which handler runs.
async fn sglang_generate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    if body.get("inputs").is_some() {
        return api::tgi::generate(State(state), Json(body)).await;
    }
    let prompt_text = body
        .get("prompt")
        .or_else(|| body.get("text"))
        .and_then(Value::as_str);
    let Some(prompt_text) = prompt_text else {
        return api_error(StatusCode::BAD_REQUEST, "prompt or text is required");
    };
    let prompt_ids = match state.tokenizer.encode(prompt_text) {
        Ok(ids) => ids.into_iter().map(i64::from).collect::<Vec<_>>(),
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    let sampling_params = body.get("sampling_params").cloned().unwrap_or(json!({}));
    let max_tokens = sampling_params
        .get("max_new_tokens")
        .or_else(|| body.get("max_new_tokens"))
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(128) as usize;
    let stop = match parse_stop_strings(sampling_params.get("stop").or_else(|| body.get("stop"))) {
        Ok(stops) => stops,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    let studio_cfg = state.studio_config.read().unwrap().clone();
    let temperature = sampling_params
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_temperature.unwrap_or(OPENAI_DEFAULT_TEMPERATURE));
    let top_p = sampling_params
        .get("top_p")
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_top_p.unwrap_or(1.0));
    let top_k = sampling_params
        .get("top_k")
        .and_then(Value::as_u64)
        .map(|k| k as usize)
        .unwrap_or(studio_cfg.override_top_k.unwrap_or(0));
    let repetition_penalty = sampling_params
        .get("repetition_penalty")
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(1.0);
    let sampling = crate::sampling::SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        ..Default::default()
    };
    let sampling = match request_sampling(&state, &sampling_params, sampling) {
        Ok(params) => params,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error),
    };
    let id = state.next_request.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("req_{id:016x}");
    let mut rx = match submit_constrained(
        &state,
        &request_id,
        prompt_ids.clone(),
        max_tokens,
        sampling,
        match request_constraint(&state, &body) {
            Ok(c) => c,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        },
        ImageOverrides::default(),
        cache_salt_of(&body),
    ) {
        Ok(rx) => rx,
        Err(response) => return response,
    };
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    let mut ids = Vec::<u32>::new();
    let mut filter = StopFilter::new(stop);
    let mut finish_reason = "length";
    while let Some(message) = rx.recv().await {
        match message {
            Ok(event) => {
                ids.push(event.token_id as u32);
                state
                    .metrics
                    .generated_tokens
                    .fetch_add(1, Ordering::Relaxed);
                let full = match state.tokenizer.decode(&ids, true) {
                    Ok(full) => full,
                    Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
                };
                let update = filter.update(&full, event.finished);
                if update.matched {
                    finish_reason = "stop";
                    break;
                }
                if event.finished {
                    finish_reason = event.finish_reason.unwrap_or("stop");
                    break;
                }
            }
            Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        }
    }
    let text = filter.text();
    Json(json!({
        "text": text,
        "meta_info": {
            "id": request_id,
            "finish_reason": {"type": finish_reason},
            "prompt_tokens": prompt_ids.len(),
            "completion_tokens": ids.len(),
        }
    }))
    .into_response()
}

async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(response) = api::ollama_extra::resolve_model(&state, &body) {
        return response;
    }
    if let Err(response) = api::ollama_extra::note_keep_alive(&state, &body) {
        return response;
    }
    // an empty prompt only loads (or, with keep_alive 0, unloads) the model
    if body.get("prompt").and_then(Value::as_str) == Some("") && body.get("images").is_none() {
        return api::ollama_extra::load_response(&state, &body, false);
    }
    // Ollama can steer the raw prompt with a system message or a FIM suffix; fLLM answers raw
    // completions, so the system text is prefixed and the suffix points at /infill.
    if body
        .get("suffix")
        .and_then(Value::as_str)
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return api::ollama_extra::error(
            StatusCode::BAD_REQUEST,
            "suffix is only supported by POST /infill",
        );
    }
    if let Some(system) = body.get("system").and_then(Value::as_str) {
        if !system.is_empty() {
            let Some(prompt) = body.get("prompt").and_then(Value::as_str).map(str::to_owned) else {
                return api::ollama_extra::error(StatusCode::BAD_REQUEST, "system requires a text prompt");
            };
            body["prompt"] = json!(format!("{system}\n\n{prompt}"));
        }
    }
    let prompt_val = match body.get("prompt") {
        Some(v) => v,
        None => return api::ollama_extra::error(StatusCode::BAD_REQUEST, "prompt is required"),
    };
    let prompt = match prompt_val {
        Value::String(s) => match state.tokenizer.encode(s) {
            Ok(ids) => ids.into_iter().map(i64::from).collect(),
            Err(e) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &e),
        },
        Value::Array(arr) => {
            let mut ids = Vec::with_capacity(arr.len());
            for item in arr {
                match item.as_i64() {
                    Some(id) => ids.push(id),
                    None => {
                        return api::ollama_extra::error(
                            StatusCode::BAD_REQUEST,
                            "prompt array elements must be integer token IDs",
                        )
                    }
                }
            }
            if ids.is_empty() {
                return api::ollama_extra::error(
                    StatusCode::BAD_REQUEST,
                    "prompt token array cannot be empty",
                );
            }
            ids
        }
        _ => {
            return api::ollama_extra::error(
                StatusCode::BAD_REQUEST,
                "prompt must be a string or array of integer token IDs",
            )
        }
    };
    let images = match body.get("images").and_then(Value::as_array) {
        Some(images) if !images.is_empty() => {
            let Some(prompt) = prompt_val.as_str() else {
                return api::ollama_extra::error(
                    StatusCode::BAD_REQUEST,
                    "token ID prompts cannot be combined with images",
                );
            };
            let chat = [json!({"role":"user","content":prompt,"images":images})];
            let (prompt, overrides) = match ollama_vision_prompt(&state, &chat, None, None).await {
                Ok(vision) => vision,
                Err(error) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &error),
            };
            return ollama_generate_inner(state, body, prompt, false, overrides).await;
        }
        _ => ImageOverrides::default(),
    };
    ollama_generate_inner(state, body, prompt, false, images).await
}

async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<Value>,
) -> Response {
    if let Err(response) = api::ollama_extra::resolve_model(&state, &body) {
        return response;
    }
    if let Err(response) = api::ollama_extra::note_keep_alive(&state, &body) {
        return response;
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return api::ollama_extra::error(
            StatusCode::BAD_REQUEST,
            "messages must be a non-empty array",
        );
    };
    if messages.is_empty() {
        return api::ollama_extra::load_response(&state, &body, true);
    }
    let tool_spec = match tool_spec_for(&state, &body) {
        Ok(spec) => spec,
        Err(error) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &error),
    };
    let messages = Value::Array(messages.clone());
    if let Some(schema) = tool_spec.forced_schema {
        if body.get("format").is_some() {
            return api::ollama_extra::error(
                StatusCode::BAD_REQUEST,
                "required tool calls cannot be combined with format",
            );
        }
        body["format"] = schema;
    }
    let tools = if tool_spec.disabled {
        None
    } else {
        body.get("tools").cloned()
    };
    let kwargs = api::ollama_extra::thinking_kwargs(&body);
    let list = messages.as_array().map_or(&[][..], Vec::as_slice);
    let (prompt, images) = match ollama_vision_prompt(&state, list, tools, kwargs).await {
        Ok(chat) => chat,
        Err(e) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &e),
    };
    ollama_generate_inner(state, body, prompt, true, images).await
}

/// Renders Ollama chat messages into prompt tokens. Messages that carry base64 images are
/// rewritten into the content parts the native vision encoder understands; requests without
/// images stay on the ordinary chat-template path.
async fn ollama_vision_prompt(
    state: &AppState,
    messages: &[Value],
    tools: Option<Value>,
    kwargs: Option<Map<String, Value>>,
) -> Result<(Vec<i64>, ImageOverrides), String> {
    if !messages.iter().any(|m| {
        m.get("images")
            .and_then(Value::as_array)
            .is_some_and(|images| !images.is_empty())
    }) {
        let prompt = render_messages(
            state,
            Value::Array(messages.to_vec()),
            tools,
            kwargs,
        )?;
        return Ok((state.tokenizer.encode(&prompt)?.into_iter().map(i64::from).collect(), ImageOverrides::default()));
    }
    if state.vision.is_none() {
        return Err("images require a vision model (a GGUF with an imported vision tower, or --mmproj)".into());
    }
    let mut normalized = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("missing chat role")?;
        let mut parts = Vec::new();
        match message.get("content") {
            Some(Value::String(text)) if !text.is_empty() => {
                parts.push(json!({"type":"text","text":text}))
            }
            None | Some(Value::Null) => {}
            Some(_) => return Err("image chat content must be a plain string".into()),
        }
        for image in message
            .get("images")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
        {
            let encoded = image
                .as_str()
                .ok_or("images must be base64 encoded strings")?;
            if encoded.is_empty() {
                return Err("images must be base64 encoded strings".into());
            }
            parts.push(json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{encoded}")}}));
        }
        if parts.is_empty() {
            parts.push(json!({"type":"text","text":""}));
        }
        normalized.push(json!({"role":role,"content":parts}));
    }
    if tools.is_some() {
        return Err("tool use is not supported together with images".into());
    }
    let (_prompt, ids, overrides) = media::prepare_media_chat(state, &normalized).await?;
    Ok((ids, overrides))
}

async fn ollama_generate_inner(
    state: Arc<AppState>,
    body: Value,
    prompt: Vec<i64>,
    is_chat: bool,
    images: ImageOverrides,
) -> Response {
    let tool_spec = match tool_spec_for(&state, &body) {
        Ok(spec) => spec,
        Err(error) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &error),
    };
    let tool_keep = keep_ids(&state, tool_spec.markers());
    let options = body.get("options").and_then(Value::as_object);
    if body.get("options").is_some_and(|v| !v.is_object()) {
        return api::ollama_extra::error(StatusCode::BAD_REQUEST, "options must be an object");
    }
    let stop_strings = match parse_stop_strings(options.and_then(|opts| opts.get("stop"))) {
        Ok(stops) => stops,
        Err(error) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &error),
    };
    let studio_cfg = state.studio_config.read().unwrap().clone();
    let temperature = options
        .and_then(|o| o.get("temperature"))
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_temperature.unwrap_or(OLLAMA_DEFAULT_TEMPERATURE));
    if !temperature.is_finite() || temperature < 0.0 {
        return api::ollama_extra::error(
            StatusCode::BAD_REQUEST,
            "temperature must be a non-negative finite number",
        );
    }
    let top_p = options
        .and_then(|o| o.get("top_p"))
        .and_then(Value::as_f64)
        .unwrap_or(studio_cfg.override_top_p.unwrap_or(OLLAMA_DEFAULT_TOP_P));
    let top_k = options
        .and_then(|o| o.get("top_k"))
        .and_then(Value::as_u64)
        .map(|k| k as usize)
        .unwrap_or(studio_cfg.override_top_k.unwrap_or(OLLAMA_DEFAULT_TOP_K));
    let repetition_penalty = options
        .and_then(|o| {
            o.get("repeat_penalty")
                .or_else(|| o.get("repetition_penalty"))
        })
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(studio_cfg.sampling.repetition_penalty);
    let sampling = crate::sampling::SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        ..Default::default()
    };
    let sampling = match request_sampling(
        &state,
        &Value::Object(options.cloned().unwrap_or_default()),
        sampling,
    ) {
        Ok(params) => params,
        Err(error) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &error),
    };
    // Ollama: num_predict -1 (the default) generates until the model stops or the context is full, -2 fills the context
    let remaining = state.max_context.saturating_sub(prompt.len());
    let max_tokens = match options.and_then(|o| o.get("num_predict")).and_then(Value::as_i64) {
        Some(n) if n >= 0 => n as usize,
        _ => remaining,
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(true);
    let model = match api::ollama_extra::resolve_model(&state, &body) {
        Ok(model) => model,
        Err(response) => return response,
    };
    let id = state.next_request.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("ollama_{id:016x}");
    if max_tokens == 0 {
        let mut out = Map::new();
        out.insert("model".into(), json!(model));
        out.insert("created_at".into(), json!(iso_now()));
        if is_chat {
            out.insert("message".into(), json!({"role":"assistant","content":""}));
        } else {
            out.insert("response".into(), json!(""));
        }
        out.insert("done".into(), json!(true));
        out.insert("prompt_eval_count".into(), json!(prompt.len()));
        out.insert("eval_count".into(), json!(0));
        out.insert("done_reason".into(), json!("length"));
        return Json(Value::Object(out)).into_response();
    }
    let mut rx = match submit_constrained(
        &state,
        &request_id,
        prompt.clone(),
        max_tokens,
        sampling,
        match request_constraint(&state, &body) {
            Ok(c) => c,
            Err(e) => return api::ollama_extra::error(StatusCode::BAD_REQUEST, &e),
        },
        images,
        cache_salt_of(&body),
    ) {
        Ok(rx) => rx,
        Err(response) => return api::ollama_extra::error_from(response).await,
    };
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    if stream {
        let state2 = state.clone();
        let stream_body = async_stream::stream! {
            let mut ids = Vec::<u32>::new();
            let mut filter = StopFilter::new(stop_strings.clone());
            let mut visible_len = 0usize;
            let mut emitted_calls = 0usize;
            let started = Instant::now();
            while let Some(message) = rx.recv().await {
                match message {
                    Ok(event) => {
                        ids.push(event.token_id as u32);
                        state2.metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);
                        let full = match state2.tokenizer.decode_keeping(&ids, &tool_keep) {
                            Ok(full) => full,
                            Err(error) => {
                                yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n", json!({"error":error}))));
                                return;
                            }
                        };
                        let update = filter.update(&full, event.finished);
                        let (visible, calls) = match tool_spec.parse_reply(filter.text(), update.matched || event.finished, &request_id) {
                            Ok(result) => result,
                            Err(error) => { yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n",json!({"error":error})))); return; }
                        };
                        let delta = visible.get(visible_len..).unwrap_or("").to_string();
                        visible_len = visible.len();
                        if !delta.is_empty() {
                            let obj = if is_chat {
                                json!({"model":model,"created_at":iso_now(),"message":{"role":"assistant","content":delta},"done":false})
                            } else {
                                json!({"model":model,"created_at":iso_now(),"response":update.delta,"done":false})
                            };
                            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n", obj)));
                        }
                        if is_chat && calls.len() > emitted_calls {
                            let fresh = ollama_tool_calls(&calls[emitted_calls..]);
                            emitted_calls = calls.len();
                            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n",json!({"model":model,"created_at":iso_now(),"message":{"role":"assistant","content":"","tool_calls":fresh},"done":false}))));
                        }
                        if update.matched || event.finished {
                            let reason = if update.matched { "stop" } else { event.finish_reason.unwrap_or("stop") };
                            let done_obj = if is_chat {
                                json!({"model":model,"created_at":iso_now(),"message":{"role":"assistant","content":""},"done":true,"total_duration":started.elapsed().as_nanos() as u64,"load_duration":0,"prompt_eval_count":prompt.len(),"eval_count":ids.len(),"done_reason":reason})
                            } else {
                                json!({"model":model,"created_at":iso_now(),"response":"","done":true,"total_duration":started.elapsed().as_nanos() as u64,"load_duration":0,"prompt_eval_count":prompt.len(),"eval_count":ids.len(),"done_reason":reason})
                            };
                            yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n", done_obj)));
                            return;
                        }
                    }
                    Err(error) => {
                        yield Ok::<Bytes, Infallible>(Bytes::from(format!("{}\n", json!({"error":error}))));
                        return;
                    }
                }
            }
        };
        Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from_stream(stream_body))
            .unwrap()
    } else {
        let started = Instant::now();
        let mut ids = Vec::<u32>::new();
        let mut filter = StopFilter::new(stop_strings);
        let mut reason = "stop";
        while let Some(message) = rx.recv().await {
            match message {
                Ok(event) => {
                    ids.push(event.token_id as u32);
                    state
                        .metrics
                        .generated_tokens
                        .fetch_add(1, Ordering::Relaxed);
                    let full = match state.tokenizer.decode_keeping(&ids, &tool_keep) {
                        Ok(full) => full,
                        Err(error) => return api::ollama_extra::error(StatusCode::INTERNAL_SERVER_ERROR, &error),
                    };
                    let update = filter.update(&full, event.finished);
                    if update.matched {
                        reason = "stop";
                        break;
                    }
                    if event.finished {
                        reason = event.finish_reason.unwrap_or("stop");
                        break;
                    }
                }
                Err(error) => return api::ollama_extra::error(StatusCode::INTERNAL_SERVER_ERROR, &error),
            }
        }
        let text = filter.text();
        let (text, calls) = match tool_spec.parse_reply(text, true, &request_id) {
            Ok(result) => result,
            Err(error) => return api::ollama_extra::error(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
        let out = if is_chat {
            let mut message = json!({"role":"assistant","content":text});
            if !calls.is_empty() {
                message["tool_calls"] = json!(ollama_tool_calls(&calls));
            }
            json!({"model":model,"created_at":iso_now(),"message":message,"done":true,"total_duration":started.elapsed().as_nanos() as u64,"load_duration":0,"prompt_eval_count":prompt.len(),"eval_count":ids.len(),"done_reason":reason})
        } else {
            json!({"model":model,"created_at":iso_now(),"response":text,"done":true,"total_duration":started.elapsed().as_nanos() as u64,"load_duration":0,"prompt_eval_count":prompt.len(),"eval_count":ids.len(),"done_reason":reason})
        };
        Json(out).into_response()
    }
}

fn ollama_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls.iter().map(|call| json!({"function":{"name":call["function"]["name"],"arguments":serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))}})).collect()
}

/// transformers picks the named `tool_use` template when tools are passed and the checkpoint has one.
fn select_chat_template<'a>(default: &'a str, tool_use: &'a str, has_tools: bool) -> &'a str {
    if has_tools && !tool_use.is_empty() {
        tool_use
    } else {
        default
    }
}

fn render_messages(
    state: &AppState,
    mut messages: Value,
    tools: Option<Value>,
    kwargs: Option<Map<String, Value>>,
) -> Result<String, String> {
    normalize_messages(&mut messages)?;
    let list = messages.as_array().unwrap();
    if state.chat_template.is_empty() {
        let mut prompt = String::new();
        for message in list {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let content = message.get("content").and_then(Value::as_str).unwrap_or("");
            prompt.push_str(&format!("{role}: {content}\n"));
        }
        prompt.push_str("assistant:");
        return Ok(prompt);
    }
    let mut context = Map::new();
    context.insert("messages".into(), messages);
    context.insert("add_generation_prompt".into(), Value::Bool(true));
    if let Some(tools) = tools {
        context.insert("tools".into(), tools);
    }
    if let Some(bos) = state.tokenizer.bos_ids().first() {
        if let Ok(token) = state.tokenizer.decode(&[*bos], false) {
            context.insert("bos_token".into(), Value::String(token));
        }
    }
    if let Some(eos) = state.tokenizer.eos_ids().first() {
        if let Ok(token) = state.tokenizer.decode(&[*eos], false) {
            context.insert("eos_token".into(), Value::String(token));
        }
    }
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs {
            context.insert(key, value);
        }
    }
    let has_tools = context.get("tools").is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let template = select_chat_template(&state.chat_template, &state.tool_chat_template, has_tools);
    let env = chat_environment();
    env.render_str(template, Value::Object(context))
        .map_err(|e| format!("chat template failed: {e}"))
}

fn normalize_messages(messages: &mut Value) -> Result<(), String> {
    for message in messages.as_array_mut().ok_or("messages must be an array")? {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("each message needs a role")?;
        if !matches!(
            role,
            "system" | "developer" | "user" | "assistant" | "tool" | "function"
        ) {
            return Err("unsupported message role".into());
        }
        match message.get("content") {
            None | Some(Value::Null)
                if role == "assistant" && message.get("tool_calls").is_some() =>
            {
                message["content"] = json!("");
            }
            Some(Value::String(_)) => {}
            Some(Value::Array(parts)) => {
                let mut content = String::new();
                for part in parts {
                    if part.get("type").and_then(Value::as_str) != Some("text") {
                        return Err("this fLLM model path supports text content; native image/audio inference is not yet implemented".into());
                    }
                    content.push_str(
                        part.get("text")
                            .and_then(Value::as_str)
                            .ok_or("text content part needs text")?,
                    );
                }
                message["content"] = json!(content);
            }
            _ => return Err("each message needs text content, or assistant tool_calls".into()),
        }
        if let Some(calls) = message.get_mut("tool_calls") {
            for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                if let Some(arguments) = call
                    .get_mut("function")
                    .and_then(|f| f.get_mut("arguments"))
                {
                    if let Some(raw) = arguments.as_str() {
                        *arguments = serde_json::from_str(raw)
                            .map_err(|e| format!("tool arguments must be JSON: {e}"))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// OpenAI's `context_length_exceeded` reply, with the numbers a client needs to fix the request.
fn context_length_error(max_context: usize, prompt: usize, max_tokens: usize) -> Response {
    let (message, param) = if prompt >= max_context {
        (
            format!(
                "This model's maximum context length is {max_context} tokens. However, your prompt has {prompt} tokens. Please reduce the length of the messages or prompt."
            ),
            "messages",
        )
    } else {
        (
            format!(
                "This model's maximum context length is {max_context} tokens. However, you requested {} tokens ({prompt} in the prompt, {max_tokens} in the completion). Please reduce the length of the prompt or the number of completion tokens.",
                prompt + max_tokens
            ),
            "max_tokens",
        )
    };
    api_error_with(StatusCode::BAD_REQUEST, &message, Some("context_length_exceeded"), Some(param))
}

/// The admission queue (`--max-queue`) is full: 429 with a Retry-After that grows with the backlog.
fn queue_full(state: &AppState) -> Response {
    state.metrics.queue_rejections.fetch_add(1, Ordering::Relaxed);
    let waiting = state.metrics.slots_waiting.load(Ordering::Relaxed);
    let running = state.metrics.slots_running.load(Ordering::Relaxed).max(1);
    let retry_after = (1 + waiting / running).min(30);
    let mut response = api_error_with(
        StatusCode::TOO_MANY_REQUESTS,
        "the request queue is full (--max-queue); retry later",
        Some("queue_full"),
        None,
    );
    if let Ok(value) = retry_after.to_string().parse() {
        response.headers_mut().insert(axum::http::header::RETRY_AFTER, value);
    }
    response
}

// Axum responses are intentionally returned directly on this error-only path.
#[allow(clippy::result_large_err)]
// Eight arguments is a lot, but this is the one place where a request turns into a worker command; a struct would only move
// the same fields one level down.
#[allow(clippy::too_many_arguments)]
fn submit_constrained(
    state: &AppState,
    request_id: &str,
    prompt: Vec<i64>,
    max_tokens: usize,
    sampling: crate::sampling::SamplingParams,
    constraint: Option<llguidance::Matcher>,
    images: ImageOverrides,
    prefix_salt: u64,
) -> Result<GenerationRx, Response> {
    submit_request(state, request_id, prompt, max_tokens, sampling, constraint, images, prefix_salt, None)
}

/// `submit_constrained`, or a beam search when `beam` is given.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn submit_request(
    state: &AppState,
    request_id: &str,
    prompt: Vec<i64>,
    max_tokens: usize,
    sampling: crate::sampling::SamplingParams,
    constraint: Option<llguidance::Matcher>,
    images: ImageOverrides,
    prefix_salt: u64,
    beam: Option<crate::beam::BeamParams>,
) -> Result<GenerationRx, Response> {
    if prompt.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "prompt must encode to at least one token",
        ));
    }
    if prompt.len().saturating_add(max_tokens) > state.max_context {
        return Err(context_length_error(state.max_context, prompt.len(), max_tokens));
    }
    if prompt
        .iter()
        .any(|id| *id < 0 || (*id as usize) >= state.tokenizer.vocab_size())
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "prompt token IDs are outside the model vocabulary",
        ));
    }
    let permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| queue_full(state))?;
    let caller = security::current_caller();
    if let Some(caller) = &caller {
        // prompt tokens count against --rate-limit's token budget when the request is admitted, generated ones as they appear
        caller.stats.prompt_tokens.fetch_add(prompt.len() as u64, Ordering::Relaxed);
        if let Some(limiter) = &caller.limiter {
            limiter.charge_tokens(prompt.len());
        }
    }
    state.metrics.observability.prompt_tokens(prompt.len());
    if let Some(caller) = &caller {
        tracing::debug!(request_id = %caller.request_id, engine_request = request_id, prompt_tokens = prompt.len(), max_tokens, "request admitted");
    }
    let (tx, rx) = tokio_mpsc::channel(256);
    let observation = state.metrics.observability.generation();
    let error = Arc::new(std::sync::Mutex::new(None));
    state.metrics.observability.queue();
    state
        .worker
        .send(WorkerCommand::Submit {
            request_id: request_id.into(),
            prompt,
            max_tokens,
            prefix_salt,
            sampling: Box::new(sampling),
            constraint: constraint.map(Box::new),
            images: Box::new(images),
            beam: beam.map(Box::new),
            output: GenerationTx {
                inner: tx,
                error: error.clone(),
                submitted: Instant::now(),
            },
        })
        .map_err(|_| {
            state.metrics.observability.dequeue();
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "model scheduler is unavailable",
            )
        })?;
    Ok(GenerationRx {
        observation,
        inner: rx,
        error,
        _permit: permit,
        caller,
    })
}

/// An error the engine reported for a request it accepted: beam-search refusals are the client's to fix (400) or a matter of
/// waiting (503); anything else is ours.
fn generation_error(message: &str) -> Response {
    let status = if message.starts_with(crate::beam::REFUSED) {
        StatusCode::BAD_REQUEST
    } else if message.starts_with(crate::beam::BUSY) {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    api_error(status, message)
}

fn api_error(status: StatusCode, message: &str) -> Response {
    errors::openai(status, message, None, None)
}

/// An OpenAI error object with a machine-readable `code` and the offending request `param`.
fn api_error_with(status: StatusCode, message: &str, code: Option<&str>, param: Option<&str>) -> Response {
    errors::openai(status, message, code, param)
}

fn empty_openai_response(
    options: (bool, bool, &str, usize),
    id: &str,
    created: u64,
    model: &str,
    is_chat: bool,
    stream: bool,
    prompt_tokens: usize,
) -> Response {
    let (include_usage, logprobs, echo_text, n) = options;
    let mut choices = if is_chat && stream {
        json!([{"index":0,"delta":{},"finish_reason":"length","logprobs":null}])
    } else if is_chat {
        json!([{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"length","logprobs":null}])
    } else {
        json!([{"index":0,"text":echo_text,"finish_reason":"length","logprobs":null}])
    };
    if logprobs {
        choices[0]["logprobs"] = if is_chat {
            json!({"content":[],"refusal":null})
        } else {
            empty_completion_logprobs()
        };
    }
    choices = json!((0..n)
        .map(|index| {
            let mut choice = choices[0].clone();
            choice["index"] = json!(index);
            choice
        })
        .collect::<Vec<_>>());
    let mut result = json!({"id":id,"object":if is_chat{if stream{"chat.completion.chunk"}else{"chat.completion"}}else{"text_completion"},"created":created,"model":model,"choices":choices,"usage":{"prompt_tokens":prompt_tokens,"completion_tokens":0,"total_tokens":prompt_tokens}});
    result["usage"] = openai_usage(prompt_tokens, 0, 0, 0);
    if stream {
        let mut usage_chunk = result.clone();
        usage_chunk["choices"] = json!([]);
        result.as_object_mut().map(|v| v.remove("usage"));
        Sse::new(async_stream::stream!{
            yield Ok::<Event,Infallible>(Event::default().data(result.to_string()));
            if include_usage { yield Ok::<Event,Infallible>(Event::default().data(usage_chunk.to_string())); }
            yield Ok::<Event,Infallible>(Event::default().data("[DONE]"));
        }).into_response()
    } else {
        Json(result).into_response()
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn iso_now() -> String {
    format!(
        "{}Z",
        chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
            .trim_end_matches('Z')
    )
}

#[cfg(test)]
mod stop_filter_tests {
    use super::{parse_stop_strings, StopFilter};
    use serde_json::json;

    #[test]
    fn incomplete_utf8_does_not_leak_replacement_or_duplicate_text() {
        let mut filter = StopFilter::new(vec![]);
        assert_eq!(filter.update("Gr\u{fffd}", false).delta, "Gr");
        assert_eq!(filter.update("Grüße", false).delta, "üße");
        assert_eq!(filter.update("Grüße\u{fffd}", true).delta, "\u{fffd}");
    }

    #[test]
    fn stop_prefix_is_held_across_token_updates_and_never_emitted() {
        let mut filter = StopFilter::new(vec!["<END>".into()]);
        let first = filter.update("answer <EN", false);
        assert_eq!(first.delta, "answer ");
        assert!(!first.matched);
        let second = filter.update("answer <END> ignored", false);
        assert_eq!(second.delta, "");
        assert_eq!(filter.text(), "answer ");
        assert!(second.matched);
    }

    #[test]
    fn final_chunk_flushes_a_partial_stop_prefix() {
        let mut filter = StopFilter::new(vec!["<END>".into()]);
        let first = filter.update("answer <EN", false);
        assert_eq!(first.delta, "answer ");
        let final_update = filter.update("answer <EN", true);
        assert_eq!(final_update.delta, "<EN");
        assert_eq!(filter.text(), "answer <EN");
        assert!(!final_update.matched);
    }

    #[test]
    fn chooses_the_earliest_of_multiple_stop_strings() {
        let mut filter = StopFilter::new(vec!["STOP".into(), "END".into()]);
        let update = filter.update("aENDbSTOP", true);
        assert_eq!(filter.text(), "a");
        assert!(update.matched);
    }

    #[test]
    fn stop_matching_handles_unicode_boundaries() {
        let mut filter = StopFilter::new(vec!["✨done".into()]);
        let first = filter.update("ok✨", false);
        assert_eq!(first.delta, "ok");
        let second = filter.update("ok✨done!", false);
        assert_eq!(filter.text(), "ok");
        assert!(second.matched);
    }

    #[test]
    fn parser_accepts_string_or_string_array_and_rejects_other_values() {
        assert_eq!(
            parse_stop_strings(Some(&json!("END"))).unwrap(),
            vec!["END"]
        );
        assert_eq!(
            parse_stop_strings(Some(&json!(["END", "STOP"]))).unwrap(),
            vec!["END", "STOP"]
        );
        assert!(parse_stop_strings(Some(&json!(["END", 3]))).is_err());
        assert!(parse_stop_strings(Some(&json!(true))).is_err());
    }
}

#[cfg(test)]
mod chat_template_tests {
    use super::*;

    #[test]
    fn tool_history_and_text_parts_are_normalized_without_losing_media() {
        let mut messages = json!([{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"{\"city\":\"Berlin\"}"}}]},
            {"role":"user","content":[{"type":"text","text":"Hi"},{"type":"text","text":" there"}]}]);
        normalize_messages(&mut messages).unwrap();
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"]["city"],
            "Berlin"
        );
        assert_eq!(messages[1]["content"], "Hi there");
        assert!(normalize_messages(&mut json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}])).is_err());
    }

    #[tokio::test]
    async fn bounded_output_reports_backpressure_and_disconnect() {
        let (inner, receiver) = tokio_mpsc::channel(2);
        let error = Arc::new(std::sync::Mutex::new(None));
        let sender = GenerationTx {
            inner,
            error: error.clone(),
            submitted: Instant::now(),
        };
        let permit = Arc::new(tokio::sync::Semaphore::new(1))
            .acquire_owned()
            .await
            .unwrap();
        let mut receiver = GenerationRx {
            inner: receiver,
            error,
            observation: Arc::new(crate::metrics::Metrics::new()).generation(),
            _permit: permit,
            caller: None,
        };
        assert!(sender.send(Err("first".into())).is_ok());
        assert!(sender.send(Err("second".into())).is_ok());
        assert!(sender.send(Err("overflow".into())).is_err());
        drop(sender);
        assert_eq!(receiver.recv().await.unwrap().unwrap_err(), "first");
        assert_eq!(receiver.recv().await.unwrap().unwrap_err(), "second");
        assert!(receiver
            .recv()
            .await
            .unwrap()
            .unwrap_err()
            .contains("buffer is full"));
        assert!(receiver.recv().await.is_none());
        let (inner, receiver) = tokio_mpsc::channel(1);
        let sender = GenerationTx {
            inner,
            error: Arc::new(std::sync::Mutex::new(None)),
            submitted: Instant::now(),
        };
        drop(receiver);
        assert!(sender.is_closed());
    }

    #[test]
    fn generation_tags_become_noop_blocks_and_keep_whitespace_control() {
        let template = "a{%- generation -%}b{% endgeneration %}c{%- endgeneration %}";
        let normalized = normalize_chat_template(template);
        assert_eq!(normalized, "a{%- if true -%}b{% endif %}c{%- endif %}");
        assert_eq!(
            normalize_chat_template("{% if x %}y{% endif %}"),
            "{% if x %}y{% endif %}"
        );
    }

    #[test]
    fn tool_use_template_is_chosen_only_with_tools() {
        assert_eq!(select_chat_template("D", "T", true), "T");
        assert_eq!(select_chat_template("D", "T", false), "D");
        assert_eq!(select_chat_template("D", "", true), "D");
    }

    #[test]
    fn hostile_chat_template_returns_err_cleanly() {
        let hostile = "{% set ns = namespace(n=0) %}{% for fm in messages %}{% set ns.n = ns %}{{ ns.n }}{% endfor %}";
        let res = render_template_for_fuzz(hostile, &json!([{"role": "user", "content": "hello"}]));
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("cyclic") || err.contains("self-referential"));

        let deep = "(".repeat(65) + &")".repeat(65);
        assert!(validate_chat_template(&deep).is_err());
    }

    #[test]
    fn counting_with_a_namespace_attribute_is_not_a_cycle() {
        let counting = "{% set ns = namespace(n=0, r='') %}{% set ns.n = ns.n + 1 %}{%- set ns.r = ns.r + part.get('x') -%}";
        assert!(validate_chat_template(counting).is_ok());
        assert!(validate_chat_template("{% set ns.n = [ns] %}").is_err());
    }

    #[test]
    fn python_style_helpers_work_in_templates() {
        let env = chat_environment();
        let out = env
            .render_str(
                "{{ 'a b  c'.split()|join('-') }}|{{ 'x\n\n'.rstrip('\n') }}|{{ {'k': [1, 2]}|tojson }}",
                Value::Null,
            )
            .unwrap();
        assert_eq!(out, "a-b-c|x|{\"k\": [1, 2]}");
        let err = env
            .render_str("{{ raise_exception('nope') }}", Value::Null)
            .unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn incomplete_characters_are_held_back_while_streaming() {
        assert_eq!(hold_incomplete_utf8("ok \u{FFFD}", false), "ok ");
        assert_eq!(hold_incomplete_utf8("ok \u{FFFD}", true), "ok \u{FFFD}");
        let spec = ReasoningSpec::from_prompt("x<ifm|think>\n");
        let early = spec.split_views(hold_incomplete_utf8("thinking \u{FFFD}", false), false);
        let late = spec.split_views(
            hold_incomplete_utf8("thinking \u{1F3AE} done", false),
            false,
        );
        assert!(late.reasoning.starts_with(&early.reasoning));
    }

    #[test]
    fn reasoning_split_for_prompts_that_open_the_think_block() {
        let spec = ReasoningSpec::from_prompt("<|ifm|im_start|>assistant\n<ifm|think>\n");
        // The prompt opened the block, so the first text already is reasoning.
        assert!(spec.split_views("so far", false).reasoning_found);
        let parts = spec.split_views("Let me think.\n</ifm|think>\n391", true);
        assert_eq!(parts.reasoning, "Let me think.");
        assert_eq!(parts.content, "391");
        // still thinking: a partial closing tag is held back
        let parts = spec.split_views("Step one\n</ifm|th", false);
        assert_eq!(parts.reasoning, "Step one");
        assert_eq!(parts.content, "");
        let parts = spec.split_views("cut off while thinking", true);
        assert_eq!(parts.reasoning, "cut off while thinking");
        assert_eq!(parts.content, "");
    }

    #[test]
    fn reasoning_split_when_the_model_opens_the_tag_itself() {
        let spec = ReasoningSpec::from_prompt("<|im_start|>assistant\n");
        let parts = spec.split_views("<think>\nhmm\n</think>\n\nAnswer", true);
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("hmm", "Answer")
        );
        let plain = spec.split_views("No thinking here", true);
        assert!(!plain.reasoning_found);
        assert_eq!(plain.content, "No thinking here");
        let closed = ReasoningSpec::from_prompt("x<think>\n\n</think>\n\n");
        assert!(!closed.split_views("Answer", true).reasoning_found);
    }
}

#[cfg(test)]
mod openai_field_tests {
    use super::*;
    #[test]
    fn grammar_control_marker_is_not_a_probability_byte() {
        use llguidance::toktrie::{TokRxInfo, TokTrie};
        let trie = TokTrie::from(
            &TokRxInfo::new(3, 2),
            &[b"a".to_vec(), vec![0xc3], b"\xff</think>".to_vec()],
        );
        assert_eq!(probability_token_bytes(&trie, 0), b"a");
        assert_eq!(probability_token_bytes(&trie, 1), &[0xc3]);
        assert_eq!(probability_token_bytes(&trie, 2), b"</think>");
    }
    #[test]
    fn stop_strings_before_minimum_do_not_stop_retroactively() {
        let mut filter = StopFilter::new(vec!["END".into()]);
        assert!(!filter.update_with_minimum("END", false, false).matched);
        assert!(!filter.update_with_minimum("END ok", false, true).matched);
        assert!(
            filter
                .update_with_minimum("END ok END", false, true)
                .matched
        );
        assert_eq!(filter.text(), "END ok ");
    }
    #[test]
    fn validate_logprobs_and_stop_controls() {
        let mut p = crate::sampling::SamplingParams::default();
        assert!(openai_sampling_controls(
            &json!({"logprobs":true,"top_logprobs":20,"min_tokens":2,"stop_token_ids":[1,2]}),
            true,
            8,
            &mut p
        )
        .is_ok());
        assert_eq!(p.logprobs, Some(20));
        for body in [
            json!({"top_logprobs":21,"logprobs":true}),
            json!({"top_logprobs":1}),
            json!({"stop_token_ids":[8]}),
            json!({"ignore_eos":"yes"}),
        ] {
            assert!(openai_sampling_controls(&body, true, 8, &mut p).is_err());
        }
    }
    #[test]
    fn visible_logprobs_exclude_hidden_and_partial_tokens() {
        let mut p = VisibleLogprobs {
            raw: b"secretOKSTOP".to_vec(),
            records: vec![
                (
                    0,
                    6,
                    json!({"token":"secret","logprob":-1.0,"top_logprobs":[]}),
                    false,
                ),
                (
                    6,
                    8,
                    json!({"token":"OK","logprob":-2.0,"top_logprobs":[]}),
                    false,
                ),
                (
                    8,
                    12,
                    json!({"token":"STOP","logprob":-3.0,"top_logprobs":[]}),
                    false,
                ),
            ],
        };
        let lp = p.take("OK", true, 0);
        assert_eq!(lp["content"].as_array().unwrap().len(), 1);
        assert_eq!(lp["content"][0]["token"], "OK");
        assert_eq!(p.take("OK", true, 0)["content"], json!([]));
    }
    #[test]
    fn completion_offsets_follow_unicode_characters_and_include_sampled_token() {
        let mut p = VisibleLogprobs {
            raw: vec![0xc3, 0xa9, b'x'],
            records: vec![
                (
                    0,
                    1,
                    json!({"token":"first-byte","logprob":-1.0,"top_logprobs":[]}),
                    false,
                ),
                (
                    1,
                    2,
                    json!({"token":"second-byte","logprob":-2.0,"top_logprobs":[]}),
                    false,
                ),
                (
                    2,
                    3,
                    json!({"token":"x","logprob":-3.0,"top_logprobs":[{"token":"y","logprob":-0.5}]}),
                    false,
                ),
            ],
        };
        let lp = p.take("\u{e9}x", false, 4);
        assert_eq!(lp["text_offset"], json!([4, 4, 5]));
        assert_eq!(lp["top_logprobs"][2]["x"], -3.0);
        assert_eq!(lp["top_logprobs"][2]["y"], -0.5);
        let mut sampling = crate::sampling::SamplingParams::default();
        assert!(openai_sampling_controls(&json!({"logprobs":6}), false, 8, &mut sampling).is_err());
    }
    #[test]
    fn usage_details_are_bounded_by_totals() {
        let u = openai_usage(4, 3, 99, 10);
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 3);
    }
}

async fn audio_transcriptions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    handle_audio_request(state, headers, body).await
}

async fn audio_translations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    handle_audio_request(state, headers, body).await
}

async fn handle_audio_request(
    _state: Arc<AppState>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let audio_bytes = if content_type.contains("multipart/form-data") {
        extract_file_from_multipart(&body, content_type).unwrap_or_else(|| body.to_vec())
    } else if content_type.contains("application/json") {
        if let Ok(v) = serde_json::from_slice::<Value>(&body) {
            if let Some(s) = v.get("file").and_then(Value::as_str) {
                if let Some(b64) = s.strip_prefix("data:audio/wav;base64,").or_else(|| s.strip_prefix("data:audio/x-wav;base64,")) {
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64).unwrap_or_default()
                } else {
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, s).unwrap_or_default()
                }
            } else {
                body.to_vec()
            }
        } else {
            body.to_vec()
        }
    } else {
        body.to_vec()
    };

    if audio_bytes.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "audio file is required");
    }

    match crate::vision::audio::transcribe_audio(&audio_bytes, None) {
        Ok(text) => Json(json!({"text": text})).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

fn extract_file_from_multipart(body: &[u8], content_type: &str) -> Option<Vec<u8>> {
    let boundary = content_type
        .split(';')
        .find_map(|p| p.trim().strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{boundary}").into_bytes();

    let mut start = 0;
    while let Some(idx) = find_subsequence(&body[start..], &delim) {
        let part_start = start + idx + delim.len();
        if part_start + 2 <= body.len() && &body[part_start..part_start + 2] == b"--" {
            break;
        }
        let header_start = if part_start + 2 <= body.len() && &body[part_start..part_start + 2] == b"\r\n" {
            part_start + 2
        } else {
            part_start
        };
        if let Some(header_end) = find_subsequence(&body[header_start..], b"\r\n\r\n") {
            let data_start = header_start + header_end + 4;
            if let Some(next_b) = find_subsequence(&body[data_start..], &delim) {
                let data_end = if next_b >= 2 && &body[data_start + next_b - 2..data_start + next_b] == b"\r\n" {
                    data_start + next_b - 2
                } else {
                    data_start + next_b
                };
                let headers_text = String::from_utf8_lossy(&body[header_start..header_start + header_end]);
                if headers_text.contains("name=\"file\"") || headers_text.contains("filename=") {
                    return Some(body[data_start..data_end].to_vec());
                }
            }
        }
        start = part_start;
    }
    None
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// `fllm-rs sandbox-exec ...` (hidden): the launcher of the Windows command sandbox. It runs one command line under a
/// restricted, low-integrity token inside a job object and returns the exit code of that command.
pub fn run_sandbox_launcher(args: &[String]) -> i32 {
    api::run_sandbox_launcher(args)
}
