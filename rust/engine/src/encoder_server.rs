//! Dedicated encoder HTTP serving. Computation is exclusively native Rust.
//!
//! Routes: `/v1/embeddings` (dense, plus the extension fields `sparse`, `multi_vector`,
//! `normalize`, `dimensions`, `input_type`, `truncate_prompt_tokens`), `/pooling`, `/classify`,
//! `/v1/score`, `/v1/rerank`, `/tokenize`, `/detokenize`. Requests are queued and run in batches
//! by one worker thread that merges queued requests up to a token budget.
use crate::{
    encoder::{BertEncoder, Outputs, Want},
    encoder_heads::{self, InputKind, SparseVector},
    server::ServerConfig,
};
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

/// Limits read from the environment when the server starts.
struct Limits {
    max_inputs: usize,
    max_request_tokens: usize,
    queue: usize,
}

impl Limits {
    fn from_env() -> Self {
        let number = |key: &str, default: usize| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(default)
        };
        Self {
            max_inputs: number("FLLM_ENCODER_MAX_INPUTS", 32),
            max_request_tokens: number("FLLM_ENCODER_MAX_REQUEST_TOKENS", 32768),
            queue: number("FLLM_ENCODER_QUEUE", 256),
        }
    }
}

static BATCH_TOKENS: AtomicUsize = AtomicUsize::new(0);

/// Token budget of one merged batch (`--max-batched-tokens`); a single larger request still runs.
pub fn set_max_batched_tokens(tokens: usize) {
    BATCH_TOKENS.store(tokens, Ordering::Relaxed);
}

fn batch_token_budget() -> usize {
    let explicit = BATCH_TOKENS.load(Ordering::Relaxed);
    if explicit > 0 {
        return explicit;
    }
    std::env::var("FLLM_ENCODER_MAX_BATCHED_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(8192)
}

enum Job {
    Encode(Vec<u32>, Want),
    Classify(Vec<u32>, Vec<u32>),
    Tokens(Vec<u32>),
}

impl Job {
    fn tokens(&self) -> usize {
        match self {
            Job::Encode(ids, _) | Job::Classify(ids, _) | Job::Tokens(ids) => ids.len(),
        }
    }
}

enum JobOut {
    Encode(Outputs),
    Classified { probs: Vec<f32>, logits: Vec<f32> },
    TokenProbs(Vec<Vec<f32>>),
}

struct Task {
    jobs: Vec<Job>,
    reply: tokio::sync::oneshot::Sender<Result<Vec<JobOut>, String>>,
}

fn run_job(model: &BertEncoder, job: &Job) -> Result<JobOut, String> {
    match job {
        Job::Encode(ids, want) => model.encode_outputs(ids, want).map(JobOut::Encode),
        Job::Classify(ids, segments) => model
            .classify_with_logits(ids, segments)
            .map(|(probs, logits)| JobOut::Classified { probs, logits }),
        Job::Tokens(ids) => model.classify_tokens(ids).map(JobOut::TokenProbs),
    }
}

/// Merges queued tasks up to the token budget and runs every job of the batch in parallel.
fn batch_worker(
    model: Arc<BertEncoder>,
    rx: std::sync::mpsc::Receiver<Task>,
    budget: usize,
    window: Duration,
) {
    while let Ok(first) = rx.recv() {
        let mut tokens: usize = first.jobs.iter().map(Job::tokens).sum();
        let mut tasks = vec![first];
        let deadline = Instant::now() + window;
        while tokens < budget {
            let next = match rx.try_recv() {
                Ok(task) => Some(task),
                Err(std::sync::mpsc::TryRecvError::Empty) if !window.is_zero() => {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        None
                    } else {
                        rx.recv_timeout(left).ok()
                    }
                }
                Err(_) => None,
            };
            match next {
                Some(task) => {
                    tokens += task.jobs.iter().map(Job::tokens).sum::<usize>();
                    tasks.push(task);
                }
                None => break,
            }
        }
        let lengths = tasks.iter().map(|t| t.jobs.len()).collect::<Vec<_>>();
        let flat = tasks
            .iter()
            .flat_map(|t| t.jobs.iter())
            .collect::<Vec<_>>();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.parallel(&flat, |job| run_job(&model, job))
        }));
        match outcome {
            Ok(results) => {
                let mut results = results.into_iter();
                for (task, count) in tasks.into_iter().zip(lengths) {
                    let own = results.by_ref().take(count).collect::<Result<Vec<_>, _>>();
                    let _ = task.reply.send(own);
                }
            }
            Err(_) => {
                for task in tasks {
                    let _ = task.reply.send(Err("encoder worker panicked".into()));
                }
            }
        }
    }
}

struct EncoderState {
    metrics: Arc<crate::metrics::Metrics>,
    model: Arc<BertEncoder>,
    name: String,
    limits: Limits,
    queue: Arc<tokio::sync::Semaphore>,
    jobs: std::sync::mpsc::Sender<Task>,
}

type Failure = (StatusCode, String);

impl EncoderState {
    /// Queues jobs for the batch worker and waits for their results.
    async fn submit(&self, jobs: Vec<Job>) -> Result<Vec<JobOut>, Failure> {
        let count = u32::try_from(jobs.len()).unwrap_or(u32::MAX);
        let tokens: usize = jobs.iter().map(Job::tokens).sum();
        let _permit = self
            .queue
            .clone()
            .try_acquire_many_owned(count)
            .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "encoder is busy".to_string()))?;
        let _active = self.metrics.embedding();
        let (reply, receive) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Task { jobs, reply })
            .map_err(|_| internal("encoder worker stopped"))?;
        let outputs = receive
            .await
            .map_err(|_| internal("encoder worker failed"))?
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        self.metrics.processed(tokens);
        Ok(outputs)
    }
}

fn internal(message: &str) -> Failure {
    (StatusCode::INTERNAL_SERVER_ERROR, message.to_owned())
}

fn bad(message: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, message.into())
}

fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":if status.is_server_error(){"server_error"}else{"invalid_request_error"}}})),
    )
        .into_response()
}
fn fail(f: Failure) -> Response {
    error(f.0, &f.1)
}
fn capabilities_json(state: &EncoderState) -> Value {
    let c = state.model.capabilities();
    json!({
        "dense": c.dense,
        "sparse": c.sparse,
        "multi_vector": c.multi_vector,
        "token_classification": c.token_classification,
        "sequence_classification": c.sequence_classification,
    })
}
async fn models(State(state): State<Arc<EncoderState>>) -> Response {
    Json(json!({"object":"list","data":[{"id":state.name,"object":"model","owned_by":"fllm","architecture":"bert","embedding_dimensions":state.model.embedding_dim,"max_model_len":state.model.context,"capabilities":capabilities_json(&state)}]})).into_response()
}
fn token_ids(value: &Value) -> Result<Vec<u32>, String> {
    value
        .as_array()
        .ok_or("expected token ID array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| "invalid token ID".into())
        })
        .collect()
}
async fn tokenize(
    State(state): State<Arc<EncoderState>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(text) = body.get("content").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "content must be text");
    };
    match state.model.tokenize(
        text,
        body.get("add_special")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    ) {
        Ok(ids) => Json(json!({"tokens":ids})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}
async fn detokenize(
    State(state): State<Arc<EncoderState>>,
    Json(body): Json<Value>,
) -> Response {
    match token_ids(&body["tokens"]).and_then(|ids| state.model.detokenize(&ids)) {
        Ok(content) => Json(json!({"content":content})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}

/// Options shared by `/v1/embeddings` and `/pooling`, parsed without touching the model.
#[derive(Debug, PartialEq)]
struct EncodeOptions {
    dimensions: Option<usize>,
    base64: bool,
    pooling: Option<String>,
    normalize: Option<bool>,
    sparse: Option<bool>,
    multi_vector: Option<bool>,
    input_type: InputKind,
    /// `Some(None)` truncates to the model context, `Some(Some(n))` to `n` tokens.
    truncate: Option<Option<usize>>,
    /// Name of the task LoRA adapter (`task` or `adapter` in the request).
    adapter: Option<String>,
}

fn optional_bool(body: &Value, key: &str) -> Result<Option<bool>, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("{key} must be boolean")),
    }
}

fn parse_options(body: &Value) -> Result<EncodeOptions, String> {
    parse_options_for(body, true)
}

/// `/pooling` uses `task` for the pooling task, so only `adapter` selects a LoRA adapter there.
fn parse_options_for(body: &Value, task_is_adapter: bool) -> Result<EncodeOptions, String> {
    let dimensions = match body.get("dimensions") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or("dimensions must be a positive integer")?,
        ),
    };
    let base64 = match body.get("encoding_format") {
        None | Some(Value::Null) => false,
        Some(v) => match v.as_str() {
            Some("float") => false,
            Some("base64") => true,
            _ => return Err("encoding_format must be float or base64".into()),
        },
    };
    let pooling = match body.get("pooling") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let name = v.as_str().ok_or("pooling must be text")?;
            crate::encoder::validate_pooling(name)?;
            Some(name.to_owned())
        }
    };
    let input_type = match body.get("input_type") {
        None | Some(Value::Null) => InputKind::Document,
        Some(v) => InputKind::parse(v.as_str().ok_or("input_type must be text")?)?,
    };
    let mut truncate = None;
    if optional_bool(body, "truncate")? == Some(true) {
        truncate = Some(None);
    }
    match body.get("truncate_prompt_tokens") {
        None | Some(Value::Null) => {}
        Some(v) => {
            let n = v
                .as_i64()
                .ok_or("truncate_prompt_tokens must be an integer")?;
            truncate = Some(match n {
                -1 => None,
                n if n > 0 => Some(n as usize),
                _ => return Err("truncate_prompt_tokens must be -1 or positive".into()),
            });
        }
    }
    let adapter = match body
        .get("adapter")
        .or_else(|| if task_is_adapter { body.get("task") } else { None })
    {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_str().ok_or("task/adapter must be text")?.to_owned()),
    };
    Ok(EncodeOptions {
        dimensions,
        base64,
        pooling,
        normalize: optional_bool(body, "normalize")?,
        sparse: optional_bool(body, "sparse")?,
        multi_vector: optional_bool(body, "multi_vector")?,
        input_type,
        truncate,
        adapter,
    })
}

/// Resolves the requested adapter to its index and the instruction text to prepend to the inputs.
fn resolve_adapter(
    model: &BertEncoder,
    name: Option<&str>,
) -> Result<(Option<usize>, String), String> {
    let Some(name) = name else {
        return Ok((None, String::new()));
    };
    let names = model.adapter_names();
    if names.is_empty() {
        return Err("this model has no task adapters; remove the task field".into());
    }
    match model.adapter_index(name) {
        Some(index) => Ok((Some(index), model.adapter_instruction(index).to_owned())),
        None => Err(format!(
            "unsupported task {name:?}; supported tasks are {}",
            names.join(", ")
        )),
    }
}

/// Splits an `input` value into token-id arrays (already tokenised) and texts.
fn input_items(input: &Value) -> Result<Vec<Value>, String> {
    if input.is_string()
        || input
            .as_array()
            .is_some_and(|a| !a.is_empty() && a[0].is_number())
    {
        return Ok(vec![input.clone()]);
    }
    input
        .as_array()
        .cloned()
        .ok_or_else(|| "invalid embedding input".to_string())
}

fn tokenize_inputs(
    state: &EncoderState,
    input: &Value,
    truncate: Option<Option<usize>>,
    instruction: &str,
) -> Result<Vec<Vec<u32>>, String> {
    let items = input_items(input)?;
    if items.is_empty() || items.len() > state.limits.max_inputs {
        return Err(format!(
            "embedding batch must contain 1 to {} inputs",
            state.limits.max_inputs
        ));
    }
    let mut batch = Vec::with_capacity(items.len());
    for value in items {
        let mut ids = if let Some(text) = value.as_str() {
            if instruction.is_empty() {
                state.model.tokenize_raw(text, true)?
            } else {
                state.model.tokenize_raw(&format!("{instruction}{text}"), true)?
            }
        } else {
            token_ids(&value)?
        };
        if let Some(limit) = truncate {
            ids = state
                .model
                .truncate_ids(ids, limit.unwrap_or(state.model.context));
        }
        state.model.validate_tokens(&ids)?;
        batch.push(ids);
    }
    let total: usize = batch.iter().map(Vec::len).sum();
    if total > state.limits.max_request_tokens {
        return Err(format!(
            "batch exceeds {} total input tokens",
            state.limits.max_request_tokens
        ));
    }
    Ok(batch)
}

fn sparse_json(sparse: &SparseVector) -> Value {
    let mut map = serde_json::Map::new();
    for (id, weight) in sparse {
        map.insert(id.to_string(), json!(weight));
    }
    Value::Object(map)
}

fn f32_base64(values: impl Iterator<Item = f32>) -> String {
    base64::engine::general_purpose::STANDARD.encode(
        values
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// Which outputs a request asks for, after applying the model's defaults.
fn wanted_outputs(
    caps: crate::encoder::Capabilities,
    options: &EncodeOptions,
    pooling: &str,
) -> Result<Want, String> {
    let sparse = options.sparse.unwrap_or(!caps.dense && caps.sparse);
    let multi = options
        .multi_vector
        .unwrap_or(!caps.dense && !caps.sparse && caps.multi_vector);
    if sparse && !caps.sparse {
        return Err("model has no sparse output".into());
    }
    if multi && !caps.multi_vector {
        return Err("model has no multi-vector output".into());
    }
    // A model with a dense output always returns it; the extension outputs come on top.
    Ok(Want {
        dense: caps.dense.then(|| (pooling.to_owned(), options.normalize)),
        sparse,
        multi: multi.then_some(options.input_type),
        adapter: None,
    })
}

fn check_model(state: &EncoderState, body: &Value) -> Result<(), Failure> {
    if body
        .get("model")
        .is_some_and(|m| m.as_str() != Some(state.name.as_str()))
    {
        return Err(bad("requested model is not loaded"));
    }
    Ok(())
}

async fn embeddings(
    State(state): State<Arc<EncoderState>>,
    Json(body): Json<Value>,
) -> Response {
    match embeddings_inner(&state, &body).await {
        Ok(response) => Json(response).into_response(),
        Err(f) => fail(f),
    }
}

async fn embeddings_inner(state: &Arc<EncoderState>, body: &Value) -> Result<Value, Failure> {
    check_model(state, body)?;
    let options = parse_options(body).map_err(bad)?;
    let caps = state.model.capabilities();
    let dimensions = options.dimensions.unwrap_or(state.model.embedding_dim);
    if dimensions > state.model.embedding_dim {
        return Err(bad("dimensions must be between 1 and encoder output size"));
    }
    let pooling = options
        .pooling
        .clone()
        .unwrap_or_else(|| state.model.default_pooling.clone());
    let (adapter, instruction) =
        resolve_adapter(&state.model, options.adapter.as_deref()).map_err(bad)?;
    let mut want = wanted_outputs(caps, &options, &pooling).map_err(bad)?;
    want.adapter = adapter;
    let input = body
        .get("input")
        .or_else(|| body.get("prompt"))
        .ok_or_else(|| bad("input is required"))?;
    let batch = tokenize_inputs(state, input, options.truncate, &instruction).map_err(bad)?;
    let count = batch.iter().map(Vec::len).sum::<usize>();
    let jobs = batch
        .into_iter()
        .map(|ids| Job::Encode(ids, want.clone()))
        .collect();
    let outputs = state.submit(jobs).await?;
    let normalized = options
        .normalize
        .unwrap_or_else(|| state.model.default_normalize());
    let mut data = Vec::new();
    let mut dense_vectors = Vec::new();
    for (index, out) in outputs.into_iter().enumerate() {
        let JobOut::Encode(out) = out else {
            return Err(internal("unexpected encoder result"));
        };
        let mut item = json!({"object":"embedding","index":index});
        let embedding = match out.dense {
            Some(mut vector) => {
                if dimensions < vector.len() {
                    vector.truncate(dimensions);
                    if normalized {
                        crate::compute::normalize(&mut vector).map_err(|e| internal(&e))?;
                    }
                }
                dense_vectors.push(vector.clone());
                if options.base64 {
                    json!(f32_base64(vector.iter().copied()))
                } else {
                    json!(vector)
                }
            }
            None => json!([]),
        };
        item["embedding"] = embedding;
        if let Some(sparse) = &out.sparse {
            item["sparse"] = sparse_json(sparse);
        }
        if let Some(multi) = &out.multi {
            let dim = multi.first().map_or(0, Vec::len);
            if options.base64 {
                item["multi_vector"] =
                    json!(f32_base64(multi.iter().flat_map(|v| v.iter().copied())));
                item["multi_vector_shape"] = json!([multi.len(), dim]);
            } else {
                item["multi_vector"] = json!(multi);
            }
        }
        data.push(item);
    }
    Ok(json!({
        "object":"list","model":state.name,"data":data,
        "embeddings":dense_vectors,"embedding":dense_vectors.first(),
        "prompt_eval_count":count,
        "usage":{"prompt_tokens":count,"total_tokens":count},
    }))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Rerank,
    Score,
    Classify,
    Pool,
}
impl Operation {
    fn error(self, status: StatusCode, message: &str) -> Response {
        if matches!(self, Self::Rerank) {
            (status, Json(json!({"message":message}))).into_response()
        } else {
            error(status, message)
        }
    }
}
fn text_batch(value: &Value, maximum: usize) -> Result<Vec<String>, String> {
    let values = if let Some(text) = value.as_str() {
        vec![text.to_owned()]
    } else {
        value
            .as_array()
            .ok_or("input must be text or an array of text")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "input array must contain only text".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    if values.is_empty() || values.len() > maximum {
        return Err(format!("input batch must contain 1 to {maximum} items"));
    }
    if values.iter().any(|v| v.len() > 65536) {
        return Err("each text must be at most 65536 bytes".into());
    }
    Ok(values)
}
fn score_pairs(body: &Value) -> Result<Vec<(String, String)>, String> {
    let left = text_batch(&body["text_1"], 32)?;
    let right = text_batch(&body["text_2"], 32)?;
    if left.len() != right.len() && left.len() != 1 && right.len() != 1 {
        return Err(
            "text_1 and text_2 must have equal lengths or one must be a single text".into(),
        );
    }
    let count = left.len().max(right.len());
    Ok((0..count)
        .map(|i| {
            (
                left[if left.len() == 1 { 0 } else { i }].clone(),
                right[if right.len() == 1 { 0 } else { i }].clone(),
            )
        })
        .collect())
}
fn documents(body: &Value) -> Result<Vec<(String, Value)>, String> {
    let values = body["documents"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 256)
        .ok_or("documents must contain 1 to 256 strings or {text} objects")?;
    values
        .iter()
        .map(|v| {
            let text = v
                .as_str()
                .or_else(|| v["text"].as_str())
                .ok_or("each document must be text or an object with text")?;
            if text.len() > 65536 {
                return Err("each document must be at most 65536 bytes".to_string());
            }
            Ok((
                text.to_owned(),
                if v.is_string() {
                    json!({"text":text})
                } else {
                    v.clone()
                },
            ))
        })
        .collect()
}

/// How a pair of texts is scored.
#[derive(Debug, PartialEq)]
enum ScoreSpec {
    /// Cross-encoder with a single-label classification head.
    CrossEncoder,
    /// Weighted sum of the dense cosine, the sparse dot product and the MaxSim score.
    Weighted { dense: f32, sparse: f32, multi: f32 },
}

fn score_spec(
    caps: crate::encoder::Capabilities,
    labels: usize,
    body: &Value,
) -> Result<ScoreSpec, String> {
    let requested = body.get("score_type").and_then(Value::as_str);
    let one = |kind: &str| -> Result<ScoreSpec, String> {
        let (dense, sparse, multi) = match kind {
            "dense" if caps.dense => (1.0, 0.0, 0.0),
            "sparse" if caps.sparse => (0.0, 1.0, 0.0),
            "multi_vector" | "colbert" if caps.multi_vector => (0.0, 0.0, 1.0),
            "dense" | "sparse" | "multi_vector" | "colbert" => {
                return Err(format!("model has no {kind} output"))
            }
            _ => return Err("score_type must be dense, sparse, multi_vector or combined".into()),
        };
        Ok(ScoreSpec::Weighted {
            dense,
            sparse,
            multi,
        })
    };
    match requested {
        Some("cross_encoder") => {
            if caps.sequence_classification && labels == 1 {
                Ok(ScoreSpec::CrossEncoder)
            } else {
                Err("model is not a single-label cross-encoder".into())
            }
        }
        Some("combined") => {
            if !(caps.dense && caps.sparse && caps.multi_vector) {
                return Err("combined scoring needs dense, sparse and multi-vector outputs".into());
            }
            let weights = match body.get("weights") {
                None | Some(Value::Null) => vec![0.4, 0.2, 0.4],
                Some(w) => w
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .and_then(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
                    .ok_or("weights must be three numbers (dense, sparse, multi_vector)")?,
            };
            Ok(ScoreSpec::Weighted {
                dense: weights[0] as f32,
                sparse: weights[1] as f32,
                multi: weights[2] as f32,
            })
        }
        Some(kind) => one(kind),
        None => {
            if caps.sequence_classification && labels == 1 {
                Ok(ScoreSpec::CrossEncoder)
            } else if caps.dense {
                one("dense")
            } else if caps.multi_vector {
                one("multi_vector")
            } else if caps.sparse {
                one("sparse")
            } else {
                Err("model has no output that can score text pairs".into())
            }
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = crate::compute::dot(a, b);
    let na = crate::compute::dot(a, a).sqrt();
    let nb = crate::compute::dot(b, b).sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Scores text pairs; returns the scores and the number of input tokens.
async fn score_texts(
    state: &Arc<EncoderState>,
    pairs: &[(String, String)],
    body: &Value,
) -> Result<(Vec<f32>, usize), Failure> {
    let caps = state.model.capabilities();
    let spec = score_spec(caps, state.model.labels.len(), body).map_err(|e| {
        if caps.sequence_classification {
            bad(e)
        } else {
            (StatusCode::NOT_IMPLEMENTED, e)
        }
    })?;
    match spec {
        ScoreSpec::CrossEncoder => {
            let instruction = body.get("instruction").and_then(Value::as_str);
            let batch = pairs
                .iter()
                .map(|(q, d)| state.model.tokenize_pair_with(q, d, instruction))
                .collect::<Result<Vec<_>, _>>()
                .map_err(bad)?;
            let tokens = batch.iter().map(|(ids, _)| ids.len()).sum::<usize>();
            if tokens > state.limits.max_request_tokens {
                return Err(bad(format!(
                    "batch exceeds {} total input tokens",
                    state.limits.max_request_tokens
                )));
            }
            let outs = state
                .submit(
                    batch
                        .into_iter()
                        .map(|(ids, types)| Job::Classify(ids, types))
                        .collect(),
                )
                .await?;
            let scores = outs
                .into_iter()
                .map(|o| match o {
                    JobOut::Classified { probs, .. } => Ok(probs[0]),
                    _ => Err(internal("unexpected encoder result")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((scores, tokens))
        }
        ScoreSpec::Weighted {
            dense,
            sparse,
            multi,
        } => {
            // Each distinct (text, role) is encoded once; the left text is the query.
            let mut index = HashMap::<(String, bool), usize>::new();
            let mut order = Vec::new();
            let mut slots = Vec::new();
            for (q, d) in pairs {
                let mut slot = |text: &String, query: bool| {
                    *index.entry((text.clone(), query)).or_insert_with(|| {
                        order.push((text.clone(), query));
                        order.len() - 1
                    })
                };
                slots.push((slot(q, true), slot(d, false)));
            }
            let pooling = state.model.default_pooling.clone();
            let (adapter, instruction) = resolve_adapter(
                &state.model,
                body
                    .get("adapter")
                    .or_else(|| body.get("task"))
                    .and_then(Value::as_str),
            )
            .map_err(bad)?;
            let mut jobs = Vec::new();
            let mut tokens = 0;
            for (text, query) in &order {
                let ids = state
                    .model
                    .tokenize(&format!("{instruction}{text}"), true)
                    .map_err(bad)?;
                state.model.validate_tokens(&ids).map_err(bad)?;
                tokens += ids.len();
                let kind = if *query {
                    InputKind::Query
                } else {
                    InputKind::Document
                };
                jobs.push(Job::Encode(
                    ids,
                    Want {
                        dense: (dense != 0.0).then(|| (pooling.clone(), Some(false))),
                        sparse: sparse != 0.0,
                        multi: (multi != 0.0).then_some(kind),
                        adapter,
                    },
                ));
            }
            if tokens > state.limits.max_request_tokens {
                return Err(bad(format!(
                    "batch exceeds {} total input tokens",
                    state.limits.max_request_tokens
                )));
            }
            let outs = state.submit(jobs).await?;
            let outs = outs
                .into_iter()
                .map(|o| match o {
                    JobOut::Encode(o) => Ok(o),
                    _ => Err(internal("unexpected encoder result")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mean = state.model.mean_maxsim();
            let scores = slots
                .iter()
                .map(|&(a, b)| {
                    let (a, b) = (&outs[a], &outs[b]);
                    let mut score = 0.0;
                    if let (Some(x), Some(y)) = (&a.dense, &b.dense) {
                        score += dense * cosine(x, y);
                    }
                    if let (Some(x), Some(y)) = (&a.sparse, &b.sparse) {
                        score += sparse * encoder_heads::sparse_dot(x, y);
                    }
                    if let (Some(x), Some(y)) = (&a.multi, &b.multi) {
                        score += multi * encoder_heads::max_sim(x, y, mean);
                    }
                    score
                })
                .collect();
            Ok((scores, tokens))
        }
    }
}

async fn operation(
    state: Arc<EncoderState>,
    body: Value,
    mode: Operation,
) -> Response {
    if let Err((status, message)) = check_model(&state, &body) {
        return mode.error(status, &message);
    }
    match run_operation(&state, &body, mode).await {
        Ok(response) => Json(response).into_response(),
        Err((status, message)) => mode.error(status, &message),
    }
}

async fn run_operation(
    state: &Arc<EncoderState>,
    body: &Value,
    mode: Operation,
) -> Result<Value, Failure> {
    let caps = state.model.capabilities();
    match mode {
        Operation::Rerank | Operation::Score => {
            let (pairs, returned_docs, top_n, return_documents) = if mode == Operation::Rerank {
                let query = body["query"]
                    .as_str()
                    .filter(|v| !v.is_empty() && v.len() <= 65536)
                    .ok_or_else(|| bad("query must be non-empty text of at most 65536 bytes"))?;
                let docs = documents(body).map_err(bad)?;
                let top_n = match body.get("top_n") {
                    None | Some(Value::Null) => docs.len(),
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| *n > 0)
                        .ok_or_else(|| bad("top_n must be a positive integer"))?
                        .min(docs.len()),
                };
                let return_documents = match body.get("return_documents") {
                    None => false,
                    Some(v) => v
                        .as_bool()
                        .ok_or_else(|| bad("return_documents must be boolean"))?,
                };
                let returned = docs.iter().map(|(_, doc)| doc.clone()).collect::<Vec<_>>();
                let pairs = docs
                    .into_iter()
                    .map(|(text, _)| (query.to_owned(), text))
                    .collect::<Vec<_>>();
                (pairs, returned, top_n, return_documents)
            } else {
                (score_pairs(body).map_err(bad)?, Vec::new(), 0, false)
            };
            if caps.sequence_classification
                && state.model.labels.len() != 1
                && body.get("score_type").is_none()
                && !caps.dense
            {
                return Err(bad("rerank/score require a classifier with exactly one label"));
            }
            let (scores, tokens) = score_texts(state, &pairs, body).await?;
            let usage = json!({"prompt_tokens":tokens,"total_tokens":tokens});
            if mode == Operation::Score {
                return Ok(json!({"object":"list","model":state.name,"data":scores.iter().enumerate().map(|(index,v)|json!({"index":index,"object":"score","score":v})).collect::<Vec<_>>(),"usage":usage}));
            }
            let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let results = ranked
                .iter()
                .take(top_n)
                .map(|&(index, score)| {
                    let mut item = json!({"index":index,"relevance_score":score});
                    if return_documents {
                        item["document"] = returned_docs[index].clone();
                    }
                    item
                })
                .collect::<Vec<_>>();
            Ok(json!({"id":format!("rerank_{}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos()),"results":results,"model":state.name,"usage":usage,"meta":{"api_version":{"version":"2"},"billed_units":{"search_units":0,"input_tokens":tokens}}}))
        }
        Operation::Classify => {
            if caps.token_classification {
                return classify_tokens(state, body).await;
            }
            if !caps.sequence_classification {
                return Err((StatusCode::NOT_IMPLEMENTED, "model has no classification head; load a sequence or token classification checkpoint (BertForSequenceClassification, an XLM-RoBERTa cross-encoder, BertForTokenClassification)".into()));
            }
            let batch = text_batch(&body["input"], state.limits.max_inputs)
                .map_err(bad)?
                .into_iter()
                .map(|text| {
                    let ids = state.model.tokenize(&text, true)?;
                    state.model.validate_tokens(&ids)?;
                    let types = vec![0; ids.len()];
                    Ok((ids, types))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(bad)?;
            let tokens = batch.iter().map(|(ids, _)| ids.len()).sum::<usize>();
            if tokens > state.limits.max_request_tokens {
                return Err(bad(format!(
                    "batch exceeds {} total input tokens",
                    state.limits.max_request_tokens
                )));
            }
            let outs = state
                .submit(
                    batch
                        .into_iter()
                        .map(|(ids, types)| Job::Classify(ids, types))
                        .collect(),
                )
                .await?;
            let data = outs
                .into_iter()
                .enumerate()
                .map(|(index, o)| {
                    let JobOut::Classified { probs, logits } = o else {
                        return Err(internal("unexpected encoder result"));
                    };
                    let best = probs.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(&a.0))).map_or(0,|(id,_)|id);
                    Ok(json!({"index":index,"label":state.model.labels[best],"probs":probs,"logits":logits,"num_classes":probs.len()}))
                })
                .collect::<Result<Vec<_>, Failure>>()?;
            Ok(json!({"object":"list","model":state.name,"data":data,"usage":{"prompt_tokens":tokens,"total_tokens":tokens}}))
        }
        Operation::Pool => pooling_inner(state, body).await,
    }
}

async fn classify_tokens(state: &Arc<EncoderState>, body: &Value) -> Result<Value, Failure> {
    let texts = text_batch(&body["input"], state.limits.max_inputs).map_err(bad)?;
    let mut batch = Vec::new();
    for text in texts {
        let ids = state.model.tokenize(&text, true).map_err(bad)?;
        state.model.validate_tokens(&ids).map_err(bad)?;
        batch.push(ids);
    }
    let tokens = batch.iter().map(Vec::len).sum::<usize>();
    if tokens > state.limits.max_request_tokens {
        return Err(bad(format!(
            "batch exceeds {} total input tokens",
            state.limits.max_request_tokens
        )));
    }
    let kept = batch
        .iter()
        .map(|ids| state.model.kept_ids(ids))
        .collect::<Vec<_>>();
    let outs = state
        .submit(batch.into_iter().map(Job::Tokens).collect())
        .await?;
    let labels = state.model.token_labels();
    let data = outs
        .into_iter()
        .zip(kept)
        .enumerate()
        .map(|(index, (o, ids))| {
            let JobOut::TokenProbs(rows) = o else {
                return Err(internal("unexpected encoder result"));
            };
            let names = rows
                .iter()
                .map(|p| {
                    let best = p.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(&a.0))).map_or(0,|(id,_)|id);
                    labels[best].clone()
                })
                .collect::<Vec<_>>();
            Ok(json!({"index":index,"tokens":ids,"labels":names,"probs":rows,"num_classes":labels.len()}))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(json!({"object":"list","model":state.name,"data":data,"usage":{"prompt_tokens":tokens,"total_tokens":tokens}}))
}

/// vLLM-style `/pooling`: the raw pooled output selected by `task`.
async fn pooling_inner(state: &Arc<EncoderState>, body: &Value) -> Result<Value, Failure> {
    let caps = state.model.capabilities();
    let options = parse_options_for(body, false).map_err(bad)?;
    let task = match body.get("task").and_then(Value::as_str) {
        Some(t) => t.to_owned(),
        None if caps.dense => "embed".into(),
        None if caps.sparse => "sparse".into(),
        None if caps.multi_vector => "token_embed".into(),
        None if caps.token_classification => "token_classify".into(),
        None => "embed".into(),
    };
    let pooling = options
        .pooling
        .clone()
        .unwrap_or_else(|| state.model.default_pooling.clone());
    let (adapter, instruction) =
        resolve_adapter(&state.model, options.adapter.as_deref()).map_err(bad)?;
    let texts = text_batch(&body["input"], state.limits.max_inputs).map_err(bad)?;
    let mut batch = Vec::new();
    for text in texts {
        let mut ids = state
            .model
            .tokenize_raw(&format!("{instruction}{text}"), true)
            .map_err(bad)?;
        if let Some(limit) = options.truncate {
            ids = state
                .model
                .truncate_ids(ids, limit.unwrap_or(state.model.context));
        }
        state.model.validate_tokens(&ids).map_err(bad)?;
        batch.push(ids);
    }
    let tokens = batch.iter().map(Vec::len).sum::<usize>();
    if tokens > state.limits.max_request_tokens {
        return Err(bad(format!(
            "batch exceeds {} total input tokens",
            state.limits.max_request_tokens
        )));
    }
    let usage = json!({"prompt_tokens":tokens,"total_tokens":tokens});
    if task == "token_classify" {
        if !caps.token_classification {
            return Err((StatusCode::NOT_IMPLEMENTED, "model has no token classification head".into()));
        }
        let outs = state
            .submit(batch.into_iter().map(Job::Tokens).collect())
            .await?;
        let data = outs
            .into_iter()
            .enumerate()
            .map(|(index, o)| match o {
                JobOut::TokenProbs(rows) => Ok(json!({"index":index,"object":"pooling","data":rows})),
                _ => Err(internal("unexpected encoder result")),
            })
            .collect::<Result<Vec<_>, Failure>>()?;
        return Ok(json!({"object":"list","model":state.name,"data":data,"usage":usage}));
    }
    let mut want = Want {
        adapter,
        ..Want::default()
    };
    match task.as_str() {
        "embed" | "encode" | "classify" | "score" => {
            if !caps.dense {
                return Err(bad("model has no pooled dense output; use task sparse or token_embed"));
            }
            want.dense = Some((pooling, options.normalize));
        }
        "sparse" => {
            if !caps.sparse {
                return Err(bad("model has no sparse output"));
            }
            want.sparse = true;
        }
        "token_embed" | "multi_vector" => {
            if !caps.multi_vector {
                return Err(bad("model has no multi-vector output"));
            }
            want.multi = Some(options.input_type);
        }
        other => return Err(bad(format!("unsupported pooling task {other}"))),
    }
    let outs = state
        .submit(batch.into_iter().map(|ids| Job::Encode(ids, want.clone())).collect())
        .await?;
    let data = outs
        .into_iter()
        .enumerate()
        .map(|(index, o)| {
            let JobOut::Encode(o) = o else {
                return Err(internal("unexpected encoder result"));
            };
            let value = if let Some(v) = o.dense {
                json!(v)
            } else if let Some(s) = o.sparse {
                sparse_json(&s)
            } else {
                json!(o.multi.unwrap_or_default())
            };
            Ok(json!({"index":index,"object":"pooling","data":value}))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(json!({"object":"list","model":state.name,"data":data,"usage":usage}))
}

async fn rerank(
    State(state): State<Arc<EncoderState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, body, Operation::Rerank).await,
        Err(e) => Operation::Rerank.error(e.status(), &e.body_text()),
    }
}
async fn score(
    State(state): State<Arc<EncoderState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, body, Operation::Score).await,
        Err(e) => Operation::Score.error(e.status(), &e.body_text()),
    }
}
async fn classify(
    State(state): State<Arc<EncoderState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, body, Operation::Classify).await,
        Err(e) => Operation::Classify.error(e.status(), &e.body_text()),
    }
}
async fn pooling(
    State(state): State<Arc<EncoderState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, body, Operation::Pool).await,
        Err(e) => Operation::Pool.error(e.status(), &e.body_text()),
    }
}
pub fn serve(model: BertEncoder, config: ServerConfig) -> Result<(), String> {
    let limits = Limits::from_env();
    let model = Arc::new(model);
    let window = Duration::from_millis(
        std::env::var("FLLM_ENCODER_BATCH_WINDOW_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0),
    );
    let (jobs, receive) = std::sync::mpsc::channel::<Task>();
    {
        let model = model.clone();
        let budget = batch_token_budget();
        std::thread::Builder::new()
            .name("encoder-batcher".into())
            .spawn(move || batch_worker(model, receive, budget, window))
            .map_err(|e| e.to_string())?;
    }
    let state = Arc::new(EncoderState {
        metrics: Arc::new(crate::metrics::Metrics::new()),
        model,
        name: config.model_name,
        queue: Arc::new(tokio::sync::Semaphore::new(limits.queue)),
        limits,
        jobs,
    });
    let context = crate::server::http::HttpContext::new(config.http.clone());
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ready_flag = ready.clone();
    let app = Router::new()
        .route("/metrics", get(metrics))
        .route(
            "/health",
            get(|| async {
                Json(json!({"status":"ok","engine":"fllm-rust","architecture":"bert"}))
            }),
        )
        .route(
            "/ready",
            get(move || {
                let ready = ready_flag.clone();
                async move {
                    let up = ready.load(std::sync::atomic::Ordering::Relaxed);
                    (
                        if up { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE },
                        Json(json!({"status": if up { "ready" } else { "not_ready" }})),
                    )
                }
            }),
        )
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        .route("/v1/score", post(score))
        .route("/score", post(score))
        .route("/classify", post(classify))
        .route("/v1/classify", post(classify))
        .route("/pooling", post(pooling))
        .route("/v1/pooling", post(pooling))
        .route("/api/embed", post(embeddings))
        .route("/api/embeddings", post(embeddings))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .layer(axum::middleware::from_fn_with_state(state.metrics.clone(), crate::metrics::observe))
        .layer(axum::middleware::from_fn_with_state(context.clone(), crate::server::http::layer))
        .with_state(state);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port))
            .await
            .map_err(|e| e.to_string())?;
        ready.store(true, std::sync::atomic::Ordering::Relaxed);
        crate::server::http::serve(listener, app, context, ready.clone(), async {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("SIGTERM handler");
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
        })
        .await
    })
}

async fn metrics(State(state): State<Arc<EncoderState>>) -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], state.metrics.render()).into_response()
}

#[cfg(test)]
mod classification_api_tests {
    use super::*;
    use crate::encoder::Capabilities;
    #[test]
    fn broadcast_pairs_and_reject_mismatched_batches() {
        let pairs = score_pairs(&json!({"text_1":"q","text_2":["a","b"]})).unwrap();
        assert_eq!(
            pairs,
            vec![("q".into(), "a".into()), ("q".into(), "b".into())]
        );
        assert!(score_pairs(&json!({"text_1":["q","r"],"text_2":["a","b","c"]})).is_err());
        assert!(text_batch(&json!([]), 32).is_err());
        assert!(text_batch(&json!([1]), 32).is_err());
        assert!(text_batch(&json!("x".repeat(65537)), 32).is_err());
    }
    #[test]
    fn preserve_document_objects_and_limit_count() {
        let docs =
            documents(&json!({"documents":["plain",{"text":"text","title":"title"}]})).unwrap();
        assert_eq!(docs[0].1, json!({"text":"plain"}));
        assert_eq!(docs[1].1["title"], "title");
        assert!(documents(&json!({"documents":vec!["x";257]})).is_err());
        assert!(documents(&json!({"documents":[{"title":"missing text"}]})).is_err());
    }
    #[test]
    fn option_parsing() {
        let o = parse_options(&json!({
            "dimensions": 8, "encoding_format": "base64", "pooling": "max", "normalize": false,
            "sparse": true, "multi_vector": false, "input_type": "query", "truncate_prompt_tokens": 16
        }))
        .unwrap();
        assert_eq!(o.dimensions, Some(8));
        assert!(o.base64);
        assert_eq!(o.pooling.as_deref(), Some("max"));
        assert_eq!(o.normalize, Some(false));
        assert_eq!(o.sparse, Some(true));
        assert_eq!(o.input_type, InputKind::Query);
        assert_eq!(o.truncate, Some(Some(16)));
        let d = parse_options(&json!({})).unwrap();
        assert_eq!(d.truncate, None);
        assert_eq!(d.input_type, InputKind::Document);
        assert_eq!(
            parse_options(&json!({"truncate_prompt_tokens": -1})).unwrap().truncate,
            Some(None)
        );
        assert!(parse_options(&json!({"truncate_prompt_tokens": 0})).is_err());
        assert!(parse_options(&json!({"encoding_format": "hex"})).is_err());
        assert!(parse_options(&json!({"dimensions": 0})).is_err());
        assert!(parse_options(&json!({"pooling": "nope"})).is_err());
        assert!(parse_options(&json!({"normalize": "yes"})).is_err());
        assert!(parse_options(&json!({"input_type": "x"})).is_err());
    }
    #[test]
    fn output_defaults_follow_capabilities() {
        let options = parse_options(&json!({})).unwrap();
        let dense = Capabilities { dense: true, sparse: true, multi_vector: true, ..Default::default() };
        let want = wanted_outputs(dense, &options, "cls").unwrap();
        assert!(want.dense.is_some() && !want.sparse && want.multi.is_none());
        let splade = Capabilities { sparse: true, ..Default::default() };
        let want = wanted_outputs(splade, &options, "cls").unwrap();
        assert!(want.dense.is_none() && want.sparse);
        let colbert = Capabilities { multi_vector: true, ..Default::default() };
        assert!(wanted_outputs(colbert, &options, "cls").unwrap().multi.is_some());
        let only_dense = Capabilities { dense: true, ..Default::default() };
        let asks = parse_options(&json!({"sparse": true})).unwrap();
        assert!(wanted_outputs(only_dense, &asks, "cls").is_err());
    }
    #[test]
    fn score_mode_selection() {
        let cross = Capabilities { dense: false, sequence_classification: true, ..Default::default() };
        assert_eq!(score_spec(cross, 1, &json!({})).unwrap(), ScoreSpec::CrossEncoder);
        assert!(score_spec(cross, 3, &json!({})).is_err());
        let m3 = Capabilities { dense: true, sparse: true, multi_vector: true, ..Default::default() };
        assert_eq!(
            score_spec(m3, 0, &json!({})).unwrap(),
            ScoreSpec::Weighted { dense: 1.0, sparse: 0.0, multi: 0.0 }
        );
        assert_eq!(
            score_spec(m3, 0, &json!({"score_type":"combined"})).unwrap(),
            ScoreSpec::Weighted { dense: 0.4, sparse: 0.2, multi: 0.4 }
        );
        assert!(score_spec(Capabilities { dense: true, ..Default::default() }, 0, &json!({"score_type":"sparse"})).is_err());
    }
}
