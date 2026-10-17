//! Offline batches and the file store behind them, following the OpenAI `/v1/batches` and
//! `/v1/files` API. Uploads are written into a server-owned directory under server-assigned
//! identifiers; a batch replays the JSONL lines of its input file in process against the
//! chat-completions and embeddings handlers with bounded concurrency, and reports progress
//! through `/v1/batches/{id}` plus an output and an error file in OpenAI's result format.
use super::*;
use axum::extract::Path as AxumPath;
use futures_util::future::join_all;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Mutex, OnceLock},
};

struct Settings {
    files_dir: PathBuf,
    max_file_size: u64,
    batch_concurrency: usize,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| Settings {
        files_dir: std::env::var_os("FLLM_FILES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| crate::hub::cache_root().join("fllm-files")),
        max_file_size: std::env::var("FLLM_MAX_FILE_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(512 * 1024 * 1024),
        batch_concurrency: std::env::var("FLLM_BATCH_CONCURRENCY")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2),
    })
}

/// Body limit of the upload route; the server-wide default is deliberately smaller.
pub(in crate::server) fn max_upload_bytes() -> usize {
    settings().max_file_size.min(usize::MAX as u64) as usize
}

fn error(status: StatusCode, message: &str) -> Response {
    api_error(status, message)
}

fn detail(status: StatusCode, message: &str, param: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":"invalid_request_error","param":param,"code":param}})),
    )
        .into_response()
}

/// Identifiers stay inside the files directory: only characters the server itself generates are
/// accepted from the URL path.
fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 96
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn stored_path(id: &str, suffix: &str) -> PathBuf {
    settings().files_dir.join(format!("{id}{suffix}"))
}

fn file_object(id: &str, filename: &str, purpose: &str, bytes: u64) -> Value {
    json!({
        "id": id,
        "object": "file",
        "bytes": bytes,
        "created_at": unix_now(),
        "filename": filename,
        "purpose": purpose,
        "status": "processed"
    })
}

/// Files and batches as persisted by the store; plain JSON so the on-disk state has the same
/// shape as the API replies.
struct Store {
    files: BTreeMap<String, Value>,
    batches: BTreeMap<String, Value>,
}

impl Store {
    fn load() -> Self {
        let mut store = match fs::read_to_string(stored_path("store", ".json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        {
            Some(Value::Object(map)) => Store {
                files: map
                    .get("files")
                    .and_then(Value::as_object)
                    .map(|files| files.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
                batches: map
                    .get("batches")
                    .and_then(Value::as_object)
                    .map(|batches| {
                        batches.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    })
                    .unwrap_or_default(),
            },
            _ => Store {
                files: BTreeMap::new(),
                batches: BTreeMap::new(),
            },
        };
        if fs::create_dir_all(&settings().files_dir).is_err() {
            return store;
        }
        // A batch that was running when the process stopped cannot be resumed: the input file is
        // still there, but the caller has to submit it again.
        let mut changed = false;
        for batch in store.batches.values_mut() {
            if matches!(
                batch["status"].as_str(),
                Some("validating" | "in_progress" | "cancelling")
            ) {
                batch["status"] = json!("failed");
                batch["failed_at"] = json!(unix_now());
                batch["errors"] = json!({
                    "object": "list",
                    "data": [{"code": "server_restarted", "message": "batch was interrupted by a server restart", "line": Value::Null, "param": Value::Null}]
                });
                changed = true;
            }
        }
        if changed {
            store.persist();
        }
        store
    }

    fn persist(&self) {
        let files: Map<String, Value> =
            self.files.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let batches: Map<String, Value> = self
            .batches
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let text = json!({"version": 1, "files": files, "batches": batches}).to_string();
        let tmp = stored_path("store", ".json.tmp");
        if fs::write(&tmp, text).is_ok() {
            let _ = fs::rename(&tmp, stored_path("store", ".json"));
        }
    }
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(Store::load()))
}

fn cancel_flags() -> &'static Mutex<BTreeMap<String, Arc<AtomicBool>>> {
    static FLAGS: OnceLock<Mutex<BTreeMap<String, Arc<AtomicBool>>>> = OnceLock::new();
    FLAGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

// ---------------------------------------------------------------------------------------------
// Multipart upload

struct Part {
    name: String,
    filename: Option<String>,
    data: Vec<u8>,
}

/// Minimal multipart reader: enough for the `purpose` and `file` fields of a file upload.
fn parse_multipart(content_type: &str, body: &[u8]) -> Option<Vec<Part>> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|piece| piece.strip_prefix("boundary="))?
        .trim_matches('"')
        .to_owned();
    if boundary.is_empty() {
        return None;
    }
    let delimiter = format!("\r\n--{boundary}").into_bytes();
    // the first boundary has no CRLF in front of it; add one so every part is found
    let mut prefixed = Vec::with_capacity(body.len() + 2);
    prefixed.extend_from_slice(b"\r\n");
    prefixed.extend_from_slice(body);
    let body: &[u8] = &prefixed;
    let mut sections: Vec<&[u8]> = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index + delimiter.len() <= body.len() {
        if body[index..index + delimiter.len()] == delimiter[..] {
            sections.push(&body[start..index]);
            index += delimiter.len();
            start = index;
        } else {
            index += 1;
        }
    }
    sections.push(&body[start..]);
    let mut parts = Vec::new();
    for section in sections.into_iter().skip(1) {
        // The closing delimiter is followed by "--".
        let section = section.strip_prefix(b"--").unwrap_or(section);
        let section = section.strip_prefix(b"\r\n").unwrap_or(section);
        let Some(split) = section.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&section[..split]);
        let mut name = String::new();
        let mut filename = None;
        for line in headers.lines() {
            if !line.to_ascii_lowercase().starts_with("content-disposition") {
                continue;
            }
            for piece in line.split(';').map(str::trim) {
                if let Some(value) = piece.strip_prefix("name=") {
                    name = value.trim_matches('"').to_owned();
                } else if let Some(value) = piece.strip_prefix("filename=") {
                    filename = Some(value.trim_matches('"').to_owned());
                }
            }
        }
        let data = section[split + 4..].strip_suffix(b"\r\n").unwrap_or(&section[split + 4..]);
        parts.push(Part {
            name,
            filename,
            data: data.to_vec(),
        });
    }
    Some(parts)
}

/// The uploaded name never reaches the file system; it is only reported back as metadata.
fn sanitize_filename(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches('.').to_owned();
    if cleaned.is_empty() {
        "upload.jsonl".to_owned()
    } else {
        cleaned.chars().take(128).collect()
    }
}

pub(in crate::server) async fn upload(
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let Some(parts) = parse_multipart(&content_type, &body) else {
        return detail(
            StatusCode::BAD_REQUEST,
            "expected multipart/form-data with a file field",
            "file",
        );
    };
    let purpose = parts
        .iter()
        .find(|part| part.name == "purpose")
        .map(|part| String::from_utf8_lossy(&part.data).trim().to_owned())
        .filter(|purpose| !purpose.is_empty())
        .unwrap_or_else(|| "batch".to_owned());
    let Some(file) = parts.iter().find(|part| part.name == "file") else {
        return detail(
            StatusCode::BAD_REQUEST,
            "multipart form field \"file\" is required",
            "file",
        );
    };
    if file.data.is_empty() {
        return detail(StatusCode::BAD_REQUEST, "uploaded file is empty", "file");
    }
    if file.data.len() as u64 > settings().max_file_size {
        return detail(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!(
                "file is larger than the configured limit of {} bytes",
                settings().max_file_size
            ),
            "file",
        );
    }
    if fs::create_dir_all(&settings().files_dir).is_err() {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot create the files directory",
        );
    }
    let id = new_id("file");
    if fs::write(stored_path(&id, ".jsonl"), &file.data).is_err() {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot store the uploaded file",
        );
    }
    let record = file_object(
        &id,
        &sanitize_filename(file.filename.as_deref().unwrap_or("upload.jsonl")),
        &purpose,
        file.data.len() as u64,
    );
    let mut store = security::lock(store());
    store.files.insert(id.clone(), record.clone());
    store.persist();
    (StatusCode::OK, Json(record)).into_response()
}

// ---------------------------------------------------------------------------------------------
// Files

pub(in crate::server) async fn list_files(
) -> Response {
    let store = security::lock(store());
    let mut data: Vec<Value> = store.files.values().cloned().collect();
    data.sort_by_key(|file| std::cmp::Reverse(file["created_at"].as_u64().unwrap_or(0)));
    (StatusCode::OK, Json(json!({"object": "list", "data": data}))).into_response()
}

pub(in crate::server) async fn get_file(
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !safe_id(&id) {
        return detail(StatusCode::BAD_REQUEST, "invalid file id", "id");
    }
    match security::with_lock(store(), |store| store.files.get(&id).cloned())
    {
        Some(file) => (StatusCode::OK, Json(file)).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such file"),
    }
}

pub(in crate::server) async fn file_content(
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !safe_id(&id) {
        return detail(StatusCode::BAD_REQUEST, "invalid file id", "id");
    }
    let record = match security::with_lock(store(), |store| store.files.get(&id).cloned())
    {
        Some(record) => record,
        None => return error(StatusCode::NOT_FOUND, "no such file"),
    };
    let suffix = if record["purpose"] == json!("batch_output") {
        "_output.jsonl"
    } else if record["purpose"] == json!("batch_error") {
        "_error.jsonl"
    } else {
        ".jsonl"
    };
    match fs::read(stored_path(&id, suffix)) {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
            bytes,
        )
            .into_response(),
        Err(_) => error(
            StatusCode::NOT_FOUND,
            "file content is no longer available",
        ),
    }
}

pub(in crate::server) async fn delete_file(
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !safe_id(&id) {
        return detail(StatusCode::BAD_REQUEST, "invalid file id", "id");
    }
    let mut store = security::lock(store());
    if store.files.remove(&id).is_none() {
        return error(StatusCode::NOT_FOUND, "no such file");
    }
    for suffix in [".jsonl", "_output.jsonl", "_error.jsonl"] {
        let _ = fs::remove_file(stored_path(&id, suffix));
    }
    store.persist();
    (
        StatusCode::OK,
        Json(json!({"id": id, "object": "file", "deleted": true})),
    )
        .into_response()
}

// ---------------------------------------------------------------------------------------------
// Batches

struct BatchLine {
    custom_id: String,
    url: String,
    body: Value,
}

/// Endpoints a batch may address; `/v1/responses` runs on the chat-completions pipeline.
fn endpoint_of(url: &str) -> Option<&'static str> {
    match url {
        "/v1/chat/completions" | "/chat/completions" | "/v1/responses" => Some("chat"),
        "/v1/completions" | "/completions" => Some("completions"),
        "/v1/embeddings" | "/embeddings" => Some("embeddings"),
        _ => None,
    }
}

#[allow(clippy::result_large_err)]
fn parse_lines(text: &str, default_url: &str) -> Result<Vec<BatchLine>, Response> {
    let mut lines = Vec::new();
    let mut seen = BTreeSet::new();
    for (number, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let line = number + 1;
        let value: Value = serde_json::from_str(raw).map_err(|e| {
            detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} is not valid JSON: {e}"),
                "body",
            )
        })?;
        let Some(object) = value.as_object() else {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} must be a JSON object"),
                "body",
            ));
        };
        let custom_id = match object.get("custom_id") {
            Some(Value::String(id)) if !id.is_empty() && id.len() <= 64 => id.clone(),
            _ => {
                return Err(detail(
                    StatusCode::BAD_REQUEST,
                    &format!("line {line} needs a custom_id of at most 64 characters"),
                    "custom_id",
                ))
            }
        };
        if !seen.insert(custom_id.clone()) {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} repeats the custom_id {custom_id:?}"),
                "custom_id",
            ));
        }
        if object.get("method").and_then(Value::as_str) != Some("POST")
            && object.get("method").is_some()
        {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} must use the POST method"),
                "method",
            ));
        }
        let url = object
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| default_url.to_owned());
        if endpoint_of(&url).is_none() {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} targets the unsupported endpoint {url}"),
                "url",
            ));
        }
        let mut body = object.get("body").cloned().unwrap_or(Value::Null);
        if !body.is_object() {
            return Err(detail(
                StatusCode::BAD_REQUEST,
                &format!("line {line} must carry an object body"),
                "body",
            ));
        }
        // Results are written to files, so a streamed reply is never what the caller wants.
        body["stream"] = json!(false);
        lines.push(BatchLine {
            custom_id,
            url,
            body,
        });
    }
    if lines.is_empty() {
        return Err(detail(
            StatusCode::BAD_REQUEST,
            "input file contains no requests",
            "input_file_id",
        ));
    }
    Ok(lines)
}

fn batch_object(lines: usize) -> Value {
    json!({
        "id": new_id("batch"),
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": Value::Null,
        "input_file_id": "",
        "completion_window": "24h",
        "status": "validating",
        "output_file_id": Value::Null,
        "error_file_id": Value::Null,
        "output_lines": 0,
        "error_lines": 0,
        "created_at": unix_now(),
        "in_progress_at": Value::Null,
        "expires_at": Value::Null,
        "finalizing_at": Value::Null,
        "completed_at": Value::Null,
        "failed_at": Value::Null,
        "cancelled_at": Value::Null,
        "request_counts": {"total": lines, "completed": 0, "failed": 0},
        "metadata": Value::Null
    })
}

pub(in crate::server) async fn create_batch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let input_file_id = match body.get("input_file_id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() => id.to_owned(),
        _ => {
            return detail(
                StatusCode::BAD_REQUEST,
                "input_file_id is required",
                "input_file_id",
            )
        }
    };
    let endpoint = body
        .get("endpoint")
        .and_then(Value::as_str)
        .unwrap_or("/v1/chat/completions")
        .to_owned();
    if endpoint_of(&endpoint).is_none() {
        return detail(
            StatusCode::BAD_REQUEST,
            &format!("endpoint {endpoint} is not supported"),
            "endpoint",
        );
    }
    let completion_window = match body.get("completion_window") {
        None | Some(Value::Null) => "24h".to_owned(),
        Some(Value::String(window)) if !window.is_empty() => window.clone(),
        Some(_) => {
            return detail(
                StatusCode::BAD_REQUEST,
                "completion_window must be a string",
                "completion_window",
            )
        }
    };
    let metadata = match body.get("metadata") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::Object(map)) if map.len() <= 16 => Value::Object(map.clone()),
        Some(_) => {
            return detail(
                StatusCode::BAD_REQUEST,
                "metadata must be an object with at most 16 entries",
                "metadata",
            )
        }
    };
    let record = match security::with_lock(store(), |store| store.files.get(&input_file_id).cloned())
    {
        Some(record) => record,
        None => return detail(StatusCode::NOT_FOUND, "no such input file", "input_file_id"),
    };
    let input_id = record["id"].as_str().unwrap_or_default().to_owned();
    let text = match fs::read_to_string(stored_path(&input_id, ".jsonl")) {
        Ok(text) => text,
        Err(_) => {
            return error(
                StatusCode::NOT_FOUND,
                "input file content is no longer available",
            )
        }
    };
    let lines = match parse_lines(&text, &endpoint) {
        Ok(lines) => lines,
        Err(response) => return response,
    };
    let mut batch = batch_object(lines.len());
    let id = batch["id"].as_str().unwrap_or_default().to_owned();
    batch["endpoint"] = json!(endpoint);
    batch["input_file_id"] = json!(input_id);
    batch["completion_window"] = json!(completion_window);
    batch["metadata"] = metadata;
    batch["status"] = json!("in_progress");
    batch["in_progress_at"] = json!(unix_now());
    batch["expires_at"] = json!(unix_now() + 24 * 3600);
    {
        let mut store = security::lock(store());
        store.batches.insert(id.clone(), batch.clone());
        store.persist();
    }
    spawn_runner(state, id, lines);
    (StatusCode::OK, Json(batch)).into_response()
}

pub(in crate::server) async fn list_batches(
) -> Response {
    let store = security::lock(store());
    let mut data: Vec<Value> = store.batches.values().cloned().collect();
    data.sort_by_key(|batch| std::cmp::Reverse(batch["created_at"].as_u64().unwrap_or(0)));
    (StatusCode::OK, Json(json!({"object": "list", "data": data}))).into_response()
}

pub(in crate::server) async fn get_batch(
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !safe_id(&id) {
        return detail(StatusCode::BAD_REQUEST, "invalid batch id", "id");
    }
    match security::with_lock(store(), |store| store.batches.get(&id).cloned())
    {
        Some(batch) => (StatusCode::OK, Json(batch)).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such batch"),
    }
}

pub(in crate::server) async fn cancel_batch(
    AxumPath(id): AxumPath<String>,
) -> Response {
    if !safe_id(&id) {
        return detail(StatusCode::BAD_REQUEST, "invalid batch id", "id");
    }
    let flag = security::with_lock(cancel_flags(), |flags| flags.get(&id).cloned());
    let mut store = security::lock(store());
    let Some(batch) = store.batches.get(&id).cloned() else {
        return error(StatusCode::NOT_FOUND, "no such batch");
    };
    match batch["status"].as_str() {
        Some("in_progress" | "validating") => {
            match &flag {
                // The runner notices the flag between chunks of requests and stops there.
                Some(flag) => {
                    flag.store(true, Ordering::Relaxed);
                    if let Some(batch) = store.batches.get_mut(&id) {
                        batch["status"] = json!("cancelling");
                    }
                }
                None => {
                    if let Some(batch) = store.batches.get_mut(&id) {
                        batch["status"] = json!("cancelled");
                        batch["cancelled_at"] = json!(unix_now());
                    }
                }
            }
        }
        Some("cancelling") => {}
        Some("completed" | "failed" | "cancelled") => {
            return detail(
                StatusCode::CONFLICT,
                "batch has already finished",
                "status",
            )
        }
        _ => {
            return detail(
                StatusCode::CONFLICT,
                "batch is not running",
                "status",
            )
        }
    }
    store.persist();
    let batch = store.batches.get(&id).cloned().unwrap_or(batch);
    (StatusCode::OK, Json(batch)).into_response()
}

// ---------------------------------------------------------------------------------------------
// Runner

/// Runs one batch line in process and returns the OpenAI batch result object.
async fn run_line(state: Arc<AppState>, line: &BatchLine, index: usize) -> Value {
    let body = Json(line.body.clone());
    let response = match endpoint_of(&line.url).unwrap_or("chat") {
        "chat" => chat_completions(State(state), body).await,
        "completions" => completions(State(state), body).await,
        _ => embeddings(State(state), body).await,
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 << 20)
        .await
        .unwrap_or_default();
    let parsed = serde_json::from_slice::<Value>(&bytes).ok().unwrap_or_else(|| {
        json!({"error": {"message": String::from_utf8_lossy(&bytes).into_owned()}})
    });
    let id = new_id("batch_req");
    if status.is_success() {
        json!({
            "id": id,
            "custom_id": line.custom_id,
            "response": {"status_code": status.as_u16(), "request_id": new_id("req"), "body": parsed},
            "error": Value::Null
        })
    } else {
        let message = parsed["error"]["message"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("request failed with status {}", status.as_u16()));
        json!({
            "id": id,
            "custom_id": line.custom_id,
            "response": Value::Null,
            "error": {"code": Value::Null, "message": message, "param": Value::Null, "line": index + 1}
        })
    }
}

fn count_results(id: &str, completed: usize, failed: usize) {
    {
        let mut store = security::lock(store());
        if let Some(batch) = store.batches.get_mut(id) {
            batch["request_counts"]["completed"] = json!(completed);
            batch["request_counts"]["failed"] = json!(failed);
        }
        store.persist();
    }
}

/// Records the terminal state of a batch together with its result files.
fn finish(
    id: &str,
    status: &str,
    output: Option<(String, u64, usize)>,
    errors: Option<(String, u64, usize)>,
    completed: usize,
    failed: usize,
) {
    let now = unix_now();
    let mut store = security::lock(store());
    let mut output_id = Value::Null;
    let mut output_lines = 0usize;
    if let Some((file_id, bytes, lines)) = output {
        store.files.insert(
            file_id.clone(),
            file_object(
                &file_id,
                &format!("{id}_output.jsonl"),
                "batch_output",
                bytes,
            ),
        );
        output_id = json!(file_id);
        output_lines = lines;
    }
    let mut error_id = Value::Null;
    let mut error_lines = 0usize;
    if let Some((file_id, bytes, lines)) = errors {
        store.files.insert(
            file_id.clone(),
            file_object(
                &file_id,
                &format!("{id}_error.jsonl"),
                "batch_error",
                bytes,
            ),
        );
        error_id = json!(file_id);
        error_lines = lines;
    }
    if let Some(batch) = store.batches.get_mut(id) {
        batch["status"] = json!(status);
        match status {
            "completed" => batch["completed_at"] = json!(now),
            "failed" => batch["failed_at"] = json!(now),
            "cancelled" => batch["cancelled_at"] = json!(now),
            _ => {}
        }
        batch["request_counts"]["completed"] = json!(completed);
        batch["request_counts"]["failed"] = json!(failed);
        batch["output_file_id"] = output_id;
        batch["output_lines"] = json!(output_lines);
        batch["error_file_id"] = error_id;
        batch["error_lines"] = json!(error_lines);
    }
    store.persist();
}

fn spawn_runner(state: Arc<AppState>, id: String, lines: Vec<BatchLine>) {
    let flag = Arc::new(AtomicBool::new(false));
    {
        let mut flags = security::lock(cancel_flags());
        flags.insert(id.clone(), flag.clone());
    }
    tokio::spawn(async move {
        run_batch(state, id, lines, flag).await;
    });
}

async fn run_batch(state: Arc<AppState>, id: String, lines: Vec<BatchLine>, flag: Arc<AtomicBool>) {
    let output_id = new_id("file");
    let error_id = new_id("file");
    let (mut out, mut err) = match (
        fs::File::create(stored_path(&output_id, "_output.jsonl")),
        fs::File::create(stored_path(&error_id, "_error.jsonl")),
    ) {
        (Ok(out), Ok(err)) => (BufWriter::new(out), BufWriter::new(err)),
        _ => {
            finish(&id, "failed", None, None, 0, lines.len());
            return;
        }
    };
    let mut out_bytes = 0u64;
    let mut err_bytes = 0u64;
    let mut out_lines = 0usize;
    let mut err_lines = 0usize;
    let mut completed = 0usize;
    let mut failed = 0usize;
    let concurrency = settings().batch_concurrency.max(1);
    let mut index = 0usize;
    while index < lines.len() {
        if flag.load(Ordering::Relaxed) {
            break;
        }
        let end = (index + concurrency).min(lines.len());
        let results = join_all(
            lines[index..end]
                .iter()
                .map(|line| run_line(Arc::clone(&state), line, index)),
        )
        .await;
        for result in results {
            let line = result.to_string();
            if result["error"].is_null() {
                let _ = writeln!(out, "{line}");
                out_bytes += line.len() as u64 + 1;
                out_lines += 1;
                completed += 1;
            } else {
                let _ = writeln!(err, "{line}");
                err_bytes += line.len() as u64 + 1;
                err_lines += 1;
                failed += 1;
            }
        }
        let _ = out.flush();
        let _ = err.flush();
        index = end;
        count_results(&id, completed, failed);
    }
    let _ = out.flush();
    let _ = err.flush();
    {
        let mut flags = security::lock(cancel_flags());
        flags.remove(&id);
    }
    let output = (out_lines > 0).then_some((output_id, out_bytes, out_lines));
    let errors = (err_lines > 0).then_some((error_id, err_bytes, err_lines));
    let status = if flag.load(Ordering::Relaxed) {
        "cancelled"
    } else if completed == 0 && failed > 0 {
        "failed"
    } else {
        "completed"
    };
    finish(&id, status, output, errors, completed, failed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_fields_are_split_out() {
        let body = b"--X\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--X\r\nContent-Disposition: form-data; name=\"file\"; filename=\"my file.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n{\"a\":1}\r\n--X--\r\n";
        let parts = parse_multipart("multipart/form-data; boundary=X", body).expect("parts");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "purpose");
        assert_eq!(parts[0].data, b"batch");
        assert_eq!(parts[1].name, "file");
        assert_eq!(parts[1].filename.as_deref(), Some("my file.jsonl"));
        assert_eq!(parts[1].data, b"{\"a\":1}");
    }

    #[test]
    fn filenames_never_escape_the_files_directory() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("C:\\tmp\\batch.jsonl"), "batch.jsonl");
        assert_eq!(sanitize_filename(""), "upload.jsonl");
    }

    #[test]
    fn identifiers_from_the_url_path_are_restricted() {
        assert!(safe_id("file_0123456789abcdef0000"));
        assert!(!safe_id(".."));
        assert!(!safe_id("a/b"));
        assert!(!safe_id(""));
    }

    #[test]
    fn lines_need_a_custom_id_and_a_known_endpoint() {
        let ok = parse_lines(
            "{\"custom_id\":\"a\",\"url\":\"/v1/chat/completions\",\"body\":{\"messages\":[]}}\n",
            "/v1/chat/completions",
        )
        .expect("valid line");
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].body["stream"], json!(false));
        assert!(parse_lines("{}\n", "/v1/chat/completions").is_err());
        assert!(parse_lines(
            "{\"custom_id\":\"a\",\"url\":\"/v1/nope\",\"body\":{}}\n",
            "/v1/chat/completions"
        )
        .is_err());
        assert!(parse_lines(
            "{\"custom_id\":\"a\"}\n{\"custom_id\":\"a\"}\n",
            "/v1/completions"
        )
        .is_err());
    }
}