//! Embeddings for the document search of the Studio: the chat server itself does not embed (a chat model usually cannot),
//! so one small setting points at a running embeddings server — `fllm-rs serve --encoder <model>` or any OpenAI-compatible
//! `/v1/embeddings` — and the page asks this route for vectors. Without a setting the search stays keyword-only (BM25) and
//! says so; nothing is pretended.
//!
//! `embeddings.json` next to `studio.json`: `{"url": "http://127.0.0.1:8100"}`.
use super::*;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;

const MAX_INPUTS: usize = 64;
const MAX_TEXT: usize = 20_000;
const TIMEOUT: Duration = Duration::from_secs(60);

fn file() -> PathBuf {
    crate::hub::studio_data_file("embeddings.json")
}

fn read_url() -> String {
    let Ok(text) = std::fs::read_to_string(file()) else { return String::new() };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return String::new() };
    value.get("url").and_then(Value::as_str).unwrap_or("").trim().to_owned()
}

/// A URL without a trailing slash; only http and https.
fn clean_url(raw: &str) -> Result<String, String> {
    let text = raw.trim().trim_end_matches('/').to_owned();
    if text.is_empty() {
        return Ok(String::new());
    }
    if !(text.starts_with("http://") || text.starts_with("https://")) {
        return Err("the address has to start with http:// or https://".to_owned());
    }
    if text.len() > 300 {
        return Err("the address is too long".to_owned());
    }
    Ok(text)
}

/// The setting for the page: the address and whether it answers.
pub(super) fn status_json() -> Value {
    let url = read_url();
    json!({"url": url, "configured": !url.is_empty(), "file": file().display().to_string()})
}

/// `POST /api/studio/embeddings` `{url}`: empty removes the setting.
pub(super) fn save(body: &Value) -> Result<Value, String> {
    let url = clean_url(body.get("url").and_then(Value::as_str).unwrap_or(""))?;
    let path = file();
    if url.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(status_json());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(&json!({"url": url})).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(status_json())
}

/// Turns whatever the server answered into plain vectors, so the page only ever sees numbers.
fn vectors_from(value: &Value) -> Result<Vec<Vec<f32>>, String> {
    let data = value.get("data").and_then(Value::as_array).ok_or_else(|| "the embeddings server sent no data array".to_owned())?;
    let mut out = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item.get("embedding").and_then(Value::as_array).ok_or_else(|| "an item has no embedding".to_owned())?;
        let mut vector = Vec::with_capacity(embedding.len());
        for number in embedding {
            let value = number.as_f64().ok_or_else(|| "an embedding holds something that is no number".to_owned())?;
            vector.push(value as f32);
        }
        if vector.is_empty() {
            return Err("an embedding is empty".to_owned());
        }
        out.push(vector);
    }
    Ok(out)
}

/// `POST /api/studio/embed` `{input: [text, ...]}` → `{vectors: [[f32, ...], ...], model, dims}`.
pub(super) fn embed(body: &Value) -> Result<Value, String> {
    embed_with(&read_url(), body)
}

fn embed_with(url: &str, body: &Value) -> Result<Value, String> {
    let url = url.trim().trim_end_matches('/').to_owned();
    if url.is_empty() {
        return Err("no embeddings server is set; add one in Settings > Search (for example `fllm-rs serve --encoder <model>`)".to_owned());
    }
    let inputs = body.get("input").and_then(Value::as_array).cloned().ok_or_else(|| "input has to be an array of texts".to_owned())?;
    if inputs.is_empty() {
        return Ok(json!({"vectors": [], "dims": 0}));
    }
    if inputs.len() > MAX_INPUTS {
        return Err(format!("at most {MAX_INPUTS} texts per call"));
    }
    let mut texts = Vec::with_capacity(inputs.len());
    for item in &inputs {
        let text = item.as_str().ok_or_else(|| "every input has to be a text".to_owned())?;
        texts.push(text.chars().take(MAX_TEXT).collect::<String>());
    }
    let client = reqwest::blocking::Client::builder().timeout(TIMEOUT).build().map_err(|e| format!("cannot build the client: {e}"))?;
    let response = client
        .post(format!("{url}/v1/embeddings"))
        .json(&json!({"input": texts, "model": "studio"}))
        .send()
        .map_err(|e| format!("the embeddings server at {url} did not answer: {e}"))?;
    let status = response.status();
    let text = response.text().map_err(|e| format!("cannot read the answer: {e}"))?;
    if !status.is_success() {
        let detail: String = text.chars().take(300).collect();
        return Err(format!("the embeddings server answered {status}: {detail}"));
    }
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("the answer is no JSON: {e}"))?;
    let vectors = vectors_from(&value)?;
    let dims = vectors.first().map(|v| v.len()).unwrap_or(0);
    let model = value.get("model").and_then(Value::as_str).unwrap_or("").to_owned();
    Ok(json!({"vectors": vectors, "dims": dims, "model": model}))
}

/// `GET /api/studio/embeddings`: the setting for the page.
pub(in crate::server) async fn route_status() -> Response {
    Json(status_json()).into_response()
}

/// `POST /api/studio/embeddings` `{url}`.
pub(in crate::server) async fn route_save(Json(body): Json<Value>) -> Response {
    match save(&body) {
        Ok(value) => Json(value).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

/// `POST /api/studio/embed` `{input: [...]}`.
pub(in crate::server) async fn route_embed(Json(body): Json<Value>) -> Response {
    match tokio::task::spawn_blocking(move || embed(&body)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, &e),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_trimmed_and_checked() {
        assert_eq!(clean_url(" http://127.0.0.1:8100/ ").unwrap(), "http://127.0.0.1:8100");
        assert_eq!(clean_url("").unwrap(), "");
        assert!(clean_url("ftp://x").is_err());
        assert!(clean_url("127.0.0.1:8100").is_err());
        assert!(clean_url(&format!("https://{}", "a".repeat(400))).is_err());
    }

    #[test]
    fn vectors_are_read_out_of_the_openai_shape() {
        let value = json!({"data": [{"embedding": [1.0, 2.5]}, {"embedding": [0, -1]}]});
        let vectors = vectors_from(&value).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 2.5], vec![0.0, -1.0]]);
    }

    #[test]
    fn a_broken_answer_is_an_error_not_an_empty_vector() {
        assert!(vectors_from(&json!({"data": [{"nope": []}]})).is_err());
        assert!(vectors_from(&json!({"data": [{"embedding": ["x"]}]})).is_err());
        assert!(vectors_from(&json!({"data": [{"embedding": []}]})).is_err());
        assert!(vectors_from(&json!({})).is_err());
    }

    #[test]
    fn without_a_server_the_page_is_told_so() {
        let error = embed_with("", &json!({"input": ["a"]})).unwrap_err();
        assert!(error.contains("embeddings server"), "{error}");
    }

    #[test]
    fn more_than_the_limit_is_refused_before_a_request() {
        let inputs: Vec<Value> = (0..MAX_INPUTS + 1).map(|_| json!("x")).collect();
        let error = embed_with("http://127.0.0.1:9", &json!({"input": inputs})).unwrap_err();
        assert!(error.contains("at most"), "{error}");
    }
}
