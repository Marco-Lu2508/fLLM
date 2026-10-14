//! Dedicated encoder HTTP serving. Computation is exclusively native Rust.
use crate::{encoder::BertEncoder, server::ServerConfig};
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use serde_json::{json, Value};
use std::sync::Arc;

struct EncoderState {
    metrics: Arc<crate::metrics::Metrics>,
    model: BertEncoder,
    name: String,
    key: Option<String>,
    admission: Arc<tokio::sync::Semaphore>,
}
fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":if status.is_server_error(){"server_error"}else{"invalid_request_error"}}})),
    )
        .into_response()
}
fn authorized(headers: &HeaderMap, state: &EncoderState) -> bool {
    state.key.as_ref().is_none_or(|key| {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.strip_prefix("Bearer ") == Some(key.as_str()))
    })
}
async fn models(State(state): State<Arc<EncoderState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    Json(json!({"object":"list","data":[{"id":state.name,"object":"model","owned_by":"fllm","architecture":"bert","embedding_dimensions":state.model.hidden}]})).into_response()
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
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
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
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    match token_ids(&body["tokens"]).and_then(|ids| state.model.detokenize(&ids)) {
        Ok(content) => Json(json!({"content":content})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}
async fn embeddings(
    State(state): State<Arc<EncoderState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    if body
        .get("model")
        .is_some_and(|m| m.as_str() != Some(state.name.as_str()))
    {
        return error(StatusCode::BAD_REQUEST, "requested model is not loaded");
    }
    let dimensions = match body.get("dimensions") {
        None => state.model.hidden,
        Some(v) => match v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0 && n <= state.model.hidden)
        {
            Some(n) => n,
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "dimensions must be between 1 and encoder hidden size",
                )
            }
        },
    };
    let format = match body.get("encoding_format") {
        None => "float",
        Some(v) => match v.as_str() {
            Some(v) => v,
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "encoding_format must be float or base64",
                )
            }
        },
    };
    if !matches!(format, "float" | "base64") {
        return error(
            StatusCode::BAD_REQUEST,
            "encoding_format must be float or base64",
        );
    }
    let pooling = body
        .get("pooling")
        .and_then(Value::as_str)
        .unwrap_or(&state.model.default_pooling)
        .to_owned();
    if let Err(e) = crate::encoder::validate_pooling(&pooling) {
        return error(StatusCode::BAD_REQUEST, &e);
    }
    let input = body.get("input").or_else(|| body.get("prompt"));
    let Some(input) = input else {
        return error(StatusCode::BAD_REQUEST, "input is required");
    };
    let inputs = if input.is_string()
        || input
            .as_array()
            .is_some_and(|a| !a.is_empty() && a[0].is_number())
    {
        vec![input.clone()]
    } else if let Some(array) = input.as_array() {
        array.clone()
    } else {
        return error(StatusCode::BAD_REQUEST, "invalid embedding input");
    };
    if inputs.is_empty() || inputs.len() > 32 {
        return error(
            StatusCode::BAD_REQUEST,
            "embedding batch must contain 1 to 32 inputs",
        );
    }
    let mut batch = Vec::new();
    for value in inputs {
        let ids = if let Some(text) = value.as_str() {
            state.model.tokenize(text, true)
        } else {
            token_ids(&value)
        };
        let ids = match ids {
            Ok(ids) => ids,
            Err(e) => return error(StatusCode::BAD_REQUEST, &e),
        };
        if let Err(e) = state.model.validate_tokens(&ids) {
            return error(StatusCode::BAD_REQUEST, &e);
        }
        batch.push(ids);
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return error(StatusCode::TOO_MANY_REQUESTS, "encoder is busy"),
    };
    let count = batch.iter().map(Vec::len).sum::<usize>();
    let cloned = state.clone();
    let base64_format = format == "base64";
    let vectors = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = cloned.metrics.embedding();
        batch
            .iter()
            .map(|ids| -> Result<Vec<f32>, String> {
                let mut vector = cloned.model.embed(ids, &pooling)?;
                cloned.metrics.processed(ids.len());
                if dimensions < vector.len() {
                    vector.truncate(dimensions);
                    crate::compute::normalize(&mut vector)?;
                }
                Ok(vector)
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await;
    let vectors = match vectors {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(_) => return error(StatusCode::INTERNAL_SERVER_ERROR, "encoder worker failed"),
    };
    let data = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let embedding = if base64_format {
                json!(base64::engine::general_purpose::STANDARD.encode(
                    vector
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                ))
            } else {
                json!(vector)
            };
            json!({"object":"embedding","index":index,"embedding":embedding})
        })
        .collect::<Vec<_>>();
    Json(json!({"object":"list","model":state.name,"data":data,"embeddings":vectors,"embedding":vectors.first(),"prompt_eval_count":count,"usage":{"prompt_tokens":count,"total_tokens":count}})).into_response()
}

#[derive(Clone, Copy)]
enum Operation {
    Rerank,
    Score,
    Classify,
    Pool,
}
type EncodedBatch = Vec<(Vec<u32>, Vec<u32>)>;
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
async fn operation(
    state: Arc<EncoderState>,
    headers: HeaderMap,
    body: Value,
    mode: Operation,
) -> Response {
    if !authorized(&headers, &state) {
        return mode.error(StatusCode::UNAUTHORIZED, "invalid API key");
    }
    if body
        .get("model")
        .is_some_and(|v| v.as_str() != Some(state.name.as_str()))
    {
        return mode.error(StatusCode::BAD_REQUEST, "requested model is not loaded");
    }
    if !matches!(mode, Operation::Pool) && !state.model.has_classifier() {
        return mode.error(StatusCode::NOT_IMPLEMENTED,"model has no sequence classification head; load BertForSequenceClassification or an XLM-RoBERTa cross-encoder");
    }
    if matches!(mode, Operation::Rerank | Operation::Score) && state.model.labels.len() != 1 {
        return mode.error(
            StatusCode::BAD_REQUEST,
            "rerank/score require a classifier with exactly one label",
        );
    }
    let mut returned_docs = Vec::new();
    let mut top_n = 0;
    let mut return_documents = false;
    let prepared: Result<EncodedBatch, String> = (|| {
        let pairs = match mode {
            Operation::Rerank => {
                let query = body["query"]
                    .as_str()
                    .filter(|v| !v.is_empty() && v.len() <= 65536)
                    .ok_or("query must be non-empty text of at most 65536 bytes")?;
                let docs = documents(&body)?;
                top_n = match body.get("top_n") {
                    None | Some(Value::Null) => docs.len(),
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| *n > 0)
                        .ok_or("top_n must be a positive integer")?
                        .min(docs.len()),
                };
                return_documents = match body.get("return_documents") {
                    None => false,
                    Some(v) => v.as_bool().ok_or("return_documents must be boolean")?,
                };
                returned_docs = docs.iter().map(|(_, doc)| doc.clone()).collect();
                docs.into_iter()
                    .map(|(text, _)| (query.to_owned(), text))
                    .collect::<Vec<_>>()
            }
            Operation::Score => score_pairs(&body)?,
            Operation::Classify | Operation::Pool => {
                return text_batch(&body["input"], 32)?
                    .into_iter()
                    .map(|text| {
                        let ids = state.model.tokenize(&text, true)?;
                        let types = vec![0; ids.len()];
                        Ok((ids, types))
                    })
                    .collect();
            }
        };
        pairs
            .iter()
            .map(|(query, document)| state.model.tokenize_pair(query, document))
            .collect()
    })();
    let batch = match prepared {
        Ok(batch) => batch,
        Err(e) => return mode.error(StatusCode::BAD_REQUEST, &e),
    };
    let tokens = batch.iter().map(|(ids, _)| ids.len()).sum::<usize>();
    if tokens > 32768 {
        return mode.error(
            StatusCode::BAD_REQUEST,
            "batch exceeds 32768 total input tokens",
        );
    }
    let pooling = body
        .get("pooling")
        .and_then(Value::as_str)
        .unwrap_or(&state.model.default_pooling)
        .to_string();
    if matches!(mode, Operation::Pool) {
        if let Err(e) = crate::encoder::validate_pooling(&pooling) {
            return mode.error(StatusCode::BAD_REQUEST, &e);
        }
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return mode.error(StatusCode::TOO_MANY_REQUESTS, "encoder is busy"),
    };
    let cloned = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = cloned.metrics.embedding();
        let output = if matches!(mode, Operation::Pool) {
            cloned.model.pool_batch(
                &batch.into_iter().map(|(ids, _)| ids).collect::<Vec<_>>(),
                &pooling,
            )
        } else {
            cloned.model.classify_batch(&batch)
        };
        if output.is_ok() {
            cloned.metrics.processed(tokens);
        }
        output
    })
    .await;
    let vectors = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return mode.error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Err(_) => return mode.error(StatusCode::INTERNAL_SERVER_ERROR, "encoder worker failed"),
    };
    let usage = json!({"prompt_tokens":tokens,"total_tokens":tokens});
    match mode {
        Operation::Rerank => {
            let mut scores=vectors.iter().enumerate().map(|(index,v)|(index,v[0])).collect::<Vec<_>>();
            scores.sort_by(|a,b|b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let results=scores.iter().take(top_n).map(|&(index,score)| {
                let mut item=json!({"index":index,"relevance_score":score});
                if return_documents {item["document"]=returned_docs[index].clone();}
                item
            }).collect::<Vec<_>>();
            Json(json!({"id":format!("rerank_{}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos()),"results":results,"model":state.name,"usage":usage,"meta":{"api_version":{"version":"2"},"billed_units":{"search_units":0,"input_tokens":tokens}}})).into_response()
        }
        Operation::Score => Json(json!({"object":"list","model":state.name,"data":vectors.iter().enumerate().map(|(index,v)|json!({"index":index,"object":"score","score":v[0]})).collect::<Vec<_>>(),"usage":usage})).into_response(),
        Operation::Classify => {
            let data=vectors.iter().enumerate().map(|(index,probs)| {
                let best=probs.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(&a.0))).map_or(0,|(id,_)|id);
                json!({"index":index,"label":state.model.labels[best],"probs":probs,"num_classes":probs.len()})
            }).collect::<Vec<_>>();
            Json(json!({"object":"list","model":state.name,"data":data,"usage":usage})).into_response()
        }
        Operation::Pool => Json(json!({"object":"list","model":state.name,"data":vectors.iter().enumerate().map(|(index,data)|json!({"index":index,"data":data})).collect::<Vec<_>>(),"usage":usage})).into_response(),
    }
}
async fn rerank(
    State(state): State<Arc<EncoderState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, headers, body, Operation::Rerank).await,
        Err(e) => Operation::Rerank.error(e.status(), &e.body_text()),
    }
}
async fn score(
    State(state): State<Arc<EncoderState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, headers, body, Operation::Score).await,
        Err(e) => Operation::Score.error(e.status(), &e.body_text()),
    }
}
async fn classify(
    State(state): State<Arc<EncoderState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, headers, body, Operation::Classify).await,
        Err(e) => Operation::Classify.error(e.status(), &e.body_text()),
    }
}
async fn pooling(
    State(state): State<Arc<EncoderState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    match body {
        Ok(Json(body)) => operation(state, headers, body, Operation::Pool).await,
        Err(e) => Operation::Pool.error(e.status(), &e.body_text()),
    }
}
pub fn serve(model: BertEncoder, config: ServerConfig) -> Result<(), String> {
    let state = Arc::new(EncoderState {
        metrics: Arc::new(crate::metrics::Metrics::new()),
        model,
        name: config.model_name,
        key: config.api_key,
        admission: Arc::new(tokio::sync::Semaphore::new(1)),
    });
    let app = Router::new()
        .route("/metrics", get(metrics))
        .route(
            "/health",
            get(|| async {
                Json(json!({"status":"ok","engine":"fllm-rust","architecture":"bert"}))
            }),
        )
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        .route("/v1/score", post(score))
        .route("/classify", post(classify))
        .route("/pooling", post(pooling))
        .route("/api/embed", post(embeddings))
        .route("/api/embeddings", post(embeddings))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .layer(axum::middleware::from_fn_with_state(state.metrics.clone(), crate::metrics::observe))
        .with_state(state);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port))
            .await
            .map_err(|e| e.to_string())?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
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
            .map_err(|e| e.to_string())
    })
}

async fn metrics(State(state): State<Arc<EncoderState>>) -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], state.metrics.render()).into_response()
}

#[cfg(test)]
mod classification_api_tests {
    use super::*;
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
}
