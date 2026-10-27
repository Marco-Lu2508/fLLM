//! The wiring between the HTTP layer and the response cache: build the fingerprint of a request, decide whether it may be
//! cached at all, and turn a cached entry back into an OpenAI-shaped answer.
//!
//! Configuration comes from the environment, so nothing has to change in the option parser:
//! `FLLM_OUTPUT_CACHE=exact|off` (default `exact`), `FLLM_OUTPUT_CACHE_BYTES` (default 64 MiB),
//! `FLLM_OUTPUT_CACHE_TTL` (seconds, default 3600), `FLLM_CACHE_SCOPE=global|tenant|off` (default `global`).
//!
//! Only requests whose answer is determined take part: `temperature = 0`, `n = 1`, no streaming, no tools, no images and no
//! `best_of`. With `scope = tenant` a request without a `user` field is not cached at all — better no hit than a hit that
//! belongs to somebody else.
use super::*;
use crate::cache::response_cache::{Entry, Fingerprint, ResponseCache, Scope};
use std::time::Duration;

/// The sampling parameters that change the text. Everything else in the body is either checked separately or is a
/// presentation detail.
const SAMPLING_KEYS: [&str; 14] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
    "repetition_penalty",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "n",
    "best_of",
];

/// Reads the cache configuration from the environment.
pub(crate) fn from_env() -> Arc<ResponseCache> {
    let scope = Scope::parse(&std::env::var("FLLM_CACHE_SCOPE").unwrap_or_else(|_| "global".to_owned()));
    let enabled = !matches!(
        std::env::var("FLLM_OUTPUT_CACHE").unwrap_or_else(|_| "exact".to_owned()).to_ascii_lowercase().as_str(),
        "off" | "0" | "no" | "false" | "none"
    );
    let bytes = std::env::var("FLLM_OUTPUT_CACHE_BYTES").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(64 << 20);
    let ttl = std::env::var("FLLM_OUTPUT_CACHE_TTL").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(3600);
    let scope = if enabled { scope } else { Scope::Off };
    Arc::new(ResponseCache::new(bytes, Duration::from_secs(ttl), scope))
}

/// A stable identity of the loaded model and its templates: any change to either makes old answers uninteresting.
pub(crate) fn identity(model: &str, arch: &str, chat_template: &str, tool_template: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"fllm-cache-identity\0");
    hasher.update(model.as_bytes());
    hasher.update(b"\0");
    hasher.update(arch.as_bytes());
    hasher.update(b"\0");
    hasher.update(chat_template.as_bytes());
    hasher.update(b"\0");
    hasher.update(tool_template.as_bytes());
    let digest = hasher.finalize();
    digest.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

/// The tenant namespace. With a tenant-scoped cache it is the API key of the caller (which the caller cannot choose) together with the
/// `user` field of the request; without keys it is the `user` field alone, and `None` when there is neither, so that the cache stays out
/// of the way.
pub(crate) fn tenant(body: &Value, scope: &Scope) -> Option<String> {
    match scope {
        Scope::Off => Some(String::new()),
        Scope::Global => Some(String::new()),
        Scope::Tenant => {
            let key = security::current_caller().and_then(|c| c.limiter).filter(|k| k.label != "anonymous").map(|k| k.id.clone());
            let user = body.get("user").and_then(Value::as_str).map(str::trim).filter(|user| !user.is_empty());
            match (key, user) {
                (Some(key), Some(user)) => Some(format!("{key}\0{user}")),
                (Some(key), None) => Some(key),
                (None, Some(user)) => Some(user.to_owned()),
                (None, None) => None,
            }
        }
    }
}

/// The fingerprint of a request, or `None` when the request takes no part in the cache.
pub(crate) fn fingerprint(state: &AppState, body: &Value, is_chat: bool) -> Option<Fingerprint> {
    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return None;
    }
    if is_chat && body.get("tools").is_some_and(|tools| tools.as_array().is_some_and(|a| !a.is_empty())) {
        return None; // tool calls have their own answer shape
    }
    for key in ["echo", "lora", "adapter", "adapters"] {
        if body.get(key).is_some_and(|value| !value.is_null()) {
            return None;
        }
    }
    // only an explicit greedy request is determined; a missing temperature means the server default, which this gate cannot see
    if !body.get("temperature").and_then(Value::as_f64).is_some_and(|t| t == 0.0) {
        return None;
    }
    if body.get("n").and_then(Value::as_u64).unwrap_or(1) > 1 || body.get("best_of").and_then(Value::as_u64).unwrap_or(1) > 1 {
        return None;
    }
    let mut sampling: Vec<(String, String)> = Vec::new();
    for key in SAMPLING_KEYS {
        if let Some(value) = body.get(key) {
            let text = match value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            sampling.push((key.to_owned(), text));
        }
    }
    if let Some(stop) = body.get("stop") {
        sampling.push(("stop".to_owned(), match stop {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }));
    }
    let grammar = ["response_format", "grammar", "guided_json", "guided_regex", "guided_choice"]
        .iter()
        .filter_map(|key| body.get(*key).map(|value| format!("{key}={value}")))
        .collect::<Vec<_>>()
        .join("\u{1f}");
    sampling.sort_by(|a, b| a.0.cmp(&b.0));
    // logprobs change the answer payload, so they belong into the key
    for key in ["logprobs", "top_logprobs"] {
        if let Some(value) = body.get(key) {
            sampling.push((key.to_owned(), value.to_string()));
        }
    }
    sampling.sort_by(|a, b| a.0.cmp(&b.0));
    let fingerprint = Fingerprint {
        model: state.cache_identity.clone(),
        template: String::new(),
        quantization: String::new(),
        loras: Vec::new(),
        sampling,
        grammar,
    };
    if fingerprint.cacheable().is_err() {
        return None;
    }
    Some(fingerprint)
}

/// The answer from the cache, in the same shape the generator produces (one choice, `n = 1`).
pub(crate) fn response(entry: &Entry, request_id: &str, created: u64, model: &str, is_chat: bool) -> Value {
    let mut choice = if is_chat {
        json!({
            "index": 0,
            "message": {"role": "assistant", "content": entry.text},
            "finish_reason": entry.finish_reason,
            "logprobs": Value::Null,
        })
    } else {
        json!({
            "index": 0,
            "text": entry.text,
            "finish_reason": entry.finish_reason,
            "logprobs": Value::Null,
        })
    };
    choice["logprobs"] = entry.logprobs.clone().unwrap_or(Value::Null);
    json!({
        "id": request_id,
        "object": if is_chat { "chat.completion" } else { "text_completion" },
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "total_tokens": entry.prompt_tokens + entry.completion_tokens,
        },
        // our own marker, like the prefix cache reports cached tokens: no client depends on it, tools can read it
        "fllm_cache": {"hit": true, "stored_at": entry.created_unix},
    })
}

/// Stores the finished non-streaming answer, if the cache is on.
/// The finished answer, in the pieces the cache stores.
pub(crate) struct Stored<'a> {
    pub text: &'a str,
    pub tokens: Vec<u32>,
    pub finish_reason: &'a str,
    pub logprobs: Option<Value>,
}

pub(crate) fn store(state: &AppState, body: &Value, prompt: &[i64], is_chat: bool, stored: Stored<'_>) {
    if !state.cache.enabled() {
        return;
    }
    let Some(fingerprint) = fingerprint(state, body, is_chat) else { return };
    let Some(salt) = tenant(body, state.cache.scope()) else { return };
    let key = fingerprint.key(prompt, &salt);
    let entry = Entry::new(stored.text.to_owned(), stored.tokens, stored.finish_reason.to_owned(), prompt.len())
        .with_logprobs(stored.logprobs);
    state.cache.put(key, entry);
}

/// The cache key of this request, when it takes part at all. Shared by the cache and the coalescer.
pub(crate) fn key_of(state: &AppState, body: &Value, prompt: &[i64], is_chat: bool) -> Option<[u8; 32]> {
    let fingerprint = fingerprint(state, body, is_chat)?;
    let salt = tenant(body, state.cache.scope())?;
    Some(fingerprint.key(prompt, &salt))
}

/// Looks up the answer for this request. `None` means: generate.
pub(crate) fn lookup(state: &AppState, body: &Value, prompt: &[i64], is_chat: bool, request_id: &str, created: u64, model: &str) -> Option<Value> {
    if !state.cache.enabled() {
        return None;
    }
    let fingerprint = fingerprint(state, body, is_chat)?;
    let salt = tenant(body, state.cache.scope())?;
    let key = fingerprint.key(prompt, &salt);
    let entry = state.cache.get(&key)?;
    Some(response(&entry, request_id, created, model, is_chat))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_changes_when_the_template_changes() {
        let a = identity("m", "llama", "tpl-a", "");
        let b = identity("m", "llama", "tpl-b", "");
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert_eq!(a, identity("m", "llama", "tpl-a", ""));
    }

    #[test]
    fn tenant_scope_without_a_user_field_keeps_the_cache_out() {
        assert_eq!(tenant(&json!({}), &Scope::Tenant), None);
        assert_eq!(tenant(&json!({"user": "  "}), &Scope::Tenant), None);
        assert_eq!(tenant(&json!({"user": "marco"}), &Scope::Tenant), Some("marco".to_owned()));
        assert_eq!(tenant(&json!({}), &Scope::Global), Some(String::new()));
    }

    #[test]
    fn an_answer_with_logprobs_keeps_its_numbers() {
        let entry = Entry::new("x".to_owned(), vec![1], "stop".to_owned(), 3)
            .with_logprobs(Some(json!({"content": [{"token": "x", "logprob": -0.5}]})));
        let payload = response(&entry, "req_1", 1, "m", true);
        assert_eq!(payload["choices"][0]["logprobs"]["content"][0]["logprob"], -0.5);
        let without = response(&Entry::new("x".to_owned(), vec![1], "stop".to_owned(), 3), "req_1", 1, "m", true);
        assert!(without["choices"][0]["logprobs"].is_null());
    }

    #[test]
    fn the_response_has_the_openai_shape() {
        let entry = Entry::new("hallo".to_owned(), vec![1, 2], "stop".to_owned(), 12);
        let chat = response(&entry, "req_1", 1700, "m", true);
        assert_eq!(chat["choices"][0]["message"]["content"], "hallo");
        assert_eq!(chat["usage"]["completion_tokens"], 2);
        assert_eq!(chat["usage"]["prompt_tokens"], 12);
        assert_eq!(chat["fllm_cache"]["hit"], true);
        let text = response(&entry, "req_1", 1700, "m", false);
        assert_eq!(text["choices"][0]["text"], "hallo");
        assert_eq!(text["object"], "text_completion");
    }
}
