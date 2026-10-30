//! Model providers: other servers and cloud services whose models can be used in the Studio next to the local one.
//! The list is `providers.json` next to `studio.json` (file mode 0600 on Unix, because it holds API keys). A provider is
//! an OpenAI-compatible base URL (`kind` "openai"), the Anthropic Messages API ("anthropic") or an Ollama server
//! ("ollama", through its OpenAI-compatible `/v1`).
//!
//! The page never sees a key: every route returns `has_key` instead, and chat requests go through
//! `POST /api/studio/providers/chat`, which adds the key on the way out. The routes are limited to this machine
//! (see `http::studio_admin_blocked`). Unless a provider is marked `local`, its host is resolved here and refused when it
//! points to a loopback, private, link-local or reserved address; the connection is pinned to the checked address and
//! redirects are never followed (a redirect would carry the key to another host).
use super::*;
use axum::http::header;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

const KINDS: [&str; 3] = ["openai", "anthropic", "ollama"];
const MAX_PROVIDERS: usize = 64;
const MAX_MODELS_PER_PROVIDER: usize = 2000;
/// Largest chat request the proxy accepts (images travel inside it as data URLs).
pub(in crate::server) const MAX_CHAT_BODY: usize = 32 << 20;
const MAX_LIST_BODY: usize = 4 << 20;
const MAX_REPLY_BODY: usize = 32 << 20;
const MAX_ERROR_BODY: usize = 64 << 10;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const LIST_TIMEOUT: Duration = Duration::from_secs(15);
/// Time to the first byte of a reply, and the total time of a reply that is not streamed.
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);
/// A stream that sends nothing for this long is cut.
const STREAM_IDLE: Duration = Duration::from_secs(120);
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 8192;

/// One-click templates of the settings page: URLs only, never keys.
const PRESETS: [(&str, &str, &str, &str, bool); 10] = [
    ("openai", "OpenAI", "openai", "https://api.openai.com/v1", false),
    ("openrouter", "OpenRouter", "openai", "https://openrouter.ai/api/v1", false),
    ("groq", "Groq", "openai", "https://api.groq.com/openai/v1", false),
    ("together", "Together AI", "openai", "https://api.together.xyz/v1", false),
    ("mistral", "Mistral", "openai", "https://api.mistral.ai/v1", false),
    ("gemini", "Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai", false),
    ("anthropic", "Anthropic", "anthropic", "https://api.anthropic.com/v1", false),
    ("ollama", "Ollama", "ollama", "http://localhost:11434", true),
    ("lmstudio", "LM Studio", "openai", "http://localhost:1234/v1", true),
    ("vllm", "vLLM", "openai", "http://localhost:8000/v1", true),
];

/// The request fields that are forwarded to OpenAI-compatible providers. Everything else (the engine-specific
/// sampling extras such as `mirostat`, `dry_*` or `tool_call_grammar`) is dropped: strict services reject unknown fields.
const FORWARDED_FIELDS: [&str; 22] = [
    "model", "messages", "stream", "stream_options", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "tools",
    "tool_choice", "parallel_tool_calls", "response_format", "seed", "presence_penalty", "frequency_penalty", "user", "n", "logprobs",
    "top_logprobs", "reasoning_effort", "logit_bias",
];

// ---------------------------------------------------------------------------------------------------------------
// Configuration file
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
struct Provider {
    id: String,
    name: String,
    kind: String,
    base_url: String,
    api_key: String,
    enabled: bool,
    /// The user says this server runs on this computer or network: private addresses are allowed.
    local: bool,
    /// The model ids found at the last refresh.
    models: Vec<String>,
}

// the key must never reach a log line through `{:?}`
impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("api_key", &if self.api_key.is_empty() { "<none>" } else { "<redacted>" })
            .field("enabled", &self.enabled)
            .field("local", &self.local)
            .field("models", &self.models.len())
            .finish()
    }
}

impl Provider {
    fn from_value(v: &Value) -> Option<Provider> {
        let text = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
        let id = text("id");
        if id.is_empty() {
            return None;
        }
        Some(Provider {
            id,
            name: text("name"),
            kind: text("kind"),
            base_url: text("base_url"),
            api_key: text("api_key"),
            enabled: v.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            local: v.get("local").and_then(Value::as_bool).unwrap_or(false),
            models: v
                .get("models")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_owned)).collect())
                .unwrap_or_default(),
        })
    }

    /// What is written to `providers.json` (with the key).
    fn stored(&self) -> Value {
        json!({
            "id": self.id, "name": self.name, "kind": self.kind, "base_url": self.base_url, "api_key": self.api_key,
            "enabled": self.enabled, "local": self.local, "models": self.models,
        })
    }

    /// What the page gets: everything but the key.
    fn public(&self) -> Value {
        json!({
            "id": self.id, "name": self.name, "kind": self.kind, "base_url": self.base_url, "has_key": !self.api_key.is_empty(),
            "enabled": self.enabled, "local": self.local, "models": self.models,
        })
    }
}

/// Lists (or updates) a server on this computer that fLLM itself starts, for example the AMD NPU runtime. The entry has no key and may use a loopback address.
pub(in crate::server) fn register_local(id: &str, name: &str, base_url: &str, models: &[String]) -> Result<(), String> {
    let _guard = store_lock();
    let mut providers = load()?;
    let entry = Provider {
        id: id.to_owned(),
        name: name.to_owned(),
        kind: "openai".to_owned(),
        base_url: base_url.to_owned(),
        api_key: String::new(),
        enabled: true,
        local: true,
        models: models.to_vec(),
    };
    match providers.iter_mut().find(|p| p.id == id) {
        Some(existing) => {
            let enabled = existing.enabled;
            *existing = Provider { enabled, ..entry };
        }
        None => providers.push(entry),
    }
    save(&providers)
}

/// Removes the entry made by [`register_local`] (nothing happens when there is none).
pub(in crate::server) fn unregister(id: &str) -> Result<(), String> {
    let _guard = store_lock();
    let mut providers = load()?;
    let before = providers.len();
    providers.retain(|p| p.id != id);
    if providers.len() == before {
        return Ok(());
    }
    save(&providers)
}

fn store_lock() -> std::sync::MutexGuard<'static, ()> {
    static STORE: Mutex<()> = Mutex::new(());
    STORE.lock().unwrap_or_else(|e| e.into_inner())
}

fn providers_path() -> std::path::PathBuf {
    crate::hub::studio_data_file("providers.json")
}

fn load_from(path: &Path) -> Result<Vec<Provider>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let root: Value =
        serde_json::from_str(&text).map_err(|e| format!("providers.json is not valid JSON ({e}); fix or delete the file first"))?;
    Ok(root.get("providers").and_then(Value::as_array).into_iter().flatten().filter_map(Provider::from_value).collect())
}

fn save_to(path: &Path, providers: &[Provider]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&json!({"providers": providers.iter().map(Provider::stored).collect::<Vec<_>>()}))
        .map_err(|e| e.to_string())?;
    // written next to the target and renamed, so a crash never leaves half a file; the key is never readable by others
    let temporary = path.with_extension("json.tmp");
    write_private(&temporary, text.as_bytes()).map_err(|e| format!("cannot write {}: {e}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    file.write_all(bytes)
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}

fn load() -> Result<Vec<Provider>, String> {
    load_from(&providers_path())
}

fn save(providers: &[Provider]) -> Result<(), String> {
    save_to(&providers_path(), providers)
}

// ---------------------------------------------------------------------------------------------------------------
// Validation, addresses, errors
// ---------------------------------------------------------------------------------------------------------------

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').chars().take(40).collect::<String>();
    if out.is_empty() { "provider".to_owned() } else { out }
}

fn parse_base_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw.trim()).map_err(|e| format!("the address is not a valid URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("the address must start with http:// or https://".to_owned());
    }
    if url.host_str().is_none() {
        return Err("the address has no host".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("the address must not contain a user name or password; use the key field".to_owned());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("the address must not contain a query or fragment".to_owned());
    }
    Ok(url)
}

/// The address of the API: Ollama and Anthropic are given without the `/v1` they need.
fn api_base(p: &Provider) -> String {
    let base = p.base_url.trim().trim_end_matches('/').to_owned();
    if matches!(p.kind.as_str(), "ollama" | "anthropic") && !base.ends_with("/v1") {
        format!("{base}/v1")
    } else {
        base
    }
}

/// Refuses the addresses a provider must not point to unless the user marked it as local.
fn check_addresses(host: &str, addresses: &[IpAddr], local: bool) -> Result<(), String> {
    if addresses.is_empty() {
        return Err(format!("the host '{host}' has no address"));
    }
    if local {
        return Ok(());
    }
    // all answers must be public: a name that also resolves to an internal address is refused outright
    if let Some(bad) = addresses.iter().find(|ip| crate::server::remote_image::is_blocked(**ip)) {
        return Err(format!(
            "the host '{host}' resolves to {bad}, a loopback, private, link-local or reserved address; if this server really runs on this computer or network, switch on \"This server runs on this computer or network\""
        ));
    }
    Ok(())
}

/// Replaces every occurrence of the key in a text (an upstream error may echo it).
fn redact(text: &str, key: &str) -> String {
    if key.len() < 4 {
        return text.to_owned();
    }
    text.replace(key, "[key]")
}

/// A client pinned to the checked address of the provider, plus the parsed base URL.
async fn client_for(p: &Provider, total_timeout: Option<Duration>) -> Result<reqwest::Client, String> {
    let url = parse_base_url(&api_base(p))?;
    let host = url.host_str().unwrap_or_default().to_owned();
    let port = url.port_or_known_default().unwrap_or(443);
    let bare = host.trim_matches(['[', ']']);
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .user_agent("fllm-studio")
        .no_proxy();
    if let Some(t) = total_timeout {
        builder = builder.timeout(t);
    }
    if let Ok(ip) = bare.parse::<IpAddr>() {
        check_addresses(&host, &[ip], p.local)?;
    } else {
        let found: Vec<SocketAddr> = tokio::net::lookup_host((bare, port))
            .await
            .map_err(|e| format!("cannot resolve the host '{host}': {e}"))?
            .collect();
        let ips: Vec<IpAddr> = found.iter().map(|a| a.ip()).collect();
        check_addresses(&host, &ips, p.local)?;
        // pin the connection to the address that was checked
        builder = builder.resolve(bare, found[0]);
    }
    builder.build().map_err(|e| format!("cannot start the connection: {e}"))
}

fn authorize(p: &Provider, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match p.kind.as_str() {
        "anthropic" => request.header("x-api-key", &p.api_key).header("anthropic-version", ANTHROPIC_VERSION),
        _ if p.api_key.is_empty() => request,
        _ => request.bearer_auth(&p.api_key),
    }
}

fn describe_send_error(p: &Provider, e: reqwest::Error) -> String {
    let e = e.without_url();
    let detail = redact(&e.to_string(), &p.api_key);
    if e.is_timeout() {
        format!("{} did not answer in time", p.name)
    } else if e.is_connect() {
        format!("cannot connect to {}: {detail}", p.name)
    } else {
        format!("the request to {} failed: {detail}", p.name)
    }
}

/// A plain sentence for an error answer of a provider.
fn upstream_error(p: &Provider, status: u16, body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let detail = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("message"))
                .or_else(|| v.get("error").filter(|e| e.is_string()))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| text.trim().chars().take(300).collect());
    let detail = redact(&detail, &p.api_key);
    let tail = if detail.is_empty() { String::new() } else { format!(": {detail}") };
    let name = &p.name;
    match status {
        401 | 403 if p.api_key.is_empty() => format!("{name} wants a key and none is set (HTTP {status}){tail}"),
        401 | 403 => format!("the key was not accepted by {name} (HTTP {status}){tail}"),
        404 => format!("{name} does not know this address or model (HTTP 404){tail}"),
        429 => format!("{name} says the rate limit or quota is reached (HTTP 429){tail}"),
        500..=599 => format!("{name} had a server error (HTTP {status}){tail}"),
        _ => format!("{name} refused the request (HTTP {status}){tail}"),
    }
}

/// The status the Studio answers with for a failed upstream call. Authentication failures become 502, so the page
/// does not mistake them for a problem with its own access.
fn relay_status(upstream: u16) -> StatusCode {
    match upstream {
        400 | 404 | 408 | 413 | 422 | 429 => StatusCode::from_u16(upstream).unwrap_or(StatusCode::BAD_GATEWAY),
        _ => StatusCode::BAD_GATEWAY,
    }
}

async fn read_limited(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| format!("the reply was cut off: {}", e.without_url()))? {
        if body.len() + chunk.len() > limit {
            return Err(format!("the reply is larger than {} MiB", limit >> 20));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

// ---------------------------------------------------------------------------------------------------------------
// Model lists
// ---------------------------------------------------------------------------------------------------------------

fn parse_model_list(body: &Value) -> Vec<String> {
    let mut ids: Vec<String> = body
        .get("data")
        .or_else(|| body.get("models"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").or_else(|| m.get("name")).and_then(Value::as_str).or_else(|| m.as_str()))
        // Gemini lists "models/<name>" and takes both forms
        .map(|id| id.strip_prefix("models/").unwrap_or(id).to_owned())
        .filter(|id| !id.is_empty())
        .collect();
    ids.sort();
    ids.dedup();
    ids.truncate(MAX_MODELS_PER_PROVIDER);
    ids
}

async fn fetch_models(p: &Provider) -> Result<Vec<String>, String> {
    let client = client_for(p, Some(LIST_TIMEOUT)).await?;
    let mut url = format!("{}/models", api_base(p));
    if p.kind == "anthropic" {
        url.push_str("?limit=1000");
    }
    let response = authorize(p, client.get(url)).send().await.map_err(|e| describe_send_error(p, e))?;
    let status = response.status();
    if !status.is_success() {
        let body = read_limited(response, MAX_ERROR_BODY).await.unwrap_or_default();
        return Err(upstream_error(p, status.as_u16(), &body));
    }
    let body = read_limited(response, MAX_LIST_BODY).await?;
    let value: Value = serde_json::from_slice(&body)
        .map_err(|_| format!("{} did not answer with a model list; check that the address is the API address (usually ending in /v1)", p.name))?;
    Ok(parse_model_list(&value))
}

/// Stores a refreshed model list, when the provider still exists.
fn cache_models(id: &str, models: &[String]) {
    let _guard = store_lock();
    if let Ok(mut all) = load() {
        if let Some(p) = all.iter_mut().find(|p| p.id == id) {
            if p.models != models {
                p.models = models.to_vec();
                let _ = save(&all);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Anthropic <-> OpenAI
// ---------------------------------------------------------------------------------------------------------------

mod anthropic {
    use serde_json::{json, Map, Value};
    use std::collections::HashMap;

    fn text_of(content: &Value) -> String {
        match content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| if p.get("type").and_then(Value::as_str) == Some("text") { p.get("text").and_then(Value::as_str) } else { None })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    fn image_block(url: &str) -> Result<Value, String> {
        if let Some(rest) = url.strip_prefix("data:") {
            let (meta, data) = rest.split_once(',').ok_or("an image data URL is malformed")?;
            let media = meta.strip_suffix(";base64").ok_or("only base64 image data URLs are supported")?;
            return Ok(json!({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}));
        }
        if url.starts_with("http://") || url.starts_with("https://") {
            return Ok(json!({"type": "image", "source": {"type": "url", "url": url}}));
        }
        Err("an image must be a data: URL or an http(s) URL".to_owned())
    }

    fn user_blocks(content: &Value) -> Result<Vec<Value>, String> {
        match content {
            Value::String(s) if !s.is_empty() => Ok(vec![json!({"type": "text", "text": s})]),
            Value::Array(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str).unwrap_or("text") {
                        "text" => {
                            let t = part.get("text").and_then(Value::as_str).unwrap_or_default();
                            if !t.is_empty() {
                                out.push(json!({"type": "text", "text": t}));
                            }
                        }
                        "image_url" => {
                            let image = part.get("image_url");
                            let url = image.and_then(|i| i.get("url").and_then(Value::as_str).or_else(|| i.as_str())).unwrap_or_default();
                            out.push(image_block(url)?);
                        }
                        other => return Err(format!("the content part type `{other}` is not supported by this provider")),
                    }
                }
                Ok(out)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Appends blocks to the last message when it has the same role (the API wants strictly alternating roles for tool results).
    fn push_message(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
        if blocks.is_empty() {
            return;
        }
        if let Some(last) = messages.last_mut() {
            if last.get("role").and_then(Value::as_str) == Some(role) {
                if let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) {
                    content.extend(blocks);
                    return;
                }
            }
        }
        messages.push(json!({"role": role, "content": blocks}));
    }

    fn tool_choice(choice: &Value) -> Option<Value> {
        match choice {
            Value::String(s) => match s.as_str() {
                "auto" => Some(json!({"type": "auto"})),
                "required" => Some(json!({"type": "any"})),
                "none" => Some(json!({"type": "none"})),
                _ => None,
            },
            Value::Object(_) => choice
                .pointer("/function/name")
                .and_then(Value::as_str)
                .map(|name| json!({"type": "tool", "name": name})),
            _ => None,
        }
    }

    /// An OpenAI chat-completions body as an Anthropic Messages body.
    pub fn request(req: &Value) -> Result<Value, String> {
        let source = req.get("messages").and_then(Value::as_array).ok_or("the request has no messages")?;
        let mut system: Vec<String> = Vec::new();
        let mut messages: Vec<Value> = Vec::new();
        for message in source {
            let content = message.get("content").unwrap_or(&Value::Null);
            match message.get("role").and_then(Value::as_str).unwrap_or_default() {
                "system" | "developer" => {
                    let t = text_of(content);
                    if !t.is_empty() {
                        system.push(t);
                    }
                }
                "user" => push_message(&mut messages, "user", user_blocks(content)?),
                "assistant" => {
                    let mut blocks = Vec::new();
                    let t = text_of(content);
                    if !t.is_empty() {
                        blocks.push(json!({"type": "text", "text": t}));
                    }
                    for call in message.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                        let function = call.get("function").unwrap_or(&Value::Null);
                        let args = function.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        let input = serde_json::from_str::<Value>(args).ok().filter(Value::is_object).unwrap_or_else(|| json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.get("id").and_then(Value::as_str).unwrap_or("call_0"),
                            "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                            "input": input,
                        }));
                    }
                    push_message(&mut messages, "assistant", blocks);
                }
                "tool" => {
                    let result = json!({
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id").and_then(Value::as_str).unwrap_or_default(),
                        "content": text_of(content),
                    });
                    push_message(&mut messages, "user", vec![result]);
                }
                other => return Err(format!("the message role `{other}` is not supported by this provider")),
            }
        }
        let mut out = Map::new();
        out.insert("model".into(), req.get("model").cloned().unwrap_or(Value::Null));
        out.insert("messages".into(), Value::Array(messages));
        if !system.is_empty() {
            out.insert("system".into(), Value::String(system.join("\n\n")));
        }
        let max_tokens = req
            .get("max_tokens")
            .or_else(|| req.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .filter(|n| *n > 0)
            .unwrap_or(super::DEFAULT_MAX_TOKENS);
        out.insert("max_tokens".into(), json!(max_tokens));
        // newer models refuse a request that sets both
        if let Some(t) = req.get("temperature").filter(|v| v.is_number()) {
            out.insert("temperature".into(), t.clone());
        } else if let Some(p) = req.get("top_p").filter(|v| v.is_number()) {
            out.insert("top_p".into(), p.clone());
        }
        match req.get("stop") {
            Some(Value::String(s)) => {
                out.insert("stop_sequences".into(), json!([s]));
            }
            Some(Value::Array(a)) if !a.is_empty() => {
                out.insert("stop_sequences".into(), Value::Array(a.clone()));
            }
            _ => {}
        }
        if req.get("stream").and_then(Value::as_bool) == Some(true) {
            out.insert("stream".into(), json!(true));
        }
        let tools: Vec<Value> = req
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|t| {
                let f = t.get("function")?;
                Some(json!({
                    "name": f.get("name")?,
                    "description": f.get("description").and_then(Value::as_str).unwrap_or_default(),
                    "input_schema": f.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                }))
            })
            .collect();
        if !tools.is_empty() {
            out.insert("tools".into(), Value::Array(tools));
            if let Some(choice) = req.get("tool_choice").and_then(tool_choice) {
                out.insert("tool_choice".into(), choice);
            }
        }
        Ok(Value::Object(out))
    }

    fn finish_reason(stop: &str) -> &'static str {
        match stop {
            "max_tokens" => "length",
            "tool_use" => "tool_calls",
            "refusal" => "content_filter",
            _ => "stop",
        }
    }

    fn usage_of(usage: &Value) -> (u64, u64) {
        let n = |k: &str| usage.get(k).and_then(Value::as_u64).unwrap_or(0);
        (n("input_tokens") + n("cache_creation_input_tokens") + n("cache_read_input_tokens"), n("output_tokens"))
    }

    fn usage_json(prompt: u64, completion: u64) -> Value {
        json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})
    }

    /// A non-streamed Messages answer as a chat completion.
    pub fn response(body: &Value) -> Value {
        let mut text = String::new();
        let mut thinking = String::new();
        let mut calls: Vec<Value> = Vec::new();
        for block in body.get("content").and_then(Value::as_array).into_iter().flatten() {
            match block.get("type").and_then(Value::as_str).unwrap_or_default() {
                "text" => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or_default()),
                "thinking" => thinking.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or_default()),
                "tool_use" => calls.push(json!({
                    "id": block.get("id"), "type": "function",
                    "function": {"name": block.get("name"), "arguments": block.get("input").cloned().unwrap_or_else(|| json!({})).to_string()},
                })),
                _ => {}
            }
        }
        let mut message = json!({"role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) }});
        if !thinking.is_empty() {
            message["reasoning_content"] = json!(thinking);
        }
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        let (prompt, completion) = usage_of(body.get("usage").unwrap_or(&Value::Null));
        json!({
            "id": body.get("id").cloned().unwrap_or_else(|| json!("msg")),
            "object": "chat.completion",
            "created": super::now_seconds(),
            "model": body.get("model").cloned().unwrap_or(Value::Null),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(body.get("stop_reason").and_then(Value::as_str).unwrap_or("end_turn"))}],
            "usage": usage_json(prompt, completion),
        })
    }

    /// Splits a byte stream of server-sent events into the payloads of its `data:` lines.
    #[derive(Default)]
    pub struct SseData {
        buffer: Vec<u8>,
    }

    impl SseData {
        pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
            self.buffer.extend_from_slice(bytes);
            let mut out = Vec::new();
            while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = self.buffer.drain(..=end).collect();
                let line = String::from_utf8_lossy(&line);
                if let Some(data) = line.trim_end_matches(['\r', '\n']).strip_prefix("data:") {
                    out.push(data.trim_start().to_owned());
                }
            }
            out
        }
    }

    /// Turns the events of a Messages stream into chat-completion chunks.
    #[derive(Default)]
    pub struct StreamTranslator {
        id: String,
        model: String,
        prompt_tokens: u64,
        completion_tokens: u64,
        /// Message block index -> index among the tool calls.
        tool_slots: HashMap<u64, usize>,
        next_tool: usize,
        started: bool,
        pub finished: bool,
    }

    impl StreamTranslator {
        fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
            json!({
                "id": self.id, "object": "chat.completion.chunk", "created": super::now_seconds(), "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
        }

        /// The chunks (or an `{"error": ...}` object) for one event.
        pub fn event(&mut self, event: &Value) -> Vec<Value> {
            let mut out = Vec::new();
            match event.get("type").and_then(Value::as_str).unwrap_or_default() {
                "message_start" => {
                    let message = event.get("message").unwrap_or(&Value::Null);
                    self.id = message.get("id").and_then(Value::as_str).unwrap_or("msg").to_owned();
                    self.model = message.get("model").and_then(Value::as_str).unwrap_or_default().to_owned();
                    let (prompt, completion) = usage_of(message.get("usage").unwrap_or(&Value::Null));
                    self.prompt_tokens = prompt;
                    self.completion_tokens = completion;
                    self.started = true;
                    out.push(self.chunk(json!({"role": "assistant", "content": ""}), None));
                }
                "content_block_start" => {
                    let block = event.get("content_block").unwrap_or(&Value::Null);
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                        let slot = self.next_tool;
                        self.next_tool += 1;
                        self.tool_slots.insert(index, slot);
                        out.push(self.chunk(
                            json!({"tool_calls": [{"index": slot, "id": block.get("id"), "type": "function",
                                "function": {"name": block.get("name"), "arguments": ""}}]}),
                            None,
                        ));
                    }
                }
                "content_block_delta" => {
                    let delta = event.get("delta").unwrap_or(&Value::Null);
                    match delta.get("type").and_then(Value::as_str).unwrap_or_default() {
                        "text_delta" => {
                            let t = delta.get("text").and_then(Value::as_str).unwrap_or_default();
                            if !t.is_empty() {
                                out.push(self.chunk(json!({"content": t}), None));
                            }
                        }
                        "thinking_delta" => {
                            let t = delta.get("thinking").and_then(Value::as_str).unwrap_or_default();
                            if !t.is_empty() {
                                out.push(self.chunk(json!({"reasoning_content": t}), None));
                            }
                        }
                        "input_json_delta" => {
                            let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                            if let Some(slot) = self.tool_slots.get(&index) {
                                let part = delta.get("partial_json").and_then(Value::as_str).unwrap_or_default();
                                out.push(self.chunk(json!({"tool_calls": [{"index": slot, "function": {"arguments": part}}]}), None));
                            }
                        }
                        _ => {}
                    }
                }
                "message_delta" => {
                    let usage = event.get("usage").unwrap_or(&Value::Null);
                    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
                        self.completion_tokens = n;
                    }
                    if usage.get("input_tokens").and_then(Value::as_u64).is_some_and(|n| n > 0) {
                        self.prompt_tokens = usage_of(usage).0;
                    }
                    if let Some(stop) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                        out.push(self.chunk(json!({}), Some(finish_reason(stop))));
                    }
                }
                "message_stop" => {
                    self.finished = true;
                    out.push(json!({
                        "id": self.id, "object": "chat.completion.chunk", "created": super::now_seconds(), "model": self.model,
                        "choices": [], "usage": usage_json(self.prompt_tokens, self.completion_tokens),
                    }));
                }
                "error" => {
                    self.finished = true;
                    let message = event.pointer("/error/message").and_then(Value::as_str).unwrap_or("the provider reported an error");
                    out.push(json!({"error": {"message": message, "type": event.pointer("/error/type")}}));
                }
                _ => {}
            }
            out
        }
    }
}

fn now_seconds() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------------------------------------------

fn presets_json() -> Value {
    Value::Array(
        PRESETS
            .iter()
            .map(|(id, name, kind, url, local)| json!({"id": id, "name": name, "kind": kind, "base_url": url, "local": local}))
            .collect(),
    )
}

/// `GET /api/studio/providers`: the providers (without keys) and the templates.
pub(in crate::server) async fn list() -> Response {
    let _guard = store_lock();
    match load() {
        Ok(all) => Json(json!({
            "providers": all.iter().map(Provider::public).collect::<Vec<_>>(),
            "presets": presets_json(),
            "config_file": providers_path().display().to_string(),
        }))
        .into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
    }
}

fn text_field(body: &Value, key: &str) -> Option<String> {
    body.get(key).and_then(Value::as_str).map(|s| s.trim().to_owned())
}

fn validate_key(key: &str) -> Result<(), String> {
    if key.len() > 4096 || !key.chars().all(|c| c.is_ascii_graphic()) {
        return Err("the key must be one line of visible characters without spaces".to_owned());
    }
    Ok(())
}

/// Applies the fields of a request to a provider (existing or new) and validates the result.
/// A key that is not sent stays; when the host changes without a new key, the old key is dropped so it never goes to another host.
fn apply_fields(mut p: Provider, body: &Value) -> Result<Provider, String> {
    let before = parse_base_url(&p.base_url).ok().and_then(|u| u.host_str().map(str::to_owned));
    if let Some(name) = text_field(body, "name") {
        p.name = name;
    }
    if let Some(kind) = text_field(body, "kind") {
        p.kind = kind;
    }
    if let Some(url) = text_field(body, "base_url") {
        p.base_url = url.trim_end_matches('/').to_owned();
    }
    if let Some(enabled) = body.get("enabled").and_then(Value::as_bool) {
        p.enabled = enabled;
    }
    if let Some(local) = body.get("local").and_then(Value::as_bool) {
        p.local = local;
    }
    let new_key = text_field(body, "api_key");
    if let Some(key) = &new_key {
        validate_key(key)?;
        p.api_key = key.clone();
    }
    if p.name.is_empty() || p.name.chars().count() > 64 {
        return Err("the name must be 1-64 characters".to_owned());
    }
    if !KINDS.contains(&p.kind.as_str()) {
        return Err("the kind must be `openai`, `anthropic` or `ollama`".to_owned());
    }
    if p.base_url.len() > 512 {
        return Err("the address is too long".to_owned());
    }
    let url = parse_base_url(&p.base_url)?;
    if new_key.is_none() && before.is_some() && before.as_deref() != url.host_str() {
        p.api_key.clear();
    }
    Ok(p)
}

fn empty_provider(id: String) -> Provider {
    Provider { id, name: String::new(), kind: "openai".into(), base_url: String::new(), api_key: String::new(), enabled: true, local: false, models: Vec::new() }
}

/// `POST /api/studio/providers` with `{id?, name, kind, base_url, api_key?, enabled?, local?}`: adds or updates a provider.
/// Without `id` a new one is created. An `api_key` that is absent keeps the saved key; an empty one removes it.
pub(in crate::server) async fn upsert(Json(body): Json<Value>) -> Response {
    let _guard = store_lock();
    let result = (|| -> Result<Provider, String> {
        let mut all = load()?;
        let requested = text_field(&body, "id").filter(|s| !s.is_empty());
        if let Some(id) = &requested {
            if !valid_id(id) {
                return Err("the id must be 1-64 letters, digits, `-` or `_`".to_owned());
            }
        }
        let position = requested.as_ref().and_then(|id| all.iter().position(|p| &p.id == id));
        let (base, existing) = match position {
            Some(i) => (all[i].clone(), true),
            None => {
                if all.len() >= MAX_PROVIDERS {
                    return Err(format!("at most {MAX_PROVIDERS} providers are supported"));
                }
                let id = requested.unwrap_or_else(|| {
                    let stem = slug(&text_field(&body, "name").unwrap_or_default());
                    let mut candidate = stem.clone();
                    let mut n = 2;
                    while all.iter().any(|p| p.id == candidate) {
                        candidate = format!("{stem}-{n}");
                        n += 1;
                    }
                    candidate
                });
                (empty_provider(id), false)
            }
        };
        let mut updated = apply_fields(base.clone(), &body)?;
        if existing && (updated.base_url != base.base_url || updated.kind != base.kind) {
            updated.models.clear();
        }
        match position {
            Some(i) => all[i] = updated.clone(),
            None => all.push(updated.clone()),
        }
        save(&all)?;
        Ok(updated)
    })();
    match result {
        Ok(p) => Json(json!({"status": "saved", "provider": p.public()})).into_response(),
        Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
    }
}

/// `DELETE /api/studio/providers` with `{id}`.
pub(in crate::server) async fn remove(Json(body): Json<Value>) -> Response {
    let id = text_field(&body, "id").unwrap_or_default();
    let _guard = store_lock();
    let result = (|| -> Result<(), String> {
        let mut all = load()?;
        let before = all.len();
        all.retain(|p| p.id != id);
        if all.len() == before {
            return Err(format!("no provider with the id `{id}`"));
        }
        save(&all)
    })();
    match result {
        Ok(()) => Json(json!({"status": "removed"})).into_response(),
        Err(e) => api_error(StatusCode::NOT_FOUND, &e),
    }
}

/// `POST /api/studio/providers/test` with `{id}` or the fields of a provider that is not saved yet (`kind`, `base_url`,
/// `api_key`, `local`). Lists the models of the provider and answers `{ok: true, models}` or `{ok: false, error}`;
/// the list of a saved provider is stored as its cache.
pub(in crate::server) async fn test_connection(Json(body): Json<Value>) -> Response {
    if !body.is_object() {
        return api_error(StatusCode::BAD_REQUEST, "the body must be a JSON object");
    }
    let id = text_field(&body, "id").unwrap_or_default();
    let stored = {
        let _guard = store_lock();
        match load() {
            Ok(all) => all.into_iter().find(|p| p.id == id),
            Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        }
    };
    let mut draft = body.clone();
    if draft.get("name").is_none() {
        draft["name"] = json!(stored.as_ref().map_or("this provider", |p| p.name.as_str()));
    }
    let base = stored.clone().unwrap_or_else(|| empty_provider(String::new()));
    let provider = match apply_fields(base, &draft) {
        Ok(p) => p,
        Err(e) => return Json(json!({"ok": false, "error": e})).into_response(),
    };
    match fetch_models(&provider).await {
        Ok(models) => {
            if stored.is_some() && provider.base_url == stored.as_ref().map_or("", |p| p.base_url.as_str()) {
                cache_models(&id, &models);
            }
            Json(json!({"ok": true, "count": models.len(), "models": models})).into_response()
        }
        Err(e) => Json(json!({"ok": false, "error": e})).into_response(),
    }
}

/// `GET /api/studio/providers/models[?refresh=1]`: the models of the enabled providers, from the cache or freshly listed.
pub(in crate::server) async fn models(axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>) -> Response {
    let refresh = query.get("refresh").is_some_and(|v| v != "0" && v != "false");
    let all = {
        let _guard = store_lock();
        match load() {
            Ok(all) => all,
            Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        }
    };
    let enabled: Vec<Provider> = all.into_iter().filter(|p| p.enabled).collect();
    let lists: Vec<Result<Vec<String>, String>> = if refresh {
        futures_util::future::join_all(enabled.iter().map(fetch_models)).await
    } else {
        enabled.iter().map(|p| Ok(p.models.clone())).collect()
    };
    let mut models = Vec::new();
    let mut errors = Vec::new();
    for (p, list) in enabled.iter().zip(lists) {
        match list {
            Ok(ids) => {
                // FastFlowLM lists its whole catalogue; only the model it has loaded can answer
                let ids = if p.id == crate::backend::npu::flm::PROVIDER_ID {
                    match crate::backend::npu::flm::serving() {
                        crate::backend::npu::flm::Serving::Running { tag, .. } => vec![tag],
                        _ => Vec::new(),
                    }
                } else {
                    ids
                };
                if refresh {
                    cache_models(&p.id, &ids);
                }
                for id in ids {
                    models.push(json!({"provider": p.id, "provider_name": p.name, "kind": p.kind, "local": p.local, "id": id}));
                }
            }
            Err(error) => errors.push(json!({"provider": p.id, "provider_name": p.name, "error": error})),
        }
    }
    Json(json!({"models": models, "errors": errors})).into_response()
}

fn sse_response(rx: tokio_mpsc::Receiver<Bytes>) -> Response {
    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<Bytes, Infallible>(b), rx)) });
    let mut response = Response::new(Body::from_stream(stream));
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("text/event-stream"));
    headers.insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-cache"));
    headers.insert("x-accel-buffering", header::HeaderValue::from_static("no"));
    response
}

fn sse_line(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn sse_error(message: &str) -> Bytes {
    Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", json!({"error": {"message": message}})))
}

/// `POST /api/studio/providers/chat` with `{provider, request}`: sends a chat-completions body to the provider and relays
/// the answer (a JSON body, or the server-sent events of a stream) in the OpenAI format.
pub(in crate::server) async fn chat(Json(body): Json<Value>) -> Response {
    let id = text_field(&body, "provider").unwrap_or_default();
    let provider = {
        let _guard = store_lock();
        match load() {
            Ok(all) => all.into_iter().find(|p| p.id == id),
            Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        }
    };
    let Some(provider) = provider else {
        return api_error(StatusCode::NOT_FOUND, &format!("no provider with the id `{id}`"));
    };
    if !provider.enabled {
        return api_error(StatusCode::BAD_REQUEST, &format!("{} is switched off in the settings", provider.name));
    }
    let Some(request) = body.get("request").and_then(Value::as_object) else {
        return api_error(StatusCode::BAD_REQUEST, "`request` must be a chat-completions body");
    };
    if !request.get("messages").is_some_and(Value::is_array) {
        return api_error(StatusCode::BAD_REQUEST, "the request has no messages");
    }
    let streaming = request.get("stream").and_then(Value::as_bool) == Some(true);
    let is_anthropic = provider.kind == "anthropic";
    let outgoing = if is_anthropic {
        match anthropic::request(&Value::Object(request.clone())) {
            Ok(v) => v,
            Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
        }
    } else {
        Value::Object(request.iter().filter(|(k, _)| FORWARDED_FIELDS.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect())
    };
    let client = match client_for(&provider, if streaming { None } else { Some(REPLY_TIMEOUT) }).await {
        Ok(c) => c,
        Err(e) => return api_error(StatusCode::BAD_GATEWAY, &e),
    };
    let url = format!("{}/{}", api_base(&provider), if is_anthropic { "messages" } else { "chat/completions" });
    let sent = authorize(&provider, client.post(url)).json(&outgoing).send();
    let response = match tokio::time::timeout(REPLY_TIMEOUT, sent).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return api_error(StatusCode::BAD_GATEWAY, &describe_send_error(&provider, e)),
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, &format!("{} did not answer in time", provider.name)),
    };
    let status = response.status();
    if !status.is_success() {
        let bytes = read_limited(response, MAX_ERROR_BODY).await.unwrap_or_default();
        return api_error(relay_status(status.as_u16()), &upstream_error(&provider, status.as_u16(), &bytes));
    }
    if !streaming {
        let bytes = match read_limited(response, MAX_REPLY_BODY).await {
            Ok(b) => b,
            Err(e) => return api_error(StatusCode::BAD_GATEWAY, &redact(&e, &provider.api_key)),
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            return api_error(StatusCode::BAD_GATEWAY, &format!("{} did not answer with JSON", provider.name));
        };
        return Json(if is_anthropic { anthropic::response(&value) } else { value }).into_response();
    }
    let (tx, rx) = tokio_mpsc::channel::<Bytes>(32);
    tokio::spawn(pump(provider, response, tx));
    sse_response(rx)
}

/// Copies the stream of the provider to the page until either side ends. When the page goes away (stop button),
/// the send fails and the provider's connection is dropped.
async fn pump(provider: Provider, mut response: reqwest::Response, tx: tokio_mpsc::Sender<Bytes>) {
    let is_anthropic = provider.kind == "anthropic";
    let mut lines = anthropic::SseData::default();
    let mut translator = anthropic::StreamTranslator::default();
    loop {
        let chunk = match tokio::time::timeout(STREAM_IDLE, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => {
                let text = redact(&format!("{} cut the stream: {}", provider.name, e.without_url()), &provider.api_key);
                let _ = tx.send(sse_error(&text)).await;
                return;
            }
            Err(_) => {
                let _ = tx.send(sse_error(&format!("{} stopped sending (no data for {} s)", provider.name, STREAM_IDLE.as_secs()))).await;
                return;
            }
        };
        if !is_anthropic {
            if tx.send(chunk).await.is_err() {
                return;
            }
            continue;
        }
        for data in lines.push(&chunk) {
            let Ok(event) = serde_json::from_str::<Value>(&data) else { continue };
            for out in translator.event(&event) {
                if tx.send(sse_line(&out)).await.is_err() {
                    return;
                }
            }
            if translator.finished {
                let _ = tx.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
                return;
            }
        }
    }
    if is_anthropic {
        let _ = tx.send(Bytes::from_static(b"data: [DONE]\n\n")).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(kind: &str, key: &str) -> Provider {
        Provider {
            id: "p".into(), name: "Test".into(), kind: kind.into(), base_url: "https://example.com/v1".into(), api_key: key.into(),
            enabled: true, local: false, models: vec!["a".into(), "b".into()],
        }
    }

    #[test]
    fn request_translation_covers_system_images_tools_and_results() {
        let req = json!({
            "model": "claude-x", "stream": true, "temperature": 0.3, "top_p": 0.9, "stop": "END", "tool_choice": "required",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": [{"type": "text", "text": "No lists."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]},
                {"role": "assistant", "content": "Let me look.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "look", "arguments": "{\"x\":1}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "zoom", "arguments": "not json"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
                {"role": "tool", "tool_call_id": "call_2", "content": "fine"},
                {"role": "user", "content": "Thanks"},
            ],
            "tools": [{"type": "function", "function": {"name": "look", "description": "Look", "parameters": {"type": "object", "properties": {"x": {"type": "number"}}}}}],
            "mirostat": 2,
        });
        let out = anthropic::request(&req).unwrap();
        assert_eq!(out["system"], "Be brief.\n\nNo lists.");
        assert_eq!(out["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(out["temperature"], 0.3);
        assert!(out.get("top_p").is_none(), "temperature and top_p are not sent together");
        assert_eq!(out["stop_sequences"], json!(["END"]));
        assert_eq!(out["stream"], true);
        assert!(out.get("mirostat").is_none());
        assert_eq!(out["tool_choice"], json!({"type": "any"}));
        assert_eq!(out["tools"][0]["input_schema"]["properties"]["x"]["type"], "number");
        let messages = out["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3, "tool results and the next user turn share one user message: {messages:?}");
        let user = &messages[0]["content"];
        assert_eq!(user[1]["source"], json!({"type": "base64", "media_type": "image/png", "data": "AAAA"}));
        assert_eq!(user[2]["source"], json!({"type": "url", "url": "https://example.com/a.jpg"}));
        let assistant = &messages[1]["content"];
        assert_eq!(assistant[0]["type"], "text");
        assert_eq!(assistant[1], json!({"type": "tool_use", "id": "call_1", "name": "look", "input": {"x": 1}}));
        assert_eq!(assistant[2]["input"], json!({}), "unparsable arguments become an empty object");
        let results = &messages[2]["content"];
        assert_eq!(results[0], json!({"type": "tool_result", "tool_use_id": "call_1", "content": "a cat"}));
        assert_eq!(results[1]["tool_use_id"], "call_2");
        assert_eq!(results[2], json!({"type": "text", "text": "Thanks"}));
    }

    #[test]
    fn request_translation_reports_problems_plainly() {
        assert!(anthropic::request(&json!({})).unwrap_err().contains("no messages"));
        let bad = json!({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]});
        assert!(anthropic::request(&bad).unwrap_err().contains("input_audio"));
        let bad = json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///x"}}]}]});
        assert!(anthropic::request(&bad).unwrap_err().contains("data: URL"));
        let top_p = anthropic::request(&json!({"messages": [], "top_p": 0.5, "max_completion_tokens": 7, "tool_choice": {"function": {"name": "f"}},
            "tools": [{"type": "function", "function": {"name": "f"}}]})).unwrap();
        assert_eq!(top_p["top_p"], 0.5);
        assert_eq!(top_p["max_tokens"], 7);
        assert_eq!(top_p["tool_choice"], json!({"type": "tool", "name": "f"}));
    }

    #[test]
    fn a_plain_answer_becomes_a_chat_completion() {
        let body = json!({
            "id": "msg_1", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "look", "input": {"x": 2}}],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
        });
        let out = anthropic::response(&body);
        let message = &out["choices"][0]["message"];
        assert_eq!(message["content"], "Checking.");
        assert_eq!(message["reasoning_content"], "hmm");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "look");
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{\"x\":2}");
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(out["usage"], json!({"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}));
    }

    #[test]
    fn a_stream_becomes_chat_completion_chunks() {
        let events = [
            json!({"type": "message_start", "message": {"id": "msg_9", "model": "claude-x", "usage": {"input_tokens": 12, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "ping"}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_7", "name": "look", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"x\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "3}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 21}}),
            json!({"type": "message_stop"}),
        ];
        let mut t = anthropic::StreamTranslator::default();
        let chunks: Vec<Value> = events.iter().flat_map(|e| t.event(e)).collect();
        assert!(t.finished);
        assert_eq!(chunks[0]["choices"][0]["delta"], json!({"role": "assistant", "content": ""}));
        let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
        assert_eq!(text, "Hello");
        let start = chunks.iter().find(|c| c["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_7").unwrap();
        assert_eq!(start["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(start["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "look");
        let args: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
            .collect();
        assert_eq!(args, "{\"x\":3}");
        let finish = chunks.iter().find(|c| c["choices"][0]["finish_reason"] == "tool_calls");
        assert!(finish.is_some());
        let usage = chunks.last().unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"], json!({"prompt_tokens": 12, "completion_tokens": 21, "total_tokens": 33}));
        assert_eq!(usage["id"], "msg_9");
    }

    #[test]
    fn stream_errors_and_stop_reasons_are_translated() {
        let mut t = anthropic::StreamTranslator::default();
        let out = t.event(&json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}));
        assert_eq!(out[0]["error"]["message"], "Overloaded");
        assert!(t.finished);
        let mut t = anthropic::StreamTranslator::default();
        let out = t.event(&json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 3}}));
        assert_eq!(out[0]["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn server_sent_events_are_split_across_chunks() {
        let mut s = anthropic::SseData::default();
        assert!(s.push(b"event: ping\r\ndata: {\"a\"").is_empty());
        assert_eq!(s.push(b":1}\r\n\r\ndata: [DONE]\n"), vec!["{\"a\":1}".to_owned(), "[DONE]".to_owned()]);
        // a multi-byte character cut in two arrives whole
        let bytes = "data: \u{e9}\n".as_bytes();
        assert!(s.push(&bytes[..7]).is_empty());
        assert_eq!(s.push(&bytes[7..]), vec!["\u{e9}".to_owned()]);
    }

    #[test]
    fn internal_hosts_are_refused_unless_the_provider_is_local() {
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        for bad in ["127.0.0.1", "10.1.2.3", "192.168.0.7", "169.254.169.254", "::1", "fd00::1", "::ffff:10.0.0.1"] {
            let err = check_addresses("h", &[ip(bad)], false).unwrap_err();
            assert!(err.contains("runs on this computer"), "{bad}: {err}");
            assert!(check_addresses("h", &[ip(bad)], true).is_ok(), "{bad} with local");
        }
        assert!(check_addresses("h", &[ip("93.184.216.34")], false).is_ok());
        assert!(check_addresses("h", &[ip("93.184.216.34"), ip("10.0.0.1")], false).is_err(), "one internal answer is enough to refuse");
        assert!(check_addresses("h", &[], true).is_err());
    }

    #[tokio::test]
    async fn clients_for_internal_hosts_are_refused_before_connecting() {
        for url in ["http://127.0.0.1:1/v1", "http://localhost:1/v1", "http://[::1]:1/v1", "http://169.254.169.254/v1"] {
            let mut p = provider("openai", "");
            p.base_url = url.into();
            let err = fetch_models(&p).await.unwrap_err();
            assert!(err.contains("loopback") || err.contains("cannot resolve"), "{url}: {err}");
        }
    }

    #[test]
    fn addresses_are_validated() {
        assert!(parse_base_url("https://api.openai.com/v1").is_ok());
        for bad in ["ftp://x/v1", "file:///etc/passwd", "https://user:pw@example.com/v1", "https://example.com/v1?key=1", "not a url", "https://example.com/#x"] {
            assert!(parse_base_url(bad).is_err(), "{bad}");
        }
        assert_eq!(api_base(&Provider { base_url: "http://localhost:11434/".into(), ..provider("ollama", "") }), "http://localhost:11434/v1");
        assert_eq!(api_base(&Provider { base_url: "https://api.anthropic.com/v1".into(), ..provider("anthropic", "") }), "https://api.anthropic.com/v1");
        assert_eq!(api_base(&Provider { base_url: "https://x.example/openai/".into(), ..provider("openai", "") }), "https://x.example/openai");
    }

    #[test]
    fn keys_are_redacted_everywhere_they_could_leak() {
        let p = provider("openai", "sk-secret-123456");
        assert!(!format!("{p:?}").contains("secret"));
        assert!(format!("{p:?}").contains("<redacted>"));
        let public = p.public();
        assert_eq!(public["has_key"], true);
        assert!(!public.to_string().contains("secret"));
        assert!(public.get("api_key").is_none());
        // an upstream error that echoes the key
        let body = br#"{"error":{"message":"Incorrect API key provided: sk-secret-123456."}}"#;
        let text = upstream_error(&p, 401, body);
        assert!(!text.contains("secret"), "{text}");
        assert!(text.contains("the key was not accepted"), "{text}");
        assert_eq!(redact("abc", "abc"), "abc", "very short keys are not replaced (nothing sensible to hide)");
    }

    #[test]
    fn upstream_errors_read_plainly() {
        let p = provider("openai", "k-12345");
        assert!(upstream_error(&p, 429, b"{}").contains("rate limit or quota"));
        assert!(upstream_error(&p, 404, b"{\"error\":{\"message\":\"no such model\"}}").contains("no such model"));
        assert!(upstream_error(&p, 503, b"busy").contains("server error"));
        assert!(upstream_error(&provider("openai", ""), 401, b"").contains("none is set"));
        assert_eq!(relay_status(401), StatusCode::BAD_GATEWAY);
        assert_eq!(relay_status(429), StatusCode::TOO_MANY_REQUESTS);
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-providers-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_config_round_trips_and_stays_private() {
        let dir = scratch("roundtrip");
        let path = dir.join("providers.json");
        assert!(load_from(&path).unwrap().is_empty(), "a missing file is an empty list");
        let all = vec![provider("openai", "sk-1"), Provider { id: "q".into(), local: true, enabled: false, ..provider("anthropic", "") }];
        save_to(&path, &all).unwrap();
        assert_eq!(load_from(&path).unwrap(), all);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        std::fs::write(&path, "{ nope").unwrap();
        assert!(load_from(&path).unwrap_err().contains("not valid JSON"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_keeps_or_drops_the_key_as_documented() {
        let saved = provider("openai", "sk-keep");
        // no key in the request: it stays
        let edited = apply_fields(saved.clone(), &json!({"name": "Renamed", "enabled": false})).unwrap();
        assert_eq!(edited.api_key, "sk-keep");
        assert_eq!(edited.name, "Renamed");
        assert!(!edited.enabled);
        // an empty key removes it, a new one replaces it
        assert_eq!(apply_fields(saved.clone(), &json!({"api_key": ""})).unwrap().api_key, "");
        assert_eq!(apply_fields(saved.clone(), &json!({"api_key": "sk-new"})).unwrap().api_key, "sk-new");
        // another host without a new key: the old key does not travel along
        assert_eq!(apply_fields(saved.clone(), &json!({"base_url": "https://other.example/v1"})).unwrap().api_key, "");
        assert_eq!(apply_fields(saved.clone(), &json!({"base_url": "https://other.example/v1", "api_key": "sk-x"})).unwrap().api_key, "sk-x");
        assert_eq!(apply_fields(saved.clone(), &json!({"base_url": "https://example.com/other"})).unwrap().api_key, "sk-keep");
        // bad input
        assert!(apply_fields(saved.clone(), &json!({"kind": "gopher"})).is_err());
        assert!(apply_fields(saved.clone(), &json!({"api_key": "two words"})).is_err());
        assert!(apply_fields(saved, &json!({"name": ""})).is_err());
    }

    #[test]
    fn identifiers_and_model_lists_are_cleaned() {
        assert_eq!(slug("My Server (Home)!"), "my-server-home");
        assert_eq!(slug("???"), "provider");
        assert!(valid_id("a_b-1") && !valid_id("a b") && !valid_id(""));
        let list = parse_model_list(&json!({"data": [{"id": "models/gemini-a"}, {"id": "b"}, {"id": "b"}, {"object": "x"}]}));
        assert_eq!(list, vec!["b".to_owned(), "gemini-a".to_owned()]);
        assert_eq!(parse_model_list(&json!({"models": [{"name": "llama3:8b"}]})), vec!["llama3:8b".to_owned()]);
    }

    #[test]
    fn presets_carry_urls_and_no_keys() {
        for (_, _, kind, url, _) in PRESETS {
            assert!(KINDS.contains(&kind));
            assert!(parse_base_url(url).is_ok(), "{url}");
        }
        assert!(!presets_json().to_string().contains("api_key"));
    }

    /// One-shot HTTP server on loopback that records each request head and answers with the canned responses.
    fn serve(responses: Vec<String>) -> (u16, std::thread::JoinHandle<Vec<String>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                seen.push(String::from_utf8_lossy(&buf[..n]).into_owned());
                let _ = stream.write_all(response.as_bytes());
            }
            seen
        });
        (port, handle)
    }

    fn http(status: &str, body: &str) -> String {
        format!("HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len())
    }

    #[tokio::test]
    async fn listing_models_sends_the_key_and_explains_a_refusal() {
        let (port, server) = serve(vec![http("200 OK", r#"{"data":[{"id":"m2"},{"id":"m1"}]}"#), http("401 Unauthorized", r#"{"error":{"message":"bad"}}"#)]);
        let mut p = provider("openai", "sk-live-0001");
        p.base_url = format!("http://127.0.0.1:{port}/v1");
        p.local = true;
        assert_eq!(fetch_models(&p).await.unwrap(), vec!["m1".to_owned(), "m2".to_owned()]);
        let error = fetch_models(&p).await.unwrap_err();
        assert!(error.contains("the key was not accepted"), "{error}");
        let seen = tokio::task::spawn_blocking(move || server.join().unwrap()).await.unwrap();
        assert!(seen[0].starts_with("GET /v1/models "), "{}", seen[0]);
        assert!(seen[0].to_ascii_lowercase().contains("authorization: bearer sk-live-0001"));
    }

    #[tokio::test]
    async fn anthropic_requests_carry_their_own_headers() {
        let (port, server) = serve(vec![http("200 OK", r#"{"data":[{"id":"claude-x"}]}"#)]);
        let mut p = provider("anthropic", "sk-ant-1");
        p.base_url = format!("http://127.0.0.1:{port}");
        p.local = true;
        assert_eq!(fetch_models(&p).await.unwrap(), vec!["claude-x".to_owned()]);
        let seen = tokio::task::spawn_blocking(move || server.join().unwrap()).await.unwrap().remove(0).to_ascii_lowercase();
        assert!(seen.starts_with("get /v1/models?limit=1000 "), "{seen}");
        assert!(seen.contains("x-api-key: sk-ant-1") && seen.contains("anthropic-version: 2023-06-01"));
        assert!(!seen.contains("authorization"));
    }

    #[tokio::test]
    async fn redirects_are_not_followed() {
        let (port, server) = serve(vec!["HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:1/x\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_owned()]);
        let mut p = provider("openai", "sk-1234");
        p.base_url = format!("http://127.0.0.1:{port}/v1");
        p.local = true;
        let error = fetch_models(&p).await.unwrap_err();
        assert!(error.contains("302"), "{error}");
        let _ = tokio::task::spawn_blocking(move || server.join()).await;
    }
}
