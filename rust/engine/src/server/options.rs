//! Options of the HTTP layer (authentication, limits, timeouts, CORS, TLS, logging) shared by the
//! decoder server and the encoder server. They are parsed from the command line by
//! [`HttpOptions::parse_flag`]; `bin/serve.rs` only forwards flags it does not know itself.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

/// Which Prometheus metric names `/metrics` exposes (`--metrics-style`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricsStyle {
    Fllm,
    Vllm,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
}

/// `--rate-limit RPM[,TPM]`: requests per minute and generated plus prompt tokens per minute, per API key.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RateLimitSpec {
    pub requests_per_minute: Option<f64>,
    pub tokens_per_minute: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct HttpOptions {
    /// Accepted API keys. Empty: no authentication.
    pub api_keys: Vec<String>,
    /// Paths (exact, or a prefix when ending in `*`) that skip authentication in addition to the built-in ones.
    pub public_routes: Vec<String>,
    pub rate_limit: RateLimitSpec,
    /// `*`, or the list of allowed browser origins. Empty: no cross-origin access.
    pub cors_origins: Vec<String>,
    /// Longest time a non-streaming request may take until its response starts (0 = unlimited).
    pub request_timeout: Duration,
    /// Longest time a client may take to send the request headers.
    pub header_read_timeout: Duration,
    /// Longest idle time of a keep-alive connection between two requests.
    pub keep_alive_timeout: Duration,
    /// Longest idle time between two SSE / streaming events before aborting the stream (0 = unlimited).
    pub stream_idle_timeout: Duration,
    pub max_connections: usize,
    /// Largest accepted request body (the file upload route has its own limit, `FLLM_MAX_FILE_SIZE`).
    pub max_body_bytes: usize,
    /// Per-route request body size limits (`--route-body-limit PATH=SIZE` or `[body_limits]` in `fllm.toml`).
    pub route_body_limits: BTreeMap<String, usize>,
    pub shutdown_grace: Duration,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub allow_remote_images: bool,
    pub access_log: bool,
    pub metrics_style: MetricsStyle,
    pub log_level: Option<String>,
    pub log_format: LogFormat,
    /// Admission limit: requests that may wait for a free sequence slot. `None`: 8 times `--max-seqs`.
    pub max_queue: Option<usize>,
    /// Further names (besides the served model name) a request may use in its `model` field; `*` matches any suffix.
    pub model_aliases: Vec<String>,
    /// Maximum number of models concurrently marked loaded in the multi-model registry (`--max-loaded-models`).
    pub max_loaded_models: usize,
    /// Optional memory budget in bytes for loaded models in the multi-model registry (`--model-memory-budget`).
    pub model_memory_budget_bytes: Option<u64>,
    /// Optional path to `fllm.toml` configuration file (`--config` or `FLLM_CONFIG`).
    pub config_path: Option<PathBuf>,
    /// Server-level settings parsed from `fllm.toml` (`model`, `host`, `port`, `ctx`, `device`, etc.).
    pub server_overrides: BTreeMap<String, String>,
    /// Where the keys made on the Studio page are kept (hashed), so they survive a restart; `None`: they live in memory only.
    pub key_store: Option<PathBuf>,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            public_routes: Vec::new(),
            rate_limit: RateLimitSpec::default(),
            cors_origins: Vec::new(),
            request_timeout: Duration::from_secs(600),
            header_read_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(75),
            stream_idle_timeout: Duration::ZERO,
            max_connections: 1024,
            max_body_bytes: 16 * 1024 * 1024,
            route_body_limits: BTreeMap::new(),
            shutdown_grace: Duration::from_secs(30),
            tls_cert: None,
            tls_key: None,
            allow_remote_images: false,
            access_log: false,
            metrics_style: MetricsStyle::Both,
            log_level: None,
            log_format: LogFormat::Text,
            max_queue: None,
            model_aliases: Vec::new(),
            max_loaded_models: 4,
            model_memory_budget_bytes: None,
            config_path: None,
            server_overrides: BTreeMap::new(),
            key_store: None,
        }
    }
}

/// The flags handled here, for the usage text.
pub const USAGE: &str = "\
       http: [--config FILE] [--api-key KEY]... [--api-key-file FILE]... [--public-routes PATHS] [--metrics-no-auth]
             [--rate-limit RPM[,TPM]] [--cors-origins '*'|ORIGIN,...] [--request-timeout S] [--header-read-timeout S]
             [--keep-alive-timeout S] [--stream-idle-timeout S] [--max-connections N] [--max-body-size BYTES|NK|NM]
             [--route-body-limit PATH=SIZE]... [--shutdown-grace S] [--tls-cert FILE --tls-key FILE]
             [--allow-remote-images] [--max-queue N] [--model-alias NAME]... [--max-loaded-models N]
             [--model-memory-budget SIZE] [--metrics-style fllm|vllm|both] [--log-level LEVEL]
             [--log-format text|json] [--access-log]
       environment: FLLM_CONFIG, FLLM_API_KEY (comma separated keys; used only when no --api-key/--api-key-file is given), RUST_LOG";

fn duration(flag: &str, raw: &str) -> Result<Duration, String> {
    let seconds: f64 = raw
        .trim()
        .trim_end_matches('s')
        .parse()
        .map_err(|_| format!("invalid value for {flag}: expected a number of seconds"))?;
    if !seconds.is_finite() || !(0.0..=86_400.0 * 30.0).contains(&seconds) {
        return Err(format!("{flag} must be between 0 and 2592000 seconds"));
    }
    Ok(Duration::from_secs_f64(seconds))
}

/// `1048576`, `64K`, `16M`, `1G` (binary multiples; a trailing `B`/`iB` is accepted).
pub fn parse_size(flag: &str, raw: &str) -> Result<usize, String> {
    let text = raw.trim().to_ascii_lowercase();
    let text = text.trim_end_matches('b').trim_end_matches('i');
    let (digits, factor) = match text.chars().last() {
        Some('k') => (&text[..text.len() - 1], 1usize << 10),
        Some('m') => (&text[..text.len() - 1], 1 << 20),
        Some('g') => (&text[..text.len() - 1], 1 << 30),
        _ => (text, 1),
    };
    let value: usize = digits
        .trim()
        .parse()
        .map_err(|_| format!("invalid size for {flag}: {raw}"))?;
    value
        .checked_mul(factor)
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("{flag} must be a positive size"))
}

fn parse_rate_limit(raw: &str) -> Result<RateLimitSpec, String> {
    let mut parts = raw.split(',');
    let mut next = |what: &str| -> Result<Option<f64>, String> {
        match parts.next().map(str::trim) {
            None | Some("") => Ok(None),
            Some(text) => {
                let value: f64 = text
                    .parse()
                    .map_err(|_| format!("--rate-limit {what} must be a number"))?;
                if !value.is_finite() || value < 0.0 {
                    return Err(format!("--rate-limit {what} must not be negative"));
                }
                Ok((value > 0.0).then_some(value))
            }
        }
    };
    let spec = RateLimitSpec {
        requests_per_minute: next("RPM")?,
        tokens_per_minute: next("TPM")?,
    };
    if parts.next().is_some() {
        return Err("--rate-limit takes RPM[,TPM]".into());
    }
    Ok(spec)
}

fn parse_route_body_limit(flag: &str, raw: &str) -> Result<(String, usize), String> {
    let Some((route, size)) = raw.split_once('=') else {
        return Err(format!("{flag} expects PATH=SIZE (e.g. /tokenize=256K)"));
    };
    let route = route.trim();
    if !route.starts_with('/') {
        return Err(format!("{flag} route must start with '/': {route}"));
    }
    let bytes = parse_size(flag, size.trim())?;
    Ok((route.to_owned(), bytes))
}

fn parse_bool(key: &str, raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean for {key}: {raw}")),
    }
}

fn unquote_toml(raw: &str) -> String {
    let trimmed = raw.trim();
    if (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
        || (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
    {
        trimmed[1..trimmed.len() - 1].to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn parse_toml_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        trimmed[1..trimmed.len() - 1]
            .split(',')
            .map(unquote_toml)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        unquote_toml(trimmed)
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    }
}

fn strip_inline_comment(line: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_double => {
                i += 2;
                continue;
            }
            b'"' if !in_single => in_double = !in_double,
            b'\'' if !in_double => in_single = !in_single,
            b'#' if !in_single && !in_double => return &line[..i],
            _ => {}
        }
        i += 1;
    }
    line
}

/// Keys of a key file: one per line, blank lines and lines starting with `#` ignored.
pub fn read_key_file(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read --api-key-file {path}: {e}"))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect())
}

impl HttpOptions {
    /// Effective body size limit in bytes for a given HTTP route path.
    pub fn body_limit_for(&self, path: &str) -> usize {
        if let Some(&limit) = self.route_body_limits.get(path) {
            return limit;
        }
        if let Some((_, &limit)) = self
            .route_body_limits
            .iter()
            .filter_map(|(pattern, limit)| pattern.strip_suffix('*').map(|prefix| (prefix, limit)))
            .filter(|(prefix, _)| path.starts_with(*prefix))
            .max_by_key(|(prefix, _)| prefix.len())
        {
            return limit;
        }
        if path == "/v1/files" || path.starts_with("/api/blobs/") {
            return super::api::batches::max_upload_bytes();
        }
        // Control and metadata routes default to at most 2 MiB (unless --max-body-size is smaller).
        if matches!(
            path,
            "/tokenize"
                | "/detokenize"
                | "/apply-template"
                | "/v1/messages/count_tokens"
                | "/api/show"
                | "/api/copy"
                | "/api/delete"
                | "/api/create"
                | "/api/pull"
                | "/api/push"
                | "/props"
                | "/lora-adapters"
                | "/v1/batches"
        ) {
            return (2 * 1024 * 1024).min(self.max_body_bytes);
        }
        self.max_body_bytes
    }

    /// Loads `fllm.toml` (or key-value config file) from `path`.
    pub fn load_config_file(&mut self, path: &Path) -> Result<(), String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read config file {}: {e}", path.display()))?;
        self.load_config_str(&text)?;
        self.config_path = Some(path.to_path_buf());
        Ok(())
    }

    /// Reloads options from `self.config_path` if one is configured. Returns `Ok(true)` if reloaded.
    pub fn reload(&mut self) -> Result<bool, String> {
        let Some(path) = self.config_path.clone() else {
            return Ok(false);
        };
        let mut fresh = HttpOptions {
            config_path: Some(path.clone()),
            ..HttpOptions::default()
        };
        fresh.load_config_file(&path)?;
        fresh.apply_environment();
        fresh.validate()?;
        *self = fresh;
        Ok(true)
    }

    /// Parses TOML / key-value configuration text into `self`.
    pub fn load_config_str(&mut self, text: &str) -> Result<(), String> {
        let mut section = String::new();
        for (lineno, raw_line) in text.lines().enumerate() {
            let line = strip_inline_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].trim().to_ascii_lowercase();
                continue;
            }
            let Some((raw_key, raw_val)) = line.split_once('=') else {
                return Err(format!("config line {}: expected KEY = VALUE", lineno + 1));
            };
            let key = unquote_toml(raw_key);
            let val = unquote_toml(raw_val);
            if section == "body_limits" || section == "route_body_limits" {
                if !key.starts_with('/') {
                    return Err(format!(
                        "config line {}: route in [{section}] must start with '/': {key}",
                        lineno + 1
                    ));
                }
                let bytes = parse_size(&key, &val)?;
                self.route_body_limits.insert(key, bytes);
                continue;
            }
            let norm_key = key.trim().to_ascii_lowercase().replace('-', "_");
            match norm_key.as_str() {
                "api_key" | "api_keys" => {
                    let keys = parse_toml_list(raw_val);
                    if keys.is_empty() {
                        return Err("api_key must not be empty".into());
                    }
                    self.api_keys.extend(keys);
                }
                "api_key_file" => {
                    let keys = read_key_file(&val)?;
                    if keys.is_empty() {
                        return Err("api_key_file contains no keys".into());
                    }
                    self.api_keys.extend(keys);
                }
                "public_routes" => {
                    self.public_routes.extend(parse_toml_list(raw_val));
                }
                "metrics_no_auth" => {
                    if parse_bool("metrics_no_auth", &val)? && !self.public_routes.iter().any(|r| r == "/metrics") {
                        self.public_routes.push("/metrics".into());
                    }
                }
                "rate_limit" => self.rate_limit = parse_rate_limit(&val)?,
                "cors_origins" => {
                    self.cors_origins = parse_toml_list(raw_val)
                        .into_iter()
                        .map(|s| s.trim_end_matches('/').to_owned())
                        .collect();
                }
                "request_timeout" => self.request_timeout = duration("--request-timeout", &val)?,
                "header_read_timeout" => self.header_read_timeout = duration("--header-read-timeout", &val)?,
                "keep_alive_timeout" => self.keep_alive_timeout = duration("--keep-alive-timeout", &val)?,
                "stream_idle_timeout" => self.stream_idle_timeout = duration("--stream-idle-timeout", &val)?,
                "shutdown_grace" => self.shutdown_grace = duration("--shutdown-grace", &val)?,
                "max_connections" => {
                    self.max_connections = val
                        .parse()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or("max_connections must be a positive integer")?;
                }
                "max_body_size" | "max_body_bytes" => {
                    self.max_body_bytes = parse_size("--max-body-size", &val)?;
                }
                "route_body_limit" => {
                    let (route, bytes) = parse_route_body_limit("--route-body-limit", &val)?;
                    self.route_body_limits.insert(route, bytes);
                }
                "tls_cert" => self.tls_cert = (!val.is_empty()).then(|| PathBuf::from(&val)),
                "tls_key" => self.tls_key = (!val.is_empty()).then(|| PathBuf::from(&val)),
                "allow_remote_images" => self.allow_remote_images = parse_bool("allow_remote_images", &val)?,
                "access_log" => self.access_log = parse_bool("access_log", &val)?,
                "max_queue" => {
                    self.max_queue = Some(
                        val.parse()
                            .map_err(|_| "max_queue must be a non-negative integer".to_owned())?,
                    );
                }
                "model_alias" | "model_aliases" => {
                    self.model_aliases.extend(parse_toml_list(raw_val));
                }
                "max_loaded_models" => {
                    self.max_loaded_models = val
                        .parse()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or("max_loaded_models must be a positive integer")?;
                }
                "model_memory_budget" | "model_memory_budget_bytes" => {
                    self.model_memory_budget_bytes = Some(parse_size("--model-memory-budget", &val)? as u64);
                }
                "metrics_style" => {
                    self.metrics_style = match val.to_ascii_lowercase().as_str() {
                        "fllm" => MetricsStyle::Fllm,
                        "vllm" => MetricsStyle::Vllm,
                        "both" => MetricsStyle::Both,
                        _ => return Err("metrics_style must be fllm, vllm or both".into()),
                    };
                }
                "log_level" => self.log_level = (!val.is_empty()).then_some(val),
                "log_format" => {
                    self.log_format = match val.to_ascii_lowercase().as_str() {
                        "text" => LogFormat::Text,
                        "json" => LogFormat::Json,
                        _ => return Err("log_format must be text or json".into()),
                    };
                }
                "host" | "port" | "model" | "model_name" | "served_name" | "ctx" | "ctx_size"
                | "max_model_len" | "max_seqs" | "max_batched_tokens" | "threads" | "device"
                | "gpu_memory_utilization" | "gpu_layers" | "kv_cache_type" | "tool_parser"
                | "reasoning_parser" | "temp" | "temperature" | "top_p" | "top_k" => {
                    self.server_overrides.insert(norm_key, val);
                }
                _ => return Err(format!("unknown config key on line {}: {key}", lineno + 1)),
            }
        }
        Ok(())
    }

    /// Handles `args[*index]` when it is one of the HTTP flags; returns `Ok(false)` for any other flag.
    /// On success `*index` points at the last argument that was consumed.
    pub fn parse_flag(&mut self, args: &[String], index: &mut usize) -> Result<bool, String> {
        let flag = args[*index].clone();
        let value = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--config" => {
                let path = PathBuf::from(value(index)?);
                self.load_config_file(&path)?;
            }
            "--api-key" => {
                let key = value(index)?;
                if key.is_empty() {
                    return Err("--api-key must not be empty".into());
                }
                self.api_keys.push(key);
            }
            "--api-key-file" => {
                let keys = read_key_file(&value(index)?)?;
                if keys.is_empty() {
                    return Err("--api-key-file contains no keys".into());
                }
                self.api_keys.extend(keys);
            }
            "--public-routes" => {
                let list = value(index)?;
                self.public_routes
                    .extend(list.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned));
            }
            "--metrics-no-auth" => self.public_routes.push("/metrics".into()),
            "--rate-limit" => self.rate_limit = parse_rate_limit(&value(index)?)?,
            "--cors-origins" => {
                let list = value(index)?;
                self.cors_origins = list
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim_end_matches('/').to_owned())
                    .collect();
            }
            "--request-timeout" => self.request_timeout = duration(&flag, &value(index)?)?,
            "--header-read-timeout" => self.header_read_timeout = duration(&flag, &value(index)?)?,
            "--keep-alive-timeout" => self.keep_alive_timeout = duration(&flag, &value(index)?)?,
            "--stream-idle-timeout" => self.stream_idle_timeout = duration(&flag, &value(index)?)?,
            "--shutdown-grace" => self.shutdown_grace = duration(&flag, &value(index)?)?,
            "--max-connections" => {
                self.max_connections = value(index)?
                    .parse()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or("--max-connections must be a positive integer")?
            }
            "--max-body-size" => self.max_body_bytes = parse_size(&flag, &value(index)?)?,
            "--route-body-limit" => {
                let (route, bytes) = parse_route_body_limit(&flag, &value(index)?)?;
                self.route_body_limits.insert(route, bytes);
            }
            "--tls-cert" => self.tls_cert = Some(PathBuf::from(value(index)?)),
            "--tls-key" => self.tls_key = Some(PathBuf::from(value(index)?)),
            "--allow-remote-images" => self.allow_remote_images = true,
            "--access-log" => self.access_log = true,
            "--max-queue" => {
                self.max_queue = Some(
                    value(index)?
                        .parse()
                        .map_err(|_| "--max-queue must be a non-negative integer".to_owned())?,
                )
            }
            "--model-alias" => {
                let alias = value(index)?;
                self.model_aliases
                    .extend(alias.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned));
            }
            "--max-loaded-models" => {
                self.max_loaded_models = value(index)?
                    .parse()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or("--max-loaded-models must be a positive integer")?;
            }
            "--model-memory-budget" => {
                self.model_memory_budget_bytes = Some(parse_size(&flag, &value(index)?)? as u64);
            }
            "--metrics-style" => {
                self.metrics_style = match value(index)?.to_ascii_lowercase().as_str() {
                    "fllm" => MetricsStyle::Fllm,
                    "vllm" => MetricsStyle::Vllm,
                    "both" => MetricsStyle::Both,
                    _ => return Err("--metrics-style must be fllm, vllm or both".into()),
                }
            }
            "--log-level" => self.log_level = Some(value(index)?),
            "--log-format" => {
                self.log_format = match value(index)?.to_ascii_lowercase().as_str() {
                    "text" => LogFormat::Text,
                    "json" => LogFormat::Json,
                    _ => return Err("--log-format must be text or json".into()),
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Keys from `FLLM_API_KEY` (comma separated) and config from `FLLM_CONFIG` when no flag supplied any.
    pub fn apply_environment(&mut self) {
        // keys made on the Studio page are kept next to its settings; FLLM_KEY_STORE=off keeps them in memory only
        if self.key_store.is_none() && std::env::var("FLLM_KEY_STORE").map_or(true, |v| !v.eq_ignore_ascii_case("off")) {
            self.key_store = Some(crate::hub::studio_data_file("api-keys.json"));
        }
        if self.config_path.is_none() {
            if let Ok(path) = std::env::var("FLLM_CONFIG") {
                let path = path.trim();
                if !path.is_empty() && Path::new(path).is_file() {
                    let _ = self.load_config_file(Path::new(path));
                }
            }
        }
        if self.api_keys.is_empty() {
            if let Ok(list) = std::env::var("FLLM_API_KEY") {
                self.api_keys = list
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
            }
        }
        if let Ok(raw) = std::env::var("FLLM_MAX_LOADED_MODELS") {
            if let Ok(n) = raw.trim().parse::<usize>() {
                if n > 0 {
                    self.max_loaded_models = n;
                }
            }
        }
        if self.stream_idle_timeout.is_zero() {
            if let Ok(raw) = std::env::var("FLLM_STREAM_IDLE_TIMEOUT") {
                if let Ok(d) = duration("FLLM_STREAM_IDLE_TIMEOUT", &raw) {
                    self.stream_idle_timeout = d;
                }
            }
        }
    }

    /// Cross-field checks that must fail before the model is loaded.
    pub fn validate(&self) -> Result<(), String> {
        if self.tls_cert.is_some() != self.tls_key.is_some() {
            return Err("--tls-cert and --tls-key must be given together".into());
        }
        if let (Some(cert), Some(key)) = (&self.tls_cert, &self.tls_key) {
            super::tls::acceptor(cert, key)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<HttpOptions, String> {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let mut options = HttpOptions::default();
        let mut i = 0;
        while i < args.len() {
            assert!(options.parse_flag(&args, &mut i)?, "unhandled {}", args[i]);
            i += 1;
        }
        Ok(options)
    }

    #[test]
    fn flags_fill_the_options() {
        let o = parse(&[
            "--api-key", "a", "--api-key", "b", "--rate-limit", "60,1000", "--cors-origins", "http://x.test/, http://y.test",
            "--request-timeout", "12.5", "--stream-idle-timeout", "45", "--max-body-size", "2M",
            "--route-body-limit", "/tokenize=128K", "--max-loaded-models", "6", "--model-memory-budget", "8G",
            "--metrics-style", "vllm", "--metrics-no-auth", "--model-alias", "gpt-4o,claude-*",
        ])
        .unwrap();
        assert_eq!(o.api_keys, ["a", "b"]);
        assert_eq!(o.rate_limit, RateLimitSpec { requests_per_minute: Some(60.0), tokens_per_minute: Some(1000.0) });
        assert_eq!(o.cors_origins, ["http://x.test", "http://y.test"]);
        assert_eq!(o.request_timeout, Duration::from_millis(12_500));
        assert_eq!(o.stream_idle_timeout, Duration::from_secs(45));
        assert_eq!(o.max_body_bytes, 2 << 20);
        assert_eq!(o.body_limit_for("/tokenize"), 128 << 10);
        assert_eq!(o.max_loaded_models, 6);
        assert_eq!(o.model_memory_budget_bytes, Some(8 << 30));
        assert_eq!(o.metrics_style, MetricsStyle::Vllm);
        assert_eq!(o.public_routes, ["/metrics"]);
        assert_eq!(o.model_aliases, ["gpt-4o", "claude-*"]);
    }

    #[test]
    fn bad_values_are_rejected() {
        for args in [
            &["--rate-limit", "x"][..],
            &["--rate-limit", "1,2,3"],
            &["--request-timeout", "-1"],
            &["--stream-idle-timeout", "-5"],
            &["--max-body-size", "0"],
            &["--route-body-limit", "tokenize=10K"],
            &["--max-loaded-models", "0"],
            &["--metrics-style", "x"],
            &["--api-key"],
            &["--max-connections", "0"],
        ] {
            assert!(parse(args).is_err(), "{args:?}");
        }
        assert!(HttpOptions { tls_cert: Some("c".into()), ..Default::default() }.validate().is_err());
    }

    #[test]
    fn sizes_accept_binary_suffixes() {
        assert_eq!(parse_size("f", "16MiB").unwrap(), 16 << 20);
        assert_eq!(parse_size("f", "64k").unwrap(), 64 << 10);
        assert_eq!(parse_size("f", "123").unwrap(), 123);
        assert!(parse_size("f", "abc").is_err());
    }

    #[test]
    fn key_files_skip_comments_and_blank_lines() {
        let path = std::env::temp_dir().join(format!("fllm-keys-{}.txt", std::process::id()));
        std::fs::write(&path, "# team keys\nalpha\n\n  beta  \n#gamma\n").unwrap();
        assert_eq!(read_key_file(path.to_str().unwrap()).unwrap(), ["alpha", "beta"]);
        let mut options = HttpOptions::default();
        let args = vec!["--api-key-file".to_owned(), path.to_str().unwrap().to_owned(), "--api-key".to_owned(), "gamma".to_owned()];
        let mut i = 0;
        while i < args.len() {
            assert!(options.parse_flag(&args, &mut i).unwrap());
            i += 1;
        }
        assert_eq!(options.api_keys, ["alpha", "beta", "gamma"]);
        std::fs::write(&path, "# nothing\n").unwrap();
        assert!(options.parse_flag(&args[..2], &mut 0).is_err(), "an empty key file must not silently disable authentication");
        let _ = std::fs::remove_file(&path);
        assert!(read_key_file("/nonexistent/keys.txt").is_err());
    }

    #[test]
    fn config_file_parses_sections_and_reloads() {
        let path = std::env::temp_dir().join(format!("fllm-cfg-{}.toml", std::process::id()));
        std::fs::write(
            &path,
            r#"
# fLLM configuration
[server]
host = "0.0.0.0"
port = 9090
model = "llama3.1:8b"

[http]
api_keys = ["k1", "k2"]
rate_limit = "120,50000"
stream_idle_timeout = 30
max_loaded_models = 3
allow_remote_images = true
access_log = true
metrics_style = "vllm"

[body_limits]
"/tokenize" = "64K"
"/v1/chat/*" = "4M"
"#,
        )
        .unwrap();
        let mut o = parse(&["--config", path.to_str().unwrap()]).unwrap();
        assert_eq!(o.api_keys, ["k1", "k2"]);
        assert_eq!(o.stream_idle_timeout, Duration::from_secs(30));
        assert_eq!(o.max_loaded_models, 3);
        assert!(o.allow_remote_images);
        assert!(o.access_log);
        assert_eq!(o.metrics_style, MetricsStyle::Vllm);
        assert_eq!(o.body_limit_for("/tokenize"), 64 << 10);
        assert_eq!(o.body_limit_for("/v1/chat/completions"), 4 << 20);
        assert_eq!(o.server_overrides.get("port").map(String::as_str), Some("9090"));

        std::fs::write(
            &path,
            "[http]\napi_keys = [\"k3\"]\nmax_loaded_models = 8\n",
        )
        .unwrap();
        assert!(o.reload().unwrap());
        assert_eq!(o.api_keys, ["k3"]);
        assert_eq!(o.max_loaded_models, 8);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_flags_are_left_to_the_caller() {
        let args = vec!["--port".to_owned()];
        assert!(!HttpOptions::default().parse_flag(&args, &mut 0).unwrap());
    }
}
