//! API-key authentication and per-key rate limits.
//!
//! Keys are never compared as text: the presented credential and every configured key are hashed
//! with SHA-256 and the digests are compared with a fixed-time loop that always visits every key,
//! so response time depends neither on where the first difference is nor on the key length.
use super::options::{HttpOptions, RateLimitSpec};
use axum::http::{header, HeaderMap, Method};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// Paths that never need a key: liveness, readiness and the browser page (which asks the user for the key).
const BUILT_IN_PUBLIC: [&str; 4] = ["/health", "/ready", "/", "/studio/runner.js"];

pub fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A panic while a lock was held (for example in a request that was isolated by catch_unwind) must not take
    // every later request down with it: the protected data in this module stays consistent after any panic.
    mutex.lock().unwrap_or_else(|poisoned| {
        mutex.clear_poison();
        poisoned.into_inner()
    })
}

/// Runs `f` with the locked data, recovering from poisoning like [`lock`].
pub fn with_lock<T, R>(mutex: &Mutex<T>, f: impl FnOnce(&mut T) -> R) -> R {
    f(&mut lock(mutex))
}

fn digest(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

/// Compares two digests in time that does not depend on their contents.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut difference = 0u8;
    for (x, y) in a.iter().zip(b) {
        difference |= x ^ y;
    }
    std::hint::black_box(difference) == 0
}

/// Token bucket refilled at `per_minute / 60` per second; a level at or below zero means "limit reached".
struct Bucket {
    capacity: f64,
    state: Mutex<(f64, Instant)>,
}

impl Bucket {
    fn new(per_minute: f64) -> Self {
        Self {
            capacity: per_minute,
            state: Mutex::new((per_minute, Instant::now())),
        }
    }

    fn refilled(&self) -> std::sync::MutexGuard<'_, (f64, Instant)> {
        let mut state = lock(&self.state);
        let now = Instant::now();
        let gained = now.duration_since(state.1).as_secs_f64() * self.capacity / 60.0;
        state.0 = (state.0 + gained).min(self.capacity);
        state.1 = now;
        state
    }

    /// Takes `cost` when at least that much is available; otherwise the time until it will be.
    fn take(&self, cost: f64) -> Result<(), Duration> {
        let mut state = self.refilled();
        if state.0 >= cost {
            state.0 -= cost;
            Ok(())
        } else {
            Err(Duration::from_secs_f64(((cost - state.0) * 60.0 / self.capacity).max(0.0)))
        }
    }

    /// Fails while the bucket is empty or overdrawn.
    fn check(&self) -> Result<(), Duration> {
        let state = self.refilled();
        if state.0 > 0.0 {
            Ok(())
        } else {
            Err(Duration::from_secs_f64(((1.0 - state.0) * 60.0 / self.capacity).max(0.0)))
        }
    }

    /// Charges tokens that were already used; the level may go negative, but never below one full minute of debt.
    fn charge(&self, cost: f64) {
        let mut state = self.refilled();
        state.0 = (state.0 - cost).max(-self.capacity);
    }
}

/// Masks a secret key for display without revealing the middle bytes.
fn mask_secret(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    if chars.len() <= 8 {
        format!("{}•••", chars.first().copied().unwrap_or('*'))
    } else {
        let prefix: String = chars[..6].iter().collect();
        let suffix: String = chars[chars.len() - 4..].iter().collect();
        format!("{prefix}••••{suffix}")
    }
}

struct StoredKey {
    digest: [u8; 32],
    label: String,
    masked: String,
    admin: bool,
    created_at: u64,
    rpm: Option<f64>,
    tpm: Option<f64>,
}

/// The keys of the key store; a missing or damaged file gives none.
fn read_store(path: &std::path::Path) -> Vec<StoredKey> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(rows) = serde_json::from_str::<Vec<serde_json::Value>>(&text) else { return Vec::new() };
    rows.iter()
        .filter_map(|row| {
            let hex = row["digest"].as_str()?;
            if hex.len() != 64 {
                return None;
            }
            let mut digest = [0u8; 32];
            for (i, byte) in digest.iter_mut().enumerate() {
                *byte = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
            }
            Some(StoredKey {
                digest,
                label: row["label"].as_str().unwrap_or("Studio Key").to_owned(),
                masked: row["masked"].as_str().unwrap_or("").to_owned(),
                admin: row["admin"].as_bool().unwrap_or(false),
                created_at: row["created_at"].as_u64().unwrap_or(0),
                rpm: row["rpm"].as_f64(),
                tpm: row["tpm"].as_f64(),
            })
        })
        .collect()
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The limits and counters of one API key (or of everybody, when no key is configured).
pub struct KeyState {
    pub id: String,
    pub label: String,
    pub masked: String,
    /// May use the administration routes (`/admin/*`, creating and removing keys) from any machine. Keys given on the command line are
    /// the operator's and have it; keys made on the Studio page have it only when asked for.
    pub admin: bool,
    pub created_at: u64,
    /// Made on the Studio page (kept in the key store), not given on the command line.
    pub runtime: bool,
    pub rpm_limit: Option<f64>,
    pub tpm_limit: Option<f64>,
    requests: Option<Bucket>,
    tokens: Option<Bucket>,
    pub accepted: AtomicU64,
    pub tokens_used: AtomicU64,
    pub rejected: AtomicU64,
}

impl KeyState {
    fn new(label: String, spec: &RateLimitSpec) -> Self {
        Self::with_meta(label.clone(), label, "cli••••key".into(), spec, false)
    }

    fn with_meta(id: String, label: String, masked: String, spec: &RateLimitSpec, admin: bool) -> Self {
        Self {
            id,
            label,
            masked,
            admin,
            created_at: unix_secs(),
            runtime: false,
            rpm_limit: spec.requests_per_minute,
            tpm_limit: spec.tokens_per_minute,
            requests: spec.requests_per_minute.map(Bucket::new),
            tokens: spec.tokens_per_minute.map(Bucket::new),
            accepted: AtomicU64::new(0),
            tokens_used: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    fn has_limits(&self) -> bool {
        self.requests.is_some() || self.tokens.is_some()
    }

    /// Admission of one request; `Err` carries the Retry-After hint.
    pub fn admit(&self) -> Result<(), Duration> {
        if let Some(tokens) = &self.tokens {
            tokens.check()?;
        }
        if let Some(requests) = &self.requests {
            requests.take(1.0)?;
        }
        Ok(())
    }

    /// Prompt and generated tokens count against the token budget as they are produced.
    pub fn charge_tokens(&self, count: usize) {
        self.tokens_used.fetch_add(count as u64, Ordering::Relaxed);
        if let Some(tokens) = &self.tokens {
            tokens.charge(count as f64);
        }
    }
}

/// Token counters of one HTTP request, for the access log and the usage metrics.
#[derive(Default)]
pub struct RequestStats {
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
}

/// Who is calling and what the call has used so far. Handlers reach it through [`current_caller`]
/// (a task-local set by the HTTP middleware), so no handler signature needs to change.
#[derive(Clone)]
pub struct Caller {
    pub limiter: Option<Arc<KeyState>>,
    pub stats: Arc<RequestStats>,
    pub request_id: String,
}

tokio::task_local! {
    pub static CALLER: Caller;
}

pub fn current_caller() -> Option<Caller> {
    CALLER.try_with(Clone::clone).ok()
}

pub enum Verdict {
    Allowed(Option<Arc<KeyState>>),
    Missing,
    Invalid,
    RateLimited { retry_after: Duration, label: String },
}

pub struct Security {
    keys: Mutex<Vec<([u8; 32], Arc<KeyState>)>>,
    open: Mutex<Arc<KeyState>>,
    public: Vec<String>,
    default_rate_limit: Mutex<RateLimitSpec>,
    next_id: AtomicU64,
    /// File that holds the keys made at run time (hashed, never the secret).
    store: Option<std::path::PathBuf>,
}

fn presented_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(value) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        let value = value.trim();
        if value.len() > 7 && value[..7].eq_ignore_ascii_case("bearer ") {
            return Some(value[7..].trim());
        }
    }
    for name in ["x-api-key", "api-key"] {
        if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
            return Some(value.trim());
        }
    }
    None
}

impl Security {
    pub fn new(options: &HttpOptions) -> Self {
        let mut keys: Vec<([u8; 32], Arc<KeyState>)> = Vec::new();
        for (index, key) in options.api_keys.iter().enumerate() {
            let d = digest(key);
            if !keys.iter().any(|(known, _)| ct_eq(known, &d)) {
                let id = format!("key-{}", index + 1);
                keys.push((
                    d,
                    Arc::new(KeyState::with_meta(
                        id.clone(),
                        id,
                        mask_secret(key),
                        &options.rate_limit,
                        true,
                    )),
                ));
            }
        }
        let mut next = keys.len() as u64 + 1;
        // keys made on the Studio page in an earlier run
        if let Some(path) = &options.key_store {
            for stored in read_store(path) {
                if keys.iter().any(|(known, _)| ct_eq(known, &stored.digest)) {
                    continue;
                }
                let mut state = KeyState::with_meta(format!("key-{next}"), stored.label, stored.masked, &RateLimitSpec { requests_per_minute: stored.rpm, tokens_per_minute: stored.tpm }, stored.admin);
                state.runtime = true;
                state.created_at = stored.created_at;
                keys.push((stored.digest, Arc::new(state)));
                next += 1;
            }
        }
        Self {
            keys: Mutex::new(keys),
            open: Mutex::new(Arc::new(KeyState::new("anonymous".into(), &options.rate_limit))),
            public: options.public_routes.clone(),
            default_rate_limit: Mutex::new(options.rate_limit),
            next_id: AtomicU64::new(next),
            store: options.key_store.clone(),
        }
    }

    /// Writes the keys made at run time (digest, label, limits; never the secret) to the key store.
    fn save(&self, keys: &[([u8; 32], Arc<KeyState>)]) {
        let Some(path) = &self.store else { return };
        let rows: Vec<serde_json::Value> = keys
            .iter()
            .filter(|(_, s)| s.runtime)
            .map(|(d, s)| {
                serde_json::json!({
                    "digest": d.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                    "label": s.label, "masked": s.masked, "admin": s.admin, "created_at": s.created_at,
                    "rpm": s.rpm_limit, "tpm": s.tpm_limit,
                })
            })
            .collect();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let temp = path.with_extension("json.tmp");
        if std::fs::write(&temp, serde_json::Value::Array(rows).to_string()).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600));
            }
            let _ = std::fs::rename(&temp, path);
        }
    }

    pub fn auth_required(&self) -> bool {
        !lock(&self.keys).is_empty()
    }

    /// Whether the request presents a key that has the admin right.
    pub fn is_admin_request(&self, headers: &HeaderMap) -> bool {
        let Some(raw) = presented_key(headers) else { return false };
        let d = digest(raw);
        lock(&self.keys).iter().any(|(known, state)| ct_eq(known, &d) && state.admin)
    }

    pub fn rate_limit_spec(&self) -> RateLimitSpec {
        *lock(&self.default_rate_limit)
    }

    pub fn set_rate_limit(&self, spec: RateLimitSpec) {
        *lock(&self.default_rate_limit) = spec;
        *lock(&self.open) = Arc::new(KeyState::new("anonymous".into(), &spec));
    }

    pub fn list_keys(&self) -> Vec<serde_json::Value> {
        let guard = lock(&self.keys);
        guard
            .iter()
            .map(|(_, state)| {
                serde_json::json!({
                    "id": state.id,
                    "label": state.label,
                    "masked": state.masked,
                    "admin": state.admin,
                    "created_at": state.created_at,
                    "rpm_limit": state.rpm_limit,
                    "tpm_limit": state.tpm_limit,
                    "requests_accepted": state.accepted.load(Ordering::Relaxed),
                    "tokens_used": state.tokens_used.load(Ordering::Relaxed),
                    "rejected": state.rejected.load(Ordering::Relaxed),
                })
            })
            .collect()
    }

    pub fn add_key(
        &self,
        raw_key: &str,
        label: &str,
        rpm: Option<f64>,
        tpm: Option<f64>,
        admin: bool,
    ) -> Result<serde_json::Value, String> {
        let raw_key = raw_key.trim();
        if raw_key.is_empty() {
            return Err("API key secret must not be empty".into());
        }
        let d = digest(raw_key);
        let mut guard = lock(&self.keys);
        if guard.iter().any(|(known, _)| ct_eq(known, &d)) {
            return Err("this API key is already registered".into());
        }
        let defaults = *lock(&self.default_rate_limit);
        let spec = RateLimitSpec {
            requests_per_minute: rpm.or(defaults.requests_per_minute),
            tokens_per_minute: tpm.or(defaults.tokens_per_minute),
        };
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("key-{seq}");
        let clean_label = if label.trim().is_empty() {
            id.clone()
        } else {
            label.trim().to_owned()
        };
        let mut made = KeyState::with_meta(id.clone(), clean_label.clone(), mask_secret(raw_key), &spec, admin);
        made.runtime = true;
        let state = Arc::new(made);
        let info = serde_json::json!({
            "id": state.id,
            "label": state.label,
            "masked": state.masked,
            "admin": state.admin,
            "created_at": state.created_at,
            "rpm_limit": state.rpm_limit,
            "tpm_limit": state.tpm_limit,
            "requests_accepted": 0,
            "tokens_used": 0,
            "rejected": 0,
        });
        guard.push((d, state));
        self.save(&guard);
        Ok(info)
    }

    pub fn remove_key(&self, id_or_label: &str) -> bool {
        let mut guard = lock(&self.keys);
        let before = guard.len();
        guard.retain(|(_, state)| state.id != id_or_label && state.label != id_or_label);
        let removed = guard.len() < before;
        if removed {
            self.save(&guard);
        }
        removed
    }

    fn is_public(&self, method: &Method, path: &str) -> bool {
        if BUILT_IN_PUBLIC.contains(&path) && (method == Method::GET || method == Method::HEAD) {
            return true;
        }
        self.public.iter().any(|rule| match rule.strip_suffix('*') {
            Some(prefix) => path.starts_with(prefix),
            None => rule == path,
        })
    }

    /// Whether a request is counted against the request-rate limit (anything that can run inference or change state).
    fn counted(method: &Method) -> bool {
        method == Method::POST || method == Method::PUT || method == Method::DELETE
    }

    pub fn check(&self, method: &Method, path: &str, headers: &HeaderMap) -> Verdict {
        if self.is_public(method, path) {
            return Verdict::Allowed(None);
        }
        let guard = lock(&self.keys);
        let limiter = if guard.is_empty() {
            drop(guard);
            lock(&self.open).clone()
        } else {
            let Some(key) = presented_key(headers) else {
                return Verdict::Missing;
            };
            let presented = digest(key);
            let mut found = None;
            for (known, state) in guard.iter() {
                // no early exit: every configured key is visited
                if ct_eq(known, &presented) & found.is_none() {
                    found = Some(state.clone());
                }
            }
            drop(guard);
            match found {
                Some(state) => state,
                None => return Verdict::Invalid,
            }
        };
        if limiter.has_limits() && Self::counted(method) {
            if let Err(retry_after) = limiter.admit() {
                limiter.rejected.fetch_add(1, Ordering::Relaxed);
                return Verdict::RateLimited { retry_after, label: limiter.label.clone() };
            }
        }
        limiter.accepted.fetch_add(1, Ordering::Relaxed);
        Verdict::Allowed(Some(limiter))
    }

    pub fn rejected_total(&self) -> u64 {
        let keys_rejected: u64 = lock(&self.keys)
            .iter()
            .map(|(_, state)| state.rejected.load(Ordering::Relaxed))
            .sum();
        keys_rejected + lock(&self.open).rejected.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn security(keys: &[&str], rate: RateLimitSpec) -> Security {
        Security::new(&HttpOptions {
            api_keys: keys.iter().map(|k| k.to_string()).collect(),
            rate_limit: rate,
            ..Default::default()
        })
    }

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn keys_made_at_run_time_survive_a_restart_as_digests_only() {
        let dir = std::env::temp_dir().join(format!("fllm-keystore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = dir.join("api-keys.json");
        let options = HttpOptions { key_store: Some(store.clone()), api_keys: vec!["from-the-command-line".into()], ..Default::default() };
        let first = Security::new(&options);
        let info = first.add_key("fllm-sk-secret-value-0123456789", "Cursor", Some(60.0), None, false).unwrap();
        let text = std::fs::read_to_string(&store).unwrap();
        assert!(!text.contains("secret-value"), "the secret itself is never written: {text}");
        assert!(!text.contains("from-the-command-line"), "keys of the command line are not stored");
        let id = info["id"].as_str().unwrap().to_owned();
        // a new process: the key works, with its label and limit, and the command line key is still there
        let second = Security::new(&options);
        let auth = |s: &Security, key: &str| matches!(s.check(&Method::POST, "/v1/models", &headers(&[("authorization", &format!("Bearer {key}"))])), Verdict::Allowed(_));
        assert!(auth(&second, "fllm-sk-secret-value-0123456789") && auth(&second, "from-the-command-line") && !auth(&second, "fllm-sk-secret-value-0123456780"));
        let listed = second.list_keys();
        let mine = listed.iter().find(|k| k["label"] == "Cursor").unwrap();
        assert_eq!(mine["rpm_limit"], 60.0);
        assert_eq!(mine["masked"], "fllm-s••••6789");
        // revoking removes it from the store as well
        let revoke = mine["id"].as_str().unwrap().to_owned();
        assert!(second.remove_key(&revoke));
        assert!(!auth(&Security::new(&options), "fllm-sk-secret-value-0123456789"));
        let _ = id;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keys_are_matched_by_digest_from_any_header() {
        let s = security(&["alpha", "beta"], RateLimitSpec::default());
        let ok = |h: &HeaderMap| matches!(s.check(&Method::POST, "/v1/chat/completions", h), Verdict::Allowed(_));
        assert!(ok(&headers(&[("authorization", "Bearer alpha")])));
        assert!(ok(&headers(&[("authorization", "bearer beta")])));
        assert!(ok(&headers(&[("x-api-key", "beta")])));
        assert!(!ok(&headers(&[("authorization", "Bearer alph")])));
        assert!(!ok(&headers(&[("authorization", "Bearer alphaa")])));
        assert!(matches!(s.check(&Method::POST, "/v1/models", &HeaderMap::new()), Verdict::Missing));
        assert!(matches!(
            s.check(&Method::GET, "/v1/models", &headers(&[("authorization", "Basic YWxwaGE=")])),
            Verdict::Missing
        ));
    }

    #[test]
    fn public_routes_skip_the_key() {
        let s = Security::new(&HttpOptions {
            api_keys: vec!["k".into()],
            public_routes: vec!["/metrics".into(), "/api/v*".into()],
            ..Default::default()
        });
        for path in ["/health", "/ready", "/", "/metrics", "/api/version"] {
            assert!(matches!(s.check(&Method::GET, path, &HeaderMap::new()), Verdict::Allowed(_)), "{path}");
        }
        for path in ["/v1/models", "/api/tags", "/get_model_info", "/props"] {
            assert!(matches!(s.check(&Method::GET, path, &HeaderMap::new()), Verdict::Missing), "{path}");
        }
        // only reading the page is public
        assert!(matches!(s.check(&Method::POST, "/", &HeaderMap::new()), Verdict::Missing));
    }

    #[test]
    fn no_keys_means_open_access() {
        let s = security(&[], RateLimitSpec::default());
        assert!(!s.auth_required());
        assert!(matches!(s.check(&Method::POST, "/v1/completions", &HeaderMap::new()), Verdict::Allowed(_)));
    }

    #[test]
    fn rate_limits_are_per_key() {
        let s = security(&["a", "b"], RateLimitSpec { requests_per_minute: Some(2.0), tokens_per_minute: None });
        let call = |key: &str| s.check(&Method::POST, "/v1/completions", &headers(&[("authorization", &format!("Bearer {key}"))]));
        assert!(matches!(call("a"), Verdict::Allowed(_)));
        assert!(matches!(call("a"), Verdict::Allowed(_)));
        match call("a") {
            Verdict::RateLimited { retry_after, .. } => assert!(retry_after > Duration::from_secs(1)),
            _ => panic!("third request must be limited"),
        }
        assert!(matches!(call("b"), Verdict::Allowed(_)), "the other key has its own budget");
        assert_eq!(s.rejected_total(), 1);
    }

    #[test]
    fn token_budget_is_charged_after_the_fact() {
        let s = security(&[], RateLimitSpec { requests_per_minute: None, tokens_per_minute: Some(100.0) });
        let Verdict::Allowed(Some(state)) = s.check(&Method::POST, "/v1/completions", &HeaderMap::new()) else {
            panic!("allowed")
        };
        state.charge_tokens(150);
        assert!(matches!(
            s.check(&Method::POST, "/v1/completions", &HeaderMap::new()),
            Verdict::RateLimited { .. }
        ));
    }

    #[test]
    fn a_poisoned_lock_is_recovered() {
        let m = Arc::new(Mutex::new(1));
        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _guard = m2.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(m.is_poisoned());
        assert_eq!(*lock(&m), 1);
        assert!(!m.is_poisoned());
    }
}
