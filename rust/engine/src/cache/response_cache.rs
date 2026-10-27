//! The response cache: an answer that was produced before is handed out again instead of being computed again.
//!
//! It is only used where the answer is determined by the request: `temperature = 0`, no `n > 1`, no logprobs, no tools,
//! no images. The key is a SHA-256 over everything that changes the answer — model name, template, quantization, LoRA
//! adapters, the sampling parameters, the grammar and the prompt **token ids** — so two requests share an entry only when
//! they would have produced the same text. A tenant salt keeps different API keys apart, the same way the prefix cache
//! does it.
//!
//! The cache is bounded twice: by bytes (LRU, the least recently used entry goes first) and by age (TTL). Every hit,
//! miss and eviction is counted, so `/metrics` and the admin route can show whether it earns its memory.
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How much a cache entry may differ between tenants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// One namespace for everybody (single user, a workstation).
    Global,
    /// One namespace per API key: an answer of one tenant is never handed to another.
    Tenant,
    /// The cache is not used at all.
    Off,
}

impl Scope {
    pub fn parse(text: &str) -> Scope {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" | "false" => Scope::Off,
            "tenant" | "key" | "per-key" => Scope::Tenant,
            _ => Scope::Global,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Scope::Global => "global",
            Scope::Tenant => "tenant",
            Scope::Off => "off",
        }
    }
}

/// Everything that makes two requests the same request.
#[derive(Debug, Clone, Default)]
pub struct Fingerprint {
    pub model: String,
    pub template: String,
    pub quantization: String,
    pub loras: Vec<String>,
    /// Canonical sampling parameters as `key=value` pairs, sorted by key by the caller.
    pub sampling: Vec<(String, String)>,
    /// Fingerprint of the grammar / JSON schema, empty when generation is unconstrained.
    pub grammar: String,
}

impl Fingerprint {
    /// The cache key for one prompt. `salt` is the tenant namespace (empty for `Scope::Global`).
    pub fn key(&self, prompt: &[i64], salt: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"fllm-output-cache-v1\0");
        hasher.update(salt.as_bytes());
        hasher.update(b"\0model\0");
        hasher.update(self.model.as_bytes());
        hasher.update(b"\0template\0");
        hasher.update(self.template.as_bytes());
        hasher.update(b"\0quant\0");
        hasher.update(self.quantization.as_bytes());
        hasher.update(b"\0lora\0");
        for lora in &self.loras {
            hasher.update(lora.as_bytes());
            hasher.update(b"\x1f");
        }
        hasher.update(b"\0sampling\0");
        for (key, value) in &self.sampling {
            hasher.update(key.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\x1f");
        }
        hasher.update(b"\0grammar\0");
        hasher.update(self.grammar.as_bytes());
        hasher.update(b"\0prompt\0");
        for token in prompt {
            hasher.update(token.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Whether the answer to this request is determined at all — only then may it be cached.
    pub fn cacheable(&self) -> Result<(), String> {
        let number = |name: &str| {
            self.sampling
                .iter()
                .find(|(k, _)| k == name)
                .and_then(|(_, v)| v.parse::<f64>().ok())
        };
        if let Some(temperature) = number("temperature") {
            if temperature > 0.0 {
                return Err("temperature is not 0, the answer would differ".to_owned());
            }
        }
        if let Some(n) = number("n") {
            if n > 1.0 {
                return Err("n > 1".to_owned());
            }
        }
        Ok(())
    }
}

/// One cached answer.
#[derive(Debug, Clone)]
pub struct Entry {
    pub text: String,
    /// The generated token ids, for a replay that keeps token accounting exact.
    pub tokens: Vec<u32>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub created_unix: u64,
    /// The logprobs payload of the answer, when the request asked for them: an answer is only reusable with its numbers.
    pub logprobs: Option<serde_json::Value>,
    stored: Instant,
    last_used: u64,
}

impl Entry {
    pub fn new(text: String, tokens: Vec<u32>, finish_reason: String, prompt_tokens: usize) -> Entry {
        let completion_tokens = tokens.len();
        Entry {
            text,
            tokens,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            created_unix: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            logprobs: None,
            stored: Instant::now(),
            last_used: 0,
        }
    }

    /// Attaches the logprobs of the answer (builder style, so existing call sites stay short).
    pub fn with_logprobs(mut self, logprobs: Option<serde_json::Value>) -> Entry {
        self.logprobs = logprobs;
        self
    }

    fn bytes(&self) -> usize {
        self.text.len() + self.tokens.len() * 4 + self.logprobs.as_ref().map(|v| v.to_string().len()).unwrap_or(0) + 96
    }
}

struct Inner {
    entries: HashMap<[u8; 32], Entry>,
    bytes: usize,
    /// Monotonic clock for the LRU order; the entry with the smallest value goes first.
    tick: u64,
}

/// The cache itself: bounded by bytes and age, one mutex, no background thread.
pub struct ResponseCache {
    inner: Mutex<Inner>,
    max_bytes: usize,
    ttl: Duration,
    scope: Scope,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
    stored: AtomicU64,
    saved_prompt_tokens: AtomicU64,
}

impl ResponseCache {
    pub fn new(max_bytes: usize, ttl: Duration, scope: Scope) -> ResponseCache {
        ResponseCache {
            inner: Mutex::new(Inner { entries: HashMap::new(), bytes: 0, tick: 0 }),
            max_bytes,
            ttl,
            scope,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            stored: AtomicU64::new(0),
            saved_prompt_tokens: AtomicU64::new(0),
        }
    }

    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    pub fn enabled(&self) -> bool {
        self.scope != Scope::Off && self.max_bytes > 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // a panic inside the cache must not take the server down with it
        self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The cached answer, if there is one that is still fresh.
    pub fn get(&self, key: &[u8; 32]) -> Option<Entry> {
        if !self.enabled() {
            return None;
        }
        let mut inner = self.lock();
        let expired = inner.entries.get(key).map(|entry| entry.stored.elapsed() > self.ttl).unwrap_or(false);
        if expired {
            if let Some(entry) = inner.entries.remove(key) {
                inner.bytes = inner.bytes.saturating_sub(entry.bytes());
            }
            self.expirations.fetch_add(1, Ordering::Relaxed);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        inner.tick += 1;
        let tick = inner.tick;
        match inner.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used = tick;
                let mut out = entry.clone();
                out.last_used = tick;
                self.hits.fetch_add(1, Ordering::Relaxed);
                self.saved_prompt_tokens.fetch_add(out.prompt_tokens as u64, Ordering::Relaxed);
                Some(out)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Stores an answer, evicting the least recently used entries until it fits.
    pub fn put(&self, key: [u8; 32], entry: Entry) {
        if !self.enabled() {
            return;
        }
        let size = entry.bytes();
        if size > self.max_bytes {
            return; // a single answer bigger than the whole cache would evict everything
        }
        let mut inner = self.lock();
        inner.tick += 1;
        let tick = inner.tick;
        let mut entry = entry;
        entry.last_used = tick;
        if let Some(old) = inner.entries.insert(key, entry) {
            inner.bytes = inner.bytes.saturating_sub(old.bytes());
        }
        inner.bytes += size;
        self.stored.fetch_add(1, Ordering::Relaxed);
        while inner.bytes > self.max_bytes {
            let oldest = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key);
            match oldest {
                Some(key) => {
                    if let Some(entry) = inner.entries.remove(&key) {
                        inner.bytes = inner.bytes.saturating_sub(entry.bytes());
                        self.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => break,
            }
        }
    }

    /// Empties the cache and returns how many answers were dropped.
    pub fn clear(&self) -> usize {
        let mut inner = self.lock();
        let count = inner.entries.len();
        inner.entries.clear();
        inner.bytes = 0;
        count
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    /// What the admin route and `/metrics` report.
    pub fn stats(&self) -> serde_json::Value {
        let inner = self.lock();
        serde_json::json!({
            "enabled": self.enabled(),
            "scope": self.scope.name(),
            "entries": inner.entries.len(),
            "bytes": inner.bytes,
            "max_bytes": self.max_bytes,
            "ttl_seconds": self.ttl.as_secs(),
            "hits": self.hits.load(Ordering::Relaxed),
            "misses": self.misses.load(Ordering::Relaxed),
            "stored": self.stored.load(Ordering::Relaxed),
            "evictions": self.evictions.load(Ordering::Relaxed),
            "expirations": self.expirations.load(Ordering::Relaxed),
            "saved_prompt_tokens": self.saved_prompt_tokens.load(Ordering::Relaxed),
        })
    }

    /// The same numbers as Prometheus counters.
    pub fn prometheus(&self) -> String {
        let mut out = String::new();
        let counters = [
            ("fllm_output_cache_hits_total", "Answers served from the output cache", self.hits.load(Ordering::Relaxed)),
            ("fllm_output_cache_misses_total", "Requests the output cache had no answer for", self.misses.load(Ordering::Relaxed)),
            ("fllm_output_cache_stored_total", "Answers written into the output cache", self.stored.load(Ordering::Relaxed)),
            ("fllm_output_cache_evictions_total", "Entries dropped because the cache was full", self.evictions.load(Ordering::Relaxed)),
            ("fllm_output_cache_expirations_total", "Entries dropped because they were too old", self.expirations.load(Ordering::Relaxed)),
            ("fllm_output_cache_saved_prompt_tokens_total", "Prompt tokens that did not have to be computed again", self.saved_prompt_tokens.load(Ordering::Relaxed)),
        ];
        for (name, help, value) in counters {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"));
        }
        out.push_str(&format!(
            "# HELP fllm_output_cache_bytes Answers in the cache, in bytes\n# TYPE fllm_output_cache_bytes gauge\nfllm_output_cache_bytes {}\n",
            self.bytes()
        ));
        out.push_str(&format!(
            "# HELP fllm_output_cache_entries Answers in the cache\n# TYPE fllm_output_cache_entries gauge\nfllm_output_cache_entries {}\n",
            self.len()
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(temperature: &str) -> Fingerprint {
        Fingerprint {
            model: "m".into(),
            template: "t".into(),
            quantization: "Q4_K_M".into(),
            loras: vec![],
            sampling: vec![("temperature".into(), temperature.into()), ("max_tokens".into(), "16".into())],
            grammar: String::new(),
        }
    }

    fn entry(text: &str, prompt_tokens: usize) -> Entry {
        Entry::new(text.to_owned(), vec![1, 2, 3], "stop".to_owned(), prompt_tokens)
    }

    fn cache(max_bytes: usize) -> ResponseCache {
        ResponseCache::new(max_bytes, Duration::from_secs(60), Scope::Global)
    }

    #[test]
    fn a_stored_answer_comes_back_and_is_counted() {
        let cache = cache(1 << 20);
        let key = fp("0").key(&[1, 2, 3], "");
        cache.put(key, entry("hallo", 512));
        let hit = cache.get(&key).expect("hit");
        assert_eq!(hit.text, "hallo");
        assert_eq!(hit.finish_reason, "stop");
        let stats = cache.stats();
        assert_eq!(stats["hits"], 1);
        assert_eq!(stats["saved_prompt_tokens"], 512);
    }

    #[test]
    fn a_different_prompt_or_quantization_is_a_different_key() {
        let base = fp("0");
        assert_ne!(base.key(&[1, 2, 3], ""), base.key(&[1, 2, 4], ""));
        let mut other = base.clone();
        other.quantization = "Q8_0".into();
        assert_ne!(base.key(&[1, 2, 3], ""), other.key(&[1, 2, 3], ""));
        let mut bigger = base.clone();
        bigger.sampling = vec![("temperature".into(), "0".into()), ("max_tokens".into(), "32".into())];
        assert_ne!(base.key(&[1, 2, 3], ""), bigger.key(&[1, 2, 3], ""));
    }

    #[test]
    fn the_tenant_salt_separates_namespaces() {
        let key_a = fp("0").key(&[9], "key-a");
        let key_b = fp("0").key(&[9], "key-b");
        assert_ne!(key_a, key_b);
        assert_eq!(key_a, fp("0").key(&[9], "key-a"));
    }

    #[test]
    fn only_deterministic_requests_are_cacheable() {
        assert!(fp("0").cacheable().is_ok());
        assert!(fp("0.7").cacheable().is_err());
        let mut many = fp("0");
        many.sampling.push(("n".into(), "3".into()));
        assert!(many.cacheable().is_err());
    }

    #[test]
    fn the_byte_budget_evicts_the_least_recently_used() {
        // one entry is about 100 bytes of text plus its overhead: 520 holds two, so the third evicts one
        let cache = ResponseCache::new(520, Duration::from_secs(60), Scope::Global);
        let first = fp("0").key(&[1], "");
        let second = fp("0").key(&[2], "");
        cache.put(first, entry(&"a".repeat(100), 10));
        cache.put(second, entry(&"b".repeat(100), 10));
        assert_eq!(cache.len(), 2);
        cache.get(&first); // first becomes the most recently used
        cache.put(fp("0").key(&[3], ""), entry(&"c".repeat(100), 10));
        assert!(cache.get(&first).is_some(), "the recently used entry survives");
        assert!(cache.get(&second).is_none(), "the least recently used entry was evicted");
    }

    #[test]
    fn an_answer_bigger_than_the_cache_is_not_stored() {
        let cache = ResponseCache::new(200, Duration::from_secs(60), Scope::Global);
        cache.put(fp("0").key(&[1], ""), entry(&"x".repeat(500), 1));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn an_old_answer_is_not_served() {
        let cache = ResponseCache::new(1 << 20, Duration::from_millis(0), Scope::Global);
        let key = fp("0").key(&[1], "");
        cache.put(key, entry("alt", 5));
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats()["expirations"], 1);
    }

    #[test]
    fn off_stores_and_serves_nothing() {
        let cache = ResponseCache::new(1 << 20, Duration::from_secs(60), Scope::Off);
        let key = fp("0").key(&[1], "");
        cache.put(key, entry("x", 1));
        assert!(cache.get(&key).is_none());
        assert!(cache.is_empty());
        assert!(!cache.enabled());
    }

    #[test]
    fn clearing_reports_what_it_dropped() {
        let cache = cache(1 << 20);
        cache.put(fp("0").key(&[1], ""), entry("a", 1));
        cache.put(fp("0").key(&[2], ""), entry("b", 1));
        assert_eq!(cache.clear(), 2);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn prometheus_output_has_the_counters() {
        let cache = cache(1 << 20);
        let key = fp("0").key(&[1], "");
        cache.put(key, entry("a", 7));
        cache.get(&key);
        let text = cache.prometheus();
        assert!(text.contains("fllm_output_cache_hits_total 1"), "{text}");
        assert!(text.contains("fllm_output_cache_entries 1"), "{text}");
    }
}
