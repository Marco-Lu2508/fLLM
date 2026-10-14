//! Per-device autotuning: pick the fastest of several candidate configurations (tile sizes, work-group shapes, split factors) by timing
//! them once on the device, and remember the choice in a cache file so the next start does not measure again.
//!
//! This is the scaffold: nothing calls it yet and nothing here has run on a device. A backend that wants a tuned parameter does
//!
//! ```ignore
//! let mut tuner = Autotuner::for_device(&device_fingerprint); // e.g. "vulkan NVIDIA GeForce RTX 4090 driver 555.42"
//! let tile = tuner
//!     .best("linear.q4_k.4096x4096.batch8", &[Tile::new(64, 64), Tile::new(128, 64)], |tile| {
//!         run_linear_with(tile)?; // enqueue, synchronize
//!         Ok(elapsed_seconds)
//!     })
//!     .copied()
//!     .unwrap_or_default();
//! ```
//!
//! # Cache
//!
//! One JSON file per device: `<cache dir>/autotune-<sanitised device id>-<hash>.json`, the cache directory being
//! `$FLLM_AUTOTUNE_DIR`, else `$XDG_CACHE_HOME/fllm`, else `~/.cache/fllm` (`%LOCALAPPDATA%\fllm` on Windows). The file records the
//! device id and a *fingerprint* (the device id plus the engine version); a file with another fingerprint (new driver string, new
//! engine version) is ignored and replaced, because the best tile of one driver says nothing about the next. Entries are keyed by a
//! string the caller composes (kernel, shape class, format) and hold the chosen candidate's label, all measured times and the
//! time of measurement. A candidate is identified by its `Display` text: if the cached label is not among the candidates (the
//! candidate list changed) the entry is measured again.
//!
//! # Policy (`FLLM_AUTOTUNE`)
//!
//! * unset / `auto` (default): use the cache; measure on a miss and store the result.
//! * `off` / `0`: never read, measure or write; `best` returns the first candidate, which is the default configuration.
//! * `retune`: ignore cached entries, measure everything again and overwrite.
//!
//! A cache that cannot be read (missing, corrupt, another version) counts as empty, and one that cannot be written is not an error
//! (`FLLM_AUTOTUNE_VERBOSE=1` prints why).

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Version of the cache file layout.
const FORMAT: u64 = 1;

/// A candidate within this fraction of the best time loses against an earlier candidate: the first entries are the defaults, and noise
/// must not make the tuner flip between configurations that are equally fast.
pub const TIE_MARGIN: f64 = 0.02;

/// How the tuner treats its cache and measurements (`FLLM_AUTOTUNE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Read the cache, measure on a miss, store the result.
    Auto,
    /// No cache, no measurement: the first candidate.
    Off,
    /// Measure again and overwrite the cached entries.
    Retune,
}

impl Mode {
    /// Parse the value of `FLLM_AUTOTUNE` (`None` and unknown values mean [`Mode::Auto`]).
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_lowercase()).as_deref() {
            Some("off" | "0" | "no" | "false") => Mode::Off,
            Some("retune" | "rebuild" | "force") => Mode::Retune,
            _ => Mode::Auto,
        }
    }

    pub fn from_env() -> Self {
        Mode::parse(std::env::var("FLLM_AUTOTUNE").ok().as_deref())
    }
}

/// How often a candidate is run: the first `warmup` runs are discarded, the best of the next `runs` counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rounds {
    pub warmup: usize,
    pub runs: usize,
}

impl Default for Rounds {
    fn default() -> Self {
        Rounds { warmup: 1, runs: 3 }
    }
}

/// The directory of the cache files; `env` is injectable for tests.
pub fn cache_dir_with(env: &dyn Fn(&str) -> Option<OsString>, windows: bool) -> Option<PathBuf> {
    let non_empty = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = non_empty("FLLM_AUTOTUNE_DIR") {
        return Some(dir);
    }
    if windows {
        if let Some(dir) = non_empty("LOCALAPPDATA") {
            return Some(dir.join("fllm"));
        }
    }
    if let Some(dir) = non_empty("XDG_CACHE_HOME") {
        return Some(dir.join("fllm"));
    }
    let home = non_empty("HOME").or_else(|| non_empty("USERPROFILE"))?;
    Some(home.join(".cache").join("fllm"))
}

/// A file name part for a device id: lower-case letters and digits, everything else a single dash, at most 48 characters.
pub fn sanitize(id: &str) -> String {
    let mut out = String::new();
    for c in id.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-');
    out.chars().take(48).collect::<String>().trim_end_matches('-').to_owned()
}

/// `autotune-<sanitised id>-<8 hex of the id's hash>.json`: two devices whose ids differ only in punctuation or beyond the length cut
/// still get different files.
pub fn cache_file_name(device_id: &str) -> String {
    let digest = Sha256::digest(device_id.as_bytes());
    let hash: String = digest.iter().take(4).map(|b| format!("{b:02x}")).collect();
    let name = sanitize(device_id);
    if name.is_empty() {
        format!("autotune-{hash}.json")
    } else {
        format!("autotune-{name}-{hash}.json")
    }
}

/// One tuned problem: the chosen label and what was measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub best: String,
    pub seconds: f64,
    /// Label and best time of every candidate that ran.
    pub timings: BTreeMap<String, f64>,
    /// Unix time of the measurement.
    pub measured: u64,
}

impl Entry {
    fn to_json(&self) -> Value {
        let timings: Map<String, Value> = self
            .timings
            .iter()
            .map(|(label, seconds)| (label.clone(), json!(seconds)))
            .collect();
        json!({ "best": self.best, "seconds": self.seconds, "timings": timings, "measured_unix": self.measured })
    }

    fn from_json(value: &Value) -> Option<Self> {
        let best = value.get("best")?.as_str()?.to_owned();
        let seconds = value.get("seconds")?.as_f64()?;
        let timings = value
            .get("timings")
            .and_then(Value::as_object)
            .map(|map| map.iter().filter_map(|(k, v)| v.as_f64().map(|s| (k.clone(), s))).collect())
            .unwrap_or_default();
        let measured = value.get("measured_unix").and_then(Value::as_u64).unwrap_or(0);
        seconds.is_finite().then_some(Entry {
            best,
            seconds,
            timings,
            measured,
        })
    }
}

/// Fingerprint stored in the cache file: the device id and the engine version.
fn fingerprint(device_id: &str) -> String {
    format!("{device_id} | fllm {}", env!("CARGO_PKG_VERSION"))
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Chooses the fastest candidate per problem and keeps the choices on disk. Not thread-safe by itself: wrap it in a `Mutex` to share it.
#[derive(Debug)]
pub struct Autotuner {
    device_id: String,
    path: Option<PathBuf>,
    mode: Mode,
    rounds: Rounds,
    entries: BTreeMap<String, Entry>,
    verbose: bool,
}

impl Autotuner {
    /// The tuner of one device with the cache in the user's cache directory and the policy of `FLLM_AUTOTUNE`.
    /// `device_id` identifies hardware and driver (`"vulkan NVIDIA GeForce RTX 4090 driver 555.42"`); the cache file is loaded here.
    pub fn for_device(device_id: &str) -> Self {
        let path = cache_dir_with(&|name| std::env::var_os(name), cfg!(windows)).map(|dir| dir.join(cache_file_name(device_id)));
        Self::with_path(device_id, path, Mode::from_env())
    }

    /// A tuner with an explicit cache file (`None`: nothing is stored) and policy; the file is loaded unless `mode` is [`Mode::Off`].
    pub fn with_path(device_id: &str, path: Option<PathBuf>, mode: Mode) -> Self {
        let mut tuner = Autotuner {
            device_id: device_id.to_owned(),
            path,
            mode,
            rounds: Rounds::default(),
            entries: BTreeMap::new(),
            verbose: std::env::var_os("FLLM_AUTOTUNE_VERBOSE").is_some(),
        };
        if mode == Mode::Auto {
            tuner.load();
        }
        tuner
    }

    /// A tuner that never reads, measures or stores anything (always the first candidate).
    pub fn disabled() -> Self {
        Self::with_path("", None, Mode::Off)
    }

    pub fn set_rounds(&mut self, rounds: Rounds) {
        self.rounds = Rounds {
            warmup: rounds.warmup,
            runs: rounds.runs.max(1),
        };
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The cached entry for `key`, if there is one.
    pub fn cached(&self, key: &str) -> Option<&Entry> {
        self.entries.get(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn log(&self, message: &str) {
        if self.verbose {
            eprintln!("[fllm-autotune] {message}");
        }
    }

    /// Replace the entries by those of the cache file, unless it is missing, unreadable or made for another device or engine version.
    fn load(&mut self) {
        self.entries.clear();
        let Some(path) = &self.path else { return };
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                self.log(&format!("{}: {error}", path.display()));
                return;
            }
        };
        let Ok(document) = serde_json::from_str::<Value>(&text) else {
            self.log(&format!("{}: not valid JSON, ignored", path.display()));
            return;
        };
        let same = document.get("format").and_then(Value::as_u64) == Some(FORMAT)
            && document.get("fingerprint").and_then(Value::as_str) == Some(fingerprint(&self.device_id).as_str());
        if !same {
            self.log(&format!(
                "{}: made for another device, driver or engine version, ignored",
                path.display()
            ));
            return;
        }
        if let Some(entries) = document.get("entries").and_then(Value::as_object) {
            self.entries = entries
                .iter()
                .filter_map(|(key, value)| Entry::from_json(value).map(|entry| (key.clone(), entry)))
                .collect();
        }
    }

    /// Write the cache file (atomically: a temporary file next to it, then a rename). Failures are logged, not returned.
    fn save(&self) {
        let Some(path) = &self.path else { return };
        let entries: Map<String, Value> = self.entries.iter().map(|(key, entry)| (key.clone(), entry.to_json())).collect();
        let document = json!({
            "format": FORMAT,
            "device": self.device_id,
            "fingerprint": fingerprint(&self.device_id),
            "entries": entries,
        });
        let Ok(text) = serde_json::to_string_pretty(&document) else {
            return;
        };
        let result = (|| -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let temporary = path.with_extension(format!("json.tmp{}", std::process::id()));
            std::fs::write(&temporary, text + "\n")?;
            std::fs::rename(&temporary, path)
        })();
        if let Err(error) = result {
            self.log(&format!("cannot write {}: {error}", path.display()));
        }
    }

    /// The fastest of `candidates` for the problem `key`.
    ///
    /// `bench` runs a candidate once and returns its time in seconds (lower is better) or the reason it cannot run; candidates that fail
    /// are skipped. It is called `warmup + runs` times per candidate (see [`Rounds`]); the best of the measured runs counts. The
    /// candidate list is ordered by preference: the first is the default and wins ties (see [`TIE_MARGIN`]).
    ///
    /// Returns `None` for an empty list or when every candidate fails (nothing is stored then); the caller keeps its default. With
    /// [`Mode::Off`] the result is simply the first candidate.
    pub fn best<'a, T, F>(&mut self, key: &str, candidates: &'a [T], mut bench: F) -> Option<&'a T>
    where
        T: Display,
        F: FnMut(&T) -> Result<f64, String>,
    {
        if candidates.is_empty() {
            return None;
        }
        if self.mode == Mode::Off || candidates.len() == 1 {
            return candidates.first();
        }
        if self.mode == Mode::Auto {
            if let Some(entry) = self.entries.get(key) {
                if let Some(hit) = candidates.iter().find(|c| c.to_string() == entry.best) {
                    return Some(hit);
                }
                self.log(&format!(
                    "{key}: cached choice '{}' is not among the candidates, measuring again",
                    entry.best
                ));
            }
        }
        let mut timings: Vec<(usize, f64)> = Vec::new();
        for (position, candidate) in candidates.iter().enumerate() {
            match self.measure(candidate, &mut bench) {
                Ok(seconds) => timings.push((position, seconds)),
                Err(reason) => self.log(&format!("{key}: candidate '{candidate}' failed: {reason}")),
            }
        }
        let winner = pick(&timings)?;
        let label = candidates[winner].to_string();
        let entry = Entry {
            best: label,
            seconds: timings.iter().find(|(p, _)| *p == winner).map_or(0.0, |(_, s)| *s),
            timings: timings.iter().map(|(p, s)| (candidates[*p].to_string(), *s)).collect(),
            measured: now_unix(),
        };
        self.log(&format!(
            "{key}: '{}' at {:.3e} s ({} candidates measured)",
            entry.best,
            entry.seconds,
            timings.len()
        ));
        self.entries.insert(key.to_owned(), entry);
        self.save();
        candidates.get(winner)
    }

    /// Best time of `runs` runs after `warmup` discarded ones; a failing or non-finite run fails the candidate.
    fn measure<T>(&self, candidate: &T, bench: &mut impl FnMut(&T) -> Result<f64, String>) -> Result<f64, String> {
        for _ in 0..self.rounds.warmup {
            bench(candidate)?;
        }
        let mut best = f64::INFINITY;
        for _ in 0..self.rounds.runs.max(1) {
            let seconds = bench(candidate)?;
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(format!("measured {seconds}"));
            }
            best = best.min(seconds);
        }
        Ok(best)
    }
}

/// Position (in the candidate list) of the winner among `(position, seconds)`: the earliest candidate whose time is within
/// [`TIE_MARGIN`] of the fastest.
fn pick(timings: &[(usize, f64)]) -> Option<usize> {
    let fastest = timings.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    if !fastest.is_finite() {
        return None;
    }
    timings
        .iter()
        .filter(|(_, s)| *s <= fastest * (1.0 + TIE_MARGIN))
        .map(|(p, _)| *p)
        .min()
}

/// Run `f` (enqueue and synchronize) and return the elapsed seconds: the usual body of a `bench` closure.
pub fn time_seconds(f: impl FnOnce() -> Result<(), String>) -> Result<f64, String> {
    let start = std::time::Instant::now();
    f()?;
    Ok(start.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// A fresh cache file path in the temporary directory, removed on drop.
    struct TempCache(PathBuf);

    impl TempCache {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("fllm-autotune-test-{}-{n}", std::process::id()));
            TempCache(dir.join("autotune-test.json"))
        }
        fn path(&self) -> Option<PathBuf> {
            Some(self.0.clone())
        }
    }

    impl Drop for TempCache {
        fn drop(&mut self) {
            if let Some(dir) = self.0.parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    /// A tuner with a single warm-up and run, and a table of seconds by label.
    fn tuner(cache: &TempCache, mode: Mode) -> Autotuner {
        let mut tuner = Autotuner::with_path("test device", cache.path(), mode);
        tuner.set_rounds(Rounds { warmup: 0, runs: 1 });
        tuner
    }

    fn table<'a>(times: &'a [(&'static str, f64)]) -> impl Fn(&&'static str) -> Result<f64, String> + 'a {
        move |label| {
            times
                .iter()
                .find(|(l, _)| l == label)
                .map(|(_, s)| *s)
                .ok_or_else(|| format!("{label} cannot run"))
        }
    }

    #[test]
    fn picks_the_fastest_and_remembers_it() {
        let cache = TempCache::new();
        let times = [("t32", 3.0), ("t64", 1.0), ("t128", 2.0)];
        let candidates = ["t32", "t64", "t128"];
        let mut first = tuner(&cache, Mode::Auto);
        assert_eq!(first.best("k", &candidates, table(&times)), Some(&"t64"));
        // a new tuner loads the file and does not measure
        let mut second = tuner(&cache, Mode::Auto);
        assert_eq!(second.len(), 1);
        let mut calls = 0;
        let hit = second.best("k", &candidates, |_| {
            calls += 1;
            Ok(1.0)
        });
        assert_eq!((hit, calls), (Some(&"t64"), 0));
        let entry = second.cached("k").unwrap();
        assert_eq!((entry.best.as_str(), entry.seconds), ("t64", 1.0));
        assert_eq!(entry.timings.len(), 3);
    }

    #[test]
    fn warmup_and_runs_are_counted_and_the_best_run_counts() {
        let mut tuner = Autotuner::with_path("d", None, Mode::Auto);
        tuner.set_rounds(Rounds { warmup: 2, runs: 3 });
        let mut calls = 0;
        // run times: the first two (warm-up) are slow, then 5, 2, 4
        let series = [100.0, 100.0, 5.0, 2.0, 4.0];
        let seconds = tuner.measure(&0, &mut |_: &i32| {
            calls += 1;
            Ok(series[calls - 1])
        });
        assert_eq!((seconds, calls), (Ok(2.0), 5));
    }

    #[test]
    fn a_changed_candidate_list_is_measured_again() {
        let cache = TempCache::new();
        let times = [("a", 2.0), ("b", 1.0), ("c", 0.5)];
        let mut first = tuner(&cache, Mode::Auto);
        assert_eq!(first.best("k", &["a", "b"], table(&times)), Some(&"b"));
        let mut second = tuner(&cache, Mode::Auto);
        // 'b' is gone: the cached choice is not a candidate any more
        assert_eq!(second.best("k", &["a", "c"], table(&times)), Some(&"c"));
        assert_eq!(second.cached("k").unwrap().best, "c");
    }

    #[test]
    fn keys_are_independent() {
        let cache = TempCache::new();
        let mut t = tuner(&cache, Mode::Auto);
        assert_eq!(t.best("small", &["a", "b"], table(&[("a", 1.0), ("b", 2.0)])), Some(&"a"));
        assert_eq!(t.best("large", &["a", "b"], table(&[("a", 2.0), ("b", 1.0)])), Some(&"b"));
        assert_eq!(t.cached("small").unwrap().best, "a");
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn a_tie_goes_to_the_earlier_candidate() {
        // the second is 1 % faster: inside the margin, the default (first) stays
        assert_eq!(pick(&[(0, 1.00), (1, 0.99)]), Some(0));
        // 10 % faster: the second wins
        assert_eq!(pick(&[(0, 1.00), (1, 0.90)]), Some(1));
        // failed candidates leave gaps in the positions
        assert_eq!(pick(&[(2, 1.0), (4, 0.5)]), Some(4));
        assert_eq!(pick(&[]), None);
        assert_eq!(pick(&[(0, f64::NAN)]), None);
    }

    #[test]
    fn failing_candidates_are_skipped_and_all_failing_stores_nothing() {
        let cache = TempCache::new();
        let mut t = tuner(&cache, Mode::Auto);
        // only "ok" can run
        assert_eq!(t.best("k", &["broken", "ok"], table(&[("ok", 1.0)])), Some(&"ok"));
        let mut none = tuner(&cache, Mode::Auto);
        assert_eq!(none.best("other", &["x", "y"], table(&[])), None);
        assert!(none.cached("other").is_none());
        // a non-finite measurement fails the candidate too
        assert_eq!(
            none.best("nan", &["x", "y"], |c: &&str| Ok(if *c == "x" { f64::NAN } else { 2.0 })),
            Some(&"y")
        );
    }

    #[test]
    fn off_returns_the_first_candidate_and_touches_nothing() {
        let cache = TempCache::new();
        let mut t = tuner(&cache, Mode::Off);
        let mut calls = 0;
        let choice = t.best("k", &["first", "second"], |_| {
            calls += 1;
            Ok(1.0)
        });
        assert_eq!((choice, calls), (Some(&"first"), 0));
        assert!(!cache.0.exists());
        assert_eq!(Autotuner::disabled().best("k", &[7, 8], |_| Ok(0.0)), Some(&7));
    }

    #[test]
    fn retune_ignores_the_cache() {
        let cache = TempCache::new();
        let mut first = tuner(&cache, Mode::Auto);
        first.best("k", &["a", "b"], table(&[("a", 1.0), ("b", 2.0)]));
        let mut again = tuner(&cache, Mode::Retune);
        assert!(again.is_empty());
        // the hardware changed its mind: "b" is the fast one now
        assert_eq!(again.best("k", &["a", "b"], table(&[("a", 2.0), ("b", 1.0)])), Some(&"b"));
        let reloaded = tuner(&cache, Mode::Auto);
        assert_eq!(reloaded.cached("k").unwrap().best, "b");
    }

    #[test]
    fn a_single_candidate_needs_no_measurement() {
        let mut t = Autotuner::with_path("d", None, Mode::Auto);
        let choice = t.best("k", &["only"], |_| panic!("must not run"));
        assert_eq!(choice, Some(&"only"));
        assert_eq!(t.best::<&str, _>("k", &[], |_| Ok(0.0)), None);
    }

    #[test]
    fn a_cache_of_another_device_or_version_is_ignored() {
        let cache = TempCache::new();
        let mut owner = Autotuner::with_path("device A", cache.path(), Mode::Auto);
        owner.set_rounds(Rounds { warmup: 0, runs: 1 });
        owner.best("k", &["a", "b"], table(&[("a", 1.0), ("b", 2.0)]));
        assert_eq!(Autotuner::with_path("device A", cache.path(), Mode::Auto).len(), 1);
        assert_eq!(Autotuner::with_path("device B", cache.path(), Mode::Auto).len(), 0);
        // another engine version
        let text = std::fs::read_to_string(&cache.0)
            .unwrap()
            .replace(&format!("fllm {}", env!("CARGO_PKG_VERSION")), "fllm 0.0.0-other");
        std::fs::write(&cache.0, text).unwrap();
        assert_eq!(Autotuner::with_path("device A", cache.path(), Mode::Auto).len(), 0);
    }

    #[test]
    fn damaged_files_count_as_empty() {
        let cache = TempCache::new();
        std::fs::create_dir_all(cache.0.parent().unwrap()).unwrap();
        for content in ["", "not json", "[1,2,3]", "{\"format\":1}", "{\"format\":99,\"entries\":{}}"] {
            std::fs::write(&cache.0, content).unwrap();
            assert_eq!(Autotuner::with_path("d", cache.path(), Mode::Auto).len(), 0, "{content}");
        }
        // a good header with a broken entry keeps the good ones
        let fp = fingerprint("d");
        let document = json!({
            "format": FORMAT, "device": "d", "fingerprint": fp,
            "entries": { "bad": { "best": 3 }, "good": { "best": "x", "seconds": 0.5, "timings": { "x": 0.5 }, "measured_unix": 9 } }
        });
        std::fs::write(&cache.0, document.to_string()).unwrap();
        let loaded = Autotuner::with_path("d", cache.path(), Mode::Auto);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.cached("good").unwrap().measured, 9);
    }

    #[test]
    fn an_unwritable_cache_is_not_an_error() {
        // the parent "directory" is a file: create_dir_all and write fail
        let cache = TempCache::new();
        let dir = cache.0.parent().unwrap();
        std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
        std::fs::write(dir, "a file").unwrap();
        let mut t = tuner(&cache, Mode::Auto);
        assert_eq!(t.best("k", &["a", "b"], table(&[("a", 2.0), ("b", 1.0)])), Some(&"b"));
        assert_eq!(t.cached("k").unwrap().best, "b");
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn file_names_and_directories() {
        assert_eq!(
            sanitize("Vulkan NVIDIA GeForce RTX 4090 (driver 555.42)"),
            "vulkan-nvidia-geforce-rtx-4090-driver-555-42"
        );
        assert_eq!(sanitize("  ***  "), "");
        assert_eq!(sanitize("a/b\\c:d"), "a-b-c-d");
        assert!(sanitize(&"x".repeat(100)).len() <= 48);
        let a = cache_file_name("gpu (1)");
        let b = cache_file_name("gpu [1]");
        assert!(a.starts_with("autotune-gpu-1-") && a.ends_with(".json"), "{a}");
        assert_ne!(a, b, "ids that differ in punctuation need different files");
        assert!(cache_file_name("").starts_with("autotune-") && cache_file_name("").ends_with(".json"));
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| OsString::from(v))
        };
        assert_eq!(
            cache_dir_with(&env(&[("FLLM_AUTOTUNE_DIR", "/x"), ("XDG_CACHE_HOME", "/y")]), false),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            cache_dir_with(&env(&[("XDG_CACHE_HOME", "/y"), ("HOME", "/h")]), false),
            Some(PathBuf::from("/y").join("fllm"))
        );
        assert_eq!(
            cache_dir_with(&env(&[("HOME", "/h")]), false),
            Some(PathBuf::from("/h").join(".cache").join("fllm"))
        );
        assert_eq!(
            cache_dir_with(&env(&[("XDG_CACHE_HOME", ""), ("HOME", "/h")]), false),
            Some(PathBuf::from("/h").join(".cache").join("fllm"))
        );
        assert_eq!(
            cache_dir_with(&env(&[("LOCALAPPDATA", "C:\\L"), ("USERPROFILE", "C:\\U")]), true),
            Some(PathBuf::from("C:\\L").join("fllm"))
        );
        assert_eq!(cache_dir_with(&env(&[]), false), None);
    }

    #[test]
    fn the_mode_comes_from_the_environment_value() {
        assert_eq!(Mode::parse(None), Mode::Auto);
        assert_eq!(Mode::parse(Some("auto")), Mode::Auto);
        assert_eq!(Mode::parse(Some("OFF")), Mode::Off);
        assert_eq!(Mode::parse(Some("0")), Mode::Off);
        assert_eq!(Mode::parse(Some("Retune")), Mode::Retune);
        assert_eq!(Mode::parse(Some("whatever")), Mode::Auto);
    }

    #[test]
    fn time_seconds_measures_and_propagates_errors() {
        assert!(time_seconds(|| Ok(())).unwrap() >= 0.0);
        assert_eq!(time_seconds(|| Err("boom".into())), Err("boom".to_owned()));
    }
}
