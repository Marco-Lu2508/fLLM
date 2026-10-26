//! `--threads auto`: the thread count of the CPU workers, chosen from the topology and confirmed by a short measurement.
//!
//! The policy value ([`crate::cpu_topology::Topology::default_threads`]) is right for most machines, but the best count of a hybrid
//! processor also depends on memory speed, power limits and the other work on the machine. At start-up (unless the model is tiny
//! or `FLLM_NO_AUTOTUNE=1`) this module times a decode-like step on the real weights, a matrix-vector product of the largest
//! projection of the first layer, for three or four counts around the policy value, takes the smallest count that is within 3 %
//! of the fastest, and caches the answer per CPU model, core counts, weight type and matrix shape in `cpu-threads.json` next to
//! the Studio settings, so later starts do not measure. The whole measurement stays under about 1.5 s.
use crate::compute::Linear;
use crate::cpu_topology::{self as topology, Topology};
use fllm_gguf_probe::MappedGguf;
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Smallest count whose time is within this factor of the best is chosen (fewer threads leave cores to the rest of the system).
const TOLERANCE: f64 = 1.03;
/// Every candidate is timed this many times, interleaved, and keeps its best time (drift and turbo ramp-up hit all alike).
const ROUNDS: usize = 2;
/// Time given to all measurements together.
const BUDGET: Duration = Duration::from_millis(1200);
/// Hard stop (cold weights, a busy machine): the policy value is used and nothing is cached.
const DEADLINE: Duration = Duration::from_millis(2500);
/// Projections smaller than this are served from cache and say nothing about memory bandwidth: no measurement.
const MIN_WEIGHT_BYTES: u64 = 8 << 20;

/// What the process runs with, for the logs, `fllm-rs devices` and the Studio's hardware page.
#[derive(Clone, Debug)]
pub struct ThreadChoice {
    pub threads: usize,
    /// Workers for batches of many rows (prompt processing); at least `threads`.
    pub prefill: usize,
    /// The policy value the choice started from.
    pub policy: usize,
    /// `flag` (`--threads N`), `policy`, `cache` or `autotune`.
    pub source: &'static str,
    pub detail: String,
}

static CHOICE: Mutex<Option<ThreadChoice>> = Mutex::new(None);

/// Records the choice of this process.
pub fn record(choice: ThreadChoice) {
    *CHOICE.lock().unwrap_or_else(|e| e.into_inner()) = Some(choice);
}

pub fn current() -> Option<ThreadChoice> {
    CHOICE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The thread count in use, or the one this machine would use by default.
pub fn active_threads() -> usize {
    current().map_or_else(|| topology::detect().default_threads(), |c| c.threads)
}

/// `threads`, `source` and the topology as JSON (the hardware page).
pub fn to_json() -> Value {
    let topology = topology::detect();
    let choice = current();
    json!({
        "threads": choice.as_ref().map_or_else(|| topology.default_threads(), |c| c.threads),
        "prefill_threads": choice.as_ref().map_or_else(|| topology.prefill_threads(), |c| c.prefill),
        "available_threads": topology.logical_count(),
        "source": choice.as_ref().map_or("policy", |c| c.source),
        "detail": choice.map(|c| c.detail),
        "cpu_topology": topology.to_json(),
    })
}

/// One line for `fllm-rs devices` and the start-up log.
pub fn summary(threads: usize, topology: &Topology) -> String {
    format!("{threads} of {} threads ({})", topology.logical_count(), topology.describe())
}

// ---------------------------------------------------------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub struct Decision {
    pub threads: usize,
    /// `(threads, best seconds per product)` in candidate order.
    pub timings: Vec<(usize, f64)>,
}

/// Times every candidate `rounds` times (round-robin, so slow drift hits all alike) with `measure(threads) -> seconds` (`None`
/// aborts: no decision) and returns the smallest count whose best time is within [`TOLERANCE`] of the fastest.
pub fn decide(candidates: &[usize], rounds: usize, measure: &mut dyn FnMut(usize) -> Option<f64>) -> Option<Decision> {
    let mut best = vec![f64::INFINITY; candidates.len()];
    for _ in 0..rounds.max(1) {
        for (slot, &threads) in best.iter_mut().zip(candidates) {
            let seconds = measure(threads)?;
            if !seconds.is_finite() || seconds <= 0.0 {
                return None;
            }
            *slot = slot.min(seconds);
        }
    }
    let fastest = best.iter().copied().fold(f64::INFINITY, f64::min);
    let threads = candidates
        .iter()
        .zip(&best)
        .filter(|(_, &t)| t <= fastest * TOLERANCE)
        .map(|(&n, _)| n)
        .min()?;
    Some(Decision { threads, timings: candidates.iter().copied().zip(best).collect() })
}

// ---------------------------------------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------------------------------------

const CACHE_FILE: &str = "cpu-threads.json";
const CACHE_LIMIT: usize = 64;

/// The cached thread count for `key` in the text of the cache file.
pub fn cache_lookup(text: &str, key: &str) -> Option<usize> {
    let value: Value = serde_json::from_str(text).ok()?;
    let n = value.get("entries")?.get(key)?.get("threads")?.as_u64()?;
    (1..=4096).contains(&n).then_some(n as usize)
}

/// The text of the cache file with `key` set to `threads` (a damaged file starts over; the file holds at most 64 entries).
pub fn cache_store(text: &str, key: &str, threads: usize) -> String {
    let mut entries = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| v.get("entries").and_then(Value::as_object).cloned())
        .unwrap_or_default();
    entries.remove(key);
    while entries.len() >= CACHE_LIMIT {
        let Some(first) = entries.keys().next().cloned() else { break };
        entries.remove(&first);
    }
    entries.insert(key.to_owned(), json!({"threads": threads}));
    json!({"version": 1, "entries": entries}).to_string()
}

/// Identifies the machine and the work: CPU model, core counts per class, weight type and matrix shape.
pub fn cache_key(cpu: &str, topology: &Topology, type_id: u32, rows: usize, cols: usize) -> String {
    let classes: Vec<String> = topology.classes().iter().map(|c| format!("{}/{}", c.cores, c.threads)).collect();
    format!("{}|{}|type{type_id}|{rows}x{cols}", cpu.trim(), classes.join("+"))
}

fn cache_path() -> std::path::PathBuf {
    crate::hub::studio_data_file(CACHE_FILE)
}

// ---------------------------------------------------------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------------------------------------------------------

/// The projection that stands for a decode step: the largest 2-D quantized weight of the first layer, as
/// `(name without ".weight", rows, cols, type id, bytes)`.
fn representative(mapping: &MappedGguf) -> Option<(String, usize, usize, u32, u64)> {
    let layer_of = |name: &str| -> Option<usize> { name.strip_prefix("blk.")?.split_once('.')?.0.parse().ok() };
    let eligible = |t: &&fllm_gguf_probe::TensorInfo| {
        t.dimensions.len() == 2 && t.name.ends_with(".weight") && crate::compute::quant_layout(t.type_id).is_some()
    };
    let tensors = &mapping.index().tensors;
    let first = tensors.iter().filter(eligible).filter_map(|t| layer_of(&t.name)).min()?;
    let t = tensors
        .iter()
        .filter(eligible)
        .filter(|t| layer_of(&t.name) == Some(first))
        .max_by_key(|t| t.dimensions[0] * t.dimensions[1])?;
    let base = t.name.strip_suffix(".weight")?.to_owned();
    Some((base, t.dimensions[1] as usize, t.dimensions[0] as usize, t.type_id, t.payload_len))
}

fn time_candidate(pool: &rayon::ThreadPool, linear: &Linear, input: &[f32], slice: Duration) -> Option<f64> {
    pool.install(|| {
        for _ in 0..2 {
            linear.forward(input).ok()?;
        }
        let start = Instant::now();
        let mut best = f64::INFINITY;
        loop {
            let t = Instant::now();
            for _ in 0..4 {
                linear.forward(input).ok()?;
            }
            best = best.min(t.elapsed().as_secs_f64() / 4.0);
            if start.elapsed() >= slice {
                return Some(best);
            }
        }
    })
}

/// Chooses the thread count of the CPU workers for the model in `mapping`. `hidden` is the model's hidden size: a pool is never
/// larger than the output rows can feed. Also lets the operating system keep the workers on the chosen cores (Windows) and out of
/// power throttling when the processor is hybrid.
pub fn tune(mapping: &MappedGguf, hidden: usize) -> ThreadChoice {
    let topology = topology::detect();
    let policy = topology::limit_for_rows(topology.default_threads(), hidden);
    let hybrid = topology.is_hybrid();
    if hybrid {
        topology::disable_power_throttling();
    }
    let settle = |threads: usize, source: &'static str, detail: String| {
        if hybrid {
            topology::pin_first(topology, threads);
        }
        ThreadChoice { threads, prefill: threads.max(topology.prefill_threads()), policy, source, detail }
    };
    if std::env::var_os("FLLM_NO_AUTOTUNE").is_some_and(|v| !v.is_empty() && v != "0") {
        return settle(policy, "policy", "auto-tuning disabled by FLLM_NO_AUTOTUNE".into());
    }
    let Some((base, rows, cols, type_id, bytes)) = representative(mapping) else {
        return settle(policy, "policy", "no projection to measure".into());
    };
    if bytes < MIN_WEIGHT_BYTES {
        return settle(policy, "policy", "model too small to measure".into());
    }
    let cpu = crate::devices::cpu_name();
    let key = cache_key(&cpu, topology, type_id, rows, cols);
    if let Some(n) = std::fs::read_to_string(cache_path()).ok().and_then(|text| cache_lookup(&text, &key)) {
        if n <= topology.logical_count() {
            return settle(n, "cache", format!("remembered from an earlier start (policy {policy})"));
        }
    }
    let candidates: Vec<usize> = topology::candidates(policy, topology.logical_count())
        .into_iter()
        .filter(|&n| topology::limit_for_rows(n, hidden) == n)
        .collect();
    if candidates.len() < 2 {
        return settle(policy, "policy", "nothing to choose between".into());
    }
    let started = Instant::now();
    let measured = (|| {
        let mut linear = Linear::load(mapping, &base, rows, cols).ok()?;
        linear.enable_fast_cpu();
        // fixed pseudo-random activations; the values do not matter for the time
        let mut state = 0x9E37_79B9u32;
        let input: Vec<f32> = (0..cols)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();
        let slice = BUDGET / (candidates.len() * ROUNDS) as u32;
        let mut pools = Vec::new();
        for &n in &candidates {
            if hybrid {
                topology::pin_first(topology, n);
            }
            pools.push((n, rayon::ThreadPoolBuilder::new().num_threads(n).build().ok()?));
        }
        decide(&candidates, ROUNDS, &mut |n| {
            if started.elapsed() > DEADLINE {
                return None;
            }
            let (_, pool) = pools.iter().find(|(m, _)| *m == n)?;
            time_candidate(pool, &linear, &input, slice)
        })
    })();
    match measured {
        Some(decision) => {
            let text = std::fs::read_to_string(cache_path()).unwrap_or_default();
            let path = cache_path();
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(path, cache_store(&text, &key, decision.threads));
            let timings: Vec<String> = decision.timings.iter().map(|(n, s)| format!("{n}: {:.2} ms", s * 1e3)).collect();
            settle(
                decision.threads,
                "autotune",
                format!("measured {} in {:.1} s; policy {policy}; {}", timings.join(", "), started.elapsed().as_secs_f64(), base),
            )
        }
        None => settle(policy, "policy", "measurement did not finish".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_prefers_the_fewest_threads_within_tolerance() {
        // the shape measured on a hybrid laptop: a clear optimum at 8, much slower beyond it
        let times = |n: usize| match n {
            6 => 1.0 / 17.7,
            8 => 1.0 / 19.6,
            10 => 1.0 / 14.5,
            12 => 1.0 / 15.0,
            _ => 1.0,
        };
        let d = decide(&[6, 8, 10, 12], 2, &mut |n| Some(times(n))).unwrap();
        assert_eq!(d.threads, 8);
        assert_eq!(d.timings.len(), 4);
        // a flat curve resolves to the smallest count, a monotone one to the largest
        let flat = decide(&[4, 5, 6], 1, &mut |n| Some(1.0 + n as f64 * 0.001)).unwrap();
        assert_eq!(flat.threads, 4);
        let scaling = decide(&[4, 8, 16], 1, &mut |n| Some(1.0 / n as f64)).unwrap();
        assert_eq!(scaling.threads, 16);
    }

    #[test]
    fn decision_keeps_the_best_of_all_rounds_and_aborts() {
        let mut calls = 0;
        // 8 is slow in round one only (a hiccup): the best time of the rounds wins
        let d = decide(&[6, 8], 2, &mut |n| {
            calls += 1;
            Some(if n == 8 && calls == 2 { 5.0 } else if n == 8 { 0.5 } else { 1.0 })
        })
        .unwrap();
        assert_eq!((d.threads, calls), (8, 4));
        assert!(decide(&[6, 8], 2, &mut |_| None).is_none());
        assert!(decide(&[6, 8], 1, &mut |_| Some(f64::NAN)).is_none());
    }

    #[test]
    fn cache_round_trip() {
        let t = Topology::uniform(4, 8, "x");
        let key = cache_key("Some CPU @ 3GHz ", &t, 12, 4096, 14336);
        assert_eq!(key, "Some CPU @ 3GHz|4/8|type12|4096x14336");
        assert_eq!(cache_lookup("", &key), None);
        let text = cache_store("", &key, 6);
        assert_eq!(cache_lookup(&text, &key), Some(6));
        let other = cache_key("Other", &t, 12, 4096, 14336);
        let text = cache_store(&text, &other, 3);
        assert_eq!((cache_lookup(&text, &key), cache_lookup(&text, &other)), (Some(6), Some(3)));
        // overwrite, damaged files and absurd values
        assert_eq!(cache_lookup(&cache_store(&text, &key, 5), &key), Some(5));
        assert_eq!(cache_lookup("{not json", &key), None);
        assert_eq!(cache_lookup(&cache_store("{not json", &key, 2), &key), Some(2));
        assert_eq!(cache_lookup(&cache_store("", &key, 0), &key), None);
    }

    #[test]
    fn cache_stays_bounded() {
        let mut text = String::new();
        for i in 0..(CACHE_LIMIT + 10) {
            text = cache_store(&text, &format!("k{i}"), 4);
        }
        let entries = serde_json::from_str::<Value>(&text).unwrap()["entries"].as_object().unwrap().len();
        assert_eq!(entries, CACHE_LIMIT);
        assert_eq!(cache_lookup(&text, &format!("k{}", CACHE_LIMIT + 9)), Some(4));
    }

    #[test]
    fn summary_text() {
        let t = Topology::from_levels(&[(8, 8), (4, 4)]);
        assert_eq!(summary(11, &t), "11 of 12 threads (hybrid processor: 8 performance cores, 4 efficiency cores)");
    }
}
