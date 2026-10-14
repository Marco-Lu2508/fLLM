//! Optional wall-clock stage tracing for CPU optimization, disabled by default.
use std::time::{Duration, Instant};

pub(crate) struct Trace {
    last: Option<Instant>,
    elapsed: [Duration; 10],
    rows: usize,
}
impl Trace {
    pub(crate) fn new(enabled: bool, rows: usize) -> Self {
        Self {
            last: enabled.then(Instant::now),
            elapsed: [Duration::ZERO; 10],
            rows,
        }
    }
    pub(crate) fn mark(&mut self, stage: usize) {
        if let Some(last) = &mut self.last {
            let now = Instant::now();
            self.elapsed[stage] += now.duration_since(*last);
            *last = now;
        }
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        if self.last.is_none() {
            return;
        }
        let labels = [
            "embedding",
            "norm",
            "qkv",
            "rope_kv",
            "attention",
            "out",
            "gate_up",
            "swiglu",
            "down",
            "head",
        ];
        let phases = labels
            .into_iter()
            .zip(self.elapsed)
            .map(|(label, time)| (label, time.as_secs_f64() * 1000.0))
            .collect::<std::collections::BTreeMap<_, _>>();
        eprintln!(
            "{}",
            serde_json::json!({"rust_cpu_profile":{"rows":self.rows,"phases_ms":phases}})
        );
    }
}
