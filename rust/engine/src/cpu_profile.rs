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

/// Per-operation wall-clock totals of one forward pass (`FLLM_CPU_OPS=1`): the time spent in each kind of work, printed as one JSON line on
/// stderr at the end of the pass. Off (one relaxed load per span) unless the variable is set. It shows where a decode or verification step of a
/// model goes on the host; only spans on the calling thread of the pass are timed (parallel work inside a span counts as wall time).
pub(crate) mod ops {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub(crate) const EMBED: usize = 0;
    pub(crate) const PLE: usize = 1;
    pub(crate) const MIX: usize = 2;
    pub(crate) const DELTA: usize = 3;
    pub(crate) const ATTENTION: usize = 4;
    pub(crate) const INJECT: usize = 5;
    pub(crate) const ROUTER: usize = 6;
    pub(crate) const EXPERTS: usize = 7;
    pub(crate) const SHARED: usize = 8;
    pub(crate) const HEAD: usize = 9;
    const COUNT: usize = 10;
    const LABELS: [&str; COUNT] = ["embed", "ple", "hc_mix", "delta", "attention", "inject", "router", "experts", "shared", "lm_head"];
    static TOTALS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];

    pub(crate) fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("FLLM_CPU_OPS").is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "false")))
    }

    /// Adds the time until it is dropped to `stage`.
    pub(crate) struct Span(Option<(usize, Instant)>);

    pub(crate) fn span(stage: usize) -> Span {
        Span(enabled().then(|| (stage, Instant::now())))
    }

    impl Drop for Span {
        fn drop(&mut self) {
            if let Some((stage, start)) = self.0 {
                TOTALS[stage].fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
            }
        }
    }

    /// Prints the totals of a pass of this many rows when it ends.
    pub(crate) struct ReportOnDrop(pub(crate) usize);

    impl Drop for ReportOnDrop {
        fn drop(&mut self) {
            report(self.0);
        }
    }

    /// Prints and clears the totals of a pass of `rows` rows.
    fn report(rows: usize) {
        if !enabled() {
            return;
        }
        let ms: std::collections::BTreeMap<&str, f64> = LABELS.iter().zip(&TOTALS).map(|(l, t)| (*l, t.swap(0, Relaxed) as f64 / 1e6)).collect();
        eprintln!("{}", serde_json::json!({"cpu_ops": {"rows": rows, "ms": ms}}));
    }
}
