//! Expert streaming for mixture-of-experts models that do not fit in memory (see docs/moe-streaming.md).
//!
//! The weights of a model file are memory-mapped, so a model of any size starts at once and an expert is read from the file when a token
//! first routes to it. What this module adds on top of that is policy and measurement:
//!
//! * a **budget** for the host memory that expert weights may occupy (`--expert-cache-mb`, by default derived from the free memory and the
//!   cgroup limit): experts beyond it are dropped from memory, least recently used first, and the hottest experts (by a decayed
//!   per-layer activation count) are pinned and never dropped;
//! * **read-ahead**: once the router of a layer has chosen its experts, all of them are requested from the file at once (a deep queue
//!   instead of one page fault at a time), and a background thread requests the experts the *next* layer is expected to use, predicted
//!   from the experts that layer used for the previous token;
//! * **statistics**: how many routed experts were already in memory, how many bytes had to be read, how good the prediction was.
//!
//! The expert bytes themselves are never copied or changed: a [`LayerTable`] only knows the address ranges of the experts inside the
//! mapping and gives the operating system hints (`os.rs`), so the output of the model does not depend on the budget. A batch of tokens
//! routes to the union of the experts of its tokens, which is what `on_route` receives, so many conversations in one batch work like one.

mod os;

use fllm_gguf_probe::TensorBytes;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};

pub use os::disk_read_bytes;

/// More rows than this in one routing call are a prompt batch, not decoding: they warm the statistics only a little and are not
/// used to predict the next layer.
const BULK_ROWS: usize = 24;
/// Fraction of the budget that the pinned (hottest) experts may take; the rest is the window of recently used experts.
const PIN_FRACTION: f64 = 0.8;
/// Statistics are rebalanced (hot set recomputed, counts decayed) every this many forward passes.
const REBALANCE_TICKS: u64 = 32;
/// An expert counts as resident when at least this share of the sampled pages is in memory.
const RESIDENT_SHARE: f64 = 0.9;
/// Smallest automatic budget.
const MIN_AUTO_BUDGET: u64 = 128 << 20;

/// How much host memory expert weights may take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    /// From the free memory of the machine and the limit of its cgroup.
    Auto,
    /// This many bytes.
    Bytes(u64),
    /// No cache layer at all: the experts are plain memory-mapped weights (what earlier versions did).
    Off,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub budget: Budget,
    /// Read ahead the predicted experts of the next layer in the background. Off by default: on the drive it was measured on only about 30 %
    /// of the experts it requested were then used, and it did not make decoding faster.
    pub prefetch: bool,
    /// Request the experts of the current layer from the file as soon as the router has chosen them, instead of faulting them in page
    /// by page while they are computed. Off by default: no measurable gain on the NVMe drive it was tried on.
    pub demand: bool,
    /// Drop experts from memory to stay within an explicit budget, and pin the hottest ones. An automatic budget is never enforced this
    /// way: the operating system already drops the least recently used pages when memory runs short, and doing it earlier measured slower.
    pub evict: bool,
    /// Turn the operating system's own read-ahead off for expert weights when they do not fit in memory. Off by default: measured on an NVMe
    /// drive it halves the speed, because the faults of the compute threads then read 4 KiB at a time.
    pub random_access: bool,
    /// Run the background work on the calling thread (tests).
    pub inline: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { budget: Budget::Auto, prefetch: false, demand: false, evict: true, random_access: false, inline: false }
    }
}

static OPTIONS: Mutex<Option<Options>> = Mutex::new(None);
thread_local! {
    static OVERRIDE: std::cell::RefCell<Option<Options>> = const { std::cell::RefCell::new(None) };
}

/// Parse `auto`, `off`, a number of MiB, or a number with a suffix (`512M`, `8G`, `8GiB`, `1.5G`).
pub fn parse_budget(text: &str) -> Result<Budget, String> {
    let t = text.trim().to_ascii_lowercase();
    match t.as_str() {
        "auto" | "" => return Ok(Budget::Auto),
        "off" | "none" | "disabled" => return Ok(Budget::Off),
        _ => {}
    }
    let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let (number, unit) = t.split_at(split);
    let n: f64 = number.parse().map_err(|_| format!("invalid expert cache size '{text}' (use auto, off, or a size such as 4096, 512M, 8G)"))?;
    let mib = match unit.trim() {
        "" | "m" | "mb" | "mib" => 1.0,
        "g" | "gb" | "gib" => 1024.0,
        "k" | "kb" | "kib" => 1.0 / 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0,
        _ => return Err(format!("invalid expert cache size '{text}' (use auto, off, or a size such as 4096, 512M, 8G)")),
    };
    if !n.is_finite() || n < 0.0 {
        return Err(format!("invalid expert cache size '{text}'"));
    }
    Ok(Budget::Bytes((n * mib * 1048576.0) as u64))
}

/// Set the options the models loaded from now on use (`--expert-cache-mb`).
pub fn configure(options: Options) {
    *OPTIONS.lock().unwrap_or_else(|e| e.into_inner()) = Some(options);
}

/// Run `f` with other options for the models loaded on this thread (tests).
pub fn with_options<R>(options: Options, f: impl FnOnce() -> R) -> R {
    let previous = OVERRIDE.with(|o| o.replace(Some(options)));
    let result = f();
    OVERRIDE.with(|o| *o.borrow_mut() = previous);
    result
}

/// The options in force: the thread override, else `configure`, else the environment (`FLLM_EXPERT_CACHE_MB`; `FLLM_EXPERT_PREFETCH=0`,
/// `FLLM_EXPERT_EVICT=0` switches eviction off, `FLLM_EXPERT_DEMAND=1`, `FLLM_EXPERT_PREFETCH=1` and `FLLM_EXPERT_RANDOM_ACCESS=1` switch read-ahead on), else the defaults.
pub fn options() -> Options {
    if let Some(o) = OVERRIDE.with(|o| o.borrow().clone()) {
        return o;
    }
    if let Some(o) = OPTIONS.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        return o;
    }
    let mut o = Options::default();
    if let Some(Ok(b)) = std::env::var("FLLM_EXPERT_CACHE_MB").ok().map(|v| parse_budget(&v)) {
        o.budget = b;
    }
    let off = |name: &str| std::env::var(name).is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false"));
    o.prefetch = std::env::var("FLLM_EXPERT_PREFETCH").is_ok_and(|v| matches!(v.trim(), "1" | "on" | "true"));
    o.demand = std::env::var("FLLM_EXPERT_DEMAND").is_ok_and(|v| matches!(v.trim(), "1" | "on" | "true"));
    o.evict = !off("FLLM_EXPERT_EVICT");
    o.random_access = std::env::var("FLLM_EXPERT_RANDOM_ACCESS").is_ok_and(|v| matches!(v.trim(), "1" | "on" | "true"));
    o
}

/// `FLLM_EXPERT_TRACE=FILE`: one line `tick layer rows e1,e2,...` for every routing event (to replay policies offline).
fn trace(layer: usize, tick: u64, rows: usize, active: &[usize]) {
    use std::io::Write;
    static FILE: OnceLock<Option<Mutex<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();
    let file = FILE.get_or_init(|| std::env::var_os("FLLM_EXPERT_TRACE").and_then(|p| std::fs::File::create(p).ok()).map(|f| Mutex::new(std::io::BufWriter::new(f))));
    if let Some(f) = file {
        let mut f = f.lock().unwrap_or_else(|e| e.into_inner());
        let list: Vec<String> = active.iter().map(usize::to_string).collect();
        let _ = writeln!(f, "{tick} {layer} {rows} {}", list.join(","));
        let _ = f.flush();
    }
}

/// Ask the operating system to read a range of a mapped file ahead (the dense weights of a model that does not fit in memory).
pub fn will_need(bytes: &[u8]) {
    if !bytes.is_empty() {
        os::will_need(bytes.as_ptr() as usize, bytes.len());
    }
}

/// Memory the machine can give to cached file data: the available physical memory, and under a cgroup memory limit (containers,
/// `systemd-run -p MemoryMax=`) what that limit leaves besides the anonymous memory in use.
pub fn effective_available_bytes() -> u64 {
    let host = crate::devices::host_free_memory_bytes();
    match cgroup_headroom() {
        Some(h) if host > 0 => host.min(h),
        Some(h) => h,
        None => host,
    }
}

#[cfg(target_os = "linux")]
fn cgroup_headroom() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = own.lines().find_map(|l| l.strip_prefix("0::"))?.trim().to_owned();
    let mut dir = std::path::PathBuf::from("/sys/fs/cgroup");
    dir.push(path.trim_start_matches('/'));
    let mut best: Option<u64> = None;
    let mut level = Some(dir.as_path());
    while let Some(d) = level {
        if !d.starts_with("/sys/fs/cgroup") {
            break;
        }
        if let Some(limit) = std::fs::read_to_string(d.join("memory.max")).ok().and_then(|t| t.trim().parse::<u64>().ok()) {
            let stat = std::fs::read_to_string(d.join("memory.stat")).unwrap_or_default();
            let get = |key: &str| stat.lines().find_map(|l| l.strip_prefix(key).and_then(|r| r.trim().parse::<u64>().ok())).unwrap_or(0);
            let pinned = get("anon ") + get("kernel_stack ") + get("slab ") + get("pagetables ") + get("sock ");
            let room = limit.saturating_sub(pinned);
            best = Some(best.map_or(room, |b| b.min(room)));
        }
        level = d.parent();
    }
    best
}

#[cfg(not(target_os = "linux"))]
fn cgroup_headroom() -> Option<u64> {
    None
}

#[derive(Default)]
struct Stats {
    /// Forward passes seen by the first expert layer, and the rows they carried.
    passes: AtomicU64,
    rows: AtomicU64,
    /// Experts the routers chose (all layers, all rows' union per layer) and their bytes.
    activations: AtomicU64,
    activated_bytes: AtomicU64,
    /// Chosen experts that were probed for residency, how many of them were in memory, and the bytes of the probed ones / those missing.
    probed: AtomicU64,
    probe_hits: AtomicU64,
    probed_bytes: AtomicU64,
    missing_bytes: AtomicU64,
    /// Read-ahead of the next layer: experts requested because they were not in memory, bytes, and how many of them the layer then used.
    prefetch_issued: AtomicU64,
    prefetch_bytes: AtomicU64,
    prefetch_useful: AtomicU64,
    prefetch_wasted: AtomicU64,
    /// Experts requested at routing time because they were not in memory (their reads overlap the computation of the first ones).
    demand_issued: AtomicU64,
    evicted_experts: AtomicU64,
    evicted_bytes: AtomicU64,
}

/// The sizes the cache works with, fixed by `ExpertCache::finalize`.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Bytes of expert weights that stay in host memory (experts on a device are not counted).
    pub expert_bytes: u64,
    pub budget: u64,
    /// The budget came from the free memory, not from the user.
    pub automatic: bool,
    /// Experts do not all fit: they are read from the file again and again.
    pub streaming: bool,
    /// The cache drops experts itself to stay within the budget (an explicit budget that is smaller than the experts).
    pub enforce: bool,
    pub layers: usize,
    /// Bytes of the experts one token uses in all those layers (average expert size x experts per token x layers).
    pub active_bytes_per_token: u64,
    pub class: &'static str,
}

impl Plan {
    /// Share of the experts that fit in the budget.
    pub fn coverage(&self) -> f64 {
        if self.expert_bytes == 0 {
            1.0
        } else {
            (self.budget as f64 / self.expert_bytes as f64).min(1.0)
        }
    }
    /// Expected bytes read from the file per token when routing is uniform over the experts.
    pub fn cold_bytes_per_token(&self) -> u64 {
        (self.active_bytes_per_token as f64 * (1.0 - self.coverage())) as u64
    }
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        if !self.streaming {
            return format!(
                "MoE experts: {:.1} GiB in {} layers fit in memory (cache budget {:.1} GiB{}), speed class: {}",
                gib(self.expert_bytes),
                self.layers,
                gib(self.budget),
                if self.automatic { ", automatic" } else { "" },
                self.class
            );
        }
        format!(
            "MoE expert streaming: {:.1} GiB of experts in {} layers, cache budget {:.1} GiB{} ({:.0} % of the experts); a token uses {:.0} MiB of experts, \
             up to {:.0} MiB of it is read from storage per token (uniform routing, cold cache); speed class: {}",
            gib(self.expert_bytes),
            self.layers,
            gib(self.budget),
            if self.automatic { ", automatic" } else if self.enforce { ", enforced" } else { "" },
            self.coverage() * 100.0,
            self.active_bytes_per_token as f64 / 1048576.0,
            self.cold_bytes_per_token() as f64 / 1048576.0,
            self.class
        )
    }
}

fn speed_class(streaming: bool, coverage: f64) -> &'static str {
    if !streaming {
        "in-memory"
    } else if coverage >= 0.75 {
        "mostly cached"
    } else if coverage >= 0.30 {
        "streaming"
    } else {
        "disk-bound"
    }
}

enum Job {
    Prefetch { table: Arc<LayerTable>, experts: Vec<u16>, tick: u64 },
    Evict,
    Rebalance,
    /// Answered when every job queued before it is done (tests).
    Barrier(mpsc::Sender<()>),
}

struct Core {
    label: String,
    opts: Options,
    tick: AtomicU64,
    plan: OnceLock<Plan>,
    /// The expert layers in order of registration (ring order of the read-ahead).
    layers: Mutex<Vec<Weak<LayerTable>>>,
    tx: Mutex<Option<mpsc::Sender<Job>>>,
    resident_bytes: AtomicU64,
    hot_bytes: AtomicU64,
    stats: Stats,
    /// `/proc/self/io` read counter when the cache was finalized.
    read_base: AtomicU64,
}

/// Cheap handle kept by the decoder; dropping it ends the background thread.
pub struct ExpertCache {
    core: Arc<Core>,
}

static REGISTRY: Mutex<Vec<Weak<Core>>> = Mutex::new(Vec::new());

impl ExpertCache {
    /// `None` when the cache layer is switched off (`--expert-cache-mb off`).
    pub fn new(label: &str) -> Option<Self> {
        let opts = options();
        if opts.budget == Budget::Off {
            return None;
        }
        let core = Arc::new(Core {
            label: label.to_owned(),
            opts,
            tick: AtomicU64::new(0),
            plan: OnceLock::new(),
            layers: Mutex::new(Vec::new()),
            tx: Mutex::new(None),
            resident_bytes: AtomicU64::new(0),
            hot_bytes: AtomicU64::new(0),
            stats: Stats::default(),
            read_base: AtomicU64::new(0),
        });
        let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        registry.retain(|w| w.strong_count() > 0);
        registry.push(Arc::downgrade(&core));
        Some(Self { core })
    }

    /// Register the experts of one layer: for every expert the tensors that hold its weights (gate, up, down). Returns `None` when the
    /// weights are not memory-mapped file data (nothing to stream).
    pub fn register_layer(&self, layer: usize, used: usize, experts: &[Vec<TensorBytes>]) -> Option<Arc<LayerTable>> {
        if experts.is_empty() || experts.iter().flatten().any(|t| !t.is_mapped() || t.is_empty()) || experts.len() > u16::MAX as usize {
            return None;
        }
        let entries: Vec<Entry> = experts
            .iter()
            .map(|parts| Entry {
                bytes: parts.iter().map(|p| p.len() as u64).sum(),
                parts: parts.clone(),
                heat: AtomicU32::new(0),
                last: AtomicU64::new(0),
                resident: AtomicBool::new(false),
                hot: AtomicBool::new(false),
                predicted: AtomicBool::new(false),
            })
            .collect();
        let mut layers = self.core.layers.lock().unwrap_or_else(|e| e.into_inner());
        let table = Arc::new(LayerTable {
            core: Arc::clone(&self.core),
            layer,
            used: used.max(1),
            entries,
            next: OnceLock::new(),
            first: AtomicBool::new(layers.iter().all(|w| w.strong_count() == 0)),
            last: Mutex::new(Vec::new()),
            device: AtomicBool::new(false),
        });
        if let Some(previous) = layers.iter().rev().find_map(Weak::upgrade) {
            let _ = previous.next.set(Arc::downgrade(&table));
        }
        layers.push(Arc::downgrade(&table));
        Some(table)
    }

    /// Fix the plan once all layers are registered (and the ones that run on a device are marked): the budget, the speed class, the
    /// hints for the operating system. Returns the plan, which is also logged by the caller.
    pub fn finalize(&self, dense_host_bytes: u64) -> Plan {
        let core = &self.core;
        if let Some(plan) = core.plan.get() {
            return plan.clone();
        }
        let tables: Vec<Arc<LayerTable>> = core.layers.lock().unwrap_or_else(|e| e.into_inner()).iter().filter_map(Weak::upgrade).filter(|t| !t.device.load(Relaxed)).collect();
        let expert_bytes: u64 = tables.iter().flat_map(|t| t.entries.iter()).map(|e| e.bytes).sum();
        let (budget, automatic) = match core.opts.budget {
            Budget::Bytes(b) => (b, false),
            _ => {
                let avail = effective_available_bytes();
                let reserve = (avail / 50).max(128 << 20);
                let b = (avail as f64 * 0.95) as u64;
                (b.saturating_sub(dense_host_bytes).saturating_sub(reserve).max(MIN_AUTO_BUDGET), true)
            }
        };
        let streaming = budget < expert_bytes;
        let active: u64 = tables.iter().map(|t| if t.entries.is_empty() { 0 } else { (t.entries.iter().map(|e| e.bytes).sum::<u64>() / t.entries.len() as u64) * t.used as u64 }).sum();
        let plan = Plan {
            expert_bytes,
            budget,
            automatic,
            streaming,
            enforce: streaming && !automatic && core.opts.evict,
            layers: tables.len(),
            active_bytes_per_token: active,
            class: speed_class(streaming, if expert_bytes == 0 { 1.0 } else { budget as f64 / expert_bytes as f64 }),
        };
        if streaming && core.opts.random_access {
            for e in tables.iter().flat_map(|t| t.entries.iter()) {
                for p in &e.parts {
                    os::random_access(p.as_ptr() as usize, p.len());
                }
            }
        }
        // the first routed layer of every pass counts the passes
        if let Some(first) = tables.first() {
            first.first.store(true, Relaxed);
        }
        for t in tables.iter().skip(1) {
            t.first.store(false, Relaxed);
        }
        if let (Some(first), Some(last)) = (tables.first(), tables.last()) {
            let _ = last.next.set(Arc::downgrade(first));
        }
        core.read_base.store(os::disk_read_bytes().unwrap_or(0), Relaxed);
        let _ = core.plan.set(plan.clone());
        plan
    }

    pub fn plan(&self) -> Option<Plan> {
        self.core.plan.get().cloned()
    }

    pub fn snapshot(&self) -> Snapshot {
        snapshot_of(&self.core)
    }

    /// Wait until the background work queued so far is done (tests).
    pub fn flush(&self) {
        let (tx, rx) = mpsc::channel();
        self.core.send(Job::Barrier(tx));
        let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
    }
}

struct Entry {
    parts: Vec<TensorBytes>,
    bytes: u64,
    /// Decayed activation count.
    heat: AtomicU32,
    /// Forward pass of the last use.
    last: AtomicU64,
    resident: AtomicBool,
    hot: AtomicBool,
    /// Requested ahead of its use, because it was not in memory.
    predicted: AtomicBool,
}

impl Entry {
    fn will_need(&self) {
        for p in &self.parts {
            os::will_need(p.as_ptr() as usize, p.len());
        }
    }
    fn evict(&self) {
        for p in &self.parts {
            os::evict(p.as_ptr() as usize, p.len());
        }
    }
    /// `(resident, sampled)` pages of a few places inside the expert.
    fn probe(&self) -> (usize, usize) {
        const SAMPLES: usize = 4;
        let page = os::page_size();
        let mut addrs = Vec::with_capacity(self.parts.len() * SAMPLES);
        for p in &self.parts {
            let n = SAMPLES.min(os::pages_of(p.len()));
            let base = p.as_ptr() as usize;
            for k in 0..n {
                addrs.push((base + p.len() * (2 * k + 1) / (2 * n)) & !(page - 1));
            }
        }
        match os::resident_pages(&addrs) {
            Some(r) => (r, addrs.len()),
            None => (addrs.len(), addrs.len()),
        }
    }
}

/// The experts of one routed layer inside the mapped model file.
pub struct LayerTable {
    core: Arc<Core>,
    layer: usize,
    used: usize,
    entries: Vec<Entry>,
    next: OnceLock<Weak<LayerTable>>,
    first: AtomicBool,
    /// Experts of the last decoding pass.
    last: Mutex<Vec<u16>>,
    /// The experts of this layer were uploaded to a device; they are never routed on the host.
    device: AtomicBool,
}

impl LayerTable {
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The experts of this layer live on a device: they do not count as host expert memory.
    pub fn mark_device_resident(&self) {
        self.device.store(true, Relaxed);
    }

    /// The router of this layer chose `active` experts (the union over `rows` tokens). Called before the experts are computed.
    pub fn on_route(self: &Arc<Self>, active: &[usize], rows: usize) {
        let core = &self.core;
        let Some(plan) = core.plan.get() else { return };
        let stats = &core.stats;
        let bulk = rows > BULK_ROWS;
        if self.first.load(Relaxed) {
            core.tick.fetch_add(1, Relaxed);
            stats.passes.fetch_add(1, Relaxed);
            stats.rows.fetch_add(rows as u64, Relaxed);
        }
        let tick = core.tick.load(Relaxed);
        trace(self.layer, tick, rows, active);
        let probe = plan.streaming || tick <= 32 || tick.is_multiple_of(8);
        let mut bytes_in_use = 0u64;
        for &e in active {
            let Some(entry) = self.entries.get(e) else { continue };
            bytes_in_use += entry.bytes;
            if entry.predicted.swap(false, Relaxed) {
                stats.prefetch_useful.fetch_add(1, Relaxed);
            }
            if probe {
                let (resident, total) = entry.probe();
                stats.probed.fetch_add(1, Relaxed);
                stats.probed_bytes.fetch_add(entry.bytes, Relaxed);
                if resident as f64 >= RESIDENT_SHARE * total as f64 {
                    stats.probe_hits.fetch_add(1, Relaxed);
                } else {
                    stats.missing_bytes.fetch_add(entry.bytes * (total - resident) as u64 / total.max(1) as u64, Relaxed);
                    if core.opts.demand {
                        // all of its pages at once: the read overlaps the computation of the experts before it
                        entry.will_need();
                        stats.demand_issued.fetch_add(1, Relaxed);
                    }
                }
            }
            if !entry.resident.swap(true, Relaxed) {
                core.resident_bytes.fetch_add(entry.bytes, Relaxed);
            }
            entry.last.store(tick, Relaxed);
            entry.heat.fetch_add(if bulk { 1 } else { 16 }, Relaxed);
        }
        stats.activations.fetch_add(active.len() as u64, Relaxed);
        stats.activated_bytes.fetch_add(bytes_in_use, Relaxed);
        // requested ahead of time but not used by this pass
        for entry in &self.entries {
            if entry.predicted.swap(false, Relaxed) {
                stats.prefetch_wasted.fetch_add(1, Relaxed);
            }
        }
        if bulk {
            return;
        }
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = active.iter().map(|&e| e as u16).collect();
        if plan.streaming && core.opts.prefetch {
            if let Some(next) = self.next.get().and_then(Weak::upgrade) {
                // the first layer of the next pass is predicted a pass ahead
                let wraps = next.first.load(Relaxed);
                let experts = next.predict();
                if !experts.is_empty() {
                    core.send(Job::Prefetch { table: next, experts, tick: tick + u64::from(wraps) });
                }
            }
        }
        if self.first.load(Relaxed) {
            if plan.enforce && core.resident_bytes.load(Relaxed) > plan.budget {
                core.send(Job::Evict);
            }
            if plan.enforce && tick.is_multiple_of(REBALANCE_TICKS) {
                core.send(Job::Rebalance);
            }
        }
    }

    /// Experts this layer is expected to use next: those of its last decoding pass, filled up with the hottest ones.
    fn predict(&self) -> Vec<u16> {
        let mut out: Vec<u16> = self.last.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if out.len() < self.used {
            let mut by_heat: Vec<(u32, u16)> = self.entries.iter().enumerate().map(|(i, e)| (e.heat.load(Relaxed), i as u16)).filter(|(h, i)| *h > 0 && !out.contains(i)).collect();
            by_heat.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            out.extend(by_heat.into_iter().take(self.used - out.len()).map(|(_, i)| i));
        }
        out
    }
}

impl Core {
    fn send(self: &Arc<Self>, job: Job) {
        if self.opts.inline {
            self.run(job);
            return;
        }
        let mut slot = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let (tx, rx) = mpsc::channel::<Job>();
            let core = Arc::downgrade(self);
            let spawned = std::thread::Builder::new().name("fllm-expert-io".into()).spawn(move || {
                // ends when the cache (and with it the sender) is gone; queued jobs hold their tables, so they are done first
                while let Ok(job) = rx.recv() {
                    match core.upgrade() {
                        Some(core) => core.run(job),
                        None => {
                            if let Job::Barrier(done) = job {
                                let _ = done.send(());
                            }
                        }
                    }
                }
            });
            if spawned.is_err() {
                return;
            }
            *slot = Some(tx);
        }
        let _ = slot.as_ref().map(|tx| tx.send(job));
    }

    fn run(&self, job: Job) {
        match job {
            Job::Prefetch { table, experts, tick } => {
                if self.tick.load(Relaxed) > tick {
                    return; // the pass moved on: reading these now would only compete with the experts it needs
                }
                for e in experts {
                    let Some(entry) = table.entries.get(e as usize) else { continue };
                    let (resident, total) = entry.probe();
                    if resident as f64 >= RESIDENT_SHARE * total as f64 {
                        continue;
                    }
                    entry.will_need();
                    entry.predicted.store(true, Relaxed);
                    if !entry.resident.swap(true, Relaxed) {
                        self.resident_bytes.fetch_add(entry.bytes, Relaxed);
                    }
                    self.stats.prefetch_issued.fetch_add(1, Relaxed);
                    self.stats.prefetch_bytes.fetch_add(entry.bytes * (total - resident) as u64 / total.max(1) as u64, Relaxed);
                }
            }
            Job::Evict => self.evict(),
            Job::Rebalance => self.rebalance(),
            Job::Barrier(done) => {
                let _ = done.send(());
            }
        }
    }

    fn tables(&self) -> Vec<Arc<LayerTable>> {
        self.layers.lock().unwrap_or_else(|e| e.into_inner()).iter().filter_map(Weak::upgrade).filter(|t| !t.device.load(Relaxed)).collect()
    }

    /// Drop the least recently used unpinned experts from memory until the resident bytes are back under the budget.
    fn evict(&self) {
        let Some(plan) = self.plan.get() else { return };
        if !plan.enforce || self.resident_bytes.load(Relaxed) <= plan.budget {
            return;
        }
        let tick = self.tick.load(Relaxed);
        let target = plan.budget - plan.budget / 32;
        let mut candidates: Vec<(u64, Arc<LayerTable>, usize)> = Vec::new();
        for t in self.tables() {
            for (i, e) in t.entries.iter().enumerate() {
                // not the experts of the pass in progress
                if e.resident.load(Relaxed) && !e.hot.load(Relaxed) && e.last.load(Relaxed) + 1 < tick.max(1) {
                    candidates.push((e.last.load(Relaxed), Arc::clone(&t), i));
                }
            }
        }
        candidates.sort_by_key(|(last, t, i)| (*last, t.layer, *i));
        for (_, t, i) in candidates {
            if self.resident_bytes.load(Relaxed) <= target {
                break;
            }
            let e = &t.entries[i];
            if e.resident.swap(false, Relaxed) {
                e.evict();
                self.resident_bytes.fetch_sub(e.bytes, Relaxed);
                self.stats.evicted_experts.fetch_add(1, Relaxed);
                self.stats.evicted_bytes.fetch_add(e.bytes, Relaxed);
            }
        }
    }

    /// Recompute the pinned set from the decayed counts, then decay the counts.
    fn rebalance(&self) {
        let Some(plan) = self.plan.get() else { return };
        let tables = self.tables();
        let mut ranked: Vec<(u32, usize, usize)> = Vec::new();
        for (ti, t) in tables.iter().enumerate() {
            for (i, e) in t.entries.iter().enumerate() {
                ranked.push((e.heat.load(Relaxed), ti, i));
            }
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let allowance = (plan.budget as f64 * PIN_FRACTION) as u64;
        let mut pinned = 0u64;
        for (heat, ti, i) in ranked {
            let e = &tables[ti].entries[i];
            let take = heat > 0 && plan.enforce && pinned + e.bytes <= allowance;
            if take {
                pinned += e.bytes;
                if !e.hot.swap(true, Relaxed) && !e.resident.load(Relaxed) && self.opts.prefetch {
                    e.will_need();
                    e.resident.store(true, Relaxed);
                    self.resident_bytes.fetch_add(e.bytes, Relaxed);
                }
            } else {
                e.hot.store(false, Relaxed);
            }
            let h = e.heat.load(Relaxed);
            e.heat.store(h - h / 8, Relaxed);
        }
        self.hot_bytes.store(pinned, Relaxed);
    }
}

/// What the Studio and `/metrics` show.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub label: String,
    pub configured: bool,
    pub streaming: bool,
    pub automatic: bool,
    pub class: &'static str,
    pub layers: usize,
    pub expert_bytes: u64,
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub hot_bytes: u64,
    pub passes: u64,
    pub rows: u64,
    pub activations: u64,
    pub probed: u64,
    pub hits: u64,
    pub hit_ratio: f64,
    /// Estimated bytes read from storage per processed row (token), from the sampled residency of routed experts.
    pub read_bytes_per_token: f64,
    pub missing_bytes: u64,
    pub demand_issued: u64,
    pub prefetch_issued: u64,
    pub prefetch_bytes: u64,
    pub prefetch_useful: u64,
    pub prefetch_wasted: u64,
    pub evicted_experts: u64,
    pub evicted_bytes: u64,
    /// Bytes the process read from storage since the cache was set up (Linux only).
    pub process_read_bytes: Option<u64>,
    pub expected_cold_bytes_per_token: u64,
}

fn snapshot_of(core: &Core) -> Snapshot {
    let s = &core.stats;
    let plan = core.plan.get();
    let probed = s.probed.load(Relaxed);
    let hits = s.probe_hits.load(Relaxed);
    let probed_bytes = s.probed_bytes.load(Relaxed);
    let missing = s.missing_bytes.load(Relaxed);
    let rows = s.rows.load(Relaxed);
    let miss_ratio = if probed_bytes == 0 { 0.0 } else { missing as f64 / probed_bytes as f64 };
    Snapshot {
        label: core.label.clone(),
        configured: plan.is_some(),
        streaming: plan.is_some_and(|p| p.streaming),
        automatic: plan.is_some_and(|p| p.automatic),
        class: plan.map_or("unknown", |p| p.class),
        layers: plan.map_or(0, |p| p.layers),
        expert_bytes: plan.map_or(0, |p| p.expert_bytes),
        budget_bytes: plan.map_or(0, |p| p.budget),
        resident_bytes: core.resident_bytes.load(Relaxed),
        hot_bytes: core.hot_bytes.load(Relaxed),
        passes: s.passes.load(Relaxed),
        rows,
        activations: s.activations.load(Relaxed),
        probed,
        hits,
        hit_ratio: if probed == 0 { 1.0 } else { hits as f64 / probed as f64 },
        read_bytes_per_token: if rows == 0 { 0.0 } else { miss_ratio * s.activated_bytes.load(Relaxed) as f64 / rows as f64 },
        missing_bytes: missing,
        demand_issued: s.demand_issued.load(Relaxed),
        prefetch_issued: s.prefetch_issued.load(Relaxed),
        prefetch_bytes: s.prefetch_bytes.load(Relaxed),
        prefetch_useful: s.prefetch_useful.load(Relaxed),
        prefetch_wasted: s.prefetch_wasted.load(Relaxed),
        evicted_experts: s.evicted_experts.load(Relaxed),
        evicted_bytes: s.evicted_bytes.load(Relaxed),
        process_read_bytes: os::disk_read_bytes().map(|n| n.saturating_sub(core.read_base.load(Relaxed))),
        expected_cold_bytes_per_token: plan.map_or(0, Plan::cold_bytes_per_token),
    }
}

/// The state of the most recently loaded model that has expert layers on the host.
pub fn snapshot() -> Option<Snapshot> {
    let registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.iter().rev().filter_map(Weak::upgrade).find(|c| c.plan.get().is_some_and(|p| p.layers > 0)).map(|c| snapshot_of(&c))
}

/// The Prometheus lines of `/metrics` (nothing when no loaded model streams experts).
pub fn render_metrics(out: &mut String) {
    use std::fmt::Write;
    let Some(s) = snapshot() else { return };
    let mut line = |name: &str, kind: &str, help: &str, value: String| {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {value}");
    };
    line("fllm_expert_cache_budget_bytes", "gauge", "Host memory the expert weights of MoE layers may occupy", s.budget_bytes.to_string());
    line("fllm_expert_cache_expert_bytes", "gauge", "Expert weights of MoE layers that run on the host", s.expert_bytes.to_string());
    line("fllm_expert_cache_resident_bytes", "gauge", "Expert weights believed to be in memory", s.resident_bytes.to_string());
    line("fllm_expert_cache_pinned_bytes", "gauge", "Hottest expert weights that are never dropped", s.hot_bytes.to_string());
    line("fllm_expert_cache_streaming", "gauge", "1 when the experts do not all fit and are read from storage on demand", u8::from(s.streaming).to_string());
    line("fllm_expert_cache_activations_total", "counter", "Experts chosen by the routers (union over the rows of a pass, per layer)", s.activations.to_string());
    line("fllm_expert_cache_probed_total", "counter", "Chosen experts whose residency was sampled", s.probed.to_string());
    line("fllm_expert_cache_hits_total", "counter", "Sampled experts that were already in memory", s.hits.to_string());
    line("fllm_expert_cache_hit_ratio", "gauge", "Share of sampled experts that were in memory when the router chose them", format!("{:.6}", s.hit_ratio));
    line("fllm_expert_cache_missing_bytes_total", "counter", "Estimated expert bytes that had to be read from storage", s.missing_bytes.to_string());
    line("fllm_expert_cache_read_bytes_per_token", "gauge", "Estimated expert bytes read from storage per processed token", format!("{:.0}", s.read_bytes_per_token));
    line("fllm_expert_cache_prefetch_bytes_total", "counter", "Expert bytes requested ahead of time for the next layer", s.prefetch_bytes.to_string());
    line("fllm_expert_cache_prefetch_useful_total", "counter", "Experts requested ahead of time that the layer then used", s.prefetch_useful.to_string());
    line("fllm_expert_cache_prefetch_wasted_total", "counter", "Experts requested ahead of time that the layer did not use", s.prefetch_wasted.to_string());
    line("fllm_expert_cache_evicted_bytes_total", "counter", "Expert bytes dropped from memory to keep within the budget", s.evicted_bytes.to_string());
    if let Some(n) = s.process_read_bytes {
        line("fllm_expert_cache_process_read_bytes_total", "counter", "Bytes this process read from storage since the model was loaded (Linux)", n.to_string());
    }
}

/// `expert_streaming` of the Studio status: `null` when no expert layer runs on the host.
pub fn status_json() -> serde_json::Value {
    let Some(s) = snapshot() else { return serde_json::Value::Null };
    serde_json::json!({
        "mode": s.class,
        "streaming": s.streaming,
        "budget_automatic": s.automatic,
        "layers": s.layers,
        "expert_bytes": s.expert_bytes,
        "budget_bytes": s.budget_bytes,
        "resident_bytes": s.resident_bytes,
        "pinned_bytes": s.hot_bytes,
        "hit_ratio": s.hit_ratio,
        "probed": s.probed,
        "read_bytes_per_token": s.read_bytes_per_token,
        "expected_cold_bytes_per_token": s.expected_cold_bytes_per_token,
        "prefetch_useful": s.prefetch_useful,
        "prefetch_wasted": s.prefetch_wasted,
        "passes": s.passes,
    })
}

#[cfg(test)]
mod tests;

/// `(expert bytes, other bytes)` of the tensors of a GGUF index: the other tensors (attention, shared experts, embeddings, head) are the
/// dense part that has to fit in memory.
pub fn split_bytes(index: &fllm_gguf_probe::GgufIndex) -> (u64, u64) {
    index.tensors.iter().fold((0, 0), |(exp, dense), t| if t.name.contains("_exps.") { (exp + t.payload_len, dense) } else { (exp, dense + t.payload_len) })
}
