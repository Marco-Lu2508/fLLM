//! The NPU linear engine: a registry of projections, lazily compiled OpenVINO models for them, and the host fallback.
//!
//! `NpuEngine` is independent of fLLM's `Backend` traits: it takes host slices (`x: &[f32]`) and returns host vectors, so a backend
//! wrapper (`npu/backend.rs`) or a host decoder can drive it. The flow for one product `y = x * W^T` (+ bias):
//!
//! 1. [`NpuEngine::register`] stores the raw tensor of a projection (done at weight upload; cheap, nothing is compiled);
//! 2. [`NpuEngine::run`] asks the [`OffloadPlan`]; when some rows should go to the NPU it converts the weight
//!    ([`super::quant::convert`]), builds the IR ([`super::model::build_linear_ir`]) and compiles it for the device the first time
//!    (OpenVINO's `CACHE_DIR` makes later processes skip the compile), then streams the rows through the model in chunks of the plan's
//!    static batch (the last chunk is zero padded) and returns the first `rows_done` rows of the result;
//! 3. whatever the NPU did not take (decode rows, a short remainder, any failure) is computed by [`host_linear`] with the CPU kernels.
//!
//! Failures never reach the caller as errors: a projection whose compile fails is marked and served by the CPU from then on, three
//! compile failures in a row switch the engine off, and a product whose activations do not fit f16 (|x| > 65504, NaN, Inf) or whose
//! result overflows f16 is left to the CPU for that call.
//!
//! Verification level: L0 (compiled; never run against a runtime or a device).

use super::devices::{self, OvDevice};
use super::model::build_linear_ir;
use super::plan::OffloadPlan;
use super::quant::{convert, QuantPolicy};
use super::session::{CompiledLinear, Runtime};
use crate::backend::QuantType;
use half::f16;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Identity of a registered projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightKey {
    pub id: u64,
    pub offset: usize,
    pub size: usize,
}

impl WeightKey {
    pub fn of(buffer: &crate::backend::DeviceBuffer) -> Self {
        Self {
            id: buffer.id,
            offset: buffer.offset,
            size: buffer.size,
        }
    }
}

enum SlotState {
    /// Nothing compiled yet.
    Idle,
    /// The background thread has the projection in its queue or is compiling it; the CPU serves it until it is `Ready`.
    Compiling,
    Ready(Arc<CompiledLinear>),
    /// Compile failed or the weight does not fit; the CPU serves this projection.
    Failed,
}

/// A registered projection.
pub struct Slot {
    pub rows: usize,
    pub cols: usize,
    pub quant: QuantType,
    /// The raw tensor, when the slot owns a copy (registered through [`NpuEngine::register`]); host decoders that keep the weights
    /// themselves pass the bytes on every call instead ([`NpuEngine::offload_host`]).
    host: Option<Arc<Vec<u8>>>,
    state: Mutex<SlotState>,
    /// Seconds since the engine opened at the last call; a compiled model that has been idle for long is dropped when the weight
    /// budget is needed for another model (a model that was unloaded leaves its slots behind).
    last_used: AtomicU64,
    /// Bytes the compiled model holds (set when it is compiled).
    resident: AtomicU64,
}

/// Seconds a compiled model must have been idle before the budget may take it away.
const IDLE_EVICT_SECONDS: u64 = 20;

impl Slot {
    fn new(rows: usize, cols: usize, quant: QuantType, host: Option<Arc<Vec<u8>>>) -> Self {
        Self { rows, cols, quant, host, state: Mutex::new(SlotState::Idle), last_used: AtomicU64::new(0), resident: AtomicU64::new(0) }
    }

    /// The raw tensor as uploaded, if this slot keeps a copy.
    pub fn host_bytes(&self) -> Option<&[u8]> {
        self.host.as_deref().map(Vec::as_slice)
    }

    /// The same copy as a shared handle (for [`crate::backend::npu::backend::hybrid::OffloadSlot`]).
    pub fn host_arc(&self) -> Option<Arc<Vec<u8>>> {
        self.host.clone()
    }
}

/// Counters for the end-of-run summary.
#[derive(Default)]
pub struct Stats {
    pub npu_calls: AtomicU64,
    pub npu_rows: AtomicU64,
    pub cpu_rows: AtomicU64,
    pub compiled: AtomicU64,
    pub compile_failures: AtomicU64,
    pub range_fallbacks: AtomicU64,
    pub compile_millis: AtomicU64,
    pub npu_millis: AtomicU64,
    /// Microseconds of the f32 -> f16 conversion of the activations, of the NPU call (copy in, run, copy out) and of the f16 -> f32
    /// conversion of the results; where the time of an offloaded projection goes (`FLLM_NPU_STATS`).
    pub to_f16_micros: AtomicU64,
    pub infer_micros: AtomicU64,
    pub from_f16_micros: AtomicU64,
    /// Models compiled by the background thread, and calls the CPU served because a model was not ready yet.
    pub background_compiles: AtomicU64,
    pub waiting_calls: AtomicU64,
}

/// A projection waiting for the background compile thread, with a private copy of its raw tensor (the thread must not depend on the
/// caller's memory staying mapped).
struct Job {
    slot: Arc<Slot>,
    raw: Vec<u8>,
}

/// Raw tensor bytes the compile queue may hold at once; further projections wait (the CPU serves them) until the thread has caught up.
const QUEUE_LIMIT_BYTES: u64 = 192 << 20;

/// A projection whose first compile took more than this many milliseconds per MiB of weights was not in the blob cache: the remaining
/// models are then compiled by the background thread instead of stalling the first prompt.
const COLD_COMPILE_MS_PER_MIB: f64 = 60.0;

#[derive(Default)]
struct CompileQueue {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    ready: std::sync::Condvar,
    pending_bytes: AtomicU64,
}

/// How the models are compiled: decided by the first compile (`Unknown`: that one runs in the caller).
const MODE_UNKNOWN: u8 = 0;
const MODE_INLINE: u8 = 1;
const MODE_BACKGROUND: u8 = 2;

/// Rows the NPU computed.
pub struct NpuOutput {
    pub rows_done: usize,
    /// `rows_done * slot.rows` values.
    pub values: Vec<f32>,
}

pub struct NpuEngine {
    /// `None` for an engine without OpenVINO (tests, or a host that only wants the CPU kernels behind the same interface).
    runtime: Option<Arc<Runtime>>,
    pub device: OvDevice,
    pub plan: OffloadPlan,
    props: Vec<(String, String)>,
    budget: u64,
    resident: AtomicU64,
    slots: Mutex<HashMap<WeightKey, Arc<Slot>>>,
    dump_dir: Option<PathBuf>,
    consecutive_failures: AtomicUsize,
    disabled: AtomicBool,
    queue: Arc<CompileQueue>,
    compile_mode: AtomicU8,
    started: Instant,
    /// The blob cache directory had no entries when the engine opened (the first start): everything is compiled for the first time.
    cache_cold: bool,
    /// `FLLM_NPU_STATS=N`: log a summary every N offloaded projections (0 = never).
    stats_every: u64,
    pub stats: Stats,
}

/// Directory for OpenVINO's compiled-model cache: `FLLM_NPU_CACHE_DIR` (empty = off), else the user cache directory.
fn cache_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("FLLM_NPU_CACHE_DIR") {
        Some(value) if value.is_empty() => return None,
        Some(value) => PathBuf::from(value),
        None => {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .filter(|v| !v.is_empty())
                        .map(|h| PathBuf::from(h).join(".cache"))
                })
                .or_else(|| {
                    std::env::var_os("LOCALAPPDATA")
                        .filter(|v| !v.is_empty())
                        .map(PathBuf::from)
                })?;
            base.join("fllm").join("npu")
        }
    };
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// `FLLM_NPU_PROPS="KEY=VALUE,KEY=VALUE"`: extra compile properties (for example `NPU_TILES=2`); at most two.
fn user_props() -> Vec<(String, String)> {
    std::env::var("FLLM_NPU_PROPS")
        .map(|text| {
            text.split(',')
                .filter_map(|pair| pair.split_once('='))
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                .filter(|(k, _)| !k.is_empty())
                .take(2)
                .collect()
        })
        .unwrap_or_default()
}

impl NpuEngine {
    /// Open the `index`-th OpenVINO NPU device (numbered like [`devices::enumerate`]).
    pub fn open(index: usize) -> Result<Arc<NpuEngine>, String> {
        let found = devices::enumerate();
        let device = found.get(index).cloned().ok_or_else(|| {
            if found.is_empty() {
                format!(
                    "no OpenVINO {} device found (OpenVINO runtime, NPU plugin and driver are required; see rust/engine/src/backend/npu/OPENVINO.md)",
                    devices::device_prefix()
                )
            } else {
                format!("OpenVINO device index {index} out of range ({} found)", found.len())
            }
        })?;
        let runtime = Runtime::shared()?;
        let plan = OffloadPlan::from_env();
        let mut props = Vec::new();
        if let Some(dir) = cache_dir() {
            props.push(("CACHE_DIR".to_owned(), dir.to_string_lossy().into_owned()));
        }
        props.extend(user_props());
        props.truncate(3);
        // keep a margin: the device memory is shared with the system, never plan for all of it
        let device_limit = device.total_memory / 4 * 3;
        let budget = match (plan.budget_bytes, device_limit) {
            (0, 0) => 0,
            (0, limit) => limit,
            (b, 0) => b,
            (b, limit) => b.min(limit),
        };
        // the memory planner: the compiled models live in the memory of the computer, next to the model, the KV cache and the
        // other programs; without a budget of the user they may take 40 % of what is free now (at least 1 GiB)
        let budget = if plan.budget_bytes == 0 { planned_budget(budget, crate::devices::host_free_memory_bytes()) } else { budget };
        let engine = NpuEngine {
            runtime: Some(runtime),
            device,
            plan,
            props,
            budget,
            resident: AtomicU64::new(0),
            slots: Mutex::new(HashMap::new()),
            dump_dir: std::env::var_os("FLLM_NPU_DUMP_IR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            consecutive_failures: AtomicUsize::new(0),
            disabled: AtomicBool::new(false),
            queue: Arc::new(CompileQueue::default()),
            started: Instant::now(),
            cache_cold: cache_dir().is_none_or(|d| std::fs::read_dir(d).map_or(true, |mut e| e.next().is_none())),
            compile_mode: AtomicU8::new(match std::env::var("FLLM_NPU_COMPILE").as_deref().map(str::trim) {
                Ok("inline") => MODE_INLINE,
                Ok("background") => MODE_BACKGROUND,
                _ => MODE_UNKNOWN,
            }),
            stats_every: std::env::var("FLLM_NPU_STATS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0),
            stats: Stats::default(),
        };
        crate::backend::log_once("npu-plan", &engine.plan.describe());
        crate::backend::log_once(
            "npu-memory",
            &format!(
                "NPU weight budget {} MiB (device {} MiB, computer free {} MiB); compiled models cached in {}",
                engine.budget >> 20,
                engine.device.total_memory >> 20,
                crate::devices::host_free_memory_bytes() >> 20,
                cache_dir().map_or_else(|| "nowhere (FLLM_NPU_CACHE_DIR is empty)".to_owned(), |d| d.display().to_string())
            ),
        );
        let engine = Arc::new(engine);
        spawn_compile_thread(&engine);
        Ok(engine)
    }

    /// An engine without a runtime: every `run` returns `None`, so everything is left to the CPU kernels. For tests and for hosts that
    /// want one code path whether or not an NPU is present.
    pub fn offline(plan: OffloadPlan) -> Arc<NpuEngine> {
        Arc::new(NpuEngine {
            runtime: None,
            device: OvDevice {
                name: "none".to_owned(),
                full_name: "no OpenVINO runtime".to_owned(),
                architecture: String::new(),
                driver: String::new(),
                total_memory: 0,
                allocated_memory: 0,
            },
            plan,
            props: Vec::new(),
            budget: 0,
            resident: AtomicU64::new(0),
            slots: Mutex::new(HashMap::new()),
            dump_dir: None,
            consecutive_failures: AtomicUsize::new(0),
            disabled: AtomicBool::new(false),
            queue: Arc::new(CompileQueue::default()),
            started: Instant::now(),
            cache_cold: false,
            compile_mode: AtomicU8::new(MODE_INLINE),
            stats_every: 0,
            stats: Stats::default(),
        })
    }

    /// Whether three compiles in a row failed (the engine then serves nothing).
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }

    /// Remember a projection. `raw` is the tensor as passed to `Backend::upload_weight`.
    pub fn register(
        &self,
        key: WeightKey,
        quant: QuantType,
        rows: usize,
        cols: usize,
        raw: &[u8],
    ) -> Arc<Slot> {
        let slot = Arc::new(Slot::new(rows, cols, quant, Some(Arc::new(raw.to_vec()))));
        if let Ok(mut slots) = self.slots.lock() {
            slots.insert(key, Arc::clone(&slot));
        }
        slot
    }

    pub fn slot(&self, key: &WeightKey) -> Option<Arc<Slot>> {
        self.slots.lock().ok()?.get(key).cloned()
    }

    /// Forget a projection and release its compiled model.
    pub fn unregister(&self, key: &WeightKey) {
        let removed = self
            .slots
            .lock()
            .ok()
            .and_then(|mut slots| slots.remove(key));
        if let Some(slot) = removed {
            if let Ok(state) = slot.state.lock() {
                if matches!(&*state, SlotState::Ready(_)) {
                    let bytes = slot.resident.swap(0, Ordering::Relaxed);
                    let mut current = self.resident.load(Ordering::Relaxed);
                    while let Err(actual) = self.resident.compare_exchange_weak(
                        current,
                        current.saturating_sub(bytes),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        current = actual;
                    }
                }
            }
        }
    }

    /// Bytes of weights the compiled models hold (estimate from the plan's conversion policy).
    pub fn resident_bytes(&self) -> u64 {
        self.resident.load(Ordering::Relaxed)
    }

    /// Convert, build and compile the model of a projection; returns it with the milliseconds the compile took.
    fn compile(&self, slot: &Slot, raw: &[u8]) -> Result<(Arc<CompiledLinear>, u64), String> {
        let weight = convert(slot.quant, raw, slot.rows, slot.cols, &self.plan.quant)?;
        let bytes = weight.bytes() as u64;
        if self.budget > 0 && self.resident.load(Ordering::Relaxed) + bytes > self.budget {
            self.evict_idle(bytes);
            if self.resident.load(Ordering::Relaxed) + bytes > self.budget {
                return Err(format!("NPU weight budget of {} MiB is used up", self.budget >> 20));
            }
        }
        let name = format!("fllm_linear_{}x{}", slot.rows, slot.cols);
        let ir = build_linear_ir(&weight, self.plan.chunk, &name)?;
        if let Some(dir) = &self.dump_dir {
            let id = self.stats.compiled.load(Ordering::Relaxed)
                + self.stats.compile_failures.load(Ordering::Relaxed);
            let base = dir.join(format!("{name}_{id}"));
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(base.with_extension("xml"), &ir.xml);
            let _ = std::fs::write(base.with_extension("bin"), &ir.bin);
        }
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "no OpenVINO runtime".to_owned())?;
        let started = Instant::now();
        let compiled = runtime.compile_ir(&ir.xml, &ir.bin, &self.device.name, &self.props)?;
        let elapsed = started.elapsed().as_millis() as u64;
        self.stats
            .compile_millis
            .fetch_add(elapsed, Ordering::Relaxed);
        self.resident.fetch_add(bytes, Ordering::Relaxed);
        slot.resident.store(bytes, Ordering::Relaxed);
        crate::backend::log_once(
            "npu-first-compile",
            &format!(
                "first NPU model ({name}, {:?}, chunk {}) compiled in {elapsed} ms",
                weight.format, self.plan.chunk
            ),
        );
        // a first compile that was slow for its size means the blob cache did not have it: the rest is compiled in the background
        if self.compile_mode.load(Ordering::Relaxed) == MODE_UNKNOWN {
            let per_mib = elapsed as f64 / (bytes as f64 / (1 << 20) as f64).max(0.25);
            let background = self.cache_cold || per_mib > COLD_COMPILE_MS_PER_MIB;
            self.compile_mode.store(if background { MODE_BACKGROUND } else { MODE_INLINE }, Ordering::Relaxed);
            crate::backend::log_once(
                "npu-compile-mode",
                &if background {
                    format!("NPU models are not in the blob cache ({elapsed} ms for the first); compiling the rest in the background, the processor serves prompts meanwhile")
                } else {
                    format!("NPU models come from the blob cache ({elapsed} ms for the first); compiling the rest while the prompt waits")
                },
            );
        }
        Ok((Arc::new(compiled), elapsed))
    }

    /// Drop the compiled models that have been idle for a while until `needed` more bytes fit the budget.
    fn evict_idle(&self, needed: u64) {
        let now = self.started.elapsed().as_secs();
        let Ok(slots) = self.slots.lock() else { return };
        let mut idle: Vec<&Arc<Slot>> = slots
            .values()
            .filter(|s| s.resident.load(Ordering::Relaxed) > 0 && now.saturating_sub(s.last_used.load(Ordering::Relaxed)) >= IDLE_EVICT_SECONDS)
            .collect();
        idle.sort_by_key(|s| s.last_used.load(Ordering::Relaxed));
        for slot in idle {
            if self.resident.load(Ordering::Relaxed) + needed <= self.budget {
                break;
            }
            // never wait for a slot that is compiling or running: skip it
            let Ok(mut state) = slot.state.try_lock() else { continue };
            if matches!(&*state, SlotState::Ready(_)) {
                *state = SlotState::Idle;
                let bytes = slot.resident.swap(0, Ordering::Relaxed);
                self.resident.fetch_sub(bytes.min(self.resident.load(Ordering::Relaxed)), Ordering::Relaxed);
            }
        }
    }

    /// The compiled model of a slot, compiling it on first use; `None` when it cannot be served by the NPU.
    fn session(&self, slot: &Arc<Slot>, raw: &[u8]) -> Option<Arc<CompiledLinear>> {
        if self.is_disabled() {
            return None;
        }
        slot.last_used.store(self.started.elapsed().as_secs(), Ordering::Relaxed);
        let mut state = slot.state.lock().ok()?;
        match &*state {
            SlotState::Ready(session) => return Some(Arc::clone(session)),
            SlotState::Failed => return None,
            SlotState::Compiling => {
                self.stats.waiting_calls.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            SlotState::Idle => {}
        }
        if self.compile_mode.load(Ordering::Relaxed) == MODE_BACKGROUND {
            // hand the projection to the compile thread (it takes a private copy of the tensor) and let the CPU serve this call
            if self.queue.pending_bytes.load(Ordering::Relaxed) + raw.len() as u64 > QUEUE_LIMIT_BYTES {
                self.stats.waiting_calls.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.queue.pending_bytes.fetch_add(raw.len() as u64, Ordering::Relaxed);
            *state = SlotState::Compiling;
            if let Ok(mut jobs) = self.queue.jobs.lock() {
                jobs.push_back(Job { slot: Arc::clone(slot), raw: raw.to_vec() });
                self.queue.ready.notify_one();
            }
            self.stats.waiting_calls.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let result = self.compile(slot, raw);
        self.settle(slot, &mut state, result)
    }

    /// Record the outcome of a compile in the slot and the counters.
    fn settle(&self, slot: &Slot, state: &mut SlotState, result: Result<(Arc<CompiledLinear>, u64), String>) -> Option<Arc<CompiledLinear>> {
        match result {
            Ok((session, _)) => {
                self.stats.compiled.fetch_add(1, Ordering::Relaxed);
                self.consecutive_failures.store(0, Ordering::Relaxed);
                *state = SlotState::Ready(Arc::clone(&session));
                Some(session)
            }
            Err(error) => {
                self.stats.compile_failures.fetch_add(1, Ordering::Relaxed);
                crate::backend::log_once(
                    &format!("npu-compile-{}x{}-{:?}", slot.rows, slot.cols, slot.quant),
                    &format!(
                        "NPU compile of a {}x{} {:?} projection failed, the CPU serves it: {error}",
                        slot.rows, slot.cols, slot.quant
                    ),
                );
                *state = SlotState::Failed;
                // a budget refusal is not a sign of a broken device
                if !error.contains("budget")
                    && self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1 >= 3
                {
                    self.disabled.store(true, Ordering::Relaxed);
                    crate::backend::log_once("npu-disabled", "three NPU compiles failed in a row; the NPU offload is switched off for this process");
                }
                None
            }
        }
    }

    /// What the background thread does with one queued projection.
    fn finish_job(&self, job: Job) {
        let result = self.compile(&job.slot, &job.raw);
        if result.is_ok() {
            self.stats.background_compiles.fetch_add(1, Ordering::Relaxed);
        }
        if let Ok(mut state) = job.slot.state.lock() {
            self.settle(&job.slot, &mut state, result);
        }
    }

    /// Models waiting for or inside the background compile (for the status line).
    pub fn compiling(&self) -> usize {
        self.slots
            .lock()
            .map(|slots| slots.values().filter(|s| s.state.try_lock().map_or(true, |st| matches!(&*st, SlotState::Compiling))).count())
            .unwrap_or(0)
    }

    /// Run the rows of `x` (`batch * slot.cols` values) that the plan assigns to the NPU through the projection `slot`.
    ///
    /// `None` when the NPU takes nothing (plan, compile failure, f16 range): the caller computes all rows itself. Otherwise the
    /// result holds the first `rows_done` rows (without bias; the caller adds it) and the caller computes rows `rows_done..batch`.
    pub fn run(&self, slot: &Arc<Slot>, x: &[f32], batch: usize) -> Option<NpuOutput> {
        let raw = slot.host_bytes()?;
        self.run_raw(slot, raw, x, batch)
    }

    /// [`run`](Self::run) for a caller that supplies the raw tensor itself (needed only for the first call, when the model is built).
    pub fn run_raw(&self, slot: &Arc<Slot>, raw: &[u8], x: &[f32], batch: usize) -> Option<NpuOutput> {
        let (n, k) = (slot.rows, slot.cols);
        if self.runtime.is_none()
            || x.len() != batch * k
            || !self.plan.should_offload(batch, n, k, slot.quant)
        {
            return None;
        }
        let take_rows = self.plan.npu_rows(batch);
        let input = &x[..take_rows * k];
        let session = self.session(slot, raw)?;
        let chunk = self.plan.chunk;
        let started = Instant::now();
        let mut values = vec![0f32; take_rows * n];
        let mut x16 = vec![0u16; chunk * k];
        let mut y16 = vec![0u16; chunk * n];
        let mut done = 0;
        let micros = |since: Instant| since.elapsed().as_micros() as u64;
        while done < take_rows {
            let take = chunk.min(take_rows - done);
            // f16 has a range of 65504 and the NPU computes in f16: activations that do not fit are left to the CPU
            let t = Instant::now();
            if !to_f16_checked(&mut x16[..take * k], &input[done * k..(done + take) * k]) {
                self.stats.range_fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            x16[take * k..].fill(0);
            self.stats.to_f16_micros.fetch_add(micros(t), Ordering::Relaxed);
            let t = Instant::now();
            if let Err(error) = session.run(&x16, &mut y16) {
                crate::backend::log_once(
                    "npu-run-error",
                    &format!("NPU inference failed, the CPU takes over: {error}"),
                );
                return None;
            }
            self.stats.infer_micros.fetch_add(micros(t), Ordering::Relaxed);
            // an f16 overflow of a sum shows up as Inf: redo the call on the CPU
            let t = Instant::now();
            if !from_f16_checked(&mut values[done * n..(done + take) * n], &y16[..take * n]) {
                self.stats.range_fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.stats.from_f16_micros.fetch_add(micros(t), Ordering::Relaxed);
            done += take;
        }
        let calls = self.stats.npu_calls.fetch_add(1, Ordering::Relaxed) + 1;
        self.stats
            .npu_rows
            .fetch_add(take_rows as u64, Ordering::Relaxed);
        self.stats
            .npu_millis
            .fetch_add(started.elapsed().as_millis() as u64, Ordering::Relaxed);
        if self.stats_every > 0 && calls.is_multiple_of(self.stats_every) {
            eprintln!("[fllm-npu] {}", self.summary());
        }
        Some(NpuOutput {
            rows_done: take_rows,
            values,
        })
    }

    /// Offload for a host decoder that keeps its weights itself (for example memory-mapped GGUF tensors): `raw` is the tensor of
    /// the projection `rows x cols` in format `quant`, `x` holds `batch` input rows. The projection is identified by the address of
    /// `raw`, so the slice must stay at the same place for the life of the process (a mapped model file does). Nothing is copied except
    /// into the compiled NPU model on the first call. Returns `None` when the NPU takes nothing (see [`run`](Self::run)); otherwise the
    /// caller computes rows `rows_done..batch` itself and adds the bias.
    pub fn offload_host(
        &self,
        quant: QuantType,
        raw: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        batch: usize,
    ) -> Option<NpuOutput> {
        if self.runtime.is_none() || !self.plan.should_offload(batch, rows, cols, quant) {
            return None;
        }
        // the address alone is not an identity: a model that was unloaded and another one mapped at the same address must not meet the
        // compiled weights of the first, so a fingerprint of the bytes is part of the key
        let key = WeightKey {
            id: raw.as_ptr() as usize as u64,
            offset: fingerprint(raw) as usize,
            size: raw.len(),
        };
        let slot = {
            let mut slots = self.slots.lock().ok()?;
            Arc::clone(slots.entry(key).or_insert_with(|| Arc::new(Slot::new(rows, cols, quant, None))))
        };
        if slot.rows != rows || slot.cols != cols || slot.quant != quant {
            return None;
        }
        self.run_raw(&slot, raw, x, batch)
    }

    /// Hardware self-check (the L1 test of this backend): compile a small Q4_0 projection for the device, run it and compare the
    /// result with the host product of the dequantized weights and the f16-rounded activations. `Ok(report)` when the largest error
    /// is below 2 % of the largest result.
    pub fn selfcheck(&self) -> Result<String, String> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "no OpenVINO runtime".to_owned())?;
        let (rows, cols, batch) = (192usize, 512usize, 64usize);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let mut raw = Vec::with_capacity(rows * cols / 32 * 18);
        for _ in 0..rows * cols / 32 {
            raw.extend_from_slice(&f16::from_f32(0.02 + next().abs() * 0.05).to_le_bytes());
            for _ in 0..16 {
                raw.push((next() * 127.0) as i8 as u8);
            }
        }
        let weight = convert(QuantType::Q4_0, &raw, rows, cols, &QuantPolicy::default())?;
        let ir = build_linear_ir(&weight, batch, "fllm_selfcheck")?;
        let started = Instant::now();
        let session = runtime.compile_ir(&ir.xml, &ir.bin, &self.device.name, &self.props)?;
        let compile_ms = started.elapsed().as_secs_f64() * 1e3;
        let x: Vec<u16> = (0..batch * cols)
            .map(|_| f16::from_f32(next() * 2.0).to_bits())
            .collect();
        let mut y = vec![0u16; batch * rows];
        session.run(&x, &mut y)?;
        let started = Instant::now();
        session.run(&x, &mut y)?;
        let run_ms = started.elapsed().as_secs_f64() * 1e3;
        let w = weight.dequantize();
        let (mut worst, mut largest) = (0f64, 0f64);
        for b in 0..batch {
            for r in 0..rows {
                let reference: f64 = (0..cols)
                    .map(|c| {
                        f64::from(f16::from_bits(x[b * cols + c]).to_f32())
                            * f64::from(w[r * cols + c])
                    })
                    .sum();
                let got = f64::from(f16::from_bits(y[b * rows + r]).to_f32());
                worst = worst.max((got - reference).abs());
                largest = largest.max(reference.abs());
            }
        }
        let relative = worst / largest.max(1e-9);
        let report = format!(
            "{} ({}): {rows}x{cols} Q4_0 as int4 + f16 scales, {batch} rows: compile {compile_ms:.0} ms, run {run_ms:.2} ms, largest error {worst:.4} = {:.3} % of the largest result {largest:.3}",
            self.device.name,
            self.device.full_name,
            relative * 100.0
        );
        if relative < 0.02 {
            Ok(report)
        } else {
            Err(format!("{report}: error above 2 %"))
        }
    }

    /// A line for the end of a run.
    pub fn summary(&self) -> String {
        let s = &self.stats;
        format!(
            "NPU offload: {} calls, {} rows on the NPU ({} ms: {} ms to f16, {} ms in the NPU call, {} ms back to f32), {} rows on the CPU, {} models compiled ({} ms, {} in the background), {} calls served by the CPU while a model was not ready, {} compile failures, {} f16 range fallbacks, ~{} MiB resident",
            s.npu_calls.load(Ordering::Relaxed),
            s.npu_rows.load(Ordering::Relaxed),
            s.npu_millis.load(Ordering::Relaxed),
            s.to_f16_micros.load(Ordering::Relaxed) / 1000,
            s.infer_micros.load(Ordering::Relaxed) / 1000,
            s.from_f16_micros.load(Ordering::Relaxed) / 1000,
            s.cpu_rows.load(Ordering::Relaxed),
            s.compiled.load(Ordering::Relaxed),
            s.compile_millis.load(Ordering::Relaxed),
            s.background_compiles.load(Ordering::Relaxed),
            s.waiting_calls.load(Ordering::Relaxed),
            s.compile_failures.load(Ordering::Relaxed),
            s.range_fallbacks.load(Ordering::Relaxed),
            self.resident_bytes() >> 20
        )
    }

    /// The state for the Studio card and `GET /api/studio/npu`: what the offload does right now.
    pub fn status_json(&self) -> serde_json::Value {
        let s = &self.stats;
        let mode = match self.compile_mode.load(Ordering::Relaxed) {
            MODE_INLINE => "inline",
            MODE_BACKGROUND => "background",
            _ => "not decided yet",
        };
        serde_json::json!({
            "engine": "openvino",
            "device": self.device.full_name,
            "platform": self.device.architecture,
            "driver": self.device.driver,
            "enabled": self.plan.enabled && !self.is_disabled(),
            "chunk": self.plan.chunk,
            "min_batch": self.plan.min_batch,
            "calls": s.npu_calls.load(Ordering::Relaxed),
            "rows": s.npu_rows.load(Ordering::Relaxed),
            "models_compiled": s.compiled.load(Ordering::Relaxed),
            "models_compiling": self.compiling(),
            "compile_mode": mode,
            "compile_failures": s.compile_failures.load(Ordering::Relaxed),
            "resident_mb": self.resident_bytes() >> 20,
            "budget_mb": self.budget >> 20,
            "npu_ms": s.npu_millis.load(Ordering::Relaxed),
            "to_f16_ms": s.to_f16_micros.load(Ordering::Relaxed) / 1000,
            "infer_ms": s.infer_micros.load(Ordering::Relaxed) / 1000,
            "from_f16_ms": s.from_f16_micros.load(Ordering::Relaxed) / 1000,
        })
    }
}

/// FNV-1a over 256 bytes at the start, the middle and the end of a tensor: cheap, and enough to tell two tensors at one address apart.
fn fingerprint(raw: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ raw.len() as u64;
    let n = raw.len();
    for start in [0, (n / 2).saturating_sub(128), n.saturating_sub(256)] {
        let start = start.min(n);
        for &byte in &raw[start..(start + 256).min(n)] {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The memory the compiled models may hold: the budget from the device and the user, and at most 40 % of the memory of the computer
/// that is free now (at least 1 GiB). `free == 0` means unknown.
fn planned_budget(budget: u64, free: u64) -> u64 {
    if free == 0 {
        return budget;
    }
    let share = (free / 5 * 2).max(1 << 30);
    if budget == 0 {
        share
    } else {
        budget.min(share)
    }
}

/// One thread per engine compiles the projections that were handed over (see `COLD_COMPILE_MS_PER_MIB`); it ends when the engine goes.
fn spawn_compile_thread(engine: &Arc<NpuEngine>) {
    let weak = Arc::downgrade(engine);
    let queue = Arc::clone(&engine.queue);
    let spawned = std::thread::Builder::new().name("fllm-npu-compile".to_owned()).spawn(move || loop {
        let job = {
            let Ok(mut jobs) = queue.jobs.lock() else { return };
            loop {
                if let Some(job) = jobs.pop_front() {
                    break job;
                }
                if weak.strong_count() == 0 {
                    return;
                }
                match queue.ready.wait_timeout(jobs, std::time::Duration::from_secs(2)) {
                    Ok((guard, _)) => jobs = guard,
                    Err(_) => return,
                }
            }
        };
        queue.pending_bytes.fetch_sub(job.raw.len() as u64, Ordering::Relaxed);
        let Some(engine) = weak.upgrade() else { return };
        engine.finish_job(job);
    });
    if let Err(error) = spawned {
        crate::backend::log_once("npu-compile-thread", &format!("no background compile thread ({error}); models are compiled while the prompt waits"));
        engine.compile_mode.store(MODE_INLINE, Ordering::Relaxed);
    }
}

/// Elements per rayon task of the f32 <-> f16 conversions.
const CONVERT_CHUNK: usize = 1 << 15;

/// `dst = f16(src)` (hardware conversion where the CPU has F16C / FP16); `false` when a result is Inf or NaN, which means the value
/// does not fit f16. `dst` and `src` must have the same length.
fn to_f16_checked(dst: &mut [u16], src: &[f32]) -> bool {
    debug_assert_eq!(dst.len(), src.len());
    dst.par_chunks_mut(CONVERT_CHUNK)
        .zip(src.par_chunks(CONVERT_CHUNK))
        .all(|(d, s)| {
            d.reinterpret_cast_mut::<f16>().convert_from_f32_slice(s);
            d.iter().all(|&bits| bits & 0x7C00 != 0x7C00)
        })
}

/// `dst = f32(src)`; `false` when a result is not finite. `dst` and `src` must have the same length.
fn from_f16_checked(dst: &mut [f32], src: &[u16]) -> bool {
    debug_assert_eq!(dst.len(), src.len());
    dst.par_chunks_mut(CONVERT_CHUNK)
        .zip(src.par_chunks(CONVERT_CHUNK))
        .all(|(d, s)| {
            s.reinterpret_cast::<f16>().convert_to_f32_slice(d);
            d.iter().all(|v| v.is_finite())
        })
}
/// Product with the engine's CPU kernels (the optimized prompt / decode kernels of the CPU path). `None` when the format has no such
/// kernel or the sizes do not match, so the caller can use the reference implementation instead. Never panics.
pub fn host_linear(
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    x: &[f32],
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    if rows == 0
        || cols == 0
        || x.is_empty()
        || !x.len().is_multiple_of(cols)
        || bias.is_some_and(|b| b.len() != rows)
    {
        return None;
    }
    let kind = quant.to_gguf_type_id();
    let expected = match quant {
        QuantType::F32 => rows * cols * 4,
        QuantType::F16 | QuantType::BF16 => rows * cols * 2,
        q if q.is_q4zp()
            || q.q4z_group().is_some()
            || q == QuantType::Fp8Marlin
            || matches!(q, QuantType::Other(_)) =>
        {
            return None
        }
        q => {
            let block = q.block_size();
            if block == 0 || !cols.is_multiple_of(block) {
                return None;
            }
            rows * (cols / block) * q.block_bytes()
        }
    };
    if raw.len() != expected {
        return None;
    }
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            crate::cpu_float::linear(raw, kind, x, rows, cols, bias)
        }
        _ => crate::cpu_quantized::linear(raw, kind, x, rows, cols, bias),
    }));
    attempt.ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_linear_matches_a_float_reference_for_f32_weights() {
        let (rows, cols, batch) = (5, 16, 3);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.05)
            .collect();
        let x: Vec<f32> = (0..batch * cols)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
        let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        // the float kernels may be unavailable on a scalar-only CPU; then None is the documented answer
        if let Some(got) = host_linear(&raw, QuantType::F32, rows, cols, &x, Some(&bias)) {
            for b in 0..batch {
                for r in 0..rows {
                    let want: f32 = (0..cols)
                        .map(|c| x[b * cols + c] * w[r * cols + c])
                        .sum::<f32>()
                        + bias[r];
                    assert!(
                        (got[b * rows + r] - want).abs() < 2e-2,
                        "row {b},{r}: {} vs {want}",
                        got[b * rows + r]
                    );
                }
            }
        }
    }

    #[test]
    fn host_linear_rejects_bad_sizes_without_panicking() {
        let x = vec![0.5f32; 64];
        assert!(host_linear(&[0u8; 17], QuantType::Q4_0, 1, 32, &x[..32], None).is_none());
        assert!(host_linear(&[0u8; 18], QuantType::Q4_0, 1, 33, &x[..33], None).is_none());
        assert!(host_linear(&[0u8; 18], QuantType::Q4_0, 1, 32, &x[..31], None).is_none());
        assert!(host_linear(
            &[0u8; 18],
            QuantType::Q4_0,
            1,
            32,
            &x[..32],
            Some(&[0.0, 1.0])
        )
        .is_none());
        assert!(host_linear(&[0u8; 8], QuantType::Other(77), 1, 4, &x[..4], None).is_none());
    }

    #[test]
    fn offline_engine_leaves_everything_to_the_cpu() {
        let engine = NpuEngine::offline(OffloadPlan::default());
        let raw = vec![0u8; 4096 * 4096 / 32 * 18];
        let x = vec![0.0f32; 256 * 4096];
        assert!(engine
            .offload_host(QuantType::Q4_0, &raw, 4096, 4096, &x, 256)
            .is_none());
        assert!(engine.is_disabled() || engine.resident_bytes() == 0);
        assert!(engine.summary().contains("0 calls"));
    }

    #[test]
    fn fingerprint_tells_tensors_at_one_address_apart() {
        let a = vec![7u8; 4096];
        let mut b = a.clone();
        assert_eq!(fingerprint(&a), fingerprint(&b));
        b[2048] = 8;
        assert_ne!(fingerprint(&a), fingerprint(&b), "a change in the middle is seen");
        b[2048] = 7;
        *b.last_mut().unwrap() = 9;
        assert_ne!(fingerprint(&a), fingerprint(&b), "a change at the end is seen");
        assert_ne!(fingerprint(&a[..4000]), fingerprint(&a), "the size is part of it");
        assert_eq!(fingerprint(&[]), fingerprint(&[]));
    }

    #[test]
    fn the_budget_follows_the_free_memory_of_the_computer() {
        // unknown free memory: the device budget stays
        assert_eq!(planned_budget(6 << 30, 0), 6 << 30);
        // 40 % of 10 GiB free is 4 GiB, below the device budget
        assert_eq!(planned_budget(12 << 30, 10 << 30), (10u64 << 30) / 5 * 2);
        // a nearly full computer still gets 1 GiB (the budget can only be lowered by the user)
        assert_eq!(planned_budget(12 << 30, 1 << 30), 1 << 30);
        assert_eq!(planned_budget(512 << 20, 64 << 30), 512 << 20);
        assert_eq!(planned_budget(0, 10 << 30), (10u64 << 30) / 5 * 2);
    }

    #[test]
    fn status_of_an_offline_engine_says_what_it_is() {
        let engine = NpuEngine::offline(OffloadPlan::default());
        let status = engine.status_json();
        assert_eq!(status["engine"], "openvino");
        assert_eq!(status["calls"], 0);
        assert_eq!(status["models_compiling"], 0);
        assert_eq!(status["chunk"], 256);
        assert_eq!(status["compile_mode"], "inline");
    }

    #[test]
    fn cache_dir_can_be_switched_off() {
        // Not touching the process environment (tests run in parallel): only the property parser is exercised.
        let props = user_props();
        assert!(props.len() <= 2);
    }

    #[test]
    fn f16_conversion_is_checked_in_both_directions() {
        let src: Vec<f32> = (0..100_000)
            .map(|i| (i % 2000) as f32 * 0.25 - 100.0)
            .collect();
        let mut half_bits = vec![0u16; src.len()];
        assert!(to_f16_checked(&mut half_bits, &src));
        for (&bits, &v) in half_bits.iter().zip(&src).step_by(997) {
            assert_eq!(f16::from_bits(bits), f16::from_f32(v));
        }
        let mut back = vec![0f32; src.len()];
        assert!(from_f16_checked(&mut back, &half_bits));
        assert!(back
            .iter()
            .zip(&src)
            .all(|(a, b)| (a - b).abs() <= b.abs() * 1e-3 + 1e-3));
        let mut too_big = src.clone();
        too_big[77_777] = 70_000.0;
        assert!(
            !to_f16_checked(&mut half_bits, &too_big),
            "overflow to Inf is detected"
        );
        too_big[77_777] = f32::NAN;
        assert!(!to_f16_checked(&mut half_bits, &too_big), "NaN is detected");
        half_bits[5] = 0x7C00;
        assert!(
            !from_f16_checked(&mut back, &half_bits),
            "Inf in the result is detected"
        );
    }
    #[test]
    fn weight_key_distinguishes_windows_of_one_buffer() {
        let a = WeightKey {
            id: 1,
            offset: 0,
            size: 64,
        };
        let b = WeightKey {
            id: 1,
            offset: 64,
            size: 64,
        };
        assert_ne!(a, b);
        let mut map = HashMap::new();
        map.insert(a, 1);
        map.insert(b, 2);
        assert_eq!(map[&a], 1);
    }
}
