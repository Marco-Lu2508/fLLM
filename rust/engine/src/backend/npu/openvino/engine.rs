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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
}

impl Slot {
    /// The raw tensor as uploaded, if this slot keeps a copy.
    pub fn host_bytes(&self) -> Option<&[u8]> {
        self.host.as_deref().map(Vec::as_slice)
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
}

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
            stats: Stats::default(),
        };
        crate::backend::log_once("npu-plan", &engine.plan.describe());
        Ok(Arc::new(engine))
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
        let slot = Arc::new(Slot {
            rows,
            cols,
            quant,
            host: Some(Arc::new(raw.to_vec())),
            state: Mutex::new(SlotState::Idle),
        });
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
                    let bytes = self.plan.resident_bytes(slot.rows, slot.cols, slot.quant);
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

    fn compile(&self, slot: &Slot, raw: &[u8]) -> Result<Arc<CompiledLinear>, String> {
        let weight = convert(slot.quant, raw, slot.rows, slot.cols, &self.plan.quant)?;
        let bytes = weight.bytes() as u64;
        if self.budget > 0 && self.resident.load(Ordering::Relaxed) + bytes > self.budget {
            return Err(format!(
                "NPU weight budget of {} MiB is used up",
                self.budget >> 20
            ));
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
        crate::backend::log_once(
            "npu-first-compile",
            &format!(
                "first NPU model ({name}, {:?}, chunk {}) compiled in {elapsed} ms",
                weight.format, self.plan.chunk
            ),
        );
        Ok(Arc::new(compiled))
    }

    /// The compiled model of a slot, compiling it on first use; `None` when it cannot be served by the NPU.
    fn session(&self, slot: &Slot, raw: &[u8]) -> Option<Arc<CompiledLinear>> {
        if self.is_disabled() {
            return None;
        }
        let mut state = slot.state.lock().ok()?;
        match &*state {
            SlotState::Ready(session) => return Some(Arc::clone(session)),
            SlotState::Failed => return None,
            SlotState::Idle => {}
        }
        match self.compile(slot, raw) {
            Ok(session) => {
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

    /// Run the rows of `x` (`batch * slot.cols` values) that the plan assigns to the NPU through the projection `slot`.
    ///
    /// `None` when the NPU takes nothing (plan, compile failure, f16 range): the caller computes all rows itself. Otherwise the
    /// result holds the first `rows_done` rows (without bias; the caller adds it) and the caller computes rows `rows_done..batch`.
    pub fn run(&self, slot: &Slot, x: &[f32], batch: usize) -> Option<NpuOutput> {
        let raw = slot.host_bytes()?;
        self.run_raw(slot, raw, x, batch)
    }

    /// [`run`](Self::run) for a caller that supplies the raw tensor itself (needed only for the first call, when the model is built).
    pub fn run_raw(&self, slot: &Slot, raw: &[u8], x: &[f32], batch: usize) -> Option<NpuOutput> {
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
        while done < take_rows {
            let take = chunk.min(take_rows - done);
            // f16 has a range of 65504 and the NPU computes in f16: activations that do not fit are left to the CPU
            if !to_f16_checked(&mut x16[..take * k], &input[done * k..(done + take) * k]) {
                self.stats.range_fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            x16[take * k..].fill(0);
            if let Err(error) = session.run(&x16, &mut y16) {
                crate::backend::log_once(
                    "npu-run-error",
                    &format!("NPU inference failed, the CPU takes over: {error}"),
                );
                return None;
            }
            // an f16 overflow of a sum shows up as Inf: redo the call on the CPU
            if !from_f16_checked(&mut values[done * n..(done + take) * n], &y16[..take * n]) {
                self.stats.range_fallbacks.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            done += take;
        }
        self.stats.npu_calls.fetch_add(1, Ordering::Relaxed);
        self.stats
            .npu_rows
            .fetch_add(take_rows as u64, Ordering::Relaxed);
        self.stats
            .npu_millis
            .fetch_add(started.elapsed().as_millis() as u64, Ordering::Relaxed);
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
        let key = WeightKey {
            id: raw.as_ptr() as usize as u64,
            offset: 0,
            size: raw.len(),
        };
        let slot = {
            let mut slots = self.slots.lock().ok()?;
            Arc::clone(slots.entry(key).or_insert_with(|| {
                Arc::new(Slot {
                    rows,
                    cols,
                    quant,
                    host: None,
                    state: Mutex::new(SlotState::Idle),
                })
            }))
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
            "NPU offload: {} calls, {} rows on the NPU ({} ms), {} rows on the CPU, {} models compiled ({} ms), {} compile failures, {} f16 range fallbacks, ~{} MiB resident",
            s.npu_calls.load(Ordering::Relaxed),
            s.npu_rows.load(Ordering::Relaxed),
            s.npu_millis.load(Ordering::Relaxed),
            s.cpu_rows.load(Ordering::Relaxed),
            s.compiled.load(Ordering::Relaxed),
            s.compile_millis.load(Ordering::Relaxed),
            s.compile_failures.load(Ordering::Relaxed),
            s.range_fallbacks.load(Ordering::Relaxed),
            self.resident_bytes() >> 20
        )
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
