//! The native XDNA engine: prompt projections `y = x . W^T` on the NPU, with our own GEMM kernel, no vendor inference library.
//!
//! * Weights stay quantized in the registry (the raw GGUF tensor, as the hybrid backend passes it at upload). On the first prompt that
//!   reaches the NPU a weight is dequantized to bf16 into buffers the NPU reads (resident, while the memory budget lasts) or, over the
//!   budget, again for every call (streamed). The dequantization is done on the CPU here ("in the shim", as the task puts it); an
//!   NPU-side dequantizing kernel is future work (`docs/npu-xdna.md`).
//! * Activations are rounded to bf16, products accumulate in f32 on the NPU, the output is f32. The depth is cut into chunks of 4096 or
//!   1024 (the instruction streams in [`super::kernels`]) whose partial sums are added on the CPU while the NPU computes the next tile.
//! * Only prompts with at least `FLLM_XDNA_MIN_ROWS` rows go to the NPU (decode is memory bound and stays on the CPU backend).
//!
//! Environment: `FLLM_XDNA=0` switches the backend off, `FLLM_XDNA_MIN_ROWS` (default 64), `FLLM_XDNA_MIN_ELEMS` (default 65536, smallest
//! weight offloaded), `FLLM_XDNA_MIN_EFF` (default 0.5, smallest share of useful work after zero padding to the tile sizes),
//! `FLLM_XDNA_RESIDENT_MB` (bf16 weight copies kept for the NPU; default 25 % of the free memory at start, 0 = stream every call).
//!
//! Verification level: L1 on one Ryzen AI 5 PRO 435 laptop under Linux (`docs/npu-xdna.md`).

use super::dequant;
use super::dqpack;
use super::kernels::{self, k_chunks, row_chunks, Variant, TILE_N, VARIANTS};
use super::shim::{Arg, Buf, Context, Run, Shim};
use crate::backend::npu::backend::hybrid::{LinearOffload, OffloadSlot};
use crate::backend::npu::openvino::engine::{NpuOutput, WeightKey};
use crate::backend::QuantType;
use half::{bf16, f16};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Settings of the offload decision (read from the environment once).
#[derive(Clone, Debug)]
pub struct Settings {
    pub enabled: bool,
    pub min_rows: usize,
    pub min_elems: usize,
    pub min_eff: f64,
    /// Most bytes of bf16 weight copies kept for the NPU (0 = none, every call re-packs).
    pub resident_budget: u64,
    /// Print the summary line every this many offloaded projections (`FLLM_XDNA_STATS`, 0 = never).
    pub stats_every: usize,
    /// `FLLM_XDNA_PRECISION`: `fast` (default: 4-bit weights as block floating point on the native bfp16 MAC), `exact` (bf16 products).
    pub weights: WeightMode,
    /// `FLLM_XDNA_SMALL=0` switches the 64-row program for small batches off.
    pub small: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightMode {
    Fast,
    Exact,
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

impl Settings {
    pub fn from_env() -> Self {
        let free = crate::devices::host_free_memory_bytes();
        let default_budget = free / 4;
        Settings {
            enabled: std::env::var("FLLM_XDNA").map(|v| !matches!(v.trim(), "0" | "off" | "false")).unwrap_or(true),
            min_rows: env_usize("FLLM_XDNA_MIN_ROWS").unwrap_or(64).max(1),
            min_elems: env_usize("FLLM_XDNA_MIN_ELEMS").unwrap_or(1 << 16),
            min_eff: std::env::var("FLLM_XDNA_MIN_EFF").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0.5),
            resident_budget: env_usize("FLLM_XDNA_RESIDENT_MB").map_or(default_budget, |mb| (mb as u64) << 20),
            stats_every: env_usize("FLLM_XDNA_STATS").unwrap_or(0),
            small: std::env::var("FLLM_XDNA_SMALL").map(|v| !matches!(v.trim(), "0" | "off" | "false")).unwrap_or(true),
            weights: match std::env::var("FLLM_XDNA_PRECISION").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
                "exact" | "bf16" => WeightMode::Exact,
                _ => WeightMode::Fast,
            },
        }
    }
}

/// Counters for the end-of-run summary.
#[derive(Default)]
pub struct Stats {
    pub calls: AtomicU64,
    pub rows: AtomicU64,
    pub cpu_rows: AtomicU64,
    pub kernel_runs: AtomicU64,
    pub npu_micros: AtomicU64,
    pub pack_micros: AtomicU64,
    /// Time filling the activation buffer (f32 -> bf16), waiting for the NPU, and adding the partial sums.
    pub fill_micros: AtomicU64,
    pub wait_micros: AtomicU64,
    pub acc_micros: AtomicU64,
    pub failures: AtomicU64,
}

enum Residency {
    Unset,
    Ready(Arc<Resident>),
    /// Over the budget (or packing failed): the tiles are packed again for every call.
    Streamed,
}

/// bf16 copy of a weight in NPU-readable buffers: tile `(jt, ci)` at `jt * chunks + ci`.
struct Resident {
    tiles: Vec<Buf>,
    bytes: u64,
    /// Which program reads these tiles: bf16 tiles, or 4-bit packed tiles (`dqpack`) for the dequantizing programs.
    kind: Kind,
}

struct Slot {
    rows: usize,
    cols: usize,
    quant: QuantType,
    raw: RawBytes,
    resident: Mutex<Residency>,
    /// Self check: use this program whatever the budget says.
    force: Option<Kind>,
}

/// The quantized tensor of a projection: a copy the engine owns (hybrid backend), or a view of memory that outlives the process's use of
/// it (the memory-mapped model file of the host decoder; the caller of [`XdnaEngine::offload_host`] promises that).
enum RawBytes {
    Owned(Arc<Vec<u8>>),
    Borrowed(*const u8, usize),
}

// SAFETY: a borrowed view is read-only memory of the mapped model that stays valid and unchanged for the life of the process.
unsafe impl Send for RawBytes {}
unsafe impl Sync for RawBytes {}

impl RawBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            RawBytes::Owned(v) => v,
            // SAFETY: see the type's documentation.
            RawBytes::Borrowed(ptr, len) => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

/// Buffers of one instruction stream.
struct VarBufs {
    instr: Buf,
    words: usize,
    a: Buf,
    c: [Buf; 2],
}

/// The NPU programs: bf16 weights (exact products), 4-bit weights expanded to bf16 in the tile (exact products), 4-bit weights
/// expanded to block floating point in the tile and multiplied on the native bfp16 MAC (fast, about 0.7 % of the largest result off).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bf16 = 0,
    Dq = 1,
    Dqb = 2,
    /// The block-floating-point program with 16-row tiles (64 rows per call) for small batches; reads the tiles of `Dqb`.
    DqbS = 3,
}

impl Kind {
    fn packed(self) -> bool {
        matches!(self, Kind::Dq | Kind::Dqb | Kind::DqbS)
    }

    fn variants(self) -> &'static [Variant] {
        match self {
            Kind::Bf16 => &VARIANTS,
            Kind::Dq => &kernels::DQ_VARIANTS,
            Kind::Dqb => &kernels::DQB_VARIANTS,
            Kind::DqbS => &kernels::DQBS_VARIANTS,
        }
    }

    fn xclbin(self) -> &'static [u8] {
        match self {
            Kind::Bf16 => kernels::XCLBIN,
            Kind::Dq => kernels::DQ_XCLBIN,
            Kind::Dqb => kernels::DQB_XCLBIN,
            Kind::DqbS => kernels::DQBS_XCLBIN,
        }
    }

    /// Rows per call of the largest and the smallest instruction streams.
    fn row_sizes(self) -> (usize, usize) {
        let v = self.variants();
        (v.iter().map(|x| x.m).max().unwrap_or(512), v.iter().map(|x| x.m).min().unwrap_or(128))
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Bf16 => "bf16 weights",
            Kind::Dq => "4-bit weights, exact products",
            Kind::Dqb => "4-bit weights, block floating point",
            Kind::DqbS => "4-bit weights, block floating point, small batches",
        }
    }
}

/// One loaded NPU program (an xclbin) with the buffers of its instruction streams.
struct Prog {
    ctx: Context,
    variants: &'static [Variant],
    vars: Vec<Option<VarBufs>>,
    scratch: Option<[Buf; 2]>,
}

struct Inner {
    /// Indexed by [`Kind`]; the bf16 program is always loaded, the others when the first weight needs them.
    progs: [Option<Prog>; 4],
    /// Loading or checking a program failed: its weights stay in the next best form.
    failed: [bool; 4],
}

impl Prog {
    fn new(ctx: Context, kind: Kind) -> Prog {
        let variants = kind.variants();
        Prog { ctx, variants, vars: (0..variants.len()).map(|_| None).collect(), scratch: None }
    }

    fn variant_index(&self, m: usize, k: usize) -> Option<usize> {
        self.variants.iter().position(|v| v.m == m && v.k == k)
    }

    fn ensure(&mut self, vi: usize) -> Result<(), String> {
        if self.vars[vi].is_some() {
            return Ok(());
        }
        let v = &self.variants[vi];
        let mut instr = self.ctx.alloc(Arg::Instructions, v.insts.len())?;
        instr.as_bytes_mut().copy_from_slice(v.insts);
        instr.to_device()?;
        let a = self.ctx.alloc(Arg::A, v.m * v.k * 2)?;
        let c = [self.ctx.alloc(Arg::C, v.m * TILE_N * 4)?, self.ctx.alloc(Arg::C, v.m * TILE_N * 4)?];
        self.vars[vi] = Some(VarBufs { instr, words: v.insts.len() / 4, a, c });
        Ok(())
    }
}

pub struct XdnaEngine {
    pub settings: Settings,
    pub device_name: String,
    shim: Arc<Shim>,
    index: usize,
    inner: Mutex<Inner>,
    slots: Mutex<HashMap<WeightKey, Arc<Slot>>>,
    resident: AtomicU64,
    resident_dq: AtomicU64,
    consecutive_failures: AtomicUsize,
    disabled: AtomicBool,
    pub stats: Stats,
}

/// Bytes of one row of `cols` values in format `quant`, or `None` when the engine cannot dequantize the format.
fn row_bytes(quant: QuantType, cols: usize) -> Option<usize> {
    match quant {
        QuantType::F32 => Some(cols * 4),
        QuantType::F16 | QuantType::BF16 => Some(cols * 2),
        q if q.is_q4zp() || q.q4z_group().is_some() || q == QuantType::Fp8Marlin || matches!(q, QuantType::Other(_)) => None,
        q => {
            let block = q.block_size();
            (block != 0 && cols.is_multiple_of(block)).then(|| cols / block * q.block_bytes())
        }
    }
}

/// Share of the work after padding `rows x cols` up to the tile sizes that is useful.
fn efficiency(rows: usize, cols: usize) -> f64 {
    let n_pad = rows.div_ceil(TILE_N) * TILE_N;
    let k_pad: usize = k_chunks(cols).iter().map(|c| c.1).sum();
    (rows * cols) as f64 / (n_pad * k_pad) as f64
}

impl XdnaEngine {
    /// Open the `index`-th NPU: load the shim, the program and create the context, then check one product against the CPU.
    pub fn open(index: usize) -> Result<Arc<XdnaEngine>, String> {
        let settings = Settings::from_env();
        if !settings.enabled {
            return Err("FLLM_XDNA=0 switches the native XDNA backend off".to_owned());
        }
        let shim = Shim::get()?;
        let count = shim.device_count()?;
        if index >= count {
            return Err(format!("XRT lists {count} NPU device(s), no device {index}"));
        }
        let ctx = shim.open_context(index, kernels::XCLBIN)?;
        let engine = Arc::new(XdnaEngine {
            settings,
            device_name: device_name(),
            shim: Arc::clone(shim),
            index,
            inner: Mutex::new(Inner { progs: [Some(Prog::new(ctx, Kind::Bf16)), None, None, None], failed: [false; 4] }),
            slots: Mutex::new(HashMap::new()),
            resident: AtomicU64::new(0),
            resident_dq: AtomicU64::new(0),
            consecutive_failures: AtomicUsize::new(0),
            disabled: AtomicBool::new(false),
            stats: Stats::default(),
        });
        engine.selfcheck()?;
        Ok(engine)
    }

    /// Runs one 128 x 1024 x 1024 product with random data on the NPU and compares it with the CPU (bf16 inputs, f32 sums). Fails when
    /// the largest error exceeds 1 % of the largest result: that is a wrong program, not rounding.
    pub fn selfcheck(&self) -> Result<(), String> {
        let (m, k, n) = (128usize, 1024usize, 1024usize);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let w: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let slot = Slot { rows: n, cols: k, quant: QuantType::F32, raw: RawBytes::Owned(Arc::new(raw)), resident: Mutex::new(Residency::Streamed), force: None };
        let got = self.gemm(&slot, &x, m, 1)?.ok_or("the self check was not accepted by the row plan")?;
        let xb: Vec<f32> = x.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let wb: Vec<f32> = w.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let mut worst = 0f32;
        let mut peak = 0f32;
        for i in (0..m).step_by(7) {
            for j in (0..n).step_by(5) {
                let reference: f32 = xb[i * k..(i + 1) * k].iter().zip(&wb[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
                worst = worst.max((reference - got.1[i * n + j]).abs());
                peak = peak.max(reference.abs());
            }
        }
        if worst > 0.01 * peak.max(1e-6) {
            return Err(format!("NPU self check failed: largest error {worst} against results up to {peak}"));
        }
        Ok(())
    }

    /// One-line state for logs and status.
    pub fn summary(&self) -> String {
        let s = &self.stats;
        format!(
            "native XDNA NPU: {} projections offloaded ({} rows, {} CPU rows), {} kernel runs, {:.2} s in the offload (of which {:.2} s waiting for the NPU, {:.2} s packing weights, {:.2} s filling activations, {:.2} s adding partial sums), resident weights {} MiB (4-bit packed: {} MiB){}",
            s.calls.load(Ordering::Relaxed),
            s.rows.load(Ordering::Relaxed),
            s.cpu_rows.load(Ordering::Relaxed),
            s.kernel_runs.load(Ordering::Relaxed),
            s.npu_micros.load(Ordering::Relaxed) as f64 / 1e6,
            s.wait_micros.load(Ordering::Relaxed) as f64 / 1e6,
            s.pack_micros.load(Ordering::Relaxed) as f64 / 1e6,
            s.fill_micros.load(Ordering::Relaxed) as f64 / 1e6,
            s.acc_micros.load(Ordering::Relaxed) as f64 / 1e6,
            self.resident.load(Ordering::Relaxed) >> 20,
            self.resident_dq.load(Ordering::Relaxed) >> 20,
            if self.disabled.load(Ordering::Relaxed) { ", switched off after repeated errors" } else { "" }
        )
    }

    /// Dequantize rows `n0..n1` of a weight to f32 (`(n1 - n0) x cols`, row major).
    fn dequant_rows(slot: &Slot, n0: usize, n1: usize) -> Result<Vec<f32>, String> {
        let rb = row_bytes(slot.quant, slot.cols).ok_or("format not supported")?;
        let kind = slot.quant.to_gguf_type_id();
        let cols = slot.cols;
        let mut out = vec![0f32; (n1 - n0) * cols];
        const GROUP: usize = 16;
        out.par_chunks_mut(GROUP * cols).enumerate().try_for_each(|(g, dst)| -> Result<(), String> {
            let r0 = n0 + g * GROUP;
            let nr = dst.len() / cols;
            let values = crate::compute::dequantize(&slot.raw.bytes()[r0 * rb..(r0 + nr) * rb], kind, nr * cols)?;
            if values.len() != dst.len() {
                return Err(format!("dequantize returned {} values, expected {}", values.len(), dst.len()));
            }
            dst.copy_from_slice(&values);
            Ok(())
        })?;
        Ok(out)
    }

    /// Pack N-tile `jt` of a weight into bf16 chunk buffers (`dst[ci]` is `TILE_N x chunk depth`, zero padded).
    fn pack_tile(slot: &Slot, jt: usize, chunks: &[(usize, usize)], dst: &mut [&mut [u16]]) -> Result<(), String> {
        let n0 = jt * TILE_N;
        let n1 = (n0 + TILE_N).min(slot.rows);
        let cols = slot.cols;
        let kind = slot.quant.to_gguf_type_id();
        if let (Some((block, bytes)), Some(rb)) = (dequant::layout(kind), row_bytes(slot.quant, cols)) {
            // the formats of common models are decoded straight to bf16
            let raw = slot.raw.bytes();
            for (ci, &(k0, kt)) in chunks.iter().enumerate() {
                let valid = kt.min(cols.saturating_sub(k0));
                let (off, len) = (k0 / block * bytes, valid / block * bytes);
                dst[ci].par_chunks_mut(kt).enumerate().for_each(|(r, row)| {
                    if r < n1 - n0 {
                        let line = &raw[(n0 + r) * rb..(n0 + r + 1) * rb];
                        dequant::decode(kind, &line[off..off + len], &mut row[..valid]);
                        row[valid..].fill(0);
                    } else {
                        row.fill(0);
                    }
                });
            }
            return Ok(());
        }
        let src = Self::dequant_rows(slot, n0, n1)?;
        for (ci, &(k0, kt)) in chunks.iter().enumerate() {
            let valid = kt.min(cols.saturating_sub(k0));
            dst[ci].par_chunks_mut(kt).enumerate().for_each(|(r, row)| {
                if r < n1 - n0 {
                    let s = &src[r * cols + k0..r * cols + k0 + valid];
                    for (d, v) in row.iter_mut().zip(s) {
                        *d = bf16::from_f32(*v).to_bits();
                    }
                    row[valid..].fill(0);
                } else {
                    row.fill(0);
                }
            });
        }
        Ok(())
    }

    /// Pack N-tile `jt`, depth chunk `(k0, kt)` into its half of the scratch pair and make it visible to the NPU.
    fn pack_into_scratch(prog: &mut Prog, slot: &Slot, jt: usize, k0: usize, kt: usize) -> Result<(), String> {
        let scratch = prog.scratch.as_mut().ok_or("scratch missing")?;
        let buf = &mut scratch[jt % 2];
        Self::pack_tile(slot, jt, &[(k0, kt)], &mut [&mut buf.as_u16_mut()[..TILE_N * kt]])?;
        buf.to_device()
    }

    /// Load a program on first use and check it against the CPU; `false` when it cannot be used.
    fn ensure_prog(&self, inner: &mut Inner, kind: Kind) -> bool {
        let i = kind as usize;
        if inner.progs[i].is_some() {
            return true;
        }
        if inner.failed[i] {
            return false;
        }
        match self.shim.open_context(self.index, kind.xclbin()) {
            Ok(ctx) => inner.progs[i] = Some(Prog::new(ctx, kind)),
            Err(error) => {
                inner.failed[i] = true;
                crate::backend::log_once(&format!("xdna-prog-{i}"), &format!("native XDNA: the program for {} did not load: {error}", kind.name()));
                return false;
            }
        }
        match self.packed_selfcheck(inner, kind) {
            Ok(()) => true,
            Err(error) => {
                inner.progs[i] = None;
                inner.failed[i] = true;
                crate::backend::log_once(&format!("xdna-prog-{i}"), &format!("native XDNA: the program for {} failed its check: {error}", kind.name()));
                false
            }
        }
    }

    /// One product with random Q4_0 weights through a packed-weight program against the CPU on the dequantized values.
    fn packed_selfcheck(&self, inner: &mut Inner, kind: Kind) -> Result<(), String> {
        let (m, k, n) = (kind.row_sizes().1, 1024usize, 1024usize);
        let mut seed = 0x1234_5678_9abc_def1_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 32 * 18);
        for _ in 0..n * k / 32 {
            raw.extend_from_slice(&f16::from_f32(0.01 + (next() % 100) as f32 / 4000.0).to_le_bytes());
            raw.extend((0..16).map(|_| next() as u8));
        }
        let x: Vec<f32> = (0..m * k).map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
        let slot = Slot {
            rows: n,
            cols: k,
            quant: QuantType::Q4_0,
            raw: RawBytes::Owned(Arc::new(raw.clone())),
            resident: Mutex::new(Residency::Unset),
            force: Some(kind),
        };
        let got = self.gemm_locked(inner, &slot, &x, m, 1)?.ok_or("the check was not accepted by the row plan")?;
        let w = crate::compute::dequantize(&raw, 2, n * k)?;
        let xb: Vec<f32> = x.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
        let (mut worst, mut peak) = (0f32, 0f32);
        for i in (0..m).step_by(9) {
            for j in (0..n).step_by(7) {
                let reference: f32 = xb[i * k..(i + 1) * k].iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * bf16::from_f32(*b).to_f32()).sum();
                worst = worst.max((reference - got.1[i * n + j]).abs());
                peak = peak.max(reference.abs());
            }
        }
        if let Ok(state) = slot.resident.lock() {
            if let Residency::Ready(r) = &*state {
                self.resident.fetch_sub(r.bytes.min(self.resident.load(Ordering::Relaxed)), Ordering::Relaxed);
                self.resident_dq.fetch_sub(r.bytes.min(self.resident_dq.load(Ordering::Relaxed)), Ordering::Relaxed);
            }
        }
        // block floating point carries 8-bit mantissas: about 0.75 % of the largest result
        let allowed = if matches!(kind, Kind::Dqb | Kind::DqbS) { 0.03 } else { 0.01 };
        if worst > allowed * peak.max(1e-6) {
            return Err(format!("largest error {worst} against results up to {peak}"));
        }
        Ok(())
    }

    /// The resident copy of a weight, built on first use, and the program that reads it: 4-bit packed tiles for the formats `dqpack`
    /// reads (block floating point program by default, the exact-product one with `FLLM_XDNA_PRECISION=exact` once the bf16 budget is
    /// used), else bf16 tiles while the budget lasts; `None` when the weight is streamed (bf16 program, packed again for every call).
    fn resident_for(&self, inner: &mut Inner, slot: &Slot) -> Result<(Option<Arc<Resident>>, Kind), String> {
        let mut state = slot.resident.lock().map_err(|_| "poisoned")?;
        match &*state {
            Residency::Ready(r) => return Ok((Some(Arc::clone(r)), r.kind)),
            Residency::Streamed => return Ok((None, Kind::Bf16)),
            Residency::Unset => {}
        }
        let chunks = k_chunks(slot.cols);
        let ntiles = slot.rows.div_ceil(TILE_N);
        let bf16_bytes: u64 = chunks.iter().map(|c| (TILE_N * c.1 * 2) as u64).sum::<u64>() * ntiles as u64;
        let kind_id = slot.quant.to_gguf_type_id();
        let packable = dqpack::supported(kind_id) && dqpack::row_bytes(kind_id, slot.cols).is_some();
        let packed_bytes: u64 = chunks.iter().map(|c| (c.1 / dqpack::TILE_K * dqpack::NTILES * dqpack::PAYLOAD) as u64).sum::<u64>() * ntiles as u64;
        let budget = self.settings.resident_budget;
        let used = self.resident.load(Ordering::Relaxed);
        let mut kind = Kind::Bf16;
        if packable {
            let wanted = match (slot.force, self.settings.weights) {
                (Some(k), _) => Some(k),
                (None, WeightMode::Fast) if used + packed_bytes <= budget => Some(Kind::Dqb),
                (None, WeightMode::Exact) if used + bf16_bytes > budget && used + packed_bytes <= budget => Some(Kind::Dq),
                _ => None,
            };
            if let Some(k) = wanted {
                if self.ensure_prog(inner, k) {
                    kind = k;
                }
            }
        }
        if kind == Kind::Bf16 && used + bf16_bytes > budget {
            *state = Residency::Streamed;
            crate::backend::log_once(
                "xdna-budget",
                &format!(
                    "native XDNA: the weight budget of {} MiB is used up; further weights are packed again on every call (FLLM_XDNA_RESIDENT_MB)",
                    budget >> 20
                ),
            );
            return Ok((None, Kind::Bf16));
        }
        let started = Instant::now();
        let mut tiles = Vec::with_capacity(ntiles * chunks.len());
        if kind.packed() {
            let prog = inner.progs[kind as usize].as_mut().ok_or("program missing")?;
            let raw = slot.raw.bytes();
            for jt in 0..ntiles {
                for &(k0, kt) in &chunks {
                    let mut buf = prog.ctx.alloc(Arg::B, kt / dqpack::TILE_K * dqpack::NTILES * dqpack::PAYLOAD)?;
                    dqpack::pack_group(kind_id, raw, slot.rows, slot.cols, jt * TILE_N, k0, kt, buf.as_bytes_mut());
                    buf.to_device()?;
                    tiles.push(buf);
                }
            }
        } else {
            let bf16 = inner.progs[0].as_mut().ok_or("program missing")?;
            for _ in 0..ntiles {
                for &(_, kt) in &chunks {
                    tiles.push(bf16.ctx.alloc(Arg::B, TILE_N * kt * 2)?);
                }
            }
            for jt in 0..ntiles {
                let group = &mut tiles[jt * chunks.len()..(jt + 1) * chunks.len()];
                let mut dst: Vec<&mut [u16]> = group.iter_mut().map(|b| b.as_u16_mut()).collect();
                Self::pack_tile(slot, jt, &chunks, &mut dst)?;
                for b in group.iter() {
                    b.to_device()?;
                }
            }
        }
        self.stats.pack_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        let bytes = if kind.packed() { packed_bytes } else { bf16_bytes };
        self.resident.fetch_add(bytes, Ordering::Relaxed);
        if kind.packed() {
            self.resident_dq.fetch_add(bytes, Ordering::Relaxed);
        }
        let resident = Arc::new(Resident { tiles, bytes, kind });
        *state = Residency::Ready(Arc::clone(&resident));
        Ok((Some(resident), kind))
    }

    /// The product of the first rows of `x` with the weight. `Ok(None)` when the row plan takes nothing; otherwise `(rows done, values)`.
    fn gemm(&self, slot: &Slot, x: &[f32], batch: usize, min_rows: usize) -> Result<Option<(usize, Vec<f32>)>, String> {
        let mut guard = self.inner.lock().map_err(|_| "poisoned")?;
        self.gemm_locked(&mut guard, slot, x, batch, min_rows)
    }

    fn gemm_locked(&self, inner: &mut Inner, slot: &Slot, x: &[f32], batch: usize, min_rows: usize) -> Result<Option<(usize, Vec<f32>)>, String> {
        let (n, k) = (slot.rows, slot.cols);
        let (resident, kind) = self.resident_for(inner, slot)?;
        // a small batch of 4-bit tiles runs on the 64-row program
        let kind = if kind == Kind::Dqb && batch <= 64 && self.settings.small && self.ensure_prog(inner, Kind::DqbS) { Kind::DqbS } else { kind };
        let (big, small) = kind.row_sizes();
        let plan = row_chunks(batch, min_rows, big, small);
        let Some(&(last0, lastn, _)) = plan.last() else {
            return Ok(None);
        };
        let take = last0 + lastn;
        let chunks = k_chunks(k);
        let ntiles = n.div_ceil(TILE_N);
        let prog: &mut Prog = inner.progs[kind as usize].as_mut().ok_or("program missing")?;
        let started = Instant::now();
        let (mut pack_micros, mut fill_micros, mut wait_micros, mut acc_micros) = (0u64, 0u64, 0u64, 0u64);
        let mut runs = 0u64;
        let mut out = vec![0f32; take * n];
        for &(r0, rn, m) in &plan {
            for (ci, &(k0, kt)) in chunks.iter().enumerate() {
                let vi = prog.variant_index(m, kt).ok_or("no instruction stream for this shape")?;
                prog.ensure(vi)?;
                let valid = kt.min(k - k0);
                let started_fill = Instant::now();
                {
                    let vb = prog.vars[vi].as_mut().ok_or("buffers missing")?;
                    vb.a.as_u16_mut().par_chunks_mut(kt).enumerate().for_each(|(i, row)| {
                        if i < rn {
                            let s = &x[(r0 + i) * k + k0..(r0 + i) * k + k0 + valid];
                            for (d, v) in row.iter_mut().zip(s) {
                                *d = bf16::from_f32(*v).to_bits();
                            }
                            row[valid..].fill(0);
                        } else {
                            row.fill(0);
                        }
                    });
                    vb.a.to_device()?;
                }
                fill_micros += started_fill.elapsed().as_micros() as u64;
                let streaming = resident.is_none();
                if streaming {
                    if prog.scratch.is_none() {
                        prog.scratch = Some([prog.ctx.alloc(Arg::B, TILE_N * 4096 * 2)?, prog.ctx.alloc(Arg::B, TILE_N * 4096 * 2)?]);
                    }
                    // tile 0 is packed before the loop, tile jt + 1 while the NPU computes tile jt
                    let started_pack = Instant::now();
                    Self::pack_into_scratch(prog, slot, 0, k0, kt)?;
                    pack_micros += started_pack.elapsed().as_micros() as u64;
                }
                let mut pending: Option<(Run, usize, usize)> = None;
                for jt in 0..ntiles {
                    let n0 = jt * TILE_N;
                    let cidx = jt % 2;
                    let b_ref: &Buf = match &resident {
                        Some(r) => &r.tiles[jt * chunks.len() + ci],
                        None => &prog.scratch.as_ref().ok_or("scratch missing")?[jt % 2],
                    };
                    let vb = prog.vars[vi].as_ref().ok_or("buffers missing")?;
                    let run = prog.ctx.start(&vb.instr, vb.words, &vb.a, b_ref, &vb.c[cidx])?;
                    runs += 1;
                    if let Some((prev, prev_c, prev_n0)) = pending.replace((run, cidx, n0)) {
                        let started_wait = Instant::now();
                        prev.wait(30_000)?;
                        wait_micros += started_wait.elapsed().as_micros() as u64;
                        let started_acc = Instant::now();
                        accumulate(&mut out, n, r0, rn, prev_n0, &vb.c[prev_c], TILE_N.min(n - prev_n0), ci == 0);
                        acc_micros += started_acc.elapsed().as_micros() as u64;
                    }
                    if streaming && jt + 1 < ntiles {
                        // the tile before this one has finished (waited just above), so its half of the scratch pair is free
                        let started_pack = Instant::now();
                        Self::pack_into_scratch(prog, slot, jt + 1, k0, kt)?;
                        pack_micros += started_pack.elapsed().as_micros() as u64;
                    }
                }
                if let Some((run, c_idx, n0)) = pending.take() {
                    let started_wait = Instant::now();
                    run.wait(30_000)?;
                    wait_micros += started_wait.elapsed().as_micros() as u64;
                    let started_acc = Instant::now();
                    let vb = prog.vars[vi].as_ref().ok_or("buffers missing")?;
                    accumulate(&mut out, n, r0, rn, n0, &vb.c[c_idx], TILE_N.min(n - n0), ci == 0);
                    acc_micros += started_acc.elapsed().as_micros() as u64;
                }
            }
        }
        self.stats.fill_micros.fetch_add(fill_micros, Ordering::Relaxed);
        self.stats.wait_micros.fetch_add(wait_micros, Ordering::Relaxed);
        self.stats.acc_micros.fetch_add(acc_micros, Ordering::Relaxed);
        self.stats.kernel_runs.fetch_add(runs, Ordering::Relaxed);
        self.stats.pack_micros.fetch_add(pack_micros, Ordering::Relaxed);
        self.stats.npu_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        Ok(Some((take, out)))
    }
}

/// Add the f32 tile in `c` (row stride `TILE_N`; the first `rn` rows, `width` columns) to `out` at rows `r0..`, columns `n0..`.
/// `c` is read after `from_device`.
#[allow(clippy::too_many_arguments)]
fn accumulate(out: &mut [f32], n: usize, r0: usize, rn: usize, n0: usize, c: &Buf, width: usize, assign: bool) {
    if c.from_device().is_err() {
        return;
    }
    let tile = c.as_f32();
    out[r0 * n..(r0 + rn) * n].par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let src = &tile[i * TILE_N..i * TILE_N + width];
        if assign {
            row[n0..n0 + width].copy_from_slice(src);
        } else {
            for (d, s) in row[n0..n0 + width].iter_mut().zip(src) {
                *d += *s;
            }
        }
    });
}

/// Name for the inventory: the NPU the operating system lists, else a generic one.
fn device_name() -> String {
    crate::backend::npu::probe::detect()
        .into_iter()
        .find(|d| d.vendor == crate::backend::npu::probe::Vendor::Amd)
        .map(|d| d.name)
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "AMD XDNA NPU".to_owned())
}

impl LinearOffload for XdnaEngine {
    fn label(&self) -> String {
        format!("{}, native XDNA kernels", self.device_name)
    }

    fn register(&self, key: WeightKey, quant: QuantType, rows: usize, cols: usize, raw: &[u8]) {
        if rows * cols < self.settings.min_elems
            || row_bytes(quant, cols).is_none_or(|rb| rb * rows != raw.len())
            || efficiency(rows, cols) < self.settings.min_eff
        {
            return;
        }
        let slot = Arc::new(Slot { rows, cols, quant, raw: RawBytes::Owned(Arc::new(raw.to_vec())), resident: Mutex::new(Residency::Unset), force: None });
        if let Ok(mut slots) = self.slots.lock() {
            slots.insert(key, slot);
        }
    }

    fn unregister(&self, key: &WeightKey) {
        let removed = self.slots.lock().ok().and_then(|mut s| s.remove(key));
        if let Some(slot) = removed {
            if let Ok(state) = slot.resident.lock() {
                if let Residency::Ready(r) = &*state {
                    self.resident.fetch_sub(r.bytes.min(self.resident.load(Ordering::Relaxed)), Ordering::Relaxed);
                }
            }
        }
    }

    fn slot(&self, key: &WeightKey) -> Option<OffloadSlot> {
        let slot = self.slots.lock().ok()?.get(key).cloned()?;
        let host = match &slot.raw {
            RawBytes::Owned(v) => Some(Arc::clone(v)),
            RawBytes::Borrowed(..) => None,
        };
        Some(OffloadSlot { rows: slot.rows, cols: slot.cols, quant: slot.quant, host })
    }

    fn run(&self, key: &WeightKey, _slot: &OffloadSlot, x: &[f32], batch: usize) -> Option<NpuOutput> {
        let slot = self.slots.lock().ok()?.get(key).cloned()?;
        self.run_slot(&slot, x, batch)
    }

    fn note_cpu_rows(&self, rows: u64) {
        self.stats.cpu_rows.fetch_add(rows, Ordering::Relaxed);
    }

    fn summary(&self) -> Option<String> {
        (self.stats.calls.load(Ordering::Relaxed) > 0).then(|| XdnaEngine::summary(self))
    }
}

impl XdnaEngine {
    /// The NPU part of a projection call: `None` when the engine takes nothing (switched off, paused, too few rows, an error: the CPU
    /// computes everything), otherwise the first `rows_done` rows without bias.
    fn run_slot(&self, slot: &Slot, x: &[f32], batch: usize) -> Option<NpuOutput> {
        if self.disabled.load(Ordering::Relaxed) || super::offload_paused() || batch < self.settings.min_rows || x.len() != batch * slot.cols {
            return None;
        }
        match self.gemm(slot, x, batch, self.settings.min_rows) {
            Ok(Some((rows_done, values))) => {
                self.consecutive_failures.store(0, Ordering::Relaxed);
                let calls = self.stats.calls.fetch_add(1, Ordering::Relaxed) + 1;
                if self.settings.stats_every > 0 && calls.is_multiple_of(self.settings.stats_every as u64) {
                    eprintln!("[fllm-rs] {}", self.summary());
                }
                self.stats.rows.fetch_add(rows_done as u64, Ordering::Relaxed);
                Some(NpuOutput { rows_done, values })
            }
            Ok(None) => None,
            Err(error) => {
                self.stats.failures.fetch_add(1, Ordering::Relaxed);
                crate::backend::log_once("xdna-run-error", &format!("native XDNA run failed, the CPU takes over: {error}"));
                if self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1 >= 3 {
                    self.disabled.store(true, Ordering::Relaxed);
                }
                None
            }
        }
    }

    /// Offload for a host decoder that keeps the quantized tensors itself (memory-mapped model file): `raw` is the tensor of the
    /// projection `rows x cols` in format `quant`, `x` holds `batch` rows. The projection is identified by the address of `raw`, so the
    /// slice must stay at the same place and unchanged for the life of the process (a mapped model file does). Nothing is copied except
    /// into the bf16 NPU buffers. `None` when the NPU takes nothing; otherwise the caller computes rows `rows_done..batch` and the bias.
    pub fn offload_host(&self, quant: QuantType, raw: &[u8], rows: usize, cols: usize, x: &[f32], batch: usize) -> Option<NpuOutput> {
        if self.disabled.load(Ordering::Relaxed) || batch < self.settings.min_rows {
            return None;
        }
        let key = WeightKey { id: raw.as_ptr() as usize as u64, offset: 0, size: raw.len() };
        let slot = {
            let mut slots = self.slots.lock().ok()?;
            if let Some(slot) = slots.get(&key) {
                Arc::clone(slot)
            } else {
                if rows * cols < self.settings.min_elems
                    || row_bytes(quant, cols).is_none_or(|rb| rb * rows != raw.len())
                    || efficiency(rows, cols) < self.settings.min_eff
                {
                    return None;
                }
                let slot = Arc::new(Slot { rows, cols, quant, raw: RawBytes::Borrowed(raw.as_ptr(), raw.len()), resident: Mutex::new(Residency::Unset), force: None });
                slots.insert(key, Arc::clone(&slot));
                slot
            }
        };
        if slot.rows != rows || slot.cols != cols || slot.quant != quant {
            return None;
        }
        self.run_slot(&slot, x, batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_sizes_follow_the_block_layouts_and_unknown_formats_are_refused() {
        assert_eq!(row_bytes(QuantType::Q4_K, 512), Some(2 * 144));
        assert_eq!(row_bytes(QuantType::Q8_0, 64), Some(2 * 34));
        assert_eq!(row_bytes(QuantType::Q6_K, 256), Some(210));
        assert_eq!(row_bytes(QuantType::F16, 10), Some(20));
        assert_eq!(row_bytes(QuantType::F32, 10), Some(40));
        assert_eq!(row_bytes(QuantType::Q4_K, 300), None, "a row must hold whole blocks");
        assert_eq!(row_bytes(QuantType::Other(999), 32), None);
        assert_eq!(row_bytes(QuantType::Q4Z32, 64), None);
    }

    #[test]
    fn padding_efficiency_prefers_multiples_of_the_tile_sizes() {
        assert!((efficiency(3072, 3072) - 1.0).abs() < 1e-9);
        assert!((efficiency(8192, 11008) - 11008.0 / 11264.0).abs() < 1e-9);
        assert!(efficiency(576, 576) < 0.4, "a small model dimension wastes most of a tile");
        assert!(efficiency(1536, 8960) > 0.5);
    }

    #[test]
    fn raw_views_read_the_bytes_they_point_to() {
        let data = [1u8, 2, 3, 4];
        let view = RawBytes::Borrowed(data.as_ptr(), data.len());
        assert_eq!(view.bytes(), &[1, 2, 3, 4]);
        assert_eq!(RawBytes::Owned(Arc::new(vec![9u8])).bytes(), &[9]);
    }

    /// Needs an AMD NPU with XRT and the shim: `cargo test --release -p fllm-engine -- --ignored xdna_hardware`.
    #[test]
    #[ignore]
    fn xdna_hardware_selfcheck_and_a_quantized_projection() {
        let report = crate::backend::npu::xdna::native::bench::run(2048, 2048, 256, "q8_0", 2).expect("the NPU run");
        eprintln!("{report}");
        assert!(report.contains("rows on the NPU"));
    }
}
