//! Development aid `FLLM_VK_BENCH_LINEAR=1`: times the quantized projections of a few typical shapes once per process, at the first projection, and
//! logs milliseconds, multiply-adds per second and weight bytes per second. Weights are random bytes (not zeros, which run cooler on a power-limited
//! GPU), several copies rotate so that a small cache cannot hold them. Not used in normal operation.

use super::linear::LinearWeight;
use super::VulkanStream;
use crate::backend::vulkan::backend::mem::{slice_table, WeightKind};
use crate::backend::{log_once, BackendError, QuantType};
use std::sync::atomic::{AtomicBool, Ordering};

static CLAIMED: AtomicBool = AtomicBool::new(false);

/// `(gguf type id, name, rows, cols)` of the benchmarked weights: the shapes of a 3B-class model's feed-forward and attention projections.
const SHAPES: &[(u32, &str, usize, usize)] = &[
    (12, "Q4_K", 8192, 3072),
    (12, "Q4_K", 3072, 3072),
    (14, "Q6_K", 3072, 8192),
    (2, "Q4_0", 8192, 3072),
    (8, "Q8_0", 3072, 3072),
    // the shape of a vocabulary projection
    (12, "Q4_K", 32768, 3072),
    (14, "Q6_K", 32768, 3072),
    (12, "Q4_K", 128256, 3072),
    (14, "Q6_K", 128256, 3072),
];
const BATCHES: &[usize] = &[1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32, 64, 128, 512];
/// Weight copies that rotate: enough bytes (about 100 MB) for a cache to be useless, at least 2.
fn copies_for(bytes: u64) -> usize {
    ((100u64 << 20) / bytes.max(1)).clamp(2, 6) as usize
}

pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLLM_VK_BENCH_LINEAR").is_some_and(|v| !v.is_empty() && v != "0")) && !CLAIMED.swap(true, Ordering::AcqRel)
}

impl VulkanStream {
    pub(super) fn bench_linear_table(&mut self) {
        if let Err(e) = self.bench_linear_inner() {
            log_once("vulkan-bench-linear-error", &format!("[fllm-vulkan-bench] stopped: {e}"));
        }
    }

    /// Raw rates of the matrix units (`bench_mma.comp`): float16 and int8 cooperative-matrix products, the int8 products with the per-group
    /// conversion / scaling of a K-quant tile, and the int8 dot product, all on registers.
    fn bench_mma(&mut self) -> Result<(), BackendError> {
        let caps = self.sh.caps();
        let sg = caps.subgroup_size.max(32);
        let (coop_f16, coop_i8) = {
            use crate::backend::vulkan::instance::ComponentType as T;
            (
                !caps.coop_matrix_for(T::F16, T::F16, T::F32, T::F32).is_empty(),
                !caps.coop_matrix_for(T::I8, T::I8, T::I32, T::I32).is_empty(),
            )
        };
        let out = self.sh.new_zeroed_buffer(4096 * 4)?;
        let (groups, iterations) = (4096u32, 512u32);
        let mut line = String::from("[fllm-vulkan-bench] raw matrix-unit rates (TMAC/s):");
        for (name, macs_per_iter_group, usable) in [
            ("bench_mma_f16", 4.0 * 4096.0 * 4.0, coop_f16),
            ("bench_mma_i8", 4.0 * 4096.0 * 4.0, coop_i8),
            ("bench_mma_i8s", 4.0 * 4096.0 * 4.0, coop_i8),
            ("bench_mma_f16b", 16.0 * 4096.0 * 4.0, coop_f16),
            ("bench_mma_i8b", 16.0 * 4096.0 * 4.0, coop_i8),
            ("bench_mma_i8sb", 16.0 * 4096.0 * 4.0, coop_i8),
            ("bench_mma_dp4a", 4.0 * 4.0 * 4.0 * f64::from(sg), true),
        ] {
            if !usable {
                line.push_str(&format!(" {name}: unsupported;"));
                continue;
            }
            let pipeline = match self.sh.pipeline(name, &[(0, 4 * sg)], 1, 4) {
                Ok(p) => p,
                Err(e) => {
                    line.push_str(&format!(" {name}: {e};"));
                    continue;
                }
            };
            let mut best = f64::MAX;
            for run in 0..4 {
                let start = std::time::Instant::now();
                for _ in 0..3 {
                    self.rec.bind_pipeline(&pipeline)?;
                    self.rec.set_buffers(&[&out])?;
                    self.rec.push_constants(&iterations.to_le_bytes())?;
                    self.rec.dispatch(groups, 1, 1)?;
                }
                self.wait_idle()?;
                if run > 0 {
                    best = best.min(start.elapsed().as_secs_f64() / 3.0);
                }
            }
            let macs = macs_per_iter_group * f64::from(iterations) * f64::from(groups);
            line.push_str(&format!(" {name}: {:.2};", macs / best / 1e12));
        }
        log_once("vulkan-bench-mma", &line);
        Ok(())
    }

    fn bench_linear_inner(&mut self) -> Result<(), BackendError> {
        if let Err(e) = self.bench_mma() {
            log_once("vulkan-bench-mma-error", &format!("[fllm-vulkan-bench] matrix-unit rates stopped: {e}"));
        }
        let shapes: Vec<String> = self
            .sh
            .caps()
            .coop_matrix
            .iter()
            .map(|c| format!("{}x{}x{} {:?}*{:?}+{:?}->{:?} {:?}", c.m, c.n, c.k, c.a, c.b, c.c, c.result, c.scope))
            .collect();
        log_once("vulkan-bench-coop-shapes", &format!("[fllm-vulkan-bench] cooperative-matrix shapes: {}", shapes.join("; ")));
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let noise: Vec<u8> = (0..1 << 20)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect();
        for &(id, name, rows, cols) in SHAPES {
            let quant = QuantType::from_gguf_type_id(id);
            let kind = WeightKind::of(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
            let slices = slice_table(kind, rows, cols, self.sh.caps().max_storage_buffer_range, self.sh.align())?;
            let bytes = slices.last().map_or(0, |s| s.offset + s.bytes) as u64;
            let mut weights = Vec::new();
            let copies = copies_for(bytes);
            for _ in 0..copies {
                let buffer = self.sh.new_zeroed_buffer(bytes)?;
                let stage = self.host.alloc(&self.sh, noise.len() as u64)?;
                stage.write(0, &noise)?;
                let mut at = 0u64;
                while at < bytes {
                    let n = (noise.len() as u64).min(bytes - at);
                    self.rec.copy_buffer(&stage, 0, &buffer, at, n)?;
                    at += n;
                }
                self.wait_idle()?;
                weights.push(LinearWeight { quant, rows, cols, scale: 1.0, weights: buffer, perm: None, bias: None });
            }
            let mut line = format!("[fllm-vulkan-bench] {name} {rows}x{cols} (ms / TMACs / GB/s):");
            for &batch in BATCHES {
                let x = self.sh.new_zeroed_buffer((batch * cols * 4) as u64)?;
                let out = self.sh.new_zeroed_buffer((batch * rows * 4) as u64)?;
                let mut best = f64::MAX;
                for run in 0..4 {
                    let start = std::time::Instant::now();
                    for w in weights.iter().cycle().take(copies * 2) {
                        self.trace = None;
                        self.linear_core(w, batch, &x, &out)?;
                    }
                    self.wait_idle()?;
                    if run > 0 {
                        best = best.min(start.elapsed().as_secs_f64() * 1e3 / (copies * 2) as f64);
                    }
                }
                let macs = (rows * cols * batch) as f64;
                line.push_str(&format!(" b{batch}: {best:.3}/{:.2}/{:.0};", macs / best / 1e9, bytes as f64 / best / 1e6));
            }
            log_once(&format!("vulkan-bench-linear-{name}-{rows}x{cols}"), &line);
        }
        Ok(())
    }
}
