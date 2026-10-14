//! Hardware check of the p06b kernels on a real Metal device (package p06b), for the Mac test round.
//!
//! Runs the same cases as the CPU emulator (`mirror.rs`: inputs, the plans of this module, the result of the engine's CPU backend as reference), but
//! with the real Metal compiler and GPU: every plan is dispatched through the command recorder of the Metal core (package p05) exactly as the stream
//! (package p07) will do it — constants struct with `bytes_raw` at buffer 0, the slots at buffer 1.. with their declared accesses (which drive the
//! barriers between the plans of one case) — and the output buffers are compared with the reference. One line per case:
//!
//! ```text
//! cargo test --release -p fllm-engine --features metal metal_ops_kernels -- --nocapture
//! FLLM_P06B_FILTER=prefill_q8 cargo test ...        # only the cases whose name contains the text
//! ```
//!
//! If a case fails: first `FLLM_METAL_SERIAL=1` (rules out missing barriers / wrong declared accesses), then `FLLM_METAL_FAST_MATH=0`, then compare the
//! pattern of `[ok]` / `[FAIL]` lines with the emulator (`emu_ops/run_emu.sh` runs the same cases on the CPU: a kernel that passes there and fails here points
//! at Metal specifics: address spaces, precision, a pipeline limit, `simdgroup_matrix` behaviour).
//!
//! Declared in `ops.rs` (`#[cfg(all(test, target_vendor = "apple"))]`) since the merge with the Metal core of package p05.

use super::attention::AttentionTuning;
use super::mirror::{
    buffer_name, gather_buffers, run_all_cases, set_device_tuning, Expect, Init, Sink,
};
use super::*;
use crate::backend::metal::buffers::{BufferRef, BufferUsage};
use crate::backend::metal::commands::MetalContext;
use std::collections::BTreeMap;

struct GpuSink<'a> {
    ctx: &'a MetalContext,
    filter: Option<String>,
    checked: usize,
    failed: usize,
}

impl GpuSink<'_> {
    fn dispatch(&self, plans: &[OpPlan], bufs: &BTreeMap<String, BufferRef>) -> Result<(), String> {
        let mut rec = self.ctx.recorder();
        for plan in plans {
            let kernel = plan.kernel_name.as_str();
            let pipe = self
                .ctx
                .pipelines
                .get_plain(kernel)
                .map_err(|e| format!("pipeline {kernel}: {e:?}"))?;
            let threads: usize = plan.threads_per_group.iter().map(|&t| t as usize).product();
            if threads > pipe.max_threads() {
                return Err(format!(
                    "{kernel}: {threads} threads per group planned, the pipeline allows {} (SIMD width {})",
                    pipe.max_threads(),
                    pipe.simd_width()
                ));
            }
            let mut d = rec
                .begin(&pipe)
                .map_err(|e| format!("begin {kernel}: {e:?}"))?;
            d.bytes_raw(CONSTANTS_BUFFER_INDEX, &plan.constants);
            for slot in &plan.slots {
                let buf = &bufs[&buffer_name(slot)];
                match slot.access {
                    Access::Read => d.read(slot.index, buf),
                    Access::Write => d.write(slot.index, buf),
                    Access::ReadWrite => d.read_write(slot.index, buf),
                };
            }
            d.threadgroups(
                plan.threadgroups.map(|g| g as usize),
                plan.threads_per_group.map(|t| t as usize),
            )
            .map_err(|e| format!("dispatch {kernel}: {e:?}"))?;
        }
        rec.synchronize().map_err(|e| format!("synchronize: {e:?}"))
    }
}

/// `(mismatches, first mismatching element, largest absolute difference)` of an output buffer.
fn compare(dtype: &str, got: &[u8], want: &[u8], abs: f64, rel: f64) -> (usize, usize, f64) {
    let (mut bad, mut first, mut worst) = (0usize, 0usize, 0.0f64);
    if dtype == "f32" {
        for (i, (g, w)) in got
            .as_chunks::<4>()
            .0
            .iter()
            .zip(want.as_chunks::<4>().0)
            .enumerate()
        {
            let (g, w) = (f32::from_le_bytes(*g), f32::from_le_bytes(*w));
            let d = (f64::from(g) - f64::from(w)).abs();
            worst = worst.max(if d.is_nan() { f64::INFINITY } else { d });
            let within = d <= abs + rel * f64::from(w).abs();
            if !within || g.is_nan() != w.is_nan() {
                if bad == 0 {
                    first = i;
                }
                bad += 1;
            }
        }
    } else {
        let width = if dtype == "i32" { 4 } else { 1 };
        for (i, (g, w)) in got
            .chunks_exact(width)
            .zip(want.chunks_exact(width))
            .enumerate()
        {
            if g != w {
                if bad == 0 {
                    first = i;
                }
                bad += 1;
            }
        }
    }
    (bad, first, worst)
}

impl Sink for GpuSink<'_> {
    fn emit(
        &mut self,
        name: &str,
        plans: &[OpPlan],
        inputs: &[(&str, Vec<u8>)],
        expects: &[Expect],
    ) {
        if self.filter.as_deref().is_some_and(|f| !name.contains(f)) {
            return;
        }
        self.checked += 1;
        let mut bufs: BTreeMap<String, BufferRef> = BTreeMap::new();
        for (bname, init) in gather_buffers(name, plans, inputs) {
            let made = match &init {
                Init::Bytes(b) => self.ctx.memory.from_bytes(b, BufferUsage::Activations),
                Init::Zero(n) => self.ctx.memory.alloc_zeroed(*n, BufferUsage::Activations),
            };
            match made {
                Ok(buf) => {
                    bufs.insert(bname, BufferRef::whole(buf));
                }
                Err(e) => {
                    println!("[FAIL] {name}: cannot allocate {bname}: {e:?}");
                    self.failed += 1;
                    return;
                }
            }
        }
        if let Err(e) = self.dispatch(plans, &bufs) {
            println!("[FAIL] {name}: {e}");
            self.failed += 1;
            return;
        }
        let mut problems = Vec::new();
        let mut worst = 0.0f64;
        for (bname, want, dtype, abs, rel) in expects {
            let mut got = vec![0u8; want.len()];
            if let Err(e) = bufs[*bname].read(0, &mut got) {
                problems.push(format!("{bname}: read back failed: {e:?}"));
                continue;
            }
            let (bad, first, w) = compare(dtype, &got, want, *abs, *rel);
            worst = worst.max(w);
            if bad > 0 {
                problems.push(format!("{bname} ({dtype}): {bad} mismatches, first at element {first}, largest difference {w:.3e}"));
            }
        }
        let kernels: Vec<&str> = plans.iter().map(|p| p.kernel_name.as_str()).collect();
        if problems.is_empty() {
            println!(
                "[ok]   {name} [{}] max abs diff {worst:.2e}",
                kernels.join("+")
            );
        } else {
            println!(
                "[FAIL] {name} [{}]: {}",
                kernels.join("+"),
                problems.join("; ")
            );
            self.failed += 1;
        }
    }
}

#[test]
fn metal_ops_kernels() {
    let ctx = match MetalContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] no Metal device: {e:?}");
            return;
        }
    };
    let info = ctx.device.info();
    println!("device: {}", info.summary());
    let simd_width = match ctx.pipelines.get_plain("fllm_residual_add_f32") {
        Ok(p) => p.simd_width(),
        Err(e) => panic!("the sources do not compile: {e:?}"),
    };
    set_device_tuning(AttentionTuning {
        simdgroup_matrix: info.simdgroup_matrix,
        simd_width,
        max_threads: info.max_threads_per_threadgroup[0],
        ..AttentionTuning::default()
    });
    let mut sink = GpuSink {
        ctx: &ctx,
        filter: std::env::var("FLLM_P06B_FILTER").ok(),
        checked: 0,
        failed: 0,
    };
    run_all_cases(&mut sink);
    println!("{} cases, {} failed", sink.checked, sink.failed);
    assert_eq!(
        sink.failed, 0,
        "{} of {} cases failed (see the [FAIL] lines)",
        sink.failed, sink.checked
    );
}
