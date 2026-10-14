//! Hardware check of the quantized linear kernels for the Metal test round (needs an Apple GPU).
//!
//! Declared in `weights.rs` (Apple targets only) since the merge with the Metal core of package p05. Verification level: L0 -
//! type-checked for `aarch64-apple-darwin`, never run.
//!
//! For every supported format the check uploads sample weights ([`super::sample`]), runs the kernels the planner selects for several batch
//! sizes (matrix-vector 1, 3, 8; matrix-matrix 9, 33 with half and float tiles; bias and per-tensor scale on) and compares the GPU output
//! with an f64 reference built from `compute::dequantize`. A failing line names format, kernel and the first wrong element.

use super::sample::{reference_weights, sample_weights, Lcg};
use super::{
    format_info, pack_weight, plan_quantized_linear, supported_formats, LinearPlan, MetalCapsLike,
};
use crate::backend::metal::buffers::{BufferRef, BufferUsage};
use crate::backend::metal::commands::{MetalContext, Recorder};
use crate::backend::metal::device_info::MetalError;
use crate::backend::metal::pipelines::FunctionConstant;
use crate::backend::QuantType;

/// Outcome of a run: one line per check, `[ok]` / `[FAIL]` / `[skip]`.
#[derive(Debug, Clone, Default)]
pub struct GpuCheckReport {
    pub device: String,
    pub lines: Vec<String>,
    pub checks: usize,
    pub failures: usize,
}

impl GpuCheckReport {
    pub fn summary(&self) -> String {
        format!(
            "{}: {} checks, {} failures",
            self.device, self.checks, self.failures
        )
    }
}

/// Opens Metal device `index` and runs [`run`].
pub fn run_default(index: usize) -> Result<GpuCheckReport, MetalError> {
    Ok(run(&MetalContext::new(index)?))
}

fn floats_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn dispatch(
    ctx: &MetalContext,
    rec: &mut Recorder,
    plan: &LinearPlan,
    buffers: [&BufferRef; 4],
) -> Result<(), MetalError> {
    let [w, x, out, bias] = buffers;
    let constants: Vec<FunctionConstant> = plan
        .constants
        .iter()
        .map(|c| FunctionConstant::u32(c.index, c.value))
        .collect();
    let pipeline = ctx.pipelines.get(&plan.kernel_name, &constants)?;
    let (groups, threads) = plan.dispatch_dims();
    let mut d = rec.begin(&pipeline)?;
    d.read(plan.slots.weights, w).read(plan.slots.x, x);
    if plan.reads_output() {
        d.read_write(plan.slots.out, out);
    } else {
        d.write(plan.slots.out, out);
    }
    d.read(plan.slots.bias, bias)
        .bytes(plan.slots.params, &plan.params);
    for (slot, bytes) in &plan.threadgroup_memory {
        d.threadgroup_memory(*slot, *bytes as usize);
    }
    d.threadgroups(groups, threads)
}

struct Case<'a> {
    quant: QuantType,
    rows: usize,
    cols: usize,
    batch: usize,
    caps: &'a MetalCapsLike,
}

/// One kernel run; returns the largest error relative to `sum |w x|` and a description of the first violation, if any.
fn check_case(
    ctx: &MetalContext,
    case: &Case<'_>,
    weights: &BufferRef,
    perm: Option<&BufferRef>,
    reference: &[f32],
) -> Result<(String, f64, Option<String>), MetalError> {
    let Case {
        quant,
        rows,
        cols,
        batch,
        caps,
    } = *case;
    let scale = 0.75f32;
    let mut rng = Lcg(0x5eed ^ (batch as u64) << 8 ^ rows as u64);
    let x: Vec<f32> = (0..batch * cols).map(|_| rng.unit() * 1.5).collect();
    let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
    let x_buf = BufferRef::whole(
        ctx.memory
            .from_bytes(&floats_to_bytes(&x), BufferUsage::Activations)?,
    );
    let bias_buf = BufferRef::whole(
        ctx.memory
            .from_bytes(&floats_to_bytes(&bias), BufferUsage::Activations)?,
    );
    let out_buf = BufferRef::whole(
        ctx.memory
            .alloc_zeroed(batch * rows * 4, BufferUsage::Activations)?,
    );
    let plan = plan_quantized_linear(quant, rows, cols, batch, caps)
        .map_err(|e| MetalError::InvalidDispatch(format!("plan: {e}")))?
        .with_bias()
        .with_scale(scale);
    let mut rec = ctx.recorder();
    let mut input = x_buf.clone();
    if let Some(pre) = &plan.activation_prepass {
        let perm = perm.expect("act-order weight without permutation buffer");
        let y = BufferRef::whole(
            ctx.memory
                .alloc(pre.output_bytes, BufferUsage::Activations)?,
        );
        let pipeline = ctx.pipelines.get_plain(pre.kernel_name)?;
        let (groups, threads) = pre.dispatch_dims();
        let mut d = rec.begin(&pipeline)?;
        d.read(pre.slots[0], &x_buf)
            .write(pre.slots[1], &y)
            .read(pre.slots[2], perm)
            .bytes(pre.slots[3], &pre.params);
        d.threadgroups(groups, threads)?;
        input = y;
    }
    dispatch(ctx, &mut rec, &plan, [weights, &input, &out_buf, &bias_buf])?;
    rec.synchronize()?;
    let mut bytes = vec![0u8; batch * rows * 4];
    out_buf.read(0, &mut bytes)?;
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let tolerance = if plan.half_tiles { 4e-3 } else { 1e-4 };
    let (mut worst, mut first_bad) = (0.0f64, None);
    for b in 0..batch {
        for r in 0..rows {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for k in 0..cols {
                let t = reference[r * cols + k] as f64 * x[b * cols + k] as f64;
                acc += t;
                mag += t.abs();
            }
            let want = acc * scale as f64 + bias[r] as f64;
            let have = got[b * rows + r] as f64;
            let err = (have - want).abs();
            worst = worst.max(err / (mag * scale as f64 + 1e-9));
            let allowed = tolerance * mag * scale as f64 + 1e-5;
            if (err > allowed || err.is_nan()) && first_bad.is_none() {
                first_bad = Some(format!("token {b} row {r}: got {have}, want {want}"));
            }
        }
    }
    Ok((plan.kernel_name.clone(), worst, first_bad))
}

/// Runs every format on `ctx`. Never panics; failures are in the report.
pub fn run(ctx: &MetalContext) -> GpuCheckReport {
    let info = ctx.device.info();
    let mut report = GpuCheckReport {
        device: info.summary(),
        ..Default::default()
    };
    let simd_width = if info.apple_family.is_some() {
        32
    } else {
        info.simd_width_hint as u32
    };
    let base_caps = MetalCapsLike::new(
        info.simdgroup_matrix,
        info.bfloat,
        simd_width,
        info.max_threads_per_threadgroup[0] as u32,
        info.has_unified_memory,
    );
    for quant in supported_formats() {
        if format_info(quant).is_none() {
            continue;
        }
        let (rows, cols) = (70usize, 256usize); // every block size divides 256
        let raw = sample_weights(quant, rows, cols, 0xfeed ^ quant.to_gguf_type_id() as u64);
        let reference = match reference_weights(quant, &raw, rows, cols) {
            Ok(v) => v,
            Err(e) => {
                report
                    .lines
                    .push(format!("[skip] {quant:?}: no CPU reference ({e})"));
                continue;
            }
        };
        let packed = match pack_weight(quant, rows, cols, &raw) {
            Ok(p) => p,
            Err(e) => {
                report.checks += 1;
                report.failures += 1;
                report
                    .lines
                    .push(format!("[FAIL] {quant:?}: pack_weight: {e}"));
                continue;
            }
        };
        let uploaded = ctx
            .memory
            .from_bytes(&packed.bytes, BufferUsage::Weights)
            .map(BufferRef::whole)
            .and_then(|w| {
                let perm = match &packed.perm {
                    Some(p) => {
                        let bytes: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
                        Some(BufferRef::whole(
                            ctx.memory.from_bytes(&bytes, BufferUsage::Weights)?,
                        ))
                    }
                    None => None,
                };
                Ok((w, perm))
            });
        let (weights, perm) = match uploaded {
            Ok(v) => v,
            Err(e) => {
                report.checks += 1;
                report.failures += 1;
                report.lines.push(format!("[FAIL] {quant:?}: upload: {e}"));
                continue;
            }
        };
        for exact in [false, true] {
            let caps = MetalCapsLike { exact, ..base_caps };
            for batch in [1usize, 3, 8, 9, 33] {
                // the matrix-vector kernels do not depend on `exact`
                if exact && batch <= 8 {
                    continue;
                }
                let case = Case {
                    quant,
                    rows,
                    cols,
                    batch,
                    caps: &caps,
                };
                report.checks += 1;
                match check_case(ctx, &case, &weights, perm.as_ref(), &reference) {
                    Ok((kernel, worst, None)) => report.lines.push(format!(
                        "[ok]   {quant:?} batch {batch:2} {kernel}: max rel err {worst:.2e}"
                    )),
                    Ok((kernel, worst, Some(first))) => {
                        report.failures += 1;
                        report.lines.push(format!(
                            "[FAIL] {quant:?} batch {batch:2} {kernel}: max rel err {worst:.2e}; {first}"
                        ));
                    }
                    Err(e) => {
                        report.failures += 1;
                        report
                            .lines
                            .push(format!("[FAIL] {quant:?} batch {batch:2}: {e}"));
                    }
                }
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    /// `cargo test --release -p fllm-engine --features metal metal_quant_kernels -- --nocapture` on a Mac.
    #[test]
    fn metal_quant_kernels_match_the_cpu_reference() {
        let report = match super::run_default(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[skip] no Metal device: {e}");
                return;
            }
        };
        for line in &report.lines {
            println!("{line}");
        }
        println!("{}", report.summary());
        assert_eq!(report.failures, 0, "{}", report.summary());
    }
}
