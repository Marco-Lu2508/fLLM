//! `Stream::quantized_linear` and `host_linear` of the Vulkan backend: slicing of big weights, choice of the shader variant, the activation
//! prepass (shared between projections of the same input) and the barrier-free dispatch of projections that read the same input.

use super::exec::WriteBack;
use super::VulkanStream;
use crate::backend::vulkan::backend::mem::{slice_table, table_bytes, view_of, SliceInfo, WeightKind};
use crate::backend::vulkan::backend::shared::{is_exact_math, verbose};
use crate::backend::vulkan::backend::upload::pack_slice;
use crate::backend::vulkan::memory::Buffer;
use crate::backend::vulkan::pipeline::ComputePipeline;
use crate::backend::vulkan::weights::exl3::{plan_exl3, MM_PUSH_BYTES, POST_PUSH_BYTES, PRE_PUSH_BYTES};
use crate::backend::vulkan::weights::{
    format_info, is_exl3, plan_quantized_linear_with, split_groups, LinearPlan, PlanOptions, PrepassRole, Variant,
};
use crate::backend::{log_once, BackendError, DeviceBuffer, DeviceWeight, QuantType};
use std::sync::Arc;

/// A planned slice: the plan, the pipeline of the main shader and the one of the activation prepass.
type Planned = (LinearPlan, Arc<ComputePipeline>, Option<Arc<ComputePipeline>>);

/// `(allocation id, absolute offset, size)` of a buffer view.
pub(super) type BufKey = (u64, u64, u64);

pub(super) fn key_of(buffer: &Buffer) -> BufKey {
    (buffer.id(), buffer.offset(), buffer.size())
}

pub(super) fn overlaps(a: &BufKey, b: &BufKey) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// The prepass result of the latest projection(s): `(shader, push constants, permutation id)` and the scratch buffer holding the result.
type PrepassEntry = ((String, Vec<u8>, u64), Buffer);

/// What the previous projection(s) of this submission read and wrote, to record the next projection of the same input without a barrier.
pub(super) struct LinTrace {
    x: BufKey,
    outs: Vec<BufKey>,
    prepass: Option<PrepassEntry>,
}

/// A weight as the projection sees it.
pub(super) struct LinearWeight {
    pub quant: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub scale: f32,
    /// The whole weight buffer (all slices).
    pub weights: Buffer,
    pub perm: Option<Buffer>,
    pub bias: Option<Buffer>,
}

/// Plan of the float-weight shader `lin_f32` / `lin_f16` / `lin_bf16` (`lin_float.comp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FloatPlan {
    pub shader: &'static str,
    /// `FPush` of the shader: rows, cols, batch, flags, scale, out_stride, row_base, pad (32 bytes).
    pub push: Vec<u8>,
    pub groups: [u32; 3],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_float_linear(
    quant: QuantType,
    rows: usize,
    cols: usize,
    batch: usize,
    row_base: usize,
    out_stride: usize,
    has_bias: bool,
    scale: f32,
) -> Result<FloatPlan, BackendError> {
    let shader = match quant {
        QuantType::F32 => "lin_f32",
        QuantType::F16 => "lin_f16",
        QuantType::BF16 => "lin_bf16",
        other => return Err(BackendError::UnsupportedQuant(other)),
    };
    let fit = |v: usize, what: &str| u32::try_from(v).map_err(|_| BackendError::InvalidShape(format!("linear {what} {v} exceeds 32 bits")));
    if rows == 0 || cols == 0 || batch == 0 {
        return Err(BackendError::InvalidShape("linear projection of an empty tensor".into()));
    }
    if (rows as u64) * (cols as u64) > u64::from(u32::MAX) {
        return Err(BackendError::InvalidShape(format!(
            "float weight slice of {rows} x {cols} exceeds 32-bit element indexing"
        )));
    }
    let words = [
        fit(rows, "rows")?,
        fit(cols, "columns")?,
        fit(batch, "batch")?,
        u32::from(has_bias),
        scale.to_bits(),
        fit(out_stride, "output stride")?,
        fit(row_base, "row base")?,
        0,
    ];
    let mut groups = split_groups(rows as u64);
    let z = batch.div_ceil(8);
    if z > 65535 {
        return Err(BackendError::InvalidShape(format!(
            "batch {batch} is too large for one float linear dispatch"
        )));
    }
    groups[2] = z as u32;
    Ok(FloatPlan {
        shader,
        push: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        groups,
    })
}

/// Smallest activation batch for which the choice between the cooperative-matrix and the int8 dot-product matrix shaders is measured.
const PROMPT_TUNE_MIN_BATCH: usize = 9;

impl VulkanStream {
    /// Once per device: times the matrix-matrix product of a Q4_K weight (4096 x 4096, 512 tokens) with the cooperative-matrix shaders and with the int8
    /// dot-product shaders and keeps the faster one for the rest of the process. Neither is faster everywhere: on an RTX 2000 Ada the cooperative
    /// matrices (tensor cores) win by a factor of 2.2, on a Radeon 840M (Mesa RADV, 4 compute units) the integer dot product wins by 22 %.
    /// `FLLM_VULKAN_NO_TUNE=1` keeps the planner's order (cooperative matrices first).
    fn tune_prompt_variant(&mut self) {
        if self.mask_coop || is_exact_math() || std::env::var_os("FLLM_VULKAN_NO_TUNE").is_some_and(|v| !v.is_empty() && v != "0") {
            return;
        }
        let caps = self.sh.caps_like();
        if !(caps.coop_matrix && caps.int_dot) || !self.sh.claim_prompt_tuning() {
            return;
        }
        match self.measure_prompt_variants() {
            Ok((coop_ms, idot_ms)) => {
                let idot_wins = idot_ms < coop_ms * 0.85;
                if idot_wins {
                    self.sh.prefer_int_dot_over_coop();
                }
                log_once(
                    "vulkan-prompt-variant",
                    &format!(
                        "[fllm-vulkan] prompt matrix shaders measured on {}: cooperative matrices {coop_ms:.1} ms, int8 dot product {idot_ms:.1} ms (4096 x 4096 Q4_K, 512 tokens): using {}",
                        self.sh.caps().device_name,
                        if idot_wins { "the int8 dot product" } else { "cooperative matrices" }
                    ),
                );
            }
            Err(e) => log_once("vulkan-prompt-variant", &format!("[fllm-vulkan] the prompt shader variants could not be measured ({e}); keeping the default order")),
        }
    }

    /// `(cooperative-matrix ms, int8-dot ms)` per product of a zero-filled 4096 x 4096 Q4_K weight with 512 tokens (best of three runs after a warm-up).
    fn measure_prompt_variants(&mut self) -> Result<(f64, f64), BackendError> {
        const ROWS: usize = 4096;
        const COLS: usize = 4096;
        const BATCH: usize = 512;
        let quant = QuantType::from_gguf_type_id(12);
        let kind = WeightKind::of(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
        let slices = slice_table(kind, ROWS, COLS, self.sh.caps().max_storage_buffer_range, self.sh.align())?;
        let bytes = slices.last().map_or(0, |s| s.offset + s.bytes) as u64;
        let weights = self.sh.new_zeroed_buffer(bytes)?;
        let x = self.sh.new_zeroed_buffer((BATCH * COLS * 4) as u64)?;
        let out = self.sh.new_zeroed_buffer((BATCH * ROWS * 4) as u64)?;
        let lw = LinearWeight { quant, rows: ROWS, cols: COLS, scale: 1.0, weights, perm: None, bias: None };
        let mut times = [f64::MAX; 2];
        for (slot, mask_coop) in [(0usize, false), (1, true)] {
            let mut stream = VulkanStream::new(Arc::clone(&self.sh))?;
            stream.mask_coop = mask_coop;
            for run in 0..4 {
                let start = std::time::Instant::now();
                for _ in 0..3 {
                    stream.trace = None;
                    stream.linear_core(&lw, BATCH, &x, &out)?;
                }
                stream.wait_idle()?;
                if run > 0 {
                    times[slot] = times[slot].min(start.elapsed().as_secs_f64() * 1e3 / 3.0);
                }
            }
        }
        Ok((times[0], times[1]))
    }

    /// `Stream::quantized_linear` for a weight described by `w`: `out[b][r] = scale * dot(x[b], w[r]) + bias[r]`, `x` and `out` already valid views.
    pub(super) fn linear_core(&mut self, w: &LinearWeight, batch: usize, x: &Buffer, out: &Buffer) -> Result<(), BackendError> {
        // The budget of a submission (`DEFAULT_MAX_MACS_PER_SUBMIT`) was calibrated on quantized weights. The float kernels (`lin_f16`, `lin_f32`: models in
        // F16 / BF16, RWKV adapters) ran at about 80 GMAC/s on a Radeon 840M, so a submission at the quantized budget lasted seconds there and the driver
        // reported "Vulkan device lost" on prompts of a 1.6B F16 model: a multiply-add of a float weight counts eightfold.
        let weight = if matches!(w.quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) { 8 } else { 1 };
        self.pending_macs = self.pending_macs.saturating_add((w.rows as u64).saturating_mul(w.cols as u64).saturating_mul(batch as u64).saturating_mul(weight));
        let kind = WeightKind::of(w.quant).ok_or(BackendError::UnsupportedQuant(w.quant))?;
        let slices = slice_table(kind, w.rows, w.cols, self.sh.caps().max_storage_buffer_range, self.sh.align())?;
        if slices.last().is_some_and(|s| (s.offset + s.bytes) as u64 > w.weights.size()) {
            return Err(BackendError::InvalidShape(format!(
                "weight buffer of {} bytes is too small for a {} x {} {:?} weight",
                w.weights.size(),
                w.rows,
                w.cols,
                w.quant
            )));
        }
        if w.bias.as_ref().is_some_and(|b| b.size() < (w.rows * 4) as u64) {
            return Err(BackendError::InvalidShape("bias shorter than the rows of the weight".into()));
        }
        let (x_key, out_key) = (key_of(x), key_of(out));
        if overlaps(&out_key, &x_key) {
            return Err(BackendError::InvalidShape("the output of a projection overlaps its input".into()));
        }
        if is_exl3(w.quant) {
            // ExLlamaV3: three dependent passes through scratch buffers; the projections that share an input cannot share a prepass here
            self.trace = None;
            let view = w.weights.slice(slices[0].offset as u64, slices[0].bytes as u64)?;
            return self.record_exl3(w, &view, batch, x, out);
        }
        let (mut trace, same_x) = match self.trace.take() {
            Some(t) if t.x == x_key && !t.outs.iter().any(|o| overlaps(o, &out_key) || overlaps(o, &x_key)) => (t, true),
            _ => (
                LinTrace {
                    x: x_key,
                    outs: Vec::new(),
                    prepass: None,
                },
                false,
            ),
        };
        trace.outs.push(out_key);
        for (index, slice) in slices.iter().enumerate() {
            let view = w.weights.slice(slice.offset as u64, slice.bytes as u64)?;
            let overlap_ok = same_x || index > 0;
            match kind {
                WeightKind::Float { .. } => self.record_float(w, slice, &view, batch, x, out, overlap_ok)?,
                WeightKind::Quant(_) => self.record_quant(w, slice, &view, batch, x, out, &mut trace, same_x, overlap_ok)?,
            }
        }
        self.trace = Some(trace);
        Ok(())
    }

    /// `y = diag(svh) H (R^T (H diag(suh) x))` for an EXL3 weight: per chunk of up to 128 tokens the input transform, the trellis product and the output
    /// transform with the scatter through the output map (`exl3_pre`, `exl3_mm`, `exl3_post`).
    fn record_exl3(&mut self, w: &LinearWeight, weights: &Buffer, batch: usize, x: &Buffer, out: &Buffer) -> Result<(), BackendError> {
        let plan = plan_exl3(w.rows, w.cols, batch, w.bias.is_some(), w.scale)?;
        let pre = self.sh.pipeline("exl3_pre", &[], 3, PRE_PUSH_BYTES)?;
        let mm = self.sh.pipeline("exl3_mm", &[(0, plan.tile_groups)], 3, MM_PUSH_BYTES)?;
        let post = self.sh.pipeline("exl3_post", &[], 4, POST_PUSH_BYTES)?;
        let x2 = self.dev.alloc(&self.sh, plan.x2_bytes as u64)?;
        let z = self.dev.alloc(&self.sh, plan.z_bytes as u64)?;
        let bias = self.bias_binding(w);
        for pass in &plan.passes {
            self.rec.bind_pipeline(&pre)?;
            self.rec.set_buffers(&[weights, x, &x2])?;
            self.rec.push_constants(&pass.pre_push)?;
            let [gx, gy, gz] = pass.pre_groups;
            self.rec.dispatch(gx, gy, gz)?;
            self.rec.bind_pipeline(&mm)?;
            self.rec.set_buffers(&[weights, &x2, &z])?;
            self.rec.push_constants(&pass.mm_push)?;
            let [gx, gy, gz] = pass.mm_groups;
            self.rec.dispatch(gx, gy, gz)?;
            self.rec.bind_pipeline(&post)?;
            self.rec.set_buffers(&[weights, &z, out, &bias])?;
            self.rec.push_constants(&pass.post_push)?;
            let [gx, gy, gz] = pass.post_groups;
            self.rec.dispatch(gx, gy, gz)?;
        }
        Ok(())
    }

    fn bias_binding(&self, w: &LinearWeight) -> Buffer {
        w.bias.clone().unwrap_or_else(|| self.sh.dummy.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_float(
        &mut self,
        w: &LinearWeight,
        slice: &SliceInfo,
        weights: &Buffer,
        batch: usize,
        x: &Buffer,
        out: &Buffer,
        independent: bool,
    ) -> Result<(), BackendError> {
        let plan = plan_float_linear(
            w.quant,
            slice.rows,
            w.cols,
            batch,
            slice.row_base,
            w.rows,
            w.bias.is_some(),
            w.scale,
        )?;
        let pipeline = self.sh.pipeline(plan.shader, &[], 4, plan.push.len() as u32)?;
        let bias = self.bias_binding(w);
        self.rec.bind_pipeline(&pipeline)?;
        self.rec.set_buffers(&[weights, x, out, &bias])?;
        self.rec.push_constants(&plan.push)?;
        let [gx, gy, gz] = plan.groups;
        if independent {
            self.rec.dispatch_independent(gx, gy, gz)?;
        } else {
            self.rec.dispatch(gx, gy, gz)?;
        }
        Ok(())
    }

    /// The plan for a slice and its pipelines. A fast variant whose pipeline the driver rejects is switched off for good and the slice is
    /// planned again.
    fn plan_with_pipelines(&mut self, w: &LinearWeight, slice: &SliceInfo, batch: usize) -> Result<Planned, BackendError> {
        let mut last_error = None;
        if super::linear_bench::enabled() {
            self.bench_linear_table();
        }
        if batch >= PROMPT_TUNE_MIN_BATCH {
            self.tune_prompt_variant();
        }
        for _ in 0..3 {
            let mut caps = self.sh.caps_like();
            if self.mask_coop {
                caps.coop_matrix = false;
            }
            let options = PlanOptions {
                has_bias: w.bias.is_some(),
                scale: w.scale,
                row_base: slice.row_base as u32,
                out_stride: Some(w.rows as u32),
                exact: is_exact_math(),
                force: None,
                available: None,
            };
            let plan = plan_quantized_linear_with(w.quant, slice.rows, w.cols, batch, &caps, &options)?;
            let spec: Vec<(u32, u32)> = plan.spec_constants.iter().map(|c| (c.id, c.value)).collect();
            let main = self.sh.pipeline(&plan.shader_name, &spec, 4, plan.push_constants.len() as u32);
            let prepass = match &plan.activation_prepass {
                Some(pp) => self
                    .sh
                    .pipeline(&pp.shader_name, &[], pp.bindings.len() as u32, pp.push_constants.len() as u32)
                    .map(Some),
                None => Ok(None),
            };
            match (main, prepass) {
                (Ok(main), Ok(prepass)) => {
                    if verbose() {
                        log_once(
                            &format!("vulkan-plan-{}", plan.shader_name),
                            &format!(
                                "quantized_linear {:?} {}x{} batch {batch}: {} ({:?}, spec {:?}, grid {:?})",
                                w.quant, slice.rows, w.cols, plan.shader_name, plan.variant, spec, plan.groups
                            ),
                        );
                    }
                    return Ok((plan, main, prepass));
                }
                (Err(error), _) | (_, Err(error)) => {
                    if plan.variant == Variant::Scalar {
                        return Err(error);
                    }
                    log_once(
                        &format!("vulkan-variant-{}", plan.shader_name),
                        &format!(
                            "the Vulkan driver rejected {} ({error}); the {:?} variants are switched off",
                            plan.shader_name, plan.variant
                        ),
                    );
                    self.sh.disable(plan.variant);
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| BackendError::DriverError("no usable Vulkan shader variant".into())))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_quant(
        &mut self,
        w: &LinearWeight,
        slice: &SliceInfo,
        weights: &Buffer,
        batch: usize,
        x: &Buffer,
        out: &Buffer,
        trace: &mut LinTrace,
        same_x: bool,
        overlap_ok: bool,
    ) -> Result<(), BackendError> {
        let (plan, main, prepass_pipeline) = self.plan_with_pipelines(w, slice, batch)?;
        let mut fresh = false;
        let mut input = x.clone();
        if let (Some(pp), Some(pipeline)) = (&plan.activation_prepass, &prepass_pipeline) {
            let perm_id = if pp.uses_perm { w.perm.as_ref().map_or(0, Buffer::id) } else { 0 };
            let key = (pp.shader_name.clone(), pp.push_constants.clone(), perm_id);
            let cached = trace.prepass.as_ref().filter(|(k, _)| *k == key).map(|(_, b)| b.clone());
            input = match cached {
                Some(buffer) => buffer,
                None => {
                    let scratch = self.dev.alloc(&self.sh, pp.output_bytes as u64)?;
                    let mut bufs: Vec<Buffer> = Vec::with_capacity(pp.bindings.len());
                    for binding in &pp.bindings {
                        bufs.push(match binding.role {
                            PrepassRole::Activations => x.clone(),
                            PrepassRole::Output => scratch.clone(),
                            PrepassRole::Perm => w
                                .perm
                                .clone()
                                .ok_or_else(|| BackendError::InvalidShape("act-order weight without its permutation".into()))?,
                        });
                    }
                    let refs: Vec<&Buffer> = bufs.iter().collect();
                    self.rec.bind_pipeline(pipeline)?;
                    self.rec.set_buffers(&refs)?;
                    self.rec.push_constants(&pp.push_constants)?;
                    let [gx, gy, gz] = pp.groups;
                    if same_x {
                        self.rec.dispatch_independent(gx, gy, gz)?;
                    } else {
                        self.rec.dispatch(gx, gy, gz)?;
                    }
                    trace.prepass = Some((key, scratch.clone()));
                    fresh = true;
                    scratch
                }
            };
        }
        let bias = self.bias_binding(w);
        self.rec.bind_pipeline(&main)?;
        self.rec.set_buffers(&[weights, &input, out, &bias])?;
        self.rec.push_constants(&plan.push_constants)?;
        let [gx, gy, gz] = plan.groups;
        if overlap_ok && !fresh {
            self.rec.dispatch_independent(gx, gy, gz)?;
        } else {
            self.rec.dispatch(gx, gy, gz)?;
        }
        Ok(())
    }

    /// `Stream::quantized_linear`.
    pub(super) fn quantized_linear_impl(
        &mut self,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
            return Err(BackendError::InvalidShape(format!(
                "linear input of {} bytes is not a whole number of rows of {cols} floats",
                x.size
            )));
        }
        let batch = x.size / (cols * 4);
        if batch == 0 {
            return Ok(());
        }
        if out.size < batch * rows * 4 {
            return Err(BackendError::InvalidShape(format!(
                "linear output of {} bytes is smaller than {batch} x {rows} floats",
                out.size
            )));
        }
        let mut writebacks: Vec<WriteBack> = Vec::new();
        let x_view = self.operand(x, false, &mut writebacks)?;
        let out_view = self.operand(out, true, &mut writebacks)?;
        // `bias` is what the caller wants added (pass `weight.bias.as_ref()` for the weight's own bias); `None` adds nothing
        let bias_view = match bias {
            Some(b) => Some(self.operand(b, false, &mut writebacks)?),
            None => None,
        };
        let perm = weight.perm.as_ref().map(view_of).transpose()?;
        let lw = LinearWeight {
            quant: weight.quant_type,
            rows,
            cols,
            scale: weight.scale,
            weights: view_of(&weight.buffer)?,
            perm,
            bias: bias_view,
        };
        self.linear_core(&lw, batch, &x_view, &out_view)?;
        self.finish_writebacks(writebacks)?;
        self.after_op()
    }

    /// `Stream::host_linear`: the weight (GGUF bytes) is packed on the host, copied through the scratch rings for this one product and used.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn host_linear_impl(
        &mut self,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        raw: &[u8],
        quant: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        if format_info(quant).is_none() {
            return Err(BackendError::UnsupportedQuant(quant));
        }
        if rows == 0 || cols == 0 || x.size == 0 || !x.size.is_multiple_of(cols * 4) {
            return Err(BackendError::InvalidShape("host_linear input is not a whole number of rows".into()));
        }
        let batch = x.size / (cols * 4);
        if out.size < batch * rows * 4 {
            return Err(BackendError::InvalidShape(
                "host_linear output is smaller than batch x rows floats".into(),
            ));
        }
        if bias.is_some_and(|b| b.len() != rows) {
            return Err(BackendError::InvalidShape(
                "host_linear: bias length does not match the rows".into(),
            ));
        }
        self.touch();
        let mut writebacks: Vec<WriteBack> = Vec::new();
        let x_view = self.operand(x, false, &mut writebacks)?;
        let out_view = self.operand(out, true, &mut writebacks)?;
        let slices = slice_table(
            WeightKind::Quant(quant),
            rows,
            cols,
            self.sh.caps().max_storage_buffer_range,
            self.sh.align(),
        )?;
        let device_weight = self.dev.alloc(&self.sh, table_bytes(&slices) as u64)?;
        let mut perm: Option<Buffer> = None;
        for slice in &slices {
            let packed = pack_slice(quant, raw, rows, cols, slice.row_base, slice.rows)?;
            let stage = self.host.alloc(&self.sh, packed.bytes.len() as u64)?;
            stage.write(0, &packed.bytes)?;
            self.rec
                .copy_buffer(&stage, 0, &device_weight, slice.offset as u64, packed.bytes.len() as u64)?;
            if perm.is_none() {
                if let Some(table) = &packed.perm {
                    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let stage = self.host.alloc(&self.sh, bytes.len() as u64)?;
                    stage.write(0, &bytes)?;
                    let device = self.dev.alloc(&self.sh, bytes.len() as u64)?;
                    self.rec.copy_buffer(&stage, 0, &device, 0, bytes.len() as u64)?;
                    perm = Some(device);
                }
            }
        }
        let bias_buffer = match bias {
            Some(values) => {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                let stage = self.host.alloc(&self.sh, bytes.len() as u64)?;
                stage.write(0, &bytes)?;
                let device = self.dev.alloc(&self.sh, bytes.len() as u64)?;
                self.rec.copy_buffer(&stage, 0, &device, 0, bytes.len() as u64)?;
                Some(device)
            }
            None => None,
        };
        let lw = LinearWeight {
            quant,
            rows,
            cols,
            scale: 1.0,
            weights: device_weight,
            perm,
            bias: bias_buffer,
        };
        self.linear_core(&lw, batch, &x_view, &out_view)?;
        self.finish_writebacks(writebacks)?;
        self.after_op()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_key_overlap() {
        assert!(overlaps(&(1, 0, 100), &(1, 50, 100)));
        assert!(!overlaps(&(1, 0, 100), &(1, 100, 100)));
        assert!(!overlaps(&(1, 0, 100), &(2, 0, 100)));
    }

    #[test]
    fn float_plan_geometry() {
        let p = plan_float_linear(QuantType::F16, 70000, 128, 19, 0, 70000, true, 0.5).unwrap();
        assert_eq!(p.shader, "lin_f16");
        assert_eq!(p.push.len(), 32);
        // 70000 rows are split over x and y, 19 tokens are three groups of 8
        assert_eq!(p.groups, [65535, 2, 3]);
        let word = |i: usize| u32::from_le_bytes(p.push[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!([word(0), word(1), word(2), word(3)], [70000, 128, 19, 1]);
        assert_eq!(word(4), 0.5f32.to_bits());
        assert_eq!([word(5), word(6), word(7)], [70000, 0, 0]);
        assert!(plan_float_linear(QuantType::Q4_0, 1, 1, 1, 0, 1, false, 1.0).is_err());
        assert!(plan_float_linear(QuantType::F32, 1 << 20, 1 << 13, 1, 0, 1, false, 1.0).is_err());
    }
}
