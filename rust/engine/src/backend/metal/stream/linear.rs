//! `quantized_linear` (and the fused residual form): one planner call, one or two dispatches.

use super::exec::invalid;
use super::MetalStream;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::buffers::BufferRef;
use crate::backend::metal::weights::{plan_quantized_linear, PlanError, WeightError};
use crate::backend::{BackendError, DeviceBuffer, DeviceWeight, QuantType};

/// Activation and prepass buffers are read as 16-byte vectors.
const ACTIVATION_ALIGN: usize = 16;

fn plan_error(weight: &DeviceWeight, error: PlanError) -> BackendError {
    match error {
        PlanError::Weight(WeightError::UnsupportedFormat(quant)) => BackendError::UnsupportedQuant(quant),
        PlanError::Device(message) => BackendError::UnsupportedOperation(format!("{}: {message}", weight.name)),
        other => invalid(format!("{}: {other}", weight.name)),
    }
}

impl MetalStream {
    /// `out = x * weight (+ bias)`, or `out += ...` with `residual`: `batch = x.size / (4 * cols)` rows.
    ///
    /// `bias` is the caller's bias only (`weight.bias` is applied when the caller passes it). `x` must not overlap `out`.
    pub(super) fn op_linear(
        &mut self,
        out: &DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
        residual: bool,
    ) -> Result<(), BackendError> {
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
            return Err(invalid("linear input dimensions do not match"));
        }
        let batch = x.size / (cols * 4);
        if batch == 0 {
            return Ok(());
        }
        if out.size < batch * rows * 4 {
            return Err(invalid("linear output smaller than batch x rows"));
        }
        if bias.is_some_and(|b| b.size < rows * 4) {
            return Err(invalid("linear bias shorter than the output rows"));
        }
        let weights = self.weight_range(weight)?;
        let dense = matches!(weight.quant_type, QuantType::F32 | QuantType::F16 | QuantType::BF16);
        let weight_align = if dense { 16 } else { 4 };
        if !weights.offset.is_multiple_of(weight_align) {
            return Err(BackendError::BufferError(format!(
                "{}: weight image at byte {} is not {weight_align}-byte aligned",
                weight.name, weights.offset
            )));
        }
        let out_range = buffer_ref(out)?;
        if !out_range.offset.is_multiple_of(4) {
            return Err(BackendError::BufferError(format!(
                "linear output at byte {} is not 4-byte aligned",
                out_range.offset
            )));
        }
        let bias_range = match bias {
            Some(b) => {
                let range = buffer_ref(b)?;
                if !range.offset.is_multiple_of(4) {
                    return Err(BackendError::BufferError("linear bias is not 4-byte aligned".into()));
                }
                range
            }
            None => self.shared.dummy.clone(),
        };
        let mut input = self.aligned(buffer_ref(x)?, ACTIVATION_ALIGN)?;

        // plan; if the pipeline of the chosen kernel cannot run the planned threadgroup, plan again with its real limit
        let mut caps = self.shared.caps;
        let mut tries = 0;
        let (plan, pipeline) = loop {
            let mut plan = plan_quantized_linear(weight.quant_type, rows, cols, batch, &caps)
                .map_err(|e| plan_error(weight, e))?;
            if bias.is_some() {
                plan = plan.with_bias();
            }
            plan = plan.with_scale(weight.scale);
            if residual {
                plan = plan.with_residual();
            }
            let pipeline = self.pipeline(&plan.kernel_name, &plan.constants)?;
            let threads: usize = plan.threads_per_group.iter().map(|&t| t as usize).product();
            if threads <= pipeline.max_threads() {
                break (plan, pipeline);
            }
            tries += 1;
            let limit = u32::try_from(pipeline.max_threads()).unwrap_or(u32::MAX);
            if tries > 2 || limit >= caps.max_threads {
                return Err(BackendError::UnsupportedOperation(format!(
                    "{}: kernel {} needs {threads} threads per group, the pipeline allows {}",
                    weight.name,
                    plan.kernel_name,
                    pipeline.max_threads()
                )));
            }
            caps.max_threads = limit;
        };
        self.shared.flush_residency();

        // act-order weights: the activations are permuted into a dense scratch copy first
        if let Some(pre) = &plan.activation_prepass {
            let perm = weight.perm.as_ref().ok_or_else(|| {
                invalid(format!("{}: act-order weight without its permutation table", weight.name))
            })?;
            let perm_range = buffer_ref(perm)?;
            let permuted = self.scratch(pre.output_bytes)?;
            let prepass = self.pipeline(pre.kernel_name, &[])?;
            let (groups, threads) = pre.dispatch_dims();
            let mut dispatch = self.rec.begin(&prepass)?;
            dispatch
                .read(pre.slots[0], &input)
                .write(pre.slots[1], &permuted)
                .read(pre.slots[2], &perm_range)
                .bytes(pre.slots[3], &pre.params);
            dispatch.threadgroups(groups, threads)?;
            input = permuted;
        }

        let (groups, threads) = plan.dispatch_dims();
        let mut dispatch = self.rec.begin(&pipeline)?;
        dispatch.read(plan.slots.weights, &weights).read(plan.slots.x, &input);
        if plan.reads_output() {
            dispatch.read_write(plan.slots.out, &out_range);
        } else {
            dispatch.write(plan.slots.out, &out_range);
        }
        dispatch
            .read(plan.slots.bias, &bias_range)
            .bytes(plan.slots.params, &plan.params);
        for (slot, bytes) in &plan.threadgroup_memory {
            dispatch.threadgroup_memory(*slot, *bytes as usize);
        }
        dispatch.threadgroups(groups, threads)?;
        Ok(())
    }

    /// The device image the kernels read for `weight`: its own buffer, except for a routed-expert MXFP4 weight (raw layout for the MoE kernel), which
    /// gets its plane-sorted twin.
    fn weight_range(&self, weight: &DeviceWeight) -> Result<BufferRef, BackendError> {
        if self.shared.is_raw_mxfp4(weight) {
            self.shared.planar_twin(weight)
        } else {
            buffer_ref(&weight.buffer)
        }
    }

    /// `range` itself if it starts at a multiple of `align`, else a copy in scratch memory (which starts at multiples of 256).
    fn aligned(&mut self, range: BufferRef, align: usize) -> Result<BufferRef, BackendError> {
        if range.offset.is_multiple_of(align) {
            return Ok(range);
        }
        let copy = self.scratch(range.len)?;
        self.rec.copy_buffer(&range, &copy, range.len)?;
        Ok(copy)
    }
}
