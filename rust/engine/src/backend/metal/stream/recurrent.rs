//! Gated DeltaNet layers (`Stream::delta_layer`).

use super::exec::{dev, invalid};
use super::MetalStream;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::ops::recurrent_plan::{
    delta_supported, plan_delta_layer, plan_mamba2_scan, plan_mamba_conv, plan_mamba_gate_norm, Mamba2Shape, MixedBuffer, RunSpec,
};
use crate::backend::{BackendError, DeltaRun, DeltaShape, DeltaWeights, DeviceBuffer, MambaShape, MambaWeights};

/// The planner's view of a [`MambaShape`].
pub(super) fn mamba2_shape(shape: &MambaShape) -> Mamba2Shape {
    Mamba2Shape {
        inner: shape.inner,
        state: shape.state,
        heads: shape.heads,
        groups: shape.groups,
        conv_kernel: shape.conv_kernel,
        norm_groups: shape.norm_groups,
        step_min: shape.step_min,
        eps: shape.eps,
    }
}

impl MetalStream {
    /// Mamba-2 mixer after the input projection: per run a convolution dispatch (into `mixed`) and a selective-scan dispatch (into `out`), then the gated
    /// group norm over all rows in place on `out`. The layer's state (convolution history, then the recurrent state) starts `state_offset` bytes into
    /// each run's state buffer and is updated in place.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_mamba2_layer(
        &mut self,
        shape: &MambaShape,
        weights: &MambaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        projected: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let planner_shape = mamba2_shape(shape);
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let (conv_dim, width) = (planner_shape.conv_dim(), planner_shape.proj_width());
        let state_end = state_offset + shape.layer_state_bytes();
        if !state_offset.is_multiple_of(4)
            || projected.size < rows * width * 4
            || mixed.size < rows * conv_dim * 4
            || out.size < rows * shape.inner * 4
            || weights.conv.size < conv_dim * shape.conv_kernel * 4
            || weights.conv_bias.is_some_and(|b| b.size < conv_dim * 4)
            || weights.dt_bias.size < shape.heads * 4
            || weights.decay.size < shape.heads * 4
            || weights.skip.size < shape.heads * 4
            || weights.norm.size < shape.inner * 4
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("Mamba-2 buffers smaller than the layer needs"));
        }
        let (history, _) = planner_shape.state_floats();
        let hist_off = state_offset / 4;
        let rec_off = hist_off + history;
        let mut plans = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            if run.rows == 0 {
                continue;
            }
            let spec = RunSpec { first_row: run.first_row, rows: run.rows, reset: run.reset };
            plans.push(plan_mamba_conv(
                i,
                &spec,
                conv_dim,
                shape.conv_kernel,
                width,
                shape.inner,
                hist_off,
                weights.conv_bias.is_some(),
                MixedBuffer::Caller,
            )?);
            plans.push(plan_mamba2_scan(
                i,
                &spec,
                &planner_shape,
                width,
                shape.inner + conv_dim,
                rec_off,
                MixedBuffer::Caller,
            )?);
        }
        plans.push(plan_mamba_gate_norm(&planner_shape, rows, width, 0)?);
        let mut states = Vec::with_capacity(runs.len());
        for run in runs {
            let range = buffer_ref(run.state)?;
            if !range.offset.is_multiple_of(4) {
                return Err(invalid("Mamba state buffer is not 4-byte aligned"));
            }
            states.push(range);
        }
        let mut named = vec![
            ("x", dev(projected)?),
            ("projected", dev(projected)?),
            ("convw", dev(weights.conv)?),
            ("mixed", dev(mixed)?),
            ("dt_bias", dev(weights.dt_bias)?),
            ("decay", dev(weights.decay)?),
            ("skip", dev(weights.skip)?),
            ("norm", dev(weights.norm)?),
            ("out", dev(out)?),
            ("y", dev(out)?),
        ];
        if let Some(bias) = weights.conv_bias {
            named.push(("convb", dev(bias)?));
        }
        self.run_plans(&plans, &named, &states)
    }

    /// One layer for `runs` (consecutive rows of one sequence each). `inputs` = `[qkv, z, beta, alpha]`. Per run a convolution dispatch
    /// (into `mixed`) and a recurrent-update dispatch (into `out`); the state of the layer starts `state_offset` bytes into each run's state
    /// buffer and is updated in place.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_delta_layer(
        &mut self,
        shape: &DeltaShape,
        weights: &DeltaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        inputs: [&DeviceBuffer; 4],
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let [qkv, z, beta, alpha] = inputs;
        if !delta_supported(shape) {
            return Err(BackendError::UnsupportedOperation(
                "delta_layer for this shape".into(),
            ));
        }
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let (conv_dim, value_dim) = (shape.conv_dim(), shape.value_dim());
        let state_end = state_offset + shape.layer_state_bytes();
        if qkv.size < rows * conv_dim * 4
            || mixed.size < rows * conv_dim * 4
            || z.size < rows * value_dim * 4
            || out.size < rows * value_dim * 4
            || beta.size < rows * shape.v_heads * 4
            || alpha.size < rows * shape.v_heads * 4
            || weights.conv.size < conv_dim * shape.conv_kernel * 4
            || weights.dt_bias.size < shape.v_heads * 4
            || weights.decay.size < shape.v_heads * 4
            || weights.norm.size < shape.v_dim * 4
            || runs
                .iter()
                .any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("DeltaNet buffers smaller than the layer needs"));
        }
        let specs: Vec<RunSpec> = runs
            .iter()
            .map(|r| RunSpec {
                first_row: r.first_row,
                rows: r.rows,
                reset: r.reset,
            })
            .collect();
        let plans = plan_delta_layer(shape, &specs, state_offset)?;
        if plans.is_empty() {
            return Ok(());
        }
        let mut states = Vec::with_capacity(runs.len());
        for run in runs {
            let range = buffer_ref(run.state)?;
            if !range.offset.is_multiple_of(4) {
                return Err(invalid("DeltaNet state buffer is not 4-byte aligned"));
            }
            states.push(range);
        }
        self.run_plans(
            &plans,
            &[
                ("qkv", dev(qkv)?),
                ("conv", dev(weights.conv)?),
                ("mixed", dev(mixed)?),
                ("z", dev(z)?),
                ("beta", dev(beta)?),
                ("alpha", dev(alpha)?),
                ("dt_bias", dev(weights.dt_bias)?),
                ("decay", dev(weights.decay)?),
                ("norm", dev(weights.norm)?),
                ("out", dev(out)?),
            ],
            &states,
        )
    }
}
