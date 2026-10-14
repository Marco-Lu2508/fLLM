//! Gated DeltaNet layers (`Stream::delta_layer`).

use super::exec::{dev, invalid};
use super::MetalStream;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::ops::recurrent_plan::{delta_supported, plan_delta_layer, RunSpec};
use crate::backend::{BackendError, DeltaRun, DeltaShape, DeltaWeights, DeviceBuffer};

impl MetalStream {
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
