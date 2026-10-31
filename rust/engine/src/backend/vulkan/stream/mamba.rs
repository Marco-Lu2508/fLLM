//! Mamba-2 mixer of the Vulkan stream, prepared for the trait method proposed in the package-3 report (`Stream::mamba_layer`).
//!
//! The trait has no such method yet, so nothing calls [`VulkanStream::mamba2_layer`]; once the integrator adds `MambaShape` / `MambaWeights` /
//! `Stream::supports_mamba` / `Stream::mamba_layer` to `backend/mod.rs`, the Vulkan implementation is a conversion of the arguments and one
//! call (see the package-4 report). The planning and the shaders exist (`ops::recurrent::plan_mamba2_mixer`, `mamba_conv`, `mamba_scan2_*`,
//! `mamba_gate_norm`).

use super::exec::Arg;
use super::VulkanStream;
use crate::backend::vulkan::ops::recurrent::{plan_mamba2_mixer_scan, Mamba2Shape, RunSpec};
use crate::backend::{BackendError, DeviceBuffer};

/// Device-resident parameters of a Mamba-2 layer (all FP32).
pub struct MambaBufs<'a> {
    /// `conv_dim x conv_kernel`
    pub conv: &'a DeviceBuffer,
    pub conv_bias: Option<&'a DeviceBuffer>,
    pub dt_bias: &'a DeviceBuffer,
    /// Negative decay rates A.
    pub decay: &'a DeviceBuffer,
    /// Skip connection D.
    pub skip: &'a DeviceBuffer,
    /// Gated group-norm weight (`inner`).
    pub norm: &'a DeviceBuffer,
}

impl VulkanStream {
    /// Mamba-2 mixer after the input projection: `projected` (`rows x proj_width`: gate, xBC, dt) of the runs; `out` (`rows x inner`) receives the
    /// gated, normalised scan output (the caller applies the output projection). The layer's state (convolution history, then recurrent state) starts
    /// `state_offset` bytes into each run's state buffer (`runs[i].1`). `mixed` is scratch of `rows x conv_dim` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_mixer(
        &mut self,
        shape: &Mamba2Shape,
        weights: &MambaBufs<'_>,
        runs: &[(RunSpec, &DeviceBuffer)],
        state_offset: usize,
        projected: &DeviceBuffer,
        mixed: &DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let width = shape.proj_width();
        if projected.size == 0 || !projected.size.is_multiple_of(width * 4) {
            return Err(BackendError::InvalidShape(format!(
                "Mamba-2 projection of {} bytes is not rows x {width} floats",
                projected.size
            )));
        }
        let rows = projected.size / (width * 4);
        let conv_dim = shape.conv_dim();
        let state_end = state_offset + shape.layer_state_bytes();
        if out.size < rows * shape.inner * 4
            || mixed.size < rows * conv_dim * 4
            || weights.conv.size < conv_dim * shape.conv_kernel * 4
            || weights.conv_bias.is_some_and(|b| b.size < conv_dim * 4)
            || weights.dt_bias.size < shape.heads * 4
            || weights.decay.size < shape.heads * 4
            || weights.skip.size < shape.heads * 4
            || weights.norm.size < shape.inner * 4
            || runs
                .iter()
                .any(|(spec, state)| state.size < state_end || spec.first_row + spec.rows > rows)
        {
            return Err(BackendError::InvalidShape(
                "Mamba-2 buffers are smaller than the layer needs".into(),
            ));
        }
        let specs: Vec<RunSpec> = runs.iter().map(|(spec, _)| *spec).collect();
        let plans = plan_mamba2_mixer_scan(shape, &specs, rows, state_offset, self.scan_shared())?;
        // a layer without a convolution bias reads zeros
        let zeros = match weights.conv_bias {
            Some(_) => None,
            None => {
                let buffer = self.dev.alloc(&self.sh, (conv_dim * 4) as u64)?;
                self.rec.fill(&buffer, 0, (conv_dim * 4).next_multiple_of(4) as u64, 0)?;
                Some(buffer)
            }
        };
        // a buffer named twice (the projection is both `x` and `projected`, the output both `out` and `y`) is staged once by run_plans
        let mut args = vec![
            Arg::r("x", projected),
            Arg::r("projected", projected),
            Arg::r("convw", weights.conv),
            Arg::r("dt_bias", weights.dt_bias),
            Arg::r("decay", weights.decay),
            Arg::r("skip", weights.skip),
            Arg::r("norm", weights.norm),
            Arg::w("out", out),
            Arg::w("y", out),
            Arg::w("mixed", mixed),
        ];
        match (weights.conv_bias, zeros) {
            (Some(bias), _) => args.push(Arg::r("convb", bias)),
            (None, Some(buffer)) => args.push(Arg::view("convb", buffer)),
            (None, None) => {}
        }
        let states: Vec<&DeviceBuffer> = runs.iter().map(|(_, state)| *state).collect();
        self.run_plans(&plans, &args, &states, false)
    }
}
