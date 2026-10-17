//! Element-wise operations of the less common architectures (plain and gated activations beyond SwiGLU / GeGLU, clamping, the softplus attention
//! gate) and the data movement of multi-head latent attention (`Stream::supports_arch_ops`).

use super::exec::{dev, invalid};
use super::MetalStream;
use crate::backend::metal::ops::arch_plan::{plan_clamp, plan_mla_assemble, plan_mla_latent, plan_softplus_gate_mul};
use crate::backend::metal::ops::{plan_glu, GluMode};
use crate::backend::{BackendError, DeviceBuffer, MlaShape};

impl MetalStream {
    /// `out = act(x)` over `x.size / 4` values; `mode` as in `Stream::activation`.
    pub(super) fn op_activation(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        let glu = GluMode::from_activation(mode, false).ok_or_else(|| invalid(format!("unknown activation mode {mode}")))?;
        let count = x.size / 4;
        if count == 0 || out.size < count * 4 {
            return Err(invalid("activation output smaller than its input"));
        }
        let plan = plan_glu(glu, count)?;
        self.run_plans(&[plan], &[("gate", dev(x)?), ("out", dev(out)?)], &[])
    }

    /// `out = act(gate) * up`; `mode` as in `Stream::activation`.
    pub(super) fn op_activation_glu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        mode: i32,
    ) -> Result<(), BackendError> {
        let glu = GluMode::from_activation(mode, true).ok_or_else(|| invalid(format!("unknown activation mode {mode}")))?;
        self.op_glu(glu, out, gate, up)
    }

    pub(super) fn op_clamp(&mut self, x: &mut DeviceBuffer, limit: f32) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 || !limit.is_finite() || limit < 0.0 {
            return Err(invalid("clamp needs a non-empty tensor and a finite non-negative limit"));
        }
        let plan = plan_clamp(count, limit)?;
        self.run_plans(&[plan], &[("x", dev(x)?)], &[])
    }

    pub(super) fn op_softplus_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer, beta: f32) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 || gate.size < count * 4 {
            return Err(invalid("softplus_gate_mul operands differ in size"));
        }
        let plan = plan_softplus_gate_mul(count, beta)?;
        self.run_plans(&[plan], &[("x", dev(x)?), ("gate", dev(gate)?)], &[])
    }

    pub(super) fn op_mla_latent(
        &mut self,
        latent: &mut DeviceBuffer,
        kv_a: &DeviceBuffer,
        rows: usize,
        kv_lora: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 || kv_a.size < rows * width * 4 || latent.size < rows * kv_lora * 4 {
            return Err(invalid("mla_latent buffers smaller than rows x width"));
        }
        let plan = plan_mla_latent(rows, kv_lora, width)?;
        self.run_plans(&[plan], &[("kv_a", dev(kv_a)?), ("latent", dev(latent)?)], &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_mla_assemble(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        v: &mut DeviceBuffer,
        q_full: &DeviceBuffer,
        kv_a: &DeviceBuffer,
        kv: &DeviceBuffer,
        rows: usize,
        shape: &MlaShape,
    ) -> Result<(), BackendError> {
        let dk = shape.nope + shape.rope;
        let pairs = rows * shape.heads;
        if rows == 0
            || q_full.size < pairs * dk * 4
            || kv_a.size < rows * (shape.kv_lora + shape.rope) * 4
            || kv.size < pairs * (shape.nope + shape.value) * 4
            || q.size < pairs * dk * 4
            || k.size < pairs * dk * 4
            || v.size < pairs * shape.value * 4
        {
            return Err(invalid("mla_assemble buffers smaller than the MLA shape needs"));
        }
        let plan = plan_mla_assemble(rows, shape)?;
        self.run_plans(
            &[plan],
            &[
                ("q_full", dev(q_full)?),
                ("kv_a", dev(kv_a)?),
                ("kv", dev(kv)?),
                ("q", dev(q)?),
                ("k", dev(k)?),
                ("v", dev(v)?),
            ],
            &[],
        )
    }
}
