//! Mixture-of-experts: routing, expert products through address tables, combination, and the row gather / scatter of grouped execution.

use super::exec::{dev, invalid, table_of, Bound};
use super::MetalStream;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::buffers::AddressTable;
use crate::backend::metal::ops::moe_plan::{
    moe_linear_supported, plan_gather_rows, plan_moe_combine, plan_moe_expert_gemm, plan_moe_linear,
    plan_moe_route, plan_moe_topk, plan_scatter_add_rows, ExpertAccess, MoeLinear,
};
use crate::backend::{BackendError, DeviceBuffer, DeviceWeight, QuantType};
use std::sync::Arc;

impl MetalStream {
    /// Softmax routing: the `k` best experts of every row with their (renormalised, scaled) scores.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_moe_topk(
        &mut self,
        logits: &DeviceBuffer,
        ids: &mut DeviceBuffer,
        weights: &mut DeviceBuffer,
        rows: usize,
        experts: usize,
        k: usize,
        normalize: bool,
        scale: f32,
    ) -> Result<(), BackendError> {
        if logits.size < rows * experts * 4 || ids.size < rows * k * 4 || weights.size < rows * k * 4 {
            return Err(invalid("moe_topk: buffers too small"));
        }
        let plan = plan_moe_topk(rows, experts, k, normalize, scale)?;
        self.run_plans(
            &[plan],
            &[("logits", dev(logits)?), ("ids", dev(ids)?), ("weights", dev(weights)?)],
            &[],
        )
    }

    /// `Stream::moe_route`: normalised top-k routing, the row count taken from the size of the logits.
    pub(super) fn op_moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        if num_experts == 0 || top_k == 0 || top_k > num_experts {
            return Err(invalid("moe_route expert counts invalid"));
        }
        let rows = router_logits.size / 4 / num_experts;
        if rows == 0 || weights_out.size < rows * top_k * 4 || indices_out.size < rows * top_k * 4 {
            return Err(invalid("moe_route buffers smaller than rows x top_k"));
        }
        let plan = plan_moe_route(rows, num_experts, top_k)?;
        self.run_plans(
            &[plan],
            &[
                ("logits", dev(router_logits)?),
                ("indices", dev(indices_out)?),
                ("weights", dev(weights_out)?),
            ],
            &[],
        )
    }

    /// `out[slot] = W[ids[slot]] x[slot / x_div]` for the experts of an address table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_moe_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        table: &DeviceBuffer,
        bias_table: Option<&DeviceBuffer>,
        ids: &DeviceBuffer,
        quant: QuantType,
        rows: usize,
        cols: usize,
        slots: usize,
        x_div: usize,
    ) -> Result<(), BackendError> {
        if !moe_linear_supported(quant) || x_div == 0 {
            return Err(BackendError::UnsupportedQuant(quant));
        }
        let inputs = slots.div_ceil(x_div);
        if x.size < inputs * cols * 4 || out.size < slots * rows * 4 || ids.size < slots * 4 {
            return Err(invalid("moe_linear: buffers too small"));
        }
        let experts = table_of(table)?;
        let plan = plan_moe_linear(&MoeLinear {
            quant,
            rows,
            cols,
            slots,
            x_div,
            n_experts: experts.len(),
            has_bias: bias_table.is_some(),
            access: ExpertAccess::Address,
        })?;
        let mut named = vec![
            ("table", Bound::Table(experts)),
            ("ids", dev(ids)?),
            ("x", dev(x)?),
            ("out", dev(out)?),
        ];
        if let Some(bias_table) = bias_table {
            named.push(("bias_table", Bound::Table(table_of(bias_table)?)));
        }
        self.run_plans(&[plan], &named, &[])
    }

    pub(super) fn op_moe_combine(
        &mut self,
        out: &mut DeviceBuffer,
        y: &DeviceBuffer,
        weights: &DeviceBuffer,
        tokens: usize,
        k: usize,
        hidden: usize,
    ) -> Result<(), BackendError> {
        if out.size < tokens * hidden * 4 || y.size < tokens * k * hidden * 4 || weights.size < tokens * k * 4 {
            return Err(invalid("moe_combine: buffers too small"));
        }
        let plan = plan_moe_combine(tokens, k, hidden)?;
        self.run_plans(
            &[plan],
            &[("out", dev(out)?), ("y", dev(y)?), ("weights", dev(weights)?)],
            &[],
        )
    }

    /// `Stream::moe_expert_gemm` for experts of one shape and format: an address table made on the spot, `moe_linear` into scratch, `moe_combine`.
    pub(super) fn op_moe_expert_gemm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        experts: &[DeviceWeight],
        indices: &DeviceBuffer,
        scores: &DeviceBuffer,
        top_k: usize,
    ) -> Result<(), BackendError> {
        let first = experts
            .first()
            .ok_or_else(|| invalid("moe_expert_gemm without experts"))?;
        let (width, hidden) = (first.rows, first.cols);
        if hidden == 0 || top_k == 0 || !x.size.is_multiple_of(hidden * 4) {
            return Err(invalid("moe_expert_gemm input is not rows of the expert width"));
        }
        if experts
            .iter()
            .any(|e| e.quant_type != first.quant_type || e.rows != width || e.cols != hidden)
        {
            return Err(invalid("moe experts differ in shape"));
        }
        if !moe_linear_supported(first.quant_type) {
            return Err(BackendError::UnsupportedQuant(first.quant_type));
        }
        let rows = x.size / (hidden * 4);
        if indices.size < rows * top_k * 4 || scores.size < rows * top_k * 4 || out.size < rows * width * 4 {
            return Err(invalid("moe routing buffers smaller than rows x top_k"));
        }
        let mut entries = Vec::with_capacity(experts.len());
        for expert in experts {
            if expert.quant_type == QuantType::Mxfp4 && !self.shared.is_raw_mxfp4(expert) {
                return Err(BackendError::UnsupportedOperation(format!(
                    "{}: MXFP4 experts must be uploaded under the name of a routed expert (blk.N.expK.*)",
                    expert.name
                )));
            }
            entries.push(buffer_ref(&expert.buffer)?);
        }
        let table = Arc::new(AddressTable::build(&self.shared.ctx.memory, &entries)?);
        let plans = plan_moe_expert_gemm(
            &MoeLinear {
                quant: first.quant_type,
                rows: width,
                cols: hidden,
                slots: rows * top_k,
                x_div: top_k,
                n_experts: experts.len(),
                has_bias: false,
                access: ExpertAccess::Address,
            },
            rows,
            top_k,
        )?;
        self.run_plans(
            &plans,
            &[
                ("table", Bound::Table(table)),
                ("expert_indices", dev(indices)?),
                ("x", dev(x)?),
                ("out", dev(out)?),
                ("expert_weights", dev(scores)?),
            ],
            &[],
        )
    }

    /// `dst[i] = src[idx[i]] * weights[i]` for `idx.len()` rows of `row_len` floats.
    pub(super) fn op_gather_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        let n = idx.len();
        if n == 0 {
            return Ok(());
        }
        if weights.is_some_and(|w| w.len() != n) || dst.size < n * row_len * 4 {
            return Err(invalid("gather_rows: operand sizes do not match"));
        }
        let max = idx.iter().copied().max().unwrap_or(0);
        if idx.iter().any(|&i| i < 0) || (max as usize + 1) * row_len * 4 > src.size {
            return Err(invalid("gather_rows: row index out of range"));
        }
        let plan = plan_gather_rows(idx, weights, row_len)?;
        self.run_plans(&[plan], &[("dst", dev(dst)?), ("src", dev(src)?)], &[])
    }

    /// `dst[idx[i]] += src[i] * weights[i]`; the indices of one call are distinct.
    pub(super) fn op_scatter_add_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        let n = idx.len();
        if n == 0 {
            return Ok(());
        }
        if weights.is_some_and(|w| w.len() != n) || src.size < n * row_len * 4 {
            return Err(invalid("scatter_add_rows: operand sizes do not match"));
        }
        let max = idx.iter().copied().max().unwrap_or(0);
        if idx.iter().any(|&i| i < 0) || (max as usize + 1) * row_len * 4 > dst.size {
            return Err(invalid("scatter_add_rows: row index out of range"));
        }
        let plan = plan_scatter_add_rows(idx, weights, row_len)?;
        self.run_plans(&[plan], &[("dst", dev(dst)?), ("src", dev(src)?)], &[])
    }
}
