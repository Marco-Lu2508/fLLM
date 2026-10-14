//! Embedding, norms, RoPE, activations, residual add, attention gating, argmax and sampling.

use super::exec::{dev, invalid, Bound};
use super::MetalStream;
use crate::backend::metal::backend::convert::raw_row_bytes;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::ops::embedding::{embed_kernel, plan_embedding, EmbeddingTable, TableLayout};
use crate::backend::metal::ops::{
    plan_argmax, plan_glu, plan_layer_norm, plan_residual_add, plan_rms_norm, plan_rope,
    plan_sigmoid_gate_mul, plan_split_q_gate, GluMode,
};
use crate::backend::{
    warn_fallback_reason, BackendError, DeviceBuffer, DeviceWeight, QuantType, RopeParams,
    SamplingParams, Stream,
};

impl MetalStream {
    pub(super) fn op_embedding(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError> {
        if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= weight.rows) {
            return Err(invalid(format!("embedding token id {id} out of bounds")));
        }
        if ids.is_empty() {
            return Ok(());
        }
        if out.size < ids.len() * weight.cols * 4 {
            return Err(invalid("embedding output smaller than ids x hidden"));
        }
        // MXFP4 matrix weights are plane-sorted; the raw routed-expert layout is the file layout
        let planar = weight.quant_type == QuantType::Mxfp4 && !self.shared.is_raw_mxfp4(weight);
        let table = EmbeddingTable {
            quant: weight.quant_type,
            rows: weight.rows,
            cols: weight.cols,
            layout: if planar {
                TableLayout::Planar
            } else {
                TableLayout::Gguf
            },
        };
        // the kernels do not apply a per-tensor scale; those tables and the formats without a kernel are decoded on the host
        if weight.scale == 1.0
            && weight.perm.is_none()
            && embed_kernel(weight.quant_type).is_some()
            && table.row_bytes().is_some()
        {
            let plan = plan_embedding(&table, ids)?;
            let weights = Bound::Buf(buffer_ref(&weight.buffer)?);
            return self.run_plans(&[plan], &[("weights", weights), ("out", dev(out)?)], &[]);
        }
        self.embedding_on_host(out, ids, weight, planar)
    }

    /// Rows of a table the embedding kernels cannot read (no kernel for the format, a per-tensor scale): decoded on the host and written to `out`.
    fn embedding_on_host(
        &mut self,
        out: &DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
        planar: bool,
    ) -> Result<(), BackendError> {
        let quant = weight.quant_type;
        let row_bytes = if planar {
            None
        } else {
            raw_row_bytes(quant, weight.cols)
        }
        .ok_or(BackendError::UnsupportedQuant(quant))?;
        let source = buffer_ref(&weight.buffer)?;
        let kind = quant.to_gguf_type_id();
        let mut row = vec![0u8; row_bytes];
        let mut bytes = Vec::with_capacity(ids.len() * weight.cols * 4);
        for &id in ids {
            source.read(id as usize * row_bytes, &mut row)?;
            let values = crate::compute::dequantize(&row, kind, weight.cols).map_err(|e| {
                BackendError::UnsupportedOperation(format!(
                    "{}: embedding rows of {quant:?} cannot be decoded on the host ({e})",
                    weight.name
                ))
            })?;
            for value in values {
                bytes.extend_from_slice(&(value * weight.scale).to_le_bytes());
            }
        }
        let range = buffer_ref(out)?;
        self.write_device(&range, &bytes)
    }

    pub(super) fn op_rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("rms_norm input is not rows of the weight width"));
        }
        let rows = x.size / (cols * 4);
        let plan = plan_rms_norm(rows, cols, eps, true)?;
        self.run_plans(
            &[plan],
            &[("x", dev(x)?), ("weight", dev(weight)?), ("out", dev(out)?)],
            &[],
        )
    }

    pub(super) fn op_layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("layer_norm input is not rows of the weight width"));
        }
        if bias.is_some_and(|b| b.size < cols * 4) {
            return Err(invalid("layer_norm bias width differs from weight"));
        }
        let rows = x.size / (cols * 4);
        let plan = plan_layer_norm(rows, cols, eps, bias.is_some())?;
        let mut named = vec![("x", dev(x)?), ("weight", dev(weight)?), ("out", dev(out)?)];
        if let Some(bias) = bias {
            named.push(("bias", dev(bias)?));
        }
        self.run_plans(&[plan], &named, &[])
    }

    pub(super) fn op_rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError> {
        params.validate(positions, q.size, k.size)?;
        if positions.is_empty() {
            return Ok(());
        }
        let table_rows = params.cos.len() / (params.rope_dim / 2);
        let cos = self.shared.table(&params.cos)?;
        let sin = self.shared.table(&params.sin)?;
        let plan = plan_rope(
            positions,
            params.heads,
            params.kv_heads,
            params.dim,
            params.rope_dim,
            params.neox,
            table_rows,
        )?;
        self.run_plans(
            &[plan],
            &[
                ("q", dev(q)?),
                ("k", dev(k)?),
                ("cos", Bound::Buf(cos)),
                ("sin", Bound::Buf(sin)),
            ],
            &[],
        )
    }

    /// SwiGLU, GeGLU and the gpt-oss variant: `out = act(gate) * up` over `gate.size / 4` values.
    pub(super) fn op_glu(
        &mut self,
        mode: GluMode,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let count = gate.size / 4;
        if count == 0 || up.size < count * 4 || out.size < count * 4 {
            return Err(invalid("gated activation operands differ in size"));
        }
        let plan = plan_glu(mode, count)?;
        self.run_plans(
            &[plan],
            &[("gate", dev(gate)?), ("up", dev(up)?), ("out", dev(out)?)],
            &[],
        )
    }

    pub(super) fn op_residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 || residual.size < count * 4 {
            return Err(invalid("residual_add operands differ in size"));
        }
        let plan = plan_residual_add(count, scale)?;
        self.run_plans(&[plan], &[("x", dev(x)?), ("residual", dev(residual)?)], &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_split_q_gate(
        &mut self,
        q: &mut DeviceBuffer,
        gate: &mut DeviceBuffer,
        projected: &DeviceBuffer,
        rows: usize,
        heads: usize,
        dk: usize,
        per_head: bool,
    ) -> Result<(), BackendError> {
        let width = rows * heads * dk * 4;
        if width == 0 || projected.size < 2 * width || q.size < width || gate.size < width {
            return Err(invalid("split_q_gate buffers smaller than rows x heads x dk"));
        }
        let plan = plan_split_q_gate(rows, heads, dk, per_head)?;
        self.run_plans(
            &[plan],
            &[
                ("projected", dev(projected)?),
                ("q", dev(q)?),
                ("gate", dev(gate)?),
            ],
            &[],
        )
    }

    pub(super) fn op_sigmoid_gate_mul(
        &mut self,
        x: &mut DeviceBuffer,
        gate: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 || gate.size < count * 4 {
            return Err(invalid("sigmoid_gate_mul operands differ in size"));
        }
        let plan = plan_sigmoid_gate_mul(count)?;
        self.run_plans(&[plan], &[("x", dev(x)?), ("gate", dev(gate)?)], &[])
    }

    /// Index of the largest of `vocab` logits for each of `rows` rows (lowest index on a tie), read back to the host.
    pub(super) fn op_argmax(
        &mut self,
        logits: &DeviceBuffer,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<i32>, BackendError> {
        self.ensure_headroom()?;
        if rows == 0 {
            return Ok(Vec::new());
        }
        if vocab == 0 || logits.size < rows * vocab * 4 {
            return Err(invalid("argmax logits smaller than rows x vocab"));
        }
        let plan = plan_argmax(rows, vocab)?;
        let result = self.scratch(rows * 4)?;
        self.run_plans(
            &[plan],
            &[("logits", dev(logits)?), ("out", Bound::Buf(result.clone()))],
            &[],
        )?;
        self.sync()?;
        let mut bytes = vec![0u8; rows * 4];
        result.read(0, &mut bytes)?;
        Ok(bytes
            .chunks(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
            .collect())
    }

    /// Greedy sampling on the device (`argmax`); sampling with a temperature runs on the host through the CPU backend.
    pub(super) fn op_sampling(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        if params.temperature <= 0.0 && params.rows > 0 && params.vocab_size > 0 {
            let ids = self.op_argmax(logits, params.rows, params.vocab_size)?;
            if token_ids_out.len() < ids.len() {
                return Err(invalid("sampling output smaller than rows"));
            }
            for (slot, id) in token_ids_out.iter_mut().zip(ids) {
                *slot = id.max(0) as u32;
            }
            return Ok(());
        }
        warn_fallback_reason("sampling_logits", "sampling with a temperature runs on the host");
        self.ensure_headroom()?;
        let host = self.copy_to_host(logits)?;
        self.cpu_stream.sampling_logits(token_ids_out, &host, params)
    }
}
