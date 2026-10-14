//! `Stream` of the TPU backend: every operation is a StableHLO program built by `ops` and run on the device.
//!
//! PJRT executes asynchronously: an operation returns when its program is enqueued, `synchronize` and every read of data wait for the
//! device. Operations that the TPU backend does not have (`delta_layer`, `host_linear`, expert GEMMs through pointer tables, the
//! CUDA-graph and half-precision fusion hooks) keep the trait's defaults and report `UnsupportedOperation`, which makes the decoder
//! use its host or generic path.
//!
//! Level of verification: L0.

use super::mem::{driver, Shared};
use super::ops::{self, Glu};
use crate::backend::{
    AttentionParams, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight, RopeParams,
    SamplingParams, Stream,
};
use std::sync::Arc;

/// Run an operation, turning a panic of the program emitter (a shape rule that an unexpected argument violates) into an error.
fn guarded<T>(f: impl FnOnce() -> Result<T, BackendError>) -> Result<T, BackendError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_else(|| "unknown panic".to_owned());
        Err(driver(format!("TPU program emitter panicked: {message}")))
    })
}
/// A stream on the TPU. All streams of a backend share its programs and its device queue.
pub struct TpuStream {
    shared: Arc<Shared>,
}

impl TpuStream {
    pub fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }
}

impl Stream for TpuStream {
    fn argmax_rows(
        &mut self,
        logits: &DeviceBuffer,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<i32>, BackendError> {
        guarded(|| ops::argmax_rows(&self.shared, logits, rows, vocab))
    }

    fn quantized_linear_add(
        &mut self,
        residual: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
        scale: f32,
    ) -> Result<bool, BackendError> {
        guarded(|| ops::linear(&self.shared, residual, x, weight, bias, Some(scale)))?;
        Ok(true)
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.shared.synchronize()
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        guarded(|| ops::upload(&self.shared, dst, src))
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        guarded(|| ops::download(&self.shared, dst, src))
    }

    fn copy_buffer(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        guarded(|| ops::copy_buffer(&self.shared, dst, src))
    }

    fn embedding_lookup(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError> {
        guarded(|| ops::embedding_lookup(&self.shared, out, ids, weight))
    }

    fn rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError> {
        guarded(|| ops::norm(&self.shared, out, x, weight, None, eps, false))
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        guarded(|| ops::norm(&self.shared, out, x, weight, bias, eps, true))
    }

    fn rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError> {
        guarded(|| ops::rope(&self.shared, q, k, positions, params))
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        guarded(|| ops::linear(&self.shared, out, x, weight, bias, None))
    }

    fn moe_topk(
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
        guarded(|| {
            ops::moe_topk(
                &self.shared,
                logits,
                ids,
                weights,
                rows,
                experts,
                k,
                normalize,
                scale,
            )
        })
    }

    fn moe_combine(
        &mut self,
        out: &mut DeviceBuffer,
        y: &DeviceBuffer,
        weights: &DeviceBuffer,
        tokens: usize,
        k: usize,
        hidden: usize,
    ) -> Result<(), BackendError> {
        guarded(|| ops::moe_combine(&self.shared, out, y, weights, tokens, k, hidden))
    }

    fn gather_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        guarded(|| ops::gather_rows(&self.shared, dst, src, idx, weights, row_len))
    }

    fn scatter_add_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        guarded(|| ops::scatter_add_rows(&self.shared, dst, src, idx, weights, row_len))
    }

    fn zero_buffer(&mut self, buf: &mut DeviceBuffer) -> Result<(), BackendError> {
        guarded(|| ops::zero_buffer(&self.shared, buf))
    }

    fn swiglu_oai(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        alpha: f32,
        limit: f32,
    ) -> Result<(), BackendError> {
        guarded(|| ops::glu(&self.shared, out, gate, up, Glu::SwigluOai { alpha, limit }))
    }

    fn swiglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        guarded(|| ops::glu(&self.shared, out, gate, up, Glu::Swiglu))
    }

    fn geglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        guarded(|| ops::glu(&self.shared, out, gate, up, Glu::Geglu))
    }

    fn split_q_gate(
        &mut self,
        q: &mut DeviceBuffer,
        gate: &mut DeviceBuffer,
        projected: &DeviceBuffer,
        rows: usize,
        heads: usize,
        dk: usize,
        per_head: bool,
    ) -> Result<(), BackendError> {
        guarded(|| ops::split_q_gate(&self.shared, q, gate, projected, rows, heads, dk, per_head))
    }

    fn sigmoid_gate_mul(
        &mut self,
        x: &mut DeviceBuffer,
        gate: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        guarded(|| ops::sigmoid_gate_mul(&self.shared, x, gate))
    }

    fn supports_attention_gate(&self) -> bool {
        true
    }

    fn residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError> {
        guarded(|| ops::residual_add(&self.shared, x, residual, scale))
    }

    fn attention_paged(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        guarded(|| {
            ops::attention_paged(
                &self.shared,
                out,
                q,
                kv,
                block_tables,
                lengths,
                layer_idx,
                params,
            )
        })
    }

    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        guarded(|| ops::kv_cache_write(&self.shared, kv, layer_idx, slots, k, v))
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        if num_experts == 0 {
            return Err(BackendError::InvalidShape(
                "moe_route expert counts invalid".into(),
            ));
        }
        let rows = router_logits.size / 4 / num_experts;
        guarded(|| {
            ops::moe_topk(
                &self.shared,
                router_logits,
                indices_out,
                weights_out,
                rows,
                num_experts,
                top_k,
                true,
                1.0,
            )
        })
    }

    fn moe_expert_gemm(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _expert_weights: &[DeviceWeight],
        _expert_indices: &DeviceBuffer,
        _expert_weights_buf: &DeviceBuffer,
        _top_k: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation(
            "tpu: moe_expert_gemm (the decoder runs the experts one by one through quantized_linear, gather_rows and scatter_add_rows)".into(),
        ))
    }

    fn sampling_logits(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        // like the CPU backend, sampling picks the most likely token (temperature and top-p only choose among tokens already ranked)
        let (rows, vocab) = (params.rows, params.vocab_size);
        if vocab == 0 || token_ids_out.len() < rows {
            return Err(BackendError::InvalidShape(
                "sampling buffers smaller than rows x vocab".into(),
            ));
        }
        let ids = guarded(|| ops::argmax_rows(&self.shared, logits, rows, vocab))?;
        if ids.len() != rows {
            return Err(driver(format!(
                "argmax returned {} ids for {rows} rows",
                ids.len()
            )));
        }
        for (slot, id) in token_ids_out.iter_mut().zip(ids) {
            *slot = id.max(0) as u32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_emitter_panic_becomes_an_error() {
        let result: Result<(), BackendError> = guarded(|| panic!("shape rule violated"));
        let message = result
            .expect_err("a panic must become an error")
            .to_string();
        assert!(
            message.contains("emitter panicked") && message.contains("shape rule violated"),
            "{message}"
        );
        assert_eq!(guarded(|| Ok(7)).expect("passes through"), 7);
    }
}
