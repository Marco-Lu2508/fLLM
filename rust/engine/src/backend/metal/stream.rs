//! Native Metal backend (objc2-metal, Apple platforms only): the `Stream`. Owner: package 7 (integration).
//!
//! One stream owns one command recorder of the Metal core: every operation turns its arguments into dispatch plans with the planners of the
//! shader packages (`weights::plan_quantized_linear`, `ops::*`, `ops::moe_plan`, `ops::recurrent_plan`), binds the buffers named by the plans
//! and appends the dispatches to the open command buffer. The recorder inserts the barriers between dispatches from the declared buffer
//! accesses and commits a command buffer every `FLLM_METAL_CB_DISPATCHES` dispatches; the CPU waits only in `synchronize` / `download` /
//! `argmax_rows`.
//!
//! * `exec`: plan runner, pipelines, host arrays and scratch, transfers; `arena`: per-step bump allocators;
//! * `basic`: embedding, norms, RoPE, activations, residual, gating, argmax, sampling; `linear`: quantized / float projections;
//! * `attention`: KV-cache write and paged attention; `moe`: routing, expert products, row gather / scatter; `recurrent`: DeltaNet.
//!
//! * `arch`: plain and gated activations, clamping, the softplus gate and the data movement of expanded MLA.
//!
//! Not implemented (the trait defaults answer "unsupported" / "not done", the decoder takes its fallback): the fused FP16 paths
//! (`linear_f16_out`, `glu_f16`, `rope_kv_write_f16`), graph capture, `host_linear` (unified memory needs no weight streaming), Mamba-1 and the
//! LFM2 short convolution (Mamba-2 and DeltaNet are in `recurrent`).
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; nothing has run on a Mac).

mod arch;
mod arena;
mod attention;
mod basic;
mod exec;
mod linear;
mod moe;
mod recurrent;

use super::backend::shared::Shared;
use super::commands::Recorder;
use super::ops::recurrent_plan;
use super::pipelines::Pipeline;
use super::weights::PlanConstant;
use crate::backend::cpu::{CpuBackend, CpuStream};
use crate::backend::{
    AttentionParams, BackendError, DeltaRun, DeltaShape, DeltaWeights, DeviceBuffer, MambaShape, MambaWeights,
    DeviceKvCache, DeviceWeight, MlaShape, QuantType, RopeParams, SamplingParams, Stream,
};
use arena::StepArena;
use std::collections::HashMap;
use std::sync::Arc;

/// A pipeline together with the function constants it was specialised with.
type SpecialisedPipeline = (Vec<PlanConstant>, Arc<Pipeline>);

pub struct MetalStream {
    shared: Arc<Shared>,
    rec: Recorder,
    /// Pipelines this stream resolved, by kernel name, then by function constants.
    pipes: HashMap<String, Vec<SpecialisedPipeline>>,
    /// Small host arrays of the dispatches of the current step (positions, slots, block tables, ...); reset by `synchronize`.
    host_arena: StepArena,
    /// Scratch storage of the operations of the current step; reset by `synchronize`.
    scratch_arena: StepArena,
    /// Host fallbacks (temperature sampling).
    cpu: CpuBackend,
    cpu_stream: CpuStream,
}

impl Drop for MetalStream {
    fn drop(&mut self) {
        // the arenas and everything the recorded dispatches use must outlive the GPU work
        let _ = self.rec.synchronize();
    }
}

impl Stream for MetalStream {
    fn argmax_rows(
        &mut self,
        logits: &DeviceBuffer,
        rows: usize,
        vocab: usize,
    ) -> Result<Vec<i32>, BackendError> {
        self.op_argmax(logits, rows, vocab)
    }

    fn quantized_linear_add(
        &mut self,
        residual: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
        scale: f32,
    ) -> Result<bool, BackendError> {
        // the kernels fold `residual += result` into the output stage; a factor other than 1 is left to the caller's separate add
        if scale != 1.0 {
            return Ok(false);
        }
        self.ensure_headroom()?;
        self.op_linear(residual, x, weight, bias, true)?;
        Ok(true)
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.sync()
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_upload(dst, src)
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        self.op_download(dst, src)
    }

    fn copy_buffer(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.op_copy(dst, src)
    }

    fn embedding_lookup(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_embedding(out, ids, weight)
    }

    fn rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_rms_norm(out, x, weight, eps)
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_layer_norm(out, x, weight, bias, eps)
    }

    fn rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_rope(q, k, positions, params)
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_linear(out, x, weight, bias, false)
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
        self.ensure_headroom()?;
        self.op_moe_topk(logits, ids, weights, rows, experts, k, normalize, scale)
    }

    fn moe_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        table: &DeviceBuffer,
        bias_table: Option<&DeviceBuffer>,
        ids: &DeviceBuffer,
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        slots: usize,
        x_div: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_moe_linear(
            out, x, table, bias_table, ids, quant_type, rows, cols, slots, x_div,
        )
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
        self.ensure_headroom()?;
        self.op_moe_combine(out, y, weights, tokens, k, hidden)
    }

    fn gather_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_gather_rows(dst, src, idx, weights, row_len)
    }

    fn scatter_add_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_scatter_add_rows(dst, src, idx, weights, row_len)
    }

    fn zero_buffer(&mut self, buf: &mut DeviceBuffer) -> Result<(), BackendError> {
        self.op_zero(buf)
    }

    fn swiglu_oai(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        alpha: f32,
        limit: f32,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_glu(
            super::ops::GluMode::SwigluOai { alpha, limit },
            out,
            gate,
            up,
        )
    }

    fn swiglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_glu(super::ops::GluMode::Swiglu, out, gate, up)
    }

    fn geglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_glu(super::ops::GluMode::Geglu, out, gate, up)
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
        self.ensure_headroom()?;
        self.op_split_q_gate(q, gate, projected, rows, heads, dk, per_head)
    }

    fn sigmoid_gate_mul(
        &mut self,
        x: &mut DeviceBuffer,
        gate: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_sigmoid_gate_mul(x, gate)
    }

    fn supports_attention_gate(&self) -> bool {
        recurrent_plan::attention_gate_supported()
    }

    fn supports_kda(&self, shape: &crate::backend::KdaShape) -> bool {
        crate::backend::metal::ops::recurrent_plan::kda_supported(shape)
    }

    fn supports_arch_ops(&self) -> bool {
        true
    }

    fn activation(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_activation(out, x, mode)
    }

    fn activation_glu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        mode: i32,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_activation_glu(out, gate, up, mode)
    }

    fn softplus_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer, beta: f32) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_softplus_gate_mul(x, gate, beta)
    }

    fn clamp(&mut self, x: &mut DeviceBuffer, limit: f32) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_clamp(x, limit)
    }

    fn mla_latent(
        &mut self,
        latent: &mut DeviceBuffer,
        kv_a: &DeviceBuffer,
        rows: usize,
        kv_lora: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_mla_latent(latent, kv_a, rows, kv_lora, width)
    }

    fn mla_assemble(
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
        self.ensure_headroom()?;
        self.op_mla_assemble(q, k, v, q_full, kv_a, kv, rows, shape)
    }

    fn supports_delta(&self, shape: &DeltaShape) -> bool {
        recurrent_plan::delta_supported(shape)
    }

    fn delta_layer(
        &mut self,
        shape: &DeltaShape,
        weights: &DeltaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        qkv: &DeviceBuffer,
        z: &DeviceBuffer,
        beta: &DeviceBuffer,
        alpha: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_delta_layer(
            shape,
            weights,
            runs,
            state_offset,
            [qkv, z, beta, alpha],
            mixed,
            out,
        )
    }

    fn supports_mamba2(&self, shape: &MambaShape) -> bool {
        shape.norm_mode == 0 && recurrent_plan::mamba2_supported(&recurrent::mamba2_shape(shape))
    }

    fn mamba2_layer(
        &mut self,
        shape: &MambaShape,
        weights: &MambaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        projected: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_mamba2_layer(shape, weights, runs, state_offset, projected, mixed, out)
    }

    fn residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_residual_add(x, residual, scale)
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
        self.ensure_headroom()?;
        self.op_attention(out, q, kv, block_tables, lengths, layer_idx, params)
    }

    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_kv_write(kv, layer_idx, slots, k, v)
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_moe_route(weights_out, indices_out, router_logits, num_experts, top_k)
    }

    fn moe_expert_gemm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        expert_weights: &[DeviceWeight],
        expert_indices: &DeviceBuffer,
        expert_weights_buf: &DeviceBuffer,
        top_k: usize,
    ) -> Result<(), BackendError> {
        self.ensure_headroom()?;
        self.op_moe_expert_gemm(
            out,
            x,
            expert_weights,
            expert_indices,
            expert_weights_buf,
            top_k,
        )
    }

    fn sampling_logits(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        self.op_sampling(token_ids_out, logits, params)
    }
}
