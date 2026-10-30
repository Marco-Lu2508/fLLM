//! Native Vulkan backend (ash): stream. Owner: package 4.
//!
//! A [`VulkanStream`] records the operations of a decode step into one command buffer (a [`Recorder`](super::commands::Recorder)) and submits
//! it when the host needs a result (`synchronize`, `download`, `argmax_rows`) or when the buffer grows large. Every operation turns its arguments
//! into dispatch plans (`weights::plan` for the projections, `ops::plan_*` for everything else, both pure CPU code), resolves the buffers of the
//! plan and records the dispatches; the recorder inserts the barriers. Host data of a dispatch (positions, block tables ...) goes through a ring
//! of host-visible scratch memory, scratch buffers through a ring of device memory (`stream/ring.rs`).
//!
//! Level of verification: L0 (compiled and unit-tested without a device; nothing here has ever run on a Vulkan driver).

mod exec;
mod fallback;
mod linear;
mod mamba;
mod ring;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use linear::plan_float_linear;
pub use mamba::MambaBufs;

use crate::backend::vulkan::backend::mem::{pool_window, view_of};
use crate::backend::vulkan::backend::shared::Shared;
use crate::backend::vulkan::commands::{Recorder, MAX_INLINE_UPDATE};
use crate::backend::vulkan::memory::{Buffer, BufferUsage};
use crate::backend::vulkan::ops::attention::{plan_attention, plan_kv_write, AttentionTuning, KvLayout};
use crate::backend::vulkan::ops::moe::{
    plan_gather_rows, plan_moe_combine, plan_moe_linear, plan_moe_topk, plan_scatter_add_rows, ExpertAccess, MoeLinear,
};
use crate::backend::vulkan::ops::recurrent::{
    delta_supported, plan_delta_layer_with, plan_mamba1_bc_norms, plan_mamba1_scan, plan_mamba2_mixer, plan_mamba_conv, Mamba1Shape as VkMamba1,
    Mamba2Shape, RunSpec,
};
use crate::backend::vulkan::ops::{
    plan_argmax, plan_embedding, plan_ewise, plan_glu, plan_glu_strided, plan_layer_norm, plan_mla_move, plan_residual_add, plan_rms_norm, plan_rms_norm_grouped,
    plan_rope, plan_sigmoid_gate_mul, plan_split_q_gate, EmbeddingTable, GluLayout, GluMode, MlaPart,
};
use crate::backend::vulkan::weights::{format_info, DeviceLayout};
use crate::backend::{
    validate_kv_write, AttentionParams, BackendError, DeltaRun, DeltaShape, DeltaWeights, DeviceBuffer, DeviceKvCache, DeviceWeight,
    Mamba1Shape, MambaShape, MambaWeights, MlaShape, QuantType, RopeParams, SamplingParams, Stream,
};
use exec::Arg;
use linear::LinTrace;
use ring::{Ring, RingKind};
use std::collections::HashMap;
use std::sync::Arc;

fn mamba2_shape(s: &MambaShape) -> Mamba2Shape {
    Mamba2Shape {
        inner: s.inner,
        state: s.state,
        heads: s.heads,
        groups: s.groups,
        conv_kernel: s.conv_kernel,
        norm_groups: s.norm_groups,
        step_min: s.step_min,
        eps: s.eps,
    }
}

fn mamba1_shape(s: &Mamba1Shape) -> VkMamba1 {
    VkMamba1 { inner: s.inner, state: s.state, rank: s.rank, conv_kernel: s.conv_kernel, bc_rms_eps: s.bc_rms_eps }
}

pub struct VulkanStream {
    sh: Arc<Shared>,
    rec: Recorder,
    /// Scratch device memory (prepass results, staged operands, partial attention results).
    dev: Ring,
    /// Host-visible scratch memory (per-dispatch arrays, the source of uploads).
    host: Ring,
    /// Timeline value of the latest submission of this stream.
    last_value: u64,
    /// Dispatch counter of the recorder at the latest submission.
    flushed_dispatches: u64,
    /// Multiply-adds of the projections recorded since the latest submission (bounds the run time of one submission, see `exec.rs`).
    pending_macs: u64,
    /// Plan without the cooperative-matrix shaders (the one-time measurement of the prompt variants).
    mask_coop: bool,
    readback: Option<Buffer>,
    /// Device copies of constant host tables (rope cos / sin), by `Arc` address.
    tables: HashMap<usize, (Arc<Vec<f32>>, Buffer)>,
    /// What the latest projection read and wrote (see `linear.rs`).
    trace: Option<LinTrace>,
    /// Host arrays already written in this submission (read-only for the shaders), by contents.
    host_cache: Vec<(Arc<Vec<u8>>, Buffer)>,
    cpu: Option<fallback::Cpu>,
}

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// `rows` and `cols` of a normalisation of the rows of `x` with a weight of `cols` floats.
fn norm_geometry(x: &DeviceBuffer, weight: &DeviceBuffer, out: &DeviceBuffer) -> Result<(usize, usize), BackendError> {
    let cols = weight.size / 4;
    if cols == 0 || x.size == 0 || !x.size.is_multiple_of(cols * 4) {
        return Err(invalid(format!(
            "normalisation input of {} bytes is not a whole number of rows of {cols} floats",
            x.size
        )));
    }
    if out.size < x.size {
        return Err(invalid("normalisation output is smaller than its input"));
    }
    Ok((x.size / (cols * 4), cols))
}

impl VulkanStream {
    pub(crate) fn new(sh: Arc<Shared>) -> Result<Self, BackendError> {
        let rec = sh.ctx.recorder()?;
        Ok(Self {
            sh,
            rec,
            dev: Ring::new(RingKind::Device),
            host: Ring::new(RingKind::Host),
            last_value: 0,
            flushed_dispatches: 0,
            pending_macs: 0,
            mask_coop: false,
            readback: None,
            tables: HashMap::new(),
            trace: None,
            host_cache: Vec::new(),
            cpu: None,
        })
    }

    /// Statistics of the recorder (dispatches, barriers, submits), for tests and diagnostics.
    pub fn recorder_stats(&self) -> crate::backend::vulkan::commands::RecorderStats {
        self.rec.stats()
    }

    fn do_upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.is_empty() {
            return Ok(());
        }
        if src.len() > dst.size {
            return Err(invalid(format!(
                "upload of {} bytes into a buffer of {} bytes",
                src.len(),
                dst.size
            )));
        }
        self.touch();
        let view = view_of(dst)?;
        if src.len() <= MAX_INLINE_UPDATE && src.len().is_multiple_of(4) && view.offset().is_multiple_of(4) {
            // small data (activations of a token, positions, norm weights): copied into the command buffer when it is recorded
            self.rec.update_buffer(&view, 0, src)?;
            return self.after_op();
        }
        let staging = self.host.alloc(&self.sh, src.len() as u64)?;
        staging.write(0, src)?;
        self.rec.copy_buffer(&staging, 0, &view, 0, src.len() as u64)?;
        self.after_op()
    }

    fn do_download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        self.touch();
        let n = dst.len().min(src.size);
        let view = view_of(src)?;
        self.read_back(&view, n as u64, dst)
    }

    /// The device copy of a constant host table, uploaded once per `Arc`.
    fn table(&mut self, values: &Arc<Vec<f32>>) -> Result<Buffer, BackendError> {
        let key = Arc::as_ptr(values) as usize;
        if let Some((_, buffer)) = self.tables.get(&key) {
            return Ok(buffer.clone());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = self
            .sh
            .ctx
            .allocator()
            .alloc_device(bytes.len().max(16).next_multiple_of(16) as u64, BufferUsage::COMPUTE)?;
        self.sh.upload_sync(&buffer, 0, &bytes)?;
        let view = buffer.slice(0, bytes.len().max(4) as u64)?;
        self.tables.insert(key, (Arc::clone(values), view.clone()));
        Ok(view)
    }

    fn glu(&mut self, mode: GluMode, out: &DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer) -> Result<(), BackendError> {
        let count = gate.size / 4;
        if count == 0 || up.size < gate.size || out.size < gate.size {
            return Err(invalid("gated activation operands differ in size"));
        }
        let plan = plan_glu(mode, count)?;
        self.run_plans(&[plan], &[Arg::r("gate", gate), Arg::r("up", up), Arg::w("out", out)], &[], false)
    }

    /// Plain (ungated) activation of `x` into `out`.
    fn plain_glu(&mut self, mode: GluMode, out: &DeviceBuffer, x: &DeviceBuffer) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 || out.size < x.size {
            return Err(invalid("activation operands differ in size"));
        }
        let plan = plan_glu(mode, count)?;
        self.run_plans(&[plan], &[Arg::r("gate", x), Arg::w("out", out)], &[], false)
    }

    /// The layout of layer `layer_idx` of both pools of `kv` and the windows that bind them.
    fn kv_windows(&self, kv: &DeviceKvCache, layer_idx: usize) -> Result<(KvLayout, Buffer, Buffer), BackendError> {
        if layer_idx >= kv.layers {
            return Err(invalid(format!("layer {layer_idx} >= {} cached layers", kv.layers)));
        }
        let align = self.sh.caps().min_storage_buffer_offset_alignment;
        let (keys, key_base) = pool_window(&kv.key_pool, layer_idx, kv.layer_key_words(layer_idx), align)?;
        let (values, value_base) = pool_window(&kv.value_pool, layer_idx, kv.layer_value_words(layer_idx), align)?;
        let layout = KvLayout {
            kind: kv.kv_kind,
            kv_heads: kv.layer_kv_heads(layer_idx),
            head_dim: kv.layer_head_dim(layer_idx),
            value_dim: kv.layer_value_dim(layer_idx),
            block_size: kv.block_size,
            pool_blocks: kv.pool_blocks,
            key_base,
            value_base,
        };
        Ok((layout, keys, values))
    }

    /// The expert arena and its offset table behind a `moe_linear` operand.
    fn moe_operands(
        table: &DeviceBuffer,
        bias_table: Option<&DeviceBuffer>,
        quant: QuantType,
        rows: usize,
        cols: usize,
    ) -> Result<(crate::backend::vulkan::backend::mem::ExpertInfo, Option<Buffer>), BackendError> {
        let info = crate::backend::vulkan::backend::mem::expert_info(table)?.clone();
        if info.is_bias || info.quant != quant || info.rows != rows || info.cols != cols {
            return Err(invalid("moe_linear: the table was made for experts of another shape or format"));
        }
        let biases = match bias_table {
            Some(b) => {
                let bi = crate::backend::vulkan::backend::mem::expert_info(b)?;
                if !bi.is_bias || bi.count != info.count || bi.rows != rows {
                    return Err(invalid("moe_linear: the bias table does not match the weight table"));
                }
                Some(bi.arena.clone())
            }
            None => None,
        };
        Ok((info, biases))
    }
}

impl Drop for VulkanStream {
    /// Work enqueued on a stream completes even when the stream is dropped without `synchronize` (as on the CUDA backend).
    fn drop(&mut self) {
        let _ = self.wait_idle();
    }
}

impl Stream for VulkanStream {
    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.wait_idle()
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        self.do_upload(dst, src)
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        self.do_download(dst, src)
    }

    fn copy_buffer(&mut self, dst: &mut DeviceBuffer, src: &DeviceBuffer) -> Result<(), BackendError> {
        let bytes = dst.size.min(src.size) as u64;
        if bytes == 0 {
            return Ok(());
        }
        self.touch();
        let (from, to) = (view_of(src)?, view_of(dst)?);
        let same = from.same_allocation(&to) && from.offset() < to.offset() + bytes && to.offset() < from.offset() + bytes;
        if same {
            // overlapping ranges of one allocation: go through scratch memory
            let scratch = self.dev.alloc(&self.sh, bytes)?;
            self.rec.copy_buffer(&from, 0, &scratch, 0, bytes)?;
            self.rec.copy_buffer(&scratch, 0, &to, 0, bytes)?;
        } else {
            self.rec.copy_buffer(&from, 0, &to, 0, bytes)?;
        }
        self.after_op()
    }

    fn zero_buffer(&mut self, buf: &mut DeviceBuffer) -> Result<(), BackendError> {
        if buf.size == 0 {
            return Ok(());
        }
        self.touch();
        let view = view_of(buf)?;
        if view.offset().is_multiple_of(4) && view.size().is_multiple_of(4) {
            self.rec.fill(&view, 0, view.size(), 0)?;
        } else {
            let zeros = vec![0u8; buf.size];
            let staging = self.host.alloc(&self.sh, zeros.len() as u64)?;
            staging.write(0, &zeros)?;
            self.rec.copy_buffer(&staging, 0, &view, 0, zeros.len() as u64)?;
        }
        self.after_op()
    }

    fn embedding_lookup(&mut self, out: &mut DeviceBuffer, ids: &[i64], weight: &DeviceWeight) -> Result<(), BackendError> {
        if ids.is_empty() {
            return Ok(());
        }
        if out.size < ids.len() * weight.cols * 4 {
            return Err(invalid("embedding output smaller than ids x hidden"));
        }
        let planes = match format_info(weight.quant_type) {
            Some(_) => {
                let layout = DeviceLayout::new(weight.quant_type, weight.rows, weight.cols)?;
                if weight.buffer.size != layout.total_bytes {
                    return Err(BackendError::UnsupportedOperation(
                        "embedding table stored in several slices".into(),
                    ));
                }
                [
                    layout.plane_word_offset[1] as usize,
                    layout.plane_word_offset[2] as usize,
                    layout.plane_word_offset[3] as usize,
                ]
            }
            None => [0; 3],
        };
        let table = EmbeddingTable {
            quant: weight.quant_type,
            rows: weight.rows,
            cols: weight.cols,
            planes,
        };
        let plan = plan_embedding(&table, ids)?;
        self.run_plans(&[plan], &[Arg::r("weights", &weight.buffer), Arg::w("out", out)], &[], false)
    }

    fn rms_norm(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, weight: &DeviceBuffer, eps: f32) -> Result<(), BackendError> {
        let (rows, cols) = norm_geometry(x, weight, out)?;
        let plan = plan_rms_norm(rows, cols, eps, true)?;
        self.run_plans(&[plan], &[Arg::r("x", x), Arg::r("weight", weight), Arg::w("out", out)], &[], false)
    }

    fn rms_norm_grouped(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
        groups: usize,
    ) -> Result<(), BackendError> {
        if groups == 0 {
            return Err(invalid("rms_norm_grouped requires groups > 0"));
        }
        if groups == 1 {
            return self.rms_norm(out, x, weight, eps);
        }
        let (rows, cols) = norm_geometry(x, weight, out)?;
        if !cols.is_multiple_of(groups) {
            return Err(invalid("rms_norm_grouped weight width is not divisible by groups"));
        }
        let plan = plan_rms_norm_grouped(rows, cols, groups, eps, true)?;
        self.run_plans(&[plan], &[Arg::r("x", x), Arg::r("weight", weight), Arg::w("out", out)], &[], false)
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let (rows, cols) = norm_geometry(x, weight, out)?;
        if bias.is_some_and(|b| b.size < weight.size) {
            return Err(invalid("layer_norm bias is shorter than its weight"));
        }
        let plan = plan_layer_norm(rows, cols, eps, bias.is_some())?;
        let mut args = vec![Arg::r("x", x), Arg::r("weight", weight), Arg::w("out", out)];
        if let Some(b) = bias {
            args.push(Arg::r("bias", b));
        }
        self.run_plans(&[plan], &args, &[], false)
    }

    fn rope(&mut self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, positions: &[usize], params: &RopeParams) -> Result<(), BackendError> {
        if positions.is_empty() {
            return Ok(());
        }
        params.validate(positions, q.size, k.size)?;
        let half = params.rope_dim / 2;
        let table_rows = params.cos.len() / half;
        let plan = plan_rope(
            positions,
            params.heads,
            params.kv_heads,
            params.dim,
            params.rope_dim,
            params.neox,
            table_rows,
        )?;
        let cos = self.table(&params.cos)?;
        let sin = self.table(&params.sin)?;
        self.run_plans(
            &[plan],
            &[Arg::w("q", q), Arg::w("k", k), Arg::view("cos", cos), Arg::view("sin", sin)],
            &[],
            false,
        )
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        self.quantized_linear_impl(out, x, weight, bias)
    }

    fn supports_host_linear(&self, quant_type: QuantType) -> bool {
        format_info(quant_type).is_some()
    }

    fn host_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        raw: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        self.host_linear_impl(out, x, raw, quant_type, rows, cols, bias)
    }

    fn swiglu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer) -> Result<(), BackendError> {
        self.glu(GluMode::Swiglu, out, gate, up)
    }

    fn geglu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer) -> Result<(), BackendError> {
        self.glu(GluMode::Geglu, out, gate, up)
    }

    fn swiglu_oai(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        alpha: f32,
        limit: f32,
    ) -> Result<(), BackendError> {
        self.glu(GluMode::SwigluOai { alpha, limit }, out, gate, up)
    }

    fn residual_add(&mut self, x: &mut DeviceBuffer, residual: &DeviceBuffer, scale: f32) -> Result<(), BackendError> {
        let count = x.size.min(residual.size) / 4;
        if count == 0 {
            return Ok(());
        }
        let plan = plan_residual_add(count, scale)?;
        self.run_plans(&[plan], &[Arg::w("x", x), Arg::r("residual", residual)], &[], false)
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
        let total = rows * heads * dk;
        if total == 0 {
            return Ok(());
        }
        if projected.size < total * 2 * 4 || q.size < total * 4 || gate.size < total * 4 {
            return Err(invalid("split_q_gate buffers are smaller than rows x heads x dk"));
        }
        let plan = plan_split_q_gate(rows, heads, dk, per_head)?;
        self.run_plans(
            &[plan],
            &[Arg::r("projected", projected), Arg::w("q", q), Arg::w("gate", gate)],
            &[],
            false,
        )
    }

    fn sigmoid_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer) -> Result<(), BackendError> {
        let count = x.size.min(gate.size) / 4;
        if count == 0 {
            return Ok(());
        }
        let plan = plan_sigmoid_gate_mul(count)?;
        self.run_plans(&[plan], &[Arg::w("x", x), Arg::r("gate", gate)], &[], false)
    }

    fn supports_attention_gate(&self) -> bool {
        true
    }

    fn supports_kda(&self, shape: &crate::backend::KdaShape) -> bool {
        crate::backend::vulkan::ops::recurrent::kda_supported(shape)
    }

    fn supports_arch_ops(&self) -> bool {
        true
    }

    fn activation(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        let mode = match mode {
            0 => GluMode::Silu,
            1 => GluMode::Gelu,
            2 => GluMode::GeluErf,
            3 => GluMode::Relu,
            4 => GluMode::Relu2,
            _ => return Err(invalid(format!("unknown activation mode {mode}"))),
        };
        self.plain_glu(mode, out, x)
    }

    fn activation_glu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        let mode = match mode {
            0 => GluMode::Swiglu,
            1 => GluMode::Geglu,
            2 => GluMode::GeluErfGated,
            3 => GluMode::ReluGated,
            4 => GluMode::Relu2Gated,
            _ => return Err(invalid(format!("unknown activation mode {mode}"))),
        };
        self.glu(mode, out, gate, up)
    }

    fn softplus_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer, beta: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || gate.size < x.size || beta <= 0.0 {
            return Err(invalid("attention gate shorter than its target"));
        }
        if x.size == 0 {
            return Ok(());
        }
        let plan = plan_ewise(true, x.size / 4, beta)?;
        self.run_plans(&[plan], &[Arg::w("x", x), Arg::r("gate", gate)], &[], false)
    }

    fn clamp(&mut self, x: &mut DeviceBuffer, limit: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) {
            return Err(invalid("clamp target is not a whole number of floats"));
        }
        if x.size == 0 {
            return Ok(());
        }
        let plan = plan_ewise(false, x.size / 4, limit)?;
        self.run_plans(&[plan], &[Arg::w("x", x)], &[], false)
    }

    fn mla_latent(&mut self, latent: &mut DeviceBuffer, kv_a: &DeviceBuffer, rows: usize, kv_lora: usize, width: usize) -> Result<(), BackendError> {
        if width < kv_lora || kv_a.size < rows * width * 4 || latent.size < rows * kv_lora * 4 {
            return Err(invalid("mla_latent buffers are smaller than rows x width"));
        }
        if rows * kv_lora == 0 {
            return Ok(());
        }
        let plan = plan_mla_move(MlaPart::Latent, rows, 1, 0, width - kv_lora, 1, kv_lora)?;
        self.run_plans(&[plan], &[Arg::r("a", kv_a), Arg::w("dst", latent)], &[], false)
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
        let (dk, width) = (shape.nope + shape.rope, shape.kv_lora + shape.rope);
        let items = rows * shape.heads;
        if q_full.size < items * dk * 4
            || kv_a.size < rows * width * 4
            || kv.size < items * (shape.nope + shape.value) * 4
            || q.size < items * dk * 4
            || k.size < items * dk * 4
            || v.size < items * shape.value * 4
        {
            return Err(invalid("mla_assemble buffers are smaller than rows x heads x width"));
        }
        if items == 0 {
            return Ok(());
        }
        let part = |p| plan_mla_move(p, rows, shape.heads, shape.nope, shape.rope, shape.value, shape.kv_lora);
        self.run_plans(&[part(MlaPart::Q)?], &[Arg::r("a", q_full), Arg::w("dst", q)], &[], false)?;
        self.run_plans(&[part(MlaPart::K)?], &[Arg::r("a", kv_a), Arg::r("b", kv), Arg::w("dst", k)], &[], false)?;
        self.run_plans(&[part(MlaPart::V)?], &[Arg::r("a", kv), Arg::w("dst", v)], &[], false)
    }

    fn supports_mamba2(&self, shape: &MambaShape) -> bool {
        // mamba_scan2 binds up to nine storage buffers
        plan_mamba2_mixer(&mamba2_shape(shape), &[RunSpec { first_row: 0, rows: 1, reset: true }], 1, 0).is_ok()
            && self.sh.caps().max_per_stage_storage_buffers >= 9
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
        if !self.supports_mamba2(shape) {
            return Err(BackendError::UnsupportedOperation("mamba2_layer for this shape".into()));
        }
        let bufs = MambaBufs {
            conv: weights.conv,
            conv_bias: weights.conv_bias,
            dt_bias: weights.dt_bias,
            decay: weights.decay,
            skip: weights.skip,
            norm: weights.norm,
        };
        let runs: Vec<(RunSpec, &DeviceBuffer)> =
            runs.iter().map(|r| (RunSpec { first_row: r.first_row, rows: r.rows, reset: r.reset }, r.state)).collect();
        self.mamba2_mixer(&mamba2_shape(shape), &bufs, &runs, state_offset, projected, mixed, out)
    }

    fn supports_mamba1(&self, shape: &Mamba1Shape) -> bool {
        let s = mamba1_shape(shape);
        plan_mamba1_scan(0, &RunSpec { first_row: 0, rows: 1, reset: true }, &s, 2 * s.inner, s.inner, 0).is_ok()
            && self.sh.caps().max_per_stage_storage_buffers >= 8
    }

    fn mamba1_conv(
        &mut self,
        shape: &Mamba1Shape,
        conv: &DeviceBuffer,
        conv_bias: Option<&DeviceBuffer>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        proj: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let rows = runs.iter().map(|r| r.first_row + r.rows).max().unwrap_or(0);
        if rows == 0 {
            return Ok(());
        }
        let s = mamba1_shape(shape);
        let state_end = state_offset + shape.layer_state_bytes();
        if proj.size < rows * 2 * s.inner * 4
            || mixed.size < rows * s.inner * 4
            || conv.size < s.inner * s.conv_kernel * 4
            || conv_bias.is_some_and(|b| b.size < s.inner * 4)
            || runs.iter().any(|r| r.state.size < state_end)
        {
            return Err(invalid("Mamba-1 buffers are smaller than the layer needs"));
        }
        // state layout of the trait: recurrent state first, then the convolution history
        let hist_off = (state_offset + shape.rec_bytes()) / 4;
        let zeros = match conv_bias {
            Some(_) => None,
            None => {
                let buffer = self.dev.alloc(&self.sh, (s.inner * 4) as u64)?;
                self.rec.fill(&buffer, 0, (s.inner * 4).next_multiple_of(4) as u64, 0)?;
                Some(buffer)
            }
        };
        let mut plans = Vec::new();
        for (i, r) in runs.iter().enumerate() {
            if r.rows == 0 {
                continue;
            }
            let spec = RunSpec { first_row: r.first_row, rows: r.rows, reset: r.reset };
            plans.push(plan_mamba_conv(i, &spec, s.inner, s.conv_kernel, 2 * s.inner, 0, hist_off, true, None)?);
        }
        let mut args = vec![Arg::r("x", proj), Arg::r("convw", conv), Arg::w("mixed", mixed)];
        match (conv_bias, zeros) {
            (Some(bias), _) => args.push(Arg::r("convb", bias)),
            (None, Some(buffer)) => args.push(Arg::view("convb", buffer)),
            (None, None) => {}
        }
        let states: Vec<&DeviceBuffer> = runs.iter().map(|r| r.state).collect();
        self.run_plans(&plans, &args, &states, false)
    }

    fn mamba1_prep(&mut self, shape: &Mamba1Shape, split: &mut DeviceBuffer, dt_in: &mut DeviceBuffer, rows: usize) -> Result<(), BackendError> {
        let s = mamba1_shape(shape);
        if rows == 0 {
            return Ok(());
        }
        if split.size < rows * s.split_width() * 4 || dt_in.size < rows * s.rank * 4 {
            return Err(invalid("Mamba-1 prep buffers are smaller than rows x width"));
        }
        for plan in plan_mamba1_bc_norms(&s, rows)? {
            self.run_plans(&[plan], &[Arg::w("x", split), Arg::w("out", split)], &[], false)?;
        }
        let layout = GluLayout { gate_stride: s.split_width(), gate_off: 0, up_stride: 0, up_off: 0, out_stride: s.rank };
        let plan = plan_glu_strided(GluMode::Copy, rows, s.rank, layout)?;
        self.run_plans(&[plan], &[Arg::r("gate", split), Arg::w("out", dt_in)], &[], false)
    }

    fn mamba1_scan(
        &mut self,
        shape: &Mamba1Shape,
        decay: &DeviceBuffer,
        skip: &DeviceBuffer,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        split: &DeviceBuffer,
        dt: &DeviceBuffer,
        mixed: &DeviceBuffer,
        proj: &DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let rows = runs.iter().map(|r| r.first_row + r.rows).max().unwrap_or(0);
        if rows == 0 {
            return Ok(());
        }
        let s = mamba1_shape(shape);
        let state_end = state_offset + shape.layer_state_bytes();
        if split.size < rows * s.split_width() * 4
            || dt.size < rows * s.inner * 4
            || mixed.size < rows * s.inner * 4
            || proj.size < rows * 2 * s.inner * 4
            || out.size < rows * s.inner * 4
            || decay.size < s.inner * s.state * 4
            || skip.size < s.inner * 4
            || runs.iter().any(|r| r.state.size < state_end)
        {
            return Err(invalid("Mamba-1 buffers are smaller than the layer needs"));
        }
        let mut plans = Vec::new();
        for (i, r) in runs.iter().enumerate() {
            if r.rows == 0 {
                continue;
            }
            let spec = RunSpec { first_row: r.first_row, rows: r.rows, reset: r.reset };
            plans.push(plan_mamba1_scan(i, &spec, &s, 2 * s.inner, s.inner, state_offset / 4)?);
        }
        let args = [
            Arg::r("mixed", mixed),
            Arg::r("dt", dt),
            Arg::r("split", split),
            Arg::r("projected", proj),
            Arg::r("decay", decay),
            Arg::r("skip", skip),
            Arg::w("out", out),
        ];
        let states: Vec<&DeviceBuffer> = runs.iter().map(|r| r.state).collect();
        self.run_plans(&plans, &args, &states, false)
    }

    fn supports_delta(&self, shape: &DeltaShape) -> bool {
        // delta_rec binds nine storage buffers
        delta_supported(shape) && self.sh.caps().max_per_stage_storage_buffers >= 9
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
        if !self.supports_delta(shape) {
            return Err(BackendError::UnsupportedOperation("delta_layer for this shape".into()));
        }
        let rows = runs.iter().map(|r| r.first_row + r.rows).max().unwrap_or(0);
        if rows == 0 {
            return Ok(());
        }
        let conv_dim = shape.conv_dim();
        let state_end = state_offset + shape.layer_state_bytes();
        if qkv.size < rows * conv_dim * 4
            || mixed.size < rows * conv_dim * 4
            || z.size < rows * shape.value_dim() * 4
            || beta.size < rows * shape.v_heads * 4
            || alpha.size < rows * shape.v_heads * 4
            || out.size < rows * shape.value_dim() * 4
            || weights.conv.size < conv_dim * shape.conv_kernel * 4
            || weights.dt_bias.size < shape.v_heads * 4
            || weights.decay.size < shape.v_heads * 4
            || weights.norm.size < shape.v_dim * 4
            || runs.iter().any(|r| r.state.size < state_end)
        {
            return Err(invalid("DeltaNet buffers are smaller than the layer needs"));
        }
        let specs: Vec<RunSpec> = runs
            .iter()
            .map(|r| RunSpec {
                first_row: r.first_row,
                rows: r.rows,
                reset: r.reset,
            })
            .collect();
        let caps = self.sh.caps();
        let min_subgroup = if caps.min_subgroup_size > 0 { caps.min_subgroup_size } else { caps.subgroup_size };
        let cols = crate::backend::vulkan::ops::recurrent::delta_cols_usable(caps.subgroup_ops.shuffle, min_subgroup);
        let plans = plan_delta_layer_with(shape, &specs, state_offset, cols)?;
        if plans.is_empty() {
            return Ok(());
        }
        let args = [
            Arg::r("qkv", qkv),
            Arg::r("conv", weights.conv),
            Arg::r("z", z),
            Arg::r("beta", beta),
            Arg::r("alpha", alpha),
            Arg::r("dt_bias", weights.dt_bias),
            Arg::r("decay", weights.decay),
            Arg::r("norm", weights.norm),
            Arg::w("out", out),
            Arg::w("mixed", mixed),
        ];
        let states: Vec<&DeviceBuffer> = runs.iter().map(|r| r.state).collect();
        self.run_plans(&plans, &args, &states, false)
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
        params.validate(kv, block_tables, lengths, layer_idx)?;
        let rows = lengths.len();
        if rows == 0 {
            return Ok(());
        }
        let (heads, dk, dv) = (params.num_heads, params.head_dim, params.value_dim);
        if q.size < rows * heads * dk * 4 || out.size < rows * heads * dv * 4 {
            return Err(invalid("attention q/out smaller than rows x heads x dim"));
        }
        if dk > crate::backend::vulkan::ops::attention::MAX_HEAD_WIDTH || dv > crate::backend::vulkan::ops::attention::MAX_HEAD_WIDTH {
            return self.attention_on_cpu(out, q, kv, block_tables, lengths, layer_idx, params);
        }
        let (layout, keys, values) = self.kv_windows(kv, layer_idx)?;
        let plans = plan_attention(&layout, params, block_tables, lengths, &AttentionTuning::for_device(self.sh.caps().max_workgroup_invocations).with_coop_prefill(&self.sh.caps_like()))?;
        self.run_plans(
            &plans,
            &[
                Arg::r("q", q),
                Arg::view("kpool", keys),
                Arg::view("vpool", values),
                Arg::w("out", out),
            ],
            &[],
            false,
        )
    }

    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        if slots.is_empty() {
            return Ok(());
        }
        let (layout, keys, values) = self.kv_windows(kv, layer_idx)?;
        let plans = plan_kv_write(&layout, slots)?;
        // the keys and the values go to different pools: the two dispatches do not depend on each other
        self.run_plans(
            &plans,
            &[
                Arg::r("k", k),
                Arg::r("v", v),
                Arg::view("key_pool", keys),
                Arg::view("value_pool", values),
            ],
            &[],
            true,
        )
    }

    fn argmax_rows(&mut self, logits: &DeviceBuffer, rows: usize, vocab: usize) -> Result<Vec<i32>, BackendError> {
        if rows == 0 || vocab == 0 {
            return Ok(Vec::new());
        }
        if logits.size < rows * vocab * 4 {
            return Err(invalid("argmax_rows: the logits buffer is smaller than rows x vocab floats"));
        }
        let plan = plan_argmax(rows, vocab)?;
        let result = self.dev.alloc(&self.sh, (rows * 4) as u64)?;
        self.run_plans(&[plan], &[Arg::r("logits", logits), Arg::view("out", result.clone())], &[], false)?;
        let mut bytes = vec![0u8; rows * 4];
        self.read_back(&result, (rows * 4) as u64, &mut bytes)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect())
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
        if rows == 0 {
            return Ok(());
        }
        if logits.size < rows * experts * 4 || ids.size < rows * k * 4 || weights.size < rows * k * 4 {
            return Err(invalid("moe_topk buffers are smaller than rows x experts / rows x k"));
        }
        let plan = plan_moe_topk(rows, experts, k, normalize, scale)?;
        self.run_plans(
            &[plan],
            &[Arg::r("logits", logits), Arg::w("ids", ids), Arg::w("weights", weights)],
            &[],
            false,
        )
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
        if slots == 0 {
            return Ok(());
        }
        let (info, biases) = Self::moe_operands(table, bias_table, quant_type, rows, cols)?;
        if x_div == 0 || x.size < slots.div_ceil(x_div) * cols * 4 || out.size < slots * rows * 4 || ids.size < slots * 4 {
            return Err(invalid("moe_linear buffers are smaller than the slot counts say"));
        }
        let call = MoeLinear {
            quant: quant_type,
            rows,
            cols,
            slots,
            x_div,
            n_experts: info.count,
            has_bias: biases.is_some(),
            access: ExpertAccess::Arena,
            planes: info.planes,
        };
        let plan = plan_moe_linear(&call)?;
        let mut args = vec![
            Arg::r("table", table),
            Arg::r("ids", ids),
            Arg::r("x", x),
            Arg::w("out", out),
            Arg::view("weights", info.arena.clone()),
        ];
        if let (Some(b), Some(arena)) = (bias_table, biases) {
            args.push(Arg::r("bias_table", b));
            args.push(Arg::view("biases", arena));
        }
        self.run_plans(&[plan], &args, &[], false)
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
        if tokens == 0 {
            return Ok(());
        }
        if y.size < tokens * k * hidden * 4 || weights.size < tokens * k * 4 || out.size < tokens * hidden * 4 {
            return Err(invalid("moe_combine buffers are smaller than tokens x k x hidden"));
        }
        let plan = plan_moe_combine(tokens, k, hidden)?;
        self.run_plans(
            &[plan],
            &[Arg::w("out", out), Arg::r("y", y), Arg::r("weights", weights)],
            &[],
            false,
        )
    }

    fn gather_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        if idx.is_empty() {
            return Ok(());
        }
        if dst.size < idx.len() * row_len * 4 {
            return Err(invalid("gather_rows destination is smaller than idx x row_len floats"));
        }
        let highest = idx.iter().copied().max().unwrap_or(0).max(0) as usize;
        if src.size < (highest + 1) * row_len * 4 {
            return Err(invalid("gather_rows source is smaller than the highest index says"));
        }
        let plan = plan_gather_rows(idx, weights, row_len)?;
        self.run_plans(&[plan], &[Arg::w("dst", dst), Arg::r("src", src)], &[], false)
    }

    fn scatter_add_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        if idx.is_empty() {
            return Ok(());
        }
        if src.size < idx.len() * row_len * 4 {
            return Err(invalid("scatter_add_rows source is smaller than idx x row_len floats"));
        }
        let highest = idx.iter().copied().max().unwrap_or(0).max(0) as usize;
        if dst.size < (highest + 1) * row_len * 4 {
            return Err(invalid("scatter_add_rows destination is smaller than the highest index says"));
        }
        let plan = plan_scatter_add_rows(idx, weights, row_len)?;
        self.run_plans(&[plan], &[Arg::w("dst", dst), Arg::r("src", src)], &[], false)
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        self.moe_route_on_cpu(weights_out, indices_out, router_logits, num_experts, top_k)
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
        // the weights are in the planar device layout, which the CPU backend cannot read; `moe_linear` + `moe_combine` is the device path
        Err(BackendError::UnsupportedOperation(
            "moe_expert_gemm on the Vulkan backend (use moe_topk, moe_linear and moe_combine)".into(),
        ))
    }

    fn sampling_logits(&mut self, token_ids_out: &mut [u32], logits: &DeviceBuffer, params: &SamplingParams) -> Result<(), BackendError> {
        if params.temperature <= 0.0 || params.top_k == 1 {
            let ids = self.argmax_rows(logits, params.rows, params.vocab_size)?;
            for (slot, id) in token_ids_out.iter_mut().zip(ids) {
                *slot = id.max(0) as u32;
            }
            return Ok(());
        }
        self.sampling_on_cpu(token_ids_out, logits, params)
    }
}
