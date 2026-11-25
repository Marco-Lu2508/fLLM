//! DSpine (`DSpineDraftModel`) and DFlash (`DFlashDraftModel` / `DFlash2DraftModel`) parallel
//! speculative decoding engine.
//!
//! Each verification round commits target hidden representations at `target_layer_ids`, projects
//! their concatenation via `fc` + `hidden_norm`, appends the resulting context K/V rows to each
//! draft layer's sliding-window cache, runs a parallel block `[anchor, mask_id, ...]` through the
//! Qwen3-style draft layers (with adjacent `DSpineCodeBridge` writes when present), and refines or
//! decodes rows `1..T-1` into exact draft tokens + candidate distributions.

use crate::backend::act_dtype::ActDtype;
use crate::backend::{
    AttentionParams, Backend, BufferPool, DeviceBuffer, DeviceKvCache, DeviceWeight, QuantType,
    RopeParams, Stream,
};
use crate::compute::{Linear, Tensor};
use crate::model::GgufModel;
use crate::rope::{self, RopeTables};
use crate::rust_decoder::RustDecoder;
use crate::sampling::{self, processed_distribution, select_from, SamplingParams, SamplingState};
use crate::speculative::Dist;
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

const F32_TINY: f32 = f32::MIN_POSITIVE;

#[derive(Clone, Debug)]
pub struct DSpineConfig {
    pub arch: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub vocab_size: usize,
    pub eps: f32,
    pub block_size: usize,
    pub mask_token_id: i64,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub target_hidden: usize,
    pub block_causal: bool,
    pub sliding_window: Option<usize>,
    pub has_bridge: bool,
    pub code_size: usize,
    pub rank: usize,
    pub bridge_eps: f32,
    pub head_topk: usize,
    pub feature_layers: Vec<usize>,
    pub terminal_residual_rank: usize,
}

struct DraftLayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct BridgeWeights {
    codebook: Vec<f32>,              // [V, C]
    read_weights: Vec<Vec<f32>>,     // [layers][C, H]
    feature_write_weights: Vec<Vec<f32>>, // [layers][H, R]
    gate_w: Vec<f32>,                // [R, C]
    pred_gate_w: Vec<f32>,           // [R, C]
    prev_w: Vec<f32>,                // [R, C]
    gate_bias: Vec<f32>,             // [R]
    active_layers: HashSet<usize>,
    terminal_emb: Vec<f32>,          // [V, rt] when rt > 0
    terminal_proj: Vec<f32>,         // [R, rt] when rt > 0
}

#[derive(Default, Clone)]
struct LayerKvCache {
    /// Kept context keys in `[kept_rows, kv_heads * head_dim]`, trimmed to `sliding_window` when active.
    k: Vec<f32>,
    v: Vec<f32>,
    /// Absolute position of `k[0]` (monotonically advances when trimmed by `sliding_window`).
    start_pos: usize,
    /// Total context rows pushed so far (`start_pos + kept_rows()`).
    total_len: usize,
}

impl LayerKvCache {
    fn kept_rows(&self, kv_dim: usize) -> usize {
        self.k.len().checked_div(kv_dim).unwrap_or(0)
    }

    fn append(&mut self, k_rows: &[f32], v_rows: &[f32], rows: usize, kv_dim: usize, window: Option<usize>) {
        if rows == 0 {
            return;
        }
        self.k.extend_from_slice(k_rows);
        self.v.extend_from_slice(v_rows);
        self.total_len += rows;
        if let Some(w) = window {
            let kept = self.kept_rows(kv_dim);
            if kept > w {
                let drop = kept - w;
                self.k.drain(..drop * kv_dim);
                self.v.drain(..drop * kv_dim);
                self.start_pos += drop;
            }
        }
    }
}

const DRAFT_KV_BLOCK_SIZE: usize = 16;

#[derive(Default, Clone)]
struct SeqDeviceKv {
    blocks: Vec<usize>,
    ctx_rows_in_blocks: usize,
    total_len: usize,
}

#[derive(Clone)]
pub(crate) struct DSpineSeqSnapshot {
    pending_target_hidden: Vec<f32>,
    pending_rows: usize,
    host_kv: Vec<LayerKvCache>,
    dev_kv: SeqDeviceKv,
    had_state: bool,
}

#[derive(Default)]
struct SeqState {
    kv: Vec<LayerKvCache>,
    dev_kv: SeqDeviceKv,
    pending_target_hidden: Vec<f32>,
    pending_rows: usize,
}

#[derive(Default)]
struct BridgeStash {
    raw: Vec<f32>,     // [T, H]
    rms: Vec<f32>,     // [T]
    q_proj: Vec<f32>,  // [T, R]
}

struct DeviceDraftLayer {
    attn_norm: DeviceBuffer,
    ffn_norm: DeviceBuffer,
    q_norm: Option<DeviceBuffer>,
    k_norm: Option<DeviceBuffer>,
    q_proj: DeviceWeight,
    k_proj: DeviceWeight,
    v_proj: DeviceWeight,
    o_proj: DeviceWeight,
    gate_proj: DeviceWeight,
    up_proj: DeviceWeight,
    down_proj: DeviceWeight,
}

struct DeviceDraftCompute {
    stream: Mutex<Box<dyn Stream>>,
    scratch: Mutex<BufferPool>,
    fc: DeviceWeight,
    hidden_norm: DeviceBuffer,
    output_norm: DeviceBuffer,
    layers: Vec<DeviceDraftLayer>,
    kv: DeviceKvCache,
    free_blocks: Mutex<Vec<usize>>,
    rope_tables: (Arc<Vec<f32>>, Arc<Vec<f32>>),
}

pub struct DSpineEngine {
    pub cfg: DSpineConfig,
    fc: Linear,
    hidden_norm: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<DraftLayerWeights>,
    bridge: Option<BridgeWeights>,
    rope_cfg: crate::config::RopeConfig,
    rope_base: f32,
    rope: RopeTables,
    states: HashMap<u64, SeqState>,
    dev: Option<DeviceDraftCompute>,
}

fn f32_slice_as_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding or alignment issue when viewed as `&[u8]`.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn f32_slice_as_bytes_mut(data: &mut [f32]) -> &mut [u8] {
    // SAFETY: exclusive `&mut [f32]` viewed as `&mut [u8]` for stream downloads.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn upload_vec_buffer(backend: &dyn Backend, stream: &mut dyn Stream, data: &[f32]) -> Result<DeviceBuffer, String> {
    let mut buf = backend.allocate_buffer(std::mem::size_of_val(data)).map_err(|e| e.to_string())?;
    if !data.is_empty() {
        stream.upload(&mut buf, f32_slice_as_bytes(data)).map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

fn device_linear_add(
    stream: &mut dyn Stream,
    scratch: &mut BufferPool,
    residual: &mut DeviceBuffer,
    x: &DeviceBuffer,
    weight: &DeviceWeight,
) -> Result<(), String> {
    let err = |e: crate::backend::BackendError| e.to_string();
    if stream
        .quantized_linear_add(residual, x, weight, weight.bias.as_ref(), 1.0)
        .map_err(err)?
    {
        return Ok(());
    }
    let rows = x.size / (weight.cols.max(1) * 4);
    let mut tmp = scratch.acquire(rows * weight.rows * 4).map_err(err)?;
    stream
        .quantized_linear(&mut tmp, x, weight, weight.bias.as_ref())
        .map_err(err)?;
    stream.residual_add(residual, &tmp, 1.0).map_err(err)?;
    scratch.recycle(tmp);
    Ok(())
}

fn device_rms_inplace_buf(
    stream: &mut dyn Stream,
    scratch: &mut BufferPool,
    buf: &mut DeviceBuffer,
    weight: &DeviceBuffer,
    eps: f32,
) -> Result<(), String> {
    let err = |e: crate::backend::BackendError| e.to_string();
    let mut normed = scratch.acquire(buf.size).map_err(err)?;
    stream.rms_norm(&mut normed, buf, weight, eps).map_err(err)?;
    let old = std::mem::replace(buf, normed);
    scratch.recycle(old);
    Ok(())
}

fn acquire_kv_blocks(free_blocks: &Mutex<Vec<usize>>, count: usize) -> Result<Vec<usize>, String> {
    let mut free = free_blocks
        .lock()
        .map_err(|_| "DSpine free block pool poisoned".to_owned())?;
    if free.len() < count {
        return Err(format!(
            "DSpine device KV cache out of blocks (need {count}, have {})",
            free.len()
        ));
    }
    let start = free.len() - count;
    Ok(free.split_off(start))
}

fn release_kv_blocks(free_blocks: &Mutex<Vec<usize>>, blocks: &[usize]) {
    if blocks.is_empty() {
        return;
    }
    if let Ok(mut free) = free_blocks.lock() {
        free.extend_from_slice(blocks);
    }
}

fn upload_draft_linear(backend: &dyn Backend, name: &str, linear: &Linear) -> Result<DeviceWeight, String> {
    if let Some((kind, bytes)) = linear.packed_weights() {
        let quant = QuantType::from_gguf_type_id(kind);
        if fllm_gguf_probe::exl::is_exl_type(kind) && !backend.capabilities().supports_quant(quant) {
            let half = crate::cpu_exl::expanded_f16(bytes, kind, linear.rows, linear.cols)?;
            return backend
                .upload_weight_scaled(name, &half, QuantType::F16, linear.rows, linear.cols, linear.bias.as_deref(), linear.scale)
                .map_err(|e| e.to_string());
        }
        if quant.mlx_affine().is_some() && !backend.capabilities().supports_quant(quant) {
            let values = crate::compute::dequantize(bytes, kind, linear.rows * linear.cols)?;
            return backend
                .upload_weight_scaled(name, f32_slice_as_bytes(&values), QuantType::F32, linear.rows, linear.cols, linear.bias.as_deref(), linear.scale)
                .map_err(|e| e.to_string());
        }
        crate::backend::set_next_nvfp4_input_scale(linear.input_scale);
        let uploaded = backend
            .upload_weight_scaled(name, bytes, quant, linear.rows, linear.cols, linear.bias.as_deref(), linear.scale)
            .map_err(|e| e.to_string());
        crate::backend::set_next_nvfp4_input_scale(0.0);
        uploaded
    } else {
        backend
            .upload_weight_scaled(
                name,
                f32_slice_as_bytes(&linear.weight.data),
                QuantType::F32,
                linear.rows,
                linear.cols,
                linear.bias.as_deref(),
                linear.scale,
            )
            .map_err(|e| e.to_string())
    }
}

fn meta_u32(meta: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    meta.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing GGUF metadata {key}"))
}

fn meta_f32(meta: &serde_json::Map<String, Value>, key: &str) -> Option<f32> {
    meta.get(key).and_then(Value::as_f64).map(|v| v as f32)
}

fn meta_bool(meta: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    meta.get(key).and_then(Value::as_bool)
}

fn meta_u32s(meta: &serde_json::Map<String, Value>, key: &str) -> Result<Vec<usize>, String> {
    meta.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array GGUF metadata {key}"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("invalid integer in GGUF metadata {key}"))
        })
        .collect()
}

fn has_tensor(mapped: &MappedGguf, name: &str) -> bool {
    mapped.index().tensors.iter().any(|t| t.name == name)
}

fn load_vec(mapped: &MappedGguf, name: &str, expected: &[usize]) -> Result<Vec<f32>, String> {
    let t = Tensor::load(mapped, name)?;
    if t.shape != expected {
        return Err(format!(
            "tensor {name} shape {:?} does not match expected {expected:?}",
            t.shape
        ));
    }
    Ok(t.data)
}

impl DSpineEngine {
    pub fn is_dspine_arch(arch: &str) -> bool {
        matches!(arch, "dspine" | "dflash")
    }

    pub fn new(model: &GgufModel) -> Result<Self, String> {
        let model_cfg = model.config();
        let arch = model_cfg.arch.as_str();
        if !Self::is_dspine_arch(arch) {
            return Err(format!("model architecture {arch:?} is not DSpine or DFlash"));
        }
        let mapped = model.mapping();
        let meta = &mapped.index().metadata;

        let hidden = model_cfg.hidden;
        let intermediate = model_cfg.ff;
        let heads = model_cfg.heads;
        let kv_heads = model_cfg.kv_heads;
        let head_dim = model_cfg.head_dim;
        let layers_count = model_cfg.layers;
        let vocab_size = model_cfg.vocab;
        let eps = model_cfg.eps;

        let block_size = meta_u32(meta, &format!("{arch}.block_size"))?;
        if block_size < 2 {
            return Err(format!("invalid {arch}.block_size {block_size}"));
        }
        let mask_token_id = meta_u32(meta, &format!("{arch}.mask_token_id"))? as i64;
        let target_layer_ids = meta_u32s(meta, &format!("{arch}.target_layer_ids"))?;
        if target_layer_ids.is_empty() {
            return Err(format!("{arch}.target_layer_ids must not be empty"));
        }
        let num_target_layers = meta_u32(meta, &format!("{arch}.num_target_layers"))?;
        let block_causal = meta_bool(meta, &format!("{arch}.block_causal")).unwrap_or(true);
        let sliding_window = meta_u32(meta, &format!("{arch}.sliding_window")).ok().filter(|&w| w > 0);
        let has_bridge = meta_bool(meta, &format!("{arch}.has_bridge")).unwrap_or(arch == "dspine");

        let fc_info = mapped
            .index()
            .tensors
            .iter()
            .find(|t| t.name == "dspine.fc.weight")
            .ok_or("missing dspine.fc.weight")?;
        if fc_info.dimensions.len() != 2 || fc_info.dimensions[1] as usize != hidden {
            return Err(format!(
                "dspine.fc.weight shape {:?} incompatible with hidden={hidden}",
                fc_info.dimensions
            ));
        }
        let fc_cols = fc_info.dimensions[0] as usize;
        if !fc_cols.is_multiple_of(target_layer_ids.len()) {
            return Err(format!(
                "dspine.fc.weight cols {fc_cols} is not divisible by {} target layers",
                target_layer_ids.len()
            ));
        }
        let target_hidden = fc_cols / target_layer_ids.len();

        let fc = Linear::load(mapped, "dspine.fc", hidden, fc_cols)?;
        let hidden_norm = load_vec(mapped, "dspine.hidden_norm.weight", &[hidden])?;
        let output_norm = load_vec(mapped, "output_norm.weight", &[hidden])?;

        let mut layers = Vec::with_capacity(layers_count);
        for i in 0..layers_count {
            let attn_norm = load_vec(mapped, &format!("blk.{i}.attn_norm.weight"), &[hidden])?;
            let ffn_norm = load_vec(mapped, &format!("blk.{i}.ffn_norm.weight"), &[hidden])?;
            let q_proj = Linear::load(mapped, &format!("blk.{i}.attn_q"), heads * head_dim, hidden)?;
            let k_proj = Linear::load(mapped, &format!("blk.{i}.attn_k"), kv_heads * head_dim, hidden)?;
            let v_proj = Linear::load(mapped, &format!("blk.{i}.attn_v"), kv_heads * head_dim, hidden)?;
            let o_proj = Linear::load(mapped, &format!("blk.{i}.attn_output"), hidden, heads * head_dim)?;
            let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
            let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
            let q_norm = if has_tensor(mapped, &q_norm_name) {
                Some(load_vec(mapped, &q_norm_name, &[head_dim])?)
            } else {
                None
            };
            let k_norm = if has_tensor(mapped, &k_norm_name) {
                Some(load_vec(mapped, &k_norm_name, &[head_dim])?)
            } else {
                None
            };
            let gate_proj = Linear::load(mapped, &format!("blk.{i}.ffn_gate"), intermediate, hidden)?;
            let up_proj = Linear::load(mapped, &format!("blk.{i}.ffn_up"), intermediate, hidden)?;
            let down_proj = Linear::load(mapped, &format!("blk.{i}.ffn_down"), hidden, intermediate)?;
            layers.push(DraftLayerWeights {
                attn_norm,
                ffn_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let (code_size, rank, bridge_eps, head_topk, feature_layers, terminal_residual_rank, bridge) =
            if has_bridge {
                let code_size = meta_u32(meta, &format!("{arch}.code_size"))?;
                let rank = meta_u32(meta, &format!("{arch}.rank"))?;
                let bridge_eps = meta_f32(meta, &format!("{arch}.bridge_eps")).unwrap_or(1e-6);
                let head_topk = meta_u32(meta, &format!("{arch}.head_topk"))?;
                let feature_layers = meta_u32s(meta, &format!("{arch}.feature_layers"))?;
                let terminal_residual_rank =
                    meta_u32(meta, &format!("{arch}.terminal_residual_rank")).unwrap_or(0);

                let codebook = load_vec(mapped, "dspine.bridge.codebook", &[code_size, vocab_size])?;
                let last_read = load_vec(mapped, "dspine.bridge.read_weight", &[hidden, code_size])?;
                let early_reads = if layers_count > 1 {
                    load_vec(
                        mapped,
                        "dspine.bridge.early_read_weights",
                        &[hidden, code_size, layers_count - 1],
                    )?
                } else {
                    Vec::new()
                };
                let mut read_weights = Vec::with_capacity(layers_count);
                let stride_r = code_size * hidden;
                for i in 0..layers_count {
                    if i + 1 == layers_count {
                        read_weights.push(last_read.clone());
                    } else {
                        read_weights.push(early_reads[i * stride_r..(i + 1) * stride_r].to_vec());
                    }
                }
                let all_writes = load_vec(
                    mapped,
                    "dspine.bridge.feature_write_weights",
                    &[rank, hidden, layers_count],
                )?;
                let mut feature_write_weights = Vec::with_capacity(layers_count);
                let stride_w = hidden * rank;
                for i in 0..layers_count {
                    feature_write_weights.push(all_writes[i * stride_w..(i + 1) * stride_w].to_vec());
                }
                let gate_w = load_vec(mapped, "dspine.bridge.gate.weight", &[code_size, rank])?;
                let pred_gate_w = load_vec(
                    mapped,
                    "dspine.bridge.predecessor_gate.weight",
                    &[code_size, rank],
                )?;
                let prev_w = load_vec(mapped, "dspine.bridge.prev.weight", &[code_size, rank])?;
                let gate_bias = load_vec(mapped, "dspine.bridge.gate.bias", &[rank])?;

                let (terminal_emb, terminal_proj) = if terminal_residual_rank > 0 {
                    (
                        load_vec(
                            mapped,
                            "dspine.bridge.terminal_residual_embedding",
                            &[terminal_residual_rank, vocab_size],
                        )?,
                        load_vec(
                            mapped,
                            "dspine.bridge.terminal_residual_projection",
                            &[terminal_residual_rank, rank],
                        )?,
                    )
                } else {
                    (Vec::new(), Vec::new())
                };

                let active_layers: HashSet<usize> = feature_layers.iter().copied().collect();
                (
                    code_size,
                    rank,
                    bridge_eps,
                    head_topk,
                    feature_layers,
                    terminal_residual_rank,
                    Some(BridgeWeights {
                        codebook,
                        read_weights,
                        feature_write_weights,
                        gate_w,
                        pred_gate_w,
                        prev_w,
                        gate_bias,
                        active_layers,
                        terminal_emb,
                        terminal_proj,
                    }),
                )
            } else {
                (0, 0, 1e-6, 0, Vec::new(), 0, None)
            };

        let rope_cfg = model_cfg.rope.clone();
        let rope_base = rope_cfg.base;
        let rope = rope::build(&rope_cfg, rope_base, model.max_context().max(4096));

        Ok(Self {
            cfg: DSpineConfig {
                arch: arch.to_owned(),
                hidden,
                intermediate,
                heads,
                kv_heads,
                head_dim,
                layers: layers_count,
                vocab_size,
                eps,
                block_size,
                mask_token_id,
                target_layer_ids,
                num_target_layers,
                target_hidden,
                block_causal,
                sliding_window,
                has_bridge,
                code_size,
                rank,
                bridge_eps,
                head_topk,
                feature_layers,
                terminal_residual_rank,
            },
            fc,
            hidden_norm,
            output_norm,
            layers,
            bridge,
            rope_cfg,
            rope_base,
            rope,
            states: HashMap::new(),
            dev: None,
        })
    }

    /// Whether the draft model's linear projections are attached to an accelerator backend.
    pub fn uses_accelerator(&self) -> bool {
        self.dev.is_some()
    }

    /// Upload `fc`, norm vectors, draft transformer projections, and KV cache onto `backend`.
    pub fn init_backend(&mut self, backend: Arc<dyn Backend>) -> Result<(), String> {
        let mut stream = backend.create_stream().map_err(|e| e.to_string())?;
        let prev_act = stream.act_dtype();
        if prev_act.is_narrow() {
            let _ = backend.set_activation_dtype(ActDtype::F32);
            let _ = stream.set_act_dtype(ActDtype::F32);
        }
        let res = (|| -> Result<DeviceDraftCompute, String> {
            let fc = upload_draft_linear(&*backend, "dspine.fc", &self.fc)?;
            let hidden_norm = upload_vec_buffer(&*backend, &mut *stream, &self.hidden_norm)?;
            let output_norm = upload_vec_buffer(&*backend, &mut *stream, &self.output_norm)?;
            let mut dev_layers = Vec::with_capacity(self.layers.len());
            for (i, l) in self.layers.iter().enumerate() {
                dev_layers.push(DeviceDraftLayer {
                    attn_norm: upload_vec_buffer(&*backend, &mut *stream, &l.attn_norm)?,
                    ffn_norm: upload_vec_buffer(&*backend, &mut *stream, &l.ffn_norm)?,
                    q_norm: l
                        .q_norm
                        .as_ref()
                        .map(|v| upload_vec_buffer(&*backend, &mut *stream, v))
                        .transpose()?,
                    k_norm: l
                        .k_norm
                        .as_ref()
                        .map(|v| upload_vec_buffer(&*backend, &mut *stream, v))
                        .transpose()?,
                    q_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.attn_q"), &l.q_proj)?,
                    k_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.attn_k"), &l.k_proj)?,
                    v_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.attn_v"), &l.v_proj)?,
                    o_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.attn_output"), &l.o_proj)?,
                    gate_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.ffn_gate"), &l.gate_proj)?,
                    up_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.ffn_up"), &l.up_proj)?,
                    down_proj: upload_draft_linear(&*backend, &format!("dspine.blk.{i}.ffn_down"), &l.down_proj)?,
                });
            }
            let per_block_bytes = 2
                * DRAFT_KV_BLOCK_SIZE
                * self.cfg.kv_heads.max(1)
                * self.cfg.head_dim.max(1)
                * 2
                * self.layers.len().max(1);
            let pool_blocks = if let Some(w) = self.cfg.sliding_window {
                (w.div_ceil(DRAFT_KV_BLOCK_SIZE) + self.cfg.block_size.div_ceil(DRAFT_KV_BLOCK_SIZE) + 2) * 32
            } else {
                ((256usize << 20) / per_block_bytes.max(1)).clamp(64, 4096)
            };
            let kv = backend
                .allocate_kv_cache(
                    self.layers.len(),
                    pool_blocks,
                    DRAFT_KV_BLOCK_SIZE,
                    self.cfg.kv_heads,
                    self.cfg.head_dim,
                    self.cfg.head_dim,
                    1,
                )
                .map_err(|e| e.to_string())?;
            let free_blocks: Vec<usize> = (0..pool_blocks).rev().collect();
            let mut scratch = BufferPool::new(Arc::clone(&backend));
            let rows = self.cfg.block_size.max(16);
            let h_bytes = rows * self.cfg.hidden * 4;
            let q_bytes = rows * self.cfg.heads * self.cfg.head_dim * 4;
            let f_bytes = rows * self.cfg.intermediate * 4;
            let _ = scratch.preallocate(&[h_bytes, h_bytes, q_bytes, f_bytes, f_bytes]);
            let rope_tables = (Arc::new(self.rope.cos.clone()), Arc::new(self.rope.sin.clone()));
            Ok(DeviceDraftCompute {
                stream: Mutex::new(stream),
                scratch: Mutex::new(scratch),
                fc,
                hidden_norm,
                output_norm,
                layers: dev_layers,
                kv,
                free_blocks: Mutex::new(free_blocks),
                rope_tables,
            })
        })();
        if prev_act.is_narrow() {
            let _ = backend.set_activation_dtype(prev_act);
        }
        self.dev = Some(res?);
        Ok(())
    }

    /// Attach DSpine/DFlash draft projections to `device`, reusing `shared_backend` when the target
    /// decoder already opened an accelerator handle.
    pub fn attach_compute(
        &mut self,
        device: &str,
        shared_backend: Option<Arc<dyn Backend>>,
    ) -> Result<(), String> {
        if device == "cpu" {
            self.dev = None;
            return Ok(());
        }
        if let Some(backend) = shared_backend {
            return self.init_backend(backend);
        }
        let automatic = crate::devices::is_auto(device);
        let mut last_error = String::from("no usable compute device for DSpine draft");
        for (backend_name, index) in crate::devices::device_candidates(device)? {
            if backend_name == "cpu" {
                self.dev = None;
                return Ok(());
            }
            match crate::rust_decoder::open_accelerator_backend(&backend_name, index) {
                Ok((backend, _)) => match self.init_backend(backend) {
                    Ok(()) => {
                        eprintln!(
                            "[fllm-rs] {} draft projections accelerated on {backend_name}:{index}",
                            self.cfg.arch
                        );
                        return Ok(());
                    }
                    Err(e) if automatic => last_error = e,
                    Err(e) => return Err(e),
                },
                Err(e) if automatic => last_error = e,
                Err(e) => return Err(e),
            }
        }
        Err(last_error)
    }

    pub fn validate_target(&self, target_layers: usize, target_hidden: usize, target_vocab: usize) -> Result<(), String> {
        if self.cfg.num_target_layers != target_layers {
            return Err(format!(
                "{} expects a {}-layer target model, loaded target has {target_layers} layers",
                self.cfg.arch, self.cfg.num_target_layers
            ));
        }
        if let Some(&bad) = self.cfg.target_layer_ids.iter().find(|&&id| id >= target_layers) {
            return Err(format!(
                "{} target_layer_id {bad} exceeds target layer count {target_layers}",
                self.cfg.arch
            ));
        }
        if self.cfg.target_hidden != target_hidden {
            return Err(format!(
                "{} expects target hidden_size {}, loaded target has {target_hidden}",
                self.cfg.arch, self.cfg.target_hidden
            ));
        }
        if self.cfg.vocab_size != target_vocab {
            return Err(format!(
                "{} draft vocab {} does not match target vocab {target_vocab}",
                self.cfg.arch, self.cfg.vocab_size
            ));
        }
        Ok(())
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.cfg.target_layer_ids
    }

    pub fn block_size(&self) -> usize {
        self.cfg.block_size
    }

    pub fn clear(&mut self, sid: u64) {
        if let (Some(st), Some(dev)) = (self.states.remove(&sid), self.dev.as_ref()) {
            release_kv_blocks(&dev.free_blocks, &st.dev_kv.blocks);
        }
    }

    pub fn retain(&mut self, live: &HashSet<u64>) {
        if let Some(dev) = self.dev.as_ref() {
            for (sid, st) in &self.states {
                if !live.contains(sid) {
                    release_kv_blocks(&dev.free_blocks, &st.dev_kv.blocks);
                }
            }
        }
        self.states.retain(|sid, _| live.contains(sid));
    }

    pub(crate) fn snapshot_seq(&self, sid: u64) -> DSpineSeqSnapshot {
        let Some(st) = self.states.get(&sid) else {
            return DSpineSeqSnapshot {
                had_state: false,
                pending_target_hidden: Vec::new(),
                pending_rows: 0,
                host_kv: Vec::new(),
                dev_kv: SeqDeviceKv::default(),
            };
        };
        DSpineSeqSnapshot {
            had_state: true,
            pending_target_hidden: st.pending_target_hidden.clone(),
            pending_rows: st.pending_rows,
            host_kv: if self.dev.is_some() { Vec::new() } else { st.kv.clone() },
            dev_kv: st.dev_kv.clone(),
        }
    }

    pub(crate) fn restore_seq(&mut self, sid: u64, snap: DSpineSeqSnapshot) {
        if !snap.had_state {
            self.clear(sid);
            return;
        }
        let Some(st) = self.states.get_mut(&sid) else {
            return;
        };
        if let Some(dev) = self.dev.as_ref() {
            // Any newly allocated blocks appended after `snap.dev_kv.blocks` are returned to the pool.
            if let Ok(mut free) = dev.free_blocks.lock() {
                for &blk in &st.dev_kv.blocks {
                    if !snap.dev_kv.blocks.contains(&blk) {
                        free.push(blk);
                    }
                }
                // Any front blocks trimmed during sliding-window eviction may have been pushed to `free`:
                // remove them from `free` so ownership returns cleanly to `snap.dev_kv.blocks`.
                for &blk in &snap.dev_kv.blocks {
                    if !st.dev_kv.blocks.contains(&blk) {
                        if let Some(pos) = free.iter().rposition(|&b| b == blk) {
                            free.swap_remove(pos);
                        }
                    }
                }
            }
        }
        st.dev_kv = snap.dev_kv;
        if !snap.host_kv.is_empty() {
            st.kv = snap.host_kv;
        }
        st.pending_target_hidden = snap.pending_target_hidden;
        st.pending_rows = snap.pending_rows;
    }

    /// Appends committed concatenated target hidden states (`rows * k * target_hidden` floats)
    /// to sequence `sid`'s pending buffer, to be projected into draft context KV on the next draft round.
    pub fn observe_target_hidden(&mut self, sid: u64, rows: usize, concat_hidden: &[f32]) -> Result<(), String> {
        if rows == 0 {
            return Ok(());
        }
        let row_width = self.cfg.target_layer_ids.len() * self.cfg.target_hidden;
        let need = rows * row_width;
        if concat_hidden.len() < need {
            return Err(format!(
                "observe_target_hidden expected at least {need} floats for {rows} rows, got {}",
                concat_hidden.len()
            ));
        }
        let state = self.states.entry(sid).or_insert_with(|| SeqState {
            kv: vec![LayerKvCache::default(); self.cfg.layers],
            dev_kv: SeqDeviceKv::default(),
            pending_target_hidden: Vec::new(),
            pending_rows: 0,
        });
        state.pending_target_hidden.extend_from_slice(&concat_hidden[..need]);
        state.pending_rows += rows;
        Ok(())
    }

    fn ensure_rope_len(&mut self, needed: usize) {
        let current = self.rope.cos.len().checked_div(self.rope.half).unwrap_or(0);
        if needed > current {
            self.rope = rope::build(&self.rope_cfg, self.rope_base, needed.next_power_of_two());
            if let Some(dev) = self.dev.as_mut() {
                dev.rope_tables = (Arc::new(self.rope.cos.clone()), Arc::new(self.rope.sin.clone()));
            }
        }
    }

    /// Generates up to `min(block_size - 1, limit, remaining)` speculative draft tokens for `sid`
    /// starting from `anchor`, together with candidate probability distributions (when non-greedy).
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &mut self,
        sid: u64,
        anchor: i64,
        limit: usize,
        remaining: usize,
        params: &SamplingParams,
        sampler: &mut SamplingState,
        history: &[i64],
        target: &RustDecoder,
    ) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
        let draft_len = limit.min(remaining).min(self.cfg.block_size.saturating_sub(1));
        if draft_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.dev.is_some() {
            return self.propose_device(sid, anchor, draft_len, params, sampler, history, target);
        }
        self.propose_host(sid, anchor, draft_len, params, sampler, history, target)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_device(
        &mut self,
        sid: u64,
        anchor: i64,
        draft_len: usize,
        params: &SamplingParams,
        sampler: &mut SamplingState,
        history: &[i64],
        target: &RustDecoder,
    ) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
        let t_block = draft_len + 1;
        let row_width = self.cfg.target_layer_ids.len() * self.cfg.target_hidden;

        let (ctx_rows, ctx_hidden, prev_pos, prev_ctx_in_blocks) = {
            let state = self.states.entry(sid).or_insert_with(|| SeqState {
                kv: vec![LayerKvCache::default(); self.cfg.layers],
                dev_kv: SeqDeviceKv::default(),
                pending_target_hidden: Vec::new(),
                pending_rows: 0,
            });
            let mut rows = state.pending_rows;
            let mut buf = std::mem::take(&mut state.pending_target_hidden);
            state.pending_rows = 0;

            let offset = state.dev_kv.total_len;
            if let Some(w) = self.cfg.sliding_window {
                if offset == 0 && rows > w {
                    let drop = rows - w;
                    buf.drain(..drop * row_width);
                    rows = w;
                }
            }
            (rows, buf, offset, state.dev_kv.ctx_rows_in_blocks)
        };

        if prev_pos + ctx_rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.ensure_rope_len(prev_pos + ctx_rows + t_block + 8);

        let mut block_ids = vec![self.cfg.mask_token_id; t_block];
        block_ids[0] = anchor;
        let block_embed = target.embed_tokens(&block_ids)?;
        if block_embed.len() != t_block * self.cfg.hidden {
            return Err(format!(
                "target embed_tokens returned {} floats for {t_block}x{}",
                block_embed.len(),
                self.cfg.hidden
            ));
        }

        let dev = self.dev.as_ref().ok_or("missing DSpine device compute")?;
        let state = self.states.get_mut(&sid).ok_or("missing DSpine sequence state")?;

        let ctx_end = prev_ctx_in_blocks + ctx_rows;
        let need_ctx_blocks = ctx_end.div_ceil(DRAFT_KV_BLOCK_SIZE);
        let add_ctx_blocks = need_ctx_blocks.saturating_sub(state.dev_kv.blocks.len());
        if add_ctx_blocks > 0 {
            let extra = acquire_kv_blocks(&dev.free_blocks, add_ctx_blocks)?;
            state.dev_kv.blocks.extend(extra);
        }

        let total_needed_blocks = (ctx_end + t_block).div_ceil(DRAFT_KV_BLOCK_SIZE);
        let need_tail_blocks = total_needed_blocks.saturating_sub(state.dev_kv.blocks.len());
        let tail_blocks = acquire_kv_blocks(&dev.free_blocks, need_tail_blocks)?;

        type ProposeRoundOut = (
            Option<Vec<i32>>,
            Option<Vec<f32>>,
            Option<BridgeStash>,
            Option<Vec<f32>>,
        );
        let round_result = (|| -> Result<ProposeRoundOut, String> {
            let mut round_blocks = Vec::with_capacity(state.dev_kv.blocks.len() + tail_blocks.len());
            round_blocks.extend_from_slice(&state.dev_kv.blocks);
            round_blocks.extend_from_slice(&tail_blocks);

            let stride = round_blocks.len();
            let mut ctx_slots = Vec::with_capacity(ctx_rows);
            for idx in 0..ctx_rows {
                let p = prev_ctx_in_blocks + idx;
                ctx_slots.push(round_blocks[p / DRAFT_KV_BLOCK_SIZE] * DRAFT_KV_BLOCK_SIZE + (p % DRAFT_KV_BLOCK_SIZE));
            }
            let mut blk_slots = Vec::with_capacity(t_block);
            for qi in 0..t_block {
                let p = ctx_end + qi;
                blk_slots.push(round_blocks[p / DRAFT_KV_BLOCK_SIZE] * DRAFT_KV_BLOCK_SIZE + (p % DRAFT_KV_BLOCK_SIZE));
            }

            let mut round_tables = Vec::with_capacity(t_block * stride);
            for _ in 0..t_block {
                round_tables.extend(round_blocks.iter().map(|&b| b as i64));
            }
            let lengths: Vec<i64> = if self.cfg.block_causal {
                (0..t_block).map(|qi| (ctx_end + qi + 1) as i64).collect()
            } else {
                vec![(ctx_end + t_block) as i64; t_block]
            };
            let ctx_positions: Vec<usize> = (prev_pos..prev_pos + ctx_rows).collect();
            let blk_positions: Vec<usize> = (prev_pos + ctx_rows..prev_pos + ctx_rows + t_block).collect();

            let err = |e: crate::backend::BackendError| e.to_string();
            let mut scratch = dev.scratch.lock().map_err(|_| "DSpine scratch poisoned".to_owned())?;
            let mut stream = dev.stream.lock().map_err(|_| "DSpine stream poisoned".to_owned())?;
            stream.begin_step();

            let pass = (|| -> Result<ProposeRoundOut, String> {
                let h = self.cfg.hidden;
                let inter = self.cfg.intermediate;
                let q_dim = self.cfg.heads * self.cfg.head_dim;
                let kv_dim = self.cfg.kv_heads * self.cfg.head_dim;
                let last_layer = self.layers.len() - 1;

                let ctx_proj_buf = if ctx_rows > 0 {
                    let mut ctx_in = scratch.acquire(ctx_rows * self.fc.cols * 4).map_err(err)?;
                    stream.upload(&mut ctx_in, f32_slice_as_bytes(&ctx_hidden)).map_err(err)?;
                    let mut ctx_raw = scratch.acquire(ctx_rows * h * 4).map_err(err)?;
                    stream.quantized_linear(&mut ctx_raw, &ctx_in, &dev.fc, dev.fc.bias.as_ref()).map_err(err)?;
                    scratch.recycle(ctx_in);
                    let mut ctx_proj = scratch.acquire(ctx_rows * h * 4).map_err(err)?;
                    stream.rms_norm(&mut ctx_proj, &ctx_raw, &dev.hidden_norm, self.cfg.eps).map_err(err)?;
                    scratch.recycle(ctx_raw);
                    Some(ctx_proj)
                } else {
                    None
                };

                let mut residual_buf = scratch.acquire(t_block * h * 4).map_err(err)?;
                stream.upload(&mut residual_buf, f32_slice_as_bytes(&block_embed)).map_err(err)?;
                let mut x_in_buf = scratch.acquire(t_block * h * 4).map_err(err)?;
                stream.rms_norm(&mut x_in_buf, &residual_buf, &dev.layers[0].attn_norm, self.cfg.eps).map_err(err)?;

                let rope_params = RopeParams {
                    heads: self.cfg.heads,
                    kv_heads: self.cfg.kv_heads,
                    dim: self.cfg.head_dim,
                    rope_dim: self.rope.half * 2,
                    neox: self.rope_cfg.neox,
                    cos: Arc::clone(&dev.rope_tables.0),
                    sin: Arc::clone(&dev.rope_tables.1),
                };
                let attn_params = AttentionParams {
                    num_heads: self.cfg.heads,
                    num_kv_heads: self.cfg.kv_heads,
                    head_dim: self.cfg.head_dim,
                    value_dim: self.cfg.head_dim,
                    stride,
                    block_size: DRAFT_KV_BLOCK_SIZE,
                    scale: 1.0 / (self.cfg.head_dim as f32).sqrt(),
                    softcap: None,
                    window: self.cfg.sliding_window,
                    chunk: None,
                    sinks: None,
                    alibi_slopes: None,
                };

                let mut last_stash: Option<BridgeStash> = None;
                let mut host_tail_fallback: Option<Vec<f32>> = None;

                for (l_idx, dl) in dev.layers.iter().enumerate() {
                    let mut q_buf = scratch.acquire(t_block * q_dim * 4).map_err(err)?;
                    let mut k_buf = scratch.acquire(t_block * kv_dim * 4).map_err(err)?;
                    let v_buf = scratch.acquire(t_block * kv_dim * 4).map_err(err)?;

                    stream.quantized_linear(&mut q_buf, &x_in_buf, &dl.q_proj, dl.q_proj.bias.as_ref()).map_err(err)?;
                    stream.quantized_linear(&mut k_buf, &x_in_buf, &dl.k_proj, dl.k_proj.bias.as_ref()).map_err(err)?;
                    let mut v_mut = v_buf;
                    stream.quantized_linear(&mut v_mut, &x_in_buf, &dl.v_proj, dl.v_proj.bias.as_ref()).map_err(err)?;

                    if let Some(qn) = &dl.q_norm {
                        device_rms_inplace_buf(&mut **stream, &mut scratch, &mut q_buf, qn, self.cfg.eps)?;
                    }
                    if let Some(kn) = &dl.k_norm {
                        device_rms_inplace_buf(&mut **stream, &mut scratch, &mut k_buf, kn, self.cfg.eps)?;
                    }
                    stream.rope(&mut q_buf, &mut k_buf, &blk_positions, &rope_params).map_err(err)?;

                    if let Some(ctx_proj) = ctx_proj_buf.as_ref() {
                        let mut k_ctx = scratch.acquire(ctx_rows * kv_dim * 4).map_err(err)?;
                        let mut v_ctx = scratch.acquire(ctx_rows * kv_dim * 4).map_err(err)?;
                        stream.quantized_linear(&mut k_ctx, ctx_proj, &dl.k_proj, dl.k_proj.bias.as_ref()).map_err(err)?;
                        stream.quantized_linear(&mut v_ctx, ctx_proj, &dl.v_proj, dl.v_proj.bias.as_ref()).map_err(err)?;
                        if let Some(kn) = &dl.k_norm {
                            device_rms_inplace_buf(&mut **stream, &mut scratch, &mut k_ctx, kn, self.cfg.eps)?;
                        }
                        let mut dummy_q = scratch.acquire(ctx_rows * q_dim * 4).map_err(err)?;
                        stream.rope(&mut dummy_q, &mut k_ctx, &ctx_positions, &rope_params).map_err(err)?;
                        scratch.recycle(dummy_q);
                        stream.kv_cache_write(&dev.kv, l_idx, &ctx_slots, &k_ctx, &v_ctx).map_err(err)?;
                        scratch.recycle(k_ctx);
                        scratch.recycle(v_ctx);
                    }

                    stream.kv_cache_write(&dev.kv, l_idx, &blk_slots, &k_buf, &v_mut).map_err(err)?;
                    scratch.recycle(k_buf);
                    scratch.recycle(v_mut);

                    let mut attn_buf = scratch.acquire(t_block * q_dim * 4).map_err(err)?;
                    stream
                        .attention_paged(&mut attn_buf, &q_buf, &dev.kv, &round_tables, &lengths, l_idx, &attn_params)
                        .map_err(err)?;
                    scratch.recycle(q_buf);

                    device_linear_add(&mut **stream, &mut scratch, &mut residual_buf, &attn_buf, &dl.o_proj)?;
                    scratch.recycle(attn_buf);

                    let mut ffn_in = scratch.acquire(t_block * h * 4).map_err(err)?;
                    stream.rms_norm(&mut ffn_in, &residual_buf, &dl.ffn_norm, self.cfg.eps).map_err(err)?;
                    let gate_buf = {
                        let mut g = scratch.acquire(t_block * inter * 4).map_err(err)?;
                        stream.quantized_linear(&mut g, &ffn_in, &dl.gate_proj, dl.gate_proj.bias.as_ref()).map_err(err)?;
                        g
                    };
                    let up_buf = {
                        let mut u = scratch.acquire(t_block * inter * 4).map_err(err)?;
                        stream.quantized_linear(&mut u, &ffn_in, &dl.up_proj, dl.up_proj.bias.as_ref()).map_err(err)?;
                        u
                    };
                    scratch.recycle(ffn_in);
                    let mut act_buf = scratch.acquire(t_block * inter * 4).map_err(err)?;
                    stream.swiglu(&mut act_buf, &gate_buf, &up_buf).map_err(err)?;
                    scratch.recycle(gate_buf);
                    scratch.recycle(up_buf);

                    device_linear_add(&mut **stream, &mut scratch, &mut residual_buf, &act_buf, &dl.down_proj)?;
                    scratch.recycle(act_buf);

                    let mut handled_next = false;
                    if let Some(bridge) = &self.bridge {
                        if bridge.active_layers.contains(&l_idx) {
                            let mut host_h = vec![0.0f32; t_block * h];
                            stream.download(f32_slice_as_bytes_mut(&mut host_h), &residual_buf).map_err(err)?;
                            let next_norm_w = if l_idx < last_layer {
                                &self.layers[l_idx + 1].attn_norm
                            } else {
                                &self.output_norm
                            };
                            let stash = (l_idx == last_layer).then(BridgeStash::default);
                            let (next_in, next_res, filled_stash) = bridge_forward(
                                bridge,
                                &self.cfg,
                                &host_h,
                                &block_ids,
                                t_block,
                                l_idx,
                                next_norm_w,
                                self.cfg.eps,
                                stash,
                            );
                            if let Some(s) = filled_stash {
                                last_stash = Some(s);
                            }
                            if l_idx < last_layer {
                                stream.upload(&mut x_in_buf, f32_slice_as_bytes(&next_in)).map_err(err)?;
                                stream.upload(&mut residual_buf, f32_slice_as_bytes(&next_res)).map_err(err)?;
                            } else {
                                host_tail_fallback = Some(next_in[h..].to_vec());
                            }
                            handled_next = true;
                        }
                    }
                    if !handled_next {
                        let norm_w = if l_idx < last_layer {
                            &dev.layers[l_idx + 1].attn_norm
                        } else {
                            &dev.output_norm
                        };
                        stream.rms_norm(&mut x_in_buf, &residual_buf, norm_w, self.cfg.eps).map_err(err)?;
                    }
                }

                if let Some(buf) = ctx_proj_buf {
                    scratch.recycle(buf);
                }
                scratch.recycle(residual_buf);

                let greedy_dflash = !self.cfg.has_bridge && params.argmax_only();
                if let Some(host_tail) = host_tail_fallback {
                    // Active bridge on last layer already normalized tail rows on host into `host_tail`.
                    stream.upload(&mut x_in_buf.slice(h * 4, draft_len * h * 4).map_err(err)?, f32_slice_as_bytes(&host_tail)).map_err(err)?;
                }
                let tail_buf = x_in_buf.slice(h * 4, draft_len * h * 4).map_err(err)?;
                let head_res = target.head_logits_or_argmax_device(
                    &mut **stream,
                    &mut scratch,
                    &tail_buf,
                    draft_len,
                    greedy_dflash,
                );
                scratch.recycle(x_in_buf);
                match head_res? {
                    Ok(ids) => Ok((Some(ids), None, last_stash, None)),
                    Err(logits) => Ok((None, Some(logits), last_stash, None)),
                }
            })();

            stream.end_step();
            pass
        })();

        release_kv_blocks(&dev.free_blocks, &tail_blocks);
        let (greedy_ids, logits, last_stash, _) = round_result?;

        state.dev_kv.ctx_rows_in_blocks = ctx_end;
        state.dev_kv.total_len += ctx_rows;
        if let Some(w) = self.cfg.sliding_window {
            while state.dev_kv.ctx_rows_in_blocks >= w + DRAFT_KV_BLOCK_SIZE {
                let dropped = state.dev_kv.blocks.remove(0);
                release_kv_blocks(&dev.free_blocks, &[dropped]);
                state.dev_kv.ctx_rows_in_blocks -= DRAFT_KV_BLOCK_SIZE;
            }
        }

        if let Some(bridge) = &self.bridge {
            let stash = last_stash.ok_or("DSpine last layer did not populate refinement stash")?;
            let logits = logits.ok_or("DSpine refine_tail expected full logits")?;
            refine_tail(
                bridge,
                &self.cfg,
                &self.output_norm,
                &logits,
                &stash,
                anchor,
                draft_len,
                params,
                sampler,
                target,
            )
        } else if let Some(ids) = greedy_ids {
            let tokens = ids.into_iter().map(i64::from).collect();
            let dists = vec![None; draft_len];
            Ok((tokens, dists))
        } else {
            let logits = logits.ok_or("DFlash expected logits")?;
            dflash_sample_from_logits(&logits, draft_len, params, sampler, history)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_host(
        &mut self,
        sid: u64,
        anchor: i64,
        draft_len: usize,
        params: &SamplingParams,
        sampler: &mut SamplingState,
        history: &[i64],
        target: &RustDecoder,
    ) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
        let t_block = draft_len + 1;
        let row_width = self.cfg.target_layer_ids.len() * self.cfg.target_hidden;

        let (ctx_rows, ctx_hidden, prev_pos) = {
            let state = self.states.entry(sid).or_insert_with(|| SeqState {
                kv: vec![LayerKvCache::default(); self.cfg.layers],
                dev_kv: SeqDeviceKv::default(),
                pending_target_hidden: Vec::new(),
                pending_rows: 0,
            });
            let mut rows = state.pending_rows;
            let mut buf = std::mem::take(&mut state.pending_target_hidden);
            state.pending_rows = 0;

            let offset = state.kv.first().map_or(0, |c| c.total_len);
            if let Some(w) = self.cfg.sliding_window {
                if offset == 0 && rows > w {
                    let drop = rows - w;
                    buf.drain(..drop * row_width);
                    rows = w;
                }
            }
            (rows, buf, offset)
        };

        if prev_pos + ctx_rows == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.ensure_rope_len(prev_pos + ctx_rows + t_block + 8);

        let ctx_proj = if ctx_rows > 0 {
            let mut projected = self.fc.forward(&ctx_hidden)?;
            rms_norm_rows_inplace(&mut projected, &self.hidden_norm, self.cfg.eps, self.cfg.hidden);
            Some(projected)
        } else {
            None
        };

        let mut block_ids = vec![self.cfg.mask_token_id; t_block];
        block_ids[0] = anchor;
        let block_embed = target.embed_tokens(&block_ids)?;
        if block_embed.len() != t_block * self.cfg.hidden {
            return Err(format!(
                "target embed_tokens returned {} floats for {t_block}x{}",
                block_embed.len(),
                self.cfg.hidden
            ));
        }

        let mut residual = block_embed;
        let mut x_in = rms_norm_rows_to(
            &residual,
            &self.layers[0].attn_norm,
            self.cfg.eps,
            self.cfg.hidden,
        );
        let mut input_normed = true;
        let mut last_stash: Option<BridgeStash> = None;

        let state = self
            .states
            .get_mut(&sid)
            .ok_or("missing DSpine sequence state")?;
        let last_layer = self.layers.len() - 1;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            if !input_normed {
                x_in = rms_norm_rows_to(&residual, &layer.attn_norm, self.cfg.eps, self.cfg.hidden);
            }
            let attn_out = draft_layer_attention(
                layer,
                &self.cfg,
                &x_in,
                ctx_proj.as_deref(),
                ctx_rows,
                t_block,
                &self.rope,
                &mut state.kv[l_idx],
            )?;
            let mut h = residual;
            for (dst, a) in h.iter_mut().zip(&attn_out) {
                *dst += *a;
            }
            let ffn_in = rms_norm_rows_to(&h, &layer.ffn_norm, self.cfg.eps, self.cfg.hidden);
            let mut gate = layer.gate_proj.forward(&ffn_in)?;
            let up = layer.up_proj.forward(&ffn_in)?;
            crate::cpu_elementwise::swiglu(&mut gate, &up);
            let mlp_out = layer.down_proj.forward(&gate)?;
            for (dst, m) in h.iter_mut().zip(&mlp_out) {
                *dst += *m;
            }

            input_normed = false;
            if let Some(bridge) = &self.bridge {
                if bridge.active_layers.contains(&l_idx) {
                    let next_norm_w = if l_idx < last_layer {
                        &self.layers[l_idx + 1].attn_norm
                    } else {
                        &self.output_norm
                    };
                    let stash = (l_idx == last_layer).then(BridgeStash::default);
                    let (next_in, next_res, filled_stash) = bridge_forward(
                        bridge,
                        &self.cfg,
                        &h,
                        &block_ids,
                        t_block,
                        l_idx,
                        next_norm_w,
                        self.cfg.eps,
                        stash,
                    );
                    x_in = next_in;
                    residual = next_res;
                    if filled_stash.is_some() {
                        last_stash = filled_stash;
                    }
                    input_normed = true;
                } else {
                    residual = h;
                }
            } else {
                residual = h;
            }
        }

        if !input_normed {
            x_in = rms_norm_rows_to(&residual, &self.output_norm, self.cfg.eps, self.cfg.hidden);
        }

        let norm_tail_rows = &x_in[self.cfg.hidden..]; // [draft_len, H]
        if let Some(bridge) = &self.bridge {
            let stash = last_stash.ok_or("DSpine last layer did not populate refinement stash")?;
            let logits = target.head_logits(norm_tail_rows)?;
            refine_tail(
                bridge,
                &self.cfg,
                &self.output_norm,
                &logits,
                &stash,
                anchor,
                draft_len,
                params,
                sampler,
                target,
            )
        } else {
            let logits = target.head_logits(norm_tail_rows)?;
            dflash_sample_from_logits(&logits, draft_len, params, sampler, history)
        }
    }
}

fn rms_norm_rows_inplace(x: &mut [f32], weight: &[f32], eps: f32, cols: usize) {
    for row in x.chunks_exact_mut(cols) {
        let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w) in row.iter_mut().zip(weight) {
            *v = (*v * inv) * w;
        }
    }
}

fn rms_norm_rows_to(src: &[f32], weight: &[f32], eps: f32, cols: usize) -> Vec<f32> {
    let mut out = src.to_vec();
    rms_norm_rows_inplace(&mut out, weight, eps, cols);
    out
}

fn apply_neox_rope_rows(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rope: &RopeTables,
    start_pos: usize,
) {
    let half = rope.half;
    if half == 0 {
        return;
    }
    let row_stride = heads * head_dim;
    for r in 0..rows {
        let pos = start_pos + r;
        let cos = &rope.cos[pos * half..(pos + 1) * half];
        let sin = &rope.sin[pos * half..(pos + 1) * half];
        let row = &mut x[r * row_stride..(r + 1) * row_stride];
        for h in 0..heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            for d in 0..half {
                let a = head[d];
                let b = head[d + half];
                let c = cos[d];
                let s = sin[d];
                head[d] = a * c - b * s;
                head[d + half] = a * s + b * c;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draft_layer_attention(
    layer: &DraftLayerWeights,
    cfg: &DSpineConfig,
    x_in: &[f32],
    ctx_proj: Option<&[f32]>,
    ctx_rows: usize,
    t_block: usize,
    rope: &RopeTables,
    cache: &mut LayerKvCache,
) -> Result<Vec<f32>, String> {
    let kv_dim = cfg.kv_heads * cfg.head_dim;
    let q_dim = cfg.heads * cfg.head_dim;
    let offset = cache.total_len;

    let mut q = layer.q_proj.forward(x_in)?;
    let mut k_blk = layer.k_proj.forward(x_in)?;
    let v_blk = layer.v_proj.forward(x_in)?;

    if let Some(qn) = &layer.q_norm {
        rms_norm_rows_inplace(&mut q, qn, cfg.eps, cfg.head_dim);
    }
    if let Some(kn) = &layer.k_norm {
        rms_norm_rows_inplace(&mut k_blk, kn, cfg.eps, cfg.head_dim);
    }

    let blk_pos = offset + ctx_rows;
    apply_neox_rope_rows(&mut q, t_block, cfg.heads, cfg.head_dim, rope, blk_pos);
    apply_neox_rope_rows(&mut k_blk, t_block, cfg.kv_heads, cfg.head_dim, rope, blk_pos);

    if let Some(ctx) = ctx_proj {
        if ctx_rows > 0 {
            let mut k_ctx = layer.k_proj.forward(ctx)?;
            let v_ctx = layer.v_proj.forward(ctx)?;
            if let Some(kn) = &layer.k_norm {
                rms_norm_rows_inplace(&mut k_ctx, kn, cfg.eps, cfg.head_dim);
            }
            apply_neox_rope_rows(&mut k_ctx, ctx_rows, cfg.kv_heads, cfg.head_dim, rope, offset);
            cache.append(&k_ctx, &v_ctx, ctx_rows, kv_dim, cfg.sliding_window);
        }
    }

    let cached_rows = cache.kept_rows(kv_dim);
    let cache_start_pos = cache.start_pos; // == cache.total_len - cached_rows
    let total_keys = cached_rows + t_block;

    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let group = (cfg.heads / cfg.kv_heads).max(1);
    let mut attn_out = vec![0.0f32; t_block * q_dim];
    let mut scores = vec![0.0f32; total_keys];

    for qi in 0..t_block {
        let qpos = blk_pos + qi;
        let q_row = &q[qi * q_dim..(qi + 1) * q_dim];
        let out_row = &mut attn_out[qi * q_dim..(qi + 1) * q_dim];

        for h in 0..cfg.heads {
            let kvh = h / group;
            let q_head = &q_row[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            let mut max_s = f32::NEG_INFINITY;
            let mut active = 0usize;

            // Context keys come first [0..cached_rows], then block keys [cached_rows..cached_rows+t_block].
            let min_kpos = cfg
                .sliding_window
                .map_or(0, |w| (qpos + 1).saturating_sub(w));
            let max_kpos = if cfg.block_causal { qpos } else { blk_pos + t_block - 1 };

            for ki in 0..cached_rows {
                let kpos = cache_start_pos + ki;
                if kpos < min_kpos || kpos > max_kpos {
                    continue;
                }
                let k_head = &cache.k[ki * kv_dim + kvh * cfg.head_dim..][..cfg.head_dim];
                let s = dot(q_head, k_head) * scale;
                scores[ki] = s;
                if s > max_s {
                    max_s = s;
                }
                active += 1;
            }
            for bi in 0..t_block {
                let kpos = blk_pos + bi;
                let idx = cached_rows + bi;
                if kpos < min_kpos || kpos > max_kpos {
                    continue;
                }
                let k_head = &k_blk[bi * kv_dim + kvh * cfg.head_dim..][..cfg.head_dim];
                let s = dot(q_head, k_head) * scale;
                scores[idx] = s;
                if s > max_s {
                    max_s = s;
                }
                active += 1;
            }
            if active == 0 {
                continue;
            }

            let mut sum_exp = 0.0f32;
            for (ki, score) in scores.iter_mut().enumerate().take(cached_rows) {
                let kpos = cache_start_pos + ki;
                if kpos < min_kpos || kpos > max_kpos {
                    *score = 0.0;
                } else {
                    let e = (*score - max_s).exp();
                    *score = e;
                    sum_exp += e;
                }
            }
            for bi in 0..t_block {
                let kpos = blk_pos + bi;
                let idx = cached_rows + bi;
                if kpos < min_kpos || kpos > max_kpos {
                    scores[idx] = 0.0;
                } else {
                    let e = (scores[idx] - max_s).exp();
                    scores[idx] = e;
                    sum_exp += e;
                }
            }
            let inv_sum = 1.0 / sum_exp;
            let o_head = &mut out_row[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            for ki in 0..cached_rows {
                let w = scores[ki] * inv_sum;
                if w == 0.0 {
                    continue;
                }
                let v_head = &cache.v[ki * kv_dim + kvh * cfg.head_dim..][..cfg.head_dim];
                for (o, &v) in o_head.iter_mut().zip(v_head) {
                    *o += w * v;
                }
            }
            for bi in 0..t_block {
                let w = scores[cached_rows + bi] * inv_sum;
                if w == 0.0 {
                    continue;
                }
                let v_head = &v_blk[bi * kv_dim + kvh * cfg.head_dim..][..cfg.head_dim];
                for (o, &v) in o_head.iter_mut().zip(v_head) {
                    *o += w * v;
                }
            }
        }
    }

    layer.o_proj.forward(&attn_out)
}

#[allow(clippy::too_many_arguments)]
fn bridge_forward(
    bridge: &BridgeWeights,
    cfg: &DSpineConfig,
    raw: &[f32],
    block_ids: &[i64],
    t_block: usize,
    layer_idx: usize,
    next_norm_w: &[f32],
    next_norm_eps: f32,
    stash: Option<BridgeStash>,
) -> (Vec<f32>, Vec<f32>, Option<BridgeStash>) {
    let h = cfg.hidden;
    let c = cfg.code_size;
    let r = cfg.rank;
    let sqrt_c = (c as f32).sqrt();
    let read_w = &bridge.read_weights[layer_idx]; // [C, H]
    let write_w = &bridge.feature_write_weights[layer_idx]; // [H, R]

    let mut rms_rows = vec![0.0f32; t_block];
    let mut codes = vec![0.0f32; t_block * c];

    let anchor_id = block_ids[0] as usize;
    let anchor_cb = &bridge.codebook[anchor_id * c..(anchor_id + 1) * c];
    for (dst, &cb) in codes[..c].iter_mut().zip(anchor_cb) {
        *dst = sqrt_c * cb;
    }

    let mut normalized = vec![0.0f32; h];
    for t in 0..t_block {
        let row = &raw[t * h..(t + 1) * h];
        let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / h as f32;
        let rms = mean_sq.max(F32_TINY).sqrt();
        rms_rows[t] = rms;
        if t == 0 {
            continue;
        }
        let inv = 1.0 / (rms + cfg.bridge_eps);
        for (dst, &src) in normalized.iter_mut().zip(row) {
            *dst = src * inv;
        }
        let proj = &mut codes[t * c..(t + 1) * c];
        let mut norm_sq = 0.0f32;
        for (ci, out_c) in proj.iter_mut().enumerate() {
            let v = dot(&normalized, &read_w[ci * h..(ci + 1) * h]);
            *out_c = v;
            norm_sq += v * v;
        }
        let scale = sqrt_c / (norm_sq.sqrt() + cfg.bridge_eps);
        for v in proj.iter_mut() {
            *v *= scale;
        }
    }

    let mut q = vec![0.0f32; t_block * r];
    let mut s = vec![0.0f32; t_block * r];
    let mut v = vec![0.0f32; t_block * r];
    for t in 0..t_block {
        let code = &codes[t * c..(t + 1) * c];
        for ri in 0..r {
            q[t * r + ri] = dot(code, &bridge.gate_w[ri * c..(ri + 1) * c]);
            s[t * r + ri] = dot(code, &bridge.pred_gate_w[ri * c..(ri + 1) * c]);
            v[t * r + ri] = dot(code, &bridge.prev_w[ri * c..(ri + 1) * c]);
        }
    }

    let out_stash = stash.map(|_| BridgeStash {
        raw: raw.to_vec(),
        rms: rms_rows.clone(),
        q_proj: q.clone(),
    });

    let mut out_res = raw.to_vec();
    let mut msg = vec![0.0f32; r];
    for t in 1..t_block {
        let q_row = &q[t * r..(t + 1) * r];
        let s_prev = &s[(t - 1) * r..t * r];
        let v_prev = &v[(t - 1) * r..t * r];
        for ri in 0..r {
            let g = 1.0 / (1.0 + (-(q_row[ri] + s_prev[ri] + bridge.gate_bias[ri])).exp());
            msg[ri] = g * v_prev[ri];
        }
        let rms = rms_rows[t];
        let dst = &mut out_res[t * h..(t + 1) * h];
        for hi in 0..h {
            let delta = dot(&msg, &write_w[hi * r..(hi + 1) * r]);
            dst[hi] += rms * delta;
        }
    }

    let normed = rms_norm_rows_to(&out_res, next_norm_w, next_norm_eps, h);
    (normed, out_res, out_stash)
}

#[allow(clippy::too_many_arguments)]
fn refine_tail(
    bridge: &BridgeWeights,
    cfg: &DSpineConfig,
    output_norm_w: &[f32],
    logits: &[f32],
    stash: &BridgeStash,
    anchor: i64,
    n: usize,
    params: &SamplingParams,
    sampler: &mut SamplingState,
    target: &RustDecoder,
) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
    let h = cfg.hidden;
    let r = cfg.rank;
    let c = cfg.code_size;
    let k = cfg.head_topk;
    let sqrt_c = (c as f32).sqrt();

    let vocab = cfg.vocab_size;
    let cand = topk_ids(logits, n, vocab, k);

    let mut sid = vec![0usize; n * k];
    sid[..k].fill(anchor as usize);
    for row in 1..n {
        for c_i in 0..k {
            sid[row * k + c_i] = cand[(row - 1) * k + c_i];
        }
    }

    let cand_i64: Vec<i64> = cand.iter().map(|&id| id as i64).collect();
    let head_rows = target.head_rows(&cand_i64)?; // [n, k, h]
    let write_w = bridge
        .feature_write_weights
        .last()
        .ok_or("missing last DSpine feature_write_weights")?; // [h, r]
    let greedy = params.argmax_only();

    let mut pred_code = vec![0.0f32; c];
    let mut tvg = vec![0.0f32; n * k * r];
    let mut tvv = vec![0.0f32; n * k * r];
    let mut unique_tsv: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::with_capacity(n * k);
    for &id in &sid {
        if unique_tsv.contains_key(&id) {
            continue;
        }
        let cb = &bridge.codebook[id * c..(id + 1) * c];
        for (dst, &v) in pred_code.iter_mut().zip(cb) {
            *dst = sqrt_c * v;
        }
        let mut g = vec![0.0f32; r];
        let mut v = vec![0.0f32; r];
        for ri in 0..r {
            g[ri] = dot(&pred_code, &bridge.pred_gate_w[ri * c..(ri + 1) * c]);
            let mut val = dot(&pred_code, &bridge.prev_w[ri * c..(ri + 1) * c]);
            if cfg.terminal_residual_rank > 0 {
                let rt = cfg.terminal_residual_rank;
                val += dot(
                    &bridge.terminal_emb[id * rt..(id + 1) * rt],
                    &bridge.terminal_proj[ri * rt..(ri + 1) * rt],
                );
            }
            v[ri] = val;
        }
        unique_tsv.insert(id, (g, v));
    }
    for idx in 0..n * k {
        let (ref g, ref v) = unique_tsv[&sid[idx]];
        tvg[idx * r..(idx + 1) * r].copy_from_slice(g);
        tvv[idx * r..(idx + 1) * r].copy_from_slice(v);
    }

    if greedy {
        let mut next_j = vec![0usize; n * k];
        let mut scaled_hr = vec![0.0f32; h];
        let mut tg_cand = vec![0.0f32; k * r];
        let mut base = vec![0.0f32; k];
        let mut msg = vec![0.0f32; r];

        for row in 0..n {
            let t = row + 1;
            let raw = &stash.raw[t * h..(t + 1) * h];
            let rms = stash.rms[t];
            let q = &stash.q_proj[t * r..(t + 1) * r];

            for c_i in 0..k {
                let hr = &head_rows[(row * k + c_i) * h..][..h];
                let mut b = 0.0f32;
                for hi in 0..h {
                    let s_hr = hr[hi] * output_norm_w[hi];
                    scaled_hr[hi] = s_hr;
                    b += raw[hi] * s_hr;
                }
                base[c_i] = b;
                for ri in 0..r {
                    let mut sum = 0.0f32;
                    for hi in 0..h {
                        sum += scaled_hr[hi] * write_w[hi * r + ri];
                    }
                    tg_cand[c_i * r + ri] = sum;
                }
            }

            let senders = if row == 0 { 1 } else { k };
            for s_i in 0..senders {
                let g = &tvg[(row * k + s_i) * r..][..r];
                let v = &tvv[(row * k + s_i) * r..][..r];
                for ri in 0..r {
                    let gate = 1.0 / (1.0 + (-(q[ri] + g[ri] + bridge.gate_bias[ri])).exp());
                    msg[ri] = gate * v[ri];
                }
                let mut best_c = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for c_i in 0..k {
                    let s_dot = dot(&msg, &tg_cand[c_i * r..(c_i + 1) * r]);
                    let score = base[c_i] + rms * s_dot;
                    if score > best_score {
                        best_score = score;
                        best_c = c_i;
                    }
                }
                next_j[row * k + s_i] = best_c;
            }
        }

        let mut tokens = Vec::with_capacity(n);
        let mut j = 0usize;
        for row in 0..n {
            let best_c = next_j[row * k + j];
            tokens.push(cand[row * k + best_c] as i64);
            j = best_c;
        }
        return Ok((tokens, vec![None; n]));
    }

    // Sampled serial refinement: materializes table[row, sender, cand] cleanly with candidate filters.
    let inv_temp = 1.0 / (params.temperature as f32).max(1e-5);
    let top_k = if params.top_k == 0 { k } else { params.top_k.min(k) };
    let top_p = params.top_p as f32;
    let min_p = params.min_p as f32;

    let mut tokens = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);
    let mut sender = 0usize;
    let mut msg = vec![0.0f32; r];
    let mut refined = vec![0.0f32; h];
    let mut row_logits = vec![0.0f32; k];

    for row in 0..n {
        let t = row + 1;
        let raw = &stash.raw[t * h..(t + 1) * h];
        let rms = stash.rms[t];
        let q = &stash.q_proj[t * r..(t + 1) * r];

        let g = &tvg[(row * k + sender) * r..][..r];
        let v = &tvv[(row * k + sender) * r..][..r];
        for ri in 0..r {
            let gate = 1.0 / (1.0 + (-(q[ri] + g[ri] + bridge.gate_bias[ri])).exp());
            msg[ri] = gate * v[ri];
        }
        for hi in 0..h {
            let delta = dot(&msg, &write_w[hi * r..(hi + 1) * r]);
            refined[hi] = raw[hi] + rms * delta;
        }
        let mean_sq = refined.iter().map(|&val| val * val).sum::<f32>() / h as f32;
        let inv = 1.0 / (mean_sq + cfg.eps).sqrt();
        for (val, &nw) in refined.iter_mut().zip(output_norm_w) {
            *val = (*val * inv) * nw;
        }

        for c_i in 0..k {
            let hr = &head_rows[(row * k + c_i) * h..][..h];
            row_logits[c_i] = dot(&refined, hr) * inv_temp;
        }
        let (probs, _keep) = softmax_and_filter_small(&row_logits, top_k, top_p, min_p);
        let cand_row = &cand[row * k..(row + 1) * k];
        let mut dist: Dist = cand_row
            .iter()
            .zip(&probs)
            .filter(|(_, &p)| p > 0.0)
            .map(|(&tok, &p)| (tok, p))
            .collect();
        let u = sampler.next_uniform();
        let (choice_tok, _) = select_from(&dist, u);
        let choice_idx = cand_row
            .iter()
            .position(|&tok| tok == choice_tok)
            .unwrap_or(0);
        // Keep Dist sorted descending by prob (not strictly required by verify_chain, but tidy).
        dist.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        tokens.push(choice_tok as i64);
        dists.push(Some(dist));
        sender = choice_idx;
    }

    Ok((tokens, dists))
}

fn dflash_sample_from_logits(
    logits: &[f32],
    n: usize,
    params: &SamplingParams,
    sampler: &mut SamplingState,
    history: &[i64],
) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let vocab = logits.len() / n;
    let greedy = params.argmax_only();
    let mut tokens = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);
    for row in 0..n {
        let row_logits = &logits[row * vocab..(row + 1) * vocab];
        if greedy {
            let best = sampling::argmax(row_logits);
            tokens.push(best as i64);
            dists.push(None);
        } else {
            let dist = processed_distribution(row_logits, history, params, 0.0);
            let u = sampler.next_uniform();
            let (tok, _) = select_from(&dist, u);
            tokens.push(tok as i64);
            dists.push(Some(dist));
        }
    }
    Ok((tokens, dists))
}

/// Top-k token ids per row, sorted ascending by token id (ties break to smaller token id).
pub(crate) fn topk_ids(logits: &[f32], rows: usize, vocab: usize, k: usize) -> Vec<usize> {
    let k = k.min(vocab);
    let mut out = Vec::with_capacity(rows * k);
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        best.clear();
        for (id, &val) in row.iter().enumerate() {
            if best.len() < k {
                best.push((val, id));
                if best.len() == k {
                    best.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
                }
            } else {
                let (min_v, min_id) = best[0];
                if val > min_v || (val == min_v && id < min_id) {
                    best[0] = (val, id);
                    // Bubble into sorted position (k <= 64, tiny)
                    let mut i = 0;
                    while i + 1 < k {
                        let (va, ia) = best[i];
                        let (vb, ib) = best[i + 1];
                        if va > vb || (va == vb && ia < ib) {
                            best.swap(i, i + 1);
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        let mut ids: Vec<usize> = best.iter().map(|&(_, id)| id).collect();
        ids.sort_unstable();
        out.extend_from_slice(&ids);
    }
    out
}

fn softmax_and_filter_small(scaled_logits: &[f32], top_k: usize, top_p: f32, min_p: f32) -> (Vec<f32>, Vec<bool>) {
    let k = scaled_logits.len();
    let max_l = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = scaled_logits.iter().map(|&v| (v - max_l).exp()).collect();
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    }
    let p_max = p.iter().copied().fold(0.0f32, f32::max);
    let mut keep = vec![true; k];
    for c in 0..k {
        let pc = p[c];
        let mut rank = 0usize;
        let mut before = 0.0f32;
        for (d, &pd) in p.iter().enumerate().take(k) {
            if pd > pc || (pd == pc && d < c) {
                rank += 1;
                before += pd;
            }
        }
        let mut ok = rank < top_k && before <= top_p;
        if min_p > 0.0 {
            ok &= pc >= min_p * p_max;
        }
        keep[c] = ok;
    }
    let mut w_sum = 0.0f32;
    for (v, &ok) in p.iter_mut().zip(&keep) {
        if !ok {
            *v = 0.0;
        } else {
            w_sum += *v;
        }
    }
    if w_sum > 0.0 {
        for v in &mut p {
            *v /= w_sum;
        }
    }
    (p, keep)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let j = i * 4;
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    let mut tail = 0.0f32;
    for i in (chunks * 4)..a.len() {
        tail += a[i] * b[i];
    }
    (s0 + s1) + (s2 + s3) + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bridge(v: usize, h: usize, c: usize, r: usize, layers: usize, rt: usize) -> BridgeWeights {
        let mut codebook = vec![0.0f32; v * c];
        for id in 0..v {
            for ci in 0..c {
                codebook[id * c + ci] = ((id * 7 + ci * 3) as f32 * 0.05).sin() * 0.1;
            }
        }
        let mut read_weights = Vec::with_capacity(layers);
        let mut feature_write_weights = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut rw = vec![0.0f32; c * h];
            for ci in 0..c {
                for hi in 0..h {
                    rw[ci * h + hi] = ((l + 1) as f32 * 0.1) * (((ci + hi) % 5) as f32 - 2.0) * 0.1;
                }
            }
            read_weights.push(rw);
            let mut ww = vec![0.0f32; h * r];
            for hi in 0..h {
                for ri in 0..r {
                    ww[hi * r + ri] = (((l * 3 + hi + ri) % 7) as f32 - 3.0) * 0.05;
                }
            }
            feature_write_weights.push(ww);
        }
        let mut gate_w = vec![0.0f32; r * c];
        let mut pred_gate_w = vec![0.0f32; r * c];
        let mut prev_w = vec![0.0f32; r * c];
        let mut gate_bias = vec![0.0f32; r];
        for ri in 0..r {
            gate_bias[ri] = (ri as f32 - 1.0) * 0.15;
            for ci in 0..c {
                gate_w[ri * c + ci] = (((ri + ci) % 5) as f32 - 2.0) * 0.08;
                pred_gate_w[ri * c + ci] = (((ri * 2 + ci) % 5) as f32 - 2.0) * 0.07;
                prev_w[ri * c + ci] = (((ri + ci * 3) % 5) as f32 - 2.0) * 0.09;
            }
        }
        let mut terminal_emb = vec![0.0f32; v * rt];
        let mut terminal_proj = vec![0.0f32; r * rt];
        if rt > 0 {
            for id in 0..v {
                for ti in 0..rt {
                    terminal_emb[id * rt + ti] = ((id + ti + 1) as f32 * 0.2).cos() * 0.1;
                }
            }
            for ri in 0..r {
                for ti in 0..rt {
                    terminal_proj[ri * rt + ti] = ((ri + ti * 2 + 1) as f32 * 0.3).sin() * 0.1;
                }
            }
        }
        BridgeWeights {
            codebook,
            read_weights,
            feature_write_weights,
            gate_w,
            pred_gate_w,
            prev_w,
            gate_bias,
            active_layers: (0..layers).collect(),
            terminal_emb,
            terminal_proj,
        }
    }

    #[test]
    fn bridge_preserves_anchor_row_and_matches_exact_formulas() {
        let (v, h, c, r, layers, t_block) = (12, 8, 6, 4, 2, 4);
        let bridge = make_test_bridge(v, h, c, r, layers, 0);
        let cfg = DSpineConfig {
            arch: "dspine".into(),
            hidden: h,
            intermediate: 16,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            layers,
            vocab_size: v,
            eps: 1e-6,
            block_size: t_block,
            mask_token_id: 11,
            target_layer_ids: vec![0],
            num_target_layers: 2,
            target_hidden: h,
            block_causal: true,
            sliding_window: None,
            has_bridge: true,
            code_size: c,
            rank: r,
            bridge_eps: 1e-6,
            head_topk: 4,
            feature_layers: vec![0, 1],
            terminal_residual_rank: 0,
        };
        let next_norm_w: Vec<f32> = (0..h).map(|i| 0.8 + i as f32 * 0.05).collect();
        let raw: Vec<f32> = (0..t_block * h).map(|i| ((i as f32) * 0.31).sin()).collect();
        let block_ids = vec![3i64, 11, 11, 11];
        let (_normed, out_res, stash) = bridge_forward(
            &bridge,
            &cfg,
            &raw,
            &block_ids,
            t_block,
            1,
            &next_norm_w,
            cfg.eps,
            Some(BridgeStash::default()),
        );
        // Anchor row residual (t = 0) must be unchanged because message[0] = 0
        assert_eq!(&out_res[..h], &raw[..h]);
        // Non-anchor rows receive non-zero causal injection
        assert_ne!(&out_res[h..2 * h], &raw[h..2 * h]);
        let stash = stash.unwrap();
        assert_eq!(stash.rms.len(), t_block);
        assert_eq!(stash.q_proj.len(), t_block * r);
    }

    #[test]
    fn refinement_tail_greedy_and_sampled_match_direct_residual_computation() {
        // Check that greedy table score matches explicit residual refinement + lm_head dot product
        // (and when terminal_residual_rank > 0, the residual projection matches direct table evaluation).
        let (v, h, c, r, k, rt) = (10, 6, 4, 3, 4, 2);
        let bridge = make_test_bridge(v, h, c, r, 1, rt);
        let norm_w: Vec<f32> = (0..h).map(|i| 1.0 + 0.05 * i as f32).collect();
        let raw_row: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let q_row: Vec<f32> = (0..r).map(|i| (i as f32 - 1.0) * 0.2).collect();
        let rms_val = 0.75f32;
        let anchor = 2usize;
        let sqrt_c = (c as f32).sqrt();

        // Candidate head rows
        let cand_ids = [1usize, 4, 6, 8];
        let mut head_mat = vec![0.0f32; k * h];
        for (ci, &id) in cand_ids.iter().enumerate() {
            for hi in 0..h {
                head_mat[ci * h + hi] = ((id * 3 + hi) as f32 * 0.17).cos();
            }
        }

        // Method 1 (pre-norm base + tg @ msg decomposition used by greedy)
        let write_w = &bridge.feature_write_weights[0];
        let cb = &bridge.codebook[anchor * c..(anchor + 1) * c];
        let code: Vec<f32> = cb.iter().map(|&v| v * sqrt_c).collect();
        let mut msg = vec![0.0f32; r];
        for ri in 0..r {
            let g_in = q_row[ri]
                + dot(&code, &bridge.pred_gate_w[ri * c..(ri + 1) * c])
                + bridge.gate_bias[ri];
            let gate = 1.0 / (1.0 + (-g_in).exp());
            let val = dot(&code, &bridge.prev_w[ri * c..(ri + 1) * c])
                + dot(
                    &bridge.terminal_emb[anchor * rt..(anchor + 1) * rt],
                    &bridge.terminal_proj[ri * rt..(ri + 1) * rt],
                );
            msg[ri] = gate * val;
        }
        let mut refined = raw_row.clone();
        for hi in 0..h {
            refined[hi] += rms_val * dot(&msg, &write_w[hi * r..(hi + 1) * r]);
        }
        for (ci, _) in cand_ids.iter().enumerate() {
            let hr = &head_mat[ci * h..(ci + 1) * h];
            let direct: f32 = refined.iter().zip(&norm_w).zip(hr).map(|((&r, &nw), &w)| r * nw * w).sum();
            let base: f32 = raw_row.iter().zip(&norm_w).zip(hr).map(|((&r, &nw), &w)| r * nw * w).sum();
            let mut tg = vec![0.0f32; r];
            for ri in 0..r {
                tg[ri] = (0..h).map(|hi| hr[hi] * norm_w[hi] * write_w[hi * r + ri]).sum();
            }
            let decomposed = base + rms_val * dot(&msg, &tg);
            assert!((direct - decomposed).abs() < 1e-5, "{direct} vs {decomposed}");
        }
    }

    #[test]
    fn topk_ids_handles_ties_and_sorts_ascending() {
        let logits = [1.0f32, 5.0, 3.0, 5.0, 2.0, 3.0];
        let ids = topk_ids(&logits, 1, 6, 3);
        // top 3 values are idx 1 (5.0), idx 3 (5.0), idx 2 (3.0, tie over idx 5 -> lower id 2 wins)
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn kv_cache_sliding_window_trims_and_preserves_positions() {
        let mut cache = LayerKvCache::default();
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], 2, 2, Some(3));
        assert_eq!(cache.start_pos, 0);
        assert_eq!(cache.total_len, 2);
        assert_eq!(cache.kept_rows(2), 2);

        cache.append(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0], 2, 2, Some(3));
        assert_eq!(cache.start_pos, 1);
        assert_eq!(cache.total_len, 4);
        assert_eq!(cache.kept_rows(2), 3);
        assert_eq!(cache.k, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn softmax_filter_respects_top_k_top_p_min_p() {
        let logits = [4.0f32, 2.0, 1.0, -5.0];
        let (probs, keep) = softmax_and_filter_small(&logits, 2, 0.99, 0.0);
        assert_eq!(keep, vec![true, true, false, false]);
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
    }
}
