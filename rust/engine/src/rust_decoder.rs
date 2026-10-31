//! Decoder forward pass for every architecture a `ModelConfig` can describe, using Rust CPU / WGSL
//! projections. The scheduler and paged KV allocations are shared with the runtime.
use crate::{
    compute::{self, Linear, Tensor},
    config::{Activation, Gating, ModelConfig, NormKind, QkNorm},
    rope::RopeTables,
    GgufModel,
};
#[cfg(target_endian = "big")]
use half::f16;
use fllm_gguf_probe::MappedGguf;
use rayon::prelude::*;
use serde_json::Value;
use std::sync::Arc;

mod mla_latent;
mod moe_device;
mod gemma;
use gemma::{AltupGlobal, GemmaLayer, PleGlobal};
// Tensor / pipeline / expert parallelism: sharding of the loaded layers and the collectives (docs/multi-gpu.md).
#[path = "parallel/decoder_shard.rs"]
mod par_shard;
mod hybrid;
mod kda;
mod rwkv;
mod deepseek4;
mod hc;
mod inkling;
mod inkling_device;
mod minimax3;
mod qwen4;
mod state_api;
mod zamba;
pub use state_api::{LayerStateSnapshot, RecurrentState, SequenceSnapshot};

/// Learned scale (and, for LayerNorm, shift) of one normalisation.
struct Norm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

struct MtpProjection {
    embedding_norm: Vec<f32>,
    hidden_norm: Vec<f32>,
    projection: Linear,
}

enum Ffn {
    /// `down(act(gate(x)) * up(x))`
    Gated {
        gate: Linear,
        up: Linear,
        down: Linear,
    },
    /// `down(act(up(x)))`
    Plain { up: Linear, down: Linear },
    /// Mixture of experts: a router picks `used` gated FFNs per token.
    Moe(Box<Moe>),
    /// Layers that are a single mixer (state-space models): nothing after the residual add.
    None,
}

struct Expert {
    /// `None` for plain up/down experts (Nemotron-H).
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
}

/// A host-resident projection ready to be streamed to the device: format, raw bytes, rows, cols.
type HostWeight<'a> = (crate::backend::QuantType, &'a [u8], usize, usize, Option<&'a [f32]>);

/// What `streamable` returns: format, raw bytes and bias of a host-resident projection.
type StreamableWeight<'a> = (crate::backend::QuantType, &'a [u8], Option<&'a [f32]>);

/// The gated activation the device implements.
#[derive(Clone, Copy)]
enum GluMode {
    Swiglu,
    Geglu,
    /// gpt-oss: alpha and clamp limit.
    SwigluOai(f32, f32),
    /// Hy4 / GLM-5-Next clamped SwiGLU, Kimi K3 SiTU.
    Clamped(crate::backend::ClampedGlu),
}

/// Where the three projections of a gated FFN / expert come from when it is computed on the device.
enum ExpertSrc<'a> {
    Device(&'a DeviceExpert),
    Host([HostWeight<'a>; 3]),
}

/// K2 Horizon MoVA: each token's values are `sum_k w_k * silu(W_k x)` over routed value experts.
struct ValueMoe {
    router: Linear,
    /// Added to the router scores for the selection only.
    bias: Option<Vec<f32>>,
    experts: Vec<Linear>,
    used: usize,
}

struct Moe {
    router: Linear,
    /// Added to the router scores for the expert selection only (DeepSeek-V3).
    bias: Option<Vec<f32>>,
    experts: Vec<Expert>,
    /// Read-ahead, residency and statistics of the experts in the model file (`expert_cache.rs`); `None` when they are not file data.
    cache: Option<Arc<crate::expert_cache::LayerTable>>,
    /// Always-active expert, optionally scaled by `sigmoid(shared_gate . x)`.
    shared: Option<(Expert, Option<Vec<f32>>)>,
    /// Latent MoE (Nemotron-H): `(down, up)` projections around the routed experts.
    latent: Option<(Linear, Linear)>,
    /// Weights of the RMS norm over the routed output in the latent space (Kimi K3).
    latent_norm: Option<Vec<f32>>,
    /// Hash routing (DeepSeek-V4's first layers): the experts of token id `t` are `table[t * used .. (t + 1) * used]`; the router scores only
    /// weight them.
    tid2eid: Option<Vec<u32>>,
}

/// Recurrent state of one Gated DeltaNet layer of one sequence.
struct LayerState {
    /// Tokens already folded into the state.
    len: usize,
    /// Last `kernel - 1` inputs of every convolution channel.
    conv: Vec<f32>,
    /// `v_heads x k_dim x v_dim`
    rec: Vec<f32>,
}

struct SeqState {
    used: u64,
    layers: Vec<Option<LayerState>>,
}

#[derive(Default)]
struct StateStore {
    map: std::collections::HashMap<i64, SeqState>,
    clock: u64,
}

/// Device copy of a Gated DeltaNet layer (projections as quantized device weights, small tensors as FP32 buffers).
struct DeviceDelta {
    qkv: crate::backend::DeviceWeight,
    gate: crate::backend::DeviceWeight,
    beta: crate::backend::DeviceWeight,
    alpha: crate::backend::DeviceWeight,
    conv: crate::backend::DeviceBuffer,
    dt_bias: crate::backend::DeviceBuffer,
    decay: crate::backend::DeviceBuffer,
    norm: crate::backend::DeviceBuffer,
    /// Index among the DeltaNet layers (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

/// Device copy of a multi-head latent attention layer (the projections; the cache holds rebuilt per-head keys and values).
struct DeviceMla {
    q_a: Option<(crate::backend::DeviceWeight, crate::backend::DeviceBuffer)>,
    q: crate::backend::DeviceWeight,
    kv_a: crate::backend::DeviceWeight,
    kv_a_norm: crate::backend::DeviceBuffer,
    /// Expanded cache: the up projection; absent with the latent cache.
    kv_b: Option<crate::backend::DeviceWeight>,
    /// Latent cache (`FLLM_MLA=latent`): the dense FP16 kv_b matrix, `heads * (nope + value)` rows of `kv_lora` values.
    kv_b_dense: Option<crate::backend::DeviceBuffer>,
}

/// Device copy of the weights of one hyper-connection block: the mixing projection of the normalised streams and the learned offsets and scales.
struct DeviceHc {
    mix: crate::backend::DeviceWeight,
    base: crate::backend::DeviceBuffer,
    scale: crate::backend::DeviceBuffer,
}

/// Hyper-connection weights of the model on the device (Hy4, GLM-5-Next).
struct DeviceHcModel {
    attn: Vec<DeviceHc>,
    ffn: Vec<DeviceHc>,
    head: Option<DeviceHc>,
    /// `hc * hidden` ones: the weight of the plain RMS norm over the flattened streams.
    ones: crate::backend::DeviceBuffer,
}

/// The parameters of the hyper-connections of a model.
struct HcParams {
    hc_mult: usize,
    eps: f32,
    iters: usize,
    /// `Some`: independent connections (Hy4).
    magnitude: Option<f32>,
}

/// What a hyper-connection block keeps between reading the streams and writing the branch output back.
struct HcBlock {
    post: crate::backend::DeviceBuffer,
    comb: Option<crate::backend::DeviceBuffer>,
    /// The collapsed block input (the branch adds its output to the residual buffer that started as this).
    base: crate::backend::DeviceBuffer,
}

/// Device copy of the lightning indexer of a sparse-attention layer (DeepSeek-V3.2, GLM-5): its projections, the key norm and the per-slot f32 key store.
struct DeviceDsa {
    q_b: crate::backend::DeviceWeight,
    k: crate::backend::DeviceWeight,
    proj: crate::backend::DeviceWeight,
    k_norm: crate::backend::DeviceBuffer,
    k_norm_bias: crate::backend::DeviceBuffer,
    /// K-pool indexer: the projection of the gate scores of the pooling and the position bias (`kpool x dim`).
    pool: Option<(crate::backend::DeviceWeight, crate::backend::DeviceBuffer)>,
    /// One `store_width`-wide f32 row per cache slot (key, and with pools the gate score).
    store: crate::backend::DeviceBuffer,
}

/// Device copy of the block indexer of a MiniMax-M3 block-sparse attention layer: its two projections, the per-head norms and the per-slot f32 key store.
struct DeviceM3 {
    q: crate::backend::DeviceWeight,
    k: crate::backend::DeviceWeight,
    qn: crate::backend::DeviceBuffer,
    kn: crate::backend::DeviceBuffer,
    /// One `index_head_dim` f32 row per cache slot.
    store: crate::backend::DeviceBuffer,
}

/// Device copy of a Mamba-2 layer (the input projection as a quantized weight, small tensors as FP32 buffers).
struct DeviceMamba {
    input: crate::backend::DeviceWeight,
    conv: crate::backend::DeviceBuffer,
    conv_bias: Option<crate::backend::DeviceBuffer>,
    dt_bias: crate::backend::DeviceBuffer,
    decay: crate::backend::DeviceBuffer,
    skip: crate::backend::DeviceBuffer,
    norm: crate::backend::DeviceBuffer,
    /// Index among the Mamba-2 layers (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

/// Device copy of an LFM2 short-convolution layer.
struct DeviceShortConv {
    input: crate::backend::DeviceWeight,
    conv: crate::backend::DeviceBuffer,
    bias: Option<crate::backend::DeviceBuffer>,
    /// Index among the ShortConv layers (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

/// Device copy of a Mamba-1 layer.
struct DeviceMamba1 {
    input: crate::backend::DeviceWeight,
    conv: crate::backend::DeviceBuffer,
    conv_bias: Option<crate::backend::DeviceBuffer>,
    x_proj: crate::backend::DeviceWeight,
    dt_proj: crate::backend::DeviceWeight,
    decay: crate::backend::DeviceBuffer,
    skip: crate::backend::DeviceBuffer,
    /// Learned weights of the RMS norms behind x_proj for the time step, B and C (Jamba).
    norms: Option<[crate::backend::DeviceBuffer; 3]>,
    /// Index among the Mamba-1 layers (selects the slice of a sequence's state buffer).
    ordinal: usize,
}

/// Recurrent state of one live sequence on the device: one buffer for all DeltaNet layers.
struct DeviceSeq {
    used: u64,
    buf: crate::backend::DeviceBuffer,
    /// Tokens already folded into the state, per DeltaNet layer.
    lens: Vec<usize>,
}

#[derive(Default)]
struct DeviceStates {
    map: std::collections::HashMap<i64, DeviceSeq>,
    clock: u64,
    /// Buffers of evicted sequences, reused for new ones.
    free: Vec<crate::backend::DeviceBuffer>,
}

/// One run of consecutive tokens of a sequence in the current batch.
struct DeltaPlanRun {
    first: i64,
    buf: crate::backend::DeviceBuffer,
    start: usize,
    n: usize,
    position: usize,
}

struct DeltaNet {
    qkv: Linear,
    gate: Linear,
    beta: Linear,
    alpha: Linear,
    /// `conv_dim x kernel`
    conv: Vec<f32>,
    dt_bias: Vec<f32>,
    /// Negative decay rates (`-exp(A_log)`).
    decay: Vec<f32>,
    norm: Vec<f32>,
}

/// Weights of a Mamba-2 mixer.
struct Mamba {
    input: Linear,
    conv: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    dt_bias: Vec<f32>,
    /// Negative decay rates (`-exp(A_log)`) per head.
    decay: Vec<f32>,
    skip: Vec<f32>,
    norm: Vec<f32>,
}

/// Weights of a Mamba-1 mixer.
struct Mamba1 {
    input: Linear,
    conv: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    x_proj: Linear,
    dt_proj: Linear,
    /// `inner x state`, already negated (`-exp(A_log)`).
    decay: Vec<f32>,
    skip: Vec<f32>,
    /// Learned weights of the RMS norms behind `x_proj` for the time step, B and C (Jamba).
    norms: Option<Box<[Vec<f32>; 3]>>,
}

struct ShortConv {
    input: Linear,
    conv: Vec<f32>,
    bias: Option<Vec<f32>>,
}

#[allow(clippy::large_enum_variant)]
enum Attn {
    ShortConv(Box<ShortConv>),
    Mamba1(Box<Mamba1>),
    /// Feed-forward-only layer (Nemotron-H keeps one mixer per layer).
    None,
    Mamba(Box<Mamba>),
    Linear(Box<DeltaNet>),
    /// Kimi Delta Attention (gated delta rule with a decay per key channel).
    Kda(Box<kda::Kda>),
    /// RWKV-6 / RWKV-7: the whole layer (time-mix and channel-mix) is computed by `rwkv_layer`.
    Rwkv(Box<rwkv::Rwkv>),
    Std {
        q: Linear,
        k: Linear,
        v: Linear,
    },
    Mla(Box<Mla>),
}

/// Weights of multi-head latent attention.
struct Mla {
    /// Low-rank query down projection and its norm.
    q_a: Option<(Linear, Vec<f32>)>,
    /// `q_b` with a low-rank query, otherwise the full query projection.
    q: Linear,
    kv_a: Linear,
    kv_a_norm: Vec<f32>,
    kv_b: Linear,
}

/// Score weights of the two attention-residual mixtures of a layer.
struct LayerRes {
    attn: Vec<f32>,
    ffn: Vec<f32>,
}

/// Softmax mixture of the earlier block sums `blocks` (each `rows x hidden`) and the running sum `prefix`: every candidate is
/// normalised by the root mean square over its hidden values and scored with `score` (the norm weight times the projection).
fn apply_attn_res(prefix: &[f32], blocks: &[Vec<f32>], score: &[f32], eps: f32, hidden: usize) -> Vec<f32> {
    let rows = prefix.len() / hidden;
    let mut out = vec![0.0f32; prefix.len()];
    for r in 0..rows {
        let candidates: Vec<&[f32]> = blocks.iter().map(|b| &b[r * hidden..(r + 1) * hidden]).chain(std::iter::once(&prefix[r * hidden..(r + 1) * hidden])).collect();
        let scores: Vec<f32> = candidates
            .iter()
            .map(|v| {
                let inv = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32 + eps).sqrt().recip();
                v.iter().zip(score).map(|(x, s)| x * inv * s).sum()
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let row = &mut out[r * hidden..(r + 1) * hidden];
        for (c, e) in candidates.iter().zip(&exps) {
            row.iter_mut().zip(*c).for_each(|(o, x)| *o += e / sum * x);
        }
    }
    out
}

/// `weights` times `scale` (a scalar that follows a normalisation is folded into its weights).
fn scaled(mut weights: Vec<f32>, scale: Option<f32>) -> Vec<f32> {
    if let Some(s) = scale {
        weights.iter_mut().for_each(|w| *w *= s);
    }
    weights
}

/// Queries, keys and values of an MLA layer.
type MlaQkv = [Vec<f32>; 3];

/// Lightning indexer of a sparse-attention layer (DeepSeek-V3.2, GLM-5): scores every earlier token and keeps the best `topk`.
struct Indexer {
    /// Queries from the normalized low-rank query of the layer: `heads * dim` rows.
    q_b: Linear,
    /// Key of the token from the layer input: `dim` rows.
    k: Linear,
    k_norm: Vec<f32>,
    k_norm_bias: Vec<f32>,
    /// Per-head score weights from the layer input: `heads` rows.
    proj: Linear,
    /// K-pool indexer (GLM-5-Next): gate scores of the pooling from the layer input (`dim` rows) and the position bias `kpool x dim`.
    pool: Option<(Linear, Vec<f32>)>,
}

/// Indexer keys of the sparse-attention layers, one `dim`-wide row per cache slot (the slot numbering of the KV pools, so a block
/// copy or swap moves them with the block), and the selection of the layer before for the layers that share it.
#[derive(Default)]
struct DsaStore {
    keys: std::sync::Mutex<Vec<Vec<f32>>>,
    selection: std::sync::Mutex<Vec<Option<Vec<u32>>>>,
}

struct Layer {
    attn: Attn,
    /// Sparse-attention indexer of an MLA layer.
    indexer: Option<Box<Indexer>>,
    /// LongCat-Flash: a mixture of experts that reads the input of this layer's feed-forward and joins the residual stream after the
    /// next layer (`takes_shortcut`).
    shortcut: Option<Box<Moe>>,
    takes_shortcut: bool,
    /// Block attention residual weights of this layer: `norm.weight * proj.weight` for the attention and the feed-forward input.
    res: Option<Box<LayerRes>>,
    /// The layer attends to the tokens the last layer with an indexer selected (GLM-5 layers marked `shared`).
    dsa_shared: bool,
    /// A second mixer that runs next to the attention of this layer (Falcon-H1: Mamba-2 and attention in parallel).
    parallel: Option<Box<hybrid::ParallelMixer>>,
    /// Zamba2: this layer starts with a call of a shared transformer block whose output joins the input of its Mamba mixer.
    zamba: Option<Box<zamba::ZambaCall>>,
    /// The query projection also yields a sigmoid gate for the attention output.
    output_gate: bool,
    /// Attention is limited to aligned chunks of this many positions (Llama 4).
    chunk: Option<usize>,
    /// Separate projection of the input that gates the attention output (AFMoE).
    attn_gate: Option<Linear>,
    /// `attn_gate` yields one value per head (Laguna) instead of one per output channel.
    gate_per_head: bool,
    value_moe: Option<ValueMoe>,
    /// xIELU parameters `[alpha_p, alpha_n, beta, eps]` of a plain feed-forward (Apertus).
    xielu: Option<[f32; 4]>,
    /// Learned per-head logit that joins the softmax denominator (GPT-OSS).
    sinks: Option<Vec<f32>>,
    /// RMS norm over the attention output in front of the output projection (BitNet).
    attn_sub_norm: Option<Vec<f32>>,
    /// RMS norm over `act(gate) * up` in front of the down projection (BitNet).
    ffn_sub_norm: Option<Vec<f32>>,
    out: Linear,
    ffn: Ffn,
    attn_norm: Option<Norm>,
    ffn_norm: Option<Norm>,
    post_attn_norm: Option<Norm>,
    post_ffn_norm: Option<Norm>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    /// LayerNorm biases of the per-head query / key norms (Persimmon).
    q_norm_bias: Option<Vec<f32>>,
    k_norm_bias: Option<Vec<f32>>,
    window: Option<usize>,
    /// Index of the rotary table of this layer (see `GgufModel::rope_index`).
    rope_table: usize,
    /// The layer reads the cache of an earlier layer (`ModelConfig::layer_kv_source`) and has no key / value projections.
    kv_shared: bool,
    /// The value projection is the key projection (Gemma 4 global layers).
    k_eq_v: bool,
    /// Gemma 4 / Gemma 3n additions of this layer.
    gemma: Option<Box<GemmaLayer>>,
}

impl Layer {
    #[allow(dead_code)]
    fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut all = Vec::new();
        let has_attention = !matches!(self.attn, Attn::None);
        match &mut self.attn {
            Attn::Std { q, k, v } => all.extend([q, k, v]),
            Attn::Linear(d) => all.extend([&mut d.qkv, &mut d.gate, &mut d.beta, &mut d.alpha]),
            Attn::Kda(k) => all.extend(k.linears_mut()),
            Attn::Rwkv(r) => all.extend(r.linears_mut()),
            Attn::ShortConv(m) => all.push(&mut m.input),
            Attn::Mamba(m) => all.push(&mut m.input),
            Attn::Mamba1(m) => all.extend([&mut m.input, &mut m.x_proj, &mut m.dt_proj]),
            Attn::None => {}
            Attn::Mla(m) => {
                if let Some((down, _)) = &mut m.q_a {
                    all.push(down);
                }
                all.extend([&mut m.q, &mut m.kv_a, &mut m.kv_b]);
            }
        }
        if let Some(ix) = &mut self.indexer {
            all.extend([&mut ix.q_b, &mut ix.k, &mut ix.proj]);
            all.extend(ix.pool.iter_mut().map(|p| &mut p.0));
        }
        if has_attention {
            all.push(&mut self.out);
        }
        if let Some(p) = &mut self.parallel {
            all.extend([&mut p.mamba.input, &mut p.out]);
        }
        match &mut self.ffn {
            Ffn::None => {}
            Ffn::Gated { gate, up, down } => all.extend([gate, up, down]),
            Ffn::Plain { up, down } => all.extend([up, down]),
            Ffn::Moe(moe) => {
                all.push(&mut moe.router);
                for e in moe
                    .experts
                    .iter_mut()
                    .chain(moe.shared.iter_mut().map(|s| &mut s.0))
                {
                    all.extend(e.gate.as_mut());
                    all.extend([&mut e.up, &mut e.down]);
                }
            }
        }
        all
    }
}

/// MLX affine weights for a backend that has no kernel for them: dequantized on the host and uploaded as f16 (the values the converter
/// used to write), one tensor at a time. The device then holds 16 bits per weight; the CPU backend and CUDA read the packed form.
fn upload_mlx_expanded(
    backend: &dyn crate::backend::Backend,
    name: &str,
    kind: u32,
    bytes: &[u8],
    linear: &Linear,
) -> Result<crate::backend::DeviceWeight, String> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[fllm-rs] {} has no kernel for MLX affine quantized weights yet; they are expanded to f16 when uploaded (first: {name})",
            backend.name()
        );
    }
    let values = compute::dequantize(bytes, kind, linear.rows * linear.cols)?;
    let halves: Vec<u8> = values.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect();
    backend
        .upload_weight(name, &halves, crate::backend::QuantType::F16, linear.rows, linear.cols, linear.bias.as_deref())
        .map_err(|e| e.to_string())
}

struct Embedding {
    data: fllm_gguf_probe::TensorBytes,
    kind: u32,
    row_bytes: usize,
    rows: usize,
    cols: usize,
    /// Vocabulary-parallel embedding: this rank stores the rows `first_row..first_row + held` (all rows otherwise).
    first_row: usize,
    held: usize,
}
impl Embedding {
    fn load_named(model: &GgufModel, name: &str) -> Result<Self, String> {
        Self::load_with_cols(model, name, model.hidden_size())
    }
    /// Embedding table whose rows are `cols` wide (the per-layer embeddings of Gemma are wider than the hidden size).
    fn load_with_cols(model: &GgufModel, name: &str, cols: usize) -> Result<Self, String> {
        let map = model.mapping();
        let info = map
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing token embeddings {name}"))?;
        let rows = *info
            .dimensions
            .get(1)
            .ok_or("invalid embedding dimensions")? as usize;
        if info.dimensions.len() != 2 || info.dimensions[0] != cols as u64 {
            return Err("invalid embedding dimensions".into());
        }
        let data = map.owned_tensor_bytes(name)?;
        if rows == 0 || !data.len().is_multiple_of(rows) {
            return Err("invalid embedding byte count".into());
        }
        let row_bytes = data.len() / rows;
        // Validate one full row with the same reader used for lookup.
        compute::dequantize(&data[..row_bytes], info.type_id, cols)?;
        Ok(Self {
            data,
            kind: info.type_id,
            row_bytes,
            rows,
            cols,
            first_row: 0,
            held: rows,
        })
    }
    fn row(&self, id: usize) -> Result<Vec<f32>, String> {
        if id < self.first_row || id >= self.first_row + self.held {
            // a row another tensor-parallel rank holds: contributes zeros to the sum over the group
            return Ok(vec![0.0; self.cols]);
        }
        let id = id - self.first_row;
        let values = compute::dequantize(
            &self.data[id * self.row_bytes..(id + 1) * self.row_bytes],
            self.kind,
            self.cols,
        )?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("non-finite token embeddings".into());
        }
        Ok(values)
    }
}

/// From this layer on, the feed-forward weights stay in host memory and the branch runs on the CPU (set by the
/// memory planner when the model does not fit the device). `(layers, first)`: only applied to a decoder with
/// exactly that many layers, so draft models are unaffected.
static FFN_HOST_FROM: std::sync::Mutex<Option<(usize, usize)>> = std::sync::Mutex::new(None);

/// Forget the host-offload plan (tensor / pipeline parallelism keeps every rank's share on its device). `true` when there was one.
pub fn clear_ffn_host_from() -> bool {
    FFN_HOST_FROM.lock().map(|mut slot| slot.take().is_some()).unwrap_or(false)
}

pub fn set_ffn_host_from(layers: usize, first: usize) {
    if let Ok(mut slot) = FFN_HOST_FROM.lock() {
        *slot = Some((layers, first));
    }
}

/// One expert of a mixture-of-experts layer whose weights live on the device.
pub struct DeviceExpert {
    pub gate: crate::backend::DeviceWeight,
    pub up: crate::backend::DeviceWeight,
    pub down: crate::backend::DeviceWeight,
}

/// The experts of a layer kept on the device: decode and prompt processing then need no host work for the weights.
pub struct DeviceMoe {
    pub experts: Vec<Option<DeviceExpert>>,
    pub shared: Option<DeviceExpert>,
    /// Router weights and per-projection pointer tables for the fused decode path (`moe_fast`).
    pub fast: Option<DeviceMoeFast>,
}

pub struct DeviceMoeFast {
    pub router: crate::backend::DeviceWeight,
    pub gate: crate::backend::DeviceBuffer,
    pub up: crate::backend::DeviceBuffer,
    pub down: crate::backend::DeviceBuffer,
    /// Tables of the experts' bias vectors (gpt-oss), when the projections have them.
    pub gate_bias: Option<crate::backend::DeviceBuffer>,
    pub up_bias: Option<crate::backend::DeviceBuffer>,
    pub down_bias: Option<crate::backend::DeviceBuffer>,
    pub kinds: [crate::backend::QuantType; 3],
    pub ff: usize,
    /// Every expert weight is a format of the int8 dot-product decode kernels (`moe_linear`).
    pub dp4a: bool,
    /// The device MoE operations (`Backend::moe_device_available`) serve this layer: general routing, grouped products for any number
    /// of rows, the shared expert and the combination without host work. `None`: the earlier softmax-only decode path.
    pub device: Option<DeviceMoeDevice>,
}

/// What the device MoE operations need besides the expert tables (see `rust_decoder/moe_device.rs`).
pub struct DeviceMoeDevice {
    pub route: crate::backend::MoeRoute,
    /// Selection bias of the router (DeepSeek `e_score_correction_bias`).
    pub bias: Option<crate::backend::DeviceBuffer>,
    /// Gate vector of the shared expert (`sigmoid(g . x)`), when it has one.
    pub shared_gate: Option<crate::backend::DeviceBuffer>,
    /// The grouped GEMM runs the formats and widths of this layer (both pair tiles).
    pub grouped: bool,
}

pub struct DeviceLayerWeights {
    pub attn_norm: Option<crate::backend::DeviceBuffer>,
    pub q: Option<crate::backend::DeviceWeight>,
    pub k: Option<crate::backend::DeviceWeight>,
    pub v: Option<crate::backend::DeviceWeight>,
    /// Output projection of the attention / mixer on the device (`None` for layers that run on the host or have no mixer).
    pub out: Option<crate::backend::DeviceWeight>,
    pub ffn_norm: Option<crate::backend::DeviceBuffer>,
    /// Biases of the four norms (LayerNorm models).
    pub attn_norm_b: Option<crate::backend::DeviceBuffer>,
    pub ffn_norm_b: Option<crate::backend::DeviceBuffer>,
    pub post_attn_norm_b: Option<crate::backend::DeviceBuffer>,
    pub post_ffn_norm_b: Option<crate::backend::DeviceBuffer>,
    /// The whole layer runs on the host (no kernels for its mixer / features): the residual stream goes down and comes back.
    pub host_layer: bool,
    /// The feed-forward runs on the host with the reference code (features without a device implementation).
    pub host_ffn: bool,
    /// Routed value experts (MoVA): the values of this layer come from the host reference, everything else runs on the device.
    pub host_values: bool,
    /// Projection of the separate attention output gate (AFMoE, K2 Horizon).
    pub attn_gate: Option<crate::backend::DeviceWeight>,
    /// BitNet sub-norms: over the attention output in front of the output projection / over the activated feed-forward product.
    pub attn_sub_norm: Option<crate::backend::DeviceBuffer>,
    pub ffn_sub_norm: Option<crate::backend::DeviceBuffer>,
    /// Query / key RMS norms (Qwen3, Gemma3 ...), applied over `weight.len()` values.
    pub q_norm: Option<crate::backend::DeviceBuffer>,
    pub k_norm: Option<crate::backend::DeviceBuffer>,
    /// Norms of the attention and feed-forward branch outputs (Gemma 2/3).
    pub post_attn_norm: Option<crate::backend::DeviceBuffer>,
    pub post_ffn_norm: Option<crate::backend::DeviceBuffer>,
    pub ffn_gate: Option<crate::backend::DeviceWeight>,
    pub ffn_up: Option<crate::backend::DeviceWeight>,
    pub ffn_down: Option<crate::backend::DeviceWeight>,
    pub moe: Option<DeviceMoe>,
    /// Gated DeltaNet mixer resident on the device.
    delta: Option<DeviceDelta>,
    /// Multi-head latent attention projections resident on the device.
    mla: Option<DeviceMla>,
    /// Lightning indexer of a sparse-attention layer resident on the device (`Stream::dsa_select`).
    dsa: Option<DeviceDsa>,
    /// Block indexer of a MiniMax-M3 block-sparse attention layer resident on the device (`Stream::block_select`).
    m3: Option<DeviceM3>,
    /// An Inkling layer resident on the device.
    ink: Option<inkling_device::DeviceInk>,
    /// Mamba-2 mixer resident on the device.
    mamba: Option<DeviceMamba>,
    /// Short convolution resident on the device.
    shortconv: Option<DeviceShortConv>,
    /// Mamba-1 mixer resident on the device.
    mamba1: Option<DeviceMamba1>,
    /// The shared transformer block call in front of the Mamba-2 mixer (Zamba2).
    zamba: Option<zamba::DeviceZamba>,
    /// The Mamba-2 branch next to the attention (Falcon-H1) and its output projection.
    par_mamba: Option<(DeviceMamba, crate::backend::DeviceWeight)>,
    /// RWKV-6 / RWKV-7 layer (time mix and channel mix) resident on the device.
    rwkv: Option<rwkv::DeviceRwkv>,
    /// Kimi Delta Attention mixer resident on the device.
    kda: Option<kda::DeviceKda>,
    /// DeepSeek-V4 attention resident on the device.
    v4: Option<deepseek4::DeviceV4Layer>,
    /// Per-head attention sink logits (GPT-OSS).
    pub sinks: Option<Arc<Vec<f32>>>,
    /// q, k and v as one matrix (`FLLM_MERGE_PROJ`, narrow activations, Marlin 4-bit): `q`, `k` and `v` are `None` then.
    pub qkv: Option<DeviceMergedProj>,
    /// gate and up as one matrix: `ffn_gate` and `ffn_up` are `None` then, `ffn_down` is set.
    pub gate_up: Option<DeviceMergedProj>,
    /// Gemma per-layer input path after the feed-forward block: gate (hidden -> per-layer width), projection back and its norm.
    pub ple: Option<(crate::backend::DeviceWeight, crate::backend::DeviceWeight, crate::backend::DeviceBuffer)>,
    /// Gemma layer output scale (multiplies the residual stream at the end of the layer).
    pub out_scale: Option<f32>,
    /// Weightless RMS norm of every value head (Gemma 4 / 3n): a vector of ones of the value width.
    pub v_norm: Option<crate::backend::DeviceBuffer>,
}

/// Several projections of a layer uploaded as one matrix (`FLLM_MERGE_PROJ`): rows of the parts in `parts` (q, k, v or gate, up, 0).
pub struct DeviceMergedProj {
    pub weight: crate::backend::DeviceWeight,
    pub parts: [usize; 3],
}

impl DeviceLayerWeights {
    /// A layer without device weights: it runs on the host (`host`), or another pipeline stage runs it.
    fn empty(host: bool) -> Self {
        Self {
            attn_norm: None,
            q: None,
            k: None,
            v: None,
            out: None,
            ffn_norm: None,
            attn_norm_b: None,
            ffn_norm_b: None,
            post_attn_norm_b: None,
            post_ffn_norm_b: None,
            host_layer: host,
            host_ffn: host,
            host_values: false,
            attn_gate: None,
            attn_sub_norm: None,
            ffn_sub_norm: None,
            q_norm: None,
            k_norm: None,
            post_attn_norm: None,
            post_ffn_norm: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
            moe: None,
            delta: None,
            mla: None,
            dsa: None,
            m3: None,
            ink: None,
            mamba: None,
            shortconv: None,
            mamba1: None,
            rwkv: None,
            kda: None,
            v4: None,
            zamba: None,
            par_mamba: None,
            sinks: None,
            qkv: None,
            gate_up: None,
            ple: None,
            out_scale: None,
            v_norm: None,
        }
    }
}

/// Tells the stream when a forward pass is over (`Stream::end_step`), on every return path of `forward_backend_inner`.
struct StepScope<'a>(&'a std::sync::Mutex<Box<dyn crate::backend::Stream>>);

impl Drop for StepScope<'_> {
    fn drop(&mut self) {
        if let Ok(mut stream) = self.0.lock() {
            stream.end_step();
        }
    }
}

/// Debug aid (`FLLM_DEBUG_SUMS=1`): prints the sum of a device buffer, to compare layer outputs with a reference.
fn debug_sum(stream: &mut dyn crate::backend::Stream, name: &str, buf: &crate::backend::DeviceBuffer) {
    if std::env::var_os("FLLM_DEBUG_SUMS").is_none() {
        return;
    }
    let mut host = vec![0.0f32; buf.size / 4];
    if stream.download(cast_f32_to_u8_mut(&mut host), buf).is_ok() {
        let n = host.len();
        eprintln!(
            "[sum] {name} {:.6} first {:.4} {:.4} {:.4} last {:.4} {:.4} {:.4}",
            host.iter().map(|&v| f64::from(v)).sum::<f64>(),
            host[0],
            host[1.min(n - 1)],
            host[2.min(n - 1)],
            host[n.saturating_sub(3)],
            host[n.saturating_sub(2)],
            host[n - 1]
        );
    }
}

/// RMS-normalizes `buf` in place over `weight.len()` values (through a pooled temporary); no-op without weights.
/// `FLLM_MERGE_PROJ=1`: q/k/v and gate/up uploaded as one matrix each. Needs narrow activations and a backend with the merged-projection operations (with FP32
/// activations the existing norm / GLU / conversion fusions of the separate projections are better). Default off: the merged GEMM has another reduction
/// order than the separate ones, so results can differ in the last bits.
fn merge_proj_requested(narrow: bool, supported: bool) -> bool {
    let asked = matches!(std::env::var("FLLM_MERGE_PROJ").ok().map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1" | "on" | "true" | "yes"));
    if asked && !(narrow && supported) {
        eprintln!("[fllm-rs] FLLM_MERGE_PROJ=1 needs FLLM_ACT_DTYPE=f16|bf16 and a CUDA GPU; projections stay separate");
        return false;
    }
    asked
}

/// `FLLM_FUSE_QK_NORM_ROPE=1`: per-head RMS norm of q and k, rotary embedding and KV cache write in one kernel (default off until measured).
fn qk_norm_rope_fusion() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("FLLM_FUSE_QK_NORM_ROPE").ok().map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1" | "on" | "true" | "yes")))
}

/// `FLLM_ACT_CHECK=1`: true for the first two forward passes of the process.
fn act_check_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static PASSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    *ON.get_or_init(|| matches!(std::env::var("FLLM_ACT_CHECK").ok().map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1" | "on" | "true" | "yes")))
        && PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2
}

/// The per-layer statistics of `FLLM_ACT_CHECK`: largest residual value per layer, first layer with NaN / Inf, and how far the largest value is
/// from the FP16 limit.
fn report_act_check(stream: &mut dyn crate::backend::Stream, rows: usize, stats: &[(f32, u32)]) {
    let dtype = stream.act_dtype();
    if stats.is_empty() {
        eprintln!("[fllm-rs] FLLM_ACT_CHECK: this backend cannot inspect its activations");
        return;
    }
    let (mut worst, mut worst_layer) = (0f32, 0usize);
    for (i, (m, _)) in stats.iter().enumerate() {
        if *m > worst {
            worst = *m;
            worst_layer = i;
        }
    }
    let bad: Vec<(usize, u32)> = stats.iter().enumerate().filter(|(_, (_, b))| *b > 0).map(|(i, (_, b))| (i, *b)).collect();
    let headroom = if worst > 0.0 { 65504.0 / worst } else { f32::INFINITY };
    eprintln!(
        "[fllm-rs] FLLM_ACT_CHECK ({dtype}, {rows} rows, {} layers): largest |residual| {worst:.4e} after layer {worst_layer}; FP16 headroom {headroom:.1}x{}",
        stats.len(),
        match bad.first() {
            Some((layer, count)) => format!("; NaN/Inf first after layer {layer} ({count} values), {} layers affected", bad.len()),
            None => "; no NaN/Inf".to_owned(),
        }
    );
    if headroom < 4.0 || !bad.is_empty() {
        eprintln!("[fllm-rs] FLLM_ACT_CHECK: the residual stream is close to or beyond the FP16 range; use FLLM_ACT_DTYPE=bf16 (or f32) for this model");
    }
    if dtype == crate::backend::act_dtype::ActDtype::F16 && (headroom <= 1.0 || !bad.is_empty()) {
        eprintln!("[fllm-rs] FLLM_ACT_CHECK: FP16 overflow encountered (worst {worst:.4e}); falling back from FP16 to BF16 at runtime");
        let _ = stream.set_act_dtype(crate::backend::act_dtype::ActDtype::Bf16);
    }
}

fn device_rms_inplace(
    stream: &mut dyn crate::backend::Stream,
    scratch: &mut crate::backend::BufferPool,
    buf: &mut crate::backend::DeviceBuffer,
    weight: Option<&crate::backend::DeviceBuffer>,
    eps: f32,
    groups: usize,
) -> Result<(), String> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let mut out = scratch.acquire(buf.size).map_err(|e| e.to_string())?;
    if groups > 1 {
        stream.rms_norm_grouped(&mut out, buf, weight, eps, groups).map_err(|e| e.to_string())?;
    } else {
        stream.rms_norm(&mut out, buf, weight, eps).map_err(|e| e.to_string())?;
    }
    std::mem::swap(buf, &mut out);
    scratch.recycle(out);
    Ok(())
}

/// Q8_0 blocks (`f16` scale, 32 `i8`) of `data`, whose length is a multiple of 32 (ggml's `quantize_row_q8_0`).
fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 32 * 34);
    for block in data.as_chunks::<32>().0 {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        out.extend(block.iter().map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8));
    }
    out
}

/// `out = norm(x)` with the model's norm kind (RMS or LayerNorm).
#[allow(clippy::too_many_arguments)]
fn device_norm(
    stream: &mut dyn crate::backend::Stream,
    kind: NormKind,
    out: &mut crate::backend::DeviceBuffer,
    x: &crate::backend::DeviceBuffer,
    weight: &crate::backend::DeviceBuffer,
    bias: Option<&crate::backend::DeviceBuffer>,
    eps: f32,
    groups: usize,
) -> Result<(), String> {
    match kind {
        NormKind::Rms => {
            if groups > 1 {
                stream.rms_norm_grouped(out, x, weight, eps, groups)
            } else {
                stream.rms_norm(out, x, weight, eps)
            }
        }
        NormKind::Layer => stream.layer_norm(out, x, weight, bias, eps),
    }
    .map_err(|e| e.to_string())
}

/// In-place norm of a branch output (Gemma 2/3 post norms); nothing without a weight.
#[allow(clippy::too_many_arguments)]
fn device_post_norm_inplace(
    stream: &mut dyn crate::backend::Stream,
    scratch: &mut crate::backend::BufferPool,
    kind: NormKind,
    buf: &mut crate::backend::DeviceBuffer,
    weight: Option<&crate::backend::DeviceBuffer>,
    bias: Option<&crate::backend::DeviceBuffer>,
    eps: f32,
    groups: usize,
) -> Result<(), String> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let mut out = scratch.acquire(buf.size).map_err(|e| e.to_string())?;
    device_norm(stream, kind, &mut out, buf, weight, bias, eps, groups)?;
    std::mem::swap(buf, &mut out);
    scratch.recycle(out);
    Ok(())
}

/// Code of an activation for `Stream::activation` / `activation_glu`.
fn act_code(a: Activation) -> i32 {
    match a {
        Activation::Silu => 0,
        Activation::GeluTanh => 1,
        Activation::GeluErf => 2,
        Activation::Relu => 3,
        Activation::ReluSquared => 4,
    }
}

pub struct DeviceModelWeights {
    pub layers: Vec<DeviceLayerWeights>,
    pub final_norm: crate::backend::DeviceBuffer,
    pub final_norm_b: Option<crate::backend::DeviceBuffer>,
    pub head: crate::backend::DeviceWeight,
    /// (cos, sin) of the global and the sliding-window rotary table, shared with every
    /// `RopeParams` so a step does not copy them.
    pub rope: Vec<SharedRopeTable>,
    pub alibi: Option<Arc<Vec<f32>>>,
    /// Gemma per-layer inputs: the projection of the embeddings to `layers x width` (the rest of the path is cheap and stays on the host).
    pub ple_proj: Option<crate::backend::DeviceWeight>,
    /// Hyper-connection weights (Hy4, GLM-5-Next, DeepSeek-V4); the streams then live in their own buffer during a pass.
    hc: Option<DeviceHcModel>,
    /// DeepSeek-V4: `head_dim` ones, the weight of the plain RMS norm of every query head.
    v4_ones: Option<crate::backend::DeviceBuffer>,
}

/// Rotary cos/sin tables shared between the decoder and device `RopeParams`.
pub type SharedRopeTable = (Arc<Vec<f32>>, Arc<Vec<f32>>);

/// Batch sizes a decode step is padded to when it runs as a CUDA graph.
const GRAPH_BUCKETS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Largest batch a decode step may have to run as a CUDA graph: `FLLM_CUDA_GRAPH_MAX_ROWS` (default 64, at most 256, in the buckets).
fn graph_max_rows() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        let asked = std::env::var("FLLM_CUDA_GRAPH_MAX_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64);
        GRAPH_BUCKETS.iter().copied().filter(|b| *b <= asked.max(1)).max().unwrap_or(1)
    })
}

/// Buckets for prompt chunk CUDA graphs (`FLLM_CUDA_GRAPH_PREFILL`).
pub const PREFILL_GRAPH_BUCKETS: [usize; 5] = [128, 256, 512, 1024, 2048];

/// `FLLM_CUDA_GRAPH_PREFILL`: piecewise CUDA graphs around attention for prompt chunks (default off).
pub fn graph_prefill_switch() -> bool {
    static SWITCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SWITCH.get_or_init(|| matches!(std::env::var("FLLM_CUDA_GRAPH_PREFILL").ok().map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("1" | "on" | "true" | "yes")))
}

/// A decode step recorded for one padded batch size: the persistent input (residual stream) and output (logits) buffers.
struct DecodeGraph {
    id: usize,
    x_in: crate::backend::DeviceBuffer,
    head_out: crate::backend::DeviceBuffer,
}
/// What `forward_backend_inner` takes from / hands to a graph capture.
struct GraphIo {
    x_in: crate::backend::DeviceBuffer,
    head_out: Option<crate::backend::DeviceBuffer>,
}
enum GraphMode<'a> {
    Off,
    Capture(&'a mut GraphIo),
}

/// Set by the runtime right before a decode-only step in which every sequence samples greedily: the next head projection then
/// returns the argmax ids (`take_greedy_ids`) instead of downloading `rows x vocab` logits.
static GREEDY_IDS_REQUEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static GREEDY_IDS: std::sync::Mutex<Option<Vec<i32>>> = std::sync::Mutex::new(None);

/// The rows of the next forward pass that the device should sample (`FLLM_DEVICE_SAMPLING=1`, see docs/speculative-decoding.md), and the
/// result. Like the greedy ids they travel through statics because the decoder is shared by the runtime and the draft engines.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeviceSampleJob {
    pub rows: Vec<crate::backend::SampleRowParams>,
    pub pen: Vec<(i32, i32)>,
    pub bias: Vec<(i32, f32)>,
    pub mask: Vec<u32>,
    pub mask_words: usize,
    pub top_stride: usize,
}
impl DeviceSampleJob {
    fn batch(&self) -> crate::backend::SampleBatch<'_> {
        crate::backend::SampleBatch { rows: &self.rows, pen: &self.pen, bias: &self.bias, mask: &self.mask, mask_words: self.mask_words, top_stride: self.top_stride }
    }
}
static SAMPLE_JOB: std::sync::Mutex<Option<DeviceSampleJob>> = std::sync::Mutex::new(None);
static SAMPLE_RESULT: std::sync::Mutex<Option<crate::backend::SampleOutput>> = std::sync::Mutex::new(None);

/// `FLLM_DEVICE_SAMPLING`: `1`, `on` or `true` sample decode steps on the device (default: off until the kernels were run on a GPU).
pub(crate) fn device_sampling_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("FLLM_DEVICE_SAMPLING").ok().as_deref(), Some("1" | "on" | "true")))
}

/// Run the job on the device if the backend can; `None` leaves the logits to the caller.
fn run_device_sample(stream: &mut dyn crate::backend::Stream, logits: &crate::backend::DeviceBuffer, vocab: usize, rows: usize) -> bool {
    let mut job_slot = SAMPLE_JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(job) = job_slot.take() else { return false };
    if job.rows.len() != rows {
        return false;
    }
    match stream.sample_rows(logits, vocab, &job.batch()) {
        Ok(out) => {
            *SAMPLE_RESULT.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(out);
            true
        }
        Err(e) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("[fllm-rs] device sampling unavailable, sampling on the host: {e}"));
            false
        }
    }
}

pub struct RustDecoder {
    mtp: Option<MtpProjection>,
    /// Owner of the expert cache tables of the MoE layers (keeps the read-ahead thread alive).
    expert_cache: Option<crate::expert_cache::ExpertCache>,
    cfg: ModelConfig,
    profile: bool,
    embedding: Embedding,
    layers: Vec<Layer>,
    head: Linear,
    final_norm: Norm,
    /// Per-layer embeddings (Gemma 4 E2B / E4B).
    ple: Option<PleGlobal>,
    /// Gemma 3n stream projections.
    altup: Option<AltupGlobal>,
    rope: Arc<Vec<RopeTables>>,
    groups: usize,
    stride: usize,
    block_size: usize,
    pool_blocks: usize,
    kv_kind: i32,
    /// The KV cache holds MLA latents (`FLLM_MLA=latent`, `GgufModel::enable_mla_latent`); the layers run `device_mla_latent`.
    mla_latent: bool,
    /// Offsets of every layer's region in the key and value pools.
    kvl: crate::kv_geometry::KvPoolLayout,
    pool: rayon::ThreadPool,
    /// A larger pool for batches of at least [`PREFILL_ROWS`] rows (prompts are compute-bound and use the second hardware thread
    /// of every core, decoding is memory-bound and does not); `None` when it would not differ from `pool`.
    prefill_pool: Option<rayon::ThreadPool>,
    /// Learned absolute position embeddings, `positions x hidden`.
    position_table: Option<Vec<f32>>,
    /// LayerNorm applied to the embeddings before the first layer (BLOOM).
    embd_norm: Option<Norm>,
    /// Recurrent state of hybrid layers, per live sequence.
    states: std::sync::Mutex<StateStore>,
    /// Recurrent state of the DeltaNet layers that run on the device.
    device_states: std::sync::Mutex<DeviceStates>,
    state_capacity: usize,
    backend: Option<Arc<dyn crate::backend::Backend>>,
    backend_stream: Option<std::sync::Mutex<Box<dyn crate::backend::Stream>>>,
    backend_stream2: Option<std::sync::Mutex<Box<dyn crate::backend::Stream>>>,
    backend_weights: Option<Arc<DeviceModelWeights>>,
    backend_kv: Option<crate::backend::DeviceKvCache>,
    backend_scratch: Option<std::sync::Mutex<crate::backend::BufferPool>>,
    /// CUDA graphs of decode steps by padded batch size (see `try_graph_decode`).
    graphs: std::sync::Mutex<std::collections::HashMap<usize, DecodeGraph>>,
    graph_ok: std::sync::OnceLock<bool>,
    graph_failed: std::sync::atomic::AtomicBool,
    /// The embedding / output tensors of the file are BF16: `FLLM_ACT_DTYPE=auto` then picks BF16 (the GGUF does not record the config dtype).
    act_source_bf16: bool,
    /// This decoder is one rank of a tensor / pipeline / expert parallel group.
    par: Option<Arc<crate::parallel::ParCtx>>,
    /// Embedding output of the current pass (Zamba2's shared blocks read it again at every call).
    embed_stash: hybrid::EmbedStash,
    pub tree_mask: Option<Vec<bool>>,
    pub tree_prefix_len: usize,
    pub tree_num_nodes: usize,
    /// Multimodal rotary layout (Qwen2-VL family): which position axis each rotary lane reads.
    mrope: Option<MropeState>,
    /// Three-axis position ids of the rows of the next forward pass; set by the runtime for batches that hold a sequence with media.
    row_positions3: std::sync::Mutex<Option<Vec<[u32; 3]>>>,
    /// DeepStack features (Qwen3-VL) of the next forward pass: per decoder layer, (row, values) added to the hidden state after that layer.
    row_deepstack: std::sync::Mutex<Option<Arc<DeepstackLevels>>>,
    /// Custom visible lengths per row in attention (bidirectional image attention, prefix-LM mask).
    row_visible_lengths: std::sync::Mutex<Option<Vec<usize>>>,
    /// DeepSeek-V4 attention and hyper-connection weights.
    v4: Option<Box<deepseek4::V4Model>>,
    /// MiniMax-M3 weights.
    minimax3: Option<Box<minimax3::Minimax3Model>>,
    qwen4: Option<Box<qwen4::Qwen4Model>>,
    /// Score weights of the attention residual over the blocks at the end of the model.
    output_res: Option<Vec<f32>>,
    /// Inkling weights.
    inkling: Option<Box<inkling::InklingModel>>,
    /// Hyper-connection weights of Hy4 and GLM-5-Next.
    hc: Option<Box<hc::HcModel>>,
    /// Token ids of the rows of the current pass (hash-routed experts).
    row_ids: std::sync::Mutex<Vec<i64>>,
    /// Indexer keys of the sparse-attention layers (host path).
    dsa: DsaStore,
    /// `FLLM_DSA=dense`: attend to every token (exact up to `top_k` tokens of context, an approximation beyond).
    dsa_dense: bool,
    /// `FLLM_DSA=exact`: run the sparse-attention layers on the host also when a device is in use. Without it a device keeps
    /// the dense attention of its own kernels (and says so once).
    dsa_exact_on_device: bool,
    /// The accelerator runs the lightning indexer and the sparse attention of the layers (`Stream::dsa_select`); set when the model is attached.
    dsa_device: bool,
    /// The accelerator runs the hyper-connections (`Stream::hc_mix`, `Stream::hc_combine`); set when the model is attached.
    hc_device: bool,
    /// MiniMax-M3 on the accelerator: the block indexer and the block-sparse attention run on the device (`Stream::block_select`); set when the model is attached.
    m3_device: bool,
    /// MiniMax-M3 runs on the accelerator at all (`m3_device`, or dense attention with `FLLM_DSA=dense`).
    m3_run: bool,
    /// Inkling runs on the accelerator (convolutions, relative-position attention, experts); set when the model is attached.
    ink_device: bool,
    /// The accelerator runs DeepSeek-V4 completely (`hc_device` is set with it); set when the model is attached.
    v4_device: bool,
}

/// DeepStack features: per decoder layer, (row, values) pairs.
type DeepstackLevels = Vec<Vec<(usize, Vec<f32>)>>;
/// Per position axis, the (cos, sin) rotary tables of the device path.
type AxisTables = Vec<(Arc<Vec<f32>>, Arc<Vec<f32>>)>;

/// Multimodal rotary embedding of a model: lane-to-axis map and (built on first use by the device path) one pair of rotary tables per axis
/// that rotate only the lanes of that axis and leave the others unchanged, so three plain rope calls compose to the three-axis rotation.
struct MropeState {
    axes: Vec<u8>,
    axis_tables: std::sync::OnceLock<AxisTables>,
}

/// Selection of the k-pool indexer for a query that sees `n` tokens: the `topk / kpool` best complete pools (a pool is visible once its
/// last token is) and, with `always_tail`, the tokens of the incomplete pool the query is in. `store(p)` is `[key, gate score]` of
/// token `p`.
fn pooled_selection<'a>(ic: &crate::config::IndexerConfig, q: &[f32], weights: &[f32], ape: &[f32], n: usize, store: impl Fn(usize) -> &'a [f32], trace: bool) -> Vec<u32> {
    let (dim, kpool, heads) = (ic.dim, ic.kpool, weights.len());
    let complete = n / kpool;
    let (scale, head_scale) = ((dim as f32).powf(-0.5), (heads as f32).powf(-0.5));
    let scored: Vec<(f32, u32)> = (0..complete)
        .map(|pool| {
            // softmax over the pool of (gate score + position bias), per channel, weighting the keys
            let mut pooled = vec![0.0f32; dim];
            for (c, out) in pooled.iter_mut().enumerate() {
                let logits: Vec<f32> = (0..kpool).map(|j| store(pool * kpool + j)[dim + c] + ape[j * dim + c]).collect();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                *out = exps.iter().enumerate().map(|(j, e)| e / sum * store(pool * kpool + j)[c]).sum();
            }
            let mut score = 0.0f32;
            for h in 0..heads {
                let dot = compute::dot(&q[h * dim..(h + 1) * dim], &pooled) * scale;
                if dot > 0.0 {
                    score += dot * weights[h] * head_scale;
                }
            }
            (score, pool as u32)
        })
        .collect();
    if trace {
        let mut all = scored.clone();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        eprintln!("[dsa-pools] visible tokens {n}: {:?}", &all[..all.len().min(ic.topk / kpool + 2)]);
    }
    let mut chosen: Vec<u32> = best_positions(scored, ic.topk / kpool).into_iter().flat_map(|pool| (0..kpool as u32).map(move |j| pool * kpool as u32 + j)).collect();
    if ic.always_tail {
        chosen.extend(((n - n % kpool) as u32)..n as u32);
    }
    chosen.sort_unstable();
    chosen
}

/// The positions of the `k` highest scores, ascending. Equal scores keep the lower position (the score of a key whose indexer heads
/// are all in the ReLU's dead zone is exactly 0, so ties are common when a query has fewer than `k` positive keys).
fn best_positions(mut scored: Vec<(f32, u32)>, k: usize) -> Vec<u32> {
    if k == 0 || scored.is_empty() {
        return Vec::new();
    }
    let k = k.min(scored.len());
    scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut best: Vec<u32> = scored[..k].iter().map(|s| s.1).collect();
    best.sort_unstable();
    best
}

fn rms(x: &mut [f32], weight: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(weight.len()) {
        let inv = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps)
            .sqrt()
            .recip();
        for (x, w) in row.iter_mut().zip(weight) {
            *x *= inv * w;
        }
    }
}
/// `x += b * scale`, spread over the pool for long vectors (elementwise, so identical results).
fn add_scaled(x: &mut [f32], b: &[f32], scale: f32) {
    let add = |x: &mut [f32], b: &[f32]| {
        for (x, b) in x.iter_mut().zip(b) {
            *x += b * scale;
        }
    };
    if x.len() >= 1 << 16 {
        x.par_chunks_mut(8192)
            .zip(b.par_chunks(8192))
            .for_each(|(x, b)| add(x, b));
    } else {
        add(x, b);
    }
}

/// Rows from which a normalisation is spread over the pool (a row is independent of the others,
/// so the result is the same bit for bit).
const PARALLEL_NORM_ROWS: usize = 8;

fn grouped_rms(x: &mut [f32], weight: &[f32], eps: f32, groups: usize) {
    let size = weight.len() / groups;
    let one = |row: &mut [f32]| {
        for (x, w) in row.chunks_exact_mut(size).zip(weight.chunks_exact(size)) {
            rms(x, w, eps);
        }
    };
    if x.len() >= PARALLEL_NORM_ROWS * weight.len() {
        x.par_chunks_exact_mut(weight.len()).for_each(one);
    } else {
        x.chunks_exact_mut(weight.len()).for_each(one);
    }
}
fn layer_norm(x: &mut [f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) {
    let one = |row: &mut [f32]| {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
        let inv = (var + eps).sqrt().recip();
        for (i, (x, w)) in row.iter_mut().zip(weight).enumerate() {
            *x = (*x - mean) * inv * w + bias.map_or(0.0, |b| b[i]);
        }
    };
    if x.len() >= PARALLEL_NORM_ROWS * weight.len() {
        x.par_chunks_exact_mut(weight.len()).for_each(one);
    } else {
        x.chunks_exact_mut(weight.len()).for_each(one);
    }
}

/// Applies the per-head query / key norm. Models with a LayerNorm there (Persimmon) carry a bias;
/// everything else uses a weight-only RMSNorm.
fn qk_norm_cpu(x: &mut [f32], weight: Option<&[f32]>, bias: Option<&[f32]>, eps: f32) {
    let Some(weight) = weight else {
        return;
    };
    if bias.is_some() {
        layer_norm(x, weight, bias, eps);
    } else {
        rms(x, weight, eps);
    }
}

/// Splits Persimmon's fused `attn_qkv` projection into separate Q / K / V projections. The HF
/// checkpoint (and the GGUF produced from it) lays the rows out head-interleaved: for every head
/// the three `head_dim`-wide blocks `[q_h, k_h, v_h]` are contiguous, so each part gathers its own
/// blocks instead of taking one contiguous slice.
fn persimmon_qkv(
    mapping: &MappedGguf,
    fused: &str,
    bias: &str,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Attn, String> {
    let tensor = Tensor::load(mapping, fused)?;
    if tensor.shape != [hidden, 3 * hidden] || tensor.data.len() != 3 * hidden * hidden {
        return Err(format!(
            "Persimmon attn_qkv must be an unquantized [{hidden}, {}] tensor",
            3 * hidden
        ));
    }
    let cols = hidden;
    let take = |part: usize, hk: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(hk * head_dim * cols);
        for h in 0..hk {
            let base = (h * 3 + part) * head_dim;
            for r in 0..head_dim {
                let row = (base + r) * cols;
                out.extend_from_slice(&tensor.data[row..row + cols]);
            }
        }
        out
    };
    let bias_full = if has_tensor(mapping, bias) {
        Some(vector(mapping, bias, 3 * hidden)?)
    } else {
        None
    };
    let take_bias = |part: usize, hk: usize| -> Option<Vec<f32>> {
        bias_full.as_ref().map(|b| {
            let mut out = Vec::with_capacity(hk * head_dim);
            for h in 0..hk {
                let base = (h * 3 + part) * head_dim;
                out.extend_from_slice(&b[base..base + head_dim]);
            }
            out
        })
    };
    let build = |part: usize, hk: usize, data: Vec<f32>| -> Result<Linear, String> {
        let mut l = Linear::from_parts(
            Tensor {
                shape: vec![cols, hk * head_dim],
                data,
            },
            hk * head_dim,
            cols,
            take_bias(part, hk),
        )?;
        l.enable_fast_cpu();
        Ok(l)
    };
    Ok(Attn::Std {
        q: build(0, heads, take(0, heads))?,
        k: build(1, kv_heads, take(1, kv_heads))?,
        v: build(2, kv_heads, take(2, kv_heads))?,
    })
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}
fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}
/// Chebyshev fit of erfc (Numerical Recipes), fractional error below 1.2e-7.
fn erf(x: f32) -> f32 {
    let x = f64::from(x);
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { 1.0 - tau } else { tau - 1.0 }) as f32
}
fn activate(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::GeluTanh => gelu_tanh(x),
        Activation::GeluErf => gelu_erf(x),
        Activation::Relu => x.max(0.0),
        Activation::ReluSquared => x.max(0.0) * x.max(0.0),
    }
}

/// Rotate the first `rope_dims` values of every head. `positions[r]` is the position of row `r`.
#[allow(clippy::too_many_arguments)]
fn apply_rope(
    x: &mut [f32],
    heads: usize,
    dim: usize,
    rope_dims: usize,
    neox: bool,
    positions: &[usize],
    tables: &RopeTables,
) {
    let half = tables.half;
    debug_assert_eq!(half * 2, rope_dims);
    for (r, &position) in positions.iter().enumerate() {
        let cos = &tables.cos[position * half..(position + 1) * half];
        let sin = &tables.sin[position * half..(position + 1) * half];
        for h in 0..heads {
            let head = &mut x[(r * heads + h) * dim..(r * heads + h) * dim + rope_dims];
            for d in 0..half {
                let (a_index, b_index) = if neox {
                    (d, d + half)
                } else {
                    (2 * d, 2 * d + 1)
                };
                let (a, b) = (head[a_index], head[b_index]);
                head[a_index] = a * cos[d] - b * sin[d];
                head[b_index] = b * cos[d] + a * sin[d];
            }
        }
    }
}

fn write_q8_cache_row(
    q_words: &mut [u16],
    scales: &mut [u16],
    slot: usize,
    kv: usize,
    group: usize,
    row: &[f32],
) {
    debug_assert!(group > 0 && kv.is_multiple_of(group));
    debug_assert_eq!(row.len(), kv);
    let groups_per_slot = kv / group;
    debug_assert!(scales.len().is_multiple_of(groups_per_slot));
    let slots = scales.len() / groups_per_slot;
    debug_assert!(slot < slots);
    debug_assert_eq!(q_words.len(), (slots * kv).div_ceil(2));

    #[cfg(target_endian = "little")]
    {
        // Q bytes and FP16 scales occupy disjoint halves of the cache allocation.
        // The caller split them before creating these slices, and this byte view
        // never reaches the optional pad byte after an odd total Q byte count.
        let q_bytes = unsafe {
            std::slice::from_raw_parts_mut(q_words.as_mut_ptr().cast::<u8>(), q_words.len() * 2)
        };
        for (g, chunk) in row.chunks_exact(group).enumerate() {
            let q_start = slot * kv + g * group;
            let scale_index = slot * groups_per_slot + g;
            crate::cpu_kernels::encode_q8_cache_group(
                chunk,
                &mut q_bytes[q_start..q_start + group],
                &mut scales[scale_index],
            );
        }
    }

    #[cfg(target_endian = "big")]
    {
        for (g, chunk) in row.chunks_exact(group).enumerate() {
            let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = amax / 127.0f32;
            scales[slot * groups_per_slot + g] = f16::from_f32(d).to_bits();
            for (j, &x) in chunk.iter().enumerate() {
                let q = if d == 0.0 {
                    0
                } else {
                    (x / d).round().clamp(-127.0, 127.0) as i8
                };
                let index = slot * kv + g * group + j;
                let word = &mut q_words[index / 2];
                let shift = index % 2 * 8;
                *word = (*word & !(255 << shift)) | ((q as u8 as u16) << shift);
            }
        }
    }
}

fn has_tensor(mapping: &MappedGguf, name: &str) -> bool {
    mapping.index().tensors.iter().any(|t| t.name == name)
}

fn tensor_rows(mapping: &MappedGguf, name: &str) -> Option<usize> {
    mapping
        .index()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.dimensions.iter().skip(1).product::<u64>() as usize)
}

fn vector(mapping: &MappedGguf, name: &str, size: usize) -> Result<Vec<f32>, String> {
    let t = Tensor::load(mapping, name)?;
    if t.data.len() != size {
        return Err(format!(
            "invalid norm {name}: {:?}, expected {size} values",
            t.shape
        ));
    }
    Ok(t.data)
}

fn load_norm(mapping: &MappedGguf, base: &str, size: usize) -> Result<Option<Norm>, String> {
    let weight = format!("{base}.weight");
    if !has_tensor(mapping, &weight) {
        return Ok(None);
    }
    let bias = format!("{base}.bias");
    Ok(Some(Norm {
        weight: vector(mapping, &weight, size)?,
        bias: if has_tensor(mapping, &bias) {
            Some(vector(mapping, &bias, size)?)
        } else {
            None
        },
    }))
}

fn cast_f32_to_u8(slice: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * 4) }
}

fn cast_f32_to_u8_mut(slice: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), slice.len() * 4) }
}

/// Batches of at least this many rows run on the larger prefill pool when there is one.
const PREFILL_ROWS: usize = 8;

impl RustDecoder {
    /// The pool for a step of `rows` rows: single rows (decoding) use the decode pool.
    fn pool_for(&self, rows: usize) -> &rayon::ThreadPool {
        match &self.prefill_pool {
            Some(pool) if rows >= PREFILL_ROWS => pool,
            _ => &self.pool,
        }
    }

    pub fn load(model: &GgufModel, device: &str) -> Result<Self, String> {
        Self::load_layers(model, device, None)
    }

    /// One rank of a parallel group: only this rank's pipeline stage is loaded, and every projection is cut to this rank's share.
    pub(crate) fn load_sharded(model: &GgufModel, device: &str, par: Arc<crate::parallel::ParCtx>) -> Result<Self, String> {
        Self::load_layers_ex(model, device, None, Some(par))
    }

    pub(crate) fn load_mtp(model: &GgufModel, device: &str) -> Result<Self, String> {
        if !matches!(device, "auto" | "cpu") && !device.starts_with("cuda") {
            return Err("MTP currently requires --device cpu or cuda; GPU hidden-state capture is not integrated with the other backends".into());
        }
        if model.is_stateful() {
            return Err(
                "MTP speculative decoding is unavailable for stateful DeltaNet/Mamba models".into(),
            );
        }
        if !matches!(model.config().arch.as_str(), "deepseek2" | "glm4moe") {
            return Err(format!(
                "MTP is supported only for DeepSeek-V2/V3 and GLM-4 MoE, got {}",
                model.config().arch
            ));
        }
        let map = model.mapping();
        let canonical = "blk.0.nextn";
        let appended = format!("blk.{}.nextn", model.config().layers);
        let prefix = if has_tensor(map, &format!("{canonical}.eh_proj.weight")) {
            canonical
        } else {
            &appended
        };
        for suffix in [
            "eh_proj",
            "enorm",
            "hnorm",
            "attn_norm",
            "ffn_norm",
            "attn_output",
        ] {
            let name = format!("{prefix}.{suffix}.weight");
            if !has_tensor(map, &name) {
                return Err(format!("MTP: missing GGUF tensor {name}; older imports retaining raw HF NextN names must be converted again"));
            }
        }
        if !has_tensor(map, &format!("{prefix}.ffn_up.weight"))
            && !has_tensor(map, &format!("{prefix}.ffn_gate_inp.weight"))
        {
            return Err(format!(
                "MTP: missing {prefix}.ffn_up.weight or {prefix}.ffn_gate_inp.weight"
            ));
        }
        Self::load_layers(model, device, Some(prefix))
    }

    fn load_layers(
        model: &GgufModel,
        device: &str,
        mtp_prefix: Option<&str>,
    ) -> Result<Self, String> {
        Self::load_layers_ex(model, device, mtp_prefix, None)
    }

    fn load_layers_ex(
        model: &GgufModel,
        device: &str,
        mtp_prefix: Option<&str>,
        par: Option<Arc<crate::parallel::ParCtx>>,
    ) -> Result<Self, String> {
        if par.is_some() && mtp_prefix.is_some() {
            return Err("MTP speculative decoding is not available with tensor / pipeline parallelism".into());
        }
        let state_capacity = model.max_sequences().max(1) * 2;
        let mapping = model.mapping();
        let cfg = model.config().clone();
        // These new mixers currently have a reference CPU implementation. Keep accelerator
        // dispatch explicit until the separate backend integration supports their operations.
        // (LFM2 runs on the device: the short convolution has CUDA kernels and everything else is attention / dense FFN.)
        let device = if matches!(cfg.arch.as_str(), "dbrx" | "lfm2moe") {
            if !matches!(device, "cpu" | "auto") {
                return Err(format!(
                    "{} currently requires --device cpu; accelerator integration is pending",
                    cfg.arch
                ));
            }
            "cpu"
        } else {
            device
        };
        let meta = &mapping.index().metadata;
        let hidden = cfg.hidden;
        // the projections of an MLA layer have the per-head widths of the model, also when the cache geometry is the latent one
        let dk = match &cfg.mla {
            Some(m) => m.nope + m.rope,
            None => cfg.head_dim,
        };
        if cfg.v4.is_none() && cfg.inkling.is_none() && cfg.minimax3.is_none() && (0..cfg.layers).any(|l| {
            let (d, r) = if cfg.mla.is_some() { (dk, cfg.rope.dims) } else { (cfg.layer_head_dim(l), cfg.layer_rope_dims(l)) };
            !d.is_multiple_of(2) || !r.is_multiple_of(2) || r > d
        }) {
            return Err("the decoder requires even head and rotary dimensions".into());
        }
        if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
            return Err("invalid normalization epsilon".into());
        }
        let groups = meta
            .get(&format!("{}.attention.group_norm_groups", cfg.arch))
            .and_then(Value::as_u64)
            .map_or(1, |n| n as usize);
        if groups == 0 || !hidden.is_multiple_of(groups) {
            return Err("invalid grouped normalization".into());
        }
        let linear = |name: &str, rows, cols| -> Result<Linear, String> {
            let mut l = Linear::load(mapping, name, rows, cols)?;
            l.replace_weights(model.adapted_projection(&format!("{name}.weight")))?;
            l.enable_fast_cpu();
            Ok(l)
        };
        let part = |name: &str, bias: &str, first, rows, cols| -> Result<Linear, String> {
            let mut l = Linear::load_rows(mapping, name, Some(bias), first, rows, cols)?;
            l.enable_fast_cpu();
            Ok(l)
        };
        let layer_range = if mtp_prefix.is_some() {
            cfg.layers..cfg.layers + 1
        } else {
            0..cfg.layers
        };
        let expert_cache = if par.is_none() && mtp_prefix.is_none() && cfg.moe.is_some() { crate::expert_cache::ExpertCache::new(&cfg.arch) } else { None };
        let mut layers = Vec::with_capacity(layer_range.len());
        let mut saw_indexer = false;
        for i in layer_range {
            if par.as_ref().is_some_and(|ctx| !ctx.layer_range(cfg.layers).contains(&i)) {
                // another pipeline stage runs this layer
                layers.push(par_shard::placeholder_layer());
                continue;
            }
            let p = mtp_prefix.map_or_else(|| format!("blk.{i}"), str::to_owned);
            let (heads, kv_heads, dk, dv) = (cfg.layer_heads(i), cfg.layer_kv_heads(i), cfg.layer_head_dim(i), cfg.layer_value_dim(i));
            let ff_i = cfg.layer_ff(i);
            let kv_shared = mtp_prefix.is_none() && cfg.layer_kv_source(i).is_some();
            let k_eq_v = !kv_shared && has_tensor(mapping, &format!("{p}.attn_k.weight")) && !has_tensor(mapping, &format!("{p}.attn_v.weight")) && cfg.arch == "gemma4";
            let (qr, kr, vr) = (heads * dk, kv_heads * dk, kv_heads * dv);
            let q_rows = tensor_rows(mapping, &format!("{p}.attn_q.weight"));
            let output_gate = q_rows == Some(2 * qr) && cfg.mla.is_none();
            let attn = if cfg
                .shortconv
                .as_ref()
                .is_some_and(|s| s.layers.get(i).copied().unwrap_or(false))
            {
                let sc = cfg
                    .shortconv
                    .as_ref()
                    .ok_or("missing ShortConv configuration")?;
                let count = hidden
                    .checked_mul(sc.kernel)
                    .ok_or("ShortConv weight size overflow")?;
                Attn::ShortConv(Box::new(ShortConv {
                    input: linear(&format!("{p}.shortconv_in"), 3 * hidden, hidden)?,
                    conv: vector(mapping, &format!("{p}.shortconv_conv.weight"), count)?,
                    bias: if has_tensor(mapping, &format!("{p}.shortconv_conv.bias")) {
                        Some(vector(
                            mapping,
                            &format!("{p}.shortconv_conv.bias"),
                            hidden,
                        )?)
                    } else {
                        None
                    },
                }))
            } else if let (Some(m), true) = (
                &cfg.mamba1,
                has_tensor(mapping, &format!("{p}.ssm_x.weight")),
            ) {
                let conv = Tensor::load(mapping, &format!("{p}.ssm_conv1d.weight"))?.data;
                if conv.len() != m.inner * m.conv_kernel {
                    return Err(format!("invalid ssm_conv1d in layer {i}"));
                }
                let bias_name = format!("{p}.ssm_conv1d.bias");
                Attn::Mamba1(Box::new(Mamba1 {
                    input: linear(&format!("{p}.ssm_in"), 2 * m.inner, hidden)?,
                    conv,
                    conv_bias: if has_tensor(mapping, &bias_name) {
                        Some(vector(mapping, &bias_name, m.inner)?)
                    } else {
                        None
                    },
                    x_proj: linear(&format!("{p}.ssm_x"), m.rank + 2 * m.state, m.inner)?,
                    dt_proj: linear(&format!("{p}.ssm_dt"), m.inner, m.rank)?,
                    decay: vector(mapping, &format!("{p}.ssm_a"), m.inner * m.state)?,
                    skip: vector(mapping, &format!("{p}.ssm_d"), m.inner)?,
                    norms: hybrid::load_inner_norms(mapping, &p, m)?,
                }))
            } else if let (Some(m), true) = (
                &cfg.mamba,
                has_tensor(mapping, &format!("{p}.ssm_in.weight")) && !has_tensor(mapping, &format!("{p}.attn_q.weight")),
            ) {
                let conv_dim = m.inner + 2 * m.groups * m.state;
                let conv = Tensor::load(mapping, &format!("{p}.ssm_conv1d.weight"))?.data;
                if conv.len() != conv_dim * m.conv_kernel {
                    return Err(format!("invalid ssm_conv1d in layer {i}"));
                }
                let bias_name = format!("{p}.ssm_conv1d.bias");
                Attn::Mamba(Box::new(Mamba {
                    input: linear(&format!("{p}.ssm_in"), m.inner + conv_dim + m.heads, hidden)?,
                    conv,
                    conv_bias: if has_tensor(mapping, &bias_name) {
                        Some(vector(mapping, &bias_name, conv_dim)?)
                    } else {
                        None
                    },
                    dt_bias: vector(mapping, &format!("{p}.ssm_dt.bias"), m.heads)?,
                    decay: vector(mapping, &format!("{p}.ssm_a"), m.heads)?,
                    skip: vector(mapping, &format!("{p}.ssm_d"), m.heads)?,
                    norm: vector(mapping, &format!("{p}.ssm_norm.weight"), m.inner)?,
                }))
            } else if let (Some(d), true) = (
                &cfg.delta,
                has_tensor(mapping, &format!("{p}.ssm_conv1d.weight")),
            ) {
                let value_dim = d.v_heads * d.v_dim;
                let conv_dim = 2 * d.k_heads * d.k_dim + value_dim;
                let conv = Tensor::load(mapping, &format!("{p}.ssm_conv1d.weight"))?.data;
                if conv.len() != conv_dim * d.conv_kernel {
                    return Err(format!("invalid ssm_conv1d in layer {i}"));
                }
                Attn::Linear(Box::new(DeltaNet {
                    qkv: linear(&format!("{p}.attn_qkv"), conv_dim, hidden)?,
                    gate: linear(&format!("{p}.attn_gate"), value_dim, hidden)?,
                    beta: linear(&format!("{p}.ssm_beta"), d.v_heads, hidden)?,
                    alpha: linear(&format!("{p}.ssm_alpha"), d.v_heads, hidden)?,
                    conv,
                    dt_bias: vector(mapping, &format!("{p}.ssm_dt.bias"), d.v_heads)?,
                    decay: vector(mapping, &format!("{p}.ssm_a"), d.v_heads)?,
                    norm: vector(mapping, &format!("{p}.ssm_norm.weight"), d.v_dim)?,
                }))
            } else if let Some(kda) = kda::Kda::load(&cfg, mapping, &p, i, &linear)? {
                Attn::Kda(Box::new(kda))
            } else if let Some(m) = &cfg.mla {
                let q_a = match m.q_lora {
                    Some(rank) => Some((
                        linear(&format!("{p}.attn_q_a"), rank, hidden)?,
                        scaled(vector(mapping, &format!("{p}.attn_q_a_norm.weight"), rank)?, cfg.mla_scales.map(|s| s.0)),
                    )),
                    None => None,
                };
                let q = match m.q_lora {
                    Some(rank) => linear(&format!("{p}.attn_q_b"), qr, rank)?,
                    None => linear(&format!("{p}.attn_q"), qr, hidden)?,
                };
                let kv_rows = heads * (m.nope + m.value);
                let kv_b = if has_tensor(mapping, &format!("{p}.attn_kv_b.weight")) {
                    linear(&format!("{p}.attn_kv_b"), kv_rows, m.kv_lora)?
                } else {
                    // split form: attn_k_b is stored transposed per head
                    let k_b = Tensor::load(mapping, &format!("{p}.attn_k_b.weight"))?.data;
                    let v_b = Tensor::load(mapping, &format!("{p}.attn_v_b.weight"))?.data;
                    if k_b.len() != heads * m.kv_lora * m.nope
                        || v_b.len() != heads * m.value * m.kv_lora
                    {
                        return Err(format!("invalid attn_k_b/attn_v_b shapes in layer {i}"));
                    }
                    let mut data = Vec::with_capacity(kv_rows * m.kv_lora);
                    for h in 0..heads {
                        for j in 0..m.nope {
                            data.extend(
                                (0..m.kv_lora).map(|c| k_b[(h * m.kv_lora + c) * m.nope + j]),
                            );
                        }
                        data.extend_from_slice(
                            &v_b[h * m.value * m.kv_lora..(h + 1) * m.value * m.kv_lora],
                        );
                    }
                    let mut l = Linear::from_parts(
                        Tensor {
                            shape: vec![m.kv_lora, kv_rows],
                            data,
                        },
                        kv_rows,
                        m.kv_lora,
                        None,
                    )?;
                    l.enable_fast_cpu();
                    l
                };
                Attn::Mla(Box::new(Mla {
                    q_a,
                    q,
                    kv_a: linear(&format!("{p}.attn_kv_a_mqa"), m.kv_lora + m.rope, hidden)?,
                    kv_a_norm: scaled(vector(mapping, &format!("{p}.attn_kv_a_norm.weight"), m.kv_lora)?, cfg.mla_scales.map(|s| s.1)),
                    kv_b,
                }))
            } else if let Some(rw) = rwkv::Rwkv::load(&cfg, mapping, &p, i, &linear)? {
                Attn::Rwkv(Box::new(rw))
            } else if !has_tensor(mapping, &format!("{p}.attn_q.weight"))
                && !has_tensor(mapping, &format!("{p}.attn_qkv.weight"))
            {
                Attn::None
            } else if has_tensor(mapping, &format!("{p}.attn_qkv.weight")) {
                let name = format!("{p}.attn_qkv.weight");
                let bias = format!("{p}.attn_qkv.bias");
                if cfg.arch == "persimmon" {
                    // Persimmon stores the fused projection head-interleaved ([q_h, k_h, v_h] per
                    // head), so the rows cannot be sliced as three contiguous blocks.
                    persimmon_qkv(mapping, &name, &bias, hidden, heads, kv_heads, dk)?
                } else {
                    Attn::Std {
                        q: part(&name, &bias, 0, qr, hidden)?,
                        k: part(&name, &bias, qr, kr, hidden)?,
                        v: part(&name, &bias, qr + kr, vr, hidden)?,
                    }
                }
            } else {
                let unused = || Linear::from_parts(Tensor { shape: vec![1, 1], data: vec![0.0] }, 1, 1, None);
                Attn::Std {
                    q: linear(
                        &format!("{p}.attn_q"),
                        if output_gate { 2 * qr } else { qr },
                        hidden,
                    )?,
                    k: if kv_shared { unused()? } else { linear(&format!("{p}.attn_k"), kr, hidden)? },
                    v: if kv_shared || k_eq_v {
                        unused()?
                    } else if has_tensor(mapping, &format!("{p}.attn_v_exps.weight")) {
                        // MoVA layers compute their values from routed experts
                        Linear::from_parts(Tensor { shape: vec![1, 1], data: vec![0.0] }, 1, 1, None)?
                    } else {
                        linear(&format!("{p}.attn_v"), vr, hidden)?
                    },
                }
            };
            let ffn = if !has_tensor(mapping, &format!("{p}.ffn_up.weight"))
                && !has_tensor(mapping, &format!("{p}.ffn_gate_inp.weight"))
            {
                Ffn::None
            } else if has_tensor(mapping, &format!("{p}.ffn_gate_inp.weight")) && !has_tensor(mapping, &format!("{p}.pre_ffw_norm_2.weight")) {
                let moe = cfg
                    .moe
                    .as_ref()
                    .ok_or("GGUF has expert tensors but no expert metadata")?;
                let expert =
                    |p: &str, suffix: &str, e: usize, eff: usize| -> Result<Expert, String> {
                        // routed experts of a latent MoE work on the latent projection, shared ones on the hidden state
                        let width = if suffix == "exps" { moe.latent.unwrap_or(hidden) } else { hidden };
                        // Qwen3.5 MoE files store gate and up of every expert in one tensor: [gate; up] rows per expert
                        let fused = suffix == "exps" && has_tensor(mapping, &format!("{p}.ffn_gate_up_exps.weight"));
                        let load = |kind: &str, rows: usize, cols: usize| {
                            let bias = format!("{p}.ffn_{kind}_{suffix}.bias");
                            let mut l = if fused && kind != "down" {
                                Linear::load_rows(
                                    mapping,
                                    &format!("{p}.ffn_gate_up_exps.weight"),
                                    None,
                                    e * 2 * rows + if kind == "up" { rows } else { 0 },
                                    rows,
                                    cols,
                                )?
                            } else {
                                Linear::load_rows(
                                    mapping,
                                    &format!("{p}.ffn_{kind}_{suffix}.weight"),
                                    Some(&bias),
                                    e * rows,
                                    rows,
                                    cols,
                                )?
                            };
                            l.enable_fast_cpu();
                            Ok::<_, String>(l)
                        };
                        let gated = fused || has_tensor(mapping, &format!("{p}.ffn_gate_{suffix}.weight"));
                        Ok(Expert {
                            gate: if gated {
                                Some(load("gate", eff, width)?)
                            } else {
                                None
                            },
                            up: load("up", eff, width)?,
                            down: load("down", width, eff)?,
                        })
                    };
                let experts = (0..moe.experts)
                    .map(|e| {
                        // expert parallelism loads only the experts of this rank
                        if let Some(ctx) = &par {
                            if ctx.cfg.expert_parallel
                                && ctx.tp_size() > 1
                                && !crate::parallel::ParCtx::expert_range(moe.experts, ctx.tp_size(), ctx.tp_rank).contains(&e)
                            {
                                return Ok(par_shard::placeholder_expert());
                            }
                        }
                        expert(&p, "exps", e, moe.expert_ff)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let shared = if has_tensor(mapping, &format!("{p}.ffn_up_shexp.weight")) {
                    let gate_name = format!("{p}.ffn_gate_inp_shexp.weight");
                    let gate = if has_tensor(mapping, &gate_name) {
                        {
                            let t = Tensor::load(mapping, &gate_name)?;
                            if t.data.len() != hidden {
                                return Err(format!("invalid shared expert gate {gate_name}"));
                            }
                            Some(t.data)
                        }
                    } else {
                        None
                    };
                    Some((expert(&p, "shexp", 0, moe.shared_ff)?, gate))
                } else {
                    None
                };
                let bias_name = format!("{p}.exp_probs_b.bias");
                let cache = expert_cache.as_ref().and_then(|c| {
                    let parts: Vec<Vec<fllm_gguf_probe::TensorBytes>> = experts
                        .iter()
                        .map(|e| [e.gate.as_ref(), Some(&e.up), Some(&e.down)].into_iter().flatten().filter_map(|l| l.packed_storage().map(|(_, b)| b.clone())).collect())
                        .collect();
                    c.register_layer(i, moe.used, &parts)
                });
                let tid2eid = if has_tensor(mapping, &format!("{p}.ffn_tid2eid.weight")) {
                    let table = Tensor::load(mapping, &format!("{p}.ffn_tid2eid.weight"))?.data;
                    if table.len() % moe.used != 0 || table.iter().any(|v| *v < 0.0 || *v as usize >= moe.experts) {
                        return Err(format!("invalid hash routing table in {p}"));
                    }
                    Some(table.iter().map(|v| *v as u32).collect())
                } else {
                    None
                };
                Ffn::Moe(Box::new(Moe {
                    tid2eid,
                    cache,
                    bias: if has_tensor(mapping, &bias_name) {
                        Some(vector(mapping, &bias_name, moe.experts)?)
                    } else {
                        None
                    },
                    router: linear(&format!("{p}.ffn_gate_inp"), moe.experts, hidden)?,
                    experts,
                    shared,
                    latent: match moe.latent {
                        Some(n) => Some((
                            linear(&format!("{p}.ffn_latent_down"), n, hidden)?,
                            linear(&format!("{p}.ffn_latent_up"), hidden, n)?,
                        )),
                        None => None,
                    },
                    latent_norm: match moe.latent {
                        Some(n) if has_tensor(mapping, &format!("{p}.ffn_latent_norm.weight")) => Some(vector(mapping, &format!("{p}.ffn_latent_norm.weight"), n)?),
                        _ => None,
                    },
                }))
            } else if has_tensor(mapping, &format!("{p}.ffn_gate.weight")) {
                Ffn::Gated {
                    gate: linear(&format!("{p}.ffn_gate"), ff_i, hidden)?,
                    up: linear(&format!("{p}.ffn_up"), ff_i, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, ff_i)?,
                }
            } else if tensor_rows(mapping, &format!("{p}.ffn_up.weight")) == Some(2 * ff_i) {
                // fused [gate; up]
                let name = format!("{p}.ffn_up.weight");
                let bias = format!("{p}.ffn_up.bias");
                Ffn::Gated {
                    gate: part(&name, &bias, 0, ff_i, hidden)?,
                    up: part(&name, &bias, ff_i, ff_i, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, ff_i)?,
                }
            } else {
                Ffn::Plain {
                    up: linear(&format!("{p}.ffn_up"), ff_i, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, ff_i)?,
                }
            };
            let qk = |name: &str, width| -> Result<Option<Vec<f32>>, String> {
                let full = format!("{p}.{name}.weight");
                if has_tensor(mapping, &full) {
                    Ok(Some(vector(mapping, &full, width)?))
                } else {
                    Ok(None)
                }
            };
            // Persimmon's query / key normalisation is a LayerNorm (a bias rides along), unlike the
            // RMSNorm used by every other model that carries a query / key norm.
            let qk_bias = |name: &str, width| -> Result<Option<Vec<f32>>, String> {
                let full = format!("{p}.{name}.bias");
                if has_tensor(mapping, &full) {
                    Ok(Some(vector(mapping, &full, width)?))
                } else {
                    Ok(None)
                }
            };
            let (q_width, k_width) = match cfg.qk_norm {
                QkNorm::Full => (heads * dk, kv_heads * dk),
                _ => (dk, dk),
            };
            let sinks_name = format!("{p}.attn_sinks.weight");
            let sinks = if has_tensor(mapping, &sinks_name) {
                Some(vector(mapping, &sinks_name, heads)?)
            } else {
                None
            };
            let out = match (&attn, &cfg.delta, &cfg.mamba) {
                (Attn::ShortConv(_), _, _) => {
                    linear(&format!("{p}.shortconv_out"), hidden, hidden)?
                }
                (Attn::None, _, _) => Linear::from_parts(
                    Tensor {
                        shape: vec![1, 1],
                        data: vec![0.0],
                    },
                    1,
                    1,
                    None,
                )?,
                (Attn::Linear(_), Some(d), _) => {
                    linear(&format!("{p}.ssm_out"), hidden, d.v_heads * d.v_dim)?
                }
                (Attn::Mamba(_), _, Some(m)) => linear(&format!("{p}.ssm_out"), hidden, m.inner)?,
                (Attn::Rwkv(_), _, _) => linear(&format!("{p}.time_mix_output"), hidden, hidden)?,
                (Attn::Kda(_), _, _) => {
                    let k = cfg.hybrid.kda.as_ref().ok_or("missing KDA configuration")?;
                    linear(&format!("{p}.attn_output"), hidden, k.heads * k.head_dim)?
                }
                (Attn::Mamba1(_), _, _) => {
                    let inner = cfg.mamba1.as_ref().map_or(0, |m| m.inner);
                    linear(&format!("{p}.ssm_out"), hidden, inner)?
                }
                _ => linear(&format!("{p}.attn_output"), hidden, heads * dv)?,
            };
            let gate_rows = tensor_rows(mapping, &format!("{p}.attn_gate.weight"));
            let gate_per_head = !cfg.gate_per_element && gate_rows == Some(heads) && heads != heads * dv;
            let attn_gate = if gate_rows.is_some() && matches!(attn, Attn::Std { .. } | Attn::Mla(_)) {
                Some(linear(&format!("{p}.attn_gate"), if gate_per_head { heads } else { heads * dv }, hidden)?)
            } else {
                None
            };
            let value_moe = if has_tensor(mapping, &format!("{p}.attn_v_exps.weight")) {
                let ve = cfg
                    .value_experts
                    .ok_or("GGUF has value-expert tensors but no attention.value_expert_count")?;
                let bias_name = format!("{p}.attn_v_gate.bias");
                let mut experts = Vec::with_capacity(ve.count);
                for e in 0..ve.count {
                    let mut l = Linear::load_rows(
                        mapping,
                        &format!("{p}.attn_v_exps.weight"),
                        None,
                        e * vr,
                        vr,
                        hidden,
                    )?;
                    l.enable_fast_cpu();
                    experts.push(l);
                }
                Some(ValueMoe {
                    router: Linear::load_rows(
                        mapping,
                        &format!("{p}.attn_v_gate.weight"),
                        None,
                        0,
                        ve.count,
                        hidden,
                    )?,
                    bias: if has_tensor(mapping, &bias_name) {
                        Some(vector(mapping, &bias_name, ve.count)?)
                    } else {
                        None
                    },
                    experts,
                    used: ve.used,
                })
            } else {
                None
            };
            let xielu_name = format!("{p}.ffn_xielu.weight");
            let xielu = if has_tensor(mapping, &xielu_name) {
                let data = Tensor::load(mapping, &xielu_name)?.data;
                Some(
                    <[f32; 4]>::try_from(data.as_slice())
                        .map_err(|_| "invalid xIELU parameters")?,
                )
            } else {
                None
            };
            // gpt-oss names the norm in front of the feed-forward `post_attention_norm` (HF: post_attention_layernorm) and
            // has no `ffn_norm`; Gemma-style models have both, and there it normalises the attention output instead.
            let mut ffn_norm = load_norm(mapping, &format!("{p}.ffn_norm"), hidden)?;
            let mut post_attn_norm = load_norm(mapping, &format!("{p}.post_attention_norm"), hidden)?;
            let post_ffn_norm = load_norm(mapping, &format!("{p}.post_ffw_norm"), hidden)?;
            if ffn_norm.is_none() && post_ffn_norm.is_none() && post_attn_norm.is_some() {
                ffn_norm = post_attn_norm.take();
            }
            let gemma_layer = gemma::load_layer(model, &cfg, &p)?;
            let indexer = match (&cfg.indexer, &cfg.mla) {
                (Some(ic), Some(mc)) if mtp_prefix.is_none() && has_tensor(mapping, &format!("{p}.indexer.attn_k.weight")) => {
                    let q_low = mc.q_lora.ok_or("a sparse-attention indexer needs a low-rank query")?;
                    Some(Box::new(Indexer {
                        q_b: linear(&format!("{p}.indexer.attn_q_b"), ic.heads * ic.dim, q_low)?,
                        k: linear(&format!("{p}.indexer.attn_k"), ic.dim, hidden)?,
                        k_norm: vector(mapping, &format!("{p}.indexer.k_norm.weight"), ic.dim)?,
                        k_norm_bias: if has_tensor(mapping, &format!("{p}.indexer.k_norm.bias")) {
                            vector(mapping, &format!("{p}.indexer.k_norm.bias"), ic.dim)?
                        } else {
                            vec![0.0; ic.dim]
                        },
                        proj: linear(&format!("{p}.indexer.proj"), ic.heads, hidden)?,
                        pool: if ic.kpool > 1 {
                            Some((linear(&format!("{p}.indexer.pool_gate"), ic.dim, hidden)?, vector(mapping, &format!("{p}.indexer.pool_ape.weight"), ic.kpool * ic.dim)?))
                        } else {
                            None
                        },
                    }))
                }
                _ => None,
            };
            // a layer of a model with an indexer that has none of its own reuses the selection of the layer before
            let dsa_shared = indexer.is_none() && cfg.indexer.is_some() && mtp_prefix.is_none() && saw_indexer && matches!(attn, Attn::Mla(_));
            saw_indexer |= indexer.is_some();
            let shortcut = if has_tensor(mapping, &format!("{p}.shortcut_gate_inp.weight")) {
                let moe = cfg.moe.as_ref().ok_or("GGUF has shortcut experts but no expert metadata")?;
                let routed = moe.experts - moe.zero_experts;
                let load = |kind: &str, e: usize, rows: usize, cols: usize| -> Result<Linear, String> {
                    let mut l = Linear::load_rows(mapping, &format!("{p}.shortcut_{kind}_exps.weight"), None, e * rows, rows, cols)?;
                    l.enable_fast_cpu();
                    Ok(l)
                };
                let experts = (0..routed)
                    .map(|e| Ok(Expert { gate: Some(load("gate", e, moe.expert_ff, hidden)?), up: load("up", e, moe.expert_ff, hidden)?, down: load("down", e, hidden, moe.expert_ff)? }))
                    .collect::<Result<Vec<_>, String>>()?;
                Some(Box::new(Moe {
                    router: linear(&format!("{p}.shortcut_gate_inp"), moe.experts, hidden)?,
                    bias: Some(vector(mapping, &format!("{p}.shortcut_exp_probs_b.bias"), moe.experts)?),
                    experts,
                    cache: None,
                    shared: None,
                    latent: None,
                    latent_norm: None,
                    tid2eid: None,
                }))
            } else {
                None
            };
            let takes_shortcut = cfg.arch == "longcat-flash" && i % 2 == 1;
            let mut layer = Layer {
                indexer,
                shortcut,
                takes_shortcut,
                res: if cfg.attn_res.is_some() && has_tensor(mapping, &format!("{p}.attn_res_norm.weight")) {
                    let score = |what: &str| -> Result<Vec<f32>, String> {
                        let norm = vector(mapping, &format!("{p}.{what}_res_norm.weight"), hidden)?;
                        let proj = vector(mapping, &format!("{p}.{what}_res_proj.weight"), hidden)?;
                        Ok(norm.iter().zip(&proj).map(|(n, w)| n * w).collect())
                    };
                    Some(Box::new(LayerRes { attn: score("attn")?, ffn: score("ffn")? }))
                } else {
                    None
                },
                dsa_shared,
                parallel: hybrid::ParallelMixer::load(&cfg, mapping, &p, i, &linear)?,
                zamba: zamba::ZambaCall::load(&cfg, mapping, i, &linear)?,
                attn,
                chunk: cfg.layer_chunk(i),
                output_gate,
                attn_gate,
                gate_per_head,
                value_moe,
                xielu,
                sinks,
                attn_sub_norm: if has_tensor(mapping, &format!("{p}.attn_sub_norm.weight")) {
                    Some(vector(mapping, &format!("{p}.attn_sub_norm.weight"), cfg.layer_heads(i) * cfg.layer_value_dim(i))?)
                } else {
                    None
                },
                ffn_sub_norm: if has_tensor(mapping, &format!("{p}.ffn_sub_norm.weight")) {
                    Some(vector(mapping, &format!("{p}.ffn_sub_norm.weight"), cfg.layer_ff(i))?)
                } else {
                    None
                },
                out,
                ffn,
                attn_norm: load_norm(mapping, &format!("{p}.attn_norm"), hidden)?,
                ffn_norm,
                post_attn_norm,
                post_ffn_norm,
                q_norm: qk("attn_q_norm", q_width)?,
                k_norm: qk("attn_k_norm", k_width)?,
                q_norm_bias: qk_bias("attn_q_norm", q_width)?,
                k_norm_bias: qk_bias("attn_k_norm", k_width)?,
                window: cfg.layer_window(i),
                rope_table: model.rope_index(i),
                kv_shared,
                k_eq_v,
                gemma: gemma_layer,
            };
            if cfg.minimax3.is_some() {
                minimax3::patch_layer(model, &cfg, i, &mut layer)?;
            }
            if let Some(ctx) = &par {
                par_shard::shard_layer(ctx, &mut layer, i, &cfg)?;
            }
            layers.push(layer);
        }
        let head_name = if has_tensor(mapping, "output.weight") {
            "output"
        } else {
            "token_embd"
        };
        let head_name = mtp_prefix
            .map(|p| format!("{p}.shared_head_head"))
            .filter(|p| has_tensor(mapping, &format!("{p}.weight")))
            .unwrap_or_else(|| head_name.to_owned());
        let final_norm_name = mtp_prefix
            .map(|p| format!("{p}.shared_head_norm"))
            .filter(|p| has_tensor(mapping, &format!("{p}.weight")))
            .unwrap_or_else(|| "output_norm".into());
        let embedding_name = mtp_prefix
            .map(|p| format!("{p}.embed_tokens.weight"))
            .filter(|p| has_tensor(mapping, p))
            .unwrap_or_else(|| "token_embd.weight".into());
        let mut embedding = Embedding::load_named(model, &embedding_name)?;
        if let Some(ctx) = &par {
            par_shard::shard_embedding(ctx, &mut embedding)?;
        }
        let head = match &par {
            // the output head lives on the last pipeline stage, cut by vocabulary rows over the tensor-parallel group
            Some(ctx) if !ctx.is_last_stage() => par_shard::dummy_linear(),
            Some(ctx) => par_shard::shard_head(ctx, linear(&head_name, cfg.vocab, hidden)?, cfg.vocab)?,
            None => linear(&head_name, cfg.vocab, hidden)?,
        };
        let cfg = match &par {
            Some(ctx) => par_shard::localize_cfg(ctx, &cfg)?,
            None => cfg,
        };
        let cfg_rope_half = cfg.rope.dims / 2;
        let mrope = match crate::vision::mrope::MropeLayout::from_metadata(meta)? {
            Some(layout) if cfg_rope_half > 0 => {
                layout.check(cfg.rope.dims)?;
                Some(MropeState { axes: layout.axes(cfg_rope_half), axis_tables: std::sync::OnceLock::new() })
            }
            _ => None,
        };
        let mtp = mtp_prefix
            .map(|p| -> Result<MtpProjection, String> {
                Ok(MtpProjection {
                    embedding_norm: vector(mapping, &format!("{p}.enorm.weight"), hidden)?,
                    hidden_norm: vector(mapping, &format!("{p}.hnorm.weight"), hidden)?,
                    projection: linear(&format!("{p}.eh_proj"), hidden, 2 * hidden)?,
                })
            })
            .transpose()?;
        let v4 = if cfg.v4.is_some() { Some(Box::new(deepseek4::V4Model::load(model, &cfg)?)) } else { None };
        let hc = if cfg.hc.is_some() { Some(Box::new(hc::HcModel::load(mapping, &cfg)?)) } else { None };
        let inkling = if cfg.inkling.is_some() { Some(Box::new(inkling::InklingModel::load(model, &cfg)?)) } else { None };
        let minimax3 = if cfg.minimax3.is_some() { Some(Box::new(minimax3::Minimax3Model::load(model, &cfg)?)) } else { None };
        let qwen4 = if cfg.qwen4.is_some() { Some(Box::new(qwen4::Qwen4Model::load(model, &cfg)?)) } else { None };
        let output_res = if cfg.attn_res.is_some() && has_tensor(mapping, "output_res_norm.weight") {
            let norm = vector(mapping, "output_res_norm.weight", hidden)?;
            let proj = vector(mapping, "output_res_proj.weight", hidden)?;
            Some(norm.iter().zip(&proj).map(|(n, w)| n * w).collect())
        } else {
            None
        };
        let mut decoder = Self {
            mtp,
            expert_cache,
            profile: std::env::var_os("FLLM_RUST_PROFILE").is_some(),
            embedding,
            layers,
            head,
            final_norm: load_norm(mapping, &final_norm_name, hidden)?
                .ok_or("missing output_norm.weight")?,
            ple: gemma::load_global(model, &cfg)?,
            altup: gemma::load_altup(model, &cfg)?,
            rope: model.rope_shared(),
            groups,
            stride: model.table_stride(),
            block_size: model.block_size(),
            pool_blocks: model.pool_blocks(),
            kv_kind: model.kv_kind(),
            mla_latent: model.mla_latent(),
            kvl: {
                // a pipeline stage lays out the cache of its own layers only
                let mut kvl = match &par {
                    Some(ctx) => model.kv_pool_layout_for(&par_shard::stage_local_kv(ctx, &cfg)?),
                    None => model.kv_pool_layout_for(&cfg),
                };
                if mtp_prefix.is_some() {
                    // the MTP block owns one layer of cache
                    kvl.key_offsets.truncate(1);
                    kvl.value_offsets.truncate(1);
                    kvl.key_words.truncate(1);
                    kvl.value_words.truncate(1);
                    kvl.key_total = kvl.key_words[0];
                    kvl.value_total = kvl.value_words[0];
                }
                kvl
            },
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(model.threads().max(1) as usize)
                .build()
                .map_err(|e| e.to_string())?,
            prefill_pool: if model.prefill_threads() > model.threads() {
                rayon::ThreadPoolBuilder::new().num_threads(model.prefill_threads() as usize).build().ok()
            } else {
                None
            },
            position_table: if has_tensor(mapping, "position_embd.weight") {
                Some(Tensor::load(mapping, "position_embd.weight")?.data)
            } else {
                None
            },
            embd_norm: load_norm(mapping, "token_embd_norm", hidden)?,
            states: std::sync::Mutex::new(StateStore::default()),
            device_states: std::sync::Mutex::new(DeviceStates::default()),
            state_capacity,
            cfg,
            backend: None,
            backend_stream: None,
            backend_stream2: None,
            backend_weights: None,
            backend_kv: None,
            backend_scratch: None,
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_ok: std::sync::OnceLock::new(),
            graph_failed: std::sync::atomic::AtomicBool::new(false),
            act_source_bf16: mapping.index().tensors.iter().any(|t| matches!(t.name.as_str(), "token_embd.weight" | "output.weight") && t.type_id == 30),
            par,
            embed_stash: Default::default(),
            tree_mask: None,
            tree_prefix_len: 0,
            tree_num_nodes: 0,
            mrope,
            row_positions3: std::sync::Mutex::new(None),
            row_deepstack: std::sync::Mutex::new(None),
            row_visible_lengths: std::sync::Mutex::new(None),
            v4,
            inkling,
            output_res,
            minimax3,
            qwen4,
            hc,
            row_ids: std::sync::Mutex::new(Vec::new()),
            dsa: DsaStore::default(),
            dsa_dense: std::env::var("FLLM_DSA").is_ok_and(|v| v.eq_ignore_ascii_case("dense")),
            dsa_exact_on_device: std::env::var("FLLM_DSA").is_ok_and(|v| v.eq_ignore_ascii_case("exact") || v.eq_ignore_ascii_case("host")),
            dsa_device: false,
            m3_device: false,
            m3_run: false,
            ink_device: false,
            hc_device: false,
            v4_device: false,
        };
        decoder.set_device(device)?;
        decoder.plan_expert_cache(mapping);
        Ok(decoder)
    }

    /// The expert cache of a MoE model (statistics, plan); `None` for other models or with `--expert-cache-mb off`.
    pub fn expert_cache(&self) -> Option<&crate::expert_cache::ExpertCache> {
        self.expert_cache.as_ref()
    }

    /// Fix the budget of the expert cache once it is known which MoE layers keep their experts in host memory, and report the plan.
    fn plan_expert_cache(&self, mapping: &MappedGguf) {
        let Some(cache) = self.expert_cache.as_ref() else { return };
        let mut tables = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let Ffn::Moe(moe) = &layer.ffn else { continue };
            let Some(table) = moe.cache.as_ref() else { continue };
            tables += 1;
            // experts that were uploaded to the device are not host memory, and the device layer never routes on the host
            if self.backend_weights.as_ref().and_then(|w| w.layers.get(i)).is_some_and(|l| l.moe.is_some()) {
                table.mark_device_resident();
            }
        }
        if tables == 0 {
            return;
        }
        let (_, dense) = crate::expert_cache::split_bytes(mapping.index());
        // the dense weights of a model on an accelerator are on the device; the processor keeps all of them in host memory
        let dense_host = if self.backend.is_some() { 0 } else { dense };
        let plan = cache.finalize(dense_host);
        if plan.layers > 0 {
            eprintln!("[fllm-rs] {}", plan.describe());
            if plan.streaming && dense_host > plan.budget.saturating_add(plan.expert_bytes) / 2 {
                eprintln!("[fllm-rs] warning: the dense weights ({:.1} GiB) are large for this machine's free memory; every token will page them as well", dense_host as f64 / (1u64 << 30) as f64);
            }
        }
    }

    pub fn set_tree_mask(&mut self, mask: Option<Vec<bool>>, prefix_len: usize, num_nodes: usize) {
        self.tree_mask = mask;
        self.tree_prefix_len = prefix_len;
        self.tree_num_nodes = num_nodes;
    }

    /// Number of decoder layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The layers this decoder runs: its pipeline stage, or all of them when it is not part of a parallel group.
    fn stage_range(&self) -> std::ops::Range<usize> {
        self.par.as_ref().map_or(0..self.layers.len(), |p| p.layer_range(self.layers.len()))
    }

    /// Sum the partial result of a row-parallel projection over the tensor-parallel group: the buffer goes down to the host, the
    /// group's all-reduce (NCCL for CUDA when present, else host memory) adds the partial sums, and the sum goes back up.
    fn device_tp_reduce(&self, stream: &mut dyn crate::backend::Stream, buf: &mut crate::backend::DeviceBuffer, len: usize) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let mut host = vec![0.0f32; len];
        stream.download(cast_f32_to_u8_mut(&mut host), buf).map_err(err)?;
        self.par_tp_reduce(&mut host)?;
        stream.upload(buf, cast_f32_to_u8(&host)).map_err(err)
    }

    /// First layer whose cache is the first layer of this decoder's device KV pool: the start of the stage when the model is split into
    /// pipeline stages (each stage allocates cache for its own layers only), else 0.
    fn kv_base(&self) -> usize {
        self.par.as_ref().filter(|p| p.pp_size() > 1).map_or(0, |p| p.layer_range(self.layers.len()).start)
    }

    /// Whether this decoder is one rank of a parallel group (the other ranks hold the rest of the model).
    pub fn is_parallel(&self) -> bool {
        self.par.is_some()
    }

    /// Whether the model carries multimodal rotary sections (`fllm.rope.mrope_sections`).
    pub fn supports_mrope(&self) -> bool {
        self.mrope.is_some()
    }

    /// Three-axis position ids (time, height, width) for the rows of the next forward pass, or `None` for the ordinary positions. The
    /// caller clears them after the pass. Rows of text carry the same id on every axis.
    pub fn set_row_positions3(&self, positions: Option<Vec<[u32; 3]>>) {
        *self.row_positions3.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = positions;
    }

    /// DeepStack features for the next forward pass (see `row_deepstack`); the caller clears them afterwards.
    pub fn set_row_deepstack(&self, features: Option<DeepstackLevels>) {
        *self.row_deepstack.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = features.map(Arc::new);
    }

    /// Visible lengths per row for the next forward pass (see `row_visible_lengths`); the caller clears them afterwards.
    pub fn set_row_visible_lengths(&self, lengths: Option<Vec<usize>>) {
        *self.row_visible_lengths.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = lengths;
    }

    fn row_visible_lengths(&self) -> Option<Vec<usize>> {
        self.row_visible_lengths.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    fn row_deepstack(&self) -> Option<Arc<DeepstackLevels>> {
        self.row_deepstack.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// The DeepStack features after the checks of a batch of `rows` rows.
    fn checked_deepstack(&self, rows: usize) -> Result<Option<Arc<DeepstackLevels>>, String> {
        let Some(levels) = self.row_deepstack() else { return Ok(None) };
        if levels.len() > self.layers.len()
            || levels.iter().flatten().any(|(r, v)| *r >= rows || v.len() != self.cfg.hidden || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid DeepStack features".into());
        }
        Ok(Some(levels))
    }

    fn row_positions3(&self) -> Option<Vec<[u32; 3]>> {
        self.row_positions3.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// The batch's three-axis positions after checks: the model must have M-RoPE sections, one entry per row, ids inside the rotary tables.
    fn checked_positions3(&self, rows: usize) -> Result<Option<Vec<[u32; 3]>>, String> {
        let Some(p3) = self.row_positions3() else { return Ok(None) };
        if self.mrope.is_none() {
            return Err("multimodal positions were given to a model without M-RoPE sections".into());
        }
        let max_position = self.max_position();
        if p3.len() != rows || p3.iter().flatten().any(|&p| p as usize >= max_position) {
            return Err("invalid multimodal position ids".into());
        }
        Ok(Some(p3))
    }

    /// Per-axis rotary tables of the device path (see `MropeState`).
    fn mrope_axis_tables(&self) -> Option<&AxisTables> {
        let state = self.mrope.as_ref()?;
        let table = &self.rope[0];
        Some(state.axis_tables.get_or_init(|| {
            (0..3u8)
                .map(|axis| {
                    let (mut cos, mut sin) = (table.cos.clone(), table.sin.clone());
                    for (i, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
                        if state.axes[i % table.half] != axis {
                            (*c, *s) = (1.0, 0.0);
                        }
                    }
                    (Arc::new(cos), Arc::new(sin))
                })
                .collect()
        }))
    }

    fn norm(&self, x: &mut [f32], norm: &Norm) {
        match self.cfg.norm {
            NormKind::Rms => grouped_rms(x, &norm.weight, self.cfg.eps, self.groups),
            NormKind::Layer => layer_norm(x, &norm.weight, norm.bias.as_deref(), self.cfg.eps),
        }
    }

    /// Norm behind the attention / feed-forward branch (its epsilon can differ from the main one).
    fn post_norm(&self, x: &mut [f32], norm: &Norm) {
        match self.cfg.norm {
            NormKind::Rms => grouped_rms(x, &norm.weight, self.cfg.post_eps, self.groups),
            NormKind::Layer => layer_norm(x, &norm.weight, norm.bias.as_deref(), self.cfg.post_eps),
        }
    }

    /// (key, value) cache formats.
    fn kv_formats(&self) -> (crate::kv_format::KvFormat, crate::kv_format::KvFormat) {
        crate::kv_format::unpack_kind(self.kv_kind)
            .unwrap_or((crate::kv_format::KvFormat::F16, crate::kv_format::KvFormat::F16))
    }
    /// Key cache region of the layer whose cache `layer` uses: `(first word, kv heads, head width)`.
    fn key_region(&self, layer: usize) -> (usize, usize, usize) {
        let owner = crate::kv_geometry::KvPoolLayout::owner(&self.cfg, layer);
        (self.kvl.key_offsets[owner], self.cfg.layer_kv_heads(owner), self.cfg.layer_head_dim(owner))
    }
    /// Value cache region, see `key_region`.
    fn value_region(&self, layer: usize) -> (usize, usize, usize) {
        let owner = crate::kv_geometry::KvPoolLayout::owner(&self.cfg, layer);
        (self.kvl.value_offsets[owner], self.cfg.layer_kv_heads(owner), self.cfg.layer_value_dim(owner))
    }
    /// Write one slot (`kv_heads x dim` values) of the region `(offset, kv_heads, dim)`.
    fn write_cache(
        &self,
        cache: &mut [u16],
        region: (usize, usize, usize),
        slot: usize,
        row: &[f32],
        format: crate::kv_format::KvFormat,
    ) {
        let (offset, kv_heads, dim) = region;
        if format == crate::kv_format::KvFormat::Q8_0 {
            // the original q8_0 writer (also used by the unit tests below)
            let kv = kv_heads * dim;
            let group = if dim.is_multiple_of(32) { 32 } else { dim };
            let slots = self.pool_blocks * self.block_size;
            let words = format.layer_words(slots, kv_heads, dim);
            let q_words = (slots * kv).div_ceil(2);
            let scale_words = slots * (kv / group);
            let (q_cache, cache_scales) = cache[offset..offset + words].split_at_mut(q_words);
            write_q8_cache_row(q_cache, &mut cache_scales[..scale_words], slot, kv, group, row);
            return;
        }
        let layout = crate::kv_format::KvLayout::new(format, self.pool_blocks * self.block_size, kv_heads, dim);
        layout.encode_slot(&mut cache[offset..offset + layout.words], slot, row);
    }
    /// Decode one cache slot (`kv_heads x dim` values) of the region `(offset, kv_heads, dim)` into `out`.
    fn read_slot(
        &self,
        cache: &[u16],
        region: (usize, usize, usize),
        slot: usize,
        format: crate::kv_format::KvFormat,
        out: &mut [f32],
    ) {
        let (offset, kv_heads, dim) = region;
        let layout = crate::kv_format::KvLayout::new(format, self.pool_blocks * self.block_size, kv_heads, dim);
        layout.decode_slot(&cache[offset..offset + layout.words], slot, out);
    }
    /// Pick the compute device for `device` (a `--device` value, see `devices`) and load the model onto it.
    ///
    /// An explicit device that cannot be created is an error; one that cannot run this model leaves it on the CPU. With `auto` every device of the chain (best first, ending in the CPU) is
    /// tried in turn: a device that cannot be created, or that cannot run this model, is reported and skipped.
    pub fn set_device(&mut self, device: &str) -> Result<(), String> {
        let automatic = crate::devices::is_auto(device);
        let mut last_error = String::from("no usable compute device");
        let mut skipped = false;
        for (backend, index) in crate::devices::device_candidates(device)? {
            if backend == "cpu" {
                if skipped {
                    eprintln!("[fllm-rs] --device auto: no accelerator could take the model; running on the CPU");
                }
                return Ok(());
            }
            match self.attach_accelerator(&backend, index) {
                Ok(()) => return Ok(()),
                Err(error) if automatic => {
                    eprintln!(
                        "[fllm-rs] --device auto: {} is not usable ({error}); trying the next device",
                        crate::devices::describe_choice(&backend, index)
                    );
                    last_error = error;
                    skipped = true;
                }
                // an NPU is experimental and often not usable (no runtime, no driver, a model it cannot take): the server must not end
                // because of it, so the model runs on the CPU and the reason is logged and shown on the Hardware page
                Err(error) if backend == "npu" => {
                    let message = crate::backend::npu::status::record_fallback(&error);
                    eprintln!("[fllm-rs] --device npu: {message} (details: {error})");
                    return Ok(());
                }
                // the device works but has no kernels for this model: run it on the CPU rather than refuse to start
                Err(error) if error.contains("cannot run on the") => {
                    eprintln!("[fllm-rs] --device {device}: {error}; running on the CPU instead");
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error)
    }

    /// Whether an accelerator backend (not the host CPU path) runs the model.
    pub fn uses_accelerator(&self) -> bool {
        self.backend.is_some()
    }

    /// Work the accelerator path leaves to the host reference code: `(whole layers, feed-forward blocks of otherwise device layers)`.
    pub fn host_fallbacks(&self) -> (usize, usize) {
        self.backend_weights.as_ref().map_or((0, 0), |w| {
            (
                w.layers.iter().filter(|l| l.host_layer).count(),
                w.layers.iter().filter(|l| !l.host_layer && l.host_ffn).count(),
            )
        })
    }

    /// Create the backend `selected_backend` (number `selected_dev_id` within that backend) and move the model onto it.
    fn attach_accelerator(&mut self, selected_backend: &str, selected_dev_id: usize) -> Result<(), String> {
        // `wgpu-<api>`: the wgpu implementation of that API even where a native backend exists
        let wgpu_only = selected_backend.strip_prefix("wgpu-");
        // native backends (features vulkan, metal, hip, sycl, npu-*, tpu-pjrt); `Ok(None)` when `selected_backend` is none of them
        if wgpu_only.is_none() {
            match crate::backend::create_native_backend(selected_backend, selected_dev_id) {
                Ok(Some(backend)) => {
                    self.init_backend(backend)?;
                    eprintln!("[fllm-rs] Unified accelerator backend enabled: {selected_backend}:{selected_dev_id} (native)");
                    return Ok(());
                }
                Ok(None) => {}
                Err(error) => {
                    // the wgpu backend serves vulkan and metal too when it is compiled in
                    if !(cfg!(feature = "rust-gpu") && matches!(selected_backend, "vulkan" | "metal")) {
                        return Err(error);
                    }
                    eprintln!("[fllm-rs] native {selected_backend} backend unavailable ({error}); falling back to wgpu");
                }
            }
        }
        #[cfg(feature = "cuda")]
        if selected_backend == "cuda" {
            let backend: Arc<dyn crate::backend::Backend> = Arc::new(
                crate::backend::cuda::CudaBackend::new(selected_dev_id).map_err(|e| e.to_string())?,
            );
            self.init_backend(backend)?;
            eprintln!("[fllm-rs] Unified accelerator backend enabled: cuda:{selected_dev_id}");
            return Ok(());
        }
        let wgpu_api = wgpu_only.unwrap_or(selected_backend);
        if matches!(wgpu_api, "vulkan" | "metal" | "dx12" | "opengl" | "gl" | "gpu" | "wgpu") {
            #[cfg(feature = "rust-gpu")]
            {
                let gpu_backend = crate::backend::wgpu_backend::WgpuBackend::new(wgpu_api, selected_dev_id)
                    .map_err(|e| e.to_string())?;
                let backend: Arc<dyn crate::backend::Backend> = Arc::new(gpu_backend);
                self.init_backend(backend)?;
                eprintln!("[fllm-rs] Unified accelerator backend enabled: {wgpu_api}:{selected_dev_id} (wgpu)");
                return Ok(());
            }
            #[cfg(not(feature = "rust-gpu"))]
            return Err(format!(
                "the {wgpu_api} backend needs a build with --features rust-gpu (wgpu) or --features {} (native) for GPU projections",
                if wgpu_api == "vulkan" || wgpu_api == "metal" { wgpu_api } else { "rust-gpu" }
            ));
        }
        Err(format!(
            "the {selected_backend} backend is not compiled into this binary (compiled: {})",
            crate::devices::compiled_backends().join(", ")
        ))
    }

    /// Model features `forward_backend` does not implement. The device path refuses such
    /// models instead of silently computing something else.
    fn backend_unsupported(&self, layer_norm_ok: bool, grouped_rms_norm_ok: bool) -> Vec<&'static str> {
        let cfg = &self.cfg;
        let mut missing = Vec::new();
        if matches!(cfg.norm, NormKind::Layer) && !layer_norm_ok {
            missing.push("LayerNorm");
        }
        if self.groups != 1 && !grouped_rms_norm_ok {
            missing.push("grouped RMSNorm");
        }
        missing
    }

    /// The largest position the rotary tables cover; a model without rotary embeddings (NoPE latent attention) has empty tables, and the
    /// block table bounds the positions instead.
    fn max_position(&self) -> usize {
        match self.rope[0].half {
            0 => self.stride * self.block_size,
            half => self.rope[0].cos.len() / half,
        }
    }

    /// The hyper-connection parameters of the model (Hy4, GLM-5-Next and DeepSeek-V4 have them).
    fn hc_params(&self) -> Option<HcParams> {
        if let Some(h) = self.cfg.hc.as_ref() {
            return Some(HcParams { hc_mult: h.hc_mult, eps: h.eps, iters: h.iters, magnitude: h.magnitude });
        }
        let v = self.cfg.v4.as_ref()?;
        Some(HcParams { hc_mult: v.hc_mult, eps: v.hc_eps, iters: v.hc_iters, magnitude: None })
    }

    /// Mode of the hyper-connection mixing of the blocks of this model.
    fn hc_block_mode(&self) -> Option<crate::backend::HcMode> {
        let h = self.hc_params()?;
        Some(match h.magnitude {
            Some(magnitude) => crate::backend::HcMode::Independent { magnitude },
            None => crate::backend::HcMode::Sinkhorn { iters: h.iters },
        })
    }

    /// Start of a hyper-connection block on the device: normalises the streams, mixes them down to the block input (written to `x_buf`, the
    /// residual buffer the layer's mixer or feed-forward adds its output to) and keeps the weights for `hc_leave`.
    #[allow(clippy::too_many_arguments)]
    fn hc_enter(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        weights: &DeviceHc,
        ones: &crate::backend::DeviceBuffer,
        streams: &crate::backend::DeviceBuffer,
        x_buf: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<HcBlock, String> {
        let (cfg, mode) = (&self.cfg, self.hc_block_mode().ok_or("hyper-connection mode")?);
        let h = self.hc_params().ok_or("hyper-connection configuration missing")?;
        let (hc, hidden) = (h.hc_mult, cfg.hidden);
        let err = |e: crate::backend::BackendError| e.to_string();
        let mut flat = scratch.acquire(rows * hc * hidden * 4).map_err(err)?;
        stream.rms_norm(&mut flat, streams, ones, cfg.eps).map_err(err)?;
        let mut mixes = scratch.acquire(rows * mode.mix_width(hc) * 4).map_err(err)?;
        stream.quantized_linear(&mut mixes, &flat, &weights.mix, weights.mix.bias.as_ref()).map_err(err)?;
        scratch.recycle(flat);
        let sinkhorn = matches!(mode, crate::backend::HcMode::Sinkhorn { .. });
        let mut post = scratch.acquire(rows * hc * 4).map_err(err)?;
        let mut comb = scratch.acquire(if sinkhorn { rows * hc * hc * 4 } else { 16 }).map_err(err)?;
        stream.hc_mix(x_buf, &mut post, &mut comb, streams, Some(&mixes), Some(&weights.base), Some(&weights.scale), rows, hc, hidden, mode, h.eps).map_err(err)?;
        scratch.recycle(mixes);
        let mut base = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.copy_buffer(&mut base, x_buf).map_err(err)?;
        let comb = if sinkhorn {
            Some(comb)
        } else {
            scratch.recycle(comb);
            None
        };
        Ok(HcBlock { post, comb, base })
    }

    /// End of a hyper-connection block: `x_buf` holds the block input plus the branch output, the difference goes into the streams.
    #[allow(clippy::too_many_arguments)]
    fn hc_leave(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        streams: &mut crate::backend::DeviceBuffer,
        x_buf: &mut crate::backend::DeviceBuffer,
        block: HcBlock,
        rows: usize,
        layer: usize,
        after_ffn: bool,
    ) -> Result<(), String> {
        let h = self.hc_params().ok_or("hyper-connection configuration missing")?;
        let err = |e: crate::backend::BackendError| e.to_string();
        let dump = std::env::var_os("FLLM_DUMP_LAYERS").is_some();
        stream.residual_add(x_buf, &block.base, -1.0).map_err(err)?;
        if dump && !after_ffn {
            let mut host = vec![0.0f32; rows * self.cfg.hidden];
            stream.download(cast_f32_to_u8_mut(&mut host), x_buf).map_err(err)?;
            dump_layer("branch", layer, &host);
        }
        stream.hc_combine(streams, &block.post, block.comb.as_ref(), x_buf, rows, h.hc_mult, self.cfg.hidden).map_err(err)?;
        if dump && after_ffn {
            let mut host = vec![0.0f32; rows * h.hc_mult * self.cfg.hidden];
            stream.download(cast_f32_to_u8_mut(&mut host), streams).map_err(err)?;
            dump_layer("ffn", layer, &host);
        }
        scratch.recycle(block.post);
        scratch.recycle(block.base);
        if let Some(comb) = block.comb {
            scratch.recycle(comb);
        }
        Ok(())
    }

    /// The final collapse of the streams into the residual buffer `x_buf`.
    #[allow(clippy::too_many_arguments)]
    fn hc_head(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        model: &DeviceHcModel,
        streams: &crate::backend::DeviceBuffer,
        x_buf: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        use crate::backend::HcMode;
        let h = self.hc_params().ok_or("hyper-connection configuration missing")?;
        let (hc, hidden) = (h.hc_mult, self.cfg.hidden);
        let err = |e: crate::backend::BackendError| e.to_string();
        let mut post = scratch.acquire(rows * hc * 4).map_err(err)?;
        let mut comb = scratch.acquire(16).map_err(err)?;
        match &model.head {
            Some(head) => {
                let mut flat = scratch.acquire(rows * hc * hidden * 4).map_err(err)?;
                stream.rms_norm(&mut flat, streams, &model.ones, self.cfg.eps).map_err(err)?;
                let mut mixes = scratch.acquire(rows * hc * 4).map_err(err)?;
                stream.quantized_linear(&mut mixes, &flat, &head.mix, head.mix.bias.as_ref()).map_err(err)?;
                stream.hc_mix(x_buf, &mut post, &mut comb, streams, Some(&mixes), Some(&head.base), Some(&head.scale), rows, hc, hidden, HcMode::Head, h.eps).map_err(err)?;
                scratch.recycle(flat);
                scratch.recycle(mixes);
            }
            None => stream.hc_mix(x_buf, &mut post, &mut comb, streams, None, None, None, rows, hc, hidden, HcMode::Mean, h.eps).map_err(err)?,
        }
        scratch.recycle(post);
        scratch.recycle(comb);
        Ok(())
    }

    /// Parameters of the lightning indexer of the sparse-attention layers (`None` without one).
    fn dsa_params(&self) -> Option<crate::backend::DsaParams> {
        let ic = self.cfg.indexer.as_ref()?;
        Some(crate::backend::DsaParams {
            heads: ic.heads,
            dim: ic.dim,
            topk: ic.topk,
            stride: self.stride,
            block_size: self.block_size,
            scale: (ic.dim as f32).powf(-0.5),
            head_scale: (ic.heads as f32).powf(-0.5),
            kpool: ic.kpool.max(1),
            always_tail: ic.always_tail,
        })
    }

    /// Can the accelerator run the indexer and the sparse attention of this model? Plain indexers over the expanded MLA cache only (pooled
    /// indexers and the tail rotation of Hy4 have no device kernel); `FLLM_DSA=host` keeps the layers on the host.
    fn dsa_device_possible(&self, backend: &dyn crate::backend::Backend) -> bool {
        let (Some(ic), Some(mc), Some(params)) = (self.cfg.indexer.as_ref(), self.cfg.mla.as_ref(), self.dsa_params()) else { return false };
        if self.dsa_dense || std::env::var("FLLM_DSA").is_ok_and(|v| v.eq_ignore_ascii_case("host")) || self.mla_latent || self.cfg.kv_heads != self.cfg.heads {
            return false;
        }
        let plain = self.layers.iter().all(|l| l.indexer.as_deref().is_none_or(|ix| ix.pool.is_some() == (ic.kpool > 1)) && (l.indexer.is_none() || l.window.is_none()));
        let q_low = self.layers.iter().all(|l| l.indexer.is_none() || matches!(&l.attn, Attn::Mla(m) if m.q_a.is_some()));
        plain && q_low && backend.create_stream().is_ok_and(|s| s.supports_dsa(&params, mc.nope + mc.rope, self.kv_kind) && (!ic.rope_last || s.supports_rotate_heads()))
    }

    /// Why layer `i` cannot run on the device (`None`: it can). Such a layer is computed by the host reference code
    /// (`host_layers`) between device layers: the residual stream is downloaded, the layer runs on the CPU with the host KV
    /// cache, and the stream goes back up. `gate_ok` / `arch_ok` / `head_ok`: the stream has the attention-gate / element-wise /
    /// per-head gate and value-scale kernels.
    #[allow(clippy::too_many_arguments)]
    fn layer_host_reason(&self, i: usize, layer: &Layer, gate_ok: bool, arch_ok: bool, head_ok: bool, value_ok: bool, conv_ok: bool, dsa_ok: bool, hc_ok: bool, rwkv_ok: bool, kda_ok: bool, mamba_ok: bool, zamba_ok: bool) -> Option<&'static str> {
        let cfg = &self.cfg;
        if cfg.v4.is_some() {
            return if self.v4_device { None } else { Some("DeepSeek-V4 hyper-connections and compressed attention (the backend has no kernels for them, or FLLM_V4=host)") };
        }
        if cfg.inkling.is_some() && !self.ink_device {
            // the host pass of this model is a pass of its own over all layers: one layer that needs it sends them all there
            return Some("Inkling convolutions and relative position bias (no kernels for this backend or shape, or FLLM_INKLING=host)");
        }
        if cfg.minimax3.is_some() && !self.m3_run {
            // the host pass of this model is a pass of its own over all layers: one layer that needs it sends them all there
            return Some("MiniMax-M3 block-sparse attention (no kernels for this backend or shape, FLLM_M3=host, or an indexer narrower than the rotary part)");
        }
        if cfg.hc.is_some() && !hc_ok {
            return Some("hyper-connections (the backend has no kernels for them, or FLLM_HC=host)");
        }
        if cfg.attn_res.is_some() {
            return Some("attention residual blocks");
        }
        if cfg.qwen4.is_some() {
            return Some("Qwen3.8-Flash-Next streams, n-gram embeddings and token indexer");
        }
        // the device pass has no kernels for the Gemma 3n / Gemma 4 extras. They carry state from layer to layer (per-layer inputs,
        // AltUp streams) that only the host pass keeps, so one such layer sends every layer to the host.
        if let Some(gap) = std::iter::once(layer).chain(&self.layers).find_map(|l| l.gemma.as_ref().and_then(|g| g.device_gap())) {
            return Some(gap);
        }
        if layer.parallel.is_some() && !(mamba_ok && layer.post_attn_norm.is_none() && matches!(layer.attn, Attn::Std { .. })) {
            // without the Mamba-2 kernels (or with a norm behind the summed branches) the layer is computed by the host reference code
            return Some("a Mamba-2 branch next to the attention");
        }
        if layer.zamba.is_some() && !zamba_ok {
            return Some("a shared transformer block in front of the Mamba-2 mixer");
        }
        match &layer.attn {
            Attn::Std { .. } => {
                // the element-wise gate kernels would read a gate with one value per head (Laguna, Step-3.5) as garbage: that gate and
                // the value scale of MiMo-V2 have their own device operations (`head_gate_mul`, `scale_inplace`)
                if layer.gate_per_head && !head_ok {
                    return Some("a per-head attention output gate");
                }
                if cfg.value_scale != 1.0 && !(head_ok && value_ok) {
                    return Some("a value scale");
                }
                // routed value experts (MoVA) are computed by the host reference inside an otherwise device layer
                let gated = layer.output_gate || (layer.attn_gate.is_some() && !layer.gate_per_head);
                if gated && (!arch_ok || !gate_ok) {
                    return Some("an attention output gate");
                }
                if cfg.qkv_clip.is_some() && (!arch_ok || layer.output_gate) {
                    return Some("QKV clipping");
                }
                if cfg.attn_temperature.is_some() && (!cfg.temperature_nope_only || !cfg.layer_rotates(i)) {
                    return Some("attention temperature scaling");
                }
                None
            }
            // the time-mix and channel-mix blocks run on the device when the stream has the RWKV operations for the shape
            Attn::Rwkv(_) if rwkv_ok => None,
            Attn::Rwkv(_) => Some("an RWKV time-mix / channel-mix layer (no device operations for this backend or shape)"),
            // Recurrent mixers keep their state on the host (or in their own device buffers) and handle the residual add themselves.
            Attn::Linear(_) | Attn::Mamba(_) | Attn::Mamba1(_) => {
                if cfg.parallel_residual {
                    Some("parallel residual with a recurrent mixer")
                } else {
                    None
                }
            }
            Attn::None => None,
            Attn::Kda(_) => {
                if kda_ok && !cfg.parallel_residual {
                    None
                } else {
                    Some("Kimi Delta Attention (no device operations for this backend or shape)")
                }
            }
            Attn::Mla(_) => {
                if layer.attn_gate.is_some() {
                    // the gate projection is computed from the normed input like the separate gates of the standard layers; the product with the
                    // attention output is the element-wise sigmoid gate (`sigmoid_gate_mul`) or the per-head gate (`head_gate_mul`)
                    let device_gate = if layer.gate_per_head { head_ok } else { gate_ok && matches!(cfg.attn_gate, crate::config::AttnGate::Sigmoid) };
                    if !device_gate {
                        return Some("a latent attention output gate");
                    }
                }
                if (layer.indexer.is_some() || layer.dsa_shared) && !self.dsa_dense && !dsa_ok {
                    if self.dsa_exact_on_device {
                        return Some("the sparse-attention indexer (host: FLLM_DSA=host, or no indexer kernels for this backend, shape or cache format)");
                    }
                    static NOTE: std::sync::Once = std::sync::Once::new();
                    NOTE.call_once(|| {
                        eprintln!(
                            "[fllm-rs] this model has a sparse-attention indexer; the device layers attend to every token, which equals the sparse result up to {} tokens of context. FLLM_DSA=exact runs those layers on the host instead",
                            cfg.indexer.as_ref().map_or(0, |i| i.topk)
                        );
                    });
                }
                if arch_ok && cfg.mla.is_some() && (cfg.kv_heads == cfg.heads || self.mla_latent) && !cfg.parallel_residual {
                    None
                } else {
                    Some("multi-head latent attention")
                }
            }
            Attn::ShortConv(_) => {
                if conv_ok && !cfg.parallel_residual {
                    None
                } else {
                    Some("the ShortConv mixer")
                }
            }
        }
    }

    /// Why the feed-forward of `layer` has to use the host reference code although the mixer runs on the device.
    fn ffn_host_reason(&self, layer: &Layer, arch_ok: bool, clamp_ok: bool) -> Option<&'static str> {
        let cfg = &self.cfg;
        let native = |a: Activation| matches!(a, Activation::Silu | Activation::GeluTanh) || arch_ok;
        // the clamped SwiGLU (Hy4, GLM-5-Next) and SiTU (Kimi K3) have no device kernel; the device GLU would apply the plain SwiGLU
        if !clamp_ok && (cfg.situ.is_some() || cfg.moe.as_ref().is_some_and(|m| m.swiglu_clamp.is_some())) && matches!(layer.ffn, Ffn::Moe(_) | Ffn::Gated { .. }) {
            return Some("a clamped SwiGLU or SiTU activation (the backend has no kernel for it)");
        }
        match &layer.ffn {
            // the all-to-all combine of expert parallelism exchanges tokens between the ranks on the host (the other combines run the
            // experts on the device and add the partial results of the group)
            Ffn::Moe(_) if self.par.as_ref().is_some_and(|p| p.tp_size() > 1 && p.cfg.expert_parallel && p.cfg.ep_dispatch == crate::parallel::EpDispatch::AllToAll) => {
                Some("all-to-all expert dispatch of a tensor-parallel group")
            }
            Ffn::Moe(moe) => {
                if moe.latent.is_some() || cfg.moe.as_ref().is_some_and(|m| m.latent.is_some()) {
                    Some("latent-space expert projections")
                } else {
                    None
                }
            }
            Ffn::Plain { .. } => {
                if layer.xielu.is_some() {
                    Some("the xIELU activation")
                } else if !arch_ok || !native(cfg.activation) {
                    Some("an activation without a device kernel")
                } else {
                    None
                }
            }
            Ffn::Gated { .. } => {
                if native(cfg.activation) {
                    None
                } else {
                    Some("an activation without a device kernel")
                }
            }
            Ffn::None => None,
        }
    }

    /// The output projection of NVFP4/BF16/F16 checkpoints is a single 0.5-1 GB matrix that is read for every
    /// generated token. Storing it as Q6_K (what llama.cpp uses for the output tensor of its Q4_K_M files) cuts
    /// that traffic by 60 %; set `FLLM_HEAD_FULL=1` to keep the original precision.
    fn upload_head(
        &self,
        backend: &dyn crate::backend::Backend,
        plain: impl Fn(&Linear) -> Result<crate::backend::DeviceWeight, String>,
    ) -> Result<crate::backend::DeviceWeight, String> {
        use crate::backend::QuantType;
        let head = &self.head;
        let (rows, cols) = (head.rows, head.cols);
        let float_bytes = match head.packed_weights() {
            None => Some(4usize),
            Some((0, _)) => Some(4),
            Some((1 | 30, _)) => Some(2),
            Some(_) => None,
        };
        let eligible = float_bytes.is_some()
            && rows * cols >= 100_000_000
            && cols.is_multiple_of(256)
            && std::env::var_os("FLLM_HEAD_FULL").is_none()
            && backend.capabilities().supported_quants.contains(&QuantType::Q6_K);
        let Some(width) = float_bytes.filter(|_| eligible) else {
            return plain(head);
        };
        let mut out = Vec::with_capacity(rows * cols / 256 * 210);
        let step = 2048;
        for first in (0..rows).step_by(step) {
            let n = step.min(rows - first);
            let values: Vec<f32> = match head.packed_weights() {
                Some((kind, bytes)) => crate::compute::dequantize(
                    &bytes[first * cols * width..(first + n) * cols * width],
                    kind,
                    n * cols,
                )?,
                None => head.weight.data[first * cols..(first + n) * cols].to_vec(),
            };
            for row in values.chunks_exact(cols) {
                fllm_checkpoint::quantize_row(fllm_checkpoint::QType::Q6K, row, &mut out);
            }
        }
        eprintln!("[fllm-rs] output projection stored as Q6_K on the device (set FLLM_HEAD_FULL=1 to keep full precision)");
        backend
            .upload_weight("output_head", &out, QuantType::Q6_K, rows, cols, head.bias.as_deref())
            .map_err(|e| e.to_string())
    }

    /// First reason the narrow-activation pipeline cannot run this model (`act_dtype::eligibility`).
    fn narrow_blocker(&self, layer_reasons: &[Option<&'static str>]) -> Option<&'static str> {
        use crate::backend::act_dtype::{eligibility, ModelShape};
        let cfg = &self.cfg;
        if self.par.is_some() {
            // the stages of a parallel group hand each other FP32 activations through host memory
            return Some("tensor / pipeline parallel stages");
        }
        let shape = ModelShape {
            all_layers_standard_attention: self.layers.iter().all(|l| matches!(l.attn, Attn::Std { .. })),
            host_layers: layer_reasons.iter().any(Option::is_some)
                || self.layers.iter().any(|l| l.value_moe.is_some() || self.ffn_host_reason(l, true, true).is_some())
                || FFN_HOST_FROM.lock().ok().and_then(|slot| *slot).is_some_and(|(layers, _)| layers == self.layers.len()),
            moe: self.layers.iter().any(|l| matches!(l.ffn, Ffn::Moe(_))),
            mla: cfg.mla.is_some(),
            gated_attention: self.layers.iter().any(|l| l.output_gate || l.attn_gate.is_some()),
            qkv_clip: cfg.qkv_clip.is_some(),
            parallel_residual: cfg.parallel_residual,
            layer_norm: !matches!(cfg.norm, NormKind::Rms) || self.layers.iter().any(|l| l.q_norm_bias.is_some() || l.k_norm_bias.is_some()),
            mtp: self.mtp.is_some(),
            recurrent: false,
        };
        eligibility(&shape).or_else(|| self.layers.iter().any(|l| l.xielu.is_some()).then_some("xIELU activation"))
    }

    /// `FLLM_ACT_DTYPE`: decide the activation type of the device pipeline and tell the backend before any weight is uploaded (BF16 needs BF16 scales
    /// for the 4-bit GEMMs). The default is f32, which leaves everything as it was.
    fn choose_activation_dtype(&self, backend: &dyn crate::backend::Backend, layer_reasons: &[Option<&'static str>]) -> crate::backend::act_dtype::ActDtype {
        use crate::backend::act_dtype::{decide, request_from_env, ActDtype, ModelFacts};
        let marlin_weights = self.layers.iter().any(|l| {
            matches!(&l.attn, Attn::Std { q, .. } if q.packed_weights().is_some_and(|(kind, _)| crate::backend::QuantType::from_gguf_type_id(kind).q4z_group().is_some_and(|g| g != 16)))
        });
        let facts = ModelFacts { arch: self.cfg.arch.clone(), source_bf16: self.act_source_bf16, blocker: self.narrow_blocker(layer_reasons), marlin_weights };
        let decision = decide(request_from_env(), &facts);
        let mut dtype = decision.dtype;
        if let Some(note) = &decision.note {
            eprintln!("[fllm-rs] {note}");
        }
        if dtype.is_narrow() && !backend.set_activation_dtype(dtype) {
            eprintln!("[fllm-rs] {} does not support {dtype} activations on this device; activations stay f32", backend.name());
            dtype = ActDtype::F32;
        }
        if !dtype.is_narrow() {
            backend.set_activation_dtype(ActDtype::F32);
        } else {
            eprintln!("[fllm-rs] activations on the device: {dtype} (FLLM_ACT_DTYPE; sums, residual adds and rotations still run in f32 registers)");
        }
        dtype
    }

    pub fn init_backend(
        &mut self,
        backend: Arc<dyn crate::backend::Backend>,
    ) -> Result<(), String> {
        if self.par.is_some() && (self.ple.is_some() || self.altup.is_some() || self.mtp.is_some() || self.mla_latent) {
            return Err(format!(
                "{} cannot run on the {} backend as a pipeline stage: its per-layer inputs depend on the embeddings of the first stage",
                self.cfg.arch,
                backend.name()
            ));
        }
        let (gate_ok, arch_ok, head_ok, value_ok) = backend
            .create_stream()
            .map(|s| (s.supports_attention_gate(), s.supports_arch_ops(), s.supports_head_gate(), s.supports_value_scale()))
            .unwrap_or((false, false, false, false));
        let caps = backend.capabilities();
        // the lightning indexer and the sparse attention on the device: plain indexers only (no pooling, no tail rotation), the expanded MLA cache
        let dsa_ok = self.dsa_device_possible(&*backend);
        self.dsa_device = dsa_ok && self.cfg.indexer.is_some();
        // MiniMax-M3: the block indexer and the attention over the selected blocks (f16 cache, the indexer queries rotate like the attention heads)
        let (m3_run, m3_device) = match self.cfg.minimax3.as_ref() {
            Some(mc) => {
                let host = std::env::var("FLLM_M3").is_ok_and(|v| v.eq_ignore_ascii_case("host"));
                let params = self.block_sparse_params(mc);
                let supported = !host
                    && self.par.is_none()
                    && mc.idx_dim >= mc.rot
                    && backend.create_stream().is_ok_and(|s| s.supports_block_sparse(&params, self.cfg.heads, mc.kv_heads, mc.head_dim, self.kv_kind));
                // FLLM_DSA=dense: every sparse layer attends to all keys (the plain device attention)
                (!host && self.par.is_none() && (self.dsa_dense || supported), supported && !self.dsa_dense)
            }
            None => (false, false),
        };
        self.m3_run = m3_run;
        self.m3_device = m3_device;
        self.ink_device = self.cfg.inkling.is_some() && self.ink_device_possible(&*backend);
        // hyper-connections on the device: the streams get their own buffer and the mixing runs in two small kernels per block; the layers' own
        // mixers and feed-forwards are the ordinary device ones (a branch output is the difference of the residual buffer before and after)
        let clamp_ok = backend.create_stream().is_ok_and(|s| s.supports_glu_clamped());
        let hc_ok = self.hc.is_some()
            && self.cfg.residual_scale == 1.0
            && self.par.is_none()
            && !std::env::var("FLLM_HC").is_ok_and(|v| v.eq_ignore_ascii_case("host"))
            && self.cfg.hc.as_ref().is_some_and(|h| backend.create_stream().is_ok_and(|s| s.supports_hc(h.hc_mult)))
            && self.layers.iter().all(|l| l.post_attn_norm.is_none() && l.post_ffn_norm.is_none() && !matches!(l.ffn, Ffn::None));
        self.hc_device = hc_ok;
        self.v4_device = self.v4_device_possible(&*backend, clamp_ok);
        if self.v4_device {
            self.hc_device = true;
        }
        let mut missing = self.backend_unsupported(caps.supports_layer_norm, caps.supports_grouped_rms_norm);
        // Layers without device kernels run on the host between the device layers (`host_layers`); a model in which no layer can
        // use the device gains nothing from it.
        // the short convolution keeps its history in a device state buffer, like the other recurrent mixers (one kind per model)
        let conv_ok = self.cfg.shortconv.as_ref().is_some_and(|sc| {
            self.delta_shape().is_none()
                && self.mamba_shape().is_none()
                && backend.create_stream().map(|st| st.supports_shortconv(self.cfg.hidden, sc.kernel)).unwrap_or(false)
                && std::env::var_os("FLLM_SHORTCONV_HOST").is_none()
        });
        // RWKV keeps the WKV matrices and the two token-shift vectors in a device state buffer (one kind of recurrent layer per model)
        let rwkv_ok = self.rwkv_shape().is_some_and(|s| {
            backend.create_stream().map(|st| st.supports_rwkv(&s)).unwrap_or(false) && std::env::var_os("FLLM_RWKV_HOST").is_none()
        });
        // Kimi Delta Attention likewise (Kimi-Linear, Ling-3.0): its state sits in a device buffer next to the latent attention's KV cache
        let kda_ok = self.kda_shape().is_some_and(|s| {
            backend.create_stream().map(|st| st.supports_kda(&s)).unwrap_or(false) && std::env::var_os("FLLM_KDA_HOST").is_none()
        });
        // Mamba-2 mixers, also the branch next to the attention of Falcon-H1: the state sits in a device buffer, which excludes DeltaNet layers on the device
        let mamba_ok = self.mamba_shape().is_some_and(|s| {
            self.delta_shape().is_none()
                && backend.create_stream().map(|st| st.supports_mamba2(&s)).unwrap_or(false)
                && std::env::var_os("FLLM_MAMBA_HOST").is_none()
        });
        // Zamba2's shared transformer blocks: attention with the model's KV cache on the device, in front of the device Mamba-2 mixers
        let zamba_ok = mamba_ok
            && self.cfg.hybrid.zamba.is_some()
            && backend.create_stream().map(|st| st.supports_copy_columns()).unwrap_or(false)
            && std::env::var_os("FLLM_ZAMBA_HOST").is_none();
        // FLLM_HOST_LAYERS=all|3,4,5 forces layers onto the host (debugging: the output must match `--device cpu`)
        let forced = std::env::var("FLLM_HOST_LAYERS").ok();
        let stage = self.stage_range();
        let mut layer_reasons: Vec<Option<&'static str>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                // layers of another pipeline stage are placeholders: nothing to decide for them
                if !stage.contains(&i) {
                    return None;
                }
                let on = forced
                    .as_deref()
                    .is_some_and(|f| f == "all" || f.split(',').any(|n| n.trim().parse::<usize>().is_ok_and(|n| n == i)));
                if on {
                    Some("forced by FLLM_HOST_LAYERS")
                } else {
                    self.layer_host_reason(i, l, gate_ok, arch_ok, head_ok, value_ok, conv_ok, dsa_ok, hc_ok, rwkv_ok, kda_ok, mamba_ok, zamba_ok)
                }
            })
            .collect();
        // LongCat-Flash: the shortcut experts of a layer join the stream after the layer that follows. The pending result lives in the pass over the
        // layers (device or host), so the model runs completely on one side; the shortcut MoE itself is computed by the host reference code
        // (routing with zero experts has no device kernel) between the device layers.
        if self.layers.iter().any(|l| l.shortcut.is_some() || l.takes_shortcut) && layer_reasons.iter().any(Option::is_some) {
            layer_reasons.iter_mut().for_each(|r| *r = Some("shortcut experts across layers (a layer of the model has no device kernels)"));
        }
        // the residual streams exist on one side only: a model with hyper-connections runs completely on the device or completely on the host
        if hc_ok && layer_reasons.iter().any(Option::is_some) {
            self.hc_device = false;
            if let Some(first) = layer_reasons.iter().flatten().next() {
                eprintln!("[fllm-rs] hyper-connections stay on the host: a layer of this model needs the host ({first})");
            }
            layer_reasons.iter_mut().for_each(|r| *r = Some("hyper-connections (a layer of the model has no device kernels)"));
        }
        // DeepSeek-V4 likewise: its streams, attention states and the experts of a layer exist on one side only
        if self.v4_device && layer_reasons.iter().any(Option::is_some) {
            self.v4_device = false;
            self.hc_device = false;
            layer_reasons.iter_mut().for_each(|r| *r = Some("DeepSeek-V4 (a layer of the model has no device kernels)"));
        }
        if self.mla_latent {
            self.check_latent_mla(&*backend, &layer_reasons)?;
        }
        if forced.is_none() && missing.is_empty() && !layer_reasons.is_empty() && layer_reasons.iter().all(Option::is_some) {
            for r in layer_reasons.iter().flatten() {
                if !missing.contains(r) {
                    missing.push(r);
                }
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "{} cannot run on the {} backend yet ({}); use --device cpu",
                self.cfg.arch,
                backend.name(),
                missing.join(", ")
            ));
        }
        let act_dtype = self.choose_activation_dtype(&*backend, &layer_reasons);
        let merge_on = merge_proj_requested(act_dtype.is_narrow(), backend.supports_merged_projections());
        let stream = backend.create_stream().map_err(|e| e.to_string())?;
        let staged = self.par.as_ref().is_some_and(|p| p.pp_size() > 1);
        let (kv_layers, kv_layout, kv_geoms) = if staged {
            // the cache of this stage's layers only, numbered from 0 (`kv_base`)
            (stage.len(), self.kvl.stage(stage.clone()), self.cfg.layer_geoms.as_ref().map(|g| g[stage.clone()].to_vec()))
        } else {
            (self.layers.len(), self.kvl.clone(), self.cfg.layer_geoms.clone())
        };
        let kv = backend
            .allocate_kv_cache_layered(
                kv_layers,
                self.pool_blocks,
                self.block_size,
                self.cfg.kv_heads,
                self.cfg.head_dim,
                self.cfg.value_dim,
                self.kv_kind,
                Some(kv_layout),
                kv_geoms,
            )
            .map_err(|e| e.to_string())?;

        let upload_linear =
            |name: &str, linear: &Linear| -> Result<crate::backend::DeviceWeight, String> {
                if let Some((kind, bytes)) = linear.packed_weights() {
                    // packed EXL2 / EXL3 blobs are expanded to F16 here when the backend has no native kernel for them
                    if fllm_gguf_probe::exl::is_exl_type(kind) && !backend.capabilities().supports_quant(crate::backend::QuantType::from_gguf_type_id(kind)) {
                        let half = crate::cpu_exl::expanded_f16(bytes, kind, linear.rows, linear.cols)?;
                        static NOTICE: std::sync::Once = std::sync::Once::new();
                        NOTICE.call_once(|| eprintln!("[fllm-rs] the {} backend has no EXL2/EXL3 kernel: packed weights are expanded to F16 on upload (first: {name})", backend.name()));
                        return backend
                            .upload_weight_scaled(name, &half, crate::backend::QuantType::F16, linear.rows, linear.cols, linear.bias.as_deref(), linear.scale)
                            .map_err(|e| e.to_string());
                    }
                    // MLX affine weights on a backend without a kernel for them: expanded to f16 once, here (documented in docs/gpu-backends.md)
                    let quant = crate::backend::QuantType::from_gguf_type_id(kind);
                    if quant.mlx_affine().is_some() && !caps.supports_quant(quant) {
                        return upload_mlx_expanded(&*backend, name, kind, bytes, linear);
                    }
                    // NVFP4: the backend stores the checkpoint's static activation scale next to the weight (read inside upload_weight_scaled)
                    crate::backend::set_next_nvfp4_input_scale(linear.input_scale);
                    let uploaded = backend
                        .upload_weight_scaled(
                            name,
                            bytes,
                            crate::backend::QuantType::from_gguf_type_id(kind),
                            linear.rows,
                            linear.cols,
                            linear.bias.as_deref(),
                            linear.scale,
                        )
                        .map_err(|e| e.to_string());
                    crate::backend::set_next_nvfp4_input_scale(0.0);
                    uploaded
                } else {
                    let bytes = cast_f32_to_u8(&linear.weight.data);
                    backend
                        .upload_weight(
                            name,
                            bytes,
                            crate::backend::QuantType::F32,
                            linear.rows,
                            linear.cols,
                            linear.bias.as_deref(),
                        )
                        .map_err(|e| e.to_string())
                }
            };

        // an unquantized linear as Q8_0 (blocks of 32 weights with an f16 scale): the int8 kernels instead of the f32 one, at a quarter of
        // the memory and a relative error below 0.4 % per weight
        let upload_linear_q8 = |name: &str, linear: &Linear| -> Result<crate::backend::DeviceWeight, String> {
            if !linear.cols.is_multiple_of(32) {
                return upload_linear(name, linear);
            }
            // f16 / bf16 weights stay packed in the linear: unpack them first (other packed formats are quantized already)
            let unpacked;
            let data: &[f32] = match linear.packed_weights() {
                Some((kind @ (1 | 30), bytes)) => {
                    unpacked = crate::compute::dequantize(bytes, kind, linear.rows * linear.cols)?;
                    &unpacked
                }
                Some(_) => return upload_linear(name, linear),
                None if linear.weight.data.len() == linear.rows * linear.cols => &linear.weight.data,
                None => return upload_linear(name, linear),
            };
            let bytes = quantize_q8_0(data);
            backend
                .upload_weight(name, &bytes, crate::backend::QuantType::Q8_0, linear.rows, linear.cols, linear.bias.as_deref())
                .map_err(|e| e.to_string())
                .or_else(|_| upload_linear(name, linear))
        };

        // Linears whose packed rows are one after the other in memory (the Marlin 4-bit formats without activation permutation,
        // NVFP4, and K-quants) as one device matrix: concatenated along the output dimension.
        // `None` when the parts differ in format or width, or have only some biases.
        let upload_merged = |name: &str, parts: &[&Linear]| -> Result<Option<DeviceMergedProj>, String> {
            let Some((kind, _)) = parts[0].packed_weights() else {
                return Ok(None);
            };
            let quant = crate::backend::QuantType::from_gguf_type_id(kind);
            let is_supported = (quant.q4z_group().is_some() && !quant.is_q4zp() && quant != crate::backend::QuantType::Nvfp4Marlin)
                || quant == crate::backend::QuantType::NvFp4
                || quant == crate::backend::QuantType::Nvfp4Marlin
                || matches!(quant, crate::backend::QuantType::Q4_K | crate::backend::QuantType::Q5_K | crate::backend::QuantType::Q6_K | crate::backend::QuantType::Q8_0);
            if !is_supported {
                return Ok(None);
            }
            let same = parts.iter().all(|p| p.packed_weights().is_some_and(|(k, _)| k == kind) && p.cols == parts[0].cols && p.scale == 1.0)
                && parts.iter().all(|p| p.bias.is_some() == parts[0].bias.is_some());
            if !same {
                return Ok(None);
            }
            if quant == crate::backend::QuantType::NvFp4 || quant == crate::backend::QuantType::Nvfp4Marlin {
                let s0 = parts[0].input_scale;
                if !parts.iter().all(|p| (p.input_scale - s0).abs() < 1e-6) {
                    return Ok(None);
                }
            }
            let mut bytes = Vec::new();
            let mut bias: Option<Vec<f32>> = parts[0].bias.as_ref().map(|_| Vec::new());
            let mut rows = 0;
            let mut sizes = [0usize; 3];
            for (n, p) in parts.iter().enumerate() {
                bytes.extend_from_slice(p.packed_weights().map(|(_, b)| b).unwrap_or_default());
                if let (Some(all), Some(b)) = (bias.as_mut(), p.bias.as_ref()) {
                    all.extend_from_slice(b);
                }
                rows += p.rows;
                sizes[n] = p.rows;
            }
            if quant == crate::backend::QuantType::NvFp4 || quant == crate::backend::QuantType::Nvfp4Marlin {
                crate::backend::set_next_nvfp4_input_scale(parts[0].input_scale);
            }
            let weight = backend
                .upload_weight_scaled(name, &bytes, quant, rows, parts[0].cols, bias.as_deref(), parts[0].scale)
                .map_err(|e| e.to_string())?;
            if quant == crate::backend::QuantType::NvFp4 || quant == crate::backend::QuantType::Nvfp4Marlin {
                crate::backend::set_next_nvfp4_input_scale(0.0);
            }
            Ok(Some(DeviceMergedProj { weight, parts: sizes }))
        };

        let upload_norm =
            |_name: &str, norm: &Norm| -> Result<crate::backend::DeviceBuffer, String> {
                let mut buf = backend
                    .allocate_buffer(norm.weight.len() * 4)
                    .map_err(|e| e.to_string())?;
                let mut s = backend.create_stream().map_err(|e| e.to_string())?;
                s.upload(&mut buf, cast_f32_to_u8(&norm.weight))
                    .map_err(|e| e.to_string())?;
                s.synchronize().map_err(|e| e.to_string())?;
                Ok(buf)
            };

        // DeltaNet layers run on the device when the stream has the kernels for their shape; otherwise on the host.
        let delta_shape = self.delta_shape().filter(|s| {
            backend.create_stream().map(|st| st.supports_delta(s)).unwrap_or(false)
                && std::env::var_os("FLLM_DELTA_HOST").is_none()
        });
        // Mamba-2 layers likewise (a model with both kinds keeps the Mamba-2 layers on the host: the state buffers are per kind)
        let mamba_shape = self.mamba_shape().filter(|s| {
            delta_shape.is_none()
                && backend.create_stream().map(|st| st.supports_mamba2(s)).unwrap_or(false)
                && std::env::var_os("FLLM_MAMBA_HOST").is_none()
        });
        let mamba1_shape = self.mamba1_shape().filter(|s| {
            delta_shape.is_none()
                && mamba_shape.is_none()
                && !conv_ok
                && backend.create_stream().map(|st| st.supports_mamba1(s)).unwrap_or(false)
                && std::env::var_os("FLLM_MAMBA_HOST").is_none()
        });
        let mut mamba_layers = 0usize;
        let mut conv_layers = 0usize;
        let mut mamba1_layers = 0usize;
        let mut rwkv_layers = 0usize;
        let mut kda_layers = 0usize;
        let mut v4_state_floats = 0usize;
        let mut delta_layers = 0usize;
        let mut dev_layers = Vec::with_capacity(self.layers.len());
        let mut host_summary: Vec<&'static str> = Vec::new();
        let mut ffn_host_blocks: (usize, &'static str) = (0, "");
        let mut host_count = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if !stage.contains(&i) {
                // another pipeline stage runs this layer: nothing is uploaded, the forward pass skips it
                dev_layers.push(DeviceLayerWeights::empty(false));
                continue;
            }
            if let Some(reason) = layer_reasons[i] {
                host_count += 1;
                if !host_summary.contains(&reason) {
                    host_summary.push(reason);
                }
                dev_layers.push(DeviceLayerWeights::empty(true));
                continue;
            }
            let ffn_host_exact = self.ffn_host_reason(layer, arch_ok, clamp_ok).is_some();
            if let Some(reason) = self.ffn_host_reason(layer, arch_ok, clamp_ok) {
                ffn_host_blocks.0 += 1;
                ffn_host_blocks.1 = reason;
            }
            let attn_norm = layer
                .attn_norm
                .as_ref()
                .map(|n| upload_norm(&format!("blk.{i}.attn_norm"), n))
                .transpose()?;
            // FLLM_MERGE_PROJ: q/k/v and gate/up as one Marlin matrix each (instead of seven GEMMs per layer, four)
            let merged_qkv = match &layer.attn {
                Attn::Std { q, k, v } if merge_on && layer.value_moe.is_none() && !layer.output_gate && self.cfg.qkv_clip.is_none() && !layer.kv_shared && !layer.k_eq_v => {
                    upload_merged(&format!("blk.{i}.attn_qkv"), &[q, k, v])?
                }
                _ => None,
            };
            let (q, k, v) = match &layer.attn {
                Attn::Std { .. } if merged_qkv.is_some() => (None, None, None),
                Attn::Std { q, k, v } => (
                    Some(upload_linear(&format!("blk.{i}.attn_q"), q)?),
                    if layer.kv_shared { None } else { Some(upload_linear(&format!("blk.{i}.attn_k"), k)?) },
                    if layer.value_moe.is_some() || layer.kv_shared || layer.k_eq_v { None } else { Some(upload_linear(&format!("blk.{i}.attn_v"), v)?) },
                ),
                _ => (None, None, None),
            };
            // only attention and the device DeltaNet use the output projection on the device (the other mixers apply it on the host)
            let out = if matches!(layer.attn, Attn::Std { .. } | Attn::Mla(_)) || (matches!(layer.attn, Attn::ShortConv(_)) && conv_ok) || (matches!(layer.attn, Attn::Mamba1(_)) && mamba1_shape.is_some()) || (matches!(layer.attn, Attn::Linear(_)) && delta_shape.is_some())
                || (matches!(layer.attn, Attn::Rwkv(_)) && rwkv_ok)
                || (matches!(layer.attn, Attn::Kda(_)) && kda_ok)
                || (matches!(layer.attn, Attn::Mamba(_)) && mamba_shape.is_some())
            {
                Some(upload_linear(&format!("blk.{i}.attn_out"), &layer.out)?)
            } else {
                None
            };
            let upload_vec = |v: &Vec<f32>| -> Result<crate::backend::DeviceBuffer, String> {
                let norm = Norm { weight: v.clone(), bias: None };
                upload_norm("", &norm)
            };
            let delta = match (&layer.attn, delta_shape.as_ref()) {
                (Attn::Linear(d), Some(_)) => {
                    delta_layers += 1;
                    Some(DeviceDelta {
                        qkv: upload_linear(&format!("blk.{i}.ssm_qkv"), &d.qkv)?,
                        gate: upload_linear(&format!("blk.{i}.ssm_gate"), &d.gate)?,
                        beta: upload_linear(&format!("blk.{i}.ssm_beta"), &d.beta)?,
                        alpha: upload_linear(&format!("blk.{i}.ssm_alpha"), &d.alpha)?,
                        conv: upload_vec(&d.conv)?,
                        dt_bias: upload_vec(&d.dt_bias)?,
                        decay: upload_vec(&d.decay)?,
                        norm: upload_vec(&d.norm)?,
                        ordinal: delta_layers - 1,
                    })
                }
                _ => None,
            };
            let mamba = match (&layer.attn, mamba_shape.as_ref()) {
                (Attn::Mamba(m), Some(_)) => {
                    mamba_layers += 1;
                    Some(DeviceMamba {
                        input: upload_linear(&format!("blk.{i}.ssm_in"), &m.input)?,
                        conv: upload_vec(&m.conv)?,
                        conv_bias: m.conv_bias.as_ref().map(upload_vec).transpose()?,
                        dt_bias: upload_vec(&m.dt_bias)?,
                        decay: upload_vec(&m.decay)?,
                        skip: upload_vec(&m.skip)?,
                        norm: if m.norm.is_empty() { upload_vec(&vec![1.0; self.cfg.mamba.as_ref().map_or(0, |c| c.inner)])? } else { upload_vec(&m.norm)? },
                        ordinal: mamba_layers - 1,
                    })
                }
                _ => None,
            };
            let zamba_dev = match (&layer.zamba, zamba_ok) {
                (Some(call), true) => Some(call.upload(&format!("blk.{i}.zamba"), &|name: &str, l: &Linear| upload_linear(name, l), &|v: &[f32]| upload_vec(&v.to_vec()))?),
                _ => None,
            };
            let par_mamba = match (&layer.parallel, mamba_shape.as_ref()) {
                (Some(par), Some(_)) if mamba_ok => {
                    mamba_layers += 1;
                    let m = &par.mamba;
                    Some((
                        DeviceMamba {
                            input: upload_linear(&format!("blk.{i}.ssm_in"), &m.input)?,
                            conv: upload_vec(&m.conv)?,
                            conv_bias: m.conv_bias.as_ref().map(upload_vec).transpose()?,
                            dt_bias: upload_vec(&m.dt_bias)?,
                            decay: upload_vec(&m.decay)?,
                            skip: upload_vec(&m.skip)?,
                            norm: if m.norm.is_empty() { upload_vec(&vec![1.0; self.cfg.mamba.as_ref().map_or(0, |c| c.inner)])? } else { upload_vec(&m.norm)? },
                            ordinal: mamba_layers - 1,
                        },
                        upload_linear(&format!("blk.{i}.ssm_out"), &par.out)?,
                    ))
                }
                _ => None,
            };
            let mamba1 = match (&layer.attn, mamba1_shape.as_ref()) {
                (Attn::Mamba1(m), Some(_)) => {
                    mamba1_layers += 1;
                    Some(DeviceMamba1 {
                        input: upload_linear(&format!("blk.{i}.ssm_in"), &m.input)?,
                        conv: upload_vec(&m.conv)?,
                        conv_bias: m.conv_bias.as_ref().map(upload_vec).transpose()?,
                        x_proj: upload_linear(&format!("blk.{i}.ssm_x"), &m.x_proj)?,
                        dt_proj: upload_linear(&format!("blk.{i}.ssm_dt"), &m.dt_proj)?,
                        decay: upload_vec(&m.decay)?,
                        skip: upload_vec(&m.skip)?,
                        norms: match &m.norms {
                            Some(w) => Some([upload_vec(&w[0])?, upload_vec(&w[1])?, upload_vec(&w[2])?]),
                            None => None,
                        },
                        ordinal: mamba1_layers - 1,
                    })
                }
                _ => None,
            };
            let kda = match &layer.attn {
                Attn::Kda(k) if kda_ok => {
                    kda_layers += 1;
                    Some(k.upload(kda_layers - 1, &|name: &str, l: &Linear| upload_linear(name, l), &|v: &[f32]| upload_vec(&v.to_vec()), &format!("blk.{i}.kda"))?)
                }
                _ => None,
            };
            let rwkv = match &layer.attn {
                Attn::Rwkv(rw) if rwkv_ok => {
                    rwkv_layers += 1;
                    Some(rw.upload(&*backend, &format!("blk.{i}.rwkv"), rwkv_layers - 1, &|name: &str, l: &Linear| upload_linear(name, l), &|v: &[f32]| upload_vec(&v.to_vec()))?)
                }
                _ => None,
            };
            let shortconv = match &layer.attn {
                Attn::ShortConv(m) if conv_ok => {
                    conv_layers += 1;
                    Some(DeviceShortConv {
                        input: upload_linear(&format!("blk.{i}.shortconv.in_proj"), &m.input)?,
                        conv: upload_vec(&m.conv)?,
                        bias: m.bias.as_ref().map(upload_vec).transpose()?,
                        ordinal: conv_layers - 1,
                    })
                }
                _ => None,
            };
            let mla = match &layer.attn {
                Attn::Mla(m) => Some(DeviceMla {
                    q_a: match &m.q_a {
                        Some((down, norm)) => Some((upload_linear(&format!("blk.{i}.attn_q_a"), down)?, upload_vec(norm)?)),
                        None => None,
                    },
                    q: upload_linear(&format!("blk.{i}.attn_q_b"), &m.q)?,
                    kv_a: upload_linear(&format!("blk.{i}.attn_kv_a"), &m.kv_a)?,
                    kv_a_norm: upload_vec(&m.kv_a_norm)?,
                    // the latent cache needs the dense FP16 kv_b matrix (absorbed into the queries, applied to the attention output) instead
                    kv_b: if self.mla_latent { None } else { Some(upload_linear(&format!("blk.{i}.attn_kv_b"), &m.kv_b)?) },
                    kv_b_dense: if self.mla_latent {
                        Some(mla_latent::upload_f16(&*backend, &crate::mla_latent::dense_kv_b_f16(&m.kv_b)?)?)
                    } else {
                        None
                    },
                }),
                _ => None,
            };
            let dsa = match (&layer.attn, layer.indexer.as_deref(), self.cfg.indexer.as_ref()) {
                (Attn::Mla(_), Some(ix), Some(ic)) if self.dsa_device => Some(DeviceDsa {
                    q_b: upload_linear(&format!("blk.{i}.indexer.attn_q_b"), &ix.q_b)?,
                    k: upload_linear(&format!("blk.{i}.indexer.attn_k"), &ix.k)?,
                    proj: upload_linear(&format!("blk.{i}.indexer.proj"), &ix.proj)?,
                    k_norm: upload_vec(&ix.k_norm)?,
                    k_norm_bias: upload_vec(&ix.k_norm_bias)?,
                    pool: match &ix.pool {
                        Some((gate, ape)) => Some((upload_linear(&format!("blk.{i}.indexer.pool_gate"), gate)?, upload_vec(ape)?)),
                        None => None,
                    },
                    store: backend.allocate_buffer(self.pool_blocks * self.block_size * ic.store_width() * 4).map_err(|e| e.to_string())?,
                }),
                _ => None,
            };
            let m3 = match (self.minimax3.as_deref().and_then(|m| m.indexers.get(i)).and_then(Option::as_ref), self.cfg.minimax3.as_ref()) {
                (Some(ix), Some(mc)) if self.m3_device => Some(DeviceM3 {
                    q: upload_linear(&format!("blk.{i}.indexer.attn_q"), &ix.q)?,
                    k: upload_linear(&format!("blk.{i}.indexer.attn_k"), &ix.k)?,
                    qn: upload_vec(&ix.qn)?,
                    kn: upload_vec(&ix.kn)?,
                    store: backend.allocate_buffer(self.pool_blocks * self.block_size * mc.idx_dim * 4).map_err(|e| e.to_string())?,
                }),
                _ => None,
            };
            let ink = match (self.inkling.as_deref().and_then(|m| m.layers.get(i)), self.ink_device) {
                (Some(il), true) => {
                    let up_expert = |e: &Expert, tag: &str| -> Result<DeviceExpert, String> {
                        let gate = e.gate.as_ref().ok_or("an Inkling expert without a gate projection")?;
                        Ok(DeviceExpert {
                            gate: upload_linear(&format!("blk.{i}.{tag}.gate"), gate)?,
                            up: upload_linear(&format!("blk.{i}.{tag}.up"), &e.up)?,
                            down: upload_linear(&format!("blk.{i}.{tag}.down"), &e.down)?,
                        })
                    };
                    let (mut experts, mut shared) = (Vec::new(), Vec::new());
                    if let Some(im) = &il.moe {
                        for (n, e) in im.experts.iter().enumerate() {
                            experts.push(up_expert(e, &format!("exp{n}"))?);
                        }
                        for (n, e) in im.shared.iter().enumerate() {
                            shared.push(up_expert(e, &format!("shared{n}"))?);
                        }
                    }
                    Some(inkling_device::DeviceInk {
                        q: upload_linear(&format!("blk.{i}.attn_query"), &il.q)?,
                        k: upload_linear(&format!("blk.{i}.attn_key"), &il.k)?,
                        v: upload_linear(&format!("blk.{i}.attn_value"), &il.v)?,
                        r: upload_linear(&format!("blk.{i}.attn_rel"), &il.r)?,
                        o: upload_linear(&format!("blk.{i}.attn_out"), &il.o)?,
                        qn: upload_vec(&il.qn)?,
                        kn: upload_vec(&il.kn)?,
                        profile: upload_vec(&il.profile)?,
                        k_conv: upload_vec(&il.k_conv.weights)?,
                        v_conv: upload_vec(&il.v_conv.weights)?,
                        attn_conv: upload_vec(&il.attn_conv.weights)?,
                        ffn_conv: upload_vec(&il.ffn_conv.weights)?,
                        experts,
                        shared,
                    })
                }
                _ => None,
            };
            let v4 = match (self.v4_device, self.v4.as_deref(), self.v4_shape(i)) {
                (true, Some(model), Some(shape)) => {
                    let offset = v4_state_floats * 4;
                    v4_state_floats += shape.layout().total;
                    let sinks = layer.sinks.as_deref().filter(|s| s.len() == self.cfg.heads).ok_or("DeepSeek-V4 layer without attention sinks")?;
                    Some(model.layers[i].upload(&format!("blk.{i}"), shape, offset, sinks, &|name: &str, l: &Linear| upload_linear(name, l), &|v: &[f32]| upload_vec(&v.to_vec()))?)
                }
                _ => None,
            };
            let q_norm = layer.q_norm.as_ref().map(upload_vec).transpose()?;
            let k_norm = layer.k_norm.as_ref().map(upload_vec).transpose()?;
            let post_attn_norm = layer.post_attn_norm.as_ref().map(|n| upload_norm("post_attn", n)).transpose()?;
            let post_ffn_norm = layer.post_ffn_norm.as_ref().map(|n| upload_norm("post_ffn", n)).transpose()?;
            let ffn_norm = layer
                .ffn_norm
                .as_ref()
                .map(|n| upload_norm(&format!("blk.{i}.ffn_norm"), n))
                .transpose()?;
            let host_ffn = ffn_host_exact
                || FFN_HOST_FROM
                    .lock()
                    .ok()
                    .and_then(|slot| *slot)
                    .is_some_and(|(layers, first)| layers == self.layers.len() && i >= first);
            let merged_gate_up = match &layer.ffn {
                Ffn::Gated { gate, up, .. } if merge_on && !host_ffn && self.cfg.inkling.is_none() && matches!(self.cfg.activation, Activation::Silu | Activation::GeluTanh) && !matches!(self.dense_glu_mode(), Some(GluMode::Clamped(_) | GluMode::SwigluOai(..))) => {
                    upload_merged(&format!("blk.{i}.ffn_gate_up"), &[gate, up])?
                }
                _ => None,
            };
            let (ffn_gate, ffn_up, ffn_down) = match &layer.ffn {
                Ffn::Gated { .. } | Ffn::Plain { .. } if host_ffn => (None, None, None),
                Ffn::Gated { down, .. } if merged_gate_up.is_some() => (None, None, Some(upload_linear(&format!("blk.{i}.ffn_down"), down)?)),
                Ffn::Gated { gate, up, down } => (
                    Some(upload_linear(&format!("blk.{i}.ffn_gate"), gate)?),
                    Some(upload_linear(&format!("blk.{i}.ffn_up"), up)?),
                    Some(upload_linear(&format!("blk.{i}.ffn_down"), down)?),
                ),
                Ffn::Plain { up, down } => (
                    None,
                    Some(upload_linear(&format!("blk.{i}.ffn_up"), up)?),
                    Some(upload_linear(&format!("blk.{i}.ffn_down"), down)?),
                ),
                _ => (None, None, None),
            };
            // the shortcut experts of LongCat-Flash sit next to a dense feed-forward and are uploaded like the experts of a mixture-of-experts layer
            let moe_src: Option<&Moe> = match &layer.ffn {
                Ffn::Moe(moe) => Some(moe),
                _ => layer.shortcut.as_deref(),
            };
            let moe_dev = match moe_src {
                Some(moe) if !host_ffn && self.device_glu_mode().is_some() => {
                    let up_expert = |e: &Expert, tag: &str| -> Result<Option<DeviceExpert>, String> {
                        let Some(gate) = e.gate.as_ref() else {
                            return Ok(None);
                        };
                        Ok(Some(DeviceExpert {
                            gate: upload_linear(&format!("blk.{i}.{tag}.gate"), gate)?,
                            up: upload_linear(&format!("blk.{i}.{tag}.up"), &e.up)?,
                            down: upload_linear(&format!("blk.{i}.{tag}.down"), &e.down)?,
                        }))
                    };
                    let mut experts = Vec::with_capacity(moe.experts.len());
                    for (n, e) in moe.experts.iter().enumerate() {
                        experts.push(up_expert(e, &format!("exp{n}"))?);
                    }
                    let shared = match &moe.shared {
                        Some((e, _)) => up_expert(e, "shared")?,
                        None => None,
                    };
                    // zero experts need `moe_zero_fold` next to the device routing
                    let device_moe = backend.moe_device_available()
                        && (self.cfg.moe.as_ref().is_none_or(|m| m.zero_experts == 0) || backend.create_stream().is_ok_and(|s| s.supports_moe_zero()));
                    let fast = if self.device_routing_ok(moe, device_moe) && (moe.shared.is_none() || shared.is_some()) {
                        let all: Option<Vec<&DeviceExpert>> = experts.iter().map(|e| e.as_ref()).collect();
                        match all {
                            Some(list) if !list.is_empty() => {
                                let kinds = [list[0].gate.quant_type, list[0].up.quant_type, list[0].down.quant_type];
                                let same = list.iter().all(|e| {
                                    e.gate.quant_type == kinds[0] && e.up.quant_type == kinds[1] && e.down.quant_type == kinds[2]
                                });
                                let ok_kind = |q: crate::backend::QuantType| matches!(q.to_gguf_type_id(), 1 | 30 | 2 | 3 | 6 | 7 | 8 | 10..=14 | 16..=23 | 29 | 39 | 40 | 0x8000_0010..=0x8000_0012);
                                let (ff, hid) = (list[0].gate.rows, list[0].gate.cols);
                                let dims = list[0].down.rows == hid
                                    && list[0].down.cols == ff
                                    && hid.is_multiple_of(16)
                                    && ff.is_multiple_of(16)
                                    && kinds.iter().all(|q| {
                                        let id = q.to_gguf_type_id();
                                        if matches!(id, 10..=14 | 16..=19 | 21 | 22 | 29) {
                                            hid.is_multiple_of(256) && ff.is_multiple_of(256)
                                        } else {
                                            true
                                        }
                                    })
                                    && {
                                        let ok = |sel: fn(&DeviceExpert) -> bool| list.iter().all(|e| sel(e)) || list.iter().all(|e| !sel(e));
                                        ok(|e| e.gate.bias.is_some()) && ok(|e| e.up.bias.is_some()) && ok(|e| e.down.bias.is_some())
                                    };
                                let dp4a = kinds.iter().all(|&q| ok_kind(q));
                                let grouped = device_moe
                                    && [16usize, 64].iter().all(|&b| {
                                        backend.moe_grouped_supported(kinds[0], hid, b)
                                            && backend.moe_grouped_supported(kinds[1], hid, b)
                                            && backend.moe_grouped_supported(kinds[2], ff, b)
                                    });
                                if same && dims && (dp4a || grouped) {
                                    let gates: Vec<&crate::backend::DeviceWeight> = list.iter().map(|e| &e.gate).collect();
                                    let ups: Vec<&crate::backend::DeviceWeight> = list.iter().map(|e| &e.up).collect();
                                    let downs: Vec<&crate::backend::DeviceWeight> = list.iter().map(|e| &e.down).collect();
                                    let tables = (
                                        backend.upload_pointer_table(&gates),
                                        backend.upload_pointer_table(&ups),
                                        backend.upload_pointer_table(&downs),
                                    );
                                    let bias_table = |ws: &[&crate::backend::DeviceWeight]| {
                                        if ws.iter().all(|w| w.bias.is_some()) {
                                            backend.upload_bias_table(ws).map(Some)
                                        } else {
                                            Ok(None)
                                        }
                                    };
                                    let biases = (bias_table(&gates), bias_table(&ups), bias_table(&downs));
                                    match (tables, biases) {
                                        ((Ok(gate), Ok(up), Ok(down)), (Ok(gate_bias), Ok(up_bias), Ok(down_bias))) => Some(DeviceMoeFast {
                                            router: upload_linear(&format!("blk.{i}.router"), &moe.router)?,
                                            gate,
                                            up,
                                            down,
                                            gate_bias,
                                            up_bias,
                                            down_bias,
                                            kinds,
                                            ff,
                                            dp4a,
                                            device: if device_moe {
                                                Some(DeviceMoeDevice {
                                                    route: self.device_route(moe),
                                                    bias: moe.bias.as_ref().map(upload_vec).transpose()?,
                                                    shared_gate: moe.shared.as_ref().and_then(|(_, g)| g.as_ref()).map(upload_vec).transpose()?,
                                                    grouped,
                                                })
                                            } else {
                                                None
                                            },
                                        }),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    // a backend whose expert tables are copies (Vulkan arenas) would hold every expert twice: the experts become slices of the tables
                    let mut experts = experts;
                    if let Some(f) = fast.as_ref() {
                        let mut live: Vec<&mut DeviceExpert> = experts.iter_mut().flatten().collect();
                        let mut gates: Vec<&mut crate::backend::DeviceWeight> = live.iter_mut().map(|e| &mut e.gate).collect();
                        let _ = backend.alias_weights_into_table(&f.gate, &mut gates);
                        let mut ups: Vec<&mut crate::backend::DeviceWeight> = live.iter_mut().map(|e| &mut e.up).collect();
                        let _ = backend.alias_weights_into_table(&f.up, &mut ups);
                        let mut downs: Vec<&mut crate::backend::DeviceWeight> = live.iter_mut().map(|e| &mut e.down).collect();
                        let _ = backend.alias_weights_into_table(&f.down, &mut downs);
                    }
                    Some(DeviceMoe { experts, shared, fast })
                }
                _ => None,
            };
            dev_layers.push(DeviceLayerWeights {
                attn_norm,
                q,
                k,
                v,
                out,
                ffn_norm,
                attn_norm_b: layer.attn_norm.as_ref().and_then(|n| n.bias.as_ref()).map(upload_vec).transpose()?,
                ffn_norm_b: layer.ffn_norm.as_ref().and_then(|n| n.bias.as_ref()).map(upload_vec).transpose()?,
                post_attn_norm_b: layer.post_attn_norm.as_ref().and_then(|n| n.bias.as_ref()).map(upload_vec).transpose()?,
                post_ffn_norm_b: layer.post_ffn_norm.as_ref().and_then(|n| n.bias.as_ref()).map(upload_vec).transpose()?,
                host_layer: false,
                host_ffn: ffn_host_exact,
                host_values: layer.value_moe.is_some(),
                attn_gate: layer.attn_gate.as_ref().map(|g| upload_linear(&format!("blk.{i}.attn_gate"), g)).transpose()?,
                attn_sub_norm: layer.attn_sub_norm.as_ref().map(upload_vec).transpose()?,
                ffn_sub_norm: layer.ffn_sub_norm.as_ref().map(upload_vec).transpose()?,
                q_norm,
                k_norm,
                post_attn_norm,
                post_ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                moe: moe_dev,
                delta,
                mla,
                dsa,
                m3,
                ink,
                mamba,
                shortconv,
                mamba1,
                rwkv,
                kda,
                v4,
                zamba: zamba_dev,
                par_mamba,
                sinks: layer.sinks.clone().map(Arc::new),
                qkv: merged_qkv,
                gate_up: merged_gate_up,
                ple: match layer.gemma.as_ref().and_then(|g| g.ple_parts()) {
                    Some((gate, proj, norm)) => Some((
                        upload_linear_q8(&format!("blk.{i}.inp_gate"), gate)?,
                        upload_linear_q8(&format!("blk.{i}.proj"), proj)?,
                        upload_norm(&format!("blk.{i}.post_norm"), norm)?,
                    )),
                    None => None,
                },
                out_scale: layer.gemma.as_ref().and_then(|g| g.output_scale()).filter(|&s| s != 1.0),
                v_norm: if self.cfg.value_norm && !layer.kv_shared {
                    Some(upload_vec(&vec![1.0f32; self.cfg.layer_value_dim(i)])?)
                } else {
                    None
                },
            });
        }

        let shortcut_layers = self.layers.iter().filter(|l| l.shortcut.is_some()).count();
        let shortcut_host = self.layers.iter().zip(&dev_layers).filter(|(l, d)| l.shortcut.is_some() && d.moe.is_none()).count();
        if shortcut_host > 0 {
            eprintln!("[fllm-rs] the shortcut experts of {shortcut_host} of {shortcut_layers} layers run on the host (device expert weights are not available for them); the rest of the model runs on the device");
        }
        if ffn_host_blocks.0 > 0 {
            eprintln!("[fllm-rs] the feed-forward blocks of {} of {} layers run on the host: {}", ffn_host_blocks.0, self.layers.len(), ffn_host_blocks.1);
        }
        if host_count > 0 {
            eprintln!(
                "[fllm-rs] {host_count} of {} layers run on the host between the device layers ({})",
                self.layers.len(),
                host_summary.join(", ")
            );
        }
        let final_norm = upload_norm("output_norm", &self.final_norm)?;
        let final_norm_b = self
            .final_norm
            .bias
            .as_ref()
            .map(|b| upload_norm("output_norm_bias", &Norm { weight: b.clone(), bias: None }))
            .transpose()?;
        let head = self.upload_head(&*backend, |l| upload_linear("output_head", l))?;

        // The input rows are built on the host (scaling, image overrides, learned positions),
        // so the token embedding table stays off the device.
        let v4_weights = match (self.v4_device, self.v4.as_deref()) {
            (true, Some(model)) => {
                let vec = |v: &Vec<f32>| upload_norm("", &Norm { weight: v.clone(), bias: None });
                let block = |name: &str, c: &deepseek4::Hc| -> Result<DeviceHc, String> {
                    Ok(DeviceHc { mix: upload_linear(&format!("{name}_fn"), &c.mix)?, base: vec(&c.base)?, scale: vec(&c.scale)? })
                };
                let hc_mult = self.cfg.v4.as_ref().map_or(1, |v| v.hc_mult);
                let attn = model.layers.iter().enumerate().map(|(i, l)| block(&format!("blk.{i}.hc_attn"), &l.hc_attn)).collect::<Result<Vec<_>, _>>()?;
                let ffn = model.layers.iter().enumerate().map(|(i, l)| block(&format!("blk.{i}.hc_ffn"), &l.hc_ffn)).collect::<Result<Vec<_>, _>>()?;
                let head = Some(block("output_hc", &model.head)?);
                let hd = self.cfg.v4.as_ref().map_or(1, |v| v.head_dim);
                Some((DeviceHcModel { attn, ffn, head, ones: vec(&vec![1.0f32; hc_mult * self.cfg.hidden])? }, vec(&vec![1.0f32; hd])?))
            }
            _ => None,
        };
        let (v4_hc, v4_ones) = match v4_weights {
            Some((hc, ones)) => (Some(hc), Some(ones)),
            None => (None, None),
        };
        let hc_weights = match (self.hc_device, self.hc.as_deref(), self.cfg.hc.as_ref()) {
            (true, Some(model), Some(h)) => {
                let vec = |v: &Vec<f32>| upload_norm("", &Norm { weight: v.clone(), bias: None });
                let block = |name: &str, c: &deepseek4::Hc| -> Result<DeviceHc, String> {
                    Ok(DeviceHc { mix: upload_linear(&format!("{name}_fn"), &c.mix)?, base: vec(&c.base)?, scale: vec(&c.scale)? })
                };
                let attn = model.attn.iter().enumerate().map(|(i, c)| block(&format!("blk.{i}.hc_attn"), c)).collect::<Result<Vec<_>, _>>()?;
                let ffn = model.ffn.iter().enumerate().map(|(i, c)| block(&format!("blk.{i}.hc_ffn"), c)).collect::<Result<Vec<_>, _>>()?;
                let head = model.head.as_ref().map(|c| block("output_hc", c)).transpose()?;
                Some(DeviceHcModel { attn, ffn, head, ones: vec(&vec![1.0f32; h.hc_mult * self.cfg.hidden])? })
            }
            _ => v4_hc,
        };
        let dev_weights = DeviceModelWeights {
            layers: dev_layers,
            final_norm,
            final_norm_b,
            head,
            rope: self
                .rope
                .iter()
                .map(|t| (Arc::new(t.cos.clone()), Arc::new(t.sin.clone())))
                .collect(),
            alibi: self.cfg.alibi_slopes.clone().map(Arc::new),
            ple_proj: match self.ple.as_ref() {
                Some(global) => Some(upload_linear_q8("per_layer_model_proj", global.projection())?),
                None => None,
            },
            hc: hc_weights,
            v4_ones,
        };

        let mut pool = crate::backend::BufferPool::new(Arc::clone(&backend));
        // FLLM_WORKSPACE_ROWS=N (default 0 = off): the buffer pool and the library scratch are sized once for batches of N rows
        let reserve_rows = std::env::var("FLLM_WORKSPACE_ROWS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0).min(16384);
        let max_tokens = reserve_rows.max(512);
        let hidden_bytes = max_tokens * self.cfg.hidden * 4;
        let ffn_bytes = max_tokens * self.cfg.ff * 4;
        pool.preallocate(&[
            hidden_bytes,
            hidden_bytes,
            hidden_bytes,
            ffn_bytes,
            ffn_bytes,
            ffn_bytes,
        ])
        .map_err(|e| e.to_string())?;

        let mut stream = stream;
        if reserve_rows > 0 {
            let wide = self.cfg.hidden.max(self.cfg.ff);
            stream.reserve_workspace(reserve_rows, wide, 2 * wide).map_err(|e| e.to_string())?;
            eprintln!("[fllm-rs] workspace reserved for batches of {reserve_rows} rows (FLLM_WORKSPACE_ROWS)");
        }
        let stream2 = backend.create_stream().ok();
        self.backend = Some(backend);
        self.backend_stream = Some(std::sync::Mutex::new(stream));
        self.backend_stream2 = stream2.map(std::sync::Mutex::new);
        self.backend_weights = Some(Arc::new(dev_weights));
        self.backend_kv = Some(kv);
        self.backend_scratch = Some(std::sync::Mutex::new(pool));
        Ok(())
    }

    /// Forward pass on the accelerator backend. Activations stay on the device from the
    /// embedding upload to the logits download; keys and values are written into the device
    /// KV cache (`kv_cache_write`) and read there by `attention_paged`, so no step moves cache
    /// contents to the host. `key`/`value` (the host cache) are not touched.
    #[allow(clippy::too_many_arguments)]
    fn forward_backend_inner(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
        capture: Option<&mut Vec<f32>>,
        mut graph: GraphMode<'_>,
    ) -> Result<Vec<f32>, String> {
        use crate::backend::{AttentionParams, RopeParams};
        let rows = ids.len();
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let mtp = usize::from(self.mtp.is_some());
        let max_position = self.max_position();
        // The CPU path's batch checks, minus the host cache sizes.
        if rows == 0
            || lengths.len() != rows
            || tables.len() != rows * self.stride
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.embedding.rows)
            || logit_rows.iter().any(|&row| row >= rows)
        {
            return Err("invalid Rust decoder batch".into());
        }
        if overrides
            .iter()
            .any(|(r, x)| *r >= rows || x.len() != hidden || x.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid image embedding overrides".into());
        }
        let positions3 = self.checked_positions3(rows)?;
        let deepstack = self.checked_deepstack(rows)?;
        let mut positions = Vec::with_capacity(rows);
        let mut slots = Vec::with_capacity(rows);
        for (r, &length) in lengths.iter().enumerate() {
            if length <= 0
                || (length as usize)
                    .checked_add(mtp)
                    .is_none_or(|n| n > max_position)
            {
                return Err("invalid Rust decoder position".into());
            }
            let table = &tables[r * self.stride..(r + 1) * self.stride];
            let position = length as usize - 1 + mtp;
            let used = (length as usize).div_ceil(self.block_size);
            let current = position / self.block_size;
            if used > self.stride || current >= self.stride {
                return Err("invalid Rust decoder KV block".into());
            }
            for &block in table[..used].iter().chain(&table[current..=current]) {
                if block < 0 || block as usize >= self.pool_blocks {
                    return Err("invalid Rust decoder KV block".into());
                }
            }
            positions.push(position);
            // Cache slot of this row's current position, written by every attention layer.
            slots.push(table[current] as usize * self.block_size + position % self.block_size);
        }

        let not_ready = "accelerator backend is not initialised";
        let stream_lock = self.backend_stream.as_ref().ok_or(not_ready)?;
        // declared before the lock guard, so that it runs after the guard is released: the stream learns that the pass is over on every way out
        let _step = StepScope(stream_lock);
        let mut stream = stream_lock.lock().map_err(|_| "accelerator stream lock poisoned")?;
        stream.begin_step();
        let dev_weights = self.backend_weights.as_ref().ok_or(not_ready)?;
        let dev_kv = self.backend_kv.as_ref().ok_or(not_ready)?;
        let mut scratch = self
            .backend_scratch
            .as_ref()
            .ok_or(not_ready)?
            .lock()
            .map_err(|_| "accelerator scratch lock poisoned")?;
        let err = |e: crate::backend::BackendError| e.to_string();

        // while a CUDA graph is recorded the residual stream is the graph's own buffer, filled before every launch
        let recording = matches!(graph, GraphMode::Capture(_));
        let mut x_buf = match &graph {
            GraphMode::Capture(io) => io.x_in.clone(),
            GraphMode::Off => scratch.acquire(rows * hidden * 4).map_err(err)?,
        };

        // pipeline stages: the first one embeds, the others continue from the activations of the previous stage
        let first_stage = self.par.as_ref().is_none_or(|p| p.is_first_stage());
        let last_stage = self.par.as_ref().is_none_or(|p| p.is_last_stage());
        let mut host_x = if first_stage { self.embed_rows(ids)? } else { self.par_recv_activations(rows)? };
        if first_stage {
            if cfg.embedding_scale != 1.0 {
                for v in &mut host_x {
                    *v *= cfg.embedding_scale;
                }
            }
            for (r, embedding) in overrides {
                host_x[*r * hidden..(*r + 1) * hidden].copy_from_slice(embedding);
            }
            if let Some(table) = &self.position_table {
                let table_rows = table.len() / hidden;
                for (r, &pos) in positions.iter().enumerate() {
                    let idx = pos + cfg.position_offset;
                    if idx >= table_rows {
                        return Err("position is beyond the learned position embeddings".into());
                    }
                    for (v, p) in host_x[r * hidden..(r + 1) * hidden]
                        .iter_mut()
                        .zip(&table[idx * hidden..(idx + 1) * hidden])
                    {
                        *v += p;
                    }
                }
            }
            if let Some(norm) = &self.embd_norm {
                self.norm(&mut host_x, norm);
            }
            if cfg.embedding_norm {
                rms(&mut host_x, &vec![1.0; hidden], cfg.eps);
            }
            self.stash_embeddings(&host_x);
        }
        if !recording {
            stream
                .upload_activations(&mut x_buf, &host_x)
                .map_err(err)?;
        }
        // hyper-connections: the residual stream is `hc` copies of the embedding, kept in a buffer of its own (`x_buf` carries the block inputs)
        let hc_dev = dev_weights.hc.as_ref().filter(|_| self.hc_device && !recording);
        let mut hc_streams: Option<crate::backend::DeviceBuffer> = None;
        if let (Some(_), Some(h)) = (hc_dev, self.hc_params()) {
            let mut expanded = Vec::with_capacity(rows * h.hc_mult * hidden);
            for r in 0..rows {
                for _ in 0..h.hc_mult {
                    expanded.extend_from_slice(&host_x[r * hidden..(r + 1) * hidden]);
                }
            }
            let mut buf = scratch.acquire(expanded.len() * 4).map_err(err)?;
            stream.upload(&mut buf, cast_f32_to_u8(&expanded)).map_err(err)?;
            hc_streams = Some(buf);
        }
        let mut hc_block: Option<HcBlock> = None;
        // Gemma per-layer inputs (rows x layers x width) from the token ids and the scaled embeddings; the large projection runs on the device
        let ple_host = match (self.ple.as_ref(), dev_weights.ple_proj.as_ref()) {
            (Some(global), Some(proj)) if !recording => {
                let total = cfg.layers * global.dim;
                let mut projected_buf = scratch.acquire(rows * total * 4).map_err(err)?;
                stream.quantized_linear(&mut projected_buf, &x_buf, proj, proj.bias.as_ref()).map_err(err)?;
                let mut projected = vec![0.0f32; rows * total];
                stream.download(cast_f32_to_u8_mut(&mut projected), &projected_buf).map_err(err)?;
                scratch.recycle(projected_buf);
                self.ple_inputs_from(ids, projected)?
            }
            (Some(_), _) => self.ple_inputs(ids, &host_x)?,
            _ => None,
        };
        // every layer's slice of them goes up now, while the device has nothing queued: an upload between the layers would wait for the
        // work recorded before it
        let mut ple_bufs: Vec<Option<crate::backend::DeviceBuffer>> = Vec::new();
        if let (Some(inputs), Some(global)) = (ple_host.as_deref(), self.ple.as_ref()) {
            let width = global.dim;
            let total = cfg.layers * width;
            for (i, dev_layer) in dev_weights.layers.iter().enumerate() {
                if dev_layer.ple.is_none() || dev_layer.host_layer {
                    ple_bufs.push(None);
                    continue;
                }
                let mut layer_in = vec![0.0f32; rows * width];
                for r in 0..rows {
                    layer_in[r * width..(r + 1) * width].copy_from_slice(&inputs[r * total + i * width..r * total + (i + 1) * width]);
                }
                let mut buf = scratch.acquire(rows * width * 4).map_err(err)?;
                stream.upload(&mut buf, cast_f32_to_u8(&layer_in)).map_err(err)?;
                ple_bufs.push(Some(buf));
            }
        }
        if !recording {
            debug_sum(&mut **stream, "embd", &x_buf);
        }

        let layer_check = std::env::var_os("FLLM_LAYER_CHECK").is_some() && !recording;
        let delta_shape = self.delta_shape();
        let mamba_shape = self.mamba_shape();
        let mamba1_shape = self.mamba1_shape();
        let rwkv_shape = self.rwkv_shape();
        let kda_shape = self.kda_shape();
        // RWKV-7: the values of layer 0 of this pass, read by the value residual of every later layer
        let mut rwkv_v_first: Option<crate::backend::DeviceBuffer> = None;
        let mut delta_plan: Option<Vec<DeltaPlanRun>> = None;
        // the norm of the next layer (or the final norm), when the residual add of this layer's feed-forward block already computed it
        let mut carried_norm: Option<crate::backend::DeviceBuffer> = None;
        // LongCat-Flash: the shortcut experts of the layer before, added after the feed-forward block of the layer that follows
        let mut pending_shortcut_dev: Option<crate::backend::DeviceBuffer> = None;
        // FLLM_LAYER_CHECK: the input of the layer that made the shortcut experts, for the host run of the pair
        let mut check_pair: Option<Vec<f32>> = None;
        // selection of the lightning indexer of the last layer that has one (the layers after it without an indexer reuse it)
        let mut dsa_sel: Option<crate::backend::DeviceBuffer> = None;
        // the head GEMM reads the final norm output directly when every row's logits are wanted in order
        let head_direct = head_copy_fused() && logit_rows.len() == rows && logit_rows.iter().enumerate().all(|(j, &r)| j == r);
        // FLLM_ACT_CHECK=1: the residual stream is inspected after every layer of the first passes (largest value, NaN / Inf)
        let act_check = !recording && act_check_pass();
        let mut act_stats: Vec<(f32, u32)> = Vec::new();
        let stage = self.stage_range();
        // a pipeline stage's device cache holds its own layers only: layer `i` is `i - kv_base` there
        let kv_base = self.kv_base();
        // tensor parallelism: the output and down projections give partial sums; the fused residual-add kernels cannot take them
        let tp = self.par.as_ref().is_some_and(|p| p.tp_size() > 1);
        if self.v4_device {
            let streams = hc_streams.as_mut().ok_or("DeepSeek-V4 needs the residual streams on the device")?;
            self.v4_device_layers(&mut **stream, &mut scratch, dev_weights, streams, &mut x_buf, ids, &positions, tables, lengths)?;
        }
        for (i, layer) in self.layers.iter().enumerate().filter(|_| !self.v4_device) {
            if !stage.contains(&i) {
                continue;
            }
            let kvi = i - kv_base;
            let dev_layer = &dev_weights.layers[i];
            let layer_heads = cfg.layer_heads(i);
            let layer_kv_heads = cfg.layer_kv_heads(i);
            let dk = cfg.layer_head_dim(i);
            let dv = cfg.layer_value_dim(i);
            if dev_layer.host_layer {
                // no kernels for this layer: the host reference code runs it (host KV cache, host recurrent state)
                let mut host_x = vec![0.0f32; rows * hidden];
                stream.download(cast_f32_to_u8_mut(&mut host_x), &x_buf).map_err(err)?;
                let mut trace = crate::cpu_profile::Trace::new(false, rows);
                self.host_layers(i..i + 1, &mut host_x, &positions, tables, lengths, key, value, &mut trace, ple_host.as_deref(), &mut Vec::new())?;
                stream.upload(&mut x_buf, cast_f32_to_u8(&host_x)).map_err(err)?;
                continue;
            }
            // FLLM_LAYER_CHECK=1: every device layer is also run by the host reference on the same input (debugging)
            let check_in = if layer_check {
                let mut host_x = vec![0.0f32; rows * hidden];
                stream.download(cast_f32_to_u8_mut(&mut host_x), &x_buf).map_err(err)?;
                Some(host_x)
            } else {
                None
            };
            if let Some(ink) = &dev_layer.ink {
                // Inkling: the whole layer (attention with convolutions, feed-forward) is one device function; the convolution histories are recurrent state
                if delta_plan.is_none() {
                    delta_plan = Some(self.state_plan(cfg.layers, self.ink_layer_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.inkling_layer_device(&mut **stream, &mut scratch, i, kvi, dev_layer, ink, &mut x_buf, (&slots, tables, lengths), plan, dev_kv)?;
                if let Some(host_in) = check_in {
                    self.layer_check(&mut **stream, (i, i), &x_buf, host_in, &positions, tables, lengths, key, value)?;
                }
                continue;
            }
            if let (Some(hcd), Some(streams)) = (hc_dev, hc_streams.as_ref()) {
                hc_block = Some(self.hc_enter(&mut **stream, &mut scratch, &hcd.attn[i], &hcd.ones, streams, &mut x_buf, rows)?);
            }
            let mut normed_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
            // parallel residual: the attention branch joins the residual stream only after the feed-forward input is taken from it
            let parallel = cfg.parallel_residual;
            let mut deferred_branch: Option<crate::backend::DeviceBuffer> = None;
            // the feed-forward norm of this layer, when the output projection's residual add already computed it
            let mut ffn_pre: Option<crate::backend::DeviceBuffer> = None;

            // Zamba2: the shared block's output joins the stream in front of the layer's norm (the residual stream itself stays as it is)
            let zamba_in = match &dev_layer.zamba {
                Some(dz) => {
                    if let Some(carried) = carried_norm.take() {
                        scratch.recycle(carried);
                    }
                    let shared = self.zamba_device(&mut **stream, &mut scratch, dz, (i, kvi), layer, &x_buf, (&positions, &slots, tables, lengths), dev_kv, dev_weights)?;
                    let mut sum = scratch.acquire(rows * hidden * 4).map_err(err)?;
                    stream.copy_buffer(&mut sum, &x_buf).map_err(err)?;
                    stream.residual_add(&mut sum, &shared, 1.0).map_err(err)?;
                    scratch.recycle(shared);
                    Some(sum)
                }
                None => None,
            };
            if let Some(carried) = carried_norm.take() {
                scratch.recycle(std::mem::replace(&mut normed_buf, carried));
            } else if let Some(norm_buf) = &dev_layer.attn_norm {
                device_norm(&mut **stream, cfg.norm, &mut normed_buf, zamba_in.as_ref().unwrap_or(&x_buf), norm_buf, dev_layer.attn_norm_b.as_ref(), cfg.eps, self.groups)?;
                if let Some(sum) = zamba_in {
                    scratch.recycle(sum);
                }
            } else {
                stream.copy_buffer(&mut normed_buf, &x_buf).map_err(err)?;
            }

            if i == 0 {
                debug_sum(&mut **stream, "attn_norm-0", &normed_buf);
            }
            let std_qkv = match (&dev_layer.q, &dev_layer.k, &dev_layer.v) {
                (Some(q), Some(k), Some(v)) => Some((q, k, Some(v))),
                (Some(q), Some(k), None) if dev_layer.host_values || layer.k_eq_v => Some((q, k, None)),
                // the layer reads the cache of an earlier layer: only the query is projected (the key weight is a placeholder)
                (Some(q), None, None) if layer.kv_shared => Some((q, q, None)),
                // one matrix for all three: the merged path below replaces the individual projections
                _ if dev_layer.qkv.is_some() => dev_layer.qkv.as_ref().map(|m| (&m.weight, &m.weight, Some(&m.weight))),
                _ => None,
            };
            if std_qkv.is_some() || dev_layer.mla.is_some() {
                // the latent cache has its own buffers (device_mla_latent); these stay placeholders there
                let lat = self.mla_latent && dev_layer.mla.is_some();
                let (q_bytes, k_bytes, v_bytes) = if lat {
                    (4, 4, 4)
                } else {
                    (rows * layer_heads * dk * 4, rows * layer_kv_heads * dk * 4, rows * layer_kv_heads * dv * 4)
                };
                let mut q_buf = scratch.acquire(q_bytes).map_err(err)?;
                let mut k_buf = scratch.acquire(k_bytes).map_err(err)?;
                let mut v_buf = scratch.acquire(v_bytes).map_err(err)?;
                let mut latent_attn: Option<crate::backend::DeviceBuffer> = None;
                // normalized low-rank query of a layer with a lightning indexer (the indexer queries are projected from it)
                let mut q_low_dev: Option<crate::backend::DeviceBuffer> = None;

                let mut gate_buf = None;
                if let Some((q_w, k_w, v_w)) = std_qkv {
                // 4-bit q/k/v: the three GEMMs leave FP16 and one kernel adds the bias, rotates q and k and writes k and v into the cache
                let mut fused_qkv = false;
                let plain_kv = !layer.kv_shared && !layer.k_eq_v && dev_layer.v_norm.is_none() && cfg.value_scale == 1.0;
                if plain_kv
                    && cfg.rope.dims > 0
                    && cfg.layer_rotates(i)
                    && self.kv_kind == 1
                    && dev_layer.q_norm.is_none()
                    && dev_layer.k_norm.is_none()
                    && dev_layer.qkv.is_none()
                    && !layer.output_gate
                    && cfg.qkv_clip.is_none()
                    && q_w.quant_type.q4z_group().is_some()
                    && k_w.quant_type.q4z_group().is_some()
                    && v_w.is_some_and(|w| w.quant_type.q4z_group().is_some())
                {
                    let v_w = v_w.ok_or("missing value projection")?;
                    let ok = stream.linear_f16_out(&mut q_buf, &normed_buf, q_w).map_err(err)?
                        && stream.linear_f16_out(&mut k_buf, &normed_buf, k_w).map_err(err)?
                        && stream.linear_f16_out(&mut v_buf, &normed_buf, v_w).map_err(err)?;
                    if ok {
                        let (cos, sin) = &dev_weights.rope[layer.rope_table];
                        let rope_params = RopeParams {
                            heads: layer_heads,
                            kv_heads: layer_kv_heads,
                            dim: dk,
                            rope_dim: cfg.layer_rope_dims(i),
                            neox: cfg.rope.neox,
                            cos: Arc::clone(cos),
                            sin: Arc::clone(sin),
                        };
                        let mut q_rot = scratch.acquire(rows * cfg.heads * dk * 4).map_err(err)?;
                        fused_qkv = stream
                            .rope_kv_write_f16(
                                &mut q_rot,
                                &q_buf,
                                &k_buf,
                                &v_buf,
                                [q_w.bias.as_ref(), k_w.bias.as_ref(), v_w.bias.as_ref()],
                                dev_kv,
                                kvi,
                                &positions,
                                &slots,
                                &rope_params,
                            )
                            .map_err(err)?;
                        if !fused_qkv {
                            return Err("the fused rope and cache write failed after the q/k/v GEMMs left half precision".into());
                        }
                        // q_buf (the half-precision q) is not needed any more; attention reads the rotated q
                        scratch.recycle(std::mem::replace(&mut q_buf, q_rot));
                    }
                }
                // merged q/k/v (FLLM_MERGE_PROJ): one GEMM; the blocks are rotated and cached by one kernel when it covers the layer, else split into dense arrays
                let mut merged_done = false;
                let mut merged_fused = false;
                if let Some(mq) = &dev_layer.qkv {
                    let (nq, nk, nv) = (mq.parts[0], mq.parts[1], mq.parts[2]);
                    let can_fuse = plain_kv
                        && cfg.rope.dims > 0
                        && cfg.layer_rotates(i)
                        && self.kv_kind == 1
                        && !cfg.qk_norm_after_rope
                        && (qk_norm_rope_fusion() || (dev_layer.q_norm.is_none() && dev_layer.k_norm.is_none()))
                        && dev_layer.q_norm.as_ref().is_none_or(|w| w.size == dk * 4)
                        && dev_layer.k_norm.as_ref().is_none_or(|w| w.size == dk * 4)
                        && dev_layer.q_norm.is_some() == dev_layer.k_norm.is_some()
                        && std::env::var_os("FLLM_DEBUG_SUMS").is_none();
                    let mut qkv = scratch.acquire(rows * (nq + nk + nv) * 4).map_err(err)?;
                    stream
                        .quantized_linear(&mut qkv, &normed_buf, &mq.weight, if can_fuse { None } else { mq.weight.bias.as_ref() })
                        .map_err(err)?;
                    if can_fuse {
                        let (cos, sin) = &dev_weights.rope[layer.rope_table];
                        let rope_params = RopeParams {
                            heads: layer_heads,
                            kv_heads: layer_kv_heads,
                            dim: dk,
                            rope_dim: cfg.layer_rope_dims(i),
                            neox: cfg.rope.neox,
                            cos: Arc::clone(cos),
                            sin: Arc::clone(sin),
                        };
                        merged_fused = stream
                            .rope_kv_write_qkv(
                                &mut q_buf,
                                &qkv,
                                [nq, nk, nv],
                                mq.weight.bias.as_ref(),
                                dev_layer.q_norm.as_ref(),
                                dev_layer.k_norm.as_ref(),
                                cfg.eps,
                                dev_kv,
                                kvi,
                                &positions,
                                &slots,
                                &rope_params,
                            )
                            .map_err(err)?;
                        if !merged_fused {
                            return Err("the fused rope and cache write declined after the merged q/k/v GEMM".into());
                        }
                    } else if !stream.split_columns([&mut q_buf, &mut k_buf, &mut v_buf], &qkv, rows, [nq, nk, nv]).map_err(err)? {
                        return Err("the backend cannot split a merged q/k/v result".into());
                    }
                    scratch.recycle(qkv);
                    merged_done = true;
                }
                // gated attention: the projection holds queries and gates; attention runs on the queries
                if !fused_qkv {
                if merged_done {
                    // q, k and v are ready
                } else if layer.output_gate {
                    let mut projected = scratch.acquire(rows * cfg.heads * dk * 8).map_err(err)?;
                    let mut gate = scratch.acquire(rows * cfg.heads * dk * 4).map_err(err)?;
                    stream
                        .quantized_linear(&mut projected, &normed_buf, q_w, q_w.bias.as_ref())
                        .map_err(err)?;
                    stream
                        .split_q_gate(&mut q_buf, &mut gate, &projected, rows, cfg.heads, dk, cfg.arch.starts_with("qwen35"))
                        .map_err(err)?;
                    scratch.recycle(projected);
                    gate_buf = Some(gate);
                } else {
                stream
                    .quantized_linear(&mut q_buf, &normed_buf, q_w, q_w.bias.as_ref())
                    .map_err(err)?;
                }
                if !merged_done && !layer.kv_shared {
                stream
                    .quantized_linear(&mut k_buf, &normed_buf, k_w, k_w.bias.as_ref())
                    .map_err(err)?;
                }
                if merged_done {
                } else if let Some(v_w) = v_w {
                    stream
                        .quantized_linear(&mut v_buf, &normed_buf, v_w, v_w.bias.as_ref())
                        .map_err(err)?;
                } else if let Some(vm) = &layer.value_moe {
                    let mut host_normed = vec![0.0f32; rows * hidden];
                    stream.download(cast_f32_to_u8_mut(&mut host_normed), &normed_buf).map_err(err)?;
                    let values = self.value_moe_forward(vm, &host_normed)?;
                    stream.upload(&mut v_buf, cast_f32_to_u8(&values)).map_err(err)?;
                } else if layer.k_eq_v {
                    // the values are the keys before their norm and rotation
                    stream.copy_buffer(&mut v_buf, &k_buf).map_err(err)?;
                }
                if !merged_done {
                    // Gemma 4 / 3n: every value head is RMS-normalised without a weight
                    device_rms_inplace(&mut **stream, &mut scratch, &mut v_buf, dev_layer.v_norm.as_ref(), cfg.eps, 1)?;
                }
                // MiMo-V2: the values are scaled before they enter the cache
                if cfg.value_scale != 1.0 && !layer.kv_shared {
                    stream.scale_inplace(&mut v_buf, cfg.value_scale).map_err(err)?;
                }
                if let Some(clip) = cfg.qkv_clip {
                    for buf in [&mut q_buf, &mut k_buf, &mut v_buf] {
                        stream.clamp(buf, clip).map_err(err)?;
                    }
                }

                if i == 0 {
                    debug_sum(&mut **stream, "Qcur-0 (proj+bias)", &q_buf);
                    debug_sum(&mut **stream, "Kcur-0 (proj+bias)", &k_buf);
                    debug_sum(&mut **stream, "Vcur-0 (proj+bias)", &v_buf);
                }
                // QK norm, rotary embedding and cache write in one kernel (FLLM_FUSE_QK_NORM_ROPE; per-head norm weights, norm before the rotation)
                let mut qk_fused = merged_fused;
                if qk_norm_rope_fusion()
                    && !qk_fused
                    && !layer.kv_shared
                    && !cfg.qk_norm_after_rope
                    && cfg.rope.dims > 0
                    && cfg.layer_rotates(i)
                    && (dev_layer.q_norm.is_some() || dev_layer.k_norm.is_some())
                    && dev_layer.q_norm.as_ref().is_none_or(|w| w.size == dk * 4)
                    && dev_layer.k_norm.as_ref().is_none_or(|w| w.size == dk * 4)
                    && dev_layer.q_norm.is_some() == dev_layer.k_norm.is_some()
                    && std::env::var_os("FLLM_DEBUG_SUMS").is_none()
                {
                    let (cos, sin) = &dev_weights.rope[layer.rope_table];
                    let rope_params = RopeParams {
                        heads: layer_heads,
                        kv_heads: layer_kv_heads,
                        dim: dk,
                        rope_dim: cfg.layer_rope_dims(i),
                        neox: cfg.rope.neox,
                        cos: Arc::clone(cos),
                        sin: Arc::clone(sin),
                    };
                    qk_fused = stream
                        .rope_kv_write_qknorm(
                            &mut q_buf,
                            &k_buf,
                            &v_buf,
                            dev_layer.q_norm.as_ref(),
                            dev_layer.k_norm.as_ref(),
                            cfg.eps,
                            dev_kv,
                            kvi,
                            &positions,
                            &slots,
                            &rope_params,
                        )
                        .map_err(err)?;
                }
                if !cfg.qk_norm_after_rope && !qk_fused {
                    device_rms_inplace(&mut **stream, &mut scratch, &mut q_buf, dev_layer.q_norm.as_ref(), cfg.eps, 1)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut k_buf, dev_layer.k_norm.as_ref(), cfg.eps, 1)?;
                }
                // rotary embedding and the cache write in one kernel (f16 cache, nothing between them)
                let mut kv_written = qk_fused;
                if cfg.rope.dims > 0 && cfg.layer_rotates(i) && !qk_fused {
                    let (cos, sin) = &dev_weights.rope[layer.rope_table];
                    let rope_params = RopeParams {
                        heads: layer_heads,
                        kv_heads: layer_kv_heads,
                        dim: dk,
                        rope_dim: cfg.layer_rope_dims(i),
                        neox: cfg.rope.neox,
                        cos: Arc::clone(cos),
                        sin: Arc::clone(sin),
                    };
                    if let (Some(p3), Some(axis_tables)) = (&positions3, self.mrope_axis_tables()) {
                        // M-RoPE: one rope call per position axis, each rotating only the lanes of its axis
                        for (axis, (cos, sin)) in axis_tables.iter().enumerate() {
                            let axis_positions: Vec<usize> = p3.iter().map(|p| p[axis] as usize).collect();
                            let axis_params = RopeParams { cos: Arc::clone(cos), sin: Arc::clone(sin), ..rope_params.clone() };
                            stream.rope(&mut q_buf, &mut k_buf, &axis_positions, &axis_params).map_err(err)?;
                        }
                    } else {
                        if !cfg.qk_norm_after_rope && !layer.kv_shared && std::env::var_os("FLLM_DEBUG_SUMS").is_none() {
                            kv_written = stream
                                .rope_kv_write(&mut q_buf, &k_buf, &v_buf, dev_kv, kvi, &positions, &slots, &rope_params)
                                .map_err(err)?;
                        }
                        if !kv_written {
                            stream
                                .rope(&mut q_buf, &mut k_buf, &positions, &rope_params)
                                .map_err(err)?;
                        }
                    }
                }
                if i == 0 {
                    debug_sum(&mut **stream, "Qcur-0 (rope)", &q_buf);
                    debug_sum(&mut **stream, "Kcur-0 (rope)", &k_buf);
                }
                if cfg.qk_norm_after_rope {
                    device_rms_inplace(&mut **stream, &mut scratch, &mut q_buf, dev_layer.q_norm.as_ref(), cfg.eps, 1)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut k_buf, dev_layer.k_norm.as_ref(), cfg.eps, 1)?;
                }
                if !kv_written && cfg.layer_kv_source(i).is_none() {
                    stream
                        .kv_cache_write(dev_kv, kvi, &slots, &k_buf, &v_buf)
                        .map_err(err)?;
                }
                }
                } else if let (true, Some(dm), Some(mc)) = (lat, &dev_layer.mla, cfg.mla.as_ref()) {
                    latent_attn = Some(self.device_mla_latent(
                        &mut **stream,
                        &mut scratch,
                        dm,
                        mc,
                        layer,
                        i,
                        dev_weights,
                        dev_kv,
                        &normed_buf,
                        &positions,
                        &slots,
                        tables,
                        lengths,
                    )?);
                } else if let (Some(dm), Some(mc)) = (&dev_layer.mla, cfg.mla.as_ref()) {
                    // multi-head latent attention, expanded: projections as GEMMs, the data movement in two small kernels, then
                    // rotary embedding, cache write and attention like any other layer
                    let shape = crate::backend::MlaShape { heads: cfg.heads, nope: mc.nope, rope: mc.rope, value: mc.value, kv_lora: mc.kv_lora };
                    let mut q_full = scratch.acquire(rows * cfg.heads * dk * 4).map_err(err)?;
                    if let Some((down, norm)) = &dm.q_a {
                        let mut low = scratch.acquire(rows * down.rows * 4).map_err(err)?;
                        stream.quantized_linear(&mut low, &normed_buf, down, down.bias.as_ref()).map_err(err)?;
                        device_rms_inplace(&mut **stream, &mut scratch, &mut low, Some(norm), cfg.eps, 1)?;
                        stream.quantized_linear(&mut q_full, &low, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
                        if dev_layer.dsa.is_some() {
                            q_low_dev = Some(low);
                        } else {
                            scratch.recycle(low);
                        }
                    } else {
                        stream.quantized_linear(&mut q_full, &normed_buf, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
                    }
                    let width = mc.kv_lora + mc.rope;
                    let mut kv_a = scratch.acquire(rows * width * 4).map_err(err)?;
                    stream.quantized_linear(&mut kv_a, &normed_buf, &dm.kv_a, dm.kv_a.bias.as_ref()).map_err(err)?;
                    let mut latent = scratch.acquire(rows * mc.kv_lora * 4).map_err(err)?;
                    stream.mla_latent(&mut latent, &kv_a, rows, mc.kv_lora, width).map_err(err)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut latent, Some(&dm.kv_a_norm), cfg.eps, 1)?;
                    let mut kv = scratch.acquire(rows * cfg.heads * (mc.nope + mc.value) * 4).map_err(err)?;
                    let kv_b = dm.kv_b.as_ref().ok_or("the up projection of the expanded MLA cache is not on the device")?;
                    stream.quantized_linear(&mut kv, &latent, kv_b, kv_b.bias.as_ref()).map_err(err)?;
                    stream.mla_assemble(&mut q_buf, &mut k_buf, &mut v_buf, &q_full, &kv_a, &kv, rows, &shape).map_err(err)?;
                    for buf in [q_full, kv_a, latent, kv] {
                        scratch.recycle(buf);
                    }
                    let (cos, sin) = &dev_weights.rope[layer.rope_table];
                    let rope_params = RopeParams {
                        heads: cfg.heads,
                        kv_heads: cfg.kv_heads,
                        dim: dk,
                        rope_dim: mc.rope,
                        neox: cfg.rope.neox,
                        cos: Arc::clone(cos),
                        sin: Arc::clone(sin),
                    };
                    // a latent attention without rotary values (GLM-5-Next, Kimi-Linear, Kimi K3) stores the keys as assembled
                    if mc.rotary && mc.rope > 0 {
                        stream.rope(&mut q_buf, &mut k_buf, &positions, &rope_params).map_err(err)?;
                    }
                    stream.kv_cache_write(dev_kv, kvi, &slots, &k_buf, &v_buf).map_err(err)?;
                }

                if let Some(g_w) = &dev_layer.attn_gate {
                    let mut gate = scratch.acquire(rows * layer_heads * dv * 4).map_err(err)?;
                    stream.quantized_linear(&mut gate, &normed_buf, g_w, g_w.bias.as_ref()).map_err(err)?;
                    gate_buf = Some(gate);
                }
                let mut attn_buf = match latent_attn.take() {
                    Some(buf) => buf,
                    None => scratch.acquire(rows * layer_heads * dv * 4).map_err(err)?,
                };
                let attn_params = AttentionParams {
                    num_heads: layer_heads,
                    num_kv_heads: layer_kv_heads,
                    head_dim: dk,
                    value_dim: dv,
                    stride: self.stride,
                    block_size: self.block_size,
                    scale: cfg.attention_scale.unwrap_or(1.0 / (dk as f32).sqrt()),
                    softcap: cfg.attention_softcap,
                    window: layer.window,
                    chunk: layer.chunk,
                    sinks: dev_layer.sinks.clone(),
                    alibi_slopes: dev_weights.alibi.clone(),
                };
                if !lat {
                    let read_layer = crate::kv_geometry::KvPoolLayout::owner(cfg, i) - kv_base;
                    if let Some(m3) = &dev_layer.m3 {
                        self.minimax3_sparse_attention(&mut **stream, &mut scratch, m3, dev_weights, layer.rope_table, &normed_buf, &q_buf, &mut attn_buf, dev_kv, read_layer, &positions, &slots, tables, lengths, &attn_params)?;
                    } else if self.dsa_device && (dev_layer.dsa.is_some() || layer.dsa_shared) {
                        // sparse attention: the lightning indexer picks the positions of every row (its own layer's, or the layer before's), attention reads those only
                        let (ic, mc, dp) = (cfg.indexer.as_ref(), cfg.mla.as_ref(), self.dsa_params());
                        let (Some(ic), Some(mc), Some(dp)) = (ic, mc, dp) else { return Err("sparse attention without an indexer configuration".into()) };
                        if let Some(ds) = &dev_layer.dsa {
                            let low = q_low_dev.take().ok_or("the sparse-attention indexer needs the low-rank query")?;
                            let mut iq = scratch.acquire(rows * ic.heads * ic.dim * 4).map_err(err)?;
                            stream.quantized_linear(&mut iq, &low, &ds.q_b, ds.q_b.bias.as_ref()).map_err(err)?;
                            scratch.recycle(low);
                            let mut ik_raw = scratch.acquire(rows * ic.dim * 4).map_err(err)?;
                            stream.quantized_linear(&mut ik_raw, &normed_buf, &ds.k, ds.k.bias.as_ref()).map_err(err)?;
                            let mut ik = scratch.acquire(rows * ic.dim * 4).map_err(err)?;
                            stream.layer_norm(&mut ik, &ik_raw, &ds.k_norm, Some(&ds.k_norm_bias), ic.norm_eps).map_err(err)?;
                            scratch.recycle(ik_raw);
                            if ic.rope_last {
                                // the rotated values are the last ones of the head: move them to the front of queries and keys alike
                                let mut moved = scratch.acquire(rows * ic.dim * 4).map_err(err)?;
                                stream.rotate_heads(&mut moved, &ik, ic.dim, mc.rope).map_err(err)?;
                                scratch.recycle(std::mem::replace(&mut ik, moved));
                                let mut moved = scratch.acquire(rows * ic.heads * ic.dim * 4).map_err(err)?;
                                stream.rotate_heads(&mut moved, &iq, ic.dim, mc.rope).map_err(err)?;
                                scratch.recycle(std::mem::replace(&mut iq, moved));
                            }
                            let mut iw = scratch.acquire(rows * ic.heads * 4).map_err(err)?;
                            stream.quantized_linear(&mut iw, &normed_buf, &ds.proj, ds.proj.bias.as_ref()).map_err(err)?;
                            let (cos, sin) = &dev_weights.rope[layer.rope_table];
                            let rope_params = RopeParams { heads: ic.heads, kv_heads: 1, dim: ic.dim, rope_dim: mc.rope, neox: ic.neox, cos: Arc::clone(cos), sin: Arc::clone(sin) };
                            if mc.rotary && mc.rope > 0 {
                                stream.rope(&mut iq, &mut ik, &positions, &rope_params).map_err(err)?;
                            }
                            let mut sel = scratch.acquire(rows * dp.sel_width() * 4).map_err(err)?;
                            let mut store = ds.store.clone();
                            let gate = match &ds.pool {
                                Some((gate_w, _)) => {
                                    let mut g = scratch.acquire(rows * ic.dim * 4).map_err(err)?;
                                    stream.quantized_linear(&mut g, &normed_buf, gate_w, gate_w.bias.as_ref()).map_err(err)?;
                                    Some(g)
                                }
                                None => None,
                            };
                            stream.dsa_select(&mut sel, &mut store, &iq, &ik, gate.as_ref(), ds.pool.as_ref().map(|(_, ape)| ape), &iw, &slots, tables, lengths, &dp).map_err(err)?;
                            if let Some(g) = gate {
                                scratch.recycle(g);
                            }
                            for buf in [iq, ik, iw] {
                                scratch.recycle(buf);
                            }
                            if let Some(old) = dsa_sel.replace(sel) {
                                scratch.recycle(old);
                            }
                        }
                        let sel = dsa_sel.as_ref().ok_or_else(|| format!("layer {i} shares the sparse-attention selection of the layer before, which did not run"))?;
                        stream.dsa_attention(&mut attn_buf, &q_buf, dev_kv, tables, read_layer, sel, dp.sel_width(), &attn_params).map_err(err)?;
                    } else {
                        stream
                            .attention_paged(&mut attn_buf, &q_buf, dev_kv, tables, lengths, read_layer, &attn_params)
                            .map_err(err)?;
                    }
                }
                if let Some(gate) = gate_buf.take() {
                    match (layer.gate_per_head, cfg.attn_gate) {
                        // one gate value per head and row (Laguna, Step-3.5)
                        (true, crate::config::AttnGate::Sigmoid) => stream.head_gate_mul(&mut attn_buf, &gate, dv, None).map_err(err)?,
                        (true, crate::config::AttnGate::Softplus(beta)) => stream.head_gate_mul(&mut attn_buf, &gate, dv, Some(beta)).map_err(err)?,
                        (false, crate::config::AttnGate::Sigmoid) => stream.sigmoid_gate_mul(&mut attn_buf, &gate).map_err(err)?,
                        (false, crate::config::AttnGate::Softplus(beta)) => stream.softplus_gate_mul(&mut attn_buf, &gate, beta).map_err(err)?,
                    }
                    scratch.recycle(gate);
                }
                // BitNet: an RMS norm over all heads of the attention output, in front of the output projection
                device_rms_inplace(&mut **stream, &mut scratch, &mut attn_buf, dev_layer.attn_sub_norm.as_ref(), cfg.eps, 1)?;
                if i == 0 {
                    debug_sum(&mut **stream, "kqv_out-0", &attn_buf);
                }

                let mut out_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                let out_w = dev_layer.out.as_ref().ok_or("attention layer without a device output projection")?;
                // the feed-forward norm (and the activation format of its first GEMM) can join the residual add of the output projection
                let ffn_target = if !tp && hc_dev.is_none() && dev_layer.par_mamba.is_none() && dev_layer.post_attn_norm.is_none() && !parallel && !matches!(layer.ffn, Ffn::None) && matches!(cfg.norm, NormKind::Rms) && dev_layer.ffn_norm_b.is_none() && self.groups == 1 {
                    dev_layer
                        .ffn_norm
                        .as_ref()
                        .map(|w| (w, dev_layer.ffn_gate.as_ref().or(dev_layer.ffn_up.as_ref())))
                } else {
                    None
                };
                let mut fused = false;
                if let Some((norm_w, first)) = ffn_target {
                    let mut pre = scratch.acquire(rows * hidden * 4).map_err(err)?;
                    if stream
                        .quantized_linear_add_norm(&mut x_buf, &attn_buf, out_w, out_w.bias.as_ref(), cfg.residual_scale, norm_w, cfg.eps, &mut pre, first)
                        .map_err(err)?
                    {
                        ffn_pre = Some(pre);
                        fused = true;
                    } else {
                        scratch.recycle(pre);
                    }
                }
                // the output projection is added straight into the residual stream when nothing sits between them
                if !fused {
                    fused = !tp
                        && dev_layer.post_attn_norm.is_none()
                        && !parallel
                        && stream
                            .quantized_linear_add(&mut x_buf, &attn_buf, out_w, out_w.bias.as_ref(), cfg.residual_scale)
                            .map_err(err)?;
                }
                if !fused {
                    stream
                        .quantized_linear(
                            &mut out_buf,
                            &attn_buf,
                            out_w,
                            out_w.bias.as_ref(),
                        )
                        .map_err(err)?;
                    if tp {
                        self.device_tp_reduce(&mut **stream, &mut out_buf, rows * hidden)?;
                    }
                    if i == 0 {
                        debug_sum(&mut **stream, "attn_out-0", &out_buf);
                    }
                    device_post_norm_inplace(
                        &mut **stream,
                        &mut scratch,
                        cfg.norm,
                        &mut out_buf,
                        dev_layer.post_attn_norm.as_ref(),
                        dev_layer.post_attn_norm_b.as_ref(),
                        cfg.post_eps,
                        self.groups,
                    )?;
                    if parallel {
                        deferred_branch = Some(std::mem::replace(
                            &mut out_buf,
                            scratch.acquire(4).map_err(err)?,
                        ));
                    } else {
                        let mut joined = false;
                        if let Some((norm_w, first)) = ffn_target {
                            let mut pre = scratch.acquire(rows * hidden * 4).map_err(err)?;
                            if stream
                                .add_rms_norm(&mut x_buf, &out_buf, cfg.residual_scale, norm_w, cfg.eps, &mut pre, first)
                                .map_err(err)?
                            {
                                ffn_pre = Some(pre);
                                joined = true;
                            } else {
                                scratch.recycle(pre);
                            }
                        }
                        if !joined {
                            stream
                                .residual_add(&mut x_buf, &out_buf, cfg.residual_scale)
                                .map_err(err)?;
                        }
                    }
                }
                // Falcon-H1: the Mamba-2 branch reads the same normed input; both branches are added to the residual stream before the feed-forward norm
                if let (Some((dm, out_w)), Some(shape)) = (&dev_layer.par_mamba, &mamba_shape) {
                    if delta_plan.is_none() {
                        let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Mamba(_)) || l.parallel.is_some()).count();
                        delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                    }
                    let plan = delta_plan.as_deref().unwrap_or(&[]);
                    self.mamba_device(&mut **stream, &mut scratch, shape, dm, dev_layer, Some(out_w), plan, &normed_buf, &mut x_buf, rows)?;
                }
                if i == 0 {
                    debug_sum(&mut **stream, "ffn_inp-0", &x_buf);
                }

                scratch.recycle(q_buf);
                scratch.recycle(k_buf);
                scratch.recycle(v_buf);
                scratch.recycle(attn_buf);
                scratch.recycle(out_buf);
            } else if let (Some(dd), Some(shape)) = (&dev_layer.delta, &delta_shape) {
                if delta_plan.is_none() {
                    delta_plan = Some(self.delta_plan(shape, tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.delta_device(&mut **stream, &mut scratch, shape, dd, dev_layer, (i, plan), &normed_buf, &mut x_buf, rows)?;
            } else if let (Some(dm), Some(shape)) = (&dev_layer.mamba, &mamba_shape) {
                if delta_plan.is_none() {
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Mamba(_)) || l.parallel.is_some()).count();
                    delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.mamba_device(&mut **stream, &mut scratch, shape, dm, dev_layer, None, plan, &normed_buf, &mut x_buf, rows)?;
            } else if let (Some(dc), Some(sc)) = (&dev_layer.shortconv, cfg.shortconv.as_ref()) {
                if delta_plan.is_none() {
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::ShortConv(_))).count();
                    let bytes = (hidden * (sc.kernel - 1) * 4).next_multiple_of(256);
                    delta_plan = Some(self.state_plan(count, bytes, tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.shortconv_device(&mut **stream, &mut scratch, sc.kernel, dc, dev_layer, plan, &normed_buf, &mut x_buf, rows)?;
            } else if let (Some(dm), Some(shape)) = (&dev_layer.mamba1, &mamba1_shape) {
                if delta_plan.is_none() {
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Mamba1(_))).count();
                    delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.mamba1_device(&mut **stream, &mut scratch, shape, dm, dev_layer, plan, &normed_buf, &mut x_buf, rows)?;
            } else if let (Some(dk), Some(shape)) = (&dev_layer.kda, &kda_shape) {
                if delta_plan.is_none() {
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Kda(_))).count();
                    delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.kda_device(&mut **stream, &mut scratch, shape, dk, dev_layer, plan, &normed_buf, &mut x_buf, rows)?;
            } else if let (Some(dr), Some(shape)) = (&dev_layer.rwkv, &rwkv_shape) {
                if delta_plan.is_none() {
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Rwkv(_))).count();
                    delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.rwkv_device(&mut **stream, &mut scratch, shape, dr, dev_layer, (i, plan), &normed_buf, &mut x_buf, rows, &mut rwkv_v_first)?;
            } else if matches!(layer.attn, Attn::None) {
                // feed-forward-only layer: its norm output is the feed-forward input (below)
            } else {
                // Recurrent mixers keep their state on the host: they get the normed input
                // and return the branch output (not cache contents).
                let mut host_normed = vec![0.0f32; rows * hidden];
                stream
                    .download(cast_f32_to_u8_mut(&mut host_normed), &normed_buf)
                    .map_err(err)?;
                let branch = match &layer.attn {
                    Attn::Linear(d) => {
                        self.delta_forward(d, &layer.out, i, &host_normed, tables, lengths)?
                    }
                    Attn::Mamba(m) => {
                        self.mamba_forward(m, &layer.out, i, &host_normed, tables, lengths)?
                    }
                    Attn::Mamba1(m) => {
                        self.mamba1_forward(m, &layer.out, i, &host_normed, tables, lengths)?
                    }
                    _ => return Err("layer kind is not supported by the accelerator path".into()),
                };
                let mut branch_buf = scratch.acquire(branch.len() * 4).map_err(err)?;
                stream
                    .upload(&mut branch_buf, cast_f32_to_u8(&branch))
                    .map_err(err)?;
                stream
                    .residual_add(&mut x_buf, &branch_buf, cfg.residual_scale)
                    .map_err(err)?;
                scratch.recycle(branch_buf);
            }

            if matches!(layer.ffn, Ffn::None) {
                if let Some(branch) = deferred_branch.take() {
                    stream.residual_add(&mut x_buf, &branch, cfg.residual_scale).map_err(err)?;
                    scratch.recycle(branch);
                }
                if let Some(host_in) = check_in {
                    self.layer_check(&mut **stream, (i, i), &x_buf, host_in, &positions, tables, lengths, key, value)?;
                }
                scratch.recycle(normed_buf);
                continue;
            }
            if let (Some(hcd), Some(streams), Some(block)) = (hc_dev, hc_streams.as_mut(), hc_block.take()) {
                self.hc_leave(&mut **stream, &mut scratch, streams, &mut x_buf, block, rows, i, false)?;
                hc_block = Some(self.hc_enter(&mut **stream, &mut scratch, &hcd.ffn[i], &hcd.ones, streams, &mut x_buf, rows)?);
            }
            let dev_ffn = match (&dev_layer.ffn_gate, &dev_layer.ffn_up, &dev_layer.ffn_down) {
                (g, Some(u), Some(d)) => Some((g.as_ref(), u, d)),
                // gate and up as one matrix: the merged branch below replaces the two projections
                (None, None, Some(d)) if dev_layer.gate_up.is_some() => dev_layer.gate_up.as_ref().map(|m| (Some(&m.weight), &m.weight, d)),
                _ => None,
            };
            let mut ffn_normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
            if matches!(layer.attn, Attn::None) || (parallel && dev_layer.ffn_norm.is_none()) {
                // feed-forward-only layers (Nemotron-H) and parallel-residual models with one shared norm (Cohere)
                stream.copy_buffer(&mut ffn_normed, &normed_buf).map_err(err)?;
            } else if let Some(pre) = ffn_pre.take() {
                scratch.recycle(std::mem::replace(&mut ffn_normed, pre));
            } else if let Some(ffn_norm_buf) = &dev_layer.ffn_norm {
                device_norm(&mut **stream, cfg.norm, &mut ffn_normed, &x_buf, ffn_norm_buf, dev_layer.ffn_norm_b.as_ref(), cfg.eps, self.groups)?;
            } else {
                stream.copy_buffer(&mut ffn_normed, &x_buf).map_err(err)?;
            }
            if let Some(branch) = deferred_branch.take() {
                stream.residual_add(&mut x_buf, &branch, cfg.residual_scale).map_err(err)?;
                scratch.recycle(branch);
            }
            if let Some(moe) = &layer.shortcut {
                let buf = self.shortcut_device(&mut **stream, &mut scratch, moe, dev_layer.moe.as_ref(), &ffn_normed, rows)?;
                if let Some(old) = pending_shortcut_dev.replace(buf) {
                    scratch.recycle(old);
                }
            }
            if i == 0 {
                debug_sum(&mut **stream, "attn_post_norm-0 (ffn input)", &ffn_normed);
            }

            if let Some((gate_w, up_w, down_w)) = dev_ffn {
                let ff = dev_layer.gate_up.as_ref().map_or(up_w.rows, |m| m.parts[0]);
                let mut gate_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let mut up_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let mut act_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let act = act_code(cfg.activation);

                if let Some(mg) = &dev_layer.gate_up {
                    // gate and up from one GEMM, the gated activation reads both blocks
                    let mut gu = scratch.acquire(rows * 2 * ff * 4).map_err(err)?;
                    stream.quantized_linear(&mut gu, &ffn_normed, &mg.weight, mg.weight.bias.as_ref()).map_err(err)?;
                    if !stream.glu_merged(&mut act_buf, &gu, rows, ff, cfg.activation == Activation::GeluTanh).map_err(err)? {
                        return Err("the backend cannot apply the gated activation to a merged gate/up result".into());
                    }
                    scratch.recycle(gu);
                } else if let Some(gate_w) = gate_w {
                // NVFP4 prompts and 4-bit Marlin: both GEMMs leave FP16 and the gated activation reads it (folded into the activation
                // quantization / conversion of the down projection)
                let mut half_glu = false;
                let dense_clamped = match self.dense_glu_mode() {
                    Some(GluMode::Clamped(kind)) => Some(kind),
                    _ => None,
                };
                // the clamped SwiGLU of gpt-oss and MiniMax-M3 (also in their dense layers)
                let dense_oai = match self.dense_glu_mode() {
                    Some(GluMode::SwigluOai(alpha, limit)) => Some((alpha, limit)),
                    _ => None,
                };
                if dense_clamped.is_none() && dense_oai.is_none()
                    && Self::half_out_weight(gate_w)
                    && matches!(cfg.activation, Activation::Silu | Activation::GeluTanh)
                    && !crate::backend::env_flag_once!("FLLM_NO_F16_FFN")
                    && gate_w.bias.is_none()
                    && dev_layer.ffn_sub_norm.is_none()
                    && up_w.bias.is_none()
                    && gate_w.quant_type == up_w.quant_type
                    && gate_w.rows == up_w.rows
                    && gate_w.cols == up_w.cols
                    && stream.linear_f16_out(&mut gate_buf, &ffn_normed, gate_w).map_err(err)?
                {
                    if !stream.linear_f16_out(&mut up_buf, &ffn_normed, up_w).map_err(err)? {
                        return Err("the up projection did not leave half precision like the gate projection".into());
                    }
                    half_glu = stream
                        .glu_f16(&mut act_buf, &gate_buf, &up_buf, cfg.activation == Activation::GeluTanh)
                        .map_err(err)?;
                    if !half_glu {
                        return Err("the gated activation of half-precision gate and up projections failed".into());
                    }
                }
                if !half_glu {
                stream
                    .quantized_linear(&mut gate_buf, &ffn_normed, gate_w, gate_w.bias.as_ref())
                    .map_err(err)?;
                stream
                    .quantized_linear(&mut up_buf, &ffn_normed, up_w, up_w.bias.as_ref())
                    .map_err(err)?;
                if let Some(kind) = dense_clamped {
                    stream.glu_clamped(&mut act_buf, &gate_buf, &up_buf, kind).map_err(err)?;
                } else if let Some((alpha, limit)) = dense_oai {
                    stream.swiglu_oai(&mut act_buf, &gate_buf, &up_buf, alpha, limit).map_err(err)?;
                } else {
                match cfg.activation {
                    Activation::GeluTanh => stream.geglu(&mut act_buf, &gate_buf, &up_buf).map_err(err)?,
                    Activation::Silu => stream.swiglu(&mut act_buf, &gate_buf, &up_buf).map_err(err)?,
                    _ => stream.activation_glu(&mut act_buf, &gate_buf, &up_buf, act).map_err(err)?,
                }
                }
                }
                } else {
                    // plain MLP: down(act(up(x)))
                    stream
                        .quantized_linear(&mut up_buf, &ffn_normed, up_w, up_w.bias.as_ref())
                        .map_err(err)?;
                    stream.activation(&mut act_buf, &up_buf, act).map_err(err)?;
                }

                // BitNet: an RMS norm over the activated product in front of the down projection
                device_rms_inplace(&mut **stream, &mut scratch, &mut act_buf, dev_layer.ffn_sub_norm.as_ref(), cfg.eps, 1)?;
                let mut ffn_out_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                // the norm of the next layer (or the final norm) can join the residual add of the down projection
                // (a layer that receives DeepStack features adds them after its residual, so the norm cannot join the add)
                let ds_here = deepstack.as_ref().is_some_and(|d| d.get(i).is_some_and(|level| !level.is_empty()));
                // (the shortcut experts of LongCat-Flash are added to the stream after this block: the norm computed in the add would miss them)
                let next_target = if !tp && hc_dev.is_none() && dev_layer.post_ffn_norm.is_none() && !parallel && !ds_here && !layer.takes_shortcut && matches!(cfg.norm, NormKind::Rms) && self.groups == 1 {
                    match dev_weights.layers.get(i + 1) {
                        Some(next) if !next.host_layer && next.attn_norm_b.is_none() => next.attn_norm.as_ref().map(|w| {
                            let first = if self.layers.get(i + 1).is_some_and(|l| matches!(l.attn, Attn::Std { .. })) { next.q.as_ref() } else { None };
                            (w, first)
                        }),
                        Some(_) => None,
                        None if dev_weights.final_norm_b.is_none() => Some((&dev_weights.final_norm, head_direct.then_some(&dev_weights.head))),
                        None => None,
                    }
                } else {
                    None
                };
                let mut fused = false;
                if let Some((norm_w, first)) = next_target {
                    let mut pre = scratch.acquire(rows * hidden * 4).map_err(err)?;
                    if stream
                        .quantized_linear_add_norm(&mut x_buf, &act_buf, down_w, down_w.bias.as_ref(), cfg.residual_scale, norm_w, cfg.eps, &mut pre, first)
                        .map_err(err)?
                    {
                        carried_norm = Some(pre);
                        fused = true;
                    } else {
                        scratch.recycle(pre);
                    }
                }
                if !fused {
                    fused = !tp
                        && dev_layer.post_ffn_norm.is_none()
                        && stream
                            .quantized_linear_add(&mut x_buf, &act_buf, down_w, down_w.bias.as_ref(), cfg.residual_scale)
                            .map_err(err)?;
                }
                if !fused {
                    stream
                        .quantized_linear(&mut ffn_out_buf, &act_buf, down_w, down_w.bias.as_ref())
                        .map_err(err)?;
                    if tp {
                        self.device_tp_reduce(&mut **stream, &mut ffn_out_buf, rows * hidden)?;
                    }
                    device_post_norm_inplace(
                        &mut **stream,
                        &mut scratch,
                        cfg.norm,
                        &mut ffn_out_buf,
                        dev_layer.post_ffn_norm.as_ref(),
                        dev_layer.post_ffn_norm_b.as_ref(),
                        cfg.post_eps,
                        self.groups,
                    )?;
                    let mut joined = false;
                    if let Some((norm_w, first)) = next_target {
                        let mut pre = scratch.acquire(rows * hidden * 4).map_err(err)?;
                        if stream
                            .add_rms_norm(&mut x_buf, &ffn_out_buf, cfg.residual_scale, norm_w, cfg.eps, &mut pre, first)
                            .map_err(err)?
                        {
                            carried_norm = Some(pre);
                            joined = true;
                        } else {
                            scratch.recycle(pre);
                        }
                    }
                    if !joined {
                        stream
                            .residual_add(&mut x_buf, &ffn_out_buf, cfg.residual_scale)
                            .map_err(err)?;
                    }
                }

                scratch.recycle(gate_buf);
                scratch.recycle(up_buf);
                scratch.recycle(act_buf);
                scratch.recycle(ffn_out_buf);
            } else {
                // Feed-forward kinds without device kernels (mixture of experts, plain MLPs) or whose weights stay in
                // host memory: large batches stream the weights to the device, the rest runs on the host from the
                // normed input and comes back as one residual contribution.
                let big = rows >= Self::host_stream_min_tokens();
                let mut streamed = None;
                let mut host_in = Vec::new();
                if let (Ffn::Moe(_), Some(dm)) = (&layer.ffn, dev_layer.moe.as_ref()) {
                    streamed = self.moe_fast(&mut **stream, &mut scratch, dm, &ffn_normed, rows)?;
                }
                if streamed.is_none() && (big || dev_layer.moe.is_some()) && !dev_layer.host_ffn {
                    match &layer.ffn {
                        Ffn::Gated { gate, up, down } if big => {
                            if let (Some(mode), Some(w)) =
                                (self.dense_glu_mode(), Self::gated_weights(&**stream, gate, up, down))
                            {
                                streamed = Some(Self::gated_run(
                                    &mut **stream,
                                    &mut scratch,
                                    mode,
                                    &ExpertSrc::Host(w),
                                    &ffn_normed,
                                    rows,
                                )?);
                            }
                        }
                        Ffn::Moe(moe) => {
                            host_in = vec![0.0f32; rows * hidden];
                            stream
                                .download(cast_f32_to_u8_mut(&mut host_in), &ffn_normed)
                                .map_err(err)?;
                            streamed = self.moe_on_device(
                                &mut **stream,
                                &mut scratch,
                                moe,
                                dev_layer.moe.as_ref(),
                                &ffn_normed,
                                &host_in,
                                rows,
                            )?;
                        }
                        _ => {}
                    }
                }
                let mut ffn_out_buf = match streamed {
                    Some(buf) => buf,
                    None => {
                        if host_in.is_empty() {
                            host_in = vec![0.0f32; rows * hidden];
                            stream
                                .download(cast_f32_to_u8_mut(&mut host_in), &ffn_normed)
                                .map_err(err)?;
                        }
                        let mut workspace = compute::ProjectionGroupWorkspace::default();
                        let host_out = self.ffn_forward(layer, &host_in, &mut workspace)?;
                        let mut buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                        stream
                            .upload(&mut buf, cast_f32_to_u8(&host_out))
                            .map_err(err)?;
                        buf
                    }
                };
                if i == 0 {
                    debug_sum(&mut **stream, "ffn_moe_out-0", &ffn_out_buf);
                }
                device_post_norm_inplace(
                    &mut **stream,
                    &mut scratch,
                    cfg.norm,
                    &mut ffn_out_buf,
                    dev_layer.post_ffn_norm.as_ref(),
                    dev_layer.post_ffn_norm_b.as_ref(),
                    cfg.post_eps,
                    self.groups,
                )?;
                stream
                    .residual_add(&mut x_buf, &ffn_out_buf, cfg.residual_scale)
                    .map_err(err)?;
                scratch.recycle(ffn_out_buf);
            }
            if layer.takes_shortcut {
                if let Some(buf) = pending_shortcut_dev.take() {
                    stream.residual_add(&mut x_buf, &buf, 1.0).map_err(err)?;
                    scratch.recycle(buf);
                }
            }
            if let (Some(streams), Some(block)) = (hc_streams.as_mut(), hc_block.take()) {
                self.hc_leave(&mut **stream, &mut scratch, streams, &mut x_buf, block, rows, i, true)?;
            }
            if let Some(level) = deepstack.as_ref().and_then(|d| d.get(i)).filter(|level| !level.is_empty()) {
                let mut added = vec![0.0f32; rows * hidden];
                for (row, values) in level {
                    added[row * hidden..(row + 1) * hidden].copy_from_slice(values);
                }
                let mut buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                stream.upload(&mut buf, cast_f32_to_u8(&added)).map_err(err)?;
                stream.residual_add(&mut x_buf, &buf, 1.0).map_err(err)?;
                scratch.recycle(buf);
            }
            // Gemma per-layer input path: x += norm(proj(act(gate(x)) * input of this layer)), then the layer output scale
            if let (Some((gate_w, proj_w, norm_w)), Some(input_buf), Some(global)) = (&dev_layer.ple, ple_bufs.get_mut(i).and_then(Option::take), self.ple.as_ref()) {
                let width = global.dim;
                let mut gated = scratch.acquire(rows * width * 4).map_err(err)?;
                stream.quantized_linear(&mut gated, &x_buf, gate_w, gate_w.bias.as_ref()).map_err(err)?;
                let mut act = scratch.acquire(rows * width * 4).map_err(err)?;
                match cfg.activation {
                    Activation::GeluTanh => stream.geglu(&mut act, &gated, &input_buf).map_err(err)?,
                    Activation::Silu => stream.swiglu(&mut act, &gated, &input_buf).map_err(err)?,
                    other => stream.activation_glu(&mut act, &gated, &input_buf, act_code(other)).map_err(err)?,
                }
                let mut back = scratch.acquire(rows * hidden * 4).map_err(err)?;
                stream.quantized_linear(&mut back, &act, proj_w, proj_w.bias.as_ref()).map_err(err)?;
                device_post_norm_inplace(&mut **stream, &mut scratch, cfg.norm, &mut back, Some(norm_w), None, cfg.post_eps, self.groups)?;
                stream.residual_add(&mut x_buf, &back, 1.0).map_err(err)?;
                for buf in [input_buf, gated, act, back] {
                    scratch.recycle(buf);
                }
            }
            if let Some(scale) = dev_layer.out_scale {
                let mut copy = scratch.acquire(rows * hidden * 4).map_err(err)?;
                stream.copy_buffer(&mut copy, &x_buf).map_err(err)?;
                stream.residual_add(&mut x_buf, &copy, scale - 1.0).map_err(err)?;
                scratch.recycle(copy);
            }
            if i == 0 || crate::backend::env_flag_once!("FLLM_DEBUG_LAYERS") {
                debug_sum(&mut **stream, &format!("l_out-{i}"), &x_buf);
            }
            if act_check {
                if let Some(stat) = stream.check_activations(&x_buf, rows * hidden).map_err(err)? {
                    act_stats.push(stat);
                }
            }
            if let Some(host_in) = check_in {
                // LongCat-Flash: the shortcut experts of a layer join the stream after the layer that follows, so that one is checked together with it
                let (first, host_in) = match (layer.takes_shortcut, check_pair.take()) {
                    (true, Some(previous)) => (i - 1, previous),
                    _ => (i, host_in),
                };
                if layer.shortcut.is_some() {
                    check_pair = Some(host_in.clone());
                }
                self.layer_check(&mut **stream, (first, i), &x_buf, host_in, &positions, tables, lengths, key, value)?;
            }
            scratch.recycle(ffn_normed);
            scratch.recycle(normed_buf);
        }
        if act_check {
            report_act_check(&mut **stream, rows, &act_stats);
        }
        if let Some(sel) = dsa_sel.take() {
            scratch.recycle(sel);
        }
        if let (Some(hcd), Some(streams)) = (hc_dev, hc_streams.take()) {
            self.hc_head(&mut **stream, &mut scratch, hcd, &streams, &mut x_buf, rows)?;
            scratch.recycle(streams);
        }

        if !last_stage {
            // not the last pipeline stage: the residual stream goes down to the host (no peer path between devices of different
            // backends) and on to the next stage, which uploads it to its own device
            if let Some(carried) = carried_norm.take() {
                scratch.recycle(carried);
            }
            let mut handover = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut handover), &x_buf).map_err(err)?;
            scratch.recycle(x_buf);
            self.par_send_activations(&handover)?;
            return Ok(Vec::new());
        }

        let mut final_normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
        if let Some(carried) = carried_norm.take() {
            scratch.recycle(std::mem::replace(&mut final_normed, carried));
        } else {
            device_norm(
                &mut **stream,
                cfg.norm,
                &mut final_normed,
                &x_buf,
                &dev_weights.final_norm,
                dev_weights.final_norm_b.as_ref(),
                cfg.eps,
                self.groups,
            )?;
        }

        // narrow activations: the hidden-state capture, the embeddings API and the output head work on FP32 rows
        if stream.act_dtype().is_narrow() && (capture.is_some() || embeddings || !logit_rows.is_empty()) {
            let mut wide = scratch.acquire(rows * hidden * 4).map_err(err)?;
            stream.widen_activations(&mut wide, &final_normed).map_err(err)?;
            scratch.recycle(std::mem::replace(&mut final_normed, wide));
        }

        if capture.is_some() || embeddings {
            let mut host_out = vec![0.0f32; rows * hidden];
            stream
                .download(cast_f32_to_u8_mut(&mut host_out), &final_normed)
                .map_err(err)?;
            if let Some(cap) = capture {
                cap.clear();
                cap.extend_from_slice(&host_out);
            }
            if embeddings {
                scratch.recycle(x_buf);
                scratch.recycle(final_normed);
                return Ok(host_out);
            }
        }

        if logit_rows.is_empty() {
            scratch.recycle(x_buf);
            scratch.recycle(final_normed);
            return Ok(Vec::new());
        }

        // Project only the requested rows: gather them into one compact buffer first.
        let selected = logit_rows.len();
        let picked = if head_direct {
            None
        } else {
            let picked = scratch.acquire(selected * hidden * 4).map_err(err)?;
            for (j, &r) in logit_rows.iter().enumerate() {
                let src = final_normed
                    .slice(r * hidden * 4, hidden * 4)
                    .map_err(err)?;
                let mut dst = picked.slice(j * hidden * 4, hidden * 4).map_err(err)?;
                stream.copy_buffer(&mut dst, &src).map_err(err)?;
            }
            Some(picked)
        };
        // a tensor-parallel rank holds a slice of the vocabulary rows
        let head_rows = dev_weights.head.rows;
        let mut head_out = scratch.acquire(selected * head_rows * 4).map_err(err)?;
        let head = &dev_weights.head;
        stream
            .quantized_linear_wide(&mut head_out, picked.as_ref().unwrap_or(&final_normed), head, head.bias.as_ref())
            .map_err(err)?;

        if let GraphMode::Capture(io) = &mut graph {
            // the logits stay in the graph's buffer; the caller downloads them (or takes their argmax) after every launch
            io.head_out = Some(head_out);
            if let Some(picked) = picked {
                scratch.recycle(picked);
            }
            scratch.recycle(final_normed);
            return Ok(Vec::new());
        }
        if self.par.is_none() && cfg.logit_scale == 1.0 && cfg.logit_softcap.is_none() && run_device_sample(&mut **stream, &head_out, cfg.vocab, selected) {
            scratch.recycle(head_out);
            if let Some(picked) = picked {
                scratch.recycle(picked);
            }
            scratch.recycle(x_buf);
            scratch.recycle(final_normed);
            return Ok(Vec::new());
        }
        if self.par.is_none() && GREEDY_IDS_REQUEST.swap(false, std::sync::atomic::Ordering::Relaxed) && cfg.logit_scale > 0.0 && cfg.logit_softcap.is_none() {
            // argmax commutes with a positive scale and with the monotonic soft cap; only the ids travel to the host
            if let Ok(ids) = stream.argmax_rows(&head_out, selected, cfg.vocab) {
                *GREEDY_IDS.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ids);
                scratch.recycle(head_out);
                if let Some(picked) = picked {
                    scratch.recycle(picked);
                }
                scratch.recycle(x_buf);
                scratch.recycle(final_normed);
                return Ok(Vec::new());
            }
        }
        let mut logits = vec![0.0f32; selected * head_rows];
        stream
            .download(cast_f32_to_u8_mut(&mut logits), &head_out)
            .map_err(err)?;

        scratch.recycle(head_out);
        if let Some(picked) = picked {
            scratch.recycle(picked);
        }
        scratch.recycle(x_buf);
        scratch.recycle(final_normed);
        let mut logits = self.par_head_gather(logits, selected)?;

        if cfg.logit_scale != 1.0 {
            for l in &mut logits {
                *l *= cfg.logit_scale;
            }
        }
        if let Some(cap) = cfg.logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }
        Ok(logits)
    }

    /// Attention for configurations the tiled kernel does not cover (sliding window, soft
    /// capping, custom scale, different key/value widths). One row at a time, in parallel.
    #[allow(clippy::too_many_arguments)]
    fn attention_reference(
        &self,
        layer: usize,
        q: &[f32],
        key: &[u16],
        value: &[u16],
        tables: &[i64],
        lengths: &[i64],
    ) -> Vec<f32> {
        self.attention_selected(layer, q, key, value, tables, lengths, None)
    }

    /// `attention_reference` over a chosen set of tokens per row (`select[r]`, sorted positions; `None`: all visible tokens).
    #[allow(clippy::too_many_arguments)]
    fn attention_selected(
        &self,
        layer: usize,
        q: &[f32],
        key: &[u16],
        value: &[u16],
        tables: &[i64],
        lengths: &[i64],
        select: Option<&[Option<Vec<u32>>]>,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let (dk, dv, heads, kv_heads) = (cfg.layer_head_dim(layer), cfg.layer_value_dim(layer), cfg.layer_heads(layer), cfg.layer_kv_heads(layer));
        let group = heads / kv_heads;
        let scale = cfg.attention_scale.unwrap_or(1.0 / (dk as f32).sqrt());
        let (fmt_k, fmt_v) = self.kv_formats();
        let (key_region, value_region) = (self.key_region(layer), self.value_region(layer));
        let window = self.layers[layer].window;
        let chunk = self.layers[layer].chunk;
        let vis_override = self.row_visible_lengths();
        let mut out = vec![0.0f32; lengths.len() * heads * dv];
        out.par_chunks_mut(heads * dv)
            .enumerate()
            .for_each(|(r, output)| {
                let n = vis_override.as_ref().and_then(|v| v.get(r).copied()).unwrap_or(lengths[r] as usize);
                let start = match chunk {
                    Some(c) => (n - 1) / c * c,
                    None => window.map_or(0, |w| n.saturating_sub(w)),
                };
                let chosen: Option<&Vec<u32>> = select.and_then(|s| s.get(r)).and_then(Option::as_ref);
                let visible: Vec<usize> = match chosen {
                    Some(list) => list.iter().map(|&p| p as usize).filter(|&p| p >= start && p < n).collect(),
                    None => (start..n).collect(),
                };
                let count = visible.len();
                let mut k_rows = vec![0.0f32; count * kv_heads * dk];
                let mut v_rows = vec![0.0f32; count * kv_heads * dv];
                for (i, &p) in visible.iter().enumerate() {
                    let block = tables[r * self.stride + p / self.block_size] as usize;
                    let slot = block * self.block_size + p % self.block_size;
                    self.read_slot(
                        key,
                        key_region,
                        slot,
                        fmt_k,
                        &mut k_rows[i * kv_heads * dk..(i + 1) * kv_heads * dk],
                    );
                    self.read_slot(
                        value,
                        value_region,
                        slot,
                        fmt_v,
                        &mut v_rows[i * kv_heads * dv..(i + 1) * kv_heads * dv],
                    );
                }
                let mut scores = vec![0.0f32; count];
                for h in 0..heads {
                    let kh = h / group;
                    let query = &q[(r * heads + h) * dk..(r * heads + h + 1) * dk];
                    for (i, score) in scores.iter_mut().enumerate() {
                        let p = visible[i];
                        if let Some(mask) = &self.tree_mask {
                            if p >= self.tree_prefix_len && r >= self.tree_prefix_len {
                                let qi = r - self.tree_prefix_len;
                                let ki = p - self.tree_prefix_len;
                                if qi < self.tree_num_nodes && ki < self.tree_num_nodes && !mask[qi * self.tree_num_nodes + ki] {
                                    *score = f32::NEG_INFINITY;
                                    continue;
                                }
                            }
                        }
                        let k = &k_rows[(i * kv_heads + kh) * dk..(i * kv_heads + kh + 1) * dk];
                        let mut s = compute::dot(query, k) * scale;
                        if let Some(slopes) = &cfg.alibi_slopes {
                            s += slopes[h] * (p as f32 - (n - 1) as f32);
                        }
                        if let Some(cap) = cfg.attention_softcap {
                            s = cap * (s / cap).tanh();
                        }
                        *score = s;
                    }
                    let sink = self.layers[layer].sinks.as_ref().map(|s| s[h]);
                    let max = scores
                        .iter()
                        .copied()
                        .fold(sink.unwrap_or(f32::NEG_INFINITY), f32::max);
                    let mut sum = sink.map_or(0.0, |s| (s - max).exp());
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let head_out = &mut output[h * dv..(h + 1) * dv];
                    for (i, s) in scores.iter().enumerate() {
                        let v = &v_rows[(i * kv_heads + kh) * dv..(i * kv_heads + kh + 1) * dv];
                        let w = s / sum;
                        for (o, x) in head_out.iter_mut().zip(v) {
                            *o += w * x;
                        }
                    }
                }
            });
        out
    }

    /// Queries, keys and values of one MLA layer for a batch of tokens, with rotary embeddings
    /// applied. Each head is laid out `[rotary | content]` for queries and keys; the dot product
    /// does not depend on the order as long as both sides agree.
    fn mla_qkv_low(
        &self,
        m: &Mla,
        x: &[f32],
        positions: &[usize],
        table: &RopeTables,
        keep_low: bool,
    ) -> Result<(MlaQkv, Option<Vec<f32>>), String> {
        let cfg = &self.cfg;
        let mc = cfg
            .mla
            .as_ref()
            .ok_or("MLA layer without MLA configuration")?;
        let rows = positions.len();
        let heads = cfg.heads;
        let (nope, rope, value) = (mc.nope, mc.rope, mc.value);
        let dk = nope + rope;
        let mut low_kept = None;
        let q_full = match &m.q_a {
            Some((down, norm)) => {
                let mut low = down.forward(x)?;
                rms(&mut low, norm, cfg.eps);
                let q = m.q.forward(&low)?;
                if keep_low {
                    low_kept = Some(low);
                }
                q
            }
            None => m.q.forward(x)?,
        };
        let kv_a = m.kv_a.forward(x)?;
        let width = mc.kv_lora + rope;
        let mut latent = Vec::with_capacity(rows * mc.kv_lora);
        for r in 0..rows {
            latent.extend_from_slice(&kv_a[r * width..r * width + mc.kv_lora]);
        }
        rms(&mut latent, &m.kv_a_norm, cfg.eps);
        let kv = m.kv_b.forward(&latent)?;
        let per_head = nope + value;
        let mut q = vec![0.0f32; rows * heads * dk];
        let mut k = vec![0.0f32; rows * heads * dk];
        let mut v = vec![0.0f32; rows * heads * value];
        for r in 0..rows {
            let pe = &kv_a[r * width + mc.kv_lora..(r + 1) * width];
            for h in 0..heads {
                let at = r * heads + h;
                let source = &q_full[at * dk..(at + 1) * dk];
                q[at * dk..at * dk + rope].copy_from_slice(&source[nope..]);
                q[at * dk + rope..(at + 1) * dk].copy_from_slice(&source[..nope]);
                k[at * dk..at * dk + rope].copy_from_slice(pe);
                let content = &kv[at * per_head..(at + 1) * per_head];
                k[at * dk + rope..(at + 1) * dk].copy_from_slice(&content[..nope]);
                v[at * value..(at + 1) * value].copy_from_slice(&content[nope..]);
            }
        }
        if mc.rotary {
            apply_rope(&mut q, heads, dk, rope, cfg.rope.neox, positions, table);
            apply_rope(&mut k, heads, dk, rope, cfg.rope.neox, positions, table);
        }
        Ok(([q, k, v], low_kept))
    }

    /// The attention tokens of every row of a sparse-attention layer: `None` for a row that attends to all of its `lengths[r]` tokens (at
    /// most `top_k` of them), otherwise the positions the lightning indexer ranks best. A layer without an indexer of its own returns the
    /// selection of the layer before. The keys of the indexer are written to the side store first (they are needed by later tokens).
    #[allow(clippy::too_many_arguments)]
    fn dsa_select(
        &self,
        i: usize,
        layer: &Layer,
        normed: &[f32],
        q_low: Option<&[f32]>,
        positions: &[usize],
        table: &RopeTables,
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<Option<Vec<u32>>>, String> {
        let rows = positions.len();
        let (Some(ic), Some(mc)) = (&self.cfg.indexer, &self.cfg.mla) else {
            return Err("sparse attention without an indexer configuration".into());
        };
        let Some(ix) = layer.indexer.as_deref() else {
            let previous = self.dsa.selection.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            if previous.len() != rows {
                return Err(format!("layer {i} shares the sparse-attention selection of the layer before, which did not run on these tokens"));
            }
            return Ok(previous);
        };
        let q_low = q_low.ok_or("sparse attention needs the low-rank query")?;
        let (heads, dim) = (ic.heads, ic.dim);
        let mut q = ix.q_b.forward(q_low)?;
        let mut k = ix.k.forward(normed)?;
        layer_norm(&mut k, &ix.k_norm, Some(&ix.k_norm_bias), ic.norm_eps);
        let weights = ix.proj.forward(normed)?;
        if ic.rope_last {
            // the rotated values are the last ones of the head: move them to the front of queries and keys alike (dot products do not
            // depend on the order of the values as long as both sides agree)
            for head in q.chunks_exact_mut(dim).chain(k.chunks_exact_mut(dim)) {
                head.rotate_right(mc.rope);
            }
        }
        apply_rope(&mut q, heads, dim, mc.rope, ic.neox, positions, table);
        apply_rope(&mut k, 1, dim, mc.rope, ic.neox, positions, table);
        let slot_of = |r: usize, p: usize| tables[r * self.stride + p / self.block_size] as usize * self.block_size + p % self.block_size;
        let mut store = self.dsa.keys.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.len() < self.cfg.layers {
            store.resize_with(self.cfg.layers, Vec::new);
        }
        if store[i].is_empty() {
            store[i] = vec![0.0; self.pool_blocks * self.block_size * ic.store_width()];
        }
        let width = ic.store_width();
        let gate = match &ix.pool {
            Some((gate, _)) => gate.forward(normed)?,
            None => Vec::new(),
        };
        let keys = &mut store[i];
        for (r, &position) in positions.iter().enumerate() {
            let slot = slot_of(r, position);
            keys[slot * width..slot * width + dim].copy_from_slice(&k[r * dim..(r + 1) * dim]);
            if !gate.is_empty() {
                keys[slot * width + dim..(slot + 1) * width].copy_from_slice(&gate[r * dim..(r + 1) * dim]);
            }
        }
        let keys: &[f32] = keys;
        let (scale, head_scale) = ((dim as f32).powf(-0.5), (heads as f32).powf(-0.5));
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace = *TRACE.get_or_init(|| std::env::var_os("FLLM_DSA_TRACE").is_some());
        let selected: Vec<Option<Vec<u32>>> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let n = lengths[r] as usize;
                if n <= ic.topk && ix.pool.is_none() {
                    return None;
                }
                if let Some((_, ape)) = &ix.pool {
                    return Some(pooled_selection(ic, &q[r * heads * dim..(r + 1) * heads * dim], &weights[r * heads..(r + 1) * heads], ape, n, |p| &keys[slot_of(r, p) * width..(slot_of(r, p) + 1) * width], trace));
                }
                let scored: Vec<(f32, u32)> = (0..n)
                    .map(|p| {
                        let slot = slot_of(r, p);
                        let key = &keys[slot * width..slot * width + dim];
                        let mut score = 0.0f32;
                        for h in 0..heads {
                            let dot = compute::dot(&q[(r * heads + h) * dim..(r * heads + h + 1) * dim], key) * scale;
                            if dot > 0.0 {
                                score += dot * weights[r * heads + h] * head_scale;
                            }
                        }
                        (score, p as u32)
                    })
                    .collect();
                if trace {
                    let mut all = scored.clone();
                    all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    eprintln!("[dsa-scores] layer {i} position {}: {:?}", positions[r], &all[..all.len().min(ic.topk + 3)]);
                }
                Some(best_positions(scored, ic.topk))
            })
            .collect();
        if trace {
            eprintln!("[dsa] indexer {ic:?}");
            for (r, rows_selection) in selected.iter().enumerate() {
                eprintln!("[dsa] layer {i} position {}: {:?}", positions[r], rows_selection);
            }
        }
        *self.dsa.selection.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = selected.clone();
        Ok(selected)
    }

    /// Bytes one cache block adds to an exported block for the indexer keys (0 without sparse attention).
    pub(crate) fn dsa_block_bytes(&self) -> usize {
        match (&self.cfg.indexer, self.dsa_dense) {
            (Some(ic), false) => self.layers.iter().filter(|l| l.indexer.is_some()).count() * self.block_size * ic.store_width() * 4,
            _ => 0,
        }
    }

    /// Calls `f(layer keys, first value of the block, values per block)` for every layer with an indexer.
    fn dsa_for_blocks(&self, block: usize, mut f: impl FnMut(&mut Vec<f32>, usize, usize)) {
        let Some(ic) = &self.cfg.indexer else { return };
        if self.dsa_device {
            // the keys live in the device stores: the block's rows go down, `f` works on them, and they go back up
            let (Some(weights), Some(stream)) = (self.backend_weights.as_ref(), self.backend_stream.as_ref()) else { return };
            let Ok(mut stream) = stream.lock() else { return };
            let per_block = self.block_size * ic.store_width();
            for dev in &weights.layers {
                let Some(ds) = &dev.dsa else { continue };
                let Ok(mut region) = ds.store.slice(block * per_block * 4, per_block * 4) else { continue };
                let mut bytes = vec![0u8; per_block * 4];
                if stream.download(&mut bytes, &region).is_err() {
                    continue;
                }
                let mut values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                f(&mut values, 0, per_block);
                let out: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = stream.upload(&mut region, &out);
            }
            return;
        }
        let mut store = self.dsa.keys.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if store.len() < self.cfg.layers {
            store.resize_with(self.cfg.layers, Vec::new);
        }
        let per_block = self.block_size * ic.store_width();
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.indexer.is_none() {
                continue;
            }
            if store[i].is_empty() {
                store[i] = vec![0.0; self.pool_blocks * per_block];
            }
            f(&mut store[i], block * per_block, per_block);
        }
    }

    /// A copy of a cache block (the fork of `n > 1` choices) takes the indexer keys along.
    pub(crate) fn dsa_copy_block(&self, src: usize, dst: usize) {
        let mut moved: Vec<Vec<f32>> = Vec::new();
        self.dsa_for_blocks(src, |keys, at, n| moved.push(keys[at..at + n].to_vec()));
        let mut it = moved.into_iter();
        self.dsa_for_blocks(dst, |keys, at, n| {
            if let Some(values) = it.next() {
                keys[at..at + n].copy_from_slice(&values);
            }
        });
    }

    /// The indexer keys of a block as little-endian f32 values (appended to an exported block).
    pub(crate) fn dsa_export_block(&self, block: usize, out: &mut Vec<u8>) {
        if self.dsa_dense {
            return;
        }
        self.dsa_for_blocks(block, |keys, at, n| {
            for v in &keys[at..at + n] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        });
    }

    /// Restores the indexer keys of a block from the tail of an exported block.
    pub(crate) fn dsa_import_block(&self, block: usize, data: &[u8]) {
        let mut at = 0;
        self.dsa_for_blocks(block, |keys, first, n| {
            for v in &mut keys[first..first + n] {
                if let Some(b) = data.get(at..at + 4) {
                    *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                at += 4;
            }
        });
    }

    /// Shape of the Gated DeltaNet layers, when the model has any.
    /// `FLLM_LAYER_CHECK=1`: runs layer `i` with the host reference on `host_in` (the residual stream the device layer started
    /// from) and prints how far the device result `x` is from it.
    #[allow(clippy::too_many_arguments)]
    fn layer_check(
        &self,
        stream: &mut dyn crate::backend::Stream,
        (first, i): (usize, usize),
        x: &crate::backend::DeviceBuffer,
        mut host_x: Vec<f32>,
        positions: &[usize],
        tables: &[i64],
        lengths: &[i64],
        key: &mut [u16],
        value: &mut [u16],
    ) -> Result<(), String> {
        let mut trace = crate::cpu_profile::Trace::new(false, positions.len());
        if self.cfg.inkling.is_some() {
            // likewise (its convolution histories are fed step by step like a host run would)
            host_x = self.inkling_layers(&host_x, positions, tables, lengths, first..i + 1)?;
        } else if self.cfg.minimax3.is_some() {
            // the host pass of this model has its own layer loop (and keeps its own per-sequence state, which the check feeds step by step like a host run would)
            host_x = self.minimax3_layers(&host_x, positions, tables, lengths, first..i + 1)?;
        } else {
            self.host_layers(first..i + 1, &mut host_x, positions, tables, lengths, key, value, &mut trace, None, &mut Vec::new())?;
        }
        let mut dev = vec![0.0f32; host_x.len()];
        stream.download(cast_f32_to_u8_mut(&mut dev), x).map_err(|e| e.to_string())?;
        let rms = (host_x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / host_x.len().max(1) as f64).sqrt();
        let worst = dev.iter().zip(&host_x).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max);
        let mean = dev.iter().zip(&host_x).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / host_x.len().max(1) as f64;
        let name = if first < i { format!("{first}..{i}") } else { i.to_string() };
        eprintln!(
            "[layer-check] layer {name} rows {}: max|dev-host| {worst:.5} mean {mean:.6} rms {rms:.4} (max/rms {:.4})",
            positions.len(),
            worst / rms.max(1e-12)
        );
        Ok(())
    }

    /// A short-convolution layer on the device: input projection, gated convolution kernel, output projection added to the residual.
    #[allow(clippy::too_many_arguments)]
    fn shortconv_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        kernel: usize,
        dc: &DeviceShortConv,
        dev_layer: &DeviceLayerWeights,
        plan: &[DeltaPlanRun],
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let hidden = self.cfg.hidden;
        let resets = self.delta_claim(dc.ordinal, plan)?;
        let mut proj = scratch.acquire(rows * 3 * hidden * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.quantized_linear(&mut proj, normed, &dc.input, dc.input.bias.as_ref()).map_err(err)?;
        let runs: Vec<crate::backend::DeltaRun<'_>> = plan
            .iter()
            .zip(&resets)
            .map(|(p, &reset)| crate::backend::DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset })
            .collect();
        let layer_bytes = (hidden * (kernel - 1) * 4).next_multiple_of(256);
        stream
            .shortconv_layer(hidden, kernel, &dc.conv, dc.bias.as_ref(), &runs, dc.ordinal * layer_bytes, &proj, &mut out)
            .map_err(err)?;
        let residual_scale = self.cfg.residual_scale;
        let out_w = dev_layer.out.as_ref().ok_or("ShortConv layer without a device output projection")?;
        let fused = dev_layer.post_attn_norm.is_none()
            && stream.quantized_linear_add(x, &out, out_w, out_w.bias.as_ref(), residual_scale).map_err(err)?;
        if !fused {
            let mut branch = scratch.acquire(rows * hidden * 4).map_err(err)?;
            stream.quantized_linear(&mut branch, &out, out_w, out_w.bias.as_ref()).map_err(err)?;
            device_post_norm_inplace(
                stream,
                scratch,
                self.cfg.norm,
                &mut branch,
                dev_layer.post_attn_norm.as_ref(),
                dev_layer.post_attn_norm_b.as_ref(),
                self.cfg.post_eps,
                self.groups,
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [proj, out] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    fn mamba1_shape(&self) -> Option<crate::backend::Mamba1Shape> {
        let mc = self.cfg.mamba1.as_ref()?;
        self.layers.iter().any(|l| matches!(l.attn, Attn::Mamba1(_))).then_some(crate::backend::Mamba1Shape {
            inner: mc.inner,
            state: mc.state,
            rank: mc.rank,
            conv_kernel: mc.conv_kernel,
            bc_rms_eps: mc.bc_rms_eps,
        })
    }

    /// A Mamba-1 layer on the device: input projection, convolution kernel, x_proj, dt/B/C normalisation, dt_proj, selective scan, output
    /// projection added to the residual stream.
    #[allow(clippy::too_many_arguments)]
    fn mamba1_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        shape: &crate::backend::Mamba1Shape,
        dm: &DeviceMamba1,
        dev_layer: &DeviceLayerWeights,
        plan: &[DeltaPlanRun],
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let resets = self.delta_claim(dm.ordinal, plan)?;
        let (inner, rank, state) = (shape.inner, shape.rank, shape.state);
        let mut proj = scratch.acquire(rows * 2 * inner * 4).map_err(err)?;
        let mut mixed = scratch.acquire(rows * inner * 4).map_err(err)?;
        let mut split = scratch.acquire(rows * (rank + 2 * state) * 4).map_err(err)?;
        let mut dt_in = scratch.acquire(rows * rank * 4).map_err(err)?;
        let mut dt = scratch.acquire(rows * inner * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * inner * 4).map_err(err)?;
        stream.quantized_linear(&mut proj, normed, &dm.input, dm.input.bias.as_ref()).map_err(err)?;
        let runs: Vec<crate::backend::DeltaRun<'_>> = plan
            .iter()
            .zip(&resets)
            .map(|(p, &reset)| crate::backend::DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset })
            .collect();
        let offset = dm.ordinal * shape.layer_state_bytes();
        stream.mamba1_conv(shape, &dm.conv, dm.conv_bias.as_ref(), &runs, offset, &proj, &mut mixed).map_err(err)?;
        stream.quantized_linear(&mut split, &mixed, &dm.x_proj, dm.x_proj.bias.as_ref()).map_err(err)?;
        stream.mamba1_prep(shape, &mut split, &mut dt_in, rows, dm.norms.as_ref().map(|w| [&w[0], &w[1], &w[2]])).map_err(err)?;
        stream.quantized_linear(&mut dt, &dt_in, &dm.dt_proj, dm.dt_proj.bias.as_ref()).map_err(err)?;
        stream.mamba1_scan(shape, &dm.decay, &dm.skip, &runs, offset, &split, &dt, &mixed, &proj, &mut out).map_err(err)?;
        let residual_scale = self.cfg.residual_scale;
        let out_w = dev_layer.out.as_ref().ok_or("Mamba layer without a device output projection")?;
        let fused = dev_layer.post_attn_norm.is_none()
            && stream.quantized_linear_add(x, &out, out_w, out_w.bias.as_ref(), residual_scale).map_err(err)?;
        if !fused {
            let mut branch = scratch.acquire(rows * self.cfg.hidden * 4).map_err(err)?;
            stream.quantized_linear(&mut branch, &out, out_w, out_w.bias.as_ref()).map_err(err)?;
            device_post_norm_inplace(
                stream,
                scratch,
                self.cfg.norm,
                &mut branch,
                dev_layer.post_attn_norm.as_ref(),
                dev_layer.post_attn_norm_b.as_ref(),
                self.cfg.post_eps,
                self.groups,
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [proj, mixed, split, dt_in, dt, out] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    fn mamba_shape(&self) -> Option<crate::backend::MambaShape> {
        let mc = self.cfg.mamba.as_ref()?;
        self.layers.iter().any(|l| matches!(l.attn, Attn::Mamba(_)) || l.parallel.is_some()).then(|| crate::backend::MambaShape {
            inner: mc.inner,
            state: mc.state,
            heads: mc.heads,
            groups: mc.groups,
            conv_kernel: mc.conv_kernel,
            norm_groups: mc.norm_groups.max(1),
            step_min: mc.step_min,
            eps: self.cfg.eps,
            norm_mode: match mc.norm {
                crate::config::MambaNorm::GateThenNorm => 0,
                crate::config::MambaNorm::NoNorm => 1,
                crate::config::MambaNorm::NormThenGate => 2,
            },
        })
    }

    /// A Mamba-2 layer on the device: input projection, convolution + scan + gated norm kernels, output projection added to the
    /// residual stream. The state lives in the sequence's device buffer (see `state_plan`).
    #[allow(clippy::too_many_arguments)]
    fn mamba_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        shape: &crate::backend::MambaShape,
        dm: &DeviceMamba,
        dev_layer: &DeviceLayerWeights,
        out_override: Option<&crate::backend::DeviceWeight>,
        plan: &[DeltaPlanRun],
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let resets = self.delta_claim(dm.ordinal, plan)?;
        let conv_dim = shape.conv_dim();
        let width = shape.inner + conv_dim + shape.heads;
        let mut proj = scratch.acquire(rows * width * 4).map_err(err)?;
        let mut mixed = scratch.acquire(rows * conv_dim * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * shape.inner * 4).map_err(err)?;
        stream.quantized_linear(&mut proj, normed, &dm.input, dm.input.bias.as_ref()).map_err(err)?;
        let runs: Vec<crate::backend::DeltaRun<'_>> = plan
            .iter()
            .zip(&resets)
            .map(|(p, &reset)| crate::backend::DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset })
            .collect();
        let weights = crate::backend::MambaWeights {
            conv: &dm.conv,
            conv_bias: dm.conv_bias.as_ref(),
            dt_bias: &dm.dt_bias,
            decay: &dm.decay,
            skip: &dm.skip,
            norm: &dm.norm,
        };
        stream
            .mamba2_layer(shape, &weights, &runs, dm.ordinal * shape.layer_state_bytes(), &proj, &mut mixed, &mut out)
            .map_err(err)?;
        let residual_scale = self.cfg.residual_scale;
        let out_w = out_override.or(dev_layer.out.as_ref()).ok_or("Mamba layer without a device output projection")?;
        let fused = dev_layer.post_attn_norm.is_none()
            && stream
                .quantized_linear_add(x, &out, out_w, out_w.bias.as_ref(), residual_scale)
                .map_err(err)?;
        if !fused {
            let mut branch = scratch.acquire(rows * self.cfg.hidden * 4).map_err(err)?;
            stream.quantized_linear(&mut branch, &out, out_w, out_w.bias.as_ref()).map_err(err)?;
            device_post_norm_inplace(
                stream,
                scratch,
                self.cfg.norm,
                &mut branch,
                dev_layer.post_attn_norm.as_ref(),
                dev_layer.post_attn_norm_b.as_ref(),
                self.cfg.post_eps,
                self.groups,
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [proj, mixed, out] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    fn kda_shape(&self) -> Option<crate::backend::KdaShape> {
        let kc = self.cfg.hybrid.kda.as_ref()?;
        self.layers.iter().any(|l| matches!(l.attn, Attn::Kda(_))).then_some(crate::backend::KdaShape {
            heads: kc.heads,
            head_dim: kc.head_dim,
            conv_kernel: kc.conv_kernel,
            eps: self.cfg.eps,
            lower_bound: kc.lower_bound,
        })
    }

    fn delta_shape(&self) -> Option<crate::backend::DeltaShape> {
        let dc = self.cfg.delta.as_ref()?;
        self.layers.iter().any(|l| matches!(l.attn, Attn::Linear(_))).then(|| crate::backend::DeltaShape {
            k_heads: dc.k_heads,
            v_heads: dc.v_heads,
            k_dim: dc.k_dim,
            v_dim: dc.v_dim,
            conv_kernel: dc.conv_kernel,
            tiled_heads: self.cfg.arch.starts_with("qwen35"),
            beta_scale: dc.beta_scale,
            eps: dc.norm_eps.unwrap_or(self.cfg.eps),
        })
    }

    /// Splits the batch into runs of consecutive tokens of one sequence (sequences are identified by their first KV block, as on
    /// the host) and finds or creates each run's state buffer on the device. Least recently used sequences make room beyond
    /// `state_capacity`; a sequence that lost its state fails the position check of `delta_claim` like it does on the host.
    fn delta_plan(
        &self,
        shape: &crate::backend::DeltaShape,
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<DeltaPlanRun>, String> {
        let layers = self.layers.iter().filter(|l| matches!(l.attn, Attn::Linear(_))).count();
        self.state_plan(layers, shape.layer_state_bytes(), tables, lengths)
    }

    /// `delta_plan` for any recurrent mixer kept in a device state buffer: `layers` layers of `layer_bytes` bytes each.
    fn state_plan(&self, layers: usize, layer_bytes: usize, tables: &[i64], lengths: &[i64]) -> Result<Vec<DeltaPlanRun>, String> {
        let backend = self.backend.as_ref().ok_or("accelerator backend is not initialised")?;
        let bytes = layers * layer_bytes;
        let rows = lengths.len();
        let mut store = self
            .device_states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        let batch_start = store.clock;
        let mut plan: Vec<DeltaPlanRun> = Vec::new();
        let mut start = 0;
        while start < rows {
            let first = tables[start * self.stride];
            let position = lengths[start] as usize - 1;
            let mut end = start + 1;
            while end < rows && tables[end * self.stride] == first && lengths[end] == lengths[end - 1] + 1 {
                end += 1;
            }
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) {
                if store.map.len() >= self.state_capacity {
                    let oldest = store
                        .map
                        .iter()
                        .min_by_key(|(_, s)| s.used)
                        .map(|(k, s)| (*k, s.used));
                    if let Some((key, used)) = oldest {
                        if used > batch_start {
                            return Err("more sequences in one batch than recurrent states".into());
                        }
                        if let Some(old) = store.map.remove(&key) {
                            store.free.push(old.buf);
                        }
                    }
                }
                let buf = match store.free.pop() {
                    Some(buf) => buf,
                    None => backend.allocate_buffer(bytes).map_err(|e| e.to_string())?,
                };
                store.map.insert(first, DeviceSeq { used: clock, buf, lens: vec![0; layers] });
            }
            let seq = store.map.get_mut(&first).ok_or("recurrent state vanished")?;
            seq.used = clock;
            plan.push(DeltaPlanRun { first, buf: seq.buf.clone(), start, n: end - start, position });
            start = end;
        }
        Ok(plan)
    }

    /// Checks that layer `ordinal` holds exactly the tokens before each run and counts the run in; returns whether the run
    /// starts a sequence (the kernels then ignore the old state).
    fn delta_claim(&self, ordinal: usize, plan: &[DeltaPlanRun]) -> Result<Vec<bool>, String> {
        let mut store = self
            .device_states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        let mut resets = Vec::with_capacity(plan.len());
        for run in plan {
            let seq = store.map.get_mut(&run.first).ok_or("recurrent state vanished")?;
            if run.position == 0 {
                seq.lens[ordinal] = 0;
            }
            if seq.lens[ordinal] != run.position {
                return Err(format!(
                    "recurrent state out of sync (holds {} tokens, got position {})",
                    seq.lens[ordinal], run.position
                ));
            }
            seq.lens[ordinal] += run.n;
            resets.push(run.position == 0);
        }
        Ok(resets)
    }

    /// Debug aid (`FLLM_DELTA_CHECK=1`): runs the host reference (`cpu_delta`) on the projections the device computed and
    /// reports the largest deviation of the kernels' output; the reference keeps its own state per sequence and layer.
    #[allow(clippy::too_many_arguments)]
    fn delta_check(
        &self,
        stream: &mut dyn crate::backend::Stream,
        layer: usize,
        shape: &crate::backend::DeltaShape,
        plan: &[DeltaPlanRun],
        inputs: [&crate::backend::DeviceBuffer; 4],
        out: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        use std::collections::HashMap;
        use std::sync::{LazyLock, Mutex};
        type Shadow = HashMap<(i64, usize), (Vec<f32>, Vec<f32>)>;
        static SHADOW: LazyLock<Mutex<Shadow>> = LazyLock::new(Default::default);
        static WORST: Mutex<f64> = Mutex::new(0.0);
        let Attn::Linear(d) = &self.layers[layer].attn else {
            return Ok(());
        };
        let (conv_dim, value_dim, vh) = (shape.conv_dim(), shape.value_dim(), shape.v_heads);
        let fetch = |stream: &mut dyn crate::backend::Stream, buf: &crate::backend::DeviceBuffer, n: usize| {
            let mut host = vec![0.0f32; n];
            stream.download(cast_f32_to_u8_mut(&mut host), buf).map(|()| host).map_err(|e| e.to_string())
        };
        let qkv = fetch(stream, inputs[0], rows * conv_dim)?;
        let z = fetch(stream, inputs[1], rows * value_dim)?;
        let beta = fetch(stream, inputs[2], rows * vh)?;
        let alpha = fetch(stream, inputs[3], rows * vh)?;
        let got = fetch(stream, out, rows * value_dim)?;
        let params = crate::cpu_delta::Layer {
            k_heads: shape.k_heads,
            v_heads: vh,
            k_dim: shape.k_dim,
            v_dim: shape.v_dim,
            conv_kernel: shape.conv_kernel,
            beta_scale: shape.beta_scale,
            eps: shape.eps,
            tiled_heads: shape.tiled_heads,
            conv: &d.conv,
            dt_bias: &d.dt_bias,
            decay: &d.decay,
            norm: &d.norm,
        };
        let mut shadow = SHADOW.lock().map_err(|_| "delta check lock poisoned")?;
        let (mut worst, mut scale) = (0.0f64, 0.0f64);
        for run in plan {
            let entry = shadow.entry((run.first, layer)).or_default();
            if run.position == 0 || entry.0.is_empty() {
                *entry = (vec![0.0; conv_dim * (shape.conv_kernel - 1)], vec![0.0; vh * shape.k_dim * shape.v_dim]);
            }
            let (a, b) = (run.start, run.start + run.n);
            let mut want = vec![0.0f32; run.n * value_dim];
            crate::cpu_delta::run(
                &params,
                &mut entry.0,
                &mut entry.1,
                (&qkv[a * conv_dim..b * conv_dim], &z[a * value_dim..b * value_dim], &beta[a * vh..b * vh], &alpha[a * vh..b * vh]),
                &mut want,
                run.n,
            );
            for (w, g) in want.iter().zip(&got[a * value_dim..b * value_dim]) {
                worst = worst.max(f64::from((w - g).abs()));
                scale = scale.max(f64::from(w.abs()));
            }
        }
        let mut all = WORST.lock().map_err(|_| "delta check lock poisoned")?;
        if worst > *all * 1.5 || worst.is_nan() {
            *all = worst;
            eprintln!("[delta-check] layer {layer}: new largest deviation {worst:.3e} (outputs up to {scale:.3e}, rows {rows})");
        }
        Ok(())
    }

    /// One Gated DeltaNet layer on the device: projections, the recurrence kernels, output projection added to the residual.
    #[allow(clippy::too_many_arguments)]
    fn delta_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        shape: &crate::backend::DeltaShape,
        dd: &DeviceDelta,
        dev_layer: &DeviceLayerWeights,
        (layer, plan): (usize, &[DeltaPlanRun]),
        normed: &crate::backend::DeviceBuffer,
        x: &mut crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<(), String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let resets = self.delta_claim(dd.ordinal, plan)?;
        let (conv_dim, value_dim, vh) = (shape.conv_dim(), shape.value_dim(), shape.v_heads);
        let mut qkv = scratch.acquire(rows * conv_dim * 4).map_err(err)?;
        let mut mixed = scratch.acquire(rows * conv_dim * 4).map_err(err)?;
        let mut z = scratch.acquire(rows * value_dim * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * value_dim * 4).map_err(err)?;
        let mut beta = scratch.acquire(rows * vh * 4).map_err(err)?;
        let mut alpha = scratch.acquire(rows * vh * 4).map_err(err)?;
        stream.quantized_linear(&mut qkv, normed, &dd.qkv, dd.qkv.bias.as_ref()).map_err(err)?;
        stream.quantized_linear(&mut z, normed, &dd.gate, dd.gate.bias.as_ref()).map_err(err)?;
        stream.quantized_linear(&mut beta, normed, &dd.beta, dd.beta.bias.as_ref()).map_err(err)?;
        stream.quantized_linear(&mut alpha, normed, &dd.alpha, dd.alpha.bias.as_ref()).map_err(err)?;
        let runs: Vec<crate::backend::DeltaRun<'_>> = plan
            .iter()
            .zip(&resets)
            .map(|(p, &reset)| crate::backend::DeltaRun { state: &p.buf, first_row: p.start, rows: p.n, reset })
            .collect();
        let weights = crate::backend::DeltaWeights {
            conv: &dd.conv,
            dt_bias: &dd.dt_bias,
            decay: &dd.decay,
            norm: &dd.norm,
        };
        stream
            .delta_layer(
                shape,
                &weights,
                &runs,
                dd.ordinal * shape.layer_state_bytes(),
                &qkv,
                &z,
                &beta,
                &alpha,
                &mut mixed,
                &mut out,
            )
            .map_err(err)?;
        if std::env::var_os("FLLM_DELTA_CHECK").is_some() {
            self.delta_check(stream, layer, shape, plan, [&qkv, &z, &beta, &alpha], &out, rows)?;
        }
        let residual_scale = self.cfg.residual_scale;
        let out_w = dev_layer.out.as_ref().ok_or("DeltaNet layer without a device output projection")?;
        let fused = dev_layer.post_attn_norm.is_none()
            && stream
                .quantized_linear_add(x, &out, out_w, out_w.bias.as_ref(), residual_scale)
                .map_err(err)?;
        if !fused {
            let mut branch = scratch.acquire(rows * self.cfg.hidden * 4).map_err(err)?;
            stream
                .quantized_linear(&mut branch, &out, out_w, out_w.bias.as_ref())
                .map_err(err)?;
            device_post_norm_inplace(
                stream,
                scratch,
                self.cfg.norm,
                &mut branch,
                dev_layer.post_attn_norm.as_ref(),
                dev_layer.post_attn_norm_b.as_ref(),
                self.cfg.post_eps,
                self.groups,
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [qkv, mixed, z, out, beta, alpha] {
            scratch.recycle(buf);
        }
        Ok(())
    }

    /// Gated DeltaNet for a batch of tokens, one token at a time per sequence because the state
    /// depends on every earlier token. Sequences are identified by their first KV block.
    fn delta_forward(
        &self,
        d: &DeltaNet,
        out_projection: &Linear,
        layer: usize,
        x: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let dc = cfg.delta.as_ref().ok_or("missing DeltaNet configuration")?;
        let rows = lengths.len();
        let (kh, vh, hk, hv, ksz) = (dc.k_heads, dc.v_heads, dc.k_dim, dc.v_dim, dc.conv_kernel);
        let key_dim = kh * hk;
        let value_dim = vh * hv;
        let conv_dim = 2 * key_dim + value_dim;
        let params = crate::cpu_delta::Layer {
            k_heads: kh,
            v_heads: vh,
            k_dim: hk,
            v_dim: hv,
            conv_kernel: ksz,
            beta_scale: dc.beta_scale,
            eps: dc.norm_eps.unwrap_or(cfg.eps),
            // HF checkpoints (and fLLM's converter) group the value heads by key head; llama.cpp's
            // Qwen3.5 files store them tiled (value head j belongs to key head j % key heads)
            tiled_heads: cfg.arch.starts_with("qwen35"),
            conv: &d.conv,
            dt_bias: &d.dt_bias,
            decay: &d.decay,
            norm: &d.norm,
        };
        // Siblings with one input share the activation packing and one work queue.
        let mut wide = Linear::forward_group(&[&d.qkv, &d.gate], x)?.into_iter();
        let (qkv, z) = wide.next().zip(wide.next()).ok_or("DeltaNet projection group")?;
        let mut narrow = Linear::forward_group(&[&d.beta, &d.alpha], x)?.into_iter();
        let (beta, alpha) = narrow.next().zip(narrow.next()).ok_or("DeltaNet gate group")?;
        let mut out = vec![0.0f32; rows * value_dim];
        let mut store = self
            .states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        // Rows of one sequence with consecutive positions form a run: the state is folded
        // through the whole run at once instead of token by token.
        let mut start = 0;
        while start < rows {
            let first = tables[start * self.stride];
            let position = lengths[start] as usize - 1;
            let mut end = start + 1;
            while end < rows
                && tables[end * self.stride] == first
                && lengths[end] == lengths[end - 1] + 1
            {
                end += 1;
            }
            let n = end - start;
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store
                    .map
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k)
                {
                    store.map.remove(&oldest);
                }
            }
            let seq = store.map.entry(first).or_insert_with(|| SeqState {
                used: clock,
                layers: (0..cfg.layers).map(|_| None).collect(),
            });
            seq.used = clock;
            if position == 0 {
                seq.layers[layer] = None;
            }
            let state = seq.layers[layer].get_or_insert_with(|| LayerState {
                len: 0,
                conv: vec![0.0; conv_dim * (ksz - 1)],
                rec: vec![0.0; vh * hk * hv],
            });
            if state.len != position {
                return Err(format!(
                    "recurrent state out of sync (holds {} tokens, got position {position})",
                    state.len
                ));
            }
            state.len += n;
            crate::cpu_delta::run(
                &params,
                &mut state.conv,
                &mut state.rec,
                (
                    &qkv[start * conv_dim..end * conv_dim],
                    &z[start * value_dim..end * value_dim],
                    &beta[start * vh..end * vh],
                    &alpha[start * vh..end * vh],
                ),
                &mut out[start * value_dim..end * value_dim],
                n,
            );
            start = end;
        }
        drop(store);
        out_projection.forward(&out)
    }

    /// LFM2: project [B,C,x], convolve B*x causally, then project C*conv(B*x).
    fn shortconv_forward(
        &self,
        m: &ShortConv,
        projection: &Linear,
        layer: usize,
        input: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let kernel = cfg
            .shortconv
            .as_ref()
            .ok_or("missing ShortConv configuration")?
            .kernel;
        let hidden = cfg.hidden;
        let projected = m.input.forward(input)?;
        let mut output = vec![0.0; lengths.len() * hidden];
        let mut store = self
            .states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        for (row, &length) in lengths.iter().enumerate() {
            let position = usize::try_from(length)
                .ok()
                .and_then(|n| n.checked_sub(1))
                .ok_or("invalid ShortConv sequence position")?;
            let first = tables[row * self.stride];
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store
                    .map
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(key, _)| *key)
                {
                    store.map.remove(&oldest);
                }
            }
            let seq = store.map.entry(first).or_insert_with(|| SeqState {
                used: clock,
                layers: (0..cfg.layers).map(|_| None).collect(),
            });
            seq.used = clock;
            if position == 0 {
                seq.layers[layer] = None;
            }
            let state = seq.layers[layer].get_or_insert_with(|| LayerState {
                len: 0,
                conv: vec![0.0; hidden * (kernel - 1)],
                rec: Vec::new(),
            });
            if state.len != position {
                return Err(format!(
                    "ShortConv layer {layer} state holds {} tokens, got position {position}",
                    state.len
                ));
            }
            let values = &projected[row * 3 * hidden..(row + 1) * 3 * hidden];
            let (b, rest) = values.split_at(hidden);
            let (c, x) = rest.split_at(hidden);
            for channel in 0..hidden {
                let history = &mut state.conv[channel * (kernel - 1)..(channel + 1) * (kernel - 1)];
                let weights = &m.conv[channel * kernel..(channel + 1) * kernel];
                let current = b[channel] * x[channel];
                let mut value = m.bias.as_ref().map_or(0.0, |bias| bias[channel]);
                for (weight, previous) in weights.iter().zip(history.iter()) {
                    value += weight * previous;
                }
                value += weights[kernel - 1] * current;
                output[row * hidden + channel] = c[channel] * value;
                if kernel > 1 {
                    history.copy_within(1.., 0);
                    history[kernel - 2] = current;
                }
            }
            state.len += 1;
        }
        drop(store);
        projection.forward(&output)
    }

    /// Mamba-1 mixer for a batch of tokens (selective scan with a per-channel state).
    fn mamba1_forward(
        &self,
        m: &Mamba1,
        out_projection: &Linear,
        layer: usize,
        x: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let mc = cfg.mamba1.as_ref().ok_or("missing Mamba configuration")?;
        let rows = lengths.len();
        let (inner, n, rank, ksz) = (mc.inner, mc.state, mc.rank, mc.conv_kernel);
        let projected = m.input.forward(x)?;
        let mut out = vec![0.0f32; rows * inner];
        let mut store = self
            .states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        for r in 0..rows {
            let position = lengths[r] as usize - 1;
            let first = tables[r * self.stride];
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store
                    .map
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k)
                {
                    store.map.remove(&oldest);
                }
            }
            let seq = store.map.entry(first).or_insert_with(|| SeqState {
                used: clock,
                layers: (0..cfg.layers).map(|_| None).collect(),
            });
            seq.used = clock;
            if position == 0 {
                seq.layers[layer] = None;
            }
            let state = seq.layers[layer].get_or_insert_with(|| LayerState {
                len: 0,
                conv: vec![0.0; inner * (ksz - 1)],
                rec: vec![0.0; inner * n],
            });
            if state.len != position {
                return Err(format!(
                    "recurrent state out of sync (holds {} tokens, got position {position})",
                    state.len
                ));
            }
            state.len += 1;
            let row = &projected[r * 2 * inner..(r + 1) * 2 * inner];
            let (xs, gate) = row.split_at(inner);
            let mut mixed = vec![0.0f32; inner];
            for (c, mixed) in mixed.iter_mut().enumerate() {
                let weights = &m.conv[c * ksz..(c + 1) * ksz];
                let history = &mut state.conv[c * (ksz - 1)..(c + 1) * (ksz - 1)];
                let mut acc = weights[ksz - 1] * xs[c] + m.conv_bias.as_ref().map_or(0.0, |b| b[c]);
                for (w, h) in weights.iter().zip(history.iter()) {
                    acc += w * h;
                }
                if ksz > 1 {
                    history.copy_within(1.., 0);
                    history[ksz - 2] = xs[c];
                }
                *mixed = acc / (1.0 + (-acc).exp());
            }
            let mut split = m.x_proj.forward(&mixed)?;
            if let Some(eps) = mc.bc_rms_eps {
                let (dt_part, rest) = split.split_at_mut(rank);
                let (b_part, c_part) = rest.split_at_mut(n);
                for (k, part) in [dt_part, b_part, c_part].into_iter().enumerate() {
                    let inv = (part.iter().map(|v| v * v).sum::<f32>() / part.len() as f32 + eps)
                        .sqrt()
                        .recip();
                    match &m.norms {
                        Some(weights) => part.iter_mut().zip(&weights[k]).for_each(|(v, w)| *v *= inv * w),
                        None => part.iter_mut().for_each(|v| *v *= inv),
                    }
                }
            }
            let dt_input = &split[..rank];
            let b = &split[rank..rank + n];
            let c = &split[rank + n..];
            let dt = m.dt_proj.forward(dt_input)?;
            let scan = &mut out[r * inner..(r + 1) * inner];
            scan.par_iter_mut()
                .zip(state.rec.par_chunks_mut(n))
                .enumerate()
                .for_each(|(channel, (y, s))| {
                    let raw = dt[channel];
                    let step = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
                    let xc = mixed[channel];
                    let mut acc = 0.0f32;
                    for k in 0..n {
                        s[k] = s[k] * (step * m.decay[channel * n + k]).exp() + step * b[k] * xc;
                        acc += s[k] * c[k];
                    }
                    let z = gate[channel];
                    *y = (acc + m.skip[channel] * xc) * (z / (1.0 + (-z).exp()));
                });
        }
        drop(store);
        out_projection.forward(&out)
    }

    /// Mamba-2 mixer for a batch of tokens, one token at a time per sequence (selective state
    /// update); see `delta_forward` for how sequences are identified.
    fn mamba_forward(
        &self,
        m: &Mamba,
        out_projection: &Linear,
        layer: usize,
        x: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let mc = cfg.mamba.as_ref().ok_or("missing Mamba configuration")?;
        let rows = lengths.len();
        let (inner, state_size, heads, groups, ksz) =
            (mc.inner, mc.state, mc.heads, mc.groups, mc.conv_kernel);
        let conv_dim = inner + 2 * groups * state_size;
        let head_dim = inner / heads;
        let per_group = heads / groups;
        let projected = m.input.forward(x)?;
        let width = inner + conv_dim + heads;
        let mut out = vec![0.0f32; rows * inner];
        let mut store = self
            .states
            .lock()
            .map_err(|_| "recurrent state lock poisoned")?;
        for r in 0..rows {
            let position = lengths[r] as usize - 1;
            let first = tables[r * self.stride];
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store
                    .map
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k)
                {
                    store.map.remove(&oldest);
                }
            }
            let seq = store.map.entry(first).or_insert_with(|| SeqState {
                used: clock,
                layers: (0..cfg.layers).map(|_| None).collect(),
            });
            seq.used = clock;
            if position == 0 {
                seq.layers[layer] = None;
            }
            let state = seq.layers[layer].get_or_insert_with(|| LayerState {
                len: 0,
                conv: vec![0.0; conv_dim * (ksz - 1)],
                rec: vec![0.0; heads * head_dim * state_size],
            });
            if state.len != position {
                return Err(format!(
                    "recurrent state out of sync (holds {} tokens, got position {position})",
                    state.len
                ));
            }
            state.len += 1;
            let row = &projected[r * width..(r + 1) * width];
            let (gate, rest) = row.split_at(inner);
            let (xbc, dt_raw) = rest.split_at(conv_dim);
            let mut mixed = vec![0.0f32; conv_dim];
            for (c, mixed) in mixed.iter_mut().enumerate() {
                let weights = &m.conv[c * ksz..(c + 1) * ksz];
                let history = &mut state.conv[c * (ksz - 1)..(c + 1) * (ksz - 1)];
                let mut acc =
                    weights[ksz - 1] * xbc[c] + m.conv_bias.as_ref().map_or(0.0, |b| b[c]);
                for (w, h) in weights.iter().zip(history.iter()) {
                    acc += w * h;
                }
                if ksz > 1 {
                    history.copy_within(1.., 0);
                    history[ksz - 2] = xbc[c];
                }
                *mixed = acc / (1.0 + (-acc).exp());
            }
            let (xs, bc) = mixed.split_at(inner);
            let (b_all, c_all) = bc.split_at(groups * state_size);
            let step_min = mc.step_min;
            let scan = &mut out[r * inner..(r + 1) * inner];
            scan.par_chunks_mut(head_dim)
                .zip(state.rec.par_chunks_mut(head_dim * state_size))
                .enumerate()
                .for_each(|(h, (y, s))| {
                    let raw = dt_raw[h] + m.dt_bias[h];
                    let dt = (if raw > 20.0 { raw } else { raw.exp().ln_1p() }).max(step_min).min(mc.step_max);
                    let da = (dt * m.decay[h]).exp();
                    let g = h / per_group.max(1);
                    let b = &b_all[g * state_size..(g + 1) * state_size];
                    let c = &c_all[g * state_size..(g + 1) * state_size];
                    for (p, y) in y.iter_mut().enumerate() {
                        let xp = xs[h * head_dim + p];
                        let cell = &mut s[p * state_size..(p + 1) * state_size];
                        let mut acc = 0.0f32;
                        for n in 0..state_size {
                            cell[n] = cell[n] * da + dt * xp * b[n];
                            acc += cell[n] * c[n];
                        }
                        *y = acc + m.skip[h] * xp;
                    }
                });
            // gate and RMS norm over `norm_groups` groups of the inner width (order and presence per `mc.norm`)
            hybrid::gate_and_norm(mc, &m.norm, mc.norm_eps.unwrap_or(cfg.eps), scan, gate);
        }
        drop(store);
        out_projection.forward(&out)
    }

    /// Smallest batch for which weights that stay in host memory are streamed to the device (below it the CPU
    /// kernels are faster than the PCIe transfer). `FLLM_NO_HOST_STREAM=1` turns the feature off.
    fn host_stream_min_tokens() -> usize {
        static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *MIN.get_or_init(|| {
            if std::env::var_os("FLLM_NO_HOST_STREAM").is_some() {
                return usize::MAX;
            }
            std::env::var("FLLM_HOST_STREAM_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32)
        })
    }

    /// The raw weight of a host-resident projection when the stream can multiply with it directly.
    fn streamable<'a>(
        stream: &dyn crate::backend::Stream,
        l: &'a Linear,
    ) -> Option<StreamableWeight<'a>> {
        if l.scale != 1.0 {
            return None;
        }
        let (kind, bytes) = l.packed_weights()?;
        let quant = crate::backend::QuantType::from_gguf_type_id(kind);
        stream.supports_host_linear(quant).then_some((quant, bytes, l.bias.as_deref()))
    }

    /// The gated activation of the device path (SwiGLU, GeGLU, gpt-oss SwiGLU); the others stay on the host.
    fn device_glu_mode(&self) -> Option<GluMode> {
        // routed experts: SiTU, or the clamped SwiGLU
        if let Some((beta, linear_beta)) = self.cfg.situ {
            return Some(GluMode::Clamped(crate::backend::ClampedGlu::Situ(beta, linear_beta)));
        }
        if let Some(limit) = self.cfg.moe.as_ref().and_then(|m| m.swiglu_clamp) {
            return Some(GluMode::Clamped(crate::backend::ClampedGlu::Clamp(limit)));
        }
        self.plain_glu_mode()
    }

    /// The gated activation of the dense feed-forward layers and of the shared expert: like the routed experts' unless only they are clamped.
    fn dense_glu_mode(&self) -> Option<GluMode> {
        if self.cfg.situ.is_some() || self.cfg.moe.as_ref().is_some_and(|m| m.swiglu_clamp.is_some() && m.swiglu_clamp_shared) {
            return self.device_glu_mode();
        }
        self.plain_glu_mode()
    }

    fn plain_glu_mode(&self) -> Option<GluMode> {
        if let Some((alpha, limit)) = self.cfg.moe.as_ref().and_then(|m| m.swiglu_oai) {
            return Some(GluMode::SwigluOai(alpha, limit));
        }
        match self.cfg.activation {
            Activation::Silu => Some(GluMode::Swiglu),
            Activation::GeluTanh => Some(GluMode::Geglu),
            _ => None,
        }
    }

    /// Weights whose GEMM can leave half precision for the gated activation that follows: 4-bit Marlin, or NVFP4 on the CUTLASS path (the
    /// backend checks the batch and declines when it cannot).
    fn half_out_weight(w: &crate::backend::DeviceWeight) -> bool {
        w.quant_type.q4z_group().is_some() || w.quant_type.to_gguf_type_id() == 40
    }

    fn device_glu(
        stream: &mut dyn crate::backend::Stream,
        mode: GluMode,
        out: &mut crate::backend::DeviceBuffer,
        gate: &crate::backend::DeviceBuffer,
        up: &crate::backend::DeviceBuffer,
    ) -> Result<(), String> {
        match mode {
            GluMode::Swiglu => stream.swiglu(out, gate, up),
            GluMode::Geglu => stream.geglu(out, gate, up),
            GluMode::SwigluOai(alpha, limit) => stream.swiglu_oai(out, gate, up, alpha, limit),
            GluMode::Clamped(kind) => stream.glu_clamped(out, gate, up, kind),
        }
        .map_err(|e| e.to_string())
    }

    /// `out = down(act(gate(x)) * up(x))` for `rows` rows of the device buffer `x`: with device-resident weights, or with
    /// every weight streamed over PCIe for its product (`host_linear`; the weights must pass `streamable`).
    fn gated_run(
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        mode: GluMode,
        src: &ExpertSrc<'_>,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<crate::backend::DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let (gr, ur, dr) = match src {
            ExpertSrc::Device(e) => (e.gate.rows, e.up.rows, e.down.rows),
            ExpertSrc::Host(w) => (w[0].2, w[1].2, w[2].2),
        };
        let mut gate_buf = scratch.acquire(rows * gr * 4).map_err(err)?;
        let mut up_buf = scratch.acquire(rows * ur * 4).map_err(err)?;
        let mut act_buf = scratch.acquire(rows * gr * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * dr * 4).map_err(err)?;
        match src {
            ExpertSrc::Device(e) => {
                stream.quantized_linear(&mut gate_buf, x, &e.gate, e.gate.bias.as_ref()).map_err(err)?;
                stream.quantized_linear(&mut up_buf, x, &e.up, e.up.bias.as_ref()).map_err(err)?;
            }
            ExpertSrc::Host([(gq, gb, gr, gc, gbias), (uq, ub, ur, uc, ubias), _]) => {
                stream.host_linear(&mut gate_buf, x, gb, *gq, *gr, *gc, *gbias).map_err(err)?;
                stream.host_linear(&mut up_buf, x, ub, *uq, *ur, *uc, *ubias).map_err(err)?;
            }
        }
        Self::device_glu(stream, mode, &mut act_buf, &gate_buf, &up_buf)?;
        match src {
            ExpertSrc::Device(e) => {
                stream.quantized_linear(&mut out, &act_buf, &e.down, e.down.bias.as_ref()).map_err(err)?;
            }
            ExpertSrc::Host([_, _, (dq, db, dr, dc, dbias)]) => {
                stream.host_linear(&mut out, &act_buf, db, *dq, *dr, *dc, *dbias).map_err(err)?;
            }
        }
        scratch.recycle(gate_buf);
        scratch.recycle(up_buf);
        scratch.recycle(act_buf);
        Ok(out)
    }

    /// The three raw weights of a gated FFN / expert when all of them can be streamed.
    fn gated_weights<'a>(
        stream: &dyn crate::backend::Stream,
        gate: &'a Linear,
        up: &'a Linear,
        down: &'a Linear,
    ) -> Option<[HostWeight<'a>; 3]> {
        let g = Self::streamable(stream, gate)?;
        let u = Self::streamable(stream, up)?;
        let d = Self::streamable(stream, down)?;
        Some([
            (g.0, g.1, gate.rows, gate.cols, g.2),
            (u.0, u.1, up.rows, up.cols, u.2),
            (d.0, d.1, down.rows, down.cols, d.2),
        ])
    }

    /// The source of one expert: its device copy, else its host weights when the stream can take them.
    fn expert_src<'a>(
        stream: &dyn crate::backend::Stream,
        e: &'a Expert,
        dev: Option<&'a DeviceExpert>,
    ) -> Option<ExpertSrc<'a>> {
        if let Some(d) = dev {
            return Some(ExpertSrc::Device(d));
        }
        let gate = e.gate.as_ref()?;
        Self::gated_weights(stream, gate, &e.up, &e.down).map(ExpertSrc::Host)
    }

    /// Whether the routing of this layer is what the device kernels implement. Without `device` (the grouped operations): softmax over all
    /// experts, the best `used`, optionally renormalised, no bias, groups or shared expert. With it: softmax or sigmoid scores, selection
    /// bias, group-limited selection, shared expert, scores applied to the expert input (see `Backend::moe_device_available`).
    fn device_routing_ok(&self, moe: &Moe, device: bool) -> bool {
        let Some(cfg) = self.cfg.moe.as_ref() else {
            return false;
        };
        // the ranks of a tensor-parallel group hold shards (or a subset) of the experts: their routing runs on the host, where the
        // routes to experts of other ranks are dropped
        if self.par.as_ref().is_some_and(|p| p.tp_size() > 1) {
            return false;
        }
        if device {
            return moe.latent.is_none()
                // zero experts (LongCat-Flash) fold into the combination's shared slot: a router with both, or with scaled inputs, keeps the other paths
                && (cfg.zero_experts == 0 || (moe.shared.is_none() && !cfg.input_scaled && cfg.experts > cfg.zero_experts))
                && moe.tid2eid.is_none()
                && matches!(cfg.gating, Gating::Softmax | Gating::Sigmoid)
                && cfg.weight_norm_p.is_none()
                && (1..=32).contains(&cfg.used)
                && cfg.experts >= cfg.used
                && cfg.experts <= 1024
                && (cfg.groups <= 1 || (cfg.groups <= 64 && cfg.experts.is_multiple_of(cfg.groups)));
        }
        matches!(cfg.gating, Gating::Softmax)
            && !cfg.input_scaled
            && cfg.groups <= 1
            && cfg.weight_norm_p.is_none()
            && moe.bias.is_none()
            && moe.shared.is_none()
            && self.cfg.arch != "lfm2moe"
            && (1..=16).contains(&cfg.used)
            && cfg.experts >= cfg.used
    }

    /// A whole mixture-of-experts layer for one to four tokens without leaving the device: router, top-k, the selected
    /// experts' three projections through pointer tables, and the weighted sum. `None` when the layer does not qualify.
    fn moe_fast(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        dm: &DeviceMoe,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<Option<crate::backend::DeviceBuffer>, String> {
        let (Some(fast), Some(cfg), Some(mode)) = (dm.fast.as_ref(), self.cfg.moe.as_ref(), self.device_glu_mode()) else {
            return Ok(None);
        };
        if let Some(device) = fast.device.as_ref() {
            return self.moe_device(stream, scratch, dm, fast, device, x, rows);
        }
        if rows == 0 || rows > 4 || cfg.zero_experts > 0 {
            return Ok(None);
        }
        let err = |e: crate::backend::BackendError| e.to_string();
        let (experts, k, hidden, ff) = (cfg.experts, cfg.used, self.cfg.hidden, fast.ff);
        let slots = rows * k;
        let mut logits = scratch.acquire(rows * experts * 4).map_err(err)?;
        stream.quantized_linear(&mut logits, x, &fast.router, fast.router.bias.as_ref()).map_err(err)?;
        let mut ids = scratch.acquire(slots * 4).map_err(err)?;
        let mut weights = scratch.acquire(slots * 4).map_err(err)?;
        stream
            .moe_topk(&logits, &mut ids, &mut weights, rows, experts, k, cfg.normalize, cfg.weight_scale)
            .map_err(err)?;
        let mut gate = scratch.acquire(slots * ff * 4).map_err(err)?;
        let mut up = scratch.acquire(slots * ff * 4).map_err(err)?;
        let mut act = scratch.acquire(slots * ff * 4).map_err(err)?;
        let mut y = scratch.acquire(slots * hidden * 4).map_err(err)?;
        let mut out = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream
            .moe_linear(&mut gate, x, &fast.gate, fast.gate_bias.as_ref(), &ids, fast.kinds[0], ff, hidden, slots, k)
            .map_err(err)?;
        stream
            .moe_linear(&mut up, x, &fast.up, fast.up_bias.as_ref(), &ids, fast.kinds[1], ff, hidden, slots, k)
            .map_err(err)?;
        Self::device_glu(stream, mode, &mut act, &gate, &up)?;
        stream
            .moe_linear(&mut y, &act, &fast.down, fast.down_bias.as_ref(), &ids, fast.kinds[2], hidden, ff, slots, 1)
            .map_err(err)?;
        stream.moe_combine(&mut out, &y, &weights, rows, k, hidden).map_err(err)?;
        for b in [logits, ids, weights, gate, up, act, y] {
            scratch.recycle(b);
        }
        Ok(Some(out))
    }

    /// A mixture-of-experts layer on the device: routing on the host from `host_in` (the normed input), the experts of
    /// the batch run one after the other (resident ones directly, the others streamed over PCIe) and accumulate into
    /// one device buffer. `None` (nothing done) when an expert that is needed has neither.
    #[allow(clippy::too_many_arguments)]
    fn moe_on_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        moe: &Moe,
        dev: Option<&DeviceMoe>,
        x: &crate::backend::DeviceBuffer,
        host_in: &[f32],
        rows: usize,
    ) -> Result<Option<crate::backend::DeviceBuffer>, String> {
        let Some(mode) = self.device_glu_mode() else {
            return Ok(None);
        };
        let Some(cfg) = self.cfg.moe.as_ref() else {
            return Ok(None);
        };
        let hidden = self.cfg.hidden;
        let mut routed = self.moe_route(moe, host_in)?;
        // expert parallelism: this rank computes the experts it holds; the rest of the group adds its share in `device_tp_reduce`
        self.par_filter_routes(&mut routed);
        let mut plan = Vec::new();
        for (n, (expert, assigned)) in moe.experts.iter().zip(&routed).enumerate() {
            if assigned.is_empty() {
                continue;
            }
            let resident = dev.and_then(|d| d.experts.get(n)).and_then(|e| e.as_ref());
            let Some(src) = Self::expert_src(stream, expert, resident) else {
                return Ok(None);
            };
            plan.push((src, assigned));
        }
        let shared = match &moe.shared {
            Some((e, gate_vec)) => {
                let Some(src) = Self::expert_src(stream, e, dev.and_then(|d| d.shared.as_ref())) else {
                    return Ok(None);
                };
                Some((src, gate_vec))
            }
            None => None,
        };
        let err = |e: crate::backend::BackendError| e.to_string();
        let mut acc = scratch.acquire(rows * hidden * 4).map_err(err)?;
        if stream.zero_buffer(&mut acc).is_err() {
            // a backend without the accumulate kernels: the host computes the layer
            scratch.recycle(acc);
            return Ok(None);
        }
        for (src, assigned) in plan {
            let idx: Vec<i32> = assigned.iter().map(|&(t, _)| t as i32).collect();
            let weights: Vec<f32> = assigned.iter().map(|&(_, wt)| wt).collect();
            let n = idx.len();
            // every token goes to this expert, in order (always so for a single token): no gather needed
            let direct = n == rows && !cfg.input_scaled && idx.iter().enumerate().all(|(i, &t)| t as usize == i);
            let mut gathered = None;
            if !direct {
                let mut xe = scratch.acquire(n * hidden * 4).map_err(err)?;
                stream
                    .gather_rows(&mut xe, x, &idx, cfg.input_scaled.then_some(weights.as_slice()), hidden)
                    .map_err(err)?;
                gathered = Some(xe);
            }
            let result = Self::gated_run(stream, scratch, mode, &src, gathered.as_ref().unwrap_or(x), n)?;
            stream
                .scatter_add_rows(&mut acc, &result, &idx, (!cfg.input_scaled).then_some(weights.as_slice()), hidden)
                .map_err(err)?;
            if let Some(xe) = gathered {
                scratch.recycle(xe);
            }
            scratch.recycle(result);
        }
        // zero experts (LongCat-Flash): the routes beyond the stored experts return the input, weighted; one pass for all of them
        if cfg.zero_experts > 0 {
            let mut total = vec![0.0f32; rows];
            for assigned in routed.iter().skip(moe.experts.len()) {
                for &(t, wt) in assigned {
                    total[t] += wt;
                }
            }
            let idx: Vec<i32> = (0..rows as i32).filter(|&t| total[t as usize] != 0.0).collect();
            if !idx.is_empty() {
                let weights: Vec<f32> = idx.iter().map(|&t| total[t as usize]).collect();
                let mut xe = scratch.acquire(idx.len() * hidden * 4).map_err(err)?;
                stream.gather_rows(&mut xe, x, &idx, None, hidden).map_err(err)?;
                stream.scatter_add_rows(&mut acc, &xe, &idx, Some(&weights), hidden).map_err(err)?;
                scratch.recycle(xe);
            }
        }
        if let Some((src, gate_vec)) = shared {
            let result = Self::gated_run(stream, scratch, self.dense_glu_mode().unwrap_or(mode), &src, x, rows)?;
            let idx: Vec<i32> = (0..rows as i32).collect();
            let scales: Option<Vec<f32>> = gate_vec.as_ref().map(|g| {
                (0..rows)
                    .map(|t| {
                        let xt = &host_in[t * hidden..(t + 1) * hidden];
                        let dot: f32 = g.iter().zip(xt).map(|(a, b)| a * b).sum();
                        1.0 / (1.0 + (-dot).exp())
                    })
                    .collect()
            });
            stream
                .scatter_add_rows(&mut acc, &result, &idx, scales.as_deref(), hidden)
                .map_err(err)?;
            scratch.recycle(result);
        }
        if self.par.as_ref().is_some_and(|p| p.tp_size() > 1) {
            self.device_tp_reduce(stream, &mut acc, rows * hidden)?;
        }
        Ok(Some(acc))
    }

    fn expert_forward(
        &self,
        e: &Expert,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<Vec<f32>, String> {
        self.expert_forward_clamped(e, input, workspace, true)
    }

    /// `expert_forward`; `clamp` false skips the clamp of the SwiGLU limit (a shared expert that is not clamped).
    fn expert_forward_clamped(
        &self,
        e: &Expert,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
        clamp: bool,
    ) -> Result<Vec<f32>, String> {
        let Some(gate) = &e.gate else {
            let mut hidden = e.up.forward(input)?;
            let activation = self.cfg.activation;
            hidden
                .par_iter_mut()
                .for_each(|v| *v = activate(activation, *v));
            return e.down.forward(&hidden);
        };
        let projected = Linear::forward_group_with_workspace(&[gate, &e.up], input, workspace)?;
        let [g, u] = projected else {
            return Err("invalid expert projection group".into());
        };
        self.gate_activation_clamped(g, u, true, clamp);
        e.down.forward(g)
    }

    /// `g = act(g) * u` of a gated expert FFN. Elementwise, so the serial form (used between
    /// the fused expert passes, where the vectors are too short for a fork/join) gives the
    /// same bits as the parallel one.
    fn gate_activation(&self, g: &mut [f32], u: &[f32], parallel: bool) {
        self.gate_activation_clamped(g, u, parallel, true);
    }

    fn gate_activation_clamped(&self, g: &mut [f32], u: &[f32], parallel: bool, clamp: bool) {
        let activation = self.cfg.activation;
        if let Some((beta, linear_beta)) = self.cfg.situ {
            let apply = |g: &mut f32, u: &f32| {
                let gate = *g;
                let up = if linear_beta > 0.0 { linear_beta * (u / linear_beta).tanh() } else { *u };
                *g = beta * (gate / beta).tanh() / (1.0 + (-gate).exp()) * up;
            };
            if parallel {
                g.par_iter_mut().zip(u.par_iter()).for_each(|(g, u)| apply(g, u));
            } else {
                g.iter_mut().zip(u).for_each(|(g, u)| apply(g, u));
            }
        } else if let Some(limit) = self.cfg.moe.as_ref().and_then(|m| m.swiglu_clamp).filter(|_| clamp) {
            let apply = |g: &mut f32, u: &f32| {
                let gate = g.min(limit);
                *g = gate / (1.0 + (-gate).exp()) * u.clamp(-limit, limit);
            };
            if parallel {
                g.par_iter_mut().zip(u.par_iter()).for_each(|(g, u)| apply(g, u));
            } else {
                g.iter_mut().zip(u).for_each(|(g, u)| apply(g, u));
            }
        } else if let Some((alpha, limit)) = self.cfg.moe.as_ref().and_then(|m| m.swiglu_oai) {
            let apply = |g: &mut f32, u: &f32| {
                let gate = g.min(limit);
                let up = u.clamp(-limit, limit);
                *g = (up + 1.0) * gate / (1.0 + (-alpha * gate).exp());
            };
            if parallel {
                g.par_iter_mut()
                    .zip(u.par_iter())
                    .for_each(|(g, u)| apply(g, u));
            } else {
                g.iter_mut().zip(u).for_each(|(g, u)| apply(g, u));
            }
        } else if activation == Activation::Silu {
            if parallel {
                crate::cpu_elementwise::swiglu(g, u);
            } else {
                crate::cpu_elementwise::swiglu_serial(g, u);
            }
        } else if parallel {
            g.par_iter_mut()
                .zip(u.par_iter())
                .for_each(|(g, u)| *g = activate(activation, *g) * u);
        } else {
            g.iter_mut()
                .zip(u)
                .for_each(|(g, u)| *g = activate(activation, *g) * u);
        }
    }

    /// Decode (one token) through the fused expert passes; `false` (with `out` untouched) when
    /// an active expert does not qualify and the caller has to run the experts one by one.
    /// Experts are accumulated in ascending expert order like the per-expert loop does.
    fn fused_experts(
        &self,
        moe: &Moe,
        routed: &[Vec<(usize, f32)>],
        input: &[f32],
        out: &mut [f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> bool {
        let mut members = Vec::with_capacity(self.cfg.moe.as_ref().map_or(8, |m| m.used));
        let mut weights = Vec::with_capacity(members.capacity());
        for (expert, assigned) in moe.experts.iter().zip(routed) {
            let Some(&(_, weight)) = assigned.first() else {
                continue;
            };
            let Some(gate) = &expert.gate else {
                return false;
            };
            members.push((gate, &expert.up, &expert.down));
            weights.push(weight);
        }
        if members.is_empty() {
            return true;
        }
        let activate = |g: &mut [f32], u: &[f32]| self.gate_activation(g, u, false);
        let Some(results) = Linear::forward_experts(&members, input, &activate, workspace) else {
            return false;
        };
        if results.iter().any(|r| r.len() != out.len()) {
            return false;
        }
        for (r, &w) in results.iter().zip(&weights) {
            for (d, r) in out.iter_mut().zip(r) {
                *d += w * r;
            }
        }
        true
    }

    /// The shortcut experts of a LongCat-Flash layer for the normed input `x` on the device: the device routing (`moe_fast`) where the backend has it,
    /// else routing on the host with the experts on the device (`moe_on_device`, as for every other mixture-of-experts layer), else the host reference.
    fn shortcut_device(
        &self,
        stream: &mut dyn crate::backend::Stream,
        scratch: &mut crate::backend::BufferPool,
        moe: &Moe,
        dev: Option<&DeviceMoe>,
        x: &crate::backend::DeviceBuffer,
        rows: usize,
    ) -> Result<crate::backend::DeviceBuffer, String> {
        let err = |e: crate::backend::BackendError| e.to_string();
        let hidden = self.cfg.hidden;
        let mut host_in = Vec::new();
        if let Some(dm) = dev {
            if let Some(buf) = self.moe_fast(stream, scratch, dm, x, rows)? {
                return Ok(buf);
            }
            host_in = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut host_in), x).map_err(err)?;
            if let Some(buf) = self.moe_on_device(stream, scratch, moe, Some(dm), x, &host_in, rows)? {
                return Ok(buf);
            }
        }
        if host_in.is_empty() {
            host_in = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut host_in), x).map_err(err)?;
        }
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        let out = self.shortcut_forward(moe, &host_in, &mut workspace)?;
        let mut buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
        stream.upload(&mut buf, cast_f32_to_u8(&out)).map_err(err)?;
        Ok(buf)
    }

    /// LongCat-Flash shortcut experts: the routed experts as usual, a zero expert (index beyond the stored experts) returns its input.
    fn shortcut_forward(&self, moe: &Moe, input: &[f32], workspace: &mut compute::ProjectionGroupWorkspace) -> Result<Vec<f32>, String> {
        let hidden = self.cfg.hidden;
        let routed = self.moe_route(moe, input)?;
        let mut out = vec![0.0f32; input.len()];
        for (e, assigned) in routed.iter().enumerate().filter(|(_, a)| !a.is_empty()) {
            match moe.experts.get(e) {
                Some(expert) => {
                    let mut rows = Vec::with_capacity(assigned.len() * hidden);
                    for &(t, _) in assigned {
                        rows.extend_from_slice(&input[t * hidden..(t + 1) * hidden]);
                    }
                    let result = self.expert_forward(expert, &rows, workspace)?;
                    for (n, &(t, w)) in assigned.iter().enumerate() {
                        out[t * hidden..(t + 1) * hidden].iter_mut().zip(&result[n * hidden..(n + 1) * hidden]).for_each(|(o, r)| *o += w * r);
                    }
                }
                None => {
                    for &(t, w) in assigned {
                        out[t * hidden..(t + 1) * hidden].iter_mut().zip(&input[t * hidden..(t + 1) * hidden]).for_each(|(o, x)| *o += w * x);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Which tokens go to which expert, with their weights: `routed[expert] = [(token, weight)]`.
    fn moe_route(&self, moe: &Moe, input: &[f32]) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let cfg = self
            .cfg
            .moe
            .as_ref()
            .ok_or("MoE layer without MoE configuration")?;
        let hidden = self.cfg.hidden;
        let tokens = input.len() / hidden;
        let mut logits = moe.router.forward(input)?;
        let hashed_ids: Vec<i64> = if moe.tid2eid.is_some() { self.row_ids.lock().map_err(|_| "row id lock poisoned")?.clone() } else { Vec::new() };
        if moe.tid2eid.is_some() && hashed_ids.len() != tokens {
            return Err(format!("hash routing needs the token ids of {tokens} rows, got {}", hashed_ids.len()));
        }
        if let Some(cap) = cfg.router_softcap {
            logits.iter_mut().for_each(|l| *l = cap * (*l / cap).tanh());
        }
        let experts = cfg.experts;
        // (token, weight) per expert
        let mut routed: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
        let mut order: Vec<usize> = Vec::with_capacity(experts);
        for t in 0..tokens {
            let row = &logits[t * experts..(t + 1) * experts];
            if let Gating::SparseMixer { jitter_eps } = cfg.gating {
                let mut selected = vec![false; experts];
                let mut chosen = Vec::with_capacity(cfg.used);
                for _ in 0..cfg.used {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_ind = None;
                    for (i, &score) in row.iter().enumerate() {
                        if !selected[i] && score > max_val {
                            max_val = score;
                            max_ind = Some(i);
                        }
                    }
                    let Some(best_i) = max_ind else { break };
                    selected[best_i] = true;
                    let mut max_masked = f32::NEG_INFINITY;
                    let mut masked = vec![f32::NEG_INFINITY; experts];
                    for i in 0..experts {
                        if !selected[i] || i == best_i {
                            let score = if selected[i] && i != best_i {
                                f32::NEG_INFINITY
                            } else {
                                row[i]
                            };
                            let factor = score.abs().max(max_val);
                            if factor > 0.0 && ((max_val - score) / factor) > 2.0 * jitter_eps {
                                // thresholded out
                            } else {
                                masked[i] = score;
                                if score > max_masked {
                                    max_masked = score;
                                }
                            }
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for &m in &masked {
                        if m != f32::NEG_INFINITY {
                            sum_exp += (m - max_masked).exp();
                        }
                    }
                    let weight = if sum_exp > 0.0 {
                        (masked[best_i] - max_masked).exp() / sum_exp
                    } else {
                        1.0
                    };
                    chosen.push((best_i, weight));
                }
                for (i, w) in chosen {
                    routed[i].push((t, w * cfg.weight_scale));
                }
                continue;
            }
            let scores: Vec<f32> = match cfg.gating {
                Gating::Softmax => {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    exps.iter().map(|x| x / sum).collect()
                }
                Gating::Sigmoid => row.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
                Gating::SqrtSoftplus => row.iter().map(|&x| (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt()).collect(),
                Gating::SparseMixer { .. } => unreachable!(),
            };
            order.clear();
            order.extend(0..experts);
            let mut selection = match &moe.bias {
                Some(bias) => scores.iter().zip(bias).map(|(s, b)| s + b).collect(),
                None => scores.clone(),
            };
            if cfg.groups > 1 {
                let per = experts / cfg.groups;
                let mut ranked: Vec<(usize, f32)> = (0..cfg.groups)
                    .map(|g| {
                        let mut group = selection[g * per..(g + 1) * per].to_vec();
                        group.sort_by(|a, b| b.total_cmp(a));
                        let value = if moe.bias.is_some() || cfg.group_top2 {
                            group.iter().take(2).sum()
                        } else {
                            group[0]
                        };
                        (g, value)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for &(g, _) in &ranked[cfg.groups_used.min(cfg.groups)..] {
                    selection[g * per..(g + 1) * per].fill(f32::NEG_INFINITY);
                }
            }
            // The comparator is a strict total order (ties go to the lower index), so selecting
            // the best `used` and sorting only those gives the order a full sort would.
            let rank = |&a: &usize, &b: &usize| selection[b].total_cmp(&selection[a]).then(a.cmp(&b));
            if cfg.used < order.len() {
                order.select_nth_unstable_by(cfg.used - 1, rank);
                order[..cfg.used].sort_by(rank);
            } else {
                order.sort_by(rank);
            }
            let hashed: Option<Vec<usize>> = moe.tid2eid.as_ref().map(|table| {
                let id = (hashed_ids[t].max(0) as usize).min(table.len() / cfg.used - 1);
                table[id * cfg.used..(id + 1) * cfg.used].iter().map(|&e| e as usize).collect()
            });
            let chosen = hashed.as_deref().unwrap_or(&order[..cfg.used]);
            let total: f32 = if let Some(p) = cfg.weight_norm_p {
                chosen
                    .iter()
                    .map(|&i| scores[i].powf(p))
                    .sum::<f32>()
                    .powf(1.0 / p)
            } else {
                chosen.iter().map(|&i| scores[i]).sum()
            };
            // LFM2-MoE's sigmoid router includes epsilon in the selected-weight denominator.
            let total = if self.cfg.arch == "lfm2moe" {
                total + 1e-6
            } else {
                total
            };
            for &i in chosen {
                let w = if cfg.normalize || cfg.weight_norm_p.is_some() {
                    scores[i] / total
                } else {
                    scores[i]
                };
                routed[i].push((t, w * cfg.weight_scale));
            }
        }
        if let Some(cache) = &moe.cache {
            let active: Vec<usize> = routed.iter().enumerate().filter(|(_, r)| !r.is_empty()).map(|(e, _)| e).collect();
            cache.on_route(&active, tokens);
        }
        if std::env::var_os("FLLM_DEBUG_SUMS").is_some() {
            let picks: Vec<(usize, f32)> = routed.iter().enumerate().flat_map(|(e, v)| v.iter().filter(|(t, _)| *t == 0).map(move |&(_, w)| (e, w))).collect();
            eprintln!("[sum] route token0 {:?} logits-sum {:.6}", picks, logits.iter().map(|&v| f64::from(v)).sum::<f64>());
        }
        Ok(routed)
    }

    /// The expert FFNs of a prompt batch through the joined all-expert queues
    /// (`Linear::forward_experts_prompt`): one result matrix per expert in `active`, or `None`
    /// when the layer does not qualify. The caller combines them in ascending expert order.
    fn moe_prompt_experts(
        &self,
        moe: &Moe,
        routed: &[Vec<(usize, f32)>],
        active: &[usize],
        input: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        let cfg = self.cfg.moe.as_ref()?;
        let hidden = self.cfg.hidden;
        let gathered = active
            .iter()
            .map(|&e| {
                let assigned = &routed[e];
                let mut rows = vec![0.0f32; assigned.len() * hidden];
                let fill = |(row, &(t, w)): (&mut [f32], &(usize, f32))| {
                    let source = &input[t * hidden..(t + 1) * hidden];
                    if cfg.input_scaled {
                        for (d, x) in row.iter_mut().zip(source) {
                            *d = x * w;
                        }
                    } else {
                        row.copy_from_slice(source);
                    }
                };
                if assigned.len() >= 16 {
                    rows.par_chunks_mut(hidden).zip(assigned.par_iter()).for_each(fill);
                } else {
                    rows.chunks_mut(hidden).zip(assigned.iter()).for_each(fill);
                }
                rows
            })
            .collect::<Vec<_>>();
        let mut members = Vec::with_capacity(active.len());
        for (&e, rows) in active.iter().zip(&gathered) {
            let expert = &moe.experts[e];
            members.push((expert.gate.as_ref()?, &expert.up, &expert.down, rows.as_slice()));
        }
        let activate = |g: &mut [f32], u: &[f32]| self.gate_activation(g, u, false);
        Linear::forward_experts_prompt(&members, &activate)
    }

    fn moe_forward(
        &self,
        moe: &Moe,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<Vec<f32>, String> {
        let cfg = self
            .cfg
            .moe
            .as_ref()
            .ok_or("MoE layer without MoE configuration")?;
        let hidden = self.cfg.hidden;
        let tokens = input.len() / hidden;
        let mut routed = self.moe_route(moe, input)?;
        self.par_filter_routes(&mut routed);
        let latent_in;
        let (expert_input, width): (&[f32], usize) = match &moe.latent {
            Some((down, _)) => {
                latent_in = down.forward(input)?;
                let w = latent_in.len() / tokens.max(1);
                (&latent_in[..], w)
            }
            None => (input, hidden),
        };
        let mut out = vec![0.0f32; tokens * width];
        let fused = tokens == 1
            && moe.latent.is_none()
            && !cfg.input_scaled
            && self.fused_experts(moe, &routed, expert_input, &mut out, workspace);
        // A prompt batch keeps every expert's rows and combines them per token afterwards, in
        // ascending expert order like the sequential accumulation, but spread over the pool.
        let batched = !fused && tokens >= 16;
        let mut results: Vec<Vec<f32>> = Vec::new();
        let mut pending: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); if batched { tokens } else { 0 }];
        // All experts at once when the batch qualifies; `done` then replaces the per-expert loop.
        let done = batched && moe.latent.is_none() && {
            let active: Vec<usize> = (0..routed.len()).filter(|&e| !routed[e].is_empty()).collect();
            match self.moe_prompt_experts(moe, &routed, &active, expert_input) {
                Some(all) => {
                    for (slot, (&expert, result)) in active.iter().zip(all).enumerate() {
                        for (i, &(t, w)) in routed[expert].iter().enumerate() {
                            pending[t].push((slot, i, if cfg.input_scaled { 1.0 } else { w }));
                        }
                        results.push(result);
                    }
                    true
                }
                None => false,
            }
        };
        for (expert, assigned) in moe.experts.iter().zip(&routed) {
            if fused || done || assigned.is_empty() {
                continue;
            }
            let mut gathered = vec![0.0f32; assigned.len() * width];
            let fill = |(row, &(t, w)): (&mut [f32], &(usize, f32))| {
                let source = &expert_input[t * width..(t + 1) * width];
                if cfg.input_scaled {
                    for (d, x) in row.iter_mut().zip(source) {
                        *d = x * w;
                    }
                } else {
                    row.copy_from_slice(source);
                }
            };
            if assigned.len() >= 16 {
                gathered
                    .par_chunks_mut(width)
                    .zip(assigned.par_iter())
                    .for_each(fill);
            } else {
                gathered.chunks_mut(width).zip(assigned.iter()).for_each(fill);
            }
            let result = self.expert_forward(expert, &gathered, workspace)?;
            if batched {
                for (i, &(t, w)) in assigned.iter().enumerate() {
                    pending[t].push((results.len(), i, if cfg.input_scaled { 1.0 } else { w }));
                }
                results.push(result);
                continue;
            }
            for (i, &(t, w)) in assigned.iter().enumerate() {
                let w = if cfg.input_scaled { 1.0 } else { w };
                let dst = &mut out[t * width..(t + 1) * width];
                for (d, r) in dst.iter_mut().zip(&result[i * width..(i + 1) * width]) {
                    *d += w * r;
                }
            }
        }
        if batched {
            out.par_chunks_mut(width)
                .zip(pending.par_iter())
                .for_each(|(dst, parts)| {
                    for &(e, i, w) in parts {
                        for (d, r) in dst.iter_mut().zip(&results[e][i * width..(i + 1) * width]) {
                            *d += w * r;
                        }
                    }
                });
        }
        if let Some((_, up)) = &moe.latent {
            if let Some(norm) = &moe.latent_norm {
                rms(&mut out, norm, self.cfg.eps);
            }
            out = up.forward(&out)?;
        }
        if let Some((shared, gate)) = &moe.shared {
            let result = self.expert_forward_clamped(shared, input, workspace, cfg.swiglu_clamp_shared)?;
            for t in 0..tokens {
                let x = &input[t * hidden..(t + 1) * hidden];
                let scale = gate.as_ref().map_or(1.0, |g| {
                    let dot: f32 = g.iter().zip(x).map(|(a, b)| a * b).sum();
                    1.0 / (1.0 + (-dot).exp())
                });
                for (d, r) in out[t * hidden..(t + 1) * hidden]
                    .iter_mut()
                    .zip(&result[t * hidden..(t + 1) * hidden])
                {
                    *d += scale * r;
                }
            }
        }
        Ok(out)
    }

    /// MoVA values for a batch of tokens: route to `used` of the value experts, `sum_k w_k * silu(W_k x)`.
    fn value_moe_forward(&self, vm: &ValueMoe, input: &[f32]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref();
        let hidden = cfg.hidden;
        let rows = input.len() / hidden;
        let width = cfg.kv_heads * cfg.value_dim;
        let logits = vm.router.forward(input)?;
        let count = vm.experts.len();
        let softmax = moe.is_some_and(|m| matches!(m.gating, crate::config::Gating::Softmax));
        let normalize = moe.is_none_or(|m| m.normalize);
        let scale = moe.map_or(1.0, |m| m.weight_scale);
        let mut out = vec![0.0f32; rows * width];
        for t in 0..rows {
            let row = &logits[t * count..(t + 1) * count];
            let scores: Vec<f32> = if softmax {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            } else {
                row.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
            };
            let mut order: Vec<usize> = (0..count).collect();
            let selection = |e: usize| scores[e] + vm.bias.as_ref().map_or(0.0, |b| b[e]);
            order.sort_by(|&a, &b| selection(b).total_cmp(&selection(a)).then(a.cmp(&b)));
            order.truncate(vm.used);
            let mut weights: Vec<f32> = order.iter().map(|&e| scores[e]).collect();
            if normalize {
                let sum: f32 = weights.iter().sum::<f32>().max(6.103_515_6e-5);
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            let x = &input[t * hidden..(t + 1) * hidden];
            let dst = &mut out[t * width..(t + 1) * width];
            for (&e, w) in order.iter().zip(&weights) {
                let value = vm.experts[e].forward(x)?;
                for (d, v) in dst.iter_mut().zip(&value) {
                    *d += w * scale * (v / (1.0 + (-v).exp()));
                }
            }
        }
        Ok(out)
    }

    /// The feed-forward of a layer on this rank's share of the weights (`ffn_forward` in `parallel/decoder_shard.rs` adds the sum
    /// over the tensor-parallel group).
    fn ffn_forward_local(
        &self,
        layer: &Layer,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
    ) -> Result<Vec<f32>, String> {
        match &layer.ffn {
            Ffn::Gated { gate, up, down } => {
                let projected =
                    Linear::forward_group_with_workspace(&[gate, up], input, workspace)?;
                let [g, u] = projected else {
                    return Err("invalid FFN projection group".into());
                };
                if let Some(factor) = layer.gemma.as_ref().map(|g| g.sparsity()).filter(|f| *f != 0.0) {
                    let width = g.len() / (input.len() / self.cfg.hidden).max(1);
                    gemma::gaussian_top_k(g, width, factor);
                }
                if self.cfg.situ.is_some() || self.cfg.moe.as_ref().is_some_and(|m| m.swiglu_clamp.is_some() && m.swiglu_clamp_shared) {
                    // the dense layers of a model with a clamped SwiGLU clamp like its shared expert (GLM-5-Next)
                    self.gate_activation_clamped(g, u, true, true);
                } else if self.cfg.activation == Activation::Silu {
                    crate::cpu_elementwise::swiglu(g, u);
                } else {
                    let activation = self.cfg.activation;
                    g.par_iter_mut()
                        .zip(u.par_iter())
                        .for_each(|(g, u)| *g = activate(activation, *g) * u);
                }
                if let Some(weight) = &layer.ffn_sub_norm {
                    rms(g, weight, self.cfg.eps);
                }
                down.forward(g)
            }
            Ffn::None => Err("layer without a feed-forward".into()),
            Ffn::Moe(moe) => self.moe_forward(moe, input, workspace),
            Ffn::Plain { up, down } => {
                let mut hidden = up.forward(input)?;
                let activation = self.cfg.activation;
                if let Some([alpha_p, alpha_n, beta, eps]) = layer.xielu {
                    hidden.par_iter_mut().for_each(|x| {
                        *x = if *x > 0.0 {
                            alpha_p * *x * *x + beta * *x
                        } else {
                            ((*x).min(eps).exp_m1() - *x) * alpha_n + beta * *x
                        };
                    });
                } else {
                    hidden
                        .par_iter_mut()
                        .for_each(|x| *x = activate(activation, *x));
                }
                down.forward(&hidden)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
    ) -> Result<Vec<f32>, String> {
        self.forward_with_embeddings(
            ids,
            lengths,
            tables,
            key,
            value,
            logit_rows,
            embeddings,
            &[],
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_embeddings(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
    ) -> Result<Vec<f32>, String> {
        self.forward_with_hidden(
            ids, lengths, tables, key, value, logit_rows, embeddings, overrides, None,
        )
    }

    /// Ask the next forward pass for greedy ids instead of logits (see `GREEDY_IDS_REQUEST`).
    pub(crate) fn request_greedy_ids() {
        *GREEDY_IDS.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        GREEDY_IDS_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whole-block operations on the accelerator's KV pools (`kv_blocks`): `None` when the model runs on the host.
    pub(crate) fn device_kv_copy(&self, src: usize, dst: usize) -> Option<Result<(), String>> {
        if self.par.is_some() {
            // the other stages hold the KV of their layers: a block operation on this rank alone would be wrong
            return None;
        }
        let kv = self.backend_kv.as_ref()?;
        let mut stream = self.backend_stream.as_ref()?.lock().ok()?;
        Some(crate::kv_blocks::device_copy(&mut **stream, kv, src, dst))
    }
    pub(crate) fn device_kv_export(&self, block: usize) -> Option<Result<Vec<u8>, String>> {
        if self.par.is_some() {
            // the other stages hold the KV of their layers: a block operation on this rank alone would be wrong
            return None;
        }
        let kv = self.backend_kv.as_ref()?;
        let mut stream = self.backend_stream.as_ref()?.lock().ok()?;
        Some(crate::kv_blocks::device_export(&mut **stream, kv, block))
    }
    pub(crate) fn device_kv_import(&self, block: usize, data: &[u8]) -> Option<Result<(), String>> {
        if self.par.is_some() {
            // the other stages hold the KV of their layers: a block operation on this rank alone would be wrong
            return None;
        }
        let kv = self.backend_kv.as_ref()?;
        let mut stream = self.backend_stream.as_ref()?.lock().ok()?;
        Some(crate::kv_blocks::device_import(&mut **stream, kv, block, data))
    }

    /// Ask the next forward pass to sample the rows of `job` on the device (see `DeviceSampleJob`).
    pub(crate) fn request_device_sample(job: DeviceSampleJob) {
        *SAMPLE_RESULT.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        *SAMPLE_JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(job);
    }

    /// The result of the last device sampling job, if the backend ran it; clears the request in any case.
    pub(crate) fn take_device_sample() -> Option<crate::backend::SampleOutput> {
        *SAMPLE_JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        SAMPLE_RESULT.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    }

    /// The ids of the last greedy forward pass, if it produced them; clears the request in any case.
    pub(crate) fn take_greedy_ids() -> Option<Vec<i32>> {
        GREEDY_IDS_REQUEST.store(false, std::sync::atomic::Ordering::Relaxed);
        GREEDY_IDS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    }

    /// Capture the main decoder state after its output RMS norm, matching h_nextn.
    /// MTP pairs row p with token p+1; logits still use only the requested rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_hidden(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
        capture: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        if let Some(par) = self.par.clone() {
            return self.par_forward(&par, ids, lengths, tables, key, value, logit_rows, embeddings, overrides, capture);
        }
        self.forward_batch(ids, lengths, tables, key, value, logit_rows, embeddings, overrides, capture)
    }

    /// One batch through this decoder's stage (the whole model when it is not part of a parallel group).
    #[allow(clippy::too_many_arguments)]
    fn forward_batch(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
        mut capture: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        if self.backend.is_some() {
            return self.forward_backend(
                ids, lengths, tables, key, value, logit_rows, embeddings, overrides, capture,
            );
        }
        let cfg = &self.cfg;
        let rows = ids.len();
        let max_position = self.max_position();
        if rows == 0
            || lengths.len() != rows
            || tables.len() != rows * self.stride
            || key.len() < self.kvl.key_total
            || value.len() < self.kvl.value_total
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= self.embedding.rows)
            || logit_rows.iter().any(|&row| row >= rows)
        {
            return Err("invalid Rust decoder batch/cache".into());
        }
        for (r, &length) in lengths.iter().enumerate() {
            if length <= 0
                || (length as usize)
                    .checked_add(usize::from(self.mtp.is_some()))
                    .is_none_or(|n| n > max_position)
            {
                return Err("invalid Rust decoder position".into());
            }
            for &block in &tables
                [r * self.stride..r * self.stride + (length as usize).div_ceil(self.block_size)]
            {
                if block < 0 || block as usize >= self.pool_blocks {
                    return Err("invalid Rust decoder KV block".into());
                }
            }
        }
        if overrides
            .iter()
            .any(|(r, x)| *r >= rows || x.len() != cfg.hidden || x.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid image embedding overrides".into());
        }
        self.checked_positions3(rows)?;
        self.checked_deepstack(rows)?;
        let positions: Vec<usize> = lengths
            .iter()
            .map(|&n| n as usize - 1 + usize::from(self.mtp.is_some()))
            .collect();
        self.pool_for(rows).install(|| {
            let mut trace = crate::cpu_profile::Trace::new(self.profile, rows);
            let first_stage = self.par.as_ref().is_none_or(|p| p.is_first_stage());
            let last_stage = self.par.as_ref().is_none_or(|p| p.is_last_stage());
            let mut x = Vec::with_capacity(rows * cfg.hidden);
            if !first_stage {
                // the embedding and the position tables belong to the first stage: continue from its activations
                x = self.par_recv_activations(rows)?;
            }
            for &id in ids.iter().filter(|_| first_stage) {
                x.extend_from_slice(&self.embedding.row(id as usize)?);
            }
            if first_stage {
                self.par_embed_reduce(&mut x)?;
            }
            if first_stage && cfg.embedding_scale != 1.0 {
                for v in &mut x {
                    *v *= cfg.embedding_scale;
                }
            }
            for (r, embedding) in overrides.iter().filter(|_| first_stage) {
                x[*r * cfg.hidden..(*r + 1) * cfg.hidden].copy_from_slice(embedding);
            }
            if let Some(table) = self.position_table.as_ref().filter(|_| first_stage) {
                let rows = table.len() / cfg.hidden;
                for (r, &position) in positions.iter().enumerate() {
                    let index = position + cfg.position_offset;
                    if index >= rows {
                        return Err("position is beyond the learned position embeddings".into());
                    }
                    for (v, p) in x[r * cfg.hidden..(r + 1) * cfg.hidden]
                        .iter_mut()
                        .zip(&table[index * cfg.hidden..(index + 1) * cfg.hidden])
                    {
                        *v += p;
                    }
                }
            }
            if let Some(norm) = self.embd_norm.as_ref().filter(|_| first_stage) {
                self.norm(&mut x, norm);
            }
            if first_stage && cfg.embedding_norm {
                rms(&mut x, &vec![1.0; cfg.hidden], cfg.eps);
            }
            trace.mark(0);
            let ple_inputs = self.ple_inputs(ids, &x)?;
            let mut streams = self.altup_expand(&x)?;
            self.stash_embeddings(&x);
            let stage_layers = self.par.as_ref().map_or(0..self.layers.len(), |p| p.layer_range(self.layers.len()));
            if cfg.v4.is_some() {
                x = self.v4_layers(&x, ids, &positions, tables, lengths)?;
            } else if cfg.inkling.is_some() {
                x = self.inkling_layers(&x, &positions, tables, lengths, 0..cfg.layers)?;
            } else if cfg.minimax3.is_some() {
                x = self.minimax3_layers(&x, &positions, tables, lengths, 0..cfg.layers)?;
            } else if cfg.qwen4.is_some() {
                x = self.qwen4_layers(&x, ids, &positions, tables, lengths, key, value)?;
            } else if cfg.hc.is_some() {
                x = self.hc_layers(&x, &positions, tables, lengths, key, value)?;
            } else {
                self.host_layers(stage_layers, &mut x, &positions, tables, lengths, key, value, &mut trace, ple_inputs.as_deref(), &mut streams)?;
            }
            if !last_stage {
                self.par_send_activations(&x)?;
                return Ok(Vec::new());
            }
            if self.altup.is_some() {
                x = self.altup_merge(&streams)?;
            }
            // Qwen3.8-Flash-Next has no output norm: the gate that collapses the streams already normalised them
            if cfg.qwen4.is_none() {
                self.norm(&mut x, &self.final_norm);
            }
            if let Some(capture) = &mut capture {
                capture.clear();
                capture.extend_from_slice(&x);
            }
            if embeddings {
                return Ok(x);
            }
            let mut selected = Vec::with_capacity(logit_rows.len() * cfg.hidden);
            for &r in logit_rows {
                selected.extend_from_slice(&x[r * cfg.hidden..(r + 1) * cfg.hidden]);
            }
            if selected.is_empty() {
                return Ok(Vec::new());
            }
            let mut logits = self.head.forward(&selected)?;
            logits = self.par_head_gather(logits, logit_rows.len())?;
            trace.mark(9);
            if cfg.logit_scale != 1.0 {
                for l in &mut logits {
                    *l *= cfg.logit_scale;
                }
            }
            if let Some(cap) = cfg.logit_softcap {
                for l in &mut logits {
                    *l = cap * (*l / cap).tanh();
                }
            }
            Ok(logits)
        })
    }

    /// Layers `range` of the host (CPU) path: `x` (rows x hidden) goes in as the residual stream and comes back updated; attention
    /// layers read and write the host KV cache `key` / `value`. The device path calls this for layers it has no kernels for.
    #[allow(clippy::too_many_arguments)]
    fn host_layers(
        &self,
        range: std::ops::Range<usize>,
        x_io: &mut Vec<f32>,
        positions: &[usize],
        tables: &[i64],
        lengths: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        trace: &mut crate::cpu_profile::Trace,
        ple: Option<&[f32]>,
        streams: &mut Vec<f32>,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let rows = positions.len();
        let positions3 = self.row_positions3().filter(|p| p.len() == rows);
        let deepstack = self.row_deepstack();
        let (fmt_k, fmt_v) = self.kv_formats();
        let mut x = std::mem::take(x_io);
        let done = self.pool_for(rows).install(|| {
            let mut qkv_workspace = compute::ProjectionGroupWorkspace::default();
            let mut ffn_workspace = compute::ProjectionGroupWorkspace::default();
            let residual = cfg.residual_scale;
            // RWKV-7: the values of layer 0 of this pass, read by the value residual of every later layer
            let mut rwkv_v_first: Vec<f32> = Vec::new();
            // LongCat-Flash: the shortcut experts of a layer join the stream after the layer that follows
            let mut pending_shortcut: Option<Vec<f32>> = None;
            // Kimi K3: sums of the finished blocks of layers
            let mut block_res: Vec<Vec<f32>> = Vec::new();
            for (i, layer) in self.layers.iter().enumerate().skip(range.start).take(range.len()) {
                let (dk, dv) = (cfg.layer_head_dim(i), cfg.layer_value_dim(i));
                let (heads, kv_heads) = (cfg.layer_heads(i), cfg.layer_kv_heads(i));
                let (kv_k, kv_v) = (kv_heads * dk, kv_heads * dv);
                let rope_dims = cfg.layer_rope_dims(i);
                let mut altup_pred: Option<Vec<f32>> = None;
                if let (Some(alt_layer), Some(alt)) = (layer.gemma.as_ref().and_then(|g| g.altup()), self.altup.as_ref()) {
                    let predictions = self.altup_predict(alt_layer, streams, rows)?;
                    let len = rows * cfg.hidden;
                    x = predictions[alt.active() * len..(alt.active() + 1) * len].to_vec();
                    altup_pred = Some(predictions);
                }
                let mut normed = match (cfg.attn_res, &layer.res) {
                    (Some(block), Some(res)) => {
                        let mixed = if block_res.is_empty() { x.clone() } else { apply_attn_res(&x, &block_res, &res.attn, cfg.eps, cfg.hidden) };
                        if i % block == 0 {
                            // a new block starts: the running sum so far becomes a block of its own
                            block_res.push(std::mem::replace(&mut x, vec![0.0; rows * cfg.hidden]));
                        }
                        mixed
                    }
                    _ => x.clone(),
                };
                if let Some(call) = &layer.zamba {
                    let shared = self.zamba_call(call, i, &x, positions, tables, lengths, key, value)?;
                    normed.iter_mut().zip(&shared).for_each(|(n, s)| *n += s);
                }
                if let Some(norm) = &layer.attn_norm {
                    self.norm(&mut normed, norm);
                }
                trace.mark(1);
                let laurel_out = match layer.gemma.as_ref().and_then(|g| g.laurel()) {
                    Some(laurel) => Some(self.laurel_forward(laurel, &normed)?),
                    None => None,
                };
                dump_layer("normed", i, &normed);
                if let Attn::Rwkv(rw) = &layer.attn {
                    self.rwkv_layer(rw, layer, i, &mut x, &normed, tables, lengths, &mut rwkv_v_first)?;
                    continue;
                }
                if matches!(layer.attn, Attn::None) {
                    let ffn = self.ffn_forward(layer, &normed, &mut ffn_workspace)?;
                    add_scaled(&mut x, &ffn, residual);
                    continue;
                }
                let mut branch = if let Attn::ShortConv(conv) = &layer.attn {
                    self.shortconv_forward(conv, &layer.out, i, &normed, tables, lengths)?
                } else if let Attn::Linear(delta) = &layer.attn {
                    self.delta_forward(delta, &layer.out, i, &normed, tables, lengths)?
                } else if let Attn::Mamba(mixer) = &layer.attn {
                    self.mamba_forward(mixer, &layer.out, i, &normed, tables, lengths)?
                } else if let Attn::Mamba1(mixer) = &layer.attn {
                    self.mamba1_forward(mixer, &layer.out, i, &normed, tables, lengths)?
                } else if let Attn::Kda(mixer) = &layer.attn {
                    self.kda_forward(mixer, &layer.out, i, &normed, tables, lengths)?
                } else {
                    let table = &self.rope[layer.rope_table];
                    let mut gate_values: Option<Vec<f32>> = None;
                    let mut dsa_rows: Option<Vec<Option<Vec<u32>>>> = None;
                    let mut latent_qkv: [Vec<f32>; 3];
                    let mut moe_values: Vec<f32>;
                    let mut shared_kv: Vec<f32> = Vec::new();
                    let mut shared_v: Vec<f32> = Vec::new();
                    let (q, k, v) = match &layer.attn {
                        Attn::Std { q, k, v } => {
                            let (q, k, v) = if let Some(value_moe) = &layer.value_moe {
                                let qk = Linear::forward_group_with_workspace(
                                    &[q, k],
                                    &normed,
                                    &mut qkv_workspace,
                                )?;
                                let [q, k] = qk else {
                                    return Err("invalid QK projection group".into());
                                };
                                moe_values = self.value_moe_forward(value_moe, &normed)?;
                                (q, k, &mut moe_values)
                            } else if layer.kv_shared {
                                let qs = Linear::forward_group_with_workspace(&[q], &normed, &mut qkv_workspace)?;
                                let [q] = qs else {
                                    return Err("invalid query projection".into());
                                };
                                (q, &mut shared_kv, &mut shared_v)
                            } else if layer.k_eq_v {
                                let qk = Linear::forward_group_with_workspace(&[q, k], &normed, &mut qkv_workspace)?;
                                let [q, k] = qk else {
                                    return Err("invalid QK projection group".into());
                                };
                                moe_values = k.clone();
                                (q, k, &mut moe_values)
                            } else {
                                let qkv = Linear::forward_group_with_workspace(
                                    &[q, k, v],
                                    &normed,
                                    &mut qkv_workspace,
                                )?;
                                let [q, k, v] = qkv else {
                                    return Err("invalid QKV projection group".into());
                                };
                                (q, k, v)
                            };
                            if cfg.value_scale != 1.0 && !layer.kv_shared {
                                v.iter_mut().for_each(|x| *x *= cfg.value_scale);
                            }
                            if cfg.value_norm && !layer.kv_shared {
                                rms(v, &vec![1.0; dv], cfg.eps);
                            }
                            trace.mark(2);
                            if let Some(clip) = cfg.qkv_clip {
                                for tensor in [&mut *q, &mut *k, &mut *v] {
                                    tensor.iter_mut().for_each(|x| *x = x.clamp(-clip, clip));
                                }
                            }
                            if let Some(gate) = &layer.attn_gate {
                                gate_values = Some(gate.forward(&normed)?);
                            }
                            if layer.output_gate {
                                let width = heads * dk;
                                let mut plain = Vec::with_capacity(rows * width);
                                let mut gate = Vec::with_capacity(rows * width);
                                // llama.cpp's Qwen3.5 files store each head's query and gate side by side
                                // ([q_h, gate_h] per head); fLLM's own converter writes all queries, then all gates
                                let per_head = cfg.arch.starts_with("qwen35");
                                for r in 0..rows {
                                    let row = &q[r * 2 * width..(r + 1) * 2 * width];
                                    if per_head {
                                        for h in 0..heads {
                                            plain.extend_from_slice(&row[h * 2 * dk..h * 2 * dk + dk]);
                                            gate.extend_from_slice(&row[h * 2 * dk + dk..(h + 1) * 2 * dk]);
                                        }
                                    } else {
                                        plain.extend_from_slice(&row[..width]);
                                        gate.extend_from_slice(&row[width..]);
                                    }
                                }
                                *q = plain;
                                gate_values = Some(gate);
                            }
                            if !cfg.qk_norm_after_rope {
                                qk_norm_cpu(
                                    q,
                                    layer.q_norm.as_deref(),
                                    layer.q_norm_bias.as_deref(),
                                    cfg.eps,
                                );
                                qk_norm_cpu(
                                    k,
                                    layer.k_norm.as_deref(),
                                    layer.k_norm_bias.as_deref(),
                                    cfg.eps,
                                );
                            }
                            if cfg.layer_rotates(i) {
                                if let (Some(p3), Some(state)) = (&positions3, &self.mrope) {
                                    crate::vision::mrope::apply_rope_mrope(q, heads, dk, rope_dims, cfg.rope.neox, p3, &state.axes, table);
                                    if !layer.kv_shared {
                                        crate::vision::mrope::apply_rope_mrope(k, kv_heads, dk, rope_dims, cfg.rope.neox, p3, &state.axes, table);
                                    }
                                } else {
                                    apply_rope(q, heads, dk, rope_dims, cfg.rope.neox, positions, table);
                                    if !layer.kv_shared {
                                        apply_rope(k, kv_heads, dk, rope_dims, cfg.rope.neox, positions, table);
                                    }
                                }
                            }
                            if cfg.qk_norm_after_rope {
                                qk_norm_cpu(
                                    q,
                                    layer.q_norm.as_deref(),
                                    layer.q_norm_bias.as_deref(),
                                    cfg.eps,
                                );
                                qk_norm_cpu(
                                    k,
                                    layer.k_norm.as_deref(),
                                    layer.k_norm_bias.as_deref(),
                                    cfg.eps,
                                );
                            }
                            if let (Some((beta, context)), true) = (
                                cfg.attn_temperature,
                                !cfg.temperature_nope_only || !cfg.layer_rotates(i),
                            ) {
                                for (r, &position) in positions.iter().enumerate() {
                                    let steps = (position + cfg.temperature_offset) / context;
                                    let scale = 1.0 + beta * (1.0 + steps as f32).ln();
                                    for value in
                                        &mut q[r * heads * dk..(r + 1) * heads * dk]
                                    {
                                        *value *= scale;
                                    }
                                }
                            }
                            (q, k, v)
                        }
                        Attn::ShortConv(_)
                        | Attn::Linear(_)
                        | Attn::Mamba(_)
                        | Attn::Mamba1(_)
                        | Attn::Kda(_)
                        | Attn::Rwkv(_)
                        | Attn::None => {
                            unreachable!("handled above")
                        }
                        Attn::Mla(m) => {
                            let sparse = cfg.indexer.is_some() && !self.dsa_dense && (layer.indexer.is_some() || layer.dsa_shared);
                            let (qkv, q_low) = self.mla_qkv_low(m, &normed, positions, table, sparse && layer.indexer.is_some())?;
                            latent_qkv = qkv;
                            if let Some(gate) = &layer.attn_gate {
                                gate_values = Some(gate.forward(&normed)?);
                            }
                            if sparse {
                                dsa_rows = Some(self.dsa_select(i, layer, &normed, q_low.as_deref(), positions, table, tables, lengths)?);
                            }
                            trace.mark(2);
                            let [q, k, v] = &mut latent_qkv;
                            (q, k, v)
                        }
                    };
                    let (key_region, value_region) = (self.key_region(i), self.value_region(i));
                    if !layer.kv_shared {
                        for (r, &position) in positions.iter().enumerate() {
                            let block = tables[r * self.stride + position / self.block_size] as usize;
                            let slot = block * self.block_size + position % self.block_size;
                            self.write_cache(key, key_region, slot, &k[r * kv_k..(r + 1) * kv_k], fmt_k);
                            self.write_cache(value, value_region, slot, &v[r * kv_v..(r + 1) * kv_v], fmt_v);
                        }
                    }
                    trace.mark(3);
                    dump_layer("q", i, q);
                    dump_layer("k", i, k);
                    dump_layer("v", i, v);
                    let tiled = cfg.alibi_slopes.is_none()
                        && layer.chunk.is_none()
                        && layer.sinks.is_none()
                        && cfg.attention_softcap.is_none()
                        && dk == dv
                        && dsa_rows.is_none()
                        && matches!(self.kv_kind, 1 | 2)
                        && self.row_visible_lengths().is_none();
                    let attn = if tiled {
                        let key_cache = crate::cpu_attention::Cache {
                            data: &key[key_region.0..key_region.0 + self.kvl.key_words[crate::kv_geometry::KvPoolLayout::owner(cfg, i)]],
                            slots: self.pool_blocks * self.block_size,
                            kv_heads,
                            dim: dk,
                            q8: self.kv_kind == 2,
                        };
                        let value_cache = crate::cpu_attention::Cache {
                            data: &value[value_region.0..value_region.0 + self.kvl.value_words[crate::kv_geometry::KvPoolLayout::owner(cfg, i)]],
                            ..key_cache
                        };
                        crate::cpu_attention::attention_with(
                            q.as_slice(),
                            key_cache,
                            value_cache,
                            tables,
                            lengths,
                            heads,
                            self.block_size,
                            self.stride,
                            &crate::cpu_attention::Options { scale: cfg.attention_scale, window: layer.window },
                        )
                    } else {
                        None
                    };
                    let mut attn = match attn {
                        Some(out) => out,
                        None => self.attention_selected(i, q, key, value, tables, lengths, dsa_rows.as_deref()),
                    };
                    if let Some(gate) = &gate_values {
                        if layer.gate_per_head {
                            for (r, row) in attn.chunks_mut(heads * dv).enumerate() {
                                for (h, head) in row.chunks_mut(dv).enumerate() {
                                    let scale = cfg.attn_gate.apply(gate[r * heads + h]);
                                    head.iter_mut().for_each(|a| *a *= scale);
                                }
                            }
                        } else {
                            for (a, g) in attn.iter_mut().zip(gate) {
                                *a *= cfg.attn_gate.apply(*g);
                            }
                        }
                    }
                    trace.mark(4);
                    if let Some(weight) = &layer.attn_sub_norm {
                        rms(&mut attn, weight, cfg.eps);
                    }
                    dump_layer("attnout", i, &attn);
                    layer.out.forward(&attn)?
                };
                if let Some(parallel) = &layer.parallel {
                    for (b, m) in branch.iter_mut().zip(self.parallel_branch(parallel, i, &normed, tables, lengths)?) {
                        *b += m;
                    }
                }
                trace.mark(5);
                // row-parallel output projection: sum the partial results of the tensor-parallel group
                self.par_tp_reduce(&mut branch)?;
                dump_layer("branch", i, &branch);
                if let Some(norm) = &layer.post_attn_norm {
                    self.post_norm(&mut branch, norm);
                }
                if matches!(layer.ffn, Ffn::None) {
                    add_scaled(&mut x, &branch, residual);
                    continue;
                }
                let ffn_input = if cfg.parallel_residual {
                    // GPT-NeoX keeps a second norm for the FFN branch; Cohere shares one
                    match &layer.ffn_norm {
                        Some(norm) => {
                            let mut separate = x.clone();
                            self.norm(&mut separate, norm);
                            separate
                        }
                        None => normed,
                    }
                } else {
                    if let Some(laurel) = &laurel_out {
                        for ((x, b), l) in x.iter_mut().zip(&branch).zip(laurel) {
                            *x = (*x + b + l) * std::f32::consts::FRAC_1_SQRT_2;
                        }
                    } else {
                        add_scaled(&mut x, &branch, residual);
                    }
                    dump_layer("attn", i, &x);
                    let mut next = match (cfg.attn_res, &layer.res) {
                        (Some(_), Some(res)) => apply_attn_res(&x, &block_res, &res.ffn, cfg.eps, cfg.hidden),
                        _ => x.clone(),
                    };
                    if let Some(norm) = &layer.ffn_norm {
                        self.norm(&mut next, norm);
                    }
                    next
                };
                trace.mark(1);
                if let Some(moe) = &layer.shortcut {
                    pending_shortcut = Some(self.shortcut_forward(moe, &ffn_input, &mut ffn_workspace)?);
                }
                let mut ffn = self.ffn_forward(layer, &ffn_input, &mut ffn_workspace)?;
                if let Some(moe) = layer.gemma.as_ref().and_then(|g| g.has_moe()) {
                    self.gemma_moe_branch(moe, &x, &mut ffn, &mut ffn_workspace)?;
                }
                trace.mark(8);
                if let Some(norm) = &layer.post_ffn_norm {
                    self.post_norm(&mut ffn, norm);
                }
                if cfg.parallel_residual {
                    for ((x, b), f) in x.iter_mut().zip(&branch).zip(&ffn) {
                        *x += (b + f) * residual;
                    }
                } else {
                    add_scaled(&mut x, &ffn, residual);
                    if layer.takes_shortcut {
                        if let Some(shortcut) = pending_shortcut.take() {
                            add_scaled(&mut x, &shortcut, 1.0);
                        }
                    }
                    dump_layer("ffn", i, &x);
                }
                if let Some(g) = &layer.gemma {
                    match &altup_pred {
                        Some(predictions) => self.altup_tail(i, g, &x, predictions, streams, ple)?,
                        None => self.gemma_tail(i, g, &mut x, ple)?,
                    }
                }
                if let Some(level) = deepstack.as_ref().and_then(|d| d.get(i)) {
                    for (row, values) in level {
                        for (a, b) in x[row * cfg.hidden..(row + 1) * cfg.hidden].iter_mut().zip(values) {
                            *a += b;
                        }
                    }
                }
            }
            if let (Some(_), Some(score)) = (cfg.attn_res, &self.output_res) {
                x = apply_attn_res(&x, &block_res, score, cfg.eps, cfg.hidden);
            }
            Ok::<(), String>(())
        });
        *x_io = x;
        done
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_mtp(
        &self,
        ids: &[i64],
        hidden: &[f32],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
    ) -> Result<Vec<f32>, String> {
        let mtp = self.mtp.as_ref().ok_or("decoder is not an MTP block")?;
        if ids.is_empty()
            || hidden.len() != ids.len() * self.cfg.hidden
            || hidden.iter().any(|v| !v.is_finite())
        {
            return Err("invalid MTP hidden-state batch".into());
        }
        let mut concatenated = Vec::with_capacity(2 * hidden.len());
        for (&token, previous) in ids.iter().zip(hidden.chunks_exact(self.cfg.hidden)) {
            if token < 0 || token as usize >= self.embedding.rows {
                return Err("MTP token ID out of vocabulary".into());
            }
            let mut embedding = self.embedding.row(token as usize)?;
            let mut previous = previous.to_vec();
            rms(&mut embedding, &mtp.embedding_norm, self.cfg.eps);
            rms(&mut previous, &mtp.hidden_norm, self.cfg.eps);
            concatenated.extend(embedding);
            concatenated.extend(previous);
        }
        let projected = mtp.projection.forward(&concatenated)?;
        let overrides: Vec<_> = projected
            .chunks_exact(self.cfg.hidden)
            .enumerate()
            .map(|(i, x)| (i, x.to_vec()))
            .collect();
        self.forward_with_embeddings(
            ids,
            lengths,
            tables,
            key,
            value,
            &[ids.len() - 1],
            false,
            &overrides,
        )
    }
}

/// `FLLM_FUSE_HEAD_COPY` (default on; unset it follows `FLLM_FUSE`): the logits of a decode step come from the final norm output as it
/// is instead of a copy of the requested rows (all rows are requested, in order).
fn head_copy_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let value = |name: &str| std::env::var(name).ok().map(|v| v.trim().to_ascii_lowercase()).filter(|v| !v.is_empty());
        let on = |v: String| !matches!(v.as_str(), "0" | "off" | "false" | "no");
        value("FLLM_FUSE_HEAD_COPY").or_else(|| value("FLLM_FUSE")).is_none_or(on)
    })
}

/// `FLLM_CUDA_GRAPH`: `0`, `off`, `false` or `no` switch the CUDA graphs of decode steps off, `1`, `on` or `true` ask for them (they still
/// need an eligible model and device, see `graph_eligible`), anything else, or no variable, is the default: on whenever eligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphSwitch {
    Auto,
    Off,
    On,
}

fn parse_graph_switch(value: Option<&str>) -> GraphSwitch {
    match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "false" | "no") => GraphSwitch::Off,
        Some("1" | "on" | "true") => GraphSwitch::On,
        _ => GraphSwitch::Auto,
    }
}

fn graph_switch() -> GraphSwitch {
    static SWITCH: std::sync::OnceLock<GraphSwitch> = std::sync::OnceLock::new();
    *SWITCH.get_or_init(|| parse_graph_switch(std::env::var("FLLM_CUDA_GRAPH").ok().as_deref()))
}

/// CUDA graphs for decode steps (on by default where the model and the device allow it, `FLLM_CUDA_GRAPH=0` switches them off). vLLM
/// records its whole forward pass once per batch size (padded sizes, fixed input buffers, a null KV block for the padding rows) and
/// replays it; a step of this engine is several hundred small kernel launches. Measured on an RTX PRO 6000 Blackwell (single user,
/// greedy): +10 to +26 % decode speed, so a graph replay is the default. The text is identical to the ordinary path.
impl RustDecoder {
    #[allow(clippy::too_many_arguments)]
    fn forward_backend(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        key: &mut [u16],
        value: &mut [u16],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
        capture: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        if capture.is_none() && self.par.is_none() && self.row_positions3().is_none() && self.row_deepstack().is_none() && self.row_visible_lengths().is_none() {
            if let Some(out) = self.try_graph_decode(ids, lengths, tables, logit_rows, embeddings, overrides)? {
                return Ok(out);
            }
        }
        self.forward_backend_inner(ids, lengths, tables, key, value, logit_rows, embeddings, overrides, capture, GraphMode::Off)
    }

    /// Dense decoders whose every layer runs on the device with plain attention: nothing in a step may touch the host.
    ///
    /// Not eligible, and why: recurrent mixers (Qwen3.5 / DeltaNet, Mamba 1 and 2, short convolutions) stage a per-step table of state
    /// slots (`stage` kind 4) that is not one of the four fixed arrays of a graph; multi-head latent attention (MLA) has no q/k/v
    /// weights and its assemble kernels were never recorded; mixtures of experts download the routed activations to the host
    /// (`moe_on_device`) or stream experts in; layers or feed-forward blocks that run on the host (`host_layer`, `host_ffn`, host
    /// values) read and write host memory in the middle of a step; parallel residual and QKV clipping take scratch buffers in the
    /// middle of a step; MTP speculation runs a second model per step. KV cache formats are not a criterion: the f16 and q8 writes
    /// and the quantized-format kernels (`kv_quant.cu`) read the staged slot, table and length arrays like the f16 ones and size
    /// their split-KV launch for the longest context while recording.
    fn graph_eligible(&self) -> bool {
        let (Some(weights), Some(stream)) = (self.backend_weights.as_ref(), self.backend_stream.as_ref()) else {
            return false;
        };
        if !stream.lock().map(|s| s.graph_supported()).unwrap_or(false) {
            return false;
        }
        // parallel residual and QKV clipping take buffers from the pool in the middle of a step: not recordable
        !self.cfg.parallel_residual
            && self.cfg.qkv_clip.is_none()
            // the block selection stages host arrays (slots, tables, lengths) that a recorded step cannot take
            && !self.m3_device
            && self.layers.len() == weights.layers.len()
            && self.layers.iter().zip(&weights.layers).all(|(layer, dev)| {
                matches!(layer.attn, Attn::Std { .. })
                    && ((dev.q.is_some() && dev.k.is_some() && dev.v.is_some()) || dev.qkv.is_some())
                    && (dev.moe.is_some()
                        || matches!(layer.ffn, Ffn::None)
                        || (dev.ffn_gate.is_some() && dev.ffn_up.is_some() && dev.ffn_down.is_some())
                        || (dev.gate_up.is_some() && dev.ffn_down.is_some()))
            })
    }

    /// Cache positions and slots of the rows (the validation of `forward_backend_inner`).
    fn step_meta(&self, lengths: &[i64], tables: &[i64]) -> Result<(Vec<usize>, Vec<usize>), String> {
        let max_position = self.max_position();
        let mut positions = Vec::with_capacity(lengths.len());
        let mut slots = Vec::with_capacity(lengths.len());
        for (r, &length) in lengths.iter().enumerate() {
            if length <= 0 || length as usize > max_position {
                return Err("invalid Rust decoder position".into());
            }
            let table = &tables[r * self.stride..(r + 1) * self.stride];
            let position = length as usize - 1;
            let used = (length as usize).div_ceil(self.block_size);
            let current = position / self.block_size;
            if used > self.stride || current >= self.stride {
                return Err("invalid Rust decoder KV block".into());
            }
            for &block in table[..used].iter().chain(&table[current..=current]) {
                if block < 0 || block as usize >= self.pool_blocks {
                    return Err("invalid Rust decoder KV block".into());
                }
            }
            positions.push(position);
            slots.push(table[current] as usize * self.block_size + position % self.block_size);
        }
        Ok((positions, slots))
    }

    /// The embedding rows of `ids` (rows x hidden values). A prompt chunk dequantizes its rows on all cores: this runs before the first kernel is
    /// queued, so the device waits for it (`FLLM_PREFILL_PAR_EMBED=0` keeps the single-threaded loop).
    fn embed_rows(&self, ids: &[i64]) -> Result<Vec<f32>, String> {
        let hidden = self.cfg.hidden;
        static PARALLEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let parallel = *PARALLEL.get_or_init(|| std::env::var("FLLM_PREFILL_PAR_EMBED").map_or(true, |v| v != "0"));
        if !parallel || ids.len() < 64 {
            let mut out = Vec::with_capacity(ids.len() * hidden);
            for &id in ids {
                out.extend_from_slice(&self.embedding.row(id as usize)?);
            }
            return Ok(out);
        }
        let mut out = vec![0.0f32; ids.len() * hidden];
        out.par_chunks_mut(hidden).zip(ids.par_iter()).try_for_each(|(dst, &id)| -> Result<(), String> {
            let row = self.embedding.row(id as usize)?;
            if row.len() != dst.len() {
                return Err("embedding row width differs from the hidden size".into());
            }
            dst.copy_from_slice(&row);
            Ok(())
        })?;
        Ok(out)
    }

    /// Embeddings of the tokens on the host (what `forward_backend_inner` uploads as the first residual stream).
    fn embed_host(&self, ids: &[i64], positions: &[usize]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let mut host_x = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            host_x.extend_from_slice(&self.embedding.row(id as usize)?);
        }
        if cfg.embedding_scale != 1.0 {
            for v in &mut host_x {
                *v *= cfg.embedding_scale;
            }
        }
        if let Some(table) = &self.position_table {
            let table_rows = table.len() / hidden;
            for (r, &pos) in positions.iter().enumerate() {
                let idx = pos + cfg.position_offset;
                if idx >= table_rows {
                    return Err("position is beyond the learned position embeddings".into());
                }
                for (v, p) in host_x[r * hidden..(r + 1) * hidden].iter_mut().zip(&table[idx * hidden..(idx + 1) * hidden]) {
                    *v += p;
                }
            }
        }
        if let Some(norm) = &self.embd_norm {
            self.norm(&mut host_x, norm);
        }
        if cfg.embedding_norm {
            rms(&mut host_x, &vec![1.0; hidden], cfg.eps);
        }
        Ok(host_x)
    }

    /// A decode-only step (one token per row, no images, no hidden-state capture) of a dense model as a CUDA graph. Returns `None`
    /// when the step has to take the ordinary path.
    fn try_graph_decode(
        &self,
        ids: &[i64],
        lengths: &[i64],
        tables: &[i64],
        logit_rows: &[usize],
        embeddings: bool,
        overrides: &[(usize, Vec<f32>)],
    ) -> Result<Option<Vec<f32>>, String> {
        use std::sync::atomic::Ordering::Relaxed;
        let rows = ids.len();
        let stride = self.stride;
        if embeddings
            || !overrides.is_empty()
            || self.mtp.is_some()
            || rows == 0
            || rows > graph_max_rows()
            || lengths.len() != rows
            || tables.len() != rows * stride
            || logit_rows.len() != rows
            || logit_rows.iter().enumerate().any(|(i, &r)| i != r)
            || self.pool_blocks < 2
            || self.graph_failed.load(Relaxed)
            || graph_switch() == GraphSwitch::Off
            || ids.iter().any(|&id| id < 0 || id as usize >= self.embedding.rows)
        {
            return Ok(None);
        }
        if !*self.graph_ok.get_or_init(|| {
            let ok = self.graph_eligible();
            let why = if graph_switch() == GraphSwitch::On { "FLLM_CUDA_GRAPH=1 asked for them, but the model or the device is not supported" } else { "model or device not supported" };
            eprintln!(
                "[fllm-rs] CUDA graphs for decode steps: {} (FLLM_CUDA_GRAPH=0 switches them off)",
                if ok { "on" } else { why }
            );
            ok
        }) {
            return Ok(None);
        }
        // consecutive rows of one sequence are served by the prompt kernels: not a plain decode step
        for r in 1..rows {
            if lengths[r] == lengths[r - 1] + 1 && tables[r * stride..(r + 1) * stride] == tables[(r - 1) * stride..r * stride] {
                return Ok(None);
            }
        }
        let bucket = GRAPH_BUCKETS.iter().copied().find(|&b| b >= rows).unwrap_or(rows);
        // rows added to fill the bucket: token 0, length 1, a table of the reserved last block of the KV pool
        let null_block = (self.pool_blocks - 1) as i64;
        let mut ids_p = ids.to_vec();
        ids_p.resize(bucket, 0);
        let mut lengths_p = lengths.to_vec();
        lengths_p.resize(bucket, 1);
        let mut tables_p = tables.to_vec();
        tables_p.resize(bucket * stride, null_block);
        let (positions, slots) = self.step_meta(&lengths_p, &tables_p)?;
        let host_x = self.embed_host(&ids_p, &positions)?;

        let mut graphs = self.graphs.lock().map_err(|_| "graph table lock poisoned")?;
        if let std::collections::hash_map::Entry::Vacant(slot) = graphs.entry(bucket) {
            match self.capture_graph(bucket, &ids_p, &lengths_p, &tables_p) {
                Ok(graph) => {
                    eprintln!("[fllm-rs] CUDA graph recorded for {bucket} rows");
                    slot.insert(graph);
                }
                Err(e) => {
                    eprintln!("[fllm-rs] CUDA graph capture failed, decode steps run without graphs: {e}");
                    self.graph_failed.store(true, Relaxed);
                    return Ok(None);
                }
            }
        }
        let graph = graphs.get(&bucket).ok_or("graph disappeared")?;
        let cfg = &self.cfg;
        let err = |e: crate::backend::BackendError| e.to_string();
        let mut stream = self
            .backend_stream
            .as_ref()
            .ok_or("accelerator backend is not initialised")?
            .lock()
            .map_err(|_| "accelerator stream lock poisoned")?;
        let mut x_in = graph.x_in.clone();
        stream.upload_activations(&mut x_in, &host_x).map_err(err)?;
        let pos32: Vec<u8> = positions.iter().flat_map(|&p| (p as i32).to_ne_bytes()).collect();
        let slot32: Vec<u8> = slots.iter().flat_map(|&p| (p as i32).to_ne_bytes()).collect();
        let table64: Vec<u8> = tables_p.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let len64: Vec<u8> = lengths_p.iter().flat_map(|v| v.to_ne_bytes()).collect();
        for (array, data) in [&pos32, &slot32, &table64, &len64].into_iter().enumerate() {
            stream.graph_set(graph.id, array, data).map_err(err)?;
        }
        if !stream.graph_launch(graph.id).map_err(err)? {
            // a scratch buffer was reallocated since the capture: record this size again on the next step
            drop(stream);
            graphs.remove(&bucket);
            return Ok(None);
        }
        if cfg.logit_scale == 1.0 && cfg.logit_softcap.is_none() && run_device_sample(&mut **stream, &graph.head_out, cfg.vocab, rows) {
            GREEDY_IDS_REQUEST.store(false, Relaxed);
            return Ok(Some(Vec::new()));
        }
        if GREEDY_IDS_REQUEST.load(Relaxed) && cfg.logit_scale > 0.0 && cfg.logit_softcap.is_none() {
            if let Ok(mut picked) = stream.argmax_rows(&graph.head_out, bucket, cfg.vocab) {
                picked.truncate(rows);
                *GREEDY_IDS.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(picked);
                GREEDY_IDS_REQUEST.store(false, Relaxed);
                return Ok(Some(Vec::new()));
            }
        }
        let mut logits = vec![0.0f32; bucket * cfg.vocab];
        stream.download(cast_f32_to_u8_mut(&mut logits), &graph.head_out).map_err(err)?;
        logits.truncate(rows * cfg.vocab);
        if cfg.logit_scale != 1.0 {
            for l in &mut logits {
                *l *= cfg.logit_scale;
            }
        }
        if let Some(cap) = cfg.logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }
        Ok(Some(logits))
    }

    /// Record the decode step of `bucket` rows: one ordinary pass first (it sizes every lazily grown scratch buffer and creates the
    /// device tables), then the recorded pass.
    fn capture_graph(&self, bucket: usize, ids: &[i64], lengths: &[i64], tables: &[i64]) -> Result<DecodeGraph, String> {
        use std::sync::atomic::Ordering::Relaxed;
        let err = |e: crate::backend::BackendError| e.to_string();
        let hidden = self.cfg.hidden;
        let x_in = self
            .backend_scratch
            .as_ref()
            .ok_or("accelerator backend is not initialised")?
            .lock()
            .map_err(|_| "accelerator scratch lock poisoned")?
            .acquire(bucket * hidden * 4)
            .map_err(err)?;
        let logit_rows: Vec<usize> = (0..bucket).collect();
        // the request for greedy ids belongs to the real step, not to these passes
        let requested = GREEDY_IDS_REQUEST.swap(false, Relaxed);
        // so does the device sampling job
        let sample_job = SAMPLE_JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        let result = (|| {
            self.forward_backend_inner(ids, lengths, tables, &mut [], &mut [], &logit_rows, false, &[], None, GraphMode::Off)?;
            let id = bucket;
            let sizes = [bucket * 4, bucket * 4, bucket * self.stride * 8, bucket * 8];
            {
                let mut stream = self.backend_stream.as_ref().ok_or("no stream")?.lock().map_err(|_| "stream lock poisoned")?;
                stream.graph_begin(id, sizes, self.stride * self.block_size).map_err(|e| e.to_string())?;
            }
            let mut io = GraphIo { x_in: x_in.clone(), head_out: None };
            let run = self.forward_backend_inner(ids, lengths, tables, &mut [], &mut [], &logit_rows, false, &[], None, GraphMode::Capture(&mut io));
            {
                let mut stream = self.backend_stream.as_ref().ok_or("no stream")?.lock().map_err(|_| "stream lock poisoned")?;
                stream.graph_end(id, run.is_err()).map_err(|e| e.to_string())?;
            }
            run?;
            let head_out = io.head_out.ok_or("the recorded step produced no logits")?;
            Ok::<DecodeGraph, String>(DecodeGraph { id, x_in: x_in.clone(), head_out })
        })();
        GREEDY_IDS_REQUEST.store(requested, Relaxed);
        *SAMPLE_JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = sample_job;
        result
    }
}

/// Debug aid (`FLLM_DUMP_LAYERS=dir`): appends the hidden state after the attention branch and
/// after the FFN branch of every host layer, and the projection results and attention output of
/// each layer, to `dir/l<layer>_<tag>.f32` (raw little-endian f32). `scripts/cmpdump.py` compares
/// two such directories (two tiers, thread counts or kernel variants).
fn dump_layer(tag: &str, layer: usize, x: &[f32]) {
    use std::io::Write;
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    let Some(dir) = DIR.get_or_init(|| std::env::var_os("FLLM_DUMP_LAYERS").map(Into::into)) else {
        return;
    };
    let path = dir.join(format!("l{layer}_{tag}.f32"));
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = file.write_all(&bytes);
    }
}

#[cfg(test)]
mod best_positions_tests {
    use super::{best_positions, pooled_selection};

    #[test]
    fn pool_selection_keeps_the_best_pools_and_the_tail() {
        let ic = crate::config::IndexerConfig { heads: 1, dim: 2, topk: 4, neox: false, rope_last: false, norm_eps: 1e-6, kpool: 2, always_tail: true };
        // token p stores [key, gate]; the keys grow with the pool number so that later pools score higher
        let tokens: Vec<[f32; 4]> = (0..7).map(|p| [(p / 2) as f32 + 1.0, 1.0, 0.0, 0.0]).collect();
        let ape = [0.0f32; 4];
        let pick = |n: usize| pooled_selection(&ic, &[1.0, 1.0], &[1.0], &ape, n, |p| &tokens[p][..], false);
        // 7 visible tokens: three complete pools (0..2, 2..4, 4..6), two are kept (the later ones) and token 6 is the tail
        assert_eq!(pick(7), vec![2, 3, 4, 5, 6]);
        // 3 visible tokens: one complete pool and the tail token 2
        assert_eq!(pick(3), vec![0, 1, 2]);
        // 1 visible token: no pool, only the tail
        assert_eq!(pick(1), vec![0]);
    }

    #[test]
    fn keeps_the_highest_scores_in_position_order() {
        let scored = vec![(0.1, 0), (0.9, 1), (0.3, 2), (0.8, 3), (-0.5, 4), (0.7, 5)];
        assert_eq!(best_positions(scored, 3), vec![1, 3, 5]);
    }

    #[test]
    fn equal_scores_keep_the_lower_position() {
        // three keys score 0 and only two of them fit next to the positive one
        let scored = vec![(0.0, 7), (0.5, 3), (0.0, 1), (0.0, 5), (-0.2, 0)];
        assert_eq!(best_positions(scored, 3), vec![1, 3, 5]);
    }

    #[test]
    fn handles_k_larger_than_the_candidates_and_empty_input() {
        assert_eq!(best_positions(vec![(1.0, 4), (2.0, 2)], 8), vec![2, 4]);
        assert!(best_positions(Vec::new(), 4).is_empty());
        assert!(best_positions(vec![(1.0, 0)], 0).is_empty());
    }
}

#[cfg(test)]
mod attn_res_tests {
    use super::apply_attn_res;

    #[test]
    fn zero_scores_average_the_candidates() {
        let blocks = vec![vec![2.0, 4.0, 6.0, 8.0]];
        let prefix = vec![4.0, 0.0, 2.0, 0.0];
        let out = apply_attn_res(&prefix, &blocks, &[0.0, 0.0], 1e-6, 2);
        assert_eq!(out, [3.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn a_dominant_score_selects_one_candidate_per_row() {
        // row 0: the block is (1, 0) and the prefix (0, 1); the score reads the first coordinate, so the block wins
        let blocks = vec![vec![1.0, 0.0, 0.0, 1.0]];
        let prefix = vec![0.0, 1.0, 1.0, 0.0];
        let out = apply_attn_res(&prefix, &blocks, &[200.0, 0.0], 1e-6, 2);
        assert!((out[0] - 1.0).abs() < 1e-4 && out[1].abs() < 1e-4);
        // row 1: the prefix is (1, 0) and wins
        assert!((out[2] - 1.0).abs() < 1e-4 && out[3].abs() < 1e-4);
    }
}

#[cfg(test)]
mod graph_switch_tests {
    use super::{parse_graph_switch, GraphSwitch};

    #[test]
    fn graphs_are_on_by_default_and_the_variable_can_switch_them_off() {
        assert_eq!(parse_graph_switch(None), GraphSwitch::Auto);
        assert_eq!(parse_graph_switch(Some("")), GraphSwitch::Auto);
        assert_eq!(parse_graph_switch(Some("auto")), GraphSwitch::Auto);
        assert_eq!(parse_graph_switch(Some("2")), GraphSwitch::Auto);
        for v in ["0", "off", "OFF", "false", "False", "no", " no "] {
            assert_eq!(parse_graph_switch(Some(v)), GraphSwitch::Off, "{v:?}");
        }
        for v in ["1", "on", "On", "true", "TRUE", " 1 "] {
            assert_eq!(parse_graph_switch(Some(v)), GraphSwitch::On, "{v:?}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod q8_cache_tests {
    use super::write_q8_cache_row;
    use half::f16;

    fn scalar_group(src: &[f32]) -> (u16, Vec<u8>) {
        let amax = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0f32;
        let q = src
            .iter()
            .map(|&x| {
                if d == 0.0 {
                    0
                } else {
                    (x / d).round().clamp(-127.0, 127.0) as i8 as u8
                }
            })
            .collect();
        (f16::from_f32(d).to_bits(), q)
    }

    fn as_le_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn q8_cache_layout_preserves_odd_slot_bytes_and_scale_slots() {
        let slots: usize = 3;
        let kv = 33;
        let group = 33;
        let slot = 1;
        let mut q_words = vec![u16::from_le_bytes([0xa5, 0x5a]); (slots * kv).div_ceil(2)];
        let mut scales = vec![0xdead; slots * (kv / group)];
        let mut expected_q = as_le_bytes(&q_words);
        let mut expected_scales = scales.clone();
        let row = (0..kv)
            .map(|i| match i {
                0 => 127.0,
                1 => -127.0,
                2 => 0.499_999_97,
                3 => -0.5,
                _ => (i as f32 % 17.0) - 8.0,
            })
            .collect::<Vec<_>>();
        let (scale, q) = scalar_group(&row);
        expected_q[slot * kv..(slot + 1) * kv].copy_from_slice(&q);
        expected_scales[slot] = scale;

        write_q8_cache_row(&mut q_words, &mut scales, slot, kv, group, &row);

        assert_eq!(as_le_bytes(&q_words), expected_q);
        assert_eq!(scales, expected_scales);
        assert_eq!(slot * kv % 2, 1, "test must start the slot on an odd byte");
        assert_eq!(q_words.len() * 2, slots * kv + 1, "odd final padding byte");
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn q8_cache_layout_keeps_qkv_head_groups_and_per_head_scales() {
        let slots: usize = 4;
        let kv_heads = 2;
        let dim = 40;
        let kv = kv_heads * dim;
        let group = dim; // Irregular head dimensions use one scale per KV head.
        let slot = 2;
        let mut q_words = vec![u16::from_le_bytes([0x39, 0xc7]); (slots * kv).div_ceil(2)];
        let mut scales = vec![0xbeef; slots * kv_heads];
        let mut expected_q = as_le_bytes(&q_words);
        let mut expected_scales = scales.clone();
        let row = (0..kv)
            .map(|i| {
                if i < dim {
                    (i as f32 - 19.0) / 3.0
                } else {
                    (i as f32 - 60.0) / 5.0
                }
            })
            .collect::<Vec<_>>();
        for head in 0..kv_heads {
            let start = head * group;
            let (scale, q) = scalar_group(&row[start..start + group]);
            expected_q[slot * kv + start..slot * kv + start + group].copy_from_slice(&q);
            expected_scales[slot * kv_heads + head] = scale;
        }

        write_q8_cache_row(&mut q_words, &mut scales, slot, kv, group, &row);

        assert_eq!(as_le_bytes(&q_words), expected_q);
        assert_eq!(scales, expected_scales);
    }
}


#[cfg(test)]
mod grouped_rms_norm_tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::Backend;

    #[test]
    fn test_device_norm_grouped_matches_grouped_rms() {
        let backend = std::sync::Arc::new(CpuBackend::new());
        let mut stream = backend.create_stream().expect("cpu stream");
        let mut pool = crate::backend::BufferPool::new(backend.clone());
        let eps = 1e-5;

        for &(rows, hidden, groups) in &[(1, 256, 4), (3, 1024, 2), (17, 3584, 8)] {
            let mut x_host = vec![0.0f32; rows * hidden];
            for (i, v) in x_host.iter_mut().enumerate() {
                *v = ((i % 100) as f32 - 50.0) * 0.05;
            }
            let mut w_host = vec![0.0f32; hidden];
            for (i, v) in w_host.iter_mut().enumerate() {
                *v = 0.5 + ((i % 50) as f32) * 0.02;
            }

            let mut want = x_host.clone();
            grouped_rms(&mut want, &w_host, eps, groups);

            let mut x_buf = backend.allocate_buffer(rows * hidden * 4).expect("alloc x");
            let mut w_buf = backend.allocate_buffer(hidden * 4).expect("alloc w");
            let mut out_buf = backend.allocate_buffer(rows * hidden * 4).expect("alloc out");

            stream.upload(&mut x_buf, cast_f32_to_u8(&x_host)).expect("upload x");
            stream.upload(&mut w_buf, cast_f32_to_u8(&w_host)).expect("upload w");

            device_norm(
                &mut *stream,
                NormKind::Rms,
                &mut out_buf,
                &x_buf,
                &w_buf,
                None,
                eps,
                groups,
            )
            .expect("device_norm");

            let mut got = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut got), &out_buf).expect("download");

            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-6,
                    "mismatch at row {} col {}: got {g}, want {w}",
                    i / hidden,
                    i % hidden
                );
            }

            // Also test device_rms_inplace
            let mut inplace_buf = backend.allocate_buffer(rows * hidden * 4).expect("alloc inplace");
            stream.upload(&mut inplace_buf, cast_f32_to_u8(&x_host)).expect("upload inplace");
            device_rms_inplace(&mut *stream, &mut pool, &mut inplace_buf, Some(&w_buf), eps, groups)
                .expect("device_rms_inplace");

            let mut got_inplace = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut got_inplace), &inplace_buf).expect("download");
            assert_eq!(got, got_inplace);

            // Also test device_post_norm_inplace
            let mut post_buf = backend.allocate_buffer(rows * hidden * 4).expect("alloc post");
            stream.upload(&mut post_buf, cast_f32_to_u8(&x_host)).expect("upload post");
            device_post_norm_inplace(
                &mut *stream,
                &mut pool,
                NormKind::Rms,
                &mut post_buf,
                Some(&w_buf),
                None,
                eps,
                groups,
            )
            .expect("device_post_norm_inplace");

            let mut got_post = vec![0.0f32; rows * hidden];
            stream.download(cast_f32_to_u8_mut(&mut got_post), &post_buf).expect("download");
            assert_eq!(got, got_post);
        }
    }
}

#[cfg(test)]
mod q8_upload_tests {
    #[test]
    fn q8_0_encoding_round_trips_through_the_gguf_decoder() {
        let data: Vec<f32> = (0..96).map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.013 + if i == 40 { 0.9 } else { 0.0 }).collect();
        let bytes = super::quantize_q8_0(&data);
        assert_eq!(bytes.len(), 3 * 34);
        let back = crate::compute::dequantize(&bytes, 8, data.len()).unwrap();
        for (block, decoded) in data.chunks(32).zip(back.chunks(32)) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (a, b) in block.iter().zip(decoded) {
                assert!((a - b).abs() <= amax / 127.0 * 0.51 + 1e-3 * amax, "{a} vs {b}");
            }
        }
        assert!(super::quantize_q8_0(&[0.0; 32]).iter().skip(2).all(|&b| b == 0));
    }
}
