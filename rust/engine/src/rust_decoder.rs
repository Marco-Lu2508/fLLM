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
    /// Always-active expert, optionally scaled by `sigmoid(shared_gate . x)`.
    shared: Option<(Expert, Option<Vec<f32>>)>,
    /// Latent MoE (Nemotron-H): `(down, up)` projections around the routed experts.
    latent: Option<(Linear, Linear)>,
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
    kv_b: crate::backend::DeviceWeight,
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

struct Layer {
    attn: Attn,
    /// The query projection also yields a sigmoid gate for the attention output.
    output_gate: bool,
    /// Attention is limited to aligned chunks of this many positions (Llama 4).
    chunk: Option<usize>,
    /// Separate projection of the input that gates the attention output (AFMoE).
    attn_gate: Option<Linear>,
    value_moe: Option<ValueMoe>,
    /// xIELU parameters `[alpha_p, alpha_n, beta, eps]` of a plain feed-forward (Apertus).
    xielu: Option<[f32; 4]>,
    /// Learned per-head logit that joins the softmax denominator (GPT-OSS).
    sinks: Option<Vec<f32>>,
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
    /// Use the sliding-window rotary table.
    local_rope: bool,
}

impl Layer {
    #[allow(dead_code)]
    fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut all = Vec::new();
        let has_attention = !matches!(self.attn, Attn::None);
        match &mut self.attn {
            Attn::Std { q, k, v } => all.extend([q, k, v]),
            Attn::Linear(d) => all.extend([&mut d.qkv, &mut d.gate, &mut d.beta, &mut d.alpha]),
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
        if has_attention {
            all.push(&mut self.out);
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

struct Embedding {
    data: fllm_gguf_probe::TensorBytes,
    kind: u32,
    row_bytes: usize,
    rows: usize,
    cols: usize,
}
impl Embedding {
    fn load_named(model: &GgufModel, name: &str) -> Result<Self, String> {
        let map = model.mapping();
        let info = map
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing token embeddings {name}"))?;
        let cols = model.hidden_size();
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
        })
    }
    fn row(&self, id: usize) -> Result<Vec<f32>, String> {
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
    /// Mamba-2 mixer resident on the device.
    mamba: Option<DeviceMamba>,
    /// Short convolution resident on the device.
    shortconv: Option<DeviceShortConv>,
    /// Mamba-1 mixer resident on the device.
    mamba1: Option<DeviceMamba1>,
    /// Per-head attention sink logits (GPT-OSS).
    pub sinks: Option<Arc<Vec<f32>>>,
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
fn device_rms_inplace(
    stream: &mut dyn crate::backend::Stream,
    scratch: &mut crate::backend::BufferPool,
    buf: &mut crate::backend::DeviceBuffer,
    weight: Option<&crate::backend::DeviceBuffer>,
    eps: f32,
) -> Result<(), String> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let mut out = scratch.acquire(buf.size).map_err(|e| e.to_string())?;
    stream.rms_norm(&mut out, buf, weight, eps).map_err(|e| e.to_string())?;
    std::mem::swap(buf, &mut out);
    scratch.recycle(out);
    Ok(())
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
) -> Result<(), String> {
    match kind {
        NormKind::Rms => stream.rms_norm(out, x, weight, eps),
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
) -> Result<(), String> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let mut out = scratch.acquire(buf.size).map_err(|e| e.to_string())?;
    device_norm(stream, kind, &mut out, buf, weight, bias, eps)?;
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
}

/// Rotary cos/sin tables shared between the decoder and device `RopeParams`.
pub type SharedRopeTable = (Arc<Vec<f32>>, Arc<Vec<f32>>);

/// Batch sizes a decode step is padded to when it runs as a CUDA graph.
const GRAPH_BUCKETS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

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

pub struct RustDecoder {
    mtp: Option<MtpProjection>,
    cfg: ModelConfig,
    profile: bool,
    embedding: Embedding,
    layers: Vec<Layer>,
    head: Linear,
    final_norm: Norm,
    rope: Arc<Vec<RopeTables>>,
    groups: usize,
    stride: usize,
    block_size: usize,
    pool_blocks: usize,
    kv_kind: i32,
    pool: rayon::ThreadPool,
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
    backend_weights: Option<Arc<DeviceModelWeights>>,
    backend_kv: Option<crate::backend::DeviceKvCache>,
    backend_scratch: Option<std::sync::Mutex<crate::backend::BufferPool>>,
    /// CUDA graphs of decode steps by padded batch size (see `try_graph_decode`).
    graphs: std::sync::Mutex<std::collections::HashMap<usize, DecodeGraph>>,
    graph_ok: std::sync::OnceLock<bool>,
    graph_failed: std::sync::atomic::AtomicBool,
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

impl RustDecoder {
    pub fn load(model: &GgufModel, device: &str) -> Result<Self, String> {
        Self::load_layers(model, device, None)
    }

    pub(crate) fn load_mtp(model: &GgufModel, device: &str) -> Result<Self, String> {
        if !matches!(device, "auto" | "cpu") {
            return Err("MTP currently requires --device cpu; GPU hidden-state capture is not integrated with the parallel backend work".into());
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
        let (heads, kv_heads, dk, dv) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.value_dim);
        if !dk.is_multiple_of(2) || !cfg.rope.dims.is_multiple_of(2) || cfg.rope.dims > dk {
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
        let mut layers = Vec::with_capacity(layer_range.len());
        for i in layer_range {
            let p = mtp_prefix.map_or_else(|| format!("blk.{i}"), str::to_owned);
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
                }))
            } else if let (Some(m), true) = (
                &cfg.mamba,
                has_tensor(mapping, &format!("{p}.ssm_in.weight")),
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
            } else if let Some(m) = &cfg.mla {
                let q_a = match m.q_lora {
                    Some(rank) => Some((
                        linear(&format!("{p}.attn_q_a"), rank, hidden)?,
                        vector(mapping, &format!("{p}.attn_q_a_norm.weight"), rank)?,
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
                    kv_a_norm: vector(mapping, &format!("{p}.attn_kv_a_norm.weight"), m.kv_lora)?,
                    kv_b,
                }))
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
                Attn::Std {
                    q: linear(
                        &format!("{p}.attn_q"),
                        if output_gate { 2 * qr } else { qr },
                        hidden,
                    )?,
                    k: linear(&format!("{p}.attn_k"), kr, hidden)?,
                    v: if has_tensor(mapping, &format!("{p}.attn_v_exps.weight")) {
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
            } else if has_tensor(mapping, &format!("{p}.ffn_gate_inp.weight")) {
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
                    .map(|e| expert(&p, "exps", e, moe.expert_ff))
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
                Ffn::Moe(Box::new(Moe {
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
                }))
            } else if has_tensor(mapping, &format!("{p}.ffn_gate.weight")) {
                Ffn::Gated {
                    gate: linear(&format!("{p}.ffn_gate"), cfg.ff, hidden)?,
                    up: linear(&format!("{p}.ffn_up"), cfg.ff, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, cfg.ff)?,
                }
            } else if tensor_rows(mapping, &format!("{p}.ffn_up.weight")) == Some(2 * cfg.ff) {
                // fused [gate; up]
                let name = format!("{p}.ffn_up.weight");
                let bias = format!("{p}.ffn_up.bias");
                Ffn::Gated {
                    gate: part(&name, &bias, 0, cfg.ff, hidden)?,
                    up: part(&name, &bias, cfg.ff, cfg.ff, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, cfg.ff)?,
                }
            } else {
                Ffn::Plain {
                    up: linear(&format!("{p}.ffn_up"), cfg.ff, hidden)?,
                    down: linear(&format!("{p}.ffn_down"), hidden, cfg.ff)?,
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
                (Attn::Mamba1(_), _, _) => {
                    let inner = cfg.mamba1.as_ref().map_or(0, |m| m.inner);
                    linear(&format!("{p}.ssm_out"), hidden, inner)?
                }
                _ => linear(&format!("{p}.attn_output"), hidden, heads * dv)?,
            };
            let attn_gate = if has_tensor(mapping, &format!("{p}.attn_gate.weight"))
                && matches!(attn, Attn::Std { .. })
            {
                Some(linear(&format!("{p}.attn_gate"), heads * dv, hidden)?)
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
            layers.push(Layer {
                attn,
                chunk: cfg.layer_chunk(i),
                output_gate,
                attn_gate,
                value_moe,
                xielu,
                sinks,
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
                local_rope: cfg.sliding_window.is_some()
                    && cfg.layer_window(i).is_some()
                    && model.rope_tables().len() > 1,
            });
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
        let embedding = Embedding::load_named(model, &embedding_name)?;
        let mtp = mtp_prefix
            .map(|p| -> Result<MtpProjection, String> {
                Ok(MtpProjection {
                    embedding_norm: vector(mapping, &format!("{p}.enorm.weight"), hidden)?,
                    hidden_norm: vector(mapping, &format!("{p}.hnorm.weight"), hidden)?,
                    projection: linear(&format!("{p}.eh_proj"), hidden, 2 * hidden)?,
                })
            })
            .transpose()?;
        let mut decoder = Self {
            mtp,
            profile: std::env::var_os("FLLM_RUST_PROFILE").is_some(),
            embedding,
            layers,
            head: linear(&head_name, cfg.vocab, hidden)?,
            final_norm: load_norm(mapping, &final_norm_name, hidden)?
                .ok_or("missing output_norm.weight")?,
            rope: model.rope_shared(),
            groups,
            stride: model.table_stride(),
            block_size: model.block_size(),
            pool_blocks: model.pool_blocks(),
            kv_kind: model.kv_kind(),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(model.threads().max(1) as usize)
                .build()
                .map_err(|e| e.to_string())?,
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
            backend_weights: None,
            backend_kv: None,
            backend_scratch: None,
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_ok: std::sync::OnceLock::new(),
            graph_failed: std::sync::atomic::AtomicBool::new(false),
        };
        decoder.set_device(device)?;
        Ok(decoder)
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
    /// 16-bit words of one layer's cache of `kv_heads x dim` values per slot.
    fn layer_words(&self, format: crate::kv_format::KvFormat, dim: usize) -> usize {
        format.layer_words(self.pool_blocks * self.block_size, self.cfg.kv_heads, dim)
    }
    fn write_cache(
        &self,
        cache: &mut [u16],
        layer: usize,
        slot: usize,
        row: &[f32],
        dim: usize,
        format: crate::kv_format::KvFormat,
    ) {
        if format == crate::kv_format::KvFormat::Q8_0 {
            // the original q8_0 writer (also used by the unit tests below)
            let kv = self.cfg.kv_heads * dim;
            let group = if dim.is_multiple_of(32) { 32 } else { dim };
            let slots = self.pool_blocks * self.block_size;
            let words = format.layer_words(slots, self.cfg.kv_heads, dim);
            let q_words = (slots * kv).div_ceil(2);
            let scale_words = slots * (kv / group);
            let (q_cache, cache_scales) = cache[layer * words..(layer + 1) * words].split_at_mut(q_words);
            write_q8_cache_row(q_cache, &mut cache_scales[..scale_words], slot, kv, group, row);
            return;
        }
        let layout = crate::kv_format::KvLayout::new(format, self.pool_blocks * self.block_size, self.cfg.kv_heads, dim);
        let offset = layer * layout.words;
        layout.encode_slot(&mut cache[offset..offset + layout.words], slot, row);
    }
    /// Decode one cache slot (`kv_heads x dim` values) into `out`.
    fn read_slot(
        &self,
        cache: &[u16],
        layer: usize,
        slot: usize,
        dim: usize,
        format: crate::kv_format::KvFormat,
        out: &mut [f32],
    ) {
        let layout = crate::kv_format::KvLayout::new(format, self.pool_blocks * self.block_size, self.cfg.kv_heads, dim);
        let offset = layer * layout.words;
        layout.decode_slot(&cache[offset..offset + layout.words], slot, out);
    }
    /// Pick the compute device for `device` (a `--device` value, see `devices`) and load the model onto it.
    ///
    /// An explicit device that cannot be created is an error. With `auto` every device of the chain (best first, ending in the CPU) is
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
                Err(error) => return Err(error),
            }
        }
        Err(last_error)
    }

    /// Whether an accelerator backend (not the host CPU path) runs the model.
    pub fn uses_accelerator(&self) -> bool {
        self.backend.is_some()
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
        if matches!(wgpu_api, "vulkan" | "metal" | "dx12" | "gpu" | "wgpu") {
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
    fn backend_unsupported(&self, layer_norm_ok: bool) -> Vec<&'static str> {
        let cfg = &self.cfg;
        let mut missing = Vec::new();
        if matches!(cfg.norm, NormKind::Layer) && !layer_norm_ok {
            missing.push("LayerNorm");
        }
        if self.groups != 1 {
            missing.push("grouped RMSNorm");
        }
        missing
    }

    /// Why layer `i` cannot run on the device (`None`: it can). Such a layer is computed by the host reference code
    /// (`host_layers`) between device layers: the residual stream is downloaded, the layer runs on the CPU with the host KV
    /// cache, and the stream goes back up. `gate_ok` / `arch_ok`: the stream has the attention-gate / element-wise kernels.
    fn layer_host_reason(&self, i: usize, layer: &Layer, gate_ok: bool, arch_ok: bool, conv_ok: bool) -> Option<&'static str> {
        let cfg = &self.cfg;
        match &layer.attn {
            Attn::Std { .. } => {
                // routed value experts (MoVA) are computed by the host reference inside an otherwise device layer
                let gated = layer.output_gate || layer.attn_gate.is_some();
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
            // Recurrent mixers keep their state on the host (or in their own device buffers) and handle the residual add themselves.
            Attn::Linear(_) | Attn::Mamba(_) | Attn::Mamba1(_) => {
                if cfg.parallel_residual {
                    Some("parallel residual with a recurrent mixer")
                } else {
                    None
                }
            }
            Attn::None => None,
            Attn::Mla(_) => {
                if arch_ok && cfg.mla.is_some() && cfg.kv_heads == cfg.heads && !cfg.parallel_residual {
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
    fn ffn_host_reason(&self, layer: &Layer, arch_ok: bool) -> Option<&'static str> {
        let cfg = &self.cfg;
        let native = |a: Activation| matches!(a, Activation::Silu | Activation::GeluTanh) || arch_ok;
        match &layer.ffn {
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

    pub fn init_backend(
        &mut self,
        backend: Arc<dyn crate::backend::Backend>,
    ) -> Result<(), String> {
        let (gate_ok, arch_ok) = backend
            .create_stream()
            .map(|s| (s.supports_attention_gate(), s.supports_arch_ops()))
            .unwrap_or((false, false));
        let mut missing = self.backend_unsupported(backend.capabilities().supports_layer_norm);
        // Layers without device kernels run on the host between the device layers (`host_layers`); a model in which no layer can
        // use the device gains nothing from it.
        // the short convolution keeps its history in a device state buffer, like the other recurrent mixers (one kind per model)
        let conv_ok = self.cfg.shortconv.as_ref().is_some_and(|sc| {
            self.delta_shape().is_none()
                && self.mamba_shape().is_none()
                && backend.create_stream().map(|st| st.supports_shortconv(self.cfg.hidden, sc.kernel)).unwrap_or(false)
                && std::env::var_os("FLLM_SHORTCONV_HOST").is_none()
        });
        // FLLM_HOST_LAYERS=all|3,4,5 forces layers onto the host (debugging: the output must match `--device cpu`)
        let forced = std::env::var("FLLM_HOST_LAYERS").ok();
        let layer_reasons: Vec<Option<&'static str>> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let on = forced
                    .as_deref()
                    .is_some_and(|f| f == "all" || f.split(',').any(|n| n.trim().parse::<usize>().is_ok_and(|n| n == i)));
                if on {
                    Some("forced by FLLM_HOST_LAYERS")
                } else {
                    self.layer_host_reason(i, l, gate_ok, arch_ok, conv_ok)
                }
            })
            .collect();
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
        let stream = backend.create_stream().map_err(|e| e.to_string())?;
        let kv = backend
            .allocate_kv_cache(
                self.layers.len(),
                self.pool_blocks,
                self.block_size,
                self.cfg.kv_heads,
                self.cfg.head_dim,
                self.cfg.value_dim,
                self.kv_kind,
            )
            .map_err(|e| e.to_string())?;

        let upload_linear =
            |name: &str, linear: &Linear| -> Result<crate::backend::DeviceWeight, String> {
                if let Some((kind, bytes)) = linear.packed_weights() {
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
        let mut delta_layers = 0usize;
        let mut dev_layers = Vec::with_capacity(self.layers.len());
        let mut host_summary: Vec<&'static str> = Vec::new();
        let mut host_count = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(reason) = layer_reasons[i] {
                host_count += 1;
                if !host_summary.contains(&reason) {
                    host_summary.push(reason);
                }
                dev_layers.push(DeviceLayerWeights {
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
                    host_layer: true,
                    host_ffn: true,
                    host_values: false,
                    attn_gate: None,
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
                    mamba: None,
                    shortconv: None,
                    mamba1: None,
                    sinks: None,
                });
                continue;
            }
            let ffn_host_exact = self.ffn_host_reason(layer, arch_ok).is_some();
            let attn_norm = layer
                .attn_norm
                .as_ref()
                .map(|n| upload_norm(&format!("blk.{i}.attn_norm"), n))
                .transpose()?;
            let (q, k, v) = match &layer.attn {
                Attn::Std { q, k, v } => (
                    Some(upload_linear(&format!("blk.{i}.attn_q"), q)?),
                    Some(upload_linear(&format!("blk.{i}.attn_k"), k)?),
                    if layer.value_moe.is_some() { None } else { Some(upload_linear(&format!("blk.{i}.attn_v"), v)?) },
                ),
                _ => (None, None, None),
            };
            // only attention and the device DeltaNet use the output projection on the device (the other mixers apply it on the host)
            let out = if matches!(layer.attn, Attn::Std { .. } | Attn::Mla(_)) || (matches!(layer.attn, Attn::ShortConv(_)) && conv_ok) || (matches!(layer.attn, Attn::Mamba1(_)) && mamba1_shape.is_some()) || (matches!(layer.attn, Attn::Linear(_)) && delta_shape.is_some())
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
                        norm: upload_vec(&m.norm)?,
                        ordinal: mamba_layers - 1,
                    })
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
                        ordinal: mamba1_layers - 1,
                    })
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
                    kv_b: upload_linear(&format!("blk.{i}.attn_kv_b"), &m.kv_b)?,
                }),
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
            let (ffn_gate, ffn_up, ffn_down) = match &layer.ffn {
                Ffn::Gated { .. } | Ffn::Plain { .. } if host_ffn => (None, None, None),
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
            let moe_dev = match &layer.ffn {
                Ffn::Moe(moe) if !host_ffn && self.device_glu_mode().is_some() => {
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
                    let fast = if self.device_routing_ok(moe) {
                        let all: Option<Vec<&DeviceExpert>> = experts.iter().map(|e| e.as_ref()).collect();
                        match all {
                            Some(list) if !list.is_empty() => {
                                let kinds = [list[0].gate.quant_type, list[0].up.quant_type, list[0].down.quant_type];
                                let same = list.iter().all(|e| {
                                    e.gate.quant_type == kinds[0] && e.up.quant_type == kinds[1] && e.down.quant_type == kinds[2]
                                });
                                let ok_kind = |q: crate::backend::QuantType| matches!(q.to_gguf_type_id(), 2 | 3 | 6 | 7 | 8 | 12..=14 | 20 | 39);
                                let (ff, hid) = (list[0].gate.rows, list[0].gate.cols);
                                let dims = list[0].down.rows == hid
                                    && list[0].down.cols == ff
                                    && hid.is_multiple_of(32)
                                    && ff.is_multiple_of(32)
                                    && kinds.iter().all(|q| !matches!(q.to_gguf_type_id(), 12..=14) || (hid.is_multiple_of(256) && ff.is_multiple_of(256)))
                                    && {
                                        let ok = |sel: fn(&DeviceExpert) -> bool| list.iter().all(|e| sel(e)) || list.iter().all(|e| !sel(e));
                                        ok(|e| e.gate.bias.is_some()) && ok(|e| e.up.bias.is_some()) && ok(|e| e.down.bias.is_some())
                                    };
                                if same && dims && kinds.iter().all(|&q| ok_kind(q)) {
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
                mamba,
                shortconv,
                mamba1,
                sinks: layer.sinks.clone().map(Arc::new),
            });
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
        };

        let mut pool = crate::backend::BufferPool::new(Arc::clone(&backend));
        let max_tokens = 512;
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

        self.backend = Some(backend);
        self.backend_stream = Some(std::sync::Mutex::new(stream));
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
        let (dk, dv) = (cfg.head_dim, cfg.value_dim);
        let mtp = usize::from(self.mtp.is_some());
        let max_position = self.rope[0].cos.len() / self.rope[0].half.max(1);
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

        let mut host_x = self.embed_rows(ids)?;
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
        if !recording {
            stream
                .upload(&mut x_buf, cast_f32_to_u8(&host_x))
                .map_err(err)?;
            debug_sum(&mut **stream, "embd", &x_buf);
        }

        let layer_check = std::env::var_os("FLLM_LAYER_CHECK").is_some() && !recording;
        let delta_shape = self.delta_shape();
        let mamba_shape = self.mamba_shape();
        let mamba1_shape = self.mamba1_shape();
        let mut delta_plan: Option<Vec<DeltaPlanRun>> = None;
        // the norm of the next layer (or the final norm), when the residual add of this layer's feed-forward block already computed it
        let mut carried_norm: Option<crate::backend::DeviceBuffer> = None;
        // the head GEMM reads the final norm output directly when every row's logits are wanted in order
        let head_direct = head_copy_fused() && logit_rows.len() == rows && logit_rows.iter().enumerate().all(|(j, &r)| j == r);
        for (i, layer) in self.layers.iter().enumerate() {
            let dev_layer = &dev_weights.layers[i];
            if dev_layer.host_layer {
                // no kernels for this layer: the host reference code runs it (host KV cache, host recurrent state)
                let mut host_x = vec![0.0f32; rows * hidden];
                stream.download(cast_f32_to_u8_mut(&mut host_x), &x_buf).map_err(err)?;
                let mut trace = crate::cpu_profile::Trace::new(false, rows);
                self.host_layers(i..i + 1, &mut host_x, &positions, tables, lengths, key, value, &mut trace)?;
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
            let mut normed_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
            // parallel residual: the attention branch joins the residual stream only after the feed-forward input is taken from it
            let parallel = cfg.parallel_residual;
            let mut deferred_branch: Option<crate::backend::DeviceBuffer> = None;
            // the feed-forward norm of this layer, when the output projection's residual add already computed it
            let mut ffn_pre: Option<crate::backend::DeviceBuffer> = None;

            if let Some(carried) = carried_norm.take() {
                scratch.recycle(std::mem::replace(&mut normed_buf, carried));
            } else if let Some(norm_buf) = &dev_layer.attn_norm {
                device_norm(&mut **stream, cfg.norm, &mut normed_buf, &x_buf, norm_buf, dev_layer.attn_norm_b.as_ref(), cfg.eps)?;
            } else {
                stream.copy_buffer(&mut normed_buf, &x_buf).map_err(err)?;
            }

            if i == 0 {
                debug_sum(&mut **stream, "attn_norm-0", &normed_buf);
            }
            let std_qkv = match (&dev_layer.q, &dev_layer.k, &dev_layer.v) {
                (Some(q), Some(k), Some(v)) => Some((q, k, Some(v))),
                (Some(q), Some(k), None) if dev_layer.host_values => Some((q, k, None)),
                _ => None,
            };
            if std_qkv.is_some() || dev_layer.mla.is_some() {
                let mut q_buf = scratch.acquire(rows * cfg.heads * dk * 4).map_err(err)?;
                let mut k_buf = scratch.acquire(rows * cfg.kv_heads * dk * 4).map_err(err)?;
                let mut v_buf = scratch.acquire(rows * cfg.kv_heads * dv * 4).map_err(err)?;

                let mut gate_buf = None;
                if let Some((q_w, k_w, v_w)) = std_qkv {
                // 4-bit q/k/v: the three GEMMs leave FP16 and one kernel adds the bias, rotates q and k and writes k and v into the cache
                let mut fused_qkv = false;
                if cfg.rope.dims > 0
                    && cfg.layer_rotates(i)
                    && self.kv_kind == 1
                    && dev_layer.q_norm.is_none()
                    && dev_layer.k_norm.is_none()
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
                        let (cos, sin) = &dev_weights.rope[usize::from(layer.local_rope)];
                        let rope_params = RopeParams {
                            heads: cfg.heads,
                            kv_heads: cfg.kv_heads,
                            dim: dk,
                            rope_dim: cfg.rope.dims,
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
                                i,
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
                // gated attention: the projection holds queries and gates; attention runs on the queries
                if !fused_qkv {
                if layer.output_gate {
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
                stream
                    .quantized_linear(&mut k_buf, &normed_buf, k_w, k_w.bias.as_ref())
                    .map_err(err)?;
                if let Some(v_w) = v_w {
                    stream
                        .quantized_linear(&mut v_buf, &normed_buf, v_w, v_w.bias.as_ref())
                        .map_err(err)?;
                } else if let Some(vm) = &layer.value_moe {
                    let mut host_normed = vec![0.0f32; rows * hidden];
                    stream.download(cast_f32_to_u8_mut(&mut host_normed), &normed_buf).map_err(err)?;
                    let values = self.value_moe_forward(vm, &host_normed)?;
                    stream.upload(&mut v_buf, cast_f32_to_u8(&values)).map_err(err)?;
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
                if !cfg.qk_norm_after_rope {
                    device_rms_inplace(&mut **stream, &mut scratch, &mut q_buf, dev_layer.q_norm.as_ref(), cfg.eps)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut k_buf, dev_layer.k_norm.as_ref(), cfg.eps)?;
                }
                // rotary embedding and the cache write in one kernel (f16 cache, nothing between them)
                let mut kv_written = false;
                if cfg.rope.dims > 0 && cfg.layer_rotates(i) {
                    let (cos, sin) = &dev_weights.rope[usize::from(layer.local_rope)];
                    let rope_params = RopeParams {
                        heads: cfg.heads,
                        kv_heads: cfg.kv_heads,
                        dim: dk,
                        rope_dim: cfg.rope.dims,
                        neox: cfg.rope.neox,
                        cos: Arc::clone(cos),
                        sin: Arc::clone(sin),
                    };
                    if !cfg.qk_norm_after_rope && std::env::var_os("FLLM_DEBUG_SUMS").is_none() {
                        kv_written = stream
                            .rope_kv_write(&mut q_buf, &k_buf, &v_buf, dev_kv, i, &positions, &slots, &rope_params)
                            .map_err(err)?;
                    }
                    if !kv_written {
                        stream
                            .rope(&mut q_buf, &mut k_buf, &positions, &rope_params)
                            .map_err(err)?;
                    }
                }
                if i == 0 {
                    debug_sum(&mut **stream, "Qcur-0 (rope)", &q_buf);
                    debug_sum(&mut **stream, "Kcur-0 (rope)", &k_buf);
                }
                if cfg.qk_norm_after_rope {
                    device_rms_inplace(&mut **stream, &mut scratch, &mut q_buf, dev_layer.q_norm.as_ref(), cfg.eps)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut k_buf, dev_layer.k_norm.as_ref(), cfg.eps)?;
                }
                if !kv_written {
                    stream
                        .kv_cache_write(dev_kv, i, &slots, &k_buf, &v_buf)
                        .map_err(err)?;
                }
                }
                } else if let (Some(dm), Some(mc)) = (&dev_layer.mla, cfg.mla.as_ref()) {
                    // multi-head latent attention, expanded: projections as GEMMs, the data movement in two small kernels, then
                    // rotary embedding, cache write and attention like any other layer
                    let shape = crate::backend::MlaShape { heads: cfg.heads, nope: mc.nope, rope: mc.rope, value: mc.value, kv_lora: mc.kv_lora };
                    let mut q_full = scratch.acquire(rows * cfg.heads * dk * 4).map_err(err)?;
                    if let Some((down, norm)) = &dm.q_a {
                        let mut low = scratch.acquire(rows * down.rows * 4).map_err(err)?;
                        stream.quantized_linear(&mut low, &normed_buf, down, down.bias.as_ref()).map_err(err)?;
                        device_rms_inplace(&mut **stream, &mut scratch, &mut low, Some(norm), cfg.eps)?;
                        stream.quantized_linear(&mut q_full, &low, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
                        scratch.recycle(low);
                    } else {
                        stream.quantized_linear(&mut q_full, &normed_buf, &dm.q, dm.q.bias.as_ref()).map_err(err)?;
                    }
                    let width = mc.kv_lora + mc.rope;
                    let mut kv_a = scratch.acquire(rows * width * 4).map_err(err)?;
                    stream.quantized_linear(&mut kv_a, &normed_buf, &dm.kv_a, dm.kv_a.bias.as_ref()).map_err(err)?;
                    let mut latent = scratch.acquire(rows * mc.kv_lora * 4).map_err(err)?;
                    stream.mla_latent(&mut latent, &kv_a, rows, mc.kv_lora, width).map_err(err)?;
                    device_rms_inplace(&mut **stream, &mut scratch, &mut latent, Some(&dm.kv_a_norm), cfg.eps)?;
                    let mut kv = scratch.acquire(rows * cfg.heads * (mc.nope + mc.value) * 4).map_err(err)?;
                    stream.quantized_linear(&mut kv, &latent, &dm.kv_b, dm.kv_b.bias.as_ref()).map_err(err)?;
                    stream.mla_assemble(&mut q_buf, &mut k_buf, &mut v_buf, &q_full, &kv_a, &kv, rows, &shape).map_err(err)?;
                    for buf in [q_full, kv_a, latent, kv] {
                        scratch.recycle(buf);
                    }
                    let (cos, sin) = &dev_weights.rope[usize::from(layer.local_rope)];
                    let rope_params = RopeParams {
                        heads: cfg.heads,
                        kv_heads: cfg.kv_heads,
                        dim: dk,
                        rope_dim: mc.rope,
                        neox: cfg.rope.neox,
                        cos: Arc::clone(cos),
                        sin: Arc::clone(sin),
                    };
                    stream.rope(&mut q_buf, &mut k_buf, &positions, &rope_params).map_err(err)?;
                    stream.kv_cache_write(dev_kv, i, &slots, &k_buf, &v_buf).map_err(err)?;
                }

                if let Some(g_w) = &dev_layer.attn_gate {
                    let mut gate = scratch.acquire(rows * cfg.heads * dv * 4).map_err(err)?;
                    stream.quantized_linear(&mut gate, &normed_buf, g_w, g_w.bias.as_ref()).map_err(err)?;
                    gate_buf = Some(gate);
                }
                let mut attn_buf = scratch.acquire(rows * cfg.heads * dv * 4).map_err(err)?;
                let attn_params = AttentionParams {
                    num_heads: cfg.heads,
                    num_kv_heads: cfg.kv_heads,
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
                stream
                    .attention_paged(&mut attn_buf, &q_buf, dev_kv, tables, lengths, i, &attn_params)
                    .map_err(err)?;
                if let Some(gate) = gate_buf.take() {
                    match cfg.attn_gate {
                        crate::config::AttnGate::Sigmoid => stream.sigmoid_gate_mul(&mut attn_buf, &gate).map_err(err)?,
                        crate::config::AttnGate::Softplus(beta) => stream.softplus_gate_mul(&mut attn_buf, &gate, beta).map_err(err)?,
                    }
                    scratch.recycle(gate);
                }
                if i == 0 {
                    debug_sum(&mut **stream, "kqv_out-0", &attn_buf);
                }

                let mut out_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                let out_w = dev_layer.out.as_ref().ok_or("attention layer without a device output projection")?;
                // the feed-forward norm (and the activation format of its first GEMM) can join the residual add of the output projection
                let ffn_target = if dev_layer.post_attn_norm.is_none() && !parallel && !matches!(layer.ffn, Ffn::None) && matches!(cfg.norm, NormKind::Rms) && dev_layer.ffn_norm_b.is_none() {
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
                    fused = dev_layer.post_attn_norm.is_none()
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
                    let count = self.layers.iter().filter(|l| matches!(l.attn, Attn::Mamba(_))).count();
                    delta_plan = Some(self.state_plan(count, shape.layer_state_bytes(), tables, lengths)?);
                }
                let plan = delta_plan.as_deref().unwrap_or(&[]);
                self.mamba_device(&mut **stream, &mut scratch, shape, dm, dev_layer, plan, &normed_buf, &mut x_buf, rows)?;
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
                    self.layer_check(&mut **stream, i, &x_buf, host_in, &positions, tables, lengths, key, value)?;
                }
                scratch.recycle(normed_buf);
                continue;
            }
            let dev_ffn = match (&dev_layer.ffn_gate, &dev_layer.ffn_up, &dev_layer.ffn_down) {
                (g, Some(u), Some(d)) => Some((g.as_ref(), u, d)),
                _ => None,
            };
            let mut ffn_normed = scratch.acquire(rows * hidden * 4).map_err(err)?;
            if matches!(layer.attn, Attn::None) || (parallel && dev_layer.ffn_norm.is_none()) {
                // feed-forward-only layers (Nemotron-H) and parallel-residual models with one shared norm (Cohere)
                stream.copy_buffer(&mut ffn_normed, &normed_buf).map_err(err)?;
            } else if let Some(pre) = ffn_pre.take() {
                scratch.recycle(std::mem::replace(&mut ffn_normed, pre));
            } else if let Some(ffn_norm_buf) = &dev_layer.ffn_norm {
                device_norm(&mut **stream, cfg.norm, &mut ffn_normed, &x_buf, ffn_norm_buf, dev_layer.ffn_norm_b.as_ref(), cfg.eps)?;
            } else {
                stream.copy_buffer(&mut ffn_normed, &x_buf).map_err(err)?;
            }
            if let Some(branch) = deferred_branch.take() {
                stream.residual_add(&mut x_buf, &branch, cfg.residual_scale).map_err(err)?;
                scratch.recycle(branch);
            }
            if i == 0 {
                debug_sum(&mut **stream, "attn_post_norm-0 (ffn input)", &ffn_normed);
            }

            if let Some((gate_w, up_w, down_w)) = dev_ffn {
                let ff = up_w.rows;
                let mut gate_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let mut up_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let mut act_buf = scratch.acquire(rows * ff * 4).map_err(err)?;
                let act = act_code(cfg.activation);

                if let Some(gate_w) = gate_w {
                // NVFP4 prompts and 4-bit Marlin: both GEMMs leave FP16 and the gated activation reads it (folded into the activation
                // quantization / conversion of the down projection)
                let mut half_glu = false;
                if Self::half_out_weight(gate_w)
                    && matches!(cfg.activation, Activation::Silu | Activation::GeluTanh)
                    && !crate::backend::env_flag_once!("FLLM_NO_F16_FFN")
                    && gate_w.bias.is_none()
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
                match cfg.activation {
                    Activation::GeluTanh => stream.geglu(&mut act_buf, &gate_buf, &up_buf).map_err(err)?,
                    Activation::Silu => stream.swiglu(&mut act_buf, &gate_buf, &up_buf).map_err(err)?,
                    _ => stream.activation_glu(&mut act_buf, &gate_buf, &up_buf, act).map_err(err)?,
                }
                }
                } else {
                    // plain MLP: down(act(up(x)))
                    stream
                        .quantized_linear(&mut up_buf, &ffn_normed, up_w, up_w.bias.as_ref())
                        .map_err(err)?;
                    stream.activation(&mut act_buf, &up_buf, act).map_err(err)?;
                }

                let mut ffn_out_buf = scratch.acquire(rows * hidden * 4).map_err(err)?;
                // the norm of the next layer (or the final norm) can join the residual add of the down projection
                let next_target = if dev_layer.post_ffn_norm.is_none() && !parallel && matches!(cfg.norm, NormKind::Rms) {
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
                    fused = dev_layer.post_ffn_norm.is_none()
                        && stream
                            .quantized_linear_add(&mut x_buf, &act_buf, down_w, down_w.bias.as_ref(), cfg.residual_scale)
                            .map_err(err)?;
                }
                if !fused {
                    stream
                        .quantized_linear(&mut ffn_out_buf, &act_buf, down_w, down_w.bias.as_ref())
                        .map_err(err)?;
                    device_post_norm_inplace(
                        &mut **stream,
                        &mut scratch,
                        cfg.norm,
                        &mut ffn_out_buf,
                        dev_layer.post_ffn_norm.as_ref(),
                        dev_layer.post_ffn_norm_b.as_ref(),
                        cfg.post_eps,
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
                                (self.device_glu_mode(), Self::gated_weights(&**stream, gate, up, down))
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
                )?;
                stream
                    .residual_add(&mut x_buf, &ffn_out_buf, cfg.residual_scale)
                    .map_err(err)?;
                scratch.recycle(ffn_out_buf);
            }
            if i == 0 || crate::backend::env_flag_once!("FLLM_DEBUG_LAYERS") {
                debug_sum(&mut **stream, &format!("l_out-{i}"), &x_buf);
            }
            if let Some(host_in) = check_in {
                self.layer_check(&mut **stream, i, &x_buf, host_in, &positions, tables, lengths, key, value)?;
            }
            scratch.recycle(ffn_normed);
            scratch.recycle(normed_buf);
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
            )?;
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
        let mut head_out = scratch.acquire(selected * cfg.vocab * 4).map_err(err)?;
        let head = &dev_weights.head;
        stream
            .quantized_linear(&mut head_out, picked.as_ref().unwrap_or(&final_normed), head, head.bias.as_ref())
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
        if GREEDY_IDS_REQUEST.swap(false, std::sync::atomic::Ordering::Relaxed) && cfg.logit_scale > 0.0 && cfg.logit_softcap.is_none() {
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
        let mut logits = vec![0.0f32; selected * cfg.vocab];
        stream
            .download(cast_f32_to_u8_mut(&mut logits), &head_out)
            .map_err(err)?;

        scratch.recycle(head_out);
        if let Some(picked) = picked {
            scratch.recycle(picked);
        }
        scratch.recycle(x_buf);
        scratch.recycle(final_normed);

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
        let cfg = &self.cfg;
        let (dk, dv, heads, kv_heads) = (cfg.head_dim, cfg.value_dim, cfg.heads, cfg.kv_heads);
        let group = heads / kv_heads;
        let scale = cfg.attention_scale.unwrap_or(1.0 / (dk as f32).sqrt());
        let (fmt_k, fmt_v) = self.kv_formats();
        let window = self.layers[layer].window;
        let chunk = self.layers[layer].chunk;
        let mut out = vec![0.0f32; lengths.len() * heads * dv];
        out.par_chunks_mut(heads * dv)
            .enumerate()
            .for_each(|(r, output)| {
                let n = lengths[r] as usize;
                let start = match chunk {
                    Some(c) => (n - 1) / c * c,
                    None => window.map_or(0, |w| n.saturating_sub(w)),
                };
                let count = n - start;
                let mut k_rows = vec![0.0f32; count * kv_heads * dk];
                let mut v_rows = vec![0.0f32; count * kv_heads * dv];
                for (i, p) in (start..n).enumerate() {
                    let block = tables[r * self.stride + p / self.block_size] as usize;
                    let slot = block * self.block_size + p % self.block_size;
                    self.read_slot(
                        key,
                        layer,
                        slot,
                        dk,
                        fmt_k,
                        &mut k_rows[i * kv_heads * dk..(i + 1) * kv_heads * dk],
                    );
                    self.read_slot(
                        value,
                        layer,
                        slot,
                        dv,
                        fmt_v,
                        &mut v_rows[i * kv_heads * dv..(i + 1) * kv_heads * dv],
                    );
                }
                let mut scores = vec![0.0f32; count];
                for h in 0..heads {
                    let kh = h / group;
                    let query = &q[(r * heads + h) * dk..(r * heads + h + 1) * dk];
                    for (i, score) in scores.iter_mut().enumerate() {
                        let k = &k_rows[(i * kv_heads + kh) * dk..(i * kv_heads + kh + 1) * dk];
                        let mut s = compute::dot(query, k) * scale;
                        if let Some(slopes) = &cfg.alibi_slopes {
                            s += slopes[h] * ((start + i) as f32 - (n - 1) as f32);
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
    fn mla_qkv(
        &self,
        m: &Mla,
        x: &[f32],
        positions: &[usize],
        table: &RopeTables,
    ) -> Result<[Vec<f32>; 3], String> {
        let cfg = &self.cfg;
        let mc = cfg
            .mla
            .as_ref()
            .ok_or("MLA layer without MLA configuration")?;
        let rows = positions.len();
        let heads = cfg.heads;
        let (nope, rope, value) = (mc.nope, mc.rope, mc.value);
        let dk = nope + rope;
        let q_full = match &m.q_a {
            Some((down, norm)) => {
                let mut low = down.forward(x)?;
                rms(&mut low, norm, cfg.eps);
                m.q.forward(&low)?
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
        apply_rope(&mut q, heads, dk, rope, cfg.rope.neox, positions, table);
        apply_rope(&mut k, heads, dk, rope, cfg.rope.neox, positions, table);
        Ok([q, k, v])
    }

    /// Shape of the Gated DeltaNet layers, when the model has any.
    /// `FLLM_LAYER_CHECK=1`: runs layer `i` with the host reference on `host_in` (the residual stream the device layer started
    /// from) and prints how far the device result `x` is from it.
    #[allow(clippy::too_many_arguments)]
    fn layer_check(
        &self,
        stream: &mut dyn crate::backend::Stream,
        i: usize,
        x: &crate::backend::DeviceBuffer,
        mut host_x: Vec<f32>,
        positions: &[usize],
        tables: &[i64],
        lengths: &[i64],
        key: &mut [u16],
        value: &mut [u16],
    ) -> Result<(), String> {
        let mut trace = crate::cpu_profile::Trace::new(false, positions.len());
        self.host_layers(i..i + 1, &mut host_x, positions, tables, lengths, key, value, &mut trace)?;
        let mut dev = vec![0.0f32; host_x.len()];
        stream.download(cast_f32_to_u8_mut(&mut dev), x).map_err(|e| e.to_string())?;
        let rms = (host_x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / host_x.len().max(1) as f64).sqrt();
        let worst = dev.iter().zip(&host_x).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max);
        let mean = dev.iter().zip(&host_x).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / host_x.len().max(1) as f64;
        eprintln!(
            "[layer-check] layer {i} rows {}: max|dev-host| {worst:.5} mean {mean:.6} rms {rms:.4} (max/rms {:.4})",
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
        stream.mamba1_prep(shape, &mut split, &mut dt_in, rows).map_err(err)?;
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
        self.layers.iter().any(|l| matches!(l.attn, Attn::Mamba(_))).then(|| crate::backend::MambaShape {
            inner: mc.inner,
            state: mc.state,
            heads: mc.heads,
            groups: mc.groups,
            conv_kernel: mc.conv_kernel,
            norm_groups: mc.norm_groups.max(1),
            step_min: mc.step_min,
            eps: self.cfg.eps,
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
        let out_w = dev_layer.out.as_ref().ok_or("Mamba layer without a device output projection")?;
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
            )?;
            stream.residual_add(x, &branch, residual_scale).map_err(err)?;
            scratch.recycle(branch);
        }
        for buf in [proj, mixed, out] {
            scratch.recycle(buf);
        }
        Ok(())
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
                for part in [dt_part, b_part, c_part] {
                    let inv = (part.iter().map(|v| v * v).sum::<f32>() / part.len() as f32 + eps)
                        .sqrt()
                        .recip();
                    part.iter_mut().for_each(|v| *v *= inv);
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
                    let dt = (if raw > 20.0 { raw } else { raw.exp().ln_1p() }).max(step_min);
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
            // gated RMS norm over `norm_groups` groups of the inner width
            for (v, g) in scan.iter_mut().zip(gate) {
                *v *= g / (1.0 + (-g).exp());
            }
            let size = inner / mc.norm_groups.max(1);
            for (chunk, weight) in scan.chunks_mut(size).zip(m.norm.chunks(size)) {
                let inv = (chunk.iter().map(|v| v * v).sum::<f32>() / size as f32 + cfg.eps)
                    .sqrt()
                    .recip();
                for (v, w) in chunk.iter_mut().zip(weight) {
                    *v *= inv * w;
                }
            }
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

    /// Whether the routing of this layer is what the device kernel implements: softmax over all experts, the best `used`,
    /// optionally renormalised, no bias, groups or shared expert.
    fn device_routing_ok(&self, moe: &Moe) -> bool {
        let Some(cfg) = self.cfg.moe.as_ref() else {
            return false;
        };
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
        if rows == 0 || rows > 4 {
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
        let routed = self.moe_route(moe, host_in)?;
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
        stream.zero_buffer(&mut acc).map_err(err)?;
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
        if let Some((src, gate_vec)) = shared {
            let result = Self::gated_run(stream, scratch, mode, &src, x, rows)?;
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
        Ok(Some(acc))
    }

    fn expert_forward(
        &self,
        e: &Expert,
        input: &[f32],
        workspace: &mut compute::ProjectionGroupWorkspace,
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
        self.gate_activation(g, u, true);
        e.down.forward(g)
    }

    /// `g = act(g) * u` of a gated expert FFN. Elementwise, so the serial form (used between
    /// the fused expert passes, where the vectors are too short for a fork/join) gives the
    /// same bits as the parallel one.
    fn gate_activation(&self, g: &mut [f32], u: &[f32], parallel: bool) {
        let activation = self.cfg.activation;
        if let Some((alpha, limit)) = self.cfg.moe.as_ref().and_then(|m| m.swiglu_oai) {
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

    /// Which tokens go to which expert, with their weights: `routed[expert] = [(token, weight)]`.
    fn moe_route(&self, moe: &Moe, input: &[f32]) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let cfg = self
            .cfg
            .moe
            .as_ref()
            .ok_or("MoE layer without MoE configuration")?;
        let hidden = self.cfg.hidden;
        let tokens = input.len() / hidden;
        let logits = moe.router.forward(input)?;
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
            let chosen = &order[..cfg.used];
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
        let routed = self.moe_route(moe, input)?;
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
            out = up.forward(&out)?;
        }
        if let Some((shared, gate)) = &moe.shared {
            let result = self.expert_forward(shared, input, workspace)?;
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

    fn ffn_forward(
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
                if self.cfg.activation == Activation::Silu {
                    crate::cpu_elementwise::swiglu(g, u);
                } else {
                    let activation = self.cfg.activation;
                    g.par_iter_mut()
                        .zip(u.par_iter())
                        .for_each(|(g, u)| *g = activate(activation, *g) * u);
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
        mut capture: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        if self.backend.is_some() {
            return self.forward_backend(
                ids, lengths, tables, key, value, logit_rows, embeddings, overrides, capture,
            );
        }
        let cfg = &self.cfg;
        let rows = ids.len();
        let (dk, dv) = (cfg.head_dim, cfg.value_dim);
        let (fmt_k, fmt_v) = self.kv_formats();
        let (words_k, words_v) = (self.layer_words(fmt_k, dk), self.layer_words(fmt_v, dv));
        let max_position = self.rope[0].cos.len() / self.rope[0].half.max(1);
        if rows == 0
            || lengths.len() != rows
            || tables.len() != rows * self.stride
            || key.len() < words_k * self.layers.len()
            || value.len() < words_v * self.layers.len()
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
        let positions: Vec<usize> = lengths
            .iter()
            .map(|&n| n as usize - 1 + usize::from(self.mtp.is_some()))
            .collect();
        self.pool.install(|| {
            let mut trace = crate::cpu_profile::Trace::new(self.profile, rows);
            let mut x = Vec::with_capacity(rows * cfg.hidden);
            for &id in ids {
                x.extend_from_slice(&self.embedding.row(id as usize)?);
            }
            if cfg.embedding_scale != 1.0 {
                for v in &mut x {
                    *v *= cfg.embedding_scale;
                }
            }
            for (r, embedding) in overrides {
                x[*r * cfg.hidden..(*r + 1) * cfg.hidden].copy_from_slice(embedding);
            }
            if let Some(table) = &self.position_table {
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
            if let Some(norm) = &self.embd_norm {
                self.norm(&mut x, norm);
            }
            if cfg.embedding_norm {
                rms(&mut x, &vec![1.0; cfg.hidden], cfg.eps);
            }
            trace.mark(0);
            self.host_layers(0..self.layers.len(), &mut x, &positions, tables, lengths, key, value, &mut trace)?;
            self.norm(&mut x, &self.final_norm);
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
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let rows = positions.len();
        let (dk, dv) = (cfg.head_dim, cfg.value_dim);
        let kv_k = cfg.kv_heads * dk;
        let kv_v = cfg.kv_heads * dv;
        let (fmt_k, fmt_v) = self.kv_formats();
        let (words_k, words_v) = (self.layer_words(fmt_k, dk), self.layer_words(fmt_v, dv));
        let mut x = std::mem::take(x_io);
        let done = self.pool.install(|| {
            let mut qkv_workspace = compute::ProjectionGroupWorkspace::default();
            let mut ffn_workspace = compute::ProjectionGroupWorkspace::default();
            let residual = cfg.residual_scale;
            for (i, layer) in self.layers.iter().enumerate().skip(range.start).take(range.len()) {
                let mut normed = x.clone();
                if let Some(norm) = &layer.attn_norm {
                    self.norm(&mut normed, norm);
                }
                trace.mark(1);
                dump_layer("normed", i, &normed);
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
                } else {
                    let table = &self.rope[usize::from(layer.local_rope)];
                    let mut gate_values: Option<Vec<f32>> = None;
                    let mut latent_qkv: [Vec<f32>; 3];
                    let mut moe_values: Vec<f32>;
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
                                let width = cfg.heads * dk;
                                let mut plain = Vec::with_capacity(rows * width);
                                let mut gate = Vec::with_capacity(rows * width);
                                // llama.cpp's Qwen3.5 files store each head's query and gate side by side
                                // ([q_h, gate_h] per head); fLLM's own converter writes all queries, then all gates
                                let per_head = cfg.arch.starts_with("qwen35");
                                for r in 0..rows {
                                    let row = &q[r * 2 * width..(r + 1) * 2 * width];
                                    if per_head {
                                        for h in 0..cfg.heads {
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
                                apply_rope(
                                    q,
                                    cfg.heads,
                                    dk,
                                    cfg.rope.dims,
                                    cfg.rope.neox,
                                    positions,
                                    table,
                                );
                                apply_rope(
                                    k,
                                    cfg.kv_heads,
                                    dk,
                                    cfg.rope.dims,
                                    cfg.rope.neox,
                                    positions,
                                    table,
                                );
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
                                        &mut q[r * cfg.heads * dk..(r + 1) * cfg.heads * dk]
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
                        | Attn::None => {
                            unreachable!("handled above")
                        }
                        Attn::Mla(m) => {
                            latent_qkv = self.mla_qkv(m, &normed, positions, table)?;
                            trace.mark(2);
                            let [q, k, v] = &mut latent_qkv;
                            (q, k, v)
                        }
                    };
                    for (r, &position) in positions.iter().enumerate() {
                        let block = tables[r * self.stride + position / self.block_size] as usize;
                        let slot = block * self.block_size + position % self.block_size;
                        self.write_cache(key, i, slot, &k[r * kv_k..(r + 1) * kv_k], dk, fmt_k);
                        self.write_cache(value, i, slot, &v[r * kv_v..(r + 1) * kv_v], dv, fmt_v);
                    }
                    trace.mark(3);
                    dump_layer("q", i, q);
                    dump_layer("k", i, k);
                    dump_layer("v", i, v);
                    let tiled = layer.window.is_none()
                        && cfg.alibi_slopes.is_none()
                        && layer.chunk.is_none()
                        && layer.sinks.is_none()
                        && cfg.attention_softcap.is_none()
                        && cfg.attention_scale.is_none()
                        && dk == dv
                        && matches!(self.kv_kind, 1 | 2);
                    let attn = if tiled {
                        let key_cache = crate::cpu_attention::Cache {
                            data: &key[i * words_k..(i + 1) * words_k],
                            slots: self.pool_blocks * self.block_size,
                            kv_heads: cfg.kv_heads,
                            dim: dk,
                            q8: self.kv_kind == 2,
                        };
                        let value_cache = crate::cpu_attention::Cache {
                            data: &value[i * words_v..(i + 1) * words_v],
                            ..key_cache
                        };
                        crate::cpu_attention::attention(
                            q.as_slice(),
                            key_cache,
                            value_cache,
                            tables,
                            lengths,
                            cfg.heads,
                            self.block_size,
                            self.stride,
                        )
                    } else {
                        None
                    };
                    let mut attn = match attn {
                        Some(out) => out,
                        None => self.attention_reference(i, q, key, value, tables, lengths),
                    };
                    if let Some(gate) = &gate_values {
                        for (a, g) in attn.iter_mut().zip(gate) {
                            *a *= cfg.attn_gate.apply(*g);
                        }
                    }
                    trace.mark(4);
                    dump_layer("attnout", i, &attn);
                    layer.out.forward(&attn)?
                };
                trace.mark(5);
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
                    add_scaled(&mut x, &branch, residual);
                    dump_layer("attn", i, &x);
                    let mut next = x.clone();
                    if let Some(norm) = &layer.ffn_norm {
                        self.norm(&mut next, norm);
                    }
                    next
                };
                trace.mark(1);
                let mut ffn = self.ffn_forward(layer, &ffn_input, &mut ffn_workspace)?;
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
                    dump_layer("ffn", i, &x);
                }
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
        if capture.is_none() {
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
            && self.layers.len() == weights.layers.len()
            && self.layers.iter().zip(&weights.layers).all(|(layer, dev)| {
                matches!(layer.attn, Attn::Std { .. })
                    && dev.q.is_some()
                    && dev.k.is_some()
                    && dev.v.is_some()
                    && dev.moe.is_none()
                    && (matches!(layer.ffn, Ffn::None)
                        || (dev.ffn_gate.is_some() && dev.ffn_up.is_some() && dev.ffn_down.is_some()))
            })
    }

    /// Cache positions and slots of the rows (the validation of `forward_backend_inner`).
    fn step_meta(&self, lengths: &[i64], tables: &[i64]) -> Result<(Vec<usize>, Vec<usize>), String> {
        let max_position = self.rope[0].cos.len() / self.rope[0].half.max(1);
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
            || rows > GRAPH_BUCKETS[GRAPH_BUCKETS.len() - 1]
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
        stream.upload(&mut x_in, cast_f32_to_u8(&host_x)).map_err(err)?;
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
