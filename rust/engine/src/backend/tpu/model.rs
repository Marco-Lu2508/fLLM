//! Google TPU through PJRT / StableHLO: whole transformer blocks as single programs. Owner: package 12.
//!
//! The `Stream` operations of the backend run one StableHLO program each, which on a TPU means one launch per operation (about
//! twenty-five per layer and token). This file emits the *whole decoder block* of a Llama-style model as one program, so that a layer
//! costs one launch:
//!
//! `x -> RMSNorm -> Q/K/V projections (+ bias, + QK-norm) -> RoPE -> KV-cache write -> GQA attention over the cache -> output
//! projection -> residual -> RMSNorm -> gated FFN (SwiGLU or GeGLU) -> residual`
//!
//! - [`emit_block`] writes the block into a [`Func`] from values the caller supplies; [`block_module`] wraps it into a module with
//!   one parameter per input (weights, tables, caches, positions), which is what a PJRT client compiles. All layers of a model share
//!   the program: the weights are arguments, not constants.
//! - The attention reads a KV cache that is an input *and* an output of the program (donated, so the update is in place): either a
//!   paged pool with block tables (the layout of `DeviceKvCache`, one array `[layers * slots, kv]` of f16) or, with
//!   [`BlockSpec::dense`], one sequence in a plain array `[context, kv]`.
//! - [`run_block`] runs the block on the backend's device with the same buffers the `Stream` operations use; [`reference_block`] is
//!   a plain f32 implementation of the same block for comparison on real hardware (see the ignored tests).
//!
//! The decoder (`rust_decoder.rs`) does not call blocks yet: the `Stream` trait has no per-layer hook. `run_block` is the entry point
//! such a hook would call (see the report of this package).
//!
//! Level of verification: L0 (the module text parses and verifies as StableHLO; no TPU has compiled or run it).

use super::backend::mem::{driver, invalid, Shared, View};
use super::backend::plan::{Meta, Plan};
use super::stablehlo::ops::{AttentionInputs, AttentionSpec, Numerics, Weight};
use super::stablehlo::{self, DType, Func, TensorType, Value};
use crate::backend::{BackendError, DeviceBuffer};
use std::sync::Arc;

/// Shape and options of one decoder block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSpec {
    /// Tokens processed together (one for decoding, the prompt length for prefill).
    pub rows: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Width of the feed-forward layer.
    pub ff: usize,
    pub eps: f32,
    /// Width of the rotated part of a head and the pairing of its values.
    pub rope_dim: usize,
    pub neox: bool,
    /// Rows of the cos / sin tables (positions the model supports).
    pub rope_rows: usize,
    /// Q/K/V projections have a bias (Qwen2).
    pub qkv_bias: bool,
    /// RMSNorm of every query and key head before RoPE (Qwen3).
    pub qk_norm: bool,
    /// GeGLU instead of SwiGLU.
    pub gelu: bool,
    /// Factor of the branch outputs before they are added to the residual stream.
    pub residual_scale: f32,
    /// Attention scale (usually `1 / sqrt(head_dim)`) and optional sliding window.
    pub scale: f32,
    pub window: Option<usize>,
    /// Paged cache geometry: positions per block, table entries per row, and the number of positions the program looks at
    /// (`blocks * block_size`, at least the longest sequence). For the dense cache `positions` is the context length.
    pub block_size: usize,
    pub stride: usize,
    pub positions: usize,
    /// Rows of the cache arrays: `layers * slots` for the paged pool, the context length for the dense cache.
    pub pool_rows: usize,
    /// One sequence in a plain `[context, kv]` array; the block table is the identity and the lengths follow from the positions.
    pub dense: bool,
    pub numerics: Numerics,
    /// Element type of the projection weights: `Bf16` or `F32`.
    pub weight_type: DType,
}

impl BlockSpec {
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Check that the options fit together.
    pub fn validate(&self) -> Result<(), String> {
        let fail = |m: String| Err(format!("block spec: {m}"));
        if self.rows == 0
            || self.hidden == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.ff == 0
        {
            return fail("a dimension is zero".into());
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return fail(format!(
                "{} heads are not a multiple of {} kv heads",
                self.heads, self.kv_heads
            ));
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim {
            return fail(format!(
                "rope width {} for head width {}",
                self.rope_dim, self.head_dim
            ));
        }
        if self.block_size == 0
            || !self.positions.is_multiple_of(self.block_size)
            || self.positions == 0
        {
            return fail(format!(
                "{} positions are not blocks of {}",
                self.positions, self.block_size
            ));
        }
        if self.positions > self.stride * self.block_size {
            return fail(format!(
                "{} positions exceed {} table entries of {}",
                self.positions, self.stride, self.block_size
            ));
        }
        if self.dense
            && (self.pool_rows != self.stride * self.block_size || self.positions > self.pool_rows)
        {
            return fail("a dense cache holds exactly stride * block_size rows".into());
        }
        if !matches!(self.weight_type, DType::Bf16 | DType::F32) {
            return fail(format!("weight type {:?}", self.weight_type));
        }
        Ok(())
    }
}

/// The values a block needs, in the function that contains it.
pub struct BlockValues {
    /// `[rows, hidden]` f32: the residual stream.
    pub x: Value,
    pub attn_norm: Value,
    pub ffn_norm: Value,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
    pub bq: Option<Value>,
    pub bk: Option<Value>,
    pub bv: Option<Value>,
    pub q_norm: Option<Value>,
    pub k_norm: Option<Value>,
    /// `[rope_rows, rope_dim / 2]` f32
    pub cos: Value,
    pub sin: Value,
    /// `[pool_rows, kv_width]` f16
    pub key_pool: Value,
    pub value_pool: Value,
    /// `[rows]` i32
    pub positions: Value,
    /// `[rows]` i32: pool row to write each token's key and value to (paged; unused when dense).
    pub slot_ids: Value,
    /// Scalar i32: first pool row of the layer (paged; unused when dense).
    pub layer_row0: Value,
    /// `[rows, stride]` i32 (paged; unused when dense).
    pub tables: Value,
    /// `[rows]` i32 (paged; unused when dense).
    pub lengths: Value,
}

/// What the block produces.
pub struct BlockOutputs {
    pub x: Value,
    pub key_pool: Value,
    pub value_pool: Value,
}

/// Write the decoder block into `f`.
pub fn emit_block(f: &mut Func, spec: &BlockSpec, v: &BlockValues) -> BlockOutputs {
    let (r, h, kvh, dk) = (
        spec.rows as i64,
        spec.heads as i64,
        spec.kv_heads as i64,
        spec.head_dim as i64,
    );
    let hidden = spec.hidden as i64;
    let numerics = spec.numerics;

    // attention branch
    let normed = f.rms_norm(&v.x, &v.attn_norm, spec.eps);
    let mut q = f.linear(&normed, &v.wq, numerics);
    let mut k = f.linear(&normed, &v.wk, numerics);
    let mut val = f.linear(&normed, &v.wv, numerics);
    if let (Some(bq), Some(bk), Some(bv)) = (&v.bq, &v.bk, &v.bv) {
        q = f.add_bias(&q, bq);
        k = f.add_bias(&k, bk);
        val = f.add_bias(&val, bv);
    }
    let mut q3 = f.reshape(&q, &[r, h, dk]);
    let mut k3 = f.reshape(&k, &[r, kvh, dk]);
    if let (Some(qn), Some(kn)) = (&v.q_norm, &v.k_norm) {
        let q2 = f.reshape(&q3, &[r * h, dk]);
        let q2 = f.rms_norm(&q2, qn, spec.eps);
        q3 = f.reshape(&q2, &[r, h, dk]);
        let k2 = f.reshape(&k3, &[r * kvh, dk]);
        let k2 = f.rms_norm(&k2, kn, spec.eps);
        k3 = f.reshape(&k2, &[r, kvh, dk]);
    }
    let cos_rows = f.gather_rows(&v.cos, &v.positions);
    let sin_rows = f.gather_rows(&v.sin, &v.positions);
    q3 = f.rope_rotate(&q3, &cos_rows, &sin_rows, spec.rope_dim, spec.neox);
    k3 = f.rope_rotate(&k3, &cos_rows, &sin_rows, spec.rope_dim, spec.neox);
    let q_flat = f.reshape(&q3, &[r, h * dk]);
    let k_rows = f.reshape(&k3, &[r, kvh * dk]);
    let v_rows = f.reshape(&val, &[r, kvh * dk]);

    // cache update, then attention over the updated cache
    let write_ids = if spec.dense {
        v.positions.clone()
    } else {
        v.slot_ids.clone()
    };
    let key_pool = f.cache_write(&v.key_pool, &write_ids, &k_rows);
    let value_pool = f.cache_write(&v.value_pool, &write_ids, &v_rows);
    let (tables, lengths, layer_row0) = if spec.dense {
        let blocks = (spec.pool_rows / spec.block_size) as i64;
        let tables = f.iota(&[r, blocks], 1, DType::I32);
        let lengths = f.add_const(&v.positions, 1.0);
        (tables, lengths, f.constant(0.0, &[], DType::I32))
    } else {
        (v.tables.clone(), v.lengths.clone(), v.layer_row0.clone())
    };
    let attention = AttentionSpec {
        rows: spec.rows,
        heads: spec.heads,
        kv_heads: spec.kv_heads,
        head_dim: spec.head_dim,
        value_dim: spec.head_dim,
        stride: if spec.dense {
            spec.pool_rows / spec.block_size
        } else {
            spec.stride
        },
        block_size: spec.block_size,
        positions: spec.positions,
        scale: spec.scale,
        softcap: None,
        window: spec.window,
        chunk: None,
        numerics,
    };
    let attended = f.paged_attention(
        &attention,
        &AttentionInputs {
            q: &q_flat,
            key_pool: &key_pool,
            value_pool: &value_pool,
            layer_row0: &layer_row0,
            tables: &tables,
            lengths: &lengths,
            sinks: None,
            alibi: None,
        },
    );
    let projected = f.linear(&attended, &v.wo, numerics);
    let branch = scaled(f, &projected, spec.residual_scale);
    let x1 = f.add(&v.x, &branch);

    // feed-forward branch
    let normed = f.rms_norm(&x1, &v.ffn_norm, spec.eps);
    let gate = f.linear(&normed, &v.gate, numerics);
    let up = f.linear(&normed, &v.up, numerics);
    let activated = if spec.gelu {
        f.geglu(&gate, &up)
    } else {
        f.swiglu(&gate, &up)
    };
    let down = f.linear(&activated, &v.down, numerics);
    let branch = scaled(f, &down, spec.residual_scale);
    let x2 = f.add(&x1, &branch);
    debug_assert_eq!(x2.dims(), [r, hidden]);
    BlockOutputs {
        x: x2,
        key_pool,
        value_pool,
    }
}

fn scaled(f: &mut Func, value: &Value, factor: f32) -> Value {
    if factor == 1.0 {
        value.clone()
    } else {
        f.mul_const(value, f64::from(factor))
    }
}

/// One parameter of a block module.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDesc {
    pub name: &'static str,
    pub ty: TensorType,
}

/// A block as a standalone module with one parameter per input.
pub struct BlockProgram {
    pub text: String,
    /// The parameters in order.
    pub params: Vec<ParamDesc>,
    /// Results: the new residual stream, the new key pool, the new value pool.
    pub results: Vec<TensorType>,
}

/// The shapes of the weights of a block: `(name, rows, cols)` of the projections in the order the parameters appear.
pub fn weight_shapes(spec: &BlockSpec) -> Vec<(&'static str, usize, usize)> {
    vec![
        ("wq", spec.q_width(), spec.hidden),
        ("wk", spec.kv_width(), spec.hidden),
        ("wv", spec.kv_width(), spec.hidden),
        ("wo", spec.hidden, spec.q_width()),
        ("w_gate", spec.ff, spec.hidden),
        ("w_up", spec.ff, spec.hidden),
        ("w_down", spec.hidden, spec.ff),
    ]
}

/// Emit `spec` as a module (`@main`). Parameter order: `x`, `attn_norm`, `ffn_norm`, the seven weights of [`weight_shapes`], then
/// `bq`, `bk`, `bv` (with `qkv_bias`), `q_norm`, `k_norm` (with `qk_norm`), `cos`, `sin`, `key_pool`, `value_pool` (donated to results
/// 1 and 2), `positions`, and, unless dense, `slot_ids`, `layer_row0`, `tables`, `lengths`.
pub fn block_module(spec: &BlockSpec) -> Result<BlockProgram, String> {
    spec.validate()?;
    let mut f = Func::new("main");
    let mut params: Vec<ParamDesc> = Vec::new();
    let mut add = |f: &mut Func, name: &'static str, dims: &[i64], dtype: DType| -> Value {
        params.push(ParamDesc {
            name,
            ty: TensorType::new(dims, dtype),
        });
        f.param(dims, dtype)
    };
    let (r, hidden, half) = (
        spec.rows as i64,
        spec.hidden as i64,
        (spec.rope_dim / 2) as i64,
    );
    let (h, kvh, dk) = (
        spec.heads as i64,
        spec.kv_heads as i64,
        spec.head_dim as i64,
    );
    let wt = spec.weight_type;
    let x = add(&mut f, "x", &[r, hidden], DType::F32);
    let attn_norm = add(&mut f, "attn_norm", &[hidden], DType::F32);
    let ffn_norm = add(&mut f, "ffn_norm", &[hidden], DType::F32);
    let mut weights = Vec::new();
    for (name, rows, cols) in weight_shapes(spec) {
        let w = add(&mut f, name, &[rows as i64, cols as i64], wt);
        weights.push(Weight { w, scales: None });
    }
    let mut weights = weights.into_iter();
    let (wq, wk, wv, wo, gate, up, down) = (
        weights.next().expect("wq"),
        weights.next().expect("wk"),
        weights.next().expect("wv"),
        weights.next().expect("wo"),
        weights.next().expect("w_gate"),
        weights.next().expect("w_up"),
        weights.next().expect("w_down"),
    );
    let (mut bq, mut bk, mut bv) = (None, None, None);
    if spec.qkv_bias {
        bq = Some(add(&mut f, "bq", &[h * dk], DType::F32));
        bk = Some(add(&mut f, "bk", &[kvh * dk], DType::F32));
        bv = Some(add(&mut f, "bv", &[kvh * dk], DType::F32));
    }
    let (mut q_norm, mut k_norm) = (None, None);
    if spec.qk_norm {
        q_norm = Some(add(&mut f, "q_norm", &[dk], DType::F32));
        k_norm = Some(add(&mut f, "k_norm", &[dk], DType::F32));
    }
    let cos = add(&mut f, "cos", &[spec.rope_rows as i64, half], DType::F32);
    let sin = add(&mut f, "sin", &[spec.rope_rows as i64, half], DType::F32);
    let key_pool = add(
        &mut f,
        "key_pool",
        &[spec.pool_rows as i64, kvh * dk],
        DType::F16,
    );
    let value_pool = add(
        &mut f,
        "value_pool",
        &[spec.pool_rows as i64, kvh * dk],
        DType::F16,
    );
    let positions = add(&mut f, "positions", &[r], DType::I32);
    let (slot_ids, layer_row0, tables, lengths) = if spec.dense {
        (
            positions.clone(),
            f.constant(0.0, &[], DType::I32),
            positions.clone(),
            positions.clone(),
        )
    } else {
        (
            add(&mut f, "slot_ids", &[r], DType::I32),
            add(&mut f, "layer_row0", &[], DType::I32),
            add(&mut f, "tables", &[r, spec.stride as i64], DType::I32),
            add(&mut f, "lengths", &[r], DType::I32),
        )
    };
    f.set_param_attr(&key_pool, "tf.aliasing_output = 1 : i32".into());
    f.set_param_attr(&value_pool, "tf.aliasing_output = 2 : i32".into());
    let values = BlockValues {
        x,
        attn_norm,
        ffn_norm,
        wq,
        wk,
        wv,
        wo,
        gate,
        up,
        down,
        bq,
        bk,
        bv,
        q_norm,
        k_norm,
        cos,
        sin,
        key_pool,
        value_pool,
        positions,
        slot_ids,
        layer_row0,
        tables,
        lengths,
    };
    let out = emit_block(&mut f, spec, &values);
    let results = vec![
        out.x.ty.clone(),
        out.key_pool.ty.clone(),
        out.value_pool.ty.clone(),
    ];
    let text = stablehlo::module(
        "decoder_block",
        &f,
        &[&out.x, &out.key_pool, &out.value_pool],
    );
    Ok(BlockProgram {
        text,
        params,
        results,
    })
}

// ----- running a block on the device ------------------------------------------------------------------------------------------------

/// The weights of one block as host f32 data (row-major, `[rows, cols]` for the matrices).
#[derive(Debug, Clone, Default)]
pub struct HostBlock {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// The weights of one block on the device.
pub struct DeviceBlock {
    pub attn_norm: Arc<super::pjrt::Buffer>,
    pub ffn_norm: Arc<super::pjrt::Buffer>,
    /// In the order of [`weight_shapes`].
    pub matrices: Vec<Arc<super::pjrt::Buffer>>,
    pub biases: Vec<Arc<super::pjrt::Buffer>>,
    pub norms: Vec<Arc<super::pjrt::Buffer>>,
}

fn host_bytes(values: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::Bf16 => values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// Upload the weights of a block (matrices in the spec's weight type, vectors in f32).
pub fn upload_block(
    shared: &Shared,
    spec: &BlockSpec,
    host: &HostBlock,
) -> Result<DeviceBlock, BackendError> {
    spec.validate().map_err(invalid)?;
    let vector = |values: &[f32],
                  len: usize,
                  what: &str|
     -> Result<Arc<super::pjrt::Buffer>, BackendError> {
        if values.len() != len {
            return Err(invalid(format!(
                "{what} has {} values, expected {len}",
                values.len()
            )));
        }
        Ok(Arc::new(shared.host_buffer(
            &host_bytes(values, DType::F32),
            DType::F32,
            &[len as i64],
        )?))
    };
    let attn_norm = vector(&host.attn_norm, spec.hidden, "attn_norm")?;
    let ffn_norm = vector(&host.ffn_norm, spec.hidden, "ffn_norm")?;
    let sources = [
        &host.wq, &host.wk, &host.wv, &host.wo, &host.gate, &host.up, &host.down,
    ];
    let mut matrices = Vec::new();
    for ((name, rows, cols), values) in weight_shapes(spec).into_iter().zip(sources) {
        if values.len() != rows * cols {
            return Err(invalid(format!(
                "{name} has {} values, expected {}",
                values.len(),
                rows * cols
            )));
        }
        matrices.push(Arc::new(shared.host_buffer(
            &host_bytes(values, spec.weight_type),
            spec.weight_type,
            &[rows as i64, cols as i64],
        )?));
    }
    let mut biases = Vec::new();
    if spec.qkv_bias {
        biases.push(vector(&host.bq, spec.q_width(), "bq")?);
        biases.push(vector(&host.bk, spec.kv_width(), "bk")?);
        biases.push(vector(&host.bv, spec.kv_width(), "bv")?);
    }
    let mut norms = Vec::new();
    if spec.qk_norm {
        norms.push(vector(&host.q_norm, spec.head_dim, "q_norm")?);
        norms.push(vector(&host.k_norm, spec.head_dim, "k_norm")?);
    }
    Ok(DeviceBlock {
        attn_norm,
        ffn_norm,
        matrices,
        biases,
        norms,
    })
}

/// Per-call data of a block.
pub struct BlockCall<'a> {
    /// First pool row of the layer (`layer * slots per layer`; paged), ignored when dense.
    pub layer_row0: usize,
    pub positions: &'a [usize],
    /// Pool slot (within the layer) of every token's key and value (paged).
    pub slots: &'a [usize],
    pub tables: &'a [i64],
    pub lengths: &'a [i64],
    pub cos: &'a Arc<Vec<f32>>,
    pub sin: &'a Arc<Vec<f32>>,
}

/// Run one decoder block: `x` (`rows x hidden` f32) is updated in place, the key and value pools get the new tokens.
pub fn run_block(
    shared: &Shared,
    spec: &BlockSpec,
    weights: &DeviceBlock,
    x: &DeviceBuffer,
    key_pool: &DeviceBuffer,
    value_pool: &DeviceBuffer,
    call: &BlockCall<'_>,
) -> Result<(), BackendError> {
    spec.validate().map_err(invalid)?;
    let rows = spec.rows;
    if call.positions.len() != rows
        || (!spec.dense
            && (call.slots.len() != rows
                || call.lengths.len() != rows
                || call.tables.len() != rows * spec.stride))
    {
        return Err(invalid(
            "block call: positions, slots, lengths and tables must have one entry per row",
        ));
    }
    if x.size < rows * spec.hidden * 4 {
        return Err(invalid(
            "block call: the residual stream is smaller than rows x hidden",
        ));
    }
    if let Some(&p) = call.positions.iter().find(|&&p| p >= spec.rope_rows) {
        return Err(invalid(format!(
            "block call: position {p} is outside the {} rows of the rope tables",
            spec.rope_rows
        )));
    }
    let (kpv, vpv) = (View::of(key_pool)?, View::of(value_pool)?);
    let xv = View::of(x)?;
    let mut meta = Meta::new();
    let xb = meta.bind(shared, &xv, DType::F32)?;
    let layer_slot = meta.push(call.layer_row0 as i64)?;
    let position_slot = meta.extend(call.positions.iter().map(|&p| p as i64))?;
    let slot_slot = meta.extend(call.slots.iter().map(|&s| (call.layer_row0 + s) as i64))?;
    let length_slot = meta.extend(call.lengths.iter().copied())?;
    let table_slot = meta.extend(call.tables.iter().copied())?;
    let cos = shared.table(call.cos, spec.rope_dim / 2)?;
    let sin = shared.table(call.sin, spec.rope_dim / 2)?;
    let mut p = Plan::new(shared, "decoder_block", meta);
    let x_in = p.input(&xb, DType::F32, rows * spec.hidden)?;
    let attn_norm = p.buffer(Arc::clone(&weights.attn_norm));
    let ffn_norm = p.buffer(Arc::clone(&weights.ffn_norm));
    let mut matrices = weights.matrices.iter().map(|m| Weight {
        w: p.buffer(Arc::clone(m)),
        scales: None,
    });
    let (wq, wk, wv, wo, gate, up, down) = (
        matrices.next().ok_or_else(|| driver("missing wq"))?,
        matrices.next().ok_or_else(|| driver("missing wk"))?,
        matrices.next().ok_or_else(|| driver("missing wv"))?,
        matrices.next().ok_or_else(|| driver("missing wo"))?,
        matrices.next().ok_or_else(|| driver("missing w_gate"))?,
        matrices.next().ok_or_else(|| driver("missing w_up"))?,
        matrices.next().ok_or_else(|| driver("missing w_down"))?,
    );
    let mut biases = weights.biases.iter().map(|b| p.buffer(Arc::clone(b)));
    let (bq, bk, bv) = (biases.next(), biases.next(), biases.next());
    let mut norms = weights.norms.iter().map(|b| p.buffer(Arc::clone(b)));
    let (q_norm, k_norm) = (norms.next(), norms.next());
    let cos_value = p.buffer(cos);
    let sin_value = p.buffer(sin);
    let key_value = p.pool(&kpv)?;
    let value_value = p.pool(&vpv)?;
    let positions = p.meta_ints(position_slot, rows);
    let (slot_ids, layer_row0, tables, lengths) = if spec.dense {
        (
            positions.clone(),
            p.f().constant(0.0, &[], DType::I32),
            positions.clone(),
            positions.clone(),
        )
    } else {
        let layer_row0 = p.meta_scalar(layer_slot);
        let slots = p.meta_ints(slot_slot, rows);
        let lengths = p.meta_ints(length_slot, rows);
        let flat = p.meta_ints(table_slot, rows * spec.stride);
        let tables = p.f().reshape(&flat, &[rows as i64, spec.stride as i64]);
        (slots, layer_row0, tables, lengths)
    };
    let x_value = p.f().reshape(&x_in, &[rows as i64, spec.hidden as i64]);
    let values = BlockValues {
        x: x_value,
        attn_norm,
        ffn_norm,
        wq,
        wk,
        wv,
        wo,
        gate,
        up,
        down,
        bq,
        bk,
        bv,
        q_norm,
        k_norm,
        cos: cos_value,
        sin: sin_value,
        key_pool: key_value,
        value_pool: value_value,
        positions,
        slot_ids,
        layer_row0,
        tables,
        lengths,
    };
    let out = emit_block(p.f(), spec, &values);
    let flat = p.f().reshape(&out.x, &[(rows * spec.hidden) as i64]);
    p.output(&xb, &flat)?;
    p.replace_pool(&kpv, &out.key_pool)?;
    p.replace_pool(&vpv, &out.value_pool)?;
    shared.run(p)?;
    Ok(())
}

// ----- reference ---------------------------------------------------------------------------------------------------------------

fn rms(row: &mut [f32], weight: &[f32], eps: f32) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let inv = (ms + eps).sqrt().recip();
    for (v, w) in row.iter_mut().zip(weight) {
        *v *= inv * w;
    }
}

fn project(x: &[f32], rows: usize, matrix: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            y[r * out_dim + o] = (0..in_dim)
                .map(|c| x[r * in_dim + c] * matrix[o * in_dim + c])
                .sum();
        }
    }
    y
}

/// A plain f32 implementation of [`emit_block`] for one sequence (`positions` are consecutive, the cache arrays are dense
/// `[context, kv_width]` f32 and updated in place). Used to check the device against; it supports the options of the spec except
/// the paged layout and the window.
#[allow(clippy::too_many_arguments)]
pub fn reference_block(
    spec: &BlockSpec,
    host: &HostBlock,
    x: &mut [f32],
    positions: &[usize],
    cos: &[f32],
    sin: &[f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
) {
    let (rows, hidden, h, kvh, dk) = (
        spec.rows,
        spec.hidden,
        spec.heads,
        spec.kv_heads,
        spec.head_dim,
    );
    let half = spec.rope_dim / 2;
    let mut normed = x.to_vec();
    for row in normed.chunks_mut(hidden) {
        rms(row, &host.attn_norm, spec.eps);
    }
    let mut q = project(&normed, rows, &host.wq, h * dk, hidden);
    let mut k = project(&normed, rows, &host.wk, kvh * dk, hidden);
    let mut v = project(&normed, rows, &host.wv, kvh * dk, hidden);
    if spec.qkv_bias {
        for r in 0..rows {
            for (i, b) in host.bq.iter().enumerate() {
                q[r * h * dk + i] += b;
            }
            for (i, b) in host.bk.iter().enumerate() {
                k[r * kvh * dk + i] += b;
            }
            for (i, b) in host.bv.iter().enumerate() {
                v[r * kvh * dk + i] += b;
            }
        }
    }
    if spec.qk_norm {
        for head in q.chunks_mut(dk) {
            rms(head, &host.q_norm, spec.eps);
        }
        for head in k.chunks_mut(dk) {
            rms(head, &host.k_norm, spec.eps);
        }
    }
    for (r, &pos) in positions.iter().enumerate() {
        for (data, heads) in [(&mut q, h), (&mut k, kvh)] {
            for head in 0..heads {
                let base = (r * heads + head) * dk;
                for d in 0..half {
                    let (a_i, b_i) = if spec.neox {
                        (d, d + half)
                    } else {
                        (2 * d, 2 * d + 1)
                    };
                    let (a, b) = (data[base + a_i], data[base + b_i]);
                    let (c, s) = (cos[pos * half + d], sin[pos * half + d]);
                    data[base + a_i] = a * c - b * s;
                    data[base + b_i] = b * c + a * s;
                }
            }
        }
    }
    let kv_width = kvh * dk;
    for (r, &pos) in positions.iter().enumerate() {
        key_cache[pos * kv_width..(pos + 1) * kv_width]
            .copy_from_slice(&k[r * kv_width..(r + 1) * kv_width]);
        value_cache[pos * kv_width..(pos + 1) * kv_width]
            .copy_from_slice(&v[r * kv_width..(r + 1) * kv_width]);
    }
    let group = h / kvh;
    let mut attended = vec![0.0f32; rows * h * dk];
    for (r, &pos) in positions.iter().enumerate() {
        for head in 0..h {
            let kv_head = head / group;
            let query = &q[(r * h + head) * dk..(r * h + head + 1) * dk];
            let scores: Vec<f32> = (0..=pos)
                .map(|p| {
                    query
                        .iter()
                        .zip(&key_cache[p * kv_width + kv_head * dk..][..dk])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * spec.scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (p, w) in weights.iter().enumerate() {
                for d in 0..dk {
                    attended[(r * h + head) * dk + d] +=
                        w / total * value_cache[p * kv_width + kv_head * dk + d];
                }
            }
        }
    }
    let projected = project(&attended, rows, &host.wo, hidden, h * dk);
    for (xv, p) in x.iter_mut().zip(&projected) {
        *xv += p * spec.residual_scale;
    }
    let mut normed = x.to_vec();
    for row in normed.chunks_mut(hidden) {
        rms(row, &host.ffn_norm, spec.eps);
    }
    let gate = project(&normed, rows, &host.gate, spec.ff, hidden);
    let up = project(&normed, rows, &host.up, spec.ff, hidden);
    let activated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            if spec.gelu {
                0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh()) * u
            } else {
                g / (1.0 + (-g).exp()) * u
            }
        })
        .collect();
    let down = project(&activated, rows, &host.down, hidden, spec.ff);
    for (xv, d) in x.iter_mut().zip(&down) {
        *xv += d * spec.residual_scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::tpu::backend::mem::Settings;
    use crate::backend::tpu::backend::weights::WeightFormat;
    use crate::backend::tpu::backend::TpuBackend;
    use crate::backend::tpu::stablehlo::validate;
    use crate::backend::Backend;

    fn spec(dense: bool, numerics: Numerics) -> BlockSpec {
        BlockSpec {
            rows: 3,
            hidden: 32,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            ff: 48,
            eps: 1e-6,
            rope_dim: 8,
            neox: true,
            rope_rows: 64,
            qkv_bias: false,
            qk_norm: false,
            gelu: false,
            residual_scale: 1.0,
            scale: 1.0 / 8f32.sqrt(),
            window: None,
            block_size: 4,
            stride: 4,
            positions: 16,
            pool_rows: if dense { 16 } else { 3 * 16 },
            dense,
            numerics,
            weight_type: if numerics == Numerics::Exact {
                DType::F32
            } else {
                DType::Bf16
            },
        }
    }

    fn variants() -> Vec<BlockSpec> {
        let mut specs = Vec::new();
        for dense in [false, true] {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                specs.push(spec(dense, numerics));
            }
        }
        let base = spec(false, Numerics::Fast);
        specs.push(BlockSpec {
            qkv_bias: true,
            qk_norm: true,
            neox: false,
            ..base.clone()
        });
        specs.push(BlockSpec {
            gelu: true,
            residual_scale: 0.5,
            rope_dim: 4,
            window: Some(6),
            ..base.clone()
        });
        specs.push(BlockSpec {
            rows: 1,
            positions: 8,
            ..base
        });
        specs
    }

    #[test]
    fn block_modules_are_well_formed() {
        for (index, s) in variants().into_iter().enumerate() {
            let program = block_module(&s).unwrap_or_else(|e| panic!("{e}"));
            let report =
                validate::check(&program.text).unwrap_or_else(|e| panic!("{e}\n{}", program.text));
            assert_eq!(report.functions, 1);
            assert_eq!(report.params, program.params.len());
            assert_eq!(program.results.len(), 3);
            assert_eq!(
                program.results[0],
                TensorType::new(&[s.rows as i64, s.hidden as i64], DType::F32)
            );
            // the pools are donated to the second and third result
            assert!(
                program.text.contains("tf.aliasing_output = 1 : i32")
                    && program.text.contains("tf.aliasing_output = 2 : i32")
            );
            if let Ok(dir) = std::env::var("FLLM_TPU_DUMP_DIR") {
                let dir = std::path::Path::new(&dir).join("blocks");
                std::fs::create_dir_all(&dir).expect("dump dir");
                let name = format!(
                    "block_{index}_{}_{:?}_{}_{}.mlir",
                    if s.dense { "dense" } else { "paged" },
                    s.numerics,
                    s.rows,
                    s.qkv_bias
                );
                std::fs::write(dir.join(name), &program.text).expect("write");
            }
        }
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let ok = spec(false, Numerics::Fast);
        assert!(BlockSpec {
            heads: 5,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(BlockSpec {
            rope_dim: 3,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(BlockSpec {
            positions: 18,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(BlockSpec {
            positions: 32,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(BlockSpec {
            weight_type: DType::F16,
            ..ok.clone()
        }
        .validate()
        .is_err());
        assert!(BlockSpec {
            dense: true,
            pool_rows: 20,
            ..ok
        }
        .validate()
        .is_err());
    }

    fn host(s: &BlockSpec) -> HostBlock {
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.2)
                .collect()
        };
        HostBlock {
            attn_norm: vec![1.0; s.hidden],
            ffn_norm: vec![1.0; s.hidden],
            wq: fill(s.q_width() * s.hidden, 0.1),
            wk: fill(s.kv_width() * s.hidden, 0.2),
            wv: fill(s.kv_width() * s.hidden, 0.3),
            wo: fill(s.hidden * s.q_width(), 0.4),
            gate: fill(s.ff * s.hidden, 0.5),
            up: fill(s.ff * s.hidden, 0.6),
            down: fill(s.hidden * s.ff, 0.7),
            bq: fill(s.q_width(), 0.8),
            bk: fill(s.kv_width(), 0.9),
            bv: fill(s.kv_width(), 1.0),
            q_norm: vec![1.0; s.head_dim],
            k_norm: vec![1.0; s.head_dim],
        }
    }

    #[test]
    fn run_block_builds_its_program_in_a_dry_run() {
        for s in variants() {
            let backend = TpuBackend::dry_run(Settings {
                numerics: s.numerics,
                weights: WeightFormat::Bf16,
            });
            let shared = backend.shared();
            let weights = upload_block(shared, &s, &host(&s)).expect("upload");
            let mut stream = backend.create_stream().expect("stream");
            let mut x = backend.allocate_buffer(s.rows * s.hidden * 4).expect("x");
            stream
                .upload(&mut x, &vec![0u8; s.rows * s.hidden * 4])
                .expect("upload x");
            let kv_width = s.kv_width();
            let cache_rows = s.pool_rows;
            let mut pools = Vec::new();
            for _ in 0..2 {
                let cell = crate::backend::tpu::backend::mem::Cell::shaped(
                    &[cache_rows as i64, kv_width as i64],
                    DType::F16,
                );
                shared.materialize(&cell, DType::F16).expect("pool");
                pools.push(DeviceBuffer::new(
                    shared.next_id(),
                    cache_rows * kv_width * 2,
                    0,
                    cell,
                ));
            }
            let half = s.rope_dim / 2;
            let table: Arc<Vec<f32>> = Arc::new(
                (0..s.rope_rows * half)
                    .map(|i| (i as f32 * 0.01).cos())
                    .collect(),
            );
            let positions: Vec<usize> = (0..s.rows).collect();
            let slots: Vec<usize> = (0..s.rows).map(|i| i + 4).collect();
            let lengths: Vec<i64> = (1..=s.rows as i64).collect();
            let mut tables = vec![-1i64; s.rows * s.stride];
            for row in tables.chunks_mut(s.stride) {
                row[0] = 0;
            }
            let call = BlockCall {
                layer_row0: s.pool_rows / 3,
                positions: &positions,
                slots: &slots,
                tables: &tables,
                lengths: &lengths,
                cos: &table,
                sin: &table,
            };
            run_block(shared, &s, &weights, &x, &pools[0], &pools[1], &call).expect("run_block");
            let programs = shared.recorded();
            assert!(!programs.is_empty());
            let text = programs.last().expect("program");
            validate::check(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        }
    }

    /// On a TPU: one dense decoder block against `reference_block`. Run with
    /// `FLLM_TPU_EXACT=1 cargo test --release --features tpu-pjrt -- --ignored tpu_hardware`.
    #[test]
    #[ignore = "needs a TPU and the libtpu plugin"]
    fn tpu_hardware_block_matches_reference() {
        let mut s = spec(true, Numerics::Exact);
        s.rows = 2;
        let backend = TpuBackend::new(0).expect("a TPU (set FLLM_TPU_LIBRARY to libtpu.so)");
        let shared = backend.shared();
        assert_eq!(
            shared.settings.numerics,
            Numerics::Exact,
            "set FLLM_TPU_EXACT=1"
        );
        let h = host(&s);
        let weights = upload_block(shared, &s, &h).expect("upload");
        let cache = backend
            .allocate_kv_cache(
                1,
                s.stride,
                s.block_size,
                s.kv_heads,
                s.head_dim,
                s.head_dim,
                1,
            )
            .expect("cache");
        let half = s.rope_dim / 2;
        let cos: Vec<f32> = (0..s.rope_rows * half)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let sin: Vec<f32> = (0..s.rope_rows * half)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let x0: Vec<f32> = (0..s.rows * s.hidden)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let mut stream = backend.create_stream().expect("stream");
        let mut x = backend.allocate_buffer(x0.len() * 4).expect("x");
        stream
            .upload(
                &mut x,
                &x0.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            )
            .expect("upload x");
        let (cos_arc, sin_arc) = (Arc::new(cos.clone()), Arc::new(sin.clone()));
        let positions = [0usize, 1];
        let call = BlockCall {
            layer_row0: 0,
            positions: &positions,
            slots: &positions,
            tables: &[],
            lengths: &[],
            cos: &cos_arc,
            sin: &sin_arc,
        };
        run_block(
            shared,
            &s,
            &weights,
            &x,
            &cache.key_pool,
            &cache.value_pool,
            &call,
        )
        .expect("run_block");
        stream.synchronize().expect("synchronize");
        let mut bytes = vec![0u8; x0.len() * 4];
        stream.download(&mut bytes, &x).expect("download");
        let device: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut want = x0.clone();
        let (mut kc, mut vc) = (
            vec![0.0f32; s.pool_rows * s.kv_width()],
            vec![0.0f32; s.pool_rows * s.kv_width()],
        );
        reference_block(&s, &h, &mut want, &positions, &cos, &sin, &mut kc, &mut vc);
        for (i, (d, w)) in device.iter().zip(&want).enumerate() {
            assert!(
                (d - w).abs() <= 2e-3 * w.abs().max(1.0),
                "element {i}: device {d}, reference {w}"
            );
        }
    }

    #[test]
    fn reference_block_is_consistent() {
        // two rows processed together equal the same rows processed one after the other
        let mut s = spec(true, Numerics::Exact);
        s.rows = 2;
        let h = host(&s);
        let cos: Vec<f32> = (0..64 * 4).map(|i| (i as f32 * 0.05).cos()).collect();
        let sin: Vec<f32> = (0..64 * 4).map(|i| (i as f32 * 0.05).sin()).collect();
        let x0: Vec<f32> = (0..2 * s.hidden).map(|i| (i as f32 * 0.11).sin()).collect();
        let (mut kc, mut vc) = (
            vec![0.0f32; 16 * s.kv_width()],
            vec![0.0f32; 16 * s.kv_width()],
        );
        let mut together = x0.clone();
        reference_block(&s, &h, &mut together, &[0, 1], &cos, &sin, &mut kc, &mut vc);
        let single = BlockSpec {
            rows: 1,
            ..s.clone()
        };
        let (mut kc2, mut vc2) = (
            vec![0.0f32; 16 * s.kv_width()],
            vec![0.0f32; 16 * s.kv_width()],
        );
        let mut first = x0[..s.hidden].to_vec();
        reference_block(
            &single,
            &h,
            &mut first,
            &[0],
            &cos,
            &sin,
            &mut kc2,
            &mut vc2,
        );
        let mut second = x0[s.hidden..].to_vec();
        reference_block(
            &single,
            &h,
            &mut second,
            &[1],
            &cos,
            &sin,
            &mut kc2,
            &mut vc2,
        );
        for (a, b) in together.iter().zip(first.iter().chain(&second)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        assert_eq!(kc, kc2);
    }
}
