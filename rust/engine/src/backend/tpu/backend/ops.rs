//! The operations of the TPU backend: each one builds a StableHLO program from the shapes of its arguments (`stablehlo::ops`),
//! binds the device arrays it reads and writes (`Plan`), and runs it.
//!
//! Buffers are windows (byte offset and size) onto arrays in device memory (`mem::Cell`). The CPU backend defines the semantics: the
//! element count of an input is its window size, outputs are written to the front of their window, and an output may alias an input.
//!
//! KV cache layout: a pool is one array `[layers * slots, kv_heads * dim]` of f16, the row of (layer, slot) being
//! `layer * slots + slot`; that is the byte layout of `DeviceKvCache` for `kv_kind == 1` (f16), so uploading or downloading the pool as
//! raw bytes (what the self test does) is the same bytes. The q8 cache (`kv_kind == 2`) is not implemented on the TPU.
//!
//! Level of verification: L0 (the programs parse and verify as StableHLO; no TPU has run them).

use super::mem::{driver, invalid, Shared, View};
use super::plan::{Bound, Meta, Plan};
use super::weights::WeightMem;
use crate::backend::tpu::stablehlo::ops::{AttentionInputs, AttentionSpec, Weight};
use crate::backend::tpu::stablehlo::DType;
use crate::backend::{
    validate_kv_write, AttentionParams, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight,
    RopeParams,
};
use std::sync::Arc;

type Res<T> = Result<T, BackendError>;

fn unsupported(what: &str) -> BackendError {
    BackendError::UnsupportedOperation(format!("tpu: {what}"))
}

/// Bind the window of an f32 buffer.
fn f32_window(meta: &mut Meta, shared: &Shared, buffer: &DeviceBuffer) -> Res<Bound> {
    let view = View::of(buffer)?;
    meta.bind(shared, &view, DType::F32)
}

fn dim(n: usize) -> i64 {
    n as i64
}

// ----- data movement -----------------------------------------------------------------------------------------------------------------

/// `Stream::upload`.
pub fn upload(shared: &Shared, dst: &DeviceBuffer, src: &[u8]) -> Res<()> {
    if src.len() > dst.size {
        return Err(BackendError::BufferError(format!(
            "upload overflow: src {} > dst {}",
            src.len(),
            dst.size
        )));
    }
    if src.is_empty() {
        return Ok(());
    }
    let view = View::of(dst)?;
    let stored = view.cell.lock()?.dtype;
    let dtype = stored.unwrap_or(
        if src.len().is_multiple_of(4) && dst.offset.is_multiple_of(4) {
            DType::F32
        } else {
            DType::U8
        },
    );
    let width = dtype.bytes();
    if !src.len().is_multiple_of(width) || !dst.offset.is_multiple_of(width) {
        return Err(BackendError::BufferError(format!("upload of {} bytes at offset {} is not aligned to the {width}-byte elements of the buffer", src.len(), dst.offset)));
    }
    if !view.cell.is_flat() {
        // a whole shaped array (a KV pool) is replaced from the host
        let (dims, cell_dtype) = view.cell.signature(dtype);
        if dst.offset != 0 || src.len() != view.cell.bytes {
            return Err(BackendError::BufferError(
                "this kind of buffer can only be uploaded as a whole".into(),
            ));
        }
        let buffer = shared.host_buffer(src, cell_dtype, &dims)?;
        let mut state = view.cell.lock()?;
        state.buf = Some(buffer);
        state.dtype = Some(cell_dtype);
        return Ok(());
    }
    let elements = src.len() / width;
    if dst.offset == 0 && src.len() == view.cell.bytes {
        let buffer = shared.host_buffer(src, dtype, &[elements as i64])?;
        let mut state = view.cell.lock()?;
        state.buf = Some(buffer);
        state.dtype = Some(dtype);
        return Ok(());
    }
    // a window of a larger array: stage the bytes on the device and write them in place
    let staged = Arc::new(shared.host_buffer(src, dtype, &[elements as i64])?);
    let mut meta = Meta::new();
    let bound = meta.bind(shared, &view, dtype)?;
    let mut p = Plan::new(shared, "upload", meta);
    let value = p.buffer(staged);
    p.output(&bound, &value)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::download`.
pub fn download(shared: &Shared, dst: &mut [u8], src: &DeviceBuffer) -> Res<()> {
    if dst.len() > src.size {
        return Err(BackendError::BufferError(format!(
            "download overflow: dst {} > src {}",
            dst.len(),
            src.size
        )));
    }
    if dst.is_empty() {
        return Ok(());
    }
    let view = View::of(src)?;
    let (dtype, written) = {
        let state = view.cell.lock()?;
        (state.dtype, state.buf.is_some())
    };
    let Some(dtype) = dtype.filter(|_| written) else {
        // never written: zeros
        dst.fill(0);
        return Ok(());
    };
    if view.offset == 0 && dst.len() == view.cell.bytes {
        let state = view.cell.lock()?;
        let buffer = state
            .buf
            .as_ref()
            .ok_or_else(|| driver("buffer vanished"))?;
        let bytes = buffer.to_host().map_err(driver)?;
        dst.copy_from_slice(&bytes[..dst.len()]);
        return Ok(());
    }
    if !view.cell.is_flat() {
        return Err(BackendError::BufferError(
            "this kind of buffer can only be downloaded as a whole".into(),
        ));
    }
    let width = dtype.bytes();
    if !view.offset.is_multiple_of(width) {
        return Err(BackendError::BufferError(format!(
            "download window at offset {} is not aligned to {width} bytes",
            view.offset
        )));
    }
    let elements = dst.len().div_ceil(width);
    if elements * width > view.size {
        return Err(BackendError::BufferError(
            "download window ends inside an element".into(),
        ));
    }
    let mut meta = Meta::new();
    let bound = meta.bind(shared, &view, dtype)?;
    let mut p = Plan::new(shared, "download", meta);
    let window = p.input(&bound, dtype, elements)?;
    p.host_result(&window);
    let mut host = shared.run(p)?;
    let bytes = host
        .pop()
        .ok_or_else(|| driver("download program returned nothing"))?
        .to_host()
        .map_err(driver)?;
    dst.copy_from_slice(&bytes[..dst.len()]);
    Ok(())
}

/// Element type for moving data between two buffers.
fn transfer_dtype(a: &View, b: &View) -> DType {
    let known = |v: &View| v.cell.state.lock().ok().and_then(|s| s.dtype);
    known(a).or_else(|| known(b)).unwrap_or(DType::F32)
}

/// `Stream::copy_buffer`: `min(dst.size, src.size)` bytes from the front of `src` to the front of `dst`.
pub fn copy_buffer(shared: &Shared, dst: &DeviceBuffer, src: &DeviceBuffer) -> Res<()> {
    let bytes = dst.size.min(src.size);
    if bytes == 0 {
        return Ok(());
    }
    let (dv, sv) = (View::of(dst)?, View::of(src)?);
    let dtype = transfer_dtype(&sv, &dv);
    let elements = bytes / dtype.bytes();
    if elements == 0 {
        return Ok(());
    }
    let mut meta = Meta::new();
    let (sb, db) = (
        meta.bind(shared, &sv, dtype)?,
        meta.bind(shared, &dv, dtype)?,
    );
    let mut p = Plan::new(shared, "copy", meta);
    let source_known = sv.cell.lock()?.dtype.is_some();
    let value = if source_known {
        p.input(&sb, dtype, elements)?
    } else {
        p.f().constant(0.0, &[dim(elements)], dtype)
    };
    p.output(&db, &value)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::zero_buffer`.
pub fn zero_buffer(shared: &Shared, buffer: &DeviceBuffer) -> Res<()> {
    if buffer.size == 0 {
        return Ok(());
    }
    let view = View::of(buffer)?;
    let dtype = view.cell.lock()?.dtype.unwrap_or(DType::F32);
    let elements = buffer.size / dtype.bytes();
    let mut meta = Meta::new();
    let bound = meta.bind(shared, &view, dtype)?;
    let mut p = Plan::new(shared, "zero", meta);
    let zeros = p.f().constant(0.0, &[dim(elements)], dtype);
    p.output(&bound, &zeros)?;
    shared.run(p)?;
    Ok(())
}

// ----- normalisation, activation, residual -------------------------------------------------------------------------------------------

/// `Stream::rms_norm` and `Stream::layer_norm` (`bias` selects the latter when `layer`).
pub fn norm(
    shared: &Shared,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    weight: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    eps: f32,
    layer: bool,
) -> Res<()> {
    let (n, c) = (x.size / 4, weight.size / 4);
    let name = if layer { "layer_norm" } else { "rms_norm" };
    if c == 0 || !n.is_multiple_of(c) {
        return Err(invalid(format!(
            "{name} input is not rows of the weight width"
        )));
    }
    if out.size < n * 4 {
        return Err(invalid(format!(
            "{name}: output holds {} values, needs {n}",
            out.size / 4
        )));
    }
    if n == 0 {
        return Ok(());
    }
    if bias.is_some_and(|b| b.size / 4 < c) {
        return Err(invalid(format!("{name} bias shorter than the weight")));
    }
    let rows = n / c;
    let mut meta = Meta::new();
    let xb = f32_window(&mut meta, shared, x)?;
    let wb = f32_window(&mut meta, shared, weight)?;
    let bb = bias.map(|b| f32_window(&mut meta, shared, b)).transpose()?;
    let ob = f32_window(&mut meta, shared, out)?;
    let mut p = Plan::new(shared, name, meta);
    let xi = p.input(&xb, DType::F32, n)?;
    let wi = p.input(&wb, DType::F32, c)?;
    let bi = bb.map(|b| p.input(&b, DType::F32, c)).transpose()?;
    let f = p.f();
    let x2 = f.reshape(&xi, &[dim(rows), dim(c)]);
    let y = if layer {
        f.layer_norm(&x2, &wi, bi.as_ref(), eps)
    } else {
        f.rms_norm(&x2, &wi, eps)
    };
    let flat = f.reshape(&y, &[dim(n)]);
    p.output(&ob, &flat)?;
    shared.run(p)?;
    Ok(())
}

/// Which gated activation.
#[derive(Debug, Clone, Copy)]
pub enum Glu {
    Swiglu,
    Geglu,
    SwigluOai { alpha: f32, limit: f32 },
}

/// `out = glu(gate, up)`.
pub fn glu(
    shared: &Shared,
    out: &DeviceBuffer,
    gate: &DeviceBuffer,
    up: &DeviceBuffer,
    kind: Glu,
) -> Res<()> {
    let n = gate.size / 4;
    if up.size / 4 != n {
        return Err(invalid("gated activation: gate and up dimensions mismatch"));
    }
    if out.size < n * 4 {
        return Err(invalid("gated activation: output smaller than its inputs"));
    }
    if n == 0 {
        return Ok(());
    }
    let mut meta = Meta::new();
    let gb = f32_window(&mut meta, shared, gate)?;
    let ub = f32_window(&mut meta, shared, up)?;
    let ob = f32_window(&mut meta, shared, out)?;
    let mut p = Plan::new(shared, "glu", meta);
    let g = p.input(&gb, DType::F32, n)?;
    let u = p.input(&ub, DType::F32, n)?;
    let f = p.f();
    let y = match kind {
        Glu::Swiglu => f.swiglu(&g, &u),
        Glu::Geglu => f.geglu(&g, &u),
        Glu::SwigluOai { alpha, limit } => f.swiglu_oai(&g, &u, alpha, limit),
    };
    p.output(&ob, &y)?;
    shared.run(p)?;
    Ok(())
}

/// `x += residual * scale`.
pub fn residual_add(
    shared: &Shared,
    x: &DeviceBuffer,
    residual: &DeviceBuffer,
    scale: f32,
) -> Res<()> {
    let n = x.size / 4;
    if residual.size / 4 < n {
        return Err(invalid("residual shorter than its target"));
    }
    if n == 0 {
        return Ok(());
    }
    let mut meta = Meta::new();
    let xb = f32_window(&mut meta, shared, x)?;
    let rb = f32_window(&mut meta, shared, residual)?;
    let mut p = Plan::new(shared, "residual_add", meta);
    let xi = p.input(&xb, DType::F32, n)?;
    let ri = p.input(&rb, DType::F32, n)?;
    let f = p.f();
    let scaled = if scale == 1.0 {
        ri
    } else {
        f.mul_const(&ri, f64::from(scale))
    };
    let sum = f.add(&xi, &scaled);
    p.output(&xb, &sum)?;
    shared.run(p)?;
    Ok(())
}

/// `out = x * sigmoid(gate)` in place on `x`.
pub fn sigmoid_gate_mul(shared: &Shared, x: &DeviceBuffer, gate: &DeviceBuffer) -> Res<()> {
    let n = x.size / 4;
    if gate.size / 4 < n {
        return Err(invalid("gate shorter than the activations"));
    }
    if n == 0 {
        return Ok(());
    }
    let mut meta = Meta::new();
    let xb = f32_window(&mut meta, shared, x)?;
    let gb = f32_window(&mut meta, shared, gate)?;
    let mut p = Plan::new(shared, "sigmoid_gate_mul", meta);
    let xi = p.input(&xb, DType::F32, n)?;
    let gi = p.input(&gb, DType::F32, n)?;
    let f = p.f();
    let sig = f.logistic(&gi);
    let y = f.mul(&xi, &sig);
    p.output(&xb, &y)?;
    shared.run(p)?;
    Ok(())
}

/// Gated attention: split the query projection into queries and gates.
#[allow(clippy::too_many_arguments)]
pub fn split_q_gate(
    shared: &Shared,
    q: &DeviceBuffer,
    gate: &DeviceBuffer,
    projected: &DeviceBuffer,
    rows: usize,
    heads: usize,
    dk: usize,
    per_head: bool,
) -> Res<()> {
    let width = heads * dk;
    if projected.size < rows * width * 8
        || q.size < rows * width * 4
        || gate.size < rows * width * 4
    {
        return Err(invalid(
            "split_q_gate buffers smaller than rows x heads x head_dim",
        ));
    }
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let (qv, gv) = (View::of(q)?, View::of(gate)?);
    if Arc::ptr_eq(&qv.cell, &gv.cell) {
        return Err(invalid(
            "split_q_gate: the query and gate outputs share one array",
        ));
    }
    let mut meta = Meta::new();
    let pb = f32_window(&mut meta, shared, projected)?;
    let qb = meta.bind(shared, &qv, DType::F32)?;
    let gb = meta.bind(shared, &gv, DType::F32)?;
    let mut p = Plan::new(shared, "split_q_gate", meta);
    let pi = p.input(&pb, DType::F32, rows * width * 2)?;
    let f = p.f();
    let p2 = f.reshape(&pi, &[dim(rows), dim(width * 2)]);
    let (qs, gs) = f.split_q_gate(&p2, heads, dk, per_head);
    let (qf, gf) = (
        f.reshape(&qs, &[dim(rows * width)]),
        f.reshape(&gs, &[dim(rows * width)]),
    );
    p.output(&qb, &qf)?;
    p.output(&gb, &gf)?;
    shared.run(p)?;
    Ok(())
}

// ----- projections --------------------------------------------------------------------------------------------------------------------

/// `Stream::quantized_linear` (`residual_scale == None`) and `Stream::quantized_linear_add`: with `Some(scale)`, `out` is the
/// residual stream and receives `out + scale * (x W^T + bias)`.
pub fn linear(
    shared: &Shared,
    out: &DeviceBuffer,
    x: &DeviceBuffer,
    weight: &DeviceWeight,
    bias: Option<&DeviceBuffer>,
    residual_scale: Option<f32>,
) -> Res<()> {
    let mem = WeightMem::of(&weight.buffer)?;
    let (rows, cols) = (mem.rows, mem.cols);
    if rows != weight.rows || cols != weight.cols {
        return Err(invalid(
            "linear: weight geometry differs from its device array",
        ));
    }
    let n = x.size / 4;
    if cols == 0 || !n.is_multiple_of(cols) {
        return Err(invalid("linear input dimensions do not match"));
    }
    let batch = n / cols;
    if batch == 0 {
        return Ok(());
    }
    if out.size < batch * rows * 4 {
        return Err(invalid(format!(
            "linear: output holds {} values, needs {}",
            out.size / 4,
            batch * rows
        )));
    }
    if bias.is_some_and(|b| b.size / 4 < rows) {
        return Err(invalid("linear bias shorter than the output rows"));
    }
    let mut meta = Meta::new();
    let xb = f32_window(&mut meta, shared, x)?;
    let bb = bias.map(|b| f32_window(&mut meta, shared, b)).transpose()?;
    let ob = f32_window(&mut meta, shared, out)?;
    let mut p = Plan::new(
        shared,
        if residual_scale.is_some() {
            "linear_add"
        } else {
            "linear"
        },
        meta,
    );
    let xi = p.input(&xb, DType::F32, n)?;
    let bi = bb.map(|b| p.input(&b, DType::F32, rows)).transpose()?;
    let residual = match residual_scale {
        Some(_) => Some(p.input(&ob, DType::F32, batch * rows)?),
        None => None,
    };
    let w = p.buffer(Arc::clone(&mem.w));
    let scales = mem.scales.as_ref().map(|s| p.buffer(Arc::clone(s)));
    let numerics = shared.settings.numerics;
    let f = p.f();
    let x2 = f.reshape(&xi, &[dim(batch), dim(cols)]);
    let mut y = f.linear(&x2, &Weight { w, scales }, numerics);
    if let Some(b) = &bi {
        y = f.add_bias(&y, b);
    }
    let flat = f.reshape(&y, &[dim(batch * rows)]);
    let result = match (residual_scale, residual) {
        (Some(scale), Some(res)) => {
            let scaled = if scale == 1.0 {
                flat
            } else {
                f.mul_const(&flat, f64::from(scale))
            };
            f.add(&res, &scaled)
        }
        _ => flat,
    };
    p.output(&ob, &result)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::embedding_lookup`.
pub fn embedding_lookup(
    shared: &Shared,
    out: &DeviceBuffer,
    ids: &[i64],
    weight: &DeviceWeight,
) -> Res<()> {
    let mem = WeightMem::of(&weight.buffer)?;
    let (rows, cols) = (mem.rows, mem.cols);
    if let Some(&bad) = ids.iter().find(|&&id| id < 0 || id as usize >= rows) {
        return Err(invalid(format!("embedding token id {bad} out of bounds")));
    }
    if ids.is_empty() {
        return Ok(());
    }
    if out.size < ids.len() * cols * 4 {
        return Err(invalid(format!(
            "embedding_lookup: output holds {} values, needs {}",
            out.size / 4,
            ids.len() * cols
        )));
    }
    let mut meta = Meta::new();
    let ob = f32_window(&mut meta, shared, out)?;
    let id_slot = meta.extend(ids.iter().copied())?;
    let mut p = Plan::new(shared, "embedding", meta);
    let w = p.buffer(Arc::clone(&mem.w));
    let scales = mem.scales.as_ref().map(|s| p.buffer(Arc::clone(s)));
    let idx = p.meta_ints(id_slot, ids.len());
    let f = p.f();
    let gathered = f.gather_rows(&w, &idx);
    let values = match scales {
        Some(s) => {
            let groups = s.dims()[1];
            let g = f.gather_rows(&s, &idx);
            let q = f.convert(&gathered, DType::F32);
            let q3 = f.reshape(&q, &[dim(ids.len()), groups, dim(cols) / groups]);
            let s3 = f.broadcast_in_dim(&g, q3.dims(), &[0, 1]);
            f.mul(&q3, &s3)
        }
        None => f.convert(&gathered, DType::F32),
    };
    let flat = f.reshape(&values, &[dim(ids.len() * cols)]);
    p.output(&ob, &flat)?;
    shared.run(p)?;
    Ok(())
}

// ----- positions, cache, attention ----------------------------------------------------------------------------------------------------

/// The inputs `rope` needs for one of q / k.
struct RopeInput<'a> {
    buffer: &'a DeviceBuffer,
    heads: usize,
}

/// Rotate q and/or k in place. Both go into one program unless they share an array.
pub fn rope(
    shared: &Shared,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    positions: &[usize],
    params: &RopeParams,
) -> Res<()> {
    params.validate(positions, q.size, k.size)?;
    if positions.is_empty() {
        return Ok(());
    }
    let (qv, kv) = (View::of(q)?, View::of(k)?);
    if Arc::ptr_eq(&qv.cell, &kv.cell) {
        rope_inputs(
            shared,
            &[RopeInput {
                buffer: q,
                heads: params.heads,
            }],
            positions,
            params,
        )?;
        return rope_inputs(
            shared,
            &[RopeInput {
                buffer: k,
                heads: params.kv_heads,
            }],
            positions,
            params,
        );
    }
    rope_inputs(
        shared,
        &[
            RopeInput {
                buffer: q,
                heads: params.heads,
            },
            RopeInput {
                buffer: k,
                heads: params.kv_heads,
            },
        ],
        positions,
        params,
    )
}

fn rope_inputs(
    shared: &Shared,
    inputs: &[RopeInput<'_>],
    positions: &[usize],
    params: &RopeParams,
) -> Res<()> {
    let (rows, half, dim_) = (positions.len(), params.rope_dim / 2, params.dim);
    let mut meta = Meta::new();
    let mut bounds = Vec::new();
    for input in inputs {
        bounds.push(f32_window(&mut meta, shared, input.buffer)?);
    }
    let position_slot = meta.extend(positions.iter().map(|&p| p as i64))?;
    let cos = shared.table(&params.cos, half)?;
    let sin = shared.table(&params.sin, half)?;
    let mut p = Plan::new(shared, "rope", meta);
    let cos_table = p.buffer(cos);
    let sin_table = p.buffer(sin);
    let pos = p.meta_ints(position_slot, rows);
    let mut values = Vec::new();
    for (input, bound) in inputs.iter().zip(&bounds) {
        values.push(p.input(bound, DType::F32, rows * input.heads * dim_)?);
    }
    let f = p.f();
    let cos_rows = f.gather_rows(&cos_table, &pos);
    let sin_rows = f.gather_rows(&sin_table, &pos);
    let mut rotated = Vec::new();
    for (input, value) in inputs.iter().zip(&values) {
        let x3 = f.reshape(value, &[dim(rows), dim(input.heads), dim(dim_)]);
        let r = f.rope_rotate(&x3, &cos_rows, &sin_rows, params.rope_dim, params.neox);
        rotated.push(f.reshape(&r, &[dim(rows * input.heads * dim_)]));
    }
    for (bound, value) in bounds.iter().zip(&rotated) {
        p.output(bound, value)?;
    }
    shared.run(p)?;
    Ok(())
}

/// `Stream::kv_cache_write`.
pub fn kv_cache_write(
    shared: &Shared,
    kv: &DeviceKvCache,
    layer: usize,
    slots: &[usize],
    k: &DeviceBuffer,
    v: &DeviceBuffer,
) -> Res<()> {
    validate_kv_write(kv, layer, slots, k.size, v.size)?;
    if kv.kv_kind != 1 {
        return Err(unsupported("the q8 KV cache"));
    }
    if slots.is_empty() {
        return Ok(());
    }
    let rows = slots.len();
    let (kw, vw) = (kv.kv_heads * kv.head_dim, kv.kv_heads * kv.value_dim);
    let (kpv, vpv) = (View::of(&kv.key_pool)?, View::of(&kv.value_pool)?);
    let (kview, vview) = (View::of(k)?, View::of(v)?);
    let mut meta = Meta::new();
    let kb = meta.bind(shared, &kview, DType::F32)?;
    let vb = meta.bind(shared, &vview, DType::F32)?;
    let first = layer * kv.slots();
    let id_slot = meta.extend(slots.iter().map(|&s| (first + s) as i64))?;
    let mut p = Plan::new(shared, "kv_write", meta);
    let ki = p.input(&kb, DType::F32, rows * kw)?;
    let vi = p.input(&vb, DType::F32, rows * vw)?;
    let key_pool = p.pool(&kpv)?;
    let value_pool = p.pool(&vpv)?;
    let ids = p.meta_ints(id_slot, rows);
    let f = p.f();
    let k2 = f.reshape(&ki, &[dim(rows), dim(kw)]);
    let v2 = f.reshape(&vi, &[dim(rows), dim(vw)]);
    let new_keys = f.cache_write(&key_pool, &ids, &k2);
    let new_values = f.cache_write(&value_pool, &ids, &v2);
    p.replace_pool(&kpv, &new_keys)?;
    p.replace_pool(&vpv, &new_values)?;
    shared.run(p)?;
    Ok(())
}

/// Blocks of a block table the program materialises: enough for `longest` positions, rounded up to a power of two so that a growing
/// sequence does not compile a new program for every block.
fn bucket_blocks(longest: usize, block_size: usize, stride: usize) -> usize {
    longest
        .div_ceil(block_size)
        .next_power_of_two()
        .clamp(1, stride.max(1))
}

/// `Stream::attention_paged`.
#[allow(clippy::too_many_arguments)]
pub fn attention_paged(
    shared: &Shared,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    kv: &DeviceKvCache,
    block_tables: &[i64],
    lengths: &[i64],
    layer: usize,
    params: &AttentionParams,
) -> Res<()> {
    if kv.kv_kind != 1 {
        return Err(unsupported("the q8 KV cache"));
    }
    params.validate(kv, block_tables, lengths, layer)?;
    let rows = lengths.len();
    let (heads, kvh, dk, dv) = (
        params.num_heads,
        params.num_kv_heads,
        params.head_dim,
        params.value_dim,
    );
    if q.size < rows * heads * dk * 4 {
        return Err(invalid(
            "attention query smaller than rows x heads x head_dim",
        ));
    }
    if out.size < rows * heads * dv * 4 {
        return Err(invalid(
            "attention output smaller than rows x heads x value_dim",
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let (qv, ov) = (View::of(q)?, View::of(out)?);
    let (kpv, vpv) = (View::of(&kv.key_pool)?, View::of(&kv.value_pool)?);
    let sinks = params
        .sinks
        .as_ref()
        .map(|s| shared.table(s, 0))
        .transpose()?;
    let alibi = params
        .alibi_slopes
        .as_ref()
        .map(|s| shared.table(s, 0))
        .transpose()?;
    let layer_row0 = (layer * kv.slots()) as i64;

    // rows per program: the gathered keys and values of a chunk (rows x positions x kv width) stay below 2^26 elements
    let longest_all = lengths.iter().copied().max().unwrap_or(1).max(1) as usize;
    let positions_all =
        bucket_blocks(longest_all, params.block_size, params.stride) * params.block_size;
    let widest = (kvh * dk).max(kvh * dv).max(1);
    let chunk_rows = ((1usize << 26) / (positions_all * widest)).clamp(1, rows);

    let mut first = 0;
    while first < rows {
        let n = chunk_rows.min(rows - first);
        let chunk_lengths = &lengths[first..first + n];
        let longest = chunk_lengths.iter().copied().max().unwrap_or(1).max(1) as usize;
        let blocks = bucket_blocks(longest, params.block_size, params.stride);
        let positions = blocks * params.block_size;

        let mut meta = Meta::new();
        let qb = meta.bind(
            shared,
            &qv.sub(first * heads * dk * 4, n * heads * dk * 4),
            DType::F32,
        )?;
        let ob = meta.bind(
            shared,
            &ov.sub(first * heads * dv * 4, n * heads * dv * 4),
            DType::F32,
        )?;
        let layer_slot = meta.push(layer_row0)?;
        let length_slot = meta.extend(chunk_lengths.iter().copied())?;
        let table_slot = meta.extend(
            block_tables[first * params.stride..(first + n) * params.stride]
                .iter()
                .copied(),
        )?;
        let mut p = Plan::new(shared, "attention", meta);
        let q_in = p.input(&qb, DType::F32, n * heads * dk)?;
        let key_pool = p.pool(&kpv)?;
        let value_pool = p.pool(&vpv)?;
        let layer_value = p.meta_scalar(layer_slot);
        let lengths_value = p.meta_ints(length_slot, n);
        let tables_flat = p.meta_ints(table_slot, n * params.stride);
        let sink_value = sinks.as_ref().map(|s| p.buffer(Arc::clone(s)));
        let alibi_value = alibi.as_ref().map(|s| p.buffer(Arc::clone(s)));
        let spec = AttentionSpec {
            rows: n,
            heads,
            kv_heads: kvh,
            head_dim: dk,
            value_dim: dv,
            stride: params.stride,
            block_size: params.block_size,
            positions,
            scale: params.scale,
            softcap: params.softcap,
            window: params.window,
            chunk: params.chunk,
            numerics: shared.settings.numerics,
        };
        let f = p.f();
        let q2 = f.reshape(&q_in, &[dim(n), dim(heads * dk)]);
        let tables = f.reshape(&tables_flat, &[dim(n), dim(params.stride)]);
        let sink_head = sink_value.as_ref().map(|s| f.slice(s, &[0], &[dim(heads)]));
        let alibi_head = alibi_value
            .as_ref()
            .map(|s| f.slice(s, &[0], &[dim(heads)]));
        let result = f.paged_attention(
            &spec,
            &AttentionInputs {
                q: &q2,
                key_pool: &key_pool,
                value_pool: &value_pool,
                layer_row0: &layer_value,
                tables: &tables,
                lengths: &lengths_value,
                sinks: sink_head.as_ref(),
                alibi: alibi_head.as_ref(),
            },
        );
        let flat = f.reshape(&result, &[dim(n * heads * dv)]);
        p.output(&ob, &flat)?;
        shared.run(p)?;
        first += n;
    }
    Ok(())
}

// ----- sampling and mixture of experts ----------------------------------------------------------------------------------------------

/// `Stream::argmax_rows`.
pub fn argmax_rows(
    shared: &Shared,
    logits: &DeviceBuffer,
    rows: usize,
    vocab: usize,
) -> Res<Vec<i32>> {
    if rows == 0 || vocab == 0 {
        return Ok(Vec::new());
    }
    if logits.size / 4 < rows * vocab {
        return Err(invalid("argmax: logits smaller than rows x vocab"));
    }
    let mut meta = Meta::new();
    let lb = f32_window(&mut meta, shared, logits)?;
    let mut p = Plan::new(shared, "argmax", meta);
    let li = p.input(&lb, DType::F32, rows * vocab)?;
    let f = p.f();
    let x2 = f.reshape(&li, &[dim(rows), dim(vocab)]);
    let best = f.argmax_rows(&x2);
    p.host_result(&best);
    let mut host = shared.run(p)?;
    let bytes = host
        .pop()
        .ok_or_else(|| driver("argmax program returned nothing"))?
        .to_host()
        .map_err(driver)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect())
}

/// `Stream::moe_topk` (and `moe_route`, which routes with renormalisation).
#[allow(clippy::too_many_arguments)]
pub fn moe_topk(
    shared: &Shared,
    logits: &DeviceBuffer,
    ids: &DeviceBuffer,
    weights: &DeviceBuffer,
    rows: usize,
    experts: usize,
    k: usize,
    normalize: bool,
    scale: f32,
) -> Res<()> {
    if experts == 0 || k == 0 || k > experts {
        return Err(invalid("moe_topk expert counts invalid"));
    }
    if logits.size / 4 < rows * experts || ids.size / 4 < rows * k || weights.size / 4 < rows * k {
        return Err(invalid(
            "moe_topk buffers smaller than rows x experts / rows x k",
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let (iv, wv) = (View::of(ids)?, View::of(weights)?);
    if Arc::ptr_eq(&iv.cell, &wv.cell) {
        return Err(invalid(
            "moe_topk: the id and weight outputs share one array",
        ));
    }
    let mut meta = Meta::new();
    let lb = f32_window(&mut meta, shared, logits)?;
    let ib = meta.bind(shared, &iv, DType::I32)?;
    let wb = meta.bind(shared, &wv, DType::F32)?;
    let mut p = Plan::new(shared, "moe_topk", meta);
    let li = p.input(&lb, DType::F32, rows * experts)?;
    let f = p.f();
    let x2 = f.reshape(&li, &[dim(rows), dim(experts)]);
    let (w, i) = f.moe_topk(&x2, k, normalize, scale);
    let wf = f.reshape(&w, &[dim(rows * k)]);
    let if_ = f.reshape(&i, &[dim(rows * k)]);
    p.output(&wb, &wf)?;
    p.output(&ib, &if_)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::moe_combine`.
pub fn moe_combine(
    shared: &Shared,
    out: &DeviceBuffer,
    y: &DeviceBuffer,
    weights: &DeviceBuffer,
    tokens: usize,
    k: usize,
    hidden: usize,
) -> Res<()> {
    if y.size / 4 < tokens * k * hidden
        || weights.size / 4 < tokens * k
        || out.size / 4 < tokens * hidden
    {
        return Err(invalid("moe_combine buffers smaller than their geometry"));
    }
    if tokens == 0 || k == 0 || hidden == 0 {
        return Ok(());
    }
    let mut meta = Meta::new();
    let yb = f32_window(&mut meta, shared, y)?;
    let wb = f32_window(&mut meta, shared, weights)?;
    let ob = f32_window(&mut meta, shared, out)?;
    let mut p = Plan::new(shared, "moe_combine", meta);
    let yi = p.input(&yb, DType::F32, tokens * k * hidden)?;
    let wi = p.input(&wb, DType::F32, tokens * k)?;
    let f = p.f();
    let y2 = f.reshape(&yi, &[dim(tokens * k), dim(hidden)]);
    let combined = f.moe_combine(&y2, &wi, tokens, k);
    let flat = f.reshape(&combined, &[dim(tokens * hidden)]);
    p.output(&ob, &flat)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::gather_rows`: `dst[i] = src[idx[i]] * weights[i]`.
pub fn gather_rows(
    shared: &Shared,
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Res<()> {
    if row_len == 0 || idx.is_empty() {
        return Ok(());
    }
    let source_rows = src.size / 4 / row_len;
    if idx.iter().any(|&i| i < 0 || i as usize >= source_rows) {
        return Err(invalid("gather_rows index outside the source"));
    }
    if weights.is_some_and(|w| w.len() < idx.len()) || dst.size / 4 < idx.len() * row_len {
        return Err(invalid("gather_rows buffers smaller than their geometry"));
    }
    let mut meta = Meta::new();
    let sb = f32_window(&mut meta, shared, src)?;
    let db = f32_window(&mut meta, shared, dst)?;
    let id_slot = meta.extend(idx.iter().map(|&i| i64::from(i)))?;
    let weight_slot = weights.map(|w| meta.extend_floats(w[..idx.len()].iter().copied()));
    let mut p = Plan::new(shared, "gather_rows", meta);
    let si = p.input(&sb, DType::F32, source_rows * row_len)?;
    let ids = p.meta_ints(id_slot, idx.len());
    let ws = weight_slot.map(|s| p.meta_floats(s, idx.len()));
    let f = p.f();
    let table = f.reshape(&si, &[dim(source_rows), dim(row_len)]);
    let mut rows = f.gather_rows(&table, &ids);
    if let Some(w) = ws {
        let wb = f.broadcast_in_dim(&w, rows.dims(), &[0]);
        rows = f.mul(&rows, &wb);
    }
    let flat = f.reshape(&rows, &[dim(idx.len() * row_len)]);
    p.output(&db, &flat)?;
    shared.run(p)?;
    Ok(())
}

/// `Stream::scatter_add_rows`: `dst[idx[i]] += src[i] * weights[i]`.
pub fn scatter_add_rows(
    shared: &Shared,
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Res<()> {
    if row_len == 0 || idx.is_empty() {
        return Ok(());
    }
    let target_rows = dst.size / 4 / row_len;
    if idx.iter().any(|&i| i < 0 || i as usize >= target_rows) {
        return Err(invalid("scatter_add_rows index outside the target"));
    }
    if weights.is_some_and(|w| w.len() < idx.len()) || src.size / 4 < idx.len() * row_len {
        return Err(invalid(
            "scatter_add_rows buffers smaller than their geometry",
        ));
    }
    let mut meta = Meta::new();
    let sb = f32_window(&mut meta, shared, src)?;
    let db = f32_window(&mut meta, shared, dst)?;
    let id_slot = meta.extend(idx.iter().map(|&i| i64::from(i)))?;
    let weight_slot = weights.map(|w| meta.extend_floats(w[..idx.len()].iter().copied()));
    let mut p = Plan::new(shared, "scatter_add_rows", meta);
    let di = p.input(&db, DType::F32, target_rows * row_len)?;
    let si = p.input(&sb, DType::F32, idx.len() * row_len)?;
    let ids = p.meta_ints(id_slot, idx.len());
    let ws = weight_slot.map(|s| p.meta_floats(s, idx.len()));
    let f = p.f();
    let table = f.reshape(&di, &[dim(target_rows), dim(row_len)]);
    let mut rows = f.reshape(&si, &[dim(idx.len()), dim(row_len)]);
    if let Some(w) = ws {
        let wb = f.broadcast_in_dim(&w, rows.dims(), &[0]);
        rows = f.mul(&rows, &wb);
    }
    let updated = f.scatter_rows(&table, &ids, &rows, true, false);
    let flat = f.reshape(&updated, &[dim(target_rows * row_len)]);
    p.output(&db, &flat)?;
    shared.run(p)?;
    Ok(())
}
