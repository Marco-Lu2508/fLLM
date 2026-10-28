//! CPU implementation of the device mixture-of-experts operations (`Stream::moe_route_ex`, `moe_align`, `moe_grouped_linear`,
//! `moe_combine_ex`, `moe_gate_scale`). Slow and simple: it makes `fllm-rs selftest-gpu --device cpu --only moe` run the whole group,
//! which checks the test code (shapes, tables, sort consumption, reference formulas) without a GPU.
//!
//! Pointer tables hold, for every weight, the id and offset of its buffer (16 bytes); the backend keeps a weak reference per buffer id, so a
//! table never keeps a weight alive and no raw address is dereferenced.

use super::*;
use crate::backend::moe::{align_reference, route_reference};
use std::collections::HashMap;
use std::sync::Weak;

type Handle = Weak<dyn std::any::Any + Send + Sync>;
/// A dequantized expert matrix and its bias.
type Dequantized = (Vec<f32>, Option<Vec<f32>>);

fn registry() -> &'static Mutex<HashMap<u64, Handle>> {
    static REGISTRY: std::sync::OnceLock<Mutex<HashMap<u64, Handle>>> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A table of buffers as device memory: (id, offset) per entry.
pub(super) fn buffer_table(backend: &CpuBackend, buffers: &[&DeviceBuffer]) -> Result<DeviceBuffer, BackendError> {
    let mut bytes = Vec::with_capacity(buffers.len() * 16);
    {
        let mut map = registry().lock().map_err(|_| BackendError::BufferError("registry lock poisoned".into()))?;
        for b in buffers {
            map.insert(b.id, Arc::downgrade(&b.handle));
            bytes.extend_from_slice(&b.id.to_le_bytes());
            bytes.extend_from_slice(&(b.offset as u64).to_le_bytes());
        }
    }
    let table = backend.allocate_buffer(bytes.len().max(16))?;
    with_buffer_slice_mut(&table, |dst| dst[..bytes.len()].copy_from_slice(&bytes))?;
    Ok(table)
}

/// The bytes of entry `e` of a table made by [`buffer_table`], `len` of them.
fn entry_bytes(table: &DeviceBuffer, e: usize, len: usize) -> Result<Vec<u8>, BackendError> {
    let (id, offset) = with_buffer_slice(table, |t| {
        let s = t.get(e * 16..e * 16 + 16).ok_or_else(|| invalid(format!("pointer table has no entry {e}")))?;
        Ok::<_, BackendError>((u64::from_le_bytes(s[..8].try_into().unwrap()), u64::from_le_bytes(s[8..].try_into().unwrap()) as usize))
    })??;
    let handle = registry()
        .lock()
        .map_err(|_| BackendError::BufferError("registry lock poisoned".into()))?
        .get(&id)
        .and_then(Weak::upgrade)
        .ok_or_else(|| BackendError::BufferError("a weight of the pointer table was freed".into()))?;
    let view = DeviceBuffer::new(id, offset + len, 0, handle).slice(offset, len)?;
    read_bytes(&view)
}

fn read_i32(buf: &DeviceBuffer, count: usize) -> Result<Vec<i32>, BackendError> {
    with_typed_mut::<i32, _>(buf, |s| s.get(..count).map(<[i32]>::to_vec))?.ok_or_else(|| invalid("integer buffer too small"))
}

fn write_i32(buf: &DeviceBuffer, values: &[i32]) -> Result<(), BackendError> {
    with_typed_mut::<i32, _>(buf, |dst| {
        if dst.len() < values.len() {
            return Err(invalid("integer buffer too small"));
        }
        dst[..values.len()].copy_from_slice(values);
        Ok(())
    })?
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_ex(
    logits: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    ids: &mut DeviceBuffer,
    weights: &mut DeviceBuffer,
    rows: usize,
    experts: usize,
    k: usize,
    route: &MoeRoute,
) -> Result<(), BackendError> {
    if k == 0 || k > experts {
        return Err(invalid("moe_route_ex: expert counts invalid"));
    }
    let l = read_f32(logits)?;
    if l.len() < rows * experts {
        return Err(invalid("moe_route_ex: logits too small"));
    }
    let b = bias.map(read_f32).transpose()?;
    let (mut out_ids, mut out_w) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for r in 0..rows {
        for (e, w) in route_reference(&l[r * experts..(r + 1) * experts], b.as_deref(), k, route) {
            out_ids.push(e as i32);
            out_w.push(w as f32);
        }
    }
    write_i32(ids, &out_ids)?;
    write_f32(weights, &out_w, "moe_route_ex")
}

pub(super) fn align(ids: &DeviceBuffer, sort: &mut MoeSort, expert_offset: usize, num_local: usize) -> Result<(), BackendError> {
    let (pairs, block) = (sort.pairs, sort.block);
    let id = read_i32(ids, pairs)?;
    let (sorted_bytes, blocks_bytes, _) = crate::backend::moe::moe_sort_bytes(pairs, num_local, block);
    let mut sorted = vec![pairs as i32; sorted_bytes / 4];
    let mut block_expert = vec![-1i32; blocks_bytes / 4];
    let mut at = 0usize;
    for (e, members) in align_reference(&id, expert_offset, num_local).into_iter().enumerate() {
        for chunk in members.chunks(block) {
            block_expert[at / block] = e as i32;
            for (i, &p) in chunk.iter().enumerate() {
                sorted[at + i] = p as i32;
            }
            at += block;
        }
    }
    write_i32(&sort.sorted, &sorted)?;
    write_i32(&sort.block_expert, &block_expert)?;
    write_i32(&sort.post, &[at as i32])
}

fn glu(g: f32, u: f32, glu: &MoeGlu) -> f32 {
    match glu.mode {
        0 => g / (1.0 + (-g).exp()) * u,
        1 => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh()) * u,
        _ => {
            let gc = g.min(glu.limit);
            let uc = u.clamp(-glu.limit, glu.limit);
            (uc + 1.0) * gc / (1.0 + (-glu.alpha * gc).exp())
        }
    }
}

pub(super) fn grouped(call: MoeGrouped<'_>) -> Result<(), BackendError> {
    let MoeGrouped { out, x, table, table_up, bias, bias_up, input_weights, sort, quant_type, rows, cols, x_div, glu: act } = call;
    let (pairs, block) = (sort.pairs, sort.block);
    let kind = quant_type.to_gguf_type_id();
    let xv = read_f32(x)?;
    let pw = input_weights.map(read_f32).transpose()?;
    let sorted = read_i32(&sort.sorted, sort.max_blocks * block)?;
    let block_expert = read_i32(&sort.block_expert, sort.max_blocks)?;
    let post = read_i32(&sort.post, 1)?[0] as usize;
    let row_bytes = cols / quant_type.block_size().max(1) * quant_type.block_bytes();
    // block-wise FP8 keeps its tile scales and trailer behind the codes of the matrix
    let entry_len = match fllm_gguf_probe::Fp8bLayout::new(kind, cols, rows, 1) {
        Some(layout) => layout.total_len(),
        None => rows * row_bytes,
    };
    // dequantized weights and biases, once per expert that has pairs
    let mut cache: HashMap<(usize, bool), Dequantized> = HashMap::new();
    let used: Vec<usize> = (0..post / block).filter_map(|b| (block_expert[b] >= 0).then_some(block_expert[b] as usize)).collect();
    for &e in &used {
        for up in [false, true] {
            if (up && act.is_none()) || cache.contains_key(&(e, up)) {
                continue;
            }
            let (t, b) = if up { (table_up, bias_up) } else { (Some(table), bias) };
            let t = t.ok_or_else(|| invalid("moe_grouped_linear: no table for the up projection"))?;
            let raw = entry_bytes(t, e, entry_len)?;
            let w = crate::compute::dequantize(&raw, kind, rows * cols).map_err(BackendError::DriverError)?;
            let bv = match b {
                Some(bt) => Some(entry_bytes(bt, e, rows * 4)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<f32>>()),
                None => None,
            };
            cache.insert((e, up), (w, bv));
        }
    }
    let mut y = vec![0.0f32; pairs * rows];
    let dot = |w: &[f32], bias: &Option<Vec<f32>>, input: &[f32], factor: f32, r: usize| -> f32 {
        let sum: f32 = w[r * cols..(r + 1) * cols].iter().zip(input).map(|(a, b)| a * (b * factor)).sum();
        sum + bias.as_ref().map_or(0.0, |b| b[r])
    };
    for b in 0..post / block {
        if block_expert[b] < 0 {
            continue;
        }
        let e = block_expert[b] as usize;
        for &p in &sorted[b * block..(b + 1) * block] {
            let p = p as usize;
            if p >= pairs {
                continue;
            }
            let input = &xv[(p / x_div) * cols..(p / x_div + 1) * cols];
            let factor = pw.as_ref().map_or(1.0, |w| w[p]);
            let (wg, bg) = &cache[&(e, false)];
            for r in 0..rows {
                let g = dot(wg, bg, input, factor, r);
                y[p * rows + r] = match &act {
                    Some(a) => {
                        let (wu, bu) = &cache[&(e, true)];
                        glu(g, dot(wu, bu, input, factor, r), a)
                    }
                    None => g,
                };
            }
        }
    }
    // rows of pairs that no block holds (experts outside the local slice) are left as they were
    let current = read_f32(out)?;
    let touched: Vec<bool> = {
        let mut t = vec![false; pairs];
        for b in 0..post / block {
            if block_expert[b] >= 0 {
                for &p in &sorted[b * block..(b + 1) * block] {
                    if (p as usize) < pairs {
                        t[p as usize] = true;
                    }
                }
            }
        }
        t
    };
    let mut merged = current;
    for p in 0..pairs {
        if touched[p] {
            merged[p * rows..(p + 1) * rows].copy_from_slice(&y[p * rows..(p + 1) * rows]);
        }
    }
    write_f32(out, &merged, "moe_grouped_linear")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn combine_ex(
    out: &mut DeviceBuffer,
    y: &DeviceBuffer,
    weights: Option<&DeviceBuffer>,
    shared: Option<&DeviceBuffer>,
    gate: Option<&DeviceBuffer>,
    tokens: usize,
    k: usize,
    hidden: usize,
) -> Result<(), BackendError> {
    let yv = read_f32(y)?;
    let w = weights.map(read_f32).transpose()?;
    let sh = shared.map(read_f32).transpose()?;
    let g = gate.map(read_f32).transpose()?;
    if yv.len() < tokens * k * hidden {
        return Err(invalid("moe_combine_ex: y too small"));
    }
    let mut o = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w.as_ref().map_or(1.0, |w| w[t * k + j]) * yv[(t * k + j) * hidden + h];
            }
            if let Some(s) = &sh {
                acc += s[t * hidden + h] * g.as_ref().map_or(1.0, |g| g[t]);
            }
            o[t * hidden + h] = acc;
        }
    }
    write_f32(out, &o, "moe_combine_ex")
}

pub(super) fn gate_scale(gate: &mut DeviceBuffer, x: &DeviceBuffer, g: &DeviceBuffer, tokens: usize, hidden: usize) -> Result<(), BackendError> {
    let xv = read_f32(x)?;
    let gv = read_f32(g)?;
    let out: Vec<f32> = (0..tokens)
        .map(|t| {
            let dot: f32 = xv[t * hidden..(t + 1) * hidden].iter().zip(&gv).map(|(a, b)| a * b).sum();
            1.0 / (1.0 + (-dot).exp())
        })
        .collect();
    write_f32(gate, &out, "moe_gate_scale")
}
