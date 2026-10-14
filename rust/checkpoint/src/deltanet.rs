//! Gated DeltaNet (Qwen3-Next, Qwen3.5) tensors that need rearranging: Qwen3-Next interleaves
//! queries, keys, values and gates per key head inside `in_proj_qkvz` / `in_proj_ba`, and stores
//! the decay as `A_log`.
use crate::gguf::{append_converted, to_f32, OutTensor};
use crate::plan::LinearDims;
use crate::safetensors::{Payload, Tensor};
use crate::{CheckpointError, OutputDType};
use std::sync::Arc;

fn out(name: String, dims: Vec<u64>, dtype: &str, bytes: Vec<u8>) -> OutTensor {
    OutTensor {
        name,
        dims,
        source_dtype: dtype.into(),
        dtype: dtype.into(),
        data: Payload::Owned(Arc::new(Vec::new()), 0..0),
        packed: None,
        data_len: bytes.len() as u64,
        source_for_output: None,
        add: 0.0,
        owned: Some(bytes),
        parts: Vec::new(),
    }
}

fn width(dtype: &str) -> Result<usize, String> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        other => Err(format!("unsupported dtype {other}")),
    }
}

fn row(t: &Tensor, cols: usize, r: usize, dtype: &str, bytes: &mut Vec<u8>) -> Result<(), String> {
    let w = width(&t.dtype)?;
    let data = t.payload.as_slice();
    for c in 0..cols {
        let at = (r * cols + c) * w;
        let v = to_f32(&t.dtype, &data[at..at + w]).map_err(|e| e.to_string())?;
        append_converted(bytes, dtype, v).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn split_linear(
    layer: u64,
    parts: Vec<(String, Tensor)>,
    dims: LinearDims,
    outtype: OutputDType,
) -> Result<Vec<OutTensor>, CheckpointError> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let ratio = dims.v_heads / dims.k_heads.max(1);
    let key_dim = dims.k_heads * dims.k_dim;
    let value_dim = dims.v_heads * dims.v_dim;
    let mut result = Vec::new();
    // Olmo-Hybrid keeps query, key and value projections apart
    let (qkv_parts, parts): (Vec<_>, Vec<_>) =
        parts.into_iter().partition(|(p, _)| p.starts_with("lin_"));
    if !qkv_parts.is_empty() {
        let mut order = qkv_parts;
        order.sort_by_key(|(p, _)| p.clone()); // lin_k, lin_q, lin_v
        let pick = |name: &str| order.iter().position(|(p, _)| p == name);
        let (Some(q), Some(k), Some(v)) = (pick("lin_q"), pick("lin_k"), pick("lin_v")) else {
            return Err("incomplete query/key/value projections".into());
        };
        let dtype = match outtype {
            OutputDType::F32 => "F32".to_owned(),
            OutputDType::F16 => "F16".to_owned(),
            OutputDType::Preserve => order[q].1.dtype.clone(),
        };
        let cols = *order[q].1.shape.last().ok_or("empty tensor")? as usize;
        let mut bytes = Vec::new();
        let mut rows = 0usize;
        for index in [q, k, v] {
            let t = &order[index].1;
            let count = t.shape[0] as usize;
            rows += count;
            for r in 0..count {
                row(t, cols, r, &dtype, &mut bytes)?;
            }
        }
        result.push(out(
            name("attn_qkv.weight"),
            vec![cols as u64, rows as u64],
            &dtype,
            bytes,
        ));
    }
    // separate convolution kernels (q, k, v channels) form one kernel in that order
    let (conv_parts, parts): (Vec<_>, Vec<_>) =
        parts.into_iter().partition(|(p, _)| p.starts_with("conv_"));
    if !conv_parts.is_empty() {
        let find = |n: &str| conv_parts.iter().find(|(p, _)| p == n).map(|(_, t)| t);
        let (Some(q), Some(k), Some(v)) = (find("conv_q"), find("conv_k"), find("conv_v")) else {
            return Err("incomplete convolution kernels".into());
        };
        let dtype = match outtype {
            OutputDType::F32 => "F32".to_owned(),
            OutputDType::F16 => "F16".to_owned(),
            OutputDType::Preserve => q.dtype.clone(),
        };
        let width_k = *q.shape.last().ok_or("empty tensor")? as usize;
        let mut bytes = Vec::new();
        let mut channels = 0usize;
        for t in [q, k, v] {
            let count = t.shape[0] as usize;
            channels += count;
            for r in 0..count {
                row(t, width_k, r, &dtype, &mut bytes)?;
            }
        }
        result.push(out(
            name("ssm_conv1d.weight"),
            vec![width_k as u64, 1, channels as u64],
            &dtype,
            bytes,
        ));
    }
    for (part, t) in parts {
        let dtype = match outtype {
            OutputDType::F32 => "F32".to_owned(),
            OutputDType::F16 => "F16".to_owned(),
            OutputDType::Preserve => {
                width(&t.dtype)?;
                t.dtype.clone()
            }
        };
        let cols = *t.shape.last().ok_or("empty tensor")? as usize;
        match part.as_str() {
            "in_proj_qkvz.weight" => {
                let group = 2 * dims.k_dim + 2 * ratio * dims.v_dim;
                if t.shape[0] as usize != dims.k_heads * group {
                    return Err(format!("unexpected in_proj_qkvz shape {:?}", t.shape).into());
                }
                let (mut qkv, mut z) = (Vec::new(), Vec::new());
                // queries, keys, values and gates are each concatenated over the key heads
                for (start, len) in [
                    (0, dims.k_dim),
                    (dims.k_dim, dims.k_dim),
                    (2 * dims.k_dim, ratio * dims.v_dim),
                ] {
                    for g in 0..dims.k_heads {
                        for i in 0..len {
                            row(&t, cols, g * group + start + i, &dtype, &mut qkv)?;
                        }
                    }
                }
                for g in 0..dims.k_heads {
                    for i in 0..ratio * dims.v_dim {
                        row(
                            &t,
                            cols,
                            g * group + 2 * dims.k_dim + ratio * dims.v_dim + i,
                            &dtype,
                            &mut z,
                        )?;
                    }
                }
                result.push(out(
                    name("attn_qkv.weight"),
                    vec![cols as u64, (2 * key_dim + value_dim) as u64],
                    &dtype,
                    qkv,
                ));
                result.push(out(
                    name("attn_gate.weight"),
                    vec![cols as u64, value_dim as u64],
                    &dtype,
                    z,
                ));
            }
            "in_proj_ba.weight" => {
                let group = 2 * ratio;
                if t.shape[0] as usize != dims.k_heads * group {
                    return Err(format!("unexpected in_proj_ba shape {:?}", t.shape).into());
                }
                for (k, target) in ["ssm_beta.weight", "ssm_alpha.weight"].iter().enumerate() {
                    let mut bytes = Vec::new();
                    for g in 0..dims.k_heads {
                        for i in 0..ratio {
                            row(&t, cols, g * group + k * ratio + i, &dtype, &mut bytes)?;
                        }
                    }
                    result.push(out(
                        name(target),
                        vec![cols as u64, dims.v_heads as u64],
                        &dtype,
                        bytes,
                    ));
                }
            }
            "A_log" => {
                let w = width(&t.dtype)?;
                let count = t.shape.iter().product::<u64>() as usize;
                let mut bytes = Vec::new();
                for i in 0..count {
                    let log = to_f32(&t.dtype, &t.payload.as_slice()[i * w..(i + 1) * w])
                        .map_err(|e| e.to_string())?;
                    bytes.extend_from_slice(&(-log.exp()).to_le_bytes());
                }
                result.push(out(
                    name("ssm_a"),
                    t.shape.iter().rev().copied().collect(),
                    "F32",
                    bytes,
                ));
            }
            other => return Err(format!("unsupported linear-attention tensor {other:?}").into()),
        }
    }
    Ok(result)
}
