//! GPT-OSS stores all experts of a layer in a few fused tensors: gate and up interleaved column by
//! column, input-major (`x @ W`), in bf16 or as MXFP4 blocks. GGUF wants one tensor per projection
//! with output rows, so the converter rebuilds them here.
use crate::gguf::{append_converted, to_f32, OutTensor};
use crate::safetensors::{Payload, Tensor};
use crate::{CheckpointError, OutputDType};
use std::sync::Arc;

fn dummy() -> Payload {
    Payload::Owned(Arc::new(Vec::new()), 0..0)
}

fn out(name: String, dims: Vec<u64>, dtype: &str, bytes: Vec<u8>) -> OutTensor {
    OutTensor {
        name,
        dims,
        source_dtype: dtype.into(),
        dtype: dtype.into(),
        data: dummy(),
        packed: None,
        data_len: bytes.len() as u64,
        source_for_output: None,
        add: 0.0,
        owned: Some(bytes),
        parts: Vec::new(),
    }
}

fn take(parts: &mut Vec<(String, Tensor)>, name: &str) -> Option<Tensor> {
    let at = parts.iter().position(|(p, _)| p == name)?;
    Some(parts.swap_remove(at).1)
}

fn width(dtype: &str) -> Result<usize, String> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        other => Err(format!("unsupported expert dtype {other}")),
    }
}

/// One element at a flat index.
fn at(t: &Tensor, w: usize, index: usize) -> Result<f32, String> {
    let bytes = t.payload.as_slice();
    to_f32(&t.dtype, &bytes[index * w..(index + 1) * w]).map_err(|e| e.to_string())
}

/// Repack one HF MXFP4 block (byte j holds elements 2j and 2j+1) into ggml's (byte j holds j and
/// j+16) and put the E8M0 scale first.
fn repack_block(scale: u8, blocks: &[u8], out: &mut Vec<u8>) {
    let element = |i: usize| {
        let byte = blocks[i / 2];
        if i.is_multiple_of(2) {
            byte & 0xF
        } else {
            byte >> 4
        }
    };
    out.push(scale);
    out.extend((0..16).map(|j| element(j) | (element(j + 16) << 4)));
}

pub(crate) fn split_experts(
    layer: u64,
    mut parts: Vec<(String, Tensor)>,
    outtype: OutputDType,
) -> Result<Vec<OutTensor>, CheckpointError> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let mut result = Vec::new();
    let float_dtype = |source: &str| -> Result<String, String> {
        Ok(match outtype {
            OutputDType::F32 => "F32".to_owned(),
            OutputDType::F16 => "F16".to_owned(),
            OutputDType::Preserve => {
                width(source)?;
                source.to_owned()
            }
        })
    };

    // biases are small: always F32
    for (part, targets) in [(
        "gate_up_proj_bias",
        ["ffn_gate_exps.bias", "ffn_up_exps.bias"],
    )] {
        if let Some(t) = take(&mut parts, part) {
            let (experts, double) = (t.shape[0] as usize, t.shape[1] as usize);
            let w = width(&t.dtype)?;
            for (k, target) in targets.iter().enumerate() {
                let mut bytes = Vec::with_capacity(experts * double / 2 * 4);
                for e in 0..experts {
                    for f in 0..double / 2 {
                        bytes.extend_from_slice(&at(&t, w, e * double + 2 * f + k)?.to_le_bytes());
                    }
                }
                result.push(out(
                    name(target),
                    vec![(double / 2) as u64, experts as u64],
                    "F32",
                    bytes,
                ));
            }
        }
    }
    if let Some(t) = take(&mut parts, "down_proj_bias") {
        let (experts, hidden) = (t.shape[0] as usize, t.shape[1] as usize);
        let w = width(&t.dtype)?;
        let mut bytes = Vec::with_capacity(experts * hidden * 4);
        for i in 0..experts * hidden {
            bytes.extend_from_slice(&at(&t, w, i)?.to_le_bytes());
        }
        result.push(out(
            name("ffn_down_exps.bias"),
            vec![hidden as u64, experts as u64],
            "F32",
            bytes,
        ));
    }

    // bf16/f16/f32 weights, input-major
    if let Some(t) = take(&mut parts, "gate_up_proj") {
        let (experts, hidden, double) = (
            t.shape[0] as usize,
            t.shape[1] as usize,
            t.shape[2] as usize,
        );
        let w = width(&t.dtype)?;
        let dtype = float_dtype(&t.dtype)?;
        for (k, target) in ["ffn_gate_exps.weight", "ffn_up_exps.weight"]
            .iter()
            .enumerate()
        {
            let mut bytes = Vec::new();
            for e in 0..experts {
                for f in 0..double / 2 {
                    for h in 0..hidden {
                        let v = at(&t, w, (e * hidden + h) * double + 2 * f + k)?;
                        append_converted(&mut bytes, &dtype, v).map_err(|e| e.to_string())?;
                    }
                }
            }
            result.push(out(
                name(target),
                vec![hidden as u64, (double / 2) as u64, experts as u64],
                &dtype,
                bytes,
            ));
        }
    }
    if let Some(t) = take(&mut parts, "down_proj") {
        let (experts, ff, hidden) = (
            t.shape[0] as usize,
            t.shape[1] as usize,
            t.shape[2] as usize,
        );
        let w = width(&t.dtype)?;
        let dtype = float_dtype(&t.dtype)?;
        let mut bytes = Vec::new();
        for e in 0..experts {
            for h in 0..hidden {
                for f in 0..ff {
                    let v = at(&t, w, (e * ff + f) * hidden + h)?;
                    append_converted(&mut bytes, &dtype, v).map_err(|e| e.to_string())?;
                }
            }
        }
        result.push(out(
            name("ffn_down_exps.weight"),
            vec![ff as u64, hidden as u64, experts as u64],
            &dtype,
            bytes,
        ));
    }

    // MXFP4 weights, output-major
    if let (Some(blocks), Some(scales)) = (
        take(&mut parts, "gate_up_proj_blocks"),
        take(&mut parts, "gate_up_proj_scales"),
    ) {
        let (experts, double, nb) = (
            blocks.shape[0] as usize,
            blocks.shape[1] as usize,
            blocks.shape[2] as usize,
        );
        let (b, s) = (blocks.payload.as_slice(), scales.payload.as_slice());
        for (k, target) in ["ffn_gate_exps.weight", "ffn_up_exps.weight"]
            .iter()
            .enumerate()
        {
            let mut bytes = Vec::with_capacity(experts * double / 2 * nb * 17);
            for e in 0..experts {
                for f in 0..double / 2 {
                    let row = e * double + 2 * f + k;
                    for blk in 0..nb {
                        let at = row * nb + blk;
                        repack_block(s[at], &b[at * 16..(at + 1) * 16], &mut bytes);
                    }
                }
            }
            result.push(out(
                name(target),
                vec![(nb * 32) as u64, (double / 2) as u64, experts as u64],
                "MXFP4",
                bytes,
            ));
        }
    }
    if let (Some(blocks), Some(scales)) = (
        take(&mut parts, "down_proj_blocks"),
        take(&mut parts, "down_proj_scales"),
    ) {
        let (experts, hidden, nb) = (
            blocks.shape[0] as usize,
            blocks.shape[1] as usize,
            blocks.shape[2] as usize,
        );
        let (b, s) = (blocks.payload.as_slice(), scales.payload.as_slice());
        let mut bytes = Vec::with_capacity(experts * hidden * nb * 17);
        for row in 0..experts * hidden {
            for blk in 0..nb {
                let at = row * nb + blk;
                repack_block(s[at], &b[at * 16..(at + 1) * 16], &mut bytes);
            }
        }
        result.push(out(
            name("ffn_down_exps.weight"),
            vec![(nb * 32) as u64, hidden as u64, experts as u64],
            "MXFP4",
            bytes,
        ));
    }

    if let Some((left, _)) = parts.first() {
        return Err(format!("unsupported GPT-OSS expert tensor {left:?}").into());
    }
    Ok(result)
}

/// Fused expert tensors whose gate and up halves are stacked (`chunk(2)`), as stored by the
/// GraniteMoE family: `[E, 2F, H]` experts, `[E, H, F]` down projections and `[2S, H]` shared MLPs.
pub(crate) fn split_chunked(
    layer: u64,
    parts: Vec<(String, Tensor)>,
    outtype: OutputDType,
) -> Result<Vec<OutTensor>, CheckpointError> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let mut result = Vec::new();
    for (part, t) in parts {
        let w = width(&t.dtype)?;
        let dtype = match outtype {
            OutputDType::F32 => "F32".to_owned(),
            OutputDType::F16 => "F16".to_owned(),
            OutputDType::Preserve => t.dtype.clone(),
        };
        let rows_of = |first: usize, count: usize, cols: usize| -> Result<Vec<u8>, String> {
            let mut bytes = Vec::with_capacity(count * cols * w);
            for r in first..first + count {
                for c in 0..cols {
                    let v = at(&t, w, r * cols + c)?;
                    append_converted(&mut bytes, &dtype, v).map_err(|e| e.to_string())?;
                }
            }
            Ok(bytes)
        };
        match part.as_str() {
            "experts_input" => {
                let (e, double, h) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let f = double / 2;
                for (k, target) in ["ffn_gate_exps.weight", "ffn_up_exps.weight"]
                    .iter()
                    .enumerate()
                {
                    let mut bytes = Vec::new();
                    for expert in 0..e {
                        bytes.extend(rows_of(expert * double + k * f, f, h)?);
                    }
                    result.push(out(
                        name(target),
                        vec![h as u64, f as u64, e as u64],
                        &dtype,
                        bytes,
                    ));
                }
            }
            "experts_gate" | "experts_up" => {
                // separate stacked tensors `[E, F, H]` (Step-3.5 Hub files)
                let (e, f, h) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let target = if part == "experts_gate" { "ffn_gate_exps.weight" } else { "ffn_up_exps.weight" };
                let bytes = rows_of(0, e * f, h)?;
                result.push(out(name(target), vec![h as u64, f as u64, e as u64], &dtype, bytes));
            }
            "experts_output" => {
                let (e, h, f) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let bytes = rows_of(0, e * h, f)?;
                result.push(out(
                    name("ffn_down_exps.weight"),
                    vec![f as u64, h as u64, e as u64],
                    &dtype,
                    bytes,
                ));
            }
            "shared_input" | "dense_input" => {
                let (double, h) = (t.shape[0] as usize, t.shape[1] as usize);
                let s = double / 2;
                let targets = if part == "shared_input" {
                    ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"]
                } else {
                    ["ffn_gate.weight", "ffn_up.weight"]
                };
                for (k, target) in targets.iter().enumerate() {
                    let bytes = rows_of(k * s, s, h)?;
                    result.push(out(name(target), vec![h as u64, s as u64], &dtype, bytes));
                }
            }
            "experts_input_t" => {
                // input-major `[E, H, 2F]`, gate and up stacked: out[e][f][h] = t[e][h][f]
                let (e, h, double) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let f = double / 2;
                for (k, target) in ["ffn_gate_exps.weight", "ffn_up_exps.weight"]
                    .iter()
                    .enumerate()
                {
                    let mut bytes = Vec::with_capacity(e * f * h * w);
                    for expert in 0..e {
                        for row in 0..f {
                            for col in 0..h {
                                let v = at(&t, w, (expert * h + col) * double + k * f + row)?;
                                append_converted(&mut bytes, &dtype, v)
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    result.push(out(
                        name(target),
                        vec![h as u64, f as u64, e as u64],
                        &dtype,
                        bytes,
                    ));
                }
            }
            "experts_output_t" => {
                // input-major `[E, F, H]` -> output rows `[E, H, F]`
                let (e, f, h) = (
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                );
                let mut bytes = Vec::with_capacity(e * h * f * w);
                for expert in 0..e {
                    for row in 0..h {
                        for col in 0..f {
                            let v = at(&t, w, (expert * f + col) * h + row)?;
                            append_converted(&mut bytes, &dtype, v).map_err(|e| e.to_string())?;
                        }
                    }
                }
                result.push(out(
                    name("ffn_down_exps.weight"),
                    vec![f as u64, h as u64, e as u64],
                    &dtype,
                    bytes,
                ));
            }
            other => return Err(format!("unsupported fused tensor {other:?}").into()),
        }
    }
    Ok(result)
}

/// xIELU keeps its parameters in four one-element tensors; fold them into
/// `[alpha_p, alpha_n, beta, eps]` as the activation uses them.
pub(crate) fn split_xielu(
    layer: u64,
    parts: Vec<(String, Tensor)>,
) -> Result<Vec<OutTensor>, CheckpointError> {
    let value = |name: &str| -> Result<f32, String> {
        let (_, t) = parts
            .iter()
            .find(|(p, _)| p == &format!("xielu.{name}"))
            .ok_or_else(|| format!("xIELU parameter {name} is missing"))?;
        at(t, width(&t.dtype)?, 0)
    };
    let softplus = |x: f32| if x > 20.0 { x } else { x.exp().ln_1p() };
    let beta = value("beta")?;
    let params = [
        softplus(value("alpha_p")?),
        beta + softplus(value("alpha_n")?),
        beta,
        value("eps")?,
    ];
    let bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(vec![out(
        format!("blk.{layer}.ffn_xielu.weight"),
        vec![4],
        "F32",
        bytes,
    )])
}
