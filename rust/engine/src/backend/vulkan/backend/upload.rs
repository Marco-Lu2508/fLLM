//! `Backend::upload_weight` of the Vulkan backend: device layouts, slicing of big weights, conversion of formats without a shader.

use super::convert::{iq4_xs_to_nl, to_f16_bytes, to_f32_bytes, u32_bytes};
use super::mem::{slice_table, table_bytes, view_of, WeightKind};
use super::shared::Shared;
use crate::backend::vulkan::weights::{format_info, pack_weight, DeviceLayout};
use crate::backend::{log_once, BackendError, DeviceBuffer, DeviceWeight, QuantType};
use std::borrow::Cow;

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// A weight ready for the device: bytes in the file layout of `quant` (or float rows), the scale the shader still has to apply.
struct Prepared<'a> {
    bytes: Cow<'a, [u8]>,
    quant: QuantType,
    scale: f32,
}

/// Upload a `rows x cols` weight of `quant` (`raw` = the bytes of the file). `scale` is applied to every product by the shaders.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_weight(
    sh: &Shared,
    name: &str,
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
    scale: f32,
) -> Result<DeviceWeight, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(invalid(format!("{name}: empty weight {rows} x {cols}")));
    }
    if let Some(b) = bias {
        if b.len() != rows {
            return Err(invalid(format!("{name}: bias has {} values for {rows} rows", b.len())));
        }
    }
    let prepared = prepare(name, raw, quant, rows, cols, scale)?;
    let (buffer, perm) = place(sh, name, &prepared, rows, cols)?;
    let bias = bias.map(|b| upload_bytes(sh, &to_f32_bytes(b))).transpose()?;
    let perm = perm.map(|p| upload_bytes(sh, &u32_bytes(&p))).transpose()?;
    Ok(DeviceWeight {
        name: name.to_owned(),
        buffer,
        quant_type: prepared.quant,
        rows,
        cols,
        bias,
        scale: prepared.scale,
        perm,
    })
}

/// A device buffer holding `bytes`.
pub(crate) fn upload_bytes(sh: &Shared, bytes: &[u8]) -> Result<DeviceBuffer, BackendError> {
    let buffer = sh.new_device_buffer(bytes.len())?;
    sh.upload_sync(&view_of(&buffer)?, 0, bytes)?;
    Ok(buffer)
}

fn prepare<'a>(name: &str, raw: &'a [u8], quant: QuantType, rows: usize, cols: usize, scale: f32) -> Result<Prepared<'a>, BackendError> {
    match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            let elem = if quant == QuantType::F32 { 4 } else { 2 };
            if raw.len() != rows * cols * elem {
                return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} {quant:?}", raw.len())));
            }
            Ok(Prepared {
                bytes: Cow::Borrowed(raw),
                quant,
                scale,
            })
        }
        QuantType::IQ4_XS if cols.is_multiple_of(256) => {
            if raw.len() != rows * (cols / 256) * 136 {
                return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} IQ4_XS", raw.len())));
            }
            Ok(Prepared {
                bytes: Cow::Owned(iq4_xs_to_nl(raw)),
                quant: QuantType::IQ4_NL,
                scale,
            })
        }
        q if format_info(q).is_some() && DeviceLayout::new(q, rows, cols).is_ok() => Ok(Prepared {
            bytes: Cow::Borrowed(raw),
            quant: q,
            scale,
        }),
        q => expand(name, raw, q, rows, cols, scale),
    }
}

/// Formats without a shader decoder (and shapes the planar layouts cannot hold): dequantized once on the host. Coarse formats with 256-wide
/// rows are re-quantized to Q5_K (5.5 bit, finer than the 2-4 bit grids they came from, and served by the fast shaders), everything else is
/// stored as binary16 (binary32 when a value does not fit), with the tensor scale folded in.
fn expand<'a>(name: &str, raw: &[u8], quant: QuantType, rows: usize, cols: usize, scale: f32) -> Result<Prepared<'a>, BackendError> {
    // (block-wise FP8 tile shapes other than 128 x 128 are `Other` types too; their codes expand exactly)
    if matches!(quant, QuantType::Other(id) if !fllm_gguf_probe::fp8b_type(id)) {
        return Err(BackendError::UnsupportedQuant(quant));
    }
    let mut values = crate::compute::dequantize(raw, quant.to_gguf_type_id(), rows * cols)
        .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
    if scale != 1.0 {
        for v in &mut values {
            *v *= scale;
        }
    }
    // (bitsandbytes NF4 / FP4 / int8 are 4 and 8-bit grids of their own: expanded to F16, never re-quantized; FP8 codes with their scales expand
    // exactly to binary16: no re-quantization to Q5_K either)
    let float_source = quant.is_bnb()
        || matches!(
            quant,
            QuantType::Fp8E4m3 | QuantType::Fp8E5m2 | QuantType::Fp8Marlin | QuantType::NvFp4
        )
        || fllm_gguf_probe::fp8b_type(quant.to_gguf_type_id());
    if cols.is_multiple_of(256) && !float_source {
        use rayon::prelude::*;
        let row_bytes = cols / 256 * 176;
        let mut out = vec![0u8; rows * row_bytes];
        out.par_chunks_mut(row_bytes).zip(values.par_chunks(cols)).for_each(|(dst, src)| {
            let mut row = Vec::with_capacity(row_bytes);
            fllm_checkpoint::quantize_row(fllm_checkpoint::QType::Q5K, src, &mut row);
            dst.copy_from_slice(&row);
        });
        log_once(
            &format!("vulkan-requant-{}", quant.to_gguf_type_id()),
            &format!(
                "{quant:?} weights have no Vulkan decoder; re-quantized to Q5_K on upload (first: {name}, {} MiB)",
                out.len() >> 20
            ),
        );
        return Ok(Prepared {
            bytes: Cow::Owned(out),
            quant: QuantType::Q5_K,
            scale: 1.0,
        });
    }
    if let Some(bytes) = to_f16_bytes(&values) {
        log_once(
            &format!("vulkan-f16-{}", quant.to_gguf_type_id()),
            &format!(
                "{quant:?} weights have no Vulkan decoder; expanded to F16 on upload (first: {name}, {} MiB)",
                (bytes.len() >> 20)
            ),
        );
        return Ok(Prepared {
            bytes: Cow::Owned(bytes),
            quant: QuantType::F16,
            scale: 1.0,
        });
    }
    log_once(
        &format!("vulkan-f32-{}", quant.to_gguf_type_id()),
        &format!(
            "{quant:?} weights have no Vulkan decoder; expanded to F32 on upload (first: {name}, {} MiB)",
            (values.len() * 4) >> 20
        ),
    );
    Ok(Prepared {
        bytes: Cow::Owned(to_f32_bytes(&values)),
        quant: QuantType::F32,
        scale: 1.0,
    })
}

/// Rows `first .. first + n` of a quantized weight in the device layout (the permutation of Q4ZP weights goes with every slice).
pub(crate) fn pack_slice(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    first: usize,
    n: usize,
) -> Result<crate::backend::vulkan::weights::PackedWeight, BackendError> {
    if first == 0 && n == rows {
        return pack_weight(quant, raw, rows, cols);
    }
    let info = format_info(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
    let row_bytes = (cols / info.qk) * info.src_block_bytes;
    let block_len = rows * row_bytes;
    if raw.len() < block_len {
        return Err(invalid(format!(
            "{quant:?} {rows} x {cols} needs {block_len} bytes, got {}",
            raw.len()
        )));
    }
    let mut part = raw[first * row_bytes..(first + n) * row_bytes].to_vec();
    if quant.is_q4zp() {
        part.extend_from_slice(&raw[block_len..]);
    }
    pack_weight(quant, &part, n, cols)
}

/// Allocate the device buffer of a prepared weight and copy it there slice by slice. Returns the buffer and the Q4ZP permutation.
fn place(
    sh: &Shared,
    name: &str,
    prepared: &Prepared<'_>,
    rows: usize,
    cols: usize,
) -> Result<(DeviceBuffer, Option<Vec<u32>>), BackendError> {
    let kind = WeightKind::of(prepared.quant).ok_or(BackendError::UnsupportedQuant(prepared.quant))?;
    let caps = sh.caps();
    let slices = slice_table(kind, rows, cols, caps.max_storage_buffer_range, sh.align())?;
    let total = table_bytes(&slices);
    let buffer = sh.new_device_buffer(total)?;
    let target = view_of(&buffer)?;
    let mut perm: Option<Vec<u32>> = None;
    for s in &slices {
        match kind {
            WeightKind::Float { elem_bytes } => {
                let range = s.row_base * cols * elem_bytes..(s.row_base + s.rows) * cols * elem_bytes;
                sh.upload_sync(&target, s.offset as u64, &prepared.bytes[range])?;
            }
            WeightKind::Quant(q) => {
                let packed = pack_slice(q, &prepared.bytes, rows, cols, s.row_base, s.rows)?;
                if packed.bytes.len() != s.bytes {
                    return Err(invalid(format!(
                        "{name}: slice of {} bytes, expected {}",
                        packed.bytes.len(),
                        s.bytes
                    )));
                }
                sh.upload_sync(&target, s.offset as u64, &packed.bytes)?;
                if perm.is_none() {
                    perm = packed.perm;
                }
            }
        }
    }
    Ok((buffer, perm))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bitsandbytes NF4 / FP4 / int8 blocks have no shader: `prepare` expands them to the F16 values of the dequantizer (not a re-quantization).
    #[test]
    fn bitsandbytes_weights_are_expanded_to_f16_on_upload() {
        use fllm_gguf_probe::bnb;
        for (quant, kind) in [
            (QuantType::Nf4B64, bnb::NF4_B64),
            (QuantType::Nf4B128, bnb::NF4_B128),
            (QuantType::Fp4B64, bnb::FP4_B64),
            (QuantType::Fp4B128, bnb::FP4_B128),
            (QuantType::Int8Row, bnb::INT8_ROW),
        ] {
            let block = bnb::layout(kind).unwrap();
            // a width that is a multiple of 256 would be re-quantized to Q5_K for other formats: these stay F16
            let (rows, cols) = (3, 256);
            let blocks = rows * cols / block.block;
            let raw: Vec<u8> = (0..blocks)
                .flat_map(|b| {
                    let mut v = (0.05 + 0.01 * b as f32).to_le_bytes().to_vec();
                    v.extend((0..block.bytes - 4).map(|i| (i * 37 + b * 11) as u8));
                    v
                })
                .collect();
            let prepared = prepare("w", &raw, quant, rows, cols, 1.0).unwrap();
            assert_eq!(prepared.quant, QuantType::F16, "{quant:?}");
            let want = crate::compute::dequantize(&raw, kind, rows * cols).unwrap();
            assert_eq!(prepared.bytes.len(), rows * cols * 2);
            for (i, pair) in prepared.bytes.as_chunks::<2>().0.iter().enumerate() {
                let got = half::f16::from_le_bytes(*pair).to_f32();
                assert_eq!(got, half::f16::from_f32(want[i]).to_f32(), "{quant:?} element {i}");
            }
        }
    }
}
