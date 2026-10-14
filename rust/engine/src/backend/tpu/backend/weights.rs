//! Weights on the TPU: GGUF tensors are decoded on the host (`compute::dequantize`, which knows every format of the engine) and
//! uploaded as bfloat16, float32 or int8 matrices with per-group scales.
//!
//! A TPU has no kernels for the block formats of GGUF; the matrix unit multiplies bfloat16, and int8 on some generations. So a weight
//! is stored in one of three forms, chosen by `Settings::weights`:
//!
//! - `Bf16` (default): `[rows, cols]` bfloat16, rounded to nearest even. Half the size of f32, a quarter of an f16 checkpoint's
//!   precision loss on Q8_0 and above, a visible loss only below Q4.
//! - `F32` (`FLLM_TPU_EXACT=1`): the exact decoded values; four bytes per weight.
//! - `Int8`: `[rows, cols]` int8 and `[rows, groups]` f32 scales (one per 128 columns when the width allows, otherwise one per row).
//!
//! Level of verification: L0.

use super::mem::{driver, Cell, Shared};
use crate::backend::tpu::pjrt::Buffer;
use crate::backend::tpu::stablehlo::DType;
use crate::backend::{BackendError, DeviceBuffer, DeviceWeight, QuantType};
use half::bf16;
use rayon::prelude::*;
use std::sync::Arc;

/// How a weight matrix is stored on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Bf16,
    F32,
    Int8,
}

/// The device arrays of a weight (the handle inside `DeviceWeight::buffer`).
pub struct WeightMem {
    /// `[rows, cols]`
    pub w: Arc<Buffer>,
    /// `[rows, groups]` f32 for `Int8`
    pub scales: Option<Arc<Buffer>>,
    pub rows: usize,
    pub cols: usize,
    pub format: WeightFormat,
}

impl WeightMem {
    /// The weight behind a `DeviceBuffer`.
    pub fn of(buffer: &DeviceBuffer) -> Result<Arc<Self>, BackendError> {
        Arc::clone(&buffer.handle).downcast::<Self>().map_err(|_| {
            BackendError::BufferError("the buffer is not a weight of the TPU backend".into())
        })
    }
}

/// Values decoded from `raw`, handed to `sink` in chunks of whole rows `(first row, values)`.
fn decode_rows(
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    mut sink: impl FnMut(usize, &[f32]),
) -> Result<(), BackendError> {
    let kind = quant.to_gguf_type_id();
    let whole_tensor_only = fllm_gguf_probe::fp8m_type(kind)
        || fllm_gguf_probe::q4zp_group(kind).is_some()
        || fllm_gguf_probe::q4z_group(kind).is_some();
    let decode = |bytes: &[u8], count: usize| {
        crate::compute::dequantize(bytes, kind, count)
            .map_err(|e| BackendError::UnsupportedOperation(format!("decoding {quant:?}: {e}")))
    };
    let block = quant.block_size().max(1);
    if whole_tensor_only
        || !cols.is_multiple_of(block)
        || rows == 0
        || !raw.len().is_multiple_of(rows)
    {
        let values = decode(raw, rows * cols)?;
        sink(0, &values);
        return Ok(());
    }
    let row_bytes = raw.len() / rows;
    let step = ((1usize << 24) / cols.max(1)).max(1);
    let mut first = 0;
    while first < rows {
        let n = step.min(rows - first);
        let values = decode(&raw[first * row_bytes..(first + n) * row_bytes], n * cols)?;
        sink(first, &values);
        first += n;
    }
    Ok(())
}

/// bfloat16 (round to nearest even) little-endian bytes of `values`.
fn bf16_bytes(values: &[f32], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + values.len() * 2, 0);
    out[start..]
        .par_chunks_mut(2)
        .zip(values.par_iter())
        .for_each(|(o, v)| o.copy_from_slice(&bf16::from_f32(*v).to_le_bytes()));
}

/// Columns per int8 scale: 128 when the width is a multiple of it, otherwise the whole row.
pub fn int8_groups(cols: usize) -> usize {
    if cols.is_multiple_of(128) {
        cols / 128
    } else {
        1
    }
}

/// The weight matrix in the device format: bytes of the main array and, for int8, the scales.
fn convert(
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    format: WeightFormat,
) -> Result<(Vec<u8>, Vec<f32>), BackendError> {
    let width = match format {
        WeightFormat::Bf16 => 2,
        WeightFormat::F32 => 4,
        WeightFormat::Int8 => 1,
    };
    let mut data: Vec<u8> = Vec::with_capacity(rows * cols * width);
    let mut scales: Vec<f32> = Vec::new();
    let groups = int8_groups(cols);
    let group = cols / groups;
    decode_rows(raw, quant, rows, cols, |_, values| match format {
        WeightFormat::F32 => {
            let start = data.len();
            data.resize(start + values.len() * 4, 0);
            data[start..]
                .par_chunks_mut(4)
                .zip(values.par_iter())
                .for_each(|(o, v)| o.copy_from_slice(&v.to_le_bytes()));
        }
        WeightFormat::Bf16 => bf16_bytes(values, &mut data),
        WeightFormat::Int8 => {
            let start = data.len();
            data.resize(start + values.len(), 0);
            let first_scale = scales.len();
            scales.resize(first_scale + values.len() / group, 0.0);
            data[start..]
                .par_chunks_mut(group)
                .zip(values.par_chunks(group))
                .zip(scales[first_scale..].par_iter_mut())
                .for_each(|((q, v), s)| {
                    let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    *s = scale;
                    for (o, x) in q.iter_mut().zip(v) {
                        *o = ((x / scale).round().clamp(-127.0, 127.0) as i8) as u8;
                    }
                });
        }
    })?;
    if data.len() != rows * cols * width {
        return Err(driver(format!(
            "decoded {} bytes for a {rows}x{cols} weight of {width}-byte values",
            data.len()
        )));
    }
    Ok((data, scales))
}

/// The `DeviceBuffer` of a freshly created cell.
pub fn device_buffer(shared: &Shared, cell: Arc<Cell>, size: usize) -> DeviceBuffer {
    DeviceBuffer::new(shared.next_id(), size, shared.index, cell)
}

/// A flat f32 vector uploaded to the device (biases).
pub fn upload_f32(shared: &Shared, values: &[f32]) -> Result<DeviceBuffer, BackendError> {
    let cell = Cell::flat(values.len() * 4);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = shared.host_buffer(&bytes, DType::F32, &[values.len() as i64])?;
    {
        let mut state = cell.lock()?;
        state.buf = Some(buffer);
        state.dtype = Some(DType::F32);
    }
    Ok(device_buffer(shared, cell, values.len() * 4))
}

/// `Backend::upload_weight` of the TPU backend.
pub fn upload(
    shared: &Shared,
    name: &str,
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Result<DeviceWeight, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(BackendError::InvalidShape(format!(
            "{name}: empty weight {rows}x{cols}"
        )));
    }
    if let Some(b) = bias {
        if b.len() != rows {
            return Err(BackendError::InvalidShape(format!(
                "{name}: bias has {} values for {rows} rows",
                b.len()
            )));
        }
    }
    let format = shared.settings.weights;
    let (data, scales) = convert(raw, quant, rows, cols, format).map_err(|e| match e {
        BackendError::UnsupportedOperation(_) => BackendError::UnsupportedQuant(quant),
        other => other,
    })?;
    let dtype = match format {
        WeightFormat::Bf16 => DType::Bf16,
        WeightFormat::F32 => DType::F32,
        WeightFormat::Int8 => DType::I8,
    };
    let w = Arc::new(shared.host_buffer(&data, dtype, &[rows as i64, cols as i64])?);
    let scale_buffer = if format == WeightFormat::Int8 {
        let bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        Some(Arc::new(shared.host_buffer(
            &bytes,
            DType::F32,
            &[rows as i64, int8_groups(cols) as i64],
        )?))
    } else {
        None
    };
    let stored = data.len() + scales.len() * 4;
    let mem = WeightMem {
        w,
        scales: scale_buffer,
        rows,
        cols,
        format,
    };
    let buffer = DeviceBuffer::new(shared.next_id(), stored, shared.index, Arc::new(mem));
    let bias = bias.map(|b| upload_f32(shared, b)).transpose()?;
    Ok(DeviceWeight {
        name: name.to_owned(),
        buffer,
        quant_type: quant,
        rows,
        cols,
        bias,
        scale: 1.0,
        perm: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups() {
        assert_eq!(int8_groups(256), 2);
        assert_eq!(int8_groups(77), 1);
        assert_eq!(int8_groups(128), 1);
    }

    #[test]
    fn f32_weights_convert_exactly() {
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.37 - 2.0).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (data, scales) =
            convert(&raw, QuantType::F32, 3, 4, WeightFormat::F32).expect("convert");
        assert_eq!(data, raw);
        assert!(scales.is_empty());
    }

    #[test]
    fn bf16_weights_round_to_nearest_even() {
        let raw: Vec<u8> = [1.0f32, 1.0039062, 1.2345, -2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (data, _) = convert(&raw, QuantType::F32, 1, 4, WeightFormat::Bf16).expect("convert");
        let back: Vec<f32> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect();
        assert_eq!(back[0], 1.0);
        // 1.0039062 = 1 + 2^-8 lies exactly between two bfloat16 values; ties go to the even mantissa (1.0)
        assert_eq!(back[1], 1.0);
        assert!((back[2] - 1.2345).abs() < 0.02);
        assert_eq!(back[3], -2.5);
    }

    #[test]
    fn int8_weights_keep_the_range() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (data, scales) =
            convert(&raw, QuantType::F32, 1, 256, WeightFormat::Int8).expect("convert");
        assert_eq!(scales.len(), 2);
        for (i, v) in values.iter().enumerate() {
            let back = f32::from(data[i] as i8) * scales[i / 128];
            assert!(
                (back - v).abs() <= scales[i / 128] * 0.5 + 1e-6,
                "{i}: {back} vs {v}"
            );
        }
    }

    #[test]
    fn q8_0_rows_decode_in_chunks() {
        // two rows of one Q8_0 block each: scale 0.5 (f16 0x3800), values 1..=32 and -1..=-32
        let mut raw = Vec::new();
        for sign in [1i8, -1] {
            raw.extend_from_slice(&0x3800u16.to_le_bytes());
            raw.extend((1..=32i8).map(|q| (q * sign) as u8));
        }
        let (data, _) = convert(&raw, QuantType::Q8_0, 2, 32, WeightFormat::F32).expect("convert");
        let values: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(values[0], 0.5);
        assert_eq!(values[31], 16.0);
        assert_eq!(values[32], -0.5);
        assert_eq!(values[63], -16.0);
    }
}
