//! Weight formats the converter expands to F16/F32 (or maps onto an existing compact type) because the engine has no
//! kernel of their own: bitsandbytes 4-bit (NF4, FP4, double quantization) and 8-bit (LLM.int8), OCP MXFP4 blocks of
//! compressed-tensors checkpoints and the other formats listed in `docs/changes/formats.md`.
//!
//! Every decoder implements [`Dequant`]: it produces row-major `[out_features, in_features]` f32 values on demand, in
//! bounded chunks, exactly as the reference library computes them (checked against fixtures written by the real
//! libraries, see `scripts/make_quant_fixtures.py`).
use crate::safetensors::{Payload, Tensor};
use serde_json::Value;
use std::collections::HashMap;

/// A packed weight that can be expanded to f32 values chunk by chunk.
pub trait Dequant: Send + Sync + std::fmt::Debug {
    /// Number of weight elements (`rows * cols`).
    fn element_count(&self) -> usize;
    /// Fill `dst` with the elements `start..start + dst.len()` in row-major order.
    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String>;
}

fn range(start: usize, len: usize, total: usize) -> Result<std::ops::Range<usize>, String> {
    let end = start.checked_add(len).filter(|&e| e <= total).ok_or_else(|| format!("range {start}+{len} exceeds {total} elements"))?;
    Ok(start..end)
}

fn f32_values(t: &Tensor) -> Result<Vec<f32>, String> {
    let b = t.payload.as_slice();
    match t.dtype.as_str() {
        "F32" => Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
        "F16" => Ok(b.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect()),
        "BF16" => Ok(b.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect()),
        other => Err(format!("{}: expected a float tensor, found {other}", t.name)),
    }
}

// ---------------------------------------------------------------- bitsandbytes

/// The absolute maxima of a bitsandbytes 4-bit weight, one per `blocksize` elements: stored as floats, or as 8-bit codes of a second
/// blockwise quantizer (double quantization). Kept as they are in the checkpoint; a value is computed when it is asked for.
#[derive(Debug, Clone)]
pub(crate) enum Absmax {
    /// F32, F16 or BF16 values.
    Plain { payload: Payload, dtype: Float },
    /// `code[q] * stats[i / block] + offset`.
    Nested { codes: Payload, code: Vec<f32>, stats: Vec<f32>, block: usize, offset: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Float {
    F32,
    F16,
    Bf16,
}

impl Float {
    fn of(dtype: &str) -> Option<Self> {
        match dtype {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }
    fn width(self) -> usize {
        if self == Self::F32 {
            4
        } else {
            2
        }
    }
}

impl Absmax {
    fn len(&self) -> usize {
        match self {
            Self::Plain { payload, dtype } => payload.as_slice().len() / dtype.width(),
            Self::Nested { codes, .. } => codes.as_slice().len(),
        }
    }

    fn get(&self, i: usize) -> f32 {
        match self {
            Self::Plain { payload, dtype } => {
                let b = payload.as_slice();
                match dtype {
                    Float::F32 => f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]),
                    Float::F16 => half::f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32(),
                    Float::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])) << 16),
                }
            }
            Self::Nested { codes, code, stats, block, offset } => fllm_gguf_probe::bnb::nested_absmax(codes.as_slice(), code, stats, *block, *offset, i),
        }
    }
}

/// bitsandbytes 4-bit: two codes per byte (the first element in the high nibble), a 16-entry code book and one absolute
/// maximum per `blocksize` elements, optionally stored as 8-bit codes themselves (double quantization). The flat element
/// order is that of the `[rows, cols]` matrix; a block may reach over several rows.
#[derive(Debug, Clone)]
pub struct Bnb4 {
    packed: Payload,
    code: [f32; 16],
    absmax: Absmax,
    blocksize: usize,
    rows: usize,
    cols: usize,
}

impl Bnb4 {
    pub(crate) fn new(packed: Payload, code: [f32; 16], absmax: Absmax, blocksize: usize, rows: usize, cols: usize) -> Result<Self, String> {
        let elements = rows.checked_mul(cols).filter(|&n| n > 0).ok_or("bitsandbytes weight shape must be nonzero")?;
        if blocksize == 0 {
            return Err("bitsandbytes block size must be nonzero".into());
        }
        if packed.as_slice().len() != elements.div_ceil(2) {
            return Err(format!("bitsandbytes 4-bit weight has {} bytes, expected {}", packed.as_slice().len(), elements.div_ceil(2)));
        }
        if absmax.len() != elements.div_ceil(blocksize) {
            return Err(format!("bitsandbytes absmax has {} entries, expected {}", absmax.len(), elements.div_ceil(blocksize)));
        }
        Ok(Self { packed, code, absmax, blocksize, rows, cols })
    }

    /// The fLLM type that holds this weight without changing a value: NF4 or FP4 blocks of 128 values when the source blocks and the
    /// rows are multiples of 128, else of 64 (a source block of 128 or more is written as several blocks with the same absmax).
    /// `None` for a code book of another kind, a block size below 64 or a row that is not a multiple of 64.
    pub(crate) fn native_type(&self) -> Option<u32> {
        use fllm_gguf_probe::bnb::{family_of_code, type_id};
        let family = family_of_code(&self.code)?;
        let block = if self.cols.is_multiple_of(128) && self.blocksize.is_multiple_of(128) {
            128
        } else if self.cols.is_multiple_of(64) && self.blocksize.is_multiple_of(64) {
            64
        } else {
            return None;
        };
        type_id(family, block)
    }

    /// Appends the blocks of row `row` in the layout of `type_id` (a result of `native_type`).
    pub(crate) fn encode_row(&self, type_id: u32, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        let layout = fllm_gguf_probe::bnb::layout(type_id).filter(|l| l.family != fllm_gguf_probe::bnb::Family::Int8).ok_or("not a 4-bit bitsandbytes type")?;
        if row >= self.rows || !self.cols.is_multiple_of(layout.block) || !self.blocksize.is_multiple_of(layout.block) {
            return Err(format!("row {row} cannot be written as {} values per block", layout.block));
        }
        let bytes = self.packed.as_slice();
        for b in 0..self.cols / layout.block {
            // the first element of the block is even: the block starts on a byte
            let first = row * self.cols + b * layout.block;
            let codes = &bytes[first / 2..first / 2 + layout.block / 2];
            fllm_gguf_probe::bnb::push_block4(dst, self.absmax.get(first / self.blocksize), codes);
        }
        Ok(())
    }
}

impl Dequant for Bnb4 {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.element_count())?;
        let bytes = self.packed.as_slice();
        for (slot, e) in dst.iter_mut().zip(r) {
            let byte = bytes[e / 2];
            let nibble = if e % 2 == 0 { byte >> 4 } else { byte & 0x0f };
            *slot = self.code[nibble as usize] * self.absmax.get(e / self.blocksize);
        }
        Ok(())
    }
}

/// LLM.int8(): int8 weights, one absmax statistic (`SCB`) per output row; `w = q * SCB / 127`. The outlier columns that
/// bitsandbytes multiplies in 16-bit at run time are columns of this same matrix (it keeps their int8 codes and dequantizes
/// them for the 16-bit product), so the weight is exactly `q * SCB / 127` everywhere and the engine, which does not quantize
/// the activations, needs no decomposition.
#[derive(Debug, Clone)]
pub struct Bnb8 {
    weight: Payload,
    scb: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Bnb8 {
    pub(crate) fn new(weight: Payload, scb: Vec<f32>, rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || weight.as_slice().len() != rows * cols {
            return Err(format!("int8 weight has {} bytes, expected {rows} x {cols}", weight.as_slice().len()));
        }
        if scb.len() != rows {
            return Err(format!("SCB has {} entries, expected one per output row ({rows})", scb.len()));
        }
        Ok(Self { weight, scb, rows, cols })
    }

    /// `INT8R` blocks when the rows are a multiple of 64 values.
    pub(crate) fn native_type(&self) -> Option<u32> {
        self.cols.is_multiple_of(64).then_some(fllm_gguf_probe::bnb::INT8_ROW)
    }

    pub(crate) fn encode_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        if row >= self.rows || !self.cols.is_multiple_of(64) {
            return Err(format!("row {row} cannot be written as int8 blocks"));
        }
        let scale = fllm_gguf_probe::bnb::int8_scale(self.scb[row]);
        let bytes = &self.weight.as_slice()[row * self.cols..(row + 1) * self.cols];
        for block in bytes.as_chunks::<64>().0 {
            fllm_gguf_probe::bnb::push_block8(dst, scale, block);
        }
        Ok(())
    }
}

impl Dequant for Bnb8 {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.rows * self.cols)?;
        let bytes = self.weight.as_slice();
        for (slot, e) in dst.iter_mut().zip(r) {
            *slot = f32::from(bytes[e] as i8) * fllm_gguf_probe::bnb::int8_scale(self.scb[e / self.cols]);
        }
        Ok(())
    }
}

/// Parses the JSON blob of `<name>.weight.quant_state.bitsandbytes__{nf4,fp4}`.
fn quant_state(t: &Tensor) -> Result<Value, String> {
    serde_json::from_slice(t.payload.as_slice()).map_err(|e| format!("{}: invalid bitsandbytes quant_state JSON: {e}", t.name))
}

/// Splits bitsandbytes tensors out of a checkpoint: returns the remaining dense tensors and the packed weights.
pub(crate) fn split_bnb(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    use crate::packed_quant::PackedData;
    const COMPANIONS: [&str; 4] = ["absmax", "quant_map", "nested_absmax", "nested_quant_map"];
    let mut dense = Vec::new();
    let mut weights: Vec<Tensor> = Vec::new();
    let mut companions: HashMap<String, HashMap<String, Tensor>> = HashMap::new();
    let mut scb: HashMap<String, Tensor> = HashMap::new();
    let mut formats: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        let name = t.name.clone();
        if let Some(base) = name.strip_suffix(".SCB") {
            scb.insert(base.to_owned(), t);
        } else if let Some(base) = name.strip_suffix(".weight_format") {
            formats.insert(base.to_owned(), t);
        } else if name.ends_with(".CB") {
            // older LLM.int8 checkpoints also carry the unscaled copy
        } else if let Some((base, suffix)) = name.split_once(".weight.") {
            let key = if COMPANIONS.contains(&suffix) {
                suffix.to_owned()
            } else if suffix.starts_with("quant_state.bitsandbytes__") {
                "quant_state".to_owned()
            } else {
                return Err(format!("unsupported bitsandbytes tensor {name}"));
            };
            companions.entry(format!("{base}.weight")).or_default().insert(key, t);
        } else if name.ends_with(".weight") && t.shape.len() == 2 && ((t.dtype == "U8" && t.shape[1] == 1) || t.dtype == "I8") {
            // 4-bit codes are stored as a [bytes, 1] column; int8 weights keep their [out, in] shape
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let mapped = map(&w.name)?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        if w.dtype == "I8" {
            let s = scb.remove(&base).ok_or_else(|| format!("int8 weight {} has no {base}.SCB", w.name))?;
            if let Some(format) = formats.remove(&base) {
                // 0 is the row-major layout; the tiled ones (col32, col_turing, col_ampere) interleave the rows for an igemm kernel
                if format.payload.as_slice().iter().any(|&b| b != 0) {
                    return Err(format!(
                        "{}: the int8 weight is stored in a tiled GPU layout (weight_format {:?}), which is not supported; use a checkpoint saved with a current bitsandbytes (row-major int8) or the original weights",
                        w.name,
                        format.payload.as_slice()
                    ));
                }
            }
            let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize);
            let d = Bnb8::new(w.payload.clone(), f32_values(&s)?, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
            packed.push((mapped, PackedData::Bnb8(d), vec![rows as u64, cols as u64], w.payload));
            continue;
        }
        let mut c = companions.remove(&w.name).ok_or_else(|| format!("4-bit weight {} has no bitsandbytes quant_state tensors", w.name))?;
        let mut take = |k: &str| c.remove(k).ok_or_else(|| format!("4-bit weight {} has no {k} tensor", w.name));
        let state = quant_state(&take("quant_state")?)?;
        let shape: Vec<u64> = state
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{}: quant_state has no shape", w.name))?
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        if shape.len() != 2 {
            return Err(format!("{}: bitsandbytes weight shape {shape:?} is not a matrix", w.name));
        }
        let blocksize = state.get("blocksize").and_then(Value::as_u64).ok_or_else(|| format!("{}: quant_state has no blocksize", w.name))? as usize;
        let code_values = f32_values(&take("quant_map")?)?;
        let code: [f32; 16] = code_values.try_into().map_err(|v: Vec<f32>| format!("{}: quant_map has {} entries, expected 16", w.name, v.len()))?;
        let absmax_tensor = take("absmax")?;
        let absmax = if absmax_tensor.dtype == "U8" {
            // double quantization: the absmax values are 8-bit codes of a second, blockwise quantizer
            let nested_blocksize = state.get("nested_blocksize").and_then(Value::as_u64).ok_or_else(|| format!("{}: nested quant_state has no nested_blocksize", w.name))? as usize;
            let offset = state.get("nested_offset").and_then(Value::as_f64).ok_or_else(|| format!("{}: nested quant_state has no nested_offset", w.name))? as f32;
            let nested_code = f32_values(&take("nested_quant_map")?)?;
            let nested_absmax = f32_values(&take("nested_absmax")?)?;
            if nested_code.len() != 256 || nested_blocksize == 0 || nested_absmax.len() != absmax_tensor.payload.as_slice().len().div_ceil(nested_blocksize) {
                return Err(format!("{}: inconsistent bitsandbytes nested statistics", w.name));
            }
            Absmax::Nested { codes: absmax_tensor.payload, code: nested_code, stats: nested_absmax, block: nested_blocksize, offset }
        } else {
            let dtype = Float::of(&absmax_tensor.dtype).ok_or_else(|| format!("{}: expected a float tensor, found {}", absmax_tensor.name, absmax_tensor.dtype))?;
            Absmax::Plain { payload: absmax_tensor.payload, dtype }
        };
        if let Some(extra) = c.keys().next() {
            return Err(format!("{}: unexpected bitsandbytes tensor {extra}", w.name));
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let d = Bnb4::new(w.payload.clone(), code, absmax, blocksize, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
        packed.push((mapped, PackedData::Bnb4(d), shape, w.payload));
    }
    if let Some(left) = scb.keys().next() {
        return Err(format!("{left}.SCB has no int8 weight"));
    }
    if let Some(left) = formats.keys().next() {
        return Err(format!("{left}.weight_format has no int8 weight"));
    }
    if let Some(left) = companions.keys().next() {
        return Err(format!("bitsandbytes tensors of {left} have no packed weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares bitsandbytes but no 4-bit or int8 weights were found".into());
    }
    Ok((dense, packed))
}

// ---------------------------------------------------------------- OCP MXFP4

/// OCP MXFP4 weights of compressed-tensors (`mxfp4-pack-quantized`): two E2M1 codes per byte (even element in the low
/// nibble) and one E8M0 exponent per 32 inputs. Written as GGUF type MXFP4 (`[e8m0][16 bytes]` per block).
#[derive(Debug, Clone)]
pub struct Mxfp4 {
    weight: Payload,
    scales: Payload,
    rows: usize,
    cols: usize,
}

/// Value of an E8M0 scale byte: `2^(e - 127)`; the lowest exponent is the f32 subnormal `2^-127`.
fn e8m0(e: u8) -> f32 {
    if e == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits(u32::from(e) << 23)
    }
}

const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

impl Mxfp4 {
    pub(crate) fn new(weight: Payload, scales: Payload, rows: usize, cols: usize) -> Result<Self, String> {
        if cols == 0 || !cols.is_multiple_of(32) {
            return Err(format!("MXFP4 needs a multiple of 32 inputs per row, found {cols}"));
        }
        if weight.as_slice().len() != rows * cols / 2 {
            return Err(format!("MXFP4 weight has {} bytes, expected {}", weight.as_slice().len(), rows * cols / 2));
        }
        if scales.as_slice().len() != rows * cols / 32 {
            return Err(format!("MXFP4 scale tensor has {} entries, expected {}", scales.as_slice().len(), rows * cols / 32));
        }
        if scales.as_slice().contains(&0xff) {
            return Err("MXFP4 scale tensor contains the E8M0 NaN code 0xff".into());
        }
        Ok(Self { weight, scales, rows, cols })
    }

    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    pub fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.element_count())?;
        let (w, s) = (self.weight.as_slice(), self.scales.as_slice());
        for (slot, e) in dst.iter_mut().zip(r) {
            let byte = w[e / 2];
            let code = if e % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *slot = E2M1[code as usize] * e8m0(s[e / 32]);
        }
        Ok(())
    }

    /// Bytes of one GGUF row: per block the E8M0 scale, then byte j holds elements j (low nibble) and j + 16 (high).
    #[cfg(test)]
    pub fn gguf_row_bytes(&self) -> usize {
        self.cols / 32 * 17
    }

    pub fn encode_gguf_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        if row >= self.rows {
            return Err(format!("row {row} is outside the {} MXFP4 rows", self.rows));
        }
        let (w, s) = (self.weight.as_slice(), self.scales.as_slice());
        let blocks = self.cols / 32;
        for b in 0..blocks {
            let at = row * blocks + b;
            let bytes = &w[at * 16..(at + 1) * 16];
            let element = |i: usize| if i.is_multiple_of(2) { bytes[i / 2] & 0xf } else { bytes[i / 2] >> 4 };
            dst.push(s[at]);
            dst.extend((0..16).map(|j| element(j) | (element(j + 16) << 4)));
        }
        Ok(())
    }
}

/// compressed-tensors `mxfp4-pack-quantized`: `weight_packed` (U8) and `weight_scale` (U8 E8M0) per linear.
pub(crate) fn split_ct_mxfp4(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut weights = Vec::new();
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        if let Some(base) = t.name.strip_suffix(".weight_scale") {
            scales.insert(base.to_owned(), t);
        } else if t.name.ends_with(".input_global_scale") {
            // activations are not quantized by the engine
        } else if t.name.ends_with(".weight_packed") {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight_packed").unwrap_or(&w.name).to_owned();
        let s = scales.remove(&base).ok_or_else(|| format!("MXFP4 weight {} has no {base}.weight_scale", w.name))?;
        if w.dtype != "U8" || w.shape.len() != 2 {
            return Err(format!("{} must be a rank-2 U8 tensor, found {} {:?}", w.name, w.dtype, w.shape));
        }
        if !matches!(s.dtype.as_str(), "U8" | "F8_E8M0") {
            return Err(format!("{} must be U8 (E8M0 exponents), found {}", s.name, s.dtype));
        }
        let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize * 2);
        let m = Mxfp4::new(w.payload.clone(), s.payload, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&format!("{base}.weight"))?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, crate::packed_quant::PackedData::Mxfp4(m), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}.weight_scale has no packed weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares MXFP4 but no weight_packed tensors were found".into());
    }
    Ok((dense, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::PackedGroup;
    use std::path::PathBuf;

    pub(crate) fn fixture(case: &str) -> (Vec<Tensor>, Vec<f32>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("quant").join(format!("{case}.safetensors"));
        let mut tensors = crate::safetensors::read_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let at = tensors.iter().position(|t| t.name == "expected").expect("fixture has an expected tensor");
        let expected = tensors.swap_remove(at);
        let values = f32_values(&expected).unwrap();
        (tensors, values)
    }

    fn decode_all(group: &PackedGroup) -> Vec<f32> {
        let mut out = vec![0f32; group.1.element_count()];
        for (i, chunk) in out.chunks_mut(37).enumerate() {
            group.1.decode_f32_chunk(i * 37, chunk).unwrap();
        }
        out
    }

    fn identity(n: &str) -> Result<Option<String>, String> {
        Ok(Some(n.to_owned()))
    }

    fn assert_close(case: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{case}");
        let mut worst = (0usize, 0f32);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        assert!(worst.1 <= tol, "{case}: max error {} at element {} (got {}, want {}), tolerance {tol}", worst.1, worst.0, got[worst.0], want[worst.0]);
    }

    #[test]
    fn bitsandbytes_4bit_matches_the_library() {
        for case in ["nf4", "fp4", "nf4_dq", "fp4_dq"] {
            for bs in [64, 128] {
                let name = format!("bnb_{case}_bs{bs}");
                let (tensors, expected) = fixture(&name);
                let (dense, packed) = split_bnb(tensors, &identity).unwrap();
                assert!(dense.is_empty(), "{name}");
                assert_eq!(packed.len(), 1);
                assert_close(&name, &decode_all(&packed[0]), &expected, 1e-6);
                assert_eq!(packed[0].2, vec![16, 256]);
            }
        }
    }

    #[test]
    fn bitsandbytes_int8_matches_the_library() {
        let (tensors, expected) = fixture("bnb_int8");
        let (dense, packed) = split_bnb(tensors, &identity).unwrap();
        assert!(dense.is_empty());
        assert_close("bnb_int8", &decode_all(&packed[0]), &expected, 1e-6);
    }

    fn native_bytes(group: &PackedGroup, type_id: u32, rows: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..rows {
            group.1.encode_bnb_row(type_id, r, &mut out).unwrap();
        }
        out
    }

    fn owned(bytes: Vec<u8>) -> Payload {
        let n = bytes.len();
        Payload::Owned(std::sync::Arc::new(bytes), 0..n)
    }

    #[test]
    fn native_bitsandbytes_blocks_decode_to_what_the_library_computes() {
        use fllm_gguf_probe::bnb;
        for (case, want_type) in [
            ("bnb_nf4_bs64", bnb::NF4_B64),
            ("bnb_nf4_bs128", bnb::NF4_B128),
            ("bnb_nf4_dq_bs64", bnb::NF4_B64),
            ("bnb_nf4_dq_bs128", bnb::NF4_B128),
            ("bnb_fp4_bs64", bnb::FP4_B64),
            ("bnb_fp4_bs128", bnb::FP4_B128),
            ("bnb_fp4_dq_bs64", bnb::FP4_B64),
            ("bnb_fp4_dq_bs128", bnb::FP4_B128),
        ] {
            let (tensors, expected) = fixture(case);
            let (_, packed) = split_bnb(tensors, &identity).unwrap();
            let (rows, cols) = (packed[0].2[0] as usize, packed[0].2[1] as usize);
            let type_id = packed[0].1.bnb_type().unwrap_or_else(|| panic!("{case}: no native type"));
            assert_eq!(type_id, want_type, "{case}");
            let bytes = native_bytes(&packed[0], type_id, rows);
            assert_eq!(bytes.len(), rows * bnb::row_bytes(type_id, cols).unwrap(), "{case}");
            let got = bnb::dequantize(&bytes, type_id, rows * cols).unwrap();
            assert_close(case, &got, &expected, 1e-6);
            if type_id != bnb::INT8_ROW {
                // 4-bit: the very same products as the checkpoint decoder, bit for bit
                assert_eq!(got, decode_all(&packed[0]), "{case}");
            }
        }
    }

    /// The int8 fixture has rows of 96 values, which no 64-value block holds: it stays on the expanding path.
    #[test]
    fn bitsandbytes_rows_that_no_block_fits_are_expanded() {
        let (tensors, expected) = fixture("bnb_int8");
        let (_, packed) = split_bnb(tensors, &identity).unwrap();
        assert_eq!(packed[0].1.bnb_type(), None);
        assert!(packed[0].1.is_expanded());
        assert_close("bnb_int8", &decode_all(&packed[0]), &expected, 1e-6);
    }

    #[test]
    fn native_bitsandbytes_blocks_follow_the_source_blocks_for_odd_shapes() {
        use fllm_gguf_probe::bnb;
        let mut state = 12345u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        // (rows, cols, blocksize, expected native type): a source block may reach over several rows or over several native blocks
        for (rows, cols, blocksize, want) in [
            (3usize, 192usize, 256usize, Some(bnb::NF4_B64)),
            (5, 128, 1024, Some(bnb::NF4_B128)),
            (7, 64, 64, Some(bnb::NF4_B64)),
            (2, 256, 128, Some(bnb::NF4_B128)),
            (4, 320, 128, Some(bnb::NF4_B64)),
            (3, 96, 64, None),
            (3, 128, 32, None),
        ] {
            let elements = rows * cols;
            let codes: Vec<u8> = (0..elements / 2).map(|_| next() as u8).collect();
            let blocks = elements.div_ceil(blocksize);
            // F16 absmax with values that differ per block
            let absmax: Vec<u8> = (0..blocks).flat_map(|b| half::f16::from_f32(0.25 + b as f32 * 0.125).to_le_bytes()).collect();
            let d = Bnb4::new(owned(codes), bnb::NF4_CODE, Absmax::Plain { payload: owned(absmax), dtype: Float::F16 }, blocksize, rows, cols).unwrap();
            assert_eq!(d.native_type(), want, "{rows}x{cols} bs{blocksize}");
            let Some(type_id) = want else { continue };
            let mut bytes = Vec::new();
            for r in 0..rows {
                d.encode_row(type_id, r, &mut bytes).unwrap();
            }
            let got = bnb::dequantize(&bytes, type_id, elements).unwrap();
            let mut reference = vec![0f32; elements];
            d.decode_chunk(0, &mut reference).unwrap();
            assert_eq!(got, reference, "{rows}x{cols} bs{blocksize}");
        }
        // a code book that is not NF4 or FP4 has no native type
        let mut odd = bnb::NF4_CODE;
        odd[5] += 0.01;
        let d = Bnb4::new(owned(vec![0; 32]), odd, Absmax::Plain { payload: owned(1f32.to_le_bytes().to_vec()), dtype: Float::F32 }, 64, 1, 64).unwrap();
        assert_eq!(d.native_type(), None);
        assert!(crate::packed_quant::PackedData::Bnb4(d).is_expanded());
    }

    #[test]
    fn stacked_bitsandbytes_experts_are_written_expert_by_expert() {
        use crate::packed_quant::PackedData;
        use fllm_gguf_probe::bnb;
        let mut state = 99u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        let (rows, cols) = (3usize, 128usize);
        let mut expert = |blocksize: usize| {
            let codes: Vec<u8> = (0..rows * cols / 2).map(|_| next()).collect();
            let absmax: Vec<u8> = (0..rows * cols / blocksize).flat_map(|b| (0.5 + b as f32 * 0.25).to_le_bytes()).collect();
            Bnb4::new(owned(codes), bnb::NF4_CODE, Absmax::Plain { payload: owned(absmax), dtype: Float::F32 }, blocksize, rows, cols).unwrap()
        };
        let (a, b) = (expert(64), expert(64));
        let stacked = PackedData::Concat { parts: vec![PackedData::Bnb4(a.clone()), PackedData::Bnb4(b.clone())], elements: 2 * rows * cols, part_rows: rows };
        let type_id = stacked.bnb_type().expect("both experts fit the same native type");
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for row in 0..2 * rows {
            stacked.encode_bnb_row(type_id, row, &mut got).unwrap();
        }
        for part in [&a, &b] {
            for row in 0..rows {
                part.encode_row(type_id, row, &mut want).unwrap();
            }
        }
        assert_eq!(got, want);
        // decoding across the expert boundary
        let mut all = vec![0f32; 2 * rows * cols];
        stacked.decode_f32_chunk(0, &mut all).unwrap();
        assert_eq!(all, bnb::dequantize(&got, type_id, 2 * rows * cols).unwrap());
        // experts that need different native types (block 64 / block 128) or another family give no common type
        let wide = expert(128);
        assert_eq!(PackedData::Bnb4(wide.clone()).bnb_type(), Some(bnb::NF4_B128));
        let mixed = PackedData::Concat { parts: vec![PackedData::Bnb4(a), PackedData::Bnb4(wide)], elements: 2 * rows * cols, part_rows: rows };
        assert_eq!(mixed.bnb_type(), None);
        assert!(mixed.is_expanded());
    }

    #[test]
    fn bitsandbytes_int8_rows_keep_their_own_scale() {
        use fllm_gguf_probe::bnb;
        let (rows, cols) = (3usize, 192usize);
        let weight: Vec<u8> = (0..rows * cols).map(|i| (i * 7 % 256) as u8).collect();
        let scb = vec![0.5f32, 2.0, 0.0];
        let d = Bnb8::new(owned(weight), scb.clone(), rows, cols).unwrap();
        assert_eq!(d.native_type(), Some(bnb::INT8_ROW));
        let mut bytes = Vec::new();
        for r in 0..rows {
            d.encode_row(r, &mut bytes).unwrap();
        }
        assert_eq!(bytes.len(), rows * 3 * 68);
        let got = bnb::dequantize(&bytes, bnb::INT8_ROW, rows * cols).unwrap();
        for (i, v) in got.iter().enumerate() {
            let q = (i * 7 % 256) as u8 as i8;
            assert_eq!(*v, f32::from(q) * (scb[i / cols] / 127.0), "element {i}");
        }
        // rows that are not a multiple of 64 have no native layout
        let d = Bnb8::new(owned(vec![1; 2 * 100]), vec![1.0; 2], 2, 100).unwrap();
        assert_eq!(d.native_type(), None);
        assert!(Bnb8::new(owned(vec![1; 2 * 100]), vec![1.0; 3], 2, 100).is_err());
    }

    #[test]
    fn bitsandbytes_tiled_int8_layouts_are_refused() {
        let (tensors, _) = fixture("bnb_int8");
        let name = tensors.iter().find(|t| t.name.ends_with(".SCB")).unwrap().name.replace(".SCB", ".weight_format");
        let tagged = |value: u8| {
            let mut all = tensors.clone();
            all.push(Tensor { name: name.clone(), dtype: "U8".into(), shape: vec![], payload: owned(vec![value]) });
            all
        };
        assert!(split_bnb(tagged(0), &identity).is_ok());
        let err = split_bnb(tagged(2), &identity).unwrap_err();
        assert!(err.contains("tiled GPU layout"), "{err}");
    }

    #[test]
    fn bitsandbytes_errors_name_the_missing_piece() {
        let (mut tensors, _) = fixture("bnb_nf4_bs64");
        tensors.retain(|t| !t.name.ends_with(".absmax"));
        assert!(split_bnb(tensors, &identity).unwrap_err().contains("absmax"));
        let (mut tensors, _) = fixture("bnb_int8");
        tensors.retain(|t| !t.name.ends_with(".SCB"));
        assert!(split_bnb(tensors, &identity).unwrap_err().contains("SCB"));
    }

    /// Reads `n` MXFP4 blocks of a GGUF tensor as f32 (scale `2^(e-127)`, byte j = elements j and j + 16).
    fn decode_mxfp4(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        for block in bytes.chunks(17) {
            let scale = e8m0(block[0]);
            out.extend((0..16).map(|j| E2M1[(block[1 + j] & 0xf) as usize] * scale));
            out.extend((0..16).map(|j| E2M1[(block[1 + j] >> 4) as usize] * scale));
        }
        out
    }

    /// A gpt-oss checkpoint in the published `quant_method: mxfp4` layout converts (it was refused before the `mxfp4`
    /// method was recognized) and the experts decode to what `transformers` dequantizes.
    #[test]
    fn gpt_oss_mxfp4_checkpoint_converts_and_decodes_like_transformers() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("gptoss_mxfp4");
        let out = std::env::temp_dir().join(format!("fllm-gptoss-mxfp4-{}.gguf", std::process::id()));
        let _ = std::fs::remove_file(&out);
        let report = crate::convert_checkpoint(&dir, &out, crate::ConvertOptions::default()).unwrap();
        assert_eq!(report.architecture, "gpt-oss");
        let gguf = fllm_gguf_probe::MappedGguf::open_single(&out).unwrap();
        let expected = crate::safetensors::read_file(&dir.join("expected.safetensors")).unwrap();
        let bf16 = |name: &str| -> Vec<f32> {
            let t = expected.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"));
            f32_values(t).unwrap()
        };
        let (e, h, f) = (2usize, 64usize, 64usize);
        for layer in 0..2 {
            let gate_up = bf16(&format!("model.layers.{layer}.mlp.experts.gate_up_proj")); // [E, H, 2F], gate/up interleaved
            let down = bf16(&format!("model.layers.{layer}.mlp.experts.down_proj")); // [E, F, H]
            for (k, part) in ["ffn_gate_exps", "ffn_up_exps"].iter().enumerate() {
                let got = decode_mxfp4(gguf.tensor_bytes(&format!("blk.{layer}.{part}.weight")).unwrap());
                assert_eq!(got.len(), e * f * h);
                for ex in 0..e {
                    for out_row in 0..f {
                        for inp in 0..h {
                            let want = gate_up[(ex * h + inp) * 2 * f + 2 * out_row + k];
                            assert_eq!(got[(ex * f + out_row) * h + inp], want, "layer {layer} {part} expert {ex} row {out_row} col {inp}");
                        }
                    }
                }
            }
            let got = decode_mxfp4(gguf.tensor_bytes(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap());
            for ex in 0..e {
                for out_row in 0..h {
                    for inp in 0..f {
                        let want = down[(ex * f + inp) * h + out_row];
                        assert_eq!(got[(ex * h + out_row) * f + inp], want, "layer {layer} down expert {ex} row {out_row} col {inp}");
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn compressed_tensors_nvfp4_matches_the_library() {
        let (tensors, expected) = fixture("ct_nvfp4");
        let (dense, packed) = crate::convert::split_packed(tensors, crate::convert::Quantization::Nvfp4, None, &identity).unwrap();
        assert!(dense.is_empty());
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].0, "layer.weight");
        // the library divides the block scale by weight_global_scale; the converter multiplies by its reciprocal
        assert_close("ct_nvfp4", &decode_all(&packed[0]), &expected, 1e-5);
        let crate::packed_quant::PackedData::Nvfp4(t) = &packed[0].1 else { panic!("not NVFP4") };
        assert!(t.input_scale().is_some_and(|v| (v - 3.1 / (448.0 * 6.0)).abs() < 1e-6), "{:?}", t.input_scale());
    }

    #[test]
    fn compressed_tensors_mxfp4_matches_the_library() {
        let (tensors, expected) = fixture("ct_mxfp4");
        let (dense, packed) = split_ct_mxfp4(tensors, &identity).unwrap();
        assert!(dense.is_empty());
        assert_eq!(packed[0].0, "layer.weight");
        assert_close("ct_mxfp4", &decode_all(&packed[0]), &expected, 1e-6);
        // the GGUF rows decode to the same values: scale * code table, byte j = elements j and j + 16
        let crate::packed_quant::PackedData::Mxfp4(m) = &packed[0].1 else { panic!("not MXFP4") };
        for row in 0..m.rows {
            let mut bytes = Vec::new();
            m.encode_gguf_row(row, &mut bytes).unwrap();
            assert_eq!(bytes.len(), m.gguf_row_bytes());
            for (b, block) in bytes.chunks(17).enumerate() {
                let scale = e8m0(block[0]);
                for j in 0..16 {
                    for (half, shift) in [(0usize, 0), (16, 4)] {
                        let got = E2M1[((block[1 + j] >> shift) & 0xf) as usize] * scale;
                        assert_eq!(got, expected[row * m.cols + b * 32 + j + half], "row {row} block {b} element {}", j + half);
                    }
                }
            }
        }
    }
}
