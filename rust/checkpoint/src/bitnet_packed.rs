//! Hugging Face `quant_method: "bitnet"` checkpoints (microsoft/bitnet-b1.58-2B-4T): every linear weight is a `uint8` matrix
//! `[out / 4, in]` holding four ternary codes per byte plus a one-element `weight_scale`. The reader expands them to the dense
//! matrix `(code - 1) * scale`, so the file is written like any other unquantized checkpoint (F16, or `--quant`).
//!
//! Packing (transformers `integrations/bitnet.py`, `unpack_weights`): code `i` (bits `2i..2i+2`) of packed row `p` is the weight
//! of output row `i * out / 4 + p`; the stored values are `-1, 0, 1` plus one. The reference implementation also rounds the
//! activations of every BitLinear to 8 bits per token; this reader does not (see docs/model-coverage.md).
use crate::dequant::Dequant;
use crate::safetensors::{Payload, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const PER_BYTE: usize = 4;

#[derive(Debug)]
pub(crate) struct BitnetTernary {
    packed: Payload,
    scale: f32,
    rows: usize,
    cols: usize,
}

impl BitnetTernary {
    pub(crate) fn new(packed: Payload, scale: f32, rows: usize, cols: usize) -> Result<Self, String> {
        if rows == 0 || cols == 0 || !rows.is_multiple_of(PER_BYTE) || packed.as_slice().len() != rows / PER_BYTE * cols {
            return Err(format!("packed ternary weight has {} bytes, expected {} x {cols}", packed.as_slice().len(), rows / PER_BYTE));
        }
        if !scale.is_finite() {
            return Err("non-finite weight_scale".into());
        }
        Ok(Self { packed, scale, rows, cols })
    }
}

impl Dequant for BitnetTernary {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.rows * self.cols).ok_or("range outside the ternary weight")?;
        let bytes = self.packed.as_slice();
        let packed_rows = self.rows / PER_BYTE;
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let (row, col) = (e / self.cols, e % self.cols);
            let (shift, p) = (row / packed_rows, row % packed_rows);
            let code = (bytes[p * self.cols + col] >> (2 * shift)) & 3;
            *slot = (f32::from(code) - 1.0) * self.scale;
        }
        Ok(())
    }
}

fn scalar(t: &Tensor) -> Result<f32, String> {
    let b = t.payload.as_slice();
    let v = match t.dtype.as_str() {
        "F32" if b.len() == 4 => f32::from_le_bytes(b.try_into().unwrap()),
        "F16" if b.len() == 2 => half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32(),
        "BF16" if b.len() == 2 => f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16),
        _ => return Err(format!("{}: expected a one-element float scale, found {} {:?}", t.name, t.dtype, t.shape)),
    };
    Ok(v)
}

/// Splits the packed ternary weights out of a checkpoint: the remaining dense tensors and the weights as expanded readers.
pub(crate) fn split_bitnet(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    let mut weights = Vec::new();
    let mut dense = Vec::new();
    for t in all {
        if let Some(base) = t.name.strip_suffix(".weight_scale") {
            scales.insert(base.to_owned(), t);
        } else if t.dtype == "U8" && t.name.ends_with(".weight") && t.shape.len() == 2 {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let scale = scales.remove(&base).ok_or_else(|| format!("packed ternary weight {} has no {base}.weight_scale", w.name))?;
        let (rows, cols) = (w.shape[0] as usize * PER_BYTE, w.shape[1] as usize);
        let t = BitnetTernary::new(w.payload.clone(), scalar(&scale)?, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&w.name)?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(t)), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}.weight_scale has no packed ternary weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares BitNet but no uint8 packed weights were found".into());
    }
    Ok((dense, packed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packing of transformers' `pack_weights`, written out for an 8 x 3 matrix of codes -1, 0 and 1.
    #[test]
    fn unpacks_four_output_rows_per_byte() {
        let (rows, cols) = (8usize, 3usize);
        let weights: Vec<i8> = (0..rows * cols).map(|i| (i % 3) as i8 - 1).collect();
        let packed_rows = rows / PER_BYTE;
        let mut bytes = vec![0u8; packed_rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let code = (weights[r * cols + c] + 1) as u8;
                bytes[(r % packed_rows) * cols + c] |= code << (2 * (r / packed_rows));
            }
        }
        let t = BitnetTernary::new(Payload::Owned(Arc::new(bytes.clone()), 0..bytes.len()), 0.5, rows, cols).unwrap();
        let mut out = vec![0f32; rows * cols];
        for (i, chunk) in out.chunks_mut(5).enumerate() {
            t.decode_chunk(i * 5, chunk).unwrap();
        }
        let want: Vec<f32> = weights.iter().map(|&w| f32::from(w) * 0.5).collect();
        assert_eq!(out, want);
        assert!(BitnetTernary::new(Payload::Owned(Arc::new(vec![0; 5]), 0..5), 1.0, 8, 3).is_err());
    }
}
