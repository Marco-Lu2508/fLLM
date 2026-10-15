//! Symmetric integer weights of `compressed-tensors` checkpoints and block-scaled FP8 weights.
//!
//! * `pack-quantized` (`weight_packed`, `weight_scale`): 4- or 8-bit codes stored with an offset of `2^(bits-1)`,
//!   `32 / bits` codes per little-endian int32 along the input dimension, so the byte layout is the plain nibble /
//!   byte order (even input index in the low nibble). `value = (code - offset) * scale`.
//! * `int-quantized` (`weight` as int8, `weight_scale`): `value = code * scale`.
//!
//! Scales are `[rows, groups]` (group-wise), `[rows, 1]` (per channel) or a single value (per tensor).
//!
//! The 4-bit codes are written unchanged as native `Q4Z` rows (zero point 8), the 8-bit codes as `Q8_0` blocks; both
//! decode to exactly `value` up to the rounding of the scale to f16.

use crate::safetensors::Payload;
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymScale {
    F16,
    Bf16,
    F32,
}

impl SymScale {
    pub fn width(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
    pub fn from_dtype(dtype: &str) -> Option<Self> {
        match dtype {
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymCodes {
    /// 4-bit codes, two per byte, offset 8 (`weight_packed`).
    Packed4,
    /// 8-bit codes with offset 128 (`weight_packed` with 8 bits).
    Packed8,
    /// Signed int8 codes (`weight` of `int-quantized`).
    Int8,
}

#[derive(Clone, Debug)]
pub struct SymInt {
    codes: Payload,
    kind: SymCodes,
    scales: Payload,
    scale_type: SymScale,
    rows: usize,
    cols: usize,
    /// Inputs per scale (`cols` for per-channel and per-tensor scales).
    group: usize,
    per_tensor: bool,
}

fn scale_at(bytes: &[u8], kind: SymScale, index: usize) -> f32 {
    let at = index * kind.width();
    match kind {
        SymScale::F16 => f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32(),
        SymScale::Bf16 => f32::from_bits((u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32) << 16),
        SymScale::F32 => f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]),
    }
}

impl SymInt {
    /// `group` is the number of inputs per scale; pass `cols` for per-channel scales.
    pub fn new(
        codes: Payload,
        kind: SymCodes,
        scales: Payload,
        scale_type: SymScale,
        rows: usize,
        cols: usize,
        group: usize,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || group == 0 || !cols.is_multiple_of(group) {
            return Err(format!(
                "integer weight [{rows}, {cols}] with group size {group}: the input dimension must be a multiple of the group size"
            ));
        }
        let (per_code_bits, align) = match kind {
            SymCodes::Packed4 => (4, 8),
            SymCodes::Packed8 => (8, 4),
            SymCodes::Int8 => (8, 1),
        };
        if !cols.is_multiple_of(align) {
            return Err(format!("integer weight input dimension {cols} must be a multiple of {align}"));
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or("integer weight size overflow")?
            * per_code_bits
            / 8;
        if codes.as_slice().len() != expected {
            return Err(format!("integer weight has {} bytes, expected {expected}", codes.as_slice().len()));
        }
        let groups = cols / group;
        let count = scales.as_slice().len() / scale_type.width();
        if !scales.as_slice().len().is_multiple_of(scale_type.width()) {
            return Err("weight_scale length is not a whole number of scalars".into());
        }
        let per_tensor = count == 1 && rows * groups != 1;
        if count != rows * groups && !per_tensor {
            return Err(format!(
                "weight_scale has {count} values, expected {} ([{rows}, {groups}]) or 1",
                rows * groups
            ));
        }
        Ok(Self {
            codes,
            kind,
            scales,
            scale_type,
            rows,
            cols,
            group,
            per_tensor,
        })
    }

    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn scale(&self, row: usize, col: usize) -> f32 {
        let index = if self.per_tensor {
            0
        } else {
            row * (self.cols / self.group) + col / self.group
        };
        scale_at(self.scales.as_slice(), self.scale_type, index)
    }

    /// Signed code value of one weight.
    fn code(&self, row: usize, col: usize) -> i32 {
        let bytes = self.codes.as_slice();
        match self.kind {
            SymCodes::Packed4 => {
                let byte = bytes[row * (self.cols / 2) + col / 2];
                ((byte >> (4 * (col & 1))) & 0x0f) as i32 - 8
            }
            SymCodes::Packed8 => bytes[row * self.cols + col] as i32 - 128,
            SymCodes::Int8 => bytes[row * self.cols + col] as i8 as i32,
        }
    }

    pub fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start
            .checked_add(dst.len())
            .filter(|&e| e <= self.element_count())
            .ok_or("integer weight decode range exceeds the tensor")?;
        for (slot, flat) in dst.iter_mut().zip(start..end) {
            let (row, col) = (flat / self.cols, flat % self.cols);
            *slot = self.code(row, col) as f32 * self.scale(row, col);
        }
        Ok(())
    }

    /// Group size when the weight can be written as native `Q4Z` rows.
    pub fn q4z_group(&self) -> Option<usize> {
        (self.kind == SymCodes::Packed4 && matches!(self.group, 32 | 64 | 128)).then_some(self.group)
    }

    /// Append the `Q4Z` blocks of one row: `group / 2` code bytes, the f16 scale and the zero point 8.
    pub fn encode_q4z_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        let g = self.q4z_group().ok_or("weight cannot be stored as Q4Z")?;
        if row >= self.rows {
            return Err(format!("row {row} out of range"));
        }
        let bytes = self.codes.as_slice();
        for block in 0..self.cols / g {
            let first = row * (self.cols / 2) + block * g / 2;
            dst.extend_from_slice(&bytes[first..first + g / 2]);
            dst.extend_from_slice(&f16::from_f32(self.scale(row, block * g)).to_bits().to_le_bytes());
            dst.push(8);
        }
        Ok(())
    }

    /// Whether the 8-bit codes fit `Q8_0` blocks (32 inputs share one scale).
    pub fn q8_0_ok(&self) -> bool {
        self.kind != SymCodes::Packed4 && self.cols.is_multiple_of(32) && self.group.is_multiple_of(32)
    }

    /// Append the `Q8_0` blocks of one row: f16 scale, 32 signed codes.
    pub fn encode_q8_0_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        if !self.q8_0_ok() {
            return Err("weight cannot be stored as Q8_0".into());
        }
        if row >= self.rows {
            return Err(format!("row {row} out of range"));
        }
        for block in 0..self.cols / 32 {
            dst.extend_from_slice(&f16::from_f32(self.scale(row, block * 32)).to_bits().to_le_bytes());
            dst.extend((0..32).map(|i| self.code(row, block * 32 + i) as i8 as u8));
        }
        Ok(())
    }
}

/// FP8 (E4M3) weights with one scale per `[block_rows, block_cols]` tile (`weight_scale_inv` of block-wise FP8
/// checkpoints). The scale multiplies the decoded FP8 value.
#[derive(Clone, Debug)]
pub struct Fp8Block {
    weight: Payload,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
}

impl Fp8Block {
    pub fn new(
        weight: Payload,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Result<Self, String> {
        if block_rows == 0 || block_cols == 0 || weight.as_slice().len() != rows * cols {
            return Err("FP8 block weight has an inconsistent shape".into());
        }
        if scales.len() != rows.div_ceil(block_rows) * cols.div_ceil(block_cols) {
            return Err(format!(
                "FP8 block scales: {} values for a [{rows}, {cols}] weight with {block_rows}x{block_cols} blocks",
                scales.len()
            ));
        }
        Ok(Self {
            weight,
            scales,
            rows,
            cols,
            block_rows,
            block_cols,
        })
    }

    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    pub fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start
            .checked_add(dst.len())
            .filter(|&e| e <= self.element_count())
            .ok_or("FP8 block decode range exceeds the tensor")?;
        let tiles = self.cols.div_ceil(self.block_cols);
        for (slot, flat) in dst.iter_mut().zip(start..end) {
            let (row, col) = (flat / self.cols, flat % self.cols);
            let scale = self.scales[row / self.block_rows * tiles + col / self.block_cols];
            *slot = crate::packed_quant::e4m3fn_to_f32(self.weight.as_slice()[flat]) * scale;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn owned(data: Vec<u8>) -> Payload {
        let len = data.len();
        Payload::Owned(Arc::new(data), 0..len)
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect()
    }

    fn pseudo_bytes(n: usize, mut seed: u32) -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn int4_group_scales_decode_and_become_q4z_rows() {
        let (rows, cols, group) = (3usize, 64usize, 32usize);
        let scales: Vec<f32> = (0..rows * cols / group).map(|i| 0.01 * (1 + i) as f32).collect();
        let t = SymInt::new(
            owned(pseudo_bytes(rows * cols / 2, 5)),
            SymCodes::Packed4,
            owned(f16_bytes(&scales)),
            SymScale::F16,
            rows,
            cols,
            group,
        )
        .unwrap();
        let mut dense = vec![0f32; rows * cols];
        t.decode_chunk(0, &mut dense).unwrap();
        assert_eq!(t.q4z_group(), Some(group));
        for row in 0..rows {
            let mut blocks = Vec::new();
            t.encode_q4z_row(row, &mut blocks).unwrap();
            let mut got = Vec::new();
            for b in blocks.chunks_exact(group / 2 + 3) {
                let scale = f16::from_le_bytes([b[group / 2], b[group / 2 + 1]]).to_f32();
                let zero = (b[group / 2 + 2] & 15) as f32;
                for &q in &b[..group / 2] {
                    got.push(((q & 15) as f32 - zero) * scale);
                    got.push(((q >> 4) as f32 - zero) * scale);
                }
            }
            assert_eq!(got, dense[row * cols..(row + 1) * cols], "row {row}");
        }
    }

    #[test]
    fn int8_per_channel_and_packed8_become_q8_0_rows() {
        let (rows, cols) = (4usize, 64usize);
        let scales = f16_bytes(&[0.5, 0.25, 0.125, 2.0]);
        let raw = pseudo_bytes(rows * cols, 11);
        for kind in [SymCodes::Int8, SymCodes::Packed8] {
            let t = SymInt::new(owned(raw.clone()), kind, owned(scales.clone()), SymScale::F16, rows, cols, cols).unwrap();
            assert!(t.q8_0_ok());
            let mut dense = vec![0f32; rows * cols];
            t.decode_chunk(0, &mut dense).unwrap();
            for row in 0..rows {
                let mut bytes = Vec::new();
                t.encode_q8_0_row(row, &mut bytes).unwrap();
                let mut got = Vec::new();
                for b in bytes.as_chunks::<34>().0 {
                    let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                    got.extend(b[2..].iter().map(|&q| q as i8 as f32 * d));
                }
                assert_eq!(got, dense[row * cols..(row + 1) * cols], "{kind:?} row {row}");
            }
        }
        // per-tensor scale and unsupported shapes
        let t = SymInt::new(owned(raw.clone()), SymCodes::Int8, owned(f16_bytes(&[0.5])), SymScale::F16, rows, cols, cols).unwrap();
        let mut one = [0f32; 1];
        t.decode_chunk(70, &mut one).unwrap();
        assert_eq!(one[0], raw[70] as i8 as f32 * 0.5);
        assert!(SymInt::new(owned(raw.clone()), SymCodes::Int8, owned(f16_bytes(&[1.0, 2.0])), SymScale::F16, rows, cols, cols).is_err());
        assert!(SymInt::new(owned(raw[1..].to_vec()), SymCodes::Int8, owned(f16_bytes(&[1.0])), SymScale::F16, rows, cols, cols).is_err());
    }

    #[test]
    fn bf16_and_f32_scales_are_read() {
        let bf = |v: f32| ((v.to_bits() >> 16) as u16).to_le_bytes();
        let t = SymInt::new(
            owned(vec![0x98; 4]),
            SymCodes::Packed4,
            owned([bf(2.0), bf(4.0)].concat()),
            SymScale::Bf16,
            1,
            8,
            4,
        )
        .unwrap();
        let mut out = [0f32; 8];
        t.decode_chunk(0, &mut out).unwrap();
        // nibbles 8 and 9 -> codes 0 and 1
        assert_eq!(out, [0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 0.0, 4.0]);
        let t = SymInt::new(owned(vec![0x98; 4]), SymCodes::Packed4, owned(1.5f32.to_le_bytes().to_vec()), SymScale::F32, 1, 8, 8).unwrap();
        t.decode_chunk(0, &mut out).unwrap();
        assert_eq!(out[1], 1.5);
    }

    #[test]
    fn fp8_blocks_multiply_the_tile_scale() {
        // 3x4 weight, 2x2 tiles -> 2x2 scales
        let weight = vec![0x38u8; 12]; // 1.0
        let t = Fp8Block::new(owned(weight), vec![1.0, 2.0, 3.0, 4.0], 3, 4, 2, 2).unwrap();
        let mut out = vec![0f32; 12];
        t.decode_chunk(0, &mut out).unwrap();
        assert_eq!(out, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        assert!(Fp8Block::new(owned(vec![0; 12]), vec![1.0; 3], 3, 4, 2, 2).is_err());
    }
}
