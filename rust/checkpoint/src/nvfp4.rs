//! NVIDIA ModelOpt NVFP4 linear weights (`quant_algo: NVFP4`) and the GGUF type 40 blocks the engine loads.
//!
//! A quantized linear `P` has these tensors in the checkpoint:
//!
//! | tensor | dtype / shape | meaning |
//! | --- | --- | --- |
//! | `P.weight` | `U8 [out, in/2]` | two E2M1 codes per byte, the even input index in the low nibble |
//! | `P.weight_scale` | `F8_E4M3 [out, in/16]` | one block scale per 16 inputs |
//! | `P.weight_scale_2` | `F32` scalar | per-tensor factor |
//! | `P.input_scale` | `F32` scalar | static activation scale (written as `<name>.input_scale`; the CUDA backend quantizes activations with it, see FLLM_NVFP4_ACT_SCALE) |
//!
//! `value = e2m1(code) * e4m3(block scale) * weight_scale_2` (ModelOpt's dequantization).
//!
//! GGUF type 40 (llama.cpp `NVFP4`) stores 64 values as 36 bytes: four UE4M3 block scales, then four groups of
//! 8 bytes; in a group, byte `j` holds value `j` in the low nibble and value `j + 8` in the high nibble. The block
//! scale bytes are copied unchanged (E4M3 with a zero sign bit is UE4M3); the per-tensor factor goes into the F32
//! tensor `<name>.scale`. Rows are independent in this layout, so the Q/K row permutation of the Llama family
//! moves whole rows (codes and scales together).

use crate::safetensors::Payload;

/// Values of the E2M1 magnitude codes 0..8.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Number of inputs that share one block scale in the checkpoint.
pub const SCALE_GROUP: usize = 16;
/// Number of values in one GGUF type 40 block.
pub const GGUF_BLOCK: usize = 64;
/// Bytes of one GGUF type 40 block.
#[cfg(test)]
pub const GGUF_BLOCK_BYTES: usize = 36;

/// Value of a non-negative E4M3 (float8_e4m3fn) byte.
pub fn e4m3_value(byte: u8) -> f32 {
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 7) as f32;
    if exp == 0 {
        mant * 2f32.powi(-9)
    } else {
        (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Value of one E2M1 code (sign in bit 3).
pub fn e2m1_value(code: u8) -> f32 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 {
        -v
    } else {
        v
    }
}

#[derive(Clone, Debug)]
pub struct Nvfp4 {
    weight: Payload,
    scales: Payload,
    rows: usize,
    cols: usize,
    global: f32,
    input_scale: Option<f32>,
}

impl Nvfp4 {
    /// `weight` holds `rows * cols / 2` bytes, `scales` holds `rows * cols / 16` E4M3 bytes.
    pub fn new(
        weight: Payload,
        scales: Payload,
        rows: usize,
        cols: usize,
        global: f32,
        input_scale: Option<f32>,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || !cols.is_multiple_of(SCALE_GROUP) {
            return Err(format!(
                "NVFP4 weight [{rows}, {cols}]: the input dimension must be a nonzero multiple of {SCALE_GROUP}"
            ));
        }
        let codes = rows.checked_mul(cols / 2).ok_or("NVFP4 size overflow")?;
        let blocks = rows
            .checked_mul(cols / SCALE_GROUP)
            .ok_or("NVFP4 size overflow")?;
        if weight.as_slice().len() != codes {
            return Err(format!(
                "NVFP4 weight has {} bytes, expected {codes}",
                weight.as_slice().len()
            ));
        }
        if scales.as_slice().len() != blocks {
            return Err(format!(
                "NVFP4 weight_scale has {} bytes, expected {blocks}",
                scales.as_slice().len()
            ));
        }
        if !global.is_finite() || global < 0.0 {
            return Err(format!(
                "NVFP4 weight_scale_2 must be finite and not negative, found {global}"
            ));
        }
        if let Some(i) = scales
            .as_slice()
            .iter()
            .position(|&b| b & 0x80 != 0 || b == 0x7f)
        {
            return Err(format!(
                "NVFP4 weight_scale byte {i} is {:#04x}: block scales must be finite and not negative",
                scales.as_slice()[i]
            ));
        }
        Ok(Self {
            weight,
            scales,
            rows,
            cols,
            global,
            input_scale,
        })
    }

    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }
    /// The per-tensor factor (`weight_scale_2`).
    pub fn global_scale(&self) -> f32 {
        self.global
    }
    pub fn input_scale(&self) -> Option<f32> {
        self.input_scale
    }
    /// Whether the rows fit whole GGUF type 40 blocks.
    pub fn fits_gguf_blocks(&self) -> bool {
        self.cols.is_multiple_of(GGUF_BLOCK)
    }
    /// Bytes of one GGUF type 40 row.
    #[cfg(test)]
    pub fn gguf_row_bytes(&self) -> usize {
        self.cols / GGUF_BLOCK * GGUF_BLOCK_BYTES
    }

    fn code(&self, row: usize, col: usize) -> u8 {
        let byte = self.weight.as_slice()[row * (self.cols / 2) + col / 2];
        (byte >> (4 * (col & 1))) & 0x0f
    }

    /// Append the GGUF type 40 blocks of one row (without the per-tensor factor).
    pub fn encode_gguf_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        if row >= self.rows {
            return Err(format!("NVFP4 row {row} out of range"));
        }
        if !self.fits_gguf_blocks() {
            return Err("NVFP4 rows need a multiple of 64 inputs for GGUF type 40".into());
        }
        let scales = self.scales.as_slice();
        let per_row = self.cols / SCALE_GROUP;
        for block in 0..self.cols / GGUF_BLOCK {
            let first = row * per_row + block * 4;
            dst.extend_from_slice(&scales[first..first + 4]);
            for group in 0..4 {
                let base = block * GGUF_BLOCK + group * SCALE_GROUP;
                for j in 0..8 {
                    dst.push(self.code(row, base + j) | (self.code(row, base + j + 8) << 4));
                }
            }
        }
        Ok(())
    }

    /// Dequantized values (`code * block scale * weight_scale_2`) of the flat range `start..start + dst.len()`
    /// of the row-major `[rows, cols]` matrix.
    pub fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start
            .checked_add(dst.len())
            .filter(|&e| e <= self.element_count())
            .ok_or("NVFP4 decode range exceeds the tensor")?;
        let scales = self.scales.as_slice();
        for (slot, flat) in dst.iter_mut().zip(start..end) {
            let (row, col) = (flat / self.cols, flat % self.cols);
            let scale = e4m3_value(scales[row * (self.cols / SCALE_GROUP) + col / SCALE_GROUP]);
            *slot = e2m1_value(self.code(row, col)) * scale * self.global;
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

    /// Decode GGUF type 40 blocks the way the engine does (`quant_extra.rs`): doubled E2M1 table times half the scale.
    fn engine_decode(bytes: &[u8]) -> Vec<f32> {
        const FP4: [f32; 16] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0,
            -12.0,
        ];
        let mut out = Vec::new();
        for b in bytes.as_chunks::<GGUF_BLOCK_BYTES>().0 {
            for sub in 0..4 {
                let d = e4m3_value(b[sub]) * 0.5;
                let qs = &b[4 + sub * 8..4 + (sub + 1) * 8];
                out.extend((0..8).map(|j| FP4[(qs[j] & 15) as usize] * d));
                out.extend((0..8).map(|j| FP4[(qs[j] >> 4) as usize] * d));
            }
        }
        out
    }

    fn pseudo(seed: &mut u32) -> u8 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 24) as u8
    }

    fn sample(rows: usize, cols: usize, global: f32) -> Nvfp4 {
        let mut seed = 7u32;
        let weight: Vec<u8> = (0..rows * cols / 2).map(|_| pseudo(&mut seed)).collect();
        // positive finite E4M3 scales: exponent 0..=14, any mantissa
        let scales: Vec<u8> = (0..rows * cols / 16)
            .map(|_| (pseudo(&mut seed) % 0x78).max(1))
            .collect();
        Nvfp4::new(owned(weight), owned(scales), rows, cols, global, Some(0.5)).unwrap()
    }

    #[test]
    fn e4m3_and_e2m1_known_values() {
        assert_eq!(e4m3_value(0x38), 1.0);
        assert_eq!(e4m3_value(0x7e), 448.0);
        assert_eq!(e4m3_value(0x01), 2f32.powi(-9));
        assert_eq!(e4m3_value(0x00), 0.0);
        let all: Vec<f32> = (0..16).map(e2m1_value).collect();
        assert_eq!(
            all,
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );
    }

    #[test]
    fn gguf_rows_decode_to_the_modelopt_dequantization() {
        let (rows, cols, global) = (6usize, 128usize, 0.0173f32);
        let t = sample(rows, cols, global);
        let mut reference = vec![0f32; rows * cols];
        t.decode_chunk(0, &mut reference).unwrap();
        for row in 0..rows {
            let mut bytes = Vec::new();
            t.encode_gguf_row(row, &mut bytes).unwrap();
            assert_eq!(bytes.len(), t.gguf_row_bytes());
            let got = engine_decode(&bytes);
            for (c, (g, r)) in got.iter().zip(&reference[row * cols..]).enumerate() {
                // the engine multiplies the per-tensor factor outside the blocks
                assert_eq!(g * global, *r, "row {row} col {c}");
            }
        }
    }

    #[test]
    fn scales_are_copied_and_nibbles_are_regrouped() {
        // one row of 64 inputs, codes 0..15 repeating, scales 1..4
        let codes: Vec<u8> = (0..64).map(|i| (i % 16) as u8).collect();
        let weight: Vec<u8> = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
        let scales = vec![0x38, 0x39, 0x3a, 0x3b];
        let t = Nvfp4::new(owned(weight), owned(scales), 1, 64, 1.0, None).unwrap();
        let mut out = Vec::new();
        t.encode_gguf_row(0, &mut out).unwrap();
        assert_eq!(&out[..4], &[0x38, 0x39, 0x3a, 0x3b]);
        // group 0: byte j = code j | code j+8 << 4
        for j in 0..8 {
            assert_eq!(out[4 + j], (j as u8) | ((j as u8 + 8) << 4));
        }
    }

    #[test]
    fn invalid_scales_and_shapes_are_rejected() {
        let w = owned(vec![0; 32]);
        assert!(Nvfp4::new(w.clone(), owned(vec![0x80, 0x38, 0x38, 0x38]), 1, 64, 1.0, None)
            .unwrap_err()
            .contains("block scales"));
        assert!(Nvfp4::new(w.clone(), owned(vec![0x7f, 0x38, 0x38, 0x38]), 1, 64, 1.0, None).is_err());
        assert!(Nvfp4::new(w.clone(), owned(vec![0x38; 3]), 1, 64, 1.0, None).is_err());
        assert!(Nvfp4::new(w.clone(), owned(vec![0x38; 4]), 1, 64, f32::NAN, None).is_err());
        assert!(Nvfp4::new(owned(vec![0; 10]), owned(vec![0x38; 2]), 1, 20, 1.0, None).is_err());
        // 48 inputs: valid for the checkpoint, not for GGUF blocks
        let t = Nvfp4::new(owned(vec![0; 24]), owned(vec![0x38; 3]), 1, 48, 1.0, None).unwrap();
        assert!(!t.fits_gguf_blocks());
        assert!(t.encode_gguf_row(0, &mut Vec::new()).is_err());
    }
}
