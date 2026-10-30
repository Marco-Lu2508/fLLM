//! Weight formats of the NPU and the conversion from fLLM's quantization formats.
//!
//! What the Intel NPU compiler handles well (OpenVINO GenAI documentation, "OpenVINO GenAI on NPU"): weights compressed to 4 bits
//! *symmetrically*, group-wise (group 128) or channel-wise, or FP16; the computation precision is FP16. This module therefore knows
//! three target formats:
//!
//! * [`NpuFormat::Int4Sym`]: signed 4-bit codes in `[-8, 7]`, one f16 scale per group of `group` values along the input dimension
//!   (`group == cols` is channel-wise); weight = code * scale. Two codes per byte, element `2j` in the low nibble (the packing of
//!   OpenVINO's `i4`).
//! * [`NpuFormat::Int8Sym`]: signed 8-bit codes, same scale layout.
//! * [`NpuFormat::Float16`]: f16 weights without scales.
//!
//! Q4_0 and Q8_0 convert *exactly* (their blocks are symmetric with one f16 scale per 32 values, group 32): `d * (q - 8)` is the code
//! `q - 8` of an i4 with scale `d`. Every other format is dequantized with [`crate::compute::dequantize`] and re-quantized by
//! round-to-nearest into INT4 (formats of up to about 4.5 bits per weight) or INT8 (5 bits and more), which is not exact: the
//! round-trip error of a re-quantized INT4 weight is that of a Q4_0-like format with the chosen group size. K-quants carry asymmetric
//! sub-block minima; their asymmetric 4-bit form (u4 with zero points, which OpenVINO also supports) is not generated because the NPU
//! compiler is documented to prefer symmetric weights.
//!
//! Verification level: L0 (hardware free unit tests compare the conversions with `compute::dequantize`).

use crate::backend::QuantType;
use half::f16;
use rayon::prelude::*;

/// How the weights of one matrix are stored for the NPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuFormat {
    Int4Sym,
    Int8Sym,
    Float16,
}

/// A weight matrix `rows x cols` (`rows` output features, `cols` input features, row-major like fLLM's tensors) in NPU form.
#[derive(Clone, Debug)]
pub struct NpuWeight {
    pub rows: usize,
    pub cols: usize,
    pub format: NpuFormat,
    /// Values per scale along `cols` (`cols` for channel-wise); 0 for [`NpuFormat::Float16`].
    pub group: usize,
    /// Int4: `rows * cols / 2` bytes; Int8: `rows * cols` bytes; Float16: `rows * cols * 2` bytes (little endian).
    pub codes: Vec<u8>,
    /// f16 bits, `rows * groups` entries, row after row; empty for [`NpuFormat::Float16`].
    pub scales: Vec<u16>,
}

impl NpuWeight {
    /// Scales per row.
    pub fn groups(&self) -> usize {
        self.cols.checked_div(self.group).unwrap_or(0)
    }

    /// Bytes the compiled NPU model will hold for this matrix (codes and scales).
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 2
    }

    /// The signed integer codes, one per weight (empty for [`NpuFormat::Float16`]).
    pub fn signed_codes(&self) -> Vec<i8> {
        match self.format {
            NpuFormat::Float16 => Vec::new(),
            NpuFormat::Int8Sym => self.codes.iter().map(|&b| b as i8).collect(),
            NpuFormat::Int4Sym => (0..self.rows * self.cols)
                .map(|e| (((self.codes[e / 2] >> ((e % 2) * 4)) << 4) as i8) >> 4)
                .collect(),
        }
    }

    /// The matrix as f32 (what the NPU computes with, apart from its f16 arithmetic). Reference for tests and for the host fallback.
    pub fn dequantize(&self) -> Vec<f32> {
        let count = self.rows * self.cols;
        let mut out = Vec::with_capacity(count);
        match self.format {
            NpuFormat::Float16 => {
                out.extend(
                    self.codes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16::from_le_bytes(*b).to_f32()),
                );
            }
            NpuFormat::Int8Sym => {
                let groups = self.groups();
                for row in 0..self.rows {
                    for c in 0..self.cols {
                        let scale =
                            f16::from_bits(self.scales[row * groups + c / self.group]).to_f32();
                        out.push(self.codes[row * self.cols + c] as i8 as f32 * scale);
                    }
                }
            }
            NpuFormat::Int4Sym => {
                let groups = self.groups();
                for row in 0..self.rows {
                    for c in 0..self.cols {
                        let e = row * self.cols + c;
                        let code = (((self.codes[e / 2] >> ((e % 2) * 4)) << 4) as i8) >> 4;
                        let scale =
                            f16::from_bits(self.scales[row * groups + c / self.group]).to_f32();
                        out.push(code as f32 * scale);
                    }
                }
            }
        }
        out
    }
}

/// How weights are converted. `group == 0` means channel-wise.
#[derive(Clone, Debug)]
pub struct QuantPolicy {
    /// Group size of re-quantized weights (the NPU documentation recommends 128, or channel-wise for larger models).
    pub group: usize,
    /// Convert Q4_0 / Q8_0 exactly (group 32) instead of re-quantizing them to `group`.
    pub lossless: bool,
    /// Store every integer format as INT8 (the 4-bit formats too). Measured on a Core Ultra 5 235U: INT8 channel-wise weights compile in about
    /// a second and run as fast as INT4, while INT4 in groups of 32 or 128 takes tens of seconds to compile for the large projections.
    pub eight: bool,
}

impl Default for QuantPolicy {
    /// The exact conversion (Q4_0 and Q8_0 in groups of 32, the rest INT4/INT8 in groups of 128): bit-identical weights, slow to compile.
    fn default() -> Self {
        Self {
            group: 128,
            lossless: true,
            eight: false,
        }
    }
}

impl QuantPolicy {
    /// INT8, one scale per output row, for every integer format: the fast default of the offload (see [`QuantPolicy::eight`]). The
    /// error against the source weights is at most 1/254 of the row maximum, small beside the 4-bit rounding the source already has.
    pub fn int8_channel() -> Self {
        Self { group: 0, lossless: false, eight: true }
    }

    /// INT4 in groups of 128 (the NNCF/OpenVINO default): half the NPU memory of INT8; the large projections take seconds to compile.
    pub fn int4_grouped() -> Self {
        Self { group: 128, lossless: false, eight: false }
    }

    /// Short name for logs and the status line.
    pub fn describe(&self) -> String {
        if self.lossless {
            "exact (Q4_0/Q8_0 in groups of 32, others INT4/INT8 in groups of 128)".to_owned()
        } else if self.eight && self.group == 0 {
            "INT8 per channel".to_owned()
        } else if self.group == 0 {
            "INT4/INT8 per channel".to_owned()
        } else if self.eight {
            format!("INT8 in groups of {}", self.group)
        } else {
            format!("INT4/INT8 in groups of {}", self.group)
        }
    }
}

/// The bit width class a format is converted to, `None` for formats the NPU path does not take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Float,
    Four,
    Eight,
}

/// Target class of a weight format; `None` for device-only layouts and formats without a Rust dequantizer.
pub fn target_of(quant: QuantType) -> Option<Target> {
    use QuantType::*;
    match quant {
        F32 | F16 | BF16 => Some(Target::Float),
        Q4_0 | Q4_1 | Q2_K | Q3_K | Q4_K | IQ1_S | IQ1_M | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS
        | IQ3_S | IQ4_NL | IQ4_XS | Mxfp4 | NvFp4 | Q4Z32 | Q4Z64 | Q4Z128 | Q4ZP32 | Q4ZP64
        | Q4ZP128 => Some(Target::Four),
        Q5_0 | Q5_1 | Q5_K | Q6_K | Q8_0 | Q8_1 | Q8_K | Fp8Marlin => Some(Target::Eight),
        _ => None,
    }
}

/// Largest group size not above `preferred` (or the preferred one itself) that divides `cols`; `cols` (channel-wise) when none does.
fn pick_group(cols: usize, preferred: usize) -> usize {
    if preferred == 0 {
        return cols;
    }
    [preferred, 128, 64, 32, 16]
        .into_iter()
        .filter(|&g| g > 0 && g <= preferred && g <= cols)
        .find(|&g| cols.is_multiple_of(g))
        .unwrap_or(cols)
}

fn set_nibble(codes: &mut [u8], element: usize, nibble: u8) {
    let byte = &mut codes[element / 2];
    if element.is_multiple_of(2) {
        *byte = (*byte & 0xF0) | (nibble & 0xF);
    } else {
        *byte = (*byte & 0x0F) | ((nibble & 0xF) << 4);
    }
}

/// Exact conversion of Q4_0: block `{d: f16, qs: [u8; 16]}`, element `j` (< 16) is the low nibble of `qs[j]`, element `j + 16` the
/// high one, value `d * (q - 8)`. The signed code `q - 8` has the nibble `q ^ 8`.
fn repack_q4_0(raw: &[u8], rows: usize, cols: usize) -> Result<NpuWeight, String> {
    if !cols.is_multiple_of(32) || raw.len() != rows * (cols / 32) * 18 {
        return Err("Q4_0 tensor size does not match rows x cols".into());
    }
    let blocks = cols / 32;
    let mut codes = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u16; rows * blocks];
    for row in 0..rows {
        for block in 0..blocks {
            let src = &raw[(row * blocks + block) * 18..][..18];
            scales[row * blocks + block] = u16::from_le_bytes([src[0], src[1]]);
            let first = row * cols + block * 32;
            for j in 0..16 {
                let byte = src[2 + j];
                set_nibble(&mut codes, first + j, (byte & 0xF) ^ 8);
                set_nibble(&mut codes, first + j + 16, (byte >> 4) ^ 8);
            }
        }
    }
    Ok(NpuWeight {
        rows,
        cols,
        format: NpuFormat::Int4Sym,
        group: 32,
        codes,
        scales,
    })
}

/// Exact conversion of Q8_0: block `{d: f16, qs: [i8; 32]}`, value `d * q`.
fn repack_q8_0(raw: &[u8], rows: usize, cols: usize) -> Result<NpuWeight, String> {
    if !cols.is_multiple_of(32) || raw.len() != rows * (cols / 32) * 34 {
        return Err("Q8_0 tensor size does not match rows x cols".into());
    }
    let blocks = cols / 32;
    let mut codes = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows * blocks);
    for block in raw.as_chunks::<34>().0 {
        scales.push(u16::from_le_bytes([block[0], block[1]]));
        codes.extend_from_slice(&block[2..]);
    }
    Ok(NpuWeight {
        rows,
        cols,
        format: NpuFormat::Int8Sym,
        group: 32,
        codes,
        scales,
    })
}

fn to_f16_bytes(values: impl Iterator<Item = f32>, count: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(count * 2);
    for v in values {
        let h = f16::from_f32(v);
        if !h.is_finite() {
            return Err("weight value outside the f16 range".into());
        }
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

fn float_weight(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<NpuWeight, String> {
    let count = rows * cols;
    let codes = match quant {
        QuantType::F16 => {
            if raw.len() != count * 2 {
                return Err("F16 tensor size does not match rows x cols".into());
            }
            // validate the range once: Inf / NaN in the weights would poison every product
            if raw
                .as_chunks::<2>()
                .0
                .iter()
                .any(|b| !f16::from_le_bytes(*b).is_finite())
            {
                return Err("F16 weights contain Inf or NaN".into());
            }
            raw.to_vec()
        }
        QuantType::BF16 => {
            if raw.len() != count * 2 {
                return Err("BF16 tensor size does not match rows x cols".into());
            }
            to_f16_bytes(
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)),
                count,
            )?
        }
        QuantType::F32 => {
            if raw.len() != count * 4 {
                return Err("F32 tensor size does not match rows x cols".into());
            }
            to_f16_bytes(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
                count,
            )?
        }
        other => return Err(format!("{other:?} is not a float format")),
    };
    Ok(NpuWeight {
        rows,
        cols,
        format: NpuFormat::Float16,
        group: 0,
        codes,
        scales: Vec::new(),
    })
}

/// Quantize `n` rows of `cols` values (symmetric, `group` values per scale) into `codes` / `scales` of just those rows.
fn quantize_rows(
    values: &[f32],
    n: usize,
    cols: usize,
    group: usize,
    bits: Target,
    codes: &mut [u8],
    scales: &mut [u16],
) {
    let groups = cols / group;
    for row in 0..n {
        for g in 0..groups {
            let chunk = &values[row * cols + g * group..][..group];
            let first = row * cols + g * group;
            match bits {
                Target::Four => {
                    // like Q4_0: the scale maps the value of largest magnitude to -8, so its sign is kept exact
                    let max =
                        chunk
                            .iter()
                            .copied()
                            .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
                    let scale = f16::from_f32(max / -8.0);
                    scales[row * groups + g] = scale.to_bits();
                    let d = scale.to_f32();
                    let inverse = if d != 0.0 && d.is_finite() {
                        1.0 / d
                    } else {
                        0.0
                    };
                    for (i, &v) in chunk.iter().enumerate() {
                        let q = (v * inverse + 8.5).floor().clamp(0.0, 15.0) as u8;
                        set_nibble(codes, first + i, q ^ 8);
                    }
                }
                _ => {
                    let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let scale = f16::from_f32(amax / 127.0);
                    scales[row * groups + g] = scale.to_bits();
                    let d = scale.to_f32();
                    let inverse = if d != 0.0 && d.is_finite() {
                        1.0 / d
                    } else {
                        0.0
                    };
                    for (i, &v) in chunk.iter().enumerate() {
                        codes[first + i] = (v * inverse).round().clamp(-127.0, 127.0) as i8 as u8;
                    }
                }
            }
        }
    }
}

/// Bytes of one row for formats whose rows can be dequantized independently (block formats without a trailing table).
fn independent_row_bytes(quant: QuantType, cols: usize) -> Option<usize> {
    if quant.is_q4zp() || quant == QuantType::Fp8Marlin {
        return None;
    }
    let block = quant.block_size();
    (block > 0 && cols.is_multiple_of(block)).then(|| cols / block * quant.block_bytes())
}

/// Convert a weight matrix from fLLM's device format into the NPU form according to `policy`.
///
/// `raw` is the tensor exactly as `Backend::upload_weight` receives it; `rows x cols` is `out_features x in_features`.
pub fn convert(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    policy: &QuantPolicy,
) -> Result<NpuWeight, String> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() {
        return Err("empty weight matrix".into());
    }
    let target = target_of(quant).ok_or_else(|| format!("{quant:?} has no NPU conversion"))?;
    if target == Target::Float {
        return float_weight(quant, raw, rows, cols);
    }
    if policy.lossless {
        match quant {
            QuantType::Q4_0 => return repack_q4_0(raw, rows, cols),
            QuantType::Q8_0 => return repack_q8_0(raw, rows, cols),
            _ => {}
        }
    }
    // 4-bit codes pack two per byte along a row, so an odd row length takes the 8-bit form
    let target = if target == Target::Four && (policy.eight || !cols.is_multiple_of(2)) {
        Target::Eight
    } else {
        target
    };
    let group = pick_group(cols, policy.group);
    let groups = cols / group;
    let code_row_bytes = if target == Target::Four {
        cols / 2
    } else {
        cols
    };
    let kind = quant.to_gguf_type_id();
    let mut codes = vec![0u8; rows * code_row_bytes];
    let mut scales = vec![0u16; rows * groups];
    match independent_row_bytes(quant, cols) {
        Some(row_bytes) => {
            if raw.len() != rows * row_bytes {
                return Err(format!(
                    "{quant:?} tensor has {} bytes, expected {}",
                    raw.len(),
                    rows * row_bytes
                ));
            }
            const CHUNK_ROWS: usize = 8;
            codes
                .par_chunks_mut(code_row_bytes * CHUNK_ROWS)
                .zip(scales.par_chunks_mut(groups * CHUNK_ROWS))
                .enumerate()
                .try_for_each(|(index, (code_chunk, scale_chunk))| -> Result<(), String> {
                    let first = index * CHUNK_ROWS;
                    let n = code_chunk.len() / code_row_bytes;
                    let values = crate::compute::dequantize(
                        &raw[first * row_bytes..(first + n) * row_bytes],
                        kind,
                        n * cols,
                    )?;
                    quantize_rows(&values, n, cols, group, target, code_chunk, scale_chunk);
                    Ok(())
                })?;
        }
        None => {
            let values = crate::compute::dequantize(raw, kind, rows * cols)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err("dequantized weights contain Inf or NaN".into());
            }
            quantize_rows(&values, rows, cols, group, target, &mut codes, &mut scales);
        }
    }
    let format = if target == Target::Four {
        NpuFormat::Int4Sym
    } else {
        NpuFormat::Int8Sym
    };
    Ok(NpuWeight {
        rows,
        cols,
        format,
        group,
        codes,
        scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator (no external crates).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next() as f32 / (1u32 << 31) as f32 * 2.0 - 1.0
        }
    }

    fn q4_0_blocks(rng: &mut Lcg, blocks: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for _ in 0..blocks {
            raw.extend_from_slice(&f16::from_f32(rng.unit() * 0.1 + 0.02).to_le_bytes());
            raw.extend((0..16).map(|_| rng.next() as u8));
        }
        raw
    }

    fn q8_0_blocks(rng: &mut Lcg, blocks: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for _ in 0..blocks {
            raw.extend_from_slice(&f16::from_f32(rng.unit() * 0.05 + 0.01).to_le_bytes());
            raw.extend((0..32).map(|_| rng.next() as u8));
        }
        raw
    }

    #[test]
    fn q4_0_converts_exactly() {
        let mut rng = Lcg(1);
        let (rows, cols) = (5, 96);
        let raw = q4_0_blocks(&mut rng, rows * cols / 32);
        let w = convert(QuantType::Q4_0, &raw, rows, cols, &QuantPolicy::default()).unwrap();
        assert_eq!((w.format, w.group, w.groups()), (NpuFormat::Int4Sym, 32, 3));
        let want = crate::compute::dequantize(&raw, 2, rows * cols).unwrap();
        assert_eq!(w.dequantize(), want, "Q4_0 -> i4 + f16 scale is bit exact");
        assert_eq!(w.codes.len(), rows * cols / 2);
    }

    #[test]
    fn q8_0_converts_exactly() {
        let mut rng = Lcg(2);
        let (rows, cols) = (3, 64);
        let raw = q8_0_blocks(&mut rng, rows * cols / 32);
        let w = convert(QuantType::Q8_0, &raw, rows, cols, &QuantPolicy::default()).unwrap();
        assert_eq!((w.format, w.group), (NpuFormat::Int8Sym, 32));
        assert_eq!(
            w.dequantize(),
            crate::compute::dequantize(&raw, 8, rows * cols).unwrap()
        );
    }

    #[test]
    fn the_fast_default_stores_four_bit_sources_as_int8_per_channel() {
        let mut rng = Lcg(11);
        let (rows, cols) = (6, 256);
        let raw = q4_0_blocks(&mut rng, rows * cols / 32);
        let w = convert(QuantType::Q4_0, &raw, rows, cols, &QuantPolicy::int8_channel()).unwrap();
        assert_eq!((w.format, w.group, w.groups()), (NpuFormat::Int8Sym, cols, 1));
        assert_eq!(w.codes.len(), rows * cols);
        let want = crate::compute::dequantize(&raw, 2, rows * cols).unwrap();
        let got = w.dequantize();
        for r in 0..rows {
            let max = want[r * cols..(r + 1) * cols].iter().fold(0f32, |m, v| m.max(v.abs()));
            // one step of the 8-bit grid is max / 127; the scale is stored as f16, which adds a relative 2^-11
            let bound = max / 254.0 + max * 1e-3;
            for c in 0..cols {
                assert!((got[r * cols + c] - want[r * cols + c]).abs() <= bound, "row {r} col {c}");
            }
        }
        let policy = QuantPolicy::int4_grouped();
        assert!(!policy.eight && policy.group == 128);
        assert_eq!(QuantPolicy::int8_channel().describe(), "INT8 per channel");
    }

    #[test]
    fn requantizing_q4_0_to_group_128_stays_close() {
        let mut rng = Lcg(3);
        let (rows, cols) = (4, 256);
        let raw = q4_0_blocks(&mut rng, rows * cols / 32);
        let policy = QuantPolicy {
            group: 128,
            lossless: false,
            eight: false,
        };
        let w = convert(QuantType::Q4_0, &raw, rows, cols, &policy).unwrap();
        assert_eq!(
            (w.format, w.group, w.groups()),
            (NpuFormat::Int4Sym, 128, 2)
        );
        let want = crate::compute::dequantize(&raw, 2, rows * cols).unwrap();
        let got = w.dequantize();
        let amax = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // like Q4_0, the scale maps the extreme value to -8, so values of the opposite sign near the extreme clamp at +7: the error is at most one step (amax / 8)
        let worst = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= amax / 8.0 * 1.05,
            "worst error {worst} vs amax {amax}"
        );
    }

    #[test]
    fn k_quant_requantizes_to_int4_and_q6_k_to_int8() {
        // Q4_K block: d, dmin, 12 scale bytes, 128 code bytes (144 bytes); Q6_K: 210 bytes.
        let mut rng = Lcg(4);
        let mut q4k = Vec::new();
        for _ in 0..2 {
            q4k.extend_from_slice(&f16::from_f32(0.01).to_le_bytes());
            q4k.extend_from_slice(&f16::from_f32(0.005).to_le_bytes());
            q4k.extend((0..12).map(|_| (rng.next() as u8 & 0x3F) | 1));
            q4k.extend((0..128).map(|_| rng.next() as u8));
        }
        let w = convert(QuantType::Q4_K, &q4k, 2, 256, &QuantPolicy::default()).unwrap();
        assert_eq!((w.format, w.group), (NpuFormat::Int4Sym, 128));
        let want = crate::compute::dequantize(&q4k, 12, 512).unwrap();
        let got = w.dequantize();
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let diff: Vec<f32> = want.iter().zip(&got).map(|(a, b)| a - b).collect();
        assert!(
            rms(&diff) < 0.2 * rms(&want),
            "int4 requantization error {} vs signal {}",
            rms(&diff),
            rms(&want)
        );

        let mut q6k = Vec::new();
        for _ in 0..2 {
            q6k.extend((0..192).map(|_| rng.next() as u8));
            q6k.extend((0..16).map(|_| (rng.next() % 40) as u8));
            q6k.extend_from_slice(&f16::from_f32(0.004).to_le_bytes());
        }
        let w = convert(QuantType::Q6_K, &q6k, 2, 256, &QuantPolicy::default()).unwrap();
        assert_eq!(w.format, NpuFormat::Int8Sym);
        let want = crate::compute::dequantize(&q6k, 14, 512).unwrap();
        let diff: Vec<f32> = want
            .iter()
            .zip(w.dequantize())
            .map(|(a, b)| a - b)
            .collect();
        assert!(
            rms(&diff) < 0.02 * rms(&want),
            "int8 requantization error {} vs signal {}",
            rms(&diff),
            rms(&want)
        );
    }

    #[test]
    fn float_weights_become_f16_and_overflow_is_rejected() {
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w = convert(QuantType::F32, &raw, 3, 4, &QuantPolicy::default()).unwrap();
        assert_eq!(w.format, NpuFormat::Float16);
        assert_eq!(w.dequantize(), values);
        let big: Vec<u8> = [1.0e9f32; 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(convert(QuantType::F32, &big, 2, 2, &QuantPolicy::default()).is_err());
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(
            convert(QuantType::BF16, &bf16, 3, 4, &QuantPolicy::default())
                .unwrap()
                .dequantize(),
            values
        );
    }

    #[test]
    fn group_selection_and_odd_widths() {
        assert_eq!(pick_group(4096, 128), 128);
        assert_eq!(pick_group(96, 128), 32, "96 is not a multiple of 128 or 64");
        assert_eq!(pick_group(4096, 0), 4096, "channel-wise");
        assert_eq!(pick_group(100, 128), 100);
        assert_eq!(pick_group(77, 32), 77);
        // an odd row length cannot hold 4-bit codes row by row: IQ4_NL needs 32-wide blocks, so use Q4_1 style size checks instead
        let mut rng = Lcg(5);
        let raw = q4_0_blocks(&mut rng, 3);
        assert!(convert(QuantType::Q4_0, &raw, 1, 95, &QuantPolicy::default()).is_err());
    }

    #[test]
    fn bad_sizes_and_unsupported_formats_are_errors() {
        assert!(convert(QuantType::Q4_0, &[0u8; 17], 1, 32, &QuantPolicy::default()).is_err());
        assert!(convert(QuantType::Q8_0, &[0u8; 33], 1, 32, &QuantPolicy::default()).is_err());
        assert!(convert(
            QuantType::Nvfp4Marlin,
            &[0u8; 8],
            1,
            16,
            &QuantPolicy::default()
        )
        .is_err());
        assert!(convert(QuantType::F16, &[0u8; 8], 0, 4, &QuantPolicy::default()).is_err());
        assert!(target_of(QuantType::Other(99)).is_none());
    }

    #[test]
    fn nibble_order_is_low_first() {
        let mut codes = [0u8; 2];
        set_nibble(&mut codes, 0, 0x1);
        set_nibble(&mut codes, 1, 0x2);
        set_nibble(&mut codes, 2, 0x3);
        set_nibble(&mut codes, 3, 0xF);
        assert_eq!(codes, [0x21, 0xF3]);
    }

    #[test]
    fn four_bit_requantizer_keeps_the_extreme_value() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let mut codes = vec![0u8; 16];
        let mut scales = vec![0u16; 1];
        quantize_rows(&values, 1, 32, 32, Target::Four, &mut codes, &mut scales);
        let w = NpuWeight {
            rows: 1,
            cols: 32,
            format: NpuFormat::Int4Sym,
            group: 32,
            codes,
            scales,
        };
        let back = w.dequantize();
        // the largest magnitude (index 31, value 1.9) is mapped to code -8 with a negative scale... it round-trips closely
        assert!(
            (back[31] - values[31]).abs() < 0.02,
            "{} vs {}",
            back[31],
            values[31]
        );
        let step = 1.9 / 8.0;
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() <= step * 0.51 + 0.005, "{a} vs {b}");
        }
    }
}
