//! Reference decoders and matrix-vector products for the 4- and 8-bit floating-point checkpoint
//! formats (NVFP4, FP8 E4M3 and E5M2).

/// Positive magnitudes of the FP4 E2M1 code, indexed by the low three bits of a nibble.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// FP8 E4M3 (finite-only variant, bias 7, no infinities, NaN = S.1111.111).
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((byte >> 3) & 0xf);
    let mantissa = f32::from(byte & 0x7);
    if exponent == 0xf && byte & 0x7 == 0x7 {
        return f32::NAN;
    }
    if exponent == 0 {
        return sign * mantissa / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
}

/// FP8 E5M2 (IEEE-like, bias 15, with infinities and NaN).
pub fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((byte >> 2) & 0x1f);
    let mantissa = f32::from(byte & 0x3);
    if exponent == 0x1f {
        return if byte & 0x3 == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if exponent == 0 {
        return sign * mantissa / 4.0 * 2f32.powi(-14);
    }
    sign * (1.0 + mantissa / 4.0) * 2f32.powi(exponent - 15)
}

fn fp4_to_f32(nibble: u8) -> f32 {
    let magnitude = E2M1[usize::from(nibble & 0x7)];
    if nibble & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// NVFP4: sixteen E2M1 values per block share an E4M3 scale; one more per-tensor scale applies to
/// the whole tensor. Two values per byte, the first element in the low nibble.
pub fn dequantize_nvfp4(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    dst: &mut [f32],
) -> Result<(), String> {
    let count = dst.len();
    if count == 0
        || !count.is_multiple_of(16)
        || packed.len() < count / 2
        || scales.len() < count / 16
    {
        return Err("invalid buffer sizes for NVFP4 dequantization".into());
    }
    for (block, out) in dst.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let scale = fp8_e4m3_to_f32(scales[block]) * global_scale;
        for (pair, values) in packed[block * 8..block * 8 + 8]
            .iter()
            .zip(out.as_chunks_mut::<2>().0.iter_mut())
        {
            values[0] = fp4_to_f32(pair & 0xf) * scale;
            values[1] = fp4_to_f32(pair >> 4) * scale;
        }
    }
    Ok(())
}

pub fn gemv_nvfp4(
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if !cols.is_multiple_of(16)
        || x.len() < cols
        || y.len() < rows
        || packed.len() < rows * (cols / 2)
        || scales.len() < rows * (cols / 16)
    {
        return Err("dimension mismatch for NVFP4 GEMV".into());
    }
    let mut row = vec![0.0f32; cols];
    for (r, out) in y.iter_mut().take(rows).enumerate() {
        dequantize_nvfp4(
            &packed[r * (cols / 2)..(r + 1) * (cols / 2)],
            &scales[r * (cols / 16)..(r + 1) * (cols / 16)],
            global_scale,
            &mut row,
        )?;
        *out = row.iter().zip(x).map(|(w, x)| w * x).sum();
    }
    Ok(())
}

pub fn dequantize_fp8_e4m3(src: &[u8], scale: f32, dst: &mut [f32]) -> Result<(), String> {
    if src.is_empty() || dst.len() < src.len() {
        return Err("invalid buffer sizes for FP8 E4M3 dequantization".into());
    }
    for (out, &byte) in dst.iter_mut().zip(src) {
        *out = fp8_e4m3_to_f32(byte) * scale;
    }
    Ok(())
}

pub fn dequantize_fp8_e5m2(src: &[u8], scale: f32, dst: &mut [f32]) -> Result<(), String> {
    if src.is_empty() || dst.len() < src.len() {
        return Err("invalid buffer sizes for FP8 E5M2 dequantization".into());
    }
    for (out, &byte) in dst.iter_mut().zip(src) {
        *out = fp8_e5m2_to_f32(byte) * scale;
    }
    Ok(())
}

fn gemv_fp8(
    decode: fn(u8) -> f32,
    x: &[f32],
    weights: &[u8],
    scale: f32,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if x.len() < cols || y.len() < rows || weights.len() < rows * cols {
        return Err("dimension mismatch for FP8 GEMV".into());
    }
    for (r, out) in y.iter_mut().take(rows).enumerate() {
        let sum: f32 = weights[r * cols..(r + 1) * cols]
            .iter()
            .zip(x)
            .map(|(&w, x)| decode(w) * x)
            .sum();
        *out = sum * scale;
    }
    Ok(())
}

pub fn gemv_fp8_e4m3(
    x: &[f32],
    weights: &[u8],
    scale: f32,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    gemv_fp8(fp8_e4m3_to_f32, x, weights, scale, y, rows, cols)
}

pub fn gemv_fp8_e5m2(
    x: &[f32],
    weights: &[u8],
    scale: f32,
    y: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    gemv_fp8(fp8_e5m2_to_f32, x, weights, scale, y, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_special_values() {
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xb8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x7e), 448.0);
        assert!(fp8_e4m3_to_f32(0x7f).is_nan());
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e5m2_to_f32(0x3c), 1.0);
        assert_eq!(fp8_e5m2_to_f32(0x7b), 57344.0);
        assert!(fp8_e5m2_to_f32(0x7c).is_infinite());
        assert!(fp8_e5m2_to_f32(0x7d).is_nan());
    }

    #[test]
    fn nvfp4_block_scales_apply_per_sixteen_values() {
        let mut packed = [0u8; 16];
        packed[0] = 0x71; // +0.5, +6.0
        packed[8] = 0x92; // +1.0, -0.5
        let scales = [0x38, 0x40]; // 1.0 and 2.0
        let mut out = [0.0f32; 32];
        dequantize_nvfp4(&packed, &scales, 0.5, &mut out).unwrap();
        assert_eq!(&out[..2], &[0.25, 3.0]);
        assert_eq!(&out[16..18], &[1.0, -0.5]);
        assert!(dequantize_nvfp4(&packed, &scales, 1.0, &mut [0.0; 15]).is_err());
    }
}
