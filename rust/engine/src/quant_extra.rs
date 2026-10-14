//! Additional GGML scalar and quantized tensor decoders.
//!
//! The block layouts and dequantization equations in this module are adapted
//! from the ggml authors' MIT-licensed llama.cpp implementation, pinned here
//! to commit b1ff4ca23630ac7c5a275405353ddb5c486332c9. The newer Q2_0
//! layout is also present in upstream master at 58cb9138e46180d45854cf45d4c1e25ad5f7e1c4.
//! See the ggml project LICENSE for the complete MIT notice (Copyright (c) 2023-2026
//! The ggml authors).

use half::f16;

fn half(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn checked_size(
    bytes: &[u8],
    kind: u32,
    count: usize,
    block: usize,
    width: usize,
) -> Result<(), String> {
    if !count.is_multiple_of(block) {
        return Err(format!(
            "GGUF type {kind} element count is not a multiple of block size {block}"
        ));
    }
    let expected = (count / block)
        .checked_mul(width)
        .ok_or("quantized tensor byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "invalid GGUF type {kind} tensor size: got {} bytes, expected {expected}",
            bytes.len()
        ));
    }
    Ok(())
}

/// Decode additional non-IQ GGML types into f32 values.
///
/// Supported ids: 9, 10, 11, 15, 24–28, 34, 35, 39–42. Returned values may
/// be non-finite when the encoded source represents or overflows to one; callers
/// decide whether that is acceptable for their use.
pub fn dequantize_extra(bytes: &[u8], kind: u32, count: usize) -> Result<Vec<f32>, String> {
    let (block, width) = match kind {
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        15 => (256, 292),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 | 28 => (1, 8),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        _ => return Err(format!("no extra Rust decoder for GGUF type {kind}")),
    };
    checked_size(bytes, kind, count, block, width)?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|e| format!("cannot allocate quantized output: {e}"))?;

    match kind {
        24 => out.extend(bytes.iter().map(|&x| (x as i8) as f32)),
        25 => out.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32),
        ),
        26 => out.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b) as f32),
        ),
        27 => out.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as f32),
        ),
        28 => out.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32),
        ),
        9 => {
            for b in bytes.as_chunks::<36>().0.iter() {
                let d = half(b);
                out.extend(b[4..].iter().map(|&q| d * (q as i8) as f32));
            }
        }
        10 => {
            for b in bytes.as_chunks::<84>().0.iter() {
                let d = half(&b[80..]);
                let dmin = half(&b[82..]);
                let mut qi = 0;
                let mut si = 0;
                for n in 0..2 {
                    let q = &b[16 + n * 32..16 + (n + 1) * 32];
                    let mut shift = 0;
                    for _ in 0..4 {
                        for half_group in 0..2 {
                            let scale = b[si];
                            si += 1;
                            let ds = d * (scale & 15) as f32;
                            let dm = dmin * (scale >> 4) as f32;
                            for l in 0..16 {
                                out.push(ds * ((q[half_group * 16 + l] >> shift) & 3) as f32 - dm);
                            }
                        }
                        shift += 2;
                    }
                    qi += 128;
                }
                debug_assert_eq!(qi, 256);
            }
        }
        11 => {
            for b in bytes.as_chunks::<110>().0.iter() {
                let d = half(&b[108..]);
                let mut scales = [0i32; 16];
                for i in 0..4 {
                    let hi = b[96 + i];
                    let lo = b[96 + 4 + i];
                    let packed = b[96 + 8 + i];
                    scales[i] = ((hi & 15) | ((packed & 3) << 4)) as i32 - 32;
                    scales[i + 4] = ((lo & 15) | (((packed >> 2) & 3) << 4)) as i32 - 32;
                    scales[i + 8] = ((hi >> 4) | (((packed >> 4) & 3) << 4)) as i32 - 32;
                    scales[i + 12] = ((lo >> 4) | (((packed >> 6) & 3) << 4)) as i32 - 32;
                }
                let q = &b[32..96];
                let hm = &b[..32];
                let mut scale_idx = 0;
                let mut mask = 1u8;
                for outer in 0..2 {
                    for sub in 0..4 {
                        let shift = sub * 2;
                        for side in 0..2 {
                            let ds = d * scales[scale_idx] as f32;
                            scale_idx += 1;
                            for l in 0..16 {
                                let ix = side * 16 + l;
                                let low = ((q[outer * 32 + ix] >> shift) & 3) as i32;
                                let high = if hm[ix] & mask != 0 { 0 } else { 4 };
                                out.push(ds * (low - high) as f32);
                            }
                        }
                        mask <<= 1;
                    }
                }
            }
        }
        15 => {
            for b in bytes.as_chunks::<292>().0.iter() {
                let d = f32::from_le_bytes(b[..4].try_into().unwrap());
                out.extend(b[4..260].iter().map(|&q| d * (q as i8) as f32));
            }
        }
        34 => {
            let pow3 = [1u16, 3, 9, 27, 81];
            for b in bytes.as_chunks::<54>().0.iter() {
                let d = half(&b[52..]);
                let qs = &b[..48];
                let qh = &b[48..52];
                for chunk in [&qs[..32], &qs[32..]] {
                    for &p in &pow3 {
                        for &qbyte in chunk {
                            let q = (qbyte as u16 * p) as u8;
                            let xi = ((q as u16 * 3) >> 8) as i32;
                            out.push((xi - 1) as f32 * d);
                        }
                    }
                }
                for &p in &pow3[..4] {
                    for &qbyte in qh {
                        let q = (qbyte as u16 * p) as u8;
                        let xi = ((q as u16 * 3) >> 8) as i32;
                        out.push((xi - 1) as f32 * d);
                    }
                }
            }
        }
        35 => {
            for b in bytes.as_chunks::<66>().0.iter() {
                let d = half(&b[64..]);
                for chunk in b[..64].as_chunks::<32>().0.iter() {
                    for shift in (0..8).step_by(2) {
                        out.extend(
                            chunk
                                .iter()
                                .map(|&q| (((q >> shift) & 3) as i8 - 1) as f32 * d),
                        );
                    }
                }
            }
        }
        39 => {
            for b in bytes.as_chunks::<17>().0.iter() {
                let d = e8m0_half(b[0]);
                for j in 0..16 {
                    let q = b[1 + j];
                    out.push(FP4[(q & 15) as usize] * d);
                }
                for j in 0..16 {
                    let q = b[1 + j];
                    out.push(FP4[(q >> 4) as usize] * d);
                }
            }
        }
        40 => {
            for b in bytes.as_chunks::<36>().0.iter() {
                for sub in 0..4 {
                    let d = ue4m3_half(b[sub]);
                    let qs = &b[4 + sub * 8..4 + (sub + 1) * 8];
                    for j in 0..8 {
                        out.push(FP4[(qs[j] & 15) as usize] * d);
                    }
                    for j in 0..8 {
                        out.push(FP4[(qs[j] >> 4) as usize] * d);
                    }
                }
            }
        }
        41 => {
            for b in bytes.as_chunks::<18>().0.iter() {
                let d = half(b);
                for i in 0..128 {
                    out.push(if b[2 + i / 8] & (1 << (i % 8)) != 0 {
                        d
                    } else {
                        -d
                    });
                }
            }
        }
        42 => {
            for b in bytes.as_chunks::<18>().0.iter() {
                let d = half(b);
                for i in 0..64 {
                    let q = (b[2 + i / 4] >> ((i % 4) * 2)) & 3;
                    out.push((q as i32 - 1) as f32 * d);
                }
            }
        }
        _ => unreachable!(),
    }
    debug_assert_eq!(out.len(), count);
    Ok(out)
}

const FP4: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

fn e8m0_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        ((x as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xf) as i32;
    let mantissa = (x & 7) as f32;
    let raw = if exp == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

#[cfg(test)]
mod tests {
    use super::dequantize_extra;
    use half::f16;

    #[test]
    fn known_scalar_and_block_vectors() {
        assert_eq!(dequantize_extra(&[0xff, 0x02], 24, 2).unwrap(), [-1.0, 2.0]);
        assert_eq!(
            dequantize_extra(&[0, 0, 0x03, 0x00], 25, 2).unwrap(),
            [0.0, 3.0]
        );

        let mut q8_1 = vec![0u8; 36];
        q8_1[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        q8_1[4] = 2;
        q8_1[5] = 0xfe;
        let values = dequantize_extra(&q8_1, 9, 32).unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -1.0);
        assert!(values[2..].iter().all(|&x| x == 0.0));

        let mut q1 = vec![0u8; 18];
        q1[..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        q1[2] = 1;
        let values = dequantize_extra(&q1, 41, 128).unwrap();
        assert_eq!(
            &values[..9],
            &[2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
        );
        assert!(values[9..].iter().all(|&x| x == -2.0));

        let mut mxfp4 = vec![0u8; 17];
        mxfp4[0] = 127; // E8M0 scale of one.
        mxfp4[1] = 0x98; // +0 and -0.5 (low nibble first).
        let values = dequantize_extra(&mxfp4, 39, 32).unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[16], -0.5);
        assert_eq!(values[17], 0.0);

        let mut q2k = vec![0u8; 84];
        q2k[0] = 1; // First 16-value scale.
        q2k[1] = 1; // Second 16-value scale.
        q2k[16] = 3;
        q2k[32] = 1;
        q2k[80..82].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let values = dequantize_extra(&q2k, 10, 256).unwrap();
        assert_eq!(values[0], 3.0);
        assert_eq!(values[16], 1.0);
        assert!(values[1..16].iter().all(|&x| x == 0.0));
        assert!(values[17..].iter().all(|&x| x == 0.0));

        let mut q8k = vec![0u8; 292];
        q8k[..4].copy_from_slice(&0.25f32.to_le_bytes());
        q8k[4] = 4;
        q8k[5] = 0xfc;
        let values = dequantize_extra(&q8k, 15, 256).unwrap();
        assert_eq!(&values[..2], &[1.0, -1.0]);
        assert!(values[2..].iter().all(|&x| x == 0.0));

        let tq1 = vec![0u8; 54];
        assert!(dequantize_extra(&tq1, 34, 256)
            .unwrap()
            .iter()
            .all(|&x| x == 0.0));

        let mut tq2 = vec![0u8; 66];
        tq2[64..].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        assert!(dequantize_extra(&tq2, 35, 256)
            .unwrap()
            .iter()
            .all(|&x| x == -1.0));

        let mut nvfp4 = vec![0u8; 36];
        nvfp4[0] = 0x38; // UE4M3 scale maps to 0.5.
        nvfp4[4] = 0x02; // low nibble is doubled E2M1 value 2.
        let values = dequantize_extra(&nvfp4, 40, 64).unwrap();
        assert_eq!(values[0], 1.0);
        assert!(values[1..16].iter().all(|&x| x == 0.0));
        assert!(values[16..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn rejects_bad_block_counts_and_truncated_data() {
        assert!(dequantize_extra(&[0; 4], 9, 31).is_err());
        assert!(dequantize_extra(&[0; 35], 9, 32).is_err());
        assert!(dequantize_extra(&[0; 3], 26, 1).is_err());
        assert!(dequantize_extra(&[], 10, usize::MAX).is_err());
        assert!(dequantize_extra(&[], 16, 256).is_err());
    }
}
