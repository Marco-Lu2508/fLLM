//! Pure Rust dequantizers for the GGML IQ formats used by GGUF tensors.
//!
//! The layouts and decoding rules below follow ggml-quants.c and
//! ggml-common.h from llama.cpp commit
//! b1ff4ca23630ac7c5a275405353ddb5c486332c9. GGML is MIT licensed; see
//! https://github.com/ggerganov/llama.cpp/blob/b1ff4ca23630ac7c5a275405353ddb5c486332c9/LICENSE.

#[path = "quant_iq_tables.rs"]
mod tables;

const QK_K: usize = 256;

fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

// Convert IEEE-754 binary16 without pulling a runtime dependency into this
// small decoder. The result rounds exactly as a normal f16-to-f32 conversion.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let value = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut mantissa = frac;
            let mut exponent = 113u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(value)
}

fn packed_bytes_u64(word: u64, index: usize) -> u8 {
    ((word >> (index * 8)) & 0xff) as u8
}

fn sign(signs: u8, lane: usize) -> f32 {
    if signs & tables::KMASK_IQ2XS[lane] != 0 {
        -1.0
    } else {
        1.0
    }
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// Dequantize one of the published GGML IQ block formats.
///
/// `kind` is the GGML type ID. IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL,
/// IQ3_S, IQ2_S, IQ4_XS, and IQ1_M are supported (IDs 16-23 and 29).
/// `count` is the number of output elements. Inputs must contain exactly the
/// complete blocks needed for that count.
pub fn dequantize_iq(bytes: &[u8], kind: u32, count: usize) -> Result<Vec<f32>, String> {
    let (block_len, block_values) = match kind {
        16 => (66, QK_K),  // IQ2_XXS
        17 => (74, QK_K),  // IQ2_XS
        18 => (98, QK_K),  // IQ3_XXS
        19 => (50, QK_K),  // IQ1_S
        20 => (18, 32),    // IQ4_NL
        21 => (110, QK_K), // IQ3_S
        22 => (82, QK_K),  // IQ2_S
        23 => (136, QK_K), // IQ4_XS
        29 => (56, QK_K),  // IQ1_M
        _ => return Err(format!("unsupported GGML IQ type {kind}")),
    };
    if !count.is_multiple_of(block_values) {
        return Err("IQ tensor element count must contain complete blocks".into());
    }
    let blocks = count / block_values;
    let expected_bytes = blocks
        .checked_mul(block_len)
        .ok_or_else(|| "IQ tensor byte size overflow".to_string())?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "invalid IQ tensor size: got {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|e| format!("cannot allocate IQ output: {e}"))?;

    for b in bytes.chunks_exact(block_len) {
        let d = half(b, 0);
        match kind {
            16 => {
                for ib32 in 0..8 {
                    let at = 2 + ib32 * 8;
                    let lo = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
                    let hi = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap());
                    let db = d * (0.5 + (hi >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid = tables::IQ2XXS_GRID[((lo >> (8 * l)) & 0xff) as usize];
                        let signs = tables::KSIGNS_IQ2XS[((hi >> (7 * l)) & 0x7f) as usize];
                        for j in 0..8 {
                            out.push(db * packed_bytes_u64(grid, j) as f32 * sign(signs, j));
                        }
                    }
                }
            }
            17 => {
                for ib32 in 0..8 {
                    let scale = b[66 + ib32];
                    let db = [
                        d * (0.5 + (scale & 0xf) as f32) * 0.25,
                        d * (0.5 + (scale >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4 {
                        let q = read_u16(b, 2 + ib32 * 8 + l * 2);
                        let grid = tables::IQ2XS_GRID[(q & 0x01ff) as usize];
                        let signs = tables::KSIGNS_IQ2XS[(q >> 9) as usize];
                        for j in 0..8 {
                            out.push(db[l / 2] * packed_bytes_u64(grid, j) as f32 * sign(signs, j));
                        }
                    }
                }
            }
            18 => {
                for ib32 in 0..8 {
                    let at = 2 + 64 + ib32 * 4;
                    let scale_signs = u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
                    let db = d * (0.5 + (scale_signs >> 28) as f32) * 0.5;
                    for l in 0..4 {
                        let signs =
                            tables::KSIGNS_IQ2XS[((scale_signs >> (7 * l)) & 0x7f) as usize];
                        let ix = 2 + ib32 * 8 + l * 2;
                        let grids = [
                            tables::IQ3XXS_GRID[b[ix] as usize],
                            tables::IQ3XXS_GRID[b[ix + 1] as usize],
                        ];
                        for j in 0..4 {
                            out.push(db * ((grids[0] >> (8 * j)) & 0xff) as f32 * sign(signs, j));
                        }
                        for j in 0..4 {
                            out.push(
                                db * ((grids[1] >> (8 * j)) & 0xff) as f32 * sign(signs, j + 4),
                            );
                        }
                    }
                }
            }
            19 => {
                for ib32 in 0..8 {
                    let qh = read_u16(b, 34 + ib32 * 2);
                    let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
                    for l in 0..4 {
                        let index =
                            b[2 + ib32 * 4 + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                        let grid = tables::IQ1S_GRID[index];
                        for j in 0..8 {
                            out.push(dl * (packed_bytes_u64(grid, j) as i8 as f32 + delta));
                        }
                    }
                }
            }
            20 => {
                for j in 0..16 {
                    let q = b[2 + j];
                    out.push(d * tables::KVALUES_IQ4NL[(q & 0xf) as usize] as f32);
                }
                for j in 0..16 {
                    let q = b[2 + j];
                    out.push(d * tables::KVALUES_IQ4NL[(q >> 4) as usize] as f32);
                }
            }
            21 => {
                for pair in 0..4 {
                    let db = [
                        d * (1 + 2 * (b[106 + pair] & 0xf) as i32) as f32,
                        d * (1 + 2 * (b[106 + pair] >> 4) as i32) as f32,
                    ];
                    for (half_group, &scale) in db.iter().enumerate() {
                        let sub = pair * 2 + half_group;
                        let qh = b[66 + sub];
                        for l in 0..4 {
                            let signs = b[74 + sub * 4 + l];
                            let q0 = b[2 + sub * 8 + l * 2] as usize
                                | (((qh >> (2 * l)) as usize & 1) << 8);
                            let q1 = b[2 + sub * 8 + l * 2 + 1] as usize
                                | (((qh >> (2 * l + 1)) as usize & 1) << 8);
                            let grids = [tables::IQ3S_GRID[q0], tables::IQ3S_GRID[q1]];
                            for j in 0..4 {
                                out.push(
                                    scale * ((grids[0] >> (8 * j)) & 0xff) as f32 * sign(signs, j),
                                );
                            }
                            for j in 0..4 {
                                out.push(
                                    scale
                                        * ((grids[1] >> (8 * j)) & 0xff) as f32
                                        * sign(signs, j + 4),
                                );
                            }
                        }
                    }
                }
            }
            22 => {
                for ib32 in 0..8 {
                    let scale = b[74 + ib32];
                    let db = [
                        d * (0.5 + (scale & 0xf) as f32) * 0.25,
                        d * (0.5 + (scale >> 4) as f32) * 0.25,
                    ];
                    let qh = b[66 + ib32];
                    for l in 0..4 {
                        let q = b[2 + ib32 * 4 + l];
                        let index = q as usize | ((((qh >> (2 * l)) & 3) as usize) << 8);
                        let signs = b[2 + 32 + ib32 * 4 + l];
                        let grid = tables::IQ2S_GRID[index];
                        for j in 0..8 {
                            out.push(db[l / 2] * packed_bytes_u64(grid, j) as f32 * sign(signs, j));
                        }
                    }
                }
            }
            23 => {
                let scales_h = read_u16(b, 2);
                for ib32 in 0..8 {
                    let low = (b[4 + ib32 / 2] >> (4 * (ib32 % 2))) & 0xf;
                    let ls = low as i32 | ((((scales_h >> (2 * ib32)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    for j in 0..16 {
                        let q = b[8 + ib32 * 16 + j];
                        out.push(dl * tables::KVALUES_IQ4NL[(q & 0xf) as usize] as f32);
                    }
                    for j in 0..16 {
                        let q = b[8 + ib32 * 16 + j];
                        out.push(dl * tables::KVALUES_IQ4NL[(q >> 4) as usize] as f32);
                    }
                }
            }
            29 => {
                let mut sc = [0u16; 4];
                for (i, scale) in sc.iter_mut().enumerate() {
                    *scale = read_u16(b, 48 + i * 2);
                }
                let scale_bits = (sc[0] >> 12)
                    | ((sc[1] >> 8) & 0x00f0)
                    | ((sc[2] >> 4) & 0x0f00)
                    | (sc[3] & 0xf000);
                let d = f16_to_f32(scale_bits);
                for ib32 in 0..8 {
                    let shift = 6 * (ib32 % 2);
                    let dl1 = d * (2 * ((sc[ib32 / 2] >> shift) & 7) + 1) as f32;
                    let dl2 = d * (2 * ((sc[ib32 / 2] >> (shift + 3)) & 7) + 1) as f32;
                    for l in 0..4 {
                        let qh = b[32 + ib32 * 2 + l / 2];
                        let high = if l % 2 == 0 { qh & 7 } else { (qh >> 4) & 7 };
                        let index = b[ib32 * 4 + l] as usize | ((high as usize) << 8);
                        let delta = if qh & (if l % 2 == 0 { 0x08 } else { 0x80 }) != 0 {
                            -0.125
                        } else {
                            0.125
                        };
                        let grid = tables::IQ1S_GRID[index];
                        let scale = if l < 2 { dl1 } else { dl2 };
                        for j in 0..8 {
                            out.push(scale * (packed_bytes_u64(grid, j) as i8 as f32 + delta));
                        }
                    }
                }
            }
            _ => unreachable!("kind was validated"),
        }
    }
    debug_assert_eq!(out.len(), count);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::dequantize_iq;

    fn block(kind: u32, len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len];
        if kind == 29 {
            // IQ1_M stores its binary16 super-block scale in scale nibbles.
            b[48..56].copy_from_slice(&[0, 0, 0, 0, 0, 0xc0, 0, 0x30]);
        } else {
            b[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        }
        b
    }

    #[test]
    fn deterministic_vectors_cover_each_iq_layout() {
        let mut b = block(16, 66);
        b[2 + 4..2 + 8].copy_from_slice(&(1u32 | (2u32 << 28)).to_le_bytes());
        let x = dequantize_iq(&b, 16, 256).unwrap();
        assert_eq!(&x[..8], &[-5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -5.0]);

        let mut b = block(17, 74);
        b[2..4].copy_from_slice(&(1u16 << 9).to_le_bytes());
        b[66] = 0x21;
        let x = dequantize_iq(&b, 17, 256).unwrap();
        assert_eq!(&x[..8], &[-3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, -3.0]);
        assert_eq!(x[16], 5.0);

        let mut b = block(18, 98);
        b[2 + 64..2 + 68].copy_from_slice(&(1u32 | (1u32 << 28)).to_le_bytes());
        let x = dequantize_iq(&b, 18, 256).unwrap();
        assert_eq!(&x[..8], &[-3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, -3.0]);

        let mut b = block(19, 50);
        b[2] = 1;
        b[34..36].copy_from_slice(&0x8000u16.to_le_bytes());
        let x = dequantize_iq(&b, 19, 256).unwrap();
        assert_eq!(
            &x[..8],
            &[0.875, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125]
        );

        let mut b = block(20, 18);
        b[2] = 0xf0;
        let x = dequantize_iq(&b, 20, 32).unwrap();
        assert_eq!(x[0], -127.0);
        assert_eq!(x[16], 113.0);

        let mut b = block(21, 110);
        b[74] = 1;
        let x = dequantize_iq(&b, 21, 256).unwrap();
        assert_eq!(&x[..8], &[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut b = block(22, 82);
        b[2 + 32] = 1;
        b[74] = 0x21;
        let x = dequantize_iq(&b, 22, 256).unwrap();
        assert_eq!(&x[..8], &[-3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);

        let mut b = block(23, 136);
        b[2..4].copy_from_slice(&1u16.to_le_bytes());
        b[8] = 0x80;
        let x = dequantize_iq(&b, 23, 256).unwrap();
        assert_eq!(x[0], 2032.0);
        assert_eq!(x[16], -16.0);

        let mut b = block(29, 56);
        b[32] = 0x08;
        let x = dequantize_iq(&b, 29, 256).unwrap();
        assert_eq!(&x[..8], &[-1.125; 8]);
    }

    #[test]
    fn rejects_incomplete_or_mismatched_iq_input() {
        assert!(dequantize_iq(&[], 16, 255).is_err());
        assert!(dequantize_iq(&[0; 65], 16, 256).is_err());
        assert!(dequantize_iq(&[0; 66], 20, 32).is_err());
        assert!(dequantize_iq(&[], 15, 0).is_err());
    }
}
