//! Reference arithmetic for the 8-bit and 4-bit floating-point formats of the data-center GPUs, on the CPU.
//!
//! * OCP E4M3 (NVIDIA GPUs, gfx950 / MI355X matrix cores, most FP8 checkpoints) and FNUZ E4M3 (gfx942 / MI300 matrix cores): decode, encode
//!   with round-to-nearest-even and saturation, and the conversion between the two (a checkpoint stored as OCP has to be re-encoded to be
//!   read by the matrix cores of an MI300).
//! * E8M0 block scales and E2M1 (FP4) elements of the OCP microscaling formats (MXFP4, MXFP8).
//! * The MXFP4-weight x MXFP8-activation product that the scaled matrix instruction of gfx950 (`v_mfma_scale_f32_16x16x128_f8f6f4`) computes,
//!   including a lane-level emulation of the kernel in `hip/hip_gfx950.cu` (same addressing, same operand layout), so that the packing and
//!   the index arithmetic are tested without the GPU.
//!
//! `hip/hip_mx.cuh` is the same code in C++; `tests::the_conversions_match_the_c_header` compares both on 2 million inputs through a
//! checksum computed from the C++ header (see the constants there). Nothing here is on the inference path; it is the specification the
//! GPU kernels are held to (`fllm-rs selftest-gpu --device hip:N` runs the GPU against the scalar reference on a gfx950).

const NAN_BITS: u32 = 0x7fc0_0000;

fn neg_if(bit: bool, v: f32) -> f32 {
    if bit {
        f32::from_bits(v.to_bits() ^ 0x8000_0000)
    } else {
        v
    }
}

/// OCP E4M3: bias 7, largest finite 448, `S.1111.111` is NaN, no infinity.
pub fn e4m3_ocp_to_f32(b: u32) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = if e == 15 && m == 7 {
        f32::from_bits(NAN_BITS)
    } else if e == 0 {
        m as f32 * (1.0 / 512.0)
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    neg_if(b & 0x80 != 0, v)
}

/// Round to nearest even, saturating at 448 (NaN stays NaN).
pub fn f32_to_e4m3_ocp(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 24) & 0x80;
    let abs_bits = bits & 0x7fff_ffff;
    if abs_bits > 0x7f80_0000 {
        return 0x7f;
    }
    if abs_bits >= 0x43e0_0000 {
        return sign | 0x7e;
    }
    if abs_bits < 0x3a80_0000 {
        return sign;
    }
    let exp = (abs_bits >> 23) as i32 - 127;
    if exp < -6 {
        let scaled = f32::from_bits(abs_bits) * 512.0;
        let fl = (scaled as i32) as f32;
        let diff = scaled - fl;
        let mut m = fl as i32;
        if diff > 0.5 || (diff == 0.5 && (m & 1) != 0) {
            m += 1;
        }
        return sign | m as u32;
    }
    let mant = abs_bits & 0x7f_ffff;
    let mut e = (exp + 7) as u32;
    let mut m = mant >> 20;
    let rest = mant & 0xf_ffff;
    if rest > 0x8_0000 || (rest == 0x8_0000 && (m & 1) != 0) {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }
    if e > 15 || (e == 15 && m == 7) {
        return sign | 0x7e;
    }
    sign | (e << 3) | m
}

/// FNUZ E4M3: bias 8, largest finite 240, `1.0000.000` is NaN, no negative zero, no infinity.
pub fn e4m3_fnuz_to_f32(b: u32) -> f32 {
    if b == 0x80 {
        return f32::from_bits(NAN_BITS);
    }
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = if e == 0 { m as f32 * (1.0 / 1024.0) } else { f32::from_bits(((e + 119) << 23) | (m << 20)) };
    neg_if(b & 0x80 != 0, v)
}

/// Round to nearest even, saturating at 240; zero is `0x00`, NaN is `0x80`.
pub fn f32_to_e4m3_fnuz(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 24) & 0x80;
    let abs_bits = bits & 0x7fff_ffff;
    if abs_bits > 0x7f80_0000 {
        return 0x80;
    }
    if abs_bits >= 0x4370_0000 {
        return sign | 0x7f;
    }
    if abs_bits < 0x3a00_0000 {
        return 0;
    }
    let exp = (abs_bits >> 23) as i32 - 127;
    if exp < -7 {
        let scaled = f32::from_bits(abs_bits) * 1024.0;
        let fl = (scaled as i32) as f32;
        let diff = scaled - fl;
        let mut m = fl as i32;
        if diff > 0.5 || (diff == 0.5 && (m & 1) != 0) {
            m += 1;
        }
        return if m == 0 { 0 } else { sign | m as u32 };
    }
    let mant = abs_bits & 0x7f_ffff;
    let mut e = (exp + 8) as u32;
    let mut m = mant >> 20;
    let rest = mant & 0xf_ffff;
    if rest > 0x8_0000 || (rest == 0x8_0000 && (m & 1) != 0) {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }
    if e > 15 {
        return sign | 0x7f;
    }
    sign | (e << 3) | m
}

/// The FNUZ byte of the same value as an OCP byte (values above 240 saturate; OCP NaN becomes the FNUZ NaN).
pub fn e4m3_ocp_to_fnuz(b: u32) -> u32 {
    f32_to_e4m3_fnuz(e4m3_ocp_to_f32(b))
}

/// The OCP byte of the same value as an FNUZ byte.
pub fn e4m3_fnuz_to_ocp(b: u32) -> u32 {
    f32_to_e4m3_ocp(e4m3_fnuz_to_f32(b))
}

/// OCP E5M2 (decode).
pub fn e5m2_ocp_to_f32(b: u32) -> f32 {
    let (e, m) = ((b >> 2) & 31, b & 3);
    let v = if e == 31 {
        if m == 0 {
            f32::from_bits(0x7f80_0000)
        } else {
            f32::from_bits(NAN_BITS)
        }
    } else if e == 0 {
        m as f32 * (1.0 / 65536.0)
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 21))
    };
    neg_if(b & 0x80 != 0, v)
}

/// E8M0 block scale: `2^(e - 127)`, 255 is NaN.
pub fn e8m0_to_f32(e: u32) -> f32 {
    match e {
        255 => f32::from_bits(NAN_BITS),
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits(e << 23),
    }
}

/// E2M1 (FP4): `+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
pub fn e2m1_to_f32(code: u32) -> f32 {
    let (e, m) = ((code >> 1) & 3, code & 1);
    let v = if e == 0 { 0.5 * m as f32 } else { f32::from_bits(((126 + e) << 23) | (m << 22)) };
    neg_if(code & 8 != 0, v)
}

/// The E2M1 code nearest to `x` (ties to even), saturating at 6.
pub fn f32_to_e2m1(x: f32) -> u32 {
    let sign = if x.is_sign_negative() && x != 0.0 { 8 } else { 0 };
    let a = x.abs();
    // midpoints between neighbours: 0.25 .75 1.25 1.75 2.5 3.5 5; ties go to the even code
    let code = if a.is_nan() {
        7
    } else if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    sign | code
}

/// Shared scale (E8M0 byte) of a block of 32 values with largest magnitude `amax`, for E4M3 elements (OCP MX: floor(log2 amax) - 8).
pub fn mx_scale_for_e4m3(amax: f32) -> u32 {
    if amax.is_nan() || amax <= 0.0 {
        return 127;
    }
    let exp = (((amax.to_bits() >> 23) & 0xff) as i32 - 127).max(-126);
    (exp - 8).clamp(-127, 127).wrapping_add(127) as u32
}

/// Shared scale of a block of E2M1 elements (largest element 6 = 1.5 * 2^2): floor(log2 amax) - 2.
pub fn mx_scale_for_e2m1(amax: f32) -> u32 {
    if amax.is_nan() || amax <= 0.0 {
        return 127;
    }
    let exp = (((amax.to_bits() >> 23) & 0xff) as i32 - 127).max(-126);
    (exp - 2).clamp(-127, 127).wrapping_add(127) as u32
}

/// Activations (`tokens` rows of `cols`, `cols % 32 == 0`) to MXFP8: E4M3 bytes and one E8M0 byte per 32 values.
pub fn quantize_mxfp8(x: &[f32], cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(cols.is_multiple_of(32) && x.len().is_multiple_of(cols));
    let mut q = vec![0u8; x.len()];
    let mut s = Vec::with_capacity(x.len() / 32);
    for (block, out) in x.chunks(32).zip(q.chunks_mut(32)) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let e = mx_scale_for_e4m3(amax);
        let inv = 1.0 / e8m0_to_f32(e);
        for (o, v) in out.iter_mut().zip(block) {
            *o = f32_to_e4m3_ocp(v * inv) as u8;
        }
        s.push(e as u8);
    }
    (q, s)
}

/// Weights (`rows` x `cols`, row-major floats) to the packed MXFP4 layout: `wq` has `cols / 2` bytes per row (element k is the low nibble
/// for even k), `ws` one E8M0 byte per 32 values.
pub fn quantize_mxfp4(w: &[f32], cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(cols.is_multiple_of(32) && w.len().is_multiple_of(cols));
    let mut wq = vec![0u8; w.len() / 2];
    let mut ws = Vec::with_capacity(w.len() / 32);
    for (b, block) in w.chunks(32).enumerate() {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let e = mx_scale_for_e2m1(amax);
        let inv = 1.0 / e8m0_to_f32(e);
        for (i, v) in block.iter().enumerate() {
            let code = f32_to_e2m1(v * inv) as u8;
            let k = b * 32 + i;
            wq[k / 2] |= if k % 2 == 0 { code } else { code << 4 };
        }
        ws.push(e as u8);
    }
    (wq, ws)
}

/// Blocks as GGUF stores MXFP4 (17 bytes: the E8M0 scale, then 16 bytes in which byte `j` holds element `j` in its low and element `j + 16`
/// in its high nibble) to the packed layout of the scaled-MFMA kernel (consecutive nibbles).
pub fn pack_mxfp4_from_gguf(blocks: &[u8]) -> (Vec<u8>, Vec<u8>) {
    assert!(blocks.len().is_multiple_of(17));
    let n = blocks.len() / 17;
    let mut wq = vec![0u8; n * 16];
    let mut ws = Vec::with_capacity(n);
    for (b, block) in blocks.chunks(17).enumerate() {
        ws.push(block[0]);
        for j in 0..16 {
            let (lo, hi) = (block[1 + j] & 15, block[1 + j] >> 4);
            for (k, code) in [(j, lo), (j + 16, hi)] {
                let at = b * 16 + k / 2;
                wq[at] |= if k % 2 == 0 { code } else { code << 4 };
            }
        }
    }
    (wq, ws)
}

fn weight_value(wq: &[u8], row: usize, k: usize, cols: usize) -> f32 {
    let byte = wq[row * (cols / 2) + k / 2] as u32;
    e2m1_to_f32(if k % 2 == 1 { byte >> 4 } else { byte & 15 })
}

/// `Y[token][row] = sum_k X[token][k] * W[row][k]` with the dequantization of the scalar reference kernel (`gemm_ref_kernel`).
pub fn gemm_ref(wq: &[u8], ws: &[u8], xq: &[u8], xs: &[u8], rows: usize, cols: usize, tokens: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * tokens];
    for t in 0..tokens {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for j in 0..cols / 32 {
                let (w_scale, x_scale) = (e8m0_to_f32(ws[r * (cols / 32) + j] as u32), e8m0_to_f32(xs[t * (cols / 32) + j] as u32));
                let mut part = 0.0f32;
                for i in 0..32 {
                    let k = j * 32 + i;
                    part += weight_value(wq, r, k, cols) * e4m3_ocp_to_f32(xq[t * cols + k] as u32);
                }
                acc += part * w_scale * x_scale;
            }
            y[t * rows + r] = acc;
        }
    }
    y
}

/// The kernel `gemm_mfma_kernel` of `hip/hip_gfx950.cu` executed lane by lane: every lane of a 64-lane wavefront loads its registers with the
/// kernel's addressing, the scaled 16x16x128 instruction is evaluated from the registers (D[m][n] += sum over the four 32-wide k groups of
/// scale_a[lane m of the group] * scale_b[lane n of the group] * sum_j A_reg * B_reg), and the accumulators are stored with the kernel's output
/// mapping. Verifies the packing, the strides and the tile edges; the instruction semantics are the documented ones, not measured.
pub fn emulate_mfma_gemm(wq: &[u8], ws: &[u8], xq: &[u8], xs: &[u8], rows: usize, cols: usize, tokens: usize) -> Vec<f32> {
    assert!(cols.is_multiple_of(128));
    let mut y = vec![0.0f32; rows * tokens];
    for tile_row in (0..rows).step_by(16) {
        for tile_tok in (0..tokens).step_by(16) {
            let mut acc = vec![[0.0f32; 4]; 64];
            for k0 in (0..cols).step_by(128) {
                // registers of the 64 lanes: A as 32 E2M1 values, B as 32 E4M3 bytes, one scale byte each
                let mut a = vec![[0u32; 32]; 64];
                let mut b = vec![[0u32; 32]; 64];
                let (mut sa, mut sb) = ([0u32; 64], [0u32; 64]);
                for lane in 0..64usize {
                    let row = tile_row + (lane & 15);
                    let tok = tile_tok + (lane & 15);
                    let lrow = if row < rows { row } else { 0 };
                    let ltok = if tok < tokens { tok } else { 0 };
                    let k = k0 + (lane >> 4) * 32;
                    // 16 bytes at wq[lrow][k / 2 ..]: the 32 nibbles of the lane, first element in the low nibble of the lowest byte
                    for (i, slot) in a[lane].iter_mut().enumerate() {
                        let byte = wq[lrow * (cols / 2) + k / 2 + i / 2] as u32;
                        *slot = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                    }
                    for (i, slot) in b[lane].iter_mut().enumerate() {
                        *slot = xq[ltok * cols + k + i] as u32;
                    }
                    sa[lane] = ws[lrow * (cols / 32) + k / 32] as u32;
                    sb[lane] = xs[ltok * (cols / 32) + k / 32] as u32;
                }
                // D[m][n] for accumulator element i of lane l: m = 4 * (l / 16) + i, n = l % 16
                for (lane, d) in acc.iter_mut().enumerate() {
                    let n = lane & 15;
                    for (i, out) in d.iter_mut().enumerate() {
                        let m = 4 * (lane >> 4) + i;
                        let mut sum = 0.0f32;
                        for g in 0..4 {
                            let (la, lb) = (g * 16 + m, g * 16 + n); // the lanes that hold row m / column n of k group g
                            let mut part = 0.0f32;
                            for j in 0..32 {
                                part += e2m1_to_f32(a[la][j]) * e4m3_ocp_to_f32(b[lb][j]);
                            }
                            sum += part * e8m0_to_f32(sa[la]) * e8m0_to_f32(sb[lb]);
                        }
                        *out += sum;
                    }
                }
            }
            for (lane, d) in acc.iter().enumerate() {
                let tok = tile_tok + (lane & 15);
                if tok < tokens {
                    for (i, v) in d.iter().enumerate() {
                        let row = tile_row + 4 * (lane >> 4) + i;
                        if row < rows {
                            y[tok * rows + row] = *v;
                        }
                    }
                }
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(h: u64, v: u32) -> u64 {
        v.to_le_bytes().iter().fold(h, |h, b| (h ^ u64::from(*b)).wrapping_mul(1_099_511_628_211))
    }

    const FNV_START: u64 = 1_469_598_103_934_665_603;

    #[test]
    fn known_values_of_both_e4m3_flavours() {
        assert_eq!(e4m3_ocp_to_f32(0x38), 1.0);
        assert_eq!(e4m3_ocp_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_ocp_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_ocp_to_f32(0x01), 2f32.powi(-9));
        assert!(e4m3_ocp_to_f32(0x7f).is_nan() && e4m3_ocp_to_f32(0xff).is_nan());
        assert_eq!(e4m3_fnuz_to_f32(0x7f), 240.0);
        assert_eq!(e4m3_fnuz_to_f32(0x01), 2f32.powi(-10));
        assert_eq!(e4m3_fnuz_to_f32(0x08), 2f32.powi(-7));
        assert!(e4m3_fnuz_to_f32(0x80).is_nan());
        assert_eq!(e4m3_fnuz_to_f32(0x00), 0.0);
        // the same bit pattern means a different number: that is the whole gfx942 / gfx950 difference
        assert_eq!(e4m3_ocp_to_f32(0x38), 1.0);
        assert_eq!(e4m3_fnuz_to_f32(0x38), 0.5);
        assert_eq!(e4m3_fnuz_to_f32(0x40), 1.0);
        assert_eq!(e5m2_ocp_to_f32(0x3c), 1.0);
        assert_eq!(e5m2_ocp_to_f32(0x7b), 57344.0);
        assert!(e5m2_ocp_to_f32(0x7c).is_infinite());
    }

    #[test]
    fn every_ocp_code_round_trips() {
        for c in 0..256u32 {
            let v = e4m3_ocp_to_f32(c);
            let back = f32_to_e4m3_ocp(v);
            if v.is_nan() {
                assert_eq!(back, 0x7f);
            } else {
                assert_eq!(back, c, "code {c:#x} = {v}");
            }
        }
    }

    #[test]
    fn every_fnuz_code_round_trips() {
        for c in 0..256u32 {
            let v = e4m3_fnuz_to_f32(c);
            let back = f32_to_e4m3_fnuz(v);
            if v.is_nan() {
                assert_eq!(back, 0x80);
            } else {
                assert_eq!(back, c, "code {c:#x} = {v}");
            }
        }
    }

    #[test]
    fn ocp_to_fnuz_keeps_the_value_up_to_240_and_saturates_above() {
        for c in 0..256u32 {
            let v = e4m3_ocp_to_f32(c);
            let f = e4m3_ocp_to_fnuz(c);
            let w = e4m3_fnuz_to_f32(f);
            if v.is_nan() {
                assert_eq!(f, 0x80);
            } else if v.abs() <= 240.0 {
                assert_eq!(w, v, "ocp {c:#x} = {v} became fnuz {f:#x} = {w}");
            } else {
                assert_eq!(w, 240.0f32.copysign(v), "ocp {c:#x} = {v} must saturate");
            }
        }
        // the other way the range is no problem (240 < 448) but the FNUZ values below 2^-6 sit on a finer grid than the OCP subnormals
        for c in 0..256u32 {
            let v = e4m3_fnuz_to_f32(c);
            let o = e4m3_fnuz_to_ocp(c);
            if !v.is_nan() && v.abs() >= 2f32.powi(-6) {
                assert_eq!(e4m3_ocp_to_f32(o), v, "fnuz {c:#x}");
            }
        }
        // negative zero has no FNUZ code
        assert_eq!(e4m3_ocp_to_fnuz(0x80), 0);
    }

    #[test]
    fn rounding_is_to_nearest_even_with_saturation() {
        // 1.0 = 0x38, next is 1.125 (0x39); the midpoint 1.0625 goes to the even mantissa (1.0), 1.1875 (between 1.125 and 1.25) to 1.25
        assert_eq!(f32_to_e4m3_ocp(1.0625), 0x38);
        assert_eq!(f32_to_e4m3_ocp(1.1875), 0x3a);
        assert_eq!(f32_to_e4m3_ocp(1.07), 0x39);
        assert_eq!(f32_to_e4m3_ocp(1000.0), 0x7e);
        assert_eq!(f32_to_e4m3_ocp(-1000.0), 0xfe);
        assert_eq!(f32_to_e4m3_ocp(f32::INFINITY), 0x7e);
        assert_eq!(f32_to_e4m3_ocp(2f32.powi(-10)), 0x00); // half of the smallest subnormal rounds to even (0)
        assert_eq!(f32_to_e4m3_ocp(1.5 * 2f32.powi(-10)), 0x01);
        assert_eq!(f32_to_e4m3_fnuz(1000.0), 0x7f);
        assert_eq!(f32_to_e4m3_fnuz(-0.0), 0x00);
        assert_eq!(f32_to_e4m3_fnuz(1e-9), 0x00);
    }

    #[test]
    fn e8m0_and_e2m1_tables() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(1), 2f32.powi(-126));
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert!(e8m0_to_f32(255).is_nan());
        let values: Vec<f32> = (0..16).map(e2m1_to_f32).collect();
        assert_eq!(values, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]);
        for c in 0..16u32 {
            let v = e2m1_to_f32(c);
            assert_eq!(e2m1_to_f32(f32_to_e2m1(v)), v, "code {c}");
        }
        assert_eq!(f32_to_e2m1(0.25), 0); // tie between 0 and 0.5: even code (0)
        assert_eq!(f32_to_e2m1(0.75), 2); // tie between 0.5 and 1: even code (1.0 = code 2)
        assert_eq!(f32_to_e2m1(100.0), 7);
        assert_eq!(f32_to_e2m1(-100.0), 15);
    }

    #[test]
    fn the_conversions_match_the_c_header() {
        let mut h_dec = FNV_START;
        for c in 0..256u32 {
            for v in [e4m3_ocp_to_f32(c), e4m3_fnuz_to_f32(c), e5m2_ocp_to_f32(c), e8m0_to_f32(c), e2m1_to_f32(c & 15)] {
                h_dec = fnv(h_dec, v.to_bits());
            }
        }
        let (mut h_ocp, mut h_fnuz, mut h_scale) = (FNV_START, FNV_START, FNV_START);
        let mut seed = 987_654_321u32;
        for i in 0..2_000_000u32 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut bits = seed;
            if i % 4 == 0 {
                bits = (bits & 0x807f_ffff) | ((100 + (bits >> 23) % 40) << 23);
            }
            let x = f32::from_bits(bits);
            h_ocp = fnv(h_ocp, f32_to_e4m3_ocp(x));
            h_fnuz = fnv(h_fnuz, f32_to_e4m3_fnuz(x));
            h_scale = fnv(h_scale, mx_scale_for_e4m3(x.abs()));
        }
        // computed by compiling hip/hip_mx.cuh as plain C++ (g++ -O2) and running the same loops
        assert_eq!(h_dec, 0x1d8e_df14_1a2d_12ea, "decode tables differ from hip_mx.cuh");
        assert_eq!(h_ocp, 0x9116_dbe7_41af_ce94, "OCP E4M3 encode differs from hip_mx.cuh");
        assert_eq!(h_fnuz, 0x4067_f31d_6d33_da8f, "FNUZ E4M3 encode differs from hip_mx.cuh");
        assert_eq!(h_scale, 0x823a_ab20_7b60_df7d, "MX scale differs from hip_mx.cuh");
    }

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*seed >> 8) & 0xffff) as f32 / 32768.0 - 1.0
    }

    #[test]
    fn mxfp8_quantization_error_is_small() {
        let mut seed = 7u32;
        let x: Vec<f32> = (0..4 * 256).map(|i| lcg(&mut seed) * if i % 97 == 0 { 40.0 } else { 3.0 }).collect();
        let (q, s) = quantize_mxfp8(&x, 256);
        for (i, v) in x.iter().enumerate() {
            let back = e4m3_ocp_to_f32(q[i] as u32) * e8m0_to_f32(s[i / 32] as u32);
            // 3 mantissa bits: relative error up to 2^-4 of the value, plus the block's smallest step for tiny values
            let step = e8m0_to_f32(s[i / 32] as u32) * 2f32.powi(-9);
            assert!((back - v).abs() <= v.abs() / 16.0 + step, "element {i}: {v} -> {back}");
        }
    }

    #[test]
    fn gguf_blocks_are_repacked_into_consecutive_nibbles() {
        // one block: element j = j % 16 (E2M1 code), scale byte 130
        let mut block = vec![130u8];
        for j in 0..16u8 {
            block.push((j & 15) | (((j + 3) & 15) << 4)); // element j = j, element j + 16 = j + 3
        }
        let (wq, ws) = pack_mxfp4_from_gguf(&block);
        assert_eq!(ws, [130]);
        for k in 0..16usize {
            let get = |k: usize| if k.is_multiple_of(2) { wq[k / 2] & 15 } else { wq[k / 2] >> 4 };
            assert_eq!(get(k) as usize, k % 16);
            assert_eq!(get(k + 16) as usize, (k + 3) % 16);
        }
    }

    type Problem = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>);

    fn random_problem(rows: usize, cols: usize, tokens: usize, seed: u32) -> Problem {
        let mut seed = seed;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed) * 2.0).collect();
        let x: Vec<f32> = (0..tokens * cols).map(|_| lcg(&mut seed) * 3.0).collect();
        let (wq, ws) = quantize_mxfp4(&w, cols);
        let (xq, xs) = quantize_mxfp8(&x, cols);
        (wq, ws, xq, xs, w, x)
    }

    #[test]
    fn mxfp4_x_mxfp8_product_approximates_the_float_product() {
        let (rows, cols, tokens) = (24, 256, 5);
        let (wq, ws, xq, xs, w, x) = random_problem(rows, cols, tokens, 11);
        let y = gemm_ref(&wq, &ws, &xq, &xs, rows, cols, tokens);
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for t in 0..tokens {
            for r in 0..rows {
                let exact: f64 = (0..cols).map(|k| f64::from(w[r * cols + k]) * f64::from(x[t * cols + k])).sum();
                scale = scale.max(exact.abs());
                worst = worst.max((exact - f64::from(y[t * rows + r])).abs());
            }
        }
        // 4-bit weights: ~10% relative error per element, averaging out in a 256-long dot product
        assert!(worst < 0.2 * scale + 2.0, "worst {worst}, largest {scale}");
    }

    #[test]
    fn the_lane_level_emulation_equals_the_reference_for_awkward_shapes() {
        for (rows, cols, tokens) in [(16, 128, 16), (24, 256, 5), (17, 384, 33), (1, 128, 1), (40, 512, 20)] {
            let (wq, ws, xq, xs, _, _) = random_problem(rows, cols, tokens, 3 + rows as u32);
            let reference = gemm_ref(&wq, &ws, &xq, &xs, rows, cols, tokens);
            let emulated = emulate_mfma_gemm(&wq, &ws, &xq, &xs, rows, cols, tokens);
            let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (i, (a, b)) in reference.iter().zip(&emulated).enumerate() {
                assert!((a - b).abs() <= 1e-5 * scale + 1e-6, "{rows}x{cols}x{tokens}: element {i}: {a} vs {b}");
            }
        }
    }
}
