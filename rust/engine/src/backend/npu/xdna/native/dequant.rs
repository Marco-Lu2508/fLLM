//! Fast weight rows -> bf16 for the formats of common models (Q4_0, Q8_0, Q4_K, Q6_K, F16, BF16, F32).
//!
//! The NPU reads bf16; the weights stay quantized in memory and are expanded when a tile is packed. For a model that does not fit in
//! memory as bf16 this runs on every prompt chunk, so it must be fast: these loops decode a block straight to bf16 (no f32 buffer, no
//! allocation). Every other format goes through the generic `crate::compute::dequantize` (slower, same values). The block layouts are
//! those of GGUF (`ggml-quants`); the unit tests compare each decoder with the generic one.

use half::f16;

/// f32 -> bf16 bits, round to nearest even (NaN and infinity keep their class).
#[inline(always)]
pub fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[inline(always)]
fn h(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

/// Bytes per block and values per block of the formats handled here.
pub fn layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        0 => Some((1, 4)),
        1 | 30 => Some((1, 2)),
        2 => Some((32, 18)),
        8 => Some((32, 34)),
        12 => Some((256, 144)),
        14 => Some((256, 210)),
        _ => None,
    }
}

#[inline(always)]
fn q4_0(src: &[u8], out: &mut [u16]) {
    for (b, o) in src.as_chunks::<18>().0.iter().zip(out.as_chunks_mut::<32>().0.iter_mut()) {
        let d = h(b);
        for l in 0..16 {
            o[l] = to_bf16(d * ((b[2 + l] & 15) as i32 - 8) as f32);
            o[l + 16] = to_bf16(d * ((b[2 + l] >> 4) as i32 - 8) as f32);
        }
    }
}

#[inline(always)]
fn q8_0(src: &[u8], out: &mut [u16]) {
    for (b, o) in src.as_chunks::<34>().0.iter().zip(out.as_chunks_mut::<32>().0.iter_mut()) {
        let d = h(b);
        for l in 0..32 {
            o[l] = to_bf16(d * (b[2 + l] as i8) as f32);
        }
    }
}

#[inline(always)]
fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        ((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
    }
}

#[inline(always)]
fn q4_k(src: &[u8], out: &mut [u16]) {
    for (b, o) in src.as_chunks::<144>().0.iter().zip(out.as_chunks_mut::<256>().0.iter_mut()) {
        let d = h(b);
        let dmin = h(&b[2..]);
        let scales = &b[4..16];
        let qs = &b[16..144];
        for j in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            let q = &qs[32 * j..32 * j + 32];
            let y = &mut o[64 * j..64 * j + 64];
            for l in 0..32 {
                y[l] = to_bf16(d1 * (q[l] & 15) as f32 - mm1);
                y[l + 32] = to_bf16(d2 * (q[l] >> 4) as f32 - mm2);
            }
        }
    }
}

#[inline(always)]
fn q6_k(src: &[u8], out: &mut [u16]) {
    for (b, o) in src.as_chunks::<210>().0.iter().zip(out.as_chunks_mut::<256>().0.iter_mut()) {
        let (ql_all, rest) = b.split_at(128);
        let (qh_all, rest) = rest.split_at(64);
        let (sc_all, dbytes) = rest.split_at(16);
        let d = h(dbytes);
        for half in 0..2 {
            let ql = &ql_all[64 * half..64 * half + 64];
            let qh = &qh_all[32 * half..32 * half + 32];
            let sc = &sc_all[8 * half..8 * half + 8];
            let y = &mut o[128 * half..128 * half + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[l] = to_bf16(d * (sc[is] as i8) as f32 * q1 as f32);
                y[l + 32] = to_bf16(d * (sc[is + 2] as i8) as f32 * q2 as f32);
                y[l + 64] = to_bf16(d * (sc[is + 4] as i8) as f32 * q3 as f32);
                y[l + 96] = to_bf16(d * (sc[is + 6] as i8) as f32 * q4 as f32);
            }
        }
    }
}

/// Decode `src` (whole blocks, `out.len()` values) of GGUF type `kind` to bf16. `false` when the format is not handled here (nothing
/// was written); the caller then uses the generic path.
pub fn decode(kind: u32, src: &[u8], out: &mut [u16]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU features were just detected.
        return unsafe { decode_avx2(kind, src, out) };
    }
    decode_impl(kind, src, out)
}

/// The same code compiled for AVX2 (the inlined loops above are vectorized with 256-bit registers); chosen at run time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_avx2(kind: u32, src: &[u8], out: &mut [u16]) -> bool {
    decode_impl(kind, src, out)
}

#[inline(always)]
fn decode_impl(kind: u32, src: &[u8], out: &mut [u16]) -> bool {
    match kind {
        0 => src.as_chunks::<4>().0.iter().zip(out.iter_mut()).for_each(|(b, o)| *o = to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        1 => src.as_chunks::<2>().0.iter().zip(out.iter_mut()).for_each(|(b, o)| *o = to_bf16(h(b))),
        30 => src.as_chunks::<2>().0.iter().zip(out.iter_mut()).for_each(|(b, o)| *o = u16::from_le_bytes([b[0], b[1]])),
        2 => q4_0(src, out),
        8 => q8_0(src, out),
        12 => q4_k(src, out),
        14 => q6_k(src, out),
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{random_weights, Lcg};
    use half::bf16;

    #[test]
    fn bf16_rounding_matches_the_half_crate() {
        let mut rng = Lcg(7);
        for _ in 0..20_000 {
            let v = (rng.next() as f32 - 1.0e9) * 1.0e-3 * f32::from_bits(0x3f80_0000 + (rng.next() as u32 % 0x0200_0000));
            if v.is_finite() {
                assert_eq!(to_bf16(v), bf16::from_f32(v).to_bits(), "{v}");
            }
        }
        assert_eq!(to_bf16(f32::INFINITY), bf16::INFINITY.to_bits());
        assert!(bf16::from_bits(to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn every_decoder_matches_the_generic_dequantizer() {
        for kind in [2u32, 8, 12, 14] {
            let (block, bytes) = layout(kind).unwrap();
            let (rows, cols) = (5usize, block * 3);
            let mut rng = Lcg(u64::from(kind) + 99);
            let raw = random_weights(kind, rows, cols, &mut rng);
            assert_eq!(raw.len(), rows * cols / block * bytes, "kind {kind}");
            let want = crate::compute::dequantize(&raw, kind, rows * cols).unwrap();
            let mut got = vec![0u16; rows * cols];
            assert!(decode(kind, &raw, &mut got));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let g = bf16::from_bits(*g).to_f32();
                // both round the same f32 up to the order of two products: at most one bf16 step apart
                assert!((g - w).abs() <= w.abs() * 2f32.powi(-7) + 1e-30, "kind {kind} element {i}: {g} vs {w}");
            }
        }
        // float formats
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.37).collect();
        let f32_raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut got = vec![0u16; 64];
        assert!(decode(0, &f32_raw, &mut got));
        assert!(got.iter().zip(&values).all(|(g, v)| *g == bf16::from_f32(*v).to_bits()));
        let f16_raw: Vec<u8> = values.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
        assert!(decode(1, &f16_raw, &mut got));
        assert!(got.iter().zip(&values).all(|(g, v)| *g == bf16::from_f32(f16::from_f32(*v).to_f32()).to_bits()));
        assert!(!decode(13, &[], &mut got), "Q5_K is left to the generic path");
    }
}
