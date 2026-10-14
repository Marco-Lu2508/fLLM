//! Host-side conversions of weights the Vulkan shaders have no decoder for. Plain CPU code, tested without a device.

use half::f16;

/// IQ4_XS super-blocks (136 bytes: f16 `d`, 16 bits of high scale bits, 4 bytes of low scale bits, 128 bytes of nibbles) as the eight IQ4_NL
/// blocks (18 bytes: f16 scale, 16 nibble bytes) they consist of: the value grid and the nibble order are the same, only the scale of every
/// group of 32 differs (`d * (ls - 32)` rounded to binary16 here).
pub(crate) fn iq4_xs_to_nl(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 136 * 8 * 18);
    for block in raw.as_chunks::<136>().0 {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        for ib in 0..8 {
            let lo = (block[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            let hi = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = i32::from(lo | (hi << 4)) - 32;
            out.extend_from_slice(&f16::from_f32(d * ls as f32).to_le_bytes());
            out.extend_from_slice(&block[8 + 16 * ib..8 + 16 * ib + 16]);
        }
    }
    out
}

/// `values` as little-endian binary16, or `None` when a value is not finite or too large for binary16 (the caller keeps F32 then).
pub(crate) fn to_f16_bytes(values: &[f32]) -> Option<Vec<u8>> {
    const LIMIT: f32 = 60000.0;
    if values.iter().any(|v| !v.is_finite() || v.abs() > LIMIT) {
        return None;
    }
    Some(values.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect())
}

/// `values` as little-endian binary32.
pub(crate) fn to_f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as little-endian 32-bit unsigned integers.
pub(crate) fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::dequantize;

    /// Deterministic pseudo-random bytes.
    fn noise(len: usize, mut state: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn iq4_xs_converts_to_the_same_values_as_iq4_nl() {
        let mut raw = noise(136 * 6, 7);
        for block in raw.as_chunks_mut::<136>().0 {
            // a sane super-block scale (binary16 of 0.0123)
            block[0..2].copy_from_slice(&f16::from_f32(0.0123).to_le_bytes());
        }
        let want = dequantize(&raw, QuantTypeIds::IQ4_XS, 6 * 256).unwrap();
        let nl = iq4_xs_to_nl(&raw);
        assert_eq!(nl.len(), 6 * 8 * 18);
        let have = dequantize(&nl, QuantTypeIds::IQ4_NL, 6 * 256).unwrap();
        let worst = want
            .iter()
            .zip(&have)
            .map(|(a, b)| (a - b).abs() / (a.abs().max(1e-3)))
            .fold(0.0f32, f32::max);
        // the only difference is the binary16 rounding of d * (ls - 32)
        assert!(worst < 2e-3, "relative difference {worst}");
    }

    struct QuantTypeIds;
    impl QuantTypeIds {
        const IQ4_NL: u32 = 20;
        const IQ4_XS: u32 = 23;
    }

    #[test]
    fn f16_expansion_checks_the_range() {
        assert!(to_f16_bytes(&[1.0, -2.5, 0.0]).is_some());
        assert!(to_f16_bytes(&[1.0e5]).is_none());
        assert!(to_f16_bytes(&[f32::NAN]).is_none());
        let bytes = to_f16_bytes(&[1.5]).unwrap();
        assert_eq!(f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(), 1.5);
    }

    #[test]
    fn word_packing() {
        assert_eq!(u32_bytes(&[1, 0x0102_0304]), vec![1, 0, 0, 0, 4, 3, 2, 1]);
        assert_eq!(to_f32_bytes(&[1.0]), 1.0f32.to_le_bytes().to_vec());
    }
}
