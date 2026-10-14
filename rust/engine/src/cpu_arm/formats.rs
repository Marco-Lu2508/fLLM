//! Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4 as the ARM kernels see them.
//!
//! Every format is cut into 32-value *groups* that become one [`Group`] record: signed 8-bit
//! weights (the two 16-value halves of a group keep their own scale where the format has
//! them), the scale of each half and the coefficient of the activation half sums. A kernel
//! computes, per group and activation record `x` (`x.q`, `x.d`, `x.sum`):
//!
//! ```text
//! out += x.d * (d[0] * dot(q[0..16], x.q[0..16]) + d[1] * dot(q[16..32], x.q[16..32]))
//!      + m[0] * x.sum[0] + m[1] * x.sum[1]
//! ```
//!
//! where `x.sum[h]` is `x.d` times the sum of the 8-bit activations of half `h`
//! (`cpu_quantized::prepare`). Formats with one scale per group (IQ4_NL, IQ4_XS, MXFP4) store
//! it twice and need only one dot and no `m`.
//!
//! The decoders here are scalar and platform independent: they define the layout, and the NEON
//! decoders in `quant.rs` are checked against them (on AArch64) bit for bit. They are also
//! checked against `compute::dequantize` on every platform.
//!
//! Block layouts (GGUF):
//!
//! | Format | Block | Bytes | Layout |
//! | --- | --- | --- | --- |
//! | Q2_K | 256 | 84 | `scales[16]` (low nibble scale, high nibble minimum, one per 16 values), `qs[64]` (four 2-bit values per byte), `d`, `dmin` (f16) |
//! | Q3_K | 256 | 110 | `hmask[32]`, `qs[64]` (2 bits), `scales[12]` (sixteen 6-bit scales), `d` (f16) |
//! | IQ4_NL | 32 | 18 | `d` (f16), `qs[16]`: low nibbles are values 0..16, high nibbles 16..32, through the 16-entry codebook |
//! | IQ4_XS | 256 | 136 | `d` (f16), `scales_h` (u16), `scales_l[4]`, `qs[128]`; eight groups with a 6-bit scale each |
//! | MXFP4 | 32 | 17 | E8M0 scale byte, `qs[16]` of E2M1 nibbles (low nibbles first) |
//!
//! Outside the tests the scalar decoders are not called by the kernels, hence the dead-code
//! allowance below: the constants and `const fn`s serve the NEON kernels on AArch64 and the
//! tests everywhere else.
#![cfg_attr(not(test), allow(dead_code))]
use half::f16;
/// GGUF type ids of the formats in this module.
pub(crate) const Q2_K: u32 = 10;
pub(crate) const Q3_K: u32 = 11;
pub(crate) const IQ4_NL: u32 = 20;
pub(crate) const IQ4_XS: u32 = 23;
pub(crate) const MXFP4: u32 = 39;

/// The IQ4_NL / IQ4_XS codebook (`kvalues_iq4nl`).
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Signed E2M1 values of MXFP4 at twice their magnitude; [`e8m0_half`] halves the scale.
pub(crate) const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// E8M0 exponent as 2^(x - 128): the scale matching [`MXFP4_VALUES`] (as `quant_extra`).
#[inline]
pub(crate) fn e8m0_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000u32 << x
    } else {
        (u32::from(x) - 1) << 23
    })
}

/// A weight format with an ARM kernel in this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Q2K,
    Q3K,
    Iq4Nl,
    Iq4Xs,
    Mxfp4,
}

impl Format {
    pub(crate) const ALL: [Format; 5] = [
        Format::Q2K,
        Format::Q3K,
        Format::Iq4Nl,
        Format::Iq4Xs,
        Format::Mxfp4,
    ];

    pub(crate) const fn from_kind(kind: u32) -> Option<Self> {
        Some(match kind {
            Q2_K => Format::Q2K,
            Q3_K => Format::Q3K,
            IQ4_NL => Format::Iq4Nl,
            IQ4_XS => Format::Iq4Xs,
            MXFP4 => Format::Mxfp4,
            _ => return None,
        })
    }

    pub(crate) const fn kind(self) -> u32 {
        match self {
            Format::Q2K => Q2_K,
            Format::Q3K => Q3_K,
            Format::Iq4Nl => IQ4_NL,
            Format::Iq4Xs => IQ4_XS,
            Format::Mxfp4 => MXFP4,
        }
    }

    /// Bytes of one block.
    pub(crate) const fn block_bytes(self) -> usize {
        block_bytes(self.kind())
    }

    /// 32-value groups in one block (the K formats and IQ4_XS: 256 values).
    pub(crate) const fn groups(self) -> usize {
        groups(self.kind())
    }

    /// Whether the two 16-value halves of a group have their own scale.
    pub(crate) const fn split_scales(self) -> bool {
        split_scales(self.kind())
    }

    /// Whether the format has a per-half minimum (`m` is non-zero).
    pub(crate) const fn has_min(self) -> bool {
        has_min(self.kind())
    }
}

/// `const fn` forms on the GGUF type id, for the const-generic NEON kernels.
pub(crate) const fn block_bytes(kind: u32) -> usize {
    match kind {
        Q2_K => 84,
        Q3_K => 110,
        IQ4_NL => 18,
        IQ4_XS => 136,
        MXFP4 => 17,
        _ => 0,
    }
}
pub(crate) const fn groups(kind: u32) -> usize {
    match kind {
        Q2_K | Q3_K | IQ4_XS => 8,
        _ => 1,
    }
}
pub(crate) const fn split_scales(kind: u32) -> bool {
    matches!(kind, Q2_K | Q3_K)
}
pub(crate) const fn has_min(kind: u32) -> bool {
    matches!(kind, Q2_K)
}

/// One decoded 32-value group; also the panel record of the prompt kernels.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Group {
    /// Signed weights; values `0..16` are the first half.
    pub(crate) q: [i8; 32],
    /// Scale of each 16-value half (equal for the formats with one scale per group).
    pub(crate) d: [f32; 2],
    /// Coefficient of the activation half sums `x.sum[0]`, `x.sum[1]`; zero except for Q2_K.
    pub(crate) m: [f32; 2],
}

impl Group {
    /// The value of lane `i` of this group, as `compute::dequantize` defines it.
    pub(crate) fn value(&self, i: usize) -> f32 {
        let h = i / 16;
        let scaled = self.d[h] * f32::from(self.q[i]);
        // Skipping a zero coefficient keeps the sign of a negative zero.
        if self.m[h] == 0.0 {
            scaled
        } else {
            scaled + self.m[h]
        }
    }
}

fn half_at(block: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

/// Sixteen signed Q3_K scales (`6-bit value - 32`).
pub(crate) fn q3k_scales(block: &[u8]) -> [i8; 16] {
    let mut scales = [0i8; 16];
    for i in 0..4 {
        let hi = block[96 + i];
        let lo = block[96 + 4 + i];
        let packed = block[96 + 8 + i];
        scales[i] = ((hi & 15) | ((packed & 3) << 4)) as i8 - 32;
        scales[i + 4] = ((lo & 15) | (((packed >> 2) & 3) << 4)) as i8 - 32;
        scales[i + 8] = ((hi >> 4) | (((packed >> 4) & 3) << 4)) as i8 - 32;
        scales[i + 12] = ((lo >> 4) | (((packed >> 6) & 3) << 4)) as i8 - 32;
    }
    scales
}

/// The 6-bit scale of IQ4_XS group `g`, minus 32.
pub(crate) fn iq4xs_scale(block: &[u8], g: usize) -> i32 {
    let high = u16::from_le_bytes([block[2], block[3]]);
    let low = (block[4 + g / 2] >> (4 * (g % 2))) & 15;
    (i32::from(low) | (i32::from((high >> (2 * g)) & 3) << 4)) - 32
}

/// Decode group `g` of one block of `format`.
pub(crate) fn decode_group(format: Format, block: &[u8], g: usize) -> Group {
    assert_eq!(block.len(), format.block_bytes());
    assert!(g < format.groups());
    let mut out = Group::default();
    match format {
        Format::Q2K => {
            let (d, dmin) = (half_at(block, 80), half_at(block, 82));
            let (n, shift) = (g / 4, 2 * (g % 4));
            let (s0, s1) = (block[2 * g], block[2 * g + 1]);
            for i in 0..32 {
                out.q[i] = ((block[16 + n * 32 + i] >> shift) & 3) as i8;
            }
            out.d = [d * f32::from(s0 & 15), d * f32::from(s1 & 15)];
            out.m = [-(dmin * f32::from(s0 >> 4)), -(dmin * f32::from(s1 >> 4))];
        }
        Format::Q3K => {
            let d = half_at(block, 108);
            let scales = q3k_scales(block);
            let (outer, shift) = (g / 4, 2 * (g % 4));
            for i in 0..32 {
                let low = ((block[32 + outer * 32 + i] >> shift) & 3) as i8;
                let high = ((block[i] >> g) & 1) as i8;
                out.q[i] = (low | (high << 2)) - 4;
            }
            out.d = [
                d * f32::from(scales[2 * g]),
                d * f32::from(scales[2 * g + 1]),
            ];
        }
        Format::Iq4Nl => {
            let d = half_at(block, 0);
            for j in 0..16 {
                out.q[j] = KVALUES_IQ4NL[usize::from(block[2 + j] & 15)];
                out.q[16 + j] = KVALUES_IQ4NL[usize::from(block[2 + j] >> 4)];
            }
            out.d = [d; 2];
        }
        Format::Iq4Xs => {
            let d = half_at(block, 0);
            for j in 0..16 {
                let byte = block[8 + g * 16 + j];
                out.q[j] = KVALUES_IQ4NL[usize::from(byte & 15)];
                out.q[16 + j] = KVALUES_IQ4NL[usize::from(byte >> 4)];
            }
            out.d = [d * iq4xs_scale(block, g) as f32; 2];
        }
        Format::Mxfp4 => {
            let d = e8m0_half(block[0]);
            for j in 0..16 {
                out.q[j] = MXFP4_VALUES[usize::from(block[1 + j] & 15)];
                out.q[16 + j] = MXFP4_VALUES[usize::from(block[1 + j] >> 4)];
            }
            out.d = [d; 2];
        }
    }
    out
}

/// Decode one row of whole blocks into `out` (one [`Group`] per 32 values).
pub(crate) fn decode_row(format: Format, row: &[u8], out: &mut [Group]) {
    let width = format.block_bytes();
    assert_eq!(row.len() % width, 0);
    assert_eq!(out.len(), row.len() / width * format.groups());
    for (block, out) in row
        .chunks_exact(width)
        .zip(out.chunks_exact_mut(format.groups()))
    {
        for (g, out) in out.iter_mut().enumerate() {
            *out = decode_group(format, block, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{random_weights, Lcg};

    #[test]
    fn const_helpers_agree_with_the_enum() {
        for format in Format::ALL {
            assert_eq!(Format::from_kind(format.kind()), Some(format));
            assert_eq!(format.block_bytes(), block_bytes(format.kind()));
            assert_eq!(
                format.groups() * 32,
                if format.groups() == 8 { 256 } else { 32 }
            );
            assert_eq!(
                crate::cpu_selftest::layout(format.kind()),
                (format.groups() * 32, format.block_bytes())
            );
        }
        assert_eq!(Format::from_kind(12), None);
        assert!(Format::Q2K.split_scales() && Format::Q2K.has_min());
        assert!(Format::Q3K.split_scales() && !Format::Q3K.has_min());
        for format in [Format::Iq4Nl, Format::Iq4Xs, Format::Mxfp4] {
            assert!(!format.split_scales() && !format.has_min());
        }
    }

    #[test]
    fn codebook_tables_match_the_reference_decoders() {
        // quant_iq / quant_extra own the reference tables; probe every nibble through them.
        for nibble in 0..16u8 {
            let mut iq = [0u8; 18];
            iq[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
            iq[2] = nibble;
            let values = crate::compute::dequantize(&iq, IQ4_NL, 32).unwrap();
            assert_eq!(values[0], f32::from(KVALUES_IQ4NL[usize::from(nibble)]));
            let mut mx = [0u8; 17];
            mx[0] = 128; // 2^(128 - 128) after halving: e8m0_half(128) is 1.0
            mx[1] = nibble;
            let values = crate::compute::dequantize(&mx, MXFP4, 32).unwrap();
            assert_eq!(values[0], f32::from(MXFP4_VALUES[usize::from(nibble)]));
        }
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(0), f32::from_bits(0x0020_0000));
        assert_eq!(e8m0_half(1), f32::from_bits(0x0040_0000));
    }

    #[test]
    fn scalar_decoders_equal_the_reference_dequantization_bit_for_bit() {
        let mut rng = Lcg(0x13_a2_7c);
        for format in Format::ALL {
            let kind = format.kind();
            let (values, width) = crate::cpu_selftest::layout(kind);
            for _ in 0..8 {
                let block = random_weights(kind, 1, values, &mut rng);
                assert_eq!(block.len(), width);
                let reference = crate::compute::dequantize(&block, kind, values).unwrap();
                let mut groups = vec![Group::default(); format.groups()];
                decode_row(format, &block, &mut groups);
                for (g, group) in groups.iter().enumerate() {
                    for i in 0..32 {
                        let have = group.value(i);
                        let want = reference[g * 32 + i];
                        assert_eq!(
                            have.to_bits(),
                            want.to_bits(),
                            "{format:?} group {g} lane {i}: {have} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q3k_scales_use_the_documented_bit_positions() {
        // Scale i lives in the low nibble of byte (i % 8) / ... : probe each of the 12 bytes.
        let mut block = [0u8; 110];
        // Only scale 5 non-zero: low part (4 bits) from byte 96 + 4 + 1 low nibble, high part
        // (2 bits) from byte 96 + 8 + 1 bits 2..3.
        block[96 + 4 + 1] = 0x0b;
        block[96 + 8 + 1] = 0b10 << 2;
        let scales = q3k_scales(&block);
        assert_eq!(scales[5], (0x0b | (0b10 << 4)) - 32);
        for (i, &s) in scales.iter().enumerate() {
            if i != 5 {
                assert_eq!(s, -32, "scale {i}");
            }
        }
        // The upper nibble of the same bytes feeds scale 13.
        let mut block = [0u8; 110];
        block[96 + 4 + 1] = 0x70;
        block[96 + 8 + 1] = 0b01 << 6;
        assert_eq!(q3k_scales(&block)[13], (0x07 | (0b01 << 4)) - 32);
    }

    #[test]
    fn iq4xs_scale_matches_the_reference_layout() {
        let mut block = [0u8; 136];
        block[2..4].copy_from_slice(&0b11_00_10_00_01_00_11_10u16.to_le_bytes());
        block[4..8].copy_from_slice(&[0x21, 0x43, 0x65, 0x87]);
        let want = |g: usize, low: i32| {
            (low | ((i32::from(0b11_00_10_00_01_00_11_10u16 >> (2 * g)) & 3) << 4)) - 32
        };
        for (g, low) in [1, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
            assert_eq!(iq4xs_scale(&block, g), want(g, low), "group {g}");
        }
    }
}
