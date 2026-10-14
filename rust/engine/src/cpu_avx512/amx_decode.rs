//! Weight panels for the AMX kernels: sixteen weight rows decoded into plain 8-bit values plus a
//! float scale (and, for formats with a minimum, a float offset) per group of 16 or 32 values.
//!
//! One uniform description covers every packed format with a CPU kernel:
//! `weight = scale[row][group] * q - min[row][group]`, with `q` an `i8` (`signed` formats: the
//! stored offset such as Q4_0's 8 or Q6_K's 32 is already subtracted, so the AMX dot needs no
//! correction) or a `u8` (Q2_K, Q4_K, Q5_K, Q4_1, Q5_1, which carry a minimum).
//!
//! Decoders are plain Rust loops compiled with `avx2`/`f16c`; the codebook formats use `pshufb`.
//! They never touch AVX-512 or AMX, so the unit tests run them on any AVX2 machine and compare
//! every weight with `compute::dequantize`.
use crate::cpu_kernels::x86::half_to_float;
use std::arch::x86_64::*;

/// IQ4_NL / IQ4_XS non-linear codebook.
use crate::cpu_kquant::IQ4_VALUES;

/// Signed MXFP4 (E2M1) values at twice their magnitude; `e8m0_half` halves the scale.
pub(crate) const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// GGUF type ids handled here.
pub(crate) const Q4_0: u32 = 2;
pub(crate) const Q4_1: u32 = 3;
pub(crate) const Q5_0: u32 = 6;
pub(crate) const Q5_1: u32 = 7;
pub(crate) const Q8_0: u32 = 8;
pub(crate) const Q2K: u32 = 10;
pub(crate) const Q3K: u32 = 11;
pub(crate) const Q4K: u32 = 12;
pub(crate) const Q5K: u32 = 13;
pub(crate) const Q6K: u32 = 14;
pub(crate) const IQ4NL: u32 = 20;
pub(crate) const IQ4XS: u32 = 23;
pub(crate) const MXFP4: u32 = 39;

/// How a format maps onto the AMX integer dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KindInfo {
    /// Values per scale group: 16 or 32.
    pub group: usize,
    /// `q` is signed (`tdpbssd`); otherwise unsigned (`tdpbusd`).
    pub signed: bool,
    /// The format has a per-group minimum/offset.
    pub has_min: bool,
    /// Values per block and bytes per block of the packed format.
    pub block: usize,
    pub bytes: usize,
}

pub(crate) fn kind_info(kind: u32) -> Option<KindInfo> {
    let (group, signed, has_min, block, bytes) = match kind {
        Q8_0 => (32, true, false, 32, 34),
        Q4_0 => (32, true, false, 32, 18),
        Q5_0 => (32, true, false, 32, 22),
        Q4_1 => (32, false, true, 32, 20),
        Q5_1 => (32, false, true, 32, 24),
        IQ4NL => (32, true, false, 32, 18),
        MXFP4 => (32, true, false, 32, 17),
        Q2K => (16, false, true, 256, 84),
        Q3K => (16, true, false, 256, 110),
        Q4K => (32, false, true, 256, 144),
        Q5K => (32, false, true, 256, 176),
        Q6K => (16, true, false, 256, 210),
        IQ4XS => (32, true, false, 256, 136),
        _ => return None,
    };
    Some(KindInfo {
        group,
        signed,
        has_min,
        block,
        bytes,
    })
}

/// The activation records a format consumes (see `cpu_quantized::activation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActKind {
    /// Q8_0 records (also what `Q8Input` holds for Q8_0).
    Q8_0,
    /// Q8_1 records: the fp16 sum in `sum[0]`.
    Q8_1,
    /// Q8_K records: a scale shared by eight records.
    Q8K,
}

pub(crate) fn act_kind(kind: u32) -> Option<ActKind> {
    match kind {
        Q8_0 | Q4_0 | Q5_0 | IQ4NL | MXFP4 => Some(ActKind::Q8_0),
        Q4_1 | Q5_1 => Some(ActKind::Q8_1),
        Q2K | Q3K | Q4K | Q5K | Q6K | IQ4XS => Some(ActKind::Q8K),
        _ => None,
    }
}

/// Rows per panel: one AMX tile of the weights.
pub(crate) const PANEL_ROWS: usize = 16;

/// Sixteen decoded weight rows of one projection.
pub(crate) struct AmxPanel {
    pub cols: usize,
    pub group: usize,
    /// Row-major `[16][cols]` 8-bit values (signed or unsigned per `KindInfo::signed`).
    pub q: Vec<u8>,
    /// `[group][16]` scales; rows past the valid count are zero.
    pub scale: Vec<f32>,
    /// `[group][16]` minimums (`weight = scale * q - min`); all zero without `has_min`.
    pub min: Vec<f32>,
    row_scale: Vec<f32>,
    row_min: Vec<f32>,
}

impl AmxPanel {
    pub(crate) fn new(cols: usize, group: usize) -> Self {
        assert!(group == 16 || group == 32);
        assert!(cols.is_multiple_of(group));
        let groups = cols / group;
        Self {
            cols,
            group,
            q: vec![0; PANEL_ROWS * cols],
            scale: vec![0.0; groups * PANEL_ROWS],
            min: vec![0.0; groups * PANEL_ROWS],
            row_scale: vec![0.0; groups],
            row_min: vec![0.0; groups],
        }
    }

    pub(crate) fn groups(&self) -> usize {
        self.cols / self.group
    }
}

#[inline(always)]
unsafe fn half(raw: &[u8], at: usize) -> f32 {
    half_to_float(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

/// E8M0 exponent as 2^(x - 128), the half scale matching `MXFP4_VALUES`.
#[inline]
pub(crate) fn e8m0_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000u32 << x
    } else {
        (u32::from(x) - 1) << 23
    })
}

/// Eight scales and eight minimums of a Q4_K / Q5_K block (bytes 4..16), one per byte of the two
/// returned words.
fn unpack_scales(raw: &[u8]) -> (u64, u64) {
    let u0 = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let u1 = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let u2 = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
    let scales_lo = u0 & 0x3f3f_3f3f;
    let scales_hi = (u2 & 0x0f0f_0f0f) | (((u0 >> 6) & 0x0303_0303) << 4);
    let mins_lo = u1 & 0x3f3f_3f3f;
    let mins_hi = ((u2 >> 4) & 0x0f0f_0f0f) | (((u1 >> 6) & 0x0303_0303) << 4);
    (
        u64::from(scales_lo) | (u64::from(scales_hi) << 32),
        u64::from(mins_lo) | (u64::from(mins_hi) << 32),
    )
}

/// The sixteen signed 6-bit Q3_K scales.
pub(crate) fn q3_scales(raw: &[u8]) -> [i32; 16] {
    let mut scales = [0; 16];
    for i in 0..4 {
        let (hi, lo, packed) = (raw[96 + i], raw[100 + i], raw[104 + i]);
        scales[i] = i32::from((hi & 15) | ((packed & 3) << 4)) - 32;
        scales[i + 4] = i32::from((lo & 15) | (((packed >> 2) & 3) << 4)) - 32;
        scales[i + 8] = i32::from((hi >> 4) | (((packed >> 4) & 3) << 4)) - 32;
        scales[i + 12] = i32::from((lo >> 4) | (((packed >> 6) & 3) << 4)) - 32;
    }
    scales
}

/// Low and high nibbles of 16 bytes, each through a 16-entry byte table, stored as the 32 values
/// `[low nibbles | high nibbles]`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lookup32(packed: *const u8, table: __m128i, dst: *mut u8) {
    let bytes = _mm_loadu_si128(packed.cast());
    let mask = _mm_set1_epi8(15);
    let low = _mm_shuffle_epi8(table, _mm_and_si128(bytes, mask));
    let high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), mask));
    _mm_storeu_si128(dst.cast(), low);
    _mm_storeu_si128(dst.add(16).cast(), high);
}

/// Decode one weight row of `kind` into `q` (`cols` bytes), the group scales and minimums.
///
/// # Safety
/// The CPU must support AVX2 and F16C; `raw` holds one row (`cols / block` blocks) of `kind`;
/// `q` has `cols` bytes and `scale`/`min` one entry per group.
#[target_feature(enable = "avx2,f16c")]
unsafe fn decode_row(
    kind: u32,
    raw: &[u8],
    cols: usize,
    q: &mut [u8],
    scale: &mut [f32],
    min: &mut [f32],
) {
    min.fill(0.0);
    match kind {
        Q8_0 => {
            for b in 0..cols / 32 {
                let blk = &raw[b * 34..b * 34 + 34];
                scale[b] = half(blk, 0);
                q[b * 32..b * 32 + 32].copy_from_slice(&blk[2..34]);
            }
        }
        Q4_0 | Q4_1 | Q5_0 | Q5_1 => {
            let width = match kind {
                Q4_0 => 18,
                Q4_1 => 20,
                Q5_0 => 22,
                _ => 24,
            };
            let five = kind == Q5_0 || kind == Q5_1;
            let offset: i32 = match kind {
                Q4_0 => 8,
                Q5_0 => 16,
                _ => 0,
            };
            for b in 0..cols / 32 {
                let blk = &raw[b * width..b * width + width];
                scale[b] = half(blk, 0);
                let (qs, qh) = match kind {
                    Q4_0 => (&blk[2..18], 0u32),
                    Q4_1 => (&blk[4..20], 0),
                    Q5_0 => (
                        &blk[6..22],
                        u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]),
                    ),
                    _ => (
                        &blk[8..24],
                        u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]),
                    ),
                };
                let dst = &mut q[b * 32..b * 32 + 32];
                for j in 0..16 {
                    let mut low = i32::from(qs[j] & 15);
                    let mut high = i32::from(qs[j] >> 4);
                    if five {
                        low |= (((qh >> j) & 1) as i32) << 4;
                        high |= (((qh >> (j + 16)) & 1) as i32) << 4;
                    }
                    dst[j] = (low - offset) as u8;
                    dst[16 + j] = (high - offset) as u8;
                }
                if kind == Q4_1 || kind == Q5_1 {
                    // weight = d * q + m = scale * q - min with min = -m
                    min[b] = -half(blk, 2);
                }
            }
        }
        IQ4NL => {
            let table = _mm_loadu_si128(IQ4_VALUES.as_ptr().cast());
            for b in 0..cols / 32 {
                let blk = &raw[b * 18..b * 18 + 18];
                scale[b] = half(blk, 0);
                lookup32(blk.as_ptr().add(2), table, q.as_mut_ptr().add(b * 32));
            }
        }
        MXFP4 => {
            let table = _mm_loadu_si128(MXFP4_VALUES.as_ptr().cast());
            for b in 0..cols / 32 {
                let blk = &raw[b * 17..b * 17 + 17];
                scale[b] = e8m0_half(blk[0]);
                lookup32(blk.as_ptr().add(1), table, q.as_mut_ptr().add(b * 32));
            }
        }
        IQ4XS => {
            let table = _mm_loadu_si128(IQ4_VALUES.as_ptr().cast());
            for sb in 0..cols / 256 {
                let blk = &raw[sb * 136..sb * 136 + 136];
                let d = half(blk, 0);
                let scales_high = u16::from_le_bytes([blk[2], blk[3]]);
                for g in 0..8 {
                    let low = (blk[4 + g / 2] >> (4 * (g % 2))) & 15;
                    let ls = i32::from(low) | ((i32::from(scales_high >> (2 * g)) & 3) << 4);
                    scale[sb * 8 + g] = d * (ls - 32) as f32;
                    lookup32(
                        blk.as_ptr().add(8 + g * 16),
                        table,
                        q.as_mut_ptr().add(sb * 256 + g * 32),
                    );
                }
            }
        }
        Q2K => {
            for sb in 0..cols / 256 {
                let blk = &raw[sb * 84..sb * 84 + 84];
                let (d, dmin) = (half(blk, 80), half(blk, 82));
                for j in 0..16 {
                    scale[sb * 16 + j] = d * f32::from(blk[j] & 15);
                    min[sb * 16 + j] = dmin * f32::from(blk[j] >> 4);
                }
                for gg in 0..8 {
                    let qs = &blk[16 + 32 * (gg / 4)..16 + 32 * (gg / 4) + 32];
                    let shift = 2 * (gg % 4);
                    let dst = &mut q[sb * 256 + gg * 32..sb * 256 + gg * 32 + 32];
                    for l in 0..32 {
                        dst[l] = (qs[l] >> shift) & 3;
                    }
                }
            }
        }
        Q3K => {
            for sb in 0..cols / 256 {
                let blk = &raw[sb * 110..sb * 110 + 110];
                let d = half(blk, 108);
                let scales = q3_scales(blk);
                for j in 0..16 {
                    scale[sb * 16 + j] = d * scales[j] as f32;
                }
                for gg in 0..8 {
                    let qs = &blk[32 + 32 * (gg / 4)..32 + 32 * (gg / 4) + 32];
                    let shift = 2 * (gg % 4);
                    let dst = &mut q[sb * 256 + gg * 32..sb * 256 + gg * 32 + 32];
                    for l in 0..32 {
                        let value =
                            i32::from((qs[l] >> shift) & 3) | (i32::from((blk[l] >> gg) & 1) << 2);
                        dst[l] = (value - 4) as u8;
                    }
                }
            }
        }
        Q4K | Q5K => {
            let width = if kind == Q5K { 176 } else { 144 };
            for sb in 0..cols / 256 {
                let blk = &raw[sb * width..sb * width + width];
                let (d, dmin) = (half(blk, 0), half(blk, 2));
                let (scales, mins) = unpack_scales(blk);
                for g in 0..8 {
                    scale[sb * 8 + g] = d * f32::from((scales >> (8 * g)) as u8);
                    min[sb * 8 + g] = dmin * f32::from((mins >> (8 * g)) as u8);
                }
                let (qs, qh) = if kind == Q5K {
                    (&blk[48..176], &blk[16..48])
                } else {
                    (&blk[16..144], &blk[0..0])
                };
                for p in 0..4 {
                    let packed = &qs[32 * p..32 * p + 32];
                    let base = sb * 256 + 64 * p;
                    for l in 0..32 {
                        let mut low = packed[l] & 15;
                        let mut high = packed[l] >> 4;
                        if kind == Q5K {
                            low |= ((qh[l] >> (2 * p)) & 1) << 4;
                            high |= ((qh[l] >> (2 * p + 1)) & 1) << 4;
                        }
                        q[base + l] = low;
                        q[base + 32 + l] = high;
                    }
                }
            }
        }
        Q6K => {
            for sb in 0..cols / 256 {
                let blk = &raw[sb * 210..sb * 210 + 210];
                let d = half(blk, 208);
                for j in 0..16 {
                    scale[sb * 16 + j] = d * f32::from(blk[192 + j] as i8);
                }
                for gg in 0..8 {
                    let (n, sub) = (gg / 4, gg % 4);
                    let low_bytes = &blk[n * 64 + (sub % 2) * 32..n * 64 + (sub % 2) * 32 + 32];
                    let high_bytes = &blk[128 + n * 32..128 + n * 32 + 32];
                    let dst = &mut q[sb * 256 + gg * 32..sb * 256 + gg * 32 + 32];
                    for l in 0..32 {
                        let low = if sub < 2 {
                            low_bytes[l] & 15
                        } else {
                            low_bytes[l] >> 4
                        };
                        let high = (high_bytes[l] >> (2 * sub)) & 3;
                        dst[l] = ((i32::from(low) | (i32::from(high) << 4)) - 32) as u8;
                    }
                }
            }
        }
        _ => unreachable!("no AMX panel decoder for GGUF type {kind}"),
    }
}

/// Decode `valid` (1..=16) rows starting at row `first` of a `kind` weight matrix with `cols`
/// columns into `panel`; the remaining panel rows become zero (value 0, scale 0, minimum 0).
///
/// # Safety
/// The CPU must support AVX2 and F16C; `w` holds complete rows of `kind` through row
/// `first + valid`; `panel` was made for `cols` and the group size of `kind`.
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn decode_panel(
    kind: u32,
    w: &[u8],
    cols: usize,
    first: usize,
    valid: usize,
    panel: &mut AmxPanel,
) {
    let info = kind_info(kind).expect("format has an AMX decoder");
    debug_assert!((1..=PANEL_ROWS).contains(&valid));
    debug_assert_eq!(panel.cols, cols);
    debug_assert_eq!(panel.group, info.group);
    let row_bytes = cols / info.block * info.bytes;
    let groups = panel.groups();
    // Take the per-row scratch out of the panel to borrow it next to `q`.
    let mut row_scale = std::mem::take(&mut panel.row_scale);
    let mut row_min = std::mem::take(&mut panel.row_min);
    for r in 0..valid {
        let raw = &w[(first + r) * row_bytes..(first + r + 1) * row_bytes];
        decode_row(
            kind,
            raw,
            cols,
            &mut panel.q[r * cols..(r + 1) * cols],
            &mut row_scale,
            &mut row_min,
        );
        for g in 0..groups {
            panel.scale[g * PANEL_ROWS + r] = row_scale[g];
            panel.min[g * PANEL_ROWS + r] = row_min[g];
        }
    }
    // Rows past `valid`: zero weights, zero scale, so they contribute nothing.
    for r in valid..PANEL_ROWS {
        panel.q[r * cols..(r + 1) * cols].fill(0);
        for g in 0..groups {
            panel.scale[g * PANEL_ROWS + r] = 0.0;
            panel.min[g * PANEL_ROWS + r] = 0.0;
        }
    }
    panel.row_scale = row_scale;
    panel.row_min = row_min;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{random_weights, Lcg};

    fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c")
    }

    const KINDS: [u32; 13] = [
        Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, IQ4NL, MXFP4, Q2K, Q3K, Q4K, Q5K, Q6K, IQ4XS,
    ];

    #[test]
    fn kind_table_matches_the_packed_layouts() {
        for kind in KINDS {
            let info = kind_info(kind).unwrap();
            let (block, bytes) = crate::cpu_selftest::layout(kind);
            assert_eq!((info.block, info.bytes), (block, bytes), "kind {kind}");
            assert!(info.group == 16 || info.group == 32);
            assert!(info.block.is_multiple_of(info.group));
            assert_eq!(info.signed, !info.has_min || kind == Q8_0, "kind {kind}");
            assert!(act_kind(kind).is_some());
        }
        assert!(kind_info(0).is_none() && kind_info(30).is_none() && kind_info(16).is_none());
    }

    /// Every decoded weight (`scale * q - min`) against the scalar dequantization.
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn decoded_panels_reproduce_the_dequantized_weights() {
        if !has_avx2() {
            return;
        }
        let mut rng = Lcg(0x14_a5_5e);
        for kind in KINDS {
            let info = kind_info(kind).unwrap();
            for (rows, first, valid) in [
                (16usize, 0usize, 16usize),
                (21, 4, 13),
                (5, 0, 5),
                (17, 16, 1),
            ] {
                let cols = if info.block == 256 { 768 } else { 352 };
                let w = random_weights(kind, rows, cols, &mut rng);
                let mut panel = AmxPanel::new(cols, info.group);
                // Poison the panel to check that rows past `valid` are cleared.
                panel.q.fill(0x55);
                panel.scale.fill(7.0);
                panel.min.fill(7.0);
                unsafe { decode_panel(kind, &w, cols, first, valid, &mut panel) };
                let row_bytes = cols / info.block * info.bytes;
                for r in 0..PANEL_ROWS {
                    if r >= valid {
                        assert!(panel.q[r * cols..(r + 1) * cols].iter().all(|&b| b == 0));
                        for g in 0..panel.groups() {
                            assert_eq!(panel.scale[g * 16 + r], 0.0);
                            assert_eq!(panel.min[g * 16 + r], 0.0);
                        }
                        continue;
                    }
                    let want = crate::compute::dequantize(
                        &w[(first + r) * row_bytes..(first + r + 1) * row_bytes],
                        kind,
                        cols,
                    )
                    .expect("dequantize");
                    for k in 0..cols {
                        let g = k / info.group;
                        let q = panel.q[r * cols + k];
                        let q = if info.signed {
                            f32::from(q as i8)
                        } else {
                            f32::from(q)
                        };
                        let have = panel.scale[g * 16 + r] * q - panel.min[g * 16 + r];
                        let tolerance = 2e-5
                            * (panel.scale[g * 16 + r] * q)
                                .abs()
                                .max(panel.min[g * 16 + r].abs())
                            + 1e-7;
                        assert!(
                            (have - want[k]).abs() <= tolerance,
                            "kind {kind} row {r} col {k}: have {have} want {}",
                            want[k]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn signed_formats_stay_inside_the_int8_range() {
        if !has_avx2() {
            return;
        }
        let mut rng = Lcg(77);
        for kind in KINDS {
            let info = kind_info(kind).unwrap();
            let cols = 512;
            let w = random_weights(kind, 16, cols, &mut rng);
            let mut panel = AmxPanel::new(cols, info.group);
            unsafe { decode_panel(kind, &w, cols, 0, 16, &mut panel) };
            if info.signed {
                // Q4_0 -8..=7, Q5_0 -16..=15, Q3_K -4..=3, Q6_K -32..=31, codebooks and Q8_0 full.
                let limit = match kind {
                    Q4_0 => (-8, 7),
                    Q5_0 => (-16, 15),
                    Q3K => (-4, 3),
                    Q6K => (-32, 31),
                    _ => (-128, 127),
                };
                assert!(
                    panel
                        .q
                        .iter()
                        .all(|&b| (limit.0..=limit.1).contains(&i32::from(b as i8))),
                    "kind {kind}"
                );
            } else {
                let limit = match kind {
                    Q2K => 3,
                    Q4_1 | Q4K => 15,
                    _ => 31,
                };
                assert!(
                    panel.q.iter().all(|&b| i32::from(b) <= limit),
                    "kind {kind}"
                );
            }
        }
    }
}
