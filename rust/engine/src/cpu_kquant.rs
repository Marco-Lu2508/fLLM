//! Q4_K, Q5_K and Q6_K projections: unsigned weights go straight into one integer dot per 32 values
//! (`vpdpbusd` with AVX-VNNI or AVX-512 VNNI, otherwise `maddubs`; 256- or 512-bit), sub-block
//! scales are applied in the integer domain
//! and each 256-value block is converted to float once. This is the same arithmetic organisation
//! as the reference K-quant dot products, so there is no per-sub-block float conversion.
//!
//! Activations arrive as the 32-value records of `cpu_quantized::X`; eight consecutive records
//! form one Q8_K block that shares a scale.
//!
//! Q2_K, Q3_K and IQ4_XS (also Q8_K activations) follow the same panel structure.
use crate::cpu_quantized::X;
use crate::cpu_schedule::Rows;
use rayon::prelude::*;
use std::arch::x86_64::*;

const Q2K: u32 = 10;
const Q3K: u32 = 11;
const Q4K: u32 = 12;
const Q5K: u32 = 13;
const Q6K: u32 = 14;
const IQ4XS: u32 = 23;
const IQ4NL: u32 = 20;

/// Non-linear 4-bit codebook of IQ4_NL and IQ4_XS (`KVALUES_IQ4NL` in the `quant_iq` tables).
pub(crate) const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const fn block_bytes(kind: u32) -> usize {
    match kind {
        Q2K => 84,
        Q3K => 110,
        IQ4XS => 136,
        Q4K => 144,
        Q5K => 176,
        _ => 210,
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
}

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn half(raw: &[u8], at: usize) -> f32 {
    crate::cpu_kernels::x86::half_to_float(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

/// Unsigned-by-signed 8-bit dot, four products per 32-bit lane, added to `init`.
#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot_vnni(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(init, u, s)
}

/// `dot_vnni` in its EVEX encoding (avx512vnni + avx512vl): the same 256-bit `vpdpbusd` for CPUs
/// with AVX-512 VNNI but without AVX-VNNI (Ice Lake, Tiger Lake, Zen 4).
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
unsafe fn dot_vnni_evex(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
    _mm256_dpbusd_epi32(init, u, s)
}

/// Same contract on plain AVX2. Weights are at most 63, so a `maddubs` pair sums to at most
/// 2 * 63 * 127 = 16002 and never saturates.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
    _mm256_add_epi32(
        _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)),
        init,
    )
}

/// `-32 * (sum of the four activations in each lane)`: the offset Q6_K weights carry, folded into
/// the dot product as its initial value.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q6_lane_bias(qx: __m256i) -> __m256i {
    let pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), qx);
    let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32::<5>(sums))
}

/// Eight scales and eight minimums of a Q4_K / Q5_K block (bytes 4..16), as two little-endian
/// 64-bit words.
#[inline]
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

/// Four copies of a signed 16-bit scale packed into one 64-bit word.
#[inline]
fn pair_scale(scale: i32) -> i64 {
    (u64::from(scale as i16 as u16) * 0x0001_0001_0001_0001) as i64
}

/// The eight 32-value activation sums of one Q8_K block as 16-bit words. `init` stores -32 times
/// each half sum, and a group sum is at most 32 * 127 in magnitude.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn group_sums16(records: &[X]) -> __m128i {
    let sum = |g: usize| (-(records[g].init[0] + records[g].init[1]) >> 5) as i16;
    _mm_setr_epi16(
        sum(0),
        sum(1),
        sum(2),
        sum(3),
        sum(4),
        sum(5),
        sum(6),
        sum(7),
    )
}

/// Q4_K/Q5_K minimum correction `sum_g min_g * group_sum_g` of one row block, as four exact
/// pair sums in the low half of the vector (the high half is zero). One `madd` replaces the
/// former 32-bit `mullo`, which costs two uops and ten cycles of latency on recent Intel cores.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn min_correction(mins_word: u64, sums: __m128i) -> __m256 {
    let mins = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(mins_word as i64));
    _mm256_zextps128_ps256(_mm_cvtepi32_ps(_mm_madd_epi16(mins, sums)))
}

/// Byte shuffle that selects 16-bit scale words `2 * pair` and `2 * pair + 1`, four copies each.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn scale_selector(pair: usize) -> __m128i {
    let low = (4 * pair) as i8;
    let high = low + 2;
    _mm_setr_epi8(
        low,
        low + 1,
        low,
        low + 1,
        low,
        low + 1,
        low,
        low + 1,
        high,
        high + 1,
        high,
        high + 1,
        high,
        high + 1,
        high,
        high + 1,
    )
}

/// One 32-value Q6_K group as unsigned 6-bit values plus its two 16-value scales.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q6_group(raw: &[u8], group: usize) -> (__m256i, i32, i32) {
    let (n, sub) = (group / 4, group % 4);
    let low = _mm256_loadu_si256(raw.as_ptr().add(n * 64 + (sub % 2) * 32).cast());
    let low = _mm256_and_si256(
        if sub < 2 {
            low
        } else {
            _mm256_srli_epi16(low, 4)
        },
        _mm256_set1_epi8(15),
    );
    let high = _mm256_loadu_si256(raw.as_ptr().add(128 + n * 32).cast());
    let high = _mm256_and_si256(
        _mm256_srl_epi16(high, _mm_cvtsi32_si128((2 * sub) as i32)),
        _mm256_set1_epi8(3),
    );
    let at = 192 + n * 8 + sub * 2;
    (
        _mm256_or_si256(low, _mm256_slli_epi16(high, 4)),
        i32::from(raw[at] as i8),
        i32::from(raw[at + 1] as i8),
    )
}

macro_rules! kernel {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Compute `rows` (1..=4) consecutive output rows starting at `first` for every token.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` must hold `first + rows` complete rows
        /// of `nb / 8` blocks and `x` must hold `batch * nb` activation records.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            kind: u32,
            first: usize,
            rows: usize,
            x: &[X],
            nb: usize,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            macro_rules! by_rows {
                ($kind:literal) => {{
                    let mut token = 0;
                    while token < batch {
                        let tokens = (batch - token).min(2);
                        match (rows, tokens) {
                            (1, 1) => $body::<$kind, 1, 1>(w, first, x, nb, batch, token, out),
                            (2, 1) => $body::<$kind, 2, 1>(w, first, x, nb, batch, token, out),
                            (3, 1) => $body::<$kind, 3, 1>(w, first, x, nb, batch, token, out),
                            (4, 1) => $body::<$kind, 4, 1>(w, first, x, nb, batch, token, out),
                            (1, _) => $body::<$kind, 1, 2>(w, first, x, nb, batch, token, out),
                            (2, _) => $body::<$kind, 2, 2>(w, first, x, nb, batch, token, out),
                            (3, _) => $body::<$kind, 3, 2>(w, first, x, nb, batch, token, out),
                            (4, _) => $body::<$kind, 4, 2>(w, first, x, nb, batch, token, out),
                            _ => unreachable!("panel has 1 to 4 rows"),
                        }
                        token += tokens;
                    }
                }};
            }
            match kind {
                Q4K => by_rows!(12),
                Q5K => by_rows!(13),
                Q6K => by_rows!(14),
                _ => unreachable!("not a K quantization"),
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            w: &[u8],
            first: usize,
            x: &[X],
            nb: usize,
            _batch: usize,
            token: usize,
            out: &mut Rows<'_>,
        ) {
            let blocks = nb / 8;
            let width = block_bytes(KIND);
            let stride = blocks * width;
            let low4 = _mm256_set1_epi8(15);
            let zero = _mm256_setzero_si256();
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            for blk in 0..blocks {
                let xs: [&[X]; T] = std::array::from_fn(|t| {
                    &x[(token + t) * nb + blk * 8..(token + t) * nb + blk * 8 + 8]
                });
                // Per-token values shared by every row of the panel.
                let mut group_sums = [_mm_setzero_si128(); T];
                let mut lane_bias = [[zero; 8]; T];
                for t in 0..T {
                    if KIND == Q6K {
                        for g in 0..8 {
                            lane_bias[t][g] =
                                q6_lane_bias(_mm256_load_si256(xs[t][g].q.as_ptr().cast()));
                        }
                    } else {
                        group_sums[t] = group_sums16(xs[t]);
                    }
                }
                for r in 0..R {
                    let start = (first + r) * stride + blk * width;
                    let raw = &w[start..start + width];
                    let mut sumi = [zero; T];
                    let d;
                    let dmin;
                    let mut mins_word = 0;
                    if KIND == Q6K {
                        d = half(raw, 208);
                        dmin = 0.0;
                        for pair in 0..4usize {
                            let (q1, a0, a1) = q6_group(raw, 2 * pair);
                            let (q2, b0, b1) = q6_group(raw, 2 * pair + 1);
                            let scales = _mm256_set_epi64x(
                                pair_scale(b1),
                                pair_scale(a1),
                                pair_scale(b0),
                                pair_scale(a0),
                            );
                            for t in 0..T {
                                let x1 = _mm256_load_si256(xs[t][2 * pair].q.as_ptr().cast());
                                let x2 = _mm256_load_si256(xs[t][2 * pair + 1].q.as_ptr().cast());
                                // Biased lane sums stay within 16 bits (4 * 32 * 127), so the two
                                // groups pack into one vector and one madd applies both scales.
                                let p1 = $dot(lane_bias[t][2 * pair], q1, x1);
                                let p2 = $dot(lane_bias[t][2 * pair + 1], q2, x2);
                                let scaled = _mm256_madd_epi16(_mm256_packs_epi32(p1, p2), scales);
                                sumi[t] = _mm256_add_epi32(sumi[t], scaled);
                            }
                        }
                    } else {
                        d = half(raw, 0);
                        dmin = half(raw, 2);
                        let scales;
                        (scales, mins_word) = unpack_scales(raw);
                        let qs = if KIND == Q5K { 48 } else { 16 };
                        let scales16 = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(scales as i64));
                        let high_bits = if KIND == Q5K {
                            _mm256_loadu_si256(raw.as_ptr().add(16).cast())
                        } else {
                            zero
                        };
                        for pair in 0..4usize {
                            let packed =
                                _mm256_loadu_si256(raw.as_ptr().add(qs + pair * 32).cast());
                            let mut lo = _mm256_and_si256(packed, low4);
                            let mut hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low4);
                            if KIND == Q5K {
                                let one = _mm256_set1_epi8(1);
                                let lo_bit = _mm256_and_si256(
                                    _mm256_srl_epi16(
                                        high_bits,
                                        _mm_cvtsi32_si128((2 * pair) as i32),
                                    ),
                                    one,
                                );
                                let hi_bit = _mm256_and_si256(
                                    _mm256_srl_epi16(
                                        high_bits,
                                        _mm_cvtsi32_si128((2 * pair + 1) as i32),
                                    ),
                                    one,
                                );
                                lo = _mm256_or_si256(lo, _mm256_slli_epi16(lo_bit, 4));
                                hi = _mm256_or_si256(hi, _mm256_slli_epi16(hi_bit, 4));
                            }
                            // words: scale of the low sub-block x4, scale of the high one x4
                            let scale_vector = _mm256_broadcastsi128_si256(_mm_shuffle_epi8(
                                scales16,
                                scale_selector(pair),
                            ));
                            for t in 0..T {
                                let x_lo = _mm256_load_si256(xs[t][2 * pair].q.as_ptr().cast());
                                let x_hi = _mm256_load_si256(xs[t][2 * pair + 1].q.as_ptr().cast());
                                // Lane sums are at most 4 * 31 * 127, so both sub-blocks pack into
                                // one vector and a single madd applies their scales.
                                let scaled = _mm256_madd_epi16(
                                    _mm256_packs_epi32($dot(zero, lo, x_lo), $dot(zero, hi, x_hi)),
                                    scale_vector,
                                );
                                sumi[t] = _mm256_add_epi32(sumi[t], scaled);
                            }
                        }
                    }
                    for t in 0..T {
                        let scale = _mm256_set1_ps(d * xs[t][0].d);
                        acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[t]), scale, acc[r][t]);
                        if KIND != Q6K {
                            acc[r][t] = _mm256_fnmadd_ps(
                                min_correction(mins_word, group_sums[t]),
                                _mm256_set1_ps(dmin * xs[t][0].d),
                                acc[r][t],
                            );
                        }
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = hsum(acc[r][t]);
                }
            }
        }
    };
}

kernel!(
    panel_vnni,
    panel_body_vnni,
    "avx2,fma,f16c,avxvnni",
    dot_vnni
);
kernel!(panel_avx2, panel_body_avx2, "avx2,fma,f16c", dot_avx2);

/// Unsigned-by-signed 8-bit dot on 512-bit vectors (AVX-512 VNNI).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot512_vnni(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
    _mm512_dpbusd_epi32(init, u, s)
}

/// Same contract on AVX-512BW; the weight bound of `dot_avx2` applies.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot512_bw(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
    _mm512_add_epi32(
        _mm512_madd_epi16(_mm512_maddubs_epi16(u, s), _mm512_set1_epi16(1)),
        init,
    )
}

/// 512-bit `q6_lane_bias`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn q6_lane_bias512(qx: __m512i) -> __m512i {
    let pairs = _mm512_maddubs_epi16(_mm512_set1_epi8(1), qx);
    let sums = _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
    _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32::<5>(sums))
}

/// Activation records `a` and `b` as the low and high halves of one vector.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn records512(a: &X, b: &X) -> __m512i {
    crate::cpu_kernels::x86::join256(
        _mm256_load_si256(a.q.as_ptr().cast()),
        _mm256_load_si256(b.q.as_ptr().cast()),
    )
}

/// 16-bit lanes 0..16 hold `low`, lanes 16..32 hold `high` (per-half shift counts).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn halves_epi16(low: i16, high: i16) -> __m512i {
    _mm512_mask_blend_epi16(0xffff_0000, _mm512_set1_epi16(low), _mm512_set1_epi16(high))
}

/// Bits `shift..` of each byte of `bits` under `mask`, moved to bit 4 (the high weight bits).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn high_bits512(bits: __m512i, shift: __m512i, mask: __m512i) -> __m512i {
    _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(bits, shift), mask))
}

/// Q4_K/Q5_K scale words for 64-byte load `k`: 128-bit lanes 0,1 select sub-block pair 2k,
/// lanes 2,3 pair 2k + 1 (see `scale_selector`).
#[inline]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn pair_selector512(k: usize) -> __m512i {
    crate::cpu_kernels::x86::join256(
        _mm256_broadcastsi128_si256(scale_selector(2 * k)),
        _mm256_broadcastsi128_si256(scale_selector(2 * k + 1)),
    )
}

/// Q6_K scale words: 128-bit lane L selects 16-value scales L and L + 4 of a half block.
#[inline]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn q6_selector512() -> __m512i {
    let mut bytes = [0i8; 64];
    for (lane, chunk) in bytes.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        for (i, byte) in chunk.iter_mut().enumerate() {
            let word = if i < 8 { lane } else { lane + 4 };
            *byte = (2 * word + i % 2) as i8;
        }
    }
    _mm512_loadu_si512(bytes.as_ptr().cast())
}

// Wide K panels. One 64-byte Q4_K/Q5_K load holds the nibbles of groups (4k, 4k+1) in its low
// half and (4k+2, 4k+3) in its high half, so the low nibbles pair groups [4k | 4k+2] and the
// high nibbles [4k+1 | 4k+3]. One 64-byte Q6_K low-bit load pairs [4n | 4n+1] (low nibbles)
// and [4n+2 | 4n+3] (high nibbles). `packs_epi32` of two such dots works per 128-bit lane, so
// lane L of the packed words holds two groups' four-lane sums; the scale words are arranged
// with one in-lane shuffle to match, and a single madd applies all four groups' scales.
macro_rules! kernel512 {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Compute `rows` (1..=8) consecutive output rows starting at `first` for every token.
        /// Panels of more than four rows run one token at a time.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` must hold `first + rows` complete rows
        /// of `nb / 8` blocks and `x` must hold `batch * nb` activation records.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            kind: u32,
            first: usize,
            rows: usize,
            x: &[X],
            nb: usize,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            macro_rules! by_rows {
                ($kind:literal) => {{
                    let mut token = 0;
                    while token < batch {
                        let tokens = if rows > 4 { 1 } else { (batch - token).min(2) };
                        match (rows, tokens) {
                            (1, 1) => $body::<$kind, 1, 1>(w, first, x, nb, token, out),
                            (2, 1) => $body::<$kind, 2, 1>(w, first, x, nb, token, out),
                            (3, 1) => $body::<$kind, 3, 1>(w, first, x, nb, token, out),
                            (4, 1) => $body::<$kind, 4, 1>(w, first, x, nb, token, out),
                            (5, 1) => $body::<$kind, 5, 1>(w, first, x, nb, token, out),
                            (6, 1) => $body::<$kind, 6, 1>(w, first, x, nb, token, out),
                            (7, 1) => $body::<$kind, 7, 1>(w, first, x, nb, token, out),
                            (8, 1) => $body::<$kind, 8, 1>(w, first, x, nb, token, out),
                            (1, _) => $body::<$kind, 1, 2>(w, first, x, nb, token, out),
                            (2, _) => $body::<$kind, 2, 2>(w, first, x, nb, token, out),
                            (3, _) => $body::<$kind, 3, 2>(w, first, x, nb, token, out),
                            (4, _) => $body::<$kind, 4, 2>(w, first, x, nb, token, out),
                            _ => unreachable!("panel has 1 to 8 rows"),
                        }
                        token += tokens;
                    }
                }};
            }
            match kind {
                Q4K => by_rows!(12),
                Q5K => by_rows!(13),
                Q6K => by_rows!(14),
                _ => unreachable!("not a K quantization"),
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            w: &[u8],
            first: usize,
            x: &[X],
            nb: usize,
            token: usize,
            out: &mut Rows<'_>,
        ) {
            let blocks = nb / 8;
            let width = block_bytes(KIND);
            let stride = blocks * width;
            let low4 = _mm512_set1_epi8(15);
            let zero = _mm512_setzero_si512();
            let pair_selectors = [pair_selector512(0), pair_selector512(1)];
            let q6_selector = q6_selector512();
            let five_shifts = [
                [halves_epi16(0, 2), halves_epi16(1, 3)],
                [halves_epi16(4, 6), halves_epi16(5, 7)],
            ];
            let six_shifts = [halves_epi16(0, 2), halves_epi16(4, 6)];
            let mut acc = [[_mm512_setzero_ps(); T]; R];
            for blk in 0..blocks {
                let xs: [&[X]; T] = std::array::from_fn(|t| {
                    &x[(token + t) * nb + blk * 8..(token + t) * nb + blk * 8 + 8]
                });
                // Activation pairs in the weight layout, shared by every row of the panel.
                let mut xv = [[zero; 4]; T];
                let mut lane_bias = [[zero; 4]; T];
                let mut group_sums = [_mm_setzero_si128(); T];
                for t in 0..T {
                    let r = xs[t];
                    if KIND == Q6K {
                        xv[t] = [
                            records512(&r[0], &r[1]),
                            records512(&r[2], &r[3]),
                            records512(&r[4], &r[5]),
                            records512(&r[6], &r[7]),
                        ];
                        for (bias, &v) in lane_bias[t].iter_mut().zip(&xv[t]) {
                            *bias = q6_lane_bias512(v);
                        }
                    } else {
                        xv[t] = [
                            records512(&r[0], &r[2]),
                            records512(&r[1], &r[3]),
                            records512(&r[4], &r[6]),
                            records512(&r[5], &r[7]),
                        ];
                        group_sums[t] = group_sums16(r);
                    }
                }
                for (r, acc) in acc.iter_mut().enumerate() {
                    let start = (first + r) * stride + blk * width;
                    let raw = &w[start..start + width];
                    let mut sumi = [zero; T];
                    let d;
                    let dmin;
                    let mut mins_word = 0;
                    if KIND == Q6K {
                        d = half(raw, 208);
                        dmin = 0.0;
                        let high = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                            raw.as_ptr().add(128).cast(),
                        ));
                        let high = [
                            high,
                            _mm512_broadcast_i64x4(_mm256_loadu_si256(
                                raw.as_ptr().add(160).cast(),
                            )),
                        ];
                        for (n, high) in high.into_iter().enumerate() {
                            let low = _mm512_loadu_si512(raw.as_ptr().add(n * 64).cast());
                            let three = _mm512_set1_epi8(3);
                            let qa = _mm512_or_si512(
                                _mm512_and_si512(low, low4),
                                high_bits512(high, six_shifts[0], three),
                            );
                            let qb = _mm512_or_si512(
                                _mm512_and_si512(_mm512_srli_epi16::<4>(low), low4),
                                high_bits512(high, six_shifts[1], three),
                            );
                            let scales16 = _mm_cvtepi8_epi16(_mm_loadl_epi64(
                                raw.as_ptr().add(192 + n * 8).cast(),
                            ));
                            let scales =
                                _mm512_shuffle_epi8(_mm512_broadcast_i32x4(scales16), q6_selector);
                            for t in 0..T {
                                // Biased lane sums stay within 16 bits, as on AVX2.
                                let pa = $dot(lane_bias[t][2 * n], qa, xv[t][2 * n]);
                                let pb = $dot(lane_bias[t][2 * n + 1], qb, xv[t][2 * n + 1]);
                                let scaled = _mm512_madd_epi16(_mm512_packs_epi32(pa, pb), scales);
                                sumi[t] = _mm512_add_epi32(sumi[t], scaled);
                            }
                        }
                    } else {
                        d = half(raw, 0);
                        dmin = half(raw, 2);
                        let scales;
                        (scales, mins_word) = unpack_scales(raw);
                        let qs = if KIND == Q5K { 48 } else { 16 };
                        let scales16 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(
                            _mm_cvtsi64_si128(scales as i64),
                        ));
                        let high_bits = if KIND == Q5K {
                            _mm512_broadcast_i64x4(_mm256_loadu_si256(raw.as_ptr().add(16).cast()))
                        } else {
                            zero
                        };
                        for k in 0..2usize {
                            let packed = _mm512_loadu_si512(raw.as_ptr().add(qs + k * 64).cast());
                            let mut lo = _mm512_and_si512(packed, low4);
                            let mut hi = _mm512_and_si512(_mm512_srli_epi16::<4>(packed), low4);
                            if KIND == Q5K {
                                let one = _mm512_set1_epi8(1);
                                lo = _mm512_or_si512(
                                    lo,
                                    high_bits512(high_bits, five_shifts[k][0], one),
                                );
                                hi = _mm512_or_si512(
                                    hi,
                                    high_bits512(high_bits, five_shifts[k][1], one),
                                );
                            }
                            let scale_vector = _mm512_shuffle_epi8(scales16, pair_selectors[k]);
                            for t in 0..T {
                                let scaled = _mm512_madd_epi16(
                                    _mm512_packs_epi32(
                                        $dot(zero, lo, xv[t][2 * k]),
                                        $dot(zero, hi, xv[t][2 * k + 1]),
                                    ),
                                    scale_vector,
                                );
                                sumi[t] = _mm512_add_epi32(sumi[t], scaled);
                            }
                        }
                    }
                    for t in 0..T {
                        let scale = _mm512_set1_ps(d * xs[t][0].d);
                        acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi[t]), scale, acc[t]);
                        if KIND != Q6K {
                            acc[t] = _mm512_fnmadd_ps(
                                _mm512_zextps256_ps512(min_correction(mins_word, group_sums[t])),
                                _mm512_set1_ps(dmin * xs[t][0].d),
                                acc[t],
                            );
                        }
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = _mm512_reduce_add_ps(acc[r][t]);
                }
            }
        }
    };
}

kernel512!(
    panel512_vnni,
    panel512_body_vnni,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
    dot512_vnni
);
kernel512!(
    panel512_bw,
    panel512_body_bw,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
    dot512_bw
);

/// One 32-value Q2_K group (values `(qs[32n + j] >> 2s) & 3` for group `4n + s`) plus the
/// 4-bit scales of its two 16-value halves.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q2_group(raw: &[u8], group: usize) -> (__m256i, i32, i32) {
    let q = _mm256_loadu_si256(raw.as_ptr().add(16 + group / 4 * 32).cast());
    let q = _mm256_and_si256(
        _mm256_srl_epi16(q, _mm_cvtsi32_si128((2 * (group % 4)) as i32)),
        _mm256_set1_epi8(3),
    );
    (
        q,
        i32::from(raw[2 * group] & 15),
        i32::from(raw[2 * group + 1] & 15),
    )
}

/// The sixteen signed 6-bit Q3_K scales, in the order of `quant_extra`.
#[inline]
fn q3_scales(raw: &[u8]) -> [i32; 16] {
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

/// One 32-value Q3_K group as `low | high << 2` in 0..8, i.e. the signed value plus 4: Q3_K
/// subtracts 4 where the `hmask` bit is clear.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q3_group(raw: &[u8], group: usize, scales: &[i32; 16]) -> (__m256i, i32, i32) {
    let low = _mm256_loadu_si256(raw.as_ptr().add(32 + group / 4 * 32).cast());
    let low = _mm256_and_si256(
        _mm256_srl_epi16(low, _mm_cvtsi32_si128((2 * (group % 4)) as i32)),
        _mm256_set1_epi8(3),
    );
    let high = _mm256_loadu_si256(raw.as_ptr().cast());
    let high = _mm256_and_si256(
        _mm256_srl_epi16(high, _mm_cvtsi32_si128(group as i32)),
        _mm256_set1_epi8(1),
    );
    (
        _mm256_or_si256(low, _mm256_slli_epi16(high, 2)),
        scales[2 * group],
        scales[2 * group + 1],
    )
}

/// `-4 * (sum of the four activations in each lane)`: the Q3_K offset as the dot's initial value.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q3_lane_bias(qx: __m256i) -> __m256i {
    let pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), qx);
    let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32::<2>(sums))
}

/// One 32-value IQ4_XS group as signed codebook values plus its signed 6-bit scale.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq4xs_group(raw: &[u8], group: usize, table: __m256i) -> (__m256i, i32) {
    let packed = _mm_loadu_si128(raw.as_ptr().add(8 + group * 16).cast());
    let low = _mm_and_si128(packed, _mm_set1_epi8(15));
    let high = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
    let index = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    let scales_high = u16::from_le_bytes([raw[2], raw[3]]);
    let scale_low = (raw[4 + group / 2] >> (4 * (group % 2))) & 15;
    let scale = i32::from(scale_low) | ((i32::from(scales_high >> (2 * group)) & 3) << 4);
    (_mm256_shuffle_epi8(table, index), scale - 32)
}

/// Signed weights by signed activations, `scale` applied to every `maddubs` pair in `madd`.
/// |w|, |x| <= 127, so a pair sum is at most 32258 and cannot saturate.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_signed_scaled(w: __m256i, x: __m256i, scale: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)),
        scale,
    )
}

// Q2_K, Q3_K and IQ4_XS share the K panel structure. Q2_K and Q3_K have 16-value scales and use
// the pack + madd scale application of Q6_K; Q2_K's minimums meet the 16-value activation sums
// in one madd per block. IQ4_XS has signed codebook weights and one scale per 32 values, which
// the sign-trick dot applies directly in its madd.
macro_rules! kernel_extra {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Compute `rows` (1..=4) consecutive Q2_K, Q3_K or IQ4_XS output rows starting at
        /// `first` for every token.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` must hold `first + rows` complete rows
        /// of `nb / 8` blocks and `x` must hold `batch * nb` activation records.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            kind: u32,
            first: usize,
            rows: usize,
            x: &[X],
            nb: usize,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            macro_rules! by_rows {
                ($kind:literal) => {{
                    let mut token = 0;
                    while token < batch {
                        let tokens = (batch - token).min(2);
                        match (rows, tokens) {
                            (1, 1) => $body::<$kind, 1, 1>(w, first, x, nb, token, out),
                            (2, 1) => $body::<$kind, 2, 1>(w, first, x, nb, token, out),
                            (3, 1) => $body::<$kind, 3, 1>(w, first, x, nb, token, out),
                            (4, 1) => $body::<$kind, 4, 1>(w, first, x, nb, token, out),
                            (1, _) => $body::<$kind, 1, 2>(w, first, x, nb, token, out),
                            (2, _) => $body::<$kind, 2, 2>(w, first, x, nb, token, out),
                            (3, _) => $body::<$kind, 3, 2>(w, first, x, nb, token, out),
                            (4, _) => $body::<$kind, 4, 2>(w, first, x, nb, token, out),
                            _ => unreachable!("panel has 1 to 4 rows"),
                        }
                        token += tokens;
                    }
                }};
            }
            match kind {
                Q2K => by_rows!(10),
                Q3K => by_rows!(11),
                IQ4XS => by_rows!(23),
                _ => unreachable!("not a Q2_K, Q3_K or IQ4_XS projection"),
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            w: &[u8],
            first: usize,
            x: &[X],
            nb: usize,
            token: usize,
            out: &mut Rows<'_>,
        ) {
            let blocks = nb / 8;
            let width = block_bytes(KIND);
            let stride = blocks * width;
            let zero = _mm256_setzero_si256();
            let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(IQ4_VALUES.as_ptr().cast()));
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            for blk in 0..blocks {
                let xs: [&[X]; T] = std::array::from_fn(|t| {
                    &x[(token + t) * nb + blk * 8..(token + t) * nb + blk * 8 + 8]
                });
                // Per-token values shared by every row of the panel.
                let mut lane_bias = [[zero; 8]; T];
                let mut half_sums = [zero; T];
                for t in 0..T {
                    if KIND == Q3K {
                        for g in 0..8 {
                            lane_bias[t][g] =
                                q3_lane_bias(_mm256_load_si256(xs[t][g].q.as_ptr().cast()));
                        }
                    } else if KIND == Q2K {
                        // 16-value activation sums; `init` stores -32 * each half sum.
                        let s = |i: usize| (-xs[t][i / 2].init[i % 2] >> 5) as i16;
                        half_sums[t] = _mm256_setr_epi16(
                            s(0),
                            s(1),
                            s(2),
                            s(3),
                            s(4),
                            s(5),
                            s(6),
                            s(7),
                            s(8),
                            s(9),
                            s(10),
                            s(11),
                            s(12),
                            s(13),
                            s(14),
                            s(15),
                        );
                    }
                }
                for r in 0..R {
                    let start = (first + r) * stride + blk * width;
                    let raw = &w[start..start + width];
                    let mut sumi = [zero; T];
                    let d;
                    if KIND == IQ4XS {
                        d = half(raw, 0);
                        for g in 0..8 {
                            let (q, scale) = iq4xs_group(raw, g, table);
                            let scale = _mm256_set1_epi16(scale as i16);
                            for t in 0..T {
                                let qx = _mm256_load_si256(xs[t][g].q.as_ptr().cast());
                                sumi[t] =
                                    _mm256_add_epi32(sumi[t], dot_signed_scaled(q, qx, scale));
                            }
                        }
                    } else {
                        d = half(raw, if KIND == Q2K { 80 } else { 108 });
                        let q3 = if KIND == Q3K { q3_scales(raw) } else { [0; 16] };
                        for pair in 0..4usize {
                            let ((q1, a0, a1), (q2, b0, b1)) = if KIND == Q2K {
                                (q2_group(raw, 2 * pair), q2_group(raw, 2 * pair + 1))
                            } else {
                                (
                                    q3_group(raw, 2 * pair, &q3),
                                    q3_group(raw, 2 * pair + 1, &q3),
                                )
                            };
                            let scales = _mm256_set_epi64x(
                                pair_scale(b1),
                                pair_scale(a1),
                                pair_scale(b0),
                                pair_scale(a0),
                            );
                            for t in 0..T {
                                let x1 = _mm256_load_si256(xs[t][2 * pair].q.as_ptr().cast());
                                let x2 = _mm256_load_si256(xs[t][2 * pair + 1].q.as_ptr().cast());
                                // Lane sums are at most 4 * 4 * 127 (Q3_K, biased), so both groups
                                // pack into one vector and a single madd applies their scales.
                                let (i1, i2) = if KIND == Q3K {
                                    (lane_bias[t][2 * pair], lane_bias[t][2 * pair + 1])
                                } else {
                                    (zero, zero)
                                };
                                let p1 = $dot(i1, q1, x1);
                                let p2 = $dot(i2, q2, x2);
                                let scaled = _mm256_madd_epi16(_mm256_packs_epi32(p1, p2), scales);
                                sumi[t] = _mm256_add_epi32(sumi[t], scaled);
                            }
                        }
                    }
                    for t in 0..T {
                        let scale = _mm256_set1_ps(d * xs[t][0].d);
                        acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[t]), scale, acc[r][t]);
                        if KIND == Q2K {
                            // minimum correction: sum_i min_i * (sum of sub-block i's activations)
                            let packed = _mm_loadu_si128(raw.as_ptr().cast());
                            let mins = _mm256_cvtepu8_epi16(_mm_and_si128(
                                _mm_srli_epi16(packed, 4),
                                _mm_set1_epi8(15),
                            ));
                            let product = _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, half_sums[t]));
                            acc[r][t] = _mm256_fnmadd_ps(
                                product,
                                _mm256_set1_ps(half(raw, 82) * xs[t][0].d),
                                acc[r][t],
                            );
                        }
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = hsum(acc[r][t]);
                }
            }
        }
    };
}

kernel_extra!(
    panel_extra_vnni,
    panel_extra_body_vnni,
    "avx2,fma,f16c,avxvnni",
    dot_vnni
);
kernel_extra!(
    panel_extra_vnni512,
    panel_extra_body_vnni512,
    "avx2,fma,f16c,avx512f,avx512vl,avx512vnni",
    dot_vnni_evex
);
kernel_extra!(
    panel_extra_avx2,
    panel_extra_body_avx2,
    "avx2,fma,f16c",
    dot_avx2
);

// ---------------------------------------------------------------------------------------------
// Predecoded panels (prompts)
//
// The streaming kernels above decode every weight block again for each pair of tokens. A prompt
// has hundreds of tokens, so a worker decodes its four-row panel once into `KBlock`s (unsigned
// weights plus ready-made scale words) and every token pair then runs only loads and dots on
// them. The panel stays in the worker's L1/L2; nothing is kept between calls, so no second copy
// of the weights exists.
//
// One formulation serves Q2_K..Q6_K. A weight is `d * scale * (q - offset)` or
// `d * scale * q - dmin * min`, with `q` unsigned. The dot product of `q` with the activations
// is scaled per 16- or 32-value group exactly as in the streaming kernels (pack + madd), and
// the offset or minimum becomes one more `madd` of `mins` and the 16-value activation sums:
// the integer correction `sum(32 * scale_h * activation_sum_h)` for Q6_K and Q3_K (offsets 32
// and 4), the float correction `dmin * sum(min_h * activation_sum_h)` for Q2_K, Q4_K and Q5_K.
// The integer sums are exact; the results equal those of the streaming kernels up to the\n// rounding of the float conversion, which sees the same sums in a different lane order.
// ---------------------------------------------------------------------------------------------

/// Number of 32-value activation records in one 256-value block.
const RECORDS: usize = 8;

/// Blocks of a row decoded at a time. A panel chunk (four rows of eight blocks, 14 KB) and the
/// activations of a token pair (8 KB) fit the L1 cache together, however long the rows are; the
/// partial sums of a row wait in a small buffer between chunks.
pub(crate) const PANEL_BLOCKS: usize = 8;

/// One 256-value block of a weight row, decoded.
#[repr(C, align(32))]
#[derive(Clone)]
pub(crate) struct KBlock {
    /// Unsigned weights; group `g` (32 values) at `32 * g`.
    q: [u8; 256],
    /// Scale words for group pair `p` at `16 * p`: `[a0 x4, b0 x4, a1 x4, b1 x4]` with `a` the
    /// even and `b` the odd group of the pair and `0`/`1` its first/second 16 values, i.e. the
    /// word order `packs_epi32` gives for the two dot products of the pair.
    scales: [i16; 64],
    /// Correction coefficients per 16-value half, index `2 * group + half` (Q4_K/Q5_K: one
    /// minimum per 32-value group at index `group`, the rest zero).
    mins: [i16; 16],
    d: f32,
    dmin: f32,
}

impl Default for KBlock {
    fn default() -> Self {
        Self {
            q: [0; 256],
            scales: [0; 64],
            mins: [0; 16],
            d: 0.0,
            dmin: 0.0,
        }
    }
}

/// Activation sums of one 256-value block as the 16-bit words `KBlock::mins` meets: sums of the
/// sixteen 16-value halves (Q2_K, Q3_K, Q6_K) or of the eight 32-value groups followed by
/// zeros (Q4_K, Q5_K).
#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub(crate) struct KSums(pub(crate) [i16; 16]);

/// Whether `kind` meets 16-value half sums (otherwise 32-value group sums).
pub(crate) const fn half_sums(kind: u32) -> bool {
    matches!(kind, Q2K | Q3K | Q6K)
}

/// The activation sums every predecoded panel needs, one `KSums` per token and block, computed
/// once per call instead of once per panel. `x` holds `batch * nb` records.
pub(crate) fn token_sums(kind: u32, x: &[X]) -> Vec<KSums> {
    let mut sums = vec![KSums::default(); x.len() / RECORDS];
    let fill = |(index, out): (usize, &mut KSums)| {
        let records = &x[index * RECORDS..(index + 1) * RECORDS];
        if half_sums(kind) {
            // `init` stores -32 times each 16-value half sum.
            for (i, word) in out.0.iter_mut().enumerate() {
                *word = (-records[i / 2].init[i % 2] >> 5) as i16;
            }
        } else {
            for (g, word) in out.0.iter_mut().take(RECORDS).enumerate() {
                *word = (-(records[g].init[0] + records[g].init[1]) >> 5) as i16;
            }
        }
    };
    if sums.len() >= 512 {
        sums.par_iter_mut().enumerate().for_each(fill);
    } else {
        sums.iter_mut().enumerate().for_each(fill);
    }
    sums
}

/// Decode one block of `raw` (a Q2_K..Q6_K block) into `out`.
///
/// # Safety
/// The CPU must support AVX2/F16C and `raw` hold one block of `KIND`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn decode_block<const KIND: u32>(raw: &[u8], out: &mut KBlock) {
    let zero = _mm256_setzero_si256();
    let (q_base, scale_base) = (out.q.as_mut_ptr(), out.scales.as_mut_ptr());
    let q_at = |pair: usize, half: usize| q_base.add(pair * 64 + half * 32).cast::<__m256i>();
    let scale_at = |pair: usize| scale_base.add(pair * 16).cast::<__m256i>();
    if KIND == Q4K || KIND == Q5K {
        out.d = half(raw, 0);
        out.dmin = half(raw, 2);
        let (scales, mins_word) = unpack_scales(raw);
        let scales16 = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(scales as i64));
        let low4 = _mm256_set1_epi8(15);
        let qs = if KIND == Q5K { 48 } else { 16 };
        let high_bits = if KIND == Q5K {
            _mm256_loadu_si256(raw.as_ptr().add(16).cast())
        } else {
            zero
        };
        for pair in 0..4usize {
            let packed = _mm256_loadu_si256(raw.as_ptr().add(qs + pair * 32).cast());
            let mut lo = _mm256_and_si256(packed, low4);
            let mut hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low4);
            if KIND == Q5K {
                let one = _mm256_set1_epi8(1);
                let lo_bit = _mm256_and_si256(
                    _mm256_srl_epi16(high_bits, _mm_cvtsi32_si128((2 * pair) as i32)),
                    one,
                );
                let hi_bit = _mm256_and_si256(
                    _mm256_srl_epi16(high_bits, _mm_cvtsi32_si128((2 * pair + 1) as i32)),
                    one,
                );
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(lo_bit, 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(hi_bit, 4));
            }
            _mm256_store_si256(q_at(pair, 0), lo);
            _mm256_store_si256(q_at(pair, 1), hi);
            _mm256_store_si256(
                scale_at(pair),
                _mm256_broadcastsi128_si256(_mm_shuffle_epi8(scales16, scale_selector(pair))),
            );
        }
        let mins = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(mins_word as i64));
        _mm256_store_si256(out.mins.as_mut_ptr().cast(), _mm256_zextsi128_si256(mins));
        return;
    }
    out.dmin = 0.0;
    let mut mins = [0i16; 16];
    let mut q3 = [0i32; 16];
    if KIND == Q3K {
        q3 = q3_scales(raw);
    }
    out.d = half(
        raw,
        match KIND {
            Q2K => 80,
            Q3K => 108,
            _ => 208,
        },
    );
    if KIND == Q2K {
        out.dmin = half(raw, 82);
        let packed = _mm_loadu_si128(raw.as_ptr().cast());
        let nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
        _mm256_store_si256(out.mins.as_mut_ptr().cast(), _mm256_cvtepu8_epi16(nibbles));
    }
    for pair in 0..4usize {
        let ((q1, a0, a1), (q2, b0, b1)) = match KIND {
            Q2K => (q2_group(raw, 2 * pair), q2_group(raw, 2 * pair + 1)),
            Q3K => (
                q3_group(raw, 2 * pair, &q3),
                q3_group(raw, 2 * pair + 1, &q3),
            ),
            _ => (q6_group(raw, 2 * pair), q6_group(raw, 2 * pair + 1)),
        };
        _mm256_store_si256(q_at(pair, 0), q1);
        _mm256_store_si256(q_at(pair, 1), q2);
        _mm256_store_si256(
            scale_at(pair),
            _mm256_set_epi64x(
                pair_scale(b1),
                pair_scale(a1),
                pair_scale(b0),
                pair_scale(a0),
            ),
        );
        // The offset of Q6_K (32) and Q3_K (4) multiplies the scale of each half.
        let factor = match KIND {
            Q6K => 32,
            Q3K => 4,
            _ => 0,
        };
        for (i, scale) in [a0, a1, b0, b1].into_iter().enumerate() {
            mins[4 * pair + i] = (factor * scale) as i16;
        }
    }
    if KIND != Q2K {
        _mm256_store_si256(
            out.mins.as_mut_ptr().cast(),
            _mm256_loadu_si256(mins.as_ptr().cast()),
        );
    }
}

/// Decode blocks `blk0..blk0 + count` (at most `PANEL_BLOCKS`) of `rows` (1..=4) rows, starting
/// at row `first`, of a weight matrix of `blocks` blocks per row into `panel`
/// (`panel[PANEL_BLOCKS * row + block - blk0]`).
///
/// # Safety
/// The CPU must support AVX2/F16C; `w` holds complete rows of `blocks` blocks of `kind` through
/// row `first + rows`.
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn decode_panel(
    w: &[u8],
    kind: u32,
    (first, rows): (usize, usize),
    blocks: usize,
    (blk0, count): (usize, usize),
    panel: &mut [KBlock],
) {
    debug_assert!(count <= PANEL_BLOCKS && blk0 + count <= blocks);
    let width = block_bytes(kind);
    let stride = blocks * width;
    for r in 0..rows {
        for j in 0..count {
            let start = (first + r) * stride + (blk0 + j) * width;
            let raw = &w[start..start + width];
            let out = &mut panel[r * PANEL_BLOCKS + j];
            match kind {
                Q2K => decode_block::<Q2K>(raw, out),
                Q3K => decode_block::<Q3K>(raw, out),
                Q4K => decode_block::<Q4K>(raw, out),
                Q5K => decode_block::<Q5K>(raw, out),
                Q6K => decode_block::<Q6K>(raw, out),
                _ => unreachable!("no predecoded kernel for this format"),
            }
        }
    }
}

macro_rules! predecoded_kernel {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Run the `rows` (1..=4) output rows of a decoded chunk of `count` blocks, starting at
        /// block `blk0`, for every token. Partial sums of a row carry from chunk to chunk in
        /// `partial` (`partial[4 * token + row]`); the last chunk writes the output.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `panel` holds the decoded chunk
        /// (`panel[PANEL_BLOCKS * row + block - blk0]`), `x` holds `batch * nb` records, `sums`
        /// `batch * nb / 8` entries made by `token_sums` for `kind`, and `partial` four entries
        /// per token (unused when the chunk is both first and last).
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            kind: u32,
            panel: &[KBlock],
            rows: usize,
            x: &[X],
            sums: &[KSums],
            nb: usize,
            batch: usize,
            (blk0, count): (usize, usize),
            partial: &mut [[f32; 8]],
            out: &mut Rows<'_>,
        ) {
            let span = (blk0, count, nb);
            macro_rules! by_rows {
                ($kind:literal) => {{
                    let mut token = 0;
                    while token < batch {
                        let tokens = (batch - token).min(2);
                        match (rows, tokens) {
                            (1, 1) => {
                                $body::<$kind, 1, 1>(panel, x, sums, span, token, partial, out)
                            }
                            (2, 1) => {
                                $body::<$kind, 2, 1>(panel, x, sums, span, token, partial, out)
                            }
                            (3, 1) => {
                                $body::<$kind, 3, 1>(panel, x, sums, span, token, partial, out)
                            }
                            (4, 1) => {
                                $body::<$kind, 4, 1>(panel, x, sums, span, token, partial, out)
                            }
                            (1, _) => {
                                $body::<$kind, 1, 2>(panel, x, sums, span, token, partial, out)
                            }
                            (2, _) => {
                                $body::<$kind, 2, 2>(panel, x, sums, span, token, partial, out)
                            }
                            (3, _) => {
                                $body::<$kind, 3, 2>(panel, x, sums, span, token, partial, out)
                            }
                            (4, _) => {
                                $body::<$kind, 4, 2>(panel, x, sums, span, token, partial, out)
                            }
                            _ => unreachable!("panel has 1 to 4 rows"),
                        }
                        token += tokens;
                    }
                }};
            }
            match kind {
                Q2K => by_rows!(10),
                Q3K => by_rows!(11),
                Q4K => by_rows!(12),
                Q5K => by_rows!(13),
                Q6K => by_rows!(14),
                _ => unreachable!("no predecoded kernel for this format"),
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            panel: &[KBlock],
            x: &[X],
            sums: &[KSums],
            (blk0, count, nb): (usize, usize, usize),
            token: usize,
            partial: &mut [[f32; 8]],
            out: &mut Rows<'_>,
        ) {
            let blocks = nb / RECORDS;
            let (first, last) = (blk0 == 0, blk0 + count == blocks);
            let zero = _mm256_setzero_si256();
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            if !first {
                for r in 0..R {
                    for t in 0..T {
                        acc[r][t] = _mm256_loadu_ps(partial[4 * (token + t) + r].as_ptr());
                    }
                }
            }
            for blk in blk0..blk0 + count {
                let xs: [&[X]; T] = std::array::from_fn(|t| {
                    &x[(token + t) * nb + blk * RECORDS..(token + t) * nb + (blk + 1) * RECORDS]
                });
                let sum_words: [__m256i; T] = std::array::from_fn(|t| {
                    _mm256_load_si256(sums[(token + t) * blocks + blk].0.as_ptr().cast())
                });
                for r in 0..R {
                    let block = &panel[r * PANEL_BLOCKS + blk - blk0];
                    let mut sumi = [zero; T];
                    for pair in 0..4usize {
                        let q1 = _mm256_load_si256(block.q.as_ptr().add(pair * 64).cast());
                        let q2 = _mm256_load_si256(block.q.as_ptr().add(pair * 64 + 32).cast());
                        let scales = _mm256_load_si256(block.scales.as_ptr().add(pair * 16).cast());
                        for t in 0..T {
                            let x1 = _mm256_load_si256(xs[t][2 * pair].q.as_ptr().cast());
                            let x2 = _mm256_load_si256(xs[t][2 * pair + 1].q.as_ptr().cast());
                            // Lane sums stay below 4 * 63 * 127 < 2^15: both groups pack into
                            // one vector and a single madd applies their scales.
                            let scaled = _mm256_madd_epi16(
                                _mm256_packs_epi32($dot(zero, q1, x1), $dot(zero, q2, x2)),
                                scales,
                            );
                            sumi[t] = _mm256_add_epi32(sumi[t], scaled);
                        }
                    }
                    let mins = _mm256_load_si256(block.mins.as_ptr().cast());
                    for t in 0..T {
                        let correction = _mm256_madd_epi16(mins, sum_words[t]);
                        if KIND == Q6K || KIND == Q3K {
                            sumi[t] = _mm256_sub_epi32(sumi[t], correction);
                        }
                        let scale = _mm256_set1_ps(block.d * xs[t][0].d);
                        acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[t]), scale, acc[r][t]);
                        if KIND != Q6K && KIND != Q3K {
                            acc[r][t] = _mm256_fnmadd_ps(
                                _mm256_cvtepi32_ps(correction),
                                _mm256_set1_ps(block.dmin * xs[t][0].d),
                                acc[r][t],
                            );
                        }
                    }
                }
            }
            if !last {
                for r in 0..R {
                    for t in 0..T {
                        _mm256_storeu_ps(partial[4 * (token + t) + r].as_mut_ptr(), acc[r][t]);
                    }
                }
                return;
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = hsum(acc[r][t]);
                }
            }
        }
    };
}

predecoded_kernel!(
    predecoded_vnni,
    predecoded_body_vnni,
    "avx2,fma,f16c,avxvnni",
    dot_vnni
);
predecoded_kernel!(
    predecoded_avx2,
    predecoded_body_avx2,
    "avx2,fma,f16c",
    dot_avx2
);


// ---------------------------------------------------------------------------------------------
// Eight-row panels with the rows in the vector lanes (prompt path of Q4_K, Q5_K and Q6_K).
//
// The weights of eight rows are transposed so that one 256-bit vector holds the same four-value
// slice of all eight rows. One `vpdpbusd` against a broadcast four-value slice of an activation
// then advances the dot products of eight rows at once and leaves them in separate lanes: no
// horizontal sums, no pack/madd scale step. Each sub-block (32 values, 16 for Q6_K) is converted
// to float once and scaled there; its minimum (or the Q6_K offset) is subtracted with the
// activation sum of the sub-block. The accumulators of a token are the eight rows of the panel.

/// Blocks decoded at a time by the eight-row kernel (a row of eight rows x four blocks, 12 KB).
pub(crate) const R8_BLOCKS: usize = 4;

/// One 256-value block of eight rows, transposed.
#[repr(C, align(32))]
#[derive(Clone)]
pub(crate) struct R8Block {
    /// Group `g` (32 values), quad `k`: the 32 bytes at `(8 * g + k) * 32`; lane `4 * row + j`
    /// holds value `4 * k + j` of the group of `row`.
    q: [u8; 2048],
    /// Row scales of sub-block `s` at `8 * s` (sub-blocks are 32 values; Q6_K: 16 values).
    scale: [f32; 128],
    /// Row minimums (Q6_K: 32 times the scale) of sub-block `s` at `8 * s`.
    min: [f32; 128],
}

impl Default for R8Block {
    fn default() -> Self {
        Self {
            q: [0; 2048],
            scale: [0.0; 128],
            min: [0.0; 128],
        }
    }
}

/// Whether the eight-row path handles `kind`.
pub(crate) const fn r8_kind(kind: u32) -> bool {
    matches!(kind, Q2K | Q3K | Q4K | Q5K | Q6K | IQ4NL)
}

/// Sub-blocks per 32-value group.
const fn subs(kind: u32) -> usize {
    if matches!(kind, Q2K | Q3K | Q6K) {
        2
    } else {
        1
    }
}

/// Transpose eight vectors of eight 32-bit words.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose8(r: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

/// Decode blocks `blk0..blk0 + count` (at most `R8_BLOCKS`) of up to eight rows from row `first`
/// into `panel`. Rows past `rows` become zero.
///
/// # Safety
/// The CPU must support AVX2/F16C; `w` holds complete rows of `blocks` blocks of `kind` through
/// row `first + rows`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn decode_r8(
    w: &[u8],
    kind: u32,
    (first, rows): (usize, usize),
    blocks: usize,
    (blk0, count): (usize, usize),
    panel: &mut [R8Block],
) {
    debug_assert!(count <= R8_BLOCKS && rows <= 8 && blk0 + count <= blocks);
    let width = if kind == IQ4NL { 144 } else { block_bytes(kind) };
    let stride = blocks * width;
    let spg = subs(kind);
    let mut decoded: Vec<KBlock> = vec![KBlock::default(); 8];
    let mut group_d = [[0.0f32; 8]; 8];
    let unsigned: [i8; 16] = std::array::from_fn(|i| IQ4_VALUES[i].wrapping_add_unsigned(128));
    let table = _mm_loadu_si128(unsigned.as_ptr().cast());
    for (j, out) in panel.iter_mut().enumerate().take(count) {
        for r in 0..rows {
            let start = (first + r) * stride + (blk0 + j) * width;
            let raw = &w[start..start + width];
            if kind == IQ4NL {
                // eight 18-byte blocks: fp16 scale, then 16 bytes with values 0..15 in the low
                // and 16..31 in the high nibbles; stored as codebook value + 128 (unsigned)
                let mask = _mm_set1_epi8(15);
                for g in 0..8 {
                    let block = &raw[g * 18..g * 18 + 18];
                    group_d[r][g] = half(block, 0);
                    let packed = _mm_loadu_si128(block.as_ptr().add(2).cast());
                    let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
                    let high = _mm_shuffle_epi8(
                        table,
                        _mm_and_si128(_mm_srli_epi16(packed, 4), mask),
                    );
                    let dst = decoded[r].q.as_mut_ptr().add(32 * g);
                    _mm_store_si128(dst.cast(), low);
                    _mm_store_si128(dst.add(16).cast(), high);
                }
                continue;
            }
            match kind {
                Q2K => decode_block::<Q2K>(raw, &mut decoded[r]),
                Q3K => decode_block::<Q3K>(raw, &mut decoded[r]),
                Q4K => decode_block::<Q4K>(raw, &mut decoded[r]),
                Q5K => decode_block::<Q5K>(raw, &mut decoded[r]),
                _ => decode_block::<Q6K>(raw, &mut decoded[r]),
            }
        }
        for g in 0..8 {
            let lines: [__m256i; 8] = std::array::from_fn(|r| {
                _mm256_load_si256(decoded[r].q.as_ptr().add(32 * g).cast())
            });
            let columns = transpose8(lines);
            for (k, column) in columns.iter().enumerate() {
                _mm256_store_si256(out.q.as_mut_ptr().add((8 * g + k) * 32).cast(), *column);
            }
        }
        for g in 0..8 {
            for h in 0..spg {
                let s = g * spg + h;
                for r in 0..8 {
                    if kind == IQ4NL {
                        out.scale[s * 8 + r] = group_d[r][g];
                        out.min[s * 8 + r] = 128.0 * group_d[r][g];
                        continue;
                    }
                    let kb = &decoded[r];
                    let (scale, min) = if spg == 2 {
                        let sc = f32::from(kb.scales[(g / 2) * 16 + (g % 2) * 4 + h * 8]);
                        match kind {
                            Q2K => (kb.d * sc, kb.dmin * f32::from(kb.mins[s])),
                            Q3K => (kb.d * sc, 4.0 * kb.d * sc),
                            _ => (kb.d * sc, 32.0 * kb.d * sc),
                        }
                    } else {
                        let sc = f32::from(kb.scales[(g / 2) * 16 + (g % 2) * 4]);
                        (kb.d * sc, kb.dmin * f32::from(kb.mins[g]))
                    };
                    out.scale[s * 8 + r] = scale;
                    out.min[s * 8 + r] = min;
                }
            }
        }
    }
}

/// Activations of a call in the layout the eight-row kernel reads.
pub(crate) struct Acts {
    cols: usize,
    blocks: usize,
    /// `batch * cols` quantized values.
    q: Vec<u8>,
    /// Block scale per token and block.
    scale: Vec<f32>,
    /// Sixteen sub-block sums per token and block (see `token_sums`).
    sums: Vec<f32>,
}

/// The 32 quantized values of every record, back to back.
fn copy_quants(x: &[X]) -> Vec<u8> {
    let mut q = vec![0u8; x.len() * 32];
    let copy = |(out, record): (&mut [u8], &X)| {
        for (o, &v) in out.iter_mut().zip(&record.q) {
            *o = v as u8;
        }
    };
    if x.len() >= 512 {
        q.par_chunks_mut(32).zip(x.par_iter()).for_each(copy);
    } else {
        q.chunks_mut(32).zip(x.iter()).for_each(copy);
    }
    q
}

/// Rearrange the records `x` (`batch * nb`) and their `sums` (`token_sums`) for the kernel.
pub(crate) fn pack_acts(x: &[X], sums: &[KSums], nb: usize) -> Acts {
    let blocks = nb / RECORDS;
    let batch = x.len() / nb;
    let q = copy_quants(x);
    let scale = (0..batch * blocks).map(|i| x[i * RECORDS].d).collect();
    let mut flat = vec![0.0f32; sums.len() * 16];
    let convert = |(out, s): (&mut [f32], &KSums)| {
        for (o, &v) in out.iter_mut().zip(&s.0) {
            *o = f32::from(v);
        }
    };
    if sums.len() >= 512 {
        flat.par_chunks_mut(16).zip(sums.par_iter()).for_each(convert);
    } else {
        flat.chunks_mut(16).zip(sums.iter()).for_each(convert);
    }
    let sums = flat;
    Acts {
        cols: nb * 32,
        blocks,
        q,
        scale,
        sums,
    }
}

/// Activations of a call in Q8_0 records (a scale and a sum per 32 values) for the kernel.
pub(crate) fn pack_acts_q8(x: &[X], nb: usize) -> Acts {
    Acts {
        cols: nb * 32,
        blocks: nb / RECORDS,
        q: copy_quants(x),
        scale: x.iter().map(|r| r.d).collect(),
        sums: x.iter().map(|r| r.sum[0] + r.sum[1]).collect(),
    }
}

macro_rules! r8_kernel {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Run a decoded chunk against every token; `total[token]` accumulates the eight rows.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `panel` holds the decoded chunk, `acts`
        /// one entry per token of `total`.
        #[allow(clippy::needless_range_loop)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            kind: u32,
            panel: &[R8Block],
            acts: &Acts,
            chunk: (usize, usize),
            total: &mut [[f32; 8]],
        ) {
            let batch = total.len();
            let mut token = 0;
            while token < batch {
                let left = batch - token;
                macro_rules! run {
                    ($spg:literal, $t:literal) => {{
                        $body::<$spg, $t>(panel, acts, chunk, token, total);
                        token += $t;
                    }};
                }
                if kind == IQ4NL {
                    match left {
                        1 => run!(0, 1),
                        2 => run!(0, 2),
                        3 => run!(0, 3),
                        _ => run!(0, 4),
                    }
                } else if subs(kind) == 2 {
                    match left {
                        1 => run!(2, 1),
                        2 => run!(2, 2),
                        3 => run!(2, 3),
                        _ => run!(2, 4),
                    }
                } else {
                    match left {
                        1 => run!(1, 1),
                        2 => run!(1, 2),
                        3 => run!(1, 3),
                        _ => run!(1, 4),
                    }
                }
            }
        }

        #[inline]
        #[allow(clippy::needless_range_loop)]
        #[target_feature(enable = $features)]
        unsafe fn $body<const SPG: usize, const T: usize>(
            panel: &[R8Block],
            acts: &Acts,
            (blk0, count): (usize, usize),
            token: usize,
            total: &mut [[f32; 8]],
        ) {
            // SPG 0: Q8_0 activations (scale and sum per 32 values), one sub-block per group
            let (sub, group_scales) = (SPG.max(1), SPG == 0);
            let quads = 8 / sub;
            for (jb, block) in panel.iter().enumerate().take(count) {
                let blk = blk0 + jb;
                let xq: [*const u8; T] = std::array::from_fn(|t| {
                    acts.q.as_ptr().add((token + t) * acts.cols + blk * 256)
                });
                let per_block = if group_scales { 8 } else { 16 };
                let sums: [*const f32; T] = std::array::from_fn(|t| {
                    acts.sums
                        .as_ptr()
                        .add(((token + t) * acts.blocks + blk) * per_block)
                });
                let scales: [*const f32; T] = std::array::from_fn(|t| {
                    acts.scale
                        .as_ptr()
                        .add(((token + t) * acts.blocks + blk) * per_block)
                });
                let mut fa = [_mm256_setzero_ps(); T];
                for g in 0..8 {
                    for h in 0..sub {
                        let s = g * sub + h;
                        let mut ia = [_mm256_setzero_si256(); T];
                        for k in h * quads..(h + 1) * quads {
                            let w = _mm256_load_si256(block.q.as_ptr().add((8 * g + k) * 32).cast());
                            for t in 0..T {
                                let y = _mm256_set1_epi32(
                                    xq[t].add(g * 32 + 4 * k).cast::<i32>().read_unaligned(),
                                );
                                ia[t] = $dot(ia[t], w, y);
                            }
                        }
                        let scale = _mm256_load_ps(block.scale.as_ptr().add(s * 8));
                        let min = _mm256_load_ps(block.min.as_ptr().add(s * 8));
                        for t in 0..T {
                            let scale = if group_scales {
                                _mm256_mul_ps(scale, _mm256_broadcast_ss(&*scales[t].add(s)))
                            } else {
                                scale
                            };
                            fa[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ia[t]), scale, fa[t]);
                            fa[t] = _mm256_fnmadd_ps(min, _mm256_broadcast_ss(&*sums[t].add(s)), fa[t]);
                        }
                    }
                }
                for t in 0..T {
                    let dy = if group_scales {
                        _mm256_set1_ps(1.0)
                    } else {
                        _mm256_set1_ps(acts.scale[(token + t) * acts.blocks + blk])
                    };
                    let cell = total[token + t].as_mut_ptr();
                    _mm256_storeu_ps(cell, _mm256_fmadd_ps(fa[t], dy, _mm256_loadu_ps(cell)));
                }
            }
        }
    };
}

r8_kernel!(r8_vnni, r8_body_vnni, "avx2,fma,f16c,avxvnni", dot_vnni);
r8_kernel!(r8_avx2, r8_body_avx2, "avx2,fma,f16c", dot_avx2);

#[cfg(test)]
mod tests {
    use crate::cpu_quantized::{linear, prepare};
    use half::f16;

    type Panel =
        unsafe fn(&[u8], u32, usize, usize, &[super::X], usize, usize, &mut super::Rows<'_>);

    /// Every panel kernel this CPU can run: (name, kernel, largest panel, formats).
    fn variants() -> Vec<(&'static str, Panel, usize, &'static [u32])> {
        const K: &[u32] = &[12, 13, 14];
        const EXTRA: &[u32] = &[10, 11, 23];
        let has = |name: &str| match name {
            "avxvnni" => std::arch::is_x86_feature_detected!("avxvnni"),
            "avx512" => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
            }
            "avx512vnni" => std::arch::is_x86_feature_detected!("avx512vnni"),
            _ => false,
        };
        let mut variants: Vec<(&'static str, Panel, usize, &'static [u32])> = vec![
            ("avx2", super::panel_avx2, 4, K),
            ("extra avx2", super::panel_extra_avx2, 4, EXTRA),
        ];
        if has("avxvnni") {
            variants.push(("avx-vnni", super::panel_vnni, 4, K));
            variants.push(("extra avx-vnni", super::panel_extra_vnni, 4, EXTRA));
        }
        if has("avx512") {
            variants.push(("avx512bw", super::panel512_bw, 8, K));
            if has("avx512vnni") {
                variants.push(("avx512-vnni", super::panel512_vnni, 8, K));
                variants.push(("extra evex vnni", super::panel_extra_vnni512, 4, EXTRA));
            }
        }
        variants
    }

    // Each K panel variant (AVX2, AVX-VNNI, AVX-512 BW/VNNI, EVEX VNNI) against the f64
    // reference: every panel row count, one and three super-blocks, odd token counts.
    #[test]
    fn every_k_panel_variant_matches_the_scalar_reference() {
        use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(91);
        for (name, panel, max_rows, kinds) in variants() {
            for &kind in kinds {
                for blocks in [1, 3] {
                    let cols = blocks * 256;
                    let rows = 9;
                    let weights = random_weights(kind, rows, cols, &mut rng);
                    for batch in [1, 2, 3] {
                        // Panels above four rows run one token at a time; dispatch only uses
                        // them for decode, but every size must stay correct.
                        for size in [1, 3, 4, max_rows] {
                            let x = rng.vector(batch * cols, 3.0);
                            let packed = prepare(&x, cols, kind);
                            let (want, scale) =
                                reference_packed(&weights, kind, rows, cols, &packed);
                            let mut output = vec![vec![0.0; batch * rows]];
                            crate::cpu_schedule::row_panels(
                                &mut output,
                                &[rows],
                                batch,
                                size,
                                || (),
                                |_, _, first, out| {
                                    let valid = (rows - first).min(size);
                                    // SAFETY: `variants` lists only kernels whose features this
                                    // CPU reports; the weights hold `rows` complete rows.
                                    unsafe {
                                        panel(
                                            &weights,
                                            kind,
                                            first,
                                            valid,
                                            &packed,
                                            cols / 32,
                                            batch,
                                            out,
                                        )
                                    };
                                },
                            );
                            let (error, ok) = compare(&output[0], &want, &scale);
                            assert!(
                                ok,
                                "{name} kind {kind} blocks {blocks} batch {batch} panel {size}: {error:e}"
                            );
                        }
                    }
                }
            }
        }
    }
    // The predecoded kernels (decode a panel once, then dot) against the f64 reference and
    // against the streaming kernels they replace for prompts: every panel row count, one and
    // three blocks, token counts around the pair boundaries.
    #[test]
    fn predecoded_panels_match_the_streaming_kernels() {
        use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(23);
        for kind in [10u32, 11, 12, 13, 14] {
            // Eleven blocks make a second, partial chunk after the full one.
            for blocks in [1, 3, 11] {
                let cols = blocks * 256;
                let rows = 7;
                let weights = random_weights(kind, rows, cols, &mut rng);
                for batch in [1, 2, 3, 4, 5, 9] {
                    let x = rng.vector(batch * cols, 3.0);
                    let packed = prepare(&x, cols, kind);
                    let nb = cols / 32;
                    let (want, scale) = reference_packed(&weights, kind, rows, cols, &packed);
                    let sums = super::token_sums(kind, &packed);
                    let mut streaming = vec![vec![0.0; batch * rows]];
                    crate::cpu_schedule::row_panels(
                        &mut streaming,
                        &[rows],
                        batch,
                        4,
                        || (),
                        |_, _, first, out| {
                            let valid = (rows - first).min(4);
                            // SAFETY: AVX2 was checked above; the weights hold `rows` rows.
                            unsafe {
                                if matches!(kind, 10 | 11) {
                                    super::panel_extra_avx2(
                                        &weights, kind, first, valid, &packed, nb, batch, out,
                                    )
                                } else {
                                    super::panel_avx2(
                                        &weights, kind, first, valid, &packed, nb, batch, out,
                                    )
                                }
                            };
                        },
                    );
                    type Decoded = unsafe fn(
                        u32,
                        &[super::KBlock],
                        usize,
                        &[super::X],
                        &[super::KSums],
                        usize,
                        usize,
                        (usize, usize),
                        &mut [[f32; 8]],
                        &mut super::Rows<'_>,
                    );
                    let mut variants: Vec<(&str, Decoded)> = vec![("avx2", super::predecoded_avx2)];
                    if std::arch::is_x86_feature_detected!("avxvnni") {
                        variants.push(("avx-vnni", super::predecoded_vnni));
                    }
                    for (name, kernel) in variants {
                        let mut have = vec![vec![0.0; batch * rows]];
                        crate::cpu_schedule::row_panels(
                            &mut have,
                            &[rows],
                            batch,
                            4,
                            || {
                                (
                                    vec![super::KBlock::default(); 4 * super::PANEL_BLOCKS],
                                    vec![[0.0f32; 8]; 4 * batch],
                                )
                            },
                            |(panel, partial), _, first, out| {
                                let valid = (rows - first).min(4);
                                // SAFETY: features checked above; the panel holds four rows of
                                // one chunk and `sums` one entry per token and block.
                                unsafe {
                                    for blk0 in (0..blocks).step_by(super::PANEL_BLOCKS) {
                                        let count = (blocks - blk0).min(super::PANEL_BLOCKS);
                                        super::decode_panel(
                                            &weights,
                                            kind,
                                            (first, valid),
                                            blocks,
                                            (blk0, count),
                                            panel,
                                        );
                                        kernel(
                                            kind,
                                            panel,
                                            valid,
                                            &packed,
                                            &sums,
                                            nb,
                                            batch,
                                            (blk0, count),
                                            partial,
                                            out,
                                        );
                                    }
                                }
                            },
                        );
                        let (error, ok) = compare(&have[0], &want, &scale);
                        assert!(
                            ok,
                            "{name} kind {kind} blocks {blocks} batch {batch}: {error:e}"
                        );
                        for (i, (a, b)) in have[0].iter().zip(&streaming[0]).enumerate() {
                            assert!(
                                (a - b).abs() as f64 <= 1e-5 * scale[i] + 1e-6,
                                "{name} kind {kind} blocks {blocks} batch {batch} cell {i}: {a} vs streaming {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 20_001) as f32 / 10_000.0 - 1.0
        }
    }

    fn random_rows(kind: u32, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
        if kind == 20 {
            let mut bytes = Vec::new();
            for _ in 0..rows * cols / 32 {
                let mut block = [0u8; 18];
                for byte in block.iter_mut() {
                    *byte = rng.next() as u8;
                }
                let d = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_le_bytes();
                block[0..2].copy_from_slice(&d);
                bytes.extend_from_slice(&block);
            }
            return bytes;
        }
        let width = super::block_bytes(kind);
        let mut bytes = Vec::new();
        for _ in 0..rows * (cols / 256) {
            let mut block = vec![0u8; width];
            for byte in block.iter_mut() {
                *byte = rng.next() as u8;
            }
            let d = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_le_bytes();
            let m = f16::from_f32((rng.next() % 100) as f32 / 4000.0).to_le_bytes();
            if kind == 14 {
                block[208..210].copy_from_slice(&d);
            } else if kind == 11 {
                block[108..110].copy_from_slice(&d);
            } else if kind == 10 {
                block[80..82].copy_from_slice(&d);
                block[82..84].copy_from_slice(&m);
            } else {
                block[0..2].copy_from_slice(&d);
                block[2..4].copy_from_slice(&m);
            }
            bytes.extend(block);
        }
        bytes
    }

    #[test]
    fn k_quant_panels_match_the_dequantized_reference() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(7);
        for kind in [10u32, 11, 12, 13, 14] {
            for (rows, cols, batch) in [
                (1, 256, 1),
                (4, 512, 1),
                (7, 512, 2),
                (9, 768, 5),
                (5, 256, 3),
                (13, 2560, 11),
                (8, 1280, 6),
                (24, 1024, 7),
                (17, 2304, 4),
            ] {
                let weights = random_rows(kind, rows, cols, &mut rng);
                let x: Vec<f32> = (0..batch * cols).map(|_| rng.unit() * 3.0).collect();
                let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
                let packed = prepare(&x, cols, kind);
                let nb = cols / 32;
                let width = cols / 256 * super::block_bytes(kind);
                let got = if batch == 1 {
                    linear(&weights, kind, &x, rows, cols, Some(&bias)).unwrap()
                } else {
                    // `linear` takes one token; run the batch through the grouped entry point
                    crate::cpu_quantized::linear_group(
                        &[crate::cpu_quantized::Projection {
                            weights: &weights,
                            rows,
                            bias: Some(&bias),
                        }],
                        kind,
                        &x,
                        cols,
                    )
                    .unwrap()
                    .pop()
                    .unwrap()
                };
                for t in 0..batch {
                    for r in 0..rows {
                        let row = crate::compute::dequantize(
                            &weights[r * width..(r + 1) * width],
                            kind,
                            cols,
                        )
                        .unwrap();
                        let mut want = f64::from(bias[r]);
                        let mut scale = 0.0f64;
                        for (i, w) in row.iter().enumerate() {
                            let record = &packed[t * nb + i / 32];
                            let v = f64::from(record.q[i % 32]) * f64::from(record.d);
                            want += f64::from(*w) * v;
                            scale += (f64::from(*w) * v).abs();
                        }
                        let have = f64::from(got[t * rows + r]);
                        assert!(
                            (have - want).abs() <= 2e-4 * scale + 1e-4,
                            "kind {kind} rows {rows} cols {cols} batch {batch} token {t} row {r}: {have} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// Prompt-path throughput of the grouped entry point; `cargo test --release bench_prompt --
    /// --ignored --nocapture` (FLLM_BENCH_KIND, FLLM_BENCH_THREADS, FLLM_BENCH_SHAPE=rows,cols,batch).
    #[test]
    #[ignore]
    fn bench_prompt_throughput() {
        let env = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let kind = env("FLLM_BENCH_KIND", 12) as u32;
        let threads = env("FLLM_BENCH_THREADS", 6);
        let rows = env("FLLM_BENCH_ROWS", 6144);
        let cols = env("FLLM_BENCH_COLS", 1024);
        let batch = env("FLLM_BENCH_BATCH", 358);
        let mut rng = Lcg(11);
        let weights = random_rows(kind, rows, cols, &mut rng);
        let x: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let run = || {
            crate::cpu_quantized::linear_group(
                &[crate::cpu_quantized::Projection { weights: &weights, rows, bias: None }],
                kind,
                &x,
                cols,
            )
            .unwrap()
        };
        pool.install(|| {
            run();
            let mut best = f64::MAX;
            for _ in 0..8 {
                let t = std::time::Instant::now();
                std::hint::black_box(run());
                best = best.min(t.elapsed().as_secs_f64());
            }
            let macs = (rows * cols * batch) as f64;
            println!(
                "bench kind {kind} {rows}x{cols} batch {batch} threads {threads}: {:.2} ms, {:.1} GMAC/s",
                best * 1e3,
                macs / best / 1e9
            );
        });
    }

    #[test]
    fn iq4_nl_eight_row_panels_match_the_dequantized_reference() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(19);
        for (rows, cols, batch) in [(9usize, 512usize, 5usize), (16, 1024, 7), (3, 256, 4), (21, 1536, 6)] {
            let mut weights = Vec::new();
            for _ in 0..rows * cols / 32 {
                let mut block = [0u8; 18];
                for byte in block.iter_mut() {
                    *byte = rng.next() as u8;
                }
                let d = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_le_bytes();
                block[0..2].copy_from_slice(&d);
                weights.extend_from_slice(&block);
            }
            let x: Vec<f32> = (0..batch * cols).map(|_| rng.unit() * 3.0).collect();
            let packed = prepare(&x, cols, 20);
            let nb = cols / 32;
            let got = crate::cpu_quantized::linear_group(
                &[crate::cpu_quantized::Projection { weights: &weights, rows, bias: None }],
                20,
                &x,
                cols,
            )
            .unwrap()
            .pop()
            .unwrap();
            let width = cols / 32 * 18;
            for t in 0..batch {
                for r in 0..rows {
                    let row = crate::compute::dequantize(&weights[r * width..(r + 1) * width], 20, cols).unwrap();
                    let mut want = 0.0f64;
                    let mut scale = 0.0f64;
                    for (i, w) in row.iter().enumerate() {
                        let record = &packed[t * nb + i / 32];
                        let v = f64::from(record.q[i % 32]) * f64::from(record.d);
                        want += f64::from(*w) * v;
                        scale += (f64::from(*w) * v).abs();
                    }
                    let have = f64::from(got[t * rows + r]);
                    assert!(
                        (have - want).abs() <= 2e-4 * scale + 1e-4,
                        "rows {rows} cols {cols} batch {batch} token {t} row {r}: {have} vs {want}"
                    );
                }
            }
        }
    }
}
