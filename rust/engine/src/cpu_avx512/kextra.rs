//! Q2_K, Q3_K and IQ4_XS streaming panels on 512-bit vectors.
//!
//! `cpu_kquant::panel_extra_*` runs these formats on 256-bit vectors only (AVX-VNNI, EVEX-encoded
//! 256-bit `vpdpbusd`, or AVX2). The kernels here follow the same arithmetic on 512-bit vectors,
//! after `cpu_kquant::kernel512` (Q4_K..Q6_K):
//!
//! - Q2_K and Q3_K: a 64-byte vector holds two 32-value groups in natural order (128-bit lanes
//!   `[g first 16, g last 16, g' first 16, g' last 16]`). Two vectors (groups `4n, 4n+1` and
//!   `4n+2, 4n+3` of a 128-value half) give four 4-lane sums per lane pair; `packs_epi32` plus one
//!   `madd` with the 16-value scales applies all four scales. The scale words of lane `L` are
//!   `[scale(8n+L) x4, scale(8n+4+L) x4]`.
//! - Q2_K's minimums meet the 16-value activation sums in one `madd` per block; Q3_K's offset (4)
//!   is folded into the dot's initial value (`-4 * lane sums`).
//! - IQ4_XS: two groups per vector in the interleaved arrangement of `codebook` (`[g low nibbles,
//!   g' low, g high, g' high]`); the sign-transfer `maddubs` dot takes the group scale in its
//!   `madd` (the scale words alternate by 128-bit lane).
//!
//! Panels are up to eight rows; more than four rows run one token at a time.
use super::amx_decode::q3_scales;
use crate::cpu_kernels::x86::{half_to_float, join256};
use crate::cpu_kquant::IQ4_VALUES;
use crate::cpu_quantized::X;
use crate::cpu_schedule::Rows;
use std::arch::x86_64::*;

const Q2K: u32 = 10;
const Q3K: u32 = 11;
const IQ4XS: u32 = 23;

const fn block_bytes(kind: u32) -> usize {
    match kind {
        Q2K => 84,
        Q3K => 110,
        _ => 136,
    }
}

#[inline]
#[target_feature(enable = "f16c")]
unsafe fn half(raw: &[u8], at: usize) -> f32 {
    half_to_float(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

/// 16-bit lanes 0..16 hold `low`, lanes 16..32 `high`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn halves_epi16(low: u16, high: u16) -> __m512i {
    _mm512_mask_blend_epi16(
        0xffff_0000,
        _mm512_set1_epi16(low as i16),
        _mm512_set1_epi16(high as i16),
    )
}

/// Byte shuffle that makes 128-bit lane `L` of a broadcast eight-word scale vector read
/// `[word L x4, word L+4 x4]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn scale_selector512() -> __m512i {
    let mut bytes = [0i8; 64];
    for (lane, chunk) in bytes.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        for (i, byte) in chunk.iter_mut().enumerate() {
            let word = if i < 8 { lane } else { lane + 4 };
            *byte = (2 * word + i % 2) as i8;
        }
    }
    _mm512_loadu_si512(bytes.as_ptr().cast())
}

/// Activation records `a` and `b` as the low and high halves of one vector (natural order).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn records512(a: &X, b: &X) -> __m512i {
    join256(
        _mm256_load_si256(a.q.as_ptr().cast()),
        _mm256_load_si256(b.q.as_ptr().cast()),
    )
}

/// `-(1 << SHIFT) * (sum of the four activations in each dword lane)`: the offset of Q3_K
/// (4, `SHIFT` 2) as the dot's initial value.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lane_bias512<const SHIFT: u32>(qx: __m512i) -> __m512i {
    let pairs = _mm512_maddubs_epi16(_mm512_set1_epi8(1), qx);
    let sums = _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
    _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32::<SHIFT>(sums))
}

/// `vpdpbusd` on 512-bit vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_vnni(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
    _mm512_dpbusd_epi32(init, u, s)
}

/// The same contract on AVX-512BW; weights are at most 7, so a `maddubs` pair never saturates.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_bw(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
    _mm512_add_epi32(
        _mm512_madd_epi16(_mm512_maddubs_epi16(u, s), _mm512_set1_epi16(1)),
        init,
    )
}

/// Signed weights by signed activations with the group scales applied in the `madd`:
/// `|w|, |x| <= 127`, so a pair sum is at most 32258 and cannot saturate.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_signed_scaled512(w: __m512i, x: __m512i, scale: __m512i) -> __m512i {
    let x = _mm512_mask_sub_epi8(x, _mm512_movepi8_mask(w), _mm512_setzero_si512(), x);
    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_abs_epi8(w), x), scale)
}

macro_rules! kernel_extra512 {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// `rows` (1..=8) consecutive Q2_K, Q3_K or IQ4_XS output rows from `first` for every
        /// token; panels of more than four rows run one token at a time.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` holds `first + rows` complete rows of
        /// `nb / 8` blocks and `x` holds `batch * nb` activation records.
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
            assert!(w.len() >= (first + R) * stride);
            let zero = _mm512_setzero_si512();
            let selector = scale_selector512();
            let iq4_table = _mm512_broadcast_i32x4(_mm_loadu_si128(IQ4_VALUES.as_ptr().cast()));
            let mut acc = [[_mm512_setzero_ps(); T]; R];
            for blk in 0..blocks {
                let xs: [&[X]; T] = std::array::from_fn(|t| {
                    &x[(token + t) * nb + blk * 8..(token + t) * nb + blk * 8 + 8]
                });
                // Activation pairs in the weight layout, shared by every row of the panel.
                let mut xv = [[zero; 4]; T];
                let mut lane_bias = [[zero; 4]; T];
                let mut half_sums = [_mm256_setzero_si256(); T];
                for t in 0..T {
                    let r = xs[t];
                    if KIND == IQ4XS {
                        for j in 0..4 {
                            let joined = records512(&r[2 * j], &r[2 * j + 1]);
                            // [a.lo16, a.hi16, b.lo16, b.hi16] -> [a.lo16, b.lo16, a.hi16, b.hi16]
                            xv[t][j] = _mm512_shuffle_i64x2::<0xD8>(joined, joined);
                        }
                    } else {
                        for j in 0..4 {
                            xv[t][j] = records512(&r[2 * j], &r[2 * j + 1]);
                        }
                        if KIND == Q3K {
                            for j in 0..4 {
                                lane_bias[t][j] = lane_bias512::<2>(xv[t][j]);
                            }
                        } else {
                            // 16-value activation sums; `init` stores -32 * each half sum.
                            let s = |i: usize| (-r[i / 2].init[i % 2] >> 5) as i16;
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
                }
                for r in 0..R {
                    let start = (first + r) * stride + blk * width;
                    let raw = &w[start..start + width];
                    let mut sumi = [zero; T];
                    let d;
                    if KIND == IQ4XS {
                        d = half(raw, 0);
                        let scales_high = u16::from_le_bytes([raw[2], raw[3]]);
                        let scale = |g: usize| -> i16 {
                            let low = (raw[4 + g / 2] >> (4 * (g % 2))) & 15;
                            let ls =
                                i32::from(low) | ((i32::from(scales_high >> (2 * g)) & 3) << 4);
                            (ls - 32) as i16
                        };
                        for j in 0..4usize {
                            let packed = _mm256_loadu_si256(raw.as_ptr().add(8 + j * 32).cast());
                            let low = _mm256_and_si256(packed, _mm256_set1_epi8(15));
                            let high = _mm256_and_si256(
                                _mm256_srli_epi16(packed, 4),
                                _mm256_set1_epi8(15),
                            );
                            let index = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high);
                            let weights = _mm512_shuffle_epi8(iq4_table, index);
                            // 128-bit lanes 1 and 3 hold the odd group of the pair
                            let scales = _mm512_mask_blend_epi16(
                                0xff00_ff00,
                                _mm512_set1_epi16(scale(2 * j)),
                                _mm512_set1_epi16(scale(2 * j + 1)),
                            );
                            for t in 0..T {
                                sumi[t] = _mm512_add_epi32(
                                    sumi[t],
                                    dot_signed_scaled512(weights, xv[t][j], scales),
                                );
                            }
                        }
                    } else {
                        d = half(raw, if KIND == Q2K { 80 } else { 108 });
                        let scales16: [i16; 16] = if KIND == Q2K {
                            std::array::from_fn(|i| i16::from(raw[i] & 15))
                        } else {
                            q3_scales(raw).map(|s| s as i16)
                        };
                        // Q2_K: values at qs[16 + 32n + l] >> 2s; Q3_K: qs[32 + 32n + l] >> 2s with
                        // the high bit from hmask[l] >> group at bit 2.
                        let qs_at = if KIND == Q2K { 16 } else { 32 };
                        let hmask = if KIND == Q3K {
                            _mm512_broadcast_i64x4(_mm256_loadu_si256(raw.as_ptr().cast()))
                        } else {
                            zero
                        };
                        let low2 = _mm512_set1_epi8(3);
                        for n in 0..2usize {
                            let qs = _mm512_broadcast_i64x4(_mm256_loadu_si256(
                                raw.as_ptr().add(qs_at + 32 * n).cast(),
                            ));
                            // vectors: groups (4n, 4n+1) shifted by (0, 2); (4n+2, 4n+3) by (4, 6)
                            let mut vectors = [zero; 2];
                            for (v, vector) in vectors.iter_mut().enumerate() {
                                let shifts = halves_epi16((4 * v) as u16, (4 * v + 2) as u16);
                                let mut q = _mm512_and_si512(_mm512_srlv_epi16(qs, shifts), low2);
                                if KIND == Q3K {
                                    let group_a = (4 * n + 2 * v) as u16;
                                    let bits = _mm512_and_si512(
                                        _mm512_srlv_epi16(
                                            hmask,
                                            halves_epi16(group_a, group_a + 1),
                                        ),
                                        _mm512_set1_epi8(1),
                                    );
                                    q = _mm512_or_si512(q, _mm512_slli_epi16::<2>(bits));
                                }
                                *vector = q;
                            }
                            let words = _mm_loadu_si128(scales16.as_ptr().add(8 * n).cast());
                            let scales =
                                _mm512_shuffle_epi8(_mm512_broadcast_i32x4(words), selector);
                            for t in 0..T {
                                let (i1, i2) = if KIND == Q3K {
                                    (lane_bias[t][2 * n], lane_bias[t][2 * n + 1])
                                } else {
                                    (zero, zero)
                                };
                                // Lane sums are small (Q2_K at most 4 * 3 * 127, Q3_K biased in
                                // [-2032, 3556]), so both vectors pack into 16-bit words.
                                let p1 = $dot(i1, vectors[0], xv[t][2 * n]);
                                let p2 = $dot(i2, vectors[1], xv[t][2 * n + 1]);
                                let scaled = _mm512_madd_epi16(_mm512_packs_epi32(p1, p2), scales);
                                sumi[t] = _mm512_add_epi32(sumi[t], scaled);
                            }
                        }
                    }
                    for t in 0..T {
                        let scale = _mm512_set1_ps(d * xs[t][0].d);
                        acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sumi[t]), scale, acc[r][t]);
                        if KIND == Q2K {
                            // minimum correction: sum_i min_i * (sum of sub-block i's activations)
                            let packed = _mm_loadu_si128(raw.as_ptr().cast());
                            let mins = _mm256_cvtepu8_epi16(_mm_and_si128(
                                _mm_srli_epi16(packed, 4),
                                _mm_set1_epi8(15),
                            ));
                            let product = _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, half_sums[t]));
                            acc[r][t] = _mm512_fnmadd_ps(
                                _mm512_zextps256_ps512(product),
                                _mm512_set1_ps(half(raw, 82) * xs[t][0].d),
                                acc[r][t],
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

kernel_extra512!(
    panel_extra512_vnni,
    panel_extra512_vnni_body,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
    dot_vnni
);
kernel_extra512!(
    panel_extra512_bw,
    panel_extra512_bw_body,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
    dot_bw
);

/// Run the 512-bit kernel of the active tier. `false` if there is none (the caller keeps the
/// 256-bit kernels).
///
/// # Safety
/// `w` holds `first + rows` complete rows of `nb / 8` blocks of `kind` (10, 11 or 23), `x` holds
/// `batch * nb` Q8_K records, `out` is this panel's exclusive view, `rows` at most
/// `panel_rows(batch)`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn panel(
    w: &[u8],
    kind: u32,
    first: usize,
    rows: usize,
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) -> bool {
    use crate::cpu_kernels::Avx512;
    match crate::cpu_kernels::avx512() {
        Some(Avx512::Vnni) => panel_extra512_vnni(w, kind, first, rows, x, nb, batch, out),
        Some(Avx512::Bw) => panel_extra512_bw(w, kind, first, rows, x, nb, batch, out),
        None => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    type Kernel = unsafe fn(&[u8], u32, usize, usize, &[X], usize, usize, &mut Rows<'_>);

    fn check(kernel: Kernel, size: usize, label: &str) {
        let mut rng = Lcg(0x4b_e7_7a ^ size as u64);
        for kind in [Q2K, Q3K, IQ4XS] {
            for blocks in [1usize, 3] {
                let cols = blocks * 256;
                let nb = cols / 32;
                for rows in [1usize, 3, 4, 5, 8, 11] {
                    let w = random_weights(kind, rows, cols, &mut rng);
                    for batch in [1usize, 2, 3, 5] {
                        let x = rng.vector(batch * cols, 3.0);
                        let packed = crate::cpu_quantized::prepare(&x, cols, 12);
                        let mut output = vec![vec![0.0f32; rows * batch]];
                        crate::cpu_schedule::row_panels(
                            &mut output,
                            &[rows],
                            batch,
                            size,
                            || (),
                            |_, _, first, out| {
                                let valid = (rows - first).min(size);
                                // SAFETY: the test checked the CPU features; shapes match.
                                unsafe { kernel(&w, kind, first, valid, &packed, nb, batch, out) };
                            },
                        );
                        let (want, scale) = reference_packed(&w, kind, rows, cols, &packed);
                        let (error, ok) = compare(&output[0], &want, &scale);
                        assert!(ok, "{label} kind {kind} blocks {blocks} rows {rows} batch {batch}: {error:e}");
                    }
                }
            }
        }
    }

    fn has_512() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    /// Scalar model of one block of `kernel_extra512!` for one row and one token, written with
    /// the operations of `model` in the order the kernel issues them. `vnni` selects the
    /// `vpdpbusd` dot over the `maddubs` dot for Q2_K and Q3_K. Returns the block's contribution.
    fn model_block(kind: u32, vnni: bool, raw: &[u8], recs: &[X]) -> f32 {
        use crate::cpu_avx512::model::*;
        let bytes = |x: &X| -> [u8; 32] { x.q.map(|v| v as u8) };
        let xv: [B64; 4] = std::array::from_fn(|j| {
            let joined = join256(&bytes(&recs[2 * j]), &bytes(&recs[2 * j + 1]));
            if kind == IQ4XS {
                shuffle_i64x2(&joined, &joined, 0xD8)
            } else {
                joined
            }
        });
        let mut sumi = [0i32; 16];
        let mut add = |v: [i32; 16]| {
            for i in 0..16 {
                sumi[i] += v[i];
            }
        };
        // SAFETY: F16C is checked by the callers of this model.
        let half_at = |at: usize| unsafe { half(raw, at) };
        let dx = recs[0].d;
        let mut result;
        if kind == IQ4XS {
            let d = half_at(0);
            let scales_high = u16::from_le_bytes([raw[2], raw[3]]);
            let scale = |g: usize| -> i16 {
                let low = (raw[4 + g / 2] >> (4 * (g % 2))) & 15;
                (i32::from(low) | ((i32::from(scales_high >> (2 * g)) & 3) << 4)) as i16 - 32
            };
            let table = broadcast128(&IQ4_VALUES.map(|v| v as u8));
            for j in 0..4usize {
                let packed: [u8; 32] = raw[8 + j * 32..8 + j * 32 + 32].try_into().unwrap();
                let index = join256(&packed.map(|v| v & 15), &packed.map(|v| (v >> 4) & 15));
                let weights = shuffle_epi8(&table, &index);
                let scales = blend_epi16(0xff00_ff00, &[scale(2 * j); 32], &[scale(2 * j + 1); 32]);
                add(dot_signed_scaled(&weights, &xv[j], &scales));
            }
            result = d * dx * sumi.iter().sum::<i32>() as f32;
        } else {
            let d = half_at(if kind == Q2K { 80 } else { 108 });
            let scales16: [i16; 16] = if kind == Q2K {
                std::array::from_fn(|i| i16::from(raw[i] & 15))
            } else {
                q3_scales(raw).map(|s| s as i16)
            };
            let qs_at = if kind == Q2K { 16 } else { 32 };
            let hmask = broadcast256(raw[..32].try_into().unwrap());
            // lane-bias of Q3_K: -4 * the sum of the four activations of each dword
            let lane_bias = |qx: &B64| -> [i32; 16] {
                let sums = madd16(&maddubs(&set1_epi8(1), qx), &[1; 32]);
                sums.map(|s| -(s << 2))
            };
            // the kernel's selector, restated: lane L reads words [L x4, L+4 x4]
            let mut selector = zero();
            for (lane_i, chunk) in selector.chunks_mut(16).enumerate() {
                for (i, byte) in chunk.iter_mut().enumerate() {
                    let word = if i < 8 { lane_i } else { lane_i + 4 };
                    *byte = (2 * word + i % 2) as u8;
                }
            }
            for n in 0..2usize {
                let qs = broadcast256(raw[qs_at + 32 * n..qs_at + 32 * n + 32].try_into().unwrap());
                let mut vectors = [zero(); 2];
                for (v, vector) in vectors.iter_mut().enumerate() {
                    let mut q = and8(
                        &srlv_epi16(&qs, &halves_epi16((4 * v) as u16, (4 * v + 2) as u16)),
                        3,
                    );
                    if kind == Q3K {
                        let group_a = (4 * n + 2 * v) as u16;
                        let bits =
                            and8(&srlv_epi16(&hmask, &halves_epi16(group_a, group_a + 1)), 1);
                        q = or(&q, &slli_epi16(&bits, 2));
                    }
                    *vector = q;
                }
                let mut word_bytes = [0u8; 16];
                for (i, w) in scales16[8 * n..8 * n + 8].iter().enumerate() {
                    word_bytes[2 * i..2 * i + 2].copy_from_slice(&w.to_le_bytes());
                }
                let scales = shuffle_epi8(&broadcast128(&word_bytes), &selector);
                let (i1, i2) = if kind == Q3K {
                    (lane_bias(&xv[2 * n]), lane_bias(&xv[2 * n + 1]))
                } else {
                    ([0; 16], [0; 16])
                };
                let dot = |init: &[i32; 16], u: &B64, s: &B64| {
                    if vnni {
                        dpbusd(init, u, s)
                    } else {
                        dot_bw(init, u, s)
                    }
                };
                let p1 = dot(&i1, &vectors[0], &xv[2 * n]);
                let p2 = dot(&i2, &vectors[1], &xv[2 * n + 1]);
                add(madd16(&packs32(&p1, &p2), &words(&scales)));
            }
            result = d * dx * sumi.iter().sum::<i32>() as f32;
            if kind == Q2K {
                // minimum: sum_i min_i * (activation sum of 16-value sub-block i)
                let mut correction = 0i64;
                for i in 0..16 {
                    let min = i64::from(raw[i] >> 4);
                    let sum = i64::from(-recs[i / 2].init[i % 2] >> 5);
                    correction += min * sum;
                }
                result -= half_at(82) * dx * correction as f32;
            }
        }
        result
    }

    /// The 512-bit lane arrangement, vector shifts and scale-word mapping of the Q2_K, Q3_K and
    /// IQ4_XS kernels, run as a scalar model, give the reference result.
    #[test]
    fn modelled_lane_arrangement_matches_the_reference() {
        if !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        let mut rng = Lcg(0x70de_0002);
        for kind in [Q2K, Q3K, IQ4XS] {
            for blocks in [1usize, 3] {
                let cols = blocks * 256;
                let w = random_weights(kind, 1, cols, &mut rng);
                let x = rng.vector(cols, 3.0);
                let packed = crate::cpu_quantized::prepare(&x, cols, 12);
                let (want, scale) = reference_packed(&w, kind, 1, cols, &packed);
                for vnni in [true, false] {
                    let width = block_bytes(kind);
                    let total: f32 = (0..blocks)
                        .map(|b| {
                            model_block(
                                kind,
                                vnni,
                                &w[b * width..(b + 1) * width],
                                &packed[b * 8..b * 8 + 8],
                            )
                        })
                        .sum();
                    let (error, ok) = compare(&[total], &want, &scale);
                    assert!(ok, "kind {kind} blocks {blocks} vnni {vnni}: {error:e}");
                }
            }
        }
    }

    #[test]
    fn avx512_bw_extra_panels_match_the_reference() {
        if has_512() {
            check(panel_extra512_bw, 4, "avx512-bw/4");
            check(panel_extra512_bw, 8, "avx512-bw/8");
        }
    }

    #[test]
    fn avx512_vnni_extra_panels_match_the_reference() {
        if has_512() && std::arch::is_x86_feature_detected!("avx512vnni") {
            check(panel_extra512_vnni, 4, "avx512-vnni/4");
            check(panel_extra512_vnni, 8, "avx512-vnni/8");
        }
    }

    /// The scale-word arrangement of the Q2_K/Q3_K kernels, modelled with plain arrays: for a
    /// 128-value half `n`, lane `L` of vector A holds 16-value scale `8n + L` and lane `L` of
    /// vector B scale `8n + 4 + L`, which is what the selector produces.
    #[test]
    fn scale_selector_pairs_lane_l_with_lane_l_plus_four() {
        for lane in 0..4usize {
            let mut bytes = [0usize; 16];
            for (i, byte) in bytes.iter_mut().enumerate() {
                let word = if i < 8 { lane } else { lane + 4 };
                *byte = 2 * word + i % 2;
            }
            // words (byte pairs) of the lane: [L x4, L+4 x4]
            for i in 0..8 {
                let word = bytes[2 * i] / 2;
                assert_eq!(word, if i < 4 { lane } else { lane + 4 });
                assert_eq!(bytes[2 * i + 1], bytes[2 * i] + 1);
            }
        }
    }

    /// Which 16-value scale each (vector, lane) pair uses for Q2_K/Q3_K, derived from the group
    /// order `group = 4n + shift_index` of `q2_group`/`q3_group`: lane `L` of the vector covering
    /// groups `(g, g + 1)` is group `g + L / 2`, half `L % 2`, i.e. scale index `2g + L`.
    #[test]
    fn vector_lanes_map_to_the_sixteen_value_scales() {
        for n in 0..2usize {
            for v in 0..2usize {
                let g = 4 * n + 2 * v; // first group of the vector
                for lane in 0..4usize {
                    let scale_index = 2 * (g + lane / 2) + lane % 2;
                    let expected = 8 * n + 4 * v + lane;
                    assert_eq!(scale_index, expected, "n {n} v {v} lane {lane}");
                }
            }
        }
    }
}
