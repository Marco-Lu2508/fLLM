//! Packed integer projections for the grid-based IQ formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18),
//! IQ3_S (21) and IQ2_S (22).
//!
//! The weights of these formats are indices into small lattices (`quant_iq_tables.rs`) plus sign bits and a
//! 4-bit scale per 16 or 32 values. A row never leaves its packed form: a kernel decodes one 256-value block
//! at a time into eight groups of 32 (unsigned lattice bytes, a vector of +-1 sign bytes, two integer scales)
//! and dots them against the Q8_K activation records `cpu_quantized::prepare` makes (eight 32-value records
//! per block sharing one f32 scale), the arithmetic of llama.cpp's `ggml_vec_dot_iq*_q8_K`:
//!
//! ```text
//! y = sum over blocks of  d_w * C * d_x * sum over groups of  scale * sum_j grid[j] * sign[j] * q8[j]
//! ```
//!
//! with `C` = 1/8 (IQ2_XXS, IQ2_XS, IQ2_S), 1/4 (IQ3_XXS) or 1 (IQ3_S). The inner sums are exact integers
//! (lattice bytes are at most 62, scales at most 31), so a kernel differs from the scalar reference only
//! in the order of its f32 accumulation. The decode is shared by the x86-64 (AVX2: `vpsignb`, `vpmaddubsw`
//! and `vpmaddwd`) and AArch64 (NEON: signed weights through `sdot`) kernels and is checked against the
//! scalar decoder `quant_iq::dequantize_iq`.
//!
//! A prompt keeps one decoded block in registers and runs up to [`TILE`] tokens over it, so the lattice
//! lookups are paid once per 16 tokens. Results do not depend on the batch size or on how rows are split
//! into panels.
use crate::quant_iq::tables::{IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};

pub(crate) const IQ2_XXS: u32 = 16;
pub(crate) const IQ2_XS: u32 = 17;
pub(crate) const IQ3_XXS: u32 = 18;
pub(crate) const IQ3_S: u32 = 21;
pub(crate) const IQ2_S: u32 = 22;

/// Values per block of every grid format.
#[cfg(test)]
const BLOCK_VALUES: usize = 256;
/// Activation records (32 values each) per block.
pub(crate) const GROUPS: usize = 8;
/// Tokens a decoded block serves at once.
#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_endian = "little")
))]
const TILE: usize = 16;

/// Whether `kind` is one of the formats of this module.
pub(crate) const fn handles(kind: u32) -> bool {
    matches!(kind, IQ2_XXS | IQ2_XS | IQ3_XXS | IQ3_S | IQ2_S)
}

/// Bytes of one 256-value block.
pub(crate) const fn block_bytes(kind: u32) -> usize {
    match kind {
        IQ2_XXS => 66,
        IQ2_XS => 74,
        IQ3_XXS => 98,
        IQ3_S => 110,
        _ => 82,
    }
}

/// Factor between the block scale `d` times the integer sum and the value: the fractions the formats
/// fold into their scale (`0.5 + s` times 1/4 or 1/2 is `(2 s + 1) / 8` or `/ 4`).
pub(crate) const fn unit(kind: u32) -> f32 {
    match kind {
        IQ3_XXS => 0.25,
        IQ3_S => 1.0,
        _ => 0.125,
    }
}

/// `+-1` bytes for the eight bits of `signs`: 0xff (-1) where the bit is set, 1 elsewhere.
const fn expand_signs(signs: usize) -> u64 {
    let mut word = 0u64;
    let mut bit = 0;
    while bit < 8 {
        let byte: u64 = if (signs >> bit) & 1 != 0 { 0xff } else { 1 };
        word |= byte << (8 * bit);
        bit += 1;
    }
    word
}

/// `expand_signs` of every sign byte (IQ2_S and IQ3_S store them whole).
static SIGNS64: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = expand_signs(i);
        i += 1;
    }
    table
};

/// `expand_signs` of the sign byte a 7-bit index stands for: bit 7 makes the parity of the byte even.
static KSIGNS64: [u64; 128] = {
    let mut table = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        table[i] = expand_signs(i | (((i as u32).count_ones() as usize & 1) << 7));
        i += 1;
    }
    table
};

/// One decoded group of 32 values.
#[cfg(any(test, all(target_arch = "aarch64", target_endian = "little")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Group {
    /// Lattice bytes in value order (little-endian words of 8 bytes).
    pub(crate) grid: [u64; 4],
    /// `+1` / `0xff` (-1) per value.
    pub(crate) sign: [u64; 4],
    /// Integer scales of values 0..16 and 16..32.
    pub(crate) scale: [i32; 2],
}

#[cfg(any(test, all(target_arch = "aarch64", target_endian = "little")))]
fn le16(raw: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

#[cfg(any(test, all(target_arch = "aarch64", target_endian = "little")))]
fn le32(raw: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
}

/// The block scale `d` of a block (its first two bytes, binary16) times [`unit`].
pub(crate) fn head(kind: u32, raw: &[u8]) -> f32 {
    half::f16::from_le_bytes([raw[0], raw[1]]).to_f32() * unit(kind)
}

/// Group `g` (0..8) of the block `raw` (`block_bytes(KIND)` bytes) of format `KIND`.
#[inline(always)]
#[cfg(any(test, all(target_arch = "aarch64", target_endian = "little")))]
pub(crate) fn group<const KIND: u32>(raw: &[u8], g: usize) -> Group {
    let mut out = Group::default();
    match KIND {
        IQ2_XXS => {
            let at = 2 + g * 8;
            let (lo, hi) = (le32(raw, at), le32(raw, at + 4));
            for l in 0..4 {
                out.grid[l] = IQ2XXS_GRID[((lo >> (8 * l)) & 0xff) as usize];
                out.sign[l] = KSIGNS64[((hi >> (7 * l)) & 0x7f) as usize];
            }
            out.scale = [2 * (hi >> 28) as i32 + 1; 2];
        }
        IQ2_XS => {
            for l in 0..4 {
                let q = le16(raw, 2 + g * 8 + l * 2);
                out.grid[l] = IQ2XS_GRID[q & 0x1ff];
                out.sign[l] = KSIGNS64[q >> 9];
            }
            let s = raw[66 + g];
            out.scale = [2 * i32::from(s & 15) + 1, 2 * i32::from(s >> 4) + 1];
        }
        IQ2_S => {
            let qh = usize::from(raw[66 + g]);
            for l in 0..4 {
                let index = usize::from(raw[2 + g * 4 + l]) | (((qh >> (2 * l)) & 3) << 8);
                out.grid[l] = IQ2S_GRID[index];
                out.sign[l] = SIGNS64[usize::from(raw[34 + g * 4 + l])];
            }
            let s = raw[74 + g];
            out.scale = [2 * i32::from(s & 15) + 1, 2 * i32::from(s >> 4) + 1];
        }
        IQ3_XXS => {
            let signs = le32(raw, 66 + g * 4);
            for l in 0..4 {
                let lo = IQ3XXS_GRID[usize::from(raw[2 + g * 8 + l * 2])];
                let hi = IQ3XXS_GRID[usize::from(raw[3 + g * 8 + l * 2])];
                out.grid[l] = u64::from(lo) | (u64::from(hi) << 32);
                out.sign[l] = KSIGNS64[((signs >> (7 * l)) & 0x7f) as usize];
            }
            out.scale = [2 * (signs >> 28) as i32 + 1; 2];
        }
        _ => {
            let qh = usize::from(raw[66 + g]);
            for l in 0..4 {
                let lo = usize::from(raw[2 + g * 8 + l * 2]) | (((qh >> (2 * l)) & 1) << 8);
                let hi = usize::from(raw[3 + g * 8 + l * 2]) | (((qh >> (2 * l + 1)) & 1) << 8);
                out.grid[l] = u64::from(IQ3S_GRID[lo]) | (u64::from(IQ3S_GRID[hi]) << 32);
                out.sign[l] = SIGNS64[usize::from(raw[74 + g * 4 + l])];
            }
            let pair = raw[106 + g / 2];
            let nibble = if g.is_multiple_of(2) { pair & 15 } else { pair >> 4 };
            out.scale = [1 + 2 * i32::from(nibble); 2];
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::{block_bytes, head, GROUPS, TILE};
    use crate::cpu_quantized::X;
    use crate::cpu_schedule::Rows;
    use std::arch::x86_64::*;

    /// `rows` consecutive output rows from `first` for every token of `x` (`batch * nb` activation
    /// records, `nb` per token).
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; `w` must hold complete rows of `nb / 8` blocks through row
    /// `first + valid` and `out` must be this panel's exclusive output view (at least `valid` wide).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn panel(
        w: &[u8],
        kind: u32,
        first: usize,
        valid: usize,
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        match kind {
            super::IQ2_XXS => panel_k::<16>(w, first, valid, x, nb, batch, out),
            super::IQ2_XS => panel_k::<17>(w, first, valid, x, nb, batch, out),
            super::IQ3_XXS => panel_k::<18>(w, first, valid, x, nb, batch, out),
            super::IQ3_S => panel_k::<21>(w, first, valid, x, nb, batch, out),
            super::IQ2_S => panel_k::<22>(w, first, valid, x, nb, batch, out),
            other => unreachable!("not a grid IQ format: {other}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn panel_k<const KIND: u32>(
        w: &[u8],
        first: usize,
        valid: usize,
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        assert!(nb.is_multiple_of(GROUPS) && x.len() >= batch * nb);
        let blocks = nb / GROUPS;
        let row_bytes = blocks * block_bytes(KIND);
        assert!(w.len() >= (first + valid) * row_bytes);
        for t0 in (0..batch).step_by(TILE) {
            let tokens = (batch - t0).min(TILE);
            for r in 0..valid {
                let row = &w[(first + r) * row_bytes..][..row_bytes];
                let sums = row_dot::<KIND>(row, blocks, &x[t0 * nb..(t0 + tokens) * nb], nb);
                for (t, &sum) in sums.iter().enumerate().take(tokens) {
                    out.row(t0 + t)[r] = hsum(sum);
                }
            }
        }
    }

    /// Decoded group in vector form: lattice bytes, +-1 sign bytes, 16-bit scales (low half for values
    /// 0..16, high half for 16..32).
    #[derive(Clone, Copy)]
    struct Vectors {
        grid: __m256i,
        sign: __m256i,
        scale: __m256i,
    }

    #[cfg(test)]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vectors(g: &super::Group) -> Vectors {
        let words = |w: &[u64; 4]| {
            _mm256_set_epi64x(w[3] as i64, w[2] as i64, w[1] as i64, w[0] as i64)
        };
        Vectors {
            grid: words(&g.grid),
            sign: words(&g.sign),
            scale: _mm256_set_m128i(
                _mm_set1_epi16(g.scale[1] as i16),
                _mm_set1_epi16(g.scale[0] as i16),
            ),
        }
    }

    /// Group `g` of a block of any grid format with two gathers instead of scalar table reads (twelve for IQ3_XXS, up to sixteen for
    /// the others) and a `set_epi64x` per vector. The lattice indices of a group are cut out of the packed bytes with widening and
    /// variable shifts, the lattice entries and the four sign words are gathered, the scales are broadcast. Same bytes as
    /// `vectors(&group::<KIND>(raw, g))`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vectors_gather<const KIND: u32>(raw: &[u8], g: usize) -> Vectors {
        assert!(raw.len() >= block_bytes(KIND) && g < GROUPS);
        let at = raw.as_ptr();
        let word = |offset: usize| std::ptr::read_unaligned(at.add(offset).cast::<u32>());
        let four = |offset: usize| _mm_cvtepu8_epi32(_mm_cvtsi32_si128(word(offset) as i32));
        // 7-bit sign indices of IQ2_XXS and IQ3_XXS: four of them in one 32-bit word, 28 bits, the scale in the top four.
        let ksign = |packed: u32| {
            let index = _mm_and_si128(
                _mm_srlv_epi32(_mm_set1_epi32(packed as i32), _mm_setr_epi32(0, 7, 14, 21)),
                _mm_set1_epi32(0x7f),
            );
            _mm256_i32gather_epi64::<8>(super::KSIGNS64.as_ptr().cast(), index)
        };
        let two = |low: i32, high: i32| _mm256_set_m128i(_mm_set1_epi16(high as i16), _mm_set1_epi16(low as i16));
        match KIND {
            super::IQ2_XXS => {
                let hi = word(2 + g * 8 + 4);
                let grid = _mm256_i32gather_epi64::<8>(super::IQ2XXS_GRID.as_ptr().cast(), four(2 + g * 8));
                let scale = (2 * (hi >> 28) + 1) as i32;
                Vectors { grid, sign: ksign(hi), scale: two(scale, scale) }
            }
            super::IQ2_XS => {
                let q = _mm_cvtepu16_epi32(_mm_loadl_epi64(at.add(2 + g * 8).cast()));
                let grid = _mm256_i32gather_epi64::<8>(super::IQ2XS_GRID.as_ptr().cast(), _mm_and_si128(q, _mm_set1_epi32(0x1ff)));
                let sign = _mm256_i32gather_epi64::<8>(super::KSIGNS64.as_ptr().cast(), _mm_srli_epi32::<9>(q));
                let s = i32::from(raw[66 + g]);
                Vectors { grid, sign, scale: two(2 * (s & 15) + 1, 2 * (s >> 4) + 1) }
            }
            super::IQ2_S => {
                let qh = _mm_set1_epi32(i32::from(raw[66 + g]));
                let high = _mm_slli_epi32::<8>(_mm_and_si128(_mm_srlv_epi32(qh, _mm_setr_epi32(0, 2, 4, 6)), _mm_set1_epi32(3)));
                let grid = _mm256_i32gather_epi64::<8>(super::IQ2S_GRID.as_ptr().cast(), _mm_or_si128(four(2 + g * 4), high));
                let sign = _mm256_i32gather_epi64::<8>(super::SIGNS64.as_ptr().cast(), four(34 + g * 4));
                let s = i32::from(raw[74 + g]);
                Vectors { grid, sign, scale: two(2 * (s & 15) + 1, 2 * (s >> 4) + 1) }
            }
            super::IQ3_XXS => {
                let signs = word(66 + g * 4);
                let index = _mm256_cvtepu8_epi32(_mm_loadl_epi64(at.add(2 + g * 8).cast()));
                let grid = _mm256_i32gather_epi32::<4>(super::IQ3XXS_GRID.as_ptr().cast(), index);
                let scale = (2 * (signs >> 28) + 1) as i32;
                Vectors { grid, sign: ksign(signs), scale: two(scale, scale) }
            }
            _ => {
                // IQ3_S: the ninth bit of each of the eight indices is bit k of the byte at 66 + g.
                let qh = _mm256_set1_epi32(i32::from(raw[66 + g]));
                let high = _mm256_slli_epi32::<8>(_mm256_and_si256(
                    _mm256_srlv_epi32(qh, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)),
                    _mm256_set1_epi32(1),
                ));
                let index = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64(at.add(2 + g * 8).cast())), high);
                let grid = _mm256_i32gather_epi32::<4>(super::IQ3S_GRID.as_ptr().cast(), index);
                let sign = _mm256_i32gather_epi64::<8>(super::SIGNS64.as_ptr().cast(), four(74 + g * 4));
                let pair = raw[106 + g / 2];
                let nibble = i32::from(if g.is_multiple_of(2) { pair & 15 } else { pair >> 4 });
                Vectors { grid, sign, scale: two(1 + 2 * nibble, 1 + 2 * nibble) }
            }
        }
    }

    /// Eight 32-bit lanes of `sum scale * lattice * sign * q` over one group.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot(v: &Vectors, record: &X) -> __m256i {
        let q = _mm256_loadu_si256(record.q.as_ptr().cast());
        // The lattice bytes are unsigned and at most 62: `maddubs` takes them as the unsigned operand
        // and the activations (sign applied first) as the signed one; a pair sums to at most 15748.
        let pairs = _mm256_maddubs_epi16(v.grid, _mm256_sign_epi8(q, v.sign));
        _mm256_madd_epi16(pairs, v.scale)
    }

    /// One row against up to [`TILE`] tokens (`x` holds `tokens * nb` records): per-lane f32 sums.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn row_dot<const KIND: u32>(
        row: &[u8],
        blocks: usize,
        x: &[X],
        nb: usize,
    ) -> [__m256; TILE] {
        let tokens = x.len() / nb;
        let width = block_bytes(KIND);
        let mut acc = [_mm256_setzero_ps(); TILE];
        for block in 0..blocks {
            let raw = &row[block * width..][..width];
            let d = head(KIND, raw);
            let decoded: [Vectors; GROUPS] = std::array::from_fn(|g| vectors_gather::<KIND>(raw, g));
            for (t, lanes) in acc.iter_mut().enumerate().take(tokens) {
                let records = &x[t * nb + block * GROUPS..][..GROUPS];
                let mut sum = _mm256_setzero_si256();
                for (v, record) in decoded.iter().zip(records) {
                    sum = _mm256_add_epi32(sum, dot(v, record));
                }
                let scale = _mm256_set1_ps(d * records[0].d);
                *lanes = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, *lanes);
            }
        }
        acc
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }

    #[cfg(test)]
    mod tests {
        use super::super::group;
        use super::*;
        use crate::cpu_selftest::{random_weights, Lcg};

        fn bytes(v: __m256i) -> [u8; 32] {
            let mut out = [0u8; 32];
            // SAFETY: `out` is 32 bytes; AVX is part of the AVX2 the test checks for.
            unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
            out
        }

        /// The gathered group is the group of the scalar decoder, byte for byte (lattice, signs and scales), for random blocks of every
        /// grid format.
        #[test]
        fn the_gathered_group_is_the_group_of_the_scalar_decoder() {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
            fn check<const KIND: u32>(rng: &mut Lcg) {
                let width = block_bytes(KIND);
                let weights = random_weights(KIND, 64, 1024, rng);
                for at in (0..weights.len()).step_by(width) {
                    let raw = &weights[at..at + width];
                    for g in 0..GROUPS {
                        // SAFETY: AVX2 was detected by the caller; `raw` is one whole block.
                        let (want, have) = unsafe { (vectors(&group::<KIND>(raw, g)), vectors_gather::<KIND>(raw, g)) };
                        assert_eq!(bytes(have.grid), bytes(want.grid), "lattice of kind {KIND} group {g}");
                        assert_eq!(bytes(have.sign), bytes(want.sign), "signs of kind {KIND} group {g}");
                        assert_eq!(bytes(have.scale), bytes(want.scale), "scales of kind {KIND} group {g}");
                    }
                }
            }
            let mut rng = Lcg(5);
            check::<16>(&mut rng);
            check::<17>(&mut rng);
            check::<18>(&mut rng);
            check::<21>(&mut rng);
            check::<22>(&mut rng);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod neon {
    use super::{block_bytes, group, head, Group, GROUPS, TILE};
    use crate::cpu_quantized::X;
    use crate::cpu_schedule::Rows;
    use std::arch::aarch64::*;

    /// `valid` consecutive output rows from `first` for every token of `x` (`batch * nb` activation
    /// records, `nb` per token).
    ///
    /// # Safety
    /// The CPU must support NEON and dotprod; `w` must hold complete rows of `nb / 8` blocks through
    /// row `first + valid` and `out` must be this panel's exclusive output view (at least `valid` wide).
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod")]
    pub(crate) unsafe fn panel(
        w: &[u8],
        kind: u32,
        first: usize,
        valid: usize,
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        match kind {
            super::IQ2_XXS => panel_k::<16>(w, first, valid, x, nb, batch, out),
            super::IQ2_XS => panel_k::<17>(w, first, valid, x, nb, batch, out),
            super::IQ3_XXS => panel_k::<18>(w, first, valid, x, nb, batch, out),
            super::IQ3_S => panel_k::<21>(w, first, valid, x, nb, batch, out),
            super::IQ2_S => panel_k::<22>(w, first, valid, x, nb, batch, out),
            other => unreachable!("not a grid IQ format: {other}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn panel_k<const KIND: u32>(
        w: &[u8],
        first: usize,
        valid: usize,
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        assert!(nb.is_multiple_of(GROUPS) && x.len() >= batch * nb);
        let blocks = nb / GROUPS;
        let row_bytes = blocks * block_bytes(KIND);
        assert!(w.len() >= (first + valid) * row_bytes);
        for t0 in (0..batch).step_by(TILE) {
            let tokens = (batch - t0).min(TILE);
            for r in 0..valid {
                let row = &w[(first + r) * row_bytes..][..row_bytes];
                let sums = row_dot::<KIND>(row, blocks, &x[t0 * nb..(t0 + tokens) * nb], nb);
                for (t, &sum) in sums.iter().enumerate().take(tokens) {
                    out.row(t0 + t)[r] = vaddvq_f32(sum);
                }
            }
        }
    }

    /// One decoded group: signed weights (lattice byte times sign) of values 0..16 and 16..32 and their
    /// integer scales.
    #[derive(Clone, Copy)]
    struct Weights {
        w: [int8x16_t; 2],
        scale: [i32; 2],
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn weights(g: &Group) -> Weights {
        let half = |i: usize| {
            let grid = vcombine_u8(vcreate_u8(g.grid[i]), vcreate_u8(g.grid[i + 1]));
            let sign = vcombine_u8(vcreate_u8(g.sign[i]), vcreate_u8(g.sign[i + 1]));
            vmulq_s8(vreinterpretq_s8_u8(grid), vreinterpretq_s8_u8(sign))
        };
        Weights {
            w: [half(0), half(2)],
            scale: g.scale,
        }
    }

    /// Four 32-bit lanes of `sum scale * weight * q` over one group.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn dot(v: &Weights, record: &X, acc: int32x4_t) -> int32x4_t {
        let zero = vdupq_n_s32(0);
        let lo = vdotq_s32(zero, v.w[0], vld1q_s8(record.q.as_ptr()));
        let hi = vdotq_s32(zero, v.w[1], vld1q_s8(record.q.as_ptr().add(16)));
        vmlaq_n_s32(vmlaq_n_s32(acc, lo, v.scale[0]), hi, v.scale[1])
    }

    /// One row against up to [`TILE`] tokens (`x` holds `tokens * nb` records): per-lane f32 sums.
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn row_dot<const KIND: u32>(
        row: &[u8],
        blocks: usize,
        x: &[X],
        nb: usize,
    ) -> [float32x4_t; TILE] {
        let tokens = x.len() / nb;
        let width = block_bytes(KIND);
        let mut acc = [vdupq_n_f32(0.0); TILE];
        for block in 0..blocks {
            let raw = &row[block * width..][..width];
            let d = head(KIND, raw);
            let decoded: [Weights; GROUPS] = std::array::from_fn(|g| weights(&group::<KIND>(raw, g)));
            for (t, lanes) in acc.iter_mut().enumerate().take(tokens) {
                let records = &x[t * nb + block * GROUPS..][..GROUPS];
                let mut sum = vdupq_n_s32(0);
                for (v, record) in decoded.iter().zip(records) {
                    sum = dot(v, record, sum);
                }
                let scale = vdupq_n_f32(d * records[0].d);
                *lanes = vfmaq_f32(*lanes, vcvtq_f32_s32(sum), scale);
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_iq::tables::KSIGNS_IQ2XS;
    use crate::cpu_quantized::{linear, linear_group, prepare, Projection};
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    const KINDS: [u32; 5] = [IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S];

    fn block_group(kind: u32, raw: &[u8], g: usize) -> Group {
        match kind {
            IQ2_XXS => group::<16>(raw, g),
            IQ2_XS => group::<17>(raw, g),
            IQ3_XXS => group::<18>(raw, g),
            IQ3_S => group::<21>(raw, g),
            _ => group::<22>(raw, g),
        }
    }

    #[test]
    fn sign_tables_match_the_ggml_sign_bytes() {
        for (i, &byte) in KSIGNS_IQ2XS.iter().enumerate() {
            assert_eq!(KSIGNS64[i], SIGNS64[usize::from(byte)], "index {i}");
        }
        assert_eq!(SIGNS64[0], 0x0101_0101_0101_0101);
        assert_eq!(SIGNS64[0x81], 0xff01_0101_0101_01ff);
    }

    #[test]
    fn decoded_groups_equal_the_scalar_dequantizer() {
        let mut rng = Lcg(31);
        for kind in KINDS {
            let weights = random_weights(kind, 6, 512, &mut rng);
            let width = block_bytes(kind);
            for raw in weights.chunks_exact(width) {
                let want = crate::quant_iq::dequantize_iq(raw, kind, BLOCK_VALUES).unwrap();
                let d = head(kind, raw);
                for g in 0..GROUPS {
                    let grp = block_group(kind, raw, g);
                    for j in 0..32 {
                        let grid = ((grp.grid[j / 8] >> (8 * (j % 8))) & 0xff) as f32;
                        let sign = ((grp.sign[j / 8] >> (8 * (j % 8))) & 0xff) as u8 as i8 as f32;
                        let have = d * grp.scale[j / 16] as f32 * grid * sign;
                        let want = want[g * 32 + j];
                        assert!(
                            (have - want).abs() <= 1e-6 * want.abs().max(1e-3),
                            "kind {kind} group {g} value {j}: {have} vs {want}"
                        );
                    }
                }
            }
        }
    }

    /// The projection, its single-token and batched forms, a grouped call and every thread count agree with
    /// the f64 reference over the same 8-bit activations, and do not depend on the batch or the panel.
    #[test]
    fn projections_match_the_scalar_reference_for_odd_shapes_batches_and_threads() {
        if !crate::cpu_kernels::packed_simd() {
            return;
        }
        let mut rng = Lcg(2026);
        for kind in KINDS {
            for (rows, cols) in [(1usize, 256usize), (13, 768), (6, 2560), (33, 512)] {
                let weights = random_weights(kind, rows, cols, &mut rng);
                let bias = rng.vector(rows, 1.0);
                for threads in [1usize, 3, 10] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    for batch in [1usize, 2, 3, 5, 16, 17, 40] {
                        let x = rng.vector(batch * cols, 3.0);
                        let packed = prepare(&x, cols, kind);
                        let have = pool
                            .install(|| linear(&weights, kind, &x, rows, cols, Some(&bias)))
                            .expect("a grid kernel on this CPU");
                        let (mut want, scale) = reference_packed(&weights, kind, rows, cols, &packed);
                        for (j, w) in want.iter_mut().enumerate() {
                            *w += f64::from(bias[j % rows]);
                        }
                        let (error, ok) = compare(&have, &want, &scale);
                        assert!(
                            ok,
                            "kind {kind} rows {rows} cols {cols} batch {batch} threads {threads}: {error:e}"
                        );
                        // token t of a batch equals the same token alone, bit for bit
                        for t in [0, batch - 1] {
                            let one = pool
                                .install(|| {
                                    linear(&weights, kind, &x[t * cols..(t + 1) * cols], rows, cols, Some(&bias))
                                })
                                .unwrap();
                            assert_eq!(
                                have[t * rows..(t + 1) * rows],
                                one[..],
                                "kind {kind} rows {rows} batch {batch} token {t}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Grid formats mix with each other and with the K formats of one Q8_K group.
    #[test]
    fn mixed_groups_match_single_format_projections() {
        if !crate::cpu_kernels::packed_simd() {
            return;
        }
        let mut rng = Lcg(77);
        let cols = 768;
        let kinds = [IQ2_XXS, 12, IQ3_S, 10, IQ2_XS, IQ2_S, IQ3_XXS, 14];
        let rows = [7usize, 13, 5, 9, 16, 3, 11, 4];
        let weights = kinds
            .iter()
            .zip(rows)
            .map(|(&k, r)| random_weights(k, r, cols, &mut rng))
            .collect::<Vec<_>>();
        let members = weights
            .iter()
            .zip(rows)
            .map(|(w, rows)| Projection { weights: w, rows, bias: None })
            .collect::<Vec<_>>();
        for batch in [1usize, 4, 19] {
            let x = rng.vector(batch * cols, 3.0);
            let packed = prepare(&x, cols, IQ2_XXS);
            let grouped = crate::cpu_quantized::linear_mixed(&members, &kinds, &x, cols).unwrap();
            for ((&kind, w), (have, &rows)) in kinds.iter().zip(&weights).zip(grouped.iter().zip(&rows)) {
                if handles(kind) {
                    let alone = linear(w, kind, &x, rows, cols, None).unwrap();
                    assert_eq!(*have, alone, "kind {kind} batch {batch}");
                } else {
                    // K members take other kernels alone (eight-row panels): only the reference is exact
                    let (want, scale) = reference_packed(w, kind, rows, cols, &packed);
                    let (error, ok) = compare(have, &want, &scale);
                    assert!(ok, "kind {kind} batch {batch}: {error:e}");
                }
            }
        }
        // a single format through `linear_group` too
        let x = rng.vector(cols, 3.0);
        let one = linear_group(&members[..1], IQ2_XXS, &x, cols).unwrap();
        assert_eq!(one[0], linear(&weights[0], IQ2_XXS, &x, rows[0], cols, None).unwrap());
    }
}
