//! Q4Z / Q4ZP projections (4-bit AWQ and GPTQ weights, groups of 32, 64 or 128 inputs) on x86-64.
//!
//! A row stores, per group of `G` inputs, `G / 2` bytes of 4-bit codes (input `2j` in the low
//! nibble of byte `j`, input `2j + 1` in the high nibble), an fp16 scale and a zero-point byte;
//! the weight is `(code - zero) * scale`. Q4ZP rows hold the same blocks over inputs sorted by
//! quantization group, followed by the permutation table `perm` (`u32` per input, stored
//! position `c` belongs to input `perm[c]`) after the last row.
//!
//! The kernels follow the K-quant kernels: the activations are quantized to 8 bits once per call
//! (32-value records with an fp16-rounded scale, `cpu_quantized::prepare`), the codes are used as
//! unsigned bytes in the integer dot (`vpdpbusd`, or `vpmaddubsw` + `vpmaddwd`; a pair of codes
//! times activations is at most `2 * 15 * 127`, so nothing saturates) and the zero point is
//! applied to the whole group: `sum((q - z) * a) = sum(q * a) - z * sum(a)`, with the activation
//! sums taken from the records. A Q4ZP projection first gathers its input into stored order,
//! `x'[c] = x[perm[c]]`, then runs the same kernels.
//!
//! Layout trick: because the nibbles of a code byte are *adjacent* inputs, `v & 15` holds the
//! even inputs of 32 consecutive bytes and `v >> 4` the odd ones. The activations are stored
//! split the same way (`even` and `odd` planes, 16 bytes per record), so a 256-bit load of
//! codes covers two records (64 inputs) and needs no shuffle: two integer dots, one against the
//! even plane and one against the odd plane, and the lanes of each 128-bit half belong to one
//! record, whose activation scale (times the group's weight scale) is applied per half.
//!
//! Test level: see `docs/cpu-kernels.md`; `selftest-cpu` checks every tier against an f64
//! reference on the same records.
use crate::cpu_quantized::X;
use crate::cpu_schedule::Rows;
use std::arch::x86_64::*;

/// Bytes of one block besides the codes: fp16 scale and zero-point byte.
const TAIL: usize = 3;

/// Activations of a call in the split layout, padded to a multiple of four records per token.
pub(crate) struct Acts {
    /// Records per token including the zero records that pad it to a multiple of four.
    stride: usize,
    /// Even inputs of each record (16 bytes), `stride` records per token.
    even: Vec<i8>,
    /// Odd inputs of each record.
    odd: Vec<i8>,
    /// Activation scale of each record, four copies (one 128-bit half of a float vector).
    scale4: Vec<f32>,
    /// `scale * sum(q)` of each record.
    sum: Vec<f32>,
}

impl Acts {
    /// Split the 32-value records of `batch` tokens (`records.len() / batch` per token).
    pub(crate) fn new(records: &[X], batch: usize) -> Acts {
        let nb = records.len() / batch;
        let stride = nb.next_multiple_of(4);
        let mut acts = Acts {
            stride,
            even: vec![0; batch * stride * 16],
            odd: vec![0; batch * stride * 16],
            scale4: vec![0.0; batch * stride * 4],
            sum: vec![0.0; batch * stride],
        };
        for t in 0..batch {
            for (b, record) in records[t * nb..(t + 1) * nb].iter().enumerate() {
                let at = t * stride + b;
                for i in 0..16 {
                    acts.even[at * 16 + i] = record.q[2 * i];
                    acts.odd[at * 16 + i] = record.q[2 * i + 1];
                }
                acts.scale4[at * 4..at * 4 + 4].fill(record.d);
                acts.sum[at] = record.sum[0] + record.sum[1];
            }
        }
        acts
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
}

/// `sum(lo * even) + sum(hi * odd)` per 32-bit lane for unsigned codes against signed
/// activations: AVX-VNNI.
#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot_vnni(lo: __m256i, hi: __m256i, even: __m256i, odd: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), lo, even),
        hi,
        odd,
    )
}

/// The same on plain AVX2: a `maddubs` pair sums to at most 2 * 15 * 127 = 3810, so adding the
/// even and the odd half in 16 bits (7620) cannot saturate.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lo: __m256i, hi: __m256i, even: __m256i, odd: __m256i) -> __m256i {
    let pairs = _mm256_add_epi16(
        _mm256_maddubs_epi16(lo, even),
        _mm256_maddubs_epi16(hi, odd),
    );
    _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))
}

macro_rules! kernel {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// Rows `first..first + rows` (1 to 4) of a projection for every token of `acts`.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` must hold `first + rows` complete rows
        /// of `cols / group` blocks, `acts` the records of `batch` tokens of `cols` values.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            group: usize,
            cols: usize,
            first: usize,
            rows: usize,
            acts: &Acts,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            macro_rules! by_group {
                ($g:literal) => {{
                    let mut r0 = 0;
                    while r0 < rows {
                        let r = match rows - r0 {
                            1 => 1,
                            2 | 3 => 2,
                            _ => 4,
                        };
                        let mut token = 0;
                        while token < batch {
                            let t = if batch - token >= 4 { 4 } else { 1 };
                            match (r, t) {
                                (1, 1) => {
                                    $body::<$g, 1, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (2, 1) => {
                                    $body::<$g, 2, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (4, 1) => {
                                    $body::<$g, 4, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (1, _) => {
                                    $body::<$g, 1, 4>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (2, _) => {
                                    $body::<$g, 2, 4>(w, cols, first + r0, acts, token, r0, out)
                                }
                                _ => $body::<$g, 4, 4>(w, cols, first + r0, acts, token, r0, out),
                            }
                            token += t;
                        }
                        r0 += r;
                    }
                }};
            }
            match group {
                32 => by_group!(32),
                64 => by_group!(64),
                128 => by_group!(128),
                _ => unreachable!("Q4Z groups hold 32, 64 or 128 inputs"),
            }
        }

        /// `R` rows by `T` tokens; the outputs go to columns `col..col + R` of tokens
        /// `token..token + T`.
        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const G: usize, const R: usize, const T: usize>(
            w: &[u8],
            cols: usize,
            row0: usize,
            acts: &Acts,
            token: usize,
            col: usize,
            out: &mut Rows<'_>,
        ) {
            let block = G / 2 + TAIL;
            let row_bytes = cols / G * block;
            let records = cols / 32;
            let records_per_group = G / 32;
            let low4 = _mm256_set1_epi8(15);
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            let mut correction = [[0.0f32; T]; R];
            // One unit is two records (64 inputs): a 256-bit load of codes.
            for unit in 0..records.div_ceil(2) {
                let record = 2 * unit;
                let pair = record + 1 < records;
                let mut even = [_mm256_setzero_si256(); T];
                let mut odd = [_mm256_setzero_si256(); T];
                let mut scale = [_mm256_setzero_ps(); T];
                let mut asum = [[0.0f32; 2]; T];
                for t in 0..T {
                    let at = (token + t) * acts.stride + record;
                    even[t] = _mm256_loadu_si256(acts.even.as_ptr().add(at * 16).cast());
                    odd[t] = _mm256_loadu_si256(acts.odd.as_ptr().add(at * 16).cast());
                    scale[t] = _mm256_loadu_ps(acts.scale4.as_ptr().add(at * 4));
                    asum[t] = [acts.sum[at], acts.sum[at + 1]];
                }
                for r in 0..R {
                    let row = w.as_ptr().add((row0 + r) * row_bytes);
                    let (codes, weight_scale);
                    if G >= 64 {
                        // Both records lie in one group: 32 contiguous code bytes.
                        let group = record / records_per_group;
                        let base = row.add(group * block);
                        let offset = record % records_per_group * 16;
                        let scale_bits =
                            u16::from_le_bytes([*base.add(G / 2), *base.add(G / 2 + 1)]);
                        let s = crate::cpu_kernels::x86::half_to_float(scale_bits);
                        let z = f32::from(*base.add(G / 2 + 2) & 15);
                        codes = if pair {
                            _mm256_loadu_si256(base.add(offset).cast())
                        } else {
                            _mm256_zextsi128_si256(_mm_loadu_si128(base.add(offset).cast()))
                        };
                        weight_scale = _mm256_set1_ps(s);
                        for t in 0..T {
                            correction[r][t] += s * z * (asum[t][0] + asum[t][1]);
                        }
                    } else {
                        // One group per record: two 16-byte code runs, one scale per half.
                        let base = row.add(record * block);
                        let scale_bits = u16::from_le_bytes([*base.add(16), *base.add(17)]);
                        let s0 = crate::cpu_kernels::x86::half_to_float(scale_bits);
                        let z0 = f32::from(*base.add(18) & 15);
                        let low = _mm_loadu_si128(base.cast());
                        let (high, s1, z1);
                        if pair {
                            let next = base.add(block);
                            high = _mm_loadu_si128(next.cast());
                            s1 = crate::cpu_kernels::x86::half_to_float(u16::from_le_bytes([
                                *next.add(16),
                                *next.add(17),
                            ]));
                            z1 = f32::from(*next.add(18) & 15);
                        } else {
                            high = _mm_setzero_si128();
                            (s1, z1) = (0.0, 0.0);
                        }
                        codes = _mm256_set_m128i(high, low);
                        weight_scale = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
                        for t in 0..T {
                            correction[r][t] += s0 * z0 * asum[t][0] + s1 * z1 * asum[t][1];
                        }
                    }
                    let lo = _mm256_and_si256(codes, low4);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(codes), low4);
                    for t in 0..T {
                        let dot = $dot(lo, hi, even[t], odd[t]);
                        acc[r][t] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(dot),
                            _mm256_mul_ps(scale[t], weight_scale),
                            acc[r][t],
                        );
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t)[col..].iter_mut().take(R).enumerate() {
                    *y = hsum(acc[r][t]) - correction[r][t];
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

/// `dot_vnni` on 512-bit vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot512_vnni(lo: __m512i, hi: __m512i, even: __m512i, odd: __m512i) -> __m512i {
    _mm512_dpbusd_epi32(
        _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, even),
        hi,
        odd,
    )
}

/// `dot_avx2` on 512-bit vectors (AVX-512BW).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot512_bw(lo: __m512i, hi: __m512i, even: __m512i, odd: __m512i) -> __m512i {
    let pairs = _mm512_add_epi16(
        _mm512_maddubs_epi16(lo, even),
        _mm512_maddubs_epi16(hi, odd),
    );
    _mm512_madd_epi16(pairs, _mm512_set1_epi16(1))
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn hsum512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Code bytes of the records `record..record + 4` of one row in a 512-bit vector (lane `i`
/// holds record `record + i`, lanes past the end of the row are zero), the weight scale of each
/// lane in a float vector, and the scale and zero point of each lane's group.
///
/// # Safety
/// `row` must point at a complete row of `records` records.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
unsafe fn load_codes512<const G: usize>(
    row: *const u8,
    record: usize,
    records: usize,
) -> (__m512i, __m512, [f32; 4], [f32; 4]) {
    let block = G / 2 + TAIL;
    let per_group = G / 32;
    let valid = (records - record).min(4);
    let params = |group: usize| {
        let base = row.add(group * block);
        let bits = u16::from_le_bytes([*base.add(G / 2), *base.add(G / 2 + 1)]);
        (
            crate::cpu_kernels::x86::half_to_float(bits),
            f32::from(*base.add(G / 2 + 2) & 15),
        )
    };
    if G == 128 && valid == 4 {
        // One group: 64 contiguous code bytes.
        let (s, z) = params(record / 4);
        let codes = _mm512_loadu_si512(row.add(record / 4 * block).cast());
        return (codes, _mm512_set1_ps(s), [s; 4], [z; 4]);
    }
    let mut codes = _mm512_setzero_si512();
    let mut weight_scale = _mm512_setzero_ps();
    let mut scales = [0.0f32; 4];
    let mut zeros = [0.0f32; 4];
    for lane in 0..valid {
        let group = (record + lane) / per_group;
        let offset = (record + lane) % per_group * 16;
        let (s, z) = params(group);
        let chunk = _mm_loadu_si128(row.add(group * block + offset).cast());
        codes = match lane {
            0 => _mm512_inserti32x4::<0>(codes, chunk),
            1 => _mm512_inserti32x4::<1>(codes, chunk),
            2 => _mm512_inserti32x4::<2>(codes, chunk),
            _ => _mm512_inserti32x4::<3>(codes, chunk),
        };
        weight_scale = _mm512_mask_mov_ps(weight_scale, 0xf << (4 * lane), _mm512_set1_ps(s));
        scales[lane] = s;
        zeros[lane] = z;
    }
    (codes, weight_scale, scales, zeros)
}

macro_rules! kernel512 {
    ($name:ident, $body:ident, $features:literal, $dot:ident) => {
        /// The 256-bit panel kernel on 512-bit vectors: four records (128 inputs) per step.
        ///
        /// # Safety
        /// As the 256-bit kernels, with the 512-bit features named in the attribute.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            group: usize,
            cols: usize,
            first: usize,
            rows: usize,
            acts: &Acts,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            macro_rules! by_group {
                ($g:literal) => {{
                    let mut r0 = 0;
                    while r0 < rows {
                        let r = match rows - r0 {
                            1 => 1,
                            2 | 3 => 2,
                            _ => 4,
                        };
                        let mut token = 0;
                        while token < batch {
                            let t = if batch - token >= 4 { 4 } else { 1 };
                            match (r, t) {
                                (1, 1) => {
                                    $body::<$g, 1, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (2, 1) => {
                                    $body::<$g, 2, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (4, 1) => {
                                    $body::<$g, 4, 1>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (1, _) => {
                                    $body::<$g, 1, 4>(w, cols, first + r0, acts, token, r0, out)
                                }
                                (2, _) => {
                                    $body::<$g, 2, 4>(w, cols, first + r0, acts, token, r0, out)
                                }
                                _ => $body::<$g, 4, 4>(w, cols, first + r0, acts, token, r0, out),
                            }
                            token += t;
                        }
                        r0 += r;
                    }
                }};
            }
            match group {
                32 => by_group!(32),
                64 => by_group!(64),
                128 => by_group!(128),
                _ => unreachable!("Q4Z groups hold 32, 64 or 128 inputs"),
            }
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const G: usize, const R: usize, const T: usize>(
            w: &[u8],
            cols: usize,
            row0: usize,
            acts: &Acts,
            token: usize,
            col: usize,
            out: &mut Rows<'_>,
        ) {
            let row_bytes = cols / G * (G / 2 + TAIL);
            let records = cols / 32;
            let low4 = _mm512_set1_epi8(15);
            let mut acc = [[_mm512_setzero_ps(); T]; R];
            let mut correction = [[0.0f32; T]; R];
            for unit in 0..records.div_ceil(4) {
                let record = 4 * unit;
                let mut even = [_mm512_setzero_si512(); T];
                let mut odd = [_mm512_setzero_si512(); T];
                let mut scale = [_mm512_setzero_ps(); T];
                let mut asum = [[0.0f32; 4]; T];
                for t in 0..T {
                    let at = (token + t) * acts.stride + record;
                    even[t] = _mm512_loadu_si512(acts.even.as_ptr().add(at * 16).cast());
                    odd[t] = _mm512_loadu_si512(acts.odd.as_ptr().add(at * 16).cast());
                    scale[t] = _mm512_loadu_ps(acts.scale4.as_ptr().add(at * 4));
                    asum[t].copy_from_slice(&acts.sum[at..at + 4]);
                }
                for r in 0..R {
                    let row = w.as_ptr().add((row0 + r) * row_bytes);
                    let (codes, weight_scale, s, z) = load_codes512::<G>(row, record, records);
                    // Zero-point term: records of one group share scale and zero point, so
                    // their activation sums are added first.
                    for t in 0..T {
                        let a = asum[t];
                        correction[r][t] += match G {
                            128 => s[0] * z[0] * ((a[0] + a[1]) + (a[2] + a[3])),
                            64 => s[0] * z[0] * (a[0] + a[1]) + s[2] * z[2] * (a[2] + a[3]),
                            _ => {
                                s[0] * z[0] * a[0]
                                    + s[1] * z[1] * a[1]
                                    + s[2] * z[2] * a[2]
                                    + s[3] * z[3] * a[3]
                            }
                        };
                    }
                    let lo = _mm512_and_si512(codes, low4);
                    let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(codes), low4);
                    for t in 0..T {
                        let dot = $dot(lo, hi, even[t], odd[t]);
                        acc[r][t] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(dot),
                            _mm512_mul_ps(scale[t], weight_scale),
                            acc[r][t],
                        );
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t)[col..].iter_mut().take(R).enumerate() {
                    *y = hsum512(acc[r][t]) - correction[r][t];
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

/// Rows of a panel: four rows share each activation load.
const PANEL_ROWS: usize = 4;

/// Whether `kind` is a Q4Z / Q4ZP type this module computes.
pub(crate) fn handles(kind: u32) -> bool {
    fllm_gguf_probe::q4z_group(kind).is_some()
}

/// A panel kernel: rows `first..first + rows` of every token.
type Panel = unsafe fn(&[u8], usize, usize, usize, usize, &Acts, usize, &mut Rows<'_>);

/// The widest kernel the CPU and the tier cap allow.
fn best_panel() -> Panel {
    use crate::cpu_kernels::Avx512;
    match crate::cpu_kernels::avx512() {
        Some(Avx512::Vnni) => panel512_vnni,
        Some(Avx512::Bw) => panel512_bw,
        None if crate::cpu_kernels::isa() == "avx-vnni" => panel_vnni,
        None => panel_avx2,
    }
}

/// `y = x * W^T + bias` for a Q4Z or Q4ZP matrix of `rows` rows over `cols` inputs and `x` of
/// `x.len() / cols` tokens. `None` when the CPU has no SIMD tier, the shape is not one of
/// the format's, or the permutation table is out of range (callers then use the generic path).
pub(crate) fn linear(
    weights: &[u8],
    kind: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    // SAFETY: `best_panel` returns a kernel whose features the tier check has confirmed.
    unsafe { linear_with(None, weights, kind, x, rows, cols, bias) }
}

/// [`linear`] on a chosen kernel (`None`: the best one).
///
/// # Safety
/// A given `panel` needs the CPU features it was compiled for.
unsafe fn linear_with(
    panel: Option<Panel>,
    weights: &[u8],
    kind: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let group = usize::try_from(fllm_gguf_probe::q4z_group(kind)?).ok()?;
    if !crate::cpu_kernels::packed_simd()
        || rows == 0
        || cols == 0
        || !cols.is_multiple_of(group)
        || !x.len().is_multiple_of(cols)
        || bias.is_some_and(|b| b.len() != rows)
    {
        return None;
    }
    let batch = x.len() / cols;
    let row_bytes = cols / group * (group / 2 + TAIL);
    let permuted = fllm_gguf_probe::q4zp_group(kind).is_some();
    if weights.len() != rows * row_bytes + if permuted { cols * 4 } else { 0 } {
        return None;
    }
    let gathered;
    let input = if permuted {
        let table = &weights[rows * row_bytes..];
        let mut values = Vec::with_capacity(x.len());
        for token in x.chunks_exact(cols) {
            for entry in table.as_chunks::<4>().0 {
                let source = u32::from_le_bytes(*entry) as usize;
                values.push(*token.get(source)?);
            }
        }
        gathered = values;
        &gathered[..]
    } else {
        x
    };
    // Q8_0-style records (fp16-rounded scale per 32 values), the packing of Q4_0 and Q8_0.
    let records = crate::cpu_quantized::prepare(input, cols, 2);
    let acts = Acts::new(&records, batch);
    drop(records);
    let mut output = vec![vec![0.0f32; batch * rows]];
    let panel = panel.unwrap_or_else(best_panel);
    // Decode streams the weights once, so a panel of four rows; prompts use the same panel and
    // four tokens per pass.
    crate::cpu_schedule::row_panels(
        &mut output,
        &[rows],
        batch,
        PANEL_ROWS,
        || (),
        |_, _, first, out| {
            let valid = (rows - first).min(PANEL_ROWS);
            // SAFETY: the caller vouches for the kernel's features; the length checks above bound
            // every row, and the scheduler gives `out` these columns exclusively until the queue
            // completes.
            unsafe { panel(weights, group, cols, first, valid, &acts, batch, out) };
            if let Some(bias) = bias {
                for token in 0..batch {
                    for (r, y) in out.row(token).iter_mut().enumerate() {
                        *y += bias[first + r];
                    }
                }
            }
        },
    );
    output.pop()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare_with, random_weights, Lcg};

    /// f64 reference on the same records: dequantized stored-order weights times the 8-bit
    /// activations the kernel uses (gathered into stored order for Q4ZP).
    fn reference(
        weights: &[u8],
        kind: u32,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let group = fllm_gguf_probe::q4z_group(kind).unwrap() as usize;
        let row_bytes = cols / group * (group / 2 + TAIL);
        let batch = x.len() / cols;
        let gathered: Vec<f32> = if fllm_gguf_probe::q4zp_group(kind).is_some() {
            let table = &weights[rows * row_bytes..];
            x.chunks_exact(cols)
                .flat_map(|token| {
                    table
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|e| token[u32::from_le_bytes(*e) as usize])
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            x.to_vec()
        };
        let records = crate::cpu_quantized::prepare(&gathered, cols, 2);
        let nb = cols / 32;
        let plain = match kind {
            0x8000_0014 => 0x8000_0010,
            0x8000_0015 => 0x8000_0011,
            0x8000_0016 => 0x8000_0012,
            other => other,
        };
        let mut want = vec![0.0; batch * rows];
        let mut scale = vec![0.0; batch * rows];
        for r in 0..rows {
            let row = crate::compute::dequantize(
                &weights[r * row_bytes..(r + 1) * row_bytes],
                plain,
                cols,
            )
            .unwrap();
            for t in 0..batch {
                let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
                for (b, record) in records[t * nb..(t + 1) * nb].iter().enumerate() {
                    for (&w, &q) in row[b * 32..b * 32 + 32].iter().zip(&record.q) {
                        let term = f64::from(w) * f64::from(q) * f64::from(record.d);
                        sum += term;
                        magnitude += term.abs();
                    }
                }
                want[t * rows + r] = sum;
                scale[t * rows + r] = magnitude;
            }
        }
        (want, scale)
    }

    /// Every kernel this CPU can run, by name. The tests call them directly instead of moving the
    /// process-wide tier cap, which other tests running in parallel would see.
    fn kernels() -> Vec<(&'static str, Panel)> {
        let mut found: Vec<(&'static str, Panel)> = Vec::new();
        let base = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        if base {
            found.push(("avx2", panel_avx2));
            if is_x86_feature_detected!("avxvnni") {
                found.push(("avx-vnni", panel_vnni));
            }
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
            {
                found.push(("avx512bw", panel512_bw));
                if is_x86_feature_detected!("avx512vnni") {
                    found.push(("avx512-vnni", panel512_vnni));
                }
            }
        }
        found
    }

    #[test]
    fn every_kernel_matches_the_reference_on_ragged_shapes() {
        let mut rng = Lcg(0x4a2b_7e11);
        let mut checked = 0;
        for (name, panel) in kernels() {
            for kind in [
                0x8000_0010,
                0x8000_0011,
                0x8000_0012,
                0x8000_0014,
                0x8000_0015,
                0x8000_0016,
            ] {
                let group = fllm_gguf_probe::q4z_group(kind).unwrap() as usize;
                for blocks in [1, 2, 3, 5] {
                    let cols = group * blocks;
                    for rows in [1, 3, 4, 5, 7, 13] {
                        let weights = random_weights(kind, rows, cols, &mut rng);
                        for batch in [1, 2, 3, 4, 5, 9] {
                            let x = rng.vector(batch * cols, 3.0);
                            // SAFETY: `kernels` lists only kernels whose features were detected.
                            let have = unsafe {
                                linear_with(Some(panel), &weights, kind, &x, rows, cols, None)
                            }
                            .expect("a SIMD tier is active");
                            let (want, scale) = reference(&weights, kind, &x, rows, cols);
                            let (error, ok) = compare_with(&have, &want, &scale, 1e-5);
                            checked += 1;
                            assert!(
                                ok,
                                "{name} kind {kind:#x} rows {rows} cols {cols} batch {batch}: {error:e}"
                            );
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "no Q4Z kernel can run on this CPU");
    }

    /// `cargo test --release -p fllm-engine cpu_q4z -- --ignored --nocapture`: weight bytes read
    /// per second at decode and weight multiply-accumulates per second at a 64-token prompt, per
    /// kernel and group size, on one thread (`FLLM_BENCH_THREADS` for more).
    #[test]
    #[ignore = "benchmark"]
    fn bench_decode_and_prompt_throughput() {
        let threads = std::env::var("FLLM_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut rng = Lcg(0xbe7c);
        let (rows, cols) = (2048, 2048);
        for (name, panel) in kernels() {
            for kind in [0x8000_0010, 0x8000_0011, 0x8000_0012] {
                let group = fllm_gguf_probe::q4z_group(kind).unwrap();
                let weights = random_weights(kind, rows, cols, &mut rng);
                let bytes = weights.len() as f64;
                for batch in [1usize, 64] {
                    let x = rng.vector(batch * cols, 1.0);
                    let run = || unsafe {
                        linear_with(Some(panel), &weights, kind, &x, rows, cols, None).unwrap()
                    };
                    pool.install(|| {
                        run();
                        let start = std::time::Instant::now();
                        let rounds = if batch == 1 { 40 } else { 4 };
                        for _ in 0..rounds {
                            std::hint::black_box(run());
                        }
                        let seconds = start.elapsed().as_secs_f64() / rounds as f64;
                        let macs = (rows * cols * batch) as f64;
                        println!(
                            "{name:12} group {group:3} batch {batch:2}: {:7.2} GB/s of weights, {:7.2} GMAC/s, {:6.2} ms",
                            bytes / seconds / 1e9,
                            macs / seconds / 1e9,
                            seconds * 1e3
                        );
                    });
                }
            }
        }
    }
    #[test]
    fn bias_and_input_rounding_agree_with_the_dequantized_matrix() {
        // Against the generic f32 path (full-precision activations): the 8-bit activations
        // contribute their usual error, far below 1e-2 of the sum of the absolute terms.
        let mut rng = Lcg(0x5107);
        for kind in [
            0x8000_0010,
            0x8000_0011,
            0x8000_0012,
            0x8000_0014,
            0x8000_0016,
        ] {
            let (rows, cols, batch) = (9, 256, 3);
            let weights = random_weights(kind, rows, cols, &mut rng);
            let bias = rng.vector(rows, 1.0);
            let x = rng.vector(batch * cols, 2.0);
            let Some(have) = linear(&weights, kind, &x, rows, cols, Some(&bias)) else {
                continue;
            };
            let dense = crate::compute::dequantize(&weights, kind, rows * cols).unwrap();
            let mut want = vec![0.0f64; batch * rows];
            let mut scale = vec![0.0f64; batch * rows];
            for t in 0..batch {
                for r in 0..rows {
                    for c in 0..cols {
                        let term = f64::from(dense[r * cols + c]) * f64::from(x[t * cols + c]);
                        want[t * rows + r] += term;
                        scale[t * rows + r] += term.abs();
                    }
                    want[t * rows + r] += f64::from(bias[r]);
                    scale[t * rows + r] += f64::from(bias[r]).abs();
                }
            }
            let (error, ok) = compare_with(&have, &want, &scale, 1e-2);
            assert!(ok, "kind {kind:#x}: {error:e}");
        }
    }
}
