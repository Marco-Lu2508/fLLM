//! F32, F16 and BF16 projections on 512-bit vectors, and BF16 with `vdpbf16ps`.
//!
//! `cpu_float` keeps weights packed and rounds the input to the weight type before the kernels
//! (so a BF16 projection multiplies bf16 weights by bf16-exact activations). Three kernels follow
//! its structure for 512-bit vectors:
//!
//! - prompts (4+ tokens): sixteen weight rows in one vector, `NR` tokens per pass over a decoded
//!   chunk of 256 columns. F32/F16/BF16 weights are decoded to f32 and transposed (`panel_fma`);
//!   BF16 weights stay bf16, transposed as pairs of consecutive columns, and `vdpbf16ps` adds two
//!   products per lane per instruction (`panel_bf16`; the activations are converted to bf16 once).
//! - decode (1-3 tokens): four rows by two tokens, 16 (or 32 bf16) columns per step with masked
//!   loads for the tail (`tile_fma`, `tile_bf16`).
//! - on a CPU with AMX-BF16 and a prompt of `amx_min_batch` tokens or more, BF16 projections go
//!   to the tile kernel (`amx_gemm::bf16_group`).
//!
//! `vdpbf16ps` multiplies two bf16 values exactly in f32 and adds them to the accumulator with one
//! rounding per pair, flushing denormal inputs and outputs to zero regardless of MXCSR; the
//! difference to the f32 FMA path is a last-bit effect.
use super::amx_gemm;
use crate::cpu_float::Projection;
use crate::cpu_schedule::Rows;
use std::arch::x86_64::*;

/// Rows per panel: one 512-bit vector.
const MR: usize = 16;
/// Columns (k values) decoded at a time.
const KC: usize = 256;
/// Tokens per pass over a decoded chunk (14 accumulators of 32 vector registers).
const NR: usize = 14;
/// Smallest batch that runs on sixteen-row panels.
const PANEL_MIN_BATCH: usize = 4;

/// Which kernels a BF16/F16/F32 projection uses on this CPU, or `None` for the AVX2 kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wide {
    /// 512-bit FMA on f32 (and f16/bf16 converted on load).
    Fma,
    /// BF16 weights through `vdpbf16ps`.
    Bf16Dot,
}

/// The kernel family for `kind` (0 F32, 1 F16, 30 BF16) under the active tiers and the CPU policy.
pub(crate) fn wide(kind: u32) -> Option<Wide> {
    use crate::cpu_kernels::{tier_enabled, CpuTier};
    let policy = super::ident::policy();
    if kind == 30 && policy.bf16_dot && tier_enabled(CpuTier::Avx512Bf16) {
        return Some(Wide::Bf16Dot);
    }
    if matches!(kind, 0 | 1 | 30) && policy.wide_float && tier_enabled(CpuTier::Avx512Bw) {
        return Some(Wide::Fma);
    }
    None
}

/// `FLLM_NO_FLOAT_PANEL` selects the four-row kernel for every batch (as in `cpu_float`).
fn panel_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("FLLM_NO_FLOAT_PANEL").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// Run all projections of one group. `x` is the input already rounded to the weight type,
/// `output[i]` holds `rows[i] * batch` floats.
pub(crate) fn run(
    projections: &[Projection<'_>],
    kind: u32,
    x: &[f32],
    cols: usize,
    batch: usize,
    kernels: Wide,
    output: &mut [Vec<f32>],
) {
    let rows: Vec<usize> = projections.iter().map(|p| p.rows).collect();
    if kind == 30
        && crate::cpu_avx512::amx_gemm::wanted(batch)
        && amx_gemm::bf16_group(projections, x, cols, batch, output)
    {
        return;
    }
    // bf16 activations, rows padded to an even length so a pair read never leaves its row.
    let cols_p = cols.next_multiple_of(2);
    let xb: Vec<u16> = if kernels == Wide::Bf16Dot {
        let mut xb = vec![0u16; batch * cols_p];
        for (dst, src) in xb.chunks_mut(cols_p).zip(x.chunks(cols)) {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = half::bf16::from_f32(v).to_bits();
            }
        }
        xb
    } else {
        Vec::new()
    };
    let add_bias = |out: &mut Rows<'_>, bias: Option<&[f32]>, first: usize| {
        if let Some(bias) = bias {
            for b in 0..batch {
                for (r, y) in out.row(b).iter_mut().enumerate() {
                    *y += bias[first + r];
                }
            }
        }
    };
    if batch >= PANEL_MIN_BATCH && !panel_disabled() {
        crate::cpu_schedule::row_panels(
            output,
            &rows,
            batch,
            MR,
            || (vec![0.0f32; KC * MR], vec![[0.0f32; MR]; batch]),
            |(panel, part), member, first, out| {
                let p = &projections[member];
                let valid = (p.rows - first).min(MR);
                // SAFETY: the tiers `wide` checked imply the target features; weights hold
                // `rows * cols` elements (asserted by `cpu_float`) and `x`/`xb` hold the batch.
                unsafe {
                    match (kernels, kind) {
                        (Wide::Bf16Dot, _) => panel_bf16(
                            p.weights,
                            &xb,
                            (first, valid),
                            cols,
                            cols_p,
                            panel,
                            part,
                            out,
                        ),
                        (Wide::Fma, 0) => {
                            panel_fma::<0>(p.weights, x, (first, valid), cols, panel, part, out)
                        }
                        (Wide::Fma, 1) => {
                            panel_fma::<1>(p.weights, x, (first, valid), cols, panel, part, out)
                        }
                        (Wide::Fma, _) => {
                            panel_fma::<30>(p.weights, x, (first, valid), cols, panel, part, out)
                        }
                    }
                }
                add_bias(out, p.bias, first);
            },
        );
        return;
    }
    crate::cpu_schedule::row_panels(
        output,
        &rows,
        batch,
        4,
        || (),
        |_, member, first, out| {
            let p = &projections[member];
            let valid = (p.rows - first).min(4);
            // SAFETY: as above.
            unsafe {
                match (kernels, kind) {
                    (Wide::Bf16Dot, _) => {
                        tile_bf16(p.weights, &xb, (first, valid), cols, cols_p, batch, out)
                    }
                    (Wide::Fma, 0) => tile_fma::<0>(p.weights, x, (first, valid), cols, batch, out),
                    (Wide::Fma, 1) => tile_fma::<1>(p.weights, x, (first, valid), cols, batch, out),
                    (Wide::Fma, _) => {
                        tile_fma::<30>(p.weights, x, (first, valid), cols, batch, out)
                    }
                }
            }
            add_bias(out, p.bias, first);
        },
    );
}

// ---------------------------------------------------------------------------------------------
// f32 FMA kernels
// ---------------------------------------------------------------------------------------------

/// 16 elements starting at element `index` as f32 (`mask` selects the valid ones, the rest are 0).
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,f16c")]
unsafe fn load_masked<const KIND: u32>(w: &[u8], index: usize, mask: __mmask16) -> __m512 {
    match KIND {
        0 => _mm512_maskz_loadu_ps(mask, w.as_ptr().add(index * 4).cast()),
        1 => _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
            mask,
            w.as_ptr().add(index * 2).cast(),
        )),
        _ => _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
            _mm256_maskz_loadu_epi16(mask, w.as_ptr().add(index * 2).cast()),
        ))),
    }
}

/// Mask of the first `n.min(16)` lanes.
#[inline]
fn lanes16(n: usize) -> __mmask16 {
    if n >= 16 {
        0xffff
    } else {
        ((1u32 << n) - 1) as u16
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn transpose8(r: [__m256; 8]) -> [__m256; 8] {
    let t0 = _mm256_unpacklo_ps(r[0], r[1]);
    let t1 = _mm256_unpackhi_ps(r[0], r[1]);
    let t2 = _mm256_unpacklo_ps(r[2], r[3]);
    let t3 = _mm256_unpackhi_ps(r[2], r[3]);
    let t4 = _mm256_unpacklo_ps(r[4], r[5]);
    let t5 = _mm256_unpackhi_ps(r[4], r[5]);
    let t6 = _mm256_unpacklo_ps(r[6], r[7]);
    let t7 = _mm256_unpackhi_ps(r[6], r[7]);
    let cast = _mm256_castps_pd;
    let back = _mm256_castpd_ps;
    let u0 = back(_mm256_unpacklo_pd(cast(t0), cast(t2)));
    let u1 = back(_mm256_unpackhi_pd(cast(t0), cast(t2)));
    let u2 = back(_mm256_unpacklo_pd(cast(t1), cast(t3)));
    let u3 = back(_mm256_unpackhi_pd(cast(t1), cast(t3)));
    let u4 = back(_mm256_unpacklo_pd(cast(t4), cast(t6)));
    let u5 = back(_mm256_unpackhi_pd(cast(t4), cast(t6)));
    let u6 = back(_mm256_unpacklo_pd(cast(t5), cast(t7)));
    let u7 = back(_mm256_unpackhi_pd(cast(t5), cast(t7)));
    [
        _mm256_permute2f128_ps::<0x20>(u0, u4),
        _mm256_permute2f128_ps::<0x20>(u1, u5),
        _mm256_permute2f128_ps::<0x20>(u2, u6),
        _mm256_permute2f128_ps::<0x20>(u3, u7),
        _mm256_permute2f128_ps::<0x31>(u0, u4),
        _mm256_permute2f128_ps::<0x31>(u1, u5),
        _mm256_permute2f128_ps::<0x31>(u2, u6),
        _mm256_permute2f128_ps::<0x31>(u3, u7),
    ]
}

fn element<const KIND: u32>(w: &[u8], index: usize) -> f32 {
    if KIND == 0 {
        f32::from_le_bytes(w[index * 4..index * 4 + 4].try_into().unwrap())
    } else {
        let v = u16::from_le_bytes(w[index * 2..index * 2 + 2].try_into().unwrap());
        if KIND == 1 {
            half::f16::from_bits(v).to_f32()
        } else {
            f32::from_bits(u32::from(v) << 16)
        }
    }
}

/// Eight elements at `index` as f32 for the 8x8 transposes.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn load8<const KIND: u32>(w: &[u8], index: usize) -> __m256 {
    if KIND == 0 {
        _mm256_loadu_ps(w.as_ptr().add(index * 4).cast())
    } else {
        let v = _mm_loadu_si128(w.as_ptr().add(index * 2).cast());
        if KIND == 1 {
            _mm256_cvtph_ps(v)
        } else {
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16))
        }
    }
}

/// Decode columns `k0..k0 + kc` of up to sixteen rows from row `first` into `panel`
/// (`panel[16 * k + row]`); rows past `valid` become zero. Same layout as the AVX2 panel.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn decode<const KIND: u32>(
    w: &[u8],
    (first, valid): (usize, usize),
    cols: usize,
    (k0, kc): (usize, usize),
    panel: &mut [f32],
) {
    let full = kc / 8 * 8;
    for half in 0..2 {
        for k in (0..full).step_by(8) {
            let rows: [__m256; 8] = std::array::from_fn(|r| {
                let row = half * 8 + r;
                if row < valid {
                    load8::<KIND>(w, (first + row) * cols + k0 + k)
                } else {
                    _mm256_setzero_ps()
                }
            });
            let columns = transpose8(rows);
            for (j, column) in columns.iter().enumerate() {
                _mm256_storeu_ps(panel.as_mut_ptr().add((k + j) * MR + half * 8), *column);
            }
        }
    }
    for k in full..kc {
        for row in 0..MR {
            panel[k * MR + row] = if row < valid {
                element::<KIND>(w, (first + row) * cols + k0 + k)
            } else {
                0.0
            };
        }
    }
}

/// `T` tokens from `token` against a decoded chunk of `kc` columns starting at `k0`.
#[inline]
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f")]
unsafe fn pass_fma<const T: usize>(
    panel: &[f32],
    x: &[f32],
    cols: usize,
    (token, k0, kc): (usize, usize, usize),
    part: &mut [[f32; MR]],
    fresh: bool,
) {
    let mut acc = [_mm512_setzero_ps(); T];
    if !fresh {
        for (t, a) in acc.iter_mut().enumerate() {
            *a = _mm512_loadu_ps(part[token + t].as_ptr());
        }
    }
    let xs: [*const f32; T] = std::array::from_fn(|t| x.as_ptr().add((token + t) * cols + k0));
    for k in 0..kc {
        let w = _mm512_loadu_ps(panel.as_ptr().add(k * MR));
        for t in 0..T {
            acc[t] = _mm512_fmadd_ps(w, _mm512_set1_ps(*xs[t].add(k)), acc[t]);
        }
    }
    for (t, a) in acc.iter().enumerate() {
        _mm512_storeu_ps(part[token + t].as_mut_ptr(), *a);
    }
}

/// One panel of up to sixteen rows against every token, f32 FMA.
///
/// # Safety
/// The CPU must support AVX-512 F/BW/VL, AVX2, FMA, F16C; `w` holds `first + valid` rows of
/// `cols` elements, `x` holds `part.len() * cols` values, `panel` `KC * MR` floats.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl")]
unsafe fn panel_fma<const KIND: u32>(
    w: &[u8],
    x: &[f32],
    rows: (usize, usize),
    cols: usize,
    panel: &mut [f32],
    part: &mut [[f32; MR]],
    out: &mut Rows<'_>,
) {
    let batch = part.len();
    for k0 in (0..cols).step_by(KC) {
        let kc = (cols - k0).min(KC);
        decode::<KIND>(w, rows, cols, (k0, kc), panel);
        let mut token = 0;
        while token < batch {
            let n = (batch - token).min(NR);
            let at = (token, k0, kc);
            let fresh = k0 == 0;
            macro_rules! pass {
                ($($t:literal),*) => {
                    match n {
                        $($t => pass_fma::<$t>(panel, x, cols, at, part, fresh),)*
                        _ => unreachable!(),
                    }
                };
            }
            pass!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
            token += n;
        }
    }
    for (token, lanes) in part.iter().enumerate() {
        for (y, v) in out.row(token).iter_mut().zip(lanes) {
            *y = *v;
        }
    }
}

/// Four rows by two tokens, 16 columns per step, masked tail.
///
/// # Safety
/// As `panel_fma`; `w` holds `first + valid` rows, `x` holds `batch * cols` values.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl")]
unsafe fn tile_fma<const KIND: u32>(
    w: &[u8],
    x: &[f32],
    (first, valid): (usize, usize),
    cols: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    for b in (0..batch).step_by(2) {
        let two = b + 1 < batch;
        let mut acc = [[_mm512_setzero_ps(); 4]; 2];
        for k in (0..cols).step_by(16) {
            let mask = lanes16(cols - k);
            let x0 = _mm512_maskz_loadu_ps(mask, x.as_ptr().add(b * cols + k));
            let x1 = if two {
                _mm512_maskz_loadu_ps(mask, x.as_ptr().add((b + 1) * cols + k))
            } else {
                _mm512_setzero_ps()
            };
            for r in 0..valid {
                let v = load_masked::<KIND>(w, (first + r) * cols + k, mask);
                acc[0][r] = _mm512_fmadd_ps(v, x0, acc[0][r]);
                acc[1][r] = _mm512_fmadd_ps(v, x1, acc[1][r]);
            }
        }
        for (j, accumulators) in acc.iter().enumerate().take((batch - b).min(2)) {
            for (r, &v) in accumulators.iter().enumerate().take(valid) {
                out.row(b + j)[r] = _mm512_reduce_add_ps(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// bf16 dot kernels
// ---------------------------------------------------------------------------------------------

/// A 512-bit integer vector as 32 bf16 values for `vdpbf16ps`.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn as_bf16(v: __m512i) -> __m512bh {
    std::mem::transmute::<__m512i, __m512bh>(v)
}

/// Decode columns `k0..k0 + kc` of up to sixteen BF16 rows into column pairs: `panel[16 * p + row]`
/// is the dword of columns `k0 + 2p` (low half) and `k0 + 2p + 1` (high half) of `row`; a missing
/// second column of an odd tail and rows past `valid` are zero. `panel` is viewed as `u32`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2")]
unsafe fn decode_pairs(
    w: &[u8],
    (first, valid): (usize, usize),
    cols: usize,
    (k0, kc): (usize, usize),
    panel: &mut [u32],
) {
    let pairs = kc.div_ceil(2);
    // Eight column pairs (16 columns) per row load; the dwords move as bit patterns.
    let full = kc / 16 * 8;
    for half in 0..2 {
        for p in (0..full).step_by(8) {
            let rows: [__m256; 8] = std::array::from_fn(|r| {
                let row = half * 8 + r;
                if row < valid {
                    _mm256_loadu_ps(
                        w.as_ptr()
                            .add(((first + row) * cols + k0 + 2 * p) * 2)
                            .cast(),
                    )
                } else {
                    _mm256_setzero_ps()
                }
            });
            let columns = transpose8(rows);
            for (j, column) in columns.iter().enumerate() {
                _mm256_storeu_ps(
                    panel.as_mut_ptr().add((p + j) * MR + half * 8).cast(),
                    *column,
                );
            }
        }
    }
    let bf = |row: usize, k: usize| -> u32 {
        if row < valid && k < kc {
            let at = ((first + row) * cols + k0 + k) * 2;
            u32::from(u16::from_le_bytes([w[at], w[at + 1]]))
        } else {
            0
        }
    };
    for p in full..pairs {
        for row in 0..MR {
            panel[p * MR + row] = bf(row, 2 * p) | (bf(row, 2 * p + 1) << 16);
        }
    }
}

/// `T` tokens against a decoded pair chunk with `vdpbf16ps`. `xb` holds bf16 activations with
/// rows of `cols_p` (even) elements.
#[inline]
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn pass_bf16<const T: usize>(
    panel: &[u32],
    xb: &[u16],
    cols_p: usize,
    (token, k0, kc): (usize, usize, usize),
    part: &mut [[f32; MR]],
    fresh: bool,
) {
    let mut acc = [_mm512_setzero_ps(); T];
    if !fresh {
        for (t, a) in acc.iter_mut().enumerate() {
            *a = _mm512_loadu_ps(part[token + t].as_ptr());
        }
    }
    let xs: [*const u8; T] =
        std::array::from_fn(|t| xb.as_ptr().add((token + t) * cols_p + k0).cast::<u8>());
    for p in 0..kc.div_ceil(2) {
        let w = as_bf16(_mm512_loadu_si512(panel.as_ptr().add(p * MR).cast()));
        for t in 0..T {
            let pair = std::ptr::read_unaligned(xs[t].add(4 * p).cast::<i32>());
            acc[t] = _mm512_dpbf16_ps(acc[t], w, as_bf16(_mm512_set1_epi32(pair)));
        }
    }
    for (t, a) in acc.iter().enumerate() {
        _mm512_storeu_ps(part[token + t].as_mut_ptr(), *a);
    }
}

/// One panel of up to sixteen BF16 rows against every token with `vdpbf16ps`.
///
/// # Safety
/// The CPU must support AVX-512 F/BW/VL and AVX512-BF16, AVX2, FMA, F16C; `w` holds
/// `first + valid` rows of `cols` bf16 elements, `xb` holds `part.len() * cols_p` bf16 values,
/// `panel` at least `KC * MR` floats.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512bf16")]
unsafe fn panel_bf16(
    w: &[u8],
    xb: &[u16],
    rows: (usize, usize),
    cols: usize,
    cols_p: usize,
    panel: &mut [f32],
    part: &mut [[f32; MR]],
    out: &mut Rows<'_>,
) {
    let batch = part.len();
    let pairs = std::slice::from_raw_parts_mut(panel.as_mut_ptr().cast::<u32>(), panel.len());
    for k0 in (0..cols).step_by(KC) {
        let kc = (cols - k0).min(KC);
        decode_pairs(w, rows, cols, (k0, kc), pairs);
        let mut token = 0;
        while token < batch {
            let n = (batch - token).min(NR);
            let at = (token, k0, kc);
            let fresh = k0 == 0;
            macro_rules! pass {
                ($($t:literal),*) => {
                    match n {
                        $($t => pass_bf16::<$t>(pairs, xb, cols_p, at, part, fresh),)*
                        _ => unreachable!(),
                    }
                };
            }
            pass!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
            token += n;
        }
    }
    for (token, lanes) in part.iter().enumerate() {
        for (y, v) in out.row(token).iter_mut().zip(lanes) {
            *y = *v;
        }
    }
}

/// Four BF16 rows by two tokens, 32 columns per step, masked tail.
///
/// # Safety
/// As `panel_bf16`.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512bf16")]
unsafe fn tile_bf16(
    w: &[u8],
    xb: &[u16],
    (first, valid): (usize, usize),
    cols: usize,
    cols_p: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    for b in (0..batch).step_by(2) {
        let two = b + 1 < batch;
        let mut acc = [[_mm512_setzero_ps(); 4]; 2];
        for k in (0..cols).step_by(32) {
            let remaining = cols - k;
            let mask: __mmask32 = if remaining >= 32 {
                u32::MAX
            } else {
                (1u32 << remaining) - 1
            };
            let x0 = as_bf16(_mm512_maskz_loadu_epi16(
                mask,
                xb.as_ptr().add(b * cols_p + k).cast(),
            ));
            let x1 = if two {
                as_bf16(_mm512_maskz_loadu_epi16(
                    mask,
                    xb.as_ptr().add((b + 1) * cols_p + k).cast(),
                ))
            } else {
                as_bf16(_mm512_setzero_si512())
            };
            for r in 0..valid {
                let v = as_bf16(_mm512_maskz_loadu_epi16(
                    mask,
                    w.as_ptr().add(((first + r) * cols + k) * 2).cast(),
                ));
                acc[0][r] = _mm512_dpbf16_ps(acc[0][r], v, x0);
                acc[1][r] = _mm512_dpbf16_ps(acc[1][r], v, x1);
            }
        }
        for (j, accumulators) in acc.iter().enumerate().take((batch - b).min(2)) {
            for (r, &v) in accumulators.iter().enumerate().take(valid) {
                out.row(b + j)[r] = _mm512_reduce_add_ps(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, Lcg};

    fn has_512() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    /// Weights as bytes of `kind`, plus the exactly rounded weight and input values.
    fn operands(
        kind: u32,
        rows: usize,
        cols: usize,
        batch: usize,
        rng: &mut Lcg,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let round = |v: f32| match kind {
            0 => v,
            1 => half::f16::from_f32(v).to_f32(),
            _ => half::bf16::from_f32(v).to_f32(),
        };
        let w: Vec<f32> = rng
            .vector(rows * cols, 1.0)
            .into_iter()
            .map(round)
            .collect();
        let bytes = w
            .iter()
            .flat_map(|&v| match kind {
                0 => v.to_le_bytes().to_vec(),
                1 => half::f16::from_f32(v).to_le_bytes().to_vec(),
                _ => half::bf16::from_f32(v).to_le_bytes().to_vec(),
            })
            .collect();
        let x: Vec<f32> = rng
            .vector(batch * cols, 3.0)
            .into_iter()
            .map(round)
            .collect();
        (bytes, w, x)
    }

    fn reference(
        w: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        batch: usize,
        bias: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut want = vec![0.0; batch * rows];
        let mut scale = vec![0.0; batch * rows];
        for t in 0..batch {
            for r in 0..rows {
                let mut sum = f64::from(bias[r]);
                let mut mag = f64::from(bias[r]).abs();
                for k in 0..cols {
                    let term = f64::from(w[r * cols + k]) * f64::from(x[t * cols + k]);
                    sum += term;
                    mag += term.abs();
                }
                want[t * rows + r] = sum;
                scale[t * rows + r] = mag;
            }
        }
        (want, scale)
    }

    fn check(kernels: Wide, kind: u32, label: &str) {
        let mut rng = Lcg(0xf10a7 ^ u64::from(kind));
        // Odd column counts, chunk boundaries (256), row tails, batches around the panel limit and
        // the 14-token pass.
        for cols in [1usize, 7, 33, 256, 257, 600] {
            for rows in [1usize, 5, 16, 21] {
                for batch in [1usize, 2, 3, 4, 15, 16, 33] {
                    let (bytes, w, x) = operands(kind, rows, cols, batch, &mut rng);
                    let bias = rng.vector(rows, 1.0);
                    let projections = [Projection {
                        weights: &bytes,
                        rows,
                        bias: Some(&bias),
                    }];
                    let mut output = vec![vec![0.0f32; rows * batch]];
                    run(&projections, kind, &x, cols, batch, kernels, &mut output);
                    let (want, scale) = reference(&w, &x, rows, cols, batch, &bias);
                    let (error, ok) = compare(&output[0], &want, &scale);
                    assert!(
                        ok,
                        "{label} kind {kind} cols {cols} rows {rows} batch {batch}: {error:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn avx512_fma_float_kernels_match_the_reference() {
        if !has_512() {
            return;
        }
        for kind in [0u32, 1, 30] {
            check(Wide::Fma, kind, "fma512");
        }
    }

    #[test]
    fn avx512_bf16_dot_kernels_match_the_reference() {
        if !(has_512() && std::arch::is_x86_feature_detected!("avx512bf16")) {
            return;
        }
        check(Wide::Bf16Dot, 30, "bf16-dot");
    }

    /// The decoded f32 panel (`panel[16 * k + row]`) and the bf16 pair panel (`panel[16 * p + row]`
    /// = columns `2p`, `2p + 1`) hold the weights at the documented places, with zero rows and a
    /// zero second column of an odd tail. These decoders need only AVX2, so they run anywhere.
    #[test]
    fn decoded_panels_have_the_documented_layouts() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return;
        }
        let mut rng = Lcg(0xdec0de);
        for (rows, cols, first, valid) in [
            (21usize, 300usize, 3usize, 16usize),
            (21, 77, 5, 11),
            (16, 256, 0, 16),
            (3, 33, 0, 3),
        ] {
            for kind in [0u32, 1, 30] {
                let (bytes, w, _) = operands(kind, rows, cols, 1, &mut rng);
                for k0 in (0..cols).step_by(KC) {
                    let kc = (cols - k0).min(KC);
                    let mut panel = vec![f32::NAN; KC * MR];
                    // SAFETY: AVX2/FMA/F16C checked above; `bytes` holds `rows` rows.
                    unsafe {
                        match kind {
                            0 => decode::<0>(&bytes, (first, valid), cols, (k0, kc), &mut panel),
                            1 => decode::<1>(&bytes, (first, valid), cols, (k0, kc), &mut panel),
                            _ => decode::<30>(&bytes, (first, valid), cols, (k0, kc), &mut panel),
                        }
                    }
                    for k in 0..kc {
                        for r in 0..MR {
                            let want = if r < valid {
                                w[(first + r) * cols + k0 + k]
                            } else {
                                0.0
                            };
                            assert_eq!(panel[k * MR + r], want, "kind {kind} k {k} row {r}");
                        }
                    }
                }
            }
            let (bytes, w, _) = operands(30, rows, cols, 1, &mut rng);
            let bf = |r: usize, k: usize| -> u32 {
                let value = w[(first + r) * cols + k];
                u32::from(half::bf16::from_f32(value).to_bits())
            };
            for k0 in (0..cols).step_by(KC) {
                let kc = (cols - k0).min(KC);
                let mut pairs = vec![0xdead_beefu32; KC * MR];
                // SAFETY: AVX2 checked above.
                unsafe { decode_pairs(&bytes, (first, valid), cols, (k0, kc), &mut pairs) };
                for p in 0..kc.div_ceil(2) {
                    for r in 0..MR {
                        let want = if r < valid {
                            let second = if 2 * p + 1 < kc {
                                bf(r, k0 + 2 * p + 1)
                            } else {
                                0
                            };
                            bf(r, k0 + 2 * p) | (second << 16)
                        } else {
                            0
                        };
                        assert_eq!(pairs[p * MR + r], want, "pair {p} row {r} k0 {k0}");
                    }
                }
            }
        }
    }

    #[test]
    fn masks_cover_exactly_the_requested_lanes() {
        assert_eq!(lanes16(0), 0);
        assert_eq!(lanes16(1), 1);
        assert_eq!(lanes16(15), 0x7fff);
        assert_eq!(lanes16(16), 0xffff);
        assert_eq!(lanes16(1000), 0xffff);
    }
}
