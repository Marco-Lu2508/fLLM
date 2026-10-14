//! Four-row/two-token AVX2 FMA for F32, F16 and BF16 projections.
//! Half weights stay packed; input rounding matches native vec_dot types.
use half::{bf16, f16};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::borrow::Cow;

pub(crate) fn encode_f16(input: &[f32], output: &mut [u16]) -> bool {
    assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::isa() != "scalar" {
        unsafe {
            x86::encode_f16(input, output);
        }
        return true;
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::cpu_kernels::neon() {
        crate::cpu_arm::float::encode_f16(input, output);
        return true;
    }
    false
}

// The x86 kernels and the tests round through here; the AArch64 kernels round in `cpu_arm::float`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn rounded_input(input: &[f32], kind: u32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let round = |out: &mut [f32], src: &[f32]| {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu_kernels::isa() != "scalar" {
            unsafe {
                if kind == 1 {
                    x86::round_input::<1>(src, out);
                } else {
                    x86::round_input::<30>(src, out);
                }
            }
            return;
        }
        for (out, &x) in out.iter_mut().zip(src) {
            *out = if kind == 1 {
                f16::from_f32(x).to_f32()
            } else {
                bf16::from_f32(x).to_f32()
            };
        }
    };
    // One decode vector is cheaper to round in place than to wake the pool.
    if input.len() <= 16384 {
        round(&mut output, input);
    } else {
        output
            .par_chunks_mut(512)
            .zip(input.par_chunks(512))
            .for_each(|(out, src)| round(out, src));
    }
    output
}

pub fn linear(
    w: &[u8],
    kind: u32,
    input: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    linear_group(
        &[Projection {
            weights: w,
            rows,
            bias,
        }],
        kind,
        input,
        cols,
    )
    .map(|mut output| output.pop().unwrap())
}

pub struct Projection<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub bias: Option<&'a [f32]>,
}

/// Smallest batch that runs on sixteen-row panels.
#[cfg(target_arch = "x86_64")]
const PANEL_MIN_BATCH: usize = 4;

/// `FLLM_NO_FLOAT_PANEL` selects the four-row kernel for every batch (for comparisons).
#[cfg(target_arch = "x86_64")]
fn panel_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("FLLM_NO_FLOAT_PANEL").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// Round a shared input once and distribute all sibling rows in one queue.
pub fn linear_group(
    projections: &[Projection<'_>],
    kind: u32,
    input: &[f32],
    cols: usize,
) -> Option<Vec<Vec<f32>>> {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::isa() != "scalar" && matches!(kind, 0 | 1 | 30) {
        assert!(cols > 0 && input.len().is_multiple_of(cols));
        let width = if kind == 0 { 4 } else { 2 };
        for p in projections {
            assert!(p.rows > 0);
            assert_eq!(p.weights.len(), p.rows * cols * width);
            assert!(p.bias.is_none_or(|b| b.len() == p.rows));
        }
        let batch = input.len() / cols;
        if batch == 0 || projections.is_empty() {
            return Some(projections.iter().map(|_| Vec::new()).collect());
        }
        let x: Cow<'_, [f32]> = if kind == 0 {
            Cow::Borrowed(input)
        } else {
            Cow::Owned(rounded_input(input, kind))
        };
        let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
        let mut output = rows
            .iter()
            .map(|&r| vec![0.0; batch * r])
            .collect::<Vec<_>>();
        // 512-bit and `vdpbf16ps` kernels, and AMX-BF16 for prompts (the `avx512` feature).
        #[cfg(feature = "avx512")]
        if let Some(kernels) = crate::cpu_avx512::float::wide(kind) {
            crate::cpu_avx512::float::run(projections, kind, &x, cols, batch, kernels, &mut output);
            return Some(output);
        }
        if batch >= PANEL_MIN_BATCH && !panel_disabled() {
            // Prompts: sixteen rows in the vector lanes, six tokens per pass over a decoded chunk.
            crate::cpu_schedule::row_panels(
                &mut output,
                &rows,
                batch,
                x86::MR,
                || (vec![0.0f32; x86::KC * x86::MR], vec![[0.0f32; x86::MR]; batch]),
                |(panel, part), member, first, out| {
                    let p = &projections[member];
                    let valid = (p.rows - first).min(x86::MR);
                    // SAFETY: `isa()` confirmed AVX2/FMA/F16C; weights hold `rows * cols`
                    // elements (asserted above) and `x` holds `batch * cols`.
                    unsafe {
                        match kind {
                            0 => x86::panel::<0>(p.weights, &x, (first, valid), cols, panel, part, out),
                            1 => x86::panel::<1>(p.weights, &x, (first, valid), cols, panel, part, out),
                            _ => x86::panel::<30>(p.weights, &x, (first, valid), cols, panel, part, out),
                        }
                    }
                    if let Some(bias) = p.bias {
                        for b in 0..batch {
                            for (r, y) in out.row(b).iter_mut().enumerate() {
                                *y += bias[first + r];
                            }
                        }
                    }
                },
            );
            return Some(output);
        }
        crate::cpu_schedule::row_panels(
            &mut output,
            &rows,
            batch,
            4,
            || (),
            |_, member, first, out| {
                let p = &projections[member];
                let valid = (p.rows - first).min(4);
                unsafe {
                    macro_rules! run {
                        ($kind:literal) => {{
                            macro_rules! rows {
                                ($rows:literal) => {
                                    if batch == 1 {
                                        x86::tile::<$kind, true, $rows>(
                                            p.weights, &x, first, cols, batch, out,
                                        );
                                    } else {
                                        x86::tile::<$kind, false, $rows>(
                                            p.weights, &x, first, cols, batch, out,
                                        );
                                    }
                                };
                            }
                            match valid {
                                1 => rows!(1),
                                2 => rows!(2),
                                3 => rows!(3),
                                4 => rows!(4),
                                _ => unreachable!(),
                            }
                        }};
                    }
                    match kind {
                        0 => run!(0),
                        1 => run!(1),
                        30 => run!(30),
                        _ => unreachable!(),
                    }
                }
                for b in 0..batch {
                    for (r, y) in out.row(b).iter_mut().enumerate() {
                        *y += p.bias.map_or(0.0, |bias| bias[first + r]);
                    }
                }
            },
        );
        return Some(output);
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::cpu_kernels::neon() && matches!(kind, 0 | 1 | 30) {
        return Some(crate::cpu_arm::float::linear_group(
            projections,
            kind,
            input,
            cols,
        ));
    }
    let _ = (projections, kind, input, cols);
    None
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    #[test]
    fn half_rounding_and_cache_encoding_preserve_ties_and_tails() {
        let mut input = (0..1031)
            .map(|i| (i as f32 - 515.0) / 17.0)
            .collect::<Vec<_>>();
        input[..8].copy_from_slice(&[
            0.0,
            -0.0,
            1.0 + 1.0 / 2048.0,
            -1.0 - 1.0 / 2048.0,
            65504.0,
            1e-8,
            1.0 / 16384.0,
            1.0 + 1.0 / 256.0,
        ]);
        for kind in [1, 30] {
            let actual = rounded_input(&input, kind);
            let expected = input
                .iter()
                .map(|&v| {
                    if kind == 1 {
                        f16::from_f32(v).to_f32()
                    } else {
                        bf16::from_f32(v).to_f32()
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let mut encoded = vec![0; input.len()];
        if encode_f16(&input, &mut encoded) {
            assert_eq!(
                encoded,
                input
                    .iter()
                    .map(|&v| f16::from_f32(v).to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn float_group_shares_rounding_across_unequal_bias_and_row_tails() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let cols = 13;
        let rows = [1, 3, 7];
        for kind in [0, 1, 30] {
            let weights = rows
                .iter()
                .map(|&r| {
                    (0..r * cols)
                        .flat_map(|i| {
                            let x = (i % 19) as f32 / 11.0 - 0.75;
                            match kind {
                                0 => x.to_le_bytes().to_vec(),
                                1 => f16::from_f32(x).to_le_bytes().to_vec(),
                                _ => bf16::from_f32(x).to_le_bytes().to_vec(),
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let bias = [0.125, -0.25, 0.375];
            let members = rows
                .iter()
                .enumerate()
                .map(|(i, &rows)| Projection {
                    weights: &weights[i],
                    rows,
                    bias: if i == 1 { Some(&bias) } else { None },
                })
                .collect::<Vec<_>>();
            for batch in [0, 1, 3] {
                let input = (0..batch * cols)
                    .map(|i| (i % 11) as f32 / 7.0 - 0.5)
                    .collect::<Vec<_>>();
                let grouped = linear_group(&members, kind, &input, cols).unwrap();
                for (member, result) in members.iter().zip(grouped) {
                    assert_eq!(
                        result,
                        linear(member.weights, kind, &input, member.rows, cols, member.bias)
                            .unwrap()
                    );
                }
            }
        }
    }
    #[test]
    fn float_projection_handles_column_row_and_token_tails_with_bias() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        for batch in [1, 3] {
            for rows in [1, 2, 3, 5] {
                let cols = 13;
                let weights = (0..rows * cols)
                    .map(|i| (i % 17) as f32 / 9.0 - 1.0)
                    .collect::<Vec<_>>();
                let input = (0..batch * cols)
                    .map(|i| (i % 11) as f32 / 7.0 - 0.5)
                    .collect::<Vec<_>>();
                for kind in [0, 1, 30] {
                    let w = match kind {
                        0 => weights
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                        1 => weights
                            .iter()
                            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                            .collect(),
                        _ => weights
                            .iter()
                            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                            .collect(),
                    };
                    let bias = (0..rows)
                        .map(|r| (r + 1) as f32 * if r % 2 == 0 { 0.1 } else { -0.1 })
                        .collect::<Vec<_>>();
                    let output = linear(&w, kind, &input, rows, cols, Some(&bias)).unwrap();
                    let round = |v: f32| match kind {
                        1 => f16::from_f32(v).to_f32(),
                        30 => bf16::from_f32(v).to_f32(),
                        _ => v,
                    };
                    for b in 0..batch {
                        for r in 0..rows {
                            let expected = (0..cols)
                                .map(|k| round(weights[r * cols + k]) * round(input[b * cols + k]))
                                .sum::<f32>()
                                + bias[r];
                            assert!((output[b * rows + r] - expected).abs() < 1e-5);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sixteen_row_panels_match_the_scalar_reference() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let round = |kind: u32, v: f32| match kind {
            1 => f16::from_f32(v).to_f32(),
            30 => bf16::from_f32(v).to_f32(),
            _ => v,
        };
        for kind in [0u32, 1, 30] {
            for (rows, cols) in [(1usize, 13usize), (17, 40), (40, 300), (33, 520)] {
                let value = |i: usize| ((i * 7919) % 211) as f32 / 105.0 - 1.0;
                let weights: Vec<f32> = (0..rows * cols).map(|i| round(kind, value(i))).collect();
                let bytes: Vec<u8> = weights
                    .iter()
                    .flat_map(|&v| match kind {
                        0 => v.to_le_bytes().to_vec(),
                        1 => f16::from_f32(v).to_le_bytes().to_vec(),
                        _ => bf16::from_f32(v).to_le_bytes().to_vec(),
                    })
                    .collect();
                let bias: Vec<f32> = (0..rows).map(|r| r as f32 / 10.0 - 1.0).collect();
                for batch in [4usize, 5, 6, 7, 13, 16] {
                    let input: Vec<f32> =
                        (0..batch * cols).map(|i| ((i * 31) % 97) as f32 / 48.0 - 1.0).collect();
                    let got = linear(&bytes, kind, &input, rows, cols, Some(&bias)).unwrap();
                    for b in 0..batch {
                        for r in 0..rows {
                            let want: f64 = (0..cols)
                                .map(|k| {
                                    f64::from(weights[r * cols + k])
                                        * f64::from(round(kind, input[b * cols + k]))
                                })
                                .sum::<f64>()
                                + f64::from(bias[r]);
                            let have = f64::from(got[b * rows + r]);
                            assert!(
                                (have - want).abs() < 1e-4 * (1.0 + want.abs()),
                                "kind {kind} rows {rows} cols {cols} batch {batch} cell ({b},{r}): {have} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn encode_f16(input: &[f32], output: &mut [u16]) {
        let end = input.len() / 8 * 8;
        for j in (0..end).step_by(8) {
            let packed = _mm256_cvtps_ph(_mm256_loadu_ps(input.as_ptr().add(j)), 0);
            _mm_storeu_si128(output.as_mut_ptr().add(j).cast(), packed);
        }
        for (dst, &x) in output[end..].iter_mut().zip(&input[end..]) {
            *dst = crate::cpu_kernels::x86::float_to_half(x);
        }
    }
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn round_input<const KIND: u32>(input: &[f32], output: &mut [f32]) {
        let end = input.len() / 8 * 8;
        for j in (0..end).step_by(8) {
            let v = _mm256_loadu_ps(input.as_ptr().add(j));
            let rounded = if KIND == 1 {
                _mm256_cvtph_ps(_mm256_cvtps_ph(v, 0))
            } else {
                let bits = _mm256_castps_si256(v);
                let lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
                let rounded =
                    _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb));
                let nan = _mm256_cmpgt_epi32(
                    _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff)),
                    _mm256_set1_epi32(0x7f800000),
                );
                let quiet = _mm256_or_si256(bits, _mm256_set1_epi32(0x00400000));
                _mm256_castsi256_ps(_mm256_and_si256(
                    _mm256_blendv_epi8(rounded, quiet, nan),
                    _mm256_set1_epi32(-65536),
                ))
            };
            _mm256_storeu_ps(output.as_mut_ptr().add(j), rounded);
        }
        for (dst, &x) in output[end..].iter_mut().zip(&input[end..]) {
            *dst = if KIND == 1 {
                half::f16::from_f32(x).to_f32()
            } else {
                half::bf16::from_f32(x).to_f32()
            };
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn load<const KIND: u32>(w: &[u8], index: usize) -> __m256 {
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
    /// Rows per panel: two vectors of eight rows.
    pub(super) const MR: usize = 16;
    /// Columns (k values) decoded at a time: `KC * MR` floats, 16 KB.
    pub(super) const KC: usize = 256;
    /// Tokens per pass over a decoded chunk.
    const NR: usize = 6;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose8(r: [__m256; 8]) -> [__m256; 8] {
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

    #[inline]
    fn element<const KIND: u32>(w: &[u8], index: usize) -> f32 {
        if KIND == 0 {
            f32::from_le_bytes(w[index * 4..index * 4 + 4].try_into().unwrap())
        } else {
            let v = u16::from_le_bytes(w[index * 2..index * 2 + 2].try_into().unwrap());
            if KIND == 1 {
                half::f16::from_bits(v).to_f32()
            } else {
                f32::from_bits((v as u32) << 16)
            }
        }
    }

    /// Decode columns `k0..k0 + kc` of up to sixteen rows from row `first` into `panel`
    /// (`panel[16 * k + row]`); rows past `valid` become zero.
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
                        load::<KIND>(w, (first + row) * cols + k0 + k)
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
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pass<const T: usize>(
        panel: &[f32],
        x: &[f32],
        cols: usize,
        (token, k0, kc): (usize, usize, usize),
        part: &mut [[f32; MR]],
        fresh: bool,
    ) {
        let mut acc = [[_mm256_setzero_ps(); 2]; T];
        if !fresh {
            for (t, a) in acc.iter_mut().enumerate() {
                a[0] = _mm256_loadu_ps(part[token + t].as_ptr());
                a[1] = _mm256_loadu_ps(part[token + t].as_ptr().add(8));
            }
        }
        let xs: [*const f32; T] =
            std::array::from_fn(|t| x.as_ptr().add((token + t) * cols + k0));
        for k in 0..kc {
            let w0 = _mm256_loadu_ps(panel.as_ptr().add(k * MR));
            let w1 = _mm256_loadu_ps(panel.as_ptr().add(k * MR + 8));
            for t in 0..T {
                let xb = _mm256_broadcast_ss(&*xs[t].add(k));
                acc[t][0] = _mm256_fmadd_ps(w0, xb, acc[t][0]);
                acc[t][1] = _mm256_fmadd_ps(w1, xb, acc[t][1]);
            }
        }
        for (t, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(part[token + t].as_mut_ptr(), a[0]);
            _mm256_storeu_ps(part[token + t].as_mut_ptr().add(8), a[1]);
        }
    }

    /// One panel of up to sixteen rows against every token of `x`.
    ///
    /// # Safety
    /// AVX2, FMA and F16C must be available; `w` holds `first + valid` rows of `cols` elements,
    /// `x` holds `part.len() * cols` values.
    #[allow(clippy::needless_range_loop)]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn panel<const KIND: u32>(
        w: &[u8],
        x: &[f32],
        rows: (usize, usize),
        cols: usize,
        panel: &mut [f32],
        part: &mut [[f32; MR]],
        out: &mut crate::cpu_schedule::Rows<'_>,
    ) {
        let batch = part.len();
        for k0 in (0..cols).step_by(KC) {
            let kc = (cols - k0).min(KC);
            decode::<KIND>(w, rows, cols, (k0, kc), panel);
            let mut token = 0;
            while token < batch {
                let at = (token, k0, kc);
                let fresh = k0 == 0;
                match (batch - token).min(NR) {
                    1 => pass::<1>(panel, x, cols, at, part, fresh),
                    2 => pass::<2>(panel, x, cols, at, part, fresh),
                    3 => pass::<3>(panel, x, cols, at, part, fresh),
                    4 => pass::<4>(panel, x, cols, at, part, fresh),
                    5 => pass::<5>(panel, x, cols, at, part, fresh),
                    _ => pass::<6>(panel, x, cols, at, part, fresh),
                }
                token += (batch - token).min(NR);
            }
        }
        for (token, lanes) in part.iter().enumerate() {
            for (y, v) in out.row(token).iter_mut().zip(lanes) {
                *y = *v;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn tile<const KIND: u32, const SINGLE: bool, const ROWS: usize>(
        w: &[u8],
        x: &[f32],
        first: usize,
        cols: usize,
        batch: usize,
        out: &mut crate::cpu_schedule::Rows<'_>,
    ) {
        let end = cols / 8 * 8;
        for b in (0..batch).step_by(2) {
            let mut a = [[_mm256_setzero_ps(); 4]; 2];
            for k in (0..end).step_by(8) {
                let x0 = _mm256_loadu_ps(x.as_ptr().add(b * cols + k));
                let x1 = if !SINGLE && b + 1 < batch {
                    _mm256_loadu_ps(x.as_ptr().add((b + 1) * cols + k))
                } else {
                    _mm256_setzero_ps()
                };
                macro_rules! row {
                    ($r:literal) => {
                        if $r < ROWS {
                            let v = load::<KIND>(w, (first + $r) * cols + k);
                            a[0][$r] = _mm256_fmadd_ps(v, x0, a[0][$r]);
                            if !SINGLE {
                                a[1][$r] = _mm256_fmadd_ps(v, x1, a[1][$r]);
                            }
                        }
                    };
                }
                row!(0);
                row!(1);
                row!(2);
                row!(3);
            }
            for (j, accumulators) in a.iter().enumerate().take((batch - b).min(2)) {
                for (r, &v) in accumulators.iter().enumerate().take(ROWS) {
                    let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                    let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
                    let mut sum = _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)));
                    for k in end..cols {
                        let i = (first + r) * cols + k;
                        let v = if KIND == 0 {
                            f32::from_le_bytes(w[i * 4..i * 4 + 4].try_into().unwrap())
                        } else {
                            let v = u16::from_le_bytes(w[i * 2..i * 2 + 2].try_into().unwrap());
                            if KIND == 1 {
                                crate::cpu_kernels::x86::half_to_float(v)
                            } else {
                                f32::from_bits((v as u32) << 16)
                            }
                        };
                        sum += v * x[(b + j) * cols + k];
                    }
                    out.row(b + j)[r] = sum;
                }
            }
        }
    }
}
