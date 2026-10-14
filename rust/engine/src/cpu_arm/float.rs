//! NEON F32, F16 and BF16 projections (and the f16 encoder of the KV cache).
//!
//! Same contract as the AVX2 kernels in `cpu_float.rs`: half-precision weights stay packed
//! (converted four at a time in registers), the input is rounded once to the weight type, and the
//! products are summed in f32. A panel is four weight rows by one, two or four tokens with
//! eight to sixteen independent accumulators, so a decode step is limited by the weight loads and
//! a prompt re-reads each weight row once per four tokens. With FEAT_BF16 the BF16 projection
//! uses `bfdot` (eight bf16 products per instruction) on an input rounded to bf16 bits.
//!
//! Loads go through byte vectors (`vld1q_u8`, `vld1_u8`), so the weight bytes need no alignment.
use crate::cpu_float::Projection;
use crate::cpu_schedule::Rows;
use half::{bf16, f16};
use rayon::prelude::*;
use std::arch::aarch64::*;
use std::borrow::Cow;

/// Bytes per element of the weight kinds 0 (F32), 1 (F16) and 30 (BF16).
const fn element_bytes(kind: u32) -> usize {
    if kind == 0 {
        4
    } else {
        2
    }
}

/// Whether the BF16 `bfdot` kernel may run: FEAT_BF16 and the `neon-i8mm` tier (both arrived
/// with Armv8.6, and `FLLM_CPU_ISA` below that tier switches it off).
pub(crate) fn bf16_dot() -> bool {
    crate::cpu_kernels::neon_i8mm() && super::features::detect().bf16
}

/// f32 to f16 bits, round to nearest even (`fcvtn`).
///
/// # Safety
/// NEON must be available.
#[target_feature(enable = "neon")]
unsafe fn encode_f16_neon(input: &[f32], output: &mut [u16]) {
    let end = input.len() / 4 * 4;
    for j in (0..end).step_by(4) {
        let half = vcvt_f16_f32(vld1q_f32(input.as_ptr().add(j)));
        vst1_u16(output.as_mut_ptr().add(j), vreinterpret_u16_f16(half));
    }
    for (dst, &x) in output[end..].iter_mut().zip(&input[end..]) {
        *dst = f16::from_f32(x).to_bits();
    }
}

/// `output[i] = f16(input[i])` as bits. The caller checked that NEON is enabled.
pub(crate) fn encode_f16(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len());
    // SAFETY: the caller (`cpu_float::encode_f16`) checked `neon()`; lengths were asserted.
    unsafe { encode_f16_neon(input, output) };
}

/// Four f32 values rounded to the nearest bf16 (ties to even, NaN quieted), as the upper 16 bits
/// of each lane with the lower 16 bits cleared.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn round_bf16(v: float32x4_t) -> uint32x4_t {
    let bits = vreinterpretq_u32_f32(v);
    let lsb = vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1));
    let rounded = vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), lsb));
    let nan = vcgtq_u32(
        vandq_u32(bits, vdupq_n_u32(0x7fff_ffff)),
        vdupq_n_u32(0x7f80_0000),
    );
    let quiet = vorrq_u32(bits, vdupq_n_u32(0x0040_0000));
    vandq_u32(vbslq_u32(nan, quiet, rounded), vdupq_n_u32(0xffff_0000))
}

/// `output = round_to_type(input)` for F16 (`KIND == 1`) and BF16 (`KIND == 30`).
///
/// # Safety
/// NEON must be available; the slices have equal length.
#[target_feature(enable = "neon")]
unsafe fn round_input_neon<const KIND: u32>(input: &[f32], output: &mut [f32]) {
    let end = input.len() / 4 * 4;
    for j in (0..end).step_by(4) {
        let v = vld1q_f32(input.as_ptr().add(j));
        let rounded = if KIND == 1 {
            vcvt_f32_f16(vcvt_f16_f32(v))
        } else {
            vreinterpretq_f32_u32(round_bf16(v))
        };
        vst1q_f32(output.as_mut_ptr().add(j), rounded);
    }
    for (dst, &x) in output[end..].iter_mut().zip(&input[end..]) {
        *dst = if KIND == 1 {
            f16::from_f32(x).to_f32()
        } else {
            bf16::from_f32(x).to_f32()
        };
    }
}

/// `output = bf16 bits of input`, rounded like [`round_bf16`].
///
/// # Safety
/// NEON must be available; the slices have equal length.
#[target_feature(enable = "neon")]
unsafe fn round_input_bf16_bits(input: &[f32], output: &mut [u16]) {
    let end = input.len() / 4 * 4;
    for j in (0..end).step_by(4) {
        let rounded = round_bf16(vld1q_f32(input.as_ptr().add(j)));
        vst1_u16(output.as_mut_ptr().add(j), vshrn_n_u32::<16>(rounded));
    }
    for (dst, &x) in output[end..].iter_mut().zip(&input[end..]) {
        *dst = bf16::from_f32(x).to_bits();
    }
}

/// The input rounded to the weight type (F32 is borrowed unchanged).
fn rounded_input(input: &[f32], kind: u32) -> Cow<'_, [f32]> {
    if kind == 0 {
        return Cow::Borrowed(input);
    }
    let mut output = vec![0.0; input.len()];
    let round = |out: &mut [f32], src: &[f32]| {
        // SAFETY: the callers checked `neon()`; `out` and `src` have equal length.
        unsafe {
            if kind == 1 {
                round_input_neon::<1>(src, out);
            } else {
                round_input_neon::<30>(src, out);
            }
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
    Cow::Owned(output)
}

/// The input as bf16 bits.
fn bf16_bits(input: &[f32]) -> Vec<u16> {
    let mut output = vec![0u16; input.len()];
    let round = |out: &mut [u16], src: &[f32]| {
        // SAFETY: the callers checked `neon()`; `out` and `src` have equal length.
        unsafe { round_input_bf16_bits(src, out) }
    };
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

/// Four weights starting at `p` as f32.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load4<const KIND: u32>(p: *const u8) -> float32x4_t {
    match KIND {
        0 => vreinterpretq_f32_u8(vld1q_u8(p)),
        1 => vcvt_f32_f16(vreinterpret_f16_u16(vreinterpret_u16_u8(vld1_u8(p)))),
        _ => vreinterpretq_f32_u32(vshll_n_u16::<16>(vreinterpret_u16_u8(vld1_u8(p)))),
    }
}

/// Weight `index` (in elements) of `w` as f32.
#[inline]
fn element<const KIND: u32>(w: &[u8], index: usize) -> f32 {
    match KIND {
        0 => f32::from_le_bytes(w[index * 4..index * 4 + 4].try_into().unwrap()),
        1 => f16::from_le_bytes([w[index * 2], w[index * 2 + 1]]).to_f32(),
        _ => f32::from_bits(u32::from(u16::from_le_bytes([w[index * 2], w[index * 2 + 1]])) << 16),
    }
}

/// Rows `first..first + valid` (1..=4; short panels repeat their last row, the extra results are
/// dropped) against tokens `token..token + T`. `U` vectors of four columns per step.
///
/// # Safety
/// NEON must be available; `w` holds `first + valid` rows of `cols` elements and `x` holds
/// `token + T` rows of `cols` values; `out` owns the columns.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tile<const KIND: u32, const T: usize, const U: usize>(
    w: &[u8],
    x: &[f32],
    (first, valid): (usize, usize),
    cols: usize,
    token: usize,
    out: &mut Rows<'_>,
) {
    let eb = element_bytes(KIND);
    assert!((1..=4).contains(&valid) && w.len() >= (first + valid) * cols * eb);
    assert!(x.len() >= (token + T) * cols);
    let rows: [*const u8; 4] =
        std::array::from_fn(|r| w.as_ptr().add((first + r.min(valid - 1)) * cols * eb));
    let xs: [*const f32; T] = std::array::from_fn(|t| x.as_ptr().add((token + t) * cols));
    let step = 4 * U;
    let end = cols / step * step;
    let mut acc = [[[vdupq_n_f32(0.0); 4]; T]; U];
    let mut k = 0;
    while k < end {
        let mut weights = [[vdupq_n_f32(0.0); U]; 4];
        for (r, row) in weights.iter_mut().enumerate() {
            for (u, v) in row.iter_mut().enumerate() {
                *v = load4::<KIND>(rows[r].add((k + 4 * u) * eb));
            }
        }
        for t in 0..T {
            for u in 0..U {
                let xv = vld1q_f32(xs[t].add(k + 4 * u));
                for r in 0..4 {
                    acc[u][t][r] = vfmaq_f32(acc[u][t][r], weights[r][u], xv);
                }
            }
        }
        k += step;
    }
    for t in 0..T {
        let row = out.row(token + t);
        for (r, y) in row.iter_mut().enumerate() {
            let mut v = acc[0][t][r];
            for acc in acc.iter().skip(1) {
                v = vaddq_f32(v, acc[t][r]);
            }
            let mut sum = vaddvq_f32(v);
            for kk in end..cols {
                sum += element::<KIND>(w, (first + r) * cols + kk) * *xs[t].add(kk);
            }
            *y = sum;
        }
    }
}

/// `bfdot` on eight bf16 pairs: `acc[i] += a[2i] * b[2i] + a[2i + 1] * b[2i + 1]`.
///
/// # Safety
/// The CPU must support NEON and FEAT_BF16.
#[inline]
#[target_feature(enable = "neon,bf16")]
unsafe fn bfdot(acc: float32x4_t, a: uint16x8_t, b: uint16x8_t) -> float32x4_t {
    let mut acc = acc;
    std::arch::asm!(
        "bfdot {acc:v}.4s, {a:v}.8h, {b:v}.8h",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

/// BF16 weights by an input rounded to bf16 bits, with `bfdot`; `U` vectors of eight columns per
/// step. Same panel shape as [`tile`].
///
/// # Safety
/// As [`tile`], plus FEAT_BF16; `x` holds `token + T` rows of `cols` bf16 values.
#[inline]
#[target_feature(enable = "neon,bf16")]
unsafe fn tile_bf16<const T: usize, const U: usize>(
    w: &[u8],
    x: &[u16],
    (first, valid): (usize, usize),
    cols: usize,
    token: usize,
    out: &mut Rows<'_>,
) {
    assert!((1..=4).contains(&valid) && w.len() >= (first + valid) * cols * 2);
    assert!(x.len() >= (token + T) * cols);
    let rows: [*const u8; 4] =
        std::array::from_fn(|r| w.as_ptr().add((first + r.min(valid - 1)) * cols * 2));
    let xs: [*const u16; T] = std::array::from_fn(|t| x.as_ptr().add((token + t) * cols));
    let step = 8 * U;
    let end = cols / step * step;
    let mut acc = [[[vdupq_n_f32(0.0); 4]; T]; U];
    let mut k = 0;
    while k < end {
        let mut weights = [[vdupq_n_u16(0); U]; 4];
        for (r, row) in weights.iter_mut().enumerate() {
            for (u, v) in row.iter_mut().enumerate() {
                *v = vreinterpretq_u16_u8(vld1q_u8(rows[r].add((k + 8 * u) * 2)));
            }
        }
        for t in 0..T {
            for u in 0..U {
                let xv = vld1q_u16(xs[t].add(k + 8 * u));
                for r in 0..4 {
                    acc[u][t][r] = bfdot(acc[u][t][r], weights[r][u], xv);
                }
            }
        }
        k += step;
    }
    for t in 0..T {
        let row = out.row(token + t);
        for (r, y) in row.iter_mut().enumerate() {
            let mut v = acc[0][t][r];
            for acc in acc.iter().skip(1) {
                v = vaddq_f32(v, acc[t][r]);
            }
            let mut sum = vaddvq_f32(v);
            for kk in end..cols {
                let xv = f32::from_bits(u32::from(*xs[t].add(kk)) << 16);
                sum += element::<30>(w, (first + r) * cols + kk) * xv;
            }
            *y = sum;
        }
    }
}

/// Activations of one `linear_group` call.
enum Input<'a> {
    /// f32 values, already rounded to the weight type.
    Float(Cow<'a, [f32]>),
    /// bf16 bits, for the `bfdot` kernel.
    Bf16(Vec<u16>),
}

/// Round a shared input once and distribute all sibling rows in one queue; the kernel for
/// F32 (`kind == 0`), F16 (1) and BF16 (30). `None` is never returned: the caller checked that
/// NEON is enabled and `kind` is one of the three.
pub(crate) fn linear_group(
    projections: &[Projection<'_>],
    kind: u32,
    input: &[f32],
    cols: usize,
) -> Vec<Vec<f32>> {
    assert!(matches!(kind, 0 | 1 | 30));
    assert!(cols > 0 && input.len().is_multiple_of(cols));
    let width = element_bytes(kind);
    for p in projections {
        assert!(p.rows > 0);
        assert_eq!(p.weights.len(), p.rows * cols * width);
        assert!(p.bias.is_none_or(|b| b.len() == p.rows));
    }
    let batch = input.len() / cols;
    if batch == 0 || projections.is_empty() {
        return projections.iter().map(|_| Vec::new()).collect();
    }
    let dot = kind == 30 && bf16_dot();
    let x = if dot {
        Input::Bf16(bf16_bits(input))
    } else {
        Input::Float(rounded_input(input, kind))
    };
    let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
    let mut output = rows
        .iter()
        .map(|&r| vec![0.0; batch * r])
        .collect::<Vec<_>>();
    crate::cpu_schedule::row_panels(
        &mut output,
        &rows,
        batch,
        4,
        || (),
        |_, member, first, out| {
            let p = &projections[member];
            let valid = (p.rows - first).min(4);
            let mut token = 0;
            while token < batch {
                let left = batch - token;
                // SAFETY: NEON (and FEAT_BF16 for `Input::Bf16`) was detected by the caller;
                // the shapes were asserted above and `out` owns these columns.
                unsafe {
                    macro_rules! run {
                        ($kind:literal) => {{
                            let Input::Float(x) = &x else { unreachable!() };
                            if left >= 4 {
                                tile::<$kind, 4, 1>(p.weights, x, (first, valid), cols, token, out);
                                4
                            } else if left >= 2 {
                                tile::<$kind, 2, 2>(p.weights, x, (first, valid), cols, token, out);
                                2
                            } else {
                                tile::<$kind, 1, 2>(p.weights, x, (first, valid), cols, token, out);
                                1
                            }
                        }};
                    }
                    token += match (&x, kind) {
                        (Input::Bf16(x), _) => {
                            if left >= 4 {
                                tile_bf16::<4, 1>(p.weights, x, (first, valid), cols, token, out);
                                4
                            } else if left >= 2 {
                                tile_bf16::<2, 2>(p.weights, x, (first, valid), cols, token, out);
                                2
                            } else {
                                tile_bf16::<1, 2>(p.weights, x, (first, valid), cols, token, out);
                                1
                            }
                        }
                        (_, 0) => run!(0),
                        (_, 1) => run!(1),
                        _ => run!(30),
                    };
                }
            }
            for b in 0..batch {
                for (r, y) in out.row(b).iter_mut().enumerate() {
                    *y += p.bias.map_or(0.0, |bias| bias[first + r]);
                }
            }
        },
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::Lcg;

    fn encode_weights(values: &[f32], kind: u32) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| match kind {
                0 => v.to_le_bytes().to_vec(),
                1 => f16::from_f32(v).to_le_bytes().to_vec(),
                _ => bf16::from_f32(v).to_le_bytes().to_vec(),
            })
            .collect()
    }

    fn round(kind: u32, v: f32) -> f32 {
        match kind {
            1 => f16::from_f32(v).to_f32(),
            30 => bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    #[test]
    fn encoders_and_rounding_preserve_ties_and_tails() {
        if !crate::cpu_kernels::neon() {
            return;
        }
        let mut input = (0..1031)
            .map(|i| (i as f32 - 515.0) / 17.0)
            .collect::<Vec<_>>();
        input[..10].copy_from_slice(&[
            0.0,
            -0.0,
            1.0 + 1.0 / 2048.0,
            -1.0 - 1.0 / 2048.0,
            65504.0,
            1e-8,
            1.0 / 16384.0,
            1.0 + 1.0 / 256.0,
            1.0 + 3.0 / 256.0,
            3.0e38,
        ]);
        let mut encoded = vec![0u16; input.len()];
        encode_f16(&input, &mut encoded);
        assert_eq!(
            encoded,
            input
                .iter()
                .map(|&v| f16::from_f32(v).to_bits())
                .collect::<Vec<_>>()
        );
        for kind in [1, 30] {
            let have = rounded_input(&input, kind);
            let want = input.iter().map(|&v| round(kind, v)).collect::<Vec<_>>();
            assert_eq!(
                have.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "kind {kind}"
            );
        }
        assert_eq!(
            bf16_bits(&input),
            input
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect::<Vec<_>>()
        );
        // NaN stays NaN.
        let nan = rounded_input(&[f32::NAN, 1.0, 2.0, 3.0, f32::NAN], 30);
        assert!(nan[0].is_nan() && nan[4].is_nan() && nan[1] == 1.0);
    }

    #[test]
    fn projections_match_the_f64_reference_at_every_tail() {
        if !crate::cpu_kernels::neon() {
            return;
        }
        let mut rng = Lcg(0xf10a7);
        for kind in [0u32, 1, 30] {
            for (rows, cols) in [
                (1usize, 13usize),
                (2, 8),
                (5, 40),
                (17, 300),
                (33, 520),
                (4, 4),
            ] {
                let weights: Vec<f32> = rng
                    .vector(rows * cols, 1.0)
                    .into_iter()
                    .map(|v| round(kind, v))
                    .collect();
                let bytes = encode_weights(&weights, kind);
                let bias: Vec<f32> = (0..rows).map(|r| r as f32 / 10.0 - 1.0).collect();
                for batch in [1usize, 2, 3, 4, 5, 7, 13] {
                    let input = rng.vector(batch * cols, 2.0);
                    let got =
                        crate::cpu_float::linear(&bytes, kind, &input, rows, cols, Some(&bias))
                            .expect("float kernel");
                    for b in 0..batch {
                        for r in 0..rows {
                            let mut want = f64::from(bias[r]);
                            let mut magnitude = want.abs();
                            for k in 0..cols {
                                let term = f64::from(weights[r * cols + k])
                                    * f64::from(round(kind, input[b * cols + k]));
                                want += term;
                                magnitude += term.abs();
                            }
                            let have = f64::from(got[b * rows + r]);
                            assert!(
                                (have - want).abs() <= 1e-4 * magnitude + 1e-6,
                                "kind {kind} rows {rows} cols {cols} batch {batch} cell ({b},{r}): {have} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_siblings_equal_single_projections() {
        if !crate::cpu_kernels::neon() {
            return;
        }
        let cols = 77;
        let rows = [1usize, 3, 7];
        let mut rng = Lcg(5);
        for kind in [0u32, 1, 30] {
            let weights = rows
                .iter()
                .map(|&r| encode_weights(&rng.vector(r * cols, 1.0), kind))
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
            for batch in [0usize, 1, 3, 6] {
                let input = rng.vector(batch * cols, 1.0);
                let grouped = crate::cpu_float::linear_group(&members, kind, &input, cols).unwrap();
                for (member, result) in members.iter().zip(grouped) {
                    assert_eq!(
                        result,
                        crate::cpu_float::linear(
                            member.weights,
                            kind,
                            &input,
                            member.rows,
                            cols,
                            member.bias
                        )
                        .unwrap()
                    );
                }
            }
        }
    }
}
