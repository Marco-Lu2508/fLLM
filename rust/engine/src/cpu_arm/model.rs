//! The dataflow of the NEON kernels in `quant.rs`, written with the scalar models of `emu.rs`
//! and the scalar decoders of `formats.rs`. Test-only.
//!
//! The NEON kernels were written line by line after these functions. Running them on x86
//! checks everything that is not an intrinsic call: block layout and scale decoding, the lane
//! algebra of `sdot` and `smmla` (which register holds which row and token), the zero padding
//! of short panels, the token-pair tail and the correction terms. What it cannot check is
//! whether a NEON intrinsic does what `emu.rs` assumes; `quant.rs` has AArch64-only tests
//! against `formats.rs` for the decoders, and `selftest-cpu` covers the kernels end to end.
use super::emu::{addv, cvt, dup, fma, sdot, smmla, zip_hi, zip_lo, F32x4, I32x4, I8x16};
use super::formats::{decode_row, Format, Group};
use crate::cpu_quantized::X;

fn halves(q: &[i8; 32]) -> [I8x16; 2] {
    [q[..16].try_into().unwrap(), q[16..].try_into().unwrap()]
}

/// One group times one activation record into a four-lane accumulator (`accumulate`).
fn accumulate(acc: &mut F32x4, split: bool, group: &Group, xq: [I8x16; 2], xd: f32) {
    let w = halves(&group.q);
    let zero: I32x4 = [0; 4];
    if split {
        let lo = sdot(zero, w[0], xq[0]);
        let hi = sdot(zero, w[1], xq[1]);
        *acc = fma(*acc, cvt(lo), dup(group.d[0] * xd));
        *acc = fma(*acc, cvt(hi), dup(group.d[1] * xd));
    } else {
        let dot = sdot(sdot(zero, w[0], xq[0]), w[1], xq[1]);
        *acc = fma(*acc, cvt(dot), dup(group.d[0] * xd));
    }
}

/// Minimum correction of one group (`offset`).
fn offset(acc: &mut f32, group: &Group, xr: &X) {
    *acc = group.m[0].mul_add(xr.sum[0], group.m[1].mul_add(xr.sum[1], *acc));
}

fn activation(xr: &X) -> [I8x16; 2] {
    halves(&xr.q)
}

/// Decode kernel: `rows` (1..=4) rows from `first` against one activation row.
pub(super) fn gemv(format: Format, w: &[u8], x: &[X], first: usize, rows: usize) -> Vec<f32> {
    assert!((1..=4).contains(&rows));
    let groups = format.groups();
    let width = format.block_bytes();
    let blocks = x.len() / groups;
    let stride = blocks * width;
    let mut acc = vec![[0.0f32; 4]; rows];
    let mut offsets = vec![0.0f32; rows];
    for block in 0..blocks {
        for g in 0..groups {
            let xr = &x[block * groups + g];
            let xq = activation(xr);
            for r in 0..rows {
                let start = (first + r) * stride + block * width;
                let group = super::formats::decode_group(format, &w[start..start + width], g);
                accumulate(&mut acc[r], format.split_scales(), &group, xq, xr.d);
                if format.has_min() {
                    offset(&mut offsets[r], &group, xr);
                }
            }
        }
    }
    (0..rows).map(|r| addv(acc[r]) + offsets[r]).collect()
}

/// Unpack up to four rows from `first` into a zero-padded four-row panel (`nb` groups per row).
pub(super) fn unpack_panel(
    format: Format,
    w: &[u8],
    width: usize,
    nb: usize,
    first: usize,
    valid: usize,
) -> Vec<Group> {
    let mut panel = vec![Group::default(); 4 * nb];
    for r in 0..valid {
        decode_row(
            format,
            &w[(first + r) * width..][..width],
            &mut panel[r * nb..(r + 1) * nb],
        );
    }
    panel
}

/// Prompt kernel with `sdot`: tokens in pairs, four panel rows. Returns `batch` rows of `valid`
/// outputs.
pub(super) fn tile(
    format: Format,
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    valid: usize,
) -> Vec<Vec<f32>> {
    assert!(panel.len() >= 4 * nb && x.len() >= batch * nb);
    let mut out = vec![vec![0.0; valid]; batch];
    for b in (0..batch).step_by(2) {
        let mut acc = [[[0.0f32; 4]; 4]; 2];
        let mut offsets = [[0.0f32; 4]; 2];
        for kb in 0..nb {
            let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
            let xq = [activation(xs[0]), activation(xs[1])];
            for r in 0..4 {
                let group = &panel[r * nb + kb];
                for t in 0..2 {
                    accumulate(&mut acc[t][r], format.split_scales(), group, xq[t], xs[t].d);
                    if format.has_min() {
                        offset(&mut offsets[t][r], group, xs[t]);
                    }
                }
            }
        }
        for j in 0..(batch - b).min(2) {
            for r in 0..valid {
                out[b + j][r] = addv(acc[j][r]) + offsets[j][r];
            }
        }
    }
    out
}

/// Prompt kernel with `smmla`: row pairs against token pairs.
pub(super) fn tile_i8mm(
    format: Format,
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    valid: usize,
) -> Vec<Vec<f32>> {
    assert!(panel.len() >= 4 * nb && x.len() >= batch * nb);
    let zero: I32x4 = [0; 4];
    let mut out = vec![vec![0.0; valid]; batch];
    for b in (0..batch).step_by(2) {
        // Lanes: [row 2k token 0, row 2k token 1, row 2k+1 token 0, row 2k+1 token 1].
        let mut acc = [[0.0f32; 4]; 2];
        let mut offsets = [[0.0f32; 4]; 2];
        for kb in 0..nb {
            let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
            let (a, c) = (activation(xs[0]), activation(xs[1]));
            let xm = [
                zip_lo(a[0], c[0]),
                zip_hi(a[0], c[0]),
                zip_lo(a[1], c[1]),
                zip_hi(a[1], c[1]),
            ];
            let xd = [xs[0].d, xs[1].d, xs[0].d, xs[1].d];
            for (k, acc) in acc.iter_mut().enumerate() {
                let (wa, wb) = (&panel[2 * k * nb + kb], &panel[(2 * k + 1) * nb + kb]);
                let (qa, qb) = (halves(&wa.q), halves(&wb.q));
                let low = smmla(
                    smmla(zero, zip_lo(qa[0], qb[0]), xm[0]),
                    zip_hi(qa[0], qb[0]),
                    xm[1],
                );
                let high = smmla(
                    smmla(zero, zip_lo(qa[1], qb[1]), xm[2]),
                    zip_hi(qa[1], qb[1]),
                    xm[3],
                );
                let scale = |half: usize| -> F32x4 {
                    let dw = [wa.d[half], wa.d[half], wb.d[half], wb.d[half]];
                    std::array::from_fn(|i| dw[i] * xd[i])
                };
                if format.split_scales() {
                    *acc = fma(*acc, cvt(low), scale(0));
                    *acc = fma(*acc, cvt(high), scale(1));
                } else {
                    let sum: I32x4 = std::array::from_fn(|i| low[i] + high[i]);
                    *acc = fma(*acc, cvt(sum), scale(0));
                }
                if format.has_min() {
                    for (t, xr) in xs.into_iter().enumerate() {
                        offset(&mut offsets[t][2 * k], wa, xr);
                        offset(&mut offsets[t][2 * k + 1], wb, xr);
                    }
                }
            }
        }
        for j in 0..(batch - b).min(2) {
            for r in 0..valid {
                out[b + j][r] = acc[r / 2][(r % 2) * 2 + j] + offsets[j][r];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    /// Reference outputs `[token][row]` of rows `first..first + valid`.
    fn reference(
        w: &[u8],
        kind: u32,
        rows: usize,
        cols: usize,
        packed: &[X],
        first: usize,
        valid: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let (want, scale) = reference_packed(w, kind, rows, cols, packed);
        let batch = packed.len() / (cols / 32);
        let pick = |v: &[f64]| {
            (0..batch)
                .flat_map(|t| (0..valid).map(move |r| v[t * rows + first + r]))
                .collect::<Vec<_>>()
        };
        (pick(&want), pick(&scale))
    }

    fn check(format: Format, rows: usize, cols: usize, batch: usize, seed: u64) {
        let kind = format.kind();
        let mut rng = Lcg(seed ^ u64::from(kind));
        let w = random_weights(kind, rows, cols, &mut rng);
        let x = rng.vector(batch * cols, 3.0);
        let packed = crate::cpu_quantized::prepare(&x, cols, kind);
        let nb = cols / 32;
        let width = cols / (format.groups() * 32) * format.block_bytes();
        let name = format!("{format:?} rows {rows} cols {cols} batch {batch}");

        // Decode kernel, one token at a time, in panels of up to four rows.
        for t in 0..batch {
            let record = &packed[t * nb..(t + 1) * nb];
            let mut first = 0;
            while first < rows {
                let valid = (rows - first).min(4);
                let have = gemv(format, &w, record, first, valid);
                let (want, scale) = reference(&w, kind, rows, cols, &packed, first, valid);
                let have64: Vec<f32> = have;
                let (error, ok) = compare(
                    &have64,
                    &want[t * valid..(t + 1) * valid],
                    &scale[t * valid..(t + 1) * valid],
                );
                assert!(ok, "gemv {name} token {t} first {first}: error {error:e}");
                first += valid;
            }
        }

        // Prompt kernels over unpacked panels.
        let mut first = 0;
        while first < rows {
            let valid = (rows - first).min(4);
            let panel = unpack_panel(format, &w, width, nb, first, valid);
            let (want, scale) = reference(&w, kind, rows, cols, &packed, first, valid);
            for (label, have) in [
                ("tile", tile(format, &panel, &packed, nb, batch, valid)),
                (
                    "tile_i8mm",
                    tile_i8mm(format, &panel, &packed, nb, batch, valid),
                ),
            ] {
                let have: Vec<f32> = have.into_iter().flatten().collect();
                let (error, ok) = compare(&have, &want, &scale);
                assert!(ok, "{label} {name} first {first}: error {error:e}");
            }
            first += valid;
        }
    }

    #[test]
    fn every_format_matches_the_reference_at_odd_shapes() {
        for format in Format::ALL {
            let cols = if format.groups() == 8 { 768 } else { 352 };
            for rows in [1, 3, 4, 5, 9] {
                for batch in [1, 2, 3, 6, 7] {
                    check(
                        format,
                        rows,
                        cols,
                        batch,
                        0xa5 + rows as u64 * 31 + batch as u64,
                    );
                }
            }
        }
    }

    #[test]
    fn single_block_rows_and_long_rows() {
        for format in Format::ALL {
            let one = format.groups() * 32;
            check(format, 4, one, 3, 17);
            check(format, 2, one * 9, 2, 19);
        }
    }

    #[test]
    fn minimum_and_scale_terms_are_not_dropped() {
        // A block with zero weights except the minimum: Q2_K must still produce the offset.
        let mut block = vec![0u8; 84];
        block[0] = 0x05; // scale 5, minimum 0 for group 0 half 0; scales of the rest are zero
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[1] = 0x30; // minimum 3, scale 0 for group 0 half 1
        let x: Vec<f32> = (0..256).map(|i| (i % 7) as f32 - 3.0).collect();
        let packed = crate::cpu_quantized::prepare(&x, 256, 10);
        let out = gemv(Format::Q2K, &block, &packed, 0, 1);
        let reference = reference_packed(&block, 10, 1, 256, &packed);
        assert!((f64::from(out[0]) - reference.0[0]).abs() <= 1e-4 * reference.1[0] + 1e-6);
        assert!(reference.1[0] > 0.0, "the minimum must contribute");
    }
}
