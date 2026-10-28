//! Native EXL2 / EXL3 projections on the CPU (portable Rust, no SIMD intrinsics; the loops auto-vectorize).
//!
//! The weights stay packed (layout: `fllm_gguf_probe::exl`); nothing is expanded to 16 bit.
//!
//! * EXL2: the activations are gathered once into stored order (`x'[k] = x[perm[k]]`), then every output row is decoded group by
//!   group (2..8-bit codes, LSB-first bitstream, codes may straddle words) into a small scratch vector and dotted with the
//!   gathered activations of every token; the group scale `(s + 1)^2 * max / 256` is applied once per group.
//! * EXL3: `y = diag(svh) H (R^T (H diag(suh) x))`. The input is scaled and transformed with the normalized 128-point
//!   Hadamard transform, every 16x16 trellis tile is decoded (tail-biting ring of K-bit symbols, 16-bit state window,
//!   65536-entry codebook table) and immediately multiplied against all tokens, then the output is transformed back and
//!   scattered through the output map. No dense weight matrix exists at any time.
//!
//! Arithmetic contract: scales and transforms are applied in F32; the importer's expanded F16 path rounds to half precision after
//! every stage, so the two agree to about 1e-3 relative (docs/exl-formats.md).

use fllm_gguf_probe::exl::{self, Exl2View, Exl3View, EXL2_TYPE, EXL3_TYPE};
use rayon::prelude::*;

/// Whether `kind` is a native EXL tensor type this module computes.
pub(crate) fn handles(kind: u32) -> bool {
    exl::is_exl_type(kind)
}

/// Full validation of a blob against the shape of its projection (call once, when the weight is loaded).
pub(crate) fn validate(blob: &[u8], kind: u32, rows: usize, cols: usize) -> Result<(), String> {
    let (r, c) = match kind {
        EXL2_TYPE => {
            let v = Exl2View::validate(blob)?;
            (v.rows(), v.cols())
        }
        EXL3_TYPE => {
            let v = Exl3View::validate(blob)?;
            (v.rows(), v.cols())
        }
        _ => return Err(format!("GGUF type {kind} is not an EXL tensor")),
    };
    if (r, c) != (rows, cols) {
        return Err(format!("EXL tensor holds {r} x {c} values, expected {rows} x {cols}"));
    }
    Ok(())
}

/// The F16 bytes of the whole matrix: what a backend without an EXL kernel is given instead of the blob (expand-at-upload fallback).
pub(crate) fn expanded_f16(blob: &[u8], kind: u32, rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    validate(blob, kind, rows, cols)?;
    let values = exl::dequantize(blob, kind, rows * cols)?;
    Ok(values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for ((lane, &p), &q) in lanes.iter_mut().zip(x).zip(y) {
            *lane += p * q;
        }
    }
    lanes.iter().sum::<f32>() + tail
}

/// `y = x W^T + bias` for a packed EXL tensor of `rows` x `cols` and `input` of `batch` tokens (`[batch][cols]`); the result is
/// `[batch][rows]`.
pub(crate) fn linear(blob: &[u8], kind: u32, input: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> Result<Vec<f32>, String> {
    if cols == 0 || !input.len().is_multiple_of(cols) {
        return Err("EXL projection input dimensions do not match".into());
    }
    if bias.is_some_and(|b| b.len() < rows) {
        return Err("EXL projection bias is shorter than the output rows".into());
    }
    let batch = input.len() / cols;
    let mut out = match kind {
        EXL2_TYPE => exl2(blob, input, rows, cols, batch)?,
        EXL3_TYPE => exl3(blob, input, rows, cols, batch)?,
        _ => return Err(format!("GGUF type {kind} is not an EXL tensor")),
    };
    if let Some(bias) = bias {
        for token in out.chunks_exact_mut(rows) {
            for (y, b) in token.iter_mut().zip(bias) {
                *y += b;
            }
        }
    }
    Ok(out)
}

fn exl2(blob: &[u8], input: &[f32], rows: usize, cols: usize, batch: usize) -> Result<Vec<f32>, String> {
    let v = Exl2View::parse(blob)?;
    if (v.rows(), v.cols()) != (rows, cols) {
        return Err(format!("EXL2 tensor holds {} x {} values, expected {rows} x {cols}", v.rows(), v.cols()));
    }
    // activations in stored order (the permutation was validated when the weight was loaded; out-of-range entries read as 0)
    let mut gathered = vec![0f32; batch * cols];
    for (dst, src) in gathered.chunks_exact_mut(cols).zip(input.chunks_exact(cols)) {
        for (k, d) in dst.iter_mut().enumerate() {
            *d = src.get(v.perm(k)).copied().unwrap_or(0.0);
        }
    }
    let mut per_row = vec![0f32; rows * batch];
    per_row.par_chunks_mut(batch).enumerate().for_each_init(Vec::<f32>::new, |codes, (r, acc)| {
        let row = v.row(r);
        for g in 0..v.groups() {
            let e = v.group(g);
            codes.resize(e.n, 0.0);
            exl::unpack_centered(v.group_words(row, &e), e.bits, codes);
            let scale = v.scale(row, g, &e);
            for (t, a) in acc.iter_mut().enumerate() {
                *a += scale * dot(codes, &gathered[t * cols + e.start..t * cols + e.start + e.n]);
            }
        }
    });
    let mut out = vec![0f32; batch * rows];
    for (r, values) in per_row.chunks_exact(batch).enumerate() {
        for (t, &y) in values.iter().enumerate() {
            out[t * rows + r] = y;
        }
    }
    Ok(out)
}

fn exl3(blob: &[u8], input: &[f32], rows: usize, cols: usize, batch: usize) -> Result<Vec<f32>, String> {
    let v = Exl3View::parse(blob)?;
    if (v.rows(), v.cols()) != (rows, cols) {
        return Err(format!("EXL3 tensor holds {} x {} values, expected {rows} x {cols}", v.rows(), v.cols()));
    }
    let lut = exl::codebook_lut(v.codebook());
    let suh: Vec<f32> = (0..cols).map(|i| v.suh(i)).collect();
    let svh: Vec<f32> = (0..rows).map(|o| v.svh(o)).collect();
    // x <- H (suh * x) on every 128-block
    let mut x = input.to_vec();
    x.par_chunks_mut(cols).for_each(|token| {
        for (value, s) in token.iter_mut().zip(&suh) {
            *value *= s;
        }
        for block in token.as_chunks_mut::<128>().0 {
            exl::hadamard128(block);
        }
    });
    let (tiles_in, tiles_out) = (cols / 16, rows / 16);
    // z = R^T x, one task per tile column of 16 outputs; every decoded tile serves all tokens
    let columns: Vec<Vec<f32>> = (0..tiles_out)
        .into_par_iter()
        .map(|no| {
            let mut acc = vec![0f32; batch * 16];
            let mut tile = [0f32; 256];
            for ki in 0..tiles_in {
                v.decode_tile(no, ki, lut, &mut tile);
                for t in 0..batch {
                    let xs = &x[t * cols + ki * 16..t * cols + ki * 16 + 16];
                    let a = &mut acc[t * 16..t * 16 + 16];
                    for (r, &xv) in xs.iter().enumerate() {
                        for (c, slot) in a.iter_mut().enumerate() {
                            *slot += xv * tile[r * 16 + c];
                        }
                    }
                }
            }
            acc
        })
        .collect();
    let mut z = vec![0f32; batch * rows];
    for (no, acc) in columns.iter().enumerate() {
        for t in 0..batch {
            z[t * rows + no * 16..t * rows + no * 16 + 16].copy_from_slice(&acc[t * 16..t * 16 + 16]);
        }
    }
    // y = svh * H z on every 128-block, scattered to the output rows
    let mut out = vec![0f32; batch * rows];
    for (zt, ot) in z.chunks_exact_mut(rows).zip(out.chunks_exact_mut(rows)) {
        for (base, block) in zt.as_chunks_mut::<128>().0.iter_mut().enumerate() {
            exl::hadamard128(block);
            for (o, &value) in block.iter().enumerate() {
                let source = base * 128 + o;
                let row = v.out_row(source);
                if let Some(slot) = ot.get_mut(row) {
                    *slot = value * svh[source];
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::Lcg;

    /// f64 reference over the expanded matrix; returns the outputs and the sums of absolute terms.
    fn reference(weights: &[f32], rows: usize, cols: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let batch = x.len() / cols;
        let mut y = vec![0f64; batch * rows];
        let mut scale = vec![0f64; batch * rows];
        for t in 0..batch {
            for r in 0..rows {
                for c in 0..cols {
                    let term = f64::from(weights[r * cols + c]) * f64::from(x[t * cols + c]);
                    y[t * rows + r] += term;
                    scale[t * rows + r] += term.abs();
                }
            }
        }
        (y, scale)
    }

    fn check(blob: &[u8], kind: u32, rows: usize, cols: usize, batch: usize, seed: u64, tolerance: f64) {
        validate(blob, kind, rows, cols).unwrap();
        let mut rng = Lcg(seed);
        let x = rng.vector(batch * cols, 1.0);
        let bias = rng.vector(rows, 0.5);
        let dense = exl::dequantize(blob, kind, rows * cols).unwrap();
        let (want, scale) = reference(&dense, rows, cols, &x);
        let got = linear(blob, kind, &x, rows, cols, Some(&bias)).unwrap();
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            let w = want[i] + f64::from(bias[i % rows]);
            let bound = tolerance * (scale[i] + 1.0);
            assert!((f64::from(got[i]) - w).abs() <= bound, "kind {kind:#x} output {i}: {} vs {w} (bound {bound})", got[i]);
        }
    }

    #[test]
    fn exl2_matches_the_dequantized_matrix_for_all_batch_shapes() {
        for (rows, cols, batch, seed) in [(8usize, 32usize, 1usize, 1u64), (40, 96, 3, 2), (24, 256, 9, 3), (136, 160, 1, 4)] {
            let mut rng = Lcg(seed);
            let blob = exl::synthetic_exl2(rows, cols, &mut || rng.next());
            check(&blob, EXL2_TYPE, rows, cols, batch, seed + 100, 1e-5);
        }
    }

    #[test]
    fn exl3_matches_the_dequantized_matrix_for_every_width_and_codebook() {
        for (bits, codebook) in [(1usize, 0u8), (2, 1), (3, 2), (4, 0), (5, 1), (6, 2), (7, 0), (8, 1)] {
            let (rows, cols) = (256usize, 128usize);
            let mut rng = Lcg(bits as u64 * 7 + u64::from(codebook));
            let blob = exl::synthetic_exl3(rows, cols, bits, codebook, &mut || rng.next());
            for batch in [1usize, 4] {
                check(&blob, EXL3_TYPE, rows, cols, batch, 50 + bits as u64, 2e-5);
            }
        }
        let mut rng = Lcg(9);
        let blob = exl::synthetic_exl3(128, 384, 3, 0, &mut || rng.next());
        check(&blob, EXL3_TYPE, 128, 384, 2, 77, 2e-5);
    }

    /// Indicative CPU speed (run with `-- --ignored --nocapture`; the numbers depend on the machine and on what else runs on it).
    #[test]
    #[ignore]
    fn bench_cpu_decode_speed() {
        for (kind, label) in [(EXL2_TYPE, "exl2"), (EXL3_TYPE, "exl3 3-bit")] {
            let (rows, cols) = (4096usize, 4096usize);
            let mut rng = Lcg(77);
            let blob = if kind == EXL2_TYPE {
                exl::synthetic_exl2(rows, cols, &mut || rng.next())
            } else {
                exl::synthetic_exl3(rows, cols, 3, 0, &mut || rng.next())
            };
            for batch in [1usize, 8, 64] {
                let x = rng.vector(batch * cols, 1.0);
                let _ = linear(&blob, kind, &x, rows, cols, None).unwrap();
                let start = std::time::Instant::now();
                let reps = 5;
                for _ in 0..reps {
                    let _ = linear(&blob, kind, &x, rows, cols, None).unwrap();
                }
                let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
                eprintln!("BENCH {label} {rows}x{cols} batch {batch}: {ms:.1} ms per call, {:.2} Gweights/s, {} threads", (rows * cols) as f64 / ms / 1e6, rayon::current_num_threads());
            }
        }
    }

    #[test]
    fn the_upload_fallback_expands_to_f16_values_of_the_native_matrix() {
        for kind in [EXL2_TYPE, EXL3_TYPE] {
            let (rows, cols) = (128usize, 128usize);
            let mut rng = Lcg(31);
            let blob = if kind == EXL2_TYPE {
                exl::synthetic_exl2(rows, cols, &mut || rng.next())
            } else {
                exl::synthetic_exl3(rows, cols, 3, 1, &mut || rng.next())
            };
            let bytes = expanded_f16(&blob, kind, rows, cols).unwrap();
            assert_eq!(bytes.len(), rows * cols * 2);
            let dense = exl::dequantize(&blob, kind, rows * cols).unwrap();
            for (value, half) in dense.iter().zip(bytes.as_chunks::<2>().0) {
                assert_eq!(half::f16::from_le_bytes(*half), half::f16::from_f32(*value));
            }
            assert!(expanded_f16(&blob, kind, rows, cols + 1).is_err());
        }
    }

    #[test]
    fn shape_and_type_mismatches_are_errors() {
        let mut rng = Lcg(5);
        let blob = exl::synthetic_exl2(8, 32, &mut || rng.next());
        assert!(validate(&blob, EXL2_TYPE, 8, 64).is_err());
        assert!(validate(&blob, EXL3_TYPE, 8, 32).is_err());
        assert!(linear(&blob, EXL2_TYPE, &[0.0; 33], 8, 32, None).is_err());
        assert!(linear(&blob, EXL2_TYPE, &[0.0; 32], 8, 32, Some(&[0.0; 3])).is_err());
        assert!(linear(&blob[..blob.len() - 4], EXL2_TYPE, &[0.0; 32], 8, 32, None).is_err());
    }
}

#[cfg(test)]
mod mirror_tests;
