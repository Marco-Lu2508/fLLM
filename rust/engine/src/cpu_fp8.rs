//! CPU kernels for block-wise FP8 weights (`FP8B`, layout: `fllm_gguf_probe::Fp8bLayout`).
//!
//! The weights stay 8-bit in memory (one byte per weight plus one f32 scale per tile). A row of the weight matrix is expanded to f32
//! in a small scratch buffer, tile by tile with the scale of the tile applied, and multiplied with every activation row in f32. The
//! expansion of an E4M3FN code is plain integer arithmetic (no table, no branch that the compiler cannot turn into a select), so the
//! loops compile to SIMD code on every target: the x86-64 build is compiled once more with AVX2 and FMA enabled and chosen at run time
//! (`cpu_kernels::isa`), AArch64 uses NEON with its native fused multiply-add, anything else the portable code. The result does not
//! depend on the number of threads: threads split the output rows, and the summation order inside a row is fixed.
//!
//! This is the reference for the GPU kernels of the same format (the self-test compares the device with an f64 evaluation of the
//! dequantized weights and runs this kernel through the CPU backend), and the path of CPU-only machines.
use fllm_gguf_probe::Fp8bLayout;
use rayon::prelude::*;

/// Output rows one task decodes and multiplies (the scratch row stays in the cache of the core).
const ROWS_PER_TASK: usize = 4;

/// Value of one unit of the subnormal mantissa of E4M3 (`2^-9`).
const SUBNORMAL_UNIT: f32 = 1.0 / 512.0;

/// The value of an E4M3FN code: the same number as `compute::e4m3fn_to_f32`, computed without a table. The exponent and mantissa
/// bits of the code are moved to the exponent and top mantissa bits of an f32 (exponent bias 127 instead of 7, hence the added
/// `120 << 23`); the subnormal codes (exponent 0) are `mantissa * 2^-9`; `0x7f` and `0xff` are NaN (E4M3FN has no infinity).
#[inline(always)]
fn decode_code(code: u8) -> f32 {
    let v = u32::from(code);
    let mag = v & 0x7f;
    let normal = f32::from_bits((mag << 20) + (120 << 23));
    let subnormal = (mag & 7) as f32 * SUBNORMAL_UNIT;
    let value = if mag >> 3 == 0 { subnormal } else { normal };
    let value = if mag == 0x7f { f32::NAN } else { value };
    f32::from_bits(value.to_bits() | ((v & 0x80) << 24))
}

/// `out[i] = value(codes[i]) * scale`.
#[inline(always)]
fn decode_run(codes: &[u8], scale: f32, out: &mut [f32]) {
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = decode_code(c) * scale;
    }
}

/// Dot product with 16 independent partial sums (two 256-bit vectors) and a fixed reduction order.
#[inline(always)]
fn dot<const FMA: bool>(x: &[f32], w: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    let ((xs, _), (ws, _)) = (x.as_chunks::<16>(), w.as_chunks::<16>());
    let full = xs.len() * 16;
    let mut acc = [0f32; 16];
    for (xc, wc) in xs.iter().zip(ws) {
        for ((a, &x), &w) in acc.iter_mut().zip(xc).zip(wc) {
            *a = if FMA { x.mul_add(w, *a) } else { x * w + *a };
        }
    }
    let mut tail = 0f32;
    for (&a, &b) in x[full..].iter().zip(&w[full..]) {
        tail = if FMA { a.mul_add(b, tail) } else { a * b + tail };
    }
    let lanes: [f32; 8] = std::array::from_fn(|i| acc[i] + acc[i + 8]);
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7])) + tail
}

/// Where the scales of a weight matrix are: one per tile behind the codes (`FP8B`), or one binary16 scale per row behind the codes (`FP8M`, the
/// per-tensor and per-channel scales of ModelOpt and compressed-tensors checkpoints).
#[derive(Clone, Copy)]
enum Scales<'a> {
    Tiles(&'a Fp8bLayout),
    PerRow { rows: usize, cols: usize },
}

impl Scales<'_> {
    fn cols(&self) -> usize {
        match self {
            Scales::Tiles(layout) => layout.cols,
            Scales::PerRow { cols, .. } => *cols,
        }
    }
}

/// Decode weight row `row` of the (first) matrix into `buf` (`cols` values), tile by tile.
#[inline(always)]
fn decode_row(src: Scales<'_>, bytes: &[u8], row: usize, buf: &mut [f32]) {
    match src {
        Scales::Tiles(layout) => {
            let base = row * layout.cols;
            let codes = &bytes[base..base + layout.cols];
            for (tile, (c, o)) in codes.chunks(layout.block_cols).zip(buf.chunks_mut(layout.block_cols)).enumerate() {
                let scale = layout.scale(bytes, layout.tile_index(0, row, tile * layout.block_cols));
                decode_run(c, scale, o);
            }
        }
        Scales::PerRow { rows, cols } => {
            let at = rows * cols + row * 2;
            let scale = half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32();
            decode_run(&bytes[row * cols..(row + 1) * cols], scale, buf);
        }
    }
}

/// Rows `row0..` of the product for one task: `out` holds `batch` values per row, row after row.
#[inline(always)]
fn rows_body<const FMA: bool>(src: Scales<'_>, bytes: &[u8], input: &[f32], batch: usize, row0: usize, out: &mut [f32], buf: &mut [f32]) {
    let cols = src.cols();
    for (i, res) in out.chunks_exact_mut(batch).enumerate() {
        decode_row(src, bytes, row0 + i, buf);
        for (b, r) in res.iter_mut().enumerate() {
            *r = dot::<FMA>(&input[b * cols..(b + 1) * cols], buf);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2(src: Scales<'_>, bytes: &[u8], input: &[f32], batch: usize, row0: usize, out: &mut [f32], buf: &mut [f32]) {
    rows_body::<true>(src, bytes, input, batch, row0, out, buf);
}

fn rows(src: Scales<'_>, bytes: &[u8], input: &[f32], batch: usize, row0: usize, out: &mut [f32], buf: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::isa() != "scalar" {
        // SAFETY: `isa` reports an AVX2 tier only when the CPU has avx2 and fma.
        unsafe { rows_avx2(src, bytes, input, batch, row0, out, buf) };
        return;
    }
    // NEON has a fused multiply-add in the base instruction set of AArch64
    rows_body::<{ cfg!(target_arch = "aarch64") }>(src, bytes, input, batch, row0, out, buf);
}

/// The product of `input` (`batch` rows of `cols` values) with the matrix whose codes and scales `bytes` hold: `batch` rows of `rows_out` values.
fn drive(src: Scales<'_>, bytes: &[u8], input: &[f32], rows_out: usize, cols: usize, bias: Option<&[f32]>) -> Result<Vec<f32>, String> {
    if cols == 0 || !input.len().is_multiple_of(cols) {
        return Err("projection input dimensions do not match".into());
    }
    if bias.is_some_and(|b| b.len() < rows_out) {
        return Err("projection bias shorter than the output rows".into());
    }
    let batch = input.len() / cols;
    if batch == 0 {
        return Ok(Vec::new());
    }
    // rows-major [r][b] first (every task owns whole rows), then transposed to [b][r]
    let mut tmp = vec![0f32; rows_out * batch];
    tmp.par_chunks_mut(ROWS_PER_TASK * batch)
        .enumerate()
        .for_each_init(|| vec![0f32; cols], |buf, (task, chunk)| rows(src, bytes, input, batch, task * ROWS_PER_TASK, chunk, buf));
    let mut out = if batch == 1 {
        tmp
    } else {
        let mut out = vec![0f32; rows_out * batch];
        for (r, values) in tmp.chunks_exact(batch).enumerate() {
            for (b, &v) in values.iter().enumerate() {
                out[b * rows_out + r] = v;
            }
        }
        out
    };
    if let Some(bias) = bias {
        for row in out.chunks_exact_mut(rows_out) {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
    }
    Ok(out)
}

/// `out[b][r] = sum_c input[b][c] * w[r][c] (+ bias[r])` for an `FP8B` tensor `packed` (GGUF type `kind`) of one `rows` x `cols` matrix.
/// `input` holds `batch` rows of `cols` values; the result `batch` rows of `rows` values.
pub fn linear(kind: u32, packed: &[u8], input: &[f32], rows_out: usize, cols: usize, bias: Option<&[f32]>) -> Result<Vec<f32>, String> {
    let layout = Fp8bLayout::parse(kind, packed)?;
    if layout.slabs != 1 || layout.rows != rows_out || layout.cols != cols {
        return Err(format!(
            "FP8B weight holds {} matrices of {} x {}, a single {rows_out} x {cols} matrix was expected",
            layout.slabs, layout.rows, layout.cols
        ));
    }
    drive(Scales::Tiles(&layout), packed, input, rows_out, cols, bias)
}

/// The same for an `FP8M` tensor (GGUF type 0x8000_0020): `rows * cols` codes, then one binary16 scale per row.
pub fn linear_row_scaled(packed: &[u8], input: &[f32], rows_out: usize, cols: usize, bias: Option<&[f32]>) -> Result<Vec<f32>, String> {
    if rows_out.checked_mul(cols).and_then(|n| n.checked_add(rows_out * 2)) != Some(packed.len()) {
        return Err(format!("FP8M weight of {} bytes is not {rows_out} x {cols} codes with a binary16 scale per row", packed.len()));
    }
    drive(Scales::PerRow { rows: rows_out, cols }, packed, input, rows_out, cols, bias)
}

/// An `FP8M` tensor (codes, one binary16 scale per row) as the `FP8B` tensor of tiles one row high and `2^ceil(log2(cols))` wide (one tile per
/// row): the same weights for the block-wise FP8 kernels, with the scales widened to f32 exactly. Lets devices without a Marlin FP8 kernel, or with
/// shapes it cannot take, keep per-row scaled FP8 weights 8-bit. `None` when the width does not fit a tile (above 32768).
pub fn fp8m_as_fp8b(raw: &[u8], rows: usize, cols: usize) -> Option<(u32, Vec<u8>)> {
    let count = rows.checked_mul(cols)?;
    if rows == 0 || cols == 0 || raw.len() != count.checked_add(rows * 2)? {
        return None;
    }
    let kind = fllm_gguf_probe::fp8b_type_id(1, cols.next_power_of_two())?;
    let layout = Fp8bLayout::new(kind, cols, rows, 1)?;
    let mut out = Vec::with_capacity(layout.total_len());
    out.extend_from_slice(&raw[..count]);
    for c in raw[count..].as_chunks::<2>().0 {
        out.extend_from_slice(&half::f16::from_le_bytes(*c).to_f32().to_le_bytes());
    }
    out.extend_from_slice(&layout.trailer());
    debug_assert_eq!(out.len(), layout.total_len());
    Some((kind, out))
}

/// The bytes of an `FP8B` tensor from its parts: `codes` (all matrices), one f32 scale per tile, trailer.
pub fn pack(layout: &Fp8bLayout, codes: &[u8], scales: &[f32]) -> Vec<u8> {
    assert_eq!(codes.len(), layout.elements());
    assert_eq!(scales.len(), layout.tiles());
    let mut out = Vec::with_capacity(layout.total_len());
    out.extend_from_slice(codes);
    for s in scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&layout.trailer());
    out
}

#[cfg(test)]
mod mirror_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{dequantize, e4m3fn_to_f32, fp8b_rows};
    use fllm_gguf_probe::{fp8b_shape, fp8b_type_id, TensorBytes};

    pub(super) struct Lcg(pub(super) u64);
    impl Lcg {
        pub(super) fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        pub(super) fn unit(&mut self) -> f32 {
            self.next() as f32 / (1u32 << 31) as f32 * 2.0 - 1.0
        }
    }

    /// Random finite codes (no NaN) and scales of moderate size.
    pub(super) fn random_tensor(kind: u32, cols: usize, rows: usize, slabs: usize, rng: &mut Lcg) -> (Fp8bLayout, Vec<u8>) {
        let layout = Fp8bLayout::new(kind, cols, rows, slabs).unwrap();
        let codes: Vec<u8> = (0..layout.elements())
            .map(|_| {
                let c = rng.next() as u8;
                if c & 0x7f == 0x7f { c & 0x80 } else { c }
            })
            .collect();
        let scales: Vec<f32> = (0..layout.tiles()).map(|_| 0.002 + 0.004 * (rng.unit() + 1.0)).collect();
        let bytes = pack(&layout, &codes, &scales);
        (layout, bytes)
    }

    /// The textbook evaluation: the dequantized weights of `compute::dequantize` and an f64 dot product. Returns the result and
    /// the sum of the absolute terms (the error scale of the accumulated rounding).
    pub(super) fn reference(kind: u32, bytes: &[u8], layout: &Fp8bLayout, input: &[f32], bias: Option<&[f32]>) -> (Vec<f64>, Vec<f64>) {
        let w = dequantize(bytes, kind, layout.elements()).unwrap();
        let (rows, cols) = (layout.rows, layout.cols);
        let batch = input.len() / cols;
        let (mut want, mut mag) = (vec![0.0; batch * rows], vec![0.0; batch * rows]);
        for b in 0..batch {
            for r in 0..rows {
                let (mut sum, mut m) = (bias.map_or(0.0, |v| f64::from(v[r])), bias.map_or(0.0, |v| f64::from(v[r]).abs()));
                for c in 0..cols {
                    let t = f64::from(input[b * cols + c]) * f64::from(w[r * cols + c]);
                    sum += t;
                    m += t.abs();
                }
                want[b * rows + r] = sum;
                mag[b * rows + r] = m;
            }
        }
        (want, mag)
    }

    #[test]
    fn decoder_matches_the_scalar_reference_for_every_code() {
        for code in 0..=255u8 {
            let (got, want) = (decode_code(code), e4m3fn_to_f32(code));
            if want.is_nan() {
                assert!(got.is_nan(), "code {code:#x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "code {code:#x}");
            }
        }
    }

    #[test]
    fn linear_matches_the_dequantized_reference_for_odd_shapes_and_every_tile_shape() {
        let mut rng = Lcg(7);
        // (tile rows, tile cols, rows, cols): partial tiles on both edges, one-row and one-column tiles, a tile larger than the matrix
        let shapes = [(128, 128, 129, 130), (128, 128, 5, 7), (128, 128, 256, 384), (32, 32, 33, 65), (64, 16, 70, 100), (1, 128, 9, 300), (16, 1, 20, 13), (256, 64, 7, 200)];
        for (br, bc, rows, cols) in shapes {
            let kind = fp8b_type_id(br, bc).unwrap();
            assert_eq!(fp8b_shape(kind), Some((br, bc)));
            let (layout, bytes) = random_tensor(kind, cols, rows, 1, &mut rng);
            for batch in [1usize, 2, 5] {
                let input: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
                let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
                for bias in [None, Some(bias.as_slice())] {
                    let got = linear(kind, &bytes, &input, rows, cols, bias).unwrap();
                    let (want, mag) = reference(kind, &bytes, &layout, &input, bias);
                    assert_eq!(got.len(), want.len());
                    // summation order and fused multiply-add differ from the f64 evaluation: |error| <= gamma(n) * sum |terms|
                    let gamma = (cols + 4) as f64 * f64::from(f32::EPSILON) * 0.5;
                    for i in 0..got.len() {
                        let err = (f64::from(got[i]) - want[i]).abs();
                        assert!(err <= gamma * mag[i] + 1e-30, "tile {br}x{bc}, {rows}x{cols}, batch {batch}, element {i}: {} vs {} (limit {})", got[i], want[i], gamma * mag[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn the_portable_and_the_fused_multiply_add_loops_agree_with_the_reference() {
        // `rows_body` is generic over the FMA flag: x86-64 runs it with FMA inside an AVX2 function when the CPU has the tier, AArch64 with its
        // native fused multiply-add, anything else without. Both instantiations run here as plain code, whatever this CPU is.
        let mut rng = Lcg(9);
        let kind = 0x8000_0021;
        let (rows, cols, batch) = (9usize, 300usize, 2usize);
        let (layout, bytes) = random_tensor(kind, cols, rows, 1, &mut rng);
        let input: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
        let (want, mag) = reference(kind, &bytes, &layout, &input, None);
        for fma in [false, true] {
            let mut buf = vec![0f32; cols];
            let mut tmp = vec![0f32; rows * batch];
            for (task, chunk) in tmp.chunks_mut(batch * ROWS_PER_TASK).enumerate() {
                if fma {
                    rows_body::<true>(Scales::Tiles(&layout), &bytes, &input, batch, task * ROWS_PER_TASK, chunk, &mut buf);
                } else {
                    rows_body::<false>(Scales::Tiles(&layout), &bytes, &input, batch, task * ROWS_PER_TASK, chunk, &mut buf);
                }
            }
            let gamma = (cols + 4) as f64 * f64::from(f32::EPSILON) * 0.5;
            for r in 0..rows {
                for b in 0..batch {
                    let (got, i) = (f64::from(tmp[r * batch + b]), b * rows + r);
                    assert!((got - want[i]).abs() <= gamma * mag[i] + 1e-30, "fma {fma}, row {r}, token {b}: {got} vs {}", want[i]);
                }
            }
        }
    }

    #[test]
    fn row_scaled_fp8m_matches_the_dequantized_reference() {
        let mut rng = Lcg(77);
        for (rows, cols, batch) in [(5usize, 130usize, 1usize), (9, 300, 3), (1, 7, 2), (33, 64, 5)] {
            // codes without the NaN pattern, one binary16 scale per row
            let mut bytes: Vec<u8> = (0..rows * cols)
                .map(|_| {
                    let c = rng.next() as u8;
                    if c & 0x7f == 0x7f { c & 0x80 } else { c }
                })
                .collect();
            for _ in 0..rows {
                bytes.extend_from_slice(&half::f16::from_f32(0.002 + 0.004 * (rng.unit() + 1.0)).to_le_bytes());
            }
            let input: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
            let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
            let w = dequantize(&bytes, 0x8000_0020, rows * cols).unwrap();
            let got = linear_row_scaled(&bytes, &input, rows, cols, Some(&bias)).unwrap();
            let gamma = (cols + 4) as f64 * f64::from(f32::EPSILON) * 0.5;
            for b in 0..batch {
                for r in 0..rows {
                    let (mut sum, mut mag) = (f64::from(bias[r]), f64::from(bias[r]).abs());
                    for c in 0..cols {
                        let t = f64::from(input[b * cols + c]) * f64::from(w[r * cols + c]);
                        sum += t;
                        mag += t.abs();
                    }
                    let err = (f64::from(got[b * rows + r]) - sum).abs();
                    assert!(err <= gamma * mag + 1e-30, "{rows}x{cols} batch {batch} element ({b}, {r}): {} vs {sum}", got[b * rows + r]);
                }
            }
            assert!(linear_row_scaled(&bytes[1..], &input, rows, cols, None).is_err());
        }
    }

    #[test]
    fn fp8m_as_row_tiles_dequantizes_to_the_same_weights_and_runs_the_block_kernel() {
        let mut rng = Lcg(5);
        for (rows, cols) in [(5usize, 130usize), (9, 256), (3, 7)] {
            let mut bytes: Vec<u8> = (0..rows * cols)
                .map(|_| {
                    let c = rng.next() as u8;
                    if c & 0x7f == 0x7f { c & 0x80 } else { c }
                })
                .collect();
            for _ in 0..rows {
                bytes.extend_from_slice(&half::f16::from_f32(0.002 + 0.004 * (rng.unit() + 1.0)).to_le_bytes());
            }
            let (kind, tiles) = fp8m_as_fp8b(&bytes, rows, cols).unwrap();
            assert_eq!(fp8b_shape(kind), Some((1, cols.next_power_of_two())));
            assert_eq!(dequantize(&tiles, kind, rows * cols).unwrap(), dequantize(&bytes, 0x8000_0020, rows * cols).unwrap());
            let input: Vec<f32> = (0..2 * cols).map(|_| rng.unit()).collect();
            let a = linear(kind, &tiles, &input, rows, cols, None).unwrap();
            let b = linear_row_scaled(&bytes, &input, rows, cols, None).unwrap();
            assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{rows}x{cols}");
        }
        assert!(fp8m_as_fp8b(&[0u8; 10], 2, 4).is_none(), "length");
        let wide = vec![0u8; 33_000 * 2 + 4];
        assert!(fp8m_as_fp8b(&wide, 2, 33_000).is_none(), "width above 32768");
    }

    #[test]
    fn linear_is_independent_of_the_thread_count() {
        let mut rng = Lcg(11);
        let kind = 0x8000_0021;
        let (_, bytes) = random_tensor(kind, 300, 70, 1, &mut rng);
        let input: Vec<f32> = (0..3 * 300).map(|_| rng.unit()).collect();
        let first = linear(kind, &bytes, &input, 70, 300, None).unwrap();
        for threads in [1usize, 3] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let again = pool.install(|| linear(kind, &bytes, &input, 70, 300, None).unwrap());
            assert_eq!(first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), again.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn scale_indexing_follows_the_tile_grid() {
        // 3 x 5 matrix, 2 x 2 tiles: tile grid 2 x 3, scale = 1 + tile index, every code is 1.0 (0x38)
        let kind = fp8b_type_id(2, 2).unwrap();
        let layout = Fp8bLayout::new(kind, 5, 3, 1).unwrap();
        assert_eq!((layout.tile_cols(), layout.tiles_per_slab(), layout.tiles()), (3, 6, 6));
        let bytes = pack(&layout, &[0x38; 15], &(0..6).map(|i| 1.0 + i as f32).collect::<Vec<_>>());
        assert_eq!(bytes.len(), layout.total_len());
        let w = dequantize(&bytes, kind, 15).unwrap();
        let want = [1., 1., 2., 2., 3., 1., 1., 2., 2., 3., 4., 4., 5., 5., 6.];
        assert_eq!(w, want);
        for (row, col, tile) in [(0, 0, 0), (1, 3, 1), (2, 4, 5), (2, 1, 3)] {
            assert_eq!(layout.tile_index(0, row, col), tile);
        }
        // stacked matrices: the tiles of the second matrix follow those of the first
        let stack = Fp8bLayout::new(kind, 5, 3, 2).unwrap();
        assert_eq!(stack.tile_index(1, 0, 0), 6);
        assert_eq!(stack.tile_index(1, 2, 4), 11);
        assert_eq!(Fp8bLayout::parse(kind, &bytes).unwrap(), layout);
    }

    #[test]
    fn trailer_is_checked() {
        let kind = 0x8000_0021;
        let mut rng = Lcg(3);
        let (layout, bytes) = random_tensor(kind, 10, 3, 1, &mut rng);
        assert_eq!(Fp8bLayout::parse(kind, &bytes).unwrap(), layout);
        let mut bad = bytes.clone();
        let n = bad.len();
        bad[n - 16] ^= 1;
        assert!(Fp8bLayout::parse(kind, &bad).is_err());
        assert!(Fp8bLayout::parse(kind, &bytes[..n - 1]).is_err());
        assert!(Fp8bLayout::parse(8, &bytes).is_err());
        assert!(linear(kind, &bytes, &[0.0; 10], 4, 10, None).is_err(), "wrong row count");
        // the probe computes the same length from the dimensions as the writer writes
        assert_eq!(Fp8bLayout::from_dims(kind, &[10, 3]).unwrap().total_len(), bytes.len());
    }

    #[test]
    fn stacked_experts_are_cut_into_matrices_and_row_ranges() {
        let mut rng = Lcg(5);
        // 3 experts of 40 x 33 with 16 x 16 tiles
        let kind = fp8b_type_id(16, 16).unwrap();
        let (layout, bytes) = random_tensor(kind, 33, 40, 3, &mut rng);
        let all = dequantize(&bytes, kind, layout.elements()).unwrap();
        let dims = [33u64, 40, 3];
        let whole = TensorBytes::from_vec(bytes);
        for (first, rows) in [(0usize, 40usize), (40, 40), (80, 40), (56, 16), (96, 20), (16, 24)] {
            let part = fp8b_rows(&whole, kind, &dims, first, rows, "test").unwrap();
            let got = dequantize(&part, kind, rows * 33).unwrap();
            assert_eq!(got, all[first * 33..(first + rows) * 33], "rows {first}..{}", first + rows);
        }
        // a cut inside a tile row, or across two matrices, is refused instead of being given wrong scales
        assert!(fp8b_rows(&whole, kind, &dims, 8, 8, "test").is_err());
        assert!(fp8b_rows(&whole, kind, &dims, 32, 16, "test").is_err());
    }
}
