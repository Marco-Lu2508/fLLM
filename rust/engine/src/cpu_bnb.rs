//! bitsandbytes projections on the CPU: NF4 and FP4 (4-bit codes into a 16-entry code book, one `f32` absolute maximum per block)
//! and LLM.int8() (int8 codes, one `f32` scale per row), the fLLM types of `fllm_gguf_probe::bnb`.
//!
//! The weights stay packed. A block of 64 or 128 values is `[f32 scale][codes]`; the activations stay `f32` (nothing is
//! quantized), so a kernel computes exactly `sum(code_book[nibble] * x) * absmax` per block and only the order of the sums differs
//! from the f64 reference ([`reference`]).
//!
//! The AVX2 kernels look the 16 code book entries up in registers: the table is two vectors of eight floats, `vpermps` picks
//! within each by the low three bits of the index and `vblendvps` by the fourth. Code byte `j` holds the elements `2j` (high
//! nibble) and `2j + 1` (low nibble), so eight code bytes give eight even and eight odd elements; the activations are split into an
//! even and an odd plane once per call and the products need no shuffle. A kernel decodes a group of 16 weights once and applies
//! it to up to four tokens, which makes prompts compute-bound instead of decode-bound. Without AVX2 (other CPUs, the scalar tier)
//! the portable kernel does the same with table loads.
//!
//! Test level: the unit tests below compare every kernel with the f64 reference on ragged shapes, and `selftest-cpu` runs the same
//! comparison for each tier (`docs/cpu-kernels.md`).
use crate::cpu_schedule::Rows;
use fllm_gguf_probe::bnb::{self, Family, Layout};

/// Rows of a panel: the scheduler hands out this many rows (all tokens) at a time.
const PANEL_ROWS: usize = 4;

/// Tokens a kernel applies one decoded group to.
const TILE: usize = 4;

/// Whether `kind` is a bitsandbytes type this module computes.
pub(crate) fn handles(kind: u32) -> bool {
    bnb::is_bnb(kind)
}

/// The activations of a call: for the 4-bit types every token split into its even and odd elements (`[even: cols / 2][odd: cols / 2]`,
/// so that a group of eight code bytes meets eight consecutive values of each plane), for int8 the tokens as they are.
struct Acts {
    values: Vec<f32>,
}

impl Acts {
    fn new(x: &[f32], cols: usize, split: bool) -> Self {
        if !split {
            return Self { values: x.to_vec() };
        }
        let half = cols / 2;
        let mut values = vec![0f32; x.len()];
        for (token, out) in x.chunks_exact(cols).zip(values.chunks_exact_mut(cols)) {
            let (even, odd) = out.split_at_mut(half);
            for (j, pair) in token.as_chunks::<2>().0.iter().enumerate() {
                even[j] = pair[0];
                odd[j] = pair[1];
            }
        }
        Self { values }
    }

    fn token(&self, t: usize, cols: usize) -> &[f32] {
        &self.values[t * cols..(t + 1) * cols]
    }
}

/// `y = x * W^T + bias` for a matrix of `rows` rows over `cols` inputs stored as `kind` and `x` of `x.len() / cols` tokens. `None`
/// when `kind` is not a bitsandbytes type or the shape does not fit it (callers then use the generic dequantizing path).
pub(crate) fn linear(weights: &[u8], kind: u32, x: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> Option<Vec<f32>> {
    linear_with(Kernel::best(), weights, kind, x, rows, cols, bias)
}

/// Which implementation runs the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// Portable table-lookup loops.
    Portable,
    /// AVX2 and FMA.
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Kernel {
    /// The widest kernel this CPU and the tier cap allow.
    pub(crate) fn best() -> Self {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu_kernels::isa() != "scalar" && std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return Self::Avx2;
        }
        Self::Portable
    }

    /// Every kernel this CPU can run.
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Self> {
        let mut found = vec![Self::Portable];
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            found.push(Self::Avx2);
        }
        found
    }
}

/// [`linear`] on a chosen kernel.
pub(crate) fn linear_with(kernel: Kernel, weights: &[u8], kind: u32, x: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> Option<Vec<f32>> {
    let layout = bnb::layout(kind)?;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(layout.block) || !x.len().is_multiple_of(cols) || bias.is_some_and(|b| b.len() != rows) {
        return None;
    }
    let row_bytes = cols / layout.block * layout.bytes;
    if weights.len() != rows * row_bytes {
        return None;
    }
    let batch = x.len() / cols;
    let book = *bnb::code_book(layout.family).unwrap_or(&[0.0; 16]);
    // the plane split is for the AVX2 4-bit kernel; the portable kernel reads the tokens as they are
    #[cfg(target_arch = "x86_64")]
    let split = kernel == Kernel::Avx2 && layout.family != Family::Int8;
    #[cfg(not(target_arch = "x86_64"))]
    let split = false;
    let acts = Acts::new(x, cols, split);
    let mut output = vec![vec![0.0f32; batch * rows]];
    crate::cpu_schedule::row_panels(
        &mut output,
        &[rows],
        batch,
        PANEL_ROWS,
        || (),
        |_, _, first, out: &mut Rows<'_>| {
            let valid = (rows - first).min(PANEL_ROWS);
            for r in first..first + valid {
                let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                let mut token = 0;
                while token < batch {
                    let n = (batch - token).min(TILE);
                    let mut sums = [0f32; TILE];
                    let xs: [&[f32]; TILE] = std::array::from_fn(|i| acts.token((token + i).min(batch - 1), cols));
                    run_row(kernel, layout, &book, row, &xs[..n], cols, &mut sums[..n]);
                    for (i, sum) in sums[..n].iter().enumerate() {
                        out.row(token + i)[r - first] = *sum + bias.map_or(0.0, |b| b[r]);
                    }
                    token += n;
                }
            }
        },
    );
    output.pop()
}

/// The dot products of one row with `xs.len()` (1 to 4) tokens.
fn run_row(kernel: Kernel, layout: Layout, book: &[f32; 16], row: &[u8], xs: &[&[f32]], cols: usize, sums: &mut [f32]) {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Kernel::Avx2` is only built after the feature check, `row` holds `cols / block` whole blocks and every token `cols` floats
        // (split into planes for the 4-bit types), which `linear_with` has checked.
        Kernel::Avx2 => unsafe {
            if layout.family == Family::Int8 {
                match xs.len() {
                    1 => avx2::int8::<1>(row, cols, xs, sums),
                    2 => avx2::int8::<2>(row, cols, xs, sums),
                    3 => avx2::int8::<3>(row, cols, xs, sums),
                    _ => avx2::int8::<4>(row, cols, xs, sums),
                }
            } else {
                match xs.len() {
                    1 => avx2::nibbles::<1>(row, layout, cols, book, xs, sums),
                    2 => avx2::nibbles::<2>(row, layout, cols, book, xs, sums),
                    3 => avx2::nibbles::<3>(row, layout, cols, book, xs, sums),
                    _ => avx2::nibbles::<4>(row, layout, cols, book, xs, sums),
                }
            }
        },
        Kernel::Portable => portable(row, layout, cols, book, xs, sums),
    }
}

/// Table-lookup loops for every token, one block at a time (the scale is applied to the block's sum, as the AVX2 kernel does).
fn portable(row: &[u8], layout: Layout, cols: usize, book: &[f32; 16], xs: &[&[f32]], sums: &mut [f32]) {
    for (x, sum) in xs.iter().zip(sums.iter_mut()) {
        let mut total = 0f32;
        for (block, xb) in row.chunks_exact(layout.bytes).zip(x[..cols].chunks_exact(layout.block)) {
            let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let codes = &block[4..];
            let mut lanes = [0f32; 8];
            if layout.family == Family::Int8 {
                for (i, (&q, &v)) in codes.iter().zip(xb).enumerate() {
                    lanes[i % 8] += f32::from(q as i8) * v;
                }
            } else {
                for (i, (&byte, pair)) in codes.iter().zip(xb.as_chunks::<2>().0).enumerate() {
                    lanes[i % 8] += book[(byte >> 4) as usize] * pair[0] + book[(byte & 15) as usize] * pair[1];
                }
            }
            total += lanes.iter().sum::<f32>() * scale;
        }
        *sum = total;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::needless_range_loop)]
mod avx2 {
    use super::Layout;
    use std::arch::x86_64::*;

    /// The 16-entry code book of `idx` (eight 32-bit indexes 0..=15): `vpermps` within each half of the table by the low three bits,
    /// the fourth bit (moved to the sign) chooses the half.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn lookup(low: __m256, high: __m256, idx: __m256i) -> __m256 {
        let from_low = _mm256_permutevar8x32_ps(low, idx);
        let from_high = _mm256_permutevar8x32_ps(high, idx);
        _mm256_blendv_ps(from_low, from_high, _mm256_castsi256_ps(_mm256_slli_epi32::<28>(idx)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sum(v: __m256) -> f32 {
        let pair = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let quad = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
        _mm_cvtss_f32(_mm_add_ss(quad, _mm_shuffle_ps::<1>(quad, quad)))
    }

    /// NF4 / FP4 row against `T` tokens whose activations are split into planes (`[even][odd]`, `cols / 2` floats each).
    ///
    /// # Safety
    /// AVX2 and FMA; `row` is `cols / layout.block` whole blocks, every `xs[t]` has `cols` floats.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn nibbles<const T: usize>(row: &[u8], layout: Layout, cols: usize, book: &[f32; 16], xs: &[&[f32]], sums: &mut [f32]) {
        let low = _mm256_loadu_ps(book.as_ptr());
        let high = _mm256_loadu_ps(book.as_ptr().add(8));
        let mask = _mm256_set1_epi32(15);
        let half = cols / 2;
        let groups = layout.block / 16;
        let mut acc = [_mm256_setzero_ps(); T];
        for b in 0..cols / layout.block {
            let block = row.as_ptr().add(b * layout.bytes);
            let scale = _mm256_set1_ps(f32::from_le_bytes([*block, *block.add(1), *block.add(2), *block.add(3)]));
            let mut part = [_mm256_setzero_ps(); T];
            for g in 0..groups {
                let codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64(block.add(4 + 8 * g).cast()));
                let w_even = lookup(low, high, _mm256_srli_epi32::<4>(codes));
                let w_odd = lookup(low, high, _mm256_and_si256(codes, mask));
                let at = (b * layout.block + 16 * g) / 2;
                for t in 0..T {
                    let plane = xs[t].as_ptr();
                    part[t] = _mm256_fmadd_ps(w_even, _mm256_loadu_ps(plane.add(at)), part[t]);
                    part[t] = _mm256_fmadd_ps(w_odd, _mm256_loadu_ps(plane.add(half + at)), part[t]);
                }
            }
            for t in 0..T {
                acc[t] = _mm256_fmadd_ps(part[t], scale, acc[t]);
            }
        }
        for t in 0..T {
            sums[t] = sum(acc[t]);
        }
    }

    /// int8 row (`[f32 scale][64 x i8]` blocks) against `T` tokens.
    ///
    /// # Safety
    /// AVX2 and FMA; `row` is `cols / 64` whole blocks, every `xs[t]` has `cols` floats.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn int8<const T: usize>(row: &[u8], cols: usize, xs: &[&[f32]], sums: &mut [f32]) {
        let mut acc = [_mm256_setzero_ps(); T];
        for b in 0..cols / 64 {
            let block = row.as_ptr().add(b * 68);
            let scale = _mm256_set1_ps(f32::from_le_bytes([*block, *block.add(1), *block.add(2), *block.add(3)]));
            let mut part = [_mm256_setzero_ps(); T];
            for g in 0..8 {
                let w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(block.add(4 + 8 * g).cast())));
                for t in 0..T {
                    part[t] = _mm256_fmadd_ps(w, _mm256_loadu_ps(xs[t].as_ptr().add(b * 64 + 8 * g)), part[t]);
                }
            }
            for t in 0..T {
                acc[t] = _mm256_fmadd_ps(part[t], scale, acc[t]);
            }
        }
        for t in 0..T {
            sums[t] = sum(acc[t]);
        }
    }
}

/// f64 reference: the products of the dequantized weights (read from the packed blocks through the code book, not through any kernel)
/// with the `f32` activations, and the sum of their absolute values (the error scale). Indexed `token * rows + row`.
pub(crate) fn reference(weights: &[u8], kind: u32, x: &[f32], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let layout = bnb::layout(kind).expect("a bitsandbytes type");
    let book = bnb::code_book(layout.family);
    let row_bytes = cols / layout.block * layout.bytes;
    let batch = x.len() / cols;
    let mut want = vec![0.0; batch * rows];
    let mut scale = vec![0.0; batch * rows];
    for r in 0..rows {
        let mut w = Vec::with_capacity(cols);
        for block in weights[r * row_bytes..(r + 1) * row_bytes].chunks_exact(layout.bytes) {
            let s = f64::from(f32::from_le_bytes([block[0], block[1], block[2], block[3]]));
            for &byte in &block[4..] {
                match book {
                    Some(book) => {
                        w.push(f64::from(book[(byte >> 4) as usize]) * s);
                        w.push(f64::from(book[(byte & 15) as usize]) * s);
                    }
                    None => w.push(f64::from(byte as i8) * s),
                }
            }
        }
        for t in 0..batch {
            let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
            for (&wv, &xv) in w.iter().zip(&x[t * cols..(t + 1) * cols]) {
                let term = wv * f64::from(xv);
                sum += term;
                magnitude += term.abs();
            }
            want[t * rows + r] = sum;
            scale[t * rows + r] = magnitude;
        }
    }
    (want, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare_with, random_weights, Lcg};

    const KINDS: [u32; 5] = [bnb::NF4_B64, bnb::NF4_B128, bnb::FP4_B64, bnb::FP4_B128, bnb::INT8_ROW];

    #[test]
    fn every_kernel_matches_the_reference_on_ragged_shapes() {
        let mut rng = Lcg(0xb5b4_0001);
        let mut checked = 0;
        for kernel in Kernel::available() {
            for kind in KINDS {
                let block = bnb::layout(kind).unwrap().block;
                for blocks in [1, 2, 3, 5, 9] {
                    let cols = block * blocks;
                    for rows in [1, 3, 4, 5, 7, 13] {
                        let weights = random_weights(kind, rows, cols, &mut rng);
                        for batch in [1, 2, 3, 4, 5, 9, 17] {
                            let x = rng.vector(batch * cols, 3.0);
                            let have = linear_with(kernel, &weights, kind, &x, rows, cols, None).expect("shape fits");
                            let (want, scale) = reference(&weights, kind, &x, rows, cols);
                            let (error, ok) = compare_with(&have, &want, &scale, 1e-5);
                            checked += 1;
                            assert!(ok, "{kernel:?} kind {kind:#x} rows {rows} cols {cols} batch {batch}: {error:e}");
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn the_reference_is_the_dequantized_matrix() {
        // `reference` reads the blocks itself; `gguf-probe` dequantizes them: the two must agree on every type
        let mut rng = Lcg(0xb5b4_0002);
        for kind in KINDS {
            let layout = bnb::layout(kind).unwrap();
            let (rows, cols) = (5, 3 * layout.block);
            let weights = random_weights(kind, rows, cols, &mut rng);
            let x = rng.vector(2 * cols, 1.0);
            let dense = bnb::dequantize(&weights, kind, rows * cols).unwrap();
            let (want, scale) = reference(&weights, kind, &x, rows, cols);
            for t in 0..2 {
                for r in 0..rows {
                    // the dequantized matrix rounds every product `code * scale` to f32, the reference does not
                    let sum: f64 = (0..cols).map(|c| f64::from(dense[r * cols + c]) * f64::from(x[t * cols + c])).sum();
                    assert!((sum - want[t * rows + r]).abs() <= 1e-6 * scale[t * rows + r], "kind {kind:#x}");
                }
            }
        }
    }

    #[test]
    fn bias_is_added_and_bad_shapes_are_declined() {
        let mut rng = Lcg(0xb5b4_0003);
        let (rows, cols, batch) = (6, 128, 3);
        let weights = random_weights(bnb::NF4_B64, rows, cols, &mut rng);
        let x = rng.vector(batch * cols, 1.0);
        let bias = rng.vector(rows, 1.0);
        let plain = linear(&weights, bnb::NF4_B64, &x, rows, cols, None).unwrap();
        let biased = linear(&weights, bnb::NF4_B64, &x, rows, cols, Some(&bias)).unwrap();
        for (i, (p, b)) in plain.iter().zip(&biased).enumerate() {
            assert!((p + bias[i % rows] - b).abs() <= 1e-6, "element {i}");
        }
        assert!(linear(&weights, bnb::NF4_B128, &x, rows, cols, None).is_none(), "row bytes do not fit 128-value blocks");
        assert!(linear(&weights, bnb::NF4_B64, &x[1..], rows, cols, None).is_none());
        assert!(linear(&weights, bnb::NF4_B64, &x, rows, cols, Some(&bias[1..])).is_none());
        assert!(linear(&weights, 12, &x, rows, cols, None).is_none());
    }

    /// `cargo test --release -p fllm-engine cpu_bnb -- --ignored --nocapture`: weight bytes read per second at decode and weight
    /// multiply-accumulates per second at a 64-token prompt, per kernel and type, on one thread (`FLLM_BENCH_THREADS` for more).
    #[test]
    #[ignore = "benchmark"]
    fn bench_decode_and_prompt_throughput() {
        let threads = std::env::var("FLLM_BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut rng = Lcg(0xbe7c);
        let (rows, cols) = (2048, 2048);
        for kernel in Kernel::available() {
            for kind in [bnb::NF4_B64, bnb::NF4_B128, bnb::INT8_ROW] {
                let weights = random_weights(kind, rows, cols, &mut rng);
                let bytes = weights.len() as f64;
                for batch in [1usize, 64] {
                    let x = rng.vector(batch * cols, 1.0);
                    pool.install(|| {
                        linear_with(kernel, &weights, kind, &x, rows, cols, None).unwrap();
                        let start = std::time::Instant::now();
                        let rounds = if batch == 1 { 40 } else { 4 };
                        for _ in 0..rounds {
                            std::hint::black_box(linear_with(kernel, &weights, kind, &x, rows, cols, None).unwrap());
                        }
                        let seconds = start.elapsed().as_secs_f64() / rounds as f64;
                        println!(
                            "{kernel:?} {kind:#x} batch {batch:2}: {:7.2} GB/s of weights, {:7.2} GMAC/s",
                            bytes / seconds / 1e9,
                            (rows * cols * batch) as f64 / seconds / 1e9
                        );
                    });
                }
            }
        }
    }
}
