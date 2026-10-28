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
        #[allow(unused_mut)] // (only x86-64 has a second kernel)
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

    /// The older route expanded the blocks to F16 weights: the native kernels (exact products, f32 sums) differ from it by the F16 rounding of
    /// every weight, a relative 2^-11 = 4.9e-4 at most, so at most that fraction of the sum of the absolute terms.
    #[test]
    fn the_kernels_agree_with_the_expanded_f16_weights_within_the_f16_rounding() {
        let mut rng = Lcg(0xb5b4_0004);
        for kind in KINDS {
            let layout = bnb::layout(kind).unwrap();
            let (rows, cols, batch) = (11, 7 * layout.block, 4);
            let weights = random_weights(kind, rows, cols, &mut rng);
            let x = rng.vector(batch * cols, 2.0);
            let have = linear(&weights, kind, &x, rows, cols, None).unwrap();
            let expanded: Vec<f32> = bnb::dequantize(&weights, kind, rows * cols).unwrap().into_iter().map(|v| half::f16::from_f32(v).to_f32()).collect();
            for t in 0..batch {
                for r in 0..rows {
                    let (mut old, mut magnitude) = (0.0f64, 0.0f64);
                    for c in 0..cols {
                        let term = f64::from(expanded[r * cols + c]) * f64::from(x[t * cols + c]);
                        old += term;
                        magnitude += term.abs();
                    }
                    let error = (f64::from(have[t * rows + r]) - old).abs();
                    assert!(error <= 5e-4 * magnitude + 1e-6, "kind {kind:#x} token {t} row {r}: native {} expanded {old}", have[t * rows + r]);
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

    // ------------------------------------------------------------------------------------------------------------------
    // Host mirrors of the CUDA kernels (cuda/common.cuh decode16, cuda/bnb.cuh): the same index arithmetic, the same order of the
    // float operations, run on the host so that a layout or indexing mistake in the device code shows up without a GPU.
    // ------------------------------------------------------------------------------------------------------------------

    /// `bnb_select16` of common.cuh: a binary select tree over the four index bits.
    fn select16(q: u32, v: &[f32; 16]) -> f32 {
        let (b0, b1, b2, b3) = (q & 1 != 0, q & 2 != 0, q & 4 != 0, q & 8 != 0);
        let a: Vec<f32> = (0..8).map(|i| if b0 { v[2 * i + 1] } else { v[2 * i] }).collect();
        let c: Vec<f32> = (0..4).map(|i| if b1 { a[2 * i + 1] } else { a[2 * i] }).collect();
        let d: Vec<f32> = (0..2).map(|i| if b2 { c[2 * i + 1] } else { c[2 * i] }).collect();
        if b3 {
            d[1]
        } else {
            d[0]
        }
    }

    /// The literals of the two `bnb_select16(q, ...)` calls in `bnb_code` of common.cuh: (FP4, NF4).
    fn cuda_code_books() -> (Vec<f32>, Vec<f32>) {
        let source = include_str!("../cuda/common.cuh");
        let mut books = Vec::new();
        let mut rest = source;
        while let Some(at) = rest.find("return bnb_select16(q, ") {
            let tail = &rest[at + "return bnb_select16(q, ".len()..];
            let end = tail.find(");").expect("end of the call");
            let values: Vec<f32> = tail[..end].split(',').map(|v| v.trim().trim_end_matches('f').parse::<f32>().unwrap_or_else(|e| panic!("{v:?}: {e}"))).collect();
            books.push(values);
            rest = &tail[end..];
        }
        assert_eq!(books.len(), 2, "bnb_code has an FP4 and an NF4 call");
        let nf4 = books.pop().unwrap();
        let fp4 = books.pop().unwrap();
        (fp4, nf4)
    }

    #[test]
    fn the_cuda_code_books_are_the_rust_ones_bit_for_bit() {
        let (fp4, nf4) = cuda_code_books();
        for (name, cuda, rust) in [("fp4", fp4, bnb::FP4_CODE), ("nf4", nf4, bnb::NF4_CODE)] {
            assert_eq!(cuda.len(), 16, "{name}");
            for i in 0..16 {
                assert_eq!(cuda[i].to_bits(), rust[i].to_bits(), "{name} entry {i}: cuda {} rust {}", cuda[i], rust[i]);
            }
        }
    }

    #[test]
    fn the_cuda_kind_ids_are_the_rust_type_ids() {
        let source = include_str!("../cuda/common.cuh");
        for (name, id) in [
            ("KIND_NF4B64", bnb::NF4_B64),
            ("KIND_NF4B128", bnb::NF4_B128),
            ("KIND_FP4B64", bnb::FP4_B64),
            ("KIND_FP4B128", bnb::FP4_B128),
            ("KIND_INT8R", bnb::INT8_ROW),
        ] {
            let at = source.find(&format!("{name} = 0x")).unwrap_or_else(|| panic!("{name} is not defined in common.cuh"));
            let digits: String = source[at + name.len() + 5..].chars().take_while(char::is_ascii_hexdigit).collect();
            assert_eq!(u32::from_str_radix(&digits, 16).unwrap(), id, "{name}");
        }
        // block sizes of the CUDA side (kind_block_elems / kind_block_bytes) are the layouts of the Rust side
        assert!(source.contains("case KIND_NF4B64: case KIND_FP4B64: return 36;"));
        assert!(source.contains("case KIND_NF4B128: case KIND_FP4B128: case KIND_INT8R: return 68;"));
        for (id, _, block, bytes) in bnb::TYPES {
            assert_eq!(bytes, if id == bnb::INT8_ROW { 4 + block } else { 4 + block / 2 }, "{id:#x}");
        }
    }

    #[test]
    fn the_select_tree_picks_every_entry() {
        let book: [f32; 16] = std::array::from_fn(|i| i as f32 * 3.0 + 1.0);
        for q in 0..16u32 {
            assert_eq!(select16(q, &book), book[q as usize], "q {q}");
        }
    }

    /// `decode16<KIND>` of common.cuh for a bitsandbytes type: 16 values of run `run` of a row (the select tree, the scale multiplied into every value).
    fn decode16_mirror(kind: u32, row: &[u8], run: usize) -> [f32; 16] {
        let layout = bnb::layout(kind).unwrap();
        let (block, bytes) = (layout.block, layout.bytes);
        let e0 = run * 16;
        let b = &row[e0 / block * bytes..];
        let scale = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let mut w = [0f32; 16];
        match layout.family {
            Family::Int8 => {
                let q = &b[4 + e0 % 64..];
                for j in 0..4 {
                    let word = u32::from_le_bytes([q[4 * j], q[4 * j + 1], q[4 * j + 2], q[4 * j + 3]]);
                    for i in 0..4 {
                        w[4 * j + i] = f32::from((word >> (8 * i)) as u8 as i8) * scale;
                    }
                }
            }
            family => {
                let book = bnb::code_book(family).unwrap();
                let q = &b[4 + e0 % block / 2..];
                let words = [u32::from_le_bytes([q[0], q[1], q[2], q[3]]), u32::from_le_bytes([q[4], q[5], q[6], q[7]])];
                for j in 0..8 {
                    let byte = (words[j >> 2] >> (8 * (j & 3))) & 0xff;
                    w[2 * j] = select16(byte >> 4, book) * scale;
                    w[2 * j + 1] = select16(byte & 15, book) * scale;
                }
            }
        }
        w
    }

    /// `bnb4_gemv_kernel` / `bnb8_gemv_kernel` of bnb.cuh: one warp (32 lanes) per row, lane `l` takes runs `l, l + 32, ...`, the byte-pair table
    /// gives two values per code byte, a run's partial sum is multiplied by the block scale, the warp sums with an xor butterfly. Returns `Y`
    /// (`batch x rows`), passes of four tokens as `fllm_bnb_gemv_try` launches them.
    fn gemv_mirror(kind: u32, weights: &[u8], x: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> Vec<f32> {
        let layout = bnb::layout(kind).unwrap();
        let batch = x.len() / cols;
        let row_bytes = cols / layout.block * layout.bytes;
        let mut y = vec![0f32; batch * rows];
        let pairs: Vec<(f32, f32)> = match bnb::code_book(layout.family) {
            Some(book) => (0..256usize).map(|e| (select16((e >> 4) as u32, book), select16((e & 15) as u32, book))).collect(),
            None => Vec::new(),
        };
        for b0 in (0..batch).step_by(4) {
            let nb = (batch - b0).min(4);
            for row in 0..rows {
                let rp = &weights[row * row_bytes..(row + 1) * row_bytes];
                let mut acc = vec![[0f32; 32]; nb];
                for lane in 0..32 {
                    for run in (lane..cols / 16).step_by(32) {
                        let e0 = run * 16;
                        let blk = &rp[e0 / layout.block * layout.bytes..];
                        let scale = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                        let mut w = [0f32; 16];
                        if layout.family == Family::Int8 {
                            let q = &blk[4 + e0 % 64..];
                            for j in 0..4 {
                                let word = u32::from_le_bytes([q[4 * j], q[4 * j + 1], q[4 * j + 2], q[4 * j + 3]]);
                                for i in 0..4 {
                                    w[4 * j + i] = f32::from((word >> (8 * i)) as u8 as i8);
                                }
                            }
                        } else {
                            let q = &blk[4 + e0 % layout.block / 2..];
                            let words = [u32::from_le_bytes([q[0], q[1], q[2], q[3]]), u32::from_le_bytes([q[4], q[5], q[6], q[7]])];
                            for j in 0..8 {
                                let p = pairs[((words[j >> 2] >> (8 * (j & 3))) & 0xff) as usize];
                                w[2 * j] = p.0;
                                w[2 * j + 1] = p.1;
                            }
                        }
                        for (b, lanes) in acc.iter_mut().enumerate() {
                            let xs = &x[(b0 + b) * cols + e0..][..16];
                            let mut s = 0f32;
                            for i in 0..16 {
                                s = w[i].mul_add(xs[i], s);
                            }
                            lanes[lane] = s.mul_add(scale, lanes[lane]);
                        }
                    }
                }
                for (b, lanes) in acc.iter().enumerate() {
                    let mut v = *lanes;
                    for offset in [16, 8, 4, 2, 1] {
                        let old = v;
                        for (i, slot) in v.iter_mut().enumerate() {
                            *slot = old[i] + old[i ^ offset];
                        }
                    }
                    y[(b0 + b) * rows + row] = v[0] + bias.map_or(0.0, |b| b[row]);
                }
            }
        }
        y
    }

    #[test]
    fn the_cuda_gemv_arithmetic_matches_the_reference() {
        let mut rng = Lcg(0xc0da_0001);
        for kind in KINDS {
            let block = bnb::layout(kind).unwrap().block;
            for blocks in [1usize, 2, 3, 5, 9, 17] {
                let cols = blocks * block;
                for (rows, batch) in [(1usize, 1usize), (5, 3), (13, 5), (4, 8), (7, 2)] {
                    let weights = random_weights(kind, rows, cols, &mut rng);
                    let x = rng.vector(batch * cols, 3.0);
                    let bias = rng.vector(rows, 1.0);
                    let have = gemv_mirror(kind, &weights, &x, rows, cols, Some(&bias));
                    let (mut want, mut scale) = reference(&weights, kind, &x, rows, cols);
                    for t in 0..batch {
                        for r in 0..rows {
                            want[t * rows + r] += f64::from(bias[r]);
                            scale[t * rows + r] += f64::from(bias[r]).abs();
                        }
                    }
                    let (error, ok) = compare_with(&have, &want, &scale, 1e-5);
                    assert!(ok, "kind {kind:#x} rows {rows} cols {cols} batch {batch}: {error:e}");
                }
            }
        }
    }

    #[test]
    fn the_cuda_run_decoder_matches_the_dequantizer_for_every_run() {
        // decode16 (tensor-core GEMM tiles, generic kernels, weight_at): every run of 16 values of every block, bit for bit
        let mut rng = Lcg(0xc0da_0002);
        for kind in KINDS {
            let layout = bnb::layout(kind).unwrap();
            let (rows, cols) = (3, 4 * layout.block);
            let weights = random_weights(kind, rows, cols, &mut rng);
            let dense = bnb::dequantize(&weights, kind, rows * cols).unwrap();
            let row_bytes = cols / layout.block * layout.bytes;
            for r in 0..rows {
                for run in 0..cols / 16 {
                    let w = decode16_mirror(kind, &weights[r * row_bytes..(r + 1) * row_bytes], run);
                    for (i, v) in w.iter().enumerate() {
                        assert_eq!(v.to_bits(), dense[r * cols + run * 16 + i].to_bits(), "kind {kind:#x} row {r} run {run} element {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_tensor_core_tiles_round_to_f16_within_the_stated_tolerance() {
        // tc_gemm.cu decodes a weight run to f32 and stores it as f16 (rounding to nearest even), the activations are f16 too: each product
        // carries a relative error of at most 2 * 2^-11 (+ the f16 range); compare with the f32 reference on the scale of the absolute terms
        let mut rng = Lcg(0xc0da_0003);
        for kind in KINDS {
            let layout = bnb::layout(kind).unwrap();
            let (rows, cols, batch) = (9, 6 * layout.block, 5);
            let weights = random_weights(kind, rows, cols, &mut rng);
            let x = rng.vector(batch * cols, 2.0);
            let row_bytes = cols / layout.block * layout.bytes;
            let mut have = vec![0f64; batch * rows];
            for r in 0..rows {
                let mut wh = Vec::with_capacity(cols);
                for run in 0..cols / 16 {
                    wh.extend(decode16_mirror(kind, &weights[r * row_bytes..(r + 1) * row_bytes], run).map(|v| half::f16::from_f32(v).to_f32()));
                }
                for t in 0..batch {
                    have[t * rows + r] = wh.iter().zip(&x[t * cols..(t + 1) * cols]).map(|(w, v)| f64::from(*w) * f64::from(half::f16::from_f32(*v).to_f32())).sum();
                }
            }
            let (want, scale) = reference(&weights, kind, &x, rows, cols);
            for i in 0..have.len() {
                assert!((have[i] - want[i]).abs() <= 1.5e-3 * scale[i] + 1e-6, "kind {kind:#x} output {i}: {} vs {}", have[i], want[i]);
            }
        }
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
