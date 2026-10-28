//! MLX affine group-quantized weights (`MLXA` tensor types, `fllm_gguf_probe::MlxAffine`) on the CPU.
//!
//! A row is `cols / group` blocks; a block is the scale and the bias of one group as two 16-bit floats (f16 or bf16, the type says which)
//! followed by `group * bits / 8` bytes of codes in the MLX order: a little-endian bit stream, code `i` of the group at bit `i * bits`,
//! so that 32 codes are exactly `bits` 32-bit words and a code of 3, 5 or 6 bits may straddle two words. The weight is
//! `w = code * scale + bias`.
//!
//! This is the reference of every other implementation (the CUDA kernels of `cuda/mlx_affine.cu` are checked against it by the
//! device self test) and, by itself, the CPU kernel: [`dequantize_row`] unpacks 32 codes at a time with the bit width as a constant
//! (the shifts and the straddling reads then fold at compile time), the rows of a tile are decoded once and dotted with every token.
//! The activations stay f32; the products are summed in 8 lanes, so a result differs from a strictly sequential sum of the
//! dequantized weights only by the order of the f32 additions.
use fllm_gguf_probe::MlxAffine;
use rayon::prelude::*;

/// Rows decoded together; every token's activations are read once per tile.
const ROW_TILE: usize = 4;

pub(crate) fn handles(kind: u32) -> bool {
    fllm_gguf_probe::mlx_affine(kind).is_some()
}

#[inline(always)]
fn half_to_f32(bytes: [u8; 2], bf16: bool) -> f32 {
    let raw = u16::from_le_bytes(bytes);
    if bf16 {
        f32::from_bits(u32::from(raw) << 16)
    } else {
        half::f16::from_bits(raw).to_f32()
    }
}

/// The 32 codes in the `BITS` words at `src` (`BITS * 4` bytes), as floats.
#[inline(always)]
fn unpack32<const BITS: usize>(src: &[u8], out: &mut [f32]) {
    // one spare word: the straddling read below is only taken when the code really crosses into the next word
    let mut words = [0u32; 9];
    for (k, w) in words.iter_mut().take(BITS).enumerate() {
        *w = u32::from_le_bytes([src[4 * k], src[4 * k + 1], src[4 * k + 2], src[4 * k + 3]]);
    }
    let mask = (1u32 << BITS) - 1;
    for (i, slot) in out.iter_mut().take(32).enumerate() {
        let (word, shift) = (i * BITS / 32, (i * BITS) % 32);
        let mut v = words[word] >> shift;
        if shift + BITS > 32 {
            v |= words[word + 1] << (32 - shift);
        }
        *slot = (v & mask) as f32;
    }
}

#[inline(always)]
fn row_impl<const BITS: usize>(row: &[u8], group: usize, bf16: bool, out: &mut [f32]) {
    let block = group * BITS / 8 + 4;
    for (b, dst) in row.chunks_exact(block).zip(out.chunks_exact_mut(group)) {
        let scale = half_to_f32([b[0], b[1]], bf16);
        let bias = half_to_f32([b[2], b[3]], bf16);
        for (codes, values) in b[4..].chunks_exact(BITS * 4).zip(dst.chunks_exact_mut(32)) {
            unpack32::<BITS>(codes, values);
            for v in values.iter_mut() {
                *v = *v * scale + bias;
            }
        }
    }
}

/// Dequantize one row of `row.len() / block_bytes * group` values into `out` (exactly that long).
pub(crate) fn dequantize_row(row: &[u8], m: MlxAffine, out: &mut [f32]) {
    let (group, bf16) = (m.group as usize, m.bf16);
    match m.bits {
        2 => row_impl::<2>(row, group, bf16, out),
        3 => row_impl::<3>(row, group, bf16, out),
        4 => row_impl::<4>(row, group, bf16, out),
        5 => row_impl::<5>(row, group, bf16, out),
        6 => row_impl::<6>(row, group, bf16, out),
        8 => row_impl::<8>(row, group, bf16, out),
        other => unreachable!("MlxAffine::bits {other} is validated by fllm_gguf_probe::mlx_affine"),
    }
}

/// `count` values of a tensor made of whole rows or of any whole number of groups.
pub(crate) fn dequantize(bytes: &[u8], kind: u32, count: usize) -> Result<Vec<f32>, String> {
    let m = fllm_gguf_probe::mlx_affine(kind).ok_or_else(|| format!("{kind:#x} is not an MLX affine type"))?;
    let group = m.group as usize;
    if !count.is_multiple_of(group) || count / group * m.block_bytes() as usize != bytes.len() {
        return Err(format!("invalid MLX tensor size: {} bytes for {count} values in groups of {group}", bytes.len()));
    }
    let mut out = vec![0f32; count];
    dequantize_row(bytes, m, &mut out);
    Ok(out)
}

#[inline(always)]
fn dot(w: &[f32], x: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    for (w, x) in w.chunks_exact(8).zip(x.chunks_exact(8)) {
        for ((lane, &a), &b) in lanes.iter_mut().zip(w).zip(x) {
            *lane += a * b;
        }
    }
    let tail = w.len() - w.len() % 8;
    let mut sum = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    for (&a, &b) in w[tail..].iter().zip(&x[tail..]) {
        sum += a * b;
    }
    sum
}

/// One tile of up to `ROW_TILE` rows: decode them into `scratch`, then dot every token: `out[row * batch + token]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_impl(rows: &[u8], m: MlxAffine, cols: usize, input: &[f32], batch: usize, bias: Option<&[f32]>, first_row: usize, scratch: &mut [f32], out: &mut [f32]) {
    let row_bytes = cols / m.group as usize * m.block_bytes() as usize;
    let n = out.len() / batch;
    for r in 0..n {
        dequantize_row(&rows[r * row_bytes..(r + 1) * row_bytes], m, &mut scratch[r * cols..(r + 1) * cols]);
    }
    for t in 0..batch {
        let x = &input[t * cols..(t + 1) * cols];
        for r in 0..n {
            out[r * batch + t] = dot(&scratch[r * cols..(r + 1) * cols], x) + bias.map_or(0.0, |b| b[first_row + r]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_avx2(rows: &[u8], m: MlxAffine, cols: usize, input: &[f32], batch: usize, bias: Option<&[f32]>, first_row: usize, scratch: &mut [f32], out: &mut [f32]) {
    tile_impl(rows, m, cols, input, batch, bias, first_row, scratch, out)
}

/// `out[token][row] = sum_k input[token][k] * w[row][k] + bias[row]` for a weight of `rows x cols` in `MLXA` blocks, or `None` when
/// `kind` is no MLX type or the sizes do not fit (the caller then reports the error through the generic path).
pub(crate) fn linear(packed: &[u8], kind: u32, input: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> Option<Vec<f32>> {
    let m = fllm_gguf_probe::mlx_affine(kind)?;
    let group = m.group as usize;
    if cols == 0 || !cols.is_multiple_of(group) || !input.len().is_multiple_of(cols) || bias.is_some_and(|b| b.len() != rows) {
        return None;
    }
    let row_bytes = cols / group * m.block_bytes() as usize;
    if packed.len() != rows * row_bytes {
        return None;
    }
    let batch = input.len() / cols;
    if batch == 0 || rows == 0 {
        return Some(Vec::new());
    }
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    // [row][token] first, so that tiles own disjoint slices; transposed below when there is more than one token
    let mut by_row = vec![0f32; rows * batch];
    by_row.par_chunks_mut(ROW_TILE * batch).enumerate().for_each_init(
        || vec![0f32; ROW_TILE * cols],
        |scratch, (tile, out)| {
            let first_row = tile * ROW_TILE;
            let n = out.len() / batch;
            let tile_rows = &packed[first_row * row_bytes..(first_row + n) * row_bytes];
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: the features were detected above; the slices are sized by the checks of `linear`.
                unsafe { tile_avx2(tile_rows, m, cols, input, batch, bias, first_row, scratch, out) };
                return;
            }
            tile_impl(tile_rows, m, cols, input, batch, bias, first_row, scratch, out);
        },
    );
    if batch == 1 {
        return Some(by_row);
    }
    let mut out = vec![0f32; rows * batch];
    for (r, values) in by_row.chunks_exact(batch).enumerate() {
        for (t, &v) in values.iter().enumerate() {
            out[t * rows + r] = v;
        }
    }
    Some(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic generator for the tests.
    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        pub(crate) fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        pub(crate) fn unit(&mut self) -> f32 {
            self.next() as f32 / (1u32 << 30) as f32 - 1.0
        }
    }

    /// Pack `codes` the way MLX does: code `i` at bit `i * bits` of a little-endian bit stream (written bit by bit, so that it shares no
    /// shifting logic with the kernels).
    pub(crate) fn pack(codes: &[u32], bits: usize) -> Vec<u8> {
        let mut out = vec![0u8; (codes.len() * bits).div_ceil(8)];
        for (i, &c) in codes.iter().enumerate() {
            for b in 0..bits {
                if c >> b & 1 != 0 {
                    let at = i * bits + b;
                    out[at / 8] |= 1 << (at % 8);
                }
            }
        }
        out
    }

    /// A random `rows x cols` weight in `MLXA` blocks: `(bytes, codes, scales, biases)`; scale and bias values are exact in the 16-bit format.
    pub(crate) fn random_weight(rng: &mut Lcg, m: MlxAffine, rows: usize, cols: usize) -> (Vec<u8>, Vec<u32>, Vec<f32>, Vec<f32>) {
        let (bits, group) = (m.bits as usize, m.group as usize);
        let round = |v: f32| if m.bf16 { half::bf16::from_f32(v).to_f32() } else { half::f16::from_f32(v).to_f32() };
        let codes: Vec<u32> = (0..rows * cols).map(|_| rng.next() & ((1 << bits) - 1)).collect();
        let groups = rows * cols / group;
        let scales: Vec<f32> = (0..groups).map(|_| round(0.01 + 0.02 * rng.unit().abs())).collect();
        let biases: Vec<f32> = (0..groups).map(|_| round(0.1 * rng.unit())).collect();
        let mut bytes = Vec::new();
        for g in 0..groups {
            for v in [scales[g], biases[g]] {
                let raw = if m.bf16 { half::bf16::from_f32(v).to_bits() } else { half::f16::from_f32(v).to_bits() };
                bytes.extend_from_slice(&raw.to_le_bytes());
            }
            bytes.extend(pack(&codes[g * group..(g + 1) * group], bits));
        }
        (bytes, codes, scales, biases)
    }

    pub(crate) const ALL_BITS: [u32; 6] = [2, 3, 4, 5, 6, 8];

    #[test]
    fn every_code_of_every_bit_width_unpacks_including_the_ones_that_straddle_words() {
        let mut rng = Lcg(7);
        for bits in ALL_BITS {
            for group in [32u32, 64, 96, 128] {
                for bf16 in [false, true] {
                    let m = MlxAffine { bits, group, bf16 };
                    let (rows, cols) = (2, group as usize * 2);
                    let (bytes, codes, scales, biases) = random_weight(&mut rng, m, rows, cols);
                    let got = dequantize(&bytes, m.type_id(), rows * cols).unwrap();
                    for (e, &v) in got.iter().enumerate() {
                        let g = e / group as usize;
                        assert_eq!(v.to_bits(), (codes[e] as f32 * scales[g] + biases[g]).to_bits(), "{bits} bits, group {group}, element {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn straddling_codes_are_read_from_both_words() {
        // 3 bits: code 10 sits at bits 30..33, its low two bits in word 0 and its high bit in word 1
        let mut words = [0u32; 3];
        words[0] |= 0b01 << 30;
        words[1] |= 0b1;
        let mut block = vec![0, 0, 0, 0];
        block.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        let m = MlxAffine { bits: 3, group: 32, bf16: false };
        // scale 1, bias 0 (f16 1.0 = 0x3c00)
        block[0] = 0x00;
        block[1] = 0x3c;
        let v = dequantize(&block, m.type_id(), 32).unwrap();
        assert_eq!(v[10], 5.0);
        assert!(v.iter().enumerate().all(|(i, &x)| (i == 10) == (x != 0.0)), "the neighbours are not disturbed");
        // 5 bits: code 6 at bits 30..35 (word 0 / word 1), code 12 at bits 60..65 (word 1 / word 2)
        let mut words = [0u32; 5];
        words[0] |= 0b11 << 30;
        words[1] |= 0b101;
        words[1] |= 0b1111 << 28;
        words[2] |= 0b1;
        let mut block = vec![0x00, 0x3c, 0, 0];
        block.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        let m = MlxAffine { bits: 5, group: 32, bf16: false };
        let v = dequantize(&block, m.type_id(), 32).unwrap();
        assert_eq!(v[6], (0b101 << 2 | 0b11) as f32);
        assert_eq!(v[12], (0b1 << 4 | 0b1111) as f32);
        // 6 bits: code 5 at bits 30..36
        let mut words = [0u32; 6];
        words[0] |= 0b10 << 30;
        words[1] |= 0b1011;
        let mut block = vec![0x00, 0x3c, 0, 0];
        block.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        let m = MlxAffine { bits: 6, group: 32, bf16: false };
        let v = dequantize(&block, m.type_id(), 32).unwrap();
        assert_eq!(v[5], (0b1011 << 2 | 0b10) as f32);
    }

    /// `y[token][row]` in f64 from the dequantized weights.
    fn reference(weights: &[f32], input: &[f32], rows: usize, cols: usize, bias: Option<&[f32]>) -> (Vec<f64>, Vec<f64>) {
        let batch = input.len() / cols;
        let mut y = vec![0f64; batch * rows];
        let mut magnitude = vec![0f64; batch * rows];
        for t in 0..batch {
            for r in 0..rows {
                let (mut s, mut a) = (0f64, 0f64);
                for k in 0..cols {
                    let p = weights[r * cols + k] as f64 * input[t * cols + k] as f64;
                    s += p;
                    a += p.abs();
                }
                y[t * rows + r] = s + bias.map_or(0.0, |b| b[r] as f64);
                magnitude[t * rows + r] = a + bias.map_or(0.0, |b| b[r].abs() as f64);
            }
        }
        (y, magnitude)
    }

    #[test]
    fn linear_matches_the_dequantized_weights_within_f32_summation_error() {
        let mut rng = Lcg(11);
        for bits in ALL_BITS {
            for (group, cols) in [(32u32, 64usize), (64, 192), (128, 384), (32, 224)] {
                for (rows, batch, bf16) in [(1usize, 1usize, false), (5, 1, true), (7, 3, false), (9, 17, true)] {
                    let m = MlxAffine { bits, group, bf16 };
                    let (bytes, ..) = random_weight(&mut rng, m, rows, cols);
                    let input: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
                    let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
                    let weights = dequantize(&bytes, m.type_id(), rows * cols).unwrap();
                    for use_bias in [false, true] {
                        let b = use_bias.then_some(bias.as_slice());
                        let got = linear(&bytes, m.type_id(), &input, rows, cols, b).expect("supported shape");
                        let (want, magnitude) = reference(&weights, &input, rows, cols, b);
                        assert_eq!(got.len(), want.len());
                        for i in 0..got.len() {
                            // f32 summation of `cols` products: the error stays below cols * 2^-24 of the sum of magnitudes
                            let tolerance = 1e-5 * magnitude[i] + 1e-6;
                            assert!((got[i] as f64 - want[i]).abs() <= tolerance, "{bits} bits, group {group}, {rows}x{cols}x{batch}: {} vs {} (tolerance {tolerance})", got[i], want[i]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn linear_declines_what_does_not_fit() {
        let m = MlxAffine { bits: 4, group: 64, bf16: false };
        let mut rng = Lcg(3);
        let (bytes, ..) = random_weight(&mut rng, m, 2, 64);
        let x = vec![1.0f32; 64];
        assert!(linear(&bytes, m.type_id(), &x, 2, 64, None).is_some());
        assert!(linear(&bytes[1..], m.type_id(), &x, 2, 64, None).is_none(), "short weight");
        assert!(linear(&bytes, m.type_id(), &x[..63], 2, 64, None).is_none(), "ragged input");
        assert!(linear(&bytes, m.type_id(), &x, 2, 64, Some(&[0.0])).is_none(), "bias length");
        assert!(linear(&bytes, 12, &x, 2, 64, None).is_none(), "not an MLX type");
        assert!(dequantize(&bytes, m.type_id(), 100).is_err());
    }
}
