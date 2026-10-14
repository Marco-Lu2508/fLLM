//! Q4Z weights (4-bit codes, one fp16 scale and one 4-bit zero point per group, rows after rows) to the arrays the Marlin GEMM of
//! vLLM expects before its repack kernel: the int4 matrix in the AutoAWQ word order, and the scales and zero points in the
//! permuted order of vLLM's `marlin_permute_scales` / `marlin_zero_points`.

use rayon::prelude::*;

/// Nibble of an AutoAWQ word that holds output column `n % 8`.
const AWQ_SLOT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
/// Column order inside a word of the Marlin zero points.
const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// vLLM `get_scale_perms()[0]`: position `j` of a block of 64 output columns takes column `PERM[j]`.
fn scale_perm() -> [usize; 64] {
    let mut p = [0usize; 64];
    let mut c = 0;
    for i in 0..8 {
        for j in 0..8 {
            p[c] = i + 8 * j;
            c += 1;
        }
    }
    p
}

/// `raw`: `rows` output rows of `cols / group` blocks of `group / 2` code bytes (input `2j` in the low nibble), an fp16 scale and a
/// zero-point byte. Returns `(awq_words [cols, rows / 8], scales [cols / group, rows] as fp16 bits, zeros [cols / group, rows / 8])`.
pub fn marlin_arrays(raw: &[u8], rows: usize, cols: usize, group: usize) -> (Vec<u32>, Vec<u16>, Vec<u32>) {
    let groups = cols / group;
    let block = group / 2 + 3;
    let row_bytes = groups * block;
    let words_per_k = rows / 8;
    let mut awq = vec![0u32; cols * words_per_k];
    awq.par_chunks_mut(64 * words_per_k).enumerate().for_each(|(chunk_index, chunk)| {
        let k0 = chunk_index * 64;
        for n in 0..rows {
            let row = &raw[n * row_bytes..(n + 1) * row_bytes];
            let (word, shift) = (n / 8, 4 * AWQ_SLOT[n % 8]);
            for kk in (0..64).step_by(2) {
                let k = k0 + kk;
                let byte = row[(k / group) * block + (k % group) / 2];
                chunk[kk * words_per_k + word] |= ((byte & 15) as u32) << shift;
                chunk[(kk + 1) * words_per_k + word] |= ((byte >> 4) as u32) << shift;
            }
        }
    });
    let perm = scale_perm();
    let at = |g: usize, n: usize| n * row_bytes + g * block + group / 2;
    let mut scales = vec![0u16; groups * rows];
    let mut plain_zeros = vec![0u8; groups * rows];
    for g in 0..groups {
        for blk in 0..rows / 64 {
            for j in 0..64 {
                let n = blk * 64 + perm[j];
                let i = at(g, n);
                scales[g * rows + blk * 64 + j] = u16::from_le_bytes([raw[i], raw[i + 1]]);
                plain_zeros[g * rows + blk * 64 + j] = raw[i + 2] & 15;
            }
        }
    }
    let mut zeros = vec![0u32; groups * rows / 8];
    for (w, word) in zeros.iter_mut().enumerate() {
        for (i, &src) in INTERLEAVE.iter().enumerate() {
            *word |= (plain_zeros[w * 8 + src] as u32) << (4 * i);
        }
    }
    (awq, scales, zeros)
}

/// Value of an E4M3 byte as an f32 (llama.cpp's NVFP4 scales are unsigned; `0x7f` and `0` give 0).
fn e4m3(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xf) as i32;
    let mantissa = (x & 7) as f32;
    if exp == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exp - 7)
    }
}

/// NVFP4 rows of the GGUF layout (blocks of 64 values: four E4M3 scales, then 32 bytes of E2M1 nibbles) to the inputs of the GPTQ
/// repack kernel of vLLM and the scales Marlin reads: `(words [cols / 8, rows], scales [cols / 16, rows])`. The scales are permuted
/// like `marlin_permute_scales` and then turned into the S0E5M3 form of `nvfp4_marlin_process_scales` (half scales, FP16 activations).
pub fn nvfp4_marlin_arrays(raw: &[u8], rows: usize, cols: usize) -> (Vec<u32>, Vec<u8>) {
    let blocks = cols / 64;
    let row_bytes = blocks * 36;
    let mut words = vec![0u32; cols / 8 * rows];
    words.par_chunks_mut(rows).enumerate().for_each(|(kw, dst)| {
        let (block, within) = (kw / 8, kw % 8);
        let (sub, high) = (within / 2, within % 2 == 1);
        for (n, out) in dst.iter_mut().enumerate() {
            let bytes = &raw[n * row_bytes + block * 36 + 4 + sub * 8..][..8];
            let mut word = 0u32;
            for (i, &b) in bytes.iter().enumerate() {
                word |= ((if high { b >> 4 } else { b & 15 }) as u32) << (4 * i);
            }
            *out = word;
        }
    });
    let groups = cols / 16;
    let perm = scale_perm();
    let mut permuted = vec![0f32; groups * rows];
    for g in 0..groups {
        for blk in 0..rows / 64 {
            for j in 0..64 {
                let n = blk * 64 + perm[j];
                permuted[g * rows + blk * 64 + j] = e4m3(raw[n * row_bytes + (g / 4) * 36 + g % 4]);
            }
        }
    }
    // inside every four consecutive scales the order is 0, 2, 1, 3 (fp8 dequantization layout)
    let mut scales = vec![0u8; groups * rows];
    for (i, quad) in permuted.as_chunks::<4>().0.iter().enumerate() {
        for (j, &src) in [0usize, 2, 1, 3].iter().enumerate() {
            let value = quad[src] * 128.0;
            let bits = if value < 2.0 { 0 } else { half::f16::from_f32(value).to_bits() };
            scales[i * 4 + j] = (bits >> 7) as u8;
        }
    }
    (words, scales)
}

/// Prepare uint32 words in GPTQ layout for FP8 repack:
/// `weight_bytes` is `rows * cols` bytes in row-major order [rows, cols].
/// Returns `[cols / 4, rows]` uint32 words where word `(k_word, n)` contains
/// 4 consecutive FP8 values along the K dimension:
/// `W[n, 4*k_word + 0] | (W[n, 4*k_word + 1] << 8) | (W[n, 4*k_word + 2] << 16) | (W[n, 4*k_word + 3] << 24)`.
pub fn fp8_marlin_words(weight_bytes: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(weight_bytes.len(), rows * cols);
    assert_eq!(cols % 4, 0);
    let k_words = cols / 4;
    let mut words = vec![0u32; k_words * rows];
    for n in 0..rows {
        let row = &weight_bytes[n * cols..(n + 1) * cols];
        for k_word in 0..k_words {
            let b0 = row[k_word * 4] as u32;
            let b1 = row[k_word * 4 + 1] as u32;
            let b2 = row[k_word * 4 + 2] as u32;
            let b3 = row[k_word * 4 + 3] as u32;
            words[k_word * rows + n] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }
    words
}

/// Permute FP16 per-channel scales with `scale_perm_single` and multiply by 256.0 (exponent bias).
pub fn fp8_marlin_scales(raw_scales: &[u8], rows: usize) -> Vec<u16> {
    assert_eq!(raw_scales.len(), rows * 2);
    assert_eq!(rows % 32, 0);
    let mut in_scales = Vec::with_capacity(rows);
    for chunk in raw_scales.as_chunks::<2>().0 {
        let u = u16::from_le_bytes([chunk[0], chunk[1]]);
        in_scales.push(u);
    }
    let perm_single: [usize; 32] = {
        let mut p = [0usize; 32];
        let mut idx = 0;
        for i in 0..4 {
            for &j in &[0, 1, 8, 9, 16, 17, 24, 25] {
                p[idx] = 2 * i + j;
                idx += 1;
            }
        }
        p
    };
    let mut out_scales = vec![0u16; rows];
    for blk in 0..rows / 32 {
        for j in 0..32 {
            let src_idx = blk * 32 + perm_single[j];
            let in_f16 = half::f16::from_bits(in_scales[src_idx]);
            let scaled = half::f16::from_f32(in_f16.to_f32() * 256.0);
            out_scales[blk * 32 + j] = scaled.to_bits();
        }
    }
    out_scales
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(rows: usize, cols: usize, group: usize) -> Vec<u8> {
        let block = group / 2 + 3;
        let mut seed = 12345u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        };
        let mut raw = vec![0u8; rows * cols / group * block];
        for b in raw.chunks_mut(block) {
            for x in b[..group / 2].iter_mut() {
                *x = next();
            }
            b[group / 2..group / 2 + 2].copy_from_slice(&(0x2000u16 + next() as u16).to_le_bytes());
            b[group / 2 + 2] = next() & 15;
        }
        raw
    }

    #[test]
    fn awq_words_hold_every_code_in_the_autoawq_order() {
        let (rows, cols, group) = (128, 128, 64);
        let raw = sample(rows, cols, group);
        let (awq, _, _) = marlin_arrays(&raw, rows, cols, group);
        let block = group / 2 + 3;
        for n in (0..rows).step_by(5) {
            for k in 0..cols {
                let byte = raw[n * (cols / group) * block + (k / group) * block + (k % group) / 2];
                let want = if k % 2 == 0 { byte & 15 } else { byte >> 4 } as u32;
                let word = awq[k * (rows / 8) + n / 8];
                assert_eq!((word >> (4 * AWQ_SLOT[n % 8])) & 15, want, "n {n} k {k}");
            }
        }
    }

    #[test]
    fn scales_and_zero_points_are_permutations_of_the_inputs() {
        let (rows, cols, group) = (128, 256, 128);
        let raw = sample(rows, cols, group);
        let (_, scales, zeros) = marlin_arrays(&raw, rows, cols, group);
        let block = group / 2 + 3;
        let mut want: Vec<u16> = (0..cols / group)
            .flat_map(|g| (0..rows).map(move |n| (g, n)))
            .map(|(g, n)| {
                let i = n * (cols / group) * block + g * block + group / 2;
                u16::from_le_bytes([raw[i], raw[i + 1]])
            })
            .collect();
        let mut got = scales.clone();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(want, got);
        let nibbles = |w: &[u32]| {
            let mut v: Vec<u8> = w.iter().flat_map(|x| (0..8).map(move |i| ((x >> (4 * i)) & 15) as u8)).collect();
            v.sort_unstable();
            v
        };
        let plain: Vec<u32> = (0..cols / group * rows / 8)
            .map(|w| {
                (0..8).fold(0u32, |acc, i| {
                    let (g, n) = ((w * 8 + i) / rows, (w * 8 + i) % rows);
                    acc | ((raw[n * (cols / group) * block + g * block + group / 2 + 2] as u32 & 15) << (4 * i))
                })
            })
            .collect();
        assert_eq!(nibbles(&plain), nibbles(&zeros));
    }
}
