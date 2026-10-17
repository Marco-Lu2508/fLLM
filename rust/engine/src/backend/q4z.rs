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
/// Nibble of a packed Q4Z code word that holds offset `t` (`0..8`) of its 8-input group.
#[inline]
pub const fn q4z_nib_of(t: usize) -> usize {
    ((t & 1) << 2) | ((t & 7) >> 1)
}

/// Host implementation of `fllm_q4z_gptq_repack` for non-CUDA builds: repacks GPTQ-ordered int4 words `[k / 8, n]`
/// (8 consecutive `k` in nibbles `0..8`) into row-major `[n, k / 8]` words with nibble `q4z_nib_of(t)`.
pub fn fllm_q4z_gptq_repack(words: &[u32], k: usize, n: usize) -> Vec<u32> {
    let k_words = k.div_ceil(8);
    let mut out = vec![0u32; n * k_words];
    for w in 0..k_words {
        for row in 0..n {
            let src = words.get(w * n + row).copied().unwrap_or(0);
            let mut word = 0u32;
            for t in 0..8 {
                let code = (src >> (4 * t)) & 15;
                word |= code << (4 * q4z_nib_of(t));
            }
            out[row * k_words + w] = word;
        }
    }
    out
}

/// Host implementation of `fllm_q4z_repack` for non-CUDA builds: repacks AutoAWQ-ordered int4 words `[k, ceil(n / 8)]`
/// into row-major `[n, ceil(k / 8)]` words with nibble `q4z_nib_of(t)`.
pub fn fllm_q4z_awq_repack(awq: &[u32], k: usize, n: usize) -> Vec<u32> {
    let k_words = k.div_ceil(8);
    let n_words = n.div_ceil(8);
    let mut out = vec![0u32; n * k_words];
    for row in 0..n {
        let shift = 4 * AWQ_SLOT[row & 7];
        for w in 0..k_words {
            let mut word = 0u32;
            for t in 0..8 {
                let kk = 8 * w + t;
                let code = if kk < k {
                    (awq[kk * n_words + row / 8] >> shift) & 15
                } else {
                    0
                };
                word |= code << (4 * q4z_nib_of(t));
            }
            out[row * k_words + w] = word;
        }
    }
    out
}

pub fn marlin_arrays(raw: &[u8], rows: usize, cols: usize, group: usize) -> (Vec<u32>, Vec<u16>, Vec<u32>) {
    let groups = cols.div_ceil(group);
    let block = group / 2 + 3;
    let row_bytes = groups * block;
    let words_per_k = rows.div_ceil(8);
    let rows_pad = rows.div_ceil(64) * 64;
    let mut awq = vec![0u32; cols * words_per_k];
    if words_per_k > 0 && cols > 0 {
        awq.par_chunks_mut(64 * words_per_k).enumerate().for_each(|(chunk_index, chunk)| {
            let k0 = chunk_index * 64;
            let chunk_k = chunk.len() / words_per_k;
            for n in 0..rows {
                let row = &raw[n * row_bytes..(n + 1) * row_bytes];
                let (word, shift) = (n / 8, 4 * AWQ_SLOT[n % 8]);
                for kk in (0..chunk_k).step_by(2) {
                    let k = k0 + kk;
                    let byte = row[(k / group) * block + (k % group) / 2];
                    chunk[kk * words_per_k + word] |= ((byte & 15) as u32) << shift;
                    if kk + 1 < chunk_k {
                        chunk[(kk + 1) * words_per_k + word] |= ((byte >> 4) as u32) << shift;
                    }
                }
            }
        });
    }
    let perm = scale_perm();
    let at = |g: usize, n: usize| n * row_bytes + g * block + group / 2;
    let mut scales = vec![0u16; groups * rows_pad];
    let mut plain_zeros = vec![8u8; groups * rows_pad];
    for g in 0..groups {
        for blk in 0..rows_pad / 64 {
            for j in 0..64 {
                let n = blk * 64 + perm[j];
                if n < rows {
                    let i = at(g, n);
                    scales[g * rows_pad + blk * 64 + j] = u16::from_le_bytes([raw[i], raw[i + 1]]);
                    plain_zeros[g * rows_pad + blk * 64 + j] = raw[i + 2] & 15;
                }
            }
        }
    }
    let mut zeros = vec![0u32; groups * rows_pad / 8];
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
    let blocks = cols.div_ceil(64);
    let row_bytes = blocks * 36;
    let k_words = cols.div_ceil(8);
    let rows_pad = rows.div_ceil(64) * 64;
    let mut words = vec![0u32; k_words * rows];
    if rows > 0 && k_words > 0 {
        words.par_chunks_mut(rows).enumerate().for_each(|(kw, dst)| {
            let (block, within) = (kw / 8, kw % 8);
            let (sub, high) = (within / 2, within % 2 == 1);
            for (n, out) in dst.iter_mut().enumerate() {
                let off = n * row_bytes + block * 36 + 4 + sub * 8;
                let mut word = 0u32;
                for i in 0..8 {
                    let b = raw.get(off + i).copied().unwrap_or(0);
                    word |= ((if high { b >> 4 } else { b & 15 }) as u32) << (4 * i);
                }
                *out = word;
            }
        });
    }
    let groups = cols.div_ceil(16).div_ceil(4) * 4;
    let perm = scale_perm();
    let mut permuted = vec![0f32; groups * rows_pad];
    for g in 0..cols.div_ceil(16) {
        for blk in 0..rows_pad / 64 {
            for j in 0..64 {
                let n = blk * 64 + perm[j];
                if n < rows {
                    permuted[g * rows_pad + blk * 64 + j] = e4m3(raw[n * row_bytes + (g / 4) * 36 + g % 4]);
                }
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

    #[test]
    fn unaligned_dimensions_and_gptq_repack_round_trip() {
        let (rows, cols, group) = (40, 96, 32);
        let raw = sample(rows, cols, group);
        let (awq, scales, zeros) = marlin_arrays(&raw, rows, cols, group);
        assert_eq!(awq.len(), cols * (rows / 8));
        assert_eq!(scales.len(), (cols / group) * 64);
        assert_eq!(zeros.len(), (cols / group) * 8);
        let repacked_awq = fllm_q4z_awq_repack(&awq, cols, rows);
        let mut gptq = vec![0u32; (cols / 8) * rows];
        let block = group / 2 + 3;
        let row_bytes = (cols / group) * block;
        for n in 0..rows {
            for w in 0..cols / 8 {
                let mut word = 0u32;
                for t in 0..8 {
                    let k = 8 * w + t;
                    let byte = raw[n * row_bytes + (k / group) * block + (k % group) / 2];
                    let code = if k % 2 == 0 { byte & 15 } else { byte >> 4 } as u32;
                    word |= code << (4 * t);
                }
                gptq[w * rows + n] = word;
            }
        }
        let repacked_gptq = fllm_q4z_gptq_repack(&gptq, cols, rows);
        assert_eq!(repacked_awq, repacked_gptq);
    }
}
