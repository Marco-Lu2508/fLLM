//! MLX affine group-quantized weights (`MLXA` types, see `fllm_gguf_probe::MlxAffine`) on GPUs: the device layout and a host model of
//! the arithmetic of the CUDA kernels (`cuda/mlx_affine.cu`).
//!
//! The GGUF tensor is `rows x (cols / group)` blocks of `[scale: 2][bias: 2][codes: group * bits / 8]`. The device image splits them into
//! three planes so that the kernels read the codes as 32-bit words and the scales and biases as 16-bit values, each coalesced:
//!
//! ```text
//! codes   rows x (cols * bits / 8) bytes   row r = the MLX words of row r (a little-endian bit stream, code i at bit i * bits)
//! scales  rows x (cols / group)    u16     f16 or bf16 as the type says
//! biases  rows x (cols / group)    u16
//! ```
//!
//! The planes follow each other in one buffer; `scales` starts at `rows * cols * bits / 8` (a multiple of 4) and `biases` right after it.
//! `w = code * scale + bias`. A chunk of 32 codes is exactly `bits` words, so a kernel works on chunks of 32 and a code that straddles two
//! words (3, 5 and 6 bits) never leaves its chunk.

use fllm_gguf_probe::MlxAffine;

/// Byte offsets of the planes and the total size of the device image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Planes {
    pub scales: usize,
    pub biases: usize,
    pub total: usize,
}

pub fn planes(m: MlxAffine, rows: usize, cols: usize) -> Planes {
    let codes = rows * cols * m.bits as usize / 8;
    let groups = rows * (cols / m.group as usize);
    Planes { scales: codes, biases: codes + groups * 2, total: codes + groups * 4 }
}

/// Whether a `rows x cols` weight of this type has the geometry the device kernels need.
pub fn device_shape_ok(m: MlxAffine, rows: usize, cols: usize) -> bool {
    rows > 0 && cols > 0 && cols.is_multiple_of(m.group as usize) && cols.is_multiple_of(32) && u32::try_from(cols).is_ok() && u32::try_from(rows).is_ok()
}

/// Bytes of the GGUF tensor of `rows x cols`.
pub fn file_bytes(m: MlxAffine, rows: usize, cols: usize) -> usize {
    rows * (cols / m.group as usize) * m.block_bytes() as usize
}

/// The device image of the GGUF bytes `raw` of a `rows x cols` weight.
pub fn device_planes(raw: &[u8], m: MlxAffine, rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    if !device_shape_ok(m, rows, cols) {
        return Err(format!("{rows} x {cols} MLX weight: columns must be a multiple of the group size {} and of 32", m.group));
    }
    if raw.len() != file_bytes(m, rows, cols) {
        return Err(format!("MLX weight has {} bytes, expected {} for {rows} x {cols}", raw.len(), file_bytes(m, rows, cols)));
    }
    let (group, bits) = (m.group as usize, m.bits as usize);
    let p = planes(m, rows, cols);
    let (per_row, group_bytes) = (cols / group, group * bits / 8);
    let mut out = vec![0u8; p.total];
    for (g, block) in raw.chunks_exact(m.block_bytes() as usize).enumerate() {
        // group g of the tensor is group g % per_row of row g / per_row; its codes continue the row's bit stream
        out[p.scales + 2 * g..p.scales + 2 * g + 2].copy_from_slice(&block[0..2]);
        out[p.biases + 2 * g..p.biases + 2 * g + 2].copy_from_slice(&block[2..4]);
        out[g * group_bytes..(g + 1) * group_bytes].copy_from_slice(&block[4..]);
    }
    debug_assert_eq!(per_row * rows * group_bytes, p.scales);
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cpu_mlx::tests::{random_weight, Lcg, ALL_BITS};

    fn half_to_f32(raw: u16, bf16: bool) -> f32 {
        if bf16 {
            f32::from_bits(u32::from(raw) << 16)
        } else {
            half::f16::from_bits(raw).to_f32()
        }
    }

    /// The code at index `i` of a chunk of 32 whose `bits` words start at `words`: the device function `mlx_code<BITS>`.
    fn code(words: &[u32], i: usize, bits: usize) -> u32 {
        let (word, shift) = (i * bits / 32, (i * bits) % 32);
        let mut v = words[word] >> shift;
        if shift + bits > 32 {
            v |= words[word + 1] << (32 - shift);
        }
        v & ((1u32 << bits) - 1)
    }

    /// Host model of `mlx_gemv_kernel`: one warp per row, lane `l` takes the chunks `l, l + 32, ...` of 32 weights, accumulates its
    /// products in order for every token, and the 32 lane sums are added by the xor butterfly of `warp_sum`.
    pub(crate) fn model_gemv(image: &[u8], m: MlxAffine, rows: usize, cols: usize, x: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        let (bits, group) = (m.bits as usize, m.group as usize);
        let p = planes(m, rows, cols);
        let batch = x.len() / cols;
        let chunks = cols / 32;
        let u16_at = |at: usize| u16::from_le_bytes([image[at], image[at + 1]]);
        let mut y = vec![0f32; batch * rows];
        for row in 0..rows {
            for t in 0..batch {
                let mut lanes = [0f32; 32];
                for (lane, sum) in lanes.iter_mut().enumerate() {
                    let mut chunk = lane;
                    while chunk < chunks {
                        let g = row * (cols / group) + chunk * 32 / group;
                        let (scale, bias_g) = (half_to_f32(u16_at(p.scales + 2 * g), m.bf16), half_to_f32(u16_at(p.biases + 2 * g), m.bf16));
                        let at = row * (cols * bits / 8) + chunk * bits * 4;
                        let words: Vec<u32> = (0..bits).map(|k| u32::from_le_bytes(image[at + 4 * k..at + 4 * k + 4].try_into().unwrap())).collect();
                        for i in 0..32 {
                            let w = (code(&words, i, bits) as f32).mul_add(scale, bias_g);
                            *sum = w.mul_add(x[t * cols + chunk * 32 + i], *sum);
                        }
                        chunk += 32;
                    }
                }
                let mut offset = 16;
                while offset > 0 {
                    let snapshot = lanes;
                    for (l, v) in lanes.iter_mut().enumerate() {
                        *v += snapshot[l ^ offset];
                    }
                    offset >>= 1;
                }
                y[t * rows + row] = lanes[0] + bias.map_or(0.0, |b| b[row]);
            }
        }
        y
    }

    #[test]
    fn planes_hold_the_words_scales_and_biases_of_every_row() {
        let mut rng = Lcg(5);
        for bits in ALL_BITS {
            for group in [32u32, 64, 128] {
                for bf16 in [false, true] {
                    let m = MlxAffine { bits, group, bf16 };
                    let (rows, cols) = (3, group as usize * 2);
                    let (raw, codes, scales, biases) = random_weight(&mut rng, m, rows, cols);
                    let image = device_planes(&raw, m, rows, cols).unwrap();
                    let p = planes(m, rows, cols);
                    assert_eq!(image.len(), p.total);
                    assert_eq!(p.scales % 4, 0);
                    for r in 0..rows {
                        let row_words: Vec<u32> = image[r * cols * bits as usize / 8..(r + 1) * cols * bits as usize / 8]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|w| u32::from_le_bytes(*w))
                            .collect();
                        for c in 0..cols {
                            let got = code(&row_words[c / 32 * bits as usize..], c % 32, bits as usize);
                            assert_eq!(got, codes[r * cols + c], "{bits} bits row {r} col {c}");
                        }
                    }
                    for g in 0..rows * cols / group as usize {
                        let s = half_to_f32(u16::from_le_bytes([image[p.scales + 2 * g], image[p.scales + 2 * g + 1]]), bf16);
                        let b = half_to_f32(u16::from_le_bytes([image[p.biases + 2 * g], image[p.biases + 2 * g + 1]]), bf16);
                        assert_eq!((s, b), (scales[g], biases[g]));
                    }
                }
            }
        }
        let m = MlxAffine { bits: 4, group: 64, bf16: false };
        assert!(device_planes(&[0; 10], m, 1, 64).is_err(), "wrong length");
        assert!(device_planes(&vec![0; file_bytes(m, 1, 96)], m, 1, 96).is_err(), "columns not a multiple of the group");
    }

    #[test]
    fn the_kernel_model_matches_the_cpu_reference_for_all_widths_and_odd_shapes() {
        let mut rng = Lcg(21);
        for bits in ALL_BITS {
            for (group, cols) in [(32u32, 32usize), (32, 96), (64, 192), (128, 128), (64, 2112)] {
                for bf16 in [false, true] {
                    // rows that are no multiple of anything, tokens that are not a power of two
                    for (rows, batch) in [(1usize, 1usize), (7, 2), (13, 4)] {
                        let m = MlxAffine { bits, group, bf16 };
                        let (raw, ..) = random_weight(&mut rng, m, rows, cols);
                        let image = device_planes(&raw, m, rows, cols).unwrap();
                        let x: Vec<f32> = (0..batch * cols).map(|_| rng.unit()).collect();
                        let bias: Vec<f32> = (0..rows).map(|_| rng.unit()).collect();
                        let got = model_gemv(&image, m, rows, cols, &x, Some(&bias));
                        let want = crate::cpu_mlx::linear(&raw, m.type_id(), &x, rows, cols, Some(&bias)).unwrap();
                        let weights = crate::cpu_mlx::dequantize(&raw, m.type_id(), rows * cols).unwrap();
                        for t in 0..batch {
                            for r in 0..rows {
                                let magnitude: f32 = (0..cols).map(|k| (weights[r * cols + k] * x[t * cols + k]).abs()).sum::<f32>() + bias[r].abs();
                                let (a, b) = (got[t * rows + r], want[t * rows + r]);
                                assert!((a - b).abs() <= 1e-5 * magnitude + 1e-6, "{bits} bits g{group} {rows}x{cols}x{batch}: {a} vs {b}");
                            }
                        }
                    }
                }
            }
        }
    }
}
