//! `I2_S`: the ternary tensor type of Microsoft's BitNet fork of llama.cpp (GGML type id 36, architecture `bitnet-b1.58`).
//!
//! Layout (read from the files of microsoft/bitnet-b1.58-2B-4T-gguf; this is an independent implementation): the weights are
//! cut into blocks of 128 consecutive elements, each block is 32 bytes, byte `p` of a block holds the elements `p`, `32 + p`,
//! `64 + p` and `96 + p` in its bit pairs 7-6, 5-4, 3-2 and 1-0. A code is 0, 1 or 2 for -1, 0 and +1 (3 does not occur).
//! After the `count / 4` code bytes the tensor ends with one `f32` scale for the whole tensor, padded to 32 bytes: the value of an
//! element is `(code - 1) * scale`. The tensor therefore holds `count / 4 + 32` bytes.

use half::f16;

/// GGML type id of `I2_S` in the BitNet fork.
pub const I2_S: u32 = 36;
const BLOCK: usize = 128;
const TRAILER: usize = 32;

/// Bytes of a tensor with `count` elements.
pub fn payload_len(count: usize) -> usize {
    count / 4 + TRAILER
}

fn check(bytes: &[u8], count: usize) -> Result<f32, String> {
    if !count.is_multiple_of(BLOCK) {
        return Err(format!("I2_S tensor with {count} elements is not a multiple of {BLOCK}"));
    }
    if bytes.len() != payload_len(count) {
        return Err(format!("invalid I2_S tensor size: got {} bytes, expected {}", bytes.len(), payload_len(count)));
    }
    let at = count / 4;
    let scale = f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if !scale.is_finite() {
        return Err("I2_S tensor scale is not finite".into());
    }
    Ok(scale)
}

/// The ternary codes (-1, 0, +1) of the tensor in element order.
fn ternary(bytes: &[u8], count: usize) -> impl Iterator<Item = i8> + '_ {
    (0..count / BLOCK).flat_map(move |block| {
        let codes = &bytes[block * 32..block * 32 + 32];
        (0..BLOCK).map(move |j| ((codes[j % 32] >> (6 - 2 * (j / 32))) & 3) as i8 - 1)
    })
}

/// All values as f32.
pub fn dequantize(bytes: &[u8], count: usize) -> Result<Vec<f32>, String> {
    let scale = check(bytes, count)?;
    Ok(ternary(bytes, count).map(|t| f32::from(t) * scale).collect())
}

/// The same values as Q8_0 blocks (GGML type 8: an `f16` scale and 32 `i8` per block). Exact when the scale is representable in
/// `f16`, which holds for the bf16 weight scales of BitNet checkpoints; this is how the tensor is kept in memory (8.5 bits per weight
/// instead of 32) and run with the existing Q8_0 kernels.
pub fn to_q8_0(bytes: &[u8], count: usize) -> Result<Vec<u8>, String> {
    let scale = check(bytes, count)?;
    let d = f16::from_f32(scale);
    if !d.to_f32().is_finite() || (d.to_f32() - scale).abs() > scale.abs() * 1e-3 {
        return Err(format!("I2_S scale {scale} does not fit a half-precision Q8_0 scale"));
    }
    let mut out = Vec::with_capacity(count / 32 * 34);
    let mut codes = ternary(bytes, count);
    for _ in 0..count / 32 {
        out.extend_from_slice(&d.to_le_bytes());
        out.extend(codes.by_ref().take(32).map(|t| t as u8));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packing of the BitNet fork's `quantize_i2_s`, written from the layout description above.
    fn pack(values: &[i8], scale: f32) -> Vec<u8> {
        let mut out = vec![0u8; payload_len(values.len())];
        for (i, &v) in values.iter().enumerate() {
            let (block, j) = (i / BLOCK, i % BLOCK);
            out[block * 32 + j % 32] |= ((v + 1) as u8) << (6 - 2 * (j / 32));
        }
        out[values.len() / 4..values.len() / 4 + 4].copy_from_slice(&scale.to_le_bytes());
        out
    }

    fn pattern(n: usize) -> Vec<i8> {
        (0..n).map(|i| ((i * 7 + i / 5) % 3) as i8 - 1).collect()
    }

    #[test]
    fn decodes_the_block_layout() {
        let values = pattern(384);
        let bytes = pack(&values, 0.0625);
        let out = dequantize(&bytes, 384).unwrap();
        for (i, (&v, &t)) in out.iter().zip(&values).enumerate() {
            assert_eq!(v, f32::from(t) * 0.0625, "element {i}");
        }
        // element 33 sits in byte 1, bits 5-4
        let mut one = vec![0i8; 128];
        one[33] = 1;
        let bytes = pack(&one, 1.0);
        assert_eq!(bytes[1], (2 << 4) | (1 << 6) | (1 << 2) | 1, "zeros are code 1");
    }

    #[test]
    fn q8_0_blocks_carry_the_same_values() {
        let values = pattern(256);
        let bytes = pack(&values, 0.09375);
        let q8 = to_q8_0(&bytes, 256).unwrap();
        assert_eq!(q8.len(), 256 / 32 * 34);
        let want = dequantize(&bytes, 256).unwrap();
        for (b, block) in q8.as_chunks::<34>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            for (k, &q) in block[2..].iter().enumerate() {
                assert!((d * f32::from(q as i8) - want[b * 32 + k]).abs() <= 1e-7, "block {b} element {k}");
            }
        }
    }

    #[test]
    fn rejects_wrong_sizes() {
        let bytes = pack(&pattern(128), 1.0);
        assert!(dequantize(&bytes[..bytes.len() - 1], 128).is_err());
        assert!(dequantize(&bytes, 192).is_err());
        assert!(to_q8_0(&pack(&pattern(128), 1.0e9), 128).is_err(), "a scale outside the f16 range is refused");
    }
}
