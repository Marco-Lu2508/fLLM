//! Integer-width EXL3 linear weights; algorithm provenance and MIT notice:
//! docs/exl-formats.md (ExLlamaV3, Copyright (c) 2025 Turboderp).
use crate::exl2::{dim, half_round, shape, take, u16_at, u32_at};
use crate::safetensors::{Payload, Tensor};
use half::f16;
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
pub struct Exl3 {
    trellis: Payload,
    left: Vec<f32>,
    right: Vec<f32>,
    bits: usize,
    codebook: u8,
    pub(crate) input: usize,
    pub(crate) output: usize,
    elements: usize,
}

fn signs(
    g: &mut HashMap<String, Tensor>,
    base: &str,
    dense: &str,
    packed: &str,
    n: usize,
) -> Result<Vec<f32>, String> {
    if let Some(t) = g.remove(dense) {
        if g.contains_key(packed) {
            return Err(format!("EXL3 {base}: conflicting {dense}/{packed} tensors"));
        }
        shape(&t, "F16", &[n])?;
        let values: Vec<f32> = (0..n)
            .map(|i| f16::from_bits(u16_at(t.payload.as_slice(), i)).to_f32())
            .collect();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(format!("tensor {}: non-finite transform", t.name));
        }
        Ok(values)
    } else {
        let t = take(g, base, packed).map_err(|_| {
            format!("EXL3 {base}: missing tensor {base}.{dense} (or {base}.{packed})")
        })?;
        shape(&t, "I16", &[n / 16])?;
        Ok((0..n)
            .map(|i| {
                if (u16_at(t.payload.as_slice(), i / 16) >> (i % 16)) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect())
    }
}

fn decode_codebook(state: u16, codebook: u8) -> f32 {
    let x = u32::from(state);
    if codebook == 2 {
        let x = x.wrapping_mul(0x83dc_d12d);
        let sum = x.to_le_bytes().iter().map(|&b| u16::from(b)).sum::<u16>() + 0x6400;
        // Half FMA: one rounding after multiplication plus addition.
        return half_round(f16::from_bits(sum).to_f32().mul_add(
            f16::from_bits(0x1eee).to_f32(),
            f16::from_bits(0xc931).to_f32(),
        ));
    }
    let x = if codebook == 1 {
        x.wrapping_mul(0xcbac_1fed)
    } else {
        x.wrapping_mul(89_226_354).wrapping_add(64_248_484)
    };
    let x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
    half_round(f16::from_bits(x as u16).to_f32() + f16::from_bits((x >> 16) as u16).to_f32())
}

/// State window in the MSB-first ring of u32 words (the packer swaps u16s).
fn state_at(bytes: &[u8], bits: usize, index: usize) -> u16 {
    let ring_bits = 256 * bits;
    let start = ((index + 1) * bits + ring_bits - 16) % ring_bits;
    let mut state = 0u16;
    for j in 0..16 {
        let bit = (start + j) % ring_bits;
        state = (state << 1) | ((u32_at(bytes, bit / 32) >> (31 - bit % 32)) & 1) as u16;
    }
    state
}

fn hadamard(values: &mut [f32]) {
    let mut stride = 1;
    while stride < 128 {
        for begin in (0..128).step_by(stride * 2) {
            for i in begin..begin + stride {
                let a = values[i];
                let b = values[i + stride];
                values[i] = a + b;
                values[i + stride] = a - b;
            }
        }
        stride *= 2;
    }
    for x in values {
        *x = half_round(*x / 128f32.sqrt());
    }
}

impl Exl3 {
    pub(crate) fn read(base: &str, g: &mut HashMap<String, Tensor>) -> Result<Self, String> {
        if g.contains_key("scale") {
            return Err(format!(
                "EXL3 {base}: deprecated separate scale tensor is unsupported"
            ));
        }
        let trellis = take(g, base, "trellis")?;
        let input = dim(&trellis, 0)?
            .checked_mul(16)
            .ok_or("EXL3 input overflow")?;
        let output = dim(&trellis, 1)?
            .checked_mul(16)
            .ok_or("EXL3 output overflow")?;
        let words = dim(&trellis, 2)?;
        let bits = words / 16;
        if !words.is_multiple_of(16) || !(1..=8).contains(&bits) {
            return Err(format!("tensor {}: unsupported trellis width {words}/16; fractional and unpacked formats are not imported", trellis.name));
        }
        shape(&trellis, "I16", &[input / 16, output / 16, words])?;
        if !input.is_multiple_of(128) || !output.is_multiple_of(128) {
            return Err(format!(
                "tensor {}: Hadamard dimensions must be divisible by 128",
                trellis.name
            ));
        }
        let elements = input.checked_mul(output).ok_or("EXL3 matrix overflow")?;
        let left = signs(g, base, "suh", "su", input)?;
        let right = signs(g, base, "svh", "sv", output)?;
        let mcg = g.remove("mcg");
        let mul1 = g.remove("mul1");
        if mcg.is_some() && mul1.is_some() {
            return Err(format!("EXL3 {base}: conflicting mcg and mul1 codebooks"));
        }
        for t in [&mcg, &mul1].into_iter().flatten() {
            shape(t, "I32", &[])?;
        }
        let codebook = if mul1.is_some() {
            2
        } else if mcg.is_some() {
            1
        } else {
            0
        };
        Ok(Self {
            trellis: trellis.payload,
            left,
            right,
            bits,
            codebook,
            input,
            output,
            elements,
        })
    }

    pub(crate) fn element_count(&self) -> usize {
        self.elements
    }

    fn tile(&self, input_base: usize, output_base: usize, scratch: &mut [f32]) {
        for k in 0..8 {
            for n in 0..8 {
                let tile_index =
                    ((input_base / 16 + k) * (self.output / 16) + output_base / 16 + n)
                        * 32
                        * self.bits;
                let bytes = &self.trellis.as_slice()[tile_index..tile_index + 32 * self.bits];
                for lane in 0..32 {
                    for j in 0..8 {
                        let row = 2 * (lane % 4) + [0, 1, 8, 9, 0, 1, 8, 9][j];
                        let col = lane / 4 + if j < 4 { 0 } else { 8 };
                        scratch[(k * 16 + row) * 128 + n * 16 + col] = decode_codebook(
                            state_at(bytes, self.bits, lane * 8 + j),
                            self.codebook,
                        );
                    }
                }
            }
        }
        let mut column = [0.0; 128];
        for out in 0..128 {
            for i in 0..128 {
                column[i] = scratch[i * 128 + out];
            }
            hadamard(&mut column);
            for i in 0..128 {
                scratch[i * 128 + out] = half_round(column[i] * self.left[input_base + i]);
            }
        }
        for row in scratch.as_chunks_mut::<128>().0 {
            hadamard(row);
            for (out, x) in row.iter_mut().enumerate() {
                *x = half_round(*x * self.right[output_base + out]);
            }
        }
    }

    pub(crate) fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start
            .checked_add(dst.len())
            .filter(|&end| end <= self.elements)
            .ok_or("EXL3 decode range out of bounds")?;
        // Enumerate intersecting blocks, then reuse one 64 KiB scratch tile.
        let mut blocks = BTreeSet::new();
        let mut flat = start;
        while flat < end {
            let input = flat % self.input;
            let output = flat / self.input;
            blocks.insert((input / 128 * 128, output / 128 * 128));
            flat += (128 - input % 128).min(end - flat);
        }
        let mut scratch = vec![0.0; 128 * 128];
        for (input, output) in blocks {
            self.tile(input, output, &mut scratch);
            for out in output..output + 128 {
                let row_start = out * self.input + input;
                let lo = row_start.max(start);
                let hi = (row_start + 128).min(end);
                if lo < hi {
                    for flat in lo..hi {
                        dst[flat - start] = scratch[(flat - row_start) * 128 + out - output];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    fn pack(bits: usize, symbols: &[u16; 256]) -> Vec<u8> {
        let mut words = vec![0u32; 8 * bits];
        for (t, &symbol) in symbols.iter().enumerate() {
            for j in 0..bits {
                let b = t * bits + j;
                words[b / 32] |= u32::from((symbol >> (bits - 1 - j)) & 1) << (31 - b % 32);
            }
        }
        words.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    #[test]
    fn synthetic_tail_biting_ring_all_integer_widths() {
        for bits in 1..=8 {
            let mut symbols = [0u16; 256];
            for (i, s) in symbols.iter_mut().enumerate() {
                *s = ((i * 13 + 7) & ((1 << bits) - 1)) as u16;
            }
            let bytes = pack(bits, &symbols);
            for t in 0..256 {
                let mut expected = 0u16;
                for age in (0..16).rev() {
                    let ring = ((t + 1) * bits + 256 * bits - 1 - age) % (256 * bits);
                    expected =
                        (expected << 1) | ((symbols[ring / bits] >> (bits - 1 - ring % bits)) & 1);
                }
                assert_eq!(state_at(&bytes, bits, t), expected);
            }
        }
    }
    #[test]
    fn constant_rotated_tiles_transform_with_signs_and_chunk_boundaries() {
        let bytes = pack(4, &[0; 256]).repeat(64);
        let len = bytes.len();
        let mut left = vec![1.0; 128];
        left[0] = -1.0;
        let reader = Exl3 {
            trellis: Payload::Owned(Arc::new(bytes), 0..len),
            left,
            right: vec![1.0; 128],
            bits: 4,
            codebook: 0,
            input: 128,
            output: 128,
            elements: 16384,
        };
        let mut values = vec![0.0; 16384];
        reader.decode_chunk(0, &mut values).unwrap();
        let expected =
            half_round(half_round(decode_codebook(0, 0) * 128f32.sqrt()) * -128f32.sqrt());
        assert_eq!(values[0], expected);
        assert!(values[1..].iter().all(|&v| v == 0.0));
        let mut part = [0.0; 29];
        reader.decode_chunk(121, &mut part).unwrap();
        assert_eq!(&part, &values[121..150]);
        assert!(reader.decode_chunk(16384, &mut part).is_err());
    }
    #[test]
    fn codebook_known_zero_state_and_fractional_rejection() {
        assert_eq!(
            decode_codebook(0, 1),
            half_round(2.0 * f16::from_bits(0x3b60).to_f32())
        );
        let mut g = HashMap::new();
        g.insert(
            "trellis".into(),
            Tensor {
                name: "w.trellis".into(),
                dtype: "I16".into(),
                shape: vec![8, 8, 24],
                payload: Payload::Owned(Arc::new(vec![0; 3072]), 0..3072),
            },
        );
        assert!(Exl3::read("w", &mut g).unwrap_err().contains("fractional"));
    }
}
