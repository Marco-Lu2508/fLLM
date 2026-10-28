//! Integer-width EXL3 linear weights; algorithm provenance and MIT notice:
//! docs/exl-formats.md (ExLlamaV3, Copyright (c) 2025 Turboderp).
use crate::exl2::{dim, half_round, shape, take, u16_at, u32_at};
use crate::safetensors::{Payload, Tensor};
use fllm_gguf_probe::exl::Exl3Layout;
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

    pub(crate) fn blob_layout(&self) -> Result<Exl3Layout, String> {
        Exl3Layout::new(self.output, self.input, self.bits, self.codebook)
    }

    pub(crate) fn blob_len(&self) -> Result<usize, String> {
        Ok(self.blob_layout()?.header.total)
    }

    /// Write the native blob. `order[i]` is the source output channel written as row `i` (the converter's row reordering);
    /// the trellis words are copied bit for bit, tile columns first.
    pub(crate) fn write_blob(&self, w: &mut impl std::io::Write, order: Option<&[usize]>) -> Result<u64, String> {
        let layout = self.blob_layout()?;
        let mut out_map: Vec<u32> = (0..self.output as u32).collect();
        if let Some(order) = order {
            if order.len() != self.output {
                return Err("invalid EXL3 row permutation".into());
            }
            let mut seen = vec![false; self.output];
            for (row, &source) in order.iter().enumerate() {
                if source >= self.output || std::mem::replace(&mut seen[source], true) {
                    return Err("invalid EXL3 row permutation".into());
                }
                out_map[source] = row as u32;
            }
        }
        let half = |v: &[f32]| v.iter().map(|&x| f16::from_f32(x).to_bits()).collect::<Vec<u16>>();
        let io = |e: std::io::Error| e.to_string();
        w.write_all(&layout.prefix(&half(&self.left), &half(&self.right), &out_map)).map_err(io)?;
        let tile_bytes = 32 * self.bits;
        let (tiles_in, tiles_out) = (self.input / 16, self.output / 16);
        let data = self.trellis.as_slice();
        for no in 0..tiles_out {
            for ki in 0..tiles_in {
                let at = (ki * tiles_out + no) * tile_bytes;
                w.write_all(data.get(at..at + tile_bytes).ok_or("EXL3 trellis truncated")?).map_err(io)?;
            }
        }
        Ok(layout.header.total as u64)
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

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn random_matrix(bits: usize, codebook: u8, input: usize, output: usize, seed: u64) -> Exl3 {
        let mut s = seed;
        let bytes: Vec<u8> = (0..input / 16 * (output / 16) * 32 * bits).map(|_| lcg(&mut s) as u8).collect();
        let len = bytes.len();
        let sign = |s: &mut u64| if lcg(s) & 1 == 0 { 1.0 } else { -1.0 };
        Exl3 {
            trellis: Payload::Owned(Arc::new(bytes), 0..len),
            left: (0..input).map(|_| sign(&mut s)).collect(),
            right: (0..output).map(|_| sign(&mut s)).collect(),
            bits,
            codebook,
            input,
            output,
            elements: input * output,
        }
    }

    #[test]
    fn native_codebook_matches_the_importer_for_every_state() {
        for codebook in 0..3u8 {
            for state in 0..=u16::MAX {
                let native = fllm_gguf_probe::exl::codebook_value(state, codebook);
                let reference = decode_codebook(state, codebook);
                assert!(native == reference || (native.is_nan() && reference.is_nan()), "codebook {codebook} state {state}");
            }
        }
    }

    #[test]
    fn native_blob_matches_the_expanded_importer_path_and_row_order() {
        use fllm_gguf_probe::exl::Exl3View;
        for (bits, codebook) in [(1usize, 0u8), (2, 0), (3, 2), (4, 1), (5, 0), (6, 1), (8, 2)] {
            let reader = random_matrix(bits, codebook, 128, 256, 7 + bits as u64);
            let mut expanded = vec![0.0; reader.element_count()];
            reader.decode_chunk(0, &mut expanded).unwrap();
            let mut blob = Vec::new();
            assert_eq!(reader.write_blob(&mut blob, None).unwrap() as usize, blob.len());
            assert_eq!(blob.len(), reader.blob_len().unwrap());
            let native = Exl3View::validate(&blob).unwrap().dequantize();
            let rms = (expanded.iter().map(|x| x * x).sum::<f32>() / expanded.len() as f32).sqrt();
            let worst = native.iter().zip(&expanded).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            // the importer rounds to F16 after every transform stage, the native path stays in F32
            assert!(worst <= 1e-2 * rms.max(1e-6), "bits {bits} codebook {codebook}: worst {worst}, rms {rms}");
            let order: Vec<usize> = (0..256).map(|i| (i * 37 + 5) % 256).collect();
            let mut permuted = Vec::new();
            reader.write_blob(&mut permuted, Some(&order)).unwrap();
            let pv = Exl3View::validate(&permuted).unwrap().dequantize();
            for (row, &source) in order.iter().enumerate() {
                assert_eq!(pv[row * 128..(row + 1) * 128], native[source * 128..(source + 1) * 128]);
            }
        }
        let reader = random_matrix(2, 0, 128, 128, 1);
        assert!(reader.write_blob(&mut Vec::new(), Some(&[0, 0])).is_err());
        let dup: Vec<usize> = vec![0; 128];
        assert!(reader.write_blob(&mut Vec::new(), Some(&dup)).is_err());
    }
}
