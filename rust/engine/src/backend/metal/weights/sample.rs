//! Deterministic sample weights and their reference values, for the CPU tests of this module and for the GPU check of the Metal
//! round (`gpu_check.rs`). Not used by the backend itself.

use super::format::format_info;
use crate::backend::QuantType;
use half::{bf16, f16};

/// Deterministic generator (64 bit LCG).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform in [-1, 1].
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() % 20_001) as f32 / 10_000.0 - 1.0
    }
}

/// `rows` x `cols` tensor bytes of a valid structure for `quant` (the layout of a GGUF tensor, the permutation table of Q4ZP and
/// the row scales of FP8M included): uniform payload bytes, finite moderate scales, no NaN / infinity patterns. Panics for formats
/// without a Metal kernel.
pub fn sample_weights(quant: QuantType, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let info = format_info(quant).expect("format with a Metal kernel");
    let sign = |rng: &mut Lcg| {
        if rng.next_u64().is_multiple_of(2) {
            0
        } else {
            0x80u8
        }
    };
    match quant {
        QuantType::F32 => {
            return (0..rows * cols)
                .flat_map(|_| (rng.unit() * 2.0).to_le_bytes())
                .collect()
        }
        QuantType::F16 => {
            return (0..rows * cols)
                .flat_map(|_| f16::from_f32(rng.unit() * 2.0).to_le_bytes())
                .collect()
        }
        QuantType::BF16 => {
            return (0..rows * cols)
                .flat_map(|_| bf16::from_f32(rng.unit() * 2.0).to_le_bytes())
                .collect()
        }
        QuantType::Fp8E4m3 | QuantType::Fp8Marlin => {
            // exponent / mantissa without the NaN pattern
            let mut bytes: Vec<u8> = (0..rows * cols)
                .map(|_| ((rng.next_u64() % 126) as u8 + 1) | sign(&mut rng))
                .collect();
            if quant == QuantType::Fp8Marlin {
                for _ in 0..rows {
                    bytes.extend_from_slice(
                        &f16::from_f32(0.001 + (rng.next_u64() % 100) as f32 / 4000.0)
                            .to_le_bytes(),
                    );
                }
            }
            return bytes;
        }
        QuantType::Fp8E5m2 => {
            // exponent below 31
            return (0..rows * cols)
                .map(|_| ((rng.next_u64() % 123) as u8 + 1) | sign(&mut rng))
                .collect();
        }
        _ => {}
    }
    let (block, width) = (info.src_qk, info.src_block_bytes);
    let mut bytes = Vec::new();
    for _ in 0..rows * (cols / block) {
        let mut b: Vec<u8> = (0..width).map(|_| rng.next_u64() as u8).collect();
        let d = f16::from_f32(0.001 + (rng.next_u64() % 100) as f32 / 4000.0).to_le_bytes();
        let m = f16::from_f32((rng.next_u64() % 100) as f32 / 4000.0).to_le_bytes();
        match quant {
            QuantType::Q4_0
            | QuantType::Q5_0
            | QuantType::Q8_0
            | QuantType::IQ4_NL
            | QuantType::IQ4_XS => b[..2].copy_from_slice(&d),
            QuantType::Q4_1 | QuantType::Q5_1 | QuantType::Q4_K | QuantType::Q5_K => {
                b[..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&m);
            }
            QuantType::Q2_K => {
                b[80..82].copy_from_slice(&d);
                b[82..84].copy_from_slice(&m);
            }
            QuantType::Q3_K => b[108..110].copy_from_slice(&d),
            QuantType::Q6_K => b[208..210].copy_from_slice(&d),
            QuantType::Mxfp4 => b[0] = 118 + (rng.next_u64() % 16) as u8,
            QuantType::NvFp4 => {
                for x in &mut b[..4] {
                    *x = 0x28 + (rng.next_u64() % 16) as u8;
                }
            }
            q if q.q4z_group().is_some() => {
                b[block / 2..block / 2 + 2].copy_from_slice(&d);
                b[block / 2 + 2] = (rng.next_u64() % 16) as u8;
            }
            other => panic!("no generator for {other:?}"),
        }
        bytes.extend(b);
    }
    if quant.is_q4zp() {
        let mut perm: Vec<u32> = (0..cols as u32).collect();
        for i in (1..cols).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        for p in perm {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
    }
    bytes
}

/// Reference weights (original column order) of a tensor in the layout of [`sample_weights`], row-major, from the CPU decoders of
/// the engine (`compute::dequantize`, `lowbit`).
pub fn reference_weights(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    match quant {
        QuantType::Fp8E4m3 => {
            let mut out = vec![0.0; rows * cols];
            crate::lowbit::dequantize_fp8_e4m3(raw, 1.0, &mut out)?;
            Ok(out)
        }
        QuantType::Fp8E5m2 => {
            let mut out = vec![0.0; rows * cols];
            crate::lowbit::dequantize_fp8_e5m2(raw, 1.0, &mut out)?;
            Ok(out)
        }
        _ => crate::compute::dequantize(raw, quant.to_gguf_type_id(), rows * cols),
    }
}
