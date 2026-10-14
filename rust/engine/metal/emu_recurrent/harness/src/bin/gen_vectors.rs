//! Writes `vectors.bin` for the CPU emulation of `moe_linear` (rust/engine/metal/emu_recurrent): random GGUF blocks of every format the kernel decodes
//! together with the weights as the Rust CPU dequantiser (`fllm_engine::compute::dequantize`) reads them.
//!
//! File (little endian): u32 count, then per format: u32 gguf type id, qk (weights per block), bytes per block, rows, cols, experts, the raw bytes of all
//! experts back to back (`rows x cols / qk` blocks each), and the dequantised f32 values (`experts x rows x cols`).
//! usage: gen_vectors <out file> [seed]
use half::f16;
use std::io::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// (gguf type id, weights per block, bytes per block, offsets of the f16 scale fields, index of an E8M0 scale byte)
const FORMATS: &[(u32, usize, usize, &[usize], Option<usize>)] = &[
    (2, 32, 18, &[0], None),       // Q4_0
    (3, 32, 20, &[0, 2], None),    // Q4_1
    (6, 32, 22, &[0], None),       // Q5_0
    (7, 32, 24, &[0, 2], None),    // Q5_1
    (8, 32, 34, &[0], None),       // Q8_0
    (20, 32, 18, &[0], None),      // IQ4_NL
    (39, 32, 17, &[], Some(0)),    // MXFP4
    (12, 256, 144, &[0, 2], None), // Q4_K
    (13, 256, 176, &[0, 2], None), // Q5_K
    (14, 256, 210, &[208], None),  // Q6_K
];

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gen_vectors <out file> [seed]");
    let seed: u64 = args
        .next()
        .map_or(0x1234_5678_9ABC_DEF1, |s| s.parse().expect("seed"));
    let mut rng = Rng(seed | 1);
    let (rows, cols, experts) = (6usize, 1024usize, 5usize);
    let mut out = Vec::new();
    out.extend_from_slice(&(FORMATS.len() as u32).to_le_bytes());
    for &(kind, qk, bytes, scales, e8m0) in FORMATS {
        let blocks_per_expert = rows * cols / qk;
        let mut raw = vec![0u8; experts * blocks_per_expert * bytes];
        for b in raw.iter_mut() {
            *b = (rng.next() >> 33) as u8;
        }
        for block in raw.chunks_mut(bytes) {
            for &off in scales {
                let v = (rng.unit() - 0.5) * 0.1;
                block[off..off + 2].copy_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            if let Some(i) = e8m0 {
                block[i] = 118 + (rng.next() % 12) as u8;
            }
        }
        let mut deq = Vec::with_capacity(experts * rows * cols);
        for e in 0..experts {
            let chunk = &raw[e * blocks_per_expert * bytes..(e + 1) * blocks_per_expert * bytes];
            deq.extend(
                fllm_engine::compute::dequantize(chunk, kind, rows * cols)
                    .unwrap_or_else(|err| panic!("dequantize type {kind}: {err}")),
            );
        }
        for v in [
            kind,
            qk as u32,
            bytes as u32,
            rows as u32,
            cols as u32,
            experts as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&raw);
        for v in &deq {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let finite = deq.iter().all(|v| v.is_finite());
        eprintln!(
            "type {kind:>2}: {} experts, {} bytes, dequantised values finite: {finite}",
            experts,
            raw.len()
        );
        assert!(finite, "non-finite weights for type {kind}");
    }
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&out))
        .expect("write");
    eprintln!("wrote {path} ({} bytes)", out.len());
}
