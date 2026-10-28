//! Host mirrors of the CUDA kernels in cuda/exl.cu.
//!
//! No GPU is available where this code is written, so the index and bit arithmetic of the device code is repeated here, one
//! function per thread / lane of the kernel, and compared with the CPU kernels. This proves the algorithm of the device code (code
//! extraction across word boundaries, scale nibbles, permutation, ring windows, tile placement, Hadamard butterflies, output map);
//! it does not prove that nvcc compiles it the way it reads (that needs a GPU run: `fllm-rs selftest`, group `exl`).
use super::*;
use crate::cpu_selftest::Lcg;

fn word_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
}

/// `exl2_kernel` for one output row and one token: 32 lanes, each handling inputs `lane, lane + 32, ...` of every group.
fn mirror_exl2_row(v: &Exl2View, r: usize, x: &[f32]) -> f32 {
    let row = v.row(r);
    let groups = v.groups();
    let scale_word0 = row.len() / 4 - groups.div_ceil(8);
    let mut acc = [0f32; 32];
    for g in 0..groups {
        let e = v.group(g);
        let (n, bits) = (e.n as i32, e.bits as i32);
        let s = (word_at(row, scale_word0 + (g >> 3)) >> (4 * (g & 7))) & 15;
        let scale = ((s + 1) * (s + 1)) as f32 * (half::f16::from_bits(e.max_bits).to_f32() * (1.0 / 256.0));
        let zero = 1i32 << (bits - 1);
        let mask = (1u32 << bits) - 1;
        for (lane, slot) in acc.iter_mut().enumerate() {
            let mut part = 0f32;
            let mut i = lane as i32;
            while i < n {
                let bit = i * bits;
                let (w, sh) = ((bit >> 5) as usize, bit & 31);
                let mut val = word_at(row, e.woff + w) >> sh;
                if sh + bits > 32 {
                    val |= word_at(row, e.woff + w + 1) << (32 - sh);
                }
                let wv = ((val & mask) as i32 - zero) as f32;
                part += wv * x[v.perm(e.start + i as usize)];
                i += 32;
            }
            *slot += scale * part;
        }
    }
    // warp_sum: xor butterfly
    for o in [16usize, 8, 4, 2, 1] {
        let prev = acc;
        for l in 0..32 {
            acc[l] = prev[l] + prev[l ^ o];
        }
    }
    acc[0]
}

/// `exl3_codebook`: the half-precision intrinsics (`__hfma`, `__hadd`) modelled with exact f64 arithmetic and one rounding.
fn mirror_codebook(state: u32, codebook: i32) -> f32 {
    use half::f16;
    if codebook == 2 {
        let x = state.wrapping_mul(0x83dc_d12d);
        let sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24) + 0x6400;
        let a = f64::from(f16::from_bits(sum as u16).to_f32());
        let b = f64::from(f16::from_bits(0x1eee).to_f32());
        let c = f64::from(f16::from_bits(0xc931).to_f32());
        return f16::from_f64(a * b + c).to_f32();
    }
    let mut x = if codebook == 1 { state.wrapping_mul(0xcbac_1fed) } else { state.wrapping_mul(89_226_354).wrapping_add(64_248_484) };
    x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
    let lo = f64::from(f16::from_bits((x & 0xffff) as u16).to_f32());
    let hi = f64::from(f16::from_bits((x >> 16) as u16).to_f32());
    f16::from_f64(lo + hi).to_f32()
}

/// `exl3_state`
fn mirror_state(words: &[u32], bits: i32, index: i32) -> u32 {
    let nw = words.len() as i32;
    let ring = 256 * bits;
    let start = ((index + 1) * bits + ring - 16) % ring;
    let (w, off) = ((start >> 5) as usize, start & 31);
    let hi = u64::from(words[w]);
    let lo = u64::from(words[if w as i32 + 1 == nw { 0 } else { w + 1 }]);
    ((((hi << 32) | lo) >> (48 - off)) & 0xffff) as u32
}

/// `exl3_hadamard128` for the 128 threads of a block (every thread reads its partner before anybody writes)
fn mirror_hadamard(values: &mut [f32]) {
    let mut stride = 1;
    while stride < 128 {
        let old = values.to_vec();
        for tid in 0..128 {
            let other = old[tid ^ stride];
            values[tid] = if tid & stride != 0 { other - old[tid] } else { old[tid] + other };
        }
        stride <<= 1;
    }
    for v in values.iter_mut() {
        *v *= 0.088_388_35;
    }
}

/// `exl3_pre_kernel`, `exl3_mm_kernel` and `exl3_post_kernel` for one token.
fn mirror_exl3(blob: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let v = Exl3View::parse(blob).unwrap();
    let (bits, codebook) = (v.bits() as i32, i32::from(v.codebook()));
    let mut x2 = x.to_vec();
    for blk in 0..cols / 128 {
        let mut s: Vec<f32> = (0..128).map(|t| x[blk * 128 + t] * v.suh(blk * 128 + t)).collect();
        mirror_hadamard(&mut s);
        x2[blk * 128..blk * 128 + 128].copy_from_slice(&s);
    }
    let mut z = vec![0f32; rows];
    for no in 0..rows / 16 {
        let mut acc = [0f32; 256];
        for ki in 0..cols / 16 {
            let tile_bytes = v.tile(no, ki);
            let words: Vec<u32> = (0..8 * bits as usize).map(|i| word_at(tile_bytes, i)).collect();
            // every thread decodes one symbol into its place of the shared tile
            let mut tile = [0f32; 256];
            for tid in 0..256usize {
                let (lane_i, j) = (tid >> 3, tid & 7);
                let pr = 2 * (lane_i & 3) + (j & 1) + if j & 2 != 0 { 8 } else { 0 };
                let pc = (lane_i >> 2) + if j & 4 != 0 { 8 } else { 0 };
                tile[pr * 16 + pc] = mirror_codebook(mirror_state(&words, bits, tid as i32), codebook);
            }
            // thread tid owns column tid & 15 of token tid >> 4 (one token here: the threads of token 0)
            for (tid, a) in acc.iter_mut().enumerate().take(16) {
                let c = tid & 15;
                let mut sum = 0f32;
                for r in 0..16 {
                    sum += x2[ki * 16 + r] * tile[r * 16 + c];
                }
                *a += sum;
            }
        }
        z[no * 16..no * 16 + 16].copy_from_slice(&acc[..16]);
    }
    let mut y = vec![0f32; rows];
    for blk in 0..rows / 128 {
        let mut s = z[blk * 128..blk * 128 + 128].to_vec();
        mirror_hadamard(&mut s);
        for (tid, value) in s.iter().enumerate() {
            let o = blk * 128 + tid;
            y[v.out_row(o)] = value * v.svh(o);
        }
    }
    y
}

#[test]
fn cuda_kernel_arithmetic_mirrored_on_the_host_matches_the_cpu_kernels() {
    for (rows, cols, seed) in [(16usize, 32usize, 1u64), (24, 96, 2), (40, 256, 3), (8, 160, 4)] {
        let mut rng = Lcg(seed);
        let blob = exl::synthetic_exl2(rows, cols, &mut || rng.next());
        let x = rng.vector(cols, 1.0);
        let v = Exl2View::validate(&blob).unwrap();
        let want = linear(&blob, EXL2_TYPE, &x, rows, cols, None).unwrap();
        for (r, &w) in want.iter().enumerate() {
            let got = mirror_exl2_row(&v, r, &x);
            assert!((got - w).abs() <= 1e-5 * (w.abs() + 1.0), "exl2 {rows}x{cols} row {r}: {got} vs {w}");
        }
    }
    for (bits, codebook) in [(1usize, 0u8), (2, 1), (3, 2), (4, 0), (5, 1), (6, 2), (7, 0), (8, 1)] {
        let (rows, cols) = (256usize, 256usize);
        let mut rng = Lcg(900 + bits as u64);
        let blob = exl::synthetic_exl3(rows, cols, bits, codebook, &mut || rng.next());
        let x = rng.vector(cols, 1.0);
        let want = linear(&blob, EXL3_TYPE, &x, rows, cols, None).unwrap();
        let got = mirror_exl3(&blob, &x, rows, cols);
        let scale = want.iter().fold(1e-3f32, |m, w| m.max(w.abs()));
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 2e-5 * scale, "exl3 bits {bits} codebook {codebook} row {r}: {g} vs {w}");
        }
    }
}

#[test]
fn device_codebook_arithmetic_equals_the_cpu_codebook_for_every_state() {
    for codebook in 0..3u8 {
        for state in 0..=u16::MAX {
            let device = mirror_codebook(u32::from(state), i32::from(codebook));
            let cpu = exl::codebook_value(state, codebook);
            assert!(device == cpu || (device.is_nan() && cpu.is_nan()), "codebook {codebook} state {state}: {device} vs {cpu}");
        }
    }
}

#[test]
fn bit_extraction_covers_every_width_at_every_alignment() {
    // the device extraction `(w0 >> sh) | (w1 << (32 - sh))` for all widths and all bit offsets of a 4-word stream
    let words: [u32; 4] = [0x0123_4567, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
    let stream = u128::from(words[0]) | u128::from(words[1]) << 32 | u128::from(words[2]) << 64 | u128::from(words[3]) << 96;
    for bits in [2u32, 3, 4, 5, 6, 8] {
        for i in 0..(96 / bits) {
            let bit = i * bits;
            let (w, sh) = ((bit >> 5) as usize, bit & 31);
            let mut v = words[w] >> sh;
            if sh + bits > 32 {
                v |= words[w + 1] << (32 - sh);
            }
            assert_eq!(v & ((1 << bits) - 1), ((stream >> bit) & ((1 << bits) - 1)) as u32, "bits {bits} index {i}");
        }
    }
}
