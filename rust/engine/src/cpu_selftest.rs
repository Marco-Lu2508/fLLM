//! `fllm-rs selftest-cpu`: run every CPU kernel tier this machine supports on random data and
//! compare each kernel with a scalar reference.
//!
//! The references follow each kernel's arithmetic contract: projections that quantize their
//! activations to 8 bits are compared with the same 8-bit activations evaluated in f64, so a
//! check measures kernel defects (indexing, signs, scales, tails), not the expected noise of
//! 8-bit activations against an f32 projection. Activation packing itself is compared with the
//! scalar packing bit for bit.

use crate::cpu_kernels::{self, CpuTier};
use crate::cpu_quantized::X;
use half::{bf16, f16};
use std::time::Instant;

/// Allowed error of a projection or attention output, relative to the sum of the absolute
/// values of its terms. Float summation order alone stays several orders of magnitude below it;
/// a wrong index, sign or scale exceeds it.
pub(crate) const TOLERANCE: f64 = 1e-4;

/// Deterministic generator for weights and activations.
pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// Uniform in [-1, 1].
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() % 20_001) as f32 / 10_000.0 - 1.0
    }
    pub(crate) fn vector(&mut self, len: usize, amplitude: f32) -> Vec<f32> {
        (0..len).map(|_| self.unit() * amplitude).collect()
    }
}

/// (values, bytes) per block of the formats the self-test covers.
pub(crate) fn layout(kind: u32) -> (usize, usize) {
    match kind {
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        40 => (64, 36),
        0x8000_0010 | 0x8000_0014 => (32, 19),
        0x8000_0011 | 0x8000_0015 => (64, 35),
        0x8000_0012 | 0x8000_0016 => (128, 67),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        21 => (256, 110),
        22 => (256, 82),
        29 => (256, 56),
        20 => (32, 18),
        23 => (256, 136),
        39 => (32, 17),
        0x8000_0020 => (1, 1),
        _ => panic!("no self-test layout for GGUF type {kind}"),
    }
}

pub(crate) fn format_name(kind: u32) -> &'static str {
    match kind {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        21 => "iq3_s",
        22 => "iq2_s",
        29 => "iq1_m",
        20 => "iq4_nl",
        23 => "iq4_xs",
        30 => "bf16",
        39 => "mxfp4",
        40 => "nvfp4",
        0x8000_0010 => "q4z32",
        0x8000_0011 => "q4z64",
        0x8000_0012 => "q4z128",
        0x8000_0014 => "q4zp32",
        0x8000_0015 => "q4zp64",
        0x8000_0016 => "q4zp128",
        0x8000_0020 => "fp8m",
        _ => "?",
    }
}

/// Random packed rows. Payload bytes are uniform; every scale field holds a finite, moderate
/// value so products stay far from overflow.
pub(crate) fn random_weights(kind: u32, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
    if kind == 0x8000_0020 {
        let mut bytes = Vec::with_capacity(rows * cols + rows * 2);
        for _ in 0..rows * cols {
            let b = 1 + (rng.next() % 126) as u8;
            let sign = if rng.next().is_multiple_of(2) { 0 } else { 0x80 };
            bytes.push(b | sign);
        }
        for _ in 0..rows {
            let d = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_le_bytes();
            bytes.extend_from_slice(&d);
        }
        return bytes;
    }
    let (block, width) = layout(kind);
    let mut bytes = Vec::with_capacity(rows * cols / block * width);
    for _ in 0..rows * (cols / block) {
        let mut b = (0..width).map(|_| rng.next() as u8).collect::<Vec<_>>();
        let d = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_le_bytes();
        let m = f16::from_f32((rng.next() % 100) as f32 / 4000.0).to_le_bytes();
        match kind {
            2 | 6 | 8 | 20 | 23 => b[..2].copy_from_slice(&d),
            3 | 7 | 12 | 13 => {
                b[..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&m);
            }
            10 => {
                b[80..82].copy_from_slice(&d);
                b[82..84].copy_from_slice(&m);
            }
            11 => b[108..110].copy_from_slice(&d),
            14 => b[208..210].copy_from_slice(&d),
            16..=19 | 21 | 22 => b[..2].copy_from_slice(&d),
            29 => {
                // the binary16 scale sits in the top nibbles of the four 16-bit scale words
                let bits = f16::from_f32(0.001 + (rng.next() % 100) as f32 / 4000.0).to_bits();
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let word = u16::from_le_bytes([b[at], b[at + 1]]) & 0x0fff | (((bits >> (4 * i)) & 15) << 12);
                    b[at..at + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
            39 => b[0] = 118 + (rng.next() % 16) as u8,
            40 => {
                // four E4M3 scales (0.25 .. 0.9), then the nibbles
                for x in &mut b[..4] {
                    *x = 0x28 + (rng.next() % 16) as u8;
                }
            }
            0x8000_0010..=0x8000_0012 | 0x8000_0014..=0x8000_0016 => {
                // group / 2 code bytes, fp16 scale, zero point
                b[block / 2..block / 2 + 2].copy_from_slice(&d);
                b[block / 2 + 2] = (rng.next() % 16) as u8;
            }
            _ => unreachable!(),
        }
        bytes.extend(b);
    }
    if matches!(kind, 0x8000_0014..=0x8000_0016) {
        let mut perm: Vec<u32> = (0..cols as u32).collect();
        for i in (1..cols).rev() {
            let j = (rng.next() as usize) % (i + 1);
            perm.swap(i, j);
        }
        for p in perm {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
    }
    bytes
}

/// f64 reference of a packed projection against the activation records the kernels consume
/// (`cpu_quantized::prepare` output for every token). Returns outputs and error scales, both
/// indexed `token * rows + row`. Q4_1/Q5_1 apply their offset through the fp16 record sum,
/// exactly as the kernels do.
pub(crate) fn reference_packed(
    weights: &[u8],
    kind: u32,
    rows: usize,
    cols: usize,
    packed: &[X],
) -> (Vec<f64>, Vec<f64>) {
    let (block, width) = layout(kind);
    let row_bytes = cols / block * width;
    let nb = cols / 32;
    let batch = packed.len() / nb;
    let affine = matches!(kind, 3 | 7);
    let is_q4zp = matches!(kind, 0x8000_0014..=0x8000_0016);
    let perm = if is_q4zp {
        let perm_bytes = &weights[rows * row_bytes..];
        Some((0..cols).map(|c| u32::from_le_bytes(perm_bytes[c * 4..(c + 1) * 4].try_into().unwrap()) as usize).collect::<Vec<_>>())
    } else {
        None
    };
    let dequant_kind = match kind {
        0x8000_0014 => 0x8000_0010,
        0x8000_0015 => 0x8000_0011,
        0x8000_0016 => 0x8000_0012,
        other => other,
    };
    let dequantized = (0..rows)
        .map(|r| {
            if fllm_gguf_probe::fp8m_type(kind) {
                let s_bytes = [weights[rows * cols + r * 2], weights[rows * cols + r * 2 + 1]];
                let scale = f16::from_le_bytes(s_bytes).to_f32();
                let r_w = &weights[r * cols..(r + 1) * cols];
                let row = r_w.iter().map(|&b| {
                    let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
                    let exp = (b >> 3) & 0x0f;
                    let mant = b & 0x07;
                    let f = if exp == 0 {
                        sign * (mant as f32) * 2f32.powi(-9)
                    } else if exp == 0x0f && mant == 0x07 {
                        f32::NAN
                    } else {
                        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
                    };
                    f * scale
                }).collect::<Vec<_>>();
                return row;
            }
            let mut row = crate::compute::dequantize(&weights[r * row_bytes..(r + 1) * row_bytes], dequant_kind, cols)
                .expect("reference dequantization");
            if let Some(p) = &perm {
                let permuted = row.clone();
                for c in 0..cols {
                    row[p[c]] = permuted[c];
                }
            }
            row
        })
        .collect::<Vec<_>>();
    let mut want = vec![0.0; batch * rows];
    let mut scale = vec![0.0; batch * rows];
    for t in 0..batch {
        for (r, row) in dequantized.iter().enumerate() {
            let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
            for (b, record) in packed[t * nb..(t + 1) * nb].iter().enumerate() {
                let offset = if affine {
                    let at = r * row_bytes + b * width + 2;
                    f64::from(f16::from_le_bytes([weights[at], weights[at + 1]]).to_f32())
                } else {
                    0.0
                };
                for (&w, &q) in row[b * 32..b * 32 + 32].iter().zip(&record.q) {
                    let term = (f64::from(w) - offset) * f64::from(q) * f64::from(record.d);
                    sum += term;
                    magnitude += term.abs();
                }
                if affine {
                    let term = offset * f64::from(record.sum[0]);
                    sum += term;
                    magnitude += term.abs();
                }
            }
            want[t * rows + r] = sum;
            scale[t * rows + r] = magnitude;
        }
    }
    (want, scale)
}

/// f64 reference of a Q8_0 projection, quantizing `x` exactly like the scalar Q8_0 packing.
pub(crate) fn reference_q8(
    weights: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let nb = cols / 32;
    let batch = x.len() / cols;
    let mut want = vec![0.0; batch * rows];
    let mut scale = vec![0.0; batch * rows];
    for t in 0..batch {
        let blocks = x[t * cols..(t + 1) * cols]
            .as_chunks::<32>()
            .0
            .iter()
            .map(|src| {
                let amax = src.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let inverse = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                let d = f16::from_f32(amax / 127.0).to_f32();
                (src.map(|v| (v * inverse).round_ties_even() as i8), d)
            })
            .collect::<Vec<_>>();
        for r in 0..rows {
            let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
            for (kb, (q, d)) in blocks.iter().enumerate() {
                let b = &weights[(r * nb + kb) * 34..][..34];
                let dw = f64::from(f16::from_le_bytes([b[0], b[1]]).to_f32());
                for (&w, &q) in b[2..].iter().zip(q) {
                    let term = dw * f64::from(w as i8) * f64::from(q) * f64::from(*d);
                    sum += term;
                    magnitude += term.abs();
                }
            }
            want[t * rows + r] = sum;
            scale[t * rows + r] = magnitude;
        }
    }
    (want, scale)
}

/// Largest error relative to its scale, and whether every output is within `TOLERANCE`.
pub(crate) fn compare(have: &[f32], want: &[f64], scale: &[f64]) -> (f64, bool) {
    assert_eq!(have.len(), want.len());
    let mut worst = 0.0f64;
    let mut ok = true;
    for ((&have, &want), &scale) in have.iter().zip(want).zip(scale) {
        let error = (f64::from(have) - want).abs();
        ok &= error <= TOLERANCE * scale + 1e-6;
        worst = worst.max(error / scale.max(1e-30));
    }
    (worst, ok)
}

/// Scalar f64 attention over the same paged caches `cpu_attention::attention` reads, with Q8
/// queries packed like the kernel when the key cache is Q8. Output layout matches `q`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reference_attention(
    q: &[f32],
    key: &crate::cpu_attention::Cache<'_>,
    value: &crate::cpu_attention::Cache<'_>,
    tables: &[i64],
    lengths: &[i64],
    heads: usize,
    block_size: usize,
    stride: usize,
) -> (Vec<f64>, Vec<f64>) {
    let dim = key.dim;
    let kv_heads = key.kv_heads;
    let g = heads / kv_heads;
    let decode = |cache: &crate::cpu_attention::Cache<'_>, slot: usize, h: usize| -> Vec<f64> {
        let start = (slot * kv_heads + h) * dim;
        if !cache.q8 {
            return cache.data[start..start + dim]
                .iter()
                .map(|&bits| f64::from(f16::from_bits(bits).to_f32()))
                .collect();
        }
        let count = cache.slots * kv_heads * dim;
        let scales = count.div_ceil(2) + (slot * kv_heads + h) * (dim / 32);
        (0..dim)
            .map(|i| {
                let word = cache.data[(start + i) / 2];
                let byte = if (start + i).is_multiple_of(2) {
                    word as u8
                } else {
                    (word >> 8) as u8
                };
                let d = f16::from_bits(cache.data[scales + i / 32]).to_f32();
                f64::from(byte as i8) * f64::from(d)
            })
            .collect()
    };
    let mut want = vec![0.0; q.len()];
    let mut magnitude = vec![0.0; q.len()];
    let scale = 1.0 / (dim as f64).sqrt();
    for (b, &length) in lengths.iter().enumerate() {
        for h in 0..heads {
            let query = &q[(b * heads + h) * dim..][..dim];
            let query = if key.q8 {
                query
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .flat_map(|x| {
                        let amax = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
                        let d = amax / 127.0;
                        let inverse = if d == 0.0 { 0.0 } else { d.recip() };
                        let stored = f16::from_f32(d).to_f32();
                        x.map(|v| {
                            f64::from((v * inverse).round_ties_even() as i8) * f64::from(stored)
                        })
                    })
                    .collect::<Vec<_>>()
            } else {
                query.iter().map(|&v| f64::from(v)).collect()
            };
            let slots = (0..length as usize)
                .map(|t| tables[b * stride + t / block_size] as usize * block_size + t % block_size)
                .collect::<Vec<_>>();
            let scores = slots
                .iter()
                .map(|&slot| {
                    let k = decode(key, slot, h / g);
                    k.iter().zip(&query).map(|(a, b)| a * b).sum::<f64>() * scale
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            let out = &mut want[(b * heads + h) * dim..][..dim];
            let mag = &mut magnitude[(b * heads + h) * dim..][..dim];
            for (&slot, &p) in slots.iter().zip(&weights) {
                for (d, v) in decode(value, slot, h / g).into_iter().enumerate() {
                    out[d] += p / total * v;
                    mag[d] += (p / total * v).abs();
                }
            }
        }
    }
    (want, magnitude)
}

/// A paged KV cache in the layout `cpu_attention::Cache` reads, from random keys or values.
/// Q8 groups are encoded with the scalar encoder.
pub(crate) fn random_cache(
    slots: usize,
    kv_heads: usize,
    dim: usize,
    q8: bool,
    rng: &mut Lcg,
) -> Vec<u16> {
    let count = slots * kv_heads * dim;
    let values = rng.vector(count, 2.0);
    if !q8 {
        return values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    }
    let mut bytes = vec![0u8; count];
    let mut scales = vec![0u16; count / 32];
    for ((src, dst), scale) in values
        .as_chunks::<32>()
        .0
        .iter()
        .zip(bytes.as_chunks_mut::<32>().0.iter_mut())
        .zip(&mut scales)
    {
        cpu_kernels::encode_q8_cache_group_scalar(src, dst, scale);
    }
    let mut words = bytes
        .chunks(2)
        .map(|pair| u16::from(pair[0]) | u16::from(*pair.get(1).unwrap_or(&0)) << 8)
        .collect::<Vec<_>>();
    words.extend(scales);
    words
}

struct Report {
    checks: usize,
    failed: usize,
    skipped: usize,
}
impl Report {
    fn line(&mut self, tier: CpuTier, name: &str, result: Option<(f64, bool)>, millis: f64) {
        match result {
            None => {
                self.skipped += 1;
                println!(
                    "[{:<12}] {name:<44} n/a (no kernel at this tier)",
                    tier.name()
                );
            }
            Some((error, ok)) => {
                self.checks += 1;
                self.failed += usize::from(!ok);
                println!(
                    "[{:<12}] {name:<44} max err {error:9.2e}  {millis:8.3} ms  {}",
                    tier.name(),
                    if ok { "ok" } else { "FAIL" }
                );
            }
        }
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_secs_f64() * 1e3)
}

fn records_equal(a: &[X], b: &[X]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.q == b.q
                && a.d.to_bits() == b.d.to_bits()
                && a.init == b.init
                && a.sum.map(f32::to_bits) == b.sum.map(f32::to_bits)
        })
}

/// Packed activations under the scalar tier, the reference every SIMD packing must match.
fn scalar_records(x: &[f32], cols: usize, kind: u32, tier: CpuTier) -> Vec<X> {
    cpu_kernels::set_tier_cap(Some(CpuTier::Scalar));
    let records = crate::cpu_quantized::prepare(x, cols, kind);
    cpu_kernels::set_tier_cap(Some(tier));
    records
}

/// Two calls of the DeltaNet recurrence (`n` tokens, then one more) on deterministic data at
/// `tier`; returns the outputs, the convolution history and the state.
fn delta_pass(
    tier: CpuTier,
    (kh, vh, hk, hv, tiled): (usize, usize, usize, usize, bool),
    n: usize,
) -> Vec<Vec<f32>> {
    cpu_kernels::set_tier_cap(Some(tier));
    let mut rng = Lcg(0xde17a ^ (n * 131 + hv) as u64);
    let ksz = 4;
    let (conv_dim, value_dim) = (2 * kh * hk + vh * hv, vh * hv);
    let conv = rng.vector(conv_dim * ksz, 0.6);
    let dt_bias = rng.vector(vh, 1.0);
    let decay: Vec<f32> = (0..vh).map(|_| -(0.3 + rng.unit().abs() * 2.0)).collect();
    let norm: Vec<f32> = (0..hv).map(|_| 1.0 + 0.25 * rng.unit()).collect();
    let layer = crate::cpu_delta::Layer {
        k_heads: kh,
        v_heads: vh,
        k_dim: hk,
        v_dim: hv,
        conv_kernel: ksz,
        beta_scale: 1.0,
        eps: 1e-6,
        tiled_heads: tiled,
        conv: &conv,
        dt_bias: &dt_bias,
        decay: &decay,
        norm: &norm,
    };
    let mut history = vec![0.0f32; conv_dim * (ksz - 1)];
    let mut state = vec![0.0f32; vh * hk * hv];
    let mut outputs = Vec::new();
    for tokens in [n, 1] {
        let qkv = rng.vector(tokens * conv_dim, 1.5);
        let z = rng.vector(tokens * value_dim, 2.0);
        let beta = rng.vector(tokens * vh, 2.0);
        let alpha = rng.vector(tokens * vh, 2.0);
        let mut out = vec![0.0f32; tokens * value_dim];
        crate::cpu_delta::run(
            &layer,
            &mut history,
            &mut state,
            (&qkv, &z, &beta, &alpha),
            &mut out,
            tokens,
        );
        outputs.push(out);
    }
    outputs.push(history);
    outputs.push(state);
    outputs
}

fn check_tier(tier: CpuTier, report: &mut Report) {
    cpu_kernels::set_tier_cap(Some(tier));
    let mut rng = Lcg(0x5e1f_7e57 ^ tier.name().len() as u64);

    // Activation packing: Q8_0, Q8_1 and Q8_K records, bit for bit against the scalar code.
    for (kind, label) in [(2, "q8_0"), (3, "q8_1"), (12, "q8_k")] {
        let x = rng.vector(5 * 768, 3.0);
        let (records, millis) = timed(|| crate::cpu_quantized::prepare(&x, 768, kind));
        let same = records_equal(&records, &scalar_records(&x, 768, kind, tier));
        report.line(
            tier,
            &format!("pack {label} records (5 x 768)"),
            Some((if same { 0.0 } else { f64::INFINITY }, same)),
            millis,
        );
    }

    // Packed projections: odd block counts, rows that are not a multiple of 4 or 8 (13) and full
    // panels (16), batches around the token-pair and prompt-panel boundaries (the prompt kernels
    // repack the weights of a panel once and run over all tokens from batch 4 on).
    for (kind, rows) in [8, 2, 3, 6, 7, 20, 39, 10, 11, 12, 13, 14, 23]
        .into_iter()
        .flat_map(|kind| [(kind, 13), (kind, 16)])
    {
        let cols = if layout(kind).0 == 256 { 768 } else { 352 };
        let weights = random_weights(kind, rows, cols, &mut rng);
        for batch in [1, 2, 3, 5, 17, 64] {
            let x = rng.vector(batch * cols, 3.0);
            let name = format!(
                "{} rows {rows} cols {cols} batch {batch}",
                format_name(kind)
            );
            let (have, millis) = timed(|| {
                if kind == 8 {
                    cpu_kernels::q8_linear(&weights, &x, rows, cols, None)
                } else {
                    crate::cpu_quantized::linear(&weights, kind, &x, rows, cols, None)
                }
            });
            let result = have.map(|have| {
                let (want, scale) = if kind == 8 {
                    reference_q8(&weights, rows, cols, &x)
                } else {
                    let records = scalar_records(&x, cols, kind, tier);
                    reference_packed(&weights, kind, rows, cols, &records)
                };
                compare(&have, &want, &scale)
            });
            report.line(tier, &name, result, millis);
        }
    }

    // Rows longer than one prompt-panel chunk (nine 256-value blocks, the chunk holds eight).
    for kind in [10, 11, 12, 13, 14] {
        let (rows, cols, batch) = (9, 2304, 7);
        let weights = random_weights(kind, rows, cols, &mut rng);
        let x = rng.vector(batch * cols, 3.0);
        let (have, millis) =
            timed(|| crate::cpu_quantized::linear(&weights, kind, &x, rows, cols, None));
        let result = have.map(|have| {
            let records = scalar_records(&x, cols, kind, tier);
            let (want, scale) = reference_packed(&weights, kind, rows, cols, &records);
            compare(&have, &want, &scale)
        });
        report.line(
            tier,
            &format!(
                "{} rows {rows} cols {cols} batch {batch}",
                format_name(kind)
            ),
            result,
            millis,
        );
    }

    // Q/K/V-like groups mixing K formats (Q4_K, Q6_K, Q5_K; then Q2_K, Q3_K and IQ4_XS with
    // Q4_K and Q6_K) with one activation packing.
    for (kinds, rows) in [
        (vec![12, 14, 13], vec![13, 5, 9]),
        (vec![10, 11, 12, 14, 23], vec![7, 16, 5, 9, 13]),
    ] {
        let cols = 768;
        let weights = kinds
            .iter()
            .zip(&rows)
            .map(|(&kind, &rows)| random_weights(kind, rows, cols, &mut rng))
            .collect::<Vec<_>>();
        let names = kinds
            .iter()
            .map(|&kind| format_name(kind))
            .collect::<Vec<_>>()
            .join("/");
        for batch in [1, 3, 17] {
            let x = rng.vector(batch * cols, 3.0);
            let members = weights
                .iter()
                .zip(&rows)
                .map(|(weights, &rows)| crate::cpu_quantized::Projection {
                    weights,
                    rows,
                    bias: None,
                })
                .collect::<Vec<_>>();
            let (have, millis) =
                timed(|| crate::cpu_quantized::linear_mixed(&members, &kinds, &x, cols));
            let result = have.map(|have| {
                let records = scalar_records(&x, cols, 12, tier);
                let mut worst = (0.0f64, true);
                for ((out, &kind), (weights, &rows)) in
                    have.iter().zip(&kinds).zip(weights.iter().zip(&rows))
                {
                    let (want, scale) = reference_packed(weights, kind, rows, cols, &records);
                    let (error, ok) = compare(out, &want, &scale);
                    worst = (worst.0.max(error), worst.1 && ok);
                }
                worst
            });
            report.line(
                tier,
                &format!("mixed {names} group batch {batch}"),
                result,
                millis,
            );
        }
    }

    // Float projections (input rounded to the weight type, as the kernels document). Besides the
    // odd width (77), 96 columns with a prompt of 20 tokens reach the sixteen-row panels and the
    // AMX-BF16 tile kernel (columns multiple of 32).
    for (kind, (rows, cols, batch)) in [0, 1, 30].into_iter().flat_map(|kind| {
        [(kind, (13, 77, 3)), (kind, (33, 520, 7)), (kind, (5, 40, 1)), (kind, (13, 96, 20))]
    }) {
        let w = rng.vector(rows * cols, 1.0);
        let bytes = w
            .iter()
            .flat_map(|&v| match kind {
                0 => v.to_le_bytes().to_vec(),
                1 => f16::from_f32(v).to_le_bytes().to_vec(),
                _ => bf16::from_f32(v).to_le_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        let x = rng.vector(batch * cols, 3.0);
        let (have, millis) = timed(|| crate::cpu_float::linear(&bytes, kind, &x, rows, cols, None));
        let result = have.map(|have| {
            let round = |v: f32| match kind {
                0 => f64::from(v),
                1 => f64::from(f16::from_f32(v).to_f32()),
                _ => f64::from(bf16::from_f32(v).to_f32()),
            };
            let mut want = vec![0.0; batch * rows];
            let mut scale = vec![0.0; batch * rows];
            for (t, x) in x.chunks_exact(cols).enumerate() {
                for (r, w) in w.chunks_exact(cols).enumerate() {
                    for (&w, &x) in w.iter().zip(x) {
                        let term = round(w) * round(x);
                        want[t * rows + r] += term;
                        scale[t * rows + r] += term.abs();
                    }
                }
            }
            compare(&have, &want, &scale)
        });
        report.line(
            tier,
            &format!(
                "{} rows {rows} cols {cols} batch {batch}",
                format_name(kind)
            ),
            result,
            millis,
        );
    }

    // Gated DeltaNet recurrence: output, convolution history and state bit for bit against the
    // scalar tier, for head shapes with a column block that is a multiple of 8, of 4 and neither.
    {
        let mut same = true;
        let mut total_millis = 0.0;
        for (kh, vh, hk, hv, tiled) in [
            (2usize, 4usize, 64usize, 32usize, true),
            (4, 8, 128, 128, false),
            (2, 4, 128, 96, false),
            (2, 2, 16, 36, false),
            (2, 2, 16, 30, true),
        ] {
            for n in [1usize, 2, 7, 37] {
                let shape = (kh, vh, hk, hv, tiled);
                let (have, millis) = timed(|| delta_pass(tier, shape, n));
                total_millis += millis;
                same &= have == delta_pass(CpuTier::Scalar, shape, n);
            }
        }
        cpu_kernels::set_tier_cap(Some(tier));
        report.line(
            tier,
            "deltanet recurrence (vs scalar tier)",
            Some((if same { 0.0 } else { f64::INFINITY }, same)),
            total_millis,
        );
    }

    // Paged decode attention: GQA 8/2, a three-group head (odd block pair), split KV tiles.
    for q8 in [false, true] {
        let (heads, kv_heads, dim, block_size) = (8, 2, 96, 16);
        let lengths = [37i64, 38, 70, 5];
        let stride = 5;
        let slots = 4 * stride * block_size;
        let tables = (0..lengths.len())
            .flat_map(|b| {
                // Rows 0 and 1 share a table so they form one query tile.
                let base = if b == 1 { 0 } else { b * stride };
                (0..stride).map(move |i| (base + i) as i64)
            })
            .collect::<Vec<_>>();
        let keys = random_cache(slots, kv_heads, dim, q8, &mut rng);
        let values = random_cache(slots, kv_heads, dim, q8, &mut rng);
        let key = crate::cpu_attention::Cache {
            data: &keys,
            slots,
            kv_heads,
            dim,
            q8,
        };
        let value = crate::cpu_attention::Cache {
            data: &values,
            ..key
        };
        let q = rng.vector(lengths.len() * heads * dim, 1.0);
        let (have, millis) = timed(|| {
            crate::cpu_attention::attention(
                &q, key, value, &tables, &lengths, heads, block_size, stride,
            )
        });
        let result = have.map(|have| {
            let (want, scale) = reference_attention(
                &q, &key, &value, &tables, &lengths, heads, block_size, stride,
            );
            compare(&have, &want, &scale)
        });
        let name = format!(
            "attention {} cache dim {dim} gqa {heads}/{kv_heads}",
            if q8 { "q8" } else { "f16" }
        );
        report.line(tier, &name, result, millis);
    }

    // Element-wise kernels, lengths with every vector tail.
    {
        let len = 1031;
        let gate = (0..len)
            .map(|i| (i as f32 - 515.0) / 4.0)
            .collect::<Vec<_>>();
        let up = rng.vector(len, 8.0);
        let mut have = gate.clone();
        let ((), millis) = timed(|| crate::cpu_elementwise::swiglu(&mut have, &up));
        let mut want = Vec::with_capacity(len);
        let mut scale = Vec::with_capacity(len);
        for (&g, &u) in gate.iter().zip(&up) {
            let (g, u) = (f64::from(g), f64::from(u));
            let value = g / (1.0 + (-g).exp()) * u;
            want.push(value);
            scale.push(value.abs().max(1e-3));
        }
        report.line(
            tier,
            "swiglu (1031 values)",
            Some(compare(&have, &want, &scale)),
            millis,
        );

        let mut worst = (0.0f64, true);
        let mut total_millis = 0.0;
        for len in 1..=70 {
            let mut scores = (0..len)
                .map(|j| j as f32 * -1.2345 + rng.unit())
                .collect::<Vec<_>>();
            let reference = scores
                .iter()
                .map(|&s| f64::from((s - 0.5).max(-80.0)).exp())
                .collect::<Vec<_>>();
            let (sum, millis) = timed(|| crate::cpu_elementwise::exp_shifted(&mut scores, 0.5));
            total_millis += millis;
            let want_sum = reference.iter().sum::<f64>();
            let (error, ok) = compare(&scores, &reference, &reference);
            let sum_error = (f64::from(sum) - want_sum).abs() / want_sum;
            worst = (
                worst.0.max(error).max(sum_error),
                worst.1 && ok && sum_error <= TOLERANCE,
            );
        }
        report.line(
            tier,
            "exp_shifted (lengths 1..=70)",
            Some(worst),
            total_millis,
        );
    }

    // Q8 KV-cache encoding, byte for byte against the scalar encoder.
    {
        let mut same = true;
        let mut total_millis = 0.0;
        for len in 1..=70 {
            let src = rng.vector(len, 5.0);
            let (mut have, mut want) = (vec![0u8; len], vec![0u8; len]);
            let (mut have_scale, mut want_scale) = (0u16, 0u16);
            let ((), millis) =
                timed(|| cpu_kernels::encode_q8_cache_group(&src, &mut have, &mut have_scale));
            total_millis += millis;
            cpu_kernels::encode_q8_cache_group_scalar(&src, &mut want, &mut want_scale);
            same &= have == want && have_scale == want_scale;
        }
        report.line(
            tier,
            "q8 kv-cache encode (lengths 1..=70)",
            Some((if same { 0.0 } else { f64::INFINITY }, same)),
            total_millis,
        );
    }
}

fn features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut found = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_x86_feature_detected!($name) { found.push($name); })*
            };
        }
        probe!(
            "avx2",
            "fma",
            "f16c",
            "avxvnni",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512vnni",
            "avx512bf16",
            "avx512vbmi",
            "avxvnniint8"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_aarch64_feature_detected!($name) { found.push($name); })*
            };
        }
        probe!("neon", "dotprod", "i8mm", "bf16", "fp16", "fhm", "sve", "sve2");
    }
    found
}

pub const HELP: &str = "usage: fllm-rs selftest-cpu [--tier TIER]
Runs every CPU kernel tier this machine supports (or only TIER) on random data against scalar
references and prints the largest relative error and the duration per kernel. Projections are
compared with the 8-bit activations they consume; packing and KV-cache encoding must match the
scalar code exactly. Exits non-zero when a check exceeds its tolerance.
TIER: scalar, avx2, avx-vnni, avx512bw, avx512-vnni, avx512-bf16, amx (the last two need a build
with the `avx512` feature), neon, neon-dotprod, neon-i8mm, neon-sve
(neon-sve, the SVE Q8_0 decode kernel, is opt-in at run time with FLLM_ARM_SVE=1; this command enables it
to check it on SVE CPUs with vectors of 256 bits or more.)";

/// Entry point of `fllm-rs selftest-cpu`.
pub fn run(args: &[String]) -> Result<(), String> {
    let mut only = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--tier" => {
                i += 1;
                let name = args.get(i).ok_or("missing value for --tier")?;
                only = Some(CpuTier::parse(name).ok_or_else(|| format!("unknown tier {name}"))?);
            }
            other => return Err(format!("unknown argument {other}\n{HELP}")),
        }
        i += 1;
    }
    // The SVE tier is opt-in for normal runs; the self-test is what validates it.
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        any(target_os = "linux", target_os = "android", target_os = "freebsd")
    ))]
    crate::cpu_arm::sve::allow();
    let supported = cpu_kernels::supported_tiers();
    let tiers = match only {
        Some(tier) if !supported.contains(&tier) => {
            return Err(format!("this CPU does not support tier {}", tier.name()));
        }
        Some(tier) => vec![tier],
        None => supported.clone(),
    };
    println!(
        "fllm-rs selftest-cpu: {} ({} threads); features: {}",
        std::env::consts::ARCH,
        rayon::current_num_threads(),
        features().join(" ")
    );
    println!(
        "supported tiers: {}",
        supported
            .iter()
            .map(|t| t.name())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let arm = crate::cpu_arm::features::detect();
    if arm.sve {
        println!(
            "SVE vector length: {} bits (the neon-sve tier needs 256 or more)",
            arm.sve_bytes * 8
        );
    }
    if let Some(line) = crate::cpu_avx512::describe() {
        println!("x86 extensions (avx512 feature): {line}");
    }
    let mut report = Report {
        checks: 0,
        failed: 0,
        skipped: 0,
    };
    for tier in tiers {
        check_tier(tier, &mut report);
    }
    cpu_kernels::set_tier_cap(None);
    println!(
        "{} checks, {} failed, {} without a kernel at their tier",
        report.checks, report.failed, report.skipped
    );
    if report.failed > 0 {
        return Err(format!(
            "{} CPU kernel checks exceeded their tolerance",
            report.failed
        ));
    }
    Ok(())
}
