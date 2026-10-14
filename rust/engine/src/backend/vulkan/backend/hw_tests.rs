//! Smoke tests of the Vulkan backend that need a real device (all `#[ignore]`d; nothing here has ever run).
//!
//! `FLLM_VULKAN_VALIDATE=1 cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::backend::hw_tests -- --ignored --nocapture --test-threads=1`
//!
//! They use device 0 among the usable native Vulkan devices (`FLLM_ALLOW_SOFTWARE_GPU=1` admits lavapipe). `--test-threads=1` is needed because the
//! tests switch the process-wide exact-math flag. The complete operation-by-operation comparison with the CPU backend is `fllm-rs selftest-gpu
//! --device vulkan`; these tests are the quick first look.

use super::{set_exact_math, VulkanBackend};
use crate::backend::vulkan::stream::VulkanStream;
use crate::backend::{Backend, DeviceBuffer, QuantType, Stream};
use half::f16;
use std::sync::Arc;

fn backend() -> VulkanBackend {
    VulkanBackend::new(0).expect("no usable native Vulkan device (FLLM_ALLOW_SOFTWARE_GPU=1 admits software renderers)")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn vector(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    fn byte(&mut self) -> u8 {
        (self.next() * 127.0 + 128.0) as u8
    }
}

fn put(backend: &VulkanBackend, stream: &mut dyn Stream, values: &[f32]) -> DeviceBuffer {
    let mut buffer = backend.allocate_buffer(values.len() * 4).unwrap();
    stream.upload(&mut buffer, &f32_bytes(values)).unwrap();
    buffer
}

fn get(stream: &mut dyn Stream, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    stream.download(&mut bytes, buffer).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// Largest error relative to the sum of the absolute terms (the bound of f32 accumulation).
fn worst(have: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    have.iter()
        .zip(want)
        .zip(scale)
        .map(|((&h, &w), &s)| (f64::from(h) - w).abs() / s.max(1e-6))
        .fold(0.0, f64::max)
}

/// Random blocks of Q4_0 (f16 scale + 16 nibble bytes) or Q8_0 (f16 scale + 32 int8).
fn random_blocks(rng: &mut Lcg, quant: QuantType, blocks: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..blocks {
        out.extend_from_slice(&f16::from_f32(0.02 + 0.01 * rng.next().abs()).to_le_bytes());
        let n = if quant == QuantType::Q4_0 { 16 } else { 32 };
        out.extend((0..n).map(|_| rng.byte()));
    }
    out
}

fn check_linear(quant: QuantType, raw: &[u8], dense: &[f32], rows: usize, cols: usize, batch: usize) {
    let backend = backend();
    let mut stream = backend.create_stream().unwrap();
    let mut rng = Lcg(99);
    let x = rng.vector(batch * cols);
    let bias = rng.vector(rows);
    let weight = backend.upload_weight("hw-test", raw, quant, rows, cols, Some(&bias)).unwrap();
    let x_buf = put(&backend, stream.as_mut(), &x);
    let mut out = backend.allocate_buffer(batch * rows * 4).unwrap();
    stream.quantized_linear(&mut out, &x_buf, &weight, weight.bias.as_ref()).unwrap();
    let have = get(stream.as_mut(), &out, batch * rows);
    let (mut want, mut scale) = (vec![0.0; batch * rows], vec![0.0; batch * rows]);
    for b in 0..batch {
        for r in 0..rows {
            let (mut sum, mut mag) = (f64::from(bias[r]), f64::from(bias[r]).abs());
            for k in 0..cols {
                let term = f64::from(x[b * cols + k]) * f64::from(dense[r * cols + k]);
                sum += term;
                mag += term.abs();
            }
            want[b * rows + r] = sum;
            scale[b * rows + r] = mag;
        }
    }
    let error = worst(&have, &want, &scale);
    eprintln!("{quant:?} {rows}x{cols} batch {batch}: worst error {error:.2e} of the absolute terms");
    assert!(error < 1e-4, "{quant:?} batch {batch}: error {error}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_backend_opens_and_describes_itself() {
    let backend = backend();
    let caps = backend.capabilities();
    eprintln!("{} - {}", backend.name(), backend.context().caps().describe());
    assert!(caps.supports_quant(QuantType::Q4_0) && caps.supports_quant(QuantType::F16));
    assert!(caps.supports_paged_attention && caps.supports_rms_norm);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn buffers_round_trip_and_copy() {
    let backend = backend();
    let mut stream = backend.create_stream().unwrap();
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 100.0).collect();
    let a = put(&backend, stream.as_mut(), &values);
    let mut b = backend.allocate_buffer(values.len() * 4).unwrap();
    stream.copy_buffer(&mut b, &a).unwrap();
    assert_eq!(get(stream.as_mut(), &b, values.len()), values);
    // a slice that starts in the middle (not a multiple of the binding alignment) is usable as an operand too
    let tail = a.slice(4 * 3, 4 * 100).unwrap();
    assert_eq!(get(stream.as_mut(), &tail, 100), values[3..103]);
    stream.zero_buffer(&mut b).unwrap();
    assert!(get(stream.as_mut(), &b, values.len()).iter().all(|&v| v == 0.0));
}

#[test]
#[ignore = "needs a Vulkan device"]
fn float_weights_match_a_plain_product() {
    set_exact_math(true);
    let mut rng = Lcg(5);
    let (rows, cols) = (70, 200);
    let dense = rng.vector(rows * cols);
    for batch in [1usize, 3, 11] {
        check_linear(QuantType::F32, &f32_bytes(&dense), &dense, rows, cols, batch);
        let halves: Vec<f16> = dense.iter().map(|&v| f16::from_f32(v)).collect();
        let raw: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let rounded: Vec<f32> = halves.iter().map(|h| h.to_f32()).collect();
        check_linear(QuantType::F16, &raw, &rounded, rows, cols, batch);
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn quantized_weights_match_the_dequantized_product() {
    set_exact_math(true);
    let mut rng = Lcg(7);
    for quant in [QuantType::Q4_0, QuantType::Q8_0] {
        let (rows, cols) = (65, 256);
        let raw = random_blocks(&mut rng, quant, rows * cols / 32);
        let dense = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).unwrap();
        for batch in [1usize, 4, 9, 40] {
            check_linear(quant, &raw, &dense, rows, cols, batch);
        }
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_decode_step_is_recorded_into_one_submission() {
    set_exact_math(true);
    let backend = backend();
    let mut stream = VulkanStream::new(Arc::clone(&backend.shared)).unwrap();
    let mut rng = Lcg(11);
    let n = 512;
    let x = put(&backend, &mut stream, &rng.vector(n));
    let weight = put(&backend, &mut stream, &rng.vector(n));
    let mut normed = backend.allocate_buffer(n * 4).unwrap();
    let mut act = backend.allocate_buffer(n * 4).unwrap();
    for _ in 0..8 {
        stream.rms_norm(&mut normed, &x, &weight, 1e-6).unwrap();
        stream.swiglu(&mut act, &normed, &x).unwrap();
    }
    assert_eq!(stream.recorder_stats().submits, 0, "nothing is submitted before a result is needed");
    let probe = get(&mut stream, &act, n);
    assert_eq!(
        stream.recorder_stats().submits,
        1,
        "uploads, 16 dispatches and the read-back are one submission"
    );
    assert!(probe.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "needs a Vulkan device"]
fn argmax_picks_the_first_maximum() {
    let backend = backend();
    let mut stream = backend.create_stream().unwrap();
    let vocab = 1000;
    let mut logits = vec![0.0f32; 3 * vocab];
    logits[7] = 2.0;
    logits[vocab + 999] = 5.0;
    logits[2 * vocab + 10] = 1.0;
    logits[2 * vocab + 20] = 1.0;
    let buf = put(&backend, stream.as_mut(), &logits);
    assert_eq!(stream.argmax_rows(&buf, 3, vocab).unwrap(), vec![7, 999, 10]);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn unaligned_outputs_are_staged_and_written_back() {
    set_exact_math(true);
    let backend = backend();
    let mut stream = backend.create_stream().unwrap();
    let mut rng = Lcg(21);
    let n = 64;
    let values = rng.vector(n);
    let x = put(&backend, stream.as_mut(), &values);
    let weight = put(&backend, stream.as_mut(), &vec![1.0; n]);
    let mut whole = backend.allocate_buffer((n + 2) * 4).unwrap();
    stream.zero_buffer(&mut whole).unwrap();
    // starts 4 bytes into the allocation: not a multiple of the 16 bytes the shaders need
    let mut out = whole.slice(4, n * 4).unwrap();
    stream.rms_norm(&mut out, &x, &weight, 1e-6).unwrap();
    let all = get(stream.as_mut(), &whole, n + 2);
    let mean = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / n as f64;
    let scale = 1.0 / (mean + 1e-6).sqrt();
    assert_eq!(all[0], 0.0, "the word in front of the operand must stay untouched");
    assert_eq!(all[n + 1], 0.0, "the word behind the operand must stay untouched");
    for (i, v) in values.iter().enumerate() {
        assert!((f64::from(all[i + 1]) - f64::from(*v) * scale).abs() < 1e-5, "element {i}");
    }
}
