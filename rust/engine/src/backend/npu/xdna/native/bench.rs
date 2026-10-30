//! `fllm-rs amd-npu native-bench ROWS COLS BATCH FORMAT [ITERS]`: one projection `y = x . W^T` (`W` is ROWS x COLS, `x` has BATCH rows) on
//! the NPU through the native engine and on the CPU with the engine's own kernels, same random data: times and the largest difference.
//!
//! The first NPU call also packs the weight into bf16 buffers (reported separately); the timed calls are the steady state of a prompt.

use super::engine::XdnaEngine;
use crate::backend::npu::backend::hybrid::LinearOffload;
use crate::backend::npu::openvino::engine::{host_linear, WeightKey};
use crate::backend::QuantType;
use crate::cpu_selftest::{random_weights, Lcg};
use std::time::Instant;

fn kind_of(name: &str) -> Option<u32> {
    Some(match name.to_ascii_lowercase().as_str() {
        "q4_0" => 2,
        "q8_0" => 8,
        "q4_k" => 12,
        "q5_k" => 13,
        "q6_k" => 14,
        _ => return None,
    })
}

pub fn run(rows: usize, cols: usize, batch: usize, format: &str, iters: usize) -> Result<String, String> {
    let kind = kind_of(format).ok_or("format: one of q4_0 q8_0 q4_k q5_k q6_k")?;
    let quant = QuantType::from_gguf_type_id(kind);
    let mut rng = Lcg(42);
    let raw = random_weights(kind, rows, cols, &mut rng);
    let x = rng.vector(batch * cols, 1.0);
    let engine = XdnaEngine::open(0)?;
    let key = WeightKey { id: 1, offset: 0, size: raw.len() };
    engine.register(key, quant, rows, cols, &raw);
    let slot = engine.slot(&key).ok_or("the engine refused this projection (too small, or too much padding: FLLM_XDNA_MIN_ELEMS / FLLM_XDNA_MIN_EFF)")?;
    let started = Instant::now();
    let first = engine.run(&key, &slot, &x, batch).ok_or("the engine took no rows (batch below FLLM_XDNA_MIN_ROWS, or the run failed: see the log)")?;
    let first_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..iters.max(1) {
        let started = Instant::now();
        engine.run(&key, &slot, &x, batch).ok_or("a repeated run failed")?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        total += ms;
    }
    let mean = total / iters.max(1) as f64;
    let started = Instant::now();
    let cpu_fast = host_linear(&raw, quant, rows, cols, &x, None);
    let cpu_ms = started.elapsed().as_secs_f64() * 1e3;
    let done = first.rows_done;
    // without the engine's CPU kernels for this format the reference is a plain f32 product of the dequantized weight (first 16 rows)
    let (reference, checked): (Vec<f32>, usize) = match cpu_fast {
        Some(v) => (v, done),
        None => {
            let checked = done.min(16);
            let w = crate::compute::dequantize(&raw, kind, rows * cols)?;
            let mut v = vec![0f32; checked * rows];
            for i in 0..checked {
                for j in 0..rows {
                    v[i * rows + j] = x[i * cols..(i + 1) * cols].iter().zip(&w[j * cols..(j + 1) * cols]).map(|(a, b)| a * b).sum();
                }
            }
            (v, checked)
        }
    };
    let cpu_text = if checked == done {
        format!("{cpu_ms:.2} ms = {:.0} GFLOPS", 2.0 * batch as f64 * rows as f64 * cols as f64 / (cpu_ms / 1e3) / 1e9)
    } else {
        "n/a (no engine kernel for this format; the reference is a plain f32 product of 16 rows)".to_owned()
    };
    let (mut worst, mut peak) = (0f32, 0f32);
    for (a, b) in first.values.iter().zip(&reference[..checked * rows]) {
        worst = worst.max((a - b).abs());
        peak = peak.max(b.abs());
    }
    let flops = 2.0 * done as f64 * rows as f64 * cols as f64;
    Ok(format!(
        "{rows}x{cols} {format} batch {batch} ({done} rows on the NPU)\n  NPU first call (packs the weight): {first_ms:.1} ms\n  NPU steady: best {best:.2} ms, mean {mean:.2} ms = {:.0} GFLOPS ({:.0} prompt tokens/s through this projection)\n  CPU kernels, {batch} rows: {cpu_text}\n  largest difference {worst:.3e} against results up to {peak:.3e} (relative {:.2e})\n  {}",
        flops / (best / 1e3) / 1e9,
        done as f64 / (best / 1e3),
        worst / peak.max(1e-12),
        engine.summary()
    ))
}
