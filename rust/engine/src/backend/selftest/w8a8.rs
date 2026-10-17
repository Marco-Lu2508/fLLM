//! Self-test cases: FP8 / int8 W8A8 GEMMs against the f64 reference (CUDA backend). Group `w8a8`.
//!
//! Every product has two references. The activations are quantized by a host copy of the kernel's recipe (E4M3 with the scale amax / 448, or int8 with
//! amax / 127, round to nearest even, IEEE division) and the products of the quantized operands are summed in f64: the device must agree with that
//! up to the accumulation order of the tensor cores (FP8: the tensor cores add the products of a k-step of 32 with a limited alignment width, 1e-3 of the
//! sum of the absolute terms; int8: the sums are exact integers, 1e-5). The FP8 engine cases are also compared with the unquantized product against the
//! worst-case quantization error of the activations, `sum |w| (|x| / 16 + s / 1024)` per output (the half spacing of E4M3 and the step of its subnormals),
//! so that a wrong scale or a skipped quantization shows up as a bound violation and not only as a mismatch with the emulation.
//!
//! The weight layouts are chosen at upload time: the cases switch `FLLM_FP8_GEMM` / `FLLM_INT8_GEMM` at run time and restore the environment's value.
//! `FLLM_SELFTEST_BENCH=1` adds large shapes that time the W8A8 kernels against the W8A16 / FP16 paths they replace.
use super::*;
use crate::backend::cuda::hopper_fp8::{direct_gemm, set_fp8_gemm_mode, set_int8_gemm_mode, Direct};
use crate::kv_format::{e4m3_to_f32, f32_to_e4m3};

/// Tolerance of the FP8 products against the emulation (relative to the sum of the absolute terms).
const FP8_TOL: f64 = 1e-3;
/// The same for the int8 products.
const INT8_TOL: f64 = 1e-5;

fn e4m3_round(v: f32) -> f64 {
    f64::from(e4m3_to_f32(f32_to_e4m3(v)))
}

/// The quantize-dequantize of one token for the FP8 kernels: one scale per `group` values (dynamic, amax / 448), or the static `scale`. Also returns the
/// per-element bound of the quantization error.
fn fp8_activations(row: &[f32], group: usize, static_scale: Option<f32>) -> (Vec<f64>, Vec<f64>) {
    let mut deq = Vec::with_capacity(row.len());
    let mut bound = Vec::with_capacity(row.len());
    for chunk in row.chunks(group) {
        let s = match static_scale {
            Some(s) => s,
            None => {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                if amax > 0.0 {
                    amax / 448.0
                } else {
                    1.0
                }
            }
        };
        let inv = 1.0f32 / s;
        for &v in chunk {
            // saturating conversion, like the kernel's __NV_SATFINITE
            deq.push(e4m3_round(v * inv) * f64::from(s));
            bound.push(f64::from(v.abs()) / 16.0 + f64::from(s) / 1024.0);
        }
    }
    (deq, bound)
}

/// Quantize-dequantize of one token for the int8 kernel: scale amax / 127, round to nearest even.
fn int8_activations(row: &[f32]) -> Vec<f64> {
    let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    row.iter().map(|&v| f64::from((v / d).round_ties_even().clamp(-127.0, 127.0)) * f64::from(d)).collect()
}

/// Activations with tokens of very different magnitude: a scale that is shared between tokens loses the small ones.
fn token_gains(rng: &mut Lcg, batch: usize, cols: usize) -> Vec<f32> {
    let mut x = rng.vector(batch * cols, 1.0);
    for b in 0..batch {
        let gain = [1.0f32, 1e-2, 3.0, 1e-3, 0.3][b % 5];
        for v in &mut x[b * cols..(b + 1) * cols] {
            *v *= gain;
        }
    }
    x
}

fn outcome(have: &[f32], want: &[f64], scale: &[f64], tol: f64, ms: f64) -> Outcome {
    Outcome { device: worst(have, want, scale), cpu: 0.0, tol, cpu_tol: tol, ms }
}

/// Mean error check that complements the maximum: a systematic error of every element is a defect even when each one is below the tolerance.
fn mean_error(have: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    have.iter().zip(want).zip(scale).map(|((&h, &w), &s)| (f64::from(h) - w).abs() / s.max(1e-30)).sum::<f64>() / have.len().max(1) as f64
}

impl Tester<'_> {
    pub(super) fn w8a8_cases(&mut self) {
        if self.device.name() == "CPU" {
            return;
        }
        // FP8M weights through the engine path (upload + quantized_linear): decode and prompt batches, tails of the 16 / 32 / 64 / 128 row tiles,
        // widths of one and several K steps, rows that are not a multiple of the tile
        for (rows, cols) in [(256usize, 128usize), (384, 1024), (200, 640)] {
            for batch in [1usize, 5, 16, 17, 33, 64, 129, 300] {
                let with_bias = batch == 5 || batch == 129;
                let shape = format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" });
                match self.fp8_engine(rows, cols, batch, with_bias) {
                    Ok((emulated, bound)) => {
                        self.record("fp8 w8a8", &shape, Ok(emulated));
                        self.record("fp8 w8a8 bound", &shape, Ok(bound));
                    }
                    Err(e) => self.record("fp8 w8a8", &shape, Err(e)),
                }
            }
        }
        // the kernels on their own: block-wise weights (128 x 128 scales, per-token-group activation scales), a static activation scale, plain int8 weights
        for (kind, label) in [(Direct::Fp8Row, "fp8 row scales"), (Direct::Fp8Block, "fp8 128x128 blocks"), (Direct::Fp8Static, "fp8 static scale"), (Direct::Int8Row, "int8 row scales")] {
            for (rows, cols, batch) in [(256usize, 256usize, 1usize), (320, 384, 7), (200, 512, 40), (384, 1024, 257)] {
                let with_bias = batch == 7;
                let shape = format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" });
                let r = self.w8a8_direct(kind, rows, cols, batch, with_bias);
                self.record(label, &shape, r);
            }
        }
        // Q8_0 weights on the int8 tensor cores against the Q8_0-activation recipe
        for (rows, cols) in [(256usize, 256usize), (384, 1024)] {
            for batch in [16usize, 17, 64, 129] {
                let with_bias = batch == 17;
                let shape = format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" });
                let r = self.int8_q8_engine(rows, cols, batch, with_bias);
                self.record("int8 w8a8 q8_0", &shape, r);
            }
        }
        if std::env::var_os("FLLM_SELFTEST_BENCH").is_some_and(|v| v != "0") {
            self.w8a8_bench();
        }
    }

    /// An FP8M weight uploaded in the W8A8 layout and run through `quantized_linear`: (agreement with the emulation, quantization bound).
    fn fp8_engine(&mut self, rows: usize, cols: usize, batch: usize, with_bias: bool) -> Result<(Outcome, Outcome), BackendError> {
        self.claim = None;
        let kind = 0x8000_0020u32;
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = token_gains(&mut self.rng, batch, cols);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let (mut want_q, mut scale_q) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        let (mut want_x, mut bound) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        for b in 0..batch {
            let row = &x[b * cols..(b + 1) * cols];
            let (deq, err) = fp8_activations(row, cols, None);
            for r in 0..rows {
                let wr = &w[r * cols..(r + 1) * cols];
                let (mut sq, mut mq, mut sx, mut eb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (((&d, &e), &xv), &wf) in deq.iter().zip(&err).zip(row).zip(wr) {
                    let wv = f64::from(wf);
                    sq += d * wv;
                    mq += (d * wv).abs();
                    sx += f64::from(xv) * wv;
                    eb += wv.abs() * e;
                }
                let bv = bias.as_ref().map_or(0.0, |bias| f64::from(bias[r]));
                want_q[b * rows + r] = sq + bv;
                scale_q[b * rows + r] = mq + bv.abs();
                want_x[b * rows + r] = sx + bv;
                // the bound is relative: a unit of `bound` is the worst case of the quantization error
                bound[b * rows + r] = eb.max(1e-30) * 1.001;
            }
        }
        set_fp8_gemm_mode(1);
        let device = self.device;
        let run = || -> Result<(Vec<f32>, f64), BackendError> {
            let weight = device.upload_weight("selftest", &weights, QuantType::Fp8Marlin, rows, cols, bias.as_deref())?;
            if weight.perm.is_none() {
                return Err(BackendError::UnsupportedOperation("the device keeps FP8M weights in the Marlin layout (FP8 tensor cores need sm_89)".into()));
            }
            let mut stream = device.create_stream()?;
            let xb = put(device, stream.as_mut(), &x)?;
            let mut out = device.allocate_buffer(batch * rows * 4)?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.quantized_linear(&mut out, &xb, &weight, weight.bias.as_ref()))?;
            Ok((get(stream.as_mut(), &out, batch * rows)?, ms))
        };
        let result = run();
        set_fp8_gemm_mode(-1);
        let (have, ms) = result?;
        let mean = mean_error(&have, &want_q, &scale_q);
        let mut emulated = outcome(&have, &want_q, &scale_q, FP8_TOL, ms);
        if mean > FP8_TOL / 10.0 {
            // a mean error of a tenth of the maximum tolerance is a systematic offset, not accumulation order
            emulated.device = emulated.device.max(mean * 10.0);
        }
        let in_bound = outcome(&have, &want_x, &bound, 1.0, ms);
        Ok((emulated, in_bound))
    }

    /// One of the kernels called directly with host data.
    fn w8a8_direct(&mut self, kind: Direct, rows: usize, cols: usize, batch: usize, with_bias: bool) -> Result<Outcome, BackendError> {
        self.claim = None;
        if self.device.name() == "CPU" {
            return Err(BackendError::UnsupportedOperation("the CPU backend has no W8A8 kernels".into()));
        }
        let int8 = kind == Direct::Int8Row;
        let weights: Vec<u8> = if int8 {
            (0..rows * cols).map(|_| (self.rng.unit() * 127.0).round() as i8 as u8).collect()
        } else {
            (0..rows * cols).map(|_| f32_to_e4m3(self.rng.unit() * 2.0)).collect()
        };
        let block = kind == Direct::Fp8Block;
        let nscales = if block { rows.div_ceil(128) * (cols / 128) } else { rows };
        let wscale: Vec<f32> = (0..nscales).map(|_| 0.25 + self.rng.unit().abs()).collect();
        let x = token_gains(&mut self.rng, batch, cols);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let static_scale = 0.01f32;
        let wv = |r: usize, k: usize| -> f64 {
            let q = if int8 { f64::from(weights[r * cols + k] as i8) } else { f64::from(e4m3_to_f32(weights[r * cols + k])) };
            let s = if block { wscale[(r / 128) * (cols / 128) + k / 128] } else { wscale[r] };
            q * f64::from(s)
        };
        let (mut want, mut scale) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        for b in 0..batch {
            let row = &x[b * cols..(b + 1) * cols];
            let deq = match kind {
                Direct::Fp8Row => fp8_activations(row, cols, None).0,
                Direct::Fp8Block => fp8_activations(row, 128, None).0,
                Direct::Fp8Static => fp8_activations(row, cols, Some(static_scale)).0,
                Direct::Int8Row => int8_activations(row),
            };
            for r in 0..rows {
                let (mut s, mut m) = (0.0f64, 0.0f64);
                for (k, &d) in deq.iter().enumerate() {
                    let t = d * wv(r, k);
                    s += t;
                    m += t.abs();
                }
                let bv = bias.as_ref().map_or(0.0, |bias| f64::from(bias[r]));
                want[b * rows + r] = s + bv;
                scale[b * rows + r] = m + bv.abs();
            }
        }
        let (have, ms) = direct_gemm(self.device.device_id(), kind, &weights, &wscale, &x, bias.as_deref(), (rows, cols, batch), static_scale, 1)?;
        let tol = if int8 { INT8_TOL } else { FP8_TOL };
        let mean = mean_error(&have, &want, &scale);
        let mut o = outcome(&have, &want, &scale, tol, ms);
        if mean > tol / 10.0 {
            o.device = o.device.max(mean * 10.0);
        }
        Ok(o)
    }

    /// A Q8_0 weight on the int8 tensor cores (`FLLM_INT8_GEMM=1`): the activations are quantized per token and 32 values.
    fn int8_q8_engine(&mut self, rows: usize, cols: usize, batch: usize, with_bias: bool) -> Result<Outcome, BackendError> {
        self.claim = None;
        let kind = 8u32;
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = token_gains(&mut self.rng, batch, cols);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let (mut want, mut scale) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        for b in 0..batch {
            let row = &x[b * cols..(b + 1) * cols];
            let mut deq = Vec::with_capacity(cols);
            for chunk in row.chunks(32) {
                deq.extend(int8_activations(chunk));
            }
            for r in 0..rows {
                let (mut s, mut m) = (0.0f64, 0.0f64);
                for (&d, &wf) in deq.iter().zip(&w[r * cols..(r + 1) * cols]) {
                    let t = d * f64::from(wf);
                    s += t;
                    m += t.abs();
                }
                let bv = bias.as_ref().map_or(0.0, |bias| f64::from(bias[r]));
                want[b * rows + r] = s + bv;
                scale[b * rows + r] = m + bv.abs();
            }
        }
        set_int8_gemm_mode(1);
        let device = self.device;
        let run = || -> Result<(Vec<f32>, f64), BackendError> {
            let weight = device.upload_weight("selftest", &weights, QuantType::Q8_0, rows, cols, bias.as_deref())?;
            let mut stream = device.create_stream()?;
            let xb = put(device, stream.as_mut(), &x)?;
            let mut out = device.allocate_buffer(batch * rows * 4)?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.quantized_linear(&mut out, &xb, &weight, weight.bias.as_ref()))?;
            Ok((get(stream.as_mut(), &out, batch * rows)?, ms))
        };
        let result = run();
        set_int8_gemm_mode(-1);
        let (have, ms) = result?;
        // the FP16 tensor-core kernel (what the engine runs without the switch) reads activations rounded to FP16 and misses this tolerance by an
        // order of magnitude, so passing means the int8 kernel ran
        Ok(outcome(&have, &want, &scale, 2e-5, ms))
    }

    /// Milliseconds per call of `quantized_linear` for `weight`, after one warm-up call (scratch buffers, tile tuning).
    pub(super) fn bench_linear(&mut self, weight: &DeviceWeight, batch: usize, reps: usize) -> Result<f64, BackendError> {
        let device = self.device;
        let x = self.rng.vector(batch * weight.cols, 1.0);
        let mut stream = device.create_stream()?;
        let xb = put(device, stream.as_mut(), &x)?;
        let mut out = device.allocate_buffer(batch * weight.rows * 4)?;
        stream.quantized_linear(&mut out, &xb, weight, None)?;
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..reps {
            stream.quantized_linear(&mut out, &xb, weight, None)?;
        }
        stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64() * 1e3 / reps as f64)
    }

    /// `FLLM_SELFTEST_BENCH=1`: the W8A8 kernels against the paths they replace, on the shapes of a 7B-8B model (attention projection, MLP up and down).
    /// The lines show the time per call and the effective rate; nothing is checked (tolerance 1).
    fn w8a8_bench(&mut self) {
        self.claim = None;
        for (rows, cols) in [(4096usize, 4096usize), (14336, 4096), (4096, 14336)] {
            let fp8 = random_kind(&mut self.rng, 0x8000_0020, rows, cols);
            let q8 = random_kind(&mut self.rng, 8, rows, cols);
            for batch in [1usize, 8, 64, 256, 2048] {
                let ops = 2.0 * rows as f64 * cols as f64 * batch as f64;
                let shape_of = |ms: f64| format!("{rows}x{cols} b{batch}: {:.1} TFLOPS", ops / ms / 1e9);
                for (label, mode) in [("bench fp8 w8a8", 1), ("bench fp8 w8a16", 0)] {
                    set_fp8_gemm_mode(mode);
                    let result = self
                        .device
                        .upload_weight("bench", &fp8, QuantType::Fp8Marlin, rows, cols, None)
                        .and_then(|w| if mode == 1 && w.perm.is_none() { Err(BackendError::UnsupportedOperation("no FP8 tensor cores".into())) } else { self.bench_linear(&w, batch, 20) });
                    set_fp8_gemm_mode(-1);
                    match result {
                        Ok(ms) => self.record(label, &shape_of(ms), Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms })),
                        Err(e) => self.record(label, &format!("{rows}x{cols} b{batch}"), Err(e)),
                    }
                }
                if batch >= 16 {
                    for (label, mode) in [("bench q8_0 int8", 1), ("bench q8_0 fp16", 0)] {
                        set_int8_gemm_mode(mode);
                        let result = self.device.upload_weight("bench", &q8, QuantType::Q8_0, rows, cols, None).and_then(|w| self.bench_linear(&w, batch, 20));
                        set_int8_gemm_mode(-1);
                        match result {
                            Ok(ms) => self.record(label, &shape_of(ms), Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms })),
                            Err(e) => self.record(label, &format!("{rows}x{cols} b{batch}"), Err(e)),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_quantization_error_stays_inside_the_bound() {
        let mut rng = Lcg(11);
        for group in [128usize, 512] {
            let row = token_gains(&mut rng, 1, 512);
            let (deq, bound) = fp8_activations(&row, group, None);
            for ((&v, d), b) in row.iter().zip(&deq).zip(&bound) {
                assert!((f64::from(v) - d).abs() <= *b, "{v} -> {d}, bound {b}");
            }
            // the largest value of every group is represented with the full-range scale: 448 steps of the scale
            for (chunk, dq) in row.chunks(group).zip(deq.chunks(group)) {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let top = dq.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!((top - f64::from(amax)).abs() <= f64::from(amax) / 16.0);
            }
        }
        // a static scale saturates at 448 scales instead of overflowing
        let (deq, _) = fp8_activations(&[1000.0, -1000.0, 0.5], 3, Some(1.0));
        assert_eq!(deq, [448.0, -448.0, 0.5]);
    }

    #[test]
    fn int8_quantization_rounds_to_even_and_keeps_the_extremum() {
        let row = [127.0f32, -63.5, 0.5, 0.0];
        // scale 1: the extremum is 127, 63.5 rounds to the even neighbour 64, 0.5 to 0
        assert_eq!(int8_activations(&row), [127.0, -64.0, 0.0, 0.0]);
        assert_eq!(int8_activations(&[0.0; 4]), [0.0; 4]);
    }
}
