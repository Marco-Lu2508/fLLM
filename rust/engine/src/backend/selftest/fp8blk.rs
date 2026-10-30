//! Self-test cases: block-wise FP8 weights (`FP8B`, the `weight_scale_inv` checkpoints of DeepSeek-V3 / Qwen3-FP8 / GLM-FP8) on devices with kernels for them
//! (CUDA, Vulkan). Group `fp8blk`.
//!
//! * `fp8b a16`: the kernels that keep the weights 8-bit and expand them in registers (FP32 GEMV for 1 .. 16 tokens, FP16 tensor cores above)
//!   against the f64 evaluation of the dequantized weights and the CPU backend, for every GEMV batch variant, odd shapes (rows that end inside a
//!   tile row, widths that are not a multiple of 16 or of the tile) and the tile shapes the format allows. The FP8 tensor-core path is switched off.
//! * `fp8b memory`: the weight really is 8-bit on the device (the buffer is the GGUF tensor, not an F16 or F32 expansion).
//! * `fp8b w8a8` / `fp8b w8a8 bound` (CUDA only, sm_89 and newer, 128 x 128 tiles): the FP8 tensor-core path through `quantized_linear`. The activations are
//!   quantized by a host copy of the kernel's recipe (one scale per token and 128 values) and the device must agree with the emulation up to the
//!   accumulation order of the tensor cores; the unquantized product must lie within the worst-case quantization error of the activations.
//!
//! The paths are chosen by `fllm_cuda_fp8_gemm_set_mode` (0 never the FP8 tensor cores, 1 always where they exist); the cases restore the default.
#[cfg(feature = "cuda")]
use super::w8a8::{fp8_activations, mean_error, outcome, token_gains, FP8_TOL};
use super::*;
#[cfg(feature = "cuda")]
use crate::backend::cuda::hopper_fp8::{device_cc, set_fp8_gemm_mode};
use fllm_gguf_probe::fp8b_type_id;

/// (the FP8 tensor-core switch exists on CUDA only)
#[cfg(not(feature = "cuda"))]
fn set_fp8_gemm_mode(_mode: i32) {}

impl Tester<'_> {
    pub(super) fn fp8blk_cases(&mut self) {
        if self.device.name() == "CPU" {
            return;
        }
        // (tile rows, tile columns, rows, columns). 128 x 128 is the format of the checkpoints; the others check the scale indexing of the kernels.
        // Widths that are not a multiple of 16 take the GEMV for every batch; a width below 16 per tile (64 x 8) the per-element scale path.
        let shapes: [(usize, usize, usize, usize); 10] = [
            (128, 128, 384, 1024),
            (128, 128, 260, 1000),
            (128, 128, 200, 640),
            (128, 128, 130, 528),
            (64, 64, 192, 512),
            (32, 32, 100, 448),
            (128, 32, 130, 256),
            (1, 128, 64, 384),
            (16, 16, 48, 160),
            (64, 8, 40, 96),
        ];
        set_fp8_gemm_mode(0);
        for (br, bc, rows, cols) in shapes {
            let kind = fp8b_type_id(br, bc).expect("tile shape with a type id");
            for batch in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 33, 129] {
                let with_bias = batch == 3 || batch == 17;
                let r = self.linear(kind, rows, cols, batch, with_bias).map(|mut o| {
                    // above 16 tokens the FP16 tensor-core kernel rounds weights and activations to FP16
                    if batch > 16 {
                        o.tol = o.tol.max(Self::FAST_TOL);
                    }
                    o
                });
                self.record("fp8b a16", &format!("{rows}x{cols} tiles {br}x{bc} batch {batch}{}", if with_bias { " +bias" } else { "" }), r);
            }
        }
        set_fp8_gemm_mode(-1);
        for (br, bc, rows, cols) in [(128usize, 128usize, 384usize, 1024usize), (32, 32, 100, 448)] {
            let r = self.fp8b_memory(fp8b_type_id(br, bc).expect("tile shape with a type id"), rows, cols);
            self.record("fp8b memory", &format!("{rows}x{cols} tiles {br}x{bc}"), r);
        }
        #[cfg(feature = "cuda")]
        for (rows, cols) in [(256usize, 128usize), (384, 1024), (200, 640), (130, 512)] {
            for batch in [17usize, 33, 64, 129, 300] {
                let with_bias = batch == 33 || batch == 129;
                let shape = format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" });
                match self.fp8b_w8a8(rows, cols, batch, with_bias) {
                    Ok((emulated, bound)) => {
                        self.record("fp8b w8a8", &shape, Ok(emulated));
                        self.record("fp8b w8a8 bound", &shape, Ok(bound));
                    }
                    Err(e) => self.record("fp8b w8a8", &shape, Err(e)),
                }
            }
        }
    }

    /// The weight uploaded to the device is the 8-bit tensor: the buffer has the size of the GGUF tensor. A device that expands block-wise FP8 by
    /// design (F16: no kernels for it, the HIP build) reports the check as unsupported; any other layout is a failure. A device with a layout of its
    /// own (Vulkan repeats the scale of a tile for each of its rows) may use more than the GGUF tensor, but never the 16-bit size.
    fn fp8b_memory(&mut self, kind: u32, rows: usize, cols: usize) -> Result<Outcome, BackendError> {
        self.claim = None;
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let weight = self.device.upload_weight("selftest", &weights, QuantType::from_gguf_type_id(kind), rows, cols, None)?;
        if weight.quant_type == QuantType::F16 {
            return Err(BackendError::UnsupportedOperation(format!(
                "the device expands block-wise FP8 to F16 ({} bytes for {} bytes of FP8B)",
                weight.buffer.size,
                weights.len()
            )));
        }
        // a device with a layout of its own (Vulkan: the scale of a tile repeated for each of its rows) keeps the weight 8-bit but not byte for byte
        // either the tensor keeps its own type id, or the backend reports one of the markers it puts
        // on a weight it decodes from the stored bytes (wgpu: high byte 0xF8 for block-wise FP8)
        let id = weight.quant_type.to_gguf_type_id();
        let decoded_in_place = id == kind || (id & 0xf800_0000) == 0xf800_0000;
        if !decoded_in_place || weight.buffer.size >= rows * cols * 2 {
            return Err(BackendError::DriverError(format!(
                "the weight is not kept as FP8B: {:?}, {} bytes on the device for {} bytes of tensor",
                weight.quant_type,
                weight.buffer.size,
                weights.len()
            )));
        }
        Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms: 0.0 })
    }

    /// An FP8B weight with 128 x 128 tiles through `quantized_linear` with the FP8 tensor cores forced on: (agreement with the emulation of the
    /// kernel's activation quantization, the unquantized product within the worst-case quantization error).
    #[cfg(feature = "cuda")]
    fn fp8b_w8a8(&mut self, rows: usize, cols: usize, batch: usize, with_bias: bool) -> Result<(Outcome, Outcome), BackendError> {
        self.claim = None;
        if device_cc(self.device.device_id()) < 89 {
            return Err(BackendError::UnsupportedOperation("the FP8 tensor cores start at sm_89".into()));
        }
        let kind = 0x8000_0021u32;
        let weights = random_kind(&mut self.rng, kind, rows, cols);
        let x = token_gains(&mut self.rng, batch, cols);
        let bias = with_bias.then(|| self.rng.vector(rows, 1.0));
        let w = crate::compute::dequantize(&weights, kind, rows * cols).map_err(BackendError::DriverError)?;
        let (mut want_q, mut scale_q) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        let (mut want_x, mut bound) = (vec![0.0f64; batch * rows], vec![0.0f64; batch * rows]);
        for b in 0..batch {
            let row = &x[b * cols..(b + 1) * cols];
            let (deq, err) = fp8_activations(row, 128, None);
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
                bound[b * rows + r] = eb.max(1e-30) * 1.001;
            }
        }
        set_fp8_gemm_mode(1);
        let device = self.device;
        let run = || -> Result<(Vec<f32>, f64), BackendError> {
            let weight = device.upload_weight("selftest", &weights, QuantType::from_gguf_type_id(kind), rows, cols, bias.as_deref())?;
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
        if crate::backend::cuda::is_exact_math() {
            // Exact mode never takes the FP8 tensor cores (`w8a8_wanted`): the product is the FP32 GEMV on the FP32 activations, so it has to match the
            // unquantized product at the exact tolerance, not the emulation of a path that does not run (that comparison is the fast pass's).
            let mut scale_x = vec![0.0f64; batch * rows];
            for b in 0..batch {
                let row = &x[b * cols..(b + 1) * cols];
                for r in 0..rows {
                    let wr = &w[r * cols..(r + 1) * cols];
                    let mag: f64 = row.iter().zip(wr).map(|(&xv, &wf)| (f64::from(xv) * f64::from(wf)).abs()).sum();
                    scale_x[b * rows + r] = mag + bias.as_ref().map_or(0.0, |bias| f64::from(bias[r]).abs());
                }
            }
            let exact = outcome(&have, &want_x, &scale_x, FP8_TOL, ms);
            let in_bound = outcome(&have, &want_x, &bound, 1.0, ms);
            return Ok((exact, in_bound));
        }
        let mean = mean_error(&have, &want_q, &scale_q);
        let mut emulated = outcome(&have, &want_q, &scale_q, FP8_TOL, ms);
        if mean > FP8_TOL / 10.0 {
            // a mean error of a tenth of the maximum tolerance is a systematic offset, not accumulation order
            emulated.device = emulated.device.max(mean * 10.0);
        }
        let in_bound = outcome(&have, &want_x, &bound, 1.0, ms);
        Ok((emulated, in_bound))
    }
}
