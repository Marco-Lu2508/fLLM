//! Self-test cases: cublasLt dense GEMM against the f64 reference (CUDA backend). Group `hgemm`.
//!
//! F16 and BF16 weights, prompt batches, with the switch forced on (`FLLM_CUBLASLT=1`). Where the library cannot be loaded the kernel declines and the
//! built-in GEMM computes the product, so a pass says the result is right, not which kernel ran: the timing lines of `FLLM_SELFTEST_BENCH=1` (cublasLt
//! against the built-in tensor-core GEMM on the same weights) tell. The activations are rounded to the weight's 16-bit format (F16: 5e-4 per term,
//! BF16: 4e-3), so the fast-kernel tolerance applies.
use super::*;
use crate::backend::cuda::hopper_gemm::set_cublaslt_mode;

impl Tester<'_> {
    pub(super) fn hgemm_cases(&mut self) {
        if self.device.name() == "CPU" {
            return;
        }
        set_cublaslt_mode(1);
        for kind in self.kinds(&[1u32, 30]) {
            for (rows, cols) in [(256usize, 512usize), (384, 1024)] {
                for batch in [16usize, 17, 64, 129, 300] {
                    let with_bias = batch == 17 || batch == 129;
                    let tol = if kind == 30 { 8e-3 } else { Self::FAST_TOL };
                    let r = self.linear(kind, rows, cols, batch, with_bias).map(|mut o| {
                        o.tol = o.tol.max(tol);
                        o
                    });
                    self.record(&format!("cublaslt {}", format_name(kind)), &format!("{rows}x{cols} batch {batch}{}", if with_bias { " +bias" } else { "" }), r);
                }
            }
        }
        set_cublaslt_mode(-1);
        if std::env::var_os("FLLM_SELFTEST_BENCH").is_some_and(|v| v != "0") {
            self.hgemm_bench();
        }
    }

    /// cublasLt against the built-in GEMM for the weights of a 7B-8B model; the lines show the time per call and the rate (nothing is checked).
    fn hgemm_bench(&mut self) {
        self.claim = None;
        for kind in self.kinds(&[1u32, 30]) {
            for (rows, cols) in [(4096usize, 4096usize), (14336, 4096)] {
                let weights = random_kind(&mut self.rng, kind, rows, cols);
                for batch in [16usize, 64, 256, 1024, 4096] {
                    for (label, mode) in [("bench cublaslt", 1), ("bench built-in", 0)] {
                        set_cublaslt_mode(mode);
                        let result = self
                            .device
                            .upload_weight("bench", &weights, QuantType::from_gguf_type_id(kind), rows, cols, None)
                            .and_then(|w| self.bench_linear(&w, batch, 20));
                        set_cublaslt_mode(-1);
                        let tflops = |ms: f64| 2.0 * rows as f64 * cols as f64 * batch as f64 / ms / 1e9;
                        match result {
                            Ok(ms) => self.record(
                                &format!("{label} {}", format_name(kind)),
                                &format!("{rows}x{cols} b{batch}: {:.1} TFLOPS", tflops(ms)),
                                Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms }),
                            ),
                            Err(e) => self.record(&format!("{label} {}", format_name(kind)), &format!("{rows}x{cols} b{batch}"), Err(e)),
                        }
                    }
                }
            }
        }
    }
}
