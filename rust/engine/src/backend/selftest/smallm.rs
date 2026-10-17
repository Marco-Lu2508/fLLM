//! Self-test cases for the small-M GEMV of the Marlin 4-bit path (`cuda/marlin_vllm/fllm_marlin_gemv.cu`, `FLLM_MARLIN_SMALL_M`).
//!
//! Two kinds of checks, both on the device only:
//! * the whole linear stack (`quantized_linear` with Q4Z weights of 1 to 4 rows) with the GEMV forced on, against the f64 reference with the
//!   tolerance of the fast kernels;
//! * synthetic Marlin-layout weights (random bytes are a valid weight matrix for both kernels): the GEMV against the Marlin GEMM for the
//!   attention and FFN shapes of 3B to 8B models, including the AWQ (zero points) and GPTQ (symmetric) variants. The line carries the
//!   time per call of both kernels in the shape column so that one run on a real GPU answers whether the GEMV is worth it.
use super::*;

extern "C" {
    fn fllm_marlin_small_m_set(mode: i32);
    fn fllm_marlin_gemv_bench(dev: i32, m: i32, n: i32, k: i32, kind: i32, group: i32, iters: i32, out: *mut f64) -> i32;
    fn fllm_cuda_last_error() -> *const std::ffi::c_char;
}

/// Largest relative difference between the GEMV and Marlin results (of the largest result) that the check accepts: both round to FP16
/// (2^-11) and dequantize in different orders.
const BENCH_TOL: f64 = 5e-3;

impl Tester<'_> {
    pub(super) fn smallm_cases(&mut self) {
        // the full stack, GEMV forced (mode 1); Marlin takes whatever the GEMV declines
        // SAFETY: plain setter of a process-wide switch.
        unsafe { fllm_marlin_small_m_set(1) };
        for kind in self.kinds(&[0x8000_0010u32, 0x8000_0011, 0x8000_0012]) {
            for batch in [1usize, 2, 3, 4] {
                let with_bias = batch == 3;
                let r = self.linear(kind, 384, 1024, batch, with_bias).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(
                    &format!("smallm linear {}", format_name(kind)),
                    &format!("384x1024 batch {batch}{}", if with_bias { " +bias" } else { "" }),
                    r,
                );
            }
        }
        // auto: the first call decides per shape (compares and times both); the result must be right whichever kernel wins
        // SAFETY: as above.
        unsafe { fllm_marlin_small_m_set(2) };
        for kind in self.kinds(&[0x8000_0012u32]) {
            for batch in [1usize, 2, 4] {
                let r = self.linear(kind, 384, 1024, batch, false).map(|mut o| {
                    o.tol = o.tol.max(Self::FAST_TOL);
                    o
                });
                self.record(&format!("smallm auto {}", format_name(kind)), &format!("384x1024 batch {batch}"), r);
            }
        }
        // SAFETY: restores the environment's choice.
        unsafe { fllm_marlin_small_m_set(-1) };

        // GEMV against the Marlin GEMM on synthetic weights; (m, n, k, kind, group): kind 0 AWQ, 1 GPTQ
        let shapes: [(i32, i32, i32, i32, i32); 12] = [
            (1, 2048, 2048, 0, 128), // 3B q_proj / o_proj
            (1, 256, 2048, 0, 128),  // 3B k_proj / v_proj
            (1, 11008, 2048, 0, 128), // 3B gate / up
            (1, 2048, 11008, 0, 128), // 3B down
            (1, 4096, 4096, 0, 128), // 8B q_proj / o_proj
            (1, 14336, 4096, 0, 128), // 8B gate / up
            (1, 4096, 14336, 0, 128), // 8B down
            (2, 2048, 2048, 0, 128),
            (4, 4096, 4096, 1, 128),
            (3, 1024, 1024, 0, 64),
            (1, 1024, 1024, 0, 32),
            (4, 512, 4096, 1, 64),
        ];
        let device = self.device.device_id() as i32;
        for (m, n, k, kind, group) in shapes {
            let mut out = [0f64; 8];
            // SAFETY: `out` has room for the eight values the routine writes.
            let rc = unsafe { fllm_marlin_gemv_bench(device, m, n, k, kind, group, 50, out.as_mut_ptr()) };
            let name = format!("m={m} n={n} k={k} {} g{group}", if kind == 0 { "awq" } else { "gptq" });
            if rc == -1 {
                self.record_skip("smallm gemv", &name, "shape not supported by the GEMV");
                continue;
            }
            let result = if rc != 0 {
                Err(BackendError::DriverError(format!("GEMV benchmark failed ({rc}): {}", last_cuda_error())))
            } else {
                let rel = if out[3] > 0.0 { out[2] / out[3] } else { f64::INFINITY };
                Ok(Outcome { device: rel, cpu: 0.0, tol: BENCH_TOL, cpu_tol: 1.0, ms: out[1] / 1e3 })
            };
            let label = if rc == 0 {
                format!("{name} marlin {:.1} us / nofp32 {:.1} / gemv {:.1} us ({:.0} GB/s)", out[0], out[6], out[1], out[5])
            } else {
                name
            };
            self.record("smallm gemv", &label, result);
        }
        self.marlin_cfg_cases();
    }

    /// `FLLM_MARLIN_PREFILL_CFG=fllm` (the M tail in the last tile) against `vllm` on the same data: the label says how many of the values differ in at least one bit,
    /// which is the evidence for or against making `fllm` the default (it may only become the default when none do).
    pub(super) fn marlin_cfg_cases(&mut self) {
        use crate::backend::cuda::{prefill_switch, set_prefill_switch, PrefillSwitch};
        let saved = prefill_switch(PrefillSwitch::MarlinCfg);
        for kind in self.kinds(&[0x8000_0012u32]) {
            for batch in [65usize, 100, 129, 300, 1000] {
                let (rows, cols) = (384usize, 1024usize);
                self.claim = Some(kind);
                let weights = random_kind(&mut self.rng, kind, rows, cols);
                let x = self.rng.vector(batch * cols, 1.0);
                let mut outputs = Vec::new();
                let mut failure = None;
                for cfg in [0, 1] {
                    set_prefill_switch(PrefillSwitch::MarlinCfg, cfg);
                    let run = (|| -> Result<Vec<f32>, BackendError> {
                        let mut stream = self.device.create_stream()?;
                        let w = self.device.upload_weight("selftest", &weights, QuantType::from_gguf_type_id(kind), rows, cols, None)?;
                        let xb = put(self.device, stream.as_mut(), &x)?;
                        let mut out = self.device.allocate_buffer(batch * rows * 4)?;
                        stream.quantized_linear(&mut out, &xb, &w, None)?;
                        get(stream.as_mut(), &out, batch * rows)
                    })();
                    match run {
                        Ok(v) => outputs.push(v),
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                set_prefill_switch(PrefillSwitch::MarlinCfg, saved);
                let name = format!("{} {rows}x{cols} batch {batch}", format_name(kind));
                if let Some(e) = failure {
                    self.record("smallm marlin cfg", &name, Err(e));
                    continue;
                }
                let differing = outputs[0].iter().zip(&outputs[1]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let scale: Vec<f64> = outputs[0].iter().map(|v| f64::from(*v).abs() + 1e-3).collect();
                let err = worst(&outputs[1], &as_f64_slice(&outputs[0]), &scale);
                let label = format!("{name}: fllm vs vllm, {differing} of {} values differ in bits", outputs[0].len());
                self.record("smallm marlin cfg", &label, Ok(Outcome { device: err, cpu: 0.0, tol: Self::FAST_TOL, cpu_tol: 1.0, ms: 0.0 }));
            }
        }
        set_prefill_switch(PrefillSwitch::MarlinCfg, saved);
    }
}

fn as_f64_slice(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn last_cuda_error() -> String {
    // SAFETY: the library returns a NUL-terminated buffer owned by thread-local storage.
    unsafe { std::ffi::CStr::from_ptr(fllm_cuda_last_error()) }.to_string_lossy().into_owned()
}
