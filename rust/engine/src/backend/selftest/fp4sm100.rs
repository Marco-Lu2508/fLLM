//! Self-test cases: NVFP4 W4A4 on datacenter Blackwell (sm_100 / sm_103 / sm_110, CUTLASS tcgen05 GEMM, cuda/nvfp4_cutlass_sm100.cu).
//!
//! The cases are the plain-result W4A4 checks of the `linear` group (`nvfp4_w4a4`: CPU simulation of the activation quantization, weights
//! through `upload_weight_scaled`, two runs that must agree bit for bit) at the batch sizes that select the tiles of the kernel. The group needs
//! `FLLM_NVFP4_SM100=1` (the path is off by default) and a binary built with the sm_100a / sm_103a / sm_110a target of the device; on any other
//! device the weights are not kept as NVFP4 and every case is reported as skipped. `FLLM_SELFTEST_BENCH=1` adds prompt-sized shapes whose
//! device time (the `ms` column) is the figure to compare with the Marlin W4A16 weights (`FLLM_NVFP4_SM100=0`).
use super::*;

impl Tester<'_> {
    pub(super) fn fp4sm100_cases(&mut self) {
        let enabled = std::env::var("FLLM_NVFP4_SM100").is_ok_and(|v| v == "1");
        let mut shapes: Vec<(usize, usize, usize)> =
            vec![(384, 1024, 1), (384, 1024, 8), (384, 1024, 40), (384, 1024, 129), (384, 1024, 356), (768, 2048, 200), (512, 4096, 1024)];
        if std::env::var("FLLM_SELFTEST_BENCH").is_ok_and(|v| !v.is_empty() && v != "0") {
            shapes.extend([(4096, 4096, 512), (4096, 4096, 2048), (14336, 4096, 2048), (4096, 14336, 2048)]);
        }
        for (rows, cols, batch) in shapes {
            let r = if enabled {
                self.nvfp4_w4a4(rows, cols, batch, 1.0, 0)
            } else {
                Err(BackendError::UnsupportedOperation("FLLM_NVFP4_SM100=1 is not set (the sm_100 NVFP4 path is off by default)".into()))
            };
            self.record("nvfp4 sm100", &format!("{rows}x{cols} batch {batch}"), r);
        }
    }
}
