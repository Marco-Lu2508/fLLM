//! Architecture report and architecture-specific gating of the CUDA backend: which kernels of this binary a device runs
//! (`fllm-rs devices`), and whether NVFP4 weights use the native datacenter-Blackwell path (cuda/nvfp4_cutlass_sm100.cu,
//! cuda/arch_report.cu).
use std::ffi::{c_char, c_int, CStr};

extern "C" {
    fn fllm_cuda_nvfp4_mma_native(dev: c_int) -> c_int;
    fn fllm_cuda_nvfp4_sm100_native(dev: c_int) -> c_int;
    fn fllm_cuda_fast_paths(dev: c_int, buf: *mut c_char, len: usize) -> c_int;
}

/// The NVFP4 tensor-core kernels (mma.sync FP4, sm_120a / sm_121a) of this binary run on device `device`. A GeForce Blackwell GPU of compute
/// capability 12.0 always has them in a binary with the FP4 target; 12.1 (DGX Spark) only when the toolkit could build sm_121a (CUDA 12.9+).
pub(super) fn nvfp4_mma_native(device: i32) -> bool {
    // SAFETY: plain query.
    unsafe { fllm_cuda_nvfp4_mma_native(device) != 0 }
}

/// NVFP4 weights run natively on this device through the CUTLASS tcgen05 GEMM (sm_100 / sm_103 / sm_110 with `FLLM_NVFP4_SM100=1` and a
/// binary built with the sm_100a / sm_103a / sm_110a target of the device). Always false on other devices.
pub(super) fn nvfp4_sm100_native(device: i32) -> bool {
    // SAFETY: plain query.
    unsafe { fllm_cuda_nvfp4_sm100_native(device) != 0 }
}

/// What this binary does on CUDA device `device`: the cubin the device loads and the fast paths it can use. Empty when the query fails.
/// (`kernels`, `paths`)
pub fn device_report(device: usize) -> (String, Vec<String>) {
    let mut buf = vec![0 as c_char; 4096];
    // SAFETY: `buf` holds 4096 bytes; the function writes a NUL-terminated string within `len`.
    if unsafe { fllm_cuda_fast_paths(device as c_int, buf.as_mut_ptr(), buf.len()) } != 0 {
        return (String::new(), Vec::new());
    }
    // SAFETY: NUL-terminated by the callee (the buffer started zeroed).
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
    parse_report(&text)
}

fn parse_report(text: &str) -> (String, Vec<String>) {
    let mut kernels = String::new();
    let mut paths = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("kernels: ") {
            kernels = rest.trim().to_owned();
        } else if let Some(rest) = line.strip_prefix("paths: ") {
            paths = rest.split("; ").map(str::trim).filter(|p| !p.is_empty() && *p != "none").map(str::to_owned).collect();
        }
    }
    (kernels, paths)
}

#[cfg(test)]
mod tests {
    use super::parse_report;

    #[test]
    fn the_report_splits_into_kernels_and_paths() {
        let (k, p) = parse_report("kernels: sm_100 cubin (runs on sm_103)\npaths: Marlin 4-bit; FlashInfer prompt attention (f16 KV); NVFP4 W4A4 tcgen05, off: FLLM_NVFP4_SM100=1");
        assert_eq!(k, "sm_100 cubin (runs on sm_103)");
        assert_eq!(p, ["Marlin 4-bit", "FlashInfer prompt attention (f16 KV)", "NVFP4 W4A4 tcgen05, off: FLLM_NVFP4_SM100=1"]);
        assert_eq!(parse_report("kernels: none for sm_61\npaths: none"), ("none for sm_61".to_owned(), Vec::new()));
        assert_eq!(parse_report(""), (String::new(), Vec::new()));
    }
}
