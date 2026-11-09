//! Dense BF16 / FP16 GEMM through cublasLt loaded at run time (cuda/cublaslt_gemm.cu).
//!
//! The kernel file opens libcublasLt with dlopen when a prompt-sized product of F16 / BF16 weights asks for it and declines (the built-in tensor-core
//! GEMM runs) when the library or an entry point is missing, so nothing is linked and no installation is required. `FLLM_CUBLASLT=auto|0|1`:
//! auto (the default) and 1 use it on every device of compute capability 8.0 and newer for 9 tokens and more, 0 never.
//! `FLLM_CUBLASLT_LIB` names the library file to open.
use super::*;

extern "C" {
    fn fllm_cuda_cublaslt_set_mode(mode: c_int) -> c_int;
    fn fllm_dense_skinny_set_mode(mode: c_int);
}

/// Overrides `FLLM_CUDA_DENSE_SKINNY` at run time: -1 environment, 0 never (CUDA-core GEMV or cuBLASLt), 1 always where the shape fits (16-bit weights,
/// one to sixteen tokens, columns a multiple of 32).
pub fn set_dense_skinny_mode(mode: i32) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_dense_skinny_set_mode(mode) };
}

/// Overrides `FLLM_CUBLASLT` at run time: -1 environment, 0 never, 1 always (where the library loads).
pub fn set_cublaslt_mode(mode: i32) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_cublaslt_set_mode(mode) };
}
