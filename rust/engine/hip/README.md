# HIP additions

New files for the AMD build (package 8): ported vLLM ROCm kernels (Apache-2.0, headers keep their license) and rocWMMA / MFMA code. The existing CUDA sources in `cuda/` are built with hipcc when `FLLM_HIP=1`; `cuda/hip_compat.h` and `cuda/hip_stubs.cu` map and replace the NVIDIA-only parts.

`hip_mx.cuh` (OCP / FNUZ E4M3, E5M2, E8M0, E2M1 in software; the Rust twin is `src/mx_ref.rs`) and `hip_gfx950.cu` (MXFP4 x MXFP8 GEMM on `v_mfma_scale_f32_16x16x128_f8f6f4`, gfx950 only, scalar reference elsewhere, `fllm-rs selftest-gpu` checks it on a gfx950) are described in [docs/gpu-backends.md](../../../docs/gpu-backends.md); level L0.
