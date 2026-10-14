# HIP additions

New files for the AMD build (package 8): ported vLLM ROCm kernels (Apache-2.0, headers keep their license) and rocWMMA / MFMA code. The existing CUDA sources in `cuda/` are built with hipcc when `FLLM_HIP=1`; `cuda/hip_compat.h` and `cuda/hip_stubs.cu` map and replace the NVIDIA-only parts.
