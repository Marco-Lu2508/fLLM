# HIP additions

New files for the AMD build (package 8): ported vLLM ROCm kernels (Apache-2.0, headers keep their license) and rocWMMA / MFMA code. The existing CUDA sources in `cuda/` are built with hipcc when `FLLM_HIP=1`; `cuda/hip_compat.h` and `cuda/hip_stubs.cu` map and replace the NVIDIA-only parts.

Which AMD GPUs these serve, and what each family runs, is in [docs/hip.md](../../../docs/hip.md). The data files of this directory:

* `arch_table.csv`: one row per gfx name (wave size, matrix unit, the kernel set fLLM uses, integer dot product, generic target). `hip_arch.h` (C++, host side, tested by
  `hip_arch_test.cpp`) and `src/backend/hip_arch.rs` (Rust) mirror it; `scripts/hip_arch_check.sh` compares it with the macros the compiler defines for each target.
* `default_archs.txt`: the offload targets of a default build (`linux=` and `windows=`), read by `build.rs` and the CI jobs.
* `hip_device.cu`: the properties of the device, and the check that decides at run time which kernel set the loaded code object may use.
