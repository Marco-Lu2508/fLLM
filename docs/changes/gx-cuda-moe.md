# CUDA mixture-of-experts completion: grouped kernels, block-wise FP8, Q4ZP slicing, CUDA graphs & stream overlap (gx-cuda-moe)

Base branch: `w-cuda-moe` (commit `a41f7f9`).

Test level of all GPU kernels: **L0** (compiles cleanly under `nvcc` for sm_75, sm_80, sm_89, sm_90, sm_120; tests pass in emulation/CPU; Colab GPU test cases included). CPU and converter tests: **L2** (unit tests, clippy, round-trip tests passing).

## Features

### 1. Grouped GEMM for NVFP4, Q4Z, Block-wise FP8, and IQ/K-quants (`moe.cu`, `moe.rs`)
The grouped MoE GEMM (`moe_tc_kernel` / `moe_tc_run`) has been extended to support all target quantization formats natively in a single launch over all experts:
* **NVFP4** (`KIND_NVFP4`, type 40): 4-bit floating point with per-sub-block scales, decoded directly into shared memory.
* **Q4Z (Marlin-MoE)**: AWQ / GPTQ / compressed-tensors int4 groups (`KIND_Q4Z32`, `KIND_Q4Z64`, `KIND_Q4Z128`) decoded inline without host-routed dispatch.
* **Block-wise FP8** (`KIND_FP8B`, `0x8000_0021`) and row-scaled FP8 (`KIND_FP8M`, `0x8000_0020`): decoded inline into tensor cores.
* **K-quants & IQ formats**: `KIND_Q2_K`, `KIND_Q3_K`, `KIND_IQ2_XXS`, `KIND_IQ2_XS`, `KIND_IQ3_XXS`, `KIND_IQ1_S`, `KIND_IQ3_S`, `KIND_IQ2_S`, `KIND_IQ1_M` are now supported directly in the grouped kernel instead of being re-quantized to Q5_K at upload time.
* Added `fllm_moe_grouped_supported` and updated `grouped_kinds_and_shapes` unit test in `rust/engine/src/backend/cuda/moe.rs` covering all new formats.

### 2. Block-wise FP8 Tensor Format (`FP8B`, `0x8000_0021`) (`gguf-probe`, `checkpoint`, `compute`)
* Registered `FP8B` with GGUF type ID `0x8000_0021` in `rust/gguf-probe/src/lib.rs` and `rust/engine/src/backend/mod.rs` (`QuantType::Fp8Block`).
* In `rust/checkpoint/src/convert.rs`, checkpoints with 128x128 blockwise scales (`weight_scale_inv`) are preserved as compact `FP8B` tensors with an appended f16 scale table instead of expanding to F16. Checkpoints with 32x32 tiles continue to expand to F16 as required by existing tests.
* In `rust/engine/src/compute.rs`, implemented `load_rows` and `dequantize_fp8b` for slicing stacked and non-stacked `FP8B` weights.

### 3. Correct Stacked Q4ZP Permutation Table Slicing (`compute.rs`)
* In `rust/engine/src/compute.rs` (`Projection::load`), fixed row slicing for `Q4ZP` weights: the appended permutation table (`cols * 4` bytes) is copied to each sliced expert rather than being dropped or misplaced.

### 4. CUDA Graph Support for MoE Layers (`api.cu`, `rust_decoder.rs`)
* In `rust/engine/cuda/api.cu`, fixed `fllm_cuda_moe_linear`: scratch buffers (`x16`) are pre-allocated and protected against reallocation during CUDA graph capture (`fllm_cuda_graph_capturing(st)`).
* In `rust/engine/src/rust_decoder.rs`, updated `graph_eligible` to allow layers with `dev.moe.is_some()`.

### 5. Shared-Expert Overlap on Secondary Stream (`cuda.rs`, `moe_device.rs`)
* Added `raw_stream_ptr` and `wait_stream` methods to `Stream` trait in `rust/engine/src/backend/mod.rs` and implemented them on `CudaStream` via `fllm_cuda_stream_wait_stream` (`cudaStreamWaitEvent`).
* In `rust/engine/src/rust_decoder/moe_device.rs`, the dense shared expert is dispatched concurrently on `backend_stream2` while routed experts run on the primary stream, followed by stream synchronization.

### 6. Atom-Free Combine & `moebench` Without Weight Aliasing
* Verified deterministic non-atomic token accumulation in `moe_combine2_kernel`.
* In `rust/engine/src/backend/selftest/moe.rs`, updated `moebench` to generate distinct weights across all experts up to the model maximum, eliminating weight aliasing and false L2 cache hits.

## Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `FLLM_CUDA_MOE` | on | `0` / `off` / `false` / `no`: disables device MoE kernels and falls back to host-routed execution. |
| `FLLM_MOE_BENCH_TOKENS` | `1,4,16,64,256,1024,4096` | Token counts timed in `--only moebench`. |
| `FLLM_MOE_BENCH_ITERS` | 20 | Number of timed iterations per stage in `moebench`. |

## THIRD_PARTY Entries

* None added. Extended the existing tensor core grouped GEMM implementation in `moe.cu` and `common.cuh`.

## Tests

* **L0 (CUDA nvcc)**: Compiles cleanly with zero warnings for sm_75, sm_80, sm_89, sm_90, sm_120 (`fllm-cuda-dev:12.8`).
* **L2 (Unit tests)**:
  - `cargo test -p fllm-engine --lib grouped_kinds_and_shapes --features cuda` passes.
  - `cargo test -p fllm-checkpoint` (43 passed).
  - `cargo test -p fllm-gguf-probe` (3 passed).
  - `cargo test -p fllm-engine --lib` (360 passed).
* **Quality Gates**:
  - `cargo clippy --release --workspace --all-targets --locked -- -D warnings`: 0 warnings.
  - `cargo clippy --release --workspace --all-targets --locked --features rust-gpu -- -D warnings`: 0 warnings.
  - CRLF preserved on `rust/checkpoint/src/gguf.rs` and `rust/gguf-probe/src/lib.rs`.
  - Privacy pattern grep: 0 hits.
