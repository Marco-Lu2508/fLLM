# CPU emulation of the quantized linear kernels (package p06a)

**No GPU and no Metal compiler are involved.** The MSL text of `common.metal`, `common_q_gemm.metal`, `common_q_gemv.metal` and
`q_prepass.metal` is turned into C++ by a mechanical `sed` (the same attribute/literal clean-up as `~/fllm-tools/mslcheck.sh`, plus the function
constant `q_fc_nb` becoming a global that the driver sets), compiled together with `harness.cpp` and **executed on the CPU**:

* `emu_runtime.h` plays the threads of a threadgroup with `std::thread` (64 for the matrix-vector kernels, 128 for the matrix-matrix kernels), a
  `std::barrier` per simdgroup for the lock-step of `simd_sum`, and one per threadgroup for `threadgroup_barrier`; threadgroups run one after the other.
* `metal_stdlib` (found before the shim's own header through `-I`) is the syntax shim of `mslcheck.sh` (types, vectors, math) with working
  `simd_sum`, `threadgroup_barrier` and `simdgroup_matrix` (`simdgroup_load` with `transpose`, `simdgroup_store`, 8x8x8 `simdgroup_multiply_accumulate` with
  float accumulation; every lane holds the whole matrix, lane 0 stores).
* `harness.cpp` reads the vectors dumped by the Rust test `backend::metal::weights::tests::dump_vectors_for_the_msl_emulation`
  (device image as `pack_weight` makes it + the reference dequantization of `compute::dequantize` / `lowbit`, Q4ZP with its permutation; shapes 24 x 256, 72 x 512,
  37 x 2048 (rows not a multiple of any tile), and 24 x 72 for the formats without blocks (K tail of the 32-wide slice)), calls the real kernel templates
  `q_mv_r4_t<F>`, `q_mv_r2_t<F>`, `q_mm_t<F, half>`, `q_mm_t<F, float>` and `fllm_q_permute_x` the way the grid would, and compares every output
  with an f64 reference. Checked per format: batch 1, 2, 3, 4, 5, 8 (both row variants), batch 11 with a second grid column, matrix-matrix batch 9, 33, 40
  (half and float tiles); plain output, bias + scale + `row_base`/`out_stride`, and the fused residual; elements outside the written range must stay untouched.
* The weight image is placed right in front of a `PROT_NONE` page (at most 15 bytes of slack when its size is not a multiple of 16): an out-of-bounds read of a kernel is a crash. The code is built with
  `-fsanitize=alignment -fsanitize-trap=alignment`, which traps on every misaligned load of the kernels (checks the alignment promises in `common.metal`).

## Run

```sh
# 1. dump the vectors (needs the host harness of the p06a package, or any build of the unit tests with FLLM_METAL_DUMP_DIR set)
FLLM_METAL_DUMP_DIR=/tmp/fllm-metal-emu/dump cargo test --release <harness> dump_vectors
# 2. build the emulation and run all formats (JOBS in parallel)
rust/engine/metal/emu/run_emu.sh            # or: run_emu.sh q4_0 q6_k ...
```

Environment: `EMU_WORK` (default `/tmp/fllm-metal-emu`), `DUMP_DIR`, `JOBS`, `MSL_SHIM` (directory with `msl_shim.h`, default `~/fllm-tools/tools/msl-shim`),
`CXX` (a clang++ with `_Float16` and `__builtin_elementwise_*`; default the ROCm clang of `~/fllm-tools/tools`). The unit tests of `weights/` run in
the engine crate on Apple targets only (module `backend::metal`); on other hosts they run in a small harness crate that includes the same files.

## Limits of the emulation

It executes the **text** of the kernels with C++ semantics: float and integer behaviour as in C++ (Metal fast-math, denormal flushing, `fma` contraction,
the order of the `simd_sum` additions are not modelled), `half` is `_Float16`, there is no memory model, no divergence, no
register/threadgroup-memory limit, no performance information, and the Metal compiler may reject constructs the shim accepts (see `INTERFACE_QUANT.md`, known weaknesses).
A pass here means: the decoders, the indexing, the tiling and the epilogues compute what `compute::dequantize` defines. It does not mean the kernels run on a GPU.
