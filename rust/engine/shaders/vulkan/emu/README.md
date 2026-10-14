# CPU checks of the quantized-linear shaders (package 2)

No GPU is involved. These scripts check the shader **text** on the CPU; the verification level of the shaders stays **L0** (never run on a Vulkan device).

| script | what it does |
|---|---|
| `check_compile.sh` | compiles all lines of `../variants_quant.list` with `glslc --target-env=vulkan1.2 -O` and validates with `spirv-val`; with `GLSLANG=<glslang >= 16>` also the hardware dot-product flavour of the `_i` shaders |
| `check_decoders.sh` | turns the chunk decoders `wchunk()` of `../_q_fmt_*.glsl` into C++ (sed) and compares `ws * q + wm` for every weight with `compute::dequantize` (exact) |
| `run_kernels.sh` | turns the whole shaders (`mv_*`, `mm_*`, `quant_act_q8*`, `quant_permute`) into C++ (cpp + `transform.sed` + `shim.h`), runs every workgroup with one thread per invocation and `std::barrier` as `barrier()`, and compares the outputs with an f64 reference (bias, scale, `row_base`, `out_stride`, 2-D workgroup grid, Q4ZP permutation; cooperative matrices are emulated per invocation with f16 rounding) |

Create the reference dumps first (Rust test, needs the `vulkan` feature only for the module; glslc is needed by `build.rs`):

```sh
FLLM_VK_DUMP_DIR=/tmp/fllm-vk-emu/dump cargo test --release -p fllm-engine --features vulkan -- dump_vectors
rust/engine/shaders/vulkan/emu/check_decoders.sh
rust/engine/shaders/vulkan/emu/run_kernels.sh
```

`EMU_WORK` (default `/tmp/fllm-vk-emu`) is the work directory, `DUMP_DIR` the dump directory (default `$EMU_WORK/dump`), `JOBS` the parallelism of `run_kernels.sh`.
Needs g++ >= 13 (C++20 `<barrier>`), `cpp`, `sed`.

Limits of the emulation: float and integer semantics are those of C++ (GLSL leaves some cases undefined or implementation-defined: denormal flushing, `roundEven` of ties,
out-of-range shifts), the memory model and subgroup lock-step execution are not modelled (`subgroupBarrier` is a workgroup barrier), and driver/compiler behaviour
(register allocation, specialization-constant folding, shared-memory bank conflicts, the integer-dot hardware path `OpSDot`) is not exercised.
