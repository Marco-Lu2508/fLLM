# CPU emulation of the MoE, gate, DeltaNet and Mamba kernels

Runs the **real kernel text** of `moe_*`, `gate_*`, `delta_*` and `mamba_*` on the CPU (one `std::thread` per GPU thread, a real barrier,
static threadgroup memory) and compares against independent references. This is **not** a test under Metal (level L0, see `../INTERFACE_RECURRENT.md`
section 9 and [test levels](../../../../docs/test-levels.md)); it demonstrates the algorithm, the index arithmetic, the block layouts and the barrier discipline, not acceptance by the MSL compiler or the behaviour of the GPU.

```
# 1. Test vectors for moe_linear (random GGUF blocks of all ten formats + weights from the Rust dequantiser)
cd harness && cp Cargo.toml.example Cargo.toml && cargo run --bin gen_vectors -- /tmp/vectors.bin          # adjust the path of fllm-engine in Cargo.toml / src/lib.rs
# 2. Emulation (g++ >= 12 or clang >= 15, C++20)
./run_emulation.sh /tmp/vectors.bin                                    # exit 0 = all checks passed
TSAN=1 ./run_emulation.sh /tmp/vectors.bin                             # with ThreadSanitizer: missing barriers become data races
# 3. Negative check: deliberately faulty kernels must be detected
TSAN=1 ./mutation_check.sh /tmp/vectors.bin
```

| File | Contents |
|---|---|
| `msl_emu/emu.h`, `msl_emu/metal_stdlib` | the small MSL substitute (address-space macros, barrier, math, `as_type`, `uint2`) |
| `emu_main.cpp` | tests and references (double-precision ports of `mamba_forward`, `mamba1_forward`, `cpu_delta.rs`, `moe_route`) |
| `run_emulation.sh` | rewrites the `.metal` files into C++ with `sed` (attributes removed, `threadgroup` fields static), builds and starts |
| `mutation_check.sh` | 16 deliberate kernel faults; each must be reported by the emulation |
| `harness/` | test crate (`gen_vectors`, planning tests); copy `Cargo.toml.example` to `Cargo.toml`. The paths to `fllm-engine` (Cargo.toml) and to the planning files (`src/lib.rs`) are set relative to the repository and may need adjusting; `cargo run --bin gen_vectors` and `cargo test --lib` |

The rewrite is deliberately small: template instantiation lines (`template [[host_name(...)]] ...`) are deleted (`mslcheck.sh` checks them), the kernels
themselves are called unchanged (the test driver sets grid and threads like a Metal dispatch).
