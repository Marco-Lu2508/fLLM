# CPU emulator of the non-matmul MSL kernels (package p06b)

Runs the **real** `.metal` sources of `norm_*`, `rope_*`, `act_*`, `misc_*`, `kv_*`, `embed_*`, `attn_*` on a host without a Mac and compares them with the
engine's CPU backend. It is a test aid, not a Metal implementation: it finds logic errors (index arithmetic, masks, softmax bookkeeping, layouts, planner/kernel
disagreements), not Metal-compiler errors, address-space errors, limits, precision of the GPU or performance. Level of verification stays **L0**.

| file | role |
|---|---|
| `msl_emu.h` | threads as Metal threads: real threadgroup / simdgroup barriers, `simd_sum/max/min/shuffle*/any/all`, `simdgroup_matrix` (8x8, replicated in every lane), attribute tags |
| `gen_glue.py` | generates the dispatch table (kernel name -> call with buffers, constants struct, thread attributes) from the signatures in the MSL text |
| `emu_main.cpp` | runner: reads the cases, executes the plans (grid, threads, constant bytes of the Rust planner) thread by thread, compares |
| `run_emu.sh` | preprocesses the MSL (`kernel`/address spaces erased by the shim of `mslcheck.sh`, kernel `threadgroup` variables made `static`), compiles with clang++, runs |

The cases come from the unit test `emu_fixtures` of the planning layer (`src/backend/metal/ops/mirror.rs`): for each operation and a set of shapes it asks the planners
of `ops.rs` / `ops/` for the dispatches, computes the expected result with the CPU backend (`backend/cpu.rs`; short references for the operations the CPU backend does
not have) and writes everything to `$FLLM_EMU_DIR`. The module `ops` is only built for Apple targets, so on a Linux host it is compiled by a small harness crate that
includes the files by symlink and uses the real engine types through a path dependency:

```
H=~/fllm-tools/p06b-harness ; WT=~/fllm-tools/wt-p06b-metal-ops        # any directories
mkdir -p $H/src/backend/metal
cat > $H/Cargo.toml <<'EOF'
[package]
name = "p06b-harness"
version = "0.0.0"
edition = "2021"
[dependencies]
fllm-engine = { path = "../wt-p06b-metal-ops/rust/engine" }
half = "2"
[workspace]
EOF
cp $WT/Cargo.lock $H/
printf 'pub mod backend {\n    pub use fllm_engine::backend::*;\n    pub mod metal {\n        pub mod ops;\n    }\n}\n' > $H/src/lib.rs
ln -sfn $WT/rust/engine/src/backend/metal/ops.rs $H/src/backend/metal/ops.rs
ln -sfn $WT/rust/engine/src/backend/metal/ops   $H/src/backend/metal/ops
cd $H && cargo test --release                                     # the 25 unit tests of the plans (they compare every plan with the MSL text)
HARNESS=$H bash $WT/rust/engine/metal/emu_ops/run_emu.sh /tmp/fix   # writes the cases with `cargo test emu_fixtures`, compiles the emulator, runs 160 cases
bash $WT/rust/engine/metal/emu_ops/run_emu.sh /tmp/fix prefill_q8   # again with a name filter (reuses the cases)
```

Needs `python3`, a clang++ with C++23 (`CXX=...`) and the syntax shim `msl-shim` of the toolchain package (`SHIM=...`; see `~/fllm-tools/TOOLS.md`).

## Mutation check (package report)

17 hand-made mutations of the kernels (wrong ALiBi offset, `exp(s - m)` instead of `exp(s - new_m)`, sink correction dropped, accumulator rescale of the prompt
kernel skipped, wrong mask comparison, off-by-one in the q8 encoder / the Q6_K decoder, wrong scale index, missing RMSNorm factor, argmax tie order, RoPE pair index,
merge without sink, last value fragment skipped, scale of the q8 writer, GeGLU replaced by SwiGLU, soft cap dropped, sink denominator halved) were each applied to a copy of
the sources and run against the cases: all 17 are caught (one needed an extra case, `decode_single_split_*`, which now exists).
