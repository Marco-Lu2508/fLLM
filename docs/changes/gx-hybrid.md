# Package H: Hybrid State-Space and Linear-Attention Architectures

This package implements GPU kernels, CPU chunked prefill, backend dispatch, selftest suites,
and model converters for modern hybrid linear attention, SSM, and hybrid Transformer/SSM models.

## Summary of Changes

### 1. KDA (Key-Dependent Attention / Recurrent Kernel)
- **CUDA Kernel (`rust/engine/cuda/kda.cu`)**:
  - Implemented fused causal depthwise 1D convolution over projected key/value states with SiLU activation.
  - Per-head L2 normalization with scaling factor $1 / \sqrt{d_{\text{head}}}$.
  - Per-channel recurrence with data-dependent decay and gating: write strength driven by $\sigma(\beta)$.
  - Sigmoid output gating and output RMSNorm.
  - Registered in `rust/engine/build.rs`, with C API bindings in `rust/engine/cuda/fllm_cuda_internal.h`, `rust/engine/cuda/api.cu`, and HIP stubs in `rust/engine/cuda/hip_stubs.cu`.
- **CPU Backend & Chunked Prefill (`rust/engine/src/rust_decoder/kda.rs`, `rust/engine/src/backend/cpu.rs`)**:
  - Chunked prefill processing with fixed chunk size 64 for bounded memory footprint and numerical parity.
  - Recurrent state snapshot management via `LayerStateSnapshot` and `RecurrentState`.
- **Metal & Vulkan**:
  - Added recurrent dispatch planning and `supports_kda` hooks in `rust/engine/src/backend/metal/` and `rust/engine/src/backend/vulkan/`.
- **Selftest Suite (`rust/engine/src/backend/selftest.rs`)**:
  - Added self-test suite for `kda` covering dimension shapes, decay dynamics, and recurrence outputs.
  - 5/5 checks verified passing on CPU backend.

### 2. Hybrid Architectures & Layer Variants
- **Falcon-H1**: Parallel Mamba-2 + self-attention block with mUP scaling and norm-order selection.
- **Zamba2**: Shared transformer block with biased linear layer support and adapter execution.
- **Jamba**: RMSNorm variant configurations, expert layout mappings, and auto time-step rank support.

### 3. Model Architecture Support (`rust/checkpoint/src/plan.rs`, `rust/checkpoint/src/kimi_linear.rs`)
- Registered architecture plans and conversion mappings for:
  - Ling 3.0 / BailingMoeV3 (MLA + KDA)
  - Kimi K3 (with `gate_lower_bound` / `safe_gate_lower_bound`)
  - GLM-5.3-Flash
  - RWKV-6 / RWKV-7
  - xLSTM
  - Granite-4 / Nemotron-H / Bamba / LFM2
  - MiniCPM-SALA
  - JetNemotron
  - HRM-text
  - ZAYA1

## Quality Gate Verification
- **Unit Tests**:
  - `cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`: 362 engine tests, 56 checkpoint tests, 3 probe tests passed (0 failed).
- **Self-Test Suite**:
  - `fllm-rs selftest-gpu --device cpu --only kda`: 5 checks, 0 failed, 5 ok.
- **Clippy Verification**:
  - `cargo clippy --workspace --all-targets --locked -- -D warnings`: Clean (0 warnings).
  - `cargo clippy --workspace --all-targets --locked --features rust-gpu -- -D warnings`: Clean (0 warnings).
  - `cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings`: Clean (0 warnings).
- **File Integrity & CRLF Conformance**:
  - Exact CRLF line terminators preserved on:
    - `rust/engine/src/runtime.rs`
    - `rust/engine/src/draft.rs`
    - `rust/checkpoint/src/gguf.rs`
    - `rust/gguf-probe/src/lib.rs`
    - `rust/checkpoint/src/tests.rs`
    - `rust/checkpoint/src/lib.rs`
- **Privacy Conformance**:
  - 0 private traces across all diffs, patches, bundles, and reports.
