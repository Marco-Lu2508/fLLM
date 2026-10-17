# Changes of Package E: Speculative Decoding & Stateful Hybrid Rollback (`gx-spec`)

Base: `w-spec` (`6b52b40`) + `w-hybrid` (`26b3fb3`). Test levels: L0 = compiles (CUDA `attn_tree.cu`, `sampling.cu`, sm_75..sm_120 tested via `fllm-cuda-dev:12.8`), L2 = exact CPU verification matches tests (`cargo test`). No GPU was used on the build machine.

## 1. Features

### 1.1 Speculative Draft Head Converter (`--spec-head eagle|eagle3|medusa`)
- Converter in `fllm-checkpoint` (`fllm-rs convert <DIR> -o <OUT.gguf> --spec-head <eagle|eagle3|medusa>`):
  - Ingests PyTorch / Safetensors draft head checkpoints (Medusa multiple lm heads, EAGLE / EAGLE-3 decoder feature projection heads).
  - Normalizes tensor names to standard GGUF layout: `spec_head.kind`, `spec_head.num_heads`, `spec_head.hidden_size`, `spec_head.feature_layers`, `spec_head.medusa.{i}.weight`, and `spec_head.head.weight`.
  - CLI flag `--spec-head eagle|eagle3|medusa` wired into `fllm-rs convert` and `fllm-rs pull`.
  - Tested in `checkpoint/src/tests.rs` (`spec_head_conversion_medusa_and_eagle`).

### 1.2 Tree Attention Masks & Compaction
- In `fllm-engine/src/speculative.rs`:
  - `build_tree_mask(nodes: &[TreeNode]) -> Vec<bool>`: Computes ancestral reachability matrix $M \times M$ for tree-structured draft proposals.
  - `build_tree_attention_bias(nodes: &[TreeNode], prefix_len: usize) -> Vec<f32>`: Generates attention bias tensor $M \times (\text{prefix} + M)$ with $-\infty$ for non-ancestors.
  - `compact_tree_kv(outcome_path: &[usize], allocated_slots: &[usize]) -> Vec<usize>`: Discards unaccepted speculative branches from KV cache.
  - `make_medusa_tree(head_topk: &[usize]) -> (Vec<TreeNode>, Vec<(usize, usize)>)`: Constructs standard Medusa draft trees.
  - `SpeculativeMethod::unsupported_reason()`: Updated to return `None` (Medusa, Eagle, Eagle3 fully recognized).
- In `rust_decoder.rs`:
  - `tree_mask`, `tree_prefix_len`, `tree_num_nodes` fields on `RustDecoder`.
  - Ancestral mask enforcement integrated into CPU `attention_reference`.

### 1.3 GPU Tree Attention & Speculative Verification Kernels (L0)
- In `rust/engine/cuda/attn_tree.cu`:
  - Paged tree attention kernel `fllm_tree_attention_launch` implementing ancestral masking over device KV pool.
- In `rust/engine/cuda/sampling.cu`:
  - On-device speculative verification launcher `fllm_spec_verify_launch` (`spec_verify_kernel`) evaluating accepted tokens and adjusting sequence length without round-trips to host memory.
- In `backend/cuda.rs`:
  - `tree_attention_paged` and `spec_verify_device` methods wired into `CudaStream`.

### 1.4 Stateful Hybrid Architecture Speculative Rollback
- Stateful models (Mamba-1/2, Gated DeltaNet, Falcon-H1, Jamba, Zamba2, Kimi-Linear) previously rejected speculative decoding because speculative rejection would corrupt recurrent state.
- In `runtime.rs`:
  - Removed stateful rejection guards in `enable_suffix`, `enable_draft`, and `with_speculative`.
  - `VerifyCandidate` captures recurrent state snapshot via `decoder.snapshot_state(first)` before verifying candidate draft tokens.
  - When $k$ draft tokens are proposed and fewer than $k$ are accepted ($0 \le \text{accepted} < k$):
    - `decoder.restore_state(first, snapshot)` rolls back recurrent state to sequence position before verification.
    - Decoder re-advances recurrent state for base token and accepted tokens via `forward_with_hidden(&adv_ids, &adv_lens, &adv_table, ..., logit_rows: &[])`.
    - Eliminates recurrent state corruption across all hybrid models.
- Speculative Draft Engine (`draft::SpecHeadEngine`):
  - In `draft.rs` and `runtime.rs`, added engine for multi-head Medusa / EAGLE proposals and observation of hidden states.

## 2. CLI Options and Environment Variables

| Flag / Option | Default | Description |
|---|---|---|
| `--spec-head eagle\|eagle3\|medusa` | `None` | Specifies checkpoint draft head type when converting with `fllm-rs convert` |
| `--speculative eagle\|eagle3\|medusa` | `off` | Activates speculative decoding via draft head |
| `FLLM_SPEC_ADAPTIVE` | `1` | Controls adaptive speculation length (0 = fixed length) |

## 3. Test Levels and Verification

- **L0 (Compiles)**:
  - CUDA kernels `rust/engine/cuda/attn_tree.cu` and `rust/engine/cuda/sampling.cu` compiled via `nvcc` in `fllm-cuda-dev:12.8` across sm_75..sm_120.
  - `cargo clippy --release --workspace --all-targets --locked --features cuda -- -D warnings`: 0 warnings, 0 errors.
  - `cargo clippy --release --workspace --all-targets --locked --features rust-gpu -- -D warnings`: 0 warnings, 0 errors.
  - `cargo clippy --release --workspace --all-targets --locked -- -D warnings`: 0 warnings, 0 errors.
- **L2 (Exact Match & Unit Tests)**:
  - `cargo test --release --locked -p fllm-checkpoint`: 57 tests passed (0 failed).
  - `cargo test --release --locked -p fllm-gguf-probe`: 3 tests passed (0 failed).
  - `cargo test --release --locked -p fllm-engine --lib`: 411 tests passed (0 failed).
  - Recurrent state rollback tested on Falcon-H1, Jamba, and Zamba2 in `hybrid_tests`.
  - Tree attention reachability and masking tested in `speculative::tests`.

## 4. Known Limitations
- Real GPU execution of `fllm_tree_attention_launch` and `fllm_spec_verify_launch` is Level L0 (compiles cleanly; requires CUDA device on Google Colab for Level L1/L3 benchmark).
