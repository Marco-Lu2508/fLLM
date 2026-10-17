# Package I: Model Formats, Quantization Encoders, RoPE Extensions

This package extends model architecture converters, RoPE position embeddings, tensor data types,
and GGUF quantization encoders.

## Summary of Changes

### 1. Model Converters & Architecture Plan Mappings (`rust/checkpoint/src/plan.rs`)
- **Ling / BailingMoe**: Dynamic family mapping distinguishing standard GQA MoE (`Family::Qwen`, `arch: bailing_moe`) from MLA variants (`Family::Deepseek`, `arch: deepseek2`) based on presence of `kv_lora_rank`.
- **Baichuan2**: Added support for fused QKV `W_pack` projection and `NormHead`.
- **Qwen1**: Added mapping for unified attention projection `c_attn.weight` to `attn_qkv.weight` alongside projection biases and token embeddings.
- **ChatGLM / GLM-4-9B**: Added mapping for fused `query_key_value` projections and transformer output layer.
- **Additional Architecture Mappings**:
  - DeepSeek-MoE v1
  - EXAONE 3.x
  - MiniCPM (muP parameter scaling)
  - TeleChat2 / TeleFLM
  - Step1 / Step3
  - PLaMo-3
  - Grok-1
  - XverseMoE
  - Cohere2Moe
  - JetMoE
  - GigaChat3.5
  - A.X-K1 / K2
  - HY-V4
  - Pangu
  - Param2MoE
- **MoE Layer Configuration**: Added architecture identifiers to MoE layer routing rules.

### 2. RoPE Extensions (`rust/checkpoint/src/plan.rs`)
- **K2-Horizon Partial Layout**: Support for rotary embeddings where `rope_head_dim < head_dim`, enabling partial rotary dimension configurations.
- **Generic Dynamic NTK**: Support for `dynamic`, `dynamic-ntk`, and `dynamic_ntk` rope scaling algorithms.
- **Llama 4 RoPE**: Support for `llama4` scaling type with attention temperature beta parameter.
- **Proportional RoPE**: Support for `proportional` scaling types.

### 3. Tensor Dtypes & Safetensors Reader (`rust/checkpoint/src/safetensors.rs`)
- Added support for 1-byte data types in safetensors reader:
  - `F8_E8M0`: 8-bit floating point format with 8 exponent bits and 0 mantissa bits, used for MXFP4 microscaling scale factors.
  - `BOOL`: 1-byte boolean mask tensors.

### 4. Quantization Encoders (`rust/checkpoint/src/quantize.rs`, `rust/checkpoint/src/requant.rs`)
- Added `Q2K`, `Q3K`, `Mxfp4`, and `Nvfp4` variants to `QType` and `Preset`:
  - `parse`: Recognizes `q2_k`/`q2k`, `q3_k`/`q3k`, `mxfp4`, and `nvfp4`.
  - `ggml_id`: Maps to GGUF type IDs 10 (`Q2_K`), 11 (`Q3_K`), 39 (`MXFP4`), 40 (`NVFP4`).
  - `block_elements` and `block_bytes`: Proper block dimensions (256 values / 84 bytes for Q2_K; 256 values / 110 bytes for Q3_K; 32 values / 17 bytes for MXFP4; 64 values / 36 bytes for NVFP4).
  - Row quantizers: Reference encoders for `q2_k`, `q3_k`, `mxfp4`, and `nvfp4` with round-trip error tests and zero-preservation tests.
  - Requant preset selection: Full compatibility with `fllm-rs quantize`.

### 5. Chat Template Reproducibility (`rust/engine/src/chat_template.rs`)
- Added thread-safe RAII atomic `TEST_NOW` mechanism in unit tests with environment variable fallback (`FLLM_NOW` and `SOURCE_DATE_EPOCH`) for `strftime_now` filter. Ensures reproducible golden test evaluation across different execution dates without changing test tolerances or golden fixtures.

## Quality Gate Verification
- **Unit Tests**:
  - `fllm-checkpoint`: 66 passed, 0 failed.
  - `fllm-engine`: 378 passed, 0 failed.
  - `fllm-gguf-probe`: 4 passed, 0 failed.
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
    - `rust/checkpoint/src/safetensors.rs`
- **Privacy Conformance**:
  - 0 private traces across all diffs, patches, bundles, and reports.
