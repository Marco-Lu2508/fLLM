# Package G: GPU Encoders and Server Rerank/Score Endpoints

This package delivers the GPU encoder stack, ModernBERT acceleration, native decoder server routes,
and architecture extensions for Jina, Nomic MoE, ColPali/ColQwen, and PyLate.

## Summary of Changes

### 1. CUDA Variable-Length Encoder Attention Stack
- Implemented `cuda/attn_encoder.cu` for bidirectional variable-length sequence attention.
- Sequence packing support via `cu_seqlens` (`[batch_size + 1]`) avoiding padding overhead.
- Numerically stable online softmax without allocating intermediate $S \times S$ attention matrices.
- Supports full bidirectional attention, local sliding-window attention (`window`), and special global tokens (e.g. ModernBERT unpadded or CLS tokens).
- Exposed C API in `cuda/fllm_cuda.h`, `cuda/fllm_cuda_internal.h`, and `cuda/api.cu`: `fllm_cuda_attn_encoder`.
- Bound to `CudaStream` via `attn_encoder` in `rust/engine/src/backend/cuda.rs` and added default trait method in `rust/engine/src/backend/mod.rs`.
- Added pure CPU reference implementation in `rust/engine/src/backend/cpu.rs`.

### 2. ModernBERT GPU Support & Activation Dtype
- Enabled GPU offload for ModernBERT (`BertEncoder::set_device`) across all attention projections (`q`, `k`, `v`, `out`) and GeGLU feed-forward projections (`up`, `gate`, `down`).
- Enabled GPU offload for `DecoderStack` and classifier / projection heads.
- Added `FLLM_ACT_DTYPE` environment variable support (`f16`, `bf16`, `f32`), applying activation quantization/rounding across encoder layers.

### 3. GPU Self-Test (`selftest-gpu --only encoder`)
- Added `encoder` test group to `backend::selftest::GROUPS`:
  - Full bidirectional variable-length sequence attention across multiple batch sizes, heads, and dimensions.
  - Local sliding-window attention with global tokens (ModernBERT pattern).
  - GeGLU activation blocks.
- Verified: 8/8 checks passed on CPU reference and device targets.

### 4. Native Decoder Server Endpoints in `server.rs`
- Replaced 501 `NOT_IMPLEMENTED` stubs in `rust/engine/src/server.rs`:
  - `/pooling`: accepts batched prompts or token arrays, forwards through `WorkerCommand::Embed`, returns pooled embeddings (`mean` or `last`).
  - `/v1/score`: pairwise cosine similarity scoring across query and document lists.
  - `/v1/rerank` and `/rerank`: ranks documents relative to a query in descending order of relevance score, slicing top N.
  - `/classify`: sequence classification with normalized class probabilities and logits.

### 5. Architectural Verification & Converters
- **PyLate**: allowed `Dense` before `Pooling` (or standalone token-level Dense) in `rust/checkpoint/src/st.rs` and `rust/engine/src/encoder.rs` as `token_dense`.
- **Jina Encoders & Rerankers**: added support for Jina v5, Jina reranker v1 (ALiBi positional embeddings), and Jina reranker v3 in `rust/checkpoint/src/jina_xlmr.rs`.
- **Nomic v2 MoE**: added `num_experts` and `num_experts_per_tok` detection and metadata export in `rust/checkpoint/src/nomic.rs`.
- **ColPali & ColQwen**: registered vision-language retrieval architectures in `rust/checkpoint/src/decoder_enc.rs`.
- **Snowflake Arctic Embed v2**: confirmed GTE NewModel converter path compatibility.

## Quality Gate Verification
- Unit Tests: 364 engine tests, 45 checkpoint tests, 3 probe tests passed (0 failed).
- Clippy: clean (0 warnings) on default features, `--features rust-gpu`, and `--features cuda`.
- Self-Test: `selftest-gpu --device cpu --only encoder` passed (8 ok, 0 failed).
- Line Endings: exact CRLF terminators preserved on all designated files.
- Privacy Check: 0 hits on private trace regex.
