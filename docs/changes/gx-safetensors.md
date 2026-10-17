# Package J: Direct Zero-Copy Safetensors Loader and Memory-Mapped Execution

This package introduces direct in-memory safetensors loading and zero-copy mapping into `MappedGguf` and the engine runtime, eliminating intermediate on-disk GGUF conversions while guaranteeing bit-for-bit tensor parity and exact logit equivalence with the conversion pipeline.

## Summary of Changes

### 1. In-Memory Safetensors Zero-Copy Loader (`rust/checkpoint/src/convert.rs`, `rust/checkpoint/src/lib.rs`)
- **`prepare_checkpoint`**: Refactored conversion pipeline into a pure preparation phase returning `(String, Vec<(String, Meta)>, Vec<OutTensor>, Vec<String>)`, decoupling tensor planning/transformation from disk serialization.
- **`materialize_mapped_gguf`**: Constructs an in-memory `MappedGguf` instance:
  - Untouched tensors (`Payload::Mapped`) are mapped directly to underlying safetensors memory-mapped regions with zero copying or reallocation.
  - Transformed, permuted (e.g. RoPE `attn_q`/`attn_k` layout), fused, quantized, or offset tensors are written directly into an anonymous read-only memory map (`MmapOptions::map_anon().make_read_only()`).
  - Assembles unified GGUF index, metadata, alignment, and per-tensor mappings.
- **`load_checkpoint_as_gguf`**: Top-level API exporting the direct checkpoint loader.
- **Split arch prepare methods (`bert.rs`, `modernbert.rs`, `nomic.rs`)**: Exposed `prepare()` returning transformed metadata and tensors for direct loading.

### 2. Multi-Region Memory Mapping in GGUF Probe (`rust/gguf-probe/src/lib.rs`)
- **`MappedGguf::from_index_and_mappings`**: Instantiates `MappedGguf` from arbitrary per-tensor `Arc<Mmap>` slices alongside anonymous memory-mapped buffers without requiring an on-disk GGUF file.
- **Exposed Accessors**: Provided public accessors `extra_mappings()` and `mapping()` on `MappedGguf` for introspection, memory bounds validation, and zero-copy verification.

### 3. Engine Model & Import Integration (`rust/engine/src/model.rs`, `rust/engine/src/import.rs`, `rust/engine/src/bin/serve.rs`)
- **`GgufModel::from_mapping`**: Public constructor enabling engine loading directly from a pre-constructed `MappedGguf` instance.
- **`GgufModel::load`**: Enhanced path detection: transparently loads directly from checkpoint directories or `.safetensors` files using `load_checkpoint_as_gguf`.
- **`import::prepare_local`**: Added `--direct` / `--zero-copy` flag support (and `FLLM_DIRECT_SAFETENSORS` environment variable) to bypass disk GGUF caching and return local checkpoint directories directly.
- **CLI Flags**: `--direct` and `--zero-copy` options exposed in `fllm serve`.

### 4. Tests and Parity Verification (`rust/checkpoint/src/tests.rs`, `rust/engine/src/import/tests.rs`, `rust/engine/src/import_tests.rs`)
- **Unit Test in `checkpoint::tests`**: Verified `load_checkpoint_as_gguf`:
  - Untouched tensors (`token_embd.weight`) reside strictly within the memory map address range of the underlying safetensors file.
  - Permuted tensors (`blk.0.attn_q.weight`) are allocated in anonymous read-only memory outside the safetensors file mapping.
  - Bit-for-bit exact tensor byte equivalence against on-disk `convert_checkpoint` reference.
- **Import Validation Test**: Checked `--direct` / `--zero-copy` flag parsing, exclusivity validation against `--quant` and `--outtype`, and cache bypass in `prepare_local`.
- **End-to-End Engine & Logit Parity**: Verified bit-for-bit tensor byte parity and forward pass logit equivalence across model decoders (`RustDecoder`) between direct safetensors and converted GGUF models.

## Quality Gate Verification
- **Unit Tests**:
  - `fllm-checkpoint`: 67 passed, 0 failed.
  - `fllm-engine`: 380 passed, 0 failed.
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
