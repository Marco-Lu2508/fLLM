# Gemma 4, Pro-Layer Models and Hybrid KV Architecture on Accelerators (gx-gemma4)

## (a) Features

**Per-layer Geometry on Accelerators.**
Extended `DeviceKvCache` across all backends (`cpu`, `cuda`, `vulkan`, `metal`, `wgpu`, `sycl`, `tpu`) with `KvPoolLayout` and per-layer geometry offsets (`layer_geoms`).
Implemented `allocate_kv_cache_layered` in backend traits and engines. The decoder dispatch loop in `rust_decoder.rs` now provisions per-layer query heads, key/value heads, head dimensions, value dimensions, and routes reading of shared KV layers (`read_layer = KvPoolLayout::owner(cfg, i)`) directly on accelerators. Rejections for `gemma4`, `gemma3n`, `laguna`, and `mimo_v2` on non-CPU devices have been removed.

**Sliding-Window Block Pruning & Hybrid KV Management.**
Added `prune_sliding_window` to `BlockAllocator` (`rust/engine/src/mem/block_allocator.rs`), actively pruning and reclaiming memory blocks that fall completely outside the sliding-window horizon for sliding layers.
Implemented `HybridBlockManager` and `KvPoolLayout::geometry_groups()` (`rust/engine/src/kv_geometry.rs`), enabling multi-pool management where layers are partitioned by KV head count, head dimension, and sliding window boundaries, mirroring vLLM's Hybrid KV Cache manager.

**Attention Head Width 512.**
Updated attention parameter validation, buffer sizing, and backend dispatch across CUDA, Metal, Vulkan, and wgpu to support head dimensions up to 512 (used by Gemma-4 global layers).

**Architecture Additions in Plan Importer.**
- Step-3.5: Support for `num_attention_heads_per_layer` in HuggingFace configuration import.
- DeciLM & Llama-3.1-Nemotron-51B/253B: Support for heterogeneous `block_configs` defining per-layer `attention_heads`, `kv_heads`, and `ffn_hidden_size`.
- Nanbeige4.2 & IQuestLoopCoder: Reading and propagation of `num_loops`.
- Phi-3-small: Importer support for `blocksparse_block_size` with dense fallback attention.
- Laguna: Support for `attention.gate_per_element` in configuration and decoder gating logic.

**Synthetic llama.cpp Gemma 4 / 3n GGUF Verification.**
Added `llama_cpp_gemma4_layout_loads_and_initializes` to `rust/engine/src/gemma_tests.rs`, validating synthetic GGUF structures following public llama.cpp conventions with per-layer global/SWA head configurations, key/value dimensions, and sliding window patterns.

## (b) Environment Variables and Flags

No new CLI flags or environment variables are required. Per-layer accelerator dispatch and sliding-window pruning are automatically active when supported models are loaded.

## (c) Third-Party Code

- Principles of Hybrid KV Cache partitioning and sliding-window block reclamation adapted from vLLM (`vllm/attention/backends/hybrid.py`, Apache-2.0).

## (d) Test Levels

- **L0 (Compile / Type-check)**: Metal, Vulkan, SYCL, TPU, CUDA accelerator dispatch paths compile cleanly and satisfy interface traits.
- **L1 / L2 (Unit Tests & Golden Logits on Host)**:
  - All 7 tests in `gemma_tests` pass (`llama_cpp_gemma4_layout_loads_and_initializes`, `gemma4_conversion_describes_the_layer_geometry`, `shared_layers_own_no_cache_and_the_pools_follow_each_layers_geometry`, `gemma4_matches_transformers`, `gemma3n_matches_transformers`, `gemma3n_decode_through_the_cache_matches_the_batch`, `gemma4_decode_through_the_cache_matches_the_batch`).
  - All 6 unit tests in `mem::block_allocator` pass (including `sliding_window_pruning_releases_blocks_outside_window` and `hybrid_block_manager_tracks_independent_geometry_groups`).
  - All 3 unit tests in `kv_geometry` pass.
  - Complete workspace test suite: 364 engine tests + 48 checkpoint tests + 3 gguf-probe tests pass (`cargo test --release --locked -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe`).

## (e) Known Limits

- Actual GPU execution of CUDA/FlashInfer/Metal/Vulkan kernels requires execution on GPU hardware (no GPU present on build host `CPU build server`; test level L0 for accelerator execution). Verification on Google Colab is required.
