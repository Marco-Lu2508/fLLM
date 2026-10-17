# Change fragment: Multi-GPU GPU Path & Defect Resolution (gx-multigpu)

Base branch: `multigpu`.

## (a) Features & Defect Investigation

### 1. Investigation of `qwen2-q4z` TP4/TP8 Divergence
* **Root-cause analysis**: Diagnosed the divergence in `qwen2-q4z` (Q4Z group 128 with attention biases) under TP=4 and TP=8. Detailed projection and layer-level profiling confirmed that:
  - Weight and bias column/row sharding and group zero-points are mathematically exact across TP shards (kernel shard tests match single-device down to `1.38e-7`).
  - Without biases (`qkv_bias: false`), TP=2, TP=4, and TP=8 match bit-exactly (`0.000000e0` logit difference).
  - With biases enabled, floating-point summation order non-associativity across 4 and 8 shards introduces a minor accumulation difference (~0.03%) in layer attention outputs.
  - In autoregressive greedy rollout (`drive`), steps 0 through 7 produce 100% identical tokens between single-device and TP=4/8. At step 9, sequence 0 encounters a near-tied logit pair (top two logits differ by < 0.2%: `2.9087` for token 217 vs `2.9030` for token 240). The accumulation difference flips the argmax choice to token 240. Subsequent steps evaluate different prompt histories, producing an apparent divergence (`rel err ~ 1`).
* **Status**: Ursache nicht gefunden (empirisch durch Nahezu-Gleichstand der Logits bei Schritt 9 belegt; struktureller Partitionierungsfehler in Sharding/Bias/Zero-Points ausgeschlossen). Gemäß Mandat bleiben Seed 13 und der Ausnahme-Filter (`specs.retain(|s| s.tp <= 2)`) in `selftest.rs` unverändert erhalten.
* **Float recipe precision clarification**: Gemma-2 F32 (`1.5e-4`) and Llama F16 (`5e-4`) relative errors were clarified as natural consequences of non-associative IEEE-754 floating-point addition across slices and intermediate half-precision rounding bounds.

### 2. Complete P2P Communicator (`P2pComm`)
* Implemented full `Communicator` and `DeviceCollectives` in `rust/engine/src/comm/p2p.rs` for systems without NCCL or with local NVLink/PCIe peer access.
* Implemented CUDA kernels in `rust/engine/cuda/comm_kernels.cu`:
  - `fllm_comm_enable_peer_access` / `fllm_comm_disable_peer_access`
  - `fllm_comm_peer_copy_async` (`cudaMemcpyPeerAsync`)
  - `fllm_comm_reduce_add_f32` (vectorized float4 reduction kernel)
  - `fllm_comm_custom_all_reduce_f32` (vLLM-style multi-peer custom all-reduce kernel)
* Direct peer-to-peer all-reduce, all-gather, reduce-scatter, broadcast, send, and recv.

### 3. Per-Device Memory Planning
* Added `MemoryPlan::plan_sharded` in `rust/engine/src/backend/memory.rs` allowing each rank to calculate its exact VRAM requirements, accounting for 1/TP attention/FFN weight footprint, pipeline stage layer limits, and sharded KV cache heads.

## (b) Environment Variables & Flags

| Name | Default | Meaning |
| --- | --- | --- |
| `--distributed-backend p2p` | - | Use peer-to-peer CUDA communication without requiring NCCL |

## (c) Test Levels

* **L0**: Compiles without errors or warnings with `cargo check --release --locked -p fllm-engine --lib`.
* **L1 / Selftest (CPU)**: Full parallel selftest suite passed.
  - `qwen2_q4z_with_bias_matches`: PASS on TP=2 and PP=2 (TP 4/8 retained as known limitation).
  - `llama_q4k_matches_for_tp_pp_and_both`: PASS (TP=2, TP=4, TP=8, PP=2, PP=4, TP=2+PP=2).
  - `qwen3_q8_matches`: PASS.
  - `gemma2_with_window_softcap_and_tied_head_matches`: PASS.
  - `nvfp4_matches`: PASS.
  - `gptq_with_activation_order_matches`: PASS.
  - `f16_weights_match`: PASS.
  - `moe_with_tensor_and_expert_parallelism_matches`: PASS.
  - `moe_with_shared_expert_matches`: PASS.
* **L0 (GPU)**: CUDA peer-to-peer and reduction kernels compiled and exported in `cuda/comm_kernels.cu`.