# Change fragment: multi-device parallelism (tensor, pipeline, expert)

Full description: `docs/multi-gpu.md`.

## (a) Features and commands

* One model over several devices, one process, one worker thread per rank, SPMD steps announced by rank 0.
  `fllm-rs serve --model M.gguf --device cpu --tensor-parallel-size 4` (CPU in this version, see limits),
  `-tp 2 -pp 2`, `--expert-parallel [--ep-dispatch alltoall]`.
* Tensor parallelism of dense and MoE decoders: column/row-parallel linears with an all-reduce per branch, head sharding with KV-head
  replication when `tp > kv_heads`, vocabulary-parallel output head (all-gather of the logits), optional vocabulary-parallel embedding,
  every rank loads only its shard from the memory-mapped GGUF.
* Pipeline parallelism: contiguous layer ranges, activations by send/recv, micro-batches split by sequence.
* Expert parallelism: contiguous expert ranges per tensor rank, combine by all-reduce or by all-to-all with variable splits.
* `Communicator` trait with `LocalComm` (host threads), a TCP rendezvous store and socket mesh for several nodes, and `NcclComm`
  (libnccl through `dlopen`, feature `cuda`).
* `fllm-rs selftest-parallel [--all] [--tp N --pp M --ep] [--case NAME | --model PATH]`: CPU proof that the logits and a 16-token
  greedy generation equal the single-device run.
* `fllm-rs selftest-comm`: exact checks of all collectives; NCCL part on GPUs.

## (b) Flags and environment variables

| Name | Default | Meaning |
| --- | --- | --- |
| `--tensor-parallel-size N`, `-tp N` | 1 | tensor-parallel group size |
| `--pipeline-parallel-size M`, `-pp M` | 1 | pipeline stages |
| `--expert-parallel`, `--enable-expert-parallel` | off | experts distributed over the tensor group |
| `--ep-dispatch allreduce\|alltoall` | allreduce | expert-parallel combine |
| `--devices 0,1,..` | first N visible | device ordinals of this node's ranks |
| `--distributed-backend nccl\|p2p\|local` | nccl if libnccl is found, else p2p; local on CPU | traffic between devices (`p2p` is not implemented yet and errors) |
| `--nnodes`, `--node-rank`, `--master-addr`, `--master-port` | 1, 0, -, 29500 | multi-node rendezvous |
| `--pipeline-microbatches K` | one per stage | micro-batches per pipeline step |
| `FLLM_VOCAB_PARALLEL_EMBED` | 0 | split the embedding table by vocabulary rows |
| `FLLM_COMM_TIMEOUT_SECS` | 600 | receive timeout of the host communicator |
| `FLLM_NCCL_PATH` | - | path of libnccl.so.2 |
| `FLLM_PARALLEL_GPU` | 0 | try a GPU device with a parallel layout (not wired into the CUDA forward path; untested) |

## (c) THIRD_PARTY

No third-party code was copied. The design follows publicly documented ideas (Megatron-LM style column/row-parallel linears, vLLM style
KV-head replication and expert parallelism); NCCL is loaded at run time and not redistributed.

## (d) Test levels

* L0 (compiles only): `comm/nccl.rs`, `comm/cuda_ffi.rs`, `comm/topology.rs`, `cuda/comm_kernels.cu` (nvcc for sm_75/80/89/90/120 and
  clippy with `--features cuda` in the CUDA 12.8 image), the GPU part of `selftest-comm`. Nothing of this was run on a GPU.
* Host-level proof (CPU, no GPU needed): shard slicing of 31 formats, collectives on 2..8 threads and 4 loopback sockets, tensor /
  pipeline / expert parallel decoding of nine tiny models (Llama, Qwen3, Qwen2 with bias, Gemma-2, MoE with and without a shared expert;
  Q4_K, Q8_0, NVFP4, Q4Z, Q4ZP, F16, F32) against the single-device decoder: tokens identical, pipeline runs bit-identical, tensor runs
  within 1e-4 (float weights) / 1e-2 (quantized, see doc) of relative L2 error.
* L1/L2/L3 on GPUs: none.

## (e) Known limits

* The CUDA forward path does not call the collectives yet: parallel layouts run with `--device cpu`; the server refuses a GPU device
  (`FLLM_PARALLEL_GPU=1` overrides, untested). Missing there: sharded device uploads, all-reduce after the row-parallel projections
  (and disabling the fused residual kernels for them), pipeline activation transfer on device, per-device memory plan, CUDA graphs
  with NCCL calls, distributed argmax on the device logits, `num_local_experts`/`expert_offset` kernels for GPU expert parallelism.
* `P2pComm` (a placeholder that returns an error) and the custom all-reduce (`FLLM_CUSTOM_ALLREDUCE`): not implemented. The peer
  matrix (`comm::topology`, from CUDA peer attributes, printed by `selftest-comm`) compiles but was never run.
* No tensor parallelism for latent attention, DeltaNet, Mamba, short convolution, value experts, latent MoE, full-width query/key norm.
* Equal shards only; no 1F1B schedule; no padding to fixed capacities for the all-to-all.
* `--gpu-layers` host offload is switched off with a parallel layout.
