# Data-center nodes: what fLLM has, what TensorRT-LLM and vLLM / SGLang have, and what was added

Goal: fLLM should be a credible engine on a node of 8 large GPUs (8x AMD MI355X, 8x NVIDIA B200 / H200 / H100, GB200) for big models
(DeepSeek-V3 / V4, Kimi K2 / K3, GLM-5, Qwen3.5 MoE, Llama 4, gpt-oss): structurally correct, plausibly fast, and every unverified part
labelled. **Nothing in this document was run on a multi-GPU machine or on an MI355X.** The hardware that exists for testing is one RTX 2000 Ada
laptop (single GPU, CUDA), a build server without a GPU (compiles, CPU stand-ins, ROCm compilers in a container) and Windows PCs. Test levels
are those of [test-levels.md](test-levels.md): L0 compiled, L1 self test passed on the real device, L2 text equals the reference, L3 speed
measured against a reference.

## 1. What the reference engines have (sources)

Read from the engines' own documentation (fetched 2026-11-03); items marked (brief) come from the hardware description this work was given, because
AMD's architecture white paper could not be fetched.

* **TensorRT-LLM** (nvidia.github.io/TensorRT-LLM, PyTorch workflow). Feature-combination matrix: overlap scheduler, CUDA graphs, tensor /
  pipeline / expert parallelism, Helix (context) parallelism, attention data parallelism, disaggregated serving, chunked prefill, speculative
  decoding (MTP, EAGLE-3, draft-target, PARD, DFlash, suffix automaton, n-gram, user-provided drafters), torch sampler, KV-cache reuse,
  sliding-window attention, logits post-processor, guided decoding, LoRA. Parallelism guide: TP, PP, DP, EP, CP, **wide EP** (expert slots
  decoupled from experts, hot experts replicated, offline and online expert-parallel load balancing, all-to-all kernels tuned for GB200
  multi-node NVLink), hybrid MoE layouts (`moe_tensor_parallel_size` x `moe_expert_parallel_size` = `tensor_parallel_size`), attention DP
  (`enable_attention_dp`: attention weights replicated, KV cache partitioned by request). Recommended layouts: dense -> TP (PP when it does not
  fit); fine-grained MoE (DeepSeek-V3/R1, Llama 4, Qwen3) -> wide EP with load balancing; throughput -> attention DP. Disaggregated serving:
  context and generation workers with separate parallelism, KV moved GPU to GPU by **NIXL** (RDMA / NVLink; UCX or libfabric underneath),
  layout conversion between different TP/PP, transfer overlapped with compute; decoder-only models with homogeneous KV only.
* **vLLM / SGLang.** Expert-parallel deployment: `--enable-expert-parallel` (EP size = TP x DP), data-parallel attention, all-to-all
  back ends `allgather_reducescatter`, `deepep_high_throughput` / `deepep_low_latency`, FlashInfer NVLink one-sided/two-sided, and **`mori` for
  ROCm** (MoRI EP; "EPv2" only on gfx950, one node or one fabric, BF16 activations); EPLB (`--enable-eplb`, redundant experts, window and
  step interval); disaggregated prefill/decode over NIXL; dual-batch overlap; multi-node data-parallel launch (`--data-parallel-address`,
  `--headless`). On ROCm, AITER supplies the fused MoE, attention and GEMM kernels.
* **AMD ROCm precision support** (rocm.docs.amd.com, precision-support page): FP8 **FNUZ** (E4M3/E5M2, bias shifted, NaN = negative zero, no
  negative zero, max 240) is the CDNA3 (MI300 series, gfx942) format; **OCP** E4M3/E5M2 (max 448, what NVIDIA GPUs and most FP8 checkpoints use)
  is CDNA4 (MI350 / MI355X, gfx950) and RDNA4. FP4 (E2M1) and FP6 (E2M3, E3M2) matrix-core types exist only on CDNA4.
* **MI355X (brief):** CDNA4 / gfx950, 288 GB HBM3E per GPU, 8 per node on Infinity Fabric (XGMI), wave64, 160 KB LDS, scaled MFMA
  `v_mfma_scale_f32_*` for FP4 / FP6 / FP8 with E8M0 block scales, RCCL, hipBLASLt, AITER, MoRI.

## 2. The matrix

Columns: the feature; TensorRT-LLM; vLLM / SGLang; **fLLM today, verified in the code** (state after this change; "new" = added by it, with
the test level); priority for a data-center node (P0 blocks large models, P1 big speed or capacity gain, P2 useful, P3 later); decision.

| Feature | TensorRT-LLM | vLLM / SGLang | fLLM today | Pri | Decision |
| --- | --- | --- | --- | --- | --- |
| Continuous batching, decode-first scheduling | yes, overlap scheduler | yes | yes (`scheduler.rs`: decode tokens reserved before prompt chunks) | - | done |
| Paged KV cache, prefix reuse | yes | yes | yes, radix prefix cache (`mem/prefix_cache.rs`) | - | done |
| Chunked prefill | yes | yes | yes (`--max-batched-tokens`) | - | done |
| KV offload to host / disk | host offload | LMCache, CPU offload | swap-out preemption to host memory, disk tier for prefix blocks (`mem/kv_tier.rs`); no asynchronous tiering | P2 | keep |
| Speculative decoding | MTP, EAGLE-3, draft-target, PARD, DFlash, suffix, n-gram | EAGLE, MTP, n-gram, draft | n-gram lookup, draft model, MTP, suffix, converted EAGLE / Medusa heads (`--speculative`); **refused with a device list** | P1 | next: allow with tensor parallelism (MTP matters for DeepSeek-class decode) |
| Guided decoding | yes | xgrammar / llguidance | yes (llguidance) | - | done |
| Multi-LoRA batching | yes | yes | adapter scale route and merge tool only (`/lora-adapters`, `merge-lora`) | P3 | later |
| CUDA graphs | yes | yes | yes on one GPU; **off in a parallel group** | P1 | next: capture the device all-reduce (NCCL is capturable; the one-shot kernel is too) |
| Tensor parallelism | yes | yes | yes: 31 weight layouts sharded, CPU stand-in equal to the single device; **NCCL device-pointer all-reduce wired into the forward pass (new, L1 with 1 rank)** | - | done, hardware run pending |
| Pipeline parallelism | yes | yes | yes, micro-batches, TP x PP, planner for uneven devices | - | done |
| Expert parallelism | EP, TP x EP hybrid | EP = TP x DP | yes (`--expert-parallel`, all-reduce or all-to-all combine), proven on the CPU stand-in; **expert-TP x EP hybrid not implemented** | P0 | next |
| Attention data parallelism | yes | yes | **no** | P0 for MLA models | planned (planner knows it, `parallel/cluster_plan.rs`) |
| MLA attention under parallelism | TP with replicated latent KV, attention DP | DP attention | **MLA cannot be split** (error: use pipeline parallelism); DeepSeek-class models run as PP only | P0 | next: replicated or data-parallel MLA + EP |
| Wide EP all-to-all kernels | NVLink one-sided, MNNVL | DeepEP, FlashInfer, MoRI | none. **Device all-to-all through NCCL / RCCL send/recv exists (new, compiled)** but the expert dispatch of the forward pass still goes through the host | P1 | wire the device all-to-all into the MoE layer, then measure |
| Expert load balancing (EPLB) | offline + online, expert slots | `--enable-eplb`, redundant experts | **new: statistics and placement planner with replication (`parallel/eplb.rs`), unit-tested**; no runtime weight movement | P1 | runtime half after a multi-GPU run shows the skew |
| Custom all-reduce (one-shot / two-shot, fused with RMSNorm) | yes | yes | **one-shot kernel over peer access (new, L1 on one GPU with 1..8 ranks sharing it)**; off unless `FLLM_CUSTOM_ALLREDUCE=1`; two-shot and fusion not built | P1 | measure on hardware, then two-shot |
| NCCL / RCCL | NCCL | NCCL / RCCL | NCCL by `dlopen` (NCCL 2.30.7 ran with 1 rank, L1); **RCCL by `dlopen` of `librccl.so.1` (new, compiled for gfx942 / gfx950 / gfx1100, never run)** | - | done |
| Peer access, topology report | yes | yes | `comm/topology.rs` (CUDA peer rank); `hipDeviceEnablePeerAccess` mapped (new, compiled) | P2 | run the report on an MI355X |
| Multi-node | MPI / Ray, NCCL | Ray / multiprocessing | TCP rendezvous + socket control mesh, NCCL unique id exchange; **non-first nodes now run headless (new, tested with two local processes on the CPU stand-in)**; no NCCL multi-node run | P1 | run on two machines |
| FP8 weights and activations | sm90 / sm100 kernels | CUTLASS, AITER | CUDA: block-wise FP8 with CUTLASS sm90, tensor-core prompt path (run on sm89 only); **AMD: FP8 checkpoints are expanded to F16 on load** | P0 for AMD | built the gfx950 path below, not yet used by the loader |
| FP4 (NVFP4 / MXFP4) | NVFP4 GEMM sm100 | FlashInfer, AITER | CUDA: NVFP4 (CUTLASS sm100, compiled only), MXFP4 as a lookup-table format; AMD: MXFP4 via lookup table in the F16 matrix-core GEMM; **new: MXFP4 x MXFP8 on the scaled MFMA of gfx950 (compiled, self test ready)** | P0 for MI355X | needs a run on an MI355X |
| FP8 KV cache | yes | yes | yes (E4M3 / E5M2 formats, software conversion, same on every GPU) | - | done; FP8 attention math not built |
| OCP vs FNUZ FP8 | n/a | handled in AITER | **new: both formats decoded / encoded in software with a checked bit-exact twin, OCP <-> FNUZ conversion** | - | done (CPU-tested) |
| MLA kernels | FlashMLA-class | FlashMLA, AITER MLA | CUDA: MLA decode and latent-cache kernels; HIP: MLA through the portable kernels only | P1 | |
| Disaggregated prefill / decode | NIXL / UCX, layout conversion | NIXL | **new: prefix-chain transfer (`kv_transfer.rs`), HTTP and shared-directory transports, `fllm-router --prefill/--decode`; same layout only, through host memory; tested with the CPU runtime (L1 on CPU)** | P1 | RDMA transport and TP-layout conversion later |
| `trtllm-bench` / `vllm bench serve` | yes | yes | `fllm-rs bench` (fixed scenarios); **new: `fllm-rs bench-serve` (length distributions, Poisson or closed loop, TTFT / ITL / TPOT percentiles, goodput, JSON + Markdown, suggestion for `--max-seqs` / `--max-batched-tokens`)** | - | done |
| GEMM tactic autotuner | yes | partly | scaffold (`autotune.rs`), nothing calls it | P2 | later |
| Parallelism planner / `auto` | manual | manual | **new: `fllm-rs plan-cluster` and `--tensor-parallel auto` choosing TP / PP / EP from the model and the GPUs** | - | done |
| Weight loading per rank | sharded checkpoints | sharded | each rank slices its share out of the memory-mapped file (row cuts are zero-copy views), layer by layer; no host copy of the whole model | - | done (CPU-tested) |
| Context / Helix parallelism | yes | yes | no | P3 | later |

## 3. What was built, and how it was verified

| Part | Built | Verified with | Only real hardware can show |
| --- | --- | --- | --- |
| Device-pointer all-reduce in the forward pass (`Stream::device_all_reduce`, `FLLM_HOST_ALLREDUCE=1` restores the host path) | the row-parallel reductions of CUDA / HIP groups no longer go through two host copies per layer | `selftest-comm` on the RTX 2000 Ada: NCCL 2.30.7 with 1 rank, host slices and device buffers pass. A real model (Llama-3.2-3B Q4_K_M, `--tensor-parallel 2`, both ranks on that one GPU with `FLLM_ALLOW_SHARED_DEVICE=1`): the host path gives the single-GPU text; the device path with the one-shot reducer gave the same text once and then hung in every later run (not diagnosed; most likely the allocations of one rank synchronize the device while the other rank's kernel spins waiting for it). That cannot happen with one rank per GPU, but it means the device path in a real model has **not** been shown to work repeatedly | NCCL / RCCL between real GPUs, link speed |
| RCCL | `librccl.so.1` / `rccl.dll` opened at run time (same API as NCCL); comm layer enabled in HIP builds; `comm_kernels.cu` built by `hipcc`; backend `hip` accepted by `--distributed-backend` | `hipcc` for gfx942, gfx950, gfx1100 in the ROCm 7 image and a clippy / release build of the engine with the `hip` feature there | everything at run time |
| One-shot custom all-reduce (`comm/oneshot.rs`, `fllm_comm_ar_oneshot`) | flag-based single-kernel reduction over peer-mapped staging buffers, f32 / f16 / bf16, two staging halves, rank-ordered sums (bit-identical on all ranks); replaced a peer-to-peer communicator that passed device addresses as `f32` and read buffers other ranks were still writing | `selftest-comm`: 55 exact checks per rank, 1 / 2 / 4 / 8 ranks sharing GPU 0 of the RTX laptop (the flag protocol runs for real, only the fabric is missing); compiled for gfx942 / gfx950 / gfx1100 | peer access over NVLink / XGMI, speed, the uncached flag allocation on AMD |
| Planner (`parallel/cluster_plan.rs`, `fllm-rs plan-cluster`, `--tensor-parallel auto`) | TP x EP x PP x attention-DP candidates, memory per rank, roofline step estimate, "ideal" and "runs in this build" | 14 unit tests (parameter counts of 7 public models within 3-6 %, 8x MI355X / B200 / H100 cases, every preset x GPU x count) | that the estimates match measured speed |
| EPLB planning (`parallel/eplb.rs`) | counters and replicating placement | 6 unit tests (skewed Zipf load: 5.3x imbalance with contiguous ranges, near 1.0x with spare slots) | the skew of real routers |
| gfx950 formats and kernels (`hip/hip_mx.cuh`, `hip/hip_gfx950.cu`, `src/mx_ref.rs`) | OCP / FNUZ E4M3, E5M2, E8M0, E2M1 in software with a C++ twin; MXFP8 activation quantizer; MXFP4 x MXFP8 GEMM on `v_mfma_scale_f32_16x16x128_f8f6f4` with a scalar fallback | 11 CPU tests including an exhaustive check of all 256 codes, a 2-million-input checksum match between the C++ header and the Rust twin, and a lane-level emulation of the kernel (addressing and tile edges); `hipcc` for gfx942 / gfx950 / gfx1100; the emitted assembly contains the scaled MFMA | the operand layout of the instruction and the numerics: `selftest-gpu --device hip:N` runs the kernel against the scalar reference on a gfx950 |
| Disaggregated prefill / decode | `kv_transfer.rs`, `/admin/disagg/export`, `/admin/disagg/import`, router mode | 8 unit tests: a prompt prefilled by one CPU runtime, shipped through the wire format and a directory, imported by a second runtime gives the same tokens with a prefix-cache hit and no block leak; two server processes and the router: see section 5 | GPU pools, large chains, RDMA |
| Multi-node launcher (`--nnodes N --node-rank R --master-addr`) | non-first nodes run their ranks headless and exit when the first node ends | two local processes on the CPU stand-in: section 5 | NCCL across machines |
| `fllm-rs bench-serve` | see matrix | 9 unit tests, including closed- and open-loop runs against a mock streaming server | numbers on real servers |

## 4. The plans for the target node

`fllm-rs plan-cluster` prints what the planner decides. Estimates are a roofline (weights read from HBM plus collectives over the link), they
rank layouts and are **not measurements**. "Ideal" is the best layout of the model; "runs in fLLM" the best one this build can execute.

Commands and the lines that matter (full output: run the command). 671B and 1T are the parameter counts the planner derives from the model shapes
(DeepSeek-V3: 671B, 37B active; the 1T model is a hypothetical GQA mixture of experts with 384 experts, 1031B, 37B active).

| Model, weights | Node | Layout the planner prefers | Runs in this build | Notes |
| --- | --- | --- | --- | --- |
| DeepSeek-V3 FP8 (630 GiB stored, 1250 GiB on AMD, where this build expands FP8 to F16) | 8x MI355X | attention data parallelism x 8 + expert parallelism 8: 175 GiB weights + 11 GiB buffers per rank, 73 GiB KV (8.9M tokens in the node); step 24.7 ms (2595 tok/s at batch 64) | pipeline x 8: 159 GiB per rank, 156 ms (410 tok/s) | with native FP8 weights (`--native-fp8`): 88 GiB per rank, 14.4 ms (4455 tok/s) ideal; pipeline x 8 83.7 ms |
| DeepSeek-V3 FP4 (352 GiB) | 8x MI355X | the same layout: 49 GiB per rank, 199 GiB KV (24M tokens), 9.7 ms (6569 tok/s) | pipeline x 8: 51 ms (1251 tok/s) | |
| DeepSeek-V3 FP8 | 8x B200 (179 GiB) | the same layout: 88 GiB per rank, 64 GiB KV (7.8M tokens), 13.4 ms (4772 tok/s) | pipeline x 8: 83.6 ms (765 tok/s) | |
| DeepSeek-V3 FP4 | 8x B200 | 49 GiB per rank, 103 GiB KV, 8.8 ms (7280 tok/s) | pipeline x 8 | |
| DeepSeek-V3 FP8 | 8x H100 (79 GiB) | nothing fits (630 GiB of weights against 569 GiB usable) | - | the planner says so instead of inventing room |
| 1T MoE FP8 (967 GiB) | 8x MI355X | tensor parallel 8 + expert parallel 8: 121 GiB weights per rank, 127 GiB KV (4.3M tokens), 19.4 ms (3308 tok/s) | the same (grouped-query attention can be split) | `--tensor-parallel 8 --pipeline-parallel-size 1 --expert-parallel` |
| 1T MoE FP4 (540 GiB) | 8x B200 | tensor parallel 8 + expert parallel 8: 68 GiB per rank, 84 GiB KV, 12.9 ms (4966 tok/s) | the same | |
| Qwen3-235B-A22B FP8 | 8x MI355X | attention data parallelism (only 4 KV heads, tensor parallelism replicates them 2x) | tensor parallel 8 + expert parallel 8 | |
| Llama 3.1 405B BF16 | 8x MI355X | tensor parallel 8: 94.5 GiB per rank, 28.5 ms (2250 tok/s) | the same | |

The estimates say what the layouts cost relative to each other (pipeline stages take turns for a batch, which is why the model that cannot be split
inside a stage loses a factor of 5 to 6); they say nothing about the absolute speed of fLLM's kernels on these GPUs.

## 5. End-to-end runs on the CPU stand-in

Two experiments with real processes on a Windows PC (the executable cross-built for Windows, CPU backend, the tiny random model of
`fllm-rs make-tiny-model` whose output is gibberish but deterministic). Greedy decoding, 158-token prompt.

* **Multi-node tensor parallelism.** `serve --device cpu --tensor-parallel 2 --nnodes 2 --node-rank 0|1 --master-addr 127.0.0.1 --master-port 29555`,
  two processes. Node 0 serves HTTP and returned exactly the text of a single process; node 1 printed "waiting for the first node to drive the model",
  opened no HTTP port and exited when node 0 was stopped. (The first attempt of this run panicked in the group setup, which indexed the devices of all
  nodes although a node knows only its own: fixed. By reading the earlier code, a node other than the first would also have started a second server.)
* **Disaggregated prefill / decode.** Prefill server, decode server and `fllm-router --prefill ... --decode ...`. The router answered with
  `x-fllm-disagg: transferred`, the text equalled the single process, the decode server reported 144 of 158 prompt tokens as cached (nine blocks of 16;
  the last partial block is computed again), and `/router/stats` counted one transfer and no fallback.
* **`bench-serve`** against the same tiny server (concurrency 1, 2, 4; then 5 req/s and "all at once"): all requests completed, the table above was
  produced, and the suggestion logic ran. The numbers are meaningless for a model of 350 KB. One property to know: the event times come from the
  moment the client reads a server-sent event, and events that the server flushed together arrive together, so the median ITL can read 0.0 ms;
  TPOT is the robust per-token figure.

None of this says anything about GPU pools, large chains or NCCL / RCCL between machines.

## 6. What comes next, in order

1. **MLA under parallelism** (P0): replicate the MLA attention weights (they are 2 % of a DeepSeek-V3) and run the experts expert-parallel, then
   attention data parallelism proper. Without it the big MLA models run as a pipeline, which leaves 7 of 8 GPUs idle for one sequence.
2. **Device expert dispatch** (P0/P1): use the device all-to-all (NCCL / RCCL send/recv, already built) for the expert exchange; today the
   host path moves every token's activation twice per MoE layer.
3. **A run on an 8-GPU machine** with the checklist below, then the two-shot all-reduce, fused all-reduce + RMSNorm and CUDA-graph capture of
   the collectives, which only make sense with numbers.
4. FP8 / FP4 weights on AMD through the scaled MFMA instead of the F16 expansion, once the gfx950 self test passes on hardware.
5. Runtime half of EPLB; RDMA / NVLink transport and TP-layout conversion for disaggregated serving; speculative decoding in a group.

## 7. Checklist for a person with an 8x MI355X node

Please send the complete output of every command (and `rocm-smi --showtopo`).

```
fllm-rs devices                                     # 8 x hip devices with gfx950 and ~288 GB each
fllm-rs doctor --gpu                                # the GPU check
for i in 0 1 2 3 4 5 6 7; do fllm-rs selftest-gpu --device hip:$i --failures-only; done
                                                    # must end with "PASS  gfx950: scaled MFMA ..." and 0 failures on every GPU
fllm-rs selftest-comm                               # RCCL part: needs librccl.so.1 (ROCm) in the loader path or FLLM_RCCL_PATH
FLLM_CUSTOM_ALLREDUCE=1 fllm-rs selftest-comm       # one-shot kernel across the 8 GPUs ("custom one-shot all-reduce across 8 GPUs")
fllm-rs plan-cluster --model deepseek-v3 --gpu mi355x --gpus 8 --weights fp8
fllm-rs plan-cluster --model MODEL.gguf --gpu mi355x --gpus 8
# a dense model on 8 GPUs (tensor parallel 8):
fllm-rs serve --model Llama-3.1-70B-Instruct-Q4_K_M.gguf --device hip:0,hip:1,hip:2,hip:3,hip:4,hip:5,hip:6,hip:7 --tensor-parallel 8
fllm-rs serve --model ... --tensor-parallel auto --device hip:0,...,hip:7      # logs the layout it chose
fllm-rs bench-serve --concurrency 1,4,16,64 --input-len 512 --output-len 128 --suggest --json tp8.json
FLLM_HOST_ALLREDUCE=1 fllm-rs serve ...             # the old host path, for comparison of bench-serve numbers
```

What to send back: the outputs above, `bench-serve` JSON files for the runs with and without `FLLM_CUSTOM_ALLREDUCE=1`, the startup log of the TP8
run (it prints the layout and the traffic back end), and any line that starts with `[fllm-rs]` and mentions RCCL, peer access or a failed rank.
