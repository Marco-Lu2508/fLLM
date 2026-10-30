# Mixture-of-experts models larger than memory

A mixture-of-experts (MoE) model reads only a few of its experts per token (OLMoE: 8 of 64 per layer, DeepSeek-V3: 8 of 256), so a model
much larger than the free memory can still run if the experts are read from the model file when a token needs them and the ones used
most stay in memory. This page says what fLLM did before, what was added, and what was measured. Every number is from the machines named
next to it; what was not run is marked.

## 1. Audit: what fLLM did before

| Question | Answer (code that decides it) |
|---|---|
| Is a GGUF file read at load time? | No. `MappedGguf::open_single` (`rust/gguf-probe/src/lib.rs`) memory-maps the file read-only; `TensorBytes` are views into the map (`owned_tensor_bytes`, `Linear::load_rows` slices a stacked expert tensor without a copy). Measured: OLMoE-1B-7B Q4_K_M (4.2 GB) is ready in 2 s with 108 MB resident. |
| Are expert weights copied or repacked? | Not for GGUF, on the CPU. The CPU kernels read the mapped bytes in place; the prompt panels (`cpu_quantized.rs`) are built per call and not kept. K-quant and Q4_0 experts stay file data. |
| Safetensors / PyTorch checkpoints | `materialize_mapped_gguf` (`rust/checkpoint/src/convert.rs`) keeps tensors the converter copies unchanged as slices of the checkpoint, but **builds every tensor it has to change into one anonymous (heap) map**. The experts of a Hugging Face checkpoint are separate tensors that get stacked, so they are built in memory (and `--quant` re-encodes into the same map). A direct load of a huge MoE checkpoint therefore needs the whole expert set in RAM (`max_built_bytes` turns that into an error before anything is built). Convert once (`fllm-rs convert`, `--no-direct-load`) and run the GGUF to stream. |
| CPU MoE path | `RustDecoder::moe_route` (router, top-k, group limits) then `moe_forward`: one token = the fused queue of `Linear::forward_experts` over the chosen experts; a batch of 16 or more rows = `forward_experts_prompt`, all experts that received rows. A batch always computes the **union** of its rows' experts. |
| Memory planner | `backend/memory.rs::plan_for_backend` is a layer split: leading layers (attention + all experts) on the device, `FFN_HOST_FROM` layers keep only attention there. It counts the experts of a layer as one block; there is no per-expert placement. A CPU run has no planner and no check against free RAM at all (that is why a model larger than RAM did start, and then relied on the kernel). |
| GPU MoE | `DeviceMoe { experts: Vec<Option<DeviceExpert>>, fast }` is uploaded for whole layers: CUDA pointer tables, Vulkan arenas (`vulkan/backend/experts.rs`), Metal buffers; the grouped kernels index a table of **all** experts of the layer. Layers beyond `offload_layers` run on the CPU from the mapping (decode) or stream expert by expert over PCIe in batches of 32 rows or more (`moe_on_device`, `host_linear`, `FLLM_HOST_STREAM_MIN`). |
| CUDA host streaming and pinned memory | `CudaStream::pin` (`backend/cuda.rs`) page-locks each streamed tensor of the mapping the first time it is used and never unlocks it. Locked pages cannot be dropped, so on a model larger than RAM this grows until the registration fails (then it falls back to pageable copies, `pin_failed`). Not changed here (see section 6). |
| Switches | `FLLM_CUDA_MOE`, `FLLM_MOE_CHUNK_MB`, `FLLM_HOST_STREAM_MIN`, `FLLM_NO_HOST_STREAM`, `FLLM_NO_HOST_PIN`, `--gpu-layers`. Nothing governed host memory for experts. |
| Page-cache behaviour | Lazy. Measured with `fincore` and `/proc/self/io` (RTX 2000 Ada laptop, 28 threads, KIOXIA NVMe): after load 33 MiB of the file is cached; after the first prompt (47 tokens) the whole 3.9 GiB, because a prompt of 8 or more tokens routes to almost every expert of every layer. RSS is mostly file pages (`RssFile` 4.0 GiB, `RssAnon` 0.12 GiB); under a cgroup memory limit those pages are reclaimed by the kernel like any page cache. |

The result of the audit: fLLM already streamed experts lazily on the CPU (the kernel's page cache was the expert cache). What was missing
were the budget, the statistics, a way to see what it costs, a start-up report, and tests that the output does not depend on any of it.

## 2. What was added

`rust/engine/src/expert_cache.rs` (+ `expert_cache/os.rs`), used by `RustDecoder` for every MoE layer whose experts are file data:

* **Registration.** For each routed layer the cache records the address ranges of every expert (gate, up, down) in the mapping and keeps the
  mapping alive. Nothing is copied. Heap-backed tensors (direct-loaded safetensors) are not registered.
* **Plan and budget (`--expert-cache-mb auto|SIZE|off`, `FLLM_EXPERT_CACHE_MB`).** After the layers are placed, the cache adds up the expert
  bytes that stay on the host, compares them with the budget and logs the *speed class*:
  `in-memory` (everything fits), `mostly cached` (>= 75 % of the experts fit), `streaming` (30 % to 75 %), `disk-bound` (< 30 %), with
  the bytes a token needs and the most it reads from storage when the cache is cold. `auto` is 95 % of the available memory (Linux:
  `MemAvailable`, further limited by the cgroup `memory.max` minus anonymous memory and kernel memory; other systems: the free physical
  memory) minus the dense weights (all non-expert tensors) minus a reserve. A number is MiB (`4096`), or `512M`, `8G`. `off` removes the layer.
  The dense part is not checked against the budget: a model starts whenever the file can be mapped; if the dense weights are large for
  the machine a warning says every token pages them too.
* **Explicit budgets are enforced, automatic ones are not.** With an explicit size smaller than the experts, the cache drops experts from
  memory (Linux `MADV_PAGEOUT`, Windows `VirtualUnlock` working-set trim; macOS and BSD: nothing) once the believed-resident bytes pass the
  budget: least recently used first, never the pinned set, never the experts of the pass in progress. The **pinned set** is the hottest
  experts by a decayed per-layer activation count (recomputed every 32 passes), up to 80 % of the budget. An automatic budget is only
  planning and reporting: the kernel already drops the least recently used pages when memory runs short, and dropping earlier measured
  slower (section 4).
* **Read-ahead (all off by default, measured no gain, section 4).** `FLLM_EXPERT_DEMAND=1` asks the system to read all pages of an expert
  as soon as the router has chosen it (`madvise(MADV_WILLNEED)`, Windows `PrefetchVirtualMemory`). `FLLM_EXPERT_PREFETCH=1` adds a thread
  (`fllm-expert-io`) that, once layer L has routed, requests the experts layer L+1 used for the previous pass (the last layer predicts the
  first layer of the next pass); stale requests are dropped. `FLLM_EXPERT_RANDOM_ACCESS=1` sets `MADV_RANDOM` on expert ranges (halves the
  speed, see below). `FLLM_EXPERT_EVICT=0` turns the enforcement off.
* **Many conversations.** A routing call receives the union of the experts of all rows of the batch, which is also what is counted,
  predicted and pinned. 16 simultaneous conversations were run (section 4).
* **Metrics** (`/metrics`, fLLM style; `--metrics-style fllm|both`): `fllm_expert_cache_budget_bytes`, `_expert_bytes`, `_resident_bytes`,
  `_pinned_bytes`, `_streaming`, `_activations_total`, `_probed_total`, `_hits_total`, `_hit_ratio`, `_missing_bytes_total`,
  `_read_bytes_per_token`, `_prefetch_bytes_total`, `_prefetch_useful_total`, `_prefetch_wasted_total`, `_evicted_bytes_total` and, on Linux,
  `_process_read_bytes_total` (the process's `read_bytes` since the model was loaded). A "hit" is a routed expert whose sampled pages (4
  per tensor, `mincore` / `QueryWorkingSetEx`) were in memory when the router chose it; the bytes read are estimated from the share of
  sampled pages that were missing. In a model that fits, only the first 32 passes and every 8th pass are sampled. The Studio footer shows
  `experts: <class>, N% cached, M MB read/token` (from `expert_streaming` in `/api/studio/status`).
* `FLLM_EXPERT_TRACE=FILE` writes `tick layer rows experts` per routing call, to replay cache policies offline.

Not done: the **pinned hot set on the GPU**. The grouped CUDA/Vulkan kernels index one table of all experts of a layer (a missing expert has
no entry), and `DeviceMoe` is immutable after upload, so a hot subset needs a table with holes, the host-routed `moe_on_device` path for
those layers, and device uploads from the forward pass. That is a separate change and could not be tested here (no spare GPU time).
Today a layer beyond the device budget runs its experts on the CPU from the mapping, and this cache manages that memory.

## 3. Platforms

| | read ahead | drop | residency | run here |
|---|---|---|---|---|
| Linux | `madvise(WILLNEED)` | `MADV_PAGEOUT` | `mincore`, `/proc/self/io` | yes (Ubuntu 26.04, x86-64) |
| Windows | `PrefetchVirtualMemory` | `VirtualUnlock` | `QueryWorkingSetEx` (working set, not file cache) | yes (Windows 11, section 4) |
| macOS / BSD | `madvise(WILLNEED)` | none | `mincore` | compiled only through the shared Unix code, not run |

The Windows I/O path is the memory manager's own prefetch (`PrefetchVirtualMemory`) on the mapped view, not `ReadFile` with overlapped I/O:
the weights are mapped, so reading them through a second path would copy them.

## 4. Measurements

Machine A: laptop with an RTX 2000 Ada (not used; CPU only), 28 threads (12 used), 31 GB RAM, KIOXIA 1 TB NVMe. Model: OLMoE-1B-7B-0125-Instruct
Q4_K_M (4.2 GB file, 3.6 GiB of experts in 16 layers, 64 experts, top 8), `--device cpu --threads 12`, one request = 47 to 50 prompt
tokens, then 48 or 64 greedy tokens (measured after the prompt, prefix-cached); each cell 3 to 4 repetitions; the file was dropped from the
page cache before each run (`posix_fadvise DONTNEED`). "Model larger than memory" is emulated with a cgroup limit
(`systemd-run --user -p MemoryMax=N -p MemorySwapMax=0 --scope`, which works on this machine; the page cache counts against it). A second
agent used the machine for GPU benchmarks during part of this, so the spread within one cell (up to 2x) is real noise.

**Disk** (O_DIRECT reads of the model file, MB/s; `bigmoe` python, no fio on the machine):

| | 4 KiB QD1 | 4 KiB QD32 | 128 KiB QD1 | 1 MiB QD1 | 1 MiB QD16 | 4 MiB QD4 |
|---|---|---|---|---|---|---|
| A: KIOXIA NVMe | 45 | 884 | 634 | 2619 | 5270 | 6284 |
| B: SK hynix PVC10 NVMe (Radeon 840M laptop) | 70 | 1107 | 812 | 2613 | 5234 | 5376 |

A token that reads 100 MB of experts therefore costs 20 ms at best on these drives (5 GB/s), but 100 to 200 ms through page faults, which
is what the numbers below show: decoding is limited by how the faults are issued, not by the drive's peak.

**Decode speed against the memory the process may use** (mean of 3, `--expert-cache-mb off` is the earlier behaviour):

| cgroup limit | experts fit | off: tok/s | auto: tok/s | auto: hit ratio | auto: est. MB read / token |
|---|---|---|---|---|---|
| 1 GiB | 13 % | 3.2 | 3.8 | 0.20 | 188 |
| 1.5 GiB | 26 % | 4.0 | 4.5 | 0.36 | 150 |
| 2 GiB | 40 % | 6.9 | 6.0 | 0.51 | 105 |
| 3 GiB | 66 % | 16.5 | 14.9 | 0.79 | 44 |
| 4 GiB | 92 % | 57 | 44 | 0.94 | 15 |
| 6 GiB (fits) | 100 % | 61 | 58 | 0.87 (sampled) | n/a |
| no limit, warm | 100 % | 58 to 66 (12 threads) | | | |

(The "auto" column of this table was run with an earlier build that also dropped experts for an automatic budget; that is why the 4 GiB
case lost a quarter of the speed. The final build does not drop for an automatic budget, section 2. Its 2 GiB run: off 6.2 / auto 5.8
tok/s, 4 repetitions each, inside the noise.) The hit ratio follows the share of experts that fit, as expected, and is better than a
uniform random router would give (at 40 % cached: 0.51 against 0.40), so OLMoE's routing has some locality. The estimated bytes read
agree with `/proc/self/io` to within 10 %: estimated 52.9 GB against 58.1 GB read by the kernel in one 2 GiB run, 3.66 GB against 3.87 GB for the first cold start of a model that fits.

**Ablation at the 2 GiB limit** (decode tok/s, mean of 4): off 6.2 and 6.9 (two runs), all hints on and `MADV_RANDOM` 3.3, prefetch thread only 5.8,
read-ahead of the chosen experts only 5.3, enforced eviction only 7.9, auto (final) 5.8, auto with `FLLM_EXPERT_DEMAND=1` 5.5,
explicit 1 GiB enforced 6.2, explicit 1.5 GiB enforced 6.8. Only `MADV_RANDOM` is clearly worse (it turns the compute threads' faults into
4 KiB reads: 2.7 to 3.3 tok/s against 6 to 9). Everything else is within the noise, so **none of the read-ahead or eviction parts speeds this
model up on this drive**, and they stay off. The prefetch thread's prediction ("same experts as the previous token") was right for 30 % of
the experts it requested (400 of 1312); a replay of the routing of 160 tokens (994 distinct experts) shows what is possible: at 40 % cache
LRU 0.65, frequency-based 0.69, best fixed set 0.72, the offline optimum 0.84 - the policies differ by a few points, the optimum is not
reachable online.

**Same text.** Greedy output of one fixed prompt (64 tokens) has the same hash with `off`, `auto`, an enforced 512 MiB, an enforced 1 GiB and
with all read-ahead switched on (`md5 65b4c237a6db`). The unit tests `*_logits_do_not_depend_on_the_expert_budget` (tiny random Qwen3-MoE
Q4_K_M, DeepSeek Q8_0 and Mixtral Q4_0 checkpoints; 6-token and 40-token prompts, bit-exact logits for budgets of 1 byte, unlimited, off and
with the background thread) pass.

**16 conversations.** Without a limit and warm: 16 conversations x 96 tokens decode at 141 tok/s together (8.8 each; 12 threads); the 16-row
union of a decode step holds about 37 of the 64 experts of a layer (58 %, from the routing trace). At the 2 GiB limit 16 conversations decode at 6.4 to 6.6 tok/s together (about 70 to 85 GB were read from the drive for the 16 prompts and 512 tokens), 4 conversations at 5.7 to 5.8.

**Windows 11** (14 logical processors, 31 GB with 5 GB free): the cross-built `fllm-rs.exe` ran OLMoE Q4_K_M with `--expert-cache-mb 1024`
(28 % of the experts): 1.5 to 1.7 tok/s, hit ratio 0.55 by `QueryWorkingSetEx`, 16.7 GB evicted through `VirtualUnlock`, the same greedy
text as with the cache off (3.1 to 4.4 tok/s) and with `auto` (3.5 to 6.6 tok/s, "fit in memory"). The residency numbers on Windows are
working-set residency: pages in the file cache but not in the working set count as missing, so hit ratios are low-biased there.
`--max-seqs 16` (the default) failed to start on this machine with a 4 GiB allocation error before the model was used (KV pool; not caused
by this change, `--max-seqs 4 --ctx 2048` was used).

## 5. How to use it

```
fllm-rs serve --model big-moe.gguf                       # auto budget, speed class in the log, metrics on
fllm-rs serve --model big-moe.gguf --expert-cache-mb 24G # keep at most 24 GiB of experts in memory
systemd-run --user -p MemoryMax=8G -p MemorySwapMax=0 --scope fllm-rs serve ...   # emulate a smaller machine
```

* Keep the model as a GGUF file on an SSD; a checkpoint is converted first (section 1). Use `--device cpu` or a GPU whose layers do not hold
  the experts; experts uploaded to a device are not host memory and are not counted.
* A prompt processes almost every expert, so the first request of a cold model reads the whole file once (OLMoE: 3.9 GB in 2.4 to 4 s
  here). Long prompts are cheap per token, short answers to short prompts are not.
* Several conversations in one batch read the union of their experts: with a cache smaller than the model, 16 conversations do not
  read 16 times as much as one, but more than half the model per step.

## 6. Open

* Hot experts on the GPU (section 2), and a GPU-side cache of streamed experts.
* CUDA: bound `CudaStream::pin` by the budget when the experts do not fit (written once, not kept: it could not be compiled on a CUDA host
  in this session).
* Prediction of the next layer from the hidden state (a small learned or cached router approximation) instead of the last token's set.
* Tests on a slow drive (SATA, network storage): on NVMe nothing beat the kernel's own LRU; read-ahead may help where latency is high.
* No model larger than the machine's memory was run; the largest real test is OLMoE (4.2 GB) under a 1 GiB limit.
