# Prefill performance notes (CUDA)

Where the prompt pass of the CUDA backend loses time that the kernels do not need, what was changed about it, and how to measure each
change. File and line references name the code of `a086f0f` (the state before these changes) unless a function name is given.

Reference measurement that started this (RTX PRO 6000 Blackwell, Qwen2.5-3B AWQ, 1863-token prompt, 5 requests under nsys): wall time
70 ms per request, GPU kernel time 52 ms, 8987 launches (about 50 per layer, 2 chunks of at most 1024 rows); vLLM needs 48 ms. The GPU is
idle 27 % of the wall time, so the host side is the first suspect; the kernel efficiency gap comes second.

## 1. The prompt pass, as it runs

`serve` hands a prompt to the scheduler in chunks of `--max-batched-tokens` rows (512 by default, raised to 1024 or 2048 by the memory
planner, `bin/serve.rs`). Every chunk is one `forward_backend_inner` call (`rust_decoder.rs:2307`): embeddings on the host, one upload,
a loop over the layers that only enqueues kernels on one stream, final norm, and the LM head for the rows that need logits. Only the
last chunk of a prompt downloads anything (`rust_decoder.rs:3058`); chunks before it return after the last launch.

Launches per layer for a 4-bit AWQ/GPTQ model with an f16 KV cache and FlashInfer prompt attention (Qwen2.5-3B shape):

| # | kernel | note |
|---|--------|------|
| 1 | `rms_norm` (attention norm) | `elementwise.cu` |
| 2 | f32 -> f16 conversion of the normed rows | `marlin_vllm_glue.cu` `f32_to_f16_kernel`, reused by k and v |
| 3-5 | Marlin q, k, v | one extra launch each when the row count is not a multiple of 64 |
| 6 | `rope_kv_write_f16` | bias, RoPE and KV-cache write in one kernel |
| 7-11 | `table_to_int32`, `float_to_half` (q), prefill attention (+ merge kernel when the KV range is split), `half_to_float` | `flashinfer_prefill.cu` |
| 12 | f32 -> f16 conversion of the attention result | output projection input |
| 13 | Marlin o | |
| 14 | `f16_to_f32_acc` (`x += scale * (y16 + bias)`) | |
| 15 | `rms_norm` (FFN norm) | |
| 16 | f32 -> f16 conversion | |
| 17-18 | Marlin gate, up (FP16 results) | |
| 19 | SwiGLU folded into the conversion for the down projection | |
| 20 | Marlin down | |
| 21 | `f16_to_f32_acc` | |

That is 21 launches (22 with the attention merge) plus the Marlin tail launches, two chunks of a 1863-token prompt: about 50 per layer
and chunk pair, which matches the nsys count. Per layer there are also 3 pageable host-to-device copies in FlashInfer's request
description, and `stage()` (below) for positions, slots, lengths and block tables.

## 2. Host-side costs per layer, chunk and request

Everything below happens while the GPU may already be running earlier kernels of the same chunk, so it only shows up as idle time when the
host is slower than the GPU for a stretch. Rows x stride terms scale with the prompt length, which is where the gap opens.

| where (a086f0f) | what | cost | status |
|---|---|---|---|
| `api.cu:976-977` -> `stage()` `api.cu:262-309`, per layer | the block tables of all rows (`rows x stride x 8` bytes, stride = max context / 16) are compared with the staged copy (`memcmp`) | O(rows x stride) per layer | fixed: staged once per step (`FLLM_PREFILL_STEP_CACHE`) |
| `api.cu:986-1004`, per layer | run detection: `memcmp` of every row's table against the first row's | O(rows x stride) per layer | fixed: computed once per step; one table per run is staged |
| `api.cu:282-290` | a table block above half the staging ring: `cudaStreamSynchronize` + `cudaFree` + `cudaMalloc` (ring = 4 x block) and a copy into a `std::vector` | once per growth, but every chunk with more rows than any before | avoided: runs stage `stride x 8` bytes |
| `cuda.rs:2518` -> `mod.rs` `AttentionParams::validate`, per layer | walks the blocks of every row | O(rows x blocks) per layer, quadratic in the prompt | fixed: walked for the first layer of a step only |
| `runtime.rs:724-732,834,847`, per chunk | `table.clone()` for every row (`stride x 8` bytes each) and `runtime.rs:902-907` flattening | O(rows x stride) per chunk | clone removed (`Arc`); the flattening stays |
| `rust_decoder.rs:2397`, per chunk | one `Vec` per token for the embedding row, single thread; the device waits for it | O(rows x hidden) before the first kernel | fixed: all cores (`FLLM_PREFILL_PAR_EMBED=0` keeps the old loop) |
| `rust_decoder.rs:2430` | upload of `rows x hidden x 4` bytes from pageable memory | one copy per chunk | not changed |
| `marlin_vllm.cu:244-251,364`, per GEMM | three `cudaDeviceGetAttribute` and one `cudaFuncSetAttribute` | about 700 x 4 driver calls per chunk | fixed: cached (`FLLM_PREFILL_STEP_CACHE`) |
| `flashinfer_prefill.cu:134-136`, per layer | three pageable `cudaMemcpyAsync` with the same bytes in every layer | 3 driver calls per layer | fixed: copied when rows, pages or last-page length change |
| `flashinfer_prefill.cu:146,158` | two `cudaStreamSynchronize` when the plan changes | once per chunk | not changed (the pipeline is drained once per chunk, after the first layers' kernels were queued) |
| `cuda.rs:1346,1852-1853,2623`, per layer | `Vec<i32>` copies of positions and slots | O(rows) | not changed (the staging dedupe makes the device copy free) |
| `cuda.rs:1772,1797,1899,1924`, `rust_decoder.rs:2835,2980`, per layer | `std::env::var_os` (environment lock) | ~10 per layer | fixed: read once |
| `api.cu:337` | `download` = copy + `cudaStreamSynchronize` | once per request (logits) | needed |
| `api.cu:377,443,511,563,849` | `cudaStreamSynchronize` + `cudaFree` + `cudaMalloc` when a scratch buffer grows | only when a chunk needs more than any before | not changed |
| `BufferPool::acquire` | `cudaMalloc` when no pooled buffer is large enough | only when a chunk needs more than any before | not changed |

Not present in the prompt pass: a synchronisation between layers, a device-to-host read before the last layer, a per-layer allocation
once the pools are warm, and any per-row host loop in the layer body.

Estimate (not measured): with a 4096-token context the block tables are 256 entries per row (2 KB). A 1024-row chunk then compares and
scans about 2 MB three times per layer (`stage` compare, run detection, plus the copy into the staging vector on a miss), roughly 6 MB x 36
layers = 216 MB of memory traffic per chunk at 10-20 GB/s single-thread memcmp: 10-20 ms per chunk, the size of the idle time that
nsys shows. With the 32k default context of Qwen2.5 (stride 2048) it would be eight times that. The step cache turns this into one pass.

## 3. Switches

All are read from the environment at process start; `selftest-gpu` flips them at run time. `1` is on.

| variable | default | what |
|---|---|---|
| `FLLM_PREFILL_STEP_CACHE` | 1 | block tables, lengths, run layout, Marlin attributes, FlashInfer request description: once per step instead of once per layer |
| `FLLM_PREFILL_PAR_EMBED` | 1 | prompt embeddings on all cores |
| `FLLM_FUSE_PREFILL_ADD_NORM` | 1 | residual add of a Marlin output projection done inside the RMS norm that follows (section 4) |
| `FLLM_FUSE_PREFILL_NORM_F16` | 1 | RMS norm also writes the FP16 rows the next Marlin GEMM needs (section 4) |
| `FLLM_FUSE_PREFILL_ATTN_F16` | 1 | prompt attention also writes the FP16 rows the output projection needs (section 4) |
| `FLLM_MARLIN_PREFILL_CFG` | `vllm` | `vllm`: Marlin launch configuration exactly as vLLM selects it; `fllm`: no second launch for the M tail (section 6) |
## 4. Kernel fusions in the prompt pass

Three fusions remove launches and one pass over the activations each. All three are bit-identical to the kernels they replace (same
arithmetic, same reduction order, same FP16 rounding), apply only to batches of 16 rows or more, and leave decode steps on the old
paths.

| switch | what | removed per layer |
|---|---|---|
| `FLLM_FUSE_PREFILL_ADD_NORM` | the output projection of a Marlin GEMM no longer adds its result to the residual stream at once; the add waits until the RMS norm that follows and happens in the norm's pass (what vLLM's `fused_add_rms_norm` does). Anything else that comes first flushes the add. | 2 launches (after `o_proj` and `down`) |
| `FLLM_FUSE_PREFILL_NORM_F16` | the RMS norm also writes the FP16 rows the next Marlin GEMM needs into the Marlin scratch and marks them as the converted activations of its output buffer, so the GEMM skips its conversion kernel. A different weight format in between, an activation permutation or a scratch reallocation fall back to the normal conversion. | 2 conversions (attention input, FFN input) |
| `FLLM_FUSE_PREFILL_ATTN_F16` | the FlashInfer path writes its FP16 attention result to the Marlin scratch while expanding it to FP32, so the `o_proj` GEMM skips the FP32 -> FP16 conversion (the round trip FP16 -> FP32 -> FP16 is exact). Only when one run covers all rows. | 1 conversion |

Already fused before this work and left alone: bias + RoPE + KV-cache write (`rope_kv_write_f16`), SwiGLU into the conversion of the
down projection (`glu_pending`), the residual add into the GEMM output conversion (`fllm_q4z_acc`). Together the prompt pass
needs 16 launches per layer instead of 21. What is not fused: the q conversion for FlashInfer (`float_to_half`) because the
rotated queries are produced in FP32 by the RoPE kernel and the half buffer belongs to FlashInfer's state; QK-norm models (no
fused RoPE path at all); non-Marlin weight formats.

The decode path uses the same kernels for single rows. The prompt fusions here are generic over the row count but gated at 16 rows
(`FLLM_PREFILL_MIN_ROWS`), so a decode fusion of the same pattern can coexist; `rms_norm_prompt_kernel` (`elementwise.cu`) is the
shared piece if both want it.

## 5. CUDA graphs for prompt chunks: not implemented

`FLLM_CUDA_GRAPH_PREFILL` does not exist. The numbers say the host launch rate is not what leaves the GPU idle, and the code is not
set up for it:

* Launch cost. 8987 launches / 5 requests = 1797 per request. At 3-5 us of host time per `cudaLaunchKernel` that is 5-9 ms per
  request, spread over 70 ms in which the GPU queue is normally full; launch overhead alone cannot produce 18 ms of idle time. The
  per-layer O(rows x stride) host work of section 2 can, and it is gone with the step cache. If the idle share is still above about
  10 % after the Colab run, the next step is a graph; until then it would hide a smaller part of the problem.
* Shapes. Rows, KV length, the FlashInfer plan and Marlin's M split differ for every chunk. vLLM's piecewise graphs capture the
  GEMM/norm pieces at padded bucket sizes and keep attention outside; fLLM would have to pad rows, mask the padding in the KV write and
  attention, and keep one graph per bucket and per Marlin launch layout.
* Capture rules. `stage()` while capturing accepts only the four fixed decode arrays (`api.cu:263-278`), FlashInfer's plan calls
  `cudaStreamSynchronize` (`flashinfer_prefill.cu:146,158`) and copies from pageable memory, and attention is switched off for the
  tensor-core prompt kernels under capture (`api.cu:985`). Each needs its own work before a prompt chunk can be recorded.

## 6. Marlin for large M

Compared with vLLM's current `csrc/quantization/marlin/marlin.cu` (read on 2026-10-14): `marlin_vllm/marlin_vllm.cu` is that host
code with the tensor glue replaced, and it makes the same choices: `max_par` 16 (128 for `n <= 4096`), `max_thread_m_blocks` 4, the
thread configurations `{64,256,256} {64,128,128} {128,64,128}` for more than 16 rows, the {128,64,128} switch when
`n / thread_n * m_tiles * 4 <= SMs`, one block per SM, no atomic add (vLLM leaves it off unless `VLLM_MARLIN_USE_ATOMIC_ADD` is set),
FP32 reduction with a `sms x 64 x 256` float scratch, a zeroed lock workspace. `FLLM_MARLIN_PREFILL_CFG=vllm` (the default) is that
selection, unchanged.

What is not free in it is the M split. A launch handles a whole number of 64-row tiles (at most `max_par`), and the remaining
rows go into a second launch with a smaller tile. The second launch reads every weight again: for 1863 rows the gate/up GEMMs
(`n = 11008`, at most 1024 rows per launch) run as 1024 + 832 + 7 rows, the others as 1856 + 7. `FLLM_MARLIN_PREFILL_CFG=fllm`
rounds batches above 64 rows up to a multiple of 64 and runs them as one sequence of full tiles. The activation and result scratch has
room for the padding rows, whose results are never read. Cost: up to 63 padded rows of compute; saving: a launch and a full weight
read per GEMM with a tail. Estimate: 11 MB of weights at 1.4 TB/s is 8 us plus launch and ramp, 7 GEMMs x 36 layers: 3-4 ms of the 52 ms
kernel time per request. Not measured. The rows of the tail are computed with a 4-tile configuration instead of the 1-tile one, so
results differ from `vllm` in the last FP16 bits; `selftest-gpu` holds them to the fast-kernel tolerance against the CPU reference.

The Q4_K/Q5_K/Q6_K/Q8_0 tensor-core GEMMs were reviewed only at the dispatch level: their tile choice is autotuned on the first call per
(kernel, shape, 64-row bucket) with events and `cudaEventSynchronize` (`fllm_cuda_internal.h`, `fllm_autotune`), which drains the
stream once per new shape during the first requests (warm up before timing). Their launch configurations were not changed.

## 7. Checks and the Colab A/B run

`fllm-rs selftest-gpu --device cuda --only prefill` (fast pass) runs, on the device only, every variant against the plain one with the
switch flipped at run time: the step cache around two-layer prompt attention (one prompt, two prompts, a prompt next to decode rows,
a long prompt), add + norm, add + norm + Marlin, add + norm + Q4_K + Marlin, add + norm + act-order GEMM, attention + `o_proj` + norm
(each batch size 16/40/129 or 24/48/130 rows): the outputs must be bit-identical. The Marlin tail variant runs the four Q4Z formats at
65, 100, 129, 300 and 1000 rows against the CPU reference with the fast-kernel tolerance.

Not covered by the self test: the FlashInfer path itself (only when the build found the headers), the NVFP4 prompt path (untouched), and
the host-side effects (embedding threads, `Arc` tables), which only change timing.

One Colab session, same model, same prompts, best of 3 per length (128, 355, 1000, 1863, 4000, 8000), `--max-batched-tokens 1024`
and the context the earlier runs used:

```sh
fllm-rs selftest-gpu --device cuda --only prefill,linear,attention      # must end with 0 failed
# 0: everything off = the numbers of 410b815 plus nothing
export FLLM_PREFILL_STEP_CACHE=0 FLLM_PREFILL_PAR_EMBED=0 FLLM_FUSE_PREFILL_ADD_NORM=0 FLLM_FUSE_PREFILL_NORM_F16=0 FLLM_FUSE_PREFILL_ATTN_F16=0 FLLM_MARLIN_PREFILL_CFG=vllm
# 1: host work only
export FLLM_PREFILL_STEP_CACHE=1 FLLM_PREFILL_PAR_EMBED=1
# 2: + fusions (the defaults)
export FLLM_FUSE_PREFILL_ADD_NORM=1 FLLM_FUSE_PREFILL_NORM_F16=1 FLLM_FUSE_PREFILL_ATTN_F16=1
# 3: + Marlin tail
export FLLM_MARLIN_PREFILL_CFG=fllm
```

Restart the server for each step (the switches are read at start-up) and warm it with one request of each length first (autotuning and
buffer growth happen in the first requests). `nsys profile --stats=true` on step 0 and step 2 shows the idle share: kernel time stays,
wall time should fall toward it.

Test level of everything in this file: L0 (compiles for `sm_75` and `sm_120`, `cargo test` and `cargo clippy` pass). No kernel has run.
