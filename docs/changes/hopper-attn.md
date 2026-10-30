# Hopper and FlashInfer attention (FA3 prefill, variants, tensor-core decode, cascade)

Test level of everything below: **L0** (compiled with CUDA 12.8 and 13.0 for the architectures named in the table at the end; the self-test group
`hattn` exists and was never run on a GPU). Nothing here was executed on a device.

## What the audit found (checked in the source)

* `attention4.cu` (decode, `rows < 16` per run or any run the prompt kernels decline) is a flash-decoding kernel on CUDA cores: one block per
  (KV head, row, split), 16 lanes per key row, the `G` query heads of a KV head share each loaded key and value row (`G` in 1..8, head widths
  64 / 128 / 256, window / chunk / soft cap / sinks). Long contexts are split over blocks and merged by `attn4_merge_kernel` through a global
  workspace. It does no tensor-core work: every query-key product is an FMA, so a GQA group of 4 .. 8 does 4 .. 8 times the FMAs a `mma`
  tile of 16 query rows would do for the same key loads.
* `attention3.cu` (prompt) is the built-in tensor-core kernel (`mma.sync`, Ampere-class). For f16 caches without sinks `flashinfer_prefill.cu`
  runs FlashInfer's FA2 kernel for **one** sequence per call, no sliding window, no soft cap, no sinks, head widths 64 / 128, never on decode.
  Prompts with a window, soft cap, sinks, or several sequences in one batch go to `attention4.cu` (one row per block).
* Hopper (sm_90) ran only Ampere-class kernels for attention.

## Features

### FA3 prefill on Hopper (`cuda/fa3_prefill.cu`)

FlashInfer's port of FlashAttention-3 (`flashinfer/attention/hopper/*`: warp-specialized producer / consumer warp groups, `wgmma`, TMA for Q
and O, cp.async for the paged K / V) for the f16 paged cache, causal, head widths 64 / 128 / 256, optional sliding window, one or several
sequences per launch (FlashInfer's `PrefillSM90Plan`). The file is compiled for `sm_90a` only (`build.rs`, `Flavor::Hopper`); other builds get a
stub that declines every call. The query is converted from FP32 to FP16 around the call and the result expanded, exactly as the FA2 path does,
including the FP16 copy for the Marlin output projection (`fllm_attn_x16`) and the pending-attention hand-over of the NVFP4 fusion.

`FLLM_ATTN_BACKEND=auto|fa2|fa3` (default `fa2` = the behaviour before FA3 existed, until the `hattn` group has passed on an H100): `auto` and `fa3` use FA3 on compute capability 9.0 for the covered shapes, `fa2` keeps the
older kernels. A shape FA3 does not cover (sinks, soft cap, `kind` other than f16, key width other than 64 / 128 / 256, decode rows) falls through
to the previous path.

Not done: bf16 (the cache is f16, converting only Q and O would not match the K / V type), FP8 Q / K / V (stretch goal), soft cap and sinks in FA3
(they need an FA3-style `OnlineSoftmax` variant with an extra `AdditionalParams`; FlashInfer ships one for sinks in `jit/attention/variants.py`,
not wired here), `-DFLLM_HAVE_FLASHINFER` hopper TMA loads for the paged cache (FlashInfer itself uses cp.async there).

### FlashInfer FA2 with variants, ragged batches, decode and cascade (`cuda/fi_attn.cu`)

One `Variant` with run-time flags (the FlashInfer variant interface: `LogitsTransform`, `LogitsMask`, `update_m_d`, `OutputTransform`):

* sliding window: `window_left = window - 1` (the engine's window counts the current position), applied by the kernel's mask and its tile skipping;
* logits soft cap: `cap * tanh(s / cap)` of the scaled score, as in `attention4.cu`;
* attention sinks: the per-head sink logit joins the softmax denominator once, in the first KV chunk of a request, following FlashInfer's
  `AttentionSink` variant; the pointer rides in `maybe_alibi_slopes` of the parameters (ALiBi is not used on this path);
* several requests per launch: every run of consecutive rows of one sequence (same block table, lengths growing by one, the grouping the engine
  already uses) is one FlashInfer request; the plan and the device copy of the page tables are made once per distinct batch (hash of the
  description), not once per layer;
* decode: runs of one row use FlashInfer's 16-row query tile. The query heads of a KV head are packed into the M dimension of the `mma`, which is
  FlashInfer's tensor-core decode for GQA groups of 4 .. 8 (also works for groups 1, 2, 3, 5, 6, 7); long contexts are split and merged by
  FlashInfer's own merge kernel, as `attention4.cu` does with its merge kernel. FlashInfer's CUDA-core `BatchDecodeWithPagedKVCache` is not used
  (it is the same arithmetic as `attention4.cu`);
* cascade (`FLLM_CASCADE_ATTN=1`): decode rows whose block tables start with the same pages (at least `FLLM_CASCADE_MIN_PREFIX` tokens, default
  128, and one more key after the prefix) form a group. The group's query rows are one non-causal request on the shared pages (the prefix is
  read once for all of them), each row's own suffix is a causal request, and the two states (output and base-2 log-sum-exp) are merged with the
  log-sum-exp rule in `merge_scatter`. Sinks join the suffix state only, so they are counted once.

The path is not used while a CUDA graph is captured (the plan synchronizes with the host). It covers f16 caches, key width = value width in
{64, 128, 256}, no ALiBi, no chunked attention; head width 256 has no non-causal instance, so cascade needs 64 or 128.

Defaults keep today's behaviour: `FLLM_ATTN_FI` is off until the path is verified on hardware (its `auto` equals 0), decode stays `builtin`,
cascade stays off.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `FLLM_ATTN_BACKEND` | `fa2` | `auto` / `fa3`: FA3 for prompts on compute capability 9.0 (f16 cache, head width 64 / 128 / 256, no sinks / soft cap); `fa2`: never FA3 |
| `FLLM_ATTN_FI` | `auto` (= `0`) | `1`: prompt runs with sliding window / soft cap / sinks and batches of several runs go through FlashInfer (FA3 when it covers them, else FA2 with the variant) |
| `FLLM_ATTN_DECODE` | `builtin` | `flashinfer`: decode rows (and speculative rows, runs shorter than 16) use FlashInfer's tensor-core kernel |
| `FLLM_CASCADE_ATTN` | `0` | `1`: decode rows sharing a prefix of cache pages use cascade attention |
| `FLLM_CASCADE_MIN_PREFIX` | `128` | shortest shared prefix, in tokens |
| `FLLM_CUDA_NO_FLASHINFER` | unset | existing switch, also turns the new FlashInfer / FA3 paths off |
| `FLLM_SELFTEST_BENCH` | unset | `1`: `selftest-gpu --only hattn` adds timing cases at model sizes |

Build (unchanged otherwise): `third_party/flashinfer` (v0.6.5) and `third_party/cutlass` (v4.8.0) headers; without CUTLASS the FA3 file is the
declining stub, without FlashInfer both new files are.

## Verification on Colab (not run yet)

Build once per runtime (the architecture list can be narrowed for speed, `FLLM_CUDA_ARCHS=<cc>`), with `rust/engine` of this tree:

```
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
git clone --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer third_party/flashinfer
FLLM_CUDA_ARCHS=90 cargo build --release --features cuda      # H100: 90 (builds sm_90 and the sm_90a FA3 file); A100: 80; L4: 89; G4 (RTX PRO 6000): 120
target/release/fllm-rs devices
target/release/fllm-rs selftest-gpu --only hattn --json hattn.json
FLLM_SELFTEST_BENCH=1 target/release/fllm-rs selftest-gpu --only hattn --json hattn-bench.json
```

Expected outcome of `--only hattn`: every line is `ok` or `SKIPPED`, `failed = 0`. Each line's name ends with the path that ran:

| device | `fa3 ...` cases | `window / soft cap / sinks / ragged prefill`, `decode ...`, `speculative rows` | `cascade ...` |
|---|---|---|---|
| H100 (9.0) | `[Fa3]`, ok | `[FlashInfer]`, ok | `[Cascade]`, ok |
| A100 (8.0), L4 (8.9), G4 (12.0) | `SKIPPED` (no FA3 on this device) | `[FlashInfer]`, ok | `[Cascade]`, ok |
| T4 (7.5, optional) | `SKIPPED` | `[FlashInfer]`, ok; head width 256 was not examined for Turing shared memory: report a failure of the d256 lines | `[Cascade]`, ok |

Measured: RTX 2000 Ada laptop (8.9, FlashInfer headers, CUDA 12.4) 2026-10-30, `--only hattn`: 28 checks, 21 ok, 7 skipped (the FA3 cases), 0 failed; the four cascade cases ran the `[Cascade]` path. Before that day the cascade cases failed: the merged FP16 rows were never converted into the FP32 output of the call (it kept the old contents), and a run too short to share the prefix kept the rest of its group from sharing it.

A case that reaches a different path than its label expects fails with `expected the X path to run, the Y path ran`, so a silent fall back to the
built-in kernel cannot pass. The tolerance is the fast-kernel one (`4e-3` of the error scale; FP16 queries and probabilities).

With `FLLM_SELFTEST_BENCH=1` the timing lines (`hattn bench ...`, column `device ms` = one call averaged over 12 calls after 3 warm-up calls, no
reference) compare: prompt 2048 of 4096 tokens, FA2 vs default (FA3 on H100); window 1024 and sinks prompts, built-in vs FlashInfer; decode 32 x 4096,
8 x 16384, 32 x 2048 builtin vs flashinfer; 64 sequences with a 2048-token shared prefix, builtin vs cascade. What to expect (estimates, not
measurements): FA3 faster than FA2 on H100 for long prompts; FlashInfer variants much faster than `attention4.cu` for windowed / sink prompts;
tensor-core decode near `attention4.cu` for short contexts, faster for GQA 4 .. 8 at long contexts; cascade faster than plain decode as the prefix
share and the batch grow. If tensor-core decode is not faster, `FLLM_ATTN_DECODE` stays `builtin`.

A text check on a real model on H100 (L2): run any f16-KV model with `FLLM_ATTN_BACKEND=auto` (FA3) and with the default (`fa2`) and compare greedy output, e.g.
`FLLM_ATTN_BACKEND=auto fllm-rs run <model> ...` against the default; for the variants repeat with `FLLM_ATTN_FI=1 FLLM_ATTN_DECODE=flashinfer`
on a sliding-window model (Mistral / Gemma / gpt-oss) against the default.

## THIRD_PARTY entries

* FlashInfer (https://github.com/flashinfer-ai/flashinfer, v0.6.5), Apache-2.0, Copyright (c) 2023 by FlashInfer team: the FA2 prefill kernel and
  planner (`attention/prefill.cuh`, `scheduler.cuh`), the variant interface and the `AttentionSink` variant (`jit/attention/variants.py`), the
  Hopper FA3 kernels (`attention/hopper/*`) and planner (`PrefillSM90Plan`), the usage pattern of `csrc/batch_prefill_sm90.cu`; used as headers
  (`third_party/flashinfer`), `cuda/fi_attn.cu` and `cuda/fa3_prefill.cu` prepare their inputs and define one run-time variant.
* FlashAttention-3 (Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao), BSD-3-Clause, as carried in FlashInfer's Hopper
  files.
* CUTLASS / CuTe v4.8.0, BSD-3-Clause (headers used by the Hopper kernels; licence in `licenses/CUTLASS-BSD-3-Clause.txt`).
* vLLM `cascade_attention` (Apache-2.0): the idea of attending a shared prefix once for all sequences that share it and merging with the suffix by
  log-sum-exp; the implementation here is new, built on FlashInfer requests.

## Test levels and limits

* L0: `cuda/fa3_prefill.cu` compiles for `sm_90a` (CUDA 12.8 and 13.0; 16 kernels in the `sm_90a` ELF, none elsewhere) and as the stub for
  `sm_75 .. sm_120`; `cuda/fi_attn.cu` compiles with CUDA 12.8 for sm_75 / 80 / 86 / 89 / 90 / 120 and with CUDA 13.0 for sm_75 / 100 / 110 / 120;
  `cargo clippy --features cuda` clean (kernels skipped); the `hattn` self-test compiles.
* L1 / L2 / L3: not run. Nothing proves the numerics of any new kernel call, the sink / soft-cap variant, the cascade merge, or the plan cache.
* Risks: FA3 `LEFT_SLIDING_WINDOW` with the sparse paged mainloop and the `window_left` convention (`window - 1`) are unchecked; the sink handling in
  split-KV chunks relies on FlashInfer applying `update_m_d` per chunk (read in `prefill.cuh`, not run); the plan cache keys on a 64-bit hash of the
  request description; the extra FlashInfer workspaces (64 MB float + two 8 MB integer plans) and FA3 buffers are allocated on first use per thread.
