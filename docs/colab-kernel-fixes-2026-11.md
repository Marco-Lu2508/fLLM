# CUDA kernel follow-up on Colab (2026-11-08)

These are paired measurements, not a new ranking against other engines. The earlier
[comparison](colab-comparison-2026-11.md) used one run per cell and counted SSE messages;
those messages are not a reliable token count if an engine emits multiple tokens in one
message. The follow-up reads `usage.completion_tokens`, records actual prompt token counts,
and switches output caching, prefix caching and speculation off.

## Q2_K and the narrow integer-dot path

`mmvq_x.cuh` decoded each 16-weight run once and reused it for up to four tokens.
At five to sixteen tokens the dispatcher instead selected generic FP32 or FP16-tile
kernels. The integer-dot path now has eight- and sixteen-token variants, including
partly filled variants. The format decoder and per-token accumulation are unchanged.
`FLLM_CUDA_EXACT` still bypasses the approximate fast kernels.

`FLLM_CUDA_MMVQX_MAXB=4` selects the old boundary; the default boundary is 16.
This switch is separate from `FLLM_CUDA_MMVQ_MAXB`, which controls the Q4_K/Q5_K/Q6_K path.

### Measurement

NVIDIA L4, Llama-3.1-8B-Instruct Q2_K, 48 generated tokens per request. Three paired
runs in alternating order, a fresh server per variant. Both variants used the same
binary and prompts. Every recorded request reported zero cached prompt tokens.
The table shows median **whole-request throughput**, including prefill, not decode-only
throughput. The prompt lengths reported by the server were 154 and 2054 tokens.

| Actual prompt tokens | Concurrent users | Old boundary 4, tokens/s | New boundary 16, tokens/s |
|---:|---:|---:|---:|
| 154 | 1 | 50.38 | 50.53 |
| 154 | 8 | 70.17 | 167.51 |
| 154 | 16 | 139.68 | 195.03 |
| 2054 | 1 | 22.31 | 23.26 |
| 2054 | 8 | 26.38 | 33.00 |
| 2054 | 16 | 31.24 | 33.86 |

The larger gain is at short prompts and eight concurrent users; long-prompt prefill
still dominates the larger requests. This is not a new integer tensor-core GEMM for
large prompts.

### Correctness boundary

The L4 `linear` selftest passed 116 checks with no skips for Q2_K, Q3_K, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ1_S. Batch sizes include 5, 8, 12 and 16;
12 exercises a partially filled sixteen-token kernel. References are the CPU and f64
format definitions, with the existing fast-kernel tolerances, not bitwise equality.

Greedy text is **not always identical** to the old FP16/generic path. In a separate
48-request check with five alternative logprobs per position, 36 requests were identical;
12 first diverged at a near tie. Each first divergence had both tokens within 0.138 nats
of the winner on both paths (the existing verification threshold is 0.25 nats).
This checks those prompts only; it is not a general quality or perplexity evaluation.

CUDA kernels were compiled for sm_75 and sm_89. The Q2_K end-to-end measurements above
were on sm_89 only; ROCm, multi-GPU and other GPU architectures were not measured.

## IQ4_XS prefill on the T4: tiles exceeded the shared-memory limit

IQ4_XS is converted to the IQ4_NL device layout at upload. Its prompt projections
use `qgemm_leg.cu`. That dispatch selected a 64-token tile above 32 tokens and a
128-token tile above 64. Even at the minimum two pipeline stages these tiles need
more than the T4's 64 KiB of shared memory, so they declined and the generic decoder
ran instead. This was a resource-limit fallback, not a missing tokenizer or an IQ
weight requantization to Q5_K.

For devices with at most 64 KiB per block, the dispatcher now selects a 128-weight-row
by 64-token tile over four K blocks: 52–60 KiB depending on the format. Larger-memory
devices keep their existing tiles. Small batches through 32 tokens are unchanged.

Tesla T4, Llama-3.1-8B-Instruct IQ4_XS, same fixed binary, three paired runs in
alternating order. `FLLM_CUDA_NO_QGEMM=1` reproduces the generic fallback. Prefix/output
caches and speculation are off, `FLLM_CUDA_PROFILE` is unset (an earlier pair of runs had
it on, which synchronises the stream after every projection; the clean runs reproduce the
same factor), completion usage is 32 tokens per request and cached prompt usage is zero.
Median time to first output delta:

| Actual prompt tokens | Generic fallback, seconds | Fitting tile, seconds |
|---:|---:|---:|
| 149 | 1.354 | 0.259 |
| 2049 | 14.631 | 2.743 |

The T4 `linear` selftest passed 145 checks with no failures or skips for Q4_0, Q4_1,
Q5_0, Q5_1, IQ4_NL, IQ4_XS and MXFP4, including odd row counts and prompt batches.

The two GEMMs sum their products in a different order, so their greedy text is not
guaranteed to agree. Three of the six clean A/B answers were identical and three
diverged (their first differing characters at offsets 7, 39 and 119 of the text). A
five-alternative logprob comparison over those three prompts found the first differing
token at steps 3, 11 and 29, and in every case both paths had the two candidate tokens
within 0.07 nats of each other (for example ` the` -0.8067 against ` ` -0.8219, the other
path ` ` -0.8058 against ` the` -0.8327; the existing verification threshold is
0.25 nats). That is a floating point near tie, not a claim of bitwise equivalence
between the two GEMM kernels.




## AVX2 decode panel width on an EPYC Rome machine

The original matrix showed fLLM's CPU decode behind llama.cpp on Zen 2 (0.58 to 0.81x
per model). One concrete cause was measurable: on a Colab high-memory VM
(AMD EPYC 7B12, family 17h model 31h, Zen 2, eight vCPUs, no AVX-512) the AVX2 K-format
decode ran four output rows per panel.

A temporary `FLLM_TEST_AVX2_PANEL_ROWS` switch in one binary compared panel widths 1, 2
and 4 with otherwise identical code: Llama-3.2-3B Q4_K_M, 149-token prompt, 128 generated
tokens, one user, four threads, caches and speculation off, five alternating runs each,
decode rate computed as `(completion - 1) / (total - first)` after dropping the warm-up
request:

| Panel rows | Median decode, tokens/s | Range |
|---:|---:|---|
| 1 | 14.41 | 14.02–15.21 |
| 2 | 11.97 | 11.49–12.24 |
| 4 | 12.32 | 11.81–12.57 |

The switch is gone. The shipped policy asks CPUID once and keeps the general AVX2 policy
at four rows; only the measured identity (AuthenticAMD, family 17h, model 31h) takes one
row, and only for single-token K-format decode on the AVX2 path — prompt and batched
paths, VNNI and AVX-512 keep what they had. Panel width changes which independent output
rows one worker owns, not the order of the sums of a token, so
`cpu_kquant::tests::batch_one_avx2_panel_sizes_have_identical_bits` compares the raw
`f32` bits of the same weights at panel widths 1, 2, 3 and 4 for Q4_K, Q5_K and Q6_K
(both tests are mutation-checked, and `avx2_decode_panel_policy_only_changes_the_measured_epyc_identity`
fails when the CPUID match is widened to another family or vendor).

Same machine, release build, 3 alternating pairs, one binary with the policy against the
binary whose panel width was pinned to four: median decode **12.75 → 15.02 tokens/s**
(+18 %), identical generated text in all three pairs. Not measured: other Zen 2 models
(only model 31h is matched), Zen 5, and the multi-user throughput of the CPU path.

## Measurement script regression checks

`compare.py` now writes `schema_version: 2`. The per-user rate includes TTFT;
it is no longer labelled post-first-token decode. Missing completion usage leaves
rates null and reports unverified chunk counts separately. The final usage-only SSE
event is read, actual prompt lengths and cached counts are retained, and seeds are
reproducible across engines and A/B runs.

Twenty offline tests pass; all thirteen instrument mutations fail the tests. A smoke
run of the new CLI on the real T4 server reported eight completion tokens, 154 prompt
tokens, zero cached prompt tokens and `token_source: usage`, with the explicit fLLM
cache/speculation controls recorded in the result. Those eight SSE messages happened
to equal eight tokens; the synthetic multi-token-delta tests cover the unequal case.
