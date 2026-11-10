# fLLM against vLLM and SGLang on an A100 (2026-11-09)

First measurement of fLLM next to the two datacenter engines. It is one shared A100-SXM4-40GB on Colab (Intel Xeon host, 12 vCPUs),
one run per cell and engine, and it is **not** the figure for the company server (AMD MI300X / MI355X, 8 GPUs, ROCm): it says where the CUDA
path stands on a checkpoint the three engines read alike.

## Setup

* One public checkpoint, two precisions: `Qwen/Qwen2.5-7B-Instruct` (bf16) and `Qwen/Qwen2.5-7B-Instruct-AWQ` (INT4). vLLM and SGLang serve the
  safetensors; fLLM serves the same files converted with `fllm-rs convert` (the same weights and the same 4-bit scheme).
* fLLM `ad73001` built from `main` for sm_80 with CUTLASS 4.8.0 and FlashInfer 0.6.5, as the release build does; vLLM and SGLang as `pip` installed them that day, with their default settings
  (the versions were not recorded by the script; `compare.py` is changed to record them next time). `scripts/colab/compare.py --seed 11`: prompts of ~100 and ~2000 tokens, 1 and 16
  simultaneous users, 64 generated tokens, server-reported token counts, every prompt different; fLLM with prefix cache, output cache and
  speculation off. Each engine runs in its own session (vLLM in one VM, SGLang in another); fLLM was measured in both and gave the same
  figures within 1 %, so the table shows their mean.
* The first answer to a fixed request is identical for fLLM and vLLM on the AWQ model; on bf16 the two texts diverge after the first words
  (a near tie, not compared further).

## Result: total tokens per second of all users, and fLLM against the best rival of each cell

| Model | Prompt | Users | fLLM | vLLM | SGLang | fLLM / best |
|---|---:|---:|---:|---:|---:|---:|
| AWQ INT4 | 100 | 1 | 163.2 | 171.8 | 186.4 | **0.88x** (SGLang) |
| AWQ INT4 | 100 | 16 | 971.1 | 1791.4 | 868.2 | **0.54x** (vLLM) |
| AWQ INT4 | 2000 | 1 | 104.4 | 126.2 | 130.1 | **0.80x** (SGLang) |
| AWQ INT4 | 2000 | 16 | 213.9 | 343.5 | 346.3 | **0.62x** (SGLang) |
| bf16 | 100 | 1 | 48.0 | 80.5 | 81.8 | **0.59x** (SGLang) |
| bf16 | 100 | 16 | 155.2 | 1075.3 | 988.2 | **0.14x** (vLLM) |
| bf16 | 2000 | 1 | 31.6 | 70.6 | 71.2 | **0.44x** (SGLang) |
| bf16 | 2000 | 16 | 60.8 | 355.9 | 356.3 | **0.17x** (SGLang) |

Time to the first output delta (seconds):

| Model | Prompt | Users | fLLM | vLLM | SGLang |
|---|---:|---:|---:|---:|---:|
| AWQ INT4 | 100 | 16 | 0.29 | 0.19 | 0.80 |
| AWQ INT4 | 2000 | 1 | 0.21 | 0.17 | 0.17 |
| AWQ INT4 | 2000 | 16 | 2.41 | 1.36 | 1.43 |
| bf16 | 100 | 1 | 0.11 | 0.03 | 0.03 |
| bf16 | 2000 | 1 | 0.75 | 0.14 | 0.14 |
| bf16 | 100 | 16 | 0.90 | 0.15 | 0.25 |
| bf16 | 2000 | 16 | 6.63 | 1.08 | 1.15 |

**fLLM is behind the best engine in all eight cells.** The one cell above 1.00x in the SGLang run (AWQ, 16 users, short prompt, 1.14x) is the
cell where SGLang is slow (868 against vLLM's 1791); against the best rival it is 0.54x.

## What the numbers point at (to be checked, not yet explained)

* **bf16 is the large gap.** One user decodes at 48 tokens/s; the 15.2 GB of weights at the A100's 1.55 TB/s allow about 100, and vLLM
  reaches 80. With 16 users fLLM gains only 3.2x over one user (155 against 48); vLLM gains 13x.
* **bf16 prefill is slow:** a 2055-token prompt takes 0.75 s (about 42 TFLOP/s of the 312 the card has), vLLM 0.14 s. The time to the first token
  with 16 users and a long prompt (6.6 s against 1.1 s) follows from it.
* **AWQ at one user is close** (0.80-0.88x), at 16 users with long prompts it falls to 0.62x, with short prompts to 0.54x of vLLM.
* The earlier [comparison with llama.cpp and ik_llama.cpp](colab-comparison-2026-11.md) put fLLM ahead on the GGUF formats; the engines that
  datacenters run (vLLM, SGLang) read bf16 and AWQ/FP8 checkpoints, where this table shows the gap. The next step is a per-kernel profile on this
  card (`FLLM_CUDA_PROFILE=1`) for the bf16 projection and attention paths, before any change.

## After the first fixes (same card, same models, `4e97034`)

The profile (`FLLM_CUDA_PROFILE=1`, nsys with `--cuda-graph-trace=node`) found three causes, which are fixed in `4978046` (cuBLASLt from 16 tokens and a
GEMV with four runs in flight), `4e97034` (sampling on the device) and the commit after them (a tensor-core GEMV for 1-8 tokens, cuBLASLt from 9). The
same compare script, vLLM and SGLang run again next to it (raw results of the vLLM and the SGLang run in `compare_result.json`):

| Qwen2.5-7B-Instruct | fLLM before | fLLM `4e97034` | best rival | ratio before | ratio now |
|---|---|---|---|---|---|
| bf16, 100-token prompt, 1 user | 48 | 76.9 | 81.8 (SGLang) | 0.59 | 0.94 |
| bf16, 100, 16 users | 155 | 756 / 749 | 1078 (vLLM) | 0.14 | 0.70 |
| bf16, 2000, 1 user | - | 59.7 | 71.7 (SGLang) | - | 0.83 |
| bf16, 2000, 16 users | - | 201 | 362 (SGLang) | - | 0.56 |
| AWQ, 100, 1 user | - | 167 | 171 (vLLM) | 0.80-0.88 | 0.98 |
| AWQ, 100, 16 users | - | 1240 | 1789 (vLLM) | 0.54 | 0.69 |
| AWQ, 2000, 1 user | - | 107 | 132 (SGLang) | - | 0.81 |
| AWQ, 2000, 16 users | - | 228 | 353 (SGLang) | 0.62 | 0.65 |

(The two fLLM columns come from one run each; the vLLM and the SGLang run each measured fLLM as well, the values differ by about 1 %.) Against SGLang alone
fLLM is ahead at 16 users with short prompts (AWQ 1.43x, bf16 1.22x).

Still behind, in the order of the gap: long prompts with 16 users (prefill of 33k tokens: 0.56-0.65x), 16 users with short prompts against vLLM (0.69-0.70x:
the CUDA-graph launch of a step takes 1.3 ms during which the GPU idles, cuBLASLt's split-K reduction and the per-projection conversion of the
activations cost 8 % of the GPU time), one user with a long prompt (0.81-0.83x).

## Second round (bf16, same card)

A 2039-token prompt takes 164 ms of GPU time in total, of which the GEMMs are 107 ms (247 TFLOPS of the 312, the same cuBLAS kernels vLLM uses),
FFN gate / activation conversions 22 ms, prompt attention 7 ms, the rest 28 ms; the changes of this round took about 6 ms off the time to the first token. A decode step with 16 users
is 15.6 ms (see the changelog for the breakdown): at 11.9 ms the GEMMs stream the weights at the card's practical bandwidth, so what is left is the
host-side path between two steps (about 1.4 ms) and about 2 ms of small kernels. Two attempts to shorten that path did nothing and were dropped: cutting the recorded graph into four (81.7 -> 79.5 tokens/s for one user) and moving the four staged-array copies out of the graph (81.8 -> 81.9); the trace shows why: the GPU cannot start before `cudaGraphLaunch` returns (0.55-0.7 ms for about 680 nodes), so the cost is the launch call itself, and only launching the next step before the host knows the sampled ids (asynchronous scheduling) or far fewer nodes would remove it. 16 users with short prompts: 875 tokens/s (0.81 of vLLM's 1078), one user 82 tokens/s
(vLLM 80.6, SGLang 81.8), four users 257 and eight users 471 sampled tokens/s (from 129 and 132 at the start of the day).

Same compare script on `e2c3fe8`, vLLM and SGLang each in a run of their own (fLLM tokens/s of the run that measured the rival, the best rival of each cell):

| Qwen2.5-7B-Instruct | prompt / users | fLLM | best rival | ratio |
|---|---|---|---|---|
| bf16 | 100 / 1 | 80.4 | 81.9 (SGLang) | 0.98 |
| bf16 | 100 / 16 | 790 | 1076 (vLLM) | 0.73 |
| bf16 | 2000 / 1 | 64.3 | 71.4 (SGLang) | 0.90 |
| bf16 | 2000 / 16 | 229 | 355 (vLLM, SGLang) | 0.65 |
| AWQ | 100 / 1 | 167.6 | 185.7 (SGLang) | 0.90 |
| AWQ | 100 / 16 | 1244 | 1770 (vLLM) | 0.70 |
| AWQ | 2000 / 1 | 106.7 | 130.1 (SGLang) | 0.82 |
| AWQ | 2000 / 16 | 225 | 345 (SGLang) | 0.65 |

The AWQ path (Marlin) was not touched in this round. The SGLang value of the bf16 100 / 16 cell moved from 614 to 1005 tokens/s between two runs of the same script, so single cells of
this table carry a spread of that size for the rival engines.

## Third round: 40 users, and why the 16-user cells fell behind when the 40-user cell was added (2026-11-10)

Same card and models, vLLM 0.33.0 and SGLang 0.5.23 (one run each, the rival column is the best of the two), users 1, 16 and 40. The first fLLM run of the day (`d561d6e`, asynchronous decode scheduling in) showed
the 16-user cells *slower* than the day before (bf16, 2000-token prompts: 181 against 230 tokens/s). The cause was not the new code: `compare.py` starts fLLM with `--ctx (max prompt + tokens + 400) x max users`,
so a 40-user cell raises `--ctx` from 39k to 98k, and every token row of a prompt chunk carried a block table of `--ctx / 16` entries (100 MB per 2000-token chunk to build and upload). With the
same binary and the same load, `--ctx 4096` gave 274 tokens/s, `--ctx 40000` 231 and `--ctx 98560` 178; the kernels were the same (3.35 s against 3.56 s of GPU time). The tables are now packed to the blocks of the longest row
(see the changelog); fLLM `compare.py` figures after that, against the best rival of each cell:

| Qwen2.5-7B-Instruct | prompt / users | fLLM | best rival | ratio |
|---|---|---|---|---|
| bf16 | 100 / 1 | 80.9 | 81.6 (SGLang) | 0.99 |
| bf16 | 100 / 16 | 852 | 1092 (vLLM) | 0.78 |
| bf16 | 100 / 40 | 1451 | 1958 (vLLM) | 0.74 |
| bf16 | 2000 / 1 | 65.3 | 71.5 (SGLang) | 0.91 |
| bf16 | 2000 / 16 | 270 | 356 (vLLM) | 0.76 |
| bf16 | 2000 / 40 | 308 | 411 (vLLM) | 0.75 |
| AWQ | 100 / 1 | 170.5 | 183.9 (SGLang) | 0.93 |
| AWQ | 100 / 16 | 1373 | 1806 (vLLM) | 0.76 |
| AWQ | 100 / 40 | 1760 | 2822 (vLLM) | 0.62 |
| AWQ | 2000 / 1 | 112.4 | 126.6 (vLLM) | 0.89 |
| AWQ | 2000 / 16 | 276 | 352 (SGLang) | 0.78 |
| AWQ | 2000 / 40 | 283 | 358 (vLLM) | 0.79 |

The two engines are given a per-sequence limit of 2464 tokens (`--max-model-len`), fLLM `--ctx` times the users (a pool for all sequences together, split by the server); the run before the packing showed how much that
mattered for fLLM and not for them. Still behind everywhere except one user with a short prompt. Where the time goes (bf16, 16 users, 2000-token prompts, A100): a 2039-token prompt costs 164 ms of GPU time, 107 ms of it GEMMs at 247 TFLOPS (the same
kernels vLLM uses) and 57 ms around them (gated-activation conversions 22 ms, prompt attention 7 ms, the rest 28 ms), against about 140 ms of time to first token for vLLM; a decode step with 16 rows is 14.4 ms, 11.9 ms of it GEMMs at the weight-streaming limit.

Later the same day (`727c450`: prompt chunks gather their embedding rows on the device, decode attention on the tensor cores, shared 16-bit conversion, split-column GEMV), the same `compare.py` run on the same card, against the rival figures above:

| Qwen2.5-7B-Instruct | prompt / users | fLLM | ratio | | prompt / users | fLLM | ratio |
|---|---|---|---|---|---|---|---|
| bf16 | 100 / 1 | 82.1 | 1.01 | AWQ | 100 / 1 | 174.8 | 0.95 |
| bf16 | 100 / 16 | 871 | 0.80 | AWQ | 100 / 16 | 1432 | 0.79 |
| bf16 | 100 / 40 | 1519 | 0.78 | AWQ | 100 / 40 | 1852 | 0.66 |
| bf16 | 2000 / 1 | 67.3 | 0.94 | AWQ | 2000 / 1 | 118.6 | 0.94 |
| bf16 | 2000 / 16 | 299.5 | 0.84 | AWQ | 2000 / 16 | 306.6 | 0.87 |
| bf16 | 2000 / 40 | 345.1 | 0.84 | AWQ | 2000 / 40 | 315.4 | 0.88 |

What is left, from nsys of a decode step (16 rows, 2000-token context): about 12.4 ms of GEMMs at 1.1 TB/s (vLLM's whole step is about 12.3 ms), 420 kernel launches (the norm, rope, GLU and conversion kernels between the GEMMs add up to about 2.5 ms), and for AWQ with many users the dequantizing GEMM.

Not measured: FP8, GPTQ, MoE models, a longer output, more than 16 users, other cards, ROCm, more than one GPU, quality (only the first answer was compared).
Raw results: `compare_result.json` of both runs (`schema_version 2`, `vs_best_rival` per engine).
