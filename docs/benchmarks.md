# Benchmarks against other engines (2026-10-29)

Two Linux laptops, one night, the same client for every engine. Read the limits at the end before quoting a number.

## Method

`scripts/bench_compare.py` starts each engine once, loads the same model file, warms it up and sends it three requests of its own:

* a text of about 380 random words (546 tokens, a different one per request, so no prompt cache can help) followed by "continue the story",
* 128 new tokens, temperature 0, `ignore_eos`, streamed through `/v1/completions` of the OpenAI interface, which all of them speak;
* **prompt speed** = prompt tokens / time to the first token (it includes producing the first token), **decode speed** = (new tokens - 1) / (last token - first token).
  Both are medians of the three requests.

GGUF engines get the same `.gguf` file (Q4_K_M from bartowski): llama.cpp `llama-server`, Ollama (the file imported with a Modelfile), fLLM. For
vLLM and SGLang, which do not run Q4_K_M, all three engines get the same 4-bit AWQ checkpoint (AMead10/Llama-3.2-3B-Instruct-AWQ). FastFlowLM, the
engine of the AMD NPU, runs its own NPU-specific model files (`llama3.2:3b`, `llama3.1:8b`): same model family, **not the same weights**. Context
4096 everywhere, one request at a time.

## Machines

| | NVIDIA laptop | AMD laptop |
|---|---|---|
| Model | Dell Precision 7680 | Dell Pro 5 14 |
| CPU | i7-13850HX (AVX2, no AVX-512), 30 GB | Ryzen AI 5 PRO 435 (Zen 5, AVX-512), 14 GB |
| GPU | RTX 2000 Ada Laptop, 8 GB, sm_89 | Radeon 840M (4 CUs, RDNA 3.5, `gfx1153`), memory shared with the system |
| Driver | NVIDIA 595.99.02 | Mesa 26.0.8 (RADV), ROCm 7.1 (Ubuntu packages) |
| NPU | none | AMD XDNA 2 (`aie2p`, 8 columns, firmware 1.1.2.64) |
| OS | Ubuntu 26.04, kernel 7.0 | the same |

Versions: fLLM `001a96b` (CUDA test build) and `79d550e` (Vulkan, HIP), llama.cpp b11988 (release binaries: CPU, Vulkan, CUDA 12.8, ROCm),
Ollama 0.40.13, vLLM 0.32.0, SGLang 0.5.22, FastFlowLM 1.0.11.

## NVIDIA RTX 2000 Ada laptop (8 GB)

Llama-3.2-3B / Llama-3.1-8B Q4_K_M. Prompt tok/s / decode tok/s.

| Engine, backend | 3B | 8B |
|---|---|---|
| **fLLM, CUDA** | **3838** / 92.5 | 1481 / **44.6** |
| llama.cpp, CUDA | 3353 / **93.5** | 1591 / 42.9 |
| Ollama, CUDA | 3644 / 92.5 | **1733** / 42.8 |
| llama.cpp, Vulkan | 2694 / 83.2 | 1275 / 39.1 |
| fLLM, Vulkan (native) | 1569 / 54.5 | 631 / 31.6 |
| **fLLM, CPU** | **267** / **32.8** | **120** / **15.6** |
| llama.cpp, CPU | 169 / 25.5 | 71 / 11.8 |

Llama-3.2-3B, the same 4-bit AWQ checkpoint:

| Engine | Prompt tok/s | Decode tok/s |
|---|---|---|
| **fLLM, CUDA** | **5244** | **99.5** |
| SGLang | 4904 | 95.2 |
| vLLM | 4792 | 93.0 |

Reading: on CUDA the decode speeds are within a few percent of each other (the memory bandwidth of the card is the limit), fLLM is ahead on the 3B
prompt, and Ollama and llama.cpp are ahead on the 8B prompt. fLLM's CPU kernels are 30 to 70 % faster than llama.cpp's on this machine. The
Vulkan backend is the weak spot: 35 % behind llama.cpp's Vulkan decode on this card (cause not found; the work-group size of the matrix-vector
shader was swept and is not it).

### Prompt length and simultaneous requests (2026-10-30)

The same client (`scripts/bench_compare.py --prompt-words 60,380,1500 --concurrency 1,4,8`, tables by `scripts/bench_table.py`), every request a new
random text, 128 new tokens, medians of three runs (one burst per cell for 4 and 8 requests), context sized for the longest prompt, `OLLAMA_NUM_PARALLEL` /
`--parallel` = the largest burst. "(1)" = one request alone: prompt tok/s, decode tok/s, time to the first token (TTFT). "(4)" and "(8)" = that many requests
sent at the same moment: **aggregate** generated tokens per second (all tokens / time until the last request ended) and the **median** TTFT.
Llama-3.1-8B was measured with 1 and 4 requests only (8 GB card). The prompts are ~130, ~580 and ~2100 tokens (the tokenizers differ by a few percent).
Versions: llama.cpp b11988, Ollama 0.40.13, vLLM 0.32.0, SGLang 0.5.22. fLLM = this tree: the CUDA binary was built on the laptop (sm_89 only, FlashInfer
prompt attention, CUDA 12.4); the "before" rows are the `colab-latest` CUDA build of commit `ea04c42` (sm_75/80/89/120), fLLM Vulkan is the native
backend (`vulkan:1`) of the same tree. The GPU was otherwise idle, one engine at a time; expect a few percent of noise between sessions (the "before"
and "now" CUDA rows for the same code differ by up to 4 %).


#### Llama-3.2-3B-Instruct-Q4_K_M


~60 words (~130 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 2663.0 | 97.2 | 0.04 | 269.9 | 0.21 | 399.6 | 0.38 |
| fLLM CUDA, before (0.1.12 build) | 2638.9 | 96.9 | 0.04 | 266.9 | 0.21 | 396.1 | 0.38 |
| **fLLM Vulkan** | 1813.8 | 76.5 | 0.06 | 142.0 | 0.23 | 182.2 | 0.38 |
| llama.cpp CUDA | 2272.3 | 94.6 | 0.05 | 154.1 | 0.19 | 315.3 | 0.35 |
| llama.cpp Vulkan | 1550.0 | 82.8 | 0.07 | 109.9 | 0.6 | 108.0 | 2.68 |
| Ollama | 2704.1 | 93.8 | 0.05 | 82.7 | 4.19 | 127.8 | 4.93 |

~380 words (~581 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 4289.0 | 91.3 | 0.13 | 221.1 | 0.46 | 293.2 | 0.73 |
| fLLM CUDA, before (0.1.12 build) | 4209.9 | 91.0 | 0.13 | 212.3 | 0.52 | 275.2 | 0.86 |
| **fLLM Vulkan** | 2660.0 | 74.5 | 0.21 | 108.0 | 0.83 | 133.2 | 1.29 |
| llama.cpp CUDA | 3323.9 | 91.5 | 0.16 | 191.8 | 0.55 | 227.4 | 1.21 |
| llama.cpp Vulkan | 2733.1 | 80.4 | 0.2 | 128.5 | 0.68 | 120.6 | 2.25 |
| Ollama | 3875.6 | 91.1 | 0.15 | 198.5 | 0.4 | 227.9 | 0.84 |

~1500 words (~2117 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 4903.5 | 89.5 | 0.42 | 131.0 | 1.32 | 155.3 | 2.18 |
| fLLM CUDA, before (0.1.12 build) | 4822.1 | 89.3 | 0.43 | 102.8 | 1.94 | 111.0 | 3.59 |
| **fLLM Vulkan** | 2816.0 | 68.3 | 0.74 | 64.0 | 2.2 | 74.0 | 4.67 |
| llama.cpp CUDA | 3383.8 | 85.5 | 0.61 | 109.2 | 1.94 | 115.7 | 3.41 |
| llama.cpp Vulkan | 2948.7 | 75.3 | 0.7 | 82.7 | 2.21 | 82.5 | 3.88 |
| Ollama | 4166.5 | 85.2 | 0.51 | 117.8 | 1.35 | 126.4 | 2.66 |

#### Meta-Llama-3.1-8B-Instruct-Q4_K_M


~60 words (~140 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 1145.5 | 46.3 | 0.09 | 134.5 | 0.51 | - | - |
| fLLM CUDA, before (0.1.12 build) | 1136.2 | 46.3 | 0.09 | 134.1 | 0.52 | - | - |
| **fLLM Vulkan** | 698.4 | 39.3 | 0.15 | 68.0 | 0.59 | - | - |
| llama.cpp CUDA | 1312.0 | 43.5 | 0.08 | 121.1 | 0.34 | - | - |
| llama.cpp Vulkan | 789.6 | 38.9 | 0.14 | 71.6 | 1.35 | - | - |
| Ollama | 1659.8 | 43.4 | 0.09 | 118.3 | 0.38 | - | - |

~380 words (~599 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 1689.7 | 44.4 | 0.32 | 106.8 | 1.16 | - | - |
| fLLM CUDA, before (0.1.12 build) | 1688.1 | 44.4 | 0.32 | 105.0 | 1.22 | - | - |
| **fLLM Vulkan** | 996.2 | 41.5 | 0.55 | 54.3 | 2.21 | - | - |
| llama.cpp CUDA | 1594.5 | 42.7 | 0.34 | 94.7 | 1.22 | - | - |
| llama.cpp Vulkan | 1252.8 | 38.3 | 0.44 | 67.3 | 1.8 | - | - |
| Ollama | 1887.0 | 42.6 | 0.31 | 96.2 | 0.89 | - | - |

~1500 words (~2158 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 1974.3 | 44.0 | 1.05 | 52.8 | 4.06 | - | - |
| fLLM CUDA, before (0.1.12 build) | 1969.7 | 44.0 | 1.05 | 45.5 | 4.88 | - | - |
| **fLLM Vulkan** | 1094.4 | 33.2 | 1.89 | 29.7 | 7.13 | - | - |
| llama.cpp CUDA | 1598.1 | 41.1 | 1.3 | 54.3 | 4.25 | - | - |
| llama.cpp Vulkan | 1352.0 | 37.1 | 1.53 | 42.4 | 5.0 | - | - |
| Ollama | 2038.1 | 41.0 | 1.04 | 60.3 | 2.84 | - | - |

#### Llama-3.2-3B, the same 4-bit AWQ checkpoint for all engines


~60 words (~96 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 4215.9 | 104.5 | 0.03 | 329.1 | 0.1 | 522.9 | 0.18 |
| fLLM CUDA, before (0.1.12 build) | 4176.6 | 104.4 | 0.03 | 328.7 | 0.1 | 516.4 | 0.18 |
| vLLM | 2988.8 | 94.7 | 0.04 | 339.9 | 0.1 | 631.7 | 0.18 |
| SGLang | 3527.2 | 96.8 | 0.03 | 223.3 | 0.92 | 645.7 | 0.12 |

~380 words (~547 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 6182.1 | 97.5 | 0.09 | 267.4 | 0.35 | 369.1 | 0.57 |
| fLLM CUDA, before (0.1.12 build) | 5805.8 | 97.3 | 0.09 | 254.8 | 0.41 | 343.9 | 0.69 |
| vLLM | 5366.8 | 92.2 | 0.1 | 269.0 | 0.39 | 412.6 | 0.51 |
| SGLang | 5491.3 | 94.7 | 0.1 | 275.2 | 0.39 | 430.5 | 0.68 |

~1500 words (~2083 prompt tokens)

| engine | prompt tok/s (1) | decode tok/s (1) | TTFT s (1) | agg tok/s (4) | TTFT s (4) | agg tok/s (8) | TTFT s (8) |
|---|---|---|---|---|---|---|---|
| **fLLM CUDA** | 6449.6 | 95.1 | 0.32 | 158.5 | 1.01 | 189.7 | 1.68 |
| fLLM CUDA, before (0.1.12 build) | 6375.6 | 94.8 | 0.32 | 118.9 | 1.64 | 123.4 | 3.18 |
| vLLM | 6222.6 | 86.4 | 0.33 | 158.7 | 0.97 | 198.1 | 1.69 |
| SGLang | 5381.1 | 87.9 | 0.39 | 151.8 | 0.99 | 194.6 | 1.87 |

#### Reading the numbers

* **CUDA, one request.** Decode speeds are within 3 to 4 % of each other on every model (the memory bandwidth of the card is the limit). The prompt of
  fLLM is the fastest for the 3B model at every length (4289 / 4904 against 3876 / 4167 tok/s for Ollama at 580 / 2100 tokens) and, on the 4-bit AWQ
  checkpoint, ahead of vLLM and SGLang too. On Llama-3.1-8B it is **10 % behind Ollama at 600 tokens (1690 against 1887) and 3 % at 2100 tokens** (in tokens each engine reports: Ollama counts 34 more tokens for the same text, the time to the first token is within 3 %, see the second pass below) and
  ahead of llama.cpp. An int8 tensor-core GEMM for Q4_K that exists in the tree (`cuda/qmma4k.cu`, never built or called) was wired in for a test: after one
  fix (it ignored the fused gate/up activation) it wrote the right text but was slower than the FP16 kernel on this card (1736 against 2017 tok/s on the
  8B prompt). It stays unused; the gap is the Q4_K prompt GEMM.
* **CUDA, simultaneous requests.** The aggregate rises with 4 and 8 requests (3B, short prompts: 97 -> 270 -> 400 tok/s; Ollama 94 -> 83 -> 128) and is the
  best of the GGUF engines everywhere except **Llama-3.1-8B with four 2100-token prompts: 53 against 60 (Ollama) and 54 tok/s (llama.cpp)**; the median
  TTFT there (4.1 s) is also worse than Ollama's (2.8 s). On the AWQ checkpoint fLLM equals vLLM and SGLang with 4 requests, is 10 % behind at 8
  requests with a medium prompt (369 against 413 / 431) and 17 % behind with short prompts (523 against 632 / 646). Profiled: the weight kernels there already
  stream at the card's bandwidth; what remained was the decode attention kernel (about 100 us per layer for 8 sequences), fixed in the second pass below (8 short requests: 530 -> 655 tok/s, vLLM 632).
* **No batching collapse on this card**, neither with CUDA nor with Vulkan: the aggregate grows with the number of requests in every row of the tables
  (Vulkan 3B: 76 -> 142 -> 182 tok/s). The collapse reported for a Radeon 840M (28 -> 10 tok/s with 8 requests) does not occur here.
* **Vulkan on the NVIDIA card** is within 3 to 5 % of llama.cpp's Vulkan prompt on the 3B model (2660 / 2816 against 2733 / 2949 tok/s) and about 20 %
  behind on the 8B model (996 / 1094 against 1253 / 1352). Decode: 8 % behind on the 3B model (76.5 against 82.8), ahead on the 8B model (41.5 against 38.3).
  With several requests fLLM's Vulkan aggregates more than llama.cpp's for short prompts (182 against 108 with 8 requests), about the same for medium ones,
  and less for 2100-token prompts (74 against 82). CUDA is 1.3 (decode) to 1.7 (prompt) times faster than Vulkan on the same card.
* **What changed for these numbers** (see the changelog): (1) CUDA: a batch that held one prompt chunk and the decode rows of other requests sent the
  **whole** batch, prompt rows included, to the per-row decode attention kernel (36 % of the GPU time); the prompt rows now keep the tensor-core kernel
  (3B, four 2100-token requests: 103 -> 131 tok/s aggregate, TTFT 1.9 -> 1.3 s; 8 requests: 111 -> 155 tok/s). (2) Vulkan: the prompt attention runs on
  cooperative matrices (4.9 ms -> 0.8 ms per layer and 512 rows at 2000 tokens of context) and the matrix product uses a 128 x 64 tile with a 64-wide K
  step, a prefetch of the next step and binary16 activations (about 1.3 times faster than the 64 x 64 tile). Llama-3.2-3B, prompt of 2100 tokens:
  1431 -> 2816 tok/s; decode at that context 55 -> 68 tok/s.
* **Still open** (see the second pass below for what was found since): the CUDA Q4_K prompt GEMM (8B) and with it the aggregate of several long 8B prompts; the Vulkan matrix product reaches
  roughly 14 TFLOPS (llama.cpp's about 20): an experiment without the weight fetch was 1.8 times faster, so the packed-weight fetch and decode in front of the
  tensor cores bound it, not the tensor cores; Vulkan decode of `Q6_K` (the output projection) streams at about 155 GB/s against 230 GB/s for the Q4_K kernels.

### Second CUDA pass (2026-10-30, same RTX 2000 Ada laptop)

Profiled with `nsys` (`ncu` is not available without root: the driver restricts the GPU performance counters), then changed. Same client and method as above
(`scripts/bench_compare.py`, medians of three runs, 128 new tokens, every request a new text); "before" = the CUDA binary of the previous section, "now" = this
tree, both built on the laptop (sm_89, FlashInfer headers, CUDA 12.4) and measured in the same session, vLLM 0.32.0 in the same session too. **The laptop is shared
with other test jobs; no other job used the GPU during these runs as far as `nvidia-smi` and `ps` showed, but a few percent of noise between runs of one
session is normal** (the 3B AWQ one-request prompt row moved from 4441 to 4174 tok/s between "before" and "now" on a 25 ms time to first token).

**What the profile showed.** (1) The decode attention kernel took about 91 us per layer for eight sequences of 200 tokens. A CUDA graph is recorded for the longest
context of the pool (32768 tokens for the 3B model), and the kernel sized its grid and its merge kernel for that: 64 splits for every row and head, 4096 blocks
of which 63 in 64 had no key to read, plus a merge over 64 partial results. The rows now use the splits their own length needs and the other blocks return
at once (about 1024 blocks at most, a row of one split writes its result directly): **Llama-3.2-3B AWQ, 8 simultaneous short requests: 530 -> 655 tok/s
(vLLM 632 in the same session)**. (2) The weight kernels of both models stream at the card's bandwidth (mmvq / Marlin, in the 8B run 22.7 ms of a 28 ms decode
step for 4 sequences read 5 GB of weights once), the GPU is 97 to 99 % busy in every burst of the 8B runs (no host gaps, sampling runs on the device), and
CUDA graphs were already on for batched decode steps; nothing is left to win from launch or scheduler overhead there.

Llama-3.2-3B-Instruct, the 4-bit AWQ checkpoint (aggregate tokens/s of 1, 4 and 8 simultaneous requests; decode tok/s of one request in brackets):

| prompt | engine | 1 request | 4 requests | 8 requests |
|---|---|---|---|---|
| ~60 words (~100 tokens) | fLLM before | (106.0) | 335.0 | 530.4 |
| | **fLLM now** | (109.4) | **374.7** | **655.3** |
| | vLLM | (94.9) | 341.9 | 632.2 |
| ~380 words (~546 tokens) | fLLM before | (99.0) | 268.9 | 376.1 |
| | **fLLM now** | (107.1) | **286.2** | **425.2** |
| | vLLM | (92.5) | 270.2 | 421.1 |

(SGLang was not run again; its row in the table above, 646 tok/s for 8 short requests, was measured in a different session.)

Llama-3.2-3B-Instruct Q4_K_M (GGUF, same session, ran after the AWQ runs; the GPU showed 13 % utilisation from another job once before the run started):

| prompt | build | decode tok/s (1) | agg tok/s (4) | agg tok/s (8) |
|---|---|---|---|---|
| ~60 words | before | 96.6 | 264.7 | 393.3 |
| | **now** | **100.9** | **293.3** | **450.4** |
| ~380 words | before | 90.3 | 216.1 | 281.3 |
| | **now** | **98.9** | **231.7** | **314.9** |


**Llama-3.1-8B Q4_K_M, the two open rows.** Measured against Ollama in one session (the previous section's numbers came from another one, and both engines were slower in this one: Ollama's time to the first token at 600 tokens was 0.31 s there, 0.36 s here):

| prompt | engine | prompt tokens counted | prompt tok/s | TTFT s | decode tok/s (1) | agg tok/s (4) | TTFT s (4) |
|---|---|---|---|---|---|---|---|
| ~380 words | fLLM before | 546 | 1471 | 0.37 | 44.5 | 101.4 | 1.44 |
| | **fLLM now** | 546 | 1500 | 0.36 | 46.8 | 104.7 | 1.41 |
| | Ollama | 580 | 1583 | 0.36 | 43.0 | 94.5 | 1.09 |
| ~1500 words | fLLM before | 2092 | 1998 | 1.04 | 44.0 | 53.8 | 3.98 |
| | **fLLM now** | 2092 | 2027 | 1.03 | 44.9 | 54.2 | 3.96 |
| | Ollama | 2126 | 2023 | 1.05 | 41.2 | 60.6 | 2.81 |

* **The "10 % behind Ollama at 600 tokens" was mostly a counting difference.** The same text is 544 tokens for llama.cpp's tokenizer and for fLLM's (checked on five of
  the benchmark prompts: identical counts), but Ollama reports 580 (and 2126 for the 2092-token prompt: 34 more at both lengths, presumably a template wrapper it adds; not examined) and the table divides that count by the time. The
  **time to the first token for the same text is within 3 %**: 0.36 s for both here (0.32 against 0.31 s in the previous session). Counted per token Ollama is 5 % faster
  at 600 tokens (1583 against 1500 for tokens it really processes: 580 against 546) and equal at 2100 tokens (2023 against 2027). The change in this pass did not
  touch the Q4_K prompt GEMM, so this stays the one place where a better kernel (int8 tensor cores: `cuda/qmma4k.cu` exists and was slower on this card when it was tried) would pay.
* **Four simultaneous 2100-token prompts: 54.2 against 60.6 tok/s (Ollama)** remains. `nsys` of that run: GPU 97 to 99 % busy; the decode phase of the four
  sequences takes 27.9 ms per step (bandwidth bound: 5 GB of weights plus 1.1 GB of key/value cache in 224 GB/s is 27 ms), the prompt phase 5.6 s for
  8400 tokens (1500 tok/s; one request alone reaches 2027, so 25 % slower per token in this phase: a prompt step of 512 rows takes 330 ms next to other
  sequences, about 250 ms alone, the GEMM kernels themselves run longer, the laptop GPU at its 60 W limit is the suspect and could not be separated from the
  other causes without hardware counters). Ollama's lead there is the prompt phase, not the scheduler or the launches.
* **Only the attention fix changed the 8B numbers, by 1 to 4 %** (decode of 4 sequences: 35.8 -> 37.0 tok/s each at 600 tokens): at 4 rows its 32 blocks of
  one or three splits were not the problem.

## AMD Radeon 840M laptop

Llama-3.2-3B / Llama-3.1-8B Q4_K_M. Prompt tok/s / decode tok/s. The 840M has 4 compute units and takes about 5 GB of the 14 GB for a model,
so the 8B does not fit and every engine splits it between GPU and CPU.

| Engine, backend | 3B | 8B |
|---|---|---|
| FastFlowLM, **NPU** (own model files) | **306** / **31.2** | **207** / **14.0** |
| llama.cpp, ROCm (`HSA_OVERRIDE_GFX_VERSION=11.5.1`) | 300 / 20.1 | not run |
| llama.cpp, Vulkan | 287 / 28.6 | 124 / 13.2 |
| Ollama, ROCm (same override) | 224 / 15.5 | not run |
| **fLLM, Vulkan (native)** | 213 / 28.4 (2026-10-30, second pass: 265 / 29.5) | 75 / 8.5 (second pass: 98 / 15.0) |
| fLLM, HIP (ROCm) | 157 / 22.0 | 67 / 9.8 |
| fLLM, CPU | 146 / 16.7 | 65 / 7.8 |
| Ollama (CPU; it does not take `gfx1153`) | 122 / 19.0 | 51 / 8.7 |
| llama.cpp, CPU | 112 / 19.0 | 47 / 8.7 |

Reading: on this small iGPU the NPU is fastest, llama.cpp's Vulkan and ROCm paths come next, fLLM's native Vulkan is on par for decode of the 3B (the 8B,
which is split between GPU and CPU, is 36 % behind) and 25 to 40 % behind for the prompt. All of these beat the laptop's own CPU for the prompt; for
decode the CPU reaches 55 to 95 % of the GPU paths. The numbers say nothing about a
Ryzen AI Max (Strix Halo: 8 to 10 times the compute units and twice the memory bandwidth); it was not available.

## Intel Core Ultra 5 235U laptop (Windows 11, no discrete GPU)

Measured 2026-10-30 with the Windows cross build of fLLM (`fllm-rs.exe`, features `rust-gpu,vulkan`, version 0.1.12 plus the changes of that day),
`scripts/bench_compare.py` on Windows (python 3.12), Q4_K_M files, 64 new tokens (not 128), 3 requests per row, medians, one request at a time, context 4096.
Machine: Core Ultra 5 235U (2 performance + 8 efficiency + 2 low-power cores, 14 threads, AVX2 and AVX-VNNI, no AVX-512), 32 GB RAM, Intel Graphics
(Arrow Lake-U iGPU, Vulkan 1.3, driver 32.0.101.8991, memory shared with the system), Intel AI Boost NPU. Prompt tok/s / decode tok/s; the prompt rows are
about 111 tokens (60 words) and 545 tokens (380 words).

| Model | Device | prompt 111 tokens | prompt 545 tokens | decode (545-token prompt) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | CPU | 125 | 196 | 29.0 (20.6 in the short-prompt run) |
| Qwen2.5-0.5B-Instruct Q4_K_M | **Vulkan (native, iGPU)** | 434 | 625 | 49.2 |
| Llama-3.2-1B-Instruct Q4_K_M | CPU | 117 | 149 | 20.9 (22.7 in the short-prompt run) |
| Llama-3.2-1B-Instruct Q4_K_M | **Vulkan (native, iGPU)** | 229 | 411 | 29.9 |

Reading: the iGPU is 1.5 to 2.8 times faster than the CPU for the prompt and 1.4 to 2.4 times for decode. CPU decode varies between runs on this hybrid processor
(the same request gave 20.6 and 29.0 tokens/s for the 0.5B model, depending on which core types the worker threads landed on; `--pin-cores` and `--threads` were not tuned).
The built-in `fllm-rs bench` (256-token prompt, chat endpoint, Qwen2.5-0.5B) gave 247 prompt tokens/s on the CPU and 1273 on the Vulkan iGPU; its prompt is shorter and it times
differently, so do not compare it with the table. The NPU path (OpenVINO, prefill matmul offload only) was run through `selftest-gpu` only, not timed; no llama.cpp or Ollama
number exists for this machine.

## What the night changed in fLLM

The benchmarks and the tests that came with them found these, all fixed in the same state (`CHANGELOG.md`):

* Vulkan: a whole model step was one GPU job, and the driver ended it as "device lost" for a prompt of about 400 tokens on this iGPU (on Windows the
  2-second TDR does the same). A job now holds at most 1.5e11 multiply-adds.
* The memory plan sent feed-forward weights to the host when the whole model would have fitted with a shorter context (Llama-3.1-8B on an 8 GB
  card: 19 tok/s before, 45 after with CUDA; 10 to 14 before, 32 to 34 after with Vulkan).
* Vulkan prompt matrix shaders: cooperative matrices are 2.2 times faster on the NVIDIA card, the int8 dot product 22 % faster on the Radeon. The backend
  now times both once per device (a few milliseconds) and keeps the faster one.
* Vulkan decode: two matrix-vector rows per group on AMD instead of one (+12 % on the 840M).
* HIP/ROCm: first run on an AMD GPU, built for `gfx1153` with the Ubuntu packages (no rocWMMA needed since 2026-10-30).
  Dense models give the CPU's text; MoE models and host-streamed layers fail with "invalid argument" on this APU.

## AMD Radeon 840M laptop, second pass (2026-10-30): concurrent users, the 8B model, what did not help

Same laptop, same client (`scripts/bench_compare.py`, medians of 2 runs, 128 generated tokens, prompts of 60 and 380 random words = about 110 and 550 tokens; the
`agg` figures are all generated tokens of the simultaneous requests divided by the wall time **including the prompt phase**). The laptop was shared with other
jobs during the runs and the GPU is power-limited (about 17 W for the whole package, clock 1.4 - 1.65 GHz while busy): single runs of the same binary differ by
5 to 15 %, so read differences below that as noise. Before = the commit before this pass, after = this pass.

| Llama-3.2-3B Q4_K_M, total tok/s | 4 requests, 60 words | 8 requests, 60 words | 4 requests, 380 words | 8 requests, 380 words |
|---|---|---|---|---|
| fLLM Vulkan before | 22.2 - 23.5 | 11.4 - 12.8 | 14.1 - 14.5 | 9.5 - 9.7 |
| **fLLM Vulkan after** | 42.2 | 35.0 | 24.2 | 22.1 |
| fLLM HIP before | 46.9 | 24.4 | 26.0 | 13.0 - 17.5 |
| **fLLM HIP after** | 46.9 | 60.6 | 26.0 | 30.0 |
| llama.cpp Vulkan (measured earlier the same day) | 37.1 | 59.0 | 30.0 | 34.4 |
| llama.cpp ROCm | 27.9 | 38.1 | 17.9 | 16.5 |
| Ollama (Vulkan) | 36.0 | 38.1 | 18.2 | 18.5 |

A single request (3B, 380 words): Vulkan prompt 215 - 243 -> 265 tok/s, decode 29.4 -> 29.5; HIP prompt 208 -> 224 - 242 tok/s with the FP16-activation GEMM (the last measurement ran while other jobs loaded the laptop; one earlier run read 157), decode 22.0 unchanged; 8 requests of 380 words 30.0 -> 34.3 tok/s in total. Llama-3.1-8B Q4_K_M, Vulkan,
547-token prompt: prompt 74.6 -> 98.0 tok/s, decode 8.5 -> 15.0 (llama.cpp Vulkan 124 / 13.2; the two fLLM figures were measured twice with 3 runs each and agreed to 0.1).

What the numbers say, and how it was found:

* **Eight users were slow because 9 to 31 activation rows left the fast kernels.** A decode step of eight sequences plus speculative drafts has 9 or more rows. On
  Vulkan those went to the old `mm_<fmt>_i` kernels (4 to 5 ms per projection of the 3B model; `FLLM_VK_PROFILE=1` showed them as 67 % of the GPU time of a step with 8
  sequences), on HIP to the FP32 GEMV (1 ms per Q4_K and 3.9 ms per Q6_K projection at 8 rows; `FLLM_CUDA_PROFILE=1`, new). Vulkan now uses the tiled kernel from 9 rows on; HIP the
  int8-dot kernel up to 16 rows (0.37 ms and 0.67 ms at 8 rows). Speculative drafting was not the cause (8 requests with `--speculative off`: the same).
* **What is left on Vulkan** (8 requests, 380 words: 22 against llama.cpp's 34): a decode step with 8 rows costs about 150 ms of GPU time, of which the matrix-vector kernel
  is 95 ms (0.45 TMAC/s of int8 dot products; the 4096 x 4096 Q4_K prompt timing is 0.8 - 0.9 TMAC/s on zero data, lower on real data because of the power limit).
* **The 8B model now stays on the GPU.** The memory plan only counted the device-local Vulkan heap (5.1 GB of the 7.6 GB the GPU can address on this 14 GB laptop) and moved 21 of 32
  layers' feed-forward weights to the CPU. It now counts the host-visible heap too (see the changelog). With the system's other processes using 6 GB or more
  of the RAM the same binary was 40 % slower in some runs and the planner fell back to the CPU for some layers, so keep the machine quiet when measuring an 8B model.
* **`FLLM_VK_SUBMIT_EVERY`** (dispatches per submission, default 32), 3B decode, 3 runs each, tok/s: 0 (one submission per token) 45.0 / 42.0 / 43.6; 16: 50.1 / 50.1 / 53.2;
  32: 48.4 / 51.1 / 46.9; 64: 48.8 / 42.2 / 53.3; 128: 46.1 / 48.6 / 43.7. One submission per step is clearly the worst (as on the NVIDIA card), 16 to 64 are the
  same within the noise; the default stays 32. (These short 32-token runs read higher than the 28 tok/s of the long runs above: the GPU clocks up for a burst.)
* **Tried without gain (kept out of the code):** a 16-byte-per-read shared-memory layout of the tiled prompt kernel (11.3 - 11.7 against 10.3 - 10.7 ms, slightly slower: the kernel is not
  limited by the number of shared-memory instructions); wave32 for all kernels (`RADV_PERFTEST=cswave32`: prompt 175 - 200 against 200 - 225 tok/s); wave32 for the
  cooperative-matrix kernel (17 against 14 ms); work-group size 32 / 64 / 128 / 256 and 1 / 2 / 4 rows per group for the matrix-vector kernel at 8 requests (all inside the noise);
  on HIP, decoding every quantized weight once into a FP16 matrix before the matrix-core GEMM (11.9 ms per Q4_K projection of a 500-token prompt without it, 16.1 ms with it: the GEMM is
  limited by its own structure, not by the weight decoding); Vulkan cooperative matrices with the weights dequantized to FP16 in shared memory: 14 ms against 10 ms for the integer dot
  product; removing the decoding and the activation conversion from that shader still left 8.4 ms, so the f16 cooperative-matrix route cannot beat the integer-dot route here
  (both have the same peak rate on RDNA 3.5; only an int8 matrix-core kernel would, and the per-block scales of the K-quants make that a larger rewrite).
* **Not done:** the prompt gap to llama.cpp (265 against 293 tok/s on the 3B, 98 against 124 on the 8B) and the HIP prompt (about 230 against llama.cpp ROCm's 298). The decode kernels of 9 to 16 rows (the 8-request gap above) and the int8 matrix-core question are in the third pass below.

## Third pass (2026-10-30, evening): small-batch kernels, the matrix units, the memory plan

Same two laptops. New: `FLLM_VK_BENCH_LINEAR=1` times typical projections once at the first projection (random weights, several copies rotated so that no cache holds them, best of three runs of
twelve products) and prints `ms / TMAC/s / GB/s` per number of activation rows. All numbers below come from it unless a table says otherwise; the clock of the 840M varies by run (power limit), so
compare columns of one run and read differences below 10 % as noise.

### Radeon 840M: matrix-vector kernels for 1 to 16 rows

Before this pass the matrix-vector shaders served up to 8 rows and the tiled int8 kernel (`mmq_<fmt>_i`) everything above. The tiled kernel costs the same for 9 as for 64 rows (a Q4_K
8192 x 3072 weight: 1.9 ms for 9 .. 32 rows, 2.0 ms for 64: it is limited by decoding the weights, 0.12 TMAC/s at 9 rows). Eight decoding sequences plus a speculative draft are 9 to 16 rows, so the
default case of eight users sat on the slow side. Now the matrix-vector range is 1 to 16 rows (without tensor cores), and the Q4_K shader got a row-group variant. ms per product:

| rows | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| Q4_K 8192x3072, before | 0.209 | 0.248 | 0.305 | 0.364 | 0.547 | 0.803 | 1.889 | 1.882 | 1.896 |
| Q4_K 8192x3072, after | 0.195 | 0.189 | 0.230 | 0.238 | 0.311 | 0.375 | 0.446 | 0.553 | 0.679 |
| Q4_K 3072x3072, before / after | 0.091 / 0.097 | 0.104 / 0.081 | 0.110 / 0.088 | 0.131 / 0.091 | 0.202 / 0.110 | 0.279 / 0.135 | 0.653 / 0.150 | 0.655 / 0.191 | 0.666 / 0.229 |
| Q6_K 3072x8192, before / after | 0.264 / 0.252 | 0.266 / 0.260 | 0.279 / 0.294 | 0.316 / 0.335 | 0.420 / 0.447 | 0.536 / 0.573 | 3.448 / 0.520 | 3.446 / 0.682 | 3.449 / 0.740 |
| Q4_0 8192x3072, before / after | 0.283 / 0.242 | 0.288 / 0.255 | 0.299 / 0.273 | 0.324 / 0.300 | 0.420 / 0.421 | 0.529 / 0.536 | 1.742 / 0.665 | 1.741 / 0.905 | 1.702 / 1.220 |
| Q8_0 3072x3072, before / after | 0.156 / 0.144 | 0.145 / 0.146 | 0.148 / 0.149 | 0.153 / 0.152 | 0.184 / 0.189 | 0.224 / 0.236 | 0.974 / 0.289 | 1.049 / 0.368 | 1.078 / 0.487 |

The Q4_K product now runs at 0.13 / 0.26 / 0.41 / 0.54 / 0.60 TMAC/s for 1 / 2 / 4 / 8 / 16 rows (before: 0.12 / 0.20 / 0.28 / 0.25 / 0.22, the last through the tiled
kernel); the peak of the integer dot product on this GPU is 2.6 TMAC/s (measured below), the tiled prompt kernel reaches 0.8 - 0.85. What it took, in the order it was found (the ISA from `RADV_DEBUG=shaders` showed
about 330 instructions for 512 multiply-adds): (1) the sum of the quantized activations of a group, which the additive term of Q4_K needs, was computed again by every invocation (8 of the
24 dot products per block and column): the prepass `quant_act_q8` writes `d * sum(q)` per block now; (2) accumulating dot products (`OpSDotAccSat`) instead of dot product + add; (3) the work-group
reduction in shared memory (233 LDS instructions, a third of the shader at K = 3072) disappears when every group of 8 invocations owns its rows (`mv_q4_k_g`); (4) groups start at different blocks:
row strides that are multiples of a power of two put the concurrent reads of a step on one memory channel (the row-group variant of Q6_K, 8192 columns, was 3.4 times slower without it; it did
not help Q6_K and is not used for it). The row-group kernel is slower than the old layout for 1 and 2 rows (fewer loads in flight), so those keep the old one.
Q6_K and the other formats gain only from the larger range (9 to 16 rows 3 to 7 times faster); their 3 .. 8 row speed is unchanged because it is limited by decoding the weights.

Whole model (`scripts/bench_compare.py`, medians of 2 runs, 128 new tokens, total tok/s of all requests including the prompt phase, native Vulkan, Radeon 840M, the machine otherwise idle):

| Llama-3.2-3B Q4_K_M | 1 request, 60 words: decode | 4 requests, 60 words | 8 requests, 60 words | 1 request, 380 words: prompt / decode | 4 requests, 380 words | 8 requests, 380 words |
|---|---|---|---|---|---|---|
| before this pass | 32.4 | 43.7 | 36.0 | 264.5 / 29.6 | 24.5 | 21.8 |
| after | 35.0 | 61.4 | 59.0 | 264.6 / 32.5 | 30.0 | 31.0 |
| llama.cpp Vulkan (earlier today) | | 37.1 | 59.0 | | 30.0 | 34.4 |

Llama-3.1-8B Q4_K_M (all layers on the GPU, 380-word prompts): one request 96.6 -> 100.3 tok/s prompt and 14.4 -> 15.2 decode; 8 requests 9.6 -> 14.6 tok/s in total (the prompt phase dominates: median
time to the first token 43 s for eight 545-token prompts). The prompt rate itself is unchanged by this pass (the same kernels), the 3B prompt gap to llama.cpp (265 against 293) is still open.

### Int8 matrix cores (WMMA) on the 840M: measured, not built

The note of the second pass said that only an int8 matrix-core kernel could beat the integer dot product. RADV exposes `16x16x16 I8*I8+I32` cooperative matrices on this GPU. `FLLM_VK_BENCH_LINEAR=1` now
measures the raw rates on registers (the fragments come from shared memory, nothing else is read), TMAC/s, three identical repetitions:

| unit | 32 x 32 subgroup tile (4 products per 16-deep step) | 64 x 64 subgroup tile (16 products per step) |
|---|---|---|
| float16 cooperative matrices (float32 accumulators) | 1.63 | 2.55 |
| **int8 cooperative matrices (int32 accumulators)** | 2.04 | 2.50 |
| int8 matrices + per quantization group: convert the int32 accumulator, multiply by a row-scale and a column-scale matrix, add (what a K-quant tile must do) | 1.56 | 1.67 |
| int8 dot product (`OpSDotAccSat`, four independent chains) | 2.60 | |

So on this RDNA 3.5 part the int8 matrix unit has **the same peak as float16 (2.5 TMAC/s) and as the dot product (2.6)**; it is not twice as fast, and the per-group scaling of a K-quant costs a third of it. A
kernel built on it cannot beat the tiled dot-product kernel (0.85 TMAC/s) by more than the margin that kernel itself loses to decoding, and the float16 matrix-core experiment of the second pass (14 against 10 ms,
the same peak) showed that this margin is not there. **No int8 matrix-core kernel was written**: the measured rates do not support it on this GPU, and the planner keeps choosing between the cooperative-matrix and the dot-product tile with the
existing timing at the first prompt. The same benchmark prints the shapes of a device (an RTX 2000 Ada offers int8 only as 16 x 16 x 32, which the benchmark shaders do not use; an RDNA 4 or Xe2 device could show a 2x
int8 rate: run the benchmark there before writing anything). HIP uses the same WMMA unit (`v_wmma_i32_16x16x16_iu8`); it was not measured separately, the conclusion follows from the identical hardware. The HIP prompt gap (about 230 against llama.cpp ROCm's
298 tok/s, whose quantized kernels use the dot product, not matrix cores) would need a dot-product tiled kernel like the Vulkan `mmq_<fmt>_i` ported to HIP; that was **not done**.

### RTX 2000 Ada (Vulkan): matrix-vector range, Q4_K loader, Q6_K

Matrix-vector kernels against the cooperative-matrix tile (8192 x 3072 Q4_K, ms): 6 / 8 / 9 / 12 / 16 rows 0.285 / 0.397 / 0.270 / 0.274 / 0.276 before (the tile from 9 rows), 0.134 / 0.159 / 0.160 / 0.214 / 0.235 with the row-group
kernel for 6 to 16 rows. Where the matrix-vector kernels lose they keep the old range: a per-format table gives 16 rows to Q4_K, 12 to Q6_K and Q8_0 and 8 to the others (Q4_0 at 12 / 16 rows: 0.28 / 0.38 ms as
matrix-vector against 0.24 / 0.25 with the tile). Whole model, Llama-3.2-3B, 8 requests of 380 words: 148.4 -> 162.4 tok/s in total (8 requests of 1500 words 83.7 -> 81.7, within noise); Llama-3.1-8B 73.0 / 72.7 and 30.4 / 31.2.

Q4_K loader of the wide tile (`mm_q4_k_cw`; one K step of 64 is one pair of scale groups, an invocation reads 16 bytes of qs with one wide load and the scales, minimums and d / dmin of the pair once, instead of five
loads per chunk of 8 weights): isolated, 64 tokens: 8192 x 3072 5.0 -> 5.7 - 6.3 TMAC/s, 3072 x 3072 3.7 -> 5.5, 32768 x 3072 5.8 -> 6.3. From 128 tokens on the tile runs at 7.7 - 8.6 TMAC/s (128) and 9.9 - 10.8 (512 tokens), about
20 TFLOPS, which is the rate of this card's tensor cores for float16 products with float32 accumulation (GeForce parts run those at half rate), so the loader cannot help there. Whole model: prompt of 380 words 2274 -> 2316 tok/s (3B), 835 -> 853 (8B), 1500 words
2959 -> 2964 and 1127 -> 1156: **+0 to 3 %, inside the run-to-run noise**. A Q4_K_M model is about a third Q6_K (and some Q5_K / Q8_0) by weight, which still use the generic loader. The "roughly 14 TFLOPS limited by the weight fetch"
of the earlier note is 11 - 13 TFLOPS at 64 tokens and about 20 at 512 tokens in isolation now; the distance to llama.cpp's matmul that note mentioned is not in the loader.

Q6_K: the claim that its decode streams at 155 against 230 GB/s for Q4_K did not reproduce in isolation. The vocabulary projection (128256 x 3072, 323 MB) streams at 233 - 235 GB/s with 1 and 2 rows, the Q4_K one at 229 GB/s; the
dip is at 3 to 12 rows (Q6_K 160 -> 59 GB/s, 0.6 - 0.9 TMAC/s against Q4_K's 1.4 - 1.7 TMAC/s), where decoding the 6-bit weights is the limit. A cheaper decode (a sign trick that saves a third of the instructions), a row-group variant and
the larger range were tried: only the range change (9 .. 12 rows) helped; neither changed the 3 .. 8 row speed on the 840M or on the RTX.

### Memory plan under memory pressure

Measured on the 14 GB Radeon 840M laptop (Llama-3.1-8B Q4_K_M, 5.0 GB of weights; another process holds N GB; two 380-word prompts after the start; the 17-word vocabulary of the prompt makes the
n-gram speculation accept a lot, so read the decode column relatively):

| RAM available at start | plan | prompt tok/s | decode tok/s |
|---|---|---|---|
| 13.6 GB | all 32 layers on the GPU, context 10256 | 97 / 106 | 29.9 / 32.8 |
| 7.7 GB | all 32 layers, context 9920 | 96 / 105 | 26.1 / 28.9 |
| 5.8 GB | **17 of 32 layers**, the feed-forward weights of the other 15 stay on the host | 76 / 99 | 12.7 / 9.5 |

The plan is deterministic for a given figure; what moved between runs was the figure (page cache, a browser tab). The planner now rounds the memory it may use down to steps of 256 MiB and logs
`memory for the plan: X MiB free on <device> (device reports Y MiB, clamped to the free system memory), taken as Z MiB`, so a small fluctuation does not flip a plan from "everything on the device" to "15 layers on the CPU"
(2.5 to 3 times slower decode) unless it crosses a step. It does not invent memory: when the weights, a context of at least 4096 tokens and the scratch memory (about 6 GB for the 8B model) do not fit, the plan still moves
feed-forward weights to the host (on this iGPU: to the CPU), and the decode speed drops by that factor; `--ctx N` trades the other way. On the 8 GB RTX 2000 Ada the same 8B model
plans all 32 layers with a 10320-token context from an empty card (budget 7.36 of 8.59 GB). Not measured: an 8 GB card with another process holding VRAM (the card was free), a CUDA run of this plan test.

### Not done in this pass

* An int8 matrix-core kernel (Vulkan or HIP): see above, the measured peak gives nothing to win on the 840M.
* The HIP prompt gap (230 against 298 tok/s) and the Vulkan 3B prompt gap (265 against 293): both need a faster dot-product tiled kernel, not matrix cores. The Vulkan one is at 0.8 - 0.85 of a 2.6 TMAC/s peak.
* Q6_K at 3 to 12 rows (decode-bound); a block-wide activation scale (Q8_K style) would take a quarter of the arithmetic out of the Q4_K / Q6_K kernels but changes the numerics of every projection.
* One laptop per vendor, one session each.
## Limits

* One machine per vendor, one run of each configuration (medians of three requests, but no repeated sessions); expect a few percent of noise.
* The first tables use one prompt length (546 tokens) and one request at a time; prompt lengths and 4 / 8 simultaneous requests are in the 2026-10-30 section (NVIDIA card only, 8 requests for the 3B model only; the AMD laptop has no such matrix here yet).
* Random-word prompts, not a quality test. Texts of fLLM and the CPU were compared separately (`scripts/gpu_test.py`) and agree on most prompts; where
  they differ it is at a near tie of two tokens.
* The 840M is a very small GPU. The NVIDIA card is a laptop part with 8 GB; models above about 7 GB of weights do not fit it.
* llama.cpp's ROCm binary and Ollama needed `HSA_OVERRIDE_GFX_VERSION` for `gfx1153`; without it they fall back to the CPU. FastFlowLM's models are not the same quantization.
