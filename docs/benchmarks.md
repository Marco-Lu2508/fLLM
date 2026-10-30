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
  checkpoint, ahead of vLLM and SGLang too. On Llama-3.1-8B it is **10 % behind Ollama at 600 tokens (1690 against 1887) and 3 % at 2100 tokens** and
  ahead of llama.cpp. An int8 tensor-core GEMM for Q4_K that exists in the tree (`cuda/qmma4k.cu`, never built or called) was wired in for a test: after one
  fix (it ignored the fused gate/up activation) it wrote the right text but was slower than the FP16 kernel on this card (1736 against 2017 tok/s on the
  8B prompt). It stays unused; the gap is the Q4_K prompt GEMM.
* **CUDA, simultaneous requests.** The aggregate rises with 4 and 8 requests (3B, short prompts: 97 -> 270 -> 400 tok/s; Ollama 94 -> 83 -> 128) and is the
  best of the GGUF engines everywhere except **Llama-3.1-8B with four 2100-token prompts: 53 against 60 (Ollama) and 54 tok/s (llama.cpp)**; the median
  TTFT there (4.1 s) is also worse than Ollama's (2.8 s). On the AWQ checkpoint fLLM equals vLLM and SGLang with 4 requests, is 10 % behind at 8
  requests with a medium prompt (369 against 413 / 431) and 17 % behind with short prompts (523 against 632 / 646). Profiled: the weight kernels there already
  stream at the card's bandwidth; what remains is the decode attention kernel (about 100 us per layer for 8 sequences) and the step overhead.
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
* **Still open:** the CUDA Q4_K prompt GEMM (8B, 10 % behind Ollama) and with it the aggregate of several long 8B prompts; the Vulkan matrix product reaches
  roughly 14 TFLOPS (llama.cpp's about 20): an experiment without the weight fetch was 1.8 times faster, so the packed-weight fetch and decode in front of the
  tensor cores bound it, not the tensor cores; Vulkan decode of `Q6_K` (the output projection) streams at about 155 GB/s against 230 GB/s for the Q4_K kernels.

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
* **Not done:** the prompt gap to llama.cpp (265 against 293 tok/s on the 3B, 98 against 124 on the 8B) and the HIP prompt (about 230 against llama.cpp ROCm's 298).

## Limits

* One machine per vendor, one run of each configuration (medians of three requests, but no repeated sessions); expect a few percent of noise.
* The first tables use one prompt length (546 tokens) and one request at a time; prompt lengths and 4 / 8 simultaneous requests are in the 2026-10-30 section (NVIDIA card only, 8 requests for the 3B model only; the AMD laptop has no such matrix here yet).
* Random-word prompts, not a quality test. Texts of fLLM and the CPU were compared separately (`scripts/gpu_test.py`) and agree on most prompts; where
  they differ it is at a near tie of two tokens.
* The 840M is a very small GPU. The NVIDIA card is a laptop part with 8 GB; models above about 7 GB of weights do not fit it.
* llama.cpp's ROCm binary and Ollama needed `HSA_OVERRIDE_GFX_VERSION` for `gfx1153`; without it they fall back to the CPU. FastFlowLM's models are not the same quantization.
