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

## AMD Radeon 840M laptop

Llama-3.2-3B / Llama-3.1-8B Q4_K_M. Prompt tok/s / decode tok/s. The 840M has 4 compute units and takes about 5 GB of the 14 GB for a model,
so the 8B does not fit and every engine splits it between GPU and CPU.

| Engine, backend | 3B | 8B |
|---|---|---|
| FastFlowLM, **NPU** (own model files) | **306** / **31.2** | **207** / **14.0** |
| llama.cpp, ROCm (`HSA_OVERRIDE_GFX_VERSION=11.5.1`) | 300 / 20.1 | not run |
| llama.cpp, Vulkan | 287 / 28.6 | 124 / 13.2 |
| Ollama, ROCm (same override) | 224 / 15.5 | not run |
| **fLLM, Vulkan (native)** | 213 / 28.4 | 75 / 8.5 |
| fLLM, HIP (ROCm) | 157 / 22.0 | 67 / 9.8 |
| fLLM, CPU | 146 / 16.7 | 65 / 7.8 |
| Ollama (CPU; it does not take `gfx1153`) | 122 / 19.0 | 51 / 8.7 |
| llama.cpp, CPU | 112 / 19.0 | 47 / 8.7 |

Reading: on this small iGPU the NPU is fastest, llama.cpp's Vulkan and ROCm paths come next, fLLM's native Vulkan is on par for decode of the 3B (the 8B,
which is split between GPU and CPU, is 36 % behind) and 25 to 40 % behind for the prompt. All of these beat the laptop's own CPU for the prompt; for
decode the CPU reaches 55 to 95 % of the GPU paths. The numbers say nothing about a
Ryzen AI Max (Strix Halo: 8 to 10 times the compute units and twice the memory bandwidth); it was not available.

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

## Limits

* One machine per vendor, one run of each configuration (medians of three requests, but no repeated sessions); expect a few percent of noise.
* One prompt length (546 tokens), one request at a time: nothing here says anything about batching or many users, where vLLM and SGLang are built to win.
* Random-word prompts, not a quality test. Texts of fLLM and the CPU were compared separately (`scripts/gpu_test.py`) and agree on most prompts; where
  they differ it is at a near tie of two tokens.
* The 840M is a very small GPU. The NVIDIA card is a laptop part with 8 GB; models above about 7 GB of weights do not fit it.
* llama.cpp's ROCm binary and Ollama needed `HSA_OVERRIDE_GFX_VERSION` for `gfx1153`; without it they fall back to the CPU. FastFlowLM's models are not the same quantization.
