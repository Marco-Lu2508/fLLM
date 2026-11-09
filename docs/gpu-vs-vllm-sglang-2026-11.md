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

Not measured: FP8, GPTQ, MoE models, a longer output, more than 16 users, other cards, ROCm, more than one GPU, quality (only the first answer was compared).
Raw results: `compare_result.json` of both runs (`schema_version 2`, `vs_best_rival` per engine).
