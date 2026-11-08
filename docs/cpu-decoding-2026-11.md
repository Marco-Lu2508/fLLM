# CPU decoding on Zen 5 (2026-11-08)

The earlier [multi-card comparison](colab-comparison-2026-11.md) counted SSE messages and
ran each cell once. The numbers here use `scripts/colab/compare.py` with `schema_version 2`:
server-reported completion tokens, one seed for every engine (`--seed 11`), prefix/output
caches and speculation off for fLLM, 64 generated tokens, and the same prompts for all three
engines. Each engine serves the model from its own process on the same machine.

Machine: Colab G4 runtime, host CPU **AMD EPYC 9B45** (Zen 5, family 26, 48 vCPUs), 8 threads
per engine, 176 GB RAM. Models: Llama-3.2-3B-Instruct Q4_K_M and Llama-3.1-8B-Instruct
Q4_K_M. Values are total tokens per second over all users; `first` is the mean time to the
first output token.

## The gap this found

fLLM's decode of several users was far behind. The 16-row AVX-512-VNNI prompt GEMM
(`cpu_avx512::vnni_gemm`) took over from four tokens on, which is right for a prefill chunk
and wrong for a small decode batch: on Zen 5 the 256-bit kernels beat it there.

| 3B Q4_K_M, 4 users, prompt 100 | fLLM | llama.cpp | ik_llama.cpp |
|---|---:|---:|---:|
| before (`FLLM_VNNI_GEMM_MIN_BATCH` = 4) | 25.1 | 70.2 | 86.7 |
| after (32) | **58.5** | 69.5 | 83.4 |

| 8B Q4_K_M, 4 users, prompt 100 | fLLM | llama.cpp | ik_llama.cpp |
|---|---:|---:|---:|
| before | 11.0 | 32.1 | 38.8 |
| after | **25.9** | 31.9 | 39.0 |

Full table after the change (`total_tok_s | first_s | each_tok_s`):

| Model | Prompt | Users | fLLM | llama.cpp | ik_llama.cpp |
|---|---:|---:|---|---|---|
| 3B Q4_K_M | 100 | 1 | 33.0 \| 0.42 \| 33.0 | 40.9 \| 0.40 \| 41.0 | 49.0 \| 0.22 \| 49.0 |
| 3B Q4_K_M | 100 | 4 | 58.5 \| 1.53 \| 14.7 | 69.5 \| 1.65 \| 17.4 | 83.4 \| 0.68 \| 20.9 |
| 3B Q4_K_M | 2000 | 1 | 7.1 \| 6.94 \| 7.1 | 7.3 \| 7.24 \| 7.3 | 11.8 \| 3.89 \| 11.8 |
| 3B Q4_K_M | 2000 | 4 | 7.8 \| 19.82 \| 2.0 | 7.8 \| 18.31 \| 2.0 | 9.3 \| 13.86 \| 2.3 |
| 8B Q4_K_M | 100 | 1 | 14.7 \| 0.95 \| 14.7 | 19.2 \| 0.91 \| 19.2 | 21.7 \| 0.51 \| 21.7 |
| 8B Q4_K_M | 100 | 4 | 25.9 \| 3.39 \| 6.5 | 31.9 \| 3.89 \| 8.0 | 39.0 \| 1.74 \| 9.8 |
| 8B Q4_K_M | 2000 | 1 | 3.8 \| 12.54 \| 3.8 | 3.4 \| 15.87 \| 3.4 | 5.9 \| 8.15 \| 5.9 |
| 8B Q4_K_M | 2000 | 4 | 4.3 \| 35.48 \| 1.1 | 3.6 \| 40.16 \| 0.9 | 5.2 \| 24.23 \| 1.3 |

## How the batch line was chosen

One binary, `FLLM_VNNI_GEMM_MIN_BATCH` set per server start, 3B Q4_K_M, 8 threads, total
tokens per second at 1, 4, 8 and 16 users and at a 2000-token prompt with 4 users:

| Line | 1 user | 4 users | 8 users | 16 users | 2000/4 users | first, 2000/4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 (old) | 33.1 | 25.1 | 41.2 | — | 7.2 | 18.3 |
| 8 | 33.3 | 59.4 | 43.8 | 64.0 | 8.4 | 18.0 |
| 16 | 33.4 | 59.6 | 70.9 | 65.7 | 8.4 | 18.1 |
| 32 (now) | 33.4 | 59.5 | 71.1 | 80.0 | 8.4 | 18.2 |
| never | 28.2 | 43.0 | 47.6 | 51.0 | — | — |

With the GEMM off everywhere the prompt loses it (time to the first token 0.40 against 0.75 s
at one user), so a line rather than a switch: decode batches under 32 take the 256-bit
kernels, prefill chunks keep the GEMM. Single-user decode is unchanged (33.1 in the sweep and
33.0 in both comparison runs, before and after), and a 2000-token prompt got slightly faster
(7.2 -> 8.4 at four users, same time to the first token).

## What is still behind

* **Single-user decode** is 0.81x of llama.cpp and 0.67x of ik_llama.cpp on the 3B model
  (0.77x / 0.68x on the 8B). Not explained.
* **ik_llama.cpp** is ahead at every batch size here (0.66 to 0.84x of it); its quantization
  kernels use different integer paths.
* **Not measured**: other AVX-512 CPUs (Intel, Zen 4), other formats than the two Q4_K_M
  models, 16 and more users for the 8B model, and the AVX-512 streaming kernels at batches
  between 16 and 32, where the new line decides by batch size alone.
* The Colab CPU runtime pool served only Zen 2 (EPYC 7B12) and Broadwell; the Zen 5 host is
  the one behind a G4 GPU runtime, measured with `--device cpu`.
