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

## Threads

fLLM does not saturate this host at eight threads (3B Q4_K_M, the batch line in place, total
tokens/s):

| Threads | 1 user | 4 users | first, 1 user |
|---:|---:|---:|---:|
| 4 | 20.0 | 33.7 | 0.71 |
| 8 | 32.9 | 58.0 | 0.42 |
| 16 | 43.7 | 88.5 | 0.27 |
| 32 | 44.6 | 95.3 | 0.27 |

The two tables above are an eight-thread comparison. At sixteen threads for all three engines,
the same `compare.py` run gives (`total_tok_s | first_s | each_tok_s`):

| Model | Prompt | Users | fLLM | llama.cpp | ik_llama.cpp |
|---|---:|---:|---|---|---|
| 3B Q4_K_M | 100 | 1 | 44.0 \| 0.28 \| 44.0 | 48.2 \| 0.22 \| 48.2 | 52.7 \| 0.12 \| 52.7 |
| 3B Q4_K_M | 100 | 4 | 88.8 \| 1.05 \| 22.3 | 113.3 \| 0.90 \| 28.3 | 141.2 \| 0.41 \| 35.3 |
| 3B Q4_K_M | 2000 | 1 | 10.7 \| 4.53 \| 10.7 | 12.3 \| 3.87 \| 12.3 | 18.9 \| 1.96 \| 18.9 |
| 3B Q4_K_M | 2000 | 4 | 12.2 \| 12.81 \| 3.1 | 14.1 \| 9.87 \| 3.5 | 18.5 \| 6.20 \| 4.6 |
| 8B Q4_K_M | 100 | 1 | 20.3 \| 0.63 \| 20.3 | 22.0 \| 0.49 \| 22.1 | 24.7 \| 0.27 \| 24.7 |
| 8B Q4_K_M | 100 | 4 | 40.9 \| 2.32 \| 10.3 | 52.6 \| 2.03 \| 13.2 | 70.2 \| 0.91 \| 17.5 |
| 8B Q4_K_M | 2000 | 1 | 6.0 \| 7.77 \| 6.0 | 5.8 \| 8.30 \| 5.8 | 9.4 \| 4.23 \| 9.4 |
| 8B Q4_K_M | 2000 | 4 | 7.1 \| 21.79 \| 1.8 | 6.8 \| 21.14 \| 1.7 | 9.9 \| 12.57 \| 2.5 |

Ratios to llama.cpp at sixteen threads: single user 0.91x (3B) and 0.92x (8B), four users
0.78x for both models, long prompt with four users 0.87x and 1.04x. Against ik_llama.cpp
fLLM is at 0.57 to 0.83x.

## What is still behind

* **Single-user decode** is 0.81x of llama.cpp at eight threads and 0.91x at sixteen (0.77x
  and 0.92x on the 8B), against ik_llama.cpp 0.67x at eight and 0.83x at sixteen. Not
  explained.
* **Four users** is 0.78x of llama.cpp at sixteen threads (0.84x at eight); the remaining
  difference is visible in the time to the first token as well (1.05 against 0.90 s on the
  3B), so it is not only the decode kernels.
* **ik_llama.cpp** is ahead at every batch size here (0.66 to 0.84x of it); its quantization
  kernels use different integer paths.
* **Not measured**: other AVX-512 CPUs (Intel, Zen 4), other formats than the two Q4_K_M
  models, 16 and more users for the 8B model, and the AVX-512 streaming kernels at batches
  between 16 and 32, where the new line decides by batch size alone.
* The Colab CPU runtime pool served only Zen 2 (EPYC 7B12) and Broadwell; the Zen 5 host is
  the one behind a G4 GPU runtime, measured with `--device cpu`.
