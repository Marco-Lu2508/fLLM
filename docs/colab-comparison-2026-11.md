# fLLM against llama.cpp and ik_llama.cpp on Colab cards and two CPUs (2026-11-08)

A first comparison on hardware that was at hand: an L4, an A100 (40 GB), an RTX PRO 6000 Blackwell (96 GB, Colab "G4") and a T4 on Google Colab Pro, and the
processors of two Colab machines. It is a measurement of one afternoon on shared cloud hardware, one run per cell, no medians. It is **not** the figure to quote against vLLM
or SGLang on the company server (an AMD MI300X / MI355X machine, 8 GPUs); it shows where the CUDA and CPU paths stand and which gaps are real.

## What was measured

- fLLM: commit `2591903` (CUDA kernels for sm_75, 80, 89, 90 and 120, built with CUTLASS and FlashInfer, 19 min 29 s on 12 cores), then the three fixes below on top of it.
- llama.cpp (`f477f79` on the T4; the other cards built the head of the same day for their own architecture) and ik_llama.cpp `cc61694`, both with CUDA and `-ngl 99`, the CPU runs
  with `-ngl 0`. Server defaults, no hand tuning of any engine.
- `scripts/colab/compare.py`: every engine serves the same GGUF file through `/v1/chat/completions` with streaming. Prompts of 100 and 2000 tokens (500 on the CPUs), 1 and 16
  simultaneous users (4 on the CPUs), 64 generated tokens (48 on the CPUs), every prompt different (no cache reuse). Reported: the speed one user sees after its first token
  (**decode**), the time to the first token of a long prompt with one user (**prefill**), and the total tokens per second of all users together.
- Models: Llama-3.1-8B-Instruct in Q2_K, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K and Q8_0, Qwen3-8B Q4_K_M and Q8_0, Llama-3.2-3B Q4_K_M and Q8_0, Gemma 4 E4B Q4_K_M,
  Qwen3-30B-A3B Q4_K_M (mixture of experts). Each engine also answers one fixed request, so a fast engine that produces nonsense would show; the answers agree up to near ties
  ("I'll" against "I will").

| card / processor | CPU of the machine | RAM |
|---|---|---|
| L4 (23 GB) | Intel Xeon 2.2 GHz, 6 cores / 12 threads, AVX-512 + VNNI | 52 GB |
| A100 SXM4 (40 GB) | Intel Xeon 2.2 GHz, 12 vCPU | 83 GB |
| RTX PRO 6000 Blackwell (96 GB) | AMD EPYC 9B45, 24 cores / 48 threads | about 180 GB |
| T4 (15 GB) | Intel Xeon 2.0 GHz, 2 vCPU | 12 GB |
| CPU run 1 | AMD EPYC 7B12 (Zen 2, AVX2), 4 cores / 8 threads | 50 GB |
| CPU run 2 | AMD EPYC 9B45 (Zen 5), 24 threads used | about 180 GB |

## Summary: the mainstream quants (geometric mean of fLLM against the engine)

Over Llama-3.1-8B Q4_K_M, Q5_K_M, Q6_K and Q8_0, Qwen3-8B Q4_K_M and Q8_0 and Llama-3.2-3B Q4_K_M (Q2_K, IQ4_XS, Gemma 4 and the mixture of experts are listed
separately below). Above 1.00 fLLM is ahead; for prefill it is the engine's time divided by fLLM's. `many users` is the total tokens per second at 16 users (4 on the CPUs).

| card | against | decode | prefill | many users, prompt 100 | many users, long prompt |
|---|---|---:|---:|---:|---:|
| A100 | llama.cpp | 1.18 | 1.97 | 1.75 | 1.51 |
| A100 | ik_llama.cpp | 1.11 | 1.60 | 1.87 | 2.32 |
| RTX PRO 6000 | llama.cpp | 0.96 | 1.19 | 1.13 | 1.07 |
| RTX PRO 6000 | ik_llama.cpp | 1.01 | 1.27 | 1.40 | 1.74 |
| L4 | llama.cpp | 0.98 | 1.16 | 1.20 | 0.98 |
| L4 | ik_llama.cpp | 0.99 | 1.09 | 1.21 | 1.80 |
| T4 (2 vCPU) | llama.cpp | 1.34 | 1.11 | 0.96 | 0.91 |
| T4 (2 vCPU) | ik_llama.cpp | 1.45 | 1.06 | 0.98 | 1.61 |
| CPU, Zen 5 (24 threads) | llama.cpp | 1.25 | 0.93 | 0.65 | 0.48 |
| CPU, Zen 5 (24 threads) | ik_llama.cpp | 1.19 | 0.44 | 0.48 | 0.30 |
| CPU, Zen 2 (8 threads) | llama.cpp | 0.71 | 1.12 | 0.91 | 0.77 |
| CPU, Zen 2 (8 threads) | ik_llama.cpp | 0.84 | 0.95 | 0.69 | 0.66 |

What it says: on the **A100** fLLM leads on all four measures, most in prefill (2x) and with many users. On the **RTX PRO 6000** and the **L4** decode is on par
(the cards are memory-bound there), prefill is 10 to 25 % faster and many short requests 13 to 40 % more. With many users **and long prompts** fLLM equals llama.cpp and is
1.7 to 1.8x ahead of ik_llama.cpp. The T4 has only 2 vCPUs, which is why its numbers carry more noise.

## Where fLLM is behind (real gaps, not noise)

1. **Gemma 4 E4B** (all cards): decode 0.46 to 0.87x of llama.cpp, many users 0.06 to 0.35x. The first cause, a step that failed at 16 users, is fixed (below); with the fix
   16 short requests reach 0.35x (L4), 0.28x (A100) and 0.25x (RTX PRO 6000) of llama.cpp, decode did not change. The rest is not explained. CUDA graphs are off for this
   model, and decode does not scale with the batch.
2. **Q2_K** (Llama-3.1-8B): decode 0.84 to 0.90x, many users 0.17 to 0.47x on every card, and on the T4 a 2000 token prompt takes 15 s against 3 s. The generic decoder of
   `iq_decode.cuh` runs these formats; there is no integer-dot-product GEMM for them.
3. **IQ4_XS on the T4**: prefill 16 s against 2 s (decode is fine); on the other cards IQ4_XS is 0.86 to 1.01x in decode.
4. **CPU, many users**: on the Zen 5 machine four simultaneous requests give fLLM 63 tok/s on Llama-3.2-3B against 116 (llama.cpp) and 161 (ik_llama.cpp); prefill of ik_llama.cpp
   is more than twice as fast. Decode with one user is on par.
5. **CPU decode on Zen 2** (AVX2, no VNNI): 0.58 to 0.81x of llama.cpp with 8 threads, 0.69 to 0.85x with 4. Prefill is 1.1x ahead there.
6. **Q8_0 on the Zen 5 CPU is the other way round**: fLLM 42.8 tok/s against 17.3 and 17.6.

## Fixed during this run

All three were found by these measurements and verified on the cards named; none is a unit test of the GPU path.

- **The Marlin Q4_K GEMM declined a shape and the step failed** (`marlin failed to launch`, Gemma 4 E4B at 16 users). The launcher now runs the plain Q4_K kernels on the first
  copy of the dual layout, and the activation scratch has the room a tall projection needs. 16 users, prompt 100, total tokens/s: L4 25.6 to 111.7, A100 28.5 to 116.4, RTX PRO 6000 63.8 to 270.5.
- **The memory plan counted one matrix per stacked expert tensor** for the Marlin copy, so Qwen3-30B-A3B planned a 62 GB cache next to 19.5 GB of weights and ran out of memory
  while uploading (A100 and RTX PRO 6000). It now starts: A100 162 tok/s decode, 2000 tokens in 0.37 s, 16 users 360 tok/s; RTX PRO 6000 257 tok/s decode, 688 tok/s with 16 users.
  Against llama.cpp on the A100 (149 / 0.56 s / 467) that is 1.08x decode and 1.5x prefill, 0.77x with 16 short requests.
  Test: `the_marlin_copy_of_an_expert_tensor_is_counted_for_every_expert` (the mutation that counts one matrix fails with 983040 against 125829120).
- **CUDA refused a copy from host memory that spans two pinned pieces** (`upload: invalid argument`, every prompt step of a model whose feed-forward weights stay in host memory:
  Llama-3.1-8B Q8_0 on the T4 and the L4, Qwen3-30B-A3B on the L4). Such copies are now done piece by piece on every backend. Q8_0 on the L4 returns the same text as with
  `FLLM_NO_HOST_PIN=1`; Qwen3-30B-A3B on the L4 (12 of 48 feed-forward blocks on the CPU) runs at 36 tok/s.

## Not measured

vLLM, SGLang and Ollama (the script supports them, they were left out to keep the run short); AMD GPUs and ROCm; more than one GPU; sm_90 (Colab refused an H100 for this account);
the prompt cache and the output cache (every prompt was different on purpose); quality beyond the fixed request. The T4 and CPU numbers come from machines with 2 and 4 physical cores.

## Per card

The tables are the output of `compare.py` for each machine (decode = tokens/s one user sees after its first token, prompt 100; prefill = seconds to the first token of the long
prompt; `same answer` compares the first words of the fixed request with llama.cpp's). The raw result files and server logs are not part of the repository.

### A100 (40 GB)

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 2000 token prompt, 1 user (less is better). 16 users = total tokens/s of 16 simultaneous requests (prompt 100 / prompt 2000). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 2000 (s) | 16 users, p100 | 16 users, p2000 | decode vs fLLM | prefill vs fLLM | 16u p100 vs fLLM | 16u p2000 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 164.5 | 0.33 | 794.3 | 166.4 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 138.2 | 0.64 | 499.7 | 114.1 | 1.19x | 1.94x | 1.59x | 1.46x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 150.0 | 0.53 | 475.1 | 74.5 | 1.10x | 1.61x | 1.67x | 2.23x | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | fllm | 124.5 | 0.37 | 864.2 | 174.3 |  |  |  |  | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | llama.cpp | 111.2 | 0.63 | 480.0 | 114.1 | 1.12x | 1.70x | 1.80x | 1.53x |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 120.7 | 0.51 | 455.1 | 75.4 | 1.03x | 1.38x | 1.90x | 2.31x | no |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | fllm | 148.2 | 0.57 | 184.4 | 70.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | llama.cpp | 165.3 | 0.89 | 409.3 | 77.6 | 0.90x | 1.56x | 0.45x | 0.91x |  |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | ik_llama.cpp | 138.7 | 0.74 | 406.6 | 59.8 | 1.07x | 1.30x | 0.45x | 1.18x | no |
| Qwen3-8B Q4_K_M (GGUF) | fllm | 153.0 | 0.31 | 728.6 | 157.2 |  |  |  |  | yes |
| Qwen3-8B Q4_K_M (GGUF) | llama.cpp | 130.2 | 0.68 | 468.2 | 106.2 | 1.18x | 2.19x | 1.56x | 1.48x |  |
| Qwen3-8B Q4_K_M (GGUF) | ik_llama.cpp | 138.2 | 0.56 | 429.8 | 69.9 | 1.11x | 1.81x | 1.70x | 2.25x | no |
| Qwen3-8B Q8_0 (GGUF) | fllm | 117.6 | 0.29 | 857.5 | 169.3 |  |  |  |  | yes |
| Qwen3-8B Q8_0 (GGUF) | llama.cpp | 105.8 | 0.69 | 442.2 | 103.0 | 1.11x | 2.38x | 1.94x | 1.64x |  |
| Qwen3-8B Q8_0 (GGUF) | ik_llama.cpp | 113.9 | 0.54 | 409.6 | 70.4 | 1.03x | 1.86x | 2.09x | 2.40x | no |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 295.4 | 0.17 | 1310.7 | 280.3 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 214.0 | 0.39 | 700.2 | 207.5 | 1.38x | 2.29x | 1.87x | 1.35x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 247.3 | 0.30 | 660.8 | 120.6 | 1.19x | 1.76x | 1.98x | 2.32x | no |
| Gemma 4 E4B Q4_K_M (GGUF) | fllm | 61.8 | 0.55 | 28.5 | 14.0 |  |  |  |  | yes |
| Gemma 4 E4B Q4_K_M (GGUF) | llama.cpp | 121.0 | 0.44 | 417.7 | 127.8 | 0.51x | 0.80x | 0.07x | 0.11x |  |
| Gemma 4 E4B Q4_K_M (GGUF) | ik_llama.cpp | - | - | - | - | - | - | - | - | no |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | fllm | 145.4 | 0.35 | 860.9 | 173.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | llama.cpp | 121.7 | 0.69 | 463.2 | 104.8 | 1.19x | 1.97x | 1.86x | 1.66x |  |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | ik_llama.cpp | 114.9 | 0.58 | 433.0 | 69.8 | 1.27x | 1.66x | 1.99x | 2.49x | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | fllm | 162.8 | 0.30 | 851.0 | 164.9 |  |  |  |  | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | llama.cpp | 161.9 | 0.62 | 517.8 | 118.7 | 1.01x | 2.07x | 1.64x | 1.39x |  |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | ik_llama.cpp | 165.0 | 0.52 | 487.0 | 74.6 | 0.99x | 1.73x | 1.75x | 2.21x | no |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | fllm | 146.4 | 0.44 | 816.4 | 164.2 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | llama.cpp | 129.4 | 0.66 | 482.6 | 110.8 | 1.13x | 1.50x | 1.69x | 1.48x |  |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | ik_llama.cpp | 136.3 | 0.54 | 455.9 | 72.6 | 1.07x | 1.23x | 1.79x | 2.26x | no |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | fllm | error: did not start: ts on the device: copying blk.27.exp39.down, 9.55 GiB c |  |  |  |  |  |  |  |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | llama.cpp | 149.4 | 0.56 | 466.9 | 115.6 |  |  |  |  |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | ik_llama.cpp | 145.7 | 0.69 | 310.9 | 57.8 |  |  |  |  | no |

### RTX PRO 6000 Blackwell (96 GB)

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 2000 token prompt, 1 user (less is better). 16 users = total tokens/s of 16 simultaneous requests (prompt 100 / prompt 2000). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 2000 (s) | 16 users, p100 | 16 users, p2000 | decode vs fLLM | prefill vs fLLM | 16u p100 vs fLLM | 16u p2000 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 223.9 | 0.18 | 1280.8 | 294.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 231.9 | 0.20 | 1314.5 | 311.7 | 0.97x | 1.11x | 0.97x | 0.95x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 222.7 | 0.22 | 1060.5 | 191.6 | 1.01x | 1.22x | 1.21x | 1.54x | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | fllm | 154.8 | 0.19 | 1385.3 | 328.6 |  |  |  |  | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | llama.cpp | 157.1 | 0.19 | 1137.8 | 310.7 | 0.99x | 1.00x | 1.22x | 1.06x |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 151.2 | 0.20 | 934.1 | 194.4 | 1.02x | 1.05x | 1.48x | 1.69x | no |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | fllm | 263.4 | 0.24 | 294.8 | 145.1 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | llama.cpp | 303.1 | 0.26 | 1267.0 | 250.8 | 0.87x | 1.08x | 0.23x | 0.58x |  |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | ik_llama.cpp | 281.7 | 0.25 | 1039.1 | 172.0 | 0.94x | 1.04x | 0.28x | 0.84x | no |
| Qwen3-8B Q4_K_M (GGUF) | fllm | 211.0 | 0.17 | 1164.5 | 284.7 |  |  |  |  | yes |
| Qwen3-8B Q4_K_M (GGUF) | llama.cpp | 219.9 | 0.21 | 1224.1 | 288.9 | 0.96x | 1.24x | 0.95x | 0.99x |  |
| Qwen3-8B Q4_K_M (GGUF) | ik_llama.cpp | 207.3 | 0.23 | 960.3 | 177.0 | 1.02x | 1.35x | 1.21x | 1.61x | no |
| Qwen3-8B Q8_0 (GGUF) | fllm | 147.6 | 0.15 | 1247.7 | 310.8 |  |  |  |  | yes |
| Qwen3-8B Q8_0 (GGUF) | llama.cpp | 151.6 | 0.21 | 1055.4 | 284.3 | 0.97x | 1.40x | 1.18x | 1.09x |  |
| Qwen3-8B Q8_0 (GGUF) | ik_llama.cpp | 144.3 | 0.21 | 843.1 | 180.2 | 1.02x | 1.40x | 1.48x | 1.72x | no |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 417.4 | 0.09 | 2063.0 | 520.2 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 409.5 | 0.13 | 1877.4 | 505.7 | 1.02x | 1.44x | 1.10x | 1.03x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 398.7 | 0.15 | 1480.6 | 265.1 | 1.05x | 1.67x | 1.39x | 1.96x | no |
| Gemma 4 E4B Q4_K_M (GGUF) | fllm | 103.0 | 0.25 | 63.8 | 31.2 |  |  |  |  | yes |
| Gemma 4 E4B Q4_K_M (GGUF) | llama.cpp | 224.6 | 0.17 | 1100.9 | 309.4 | 0.46x | 0.68x | 0.06x | 0.10x |  |
| Gemma 4 E4B Q4_K_M (GGUF) | ik_llama.cpp | - | - | - | - | - | - | - | - | no |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | fllm | 171.8 | 0.17 | 1504.7 | 351.7 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | llama.cpp | 190.3 | 0.23 | 1129.2 | 271.0 | 0.90x | 1.35x | 1.33x | 1.30x |  |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | ik_llama.cpp | 179.2 | 0.24 | 941.9 | 176.8 | 0.96x | 1.41x | 1.60x | 1.99x | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | fllm | 222.2 | 0.14 | 1709.9 | 362.2 |  |  |  |  | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | llama.cpp | 258.1 | 0.19 | 1404.2 | 343.4 | 0.86x | 1.36x | 1.22x | 1.05x |  |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | ik_llama.cpp | 243.2 | 0.20 | 1117.5 | 203.2 | 0.91x | 1.43x | 1.53x | 1.78x | no |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | fllm | 199.1 | 0.22 | 1496.0 | 331.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | llama.cpp | 209.5 | 0.20 | 1262.8 | 305.5 | 0.95x | 0.91x | 1.18x | 1.09x |  |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | ik_llama.cpp | 200.4 | 0.21 | 1030.5 | 190.4 | 0.99x | 0.95x | 1.45x | 1.74x | no |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | fllm | error: did not start: tchpad: 1184.13 MB | [fllm-rs] uploading the weights of |  |  |  |  |  |  |  |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | llama.cpp | 297.8 | 0.19 | 1165.6 | 278.0 |  |  |  |  |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | ik_llama.cpp | 246.5 | 0.34 | 655.5 | 125.1 |  |  |  |  | no |

### L4 (23 GB)

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 2000 token prompt, 1 user (less is better). 16 users = total tokens/s of 16 simultaneous requests (prompt 100 / prompt 2000). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 2000 (s) | 16 users, p100 | 16 users, p2000 | decode vs fLLM | prefill vs fLLM | 16u p100 vs fLLM | 16u p2000 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 48.7 | 0.80 | 383.3 | 64.1 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 49.3 | 0.91 | 311.7 | 66.0 | 0.99x | 1.14x | 1.23x | 0.97x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 48.4 | 0.86 | 308.6 | 36.9 | 1.01x | 1.07x | 1.24x | 1.74x | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | fllm | 30.5 | 0.87 | 258.9 | 57.1 |  |  |  |  | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | llama.cpp | 29.8 | 0.90 | 243.8 | 61.0 | 1.02x | 1.03x | 1.06x | 0.94x |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 29.9 | 0.89 | 239.2 | 33.2 | 1.02x | 1.02x | 1.08x | 1.72x | no |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | fllm | 59.4 | 1.29 | 149.7 | 37.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | llama.cpp | 70.4 | 1.11 | 318.3 | 55.8 | 0.84x | 0.86x | 0.47x | 0.67x |  |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | ik_llama.cpp | 59.5 | 1.02 | 320.4 | 33.9 | 1.00x | 0.79x | 0.47x | 1.11x | no |
| Qwen3-8B Q4_K_M (GGUF) | fllm | 46.8 | 0.84 | 355.6 | 60.1 |  |  |  |  | yes |
| Qwen3-8B Q4_K_M (GGUF) | llama.cpp | 47.5 | 0.99 | 285.1 | 61.0 | 0.99x | 1.18x | 1.25x | 0.99x |  |
| Qwen3-8B Q4_K_M (GGUF) | ik_llama.cpp | 46.6 | 0.91 | 283.6 | 33.9 | 1.00x | 1.08x | 1.25x | 1.77x | no |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 103.4 | 0.42 | 786.5 | 122.7 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 100.8 | 0.51 | 509.4 | 123.5 | 1.03x | 1.21x | 1.54x | 0.99x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 99.8 | 0.45 | 502.5 | 61.1 | 1.04x | 1.07x | 1.57x | 2.01x | no |
| Gemma 4 E4B Q4_K_M (GGUF) | fllm | 44.2 | 1.01 | 25.6 | 12.7 |  |  |  |  | yes |
| Gemma 4 E4B Q4_K_M (GGUF) | llama.cpp | 63.8 | 0.62 | 318.6 | 85.6 | 0.69x | 0.61x | 0.08x | 0.15x |  |
| Gemma 4 E4B Q4_K_M (GGUF) | ik_llama.cpp | - | - | - | - | - | - | - | - | no |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | fllm | 33.2 | 0.80 | 297.0 | 61.0 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | llama.cpp | 37.8 | 1.05 | 257.3 | 56.3 | 0.88x | 1.31x | 1.15x | 1.08x |  |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | ik_llama.cpp | 36.4 | 1.05 | 253.0 | 31.5 | 0.91x | 1.31x | 1.17x | 1.94x | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | fllm | 48.2 | 0.79 | 372.1 | 63.1 |  |  |  |  | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | llama.cpp | 54.6 | 0.87 | 329.3 | 70.9 | 0.88x | 1.10x | 1.13x | 0.89x |  |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | ik_llama.cpp | 54.2 | 0.81 | 325.0 | 36.9 | 0.89x | 1.03x | 1.14x | 1.71x | no |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | fllm | 41.8 | 0.90 | 289.5 | 56.7 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | llama.cpp | 42.9 | 1.00 | 283.1 | 60.3 | 0.97x | 1.11x | 1.02x | 0.94x |  |
| Llama-3.1-8B-Instruct Q5_K_M (GGUF) | ik_llama.cpp | 42.3 | 0.92 | 282.7 | 34.3 | 0.99x | 1.02x | 1.02x | 1.65x | no |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | fllm | 0.0 | - | - | - |  |  |  |  |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | llama.cpp | 92.4 | 0.81 | 337.3 | 66.9 | - | - | - | - |  |
| Qwen3-30B-A3B Q4_K_M (GGUF, MoE) | ik_llama.cpp | 87.3 | 0.91 | 251.4 | 32.2 | - | - | - | - | no |

### T4 (15 GB)

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 2000 token prompt, 1 user (less is better). 16 users = total tokens/s of 16 simultaneous requests (prompt 100 / prompt 2000). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 2000 (s) | 16 users, p100 | 16 users, p2000 | decode vs fLLM | prefill vs fLLM | 16u p100 vs fLLM | 16u p2000 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 47.4 | 2.00 | 171.7 | 26.5 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 39.6 | 2.04 | 193.1 | 29.8 | 1.20x | 1.02x | 0.89x | 0.89x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 36.3 | 2.13 | 184.0 | 17.2 | 1.31x | 1.06x | 0.93x | 1.54x | no |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | fllm | 37.0 | 14.98 | 23.9 | 3.8 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | llama.cpp | 33.8 | 3.13 | 141.2 | 19.7 | 1.09x | 0.21x | 0.17x | 0.19x |  |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | ik_llama.cpp | 29.4 | 3.12 | 150.1 | 13.5 | 1.26x | 0.21x | 0.16x | 0.28x | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | fllm | 45.5 | 15.99 | 49.7 | 3.9 |  |  |  |  | no |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | llama.cpp | 35.1 | 2.12 | 183.1 | 28.4 | 1.30x | 0.13x | 0.27x | 0.14x |  |
| Llama-3.1-8B-Instruct IQ4_XS (GGUF) | ik_llama.cpp | 37.6 | 2.02 | 190.4 | 17.6 | 1.21x | 0.13x | 0.26x | 0.22x | no |
| Qwen3-8B Q4_K_M (GGUF) | fllm | 44.3 | 2.21 | 148.3 | 23.5 |  |  |  |  | yes |
| Qwen3-8B Q4_K_M (GGUF) | llama.cpp | 29.0 | 2.58 | 165.5 | 24.9 | 1.53x | 1.17x | 0.90x | 0.94x |  |
| Qwen3-8B Q4_K_M (GGUF) | ik_llama.cpp | 29.5 | 2.35 | 168.6 | 15.5 | 1.50x | 1.06x | 0.88x | 1.52x | no |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 93.3 | 0.94 | 326.2 | 55.0 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 71.3 | 1.32 | 302.1 | 54.7 | 1.31x | 1.40x | 1.08x | 1.01x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 59.9 | 1.15 | 289.7 | 26.9 | 1.56x | 1.22x | 1.13x | 2.04x | no |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | fllm | 62.8 | 1.16 | 281.7 | 44.2 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | llama.cpp | 55.2 | 1.13 | 297.9 | 54.4 | 1.14x | 0.97x | 0.95x | 0.81x |  |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 51.2 | 1.14 | 274.3 | 26.9 | 1.23x | 0.98x | 1.03x | 1.64x | no |
| Gemma 4 E4B Q4_K_M (GGUF) | fllm | 41.8 | 2.58 | 68.7 | 20.3 |  |  |  |  | yes |
| Gemma 4 E4B Q4_K_M (GGUF) | llama.cpp | 47.8 | 1.47 | 197.4 | 37.6 | 0.87x | 0.57x | 0.35x | 0.54x |  |
| Gemma 4 E4B Q4_K_M (GGUF) | ik_llama.cpp | - | - | - | - | - | - | - | - | no |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | fllm | 34.6 | 2.43 | 149.9 | 21.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | llama.cpp | 21.7 | 2.47 | 149.8 | 24.2 | 1.59x | 1.02x | 1.00x | 0.89x |  |
| Llama-3.1-8B-Instruct Q6_K (GGUF) | ik_llama.cpp | 19.9 | 2.40 | 154.3 | 15.6 | 1.74x | 0.99x | 0.97x | 1.38x | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | fllm | 0.0 | - | - | - |  |  |  |  |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | llama.cpp | 23.2 | 2.43 | 157.3 | 26.1 | - | - | - | - |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 25.2 | 2.19 | 161.0 | 16.4 | - | - | - | - | no |

### CPU, AMD EPYC 9B45 (Zen 5), 24 threads

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 500 token prompt, 1 user (less is better). 4 users = total tokens/s of 4 simultaneous requests (prompt 100 / prompt 500). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 500 (s) | 4 users, p100 | 4 users, p500 | decode vs fLLM | prefill vs fLLM | 4u p100 vs fLLM | 4u p500 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 67.0 | 0.91 | 63.0 | 18.9 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 59.6 | 0.68 | 116.5 | 49.1 | 1.12x | 0.75x | 0.54x | 0.38x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 64.7 | 0.37 | 160.8 | 74.2 | 1.04x | 0.41x | 0.39x | 0.25x | no |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 27.0 | 1.55 | 31.3 | 8.6 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 27.9 | 1.53 | 55.2 | 23.3 | 0.97x | 0.99x | 0.57x | 0.37x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 29.0 | 0.78 | 73.7 | 35.1 | 0.93x | 0.50x | 0.42x | 0.25x | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | fllm | 42.8 | 2.13 | 39.3 | 15.2 |  |  |  |  | no |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | llama.cpp | 17.3 | 2.27 | 33.7 | 15.2 | 2.47x | 1.07x | 1.17x | 1.00x |  |
| Llama-3.1-8B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 17.6 | 0.75 | 50.2 | 30.8 | 2.43x | 0.35x | 0.78x | 0.49x | no |
| Qwen3-8B Q4_K_M (GGUF) | fllm | 24.6 | 1.67 | 26.9 | 8.2 |  |  |  |  | yes |
| Qwen3-8B Q4_K_M (GGUF) | llama.cpp | 26.9 | 1.56 | 54.6 | 22.4 | 0.91x | 0.93x | 0.49x | 0.37x |  |
| Qwen3-8B Q4_K_M (GGUF) | ik_llama.cpp | 29.1 | 0.88 | 66.7 | 32.9 | 0.85x | 0.53x | 0.40x | 0.25x | no |

### CPU, AMD EPYC 7B12 (Zen 2), 8 threads

decode = tokens/s one user sees after its first token (prompt 100). prefill = seconds to the first token with a 500 token prompt, 1 user (less is better). 4 users = total tokens/s of 4 simultaneous requests (prompt 100 / prompt 500). Ratios are fLLM against the engine: above 1.00 fLLM is ahead (for prefill the ratio is the engine's time divided by fLLM's).

| model | engine | decode | prefill 500 (s) | 4 users, p100 | 4 users, p500 | decode vs fLLM | prefill vs fLLM | 4u p100 vs fLLM | 4u p500 vs fLLM | same answer as llama.cpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | fllm | 16.4 | 7.07 | 10.4 | 3.3 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | llama.cpp | 21.5 | 8.43 | 13.1 | 4.6 | 0.76x | 1.19x | 0.79x | 0.72x |  |
| Llama-3.2-3B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 22.7 | 7.81 | 16.2 | 5.2 | 0.72x | 1.10x | 0.64x | 0.63x | no |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | fllm | 12.2 | 11.91 | 10.3 | 3.2 |  |  |  |  | yes |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | llama.cpp | 15.0 | 12.34 | 9.4 | 3.3 | 0.81x | 1.04x | 1.10x | 0.97x |  |
| Llama-3.2-3B-Instruct Q8_0 (GGUF) | ik_llama.cpp | 11.0 | 9.42 | 13.9 | 4.1 | 1.11x | 0.79x | 0.74x | 0.78x | no |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | fllm | 6.0 | 17.98 | 4.8 | 1.3 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | llama.cpp | 10.4 | 20.28 | 5.6 | 2.0 | 0.58x | 1.13x | 0.86x | 0.65x |  |
| Llama-3.1-8B-Instruct Q4_K_M (GGUF) | ik_llama.cpp | 8.1 | 17.64 | 7.1 | 2.2 | 0.74x | 0.98x | 0.68x | 0.59x | no |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | fllm | 6.3 | 18.77 | 4.8 | 1.4 |  |  |  |  | yes |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | llama.cpp | 10.9 | 17.62 | 5.6 | 2.2 | 0.58x | 0.94x | 0.86x | 0.64x |  |
| Llama-3.1-8B-Instruct Q2_K (GGUF) | ik_llama.cpp | 13.0 | 17.75 | 6.9 | 2.0 | 0.48x | 0.95x | 0.70x | 0.70x | no |

The Zen 2 machine was also run with 4 threads for both engines (the 4 physical cores): Llama-3.2-3B Q4_K_M decode 14.5 (fLLM) against 17.0 (llama.cpp), Llama-3.1-8B Q4_K_M 5.1 against 7.4.
