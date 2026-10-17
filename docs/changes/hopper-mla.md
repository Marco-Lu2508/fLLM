# Latent-cache multi-head latent attention (CUDA)

Fragment of the Hopper / Blackwell work (part: MLA). Merge into the central documents; nothing in this file is measured on a GPU (see "Test levels").

## What it does

DeepSeek-V2/V3, Kimi K2 and GLM-MoE-DSA models use multi-head latent attention (MLA). The default path (`FLLM_MLA=expanded`) rebuilds per-head keys and
values from the compressed latent and caches those: `heads * (nope + rope)` key values and `heads * value` value values per token and layer
(5120 for DeepSeek-V2-Lite, 40960 for the 128-head models), attended with FP32 CUDA-core kernels because `dk != dv` excludes the tensor-core attention kernels.

`FLLM_MLA=latent` caches what MLA compresses to, the normalized latent and the rotated shared rope key: `kv_lora + rope` = 576 values per token and layer
(1156 bytes in f16 against 10240 / 81920 bytes: 8.9x / 70.9x less cache). Decode and prompt rows attend in the latent space (matrix absorption):

* queries: `q_lat[h] = [q_nope[h] W_UK[h] | rope(q_pe[h])]` (tensor cores, `mla_latent_query`),
* attention over the shared rows `[latent | rope key]` (keys = the row, values = its first `kv_lora` values; mma.sync m16n8k16, online softmax, split over
  keys for long contexts; `mla_latent_attention`),
* output: `W_UV[h] (sum_t p_t latent_t)` (tensor cores, `mla_latent_output`).

Cache formats: f16, q8_0, int8, fp8_e4m3 (`--kv-cache-type`). Pages of 64 tokens. The CPU decoder and every other backend keep the expanded path; the latent cache is
CUDA only and falls back (with a warning) when the device, the dimensions or the cache format are not covered (kv_lora / rope in {512/64, 256/64, 256/32,
128/64, 128/32, 64/32}, nope a multiple of 16 up to 256, value a multiple of 32 up to 256, compute capability 7.5 or newer, NVIDIA only).

The kernels cover prompts too (a run of 16 or more rows of one sequence packs 16 rows into the M dimension of one CTA); that costs about 3.4x the attention
FLOPs of the expanded prompt attention (576 + 512 against 192 + 128 per key and head), in exchange for 9-70x less cache. A prompt path that expands K/V on
the fly from the latent (vLLM's chunked-context prefill) is not implemented.

## Commands

```
# build: the latent kernels are part of the normal CUDA build (cuda/mla_latent.cu, cuda/mla_decode.cu)
cargo build --release --features cuda

# self test on the GPU (H100, L4, A100, G4 / RTX PRO 6000 Blackwell): each stage against an f64 reference, then the whole chain against the expanded formulation
./target/release/fllm-rs selftest-gpu --only hmla
FLLM_SELFTEST_BENCH=1 ./target/release/fllm-rs selftest-gpu --only hmla      # adds: latent against expanded attention, 8 sequences of 4095 tokens, cache MB and ms
FLLM_MLA_KERNEL=ref ./target/release/fllm-rs selftest-gpu --only hmla        # the CUDA-core reference kernel everywhere (isolates the data movement from the MMA kernel)

# a model (DeepSeek-V2-Lite, any MLA GGUF): the log says "Activated the latent MLA cache" and the KV pool plan shows the smaller cache
FLLM_MLA=latent ./target/release/fllm-rs serve --model <model.gguf> --device cuda --ctx 32768
FLLM_MLA=latent ./target/release/fllm-rs serve --model <model.gguf> --device cuda --kv-cache-type fp8_e4m3
```

Expected outcomes (to be confirmed on the GPUs; none of this has run): every `mla latent ...` check `ok`; the checks name the failing stage
(`mla latent kv`, `query`, `attention`, `attn ref`, `output`, `vs expanded`). `attn ref` ok and `attention` FAIL means the MMA kernel; both FAIL with `query` ok
means the cache layout or the block tables; `kv` FAIL means the rotary tables or the cache write. With greedy sampling the first generated tokens of the latent and
the expanded path should agree (FP16 rounding of the absorbed queries and probabilities can flip a late token). With `FLLM_SELFTEST_BENCH=1` the latent attention should
read about 9x (16 heads) / 71x (128 heads) fewer cache bytes than the expanded one; the time ratio is the result to report.

## Environment

| variable | default | meaning |
|---|---|---|
| `FLLM_MLA` | `expanded` | `latent`: latent cache and absorbed attention (CUDA only, falls back with a warning); `expanded`: the previous behavior |
| `FLLM_MLA_KERNEL` | tensor cores | `ref`: the CUDA-core reference attention kernel of the latent path (slow, for diagnosis) |
| `FLLM_SELFTEST_BENCH` | off | `1`: the Hopper / Blackwell self-test groups add timing cases |

## THIRD_PARTY entries

* DeepSeek-V2 paper (matrix absorption of the up projections), formulation only.
* FlashMLA, https://github.com/deepseek-ai/FlashMLA, MIT License, Copyright (c) 2025 DeepSeek: the design of the absorbed decode (all heads attend over one
  shared key/value row of `kv_lora + rope` values, head-parallel M dimension of the MMAs, split over keys with a merge). No source file was copied.
* vLLM, https://github.com/vllm-project/vllm, `vllm/v1/attention/backends/mla`, Apache License 2.0: the split between the absorbed decode path and the up-projections,
  `W_UK` / `W_UV` taken from `kv_b_proj`. No source file was copied.
* SGLang, https://github.com/sgl-project/sglang, Apache License 2.0: the weight-absorption formulation of its MLA backends (reference). No source file was copied.

## Test levels

* L0 (compiles): `cuda/mla_latent.cu` and `cuda/mla_decode.cu` compile for sm_61 .. sm_100 and sm_120 with CUDA 12.8 (sm_61 / sm_70 bodies are empty, as for the
  other tensor-core files), and for sm_80 / sm_90 / sm_100 / sm_120 with CUDA 13.0; no register spills at sm_89 (126 registers for 512 + 64 in f16, 95 for the
  scaled formats). `cargo clippy` clean with `--features cuda` (kernel build skipped for the check). The ROCm build gets declining stubs (`hip_stubs.cu`).
* L0 on the CPU (run): `cargo test -p fllm-engine --lib mla_latent`: the absorbed and the expanded formulation agree to 1e-9 (several shapes, neox and pair rotary, causal lengths);
  cache memory per token (8.9x / 70.9x); fp16 kv_b conversion; chunking.
* L1 (self test on a device): not run. Group `hmla` exists (selftest-gpu --only hmla).
* L2 / L3: not run.

## Not done

* Nothing was run on a GPU: every kernel is unproven (no L1 result). The MMA fragment layouts, the shared-memory addressing and the softmax bookkeeping were
  checked by reading and by compilation only; the CUDA-core reference kernel (`FLLM_MLA_KERNEL=ref`) and the staged checks of the `hmla` group exist to find the
  defect quickly if there is one.
* No end-to-end model run, no comparison with `transformers` on a tiny random DeepSeek checkpoint (the engine has no tiny MLA fixture; the CPU MLA
  path itself is untouched).
* Prompt attention in the expanded form from the latent cache (vLLM's chunked-context prefill), a wgmma / Hopper variant, a sparse-attention (DSA) indexer, a
  window or chunked variant, and bf16 / 4-bit latent caches.
* Latent cache for the draft model / MTP layer, and CUDA graphs for MLA layers.
* CUDA 13 compile check covers sm_80, sm_90, sm_100 and sm_120 only (not sm_103 / sm_110 / sm_121, which need no special code: the kernels are architecture-neutral).

## Known limits

* Prompt attention uses the absorbed kernel (about 3.4x the FLOPs of expanded prompt attention).
* Heads are processed 16 per CTA; models with 128 heads re-read the cache once per head tile (L2 traffic, not DRAM, for typical batches).
* Windows, chunked attention, sinks and soft-capping are not covered (the decoder refuses); DeepSeek/Kimi/GLM-MoE-DSA do not use them. The sparse (DSA) indexer of
  GLM-MoE-DSA is not part of this work.
* The kv_b matrix is dequantized to dense FP16 at load (`heads * (nope + value) * kv_lora * 2` bytes per layer: 33 MB per layer for 128 heads).
* No CUDA graph capture of latent layers (MLA layers are excluded from graphs already).
* No wgmma variant. MTP speculation needs the CPU backend and is not combined with the latent cache.
* With a speculative or shared-prompt block copy the cache pages are 64 tokens; shared prefixes are shared at that granularity.
