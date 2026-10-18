# Open-model coverage (Artificial Analysis)

Snapshot of 2026-10-13. Source: the open-weights views of <https://artificialanalysis.ai/models/open-source>
(overall list, size classes `tiny`, `small`, `medium`, `large`). Artificial Analysis shows only the top 12 of each
size class on a page (tiny: 12 of 49, medium: 12 of 65, ...), so this table covers the **43 distinct open
models that are visible there**, not every open model AA tracks. Reasoning-effort variants of one checkpoint
(for example "Qwen3.8 27B (Xhigh / Medium / Low)") are one row. Closed models (GPT, Claude, Gemini, Grok) are out of scope.

Architecture names come from each model's `config.json` on the Hugging Face hub (`architectures` / `model_type`).

Status legend:

* **V** supported and verified: fLLM's conversion of a tiny random checkpoint of that architecture matches
  `transformers` (logits, `scripts/verify_architectures.py`, relative difference < 2e-3). "+GGUF" means a real
  llama.cpp-style GGUF was additionally run and compared with llama.cpp (a build of ggml-org/llama.cpp, used only as a
  reference program).
* **U** supported but not (fully) verified, see note.
* **N** not supported (reason given). `--device cpu` is the only way to run anything new in this table; the GPU
  backend rejects the new layer types with an explanatory error.

"Verified" never means "the real checkpoint was compared with `transformers`": none of the large checkpoints fit
into the 30 GB test machine.

## Result

| | count |
| --- | --- |
| open models visible on AA (distinct) | 43 |
| V verified (architecture, tiny checkpoint vs `transformers` or the model's remote code) | 25 |
| U supported, but not fully verified or with a documented gap | 2 (LFM2.5, K2 Horizon 375B) |
| N not supported | 16 |

Newly supported by this work: Muse Glimmer, K2 Horizon MoE/MoVA (36B A4B, and the 375B A23B text model except its
partial rope), G9v3, Nemotron-H with latent experts (Nemotron 3 Super / Ultra), llama.cpp `nemotron_h_moe`
GGUFs (Nemotron 3 Nano), and Qwen3.8 27B GGUF checks.

## Home-user models first (8-24 GB VRAM at 4-bit)

| Model (AA rank in class) | Params | HF architecture | GGUF architecture | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8 27B (small #1) | 27B dense | `Qwen3_5ForConditionalGeneration` / `qwen3_5_text` | `qwen35` | V +GGUF | Same layers as Qwen3.5 (gated DeltaNet + gated attention). The unsloth Q4_0 file loads, layer sums agree with llama.cpp within ~0.5 %, greedy text equals llama.cpp wherever the next token is not a near tie. `output_gate_type: swish` is not read by `transformers` 5.19; llama.cpp ignores it too. Vision tower and MTP head are not imported. |
| K2 Horizon MoVA 36B A4B (small #4) | 36B / 4B active | `K2HorizonForCausalLM` | `k2-horizon` | V (tiny) +GGUF | New: sigmoid MoE with selection bias, shared expert, leading dense layers, grouped RMSNorm, softplus attention gate and routed value experts ("MoVA"). Verified against the model's own remote code. IFM's Q4_K_M GGUF (22 GB) gives the same 24 tokens as llama.cpp on three prompts, 2.1 tok/s on 10 CPU cores. |
| G9v3-39A5B (small #5) | 39B / 5B | `G9v3ForCausalLM` | `g9v3` | V (tiny), U (GGUF) | New: DeepSeek-style sigmoid MoE (320 experts, top 32) with per-head sigmoid-gated attention. Verified against the model's remote code. The community GGUFs (arch `g9v3`) are accepted by name, rope pairing of those files is not checked (no reference program knows the arch). |
| K2 Horizon 7B / 3.7B / 0.9B (small/tiny) | dense | `K2HorizonForCausalLM` | `k2-horizon` | V | The converter writes `attention.group_norm_groups` and converts dense checkpoints. Grouped RMSNorm is implemented on CPU, CUDA (compiled) and Vulkan (verified with software rasterizer). |
| Muse Glimmer 30B (small #9) | 30B dense | `MuseGlimmerForConditionalGeneration` | `muse-glimmer` | V +GGUF | New. Gemma-style sandwich norms with `1+w` folded in the GGUF, silu MLP, per-head sigmoid gate, weightless QK norm (q scaled by 3.87), NoPE global layers, embedding RMSNorm, logits * 0.196 then tanh cap 20. Unsloth Q4_K_XL produces the same 40 tokens as llama.cpp. Vision tower not imported. |
| Qwen3.6 35B A3B (small) | 35B / 3B | `Qwen3_5MoeForConditionalGeneration` | `qwen35moe` | V (+GGUF runs) | Previously supported. |
| Gemma 4 26B A4B, Gemma 4 31B (small) | 26B / 31B | `Gemma4ForConditionalGeneration` | `gemma4` | **N** | Different head width and KV head count for sliding (256 x 8) and global (512 x 2) layers, K = V on global layers, proportional partial rope, an MoE block next to a dense MLP in the 26B. fLLM has one head geometry per model: needs per-layer KV geometry in the paged cache (estimated > 3 h with verification). **Biggest remaining gap for home users.** |
| MiniCPM5 2B / 1B, G9v3-3B, Nanbeige4.1-3B (tiny) | dense | `LlamaForCausalLM` | `llama` | V | Tokenizer `minicpm5` exists. |
| Granite 4.2 3B (tiny) | dense | `GraniteForCausalLM` | `granite` | V | |
| Phi-4 Mini (tiny) | 3.8B | `Phi3ForCausalLM` | `phi3` | V | LongRoPE. |
| Qwen3.5 2B / 0.8B (tiny) | | `Qwen3_5ForConditionalGeneration` | `qwen35` | V +GGUF | 0.8B GGUF was compared earlier. |
| LFM2.5 2.6B (tiny) | | `Lfm2ForCausalLM` | `lfm2` | U | Implemented; reference comparison still pending, CPU only. |

## Medium (25-130B)

| Model | Params | HF architecture | Status | Notes |
| --- | --- | --- | --- | --- |
| Ling-3.0-flash(-VL/-Fin) | 124B / 5.5B | `BailingMoeV3ForCausalLM` (`bailing_hybrid`) | **N** | MLA mixed with KDA linear-attention layers (`layer_group_size`); the converter's `bailing_hybrid` entry only knows the MLA part and refuses KDA checkpoints (missing tensors). |
| Qwen3.5 122B A10B | | `qwen3_5_moe` | V | |
| Mistral Medium 3.5 128B | dense | `Mistral3ForConditionalGeneration` / `ministral3` | V | Text model only. |
| Nemotron 3 Super 120B A12B | | `NemotronHForCausalLM` | V (tiny, new) | Needs the latent expert projections added here (`moe_latent_size: 1024`). Real weights not run (120B). |
| HyperNova 60B, gpt-oss-120b | | `GptOssForCausalLM` | V | MXFP4 import. |
| K2 Think V2 70B | dense | `LlamaForCausalLM` | V | |
| LongCat Flash Lite | 68.5B / 3B | `LongcatFlashNgramForCausalLM` | **N** | n-gram embeddings plus LongCat dual-sublayer topology. |
| Mistral Small 4 119B | | `mistral4` (MLA MoE) | V | |
| Qwen3 Next 80B A3B | | `Qwen3NextForCausalLM` | V | |
| Nemotron 3 Nano 30B A3B (not in the top 12, checked as a real GGUF) | | `NemotronHForCausalLM` | V (tiny) +GGUF | The llama.cpp GGUF uses architecture `nemotron_h_moe` and per-layer arrays for `head_count_kv` / `feed_forward_length`; both were unsupported before. Unsloth Q4_0 loads, text equals llama.cpp on peaked prompts. |

## Large (> 150B), not practical at home, listed for completeness

| Model | Params | HF architecture | Status | Reason |
| --- | --- | --- | --- | --- |
| MiMo-V2.6-Pro / -Flash | 1.0T / 309B | `MiMoV2ForCausalLM` | **N** | Different head counts and widths for sliding and global layers, learned attention sinks, fused QKV, value scale, FP8. |
| GLM-5.3 (Max / Low) | 753B | `GlmMoeDsaForCausalLM` | **N** | DeepSeek sparse attention indexer (shared/full indexer layers). |
| GLM-5.3-Flash | 320B | `Glm5NextForConditionalGeneration` | **N** | KDA linear attention + DSA + multi-stream residuals (`mhc`). |
| Kimi K3 | 2.8T | `KimiK3ForConditionalGeneration` | **N** | KDA layers, attention residual blocks, `situ` activation, latent MoE, MXFP4 compressed-tensors. |
| Qwen3.8 2.4T A95B | 2.4T | `Qwen3_5MoeForCausalLM` | V (arch) | Same layers as Qwen3.5-MoE; too large to run. |
| Qwen3.8-Flash-Next | 180B / 6B | `Qwen4ExpForConditionalGeneration` | **N** | Hyper-connections, n-gram embeddings, per-layer embeddings, sparse indexer. |
| DeepSeek V4.1 Flash, V4 Pro 0813 | 552B, 1.6T | `DeepseekV41ForCausalLM`, `DeepseekV4ForCausalLM` | **N** | Hyper-connections, engram embeddings, compressed/sparse attention. |
| Motif 3 | 314B | `MotifForCausalLM` | **N** | GDLA attention, multi-stream residuals, PolyNorm activation. |
| MiniMax-M3 | 428B | `MiniMaxM3SparseForConditionalGeneration` | **N** | Block-sparse attention index on top of MiniMax-M2, `swigluoai`. |
| Inkling | 975B | `InklingForConditionalGeneration` | **N** | Short convs, relative position tables, log scaling, shared expert sink. |
| Nemotron 3 Ultra | 550B / 55B | `NemotronHForCausalLM` | V (tiny, new) | Latent MoE, same code as Super. MTP head is not imported. |
| K2 Horizon 375B A23B | 375B / 23B | `K2HorizonForCausalLM` | U | MoE path verified (tiny). The checkpoint uses `rope_head_dim: 64` < `head_dim: 128` with an interleaved partial layout; the converter rejects it instead of guessing. |

## Priority of the gaps

1. **Gemma 4 (26B A4B, 31B)** - most downloaded gap (11 M and 9.5 M downloads) and the only one that fits 24 GB.
   Needs per-layer KV geometry, K = V sharing, proportional rope and the dense + MoE block.
2. Ling-3.0-flash (124B, 5 GB active at 4-bit with CPU offload) - KDA layers, which are also needed by Kimi-Linear/K3 and GLM-5.3-Flash.
3. LongCat Flash Lite, Qwen3.8-Flash-Next (n-gram embeddings, hyper-connections): small active size but exotic layers.
4. The 300B+ models (MiMo, GLM, DeepSeek V4, Kimi K3, MiniMax-M3, Inkling, Motif): not usable on a home machine, and each
   adds sparse attention indexers or multi-stream residuals; worth doing only after the above.

## Real GGUF runs (CPU, 10 cores)

| File | Result |
| --- | --- |
| unsloth Muse-Glimmer-30B-UD-Q4_K_XL | 40 greedy tokens identical to llama.cpp for the 68-token chat prompt and the multi-turn prompt. 3.9 tok/s. Found and fixed: GGUF rope is interleaved with permuted Q/K rows (llama style), not NeoX. |
| unsloth Qwen3.8-27B-Q4_0 | Layer-output sums within 0.5 % of llama.cpp over 64 layers; `1, 2, 3 ...`, the arithmetic prompt (`391`) and the first tokens of two other peaked prompts are identical; flat prompts diverge at near ties (the top-2 log-probabilities differ by 0.27). 5.4 tok/s. |
| unsloth Nemotron-3-Nano-30B-A3B-Q4_0 | Loads after the `nemotron_h_moe` / per-layer-array fixes; first tokens equal llama.cpp on peaked prompts. |
| IFM K2-Horizon-MoVA-36B-A4B-Q4_K_M | 24 greedy tokens identical to llama.cpp on three prompts (counting, chemistry facts, numpy code). Exercises MoE, value experts, softplus gate and grouped norms. |

## How the checks were done

* `scripts/hf_golden.py` builds tiny random checkpoints (new fixtures: `muse_glimmer`, `nemotron_h_latent`,
  `k2_horizon_moe`, `k2_horizon_mova`, `g9v3`; the last three use the models' remote code, see the comments in the
  script) and `scripts/verify_architectures.py` compares all positions' logits with `transformers` 5.19.
  Results: Muse 6.2e-4 (12 tokens) / 1.6e-3 (70 tokens), Nemotron latent 1.4e-4, K2 MoE 1.9e-4, K2 MoVA 1.5e-4
  (2.1e-4 for 60 tokens), G9v3 1.6e-4, no arg-max mismatch.
* GGUF files were compared with a CPU build of llama.cpp (`llama-completion`, `llama-eval-callback`) used as an
  independent reference program; no llama.cpp code is part of fLLM.
* `scripts/verify_generation.py` reports a 1-of-4 text mismatch for both `nemotron_h` and the new latent variant
  (random models, near ties); it predates this work.
