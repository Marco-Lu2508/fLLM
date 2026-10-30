# Model and format coverage

Audit of 2026-10-30 (code, `scripts/hf_golden.py` fixtures, transformers 5.22.1 as reference). Three parts: the
architecture matrix (what runs where and how it was verified), the quantization / checkpoint formats, and the older
snapshot against the open models listed on artificialanalysis.ai (further below). The checklist against the vLLM / SGLang
registries is [model-coverage-vllm.md](model-coverage-vllm.md); the per-family tensor and config notes are in
[architectures.md](architectures.md).

## How to read the matrix

Columns CPU / CUDA / Vulkan / HIP / Metal say what the decoder does for the layer types of the family. They are **not**
measurements: hardware evidence is listed separately under "Runs on real devices" and everything else is the code path.

* **D** the layer types have device kernels in that backend (compiled; see [backends.md](backends.md) for what ran).
* **H** some layers run on the host between device layers (`layer_host_reason`, [gpu-backends.md](gpu-backends.md) section 3a):
  the model runs, the host part is not accelerated.
* **C** CPU path (the reference path) only.

Verified column:

* **T** the converted tiny random checkpoint of that architecture gives the same logits as `transformers` for every
  position (`scripts/verify_architectures.py`, relative difference below 2e-3, the table shows the measured value). Where
  the model lives in remote code, the reference is that code. Cached/batched decoding is covered by
  `scripts/verify_generation.py` for the recurrent families only.
* **T+G** as T and a real GGUF file of the family was compared with llama.cpp.
* **R** a real (small) checkpoint was run.
* **L0** mapped and compiled, no reference comparison: it may still fail on a real checkpoint.
* **N** not supported; the converter refuses with the reason.

## Decoder-only language models

| Architecture (HF `model_type` / GGUF arch) | Checkpoint / GGUF | CPU | CUDA | Vulkan | HIP | Metal | Verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 1-3, Mistral, Mixtral, Yi, Vicuna, DeepSeek-R1-Distill-Llama (`llama`, `mistral`, `mixtral`) | safetensors, GGUF | D | D | D | D | D | T 2e-4 (also `llama_wide` 1e-3, `mistral_swa`) |
| Ministral 8B (`ministral`, Mistral with a sliding/full pattern) | safetensors | D | D | D | D | D | T 1.7e-4 (new) |
| Ministral 3 / Mistral Medium 3.x (`ministral3`) | safetensors, GGUF | D | D | D | D | D | T 1.9e-4 |
| Helium (`helium`, interleaved rotary pairs) | safetensors | D | D | D | D | D | T 1.8e-4 (new) |
| BitNet b1.58 (`bitnet`: relu2, attention and FFN sub-norms) | safetensors (dense and HF-packed ternary), GGUF `bitnet` and the BitNet fork's `bitnet-b1.58` (I2_S) | D | D | D | D | H (sub-norms use the ordinary RMS norm; not run on Metal) | T 2.7e-4 dense, 2.4e-4 packed; R: the real I2_S GGUF against transformers 5.2e-3 (see below) |
| Granite 3/4 dense, Granite SWA (`granite`, `granite_swa`) | safetensors, GGUF | D | D | D | D | D | T 8e-5; `granite_swa` 8e-5 with a generic probe config |
| Qwen 1 (`qwen`), Qwen2/2.5 (`qwen2`), Qwen3 (`qwen3`), Qwen-VL text parts (`qwen2_vl_text`, `qwen3_vl_text`) | safetensors, GGUF | D | D | D | D | D | T 7e-5 / 1.5e-4 / 1.6e-4; T+G for Qwen3 |
| Gemma 1/2/3, VaultGemma (`gemma`, `gemma2`, `gemma3_text`, `vaultgemma`) | safetensors, GGUF | D | D | D | D | D | T 2.2e-4 / 2.4e-4 / 4.6e-4 / 4.9e-4 |
| Gemma 3n, Gemma 4 dense and MoE (`gemma3n`, `gemma4`) | safetensors, GGUF | D | H (AltUp, per-layer embeddings: the whole model runs on the host) | H | H | H | T (transformers 5.19, `scripts/verify_gemma4.py`; not re-run on 5.22) |
| Muse Glimmer (`muse-glimmer`) | safetensors, GGUF | D | D | D | D | D | T+G 6e-4 |
| Phi-1/2/3/3.5/4-mini (`phi`, `phi3`, LongRoPE), Phi-3.5-MoE (`phimoe`) | safetensors, GGUF | D | D | D | D | D | T 7.7e-5 / 2.1e-4 / 3.3e-4 / 2.0e-4 |
| OLMo 1/2/3 (`olmo`, `olmo2`, `olmo3`), OLMoE, FlexOlmo | safetensors, GGUF | D | D | D | D | D | T 1.9e-4 .. 4.1e-4 |
| OLMo hybrid (`olmo_hybrid`, gated DeltaNet + attention) | safetensors | D | D | D | H | D | T 2.1e-4 |
| Cohere / Command-R, Command-A (`cohere`, `cohere2`, `cohere2_moe`) | safetensors, GGUF | D | D | D | D | D | T 1.3e-4 .. 1.6e-4 |
| StarCoder2, GPT-NeoX/Pythia, StableLM, GPT-J, GPT-2, GPT-Neo, GPT-BigCode, OPT, Falcon (1/2), Bloom, MPT | safetensors, GGUF | D | D | D | D | D | T 4e-5 .. 2.2e-4 (all fixtures) |
| GLM-4 / GLM-4-0414 (`glm`, `glm4`), ChatGLM2/3 (`chatglm`) | safetensors | D | D | D | D | D | T 1.4e-4 / 2.3e-4; the `chatglm` fixture needs the model's remote code and does not build with transformers 5.22 (earlier run 3.2e-4) |
| Nemotron-4 (`nemotron`), InternLM2/3, Baichuan 1/2, XVERSE, Orion, Solar, MiMo (dense), IQuest, EXAONE 3/4, SmolLM3, Seed-OSS, Arcee, ERNIE 4.5, Hunyuan dense, Jais 2, CWM, HyperCLOVAX, NanoChat, Apertus, Persimmon, MiniCPM 1/2 | safetensors, GGUF | D | D | D | D | D | T 4e-5 .. 1.6e-3 (one fixture each) |
| Llama 4 (`llama4`, chunked attention, NoPE layers) | safetensors, GGUF | D | D | D | D | D | T 1.6e-4 |
| gpt-oss (`gpt_oss`, sinks, clamped SwiGLU) | safetensors (MXFP4), GGUF | D | D | D | D | D | T 1.2e-4 |
| Qwen2-MoE, Qwen3-MoE, Mellum (`qwen2_moe`, `qwen3_moe`, `mellum`) | safetensors, GGUF | D | D | D | D | D | T 1.3e-4 / 2.0e-4 / 1.9e-4 |
| GLM-4.5/4.6/4.7 (`glm4_moe`), GLM-4.x-Flash (`glm4_moe_lite`), dots.llm1, ERNIE 4.5 MoE, Hunyuan MoE/Hy3, MiniMax-M2, EXAONE-MoE, AFMoE/Trinity, Solar-Open, Sarvam-MoE, GraniteMoE (+shared, SWA) | safetensors, GGUF | D | D | D | D | D | T 1.1e-4 .. 2.7e-4 |
| Step-3.5-Flash / Step-3.7-Flash text (`step3p5`) | safetensors (Hub layout); llama.cpp `step35` GGUFs are refused (clamped SwiGLU, scaled partial rotary factors) | D | D (per-head gate) | D | D | H (attention layers: per-head gate) | T 6.0e-4 (new; the real 196B checkpoint was not run). The clamped SwiGLU of the last layers (`swiglu_limits`) is not implemented |
| MiMo-V2-Flash (`mimo_v2_flash`: per-layer KV heads, sinks, value scale) | safetensors, llama.cpp `mimo2` GGUF (alias) | D | D (value scale) | D | H on AMD GPUs (attention layers stay on the host: the HIP decode step differs, see device runs) | H (attention layers: value scale) | T 2.7e-4 (new fixture; the real 309B checkpoint was not run); the `mimo2` header of a real file was checked against the metadata keys, no such file was run |
| DeepSeek V2/V3/R1, Kimi K2, Mistral Large 3 / Mistral 4, Youtu, Sarvam-MLA, MiniCPM3, Ling-2 (`deepseek_v2`, `deepseek_v3`, `kimi_k2`, `mistral4`, ...) | safetensors (also FP8 block), GGUF `deepseek2` | D | D | D (equal KV head count) / H | D | D | T 1.4e-4 .. 2.0e-4 |
| DeepSeek-V3.2, GLM-5 / GLM-5.x (`deepseek_v32`, `glm_moe_dsa`; GGUF `deepseek32`, `glm-dsa` by name) | safetensors | D | D | D / H | D | D | T 2.3e-4 / 1.8e-4 with `index_topk` above the sequence length; **T 1.7e-4 / 3.5e-4 with the indexer cutting 4 of 12 tokens** (`deepseek_v32_sparse`, `glm_moe_dsa_sparse`). The exact lightning-indexer selection runs on the CPU path; accelerators attend densely (exact up to `index_topk` tokens) unless `FLLM_DSA=exact` puts the layers on the host. No device kernel for the indexer |
| DeepSeek-V4 / V4.1 (`deepseek_v4`; GGUF `deepseek4`) | safetensors (transformers and Hub tensor names; bf16/f16/f32) | D | C | C | C | C | T 5e-7 .. 1e-6 (new; 12..40 positions, also cached decoding and server completions). Accelerators run on the CPU with a notice; no MTP head; FP8/FP4 Hub weights not read; the real checkpoints were not run |
| MiniMax-M3 text (`minimax_m3_vl_text`; GGUF `minimax3`) | safetensors (vision tower not imported) | D | C | C | C | C | T 7e-7 .. 1.5e-6 (new; 12..40 positions, cached decoding, server completions; the block selection matters: dense attention is off by 0.40). Accelerators run on the CPU with a notice; the real 428B checkpoint was not run |
| Inkling text (`inkling_text`; GGUF `inkling`) | safetensors (vision, audio and MTP tensors not imported) | D | C | C | C | C | T 1.7e-7 .. 1.8e-7 (new; 12..40 positions, cached decoding, server completions; sliding window, distance clamp, log scaling, shared experts exercised). Accelerators run on the CPU with a notice; the real 975B checkpoint was not run |
| LongCat-Flash (`longcat_flash`; GGUF `longcat-flash`) | safetensors | D | C | C | C | C | T 1.4e-4 .. 1.7e-4 (new; 12..40 positions, cached decoding, server completions; zero experts and the cross-layer shortcut exercised). Accelerators run on the CPU with a notice; the real 560B checkpoint was not run |
| GLM-5-Next / GLM-5.3-Flash text (`glm5_next`; GGUF `kimi-linear` + hyper-connections) | safetensors (vision tower skipped) | D | C | C | C | C | T 8.7e-5 .. 9.2e-5 (new; 12..40 positions, cached decoding, server completions; the k-pool indexer cuts the context to 2 pools + tail). Accelerators run on the CPU with a notice; the real 320B checkpoint was not run |
| Ling-3.0 (`bailing_hybrid`: Kimi delta layers with direct bounded gates + rotary latent attention with a head gate + MoE; GGUF `kimi-linear`) | safetensors (Hub `attention.*` and transformers 5 names) | D | H | H | H | H | T 3.9e-5 .. 4.7e-5 (new; 12..40 positions, cached decoding, server completions). The latent attention layers (output gate) run on the host between device layers; Vulkan on an RTX 2000 Ada gave the same text as the CPU for the tiny model, the other devices were not run; the real checkpoint was not run |
| Hy4 (`hy_v4`; GGUF `hy4`) | safetensors (transformers and Hub names) | D | C | C | C | C | T 1.4e-4 .. 1.7e-4 (new; indexer cutting 4 keys, 12..40 positions, cached decoding, server completions). Accelerators run on the CPU with a notice; the real checkpoint was not run |
| Laguna (`laguna`, per-head attention gate) | safetensors | D | H (attention layers) | H | H | H | T 2.6e-4 |
| DeepSeek-V3.2, GLM-5 / GLM-5.x (`deepseek_v32`, `glm_moe_dsa`) | safetensors | D | D | D / H | D | D | T 2.3e-4 / 1.8e-4 (new). **Dense attention:** the sparse-attention indexer is not run; that equals DSA while the context is at most `index_topk` (2048) and is approximate beyond (see Open gaps) |
| Laguna (`laguna`, per-head attention gate) | safetensors | D | D | D | D | H (attention layers) | T 2.6e-4 |
| K2 Horizon, G9v3 (`k2_horizon`, `g9v3`) | safetensors, GGUF | D | D | D | D | D | T 1.5e-4 .. 1.9e-4 (remote-code reference) |
| Qwen3-Next, Qwen3.5/3.6/3.8, MoE (`qwen3_next`, `qwen3_5`, `qwen3_5_moe`; GGUF `qwen35`, `qwen35moe`) | safetensors, GGUF | D | D | D | H | D | T 1.2e-4 .. 3.0e-4; T+G for the Qwen3.5/3.8 GGUFs |
| Kimi-Linear 48B-A3B (`kimi_linear`: KDA + latent attention without rope + MoE) | safetensors | D | D | D | H | D | T 8.2e-5 (new fixture; the KDA recurrence is also checked against `fla.ops.kda.naive` in `rust/engine/testdata`) |
| RWKV-7, RWKV-6 (`rwkv7`, `rwkv6`; GGUF `rwkv7`, `rwkv6`) | safetensors (fla layout / HF RWKV-6), GGUF | D | C | C | C | C | T 4.7e-7 / 1.1e-6 against `scripts/rwkv_reference.py` (transformers has no RWKV-6/7); R: real 0.1B RWKV-7 and 1.6B RWKV-6, see below |
| Mamba 1/2, Falcon-Mamba (`mamba`, `mamba2`, `falcon_mamba`) | safetensors, GGUF | D | D | D (Mamba-2) / H | H | H | T 3e-7 .. 6e-7 |
| Granite 4 hybrid, Bamba, Nemotron-H and Nemotron-3 (`granitemoehybrid`, `bamba`, `nemotron_h`, latent MoE) | safetensors, GGUF | D | D | D / H | H | H | T 3e-5 .. 1.6e-4; T+G for Nemotron-3-Nano |
| Falcon-H1, Zamba2 | safetensors | D | H | H | H | H | T 1.8e-4 / 1.6e-3 |
| Jamba (`jamba`) | safetensors | D | D (CUDA scan) | H | H | H | T 1.2e-4 |
| LFM2, LFM2-MoE | safetensors | D | D (dense) / C (MoE) | H | H | H | T 1.8e-4 / 2.9e-4 |
| DBRX | safetensors | D | C | C | C | C | T 1.3e-4 |
| Grok-1, JetMoE, Step-3 (not 3.5), PLaMo, TeleChat, Param2, A.X, Hy4, GigaChat3.5, Pangu, xLSTM, HRM-Text, ZAYA | - | - | - | - | - | - | N (config name recognised, no decoder: the converter or the loader refuses with the reason) |
| DeepSeek-V4, Qwen3.8-Flash-Next (`qwen4_exp`), Inkling, Motif, Kimi K3, GLM-5-Next, Ling-3.0, MiniMax-M3, LongCat-Flash, DiffLlama, Doge, Kolibri1, RecurrentGemma, MiniMax-M1 | - | - | - | - | - | - | N (see Open gaps) |
| Grok-1, JetMoE, Step-3 (not 3.5), PLaMo, TeleChat, Param2, A.X, Hy4, GigaChat3.5, Pangu, xLSTM, RWKV 6/7, HRM-Text, ZAYA | - | - | - | - | - | - | N (config name recognised, no decoder: the converter or the loader refuses with the reason) |
| Grok-1, JetMoE, Step-3 (not 3.5), PLaMo, TeleChat, Param2, A.X, GigaChat3.5, Pangu, xLSTM, RWKV 6/7, HRM-Text, ZAYA | - | - | - | - | - | - | N (config name recognised, no decoder: the converter or the loader refuses with the reason) |
| Qwen3.8-Flash-Next (`qwen4_exp`), Motif, Kimi K3, DiffLlama, Doge, Kolibri1, RecurrentGemma, MiniMax-M1 | - | - | - | - | - | - | N (see Open gaps) |

## Encoders, rerankers, embedders

All use the `bert` GGUF architecture family and the Rust encoder (`encoder.rs`); no generation. Reference scripts are
`scripts/encoder_verify/*.py` (each converts a tiny random model, serves it and compares with transformers /
sentence-transformers). Encoders run on the CPU; the GPU is used only through the helper path (`helper_model_uses_gpu`).

| Family | Formats | Verified (run of 2026-10-30 on the build host) |
| --- | --- | --- |
| BERT, RoBERTa, XLM-R, DistilBERT, ELECTRA (embedding, cross-encoder rerank, classifier) | safetensors, GGUF | T: rerank and score within 2e-7, RoBERTa mean cosine 1.0 |
| ModernBERT (`modernbert`) | safetensors | T: cosine 0.99999997, max error 1.4e-4 .. 3.5e-4 of the reference range |
| NomicBERT / nomic-embed | safetensors | T (earlier run, also the real nomic-embed-text-v1.5) |
| BGE-M3 dense / sparse / multi-vector, ColBERT v2, SPLADE-style MLM heads | safetensors | T: dense cosine 1.0, ColBERT rows 1.0, sparse entries equal, scores within 2e-6 |
| GTE-new (`new`), Jina-XLM-R | safetensors | verification scripts need third-party remote code (`GTE_REMOTE_CODE_DIR`, `JINA_REMOTE_CODE_DIR`); not run in this audit |
| Qwen3-Embedding / Qwen3-Reranker, LlamaBidirectional, EmbeddingGemma (bidirectional Gemma-3 + Dense modules), reward models (Qwen2/Qwen3 sequence classification) | safetensors | T: cosine 1.0, rerank score within 1e-7 |
| sentence-transformers pipelines (pooling modes, Dense, Normalize, dimensions) | safetensors | T: cosine 1.0 (`verify_st_pipeline.py`) |

## Vision, audio, speech

| Part | State |
| --- | --- |
| Vision towers: Qwen2-VL, Qwen2.5-VL, Qwen3-VL | implemented in Rust, T against transformers ([vision-audio.md](vision-audio.md)) |
| SmolVLM / Idefics3 through a llama.cpp `mmproj` file | implemented, unit tests only |
| Gemma 3/4, Llama 4, Mistral 3, GLM-4V, InternVL, Pixtral, Step-3.7 and other vision towers | not imported: the text model converts, image input is not available (the tower tensors are skipped) |
| Audio input, ASR (Whisper, Parakeet, Qwen3-ASR), TTS, audio language models | not implemented in the engine |
| MTP / NextN heads | read for a few families (`nextn.*`), dropped for MiMo and Step-3.5 |

## Quantization and checkpoint formats

| Format | Read natively | Notes / verified |
| --- | --- | --- |
| GGUF F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | yes, native kernels | the writer is byte-identical to `llama-quantize` for the classic types |
| GGUF Q2_K..Q6_K (K-quants), IQ1..IQ4 (I-quants), TQ1_0/TQ2_0, MXFP4, NVFP4, Q1_0/Q2_0 | yes (35 types, `fllm-rs formats`) | native kernels for Q4_0..Q8_0 and Q4_K/Q5_K/Q6_K; the other types are decoded (expanded to F16/F32 at upload on GPUs) |
| safetensors F32/F16/BF16, sharded; PyTorch `.bin`/`.pt` | yes | direct loading without writing a GGUF file |
| GPTQ 2/3/4/8-bit, `gptq_v2`, `g_idx`, sym/asym | yes | checked against gptqmodel fixtures |
| AWQ 4-bit GEMM, asymmetric and symmetric (`zero_point: false`) | yes | symmetric added 2026-10-30; the layout of a real AutoRound-AWQ repository was checked from its tensor headers (qzeros all 8) |
| Intel AutoRound (`quant_method: auto-round`, `auto_round:auto_gptq`, `auto_round:auto_awq`) | yes (new) | read as GPTQ / AWQ; tensor names, shapes and the zero-point convention (stored minus one) were checked on the headers of real exports; no AutoRound checkpoint was run end to end. Other packings: refused with the reason |
| FP8 (per tensor, block-wise), `fbgemm_fp8`, ModelOpt FP8 | yes | stays FP8 on devices that have kernels |
| NVFP4 (ModelOpt, compressed-tensors), MXFP4 (gpt-oss, compressed-tensors) | yes | |
| bitsandbytes NF4/FP4 (double quant), LLM.int8 | yes | checked against bitsandbytes fixtures |
| MLX affine (2-8 bit, any group size) | yes | |
| MLX `mxfp4`, `nvfp4`, `mxfp8` modes | yes (new), expanded to F16 | decoded values equal `mx.dequantize` on tiny matrices (`scripts/make_mlx_fp_fixtures.py`, mlx 0.32.4 CPU); no real MLX fp-mode repository was run |
| EXL2, EXL3 | yes | [exl-formats.md](exl-formats.md) |
| compressed-tensors int4/int8 symmetric (pack-quantized, int-quantized) | yes | native `Q4Z` / `Q8_0` |
| compressed-tensors integer weights with zero points (asymmetric) or `weight_g_idx` (activation order), 1-8 bits | yes (new), expanded to F16 | zero points: decoded values equal the compressed-tensors 0.20.1 library's own decompression (4 and 8 bits, group and channel strategies, `scripts/make_ct_int_fixtures.py`). **Activation order is covered only by a hand-made unit test** of the group-per-column rule: compressed-tensors 0.20.1 no longer writes `weight_g_idx`, so no library reference exists, and no real act-order repository was run |
| GPTQ-Marlin / IST-DASLab Marlin repacked checkpoints (`checkpoint_format: marlin`; tensors `B`/`s` or `qweight`/`scales`), symmetric 4 and 8 bits | yes (new), read back to plain values and expanded to F16 | the inverse of vLLM's tile and scale permutation, checked against that permutation code on tiny matrices (`scripts/make_marlin_fixtures.py`) and on a real 4-bit TinyLlama Marlin repository (`B`/`s` layout, group 128): the converted model (CPU, f16) writes coherent, correct text on four prompts (Paris, counting, `fibonacci(n-1) + fibonacci(n-2)`, 100 degrees C); none of the four greedy texts is identical to the unquantized model's, they share 5 to 35 leading characters (4-bit quantization, not checked further); a wrong tile permutation would produce noise. `desc_act` and zero-point (`sym: false`) Marlin checkpoints are refused |
| AQLM (`quant_method: aqlm`, codes + codebooks + scales) | yes (new), expanded to F16 | decoded values equal `aqlm.utils._dequantize_weight` on tiny layers (8-bit codes in 2 codebooks; 12-bit codes with output groups, `scripts/make_aqlm_fixtures.py`); the tensor names, dtypes and shapes were checked on the header of a real 2-bit Llama-2 repository (`scripts/st_peek.py`, I16 codes `[4096, 512, 1]`, 1 x 16-bit codebooks); that 2.4 GB repository was not converted or run |
| Hugging Face BitNet packed ternary (`quant_method: bitnet`) | yes (new) | T 2.4e-4 (`scripts/verify_bitnet_packed.py`, packed with transformers' own `pack_weights`); the tensor layout also matches the headers of microsoft/bitnet-b1.58-2B-4T. The reference BitLinear rounds activations to 8 bits per token; fLLM does not |
| HQQ, torchao, Quark, EETQ, QuIP#, SpQR, VPTQ, HIGGS, fp_quant, BitBLAS / IPEX repacks | no | the converter names the method, the reason and the way out (re-export as GPTQ/AWQ/compressed-tensors, or use the original weights). Header checks of real repositories: HQQ repositories ship pickled `qmodel.pt` files (Python `meta` dicts); a torchao "HQQ" export (`SocialLocalMobile/Qwen3.5-35B-A3B-HQQ-INT4`) stores `Int4TilePackedTo4dTensor` inner tensors (`__qdata` I32 `[n/8, k/128, 32, 4]`, `__scale_and_zero`) in the device-specific tinygemm tile layout under ExecuTorch tensor names; EETQ and the rotation-based methods (QuIP#, HIGGS, fp_quant) cannot be expanded layer by layer. Not decoded: no reference could be built in the time available, and guessing a layout would give wrong weights |

## Real checkpoints run in this audit

CPU only, on the shared build host, small models only (the downloads were deleted afterwards). `scripts/verify_real_model.py` compares
greedy continuations of four short prompts (24 tokens, f16 KV cache, so near ties can flip) with transformers in float32;
logits were compared with `verify_checkpoint` (f32 KV cache) where noted.

| Model | What it exercises | Result |
| --- | --- | --- |
| microsoft/bitnet-b1.58-2B-4T (packed uint8 ternary weights) | `quant_method: bitnet` reader, `bitnet` architecture with sub-norms, tied embeddings | logits of 19 positions against transformers on the unpacked weights (`scripts/unpack_bitnet.py`): relative 1.7e-4, arg-max equal everywhere. Greedy text: the first audit saw divergences after 4 to 38 characters and blamed the f16 KV cache. **That was a prompt bug, not numerics:** the converter ignored the BOS template of `tokenizer.json`, so fLLM prompted the model without BOS. With the fix, four of four continuations are identical to transformers with the f16 and with the f32 KV cache (`scripts/verify_kv_numerics.py`). The reference BitLinear's 8-bit activation rounding is applied by neither side |
| microsoft/bitnet-b1.58-2B-4T-gguf (`ggml-model-i2_s.gguf`, architecture `bitnet-b1.58`, 1.19 GB, I2_S tensors) | the BitNet fork's GGUF layout, I2_S decoding (kept as exact Q8_0 blocks), neox rotary pairs | logits of 16 positions against transformers on the unpacked bf16 weights: relative 5.2e-3 (max abs 0.23 on a span of 45), arg-max equal everywhere. The difference is larger than for the safetensors path because the file keeps the f32 weight scale while the reference rounds it to bf16 (not isolated further). Greedy text: three of four continuations identical to transformers, the fourth differs after 48 characters (`is_prime(num)` against `is_prime(n)`). The decoded weights of four tensors of layer 0 agree with the unpacked reference up to that scale rounding (0.15 %), so the block layout is right |
| kyutai/helium-1-2b | The Hub config of this model is plain `llama`, so it runs through the Llama path (the new `helium` model type is for transformers-native Helium configs only) | logits of 18 positions: relative 5.4e-4, arg-max equal. Greedy text: the first audit saw two of four prompts diverge. Cause: `config.json` names BOS id 0 while the tokenizer template puts `<s>` (id 1) in front; fLLM used 0. Fixed (the template wins over the configuration): four of four identical to transformers with the f16 and the f32 KV cache |
| RWKV/RWKV7-Goose-World2.8-0.1B-HF (RWKV-7, 0.1B) | `rwkv7` converter (fla layout, bf16 weights) and the llama.cpp F16 GGUF of the same model (fused mixing vectors, `rwkv` tokenizer) | logits of 12 positions against `scripts/rwkv_reference.py` (float32 on the original weights): converted checkpoint relative 1.9e-6, llama.cpp F16 GGUF 2.4e-4, arg-max equal at every position; the GGUF completes "The Eiffel Tower is located in the city of" with " Paris, France. It is the tallest building in the world" on the CPU |
| RWKV/v6-Finch-1B6-HF (RWKV-6, 1.6B) | `rwkv6` converter and the llama.cpp F16 GGUF (pre-scaled weights, `rescale_every_n_layers` 6) | 12 positions against the reference: converted checkpoint 1.8e-6, llama.cpp F16 GGUF 4.2e-4, arg-max equal; the reference itself is a transcription of the official recurrence, not the model's own remote code (that needs flash-linear-attention kernels) |
| Qwen/Qwen3-VL-2B-Instruct, text only | `qwen3_vl_text` path | four of four continuations identical to transformers |
| Qwen3-VL-2B-Instruct Q8_0 GGUF of the Qwen organisation (architecture `qwen3vl`) | the new `qwen3vl` alias | loads and generates; two of four continuations identical to the float32 transformers text, the others differ after 20 and 52 characters (Q8_0 against float32) |

Header-only checks (no weights downloaded) of what real repositories contain: Step-3.5-Flash Hub files use `moe.*` / `share_expert.*` names and
stacked expert tensors, which the new reader handles (shapes and dtypes read from the safetensors headers); the AutoRound repositories
store GPTQ tensors with zero points stored minus one, AutoRound-AWQ stores `qzeros` of all 8.

llama.cpp GGUF architecture names, checked on the headers of real files (`scripts/gguf_peek.py`, range requests, nothing downloaded except the BitNet file above):

| GGUF architecture | State |
| --- | --- |
| `bitnet-b1.58` (BitNet fork, `I2_S`) | loads (alias of `bitnet`, neox rotary pairs, I2_S decoder); the real 2B file was run, see above |
| `mimo2` (MiMo-V2(.6)-Flash) | loads as `mimo_v2` (same metadata keys and tensor names; the sliding-window pattern as 0/1 integers is accepted); only the header of `ggml-org/MiMo-V2.6-Flash-RL-GGUF` was read, the 160 GB model was not run |
| `step35` (Step-3.5-Flash) | **refused with the reason**: the file clamps the SwiGLU of the experts on layers 43 and 44 (limit 7) and of the shared expert on layer 44 (limit 16), and has a `rope_freqs.weight` of 64 entries for the scaled partial rotary of the full-attention layers; neither exists in fLLM. Header of `ggml-org/Step-3.5-Flash-GGUF` read |
| `glm5-next` (GLM-5.3-Flash) | **refused with the reason**: multi-stream residuals (`hyper_connection.*`), the sparse indexer and the KDA / latent hybrid (header of `unsloth/GLM-5.3-Flash-GGUF` read) |
| `qwen3vl` | the text model loads (alias, earlier audit); the vision part of such files lives in a separate `clip` mmproj file (`qwen3vl_merger`, deepstack layers) that fLLM does not read |
| `deepseek2`, `kimi-linear` | same names as fLLM's; no such file was run |

## Runs on accelerators (this audit)

Only the greedy-completion smoke test `scripts/gpu_arch_smoke.py`, `selftest-gpu` and `FLLM_LAYER_CHECK=1` ran the new architectures on an
accelerator (tiny random models, CPU against the device, same text expected). Nothing else in this document claims a GPU result.

Device: an NVIDIA RTX 2000 Ada Laptop GPU through the native Vulkan backend (`--device vulkan:1`, driver 595.99.02); CPU: the same machine's
processor. Each row is one tiny random-weight model, 12 greedy tokens, the text from the device against the text from the CPU:

| Model | Result |
| --- | --- |
| BitNet, Helium, Ministral, DeepSeek-V3.2, GLM-5 (DSA, index_topk above the sequence length), Kimi-Linear, XVERSE | same text |
| MiMo-V2-Flash, Step-3.5, Laguna | **different text on the first run**: `FLLM_LAYER_CHECK=1` showed layer 0 of the device differing from the host reference by 1.7 to 2.5 times its rms (a stablelm, glm4, phi, olmoe, gemma2 control showed 0.001 to 0.003). Cause found by comparing intermediate sums: the device multiplied the attention output by a per-head gate element by element (Laguna, Step-3.5) and had no value scale (MiMo-V2). Those attention layers now run on the host (`layer_host_reason`); the same text as the CPU on the second run, together with BitNet |

This shows the architectures load and run on that device and found one defect of the shared device path; it is not a performance or
accuracy measurement.

### Device operations for the per-head gate, the value scale and the sub-norms (2026-10-30)

The host fallback was replaced by device operations (`head_gate_mul`, `scale_inplace`; the BitNet sub-norms use the ordinary device RMS
norm). Tiny random models (`laguna`, `step3p5`, `mimo_v2_flash`, `bitnet`; f32 and f16 GGUF files, 12 greedy tokens, `scripts/gpu_arch_smoke.py`):

| Device | Backend | Result |
| --- | --- | --- |
| Radeon 840M (RADV, gfx1153) | Vulkan, `vulkan:0` | 8 of 8 files give the CPU's text; `FLLM_LAYER_CHECK=1`: every layer of all four models equals the host reference (max/rms 0.0000; the layers run on the device, no "run on the host" line); `selftest-gpu --only arch`: 23 of 23 checks ok including the new `head_gate` (sigmoid and softplus, errors 1e-7 .. 2e-7 against an f64 reference, tolerance 2e-6) and `scale` cases |
| RTX 2000 Ada Laptop | Vulkan, `vulkan:1` (native) | 8 of 8 files give the CPU's text; `selftest-gpu --only arch`: 23 of 23 ok, `head_gate` errors 9e-8 .. 3.6e-7, `scale` 3e-8 |
| Radeon 840M | HIP (ROCm 7.1, gfx1153) | prompt "the cat" (3 tokens), 8 of 8 files give the CPU's text; `selftest-gpu --only arch`: 16 of 16 ok, `head_gate` and `scale` errors 9e-8 .. 1.9e-7 / 3e-8. **The MiMo-V2 attention layers stay on the host on AMD GPUs** (`supports_value_scale`; `FLLM_HIP_VALUE_SCALE=1` lifts it): with them on the device the text differed and `FLLM_LAYER_CHECK=1` showed layer 0 differing in decode steps (max/rms 0.04 .. 0.2) while prefill agreed; the text also differed with the value scale patched out of the file (1.0), so the cause is the HIP attention path for this shape (per-layer KV head counts, sinks, value width half of the key width), not the new operations; not investigated further |
| RTX 2000 Ada Laptop (sm_89, built on the laptop with `FLLM_CUDA_ARCHS=89`) | CUDA | prompt "the cat": f32 files of all four models and `bitnet_f16` give the CPU's text; `selftest-gpu --only arch`: 26 of 26 ok, `head_gate` errors 9e-8 .. 1.3e-7, `scale` 3e-8. The f16 files of the other three models do not run on CUDA (`Unsupported quantization format` or no response; not looked into) |

**Prompts of 16 or more tokens failed on CUDA and HIP with an empty `Driver error:` for these tiny models** (`laguna`, `step3p5`, `bitnet`, and a plain tiny Qwen3 fixture that uses none of the new operations). Cause (found, fixed): attention for a prompt of 16 or more rows of one sequence, inside a forward pass, tried the tensor-core kernel (CUDA only), then the two decode kernels, and returned the last one's "not handled" code as an error when none of them covers the head width (16 here; 32, 80, 96 and others as well, the decode kernels take 64, 128 and 256). The general attention kernel, which handled these shapes before, is the last step again. The three-token prompt of the table above was chosen to avoid it. After the fix, on an RTX 2000 Ada Laptop (CUDA) and a Radeon 840M (HIP), the tiny `make-tiny-model` Llama and f32 GGUF files of `laguna`, `step3p5`, `bitnet` and a Qwen3 fixture (`scripts/hf_golden.py`) give the CPU's text at prompts of 3, 8, 16, 40 and 70 characters (the fixtures' tokenizers merge characters, so the token counts are lower); the same on Vulkan (`vulkan:1`, RTX 2000 Ada) for the CUDA laptop. New `selftest-gpu` attention cases (fast pass, head widths 16, 32 and 80, runs of 16 to 24 rows, inside and outside a forward pass) failed with the old code on both devices and pass now.

Metal, SYCL, wgpu, the NPU and the TPU backend do not
report `supports_head_gate` and keep the host fallback.

### Sparse-attention indexer (DeepSeek-V3.2, GLM-5): comparison and devices

`scripts/hf_golden.py` has the fixtures `deepseek_v32_sparse` and `glm_moe_dsa_sparse` (`index_topk` 4, so from the fifth token on the
indexer drops keys; the older fixtures with `index_topk` 64 never drop one). Reference: transformers 5.22.1 on the build host, float32.
The indexer score of a key is `sum_h w_h * relu(q_h . k)`; keys whose heads are all in the ReLU dead zone score exactly 0 and
transformers' `topk` picks among such ties in an undefined order, so the fixtures use a stable `topk` (lower position first, as fLLM
does) and the builder chooses a seed in which the last kept and the first dropped score differ by more than 5e-4.

| Check | Result |
| --- | --- |
| Logits of every position, `scripts/verify_architectures.py`, f32 weights, f16 KV cache | `deepseek_v32_sparse` 1.7e-4 (12 tokens), 2.5e-4 (28), 1.8e-4 (33 and 36); `glm_moe_dsa_sparse` 3.5e-4 (12), 3.0e-4 (28, 33, 36), 2.2e-4 (40, 45); the selected positions of all 109 (layer, position) pairs equal those of transformers |
| The same fixtures with `FLLM_DSA=dense` (the earlier behaviour) | 0.57 / 0.64 relative: the test discriminates |
| `deepseek_v32_sparse` at 40 and 45 tokens | 0.13 relative at position 37 only: the attention output of that position agrees (below 1e-3) and the MoE output of layer 2 does not, a routing near-tie of the random fixture that is not related to the indexer; not counted as a pass |
| Cached decoding: first K tokens as a batch, then one token per call (`FLLM_VERIFY_SPLIT`) | same logits as the single batch for both fixtures |
| Greedy server completions, 4 concurrent requests, `scripts/verify_generation.py` | 4 of 4 identical to transformers for both fixtures |

Devices (RTX 2000 Ada Laptop GPU through Vulkan, `vulkan:1`; tiny fixtures; `scripts/gpu_arch_smoke.py`): the default device run keeps
the dense device attention and gives a different text from the CPU (the indexer cuts 4 of 19 tokens); with `FLLM_DSA=exact` the text equals the
CPU text for both fixtures, and with `FLLM_DSA=dense` the device text equals the dense CPU text. CUDA, HIP, Metal and Vulkan on AMD were not run.

## Open gaps

Not implemented, ordered by how many current models need them:

1. **Multi-stream residuals ("hyper-connections", mHC/iHC)** on accelerators, and for Qwen3.8-Flash-Next and
   Motif at all. DeepSeek-V4, Hy4 and GLM-5-Next have them on the CPU (see the matrix); the device kernels for the stream mixing, the compressors and the
   per-sequence state are missing, as is its MTP head and the FP8 / FP4 Hub weight formats.
2. **A device kernel for the DeepSeek-V3.2 / GLM-5 indexer.** The CPU path is exact at any context length (indexer keys per
   slot beside the KV pools); accelerators keep dense attention, exact up to `index_topk` tokens, or run those layers on the
   host (`FLLM_DSA=exact`). A real 685B / 753B checkpoint was not run.
3. **KDA variants**: Ling-3.0 (bounded forget gate, rotary latent attention with a head gate, Hub layout `attention.*`),
   GLM-5-Next, Kimi K3 (attention residual blocks, `situ` activation, latent MoE, MXFP4 compressed-tensors).
4. **xLSTM, RecurrentGemma, HRM-Text, ZAYA (CCA attention), JetMoE (mixture of attention heads), Doge,
3. **KDA variants**: GLM-5-Next, Kimi K3 (attention residual blocks, `situ` activation, latent MoE, MXFP4 compressed-tensors).
3. **KDA variants**: Kimi K3 (attention residual blocks, `situ` activation, latent MoE, MXFP4 compressed-tensors).
4. **RWKV 6/7, xLSTM, RecurrentGemma, HRM-Text, ZAYA (CCA attention), JetMoE (mixture of attention heads), Doge,
   DiffLlama**: their own recurrent or attention mechanisms.
5. **LongCat-Flash-Lite** (n-gram embeddings), **MiniMax-M1**
   (lightning attention).
6. Vision towers beyond Qwen-VL; any audio model.
7. The quantization formats listed as "no" above (HQQ, torchao, Quark, EETQ, QuIP#, SpQR, VPTQ, HIGGS, fp_quant, BitBLAS / IPEX repacks): each has a clear converter error.

A 2.8T or 1T model cannot run on a laptop whatever the architecture support is; "T" means the layer maths is right on a
tiny random model, not that a real checkpoint of that size was run.

## Snapshot against artificialanalysis.ai (2026-10-13, statuses updated 2026-10-30)

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
| V verified (architecture, tiny checkpoint vs `transformers` or the model's remote code) | 28 |
| U supported, but not fully verified or with a documented gap | 2 (LFM2.5, K2 Horizon 375B) |
| N not supported | 13 |

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
| Gemma 4 26B A4B, Gemma 4 31B (small) | 26B / 31B | `Gemma4ForConditionalGeneration` | `gemma4` | V | Supported and verified against transformers 5.19: heterogeneous sliding (256 x 8) and global (512 x 2) layers, K = V, proportional rope, dense + MoE block, per-layer embeddings and shared KV layers (E2B/E4B). Layered KV cache layout across CPU and accelerator traits. |
| MiniCPM5 2B / 1B, G9v3-3B, Nanbeige4.1-3B (tiny) | dense | `LlamaForCausalLM` | `llama` | V | Tokenizer `minicpm5` exists. |
| Granite 4.2 3B (tiny) | dense | `GraniteForCausalLM` | `granite` | V | |
| Phi-4 Mini (tiny) | 3.8B | `Phi3ForCausalLM` | `phi3` | V | LongRoPE. |
| Qwen3.5 2B / 0.8B (tiny) | | `Qwen3_5ForConditionalGeneration` | `qwen35` | V +GGUF | 0.8B GGUF was compared earlier. |
| LFM2.5 2.6B (tiny) | | `Lfm2ForCausalLM` | `lfm2` | U | Implemented; reference comparison still pending, CPU only. |

## Medium (25-130B)

| Model | Params | HF architecture | Status | Notes |
| --- | --- | --- | --- | --- |
| Ling-3.0-flash(-VL/-Fin) | 124B / 5.5B | `BailingMoeV3ForCausalLM` (`bailing_hybrid`) | **N** | MLA mixed with KDA layers like Kimi-Linear (which is supported and verified), but with a bounded forget gate, direct (not low-rank) gate projections, rotary latent attention with a head-wise output gate and the Hub tensor names `attention.*`. |
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
| MiMo-V2.6-Pro / -Flash | 1.0T / 309B | `MiMoV2ForCausalLM` | V (tiny) | The `mimo_v2_flash` layer is supported (different KV heads for sliding and global layers, sinks, value scale, partial rope) and matches transformers on a tiny checkpoint; the real checkpoints (FP8, 309B and 1.0T) were not run. Pro-specific differences, if any, are unchecked. |
| GLM-5.3 (Max / Low) | 753B | `GlmMoeDsaForCausalLM` | V (tiny, dense attention) | MLA + MoE as before; the indexer tensors are dropped and attention is dense, which is exact up to 2048 tokens of context (see Open gaps). |
| GLM-5.3-Flash | 320B | `Glm5NextForConditionalGeneration` | V (tiny, CPU) | KDA linear attention + k-pool DSA + multi-stream residuals (`mhc`) as in `glm5_next`; text part only, checkpoint not run. |
| Kimi K3 | 2.8T | `KimiK3ForConditionalGeneration` | **N** | KDA layers, attention residual blocks, `situ` activation, latent MoE, MXFP4 compressed-tensors. |
| Qwen3.8 2.4T A95B | 2.4T | `Qwen3_5MoeForCausalLM` | V (arch) | Same layers as Qwen3.5-MoE; too large to run. |

| Qwen3.8-Flash-Next | 180B / 6B | `Qwen4ExpForConditionalGeneration` | **N** | Hyper-connections, n-gram embeddings, per-layer embeddings, sparse indexer. |
| DeepSeek V4.1 Flash, V4 Pro 0813 | 552B, 1.6T | `DeepseekV41ForCausalLM`, `DeepseekV4ForCausalLM` | V (tiny, CPU) | Hyper-connections, compressed/sparse attention, hash routing as in `deepseek_v4`; the V4.1 engram embeddings are not implemented. Checkpoints not run. |
| Motif 3 | 314B | `MotifForCausalLM` | **N** | GDLA attention, multi-stream residuals, PolyNorm activation. |
| MiniMax-M3 | 428B | `MiniMaxM3SparseForConditionalGeneration` | V (tiny, CPU) | Block-sparse attention index on top of MiniMax-M2, `swigluoai`, as in `minimax_m3_vl_text`; text part only, checkpoint not run. |
| Inkling | 975B | `InklingForConditionalGeneration` | V (tiny, CPU) | Short convs, relative position tables, log scaling, shared expert sink as in `inkling_text`; text part only, checkpoint not run. |
| Nemotron 3 Ultra | 550B / 55B | `NemotronHForCausalLM` | V (tiny, new) | Latent MoE, same code as Super. MTP head is not imported. |
| K2 Horizon 375B A23B | 375B / 23B | `K2HorizonForCausalLM` | U | MoE path verified (tiny). The checkpoint uses `rope_head_dim: 64` < `head_dim: 128` with an interleaved partial layout; the converter rejects it instead of guessing. |

## Priority of the gaps

1. Ling-3.0-flash (124B, 5 GB active at 4-bit with CPU offload) - KDA layers, which are also needed by Kimi-Linear/K3 and GLM-5.3-Flash.
2. LongCat Flash Lite, Qwen3.8-Flash-Next (n-gram embeddings, hyper-connections): small active size but exotic layers.
3. The 300B+ models (MiMo, GLM, DeepSeek V4, Kimi K3, MiniMax-M3, Inkling, Motif): not usable on a home machine, and each
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
