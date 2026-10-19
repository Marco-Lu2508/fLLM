# Supported architectures

fLLM reads llama.cpp-style GGUF files and converts Hugging Face checkpoints into them
(`fllm-rs convert`). Everything marked as verified below was compared with `transformers` on tiny
random checkpoints (`scripts/verify_architectures.py`): every position's logits must agree to a
relative difference below 2e-3 and the arg-max must not differ beyond the numeric error. Hybrid and
state-space models are additionally checked through the HTTP server with concurrent requests
(`scripts/verify_generation.py`). Real checkpoints of most families were **not** available for
testing; the tensor names follow what `transformers` writes, which can differ from a model's hub
files for a few families, so report mismatches.

Multimodal checkpoints (`text_config` + `language_model.*` / `model.language_model.*` tensors)
convert to their text model; vision towers are not imported.

## Dense decoders

| `model_type` | Notes |
| --- | --- |
| `llama`, `mistral`, `mistral_large`, `granite`, `k2_horizon`, `intellect_3` | GQA, sliding window, linear/YaRN/Llama-3 rope scaling |
| `baichuan`, `qwen`, `qwen2`, `qwen3` (+ `*_vl_text`) | Q/K bias, per-head Q/K norm, fused QKV (baichuan, qwen) |
| `gemma`, `gemma2`, `gemma3_text`, `vaultgemma` | `1+w` norms, soft-capping, local/global rope, sandwich norms (Gemma 2/3) |
| `gemma4`, `gemma4_text`, `gemma3n` | sliding and global layers with heterogeneous head width, K = V on global layers, proportional rope, value norm, AltUp, LAuReL, per-layer embeddings, shared KV layers, layer output scale; verified against transformers 5.19 |
| `muse_glimmer` (text model of Muse Glimmer) | Gemma-style sandwich norms with a silu MLP, per-head sigmoid output gate (`attn_gate`), weightless QK norm (queries scaled by `qk_scale_factor`), NoPE global layers, embedding RMS norm, `output_multiplier` before the tanh soft cap, separate epsilon for the post norms. llama.cpp GGUFs use interleaved rope with permuted Q/K rows; checked against an unsloth GGUF and llama.cpp. Vision tower not imported |
| `olmo`, `olmo2`, `olmo3` | non-parametric LN (v1), post norms, sliding layers with own rope |
| `phi`, `phi3` | parallel block + biases (phi-2), LongRoPE (phi-3) |
| `cohere`, `cohere2`, `cohere2_moe` | bias-free LayerNorm, parallel residual, logit scale, sliding/NoPE layers |
| `starcoder2`, `gpt_neox`, `stablelm`, `gptj`, `gpt2`, `gpt_bigcode`, `gpt_neo`, `opt` | LayerNorm families, learned positions, parallel residual |
| `falcon`, `bloom`, `mpt` | multi-query/GQA fused qkv layouts, ALiBi (Falcon-RW's reference rounds the bias to bfloat16, so only the maths is implemented) |
| `glm`, `glm4`, `glm4v_text` | partial interleaved rope, fused gate/up, sandwich norms |
| `nemotron` | LayerNorm1p (`weight + 1`, folded at conversion), ReLU² MLP without gate, partial NeoX rope (Nemotron-4-Minitron, Nemotron-Mini-4B; also compared with the real Nemotron-Mini-4B weights) |
| `internlm2` | InternLM2/2.5: fused `wqkv` (rows grouped per KV head) reordered into q/k/v with interleaved rotary pairs, `w1/w2/w3` MLP; dynamic NTK scaling is ignored within `max_position_embeddings`; verified against the model remote code |
| `seed_oss`, `smollm3`, `arcee`, `ernie4_5`, `exaone4`, `hunyuan_v1_dense`, `ministral3`, `jais2`, `cwm`, `hyperclovax`, `nanochat`, `apertus` | per-layer NoPE, post-rope QK norm, query temperature, ReLU², xIELU, weightless norms, flipped rope |
| `minicpm` | muP scaling: `scale_emb` on embeddings, `residual_scale` (`scale_depth / sqrt(layers)`), `logit_scale` (`dim_model_base / hidden`); verified against transformers 5.19 (4.3e-5) |
| `internlm3` | Llama tensor layout; dynamic NTK rope is the identity within `max_position_embeddings` (as for `internlm2`); checked against transformers 5.19 (relative logit difference 1.4e-4) |
| `mimo` | Qwen2 layout with Q/K/V biases; `model.mtp_layers.*` (NextN heads) are dropped on import; checked against transformers 5.19 (1.9e-4) |
| `solar` (Solar-Pro) | Llama layout; `bskcn_*` config fields carry no tensors and are ignored; a `sliding_window` key is present but attention is dense, so no window pattern is written; checked against transformers 5.19 with that key present (1.7e-4) |
| `orion` | Llama layout with MHA and biased LayerNorms (input/post-attention/final); the norm biases are mapped generically; checked against transformers 5.19 (1.2e-4) |
| `iquestcoder` | Plain Llama tensor layout (GQA); checked against transformers 5.19 (1.3e-4) |
| `xverse` | Llama layout (MHA, no `num_key_value_heads`, no `rope_theta` -> default 10000); the persistent `rotary_emb.inv_freq` buffer is dropped; checked against transformers 5.19 (1.8e-4) |
| `persimmon` | MHA with biased LayerNorm instead of RMSNorm; fused `query_key_value` is head-interleaved (`[q_h, k_h, v_h]` per head) and split in the loader; per-head Q/K LayerNorm *with* bias applied before rope; gapless ReLU² MLP (`dense_h_to_4h`/`dense_4h_to_h`); partial rope 0.5 (only the first `head_dim/2` dims, Llama `rotate_half`); checked against transformers 5.19 (2.1e-4) |
| `exaone` (EXAONE 3.x) | Llama layout with `transformer.h.N` prefix, `attn.attention` / `mlp.c_fc_0/1/c_proj` tensor names, Llama-3 RoPE scaling; canonicalized to `llama` in loader; verified against remote code via EXAONE_DIR (9.4e-5) |

## Mixture of experts

| `model_type` | Router |
| --- | --- |
| `mixtral`, `qwen2_moe`, `qwen3_moe`, `mellum`, `sarvam_moe`, `olmoe`, `flex_olmo`, `granitemoe(shared)`, `xverse_moe` | softmax, top-k, optional renormalisation, shared experts (sigmoid-gated for Qwen2-MoE), sliding/NoPE layers for Mellum |
| `deepseek_v2`, `deepseek_v3`, `kimi_k2`, `glm4_moe_lite`, `bailing_moe`, `sarvam_mla`, `mistral4`, `youtu`, `minicpm3` | multi-head latent attention (the cache holds the rebuilt per-head keys/values), sigmoid or softmax routing, selection bias, group-limited experts, shared experts |
| `glm4_moe`, `dots1`, `exaone_moe`, `solar_open`, `hy_v3`, `afmoe`, `trinity`, `minimax_m2`, `ernie4_5_moe`, `hunyuan_v1_moe`, `laguna` | DeepSeek-style sigmoid routing with bias, softplus attention gate, per-layer head counts, and dense/sparse layer patterns |
| `dbrx` | flat experts with transposed down weights, clipped fused QKV, bias-free LayerNorm, selected-weight p-norm; CPU implementation; logits agree with transformers 5.19 on a tiny checkpoint (relative difference 1.3e-4) |
| `gpt_oss` | attention sinks, clamped SwiGLU, biased experts, MXFP4 import, untruncated YaRN |
| `llama4` | chunked attention, NoPE layers, router score scales the expert input |
| `phimoe` | LongRoPE, separate Q/K/V/O, fused expert gate/up, SparseMixer thresholded routing |
| `k2_horizon` (MoE and MoVA variants) | sigmoid routing with selection bias, shared expert, leading dense layers (`mlp_only_layers` prefix), grouped RMSNorm, softplus attention gate and, for MoVA, routed value experts (`attn_v_exps`, `attn_v_gate`); verified against the model's remote code on tiny checkpoints. Not supported: `query_key_norm`, `rope_head_dim` < `head_dim` (375B A23B) |
| `g9v3` | DeepSeek-V3-style sigmoid MoE with shared expert and per-head sigmoid-gated attention (`q_proj` holds `[q; gate]` per head, converted to llama.cpp's separate-gate layout); verified against the model's remote code on a tiny checkpoint |
| `gemma4` (26B-A4B) | dense MLP and routed expert block side by side, normalized router with top-k experts; verified against transformers 5.19 |
| `deepseek` | DeepSeek-MoE v1 (16B): MHA + MoE with shared experts and first layer dense (`first_k_dense_replace`), softmax routing; verified against remote code via DSPKG_DIR (2.7e-4) |
| `chatglm` | ChatGLM2/3, GLM-4-9B-chat: fused QKV (multi-query), 2D/partial RoPE (adjacent pairs), fused SwiGLU dense_h_to_4h, RMSNorm; verified against remote code via CHATGLMPKG_DIR (3.2e-4) |
| `falcon-h1` | Falcon-H1: concurrent Mamba-2 and attention branches in parallel in every layer, mu-P multipliers folded into weights; verified against transformers 5.19 (1.8e-4) |
| `jamba` | Jamba: interleaved Mamba-1 with inner dt/B/C RMSNorms and attention, MoE with per-layer allocation; verified against transformers 5.19 (1.3e-4) |
| `zamba2` | Zamba2: Mamba-2 backbone with shared transformer blocks (per-call LoRA adapters, `[x, embeddings]` input); verified against transformers 5.19 (1.6e-3) |


DBRX currently uses the CPU decoder (`auto` selects CPU for this architecture); an explicit
accelerator request is rejected until its LayerNorm, clipping and router normalization are
implemented in the accelerator path. Its tiny reference fixture includes the Transformers 5.19
`attn_config.rope_theta` compatibility workaround. The tiny fixture agrees with Transformers 5.19 (relative logit difference 1.3e-4, no arg-max mismatch).

Dense LFM2 runs on the CUDA backend (the ShortConv has CUDA kernels; everything else is attention and a dense FFN) and
selects the GPU under `auto` like other decoders; LFM2-MoE is CPU-only (see below). The importer applies the reference FFN width adjustment
(including truncation, multiplier and rounding to `block_multiple_of`) and maps the final
`embedding_norm` to the output RMS norm. ShortConv caches `L_cache - 1` products `B*x` per channel;
the current token participates in the last convolution tap, and `C` gates the result without
an additional activation. Sequence restart and position mismatch use the existing recurrent
state rules. Cached and batched numerical comparisons remain pending.

LFM2-MoE selects CPU under `auto` and rejects explicit accelerators until its expert routing is integrated there.
It uses the same recurrent state rules, preserves the dense FFN width, and imports the
stacked expert gate/up and down matrices.
Its router normalizes selected sigmoid scores with `sum + 1e-6`, then applies the routing scale;
the bias affects selection only. Unlike LFM2, the installed LFM2-MoE configuration does not infer
`layer_types`, so import requires an explicit list. The fixture includes nonzero selection bias.

## Recurrent and hybrid models

These keep per-sequence state outside the KV cache (`DeltaNet`, Mamba, LFM2 ShortConv), so prefix sharing and
speculative decoding are switched off for them.

| `model_type` | Layers |
| --- | --- |
| `lfm2_moe` | ShortConv/attention with leading dense FFNs, sigmoid top-k routing, selection bias and epsilon normalization; CPU only; tiny fixture agrees with transformers 5.19 (2.9e-4) |
| `lfm2` | gated ShortConv with per-sequence history + Q/K-normalized attention; CPU and CUDA; tiny fixture agrees with transformers 5.19 (1.8e-4) |
| `mamba`, `falcon_mamba`, `mamba2` | selective state-space mixers |
| `qwen3_next`, `qwen3_5`, `qwen3_5_moe` | gated DeltaNet + gated attention (+ MoE) |
| GGUF `qwen35`, `qwen35moe` (llama.cpp / unsloth files) | same layers; llama.cpp stores each head's query and gate side by side, the value heads of the DeltaNet layers tiled (value head `j` belongs to key head `j % key heads`) and the MoE experts as one `ffn_gate_up_exps` tensor with `[gate; up]` rows per expert. Loaded as such (checked against fLLM's own conversion of Qwen3.5-0.8B: identical text; Qwen3.6-35B-A3B Q4_K_S runs). The DeltaNet recurrence runs on the device with the CUDA backend (`deltanet.cu`); on the other backends it runs on the CPU between device layers. |
| `olmo_hybrid` | gated DeltaNet with separate projections + attention |
| `granitemoehybrid`, `bamba`, `nemotron_h` | Mamba-2 + attention (+ MLP / experts). Nemotron-H with `moe_latent_size` (Nemotron 3 Super / Ultra) is supported: router on the hidden state, routed experts on a latent projection (`ffn_latent_down` / `ffn_latent_up`), shared expert on the hidden state. Expert variants are written as GGUF architecture `nemotron_h_moe` like llama.cpp; both names load, per-layer `head_count_kv` / `feed_forward_length` arrays are read, attention has no rope |

## Not supported

* MiMo-V2 (Flash and Pro), Kimi-Linear (KDA), DeepSeek V3.2/V4 and GLM-5
  (sparse attention indexer), JetMoE, Inkling, ZAYA.
* LongCat Flash: dual-sublayer self-attention and MLP topology with shortcut MoE routing.
* Step 3.5 / Step 3.7: heterogeneous per-layer attention head counts (`num_attention_heads_per_layer`).
* A.X-K2: Gated RMSNorm wrapper (`AXK2GatedRMSNorm`) with internal `gate_mlp` projection.
* Encoder families beyond the BERT GGUF layout (ModernBERT and NomicBERT are supported (NomicBERT verified 2026-10-14 against transformers 5.19, fixture + nomic-embed-text-v1.5); BGE-M3 dense verified via the XLM-R path; still open: GTE-new, Jina, SPLADE, ColBERT),
  and classification heads outside BERT/RoBERTa/XLM-R.
* Vision, audio and speech towers of multimodal checkpoints.

## Quantization

`fllm-rs quantize IN.gguf OUT.gguf TYPE` (and `convert --outtype TYPE`) writes `q4_0`, `q4_1`, `q5_0`,
`q5_1`, `q8_0`, `q4_k_s`, `q4_k_m`, `q5_k_s`, `q5_k_m` and `q6_k` with the same per-tensor mixes as
llama.cpp. `q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0` are byte-identical to `llama-quantize`; the K-quants differ
only in rare float-summation ties. `scripts/compare_gguf.py` compares two files tensor by tensor.
Reading supports every ggml type listed by `fllm-rs formats`.

## Tokenizers

Byte-level BPE models use the pre-tokenizer matching `tokenizer.ggml.pre` (GPT-2, Llama-3, Qwen2,
GPT-4o, DeepSeek-V3, K2); other values fall back to the GPT-2 split. Check a new model with
`scripts/verify_tokenizer.py`.

## Verifying a model

```sh
python scripts/verify_architectures.py --fllm target/release/fllm-rs \
    --example target/release/examples/verify_checkpoint llama qwen3 gemma3 deepseek exaone deepseek_v3
python scripts/verify_generation.py --fllm target/release/fllm-rs qwen3_5 mamba2
python scripts/verify_tokenizer.py MODEL.gguf HF_TOKENIZER_DIR target/release/examples/tokenize
```


## Encoders and sequence classifiers

| HF family | Import and runtime |
| --- | --- |
| BERT | Bidirectional embeddings; sequence classifier with CLS pooler, tanh and linear head |
| RoBERTa, XLM-RoBERTa | Bidirectional embeddings with non-padding position offset; CLS dense/tanh/out_proj classifier |
| DistilBERT, ELECTRA | Existing encoder embeddings; sequence classification import is unsupported |
| BGE cross-encoders | BERT or XLM-R sequence-classification checkpoints with compatible standard tensor names |
| ModernBERT | Fused QKV/GeGLU, dual-theta RoPE, alternating local (128) / global attention, bias-free pre-norms plus final norm; CLS/mean/last pooling, optional Matryoshka truncation |

All use the `bert` GGUF architecture plus `bert.model_type`. Classifier imports preserve
`classifier.*`, the BERT `pooler.*` or RoBERTa `classifier_dense.*`, and
`bert.classifier.label_count`, `kind`, `labels`. Missing or wrongly shaped required
head tensors fail conversion/loading with the tensor name. Embedding output stays
normalized; classification uses unnormalized hidden states. A one-label head receives
sigmoid postprocessing, multi-class heads softmax, with no inferred calibration.
The encoder has no generation, prefix cache or speculative path.

Rerank/score/classify/pooling routes are described in [server-api.md](server-api.md).
Fixtures `bert_classifier_1`, `bert_classifier_3`, `xlm_roberta_classifier_1` and `modernbert` are
available in `scripts/hf_golden.py`; the external checker `verify_encoder.py` reads dimensions,
layer count, positions, pooling and normalization from the GGUF metadata and compares
embeddings against a matching PyTorch reference (max abs error < 3e-5, cosine > 0.99999).
ModernBERT agrees with transformers 5.19 on the `modernbert` fixture (max abs error ~3e-8,
cosine 1.0 for cls/mean/last) and on the real `answerdotai/ModernBERT-base` checkpoint
(cosine 0.9999995-1.0 on three sentences; per-component relative error is inflated at
near-zero components, the fixture covers architecture fidelity with rel. error 8e-6 < 1e-3).
Decoder embedding pooling (mean/last + L2, e.g. Qwen3-Embedding) is verified against
transformers on a tiny Qwen3 fixture (cosine 1.0).

See [model-coverage.md](model-coverage.md) for the comparison with the open models listed on artificialanalysis.ai and
[model-coverage-vllm.md](model-coverage-vllm.md) for the checklist against the architectures of vLLM and SGLang.
