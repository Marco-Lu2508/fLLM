# Wave 2 Architecture Analysis (Part C)

Technical analysis of complex and upcoming architectures for fLLM, evaluating architectural requirements, complexity, and available reference implementations in `transformers` 5.19.0.

---

## 1. KimiLinear (`KimiLinearForCausalLM`)

* **Reference Implementation:** Available in `transformers` 5.19 (`transformers.models.kimi_linear.modeling_kimi_linear.KimiLinearForCausalLM`, `KimiLinearConfig`).
* **Core Mechanisms:**
  * Uses **Kimi Delta Attention (KDA)**, an input-dependent linear attention mechanism with gated delta-rule recurrent memory updates.
  * Employs short convolutions over keys, values, and beta gates before the recurrent update step.
  * Uses grouped query/key/value projections and learned channel-wise decay factors.
* **Technical Gap in fLLM:**
  * While `fllm` has initial KDA geometry parsing in `checkpoint/src/kimi_linear.rs` and `Attn::Kda` stubs in `rust_decoder.rs`, it lacks the high-performance recurrent update and chunked prefill kernels on CUDA and Vulkan.
  * Memory footprint: recurrent state of size `[batch, heads, d_k, d_v]` must be managed persistently in device VRAM per sequence.
* **Effort & Feasibility:** Medium to Large (M-L). Requires dedicated CUDA/Vulkan KDA scan kernels and chunked prefill algorithms.

---

## 2. GlmMoeDsa / DeepseekV32 (`GlmMoeDsaForCausalLM` / `DeepseekV32ForCausalLM`)

* **Reference Implementation:** Available in `transformers` 5.19 (`transformers.models.glm_moe_dsa`, `transformers.models.deepseek_v32`).
* **Core Mechanisms:**
  * **Dynamic Sparse Attention (DSA)**: Extends Multi-Head Latent Attention (MLA) with a learned sparse indexer module (`indexer_heads`, `indexer_dim`).
  * Before computing dense attention over the full context, an indexer projection selects the top-$k$ most relevant token indices per head.
  * Dense multi-head attention is then computed exclusively over the dynamically selected sparse token set.
* **Technical Gap in fLLM:**
  * Requires a new indexer pipeline before the paged attention kernel.
  * Paged attention kernels on CUDA/Vulkan currently assume contiguous block tables or contiguous sequence tokens; evaluating attention over non-contiguous top-$k$ sparse token sets requires a specialized indirect gathering attention kernel or indexer pre-filter.
* **Effort & Feasibility:** Large (L). Highly complex kernel development and custom sparse paged attention required.

---

## 3. Step3p5 / Step3p7 (`Step3p7ForConditionalGeneration` / `Step3p7TextModel`)

* **Reference Implementation:** Step-3.7 text backbone is present in `transformers` 5.19 (`transformers.models.step3p7.modeling_step3p7.Step3p7TextModel`).
* **Core Mechanisms:**
  * Heterogeneous per-layer KV dimensions (alternating sliding-window local layers and global attention layers with differing head counts and head dimensions).
  * Fine-grained Mixture-of-Experts with 64+ routed experts and shared experts, with router normalization and top-$k$ gating.
* **Technical Gap in fLLM:**
  * Requires generalized per-layer KV cache layouts in the device cache allocator (similar to Gemma 4, but with varying head dimensions across layers).
  * Checkpoint conversion must map dynamic expert counts and per-layer rotary configurations.
* **Effort & Feasibility:** Medium (M). Feasible once per-layer heterogeneous KV cache pooling is generalized across all backends.

---

## 4. MiMoV2 (`MiMoV2FlashForCausalLM`)

* **Reference Implementation:** Present in `transformers` 5.19 (`transformers.models.mimo_v2_flash`).
* **Core Mechanisms:**
  * Hybrid architecture combining sliding-window GQA, compressed attention, and Multi-Token Prediction (MTP) draft blocks.
  * MTP layers predict $k$ future tokens concurrently during training and speculative draft generation.
* **Technical Gap in fLLM:**
  * For primary causal decoding, the backbone behaves like a sliding-window GQA model (which fLLM supports).
  * The MTP draft heads require integration with `rust/engine/src/draft.rs` to support multi-token speculative verification in the server.
* **Effort & Feasibility:** Small to Medium (S-M) for the backbone; Large (L) for full MTP speculative decoding.

---

## 5. LongcatFlash (`LongcatFlashForCausalLM`)

* **Reference Implementation:** Present in `transformers` 5.19 (`transformers.models.longcat_flash`).
* **Core Mechanisms:**
  * Engineered for ultra-long context processing (up to 512k-1M tokens).
  * Employs dual-path attention: local sliding window combined with flash-compressed landmark tokens and n-gram embedding projections.
* **Technical Gap in fLLM:**
  * Requires landmark token pooling and dual KV cache buffers (one for local high-resolution tokens, one for compressed landmark tokens).
* **Effort & Feasibility:** Large (L). Requires architectural changes to the sequence context pipeline.

---

## 6. Grok-1 (`Grok1ForCausalLM`)

* **Reference Implementation:** Open-source 314B MoE released by xAI; reference implementations available in JAX and vLLM (`vllm.model_executor.models.grok1`).
* **Core Mechanisms:**
  * Standard dense transformer backbone with 64 layers, 48 attention heads, 8 KV heads (GQA), and 8 routed experts per layer (top-2 active, 86B active parameters).
  * Uses standard RMSNorm, Rotary Position Embeddings (RoPE), and GeLU activation.
* **Technical Gap in fLLM:**
  * Architecturally, Grok-1 does not introduce novel mathematical operators: it is a scaled-up Mixtral variant.
  * The primary challenge is sheer scale (314B parameters, ~628 GB in FP16, ~160 GB in 4-bit).
  * Running Grok-1 requires multi-GPU tensor parallelism (`rust/engine/src/parallel/shard.rs`) and sharded checkpoint loading across multiple devices.
* **Effort & Feasibility:** Medium (M) for architecture support in converter/engine; Large (L) for multi-GPU memory management and testing.
