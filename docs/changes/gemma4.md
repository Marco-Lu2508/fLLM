# Per-layer attention geometry, Gemma 4, Gemma 3n and related models

## (a) Features

**Per-layer geometry.** `ModelConfig::layer_geoms` (rust/engine/src/config.rs) holds, for models whose layers differ, the
query heads, KV heads, key width, value width, feed-forward width and the KV-sharing source of every layer. Models with
one geometry keep `layer_geoms = None` and the old code paths and pool layout. Accessors: `layer_heads`, `layer_kv_heads`,
`layer_head_dim`, `layer_value_dim`, `layer_ff`, `layer_kv_source`, `layer_rope_dims`. Rotary tables: the sliding-window
table can have its own rotating width (`RopeConfig::local_dims`), and Granite SWA models list a rotary base per layer
(`ModelConfig::layer_rope_bases`, `GgufModel::rope_index`).

**KV pools.** `rust/engine/src/kv_geometry.rs` lays out one region per layer in the key pool and in the value pool
(`KvPoolLayout`); a layer that reads the cache of an earlier layer owns an empty region. `GgufModel::key_buffer_len`,
`value_buffer_len`, the host decoder, the prompt-block copy of the runtime and the memory planner
(`backend/memory.rs`, bytes per token over all layers) use it. All layers share one block table (see limits).

**GGUF keys** (llama.cpp names; fLLM additions marked *new*):

| Key | Meaning |
| --- | --- |
| `{arch}.attention.head_count`, `.head_count_kv`, `.feed_forward_length` | scalar or one entry per layer |
| `{arch}.attention.key_length`, `.value_length` | width of the global layers |
| `{arch}.attention.key_length_swa`, `.value_length_swa` | width of the sliding-window layers |
| `{arch}.attention.sliding_window`, `.sliding_window_pattern` | window and one bool per layer |
| `{arch}.attention.shared_kv_layers` | the last n layers read the cache of the last earlier layer of their kind |
| `{arch}.rope.dimension_count`, `.dimension_count_swa`, `.freq_base`, `.freq_base_swa` | rotating width and base per layer kind |
| `{arch}.embedding_length_per_layer_input` | width of the per-layer embedding (Gemma 4 E2B/E4B, Gemma 3n) |
| `{arch}.altup.num_inputs`, `.altup.active_idx`, `.altup.correct_scale`, `.laurel_rank`, `.activation_sparsity_scale` | Gemma 3n |
| `{arch}.rope.layer_bases` *new* | rotary base of every layer, 0 = no rotary embedding (Granite SWA) |
| `{arch}.attention.value_scale` *new* | constant multiplier on the values (MiMo-V2) |
| `{arch}.router_logit_softcapping` *new* | `cap * tanh(logit / cap)` on the router logits (Laguna) |
| `{arch}.attention.gate_per_element` *new* | reserved: per-element output gate (Laguna `gating: per-element`) |

Tensors added: `layer_output_scale`, `inp_gate`, `proj`, `post_norm`, `per_layer_token_embd`, `per_layer_model_proj`,
`per_layer_proj_norm`, `ffn_gate_inp.scale`, `ffn_down_exps.scale`, `pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`
(Gemma 4); `altup_*`, `laurel_*` (Gemma 3n). Layers with K = V have no `attn_v`, layers that share a cache have no
`attn_k` / `attn_v` / `attn_k_norm`.

**Gemma 4** (`gemma4`, text part of `Gemma4ForConditionalGeneration` / `Gemma4ForCausalLM`; vision and audio towers are
skipped): sliding layers and global layers with different head width and KV heads, K = V on global layers, proportional
rotary embedding (written as `rope_freqs.weight`), weightless value norm, attention scale 1, GELU-tanh, sandwich norms,
dense MLP and expert block side by side (26B-A4B), per-layer embeddings, shared KV layers with double-wide MLP
(E2B/E4B), per-layer output scalar, final logit soft cap. **Gemma 3n** (`gemma3n`): AltUp, LAuReL, per-layer embeddings,
Gaussian top-k activation sparsity, KV sharing. Also converted and checked (see (d)): Mellum (`qwen3moe` with sliding
layers and dense / sparse MLP layers), Laguna (`laguna`), MiMo-V2-Flash (`mimo_v2`), Granite SWA and GraniteMoE SWA
(`granite`, `granitemoe`).

Commands:

```
fllm-rs convert google/gemma-4-E2B-it gemma4.gguf --outtype f16     # or any local checkpoint directory
fllm-rs serve --model gemma4.gguf --device cpu
python scripts/verify_gemma4.py --list
python scripts/verify_gemma4.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint --generate
```

## (b) Environment variables and flags

| Name | Default | Meaning |
| --- | --- | --- |
| `FLLM_VERIFY_KV` (example `verify_checkpoint`) | `f16` | `f32` keeps the example's KV cache in single precision; `scripts/verify_gemma4.py` sets it (`--kv f16` to use the server precision) |
| `scripts/verify_gemma4.py --tol` | `1e-3` | relative logit error (max abs error / logit range) |
| `scripts/verify_gemma4.py --generate` | off | also compare a greedy continuation served by `fllm-rs serve` (CPU, paged cache) |

No runtime switch was added: models with per-layer geometry run on the host (CPU) path; `--device cuda|vulkan|...` for
`gemma4`, `gemma3n`, `laguna`, `mimo_v2` and any model with `layer_geoms` returns an error for an explicit device and
selects the CPU with `--device auto`.

## (c) THIRD_PARTY

No third-party code was adapted. The tensor and metadata names follow the public llama.cpp GGUF conventions (names only;
nothing was copied from ggml-org sources). `normal_quantile` implements Wichura's algorithm AS 241 (Applied Statistics 37,
1988) from its published coefficients.

## (d) Test levels

* L2 (CPU): tiny random checkpoints built with `transformers` 5.19.0 (hidden 64, 4-6 layers), converted with
  `fllm-rs convert --outtype f32`, logits of 14 positions compared with transformers on the CPU backend with an f32 KV
  cache, plus a greedy continuation through `fllm-rs serve`:

  | fixture | relative logit error | argmax | served greedy text |
  | --- | --- | --- | --- |
  | gemma4-dense | 1.8e-6 | equal | equal |
  | gemma4-moe | 9.6e-7 | equal | equal |
  | gemma4-ple (shared KV, double-wide MLP) | 8.6e-7 | equal | equal |
  | gemma4-vlm (`model.language_model.*` names) | 1.8e-6 | equal | equal |
  | gemma3n | 9.6e-7 | equal | equal |
  | mellum | 3.4e-7 | equal | equal |
  | laguna | 5.7e-7 | equal | equal |
  | mimo (MiMo-V2-Flash) | 3.8e-7 | equal | equal |
  | granite_swa | 3.2e-7 | equal | equal |
  | granitemoe_swa | 3.8e-7 | equal | equal |
  | gemma3 (existing architecture, baseline) | 3.5e-6 | equal | equal |

  With the f16 KV cache of the server the same fixtures stay below 1e-3 (gemma4-dense 8.2e-4, gemma4-moe 6.7e-4,
  gemma4-ple 6.3e-4); the existing gemma3 fixture shows 1.8e-3 in that mode, so the f16 cache, not the new code, sets that floor.
* Rust tests without Python: `rust/engine/src/gemma_tests.rs` (golden logits of the checked-in tiny checkpoints in
  `rust/engine/tests/fixtures/{gemma4,gemma3n}-tiny`, batch versus one-token-at-a-time decoding through the cache, layer
  geometry and pool layout), importer tests in `rust/checkpoint/src/gemma.rs` and `gemma3n.rs`, `kv_geometry.rs`.
  `scripts/make_gemma_fixtures.py` regenerates the fixtures.
* L0 only: nothing touches the GPU kernels. No GPU backend (CUDA, wgpu, Metal, HIP, SYCL) was changed or run for these
  models.

## (e) Known limits

* Per-layer geometry is CPU-only. The device attention kernels, `DeviceKvCache` and the CUDA / wgpu cache layouts still
  assume one geometry; a model with `layer_geoms` (and `gemma4`, `gemma3n`, `laguna`, `mimo_v2`) is refused on an explicit
  accelerator and runs on the CPU with `--device auto`. Per-layer fallback to the host inside a device run is not
  implemented for these layers.
* Sliding-window layers keep all blocks of the sequence (one block table for all layers); blocks outside the window are
  not freed and there are no separate pools per geometry group. Long contexts therefore cost the KV of every layer; the
  memory planner counts it correctly, but it is far above what a windowed pool would need.
* Gemma 4: text only (no vision / audio), no llama.cpp GGUF loading test (the layout follows the public key and tensor
  names but no real llama.cpp file was available), `attention.gate_per_element` of Laguna is only written, not read.
* Granite SWA and GraniteMoE SWA have uniform geometry and are not refused on accelerators, but the sinks and the per-layer rotary tables were only run on the CPU; on a GPU they are L0 (compiled, not run).
* Gemma 3n: AltUp coefficient clipping (training only) is ignored.
* Not done: Step-3.5, DeciLM / Nemotron-51B/253B (`block_configs`), NemotronH-Puzzle, Nanbeige4.2 / IQuestLoopCoder
  (`num_loops`), Phi-3-small block-sparse attention, Cohere2 and gpt-oss re-checks.
* Real checkpoints were not available (no download in tests): the 26B-A4B / 31B / E2B / E4B configurations are covered by
  the same code paths in tiny form only.

## architectures.md entries

| model_type | HF class | GGUF arch | status | verification |
| --- | --- | --- | --- | --- |
| `gemma4`, `gemma4_text` | `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | `gemma4` | V (CPU, text) | `python scripts/verify_gemma4.py gemma4-dense gemma4-moe gemma4-ple gemma4-vlm` |
| `gemma3n`, `gemma3n_text` | `Gemma3nForConditionalGeneration`, `Gemma3nForCausalLM` | `gemma3n` | V (CPU, text) | `python scripts/verify_gemma4.py gemma3n` |
| `mellum` | `MellumForCausalLM` | `qwen3moe` | V (CPU) | `python scripts/verify_gemma4.py mellum` |
| `laguna` | `LagunaForCausalLM` | `laguna` | V (CPU) | `python scripts/verify_gemma4.py laguna` |
| `mimo_v2_flash` | `MiMoV2FlashForCausalLM` | `mimo_v2` | V (CPU) | `python scripts/verify_gemma4.py mimo` |
| `granite_swa` | `GraniteSWAForCausalLM` | `granite` | V | `python scripts/verify_gemma4.py granite_swa` |
| `granitemoe_swa` | `GraniteMoeSWAForCausalLM` | `granitemoe` | V | `python scripts/verify_gemma4.py granitemoe_swa` |
