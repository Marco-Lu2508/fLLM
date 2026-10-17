# Hybrid state-space / linear-attention families

New architectures, all checked logit-for-logit against the official `transformers` implementation on tiny random
checkpoints (CPU backend), plus a snapshot / restore API for the recurrent state of every stateful layer kind.

| GGUF architecture | Hugging Face `model_type` | What it is |
| --- | --- | --- |
| `falcon-h1` | `falcon_h1` | Mamba-2 and attention in parallel in every layer, mu-P multipliers |
| `jamba` | `jamba` | Mamba-1 with learned dt/B/C norms, attention every N layers, MoE every second layer |
| `zamba2` | `zamba2` | Mamba-2 backbone with shared transformer blocks (per-call LoRA adapters, `[x, embeddings]` input) |
| `kimi-linear` | `kimi_linear` | Kimi Delta Attention (KDA) layers + latent attention without RoPE + DeepSeek-V3 style MoE |

## 1. Features and commands

Conversion and a quick run work as for every other model:

```sh
fllm-rs convert tiiuae/Falcon-H1-1.5B-Deep-Instruct falcon-h1.gguf        # directory of the HF checkpoint
fllm-rs convert ai21labs/Jamba-v0.1 jamba.gguf --outtype f16               # Hub (per-expert) and transformers-5 (stacked) layouts
fllm-rs convert Zyphra/Zamba2-2.7B zamba2.gguf
fllm-rs convert moonshotai/Kimi-Linear-48B-A3B-Instruct kimi.gguf          # Hub names (`block_sparse_moe.*`, `q_conv1d`, ...)
```

(Arguments are local checkpoint directories; nothing here downloads. The model names are the intended targets; only tiny
random checkpoints built with `transformers` 5.19 were run, see the test levels.)

### Falcon-H1

* Both mixers read the same input norm; their outputs, each through its own output projection, are summed before the residual
  add. The decoder keeps the Mamba-2 branch in `Layer::parallel` next to the ordinary attention weights.
* The mu-P multipliers are folded into the weights while converting (`rust/checkpoint/src/hybrid.rs`, `ScaleSet`):
  `ssm_in_multiplier * ssm_multipliers[0..5]` into the rows of `mamba.in_proj` (`[z | x | B | C | dt]`, the bias only with its
  mu-P part), `ssm_out_multiplier` into `mamba.out_proj`, `attention_in_multiplier` into q/k/v, `key_multiplier` into k,
  `attention_out_multiplier` into `o_proj`, `mlp_multipliers` into gate and down. `embedding_multiplier` and
  `lm_head_multiplier` stay metadata (`embedding_scale`, `logit_scale`) because the head may share the embedding. BF16/F16
  sources are re-rounded to their own dtype after the multiplication; quantised sources with non-unit multipliers are refused.
* Mamba-2 variants: `mamba_d_ssm` different from `expand * hidden`, `n_groups`, optional gated norm (`ssm.norm_mode`: 0 gate then
  norm, 1 no norm, 2 norm then gate), `time_step_limit` (`ssm.time_step_min` / `ssm.time_step_max`). The chunk size does not
  change the result and is ignored. Tied embeddings are handled by the generic converter.

### Jamba

* Mamba-1 layers whose `dt`, `B` and `C` projections go through RMS norms with learned weights (`ssm_dt_norm`, `ssm_b_norm`,
  `ssm_c_norm`); attention layers (`attn_layer_period` / `attn_layer_offset`) have no rotary embeddings
  (`jamba.rope.layer_flags` all false); `expert_layer_period` / `expert_layer_offset` select the MoE layers (`layers_num_experts`
  is 1 for dense layers: the engine decides per layer from the tensors, so a dense layer is an ordinary gated FFN).
* Router: softmax over all experts, top-k, no renormalisation. Experts are accepted as one tensor per expert (Hub files) or as
  the stacked `experts.gate_up_proj` / `experts.down_proj` of transformers 5.

### Zamba2

* The tied transformer blocks are stored once as `shared.<b>.*`; call `n` uses block `n % num_mem_blocks`. Every call owns
  `blk.<layer>.zamba_linear.weight`, its adapters (`adapter_{q,k,v}_{a,b}`, `adapter_ffn_{a,b}`) and the KV layer of the
  layer it sits in (an application of a shared block has its own KV cache). Later calls of a block repeat its weights in a
  Hugging Face checkpoint only as tied copies, which the converter drops.
* The shared block has no residual of its own: its output goes through the call's linear layer and is added to the input of the
  layer's Mamba-2 mixer (`x + t -> norm -> mamba`, then `x += mamba`). The block reads `[x, embeddings]`, so the decoder keeps the
  embedding output of the pass (`EmbedStash`). Attention scale `1/sqrt(head_dim/2)`, optional rotary embeddings (`use_mem_rope`).

### Kimi Delta Attention (Kimi-Linear)

* `rust/engine/src/cpu_kda.rs`: causal short convolution (q, k, v channels, SiLU), L2-normalised q/k (`x / sqrt(sum x^2 + 1e-6)`),
  decay per key channel `g = -exp(A_log) * softplus(f + dt_bias)`, `S *= exp(g)`, delta-rule update with `beta = sigmoid(b)`,
  readout, per-head RMS norm times `sigmoid(output gate)`. The recurrence is evaluated token by token (the chunked form is an
  exact reformulation); `gate_lower_bound` selects the GLM-5 "safe gate" `bound * sigmoid(exp(A_log) * f)`.
* Verified against fla: `scripts/gen_kda_reference.py` runs `fla.ops.kda.naive_recurrent_kda` and `naive_chunk_kda` on the CPU,
  asserts they agree, and writes `rust/engine/testdata/kda_reference.json`; the unit test `cpu_kda::tests::recurrence_matches_the_fla_reference_vectors`
  compares outputs, final state and the state after a split run with it.
* Kimi-Linear's latent-attention layers use no rotary embedding (`attention.mla_no_rope`, `MlaConfig::rotary`); the first
  layer may be a KDA layer (latent attention is detected from any layer).

### State snapshots (`rust/engine/src/rust_decoder/state_api.rs`, exported from `rust_decoder`)

```rust
pub trait RecurrentState { type Snapshot: Clone; fn tokens(&self) -> usize; fn snapshot(&self) -> Self::Snapshot; fn restore(&mut self, s: &Self::Snapshot); }
impl RustDecoder {
    pub fn has_recurrent_state(&self) -> bool;
    pub fn snapshot_state(&self, first_block: i64) -> Result<Option<SequenceSnapshot>, String>;
    pub fn restore_state(&self, first_block: i64, snapshot: &SequenceSnapshot) -> Result<(), String>;
    pub fn forget_state(&self, first_block: i64);
}
pub struct SequenceSnapshot { pub host: Vec<Option<LayerStateSnapshot>>, pub device: Option<(Vec<u8>, Vec<usize>)> }
// SequenceSnapshot::tokens() -> Option<usize> (smallest token count over all layers), ::bytes()
```

* A sequence is identified by its first KV block (`tables[row * stride]`), as everywhere in the decoder. The snapshot covers
  every layer kind that keeps state outside the KV cache: Mamba-1/2 (also next to attention and in Zamba2), Gated DeltaNet,
  KDA, LFM2 ShortConv (host state: `LayerStateSnapshot { tokens, conv, rec }`) and the device state buffer of the layers that run
  on the accelerator (`device`: one buffer per sequence, downloaded / uploaded through the decoder's stream).
* Speculative decoding: `snapshot_state(first)` before the verification pass, `restore_state(first, &snapshot)` when drafts are
  rejected, then decode the accepted prefix again; the state must be restored to a position `<=` the one to continue from
  (`tokens()` tells which). Prefix cache: store one snapshot per cached prefix next to its KV blocks and `restore_state` it under
  the first block of the new sequence. The KV cache of attention layers needs no extra handling: positions after the
  restore point are overwritten.
* A new stateful layer kind plugs in by storing its state in the decoder's `LayerState { len, conv, rec }` (host) or in the
  per-sequence device buffer; nothing else has to change. `ModelConfig::hybrid.stateful()` / `GgufModel::is_stateful()`
  keep the scheduler's prefix cache and speculative lookup switched off for such models until the `spec` work uses this API.
* Tested (`hybrid_tests::*_decodes_incrementally_and_rolls_back`, CPU): token-by-token decoding equals one batched pass (1e-4),
  a restored snapshot reproduces the continuation (1e-6), and a rolled-back run equals a fresh run of the same tokens.
  The device half of the API (`SequenceSnapshot::device`) is untested (no GPU).

## 2. Backends and fallbacks (`layer_host_reason`)

| Layer kind | CPU | Accelerator |
| --- | --- | --- |
| Falcon-H1 layer (attention + parallel Mamba-2) | yes | host reference between device layers ("a Mamba-2 branch next to the attention") |
| Jamba Mamba-1 layer with learned dt/B/C norms | yes | host ("Mamba-1 with learned dt/B/C norms"); Jamba attention and MoE layers run on the device |
| Zamba2 layer with a shared-block call | yes | host ("a shared transformer block in front of the Mamba-2 mixer"); plain Mamba-2 layers use the existing device kernels |
| KDA layer | yes | host ("Kimi Delta Attention"); no CUDA kernel yet |
| Latent attention without RoPE | yes | host (the device MLA kernel always rotates) |

Layers that fall back to the host download and upload the residual stream around them (existing mechanism,
`FLLM_HOST_LAYERS=all` forces it for debugging). CUDA graphs need every layer on the device (`graph_eligible`), so these
models decode without graphs; Jamba's attention layers are the only ones among them that would otherwise be eligible.

## 3. Verification

```sh
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors numpy tokenizers einops          # transformers >= 5.19 has all four classes
cargo build --release -p fllm-engine -p fllm-checkpoint --bins --examples
python scripts/verify_hybrid.py --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint \
    falcon_h1 falcon_h1_norm falcon_h1_norm_after falcon_h1_tied jamba jamba_fused_experts jamba_biased \
    zamba2 zamba2_rope zamba2_one_block kimi_linear kimi_linear_mla_first
cargo test --release -p fllm-engine -p fllm-checkpoint -p fllm-gguf-probe
```

The example runs the engine with an f16 KV cache, so the script rounds keys and values to f16 in the reference too
(`HYBRID_REF_KV16=0` switches that off; without it the wide shared attention of Zamba2 differs by about 1.5e-3 of the logit range,
with it by 4e-5). `HYBRID_REF_FP64=1` computes the reference in double precision. `HYBRID_USE_FLA=1` lets transformers use
flash-linear-attention (needs a GPU); by default the pure PyTorch layers are used.

## 4. Environment variables and flags

No new engine variables. Script-only switches: `HYBRID_REF_KV16` (default on), `HYBRID_REF_FP64` (off), `HYBRID_USE_FLA` (off).
Existing `FLLM_HOST_LAYERS`, `FLLM_MAMBA_HOST`, `FLLM_DELTA_HOST` behave as before.

## 5. Rows for `docs/architectures.md`

| `falcon_h1` | Mamba-2 and attention branches in parallel in every layer; mu-P multipliers folded at conversion; compared with transformers 5.19 on tiny checkpoints for all three Mamba norm orders and tied embeddings (1e-5 of the logit range) |
| `jamba` | Mamba-1 with learned dt/B/C norms + attention without RoPE + MoE every second layer (per-layer experts), per-expert and stacked expert layouts; compared with transformers 5.19 (3e-7) |
| `zamba2` | Mamba-2 + shared transformer blocks with per-call adapters, `[x, embeddings]` input, one KV layer per call; compared with transformers 5.19 with and without RoPE / attention adapters / two blocks (5e-5) |
| `kimi_linear` | KDA + latent attention without RoPE + DeepSeek-V3 MoE; compared with transformers 5.19 (4e-5 with a KDA first layer, 5e-4 with latent attention first) and with fla's recurrent/chunked reference vectors |

Remove Jamba, Falcon-H1, Zamba2 and Kimi-Linear (KDA) from the "Not supported" list.

## 6. THIRD_PARTY entries

* `rust/engine/testdata/kda_reference.json`: output vectors of `fla.ops.kda.naive_recurrent_kda` / `naive_chunk_kda`
  (flash-linear-attention, MIT licence, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li et al.). Data only, generated by
  `scripts/gen_kda_reference.py`; no fla code is included.
* The layer equations were derived from the `transformers` models `falcon_h1`, `jamba`, `zamba2`, `kimi_linear` (Apache-2.0);
  no code was copied. GGUF tensor names for Kimi-Linear follow the layout facts of llama.cpp (names only).

## 7. Test levels

* L2-equivalent on the CPU backend: all fixtures of section 3 agree with `transformers` (the numbers above), run on the build host
  with transformers 5.19.0 / torch 2.14.1 (CPU).
* Rust unit tests: `hybrid_tests` (state round trips for Falcon-H1, Jamba, Zamba2), `cpu_kda` (fla vectors, convolution carry-over),
  `hybrid_config`, `state_api`, converter tests for multipliers, Jamba / Zamba2 / Kimi-Linear names, metadata and layer requirements.
* No GPU was available: nothing here ran on an accelerator (L0 at best: the host fallbacks do not add device code). No CUDA kernels were
  written for these layer kinds.

## 8. Known limits and what is not done

* No CUDA/ROCm/Metal kernels for Falcon-H1's parallel branch, Jamba's normed Mamba-1, Zamba2's shared block or KDA: they run on the
  host between device layers (correct, slow). The `selftest-gpu` cases for a KDA kernel were therefore not written.
* GPU states are covered by the snapshot API but untested; the scheduler does not use snapshots yet (prefix cache and speculative
  lookup stay off for stateful models).
* Not implemented: Ling 3.0 / BailingMoeV3, Kimi K3, GLM-5.3-Flash (`glm5_next` also has hyper-connections and an indexer; only its
  KDA layer maps onto `cpu_kda` through `gate_lower_bound`), RWKV-6/7, xLSTM, Granite-4 / Nemotron-H / Bamba / LFM2 variants beyond
  what already existed, MiniCPM-SALA, JetNemotron, HRM-text, ZAYA1. Zamba2 with `add_bias_linear`, a non-GELU activation or
  `num_mem_blocks` adapters on k/v when `num_key_value_heads != num_attention_heads` is refused or unverified.
* Real checkpoints (Falcon-H1-*, Jamba-*, Zamba2-*, Kimi-Linear-48B) were not available here; tensor names of the Kimi-Linear Hub layout
  were taken from the `transformers` conversion mapping and the saved tiny model, not from the Hub files.
* Falcon-H1 BF16 weights are multiplied and rounded to BF16 again (one extra rounding of 2^-9 relative per scaled tensor).
