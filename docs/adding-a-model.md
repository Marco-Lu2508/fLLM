# Adding a model architecture

A new architecture touches three places that must agree with each other:

1. **`rust/checkpoint/src/plan.rs`** — how the Hugging Face checkpoint becomes a GGUF file.
2. **`rust/engine/src/config.rs`** — how the GGUF metadata becomes the decoder description.
3. **The verification harness** (`scripts/verify_*.py`) and the tables in
   [architectures.md](architectures.md).

The decoder itself has no per-architecture forward code: `rust/engine/src/rust_decoder.rs`
executes whatever `ModelConfig` describes, so the whole task is producing and describing the
right tensors.

## 1. Conversion plan (`rust/checkpoint/src/plan.rs`)

`convert_checkpoint` (`rust/checkpoint/src/convert.rs`) calls `Plan::from_config(&config.json)`:

- Older families are tried first via `classic::plan_for` (GPT-2, GPT-NeoX, Falcon, BLOOM, OPT,
  MPT, GPT-J and similar, see `rust/checkpoint/src/classic.rs`).
- Otherwise the `model_type` string is looked up in `family_for(model_type)`, which returns a
  `(Family, arch)` pair. An unknown `model_type` aborts the conversion with
  `unsupported model_type …; refusing fallback architecture mapping` — there is deliberately
  no fallback guess.

Adding an entry there decides two things:

- **`Family`** selects the conversion behaviour: tensor-name conventions and HF split-half →
  GGUF interleaved rotary permutation (`Llama`, `Qwen`), the `1 + w` norm constant (`Gemma`),
  post-norm handling (`Olmo2`), MoE naming/splitting, MLA layouts (`Deepseek`), plus the
  dedicated constructors `from_mamba` and `from_nemotron`. Reuse an existing family when the
  layout matches; only add a new `Family` when the conversion logic itself differs.
- **`arch`** becomes `general.architecture` in the GGUF (llama.cpp naming, e.g. `qwen2moe`,
  `command-r`, `deepseek2`). It must be the same string `config.rs` will read back.

The `Plan` struct carries everything the converter needs: `meta` (the GGUF key/value pairs it
writes, `{arch}.block_count`, `{arch}.feed_forward_length`, rope and norm keys …), `layers`,
`moe`/`moe_layers`/`gated`, `q_lora`, `mixtral_names`, `synthesized` (tensors fLLM builds
itself), `linear` (gated-DeltaNet dimensions), `kinds` (per-layer mixer kinds for Nemotron-H)
and `norm_add`. Per tensor, `plan.map(hf_name)` returns one of:

| Variant | Meaning |
| --- | --- |
| `Skip` | Not needed in GGUF (rotary caches and similar buffers) |
| `Tensor { name, add }` | Rename; `add` is a constant added to every element (Gemma norms) |
| `Fused { layer, part }` | One whole-layer fused expert tensor the converter splits itself |
| `Transposed { name }` | `[in, out]` matrix (GPT-2 Conv1D) written to GGUF as `[out, in]` |
| `Expert { name, index }` | One expert, stacked into `name` in index order |

Consistency checks that will reject a wrong mapping: the expert counts found in
`config.json` must match the architecture's MoE list, MoE models with mixed dense layers
(`decoder_sparse_step`, `mlp_only_layers`) are refused, and required keys (`hidden_size`,
`num_attention_heads`, `num_hidden_layers`, …) must be present.

Output tensor names follow the llama.cpp convention the rest of fLLM reads:
`token_embd.weight`, `output.weight`, `output_norm.weight`, `blk.<i>.attn_q.weight`,
`blk.<i>.ffn_down.weight`, expert tensors under `blk.<i>.ffn_gate_exps.weight` and so on.

## 2. Decoder description (`rust/engine/src/config.rs`)

`ModelConfig::from_gguf` reads `general.architecture` (it falls back to `qwen3` when the key is
missing) and then looks up keys through `key(meta, arch, name)`, i.e. `"{arch}.{name}"`:

- Dimensions: `embedding_length`, `block_count`, `attention.head_count`,
  `attention.head_count_kv`, `attention.key_length`/`attention.head_dim`,
  `attention.value_length`, `feed_forward_length`, `attention.layer_norm_rms_epsilon` /
  `attention.layer_norm_epsilon`.
- RoPE: `rope.dimension_count`, `rope.freq_base`, `rope.freq_base_swa`, `rope.flipped`,
  `rope.scaling` plus the `rope_freqs.weight` / `rope_factors_short.weight` /
  `rope_factors_long.weight` tensors.
- Experts: `expert_count`, `expert_used_count`, `expert_feed_forward_length`,
  `expert_gating_func`, `expert_weights_norm`, `expert_weights_scale`,
  `expert_shared_count`, `leading_dense_block_count`.
- State space: `ssm.group_count`, `ssm.conv_kernel`, `ssm.inner_size`, `ssm.state_size`,
  `ssm.time_step_rank`, `ssm.time_step_min`, `ssm.beta_scale`, `ssm.norm_eps`
  (`parse_mamba`, `parse_delta`).

Decisions that mix metadata with tensor presence:

- Norm kind: `blk.0.attn_norm.bias` / `output_norm.bias` present (or an arch-name allowlist
  such as `command-r`, `cohere2`, `jais2`, `mpt`, `olmo`, `stablelm`) selects
  `NormKind::Layer`, otherwise `NormKind::Rms`.
- RoPE layout: `neox` style is the default; the arch names `llama`, `mistral`, `deepseek`,
  `deepseek2`, `ernie4_5`, `llama4`, `glm`, `glm4`, `granite`, `minicpm`, `gptj`,
  `internlm2`, `command-r`, `cohere`, `cohere2` use the GPT-J style instead — a new arch with
  the other layout must be added to that list.
- Q/K norm, activations, sliding windows, MLA and tied embeddings are derived from metadata
  and tensor presence (`output.weight` missing ⇒ tied head).

So the checklist for this file is: the arch string written by `plan.rs` appears where the
decoder needs an arch-specific exception, and every `{arch}.*` key the plan writes is read (or
deliberately ignored) here. Anything the decoder then needs at load time is validated in
`RustDecoder::load` (even head/rope dimensions, positive epsilon, group norms, tensor shapes).

## 3. Verification

Test-only Python (`torch`, `transformers`, `tokenizers`) provides the reference; production
code does not import it.

**a. Register a fixture.** `scripts/hf_golden.py` builds a tiny random checkpoint and its
transformers reference logits per architecture; `config_for()` holds the registry
(`"qwen3": lambda: T.Qwen3Config(...)`). Add an entry with a transformers config class if the
architecture is not in the list yet.

**b. Logit comparison** — convert with fLLM and compare every position against transformers
(relative difference below `2e-3`, arg-max must not differ beyond numeric noise):

```sh
make build-rust
cargo build --release --locked -p fllm-engine --example verify_checkpoint
python scripts/verify_architectures.py --fllm target/release/fllm-rs \
    --example target/release/examples/verify_checkpoint qwen3 my_new_arch
```

Useful options: `--outtype f32` (default), `--workdir DIR`, `--keep` (keep the generated
checkpoint and GGUF).

**c. Generation through the HTTP server** (incremental decoding, concurrency, and the
recurrent state of hybrid/state-space models):

```sh
python scripts/verify_generation.py --fllm target/release/fllm-rs my_new_arch
# default port 8741; --port changes it
```

**d. Tokenizer** (the GGUF keeps the complete HF tokenizer JSON; check the pre-tokenizer
behaviour with):

```sh
python scripts/verify_tokenizer.py MODEL.gguf HF_TOKENIZER_DIR target/release/examples/tokenize
```

**e. Regressions and style:**

```sh
make test
make lint
```

**f. Encoder and vision models** use their own harnesses:
`scripts/verify_encoder.py --binary target/release/fllm-rs --model bge-small.gguf` and
`scripts/verify_vision.py --binary target/release/fllm-rs --model SmolVLM.gguf --mmproj mmproj.gguf`.

## 4. Document it

Add the `model_type` to the right table in [architectures.md](architectures.md) (dense, MoE,
recurrent/hybrid) and to the "Not supported" list when it is deliberately rejected. The
document states what has actually been compared with `transformers`; real checkpoints of most
families were not available, so keep the wording aligned with what the scripts prove.

## See also

- [rust-formats-and-import.md](rust-formats-and-import.md) — containers, tokenizer retention
  and the import limits
- [quantization.md](quantization.md) — output types for `convert --outtype`
