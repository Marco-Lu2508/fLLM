//! Architecture-independent description of a decoder-only transformer, read from GGUF metadata
//! (llama.cpp key names) and tensor presence. The decoder is driven entirely by this description.
use fllm_gguf_probe::MappedGguf;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `x / rms(x) * w`
    Rms,
    /// `(x - mean) / std * w + b`
    Layer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
    GeluErf,
    Relu,
    /// `relu(x)^2`
    ReluSquared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkNorm {
    None,
    /// RMS norm over each head's `head_dim` values (Qwen3, Gemma3)
    PerHead,
    /// RMS norm over the whole projected vector (OLMo2)
    Full,
}

/// How rotary frequencies are adjusted for long contexts.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    /// positions are divided by `factor`
    Linear {
        factor: f32,
    },
    /// Phi-3: `RopeConfig::factors` are the short factors; contexts beyond `original_context`
    /// use `long` and scale cos/sin by `attention`.
    LongRope {
        original_context: usize,
        attention: f32,
        long: Vec<f32>,
    },
    Yarn {
        factor: f32,
        original_context: usize,
        extrapolation: f32,
        attention: f32,
        beta_fast: f32,
        beta_slow: f32,
        /// Round the correction range to whole dimensions (ggml does; GPT-OSS turns it off).
        truncate: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    /// Number of leading head dimensions that rotate.
    pub dims: usize,
    pub base: f32,
    /// Alternative base for sliding-window layers (Gemma3).
    pub local_base: Option<f32>,
    /// Pair `(i, i + dims/2)` (NeoX) instead of `(2i, 2i + 1)`.
    pub neox: bool,
    pub scaling: RopeScaling,
    /// Per-frequency divisors (`rope_freqs.weight`), length `dims / 2`.
    pub factors: Option<Vec<f32>>,
    /// Rotate by the opposite angle (NanoChat's flipped `rotate_half`).
    pub flipped: bool,
}

/// Activation of the per-head attention output gate (`attn_gate.weight`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttnGate {
    Sigmoid,
    /// `softplus` with the given `beta` (K2 Horizon: ln 2)
    Softplus(f32),
}

impl AttnGate {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            AttnGate::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            AttnGate::Softplus(beta) => {
                let scaled = beta * x;
                if scaled > 20.0 {
                    x
                } else {
                    scaled.exp().ln_1p() / beta
                }
            }
        }
    }
}

/// K2 Horizon MoVA: the values of a layer are a sum of routed per-expert projections.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueExperts {
    pub count: usize,
    pub used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gating {
    Softmax,
    Sigmoid,
    SparseMixer { jitter_eps: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub experts: usize,
    pub used: usize,
    pub expert_ff: usize,
    pub gating: Gating,
    /// Renormalise the selected weights so they sum to one.
    pub normalize: bool,
    /// DBRX divides the selected softmax scores by their p-norm (null disables it).
    pub weight_norm_p: Option<f32>,
    pub weight_scale: f32,
    /// Number of always-active shared experts (their projections are fused in one FFN).
    pub shared: usize,
    pub shared_ff: usize,
    /// First layers use an ordinary dense FFN.
    pub dense_layers: usize,
    /// Group-limited routing (DeepSeek): experts are split in `groups`, only the best
    /// `groups_used` groups may be chosen from. `groups <= 1` disables it.
    pub groups: usize,
    pub groups_used: usize,
    /// Rank groups by the sum of their two best scores instead of the best one.
    pub group_top2: bool,
    /// The router score scales the expert's input instead of its output (Llama 4).
    pub input_scaled: bool,
    /// GPT-OSS clamped SwiGLU: `(alpha, limit)`.
    pub swiglu_oai: Option<(f32, f32)>,
    /// Nemotron-H latent MoE: the routed experts work on a `latent`-wide projection of the hidden state.
    pub latent: Option<usize>,
}

/// Gated DeltaNet layers (Qwen3-Next, Qwen3.5): a recurrent state replaces the KV cache.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaConfig {
    pub conv_kernel: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    /// Multiplier on the write strength (`2.0` lets Olmo-Hybrid reach negative eigenvalues).
    pub beta_scale: f32,
    /// Epsilon of the gated output norm when it differs from the model's.
    pub norm_eps: Option<f32>,
}

/// Mamba-2 state-space layers: a recurrent state replaces the KV cache.
#[derive(Clone, Debug, PartialEq)]
pub struct MambaConfig {
    pub conv_kernel: usize,
    pub inner: usize,
    pub state: usize,
    pub heads: usize,
    pub groups: usize,
    /// Groups of the gated RMS norm over the inner width (1 = one norm over everything).
    pub norm_groups: usize,
    pub step_min: f32,
}

/// LFM2 gated depthwise convolution, with an independent history for each sequence/layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ShortConvConfig {
    pub kernel: usize,
    pub layers: Vec<bool>,
}

/// Mamba-1 layers (per-channel state, low-rank time step); Falcon-Mamba also normalises the
/// time step, B and C.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba1Config {
    pub conv_kernel: usize,
    pub inner: usize,
    pub state: usize,
    pub rank: usize,
    pub bc_rms_eps: Option<f32>,
}

/// Multi-head latent attention (DeepSeek-V2/V3, Kimi-K2): keys and values are rebuilt from a
/// compressed `kv_lora`-wide latent. The cache holds the rebuilt per-head keys and values.
#[derive(Clone, Debug, PartialEq)]
pub struct MlaConfig {
    pub q_lora: Option<usize>,
    pub kv_lora: usize,
    /// Per-head query/key width without position information.
    pub nope: usize,
    /// Rotary part, shared by all heads on the key side.
    pub rope: usize,
    pub value: usize,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// Per-head query/key width.
    pub head_dim: usize,
    /// Per-head value width.
    pub value_dim: usize,
    pub ff: usize,
    pub eps: f32,
    /// Epsilon of the norms behind the attention / feed-forward branches (Muse Glimmer: 1e-8).
    pub post_eps: f32,
    pub norm: NormKind,
    pub rope: RopeConfig,
    pub qk_norm: QkNorm,
    pub activation: Activation,
    /// Gate and up projections (false: a single up projection and an activation).
    pub gated_ffn: bool,
    /// Attention and FFN read the same normalised input and add to the residual together.
    pub parallel_residual: bool,
    /// Norms applied to the attention / FFN output before it joins the residual.
    pub post_norms: bool,
    /// Attention scale; `None` means `1 / sqrt(head_dim)`.
    pub attention_scale: Option<f32>,
    pub attention_softcap: Option<f32>,
    pub logit_softcap: Option<f32>,
    pub sliding_window: Option<usize>,
    /// Layer `i` uses the window when `i % n < n - 1` (n = pattern); 1 means every layer.
    pub window_pattern: usize,
    /// Explicit per-layer flags (override `window_pattern` when present).
    pub window_flags: Option<Vec<bool>>,
    pub embedding_scale: f32,
    pub logit_scale: f32,
    pub residual_scale: f32,
    pub tied_embeddings: bool,
    pub moe: Option<MoeConfig>,
    /// The embeddings go through a parameter-free RMS norm before the first layer.
    pub embedding_norm: bool,
    /// Learned absolute positions are added to the embeddings (`position_embd.weight`); OPT
    /// shifts the index by `position_offset`.
    pub position_offset: usize,
    /// ALiBi slope per head, added as `slope * (key_position - query_position)` to the scores.
    pub alibi_slopes: Option<Vec<f32>>,
    /// Queries, keys and values are clamped to `[-clip, clip]` after projection.
    pub qkv_clip: Option<f32>,
    pub mla: Option<MlaConfig>,
    pub delta: Option<DeltaConfig>,
    pub mamba: Option<MambaConfig>,
    pub mamba1: Option<Mamba1Config>,
    pub shortconv: Option<ShortConvConfig>,
    /// Layers that skip rotary embeddings (`None`: all rotate).
    pub rope_layers: Option<Vec<bool>>,
    /// Query/key norms run after the rotary embedding (Hunyuan).
    pub qk_norm_after_rope: bool,
    /// Queries are scaled by `1 + beta * ln(1 + floor((position + offset) / context))`.
    pub attn_temperature: Option<(f32, usize)>,
    pub temperature_offset: usize,
    /// Only layers without rotary embeddings use the temperature scale (Llama 4).
    pub temperature_nope_only: bool,
    /// Layers that rotate attend within aligned chunks of this many positions (Llama 4).
    pub attention_chunk: Option<usize>,
    pub attn_gate: AttnGate,
    pub value_experts: Option<ValueExperts>,
}

fn key<'a>(meta: &'a Map<String, Value>, arch: &str, name: &str) -> Option<&'a Value> {
    meta.get(&format!("{arch}.{name}"))
        .or_else(|| meta.get(name))
}

fn uint(meta: &Map<String, Value>, arch: &str, name: &str) -> Option<usize> {
    key(meta, arch, name)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
}

fn float(meta: &Map<String, Value>, arch: &str, name: &str) -> Option<f32> {
    key(meta, arch, name)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
}

fn has(mapping: &MappedGguf, name: &str) -> bool {
    mapping.index().tensors.iter().any(|t| t.name == name)
}

/// First non-zero element when GGUF stores a per-layer array (hybrid models put 0 on layers without
/// the feature), otherwise the scalar itself.
fn uniform(meta: &Map<String, Value>, arch: &str, name: &str) -> Option<usize> {
    match key(meta, arch, name)? {
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_u64)
            .find(|&v| v > 0)
            .or_else(|| values.first().and_then(Value::as_u64))
            .map(|v| v as usize),
        value => value.as_u64().map(|v| v as usize),
    }
}

impl ModelConfig {
    pub fn from_gguf(mapping: &MappedGguf) -> Result<Self, String> {
        let meta = &mapping.index().metadata;
        let arch = meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("qwen3")
            .to_owned();
        let name = arch.clone();
        let a = name.as_str();
        let hidden = uint(meta, a, "embedding_length").ok_or("missing embedding_length")?;
        let layers = uint(meta, a, "block_count").ok_or("missing block_count")?;
        // state-space models have no attention; a one-head dummy keeps the cache tiny
        let state_space = matches!(a, "mamba2" | "mamba");
        let heads = if state_space {
            1
        } else {
            uniform(meta, a, "attention.head_count").ok_or("missing head_count")?
        };
        let kv_heads = uniform(meta, a, "attention.head_count_kv").unwrap_or(heads);
        let head_dim = uint(meta, a, "attention.key_length")
            .or_else(|| uint(meta, a, "attention.head_dim"))
            .unwrap_or(if state_space {
                2
            } else {
                hidden / heads.max(1)
            });
        let value_dim = uint(meta, a, "attention.value_length").unwrap_or(head_dim);
        let ff = uniform(meta, a, "feed_forward_length").unwrap_or(hidden * 4);
        if hidden == 0 || layers == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(format!("GGUF has invalid {arch} dimensions"));
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(format!(
                "{arch}: head count {heads} is not a multiple of {kv_heads}"
            ));
        }
        let embedding = mapping
            .index()
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("missing token_embd.weight")?;
        if embedding.dimensions.len() != 2 || embedding.dimensions[0] as usize != hidden {
            return Err(format!(
                "embedding width {:?} does not match metadata {hidden}",
                embedding.dimensions
            ));
        }
        let tied = !has(mapping, "output.weight");
        let vocab = if tied {
            embedding.dimensions[1] as usize
        } else {
            mapping
                .index()
                .tensors
                .iter()
                .find(|t| t.name == "output.weight")
                .map(|t| t.dimensions.get(1).copied().unwrap_or(0) as usize)
                .unwrap_or(0)
        };
        let eps = float(meta, a, "attention.layer_norm_rms_epsilon")
            .or_else(|| float(meta, a, "attention.layer_norm_epsilon"))
            .unwrap_or(1e-6);
        let norm = if has(mapping, "blk.0.attn_norm.bias")
            || has(mapping, "output_norm.bias")
            || matches!(
                a,
                "command-r" | "cohere" | "cohere2" | "jais2" | "nemotron" | "mpt" | "olmo" | "stablelm" | "dbrx"
            ) {
            NormKind::Layer
        } else {
            NormKind::Rms
        };
        let rope_dims = uint(meta, a, "rope.dimension_count").unwrap_or(head_dim);
        let rope = RopeConfig {
            dims: rope_dims,
            base: float(meta, a, "rope.freq_base").unwrap_or(1_000_000.0),
            // Gemma 3 GGUFs do not store the sliding-window layers' base; llama.cpp hardcodes 10000 for them
            local_base: float(meta, a, "rope.freq_base_swa").or((a == "gemma3").then_some(10_000.0)),
            neox: !matches!(
                a,
                "llama"
                    | "muse-glimmer"
                    | "mistral"
                    | "deepseek"
                    | "deepseek2"
                    | "ernie4_5"
                    | "ernie4_5moe"
                    | "llama4"
                    | "glm"
                    | "glm4"
                    | "granite"
                    | "minicpm"
                    | "gptj"
                    | "internlm2"
                    | "command-r"
                    | "cohere2"
                    | "cohere"
            ),
            scaling: {
                let mut scaling = parse_scaling(meta, a)?;
                // Phi-3 GGUFs carry no rope.scaling.type: the long / short factor tensors and rope.scaling.attn_factor
                // are what llama.cpp goes by (the factors pick by context length, the factor scales cos / sin)
                if matches!(scaling, RopeScaling::None)
                    && a == "phi3"
                    && has(mapping, "rope_factors_long.weight")
                    && float(meta, a, "rope.scaling.attn_factor").is_some()
                {
                    if let Some(original_context) = uint(meta, a, "rope.scaling.original_context_length") {
                        scaling = RopeScaling::LongRope {
                            original_context,
                            attention: float(meta, a, "rope.scaling.attn_factor").unwrap_or(1.0),
                            long: Vec::new(),
                        };
                    }
                }
                scaling
            },
            flipped: key(meta, a, "rope.flipped")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            factors: rope_factors(mapping, rope_dims, "rope_freqs.weight")?.or(rope_factors(
                mapping,
                rope_dims,
                "rope_factors_short.weight",
            )?),
        };
        let mut rope = rope;
        if let RopeScaling::LongRope { long, .. } = &mut rope.scaling {
            *long = rope_factors(mapping, rope_dims, "rope_factors_long.weight")?
                .ok_or("longrope scaling needs rope_factors_long.weight")?;
            if rope.factors.is_none() {
                return Err("longrope scaling needs rope_factors_short.weight".into());
            }
        }
        let qk_norm = if mapping
            .index()
            .tensors
            .iter()
            .any(|t| t.name.ends_with(".attn_q_norm.weight"))
        {
            let width = mapping
                .index()
                .tensors
                .iter()
                .find(|t| t.name.ends_with(".attn_q_norm.weight"))
                .map_or(0, |t| t.dimensions.iter().product::<u64>() as usize);
            if width == heads * head_dim && width != head_dim {
                QkNorm::Full
            } else {
                QkNorm::PerHead
            }
        } else {
            QkNorm::None
        };
        let post_norms = has(mapping, "blk.0.post_attention_norm.weight")
            && has(mapping, "blk.0.post_ffw_norm.weight");
        let sliding_window = uint(meta, a, "attention.sliding_window").filter(|&w| w > 0);
        let window_flags = match key(meta, a, "attention.sliding_window_pattern") {
            Some(Value::Array(values)) if values.len() == layers => values
                .iter()
                .map(Value::as_bool)
                .collect::<Option<Vec<bool>>>(),
            _ => None,
        };
        let gemma = a.starts_with("gemma");
        let window_pattern = uint(meta, a, "attention.sliding_window_pattern")
            .unwrap_or(match a {
                "gemma2" => 2,
                "gemma3" => 6,
                _ => 1,
            })
            .max(1);
        let gated_ffn =
            has(mapping, "blk.0.ffn_gate.weight") || has(mapping, "blk.0.ffn_gate_exps.weight") || has(mapping, "blk.0.ffn_gate_up_exps.weight");
        let mla = if has(mapping, "blk.0.attn_kv_a_mqa.weight") {
            let rope = rope_dims;
            let key_width = uint(meta, a, "attention.key_length_mla")
                .or_else(|| uint(meta, a, "attention.key_length"))
                .ok_or("MLA model needs attention.key_length")?;
            Some(MlaConfig {
                q_lora: has(mapping, "blk.0.attn_q_a.weight")
                    .then(|| uint(meta, a, "attention.q_lora_rank"))
                    .flatten(),
                kv_lora: uint(meta, a, "attention.kv_lora_rank").ok_or("missing kv_lora_rank")?,
                nope: key_width
                    .checked_sub(rope)
                    .filter(|&n| n > 0)
                    .ok_or("invalid MLA key width")?,
                rope,
                value: uint(meta, a, "attention.value_length_mla")
                    .or_else(|| uint(meta, a, "attention.value_length"))
                    .ok_or("MLA model needs attention.value_length")?,
            })
        } else {
            None
        };
        let (kv_heads, head_dim, value_dim) = match &mla {
            Some(m) => (heads, m.nope + m.rope, m.value),
            None => (kv_heads, head_dim, value_dim),
        };
        let mut attention_scale = float(meta, a, "attention.scale");
        if mla.is_some() {
            // ggml folds the YaRN magnitude into the softmax scale for DeepSeek models
            let multiplier = float(meta, a, "rope.scaling.yarn_log_multiplier").unwrap_or(0.0);
            let mut mscale = 1.0f32;
            if let RopeScaling::Yarn {
                factor, attention, ..
            } = &mut rope.scaling
            {
                mscale = 1.0 + multiplier * factor.ln();
                *attention = 1.0 / (1.0 + 0.1 * factor.ln());
            }
            attention_scale = Some(mscale * mscale / (head_dim as f32).sqrt());
        }
        Ok(Self {
            arch,
            hidden,
            layers,
            vocab,
            heads,
            kv_heads,
            head_dim,
            value_dim,
            ff,
            eps,
            post_eps: float(meta, a, "attention.layer_norm_post_epsilon")
                .unwrap_or(if a == "muse-glimmer" { 1e-8 } else { eps }),
            norm,
            rope,
            qk_norm,
            activation: match a {
                _ if gemma => Activation::GeluTanh,
                "starcoder2" | "gpt2" | "starcoder" | "gpt_neo" | "gptj" | "bloom" | "phi" => {
                    Activation::GeluTanh
                }
                "falcon" | "mpt" => Activation::GeluErf,
                "opt" => Activation::Relu,
                "arcee" | "nemotron" | "nemotron_h" | "nemotron_h_moe" | "jais2" | "nanochat"
                | "persimmon" => Activation::ReluSquared,
                "gptneox" => Activation::GeluErf,
                _ => Activation::Silu,
            },
            gated_ffn,
            parallel_residual: key(meta, a, "use_parallel_residual")
                .and_then(Value::as_bool)
                .unwrap_or(matches!(a, "command-r" | "cohere" | "cohere2")),
            post_norms,
            attention_scale,
            attention_softcap: float(meta, a, "attn_logit_softcapping"),
            logit_softcap: float(meta, a, "final_logit_softcapping"),
            sliding_window,
            window_pattern,
            window_flags,
            embedding_scale: float(meta, a, "embedding_scale").unwrap_or(if gemma {
                (hidden as f32).sqrt()
            } else {
                1.0
            }),
            // Granite divides the logits, Cohere multiplies them.
            logit_scale: match (float(meta, a, "logit_scale"), a) {
                (Some(v), "granite" | "granitemoe" | "granitehybrid") if v != 0.0 => 1.0 / v,
                (Some(v), _) => v,
                (None, _) => 1.0,
            },
            residual_scale: float(meta, a, "residual_scale").unwrap_or(1.0),
            tied_embeddings: tied,
            moe: parse_moe(mapping, a, ff)?,
            position_offset: uint(meta, a, "position_embedding_offset").unwrap_or(0),
            alibi_slopes: match key(meta, a, "attention.alibi_slopes") {
                Some(Value::Array(values)) if values.len() == heads => values
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect(),
                _ => None,
            },
            qkv_clip: float(meta, a, "attention.clip_qkv"),
            embedding_norm: key(meta, a, "embedding_norm")
                .and_then(Value::as_bool)
                .unwrap_or(a == "muse-glimmer"),
            mla,
            shortconv: parse_shortconv(meta, a, layers)?,
            delta: parse_delta(meta, a),
            mamba: parse_mamba(meta, a),
            mamba1: (a == "mamba")
                .then(|| {
                    Some(Mamba1Config {
                        conv_kernel: uint(meta, a, "ssm.conv_kernel")?,
                        inner: uint(meta, a, "ssm.inner_size")?,
                        state: uint(meta, a, "ssm.state_size")?,
                        rank: uint(meta, a, "ssm.time_step_rank")?,
                        bc_rms_eps: float(meta, a, "ssm.dt_b_c_rms_eps"),
                    })
                })
                .flatten(),
            rope_layers: match key(meta, a, "rope.layer_flags") {
                Some(Value::Array(values)) if values.len() == layers => {
                    values.iter().map(Value::as_bool).collect()
                }
                _ => None,
            },
            qk_norm_after_rope: key(meta, a, "attention.qk_norm_after_rope")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            temperature_offset: uint(meta, a, "attention.temperature_offset").unwrap_or(0),
            temperature_nope_only: key(meta, a, "attention.temperature_nope_only")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            attention_chunk: uint(meta, a, "attention.chunk_size"),
            attn_gate: if is_k2(a) {
                AttnGate::Softplus(std::f32::consts::LN_2)
            } else {
                AttnGate::Sigmoid
            },
            value_experts: uint(meta, a, "attention.value_expert_count")
                .filter(|&n| n > 0)
                .map(|count| ValueExperts {
                    count,
                    used: uint(meta, a, "attention.value_expert_used_count").unwrap_or(1).clamp(1, count),
                }),
            attn_temperature: float(meta, a, "attention.temperature_beta").map(|beta| {
                (
                    beta,
                    uint(meta, a, "attention.temperature_context").unwrap_or(8192),
                )
            }),
        })
    }

    /// Chunked attention applies to the layers that rotate.
    pub fn layer_chunk(&self, layer: usize) -> Option<usize> {
        self.attention_chunk.filter(|_| self.layer_rotates(layer))
    }

    /// Whether `layer` applies rotary embeddings.
    pub fn layer_rotates(&self, layer: usize) -> bool {
        match &self.rope_layers {
            Some(flags) => flags.get(layer).copied().unwrap_or(true),
            None => match self.arch.as_str() {
                "smollm3" => !(layer + 1).is_multiple_of(4),
                // Nemotron-H attention has no positional encoding (llama.cpp files still carry rope.dimension_count)
                "nemotron_h" | "nemotron_h_moe" => false,
                // Muse Glimmer: the global-attention layers have no position encoding
                "muse-glimmer" if self.sliding_window.is_some() => self.layer_window(layer).is_some(),
                "exaone4" | "exaone_moe" | "cohere2" if self.sliding_window.is_some() => {
                    self.layer_window(layer).is_some()
                }
                _ => true,
            },
        }
    }

    /// Whether `layer` attends only to the last `sliding_window` positions.
    pub fn layer_window(&self, layer: usize) -> Option<usize> {
        let window = self.sliding_window?;
        if let Some(flags) = &self.window_flags {
            return flags.get(layer).copied().unwrap_or(false).then_some(window);
        }
        (self.window_pattern <= 1 || layer % self.window_pattern < self.window_pattern - 1)
            .then_some(window)
    }
}

pub(crate) fn is_k2(arch: &str) -> bool {
    matches!(arch, "k2_horizon" | "k2-horizon" | "k2horizon" | "k2")
}

fn parse_shortconv(
    meta: &Map<String, Value>,
    arch: &str,
    layers: usize,
) -> Result<Option<ShortConvConfig>, String> {
    if !matches!(arch, "lfm2" | "lfm2moe") {
        return Ok(None);
    }
    let kernel = uint(meta, arch, "shortconv.kernel").ok_or("missing LFM2 shortconv.kernel")?;
    if kernel == 0 {
        return Err("LFM2 shortconv.kernel must be positive".into());
    }
    let flags = key(meta, arch, "shortconv.layer_flags")
        .and_then(Value::as_array)
        .ok_or("missing LFM2 shortconv.layer_flags")?;
    let flags = flags
        .iter()
        .map(|v| {
            v.as_bool()
                .ok_or("LFM2 shortconv.layer_flags must be booleans")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if flags.len() != layers {
        return Err("LFM2 shortconv.layer_flags length differs from block_count".into());
    }
    Ok(Some(ShortConvConfig {
        kernel,
        layers: flags,
    }))
}

fn parse_mamba(meta: &Map<String, Value>, arch: &str) -> Option<MambaConfig> {
    if !matches!(arch, "mamba2" | "granitehybrid" | "bamba" | "nemotron_h" | "nemotron_h_moe") {
        return None;
    }
    let groups = uint(meta, arch, "ssm.group_count").unwrap_or(1).max(1);
    Some(MambaConfig {
        conv_kernel: uint(meta, arch, "ssm.conv_kernel")?,
        inner: uint(meta, arch, "ssm.inner_size")?,
        state: uint(meta, arch, "ssm.state_size")?,
        heads: uint(meta, arch, "ssm.time_step_rank")?,
        groups,
        norm_groups: uint(meta, arch, "ssm.norm_groups").unwrap_or(groups),
        step_min: float(meta, arch, "ssm.time_step_min").unwrap_or(0.0),
    })
}

fn parse_delta(meta: &Map<String, Value>, arch: &str) -> Option<DeltaConfig> {
    if !matches!(arch, "qwen3next" | "qwen35" | "qwen35moe" | "olmo_hybrid") {
        return None;
    }
    let v_heads = uint(meta, arch, "ssm.time_step_rank")?;
    Some(DeltaConfig {
        conv_kernel: uint(meta, arch, "ssm.conv_kernel")?,
        k_heads: uint(meta, arch, "ssm.group_count")?,
        v_heads,
        k_dim: uint(meta, arch, "ssm.state_size")?,
        v_dim: uint(meta, arch, "ssm.inner_size")? / v_heads.max(1),
        beta_scale: float(meta, arch, "ssm.beta_scale").unwrap_or(1.0),
        norm_eps: float(meta, arch, "ssm.norm_eps"),
    })
}

fn parse_moe(mapping: &MappedGguf, arch: &str, ff: usize) -> Result<Option<MoeConfig>, String> {
    let meta = &mapping.index().metadata;
    let experts = uint(meta, arch, "expert_count").unwrap_or(0);
    if experts == 0 {
        return Ok(None);
    }
    let used = uint(meta, arch, "expert_used_count").ok_or("missing expert_used_count")?;
    if used == 0 || used > experts {
        return Err(format!(
            "{arch}: invalid expert_used_count {used} of {experts}"
        ));
    }
    let expert_ff = uint(meta, arch, "expert_feed_forward_length").unwrap_or(ff);
    let shared_rows = mapping
        .index()
        .tensors
        .iter()
        .find(|t| t.name.ends_with(".ffn_up_shexp.weight"))
        .map(|t| t.dimensions.iter().skip(1).product::<u64>() as usize);
    let gating = match uint(meta, arch, "expert_gating_func") {
        Some(2) => Gating::Sigmoid,
        Some(3) => Gating::SparseMixer {
            jitter_eps: float(meta, arch, "expert_router_jitter_noise").unwrap_or(0.01),
        },
        // llama.cpp defaults K2 Horizon to sigmoid scores
        None if is_k2(arch) => Gating::Sigmoid,
        _ => Gating::Softmax,
    };
    // Mixtral-style (llama arch) models always renormalise; others say so in the metadata.
    let normalize = key(meta, arch, "expert_weights_norm")
        .and_then(Value::as_bool)
        .unwrap_or(matches!(arch, "llama" | "mistral" | "granitemoe" | "gpt-oss")); // gpt-oss: softmax over the selected logits
    let weight_norm_p = if arch == "dbrx" {
        float(meta, arch, "expert_weights_norm_p")
    } else {
        None
    };
    if weight_norm_p.is_some_and(|p| !p.is_finite() || p <= 0.0) {
        return Err("DBRX expert_weights_norm_p must be finite and positive".into());
    }
    Ok(Some(MoeConfig {
        experts,
        used,
        expert_ff,
        gating,
        normalize,
        weight_norm_p,
        weight_scale: float(meta, arch, "expert_weights_scale").unwrap_or(1.0),
        shared: uint(meta, arch, "expert_shared_count")
            .unwrap_or(usize::from(shared_rows.is_some())),
        shared_ff: shared_rows.unwrap_or(0),
        dense_layers: uint(meta, arch, "leading_dense_block_count").unwrap_or(0),
        groups: uint(meta, arch, "expert_group_count").unwrap_or(1),
        groups_used: uint(meta, arch, "expert_group_used_count").unwrap_or(1),
        input_scaled: key(meta, arch, "expert_weights_apply_to_input")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        group_top2: key(meta, arch, "expert_group_top2")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        latent: uint(meta, arch, "moe_latent_size").filter(|&n| n > 0),
        swiglu_oai: match (arch, float(meta, arch, "expert_swiglu_limit")) {
            (_, Some(limit)) => Some((
                float(meta, arch, "expert_swiglu_alpha").unwrap_or(1.702),
                limit,
            )),
            ("gpt-oss", None) => Some((1.702, 7.0)),
            _ => None,
        },
    }))
}

fn parse_scaling(meta: &Map<String, Value>, arch: &str) -> Result<RopeScaling, String> {
    let kind = key(meta, arch, "rope.scaling.type")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let factor = float(meta, arch, "rope.scaling.factor").unwrap_or(1.0);
    match kind {
        "none" => Ok(RopeScaling::None),
        "linear" if factor > 0.0 => Ok(RopeScaling::Linear { factor }),
        "longrope" => Ok(RopeScaling::LongRope {
            original_context: uint(meta, arch, "rope.scaling.original_context_length")
                .ok_or("longrope scaling needs rope.scaling.original_context_length")?,
            attention: float(meta, arch, "rope.scaling.attn_factor").unwrap_or(1.0),
            long: Vec::new(),
        }),
        "yarn" if factor > 0.0 => Ok(RopeScaling::Yarn {
            factor,
            original_context: uint(meta, arch, "rope.scaling.original_context_length")
                .ok_or("YaRN scaling needs rope.scaling.original_context_length")?,
            extrapolation: float(meta, arch, "rope.scaling.yarn_ext_factor").unwrap_or(1.0),
            attention: float(meta, arch, "rope.scaling.yarn_attn_factor").unwrap_or(1.0),
            beta_fast: float(meta, arch, "rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: float(meta, arch, "rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            truncate: key(meta, arch, "rope.scaling.yarn_truncate")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        }),
        other => Err(format!("unsupported rope scaling type {other:?}")),
    }
}

fn rope_factors(mapping: &MappedGguf, dims: usize, name: &str) -> Result<Option<Vec<f32>>, String> {
    if !has(mapping, name) {
        return Ok(None);
    }
    let tensor = crate::compute::Tensor::load(mapping, name)?;
    if tensor.data.len() != dims / 2 {
        return Err(format!(
            "{name} has {} entries, expected {}",
            tensor.data.len(),
            dims / 2
        ));
    }
    Ok(Some(tensor.data))
}
