//! Architecture plans: how a Hugging Face `config.json` and tensor names become GGUF metadata and
//! tensor names (llama.cpp conventions, so the files also load elsewhere).
use crate::gguf::Meta;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    Llama,
    Qwen,
    Gemma,
    Olmo2,
    Phi3,
    Deepseek,
    Cohere,
    Starcoder2,
    Neox,
    Glm,
    GptOss,
    Qwen3Next,
    Mamba,
    Nemotron,
    Classic,
}

/// Result of mapping one Hugging Face tensor name.
#[derive(Debug, PartialEq)]
pub(crate) enum Mapped {
    /// Not needed (rotary caches and similar buffers).
    Skip,
    Tensor {
        name: String,
        /// Constant added to every element (Gemma stores `1 + weight` for its RMS norms).
        add: f32,
    },
    /// A whole-layer fused expert tensor (GPT-OSS) that the converter splits itself.
    Fused { layer: u64, part: String },
    /// A matrix stored `[in, out]` (GPT-2's Conv1D) that is written `[out, in]`.
    Transposed { name: String },
    /// One expert's matrix; all experts of a layer are stacked into `name` in index order.
    Expert { name: String, index: usize },
}

pub(crate) struct Plan {
    pub arch: String,
    pub family: Family,
    pub meta: Vec<(String, Meta)>,
    pub norm_add: f32,
    pub layers: u64,
    /// Mixture-of-experts feed-forward layers.
    pub moe: bool,
    /// Gate and up projections (false: a single up projection).
    pub gated: bool,
    /// Which layers use experts (the others keep a dense feed-forward).
    pub moe_layers: Vec<bool>,
    /// DeepSeek: queries go through a low-rank bottleneck.
    pub q_lora: bool,
    /// Expert tensors use the Mixtral `w1/w2/w3` names.
    pub(crate) mixtral_names: bool,
    /// Synthesised tensors: name, f32 values.
    pub synthesized: Vec<(String, Vec<f32>)>,
    /// Gated DeltaNet layers (Qwen3-Next / Qwen3.5).
    pub linear: Option<LinearDims>,
    /// Nemotron-H layer kinds: `M` Mamba, `*` attention, `-` MLP, `E` experts.
    pub kinds: Vec<u8>,
    /// Older families with their own tensor names (GPT-2, Falcon, BLOOM, ...).
    pub classic: Option<crate::classic::Classic>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearDims {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
}

fn num(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::as_u64)
}
fn flt(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

/// RoPE settings for one layer kind, from either `rope_parameters` (transformers 5) or the older
/// `rope_theta` / `rope_scaling` pair.
pub(crate) struct RopeParams {
    pub theta: Option<f64>,
    pub kind: String,
    pub raw: Value,
}

pub(crate) fn rope_params(cfg: &Value, layer_type: Option<&str>) -> RopeParams {
    let mut raw = Value::Null;
    if let Some(p) = cfg.get("rope_parameters").filter(|v| v.is_object()) {
        raw = match layer_type.and_then(|t| p.get(t)).filter(|v| v.is_object()) {
            Some(per_layer) => per_layer.clone(),
            None if p.get("rope_type").is_some() || p.get("rope_theta").is_some() => p.clone(),
            None => p
                .as_object()
                .and_then(|m| m.values().find(|v| v.is_object()))
                .cloned()
                .unwrap_or(Value::Null),
        };
    }
    if raw.is_null() {
        raw = cfg
            .get("rope_scaling")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or(Value::Null);
    }
    let kind = raw
        .get("rope_type")
        .or_else(|| raw.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_ascii_lowercase();
    let theta = flt(&raw, "rope_theta").or_else(|| flt(cfg, "rope_theta"));
    RopeParams { theta, kind, raw }
}

/// Which Llama-4 layers apply rotary embeddings (the others are NoPE layers).
fn llama4_rope_flags(cfg: &Value, layers: u64) -> Vec<bool> {
    match cfg.get("no_rope_layers").and_then(Value::as_array) {
        Some(list) if list.len() == layers as usize => {
            list.iter().map(|v| v.as_u64() != Some(0)).collect()
        }
        _ => {
            let interval = num(cfg, "no_rope_layer_interval").unwrap_or(4).max(1);
            (0..layers).map(|i| (i + 1) % interval != 0).collect()
        }
    }
}

/// Replace the metadata entry `key` (or add it).
fn set_meta(meta: &mut Vec<(String, Meta)>, key: &str, value: Meta) {
    match meta.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => meta.push((key.to_owned(), value)),
    }
}

fn family_for(model_type: &str) -> Option<(Family, &'static str)> {
    Some(match model_type {
        "llama" => (Family::Llama, "llama"),
        "mistral" => (Family::Llama, "mistral"),
        // Ministral 8B: Mistral with a sliding/full attention pattern per layer (`layer_types`)
        "ministral" => (Family::Llama, "mistral"),
        "mixtral" => (Family::Llama, "llama"),
        "qwen2_moe" => (Family::Qwen, "qwen2moe"),
        "qwen3_moe" => (Family::Qwen, "qwen3moe"),
        "olmoe" => (Family::Qwen, "olmoe"),
        "qwen2" => (Family::Qwen, "qwen2"),
        "qwen3" => (Family::Qwen, "qwen3"),
        "k2_horizon" => (Family::Llama, "k2_horizon"),
        "g9v3" => (Family::Llama, "g9v3"),
        "gemma" => (Family::Gemma, "gemma"),
        "gemma2" => (Family::Gemma, "gemma2"),
        "gemma3_text" | "gemma3" => (Family::Gemma, "gemma3"),
        "gemma4_text" | "gemma4" => (Family::Gemma, "gemma4"),
        "gemma3n_text" | "gemma3n" => (Family::Gemma, "gemma3n"),
        "muse_glimmer" | "muse_glimmer_text" => (Family::Gemma, "muse-glimmer"),
        "vaultgemma" => (Family::Gemma, "vaultgemma"),
        "olmo2" => (Family::Olmo2, "olmo2"),
        "phi3" => (Family::Phi3, "phi3"),
        "phimoe" => (Family::Llama, "phimoe"),
        "granite" => (Family::Llama, "granite"),
        "granite_swa" => (Family::Llama, "granite"),
        "granitemoe_swa" => (Family::Llama, "granitemoe"),
        "deepseek_v2" | "deepseek_v3" | "kimi_k2" | "glm4_moe_lite" | "bailing_moe" | "bailing_hybrid" | "sarvam_mla" => (Family::Deepseek, "deepseek2"),
        // DeepSeek-V3.2 / GLM-5: MLA + MoE with a lightning indexer (sparse attention). The indexer tensors are dropped and attention is
        // dense, which equals the sparse result while the context is not longer than `index_topk`.
        "deepseek_v32" | "glm_moe_dsa" => (Family::Deepseek, "deepseek2"),
        "sarvam_moe" => (Family::Qwen, "qwen2moe"),
        "trinity" => (Family::Llama, "afmoe"),
        "mistral_large" => (Family::Llama, "mistral"),
        // Mistral Large 3 is a DeepSeek-V3-style model (MLA + MoE), not dense Llama.
        "mistral_large_3" => (Family::Deepseek, "deepseek2"),
        "internlm3" => (Family::Llama, "llama"),
        "mimo" => (Family::Qwen, "qwen2"),
        "solar" => (Family::Llama, "llama"),
        "orion" => (Family::Llama, "llama"),
        "iquestcoder" => (Family::Llama, "llama"),
        "xverse" => (Family::Llama, "llama"),
        "persimmon" => (Family::Llama, "persimmon"),
        "intellect" | "intellect_3" => (Family::Llama, "llama"),
        "cohere" => (Family::Cohere, "command-r"),
        "cohere2" => (Family::Cohere, "cohere2"),
        "starcoder2" => (Family::Starcoder2, "starcoder2"),
        "gpt_neox" => (Family::Neox, "gptneox"),
        "glm" => (Family::Glm, "glm"),
        "glm4" => (Family::Glm, "glm4"),
        "seed_oss" => (Family::Llama, "seed_oss"),
        "smollm3" => (Family::Llama, "smollm3"),
        "arcee" => (Family::Llama, "arcee"),
        "ministral3" => (Family::Llama, "ministral3"),
        "ernie4_5" => (Family::Llama, "ernie4_5"),
        "helium" => (Family::Llama, "helium"),
        "bitnet" => (Family::Llama, "bitnet"),
        "hunyuan_v1_dense" => (Family::Llama, "hunyuan-dense"),
        "exaone4" => (Family::Olmo2, "exaone4"),
        "olmo3" => (Family::Olmo2, "olmo3"),
        "gpt_oss" => (Family::GptOss, "gpt-oss"),
        "qwen3_next" => (Family::Qwen3Next, "qwen3next"),
        "mamba2" => (Family::Mamba, "mamba2"),
        "mamba" | "falcon_mamba" => (Family::Mamba, "mamba"),
        "nemotron_h" => (Family::Nemotron, "nemotron_h"),
        "granitemoehybrid" | "granite4" | "granite_4" => (Family::Llama, "granitehybrid"),
        "jetnemotron" => (Family::Nemotron, "nemotron_h"),
        "bamba" => (Family::Llama, "bamba"),
        "rwkv6" | "rwkv_6" => (Family::Mamba, "rwkv6"),
        "rwkv7" | "rwkv_7" => (Family::Mamba, "rwkv7"),
        "xlstm" | "xlstm_lm" => (Family::Mamba, "xlstm"),
        "hrm_text" | "hrm" => (Family::Llama, "hrm"),
        "zaya1" => (Family::Llama, "zaya1"),
        "falcon_h1" => (Family::Llama, "falcon-h1"),
        "jamba" => (Family::Llama, "jamba"),
        "solar_open" => (Family::Qwen, "glm4moe"),
        "jais2" => (Family::Llama, "jais2"),
        "nemotron" => (Family::Llama, "nemotron"),
        "internlm2" => (Family::Llama, "internlm2"),
        "cwm" => (Family::Llama, "cwm"),
        "hy_v3" => (Family::Llama, "hy_v3"),
        "minicpm3" | "minicpm_sala" => (Family::Deepseek, "minicpm3"),
        "kimi_linear" | "kimi_k3" => (Family::Deepseek, "kimi-linear"),
        "bailing_moe_v3" | "bailingmoev3" | "ling3" | "ling" => (Family::Deepseek, "kimi-linear"),
        "glm5_flash" | "glm_5_3_flash" | "glm5_next" => (Family::Deepseek, "kimi-linear"),
        "apertus" => (Family::Llama, "apertus"),
        "llama4_text" | "llama4" => (Family::Llama, "llama4"),
        "olmo_hybrid" => (Family::Llama, "olmo_hybrid"),
        "stablelm" => (Family::Llama, "stablelm"),
        "granitemoe" | "granitemoeshared" => (Family::Llama, "granitemoe"),
        "glm4v_text" => (Family::Glm, "glm4"),
        "glm4v_moe_text" => (Family::Qwen, "glm4moe"),
        "olmo" => (Family::Llama, "olmo"),
        "phi" => (Family::Llama, "phi"),
        "exaone_moe" => (Family::Qwen, "exaone_moe"),
        "afmoe" => (Family::Llama, "afmoe"),
        "hyperclovax" => (Family::Llama, "hyperclovax"),
        "flex_olmo" => (Family::Olmo2, "flex_olmo"),
        "nanochat" => (Family::Llama, "nanochat"),
        "youtu" => (Family::Deepseek, "deepseek2"),
        "mistral4" => (Family::Deepseek, "deepseek2"),
        "qwen2_vl_text" | "qwen2_5_vl_text" => (Family::Qwen, "qwen2"),
        "qwen3_vl_text" => (Family::Qwen, "qwen3"),
        "qwen3_vl_moe_text" => (Family::Qwen, "qwen3moe"),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => (Family::Qwen3Next, "qwen3next"),
        "glm4_moe" => (Family::Qwen, "glm4moe"),
        "dots1" => (Family::Qwen, "dots1"),
        "minimax_m2" => (Family::Qwen, "minimax-m2"),
        "ernie4_5_moe" => (Family::Llama, "ernie4_5moe"),
        "hunyuan_v1_moe" => (Family::Llama, "hunyuan-moe"),
        "deepseek" => (Family::Llama, "deepseek"),
        "deepseek_moe" => (Family::Qwen, "deepseek_moe"),
        "baichuan" | "baichuan2" => (Family::Llama, "baichuan"),
        "qwen" => (Family::Qwen, "qwen"),
        "exaone" | "exaone3" => (Family::Llama, "exaone"),
        "chatglm" => (Family::Glm, "chatglm"),
        "minicpm" => (Family::Llama, "minicpm"),
        "telechat" | "telechat2" | "teleflm" => (Family::Llama, "telechat"),
        "step1" => (Family::Llama, "step1"),
        "step3" | "step3_moe" => (Family::Qwen, "step3"),
        "plamo" | "plamo3" | "plamo_3" => (Family::Llama, "plamo"),
        "grok" | "grok1" | "grok_1" | "grok-1" => (Family::Llama, "grok"),
        "xverse_moe" => (Family::Llama, "xverse_moe"),
        "cohere2_moe" => (Family::Cohere, "cohere2_moe"),
        "jetmoe" => (Family::Qwen, "jetmoe"),
        "gigachat" | "gigachat3_5" => (Family::Llama, "gigachat"),
        "ax_k1" | "ax_k2" | "a_x" => (Family::Llama, "ax"),
        "hy_v4" => (Family::Llama, "hy_v4"),
        "pangu" => (Family::Llama, "pangu"),
        "param2_moe" => (Family::Qwen, "param2_moe"),
        "ling_moe" | "bailing" | "bailing_v2" => (Family::Deepseek, "deepseek2"),
        "mellum" => (Family::Qwen, "qwen3moe"),
        "laguna" => (Family::Qwen, "laguna"),
        "mimo_v2_flash" => (Family::Qwen, "mimo_v2"),
        _ => return None,
    })
}

/// DBRX nests its attention/expert dimensions and uses different top-level names.
pub(crate) fn dbrx_config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    for (target, source) in [
        ("hidden_size", "d_model"),
        ("num_hidden_layers", "n_layers"),
        ("num_attention_heads", "n_heads"),
        ("max_position_embeddings", "max_seq_len"),
    ] {
        out[target] = cfg
            .get(source)
            .cloned()
            .ok_or_else(|| format!("DBRX config requires {source}"))?;
    }
    let attn = cfg
        .get("attn_config")
        .ok_or("DBRX config requires attn_config")?;
    let ffn = cfg
        .get("ffn_config")
        .ok_or("DBRX config requires ffn_config")?;
    for (target, source) in [
        ("intermediate_size", "ffn_hidden_size"),
        ("num_experts", "moe_num_experts"),
        ("num_experts_per_tok", "moe_top_k"),
    ] {
        out[target] = ffn
            .get(source)
            .cloned()
            .ok_or_else(|| format!("DBRX ffn_config requires {source}"))?;
    }
    out["num_key_value_heads"] = attn.get("kv_n_heads").cloned().unwrap_or(1.into());
    out["hidden_act"] = ffn
        .get("ffn_act_fn")
        .and_then(|v| v.get("name"))
        .cloned()
        .unwrap_or("silu".into());
    // All three torch.nn.LayerNorm modules use the PyTorch default epsilon.
    out["layer_norm_eps"] = 1e-5.into();
    out["rope_theta"] = attn.get("rope_theta").cloned().unwrap_or(10000.0.into());
    Ok(out)
}

/// LFM2 adjusts the dense FFN width before creating its projections.
pub(crate) fn lfm2_config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    let width = cfg
        .get("block_ff_dim")
        .or_else(|| cfg.get("intermediate_size"))
        .and_then(Value::as_u64)
        .ok_or("LFM2 requires intermediate_size")?;
    let mut width = width;
    if cfg.get("model_type").and_then(Value::as_str) != Some("lfm2_moe")
        && cfg.get("block_auto_adjust_ff_dim").and_then(Value::as_bool) != Some(false)
    {
        width = width.checked_mul(2).ok_or("LFM2 FFN width overflow")? / 3;
        let multiplier = cfg.get("block_ffn_dim_multiplier");
        if multiplier.is_none_or(|v| !v.is_null()) {
            let multiplier = multiplier.and_then(Value::as_f64).unwrap_or(1.0);
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return Err("LFM2 block_ffn_dim_multiplier must be finite and positive".into());
            }
            let adjusted = multiplier * width as f64;
            if adjusted >= u32::MAX as f64 {
                return Err("LFM2 FFN width exceeds u32".into());
            }
            width = adjusted as u64;
            let multiple = num(cfg, "block_multiple_of").unwrap_or(256);
            if multiple == 0 {
                return Err("LFM2 block_multiple_of must be positive".into());
            }
            width = width
                .checked_add(multiple - 1)
                .ok_or("LFM2 FFN width overflow")?
                / multiple
                * multiple;
        }
    }
    if width == 0 || width > u32::MAX as u64 {
        return Err("invalid LFM2 FFN width".into());
    }
    out["intermediate_size"] = width.into();
    out["rms_norm_eps"] = cfg.get("norm_eps").cloned().unwrap_or(1e-5.into());
    out["tie_word_embeddings"] = cfg
        .get("tie_embedding")
        .or_else(|| cfg.get("tie_word_embeddings"))
        .cloned()
        .unwrap_or(true.into());
    Ok(out)
}

impl Plan {
    fn from_lfm2(cfg: &Value) -> Result<Self, String> {
        let mut normalized = lfm2_config(cfg)?;
        let moe = cfg.get("model_type").and_then(Value::as_str) == Some("lfm2_moe");
        let (family_arch, arch) = if moe {
            ("qwen3moe", "lfm2moe")
        } else {
            ("qwen2", "lfm2")
        };
        normalized["model_type"] = if moe { "qwen3_moe" } else { "qwen2" }.into();
        if moe {
            normalized["norm_topk_prob"] =
                cfg.get("norm_topk_prob").cloned().unwrap_or(true.into());
            if cfg.get("layer_types").and_then(Value::as_array).is_none() {
                return Err("LFM2-MoE requires an explicit layer_types list".into());
            }
        }
        let mut plan = Self::from_config(&normalized)?;
        plan.arch = arch.into();
        for (key, value) in &mut plan.meta {
            *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
            if key == "general.architecture"
                || (key == "general.name" && cfg.get("_name_or_path").is_none())
            {
                *value = Meta::Str(arch.into());
            }
        }
        let kinds = if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
            if types.len() != plan.layers as usize {
                return Err("LFM2 layer_types length differs from num_hidden_layers".into());
            }
            types
                .iter()
                .map(|kind| match kind.as_str() {
                    Some("full_attention") => Ok(b'A'),
                    Some("conv") => Ok(b'C'),
                    _ => Err(format!("unsupported LFM2 layer type {kind}")),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let indices = cfg.get("full_attn_idxs").and_then(Value::as_array);
            if let Some(indices) = indices {
                if indices
                    .iter()
                    .any(|i| i.as_u64().is_none_or(|n| n >= plan.layers))
                {
                    return Err("LFM2 full_attn_idxs contains an invalid layer index".into());
                }
            }
            (0..plan.layers)
                .map(|i| {
                    if indices.is_none_or(|list| list.iter().any(|v| v.as_u64() == Some(i))) {
                        b'A'
                    } else {
                        b'C'
                    }
                })
                .collect()
        };
        let kernel = num(cfg, "conv_L_cache").unwrap_or(3);
        if kernel == 0 || kernel > u32::MAX as u64 {
            return Err("LFM2 conv_L_cache must be a positive u32".into());
        }
        plan.meta
            .push((format!("{arch}.shortconv.kernel"), Meta::U32(kernel as u32)));
        plan.meta.push((
            format!("{arch}.shortconv.layer_flags"),
            Meta::Bools(kinds.iter().map(|&k| k == b'C').collect()),
        ));
        plan.kinds = kinds;
        if moe {
            let dense = num(cfg, "num_dense_layers").unwrap_or(2);
            if dense > plan.layers {
                return Err("LFM2-MoE num_dense_layers exceeds num_hidden_layers".into());
            }
            let scale = flt(cfg, "routed_scaling_factor").unwrap_or(1.0);
            if !scale.is_finite() {
                return Err("LFM2-MoE routed_scaling_factor must be finite".into());
            }
            plan.moe_layers = (0..plan.layers).map(|i| i >= dense).collect();
            plan.meta.push((
                format!("{arch}.leading_dense_block_count"),
                Meta::U32(dense as u32),
            ));
            plan.meta
                .push((format!("{arch}.expert_gating_func"), Meta::U32(2)));
            plan.meta.push((
                format!("{arch}.expert_weights_scale"),
                Meta::F32(scale as f32),
            ));
        }
        Ok(plan)
    }

    /// Nemotron-H: every layer is one mixer (Mamba-2, attention, MLP or experts).
    fn from_nemotron(cfg: &Value, arch: &str) -> Result<Self, String> {
        let get = |k: &str| num(cfg, k).ok_or_else(|| format!("config.json requires {k}"));
        let hidden = get("hidden_size")?;
        let kinds: Vec<u8> = if let Some(list) =
            cfg.get("layers_block_type").and_then(Value::as_array)
        {
            list.iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention" | "mamba") => Ok(b'M'),
                    Some("full_attention" | "attention") => Ok(b'*'),
                    Some("mlp") => Ok(b'-'),
                    Some("moe") => Ok(b'E'),
                    other => Err(format!("unsupported Nemotron-H layer type {other:?}")),
                })
                .collect::<Result<_, _>>()?
        } else if let Some(pattern) = cfg.get("hybrid_override_pattern").and_then(Value::as_str) {
            pattern.bytes().collect()
        } else {
            return Err(
                "Nemotron-H config needs layers_block_type or hybrid_override_pattern".into(),
            );
        };
        let layers = kinds.len() as u64;
        // llama.cpp names the expert variants (Nemotron 3 Nano/Super/Ultra) `nemotron_h_moe`
        let arch = if kinds.contains(&b'E') { "nemotron_h_moe" } else { arch };
        let heads = get("num_attention_heads")?;
        let kv = num(cfg, "num_key_value_heads").or_else(|| num(cfg, "multi_query_group_num")).unwrap_or(heads);
        let head_dim = num(cfg, "head_dim").or_else(|| num(cfg, "kv_channels")).unwrap_or(hidden / heads.max(1));
        let (m_heads, m_dim) = (get("mamba_num_heads")?, get("mamba_head_dim")?);
        let eps = flt(cfg, "layer_norm_epsilon").unwrap_or(1e-5) as f32;
        let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
        let mut meta = vec![
            ("general.architecture".to_owned(), Meta::Str(arch.into())),
            ("general.name".to_owned(), Meta::Str(arch.into())),
            ("general.alignment".to_owned(), Meta::U32(32)),
            u(
                "context_length",
                num(cfg, "max_position_embeddings").unwrap_or(1 << 20),
            ),
            u("embedding_length", hidden),
            u("block_count", layers),
            u(
                "feed_forward_length",
                num(cfg, "intermediate_size").unwrap_or(0),
            ),
            u("attention.head_count", heads),
            u("attention.head_count_kv", kv),
            u("attention.key_length", head_dim),
            u("attention.value_length", head_dim),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                Meta::F32(eps),
            ),
            (
                format!("{arch}.rope.layer_flags"),
                Meta::Bools(vec![false; layers as usize]),
            ),
            u("ssm.conv_kernel", num(cfg, "conv_kernel").unwrap_or(4)),
            u("ssm.inner_size", m_heads * m_dim),
            u("ssm.state_size", get("ssm_state_size")?),
            u("ssm.time_step_rank", m_heads),
            u("ssm.group_count", num(cfg, "n_groups").unwrap_or(1)),
            u("ssm.norm_groups", num(cfg, "n_groups").unwrap_or(1)),
            (
                format!("{arch}.ssm.time_step_min"),
                Meta::F32(flt(cfg, "time_step_min").unwrap_or(0.0) as f32),
            ),
        ];
        if let Some(nextn) = num(cfg, "num_nextn_predict_layers").or_else(|| num(cfg, "num_mtp_layers")).filter(|&n| n > 0) {
            meta.push(u("nextn_predict_layers", nextn));
        }
        let moe = kinds.contains(&b'E');
        if moe {
            meta.push(u("expert_count", get("n_routed_experts")?));
            meta.push(u("expert_used_count", get("num_experts_per_tok")?));
            if let Some(latent) = num(cfg, "moe_latent_size").filter(|&n| n > 0) {
                meta.push(u("moe_latent_size", latent));
            }
            meta.push(u(
                "expert_feed_forward_length",
                get("moe_intermediate_size")?,
            ));
            meta.push(u(
                "expert_shared_feed_forward_length",
                num(cfg, "moe_shared_expert_intermediate_size").unwrap_or(0),
            ));
            meta.push(u("expert_gating_func", 2));
            if let Some(g) = num(cfg, "n_group").filter(|&g| g > 1) {
                meta.push(u("expert_group_count", g));
                meta.push(u(
                    "expert_group_used_count",
                    num(cfg, "topk_group").unwrap_or(g),
                ));
            }
            meta.push((
                format!("{arch}.expert_weights_scale"),
                Meta::F32(flt(cfg, "routed_scaling_factor").unwrap_or(1.0) as f32),
            ));
            meta.push((
                format!("{arch}.expert_weights_norm"),
                Meta::Bool(
                    cfg.get("norm_topk_prob")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                ),
            ));
        }
        Ok(Self {
            arch: arch.to_owned(),
            family: Family::Nemotron,
            meta,
            norm_add: 0.0,
            layers,
            moe,
            gated: false,
            moe_layers: kinds.iter().map(|&k| k == b'E').collect(),
            q_lora: false,
            mixtral_names: false,
            synthesized: Vec::new(),
            linear: Some(LinearDims {
                k_heads: 1,
                v_heads: 1,
                k_dim: 1,
                v_dim: 1,
            }),
            kinds,
            classic: None,
        })
    }

    /// RWKV-6 ("Finch") and RWKV-7 ("Goose"): time-mix and channel-mix blocks with a matrix-valued state per head. The metadata
    /// keys are the ones llama.cpp writes (`wkv.head_size`, `rescale_every_n_layers`) plus `attention.group_norm_epsilon`.
    fn from_rwkv(cfg: &Value, arch: &str) -> Result<Self, String> {
        let get = |k: &str| num(cfg, k).ok_or_else(|| format!("config.json requires {k}"));
        let hidden = get("hidden_size")?;
        let layers = get("num_hidden_layers")?;
        let v7 = arch == "rwkv7";
        let head = if v7 {
            num(cfg, "head_dim").or_else(|| num(cfg, "num_heads").filter(|&h| h > 0).map(|h| hidden / h)).unwrap_or(64)
        } else {
            num(cfg, "head_size").unwrap_or(64)
        };
        if head == 0 || hidden % head != 0 {
            return Err(format!("RWKV hidden size {hidden} is not a multiple of the head size {head}"));
        }
        let ff = match num(cfg, "intermediate_size") {
            Some(n) if n > 0 => n,
            _ if v7 => (flt(cfg, "hidden_ratio").unwrap_or(4.0) * hidden as f64) as u64,
            _ => (hidden as f64 * 3.5) as u64 / 32 * 32,
        };
        let eps = flt(cfg, if v7 { "norm_eps" } else { "layer_norm_epsilon" }).unwrap_or(1e-5) as f32;
        // RWKV-7 (flash-linear-attention) normalises each head with eps = head_dim * norm_eps; RWKV-6 with 1e-5 * head_size_divisor^2
        let group_eps = if v7 { head as f32 * eps } else { 1e-5 * (flt(cfg, "head_size_divisor").unwrap_or(8.0) as f32).powi(2) };
        let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
        let meta = vec![
            ("general.architecture".to_owned(), Meta::Str(arch.into())),
            ("general.name".to_owned(), Meta::Str(arch.into())),
            ("general.alignment".to_owned(), Meta::U32(32)),
            u("context_length", 1 << 20),
            u("embedding_length", hidden),
            u("block_count", layers),
            u("feed_forward_length", ff),
            u("wkv.head_size", head),
            u("attention.head_count", 0),
            (format!("{arch}.attention.layer_norm_epsilon"), Meta::F32(eps)),
            (format!("{arch}.attention.group_norm_epsilon"), Meta::F32(group_eps)),
        ];
        Ok(Self {
            arch: arch.to_owned(),
            family: Family::Mamba,
            meta,
            norm_add: 0.0,
            layers,
            moe: false,
            gated: false,
            moe_layers: vec![false; layers as usize],
            q_lora: false,
            mixtral_names: false,
            synthesized: Vec::new(),
            linear: Some(LinearDims { k_heads: 1, v_heads: 1, k_dim: 1, v_dim: 1 }),
            kinds: Vec::new(),
            classic: None,
        })
    }

    /// Tensor names of an RWKV-7 (flash-linear-attention layout) or RWKV-6 (`Rwkv6ForCausalLM` layout) checkpoint.
    fn map_rwkv(&self, n: &str) -> Result<Mapped, String> {
        let plain = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
        let v7 = self.arch == "rwkv7";
        // the embedding, final norm, head and the norm in front of layer 0 (applied to the embeddings)
        let (emb, out_norm, head, pre) = if v7 {
            ("model.embeddings.weight", "model.norm", "lm_head.weight", "model.layers.0.pre_norm")
        } else {
            ("rwkv.embeddings.weight", "rwkv.ln_out", "head.weight", "rwkv.blocks.0.pre_ln")
        };
        if n == emb {
            return plain("token_embd.weight".into());
        }
        if n == head {
            return plain("output.weight".into());
        }
        for (prefix, target) in [(out_norm, "output_norm"), (pre, "token_embd_norm")] {
            if let Some(kind) = n.strip_prefix(prefix).and_then(|r| r.strip_prefix('.')) {
                if matches!(kind, "weight" | "bias") {
                    return plain(format!("{target}.{kind}"));
                }
            }
        }
        let rest = n
            .strip_prefix(if v7 { "model.layers." } else { "rwkv.blocks." })
            .ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
        let (i, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
        if i.parse::<u64>().map_or(true, |i| i >= self.layers) {
            return Err(format!("tensor {n:?} refers to an invalid layer"));
        }
        let to = |name: &str| plain(format!("blk.{i}.{name}"));
        let transposed = |name: &str| Ok(Mapped::Transposed { name: format!("blk.{i}.{name}") });
        let (stem, kind) = suf.rsplit_once('.').unwrap_or((suf, ""));
        let norm = |base: &str| to(&format!("{base}.{kind}"));
        if v7 {
            return match stem {
                "attn_norm" => norm("attn_norm"),
                "ffn_norm" => norm("attn_norm_2"),
                "attn.g_norm" => norm("time_mix_ln"),
                "attn.r_proj" if kind == "weight" => to("time_mix_receptance.weight"),
                "attn.k_proj" if kind == "weight" => to("time_mix_key.weight"),
                "attn.v_proj" if kind == "weight" => to("time_mix_value.weight"),
                "attn.o_proj" if kind == "weight" => to("time_mix_output.weight"),
                "ffn.key" if kind == "weight" => to("channel_mix_key.weight"),
                "ffn.value" if kind == "weight" => to("channel_mix_value.weight"),
                _ => {
                    let fixed = match suf {
                        "attn.k_k" => "time_mix_k_k.weight",
                        "attn.k_a" => "time_mix_k_a.weight",
                        "attn.r_k" => "time_mix_r_k.weight",
                        "ffn.x_k" => "channel_mix_lerp_k.weight",
                        "attn.x_r" => "time_mix_lerp_r.weight",
                        "attn.x_w" => "time_mix_lerp_w.weight",
                        "attn.x_k" => "time_mix_lerp_k.weight",
                        "attn.x_v" => "time_mix_lerp_v.weight",
                        "attn.x_a" => "time_mix_lerp_a.weight",
                        "attn.x_g" => "time_mix_lerp_g.weight",
                        _ => "",
                    };
                    if !fixed.is_empty() {
                        return to(fixed);
                    }
                    // low-rank adapters: lora.0 is the down projection, lora.2 the up projection, its bias the constant term
                    for lora in ["w", "a", "v", "g"] {
                        if let Some(r) = suf.strip_prefix(format!("attn.{lora}_lora.lora.").as_str()) {
                            return match r {
                                "0.weight" => to(&format!("time_mix_{lora}1.weight")),
                                "2.weight" => to(&format!("time_mix_{lora}2.weight")),
                                "2.bias" => to(&format!("time_mix_{lora}0.weight")),
                                _ => Err(format!("unsupported forward-affecting layer tensor {n:?}")),
                            };
                        }
                    }
                    Err(format!("unsupported forward-affecting layer tensor {n:?}"))
                }
            };
        }
        match stem {
            "ln1" => norm("attn_norm"),
            "ln2" => norm("attn_norm_2"),
            "attention.ln_x" => norm("time_mix_ln"),
            "attention.receptance" if kind == "weight" => to("time_mix_receptance.weight"),
            "attention.key" if kind == "weight" => to("time_mix_key.weight"),
            "attention.value" if kind == "weight" => to("time_mix_value.weight"),
            "attention.gate" if kind == "weight" => to("time_mix_gate.weight"),
            "attention.output" if kind == "weight" => to("time_mix_output.weight"),
            "feed_forward.key" if kind == "weight" => to("channel_mix_key.weight"),
            "feed_forward.receptance" if kind == "weight" => to("channel_mix_receptance.weight"),
            "feed_forward.value" if kind == "weight" => to("channel_mix_value.weight"),
            _ => match suf {
                "attention.time_maa_x" => to("time_mix_lerp_x.weight"),
                "attention.time_maa_w" => to("time_mix_lerp_w.weight"),
                "attention.time_maa_k" => to("time_mix_lerp_k.weight"),
                "attention.time_maa_v" => to("time_mix_lerp_v.weight"),
                "attention.time_maa_r" => to("time_mix_lerp_r.weight"),
                "attention.time_maa_g" => to("time_mix_lerp_g.weight"),
                "attention.time_maa_w1" => transposed("time_mix_w1.weight"),
                "attention.time_maa_w2" => transposed("time_mix_w2.weight"),
                "attention.time_decay" => to("time_mix_decay.weight"),
                "attention.time_decay_w1" => transposed("time_mix_decay_w1.weight"),
                "attention.time_decay_w2" => transposed("time_mix_decay_w2.weight"),
                "attention.time_faaaa" => to("time_mix_first.weight"),
                "feed_forward.time_maa_k" => to("channel_mix_lerp_k.weight"),
                "feed_forward.time_maa_r" => to("channel_mix_lerp_r.weight"),
                _ => Err(format!("unsupported forward-affecting layer tensor {n:?}")),
            },
        }
    }

    /// Pure Mamba-2 stacks: no attention, no feed-forward.
    /// Mamba-1 (and Falcon-Mamba): per-channel state, low-rank time step.
    fn from_mamba1(cfg: &Value, arch: &str) -> Result<Self, String> {
        let get = |k: &str| num(cfg, k).ok_or_else(|| format!("config.json requires {k}"));
        let hidden = get("hidden_size")?;
        let layers = get("num_hidden_layers")?;
        let inner =
            num(cfg, "intermediate_size").unwrap_or(hidden * num(cfg, "expand").unwrap_or(2));
        let rank = num(cfg, "time_step_rank").unwrap_or(hidden.div_ceil(16));
        let eps = flt(cfg, "layer_norm_epsilon").unwrap_or(1e-5) as f32;
        let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
        let mut meta = vec![
            ("general.architecture".to_owned(), Meta::Str(arch.into())),
            ("general.name".to_owned(), Meta::Str(arch.into())),
            ("general.alignment".to_owned(), Meta::U32(32)),
            u("context_length", 1 << 20),
            u("embedding_length", hidden),
            u("block_count", layers),
            u("feed_forward_length", 0),
            u("ssm.conv_kernel", num(cfg, "conv_kernel").unwrap_or(4)),
            u("ssm.inner_size", inner),
            u("ssm.state_size", get("state_size")?),
            u("ssm.time_step_rank", rank),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                Meta::F32(eps),
            ),
        ];
        if cfg.get("mixer_rms_eps").is_some() {
            // Falcon-Mamba normalises the time step, B and C
            meta.push((
                format!("{arch}.ssm.dt_b_c_rms_eps"),
                Meta::F32(flt(cfg, "mixer_rms_eps").unwrap_or(1e-6) as f32),
            ));
        }
        Ok(Self {
            arch: arch.to_owned(),
            family: Family::Mamba,
            meta,
            norm_add: 0.0,
            layers,
            moe: false,
            gated: false,
            moe_layers: vec![false; layers as usize],
            q_lora: false,
            mixtral_names: false,
            synthesized: Vec::new(),
            linear: Some(LinearDims {
                k_heads: 1,
                v_heads: 1,
                k_dim: 1,
                v_dim: 1,
            }),
            kinds: Vec::new(),
            classic: None,
        })
    }

    fn from_mamba(cfg: &Value, arch: &str) -> Result<Self, String> {
        if arch == "mamba" {
            return Self::from_mamba1(cfg, arch);
        }
        if matches!(arch, "rwkv6" | "rwkv7") {
            return Self::from_rwkv(cfg, arch);
        }
        let get = |k: &str| num(cfg, k).ok_or_else(|| format!("config.json requires {k}"));
        let hidden = get("hidden_size")?;
        let layers = get("num_hidden_layers")?;
        let heads = get("num_heads")?;
        let head_dim = get("head_dim")?;
        let expand = num(cfg, "expand").unwrap_or(2);
        if hidden * expand != heads * head_dim {
            return Err("inconsistent Mamba-2 dimensions".into());
        }
        let eps = flt(cfg, "layer_norm_epsilon").unwrap_or(1e-5) as f32;
        let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
        let meta = vec![
            ("general.architecture".to_owned(), Meta::Str(arch.into())),
            ("general.name".to_owned(), Meta::Str(arch.into())),
            ("general.alignment".to_owned(), Meta::U32(32)),
            u("context_length", 1 << 20),
            u("embedding_length", hidden),
            u("block_count", layers),
            u("feed_forward_length", 0),
            u("ssm.conv_kernel", num(cfg, "conv_kernel").unwrap_or(4)),
            u("ssm.inner_size", heads * head_dim),
            u("ssm.state_size", get("state_size")?),
            u("ssm.time_step_rank", heads),
            u("ssm.group_count", num(cfg, "n_groups").unwrap_or(1)),
            // transformers normalises over the whole inner width
            u("ssm.norm_groups", 1),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                Meta::F32(eps),
            ),
        ];
        Ok(Self {
            arch: arch.to_owned(),
            family: Family::Mamba,
            meta,
            norm_add: 0.0,
            layers,
            moe: false,
            gated: false,
            moe_layers: vec![false; layers as usize],
            q_lora: false,
            mixtral_names: false,
            synthesized: Vec::new(),
            linear: Some(LinearDims {
                k_heads: 1,
                v_heads: 1,
                k_dim: 1,
                v_dim: 1,
            }),
            kinds: Vec::new(),
            classic: None,
        })
    }

    pub(crate) fn from_config(cfg: &Value) -> Result<Self, String> {
        let raw = cfg
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or("config.json requires model_type")?
            .to_ascii_lowercase();
        if matches!(raw.as_str(), "lfm2" | "lfm2_moe") {
            return Self::from_lfm2(cfg);
        }
        if raw == "dbrx" {
            let mut normalized = dbrx_config(cfg)?;
            normalized["model_type"] = "qwen3_moe".into();
            normalized["norm_topk_prob"] = false.into();
            let mut plan = Self::from_config(&normalized)?;
            plan.arch = "dbrx".into();
            for (key, value) in &mut plan.meta {
                *key = key.replace("qwen3moe.", "dbrx.");
                if key == "general.architecture"
                    || (key == "general.name" && cfg.get("_name_or_path").is_none())
                {
                    *value = Meta::Str("dbrx".into());
                }
            }
            if let Some(clip) = flt(&cfg["attn_config"], "clip_qkv") {
                if !clip.is_finite() || clip <= 0.0 {
                    return Err("DBRX clip_qkv must be finite and positive".into());
                }
                plan.meta
                    .push(("dbrx.attention.clip_qkv".into(), Meta::F32(clip as f32)));
            }
            let p = cfg["ffn_config"].get("moe_normalize_expert_weights");
            if let Some(p) = p.filter(|v| !v.is_null()) {
                let p = p
                    .as_f64()
                    .ok_or("DBRX router p-norm must be numeric or null")?;
                if !p.is_finite() || p <= 0.0 {
                    return Err("DBRX router p-norm must be finite and positive".into());
                }
                plan.meta
                    .push(("dbrx.expert_weights_norm_p".into(), Meta::F32(p as f32)));
            } else if p.is_none() {
                plan.meta
                    .push(("dbrx.expert_weights_norm_p".into(), Meta::F32(1.0)));
            }
            return Ok(plan);
        }
        if raw == "zamba2" {
            return crate::zamba2::plan(cfg);
        }
        if raw == "deepseek_v4" {
            return crate::deepseek4::plan(cfg);
        }
        if matches!(raw.as_str(), "kimi_linear" | "kimi_k3" | "bailing_moe_v3" | "bailingmoev3" | "ling3" | "ling" | "glm5_flash" | "glm_5_3_flash" | "glm5_next") {
            return crate::kimi_linear::plan(cfg);
        }
        if matches!(raw.as_str(), "step3p5" | "step3p7_text") {
            // Step-3.5 / 3.7 text: the Laguna layer with `weight - 1` norms
            let mut plan = Self::from_config(&crate::step3p5::config(cfg)?)?;
            plan.norm_add = 1.0;
            plan.meta.push((format!("{}.attention.gate_sigmoid", plan.arch), Meta::Bool(true)));
            return Ok(plan);
        }
        if let Some(plan) = crate::classic::plan_for(&raw, cfg)? {
            return Ok(plan);
        }
        if matches!(raw.as_str(), "gemma4" | "gemma4_text") {
            return crate::gemma::plan(cfg);
        }
        if matches!(raw.as_str(), "gemma3n" | "gemma3n_text") {
            return crate::gemma3n::plan(cfg);
        }
        let (family, arch) = if matches!(raw.as_str(), "bailing_moe" | "bailing_hybrid" | "ling" | "ling_moe" | "bailing" | "bailing_v2") && num(cfg, "kv_lora_rank").is_none() {
            (Family::Qwen, "bailing_moe")
        } else if raw == "xverse" && num(cfg, "num_experts").unwrap_or(0) > 0 {
            (Family::Llama, "xverse_moe")
        } else {
            family_for(&raw).ok_or_else(|| {
                format!("unsupported model_type {raw:?}; refusing fallback architecture mapping")
            })?
        };
        if family == Family::Mamba {
            return Self::from_mamba(cfg, arch);
        }
        if family == Family::Nemotron {
            return Self::from_nemotron(cfg, arch);
        }
        let experts = [
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
        ]
        .iter()
        .find_map(|k| num(cfg, k))
        .unwrap_or(0);
        let moe = experts > 0;
        let known_moe = matches!(
                    raw.as_str(),
                    "mixtral"
                        | "qwen2_moe"
                        | "qwen3_moe"
                        | "olmoe"
                        | "deepseek_v2"
                        | "deepseek_v3"
                        | "kimi_k2"
                        | "glm4_moe_lite"
                        | "deepseek_v32"
                        | "glm_moe_dsa"
                        | "bailing_moe"
                        | "bailing_hybrid"
                        | "sarvam_mla"
                        | "sarvam_moe"
                        | "trinity"
                        | "glm4_moe"
                        | "dots1"
                        | "minimax_m2"
                        | "ernie4_5_moe"
                        | "hunyuan_v1_moe"
                        | "mellum"
                        | "laguna"
                        | "mimo_v2_flash"
                        | "gpt_oss"
                        | "qwen3_next"
                        | "qwen3_5_moe"
                        | "qwen3_5_moe_text"
                        | "cohere2_moe"
                        | "xverse_moe"
                        | "solar_open"
                        | "hy_v3"
                        | "exaone_moe"
                        | "afmoe"
                        | "flex_olmo"
                        | "llama4_text"
                        | "llama4"
                        | "granitemoe"
                        | "granitemoeshared"
                        | "granitemoe_swa"
                        | "glm4v_moe_text"
                        | "mistral4"
                        | "qwen3_vl_moe_text"
                        | "phimoe"
                        | "k2_horizon"
                        | "g9v3"
                        | "deepseek_moe"
                        | "step3"
                        | "step3_moe"
                        | "grok"
                        | "grok1"
                        | "grok_1"
                        | "grok-1"
                                                | "jetmoe"
                        | "param2_moe"
                        | "ling_moe"
                        | "jamba"
                ) || (matches!(raw.as_str(), "deepseek" | "xverse") && moe);
        if raw != "granitemoehybrid" && raw != "k2_horizon" && moe != known_moe {
            return Err(format!(
                "{raw}: expert configuration does not match the architecture"
            ));
        }
        if moe
            && raw != "k2_horizon"
            && raw != "deepseek"
            && (num(cfg, "decoder_sparse_step").unwrap_or(1) != 1
                || cfg
                    .get("mlp_only_layers")
                    .and_then(Value::as_array)
                    .is_some_and(|l| !l.is_empty()))
        {
            return Err("MoE models with dense layers mixed in are unsupported".into());
        }
        let hidden = num(cfg, "hidden_size").ok_or("config.json requires valid hidden_size")?;
        let heads = num(cfg, "num_attention_heads")
            .ok_or("config.json requires valid num_attention_heads")?;
        let layers =
            num(cfg, "num_hidden_layers").or_else(|| num(cfg, "num_layers")).ok_or("config.json requires valid num_hidden_layers")?;
        let inter = num(cfg, "intermediate_size")
            .or_else(|| num(cfg, "ffn_hidden_size"))
            .ok_or("config.json requires valid intermediate_size")?;
        let inter = if arch == "qwen" { inter / 2 } else { inter };
        // Persimmon is multi-head only: the HF reference ignores `num_key_value_heads`.
        let kv_heads = if raw == "persimmon" {
            heads
        } else {
            num(cfg, "num_key_value_heads").or_else(|| num(cfg, "multi_query_group_num")).unwrap_or(heads)
        };
        let ctx = num(cfg, "max_position_embeddings").or_else(|| num(cfg, "seq_length")).unwrap_or(40960);
        let (nope, rope_dims, value_dim) = (
            num(cfg, "qk_nope_head_dim").unwrap_or(0),
            num(cfg, "qk_rope_head_dim").unwrap_or(0),
            num(cfg, "v_head_dim")
                .filter(|&v| v > 0)
                .unwrap_or(num(cfg, "qk_nope_head_dim").unwrap_or(0)),
        );
        let mla = family == Family::Deepseek;
        if mla
            && (nope == 0 || rope_dims == 0 || value_dim == 0 || num(cfg, "kv_lora_rank").is_none())
        {
            return Err("DeepSeek config requires qk_nope_head_dim, qk_rope_head_dim, v_head_dim and kv_lora_rank".into());
        }
        let head_dim = if mla {
            nope + rope_dims
        } else {
            num(cfg, "head_dim").or_else(|| num(cfg, "kv_channels")).unwrap_or(hidden / heads.max(1))
        };
        let value_dim = if mla { value_dim } else { head_dim };
        if hidden == 0 || heads == 0 || layers == 0 || inter == 0 || kv_heads == 0 {
            return Err("invalid or inconsistent model dimensions in config.json".into());
        }
        if !mla && num(cfg, "head_dim").is_none() && !hidden.is_multiple_of(heads) {
            return Err("invalid or inconsistent model dimensions in config.json".into());
        }
        for (key, value) in [
            ("hidden_size", hidden),
            ("num_attention_heads", heads),
            ("num_hidden_layers", layers),
            ("intermediate_size", inter),
            ("num_key_value_heads", kv_heads),
            ("max_position_embeddings", ctx),
            ("head_dim", head_dim),
        ] {
            if value > u64::from(u32::MAX) {
                return Err(format!(
                    "config dimension {key} exceeds GGUF's u32 metadata range"
                ));
            }
        }
        heads
            .checked_mul(head_dim)
            .ok_or("attention head dimension overflow")?;
        kv_heads
            .checked_mul(head_dim)
            .ok_or("key/value head dimension overflow")?;
        let activation = cfg
            .get(if family == Family::Gemma {
                "hidden_activation"
            } else {
                "hidden_act"
            })
            .or_else(|| cfg.get("hidden_act"))
            .and_then(Value::as_str)
            .unwrap_or(match family {
                _ if matches!(arch, "arcee" | "bitnet") => "relu2",
                _ if matches!(arch, "jais2" | "nanochat" | "nemotron") => "relu2",
                _ if arch == "persimmon" => "relu2",
                _ if arch == "apertus" => "xielu",
                _ if arch == "phi" => "gelu_new",
                _ if arch == "muse-glimmer" => "silu",
                Family::Gemma | Family::Starcoder2 => "gelu_pytorch_tanh",
                Family::Neox => "gelu",
                _ => "silu",
            })
            .to_ascii_lowercase();
        let wanted: &[&str] = match family {
            _ if matches!(arch, "arcee" | "bitnet" | "jais2" | "nanochat" | "nemotron" | "persimmon") => {
                &["relu2"]
            }
            _ if arch == "apertus" => &["xielu"],
            _ if arch == "phi" => &["gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu"],
            _ if arch == "muse-glimmer" => &["silu"],
            Family::Gemma => &["gelu_pytorch_tanh", "gelu_new", "gelu_fast", "gelu"],
            Family::Starcoder2 => &["gelu_pytorch_tanh", "gelu_new", "gelu_fast"],
            Family::Neox => &["gelu"],
            _ => &["silu", "swish"],
        };
        if !wanted.contains(&activation.as_str()) {
            return Err(format!(
                "unsupported activation {activation:?}; {arch} uses {}",
                wanted[0]
            ));
        }

        let mut meta: Vec<(String, Meta)> = vec![
            ("general.architecture".into(), Meta::Str(arch.into())),
            (
                "general.name".into(),
                Meta::Str(
                    cfg.get("_name_or_path")
                        .and_then(Value::as_str)
                        .unwrap_or(arch)
                        .into(),
                ),
            ),
            ("general.alignment".into(), Meta::U32(32)),
        ];
        macro_rules! u {
            ($k:expr, $v:expr) => {
                meta.push((format!("{arch}.{}", $k), Meta::U32($v as u32)))
            };
        }
        macro_rules! f {
            ($k:expr, $v:expr) => {
                meta.push((format!("{arch}.{}", $k), Meta::F32($v as f32)))
            };
        }
        u!("context_length", ctx);
        u!("embedding_length", hidden);
        u!("block_count", layers);
        if let Some(nextn) = num(cfg, "num_nextn_predict_layers").or_else(|| num(cfg, "num_mtp_layers")).filter(|&n| n > 0 && raw != "mimo") {
            u!("nextn_predict_layers", nextn);
        }
        u!("feed_forward_length", inter);
        u!("attention.head_count", heads);
        u!("attention.head_count_kv", kv_heads);
        u!("attention.key_length", head_dim);
        u!("attention.value_length", value_dim);
        if mla {
            u!("attention.key_length_mla", head_dim);
            u!("attention.value_length_mla", value_dim);
            u!(
                "attention.kv_lora_rank",
                num(cfg, "kv_lora_rank").unwrap_or(0)
            );
            if let Some(rank) = num(cfg, "q_lora_rank").filter(|&r| r > 0) {
                u!("attention.q_lora_rank", rank);
            }
            u!("rope.dimension_count", rope_dims);
            // DeepSeek sparse attention (DeepSeek-V3.2, GLM-5): the lightning indexer of the layers
            if let (Some(h), Some(d), Some(k)) = (num(cfg, "index_n_heads"), num(cfg, "index_head_dim"), num(cfg, "index_topk")) {
                u!("attention.indexer.head_count", h);
                u!("attention.indexer.key_length", d);
                u!("attention.indexer.top_k", k);
                // DeepSeek-V3.2 rotates half-split pairs in the indexer, GLM-5 interleaved pairs
                meta.push((format!("{arch}.attention.indexer.rope_neox"), Meta::Bool(raw == "deepseek_v32")));
            }
        }
        let eps = [
            "rms_norm_eps",
            "layer_norm_eps",
            "norm_epsilon",
            "layer_norm_epsilon",
            "norm_eps",
            "layernorm_epsilon",
        ]
        .iter()
        .find_map(|k| flt(cfg, k))
        .unwrap_or(if arch == "olmo" { 1e-5 } else { 1e-6 });
        if !eps.is_finite() {
            return Err("invalid rms_norm_eps".into());
        }
        f!("attention.layer_norm_rms_epsilon", eps);

        // sliding window attention
        let layer_types: Vec<String> = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let window = num(cfg, "sliding_window").filter(|&w| w > 0);
        let use_window = cfg.get("use_sliding_window").and_then(Value::as_bool) != Some(false)
            // Solar-pro carries a `sliding_window` key for compatibility but attends
            // densely, so the window pattern must not be applied.
            && raw != "solar";
        let mut sliding_layers: Option<Vec<bool>> = None;
        if let (Some(window), true) = (window, use_window) {
            if matches!(raw.as_str(), "qwen2" | "qwen3") {
                // Qwen keeps the key for compatibility but only uses it when asked to.
                if cfg.get("use_sliding_window").and_then(Value::as_bool) == Some(true) {
                    return Err(
                        "Qwen sliding-window layers (max_window_layers) are unsupported".into(),
                    );
                }
            } else {
                let flags: Vec<bool> = if layer_types.len() == layers as usize {
                    layer_types
                        .iter()
                        .map(|t| t == "sliding_attention")
                        .collect()
                } else if raw == "gemma2" {
                    (0..layers).map(|i| i % 2 == 0).collect()
                } else if raw.starts_with("gemma3") {
                    let pattern = num(cfg, "sliding_window_pattern")
                        .or_else(|| num(cfg, "_sliding_window_pattern"))
                        .unwrap_or(6);
                    (0..layers).map(|i| i % pattern < pattern - 1).collect()
                } else {
                    vec![true; layers as usize]
                };
                if flags.iter().any(|&f| f) {
                    u!("attention.sliding_window", window);
                    sliding_layers = Some(flags);
                }
            }
        }
        if let Some(flags) = &sliding_layers {
            meta.push((
                format!("{arch}.attention.sliding_window_pattern"),
                Meta::Bools(flags.clone()),
            ));
        }

        // layers that skip rotary embeddings (true = rotates)
        let rope_flags: Option<Vec<bool>> = match arch {
            "smollm3" => cfg
                .get("no_rope_layers")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|v| v.as_u64() != Some(0) && v.as_bool() != Some(false))
                        .collect()
                }),
            "exaone4" | "exaone_moe" | "cohere2" | "cohere2_moe" => sliding_layers.clone(),
            "muse-glimmer" => cfg
                .get("layer_rope_theta")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|v| v.as_f64().is_some_and(|t| t != 0.0)).collect())
                .or_else(|| sliding_layers.clone()),
            "llama4" => Some(llama4_rope_flags(cfg, layers)),
            "afmoe" => sliding_layers.clone(),
            _ => None,
        };
        if let Some(flags) = rope_flags.filter(|f| f.len() == layers as usize) {
            meta.push((format!("{arch}.rope.layer_flags"), Meta::Bools(flags)));
        }
        if matches!(arch, "hunyuan-dense" | "hunyuan-moe") {
            meta.push((
                format!("{arch}.attention.qk_norm_after_rope"),
                Meta::Bool(true),
            ));
        }

        // rotary embeddings
        let global = rope_params(cfg, Some("full_attention"));
        let local = rope_params(cfg, Some("sliding_attention"));
        let default_theta = if matches!(
            raw.as_str(),
            "llama" | "mistral" | "granite" | "phi3" | "gpt_neox" | "xverse" | "xverse_moe" | "minicpm"
        ) {
            10_000.0
        } else if raw == "chatglm" {
            10_000.0 * flt(cfg, "rope_ratio").unwrap_or(1.0)
        } else {
            1_000_000.0
        };
        let mut theta = global
            .theta
            .or_else(|| flt(cfg, "rotary_emb_base"))
            .unwrap_or(default_theta);
        if let (true, Some(alpha)) = (global.kind == "dynamic", flt(&global.raw, "alpha")) {
            // NTK-aware base used by Hunyuan
            let dim = head_dim as f64;
            theta *= alpha.powf(dim / (dim - 2.0));
        }
        f!("rope.freq_base", theta);
        if arch == "olmo3" || raw == "mellum" {
            f!("rope.freq_base_swa", local.theta.unwrap_or(theta));
        }
        if let Some(beta) = flt(&global.raw, "llama_4_scaling_beta") {
            f!("attention.temperature_beta", beta);
            u!(
                "attention.temperature_context",
                num(&global.raw, "original_max_position_embeddings").unwrap_or(ctx)
            );
        }
        if raw.starts_with("gemma3") {
            let base = local
                .theta
                .or_else(|| flt(cfg, "rope_local_base_freq"))
                .unwrap_or(10_000.0);
            f!("rope.freq_base_swa", base);
        }
        let partial = flt(&global.raw, "partial_rotary_factor")
            .or_else(|| flt(cfg, "partial_rotary_factor"))
            .or_else(|| num(cfg, "rope_head_dim").map(|d| d as f64 / head_dim as f64))
            .or_else(|| num(cfg, "rotary_dim").map(|d| d as f64 / head_dim as f64))
            .or_else(|| (arch == "chatglm").then_some(0.5))
            .or_else(|| flt(cfg, "rotary_pct"))
            .filter(|_| !mla); // MLA states its rotary width explicitly
        if let Some(p) = partial.filter(|p| (*p - 1.0).abs() > 1e-9) {
            let dims = (head_dim as f64 * p).round() as u64;
            u!("rope.dimension_count", dims);
        }
        let mut synthesized = Vec::new();
        match global.kind.as_str() {
            "default" | "mrope" => {}
            "dynamic" if flt(&global.raw, "alpha").is_some() => {}
            // InternLM2/3: dynamic NTK scaling is the identity up to max_position_embeddings
            "dynamic" if arch == "internlm2" || raw == "internlm3" => {}
            "dynamic" | "dynamic-ntk" | "dynamic_ntk" => {
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("dynamic".into()),
                ));
                if let Some(factor) = flt(&global.raw, "factor") {
                    f!("rope.scaling.factor", factor);
                }
                if let Some(orig) = num(&global.raw, "original_max_position_embeddings")
                    .or_else(|| num(cfg, "original_max_position_embeddings"))
                {
                    u!("rope.scaling.original_context_length", orig);
                }
            }
            "llama4" => {
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("llama4".into()),
                ));
                if let Some(factor) = flt(&global.raw, "factor") {
                    f!("rope.scaling.factor", factor);
                }
                if let Some(orig) = num(&global.raw, "original_max_position_embeddings")
                    .or_else(|| num(cfg, "original_max_position_embeddings"))
                {
                    u!("rope.scaling.original_context_length", orig);
                }
                if let Some(beta) = flt(&global.raw, "llama_4_scaling_beta").or_else(|| flt(&global.raw, "beta")) {
                    f!("attention.temperature_beta", beta);
                }
            }
            "proportional" => {
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("proportional".into()),
                ));
                if let Some(factor) = flt(&global.raw, "factor") {
                    f!("rope.scaling.factor", factor);
                }
                if let Some(orig) = num(&global.raw, "original_max_position_embeddings")
                    .or_else(|| num(cfg, "original_max_position_embeddings"))
                {
                    u!("rope.scaling.original_context_length", orig);
                }
            }
            "linear" => {
                let factor =
                    flt(&global.raw, "factor").ok_or("linear rope_scaling requires factor")?;
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("linear".into()),
                ));
                f!("rope.scaling.factor", factor);
            }
            "yarn" => {
                let factor =
                    flt(&global.raw, "factor").ok_or("yarn rope_scaling requires factor")?;
                let original = num(&global.raw, "original_max_position_embeddings")
                    .or_else(|| num(cfg, "original_max_position_embeddings"))
                    .ok_or("yarn rope_scaling requires original_max_position_embeddings")?;
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("yarn".into()),
                ));
                f!("rope.scaling.factor", factor);
                u!("rope.scaling.original_context_length", original);
                if global.raw.get("truncate").and_then(Value::as_bool) == Some(false) {
                    meta.push((
                        format!("{arch}.rope.scaling.yarn_truncate"),
                        Meta::Bool(false),
                    ));
                }
                if let Some(m) = flt(&global.raw, "mscale_all_dim") {
                    f!("rope.scaling.yarn_log_multiplier", 0.1 * m);
                }
                // transformers' attention factor replaces ggml's built-in 1 + 0.1 ln(factor)
                let magnitude = |m: f64| 0.1 * m * factor.ln() + 1.0;
                let hf_factor = flt(&global.raw, "attention_factor").or_else(|| {
                    match (
                        flt(&global.raw, "mscale"),
                        flt(&global.raw, "mscale_all_dim"),
                    ) {
                        (Some(a), Some(b)) => Some(magnitude(a) / magnitude(b)),
                        _ => None,
                    }
                });
                if let Some(h) = hf_factor {
                    f!("rope.scaling.yarn_attn_factor", h / magnitude(1.0));
                }
                for (src, dst) in [
                    ("beta_fast", "rope.scaling.yarn_beta_fast"),
                    ("beta_slow", "rope.scaling.yarn_beta_slow"),
                    ("extrapolation_factor", "rope.scaling.yarn_ext_factor"),
                ] {
                    if let Some(v) = flt(&global.raw, src) {
                        f!(dst, v);
                    }
                }
            }
            "llama3" => {
                let theta = global.theta.unwrap_or(default_theta);
                let factor =
                    flt(&global.raw, "factor").ok_or("llama3 rope_scaling requires factor")?;
                let low = flt(&global.raw, "low_freq_factor").unwrap_or(1.0);
                let high = flt(&global.raw, "high_freq_factor").unwrap_or(4.0);
                let original = flt(&global.raw, "original_max_position_embeddings")
                    .ok_or("llama3 rope_scaling requires original_max_position_embeddings")?;
                let dims =
                    partial.map_or(head_dim, |p| (head_dim as f64 * p).round() as u64) as usize;
                let low_wavelength = original / low;
                let high_wavelength = original / high;
                let factors = (0..dims / 2)
                    .map(|i| {
                        let frequency = 1.0 / theta.powf((2 * i) as f64 / dims as f64);
                        let wavelength = 2.0 * std::f64::consts::PI / frequency;
                        let scale = if wavelength < high_wavelength {
                            1.0
                        } else if wavelength > low_wavelength {
                            factor
                        } else {
                            let smooth = (original / wavelength - low) / (high - low);
                            1.0 / ((1.0 - smooth) / factor + smooth)
                        };
                        scale as f32
                    })
                    .collect::<Vec<_>>();
                synthesized.push(("rope_freqs.weight".to_owned(), factors));
            }
            "longrope" | "su" => {
                let list = |key: &str| -> Result<Vec<f32>, String> {
                    global
                        .raw
                        .get(key)
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(Value::as_f64)
                                .map(|v| v as f32)
                                .collect()
                        })
                        .ok_or_else(|| format!("longrope scaling requires {key}"))
                };
                let (short, long) = (list("short_factor")?, list("long_factor")?);
                let dims = if mla {
                    rope_dims as usize / 2
                } else {
                    partial.map_or(head_dim, |p| (head_dim as f64 * p).round() as u64) as usize / 2
                };
                if short.len() != dims || long.len() != dims {
                    return Err(format!("longrope factors must have {dims} entries"));
                }
                let original = num(&global.raw, "original_max_position_embeddings")
                    .or_else(|| num(cfg, "original_max_position_embeddings"))
                    .unwrap_or(ctx);
                let scale = ctx as f64 / original as f64;
                let attention = flt(&global.raw, "attention_factor").unwrap_or(if scale > 1.0 {
                    (1.0 + scale.ln() / (original as f64).ln()).sqrt()
                } else {
                    1.0
                });
                meta.push((
                    format!("{arch}.rope.scaling.type"),
                    Meta::Str("longrope".into()),
                ));
                u!("rope.scaling.original_context_length", original);
                f!("rope.scaling.attn_factor", attention);
                synthesized.push(("rope_factors_short.weight".to_owned(), short));
                synthesized.push(("rope_factors_long.weight".to_owned(), long));
            }
            other => return Err(format!("unsupported rope_scaling type {other:?}")),
        }

        if moe {
            let used = num(cfg, "num_experts_per_tok")
                .or_else(|| num(cfg, "moe_topk"))
                .or_else(|| num(cfg, "moe_top_k"))
                .or_else(|| num(cfg, "moe_k"))
                .ok_or("MoE config requires num_experts_per_tok")?;
            if used == 0 || used > experts {
                return Err("invalid num_experts_per_tok".into());
            }
            u!("expert_count", experts);
            u!("expert_used_count", used);
            u!(
                "expert_feed_forward_length",
                num(cfg, "moe_intermediate_size").unwrap_or(inter)
            );
            if let Some(shared) = num(cfg, "shared_expert_intermediate_size").or_else(|| num(cfg, "moe_shared_expert_intermediate_size")).filter(|&s| s > 0) {
                u!("expert_shared_feed_forward_length", shared);
            }
            if matches!(raw.as_str(), "minimax_m2") {
                u!("expert_gating_func", 2);
            }
            if raw == "phimoe" {
                u!("expert_gating_func", 3);
                f!(
                    "expert_router_jitter_noise",
                    flt(cfg, "router_jitter_noise").unwrap_or(0.01)
                );
            }
            if raw == "gpt_oss" {
                f!(
                    "expert_swiglu_alpha",
                    flt(cfg, "swiglu_alpha").unwrap_or(1.702)
                );
                f!(
                    "expert_swiglu_limit",
                    flt(cfg, "swiglu_limit").unwrap_or(7.0)
                );
            }
            if matches!(raw.as_str(), "hy_v3" | "afmoe") {
                // sigmoid router with selection bias, renormalised and scaled
                u!("expert_gating_func", 2);
                f!(
                    "expert_weights_scale",
                    flt(cfg, "router_scaling_factor")
                        .or_else(|| flt(cfg, "route_scale"))
                        .unwrap_or(1.0)
                );
            }
            if mla
                || matches!(
                    raw.as_str(),
                    "glm4_moe" | "dots1" | "solar_open" | "exaone_moe" | "g9v3" | "deepseek" | "xverse_moe" | "xverse"
                )
            {
                let per = num(cfg, "moe_intermediate_size").unwrap_or(inter);
                if let Some(n) = num(cfg, "n_shared_experts").or_else(|| num(cfg, "num_shared_experts")).filter(|&n| n > 0) {
                    u!("expert_shared_count", n);
                    u!("expert_shared_feed_forward_length", n * per);
                }
                if let Some(d) = num(cfg, "first_k_dense_replace") {
                    u!("leading_dense_block_count", d);
                }
                if let Some(s) = flt(cfg, "routed_scaling_factor") {
                    f!("expert_weights_scale", s);
                }
                // V3 and Kimi routers are always sigmoid-scored
                let gating = match cfg.get("scoring_func").and_then(Value::as_str) {
                    Some("sigmoid") => 2,
                    Some(_) => 1,
                    None if !matches!(raw.as_str(), "deepseek_v2" | "mistral4" | "deepseek" | "xverse_moe" | "xverse") => 2,
                    None => 1,
                };
                u!("expert_gating_func", gating);
                if raw == "mistral4" {
                    meta.push((format!("{arch}.expert_group_top2"), Meta::Bool(true)));
                }
                if let Some(g) = num(cfg, "n_group").filter(|&g| g > 1) {
                    u!("expert_group_count", g);
                    u!(
                        "expert_group_used_count",
                        num(cfg, "topk_group").unwrap_or(g)
                    );
                }
            }
            if raw == "k2_horizon" {
                if num(cfg, "decoder_sparse_step").unwrap_or(1) != 1 {
                    return Err("K2 Horizon decoder_sparse_step other than 1 is unsupported".into());
                }
                let only: Vec<u64> = cfg
                    .get("mlp_only_layers")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                let mut dense = 0u64;
                while only.contains(&dense) {
                    dense += 1;
                }
                if only.iter().any(|&l| l >= dense) {
                    return Err("K2 Horizon mlp_only_layers must be a leading prefix".into());
                }
                u!("leading_dense_block_count", dense);
                u!(
                    "expert_gating_func",
                    if cfg.get("router_score_func").and_then(Value::as_str) == Some("softmax") {
                        1
                    } else {
                        2
                    }
                );
                f!("expert_weights_scale", flt(cfg, "router_scaling_factor").unwrap_or(1.0));
                let per = num(cfg, "moe_intermediate_size").unwrap_or(inter);
                if let Some(n) = num(cfg, "num_shared_experts").filter(|&n| n > 0) {
                    u!("expert_shared_count", n);
                    u!("expert_shared_feed_forward_length", n * per);
                }
            }
            let normalize = match cfg.get("norm_topk_prob").and_then(Value::as_bool) {
                Some(b) => b,
                None => matches!(
                    raw.as_str(),
                    "mixtral"
                        | "hy_v3"
                        | "afmoe"
                        | "minimax_m2"
                        | "ernie4_5_moe"
                        | "hunyuan_v1_moe"
                        | "laguna"
                        | "gpt_oss"
                        | "granitemoe"
                        | "granitemoeshared"
                        | "granitemoe_swa"
                        | "granitemoehybrid"
                        | "qwen3_5_moe"
                        | "qwen3_5_moe_text"
                        | "cohere2_moe"
                        | "xverse_moe"
                        | "xverse"
                ),
            };
            meta.push((format!("{arch}.expert_weights_norm"), Meta::Bool(normalize)));
        }

        if raw == "k2_horizon" {
            u!("attention.group_norm_groups", num(cfg, "layernorm_num_groups").unwrap_or(1));
            if cfg.get("query_key_norm").and_then(Value::as_bool).unwrap_or(true) {
                return Err("K2 Horizon query_key_norm is unsupported".into());
            }
            if num(cfg, "rope_head_dim").is_some_and(|d| d != head_dim) {
                return Err("K2 Horizon rope_head_dim narrower than head_dim is unsupported".into());
            }
            match cfg.get("attention_gate_func").and_then(Value::as_str) {
                None | Some("softplus") => {}
                Some(other) => return Err(format!("K2 Horizon attention_gate_func {other:?} is unsupported")),
            }
            let value_experts = num(cfg, "mova_num_experts").unwrap_or(0);
            if value_experts > 0 {
                u!("attention.value_expert_count", value_experts);
                u!("attention.value_expert_used_count", num(cfg, "mova_num_experts_per_tok").unwrap_or(1));
            }
        }
        if matches!(arch, "granitehybrid" | "bamba") {
            let get = |k: &str| num(cfg, k).unwrap_or(0);
            u!("ssm.conv_kernel", get("mamba_d_conv"));
            u!("ssm.state_size", get("mamba_d_state"));
            u!("ssm.group_count", get("mamba_n_groups"));
            u!("ssm.time_step_rank", get("mamba_n_heads"));
            u!("ssm.inner_size", get("mamba_expand") * hidden);
            u!("ssm.norm_groups", 1);
            if let (Some(shared), "granitehybrid") = (num(cfg, "shared_intermediate_size"), arch) {
                if moe {
                    u!("expert_shared_feed_forward_length", shared);
                }
                if !moe {
                    if let Some(entry) = meta
                        .iter_mut()
                        .find(|(k, _)| k.ends_with(".feed_forward_length"))
                    {
                        entry.1 = Meta::U32(shared as u32);
                    }
                }
            }
            // the attention layers only rotate when asked to
            if arch == "granitehybrid"
                && cfg.get("position_embedding_type").and_then(Value::as_str) != Some("rope")
            {
                meta.push((
                    format!("{arch}.rope.layer_flags"),
                    Meta::Bools(vec![false; layers as usize]),
                ));
            }
        }
        if arch == "falcon-h1" {
            crate::hybrid::falcon_h1_meta(cfg, hidden, &mut meta)?;
        }
        if arch == "jamba" {
            crate::hybrid::jamba_meta(cfg, hidden, layers, &mut meta)?;
        }
        if family == Family::Qwen3Next || arch == "olmo_hybrid" {
            if arch == "olmo_hybrid" {
                f!(
                    "ssm.beta_scale",
                    if cfg.get("linear_allow_neg_eigval").and_then(Value::as_bool) == Some(true) {
                        2.0
                    } else {
                        1.0
                    }
                );
                f!("ssm.norm_eps", 1e-5);
            }
            let get = |k: &str| num(cfg, k).unwrap_or(0);
            u!("ssm.conv_kernel", get("linear_conv_kernel_dim"));
            u!("ssm.state_size", get("linear_key_head_dim"));
            u!("ssm.group_count", get("linear_num_key_heads"));
            u!("ssm.time_step_rank", get("linear_num_value_heads"));
            u!(
                "ssm.inner_size",
                get("linear_num_value_heads") * get("linear_value_head_dim")
            );
        }

        // family specifics
        match arch {
            "command-r" | "cohere2" | "cohere2_moe" => {
                if cfg.get("use_qk_norm").and_then(Value::as_bool) == Some(true) {
                    return Err("Cohere models with use_qk_norm are unsupported".into());
                }
                f!("logit_scale", flt(cfg, "logit_scale").unwrap_or(1.0));
            }
            "gptneox" => {
                let parallel = cfg
                    .get("use_parallel_residual")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                meta.push((
                    format!("{arch}.use_parallel_residual"),
                    Meta::Bool(parallel),
                ));
            }
            "muse-glimmer" => {
                // weightless RMS norm on the embeddings, qk norms without learned weights (queries
                // are scaled by qk_scale_factor), logits scaled before the tanh soft cap
                meta.push((format!("{arch}.embedding_norm"), Meta::Bool(true)));
                f!("attention.layer_norm_post_epsilon", flt(cfg, "post_norm_eps").unwrap_or(1e-8));
                if let Some(cap) = flt(cfg, "final_logit_softcapping") {
                    f!("final_logit_softcapping", cap);
                }
                f!("logit_scale", flt(cfg, "output_multiplier").unwrap_or(1.0));
                let q_scale = flt(cfg, "qk_scale_factor").unwrap_or(3.87) as f32;
                for i in 0..layers {
                    synthesized.push((format!("blk.{i}.attn_q_norm.weight"), vec![q_scale; head_dim as usize]));
                    synthesized.push((format!("blk.{i}.attn_k_norm.weight"), vec![1.0; head_dim as usize]));
                }
            }
            "gemma" | "gemma2" | "gemma3" | "vaultgemma" => {
                f!("embedding_scale", (hidden as f64).sqrt());
                if arch != "gemma" {
                    let scalar = flt(cfg, "query_pre_attn_scalar").unwrap_or(head_dim as f64);
                    f!("attention.scale", 1.0 / scalar.sqrt());
                }
                for (src, dst) in [
                    ("attn_logit_softcapping", "attn_logit_softcapping"),
                    ("final_logit_softcapping", "final_logit_softcapping"),
                ] {
                    if let Some(v) = flt(cfg, src) {
                        f!(dst, v);
                    }
                }
            }
            "mimo_v2" => {
                let swa = |l: usize| layer_types.get(l).map(String::as_str) == Some("sliding_attention");
                let kv_per_layer: Vec<u32> = (0..layers as usize).map(|l| (kv_heads * if swa(l) { 2 } else { 1 }) as u32).collect();
                set_meta(&mut meta, &format!("{arch}.attention.head_count_kv"), Meta::U32s(kv_per_layer));
                let v_dim = num(cfg, "v_head_dim").unwrap_or(128);
                set_meta(&mut meta, &format!("{arch}.attention.value_length"), Meta::U32(v_dim as u32));
                let rotary = |raw: &Value| (head_dim as f64 * flt(raw, "partial_rotary_factor").unwrap_or(0.334)).floor() as u32;
                set_meta(&mut meta, &format!("{arch}.rope.dimension_count"), Meta::U32(rotary(&global.raw)));
                set_meta(&mut meta, &format!("{arch}.rope.dimension_count_swa"), Meta::U32(rotary(&local.raw)));
                f!("rope.freq_base_swa", local.theta.unwrap_or(10_000.0));
                f!("attention.value_scale", flt(cfg, "attention_value_scale").unwrap_or(1.0));
                u!("expert_gating_func", 2);
                f!("expert_weights_scale", flt(cfg, "routed_scaling_factor").unwrap_or(1.0));
                if let Some(g) = num(cfg, "n_group").filter(|&g| g > 1) {
                    u!("expert_group_count", g);
                    u!("expert_group_used_count", num(cfg, "topk_group").unwrap_or(g));
                }
            }
            "laguna" => {
                if cfg.get("moe_apply_router_weight_on_input").and_then(Value::as_bool) == Some(true) {
                    return Err("Laguna with moe_apply_router_weight_on_input is unsupported".into());
                }
                let per_layer: Vec<u32> = match cfg.get("num_attention_heads_per_layer").and_then(Value::as_array) {
                    Some(list) if list.len() == layers as usize => list.iter().map(|v| v.as_u64().unwrap_or(heads) as u32).collect(),
                    Some(_) => return Err("num_attention_heads_per_layer does not match num_hidden_layers".into()),
                    None => vec![heads as u32; layers as usize],
                };
                if let Some(entry) = meta.iter_mut().find(|(k, _)| k.ends_with(".attention.head_count")) {
                    entry.1 = Meta::U32s(per_layer);
                }
                u!("rope.dimension_count_swa", (head_dim as f64 * flt(&local.raw, "partial_rotary_factor").unwrap_or(1.0)).round() as u64);
                f!("rope.freq_base_swa", local.theta.unwrap_or(10_000.0));
                u!("expert_gating_func", 2);
                f!("expert_weights_scale", flt(cfg, "moe_routed_scaling_factor").unwrap_or(1.0));
                let cap = flt(cfg, "moe_router_logit_softcapping").unwrap_or(0.0);
                if cap > 0.0 {
                    f!("router_logit_softcapping", cap);
                }
                match cfg.get("gating") {
                    Some(Value::String(s)) if s == "per-element" => meta.push((format!("{arch}.attention.gate_per_element"), Meta::Bool(true))),
                    _ => {}
                }
            }
            "minicpm" | "minicpm3" => {
                f!("embedding_scale", flt(cfg, "scale_emb").unwrap_or(1.0));
                f!(
                    "residual_scale",
                    flt(cfg, "scale_depth").unwrap_or(1.0) / (layers as f64).sqrt()
                );
                f!(
                    "logit_scale",
                    flt(cfg, "dim_model_base").unwrap_or(hidden as f64) / hidden as f64
                );
            }
            "llama4" => {
                if let Some(chunk) = num(cfg, "attention_chunk_size") {
                    u!("attention.chunk_size", chunk);
                }
                if cfg.get("attn_temperature_tuning").and_then(Value::as_bool) != Some(false) {
                    f!(
                        "attention.temperature_beta",
                        flt(cfg, "attn_scale").unwrap_or(0.1)
                    );
                    u!(
                        "attention.temperature_context",
                        num(cfg, "floor_scale").unwrap_or(8192)
                    );
                    meta.push((
                        format!("{arch}.attention.temperature_nope_only"),
                        Meta::Bool(true),
                    ));
                    u!("attention.temperature_offset", 1);
                }
                meta.push((
                    format!("{arch}.expert_weights_apply_to_input"),
                    Meta::Bool(true),
                ));
                u!("expert_gating_func", 2);
                if cfg.get("use_qk_norm").and_then(Value::as_bool) != Some(false) {
                    // weightless L2 norm after the rotary embedding on the layers that rotate
                    meta.push((
                        format!("{arch}.attention.qk_norm_after_rope"),
                        Meta::Bool(true),
                    ));
                    for (i, rotates) in llama4_rope_flags(cfg, layers).into_iter().enumerate() {
                        if rotates {
                            synthesized.push((
                                format!("blk.{i}.attn_q_norm.weight"),
                                vec![1.0; head_dim as usize],
                            ));
                            synthesized.push((
                                format!("blk.{i}.attn_k_norm.weight"),
                                vec![1.0; head_dim as usize],
                            ));
                        }
                    }
                }
                if let Some(entry) = meta
                    .iter_mut()
                    .find(|(k, _)| k.ends_with(".feed_forward_length"))
                {
                    entry.1 = Meta::U32(num(cfg, "intermediate_size_mlp").unwrap_or(inter) as u32);
                }
            }
            "afmoe" => {
                if cfg.get("mup_enabled").and_then(Value::as_bool) == Some(true) {
                    f!("embedding_scale", (hidden as f64).sqrt());
                }
            }
            "phi" | "stablelm" | "olmo" => {
                if cfg.get("qk_layernorm").and_then(Value::as_bool) == Some(true) {
                    return Err(format!("{arch} models with qk_layernorm are unsupported"));
                }
                let parallel = arch == "phi"
                    || cfg.get("use_parallel_residual").and_then(Value::as_bool) == Some(true);
                if parallel {
                    meta.push((format!("{arch}.use_parallel_residual"), Meta::Bool(true)));
                }
                if let Some(clip) = flt(cfg, "clip_qkv") {
                    f!("attention.clip_qkv", clip);
                }
                if arch == "olmo" {
                    // non-parametric LayerNorm: identity weights keep the generic decoder
                    synthesized.push(("output_norm.weight".to_owned(), vec![1.0; hidden as usize]));
                    for i in 0..layers {
                        synthesized.push((
                            format!("blk.{i}.attn_norm.weight"),
                            vec![1.0; hidden as usize],
                        ));
                        synthesized.push((
                            format!("blk.{i}.ffn_norm.weight"),
                            vec![1.0; hidden as usize],
                        ));
                    }
                }
            }
            "nanochat" => {
                if let Some(cap) = flt(cfg, "final_logit_softcapping") {
                    f!("final_logit_softcapping", cap);
                }
                meta.push((
                    format!("{arch}.attention.qk_norm_after_rope"),
                    Meta::Bool(true),
                ));
                meta.push((format!("{arch}.rope.flipped"), Meta::Bool(true)));
                meta.push((format!("{arch}.embedding_norm"), Meta::Bool(true)));
                // parameter-free RMS norms: identity weights keep the generic decoder
                let ones = |n: u64| vec![1.0f32; n as usize];
                synthesized.push(("output_norm.weight".to_owned(), ones(hidden)));
                for i in 0..layers {
                    synthesized.push((format!("blk.{i}.attn_norm.weight"), ones(hidden)));
                    synthesized.push((format!("blk.{i}.ffn_norm.weight"), ones(hidden)));
                    synthesized.push((format!("blk.{i}.attn_q_norm.weight"), ones(head_dim)));
                    synthesized.push((format!("blk.{i}.attn_k_norm.weight"), ones(head_dim)));
                }
            }
            "granite" | "granitehybrid" | "granitemoe" | "hyperclovax" if matches!(raw.as_str(), "granite_swa" | "granitemoe_swa") => {
                // every layer has its own rotary base; 0 means no position encoding
                if let Some(list) = cfg.get("layer_rope_theta").and_then(Value::as_array).filter(|l| l.len() == layers as usize) {
                    let bases: Vec<f32> = list.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
                    meta.push((format!("{arch}.rope.layer_flags"), Meta::Bools(bases.iter().map(|b| *b != 0.0).collect())));
                    meta.push((format!("{arch}.rope.layer_bases"), Meta::F32s(bases)));
                }
                for (src, dst) in [
                    ("embedding_multiplier", "embedding_scale"),
                    ("attention_multiplier", "attention.scale"),
                    ("residual_multiplier", "residual_scale"),
                    ("logits_scaling", "logit_scale"),
                ] {
                    if let Some(v) = flt(cfg, src) {
                        f!(dst, v);
                    }
                }
            }
            "granite" | "granitehybrid" | "granitemoe" | "hyperclovax" => {
                for (src, dst) in [
                    ("embedding_multiplier", "embedding_scale"),
                    ("attention_multiplier", "attention.scale"),
                    ("residual_multiplier", "residual_scale"),
                    ("logits_scaling", "logit_scale"),
                ] {
                    if let Some(v) = flt(cfg, src) {
                        f!(dst, v);
                    }
                }
            }
            _ => {}
        }
        if let Some(blocks) = cfg.get("block_configs").and_then(Value::as_array) {
            if blocks.len() == layers as usize {
                let mut heads_l = Vec::with_capacity(blocks.len());
                let mut kv_l = Vec::with_capacity(blocks.len());
                let mut ff_l = Vec::with_capacity(blocks.len());
                for b in blocks {
                    let h = num(b, "attention_heads").or_else(|| num(b, "num_attention_heads")).unwrap_or(heads);
                    let k = num(b, "kv_heads").or_else(|| num(b, "num_key_value_heads")).unwrap_or(kv_heads);
                    let f = num(b, "ffn_hidden_size").or_else(|| num(b, "intermediate_size")).unwrap_or(inter);
                    heads_l.push(h as u32);
                    kv_l.push(k as u32);
                    ff_l.push(f as u32);
                }
                set_meta(&mut meta, &format!("{arch}.attention.head_count"), Meta::U32s(heads_l));
                set_meta(&mut meta, &format!("{arch}.attention.head_count_kv"), Meta::U32s(kv_l));
                set_meta(&mut meta, &format!("{arch}.feed_forward_length"), Meta::U32s(ff_l));
            }
        }
        if let Some(list) = cfg.get("num_attention_heads_per_layer").and_then(Value::as_array) {
            let heads_l: Vec<u32> = list.iter().map(|v| v.as_u64().unwrap_or(heads) as u32).collect();
            if heads_l.len() == layers as usize {
                set_meta(&mut meta, &format!("{arch}.attention.head_count"), Meta::U32s(heads_l));
            }
        }
        if let Some(nl) = num(cfg, "num_loops") {
            meta.push((format!("{arch}.num_loops"), Meta::U32(nl as u32)));
        }
        if let Some(bs) = num(cfg, "blocksparse_block_size") {
            meta.push((format!("{arch}.blocksparse_block_size"), Meta::U32(bs as u32)));
        }
        Ok(Self {
            arch: arch.to_owned(),
            family,
            meta,
            // Nemotron's LayerNorm1p computes layer_norm(x, weight + 1, bias)
            norm_add: if matches!(family, Family::Gemma | Family::Qwen3Next) || arch == "nemotron" {
                1.0
            } else {
                0.0
            },
            classic: None,
            kinds: if arch == "olmo_hybrid" {
                cfg.get("layer_types")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .map(|v| {
                                if v.as_str() == Some("linear_attention") {
                                    b'L'
                                } else {
                                    b'A'
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
            linear: if family == Family::Qwen3Next || arch == "olmo_hybrid" {
                let get = |k: &str| num(cfg, k).filter(|&v| v > 0).map(|v| v as usize);
                Some(LinearDims {
                    k_heads: get("linear_num_key_heads").ok_or("missing linear_num_key_heads")?,
                    v_heads: get("linear_num_value_heads")
                        .ok_or("missing linear_num_value_heads")?,
                    k_dim: get("linear_key_head_dim").ok_or("missing linear_key_head_dim")?,
                    v_dim: get("linear_value_head_dim").ok_or("missing linear_value_head_dim")?,
                })
            } else {
                None
            },
            layers,
            moe,
            gated: !matches!(
                family,
                Family::Starcoder2 | Family::Neox | Family::Phi3 | Family::Glm
            ) && !matches!(arch, "arcee" | "jais2" | "nanochat" | "nemotron" | "apertus" | "phi" | "persimmon"),
            moe_layers: (0..layers)
                .map(|i| match raw.as_str() {
                    "deepseek_v2" | "deepseek_v3" | "kimi_k2" | "glm4_moe" | "dots1"
                    | "mistral4" | "g9v3" | "deepseek" => i >= num(cfg, "first_k_dense_replace").unwrap_or(0),
                    "hy_v3" | "exaone_moe" | "glm4_moe_lite" | "deepseek_v32" | "glm_moe_dsa" | "mellum" | "laguna" | "mimo_v2_flash" => cfg
                        .get("mlp_layer_types")
                        .and_then(Value::as_array)
                        .and_then(|a| a.get(i as usize))
                        .and_then(Value::as_str)
                        .map_or(i >= num(cfg, "first_k_dense_replace").unwrap_or(u64::from(matches!(raw.as_str(), "laguna" | "mimo_v2_flash"))), |t| {
                            t == "sparse"
                        }),
                    "afmoe" => i >= num(cfg, "num_dense_layers").unwrap_or(1),
                    "k2_horizon" => !cfg
                        .get("mlp_only_layers")
                        .and_then(Value::as_array)
                        .is_some_and(|a| a.iter().any(|v| v.as_u64() == Some(i))),
                    "llama4_text" | "llama4" => {
                        match cfg.get("moe_layers").and_then(Value::as_array) {
                            Some(list) => list.iter().any(|v| v.as_u64() == Some(i)),
                            None => {
                                let step =
                                    num(cfg, "interleave_moe_layer_step").unwrap_or(1).max(1);
                                i % step == step - 1
                            }
                        }
                    }
                    "jamba" => crate::hybrid::jamba_expert_layer(cfg, i),
                    "ernie4_5_moe" => {
                        let interval = num(cfg, "moe_layer_interval").unwrap_or(1).max(1);
                        (i + 1) % interval == 0
                            && i >= num(cfg, "moe_layer_start_index").unwrap_or(0)
                            && i <= num(cfg, "moe_layer_end_index").unwrap_or(u64::MAX)
                    }
                    _ => true,
                })
                .collect(),
            q_lora: mla && num(cfg, "q_lora_rank").is_some_and(|r| r > 0),
            mixtral_names: matches!(raw.as_str(), "mixtral" | "minimax_m2" | "phimoe"),
            synthesized,
        })
    }

    fn map_moe(&self, layer: &str, suf: &str) -> Option<Mapped> {
        let (router, experts) = if self.mixtral_names {
            ("block_sparse_moe.gate.weight", "block_sparse_moe.experts.")
        } else {
            ("mlp.gate.weight", "mlp.experts.")
        };
        let plain = |name: &str| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add: 0.0,
            })
        };
        if self.arch == "k2_horizon" {
            match suf {
                "mlp.gate.bias" => return plain("exp_probs_b.bias"),
                "mlp.shared_experts.gate_proj.weight" => return plain("ffn_gate_shexp.weight"),
                "mlp.shared_experts.up_proj.weight" => return plain("ffn_up_shexp.weight"),
                "mlp.shared_experts.down_proj.weight" => return plain("ffn_down_shexp.weight"),
                _ => {}
            }
        }
        if suf == router || matches!(suf, "mlp.gate.wg.weight" | "mlp.router.gate.weight" | "mlp.router.weight" | "moe.gate.weight") {
            return plain("ffn_gate_inp.weight");
        }
        // experts stored as two fused tensors (newer transformers layouts)
        let fused = |part: &str| {
            Some(Mapped::Fused {
                layer: layer.parse().unwrap_or(0),
                part: part.to_owned(),
            })
        };
        // Step-3.5 Hub files: stacked experts `moe.{gate,up,down}_proj.weight` ([E, F, H], [E, F, H], [E, H, F]) and `share_expert.*`
        if self.arch == "laguna" {
            match suf {
                "moe.router_bias" => return plain("exp_probs_b.bias"),
                "moe.gate_proj.weight" => return fused("experts_gate"),
                "moe.up_proj.weight" => return fused("experts_up"),
                "moe.down_proj.weight" => return fused("experts_output"),
                "share_expert.gate_proj.weight" => return plain("ffn_gate_shexp.weight"),
                "share_expert.up_proj.weight" => return plain("ffn_up_shexp.weight"),
                "share_expert.down_proj.weight" => return plain("ffn_down_shexp.weight"),
                _ => {}
            }
        }
        match suf {
            "mlp.experts.gate_up_proj" => return fused("experts_input"),
            "mlp.experts.down_proj" => return fused("experts_output"),
            _ => {}
        }
        if suf == "mlp.experts.e_score_correction_bias" {
            return plain("exp_probs_b.bias");
        }
        if let Some(rest) = suf.strip_prefix(experts) {
            let (index, kind) = rest.split_once('.')?;
            let index = index.parse().ok()?;
            let kind = match kind {
                "w1.weight" | "gate_proj.weight" => "gate",
                "w3.weight" | "up_proj.weight" => "up",
                "w2.weight" | "down_proj.weight" => "down",
                _ => return None,
            };
            return Some(Mapped::Expert {
                name: format!("blk.{layer}.ffn_{kind}_exps.weight"),
                index,
            });
        }
        match suf {
            "mlp.shared_experts.gate_proj.weight" => plain("ffn_gate_shexp.weight"),
            "mlp.shared_experts.up_proj.weight" => plain("ffn_up_shexp.weight"),
            "mlp.shared_experts.down_proj.weight" => plain("ffn_down_shexp.weight"),
            "mlp.gate.e_score_correction_bias"
            | "mlp.expert_bias"
            | "mlp.e_score_correction_bias"
            | "mlp.experts.e_score_correction_bias"
            | "mlp.moe_statics.e_score_correction_bias"
            | "block_sparse_moe.e_score_correction_bias" => plain("exp_probs_b.bias"),
            "mlp.shared_mlp.gate_proj.weight" => plain("ffn_gate_shexp.weight"),
            "mlp.shared_mlp.up_proj.weight" => plain("ffn_up_shexp.weight"),
            "mlp.shared_mlp.down_proj.weight" => plain("ffn_down_shexp.weight"),
            "mlp.shared_expert.gate_proj.weight" => plain("ffn_gate_shexp.weight"),
            "mlp.shared_expert.up_proj.weight" => plain("ffn_up_shexp.weight"),
            "mlp.shared_expert.down_proj.weight" => plain("ffn_down_shexp.weight"),
            "mlp.shared_expert_gate.weight" => plain("ffn_gate_inp_shexp.weight"),
            _ => None,
        }
    }

    fn map_hybrid(&self, layer: &str, suf: &str) -> Option<Mapped> {
        let plain = |name: &str| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add: 0.0,
            })
        };
        let fused = |part: &str| {
            Some(Mapped::Fused {
                layer: layer.parse().unwrap_or(0),
                part: part.to_owned(),
            })
        };
        match suf {
            "feed_forward.router.weight" => plain("ffn_gate_inp.weight"),
            "feed_forward.experts.gate_up_proj" => fused("experts_input_t"),
            "feed_forward.experts.down_proj" => fused("experts_output_t"),
            "feed_forward.shared_expert.gate_proj.weight" => plain("ffn_gate_shexp.weight"),
            "feed_forward.shared_expert.up_proj.weight" => plain("ffn_up_shexp.weight"),
            "feed_forward.shared_expert.down_proj.weight" => plain("ffn_down_shexp.weight"),
            "feed_forward.gate_proj.weight" => plain("ffn_gate.weight"),
            "feed_forward.up_proj.weight" => plain("ffn_up.weight"),
            "feed_forward.down_proj.weight" => plain("ffn_down.weight"),
            "feed_forward.gate_proj.bias" => plain("ffn_gate.bias"),
            "feed_forward.up_proj.bias" => plain("ffn_up.bias"),
            "feed_forward.down_proj.bias" => plain("ffn_down.bias"),
            "pre_ff_layernorm.weight" => plain("ffn_norm.weight"),
            "block_sparse_moe.router.layer.weight" => plain("ffn_gate_inp.weight"),
            "block_sparse_moe.router.weight" => plain("ffn_gate_inp.weight"),
            "block_sparse_moe.experts.gate_up_proj" => fused("experts_input"),
            "block_sparse_moe.experts.down_proj" => fused("experts_output"),
            "block_sparse_moe.input_linear.weight" => fused("experts_input"),
            "block_sparse_moe.output_linear.weight" => fused("experts_output"),
            "shared_mlp.input_linear.weight" => fused(if self.moe {
                "shared_input"
            } else {
                "dense_input"
            }),
            "shared_mlp.output_linear.weight" => plain(if self.moe {
                "ffn_down_shexp.weight"
            } else {
                "ffn_down.weight"
            }),
            "mamba.in_proj.weight" => plain("ssm_in.weight"),
            "mamba.in_proj.bias" => plain("ssm_in.bias"),
            "mamba.out_proj.bias" => plain("ssm_out.bias"),
            "mamba.conv1d.weight" => plain("ssm_conv1d.weight"),
            "mamba.conv1d.bias" => plain("ssm_conv1d.bias"),
            "mamba.dt_bias" => plain("ssm_dt.bias"),
            "mamba.D" => plain("ssm_d"),
            "mamba.norm.weight" => plain("ssm_norm.weight"),
            "mamba.out_proj.weight" => plain("ssm_out.weight"),
            "mamba.A_log" => fused("A_log"),
            _ => None,
        }
    }

    fn map_olmo_hybrid(&self, layer: &str, suf: &str, linear: bool) -> Option<Mapped> {
        let plain = |name: &str| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add: 0.0,
            })
        };
        let fused = |part: &str| {
            Some(Mapped::Fused {
                layer: layer.parse().unwrap_or(0),
                part: part.to_owned(),
            })
        };
        match (linear, suf) {
            (true, "linear_attn.q_proj.weight") => fused("lin_q"),
            (true, "linear_attn.k_proj.weight") => fused("lin_k"),
            (true, "linear_attn.v_proj.weight") => fused("lin_v"),
            (true, "linear_attn.a_proj.weight") => plain("ssm_alpha.weight"),
            (true, "linear_attn.b_proj.weight") => plain("ssm_beta.weight"),
            (true, "linear_attn.g_proj.weight") => plain("attn_gate.weight"),
            (true, "linear_attn.o_proj.weight") => plain("ssm_out.weight"),
            (true, "linear_attn.conv1d.weight") => plain("ssm_conv1d.weight"),
            (true, "linear_attn.q_conv1d.weight") => fused("conv_q"),
            (true, "linear_attn.k_conv1d.weight") => fused("conv_k"),
            (true, "linear_attn.v_conv1d.weight") => fused("conv_v"),
            (true, "attention_layer_norm.weight") => plain("attn_norm.weight"),
            (true, "feedforward_layer_norm.weight") => plain("ffn_norm.weight"),
            // the hub names the norm applied after attention "feedforward_layer_norm"
            (false, "feedforward_layer_norm.weight") => plain("post_attention_norm.weight"),
            (true, "linear_attn.dt_bias") => plain("ssm_dt.bias"),
            (true, "linear_attn.o_norm.weight") => plain("ssm_norm.weight"),
            (true, "linear_attn.A_log") => fused("A_log"),
            (true, "post_attention_layernorm.weight") => plain("ffn_norm.weight"),
            (false, "post_attention_layernorm.weight") => plain("post_attention_norm.weight"),
            _ => None,
        }
    }

    fn map_linear(&self, layer: &str, suf: &str) -> Option<Mapped> {
        let plain = |name: &str, add: f32| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add,
            })
        };
        let rest = suf.strip_prefix("linear_attn.")?;
        match rest {
            "in_proj_qkv.weight" => plain("attn_qkv.weight", 0.0),
            "in_proj_z.weight" => plain("attn_gate.weight", 0.0),
            "in_proj_b.weight" => plain("ssm_beta.weight", 0.0),
            "in_proj_a.weight" => plain("ssm_alpha.weight", 0.0),
            "conv1d.weight" => plain("ssm_conv1d.weight", 0.0),
            "dt_bias" => plain("ssm_dt.bias", 0.0),
            "norm.weight" => plain("ssm_norm.weight", 0.0),
            "out_proj.weight" => plain("ssm_out.weight", 0.0),
            "in_proj_qkvz.weight" | "in_proj_ba.weight" | "A_log" => Some(Mapped::Fused {
                layer: layer.parse().unwrap_or(0),
                part: rest.to_owned(),
            }),
            _ => None,
        }
    }

    fn map_gptoss(&self, layer: &str, suf: &str) -> Option<Mapped> {
        let plain = |name: &str| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add: 0.0,
            })
        };
        match suf {
            "mlp.router.weight" => plain("ffn_gate_inp.weight"),
            "mlp.router.bias" => plain("ffn_gate_inp.bias"),
            "self_attn.sinks" => plain("attn_sinks.weight"),
            _ => suf.strip_prefix("mlp.experts.").map(|part| Mapped::Fused {
                layer: layer.parse().unwrap_or(0),
                part: part.to_owned(),
            }),
        }
    }

    /// Attention, MLP and expert mixers of Nemotron-H (the Mamba mixer shares `map_mamba`).
    fn map_nemotron(&self, layer: &str, suf: &str) -> Option<Mapped> {
        let plain = |name: &str| {
            Some(Mapped::Tensor {
                name: format!("blk.{layer}.{name}"),
                add: 0.0,
            })
        };
        let rest = suf.strip_prefix("mixer.")?;
        if let Some(expert) = rest.strip_prefix("experts.") {
            let (index, kind) = expert.split_once('.')?;
            let name = match kind {
                "up_proj.weight" => "ffn_up_exps.weight",
                "down_proj.weight" => "ffn_down_exps.weight",
                _ => return None,
            };
            return Some(Mapped::Expert {
                name: format!("blk.{layer}.{name}"),
                index: index.parse().ok()?,
            });
        }
        match rest {
            "q_proj.weight" => plain("attn_q.weight"),
            "k_proj.weight" => plain("attn_k.weight"),
            "v_proj.weight" => plain("attn_v.weight"),
            "o_proj.weight" => plain("attn_output.weight"),
            "up_proj.weight" => plain("ffn_up.weight"),
            "down_proj.weight" => plain("ffn_down.weight"),
            "gate.weight" => plain("ffn_gate_inp.weight"),
            "gate.e_score_correction_bias" => plain("exp_probs_b.bias"),
            "fc1_latent_proj.weight" => plain("ffn_latent_down.weight"),
            "fc2_latent_proj.weight" => plain("ffn_latent_up.weight"),
            "shared_experts.up_proj.weight" => plain("ffn_up_shexp.weight"),
            "shared_experts.down_proj.weight" => plain("ffn_down_shexp.weight"),
            _ => None,
        }
    }

    fn map_mamba(&self, n: &str) -> Result<Mapped, String> {
        let plain = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
        match n {
            "backbone.embeddings.weight"
            | "backbone.embedding.weight"
            | "model.embeddings.weight" => return plain("token_embd.weight".into()),
            "backbone.norm_f.weight" | "model.norm_f.weight" => {
                return plain("output_norm.weight".into())
            }
            "lm_head.weight" => return plain("output.weight".into()),
            _ => {}
        }
        let rest = n
            .strip_prefix("backbone.layers.")
            .or_else(|| n.strip_prefix("model.layers."))
            .ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
        let (i, suf) = rest
            .split_once('.')
            .ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
        if i.parse::<u64>().map_or(true, |i| i >= self.layers) {
            return Err(format!("tensor {n:?} refers to an invalid layer"));
        }
        if self.family == Family::Nemotron {
            if let Some(m) = self.map_nemotron(i, suf) {
                return Ok(m);
            }
        }
        let mapped = match suf {
            "norm.weight" => "attn_norm.weight",
            "mixer.in_proj.weight" => "ssm_in.weight",
            "mixer.conv1d.weight" => "ssm_conv1d.weight",
            "mixer.conv1d.bias" => "ssm_conv1d.bias",
            "mixer.dt_bias" => "ssm_dt.bias",
            "mixer.D" => "ssm_d",
            "mixer.x_proj.weight" => "ssm_x.weight",
            "mixer.dt_proj.weight" => "ssm_dt.weight",
            "mixer.dt_proj.bias" => "ssm_dt.bias",
            "mixer.norm.weight" => "ssm_norm.weight",
            "mixer.out_proj.weight" => "ssm_out.weight",
            "mixer.A_log" => {
                return Ok(Mapped::Fused {
                    layer: i.parse().unwrap_or(0),
                    part: "A_log".into(),
                })
            }
            _ => return Err(format!("unsupported forward-affecting layer tensor {n:?}")),
        };
        plain(format!("blk.{i}.{mapped}"))
    }

    /// Persimmon (`PersimmonForCausalLM`): a Llama-style block with LayerNorm instead of RMSNorm,
    /// a fused `query_key_value` projection with per-head `q_layernorm`/`k_layernorm` and a
    /// `relu2` (squared ReLU) feed-forward. Tensor names follow the GPT-NeoX/Llama mix used by the
    /// HF reference implementation.
    fn map_persimmon(&self, n: &str) -> Result<Mapped, String> {
        let plain = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
        match n {
            "model.embed_tokens.weight" => return plain("token_embd.weight".into()),
            "model.final_layernorm.weight" => return plain("output_norm.weight".into()),
            "model.final_layernorm.bias" => return plain("output_norm.bias".into()),
            "lm_head.weight" => return plain("output.weight".into()),
            _ => {}
        }
        let rest = n
            .strip_prefix("model.layers.")
            .ok_or_else(|| format!("unrecognized Persimmon tensor {n:?}"))?;
        let (i, suf) = rest
            .split_once('.')
            .ok_or_else(|| format!("malformed Persimmon tensor {n:?}"))?;
        let idx = i
            .parse::<u64>()
            .map_err(|_| format!("invalid Persimmon layer in {n:?}"))?;
        if idx >= self.layers {
            return Err(format!("Persimmon tensor {n:?} refers to out-of-range layer {idx}"));
        }
        let mapped = match suf {
            "input_layernorm.weight" => "attn_norm.weight",
            "input_layernorm.bias" => "attn_norm.bias",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "post_attention_layernorm.bias" => "ffn_norm.bias",
            "self_attn.query_key_value.weight" => "attn_qkv.weight",
            "self_attn.query_key_value.bias" => "attn_qkv.bias",
            "self_attn.dense.weight" => "attn_output.weight",
            "self_attn.dense.bias" => "attn_output.bias",
            "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
            "self_attn.q_layernorm.bias" => "attn_q_norm.bias",
            "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
            "self_attn.k_layernorm.bias" => "attn_k_norm.bias",
            "mlp.dense_h_to_4h.weight" => "ffn_up.weight",
            "mlp.dense_h_to_4h.bias" => "ffn_up.bias",
            "mlp.dense_4h_to_h.weight" => "ffn_down.weight",
            "mlp.dense_4h_to_h.bias" => "ffn_down.bias",
            _ => return Err(format!("unrecognized Persimmon tensor {n:?}")),
        };
        plain(format!("blk.{idx}.{mapped}"))
    }

    fn map_neox(&self, n: &str) -> Result<Mapped, String> {
        let plain = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
        match n {
            "gpt_neox.embed_in.weight" => return plain("token_embd.weight".into()),
            "gpt_neox.final_layer_norm.weight" => return plain("output_norm.weight".into()),
            "gpt_neox.final_layer_norm.bias" => return plain("output_norm.bias".into()),
            "embed_out.weight" => return plain("output.weight".into()),
            _ => {}
        }
        let rest = n
            .strip_prefix("gpt_neox.layers.")
            .ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
        let (i, suf) = rest
            .split_once('.')
            .ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
        let idx = i
            .parse::<u64>()
            .map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
        if idx >= self.layers {
            return Err(format!("tensor {n:?} refers to out-of-range layer {idx}"));
        }
        let mapped = match suf {
            "input_layernorm.weight" => "attn_norm.weight",
            "input_layernorm.bias" => "attn_norm.bias",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            "post_attention_layernorm.bias" => "ffn_norm.bias",
            "attention.query_key_value.weight" => "attn_qkv.weight",
            "attention.query_key_value.bias" => "attn_qkv.bias",
            "attention.dense.weight" => "attn_output.weight",
            "attention.dense.bias" => "attn_output.bias",
            "mlp.dense_h_to_4h.weight" => "ffn_up.weight",
            "mlp.dense_h_to_4h.bias" => "ffn_up.bias",
            "mlp.dense_4h_to_h.weight" => "ffn_down.weight",
            "mlp.dense_4h_to_h.bias" => "ffn_down.bias",
            "attention.rotary_emb.inv_freq" | "attention.bias" | "attention.masked_bias" => {
                return Ok(Mapped::Skip)
            }
            _ => return Err(format!("unsupported forward-affecting layer tensor {n:?}")),
        };
        plain(format!("blk.{i}.{mapped}"))
    }

    fn map_mtp(&self, rest: &str) -> Result<Mapped, String> {
        let (idx, suf) = if let Some(stripped) = rest.strip_prefix("layers.") {
            let (i, s) = stripped
                .split_once('.')
                .ok_or_else(|| format!("malformed mtp layer tensor {rest:?}"))?;
            (
                i.parse::<u64>()
                    .map_err(|_| format!("invalid mtp layer index {i:?}"))?,
                s,
            )
        } else if let Some((i, s)) = rest.split_once('.') {
            if let Ok(parsed) = i.parse::<u64>() {
                (parsed, s)
            } else {
                (0, rest)
            }
        } else {
            (0, rest)
        };
        let suf = suf.strip_prefix("mtp_block.").or_else(|| suf.strip_prefix("transformer_block.")).unwrap_or(suf);
        if self.layers > 0 && (suf.starts_with("self_attn.") || suf.starts_with("mlp.") || suf.starts_with("input_layernorm.") || suf.starts_with("post_attention_layernorm.")) {
            let mapped = self.map(&format!("model.layers.0.{suf}"))?;
            let rename = |name: String| -> Result<String, String> {
                let suffix = name.strip_prefix("blk.0.").ok_or("invalid MTP transformer mapping")?;
                Ok(format!("blk.{idx}.nextn.{suffix}"))
            };
            return match mapped {
                Mapped::Tensor { name, add } => Ok(Mapped::Tensor { name: rename(name)?, add }),
                Mapped::Expert { name, index } => Ok(Mapped::Expert { name: rename(name)?, index }),
                Mapped::Skip => Ok(Mapped::Skip),
                _ => Err(format!("unsupported fused/transposed MTP tensor {rest}")),
            };
        }
        let target = match suf {
            "eh_proj.weight" | "fc.weight" | "nextn.eh_proj.weight" => "eh_proj.weight",
            "enorm.weight" | "pre_fc_norm_embedding.weight" | "nextn.enorm.weight" => "enorm.weight",
            "hnorm.weight" | "pre_fc_norm_hidden.weight" | "nextn.hnorm.weight" => "hnorm.weight",
            "shared_head_norm.weight" | "shared_head.norm.weight" | "post_norm.weight" | "norm.weight" | "nextn.shared_head_norm.weight" => "shared_head_norm.weight",
            "shared_head_head.weight" | "shared_head.head.weight" | "head.weight" | "nextn.shared_head_head.weight" => "shared_head_head.weight",
            "embed_tokens.weight" | "nextn.embed_tokens.weight" => "embed_tokens.weight",
            other => {
                if let Some(s) = other.strip_prefix("nextn.") {
                    s
                } else {
                    other
                }
            }
        };
        Ok(Mapped::Tensor {
            name: format!("blk.{idx}.nextn.{target}"),
            add: 0.0,
        })
    }

    /// Map a Hugging Face tensor name to its GGUF name.
    /// InternLM2 names: `wqkv` is one fused tensor (rows grouped per KV head as
    /// `[q .. q, k, v]`, reordered by `row_permutation`), the MLP is `w1` (gate), `w3` (up), `w2` (down).
    fn map_internlm2(&self, n: &str) -> Result<Mapped, String> {
        let plain = |name: String| Mapped::Tensor { name, add: 0.0 };
        match n {
            "model.tok_embeddings.weight" => return Ok(plain("token_embd.weight".into())),
            "model.norm.weight" => return Ok(plain("output_norm.weight".into())),
            "output.weight" => return Ok(plain("output.weight".into())),
            _ => {}
        }
        let rest = n
            .strip_prefix("model.layers.")
            .ok_or_else(|| format!("unrecognized InternLM2 tensor {n:?}"))?;
        let (i, suf) = rest
            .split_once('.')
            .ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
        let mapped = match suf {
            "attention.wqkv.weight" => "attn_qkv.weight",
            "attention.wqkv.bias" => "attn_qkv.bias",
            "attention.wo.weight" => "attn_output.weight",
            "attention.wo.bias" => "attn_output.bias",
            "attention_norm.weight" => "attn_norm.weight",
            "ffn_norm.weight" => "ffn_norm.weight",
            "feed_forward.w1.weight" => "ffn_gate.weight",
            "feed_forward.w3.weight" => "ffn_up.weight",
            "feed_forward.w2.weight" => "ffn_down.weight",
            "attention.rotary_emb.inv_freq" => return Ok(Mapped::Skip),
            _ => return Err(format!("unrecognized InternLM2 tensor {n:?}")),
        };
        Ok(plain(format!("blk.{i}.{mapped}")))
    }

    pub(crate) fn map(&self, n: &str) -> Result<Mapped, String> {
        let norm = |name: &str| Mapped::Tensor {
            name: name.to_owned(),
            add: self.norm_add,
        };
        let plain = |name: &str| Mapped::Tensor {
            name: name.to_owned(),
            add: 0.0,
        };
        // multimodal checkpoints nest the language model; the vision tower is not imported
        let renamed;
        let n = if let Some(rest) = n.strip_prefix("model.language_model.") {
            renamed = format!("model.{rest}");
            renamed.as_str()
        } else if let Some(rest) = n.strip_prefix("language_model.") {
            renamed = if rest.starts_with("model.") || rest.starts_with("lm_head.") {
                rest.to_owned()
            } else {
                format!("model.{rest}")
            };
            renamed.as_str()
        } else {
            n
        };
        if n.starts_with("model.visual.")
            || n.starts_with("visual.")
            || n.starts_with("vision_tower.")
            || n.starts_with("model.vision_tower.")
            || n.starts_with("multi_modal_projector.")
            || n.starts_with("model.multi_modal_projector.")
            || n.starts_with("model.vision_")
            || n.starts_with("vision_model.")
            || n.starts_with("vit_large_projector.")
        {
            return Ok(Mapped::Skip);
        }
        if self.arch == "deepseek4" {
            return crate::deepseek4::map(self, n);
        }
        if self.arch == "gemma4" {
            return crate::gemma::map(self, n);
        }
        if self.arch == "gemma3n" {
            return crate::gemma3n::map(self, n);
        }
        if let Some(rest) = n.strip_prefix("mtp.") {
            return self.map_mtp(rest);
        }
        // MiMo keeps its NextN/MTP prediction block on a `model.mtp_layers.N.` module
        // (a separate multi-token head). It is unused by the main single-token
        // forward pass, so these tensors are dropped.
        if n.starts_with("model.mtp_layers.") {
            return Ok(Mapped::Skip);
        }
        if self.arch == "dbrx" {
            let global = match n {
                "transformer.wte.weight" => Some("token_embd.weight"),
                "transformer.norm_f.weight" => Some("output_norm.weight"),
                "lm_head.weight" => Some("output.weight"),
                _ => None,
            };
            if let Some(name) = global {
                return Ok(plain(name));
            }
            if let Some(rest) = n.strip_prefix("transformer.blocks.") {
                let (i, suffix) = rest
                    .split_once('.')
                    .ok_or_else(|| format!("malformed DBRX tensor {n}"))?;
                let layer: u64 = i
                    .parse()
                    .map_err(|_| format!("invalid DBRX layer in {n}"))?;
                if layer >= self.layers {
                    return Err(format!("DBRX layer out of range: {n}"));
                }
                let name = match suffix {
                    "norm_attn_norm.norm_1.weight" => "attn_norm.weight",
                    "norm_attn_norm.norm_2.weight" => "ffn_norm.weight",
                    "norm_attn_norm.attn.Wqkv.weight" => "attn_qkv.weight",
                    "norm_attn_norm.attn.out_proj.weight" => "attn_output.weight",
                    "ffn.router.layer.weight" => "ffn_gate_inp.weight",
                    "ffn.experts.mlp.w1" => "ffn_gate_exps.weight",
                    "ffn.experts.mlp.v1" => "ffn_up_exps.weight",
                    "ffn.experts.mlp.w2" => {
                        return Ok(Mapped::Fused {
                            layer,
                            part: "experts_output_t".into(),
                        })
                    }
                    _ => return Err(format!("unrecognized DBRX tensor {n}")),
                };
                return Ok(plain(&format!("blk.{layer}.{name}")));
            }
            return Err(format!("unrecognized DBRX tensor {n}"));
        }
        if self.family == Family::Classic {
            return crate::classic::map(self, n);
        }
        if self.arch == "zamba2" {
            return crate::zamba2::map(self, n);
        }
        if self.arch == "internlm2" {
            return self.map_internlm2(n);
        }
        if self.arch == "persimmon" {
            return self.map_persimmon(n);
        }
        if self.family == Family::Neox {
            return self.map_neox(n);
        }
        if matches!(self.arch.as_str(), "rwkv6" | "rwkv7") {
            return self.map_rwkv(n);
        }
        if matches!(self.family, Family::Mamba | Family::Nemotron) {
            return self.map_mamba(n);
        }
        match n {
            "model.embedding_norm.weight" if matches!(self.arch.as_str(), "lfm2" | "lfm2moe") => {
                return Ok(plain("output_norm.weight"))
            }
            "model.embed_tokens.weight"
            | "transformer.wte.weight"
            | "transformer.embedding.word_embeddings.weight" => return Ok(plain("token_embd.weight")),
            "model.norm.weight" if self.arch == "muse-glimmer" => return Ok(plain("output_norm.weight")),
            "model.norm.weight"
            | "transformer.ln_f.weight"
            | "transformer.final_layernorm.weight"
            | "transformer.encoder.final_layernorm.weight" => return Ok(norm("output_norm.weight")),
            "model.norm.bias"
            | "transformer.ln_f.bias"
            | "transformer.encoder.final_layernorm.bias" => return Ok(plain("output_norm.bias")),
            "model.final_layernorm.weight" => return Ok(plain("output_norm.weight")),
            "model.final_layernorm.bias" => return Ok(plain("output_norm.bias")),
            "lm_head.bias" => return Ok(plain("output.bias")),
            "lm_head.weight" | "transformer.output_layer.weight" => return Ok(plain("output.weight")),
            "transformer.rotary_pos_emb.inv_freq" => return Ok(Mapped::Skip),
            _ => {}
        }
        let layer_rest = n.strip_prefix("model.layers.")
            .or_else(|| n.strip_prefix("transformer.layers."))
            .or_else(|| n.strip_prefix("transformer.encoder.layers."))
            .or_else(|| n.strip_prefix("transformer.h."));
        if let Some(rest) = layer_rest {
            let (i, suf) = rest
                .split_once('.')
                .ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
            let idx = i
                .parse::<u64>()
                .map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
            if let Some(tensor) = suf.strip_prefix("self_attn.indexer.") {
                // DeepSeek sparse attention indexer (DeepSeek-V3.2, GLM-5). The appended NextN block keeps no indexer: it attends densely.
                let name = match tensor {
                    "wq_b.weight" => "attn_q_b.weight",
                    "wk.weight" => "attn_k.weight",
                    "k_norm.weight" => "k_norm.weight",
                    "k_norm.bias" => "k_norm.bias",
                    "weights_proj.weight" => "proj.weight",
                    _ => return Ok(Mapped::Skip),
                };
                if idx >= self.layers {
                    return Ok(Mapped::Skip);
                }
                return Ok(Mapped::Tensor { name: format!("blk.{idx}.indexer.{name}"), add: 0.0 });
            }
            if idx >= self.layers && self.arch == "laguna" {
                // the multi-token-prediction blocks of Step-3.5 are not imported
                return Ok(Mapped::Skip);
            }
            if idx >= self.layers {
                let mtp_idx = idx - self.layers;
                return self.map_mtp(&format!("{mtp_idx}.{suf}"));
            }
            if matches!(self.arch.as_str(), "lfm2" | "lfm2moe") {
                match suf {
                    "feed_forward.experts.gate_up_proj" | "feed_forward.experts.down_proj"
                        if self.arch == "lfm2moe" =>
                    {
                        return Ok(Mapped::Fused {
                            layer: idx,
                            part: if suf.ends_with("gate_up_proj") {
                                "experts_input"
                            } else {
                                "experts_output"
                            }
                            .into(),
                        });
                    }
                    _ => {}
                }
                // Checkpoints saved by transformers (and the hub files) keep one tensor per expert:
                // feed_forward.experts.<n>.w1 (gate), w3 (up), w2 (down).
                if self.arch == "lfm2moe" {
                    if let Some(rest) = suf.strip_prefix("feed_forward.experts.") {
                        if let Some((index, kind)) = rest.split_once('.') {
                            if let Ok(index) = index.parse::<usize>() {
                                let kind = match kind {
                                    "w1.weight" => "gate",
                                    "w3.weight" => "up",
                                    "w2.weight" => "down",
                                    _ => "",
                                };
                                if !kind.is_empty() {
                                    return Ok(Mapped::Expert {
                                        name: format!("blk.{i}.ffn_{kind}_exps.weight"),
                                        index,
                                    });
                                }
                            }
                        }
                    }
                }
                let name = match suf {
                    "feed_forward.gate.weight" if self.arch == "lfm2moe" => "ffn_gate_inp.weight",
                    "feed_forward.expert_bias" if self.arch == "lfm2moe" => "exp_probs_b.bias",
                    "operator_norm.weight" => "attn_norm.weight",
                    "ffn_norm.weight" => "ffn_norm.weight",
                    "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
                    "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
                    "self_attn.out_proj.weight" => "attn_output.weight",
                    "conv.in_proj.weight" => "shortconv_in.weight",
                    "conv.in_proj.bias" => "shortconv_in.bias",
                    "conv.out_proj.weight" => "shortconv_out.weight",
                    "conv.out_proj.bias" => "shortconv_out.bias",
                    "conv.conv.weight" => "shortconv_conv.weight",
                    "conv.conv.bias" => "shortconv_conv.bias",
                    "feed_forward.w1.weight" => "ffn_gate.weight",
                    "feed_forward.w3.weight" => "ffn_up.weight",
                    "feed_forward.w2.weight" => "ffn_down.weight",
                    _ => "",
                };
                if !name.is_empty() {
                    return Ok(plain(&format!("blk.{i}.{name}")));
                }
            }
            if self.arch == "jamba" {
                if let Some(m) = crate::hybrid::map_jamba(i, suf) {
                    return Ok(m);
                }
            }
            if self.arch == "kimi-linear" {
                if let Some(m) = crate::kimi_linear::map(i, suf) {
                    return Ok(m);
                }
            }
            let sandwich = matches!(self.family, Family::Gemma) && self.arch != "gemma" && self.arch != "vaultgemma";
            let post_norm_family = sandwich || self.family == Family::Olmo2 || self.arch == "afmoe";
            if matches!(
                self.arch.as_str(),
                "granitehybrid" | "bamba" | "llama4" | "granitemoe" | "falcon-h1" | "jamba"
            ) {
                if let Some(m) = self.map_hybrid(i, suf) {
                    return Ok(m);
                }
            }
            if self.arch == "olmo_hybrid" {
                let linear = self.kinds.get(idx as usize) == Some(&b'L');
                if let Some(m) = self.map_olmo_hybrid(i, suf, linear) {
                    return Ok(m);
                }
            }
            if self.family == Family::Qwen3Next {
                if let Some(m) = self.map_linear(i, suf) {
                    return Ok(m);
                }
            }
            if let Some(param) = suf.strip_prefix("mlp.act_fn.") {
                // xIELU parameters are folded into one tensor per layer
                return Ok(Mapped::Fused {
                    layer: idx,
                    part: format!("xielu.{param}"),
                });
            }
            if self.family == Family::GptOss {
                if let Some(m) = self.map_gptoss(i, suf) {
                    return Ok(m);
                }
            }
            if self.arch == "k2_horizon" {
                match suf {
                    "self_attn.v_router.weight" => return Ok(plain(&format!("blk.{i}.attn_v_gate.weight"))),
                    "self_attn.v_router.bias" => return Ok(plain(&format!("blk.{i}.attn_v_gate.bias"))),
                    _ => {
                        if let Some(rest) = suf.strip_prefix("self_attn.v_experts.") {
                            if let Some(index) = rest.strip_suffix(".weight").and_then(|r| r.parse().ok()) {
                                return Ok(Mapped::Expert {
                                    name: format!("blk.{i}.attn_v_exps.weight"),
                                    index,
                                });
                            }
                        }
                    }
                }
            }
            if self.moe {
                if let Some(m) = self.map_moe(i, suf) {
                    return Ok(m);
                }
            }
            let mapped = match suf {
                "input_layernorm.weight" => Some(("attn_norm.weight", true)),
                "post_attention_layernorm.weight" if post_norm_family => {
                    Some(("post_attention_norm.weight", true))
                }
                "post_attention_layernorm.weight" => Some(("ffn_norm.weight", true)),
                "pre_feedforward_layernorm.weight" => Some(("ffn_norm.weight", true)),
                "post_feedforward_layernorm.weight" => Some(("post_ffw_norm.weight", true)),
                "self_attn.W_pack.weight" => Some(("attn_qkv.weight", false)),
                "self_attn.W_pack.bias" => Some(("attn_qkv.bias", false)),
                "attn.c_attn.weight" => Some(("attn_qkv.weight", false)),
                "attn.c_attn.bias" => Some(("attn_qkv.bias", false)),
                "attn.c_proj.weight" => Some(("attn_output.weight", false)),
                "attn.c_proj.bias" => Some(("attn_output.bias", false)),
                "ln_1.weight" => Some(("attn_norm.weight", true)),
                "ln_1.bias" => Some(("attn_norm.bias", false)),
                "ln_2.weight" => Some(("ffn_norm.weight", true)),
                "ln_2.bias" => Some(("ffn_norm.bias", false)),
                "attn.attention.q_proj.weight" | "attn.q_proj.weight" => Some(("attn_q.weight", false)),
                "attn.attention.k_proj.weight" | "attn.k_proj.weight" => Some(("attn_k.weight", false)),
                "attn.attention.v_proj.weight" | "attn.v_proj.weight" => Some(("attn_v.weight", false)),
                "attn.attention.out_proj.weight" | "attn.out_proj.weight" => Some(("attn_output.weight", false)),
                "mlp.c_fc_0.weight" => Some(("ffn_gate.weight", false)),
                "mlp.c_fc_1.weight" => Some(("ffn_up.weight", false)),
                "attn.attention.q_proj.bias" | "attn.q_proj.bias" => Some(("attn_q.bias", false)),
                "attn.attention.k_proj.bias" | "attn.k_proj.bias" => Some(("attn_k.bias", false)),
                "attn.attention.v_proj.bias" | "attn.v_proj.bias" => Some(("attn_v.bias", false)),
                "attn.attention.out_proj.bias" | "attn.out_proj.bias" => Some(("attn_output.bias", false)),
                "mlp.c_fc_0.bias" => Some(("ffn_gate.bias", false)),
                "mlp.c_fc_1.bias" => Some(("ffn_up.bias", false)),
                "mlp.w1.weight" => Some(("ffn_up.weight", false)),
                "mlp.w1.bias" => Some(("ffn_up.bias", false)),
                "mlp.w2.weight" => Some(("ffn_gate.weight", false)),
                "mlp.w2.bias" => Some(("ffn_gate.bias", false)),
                "attention.query_key_value.weight" | "self_attention.query_key_value.weight" => Some(("attn_qkv.weight", false)),
                "attention.query_key_value.bias" | "self_attention.query_key_value.bias" => Some(("attn_qkv.bias", false)),
                "attention.dense.weight" | "self_attention.dense.weight" => Some(("attn_output.weight", false)),
                "attention.dense.bias" | "self_attention.dense.bias" => Some(("attn_output.bias", false)),
                "mlp.dense_h_to_4h.weight" => Some(("ffn_up.weight", false)),
                "mlp.dense_4h_to_h.weight" => Some(("ffn_down.weight", false)),
                "self_attn.q_proj.weight" => Some(("attn_q.weight", false)),
                "self_attn.k_proj.weight" => Some(("attn_k.weight", false)),
                "self_attn.v_proj.weight" => Some(("attn_v.weight", false)),
                "self_attn.o_proj.weight" => Some(("attn_output.weight", false)),
                "self_attn.qkv_proj.weight" => Some(("attn_qkv.weight", false)),
                "input_layernorm.bias" => Some(("attn_norm.bias", false)),
                "post_attention_layernorm.bias" => Some(("ffn_norm.bias", false)),
                "mlp.c_fc.weight" => Some(("ffn_up.weight", false)),
                "mlp.c_fc.bias" => Some(("ffn_up.bias", false)),
                "mlp.c_proj.weight" => Some(("ffn_down.weight", false)),
                "mlp.c_proj.bias" => Some(("ffn_down.bias", false)),
                "self_attn.query_layernorm.weight" => Some(("attn_q_norm.weight", false)),
                "self_attn.key_layernorm.weight" => Some(("attn_k_norm.weight", false)),
                "post_self_attn_layernorm.weight" => Some(("post_attention_norm.weight", false)),
                "pre_mlp_layernorm.weight" => Some(("ffn_norm.weight", false)),
                "attention_layernorm.weight" => Some(("attn_norm.weight", true)),
                "feedforward_layernorm.weight" => Some(("ffn_norm.weight", true)),
                "post_mlp_layernorm.weight" if self.arch == "afmoe" => {
                    Some(("post_ffw_norm.weight", false))
                }
                "self_attn.attn_sub_norm.weight" => Some(("attn_sub_norm.weight", false)),
                "mlp.ffn_sub_norm.weight" => Some(("ffn_sub_norm.weight", false)),
                "self_attn.gate_proj.weight" | "self_attn.g_proj.weight" => Some(("attn_gate.weight", false)),
                "self_attn.sinks" | "self_attn.attention_sink_bias" => Some(("attn_sinks.weight", false)),
                "post_norm1.weight" => Some(("post_attention_norm.weight", false)),
                "post_norm2.weight" => Some(("post_ffw_norm.weight", false)),
                "mlp.fc1.weight" => Some(("ffn_up.weight", false)),
                "mlp.fc2.weight" => Some(("ffn_down.weight", false)),
                "post_mlp_layernorm.weight" => Some(("post_ffw_norm.weight", false)),
                "self_attn.q_a_proj.weight" => Some(("attn_q_a.weight", false)),
                "self_attn.q_a_layernorm.weight" => Some(("attn_q_a_norm.weight", false)),
                "self_attn.q_b_proj.weight" => Some(("attn_q_b.weight", false)),
                "self_attn.kv_a_proj_with_mqa.weight" => Some(("attn_kv_a_mqa.weight", false)),
                "self_attn.kv_a_layernorm.weight" => Some(("attn_kv_a_norm.weight", false)),
                "self_attn.kv_b_proj.weight" => Some(("attn_kv_b.weight", false)),
                "mlp.gate_up_proj.weight" => Some(("ffn_up.weight", false)),
                "self_attn.q_norm.weight" => Some(("attn_q_norm.weight", true)),
                "self_attn.k_norm.weight" => Some(("attn_k_norm.weight", true)),
                "mlp.gate_proj.weight" => Some(("ffn_gate.weight", false)),
                "mlp.up_proj.weight" => Some(("ffn_up.weight", false)),
                "mlp.down_proj.weight" => Some(("ffn_down.weight", false)),
                "self_attn.q_proj.bias" => Some(("attn_q.bias", false)),
                "self_attn.k_proj.bias" => Some(("attn_k.bias", false)),
                "self_attn.v_proj.bias" => Some(("attn_v.bias", false)),
                "self_attn.o_proj.bias" => Some(("attn_output.bias", false)),
                "self_attn.dense.weight" => Some(("attn_output.weight", false)),
                "self_attn.dense.bias" => Some(("attn_output.bias", false)),
                "mlp.fc1.bias" => Some(("ffn_up.bias", false)),
                "mlp.fc2.bias" => Some(("ffn_down.bias", false)),
                "mlp.gate_proj.bias" => Some(("ffn_gate.bias", false)),
                "mlp.up_proj.bias" => Some(("ffn_up.bias", false)),
                "mlp.down_proj.bias" => Some(("ffn_down.bias", false)),
                _ => None,
            };
            if let Some((mapped, is_norm)) = mapped {
                return Ok(Mapped::Tensor {
                    name: format!("blk.{i}.{mapped}"),
                    add: if is_norm { self.norm_add } else { 0.0 },
                });
            }
            if matches!(
                suf,
                "self_attn.rotary_emb.inv_freq"
                    | "self_attn.rotary_emb.cos_cached"
                    | "self_attn.rotary_emb.sin_cached"
            ) {
                return Ok(Mapped::Skip);
            }
            if suf.ends_with(".weight")
                || suf.ends_with(".bias")
                || suf.contains("qweight")
                || suf.contains("scales")
                || suf.contains("zeros")
            {
                return Err(format!("unsupported forward-affecting layer tensor {n:?}"));
            }
            return Err(format!("unrecognized layer tensor {n:?}"));
        }
        if matches!(
            n,
            "model.rotary_emb.inv_freq"
                | "model.rotary_emb.cos_cached"
                | "model.rotary_emb.sin_cached"
        ) {
            return Ok(Mapped::Skip);
        }
        if n.ends_with(".weight")
            || n.ends_with(".bias")
            || n.contains("qweight")
            || n.contains("scales")
            || n.contains("zeros")
        {
            return Err(format!("unsupported forward-affecting tensor {n:?}"));
        }
        Err(format!("unrecognized checkpoint tensor {n:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base(model_type: &str) -> Value {
        json!({"model_type": model_type, "hidden_size": 32, "num_attention_heads": 4,
               "num_hidden_layers": 4, "intermediate_size": 64, "num_key_value_heads": 2})
    }

    #[test]
    fn dsa_models_convert_as_mla_and_keep_the_indexer() {
        for model_type in ["deepseek_v32", "glm_moe_dsa"] {
            let mut cfg = base(model_type);
            cfg["n_routed_experts"] = json!(4);
            cfg["num_experts_per_tok"] = json!(2);
            cfg["moe_intermediate_size"] = json!(16);
            cfg["first_k_dense_replace"] = json!(1);
            cfg["kv_lora_rank"] = json!(8);
            cfg["q_lora_rank"] = json!(8);
            cfg["qk_nope_head_dim"] = json!(8);
            cfg["qk_rope_head_dim"] = json!(4);
            cfg["v_head_dim"] = json!(8);
            cfg["index_n_heads"] = json!(2);
            cfg["index_head_dim"] = json!(8);
            cfg["index_topk"] = json!(16);
            let plan = Plan::from_config(&cfg).unwrap();
            assert_eq!(plan.arch, "deepseek2");
            let meta = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v);
            assert!(matches!(meta("deepseek2.attention.indexer.head_count"), Some(Meta::U32(2))));
            assert!(matches!(meta("deepseek2.attention.indexer.key_length"), Some(Meta::U32(8))));
            assert!(matches!(meta("deepseek2.attention.indexer.top_k"), Some(Meta::U32(16))));
            // DeepSeek-V3.2 rotates half-split pairs in the indexer, GLM-5 interleaved pairs
            assert!(matches!(
                meta("deepseek2.attention.indexer.rope_neox"),
                Some(Meta::Bool(neox)) if *neox == (model_type == "deepseek_v32")
            ));
            for (name, gguf) in [
                ("model.layers.1.self_attn.indexer.wq_b.weight", "blk.1.indexer.attn_q_b.weight"),
                ("model.layers.3.self_attn.indexer.wk.weight", "blk.3.indexer.attn_k.weight"),
                ("model.layers.0.self_attn.indexer.k_norm.bias", "blk.0.indexer.k_norm.bias"),
                ("model.layers.2.self_attn.indexer.weights_proj.weight", "blk.2.indexer.proj.weight"),
            ] {
                assert_eq!(plan.map(name).unwrap(), Mapped::Tensor { name: gguf.into(), add: 0.0 }, "{model_type} {name}");
            }
            // the NextN block (index == layer count) keeps no indexer
            assert_eq!(plan.map("model.layers.4.self_attn.indexer.wk.weight").unwrap(), Mapped::Skip);
            assert_eq!(
                plan.map("model.layers.1.self_attn.q_b_proj.weight").unwrap(),
                Mapped::Tensor { name: "blk.1.attn_q_b.weight".into(), add: 0.0 }
            );
        }
    }

    #[test]
    fn nextn_maps_attention_and_shared_head_for_appended_and_separate_blocks() {
        let mut config = base("glm4_moe");
        config["n_routed_experts"] = json!(4);
        config["num_experts_per_tok"] = json!(2);
        config["moe_intermediate_size"] = json!(16);
        config["first_k_dense_replace"] = json!(1);
        let plan = Plan::from_config(&config).unwrap();
        for source in ["model.layers.4.self_attn.q_proj.weight", "mtp.layers.0.self_attn.q_proj.weight"] {
            assert_eq!(plan.map(source).unwrap(), Mapped::Tensor { name:"blk.0.nextn.attn_q.weight".into(), add:0.0 });
        }
        assert_eq!(plan.map("model.layers.4.shared_head.norm.weight").unwrap(), Mapped::Tensor { name:"blk.0.nextn.shared_head_norm.weight".into(), add:0.0 });
    }

    #[test]
    fn gemma_norms_get_the_offset_and_sandwich_names() {
        let plan = Plan::from_config(&base("gemma2")).unwrap();
        assert_eq!(
            plan.map("model.layers.1.pre_feedforward_layernorm.weight")
                .unwrap(),
            Mapped::Tensor {
                name: "blk.1.ffn_norm.weight".into(),
                add: 1.0
            }
        );
        assert_eq!(
            plan.map("model.layers.1.post_attention_layernorm.weight")
                .unwrap(),
            Mapped::Tensor {
                name: "blk.1.post_attention_norm.weight".into(),
                add: 1.0
            }
        );
        assert_eq!(
            plan.map("model.layers.1.self_attn.q_proj.weight").unwrap(),
            Mapped::Tensor {
                name: "blk.1.attn_q.weight".into(),
                add: 0.0
            }
        );
    }

    #[test]
    fn rope_parameters_follow_layer_kinds_and_legacy_keys() {
        let mut cfg = base("gemma3_text");
        cfg["rope_parameters"] = json!({
            "full_attention": {"rope_theta": 1.0e6, "rope_type": "linear", "factor": 8.0},
            "sliding_attention": {"rope_theta": 1.0e4, "rope_type": "default"}
        });
        let plan = Plan::from_config(&cfg).unwrap();
        let find = |key: &str| {
            plan.meta
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        assert!(matches!(find("gemma3.rope.freq_base"), Some(Meta::F32(v)) if v == 1.0e6));
        assert!(matches!(find("gemma3.rope.freq_base_swa"), Some(Meta::F32(v)) if v == 1.0e4));
        assert!(matches!(find("gemma3.rope.scaling.factor"), Some(Meta::F32(v)) if v == 8.0));
        let mut legacy = base("llama");
        legacy["rope_theta"] = json!(500000.0);
        let plan = Plan::from_config(&legacy).unwrap();
        assert!(plan.meta.iter().any(
            |(k, v)| k == "llama.rope.freq_base" && matches!(v, Meta::F32(x) if *x == 500000.0)
        ));
    }

    #[test]
    fn llama3_scaling_is_written_as_per_frequency_divisors() {
        let mut cfg = base("llama");
        cfg["head_dim"] = json!(16);
        cfg["rope_theta"] = json!(500000.0);
        cfg["rope_scaling"] = json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192});
        let plan = Plan::from_config(&cfg).unwrap();
        let (name, factors) = &plan.synthesized[0];
        assert_eq!(name, "rope_freqs.weight");
        assert_eq!(factors.len(), 8);
        assert_eq!(factors[0], 1.0);
        assert_eq!(factors[7], 8.0);
    }

    #[test]
    fn muse_glimmer_norms_gates_and_synthesized_qk_norms() {
        let mut cfg = base("muse_glimmer_text");
        cfg["head_dim"] = json!(16);
        cfg["layer_types"] = json!(["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]);
        cfg["layer_rope_theta"] = json!([1.0e5, 1.0e5, 1.0e5, 0]);
        cfg["sliding_window"] = json!(8);
        cfg["hidden_activation"] = json!("silu");
        cfg["qk_scale_factor"] = json!(3.0);
        cfg["output_multiplier"] = json!(0.5);
        let plan = Plan::from_config(&cfg).unwrap();
        assert_eq!(plan.arch, "muse-glimmer");
        // centred norms store 1 + w, the final norm and the gate projection are plain
        assert_eq!(
            plan.map("model.language_model.layers.0.input_layernorm.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_norm.weight".into(), add: 1.0 }
        );
        assert_eq!(
            plan.map("model.language_model.norm.weight").unwrap(),
            Mapped::Tensor { name: "output_norm.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("model.language_model.layers.2.self_attn.gate_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.2.attn_gate.weight".into(), add: 0.0 }
        );
        assert_eq!(plan.map("model.vision_projection.weight").unwrap(), Mapped::Skip);
        let q = plan.synthesized.iter().find(|(n, _)| n == "blk.1.attn_q_norm.weight").unwrap();
        assert!(q.1.iter().all(|&v| v == 3.0) && q.1.len() == 16);
        let flags = plan.meta.iter().find(|(k, _)| k == "muse-glimmer.rope.layer_flags").unwrap();
        assert!(matches!(&flags.1, Meta::Bools(f) if f == &vec![true, true, true, false]));
    }

    #[test]
    fn nemotron_latent_moe_keeps_projections_and_uses_the_moe_architecture_name() {
        let mut cfg = base("nemotron_h");
        cfg["layers_block_type"] = json!(["mamba", "moe", "attention", "moe"]);
        cfg["mamba_num_heads"] = json!(4);
        cfg["mamba_head_dim"] = json!(8);
        cfg["ssm_state_size"] = json!(8);
        cfg["n_routed_experts"] = json!(4);
        cfg["num_experts_per_tok"] = json!(2);
        cfg["moe_intermediate_size"] = json!(16);
        cfg["moe_latent_size"] = json!(12);
        let plan = Plan::from_config(&cfg).unwrap();
        assert_eq!(plan.arch, "nemotron_h_moe");
        assert!(plan.meta.iter().any(|(k, v)| k == "nemotron_h_moe.moe_latent_size" && matches!(v, Meta::U32(12))));
        assert_eq!(
            plan.map("backbone.layers.1.mixer.fc1_latent_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.1.ffn_latent_down.weight".into(), add: 0.0 }
        );
    }

    #[test]
    fn k2_horizon_value_experts_map_to_stacked_tensors() {
        let mut cfg = base("k2_horizon");
        cfg["head_dim"] = json!(8);
        cfg["num_experts"] = json!(4);
        cfg["num_experts_per_tok"] = json!(2);
        cfg["moe_intermediate_size"] = json!(16);
        cfg["mlp_only_layers"] = json!([0]);
        cfg["num_shared_experts"] = json!(1);
        cfg["query_key_norm"] = json!(false);
        cfg["mova_num_experts"] = json!(5);
        cfg["mova_num_experts_per_tok"] = json!(2);
        cfg["layernorm_num_groups"] = json!(2);
        let plan = Plan::from_config(&cfg).unwrap();
        assert!(plan.meta.iter().any(|(k, v)| k == "k2_horizon.attention.value_expert_count" && matches!(v, Meta::U32(5))));
        assert!(plan.meta.iter().any(|(k, v)| k == "k2_horizon.attention.group_norm_groups" && matches!(v, Meta::U32(2))));
        assert!(plan.meta.iter().any(|(k, v)| k == "k2_horizon.leading_dense_block_count" && matches!(v, Meta::U32(1))));
        assert_eq!(
            plan.map("model.layers.2.self_attn.v_experts.3.weight").unwrap(),
            Mapped::Expert { name: "blk.2.attn_v_exps.weight".into(), index: 3 }
        );
        assert_eq!(
            plan.map("model.layers.2.self_attn.v_router.bias").unwrap(),
            Mapped::Tensor { name: "blk.2.attn_v_gate.bias".into(), add: 0.0 }
        );
        assert_eq!(plan.moe_layers, vec![false, true, true, true]);
    }

    #[test]
    fn bailing_moe_gqa_vs_mla_mapping() {
        // GQA variant (no kv_lora_rank) maps to Qwen family (bailing_moe)
        let mut cfg_gqa = base("bailing_moe");
        cfg_gqa["num_experts"] = json!(8);
        cfg_gqa["num_experts_per_tok"] = json!(2);
        let plan_gqa = Plan::from_config(&cfg_gqa).unwrap();
        assert_eq!(plan_gqa.family, Family::Qwen);
        assert_eq!(plan_gqa.arch, "bailing_moe");

        // MLA variant (with kv_lora_rank) maps to Deepseek family
        let mut cfg_mla = base("bailing_moe");
        cfg_mla["num_experts"] = json!(8);
        cfg_mla["num_experts_per_tok"] = json!(2);
        cfg_mla["kv_lora_rank"] = json!(32);
        cfg_mla["qk_nope_head_dim"] = json!(16);
        cfg_mla["qk_rope_head_dim"] = json!(16);
        cfg_mla["v_head_dim"] = json!(16);
        let plan_mla = Plan::from_config(&cfg_mla).unwrap();
        assert_eq!(plan_mla.family, Family::Deepseek);
        assert_eq!(plan_mla.arch, "deepseek2");
    }

    #[test]
    fn baichuan_qwen_chatglm_tensor_mappings() {
        let baichuan = Plan::from_config(&base("baichuan2")).unwrap();
        assert_eq!(
            baichuan.map("model.layers.0.self_attn.W_pack.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_qkv.weight".into(), add: 0.0 }
        );

        let qwen = Plan::from_config(&base("qwen")).unwrap();
        assert_eq!(
            qwen.map("transformer.h.0.attn.c_attn.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_qkv.weight".into(), add: 0.0 }
        );
        assert_eq!(
            qwen.map("transformer.h.0.attn.c_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_output.weight".into(), add: 0.0 }
        );
        assert_eq!(
            qwen.map("transformer.wte.weight").unwrap(),
            Mapped::Tensor { name: "token_embd.weight".into(), add: 0.0 }
        );

        let chatglm = Plan::from_config(&base("chatglm")).unwrap();
        assert_eq!(
            chatglm.map("transformer.layers.0.attention.query_key_value.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_qkv.weight".into(), add: 0.0 }
        );
        assert_eq!(
            chatglm.map("transformer.output_layer.weight").unwrap(),
            Mapped::Tensor { name: "output.weight".into(), add: 0.0 }
        );
        assert_eq!(
            chatglm.map("transformer.encoder.layers.0.self_attention.query_key_value.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_qkv.weight".into(), add: 0.0 }
        );
        assert_eq!(
            chatglm.map("transformer.encoder.layers.0.mlp.dense_h_to_4h.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_up.weight".into(), add: 0.0 }
        );
        assert_eq!(
            chatglm.map("transformer.encoder.final_layernorm.weight").unwrap(),
            Mapped::Tensor { name: "output_norm.weight".into(), add: 0.0 }
        );
        assert_eq!(
            chatglm.map("transformer.rotary_pos_emb.inv_freq").unwrap(),
            Mapped::Skip
        );
    }

    #[test]
    fn chatglm_plan() {
        let cfg = json!({
            "model_type": "chatglm",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_layers": 2,
            "ffn_hidden_size": 128,
            "multi_query_group_num": 2,
            "kv_channels": 16,
            "seq_length": 2048,
            "layernorm_epsilon": 1e-5,
            "rope_ratio": 1
        });
        let plan = Plan::from_config(&cfg).unwrap();
        assert_eq!(plan.arch, "chatglm");
        assert_eq!(plan.layers, 2);
        assert_eq!(plan.family, Family::Glm);
        assert!(plan.meta.iter().any(|(k, v)| k == "chatglm.feed_forward_length" && matches!(v, Meta::U32(128))));
        assert!(plan.meta.iter().any(|(k, v)| k == "chatglm.attention.head_count_kv" && matches!(v, Meta::U32(2))));
        assert!(plan.meta.iter().any(|(k, v)| k == "chatglm.rope.dimension_count" && matches!(v, Meta::U32(8))));
        assert!(plan.meta.iter().any(|(k, v)| k == "chatglm.attention.layer_norm_rms_epsilon" && matches!(v, Meta::F32(e) if (*e - 1e-5).abs() < 1e-8)));
    }

    #[test]
    fn rope_extensions_dynamic_llama4_proportional() {
        let mut cfg_dyn = base("llama");
        cfg_dyn["rope_scaling"] = json!({"rope_type": "dynamic", "factor": 4.0, "original_max_position_embeddings": 2048});
        let plan_dyn = Plan::from_config(&cfg_dyn).unwrap();
        assert!(plan_dyn.meta.iter().any(|(k, v)| k == "llama.rope.scaling.type" && matches!(v, Meta::Str(s) if s == "dynamic")));
        assert!(plan_dyn.meta.iter().any(|(k, v)| k == "llama.rope.scaling.factor" && matches!(v, Meta::F32(f) if *f == 4.0)));

        let mut cfg_l4 = base("llama");
        cfg_l4["rope_scaling"] = json!({"rope_type": "llama4", "factor": 8.0, "beta": 0.25});
        let plan_l4 = Plan::from_config(&cfg_l4).unwrap();
        assert!(plan_l4.meta.iter().any(|(k, v)| k == "llama.rope.scaling.type" && matches!(v, Meta::Str(s) if s == "llama4")));
        assert!(plan_l4.meta.iter().any(|(k, v)| k == "llama.attention.temperature_beta" && matches!(v, Meta::F32(f) if *f == 0.25)));

        let mut cfg_prop = base("llama");
        cfg_prop["rope_scaling"] = json!({"rope_type": "proportional", "factor": 2.0});
        let plan_prop = Plan::from_config(&cfg_prop).unwrap();
        assert!(plan_prop.meta.iter().any(|(k, v)| k == "llama.rope.scaling.type" && matches!(v, Meta::Str(s) if s == "proportional")));

        // Partial rotary via rope_head_dim < head_dim
        let mut cfg_part = base("llama");
        cfg_part["head_dim"] = json!(128);
        cfg_part["rope_head_dim"] = json!(64);
        let plan_part = Plan::from_config(&cfg_part).unwrap();
        assert!(plan_part.meta.iter().any(|(k, v)| k == "llama.rope.dimension_count" && matches!(v, Meta::U32(64))));
    }

    #[test]
    fn deepseek_v1_moe_plan() {
        let mut config = base("deepseek");
        config["n_routed_experts"] = json!(4);
        config["num_experts_per_tok"] = json!(2);
        config["n_shared_experts"] = json!(1);
        config["moe_intermediate_size"] = json!(16);
        config["first_k_dense_replace"] = json!(1);
        let plan = Plan::from_config(&config).unwrap();
        assert_eq!(plan.family, Family::Llama);
        assert_eq!(plan.arch, "deepseek");
        assert_eq!(plan.moe_layers, vec![false, true, true, true]);
        assert_eq!(
            plan.map("model.layers.0.mlp.gate_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_gate.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("model.layers.1.mlp.gate.weight").unwrap(),
            Mapped::Tensor { name: "blk.1.ffn_gate_inp.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("model.layers.1.mlp.experts.0.gate_proj.weight").unwrap(),
            Mapped::Expert { name: "blk.1.ffn_gate_exps.weight".into(), index: 0 }
        );
        assert_eq!(
            plan.map("model.layers.1.mlp.shared_experts.gate_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.1.ffn_gate_shexp.weight".into(), add: 0.0 }
        );
        assert!(plan.meta.iter().any(|(k, v)| k == "deepseek.expert_count" && matches!(v, Meta::U32(4))));
        assert!(plan.meta.iter().any(|(k, v)| k == "deepseek.expert_used_count" && matches!(v, Meta::U32(2))));
        assert!(plan.meta.iter().any(|(k, v)| k == "deepseek.expert_shared_count" && matches!(v, Meta::U32(1))));
        assert!(plan.meta.iter().any(|(k, v)| k == "deepseek.leading_dense_block_count" && matches!(v, Meta::U32(1))));
        assert!(plan.meta.iter().any(|(k, v)| k == "deepseek.expert_gating_func" && matches!(v, Meta::U32(1))));
    }

    #[test]
    fn exaone_3_plan() {
        let config = json!({
            "model_type": "exaone",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_layers": 4,
            "intermediate_size": 64,
            "activation_function": "silu",
            "layer_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
        });
        let plan = Plan::from_config(&config).unwrap();
        assert_eq!(plan.family, Family::Llama);
        assert_eq!(plan.arch, "exaone");
        assert_eq!(
            plan.map("transformer.wte.weight").unwrap(),
            Mapped::Tensor { name: "token_embd.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.ln_f.weight").unwrap(),
            Mapped::Tensor { name: "output_norm.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("lm_head.weight").unwrap(),
            Mapped::Tensor { name: "output.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.ln_1.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_norm.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.attn.attention.q_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_q.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.attn.attention.k_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_k.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.attn.attention.v_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_v.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.attn.attention.out_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.attn_output.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.ln_2.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_norm.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.mlp.c_fc_0.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_gate.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.mlp.c_fc_1.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_up.weight".into(), add: 0.0 }
        );
        assert_eq!(
            plan.map("transformer.h.0.mlp.c_proj.weight").unwrap(),
            Mapped::Tensor { name: "blk.0.ffn_down.weight".into(), add: 0.0 }
        );
    }

    #[test]
    fn rwkv_tensors_map_to_the_llama_cpp_names() {
        let t = |name: &str| Mapped::Tensor { name: name.to_owned(), add: 0.0 };
        let v7 = Plan::from_config(&json!({"model_type": "rwkv7", "hidden_size": 64, "num_hidden_layers": 2, "head_dim": 16, "intermediate_size": 256, "norm_eps": 1e-5})).unwrap();
        assert_eq!(v7.arch, "rwkv7");
        for (hf, gguf) in [
            ("model.embeddings.weight", "token_embd.weight"),
            ("model.layers.0.pre_norm.bias", "token_embd_norm.bias"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.layers.1.attn_norm.weight", "blk.1.attn_norm.weight"),
            ("model.layers.1.ffn_norm.bias", "blk.1.attn_norm_2.bias"),
            ("model.layers.1.attn.x_a", "blk.1.time_mix_lerp_a.weight"),
            ("model.layers.1.attn.w_lora.lora.0.weight", "blk.1.time_mix_w1.weight"),
            ("model.layers.1.attn.w_lora.lora.2.weight", "blk.1.time_mix_w2.weight"),
            ("model.layers.1.attn.w_lora.lora.2.bias", "blk.1.time_mix_w0.weight"),
            ("model.layers.1.attn.g_lora.lora.2.weight", "blk.1.time_mix_g2.weight"),
            ("model.layers.1.attn.r_k", "blk.1.time_mix_r_k.weight"),
            ("model.layers.1.attn.g_norm.weight", "blk.1.time_mix_ln.weight"),
            ("model.layers.1.attn.o_proj.weight", "blk.1.time_mix_output.weight"),
            ("model.layers.1.ffn.x_k", "blk.1.channel_mix_lerp_k.weight"),
            ("model.layers.1.ffn.value.weight", "blk.1.channel_mix_value.weight"),
        ] {
            assert_eq!(v7.map(hf).unwrap(), t(gguf), "{hf}");
        }
        assert!(v7.map("model.layers.1.attn.unknown").is_err());
        assert!(v7.map("model.layers.9.attn.k_k").is_err());
        let v6 = Plan::from_config(&json!({"model_type": "rwkv6", "hidden_size": 64, "num_hidden_layers": 2, "head_size": 16, "intermediate_size": null})).unwrap();
        assert_eq!(v6.arch, "rwkv6");
        assert_eq!(v6.map("rwkv.blocks.0.pre_ln.weight").unwrap(), t("token_embd_norm.weight"));
        assert_eq!(v6.map("rwkv.blocks.1.attention.time_maa_r").unwrap(), t("blk.1.time_mix_lerp_r.weight"));
        assert_eq!(v6.map("rwkv.blocks.1.attention.time_faaaa").unwrap(), t("blk.1.time_mix_first.weight"));
        assert_eq!(v6.map("rwkv.blocks.1.attention.time_maa_w2").unwrap(), Mapped::Transposed { name: "blk.1.time_mix_w2.weight".into() });
        assert_eq!(v6.map("rwkv.blocks.1.feed_forward.receptance.weight").unwrap(), t("blk.1.channel_mix_receptance.weight"));
        assert_eq!(v6.map("head.weight").unwrap(), t("output.weight"));
    }
}
