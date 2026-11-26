//! RecurrentGemma / Griffin (`RecurrentGemmaForCausalLM`, google/recurrentgemma-*): blocks of two RG-LRU recurrent mixers and one local (windowed,
//! multi-query, half-rotated) attention mixer, each followed by a gated GELU feed-forward with biases. The plan is the Gemma plan of a normalised
//! configuration (`1 + w` norms, embedding scale, tied embeddings), renamed to the GGUF architecture `recurrent_gemma`, plus the RG-LRU metadata
//! and tensors (GGUF names `rglru_*`, read by `rust_decoder/rglru.rs`).
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

/// Per layer: `true` for an RG-LRU recurrent block, `false` for local attention (`block_types` repeated over the layers).
pub(crate) fn recurrent_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let pattern: Vec<bool> = match cfg.get("block_types").or_else(|| cfg.get("_block_pattern")).and_then(Value::as_array) {
        Some(list) if !list.is_empty() => list
            .iter()
            .map(|t| match t.as_str() {
                Some("recurrent") => Ok(true),
                Some("attention") => Ok(false),
                other => Err(format!("unsupported RecurrentGemma block type {other:?}")),
            })
            .collect::<Result<_, _>>()?,
        _ => vec![true, true, false],
    };
    Ok((0..layers).map(|i| pattern[i % pattern.len()]).collect())
}

/// The Gemma configuration the plan and the shape checks use (the feed-forward width is half of `intermediate_size`).
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?;
    let intermediate = num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?;
    if heads == 0 || !hidden.is_multiple_of(heads) {
        return Err("RecurrentGemma hidden_size must be a multiple of num_attention_heads".into());
    }
    let head_dim = num(cfg, "head_dim").unwrap_or(hidden / heads);
    let kv_heads = num(cfg, "num_key_value_heads").unwrap_or(heads);
    let rope = cfg.get("rope_parameters").cloned().unwrap_or(Value::Null);
    let theta = flt(&rope, "rope_theta").or_else(|| flt(cfg, "rope_theta")).unwrap_or(10000.0);
    let mut normalized = json!({
        "model_type": "gemma",
        "architectures": ["GemmaForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": layers,
        // the feed-forward holds half of `intermediate_size` per projection (RecurrentGemmaMlp)
        "intermediate_size": intermediate / 2,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "head_dim": head_dim,
        "rms_norm_eps": flt(cfg, "rms_norm_eps").unwrap_or(1e-6),
        "rope_theta": theta,
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(8192),
        "hidden_activation": "gelu_pytorch_tanh",
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(true),
    });
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "_name_or_path"] {
        if let Some(v) = cfg.get(key) {
            normalized[key] = v.clone();
        }
    }
    Ok(normalized)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    if heads == 0 || !hidden.is_multiple_of(heads) {
        return Err("RecurrentGemma hidden_size must be a multiple of num_attention_heads".into());
    }
    let head_dim = num(cfg, "head_dim").unwrap_or(hidden / heads);
    let lru_width = num(cfg, "lru_width").unwrap_or(hidden);
    let kernel = num(cfg, "conv1d_width").unwrap_or(4);
    let window = num(cfg, "attention_window_size").unwrap_or(2048);
    let rope = cfg.get("rope_parameters").cloned().unwrap_or(Value::Null);
    let partial = flt(&rope, "partial_rotary_factor").or_else(|| flt(cfg, "partial_rotary_factor")).unwrap_or(0.5);
    let rot = ((head_dim as f64 * partial) as u64) & !1;
    if rot == 0 || kernel == 0 || lru_width % heads != 0 {
        return Err("invalid RecurrentGemma rotary, convolution or LRU width".into());
    }
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    for (key, value) in &mut plan.meta {
        *key = key.replace("gemma.", "recurrent_gemma.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("recurrent_gemma".into());
        }
    }
    plan.arch = "recurrent_gemma".into();
    plan.norm_add = 1.0;
    let flags = recurrent_layers(cfg)?;
    let a = "recurrent_gemma";
    plan.meta.retain(|(k, _)| {
        !matches!(
            k.as_str(),
            "recurrent_gemma.rope.dimension_count" | "recurrent_gemma.attention.sliding_window" | "recurrent_gemma.embedding_scale" | "recurrent_gemma.final_logit_softcapping"
        )
    });
    plan.meta.push((format!("{a}.rope.dimension_count"), Meta::U32(rot as u32)));
    plan.meta.push((format!("{a}.attention.sliding_window"), Meta::U32(window as u32)));
    plan.meta.push((format!("{a}.embedding_scale"), Meta::F32((hidden as f64).sqrt() as f32)));
    plan.meta.push((format!("{a}.final_logit_softcapping"), Meta::F32(flt(cfg, "logits_soft_cap").unwrap_or(30.0) as f32)));
    plan.meta.push((format!("{a}.rglru.width"), Meta::U32(lru_width as u32)));
    plan.meta.push((format!("{a}.rglru.kernel"), Meta::U32(kernel as u32)));
    plan.meta.push((format!("{a}.rglru.blocks"), Meta::U32(heads as u32)));
    plan.meta.push((format!("{a}.rglru.layer_flags"), Meta::Bools(flags.clone())));
    plan.kinds = flags.iter().map(|&r| if r { b'R' } else { b'A' }).collect();
    Ok(plan)
}

/// GGUF name of a per-layer tensor of the layer `i` with the checkpoint suffix `suf` (`add` is the norm offset of the plan for the two norms).
pub(crate) fn map(i: u64, suf: &str, add: f32) -> Option<Mapped> {
    let plain = |name: &str| Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 });
    let norm = |name: &str| Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add });
    match suf {
        "temporal_pre_norm.weight" => norm("attn_norm.weight"),
        "channel_pre_norm.weight" => norm("ffn_norm.weight"),
        "temporal_block.linear_y.weight" => plain("rglru_y.weight"),
        "temporal_block.linear_y.bias" => plain("rglru_y.bias"),
        "temporal_block.linear_x.weight" => plain("rglru_x.weight"),
        "temporal_block.linear_x.bias" => plain("rglru_x.bias"),
        "temporal_block.linear_out.weight" => plain("rglru_out.weight"),
        "temporal_block.linear_out.bias" => plain("rglru_out.bias"),
        "temporal_block.conv_1d.weight" => plain("rglru_conv.weight"),
        "temporal_block.conv_1d.bias" => plain("rglru_conv.bias"),
        "temporal_block.rg_lru.recurrent_param" => plain("rglru_a.weight"),
        "temporal_block.rg_lru.input_gate_weight" => plain("rglru_input_gate.weight"),
        "temporal_block.rg_lru.input_gate_bias" => plain("rglru_input_gate.bias"),
        "temporal_block.rg_lru.recurrent_gate_weight" => plain("rglru_recurrent_gate.weight"),
        "temporal_block.rg_lru.recurrent_gate_bias" => plain("rglru_recurrent_gate.bias"),
        "temporal_block.q_proj.weight" => plain("attn_q.weight"),
        "temporal_block.k_proj.weight" => plain("attn_k.weight"),
        "temporal_block.v_proj.weight" => plain("attn_v.weight"),
        "temporal_block.o_proj.weight" => plain("attn_output.weight"),
        "temporal_block.o_proj.bias" => plain("attn_output.bias"),
        "mlp_block.gate_proj.weight" => plain("ffn_gate.weight"),
        "mlp_block.gate_proj.bias" => plain("ffn_gate.bias"),
        "mlp_block.up_proj.weight" => plain("ffn_up.weight"),
        "mlp_block.up_proj.bias" => plain("ffn_up.bias"),
        "mlp_block.down_proj.weight" => plain("ffn_down.weight"),
        "mlp_block.down_proj.bias" => plain("ffn_down.bias"),
        _ => None,
    }
}

/// Tensors every layer of the given kind must have after the conversion.
pub(crate) fn layer_requirements(recurrent: bool) -> Vec<&'static str> {
    let mut need = vec!["attn_norm.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_gate.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias"];
    if recurrent {
        need.extend([
            "rglru_y.weight", "rglru_y.bias", "rglru_x.weight", "rglru_x.bias", "rglru_out.weight", "rglru_out.bias", "rglru_conv.weight", "rglru_conv.bias", "rglru_a.weight",
            "rglru_input_gate.weight", "rglru_input_gate.bias", "rglru_recurrent_gate.weight", "rglru_recurrent_gate.bias",
        ]);
    } else {
        need.extend(["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_output.bias"]);
    }
    need
}
