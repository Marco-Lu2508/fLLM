//! ZAYA (`ZayaForCausalLM`): every layer is compressed convolutional attention plus a mixture of experts with a learned MLP router.
//! * Attention: queries and keys (`q_proj`, `k_proj`) go through a causal depthwise and a grouped convolution over time and keep a residual path
//!   (`(q + k) / 2` per head, averaged over a key head's query group for the key); the value is half current (`v_proj_current`) and half the
//!   previous token's `v_proj_delayed`. Queries and keys are L2-normalised to `sqrt(head_dim)` (keys times a learned scale per key head), then rotated.
//! * Residual stream: `(branch + hb) * hs + (x + rb) * rs` with four learned vectors per block, and a learned scale and bias on the embeddings.
//! * Feed-forward: the router (down projection, state of the layer before, MLP) ranks `experts + 1` classes (the last one skips); the experts are plain gated.
//! The plan is the Mixtral plan of a normalised configuration renamed to the GGUF architecture `zaya`.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn rope(cfg: &Value, kind: &str) -> (f64, f64) {
    let r = cfg.get("rope_parameters").and_then(|r| r.get(kind)).or_else(|| cfg.get("rope_parameters"));
    let theta = r.and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).unwrap_or(10000.0);
    let partial = r.and_then(|r| r.get("partial_rotary_factor")).and_then(Value::as_f64).unwrap_or(1.0);
    (theta, partial)
}

pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?;
    if cfg.get("attention_bias").and_then(Value::as_bool).unwrap_or(false) || cfg.get("lm_head_bias").and_then(Value::as_bool).unwrap_or(false) {
        return Err("ZAYA with attention_bias or lm_head_bias is not supported".into());
    }
    let (theta, partial) = rope(cfg, "hybrid");
    let mut out = json!({
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": num(cfg, "num_key_value_heads").unwrap_or(heads),
        "head_dim": num(cfg, "head_dim").unwrap_or(hidden / heads.max(1)),
        "num_hidden_layers": layers,
        "intermediate_size": num(cfg, "moe_intermediate_size").ok_or("config.json requires moe_intermediate_size")?,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": cfg.get("rms_norm_eps").cloned().unwrap_or(1e-5.into()),
        "rope_theta": theta,
        "partial_rotary_factor": partial,
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(4096),
        "hidden_act": cfg.get("hidden_act").cloned().unwrap_or("silu".into()),
        "num_local_experts": num(cfg, "num_experts").ok_or("config.json requires num_experts")?,
        "num_experts_per_tok": num(cfg, "num_experts_per_tok").ok_or("config.json requires num_experts_per_tok")?,
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(true),
    });
    if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
        out["layer_types"] = types.iter().map(|t| if t.as_str() == Some("hybrid_sliding") { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>().into();
        if let Some(w) = num(cfg, "sliding_window") {
            out["sliding_window"] = w.into();
        }
    }
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "_name_or_path"] {
        if let Some(v) = cfg.get(key) {
            out[key] = v.clone();
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    for (key, value) in &mut plan.meta {
        *key = key.replace("llama.", "zaya.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("zaya".into());
        }
    }
    plan.arch = "zaya".into();
    let (swa_theta, _) = rope(cfg, "hybrid_sliding");
    plan.meta.retain(|(k, _)| k != "zaya.rope.freq_base_swa");
    plan.meta.push(("zaya.rope.freq_base_swa".into(), Meta::F32(swa_theta as f32)));
    plan.meta.push(("zaya.attention.cca_kernels".into(), Meta::U32s(vec![num(cfg, "cca_time0").unwrap_or(2) as u32, num(cfg, "cca_time1").unwrap_or(2) as u32])));
    plan.meta.push(("zaya.router.hidden_size".into(), Meta::U32(num(cfg, "router_hidden_size").ok_or("config.json requires router_hidden_size")? as u32)));
    Ok(plan)
}

/// GGUF names of the global tensors ZAYA adds.
pub(crate) fn map_global(n: &str) -> Option<Mapped> {
    let name = match n {
        "model.input_hidden_states_scale" => "zaya_input_scale.weight",
        "model.input_hidden_states_bias" => "zaya_input_bias.weight",
        _ => return None,
    };
    Some(Mapped::Tensor { name: name.into(), add: 0.0 })
}

/// GGUF name of a per-layer tensor of layer `i` with the checkpoint suffix `suf`.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let fused = |part: &str| Some(Mapped::Fused { layer: i, part: part.to_owned() });
    let name = match suf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.qkv_proj.q_proj.weight" => "attn_q.weight",
        "self_attn.qkv_proj.k_proj.weight" => "attn_k.weight",
        "self_attn.qkv_proj.v_proj_current.weight" => "zaya_v_cur.weight",
        "self_attn.qkv_proj.v_proj_delayed.weight" => "zaya_v_del.weight",
        "self_attn.qkv_proj.conv_qk_depthwise.weight" => "zaya_conv_dw.weight",
        "self_attn.qkv_proj.conv_qk_depthwise.bias" => "zaya_conv_dw.bias",
        "self_attn.qkv_proj.conv_qk_grouped.weight" => "zaya_conv_g.weight",
        "self_attn.qkv_proj.conv_qk_grouped.bias" => "zaya_conv_g.bias",
        "self_attn.qk_norm.temp" => "zaya_temp.weight",
        "post_attention_residual_scale.hidden_states_scale" => "zaya_res_attn_hs.weight",
        "post_attention_residual_scale.hidden_states_bias" => "zaya_res_attn_hb.weight",
        "post_attention_residual_scale.residual_scale" => "zaya_res_attn_rs.weight",
        "post_attention_residual_scale.residual_bias" => "zaya_res_attn_rb.weight",
        "post_mlp_residual_scale.hidden_states_scale" => "zaya_res_mlp_hs.weight",
        "post_mlp_residual_scale.hidden_states_bias" => "zaya_res_mlp_hb.weight",
        "post_mlp_residual_scale.residual_scale" => "zaya_res_mlp_rs.weight",
        "post_mlp_residual_scale.residual_bias" => "zaya_res_mlp_rb.weight",
        "mlp.gate.down_proj.weight" => "zaya_router_down.weight",
        "mlp.gate.down_proj.bias" => "zaya_router_down.bias",
        "mlp.gate.router_states_scale" => "zaya_router_scale.weight",
        "mlp.gate.router_mlp.norm.weight" => "zaya_router_norm.weight",
        "mlp.gate.router_mlp.fc1.weight" => "zaya_router_fc1.weight",
        "mlp.gate.router_mlp.fc1.bias" => "zaya_router_fc1.bias",
        "mlp.gate.router_mlp.fc2.weight" => "zaya_router_fc2.weight",
        "mlp.gate.router_mlp.fc2.bias" => "zaya_router_fc2.bias",
        "mlp.gate.router_mlp.out_proj.weight" => "ffn_gate_inp.weight",
        "mlp.gate.balancing_biases" => "zaya_router_bias.weight",
        "mlp.experts.gate_up_proj" => return fused("experts_input"),
        "mlp.experts.down_proj" => return fused("experts_output"),
        _ => return None,
    };
    Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 })
}

pub(crate) fn layer_requirements(first: bool) -> Vec<&'static str> {
    let mut v = vec![
        "attn_norm.weight", "ffn_norm.weight", "attn_output.weight", "attn_q.weight", "attn_k.weight", "zaya_v_cur.weight", "zaya_v_del.weight",
        "zaya_conv_dw.weight", "zaya_conv_dw.bias", "zaya_conv_g.weight", "zaya_conv_g.bias", "zaya_temp.weight",
        "zaya_res_attn_hs.weight", "zaya_res_attn_hb.weight", "zaya_res_attn_rs.weight", "zaya_res_attn_rb.weight",
        "zaya_res_mlp_hs.weight", "zaya_res_mlp_hb.weight", "zaya_res_mlp_rs.weight", "zaya_res_mlp_rb.weight",
        "zaya_router_down.weight", "zaya_router_down.bias", "zaya_router_norm.weight", "zaya_router_fc1.weight", "zaya_router_fc1.bias",
        "zaya_router_fc2.weight", "zaya_router_fc2.bias", "ffn_gate_inp.weight", "zaya_router_bias.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    ];
    if !first {
        v.push("zaya_router_scale.weight");
    }
    v
}
