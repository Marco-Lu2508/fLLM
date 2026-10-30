//! Inkling (`InklingForCausalLM`, Thinking Machines): attention without rotary embeddings (a position bias that is a learned
//! distance profile mixed per token and head by a projection of the hidden state), short causal convolutions on keys, values and both
//! residual branches, global layers with log-scaled attention and sliding layers with their own head counts, sigmoid-routed MoE layers
//! whose shared experts are weighted by the same normalisation as the routed ones, scaled dense layers and a mu-P output scale.
//!
//! GGUF architecture `inkling`; tensors of `blk.N`: `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_rel` (the bias
//! projection), `attn_output`, `attn_q_norm`, `attn_k_norm`, `attn_rel_profile` (`d_rel x extent`), the convolutions `attn_k_sconv`,
//! `attn_v_sconv`, `attn_sconv`, `ffn_sconv`; a dense layer has `ffn_gate`, `ffn_up`, `ffn_down` and `ffn_scale`; a MoE layer has
//! `ffn_gate_inp`, `ffn_gate_scale`, `exp_probs_b`, `ffn_{gate,up,down}_exps` and the stacked shared experts `ffn_{gate,up,down}_shexp`.
//! The embedding norm is `token_embd_norm`, and `logit_scale` is the inverse of the mu-P multiplier.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

fn layers(cfg: &Value) -> Result<usize, String> {
    num(cfg, "num_hidden_layers").map(|n| n as usize).ok_or_else(|| "config.json requires num_hidden_layers".to_owned())
}

/// Per layer: sliding window layer.
pub(crate) fn sliding_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layers(cfg)?;
    if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
        return Ok(types.iter().take(n).map(|t| t.as_str() == Some("hybrid_sliding")).collect());
    }
    let local: Option<Vec<usize>> = cfg.get("local_layer_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect());
    Ok((0..n).map(|i| local.as_ref().map_or((i + 1) % 6 != 0, |l| l.contains(&i))).collect())
}

/// Per layer: mixture-of-experts layer.
pub(crate) fn sparse_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layers(cfg)?;
    if let Some(types) = cfg.get("mlp_layer_types").and_then(Value::as_array) {
        return Ok(types.iter().take(n).map(|t| t.as_str() == Some("sparse")).collect());
    }
    let dense = num(cfg, "dense_mlp_idx").unwrap_or(0) as usize;
    Ok((0..n).map(|i| i >= dense).collect())
}

/// The configuration for the shape checks of the conversion.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    for key in ["num_key_value_heads", "num_attention_heads", "intermediate_size"] {
        if out.get(key).is_none() {
            out[key] = 1.into();
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let n = layers(cfg)?;
    let llama = json!({
        "model_type": "llama",
        "hidden_size": num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?,
        "num_attention_heads": num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?,
        "num_key_value_heads": num(cfg, "num_key_value_heads").unwrap_or(1),
        "head_dim": num(cfg, "head_dim").unwrap_or(128),
        "intermediate_size": num(cfg, "intermediate_size").unwrap_or(1),
        "num_hidden_layers": n,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": flt(cfg, "rms_norm_eps").unwrap_or(1e-6),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(131072),
        "tie_word_embeddings": false,
    });
    let mut plan = Plan::from_config(&llama)?;
    let (family_arch, arch) = ("llama", "inkling");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let sliding = sliding_layers(cfg)?;
    let sparse = sparse_layers(cfg)?;
    let (routed, shared) = (num(cfg, "n_routed_experts").unwrap_or(0), num(cfg, "n_shared_experts").unwrap_or(0));
    let mup = flt(cfg, "logits_mup_width_multiplier").unwrap_or(1.0);
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    plan.meta.retain(|(k, _)| !k.ends_with(".rope.layer_flags"));
    plan.meta.extend([
        u("swa.head_count", num(cfg, "swa_num_attention_heads").unwrap_or(0)),
        u("swa.head_count_kv", num(cfg, "swa_num_key_value_heads").unwrap_or(0)),
        u("swa.head_dim", num(cfg, "swa_head_dim").unwrap_or(0)),
        u("swa.window", num(cfg, "sliding_window_size").unwrap_or(512)),
        u("rel.dim", num(cfg, "d_rel").unwrap_or(16)),
        u("rel.extent", num(cfg, "rel_extent").unwrap_or(1024)),
        u("log_scaling_floor", num(cfg, "log_scaling_n_floor").unwrap_or(0)),
        f("log_scaling_alpha", flt(cfg, "log_scaling_alpha").unwrap_or(0.1)),
        u("conv_kernel", num(cfg, "sconv_kernel_size").or_else(|| num(cfg, "conv_kernel_size")).unwrap_or(4)),
        u("block_conv_kernel", num(cfg, "conv_kernel_size").unwrap_or(4)),
        (format!("{arch}.layer_sliding"), Meta::Bools(sliding)),
        (format!("{arch}.layer_sparse"), Meta::Bools(sparse.clone())),
        f("logit_scale", 1.0 / mup),
    ]);
    if routed > 0 {
        plan.meta.extend([
            u("moe.routed", routed),
            u("moe.used", num(cfg, "num_experts_per_tok").unwrap_or(1)),
            u("moe.feed_forward_length", num(cfg, "moe_intermediate_size").unwrap_or(1)),
            u("moe.shared", shared),
            f("moe.route_scale", flt(cfg, "route_scale").unwrap_or(1.0)),
        ]);
    }
    plan.moe = routed > 0;
    plan.moe_layers = sparse;
    Ok(plan)
}

pub(crate) fn map(_plan: &Plan, n: &str) -> Result<Mapped, String> {
    let tensor = |name: &str| Ok(Mapped::Tensor { name: name.to_owned(), add: 0.0 });
    match n {
        "model.embed_tokens.weight" => return tensor("token_embd.weight"),
        "model.embed_tokens.embed_norm.weight" => return tensor("token_embd_norm.weight"),
        "model.norm.weight" => return tensor("output_norm.weight"),
        "lm_head.weight" => return tensor("output.weight"),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.").ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    let (layer, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: u64 = layer.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let name = match suf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_query.weight",
        "self_attn.k_proj.weight" => "attn_key.weight",
        "self_attn.v_proj.weight" => "attn_value.weight",
        "self_attn.r_proj.weight" => "attn_rel.weight",
        "self_attn.o_proj.weight" => "attn_out.weight",
        "self_attn.q_norm.weight" => "attn_qn.weight",
        "self_attn.k_norm.weight" => "attn_kn.weight",
        "self_attn.rel_logits_proj.proj" => "attn_rel_profile.weight",
        "self_attn.k_sconv.conv1d.weight" => "attn_k_sconv.weight",
        "self_attn.v_sconv.conv1d.weight" => "attn_v_sconv.weight",
        "attn_sconv.conv1d.weight" => "attn_sconv.weight",
        "mlp_sconv.conv1d.weight" => "ffn_sconv.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.global_scale" => "ffn_scale.weight",
        "mlp.gate.weight" => "ffn_router.weight",
        "mlp.gate.global_scale" => "ffn_router_scale.weight",
        "mlp.gate.e_score_correction_bias" => "ffn_router_bias.bias",
        "mlp.shared_experts.gate_proj" => "ffn_shared_gate.weight",
        "mlp.shared_experts.up_proj" => "ffn_shared_up.weight",
        "mlp.shared_experts.down_proj" => "ffn_shared_down.weight",
        "mlp.experts.gate_up_proj" => return Ok(Mapped::Fused { layer, part: "experts_input".into() }),
        "mlp.experts.down_proj" => return Ok(Mapped::Fused { layer, part: "experts_output".into() }),
        _ => {
            if suf.starts_with("mtp.") || n.contains(".mtp.") {
                return Ok(Mapped::Skip);
            }
            return Err(format!("unrecognized Inkling tensor {n:?}"));
        }
    };
    Ok(Mapped::Tensor { name: format!("blk.{layer}.{name}"), add: 0.0 })
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let sparse = *sparse_layers(cfg).ok()?.get(i as usize)?;
    let mut need = vec![
        "attn_norm.weight", "ffn_norm.weight", "attn_query.weight", "attn_key.weight", "attn_value.weight", "attn_rel.weight", "attn_out.weight", "attn_qn.weight",
        "attn_kn.weight", "attn_rel_profile.weight", "attn_k_sconv.weight", "attn_v_sconv.weight", "attn_sconv.weight", "ffn_sconv.weight",
    ];
    if sparse {
        need.extend([
            "ffn_router.weight", "ffn_router_scale.weight", "ffn_router_bias.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_shared_gate.weight", "ffn_shared_up.weight", "ffn_shared_down.weight",
        ]);
    } else {
        need.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_scale.weight"]);
    }
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_and_names() {
        let cfg = json!({"model_type": "inkling", "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64,
            "intermediate_size": 64, "moe_intermediate_size": 16, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 2,
            "swa_num_attention_heads": 4, "swa_num_key_value_heads": 1, "swa_head_dim": 4, "sliding_window_size": 8, "d_rel": 4, "rel_extent": 16,
            "mlp_layer_types": ["dense", "sparse", "sparse"], "layer_types": ["hybrid_sliding", "hybrid", "hybrid_sliding"], "logits_mup_width_multiplier": 4.0});
        let plan = plan(&cfg).unwrap();
        assert_eq!(plan.arch, "inkling");
        let meta = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert!(matches!(meta("inkling.logit_scale"), Some(Meta::F32(s)) if (s - 0.25).abs() < 1e-7));
        assert!(matches!(meta("inkling.layer_sliding"), Some(Meta::Bools(b)) if b == [true, false, true]));
        assert!(matches!(meta("inkling.moe.shared"), Some(Meta::U32(2))));
        let t = |n: &str| match plan.map(n).unwrap() {
            Mapped::Tensor { name, .. } => name,
            other => format!("{other:?}"),
        };
        assert_eq!(t("model.layers.1.self_attn.rel_logits_proj.proj"), "blk.1.attn_rel_profile.weight");
        assert_eq!(t("model.layers.2.mlp_sconv.conv1d.weight"), "blk.2.ffn_sconv.weight");
        assert_eq!(t("model.embed_tokens.embed_norm.weight"), "token_embd_norm.weight");
        assert_eq!(requirements(&cfg, 0).unwrap().len(), 18);
        assert_eq!(requirements(&cfg, 1).unwrap().len(), 23);
    }
}
