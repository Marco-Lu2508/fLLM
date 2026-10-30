//! Step-3.5 / Step-3.7 (`model_type: "step3p5"`, stepfun-ai/Step-3.5-Flash and Step-3.7-Flash text models).
//!
//! The layer is the Laguna layer of fLLM: GQA with per-head QK norms, a sigmoid gate per head on the attention output, sliding
//! and full attention layers with their own head counts and rotary parameters, a leading dense MLP, then sigmoid-routed experts
//! with a selection bias, a routed scaling factor and one shared expert. Differences handled here: the norms store `weight - 1`
//! (`plan.norm_add = 1`), and both the transformers configuration and the older Hub configuration (`num_attention_groups`,
//! `moe_num_experts`, `share_expert_dim`, `moe_layers_enum`, `rope_theta` lists, `attention_other_setting`) are rewritten into the
//! Laguna configuration. The Hub tensor names (`moe.*`, `share_expert.*`) are mapped in `Plan::map_moe`.
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn first_num(cfg: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| num(cfg, k))
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

/// The clamp bounds of `swiglu_limits` / `swiglu_limits_shared` (Step-3.7-Flash clamps the last layers): zero or null means none.
fn clamps(cfg: &Value, key: &str) -> bool {
    cfg.get(key).and_then(Value::as_array).is_some_and(|l| l.iter().any(|v| v.as_f64().is_some_and(|f| f != 0.0)))
}

/// Laguna configuration of a Step-3.5 / 3.7 text configuration.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let head_dim = num(cfg, "head_dim").unwrap_or(0);
    if clamps(cfg, "swiglu_limits") || clamps(cfg, "swiglu_limits_shared") {
        return Err("Step-3.7 layers with clamped SwiGLU (swiglu_limits) are unsupported".into());
    }
    if let Some(scalar) = flt(cfg, "query_pre_attn_scalar") {
        if head_dim > 0 && (scalar - head_dim as f64).abs() > 1e-9 {
            return Err("Step-3.5 with query_pre_attn_scalar different from head_dim is unsupported".into());
        }
    }
    if cfg.get("norm_expert_weight").and_then(Value::as_bool) == Some(false) {
        return Err("Step-3.5 routers without normalised expert weights are unsupported".into());
    }
    if cfg.get("moe_router_activation").and_then(Value::as_str).is_some_and(|a| a != "sigmoid") {
        return Err("Step-3.5 routers use sigmoid scores only".into());
    }
    if cfg.get("sink").and_then(Value::as_bool) == Some(true) {
        return Err("Step-3.5 with attention sinks is unsupported".into());
    }
    if cfg.get("gating").and_then(Value::as_bool) == Some(false) || cfg.get("use_head_wise_attn_gate").and_then(Value::as_bool) == Some(false) {
        return Err("Step-3.5 without the attention output gate is unsupported".into());
    }
    let layer_types: Vec<String> = match cfg.get("layer_types").and_then(Value::as_array) {
        Some(types) if types.len() >= layers => types.iter().take(layers).map(|t| t.as_str().unwrap_or("full_attention").to_owned()).collect(),
        _ => vec!["full_attention".to_owned(); layers],
    };
    let sliding_heads = num(cfg, "num_sliding_attention_heads")
        .or_else(|| cfg.get("attention_other_setting").and_then(|o| num(o, "num_attention_heads")))
        .unwrap_or(heads);
    let per_layer: Vec<u64> = layer_types.iter().map(|t| if t == "sliding_attention" { sliding_heads } else { heads }).collect();

    let mut out = cfg.clone();
    out["model_type"] = "laguna".into();
    out["num_attention_heads_per_layer"] = json!(per_layer);
    out["layer_types"] = json!(layer_types);
    out["num_key_value_heads"] = first_num(cfg, &["num_key_value_heads", "num_attention_groups"]).ok_or("config.json requires num_key_value_heads")?.into();
    out["num_experts"] = first_num(cfg, &["n_routed_experts", "moe_num_experts", "num_local_experts", "num_experts"]).ok_or("config.json requires the number of experts")?.into();
    out["num_experts_per_tok"] = first_num(cfg, &["num_experts_per_tok", "moe_top_k"]).ok_or("config.json requires num_experts_per_tok")?.into();
    if let Some(shared) = first_num(cfg, &["share_expert_dim", "share_expert_dims", "shared_expert_intermediate_size"]) {
        out["shared_expert_intermediate_size"] = shared.into();
    }
    out["moe_routed_scaling_factor"] = flt(cfg, "moe_router_scaling_factor").or_else(|| flt(cfg, "moe_routed_scaling_factor")).unwrap_or(1.0).into();
    if num(cfg, "max_position_embeddings").is_none() {
        if let Some(n) = num(cfg, "max_seq_len") {
            out["max_position_embeddings"] = n.into();
        }
    }
    out["gating"] = true.into();
    out["hidden_act"] = "silu".into();
    if cfg.get("rms_norm_eps").is_none() {
        out["rms_norm_eps"] = 1e-5.into();
    }
    // which layers carry experts
    let mlp_types: Vec<&str> = match cfg.get("mlp_layer_types").and_then(Value::as_array) {
        Some(types) if types.len() >= layers => types.iter().take(layers).map(|t| t.as_str().unwrap_or("dense")).collect(),
        _ => {
            let enumerated: Option<Vec<u64>> = match cfg.get("moe_layers_enum") {
                Some(Value::String(s)) => Some(s.split(',').filter_map(|p| p.trim().parse().ok()).collect()),
                Some(Value::Array(a)) => Some(a.iter().filter_map(Value::as_u64).collect()),
                _ => None,
            };
            (0..layers as u64)
                .map(|i| match &enumerated {
                    Some(set) => set.contains(&i),
                    None => i >= 3,
                })
                .map(|moe| if moe { "sparse" } else { "dense" })
                .collect()
        }
    };
    out["mlp_layer_types"] = json!(mlp_types);
    // rotary parameters per layer kind
    let by_type = |rp: &Value| layer_types.iter().any(|t| rp.get(t).is_some_and(Value::is_object));
    if !cfg.get("rope_parameters").is_some_and(|rp| rp.is_object() && by_type(rp)) {
        let pick = |key: &str, kind: &str| -> Option<Value> {
            let index = layer_types.iter().position(|t| t == kind)?;
            match cfg.get(key)? {
                Value::Array(list) => list.get(index).cloned(),
                other if !other.is_null() => Some(other.clone()),
                _ => None,
            }
        };
        let mut params = serde_json::Map::new();
        for kind in ["full_attention", "sliding_attention"] {
            if !layer_types.iter().any(|t| t == kind) {
                continue;
            }
            let mut p = json!({"rope_type": "default", "rope_theta": pick("rope_theta", kind).unwrap_or(json!(10000.0))});
            if let Some(f) = pick("partial_rotary_factors", kind) {
                p["partial_rotary_factor"] = f;
            }
            if kind == "full_attention" {
                if let Some(scaling) = cfg.get("rope_scaling").filter(|s| s.is_object()) {
                    for (k, v) in scaling.as_object().unwrap() {
                        p[k] = v.clone();
                    }
                }
            }
            params.insert(kind.to_owned(), p);
        }
        out["rope_parameters"] = Value::Object(params);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hub() -> Value {
        json!({
            "model_type": "step3p5", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 8, "num_attention_groups": 2, "head_dim": 8,
            "moe_num_experts": 8, "moe_top_k": 2, "moe_intermediate_size": 16, "share_expert_dim": 24, "moe_router_scaling_factor": 3.0,
            "moe_layers_enum": "1,2,3", "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "attention_other_setting": {"num_attention_heads": 12, "num_attention_groups": 2}, "sliding_window": 16, "max_seq_len": 512,
            "rope_theta": [5.0e6, 1.0e4, 1.0e4, 5.0e6], "partial_rotary_factors": [0.5, 1.0, 1.0, 0.5], "intermediate_size": 32,
            "rope_scaling": {"rope_type": "llama3", "factor": 2.0}
        })
    }

    #[test]
    fn hub_configuration_becomes_a_laguna_configuration() {
        let c = config(&hub()).unwrap();
        assert_eq!(c["model_type"], "laguna");
        assert_eq!(c["num_attention_heads_per_layer"], json!([8, 12, 12, 8]));
        assert_eq!(c["mlp_layer_types"], json!(["dense", "sparse", "sparse", "sparse"]));
        assert_eq!(c["num_experts"], 8);
        assert_eq!(c["shared_expert_intermediate_size"], 24);
        assert_eq!(c["moe_routed_scaling_factor"], 3.0);
        assert_eq!(c["max_position_embeddings"], 512);
        assert_eq!(c["rope_parameters"]["sliding_attention"]["rope_theta"], 1.0e4);
        assert_eq!(c["rope_parameters"]["full_attention"]["partial_rotary_factor"], 0.5);
        assert_eq!(c["rope_parameters"]["full_attention"]["rope_type"], "llama3");
        assert_eq!(c["rope_parameters"]["sliding_attention"]["rope_type"], "default");
    }

    #[test]
    fn unsupported_variants_are_refused() {
        let mut c = hub();
        c["swiglu_limits"] = json!([0.0, 7.0, 0.0, 0.0]);
        assert!(config(&c).unwrap_err().contains("swiglu"));
        let mut c = hub();
        c["sink"] = true.into();
        assert!(config(&c).unwrap_err().contains("sinks"));
        let mut c = hub();
        c["moe_router_activation"] = "softmax".into();
        assert!(config(&c).is_err());
    }
}
