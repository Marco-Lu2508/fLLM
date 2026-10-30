//! Kimi-Linear (`KimiLinearForCausalLM`, moonshotai/Kimi-Linear-*): Kimi Delta Attention layers interleaved with multi-head
//! latent attention layers (no rotary embeddings), dense or DeepSeek-V3 style MoE feed-forwards.
//!
//! The plan is the DeepSeek (`kimi_k2`) plan of a normalised configuration, renamed to the GGUF architecture `kimi-linear`,
//! plus the KDA metadata and tensors (see `rust_decoder/kda.rs` for the GGUF names). Both the Hub layout (`block_sparse_moe.*`
//! experts named `w1`/`w2`/`w3`, separate `q_conv1d`/`k_conv1d`/`v_conv1d`) and the transformers 5 layout are accepted for the
//! tensors that do not need splitting.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::Value;

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

fn first<'a>(cfg: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| cfg.get(*k).filter(|v| !v.is_null()))
}

/// Ling-3.0 (`bailing_hybrid`): the same layer mix, but rotary latent attention with a head-wise output gate and direct (not low-rank)
/// forget and output gate projections of the linear-attention layers.
pub(crate) fn is_ling(cfg: &Value) -> bool {
    cfg.get("model_type").and_then(Value::as_str) == Some("bailing_hybrid") || cfg.get("_ling").and_then(Value::as_bool) == Some(true)
}

fn attn_config(cfg: &Value) -> &Value {
    cfg.get("linear_attn_config").filter(|v| v.is_object()).unwrap_or(&Value::Null)
}

/// Per layer: `true` for KDA, `false` for latent attention.
pub(crate) fn kda_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
        if types.len() != layers {
            return Err("layer_types must have one entry per layer".into());
        }
        return types
            .iter()
            .map(|t| match t.as_str() {
                Some("linear_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                other => Err(format!("unsupported Kimi-Linear layer type {other:?}")),
            })
            .collect();
    }
    let attn = attn_config(cfg);
    let listed = |key: &str| -> Option<Vec<usize>> {
        attn.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
    };
    match (listed("kda_layers"), listed("full_attn_layers")) {
        // the Hub lists layers 1-based
        (Some(kda), Some(full)) => {
            let mut flags = vec![None; layers];
            for (list, value) in [(&kda, true), (&full, false)] {
                for &l in list {
                    *flags.get_mut(l.checked_sub(1).ok_or("layer numbers of linear_attn_config start at 1")?).ok_or("layer number out of range")? = Some(value);
                }
            }
            flags.into_iter().map(|f| f.ok_or_else(|| "linear_attn_config leaves a layer without a type".to_owned())).collect()
        }
        // the default of the configuration class: every fourth layer after the first is full attention
        _ => Ok((0..layers).map(|i| !(i > 0 && i % 4 == 0)).collect()),
    }
}

/// Number of leading dense feed-forward layers (`mlp_layer_types` of transformers 5, `first_k_dense_replace` on the Hub).
fn dense_prefix(cfg: &Value) -> Result<u64, String> {
    if let Some(types) = cfg.get("mlp_layer_types").and_then(Value::as_array) {
        let dense = types.iter().take_while(|t| t.as_str() == Some("dense")).count();
        if types.iter().skip(dense).any(|t| t.as_str() != Some("sparse")) {
            return Err("Kimi-Linear dense feed-forward layers must come first".into());
        }
        return Ok(dense as u64);
    }
    Ok(num(cfg, "first_k_dense_replace").unwrap_or(1))
}

/// The configuration as the DeepSeek plan expects it.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let layers = kda_layers(cfg)?;
    let mut out = cfg.clone();
    out["_ling"] = is_ling(cfg).into();
    out["model_type"] = "kimi_k2".into();
    out["num_hidden_layers"] = (layers.len() as u64).into();
    let copy = |out: &mut Value, to: &str, keys: &[&str]| {
        if let Some(v) = first(cfg, keys).cloned() {
            out[to] = v;
        }
    };
    copy(&mut out, "max_position_embeddings", &["max_position_embeddings", "model_max_length"]);
    copy(&mut out, "num_experts", &["num_experts", "num_local_experts", "n_routed_experts"]);
    copy(&mut out, "num_experts_per_tok", &["num_experts_per_tok", "num_experts_per_token"]);
    copy(&mut out, "n_shared_experts", &["n_shared_experts", "num_shared_experts"]);
    copy(&mut out, "norm_topk_prob", &["norm_topk_prob", "moe_renormalize"]);
    copy(&mut out, "n_group", &["n_group", "num_expert_group"]);
    out["first_k_dense_replace"] = dense_prefix(cfg)?.into();
    out["scoring_func"] = "sigmoid".into();
    if first(cfg, &["moe_router_activation_func"]).and_then(Value::as_str).is_some_and(|f| f != "sigmoid") {
        return Err("Kimi-Linear routers use sigmoid scores only".into());
    }
    if out.get("q_lora_rank").is_none() {
        out["q_lora_rank"] = Value::Null;
    }
    if cfg.get("mla_use_nope").and_then(Value::as_bool) == Some(false) {
        return Err("Kimi-Linear with rotary latent attention (mla_use_nope = false) is unsupported".into());
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let kda = kda_layers(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    let (family_arch, arch) = ("deepseek2", "kimi-linear");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let attn = attn_config(cfg);
    let pick = |hub: &str, hf: &str, default: u64| attn.get(hub).and_then(Value::as_u64).or_else(|| num(cfg, hf)).unwrap_or(default);
    let (heads, head_dim, kernel) = (pick("num_heads", "linear_num_heads", 32), pick("head_dim", "linear_head_dim", 128), pick("short_conv_kernel_size", "linear_conv_kernel_dim", 4));
    if kernel < 2 {
        return Err("the KDA short convolution needs at least two taps".into());
    }
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    plan.meta.retain(|(k, _)| !k.ends_with(".rope.layer_flags"));
    plan.meta.extend([
        u("ssm.conv_kernel", kernel),
        u("kda.head_count", heads),
        u("kda.head_dim", head_dim),
        (format!("{arch}.kda.layer_flags"), Meta::Bools(kda.clone())),
        (format!("{arch}.attention.mla_no_rope"), Meta::Bool(!is_ling(cfg))),
        (format!("{arch}.rope.layer_flags"), Meta::Bools(vec![false; kda.len()])),
    ]);
    if is_ling(cfg) && cfg.get("kda_safe_gate").and_then(Value::as_bool).unwrap_or(true) {
        plan.meta.push((format!("{arch}.kda.gate_lower_bound"), Meta::F32(flt(cfg, "kda_lower_bound").unwrap_or(-5.0) as f32)));
    } else if let Some(lb) = attn.get("gate_lower_bound").and_then(Value::as_f64).or_else(|| cfg.get("safe_gate_lower_bound").and_then(Value::as_f64)).or_else(|| cfg.get("gate_lower_bound").and_then(Value::as_f64)) {
        plan.meta.push((format!("{arch}.kda.gate_lower_bound"), Meta::F32(lb as f32)));
    }
    plan.kinds = kda.iter().map(|&k| if k { b'K' } else { b'A' }).collect();
    Ok(plan)
}

/// Ling-3.0 Hub names to the transformers 5 names (the renames of its conversion mapping); `None` when `n` is not in the Hub layout.
pub(crate) fn normalize(n: &str) -> Option<String> {
    if !(n.contains(".attention.") || n.starts_with("model.word_embeddings.") || n.ends_with(".mlp.gate.expert_bias")) {
        return None;
    }
    Some(
        n.replacen("model.word_embeddings.", "model.embed_tokens.", 1)
            .replace(".attention.dense.", ".self_attn.o_proj.")
            .replace(".attention.", ".self_attn.")
            .replace(".mlp.gate.expert_bias", ".mlp.gate.e_score_correction_bias"),
    )
}

/// Tensor names of Kimi-Linear that the generic plan does not know. `layer` is the layer index, `suf` the name below
/// `model.layers.<layer>.`.
pub(crate) fn map(layer: &str, suf: &str, linear: bool) -> Option<Mapped> {
    let plain = |name: &str| Some(Mapped::Tensor { name: format!("blk.{layer}.{name}"), add: 0.0 });
    let kda = suf.strip_prefix("self_attn.");
    if let Some(rest) = kda {
        // Ling-3.0: the head-wise output gate of the latent attention layers
        if !linear && rest == "g_proj.weight" {
            return plain("attn_gate.weight");
        }
        let rest = rest.strip_prefix("forget_gate.").unwrap_or(rest);
        let name = match rest {
            "conv1d.weight" if linear => return Some(Mapped::Fused { layer: layer.parse().unwrap_or(0), part: "conv1d_qkv".into() }),
            "f_proj.weight" => "ssm_f.weight",
            "g_proj.weight" if linear => "ssm_g.weight",
            "q_conv1d.weight" => "ssm_conv1d_q.weight",
            "k_conv1d.weight" => "ssm_conv1d_k.weight",
            "v_conv1d.weight" => "ssm_conv1d_v.weight",
            "f_a_proj.weight" => "ssm_f_a.weight",
            "f_b_proj.weight" => "ssm_f_b.weight",
            "b_proj.weight" => "ssm_beta.weight",
            "g_a_proj.weight" => "ssm_g_a.weight",
            "g_b_proj.weight" => "ssm_g_b.weight",
            "dt_bias" => "ssm_dt.bias",
            "o_norm.weight" => "ssm_o_norm.weight",
            "A_log" => return Some(Mapped::Fused { layer: layer.parse().unwrap_or(0), part: "A_log".into() }),
            _ => return None,
        };
        return plain(name);
    }
    // Hub names of the experts: w1 = gate, w3 = up, w2 = down
    let moe = suf.strip_prefix("block_sparse_moe.")?;
    match moe {
        // the Hub files name the dense feed-forward of the leading layers like the experts' parent
        "gate_proj.weight" => return plain("ffn_gate.weight"),
        "up_proj.weight" => return plain("ffn_up.weight"),
        "down_proj.weight" => return plain("ffn_down.weight"),
        "gate.weight" => return plain("ffn_gate_inp.weight"),
        "gate.e_score_correction_bias" => return plain("exp_probs_b.bias"),
        "shared_experts.gate_proj.weight" => return plain("ffn_gate_shexp.weight"),
        "shared_experts.up_proj.weight" => return plain("ffn_up_shexp.weight"),
        "shared_experts.down_proj.weight" => return plain("ffn_down_shexp.weight"),
        _ => {}
    }
    let (index, kind) = moe.strip_prefix("experts.")?.split_once('.')?;
    let kind = match kind {
        "w1.weight" => "gate",
        "w3.weight" => "up",
        "w2.weight" => "down",
        _ => return None,
    };
    Some(Mapped::Expert { name: format!("blk.{layer}.ffn_{kind}_exps.weight"), index: index.parse().ok()? })
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let kda = *kda_layers(cfg).ok()?.get(i as usize)?;
    let mut need: Vec<&str> = vec!["attn_norm.weight", "ffn_norm.weight", "attn_output.weight"];
    if kda {
        need.extend([
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
            "ssm_beta.weight", "ssm_dt.bias", "ssm_a", "ssm_o_norm.weight",
        ]);
        if is_ling(cfg) {
            need.extend(["ssm_f.weight", "ssm_g.weight"]);
        } else {
            need.extend(["ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight"]);
        }
    } else {
        if is_ling(cfg) {
            need.push("attn_gate.weight");
        }
        need.extend(["attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_kv_b.weight"]);
        need.push(if num(cfg, "q_lora_rank").is_some_and(|r| r > 0) { "attn_q_a.weight" } else { "attn_q.weight" });
    }
    if i < dense_prefix(cfg).ok()? {
        need.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
    } else {
        need.extend(["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_up_shexp.weight"]);
    }
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hub() -> Value {
        json!({
            "model_type": "kimi_linear", "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4,
            "intermediate_size": 64, "moe_intermediate_size": 16, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
            "v_head_dim": 8, "num_experts": 4, "num_experts_per_token": 2, "num_shared_experts": 1, "moe_renormalize": true,
            "routed_scaling_factor": 2.0, "first_k_dense_replace": 1, "model_max_length": 128, "mla_use_nope": true,
            "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4], "head_dim": 8, "num_heads": 2, "short_conv_kernel_size": 4}
        })
    }

    #[test]
    fn hub_layer_lists_are_one_based() {
        assert_eq!(kda_layers(&hub()).unwrap(), [true, true, true, false]);
        let mut c = hub();
        c["linear_attn_config"]["kda_layers"] = json!([1, 2]);
        assert!(kda_layers(&c).is_err());
        let default = json!({"num_hidden_layers": 9});
        assert_eq!(kda_layers(&default).unwrap(), [true, true, true, true, false, true, true, true, false]);
        let typed = json!({"num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]});
        assert_eq!(kda_layers(&typed).unwrap(), [true, false]);
    }

    #[test]
    fn plan_is_a_deepseek_plan_with_kda_metadata() {
        let plan = plan(&hub()).unwrap();
        assert_eq!(plan.arch, "kimi-linear");
        assert_eq!(plan.kinds, b"KKKA");
        assert!(plan.meta.iter().all(|(k, _)| !k.starts_with("deepseek2.")));
        let get = |k: &str| plan.meta.iter().find(|(n, _)| n == &format!("kimi-linear.{k}")).map(|(_, v)| v.clone());
        assert!(matches!(get("kda.head_count"), Some(Meta::U32(2))));
        assert!(matches!(get("kda.layer_flags"), Some(Meta::Bools(f)) if f == [true, true, true, false]));
        assert!(matches!(get("leading_dense_block_count"), Some(Meta::U32(1))));
        assert!(matches!(get("expert_used_count"), Some(Meta::U32(2))));
        assert!(matches!(get("attention.mla_no_rope"), Some(Meta::Bool(true))));
        assert_eq!(plan.moe_layers, [false, true, true, true]);
    }

    #[test]
    fn hub_tensor_names() {
        let name = |m: Option<Mapped>| match m {
            Some(Mapped::Tensor { name, .. }) => name,
            Some(Mapped::Expert { name, index }) => format!("{name}#{index}"),
            Some(Mapped::Fused { layer, part }) => format!("fused:{layer}:{part}"),
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(name(map("0", "self_attn.q_conv1d.weight", true)), "blk.0.ssm_conv1d_q.weight");
        assert_eq!(name(map("2", "self_attn.A_log", true)), "fused:2:A_log");
        assert_eq!(name(map("2", "self_attn.forget_gate.f_a_proj.weight", true)), "blk.2.ssm_f_a.weight");
        assert_eq!(name(map("1", "block_sparse_moe.experts.3.w3.weight", true)), "blk.1.ffn_up_exps.weight#3");
        assert_eq!(name(map("1", "block_sparse_moe.gate.e_score_correction_bias", true)), "blk.1.exp_probs_b.bias");
        assert_eq!(name(map("0", "block_sparse_moe.down_proj.weight", true)), "blk.0.ffn_down.weight");
        assert!(map("1", "self_attn.q_proj.weight", true).is_none());
        // Ling-3.0: the same tensor name is the output gate of a latent attention layer and of a linear-attention layer
        assert_eq!(name(map("3", "self_attn.g_proj.weight", false)), "blk.3.attn_gate.weight");
        assert_eq!(name(map("2", "self_attn.g_proj.weight", true)), "blk.2.ssm_g.weight");
        assert_eq!(name(map("2", "self_attn.forget_gate.f_proj.weight", true)), "blk.2.ssm_f.weight");
        assert_eq!(name(map("2", "self_attn.conv1d.weight", true)), "fused:2:conv1d_qkv");
        assert!(map("1", "mlp.gate_proj.weight", true).is_none());
    }

    #[test]
    fn requirements_follow_the_layer_kinds() {
        let c = hub();
        let kda = requirements(&c, 1).unwrap();
        assert!(kda.contains(&"blk.1.ssm_o_norm.weight".to_owned()) && kda.contains(&"blk.1.ffn_up_shexp.weight".to_owned()));
        let mla = requirements(&c, 3).unwrap();
        assert!(mla.contains(&"blk.3.attn_kv_b.weight".to_owned()) && !mla.iter().any(|n| n.contains("ssm_")));
        let dense = requirements(&c, 0).unwrap();
        assert!(dense.contains(&"blk.0.ffn_gate.weight".to_owned()));
    }

    #[test]
    fn rotary_latent_attention_is_refused() {
        let mut c = hub();
        c["mla_use_nope"] = false.into();
        assert!(config(&c).is_err());
    }
}
