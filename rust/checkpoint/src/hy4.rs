//! Hy4 (`HYV4ForCausalLM`, Tencent Hy4): DeepSeek-V3.2 style latent attention with the lightning indexer (layers marked `shared` reuse the
//! selection of the layer before), a per-element sigmoid gate and sinks on the attention output, rotary embeddings that rotate
//! half-split pairs, independent hyper-connections (`hc_mult` residual streams without a mixing matrix) and a sigmoid-routed MoE whose
//! routed experts use a clamped SwiGLU.
//!
//! The plan is the one of `glm_moe_dsa` of the same configuration, renamed to the GGUF architecture `hy4`; the indexer rotates the
//! trailing `qk_rope_head_dim` values of its heads (`attention.indexer.rope_last`) with half-split pairs and its key norm uses the model's
//! epsilon. Extra tensors per layer: `hc_attn_{fn,base,scale}`, `hc_ffn_{fn,base,scale}`; globally `output_hc_{fn,base,scale}`.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::Value;

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

/// The configuration as the `glm_moe_dsa` plan expects it.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    out["model_type"] = "glm_moe_dsa".into();
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?;
    if out.get("mlp_layer_types").is_none() {
        out["mlp_layer_types"] = (0..layers).map(|i| if i == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>().into();
    }
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    out["num_key_value_heads"] = heads.into();
    // the dense layers' width is `intermediate_size`, the experts' `moe_intermediate_size`
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    let (family_arch, arch) = ("deepseek2", "hy4");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    plan.meta.retain(|(k, _)| !k.ends_with("indexer.rope_neox"));
    plan.meta.extend([
        (format!("{arch}.attention.indexer.rope_neox"), Meta::Bool(true)),
        (format!("{arch}.attention.indexer.rope_last"), Meta::Bool(true)),
        f("attention.indexer.norm_eps", flt(cfg, "rms_norm_eps").unwrap_or(1e-5)),
        u("hc_mult", num(cfg, "hc_mult").unwrap_or(4)),
        f("hc_magnitude", flt(cfg, "hc_magnitude").unwrap_or(2.0)),
        f("hc_eps", flt(cfg, "hc_eps").unwrap_or(1e-6)),
    ]);
    let limit = flt(cfg, "swiglu_limit").unwrap_or(10.0);
    if limit > 0.0 {
        plan.meta.push((format!("{arch}.expert_swiglu_clamp"), Meta::F32(limit as f32)));
        // the shared expert is a plain SwiGLU
        plan.meta.push((format!("{arch}.expert_swiglu_clamp_shared"), Meta::Bool(false)));
    }
    Ok(plan)
}

/// Hub renames of the transformers conversion mapping, then the hyper-connection tensors; `None` for everything the generic plan maps.
pub(crate) fn map_hc(n: &str) -> Option<Mapped> {
    let hf = n
        .replace(".hc_pre.hc_", ".hc_")
        .replace(".learnable_sink_param", ".sinks")
        .replace(".linear_gate", ".gate_proj")
        .replace(".hc_attn_layer.hc_", ".attn_hc.")
        .replace(".hc_mlp_layer.hc_", ".ffn_hc.")
        .replace(".hc_head_fn", ".hc_fn")
        .replace(".hc_head_base", ".hc_base")
        .replace(".hc_head_scale", ".hc_scale");
    let tensor = |name: String| Some(Mapped::Tensor { name, add: 0.0 });
    match hf.as_str() {
        "model.hc_head.hc_fn" => return tensor("output_hc_fn.weight".into()),
        "model.hc_head.hc_base" => return tensor("output_hc_base.weight".into()),
        "model.hc_head.hc_scale" => return tensor("output_hc_scale.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, suf) = rest.split_once('.')?;
    let name = match suf {
        "attn_hc.fn" => "hc_attn_fn.weight",
        "attn_hc.base" => "hc_attn_base.weight",
        "attn_hc.scale" => "hc_attn_scale.weight",
        "ffn_hc.fn" => "hc_ffn_fn.weight",
        "ffn_hc.base" => "hc_ffn_base.weight",
        "ffn_hc.scale" => "hc_ffn_scale.weight",
        _ => return None,
    };
    tensor(format!("blk.{layer}.{name}"))
}

/// The tensor name after the Hub renames that the generic plan understands (`None`: unchanged).
pub(crate) fn rename(n: &str) -> Option<String> {
    let renamed = n.replace(".learnable_sink_param", ".sinks").replace(".linear_gate", ".gate_proj");
    (renamed != n).then_some(renamed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maps_the_hyper_connection_tensors_of_both_layouts() {
        let t = |m: Option<Mapped>| match m {
            Some(Mapped::Tensor { name, .. }) => name,
            other => panic!("{other:?}"),
        };
        assert_eq!(t(map_hc("model.layers.2.attn_hc.fn")), "blk.2.hc_attn_fn.weight");
        assert_eq!(t(map_hc("model.layers.2.hc_mlp_layer.hc_scale")), "blk.2.hc_ffn_scale.weight");
        assert_eq!(t(map_hc("model.layers.2.hc_pre.hc_attn_layer.hc_base")), "blk.2.hc_attn_base.weight");
        assert_eq!(t(map_hc("model.hc_head.hc_fn")), "output_hc_fn.weight");
        assert!(map_hc("model.layers.2.self_attn.q_a_proj.weight").is_none());
        assert_eq!(rename("model.layers.0.self_attn.learnable_sink_param").as_deref(), Some("model.layers.0.self_attn.sinks"));
    }

    #[test]
    fn plan_is_the_latent_attention_plan_with_the_hy4_metadata() {
        let cfg = json!({"model_type": "hy_v4", "hidden_size": 32, "num_attention_heads": 4, "num_hidden_layers": 3, "intermediate_size": 64, "moe_intermediate_size": 16,
            "vocab_size": 64, "n_routed_experts": 4, "num_experts_per_tok": 2, "kv_lora_rank": 8, "q_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
            "v_head_dim": 8, "index_n_heads": 2, "index_head_dim": 8, "index_topk": 5, "hc_mult": 3, "swiglu_limit": 2.0});
        let plan = plan(&cfg).unwrap();
        assert_eq!(plan.arch, "hy4");
        let meta = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert!(matches!(meta("hy4.attention.indexer.top_k"), Some(Meta::U32(5))));
        assert!(matches!(meta("hy4.attention.indexer.rope_last"), Some(Meta::Bool(true))));
        assert!(matches!(meta("hy4.hc_mult"), Some(Meta::U32(3))));
        assert!(matches!(meta("hy4.expert_swiglu_clamp_shared"), Some(Meta::Bool(false))));
        assert_eq!(plan.map("model.layers.1.self_attn.q_a_proj.weight").unwrap(), Mapped::Tensor { name: "blk.1.attn_q_a.weight".into(), add: 0.0 });
    }
}
