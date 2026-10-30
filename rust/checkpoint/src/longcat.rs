//! LongCat-Flash (`LongcatFlashForCausalLM`, meituan-longcat): every layer holds two latent-attention sublayers, each followed by a dense
//! MLP, and a mixture-of-experts "shortcut" that reads the input of the first MLP and joins the residual stream after the second
//! sublayer. The router scores `n_routed_experts + zero_expert_num` experts; a zero expert returns its input.
//!
//! The GGUF architecture `longcat-flash` has `2 * num_layers` blocks, one per sublayer: `blk.2l + s` carries the attention
//! (`attn_*`, the usual latent-attention names) and the dense MLP (`ffn_*`) of sublayer `s` of layer `l`; the shortcut MoE sits on the
//! even block as `shortcut_gate_inp`, `shortcut_exp_probs_b`, `shortcut_{gate,up,down}_exps`. The LoRA scalings
//! `sqrt(hidden / q_lora_rank)` and `sqrt(hidden / kv_lora_rank)` of the latent attention are metadata (`attention.q_lora_scale`,
//! `attention.kv_lora_scale`) that the engine folds into the norm weights.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

/// Number of blocks of the GGUF file.
fn blocks(cfg: &Value) -> Result<u64, String> {
    num(cfg, "num_layers").map(|l| 2 * l).or_else(|| num(cfg, "num_hidden_layers")).ok_or_else(|| "config.json requires num_layers".to_owned())
}

/// The configuration for the shape checks of the conversion.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    out["num_hidden_layers"] = blocks(cfg)?.into();
    out["intermediate_size"] = num(cfg, "ffn_hidden_size").ok_or("config.json requires ffn_hidden_size")?.into();
    out["num_key_value_heads"] = num(cfg, "num_key_value_heads").or_else(|| num(cfg, "num_attention_heads")).ok_or("config.json requires num_attention_heads")?.into();
    Ok(out)
}

/// A DeepSeek-style latent-attention configuration without any expert layer.
fn deepseek_config(cfg: &Value) -> Result<Value, String> {
    let layers = blocks(cfg)?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let mut out = json!({
        "model_type": "deepseek_v3",
        "hidden_size": num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?,
        "num_attention_heads": heads,
        "num_key_value_heads": num(cfg, "num_key_value_heads").unwrap_or(heads),
        "intermediate_size": num(cfg, "ffn_hidden_size").ok_or("config.json requires ffn_hidden_size")?,
        "num_hidden_layers": layers,
        "first_k_dense_replace": layers,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": flt(cfg, "rms_norm_eps").unwrap_or(1e-5),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(131072),
        "q_lora_rank": num(cfg, "q_lora_rank").ok_or("config.json requires q_lora_rank")?,
        "kv_lora_rank": num(cfg, "kv_lora_rank").ok_or("config.json requires kv_lora_rank")?,
        "qk_nope_head_dim": num(cfg, "qk_nope_head_dim").ok_or("config.json requires qk_nope_head_dim")?,
        "qk_rope_head_dim": num(cfg, "qk_rope_head_dim").ok_or("config.json requires qk_rope_head_dim")?,
        "v_head_dim": num(cfg, "v_head_dim").ok_or("config.json requires v_head_dim")?,
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
        // the plan wants an expert configuration; no layer of the plan is an expert layer
        "n_routed_experts": num(cfg, "n_routed_experts").unwrap_or(1),
        "num_experts_per_tok": 1,
        "moe_intermediate_size": num(cfg, "expert_ffn_hidden_size").unwrap_or(1),
        "n_shared_experts": 0,
    });
    for key in ["rope_parameters", "rope_theta", "rope_scaling", "bos_token_id", "eos_token_id", "pad_token_id"] {
        if let Some(v) = cfg.get(key) {
            out[key] = v.clone();
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let mut plan = Plan::from_config(&deepseek_config(cfg)?)?;
    let (family_arch, arch) = ("deepseek2", "longcat-flash");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    plan.meta.retain(|(k, _)| !k.contains(".expert_") && !k.ends_with(".leading_dense_block_count"));
    let routed = num(cfg, "n_routed_experts").ok_or("config.json requires n_routed_experts")?;
    let zero = num(cfg, "zero_expert_num").unwrap_or(0);
    let hidden = num(cfg, "hidden_size").unwrap_or(1) as f64;
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    plan.meta.extend([
        u("expert_count", routed + zero),
        u("expert_zero_count", zero),
        u("expert_used_count", num(cfg, "moe_topk").or_else(|| num(cfg, "num_experts_per_tok")).ok_or("config.json requires moe_topk")?),
        u("expert_feed_forward_length", num(cfg, "expert_ffn_hidden_size").ok_or("config.json requires expert_ffn_hidden_size")?),
        f("expert_weights_scale", flt(cfg, "routed_scaling_factor").unwrap_or(6.0)),
        (format!("{arch}.expert_weights_norm"), Meta::Bool(false)),
        u("expert_gating_func", 1),
        f("attention.q_lora_scale", (hidden / num(cfg, "q_lora_rank").unwrap_or(1) as f64).sqrt()),
        f("attention.kv_lora_scale", (hidden / num(cfg, "kv_lora_rank").unwrap_or(1) as f64).sqrt()),
    ]);
    plan.moe_layers = vec![false; plan.layers as usize];
    Ok(plan)
}

/// Routed experts (those with a down projection in the checkpoint) of the plan.
fn routed(plan: &Plan) -> Option<usize> {
    let get = |key: &str| {
        plan.meta.iter().find_map(|(k, v)| match v {
            Meta::U32(n) if k == key => Some(*n as usize),
            _ => None,
        })
    };
    Some(get("longcat-flash.expert_count")? - get("longcat-flash.expert_zero_count")?)
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
    let tensor = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
    match n {
        "model.embed_tokens.weight" => return tensor("token_embd.weight".into()),
        "model.norm.weight" => return tensor("output_norm.weight".into()),
        "lm_head.weight" => return tensor("output.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.").ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    let (layer, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: u64 = layer.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let shortcut = |name: &str| tensor(format!("blk.{}.{name}", 2 * layer));
    // sublayer tensors: `<what>.<sublayer>.<rest>`
    let sub = |what: &str| -> Option<(u64, &str)> {
        let tail = suf.strip_prefix(what)?.strip_prefix('.')?;
        let (s, tail) = tail.split_once('.')?;
        Some((s.parse().ok().filter(|s| *s < 2)?, tail))
    };
    if let Some((s, tail)) = sub("self_attn") {
        let name = match tail {
            "q_a_proj.weight" => "attn_q_a.weight",
            "q_a_layernorm.weight" => "attn_q_a_norm.weight",
            "q_b_proj.weight" => "attn_q_b.weight",
            "kv_a_proj_with_mqa.weight" => "attn_kv_a_mqa.weight",
            "kv_a_layernorm.weight" => "attn_kv_a_norm.weight",
            "kv_b_proj.weight" => "attn_kv_b.weight",
            "o_proj.weight" => "attn_output.weight",
            _ => return Err(format!("unrecognized LongCat attention tensor {n:?}")),
        };
        return tensor(format!("blk.{}.{name}", 2 * layer + s));
    }
    if let Some((s, tail)) = sub("mlps") {
        let name = match tail {
            "gate_proj.weight" => "ffn_gate.weight",
            "up_proj.weight" => "ffn_up.weight",
            "down_proj.weight" => "ffn_down.weight",
            _ => return Err(format!("unrecognized LongCat MLP tensor {n:?}")),
        };
        return tensor(format!("blk.{}.{name}", 2 * layer + s));
    }
    if let Some((s, tail)) = sub("input_layernorm") {
        return if tail == "weight" { tensor(format!("blk.{}.attn_norm.weight", 2 * layer + s)) } else { Err(format!("unrecognized tensor {n:?}")) };
    }
    if let Some((s, tail)) = sub("post_attention_layernorm") {
        return if tail == "weight" { tensor(format!("blk.{}.ffn_norm.weight", 2 * layer + s)) } else { Err(format!("unrecognized tensor {n:?}")) };
    }
    match suf {
        "input_layernorm.weight" | "post_attention_layernorm.weight" => return Err(format!("unrecognized tensor {n:?}")),
        "mlp.router.classifier.weight" => return shortcut("shortcut_gate_inp.weight"),
        "mlp.router.e_score_correction_bias" => return shortcut("shortcut_exp_probs_b.bias"),
        _ => {}
    }
    if let Some((index, kind)) = suf.strip_prefix("mlp.experts.").and_then(|r| r.split_once('.')) {
        let index: usize = index.parse().map_err(|_| format!("invalid expert index in {n:?}"))?;
        // the zero experts keep unused gate and up projections in some exports: they are dropped
        if routed(plan).is_some_and(|r| index >= r) {
            return Ok(Mapped::Skip);
        }
        let kind = match kind {
            "gate_proj.weight" | "w1.weight" => "gate",
            "up_proj.weight" | "w3.weight" => "up",
            "down_proj.weight" | "w2.weight" => "down",
            _ => return Err(format!("unrecognized LongCat expert tensor {n:?}")),
        };
        return Ok(Mapped::Expert { name: format!("blk.{}.shortcut_{kind}_exps.weight", 2 * layer), index });
    }
    Err(format!("unrecognized LongCat tensor {n:?}"))
}

/// GGUF tensors block `i` must have after conversion.
pub(crate) fn requirements(_cfg: &Value, i: u64) -> Option<Vec<String>> {
    let mut need = vec![
        "attn_norm.weight", "ffn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
        "attn_kv_b.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ];
    if i.is_multiple_of(2) {
        need.extend(["shortcut_gate_inp.weight", "shortcut_exp_probs_b.bias", "shortcut_gate_exps.weight", "shortcut_up_exps.weight", "shortcut_down_exps.weight"]);
    }
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Value {
        json!({"model_type": "longcat_flash", "hidden_size": 32, "num_layers": 2, "num_attention_heads": 4, "ffn_hidden_size": 64, "expert_ffn_hidden_size": 16,
            "n_routed_experts": 4, "zero_expert_num": 2, "moe_topk": 2, "q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
            "vocab_size": 64})
    }

    #[test]
    fn sublayers_become_blocks_and_the_shortcut_sits_on_the_even_one() {
        let plan = plan(&cfg()).unwrap();
        assert_eq!(plan.arch, "longcat-flash");
        assert_eq!(plan.layers, 4);
        let t = |n: &str| match plan.map(n).unwrap() {
            Mapped::Tensor { name, .. } => name,
            Mapped::Expert { name, index } => format!("{name}#{index}"),
            other => format!("{other:?}"),
        };
        assert_eq!(t("model.layers.1.self_attn.1.q_a_proj.weight"), "blk.3.attn_q_a.weight");
        assert_eq!(t("model.layers.1.mlps.0.down_proj.weight"), "blk.2.ffn_down.weight");
        assert_eq!(t("model.layers.1.input_layernorm.1.weight"), "blk.3.attn_norm.weight");
        assert_eq!(t("model.layers.1.mlp.router.classifier.weight"), "blk.2.shortcut_gate_inp.weight");
        assert_eq!(t("model.layers.0.mlp.experts.3.up_proj.weight"), "blk.0.shortcut_up_exps.weight#3");
        assert_eq!(t("model.layers.0.mlp.experts.4.up_proj.weight"), "Skip");
        let meta = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert!(matches!(meta("longcat-flash.expert_count"), Some(Meta::U32(6))));
        assert!(matches!(meta("longcat-flash.block_count"), Some(Meta::U32(4))));
        assert_eq!(requirements(&cfg(), 2).unwrap().len(), 17);
        assert_eq!(requirements(&cfg(), 3).unwrap().len(), 12);
    }
}
