//! Doge (`DogeForCausalLM`): Llama-style blocks with three changes. (1) Dynamic mask attention: every key gets a decay `exp(A * softplus(dt_proj(value)))` per key head that
//! is added to the attention scores, and a query keeps only the `keep_window_size` keys with the largest decay (the others are masked). Queries and keys are RMS-normalised
//! per head before the rotary embedding. (2) Every layer has two learned vectors that scale the residual stream (`x = input_residual * x + attention`, likewise behind
//! the feed-forward). (3) The feed-forward is either dense or a cross-domain mixture of experts: a shared gated expert plus `top_k` of `num_experts` single-vector experts
//! found by a product-key lookup (`router_gate` scores two key halves, `down_embed` / `up_embed` are the expert vectors). The plan is the Llama plan renamed to `doge`.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

/// The Llama configuration of a Doge checkpoint (the dense feed-forward, or the shared expert of the mixture).
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let mut out = json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?,
        "num_attention_heads": heads,
        "num_key_value_heads": num(cfg, "num_key_value_heads").unwrap_or(heads),
        "num_hidden_layers": num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?,
        "intermediate_size": num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": cfg.get("rms_norm_eps").cloned().unwrap_or(1e-6.into()),
        "rope_theta": cfg.get("rope_theta").cloned().or_else(|| cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).cloned()).unwrap_or(10000.0.into()),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(2048),
        "hidden_act": cfg.get("hidden_act").cloned().unwrap_or("silu".into()),
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
        "attention_bias": cfg.get("attention_bias").and_then(Value::as_bool).unwrap_or(false),
        "mlp_bias": cfg.get("mlp_bias").and_then(Value::as_bool).unwrap_or(false),
    });
    if let Some(d) = num(cfg, "head_dim") {
        out["head_dim"] = d.into();
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
        *key = key.replace("llama.", "doge.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("doge".into());
        }
    }
    plan.arch = "doge".into();
    plan.meta.push(("doge.attention.keep_window_size".into(), Meta::U32(num(cfg, "keep_window_size").unwrap_or(2048) as u32)));
    if cfg.get("is_moe").and_then(Value::as_bool).unwrap_or(false) {
        let experts = num(cfg, "num_experts").ok_or("Doge with is_moe requires num_experts")?;
        let keys = (experts as f64).sqrt().floor() as u64;
        let top_k = num(cfg, "num_experts_per_tok").ok_or("Doge with is_moe requires num_experts_per_tok")?;
        if keys == 0 || top_k == 0 || top_k > keys * keys {
            return Err("Doge: num_experts_per_tok must be between 1 and floor(sqrt(num_experts))^2".into());
        }
        plan.meta.push(("doge.expert_count".into(), Meta::U32(experts as u32)));
        plan.meta.push(("doge.expert_used_count".into(), Meta::U32(top_k as u32)));
        plan.meta.push(("doge.norm_topk_prob".into(), Meta::Bool(cfg.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(false))));
    }
    Ok(plan)
}

/// GGUF name of the per-layer tensors Doge adds to (or renames in) the Llama set.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let name = match suf {
        "input_residual" => "doge_input_residual.weight",
        "post_attention_residual" => "doge_post_residual.weight",
        "self_attn.A" => "doge_a.weight",
        "self_attn.dt_proj.weight" => "doge_dt.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.router_gate.weight" => "doge_router.weight",
        "mlp.down_embed.weight" => "doge_down_embed.weight",
        "mlp.up_embed.weight" => "doge_up_embed.weight",
        _ => return None,
    };
    Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 })
}

pub(crate) fn layer_requirements(moe: bool) -> Vec<&'static str> {
    let mut v = vec!["doge_input_residual.weight", "doge_post_residual.weight", "doge_a.weight", "doge_dt.weight", "attn_q_norm.weight", "attn_k_norm.weight"];
    if moe {
        v.extend(["doge_router.weight", "doge_down_embed.weight", "doge_up_embed.weight"]);
    }
    v
}
