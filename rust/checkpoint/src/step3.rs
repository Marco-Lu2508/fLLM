//! Step-3 text model (`Step3vForConditionalGeneration` of stepfun-ai/step3; the vision tower is not imported): multi-query attention with one key / value head of
//! `head_dim` values, the query goes through a low-rank pair (`q_proj` to `share_q_dim`, an RMS norm `inter_norm`, then `wq` to `heads * head_dim`), rotary
//! embeddings over the whole head, scores scaled by `head_dim^-0.5`. The feed-forward is dense (`mlp`) or, in the layers of `moe_layers_enum`, softmax-routed
//! experts (stacked `gate_proj` / `up_proj` / `down_proj`, the weights are the routing probabilities as they are) plus a shared expert that always runs.
//!
//! The plan is the Mixtral plan of a normalised configuration renamed to the GGUF architecture `step3`.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

/// The layers with experts: `moe_layers_enum` ("1,2,3") or every layer but the first.
fn moe_layers(cfg: &Value, layers: u64) -> Vec<bool> {
    match cfg.get("moe_layers_enum").and_then(Value::as_str) {
        Some(list) => {
            let idx: Vec<u64> = list.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            (0..layers).map(|l| idx.contains(&l)).collect()
        }
        None => (0..layers).map(|l| l >= 1).collect(),
    }
}

pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let rope = cfg.get("rope_scaling").filter(|v| !v.is_null());
    if rope.is_some() {
        return Err("Step-3 with rope_scaling is not supported".into());
    }
    let mut out = json!({
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": 1,
        "head_dim": num(cfg, "head_dim").unwrap_or(hidden / heads.max(1)),
        "num_hidden_layers": num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?,
        "intermediate_size": num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": cfg.get("rms_norm_eps").cloned().unwrap_or(1e-5.into()),
        "rope_theta": cfg.get("rope_theta").cloned().unwrap_or(500000.0.into()),
        "max_position_embeddings": num(cfg, "max_position_embedding").or_else(|| num(cfg, "max_position_embeddings")).unwrap_or(65536),
        "hidden_act": "silu",
        "num_local_experts": num(cfg, "moe_num_experts").ok_or("config.json requires moe_num_experts")?,
        "num_experts_per_tok": num(cfg, "moe_top_k").ok_or("config.json requires moe_top_k")?,
        "tie_word_embeddings": false,
    });
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
        *key = key.replace("llama.", "step3.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("step3".into());
        }
    }
    plan.arch = "step3".into();
    let share = num(cfg, "share_expert_dim").ok_or("config.json requires share_expert_dim")? as u32;
    let moe_ff = num(cfg, "moe_intermediate_size").ok_or("config.json requires moe_intermediate_size")? as u32;
    plan.meta.retain(|(k, _)| !matches!(k.as_str(), "step3.expert_feed_forward_length" | "step3.expert_weights_norm" | "step3.expert_shared_count" | "step3.expert_shared_feed_forward_length"));
    plan.meta.push(("step3.expert_feed_forward_length".into(), Meta::U32(moe_ff)));
    plan.meta.push(("step3.expert_weights_norm".into(), Meta::Bool(false)));
    plan.meta.push(("step3.expert_shared_count".into(), Meta::U32(1)));
    plan.meta.push(("step3.expert_shared_feed_forward_length".into(), Meta::U32(share)));
    plan.meta.push(("step3.attention.q_low_rank".into(), Meta::U32(num(cfg, "share_q_dim").ok_or("config.json requires share_q_dim")? as u32)));
    plan.moe_layers = moe_layers(cfg, num(cfg, "num_hidden_layers").unwrap_or(0));
    Ok(plan)
}

/// GGUF name of a per-layer tensor of layer `i` with the checkpoint suffix `suf`.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let name = match suf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.inter_norm.weight" => "attn_q_inter_norm.weight",
        "self_attn.wq.weight" => "attn_q_low.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "moe.gate.weight" => "ffn_gate_inp.weight",
        "moe.gate_proj.weight" => "ffn_gate_exps.weight",
        "moe.up_proj.weight" => "ffn_up_exps.weight",
        "moe.down_proj.weight" => "ffn_down_exps.weight",
        "share_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
        "share_expert.up_proj.weight" => "ffn_up_shexp.weight",
        "share_expert.down_proj.weight" => "ffn_down_shexp.weight",
        _ => return None,
    };
    Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 })
}

pub(crate) fn layer_requirements(moe: bool) -> Vec<&'static str> {
    let mut v = vec!["attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_inter_norm.weight", "attn_q_low.weight"];
    if moe {
        v.extend(["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]);
    } else {
        v.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
    }
    v
}
