//! A.X-K2 (`AXK2ForCausalLM`): DeepSeek-V3.2 (MLA with a sparse-attention indexer, shared + routed experts) with three changes. The query projection is one matrix over
//! `[normalised, raw]` low-rank query (`q_gate_proj`) whose rows per head are the query followed by an output gate (a sigmoid over the attention output of the head);
//! the input norm of every layer and the post-attention norm of the expert layers are RMS norms followed by a low-rank sigmoid gate (`y * sigmoid(fc2(silu(fc1(y))))`);
//! there is no norm behind the experts. The plan is the DeepSeek-V3.2 plan of a normalised configuration; the gate rows are moved behind the query rows
//! (`attn_q_b.weight`: all queries, then all gates) and the metadata `attention.q_gate` / `gated_norm_rank` switch the engine to the new parts.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    out["model_type"] = "deepseek_v32".into();
    out["architectures"] = json!(["DeepseekV32ForCausalLM"]);
    let types: Vec<&str> = cfg.get("mlp_layer_types").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let dense = types.iter().take_while(|t| **t == "dense").count();
    if types.iter().skip(dense).any(|t| *t != "sparse") {
        return Err("A.X-K2 supports dense layers only in front of the expert layers".into());
    }
    out["first_k_dense_replace"] = (if types.is_empty() { 0 } else { dense as u64 }).into();
    out["moe_layer_freq"] = 1.into();
    if out.get("rope_theta").is_none() {
        if let Some(t) = cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")) {
            out["rope_theta"] = t.clone();
        }
    }
    out["scoring_func"] = cfg.get("scoring_func").cloned().unwrap_or("sigmoid".into());
    out["topk_method"] = cfg.get("topk_method").cloned().unwrap_or("noaux_tc".into());
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    let arch = plan.arch.clone();
    plan.meta.push((format!("{arch}.attention.q_gate"), Meta::Bool(true)));
    plan.meta.push((format!("{arch}.gated_norm_rank"), Meta::U32(num(cfg, "gated_norm_rank").ok_or("config.json requires gated_norm_rank")? as u32)));
    Ok(plan)
}

/// GGUF name of the per-layer tensors A.X-K2 adds to (or renames in) the DeepSeek set.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let name = match suf {
        "self_attn.q_gate_proj.weight" | "self_attn.q_b_proj.weight" => "attn_q_b.weight",
        "input_layernorm.norm.weight" => "attn_norm.weight",
        "input_layernorm.mlp.fc1.weight" | "input_layernorm.W_down.weight" => "attn_norm_gate_fc1.weight",
        "input_layernorm.mlp.fc2.weight" | "input_layernorm.W_up.weight" => "attn_norm_gate_fc2.weight",
        "post_attention_layernorm.norm.weight" => "ffn_norm.weight",
        "post_attention_layernorm.mlp.fc1.weight" | "post_attention_layernorm.W_down.weight" => "ffn_norm_gate_fc1.weight",
        "post_attention_layernorm.mlp.fc2.weight" | "post_attention_layernorm.W_up.weight" => "ffn_norm_gate_fc2.weight",
        _ => return None,
    };
    Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 })
}

/// Row order of `attn_q_b.weight`: the checkpoint stores `[query; gate]` per head, the GGUF all queries, then all gates.
pub(crate) fn q_gate_order(cfg: &Value) -> Option<Vec<usize>> {
    let (heads, nope, rope, v) = (num(cfg, "num_attention_heads")? as usize, num(cfg, "qk_nope_head_dim")? as usize, num(cfg, "qk_rope_head_dim")? as usize, num(cfg, "v_head_dim")? as usize);
    let qk = nope + rope;
    let mut order = vec![0usize; heads * (qk + v)];
    for h in 0..heads {
        for j in 0..qk {
            order[h * qk + j] = h * (qk + v) + j;
        }
        for j in 0..v {
            order[heads * qk + h * v + j] = h * (qk + v) + qk + j;
        }
    }
    Some(order)
}
