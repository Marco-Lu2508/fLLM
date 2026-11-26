//! DiffLlama (`DiffLlamaForCausalLM`): Llama with differential attention, `(softmax(Q1 K1) - lambda softmax(Q2 K2)) V`. The checkpoint pairs head `h` with head
//! `h + heads / 2` and gives every layer four vectors `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2` (`rust_decoder` combines them with `lambda_init` of the layer
//! index). The plan is the Llama plan renamed to the GGUF architecture `diffllama` with a value width of twice the head width (the two value halves of a key head).
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::Value;

/// The Llama configuration of a DiffLlama checkpoint.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    out["model_type"] = "llama".into();
    out["architectures"] = serde_json::json!(["LlamaForCausalLM"]);
    let kv = out.get("num_key_value_heads").and_then(Value::as_u64).or_else(|| out.get("num_attention_heads").and_then(Value::as_u64)).unwrap_or(0);
    let heads = out.get("num_attention_heads").and_then(Value::as_u64).unwrap_or(0);
    if kv < 2 || kv % 2 != 0 || heads % 2 != 0 {
        return Err("DiffLlama needs an even number of key-value heads (at least 2) and of attention heads".into());
    }
    out["num_key_value_heads"] = kv.into();
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let hidden = normalized.get("hidden_size").and_then(Value::as_u64).ok_or("config.json requires hidden_size")?;
    let heads = normalized.get("num_attention_heads").and_then(Value::as_u64).ok_or("config.json requires num_attention_heads")?;
    let head_dim = normalized.get("head_dim").and_then(Value::as_u64).unwrap_or(hidden / heads.max(1));
    let mut plan = Plan::from_config(&normalized)?;
    for (key, value) in &mut plan.meta {
        *key = key.replace("llama.", "diffllama.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("diffllama".into());
        }
    }
    plan.arch = "diffllama".into();
    plan.meta.retain(|(k, _)| k != "diffllama.attention.value_length");
    plan.meta.push(("diffllama.attention.value_length".into(), Meta::U32((2 * head_dim) as u32)));
    Ok(plan)
}

/// GGUF name of the per-layer tensors DiffLlama adds to the Llama set.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let name = match suf {
        "self_attn.lambda_q1" => "attn_lambda_q1.weight",
        "self_attn.lambda_k1" => "attn_lambda_k1.weight",
        "self_attn.lambda_q2" => "attn_lambda_q2.weight",
        "self_attn.lambda_k2" => "attn_lambda_k2.weight",
        _ => return None,
    };
    Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 })
}
