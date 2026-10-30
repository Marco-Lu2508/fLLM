//! MiniMax-M3 text model (`MiniMaxM3VLForCausalLM` of MiniMaxAI/MiniMax-M3, its vision tower is not imported): grouped-query attention
//! with Gemma-style (`1 + weight`) per-head query and key norms and partial half-split rotary pairs, block-sparse attention on some
//! layers (a small indexer scores the keys, pools the scores per block of `index_block_size` keys and keeps the best
//! `index_topk_blocks` blocks plus the local ones for every KV head), dense or sigmoid-routed MoE feed-forwards whose experts (and
//! the shared expert) use the gpt-oss style SwiGLU `(up + 1) * gate * sigmoid(alpha * gate)` with clamping.
//!
//! GGUF architecture `minimax3`; tensors of `blk.N`: `attn_norm`, `ffn_norm` (the `1 + weight` is folded in), `attn_query`, `attn_key`,
//! `attn_value`, `attn_out`, `attn_qn`, `attn_kn`; on sparse layers `indexer.attn_q`, `indexer.attn_k`, `indexer.q_norm`, `indexer.k_norm`;
//! dense feed-forward `ffn_gate`, `ffn_up`, `ffn_down`; MoE `ffn_router`, `ffn_router_bias`, `ffn_{gate,up,down}_exps`,
//! `ffn_{gate,up}_shexp`, `ffn_shared_down`. The names differ from the usual ones on purpose: the model has its own layer loader.
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

/// Per layer: block-sparse attention.
pub(crate) fn sparse_attention(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layers(cfg)?;
    Ok(match cfg.get("layer_types").and_then(Value::as_array) {
        Some(types) => types.iter().take(n).map(|t| t.as_str() == Some("minimax_m3_sparse")).collect(),
        None => vec![false; n],
    })
}

/// Per layer: MoE feed-forward.
pub(crate) fn sparse_mlp(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layers(cfg)?;
    Ok(match cfg.get("mlp_layer_types").and_then(Value::as_array) {
        Some(types) => types.iter().take(n).map(|t| t.as_str() == Some("sparse")).collect(),
        None => vec![true; n],
    })
}

/// Rotated values per head (the whole head unless the rotary parameters say otherwise).
fn rotary_dims(cfg: &Value) -> u64 {
    let head_dim = num(cfg, "head_dim").unwrap_or(128);
    let factor = cfg.get("rope_parameters").and_then(|r| flt(r, "partial_rotary_factor")).or_else(|| flt(cfg, "partial_rotary_factor")).unwrap_or(1.0);
    (head_dim as f64 * factor) as u64
}

/// The configuration for the shape checks of the conversion.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    for key in ["num_key_value_heads", "num_attention_heads"] {
        if out.get(key).is_none() {
            out[key] = 1.into();
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let n = layers(cfg)?;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let llama = json!({
        "model_type": "llama",
        "hidden_size": num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?,
        "num_attention_heads": heads,
        "num_key_value_heads": num(cfg, "num_key_value_heads").unwrap_or(heads),
        "head_dim": num(cfg, "head_dim").unwrap_or(128),
        "intermediate_size": num(cfg, "dense_intermediate_size").unwrap_or(1),
        "num_hidden_layers": n,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": flt(cfg, "rms_norm_eps").unwrap_or(1e-6),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(524288),
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
    });
    let mut plan = Plan::from_config(&llama)?;
    plan.norm_add = 1.0;
    let (family_arch, arch) = ("llama", "minimax3");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let theta = cfg.get("rope_parameters").and_then(|r| flt(r, "rope_theta")).or_else(|| flt(cfg, "rope_theta")).unwrap_or(10000.0);
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    plan.meta.retain(|(k, _)| !k.ends_with(".rope.layer_flags") && !k.ends_with(".rope.freq_base") && !k.ends_with(".rope.dimension_count"));
    plan.meta.extend([
        u("rope.dimension_count", rotary_dims(cfg)),
        f("rope.freq_base", theta),
        u("index.head_count", num(cfg, "index_n_heads").unwrap_or(4)),
        u("index.head_dim", num(cfg, "index_head_dim").unwrap_or(128)),
        u("index.block_size", num(cfg, "index_block_size").unwrap_or(128)),
        u("index.topk_blocks", num(cfg, "index_topk_blocks").unwrap_or(16)),
        u("index.local_blocks", num(cfg, "index_local_blocks").unwrap_or(1)),
        f("swiglu_alpha", flt(cfg, "swiglu_alpha").unwrap_or(1.702)),
        f("swiglu_limit", flt(cfg, "swiglu_limit").unwrap_or(7.0)),
        u("moe.experts", num(cfg, "num_local_experts").ok_or("config.json requires num_local_experts")?),
        u("moe.used", num(cfg, "num_experts_per_tok").unwrap_or(4)),
        u("moe.feed_forward_length", num(cfg, "intermediate_size").unwrap_or(1)),
        u("moe.shared_feed_forward_length", num(cfg, "shared_intermediate_size").unwrap_or(1)),
        f("moe.route_scale", flt(cfg, "routed_scaling_factor").unwrap_or(1.0)),
        (format!("{arch}.layer_sparse_attention"), Meta::Bools(sparse_attention(cfg)?)),
        (format!("{arch}.layer_sparse_mlp"), Meta::Bools(sparse_mlp(cfg)?)),
    ]);
    plan.moe = false;
    Ok(plan)
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
    let norm = |name: String| Ok(Mapped::Tensor { name, add: plan.norm_add });
    let plain = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
    match n {
        "model.embed_tokens.weight" => return plain("token_embd.weight".into()),
        "model.norm.weight" => return norm("output_norm.weight".into()),
        "lm_head.weight" => return plain("output.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.").ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    let (layer, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: u64 = layer.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let fused = |part: &str| Ok(Mapped::Fused { layer, part: part.to_owned() });
    let at = |name: &str| format!("blk.{layer}.{name}");
    match suf {
        "input_layernorm.weight" => norm(at("attn_norm.weight")),
        "post_attention_layernorm.weight" => norm(at("ffn_norm.weight")),
        "self_attn.q_proj.weight" => plain(at("attn_query.weight")),
        "self_attn.k_proj.weight" => plain(at("attn_key.weight")),
        "self_attn.v_proj.weight" => plain(at("attn_value.weight")),
        "self_attn.o_proj.weight" => plain(at("attn_out.weight")),
        "self_attn.q_norm.weight" => norm(at("attn_qn.weight")),
        "self_attn.k_norm.weight" => norm(at("attn_kn.weight")),
        "self_attn.indexer.q_proj.weight" => plain(at("indexer.attn_q.weight")),
        "self_attn.indexer.k_proj.weight" => plain(at("indexer.attn_k.weight")),
        "self_attn.indexer.q_norm.weight" => norm(at("indexer.q_norm.weight")),
        "self_attn.indexer.k_norm.weight" => norm(at("indexer.k_norm.weight")),
        "mlp.gate_up_proj.weight" => fused("dense_input"),
        "mlp.down_proj.weight" => plain(at("ffn_down.weight")),
        "mlp.gate.weight" => plain(at("ffn_router.weight")),
        "mlp.gate.e_score_correction_bias" => plain(at("ffn_router_bias.bias")),
        "mlp.shared_experts.gate_up_proj.weight" => fused("shared_input"),
        "mlp.shared_experts.down_proj.weight" => plain(at("ffn_shared_down.weight")),
        "mlp.experts.gate_up_proj" => fused("experts_input"),
        "mlp.experts.down_proj" => fused("experts_output"),
        _ => {
            if suf.starts_with("mtp.") || n.contains(".mtp.") {
                return Ok(Mapped::Skip);
            }
            Err(format!("unrecognized MiniMax-M3 tensor {n:?}"))
        }
    }
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let sparse_attn = *sparse_attention(cfg).ok()?.get(i as usize)?;
    let sparse = *sparse_mlp(cfg).ok()?.get(i as usize)?;
    let mut need = vec!["attn_norm.weight", "ffn_norm.weight", "attn_query.weight", "attn_key.weight", "attn_value.weight", "attn_out.weight", "attn_qn.weight", "attn_kn.weight"];
    if sparse_attn {
        need.extend(["indexer.attn_q.weight", "indexer.attn_k.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"]);
    }
    if sparse {
        need.extend(["ffn_router.weight", "ffn_router_bias.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_shared_down.weight"]);
    } else {
        need.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
    }
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norms_fold_the_gemma_offset_and_fused_projections_split() {
        let cfg = json!({"model_type": "minimax_m3_vl_text", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "dense_intermediate_size": 64, "intermediate_size": 16, "shared_intermediate_size": 16, "num_local_experts": 4, "num_experts_per_tok": 2,
            "layer_types": ["full_attention", "minimax_m3_sparse"], "mlp_layer_types": ["dense", "sparse"], "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5}});
        let plan = plan(&cfg).unwrap();
        assert_eq!(plan.arch, "minimax3");
        assert_eq!(plan.map("model.layers.1.self_attn.q_norm.weight").unwrap(), Mapped::Tensor { name: "blk.1.attn_qn.weight".into(), add: 1.0 });
        assert_eq!(plan.map("model.norm.weight").unwrap(), Mapped::Tensor { name: "output_norm.weight".into(), add: 1.0 });
        assert_eq!(plan.map("model.layers.0.mlp.gate_up_proj.weight").unwrap(), Mapped::Fused { layer: 0, part: "dense_input".into() });
        assert_eq!(plan.map("model.layers.1.mlp.shared_experts.gate_up_proj.weight").unwrap(), Mapped::Fused { layer: 1, part: "shared_input".into() });
        let meta = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        assert!(matches!(meta("minimax3.rope.dimension_count"), Some(Meta::U32(4))));
        assert!(matches!(meta("minimax3.layer_sparse_attention"), Some(Meta::Bools(b)) if b == [false, true]));
        assert_eq!(requirements(&cfg, 0).unwrap().len(), 11);
        assert_eq!(requirements(&cfg, 1).unwrap().len(), 20);
    }
}
