//! Qwen3.8-Flash-Next (`qwen4_exp`, text part of `Qwen4ExpForConditionalGeneration`). The layers are those of Qwen3.5 MoE (gated delta
//! layers and gated softmax attention with grouped queries, a softmax-routed MoE with a gated shared expert), so the plan is the Qwen3.5
//! plan of a normalised configuration with the GGUF architecture `qwen3next`; what is new is carried by extra metadata and tensors:
//!
//! * several residual streams per token (`hc.count`) with learned gating instead of input norms: `blk.N.hc_{attn,ffn}_{norm,down,up,inject}`
//!   and the collapse before the output norm `output_hc_{norm,down,up}`;
//! * per-layer hashed n-gram embeddings (PLE) on some delta layers: `blk.N.ple_{embd,key,value,norm_key,norm_query,norm_conv,conv}`;
//! * the token indexer (QSA) of the attention layers: `blk.N.indexer.{qk,q_norm,k_norm}.weight` and the metadata `indexer.*`.
//!
//! The `1 + weight` of the RMS norms is folded in. The hash multipliers and the table layout of the n-gram embedding are recomputed by the
//! engine from `ple.*`; the buffers of the checkpoint that hold them are skipped.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::Value;

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

/// Is this the configuration of a Qwen3.8-Flash-Next text model?
pub(crate) fn is_qwen4(cfg: &Value) -> bool {
    // the normalised configuration (`config`) has the Qwen3.5 model type but keeps the stream keys
    cfg.get("model_type").and_then(Value::as_str).is_some_and(|t| t.starts_with("qwen4_exp")) || cfg.get("hc_count").is_some_and(|v| !v.is_null())
}

fn need_hidden(cfg: &Value) -> Result<usize, String> {
    num(cfg, "hidden_size").map(|n| n as usize).ok_or_else(|| "config.json requires hidden_size".to_owned())
}

fn layer_count(cfg: &Value) -> Result<usize, String> {
    num(cfg, "num_hidden_layers").map(|n| n as usize).ok_or_else(|| "config.json requires num_hidden_layers".to_owned())
}

/// Per layer: a gated delta layer (`true`) or an attention layer with the token indexer (`false`).
pub(crate) fn delta_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layer_count(cfg)?;
    Ok(match cfg.get("layer_types").and_then(Value::as_array) {
        Some(types) if types.len() >= n => types.iter().take(n).map(|t| t.as_str() == Some("linear_attention")).collect(),
        _ => {
            let interval = num(cfg, "full_attention_interval").unwrap_or(4).max(1) as usize;
            (0..n).map(|i| (i + 1) % interval != 0).collect()
        }
    })
}

/// One-indexed layer ids with an n-gram embedding, as per-layer flags.
pub(crate) fn ple_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let n = layer_count(cfg)?;
    let ids: Vec<u64> = cfg.get("ple_layer_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    if let Some(bad) = ids.iter().find(|&&id| id < 1 || id as usize > n) {
        return Err(format!("ple_layer_ids holds {bad}, outside 1..={n}"));
    }
    Ok((0..n).map(|i| ids.contains(&(i as u64 + 1))).collect())
}

/// The configuration of a Qwen3.5 MoE text model with the same layers.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    out["model_type"] = "qwen3_5_moe_text".into();
    let types: Vec<Value> = delta_layers(cfg)?.into_iter().map(|d| if d { "linear_attention" } else { "full_attention" }.into()).collect();
    out["layer_types"] = Value::Array(types);
    // every layer is a MoE layer: the dense width only has to be valid
    if num(cfg, "intermediate_size").is_none() {
        out["intermediate_size"] = num(cfg, "shared_expert_intermediate_size").or_else(|| num(cfg, "moe_intermediate_size")).unwrap_or(1).into();
    }
    if let Some(kind) = cfg.get("output_gate_type").and_then(Value::as_str) {
        if kind != "silu" {
            return Err(format!("the output gate activation {kind:?} of the delta layers is unsupported (silu only)"));
        }
    }
    if let Some(act) = cfg.get("hidden_act").and_then(Value::as_str) {
        if act != "silu" {
            return Err(format!("unsupported activation {act:?}; qwen4_exp uses silu"));
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    let arch = plan.arch.clone();
    // the model has no output norm; the file still carries one (all ones), which the engine does not apply
    plan.synthesized.push(("output_norm.weight".into(), vec![1.0; need_hidden(cfg)?]));
    let need = |key: &str| num(cfg, key).filter(|&v| v > 0).ok_or_else(|| format!("config.json requires {key}"));
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let hc = num(cfg, "hc_count").unwrap_or(4);
    if hc < 2 {
        return Err("qwen4_exp needs hc_count of at least 2".into());
    }
    plan.meta.push(u("hc.count", hc));
    plan.meta.push(u("hc.lowrank", num(cfg, "hc_lowrank").unwrap_or(320)));
    let fields = ["indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"];
    if fields.iter().any(|f| cfg.get(*f).is_some_and(|v| !v.is_null())) {
        if num(cfg, "indexer_kv_heads") != Some(1) {
            return Err("qwen4_exp token indexer needs indexer_kv_heads = 1".into());
        }
        let (budget, ratio) = (need("indexer_budget")?, need("indexer_compress_ratio")?);
        if !budget.is_multiple_of(ratio) {
            return Err("indexer_budget must be divisible by indexer_compress_ratio".into());
        }
        plan.meta.extend([u("indexer.heads", need("indexer_n_heads")?), u("indexer.head_dim", need("indexer_head_dim")?), u("indexer.budget", budget), u("indexer.compress_ratio", ratio)]);
    }
    let flags = ple_layers(cfg)?;
    if flags.iter().any(|&f| f) {
        let hidden = need("hidden_size")?;
        let eos = match cfg.get("eos_token_id") {
            Some(Value::Array(list)) => list.first().and_then(Value::as_u64),
            Some(v) => v.as_u64(),
            None => None,
        }
        .ok_or("eos_token_id must be set when PLE layers are enabled")?;
        let (ngram, heads) = (num(cfg, "ngram_size").unwrap_or(3), num(cfg, "heads_per_ngram").unwrap_or(8));
        let embed = num(cfg, "ple_embed_dim").unwrap_or(hidden);
        if ngram < 2 || heads == 0 || !embed.is_multiple_of((ngram - 1) * heads) {
            return Err("ple_embed_dim must be divisible by the number of n-gram heads".into());
        }
        let delta = delta_layers(cfg)?;
        if flags.iter().zip(&delta).any(|(&p, &d)| p && !d) {
            return Err("PLE is only supported on delta layers".into());
        }
        plan.meta.extend([
            (format!("{arch}.ple.layer_flags"), Meta::Bools(flags)),
            u("ple.embed_dim", embed),
            u("ple.conv_kernel", num(cfg, "ple_conv_kernel_size").unwrap_or(4)),
            u("ple.ngram_size", ngram),
            u("ple.heads_per_ngram", heads),
            u("ple.vocab_base", num(cfg, "ngram_vocab_size_base").unwrap_or(20_000_000)),
            u("ple.divisible", num(cfg, "make_ngram_vocab_size_divisible_by").unwrap_or(128)),
            u("ple.seed", num(cfg, "seed").unwrap_or(1234)),
            u("ple.eos", eos),
            u("ple.vocab", need("vocab_size")?),
        ]);
    }
    Ok(plan)
}

/// Does this plan belong to a Qwen3.8-Flash-Next model (the extra metadata is there)?
pub(crate) fn is_plan(plan: &Plan) -> bool {
    plan.meta.iter().any(|(k, _)| k.ends_with(".hc.count"))
}

/// The tensors that are new in this model; `None` for the Qwen3.5 ones.
pub(crate) fn map(n: &str) -> Option<Mapped> {
    let plain = |name: String| Some(Mapped::Tensor { name, add: 0.0 });
    let norm = |name: String| Some(Mapped::Tensor { name, add: 1.0 });
    match n {
        "model.hyper_connection_mixer.hc_norm.weight" => return norm("output_hc_norm.weight".into()),
        "model.hyper_connection_mixer.input_mix_weight_down.weight" => return plain("output_hc_down.weight".into()),
        "model.hyper_connection_mixer.input_mix_weight_up.weight" => return plain("output_hc_up.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.")?;
    let (layer, suf) = rest.split_once('.')?;
    layer.parse::<u64>().ok()?;
    let at = |name: &str| format!("blk.{layer}.{name}");
    if let Some((which, part)) = suf.split_once("_hyper_connection.") {
        let which = match which {
            "attn" => "attn",
            "mlp" => "ffn",
            _ => return None,
        };
        return match part {
            "hc_norm.weight" => norm(at(&format!("hc_{which}_norm.weight"))),
            "input_mix_weight_down.weight" => plain(at(&format!("hc_{which}_down.weight"))),
            "input_mix_weight_up.weight" => plain(at(&format!("hc_{which}_up.weight"))),
            "block_inject_weight.weight" => plain(at(&format!("hc_{which}_inject.weight"))),
            _ => None,
        };
    }
    match suf {
        "self_attn.indexer.index_qk_proj.weight" => plain(at("indexer.qk.weight")),
        "self_attn.indexer.q_layernorm.weight" => norm(at("indexer.q_norm.weight")),
        "self_attn.indexer.k_layernorm.weight" => norm(at("indexer.k_norm.weight")),
        "ple.key_proj.weight" => plain(at("ple_key.weight")),
        "ple.value_proj.weight" => plain(at("ple_value.weight")),
        "ple.norm_key.weight" => norm(at("ple_norm_key.weight")),
        "ple.norm_query.weight" => norm(at("ple_norm_query.weight")),
        "ple.norm_conv.weight" => norm(at("ple_norm_conv.weight")),
        "ple.conv1d.weight" => plain(at("ple_conv.weight")),
        "ple.ple_embedding.ngram_embedding.weight" => plain(at("ple_embd.weight")),
        // buffers of the hash tables: recomputed from the metadata
        "ple.ple_embedding.layer_multipliers" | "ple.ple_embedding.ngram_heads_vocab_sizes" | "ple.ple_embedding.ngram_heads_offsets" => Some(Mapped::Skip),
        _ => {
            // the table is stored in `split_ngram_parts` shards along the rows
            let shard = suf.strip_prefix("ple.ple_embedding.ngram_embedding.shard_")?.strip_suffix(".weight")?.parse::<usize>().ok()?;
            Some(Mapped::Expert { name: at("ple_embd.weight"), index: shard })
        }
    }
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let delta = *delta_layers(cfg).ok()?.get(i as usize)?;
    let mut need = vec!["hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight", "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"];
    if delta {
        need.extend(["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight"]);
    } else {
        need.extend(["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight"]);
        if cfg.get("indexer_n_heads").is_some_and(|v| !v.is_null()) {
            need.extend(["indexer.qk.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"]);
        }
    }
    if *ple_layers(cfg).ok()?.get(i as usize)? {
        need.extend(["ple_embd.weight", "ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv.weight"]);
    }
    need.extend(["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]);
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny() -> Value {
        json!({
            "model_type": "qwen4_exp_text", "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "eos_token_id": 2, "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16, "num_experts": 4, "num_experts_per_tok": 2,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "indexed_attention"], "ple_layer_ids": [2], "ple_embed_dim": 32,
            "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 100, "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
            "indexer_budget": 8, "indexer_compress_ratio": 4, "hc_count": 2, "hc_lowrank": 8,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]}, "max_position_embeddings": 256
        })
    }

    #[test]
    fn plan_is_a_qwen35_plan_with_stream_embedding_and_indexer_metadata() {
        let plan = plan(&tiny()).unwrap();
        assert_eq!(plan.arch, "qwen3next");
        assert!(is_plan(&plan));
        let get = |k: &str| plan.meta.iter().find(|(n, _)| n == &format!("qwen3next.{k}")).map(|(_, v)| v.clone());
        assert!(matches!(get("hc.count"), Some(Meta::U32(2))));
        assert!(matches!(get("indexer.compress_ratio"), Some(Meta::U32(4))));
        assert!(matches!(get("ple.layer_flags"), Some(Meta::Bools(f)) if f == [false, true, false, false]));
        assert!(matches!(get("ple.eos"), Some(Meta::U32(2))));
    }

    #[test]
    fn new_tensors_map_and_norms_fold_the_offset() {
        let norm = |m: Option<Mapped>| match m {
            Some(Mapped::Tensor { name, add }) => (name, add),
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(norm(map("model.layers.2.attn_hyper_connection.hc_norm.weight")), ("blk.2.hc_attn_norm.weight".into(), 1.0));
        assert_eq!(norm(map("model.layers.2.mlp_hyper_connection.block_inject_weight.weight")), ("blk.2.hc_ffn_inject.weight".into(), 0.0));
        assert_eq!(norm(map("model.hyper_connection_mixer.input_mix_weight_up.weight")), ("output_hc_up.weight".into(), 0.0));
        assert_eq!(norm(map("model.layers.1.self_attn.indexer.k_layernorm.weight")), ("blk.1.indexer.k_norm.weight".into(), 1.0));
        assert_eq!(norm(map("model.layers.1.ple.conv1d.weight")), ("blk.1.ple_conv.weight".into(), 0.0));
        assert!(matches!(map("model.layers.1.ple.ple_embedding.layer_multipliers"), Some(Mapped::Skip)));
        assert!(matches!(map("model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"), Some(Mapped::Expert { index: 3, .. })));
        assert!(map("model.layers.1.mlp.gate.weight").is_none());
    }

    #[test]
    fn requirements_follow_the_layer_kind() {
        let c = tiny();
        let delta = requirements(&c, 1).unwrap();
        assert!(delta.contains(&"blk.1.ssm_conv1d.weight".to_owned()) && delta.contains(&"blk.1.ple_embd.weight".to_owned()));
        let attn = requirements(&c, 3).unwrap();
        assert!(attn.contains(&"blk.3.indexer.qk.weight".to_owned()) && !attn.iter().any(|n| n.contains("ple_")));
    }

    #[test]
    fn unsupported_output_gate_is_refused() {
        let mut c = tiny();
        c["output_gate_type"] = "sigmoid".into();
        assert!(config(&c).is_err());
    }
}
