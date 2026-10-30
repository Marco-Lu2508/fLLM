//! DeepSeek-V4 (`DeepseekV4ForCausalLM`, deepseek-ai/DeepSeek-V4-*): manifold-constrained hyper-connections (several residual
//! streams per token), shared-KV multi-query attention with a sliding window, compressed (CSA) and heavily compressed (HCA) long-range
//! entries, grouped low-rank output projection, hash-routed and top-k MoE layers with a clamped SwiGLU.
//!
//! The plan is a Llama plan of a normalised configuration, renamed to the GGUF architecture `deepseek4`, plus the V4 metadata and tensors.
//! GGUF tensors of layer `blk.N`: `attn_norm`, `ffn_norm`, `hc_attn_{fn,base,scale}`, `hc_ffn_{fn,base,scale}`, `attn_q_a`, `attn_q_a_norm`,
//! `attn_q_b`, `attn_kv`, `attn_kv_norm`, `attn_output_a`, `attn_output_b`, `attn_sinks`; on compressing layers `attn_comp_{kv,gate,pos,norm}` and
//! (compressed sparse layers) `indexer.{comp_kv,comp_gate,comp_pos,comp_norm,attn_q_b,proj}`; the feed-forward `ffn_gate_inp`, `exp_probs_b`
//! or `ffn_tid2eid` (hash-routed layers, token id to experts, stored as f32), `ffn_{gate,up,down}_exps` and `ffn_{gate,up,down}_shexp`.
//! Globally: `output_hc_{fn,base,scale}` and the synthesised rotary frequency tables `rope_freqs_main.weight` / `rope_freqs_compress.weight`
//! (the inverse frequencies, with the YaRN correction of the compressing layers applied).
//! Both the Hugging Face layout and the Hub layout (`layers.N.attn.wq_a`, `hc_attn_fn`, per-expert `w1`/`w2`/`w3`, `embed.weight`) are mapped
//! (the renames follow the transformers conversion mapping for `deepseek_v4`). Hub checkpoints store the experts in FP4 and the dense weights in
//! FP8 with `.scale` companions; those formats are not read (convert from a bf16 export).
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

/// Compression rate per layer: 0 (window only), the rate of compressed sparse attention or of heavily compressed attention.
pub(crate) fn compress_ratios(cfg: &Value) -> Result<Vec<u64>, String> {
    Ok(layer_kinds(cfg)?.into_iter().map(|(ratio, _)| ratio).collect())
}

/// Per layer: the compression rate and whether the layer is a compressed sparse one (two overlapping series and an indexer).
pub(crate) fn layer_kinds(cfg: &Value) -> Result<Vec<(u64, bool)>, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let rates = cfg.get("compress_rates");
    let rate = |name: &str, legacy: &str, default: u64| rates.and_then(|r| r.get(name)).and_then(Value::as_u64).or_else(|| num(cfg, legacy)).unwrap_or(default);
    let (csa, hca) = (rate("compressed_sparse_attention", "compress_rate_csa", 4), rate("heavily_compressed_attention", "compress_rate_hca", 128));
    if let Some(list) = cfg.get("compress_ratios").and_then(Value::as_array) {
        // legacy per-layer integers
        return Ok(list.iter().take(layers).map(|v| v.as_u64().unwrap_or(0)).map(|r| (r, r == 4)).collect());
    }
    let types: Vec<String> = match cfg.get("layer_types").and_then(Value::as_array) {
        Some(list) => list.iter().map(|v| v.as_str().unwrap_or_default().to_owned()).collect(),
        None => {
            let mut v = vec!["heavily_compressed_attention".to_owned(); layers.min(2)];
            v.extend((0..layers.saturating_sub(2)).map(|i| if i % 2 == 1 { "compressed_sparse_attention" } else { "heavily_compressed_attention" }.to_owned()));
            v
        }
    };
    if types.len() < layers {
        return Err("layer_types must have one entry per layer".into());
    }
    types
        .iter()
        .take(layers)
        .map(|t| match t.as_str() {
            "sliding_attention" => Ok((0, false)),
            "compressed_sparse_attention" => Ok((csa, true)),
            "heavily_compressed_attention" => Ok((hca, false)),
            other => Err(format!("unsupported DeepSeek-V4 layer type {other:?}")),
        })
        .collect()
}

/// Per layer: hash-routed feed-forward.
pub(crate) fn hash_layers(cfg: &Value) -> Result<Vec<bool>, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    if let Some(list) = cfg.get("mlp_layer_types").and_then(Value::as_array) {
        return Ok(list.iter().take(layers).map(|v| v.as_str() == Some("hash_moe")).collect());
    }
    let hashed = num(cfg, "num_hash_layers").unwrap_or(3) as usize;
    Ok((0..layers).map(|i| i < hashed).collect())
}

fn rope_dims(cfg: &Value) -> Result<u64, String> {
    let head_dim = num(cfg, "head_dim").unwrap_or(512);
    if let Some(d) = num(cfg, "qk_rope_head_dim") {
        return Ok(d);
    }
    let factor = flt(cfg, "partial_rotary_factor").unwrap_or(64.0 / 512.0);
    Ok((head_dim as f64 * factor) as u64)
}

/// The configuration for the shape checks of the conversion: the original one with the keys of a plain decoder filled in.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let mut out = cfg.clone();
    let llama = llama_config(cfg)?;
    for key in ["num_key_value_heads", "intermediate_size"] {
        out[key] = llama[key].clone();
    }
    Ok(out)
}

/// The configuration as the Llama plan expects it.
fn llama_config(cfg: &Value) -> Result<Value, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?;
    let mut out = json!({
        "model_type": "llama",
        "hidden_size": num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?,
        "num_attention_heads": num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?,
        "num_key_value_heads": 1,
        "head_dim": num(cfg, "head_dim").unwrap_or(512),
        "intermediate_size": num(cfg, "moe_intermediate_size").ok_or("config.json requires moe_intermediate_size")?,
        "num_hidden_layers": layers,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": flt(cfg, "rms_norm_eps").unwrap_or(1e-6),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(1_048_576),
        "rope_theta": flt(cfg, "rope_theta").unwrap_or(10000.0),
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
    });
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
        if let Some(v) = cfg.get(key) {
            out[key] = v.clone();
        }
    }
    if num(cfg, "n_shared_experts").unwrap_or(1) != 1 {
        return Err("DeepSeek-V4 with more than one shared expert is unsupported".into());
    }
    Ok(out)
}

/// Inverse rotary frequencies (`dim / 2` of them) of the rope parameters `params` (transformers' default and YaRN initialisation;
/// the YaRN attention factor is not applied to cos / sin by the V4 reference).
fn inv_freq(params: &Value, theta: f64, dim: u64, max_positions: u64) -> Vec<f32> {
    let half = (dim / 2) as usize;
    let pos_freqs: Vec<f64> = (0..half).map(|i| theta.powf(2.0 * i as f64 / dim as f64)).collect();
    if params.get("rope_type").and_then(Value::as_str) != Some("yarn") {
        return pos_freqs.iter().map(|p| (1.0 / p) as f32).collect();
    }
    let factor = flt(params, "factor").unwrap_or(1.0);
    let original = num(params, "original_max_position_embeddings").unwrap_or(max_positions) as f64;
    let beta_fast = flt(params, "beta_fast").filter(|v| *v != 0.0).unwrap_or(32.0);
    let beta_slow = flt(params, "beta_slow").filter(|v| *v != 0.0).unwrap_or(1.0);
    let truncate = params.get("truncate").and_then(Value::as_bool).unwrap_or(true);
    let correction = |rotations: f64| dim as f64 * (original / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln());
    let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    let (low, mut high) = (low.max(0.0), high.min(dim as f64 - 1.0));
    if low == high {
        high += 0.001;
    }
    pos_freqs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation = 1.0 - ramp;
            ((1.0 / (factor * p)) * (1.0 - extrapolation) + (1.0 / p) * extrapolation) as f32
        })
        .collect()
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = llama_config(cfg)?;
    let ratios = compress_ratios(cfg)?;
    let hashed = hash_layers(cfg)?;
    let layers = ratios.len() as u64;
    let mut plan = Plan::from_config(&normalized)?;
    let (family_arch, arch) = ("llama", "deepseek4");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let rope = rope_dims(cfg)?;
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    let gating = match cfg.get("scoring_func").and_then(Value::as_str).unwrap_or("sqrtsoftplus") {
        "softmax" => 1,
        "sigmoid" => 2,
        "sqrtsoftplus" => 4,
        other => return Err(format!("unsupported DeepSeek-V4 scoring function {other:?}")),
    };
    plan.meta.retain(|(k, _)| !k.ends_with(".rope.layer_flags") && !k.ends_with(".rope.dimension_count"));
    plan.meta.extend([
        u("rope.dimension_count", rope),
        u("attention.q_lora_rank", num(cfg, "q_lora_rank").unwrap_or(1024)),
        u("v4.window", num(cfg, "sliding_window").unwrap_or(128)),
        u("v4.hc_mult", num(cfg, "hc_mult").unwrap_or(4)),
        u("v4.hc_sinkhorn_iters", num(cfg, "hc_sinkhorn_iters").unwrap_or(20)),
        f("v4.hc_eps", flt(cfg, "hc_eps").unwrap_or(1e-6)),
        u("v4.o_groups", num(cfg, "o_groups").unwrap_or(8)),
        u("v4.o_lora_rank", num(cfg, "o_lora_rank").unwrap_or(1024)),
        u("v4.index_head_count", num(cfg, "index_n_heads").unwrap_or(64)),
        u("v4.index_head_dim", num(cfg, "index_head_dim").unwrap_or(128)),
        u("v4.index_top_k", num(cfg, "index_topk").unwrap_or(512)),
        (format!("{arch}.v4.compress_ratio"), Meta::U32s(ratios.iter().map(|&r| r as u32).collect())),
        (format!("{arch}.v4.hash_layers"), Meta::Bools(hashed)),
        u("expert_count", num(cfg, "n_routed_experts").or_else(|| num(cfg, "num_local_experts")).ok_or("config.json requires n_routed_experts")?),
        u("expert_used_count", num(cfg, "num_experts_per_tok").unwrap_or(6)),
        u("expert_feed_forward_length", num(cfg, "moe_intermediate_size").unwrap_or(0)),
        u("expert_shared_count", 1),
        f("expert_weights_scale", flt(cfg, "routed_scaling_factor").unwrap_or(1.5)),
        (format!("{arch}.expert_weights_norm"), Meta::Bool(true)),
        u("expert_gating_func", gating),
        f("expert_swiglu_clamp", flt(cfg, "swiglu_limit").unwrap_or(10.0)),
    ]);
    // rotary tables: plain rotary for the window-only layers, the (YaRN-scaled) compress rotary for the others
    let max_positions = num(cfg, "max_position_embeddings").unwrap_or(1_048_576);
    let rp = cfg.get("rope_parameters").cloned().unwrap_or(Value::Null);
    let nested = rp.get("main").is_some_and(Value::is_object) && rp.get("compress").is_some_and(Value::is_object);
    let compress_params = if nested {
        rp["compress"].clone()
    } else {
        rp.as_object().map(|m| Value::Object(m.iter().filter(|(k, _)| k.as_str() != "main" && k.as_str() != "compress").map(|(k, v)| (k.clone(), v.clone())).collect())).unwrap_or(Value::Null)
    };
    let main_theta = if nested { flt(&rp["main"], "rope_theta") } else { None }.or_else(|| flt(cfg, "rope_theta")).unwrap_or(10000.0);
    let compress_theta = if nested { flt(&rp["compress"], "rope_theta") } else { None }.or_else(|| flt(cfg, "compress_rope_theta")).unwrap_or(160000.0);
    plan.synthesized.push(("rope_freqs_main.weight".into(), inv_freq(&json!({}), main_theta, rope, max_positions)));
    plan.synthesized.push(("rope_freqs_compress.weight".into(), inv_freq(&compress_params, compress_theta, rope, max_positions)));
    plan.moe = true;
    plan.moe_layers = vec![true; layers as usize];
    plan.layers = layers;
    Ok(plan)
}

/// Rewrites a Hub tensor name to the transformers layout (the conversion mapping of `deepseek_v4`).
fn hub_to_hf(n: &str) -> String {
    let mut s = n.to_owned();
    let swap = |s: &mut String, from: &str, to: &str| {
        if s.contains(from) {
            *s = s.replace(from, to);
        }
    };
    if let Some(rest) = s.strip_prefix("layers.") {
        s = format!("model.layers.{rest}");
    }
    swap(&mut s, ".attn.", ".self_attn.");
    swap(&mut s, ".ffn.", ".mlp.");
    swap(&mut s, ".indexer.compressor.", ".compressor.indexer.");
    for (from, to) in [("embed.weight", "model.embed_tokens.weight"), ("head.weight", "lm_head.weight"), ("hc_head_fn", "model.hc_head.hc_fn"), ("hc_head_base", "model.hc_head.hc_base"), ("hc_head_scale", "model.hc_head.hc_scale")] {
        if s == from {
            s = to.to_owned();
        }
    }
    if s == "norm.weight" {
        s = "model.norm.weight".into();
    }
    swap(&mut s, ".attn_norm.", ".input_layernorm.");
    swap(&mut s, ".ffn_norm.", ".post_attention_layernorm.");
    for (from, to) in [("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"), ("hc_attn_scale", "attn_hc.scale"), ("hc_ffn_fn", "ffn_hc.fn"), ("hc_ffn_base", "ffn_hc.base"), ("hc_ffn_scale", "ffn_hc.scale")] {
        swap(&mut s, &format!(".{from}"), &format!(".{to}"));
    }
    swap(&mut s, ".attn_sink", ".sinks");
    swap(&mut s, ".indexer.weights_proj.", ".compressor.indexer.scorer.weights_proj.");
    swap(&mut s, ".indexer.wq_b.", ".compressor.indexer.q_b_proj.");
    if s.contains(".self_attn.") {
        swap(&mut s, ".norm.", ".kv_norm.");
        swap(&mut s, ".ape", ".position_bias");
        swap(&mut s, ".wq_a.", ".q_a_proj.");
        swap(&mut s, ".self_attn.wq_b.", ".self_attn.q_b_proj.");
        swap(&mut s, ".wkv.", ".kv_proj.");
        swap(&mut s, ".wgate.", ".gate_proj.");
        swap(&mut s, ".wo_a.", ".o_a_proj.");
        swap(&mut s, ".wo_b.", ".o_b_proj.");
        swap(&mut s, ".q_norm.", ".q_a_norm.");
    }
    swap(&mut s, ".gate.bias", ".gate.e_score_correction_bias");
    swap(&mut s, ".shared_experts.w1.", ".shared_experts.gate_proj.");
    swap(&mut s, ".shared_experts.w2.", ".shared_experts.down_proj.");
    swap(&mut s, ".shared_experts.w3.", ".shared_experts.up_proj.");
    s
}

pub(crate) fn map(_plan: &Plan, n: &str) -> Result<Mapped, String> {
    let tensor = |name: &str| Ok(Mapped::Tensor { name: name.to_owned(), add: 0.0 });
    let hf = hub_to_hf(n);
    let n = hf.as_str();
    match n {
        "model.embed_tokens.weight" => return tensor("token_embd.weight"),
        "model.norm.weight" => return tensor("output_norm.weight"),
        "lm_head.weight" => return tensor("output.weight"),
        "model.hc_head.hc_fn" => return tensor("output_hc_fn.weight"),
        "model.hc_head.hc_base" => return tensor("output_hc_base.weight"),
        "model.hc_head.hc_scale" => return tensor("output_hc_scale.weight"),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.").ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    let (i, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: u64 = i.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let name = match suf {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "attn_hc.fn" => "hc_attn_fn.weight",
        "attn_hc.base" => "hc_attn_base.weight",
        "attn_hc.scale" => "hc_attn_scale.weight",
        "ffn_hc.fn" => "hc_ffn_fn.weight",
        "ffn_hc.base" => "hc_ffn_base.weight",
        "ffn_hc.scale" => "hc_ffn_scale.weight",
        "self_attn.q_a_proj.weight" => "attn_q_a.weight",
        "self_attn.q_a_norm.weight" => "attn_q_a_norm.weight",
        "self_attn.q_b_proj.weight" => "attn_q_b.weight",
        "self_attn.kv_proj.weight" => "attn_kv.weight",
        "self_attn.kv_norm.weight" => "attn_kv_norm.weight",
        "self_attn.o_a_proj.weight" => "attn_output_a.weight",
        "self_attn.o_b_proj.weight" => "attn_output_b.weight",
        "self_attn.sinks" => "attn_sinks.weight",
        "self_attn.compressor.kv_proj.weight" => "attn_comp_kv.weight",
        "self_attn.compressor.gate_proj.weight" => "attn_comp_gate.weight",
        "self_attn.compressor.position_bias" => "attn_comp_pos.weight",
        "self_attn.compressor.kv_norm.weight" => "attn_comp_norm.weight",
        "self_attn.compressor.indexer.kv_proj.weight" => "indexer.comp_kv.weight",
        "self_attn.compressor.indexer.gate_proj.weight" => "indexer.comp_gate.weight",
        "self_attn.compressor.indexer.position_bias" => "indexer.comp_pos.weight",
        "self_attn.compressor.indexer.kv_norm.weight" => "indexer.comp_norm.weight",
        "self_attn.compressor.indexer.q_b_proj.weight" => "indexer.attn_q_b.weight",
        "self_attn.compressor.indexer.scorer.weights_proj.weight" => "indexer.proj.weight",
        "mlp.gate.weight" => "ffn_gate_inp.weight",
        "mlp.gate.e_score_correction_bias" => "exp_probs_b.bias",
        "mlp.gate.tid2eid" => "ffn_tid2eid.weight",
        "mlp.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
        "mlp.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
        "mlp.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
        "mlp.experts.gate_up_proj" => return Ok(Mapped::Fused { layer, part: "experts_input".into() }),
        "mlp.experts.down_proj" => return Ok(Mapped::Fused { layer, part: "experts_output".into() }),
        _ => {
            // Hub layout: one tensor per expert, w1 = gate, w3 = up, w2 = down
            if let Some((index, kind)) = suf.strip_prefix("mlp.experts.").and_then(|r| r.split_once('.')) {
                let kind = match kind {
                    "w1.weight" => "gate",
                    "w3.weight" => "up",
                    "w2.weight" => "down",
                    _ => return Err(format!("unrecognized DeepSeek-V4 expert tensor {n:?}")),
                };
                let index = index.parse().map_err(|_| format!("invalid expert index in {n:?}"))?;
                return Ok(Mapped::Expert { name: format!("blk.{layer}.ffn_{kind}_exps.weight"), index });
            }
            if suf.starts_with("mtp.") || n.contains(".mtp.") {
                return Ok(Mapped::Skip);
            }
            return Err(format!("unrecognized DeepSeek-V4 tensor {n:?}"));
        }
    };
    Ok(Mapped::Tensor { name: format!("blk.{layer}.{name}"), add: 0.0 })
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let (ratio, sparse) = *layer_kinds(cfg).ok()?.get(i as usize)?;
    let hashed = *hash_layers(cfg).ok()?.get(i as usize)?;
    let mut need: Vec<&str> = vec![
        "attn_norm.weight", "ffn_norm.weight", "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight", "hc_ffn_fn.weight",
        "hc_ffn_base.weight", "hc_ffn_scale.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv.weight",
        "attn_kv_norm.weight", "attn_output_a.weight", "attn_output_b.weight", "attn_sinks.weight", "ffn_gate_inp.weight",
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_up_shexp.weight",
    ];
    if ratio > 0 {
        need.extend(["attn_comp_kv.weight", "attn_comp_gate.weight", "attn_comp_pos.weight", "attn_comp_norm.weight"]);
    }
    if sparse {
        need.extend(["indexer.comp_kv.weight", "indexer.comp_gate.weight", "indexer.comp_pos.weight", "indexer.comp_norm.weight", "indexer.attn_q_b.weight", "indexer.proj.weight"]);
    }
    need.push(if hashed { "ffn_tid2eid.weight" } else { "exp_probs_b.bias" });
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_names_map_like_the_transformers_layout() {
        for (hub, hf) in [
            ("layers.3.attn.wq_a.weight", "model.layers.3.self_attn.q_a_proj.weight"),
            ("layers.3.attn.q_norm.weight", "model.layers.3.self_attn.q_a_norm.weight"),
            ("layers.3.attn.wo_a.weight", "model.layers.3.self_attn.o_a_proj.weight"),
            ("layers.3.attn.attn_sink", "model.layers.3.self_attn.sinks"),
            ("layers.3.attn.compressor.wkv.weight", "model.layers.3.self_attn.compressor.kv_proj.weight"),
            ("layers.3.attn.compressor.ape", "model.layers.3.self_attn.compressor.position_bias"),
            ("layers.3.attn.compressor.norm.weight", "model.layers.3.self_attn.compressor.kv_norm.weight"),
            ("layers.3.attn.indexer.compressor.wgate.weight", "model.layers.3.self_attn.compressor.indexer.gate_proj.weight"),
            ("layers.3.attn.indexer.wq_b.weight", "model.layers.3.self_attn.compressor.indexer.q_b_proj.weight"),
            ("layers.3.attn.indexer.weights_proj.weight", "model.layers.3.self_attn.compressor.indexer.scorer.weights_proj.weight"),
            ("layers.3.hc_ffn_scale", "model.layers.3.ffn_hc.scale"),
            ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
            ("layers.3.ffn.gate.bias", "model.layers.3.mlp.gate.e_score_correction_bias"),
            ("layers.3.ffn.shared_experts.w2.weight", "model.layers.3.mlp.shared_experts.down_proj.weight"),
            ("embed.weight", "model.embed_tokens.weight"),
            ("head.weight", "lm_head.weight"),
            ("hc_head_fn", "model.hc_head.hc_fn"),
        ] {
            assert_eq!(hub_to_hf(hub), hf, "{hub}");
        }
    }

    #[test]
    fn layer_schedules_follow_the_configuration_defaults() {
        let cfg = json!({"num_hidden_layers": 6});
        assert_eq!(compress_ratios(&cfg).unwrap(), vec![128, 128, 128, 4, 128, 4]);
        assert_eq!(hash_layers(&cfg).unwrap(), vec![true, true, true, false, false, false]);
        let cfg = json!({"num_hidden_layers": 3, "layer_types": ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"], "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 8}});
        assert_eq!(compress_ratios(&cfg).unwrap(), vec![0, 2, 8]);
    }

    #[test]
    fn yarn_keeps_high_frequencies_and_interpolates_low_ones() {
        let plain = inv_freq(&json!({}), 10000.0, 8, 1024);
        let yarn = inv_freq(&json!({"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1}), 10000.0, 8, 1024);
        assert_eq!(plain.len(), 4);
        assert!((yarn[0] - plain[0]).abs() < 1e-9, "the fastest rotation is not scaled");
        assert!((yarn[3] - plain[3] / 16.0).abs() < 1e-9 * plain[3].max(1.0), "the slowest rotation is divided by the factor");
    }

    #[test]
    fn maps_stacked_experts_and_routing_tensors() {
        let plan = plan(&json!({"model_type": "deepseek_v4", "hidden_size": 32, "num_attention_heads": 4, "num_hidden_layers": 3, "head_dim": 16, "moe_intermediate_size": 16,
            "vocab_size": 64, "n_routed_experts": 4, "num_experts_per_tok": 2, "qk_rope_head_dim": 8, "q_lora_rank": 8})).unwrap();
        assert_eq!(plan.arch, "deepseek4");
        assert_eq!(plan.map("model.layers.1.mlp.gate.tid2eid").unwrap(), Mapped::Tensor { name: "blk.1.ffn_tid2eid.weight".into(), add: 0.0 });
        assert_eq!(plan.map("model.layers.1.mlp.experts.gate_up_proj").unwrap(), Mapped::Fused { layer: 1, part: "experts_input".into() });
        assert_eq!(plan.map("layers.2.ffn.experts.5.w3.weight").unwrap(), Mapped::Expert { name: "blk.2.ffn_up_exps.weight".into(), index: 5 });
        assert_eq!(plan.map("model.hc_head.hc_scale").unwrap(), Mapped::Tensor { name: "output_hc_scale.weight".into(), add: 0.0 });
        assert!(plan.synthesized.iter().any(|(n, v)| n == "rope_freqs_main.weight" && v.len() == 4));
    }
}
