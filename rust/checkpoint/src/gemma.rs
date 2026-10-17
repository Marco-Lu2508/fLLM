//! Gemma 4 (`gemma4`, `gemma4_text`): Hugging Face config and tensor names to the fLLM GGUF layout.
//!
//! Metadata written (`gemma4.` prefix, llama.cpp key names): `attention.head_count`, `attention.head_count_kv` (one entry
//! per layer), `feed_forward_length` (one entry per layer: layers that share a KV cache of an earlier layer use a double-wide
//! MLP when `use_double_wide_mlp` is set), `attention.key_length` / `attention.value_length` (global layers) and
//! `attention.key_length_swa` / `attention.value_length_swa` (sliding-window layers), `attention.sliding_window`,
//! `attention.sliding_window_pattern` (bool per layer), `attention.shared_kv_layers`, `attention.scale` (1.0: the reference
//! does not scale the scores), `rope.freq_base`, `rope.freq_base_swa`, `rope.dimension_count`, `rope.dimension_count_swa`,
//! `embedding_length_per_layer_input`, `final_logit_softcapping`, and for the expert block `expert_count`,
//! `expert_used_count`, `expert_feed_forward_length`.
//!
//! The proportional rotary embedding of the global layers rotates the first `partial_rotary_factor * head_dim` dimensions
//! with frequencies `theta^(-2i / head_dim)` and leaves the rest untouched. It is written as `rope_freqs.weight` (one
//! divisor per frequency: 1 for the rotating ones, 1e30 for the others), the tensor the engine already reads.
//!
//! Tensors: attention `attn_{q,k,v,output}`, `attn_{q,k}_norm` (no value norm weight: the value norm has none),
//! sandwich norms `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`, `layer_output_scale`; per-layer
//! embeddings `per_layer_token_embd`, `per_layer_model_proj`, `per_layer_proj_norm`, `blk.N.inp_gate`, `blk.N.proj`,
//! `blk.N.post_norm`; expert block `ffn_gate_inp` (+ `.scale`), `ffn_gate_up_exps`, `ffn_down_exps` (+ `.scale` = per-expert
//! scale), `pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`. A global layer with `attention_k_eq_v` has no `attn_v`
//! tensor; layers that read the KV cache of an earlier layer have no `attn_k` / `attn_v` / `attn_k_norm`.
use crate::{
    gguf::Meta,
    plan::{rope_params, Family, Mapped, Plan},
};
use serde_json::Value;

fn num(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::as_u64)
}
fn flt(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

/// Effective `(head_dim, kv_heads)` of a layer: the explicit `per_layer_config` entry of the saved configuration, otherwise
/// the `global_head_dim` / `num_global_key_value_heads` rule of the transformers configuration class.
fn layer_shape(cfg: &Value, layer: usize, full: bool) -> Result<(u64, u64), String> {
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let mut head_dim = num(cfg, "head_dim").ok_or("Gemma 4 config requires head_dim")?;
    let mut kv = num(cfg, "num_key_value_heads").unwrap_or(heads);
    let k_eq_v = cfg.get("attention_k_eq_v").and_then(Value::as_bool).unwrap_or(false);
    if let Some(overrides) = cfg.get("per_layer_config").filter(|v| v.is_object()) {
        if let Some(over) = overrides.get(layer.to_string()) {
            head_dim = num(over, "head_dim").unwrap_or(head_dim);
            kv = num(over, "num_key_value_heads").unwrap_or(kv);
        }
    } else if full {
        head_dim = num(cfg, "global_head_dim").unwrap_or(512);
        if k_eq_v {
            kv = num(cfg, "num_global_key_value_heads").unwrap_or(kv);
        }
    }
    Ok((head_dim, kv))
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let inter = num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?;
    let ctx = num(cfg, "max_position_embeddings").unwrap_or(131_072);
    let activation = cfg
        .get("hidden_activation")
        .or_else(|| cfg.get("hidden_act"))
        .and_then(Value::as_str)
        .unwrap_or("gelu_pytorch_tanh");
    if activation != "gelu_pytorch_tanh" {
        return Err(format!("unsupported activation {activation:?}; gemma4 uses gelu_pytorch_tanh"));
    }
    if cfg.get("use_bidirectional_attention").is_some_and(|v| !v.is_null()) {
        return Err("Gemma 4 with use_bidirectional_attention is unsupported".into());
    }
    if cfg.get("attention_bias").and_then(Value::as_bool) == Some(true) {
        return Err("Gemma 4 with attention_bias is unsupported".into());
    }
    let types: Vec<String> = match cfg.get("layer_types").and_then(Value::as_array) {
        Some(list) if list.len() == layers => list.iter().map(|v| v.as_str().unwrap_or("").to_owned()).collect(),
        Some(_) => return Err("layer_types does not match num_hidden_layers".into()),
        None => (0..layers)
            .map(|i| if (i + 1) % 6 != 0 { "sliding_attention" } else { "full_attention" }.to_owned())
            .collect(),
    };
    let mut types = types;
    if let Some(last) = types.last_mut() {
        // the reference forces the last layer to full attention
        *last = "full_attention".into();
    }
    if types.iter().any(|t| t != "sliding_attention" && t != "full_attention") {
        return Err("unsupported layer type in Gemma 4 layer_types".into());
    }
    let sliding: Vec<bool> = types.iter().map(|t| t == "sliding_attention").collect();
    let window = num(cfg, "sliding_window").filter(|&w| w > 0).unwrap_or(512);

    let shared = num(cfg, "num_kv_shared_layers").unwrap_or(0) as usize;
    if shared >= layers {
        return Err("num_kv_shared_layers leaves no layer with its own KV cache".into());
    }
    let double_wide = cfg.get("use_double_wide_mlp").and_then(Value::as_bool).unwrap_or(false);

    let mut kv_heads = Vec::with_capacity(layers);
    let mut ff = Vec::with_capacity(layers);
    let (mut full_dim, mut sliding_dim) = (None, None);
    for (i, &is_sliding) in sliding.iter().enumerate() {
        let (head_dim, kv) = layer_shape(cfg, i, !is_sliding)?;
        let slot = if is_sliding { &mut sliding_dim } else { &mut full_dim };
        match slot {
            None => *slot = Some(head_dim),
            Some(d) if *d != head_dim => return Err("Gemma 4 layers of one kind must share a head_dim".into()),
            Some(_) => {}
        }
        kv_heads.push(kv as u32);
        ff.push((inter * if double_wide && i >= layers - shared { 2 } else { 1 }) as u32);
    }
    let global_dim = full_dim.ok_or("Gemma 4 needs a full-attention layer")?;
    let local_dim = sliding_dim.unwrap_or(global_dim);

    let eps = flt(cfg, "rms_norm_eps").unwrap_or(1e-6);
    let arch = "gemma4";
    let mut meta: Vec<(String, Meta)> = vec![
        ("general.architecture".into(), Meta::Str(arch.into())),
        (
            "general.name".into(),
            Meta::Str(cfg.get("_name_or_path").and_then(Value::as_str).unwrap_or(arch).into()),
        ),
        ("general.alignment".into(), Meta::U32(32)),
    ];
    let mut u = |k: &str, v: u64| meta.push((format!("{arch}.{k}"), Meta::U32(v as u32)));
    u("context_length", ctx);
    u("embedding_length", hidden);
    u("block_count", layers as u64);
    u("attention.head_count", heads);
    u("attention.key_length", global_dim);
    u("attention.value_length", global_dim);
    u("attention.key_length_swa", local_dim);
    u("attention.value_length_swa", local_dim);
    u("attention.sliding_window", window);
    u("rope.dimension_count", global_dim);
    u("rope.dimension_count_swa", local_dim);
    if shared > 0 {
        u("attention.shared_kv_layers", shared as u64);
    }
    let ple = num(cfg, "hidden_size_per_layer_input").unwrap_or(0);
    if ple > 0 {
        u("embedding_length_per_layer_input", ple);
    }
    let moe = cfg.get("enable_moe_block").and_then(Value::as_bool).unwrap_or(false);
    if moe {
        let experts = num(cfg, "num_experts").ok_or("Gemma 4 MoE config requires num_experts")?;
        u("expert_count", experts);
        u("expert_used_count", num(cfg, "top_k_experts").ok_or("Gemma 4 MoE config requires top_k_experts")?);
        u("expert_feed_forward_length", num(cfg, "moe_intermediate_size").ok_or("Gemma 4 MoE config requires moe_intermediate_size")?);
    }
    let mut f = |k: &str, v: f64| meta.push((format!("{arch}.{k}"), Meta::F32(v as f32)));
    f("attention.layer_norm_rms_epsilon", eps);
    f("attention.scale", 1.0);
    f("embedding_scale", (hidden as f64).sqrt());
    if let Some(cap) = flt(cfg, "final_logit_softcapping") {
        f("final_logit_softcapping", cap);
    }
    // rotary embeddings
    let global = rope_params(cfg, Some("full_attention"));
    let local = rope_params(cfg, Some("sliding_attention"));
    f("rope.freq_base", global.theta.unwrap_or(1_000_000.0));
    f("rope.freq_base_swa", local.theta.unwrap_or(10_000.0));
    if local.kind != "default" {
        return Err(format!("unsupported rope type {:?} for the sliding-window layers", local.kind));
    }
    let mut synthesized = Vec::new();
    match global.kind.as_str() {
        "default" => {}
        "proportional" => {
            let proportion = flt(&global.raw, "partial_rotary_factor").unwrap_or(1.0);
            let angles = ((proportion * global_dim as f64) as u64 / 2) as usize;
            let half = (global_dim / 2) as usize;
            if angles > half {
                return Err("partial_rotary_factor above 1".into());
            }
            if flt(&global.raw, "factor").is_some_and(|x| (x - 1.0).abs() > 1e-9) {
                return Err("proportional rope with a scaling factor is unsupported".into());
            }
            if angles < half {
                let mut divisors = vec![1.0f32; angles];
                divisors.resize(half, 1.0e30);
                synthesized.push(("rope_freqs.weight".to_owned(), divisors));
            }
        }
        other => return Err(format!("unsupported rope type {other:?} for the full-attention layers")),
    }
    meta.push((format!("{arch}.attention.head_count_kv"), Meta::U32s(kv_heads)));
    meta.push((format!("{arch}.feed_forward_length"), Meta::U32s(ff)));
    meta.push((format!("{arch}.attention.sliding_window_pattern"), Meta::Bools(sliding)));
    Ok(Plan {
        arch: arch.into(),
        family: Family::Gemma,
        meta,
        // the reference multiplies by the stored weight (no 1 + w)
        norm_add: 0.0,
        layers: layers as u64,
        moe: false,
        gated: true,
        moe_layers: vec![false; layers],
        q_lora: false,
        mixtral_names: false,
        synthesized,
        linear: None,
        kinds: Vec::new(),
        classic: None,
    })
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
    let layers = plan.layers as usize;
    let shared = plan.meta.iter().find_map(|(k, v)| match v {
        Meta::U32(n) if k == "gemma4.attention.shared_kv_layers" => Some(*n as usize),
        _ => None,
    });
    let shared_from = layers - shared.unwrap_or(0);
    let plain = |name: &str| Mapped::Tensor { name: name.to_owned(), add: 0.0 };
    if n.starts_with("model.audio_tower.")
        || n.starts_with("model.embed_audio.")
        || n.starts_with("model.embed_vision.")
        || n.starts_with("audio_tower.")
        || n.starts_with("embed_audio.")
        || n.starts_with("embed_vision.")
    {
        return Ok(Mapped::Skip);
    }
    match n {
        "model.embed_tokens.weight" => return Ok(plain("token_embd.weight")),
        "model.embed_tokens_per_layer.weight" => return Ok(plain("per_layer_token_embd.weight")),
        "model.per_layer_model_projection.weight" => return Ok(plain("per_layer_model_proj.weight")),
        "model.per_layer_projection_norm.weight" => return Ok(plain("per_layer_proj_norm.weight")),
        "model.norm.weight" => return Ok(plain("output_norm.weight")),
        "lm_head.weight" => return Ok(plain("output.weight")),
        _ => {}
    }
    let rest = n
        .strip_prefix("model.layers.")
        .ok_or_else(|| format!("unrecognized Gemma 4 tensor {n:?}"))?;
    let (index, suffix) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: usize = index.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    if layer >= plan.layers as usize {
        return Err(format!("layer index out of range in {n:?}"));
    }
    if suffix == "self_attn.rotary_emb.inv_freq" {
        return Ok(Mapped::Skip);
    }
    // layers that read the KV cache of an earlier layer have no key / value path
    if layer >= shared_from
        && matches!(
            suffix,
            "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.k_norm.weight" | "self_attn.v_norm.weight"
        )
    {
        return Ok(Mapped::Skip);
    }
    let name = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "layer_scalar" => "layer_output_scale.weight",
        "per_layer_input_gate.weight" => "inp_gate.weight",
        "per_layer_projection.weight" => "proj.weight",
        "post_per_layer_input_norm.weight" => "post_norm.weight",
        "router.proj.weight" => "ffn_gate_inp.weight",
        "router.scale" => "ffn_gate_inp.scale",
        "router.per_expert_scale" => "ffn_down_exps.scale",
        "experts.gate_up_proj" => "ffn_gate_up_exps.weight",
        "experts.down_proj" => "ffn_down_exps.weight",
        "pre_feedforward_layernorm_2.weight" => "pre_ffw_norm_2.weight",
        "post_feedforward_layernorm_1.weight" => "post_ffw_norm_1.weight",
        "post_feedforward_layernorm_2.weight" => "post_ffw_norm_2.weight",
        _ => return Err(format!("unrecognized Gemma 4 layer tensor {n:?}")),
    };
    Ok(plain(&format!("blk.{layer}.{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Value {
        json!({"model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 6, "num_attention_heads": 4,
               "num_key_value_heads": 4, "head_dim": 8, "global_head_dim": 16, "num_global_key_value_heads": 1,
               "attention_k_eq_v": true, "intermediate_size": 64, "sliding_window": 16,
               "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
               "num_kv_shared_layers": 2, "use_double_wide_mlp": true, "hidden_size_per_layer_input": 8,
               "enable_moe_block": true, "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 16,
               "rope_parameters": {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1.0e6},
                                   "sliding_attention": {"rope_type": "default", "rope_theta": 1.0e4}}})
    }

    #[test]
    fn metadata_describes_both_kinds_of_layers() {
        let plan = plan(&config()).unwrap();
        let find = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert!(matches!(find("gemma4.attention.head_count_kv"), Some(Meta::U32s(v)) if v == &[4, 4, 4, 4, 4, 1]));
        assert!(matches!(find("gemma4.feed_forward_length"), Some(Meta::U32s(v)) if v == &[64, 64, 64, 64, 128, 128]));
        assert!(matches!(find("gemma4.attention.key_length"), Some(Meta::U32(16))));
        assert!(matches!(find("gemma4.attention.key_length_swa"), Some(Meta::U32(8))));
        assert!(matches!(find("gemma4.rope.dimension_count"), Some(Meta::U32(16))));
        assert!(matches!(find("gemma4.attention.scale"), Some(Meta::F32(v)) if *v == 1.0));
        // 25 % of the 16-wide global head rotate: 2 of the 8 frequencies
        let freqs = &plan.synthesized.iter().find(|(n, _)| n == "rope_freqs.weight").unwrap().1;
        assert_eq!(freqs.len(), 8);
        assert_eq!(&freqs[..2], &[1.0, 1.0]);
        assert!(freqs[2..].iter().all(|f| *f == 1.0e30));
        assert_eq!(plan.norm_add, 0.0);
    }

    #[test]
    fn the_saved_per_layer_config_is_honoured() {
        let mut cfg = config();
        cfg.as_object_mut().unwrap().remove("global_head_dim");
        cfg["per_layer_config"] = json!({"5": {"head_dim": 32, "num_key_value_heads": 2}});
        let plan = plan(&cfg).unwrap();
        let find = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert!(matches!(find("gemma4.attention.key_length"), Some(Meta::U32(32))));
        assert!(matches!(find("gemma4.attention.head_count_kv"), Some(Meta::U32s(v)) if v == &[4, 4, 4, 4, 4, 2]));
    }

    #[test]
    fn tensors_map_to_the_expected_names() {
        let plan = plan(&config()).unwrap();
        let name = |n: &str| match map(&plan, n).unwrap() {
            Mapped::Tensor { name, .. } => name,
            other => panic!("{other:?}"),
        };
        assert_eq!(name("model.layers.0.pre_feedforward_layernorm_2.weight"), "blk.0.pre_ffw_norm_2.weight");
        assert_eq!(name("model.layers.0.router.per_expert_scale"), "blk.0.ffn_down_exps.scale");
        assert_eq!(name("model.layers.5.experts.gate_up_proj"), "blk.5.ffn_gate_up_exps.weight");
        assert_eq!(name("model.layers.1.layer_scalar"), "blk.1.layer_output_scale.weight");
        assert_eq!(name("model.embed_tokens_per_layer.weight"), "per_layer_token_embd.weight");
        // layers 4 and 5 read the cache of an earlier layer
        assert_eq!(map(&plan, "model.layers.4.self_attn.k_proj.weight").unwrap(), Mapped::Skip);
        assert_eq!(map(&plan, "model.layers.5.self_attn.v_norm.weight").unwrap(), Mapped::Skip);
        assert_eq!(name("model.layers.3.self_attn.k_norm.weight"), "blk.3.attn_k_norm.weight");
        assert!(map(&plan, "model.layers.0.unknown.weight").is_err());
    }

    #[test]
    fn unsupported_variants_are_refused() {
        let mut cfg = config();
        cfg["hidden_activation"] = json!("silu");
        assert!(plan(&cfg).err().unwrap().contains("gelu_pytorch_tanh"));
        let mut cfg = config();
        cfg["use_bidirectional_attention"] = json!("all");
        assert!(plan(&cfg).is_err());
        let mut cfg = config();
        cfg["rope_parameters"]["full_attention"]["rope_type"] = json!("yarn");
        assert!(plan(&cfg).is_err());
    }
}
