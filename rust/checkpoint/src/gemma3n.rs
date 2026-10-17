//! Gemma 3n text model (`gemma3n_text`, `gemma3n`): Hugging Face config and tensor names to the fLLM GGUF layout.
//!
//! Metadata (llama.cpp names, `gemma3n.` prefix): the common `attention.*` keys, `feed_forward_length` (one entry per layer),
//! `attention.shared_kv_layers`, `embedding_length_per_layer_input`, `altup.num_inputs`, `altup.active_idx`,
//! `altup.correct_scale`, `laurel_rank` and `activation_sparsity_scale` with one entry per layer: the factor of the standard
//! deviation that is added to the mean to form the cut-off of the Gaussian top-k of the gate activations (0: dense layer;
//! the reference derives it from the sparsity as the normal quantile).
//!
//! Tensors: `altup_proj.weight` and `altup_unembd_proj.weight` (the three projections of the model stacked on a third axis),
//! per layer `altup_predict_coef`, `altup_correct_coef`, `altup_router`, `altup_router_norm`, `altup_correct_scale`,
//! `laurel_l`, `laurel_r`, `laurel_post_norm`, `inp_gate`, `proj`, `post_norm`, plus the sandwich-normed attention and MLP
//! tensors of Gemma 3. Layers that read the KV cache of an earlier layer have no key / value tensors.
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

/// Inverse of the standard normal distribution function (Wichura's algorithm AS 241).
// the coefficients are the published ones, with all their digits
#[allow(clippy::excessive_precision)]
pub(crate) fn normal_quantile(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let value = if r <= 5.0 {
        r -= 1.6;
        (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else {
        r -= 5.0;
        (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -value
    } else {
        value
    }
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let heads = num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let kv = num(cfg, "num_key_value_heads").unwrap_or(heads);
    let head_dim = num(cfg, "head_dim").unwrap_or(hidden / heads.max(1));
    let ctx = num(cfg, "max_position_embeddings").unwrap_or(32_768);
    let activation = cfg
        .get("hidden_activation")
        .or_else(|| cfg.get("hidden_act"))
        .and_then(Value::as_str)
        .unwrap_or("gelu_pytorch_tanh");
    if activation != "gelu_pytorch_tanh" {
        return Err(format!("unsupported activation {activation:?}; gemma3n uses gelu_pytorch_tanh"));
    }
    let inter: Vec<u32> = match cfg.get("intermediate_size") {
        Some(Value::Array(list)) if list.len() == layers => list.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect(),
        Some(v) if v.as_u64().is_some() => vec![v.as_u64().unwrap_or(0) as u32; layers],
        _ => return Err("Gemma 3n intermediate_size must be a number or one entry per layer".into()),
    };
    let sliding: Vec<bool> = match cfg.get("layer_types").and_then(Value::as_array) {
        Some(list) if list.len() == layers => list.iter().map(|v| v.as_str() == Some("sliding_attention")).collect(),
        Some(_) => return Err("layer_types does not match num_hidden_layers".into()),
        None => (0..layers).map(|i| (i + 1) % 5 != 0).collect(),
    };
    let shared = num(cfg, "num_kv_shared_layers").unwrap_or(0) as usize;
    if shared >= layers {
        return Err("num_kv_shared_layers leaves no layer with its own KV cache".into());
    }
    let sparsity: Vec<f32> = match cfg.get("activation_sparsity_pattern").and_then(Value::as_array) {
        Some(list) if list.len() == layers => list
            .iter()
            .map(|v| {
                let p = v.as_f64().unwrap_or(0.0);
                if p > 0.0 {
                    normal_quantile(p) as f32
                } else {
                    0.0
                }
            })
            .collect(),
        Some(_) => return Err("activation_sparsity_pattern does not match num_hidden_layers".into()),
        None => vec![0.0; layers],
    };
    let arch = "gemma3n";
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
    u("attention.head_count_kv", kv);
    u("attention.key_length", head_dim);
    u("attention.value_length", head_dim);
    u("attention.sliding_window", num(cfg, "sliding_window").filter(|&w| w > 0).unwrap_or(512));
    u("rope.dimension_count", head_dim);
    u("embedding_length_per_layer_input", num(cfg, "hidden_size_per_layer_input").unwrap_or(0));
    u("altup.num_inputs", num(cfg, "altup_num_inputs").unwrap_or(4));
    u("altup.active_idx", num(cfg, "altup_active_idx").unwrap_or(0));
    u("laurel_rank", num(cfg, "laurel_rank").unwrap_or(0));
    if shared > 0 {
        u("attention.shared_kv_layers", shared as u64);
    }
    let mut f = |k: &str, v: f64| meta.push((format!("{arch}.{k}"), Meta::F32(v as f32)));
    f("attention.layer_norm_rms_epsilon", flt(cfg, "rms_norm_eps").unwrap_or(1e-6));
    f("attention.scale", 1.0);
    f("embedding_scale", (hidden as f64).sqrt());
    if let Some(cap) = flt(cfg, "final_logit_softcapping") {
        f("final_logit_softcapping", cap);
    }
    let global = rope_params(cfg, Some("full_attention"));
    let local = rope_params(cfg, Some("sliding_attention"));
    if global.kind != "default" || local.kind != "default" {
        return Err("Gemma 3n with scaled rotary embeddings is unsupported".into());
    }
    f("rope.freq_base", global.theta.unwrap_or(1_000_000.0));
    f("rope.freq_base_swa", local.theta.or_else(|| flt(cfg, "rope_local_base_freq")).unwrap_or(10_000.0));
    meta.push((
        format!("{arch}.altup.correct_scale"),
        Meta::Bool(cfg.get("altup_correct_scale").and_then(Value::as_bool).unwrap_or(true)),
    ));
    meta.push((format!("{arch}.feed_forward_length"), Meta::U32s(inter)));
    meta.push((format!("{arch}.activation_sparsity_scale"), Meta::F32s(sparsity)));
    meta.push((format!("{arch}.attention.sliding_window_pattern"), Meta::Bools(sliding)));
    Ok(Plan {
        arch: arch.into(),
        family: Family::Gemma,
        meta,
        norm_add: 0.0,
        layers: layers as u64,
        moe: false,
        gated: true,
        moe_layers: vec![false; layers],
        q_lora: false,
        mixtral_names: false,
        synthesized: Vec::new(),
        linear: None,
        kinds: Vec::new(),
        classic: None,
    })
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
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
    for (prefix, name) in [
        ("model.altup_projections.", "altup_proj.weight"),
        ("model.altup_unembed_projections.", "altup_unembd_proj.weight"),
    ] {
        if let Some(rest) = n.strip_prefix(prefix) {
            let index = rest
                .strip_suffix(".weight")
                .and_then(|i| i.parse::<usize>().ok())
                .ok_or_else(|| format!("unrecognized Gemma 3n tensor {n:?}"))?;
            return Ok(Mapped::Expert { name: name.to_owned(), index });
        }
    }
    let rest = n
        .strip_prefix("model.layers.")
        .ok_or_else(|| format!("unrecognized Gemma 3n tensor {n:?}"))?;
    let (index, suffix) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: usize = index.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let layers = plan.layers as usize;
    if layer >= layers {
        return Err(format!("layer index out of range in {n:?}"));
    }
    let shared = plan
        .meta
        .iter()
        .find_map(|(k, v)| match v {
            Meta::U32(n) if k == "gemma3n.attention.shared_kv_layers" => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(0);
    if layer >= layers - shared
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
        "per_layer_input_gate.weight" => "inp_gate.weight",
        "per_layer_projection.weight" => "proj.weight",
        "post_per_layer_input_norm.weight" => "post_norm.weight",
        "altup.correct_output_scale" => "altup_correct_scale.weight",
        "altup.correction_coefs.weight" => "altup_correct_coef.weight",
        "altup.prediction_coefs.weight" => "altup_predict_coef.weight",
        "altup.modality_router.weight" => "altup_router.weight",
        "altup.router_norm.weight" => "altup_router_norm.weight",
        "laurel.linear_left.weight" => "laurel_l.weight",
        "laurel.linear_right.weight" => "laurel_r.weight",
        "laurel.post_laurel_norm.weight" => "laurel_post_norm.weight",
        _ => return Err(format!("unrecognized Gemma 3n layer tensor {n:?}")),
    };
    Ok(plain(&format!("blk.{layer}.{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Value {
        json!({"model_type": "gemma3n_text", "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
               "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": [64, 64, 128, 128],
               "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
               "num_kv_shared_layers": 2, "activation_sparsity_pattern": [0.95, 0.0, 0.0, 0.0],
               "hidden_size_per_layer_input": 8, "laurel_rank": 4, "altup_num_inputs": 4, "sliding_window": 16,
               "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1.0e6},
                                   "sliding_attention": {"rope_type": "default", "rope_theta": 1.0e4}}})
    }

    #[test]
    fn normal_quantile_matches_known_values() {
        assert!((normal_quantile(0.95) - 1.644_853_626_951_472_7).abs() < 1e-12);
        assert!((normal_quantile(0.5)).abs() < 1e-15);
        assert!((normal_quantile(0.999_999) - 4.753_424_308_822_899).abs() < 1e-9);
        assert!((normal_quantile(0.01) + 2.326_347_874_040_841).abs() < 1e-12);
    }

    #[test]
    fn metadata_carries_the_per_layer_values() {
        let plan = plan(&config()).unwrap();
        let find = |key: &str| plan.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert!(matches!(find("gemma3n.feed_forward_length"), Some(Meta::U32s(v)) if v == &[64, 64, 128, 128]));
        assert!(matches!(find("gemma3n.attention.shared_kv_layers"), Some(Meta::U32(2))));
        assert!(matches!(find("gemma3n.activation_sparsity_scale"), Some(Meta::F32s(v)) if (v[0] - 1.644_853_6).abs() < 1e-6 && v[1] == 0.0));
        assert!(matches!(find("gemma3n.attention.sliding_window_pattern"), Some(Meta::Bools(v)) if v == &[true, true, true, false]));
        assert!(matches!(find("gemma3n.rope.freq_base_swa"), Some(Meta::F32(v)) if *v == 1.0e4));
    }

    #[test]
    fn tensor_names_follow_the_gguf_layout_and_shared_layers_drop_their_key_value_path() {
        let plan = plan(&config()).unwrap();
        let name = |n: &str| match map(&plan, n).unwrap() {
            Mapped::Tensor { name, add } => {
                assert_eq!(add, 0.0, "Gemma 3n norms are not offset");
                name
            }
            other => panic!("{other:?}"),
        };
        assert_eq!(name("model.layers.1.altup.modality_router.weight"), "blk.1.altup_router.weight");
        assert_eq!(name("model.layers.0.laurel.linear_left.weight"), "blk.0.laurel_l.weight");
        assert_eq!(name("model.layers.0.self_attn.k_proj.weight"), "blk.0.attn_k.weight");
        assert_eq!(map(&plan, "model.layers.2.self_attn.k_proj.weight").unwrap(), Mapped::Skip);
        assert_eq!(map(&plan, "model.layers.3.self_attn.v_norm.weight").unwrap(), Mapped::Skip);
        assert_eq!(
            map(&plan, "model.altup_projections.2.weight").unwrap(),
            Mapped::Expert { name: "altup_proj.weight".into(), index: 2 }
        );
        assert!(map(&plan, "model.layers.0.something.weight").is_err());
    }
}
