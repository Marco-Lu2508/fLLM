//! GLM-5-Next / GLM-5.3-Flash as llama.cpp writes it (`general.architecture = glm5-next`).
//!
//! The model is a hybrid of Kimi Delta Attention (KDA) layers and NoPE multi-head latent attention
//! (MLA) layers with a k-pool lightning indexer, manifold-constrained hyper-connections (`hc_*`),
//! and clamped-SwiGLU MoE feed-forwards (`kimi-linear` in fLLM).
//!
//! | llama.cpp (`glm5-next.`)                                   | engine view (`kimi-linear.`)                                |
//! |------------------------------------------------------------|-------------------------------------------------------------|
//! | `block_count` - `nextn_predict_layers`                     | `block_count`                                               |
//! | `hyper_connection.count`, `.sinkhorn_iters`, `.eps`        | `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `hc_head_mean`    |
//! | `ssm.conv_kernel`, `kda.head_count`, `kda.head_dim`        | `ssm.conv_kernel`, `kda.head_count`, `kda.head_dim`         |
//! | `kda.layer_flags` (or inferred from `blk.N.ssm_*`)         | `kda.layer_flags`                                           |
//! | NoPE MLA                                                   | `rope.dimension_count = 0`, `attention.mla_no_rope = true`  |
//! | `attention.indexer.*` (`kpool`, `kpool_always_tail`)       | `attention.indexer.*`                                       |
//! | `expert_swiglu_clamp` / `swiglu_limit` / `swiglu_clamp_exp`| `expert_swiglu_clamp`                                       |
use fllm_gguf_probe::{MappedGguf, TensorInfo};
use serde_json::{Map, Value};

const FROM: &str = "glm5-next";
const TO: &str = "kimi-linear";

fn uint(meta: &Map<String, Value>, name: &str) -> Option<u64> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_u64)
}

fn flt(meta: &Map<String, Value>, name: &str) -> Option<f64> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_f64)
}

fn flag(meta: &Map<String, Value>, name: &str) -> Option<bool> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_bool)
}

fn list<'a>(meta: &'a Map<String, Value>, name: &str) -> Option<&'a Vec<Value>> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_array)
}

fn need(meta: &Map<String, Value>, name: &str) -> Result<u64, String> {
    uint(meta, name)
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("glm5-next GGUF needs {FROM}.{name}"))
}

/// `(translated metadata, per-layer kda flags, trunk layers, optional (kda_heads, kda_head_dim))`.
type Translated = (Vec<(String, Value)>, Vec<bool>, usize, Option<(usize, usize)>);

fn kda_layer_flags(meta: &Map<String, Value>, tensors: &[TensorInfo], layers: usize) -> Result<Vec<bool>, String> {
    if let Some(arr) = list(meta, "kda.layer_flags").or_else(|| list(meta, "ssm.layer_flags")) {
        if arr.len() < layers {
            return Err(format!("{FROM}.kda.layer_flags has {} entries, expected at least {layers}", arr.len()));
        }
        return arr
            .iter()
            .take(layers)
            .map(|v| {
                v.as_bool()
                    .or_else(|| v.as_u64().map(|n| n != 0))
                    .ok_or_else(|| format!("{FROM}.kda.layer_flags must hold booleans or 0/1 integers"))
            })
            .collect();
    }
    if let Some(arr) = list(meta, "layer_types") {
        if arr.len() < layers {
            return Err(format!("{FROM}.layer_types has {} entries, expected at least {layers}", arr.len()));
        }
        return arr
            .iter()
            .take(layers)
            .map(|v| match v.as_str() {
                Some("linear_attention" | "kda") => Ok(true),
                Some("full_attention" | "indexed_attention" | "mla") => Ok(false),
                other => Err(format!("unsupported {FROM} layer type {other:?}")),
            })
            .collect();
    }
    Ok((0..layers)
        .map(|l| {
            let prefix = format!("blk.{l}.");
            let has_ssm = tensors.iter().any(|t| {
                t.name
                    .strip_prefix(prefix.as_str())
                    .is_some_and(|rest| rest.starts_with("ssm_") || rest.starts_with("kda_"))
            });
            let has_mla = tensors
                .iter()
                .any(|t| t.name.strip_prefix(prefix.as_str()) == Some("attn_kv_a_mqa.weight"));
            has_ssm && !has_mla
        })
        .collect())
}

/// Metadata of the engine view (under the `kimi-linear.` prefix).
fn translated(meta: &Map<String, Value>, tensors: &[TensorInfo]) -> Result<Translated, String> {
    let all_layers = need(meta, "block_count")? as usize;
    let layers = all_layers.saturating_sub(uint(meta, "nextn_predict_layers").unwrap_or(0) as usize);
    if layers == 0 {
        return Err("glm5-next GGUF without layers".into());
    }
    let mut out: Vec<(String, Value)> = Vec::new();
    let mut put = |key: &str, value: Value| out.push((format!("{TO}.{key}"), value));
    put("block_count", Value::from(layers as u64));

    let hc_mult = uint(meta, "hyper_connection.count")
        .or_else(|| uint(meta, "hc_mult"))
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("glm5-next GGUF needs {FROM}.hyper_connection.count"))?;
    let hc_iters = uint(meta, "hyper_connection.sinkhorn_iters")
        .or_else(|| uint(meta, "hc_sinkhorn_iters"))
        .filter(|&v| v > 0)
        .unwrap_or(20);
    let hc_eps = flt(meta, "hyper_connection.eps")
        .or_else(|| flt(meta, "hc_eps"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1e-6);
    put("hc_mult", Value::from(hc_mult));
    put("hc_sinkhorn_iters", Value::from(hc_iters));
    put("hc_eps", Value::from(hc_eps));
    put("hc_head_mean", Value::Bool(true));

    let kda_flags = kda_layer_flags(meta, tensors, layers)?;
    put("kda.layer_flags", Value::Array(kda_flags.iter().copied().map(Value::Bool).collect()));
    let kda_dims = if kda_flags.iter().any(|&f| f) {
        let conv_kernel = uint(meta, "ssm.conv_kernel")
            .or_else(|| uint(meta, "kda.conv_kernel"))
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("glm5-next GGUF needs {FROM}.ssm.conv_kernel"))?;
        if conv_kernel < 2 {
            return Err("glm5-next ssm.conv_kernel must be at least 2".into());
        }
        let kda_heads = uint(meta, "kda.head_count")
            .or_else(|| uint(meta, "ssm.time_step_rank"))
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("glm5-next GGUF needs {FROM}.kda.head_count"))?;
        let kda_head_dim = uint(meta, "kda.head_dim")
            .or_else(|| uint(meta, "ssm.state_size"))
            .or_else(|| uint(meta, "ssm.inner_size").map(|inner| inner / kda_heads))
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("glm5-next GGUF needs {FROM}.kda.head_dim"))?;
        put("ssm.conv_kernel", Value::from(conv_kernel));
        put("kda.head_count", Value::from(kda_heads));
        put("kda.head_dim", Value::from(kda_head_dim));
        if let Some(lb) = flt(meta, "kda.gate_lower_bound")
            .or_else(|| flt(meta, "kda.lower_bound"))
            .or_else(|| flt(meta, "ssm.gate_lower_bound"))
            .or_else(|| flt(meta, "linear_lower_bound"))
        {
            put("kda.gate_lower_bound", Value::from(lb));
        }
        Some((kda_heads as usize, kda_head_dim as usize))
    } else {
        None
    };

    // NoPE MLA: keys/queries carry no rotary dimensions.
    put("rope.dimension_count", Value::from(0u64));
    put("attention.mla_no_rope", Value::Bool(true));
    put("rope.layer_flags", Value::Array(vec![Value::Bool(false); layers]));

    if uint(meta, "attention.indexer.head_count").is_some()
        || uint(meta, "attention.indexer.key_length").is_some()
        || uint(meta, "attention.indexer.top_k").is_some()
    {
        let idx_heads = need(meta, "attention.indexer.head_count")?;
        let idx_dim = need(meta, "attention.indexer.key_length")?;
        let top_k = need(meta, "attention.indexer.top_k")?;
        put("attention.indexer.head_count", Value::from(idx_heads));
        put("attention.indexer.key_length", Value::from(idx_dim));
        put("attention.indexer.top_k", Value::from(top_k));
        let kpool = uint(meta, "attention.indexer.kpool")
            .or_else(|| uint(meta, "attention.indexer.kpool_size"))
            .or_else(|| uint(meta, "attention.index_kpool"))
            .or_else(|| uint(meta, "index_kpool"))
            .unwrap_or(1)
            .max(1);
        if kpool > 1 && top_k < kpool {
            return Err(format!("glm5-next attention.indexer.top_k ({top_k}) must be at least kpool ({kpool})"));
        }
        put("attention.indexer.kpool", Value::from(kpool));
        let always_tail = flag(meta, "attention.indexer.kpool_always_tail")
            .or_else(|| flag(meta, "attention.indexer.kpool_always_select_tail"))
            .or_else(|| flag(meta, "index_kpool_always_select_tail"))
            .unwrap_or(true);
        put("attention.indexer.kpool_always_tail", Value::Bool(always_tail));
        if let Some(neox) = flag(meta, "attention.indexer.rope_neox") {
            put("attention.indexer.rope_neox", Value::Bool(neox));
        }
        if let Some(eps) = flt(meta, "attention.indexer.norm_eps") {
            put("attention.indexer.norm_eps", Value::from(eps));
        }
    }

    if let Some(clamp) = meta
        .get(&format!("{FROM}.expert_swiglu_clamp"))
        .or_else(|| meta.get(&format!("{FROM}.swiglu_limit")))
        .or_else(|| meta.get(&format!("{FROM}.swiglu_clamp_exp")))
        .cloned()
    {
        put("expert_swiglu_clamp", clamp);
    }
    if let Some(clamp_shared) = meta
        .get(&format!("{FROM}.expert_swiglu_clamp_shared"))
        .or_else(|| meta.get(&format!("{FROM}.swiglu_clamp_shexp")))
        .cloned()
    {
        put("expert_swiglu_clamp_shared", clamp_shared);
    }

    Ok((out, kda_flags, layers, kda_dims))
}

/// The engine view of a llama.cpp `glm5-next` file.
pub(crate) fn view(mapping: MappedGguf) -> Result<MappedGguf, String> {
    let meta = mapping.index().metadata.clone();
    let tensors = mapping.index().tensors.clone();
    let (keys, kda_flags, layers, kda_dims) = translated(&meta, &tensors)?;
    let mut view = mapping.with_architecture_alias(FROM, TO);
    for (key, value) in keys {
        view = view.with_metadata(&key, value);
    }
    view = view
        .with_metadata_default(&format!("{TO}.expert_gating_func"), Value::from(2u64))
        .with_metadata_default(&format!("{TO}.expert_weights_norm"), Value::Bool(true));

    let has = |name: &str| tensors.iter().any(|t| t.name == name);
    let dims = |name: &str| tensors.iter().find(|t| t.name == name).map(|t| t.dimensions.as_slice());
    let mut aliases: Vec<(String, String)> = Vec::new();
    let mut slices: Vec<(String, String, usize, usize)> = Vec::new();
    let alias = |aliases: &mut Vec<(String, String)>, slices: &[(String, String, usize, usize)], target: String, candidates: &[String]| {
        if has(&target)
            || aliases.iter().any(|(a, _)| a == &target)
            || slices.iter().any(|(a, _, _, _)| a == &target)
        {
            return;
        }
        if let Some(src) = candidates.iter().find(|c| has(c)) {
            aliases.push((target, src.clone()));
        }
    };

    for (l, &is_kda) in kda_flags.iter().enumerate().take(layers) {
        let p = format!("blk.{l}");
        for which in ["attn", "ffn"] {
            alias(
                &mut aliases,
                &slices,
                format!("{p}.hc_{which}_fn.weight"),
                &[
                    format!("{p}.hc_{which}.fn.weight"),
                    format!("{p}.{which}_hc_fn.weight"),
                    format!("{p}.{which}_hc.fn.weight"),
                    format!("{p}.hyper_connection_{which}_fn.weight"),
                    format!("{p}.hyper_connection.{which}.fn.weight"),
                ],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.hc_{which}_base.weight"),
                &[
                    format!("{p}.hc_{which}_base.bias"),
                    format!("{p}.hc_{which}_fn.bias"),
                    format!("{p}.hc_{which}.base.weight"),
                    format!("{p}.hc_{which}.base"),
                    format!("{p}.{which}_hc_base.weight"),
                    format!("{p}.{which}_hc_base.bias"),
                    format!("{p}.{which}_hc.base.weight"),
                    format!("{p}.hyper_connection_{which}_base.weight"),
                    format!("{p}.hyper_connection.{which}.base.weight"),
                ],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.hc_{which}_scale.weight"),
                &[
                    format!("{p}.hc_{which}.scale.weight"),
                    format!("{p}.hc_{which}.scale"),
                    format!("{p}.{which}_hc_scale.weight"),
                    format!("{p}.{which}_hc.scale.weight"),
                    format!("{p}.hyper_connection_{which}_scale.weight"),
                    format!("{p}.hyper_connection.{which}.scale.weight"),
                ],
            );
        }

        if is_kda {
            if let Some((kda_heads, kda_head_dim)) = kda_dims {
                let width = kda_heads * kda_head_dim;
                if !has(&format!("{p}.attn_q.weight")) {
                    let fused_qkv = [format!("{p}.attn_qkv.weight"), format!("{p}.ssm_in.weight")]
                        .into_iter()
                        .find(|n| dims(n).is_some_and(|d| d.len() == 2 && d[1] as usize == 3 * width));
                    if let Some(fused) = fused_qkv {
                        slices.push((format!("{p}.attn_q.weight"), fused.clone(), 0, width));
                        slices.push((format!("{p}.attn_k.weight"), fused.clone(), width, width));
                        slices.push((format!("{p}.attn_v.weight"), fused, 2 * width, width));
                    } else {
                        alias(&mut aliases, &slices, format!("{p}.attn_q.weight"), &[format!("{p}.kda_q.weight")]);
                        alias(&mut aliases, &slices, format!("{p}.attn_k.weight"), &[format!("{p}.kda_k.weight")]);
                        alias(&mut aliases, &slices, format!("{p}.attn_v.weight"), &[format!("{p}.kda_v.weight")]);
                    }
                }
                if !has(&format!("{p}.ssm_conv1d_q.weight")) {
                    let conv_fused = format!("{p}.ssm_conv1d.weight");
                    if dims(&conv_fused).is_some_and(|d| d.len() == 2 && d[1] as usize == 3 * width) {
                        slices.push((format!("{p}.ssm_conv1d_q.weight"), conv_fused.clone(), 0, width));
                        slices.push((format!("{p}.ssm_conv1d_k.weight"), conv_fused.clone(), width, width));
                        slices.push((format!("{p}.ssm_conv1d_v.weight"), conv_fused, 2 * width, width));
                    } else {
                        alias(
                            &mut aliases,
                            &slices,
                            format!("{p}.ssm_conv1d_q.weight"),
                            &[format!("{p}.kda_q_conv1d.weight"), format!("{p}.ssm_q_conv1d.weight"), format!("{p}.ssm_conv_q.weight")],
                        );
                        alias(
                            &mut aliases,
                            &slices,
                            format!("{p}.ssm_conv1d_k.weight"),
                            &[format!("{p}.kda_k_conv1d.weight"), format!("{p}.ssm_k_conv1d.weight"), format!("{p}.ssm_conv_k.weight")],
                        );
                        alias(
                            &mut aliases,
                            &slices,
                            format!("{p}.ssm_conv1d_v.weight"),
                            &[format!("{p}.kda_v_conv1d.weight"), format!("{p}.ssm_v_conv1d.weight"), format!("{p}.ssm_conv_v.weight")],
                        );
                    }
                }
                if !has(&format!("{p}.ssm_f_a.weight")) && !has(&format!("{p}.ssm_f.weight")) {
                    let fg_fused = [format!("{p}.ssm_fg_a.weight"), format!("{p}.ssm_f_a_g_a.weight")]
                        .into_iter()
                        .find(|n| dims(n).is_some_and(|d| d.len() == 2 && d[1] as usize == 2 * kda_head_dim));
                    if let Some(fused) = fg_fused {
                        slices.push((format!("{p}.ssm_f_a.weight"), fused.clone(), 0, kda_head_dim));
                        slices.push((format!("{p}.ssm_g_a.weight"), fused, kda_head_dim, kda_head_dim));
                    }
                }
            }
            alias(
                &mut aliases,
                &slices,
                format!("{p}.attn_output.weight"),
                &[format!("{p}.ssm_out.weight"), format!("{p}.kda_out.weight"), format!("{p}.attn_o.weight")],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.ssm_o_norm.weight"),
                &[format!("{p}.ssm_norm.weight"), format!("{p}.kda_o_norm.weight"), format!("{p}.kda_norm.weight")],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.ssm_dt.bias"),
                &[format!("{p}.ssm_dt_bias"), format!("{p}.ssm_dt_bias.weight"), format!("{p}.ssm_dt.weight"), format!("{p}.kda_dt.bias")],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.ssm_a"),
                &[format!("{p}.ssm_a.weight"), format!("{p}.ssm_A"), format!("{p}.ssm_a_log"), format!("{p}.kda_a")],
            );
            alias(&mut aliases, &slices, format!("{p}.ssm_beta.weight"), &[format!("{p}.kda_beta.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_f_a.weight"), &[format!("{p}.kda_f_a.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_f_b.weight"), &[format!("{p}.kda_f_b.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_g_a.weight"), &[format!("{p}.kda_g_a.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_g_b.weight"), &[format!("{p}.kda_g_b.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_f.weight"), &[format!("{p}.kda_f.weight")]);
            alias(&mut aliases, &slices, format!("{p}.ssm_g.weight"), &[format!("{p}.kda_g.weight")]);
        } else {
            alias(
                &mut aliases,
                &slices,
                format!("{p}.indexer.attn_q_b.weight"),
                &[
                    format!("{p}.indexer.q_b.weight"),
                    format!("{p}.indexer.wq_b.weight"),
                    format!("{p}.indexer.q_proj.weight"),
                    format!("{p}.indexer.q.weight"),
                ],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.indexer.attn_k.weight"),
                &[format!("{p}.indexer.k.weight"), format!("{p}.indexer.wk.weight"), format!("{p}.indexer.k_proj.weight")],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.indexer.proj.weight"),
                &[format!("{p}.indexer.weights_proj.weight"), format!("{p}.indexer.w_proj.weight")],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.indexer.pool_ape.weight"),
                &[
                    format!("{p}.indexer.kpool_ape.weight"),
                    format!("{p}.indexer.kpool_compress_ape.weight"),
                    format!("{p}.indexer.kpool_compress_ape"),
                    format!("{p}.indexer.index_kpool_compress_ape"),
                    format!("{p}.indexer.pool_ape"),
                ],
            );
            alias(
                &mut aliases,
                &slices,
                format!("{p}.indexer.pool_gate.weight"),
                &[
                    format!("{p}.indexer.kpool_gate.weight"),
                    format!("{p}.indexer.kpool_compress_gate.weight"),
                    format!("{p}.indexer.kpool_compress_gate"),
                    format!("{p}.indexer.index_kpool_compress_gate"),
                    format!("{p}.indexer.pool_gate"),
                ],
            );
        }
    }

    view = view.with_tensor_aliases(&aliases)?;
    view = view.with_row_slices(&slices)?;
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use fllm_checkpoint::gguf::{write_to, Meta, OutTensor};
    use fllm_checkpoint::safetensors::Payload;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    fn meta_json() -> Map<String, Value> {
        let Value::Object(map) = json!({
            "glm5-next.block_count": 5,
            "glm5-next.nextn_predict_layers": 1,
            "glm5-next.hyper_connection.count": 4,
            "glm5-next.hyper_connection.sinkhorn_iters": 16,
            "glm5-next.hyper_connection.eps": 1e-5,
            "glm5-next.ssm.conv_kernel": 4,
            "glm5-next.kda.head_count": 2,
            "glm5-next.kda.head_dim": 8,
            "glm5-next.attention.indexer.head_count": 2,
            "glm5-next.attention.indexer.key_length": 8,
            "glm5-next.attention.indexer.top_k": 16,
            "glm5-next.attention.indexer.kpool": 4,
            "glm5-next.attention.indexer.kpool_always_tail": false,
            "glm5-next.expert_swiglu_clamp": 7.0
        }) else {
            unreachable!()
        };
        map
    }

    fn tensor_info(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_owned(),
            dimensions: vec![16, 16],
            type_id: 0,
            data_offset: 0,
            payload_len: 16 * 16 * 4,
        }
    }

    #[test]
    fn metadata_is_translated_and_kda_flags_are_inferred() {
        let tensors = vec![
            tensor_info("blk.0.ssm_conv1d.weight"),
            tensor_info("blk.1.ssm_a"),
            tensor_info("blk.2.ssm_beta.weight"),
            tensor_info("blk.3.attn_kv_a_mqa.weight"),
            tensor_info("blk.4.attn_kv_a_mqa.weight"),
        ];
        let (keys, kda_flags, layers, kda_dims) = translated(&meta_json(), &tensors).unwrap();
        assert_eq!(layers, 4);
        assert_eq!(kda_flags, vec![true, true, true, false]);
        assert_eq!(kda_dims, Some((2, 8)));
        let get = |k: &str| keys.iter().find(|(n, _)| n == &format!("kimi-linear.{k}")).map(|(_, v)| v.clone());
        assert_eq!(get("block_count"), Some(json!(4)));
        assert_eq!(get("hc_mult"), Some(json!(4)));
        assert_eq!(get("hc_sinkhorn_iters"), Some(json!(16)));
        assert_eq!(get("hc_head_mean"), Some(json!(true)));
        assert_eq!(get("kda.layer_flags"), Some(json!([true, true, true, false])));
        assert_eq!(get("ssm.conv_kernel"), Some(json!(4)));
        assert_eq!(get("kda.head_count"), Some(json!(2)));
        assert_eq!(get("kda.head_dim"), Some(json!(8)));
        assert_eq!(get("rope.dimension_count"), Some(json!(0)));
        assert_eq!(get("attention.mla_no_rope"), Some(json!(true)));
        assert_eq!(get("attention.indexer.kpool"), Some(json!(4)));
        assert_eq!(get("attention.indexer.kpool_always_tail"), Some(json!(false)));
        assert_eq!(get("expert_swiglu_clamp"), Some(json!(7.0)));
    }

    #[test]
    fn inconsistent_files_are_refused() {
        let tensors = vec![tensor_info("blk.0.ssm_a"), tensor_info("blk.3.attn_kv_a_mqa.weight")];
        let mut m = meta_json();
        m.insert("glm5-next.nextn_predict_layers".into(), json!(5));
        assert!(translated(&m, &tensors).is_err());

        let mut m = meta_json();
        m.remove("glm5-next.hyper_connection.count");
        assert!(translated(&m, &tensors).is_err());

        let mut m = meta_json();
        m.insert("glm5-next.ssm.conv_kernel".into(), json!(1));
        assert!(translated(&m, &tensors).is_err());

        let mut m = meta_json();
        m.insert("glm5-next.attention.indexer.top_k".into(), json!(2));
        assert!(translated(&m, &tensors).is_err());
    }

    fn f32_tensor(name: &str, dims: Vec<u64>, data: &[f32]) -> OutTensor {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let data_len = bytes.len() as u64;
        OutTensor {
            name: name.to_owned(),
            dims,
            source_dtype: "F32".into(),
            dtype: "F32".into(),
            data: Payload::Owned(Arc::new(Vec::new()), 0..0),
            packed: None,
            data_len,
            source_for_output: None,
            add: 0.0,
            owned: Some(bytes),
            parts: Vec::new(),
        }
    }

    fn write_test_gguf(meta: Vec<(String, Meta)>, tensors: Vec<OutTensor>) -> MappedGguf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("fllm_glm5_test_{}_{}.gguf", std::process::id(), id));
        {
            let mut file = std::fs::File::create(&path).unwrap();
            write_to(&mut file, &meta, &tensors).unwrap();
        }
        let mapped = MappedGguf::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        mapped
    }

    #[test]
    fn view_translates_synthetic_gguf_and_parses_model_config() {
        let hidden = 32usize;
        let hc = 4usize;
        let hc_out = (2 + hc) * hc;
        let kda_heads = 2usize;
        let kda_dim = 8usize;
        let kda_w = kda_heads * kda_dim;
        let meta = vec![
            ("general.architecture".into(), Meta::Str("glm5-next".into())),
            ("glm5-next.embedding_length".into(), Meta::U32(hidden as u32)),
            ("glm5-next.block_count".into(), Meta::U32(3)),
            ("glm5-next.nextn_predict_layers".into(), Meta::U32(1)),
            ("glm5-next.feed_forward_length".into(), Meta::U32(64)),
            ("glm5-next.attention.head_count".into(), Meta::U32(2)),
            ("glm5-next.attention.head_count_kv".into(), Meta::U32(2)),
            ("glm5-next.attention.key_length".into(), Meta::U32(16)),
            ("glm5-next.attention.value_length".into(), Meta::U32(16)),
            ("glm5-next.attention.q_lora_rank".into(), Meta::U32(8)),
            ("glm5-next.attention.kv_lora_rank".into(), Meta::U32(8)),
            ("glm5-next.rope.dimension_count".into(), Meta::U32(8)),
            ("glm5-next.hyper_connection.count".into(), Meta::U32(hc as u32)),
            ("glm5-next.hyper_connection.sinkhorn_iters".into(), Meta::U32(12)),
            ("glm5-next.hyper_connection.eps".into(), Meta::F32(1e-5)),
            ("glm5-next.ssm.conv_kernel".into(), Meta::U32(4)),
            ("glm5-next.kda.head_count".into(), Meta::U32(kda_heads as u32)),
            ("glm5-next.kda.head_dim".into(), Meta::U32(kda_dim as u32)),
            ("glm5-next.kda.gate_lower_bound".into(), Meta::F32(-5.0)),
            ("glm5-next.attention.indexer.head_count".into(), Meta::U32(2)),
            ("glm5-next.attention.indexer.key_length".into(), Meta::U32(8)),
            ("glm5-next.attention.indexer.top_k".into(), Meta::U32(8)),
            ("glm5-next.attention.indexer.kpool".into(), Meta::U32(4)),
            ("glm5-next.attention.indexer.kpool_always_tail".into(), Meta::Bool(false)),
            ("glm5-next.expert_count".into(), Meta::U32(4)),
            ("glm5-next.expert_used_count".into(), Meta::U32(2)),
            ("glm5-next.expert_feed_forward_length".into(), Meta::U32(16)),
            ("glm5-next.expert_swiglu_clamp".into(), Meta::F32(7.5)),
        ];

        let mut qkv = Vec::with_capacity(3 * kda_w * hidden);
        qkv.extend(vec![1.0f32; kda_w * hidden]);
        qkv.extend(vec![2.0f32; kda_w * hidden]);
        qkv.extend(vec![3.0f32; kda_w * hidden]);
        let mut conv = Vec::with_capacity(3 * kda_w * 4);
        conv.extend(vec![4.0f32; kda_w * 4]);
        conv.extend(vec![5.0f32; kda_w * 4]);
        conv.extend(vec![6.0f32; kda_w * 4]);
        let mut fg = Vec::with_capacity(2 * kda_dim * hidden);
        fg.extend(vec![7.0f32; kda_dim * hidden]);
        fg.extend(vec![8.0f32; kda_dim * hidden]);

        let tensors = vec![
            f32_tensor("token_embd.weight", vec![hidden as u64, 16], &vec![0.0; hidden * 16]),
            // Layer 0 (KDA with llama.cpp style names and fused matrices)
            f32_tensor("blk.0.hc_attn.fn.weight", vec![(hc * hidden) as u64, hc_out as u64], &vec![0.1; hc * hidden * hc_out]),
            f32_tensor("blk.0.hc_attn_fn.bias", vec![hc_out as u64], &vec![0.2; hc_out]),
            f32_tensor("blk.0.hc_attn.scale.weight", vec![3], &[1.0, 2.0, 3.0]),
            f32_tensor("blk.0.hc_ffn_fn.weight", vec![(hc * hidden) as u64, hc_out as u64], &vec![0.1; hc * hidden * hc_out]),
            f32_tensor("blk.0.hc_ffn_base.weight", vec![hc_out as u64], &vec![0.2; hc_out]),
            f32_tensor("blk.0.hc_ffn_scale.weight", vec![3], &[1.0, 2.0, 3.0]),
            f32_tensor("blk.0.attn_qkv.weight", vec![hidden as u64, (3 * kda_w) as u64], &qkv),
            f32_tensor("blk.0.ssm_conv1d.weight", vec![4, (3 * kda_w) as u64], &conv),
            f32_tensor("blk.0.ssm_out.weight", vec![kda_w as u64, hidden as u64], &vec![0.5; kda_w * hidden]),
            f32_tensor("blk.0.ssm_fg_a.weight", vec![hidden as u64, (2 * kda_dim) as u64], &fg),
            f32_tensor("blk.0.ssm_norm.weight", vec![kda_dim as u64], &vec![1.0; kda_dim]),
            f32_tensor("blk.0.ssm_dt_bias", vec![kda_w as u64], &vec![0.0; kda_w]),
            f32_tensor("blk.0.ssm_a.weight", vec![kda_heads as u64], &vec![1.0; kda_heads]),
            // Layer 1 (MLA + Indexer with llama.cpp aliases)
            f32_tensor("blk.1.hc_attn_fn.weight", vec![(hc * hidden) as u64, hc_out as u64], &vec![0.1; hc * hidden * hc_out]),
            f32_tensor("blk.1.hc_attn_base.weight", vec![hc_out as u64], &vec![0.2; hc_out]),
            f32_tensor("blk.1.hc_attn_scale.weight", vec![3], &[1.0, 2.0, 3.0]),
            f32_tensor("blk.1.hc_ffn_fn.weight", vec![(hc * hidden) as u64, hc_out as u64], &vec![0.1; hc * hidden * hc_out]),
            f32_tensor("blk.1.hc_ffn_base.weight", vec![hc_out as u64], &vec![0.2; hc_out]),
            f32_tensor("blk.1.hc_ffn_scale.weight", vec![3], &[1.0, 2.0, 3.0]),
            f32_tensor("blk.1.attn_q_a.weight", vec![hidden as u64, 8], &vec![0.0; hidden * 8]),
            f32_tensor("blk.1.attn_kv_a_mqa.weight", vec![hidden as u64, 8], &vec![0.0; hidden * 8]),
            f32_tensor("blk.1.indexer.wq_b.weight", vec![8, 16], &vec![0.0; 8 * 16]),
            f32_tensor("blk.1.indexer.wk.weight", vec![hidden as u64, 8], &vec![0.0; hidden * 8]),
            f32_tensor("blk.1.indexer.weights_proj.weight", vec![hidden as u64, 2], &vec![0.0; hidden * 2]),
            f32_tensor("blk.1.indexer.kpool_compress_ape", vec![8, 4], &[0.25; 32]),
            f32_tensor("blk.1.indexer.kpool_compress_gate", vec![hidden as u64, 8], &vec![0.5; hidden * 8]),
        ];

        let mapped = write_test_gguf(meta, tensors);
        let translated_view = view(mapped).unwrap();
        let cfg = ModelConfig::from_gguf(&translated_view).unwrap();
        assert_eq!(cfg.arch, "kimi-linear");
        assert_eq!(cfg.layers, 2);
        let hc_cfg = cfg.hc.as_ref().unwrap();
        assert_eq!((hc_cfg.hc_mult, hc_cfg.iters, hc_cfg.head_mean, hc_cfg.magnitude), (4, 12, true, None));
        let kda_cfg = cfg.hybrid.kda.as_ref().unwrap();
        assert_eq!((&kda_cfg.layers[..], kda_cfg.conv_kernel, kda_cfg.heads, kda_cfg.head_dim, kda_cfg.lower_bound), (&[true, false][..], 4, 2, 8, Some(-5.0)));
        let mla_cfg = cfg.mla.as_ref().unwrap();
        assert_eq!((mla_cfg.nope, mla_cfg.rope, mla_cfg.rotary, mla_cfg.kv_lora, mla_cfg.q_lora, mla_cfg.value), (16, 0, false, 8, Some(8), 16));
        let idx_cfg = cfg.indexer.as_ref().unwrap();
        assert_eq!((idx_cfg.heads, idx_cfg.dim, idx_cfg.topk, idx_cfg.kpool, idx_cfg.always_tail), (2, 8, 8, 4, false));
        let moe_cfg = cfg.moe.as_ref().unwrap();
        assert_eq!(moe_cfg.swiglu_clamp, Some(7.5));
        assert_eq!(moe_cfg.routed_clamp(0), Some(7.5));

        // Verify row-sliced and aliased tensors resolve with the expected payloads.
        let load = |name: &str| crate::compute::Tensor::load(&translated_view, name).unwrap().data;
        assert_eq!(load("blk.0.attn_q.weight"), vec![1.0; kda_w * hidden]);
        assert_eq!(load("blk.0.attn_k.weight"), vec![2.0; kda_w * hidden]);
        assert_eq!(load("blk.0.attn_v.weight"), vec![3.0; kda_w * hidden]);
        assert_eq!(load("blk.0.ssm_conv1d_q.weight"), vec![4.0; kda_w * 4]);
        assert_eq!(load("blk.0.ssm_conv1d_k.weight"), vec![5.0; kda_w * 4]);
        assert_eq!(load("blk.0.ssm_conv1d_v.weight"), vec![6.0; kda_w * 4]);
        assert_eq!(load("blk.0.ssm_f_a.weight"), vec![7.0; kda_dim * hidden]);
        assert_eq!(load("blk.0.ssm_g_a.weight"), vec![8.0; kda_dim * hidden]);
        assert_eq!(load("blk.0.attn_output.weight"), vec![0.5; kda_w * hidden]);
        assert_eq!(load("blk.0.hc_attn_base.weight"), vec![0.2; hc_out]);
        assert_eq!(load("blk.0.hc_attn_scale.weight"), vec![1.0, 2.0, 3.0]);
        assert_eq!(load("blk.0.ssm_o_norm.weight"), vec![1.0; kda_dim]);
        assert_eq!(load("blk.0.ssm_dt.bias"), vec![0.0; kda_w]);
        assert_eq!(load("blk.0.ssm_a"), vec![1.0; kda_heads]);
        assert_eq!(load("blk.1.indexer.pool_ape.weight"), vec![0.25; 32]);
        assert_eq!(load("blk.1.indexer.pool_gate.weight"), vec![0.5; hidden * 8]);
        assert!(translated_view.tensor_bytes("blk.1.indexer.attn_q_b.weight").is_ok());
        assert!(translated_view.tensor_bytes("blk.1.indexer.attn_k.weight").is_ok());
        assert!(translated_view.tensor_bytes("blk.1.indexer.proj.weight").is_ok());
    }
}
