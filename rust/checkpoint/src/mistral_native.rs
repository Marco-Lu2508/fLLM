//! Mistral's own checkpoint layout (`params.json` + `consolidated*.safetensors`), used by repositories that ship no
//! `config.json` or whose Hugging Face copy is incomplete.
//!
//! * `params.json` becomes an equivalent Hugging Face `config.json` (`mistral`, `ministral3` or `mixtral`).
//! * Tensor names (`layers.N.attention.wq.weight`, `layers.N.feed_forward.w1.weight`, ...) are renamed to the
//!   Hugging Face names, so the regular plan, shape checks and quantization paths apply unchanged. FP8 scales
//!   (`qscale_weight`, `qscale_act`) become `weight_scale` and `input_scale`.
//! * Q/K rows stay in the original interleaved rotary order (the Hugging Face copy is permuted to split-half order);
//!   `native_row_permutation` accounts for that.
//!
//! Vision encoder tensors are dropped, as for the Hugging Face layout. Multi-head latent attention variants
//! (Mistral Large 3, `q_lora_rank` set) are rejected with a precise message.
use serde_json::{json, Map, Value};
use std::fs;
use std::path::Path;

/// True for a checkpoint directory in Mistral's layout: `params.json` and no `config.json`.
pub(crate) fn is_native(root: &Path) -> bool {
    !root.join("config.json").is_file() && root.join("params.json").is_file()
}

/// `config.json` of the directory, or the one derived from `params.json` for the native layout.
pub(crate) fn read_config(root: &Path) -> Result<Value, String> {
    let config = root.join("config.json");
    if config.is_file() {
        return serde_json::from_slice(&fs::read(&config).map_err(|e| format!("read config.json: {e}"))?)
            .map_err(|e| format!("invalid config.json: {e}"));
    }
    let params = root.join("params.json");
    if !params.is_file() {
        return Err(format!("read config.json: {} has neither config.json nor params.json", root.display()));
    }
    let params: Value = serde_json::from_slice(&fs::read(&params).map_err(|e| format!("read params.json: {e}"))?)
        .map_err(|e| format!("invalid params.json: {e}"))?;
    config_from_params(&params)
}

fn uint(params: &Value, key: &str) -> Option<u64> {
    params.get(key).and_then(Value::as_u64)
}

fn float(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(Value::as_f64)
}

/// Hugging Face `config.json` equivalent of a Mistral `params.json`.
pub(crate) fn config_from_params(params: &Value) -> Result<Value, String> {
    let need = |key: &str| uint(params, key).ok_or_else(|| format!("params.json has no {key}"));
    let dim = need("dim")?;
    let layers = need("n_layers")?;
    let heads = need("n_heads")?;
    let kv_heads = uint(params, "n_kv_heads").unwrap_or(heads);
    let head_dim = uint(params, "head_dim").unwrap_or(dim / heads.max(1));
    let vocab = need("vocab_size")?;
    if params.get("q_lora_rank").is_some_and(|v| !v.is_null()) || params.get("kv_lora_rank").is_some_and(|v| !v.is_null()) {
        return Err("params.json describes multi-head latent attention (Mistral Large 3 layout); only the dense and Mixtral-style \
                    native layouts are supported, use the Hugging Face copy of this model"
            .into());
    }
    let mut cfg = Map::new();
    let moe = params.get("moe").filter(|v| v.is_object());
    let ministral = params.get("llama_4_scaling").is_some_and(|v| v.is_object());
    let model_type = if moe.is_some() {
        "mixtral"
    } else if ministral {
        "ministral3"
    } else {
        "mistral"
    };
    cfg.insert("model_type".into(), json!(model_type));
    cfg.insert(
        "architectures".into(),
        json!([match model_type {
            "mixtral" => "MixtralForCausalLM",
            "ministral3" => "Ministral3ForCausalLM",
            _ => "MistralForCausalLM",
        }]),
    );
    cfg.insert("hidden_size".into(), json!(dim));
    cfg.insert("num_hidden_layers".into(), json!(layers));
    cfg.insert("num_attention_heads".into(), json!(heads));
    cfg.insert("num_key_value_heads".into(), json!(kv_heads));
    cfg.insert("head_dim".into(), json!(head_dim));
    cfg.insert("vocab_size".into(), json!(vocab));
    cfg.insert("intermediate_size".into(), json!(uint(params, "hidden_dim").ok_or("params.json has no hidden_dim")?));
    cfg.insert("rms_norm_eps".into(), json!(float(params, "norm_eps").unwrap_or(1e-5)));
    cfg.insert("hidden_act".into(), json!("silu"));
    let max_pos = ["max_position_embeddings", "max_seq_len", "max_seq_length"].iter().find_map(|k| uint(params, k)).unwrap_or(32768);
    cfg.insert("max_position_embeddings".into(), json!(max_pos));
    let theta = float(params, "rope_theta").unwrap_or(10_000.0);
    cfg.insert("rope_theta".into(), json!(theta));
    cfg.insert(
        "tie_word_embeddings".into(),
        json!(params.get("tied_embeddings").or_else(|| params.get("tie_embeddings")).and_then(Value::as_bool).unwrap_or(false)),
    );
    if let Some(w) = uint(params, "sliding_window") {
        cfg.insert("sliding_window".into(), json!(w));
    }
    if let Some(yarn) = params.get("yarn").filter(|v| v.is_object()) {
        let mut rope = Map::new();
        rope.insert("rope_type".into(), json!("yarn"));
        rope.insert("rope_theta".into(), json!(theta));
        rope.insert("factor".into(), json!(float(yarn, "factor").ok_or("params.json yarn has no factor")?));
        rope.insert(
            "original_max_position_embeddings".into(),
            json!(uint(yarn, "original_max_position_embeddings").ok_or("params.json yarn has no original_max_position_embeddings")?),
        );
        rope.insert("beta_fast".into(), json!(float(yarn, "beta").unwrap_or(32.0)));
        rope.insert("beta_slow".into(), json!(float(yarn, "alpha").unwrap_or(1.0)));
        if let Some(scaling) = params.get("llama_4_scaling") {
            if let Some(beta) = float(scaling, "beta") {
                rope.insert("llama_4_scaling_beta".into(), json!(beta));
            }
        }
        cfg.insert("rope_parameters".into(), Value::Object(rope));
    } else if let Some(scaling) = params.get("llama_4_scaling") {
        let mut rope = Map::new();
        rope.insert("rope_type".into(), json!("default"));
        rope.insert("rope_theta".into(), json!(theta));
        if let Some(beta) = float(scaling, "beta") {
            rope.insert("llama_4_scaling_beta".into(), json!(beta));
        }
        if let Some(orig) = uint(scaling, "original_max_position_embeddings") {
            rope.insert("original_max_position_embeddings".into(), json!(orig));
        }
        cfg.insert("rope_parameters".into(), Value::Object(rope));
    }
    if let Some(moe) = moe {
        cfg.insert("num_local_experts".into(), json!(uint(moe, "num_experts").ok_or("params.json moe has no num_experts")?));
        cfg.insert("num_experts_per_tok".into(), json!(uint(moe, "num_experts_per_tok").ok_or("params.json moe has no num_experts_per_tok")?));
    }
    if let Some(q) = params.get("quantization").filter(|v| v.is_object()) {
        match q.get("qformat_weight").and_then(Value::as_str) {
            Some("fp8_e4m3") => {
                cfg.insert("quantization_config".into(), json!({"quant_method": "fp8", "activation_scheme": "static", "fmt": "e4m3"}));
            }
            other => return Err(format!("params.json declares weight quantization {other:?}; only fp8_e4m3 is supported for the native layout")),
        }
    }
    Ok(Value::Object(cfg))
}

fn is_vision(name: &str) -> bool {
    ["vision_encoder.", "vision_language_adapter.", "patch_merger.", "pre_mm_projector_norm.", "audio_", "whisper_encoder."].iter().any(|p| name.starts_with(p))
}

/// Hugging Face name of a native tensor; `Ok(None)` for tensors that are dropped (vision, audio, cached rotary table).
pub(crate) fn rename(name: &str) -> Result<Option<String>, String> {
    match name {
        "tok_embeddings.weight" => return Ok(Some("model.embed_tokens.weight".into())),
        "norm.weight" => return Ok(Some("model.norm.weight".into())),
        "output.weight" => return Ok(Some("lm_head.weight".into())),
        "rope.freqs" => return Ok(None),
        _ => {}
    }
    if is_vision(name) {
        return Ok(None);
    }
    let unsupported = || format!("unsupported tensor {name:?} in a native Mistral checkpoint");
    let rest = name.strip_prefix("layers.").ok_or_else(unsupported)?;
    let (layer, tail) = rest.split_once('.').ok_or_else(unsupported)?;
    layer.parse::<u64>().map_err(|_| unsupported())?;
    // the quantization scales follow the weight's name
    let scale_suffix = |suffix: &str| -> Option<&'static str> {
        match suffix {
            "weight" => Some("weight"),
            "qscale_weight" => Some("weight_scale"),
            "qscale_act" => Some("input_scale"),
            _ => None,
        }
    };
    let prefix = format!("model.layers.{layer}");
    match tail {
        "attention_norm.weight" => return Ok(Some(format!("{prefix}.input_layernorm.weight"))),
        "ffn_norm.weight" => return Ok(Some(format!("{prefix}.post_attention_layernorm.weight"))),
        "feed_forward.gate.weight" => return Ok(Some(format!("{prefix}.block_sparse_moe.gate.weight"))),
        _ => {}
    }
    let parts: Vec<&str> = tail.split('.').collect();
    let mapped = match parts.as_slice() {
        ["attention", proj @ ("wq" | "wk" | "wv" | "wo"), suffix] => {
            let hf = match *proj {
                "wq" => "q_proj",
                "wk" => "k_proj",
                "wv" => "v_proj",
                _ => "o_proj",
            };
            scale_suffix(suffix).map(|s| format!("{prefix}.self_attn.{hf}.{s}"))
        }
        ["feed_forward", proj @ ("w1" | "w2" | "w3"), suffix] => {
            let hf = match *proj {
                "w1" => "gate_proj",
                "w3" => "up_proj",
                _ => "down_proj",
            };
            scale_suffix(suffix).map(|s| format!("{prefix}.mlp.{hf}.{s}"))
        }
        ["feed_forward", "experts", expert, proj @ ("w1" | "w2" | "w3"), suffix] if expert.parse::<u64>().is_ok() => {
            scale_suffix(suffix).map(|s| format!("{prefix}.block_sparse_moe.experts.{expert}.{proj}.{s}"))
        }
        _ => None,
    };
    mapped.map(Some).ok_or_else(unsupported)
}

/// `row_permutation` for native weights, whose Q/K rows are already in interleaved rotary order: architectures whose
/// GGUF layout is interleaved take the rows as they are, split-half architectures need the inverse of the usual
/// conversion.
pub(crate) fn native_row_permutation(arch: &str, name: &str, heads: u64, kv_heads: u64, head_dim: u64) -> Result<Option<Vec<usize>>, String> {
    if crate::convert::row_permutation(arch, name, heads, kv_heads, head_dim)?.is_some() {
        // the architecture expects interleaved rows: native weights already are
        return Ok(None);
    }
    if !(name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight")) {
        return Ok(None);
    }
    let Some(forward) = crate::convert::row_permutation("llama", name, heads, kv_heads, head_dim)? else {
        return Ok(None);
    };
    let mut inverse = vec![0usize; forward.len()];
    for (output_row, &source_row) in forward.iter().enumerate() {
        inverse[source_row] = output_row;
    }
    Ok(Some(inverse))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_become_a_mistral_config() {
        let params = json!({"dim": 5120, "n_layers": 40, "head_dim": 128, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8, "norm_eps": 1e-5,
            "vocab_size": 131072, "rope_theta": 1e6, "max_position_embeddings": 131072, "vision_encoder": {"hidden_size": 1024}});
        let cfg = config_from_params(&params).unwrap();
        assert_eq!(cfg["model_type"], "mistral");
        assert_eq!(cfg["hidden_size"], 5120);
        assert_eq!(cfg["num_key_value_heads"], 8);
        assert_eq!(cfg["intermediate_size"], 14336);
        assert_eq!(cfg["tie_word_embeddings"], false);
        assert!(cfg.get("quantization_config").is_none());
    }

    #[test]
    fn ministral_fp8_params_carry_yarn_and_scaling() {
        let params = json!({"dim": 4096, "n_layers": 34, "head_dim": 128, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8, "rope_theta": 1e6,
            "norm_eps": 1e-5, "vocab_size": 131072, "tied_embeddings": false, "max_position_embeddings": 262144,
            "llama_4_scaling": {"original_max_position_embeddings": 16384, "beta": 0.1}, "q_lora_rank": null,
            "quantization": {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"},
            "yarn": {"original_max_position_embeddings": 16384, "factor": 16, "apply_scale": false, "beta": 32, "alpha": 1}});
        let cfg = config_from_params(&params).unwrap();
        assert_eq!(cfg["model_type"], "ministral3");
        assert_eq!(cfg["rope_parameters"]["rope_type"], "yarn");
        assert_eq!(cfg["rope_parameters"]["llama_4_scaling_beta"], 0.1);
        assert_eq!(cfg["rope_parameters"]["beta_fast"], 32.0);
        assert_eq!(cfg["quantization_config"]["quant_method"], "fp8");
    }

    #[test]
    fn moe_and_mla_params() {
        let moe = json!({"dim": 64, "n_layers": 2, "hidden_dim": 128, "n_heads": 4, "vocab_size": 100, "moe": {"num_experts": 8, "num_experts_per_tok": 2}});
        let cfg = config_from_params(&moe).unwrap();
        assert_eq!(cfg["model_type"], "mixtral");
        assert_eq!(cfg["num_local_experts"], 8);
        let mla = json!({"dim": 64, "n_layers": 2, "hidden_dim": 128, "n_heads": 4, "vocab_size": 100, "q_lora_rank": 32});
        assert!(config_from_params(&mla).unwrap_err().contains("latent attention"));
        let int4 = json!({"dim": 64, "n_layers": 2, "hidden_dim": 128, "n_heads": 4, "vocab_size": 100, "quantization": {"qformat_weight": "int4"}});
        assert!(config_from_params(&int4).unwrap_err().contains("only fp8_e4m3"));
    }

    #[test]
    fn tensor_names_map_to_the_hugging_face_layout() {
        let m = |n: &str| rename(n).unwrap();
        assert_eq!(m("tok_embeddings.weight").unwrap(), "model.embed_tokens.weight");
        assert_eq!(m("output.weight").unwrap(), "lm_head.weight");
        assert_eq!(m("layers.3.attention.wq.weight").unwrap(), "model.layers.3.self_attn.q_proj.weight");
        assert_eq!(m("layers.3.attention.wo.qscale_weight").unwrap(), "model.layers.3.self_attn.o_proj.weight_scale");
        assert_eq!(m("layers.3.feed_forward.w3.qscale_act").unwrap(), "model.layers.3.mlp.up_proj.input_scale");
        assert_eq!(m("layers.0.feed_forward.w2.weight").unwrap(), "model.layers.0.mlp.down_proj.weight");
        assert_eq!(m("layers.1.attention_norm.weight").unwrap(), "model.layers.1.input_layernorm.weight");
        assert_eq!(m("layers.1.ffn_norm.weight").unwrap(), "model.layers.1.post_attention_layernorm.weight");
        assert_eq!(m("layers.1.feed_forward.gate.weight").unwrap(), "model.layers.1.block_sparse_moe.gate.weight");
        assert_eq!(m("layers.1.feed_forward.experts.5.w1.weight").unwrap(), "model.layers.1.block_sparse_moe.experts.5.w1.weight");
        assert!(m("vision_encoder.transformer.layers.0.attention.wq.weight").is_none());
        assert!(m("rope.freqs").is_none());
        assert!(rename("layers.1.attention.wq.bias").unwrap_err().contains("unsupported tensor"));
        assert!(rename("lm_head.weight").is_err());
    }

    #[test]
    fn native_rows_need_no_permutation_for_llama_and_the_inverse_elsewhere() {
        assert_eq!(native_row_permutation("llama", "blk.0.attn_q.weight", 1, 1, 4).unwrap(), None);
        // a split-half architecture receives the inverse of the llama conversion
        assert_eq!(native_row_permutation("ministral3", "blk.0.attn_q.weight", 1, 1, 4).unwrap(), Some(vec![0, 2, 1, 3]));
        assert_eq!(native_row_permutation("ministral3", "blk.0.attn_v.weight", 1, 1, 4).unwrap(), None);
        let order = native_row_permutation("ministral3", "blk.0.attn_k.weight", 1, 1, 8).unwrap().unwrap();
        let forward = crate::convert::row_permutation("llama", "blk.0.attn_k.weight", 1, 1, 8).unwrap().unwrap();
        // applying the llama conversion after the inverse restores the original row order
        let composed: Vec<usize> = (0..8).map(|i| order[forward[i]]).collect();
        assert_eq!(composed, (0..8).collect::<Vec<_>>());
    }
}
