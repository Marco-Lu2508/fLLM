//! Qwen3.8-Flash-Next as llama.cpp writes it (`general.architecture = qwen4exp`).
//!
//! The model is the Qwen3.5 MoE (`qwen35moe`) with `hc` residual streams, a token indexer on the attention layers and a hashed n-gram
//! embedding on one delta layer. llama.cpp (`conversion/qwen4exp.py`, `src/models/qwen4exp.cpp`) and fLLM's own converter
//! (`rust/checkpoint/src/qwen4_exp.rs`) describe the same network with different names, so the file is presented to the engine as an
//! fLLM-style `qwen35moe` file: a metadata and tensor-name view over the same mapped bytes, nothing is copied.
//!
//! | llama.cpp (`qwen4exp.`)                                  | engine view (`qwen35moe.`)                   |
//! |----------------------------------------------------------|----------------------------------------------|
//! | `hyper_connection.count`, `.low_rank`                    | `hc.count`, `hc.lowrank`                     |
//! | `attention.indexer.head_count`, `.key_length`, `.top_k`  | `indexer.heads`, `.head_dim`, `.budget`      |
//! | `attention.compress_ratios` (per layer, 0: no indexer)   | `indexer.compress_ratio`                     |
//! | `ple.layers` (0-based list)                              | `ple.layer_flags` (one flag per layer)       |
//! | `ple.ngram_size`, `heads_per_ngram`, `conv_kernel`       | `ple.ngram_size`, `heads_per_ngram`, `conv_kernel` |
//! | `ple.eos_token_id`                                       | `ple.eos`                                    |
//! | `embedding_length_per_layer_input` (row width of a head) | `ple.embed_dim` (heads x row width)          |
//! | `ple.layer_multipliers`, `head_offsets`, `head_vocab_sizes` | `ple.multipliers`, `head_offsets`, `head_sizes` |
//!
//! Tensors: `per_layer_token_embd.weight` (one table for the file) is `blk.N.ple_embd.weight` of the n-gram layer, `ple_conv1d` is
//! `ple_conv`; the fused weights the converter writes when the checkpoint is not pre-quantized (`hc_*_down_inject`, `attn_qkv_gate`,
//! `ssm_beta_alpha`) are read as the pairs they were stacked from (`with_row_slices`); `indexer.q_proj` / `k_proj` are read as they are.
//!
//! What the layout needs besides names, and where it is handled:
//! * the gated delta layers store the value heads tiled and the attention layers store `[q_h, gate_h]` per head, like every llama.cpp
//!   Qwen3.5 file: both follow from the architecture name `qwen35moe`;
//! * the gate of the delta output norm is `sigmoid(z)` (`delta.gate_sigmoid`), not `silu(z)`;
//! * the converter folds `1 + w` into every norm of the hyper-connection mixers, the indexer and the n-gram layer, the same fold fLLM's
//!   converter does, so no norm is touched;
//! * the model has no `output_norm`: the engine uses a norm of ones where the file has none;
//! * the routed experts are normalised (llama.cpp passes `norm_w = true` for this architecture).
use fllm_gguf_probe::MappedGguf;
use serde_json::{Map, Value};

const FROM: &str = "qwen4exp";
const TO: &str = "qwen35moe";

fn uint(meta: &Map<String, Value>, name: &str) -> Option<u64> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_u64)
}

fn list<'a>(meta: &'a Map<String, Value>, name: &str) -> Option<&'a Vec<Value>> {
    meta.get(&format!("{FROM}.{name}")).and_then(Value::as_array)
}

fn need(meta: &Map<String, Value>, name: &str) -> Result<u64, String> {
    uint(meta, name).filter(|&v| v > 0).ok_or_else(|| format!("qwen4exp GGUF needs {FROM}.{name}"))
}

/// The translated keys, the layers with an n-gram embedding and the number of trunk layers.
type Translated = (Vec<(String, Value)>, Vec<usize>, usize);

/// Metadata of the engine view: the translated keys of the table above (under the `qwen35moe.` prefix).
fn translated(meta: &Map<String, Value>) -> Result<Translated, String> {
    let all_layers = need(meta, "block_count")? as usize;
    // the draft block of a file with a multi-token-prediction head is not part of the trunk
    let layers = all_layers.saturating_sub(uint(meta, "nextn_predict_layers").unwrap_or(0) as usize);
    if layers == 0 {
        return Err("qwen4exp GGUF without layers".into());
    }
    let mut out: Vec<(String, Value)> = Vec::new();
    let mut put = |key: &str, value: Value| out.push((format!("{TO}.{key}"), value));
    put("block_count", Value::from(layers as u64));
    put("hc.count", Value::from(need(meta, "hyper_connection.count")?));
    put("hc.lowrank", Value::from(need(meta, "hyper_connection.low_rank")?));
    let budget = need(meta, "attention.indexer.top_k")?;
    let ratios: Vec<u64> = list(meta, "attention.compress_ratios").map(|a| a.iter().take(layers).map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default();
    // no layer with a block size: the trunk has no attention layer that selects blocks (a slice of the first layers of a model)
    if let Some(ratio) = ratios.iter().copied().find(|&r| r > 0) {
        if ratios.iter().any(|&r| r > 0 && r != ratio) {
            return Err("the attention layers of a qwen4exp GGUF must share one compress ratio".into());
        }
        if ratio < 2 || !budget.is_multiple_of(ratio) {
            return Err(format!("qwen4exp indexer budget {budget} must be a multiple of the compress ratio {ratio} (above 1)"));
        }
        put("indexer.heads", Value::from(need(meta, "attention.indexer.head_count")?));
        put("indexer.head_dim", Value::from(need(meta, "attention.indexer.key_length")?));
        put("indexer.budget", Value::from(budget));
        put("indexer.compress_ratio", Value::from(ratio));
    }
    let ple_layers: Vec<usize> = list(meta, "ple.layers").map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()).unwrap_or_default();
    if let Some(bad) = ple_layers.iter().find(|&&l| l >= layers) {
        return Err(format!("qwen4exp ple.layers holds {bad}, outside 0..{layers}"));
    }
    if ple_layers.len() > 1 {
        return Err("a qwen4exp GGUF carries the n-gram hash constants of one layer; several n-gram layers are not supported".into());
    }
    if !ple_layers.is_empty() {
        let (ngram, per_ngram, row) = (need(meta, "ple.ngram_size")?, need(meta, "ple.heads_per_ngram")?, need(meta, "embedding_length_per_layer_input")?);
        if ngram < 2 {
            return Err("qwen4exp ple.ngram_size must be at least 2".into());
        }
        let heads = ((ngram - 1) * per_ngram) as usize;
        let table = |name: &str, min: usize| -> Result<Value, String> {
            let values = list(meta, name).ok_or_else(|| format!("qwen4exp GGUF needs {FROM}.{name}"))?;
            if values.len() < min || values.iter().any(|v| v.as_u64().is_none() && v.as_i64().is_none()) {
                return Err(format!("{FROM}.{name} needs {min} integers"));
            }
            Ok(Value::Array(values.clone()))
        };
        put("ple.layer_flags", Value::Array((0..layers).map(|l| Value::Bool(ple_layers.contains(&l))).collect()));
        put("ple.ngram_size", Value::from(ngram));
        put("ple.heads_per_ngram", Value::from(per_ngram));
        put("ple.conv_kernel", Value::from(need(meta, "ple.conv_kernel")?));
        put("ple.eos", Value::from(uint(meta, "ple.eos_token_id").ok_or("qwen4exp GGUF needs ple.eos_token_id")?));
        put("ple.embed_dim", Value::from(row * heads as u64));
        put("ple.multipliers", table("ple.layer_multipliers", ngram as usize)?);
        put("ple.head_offsets", table("ple.head_offsets", heads)?);
        put("ple.head_sizes", table("ple.head_vocab_sizes", heads)?);
    }
    // the gate of the delta output norm
    put("delta.gate_sigmoid", Value::Bool(true));
    Ok((out, ple_layers, layers))
}

/// The engine view of a llama.cpp `qwen4exp` file.
pub(crate) fn view(mapping: MappedGguf) -> Result<MappedGguf, String> {
    let meta = mapping.index().metadata.clone();
    let (keys, ple_layers, layers) = translated(&meta)?;
    let mut view = mapping.with_architecture_alias(FROM, TO);
    for (key, value) in keys {
        view = view.with_metadata(&key, value);
    }
    // llama.cpp normalises the weights of the routed experts for this architecture without storing the flag
    view = view.with_metadata_default(&format!("{TO}.expert_weights_norm"), Value::Bool(true));
    let has = |view: &MappedGguf, name: &str| view.index().tensors.iter().any(|t| t.name == name);
    let dims = |view: &MappedGguf, name: &str| view.index().tensors.iter().find(|t| t.name == name).map(|t| t.dimensions.clone());
    let lowrank = need(&meta, "hyper_connection.low_rank")? as usize;
    let hc = need(&meta, "hyper_connection.count")? as usize;
    let inner = need(&meta, "ssm.inner_size")? as usize;
    let v_heads = need(&meta, "ssm.time_step_rank")? as usize;
    let mut aliases: Vec<(String, String)> = Vec::new();
    let mut slices: Vec<(String, String, usize, usize)> = Vec::new();
    for l in 0..layers {
        let p = format!("blk.{l}");
        if has(&view, &format!("{p}.ple_conv1d.weight")) {
            aliases.push((format!("{p}.ple_conv.weight"), format!("{p}.ple_conv1d.weight")));
        }
        if ple_layers.contains(&l) && has(&view, "per_layer_token_embd.weight") {
            aliases.push((format!("{p}.ple_embd.weight"), "per_layer_token_embd.weight".into()));
        }
        for which in ["attn", "ffn"] {
            let fused = format!("{p}.hc_{which}_down_inject.weight");
            if has(&view, &fused) {
                slices.push((format!("{p}.hc_{which}_down.weight"), fused.clone(), 0, lowrank));
                slices.push((format!("{p}.hc_{which}_inject.weight"), fused, lowrank, hc));
            }
        }
        let fused = format!("{p}.attn_qkv_gate.weight");
        if let Some(d) = dims(&view, &fused) {
            let rows = *d.get(1).ok_or("attn_qkv_gate is not a matrix")? as usize;
            let qkv = rows.checked_sub(inner).filter(|&r| r > 0).ok_or("attn_qkv_gate is narrower than the value width")?;
            slices.push((format!("{p}.attn_qkv.weight"), fused.clone(), 0, qkv));
            slices.push((format!("{p}.attn_gate.weight"), fused, qkv, inner));
        }
        let fused = format!("{p}.ssm_beta_alpha.weight");
        if has(&view, &fused) {
            slices.push((format!("{p}.ssm_beta.weight"), fused.clone(), 0, v_heads));
            slices.push((format!("{p}.ssm_alpha.weight"), fused, v_heads, v_heads));
        }
    }
    view = view.with_tensor_aliases(&aliases)?;
    view = view.with_row_slices(&slices)?;
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta() -> Map<String, Value> {
        let Value::Object(map) = json!({
            "qwen4exp.block_count": 8, "qwen4exp.hyper_connection.count": 4, "qwen4exp.hyper_connection.low_rank": 16,
            "qwen4exp.attention.indexer.head_count": 4, "qwen4exp.attention.indexer.key_length": 32, "qwen4exp.attention.indexer.top_k": 64,
            "qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0, 0, 4],
            "qwen4exp.ple.layers": [1], "qwen4exp.ple.ngram_size": 3, "qwen4exp.ple.heads_per_ngram": 2, "qwen4exp.ple.conv_kernel": 4,
            "qwen4exp.ple.eos_token_id": 9, "qwen4exp.embedding_length_per_layer_input": 8,
            "qwen4exp.ple.layer_multipliers": [3, 5, 7], "qwen4exp.ple.head_offsets": [0, 11, 22, 33], "qwen4exp.ple.head_vocab_sizes": [11, 11, 11, 11],
            "qwen4exp.ssm.inner_size": 64, "qwen4exp.ssm.time_step_rank": 4,
        }) else { unreachable!() };
        map
    }

    #[test]
    fn metadata_is_translated() {
        let (keys, ple, layers) = translated(&meta()).unwrap();
        assert_eq!((ple, layers), (vec![1], 8));
        let get = |k: &str| keys.iter().find(|(n, _)| n == &format!("qwen35moe.{k}")).map(|(_, v)| v.clone());
        assert_eq!(get("hc.count"), Some(json!(4)));
        assert_eq!(get("indexer.compress_ratio"), Some(json!(4)));
        assert_eq!(get("indexer.budget"), Some(json!(64)));
        assert_eq!(get("ple.embed_dim"), Some(json!(32)));
        assert_eq!(get("ple.layer_flags"), Some(json!([false, true, false, false, false, false, false, false])));
        assert_eq!(get("ple.head_sizes"), Some(json!([11, 11, 11, 11])));
        assert_eq!(get("delta.gate_sigmoid"), Some(json!(true)));
    }

    #[test]
    fn a_draft_block_is_not_a_trunk_layer() {
        let mut m = meta();
        m.insert("qwen4exp.block_count".into(), json!(9));
        m.insert("qwen4exp.nextn_predict_layers".into(), json!(1));
        assert_eq!(translated(&m).unwrap().2, 8);
    }

    #[test]
    fn inconsistent_files_are_refused() {
        let mut m = meta();
        m.insert("qwen4exp.attention.indexer.top_k".into(), json!(63));
        assert!(translated(&m).is_err());
        let mut m = meta();
        m.insert("qwen4exp.ple.layers".into(), json!([1, 2]));
        assert!(translated(&m).is_err());
        let mut m = meta();
        m.insert("qwen4exp.attention.compress_ratios".into(), json!([0, 0, 0, 4, 0, 0, 0, 8]));
        assert!(translated(&m).is_err());
    }
}
