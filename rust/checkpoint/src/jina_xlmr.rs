//! Jina XLM-RoBERTa "flash implementation" checkpoints (jina-embeddings-v3, jina-reranker-v2)
//! exported in the BERT GGUF layout.
//!
//! They are post-norm XLM-R encoders with fused `mixer.Wqkv` / `mixer.out_proj` / `mlp.fc1` /
//! `mlp.fc2` tensors and either absolute positions (reranker v2) or rotary positions (embeddings
//! v3). jina-embeddings-v3 additionally carries one LoRA adapter set per task as torch
//! parametrizations (`...parametrizations.weight.original` plus `...0.lora_A` / `lora_B`); the
//! adapters are kept separate and written as `*.lora_a` / `*.lora_b` (see `encoder_lora.rs`).
use crate::{
    bert,
    gguf::{Meta, OutTensor},
    safetensors::{self, Tensor},
    st, CheckpointError, ConvertOptions,
};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

pub(crate) fn supports(cfg: &Value) -> bool {
    cfg["auto_map"]["AutoConfig"]
        .as_str()
        .is_some_and(|s| s.contains("XLMRobertaFlashConfig"))
        || cfg["lora_adaptations"].is_array()
        || cfg["model_type"].as_str().is_some_and(|s| s.starts_with("jina"))
        || cfg["architectures"].as_array().is_some_and(|a| a.iter().any(|v| v.as_str().is_some_and(|s| s.starts_with("Jina"))))
}

fn number(cfg: &Value, key: &str) -> Result<u32, CheckpointError> {
    cfg[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("config.json requires positive {key}").into())
}

/// Same as [`st::out_tensor`] but always stores F32 (LoRA factors are small and precision-critical).
fn out_f32(t: Tensor, name: String) -> Result<OutTensor, CheckpointError> {
    let mut out = st::out_tensor(
        t,
        name,
        &ConvertOptions {
            output_dtype: crate::OutputDType::F32,
            overwrite: false,
            spec_head: None,
        },
    )?;
    out.dtype = "F32".into();
    Ok(out)
}


type Sources = HashMap<String, Tensor>;

/// A possibly parametrized weight: the base tensor plus the per-adapter LoRA factors when asked for.
fn weight(
    sources: &mut Sources,
    tensors: &mut Vec<OutTensor>,
    options: &ConvertOptions,
    src: &str,
    dst: &str,
    shape: &[u64],
    lora: Option<([u64; 3], [u64; 3])>,
) -> Result<(), CheckpointError> {
    let base = sources
        .remove(&format!("{src}.parametrizations.weight.original"))
        .or_else(|| sources.remove(&format!("{src}.weight")))
        .ok_or_else(|| format!("missing tensor {src}.weight"))?;
    if base.shape != shape {
        return Err(format!("tensor {src}.weight has shape {:?}, expected {shape:?}", base.shape).into());
    }
    tensors.push(st::out_tensor(base, format!("{dst}.weight"), options)?);
    if let Some((a_shape, b_shape)) = lora {
        for (suffix, shape, name) in [("lora_A", a_shape, "lora_a"), ("lora_B", b_shape, "lora_b")] {
            let t = sources
                .remove(&format!("{src}.parametrizations.weight.0.{suffix}"))
                .ok_or_else(|| format!("missing LoRA tensor {src}.{suffix}"))?;
            if t.shape != shape {
                return Err(format!("LoRA tensor {src}.{suffix} has shape {:?}, expected {shape:?}", t.shape).into());
            }
            tensors.push(out_f32(t, format!("{dst}.{name}"))?);
        }
    }
    Ok(())
}

/// A plain one-dimensional tensor of a known length.
fn vector(
    sources: &mut Sources,
    tensors: &mut Vec<OutTensor>,
    options: &ConvertOptions,
    src: &str,
    dst: &str,
    len: u64,
) -> Result<(), CheckpointError> {
    let t = sources
        .remove(src)
        .ok_or_else(|| format!("missing tensor {src}"))?;
    if t.shape != [len] {
        return Err(format!("tensor {src} has shape {:?}, expected [{len}]", t.shape).into());
    }
    tensors.push(st::out_tensor(t, dst.to_owned(), options)?);
    Ok(())
}

pub(crate) fn prepare(
    input: &Path,
    options: ConvertOptions,
    cfg: &Value,
) -> Result<crate::enc_out::Prepared, CheckpointError> {
    let root = if input.is_dir() {
        input
    } else {
        input.parent().ok_or("checkpoint has no parent directory")?
    };
    let hidden = number(cfg, "hidden_size")?;
    let heads = number(cfg, "num_attention_heads")?;
    let layers = number(cfg, "num_hidden_layers")?;
    let inter = number(cfg, "intermediate_size")?;
    let vocab = number(cfg, "vocab_size")?;
    let positions = number(cfg, "max_position_embeddings")?;
    if !hidden.is_multiple_of(heads) || hidden > 8192 || layers > 128 || vocab > 10_000_000 {
        return Err("unsupported encoder dimensions".into());
    }
    let head_dim = hidden / heads;
    let rotary = match cfg["position_embedding_type"].as_str().unwrap_or("absolute") {
        "rotary" => true,
        "absolute" => false,
        other => return Err(format!("unsupported position_embedding_type {other}").into()),
    };
    if rotary
        && (cfg["rotary_emb_interleaved"].as_bool() == Some(true)
            || !cfg["rotary_emb_scale_base"].is_null()
            || cfg["rotary_emb_dim"]
                .as_f64()
                .is_some_and(|d| d as u32 != head_dim))
    {
        return Err("only full-width, non-interleaved, unscaled rotary embeddings are supported".into());
    }
    let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
    let act = cfg["hidden_act"].as_str().unwrap_or("gelu");
    if !matches!(act, "gelu" | "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh") {
        return Err(format!("unsupported activation {act}").into());
    }
    let act = if act == "gelu_pytorch_tanh" { "gelu_new" } else { act };
    let pad = cfg["pad_token_id"].as_u64().unwrap_or(1) as u32;
    let offset = if rotary { 0 } else { pad + 1 };
    let context = positions
        .checked_sub(if rotary { 2 } else { offset })
        .filter(|&n| n > 0 && n <= 8192)
        .ok_or("unsupported context length")?;
    let type_vocab = cfg["type_vocab_size"].as_u64().unwrap_or(1) as u32;
    let model_type = if rotary { "jina_xlmr" } else { "xlm-roberta" };
    let mut meta = vec![
        ("general.architecture".into(), Meta::Str("bert".into())),
        (
            "general.name".into(),
            Meta::Str(root.file_name().unwrap_or_default().to_string_lossy().into()),
        ),
        ("bert.attention.causal".into(), Meta::Bool(false)),
        ("bert.attention.layer_norm_epsilon".into(), Meta::F32(eps)),
        ("bert.activation".into(), Meta::Str(act.into())),
        ("bert.model_type".into(), Meta::Str(model_type.into())),
    ];
    for (key, n) in [
        ("embedding_length", hidden),
        ("context_length", context),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("position_offset", offset),
        ("token_type_count", type_vocab),
        ("pooling_type", st::pooling_or(root, || st::MEAN)?),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    if rotary {
        let base = cfg["rotary_emb_base"].as_f64().unwrap_or(10000.0);
        meta.push(("bert.rope_theta".into(), Meta::F32(base as f32)));
    }
    meta.extend(bert::tokenizer(root, vocab, cfg, false)?);

    // LoRA configuration.
    let adaptations = cfg["lora_adaptations"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect::<Vec<_>>())
        .unwrap_or_default();
    let rank = cfg["lora_rank"].as_u64().unwrap_or(4);
    if !adaptations.is_empty() {
        let instructions = adaptations
            .iter()
            .map(|n| cfg["task_instructions"][n.as_str()].as_str().unwrap_or_default().to_owned())
            .collect::<Vec<_>>();
        meta.push(("bert.lora.names".into(), Meta::Strs(adaptations.clone())));
        meta.push(("bert.lora.instructions".into(), Meta::Strs(instructions)));
        meta.push(("bert.lora.rank".into(), Meta::U32(rank as u32)));
        meta.push((
            "bert.lora.alpha".into(),
            Meta::F32(cfg["lora_alpha"].as_f64().unwrap_or(1.0) as f32),
        ));
    }
    let n = adaptations.len() as u64;

    let mut sources: Sources = safetensors::read_checkpoint(input)?
        .into_iter()
        .map(|t| (t.name.strip_prefix("roberta.").unwrap_or(&t.name).to_owned(), t))
        .collect();
    let mut tensors = Vec::new();
    let h = u64::from(hidden);
    let i = u64::from(inter);
    let lora = !adaptations.is_empty();
    let v = u64::from(vocab);
    let t = &mut tensors;
    let o = &options;
    weight(
        &mut sources, t, o,
        "embeddings.word_embeddings",
        "token_embd",
        &[v, h],
        lora.then_some(([n, v, rank], [n, rank, h])),
    )?;
    if type_vocab > 0 {
        let tv = u64::from(type_vocab);
        weight(
            &mut sources, t, o,
            "embeddings.token_type_embeddings",
            "token_types",
            &[tv, h],
            lora.then_some(([n, tv, rank], [n, rank, h])),
        )?;
    }
    if !rotary {
        weight(
            &mut sources, t, o,
            "embeddings.position_embeddings",
            "position_embd",
            &[u64::from(positions), h],
            None,
        )?;
    }
    for suffix in ["weight", "bias"] {
        vector(&mut sources, t, o, &format!("emb_ln.{suffix}"), &format!("token_embd_norm.{suffix}"), h)?;
    }
    for layer in 0..layers {
        let p = format!("encoder.layers.{layer}");
        let b = format!("blk.{layer}");
        for (src, dst, rows, cols) in [
            ("mixer.Wqkv", "attn_qkv", 3 * h, h),
            ("mixer.out_proj", "attn_output", h, h),
            ("mlp.fc1", "ffn_up", i, h),
            ("mlp.fc2", "ffn_down", h, i),
        ] {
            weight(
                &mut sources, t, o,
                &format!("{p}.{src}"),
                &format!("{b}.{dst}"),
                &[rows, cols],
                lora.then_some(([n, rank, cols], [n, rows, rank])),
            )?;
            vector(&mut sources, t, o, &format!("{p}.{src}.bias"), &format!("{b}.{dst}.bias"), rows)?;
        }
        for (src, dst) in [("norm1", "attn_output_norm"), ("norm2", "layer_output_norm")] {
            for suffix in ["weight", "bias"] {
                vector(&mut sources, t, o, &format!("{p}.{src}.{suffix}"), &format!("{b}.{dst}.{suffix}"), h)?;
            }
        }
    }
    // Sequence classification head (jina-reranker-v2): dense + tanh + out_proj like RoBERTa.
    if let Some(labels) = sources.get("classifier.out_proj.weight").and_then(|t| t.shape.first().copied()) {
        for (src, dst, rows, cols) in [
            ("classifier.dense", "classifier_dense", h, h),
            ("classifier.out_proj", "classifier", labels, h),
        ] {
            let w = sources
                .remove(&format!("{src}.weight"))
                .ok_or_else(|| format!("missing tensor {src}.weight"))?;
            if w.shape != [rows, cols] {
                return Err(format!("tensor {src}.weight has an unexpected shape").into());
            }
            tensors.push(st::out_tensor(w, format!("{dst}.weight"), &options)?);
            let b = sources
                .remove(&format!("{src}.bias"))
                .ok_or_else(|| format!("missing tensor {src}.bias"))?;
            tensors.push(st::out_tensor(b, format!("{dst}.bias"), &options)?);
        }
        let names = (0..labels)
            .map(|id| {
                cfg["id2label"][id.to_string()]
                    .as_str()
                    .map_or_else(|| format!("LABEL_{id}"), str::to_owned)
            })
            .collect::<Vec<_>>();
        meta.push(("bert.classifier.label_count".into(), Meta::U32(labels as u32)));
        meta.push(("bert.classifier.kind".into(), Meta::Str("roberta".into())));
        meta.push(("bert.classifier.labels".into(), Meta::Strs(names)));
    }
    let skipped = sources.into_keys().collect::<Vec<_>>();
    st::extend(root, hidden, &options, &mut meta, &mut tensors)?;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((meta, tensors, skipped))
}
