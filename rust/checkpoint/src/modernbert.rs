//! ModernBERT encoder checkpoints exported in the BERT GGUF tensor layout.
//!
//! ModernBERT differs from BERT in every sub-tensor: rotary instead of learned
//! positions (two thetas, global vs sliding layers), fused QKV / GeGLU inputs,
//! bias-free norms in pre-norm arrangement with a final output norm, and no
//! token type embeddings. The loader (`rust/engine/src/encoder.rs`) reads the
//! `bert.*` keys written here.
use crate::{
    bert,
    gguf::{self, Meta, OutTensor},
    safetensors, CheckpointError, ConversionReport, ConvertOptions, OutputDType,
};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

pub(crate) fn supports(cfg: &Value) -> bool {
    cfg["model_type"].as_str() == Some("modernbert")
}

fn number(cfg: &Value, key: &str) -> Result<u32, CheckpointError> {
    cfg[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("config.json requires positive {key}").into())
}

fn pooling(root: &Path) -> Result<u32, CheckpointError> {
    crate::st::pooling_or(root, || crate::st::MEAN)
}

fn theta(cfg: &Value, kind: &str, legacy: &str, default: f64) -> Result<f32, CheckpointError> {
    let value = cfg["rope_parameters"][kind]["rope_theta"]
        .as_f64()
        .or_else(|| cfg[legacy].as_f64())
        .or_else(|| cfg["rope_theta"].as_f64())
        .unwrap_or(default);
    if !value.is_finite() || value <= 0.0 {
        return Err("invalid modernbert rotary theta".into());
    }
    Ok(value as f32)
}

#[allow(clippy::type_complexity)]
pub(crate) fn prepare(
    input: &Path,
    options: ConvertOptions,
    cfg: &Value,
) -> Result<(Vec<(String, Meta)>, Vec<OutTensor>, Vec<String>), CheckpointError> {
    let root = if input.is_dir() {
        input
    } else {
        input.parent().ok_or("checkpoint has no parent directory")?
    };
    if cfg["is_decoder"].as_bool() == Some(true)
        || cfg["add_cross_attention"].as_bool() == Some(true)
    {
        return Err(
            "encoder conversion requires bidirectional attention without cross-attention".into(),
        );
    }
    let hidden = number(cfg, "hidden_size")?;
    let heads = number(cfg, "num_attention_heads")?;
    let layers = number(cfg, "num_hidden_layers")?;
    let inter = number(cfg, "intermediate_size")?;
    let positions = number(cfg, "max_position_embeddings")?;
    let vocab = number(cfg, "vocab_size")?;
    if hidden % heads != 0
        || hidden / heads % 2 != 0
        || hidden > 8192
        || layers > 128
        || positions > 8192
        || vocab > 10_000_000
    {
        return Err("unsupported encoder dimensions".into());
    }
    let eps = cfg["norm_eps"]
        .as_f64()
        .or_else(|| cfg["layer_norm_eps"].as_f64())
        .unwrap_or(1e-5) as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err("invalid layer norm epsilon".into());
    }
    let local_attention = cfg["local_attention"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(128);
    if local_attention == 0 || !local_attention.is_multiple_of(2) || local_attention > positions {
        return Err("invalid modernbert local attention window".into());
    }
    let layer_global: Vec<u32> = match cfg["layer_types"].as_array() {
        Some(types) => types
            .iter()
            .map(|v| match v.as_str() {
                Some("full_attention") => Ok(1),
                Some("sliding_attention") => Ok(0),
                _ => Err("modernbert layer_types must be full_attention or sliding_attention"),
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            let every = cfg["global_attn_every_n_layers"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(3);
            if every == 0 {
                return Err("invalid modernbert global attention stride".into());
            }
            (0..layers).map(|i| u32::from(i % every == 0)).collect()
        }
    };
    if layer_global.len() as u32 != layers {
        return Err("modernbert layer_types disagree with num_hidden_layers".into());
    }
    let theta_global = theta(cfg, "full_attention", "global_rope_theta", 160_000.0)?;
    let theta_local = theta(cfg, "sliding_attention", "local_rope_theta", 10_000.0)?;
    let mut meta = vec![
        ("general.architecture".into(), Meta::Str("bert".into())),
        (
            "general.name".into(),
            Meta::Str(
                root.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into(),
            ),
        ),
        ("bert.attention.causal".into(), Meta::Bool(false)),
        ("bert.attention.layer_norm_epsilon".into(), Meta::F32(eps)),
        ("bert.activation".into(), Meta::Str("geglu".into())),
    ];
    for (key, n) in [
        ("embedding_length", hidden),
        ("context_length", positions),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("position_offset", 0),
        ("token_type_count", 0),
        ("pooling_type", pooling(root)?),
        ("local_attention", local_attention),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    meta.push(("bert.rope_theta_global".into(), Meta::F32(theta_global)));
    meta.push(("bert.rope_theta_local".into(), Meta::F32(theta_local)));
    meta.push(("bert.layer_global".into(), Meta::U32s(layer_global)));
    meta.extend(bert::tokenizer(root, vocab, cfg, false)?);
    meta.push(("bert.model_type".into(), Meta::Str("modernbert".into())));
    let sources = safetensors::read_checkpoint(input)?;
    for t in &sources {
        let name = t
            .name
            .strip_prefix("model.")
            .or_else(|| t.name.strip_prefix("modernbert."))
            .unwrap_or(&t.name);
        if (name.starts_with("embeddings.position_embeddings.")
            || name.starts_with("embeddings.token_type_embeddings."))
            || (name.contains(".attn.Wqkv.bias") || name.contains(".mlp.Wi.bias"))
        {
            return Err(format!("unexpected modernbert tensor {}", t.name).into());
        }
    }
    let wants_head = cfg["architectures"].as_array().is_some_and(|a| {
        a.iter().any(|v| {
            v.as_str()
                .is_some_and(|v| v.ends_with("ForSequenceClassification"))
        })
    });
    let head = sources.iter().find(|t| {
        t.name == "classifier.weight"
            || t.name == "model.classifier.weight"
            || t.name == "modernbert.classifier.weight"
    });
    let num_labels = if let Some(t) = head {
        let labels = t
            .shape
            .first()
            .copied()
            .filter(|v| *v > 0 && *v <= 4096)
            .ok_or("invalid classifier label count")?;
        if cfg
            .get("num_labels")
            .is_some_and(|v| v.as_u64() != Some(labels))
            || cfg.get("id2label").is_some_and(|v| {
                !v.is_null() && v.as_object().is_none_or(|m| m.len() as u64 != labels)
            })
        {
            return Err("classifier label count disagrees with config.json".into());
        }
        Some(labels)
    } else if wants_head {
        return Err("missing required classifier tensor classifier.weight".into());
    } else {
        None
    };
    let mut expected = HashMap::<String, (String, Vec<u64>)>::new();
    let stripped: HashSet<String> = sources
        .iter()
        .map(|t| {
            t.name
                .strip_prefix("model.")
                .or_else(|| t.name.strip_prefix("modernbert."))
                .unwrap_or(&t.name)
                .to_owned()
        })
        .collect();
    if let Some(labels) = num_labels {
        meta.push((
            "bert.classifier.label_count".into(),
            Meta::U32(labels as u32),
        ));
        meta.push((
            "bert.classifier.kind".into(),
            Meta::Str("modernbert".into()),
        ));
        let pooling = cfg["classifier_pooling"]
            .as_str()
            .unwrap_or("cls")
            .to_owned();
        if !matches!(pooling.as_str(), "cls" | "mean") {
            return Err("modernbert classifier_pooling must be cls or mean".into());
        }
        meta.push(("bert.classifier.pooling".into(), Meta::Str(pooling)));
        let names = (0..labels)
            .map(|id| {
                if cfg["id2label"].is_null() {
                    Ok(format!("LABEL_{id}"))
                } else {
                    cfg["id2label"][id.to_string()]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("config.json id2label is missing string label {id}"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        meta.push(("bert.classifier.labels".into(), Meta::Strs(names)));
        if stripped.contains("classifier.bias") {
            expected.insert(
                "classifier.bias".into(),
                ("classifier.bias".into(), vec![labels]),
            );
        }
        expected.insert(
            "classifier.weight".into(),
            (
                "classifier.weight".into(),
                vec![labels, hidden as u64],
            ),
        );
        expected.insert(
            "head.dense.weight".into(),
            (
                "classifier_dense.weight".into(),
                vec![hidden as u64, hidden as u64],
            ),
        );
        if stripped.contains("head.dense.bias") {
            expected.insert(
                "head.dense.bias".into(),
                ("classifier_dense.bias".into(), vec![hidden as u64]),
            );
        }
        expected.insert(
            "head.norm.weight".into(),
            ("classifier_norm.weight".into(), vec![hidden as u64]),
        );
    }
    expected.insert(
        "embeddings.tok_embeddings.weight".into(),
        ("token_embd.weight".into(), vec![vocab as u64, hidden as u64]),
    );
    expected.insert(
        "embeddings.norm.weight".into(),
        ("token_embd_norm.weight".into(), vec![hidden as u64]),
    );
    expected.insert(
        "final_norm.weight".into(),
        ("output_norm.weight".into(), vec![hidden as u64]),
    );
    for i in 0..layers {
        let layer = format!("layers.{i}");
        expected.insert(
            format!("{layer}.attn.Wqkv.weight"),
            (
                format!("blk.{i}.attn_qkv.weight"),
                vec![3 * hidden as u64, hidden as u64],
            ),
        );
        expected.insert(
            format!("{layer}.attn.Wo.weight"),
            (
                format!("blk.{i}.attn_output.weight"),
                vec![hidden as u64, hidden as u64],
            ),
        );
        if i > 0 {
            expected.insert(
                format!("{layer}.attn_norm.weight"),
                (
                    format!("blk.{i}.attn_output_norm.weight"),
                    vec![hidden as u64],
                ),
            );
        }
        expected.insert(
            format!("{layer}.mlp_norm.weight"),
            (format!("blk.{i}.ffn_norm.weight"), vec![hidden as u64]),
        );
        expected.insert(
            format!("{layer}.mlp.Wi.weight"),
            (
                format!("blk.{i}.ffn_up.weight"),
                vec![2 * inter as u64, hidden as u64],
            ),
        );
        expected.insert(
            format!("{layer}.mlp.Wo.weight"),
            (
                format!("blk.{i}.ffn_down.weight"),
                vec![hidden as u64, inter as u64],
            ),
        );
    }
    if stripped.contains("layers.0.attn_norm.weight") {
        expected.insert(
            "layers.0.attn_norm.weight".into(),
            ("blk.0.attn_output_norm.weight".into(), vec![hidden as u64]),
        );
    }
    let mut tensors = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for t in sources {
        let source = t
            .name
            .strip_prefix("model.")
            .or_else(|| t.name.strip_prefix("modernbert."))
            .unwrap_or(&t.name);
        let Some((name, shape)) = expected.get(source) else {
            skipped.push(t.name);
            continue;
        };
        if &t.shape != shape {
            return Err(format!(
                "tensor {} has shape {:?}, expected {shape:?}",
                t.name, t.shape
            )
            .into());
        }
        if !seen.insert(name.clone()) {
            return Err(format!("duplicate encoder tensor {name}").into());
        }
        if !matches!(t.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(format!("unsupported encoder dtype {}", t.dtype).into());
        }
        let dtype = match options.output_dtype {
            OutputDType::F32 => "F32",
            OutputDType::F16 => "F16",
            OutputDType::Preserve => {
                if t.dtype == "BF16" {
                    "F16"
                } else {
                    &t.dtype
                }
            }
        }
        .to_owned();
        let count = t
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or("encoder tensor size overflow")?;
        let data_len = count
            .checked_mul(if dtype == "F32" { 4 } else { 2 })
            .ok_or("encoder tensor byte size overflow")?;
        tensors.push(OutTensor {
            name: name.clone(),
            dims: t.shape.iter().rev().copied().collect(),
            source_dtype: t.dtype,
            dtype,
            data: t.payload,
            packed: None,
            data_len,
            source_for_output: None,
            add: 0.0,
            owned: None,
            parts: Vec::new(),
        });
    }
    for (name, _) in expected.values() {
        if !seen.contains(name) {
            return Err(format!("missing required encoder tensor {name}").into());
        }
    }
    crate::st::extend(root, hidden, &options, &mut meta, &mut tensors)?;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((meta, tensors, skipped))
}

#[allow(dead_code)]
pub(crate) fn convert(
    input: &Path,
    output: &Path,
    options: ConvertOptions,
    cfg: &Value,
) -> Result<ConversionReport, CheckpointError> {
    let (meta, tensors, skipped) = prepare(input, options, cfg)?;
    if output.exists() && !options.overwrite {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    if input.is_file() && fs::canonicalize(input).ok() == fs::canonicalize(output).ok() {
        return Err("input and output must differ".into());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut tmp = output.as_os_str().to_owned();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let tmp = std::path::PathBuf::from(tmp);
    let result = (|| -> Result<u64, CheckpointError> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let size = gguf::write_to(&mut file, &meta, &tensors)?;
        file.sync_all()?;
        drop(file);
        if options.overwrite {
            fs::rename(&tmp, output)?;
        } else {
            fs::hard_link(&tmp, output)?;
            fs::remove_file(&tmp)?;
        }
        Ok(size)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(ConversionReport {
        architecture: "bert".into(),
        tensors_written: tensors.len(),
        bytes_written: result?,
        skipped_tensors: skipped,
        output: output.to_owned(),
    })
}

